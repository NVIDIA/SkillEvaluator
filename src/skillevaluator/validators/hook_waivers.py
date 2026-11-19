# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load explicit, content-pinned hook review waivers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_yaml

_MAX_WAIVER_BYTES = 1024 * 1024
_MAX_WAIVERS = 32
_MAX_PAYLOAD_FILES = 512
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_WAIVER_FIELDS = frozenset(
    {"id", "manifest", "manifest_sha256", "artifact_sha256", "payload", "reason", "owner", "created", "expires"}
)


@dataclass(frozen=True)
class HookWaiver:
    """A review bound to a manifest and exact packaged artifact contents."""

    id: str
    manifest: str
    manifest_sha256: str
    artifact_sha256: str
    payload: Mapping[str, str]
    reason: str
    owner: str
    created: date
    expires: date


def _text_field(value: object, label: str, *, minimum_length: int = 1) -> str:
    if isinstance(value, str) and len(value.strip()) >= minimum_length:
        return value.strip()
    raise ValueError(f"{label} must be a string with at least {minimum_length} non-padding characters")


def _path_field(value: object, label: str) -> str:
    if not isinstance(value, str) or value != value.strip() or any(marker in value for marker in ("\\", ":", "!/")):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _sha256_field(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _date_field(value: object, label: str) -> date:
    if type(value) is date:
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a date (YYYY-MM-DD)") from exc
        if parsed.isoformat() == value:
            return parsed
    raise ValueError(f"{label} must be a date (YYYY-MM-DD)")


def _payload_field(value: object, manifest: str, label: str) -> Mapping[str, str]:
    if not isinstance(value, list) or len(value) > _MAX_PAYLOAD_FILES:
        raise ValueError(f"{label} must be a list of at most {_MAX_PAYLOAD_FILES} files")
    payload: dict[str, str] = {}
    for index, entry in enumerate(value):
        location = f"{label}[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"{location} must contain only path and sha256")
        path = _path_field(entry["path"], f"{location}.path")
        if path == manifest or path in payload:
            raise ValueError(f"{location}.path duplicates a manifest or payload path")
        payload[path] = _sha256_field(entry["sha256"], f"{location}.sha256")
    return MappingProxyType(dict(sorted(payload.items())))


def _parse_waiver(entry: object, index: int, today: date) -> HookWaiver:
    label = f"waivers[{index}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{label} must be a mapping")
    missing = _WAIVER_FIELDS - entry.keys()
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(sorted(missing))}")
    if entry.keys() - _WAIVER_FIELDS:
        raise ValueError(f"{label} contains unsupported fields")

    created = _date_field(entry["created"], f"{label}.created")
    expires = _date_field(entry["expires"], f"{label}.expires")
    if created > today or expires < created:
        raise ValueError(f"{label}.created cannot be in the future, and expires cannot be before created")
    manifest = _path_field(entry["manifest"], f"{label}.manifest path")
    return HookWaiver(
        id=_text_field(entry["id"], f"{label}.id"),
        manifest=manifest,
        manifest_sha256=_sha256_field(entry["manifest_sha256"], f"{label}.manifest_sha256"),
        artifact_sha256=_sha256_field(entry["artifact_sha256"], f"{label}.artifact_sha256"),
        payload=_payload_field(entry["payload"], manifest, f"{label}.payload"),
        reason=_text_field(entry["reason"], f"{label}.reason", minimum_length=8),
        owner=_text_field(entry["owner"], f"{label}.owner", minimum_length=2),
        created=created,
        expires=expires,
    )


def load_hook_waivers(path: Path) -> tuple[HookWaiver, ...]:
    """Read one explicitly supplied approval file or raise ValueError.

    Expired waivers still load for reporting. A consumer may apply a matching
    waiver through its expiry date in UTC.
    """
    try:
        with SecureRoot(path.parent) as root:
            data, _ = root.read_bytes(Path(path.name), _MAX_WAIVER_BYTES)
    except (OSError, SecurePathError) as exc:
        raise ValueError("cannot securely read the hook waiver file as a regular file of at most 1 MiB") from exc
    try:
        document = load_bounded_yaml(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("hook waiver file must be valid UTF-8") from exc
    except StructuredDataError as exc:
        raise ValueError(f"invalid hook waiver YAML: {exc}") from exc

    if not isinstance(document, dict) or set(document) != {"version", "waivers"}:
        raise ValueError("hook waiver file must contain only version and waivers")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("hook waiver file version must be 1")
    entries = document["waivers"]
    if not isinstance(entries, list) or len(entries) > _MAX_WAIVERS:
        raise ValueError(f"waivers must be a list of at most {_MAX_WAIVERS} entries")

    today = datetime.now(UTC).date()
    waivers = tuple(_parse_waiver(entry, index, today) for index, entry in enumerate(entries))
    if len({waiver.id for waiver in waivers}) != len(waivers):
        raise ValueError("waivers contain a duplicate id")
    return waivers
