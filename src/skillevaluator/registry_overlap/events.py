# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, checksum-verified local replay batches for the registry pilot."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from skillevaluator.embedding.extractor import MAX_MANIFEST_BYTES, parse_skill_manifest
from skillevaluator.embedding.registry import _validate_catalog_string, _validate_embedding_text
from skillevaluator.utils.secure_fs import SecureRoot
from skillevaluator.utils.structured_data import preflight_json_structure, reject_duplicate_object_pairs

MAX_BATCH_BYTES = 8 * 1024 * 1024
MAX_BATCH_EVENTS = 250
_MAX_SEQUENCE = 2**63 - 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_OPAQUE_REFERENCE = re.compile(r"ref:[A-Za-z0-9][A-Za-z0-9_.+-]{0,255}")
_EVENT_FIELDS = {"id", "sequence", "revision", "kind"}


@dataclass(frozen=True)
class RegistryEvent:
    skill_id: str
    sequence: int
    revision: str
    digest: str
    text: str | None = None
    artifact: str | None = None
    manifest_sha256: str | None = None
    input_sha256: str | None = None


@dataclass(frozen=True)
class RegistryBatch:
    source: str
    cursor: int
    digest: str
    events: tuple[RegistryEvent, ...]


def identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid registry {field}")
    return value


def _sequence(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_SEQUENCE:
        raise ValueError("Expected a positive 64-bit integer")
    return value


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def embedding_input(manifest: str) -> str:
    entry = parse_skill_manifest(manifest)
    if entry is None:
        raise ValueError("Expected skill name and description in bounded YAML frontmatter")
    _validate_catalog_string(entry.name, "Skill name")
    _validate_catalog_string(entry.description, "Skill description", allow_text_whitespace=True)
    _validate_embedding_text(entry.embedding_text, full_body=False)
    return entry.embedding_text


def _artifact_reference(value: object) -> str:
    if isinstance(value, str) and _OPAQUE_REFERENCE.fullmatch(value):
        return value
    if (
        isinstance(value, str)
        and len(value) <= 1024
        and value.isprintable()
        and not any(character in value for character in " \\?#")
    ):
        url = urlsplit(value)
        _ = url.port  # Accessing port rejects malformed or out-of-range ports.
        if url.scheme == "https" and url.hostname and url.username is None:
            return value
    raise ValueError("Expected a credential-free HTTPS artifact URL or opaque ref: identifier")


def _event(data: object) -> RegistryEvent:
    if not isinstance(data, dict):
        raise ValueError("Expected an event object")
    kind = data.get("kind")
    required = _EVENT_FIELDS | {"manifest", "artifact"} if kind == "upsert" else _EVENT_FIELDS
    if kind not in ("upsert", "remove") or data.keys() != required:
        raise ValueError("Unexpected event fields or kind")
    identity = identifier(data["id"], "skill ID")
    sequence = _sequence(data["sequence"])
    revision = identifier(data["revision"], "revision")
    digest = _digest(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    if kind == "remove":
        return RegistryEvent(identity, sequence, revision, digest)

    manifest = data["manifest"]
    if not isinstance(manifest, str):
        raise ValueError("Expected manifest text")
    artifact = _artifact_reference(data["artifact"])
    text = embedding_input(manifest)
    return RegistryEvent(
        identity, sequence, revision, digest, text, artifact, _digest(manifest.encode()), _digest(text.encode())
    )


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant")


def _parse_integer(value: str) -> int:
    if len(value.removeprefix("-")) > 19:
        raise ValueError("Registry JSON integer exceeds 19 digits")
    return int(value)


def parse_batch(raw: bytes, expected_sha256: str) -> RegistryBatch:
    """Verify bytes and validate every event before returning any work."""
    if len(raw) > MAX_BATCH_BYTES:
        raise ValueError("Registry batch exceeds the byte limit")
    digest = _digest(raw)
    if not _DIGEST.fullmatch(expected_sha256) or not hmac.compare_digest(digest, expected_sha256):
        raise ValueError("Registry batch checksum does not match")
    try:
        text = raw.decode("utf-8")
        preflight_json_structure(
            text,
            max_depth=4,
            max_tokens=6000,
            max_collection_items=MAX_BATCH_EVENTS,
            max_mapping_items=8,
            max_string_chars=MAX_MANIFEST_BYTES,
        )
        data = json.loads(
            text,
            object_pairs_hook=reject_duplicate_object_pairs,
            parse_constant=_reject_constant,
            parse_int=_parse_integer,
        )
    except ValueError:
        raise ValueError("Invalid or oversized registry batch JSON") from None
    if (
        not isinstance(data, dict)
        or data.keys() != {"schema_version", "source", "cursor", "events"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or not isinstance(data["events"], list)
        or not 1 <= len(data["events"]) <= MAX_BATCH_EVENTS
    ):
        raise ValueError("Invalid registry batch envelope")
    source = identifier(data["source"], "source")
    cursor = _sequence(data["cursor"])
    events: list[RegistryEvent] = []
    for number, candidate in enumerate(data["events"], start=1):
        try:
            events.append(_event(candidate))
        except (UnicodeError, ValueError):
            raise ValueError(f"Registry event {number} has invalid fields or manifest") from None
    return RegistryBatch(source, cursor, digest, tuple(events))


def read_batch(path: Path, expected_sha256: str) -> RegistryBatch:
    """Read an operator-supplied batch without following a file link."""
    try:
        with SecureRoot(path.parent) as root:
            raw, _ = root.read_bytes(Path(path.name), MAX_BATCH_BYTES)
    except (OSError, ValueError):
        raise ValueError("Cannot securely read registry batch") from None
    return parse_batch(raw, expected_sha256)
