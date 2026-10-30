# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select local workspace decoys once, then verify their frozen identities.

Description similarity is a selection policy, not evidence that a skill is a
prerequisite or that two skills are confusing to an agent. The manifest records
the policy and source identities so evaluation never repeats retrieval.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from skillevaluator.constants import CONTENT_TYPE_SKILL
from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.extractor import ContentEntry, _ensure_safe_content_path, extract_from_skill
from skillevaluator.embedding.registry import (
    MAX_SIMILARITY_MATCHES,
    EmbeddingRegistry,
    _catalog_object,
    _read_catalog_text,
    _reject_json_constant,
    _validate_catalog_identity,
    _validate_catalog_string,
    _write_catalog_atomically,
)
from skillevaluator.publication_identity import PUBLICATION_TARGET_DIGEST_ALGORITHM, publication_source_digest
from skillevaluator.utils.path_security import canonicalize_trusted_root_alias
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

DECOY_MANIFEST_VERSION = 1
DECOY_SELECTION_STRATEGY = "description-similarity"
MAX_DECOYS = MAX_SIMILARITY_MATCHES
_HASH = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = {
    "schema_version",
    "strategy",
    "catalog_root",
    "catalog",
    "target",
    "count",
    "exclude",
    "selected",
    "source_digest_algorithm",
}
_CATALOG_FIELDS = {"sha256", "provider", "model", "mode", "endpoint_fingerprint"}
_TARGET_FIELDS = {"name", "skill_name", "manifest_sha256"}
_SELECTED_FIELDS = {"path", "name", "skill_name", "score", "source_digest"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _absolute_root(path: Path) -> Path:
    # Keep later path components lexical until no-follow checks have run.
    root = canonicalize_trusted_root_alias(Path(os.path.abspath(path.expanduser())))  # noqa: PTH100
    _ensure_safe_content_path(Path(root.anchor), root)
    if not root.is_dir():
        raise ValueError(f"Decoy skill root must be a directory: {root}")
    return root


def _skill(path: Path) -> ContentEntry:
    entry = extract_from_skill(path)
    if entry is None or not (path / "SKILL.md").is_file():
        raise ValueError(f"Decoy selection requires a skill with SKILL.md and name/description: {path}")
    _validate_catalog_string(entry.name, "Skill name")
    return entry


def _relative_path(value: object) -> str:
    _validate_catalog_identity(f"skill:{value}", value, CONTENT_TYPE_SKILL)
    return str(value)


def _selected_path(root: Path, relative: object) -> Path:
    relative = _relative_path(relative)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    _ensure_safe_content_path(root, candidate)
    if not candidate.is_dir():
        raise ValueError(f"Selected decoy is not a skill directory: {relative}")
    return candidate


def _instructions(entry: ContentEntry) -> str:
    match = FRONTMATTER_PATTERN.match(entry.full_text)
    return match.group(2).strip() if match else entry.full_text.strip()


def _aliases_target(entry: ContentEntry, target: ContentEntry) -> bool:
    return target.full_text in entry.full_text or bool(
        _instructions(target) and _instructions(entry) == _instructions(target)
    )


def _source_digest(path: Path) -> str:
    digest = publication_source_digest(path)
    if digest is None:
        raise ValueError(f"Selected decoy has an unsafe or unstable source tree: {path}")
    return digest


def _contains_target_payload(path: Path, target: Path) -> bool:
    # Match the baseline adapter's policy, including renamed files and nested
    # packages, while ignoring files that never enter the runtime projection.
    # The caller validates the source tree before this bounded-memory scan.
    from skillevaluator.tier3.harbor.adapter import _find_target_manifest_payload

    return _find_target_manifest_payload(path, target, runtime_projection=True) is not None


def _count(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_DECOYS:
        raise ValueError(f"Decoy count must be an integer from 1 to {MAX_DECOYS}")
    return value


def select_decoys(
    target: Path,
    *,
    catalog_path: Path,
    catalog_root: Path,
    count: int,
    exclude: tuple[str, ...] = (),
    client: EmbeddingClient | None = None,
) -> dict[str, Any]:
    """Rank local skills from a description-mode Tier 2 catalog.

    Exact target copies, duplicate identities, and negative-similarity matches
    are ineligible. Ties use catalog-relative path order. No evaluation outcome
    is consulted. Invalid/stale candidate sources fail instead of being silently
    replaced with a lower-ranked candidate.
    """
    count = _count(count)
    root = _absolute_root(catalog_root)
    target = _absolute_root(target)
    target_entry = _skill(target)
    exclusions = sorted({_relative_path(path) for path in exclude})
    registry = EmbeddingRegistry(client or EmbeddingClient())
    before = _read_catalog_text(catalog_path)
    registry.load_catalog(catalog_path)
    if _read_catalog_text(catalog_path) != before:
        raise ValueError("Decoy catalog changed while being loaded")
    catalog = json.loads(before)
    if registry.size > MAX_SIMILARITY_MATCHES:
        raise ValueError(f"Decoy selection supports at most {MAX_SIMILARITY_MATCHES} catalog entries")
    if any(entry.content_type != CONTENT_TYPE_SKILL for entry in registry._entries.values()):
        raise ValueError("Decoy selection requires a skill-only catalog")
    catalog_entries = {entry.path: entry for entry in registry._entries.values()}
    unknown_exclusions = set(exclusions) - catalog_entries.keys()
    if unknown_exclusions:
        raise ValueError("Excluded decoy path is absent from catalog: " + ", ".join(sorted(unknown_exclusions)))
    matches = registry.query_entry(target_entry, threshold=0.0)
    matches.sort(key=lambda match: (-match.score, match.path_b))

    names = {_identity(target.name)}
    skill_names = {_identity(target_entry.name)}
    digests: set[str] = set()
    selected: list[dict[str, Any]] = []
    for match in matches:
        stored = catalog_entries[match.path_b]
        if stored.content_type != CONTENT_TYPE_SKILL or stored.path in exclusions:
            continue
        candidate = _selected_path(root, stored.path)
        if candidate == target or _identity(candidate.name) in names or _identity(stored.name) in skill_names:
            continue
        entry = _skill(candidate)
        if _sha256(entry.embedding_text) != stored.content_fingerprint:
            raise ValueError(f"Decoy catalog entry is stale; rebuild the catalog: {stored.path}")
        if _aliases_target(entry, target_entry):
            continue
        digest = _source_digest(candidate)
        if digest in digests or _contains_target_payload(candidate, target):
            continue
        selected.append(
            {
                "path": stored.path,
                "name": candidate.name,
                "skill_name": entry.name,
                "score": match.score,
                "source_digest": digest,
            }
        )
        names.add(_identity(candidate.name))
        skill_names.add(_identity(entry.name))
        digests.add(digest)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Requested {count} decoys but only {len(selected)} eligible skills were found")
    if _skill(target).full_text != target_entry.full_text:
        raise ValueError("Target skill changed during decoy selection")
    return {
        "schema_version": DECOY_MANIFEST_VERSION,
        "strategy": DECOY_SELECTION_STRATEGY,
        "catalog_root": str(root),
        "catalog": {
            "sha256": _sha256(before),
            **{key: catalog[key] for key in _CATALOG_FIELDS - {"sha256"}},
        },
        "target": {
            "name": target.name,
            "skill_name": target_entry.name,
            "manifest_sha256": _sha256(target_entry.full_text),
        },
        "count": count,
        "exclude": exclusions,
        "selected": selected,
        "source_digest_algorithm": PUBLICATION_TARGET_DIGEST_ALGORITHM,
    }


def _object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.keys() != keys:
        raise ValueError(f"Invalid decoy manifest {label} fields")
    return value


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"Invalid decoy manifest {label} hash")
    return value


def _validate_manifest(manifest: object, target: Path) -> tuple[list[Path], dict[str, Any]]:
    data = _object(manifest, _MANIFEST_FIELDS, "root")
    if type(data["schema_version"]) is not int or data["schema_version"] != DECOY_MANIFEST_VERSION:
        raise ValueError("Unsupported decoy manifest schema version")
    if data["strategy"] != DECOY_SELECTION_STRATEGY:
        raise ValueError("Unsupported decoy selection strategy")
    if data["source_digest_algorithm"] != PUBLICATION_TARGET_DIGEST_ALGORITHM:
        raise ValueError("Unsupported decoy source digest algorithm")
    count = _count(data["count"])
    catalog = _object(data["catalog"], _CATALOG_FIELDS, "catalog")
    _hash(catalog["sha256"], "catalog")
    for key in ("provider", "model", "mode", "endpoint_fingerprint"):
        _validate_catalog_string(catalog[key], f"Decoy catalog {key}")
    if catalog["mode"] != "description" or not re.fullmatch(r"sha256:[0-9a-f]{64}", catalog["endpoint_fingerprint"]):
        raise ValueError("Invalid decoy catalog mode or endpoint fingerprint")
    root_value = _validate_catalog_string(data["catalog_root"], "Decoy catalog root")
    if not Path(root_value).is_absolute():
        raise ValueError("Decoy catalog root must be absolute")
    root = _absolute_root(Path(root_value))
    target = _absolute_root(target)
    entry = _skill(target)
    target_data = _object(data["target"], _TARGET_FIELDS, "target")
    _hash(target_data["manifest_sha256"], "target")
    if target_data != {
        "name": target.name,
        "skill_name": entry.name,
        "manifest_sha256": _sha256(entry.full_text),
    }:
        raise ValueError("Decoy manifest target changed; regenerate the selection")
    exclusions = data["exclude"]
    if not isinstance(exclusions, list) or len(exclusions) > MAX_SIMILARITY_MATCHES:
        raise ValueError("Invalid decoy manifest exclusions")
    exclusions = [_relative_path(value) for value in exclusions]
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("Duplicate decoy manifest exclusions")
    selected = data["selected"]
    if not isinstance(selected, list) or len(selected) != count:
        raise ValueError("Decoy manifest selected count does not match count")
    paths: list[Path] = []
    names = {_identity(target.name)}
    skill_names = {_identity(entry.name)}
    digests: set[str] = set()
    for item in selected:
        item = _object(item, _SELECTED_FIELDS, "selected skill")
        path = _selected_path(root, item["path"])
        current = _skill(path)
        score = item["score"]
        if type(score) not in (int, float) or not 0 <= score <= 1 or not math.isfinite(score):
            raise ValueError("Invalid decoy manifest similarity score")
        digest = _hash(item["source_digest"], "source")
        if (
            path == target
            or item["path"] in exclusions
            or _identity(path.name) in names
            or _identity(current.name) in skill_names
            or digest in digests
            or _aliases_target(current, entry)
        ):
            raise ValueError("Decoy manifest contains a target alias or duplicate skill identity")
        if item["name"] != path.name or item["skill_name"] != current.name:
            raise ValueError(f"Selected decoy identity changed: {item['path']}")
        if _source_digest(path) != digest:
            raise ValueError(f"Selected decoy source changed; regenerate the selection: {item['path']}")
        if _contains_target_payload(path, target):
            raise ValueError(f"Selected decoy contains target instructions: {item['path']}")
        names.add(_identity(path.name))
        skill_names.add(_identity(current.name))
        digests.add(digest)
        paths.append(path)
    return paths, data


def write_decoy_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Atomically save a bounded JSON selection without following output links."""
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    from skillevaluator.embedding.registry import MAX_CATALOG_BYTES

    if len(payload) > MAX_CATALOG_BYTES:
        raise ValueError("Decoy manifest exceeds its byte limit")
    _write_catalog_atomically(output_path, payload)


def load_decoy_manifest(target: Path, manifest_path: Path) -> tuple[list[Path], dict[str, Any]]:
    """Verify a saved selection against local sources, without model access.

    Paths and content hashes are checked, not silently refreshed. Callers that
    stage multiple paired environments should snapshot these verified sources
    once and verify their hashes again before running either arm.
    """
    try:
        raw = json.loads(
            _read_catalog_text(manifest_path),
            object_pairs_hook=_catalog_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid decoy manifest JSON: {exc}") from exc
    return _validate_manifest(raw, target)
