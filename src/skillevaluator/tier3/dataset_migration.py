# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preview-first migration of authored datasets without a generator allowlist."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from skillevaluator.tier3.case_ids import validate_case_ids

_ALIASES = {
    "question": "prompt",
    "ground_truth": "expected_output",
    "expected_behavior": "assertions",
    "acceptable_alternates": "acceptable_skills",
}


@dataclass(frozen=True)
class MigrationResult:
    """The complete preview payload and the optional newly written destination."""

    dataset: dict[str, Any]
    destination: Path | None = None


def _unique_mapping(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError(f"Dataset object keys must be strings, got {key!r}")
        if key in mapping:
            raise ValueError(f"duplicate authored key {key!r}; choose one value before migration")
        mapping[key] = value
    return mapping


class _MigrationLoader(yaml.SafeLoader):
    """Keep YAML's safe constructors while rejecting overwritten authored keys."""


def _yaml_mapping(loader: _MigrationLoader, node: yaml.MappingNode) -> dict[str, Any]:
    loader.flatten_mapping(node)
    return _unique_mapping(loader.construct_pairs(node, deep=True))


_MigrationLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def _read_source(source: Path) -> Any:
    text = source.read_text(encoding="utf-8")
    try:
        if source.suffix.lower() == ".jsonl":
            return [json.loads(line, object_pairs_hook=_unique_mapping) for line in text.splitlines() if line.strip()]
        if source.suffix.lower() == ".json":
            return json.loads(text, object_pairs_hook=_unique_mapping)
        if source.suffix.lower() in {".yaml", ".yml"}:
            return yaml.load(text, Loader=_MigrationLoader)
    except (ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot parse dataset {source}: {exc}") from exc
    raise ValueError(f"Unsupported source format {source.suffix!r}; use JSON, JSONL, YAML or YML")


def _unpack(data: Any) -> tuple[dict[str, Any], list[Any], bool]:
    if isinstance(data, list):
        return {}, data, False
    if not isinstance(data, dict):
        raise ValueError("Dataset must be an object or an array of case objects")
    if "evals" in data and "cases" in data:
        raise ValueError("Dataset cannot contain both evals and cases arrays; choose one source")
    for key in ("evals", "cases"):
        if key in data:
            if not isinstance(data[key], list):
                raise ValueError(f"Dataset {key} must be an array")
            metadata = deepcopy(data)
            entries = metadata.pop(key)
            return metadata, entries, key == "evals"
    return {}, [data], False


def _identity(metadata: dict[str, Any], entries: list[Any], skill_name: str | None) -> str:
    authored = metadata.get("skill_name")
    if "skill_name" in metadata and (not isinstance(authored, str) or not authored.strip()):
        raise ValueError("Dataset skill_name must be a non-empty string")
    if skill_name is not None and (not isinstance(skill_name, str) or not skill_name.strip()):
        raise ValueError("--skill-name must be a non-empty string")
    if authored is not None and skill_name is not None and authored != skill_name:
        raise ValueError("conflicting skill identities: --skill-name differs from top-level skill_name")
    identity = skill_name if skill_name is not None else authored
    if identity is None:
        names = {
            entry.get("expected_skill")
            for entry in entries
            if isinstance(entry.get("expected_skill"), str) and entry["expected_skill"]
        }
        if len(names) > 1:
            raise ValueError("ambiguous target skill identity; supply --skill-name for the skill under evaluation")
        if not names:
            raise ValueError("Cannot infer target skill identity; supply --skill-name")
        identity = names.pop()
        if not identity.strip():
            raise ValueError("Inferred skill identity must be a non-empty string; supply --skill-name")
    return identity


def _convert_entry(entry: dict[str, Any], index: int, *, preferred: bool) -> dict[str, Any]:
    converted = deepcopy(entry)
    for old, new in _ALIASES.items():
        if old not in converted:
            continue
        # JSON comparison distinguishes nested false/0 values that Python's
        # container equality would otherwise silently consider equal.
        if new in converted and json.dumps(converted[old], sort_keys=True, allow_nan=False) != json.dumps(
            converted[new], sort_keys=True, allow_nan=False
        ):
            raise ValueError(f"entry {index}: conflicting aliases {old} and {new}; choose one authored value")
        converted[new] = converted.pop(old)
    if not isinstance(converted.get("prompt"), str) or not converted["prompt"].strip():
        raise ValueError(f"entry {index}: prompt/question must be a non-empty string")
    if "expected_output" not in converted:
        raise ValueError(f"entry {index}: author expected_output/ground_truth before migration")
    if (
        "expected_skill" in converted
        and converted["expected_skill"] is not None
        and not isinstance(converted["expected_skill"], str)
    ):
        raise ValueError(f"entry {index}: expected_skill must be a string or null")
    if not preferred and "expected_skill" not in converted:
        if "should_trigger" not in converted:
            raise ValueError(
                f"entry {index}: unlabeled routing cannot migrate losslessly; author expected_skill or should_trigger first"
            )
        # Prevent the object loader's implicit positive skill from changing an
        # explicitly labeled legacy case which had no expected skill identity.
        converted["expected_skill"] = None
    return converted


def migrate_dataset(source: Path, *, skill_name: str | None = None, destination: Path | None = None) -> MigrationResult:
    """Return a lossless object preview; write only an explicitly new JSON/YAML file.

    Case IDs retain their authored numeric/string types. Legacy unlabeled routing
    must be authored explicitly first because object loading defaults it positive.
    JSONL is accepted as input; object output uses JSON or YAML, never JSONL.
    """
    metadata, entries, preferred = _unpack(_read_source(source))
    if not entries:
        raise ValueError("Dataset is empty; author at least one case before migration")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {index} must be an object, got {type(entry).__name__}")
    validate_case_ids(entry.get("id") for entry in entries)
    identity = _identity(metadata, entries, skill_name)
    converted = [_convert_entry(entry, index, preferred=preferred) for index, entry in enumerate(entries)]
    dataset = {**metadata, "skill_name": identity, "evals": converted}
    # Validate JSON serializability even for previews and YAML sources. This
    # rejects YAML-only values rather than silently stringifying authored data.
    rendered = json.dumps(dataset, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if destination is not None:
        if destination.resolve() == source.resolve():
            raise ValueError("Destination must be distinct from the source; choose a new output path")
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Destination already exists: {destination}; choose a new output path")
        suffix = destination.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml"}:
            raise ValueError("Object output requires .json, .yaml or .yml; JSONL is input-only for migration")
        if suffix != ".json":
            rendered = yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False)
        try:
            with destination.open("x", encoding="utf-8") as output:
                output.write(rendered)
        except FileExistsError as exc:
            raise ValueError(f"Destination already exists: {destination}; choose a new output path") from exc
    return MigrationResult(dataset=dataset, destination=destination)
