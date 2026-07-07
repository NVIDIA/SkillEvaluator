# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared dataset loading utilities.

Supports JSONL, JSON, YAML, and YML dataset formats. All functions are used
across the evaluator, scripts, and UI to avoid duplicating format-detection
and parsing logic.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)

DATASET_EXTENSIONS = (".json", ".jsonl", ".yaml", ".yml")
DatasetFormat = Literal["agentskills", "legacy"]


def find_eval_file(skill_dir: Path) -> Path | None:
    """Find an eval file for a skill, checking both new and legacy locations.

    Search order:
      1. ``skill_dir/evals/evals.{json,jsonl,yaml,yml}`` (new convention)
      2. ``skill_dir/eval/dataset.{jsonl,yaml,yml,json}`` (legacy convention)

    Args:
        skill_dir: Root directory of the skill (e.g. ``skills/git-skill``).

    Returns:
        Path to the first matching file, or ``None``.
    """
    # New convention: evals/evals.*
    for ext in DATASET_EXTENSIONS:
        candidate = skill_dir / "evals" / f"evals{ext}"
        if candidate.exists():
            return candidate

    # Legacy convention: eval/dataset.*
    for ext in DATASET_EXTENSIONS:
        candidate = skill_dir / "eval" / f"dataset{ext}"
        if candidate.exists():
            return candidate

    return None


def find_dataset_file(eval_dir: Path, stem: str = "dataset") -> Path | None:
    """Find a dataset file in *eval_dir*, trying extensions in priority order.

    Legacy helper — prefer :func:`find_eval_file` for new code.
    """
    for ext in DATASET_EXTENSIONS:
        candidate = eval_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_dataset_entries(dataset_path: Path) -> list[dict[str, Any]]:
    """Load normalized dataset entries from a JSONL, JSON, YAML, or YML file.

    Legacy SkillEvaluator datasets use a flat list of entries with ``question``,
    ``ground_truth``, and ``expected_behavior`` fields.  The agentskills.io
    convention wraps entries in ``{"skill_name": "...", "evals": [...]}`` and
    names those fields ``prompt``, ``expected_output``, and ``assertions``.
    Runtime code consumes the legacy field names, so this loader accepts
    both authoring formats and normalizes the agentskills.io shape at the
    boundary.

    Args:
        dataset_path: Path to the dataset file.

    Returns:
        List of entry dicts.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    entries, _format = load_dataset_entries_with_format(dataset_path)
    return entries


def load_dataset_entries_with_format(dataset_path: Path) -> tuple[list[dict[str, Any]], DatasetFormat]:
    """Load normalized dataset entries and report the detected authoring format."""
    suffix = dataset_path.suffix.lower()

    if suffix == ".jsonl":
        data = _load_jsonl(dataset_path)
    elif suffix in (".yaml", ".yml"):
        data = _load_yaml(dataset_path)
    elif suffix == ".json":
        data = _load_json(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    dataset_format = detect_dataset_format(data)
    return normalize_dataset_entries(data), dataset_format


def detect_dataset_format(data: Any) -> DatasetFormat:
    """Return the authoring format used by a parsed dataset payload."""
    if isinstance(data, dict) and isinstance(data.get("evals"), list):
        return "agentskills"
    return "legacy"


def normalize_dataset_entries(data: Any) -> list[dict[str, Any]]:
    """Normalize supported eval dataset shapes to runtime fields."""
    if isinstance(data, dict):
        if isinstance(data.get("evals"), list):
            skill_name = data.get("skill_name")
            if not isinstance(skill_name, str) or not skill_name.strip():
                raise ValueError("agentskills dataset requires a non-empty top-level skill_name")
            entries: list[dict[str, Any]] = []
            for idx, entry in enumerate(data["evals"]):
                if not isinstance(entry, dict):
                    raise ValueError(f"agentskills dataset evals[{idx}] must be an object, got {type(entry).__name__}")
                entries.append(_normalize_entry(entry, skill_name=skill_name))
            return entries
        if isinstance(data.get("cases"), list):
            return [_normalize_entry(entry) for entry in data["cases"] if isinstance(entry, dict)]
        return [_normalize_entry(data)]

    if isinstance(data, list):
        return [_normalize_entry(entry) for entry in data if isinstance(entry, dict)]

    return []


# ── private helpers ──────────────────────────────────────────────────────────


def _normalize_entry(entry: dict[str, Any], *, skill_name: Any | None = None) -> dict[str, Any]:
    """Map agentskills.io field names onto the existing runtime schema."""
    normalized = dict(entry)
    if "question" not in normalized and "prompt" in entry:
        normalized["question"] = entry["prompt"]
    if "ground_truth" not in normalized and "expected_output" in entry:
        normalized["ground_truth"] = entry["expected_output"]
    if "expected_behavior" not in normalized and "assertions" in entry:
        normalized["expected_behavior"] = entry["assertions"]
    if "expected_skill" not in normalized and isinstance(skill_name, str) and skill_name:
        normalized["expected_skill"] = skill_name
    return normalized


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_yaml(path: Path) -> Any:
    with path.open() as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)
