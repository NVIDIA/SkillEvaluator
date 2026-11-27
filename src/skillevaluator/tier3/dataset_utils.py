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
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml

from skillevaluator.quality_config import _yaml_error_detail
from skillevaluator.tier3.eval_core.optional_reference import normalize_expected_behaviors, normalize_optional_reference
from skillevaluator.utils.structured_data import ScalarConstructionError, StructuredDataSyntaxError, load_bounded_yaml

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


def load_dataset_entries_with_format(
    dataset_path: Path,
    *,
    inspect_authored_entry: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], DatasetFormat]:
    """Load normalized entries and optionally inspect each entry as authored."""
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
    return normalize_dataset_entries(data, inspect_authored_entry=inspect_authored_entry), dataset_format


def detect_dataset_format(data: Any) -> DatasetFormat:
    """Return the authoring format used by a parsed dataset payload."""
    if isinstance(data, dict) and isinstance(data.get("evals"), list):
        return "agentskills"
    return "legacy"


def normalize_dataset_entries(
    data: Any, *, inspect_authored_entry: Callable[[dict[str, Any]], None] | None = None
) -> list[dict[str, Any]]:
    """Normalize supported eval dataset shapes to runtime fields."""
    if isinstance(data, dict):
        if isinstance(data.get("evals"), list):
            skill_name = data.get("skill_name")
            if not isinstance(skill_name, str) or not skill_name.strip():
                raise ValueError("agentskills dataset requires a non-empty top-level skill_name")
            return _normalize_entries(
                data["evals"], skill_name=skill_name, inspect_authored_entry=inspect_authored_entry
            )
        cases = data.get("cases")
        data = cases if isinstance(cases, list) else [data]

    if isinstance(data, list):
        return _normalize_entries(data, inspect_authored_entry=inspect_authored_entry)

    return []


# ── private helpers ──────────────────────────────────────────────────────────


def _normalize_entries(
    entries: list[Any],
    *,
    inspect_authored_entry: Callable[[dict[str, Any]], None] | None,
    skill_name: str | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            location = (
                f"agentskills dataset evals[{index}]" if skill_name is not None else f"legacy dataset entry {index}"
            )
            raise ValueError(f"{location} must be an object, got {type(entry).__name__}")
        if inspect_authored_entry is not None:
            inspect_authored_entry(entry)
        normalized.append(_normalize_entry(entry, skill_name=skill_name, entry_index=index))
    return normalized


def _normalize_entry(
    entry: dict[str, Any],
    *,
    skill_name: Any | None = None,
    entry_index: int = 0,
) -> dict[str, Any]:
    """Map agentskills.io field names onto the existing runtime schema."""
    normalized = dict(entry)
    if "question" not in normalized and "prompt" in entry:
        normalized["question"] = entry["prompt"]
    if "ground_truth" not in normalized and "expected_output" in entry:
        normalized["ground_truth"] = entry["expected_output"]
    if "expected_behavior" not in normalized and "assertions" in entry:
        normalized["expected_behavior"] = entry["assertions"]
    if "expected_script" in normalized:
        try:
            normalized["expected_script"] = normalize_optional_reference(normalized["expected_script"])
        except TypeError as exc:
            raise ValueError(f"invalid eval dataset entry {entry_index}: {exc}") from exc
    if "expected_behavior" in normalized:
        normalized["expected_behavior"] = normalize_expected_behaviors(normalized["expected_behavior"])
    if isinstance(normalized.get("assertions"), list):
        normalized["assertions"] = normalize_expected_behaviors(normalized["assertions"])
    if "expected_skill" not in normalized and isinstance(skill_name, str) and skill_name:
        normalized["expected_skill"] = skill_name
    return normalized


def parse_dataset_text(text: str, suffix: str) -> Any:
    """Parse the text of a ``.json``, ``.jsonl``, ``.yaml`` or ``.yml`` dataset.

    A dataset ships with the skill, so YAML is read within the structured-data limits: plain
    ``yaml.safe_load`` expands chained merge keys inside the parser and shares aliases that a
    later ``str()`` or ``json.dumps`` expands without limit. Every failure, including nesting
    deeper than the JSON parser allows, is a ``ValueError``; the YAML limit errors are
    ``StructuredDataError``, which subclasses it.
    """
    suffix = suffix.casefold()
    try:
        if suffix == ".jsonl":
            # Records end at LF, CRLF or CR only, as when a file is read line by line. ``splitlines``
            # would also split at U+2028, U+2029 and U+0085, which JSON allows inside a string.
            lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            return [json.loads(line) for raw_line in lines if (line := raw_line.strip())]
        if suffix in {".yaml", ".yml"}:
            try:
                return load_bounded_yaml(text, last_key_wins=True)
            except StructuredDataSyntaxError as exc:
                raise StructuredDataSyntaxError(_yaml_syntax_message(exc)) from exc
        if suffix == ".json":
            return json.loads(text)
    except RecursionError as exc:
        raise ValueError("dataset is nested too deeply to read") from exc
    raise ValueError(f"Unsupported dataset format: {suffix}")


def _yaml_syntax_message(error: StructuredDataSyntaxError) -> str:
    """Keep what PyYAML says is wrong and where, but never the source text around it.

    The bounded loader reports only that the input is not valid YAML; the error it wraps has the
    parser's context, its problem, and a line and column. A dataset can be large, so the line and
    column are what the author needs to find the mistake.
    """
    cause = error.__cause__
    mark = getattr(cause, "problem_mark", None)
    if not isinstance(cause, yaml.YAMLError) or mark is None:
        return str(error)
    if isinstance(cause, ScalarConstructionError):
        return f"{error!s}: line {mark.line + 1}, column {mark.column + 1}: cannot parse a value"
    context = getattr(cause, "context", None)
    detail = _yaml_error_detail(cause)
    return f"{error!s}: {context}; {detail}" if context else f"{error!s}: {detail}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return parse_dataset_text(path.read_text(encoding="utf-8"), ".jsonl")


def _load_yaml(path: Path) -> Any:
    return parse_dataset_text(path.read_text(encoding="utf-8"), ".yaml")


def _load_json(path: Path) -> Any:
    return parse_dataset_text(path.read_text(encoding="utf-8"), ".json")
