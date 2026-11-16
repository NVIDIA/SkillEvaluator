# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map changed files to skill roots and run one offline Tier 1 check per skill."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.constants import QUALITY_DEFAULT_MIN_SCORE, SKILL_MANIFEST_FILE
from skillevaluator.tier1.commands import run_lint_scripts, run_quality_check
from skillevaluator.validators.schema import SchemaValidator

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

CHECK_QUALITY = "quality"
CHECK_SCHEMA = "schema"
CHECK_LINT_SCRIPTS = "lint-scripts"
CHECKS = (CHECK_QUALITY, CHECK_SCHEMA, CHECK_LINT_SCRIPTS)


def parse_hook_path(value: str | Path) -> Path:
    """Interpret a pre-commit path, including Windows-style separators on POSIX."""
    text = str(value)
    if "\\" in text:
        text = text.replace("\\", "/")
    return Path(text)


def skill_root_for(path: str | Path, *, cwd: Path | None = None) -> Path | None:
    """Return the nearest ancestor of *path* that contains ``SKILL.md``.

    Deleted skills and paths that never sit under a skill are ignored. Search
    stops at *cwd* (the repository root pre-commit runs from) so a stray
    manifest above the project is not treated as a skill root.
    """
    cwd = Path.cwd() if cwd is None else Path(cwd)
    candidate = parse_hook_path(path)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    if candidate.name == SKILL_MANIFEST_FILE:
        start = candidate.parent
    elif candidate.exists() and candidate.is_dir():
        start = candidate
    else:
        start = candidate.parent

    try:
        limit = cwd.resolve()
    except OSError:
        limit = cwd

    for ancestor in [start, *start.parents]:
        manifest = ancestor / SKILL_MANIFEST_FILE
        try:
            if manifest.is_file():
                return ancestor.resolve()
        except OSError:
            pass
        try:
            if ancestor.resolve() == limit:
                break
        except OSError:
            if ancestor == cwd:
                break
    return None


def collect_skill_roots(paths: Iterable[str | Path], *, cwd: Path | None = None) -> list[Path]:
    """Return unique skill roots for *paths*, in first-seen order."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        root = skill_root_for(raw, cwd=cwd)
        if root is None or root in seen:
            continue
        seen.add(root)
        roots.append(root)
    return roots


def _quality_score(results: Sequence[ValidationResult]) -> float | None:
    for result in results:
        scores = result.metadata.get("quality_scores")
        if isinstance(scores, dict) and "overall_score" in scores:
            try:
                return float(scores["overall_score"])
            except (TypeError, ValueError):
                return None
    return None


def _first_error(results: Sequence[ValidationResult]) -> str:
    for result in results:
        for message in (*result.gate_failures, *result.errors):
            return message.split(" in /", 1)[0]
    return ""


def format_result_line(
    skill_name: str,
    check: str,
    results: Sequence[ValidationResult],
    *,
    min_score: int,
) -> str:
    """Return one compact PASS/FAIL line for a skill."""
    passed = all(result.passed for result in results)
    status = "PASS" if passed else "FAIL"
    extra = ""
    if check == CHECK_QUALITY:
        score = _quality_score(results)
        if score is not None:
            extra = f"  {score:.1f}/100"
            if not passed:
                extra += f"  (min {min_score})"
    elif not passed:
        detail = _first_error(results)
        if detail:
            extra = f"  {detail}"
    return f"{status}  {skill_name}  {check}{extra}"


def run_check(check: str, skill_root: Path, *, min_score: int) -> list[ValidationResult]:
    """Run one offline check against a single skill directory. Never writes reports."""
    if check == CHECK_QUALITY:
        return run_quality_check(skill_root, min_score=min_score)
    if check == CHECK_SCHEMA:
        return [SchemaValidator().validate(skill_root)]
    if check == CHECK_LINT_SCRIPTS:
        return run_lint_scripts(skill_root)
    raise ValueError(f"Unknown check: {check}")


def run_precommit(
    check: str,
    paths: Sequence[str | Path],
    *,
    min_score: int = QUALITY_DEFAULT_MIN_SCORE,
    cwd: Path | None = None,
) -> tuple[int, list[str]]:
    """Run *check* once per skill touched by *paths*.

    Returns ``(exit_code, lines)``. Exit code is 1 when any skill fails and 0
    otherwise, including when every path is outside a skill or the skill was
    deleted.
    """
    if check not in CHECKS:
        raise ValueError(f"Unknown check: {check}")
    lines: list[str] = []
    failed = False
    for root in collect_skill_roots(paths, cwd=cwd):
        results = run_check(check, root, min_score=min_score)
        line = format_result_line(root.name, check, results, min_score=min_score)
        lines.append(line)
        if not all(result.passed for result in results):
            failed = True
    return (1 if failed else 0, lines)
