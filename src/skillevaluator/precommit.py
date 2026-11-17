# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map changed files to skill roots and run one offline Tier 1 check per skill."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.constants import QUALITY_DEFAULT_MIN_SCORE, SCAN_EXCLUDED_DIRS, SKILL_MANIFEST_VARIANTS
from skillevaluator.tier1.commands import run_lint_scripts, run_quality_check
from skillevaluator.validators.policy import resolve_policy
from skillevaluator.validators.schema import SchemaValidator

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

CHECK_QUALITY = "quality"
CHECK_SCHEMA = "schema"
CHECK_LINT_SCRIPTS = "lint-scripts"
CHECKS = (CHECK_QUALITY, CHECK_SCHEMA, CHECK_LINT_SCRIPTS)
LINT_FINDINGS_SHOWN = 3


def parse_hook_path(value: str | Path) -> Path:
    """Interpret a pre-commit path in the host platform's own spelling.

    pre-commit passes tracked paths as the host spells them. On POSIX a
    backslash is a valid filename character, so it is kept as is; on Windows
    ``Path`` already treats it as a separator.
    """
    return Path(value)


def _has_skill_manifest(directory: Path, cache: dict[Path, bool] | None = None) -> bool:
    """Return True when *directory* holds a manifest named exactly as in ``SKILL_MANIFEST_VARIANTS``.

    Names are compared as stored on disk, so the result is the same on
    case-insensitive (macOS, Windows) and case-sensitive (Linux) filesystems.
    When *cache* is given, each directory is listed at most once per cache.
    """
    if cache is not None and directory in cache:
        return cache[directory]
    try:
        found = any(child.name in SKILL_MANIFEST_VARIANTS and child.is_file() for child in directory.iterdir())
    except OSError:
        found = False
    if cache is not None:
        cache[directory] = found
    return found


def _in_excluded_dir(directory: Path, limit: Path) -> bool:
    """Return True when *directory*, relative to *limit*, sits in a ``SCAN_EXCLUDED_DIRS`` folder.

    Folder-mode validation skips manifests under these folders (eval fixtures,
    version snapshots, results), so the hook does the same.
    """
    try:
        relative = directory.resolve().relative_to(limit)
    except (OSError, ValueError):
        return False
    return any(part in SCAN_EXCLUDED_DIRS for part in relative.parts)


def skill_root_for(
    path: str | Path,
    *,
    cwd: Path | None = None,
    _cache: dict[Path, Path | None] | None = None,
    _manifest_cache: dict[Path, bool] | None = None,
) -> Path | None:
    """Return the nearest ancestor of *path* that contains ``SKILL.md`` or ``skill.md``.

    Deleted skills and paths that never sit under a skill are ignored. Search
    stops at *cwd* (the repository root pre-commit runs from) so a stray
    manifest above the project is not treated as a skill root. A manifest in
    a ``SCAN_EXCLUDED_DIRS`` folder (``evals/``, ``.versions/``, ...) is not
    a skill root either; the walk continues up to the real skill.

    ``_cache`` and ``_manifest_cache`` let ``collect_skill_roots`` share
    lookups across paths so each directory is listed once per invocation.
    """
    cwd = Path.cwd() if cwd is None else Path(cwd)
    candidate = parse_hook_path(path)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    if candidate.name in SKILL_MANIFEST_VARIANTS:
        start = candidate.parent
    elif candidate.exists() and candidate.is_dir():
        start = candidate
    else:
        start = candidate.parent

    if _cache is not None and start in _cache:
        return _cache[start]

    try:
        limit = cwd.resolve()
    except OSError:
        limit = cwd

    root: Path | None = None
    for ancestor in [start, *start.parents]:
        if not _in_excluded_dir(ancestor, limit) and _has_skill_manifest(ancestor, _manifest_cache):
            try:
                root = ancestor.resolve()
            except OSError:
                root = ancestor
            break
        try:
            if ancestor.resolve() == limit:
                break
        except OSError:
            if ancestor == cwd:
                break
    if _cache is not None:
        _cache[start] = root
    return root


def collect_skill_roots(paths: Iterable[str | Path], *, cwd: Path | None = None) -> list[Path]:
    """Return unique skill roots for *paths*, in first-seen order."""
    roots: list[Path] = []
    seen: set[Path] = set()
    start_cache: dict[Path, Path | None] = {}
    manifest_cache: dict[Path, bool] = {}
    for raw in paths:
        root = skill_root_for(raw, cwd=cwd, _cache=start_cache, _manifest_cache=manifest_cache)
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


def _strip_location(message: str) -> str:
    """Return the first line of *message* without its absolute-path location.

    YAML parse errors carry context lines (a copy of the offending source), so
    only the first line is kept to hold the one-line-per-skill format.
    """
    first_line = message.splitlines()[0] if message else ""
    return first_line.split(" in /", 1)[0]


def _first_error(results: Sequence[ValidationResult]) -> str:
    for result in results:
        for message in (*result.gate_failures, *result.errors):
            return _strip_location(message)
    return ""


def _lint_findings(results: Sequence[ValidationResult]) -> list[str]:
    return [_strip_location(message) for result in results for message in (*result.errors, *result.warnings)]


def format_lint_findings(results: Sequence[ValidationResult], *, limit: int = LINT_FINDINGS_SHOWN) -> list[str]:
    """Return indented lines for the first *limit* script lint findings."""
    findings = _lint_findings(results)
    lines = [f"    {message}" for message in findings[:limit]]
    if len(findings) > limit:
        lines.append(f"    ... and {len(findings) - limit} more (run `skillevaluator lint-scripts` for all)")
    return lines


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
            if score is not None and score < min_score:
                extra += f"  (min {min_score})"
            elif detail := _first_error(results):
                extra += f"  {detail}"
    elif check == CHECK_LINT_SCRIPTS:
        count = len(_lint_findings(results))
        extra = f"  {count} finding{'' if count == 1 else 's'}"
    elif not passed:
        detail = _first_error(results)
        if detail:
            extra = f"  {detail}"
    return f"{status}  {skill_name}  {check}{extra}"


def run_check(
    check: str, skill_root: Path, *, min_score: int, quality_config: Path | None = None
) -> list[ValidationResult]:
    """Run one offline check against a single skill directory. Never writes reports.

    *quality_config* replaces the packaged quality-default.yaml for the quality
    check, the same way ``quality-check --quality-config`` does.
    """
    if check == CHECK_QUALITY:
        return run_quality_check(skill_root, min_score=min_score, quality_config=quality_config)
    if check == CHECK_SCHEMA:
        # Same policy as ``validate``: ``SKILLEVALUATOR_PROFILE`` picks the profile.
        return [SchemaValidator(policy=resolve_policy()).validate(skill_root)]
    if check == CHECK_LINT_SCRIPTS:
        return run_lint_scripts(skill_root)
    raise ValueError(f"Unknown check: {check}")


def run_precommit(
    check: str,
    paths: Sequence[str | Path],
    *,
    min_score: int = QUALITY_DEFAULT_MIN_SCORE,
    quality_config: Path | None = None,
    cwd: Path | None = None,
) -> tuple[int, list[str]]:
    """Run *check* once per skill touched by *paths*.

    Returns ``(exit_code, lines)``. Exit code is 1 when any skill fails and 0
    otherwise, including when every path is outside a skill or the skill was
    deleted. A check that raises for one skill is reported as a FAIL line for
    that skill and the remaining skills are still checked.
    """
    if check not in CHECKS:
        raise ValueError(f"Unknown check: {check}")
    lines: list[str] = []
    failed = False
    for root in collect_skill_roots(paths, cwd=cwd):
        try:
            results = run_check(check, root, min_score=min_score, quality_config=quality_config)
        except Exception as exc:  # one broken skill must not hide the others
            lines.append(f"FAIL  {root.name}  {check}  error: {type(exc).__name__}: {exc}")
            failed = True
            continue
        line = format_result_line(root.name, check, results, min_score=min_score)
        lines.append(line)
        if check == CHECK_LINT_SCRIPTS:
            lines.extend(format_lint_findings(results))
        if not all(result.passed for result in results):
            failed = True
    return (1 if failed else 0, lines)
