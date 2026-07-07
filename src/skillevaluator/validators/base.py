# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base validator class and result model.

This module provides the foundation for all SkillEvaluator validators with:
- Finding: Structured validation finding with full context for CI reporting
- ValidationResult: Dataclass for collecting validation outcomes
- ValidatorBase: Abstract base class with folder-level validation support

Note: The Finding and ValidationResult classes are now defined in skillevaluator.models.result
and re-exported here for backward compatibility. New code should import from
skillevaluator.models directly.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from skillevaluator.constants import SCAN_EXCLUDED_DIRS, SCAN_EXCLUDED_FILES, SKILL_MANIFEST_VARIANTS

# Import from the canonical location in models
from skillevaluator.models.result import (
    Finding,
    Severity,
    SuccessDetail,
    ValidationResult,
    ValidationSummary,
)

# Re-export for backward compatibility
__all__ = [
    "Finding",
    "Severity",
    "SuccessDetail",
    "ValidationResult",
    "ValidationSummary",
    "ValidatorBase",
    "continue_on_failure_scope",
    "iter_scannable_files",
]

# When enabled (via ``run_validation`` honoring ``--continue-on-failure``),
# batch folder validation records every skill instead of stopping at the first
# CRITICAL finding. Defaults to Skill Evaluator's historical stop-on-critical behavior.
_CONTINUE_ON_FAILURE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "skillevaluator_continue_on_failure", default=False
)


@contextlib.contextmanager
def continue_on_failure_scope(enabled: bool) -> Iterator[None]:
    """Toggle continue-on-failure behavior for the duration of the scope.

    While *enabled* is True, :meth:`ValidatorBase._validate_folder_or_skill`
    keeps scanning and recording every skill in a folder even after a CRITICAL
    finding, rather than returning early (parity with Skill Evaluator
    ``--continue-on-failure``).
    """
    token = _CONTINUE_ON_FAILURE.set(enabled)
    try:
        yield
    finally:
        _CONTINUE_ON_FAILURE.reset(token)


def iter_scannable_files(
    root: Path,
    extensions: Iterable[str],
    excluded_dirs: Iterable[str] | None = None,
    excluded_files: Iterable[str] | None = None,
) -> list[Path]:
    """Walk ``root`` and return files matching any of ``extensions``.

    Skips files whose path traverses any directory listed in ``excluded_dirs``
    (defaults to :data:`SCAN_EXCLUDED_DIRS`). The exclusion matches at any
    depth so that ``evals/results/.../foo.txt`` and a re-occurrence under
    ``references/evals/foo.md`` are both filtered out. Excluded directories
    are pruned during the walk (never descended into), and symlinked
    directories are not followed.
    Skips generated publishing artifacts listed in ``excluded_files`` (defaults
    to :data:`SCAN_EXCLUDED_FILES`) by basename, case-insensitively.

    If ``root`` is itself a file, returns ``[root]`` when its suffix matches
    one of ``extensions`` and its basename is not excluded (no directory walk
    performed).

    Args:
        root: Directory or file path to scan.
        extensions: Iterable of file extensions including the leading dot
            (e.g. ``{".md", ".py"}``). Matching is case-sensitive on
            case-sensitive filesystems (Linux), preserving the prior
            ``rglob("*.md")`` behavior of every Tier 1 call site this
            helper replaced.
        excluded_dirs: Directory names to skip at any depth. Pass an empty
            iterable to disable exclusion entirely (e.g. for diagnostic
            scans). ``None`` uses :data:`SCAN_EXCLUDED_DIRS`.
        excluded_files: File basenames to skip. Pass an empty iterable to
            disable filename exclusion. ``None`` uses
            :data:`SCAN_EXCLUDED_FILES`.

    Returns:
        List of file paths in arbitrary order (callers that need a stable
        order should sort the result themselves).
    """
    excluded = SCAN_EXCLUDED_DIRS if excluded_dirs is None else frozenset(excluded_dirs)
    excluded_basenames = (
        SCAN_EXCLUDED_FILES if excluded_files is None else frozenset(name.lower() for name in excluded_files)
    )
    ext_set = set(extensions)

    if root.is_file():
        if root.name.lower() in excluded_basenames:
            return []
        return [root] if root.suffix in ext_set else []

    # Prune excluded dirs during traversal instead of filtering rglob
    # output: a skill that has been through Tier 3 can hold tens of
    # thousands of artifact files under evals/results/, and an
    # enumerate-then-filter walk still descends into all of them —
    # once per extension. One pruned walk covers every extension.
    suffixes = tuple(ext_set)
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excluded]
        for name in filenames:
            if not name.endswith(suffixes):
                continue
            if name.lower() in excluded_basenames:
                continue
            out.append(Path(dirpath) / name)
    return out


class ValidatorBase(ABC):
    """Abstract base class for skill validators.

    Provides common functionality for:
    - Finding SKILL.md files in directories
    - Folder-level validation (validating multiple skills at once)
    - Template method for consistent validation patterns
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the validator name for display purposes."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a brief description of what this validator checks."""
        ...

    @abstractmethod
    def validate(self, skill_path: Path) -> ValidationResult:
        """Validate a skill or folder of skills.

        Args:
            skill_path: Path to skill directory or folder containing skills

        Returns:
            ValidationResult with pass/fail status and details
        """
        ...

    def _find_exact_child(self, directory: Path, filename: str) -> Path | None:
        """Return a child path whose on-disk basename exactly matches ``filename``."""
        try:
            for child in directory.iterdir():
                if child.is_file() and child.name == filename:
                    return child
        except OSError:
            return None
        return None

    def _find_skill_manifest(self, skill_path: Path) -> Path | None:
        """Locate the skill manifest, preferring canonical exact casing."""
        for name in SKILL_MANIFEST_VARIANTS:
            if manifest := self._find_exact_child(skill_path, name):
                return manifest
        return None

    def _is_lowercase_manifest(self, skill_path: Path) -> bool:
        """Check if the skill manifest uses non-canonical lowercase naming.

        Per agentskills.io spec, the canonical name is SKILL.md (uppercase).
        On case-sensitive filesystems (Linux), skill.md is a distinct file
        that won't be recognized by spec-compliant tooling.
        """
        canonical = self._find_exact_child(skill_path, "SKILL.md")
        lowercase = self._find_exact_child(skill_path, "skill.md")
        return lowercase is not None and canonical is None

    def _find_all_skills(self, root_path: Path) -> list[Path]:
        """Recursively find all skill directories containing SKILL.md files.

        Manifests beneath any directory in :data:`SCAN_EXCLUDED_DIRS` (e.g.
        ``evals/``, ``results/``, ``.versions/``) are skipped so that
        evaluation snapshots and version archives do not get scanned as
        live skills.
        """
        skill_dirs: set[Path] = set()
        for manifest_name in SKILL_MANIFEST_VARIANTS:
            for skill_md in root_path.rglob(manifest_name):
                if any(part in SCAN_EXCLUDED_DIRS for part in skill_md.parts):
                    continue
                skill_dirs.add(skill_md.parent)
        return sorted(skill_dirs)

    def _is_skill_directory(self, path: Path) -> bool:
        """Check if path is a skill directory (contains SKILL.md)."""
        return self._find_skill_manifest(path) is not None

    def _validate_folder_or_skill(
        self,
        skill_path: Path,
        single_skill_validator: Callable[[Path], ValidationResult],
        action_description: str,
        no_skills_fallback: bool = True,
    ) -> ValidationResult:
        """Template method for folder-level or single-skill validation.

        Eliminates duplicated folder detection logic across validators.

        Args:
            skill_path: Path to validate
            single_skill_validator: Function to validate a single skill
            action_description: Description for progress messages (e.g., "Scanning")
            no_skills_fallback: If True, validate folder directly when no skills found

        Returns:
            Aggregated ValidationResult for all skills
        """
        if self._is_skill_directory(skill_path):
            return single_skill_validator(skill_path)

        result = ValidationResult()
        skill_dirs = self._find_all_skills(skill_path)

        if not skill_dirs:
            if no_skills_fallback:
                result.add_warning(f"No skills found in {skill_path}. {action_description} folder directly.")
                return single_skill_validator(skill_path)
            result.add_error(f"No skills found in {skill_path}. Expected SKILL.md files in skill directories.")
            return result

        # Use structured success details
        result.add_success(
            check_name="skill_discovery",
            message=f"{action_description} {len(skill_dirs)} skill(s)",
            skill_count=len(skill_dirs),
        )

        for skill_dir in skill_dirs:
            skill_result = single_skill_validator(skill_dir)
            skill_name = skill_dir.name

            if skill_result.passed:
                # Collect detailed check information from the skill result
                check_details = [
                    {
                        "name": detail.check_name,
                        "description": detail.message,
                        "metadata": detail.metadata,
                    }
                    for detail in skill_result.success_details
                ]

                result.add_success(
                    check_name=skill_name,
                    message="All checks passed",
                    checks=check_details,
                    total_checks=len(check_details),
                )
            else:
                result.merge_with_prefix(skill_result, skill_name)

            # Fail fast on CRITICAL findings (e.g., missing API keys). These are
            # typically configuration errors that affect all skills, so by
            # default we stop early. --continue-on-failure suppresses this so
            # every skill is still scanned and recorded.
            critical_findings = [f for f in skill_result.findings if f.severity == Severity.CRITICAL]
            if critical_findings and not _CONTINUE_ON_FAILURE.get():
                # Return early with just this critical finding
                return result

        return result
