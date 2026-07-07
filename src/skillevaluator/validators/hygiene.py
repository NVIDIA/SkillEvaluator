# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code Integrity and Hygiene Validator.

Validates code hygiene: dead links, dependency auditing, test-file discovery.
"""

import re
from fnmatch import fnmatchcase
from pathlib import Path

from skillevaluator.constants import BANNED_PACKAGES
from skillevaluator.logging_config import get_logger
from skillevaluator.validators.base import ValidationResult, ValidatorBase, iter_scannable_files

logger = get_logger(__name__)

_TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")


class HygieneValidator(ValidatorBase):
    """Validates code integrity: dead links, dependencies, and test-file presence."""

    @property
    def name(self) -> str:
        return "Code Integrity & Hygiene"

    @property
    def description(self) -> str:
        return "Validate dead links, dependencies, and static Python test-file discovery"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Run hygiene checks on skill(s) at path."""
        return self._validate_folder_or_skill(
            skill_path,
            self._validate_single_skill,
            action_description="Checking code integrity for",
        )

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Run all hygiene checks on a single skill directory."""
        result = ValidationResult()
        result.merge(self._check_dead_links(skill_path))
        result.merge(self._audit_dependencies(skill_path))
        result.merge(self._check_test_presence(skill_path))
        return result

    def _check_dead_links(self, skill_path: Path) -> ValidationResult:
        """Verify all relative markdown links point to existing files.

        Markdown files under Tier 1 artifact directories (``evals/``,
        ``results/``, ``versions/`` and dot-prefixed variants) are skipped
        via :func:`iter_scannable_files` so that snapshot copies of the
        live skill do not produce duplicate dead-link reports.
        """
        result = ValidationResult()
        md_files = iter_scannable_files(skill_path, {".md"})

        if not md_files:
            result.add_success(
                check_name="dead_links",
                message="No markdown files found to check for links",
            )
            return result

        result.summary.files_scanned += len(md_files)
        result.add_success(
            check_name="dead_links_scan",
            message=f"Checking {len(md_files)} markdown files for dead links",
            file_count=len(md_files),
        )
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        # Pattern to match fenced code blocks (``` or ~~~)
        code_block_pattern = re.compile(r"(```|~~~).*?\1", re.DOTALL)

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception as e:
                result.add_warning(f"Could not read {md_file}: {e}")
                continue

            # Remove fenced code blocks before checking links
            # This prevents false positives from example links in documentation
            content_without_code = code_block_pattern.sub("", content)

            for link_text, link_href in link_pattern.findall(content_without_code):
                # Skip external URLs and anchors
                if link_href.startswith(("http://", "https://", "mailto:", "#")):
                    continue

                # Skip links inside inline code (backticks)
                # Check if this link appears in the original content within backticks
                inline_code_check = f"`[{link_text}]({link_href})`"
                if inline_code_check in content:
                    continue

                # Resolve and validate relative path
                clean_path = link_href.removeprefix("./").split("#")[0]
                target = md_file.parent / clean_path

                if not target.exists():
                    result.add_error(f"Dead link in {md_file.name}: [{link_text}]({link_href})")

        if not result.errors:
            result.add_success(
                check_name="dead_links",
                message=f"All relative links valid in {len(md_files)} markdown file(s)",
                files_checked=len(md_files),
            )

        return result

    def _audit_dependencies(self, skill_path: Path) -> ValidationResult:
        """Check for unpinned or banned packages in dependency files."""
        result = ValidationResult()
        found_any = False

        for req_name in ("requirements.txt", "requirements-dev.txt"):
            req_file = skill_path / req_name
            if req_file.exists():
                found_any = True
                result.merge(self._check_requirements_file(req_file))

        pyproject = skill_path / "pyproject.toml"
        if pyproject.exists():
            found_any = True
            result.add_success(
                check_name="pyproject_toml",
                message="Found pyproject.toml - dependencies managed by uv/pip",
            )

        if not found_any:
            result.add_success(
                check_name="dependencies",
                message="No dependency files found (requirements.txt, pyproject.toml)",
            )

        return result

    def _check_requirements_file(self, req_file: Path) -> ValidationResult:
        """Audit a single requirements.txt for banned/unpinned packages."""
        result = ValidationResult()

        try:
            lines = req_file.read_text(encoding="utf-8").strip().split("\n")
        except Exception as e:
            result.add_warning(f"Could not read {req_file}: {e}")
            return result

        result.add_success(
            check_name="dependency_audit",
            message=f"Auditing {req_file.name}",
        )
        pkg_pattern = re.compile(r"^([a-zA-Z0-9_-]+)")
        banned_lower = {p.lower() for p in BANNED_PACKAGES}

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith(("#", "-")):
                continue

            match = pkg_pattern.match(line)
            if not match:
                continue

            pkg_name = match.group(1).lower()

            if pkg_name in banned_lower:
                result.add_error(f"{req_file.name}:{line_num} - Banned package: {pkg_name}")
            elif not re.search(r"[=<>!]", line):
                result.add_warning(f"{req_file.name}:{line_num} - Unpinned: {line}")

        if not result.errors and not result.warnings:
            result.add_success(
                check_name="dependency_audit",
                message=f"{req_file.name} passed dependency audit",
            )

        return result

    def _check_test_presence(self, skill_path: Path) -> ValidationResult:
        """Discover conventional Python test filenames without reading or executing them."""
        result = ValidationResult()
        try:
            skill_root = skill_path.resolve(strict=True)
        except OSError:
            skill_root = skill_path.resolve()

        test_files: list[Path] = []
        for candidate in iter_scannable_files(skill_path, {".py"}):
            if candidate.is_symlink():
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_file() or not resolved.is_relative_to(skill_root):
                continue
            if any(fnmatchcase(candidate.name, pattern) for pattern in _TEST_FILE_PATTERNS):
                test_files.append(candidate)

        result.summary.files_scanned += len(test_files)
        metadata = {
            "test_count": len(test_files),
            "execution_performed": False,
            "coverage_measured": False,
            "patterns": list(_TEST_FILE_PATTERNS),
        }
        if test_files:
            message = (
                f"Found {len(test_files)} standard Python test-file candidate(s); "
                "target tests were not executed and coverage was not measured"
            )
        else:
            message = (
                "No standard Python test-file candidates found; target tests were not executed "
                "and coverage was not measured. Consider adding tests."
            )
            result.add_warning("No standard Python test-file candidates found; consider adding tests")
        result.add_success(check_name="test_discovery", message=message, **metadata)
        return result
