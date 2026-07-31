# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic version validation for skills.

The previous-version bound is supplied via the ``previous_version`` argument or
the ``SKILLEVALUATOR_PREVIOUS_VERSION`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.models.skill import SEMVER_RE
from skillevaluator.validators.base import ValidatorBase
from skillevaluator.validators.frontmatter_parser import parse_frontmatter

PREVIOUS_VERSION_ENV = "SKILLEVALUATOR_PREVIOUS_VERSION"


def _extract_version(frontmatter: dict[str, Any]) -> str:
    """Return ``metadata.version`` only.

    The legacy top-level ``version`` field is intentionally ignored: prior to
    the optional-version proposal, ``quality_score.py`` recommended a top-level
    ``version`` without enforcing strict semver, so existing skills may carry
    values like ``"1.0"`` or ``1`` that would now produce HIGH findings. Scoping
    extraction to ``metadata.version`` keeps the label itself optional and
    avoids breaking those skills when the validator runs by default.
    """
    metadata = frontmatter.get("metadata")
    if isinstance(metadata, dict) and "version" in metadata:
        raw_value = metadata["version"]
        if raw_value is None:
            return ""
        value = str(raw_value).strip()
        if value:
            return value
    return ""


def _version_tuple(version: str) -> tuple[int, int, int] | None:
    """Parse a prevalidated version without allowing conversion failures to escape."""
    try:
        major, minor, patch = version.split(".")
        return int(major), int(minor), int(patch)
    except (TypeError, ValueError):
        return None


class VersionValidator(ValidatorBase):
    """Validate optional skill semantic version labels."""

    def __init__(self, previous_version: str | None = None) -> None:
        self.previous_version = previous_version if previous_version is not None else os.getenv(PREVIOUS_VERSION_ENV)

    @property
    def name(self) -> str:
        return "Semantic Version Validation"

    @property
    def description(self) -> str:
        return "Validate optional metadata.version labels and require strict bumps"

    def validate(self, skill_path: Path) -> ValidationResult:
        return self._validate_folder_or_skill(
            skill_path,
            self._validate_single_skill,
            "Validating semantic versions for",
            no_skills_fallback=False,
        )

    def _validate_previous_version(
        self,
        current: str,
        manifest: Path,
        result: ValidationResult,
    ) -> bool:
        """Validate the optional previous version bound.

        Returns ``True`` when validation can continue.
        """
        if not self.previous_version:
            return True

        previous = self.previous_version.strip()
        if not SEMVER_RE.fullmatch(previous):
            result.add_finding(
                Finding(
                    category="VERSION",
                    severity=Severity.HIGH,
                    check_name="previous_version_semver",
                    message=f"Previous version '{previous}' is not strict semantic version x.y.z",
                    file_path=str(manifest),
                    suggestion="Pass --previous-version as major.minor.patch",
                )
            )
            return False

        if not current:
            result.add_finding(
                Finding(
                    category="VERSION",
                    severity=Severity.HIGH,
                    check_name="version_missing",
                    message=(
                        "metadata.version is missing even though a previous version "
                        f"('{previous}') was supplied"
                    ),
                    file_path=str(manifest),
                    suggestion="Keep metadata.version and bump major, minor, or patch",
                )
            )
            return False

        current_tuple = _version_tuple(current)
        previous_tuple = _version_tuple(previous)
        if current_tuple is None or previous_tuple is None:
            check_name = "version_semver" if current_tuple is None else "previous_version_semver"
            invalid_value = current if current_tuple is None else previous
            result.add_finding(
                Finding(
                    category="VERSION",
                    severity=Severity.HIGH,
                    check_name=check_name,
                    message=f"Version '{invalid_value}' could not be parsed as strict semantic version x.y.z",
                    file_path=str(manifest),
                    suggestion="Use bounded ASCII major.minor.patch components without leading zeroes",
                )
            )
            return False

        if current_tuple <= previous_tuple:
            result.add_finding(
                Finding(
                    category="VERSION",
                    severity=Severity.HIGH,
                    check_name="version_monotonic",
                    message=f"Version '{current}' must be greater than previous '{previous}'",
                    file_path=str(manifest),
                    suggestion=(
                        "Bump major, minor, or patch before submitting the change; "
                        "reusing the previous version label is not allowed"
                    ),
                )
            )
            return False

        return True

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Validate one skill's optional ``metadata.version`` label."""
        result = ValidationResult(
            validator_name=self.name,
            validator_description=self.description,
        )
        manifest = self._find_skill_manifest(skill_path)
        if manifest is None:
            result.add_error(f"No SKILL.md found in {skill_path}")
            return result

        parsed, parse_result = parse_frontmatter(manifest)
        result.merge(parse_result)
        if parsed is None or parsed.yaml_data is None:
            return result

        current = _extract_version(parsed.yaml_data)
        if not current:
            if not self._validate_previous_version(current, manifest, result):
                return result
            result.add_success(
                check_name="version_optional",
                message="No semantic version label present; resource will use commit-hash history",
                previous_version=self.previous_version,
            )
            return result

        if not SEMVER_RE.fullmatch(current):
            result.add_finding(
                Finding(
                    category="VERSION",
                    severity=Severity.HIGH,
                    check_name="version_semver",
                    message=f"Version '{current}' is not strict semantic version x.y.z",
                    file_path=str(manifest),
                    suggestion=(
                        "Use a quoted string in major.minor.patch form, e.g. "
                        'version: "1.2.3" (unquoted YAML values like 1.0 are '
                        "parsed as numbers and stringified without a patch component)"
                    ),
                )
            )
            return result

        if not self._validate_previous_version(current, manifest, result):
            return result

        result.add_success(
            check_name="version_semver",
            message=f"Valid semantic version: {current}",
            version=current,
            previous_version=self.previous_version,
        )
        return result
