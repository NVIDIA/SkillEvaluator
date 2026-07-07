# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""License Compliance Validator.

Validates license compliance for Skills, Rules, and Workflows using multi-tier detection:
1. Explicit declaration in YAML frontmatter
2. LICENSE file pattern matching
3. SPDX header scanning in source files
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path

from skillevaluator.config import load_license_config
from skillevaluator.constants import (
    LICENSE_FILE_NAMES,
    LICENSE_HEADER_EXTENSIONS,
    LICENSE_HEADER_SCAN_LINES,
    SPDX_LICENSE_PATTERN,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.validators.base import Finding, ValidationResult, ValidatorBase, iter_scannable_files
from skillevaluator.validators.frontmatter_parser import parse_frontmatter

logger = get_logger(__name__)

# Compile regex once at module level for performance
_SPDX_PATTERN = re.compile(SPDX_LICENSE_PATTERN, re.IGNORECASE)
_LICENSE_SUFFIX_PATTERN = re.compile(r"(-license|-licence)$")
_THE_PREFIX_PATTERN = re.compile(r"^(the-)?")

# File reference indicators in license field values
_FILE_REFERENCE_KEYWORDS = frozenset(["see ", "refer to ", "license.txt", "license.md", "copying"])


@dataclass(slots=True)
class LicenseDetection:
    """Result of license detection attempt."""

    license_id: str | None
    source: str
    confidence: str
    file_path: str | None = None
    details: str | None = None


class LicenseValidator(ValidatorBase):
    """Validates license compliance for Skills, Rules, and Workflows.

    Uses a multi-tier detection approach to find license information and validates
    against a configurable allowlist of permissive open-source licenses.

    Detection tiers (in order):
    1. Frontmatter: Check 'license' field in SKILL.md or workflow-rules.mdc
    2. LICENSE file: Parse common license files and match against known patterns
    3. Source headers: Scan source files for SPDX-License-Identifier headers

    Configuration:
        The validator uses skillevaluator/config/license_config.yaml which defines:
        - allowed_licenses: Permissive licenses that pass validation
        - blocked_licenses: Restrictive licenses that fail validation
        - license_patterns: Patterns to detect licenses from file content
        - proprietary_indicators: Strings indicating proprietary licensing
    """

    def __init__(self, strict_mode: bool = False):
        """Initialize validator.

        Args:
            strict_mode: If True, fail on UNKNOWN licenses. If False, warn only.
        """
        self.strict_mode = strict_mode
        self._config: dict | None = None
        self._allowed_normalized: frozenset[str] | None = None
        self._blocked_normalized: frozenset[str] | None = None

    @cached_property
    def config(self) -> dict:
        """Lazy-load license configuration."""
        return load_license_config()

    @cached_property
    def _normalized_allowlist(self) -> frozenset[str]:
        """Pre-compute normalized allowlist for O(1) lookups."""
        return frozenset(self._normalize_license_id(lic) for lic in self.config.get("allowed_licenses", []))

    @cached_property
    def _normalized_blocklist(self) -> frozenset[str]:
        """Pre-compute normalized blocklist for O(1) lookups."""
        return frozenset(self._normalize_license_id(lic) for lic in self.config.get("blocked_licenses", []))

    @property
    def name(self) -> str:
        return "License Compliance"

    @property
    def description(self) -> str:
        return "Validate license compliance for Skills, Rules, and Workflows"

    def validate(self, asset_path: Path) -> ValidationResult:
        """Run license compliance validation on asset(s) at path."""
        if asset_path.is_dir() and not self._is_asset_directory(asset_path):
            return self._validate_folder_or_skill(
                asset_path,
                self._validate_single_asset,
                action_description="Checking license compliance for",
            )
        return self._validate_single_asset(asset_path)

    def _is_asset_directory(self, path: Path) -> bool:
        """Check if path is an asset directory (skill, rule, or workflow)."""
        return (
            self._find_skill_manifest(path) is not None
            or (path / "workflow-rules.mdc").exists()
            or any(path.glob("*.mdc"))
        )

    def _validate_single_asset(self, asset_path: Path) -> ValidationResult:
        """Validate license compliance for a single asset directory."""
        result = ValidationResult()
        detection = self._detect_license(asset_path, result)

        if detection is None:
            self._handle_no_license(result)
        else:
            self._validate_license(detection, result)

        return result

    def _handle_no_license(self, result: ValidationResult) -> None:
        """Handle case when no license is detected."""
        msg = "No license information found"
        if self.strict_mode:
            result.add_error(f"{msg} - cannot verify compliance")
        else:
            result.add_warning(f"{msg} - manual review required. Add a LICENSE file or 'license' field in frontmatter.")

    def _detect_license(self, asset_path: Path, result: ValidationResult) -> LicenseDetection | None:
        """Attempt to detect license using multi-tier approach."""
        # Tier 1: Frontmatter declaration
        if detection := self._check_frontmatter(asset_path):
            result.add_message(f"Tier 1: Found license declaration in {detection.source}")
            if detection.license_id and self._is_file_reference(detection.license_id):
                result.add_message(f"  License field references file: '{detection.license_id}'")
                if file_detection := self._check_license_file(asset_path):
                    return file_detection
            else:
                return detection

        # Tier 2: LICENSE file
        if detection := self._check_license_file(asset_path):
            result.add_message(f"Tier 2: Detected {detection.license_id} from {detection.file_path}")
            return detection

        # Tier 3: SPDX headers in source files
        if detection := self._scan_source_headers(asset_path):
            result.add_message(f"Tier 3: Found SPDX header '{detection.license_id}' in {detection.file_path}")
            return detection

        result.add_message("No license detected in any tier")
        return None

    def _check_frontmatter(self, asset_path: Path) -> LicenseDetection | None:
        """Tier 1: Extract license from YAML frontmatter."""
        manifest_candidates = [
            asset_path / "SKILL.md",
            asset_path / "skill.md",
            asset_path / "workflow-rules.mdc",
            *asset_path.glob("*.mdc"),
        ]

        for manifest in manifest_candidates:
            if not manifest.exists():
                continue
            try:
                parsed, _ = parse_frontmatter(manifest)
                if parsed and parsed.yaml_data and "license" in parsed.yaml_data:
                    return LicenseDetection(
                        license_id=str(parsed.yaml_data["license"]).strip(),
                        source=f"frontmatter ({manifest.name})",
                        confidence="high",
                        file_path=manifest.name,
                    )
            except Exception as e:
                logger.debug("Could not parse frontmatter in %s: %s", manifest, e)

        return None

    @staticmethod
    def _is_file_reference(license_value: str) -> bool:
        """Check if license value references a file."""
        lower = license_value.lower()
        return any(ref in lower for ref in _FILE_REFERENCE_KEYWORDS)

    def _check_license_file(self, asset_path: Path) -> LicenseDetection | None:
        """Tier 2: Parse LICENSE files and match against known patterns."""
        for license_name in LICENSE_FILE_NAMES:
            license_path = asset_path / license_name
            if not license_path.exists():
                continue

            try:
                content = license_path.read_text(encoding="utf-8", errors="ignore")

                if detected := self._identify_license_from_content(content):
                    return LicenseDetection(
                        license_id=detected["license_id"],
                        source="license_file",
                        confidence=detected["confidence"],
                        file_path=license_name,
                    )

                if indicator := self._find_proprietary_indicator(content):
                    return LicenseDetection(
                        license_id="Proprietary",
                        source="license_file",
                        confidence="high",
                        file_path=license_name,
                        details=f"Found proprietary indicator: '{indicator}'",
                    )
            except Exception as e:
                logger.debug("Could not read %s: %s", license_path, e)

        return None

    def _identify_license_from_content(self, content: str) -> dict | None:
        """Match LICENSE content against known patterns."""
        content_upper = content.upper()

        for license_id, pattern_config in self.config.get("license_patterns", {}).items():
            required = pattern_config.get("required", [])
            exclude = pattern_config.get("exclude", [])

            if not self._all_patterns_match(required, content, content_upper):
                continue
            if self._any_pattern_matches(exclude, content_upper):
                continue

            return {
                "license_id": license_id,
                "confidence": pattern_config.get("confidence", "medium"),
            }

        return None

    @staticmethod
    def _all_patterns_match(patterns: list[str], content: str, content_upper: str) -> bool:
        """Check if all required patterns match the content."""
        for pattern in patterns:
            if "|" in pattern:
                if not re.search(pattern, content, re.IGNORECASE):
                    return False
            elif pattern.upper() not in content_upper:
                return False
        return True

    @staticmethod
    def _any_pattern_matches(patterns: list[str], content_upper: str) -> bool:
        """Check if any exclusion pattern matches."""
        return any(p.upper() in content_upper for p in patterns)

    def _find_proprietary_indicator(self, content: str) -> str | None:
        """Find proprietary/restrictive license indicator in content."""
        content_lower = content.lower()
        for indicator in self.config.get("proprietary_indicators", []):
            if indicator.lower() in content_lower:
                return indicator
        return None

    def _scan_source_headers(self, asset_path: Path) -> LicenseDetection | None:
        """Tier 3: Scan source files for SPDX-License-Identifier headers.

        Files under Tier 1 artifact directories (``evals/``, ``results/``,
        ``versions/`` and dot-prefixed variants) are skipped via
        :func:`iter_scannable_files` so a vendored skill snapshot inside
        ``evals/`` cannot skew header detection.
        """
        found_licenses: dict[str, list[str]] = {}

        for file_path in iter_scannable_files(asset_path, LICENSE_HEADER_EXTENSIONS):
            if license_id := self._extract_spdx_from_file(file_path):
                found_licenses.setdefault(license_id, []).append(str(file_path.relative_to(asset_path)))

        if not found_licenses:
            return None

        # Return the most common license
        most_common = max(found_licenses, key=lambda k: len(found_licenses[k]))
        files = found_licenses[most_common]

        return LicenseDetection(
            license_id=most_common,
            source="spdx_header",
            confidence="high" if len(files) > 1 else "medium",
            file_path=files[0],
            details=f"Found in {len(files)} file(s)",
        )

    @staticmethod
    def _extract_spdx_from_file(file_path: Path) -> str | None:
        """Extract SPDX license identifier from file header."""
        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                header = "".join(islice(f, LICENSE_HEADER_SCAN_LINES))
                if match := _SPDX_PATTERN.search(header):
                    return match.group(1).strip()
        except Exception as e:
            logger.debug("Could not scan %s: %s", file_path, e)
        return None

    def _validate_license(self, detection: LicenseDetection, result: ValidationResult) -> None:
        """Validate detected license against allowlist/blocklist."""
        license_id = detection.license_id
        if not license_id:
            result.add_warning("License detection returned empty identifier")
            return

        normalized = self._normalize_license_id(license_id)

        if normalized in self._normalized_allowlist:
            self._set_license_metadata(result, detection, "allowed")
            result.add_message(f"License: {license_id} (ALLOWED - permissive)")
        elif normalized in self._normalized_blocklist:
            self._set_license_metadata(result, detection, "blocked")
            self._add_blocked_license_finding(result, license_id, detection.file_path)
        else:
            self._set_license_metadata(result, detection, "unknown")
            self._handle_unknown_license(result, license_id, detection.file_path)

    @staticmethod
    def _set_license_metadata(result: ValidationResult, detection: LicenseDetection, status: str) -> None:
        """Set license metadata on validation result."""
        result.metadata.update(
            {
                "license": detection.license_id,
                "license_status": status,
                "license_source": detection.source,
            }
        )

    @staticmethod
    def _add_blocked_license_finding(result: ValidationResult, license_id: str, file_path: str | None) -> None:
        """Add structured finding for blocked license."""
        result.add_structured_finding(
            Finding(
                category="LICENSE",
                severity="HIGH",
                check_name="blocked_license",
                message=f"License '{license_id}' is not allowed (restrictive/copyleft)",
                file_path=file_path or "unknown",
                suggestion=(
                    "This asset uses a restrictive license that is not permitted. "
                    "Contact the asset owner about re-licensing, or find an alternative."
                ),
            ),
            is_error=True,
        )

    def _handle_unknown_license(self, result: ValidationResult, license_id: str, file_path: str | None) -> None:
        """Handle license not in allowlist or blocklist."""
        if self.strict_mode:
            result.add_structured_finding(
                Finding(
                    category="LICENSE",
                    severity="MEDIUM",
                    check_name="unknown_license",
                    message=f"License '{license_id}' is not in the allowlist",
                    file_path=file_path or "unknown",
                    suggestion=(
                        f"Review license '{license_id}' and add to allowlist if permissive, "
                        "or blocklist if restrictive. See skillevaluator/config/license_config.yaml"
                    ),
                ),
                is_error=True,
            )
        else:
            result.add_warning(f"License '{license_id}' not in allowlist - manual review required")

    @staticmethod
    def _normalize_license_id(license_id: str) -> str:
        """Normalize license ID for case-insensitive comparison."""
        normalized = license_id.strip().lower().replace(" ", "-").replace("_", "-")
        normalized = _THE_PREFIX_PATTERN.sub("", normalized)
        return _LICENSE_SUFFIX_PATTERN.sub("", normalized)
