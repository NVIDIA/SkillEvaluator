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
from dataclasses import dataclass, replace
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
_SPDX_EXPRESSION_TOKEN = re.compile(r"[()]|[^\s()]+")
_SPDX_OPERATORS = frozenset({"AND", "OR", "WITH"})
_SPDX_VERSION_VARIANT_SUFFIX = re.compile(r"-(?:only|or-later)$")
# Parser states where a license name or "(" comes next, and states where the
# expression may end or continue with AND, OR or ")".
_EXPRESSION_OPERAND_STATES = frozenset({"start", "(", "AND", "OR"})
_EXPRESSION_COMPLETE_STATES = frozenset({"license", "exception", ")"})


def _strip_spdx_capture(raw: str, line_prefix: str = "") -> str:
    """Drop inline comment terminators and trailing source after the SPDX value.

    A backtick or ``|`` ends the value only when the same delimiter opens before
    the tag on that line (Markdown inline code or a table cell), so a header such
    as ``MIT | GPL-3.0`` is not shortened to ``MIT``.
    """
    raw = raw.split("#", 1)[0]
    markers = ["*/", "-->", *(delimiter for delimiter in ("`", "|") if delimiter in line_prefix)]
    for marker in markers:
        idx = raw.find(marker)
        if idx != -1:
            raw = raw[:idx]
    return raw.strip().rstrip(";").rstrip()


@dataclass(slots=True, frozen=True)
class _LicenseExpression:
    """License names and ``WITH`` exceptions of a license value, or its syntax error.

    ``names`` holds every name written in the value, whatever its position and
    even when the value does not parse, so the blocked check stays fail-closed.
    """

    licenses: tuple[str, ...] = ()
    exceptions: tuple[tuple[str, str], ...] = ()
    error: str | None = None
    names: tuple[str, ...] = ()


def _parse_license_expression(value: str) -> _LicenseExpression:
    """Check SPDX expression syntax and collect the names it references.

    Supports ``AND``, ``OR`` and ``WITH`` (matched case-insensitively) and
    parentheses. Adjacent words without an operator between them form one name,
    so free-text values such as ``MIT License`` keep the existing name matching.
    Misplaced operators and unbalanced parentheses are syntax errors when the
    value uses an SPDX operator (upper case, as the SPDX specification writes
    them); otherwise the whole value is one free-text name, such as
    ``MIT License (MIT)``, and goes to manual review like other unknown names.
    The names between operators and parentheses are kept in ``names`` either
    way, so a blocked license in ``GPL-3.0 (or later)`` still blocks the value.
    """
    items: list[tuple[str, str]] = []
    words: list[str] = []
    for token in _SPDX_EXPRESSION_TOKEN.findall(value):
        keyword = token.upper()
        if token in ("(", ")") or keyword in _SPDX_OPERATORS:
            if words:
                items.append(("name", " ".join(words)))
                words = []
            items.append((keyword, token))
        else:
            words.append(token)
    if words:
        items.append(("name", " ".join(words)))

    names = tuple(text for kind, text in items if kind == "name")
    expression = _parse_expression_items(items)
    if expression.error and not any(text in _SPDX_OPERATORS for _kind, text in items):
        whole = value.strip()
        return _LicenseExpression((whole,), names=(whole, *names))
    return replace(expression, names=names)


def _parse_expression_items(items: list[tuple[str, str]]) -> _LicenseExpression:
    """Run the SPDX expression grammar over grouped names, operators and parentheses."""
    licenses: list[str] = []
    exceptions: list[tuple[str, str]] = []
    depth = 0
    state = "start"
    for kind, text in items:
        if kind == "name" and state == "WITH":
            exceptions.append((licenses[-1], text))
            state = "exception"
        elif kind == "name" and state in _EXPRESSION_OPERAND_STATES:
            licenses.append(text)
            state = "license"
        elif kind == "name" or (kind == "(" and state not in _EXPRESSION_OPERAND_STATES):
            return _LicenseExpression(error=f"missing operator before '{text}'")
        elif kind == "(":
            depth += 1
            state = "("
        elif kind == "WITH" and state != "license":
            return _LicenseExpression(error="WITH must follow a single license identifier")
        elif kind == "WITH":
            state = "WITH"
        elif state not in _EXPRESSION_COMPLETE_STATES:
            return _LicenseExpression(error=f"missing license identifier before '{text}'")
        elif kind == ")" and depth == 0:
            return _LicenseExpression(error="unbalanced parenthesis")
        elif kind == ")":
            depth -= 1
            state = ")"
        else:
            state = kind
    if depth:
        return _LicenseExpression(error="unbalanced parenthesis")
    if items and state not in _EXPRESSION_COMPLETE_STATES:
        missing = "exception" if state == "WITH" else "license"
        return _LicenseExpression(error=f"missing {missing} identifier after '{items[-1][1]}'")
    return _LicenseExpression(tuple(licenses), tuple(exceptions))


# File reference indicators in license field values
_FILE_REFERENCE_KEYWORDS = frozenset(["see ", "refer to ", "license.txt", "license.md", "copying"])
_LICENSE_DECLARATION_STEMS = frozenset({"license", "licence", "copying"})
_TERMINAL_LICENSE_STATUSES = frozenset({"conflict", "unrecognized"})


@dataclass(slots=True)
class LicenseDetection:
    """Result of license detection attempt."""

    license_id: str | None
    source: str
    confidence: str
    file_path: str | None = None
    details: str | None = None


def _is_license_declaration_filename(name: str) -> bool:
    """Return True for LICENSE/COPYING-style names, not NOTICE files."""
    return Path(name).stem.lower() in _LICENSE_DECLARATION_STEMS


@dataclass(slots=True)
class _LicenseFileScan:
    """All detections and unidentified declaration files under an asset."""

    detections: list[LicenseDetection]
    unrecognized_declaration_files: list[str]


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
        - allowed_license_exceptions: SPDX exceptions allowed after WITH
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

    @cached_property
    def _normalized_exception_allowlist(self) -> frozenset[str]:
        """Pre-compute normalized SPDX exceptions that may follow an allowed license."""
        return frozenset(
            self._normalize_license_id(exception) for exception in self.config.get("allowed_license_exceptions", [])
        )

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

        if result.metadata.get("license_status") in _TERMINAL_LICENSE_STATUSES:
            return result
        if detection is None:
            self._handle_no_license(result)
        else:
            self._validate_license(detection, result)

        if result.passed:
            license_status = str(result.metadata.get("license_status") or "not_detected")
            result.add_success(
                check_name="license_compliance",
                message=f"License compliance scan completed ({license_status.replace('_', ' ')})",
                license_status=license_status,
            )

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
        scan = self._collect_license_files(asset_path)

        if detection := self._check_frontmatter(asset_path):
            result.add_message(f"Tier 1: Found license declaration in {detection.source}")
            if detection.license_id and self._is_file_reference(detection.license_id):
                result.add_message(f"  License field references file: '{detection.license_id}'")
                chosen = self._choose_from_license_files(scan, result)
                if chosen is not None or result.metadata.get("license_status") in _TERMINAL_LICENSE_STATUSES:
                    return chosen
            else:
                return self._resolve_frontmatter_and_files(detection, scan, result)

        chosen = self._choose_from_license_files(scan, result)
        if result.metadata.get("license_status") in _TERMINAL_LICENSE_STATUSES:
            return chosen
        if chosen is not None:
            result.add_message(f"Tier 2: Detected {chosen.license_id} from {chosen.file_path}")
            return chosen

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

    def _collect_license_files(self, asset_path: Path) -> _LicenseFileScan:
        """Read every conventional license filename and collect all matches."""
        detections: list[LicenseDetection] = []
        unrecognized: list[str] = []
        for license_name in LICENSE_FILE_NAMES:
            license_path = asset_path / license_name
            if not license_path.is_file() or not _is_license_declaration_filename(license_name):
                continue
            try:
                content = license_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logger.debug("Could not read %s: %s", license_path, exc)
                if _is_license_declaration_filename(license_name):
                    unrecognized.append(license_name)
                continue
            matches = self._identify_all_licenses_from_content(content)
            if matches:
                detections.extend(
                    LicenseDetection(
                        license_id=match["license_id"],
                        source="license_file",
                        confidence=match["confidence"],
                        file_path=license_name,
                    )
                    for match in matches
                )
                continue
            if indicator := self._find_proprietary_indicator(content):
                detections.append(
                    LicenseDetection(
                        license_id="Proprietary",
                        source="license_file",
                        confidence="high",
                        file_path=license_name,
                        details=f"Found proprietary indicator: '{indicator}'",
                    )
                )
                continue
            if _is_license_declaration_filename(license_name):
                unrecognized.append(license_name)
        return _LicenseFileScan(detections, unrecognized)

    def _identify_all_licenses_from_content(self, content: str) -> list[dict]:
        """Return every configured license pattern that matches the file text."""
        content_upper = content.upper()
        matches: list[dict] = []
        for license_id, pattern_config in self.config.get("license_patterns", {}).items():
            required = pattern_config.get("required", [])
            exclude = pattern_config.get("exclude", [])
            if not self._all_patterns_match(required, content, content_upper):
                continue
            if self._any_pattern_matches(exclude, content_upper):
                continue
            matches.append(
                {
                    "license_id": license_id,
                    "confidence": pattern_config.get("confidence", "medium"),
                }
            )
        return matches

    def _identify_license_from_content(self, content: str) -> dict | None:
        """Match LICENSE content against known patterns."""
        matches = self._identify_all_licenses_from_content(content)
        return matches[0] if matches else None

    def _license_bucket(self, license_id: str | None) -> str:
        """Classify a license value as allowed, blocked, unknown, or unsupported syntax."""
        if not license_id:
            return "unknown"
        return self._license_policy(_parse_license_expression(license_id))[0]

    def _license_policy(self, expression: _LicenseExpression) -> tuple[str, str | None]:
        """Return the policy status of a parsed expression and the name that decided it.

        Any blocked name blocks the expression, even one that is not valid SPDX
        syntax. Otherwise a syntax error makes it unsupported. It is allowed only
        when every license is allowed and every ``WITH`` exception is listed.
        """
        if blocked := next((name for name in expression.names if self._is_blocked_license(name)), None):
            return "blocked", blocked
        if expression.error:
            return "unsupported", None
        for name in expression.licenses:
            if self._normalize_license_id(name) not in self._normalized_allowlist:
                return "unknown", name
        for license_id, exception in expression.exceptions:
            if self._normalize_license_id(exception) not in self._normalized_exception_allowlist:
                return "unknown", f"{license_id} WITH {exception}"
        return "allowed", None

    def _is_blocked_license(self, license_id: str) -> bool:
        """Return True for a blocked id or one of its SPDX variants.

        Variants are the ``-only``/``-or-later``/``+`` forms and the deprecated
        ids with a built-in exception, such as ``GPL-2.0-with-classpath-exception``.
        """
        normalized = self._normalize_license_id(license_id)
        base, separator, exception = normalized.partition("-with-")
        if not (separator and exception.endswith("-exception")):
            base = normalized
        return any(
            candidate in self._normalized_blocklist
            for candidate in (normalized, base, _SPDX_VERSION_VARIANT_SUFFIX.sub("", base))
        )

    def _record_unrecognized_license_file(
        self,
        result: ValidationResult,
        filenames: list[str],
        frontmatter: LicenseDetection | None = None,
    ) -> None:
        """Fail closed when a LICENSE/COPYING file exists but cannot be identified."""
        joined = ", ".join(filenames)
        declared = f" Frontmatter declares '{frontmatter.license_id}'." if frontmatter and frontmatter.license_id else ""
        result.add_structured_finding(
            Finding(
                category="LICENSE",
                severity="HIGH",
                check_name="unrecognized_license_file",
                message=(
                    f"License file '{joined}' is present but could not be identified.{declared} "
                    "Manual review is required."
                ),
                file_path=filenames[0],
                suggestion=(
                    "Identify the license in the file, add a detection pattern, or remove the file."
                ),
            ),
            is_error=True,
        )
        result.metadata.update(
            {
                "license": frontmatter.license_id if frontmatter else None,
                "license_status": "unrecognized",
                "license_source": "license_file",
            }
        )

    def _choose_from_license_files(
        self,
        scan: _LicenseFileScan,
        result: ValidationResult,
    ) -> LicenseDetection | None:
        """Pick a file detection, failing closed on blocked or conflicting ids."""
        if scan.unrecognized_declaration_files:
            self._record_unrecognized_license_file(result, scan.unrecognized_declaration_files)
            return None
        if not scan.detections:
            return None
        blocked = [item for item in scan.detections if self._license_bucket(item.license_id) == "blocked"]
        if blocked:
            return blocked[0]
        unique: dict[str, LicenseDetection] = {}
        for item in scan.detections:
            if not item.license_id:
                continue
            unique.setdefault(self._normalize_license_id(item.license_id), item)
        if len(unique) == 1:
            return next(iter(unique.values()))
        first, second = list(unique.values())[:2]
        self._add_license_mismatch_finding(result, first, second, check_name="license_file_mismatch")
        return None

    def _resolve_frontmatter_and_files(
        self,
        frontmatter: LicenseDetection,
        scan: _LicenseFileScan,
        result: ValidationResult,
    ) -> LicenseDetection | None:
        """Reconcile a concrete frontmatter license with every license file."""
        if scan.unrecognized_declaration_files:
            self._record_unrecognized_license_file(result, scan.unrecognized_declaration_files, frontmatter)
            return None
        if not scan.detections:
            return frontmatter

        blocked_files = [item for item in scan.detections if self._license_bucket(item.license_id) == "blocked"]
        if blocked_files:
            chosen = blocked_files[0]
            result.add_message(
                f"Frontmatter license '{frontmatter.license_id}' differs from "
                f"{chosen.file_path} license '{chosen.license_id}'"
            )
            result.add_message(f"Tier 2: Detected {chosen.license_id} from {chosen.file_path}")
            return chosen
        if self._license_bucket(frontmatter.license_id) in {"blocked", "unsupported"}:
            return frontmatter

        file_by_id: dict[str, LicenseDetection] = {}
        for item in scan.detections:
            if item.license_id:
                file_by_id.setdefault(self._normalize_license_id(item.license_id), item)
        declared = {
            self._normalize_license_id(name)
            for name in _parse_license_expression(frontmatter.license_id or "").licenses
        }
        if set(file_by_id) <= declared:
            return frontmatter

        other = next((item for key, item in file_by_id.items() if key not in declared), None)
        if (
            other is not None
            and self._license_bucket(frontmatter.license_id) == "allowed"
            and self._license_bucket(other.license_id) == "allowed"
        ):
            result.add_message(
                f"Frontmatter license '{frontmatter.license_id}' differs from "
                f"{other.file_path} license '{other.license_id}'"
            )
            self._add_license_mismatch_finding(result, frontmatter, other)
            return None
        if other is not None and self._license_bucket(other.license_id) == "unknown":
            return other
        return frontmatter

    def _add_license_mismatch_finding(
        self,
        result: ValidationResult,
        first: LicenseDetection,
        second: LicenseDetection,
        check_name: str = "frontmatter_license_mismatch",
    ) -> None:
        """Fail closed when two allowed licenses disagree, without marking allowed."""
        result.add_structured_finding(
            Finding(
                category="LICENSE",
                severity="HIGH",
                check_name=check_name,
                message=(
                    f"License '{first.license_id}' conflicts with "
                    f"{second.file_path} license '{second.license_id}'"
                ),
                file_path=second.file_path or first.file_path or "unknown",
                suggestion=(
                    "Make every license declaration agree, or remove the conflicting file."
                ),
            ),
            is_error=True,
        )
        result.metadata.update(
            {
                "license": f"{first.license_id} vs {second.license_id}",
                "license_status": "conflict",
                "license_source": "frontmatter+license_file" if first.source.startswith("frontmatter") else "license_file",
            }
        )

    def _check_license_file(self, asset_path: Path) -> LicenseDetection | None:
        """Tier 2 helper: first blocked file detection, else the sole identified id."""
        scan = self._collect_license_files(asset_path)
        if scan.unrecognized_declaration_files or not scan.detections:
            return None
        blocked = [item for item in scan.detections if self._license_bucket(item.license_id) == "blocked"]
        if blocked:
            return blocked[0]
        unique: dict[str, LicenseDetection] = {}
        for item in scan.detections:
            if item.license_id:
                unique.setdefault(self._normalize_license_id(item.license_id), item)
        if len(unique) == 1:
            return next(iter(unique.values()))
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
                    line_start = header.rfind("\n", 0, match.start()) + 1
                    raw = _strip_spdx_capture(match.group(1), header[line_start : match.start()])
                    return raw or None
        except Exception as e:
            logger.debug("Could not scan %s: %s", file_path, e)
        return None

    def _validate_license(self, detection: LicenseDetection, result: ValidationResult) -> None:
        """Validate detected license against allowlist/blocklist."""
        license_id = detection.license_id
        if not license_id:
            result.add_warning("License detection returned empty identifier")
            return

        # Fail closed: one blocked name blocks the value, even a malformed one;
        # findings for a compound value name the license or exception that decided it.
        expression = _parse_license_expression(license_id)
        status, name = self._license_policy(expression)
        self._set_license_metadata(result, detection, status)
        if status == "allowed":
            result.add_message(f"License: {license_id} (ALLOWED - permissive)")
        elif status == "blocked":
            subject = name if name and len(expression.names) > 1 else license_id
            if subject != license_id:
                result.add_message(f"SPDX expression '{license_id}' includes blocked license '{subject}'")
            if expression.error:
                result.add_message(f"License expression '{license_id}' is not valid SPDX syntax: {expression.error}")
            self._add_blocked_license_finding(result, subject, detection.file_path)
        elif expression.error:
            self._add_unsupported_license_expression_finding(result, license_id, detection.file_path, expression.error)
        else:
            subject = name if name and (len(expression.licenses) > 1 or expression.exceptions) else license_id
            self._handle_unknown_license(result, subject, detection.file_path)

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

    @staticmethod
    def _add_unsupported_license_expression_finding(
        result: ValidationResult,
        license_id: str,
        file_path: str | None,
        reason: str,
    ) -> None:
        """Add structured finding for a license expression that does not parse."""
        result.add_structured_finding(
            Finding(
                category="LICENSE",
                severity="HIGH",
                check_name="unsupported_license_expression",
                message=f"License expression '{license_id}' is not valid SPDX syntax: {reason}",
                file_path=file_path or "unknown",
                suggestion="Use a valid SPDX license expression, such as 'MIT OR Apache-2.0'.",
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
        normalized = _LICENSE_SUFFIX_PATTERN.sub("", normalized)
        # SPDX "+" means "this version or any later version" (GPL-3.0+ is GPL-3.0-or-later).
        if normalized.endswith("+"):
            normalized = f"{normalized[:-1]}-or-later"
        return normalized
