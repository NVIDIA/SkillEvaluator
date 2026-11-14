# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation result models for SkillEvaluator.

This module provides the core data structures for validation results:
- Severity: Standardized severity levels for findings
- Finding: Structured validation issue with full context
- SuccessDetail: Structured success information for passed checks
- ValidationSummary: Summary statistics for a validation run
- ValidationResult: Aggregated validation outcomes from a validator

These models are consumed by reporters (CLI, HTML, JSON, Markdown) to
render validation results in various formats.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from skillevaluator.skillspector_diagnostics import merge_completeness


class Severity(StrEnum):
    """Standardized severity levels for validation findings.

    Severity levels follow industry standards:
    - CRITICAL: Must fix immediately (security breach, blocked license)
    - HIGH: Should fix before merge (significant issues)
    - MEDIUM: Should fix soon (moderate risk)
    - LOW: Consider fixing (minor issues)
    - INFO: Informational only (no action required)
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def is_error(self, *, fail_on_medium: bool = False, fail_on_low: bool = False) -> bool:
        """Determine if this severity should cause validation failure.

        Args:
            fail_on_medium: Treat MEDIUM severity as error
            fail_on_low: Treat LOW severity as error

        Returns:
            True if this severity should fail validation
        """
        if self in (Severity.CRITICAL, Severity.HIGH):
            return True
        if self == Severity.MEDIUM and fail_on_medium:
            return True
        return bool(self == Severity.LOW and fail_on_low)

    @property
    def color(self) -> str:
        """Return Rich color for this severity level."""
        return {
            Severity.CRITICAL: "red bold",
            Severity.HIGH: "red",
            Severity.MEDIUM: "yellow",
            Severity.LOW: "dim",
            Severity.INFO: "blue",
        }.get(self, "white")

    @property
    def emoji(self) -> str:
        """Return text indicator for this severity level (emoji-free)."""
        return {
            Severity.CRITICAL: "[CRIT]",
            Severity.HIGH: "[HIGH]",
            Severity.MEDIUM: "[MED]",
            Severity.LOW: "[LOW]",
            Severity.INFO: "[INFO]",
        }.get(self, "[--]")


@dataclass
class Finding:
    """Structured validation issue with full context for CI reporting.

    Provides detailed information about each validation issue to help
    developers quickly understand and fix problems in CI pipelines.

    Attributes:
        category: Issue category (e.g., "SCHEMA", "SECRET", "CVE")
        severity: Severity level (accepts Severity enum or string for backward compat)
        check_name: Specific check identifier (e.g., "aws-access-key-id")
        message: Human-readable description of the issue
        file_path: Relative file path where issue was found
        line_number: Line number (if applicable)
        line_content: Actual content of the line (truncated if long)
        suggestion: How to fix the issue
        metadata: Additional context (e.g., CWE, CVSS score)
    """

    category: str
    severity: Severity | str  # Accept both for backward compatibility
    check_name: str
    message: str
    file_path: str
    line_number: int | None = None
    line_content: str | None = None
    suggestion: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert string severity to Severity enum if needed."""
        if isinstance(self.severity, str):
            # Convert string to Severity enum (case-insensitive)
            try:
                self.severity = Severity(self.severity.lower())
            except ValueError:
                # If not a valid enum value, map common strings
                severity_map = {
                    "critical": Severity.CRITICAL,
                    "high": Severity.HIGH,
                    "medium": Severity.MEDIUM,
                    "low": Severity.LOW,
                    "info": Severity.INFO,
                }
                self.severity = severity_map.get(self.severity.lower(), Severity.MEDIUM)

    @property
    def tag(self) -> str:
        """Return formatted tag like [SCHEMA-HIGH]."""
        if isinstance(self.severity, Severity):
            return f"[{self.category}-{self.severity.value.upper()}]"
        return f"[{self.category}-{str(self.severity).upper()}]"

    @staticmethod
    def _normalize_identity_text(value: object) -> str:
        """Normalize human text without coupling identity to formatting."""
        return " ".join(("" if value is None else str(value)).casefold().split())

    @staticmethod
    def _location_paths(file_path: object) -> list[str]:
        """Return ``file_path`` and the shorter paths a message may name it by.

        A folder run prefixes each skill's ``file_path`` with ``[skill] ``
        and a plugin scan prepends the member's directory, but neither
        rewrites the message, which still names the path the scanner saw.
        Longest first, so the most specific match is removed.
        """
        path = str(file_path)
        paths = [path]
        while path.startswith("[") and "] " in path:
            path = path.split("] ", 1)[1]
            paths.append(path)
        paths.extend(path[index + 1 :] for index, char in enumerate(path[:-1]) if char in "/\\")
        return paths

    def _identity_message(self) -> str:
        """Return the message without location text the finding records elsewhere.

        Secret detection writes the finding's own ``in <file>:<line>`` into
        its message (with the path as the scanner saw it, which may be a
        suffix of ``file_path``), and the PII scan appends
        ``— N occurrences (...)`` when it folds repeats of one value into a
        finding whose ``metadata`` lists them. Leaving that text out keeps ``finding_id`` stable when
        lines move. Other text, such as a file name a check puts in its
        message, stays part of the identity.
        """
        message = "" if self.message is None else str(self.message)
        if self.file_path is not None and self.line_number is not None:
            # Compare with "/" separators: a plugin scan writes file_path with
            # "/" while the message keeps the scanner's backslashes on Windows.
            # Each backslash maps to one "/", so a match spans the same text.
            slashed = message.replace("\\", "/")
            for location in self._location_paths(self.file_path):
                text = f" in {location}:{self.line_number}".replace("\\", "/")
                index = slashed.find(text)
                if index >= 0:
                    message = message[:index] + message[index + len(text) :]
                    break
        occurrences = self.metadata.get("occurrences") if isinstance(self.metadata, dict) else None
        if isinstance(occurrences, list) and len(occurrences) > 1:
            marker = f" \u2014 {len(occurrences)} occurrences ("
            index = message.rfind(marker)
            if index >= 0 and message.endswith(")"):
                message = message[:index]
        return message

    @property
    def finding_id(self) -> str:
        """Return a deterministic semantic identity independent of location.

        File and line belong to an occurrence. Excluding them, and any copy
        of them in the message (see ``_identity_message``), lets reporters
        group the same check across paths and validators while retaining every
        concrete occurrence in the reconciliation contract. Severity is also
        excluded, so a policy override or verifier downgrade keeps the ID.
        """
        identity = "\x1f".join(
            (
                self._normalize_identity_text(self.category),
                self._normalize_identity_text(self.check_name),
                self._normalize_identity_text(self._identity_message()),
                self._normalize_identity_text(self.suggestion),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8"), usedforsecurity=False).hexdigest()[:20]
        return f"SEF-{digest}"

    def location_occurrence_id(self, file_path: object, line_number: object, line_content: object = None) -> str:
        """Return the occurrence ID this finding has at another location.

        A check that folds repeats into one finding, such as the PII scan,
        lists the other locations in ``metadata["occurrences"]``; reporters
        give each of them an ID this way. At the finding's own location and
        line content it equals ``occurrence_id``.
        """
        identity = "\x1f".join(
            (
                self.finding_id,
                ("" if file_path is None else str(file_path)).replace("\\", "/"),
                str(line_number) if line_number is not None else "",
                self._normalize_identity_text(line_content),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8"), usedforsecurity=False).hexdigest()[:20]
        return f"SEO-{digest}"

    @property
    def occurrence_id(self) -> str:
        """Return a deterministic identity for this concrete location."""
        return self.location_occurrence_id(self.file_path, self.line_number, self.line_content)

    @property
    def location(self) -> str:
        """Return formatted location like 'file.py:42'."""
        loc = self.file_path
        if self.line_number:
            loc += f":{self.line_number}"
        return loc

    def to_legacy_string(self) -> str:
        """Convert to legacy error string format for backward compatibility."""
        return f"{self.tag} {self.message} in {self.location}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        severity_value = self.severity.value if isinstance(self.severity, Severity) else str(self.severity).lower()
        return {
            "finding_id": self.finding_id,
            "occurrence_id": self.occurrence_id,
            "category": self.category,
            "severity": severity_value,
            "check_name": self.check_name,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "suggestion": self.suggestion,
            "metadata": self.metadata,
        }


@dataclass
class SuccessDetail:
    """Structured success information for passed checks.

    Records details about successful validation checks for comprehensive
    reporting of what was verified.

    Attributes:
        check_name: Check identifier (e.g., "manifest_found")
        message: Human-readable description of what passed
        metadata: Additional context (e.g., file count, version)
    """

    check_name: str
    message: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "check": self.check_name,
            "message": self.message,
            "metadata": self.metadata or None,
        }


# Maps a finding severity to its ValidationSummary counter attribute. INFO
# (and any non-enum severity) is intentionally untracked -- it has no counter.
_SEVERITY_COUNT_ATTRS: dict[Severity, str] = {
    Severity.CRITICAL: "critical_count",
    Severity.HIGH: "high_count",
    Severity.MEDIUM: "medium_count",
    Severity.LOW: "low_count",
}


@dataclass
class ValidationSummary:
    """Summary statistics for a validation run.

    Provides aggregate counts for reporting summaries.
    """

    files_scanned: int = 0
    checks_performed: int = 0
    errors: int = 0
    warnings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    def tally_severity(self, severity: object) -> None:
        """Increment the per-severity counter for a finding.

        INFO and any non-:class:`Severity` value are a no-op, matching the
        historical behavior where only CRITICAL/HIGH/MEDIUM/LOW were counted.
        """
        if isinstance(severity, Severity):
            attr = _SEVERITY_COUNT_ATTRS.get(severity)
            if attr is not None:
                setattr(self, attr, getattr(self, attr) + 1)

    def reset_finding_counts(self) -> None:
        """Zero the error/warning totals and per-severity counters."""
        self.errors = 0
        self.warnings = 0
        self.critical_count = 0
        self.high_count = 0
        self.medium_count = 0
        self.low_count = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "files_scanned": self.files_scanned,
            "checks_performed": self.checks_performed,
            "errors": self.errors,
            "warnings": self.warnings,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
        }




def prefix_finding(finding: Finding, prefix: str, extra_metadata: dict | None = None) -> Finding:
    """Copy *finding* with *prefix* applied to every location it names.

    Used when a sub-result is folded into a parent result (a skill inside a
    collection, a rule inside a rules file) so a reader can still tell which
    member each finding came from. *extra_metadata* is merged on top, for a
    caller that can also name the member in a machine-readable field.
    """
    metadata = finding.metadata
    for key in ("occurrences", "skillspector_occurrences"):
        listed = metadata.get(key) if isinstance(metadata, dict) else None
        if not isinstance(listed, list):
            continue
        # A check that folds repeats (the PII scan) lists each location
        # in metadata, and a SkillSpector AE1 finding lists the lines
        # that name its file; prefix those like file_path so the
        # finding's own location still matches its first listed one.
        metadata = {
            **metadata,
            key: [
                {**item, "file": f"[{prefix}] {item['file']}"}
                if isinstance(item, dict) and isinstance(item.get("file"), str)
                else item
                for item in listed
            ],
        }
    if extra_metadata:
        metadata = {**metadata, **extra_metadata} if isinstance(metadata, dict) else dict(extra_metadata)
    return Finding(
        category=finding.category,
        severity=finding.severity,
        check_name=finding.check_name,
        message=finding.message,
        file_path=f"[{prefix}] {finding.file_path}",
        line_number=finding.line_number,
        line_content=finding.line_content,
        suggestion=finding.suggestion,
        metadata=metadata,
    )

@dataclass
class ValidationResult:
    """Aggregates validation outcomes from a validator.

    This is the primary data structure passed to reporters. It contains
    both structured findings and legacy string-based errors for backward
    compatibility.

    Attributes:
        validator_name: Name of the validator (e.g., "SCHEMA", "SECRETS")
        validator_description: Brief description of what the validator checks
        passed: Whether validation passed (no errors)
        findings: List of structured Finding objects
        success_details: List of SuccessDetail objects for passed checks
        summary: ValidationSummary with aggregate statistics
        errors: Legacy list of error strings (for backward compatibility)
        warnings: Legacy list of warning strings
        messages: Legacy list of informational messages
        metadata: Additional context dictionary
    """

    validator_name: str = ""
    validator_description: str = ""
    passed: bool = True
    findings: list[Finding] = field(default_factory=list)
    success_details: list[SuccessDetail] = field(default_factory=list)
    summary: ValidationSummary = field(default_factory=ValidationSummary)

    # Legacy fields for backward compatibility
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def incomplete_scans(self) -> list[str]:
        """External scanners that did not produce complete trustworthy evidence."""
        value = self.metadata.get("incomplete_scans")
        if not isinstance(value, list):
            return []
        return [name for name in value if isinstance(name, str) and name]

    @property
    def gate_failures(self) -> list[str]:
        """Failed requirements independent of finding severity, such as a minimum score."""
        value = self.metadata.get("gate_failures")
        if not isinstance(value, list):
            return []
        return [message for message in value if isinstance(message, str) and message]

    @property
    def is_incomplete(self) -> bool:
        """Whether required validation evidence is missing or unusable."""
        return bool(self.incomplete_scans)

    @property
    def status(self) -> str:
        """Canonical report status: ``passed``, ``failed``, or ``incomplete``."""
        if self.is_incomplete:
            return "incomplete"
        return "passed" if self.passed else "failed"

    def add_finding(
        self,
        finding_or_tag: Finding | str | None = None,
        severity: Severity | None = None,
        message: str | None = None,
        *,
        tag: str | None = None,
        fail_on_medium: bool = False,
        fail_on_low: bool = False,
    ) -> None:
        """Add a structured finding and update summary.

        Supports multiple calling conventions for backward compatibility:
        1. New style: add_finding(Finding(...))
        2. Legacy positional: add_finding(tag, severity, message)
        3. Legacy keyword: add_finding(tag="CVE", severity=..., message=...)

        Args:
            finding_or_tag: Either a Finding object (new style) or a tag string (legacy)
            severity: Severity level (only for legacy style)
            message: Finding description (only for legacy style)
            tag: Category tag as keyword argument (alternate for finding_or_tag)
            fail_on_medium: Treat MEDIUM severity as error
            fail_on_low: Treat LOW severity as error
        """
        # Handle keyword-only tag argument: add_finding(tag="CVE", severity=..., message=...)
        if finding_or_tag is None and tag is not None:
            finding_or_tag = tag

        # Handle legacy calling convention: add_finding(tag, severity, message)
        if isinstance(finding_or_tag, str):
            if severity is None or message is None:
                raise ValueError("Legacy add_finding requires tag, severity, and message arguments")
            # Legacy style - just format and add as error/warning string
            formatted = f"[{finding_or_tag}-{severity.value.upper()}] {message}"
            if severity.is_error(fail_on_medium=fail_on_medium, fail_on_low=fail_on_low):
                self.add_error(formatted)
            else:
                self.add_warning(formatted)
            return

        if finding_or_tag is None:
            raise ValueError("add_finding requires either a Finding object or tag arguments")

        # New style - finding is a Finding object
        finding = finding_or_tag
        self.findings.append(finding)

        self.summary.tally_severity(finding.severity)

        # Determine if this is an error or warning
        is_error = finding.severity.is_error(fail_on_medium=fail_on_medium, fail_on_low=fail_on_low)

        if is_error:
            self.passed = False
            self.summary.errors += 1
            # Add legacy error string
            self.errors.append(finding.to_legacy_string())
        else:
            self.summary.warnings += 1
            # Add legacy warning string
            self.warnings.append(finding.to_legacy_string())

    def add_success(
        self,
        check_name: str,
        message: str,
        **metadata: dict,
    ) -> None:
        """Add a success detail.

        Args:
            check_name: Identifier for the check
            message: Human-readable description
            **metadata: Additional context
        """
        self.success_details.append(SuccessDetail(check_name=check_name, message=message, metadata=metadata))
        self.summary.checks_performed += 1
        # Also add to legacy messages
        self.messages.append(f"[OK] {check_name}: {message}")

    def add_structured_finding(self, finding: Finding, *, is_error: bool = True) -> None:
        """Add a structured finding with full context for CI reporting.

        This method stores the Finding object for detailed failure analysis
        and also creates a legacy error/warning string for backward compatibility.

        Args:
            finding: Structured Finding object with full context
            is_error: If True, marks validation as failed; if False, adds as warning
        """
        self.findings.append(finding)

        self.summary.tally_severity(finding.severity)

        # Create legacy error string for backward compatibility
        location = f"{finding.file_path}"
        if finding.line_number:
            location += f":{finding.line_number}"

        # Handle both string and Severity enum for severity
        if isinstance(finding.severity, Severity):
            severity_str = finding.severity.value.upper()
        else:
            severity_str = str(finding.severity).upper()

        legacy_msg = f"[{finding.category}-{severity_str}] {finding.message} in {location}"

        if is_error:
            self.add_error(legacy_msg)
        else:
            self.add_warning(legacy_msg)

    def recalculate_from_findings(self) -> None:
        """Rebuild passed, errors, warnings, and summary counts from current findings.

        Call after mutating finding severities (e.g. LLM verification downgrade)
        so that result state is consistent with the actual finding objects.
        Independent gate failures remain errors regardless of those severities.
        HIGH and CRITICAL findings are errors, and so is a MEDIUM finding whose
        metadata sets ``fail_on_medium`` (a dependency audit with that option).
        """
        self.errors.clear()
        self.warnings.clear()
        self.passed = not self.is_incomplete
        self.summary.reset_finding_counts()

        for message in self.gate_failures:
            self.add_error(message)

        for finding in self.findings:
            sev = (
                finding.severity if isinstance(finding.severity, Severity) else Severity(str(finding.severity).lower())
            )
            self.summary.tally_severity(sev)

            fail_on_medium = isinstance(finding.metadata, dict) and finding.metadata.get("fail_on_medium") is True
            is_error = sev.is_error(fail_on_medium=fail_on_medium)
            legacy_msg = finding.to_legacy_string()
            if is_error:
                self.errors.append(legacy_msg)
                self.passed = False
                self.summary.errors += 1
            else:
                self.warnings.append(legacy_msg)
                self.summary.warnings += 1

    def _finding_legacy_lines(self) -> set[str]:
        """Return the legacy error and warning lines that only repeat a structured finding."""
        represented = {finding.to_legacy_string() for finding in self.findings}
        for finding in self.findings:
            # merge_with_prefix puts skill labels before legacy messages but inside
            # structured finding paths. Recognize both forms, including nested merges.
            location = finding.location
            prefix = ""
            while location.startswith("[") and "] " in location:
                label, _, location = location.partition("] ")
                prefix += label + "] "
                represented.add(f"{prefix}{finding.tag} {finding.message} in {location}")
        return represented

    def independent_errors(self) -> list[str]:
        """Return errors no structured finding repeats, such as execution diagnostics."""
        represented = self._finding_legacy_lines()
        return [error for error in self.errors if error not in represented]

    def independent_warnings(self) -> list[str]:
        """Return warnings no structured finding repeats, such as why a scanner did not complete."""
        represented = self._finding_legacy_lines()
        return [warning for warning in self.warnings if warning not in represented]

    def add_error(self, message: str) -> None:
        """Add a legacy error string and mark validation as failed.

        For backward compatibility with existing validators.
        """
        self.errors.append(message)
        self.passed = False
        self.summary.errors += 1

    def add_warning(self, message: str) -> None:
        """Add a legacy warning string.

        For backward compatibility with existing validators.
        """
        self.warnings.append(message)
        self.summary.warnings += 1

    def fail_gate(self, message: str) -> None:
        """Record a failed requirement that severity overrides cannot waive."""
        failures = self.gate_failures
        if message not in failures:
            failures.append(message)
            self.metadata["gate_failures"] = failures
            self.add_error(message)

    def add_message(self, message: str) -> None:
        """Add an informational message.

        For backward compatibility with existing validators.
        """
        self.messages.append(message)

    def mark_scan_incomplete(self, tool_name: str) -> None:
        """Record that an external scanner did not produce complete trustworthy results.

        Partial, missing, timed-out, crashed, and malformed scanner runs are
        gate failures. Trustworthy partial findings remain available, while
        reporters distinguish these runs from completed scans with policy
        findings by using the canonical ``incomplete`` status.
        """
        scans = self.metadata.setdefault("incomplete_scans", [])
        if tool_name not in scans:
            scans.append(tool_name)
        self.passed = False

    @property
    def scanners(self) -> dict[str, str]:
        """External scanners that produced this result, mapped to their versions."""
        value = self.metadata.get("scanners")
        return dict(value) if isinstance(value, dict) else {}

    def record_scanner(self, name: str, version: str | None) -> None:
        """Record that scanner *name* ran, so reports show which scanner versions produced the result."""
        self.metadata["scanners"] = {**self.scanners, name: version or "unknown"}

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one, combining all fields."""
        if not other.passed:
            self.passed = False

        self.findings.extend(other.findings)
        self.success_details.extend(other.success_details)
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.messages.extend(other.messages)
        # dict.update would let one sub-result's incomplete_scans clobber
        # another's (e.g. bandit and semgrep both failing); concatenate.
        incomplete = list(dict.fromkeys([*self.incomplete_scans, *other.incomplete_scans]))
        gate_failures = [*self.gate_failures, *other.gate_failures]
        scanners = {**self.scanners, **other.scanners}
        completeness = merge_completeness(
            self.metadata.get("skillspector_completeness"), other.metadata.get("skillspector_completeness")
        )
        self.metadata.update(other.metadata)
        if completeness:
            self.metadata["skillspector_completeness"] = completeness
        if incomplete:
            self.metadata["incomplete_scans"] = incomplete
        if scanners:
            self.metadata["scanners"] = scanners
        if gate_failures:
            self.metadata["gate_failures"] = gate_failures

        # Merge summary
        self.summary.files_scanned += other.summary.files_scanned
        self.summary.checks_performed += other.summary.checks_performed
        self.summary.errors += other.summary.errors
        self.summary.warnings += other.summary.warnings
        self.summary.critical_count += other.summary.critical_count
        self.summary.high_count += other.summary.high_count
        self.summary.medium_count += other.summary.medium_count
        self.summary.low_count += other.summary.low_count

    def merge_with_prefix(self, other: ValidationResult, prefix: str) -> None:
        """Merge another result, prefixing all errors/warnings with skill name."""
        completeness = merge_completeness(
            self.metadata.get("skillspector_completeness"),
            other.metadata.get("skillspector_completeness"),
            prefix=prefix,
        )
        if completeness:
            self.metadata["skillspector_completeness"] = completeness
        if not other.passed:
            self.passed = False

        for error in other.errors:
            self.errors.append(f"[{prefix}] {error}")
        for warning in other.warnings:
            self.warnings.append(f"[{prefix}] {warning}")

        # Prefix findings with skill name in file_path
        for finding in other.findings:
            self.findings.append(prefix_finding(finding, prefix))

        # Carry over success_details (prefixed) so contributor info is preserved
        for detail in other.success_details:
            prefixed_detail = SuccessDetail(
                check_name=f"[{prefix}] {detail.check_name}",
                message=detail.message,
                metadata=detail.metadata,
            )
            self.success_details.append(prefixed_detail)

        incomplete = list(dict.fromkeys([*self.incomplete_scans, *other.incomplete_scans]))
        if incomplete:
            self.metadata["incomplete_scans"] = incomplete
        gate_failures = [*self.gate_failures, *(f"[{prefix}] {message}" for message in other.gate_failures)]
        if gate_failures:
            self.metadata["gate_failures"] = gate_failures
        if scanners := {**self.scanners, **other.scanners}:
            self.metadata["scanners"] = scanners
        # Keep each skill's unparsed files, prefixed like errors and findings so
        # skills stay distinguishable when a folder holds several.
        unparsed = other.metadata.get("semgrep_unparsed_files")
        if isinstance(unparsed, list) and unparsed:
            carried = self.metadata.get("semgrep_unparsed_files")
            self.metadata["semgrep_unparsed_files"] = [
                *(carried if isinstance(carried, list) else []),
                *(f"[{prefix}] {path}" for path in unparsed if isinstance(path, str)),
            ]
        omitted = other.metadata.get("semgrep_unparsed_files_omitted")
        if type(omitted) is int and omitted > 0:
            carried_omitted = self.metadata.get("semgrep_unparsed_files_omitted")
            self.metadata["semgrep_unparsed_files_omitted"] = omitted + (
                carried_omitted if type(carried_omitted) is int else 0
            )

        # Merge summary
        self.summary.errors += other.summary.errors
        self.summary.warnings += other.summary.warnings

    def merge_advisories_with_prefix(self, other: ValidationResult, prefix: str) -> None:
        """Carry a *passing* sub-result's nonblocking findings into this one.

        ``merge_with_prefix`` is for a sub-result that failed. A member can
        pass and still report advisories (INFO/LOW/MEDIUM findings such as
        ``author_without_email`` or ``metadata_id_format``), and those are
        exactly what a collection run is expected to surface.

        This never touches ``passed``: an advisory from a passing member stays
        advisory and cannot fail the collection. The member is named both in
        the finding's location and in ``metadata.collection_skill``, so a
        consumer does not have to parse the ``[prefix]`` location to tell
        where the finding came from.
        """
        for finding in other.findings:
            self.findings.append(prefix_finding(finding, prefix, {"collection_skill": prefix}))
            self.summary.tally_severity(finding.severity)
        for warning in other.warnings:
            self.warnings.append(f"[{prefix}] {warning}")
        self.summary.warnings += other.summary.warnings

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "validator": self.validator_name,
            "description": self.validator_description,
            "passed": self.passed,
            "status": self.status,
            "incomplete_scans": self.incomplete_scans,
            "summary": self.summary.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "success_details": [s.to_dict() for s in self.success_details],
        }
        if self.gate_failures:
            result["gate_failures"] = self.gate_failures
        if self.scanners:
            result["scanners"] = self.scanners
        return result
