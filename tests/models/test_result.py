# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.models.result.

Focuses on severity tallying, error/warning classification, and the
``recalculate_from_findings`` path used by the LLM false-positive downgrade
flow (validators/security.py, validators/policy.py).
"""

import pytest

from skillevaluator.models.result import (
    Finding,
    Severity,
    ValidationResult,
    ValidationSummary,
)


def _finding(severity: Severity, *, category: str = "TEST", check: str = "check") -> Finding:
    return Finding(
        category=category,
        severity=severity,
        check_name=check,
        message="something happened",
        file_path="skill/SKILL.md",
        line_number=3,
    )


class TestSeverity:
    """Tests for the Severity enum helpers."""

    @pytest.mark.parametrize(
        ("severity", "expected"),
        [
            (Severity.CRITICAL, True),
            (Severity.HIGH, True),
            (Severity.MEDIUM, False),
            (Severity.LOW, False),
            (Severity.INFO, False),
        ],
    )
    def test_is_error_default(self, severity: Severity, expected: bool):
        assert severity.is_error() is expected

    def test_is_error_fail_on_medium_and_low(self):
        assert Severity.MEDIUM.is_error(fail_on_medium=True) is True
        assert Severity.LOW.is_error(fail_on_low=True) is True
        assert Severity.LOW.is_error(fail_on_medium=True) is False


class TestFindingPostInit:
    """Finding normalizes string severities to the Severity enum."""

    def test_string_severity_is_coerced(self):
        finding = Finding(category="C", severity="high", check_name="c", message="m", file_path="f")
        assert finding.severity is Severity.HIGH

    def test_unknown_string_severity_falls_back_to_medium(self):
        finding = Finding(category="C", severity="bogus", check_name="c", message="m", file_path="f")
        assert finding.severity is Severity.MEDIUM


class TestValidationSummaryTally:
    """Direct tests for the tally / reset helpers."""

    def test_tally_increments_matching_counter(self):
        summary = ValidationSummary()
        summary.tally_severity(Severity.CRITICAL)
        summary.tally_severity(Severity.CRITICAL)
        summary.tally_severity(Severity.LOW)
        assert summary.critical_count == 2
        assert summary.low_count == 1
        assert summary.high_count == 0

    def test_info_and_non_enum_are_noops(self):
        summary = ValidationSummary()
        summary.tally_severity(Severity.INFO)
        summary.tally_severity("critical")  # raw string, not a Severity instance
        assert (summary.critical_count, summary.high_count, summary.medium_count, summary.low_count) == (0, 0, 0, 0)

    def test_reset_finding_counts_zeros_tracked_fields(self):
        summary = ValidationSummary(
            files_scanned=4,
            checks_performed=9,
            errors=2,
            warnings=1,
            critical_count=1,
            high_count=1,
            medium_count=3,
            low_count=2,
        )
        summary.reset_finding_counts()
        assert (summary.errors, summary.warnings) == (0, 0)
        assert (summary.critical_count, summary.high_count, summary.medium_count, summary.low_count) == (0, 0, 0, 0)
        # Untracked aggregate fields are preserved.
        assert summary.files_scanned == 4
        assert summary.checks_performed == 9


class TestAddFinding:
    """Tests for ValidationResult.add_finding (new + legacy calling styles)."""

    def test_high_finding_fails_and_counts(self):
        result = ValidationResult()
        result.add_finding(_finding(Severity.HIGH))
        assert result.passed is False
        assert result.summary.high_count == 1
        assert result.summary.errors == 1
        assert len(result.errors) == 1
        assert result.warnings == []

    def test_medium_finding_warns_by_default(self):
        result = ValidationResult()
        result.add_finding(_finding(Severity.MEDIUM))
        assert result.passed is True
        assert result.summary.medium_count == 1
        assert result.summary.warnings == 1
        assert len(result.warnings) == 1

    def test_medium_finding_fails_when_fail_on_medium(self):
        result = ValidationResult()
        result.add_finding(_finding(Severity.MEDIUM), fail_on_medium=True)
        assert result.passed is False
        assert result.summary.errors == 1

    def test_info_finding_is_not_severity_counted(self):
        result = ValidationResult()
        result.add_finding(_finding(Severity.INFO))
        assert result.passed is True
        assert (result.summary.critical_count, result.summary.high_count) == (0, 0)
        assert result.summary.medium_count == 0
        assert result.summary.low_count == 0
        assert result.summary.warnings == 1

    def test_legacy_positional_style(self):
        result = ValidationResult()
        result.add_finding("CVE", Severity.HIGH, "vulnerable dependency")
        assert result.passed is False
        assert result.errors == ["[CVE-HIGH] vulnerable dependency"]
        # Legacy string style does not populate structured findings.
        assert result.findings == []

    def test_legacy_style_requires_severity_and_message(self):
        result = ValidationResult()
        with pytest.raises(ValueError, match="Legacy add_finding"):
            result.add_finding("CVE")


class TestAddStructuredFinding:
    """Tests for add_structured_finding error/warning routing."""

    def test_error_marks_failed(self):
        result = ValidationResult()
        result.add_structured_finding(_finding(Severity.CRITICAL), is_error=True)
        assert result.passed is False
        assert result.summary.critical_count == 1
        assert len(result.errors) == 1

    def test_warning_does_not_fail(self):
        result = ValidationResult()
        result.add_structured_finding(_finding(Severity.LOW), is_error=False)
        assert result.passed is True
        assert result.summary.low_count == 1
        assert len(result.warnings) == 1


class TestRecalculateFromFindings:
    """Tests mirroring the LLM false-positive downgrade flow."""

    def test_downgrade_to_info_clears_error_state(self):
        result = ValidationResult()
        result.add_finding(_finding(Severity.HIGH))
        result.add_finding(_finding(Severity.CRITICAL))
        assert result.passed is False

        # Simulate the security validator downgrading both findings to INFO.
        for finding in result.findings:
            finding.severity = Severity.INFO
        result.recalculate_from_findings()

        assert result.passed is True
        assert result.errors == []
        assert result.summary.errors == 0
        assert (result.summary.critical_count, result.summary.high_count) == (0, 0)
        # Both findings now warn (INFO is not an error) but are not severity-counted.
        assert result.summary.warnings == 2

    def test_recalculate_is_idempotent(self):
        result = ValidationResult()
        result.add_finding(_finding(Severity.MEDIUM))
        result.add_finding(_finding(Severity.HIGH))

        before = result.summary.to_dict()
        result.recalculate_from_findings()
        after = result.summary.to_dict()

        # files_scanned / checks_performed are not rebuilt, so compare the rest.
        for key in ("errors", "warnings", "critical_count", "high_count", "medium_count", "low_count"):
            assert before[key] == after[key]


class TestMerge:
    """Tests for merge and merge_with_prefix aggregation."""

    def test_merge_combines_findings_and_counts(self):
        a = ValidationResult()
        a.add_finding(_finding(Severity.MEDIUM))
        b = ValidationResult()
        b.add_finding(_finding(Severity.HIGH))

        a.merge(b)
        assert a.passed is False
        assert len(a.findings) == 2
        assert a.summary.medium_count == 1
        assert a.summary.high_count == 1

    def test_merge_with_prefix_tags_messages(self):
        parent = ValidationResult()
        child = ValidationResult()
        child.add_finding(_finding(Severity.HIGH))

        parent.merge_with_prefix(child, "my-skill")
        assert parent.passed is False
        assert all("[my-skill]" in err for err in parent.errors)
        assert parent.findings[0].file_path.startswith("[my-skill] ")
