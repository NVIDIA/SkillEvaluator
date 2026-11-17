# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordering, multiplicity and summary contracts for diagnostic reconciliation."""

from copy import deepcopy

import pytest

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.validators.policy import ValidationPolicy, apply_policy
from skillevaluator.validators.security import SecurityValidator


def _finding(severity):
    return Finding("SECURITY", severity, "example", "Check this source", file_path="SKILL.md")


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("keep_errors", [False, True])
@pytest.mark.parametrize("repeat_warnings", [False, True])
def test_reconciliation_preserves_order_and_existing_multiplicity(incomplete, keep_errors, repeat_warnings):
    result = ValidationResult()
    high = _finding(Severity.HIGH)
    medium = _finding(Severity.MEDIUM)
    result.add_finding(high)
    result.add_finding(medium)
    result.add_finding(deepcopy(medium))
    result.fail_gate("independent gate failure")
    if incomplete:
        result.mark_scan_incomplete("scanner")
    errors = [high.to_legacy_string(), "independent gate failure", "error b", "error a", "error b"]
    warnings = [medium.to_legacy_string(), "warning b", "warning a", "warning b", medium.to_legacy_string()]
    original_errors, original_warnings = errors.copy(), warnings.copy()

    result.recalculate_keeping_independent(
        errors, warnings, keep_all_errors=keep_errors, keep_repeated_warnings=repeat_warnings
    )

    expected_errors = ["independent gate failure", high.to_legacy_string()]
    if keep_errors or incomplete:
        expected_errors += ["error b", "error a"]
    expected_warnings = [medium.to_legacy_string()] * 2
    expected_warnings += warnings if repeat_warnings else ["warning b", "warning a"]
    assert result.errors == expected_errors
    assert result.warnings == expected_warnings
    assert result.summary.errors == len(expected_errors)
    assert result.summary.warnings == len(expected_warnings)
    assert (result.summary.high_count, result.summary.medium_count) == (1, 2)
    assert not result.passed
    assert errors == original_errors
    assert warnings == original_warnings


def test_skillspector_dedup_preserves_independent_diagnostic_order():
    result = ValidationResult()
    finding = _finding(Severity.HIGH)
    result.add_finding(finding)
    result.add_finding(deepcopy(finding))
    result.fail_gate("scanner gate")
    for message in ["error b", "error a", "error b"]:
        result.add_error(message)
    for message in ["warning b", "warning a", "warning b"]:
        result.add_warning(message)

    SecurityValidator._deduplicate_skillspector_findings(result)

    assert result.findings == [finding]
    assert result.errors == ["scanner gate", finding.to_legacy_string(), "error b", "error a"]
    assert result.warnings == ["warning b", "warning a"]
    assert result.summary.errors == 4
    assert result.summary.warnings == 2
    assert result.summary.high_count == 1


def test_policy_override_keeps_repeated_warnings_and_incomplete_errors():
    result = ValidationResult()
    finding = _finding(Severity.HIGH)
    result.add_finding(finding)
    result.mark_scan_incomplete("scanner")
    for message in ["error b", "error a", "error b"]:
        result.add_error(message)
    warnings = ["warning b", "warning a", "warning b"]
    for message in warnings:
        result.add_warning(message)

    apply_policy([result], ValidationPolicy(severity_overrides={"SECURITY.example": Severity.INFO}))

    assert result.findings[0].severity == Severity.INFO
    assert result.errors == ["error b", "error a"]
    assert result.warnings == [finding.to_legacy_string(), *warnings]
    assert result.summary.errors == 2
    assert result.summary.warnings == 4
    assert result.summary.high_count == 0
    assert result.is_incomplete
    assert not result.passed
