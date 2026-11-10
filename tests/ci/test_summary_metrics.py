# SPDX-License-Identifier: Apache-2.0
"""Compact comment metrics preserve findings and uncertainty from native results."""

import json

import pytest
from ci.summary_metrics import summarize_results

from skillevaluator.models import Finding, ValidationResult
from skillevaluator.validators.license import LicenseValidator


def _finding(severity, message):
    return Finding(category="TEST", severity=severity, check_name="check", message=message, file_path="SKILL.md")


def test_counts_findings_and_independent_diagnostics_without_success_info_or_mirrors():
    result = ValidationResult(validator_name="Schema")
    result.add_success("good", "Valid metadata")
    for severity, message in (
        ("low", "Small issue"),
        ("info", "Nothing needs changing"),
        ("high", "Risky operation"),
        ("critical", "Unsafe secret handling"),
        ("medium", "Missing guidance"),
    ):
        result.add_finding(_finding(severity, message))
    result.summary.high_count = 900
    result.add_warning("Independent diagnostic counts once without being copied")

    assert summarize_results([result], tier=1, native={"metrics": {"issue_count": 99}}) == {
        "issue_count": 5,
        "issue_count_known": True,
        "critical_count": 1,
        "high_count": 1,
        "top_issue": "Unsafe secret handling",
    }


@pytest.mark.parametrize("incomplete", [False, True])
def test_clean_and_partial_scans_have_distinct_count_certainty(incomplete):
    result = ValidationResult(validator_name="Security")
    if incomplete:
        result.mark_scan_incomplete("scanner")

    summary = summarize_results([result], tier=1)

    assert summary["issue_count"] == 0
    assert summary["issue_count_known"] is not incomplete
    assert summary["top_issue"] is None


def test_partial_scan_with_confirmed_findings_keeps_observed_count_unknown_total():
    result = ValidationResult(validator_name="Security")
    result.mark_scan_incomplete("scanner")
    result.add_finding(_finding("critical", "Confirmed issue"))

    summary = summarize_results([result], tier=1)

    assert summary["issue_count"] == 1
    assert summary["critical_count"] == 1
    assert summary["issue_count_known"] is False


@pytest.mark.parametrize("results", [[], [ValidationResult(validator_name="QUALITY", metadata={"skipped": True})]])
def test_absent_and_skipped_scans_do_not_claim_zero_known_issues(results):
    summary = summarize_results(results, tier=2)

    assert summary["issue_count"] == 0
    assert summary["issue_count_known"] is False


def test_top_issue_is_bounded_single_line_plain_text_from_highest_severity_only():
    result = ValidationResult(validator_name="Security")
    result.add_finding(_finding("low", "Earlier low finding"))
    result.add_finding(
        _finding("high", "First high\n\x00\u202e<script> [link](javascript:x) | @team `code` " + "x" * 300)
    )
    result.add_finding(_finding("high", "Second high finding"))

    summary = summarize_results([result], tier=1)
    text = summary["top_issue"]

    assert text.startswith("First high")
    assert len(text) <= 160
    assert text.endswith("…")
    assert not any(character in text for character in "\n\x00\u202e<>[]|@`")
    assert "Earlier low finding" not in text
    assert "Second high finding" not in text
    assert json.loads(json.dumps(summary)) == summary


def test_info_only_scan_has_no_top_issue():
    result = ValidationResult(validator_name="Schema", findings=[_finding("info", "Verified safe behavior")])

    assert summarize_results([result], tier=1)["top_issue"] is None


def test_repeated_finding_across_locations_and_validators_counts_one_highest_severity_issue():
    first = _finding("medium", "Repeated problem")
    second = _finding("high", "Repeated problem")
    second.file_path = "other/SKILL.md"
    results = [
        ValidationResult(validator_name="One", findings=[first]),
        ValidationResult(validator_name="Two", findings=[second, second]),
    ]

    summary = summarize_results(results, tier=1)

    assert summary["issue_count"] == 1
    assert summary["high_count"] == 1
    assert summary["top_issue"] == "Repeated problem"


def test_gate_failure_without_findings_keeps_safe_gate_reason_in_key_information():
    result = ValidationResult(validator_name="QUALITY")
    result.fail_gate("Score 40 is below the required minimum 70\n<details> | @team")
    result.errors.append("Provider diagnostic must not be copied")

    summary = summarize_results([result], tier=2)

    assert summary["issue_count"] == 2
    assert summary["issue_count_known"] is True
    assert summary["top_issue"] == "Score 40 is below the required minimum 70 details (at)team"


def test_gate_reason_does_not_hide_a_structured_finding():
    result = ValidationResult(validator_name="QUALITY")
    result.fail_gate("Score below minimum")
    result.add_finding(_finding("high", "Fix this concrete problem"))

    assert summarize_results([result], tier=2)["top_issue"] == "Fix this concrete problem"


@pytest.mark.parametrize("strict", [False, True])
def test_missing_license_legacy_error_or_warning_counts_as_an_issue(tmp_path, strict):
    (tmp_path / "SKILL.md").write_text("---\nname: example\ndescription: An example skill\n---\n")
    result = LicenseValidator(strict_mode=strict).validate(tmp_path)

    summary = summarize_results([result], tier=1)

    assert summary["issue_count"] == 1
    assert summary["issue_count_known"] is True
    assert summary["critical_count"] == 0
    assert summary["high_count"] == 0
    assert summary["top_issue"] == "Additional validation errors/warnings; see report"


def test_workflow_scope_warning_does_not_count_as_an_issue():
    result = ValidationResult(validator_name="Workflow")
    result.add_warning("Workflow scope: semantic validation was disabled")

    summary = summarize_results([result], tier=1)

    assert summary["issue_count"] == 0
    assert summary["top_issue"] is None


def test_legacy_finding_mirrors_do_not_increase_issue_count():
    result = ValidationResult(validator_name="Schema")
    result.add_finding(_finding("high", "Fix a serious problem"))
    result.add_finding(_finding("low", "Fix a minor problem"))
    result.add_finding(_finding("info", "Informational finding"))

    assert summarize_results([result], tier=1)["issue_count"] == 2


def test_independent_provider_diagnostic_is_counted_without_copying_raw_text():
    result = ValidationResult(validator_name="QUALITY")
    result.add_error("Provider diagnostic contains private source content and a credential")
    result.mark_scan_incomplete("provider")

    summary = summarize_results([result], tier=2)

    assert summary["issue_count"] == 1
    assert summary["issue_count_known"] is False
    assert summary["top_issue"] == "Additional validation errors/warnings; see report"
    assert "private source content" not in json.dumps(summary)


def test_tier3_uses_validated_result_metadata_for_scores_and_coverage():
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        metadata={
            "agent_eval": {
                "execution_status": "succeeded",
                "verdict": "pass",
                "summary": {
                    "overall_score": 0.8,
                    "overall_lift": 0.25,
                    "scored_attempts": 4,
                    "expected_attempts": 4,
                },
            }
        },
    )

    summary = summarize_results([result], tier=3, native={"tier3": {"overall_score": 1.0}})

    assert summary["score"] == 0.8
    assert summary["skill_lift"] == 0.25
    assert summary["coverage"] == "4/4"
    assert summary["issue_count_known"] is True


def test_tier3_missing_execution_does_not_invent_scores_or_zero_coverage():
    result = ValidationResult(validator_name="AGENT_EVAL")

    summary = summarize_results([result], tier=3)

    assert summary["issue_count_known"] is False
    assert "score" not in summary
    assert "skill_lift" not in summary
    assert "coverage" not in summary


def test_tier3_partial_execution_retains_coverage_without_claiming_clean_counts_or_scores():
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        metadata={
            "agent_eval": {
                "execution_status": "failed",
                "summary": {
                    "overall_score": 0.9,
                    "overall_lift": 0.2,
                    "scored_attempts": 1,
                    "expected_attempts": 4,
                },
            }
        },
    )

    summary = summarize_results([result], tier=3)

    assert summary["issue_count_known"] is False
    assert summary["coverage"] == "1/4"
    assert "score" not in summary
    assert "skill_lift" not in summary
