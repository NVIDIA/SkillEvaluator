# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from skillevaluator.tier3.eval_core.llm_judge import _ACCURACY_DEDUCTION_SENTENCES
from skillevaluator.tier3.harbor import report


@pytest.mark.parametrize(("criterion", "sentence"), _ACCURACY_DEDUCTION_SENTENCES.items())
def test_accuracy_criteria_have_plain_language_reasons(criterion, sentence):
    reasons = report._collect_fail_reasons("accuracy", [{"detail": {"criteria": {criterion: False}}}])

    assert reasons == [f"{sentence} (criterion {criterion})"]


def test_unknown_accuracy_criterion_has_safe_fallback():
    assert report._collect_fail_reasons("accuracy", [{"detail": {"criteria": {"NEW_CHECK": False}}}]) == [
        "Criterion NEW_CHECK was not met"
    ]


def test_excluded_accuracy_criterion_does_not_create_a_deduction():
    detail = {"criteria": {"ACTION_CORRECT": False}, "excluded_criteria": ["ACTION_CORRECT"]}

    assert report._collect_fail_reasons("accuracy", [{"detail": detail}]) == []


@pytest.mark.parametrize("metric", ["skill_execution", "skill_efficiency"])
def test_skill_check_without_reason_uses_plain_fallback(metric):
    detail = {"error_recovery_check": {"passed": False}}

    assert report._collect_fail_reasons(metric, [{"detail": detail}]) == [
        "Check `error recovery check` did not pass; no reason was recorded"
    ]


def test_expected_and_observed_are_appended_once_and_bounded():
    detail = {
        "criteria": {"ACTION_CORRECT": False, "FACTUALLY_ACCURATE": False},
        "ground_truth": "e" * 200,
        "final_answer": "o" * 200,
    }

    reasons = report._collect_fail_reasons("accuracy", [{"detail": detail}])

    assert sum(reason.count("Expected:") for reason in reasons) == 1
    assert len(reasons[-1].split("Expected: ", 1)[1].split("; observed: ", 1)[0]) == 160
    assert len(reasons[-1].split("; observed: ", 1)[1]) == 160


def test_expected_and_observed_are_omitted_when_not_recorded():
    reasons = report._collect_fail_reasons("goal_accuracy", [{"detail": {"reason": "goal missed"}}])

    assert reasons == ["goal missed"]


def _goal_reward(detail, score=0.2):
    return {
        "entry_id": "case-1",
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
        "goal_accuracy": score,
        "behavior_check": 1.0,
        "details": detail,
    }


def test_failed_finding_and_json_include_evidence_unavailable_reason():
    reward = _goal_reward({"goal_accuracy": {"reason": "goal missed"}})
    [finding] = report._extract_findings([reward], canonical_scores={"goal_accuracy": 0.2})
    payload = report._bounded_findings_payload(
        skill_name="demo",
        agent="agent",
        findings=[finding],
        suggestions=[],
        suggestion_mode="remediation",
        suggestions_v2=None,
    )

    assert finding["evidence_unavailable_reason"] == "No trajectory evidence was recorded for this metric"
    assert payload["findings"][0]["evidence_unavailable_reason"] == finding["evidence_unavailable_reason"]
    assert (
        "evidence: unavailable — No trajectory evidence was recorded for this metric"
        in report._render_findings_body([finding]).plain
    )


def test_trajectory_load_failure_replaces_default_evidence_reason():
    reward = _goal_reward({"goal_accuracy": {"reason": "goal missed"}})
    reward["trajectory_detail"] = "trajectory could not be read"
    [finding] = report._extract_findings([reward], canonical_scores={"goal_accuracy": 0.2})

    assert (
        finding["evidence_unavailable_reason"]
        == "Trajectory evidence could not be loaded: trajectory could not be read"
    )


def test_trajectory_load_metadata_is_not_reported_as_a_load_failure():
    reward = _goal_reward(
        {
            "_trajectory_load": {
                "source": "claude-code.txt",
                "note": "Synthetic ATIF from Claude Code stream JSONL",
                "warning": "trajectory.json has no steps[]",
            },
            "goal_accuracy": {"reason": "goal missed"},
        }
    )
    [finding] = report._extract_findings([reward], canonical_scores={"goal_accuracy": 0.2})

    assert finding["evidence_unavailable_reason"] == "No trajectory evidence was recorded for this metric"


def test_evidence_unavailable_labels_are_limited_to_evidence_ref_metrics():
    reward = _goal_reward(
        {
            "security": {"reason": "unsafe action"},
            "goal_accuracy": {"reason": "goal missed"},
        }
    )
    reward["security"] = 0.0
    findings = report._extract_findings(
        [reward],
        canonical_scores={"security": 0.0, "goal_accuracy": 0.2},
    )
    by_metric = {finding["metric"]: finding for finding in findings}

    assert "evidence_unavailable_reason" not in by_metric["security"]
    assert by_metric["goal_accuracy"]["evidence_unavailable_reason"] == (
        "No trajectory evidence was recorded for this metric"
    )
    security_payload = report._bounded_findings_payload(
        skill_name="demo",
        agent="agent",
        findings=[by_metric["security"]],
        suggestions=[],
        suggestion_mode="remediation",
        suggestions_v2=None,
    )
    assert "evidence_unavailable_reason" not in security_payload["findings"][0]


def test_evidence_reason_is_absent_when_refs_are_present_or_finding_is_ok():
    failed = _goal_reward(
        {
            "goal_accuracy": {
                "reason": "goal missed",
                "evidence_refs": [{"source": "trajectory.json", "json_pointer": "/steps/1"}],
            }
        }
    )
    [finding] = report._extract_findings([failed], canonical_scores={"goal_accuracy": 0.2})
    assert "evidence_unavailable_reason" not in finding

    passed = _goal_reward({"goal_accuracy": {"reason": "goal reached"}}, score=1.0)
    [finding] = report._extract_findings([passed], canonical_scores={"goal_accuracy": 1.0})
    assert finding["severity"] == "ok"
    assert "evidence_unavailable_reason" not in finding
