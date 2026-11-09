# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import pytest
from rich.console import Console

from skillevaluator.evaluation.tier3_report import _build_agent, build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter
from skillevaluator.tier3.evidence_volume import (
    evidence_volume,
    evidence_volume_for_conditions,
    format_evidence_volume,
)
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, DEFAULT_SCORE_POLICY
from skillevaluator.tier3.result_display import render_result


def _condition(*, cases: int, planned: int, scored: int, judge_failures: int = 0) -> dict:
    condition = {
        "execution_status": "succeeded" if scored == planned else "failed",
        "expected_cases": cases,
        "scored_cases": cases,
        "expected_attempts": planned,
        "scored_attempts": scored,
        "execution_errors": [],
    }
    if judge_failures:
        condition["judge_failures"] = judge_failures
    return condition


def _volume(status: str) -> dict:
    if status == "sufficient":
        return evidence_volume(
            cases=10,
            attempts_planned=20,
            attempts_scored=20,
            execution_failures=0,
            judge_failures=0,
            by_arm={
                "with_skill": {
                    "cases": 10,
                    "attempts_planned": 20,
                    "attempts_scored": 20,
                    "execution_failures": 0,
                    "judge_failures": 0,
                },
                "without_skill": {
                    "cases": 10,
                    "attempts_planned": 20,
                    "attempts_scored": 20,
                    "execution_failures": 0,
                    "judge_failures": 0,
                },
            },
        )
    return evidence_volume(
        cases=3,
        attempts_planned=30,
        attempts_scored=27,
        execution_failures=2,
        judge_failures=1,
        by_arm={
            "with_skill": {
                "cases": 3,
                "attempts_planned": 30,
                "attempts_scored": 27,
                "execution_failures": 2,
                "judge_failures": 1,
            }
        },
    )


def _agent_eval(volume: dict) -> dict:
    conditions = {
        "with_skill": _condition(
            cases=volume["cases"],
            planned=volume["attempts_planned"],
            scored=volume["attempts_scored"],
            judge_failures=volume["judge_failures"],
        ),
        "without_skill": _condition(cases=10, planned=20, scored=20),
    }
    return {
        "execution_status": "succeeded",
        "verdict": "pass",
        "best_agent": "test-agent",
        "skill_name": "test-skill",
        "overall_score": 0.8,
        "overall_lift": 0.1,
        "agents_run": ["test-agent"],
        "agents": {
            "test-agent": {
                "execution_status": "succeeded",
                "with_skill": 0.8,
                "baseline": 0.7,
                "lift": 0.1,
                "expected_attempts": 40,
                "scored_attempts": 40,
                "metrics_with_skill": ["security"],
                "metrics_without_skill": ["security"],
                "conditions": {
                    "with_skill": _condition(cases=10, planned=20, scored=20),
                    "without_skill": _condition(cases=10, planned=20, scored=20),
                },
                "evidence_volume": volume,
                "dimensions": [],
                "evaluators": {},
                "pass_at_k": {},
                "trials": [],
                "cases": [],
            }
        },
        "conditions": conditions,
        "evidence_volume": volume,
        "dataset_summary": {"total_tasks": 10, "negative_tasks": 1},
        "dataset_sufficiency": {
            "status": "sufficient" if not volume["limited"] else "provisional",
            "coverage_summary": "Dataset size: test",
        },
        "attempt_policy": {"max_attempts": 2, "stop_on_pass": False},
        "verdict_policy": {"attempt_pass_threshold": 0.5},
        "evaluators": {},
        "dimensions": [],
        "evaluator_cards": [],
        "cases": [],
        "trials": [],
        "pass_at_k": {},
        "execution_errors": [],
    }


def _provisional_agent_eval() -> dict:
    conditions = {
        "with_skill": _condition(cases=3, planned=30, scored=27, judge_failures=1),
        "without_skill": _condition(cases=3, planned=30, scored=30),
    }
    for condition in conditions.values():
        condition["execution_status"] = "succeeded"
    payload = build_agent_eval_payload(
        "test-skill",
        {
            "test-agent": {
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.7),
                "conditions": conditions,
                "execution_status": "succeeded",
                "expected_attempts": 60,
                "scored_attempts": 57,
                "rewards": [],
                "rewards_baseline": [],
                "rewards_complete": True,
                "rewards_baseline_complete": True,
            }
        },
        attempt_policy={"max_attempts": 10, "stop_on_pass": False},
        persisted_dataset_summary={"total_tasks": 3, "negative_tasks": 1, "unclassified_tasks": 0},
    )
    assert payload is not None
    return payload


@pytest.mark.parametrize("status", ("provisional", "sufficient"))
def test_evidence_volume_is_rendered_on_all_report_surfaces(status):
    ae = _provisional_agent_eval() if status == "provisional" else _agent_eval(_volume(status))
    volume = ae["evidence_volume"]
    expected = format_evidence_volume(volume)
    assert expected.startswith("Limited evidence:") == (status == "provisional")

    cli = StringIO()
    CLIReporter._print_agent_eval_tables(ae, Console(file=cli, force_terminal=False))
    assert expected in " ".join(cli.getvalue().split())

    evaluate_output = render_result(ae)
    assert expected in " ".join(evaluate_output.split())
    assert "test-agent: " not in evaluate_output

    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = ae
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert expected in html
    assert ("t3-evidence-volume warning" in html) == (status == "provisional")

    benchmark = BenchmarkReporter(include_timestamp=False, report_scope="tier3").render_all([result])
    assert f"Evidence volume: {expected}." in benchmark


def test_default_benchmark_publication_scope_renders_evidence_volume_once():
    ae = _agent_eval(_volume("sufficient"))
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = ae
    output = BenchmarkReporter(include_timestamp=False).render_all([result])
    expected = f"- Evidence volume: {format_evidence_volume(ae['evidence_volume'])}."
    assert expected in output
    assert output.count(expected) == 1


def test_evidence_volume_case_fallbacks_and_unknown_case_counts():
    from_summary = evidence_volume_for_conditions(
        {"with_skill": {"expected_attempts": 2, "scored_attempts": 2}},
        dataset_summary={"total_tasks": 4},
    )
    assert from_summary["cases"] == 4
    assert from_summary["by_arm"]["with_skill"]["cases"] == 4

    from_rewards = evidence_volume_for_conditions(
        {"with_skill": {"expected_attempts": 2, "scored_attempts": 2}},
        rewards_by_arm={"with_skill": [{"entry_id": "case-1"}, {"entry_id": "case-2"}]},
    )
    assert from_rewards["cases"] == 2

    unknown = evidence_volume_for_conditions({"with_skill": {"expected_attempts": 2, "scored_attempts": 2}})
    assert unknown["cases"] is None
    assert unknown["limited"] is False
    assert "0 cases" not in format_evidence_volume(unknown)


def test_attempt_policy_marks_nonuniform_evidence_in_agent_payload():
    conditions = {
        "with_skill": _condition(cases=3, planned=5, scored=5),
        "without_skill": _condition(cases=3, planned=5, scored=5),
    }
    for condition in conditions.values():
        condition["execution_status"] = "succeeded"
    payload = build_agent_eval_payload(
        "test-skill",
        {
            "test-agent": {
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.7),
                "conditions": conditions,
                "execution_status": "succeeded",
                "rewards": [],
                "rewards_baseline": [],
                "rewards_complete": True,
                "rewards_baseline_complete": True,
            }
        },
        attempt_policy={"max_attempts": 2, "stop_on_pass": True},
        persisted_dataset_summary={"total_tasks": 3, "negative_tasks": 1, "unclassified_tasks": 0},
    )
    assert payload is not None
    evidence = payload["evidence_volume"]
    assert evidence["uniform_attempts"] is False
    assert evidence["attempts_per_case"] == 2
    assert "Based on 3 cases, 5 attempts (up to 2 per case)" in evidence["display"]


def test_any_incomplete_non_skipped_arm_marks_evidence_limited():
    volume = evidence_volume_for_conditions(
        {
            "with_skill": _condition(cases=10, planned=10, scored=10),
            "without_skill": _condition(cases=10, planned=10, scored=9),
        }
    )
    assert volume["limited"] is True
    assert format_evidence_volume(volume).startswith("Limited evidence:")


def test_failure_case_ids_do_not_inflate_known_case_count():
    volume = evidence_volume_for_conditions(
        {
            "with_skill": {
                **_condition(cases=1, planned=2, scored=1),
                "failure_cases": 8,
            }
        },
        trial_failures={
            "with_skill": [
                {"trial": "case-001__0IgxLd6", "reason": "execution failure"},
                {"trial": "case-002__0IgxLd6", "reason": "execution failure"},
            ]
        },
    )
    assert volume["cases"] == 1


def test_nonuniform_attempts_use_per_case_maximum():
    volume = evidence_volume(
        cases=3,
        attempts_planned=5,
        attempts_scored=5,
        execution_failures=0,
        judge_failures=0,
        uniform_attempts=False,
        attempts_per_case=2,
    )
    assert "Based on 3 cases, 5 attempts (up to 2 per case)" in format_evidence_volume(volume)


def test_html_evidence_volume_survives_bounded_large_payload():
    ae = _provisional_agent_eval()
    ae["trials"] = [{"trial_id": f"case-{index}", "output": "x" * 300} for index in range(1200)]
    expected = format_evidence_volume(ae["evidence_volume"])
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = ae
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert expected in html


def test_result_display_labels_each_agent_evidence_line():
    result = {
        "execution_status": "succeeded",
        "attempt_policy": {"max_attempts": 2, "stop_on_pass": False},
        "agents": {
            "codex": {"conditions": {"with_skill": _condition(cases=10, planned=20, scored=18)}},
            "opencode": {"conditions": {"with_skill": _condition(cases=10, planned=20, scored=20)}},
        },
    }
    output = render_result(result)
    assert "codex: Limited evidence:" in output
    assert "opencode: Based on" in output


def test_evidence_volume_counts_cases_once_and_separates_failures_by_arm():
    conditions = {
        "with_skill": _condition(cases=3, planned=30, scored=27),
        "without_skill": _condition(cases=3, planned=30, scored=29),
    }
    volume = evidence_volume_for_conditions(
        conditions,
        trial_failures={
            "with_skill": [{"trial": "case-1__attempt-10", "reason": "Required judge evaluation failed: accuracy"}]
        },
    )

    assert volume["cases"] == 3
    assert volume["attempts_planned"] == 30
    assert volume["attempts_scored"] == 27
    assert volume["execution_failures"] == 2
    assert volume["judge_failures"] == 1
    assert volume["by_arm"]["without_skill"] == {
        "cases": 3,
        "attempts_planned": 30,
        "attempts_scored": 29,
        "execution_failures": 1,
        "judge_failures": 0,
    }


def test_evidence_volume_formats_singular_and_complete_coverage():
    value = evidence_volume(
        cases=1,
        attempts_planned=1,
        attempts_scored=1,
        execution_failures=0,
        judge_failures=0,
    )

    assert format_evidence_volume(value) == (
        "Limited evidence: Based on 1 case \u00d7 1 attempt = 1 scored attempt (0 execution failures, 0 judge failures)"
    )


def test_agent_payload_includes_evidence_volume_with_per_arm_counts():
    conditions = {
        "with_skill": _condition(cases=3, planned=30, scored=27),
        "without_skill": _condition(cases=3, planned=30, scored=29),
    }
    conditions["with_skill"]["judge_failures"] = 1
    agent = _build_agent(
        "test-agent",
        {
            "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
            "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.7),
            "conditions": conditions,
            "execution_status": "failed",
            "expected_attempts": 60,
            "scored_attempts": 56,
            "rewards": [],
            "rewards_baseline": [],
            "rewards_complete": True,
            "rewards_baseline_complete": True,
        },
        list(DEFAULT_METRICS),
        list(DEFAULT_METRICS),
        None,
        DEFAULT_SCORE_POLICY,
    )

    assert agent["evidence_volume"]["cases"] == 3
    assert agent["evidence_volume"]["attempts_planned"] == 30
    assert agent["evidence_volume"]["attempts_scored"] == 27
    assert agent["evidence_volume"]["execution_failures"] == 2
    assert agent["evidence_volume"]["judge_failures"] == 1
    assert set(agent["evidence_volume"]["by_arm"]) == {"with_skill", "without_skill"}
