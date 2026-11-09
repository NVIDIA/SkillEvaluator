# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import pytest
from rich.console import Console

from skillevaluator.evaluation.tier3_report import _build_agent
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


@pytest.mark.parametrize("status", ("provisional", "sufficient"))
def test_evidence_volume_is_rendered_on_all_report_surfaces(status):
    volume = _volume(status)
    expected = format_evidence_volume(volume)
    ae = _agent_eval(volume)
    assert expected.startswith("Limited evidence:") == (status == "provisional")

    cli = StringIO()
    CLIReporter._print_agent_eval_tables(ae, Console(file=cli, force_terminal=False))
    assert expected in " ".join(cli.getvalue().split())

    evaluate_output = render_result(
        {
            "execution_status": "succeeded",
            "best_agent": "test-agent",
            "agents": {
                "test-agent": {
                    "conditions": ae["conditions"],
                    "trial_failures": {},
                }
            },
        }
    )
    assert expected in " ".join(evaluate_output.split())

    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = ae
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert expected in html
    assert ("t3-evidence-volume warning" in html) == (status == "provisional")

    benchmark = BenchmarkReporter(include_timestamp=False, report_scope="tier3").render_all([result])
    assert f"Evidence volume: {expected}." in benchmark


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
