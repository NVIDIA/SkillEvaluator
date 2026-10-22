# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from skillevaluator.evaluation.tier3_report import (
    _build_agent,
    _build_integration_report,
    _validation_result_from_payload,
)


def test_integration_report_is_plugin_only_and_reconciles_operands() -> None:
    best = {
        "with_skill": 0.80,
        "baseline": 0.30,
        "sum_of_parts": 0.65,
        "integration_completeness": {"complete": True},
    }
    config = {
        "eval_target": {"kind": "plugin"},
        "skill_workspace": {
            "staged_skills": ["loader", "summarizer"],
            "baseline_includes_workspace_skills": False,
            "sum_of_parts_arm": True,
        },
    }
    report = _build_integration_report(best, config)
    assert report is not None
    assert report["integration_lift"] == 0.15
    assert report["verdict"] == "real_integration"
    assert report["point_verdict"] == "real_integration"
    assert report["measured"] is True
    assert report["reason"] is None
    assert report["lift_mode_effective"] == "both"
    assert report["report_only"] is True

    assert _build_integration_report(best, {**config, "eval_target": {"kind": "skill"}}) is None


def test_incomplete_sum_of_parts_never_claims_integration() -> None:
    report = _build_integration_report(
        {"with_skill": 0.9, "sum_of_parts": 0.2, "integration_completeness": {"complete": False}},
        {
            "eval_target": {"kind": "plugin"},
            "skill_workspace": {"staged_skills": ["member"], "sum_of_parts_arm": True},
        },
    )
    assert report is not None
    assert report["verdict"] == "inconclusive"
    assert report["point_verdict"] == "real_integration"
    assert report["complete"] is False
    assert "did not cover the same expected cases" in report["reason"]


def test_custom_only_sum_of_parts_produces_integration_lift() -> None:
    agent = _build_agent(
        "codex",
        {
            "execution_status": "succeeded",
            "conditions": {
                "with_skill": {"execution_status": "succeeded"},
                "without_skill": {"execution_status": "skipped"},
                "sum_of_parts": {"execution_status": "succeeded"},
            },
            "overall_with_skill": 0.8,
            "overall_sum_of_parts": 0.5,
            "integration_completeness": {"complete": True},
        },
        [],
        [],
        None,
    )

    assert agent["sum_of_parts"] == 0.5
    assert agent["integration_lift"] == 0.3
    report = _build_integration_report(
        agent,
        {
            "eval_target": {"kind": "plugin"},
            "skill_workspace": {"staged_skills": ["member"], "sum_of_parts_arm": True},
        },
    )
    assert report is not None
    assert report["complete"] is True
    assert report["verdict"] == "real_integration"


def test_partial_plugin_payload_is_never_reported_as_a_pass() -> None:
    result = _validation_result_from_payload(
        {
            "best_agent": "codex",
            "execution_status": "succeeded",
            "overall_score": 0.8,
            "verdict": "positive",
            "plugin_provenance": {
                "partial": True,
                "unresolved_skill_refs": ["github::other/repo::skills::member"],
            },
        }
    )

    assert result is not None
    assert result.passed is False
    assert result.metadata["execution_status"] == "skipped"
    assert result.metadata["skip_reason"].startswith("INCOMPLETE:")


_SUM_OF_PARTS_RAN = {"conditions": {"sum_of_parts": {"execution_status": "succeeded"}}, "execution_status": "succeeded"}


def test_sum_of_parts_overall_averages_the_rounded_dimension_scores() -> None:
    info = {
        **_SUM_OF_PARTS_RAN,
        "dimensions_sum_of_parts": {
            "security": {"score": 0.00004},
            "correctness": {"score": 0.00004},
            "discoverability": {"score": 0.00007},
        },
    }

    # Each dimension rounds to four places first (0.0, 0.0, 0.0001), as the dimension rows do.
    assert _build_agent("codex", info, ["accuracy"], ["accuracy"], None)["sum_of_parts"] == 0.0


def test_an_arm_without_metrics_falls_back_to_the_engine_overall_only_when_it_ran() -> None:
    ran = {**_SUM_OF_PARTS_RAN, "overall_sum_of_parts": 0.42, "overall_with_skill": 0.9}
    failed = {**ran, "conditions": {"sum_of_parts": {"execution_status": "failed"}}}

    assert _build_agent("codex", ran, [], [], None)["sum_of_parts"] == 0.42
    assert _build_agent("codex", ran, [], [], None)["integration_lift"] == 0.48
    assert _build_agent("codex", failed, [], [], None)["sum_of_parts"] is None


def test_integration_reports_for_match_the_report_payload() -> None:
    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload, integration_reports_for

    succeeded = {"execution_status": "succeeded"}

    def agent(with_plugin: float) -> dict:
        return {
            "with_skill": {"accuracy": with_plugin},
            "without_skill": {"accuracy": 0.5},
            "sum_of_parts": {"accuracy": 0.7},
            "execution_status": "succeeded",
            "conditions": {"with_skill": succeeded, "without_skill": succeeded, "sum_of_parts": succeeded},
            "integration_completeness": {"complete": True, "ratio": float("nan")},
        }

    agents = {"claude-code": agent(0.75), "codex": agent(0.9)}
    run_config = {
        "eval_target": {"kind": "plugin"},
        "lift_mode": {"requested": "both", "effective": "both"},
        "skill_workspace": {"sum_of_parts_arm": True, "staged_skills": ["skills/a", "skills/b"]},
    }

    payload = build_agent_eval_payload("demo-plugin", agents, run_config=run_config, use_llm_judge=False)
    integration, per_agent = integration_reports_for(agents, run_config)

    assert payload is not None and integration is not None
    # The run-level block is the best agent's, and each agent keeps its own named block.
    assert integration == payload["integration"]
    assert integration["agent"] == payload["best_agent"] == "codex"
    assert per_agent == {name: payload["agents"][name]["integration"] for name in agents}
    assert integration["completeness"]["ratio"] is None  # sanitized like the payload
    assert integration_reports_for(agents, {**run_config, "eval_target": {"kind": "skill"}}) == (None, {})
