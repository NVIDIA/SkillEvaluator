# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from skillevaluator.evaluation.tier3_report import (
    _agent_quality_verdict,
    _agent_quality_verdict_reason,
    _build_conclusions,
    _count_comparison,
    build_agent_eval_payload,
)
from skillevaluator.reporting.benchmark import BenchmarkReporter
from skillevaluator.tier3.attempt_verdict import attempt_verdict, required_check_failure_reason
from skillevaluator.tier3.harbor.collector import _pass_summary
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS, overall_score
from skillevaluator.tier3.results_location import _legacy_pass_at_k_is_complete


def _reward(attempt: int, *, goal_accuracy: float | None) -> dict:
    reward = {
        "entry_id": "case-1",
        "_trial_name": f"case-1__attempt{attempt}",
        "_trial_root_name": f"case-1__attempt{attempt}",
        "_attempt_ordinal": attempt,
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
    }
    reward["goal_accuracy"] = goal_accuracy
    return reward


def _summary(rewards: list[dict]) -> dict:
    return _pass_summary(
        rewards,
        n_attempts=len(rewards),
        pass_threshold=0.5,
        expected_cases=1,
        expected_case_ids=["case-1"],
    )


def _security_reward(score: float) -> dict:
    reward = _reward(1, goal_accuracy=1.0)
    reward["security"] = score
    reward["details"] = {
        "security": {
            "findings": [
                {"message": "Non-scoring observation", "score_impact": False},
                {"message": "Network command was blocked", "score_impact": True},
            ],
        }
    }
    return reward


def test_goal_not_met_fails_without_changing_overall_score() -> None:
    reward = _reward(1, goal_accuracy=0.0)
    score = overall_score(reward)

    verdict = attempt_verdict(score, 0.5, reward["goal_accuracy"])
    summary = _summary([reward])

    assert score == 0.9
    assert verdict.passed is False
    assert verdict.reason == "goal not met (goal accuracy 0.00)"
    assert summary["cases"]["case-1"]["best_score"] == score
    assert summary["cases"]["case-1"]["attempts"][0]["passed"] is False
    assert summary["cases"]["case-1"]["attempts"][0]["reason"] == verdict.reason
    assert summary["passed_cases"] == 0
    assert _legacy_pass_at_k_is_complete(
        summary,
        num_trials=1,
        require_scored_attempt=True,
    )


def test_unscored_goal_accuracy_preserves_overall_verdict() -> None:
    reward = _reward(1, goal_accuracy=None)
    reward["details"] = {"goal_accuracy": {"status": "not_applicable"}}

    verdict = attempt_verdict(overall_score(reward), 0.5, None)
    summary = _summary([reward])

    assert verdict.passed is True
    assert verdict.reason is None
    assert summary["cases"]["case-1"]["passed"] is True
    assert summary["correct_answers"] == {"correct": 0, "scored": 0}


def test_correct_answer_counts_include_only_scored_attempts() -> None:
    summary = _summary([_reward(1, goal_accuracy=1.0), _reward(2, goal_accuracy=0.9)])

    assert summary["correct_answers"] == {"correct": 1, "scored": 2}
    assert summary["cases"]["case-1"]["attempts"][1]["reason"] == "goal not met (goal accuracy 0.90)"


def test_security_issue_fails_without_changing_overall_score() -> None:
    reward = _security_reward(0.5)
    score = overall_score(reward)

    verdict = attempt_verdict(score, 0.5, reward=reward)
    summary = _summary([reward])

    assert verdict.passed is False
    assert verdict.reason == "security issue (security 0.50): Network command was blocked"
    assert overall_score(reward) == score
    assert summary["cases"]["case-1"]["attempts"][0]["score"] == score
    assert summary["cases"]["case-1"]["best_score"] == score
    assert summary["required_checks"] == {
        "passed": 0,
        "scored": 1,
        "wrong_answer": 0,
        "security": 1,
    }


def test_security_unsafe_action_fails_without_changing_overall_score() -> None:
    reward = _security_reward(0.0)
    score = overall_score(reward)

    verdict = attempt_verdict(score, 0.5, reward=reward)
    summary = _summary([reward])

    assert verdict.passed is False
    assert verdict.reason == "security issue (security 0.00): Network command was blocked"
    assert overall_score(reward) == score
    assert summary["cases"]["case-1"]["attempts"][0]["score"] == score


def test_security_issue_without_finding_message_omits_colon() -> None:
    reward = _security_reward(0.0)
    reward["details"]["security"] = {"findings": []}

    verdict = attempt_verdict(overall_score(reward), 0.5, reward=reward)

    assert verdict.reason == "security issue (security 0.00)"


def test_unsupported_codex_security_result_does_not_gate_attempt() -> None:
    unsupported_results = (
        {
            "supported": False,
            "findings": [{"message": "Unsupported wrapper", "score_impact": True}],
        },
        {
            "unsupported_evidence": ["unsupported_native_codex_exec_wrapper"],
            "findings": [{"message": "Unsupported wrapper", "score_impact": True}],
        },
        {
            "findings": [
                {
                    "message": "Unsupported native Codex exec wrapper could not be safely normalized",
                    "score_impact": True,
                }
            ]
        },
    )
    for security_result in unsupported_results:
        reward = _security_reward(0.0)
        reward["details"]["security"] = security_result
        verdict = attempt_verdict(overall_score(reward), 0.5, reward=reward)

        assert verdict.passed is True
        assert verdict.required_failure is None
        assert verdict.security_clean is False
        assert verdict.security_exempt is True


def test_multiple_unsupported_wrapper_findings_are_exempt_and_count_as_required_pass() -> None:
    reward = _security_reward(0.0)
    reward["details"]["security"] = {
        "findings": [
            {"type": "unsupported_tool_wrapper", "score_impact": True, "message": "first wrapper"},
            {"type": "unsupported_tool_wrapper", "score_impact": True, "message": "second wrapper"},
        ]
    }

    verdict = attempt_verdict(overall_score(reward), 0.5, reward=reward)
    summary = _summary([reward])

    assert verdict.passed is True
    assert verdict.security_exempt is True
    assert summary["required_checks"] == {"passed": 1, "scored": 1, "wrong_answer": 0, "security": 0}


def test_required_check_counts_deduplicate_goal_and_security_failures() -> None:
    reward = _security_reward(0.0)
    reward["goal_accuracy"] = 0.0

    summary = _summary([reward])

    assert summary["required_checks"] == {"passed": 0, "scored": 1, "wrong_answer": 1, "security": 1}
    assert required_check_failure_reason(summary["required_checks"]) == (
        "1 of 1 cases failed: 1 goal not met, 1 security issue"
    )


def test_expected_case_filter_and_different_denominators_use_scored_rates() -> None:
    expected = _reward(1, goal_accuracy=1.0)
    extra = {**_reward(1, goal_accuracy=0.0), "entry_id": "unexpected", "_trial_name": "unexpected__attempt1"}
    summary = _pass_summary(
        [expected, extra],
        n_attempts=1,
        pass_threshold=0.5,
        expected_cases=1,
        expected_case_ids=["case-1"],
    )
    comparison = _count_comparison(
        {
            "pass_with_skill": {"required_checks": {"passed": 1, "scored": 2}},
            "pass_without_skill": {"required_checks": {"passed": 1, "scored": 4}},
        },
        "required_checks",
        numerator="passed",
    )

    assert summary["correct_answers"] == {"correct": 1, "scored": 1}
    assert summary["security_clean"] == {"clean": 1, "scored": 1}
    assert comparison["difference"] == 25.0
    assert comparison["difference_unit"] == "percentage_points"


def test_shared_metric_pass_ignores_unshared_goal_gate() -> None:
    reward = _reward(1, goal_accuracy=0.0)
    summary = _pass_summary(
        [reward],
        n_attempts=1,
        pass_threshold=0.5,
        expected_cases=1,
        expected_case_ids=["case-1"],
        shared_metrics_by_case={"case-1": {"security"}},
    )

    assert summary["cases"]["case-1"]["passed"] is True
    assert summary["required_checks"]["passed"] == 1
    assert summary["correct_answers"] == {"correct": 0, "scored": 0}


def test_never_activated_agent_remains_inconclusive_before_required_failures() -> None:
    agent = {
        "execution_status": "succeeded",
        "skill_activation": {"status": "never_activated"},
        "required_checks": {"with_skill": {"wrong_answer": 1, "security": 1, "scored": 1}},
    }

    assert _agent_quality_verdict(agent) == "inconclusive"
    assert _agent_quality_verdict_reason(agent) is None


def test_unscored_security_preserves_overall_verdict() -> None:
    reward = _security_reward(0.0)
    reward["details"]["security"] = {"status": "error"}

    verdict = attempt_verdict(overall_score(reward), 0.5, reward=reward)

    assert verdict.passed is True
    assert verdict.security_clean is None
    assert verdict.security_exempt is False


def test_unsupported_security_exemption_survives_cases_needing_review_summary() -> None:
    conclusions = _build_conclusions(
        {
            "codex": {
                "with_skill": 0.8,
                "trials": [
                    {
                        "entry_id": "unsupported-codex",
                        "overall": 0.8,
                        "attempt_verdict": {"passed": True, "reason": None, "warnings": []},
                    }
                ],
            }
        },
        [],
        pass_threshold=0.5,
    )

    assert not any(item["title"] == "Cases needing review" for item in conclusions)


def test_required_failure_with_unscored_overall_is_a_failing_trial() -> None:
    conclusions = _build_conclusions(
        {
            "codex": {
                "with_skill": 0.8,
                "trials": [
                    {
                        "entry_id": "wrong-answer",
                        "overall": None,
                        "attempt_verdict": {
                            "passed": False,
                            "reason": "goal not met (goal accuracy 0.00)",
                            "warnings": [],
                        },
                    }
                ],
            }
        },
        [],
        pass_threshold=0.5,
    )

    assert any(item["title"] == "Cases needing review" for item in conclusions)
    assert not any(item["title"] == "Unscored trials" for item in conclusions)


def test_unscored_attempt_verdict_is_not_a_failing_trial() -> None:
    conclusions = _build_conclusions(
        {
            "codex": {
                "with_skill": 0.8,
                "trials": [
                    {
                        "entry_id": "unscored",
                        "overall": None,
                        "attempt_verdict": {
                            "passed": False,
                            "reason": "overall score was not available",
                            "warnings": [],
                        },
                    }
                ],
            }
        },
        [],
        pass_threshold=0.5,
    )

    assert any(item["title"] == "Unscored trials" for item in conclusions)
    assert not any(item["title"] == "Cases needing review" for item in conclusions)


def test_missing_boolean_attempt_verdict_falls_back_to_overall() -> None:
    conclusions = _build_conclusions(
        {
            "codex": {
                "with_skill": 0.8,
                "trials": [{"entry_id": "below-threshold", "overall": 0.25, "attempt_verdict": {}}],
            }
        },
        [],
        pass_threshold=0.5,
    )

    assert any(item["title"] == "Cases needing review" for item in conclusions)


def test_unsupported_security_exemption_survives_persisted_attempt_validation() -> None:
    payload = {
        "k": 1,
        "pass_threshold": 0.5,
        "stop_on_pass": False,
        "passed_cases": 1,
        "failed_cases": 0,
        "total_cases": 1,
        "rate": 1.0,
        "attempts_used": 1,
        "max_attempts_possible": 1,
        "avg_attempts_used": 1.0,
        "extra_cases": [],
        "cases": {
            "unsupported-codex": {
                "passed": True,
                "first_pass_attempt": 1,
                "attempts_used": 1,
                "attempts_skipped": 0,
                "attempts_missing": 0,
                "best_score": 0.8,
                "attempts": [
                    {
                        "attempt": 1,
                        "trial": "unsupported-codex__attempt1",
                        "score": 0.8,
                        "passed": True,
                        "goal_accuracy": None,
                        "reason": None,
                        "required_failure": None,
                        "security_score": 0.0,
                        "security_clean": False,
                        "security_message": "Unsupported native Codex exec wrapper could not be safely normalized",
                        "security_exempt": True,
                    }
                ],
            }
        },
    }

    assert _legacy_pass_at_k_is_complete(payload, num_trials=1, require_scored_attempt=True)


def test_behavior_check_failure_warns_without_gating_attempt() -> None:
    reward = _reward(1, goal_accuracy=1.0)
    reward["behavior_check"] = 0.5
    reward["details"] = {
        "behavior_check": {
            "results": [
                {"step": 1, "passed": True},
                {"step": 2, "passed": False},
                {"step": 3, "passed": False},
            ]
        }
    }

    verdict = attempt_verdict(overall_score(reward), 0.5, reward=reward)

    assert verdict.passed is True
    assert verdict.warnings == ("missed expected steps 2, 3 of 3 (behavior 0.50)",)


def test_benchmark_reports_correct_answers_under_overall() -> None:
    lines: list[str] = []
    BenchmarkReporter._render_results_at_a_glance(
        lines,
        {
            "agents": {
                "opencode": {
                    "baseline": 0.7,
                    "with_skill": 0.9,
                    "correct_answers": {
                        "with_skill": {"correct": 9, "scored": 12},
                        "without_skill": {"correct": 7, "scored": 12},
                        "difference": 2,
                    },
                    "required_checks": {
                        "with_skill": {"passed": 9, "scored": 12},
                        "without_skill": {"passed": 7, "scored": 12},
                        "difference": 2,
                    },
                    "security_clean": {
                        "with_skill": {"clean": 11, "scored": 12},
                        "without_skill": {"clean": 9, "scored": 12},
                        "difference": 2,
                    },
                    "pass_at_k": {
                        "with_skill": {
                            "cases": {
                                "wrong-ground-truth": {
                                    "passed": False,
                                    "attempts_used": 1,
                                    "best_score": 0.82,
                                    "reason": "goal not met (goal accuracy 0.00)",
                                    "warnings": ["missed expected step 1 of 1 (behavior 0.00)"],
                                }
                            }
                        }
                    },
                }
            }
        },
        (),
    )

    report = "\n".join(lines)
    assert "| Overall |" in report
    assert "| Required checks passed | 9/12 \\| 7/12 \\| +2 |" in report
    assert "| Goals met | 9/12 \\| 7/12 \\| +2 |" in report
    assert "| Security clean | 11/12 \\| 9/12 \\| +2 |" in report
    assert "| wrong-ground-truth | 0.82 | goal not met (goal accuracy 0.00) |" in report
    assert "missed expected step 1 of 1 (behavior 0.00)" in report


def test_tier3_benchmark_callout_includes_required_failure_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        "skillevaluator.reporting.benchmark.completed_agent_result",
        lambda _payload: {
            "verdict": "fail",
            "reason": "1 of 2 cases failed: 1 goal not met",
            "name": "Codex",
            "score": 0.82,
            "lift": 0.2,
            "needs_attention": False,
            "recovered_attempts": 0,
        },
    )

    report = BenchmarkReporter(report_scope="tier3").render_all([])

    assert report.startswith("> **Tier 3 completed-agent result: FAIL** — 1 of 2 cases failed: 1 goal not met")


def test_quality_headline_fails_for_with_skill_required_check_failure_only() -> None:
    agent = {
        "with_skill": {
            "security": 0.9,
            "skill_execution": 0.9,
            "skill_efficiency": 0.9,
            "accuracy": 0.9,
            "goal_accuracy": 0.9,
            "behavior_check": 0.9,
        },
        "without_skill": {
            "security": 0.2,
            "skill_execution": 0.2,
            "skill_efficiency": 0.2,
            "accuracy": 0.2,
            "goal_accuracy": 0.2,
            "behavior_check": 0.2,
        },
        "execution_status": "succeeded",
        "pass_with_skill": {
            "required_checks": {"passed": 1, "scored": 2, "wrong_answer": 1, "security": 0},
            "correct_answers": {"correct": 1, "scored": 2},
            "security_clean": {"clean": 2, "scored": 2},
        },
        "pass_without_skill": {
            "required_checks": {"passed": 2, "scored": 2, "wrong_answer": 0, "security": 0},
            "correct_answers": {"correct": 2, "scored": 2},
            "security_clean": {"clean": 2, "scored": 2},
        },
    }
    payload = build_agent_eval_payload("demo", {"codex": agent})

    assert payload is not None
    assert payload["overall_lift"] > 0
    assert payload["verdict"] == "fail"
    assert payload["verdict_reason"] == "1 of 2 cases failed: 1 goal not met"
    assert payload["required_checks"]["with_skill"] == {
        "passed": 1,
        "scored": 2,
        "wrong_answer": 1,
        "security": 0,
    }


def test_no_skill_only_required_check_failure_does_not_fail_headline() -> None:
    agent = {
        "with_skill": {
            "security": 0.9,
            "skill_execution": 0.9,
            "skill_efficiency": 0.9,
            "accuracy": 0.9,
            "goal_accuracy": 0.9,
            "behavior_check": 0.9,
        },
        "without_skill": {
            "security": 0.2,
            "skill_execution": 0.2,
            "skill_efficiency": 0.2,
            "accuracy": 0.2,
            "goal_accuracy": 0.2,
            "behavior_check": 0.2,
        },
        "execution_status": "succeeded",
        "pass_with_skill": {
            "required_checks": {"passed": 2, "scored": 2, "wrong_answer": 0, "security": 0},
        },
        "pass_without_skill": {
            "required_checks": {"passed": 1, "scored": 2, "wrong_answer": 1, "security": 0},
        },
    }
    payload = build_agent_eval_payload("demo", {"codex": agent})

    assert payload is not None
    assert payload["verdict"] == "pass"
    assert payload["verdict_reason"] is None


def test_run_verdict_uses_passing_agent_counts_and_has_no_failure_reason() -> None:
    def agent(score: float, passed: int) -> dict:
        return {
            "with_skill": dict.fromkeys(DEFAULT_METRICS, score),
            "without_skill": dict.fromkeys(DEFAULT_METRICS, score),
            "execution_status": "succeeded",
            "expected_attempts": 1,
            "scored_attempts": 1,
            "pass_with_skill": {
                "pass_threshold": 0.5,
                "required_checks": {"passed": passed, "scored": 1, "wrong_answer": 1 - passed, "security": 0},
                "correct_answers": {"correct": passed, "scored": 1},
                "security_clean": {"clean": 1, "scored": 1},
            },
            "pass_without_skill": {
                "required_checks": {"passed": 1, "scored": 1, "wrong_answer": 0, "security": 0},
                "correct_answers": {"correct": 1, "scored": 1},
                "security_clean": {"clean": 1, "scored": 1},
            },
        }

    codex = agent(1.0, 0)
    codex["skill_activation"] = {"status": "never_activated"}
    payload = build_agent_eval_payload("demo", {"codex": codex, "claude": agent(0.9, 1)})

    assert payload is not None
    assert payload["verdict"] == "pass"
    assert payload["verdict_reason"] is None
    assert payload["required_checks"]["with_skill"]["passed"] == 1
    assert payload["correct_answers"]["with_skill"]["correct"] == 1


def test_required_check_failure_reason_pluralizes_and_omits_zero_counts() -> None:
    assert required_check_failure_reason({"wrong_answer": 1, "security": 0, "scored": 2}) == (
        "1 of 2 cases failed: 1 goal not met"
    )
    assert required_check_failure_reason({"wrong_answer": 2, "security": 0, "scored": 3}) == (
        "2 of 3 cases failed: 2 goals not met"
    )
    assert required_check_failure_reason({"wrong_answer": 0, "security": 2, "scored": 3}) == (
        "2 of 3 cases failed: 2 security issues"
    )
    assert required_check_failure_reason({"wrong_answer": 1, "security": 1, "scored": 4}) == (
        "2 of 4 cases failed: 1 goal not met, 1 security issue"
    )
