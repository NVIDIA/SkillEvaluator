# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from skillevaluator.reporting.benchmark import BenchmarkReporter
from skillevaluator.tier3.attempt_verdict import attempt_verdict
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


def test_wrong_answer_fails_without_changing_overall_score() -> None:
    reward = _reward(1, goal_accuracy=0.0)
    score = overall_score(reward)

    verdict = attempt_verdict(score, 0.5, reward["goal_accuracy"])
    summary = _summary([reward])

    assert score == 0.9
    assert verdict.passed is False
    assert verdict.reason == "wrong answer (goal accuracy 0.00)"
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
    assert summary["cases"]["case-1"]["attempts"][1]["reason"] == "wrong answer (goal accuracy 0.90)"


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
                }
            }
        },
        (),
    )

    report = "\n".join(lines)
    assert "| Overall |" in report
    assert "| Correct answers | 9/12 \\| 7/12 \\| +2 |" in report
