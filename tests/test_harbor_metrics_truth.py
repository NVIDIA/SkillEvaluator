# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest

from skillevaluator.tier3.harbor.collector import (
    _compute_lift,
    _condition_execution_summary,
    _paired_pass_comparison,
    _pass_summary,
    _wilson_score_interval,
)
from skillevaluator.tier3.harbor.metrics import (
    CUSTOM_ONLY_METRIC_SET,
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    average_metrics,
    extract_custom_metrics,
    metric_value,
    overall_score,
)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_metric_inputs_reject_nonfinite_values(invalid: float) -> None:
    assert metric_value({"security": invalid}, "security") is None
    assert metric_value({"metrics": {"security": {"score": invalid}}}, "security") is None
    assert extract_custom_metrics({"custom_metrics": {"latency": invalid}}) == {}


def test_average_metrics_omits_unavailable_canonical_metrics() -> None:
    scores, metric_set, metrics = average_metrics(
        [
            {
                "metric_set": DEFAULT_METRIC_SET,
                "security": 1.0,
                "accuracy": float("nan"),
            }
        ]
    )

    assert metric_set == DEFAULT_METRIC_SET
    assert metrics == DEFAULT_METRICS
    assert scores == {"security": 1.0}


def test_overall_score_requires_a_complete_finite_metric_set() -> None:
    complete = dict.fromkeys(DEFAULT_METRICS, 0.8)
    incomplete = dict(complete)
    incomplete.pop("behavior_check")
    invalid = dict(complete)
    invalid["behavior_check"] = float("inf")

    assert overall_score(complete) == pytest.approx(0.8)
    assert overall_score(incomplete) is None
    assert overall_score(invalid) is None
    assert overall_score({"metric_set": CUSTOM_ONLY_METRIC_SET, "overall": float("nan")}) is None


def test_lift_omits_unpaired_metrics_and_incomplete_overall() -> None:
    lift = _compute_lift(
        {"security": 1.0, "accuracy": 0.8},
        {"security": 0.5, "goal_accuracy": 0.4},
    )

    assert lift == {
        "security": {
            "with_skill": 1.0,
            "without_skill": 0.5,
            "delta": 0.5,
            "direction": "up",
        }
    }


def test_pass_summary_marks_incomplete_reward_unscored() -> None:
    reward = {
        "entry_id": "case-001",
        "_trial_name": "case-001__attempt1",
        "_trial_root_name": "trial-1",
        "security": 1.0,
    }

    summary = _pass_summary(
        [reward],
        n_attempts=1,
        pass_threshold=0.5,
        expected_cases=1,
        expected_case_ids=["case-001"],
    )

    case = summary["cases"]["case-001"]
    # An incomplete reward is not a scored attempt: it contributes no pass@k
    # attempt row and the attempt is reported as missing instead.
    assert case["attempts"] == []
    assert case["passed"] is False
    assert case["attempts_used"] == 0
    assert case["attempts_missing"] == 1
    assert case["best_score"] is None
    assert summary["attempts_used"] == 0
    assert math.isfinite(summary["rate"])


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [
        (0, 4, (0.0, 0.4899)),
        (2, 4, (0.15, 0.85)),
        (4, 4, (0.5101, 1.0)),
    ],
)
def test_wilson_interval_reports_bounded_case_rate_uncertainty(
    successes: int,
    total: int,
    expected: tuple[float, float],
) -> None:
    interval = _wilson_score_interval(successes, total)

    assert interval is not None
    assert interval["method"] == "wilson_score"
    assert interval["confidence_level"] == 0.95
    assert interval["lower"] == pytest.approx(expected[0], abs=0.0001)
    assert interval["upper"] == pytest.approx(expected[1], abs=0.0001)


def test_wilson_interval_is_unavailable_without_cases() -> None:
    assert _wilson_score_interval(0, 0) is None


def test_paired_pass_comparison_preserves_case_direction_and_exact_test() -> None:
    with_skill = {
        "total_cases": 4,
        "cases": {
            "both": {"passed": True},
            "improved": {"passed": True},
            "regressed": {"passed": False},
            "neither": {"passed": False},
        },
    }
    without_skill = {
        "total_cases": 4,
        "cases": {
            "both": {"passed": True},
            "improved": {"passed": False},
            "regressed": {"passed": True},
            "neither": {"passed": False},
        },
    }

    paired = _paired_pass_comparison(with_skill, without_skill)

    assert paired["pairing_status"] == "complete"
    assert paired["paired_cases"] == 4
    assert paired["both_pass"] == 1
    assert paired["with_skill_only_pass"] == 1
    assert paired["without_skill_only_pass"] == 1
    assert paired["neither_pass"] == 1
    assert paired["paired_rate_delta"] == 0.0
    assert paired["mcnemar_exact"]["p_value"] == 1.0


def test_paired_pass_comparison_does_not_issue_exact_test_for_partial_pairing() -> None:
    paired = _paired_pass_comparison(
        {"total_cases": 2, "cases": {"shared": {"passed": True}, "with-only": {"passed": True}}},
        {"total_cases": 2, "cases": {"shared": {"passed": False}, "without-only": {"passed": False}}},
    )

    assert paired["pairing_status"] == "partial"
    assert paired["paired_cases"] == 1
    assert paired["with_skill_unpaired_case_ids"] == ["with-only"]
    assert paired["without_skill_unpaired_case_ids"] == ["without-only"]
    assert "mcnemar_exact" not in paired


def test_condition_marks_incomplete_reward_as_unscored() -> None:
    reward = {
        "entry_id": "case-001",
        "_trial_root_name": "trial-1",
        "security": 1.0,
    }

    summary = _condition_execution_summary(
        [reward],
        expected_case_ids=["case-001"],
        expected_cases=1,
        n_attempts=1,
        job_failure="",
    )

    assert summary["execution_status"] == "failed"
    assert summary["scored_attempts"] == 0
    assert any("incomplete or non-finite" in error for error in summary["execution_errors"])
