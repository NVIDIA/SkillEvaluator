# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-arm applicability for the no-skill baseline."""

from __future__ import annotations

from typing import Any

from skillevaluator.constants import DEFAULT_SCORE_POLICY
from skillevaluator.tier3.harbor import metrics
from skillevaluator.tier3.harbor.collector import _compute_lift
from skillevaluator.tier3.harbor.templates import eval as verifier


def reward(case: str, **overrides: float | None) -> dict[str, Any]:
    result = dict.fromkeys(metrics.DEFAULT_METRICS, 1.0)
    result.update(
        metric_set=metrics.DEFAULT_METRIC_SET,
        score_policy=DEFAULT_SCORE_POLICY,
        entry_id=case,
        details={},
    )
    for metric, value in overrides.items():
        result[metric] = value
        if value is None:
            result["details"][metric] = {
                "score": None,
                "status": "not_applicable",
                "reason": "N/A: this metric has no applicable reference or skill",
            }
    return result


def test_skill_metrics_apply_to_skill_and_matching_acceptable_alternate():
    assert not verifier.skill_metrics_applicable(
        False,
        True,
        "target-skill",
        workspace_skill_names=[],
    )
    assert verifier.skill_metrics_applicable(True, True, "target-skill")
    assert verifier.skill_metrics_applicable(
        False,
        True,
        "target-skill",
        acceptable_skills=["alternate-skill"],
        workspace_skill_names=["alternate-skill"],
    )


def test_reward_overall_averages_three_scored_dimensions_when_skill_metrics_are_na():
    result = {
        "security": 0.8,
        "accuracy": 0.6,
        "skill_execution": None,
        "goal_accuracy": 0.4,
        "behavior_check": 0.8,
        "skill_efficiency": None,
    }
    details = {
        metric: {
            "score": None,
            "status": "not_applicable",
            "reason": verifier.SKILL_METRICS_NOT_APPLICABLE_REASON,
        }
        for metric in ("skill_execution", "skill_efficiency")
    }

    assert verifier._reward_overall(result, details) == 0.6667


def test_lift_basis_excludes_baseline_na_skill_metrics_but_pairs_other_metrics():
    with_skill = reward(
        "case-1",
        security=0.9,
        accuracy=0.8,
        skill_execution=0.7,
        goal_accuracy=0.6,
        behavior_check=0.8,
        skill_efficiency=0.9,
    )
    without_skill = reward(
        "case-1",
        security=0.7,
        accuracy=0.6,
        skill_execution=None,
        goal_accuracy=0.4,
        behavior_check=0.6,
        skill_efficiency=None,
    )
    lift = _compute_lift(
        metrics.average_metrics([with_skill])[0],
        metrics.average_metrics([without_skill])[0],
        [with_skill],
        [without_skill],
        ["case-1"],
        score_policy=DEFAULT_SCORE_POLICY,
    )

    assert "skill_execution" not in lift
    assert "skill_efficiency" not in lift
    excluded = {item["metric"]: item for item in lift["basis"]["excluded_metrics"]}
    assert excluded["skill_execution"] == {
        "metric": "skill_execution",
        "reason": "not_applicable",
        "arms": ["without_skill"],
    }
    assert excluded["skill_efficiency"] == {
        "metric": "skill_efficiency",
        "reason": "not_applicable",
        "arms": ["without_skill"],
    }
    for metric in ("security", "accuracy", "goal_accuracy", "behavior_check"):
        assert lift[metric]["delta"] == 0.2
        assert lift["basis"]["metrics"][metric]["cases"] == 1
