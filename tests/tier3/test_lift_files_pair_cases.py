# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-metric lift files compare both arms on the same cases and say what they left out.

``lift.json`` and ``integration_lift.json`` used to subtract two arm averages,
each over the trials that arm scored. When one arm records a metric as not
applicable on some cases (the skills-only arm on a case with no expected skill)
the two averages covered different cases: the release-kit pilot compared 32
full-plugin trials with 28 skills-only and 28 loose-parts trials. Now every
metric keeps only the cases both arms scored it on, and a ``basis`` block lists
the counts and the exclusions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _f90_lift_fixtures import collect, metrics

from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS

AGENT = "codex"


def _run(tmp_path: Path) -> dict:
    # Case c has no expected skill: the skills-only arm records the skill metrics as N/A there, while the
    # full plugin scores them (low). The full plugin records accuracy N/A on case b (no ground truth seen).
    arms = {
        AGENT: {
            "with": {
                "a": [metrics(0.9)],
                "b": [metrics(0.9, accuracy=None)],
                "c": [metrics(0.9, skill_execution=0.3, skill_efficiency=0.3)],
            },
            "without": {
                "a": [metrics(0.4, skill=False)],
                "b": [metrics(0.4, skill=False, accuracy=0.0)],
                "c": [metrics(0.4, skill=False)],
            },
            "sumofparts": {
                "a": [metrics(0.6)],
                "b": [metrics(0.6, accuracy=0.0)],
                "c": [metrics(0.6, skill=False)],
            },
        }
    }
    return collect(tmp_path, arms, n_attempts=1)


def _file(tmp_path: Path, name: str) -> dict:
    return json.loads((tmp_path / "results" / AGENT / name).read_text(encoding="utf-8"))


def test_integration_lift_compares_the_cases_both_arms_scored(tmp_path: Path) -> None:
    result = _run(tmp_path)

    lift = _file(tmp_path, "integration_lift.json")
    # Skill metrics: only a and b, where both arms scored them (the old file averaged 0.7 over a, b, c).
    assert lift["skill_execution"] == {"with_skill": 0.9, "without_skill": 0.6, "delta": 0.3, "direction": "up"}
    assert lift["skill_efficiency"]["delta"] == 0.3
    # Accuracy: only a and c, where both scored it (the old file averaged skills-only 0.4 over a, b, c).
    assert lift["accuracy"] == {"with_skill": 0.9, "without_skill": 0.6, "delta": 0.3, "direction": "up"}
    basis = lift["basis"]
    assert basis["arms"] == ["with_skill", "sum_of_parts"]
    assert basis["metrics"]["skill_execution"] == {"cases": 2, "trials": {"with_skill": 2, "sum_of_parts": 2}}
    assert basis["metrics"]["accuracy"] == {"cases": 2, "trials": {"with_skill": 2, "sum_of_parts": 2}}
    assert basis["metrics"]["security"] == {"cases": 3, "trials": {"with_skill": 3, "sum_of_parts": 3}}
    assert basis["excluded_metrics"] == []
    assert basis["excluded_cases"] == [
        {"case": "b", "reason": "not_applicable", "arms": ["with_skill"], "metrics": ["accuracy"]},
        {
            "case": "c",
            "reason": "not_applicable",
            "arms": ["sum_of_parts"],
            "metrics": ["skill_execution", "skill_efficiency"],
        },
    ]
    # The in-memory result carries the same file.
    assert result["agents"][AGENT]["integration_lift"] == lift


def test_lift_json_pairs_cases_and_names_the_metrics_the_baseline_never_scores(tmp_path: Path) -> None:
    _run(tmp_path)

    lift = _file(tmp_path, "lift.json")
    # The no-plugin arm has no skill: its skill metrics are N/A on every trial, so they are not compared.
    assert "skill_execution" not in lift and "skill_efficiency" not in lift
    assert lift["basis"]["excluded_metrics"] == [
        {"metric": "skill_execution", "reason": "not_applicable", "arms": ["without_skill"]},
        {"metric": "skill_efficiency", "reason": "not_applicable", "arms": ["without_skill"]},
    ]
    # Accuracy on a and c only (the old file averaged the no-plugin arm over a, b, c: 0.2667).
    assert lift["accuracy"] == {"with_skill": 0.9, "without_skill": 0.4, "delta": 0.5, "direction": "up"}
    assert lift["basis"]["excluded_cases"] == [
        {"case": "b", "reason": "not_applicable", "arms": ["with_skill"], "metrics": ["accuracy"]},
    ]
    # Rows with nothing excluded keep the arm averages, and the overall is still the mean of the rows.
    assert lift["security"]["with_skill"] == 0.9 and lift["goal_accuracy"]["without_skill"] == 0.4
    assert lift["overall"] == {"with_skill": 0.9, "without_skill": 0.4, "delta": 0.5}


def test_nothing_excluded_keeps_every_value_and_lists_nothing(tmp_path: Path) -> None:
    arms = {
        AGENT: {
            "with": {"a": [metrics(0.8), metrics(0.6)], "b": [metrics(0.9), metrics(0.7)]},
            "without": {"a": [metrics(0.5), metrics(0.3)], "b": [metrics(0.4), metrics(0.2)]},
        }
    }
    result = collect(tmp_path, arms, n_attempts=2)

    lift = _file(tmp_path, "lift.json")
    agent = result["agents"][AGENT]
    for metric in ("security", "skill_execution", "accuracy"):
        assert lift[metric]["with_skill"] == agent["with_skill"][metric]
        assert lift[metric]["without_skill"] == agent["without_skill"][metric]
    assert lift["basis"]["excluded_metrics"] == [] and lift["basis"]["excluded_cases"] == []
    assert lift["basis"]["metrics"]["security"] == {"cases": 2, "trials": {"with_skill": 4, "without_skill": 4}}


# --------------------------------------------------------------------------- #
# The pairing itself: every reason a case or metric is left out               #
# --------------------------------------------------------------------------- #
def _reward(**scores: float | None) -> dict:
    """A default-metric-set reward; ``None`` is N/A (with the verifier's marker), a missing key has no value."""
    reward: dict = {"metric_set": DEFAULT_METRIC_SET, "details": {}}
    for metric, value in scores.items():
        reward[metric] = value
        if value is None:
            reward["details"][metric] = {"score": None, "status": "not_applicable"}
    return reward


def test_pairing_names_every_reason_it_leaves_something_out() -> None:
    from skillevaluator.tier3.harbor.lift_basis import pair_metric_scores

    full = [
        ("a", _reward(security=1.0, accuracy=0.8, goal_accuracy=0.9)),
        ("b", _reward(security=1.0, accuracy=0.4, goal_accuracy=None)),
        ("c", _reward(security=0.5, accuracy=0.2)),  # the other arm never ran c
        ("unknown", _reward(security=0.0, accuracy=0.0)),  # no resolvable case id
    ]
    loose = [
        ("a", _reward(security=1.0, accuracy=0.6, goal_accuracy=None)),
        ("b", _reward(security=1.0, goal_accuracy=0.7)),  # accuracy: no number and no N/A marker
    ]

    paired = pair_metric_scores(
        full,
        loose,
        arms=("with_skill", "loose_parts"),
        metrics=["security", "accuracy", "goal_accuracy"],
        expected_case_ids=["a", "b", "c", "d"],
    )

    assert paired.treatment == {"security": 1.0, "accuracy": 0.8}
    assert paired.control == {"security": 1.0, "accuracy": 0.6}
    # goal_accuracy was scored by both arms, never on the same case, and only N/A kept them apart.
    assert paired.not_comparable == ("goal_accuracy",)
    basis = paired.basis
    assert basis["metrics"]["security"] == {"cases": 2, "trials": {"with_skill": 2, "loose_parts": 2}}
    assert basis["metrics"]["goal_accuracy"] == {"cases": 0, "trials": {"with_skill": 0, "loose_parts": 0}}
    assert basis["excluded_metrics"] == [
        {"metric": "goal_accuracy", "reason": "no_shared_case", "arms": ["with_skill", "loose_parts"]}
    ]
    every = ["security", "accuracy", "goal_accuracy"]
    assert basis["excluded_cases"] == [
        {"case": "a", "reason": "not_applicable", "arms": ["loose_parts"], "metrics": ["goal_accuracy"]},
        {"case": "b", "reason": "no_value", "arms": ["loose_parts"], "metrics": ["accuracy"]},
        {"case": "b", "reason": "not_applicable", "arms": ["with_skill"], "metrics": ["goal_accuracy"]},
        {"case": "c", "reason": "not_scored", "arms": ["loose_parts"], "metrics": every},
        {"case": "d", "reason": "not_scored", "arms": ["with_skill", "loose_parts"], "metrics": every},
        {
            "case": None,
            "reason": "unidentified_case",
            "arms": ["with_skill"],
            "metrics": every,
            "trials": {"with_skill": 1},
        },
    ]
    assert (basis["excluded_case_entries"], basis["excluded_cases_truncated"]) == (6, False)


def test_a_long_exclusion_list_is_capped_so_the_report_can_still_load_the_file() -> None:
    from skillevaluator.tier3.harbor.lift_basis import MAX_EXCLUDED_CASE_ENTRIES, pair_metric_scores

    many = [f"case-{index:04d}" for index in range(MAX_EXCLUDED_CASE_ENTRIES + 10)]
    paired = pair_metric_scores(
        [("kept", _reward(security=1.0)), *((case, _reward(security=1.0)) for case in many)],
        [("kept", _reward(security=0.5))],
        arms=("with_skill", "without_skill"),
        metrics=["security"],
    )

    assert paired.treatment == {"security": 1.0} and paired.basis["metrics"]["security"]["cases"] == 1
    assert len(paired.basis["excluded_cases"]) == MAX_EXCLUDED_CASE_ENTRIES
    assert paired.basis["excluded_case_entries"] == len(many)
    assert paired.basis["excluded_cases_truncated"] is True


def _judged(case_id: str, accuracy: float, judge_model: str) -> dict:
    reward = {"entry_id": case_id, **_reward(**dict.fromkeys(DEFAULT_METRICS, 0.5)), "accuracy": accuracy}
    reward["details"] = {"accuracy": {"score": accuracy, "judge_model_used": judge_model}}
    return reward


def test_paired_lift_leaves_out_cases_whose_arms_used_different_judges() -> None:
    from skillevaluator.tier3.harbor.collector import _CollectedArm, _paired_lift

    treatment = _CollectedArm(
        execution={},
        logical_rewards=[_judged("case-1", 1.0, "claude-opus-4-8"), _judged("case-2", 0.8, "judge")],
    )
    control = _CollectedArm(
        execution={},
        logical_rewards=[_judged("case-1", 0.0, "judge"), _judged("case-2", 0.4, "judge")],
    )

    lift = _paired_lift(
        treatment, control, arms=("with_skill", "without_skill"), expected_case_ids=["case-1", "case-2"]
    )

    # Accuracy compares case-2 only; every other metric keeps both cases.
    assert lift["accuracy"] == {"with_skill": 0.8, "without_skill": 0.4, "delta": 0.4, "direction": "up"}
    assert lift["judge_mixed_cases"] == {"accuracy": 1}
    assert lift["overall"]["delta"] == round(0.4 / len(DEFAULT_METRICS), 4)
    basis = lift["basis"]
    assert basis["metrics"]["accuracy"]["cases"] == 1
    assert basis["metrics"]["security"]["cases"] == 2
    assert basis["excluded_cases"] == [
        {"case": "case-1", "reason": "judge_mixed", "arms": ["with_skill", "without_skill"], "metrics": ["accuracy"]}
    ]


def test_paired_lift_without_a_judge_fallback_reports_no_judge_mixed_cases() -> None:
    from skillevaluator.tier3.harbor.collector import _CollectedArm, _paired_lift

    treatment = _CollectedArm(execution={}, logical_rewards=[_judged("case-1", 1.0, "judge")])
    control = _CollectedArm(execution={}, logical_rewards=[_judged("case-1", 0.0, "judge")])

    lift = _paired_lift(treatment, control, arms=("with_skill", "without_skill"), expected_case_ids=["case-1"])

    assert "judge_mixed_cases" not in lift
    assert lift["accuracy"]["delta"] == 1.0
    assert lift["basis"]["excluded_cases"] == []
