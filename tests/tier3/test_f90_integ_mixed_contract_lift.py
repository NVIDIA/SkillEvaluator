# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared lift basis (proof H5, M10) and Harbor's mixed metric contracts.

A native multi-step task can write one standard reward row and one
custom-only row for the same attempt. The attempt's score is then its
logical overall (the mean of both rows), which pass@k and the report
headline use. The paired lift statistics and the pass@k lift must use it
too: comparing only the standard rows' dimensions would drop the custom
step and disagree with the headline.
"""

from __future__ import annotations

from skillevaluator.tier3.harbor.collector import _logical_attempt_rewards, _shared_metric_pass_lift
from skillevaluator.tier3.harbor.metrics import CUSTOM_ONLY_METRIC_SET, DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.harbor.stats import TrialObservation, shared_case_scores


def _mixed_attempt(arm: str, standard: float, custom: float) -> list[dict[str, object]]:
    root = f"case-001__{arm}"
    return [
        {
            "entry_id": "case-001",
            "_trial_root_name": root,
            "_step_name": "step-1",
            "metric_set": DEFAULT_METRIC_SET,
            **dict.fromkeys(DEFAULT_METRICS, standard),
        },
        {
            "entry_id": "case-001",
            "_trial_root_name": root,
            "_step_name": "step-2",
            "metric_set": CUSTOM_ONLY_METRIC_SET,
            "overall": custom,
            "domain_quality": custom,
        },
    ]


def test_mixed_contract_attempts_pair_on_their_logical_overall() -> None:
    treatment = TrialObservation(case_id="case-001", reward={}, logical_overall=0.6)
    control = TrialObservation(case_id="case-001", reward={}, logical_overall=0.4)

    treatment_scores, control_scores, dimensions = shared_case_scores([treatment], [control])

    assert (treatment_scores, control_scores, dimensions) == ({"case-001": 0.6}, {"case-001": 0.4}, ["overall"])


def test_an_overall_only_case_compares_with_the_other_arms_dimension_mean() -> None:
    treatment = TrialObservation(case_id="case-001", reward={}, logical_overall=0.6)
    control = TrialObservation(
        case_id="case-001",
        reward={"entry_id": "case-001", "metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, 0.4)},
    )

    treatment_scores, control_scores, dimensions = shared_case_scores([treatment], [control])

    assert dimensions == ["overall"]
    assert treatment_scores == {"case-001": 0.6}
    assert round(control_scores["case-001"], 4) == 0.4


def test_pass_lift_scores_mixed_contract_attempts_by_their_logical_overall() -> None:
    with_rewards = _logical_attempt_rewards(_mixed_attempt("with", 1.0, 0.2))
    without_rewards = _logical_attempt_rewards(_mixed_attempt("without", 0.6, 0.2))

    lift = _shared_metric_pass_lift(
        with_rewards,
        without_rewards,
        n_attempts=1,
        pass_threshold=0.5,
        stop_on_pass=False,
        expected_cases=1,
        expected_case_ids=["case-001"],
    )

    # Logical overalls 0.6 and 0.4 against a 0.5 threshold: only the plugin arm passes. Scoring the
    # standard rows alone (1.0 and 0.6) would pass both and hide the custom step.
    assert (lift["with_skill_passed_cases"], lift["without_skill_passed_cases"]) == (1, 0)
