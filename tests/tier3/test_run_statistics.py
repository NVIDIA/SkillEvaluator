# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only statistics of an ordinary skill run: arm observations and the blocks they feed."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from tests.statistics_run_fixtures import CASES, Trial, graded_baseline, reward, uniform

from skillevaluator.constants import LIFT_CI_LOW_PRECISION_WIDTH, LIFT_CI_MIN_PAIRED_CASES
from skillevaluator.tier3.harbor import collector, run_statistics, stats
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.harbor.run_statistics import ArmInput


def _arm(
    trials: Mapping[str, Sequence[Trial]],
    *,
    status: str = "succeeded",
    n_attempts: int = 1,
    stop_on_pass: bool = False,
) -> ArmInput:
    """Describe an arm the way the collector does."""
    rewards = [
        {
            **reward(case, trial),
            "_trial_root_name": f"{case}__attempt{ordinal}",
            "_trial_name": f"{case}__attempt{ordinal}",
            "_attempt_ordinal": ordinal,
        }
        for case, attempts in trials.items()
        for ordinal, trial in enumerate(attempts, 1)
    ]
    summary = collector._pass_summary(
        rewards,
        n_attempts=n_attempts,
        pass_threshold=0.5,
        stop_on_pass=stop_on_pass,
        expected_cases=len(trials),
        expected_case_ids=list(trials),
        recovered_attempts=[],
    )
    return ArmInput(rewards, status, summary)


def _statistics(arms: Mapping[str, ArmInput], **options: Any) -> dict[str, Any]:
    settings = {"expected_case_ids": CASES, "stop_on_pass": False, "lift_eligible": True, **options}
    return run_statistics.agent_statistics(arms, **settings)


def test_agent_statistics_two_arms_blocks_and_values() -> None:
    statistics = _statistics({"with_skill": _arm(uniform(0.9)), "without_skill": _arm(graded_baseline())})

    assert set(statistics) == set(stats.RUN_STATISTICS_FIELDS)
    assert set(statistics["lift_uncertainty"]) == {"effectiveness"}
    interval = statistics["lift_uncertainty"]["effectiveness"]
    assert interval["n_cases"] == interval["expected_cases"] == 6
    assert interval["precision"] == "low"
    assert interval["ci_low"] < interval["estimate"] < interval["ci_high"]
    assert interval["ci_includes_zero"] is False
    assert interval["partial"] is False
    assert statistics["reliability"]["with_skill"]["pass_hat_k"] == 1.0
    assert statistics["reliability"]["without_skill"]["pass_at_k"] == 0.5
    json.dumps(statistics)


@pytest.mark.parametrize(
    ("cases", "deltas", "expected"),
    [
        (LIFT_CI_MIN_PAIRED_CASES - 1, [0.1], "insufficient"),
        (stats.LIFT_CI_ADEQUATE_MIN_CASES - 4, [0.1], "low"),
        (stats.LIFT_CI_ADEQUATE_MIN_CASES + 2, [0.1], "adequate"),
        (stats.LIFT_CI_ADEQUATE_MIN_CASES + 2, [0.45, -0.4], "low"),
    ],
    ids=["too-few-cases", "few-cases", "narrow-interval", "wide-interval"],
)
def test_precision_labels_follow_case_count_and_width(cases: int, deltas: list[float], expected: str) -> None:
    names = [f"case-{index}" for index in range(1, cases + 1)]
    with_skill = {case: [Trial(0.5 + deltas[index % len(deltas)])] for index, case in enumerate(names)}
    baseline = {case: [Trial(0.5)] for case in names}
    arms = {"with_skill": _arm(with_skill), "without_skill": _arm(baseline)}

    interval = _statistics(arms, expected_case_ids=names)["lift_uncertainty"]["effectiveness"]

    assert interval["precision"] == expected
    assert interval["n_cases"] == cases
    if expected == "adequate":
        assert interval["ci_high"] - interval["ci_low"] <= LIFT_CI_LOW_PRECISION_WIDTH
    if expected == "low" and cases >= stats.LIFT_CI_ADEQUATE_MIN_CASES:
        assert interval["ci_high"] - interval["ci_low"] > LIFT_CI_LOW_PRECISION_WIDTH


def test_failed_arm_withholds_reliability_and_lift_interval() -> None:
    failed = _arm(uniform(0.9), status="failed")

    statistics = _statistics({"with_skill": failed, "without_skill": _arm(uniform(0.5))}, lift_eligible=False)

    assert statistics["lift_uncertainty"] == {"effectiveness": None}
    assert set(statistics["reliability"]) == {"without_skill"}


def test_an_ineligible_lift_has_no_interval_but_keeps_reliability() -> None:
    arms = {"with_skill": _arm(uniform(0.9)), "without_skill": _arm(uniform(0.5))}

    eligible = _statistics(arms)
    ineligible = _statistics(arms, lift_eligible=False)

    assert eligible["lift_uncertainty"]["effectiveness"]["n_cases"] == 6
    assert ineligible["lift_uncertainty"] == {"effectiveness": None}
    assert ineligible["reliability"] == eligible["reliability"]


def test_interval_is_withheld_when_an_arms_cases_made_different_numbers_of_attempts() -> None:
    # After an early stop one case ran three attempts and the rest one. The headline Skill Lift
    # weights every attempt and the interval weights every case, so they would disagree.
    uneven = {**uniform(0.9), "case-1": [Trial(0.1, passed=False), Trial(0.1, passed=False), Trial(0.9)]}
    baseline = _arm(uniform(0.5))

    withheld = _statistics({"with_skill": _arm(uneven, n_attempts=3, stop_on_pass=True), "without_skill": baseline})
    # Each arm has one attempt count of its own, so its mean is the same either way: the interval stays.
    steady = {case: [Trial(0.3, passed=False)] * 3 for case in CASES}
    kept = _statistics({"with_skill": _arm(uniform(0.9)), "without_skill": _arm(steady, n_attempts=3)})

    assert withheld["lift_uncertainty"] == {"effectiveness": None}
    assert kept["lift_uncertainty"]["effectiveness"]["n_cases"] == 6


def test_no_arm_means_no_statistics() -> None:
    assert _statistics({}) == {}


def test_an_attempt_without_an_overall_score_is_left_out_of_the_interval() -> None:
    arm = _arm(uniform(0.9))
    unscored = [{**arm.rewards[0], **dict.fromkeys(DEFAULT_METRICS)}, *arm.rewards[1:]]

    observed = run_statistics.arm_observations(
        ArmInput(unscored, arm.execution_status, arm.pass_summary), expected_case_ids=CASES
    )

    assert [trial.case_id for trial in observed.trials] == CASES[1:]


@pytest.mark.parametrize(("stop_on_pass", "pass_hat_k"), [(False, 1.0), (True, None)])
def test_pass_hat_k_is_null_when_stop_on_pass_truncates_attempts(stop_on_pass: bool, pass_hat_k: float | None) -> None:
    trials = {case: [Trial(0.9), Trial(0.9)] for case in CASES}
    arm = _arm(trials, n_attempts=2, stop_on_pass=stop_on_pass)

    reliability = _statistics({"with_skill": arm}, stop_on_pass=stop_on_pass)["reliability"]["with_skill"]

    assert reliability["pass_hat_k"] == pass_hat_k
    assert reliability["pass_at_k"] == 1.0
    assert reliability["k"] == 2


def test_pass_hat_k_tells_a_flaky_arm_from_a_steady_one() -> None:
    flaky = {case: [Trial(0.9), Trial(0.1, passed=False)] for case in CASES}
    steady = {case: [Trial(0.9), Trial(0.9)] for case in CASES}

    flaky_block = _statistics({"with_skill": _arm(flaky, n_attempts=2)})["reliability"]["with_skill"]
    steady_block = _statistics({"with_skill": _arm(steady, n_attempts=2)})["reliability"]["with_skill"]

    # pass@k cannot tell them apart; pass^k can.
    assert flaky_block["pass_at_k"] == steady_block["pass_at_k"] == 1.0
    assert (flaky_block["pass_hat_k"], steady_block["pass_hat_k"]) == (0.0, 1.0)


def test_attach_does_not_publish_blocks_when_the_file_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_dir = tmp_path / "run" / "codex"
    agent_dir.mkdir(parents=True)
    results = {"agents": {"codex": {}}}

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(collector, "_write_generated_root_json", refuse)

    run_statistics.attach(
        results,
        "codex",
        agent_dir=agent_dir,
        output_dir=tmp_path / "run",
        arms={"with_skill": _arm(uniform(0.9))},
        expected_case_ids=CASES,
        stop_on_pass=False,
        lift_eligible=False,
    )

    assert results["agents"] == {"codex": {}}
    assert results["warnings"] == ["Report-only run statistics were not computed for codex (OSError)."]


def test_the_comparison_trial_limit_applies_to_run_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats, "MAX_COMPARISON_TRIALS", 3)
    arms = {"with_skill": _arm(uniform(0.9)), "without_skill": _arm(uniform(0.5))}

    with pytest.raises(ValueError, match="trial limit"):
        _statistics(arms)
