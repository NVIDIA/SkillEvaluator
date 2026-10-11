# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lift ladder: four arms, one common basis, four steps that add up.

no plugin -> skills only -> loose parts -> full plugin. The steps are
``skill`` (skills only minus no plugin), ``capability`` (loose parts minus
skills only), ``integration`` (full plugin minus loose parts, the glue) and
``plugin`` (full plugin minus no plugin). On one shared per-case basis
across all four arms they add up per case, on the means, and in every
bootstrap resample. The glue verdict reads cross-component cases only, on a
second basis over the three skill arms that keeps all five dimensions.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Any

import pytest

from skillevaluator.constants import LIFT_BOOTSTRAP_SEED, LIFT_CI_MIN_PAIRED_CASES
from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.harbor.stats import ArmObservations, TrialObservation

NONE, SKILLS, LOOSE, FULL = "without_skill", "sum_of_parts", "loose_parts", "with_skill"
LADDER_ARMS = (NONE, SKILLS, LOOSE, FULL)
STEP_ARMS = {
    "skill": (SKILLS, NONE),
    "capability": (LOOSE, SKILLS),
    "integration": (FULL, LOOSE),
    "plugin": (FULL, NONE),
}
STEP_FIELDS = {"estimate", "ci_low", "ci_high", "n_cases", "precision"}


def _reward(case_id: str, value: float | None, **overrides: float | None) -> dict[str, Any]:
    """All six default metrics at *value*; a keyword sets one metric (``None`` = not applicable)."""
    reward: dict[str, Any] = {"entry_id": case_id, "metric_set": DEFAULT_METRIC_SET}
    reward.update(dict.fromkeys(DEFAULT_METRICS, value))
    reward.update(overrides)
    return reward


def _no_skill(case_id: str, value: float) -> dict[str, Any]:
    """The no-plugin arm: nothing to discover, so the skill metrics are N/A."""
    return _reward(case_id, value, skill_execution=None, skill_efficiency=None)


def _arm(trials: list[TrialObservation], status: str = "succeeded", job_failure: str = "") -> ArmObservations:
    return ArmObservations(status, tuple(trials), job_failure=job_failure)


def _flat_arms(cases: list[str], none: float, skills: float, loose: float, full: float) -> dict[str, ArmObservations]:
    return {
        NONE: _arm([TrialObservation(c, _no_skill(c, none)) for c in cases]),
        SKILLS: _arm([TrialObservation(c, _reward(c, skills)) for c in cases]),
        LOOSE: _arm([TrialObservation(c, _reward(c, loose)) for c in cases]),
        FULL: _arm([TrialObservation(c, _reward(c, full)) for c in cases]),
    }


CASES = [f"c-{index}" for index in range(1, 10)]
CROSS = CASES[:5]


# ---------------------------------------------------------------------------
# common_case_scores: one basis for every arm
# ---------------------------------------------------------------------------


def test_common_basis_keeps_only_the_dimensions_every_arm_scored_for_that_case() -> None:
    arms = _flat_arms(["a", "b"], 0.4, 0.6, 0.7, 0.8)
    # Case b: the loose arm has no accuracy score, so correctness drops out of b for every arm.
    arms[LOOSE] = _arm(
        [TrialObservation("a", _reward("a", 0.7)), TrialObservation("b", _reward("b", 0.7, accuracy=None))]
    )
    arms[FULL] = _arm(
        [TrialObservation("a", _reward("a", 0.8)), TrialObservation("b", _reward("b", 0.8, accuracy=0.2))]
    )

    basis = stats.common_case_scores(arms)

    assert basis.case_ids == ("a", "b")
    assert basis.dimensions == ("security", "correctness", "effectiveness")
    assert basis.excluded_cases == ()
    # a: security, correctness, effectiveness all at 0.8. b: only security and effectiveness, so the 0.2 accuracy
    # never counts.
    assert basis.scores[FULL] == {"a": pytest.approx(0.8), "b": pytest.approx(0.8)}
    assert basis.scores[NONE] == {"a": pytest.approx(0.4), "b": pytest.approx(0.4)}
    # Discoverability and efficiency are not in the common basis, even where every arm has them.
    assert "discoverability" not in basis.dimensions and "efficiency" not in basis.dimensions


def test_attempts_are_averaged_inside_a_case_before_the_dimensions() -> None:
    arms = _flat_arms(["a"], 0.4, 0.6, 0.7, 0.8)
    arms[FULL] = _arm([TrialObservation("a", _reward("a", 0.6)), TrialObservation("a", _reward("a", 1.0))])

    basis = stats.common_case_scores(arms)

    assert basis.scores[FULL]["a"] == pytest.approx(0.8)


def test_a_case_one_arm_did_not_score_is_dropped_from_every_arm_and_listed() -> None:
    arms = _flat_arms(CASES, 0.4, 0.6, 0.7, 0.8)
    arms[LOOSE] = _arm([TrialObservation(c, _reward(c, 0.7)) for c in CASES if c != "c-3"])

    basis = stats.common_case_scores(arms)

    assert "c-3" not in basis.case_ids
    assert all("c-3" not in scores for scores in basis.scores.values())
    assert basis.excluded_cases == (
        {"case": "c-3", "reason": "not_scored", "missing_arms": [LOOSE], "load_failures": {}},
    )


def test_a_case_in_one_arm_only_lists_the_three_arms_it_lacks() -> None:
    arms = _flat_arms(["a", "b"], 0.4, 0.6, 0.7, 0.8)
    arms[FULL] = _arm([*arms[FULL].trials, TrialObservation("only-full", _reward("only-full", 0.9))])

    basis = stats.common_case_scores(arms)

    assert basis.case_ids == ("a", "b")
    assert basis.excluded_cases == (
        {"case": "only-full", "reason": "not_scored", "missing_arms": [NONE, SKILLS, LOOSE], "load_failures": {}},
    )


def test_a_case_whose_common_set_is_empty_is_dropped_with_its_own_reason() -> None:
    arms = _flat_arms(["a", "b"], 0.4, 0.6, 0.7, 0.8)
    # Case b: every arm scored it, but the no-plugin arm only on accuracy and the full arm only on security.
    only_accuracy = _reward("b", None, accuracy=0.5)
    only_security = _reward("b", None, security=0.9, skill_execution=0.9)
    arms[NONE] = _arm([TrialObservation("a", _no_skill("a", 0.4)), TrialObservation("b", only_accuracy)])
    arms[FULL] = _arm([TrialObservation("a", _reward("a", 0.8)), TrialObservation("b", only_security)])

    basis = stats.common_case_scores(arms)

    assert basis.case_ids == ("a",)
    assert basis.excluded_cases == (
        {"case": "b", "reason": "no_common_dimension", "missing_arms": [], "load_failures": {}},
    )


def test_an_expected_case_no_arm_scored_is_listed_with_every_arm() -> None:
    arms = _flat_arms(["a"], 0.4, 0.6, 0.7, 0.8)

    basis = stats.common_case_scores(arms, expected_case_ids=["a", "never-ran"])

    assert basis.excluded_cases == (
        {"case": "never-ran", "reason": "not_scored", "missing_arms": list(LADDER_ARMS), "load_failures": {}},
    )


def test_a_load_failure_in_any_arm_drops_the_case_from_every_arm() -> None:
    """A part Claude Code did not report loaded (and never called) voids the case in all four arms."""
    arms = _flat_arms(["a", "b", "c"], 0.4, 0.6, 0.7, 0.8)
    arms[SKILLS] = _arm([TrialObservation(c, _reward(c, 0.6)) for c in ("a", "b")])

    basis = stats.common_case_scores(
        arms,
        load_failures_by_case={
            "a": {LOOSE: "rule:release-policy.md not in the startup report and never read"},
            "c": {FULL: "mcp:reltools not loaded"},
        },
    )

    assert basis.case_ids == ("b",)
    assert basis.excluded_cases == (
        {
            "case": "a",
            "reason": "load_failed",
            "missing_arms": [],
            "load_failures": {LOOSE: "rule:release-policy.md not in the startup report and never read"},
        },
        # Not scored by the skills arm AND a load failure: load failure is the reason, both are listed.
        {
            "case": "c",
            "reason": "load_failed",
            "missing_arms": [SKILLS],
            "load_failures": {FULL: "mcp:reltools not loaded"},
        },
    )


def test_an_empty_load_failure_map_changes_nothing() -> None:
    arms = _flat_arms(CASES, 0.4, 0.6, 0.7, 0.8)
    assert stats.common_case_scores(arms, load_failures_by_case={}) == stats.common_case_scores(arms)
    assert stats.common_case_scores(arms, load_failures_by_case={"c-1": {}}) == stats.common_case_scores(arms)


def test_a_custom_basis_keeps_discoverability_and_efficiency() -> None:
    arms = {arm: obs for arm, obs in _flat_arms(["a"], 0.4, 0.6, 0.7, 0.8).items() if arm != NONE}
    arms[FULL] = _arm([TrialObservation("a", _reward("a", 0.8, skill_execution=1.0))])

    basis = stats.common_case_scores(arms, dimensions=stats.GLUE_BASIS_DIMENSIONS)

    assert basis.dimensions == ("security", "correctness", "discoverability", "effectiveness", "efficiency")
    assert basis.scores[FULL]["a"] == pytest.approx((0.8 * 4 + 1.0) / 5)


# ---------------------------------------------------------------------------
# lift_ladder: the block
# ---------------------------------------------------------------------------


def test_a_clean_four_arm_run_gives_every_step_and_the_bundle() -> None:
    block = stats.lift_ladder(_flat_arms(CASES, 0.4, 0.6, 0.7, 0.8), CROSS, expected_case_ids=CASES)

    assert block["schema_version"] == "1.0"
    assert block["advisory"] is True and block["report_only"] is True
    assert block["arms"] == {"none": NONE, "skills": SKILLS, "loose": LOOSE, "full": FULL}
    assert block["basis"] == "common_dimensions_all_arms"
    assert block["dimensions"] == ["security", "correctness", "effectiveness"]
    assert (block["n_cases"], block["expected_cases"], block["partial"]) == (9, 9, False)
    assert block["excluded_cases"] == [] and block["failed_arms"] == []
    assert set(block["steps"]) == {"skill", "capability", "integration", "plugin"}
    for step in (*block["steps"].values(), block["bundle"]):
        assert set(step) == STEP_FIELDS
        assert step["n_cases"] == 9
    estimates = {name: step["estimate"] for name, step in block["steps"].items()}
    assert estimates == pytest.approx({"skill": 0.2, "capability": 0.1, "integration": 0.1, "plugin": 0.4})
    assert block["bundle"]["estimate"] == pytest.approx(0.2)
    assert block["bootstrap"]["seed"] == LIFT_BOOTSTRAP_SEED
    assert block["bootstrap"]["method"] == "paired_case_bootstrap"
    assert block["bootstrap"]["interval"] == "expanded_percentile"


def test_case_type_split_has_the_same_steps_on_each_group() -> None:
    block = stats.lift_ladder(_flat_arms(CASES, 0.4, 0.6, 0.7, 0.8), CROSS)

    split = block["by_case_type"]
    assert set(split) == {"cross_component", "other"}
    assert split["cross_component"]["n_cases"] == 5 and split["other"]["n_cases"] == 4
    for group in split.values():
        assert set(group["steps"]) == {"skill", "capability", "integration", "plugin"}
        assert group["bundle"]["estimate"] == pytest.approx(0.2)


def test_glue_reads_cross_component_cases_on_the_five_dimension_basis() -> None:
    """Glue that only shows in routing (discoverability) is 0 on the common basis but not in the glue verdict."""
    arms = _flat_arms(CASES, 0.4, 0.6, 0.7, 0.7)
    arms[FULL] = _arm([TrialObservation(c, _reward(c, 0.7, skill_execution=1.0)) for c in CASES])

    block = stats.lift_ladder(arms, CROSS)

    assert block["steps"]["integration"]["estimate"] == pytest.approx(0.0)
    glue = block["glue"]
    assert glue["case_type"] == "cross_component"
    assert glue["basis"] == "common_dimensions_skill_arms"
    assert glue["arms"] == [SKILLS, LOOSE, FULL]
    assert glue["basis_dimensions"] == ["security", "correctness", "discoverability", "effectiveness", "efficiency"]
    assert glue["n"] == 5
    assert glue["lift"] == pytest.approx(0.3 / 5)
    assert glue["ci_low"] == pytest.approx(0.06) and glue["ci_high"] == pytest.approx(0.06)
    assert glue["partial"] is False and glue["reason"] is None and glue["excluded_cases"] == []


def test_fewer_than_five_cross_component_pairs_give_no_usable_glue_interval() -> None:
    block = stats.lift_ladder(_flat_arms(CASES, 0.4, 0.6, 0.7, 0.8), CASES[:4])

    glue = block["glue"]
    assert glue["n"] == 4 < LIFT_CI_MIN_PAIRED_CASES
    assert glue["lift"] == pytest.approx(0.1)
    assert glue["precision"] == "insufficient"
    # No interval below 5 pairs, as for the Bundle lift's case-type split.
    assert (glue["ci_low"], glue["ci_high"]) == (None, None)
    group = block["by_case_type"]["cross_component"]
    assert group["n_cases"] == 4
    assert all((step["ci_low"], step["ci_high"]) == (None, None) for step in group["steps"].values())
    assert group["steps"]["skill"]["estimate"] == pytest.approx(0.2)
    # The whole ladder (9 cases) keeps its intervals.
    assert block["steps"]["skill"]["ci_low"] is not None
    assert glue["reason"] == "Only 4 paired cross-component case(s); at least 5 are needed for an interval."


def test_no_cross_component_case_leaves_the_glue_unmeasured() -> None:
    block = stats.lift_ladder(_flat_arms(CASES, 0.4, 0.6, 0.7, 0.8), [])

    assert block["by_case_type"]["cross_component"]["n_cases"] == 0
    assert all(step["estimate"] is None for step in block["by_case_type"]["cross_component"]["steps"].values())
    assert block["by_case_type"]["other"]["n_cases"] == 9
    glue = block["glue"]
    assert (glue["n"], glue["lift"], glue["ci_low"], glue["ci_high"]) == (0, None, None, None)
    assert glue["reason"] == "No cross-component case was scored in every arm."


def test_unrecorded_case_types_record_no_split_and_no_glue() -> None:
    block = stats.lift_ladder(_flat_arms(CASES, 0.4, 0.6, 0.7, 0.8), None)

    assert block["by_case_type"] is None
    assert block["glue"]["lift"] is None
    assert block["glue"]["reason"] == "The run recorded no case types, so no case is known to be cross-component."
    assert block["steps"]["integration"]["estimate"] == pytest.approx(0.1)


def test_all_cases_cross_component_makes_the_cross_group_the_whole_ladder() -> None:
    block = stats.lift_ladder(_flat_arms(CASES, 0.4, 0.6, 0.7, 0.8), CASES)

    assert block["by_case_type"]["cross_component"]["steps"] == block["steps"]
    assert block["by_case_type"]["other"]["n_cases"] == 0
    assert block["glue"]["n"] == 9


def test_an_excluded_case_is_out_of_every_step_and_the_glue() -> None:
    arms = _flat_arms(CASES, 0.4, 0.6, 0.7, 0.8)
    # c-2 is cross-component; the no-plugin arm never scored it. The glue does not use that arm, but the case
    # is still dropped everywhere, so the glue and the steps describe one case set.
    arms[NONE] = _arm([TrialObservation(c, _no_skill(c, 0.4)) for c in CASES if c != "c-2"])

    block = stats.lift_ladder(arms, CROSS, expected_case_ids=CASES)

    assert block["n_cases"] == 8 and block["partial"] is True
    assert {step["n_cases"] for step in block["steps"].values()} == {8}
    assert block["excluded_cases"] == [
        {"case": "c-2", "reason": "not_scored", "missing_arms": [NONE], "load_failures": {}}
    ]
    assert block["glue"]["n"] == 4
    assert block["glue"]["partial"] is True
    assert block["glue"]["excluded_cases"] == ["c-2"]


def test_load_failures_exclude_the_case_from_the_ladder_and_the_glue() -> None:
    block = stats.lift_ladder(
        _flat_arms(CASES, 0.4, 0.6, 0.7, 0.8),
        CROSS,
        load_failures_by_case={"c-1": {LOOSE: "rule not loaded"}},
    )

    assert block["n_cases"] == 8 and block["partial"] is True
    assert block["excluded_cases"][0]["reason"] == "load_failed"
    assert block["glue"]["n"] == 4 and block["glue"]["excluded_cases"] == ["c-1"]


def test_a_failed_arm_marks_the_ladder_partial_and_names_the_arm() -> None:
    arms = _flat_arms(CASES, 0.4, 0.6, 0.7, 0.8)
    arms[LOOSE] = _arm(list(arms[LOOSE].trials), status="failed", job_failure="looseparts job failed")

    block = stats.lift_ladder(arms, CROSS)

    assert block["partial"] is True
    assert block["failed_arms"] == [LOOSE]
    assert block["glue"]["partial"] is True
    # The scored cases still count: a failed job keeps the cases its trials scored.
    assert block["n_cases"] == 9


def test_a_missing_arm_leaves_no_case_and_is_named() -> None:
    arms = _flat_arms(CASES, 0.4, 0.6, 0.7, 0.8)
    del arms[LOOSE]

    block = stats.lift_ladder(arms, CROSS, expected_case_ids=CASES)

    assert block["n_cases"] == 0 and block["partial"] is True
    assert block["failed_arms"] == [LOOSE]
    assert all(step["estimate"] is None for step in block["steps"].values())
    assert {entry["reason"] for entry in block["excluded_cases"]} == {"not_scored"}
    assert block["excluded_cases"][0]["missing_arms"] == [LOOSE]


# ---------------------------------------------------------------------------
# Property: the steps add up, per case, on the means and in every resample
# ---------------------------------------------------------------------------


def _random_reward(case_id: str, rng: random.Random, *, skill: bool) -> dict[str, Any]:
    reward: dict[str, Any] = {"entry_id": case_id, "metric_set": DEFAULT_METRIC_SET}
    for metric in DEFAULT_METRICS:
        if rng.random() < 0.08 or (not skill and metric in ("skill_execution", "skill_efficiency")):
            reward[metric] = None
        else:
            reward[metric] = round(rng.random(), 3)
    return reward


def _random_ladder_arms(rng: random.Random) -> tuple[dict[str, ArmObservations], set[str], list[str]]:
    """Four random arms: 0-24 cases, 1-3 attempts, some cases missing from an arm, some N/A metrics."""
    cases = [f"case-{index}" for index in range(rng.randint(0, 24))]
    trials: dict[str, list[TrialObservation]] = {arm: [] for arm in LADDER_ARMS}
    for case in cases:
        for arm in LADDER_ARMS:
            if rng.random() < 0.06:
                continue  # this arm lost the case
            for _ in range(rng.randint(1, 3)):
                trials[arm].append(TrialObservation(case, _random_reward(case, rng, skill=arm != NONE)))
    share = rng.choice((0.0, 0.3, 0.6, 1.0))
    cross = {case for case in cases if rng.random() < share} | {"not-a-case"}
    return {arm: _arm(observed) for arm, observed in trials.items()}, cross, cases


def test_steps_add_up_per_case_on_the_means_and_in_every_resample_on_random_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property, 80 random runs.

    Every step is bootstrapped over the same cases with the same seed, so
    ``rng.choices`` draws the same case indices for each step; the per-case
    deltas telescope, so every resample mean adds up too. The rounded
    estimates add up within 0.0002 (four values rounded to 4 places).
    """
    monkeypatch.setattr(stats, "LIFT_BOOTSTRAP_TAIL_RESAMPLES", 5)
    calls: list[tuple[dict[str, float], dict[str, float], dict[str, Any]]] = []
    real_bootstrap = stats.paired_case_bootstrap

    def spy(treatment: Any, control: Any, **kwargs: Any) -> dict[str, Any]:
        result = real_bootstrap(treatment, control, **kwargs)
        calls.append((dict(treatment), dict(control), result))
        return result

    monkeypatch.setattr(stats, "paired_case_bootstrap", spy)
    rng = random.Random(20261010)
    checked = 0
    for _ in range(80):
        arms, cross, cases = _random_ladder_arms(rng)
        calls.clear()
        block = stats.lift_ladder(arms, cross, expected_case_ids=cases)
        basis = stats.common_case_scores(arms, expected_case_ids=cases)
        # The first four bootstrap calls are the four steps over all common cases, in step order.
        steps = dict(zip(("skill", "capability", "integration", "plugin"), calls[:4], strict=True))
        case_ids = set(basis.case_ids)
        for name, (treatment, control, result) in steps.items():
            treatment_arm, control_arm = STEP_ARMS[name]
            # Identical case sets for every step: the common cases, nothing else.
            assert set(treatment) == set(control) == case_ids
            assert treatment == dict(basis.scores[treatment_arm]) and control == dict(basis.scores[control_arm])
            assert result["n_cases"] == len(case_ids) == block["steps"][name]["n_cases"] == block["n_cases"]
            assert result["seed"] == LIFT_BOOTSTRAP_SEED
        assert len({result["resamples"] for _, _, result in steps.values()}) == 1
        # The steps share their arm scores exactly, so per case they telescope.
        for case in case_ids:
            skill, capability = steps["skill"], steps["capability"]
            integration, plugin = steps["integration"], steps["plugin"]
            assert skill[0][case] == capability[1][case]  # skills only
            assert capability[0][case] == integration[1][case]  # loose parts
            assert integration[0][case] == plugin[0][case]  # full plugin
            assert skill[1][case] == plugin[1][case]  # no plugin
            exact = sum(Fraction(t[case]) - Fraction(c[case]) for t, c, _ in (skill, capability, integration))
            assert exact == Fraction(plugin[0][case]) - Fraction(plugin[1][case])
            floats = [t[case] - c[case] for t, c, _ in (skill, capability, integration)]
            assert math.isclose(math.fsum(floats), plugin[0][case] - plugin[1][case], abs_tol=1e-12)
        if not case_ids:
            assert all(step["estimate"] is None for step in block["steps"].values())
            continue
        deltas = {
            name: [treatment[case] - control[case] for case in sorted(case_ids)]
            for name, (treatment, control, _) in steps.items()
        }
        means = {name: math.fsum(values) / len(values) for name, values in deltas.items()}
        assert math.isclose(means["skill"] + means["capability"] + means["integration"], means["plugin"], abs_tol=1e-12)
        estimates = {name: block["steps"][name]["estimate"] for name in steps}
        assert (
            abs(estimates["skill"] + estimates["capability"] + estimates["integration"] - estimates["plugin"]) <= 2e-4
        )
        assert abs(estimates["capability"] + estimates["integration"] - block["bundle"]["estimate"]) <= 2e-4
        # Every resample adds up: the same seed and n draw the same case indices for each step.
        resamples = steps["skill"][2]["resamples"]
        draws = {
            name: stats._resample_means(values, resamples=resamples, seed=LIFT_BOOTSTRAP_SEED)
            for name, values in deltas.items()
        }
        for index in range(resamples):
            total = draws["skill"][index] + draws["capability"][index] + draws["integration"][index]
            assert math.isclose(total, draws["plugin"][index], abs_tol=1e-12)
        # The case-type groups partition the common cases, and add up inside each group too.
        split = block["by_case_type"]
        assert split["cross_component"]["n_cases"] + split["other"]["n_cases"] == len(case_ids)
        for group in split.values():
            group_steps = group["steps"]
            if group["n_cases"]:
                total = sum(group_steps[name]["estimate"] for name in ("skill", "capability", "integration"))
                assert abs(total - group_steps["plugin"]["estimate"]) <= 2e-4
        # The glue reads exactly the cross-component common cases.
        assert block["glue"]["n"] == len(case_ids & cross)
        checked += 1
    assert checked >= 60


def test_resample_means_draw_the_same_indices_whatever_the_values() -> None:
    """The seam the additivity rests on: ``rng.choices`` picks indices from the seed and n alone."""
    first = stats._resample_means([1.0, 2.0, 3.0, 4.0, 5.0], resamples=50, seed=0)
    second = stats._resample_means([10.0, 20.0, 30.0, 40.0, 50.0], resamples=50, seed=0)
    assert [10 * value for value in first] == pytest.approx(second)


# ---------------------------------------------------------------------------
# Integration completeness checks loose parts against the full plugin too
# ---------------------------------------------------------------------------


def _completeness(**kwargs: Any) -> dict[str, Any]:
    arms = _flat_arms(["a", "b"], 0.4, 0.6, 0.7, 0.8)
    params: dict[str, Any] = {
        "expected_case_ids": ["a", "b"],
        "n_attempts": 1,
        "stop_on_pass": False,
        "pass_threshold": 0.5,
    }
    params.update(kwargs)
    return stats.integration_completeness(arms[FULL], arms[SKILLS], **params)


def test_completeness_without_a_loose_arm_is_unchanged() -> None:
    result = _completeness()
    assert "loose_parts" not in result
    assert result == {"complete": True, "missing_cases": [], "failed_arms": [], "attempt_shortfall": []}


def test_completeness_checks_loose_parts_against_the_full_plugin() -> None:
    loose = _arm([TrialObservation("a", _reward("a", 0.7))], status="failed", job_failure="boom")

    result = _completeness(loose_parts_arm=loose)

    # The full-versus-skills check is the same as before.
    assert {key: result[key] for key in ("complete", "missing_cases", "failed_arms", "attempt_shortfall")} == {
        "complete": True,
        "missing_cases": [],
        "failed_arms": [],
        "attempt_shortfall": [],
    }
    assert result["loose_parts"] == {
        "complete": False,
        "missing_cases": ["b"],
        "failed_arms": [LOOSE],
        "attempt_shortfall": [{"case": "b", "arm": LOOSE, "expected": 1, "observed": 0}],
    }


def test_a_complete_loose_arm_is_complete() -> None:
    loose = _arm([TrialObservation(c, _reward(c, 0.7)) for c in ("a", "b")])
    assert _completeness(loose_parts_arm=loose)["loose_parts"]["complete"] is True


# ---------------------------------------------------------------------------
# build_agent_statistics: the block is optional
# ---------------------------------------------------------------------------


def _agent_statistics(arms: dict[str, ArmObservations], **kwargs: Any) -> dict[str, Any]:
    return stats.build_agent_statistics(
        arms,
        expected_case_ids=CASES,
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=True,
        cross_component_case_ids=frozenset(CROSS),
        **kwargs,
    )


def test_statistics_without_loose_observations_carry_no_ladder() -> None:
    arms = _flat_arms(CASES, 0.4, 0.6, 0.7, 0.8)
    del arms[LOOSE]

    result = _agent_statistics(arms)

    assert "lift_ladder" in stats.STATISTICS_BLOCKS
    assert tuple(result) == stats.STATISTICS_BLOCKS
    assert result["lift_ladder"] is None
    assert "loose_parts" not in result["integration_completeness"]


def test_statistics_with_loose_observations_carry_the_ladder_and_its_completeness() -> None:
    arms = _flat_arms(CASES, 0.4, 0.6, 0.7, 0.8)
    failures = {"c-9": {LOOSE: "rule not loaded"}}

    result = _agent_statistics(arms, load_failures_by_case=failures)

    assert tuple(result) == stats.STATISTICS_BLOCKS
    assert result["lift_ladder"] == stats.lift_ladder(
        arms, frozenset(CROSS), expected_case_ids=CASES, load_failures_by_case=failures
    )
    assert result["lift_ladder"]["n_cases"] == 8
    assert result["integration_completeness"]["loose_parts"]["complete"] is True
    # The loose arm gets its own reliability and cost rows like any arm that ran.
    assert LOOSE in result["reliability"] or LOOSE in result["cost"]
    # The existing Bundle lift is untouched by the extra arm.
    without_loose = {arm: obs for arm, obs in arms.items() if arm != LOOSE}
    assert result["lift_uncertainty"] == _agent_statistics(without_loose)["lift_uncertainty"]


def test_statistics_respect_an_unrequested_skills_arm() -> None:
    """The skills-only rung uses the arm only when it was requested, like the Bundle lift."""
    result = stats.build_agent_statistics(
        _flat_arms(CASES, 0.4, 0.6, 0.7, 0.8),
        expected_case_ids=CASES,
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=False,
        cross_component_case_ids=frozenset(CROSS),
    )

    ladder = result["lift_ladder"]
    assert ladder["failed_arms"] == [SKILLS]
    assert ladder["n_cases"] == 0 and ladder["partial"] is True


def test_a_legacy_members_baseline_gets_no_ladder() -> None:
    """Legacy ``--lift-mode integration``: the only baseline holds the member skills, so there is no no-plugin rung."""
    result = _agent_statistics(_flat_arms(CASES, 0.4, 0.6, 0.7, 0.8), baseline_is_sum_of_parts=True)
    assert result["lift_ladder"] is None
