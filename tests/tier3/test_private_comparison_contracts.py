# SPDX-License-Identifier: Apache-2.0
"""Private comparison contracts; deterministic rewards and staged files only."""

from __future__ import annotations

import math

import pytest

from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS, LEGACY_SCORE_POLICY, overall_score


def reward(case, value=0.5, *, accuracy=None, judge=None):
    row = {"entry_id": case, "metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, value)}
    if accuracy is not None:
        row["accuracy"] = accuracy
    if judge:
        row["details"] = {"accuracy": {"judge_model_used": judge}}
    return row


def arms(cases=("a", "b", "c", "d", "e")):
    return {
        arm: stats.ArmObservations(
            "succeeded", tuple(stats.TrialObservation(case, reward(case, value)) for case in cases)
        )
        for arm, value in (("without_skill", 0.2), ("sum_of_parts", 0.4), ("loose_parts", 0.6), ("with_skill", 0.8))
    }


def test_ladder_uses_one_shared_bootstrap_draw_stream(monkeypatch):
    calls = []
    original = stats.random.Random.choices

    def choices(self, population, *args, **kwargs):
        calls.append(len(population))
        return original(self, population, *args, **kwargs)

    monkeypatch.setattr(stats.random.Random, "choices", choices)
    result = stats.lift_ladder(arms(), None)
    assert len(calls) == result["bootstrap"]["resamples"]
    steps = result["steps"]
    assert math.isclose(
        sum(steps[key]["estimate"] for key in ("skill", "capability", "integration")), steps["plugin"]["estimate"]
    )


def test_judge_mismatch_does_not_enter_paired_interval_or_ladder():
    observed = {}
    for arm in arms():
        high = arm == "with_skill"
        observed[arm] = stats.ArmObservations(
            "succeeded",
            tuple(
                stats.TrialObservation(
                    case, reward(case, accuracy=0.9 if high else 0.1, judge="fallback" if high else "primary")
                )
                for case in ("a", "b", "c", "d", "e")
            ),
        )
    pair = stats.paired_lift(
        observed["with_skill"], observed["without_skill"], treatment_arm="with_skill", control_arm="without_skill"
    )
    assert pair["estimate"] == 0.0
    ladder = stats.lift_ladder(observed, None)
    assert ladder["steps"]["plugin"]["estimate"] == 0.0
    assert ladder["judge_mixed_cases"]["accuracy"] == 5


def test_within_arm_judge_mixture_is_excluded_when_other_arm_has_no_model():
    treatment = stats.ArmObservations(
        "succeeded",
        tuple(stats.TrialObservation("a", reward("a", accuracy=0.9, judge=model)) for model in ("primary", "fallback")),
    )
    control = stats.ArmObservations("succeeded", (stats.TrialObservation("a", reward("a", accuracy=0.1)),))
    pair = stats.paired_lift(treatment, control, treatment_arm="with_skill", control_arm="without_skill")
    assert pair["estimate"] == 0.0


def test_unscored_executed_attempt_participates_in_cost():
    usage = {"prompt_tokens": 10, "completion_tokens": 10, "cost_usd": 0.4}
    unscored = {"prompt_tokens": 20, "completion_tokens": 10, "cost_usd": 0.6}
    arm = stats.ArmObservations(
        "succeeded",
        (stats.TrialObservation("a", reward("a"), usage),),
        {"passed_cases": 1},
        unscored=(stats.TrialObservation("b", {}, unscored),),
    )
    result = stats.build_agent_statistics(
        {"with_skill": arm},
        expected_case_ids=["a", "b"],
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=False,
    )
    assert result["cost"]["with_skill"]["total_tokens"] == 50
    assert result["cost"]["with_skill"]["total_usd"] == 1.0


def test_missing_unscored_counter_cannot_become_a_partial_cost_total():
    arm = stats.ArmObservations(
        "succeeded",
        (stats.TrialObservation("a", reward("a"), {"prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.0}),),
        {"passed_cases": 1},
        unscored=(stats.TrialObservation("b", {}, {}),),
    )
    result = stats.build_agent_statistics(
        {"with_skill": arm},
        expected_case_ids=["a", "b"],
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=False,
    )
    assert result["cost"]["with_skill"]["total_tokens"] is None
    assert result["cost"]["with_skill"]["total_usd"] is None
    assert stats.arm_cost(arm.trials, successes=1)["total_usd"] == 0.0


def test_bounded_or_incomplete_observation_coverage_cannot_claim_total_cost():
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.4, "first_turn_prompt_tokens": 10}
    arm = stats.ArmObservations(
        "succeeded",
        tuple(stats.TrialObservation(case, reward(case), usage) for case in ("a", "b", "c")),
        {"passed_cases": 3},
        observations_complete=False,
    )
    result = stats.build_agent_statistics(
        {"with_skill": arm, "without_skill": arm},
        expected_case_ids=["a", "b", "c"],
        n_attempts=2,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=False,
    )
    assert result["cost"]["with_skill"]["total_tokens"] is None
    assert result["cost"]["with_skill"]["total_usd"] is None
    assert result["token_efficiency"]["with_skill"] is None
    assert result["context_cost_measured"]["partial"] is True


def test_failed_arm_keeps_cost_when_all_executed_observations_were_retained():
    arm = stats.ArmObservations(
        "failed",
        unscored=(stats.TrialObservation("a", {}, {"prompt_tokens": 3, "completion_tokens": 2, "cost_usd": 0.25}),),
        job_failure="setup failed after a recorded call",
    )
    result = stats.build_agent_statistics(
        {"with_skill": arm},
        expected_case_ids=["a"],
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=False,
    )
    assert result["cost"]["with_skill"]["total_tokens"] == 5
    assert result["cost"]["with_skill"]["total_usd"] == 0.25
    assert result["reliability"] == {}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, -1.0, 2.0])
def test_invalid_logical_scores_are_unavailable(bad):
    trial = stats.TrialObservation("a", reward("a"), logical_overall=bad)
    assert stats.case_dimension_means([trial]) == {}


def test_unidentified_cases_do_not_pair():
    result = stats.common_case_scores(arms(("unknown", "")))
    assert not result.case_ids


def test_completeness_rejects_missing_attempts():
    observed = arms(("a",))
    result = stats.integration_completeness(
        observed["with_skill"],
        observed["sum_of_parts"],
        expected_case_ids=["a"],
        n_attempts=2,
        stop_on_pass=False,
        pass_threshold=0.5,
    )
    assert result["complete"] is False
    assert len(result["attempt_shortfall"]) == 2


def test_expected_case_missing_from_both_arms_makes_context_partial():
    usage = {"first_turn_prompt_tokens": 20}
    arm = stats.ArmObservations(
        "succeeded", tuple(stats.TrialObservation(case, reward(case), usage) for case in ("a", "b", "c"))
    )
    result = stats.build_agent_statistics(
        {"with_skill": arm, "without_skill": arm},
        expected_case_ids=["a", "b", "c", "missing"],
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=False,
    )
    assert result["context_cost_measured"]["partial"] is True
    assert result["context_cost_measured"]["missing_cases"] == ["missing"]


def test_judge_exclusion_cannot_remove_security():
    with pytest.raises(ValueError, match="judge"):
        stats.common_case_scores(arms(), judge_mixed_cases={"security": ["a"]})


def test_case_index_reads_each_judge_provenance_once(monkeypatch):
    from skillevaluator.tier3.harbor import collector

    calls = []
    original = collector._reward_judge_models

    def read(row, metric):
        calls.append((id(row), metric))
        return original(row, metric)

    monkeypatch.setattr(collector, "_reward_judge_models", read)
    observed = arms(("a", "b"))
    stats.build_agent_statistics(
        observed,
        expected_case_ids=["a", "b"],
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=True,
        cross_component_case_ids=["a"],
    )
    assert len(calls) == 4 * 2 * 3


def test_pairwise_judges_do_not_inherit_another_arms_fallback():
    observed = {}
    for arm in arms():
        primary = arm in {"with_skill", "without_skill"}
        observed[arm] = stats.ArmObservations(
            "succeeded",
            tuple(
                stats.TrialObservation(
                    case,
                    reward(
                        case, accuracy=0.9 if arm == "with_skill" else 0.1, judge="primary" if primary else "fallback"
                    ),
                )
                for case in ("a", "b", "c", "d", "e")
            ),
        )
    result = stats.build_agent_statistics(
        observed,
        expected_case_ids=["a", "b", "c", "d", "e"],
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=True,
    )
    assert result["lift_uncertainty"]["effectiveness"]["estimate"] > 0
    assert result["lift_ladder"]["steps"]["plugin"]["estimate"] == 0


def test_load_failure_is_excluded_from_each_affected_pair():
    observed = arms(("a", "b"))
    kwargs = {
        "expected_case_ids": ["a", "b"],
        "n_attempts": 1,
        "stop_on_pass": False,
        "pass_threshold": 0.5,
        "sum_of_parts_requested": True,
    }
    result = stats.build_agent_statistics(
        observed, load_failures_by_case={"a": {"with_skill": "MCP did not load"}}, **kwargs
    )
    assert result["lift_uncertainty"]["effectiveness"]["n_cases"] == 1
    assert result["lift_uncertainty"]["integration"]["n_cases"] == 1
    assert result["lift_ladder"]["n_cases"] == 1
    unrelated = stats.build_agent_statistics(
        observed, load_failures_by_case={"a": {"loose_parts": "MCP did not load"}}, **kwargs
    )
    assert unrelated["lift_uncertainty"]["effectiveness"]["n_cases"] == 2
    assert unrelated["lift_ladder"]["n_cases"] == 1


def test_complete_legacy_policy_keeps_its_metric_mean():
    row = reward("a", 0.0)
    row.update(skill_execution=1.0, score_policy=LEGACY_SCORE_POLICY)
    assert stats.trial_quality_score(row) == overall_score(row)


def test_custom_only_scores_keep_the_existing_overall_basis():
    treatment = stats.ArmObservations("succeeded", (stats.TrialObservation("a", {"overall": 0.75}),))
    control = stats.ArmObservations("succeeded", (stats.TrialObservation("a", {"overall": 0.25}),))
    pair = stats.paired_lift(treatment, control, treatment_arm="with_skill", control_arm="without_skill")
    assert pair["dimensions"] == ["overall"]
    assert pair["estimate"] == 0.5


def test_mixed_logical_attempt_retains_the_collectors_overall_score():
    treatment = stats.ArmObservations(
        "succeeded", (stats.TrialObservation("a", reward("a", 0.9), logical_overall=0.7),)
    )
    control = stats.ArmObservations("succeeded", (stats.TrialObservation("a", reward("a", 0.2)),))
    pair = stats.paired_lift(treatment, control, treatment_arm="with_skill", control_arm="without_skill")
    assert pair["treatment_score"] == 0.7
    assert pair["estimate"] == 0.5


def test_unknown_policy_and_unexplained_missing_dimension_are_unscored():
    row = reward("a")
    row["score_policy"] = "unrecognized-policy"
    assert stats.trial_quality_score(row) is None
    row = reward("a")
    del row["accuracy"]
    assert stats.trial_quality_score(row) is None
    row["accuracy"] = None
    row["details"] = {"accuracy": {"status": "not_applicable"}}
    assert stats.trial_quality_score(row) == pytest.approx(overall_score(row), abs=0.0001)


def test_different_declared_policies_are_not_called_one_overall_lift():
    normal = reward("a")
    legacy = {**normal, "score_policy": LEGACY_SCORE_POLICY}
    pair = stats.paired_lift(
        stats.ArmObservations("succeeded", (stats.TrialObservation("a", legacy),)),
        stats.ArmObservations("succeeded", (stats.TrialObservation("a", normal),)),
        treatment_arm="with_skill",
        control_arm="without_skill",
    )
    assert pair is None


def test_common_metric_is_selected_before_dimension_weighting():
    treatment = reward("a", 0.0)
    treatment["goal_accuracy"] = 1.0
    control = reward("a", 0.0)
    control["goal_accuracy"] = None
    control["details"] = {"goal_accuracy": {"status": "not_applicable"}}
    from skillevaluator.tier3.harbor.lift_basis import pair_metric_scores

    expected = pair_metric_scores(
        [("a", treatment)], [("a", control)], arms=("with_skill", "without_skill"), metrics=DEFAULT_METRICS
    )
    assert expected.treatment == expected.control
    paired = stats.paired_lift(
        stats.ArmObservations("succeeded", (stats.TrialObservation("a", treatment),)),
        stats.ArmObservations("succeeded", (stats.TrialObservation("a", control),)),
        treatment_arm="with_skill",
        control_arm="without_skill",
    )
    assert paired["estimate"] == 0.0
    observed = {
        name: stats.ArmObservations(
            "succeeded", (stats.TrialObservation("a", treatment if name == "with_skill" else control),)
        )
        for name in arms()
    }
    ladder = stats.lift_ladder(observed, ["a"])
    assert ladder["steps"]["plugin"]["estimate"] == 0.0
    assert ladder["glue"]["lift"] == 0.0


def test_per_metric_attempt_means_precede_dimension_weighting():
    scored = reward("a", 0.0)
    scored["goal_accuracy"] = 1.0
    not_applicable = {**scored, "goal_accuracy": None, "details": {"goal_accuracy": {"status": "not_applicable"}}}
    treatment = stats.ArmObservations(
        "succeeded", tuple(stats.TrialObservation("a", row) for row in (scored, not_applicable))
    )
    control = stats.ArmObservations("succeeded", tuple(stats.TrialObservation("a", scored) for _ in range(2)))
    pair = stats.paired_lift(treatment, control, treatment_arm="with_skill", control_arm="without_skill")
    assert pair["estimate"] == 0.0


@pytest.mark.parametrize(
    "low,high,verdict",
    [
        (0.05, 0.2, "real_integration"),
        (-0.2, -0.05, "negative_integration"),
        (-0.04, 0.04, "cosmetic_bundling"),
        (-0.05, 0.05, "inconclusive"),
        (0.02, 0.1, "inconclusive"),
    ],
)
def test_integration_claim_uses_the_whole_eligible_interval(low, high, verdict):
    ladder = {
        "partial": False,
        "glue": {
            "n": 5,
            "lift": (low + high) / 2,
            "ci_low": low,
            "ci_high": high,
            "precision": "adequate",
            "partial": False,
        },
    }
    content = {"equal_across_arms": True, "tasks_hashed": 5, "content_basis": "emitted Harbor tasks"}
    complete = {"complete": True, "loose_parts": {"complete": True}, "all_arms": {"complete": True}}
    result = stats.integration_claim(ladder, content, complete, agent="claude-code", full_arm_load="native")
    assert result["verdict"] == verdict
    assert result["eligible"] is (verdict != "inconclusive")
    assert (
        stats.integration_claim(
            ladder, {**content, "equal_across_arms": False}, complete, agent="claude-code", full_arm_load="native"
        )["eligible"]
        is False
    )
    assert (
        stats.integration_claim(
            ladder, content, {**complete, "all_arms": {"complete": False}}, agent="claude-code", full_arm_load="native"
        )["eligible"]
        is False
    )
    wrapper = stats.integration_claim(ladder, content, complete, agent="claude-code", full_arm_load="wrapper")
    assert wrapper["verdict"] is None and "Wrapper effect" in wrapper["label"]


@pytest.mark.parametrize(
    "bad_glue",
    [{"n": 4}, {"n": True}, {"ci_low": float("nan")}, {"ci_high": float("inf")}, {"lift": 2.0}, {"partial": True}],
)
def test_integration_claim_rejects_unusable_or_incomplete_glue(bad_glue):
    ladder = {
        "partial": False,
        "glue": {"n": 5, "lift": 0.15, "ci_low": 0.1, "ci_high": 0.2, "precision": "low", "partial": False, **bad_glue},
    }
    content = {"equal_across_arms": True, "tasks_hashed": 5, "content_basis": "emitted Harbor tasks"}
    complete = {"complete": True, "loose_parts": {"complete": True}, "all_arms": {"complete": True}}
    result = stats.integration_claim(ladder, content, complete, agent="claude-code", full_arm_load="native")
    assert result["eligible"] is False
    assert result["verdict"] == "inconclusive"
