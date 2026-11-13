# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token usage per case, agent and run: measured, never estimated and never scored."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from skillevaluator.tier3.harbor.collector import _pass_summary
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.token_usage import build_token_usage, normalize_token_counts


def _trial(case: str, prompt: int, completion: int, cached: int | None = None) -> dict[str, Any]:
    tokens = {"prompt": prompt, "completion": completion}
    if cached is not None:
        tokens["cached"] = cached
    return {"entry_id": case, "trial_id": f"{case}__a", "tokens": tokens}


def _agent(
    with_skill: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    with_status: str = "succeeded",
    baseline_status: str = "succeeded",
    unscored: list[dict[str, Any]] | None = None,
    unscored_baseline: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    unscored = unscored or []
    unscored_baseline = unscored_baseline or []
    return {
        "trials": with_skill,
        "trials_baseline": baseline,
        "usage_only_attempts": [{**row, "scored": False} for row in unscored],
        "usage_only_attempts_baseline": [{**row, "scored": False} for row in unscored_baseline],
        "conditions": {
            "with_skill": {
                "execution_status": with_status,
                "expected_attempts": len(with_skill) + len(unscored),
                "scored_attempts": len(with_skill),
            },
            "without_skill": {
                "execution_status": baseline_status,
                "expected_attempts": len(baseline) + len(unscored_baseline),
                "scored_attempts": len(baseline),
            },
        },
        "execution_status": "failed" if "failed" in (with_status, baseline_status) else "succeeded",
    }


def _collector_pass_at_k(scored: dict[str, int], *, k: int, planned: dict[str, int] | None = None) -> dict[str, Any]:
    """One arm's pass@k as the collector records it: ``scored`` attempts per case, ``planned`` under a run budget."""
    scores = dict.fromkeys(DEFAULT_METRICS, 0.9)
    rewards = [
        {"entry_id": case, "_trial_name": f"{case}__{index}", **scores}
        for case, count in scored.items()
        for index in range(count)
    ]
    return _pass_summary(
        rewards,
        n_attempts=k,
        pass_threshold=0.5,
        expected_cases=len(scored),
        expected_case_ids=list(scored),
        expected_attempts_by_case=planned,
    )


def _scopes(usage: dict[str, Any], agent: str = "codex") -> list[dict[str, Any]]:
    return [usage["dataset"], usage["agents"][agent], *usage["agents"][agent]["cases"]]


# -- one attempt's counters --------------------------------------------------------


def test_cached_tokens_are_part_of_the_prompt_and_never_added_again() -> None:
    assert normalize_token_counts({"prompt": 100, "completion": 20, "cached": 40}) == {
        "prompt_tokens": 100,
        "cached_tokens": 40,
        "uncached_prompt_tokens": 60,
        "completion_tokens": 20,
        "total_tokens": 120,
        "complete": True,
    }


def test_an_omitted_cache_counter_is_zero_and_atif_names_are_read() -> None:
    normalized = normalize_token_counts({"prompt_tokens": 100, "completion_tokens": 20})
    assert normalized is not None
    assert (normalized["cached_tokens"], normalized["total_tokens"], normalized["complete"]) == (0, 120, True)
    assert normalize_token_counts({}) is None
    assert normalize_token_counts(None) is None


@pytest.mark.parametrize("field", ["prompt", "completion", "cached"])
@pytest.mark.parametrize("value", [-1, True, "100", float("nan"), 1.5, 2**60])
def test_invalid_counters_are_rejected_and_named_without_their_values(field: str, value: object) -> None:
    raw: dict[str, Any] = {"prompt": 100, "completion": 20, "cached": 10, field: value}

    normalized = normalize_token_counts(raw)

    assert normalized is not None
    assert normalized["complete"] is False
    assert normalized["invalid_counters"] == [f"{field}_tokens"]
    assert "100" not in json.dumps(normalized["invalid_counters"])


def test_a_counter_the_trajectory_recorded_invalid_is_rejected_although_tokens_leave_it_out() -> None:
    # A report trial's ``tokens`` leave out an invalid counter; ``invalid_token_counters`` names it.
    normalized = normalize_token_counts({"prompt": 100, "completion": 20}, invalid_counters=["cached_tokens"])
    assert normalized is not None
    assert (normalized["complete"], normalized["cached_tokens"], normalized["invalid_counters"]) == (
        False,
        None,
        ["cached_tokens"],
    )
    without_tokens = normalize_token_counts(None, invalid_counters=["prompt_tokens"])
    assert without_tokens is not None
    assert (without_tokens["complete"], without_tokens["invalid_counters"]) == (False, ["prompt_tokens"])
    # Anything but known counter names is ignored.
    assert normalize_token_counts(None, invalid_counters=["steps", {"x": 1}]) is None

    trial = {**_trial("case-1", 100, 20), "invalid_token_counters": ["cached_tokens"]}
    usage = build_token_usage({"codex": _agent([trial, _trial("case-2", 50, 5)], [_trial("case-1", 90, 10)])})

    assert usage is not None
    with_skill = usage["agents"]["codex"]["with_skill"]
    assert (with_skill["coverage"]["measured_attempts"], with_skill["coverage"]["reason_codes"]) == (
        1,
        ["invalid_token_counters"],
    )
    assert with_skill["totals"]["total_tokens"] == 55


def test_cached_above_prompt_is_invalid() -> None:
    normalized = normalize_token_counts({"prompt": 10, "completion": 5, "cached": 11})
    assert normalized is not None
    assert (normalized["complete"], normalized["invalid_counters"]) == (False, ["cached_tokens"])


# -- sums per case, agent and run -----------------------------------------------------


def test_sums_each_case_each_agent_and_all_agents() -> None:
    usage = build_token_usage(
        {
            "codex": _agent(
                [_trial("case-1", 100, 20, 40), _trial("case-2", 200, 30, 50)],
                [_trial("case-1", 150, 25, 30), _trial("case-2", 250, 35, 40)],
            ),
            "opencode": _agent([_trial("case-1", 80, 10, 20)], [_trial("case-1", 90, 15, 10)]),
        }
    )

    assert usage is not None
    assert usage["scored"] is False
    codex = usage["agents"]["codex"]
    assert codex["with_skill"]["totals"] == {
        "prompt_tokens": 300,
        "cached_tokens": 90,
        "uncached_prompt_tokens": 210,
        "completion_tokens": 50,
        "total_tokens": 350,
    }
    cases = {row["entry_id"]: row for row in codex["cases"]}
    assert cases["case-1"]["with_skill"]["totals"]["total_tokens"] == 120
    assert cases["case-1"]["baseline"]["totals"]["total_tokens"] == 175
    assert (cases["case-1"]["delta"]["total_tokens"], cases["case-1"]["delta"]["total_tokens_change_percent"]) == (
        -55,
        -31.43,
    )
    assert codex["delta"]["total_tokens"] == 350 - 460
    assert usage["dataset"]["with_skill"]["totals"]["total_tokens"] == 440
    assert usage["dataset"]["baseline"]["totals"]["total_tokens"] == 565
    assert usage["dataset"]["delta"]["total_tokens"] == -125
    assert usage["dataset"]["with_skill"]["coverage"]["measured_attempts"] == 3


def test_a_failed_attempt_is_counted_for_usage_but_not_as_a_scored_attempt() -> None:
    failed = {"entry_id": "case-1", "trial_id": "case-1__b", "tokens": {"prompt": 400, "completion": 4000}}
    usage = build_token_usage(
        {
            "codex": _agent(
                [_trial("case-1", 100, 20)],
                [_trial("case-1", 90, 10)],
                unscored=[failed],
            )
        }
    )

    assert usage is not None
    with_skill = usage["agents"]["codex"]["with_skill"]
    assert with_skill["totals"]["total_tokens"] == 120 + 4400
    assert with_skill["coverage"] == {
        "status": "complete",
        "expected_attempts": 2,
        "observed_attempts": 2,
        "measured_attempts": 2,
        "unscored_attempts": 1,
        "reason_codes": [],
    }


def test_a_failed_attempt_with_a_cut_harbor_trial_name_is_matched_to_its_case() -> None:
    case = "a-very-long-case-identifier-that-harbor-cuts-short"
    trial_prefix = case[:32].rstrip("_-")
    failed = {"entry_id": trial_prefix, "trial_id": f"{trial_prefix}__b", "tokens": {"prompt": 50, "completion": 5}}
    usage = build_token_usage({"codex": _agent([_trial(case, 100, 20)], [_trial(case, 90, 10)], unscored=[failed])})

    assert usage is not None
    [row] = usage["agents"]["codex"]["cases"]
    assert row["entry_id"] == case
    assert row["with_skill"]["totals"]["total_tokens"] == 175


def test_a_stop_on_pass_trial_name_is_matched_to_its_case() -> None:
    def timed_out(arm: str) -> dict[str, Any]:
        # Without a Harbor result, the loader keeps the trial name before Harbor's ``__<id>``.
        name = f"demo-codex-{arm}-case-1-attempt001__case-1"
        return {"entry_id": name, "trial_id": f"{name}__a", "tokens": {"prompt": 40, "completion": 4}}

    usage = build_token_usage(
        {
            "codex": _agent(
                [_trial("case-1", 100, 20)],
                [_trial("case-1", 90, 10)],
                unscored=[timed_out("with")],
                unscored_baseline=[timed_out("without")],
            )
        }
    )

    assert usage is not None
    codex = usage["agents"]["codex"]
    assert [row["entry_id"] for row in codex["cases"]] == ["case-1"]
    assert codex["delta"]["status"] == "available"


def test_partial_coverage_sums_only_measured_attempts_and_never_estimates() -> None:
    missing = {"entry_id": "case-2", "trial_id": "case-2__a"}
    invalid = {"entry_id": "case-3", "trial_id": "case-3__a", "tokens": {"prompt": -5, "completion": 9}}
    usage = build_token_usage(
        {
            "codex": _agent(
                [_trial("case-1", 100, 20), missing, invalid],
                [_trial("case-1", 90, 10), _trial("case-2", 90, 10), _trial("case-3", 90, 10)],
            )
        }
    )

    assert usage is not None
    with_skill = usage["agents"]["codex"]["with_skill"]
    assert with_skill["totals"]["total_tokens"] == 120
    assert with_skill["totals"]["prompt_tokens"] == 100
    assert with_skill["coverage"]["status"] == "partial"
    assert (with_skill["coverage"]["measured_attempts"], with_skill["coverage"]["expected_attempts"]) == (1, 3)
    assert with_skill["coverage"]["reason_codes"] == ["invalid_token_counters", "missing_token_counters"]
    for scope in (usage["dataset"], usage["agents"]["codex"]):
        assert scope["with_skill"]["coverage"]["status"] == "partial"
        assert scope["delta"]["status"] == "unavailable"
        assert scope["delta"]["total_tokens"] is None
        assert "with_skill_tokens_incomplete" in scope["delta"]["reason_codes"]
    assert "With-skill run token counts cover 1 of 3 attempts." in usage["agents"]["codex"]["delta"]["reason"]
    cases = {row["entry_id"]: row for row in usage["agents"]["codex"]["cases"]}
    assert cases["case-1"]["delta"]["total_tokens"] == 20
    assert cases["case-2"]["with_skill"]["totals"] is None
    assert cases["case-2"]["with_skill"]["coverage"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "pass_at_k",
    [
        _collector_pass_at_k({"case-1": 1}, k=2),
        # A result file from before the collector counted each case's attempts.
        {"k": 2, "cases": {"case-1": {"attempts_skipped": 0}}},
    ],
    ids=["collector", "k-only"],
)
def test_an_attempt_that_never_ran_leaves_the_arm_partial(pass_at_k: dict[str, Any]) -> None:
    agent = _agent([_trial("case-1", 100, 20)], [_trial("case-1", 90, 10)])
    agent["conditions"]["with_skill"]["expected_attempts"] = 2
    agent["pass_at_k"] = {"with_skill": pass_at_k}

    usage = build_token_usage({"codex": agent})

    assert usage is not None
    coverage = usage["agents"]["codex"]["with_skill"]["coverage"]
    assert (coverage["status"], coverage["observed_attempts"], coverage["reason_codes"]) == (
        "partial",
        1,
        ["missing_attempts"],
    )
    [case] = usage["agents"]["codex"]["cases"]
    assert case["with_skill"]["coverage"]["expected_attempts"] == 2
    assert case["delta"]["total_tokens"] is None


def test_a_planned_case_without_attempts_gets_a_row_with_no_tokens() -> None:
    agent = _agent([_trial("case-1", 100, 20)], [])
    agent["pass_at_k"] = {
        "with_skill": {"k": 1, "cases": {"case-1": {"attempts_skipped": 0}, "case-2": {"attempts_skipped": 0}}}
    }
    agent["conditions"]["without_skill"] = {"execution_status": "skipped", "expected_attempts": 0}

    usage = build_token_usage({"codex": agent})

    assert usage is not None
    cases = {row["entry_id"]: row for row in usage["agents"]["codex"]["cases"]}
    assert cases["case-2"]["with_skill"] == {
        "totals": None,
        "coverage": {
            "status": "unavailable",
            "expected_attempts": 1,
            "observed_attempts": 0,
            "measured_attempts": 0,
            "unscored_attempts": 0,
            "reason_codes": ["missing_attempts"],
        },
    }
    assert cases["case-2"]["baseline"]["coverage"]["status"] == "not_run"


def test_attempts_skipped_after_a_pass_are_not_missing() -> None:
    agent = _agent([_trial("case-1", 100, 20)], [_trial("case-1", 90, 10)])
    agent["pass_at_k"] = {"with_skill": {"k": 3, "cases": {"case-1": {"attempts_skipped": 2}}}}

    usage = build_token_usage({"codex": agent})

    assert usage is not None
    [case] = usage["agents"]["codex"]["cases"]
    assert case["with_skill"]["coverage"]["status"] == "complete"
    assert case["delta"]["total_tokens"] == 20


def test_a_case_that_a_run_budget_cut_short_expects_only_the_attempts_it_admitted() -> None:
    # The budget admitted all 3 attempts of case-1 but only 1 of case-2, in both arms.
    trials = [
        {**_trial(case, 100, 10), "trial_id": f"{case}__{index}"}
        for case, count in (("case-1", 3), ("case-2", 1))
        for index in range(count)
    ]
    agent = _agent(trials, copy.deepcopy(trials))
    pass_at_k = _collector_pass_at_k({"case-1": 3, "case-2": 1}, k=3, planned={"case-1": 3, "case-2": 1})
    agent["pass_at_k"] = {"with_skill": pass_at_k, "without_skill": copy.deepcopy(pass_at_k)}

    usage = build_token_usage({"codex": agent})

    assert usage is not None
    cases = {row["entry_id"]: row for row in usage["agents"]["codex"]["cases"]}
    for arm in ("with_skill", "baseline"):
        coverage = cases["case-2"][arm]["coverage"]
        assert (coverage["status"], coverage["expected_attempts"], coverage["reason_codes"]) == ("complete", 1, [])
    assert cases["case-2"]["delta"]["total_tokens"] == 0
    assert usage["agents"]["codex"]["delta"]["status"] == "available"


def test_a_recovered_timeout_attempt_is_expected_usage() -> None:
    history = {"entry_id": "case-1", "trial_id": "case-1__x", "tokens": {"prompt": 10, "completion": 1}}
    agent = _agent([_trial("case-1", 100, 20)], [_trial("case-1", 90, 10)])
    agent["usage_only_attempts"] = [{**history, "scored": False}]
    agent["conditions"]["with_skill"]["recovered_attempts_total"] = 1

    usage = build_token_usage({"codex": agent})

    assert usage is not None
    with_skill = usage["agents"]["codex"]["with_skill"]
    assert (with_skill["coverage"]["status"], with_skill["coverage"]["expected_attempts"]) == ("complete", 2)
    assert with_skill["totals"]["total_tokens"] == 131
    # The arms ran different attempts, so they are not compared.
    assert usage["agents"]["codex"]["delta"]["reason_codes"] == ["case_scope_mismatch"]


# -- when a change is given ---------------------------------------------------------------


@pytest.mark.parametrize(("with_total", "percent"), [(590, -41.0), (1000, 0.0), (1250, 25.0)])
def test_percentage_change_is_given_at_every_scope(with_total: int, percent: float) -> None:
    usage = build_token_usage({"codex": _agent([_trial("case-1", with_total, 0)], [_trial("case-1", 1000, 0)])})

    assert usage is not None
    for scope in _scopes(usage):
        assert scope["delta"] == {
            "status": "available",
            "total_tokens": with_total - 1000,
            "total_tokens_change_percent": percent,
            "reason_codes": [],
            "reason": None,
        }


def test_a_zero_token_baseline_keeps_the_difference_without_a_percentage() -> None:
    usage = build_token_usage({"codex": _agent([_trial("case-1", 10, 0)], [_trial("case-1", 0, 0)])})

    assert usage is not None
    delta = usage["dataset"]["delta"]
    assert (delta["total_tokens"], delta["total_tokens_change_percent"], delta["reason_codes"]) == (
        10,
        None,
        ["zero_baseline"],
    )


def test_a_reused_baseline_is_labelled_and_otherwise_unchanged() -> None:
    agents = {"codex": _agent([_trial("case-1", 100, 10)], [_trial("case-1", 200, 20)])}

    plain = build_token_usage(agents)
    reused = build_token_usage(agents, baseline_reused=True)

    assert plain is not None and reused is not None
    assert "baseline_reused" not in plain
    assert reused["baseline_reused"] is True
    # The reused rows are measurements like any other, so the sums and the change stay.
    assert {key: value for key, value in reused.items() if key != "baseline_reused"} == plain
    assert reused["dataset"]["delta"]["total_tokens"] == -110


def test_a_skipped_baseline_is_not_run_and_has_no_change() -> None:
    agent = _agent([_trial("case-1", 100, 20)], [])
    agent["conditions"]["without_skill"] = {"execution_status": "skipped", "expected_attempts": 0}

    usage = build_token_usage({"codex": agent})

    assert usage is not None
    for scope in _scopes(usage):
        assert scope["baseline"]["coverage"]["status"] == "not_run"
        assert scope["baseline"]["totals"] is None
        assert scope["delta"]["total_tokens"] is None
        assert scope["delta"]["reason_codes"][0] == "baseline_not_run"
    # The skip is the one reason: the arms did not run different cases.
    codex = usage["agents"]["codex"]
    assert (codex["delta"]["reason_codes"], codex["delta"]["reason"]) == (
        ["baseline_not_run"],
        "The baseline did not run.",
    )


@pytest.mark.parametrize("mode", ["both", "with_skill_only"])
def test_a_baseline_without_a_record_or_attempts_did_not_run(mode: str) -> None:
    agent = _agent([_trial("case-1", 100, 20)], [])
    del agent["conditions"]["without_skill"]

    usage = build_token_usage({"codex": agent}, mode=mode)

    assert usage is not None
    for scope in _scopes(usage)[1:]:
        assert scope["baseline"]["coverage"] == {
            "status": "not_run",
            "expected_attempts": 0,
            "observed_attempts": 0,
            "measured_attempts": 0,
            "unscored_attempts": 0,
            "reason_codes": [],
        }
        assert (scope["delta"]["reason_codes"], scope["delta"]["reason"]) == (
            ["baseline_not_run"],
            "The baseline did not run.",
        )
    assert usage["dataset"]["baseline"]["coverage"]["status"] == "not_run"


def test_an_arm_the_run_mode_leaves_out_did_not_run() -> None:
    agent = _agent([_trial("case-1", 100, 20)], [])
    agent["conditions"]["without_skill"] = {"expected_attempts": 1}

    usage = build_token_usage({"codex": agent}, mode="with_skill_only")

    assert usage is not None
    assert usage["agents"]["codex"]["baseline"]["coverage"]["status"] == "not_run"
    assert usage["agents"]["codex"]["delta"]["reason_codes"] == ["baseline_not_run"]
    # In a run of both arms the same arm planned an attempt and recorded none.
    usage = build_token_usage({"codex": agent})
    assert usage is not None
    codex = usage["agents"]["codex"]
    assert codex["baseline"]["coverage"]["status"] == "unavailable"
    assert (codex["delta"]["reason_codes"], codex["delta"]["reason"]) == (
        ["baseline_tokens_incomplete"],
        "No baseline attempts were recorded.",
    )

    with_skill = _agent([], [_trial("case-1", 100, 20)])
    del with_skill["conditions"]["with_skill"]
    usage = build_token_usage({"codex": with_skill}, mode="baseline_only")
    assert usage is not None
    assert usage["agents"]["codex"]["with_skill"]["coverage"]["status"] == "not_run"
    assert usage["agents"]["codex"]["delta"]["reason_codes"] == ["with_skill_not_run"]


def test_absent_baseline_evidence_does_not_claim_the_baseline_never_ran() -> None:
    agent = _agent([_trial("case-1", 100, 20)], [])
    for record in ({}, {"execution_status": "succeeded"}, {"scored_attempts": 1}):
        agent["conditions"]["without_skill"] = record
        usage = build_token_usage({"codex": agent})
        assert usage is not None
        baseline = usage["agents"]["codex"]["baseline"]
        assert baseline["coverage"]["status"] == "unavailable"
        assert "no_attempts_recorded" in baseline["coverage"]["reason_codes"]
        # One reason, not a status or case mismatch on top of it.
        assert (usage["agents"]["codex"]["delta"]["reason_codes"], usage["agents"]["codex"]["delta"]["reason"]) == (
            ["baseline_tokens_incomplete"],
            "No baseline attempts were recorded.",
        )


def test_arms_with_different_attempt_counts_or_cases_are_not_compared() -> None:
    usage = build_token_usage(
        {"codex": _agent([_trial("case-1", 590, 0)], [_trial("case-1", 1000, 0), _trial("case-2", 1000, 0)])}
    )
    assert usage is not None
    assert usage["agents"]["codex"]["delta"]["reason_codes"] == ["case_scope_mismatch"]
    assert usage["dataset"]["delta"]["reason_codes"] == ["agent_scope_mismatch"]

    usage = build_token_usage({"codex": _agent([_trial("case-1", 50, 0)], [_trial("case-2", 100, 0)])})
    assert usage is not None
    assert usage["agents"]["codex"]["delta"]["reason_codes"] == ["case_scope_mismatch"]
    assert usage["dataset"]["delta"]["reason_codes"] == ["agent_scope_mismatch"]


def test_one_agent_without_a_change_blocks_the_change_over_all_agents() -> None:
    usage = build_token_usage(
        {
            "codex": _agent([_trial("case-1", 100, 0)], [_trial("case-1", 200, 0)]),
            "opencode": _agent([_trial("case-1", 100, 0)], [_trial("case-1", 200, 0)], baseline_status="failed"),
        }
    )

    assert usage is not None
    assert usage["agents"]["codex"]["delta"]["total_tokens"] == -100
    assert usage["dataset"]["baseline"]["totals"]["total_tokens"] == 400
    assert usage["dataset"]["delta"]["reason_codes"] == ["agent_scope_mismatch"]


def test_a_failed_baseline_arm_keeps_its_usage_but_gets_no_change() -> None:
    timed_out = {"entry_id": "case-1", "trial_id": "case-1__b", "tokens": {"prompt": 400, "completion": 47522}}
    agent = _agent(
        [_trial("case-1", 100, 20), _trial("case-1", 110, 20)],
        [_trial("case-1", 90, 10)],
        baseline_status="failed",
        unscored_baseline=[timed_out],
    )

    usage = build_token_usage({"claude-code": agent})

    assert usage is not None
    agent_usage = usage["agents"]["claude-code"]
    assert agent_usage["baseline"]["totals"]["total_tokens"] == 100 + 47922
    assert agent_usage["baseline"]["coverage"]["status"] == "complete"
    assert agent_usage["baseline"]["coverage"]["unscored_attempts"] == 1
    for delta in (agent_usage["delta"], *(case["delta"] for case in agent_usage["cases"])):
        assert delta["status"] == "unavailable"
        assert delta["reason_codes"] == ["baseline_failed"]
        assert delta["reason"] == "No change: baseline incomplete (1 of 2 attempts scored)."


@pytest.mark.parametrize("status", ["unknown", "interrupted", None])
def test_an_arm_without_a_known_run_status_gets_no_change(status: str | None) -> None:
    agent = _agent([_trial("case-1", 100, 20)], [_trial("case-1", 500, 20)], baseline_status="succeeded")
    agent["conditions"]["without_skill"]["execution_status"] = status

    usage = build_token_usage({"codex": agent})

    assert usage is not None
    codex = usage["agents"]["codex"]
    # The usage is still measured, but the report withholds this arm's scores.
    assert codex["baseline"]["coverage"]["status"] == "complete"
    for delta in (codex["delta"], *(case["delta"] for case in codex["cases"]), usage["dataset"]["delta"]):
        assert delta["status"] == "unavailable"
        assert delta["total_tokens"] is None
    assert codex["delta"]["reason_codes"] == ["baseline_status_unknown"]
    assert codex["delta"]["reason"] == "No change: baseline status unknown."


def test_a_sum_above_the_exact_json_range_is_not_reported() -> None:
    # Each counter is valid, but each arm's sum is above 2**53 - 1.
    half = 2**52
    usage = build_token_usage(
        {
            "codex": _agent(
                [_trial("case-1", half + 2, 0), _trial("case-1", half + 2, 0)],
                [_trial("case-1", half, 0), _trial("case-1", half, 0)],
            )
        }
    )

    assert usage is not None
    codex = usage["agents"]["codex"]
    for scope in (usage["dataset"], codex, codex["cases"][0]):
        for arm in ("with_skill", "baseline"):
            assert scope[arm]["totals"] is None
            assert scope[arm]["coverage"]["status"] == "unavailable"
            assert "total_out_of_range" in scope[arm]["coverage"]["reason_codes"]
        assert scope["delta"]["status"] == "unavailable"
        assert scope["delta"]["total_tokens"] is None
    assert codex["delta"]["reason_codes"] == ["with_skill_total_out_of_range", "baseline_total_out_of_range"]
    assert "largest integer JSON keeps exact" in codex["delta"]["reason"]


def test_agent_sums_within_range_can_still_pass_it_over_all_agents() -> None:
    half = 2**52
    usage = build_token_usage(
        {
            "codex": _agent([_trial("case-1", half, 0)], [_trial("case-1", half, 0)]),
            "opencode": _agent([_trial("case-1", half, 0)], [_trial("case-1", half, 0)]),
        }
    )

    assert usage is not None
    assert usage["agents"]["codex"]["with_skill"]["totals"]["total_tokens"] == half
    dataset = usage["dataset"]
    assert dataset["with_skill"]["totals"] is None
    assert dataset["with_skill"]["coverage"]["reason_codes"] == ["total_out_of_range"]
    assert "with_skill_total_out_of_range" in dataset["delta"]["reason_codes"]


def test_no_measured_attempt_gives_no_token_usage() -> None:
    bare = {"entry_id": "case-1", "trial_id": "case-1__a"}
    assert build_token_usage({"codex": _agent([bare], [bare])}) is None
    assert build_token_usage({}) is None


# -- case rows are validated against the dataset's canonical IDs (issue #192) ----------

# A real dataset ID that Harbor's 32-character trial name cuts short.
_LONG_CASE = "huggingface-factory-peft-sft-local-return"
_CUT_CASE = _LONG_CASE[:32].rstrip("_-")


def test_a_cut_case_name_never_becomes_a_second_row_beside_its_dataset_case() -> None:
    # The scored arm recorded Harbor's cut trial name; the other arm the full one.
    agent = _agent([_trial(_CUT_CASE, 100, 20)], [_trial(_LONG_CASE, 90, 10)])

    usage = build_token_usage({"codex": agent}, dataset_case_ids=[_LONG_CASE])

    assert usage is not None
    [row] = usage["agents"]["codex"]["cases"]
    assert row["entry_id"] == _LONG_CASE
    assert row["with_skill"]["totals"]["total_tokens"] == 120
    assert row["baseline"]["totals"]["total_tokens"] == 100
    assert usage["case_id_validation"] == {
        "status": "validated",
        "dataset_case_ids": 1,
        "unmatched_case_ids": [],
    }


def test_an_arm_that_died_keeps_its_tokens_on_the_canonical_case() -> None:
    # Both arms were killed before scoring, so only the cut trial name exists.
    def killed(suffix: str) -> dict[str, Any]:
        return {
            "entry_id": _CUT_CASE,
            "trial_id": f"{_CUT_CASE}__{suffix}",
            "tokens": {"prompt": 40, "completion": 4},
        }

    agent = _agent(
        [_trial("case-1", 100, 20)],
        [_trial("case-1", 90, 10)],
        unscored=[killed("a")],
        unscored_baseline=[killed("b")],
    )

    usage = build_token_usage({"codex": agent}, dataset_case_ids=["case-1", _LONG_CASE])

    assert usage is not None
    rows = {row["entry_id"]: row for row in usage["agents"]["codex"]["cases"]}
    assert sorted(rows) == ["case-1", _LONG_CASE]
    assert rows[_LONG_CASE]["with_skill"]["totals"]["total_tokens"] == 44
    assert rows[_LONG_CASE]["baseline"]["totals"]["total_tokens"] == 44
    assert usage["case_id_validation"]["unmatched_case_ids"] == []


def test_every_case_row_uses_a_dataset_id_at_most_once() -> None:
    other = "huggingface-global-negative-serving"
    agent = _agent(
        [_trial(_LONG_CASE, 100, 20), _trial(other[:32].rstrip("_-"), 50, 5)],
        [_trial(_CUT_CASE, 90, 10), _trial(other, 40, 4)],
    )

    usage = build_token_usage({"codex": agent}, dataset_case_ids=[_LONG_CASE, other])

    assert usage is not None
    ids = [row["entry_id"] for row in usage["agents"]["codex"]["cases"]]
    assert ids == sorted({_LONG_CASE, other})
    assert len(ids) == len(set(ids))


def test_a_row_that_no_dataset_id_explains_is_marked_not_rendered_as_a_case() -> None:
    agent = _agent([_trial("case-1", 100, 20)], [_trial("ghost-case", 90, 10)])

    usage = build_token_usage({"codex": agent}, dataset_case_ids=["case-1"])

    assert usage is not None
    rows = {row["entry_id"]: row for row in usage["agents"]["codex"]["cases"]}
    assert rows["case-1"].get("case_id_in_dataset") is None
    assert rows["ghost-case"]["case_id_in_dataset"] is False
    assert usage["case_id_validation"] == {
        "status": "unverified_case_ids",
        "dataset_case_ids": 1,
        "unmatched_case_ids": ["ghost-case"],
    }


def test_two_dataset_ids_with_the_same_cut_name_are_never_guessed() -> None:
    first = "huggingface-factory-peft-sft-local-return"
    second = "huggingface-factory-peft-sft-local-detach"
    assert first[:32] == second[:32]
    agent = _agent([_trial("case-1", 100, 20)], [_trial("case-1", 90, 10)], unscored=[_trial(_CUT_CASE, 40, 4)])

    usage = build_token_usage({"codex": agent}, dataset_case_ids=["case-1", first, second])

    assert usage is not None
    rows = {row["entry_id"]: row for row in usage["agents"]["codex"]["cases"]}
    assert rows[_CUT_CASE]["case_id_in_dataset"] is False
    assert usage["case_id_validation"]["unmatched_case_ids"] == [_CUT_CASE]


def test_an_agent_without_scores_still_gets_canonical_case_rows() -> None:
    # An optional agent the deadline killed: no scored attempt, usage only.
    killed = {"entry_id": _CUT_CASE, "trial_id": f"{_CUT_CASE}__a", "tokens": {"prompt": 40, "completion": 4}}
    claude = {
        "trials": [],
        "trials_baseline": [],
        "usage_only_attempts": [{**killed, "scored": False}],
        "usage_only_attempts_baseline": [],
        "conditions": {
            "with_skill": {"execution_status": "failed", "expected_attempts": 1, "scored_attempts": 0},
            "without_skill": {"execution_status": "failed", "expected_attempts": 1, "scored_attempts": 0},
        },
        "execution_status": "failed",
    }

    usage = build_token_usage(
        {"claude": claude, "codex": _agent([_trial(_LONG_CASE, 100, 20)], [_trial(_LONG_CASE, 90, 10)])},
        dataset_case_ids=[_LONG_CASE],
    )

    assert usage is not None
    assert [row["entry_id"] for row in usage["agents"]["claude"]["cases"]] == [_LONG_CASE]
    assert usage["agents"]["claude"]["baseline"]["coverage"]["status"] == "unavailable"
    assert usage["agents"]["claude"]["delta"]["status"] == "unavailable"
    assert usage["case_id_validation"]["unmatched_case_ids"] == []


def test_without_a_dataset_nothing_is_claimed_about_case_ids() -> None:
    usage = build_token_usage({"codex": _agent([_trial("case-1", 100, 20)], [_trial("case-1", 90, 10)])})

    assert usage is not None
    assert "case_id_validation" not in usage
    assert all("case_id_in_dataset" not in row for row in usage["agents"]["codex"]["cases"])


def test_a_pass_at_k_case_named_by_its_cut_name_is_one_canonical_row() -> None:
    agent = _agent([_trial(_LONG_CASE, 100, 20)], [_trial(_LONG_CASE, 90, 10)])
    agent["pass_at_k"] = {"with_skill": {"k": 2, "cases": {_CUT_CASE: {"attempts_skipped": 0}}}}

    usage = build_token_usage({"codex": agent}, dataset_case_ids=[_LONG_CASE])

    assert usage is not None
    [row] = usage["agents"]["codex"]["cases"]
    assert row["entry_id"] == _LONG_CASE
    assert row["with_skill"]["coverage"]["expected_attempts"] == 2


@pytest.mark.parametrize("cut_first", [True, False], ids=["cut-first", "dataset-id-first"])
def test_two_pass_at_k_keys_for_one_case_merge_the_same_way_in_either_order(cut_first: bool) -> None:
    # Harbor's cut name and the dataset ID canonicalize onto one case. Which
    # record survives must not depend on the order the payload lists them in:
    # the key that is already a dataset ID wins, whichever comes first.
    agent = _agent([_trial(_LONG_CASE, 100, 20)], [_trial(_LONG_CASE, 90, 10)])
    records = [(_CUT_CASE, {"attempts_skipped": 2}), (_LONG_CASE, {"attempts_skipped": 0})]
    agent["pass_at_k"] = {
        "with_skill": {"k": 3, "cases": dict(records if cut_first else reversed(records))},
    }

    usage = build_token_usage({"codex": agent}, dataset_case_ids=[_LONG_CASE])

    assert usage is not None
    [row] = usage["agents"]["codex"]["cases"]
    assert row["entry_id"] == _LONG_CASE
    assert row["with_skill"]["coverage"]["expected_attempts"] == 3
