# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token usage per case, agent and run: measured, never estimated and never scored."""

from __future__ import annotations

import json
from typing import Any

import pytest

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
        "unscored_attempts": unscored,
        "unscored_attempts_baseline": unscored_baseline,
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


def test_an_attempt_that_never_ran_leaves_the_arm_partial() -> None:
    agent = _agent([_trial("case-1", 100, 20)], [_trial("case-1", 90, 10)])
    agent["conditions"]["with_skill"]["expected_attempts"] = 2
    agent["pass_at_k"] = {"with_skill": {"k": 2, "cases": {"case-1": {"attempts_skipped": 0}}}}

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


def test_a_recovered_timeout_attempt_is_expected_usage() -> None:
    history = {"entry_id": "case-1", "trial_id": "case-1__x", "tokens": {"prompt": 10, "completion": 1}}
    agent = _agent([_trial("case-1", 100, 20)], [_trial("case-1", 90, 10)])
    agent["unscored_attempts"] = [history]
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


def test_absent_baseline_evidence_does_not_claim_the_baseline_never_ran() -> None:
    agent = _agent([_trial("case-1", 100, 20)], [])
    for record in ({}, {"execution_status": "succeeded"}, {"scored_attempts": 1}):
        agent["conditions"]["without_skill"] = record
        usage = build_token_usage({"codex": agent})
        assert usage is not None
        baseline = usage["agents"]["codex"]["baseline"]
        assert baseline["coverage"]["status"] == "unavailable"
        assert "no_attempts_recorded" in baseline["coverage"]["reason_codes"]
        assert "baseline_not_run" not in usage["agents"]["codex"]["delta"]["reason_codes"]


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
        assert delta["reason"] == "Baseline incomplete (1 of 2 attempts scored)."


def test_no_measured_attempt_gives_no_token_usage() -> None:
    bare = {"entry_id": "case-1", "trial_id": "case-1__a"}
    assert build_token_usage({"codex": _agent([bare], [bare])}) is None
    assert build_token_usage({}) is None
