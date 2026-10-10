# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured always-on context cost (check 25): hosted search, failed trials, few pairs, load mode.

The job directories are small synthetic copies of the shapes in the check-25
verification examples: Codex trajectories whose first model call ran a hosted
``web_search_call`` (tier3-06), runs where one trial failed (tier3-04), one-case
runs (tier3-12). Nothing starts Harbor, Docker, or a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.harbor.collector import _trial_usage, collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.harbor.stats import ArmObservations, TrialObservation

CASES = [f"rd-case-{index}" for index in range(1, 7)]


def _reward(case_id: str, score: float = 0.8) -> dict[str, Any]:
    return {"entry_id": case_id, "metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, score)}


def _codex_trajectory(first_prompt: int, *, web_search: bool) -> dict[str, Any]:
    """A Harbor ATIF trajectory shaped like Codex's: one agent step per model call, hosted search as a tool call."""
    calls: list[dict[str, Any]] = []
    if web_search:
        calls.append(
            {
                "tool_call_id": "ws_1",
                "function_name": "web_search_call",
                "arguments": {"action_type": "search", "query": "library docs"},
            }
        )
    calls.append({"tool_call_id": "call_1", "function_name": "exec_command", "arguments": {"cmd": "ls"}})
    return {
        "schema_version": "ATIF-v1.5",
        "steps": [
            {"step_id": 1, "source": "user", "message": "task"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "",
                "llm_call_count": 1,
                "tool_calls": calls,
                "metrics": {"prompt_tokens": first_prompt, "completion_tokens": 30, "cached_tokens": 0},
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "done",
                "llm_call_count": 1,
                "metrics": {"prompt_tokens": first_prompt + 900, "completion_tokens": 20},
            },
        ],
        "final_metrics": {"total_prompt_tokens": 2 * first_prompt, "total_completion_tokens": 50},
    }


def _write_job(jobs: Path, variant: str, prompts: dict[tuple[str, int], tuple[int, bool]]) -> None:
    job_dir = jobs / f"demo-codex-{variant}"
    names: list[str] = []
    for (case_id, attempt), (first_prompt, web_search) in prompts.items():
        name = f"{case_id}__attempt{attempt:03d}"
        names.append(name)
        trial = job_dir / name
        (trial / "verifier").mkdir(parents=True)
        (trial / "agent").mkdir()
        (trial / "verifier" / "reward.json").write_text(json.dumps(_reward(case_id)), encoding="utf-8")
        trajectory = _codex_trajectory(first_prompt, web_search=web_search)
        (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
        (trial / "result.json").write_text(json.dumps({"trial_name": name, "task_name": case_id}), encoding="utf-8")
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(names),
                "stats": {
                    "n_trials": len(names),
                    "n_errors": 0,
                    "evals": {
                        "codex__model": {
                            "n_trials": len(names),
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.8": names}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _collect(tmp_path: Path) -> dict[str, Any]:
    result = collect_harbor_results(
        skill_name="demo",
        agents=["codex"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        n_attempts=2,
        expected_cases=len(CASES),
        expected_case_ids=CASES,
        expected_trials=2 * len(CASES),
    )
    return result["agents"]["codex"]["context_cost_measured"]


# --------------------------------------------------------------------------- #
# M33: Codex hosted web search in the first model call                       #
# --------------------------------------------------------------------------- #
def test_m33_first_call_with_hosted_web_search_is_not_a_first_turn_cost(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    trial = job_dir / "case-1__attempt001" / "agent"
    trial.mkdir(parents=True)
    (trial / "trajectory.json").write_text(json.dumps(_codex_trajectory(19_260, web_search=True)), encoding="utf-8")

    usage = _trial_usage(job_dir, {"_trial_root_name": "case-1__attempt001"})

    assert "first_turn_prompt_tokens" not in usage
    assert usage["first_turn_hosted_search"] == 1


def test_m33_codex_web_search_pairs_are_dropped_and_clean_pairs_measured(tmp_path: Path) -> None:
    """tier3-06 C native shape: the baseline searches on its first call; clean pairs give the real +593."""
    jobs = tmp_path / "jobs"
    with_prompts = {(case, attempt): (13_730, False) for case in CASES for attempt in (1, 2)}
    with_prompts[(CASES[0], 1)] = (25_000, True)
    # Attempt 1 of every baseline case searched the web inside its first model call.
    without_prompts = {
        (case, attempt): (21_000, True) if attempt == 1 else (13_137, False) for case in CASES for attempt in (1, 2)
    }
    _write_job(jobs, "with", with_prompts)
    _write_job(jobs, "without", without_prompts)

    measured = _collect(tmp_path)

    assert measured["status"] == "measured"
    assert measured["delta_tokens_mean"] == 593.0
    assert measured["n_pairs"] == 6
    assert measured["excluded"]["hosted_search"] == 7


def test_m33_one_clean_pair_is_reported_but_not_called_measured(tmp_path: Path) -> None:
    """tier3-06 C native: 10 of 12 baseline first calls searched; PR #28 printed -5,699 as the plugin's cost."""
    jobs = tmp_path / "jobs"
    with_prompts = {(case, attempt): (13_730, False) for case in CASES for attempt in (1, 2)}
    with_prompts[(CASES[0], 1)] = (25_000, True)
    without_prompts = {
        (case, attempt): (13_137, False) if case == CASES[-1] else (21_000, True)
        for case in CASES
        for attempt in (1, 2)
    }
    _write_job(jobs, "with", with_prompts)
    _write_job(jobs, "without", without_prompts)

    measured = _collect(tmp_path)

    assert measured["delta_tokens_mean"] == 593.0
    assert measured["n_pairs"] == 1
    assert measured["status"] == "insufficient"
    assert "hosted web search" in measured["reason"]


# --------------------------------------------------------------------------- #
# M4 (measured half): one failed trial does not delete the measurement         #
# --------------------------------------------------------------------------- #
def _trial(case_id: str, **usage: float) -> TrialObservation:
    return TrialObservation(case_id=case_id, reward=_reward(case_id), usage=usage)


def test_m4_failed_trial_keeps_the_measured_context_cost() -> None:
    """tier3-04 (Claude Code A): a judge failure failed the arm; every other trajectory had its counts (+884, 9 pairs)."""
    cases = [f"case-{index}" for index in range(1, 11)]
    with_trials = [_trial(case, first_turn_prompt_tokens=25_884) for case in cases[:9]]
    with_trials.append(_trial(cases[9]))  # judge failure or timeout: no trajectory counts
    with_arm = ArmObservations(execution_status="failed", trials=tuple(with_trials))
    without_arm = ArmObservations(
        execution_status="succeeded", trials=tuple(_trial(case, first_turn_prompt_tokens=25_000) for case in cases)
    )

    measured = stats.context_cost_measured(with_arm, without_arm)

    assert measured["status"] == "measured"
    assert measured["delta_tokens_mean"] == 884.0
    assert measured["n_pairs"] == 9
    assert measured["partial"] is True
    assert measured["excluded"]["missing_tokens"] == 1
    assert "with-plugin arm" in measured["reason"]


def test_m4_reason_names_the_arm_without_counts() -> None:
    with_arm = ArmObservations(execution_status="succeeded", trials=(_trial("a", first_turn_prompt_tokens=1_500),))
    without_arm = ArmObservations(execution_status="failed", trials=(_trial("a"),))

    measured = stats.context_cost_measured(with_arm, without_arm)

    assert measured["status"] == "unavailable"
    assert "baseline" in measured["reason"]
    assert "with and without arms did not both complete" not in measured["reason"]


def test_m4_reason_says_no_usage_was_collected_rather_than_no_count() -> None:
    """Verifier: with usage not read for a failed arm, the reason claimed the trials had no first-turn count."""
    with_arm = ArmObservations(execution_status="failed", trials=tuple(_trial(case) for case in CASES))
    without_arm = ArmObservations(
        execution_status="succeeded", trials=tuple(_trial(case, first_turn_prompt_tokens=25_000) for case in CASES)
    )

    measured = stats.context_cost_measured(with_arm, without_arm)

    assert measured["status"] == "unavailable"
    assert f"no usage was collected for {len(CASES)} trial(s) of the with-plugin arm" in measured["reason"]
    assert "have no first-turn prompt token count" not in measured["reason"]
    assert measured["excluded"]["missing_tokens"] == len(CASES)


def test_m4_usage_without_a_first_turn_count_is_named_as_such() -> None:
    with_trials = [_trial(case, first_turn_prompt_tokens=25_884) for case in CASES[:-1]]
    with_trials.append(_trial(CASES[-1], prompt_tokens=40_000.0))
    without_trials = [_trial(case, first_turn_prompt_tokens=25_000) for case in CASES]

    measured = stats.context_cost_measured(
        ArmObservations(execution_status="succeeded", trials=tuple(with_trials)),
        ArmObservations(execution_status="succeeded", trials=tuple(without_trials)),
    )

    assert measured["partial"] is True
    assert "1 trial(s) of the with-plugin arm have no first-turn prompt token count" in measured["reason"]
    assert "no usage was collected" not in measured["reason"]


def test_m4_failed_arm_with_every_count_says_collected_trials() -> None:
    """Verifier nit: a judge-failed trial is not observed, so "every trial has its count" overstated it."""
    with_arm = ArmObservations(
        execution_status="failed", trials=tuple(_trial(case, first_turn_prompt_tokens=25_884) for case in CASES)
    )
    without_arm = ArmObservations(
        execution_status="succeeded", trials=tuple(_trial(case, first_turn_prompt_tokens=25_000) for case in CASES)
    )

    measured = stats.context_cost_measured(with_arm, without_arm)

    assert measured["status"] == "measured"
    assert measured["partial"] is True
    assert "every collected trial has its count" in measured["reason"]
    assert "every trial has its count" not in measured["reason"]


# --------------------------------------------------------------------------- #
# L29: minimum number of pairs                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("n_cases", "status"), [(1, "insufficient"), (2, "insufficient"), (3, "measured")])
def test_l29_a_single_pair_is_not_a_measurement(n_cases: int, status: str) -> None:
    """tier3-12: one-case runs printed `measured` "(1 pairs)" next to an `insufficient` lift interval."""
    cases = [f"case-{index}" for index in range(n_cases)]
    with_arm = ArmObservations(
        execution_status="succeeded", trials=tuple(_trial(case, first_turn_prompt_tokens=26_351) for case in cases)
    )
    without_arm = ArmObservations(
        execution_status="succeeded", trials=tuple(_trial(case, first_turn_prompt_tokens=25_000) for case in cases)
    )

    measured = stats.context_cost_measured(with_arm, without_arm)

    assert measured["status"] == status
    assert measured["delta_tokens_mean"] == 1_351.0
    assert measured["min_pairs"] == 3
