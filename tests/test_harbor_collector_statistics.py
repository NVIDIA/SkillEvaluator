# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection writes report-only statistics beside the scores and never changes the scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.statistics_run_fixtures import AGENT, CASES, SKILL, Trial, collect, graded_baseline, uniform

from skillevaluator.tier3.harbor.collector import collect_harbor_results

# The blocks a skill run adds to each agent's results.
FIELDS = ("lift_uncertainty", "reliability")


def _agent(collected: dict[str, Any]) -> dict[str, Any]:
    return collected["agents"][AGENT]


def _bytes(run: Path, relative: str) -> bytes:
    return (run / AGENT / relative).read_bytes()


def test_collect_writes_statistics_json_and_agent_blocks(tmp_path: Path) -> None:
    collected, run = collect(tmp_path, uniform(0.9), graded_baseline())

    assert collected["execution_status"] == "succeeded"
    assert "warnings" not in collected
    written = json.loads((run / AGENT / "statistics.json").read_text(encoding="utf-8"))
    assert set(written) == set(FIELDS)
    agent = _agent(collected)
    assert {field: agent[field] for field in FIELDS} == written
    interval = agent["lift_uncertainty"]["effectiveness"]
    assert interval["n_cases"] == 6
    assert interval["precision"] == "low"
    assert agent["reliability"]["with_skill"]["pass_hat_k"] == 1.0
    assert agent["reliability"]["without_skill"]["pass_hat_k"] == 0.5


def test_statistics_do_not_change_scores_lift_pass_at_k_or_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3.harbor import run_statistics

    with_skill, baseline = uniform(0.9), graded_baseline()
    first, run = collect(tmp_path, with_skill, baseline)
    held = {
        name: _bytes(run, name)
        for name in ("lift.json", "pass_at_k_lift.json", "with-skill/summary.json", "without-skill/summary.json")
    }
    assert (run / AGENT / "statistics.json").exists()

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("do not leak this")

    monkeypatch.setattr(run_statistics, "agent_statistics", explode)
    second, run_again = collect(tmp_path, with_skill, baseline)

    assert run_again == run
    stripped = [
        {key: value for key, value in _agent(result).items() if key not in FIELDS} for result in (first, second)
    ]
    assert stripped[0] == stripped[1]
    assert {key for key in _agent(first) if key in FIELDS} == set(FIELDS)
    assert not {key for key in _agent(second) if key in FIELDS}
    assert {name: _bytes(run, name) for name in held} == held
    assert not (run / AGENT / "statistics.json").exists()
    for key in ("execution_status", "score_policy", "attempt_policy", "metrics", "metric_set"):
        assert second[key] == first[key]
    assert second["warnings"] == ["Report-only run statistics were not computed for codex (RuntimeError)."]
    assert "warnings" not in first


def test_estimate_matches_headline_lift_on_fully_paired_cases(tmp_path: Path) -> None:
    collected, _run = collect(tmp_path, uniform(0.9), graded_baseline())

    agent = _agent(collected)
    interval = agent["lift_uncertainty"]["effectiveness"]
    overall = agent["lift"]["overall"]
    assert interval["estimate"] == pytest.approx(overall["delta"], abs=1e-4)
    assert interval["treatment_score"] == pytest.approx(overall["with_skill"], abs=1e-4)
    assert interval["control_score"] == pytest.approx(overall["without_skill"], abs=1e-4)
    assert interval["ci_low"] < interval["estimate"] < interval["ci_high"]


def test_judge_mixed_case_is_left_out_of_the_interval(tmp_path: Path) -> None:
    # case-1 is judged by a different model in each arm and its accuracy differs; with that
    # judge metric left out, no case differs between the arms.
    with_skill = {case: [Trial(0.8, judge="primary")] for case in CASES}
    with_skill["case-1"] = [Trial(0.8, accuracy=0.0, judge="fallback")]
    baseline = {case: [Trial(0.8, judge="primary")] for case in CASES}

    collected, _run = collect(tmp_path, with_skill, baseline)

    agent = _agent(collected)
    assert agent["lift"]["judge_mixed_cases"] == {"accuracy": 1}
    interval = agent["lift_uncertainty"]["effectiveness"]
    assert interval["n_cases"] == 6
    assert (interval["estimate"], interval["ci_low"], interval["ci_high"]) == (0.0, 0.0, 0.0)
    assert interval["ci_includes_zero"] is True
    assert agent["lift"]["overall"]["delta"] == 0.0


def test_statistics_json_is_removed_when_a_rerun_produces_none(tmp_path: Path) -> None:
    _collected, run = collect(tmp_path, uniform(0.9), uniform(0.5))
    assert (run / AGENT / "statistics.json").exists()

    # An agent left out of a later collection into the same directory loses its stale file.
    collected = collect_harbor_results(SKILL, ["other"], run, tmp_path / "jobs", expected_case_ids=CASES)

    assert not (run / AGENT / "statistics.json").exists()
    assert AGENT not in collected["agents"]


def test_reused_baseline_collection_reports_both_arms(tmp_path: Path) -> None:
    _fresh, source = collect(tmp_path, uniform(0.9), graded_baseline(), run_name="source")

    collected, run = collect(tmp_path, uniform(0.9), None, reused_baseline_dir=source)

    agent = _agent(collected)
    assert agent["execution_status"] == "succeeded"
    assert agent["reliability"]["without_skill"]["pass_hat_k"] == 0.5
    assert agent["reliability"]["with_skill"]["pass_hat_k"] == 1.0
    assert agent["lift_uncertainty"]["effectiveness"]["n_cases"] == 6
    assert (run / AGENT / "statistics.json").exists()


def test_baseline_only_and_skill_only_runs_report_their_own_arm(tmp_path: Path) -> None:
    skill_only, _run = collect(tmp_path, uniform(0.9), None, skip_baseline=True, run_name="skill")
    baseline_only, _run = collect(tmp_path, None, uniform(0.5), skip_with_skill=True, run_name="baseline")

    assert set(_agent(skill_only)["reliability"]) == {"with_skill"}
    assert _agent(skill_only)["lift_uncertainty"] == {"effectiveness": None}
    assert set(_agent(baseline_only)["reliability"]) == {"without_skill"}
    assert _agent(baseline_only)["lift_uncertainty"] == {"effectiveness": None}


def test_a_never_activated_skill_has_no_interval_but_keeps_reliability(tmp_path: Path) -> None:
    with_skill = {case: [Trial(0.9, activated=False)] for case in CASES}

    collected, _run = collect(tmp_path, with_skill, uniform(0.5))

    agent = _agent(collected)
    assert agent["skill_activation"]["status"] == "never_activated"
    assert agent["lift"] == {}
    assert agent["lift_uncertainty"] == {"effectiveness": None}
    assert agent["reliability"]["with_skill"]["pass_hat_k"] == 1.0


def test_a_failed_arm_has_no_interval_or_reliability(tmp_path: Path) -> None:
    jobs_without = uniform(0.5)

    collected, _run = collect(tmp_path, uniform(0.9), jobs_without, expected_trials=7)

    agent = _agent(collected)
    assert agent["execution_status"] == "failed"
    assert agent["lift_uncertainty"] == {"effectiveness": None}
    assert agent["reliability"] == {}


def test_stop_on_pass_with_uneven_attempts_has_no_interval_to_disagree_with_the_headline(tmp_path: Path) -> None:
    # case-1 needed three attempts and case-2 two in the with-skill arm, case-3 and case-4 in the baseline.
    with_skill = {**uniform(0.9), "case-1": [Trial(0.1, False), Trial(0.1, False), Trial(0.9)]}
    with_skill["case-2"] = [Trial(0.2, False), Trial(0.9)]
    baseline = {**uniform(0.5), "case-3": [Trial(0.3, False), Trial(0.3, False), Trial(0.5)]}
    baseline["case-4"] = [Trial(0.3, False), Trial(0.5)]

    collected, _run = collect(tmp_path, with_skill, baseline, n_attempts=3, stop_on_pass=True, expected_trials=9)

    agent = _agent(collected)
    assert agent["execution_status"] == "succeeded"
    assert agent["lift"]["overall"]["delta"] > 0
    assert agent["lift_uncertainty"] == {"effectiveness": None}
    assert agent["reliability"]["with_skill"]["pass_hat_k"] is None
