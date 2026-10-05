# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Not-applicable judged metrics: reward.json -> collector -> summaries, lift, and reports.

A judged metric (accuracy, goal_accuracy, behavior_check) whose eval case has no
ground_truth / expected_behavior is recorded by the verifier as ``null`` with
``details[metric].status == "not_applicable"`` in ``skill_evaluator_reward.json``.
Harbor's ``reward.json`` stays numeric-only, so the metric is simply absent
there. N/A metrics must be excluded from the trial overall, the arm averages,
and lift, and must render as N/A, never as 0 or 1.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.tier3.commands import compare_results
from skillevaluator.tier3.harbor import report, report_data
from skillevaluator.tier3.harbor.collector import _compute_lift, _logical_attempt_rewards, collect_harbor_results
from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    average_metrics,
    dimension_scores,
    metric_is_not_applicable,
    not_applicable_counts,
    not_applicable_metrics,
    overall_score,
)
from skillevaluator.tier3.harbor.templates import metric as harbor_metric_template
from skillevaluator.tier3.result_display import _with_skill_overall, render_evaluation_result

N = None  # a judged metric the verifier recorded as not applicable
JUDGED = ("accuracy", "goal_accuracy", "behavior_check")


# ---------------------------------------------------------------------------
# Fixtures that mirror the verifier's reward.json + skill_evaluator_reward.json
# ---------------------------------------------------------------------------


def _verifier_outputs(entry_id: str, scores: tuple[float | None, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (numeric reward.json, rich sidecar) as the verifier template writes them."""
    values = dict(zip(DEFAULT_METRICS, scores, strict=True))
    details: dict[str, Any] = {}
    sidecar: dict[str, Any] = {"metric_set": DEFAULT_METRIC_SET, "entry_id": entry_id, "has_skill": True}
    numeric: dict[str, Any] = {}
    for metric, value in values.items():
        sidecar[metric] = value
        if value is None:
            details[metric] = {"score": None, "status": "not_applicable", "reason": "N/A: no reference"}
        else:
            numeric[metric] = value
            details[metric] = {"score": value, "reason": "judged"}
    sidecar["details"] = details
    scored = [value for value in values.values() if value is not None]
    numeric["overall"] = round(sum(scored) / len(scored), 4)
    return numeric, sidecar


def _write_complete_job_result(job_dir: Path, trial_names: list[str]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(trial_names),
                "stats": {
                    "n_completed_trials": len(trial_names),
                    "n_errored_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_retries": 0,
                    "evals": {
                        "agent__model___harbor-tasks": {
                            "n_trials": len(trial_names),
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.1": trial_names}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_job(
    jobs_dir: Path,
    job_name: str,
    cases: dict[str, tuple[float | None, ...]],
    *,
    with_sidecar: bool = True,
) -> None:
    job_dir = jobs_dir / job_name
    trial_names = []
    for case_id, scores in cases.items():
        trial_name = f"{case_id}__attempt1"
        trial_names.append(trial_name)
        numeric, sidecar = _verifier_outputs(case_id, scores)
        verifier = job_dir / trial_name / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.json").write_text(json.dumps(numeric), encoding="utf-8")
        if with_sidecar:
            (verifier / "skill_evaluator_reward.json").write_text(json.dumps(sidecar), encoding="utf-8")
    _write_complete_job_result(job_dir, trial_names)


def _collect(tmp_path: Path, case_ids: list[str], *, skip_baseline: bool = False) -> dict[str, Any]:
    return collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=skip_baseline,
        expected_cases=len(case_ids),
        expected_case_ids=case_ids,
        expected_trials=len(case_ids),
    )


# Two cases: behavior_check is N/A everywhere, accuracy/goal_accuracy only for case-b.
#                sec  exec  eff  acc  goal  behavior
WITH_PARTIAL = {
    "case-a": (1.0, 1.0, 1.0, 0.8, 1.0, N),
    "case-b": (1.0, 1.0, 0.5, N, N, N),
}
WITHOUT_PARTIAL = {
    "case-a": (1.0, 0.0, 0.0, 0.4, 0.5, N),
    "case-b": (1.0, 0.0, 0.0, N, N, N),
}


@pytest.fixture
def partial_run(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    _write_job(tmp_path / "jobs", "demo-opencode-with", WITH_PARTIAL)
    _write_job(tmp_path / "jobs", "demo-opencode-without", WITHOUT_PARTIAL)
    return tmp_path, _collect(tmp_path, ["case-a", "case-b"])


@pytest.fixture
def all_judged_na_run(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    _write_job(tmp_path / "jobs", "demo-opencode-with", {"case-a": (1.0, 1.0, 1.0, N, N, N)})
    _write_job(tmp_path / "jobs", "demo-opencode-without", {"case-a": (1.0, 0.0, 0.5, N, N, N)})
    return tmp_path, _collect(tmp_path, ["case-a"])


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


def _merged(scores: tuple[float | None, ...]) -> dict[str, Any]:
    """Numeric reward.json with the sidecar merged back, as the collector does."""
    numeric, sidecar = _verifier_outputs("case", scores)
    return {**sidecar, **numeric}


def test_not_applicable_requires_the_explicit_verifier_marker() -> None:
    reward = _merged((1.0, 1.0, 1.0, 0.5, N, N))

    assert metric_is_not_applicable(reward, "goal_accuracy")
    assert metric_is_not_applicable(reward, "behavior_check")
    assert not metric_is_not_applicable(reward, "accuracy")  # numeric wins

    missing = dict(reward)
    missing.pop("details")
    assert not metric_is_not_applicable(missing, "goal_accuracy")  # absence alone is never N/A

    errored = json.loads(json.dumps(reward))
    errored["details"]["goal_accuracy"]["status"] = "error"
    assert not metric_is_not_applicable(errored, "goal_accuracy")

    garbage = json.loads(json.dumps(reward))
    garbage["goal_accuracy"] = "n/a"
    assert not metric_is_not_applicable(garbage, "goal_accuracy")


def test_security_can_never_be_not_applicable() -> None:
    reward = _merged((1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    reward["security"] = None
    reward["details"]["security"] = {"score": None, "status": "not_applicable"}

    assert not metric_is_not_applicable(reward, "security")
    assert overall_score(reward) is None


def test_skill_metrics_are_not_applicable_only_with_the_verifier_marker() -> None:
    # Proof H5: an arm without the skill records the skill metrics as N/A.
    reward = _merged((1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    reward["skill_execution"] = None
    reward["details"]["skill_execution"] = {"score": None, "status": "not_applicable"}

    assert metric_is_not_applicable(reward, "skill_execution")
    assert overall_score(reward) == pytest.approx(1.0)
    del reward["details"]["skill_execution"]
    assert not metric_is_not_applicable(reward, "skill_execution")
    assert overall_score(reward) is None


def test_overall_score_excludes_not_applicable_metrics_only() -> None:
    assert overall_score(_merged((1.0, 1.0, 1.0, 0.0, 0.0, N))) == pytest.approx(0.6)
    # Every judged metric N/A: mean of the deterministic metrics.
    assert overall_score(_merged((1.0, 0.5, 0.0, N, N, N))) == pytest.approx(0.5)
    # A genuine 0.0 stays in the mean.
    assert overall_score(_merged((1.0, 1.0, 1.0, 0.0, 0.0, 0.0))) == pytest.approx(0.5)


def test_overall_score_keeps_missing_metric_without_marker_unscored() -> None:
    numeric, _sidecar = _verifier_outputs("case", (1.0, 1.0, 1.0, 1.0, 1.0, N))

    assert "behavior_check" not in numeric
    assert overall_score(numeric) is None


def test_arm_averages_and_counts_exclude_not_applicable_trials() -> None:
    rewards = [_merged(scores) for scores in WITH_PARTIAL.values()]

    averages, _metric_set, metrics = average_metrics(rewards)

    assert averages == {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 0.75,
        "accuracy": 0.8,
        "goal_accuracy": 1.0,
    }
    assert not_applicable_metrics(rewards, metrics) == ["behavior_check"]
    assert not_applicable_counts(rewards, metrics) == {"accuracy": 1, "goal_accuracy": 1, "behavior_check": 2}


def test_dimension_scores_renormalize_over_applicable_sources() -> None:
    scores = {"security": 1.0, "skill_execution": 0.5, "skill_efficiency": 0.5, "accuracy": 0.8, "goal_accuracy": 0.6}

    dimensions = dimension_scores(scores, ["behavior_check"])

    assert dimensions["effectiveness"] == {
        "score": 0.6,
        "sources": {"goal_accuracy": 0.5},
        "not_applicable_sources": ["behavior_check"],
    }
    # Without the N/A declaration a missing source still omits the dimension.
    assert "effectiveness" not in dimension_scores(scores)


def test_dimension_scores_omit_dimensions_whose_sources_are_all_not_applicable() -> None:
    scores = {"security": 1.0, "skill_execution": 0.5, "skill_efficiency": 0.5}

    dimensions = dimension_scores(scores, list(JUDGED))

    assert set(dimensions) == {"security", "discoverability", "efficiency"}


# ---------------------------------------------------------------------------
# collector: lift and logical attempts
# ---------------------------------------------------------------------------


def test_lift_overall_uses_the_shared_basket_when_metrics_are_not_applicable() -> None:
    with_scores = {"security": 1.0, "skill_execution": 1.0, "skill_efficiency": 1.0, "accuracy": 1.0}
    without_scores = {"security": 1.0, "skill_execution": 0.0, "skill_efficiency": 0.0, "accuracy": 0.0}

    lift = _compute_lift(
        with_scores,
        without_scores,
        with_not_applicable=["goal_accuracy", "behavior_check"],
        without_not_applicable=["goal_accuracy", "behavior_check"],
    )

    assert set(lift) == {"security", "skill_execution", "skill_efficiency", "accuracy", "overall"}
    assert lift["overall"] == {"with_skill": 1.0, "without_skill": 0.25, "delta": 0.75}


def test_lift_overall_is_suppressed_when_a_metric_is_missing_without_marker() -> None:
    with_scores = {"security": 1.0, "skill_execution": 1.0, "skill_efficiency": 1.0, "accuracy": 1.0}
    without_scores = dict(with_scores)

    lift = _compute_lift(with_scores, without_scores, with_not_applicable=["behavior_check"])

    assert "overall" not in lift
    assert "behavior_check" not in lift


def test_lift_is_unchanged_when_nothing_is_not_applicable() -> None:
    scores = dict.fromkeys(DEFAULT_METRICS, 0.5)

    assert _compute_lift(scores, scores)["overall"] == {"with_skill": 0.5, "without_skill": 0.5, "delta": 0.0}


def test_multistep_fallback_keeps_not_applicable_markers_on_the_logical_trial() -> None:
    rows = []
    for index, scores in enumerate([(1.0, 1.0, 1.0, 0.5, N, N), (1.0, 0.5, 1.0, 1.0, N, N)]):
        reward = _merged(scores)
        reward.update({"_trial_root_name": "case__attempt1", "_step_name": f"step-{index}"})
        rows.append(reward)

    [logical] = _logical_attempt_rewards(rows)

    assert logical["goal_accuracy"] is None
    assert metric_is_not_applicable(logical, "goal_accuracy")
    assert metric_is_not_applicable(logical, "behavior_check")
    assert overall_score(logical) == pytest.approx((1.0 + 0.75 + 1.0 + 0.75) / 4)


# ---------------------------------------------------------------------------
# collector: end to end
# ---------------------------------------------------------------------------


def test_collector_excludes_not_applicable_metrics_from_scores_lift_and_pass_at_k(
    partial_run: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, result = partial_run
    agent = result["agents"]["opencode"]

    assert result["execution_status"] == "succeeded"
    assert agent["with_skill"] == {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 0.75,
        "accuracy": 0.8,
        "goal_accuracy": 1.0,
    }
    assert agent["not_applicable_metrics"] == {
        "with_skill": ["behavior_check"],
        "without_skill": ["behavior_check"],
        "sum_of_parts": [],
    }
    assert "behavior_check" not in agent["lift"]
    assert agent["lift"]["accuracy"]["delta"] == 0.4
    assert agent["lift"]["overall"] == {"with_skill": 0.91, "without_skill": 0.38, "delta": 0.53}
    assert agent["dimensions_with_skill"]["effectiveness"]["sources"] == {"goal_accuracy": 0.5}
    assert agent["dimensions_with_skill"]["correctness"]["score"] == 0.8

    cases = agent["pass_at_k"]["with_skill"]["cases"]
    assert cases["case-a"]["best_score"] == 0.96  # mean of 5 scored metrics
    assert cases["case-b"]["best_score"] == 0.8333  # mean of 3 scored metrics

    summary = json.loads((tmp_path / "results/opencode/with-skill/summary.json").read_text(encoding="utf-8"))
    assert summary["not_applicable_metrics"] == ["behavior_check"]
    assert summary["not_applicable_counts"] == {"accuracy": 1, "goal_accuracy": 1, "behavior_check": 2}
    assert summary["overall_score"] == round((0.96 + 2.5 / 3) / 2, 4)

    persisted = json.loads(
        (tmp_path / "results/opencode/with-skill/trials/case-b__attempt1/reward.json").read_text(encoding="utf-8")
    )
    assert persisted["accuracy"] is None
    assert persisted["details"]["accuracy"]["status"] == "not_applicable"


def test_collector_handles_every_judged_metric_not_applicable(
    all_judged_na_run: tuple[Path, dict[str, Any]],
) -> None:
    _tmp_path, result = all_judged_na_run
    agent = result["agents"]["opencode"]

    assert result["execution_status"] == "succeeded"
    assert agent["with_skill"] == {"security": 1.0, "skill_execution": 1.0, "skill_efficiency": 1.0}
    assert agent["not_applicable_metrics"]["with_skill"] == list(JUDGED)
    assert set(agent["lift"]) == {"security", "skill_execution", "skill_efficiency", "overall"}
    assert agent["lift"]["overall"] == {"with_skill": 1.0, "without_skill": 0.5, "delta": 0.5}
    assert set(agent["dimensions_with_skill"]) == {"security", "discoverability", "efficiency"}
    assert agent["pass_at_k"]["with_skill"]["passed_cases"] == 1


def test_collector_fails_closed_when_the_not_applicable_sidecar_is_missing(tmp_path: Path) -> None:
    _write_job(tmp_path / "jobs", "demo-opencode-with", {"case-a": (1.0, 1.0, 1.0, 1.0, 1.0, N)}, with_sidecar=False)

    result = _collect(tmp_path, ["case-a"], skip_baseline=True)

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "failed"
    assert agent["conditions"]["with_skill"]["scored_attempts"] == 0
    assert agent["with_skill"] == {}


_EVAL_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)


def _run_verifier_without_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the verifier template's main() for an agent that left no trajectory."""
    module_name = f"harbor_eval_no_trajectory_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(module_name, _EVAL_TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    logs_dir = tmp_path / "logs"
    agent_dir = logs_dir / "agent"
    verifier_dir = logs_dir / "verifier"
    tests_dir = tmp_path / "tests"
    for directory in (agent_dir, verifier_dir, tests_dir):
        directory.mkdir(parents=True)
    for name, value in {
        "LOGS_DIR": logs_dir,
        "AGENT_LOGS_DIR": agent_dir,
        "VERIFIER_DIR": verifier_dir,
        "TESTS_DIR": tests_dir,
        "ATIF_PATH": agent_dir / "trajectory.json",
        "ENTRY_PATH": tests_dir / "entry.json",
        "REWARD_JSON": verifier_dir / "reward.json",
        "REWARD_TXT": verifier_dir / "reward.txt",
        "SKILL_EVALUATOR_REWARD_JSON": verifier_dir / "skill_evaluator_reward.json",
    }.items():
        monkeypatch.setattr(module, name, value)
    (tests_dir / "entry.json").write_text(json.dumps(entry), encoding="utf-8")

    module.main()

    reward = json.loads((verifier_dir / "reward.json").read_text(encoding="utf-8"))
    sidecar = json.loads((verifier_dir / "skill_evaluator_reward.json").read_text(encoding="utf-8"))
    return reward, sidecar


def test_no_trajectory_trial_keeps_reference_less_judges_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The agent crashed before writing a trajectory. The case has no
    # ground_truth / expected_behavior, so its judge metrics stay N/A instead
    # of turning the arm's N/A accuracy into 0.0.
    entry = {"id": "case-1", "question": "Summarize the report.", "expected_skill": "reporter", "has_skill": True}

    reward, sidecar = _run_verifier_without_trajectory(tmp_path, monkeypatch, entry)

    assert sidecar["error"] == "No trajectory or reconstructible agent log"
    assert set(reward) == {"security", "skill_execution", "skill_efficiency", "overall"}
    assert reward["overall"] == 0.0
    merged = {**sidecar, **reward}
    for metric in JUDGED:
        assert sidecar[metric] is None
        assert sidecar["details"][metric]["status"] == "not_applicable"
        assert sidecar["details"][metric]["reason"].startswith("N/A")
        assert metric_is_not_applicable(merged, metric)
    assert overall_score(merged) == 0.0

    # Nine judged-N/A trials plus this crash: the arm keeps the judges N/A.
    arm = [_merged((1.0, 1.0, 1.0, N, N, N)) for _ in range(9)] + [merged]
    averages, _metric_set, metrics = average_metrics(arm)
    assert not set(JUDGED) & set(averages)
    assert set(not_applicable_metrics(arm, metrics)) == set(JUDGED)


def test_no_trajectory_trial_fails_every_judge_the_case_defines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = {
        "id": "case-1",
        "question": "Summarize the report.",
        "ground_truth": "A summary.",
        "expected_behavior": ["Summarizes the report"],
        "has_skill": True,
    }

    reward, sidecar = _run_verifier_without_trajectory(tmp_path, monkeypatch, entry)

    assert [sidecar[metric] for metric in JUDGED] == [0, 0, 0]
    assert [reward[metric] for metric in JUDGED] == [0.0, 0.0, 0.0]
    assert "details" not in sidecar
    assert overall_score({**sidecar, **reward}) == 0.0


def _write_multistep_trial(
    job_dir: Path,
    trial_name: str,
    step_scores: dict[str, tuple[float | None, ...]],
    *,
    sidecar_steps: set[str],
    authoritative: bool = True,
) -> None:
    trial_dir = job_dir / trial_name
    step_results = []
    numeric_rows = []
    for step_name, scores in step_scores.items():
        numeric, sidecar = _verifier_outputs("case-a", scores)
        verifier = trial_dir / "steps" / step_name / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.json").write_text(json.dumps(numeric), encoding="utf-8")
        if step_name in sidecar_steps:
            (verifier / "skill_evaluator_reward.json").write_text(json.dumps(sidecar), encoding="utf-8")
        step_results.append({"step_name": step_name, "verifier_result": {"rewards": numeric}})
        numeric_rows.append(numeric)
    aggregate = {
        key: sum(row[key] for row in numeric_rows) / len(numeric_rows)
        for key in numeric_rows[0]
        if all(key in row for row in numeric_rows)
    }
    result: dict[str, Any] = {"trial_name": trial_name, "task_name": "case-a", "step_results": step_results}
    if authoritative:
        result["verifier_result"] = {"rewards": aggregate}
    (trial_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_authoritative_multistep_aggregate_restores_not_applicable_from_step_sidecars(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    _write_multistep_trial(
        job_dir,
        "case-a__attempt1",
        {"prepare": (1.0, 1.0, 1.0, 0.5, 1.0, N), "finish": (1.0, 0.5, 1.0, 1.0, 1.0, N)},
        sidecar_steps={"prepare", "finish"},
    )
    _write_complete_job_result(job_dir, ["case-a__attempt1"])

    result = _collect(tmp_path, ["case-a"], skip_baseline=True)

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert "behavior_check" not in agent["with_skill"]
    assert agent["with_skill"]["accuracy"] == 0.75
    assert agent["not_applicable_metrics"]["with_skill"] == ["behavior_check"]


def test_multistep_step_rewards_without_root_aggregate_keep_not_applicable(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    _write_multistep_trial(
        job_dir,
        "case-a__attempt1",
        {"prepare": (1.0, 1.0, 1.0, 0.5, N, N), "finish": (1.0, 0.5, 1.0, 1.0, N, N)},
        sidecar_steps={"prepare", "finish"},
        authoritative=False,
    )
    _write_complete_job_result(job_dir, ["case-a__attempt1"])

    result = _collect(tmp_path, ["case-a"], skip_baseline=True)

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert agent["with_skill"] == {
        "security": 1.0,
        "skill_execution": 0.75,
        "skill_efficiency": 1.0,
        "accuracy": 0.75,
    }
    assert agent["not_applicable_metrics"]["with_skill"] == ["goal_accuracy", "behavior_check"]
    assert agent["pass_at_k"]["with_skill"]["cases"]["case-a"]["best_score"] == 0.875


def test_authoritative_multistep_aggregate_without_step_marker_stays_unscored(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    _write_multistep_trial(
        job_dir,
        "case-a__attempt1",
        {"prepare": (1.0, 1.0, 1.0, 0.5, 1.0, N), "finish": (1.0, 0.5, 1.0, 1.0, 1.0, N)},
        sidecar_steps={"prepare"},
    )
    _write_complete_job_result(job_dir, ["case-a__attempt1"])

    result = _collect(tmp_path, ["case-a"], skip_baseline=True)

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["scored_attempts"] == 0


# ---------------------------------------------------------------------------
# rendering: report_data, CLI, HTML, compare, findings, Harbor metric
# ---------------------------------------------------------------------------


def test_report_data_loads_arm_not_applicable_metrics(partial_run: tuple[Path, dict[str, Any]]) -> None:
    tmp_path, _result = partial_run

    agents = report_data.load_agent_data(tmp_path / "results")

    assert agents["opencode"]["not_applicable_with_skill"] == ["behavior_check"]
    assert agents["opencode"]["not_applicable_without_skill"] == ["behavior_check"]


def _render_cli(result: dict[str, Any]) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=160, force_terminal=False, color_system=None)
    render_evaluation_result(result, console=console)
    return buffer.getvalue()


def _row(output: str, label: str) -> str:
    [line] = [line for line in output.splitlines() if re.search(rf"^\W*{label}\b", line.strip())]
    return line


def test_cli_renders_not_applicable_metrics_as_na(partial_run: tuple[Path, dict[str, Any]]) -> None:
    _tmp_path, result = partial_run

    output = _render_cli(result)

    behavior_row = _row(output, "Behavior Check")
    assert behavior_row.count("N/A") == 3  # with skill, no skill, lift
    assert "NO SCORE" not in behavior_row
    assert not re.search(r"\d\.\d\d", behavior_row)
    assert "N/A = not applicable" in output
    assert re.search(r"\b0\.80\b", _row(output, "Accuracy"))


def test_cli_renders_not_applicable_dimensions_when_every_judge_is_na(
    all_judged_na_run: tuple[Path, dict[str, Any]],
) -> None:
    _tmp_path, result = all_judged_na_run

    output = _render_cli(result)

    for label in ("Correctness", "Effectiveness"):
        row = _row(output, label)
        assert "N/A" in row
        assert "NO SCORE" not in row
        assert not re.search(r"\d\.\d\d", row)
    for label in ("Accuracy", "Goal Accuracy", "Behavior Check"):
        assert "N/A" in _row(output, label)


def test_cli_skip_baseline_overall_excludes_not_applicable_metrics(tmp_path: Path) -> None:
    _write_job(tmp_path / "jobs", "demo-opencode-with", {"case-a": (1.0, 1.0, 0.5, N, N, N)})
    result = _collect(tmp_path, ["case-a"], skip_baseline=True)

    agent = result["agents"]["opencode"]

    assert _with_skill_overall(agent, DEFAULT_METRIC_SET) == pytest.approx(0.8333)


def _report_payload(tmp_path: Path) -> tuple[str, dict[str, Any]]:
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir(exist_ok=True)
    report_path = render_agent_eval_html_report(skill_dir, tmp_path / "results", use_llm_judge=False)
    html = report_path.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="tier3-full">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    return html, json.loads(match.group(1))


def test_html_report_renders_not_applicable_evaluators_and_trial_cells(
    partial_run: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, _result = partial_run

    html, payload = _report_payload(tmp_path)

    agent = payload["agents"]["opencode"]
    # behavior_check N/A everywhere: Effectiveness falls back to goal_accuracy, so the gate still decides.
    assert payload["verdict"] == "pass"
    assert "behavior_check" not in agent["evaluators"]
    assert [item["id"] for item in agent["not_applicable_evaluators"]] == ["behavior_check"]
    assert [item["id"] for item in payload["not_applicable_evaluators"]] == ["behavior_check"]
    trials = {trial["entry_id"]: trial for trial in agent["trials"]}
    assert set(trials["case-b"]["not_applicable"]) == set(JUDGED)
    assert "accuracy" not in trials["case-b"]["scores"]
    assert trials["case-a"]["scores"]["accuracy"] == 0.8
    assert "Not applicable (excluded from scores and lift)" in html
    assert 'data-not-applicable="true"' in html
    assert "N/A (not applicable)" in html


def test_html_report_lists_not_applicable_dimensions_when_every_judge_is_na(
    all_judged_na_run: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, _result = all_judged_na_run

    html, payload = _report_payload(tmp_path)

    assert {dimension["id"] for dimension in payload["dimensions"]} == {"security", "discoverability", "efficiency"}
    assert [item["id"] for item in payload["not_applicable_dimensions"]] == ["correctness", "effectiveness"]
    assert payload["overall_score"] == pytest.approx(1.0)
    # The every-dimension gate cannot pass without Correctness/Effectiveness evidence.
    assert payload["verdict"] == "neutral"
    assert "had nothing to judge against in any eval case" in html


def test_compare_renders_not_applicable_metrics_and_excludes_them_from_overall(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    run_dir = tmp_path / "results" / "demo" / "20260709_010000"
    agent_dir = run_dir / "opencode"
    for variant, scores in (
        ("with-skill", {"security": 1.0, "skill_execution": 1.0, "skill_efficiency": 1.0}),
        ("without-skill", {"security": 1.0, "skill_execution": 0.0, "skill_efficiency": 0.5}),
    ):
        (agent_dir / variant).mkdir(parents=True)
        (agent_dir / variant / "summary.json").write_text(
            json.dumps(
                {
                    "execution_status": "succeeded",
                    "scores": scores,
                    "not_applicable_metrics": list(JUDGED),
                }
            ),
            encoding="utf-8",
        )
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_dir.name, "agents": {}}), encoding="utf-8")

    assert compare_results(skill_path, results_dir=tmp_path / "results") == 0

    output = capsys.readouterr().out
    for metric in JUDGED:
        row = _row(output, metric)
        assert row.count("N/A") == 2
        assert not re.search(r"\d\.\d\d", row)
    overall = _row(output, "Overall")
    assert "1.00" in overall
    assert "+0.50" in overall


def test_findings_ignore_not_applicable_reasons() -> None:
    rewards = [_merged(scores) for scores in WITH_PARTIAL.values()]

    findings = {finding["metric"]: finding for finding in report._extract_findings(rewards)}

    assert "behavior_check" not in findings
    assert findings["accuracy"]["score"] == 0.8
    assert all("N/A" not in reason for reason in findings["accuracy"]["reasons"])


def test_harbor_dataset_metric_omits_metrics_no_task_scored(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rewards.jsonl"
    rows = [_verifier_outputs(case, scores)[0] for case, scores in WITH_PARTIAL.items()]
    rewards_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output_path = tmp_path / "metrics.json"

    harbor_metric_template.main(rewards_path, output_path)

    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert "behavior_check" not in metrics
    assert metrics["accuracy"] == 0.8
    assert metrics["overall"] == round((1.0 + 1.0 + 0.75 + 0.8 + 1.0) / 5, 4)
