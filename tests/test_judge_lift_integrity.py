# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge fallback exclusions survive collection and canonical report reloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.tier3.harbor.collector import _extract_rewards, collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS, DEFAULT_SCORE_POLICY


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_job(job: Path, accuracies: list[float], judges: list[tuple[str, ...]], *, native: bool) -> None:
    trials = []
    for index, (accuracy, step_judges) in enumerate(zip(accuracies, judges, strict=True), 1):
        case = f"case-{index}"
        trial_name = f"{case}__trial"
        trials.append(trial_name)
        trial = job / trial_name
        scores = {**dict.fromkeys(DEFAULT_METRICS, 0.5), "accuracy": accuracy}
        steps = []
        for step_index, judge in enumerate(step_judges):
            name = f"step-{step_index}"
            verifier = trial / "steps" / name / "verifier" if native else trial / "verifier"
            rich = {
                **scores,
                "entry_id": case,
                "metric_set": DEFAULT_METRIC_SET,
                "score_policy": DEFAULT_SCORE_POLICY,
                "details": {"accuracy": {"score": accuracy, "judge_model_used": judge}},
            }
            _write_json(verifier / "reward.json", scores if native else rich)
            if native:
                _write_json(verifier / "skill_evaluator_reward.json", rich)
                steps.append({"step_name": name, "verifier_result": {"rewards": scores}})
        if native:
            _write_json(
                trial / "result.json",
                {
                    "trial_name": trial_name,
                    "task_name": case,
                    "verifier_result": {"rewards": scores},
                    "step_results": steps,
                },
            )
    _write_json(
        job / "result.json",
        {
            "n_total_trials": len(trials),
            "stats": {
                "n_completed_trials": len(trials),
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
                "n_retries": 0,
                "evals": {
                    "agent__model___tasks": {
                        "n_trials": len(trials),
                        "n_errors": 0,
                        "reward_stats": {"overall": {"0.5": trials}},
                    }
                },
            },
        },
    )


def _collect(root: Path, *, native: bool = False, all_mixed: bool = False) -> tuple[dict, Path]:
    judges_with = [("primary", "fallback"), ("primary", "fallback" if all_mixed else "primary")]
    judges_without = [("primary", "primary"), ("primary", "primary")]
    if not native:
        judges_with = [(row[-1],) for row in judges_with]
        judges_without = [(row[-1],) for row in judges_without]
    _write_job(root / "jobs" / "demo-codex-with", [1.0, 0.0], judges_with, native=native)
    _write_job(root / "jobs" / "demo-codex-without", [0.0, 0.4], judges_without, native=native)
    run = root / "run"
    collected = collect_harbor_results(
        "demo", ["codex"], run, root / "jobs", expected_case_ids=["case-1", "case-2"], expected_trials=2
    )
    assert collected["execution_status"] == "succeeded"
    return collected, run


def _report(run: Path) -> dict:
    return agent_eval_result_from_directory(run.parent / "demo", run, use_llm_judge=False).metadata["agent_eval"]


@pytest.mark.parametrize("native", [False, True])
def test_report_retains_judge_filtered_regression(tmp_path: Path, native: bool) -> None:
    collected, run = _collect(tmp_path, native=native)
    lift = collected["agents"]["codex"]["lift"]
    assert lift["judge_mixed_cases"] == {"accuracy": 1}
    assert lift["accuracy"]["delta"] == -0.4
    assert lift["overall"]["delta"] == -0.08

    # A stale cache must not reverse the comparison or its explanation.
    cached = run / "codex" / "lift.json"
    payload = json.loads(cached.read_text())
    payload["accuracy"]["delta"] = 0.9
    payload["overall"]["delta"] = 0.9
    _write_json(cached, payload)
    summary = run / "codex" / "with-skill" / "summary.json"
    payload = json.loads(summary.read_text())
    payload["dimensions"]["correctness"].update(
        explanation="+0.30 lift over baseline 0.20.",
        reasoning_bullets=["Improvement!"],
        sources={"goal_accuracy": 0.5},
    )
    _write_json(summary, payload)

    report = _report(run)
    agent = report["agents"]["codex"]
    assert report["overall_lift"] == agent["lift"] == -0.08
    assert agent["with_skill"] == 0.5
    assert agent["baseline"] == 0.44
    assert agent["evaluators"]["accuracy"]["lift"] == -0.4
    correctness = next(dim for dim in agent["dimensions"] if dim["id"] == "correctness")
    assert correctness["lift"] == -0.4
    assert "-0.40 regression versus baseline 0.40" in correctness["explanation"]
    assert "different judge models" in correctness["explanation"]
    assert "+0.30" not in correctness["explanation"]


@pytest.mark.parametrize("native", [False, True])
def test_all_mixed_judges_leave_metric_dimension_and_overall_unavailable(tmp_path: Path, native: bool) -> None:
    collected, run = _collect(tmp_path, native=native, all_mixed=True)
    lift = collected["agents"]["codex"]["lift"]
    assert lift["judge_mixed_cases"] == {"accuracy": 2}
    assert "accuracy" not in lift
    assert "overall" not in lift
    report = _report(run)
    agent = report["agents"]["codex"]
    assert report["overall_lift"] is None
    assert agent["lift"] is None
    assert agent["evaluators"]["accuracy"]["lift"] is None
    correctness = next(dim for dim in agent["dimensions"] if dim["id"] == "correctness")
    assert correctness["lift"] is None
    assert "Lift unavailable" in correctness["explanation"]
    assert "No baseline run" not in correctness["explanation"]


@pytest.mark.parametrize(
    "missing_evidence", ["reward", "judge_metadata", "partial_judge_metadata", "retained_judge_metadata"]
)
def test_exclusions_cannot_be_reconstructed_from_incomplete_evidence(tmp_path: Path, missing_evidence: str) -> None:
    _, run = _collect(tmp_path, all_mixed=missing_evidence == "partial_judge_metadata")
    reward = next((run / "codex" / "with-skill" / "trials").glob("*/reward.json"))
    if missing_evidence == "reward":
        reward.unlink()
    else:
        for reward in (run / "codex").glob("*/trials/*/reward.json"):
            payload = json.loads(reward.read_text())
            if (
                missing_evidence in {"partial_judge_metadata", "retained_judge_metadata"}
                and payload["entry_id"] != "case-2"
            ):
                continue
            payload.pop("details", None)
            _write_json(reward, payload)
    report = _report(run)
    assert report["overall_lift"] is None
    assert report["agents"]["codex"]["evaluators"]["accuracy"]["lift"] is None


def test_native_aggregate_keeps_numeric_authority_and_only_declared_step_judges(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job, [0.25], [("primary", "fallback")], native=True)
    trial = job / "case-1__trial"
    for sidecar in trial.glob("steps/*/verifier/skill_evaluator_reward.json"):
        payload = json.loads(sidecar.read_text())
        payload["accuracy"] = 1.0
        _write_json(sidecar, payload)
    _write_json(
        trial / "steps" / "orphan" / "verifier" / "skill_evaluator_reward.json",
        {"details": {"accuracy": {"judge_model_used": "unrelated"}}},
    )
    rows = _extract_rewards(job)
    assert len(rows) == 1
    assert rows[0]["accuracy"] == 0.25
    assert rows[0]["_judge_models"] == {"accuracy": ["fallback", "primary"]}
    assert rows[0]["details"]["accuracy"]["judge_models"] == ["fallback", "primary"]
