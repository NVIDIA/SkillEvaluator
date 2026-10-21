# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native multi-step judge provenance follows Harbor's scoring strategy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def _write_native_job(job: Path, *, strategy: str | None, equal_scores: bool, with_skill: bool) -> None:
    trial_name = "case-1__trial"
    trial = job / trial_name
    final_accuracy = 1.0 if with_skill else 0.6
    initial_accuracy = final_accuracy if equal_scores else 0.0 if with_skill else 0.4
    steps = []
    for name, accuracy in (("prepare", initial_accuracy), ("finish", final_accuracy)):
        scores = {**dict.fromkeys(DEFAULT_METRICS, 0.5), "accuracy": accuracy}
        verifier = trial / "steps" / name / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.json").write_text(json.dumps(scores), encoding="utf-8")
        (verifier / "skill_evaluator_reward.json").write_text(
            json.dumps(
                {
                    **scores,
                    "details": {
                        "accuracy": {"judge_model_used": "fallback" if with_skill and name == "prepare" else "primary"}
                    },
                }
            ),
            encoding="utf-8",
        )
        steps.append({"step_name": name, "verifier_result": {"rewards": scores}})
    root_accuracy = (initial_accuracy + final_accuracy) / 2 if strategy == "mean" else final_accuracy
    result = {
        "trial_name": trial_name,
        "task_name": "case-1",
        "verifier_result": {"rewards": {**dict.fromkeys(DEFAULT_METRICS, 0.5), "accuracy": root_accuracy}},
        "step_results": steps,
    }
    if strategy is not None:
        task = job / "task" / "case-1"
        task.mkdir(parents=True)
        (task / "task.toml").write_text(f'multi_step_reward_strategy = "{strategy}"\n', encoding="utf-8")
        result["config"] = {"task": {"path": str(task)}}
    (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (job / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_retries": 0,
                    "evals": {
                        "agent__model___tasks": {
                            "n_trials": 1,
                            "n_errors": 0,
                            "reward_stats": {"overall": {"0.5": [trial_name]}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("strategy", "equal_scores", "final_only"),
    [
        ("final", False, True),
        ("final", True, True),
        (None, False, True),
        ("mean", False, False),
        ("mean", True, False),
        (None, True, False),
    ],
    ids=("declared-final", "equal-final", "inferred-final", "declared-mean", "equal-mean", "ambiguous"),
)
def test_native_judge_provenance_uses_only_scoring_steps(
    tmp_path: Path, strategy: str | None, equal_scores: bool, final_only: bool
) -> None:
    jobs = tmp_path / "jobs"
    for arm in ("with", "without"):
        _write_native_job(
            jobs / f"demo-codex-{arm}", strategy=strategy, equal_scores=equal_scores, with_skill=arm == "with"
        )
    run = tmp_path / "run"
    collected = collect_harbor_results("demo", ["codex"], run, jobs, expected_case_ids=["case-1"], expected_trials=1)

    assert collected["execution_status"] == "succeeded"
    reward_file = next((run / "codex" / "with-skill" / "trials").glob("*/reward.json"))
    reward = json.loads(reward_file.read_text(encoding="utf-8"))
    expected_judges = ["primary"] if final_only else ["fallback", "primary"]
    assert reward["details"]["accuracy"]["judge_models"] == expected_judges
    lift = collected["agents"]["codex"]["lift"]
    if final_only:
        assert lift["accuracy"]["delta"] == pytest.approx(0.4)
        assert "judge_mixed_cases" not in lift
    else:
        assert lift["judge_mixed_cases"] == {"accuracy": 1}
        assert "accuracy" not in lift
    report = agent_eval_result_from_directory(tmp_path / "demo", run, use_llm_judge=False).metadata["agent_eval"]
    agent = report["agents"]["codex"]
    assert agent["evaluators"]["accuracy"]["lift"] == (pytest.approx(0.4) if final_only else None)
    assert agent["lift"] == (pytest.approx(0.08) if final_only else None)
