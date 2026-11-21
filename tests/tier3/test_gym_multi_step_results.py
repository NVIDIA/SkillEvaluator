# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A multi-step NeMo Gym episode keeps its ordered steps and is scored by its declared strategy.

Gym records each step's verifier code, logs and rewards on ``turns[]`` and leaves the
top-level verifier fields unset, so the adapter writes Harbor's multi-step trial layout
(``step_results`` plus ``steps/<name>/``) for the shared collector instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.nemo_gym.results import write_job_from_rollouts

CASES = {"case-1": "case-1"}


def _trajectory(name: str) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": f"session-{name}",
        "trajectory_id": f"trajectory-{name}",
        "agent": {"name": "codex", "version": "1"},
        "steps": [{"step_id": 1, "source": "agent", "message": name}],
        "final_metrics": {"total_prompt_tokens": 10, "total_completion_tokens": 2},
    }


def _turn(base: Path, name: str, score: float | None, **fields: Any) -> dict[str, Any]:
    logs = base / "gym-logs" / name
    logs.mkdir(parents=True)
    rewards = None if score is None else {"overall": score, "correctness": score}
    if rewards is not None:
        (logs / "reward.json").write_text(json.dumps(rewards), encoding="utf-8")
        (logs / "skill_evaluator_reward.json").write_text(json.dumps(rewards), encoding="utf-8")
    (logs / "trajectory.json").write_text(json.dumps(_trajectory(name)), encoding="utf-8")
    turn = {
        "name": name,
        "reward": score or 0.0,
        "mask_sample": False,
        "verifier_rewards": rewards,
        "verifier_return_code": None if score is None else 0,
        "verifier_logs_dir": str(logs),
    }
    return turn | fields


def _episode(turns: list[dict[str, Any]], *, task: str = "case-1", **fields: Any) -> dict[str, Any]:
    row = {
        "_ng_task_id": {"taskset": "t", "task_id": task},
        "episode_id": {"rollout_id": "r1", "attempt": 0},
        "reward": 99.0,
        "mask_sample": False,
        "verifier_return_code": None,
        "verifier_logs_dir": None,
        "turns": turns,
        "turn_sessions": "fresh",
        "turn_reward_strategy": "mean",
        "turns_without_verifier_result": sum(1 for turn in turns if turn.get("verifier_rewards") is None),
        "stopped_after_turn": None,
    }
    return row | fields


def _write(rows: list[dict[str, Any]], job_dir: Path, cases: dict[str, str] = CASES) -> list[str]:
    return write_job_from_rollouts(rows, job_dir, eval_name="codex__m", task_case_ids=cases, n_attempts=1)


@pytest.mark.parametrize(("strategy", "expected"), [("mean", 0.625), ("final", 1.0)])
def test_successful_turns_become_ordered_step_results(tmp_path: Path, strategy: str, expected: float) -> None:
    job_dir = tmp_path / "job"
    row = _episode([_turn(tmp_path, "first", 0.25), _turn(tmp_path, "second", 1.0)], turn_reward_strategy=strategy)

    assert _write([row], job_dir) == []

    trial = job_dir / "case-1__gym001"
    result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    assert result["exception_info"] is None
    assert result["verifier_result"]["rewards"] == {"overall": expected, "correctness": expected, "entry_id": "case-1"}
    assert [step["step_name"] for step in result["step_results"]] == ["first", "second"]
    assert [step["verifier_result"]["rewards"]["overall"] for step in result["step_results"]] == [0.25, 1.0]
    assert all(step["exception_info"] is None for step in result["step_results"])
    assert result["config"] == {"agent": {"resume_trajectory": False}}
    assert result["nemo_gym"] == {
        "turn_sessions": "fresh",
        "turn_reward_strategy": strategy,
        "stopped_after_turn": None,
        "turns_without_verifier_result": 0,
    }
    for name, score in (("first", 0.25), ("second", 1.0)):
        step = trial / "steps" / name
        assert json.loads((step / "verifier" / "reward.json").read_text(encoding="utf-8"))["overall"] == score
        assert (step / "verifier" / "skill_evaluator_reward.json").is_file()
        assert (
            json.loads((step / "agent" / "trajectory.json").read_text(encoding="utf-8"))["steps"][0]["message"] == name
        )
    # The last step's evidence never stands in for the episode's.
    assert not (trial / "verifier").exists()
    assert not (trial / "agent").exists()
    job = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert job["stats"]["n_errors"] == 0
    assert job["stats"]["evals"]["codex__m"]["reward_stats"]["reward"] == {str(expected): ["case-1__gym001"]}


def test_issue_reproduction_with_reward_only_files_is_scored(tmp_path: Path) -> None:
    turns = []
    for name, score in (("first", 0.25), ("second", 1.0)):
        logs = tmp_path / name
        logs.mkdir()
        (logs / "reward.json").write_text(json.dumps({"reward": score}), encoding="utf-8")
        turns.append(
            {
                "name": name,
                "reward": score,
                "mask_sample": False,
                "verifier_rewards": {"reward": score},
                "verifier_return_code": 0,
                "verifier_logs_dir": str(logs),
            }
        )
    row = _episode(turns, task_id={"task_id": "case-1"}, reward=0.625, verifier_rewards={"reward": 0.625})
    del row["_ng_task_id"]

    assert _write([row], tmp_path / "job") == []

    result = json.loads((tmp_path / "job" / "case-1__gym001" / "result.json").read_text(encoding="utf-8"))
    assert result["verifier_result"]["rewards"] == {"reward": 0.625, "entry_id": "case-1"}
    assert len(result["step_results"]) == 2


def test_continued_sessions_merge_step_trajectories_as_resumed(tmp_path: Path) -> None:
    row = _episode([_turn(tmp_path, "first", 1.0), _turn(tmp_path, "second", 1.0)], turn_sessions="continue")

    assert _write([row], tmp_path / "job") == []

    result = json.loads((tmp_path / "job" / "case-1__gym001" / "result.json").read_text(encoding="utf-8"))
    assert result["config"] == {"agent": {"resume_trajectory": True}}
    assert result["nemo_gym"]["turn_sessions"] == "continue"


def test_a_missed_min_reward_is_scored_over_the_steps_that_ran(tmp_path: Path) -> None:
    row = _episode([_turn(tmp_path, "first", 0.2)], stopped_after_turn="first")

    assert _write([row], tmp_path / "job") == []

    result = json.loads((tmp_path / "job" / "case-1__gym001" / "result.json").read_text(encoding="utf-8"))
    assert result["verifier_result"]["rewards"]["overall"] == 0.2
    assert [step["step_name"] for step in result["step_results"]] == ["first"]
    assert result["nemo_gym"]["stopped_after_turn"] == "first"


def test_mean_counts_a_metric_a_step_did_not_report_as_zero(tmp_path: Path) -> None:
    first = _turn(tmp_path, "first", 1.0)
    Path(first["verifier_logs_dir"], "reward.json").write_text(json.dumps({"overall": 1.0}), encoding="utf-8")
    row = _episode([first, _turn(tmp_path, "second", 0.5)])

    assert _write([row], tmp_path / "job") == []

    result = json.loads((tmp_path / "job" / "case-1__gym001" / "result.json").read_text(encoding="utf-8"))
    assert result["verifier_result"]["rewards"] == {"overall": 0.75, "correctness": 0.25, "entry_id": "case-1"}


def _second_turn_failure(base: Path, kind: str) -> tuple[dict[str, Any], str]:
    if kind == "masked":
        turn = _turn(base, "second", 1.0, mask_sample=True, failure_kind="harbor:trajectory_export_failed")
        return turn, "harbor:trajectory_export_failed"
    if kind == "no-verifier-result":
        turn = _turn(base, "second", None, failure_kind="harbor:no_turn_result", failure_reason="test.sh timed out")
        return turn, "test.sh timed out"
    if kind == "setup-failed":
        turn = {
            "name": "second",
            "reward": 0.0,
            "failure_kind": "harbor:step_setup",
            "failure_reason": "setup.sh exit 1",
        }
        return turn, "setup.sh exit 1"
    if kind == "missing-reward-file":
        turn = _turn(base, "second", 1.0)
        Path(turn["verifier_logs_dir"], "reward.json").unlink()
        return turn, "the task verifier wrote no reward.json or numeric reward.txt"
    turn = _turn(base, "second", 1.0)
    Path(turn["verifier_logs_dir"], "reward.json").write_text(json.dumps({"entry_id": "other", "overall": 1.0}))
    return turn, "the task verifier reported case 'other' for staged case 'case-1'"


@pytest.mark.parametrize(
    "kind", ["masked", "no-verifier-result", "setup-failed", "missing-reward-file", "case-mismatch"]
)
def test_a_failed_step_leaves_the_episode_unscored_with_its_step_evidence(tmp_path: Path, kind: str) -> None:
    second, reason = _second_turn_failure(tmp_path, kind)
    row = _episode([_turn(tmp_path, "first", 1.0), second])

    errors = _write([row], tmp_path / "job")

    assert errors == [f"case case-1 attempt 1: step 'second': {reason}"]
    trial = tmp_path / "job" / "case-1__gym001"
    result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    assert result["verifier_result"] is None
    assert result["exception_info"] == {
        "exception_type": "NeMoGymEpisodeFailure",
        "exception_message": f"step 'second': {reason}",
    }
    first, failed = result["step_results"]
    assert first["verifier_result"]["rewards"]["overall"] == 1.0 and first["exception_info"] is None
    assert failed["verifier_result"] is None
    assert failed["exception_info"] == {"exception_type": "NeMoGymStepFailure", "exception_message": reason}
    assert (trial / "steps" / "first" / "verifier" / "reward.json").is_file()
    job = json.loads((tmp_path / "job" / "result.json").read_text(encoding="utf-8"))
    assert job["stats"]["n_errors"] == 1
    assert job["stats"]["evals"]["codex__m"]["reward_stats"]["reward"] == {}


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"turns": []}, "NeMo Gym returned malformed multi-step turns"),
        ({"turns": "first"}, "NeMo Gym returned malformed multi-step turns"),
        ({"turn_reward_strategy": None}, "NeMo Gym reported an unsupported turn_reward_strategy None"),
        ({"turn_reward_strategy": "sum"}, "NeMo Gym reported an unsupported turn_reward_strategy 'sum'"),
        ({"stopped_after_turn": "first"}, "NeMo Gym stopped after turn 'first', which is not its last turn"),
        ({"mask_sample": True}, "NeMo Gym masked the sample (Gym-side failure)"),
    ],
)
def test_an_inconsistent_episode_is_not_scored(tmp_path: Path, fields: dict[str, Any], reason: str) -> None:
    row = _episode([_turn(tmp_path, "first", 1.0), _turn(tmp_path, "second", 1.0)]) | fields

    assert _write([row], tmp_path / "job") == [f"case case-1 attempt 1: {reason}"]

    result = json.loads((tmp_path / "job" / "case-1__gym001" / "result.json").read_text(encoding="utf-8"))
    assert result["verifier_result"] is None
    assert result["exception_info"]["exception_message"] == reason


@pytest.mark.parametrize("names", [("same", "same"), ("Step", "step"), ("ok", "../escape"), ("ok", "trailing.")])
def test_step_names_that_cannot_be_distinct_directories_are_rejected(tmp_path: Path, names: tuple[str, str]) -> None:
    turns = [_turn(tmp_path / str(index), name.replace("/", "_"), 1.0) for index, name in enumerate(names)]
    for turn, name in zip(turns, names, strict=True):
        turn["name"] = name
    job_dir = tmp_path / "job"

    errors = _write([_episode(turns)], job_dir)

    assert errors == ["case case-1 attempt 1: NeMo Gym returned malformed multi-step turns"]
    assert not (job_dir / "case-1__gym001" / "steps").exists()
    assert not (job_dir / "escape").exists()


def test_a_single_step_row_with_null_turns_keeps_the_single_step_layout(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text(json.dumps({"overall": 0.5}), encoding="utf-8")
    row = {"task_id": "case-1", "turns": None, "verifier_return_code": 0, "verifier_logs_dir": str(logs)}

    assert _write([row], tmp_path / "job") == []

    trial = tmp_path / "job" / "case-1__gym001"
    result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    assert "step_results" not in result
    assert result["verifier_result"]["rewards"] == {"overall": 0.5, "entry_id": "case-1"}
    assert (trial / "verifier" / "reward.json").is_file()


def _stage_native_task(out: Path, arm: str) -> None:
    task = out / "_harbor-tasks" / "codex" / arm / "login-task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        f'[task]\nname = "nvidia/login-task-{arm}-skill"\n\n[metadata]\nentry_id = "auth-login"\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize(("strategy", "expected"), [("mean", 0.625), ("final", 1.0)])
def test_shared_collector_scores_the_episode_once_with_merged_step_evidence(
    tmp_path: Path, strategy: str, expected: float
) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    for arm in ("with", "without"):
        _stage_native_task(out, arm)
        turns = [_turn(tmp_path / arm, "first", 0.25), _turn(tmp_path / arm, "second", 1.0)]
        row = _episode(turns, task="login-task", turn_reward_strategy=strategy)
        assert _write([row], jobs / f"demo-codex-{arm}", {"login-task": "auth-login"}) == []

    results = collect_harbor_results(
        "demo", ["codex"], out, jobs, expected_cases=1, expected_case_ids=["auth-login"], expected_trials=1
    )

    agent = results["agents"]["codex"]
    assert agent["overall_with_skill"] == pytest.approx(expected)
    assert agent["overall_without_skill"] == pytest.approx(expected)
    assert agent["num_trials_with"] == 1
    trial = out / "codex" / "with-skill" / "trials" / "login-task__gym001"
    reward = json.loads((trial / "reward.json").read_text(encoding="utf-8"))
    assert reward["entry_id"] == "auth-login"
    assert reward["overall"] == pytest.approx(expected)
    assert reward.get("evaluation_status") != "failed"
    persisted = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    assert [step["step_name"] for step in persisted["step_results"]] == ["first", "second"]
    trajectory = json.loads((trial / "trajectory.json").read_text(encoding="utf-8"))
    assert [step["message"] for step in trajectory["steps"]] == ["first", "second"]
    assert [step["extra"]["harbor_step_name"] for step in trajectory["steps"]] == ["first", "second"]


def test_shared_collector_records_a_failed_step_as_an_unscored_attempt(tmp_path: Path) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")
    second, reason = _second_turn_failure(tmp_path, "no-verifier-result")
    row = _episode([_turn(tmp_path, "first", 1.0), second], task="login-task")
    _write([row], jobs / "demo-codex-with", {"login-task": "auth-login"})

    results = collect_harbor_results(
        "demo",
        ["codex"],
        out,
        jobs,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["auth-login"],
        expected_trials=1,
    )

    agent = results["agents"]["codex"]
    assert agent["scored_attempts"] == 0
    assert agent["overall_with_skill"] is None
    trial = out / "codex" / "with-skill" / "trials" / "login-task__gym001"
    failure = json.loads((trial / "failure.json").read_text(encoding="utf-8"))
    assert failure["entry_id"] == "auth-login"
    assert failure["error"]["message"] == f"step 'second': {reason}"
    persisted = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    assert [step["step_name"] for step in persisted["step_results"]] == ["first", "second"]
