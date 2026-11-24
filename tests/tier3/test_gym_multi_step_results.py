# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A multi-step NeMo Gym episode keeps its ordered steps and is scored by its declared strategy.

Gym records each step's verifier code, logs and rewards on ``turns[]`` and leaves the
top-level verifier fields unset, so the adapter writes Harbor's multi-step trial layout
(``step_results`` plus ``steps/<name>/``) for the shared collector instead.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.case_summary import ERROR, FAIL, PASS, case_marks, live_case_marks
from skillevaluator.tier3.harbor.collector import DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES, collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, mark_not_applicable
from skillevaluator.tier3.nemo_gym.results import write_job_from_rollouts, write_task_case_ids

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
    (logs / "test-stdout.txt").write_text(f"{name} verifier output\n", encoding="utf-8")
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


def test_continued_sessions_keep_each_turns_own_transcript(tmp_path: Path) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")
    first, second = _turn(tmp_path, "first", 1.0), _turn(tmp_path, "second", 1.0)
    # Gym exports each turn's own input and new output under the episode's one session id, even when the
    # harness continues its session, so a turn that opens like the one before it is still new work.
    opening = [
        {"step_id": 1, "source": "agent", "message": "listed the repository"},
        {"step_id": 2, "source": "agent", "message": "read login.py"},
    ]
    later = [*opening, {"step_id": 3, "source": "agent", "message": "fixed the login check"}]
    for turn, steps in ((first, opening), (second, later)):
        trajectory = {**_trajectory("episode"), "steps": steps}
        Path(turn["verifier_logs_dir"], "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    row = _episode([first, second], task="login-task", turn_sessions="continue")
    assert _write([row], jobs / "demo-codex-with", {"login-task": "auth-login"}) == []

    collect_harbor_results(
        "demo", ["codex"], out, jobs, skip_baseline=True, expected_cases=1, expected_case_ids=["auth-login"]
    )

    trial = out / "codex" / "with-skill" / "trials" / "login-task__gym001"
    trajectory = json.loads((trial / "trajectory.json").read_text(encoding="utf-8"))
    assert [step["extra"]["harbor_step_name"] for step in trajectory["steps"]] == ["first"] * 2 + ["second"] * 3
    persisted = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    assert persisted["config"] == {"agent": {"resume_trajectory": False}}
    assert persisted["nemo_gym"]["turn_sessions"] == "continue"


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


def _judged_turn(base: Path, name: str, accuracy: float, judge: str) -> dict[str, Any]:
    logs = base / "gym-logs" / name
    logs.mkdir(parents=True)
    scores = {**dict.fromkeys(DEFAULT_METRICS, 0.5), "accuracy": accuracy}
    (logs / "reward.json").write_text(json.dumps(scores), encoding="utf-8")
    sidecar = {**scores, "details": {"accuracy": {"judge_model_used": judge}}}
    (logs / "skill_evaluator_reward.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (logs / "trajectory.json").write_text(json.dumps(_trajectory(name)), encoding="utf-8")
    return {
        "name": name,
        "reward": accuracy,
        "mask_sample": False,
        "verifier_return_code": 0,
        "verifier_logs_dir": str(logs),
    }


@pytest.mark.parametrize(("strategy", "final_only"), [("final", True), ("mean", False)])
def test_shared_collector_attributes_judges_by_gyms_declared_strategy(
    tmp_path: Path, strategy: str, final_only: bool
) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    for arm in ("with", "without"):
        _stage_native_task(out, arm)
        accuracy = 1.0 if arm == "with" else 0.6
        # Equal step scores leave the strategy impossible to infer from the rewards alone.
        turns = [
            _judged_turn(tmp_path / arm, "prepare", accuracy, "fallback" if arm == "with" else "primary"),
            _judged_turn(tmp_path / arm, "finish", accuracy, "primary"),
        ]
        row = _episode(turns, task="login-task", turn_reward_strategy=strategy)
        assert _write([row], jobs / f"demo-codex-{arm}", {"login-task": "auth-login"}) == []

    results = collect_harbor_results(
        "demo", ["codex"], out, jobs, expected_cases=1, expected_case_ids=["auth-login"], expected_trials=1
    )

    reward_path = out / "codex" / "with-skill" / "trials" / "login-task__gym001" / "reward.json"
    reward = json.loads(reward_path.read_text(encoding="utf-8"))
    lift = results["agents"]["codex"]["lift"]
    if final_only:
        assert reward["details"]["accuracy"]["judge_models"] == ["primary"]
        assert lift["accuracy"]["delta"] == pytest.approx(0.4)
        assert "judge_mixed_cases" not in lift
    else:
        assert reward["details"]["accuracy"]["judge_models"] == ["fallback", "primary"]
        assert lift["judge_mixed_cases"] == {"accuracy": 1}
        assert "accuracy" not in lift


@pytest.mark.parametrize("outcome", ["scored", "second-step-setup-failed"])
def test_named_step_evidence_survives_collection_and_job_cleanup(tmp_path: Path, outcome: str) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")
    if outcome == "scored":
        second = _turn(tmp_path, "second", 1.0)
    else:
        second, _reason = _second_turn_failure(tmp_path, "setup-failed")
    row = _episode([_turn(tmp_path, "first", 0.25), second], task="login-task")
    _write([row], jobs / "demo-codex-with", {"login-task": "auth-login"})

    collect_harbor_results(
        "demo", ["codex"], out, jobs, skip_baseline=True, expected_cases=1, expected_case_ids=["auth-login"]
    )
    shutil.rmtree(jobs)

    steps = out / "codex" / "with-skill" / "trials" / "login-task__gym001" / "steps"
    first = steps / "first"
    assert json.loads((first / "trajectory.json").read_text(encoding="utf-8"))["steps"][0]["message"] == "first"
    assert (first / "test-stdout.txt").read_text(encoding="utf-8") == "first verifier output\n"
    if outcome == "scored":
        assert json.loads((steps / "second" / "trajectory.json").read_text(encoding="utf-8"))["session_id"] == (
            "session-second"
        )
        assert (steps / "second" / "test-stdout.txt").is_file()
    else:
        # The step whose setup failed left no evidence to keep.
        assert not (steps / "second").exists()


@pytest.mark.parametrize("partial", [False, True], ids=["scored", "partial"])
def test_retained_trajectories_mask_arm_values(tmp_path: Path, partial: bool) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")
    first = _turn(tmp_path, "first", 1.0)
    second = _second_turn_failure(tmp_path, "setup-failed")[0] if partial else _turn(tmp_path, "second", 1.0)
    secret = 'opaque "sample"\nvalue'
    trajectory = _trajectory("first")
    trajectory["steps"][0].update(
        message=f"before {secret} after", timestamp="2026-11-24T12:34:56.808012Z", extra={"nested": [secret, 7]}
    )
    (Path(first["verifier_logs_dir"]) / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    errors = write_job_from_rollouts(
        [_episode([first, second], task="login-task")],
        jobs / "demo-codex-with",
        eval_name="codex__m",
        task_case_ids={"login-task": "auth-login"},
        n_attempts=1,
        secret_values={secret, "name", "agent", "8080"},
    )
    assert bool(errors) == partial
    collect_harbor_results("demo", ["codex"], out, jobs, skip_baseline=True, expected_case_ids=["auth-login"])
    trial = out / "codex/with-skill/trials/login-task__gym001"
    paths = [trial / "steps/first/trajectory.json"]
    if not partial:
        paths.append(trial / "trajectory.json")
    exposed = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["agent"]["name"] == "codex"
        step = document["steps"][0]
        assert step["message"].startswith("before ") and step["message"].endswith(" after")
        assert step["extra"]["nested"][1] == 7
        if secret in step["message"] or secret in step["extra"]["nested"][0]:
            exposed.append(path.relative_to(trial))
    assert not exposed
    assert "timestamp" not in json.loads(paths[0].read_text(encoding="utf-8"))["steps"][0]


@pytest.mark.parametrize("failed", [False, True], ids=["scored", "failed"])
def test_single_step_published_trajectory_masks_arm_value(tmp_path: Path, failed: bool) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")
    logs = tmp_path / "single"
    row = _graded_rollout(logs)
    if failed:
        row.update(failure="verifier failed", verifier_return_code=1)
    secret = 'opaque "example"\nvalue'
    trajectory = _trajectory("single") | {"final_metrics": {}}
    trajectory["steps"][0]["message"] = f"before {secret}401 after"
    (logs / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    errors = write_job_from_rollouts(
        [row],
        jobs / "demo-codex-with",
        eval_name="codex__m",
        task_case_ids={"login-task": "auth-login"},
        n_attempts=1,
        secret_values={secret},
    )
    assert bool(errors) == failed
    results = collect_harbor_results("demo", ["codex"], out, jobs, skip_baseline=True, expected_case_ids=["auth-login"])
    assert results["agents"]["codex"]["scored_attempts"] == (0 if failed else 1)
    published = out / "codex/with-skill/trials/login-task__gym001/trajectory.json"
    message = json.loads(published.read_text(encoding="utf-8"))["steps"][0]["message"]
    assert message.startswith("before ") and message.endswith(" after")
    assert secret not in message


def test_step_evidence_publication_refuses_linked_files(tmp_path: Path) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")
    row = _episode([_turn(tmp_path, "first", 0.25), _turn(tmp_path, "second", 1.0)], task="login-task")
    _write([row], jobs / "demo-codex-with", {"login-task": "auth-login"})
    secret = tmp_path / "secret.txt"
    secret.write_text("outside the trial\n", encoding="utf-8")
    linked = jobs / "demo-codex-with" / "login-task__gym001" / "steps" / "first" / "verifier" / "test-stdout.txt"
    linked.unlink()
    linked.symlink_to(secret)

    collect_harbor_results(
        "demo", ["codex"], out, jobs, skip_baseline=True, expected_cases=1, expected_case_ids=["auth-login"]
    )

    steps = out / "codex" / "with-skill" / "trials" / "login-task__gym001" / "steps"
    assert not (steps / "first" / "test-stdout.txt").exists()
    assert (steps / "first" / "trajectory.json").is_file()
    assert (steps / "second" / "test-stdout.txt").is_file()


def test_live_marks_score_multi_step_episodes_as_the_final_result_does(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "_harbor-tasks" / "codex" / "with" / "case-1").mkdir(parents=True)
    arm = run_dir / "_nemo-gym" / "codex-with"
    arm.mkdir(parents=True)
    failed_second, _reason = _second_turn_failure(tmp_path / "failed", "no-verifier-result")
    rows = [
        _episode([_turn(tmp_path / "pass", "first", 0.25), _turn(tmp_path / "pass", "second", 1.0)]),
        # Gym stopped after a step missed its min_reward: scored over the steps that ran, below the threshold.
        _episode([_turn(tmp_path / "stop", "first", 0.2)], stopped_after_turn="first"),
        _episode([_turn(tmp_path / "failed", "first", 1.0), failed_second]),
    ]
    (arm / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    marks, running = live_case_marks(run_dir, ["codex"], pass_threshold=0.5)

    assert marks == {"codex": {"with_skill": {"case-1": [PASS, FAIL, ERROR]}}}
    assert running == set()
    final = write_job_from_rollouts(rows, tmp_path / "job", eval_name="codex__m", task_case_ids=CASES, n_attempts=3)
    assert final == ["case case-1 attempt 3: step 'second': test.sh timed out"]


def _graded(
    logs: Path, goal_accuracy: float, *, entry_id: str | None = None, overall: float = 0.95, **sidecar: Any
) -> None:
    rewards: dict[str, Any] = {
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "goal_accuracy": goal_accuracy,
        "overall": overall,
    }
    if entry_id is not None:
        rewards["entry_id"] = entry_id
    (logs / "reward.json").write_text(json.dumps(rewards), encoding="utf-8")
    (logs / "skill_evaluator_reward.json").write_text(json.dumps(rewards | sidecar), encoding="utf-8")


def _graded_turn(
    base: Path, name: str, goal_accuracy: float, entry_id: str | None = None, overall: float = 0.95
) -> dict[str, Any]:
    turn = _turn(base, name, overall)
    _graded(Path(turn["verifier_logs_dir"]), goal_accuracy, entry_id=entry_id, overall=overall)
    return turn


def _judge_failed(logs: Path) -> dict[str, Any]:
    """Write what the built-in verifier writes when a required judge fails, and return how Gym records its exit."""
    scores: dict[str, Any] = {**dict.fromkeys(DEFAULT_METRICS, 1.0), "accuracy": None}
    rewards = {metric: score for metric, score in scores.items() if score is not None} | {"overall": 0.0}
    (logs / "reward.json").write_text(json.dumps(rewards), encoding="utf-8")
    sidecar = scores | {
        "details": {"accuracy": {"score": None, "status": "error", "reason": "judge request timed out"}},
        "evaluation_status": "failed",
        "evaluation_errors": {"accuracy": "judge request timed out"},
    }
    (logs / "skill_evaluator_reward.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return {"reward": 0.0, "verifier_rewards": rewards, "verifier_return_code": 1, "verifier_logs_dir": str(logs)}


def _live_arm(out: Path, rows: list[dict[str, Any]]) -> None:
    arm = out / "_nemo-gym" / "codex-with"
    write_task_case_ids(arm, {"login-task": "auth-login"})
    (arm / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _live_and_final_marks(out: Path, jobs: Path, rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    _live_arm(out, rows)
    live, running = live_case_marks(out, ["codex"], pass_threshold=0.5)
    assert running == set()
    write_job_from_rollouts(
        rows,
        jobs / "demo-codex-with",
        eval_name="codex__m",
        task_case_ids={"login-task": "auth-login"},
        n_attempts=len(rows),
    )
    results = collect_harbor_results(
        "demo",
        ["codex"],
        out,
        jobs,
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["auth-login"],
        expected_trials=len(rows),
    )
    return live["codex"]["with_skill"]["login-task"], case_marks(results)["codex"]["with_skill"]["auth-login"]


@pytest.mark.parametrize("strategy", ["mean", "final"])
def test_live_marks_apply_the_final_verdict_and_staged_case_to_episodes(tmp_path: Path, strategy: str) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")

    def episode(label: str, goal_accuracy: float, second_entry_id: str | None = None) -> dict[str, Any]:
        turns = [
            _graded_turn(tmp_path / label, "first", goal_accuracy),
            _graded_turn(tmp_path / label, "second", goal_accuracy, second_entry_id),
        ]
        return _episode(turns, task="login-task", turn_reward_strategy=strategy)

    judged = _graded_turn(tmp_path / "judge-failed", "first", 1.0, overall=1.0)
    failed_judge = _turn(tmp_path / "judge-failed", "second", 1.0)
    failed_judge |= _judge_failed(Path(failed_judge["verifier_logs_dir"]))
    rows = [
        episode("goal-met", 1.0),
        # Overall clears the threshold, but standard grading requires the goal.
        episode("goal-missed", 0.5),
        episode("other-case", 1.0, second_entry_id="other-case"),
        episode("staged-case", 1.0, second_entry_id="auth-login"),
        # Under mean the two steps still reach the threshold, but a step's required judge failed.
        _episode([judged, failed_judge], task="login-task", turn_reward_strategy=strategy),
    ]

    live, final = _live_and_final_marks(out, jobs, rows)

    assert live == [PASS, FAIL, ERROR, PASS, ERROR]
    assert sorted(live) == sorted(final)


@pytest.mark.parametrize("failed_prepare", [False, True], ids=["healthy", "failed-prepare"])
def test_final_turn_determines_metric_applicability(tmp_path: Path, failed_prepare: bool) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")
    prepare = _graded_turn(tmp_path, "prepare", 1.0, overall=1.0)
    if failed_prepare:
        _judge_failed(Path(prepare["verifier_logs_dir"]))
    final = _graded_turn(tmp_path, "verify", 1.0, overall=1.0)
    logs = Path(final["verifier_logs_dir"])
    numeric = json.loads((logs / "reward.json").read_text())
    del numeric["behavior_check"]
    (logs / "reward.json").write_text(json.dumps(numeric))
    sidecar = numeric.copy()
    mark_not_applicable(sidecar, "behavior_check", "No behavior assertions")
    (logs / "skill_evaluator_reward.json").write_text(json.dumps(sidecar))
    row = _episode([prepare, final], task="login-task", turn_reward_strategy="final")

    live, finished = _live_and_final_marks(out, jobs, [row])

    expected = [ERROR] if failed_prepare else [PASS]
    assert (live, finished) == (expected, expected)
    if not failed_prepare:
        reward = json.loads((out / "codex/with-skill/trials/login-task__gym001/reward.json").read_text())
        assert reward["overall"] == 1.0
        assert reward["details"]["behavior_check"]["reason"] == "No behavior assertions"


def test_live_marks_keep_the_staged_case_after_the_staged_tasks_are_removed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _stage_native_task(out, "with")
    rows = [
        _episode([_graded_turn(tmp_path / case, name, 1.0, case) for name in ("first", "second")], task="login-task")
        for case in ("auth-login", "other-case")
    ]
    _live_arm(out, rows)
    # The run removes its staged tasks before the live view stops polling.
    shutil.rmtree(out / "_harbor-tasks")

    marks, _running = live_case_marks(out, ["codex"], pass_threshold=0.5)

    assert marks == {"codex": {"with_skill": {"login-task": [PASS, ERROR]}}}


def test_live_marks_judge_each_rollout_once_and_leave_no_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3.nemo_gym import results

    out = tmp_path / "out"
    _live_arm(out, [_episode([_graded_turn(tmp_path, name, 1.0) for name in ("first", "second")], task="login-task")])
    judged: list[str] = []
    verdict = results.rollout_verdict
    monkeypatch.setattr(results, "rollout_verdict", lambda row, *args: judged.append(args[0]) or verdict(row, *args))

    polls = [live_case_marks(out, ["codex"], pass_threshold=0.5)[0] for _ in range(2)]

    assert polls == [{"codex": {"with_skill": {"login-task": [PASS]}}}] * 2
    assert judged == ["login-task"]
    assert [path.name for path in (out / "_nemo-gym").iterdir()] == ["codex-with"]


def test_live_marks_keep_the_polled_runs_verdicts_past_the_cache_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3 import case_summary
    from skillevaluator.tier3.nemo_gym import results

    monkeypatch.setattr(case_summary, "_GYM_LIVE_MARKS", {})
    monkeypatch.setattr(case_summary, "_MAX_GYM_LIVE_MARKS", 2)
    judged: list[str] = []
    verdict = results.rollout_verdict
    monkeypatch.setattr(results, "rollout_verdict", lambda row, *args: judged.append(args[0]) or verdict(row, *args))
    out = tmp_path / "out"
    rows = [_graded_rollout(tmp_path / f"rollout-{index}") for index in range(3)]

    judged_per_poll = []
    for count in (2, 2, 3, 3, 3):
        _live_arm(out, rows[:count])
        before = len(judged)
        marks, _running = live_case_marks(out, ["codex"], pass_threshold=0.5)
        judged_per_poll.append(len(judged) - before)
        assert marks == {"codex": {"with_skill": {"login-task": [PASS] * count}}}

    # The run being polled keeps every verdict, even with more rollouts than the cap.
    assert judged_per_poll == [2, 0, 1, 0, 0]
    # Past the cap, a run that is no longer polled gives up its verdicts.
    other = tmp_path / "other"
    _live_arm(other, [_graded_rollout(tmp_path / "other-rollout")])
    live_case_marks(other, ["codex"], pass_threshold=0.5)
    assert list(case_summary._GYM_LIVE_MARKS) == [str(other / "_nemo-gym")]


def _graded_rollout(logs: Path, trajectory_bytes: int = 0) -> dict[str, Any]:
    logs.mkdir()
    _graded(logs, 1.0)
    trajectory = json.dumps(_trajectory(logs.name)).ljust(trajectory_bytes)
    (logs / "trajectory.json").write_text(trajectory, encoding="utf-8")
    (logs / "test-stdout.txt").write_text("verifier output\n", encoding="utf-8")
    return {"task_id": "login-task", "verifier_return_code": 0, "verifier_logs_dir": str(logs)}


def test_live_marks_need_no_write_access_to_the_run_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out, scratch = tmp_path / "out", tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    episode = _episode([_graded_turn(tmp_path, name, 1.0) for name in ("first", "second")], task="login-task")
    _live_arm(out, [episode, _graded_rollout(tmp_path / "single")])
    gym_root = out / "_nemo-gym"
    for folder in (gym_root / "codex-with", gym_root):
        folder.chmod(0o555)
    try:
        if os.access(gym_root, os.W_OK):
            pytest.skip("this account can write to a read-only folder")
        marks, _running = live_case_marks(out, ["codex"], pass_threshold=0.5)
    finally:
        for folder in (gym_root, gym_root / "codex-with"):
            folder.chmod(0o755)

    assert marks == {"codex": {"with_skill": {"login-task": [PASS, PASS]}}}
    assert list(scratch.iterdir()) == []


def test_live_marks_copy_only_what_the_verdict_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    episode = _episode(
        [_graded_turn(tmp_path / "episode", name, 1.0) for name in ("first", "second")], task="login-task"
    )
    rows = [
        episode,
        _graded_rollout(tmp_path / "small"),
        # The collector reads no more of a trajectory than this, so a larger one cannot change the verdict.
        _graded_rollout(tmp_path / "large", DEFAULT_DIAGNOSTIC_ARTIFACT_MAX_BYTES + 1),
    ]
    _live_arm(out, rows)
    copied: list[str] = []
    copyfile = shutil.copyfile

    def record(source: Path, target: Path) -> Any:
        copied.append(Path(source).relative_to(tmp_path).as_posix())
        return copyfile(source, target)

    monkeypatch.setattr(shutil, "copyfile", record)

    marks, _running = live_case_marks(out, ["codex"], pass_threshold=0.5)

    assert marks == {"codex": {"with_skill": {"login-task": [PASS, PASS, PASS]}}}
    rewards = ("reward.json", "skill_evaluator_reward.json")
    assert sorted(copied) == sorted(
        [
            *(f"episode/gym-logs/{step}/{name}" for step in ("first", "second") for name in rewards),
            *(f"{label}/{name}" for label in ("small", "large") for name in rewards),
            "small/trajectory.json",
        ]
    )


def test_live_marks_apply_the_final_verdict_and_staged_case_to_single_steps(tmp_path: Path) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")

    def rollout(label: str, goal_accuracy: float, **reward: Any) -> dict[str, Any]:
        logs = tmp_path / label
        logs.mkdir()
        _graded(logs, goal_accuracy, **reward)
        return {"task_id": "login-task", "verifier_return_code": 0, "verifier_logs_dir": str(logs)}

    not_applicable = {"details": {"goal_accuracy": {"status": "not_applicable"}}}
    judge_failed = tmp_path / "judge-failed"
    judge_failed.mkdir()
    rows = [
        rollout("goal-missed", 0.5),
        rollout("other-case", 1.0, entry_id="other-case"),
        # The collector reads the sidecar's N/A goal back into the reward, so only Overall decides.
        rollout("goal-not-applicable", 0.0, **not_applicable),
        {"task_id": "login-task", **_judge_failed(judge_failed)},
    ]

    live, final = _live_and_final_marks(out, jobs, rows)

    assert live == [FAIL, ERROR, PASS, ERROR]
    assert sorted(live) == sorted(final)


def test_live_marks_still_read_the_trajectory_of_an_agent_that_never_started(tmp_path: Path) -> None:
    out, jobs = tmp_path / "out", tmp_path / "jobs"
    _stage_native_task(out, "with")
    row = _graded_rollout(tmp_path / "never-started")
    # A tokenless trajectory that names an auth failure is an agent that never started, which stays unscored.
    tokenless = _trajectory("never-started") | {"final_metrics": {}}
    tokenless["steps"][0]["message"] = "AuthenticationError: provider refused request"
    Path(row["verifier_logs_dir"], "trajectory.json").write_text(json.dumps(tokenless), encoding="utf-8")
    _live_arm(out, [row])

    marks, _running = live_case_marks(out, ["codex"], pass_threshold=0.5)

    assert marks == {"codex": {"with_skill": {"login-task": [ERROR]}}}
    write_job_from_rollouts(
        [row],
        jobs / "demo-codex-with",
        eval_name="codex__m",
        task_case_ids={"login-task": "auth-login"},
        n_attempts=1,
        secret_values={"Error"},
    )
    results = collect_harbor_results(
        "demo", ["codex"], out, jobs, skip_baseline=True, expected_cases=1, expected_case_ids=["auth-login"]
    )
    assert results["agents"]["codex"]["scored_attempts"] == 0
