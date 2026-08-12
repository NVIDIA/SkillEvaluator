# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor subprocess completion is not sufficient proof of successful trials."""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import collector, runner


@pytest.mark.parametrize("configured_concurrency", [1, 3, 4])
def test_agent_pair_treats_concurrency_as_a_global_condition_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_concurrency: int,
) -> None:
    lock = threading.Lock()
    active_budget = 0
    maximum_active_budget = 0
    launched: list[tuple[str, int]] = []

    def _run_harbor(**kwargs: object) -> tuple[bool, str]:
        nonlocal active_budget, maximum_active_budget
        budget = int(kwargs["n_concurrent"])
        with lock:
            active_budget += budget
            maximum_active_budget = max(maximum_active_budget, active_budget)
            launched.append((str(kwargs["job_name"]), budget))
        time.sleep(0.05)
        with lock:
            active_budget -= budget
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", _run_harbor)

    errors = runner._run_agent_pair(
        skill_name="demo",
        agent="opencode",
        model="nvidia/openai/gpt-oss-120b",
        env_mode="docker",
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=1,
        n_concurrent=configured_concurrency,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=4,
    )

    assert errors == []
    assert {name for name, _budget in launched} == {"demo-opencode-with", "demo-opencode-without"}
    assert maximum_active_budget <= configured_concurrency


def test_agent_pair_assigns_the_full_concurrency_budget_when_baseline_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    budgets: list[int] = []
    monkeypatch.setattr(
        runner,
        "_run_harbor",
        lambda **kwargs: budgets.append(int(kwargs["n_concurrent"])) or (True, ""),
    )

    errors = runner._run_agent_pair(
        skill_name="demo",
        agent="opencode",
        model="nvidia/openai/gpt-oss-120b",
        env_mode="docker",
        with_skill=tmp_path / "with",
        baseline=None,
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=1,
        n_concurrent=4,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=4,
    )

    assert errors == []
    assert budgets == [4]


@pytest.mark.parametrize(
    ("env_mode", "agent", "import_path"),
    [
        (
            "docker",
            "codex",
            "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex",
        ),
        (
            "docker",
            "claude-code",
            "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode",
        ),
        (
            "local",
            "codex",
            "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildCodex",
        ),
        (
            "local",
            "claude-code",
            "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildClaudeCode",
        ),
    ],
)
def test_stop_on_pass_preserves_nvidia_build_agent_import_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_mode: str,
    agent: str,
    import_path: str,
) -> None:
    launches: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "_run_harbor",
        lambda **kwargs: launches.append(kwargs) or (True, ""),
    )
    monkeypatch.setattr(runner, "_job_passed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", lambda *_args, **_kwargs: None)

    errors = runner._run_agent_pair(
        skill_name="demo",
        agent=agent,
        model="nvidia/nemotron-3-super-120b-a12b",
        env_mode=env_mode,
        with_skill=tmp_path / "with",
        baseline=None,
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=2,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=2,
        agent_import_path=import_path,
        stop_on_pass=True,
        task_names=["case-001"],
    )

    assert errors == []
    assert len(launches) == 1
    assert launches[0]["agent_import_path"] == import_path
    assert launches[0]["include_task_names"] == ["case-001"]


_UNSAFE_LINK = r"symlink|reparse"


@pytest.mark.parametrize("link_kind", ["directory", "dangling"])
def test_merge_attempt_jobs_rejects_linked_whole_job_root(tmp_path: Path, link_kind: str) -> None:
    target = tmp_path / "real-job"
    if link_kind == "directory":
        target.mkdir()
    job_link = tmp_path / "attempt-001"
    job_link.symlink_to(target, target_is_directory=True)
    aggregate_dir = tmp_path / "aggregate"

    with pytest.raises(ValueError, match=r"non-linked|symlink|reparse"):
        runner._merge_attempt_jobs([job_link], aggregate_dir)

    assert not aggregate_dir.exists()


def test_merge_attempt_jobs_rejects_mocked_reparse_whole_job_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    detect_link = runner._job_path_is_link_or_reparse

    def mocked_reparse(path: Path, metadata: object) -> bool:
        return path == job_dir or detect_link(path, metadata)

    monkeypatch.setattr(runner, "_job_path_is_link_or_reparse", mocked_reparse)

    with pytest.raises(ValueError, match="non-linked"):
        runner._merge_attempt_jobs([job_dir], tmp_path / "aggregate")


def test_merge_attempt_jobs_rejects_non_directory_whole_job_root(tmp_path: Path) -> None:
    job_file = tmp_path / "attempt-001"
    job_file.write_text("not a job", encoding="utf-8")

    with pytest.raises(ValueError, match="non-linked directory"):
        runner._merge_attempt_jobs([job_file], tmp_path / "aggregate")


@pytest.mark.parametrize("artifact_kind", ["symlink", "hardlink"])
def test_merge_attempt_jobs_rejects_forged_root_result(tmp_path: Path, artifact_kind: str) -> None:
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    forged = tmp_path / "forged-result.json"
    forged.write_text('{"n_total_trials": 999, "stats": {}}', encoding="utf-8")
    result = job_dir / "result.json"
    try:
        if artifact_kind == "symlink":
            result.symlink_to(forged)
        else:
            result.hardlink_to(forged)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"{artifact_kind} creation unavailable: {exc}")

    with pytest.raises(ValueError, match=r"symlink|reparse|hard.?link|multiple links"):
        runner._merge_attempt_jobs([job_dir], tmp_path / "aggregate")


def test_merge_attempt_jobs_rejects_root_result_source_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secure_copy = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    result = job_dir / "result.json"
    result.write_text('{"n_total_trials": 1, "stats": {}}', encoding="utf-8")
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    marker = aggregate_dir / "keep.txt"
    marker.write_text("old aggregate", encoding="utf-8")
    original = secure_copy._build_tree_manifest

    def validate_then_replace(source: Path, *args: object, **kwargs: object):
        manifest = original(source, *args, **kwargs)
        if Path(source).resolve() == job_dir.resolve():
            result.unlink()
            result.write_text('{"n_total_trials": 999, "stats": {}}', encoding="utf-8")
        return manifest

    monkeypatch.setattr(secure_copy, "_build_tree_manifest", validate_then_replace)

    with pytest.raises(ValueError, match="source changed after validation"):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "old aggregate"


def test_merge_attempt_jobs_rejects_regular_root_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    (job_dir / "result.json").write_text('{"n_total_trials": 1, "stats": {}}', encoding="utf-8")
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    marker = aggregate_dir / "keep.txt"
    marker.write_text("old aggregate", encoding="utf-8")
    original_copy = runner.copytree_secure
    replaced = False

    def replace_root_then_copy(source: Path, destination: Path, **kwargs: object) -> None:
        nonlocal replaced
        if not replaced and Path(source) == job_dir:
            replaced = True
            job_dir.rename(tmp_path / "original-job")
            job_dir.mkdir()
            (job_dir / "result.json").write_text('{"n_total_trials": 999, "stats": {}}', encoding="utf-8")
        original_copy(source, destination, **kwargs)

    monkeypatch.setattr(runner, "copytree_secure", replace_root_then_copy)

    with pytest.raises(ValueError, match="root changed during snapshot"):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "old aggregate"


def _write_multistep_attempt_job(
    job_dir: Path,
    *,
    root_score: float,
    step_scores: tuple[float, ...],
    failed: bool = False,
) -> Path:
    trial = job_dir / "case-001_attempt001"
    for index, score in enumerate(step_scores, start=1):
        verifier = trial / "steps" / f"step-{index}" / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.json").write_text(json.dumps({"overall": score}), encoding="utf-8")
    result: dict[str, object] = {
        "trial_name": trial.name,
        "task_name": "case-001",
        "verifier_result": {"rewards": {"overall": root_score}},
        "step_results": [
            {"step_name": f"step-{index}", "verifier_result": {"rewards": {"overall": score}}}
            for index, score in enumerate(step_scores, start=1)
        ],
    }
    if failed:
        result["exception_info"] = {
            "exception_type": "TaskFailure",
            "exception_message": "attempt crashed",
        }
    (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
    _write_successful_job_result(job_dir, trial.name)
    return trial


def _write_successful_job_result(job_dir: Path, trial_name: str) -> None:
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_trials": 1,
                    "n_errors": 0,
                    "evals": {
                        "demo": {
                            "n_trials": 1,
                            "n_errors": 0,
                            "reward_stats": {"overall": {"1.0": [trial_name]}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_job_passed_uses_authoritative_multistep_root_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_multistep_attempt_job(job_dir, root_score=0.5, step_scores=(1.0, 0.0))

    assert runner._job_passed(job_dir, 0.75) is False
    rewards = collector._extract_rewards(job_dir)
    assert len(rewards) == 1
    assert collector._average_overall(rewards) == 0.5


def test_job_passed_accepts_passing_authoritative_multistep_root_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_multistep_attempt_job(job_dir, root_score=0.8, step_scores=(0.0, 0.0))

    assert runner._job_passed(job_dir, 0.75) is True


def test_job_passed_rejects_failed_trial_even_with_passing_rewards(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_multistep_attempt_job(job_dir, root_score=1.0, step_scores=(1.0,), failed=True)

    assert runner._job_passed(job_dir, 0.75) is False


def test_job_passed_rejects_failed_job_result_even_with_passing_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_multistep_attempt_job(job_dir, root_score=1.0, step_scores=(1.0,))
    job_result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    job_result["stats"]["n_errors"] = 1
    (job_dir / "result.json").write_text(json.dumps(job_result), encoding="utf-8")

    assert runner._job_passed(job_dir, 0.75) is False


def test_job_passed_preserves_legacy_single_step_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    trial_name = "case-001_attempt001"
    verifier = job_dir / trial_name / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(json.dumps({"overall": 0.9}), encoding="utf-8")
    _write_successful_job_result(job_dir, trial_name)

    assert runner._job_passed(job_dir, 0.75) is True


def test_merge_attempt_jobs_rejects_symlinked_trial_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside-trial"
    outside.mkdir()
    (outside / "host-secret.txt").write_text("secret", encoding="utf-8")
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    trial_link = job_dir / "case-001__trial"
    trial_link.symlink_to(outside, target_is_directory=True)
    aggregate_dir = tmp_path / "aggregate"

    with pytest.raises(ValueError, match=_UNSAFE_LINK):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert not (aggregate_dir / f"{job_dir.name}__{trial_link.name}" / "host-secret.txt").exists()
    assert not (aggregate_dir / "result.json").exists()


def test_merge_attempt_jobs_rejects_symlinked_trial_file(tmp_path: Path) -> None:
    outside = tmp_path / "host-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    job_dir = tmp_path / "attempt-001"
    trial_dir = job_dir / "case-001__trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "artifact.txt").symlink_to(outside)
    aggregate_dir = tmp_path / "aggregate"

    with pytest.raises(ValueError, match=_UNSAFE_LINK):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert not (aggregate_dir / f"{job_dir.name}__{trial_dir.name}" / "artifact.txt").exists()
    assert not (aggregate_dir / "result.json").exists()


def test_merge_attempt_jobs_rejects_nested_directory_link_like_reparse_point(tmp_path: Path) -> None:
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    (outside / "host-secret.txt").write_text("secret", encoding="utf-8")
    job_dir = tmp_path / "attempt-001"
    nested = job_dir / "case-001__trial" / "artifacts"
    nested.mkdir(parents=True)
    linked_dir = nested / "external"
    linked_dir.symlink_to(outside, target_is_directory=True)
    aggregate_dir = tmp_path / "aggregate"

    with pytest.raises(ValueError, match=_UNSAFE_LINK):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    copied_secret = aggregate_dir / f"{job_dir.name}__case-001__trial" / "artifacts" / "external" / "host-secret.txt"
    assert not copied_secret.exists()
    assert not (aggregate_dir / "result.json").exists()


def test_merge_attempt_jobs_preserves_regular_trial_artifacts(tmp_path: Path) -> None:
    job_dir = tmp_path / "attempt-001"
    trial_dir = job_dir / "case-001__trial"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "output.txt").write_text("expected", encoding="utf-8")
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_trials": 1,
                    "n_errors": 0,
                    "evals": {
                        "demo": {
                            "n_trials": 1,
                            "n_errors": 0,
                            "reward_stats": {"reward": {"1.0": [trial_dir.name]}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([job_dir], aggregate_dir)

    merged_trial = aggregate_dir / f"{job_dir.name}__{trial_dir.name}"
    assert (merged_trial / "artifacts" / "output.txt").read_text(encoding="utf-8") == "expected"
    merged_result = json.loads((aggregate_dir / "result.json").read_text(encoding="utf-8"))
    assert merged_result["stats"]["evals"]["demo"]["reward_stats"]["reward"]["1.0"] == [merged_trial.name]


def test_merge_attempt_jobs_ignores_tmpdir_inside_attempt_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "attempt-001"
    trial_dir = job_dir / "case-001__trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "artifact.txt").write_text("expected", encoding="utf-8")
    monkeypatch.setattr(runner.tempfile, "tempdir", str(job_dir))
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert (aggregate_dir / f"{job_dir.name}__{trial_dir.name}" / "artifact.txt").read_text() == "expected"
    assert not list(tmp_path.glob(".aggregate-merge-*"))


def test_merge_attempt_jobs_preserves_existing_aggregate_on_unsafe_source(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    safe_job = tmp_path / "attempt-001"
    safe_trial = safe_job / "case-001__trial"
    safe_trial.mkdir(parents=True)
    (safe_trial / "safe.txt").write_text("staged first", encoding="utf-8")
    unsafe_job = tmp_path / "attempt-002"
    unsafe_trial = unsafe_job / "case-001__trial"
    unsafe_trial.mkdir(parents=True)
    (unsafe_trial / "unsafe").symlink_to(outside, target_is_directory=True)
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    marker = aggregate_dir / "keep.txt"
    marker.write_text("old aggregate", encoding="utf-8")

    with pytest.raises(ValueError, match=_UNSAFE_LINK):
        runner._merge_attempt_jobs([safe_job, unsafe_job], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "old aggregate"
    assert not (aggregate_dir / f"{safe_job.name}__{safe_trial.name}").exists()
    assert not (aggregate_dir / f"{unsafe_job.name}__{unsafe_trial.name}").exists()
    assert not list(tmp_path.glob(".aggregate-merge-*"))


def test_merge_attempt_jobs_preserves_existing_aggregate_when_private_staging_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    marker = aggregate_dir / "keep.txt"
    marker.write_text("old aggregate", encoding="utf-8")

    def fail_private_staging(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected temp failure")

    monkeypatch.setattr(runner.tempfile, "TemporaryDirectory", fail_private_staging)

    with pytest.raises(OSError, match="injected temp failure"):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "old aggregate"


@pytest.mark.parametrize("relationship", ["aggregate-in-job", "job-in-aggregate"])
def test_merge_attempt_jobs_rejects_source_destination_overlap(tmp_path: Path, relationship: str) -> None:
    if relationship == "aggregate-in-job":
        job_dir = tmp_path / "attempt-001"
        job_dir.mkdir()
        aggregate_dir = job_dir / "aggregate"
    else:
        aggregate_dir = tmp_path / "aggregate"
        job_dir = aggregate_dir / "attempt-001"
        job_dir.mkdir(parents=True)
    marker = job_dir / "keep.txt"
    marker.write_text("source", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overlap"):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "source"


def _run(
    monkeypatch: pytest.MonkeyPatch,
    jobs_dir: Path,
    job_name: str = "demo-opencode-with",
    expected_total_trials: int | None = None,
) -> tuple[bool, str]:
    monkeypatch.setattr(
        runner,
        "build_harbor_run_command",
        lambda **_kwargs: [sys.executable, "-c", "pass"],
    )
    kwargs = {"expected_total_trials": expected_total_trials} if expected_total_trials is not None else {}
    return runner._run_harbor(
        dataset=jobs_dir / "dataset",
        agent="opencode",
        job_name=job_name,
        env_mode="docker",
        model="nvidia/openai/gpt-oss-120b",
        jobs_dir=jobs_dir,
        run_env={},
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        **kwargs,
    )


def _write_job_result(jobs_dir: Path, stats: dict[str, object], *, total: int = 1) -> None:
    job_dir = jobs_dir / "demo-opencode-with"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps({"n_total_trials": total, "stats": stats}),
        encoding="utf-8",
    )


def _complete_stats(**overrides: object) -> dict[str, object]:
    stats: dict[str, object] = {
        "n_trials": 1,
        "n_errors": 0,
        "evals": {
            "codex__model___harbor-tasks": {
                "n_trials": 1,
                "n_errors": 0,
                "reward_stats": {"reward": {"1.0": ["case-001__abc"]}},
            }
        },
    }
    stats.update(overrides)
    return stats


def test_run_harbor_rejects_missing_job_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert "result.json" in detail


def test_run_harbor_rejects_zero_trials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_job_result(tmp_path, _complete_stats(n_trials=0, evals={}), total=0)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert "zero trials" in detail


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"n_errors": 1}, "1 errored"),
        ({"n_trials": 0}, "completed 0/1"),
        ({"n_trials": 2}, "completed 2/1"),
    ],
)
def test_run_harbor_rejects_non_successful_trial_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, object],
    expected: str,
) -> None:
    _write_job_result(tmp_path, _complete_stats(**overrides))

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert expected in detail


def test_run_harbor_accepts_complete_successful_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_job_result(tmp_path, _complete_stats())

    assert _run(monkeypatch, tmp_path) == (True, "")


def test_run_harbor_accepts_pinned_harbor_0132_job_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stats = _complete_stats()
    stats.pop("n_trials")
    stats.pop("n_errors")
    stats.update(
        {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": 0,
        }
    )
    _write_job_result(tmp_path, stats)

    assert _run(monkeypatch, tmp_path) == (True, "")


@pytest.mark.parametrize(
    ("counter", "expected"),
    [
        ("n_errored_trials", "1 errored"),
        ("n_running_trials", "1 running"),
        ("n_pending_trials", "1 pending"),
        ("n_cancelled_trials", "1 cancelled"),
    ],
)
def test_run_harbor_rejects_incomplete_pinned_harbor_0132_job_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    counter: str,
    expected: str,
) -> None:
    stats = _complete_stats()
    stats.pop("n_trials")
    stats.pop("n_errors")
    stats.update(
        {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": 0,
        }
    )
    stats[counter] = 1
    _write_job_result(tmp_path, stats)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert expected in detail


def test_job_result_must_match_requested_trial_count(tmp_path: Path) -> None:
    _write_job_result(tmp_path, _complete_stats())

    ok, detail = runner._validate_harbor_job_result(
        tmp_path,
        "demo-opencode-with",
        expected_trials=2,
    )

    assert ok is False
    assert "declared 1 trials; expected 2" in detail


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({"n_trials": True, "n_errors": 0, "evals": {}}, "invalid n_trials"),
        ({"n_trials": 1, "n_errors": -1, "evals": {}}, "invalid n_errors"),
        ({"n_trials": 1, "n_errors": 0, "evals": {}}, "no evaluation statistics"),
        (
            {
                "n_trials": 1,
                "n_errors": 0,
                "evals": {"eval": {"n_trials": 0, "n_errors": 0, "reward_stats": {}}},
            },
            "account for 0/1",
        ),
        (
            {
                "n_trials": 1,
                "n_errors": 0,
                "evals": {"eval": {"n_trials": 1, "n_errors": 0, "reward_stats": {}}},
            },
            "no scored trial names",
        ),
    ],
)
def test_run_harbor_rejects_incomplete_real_harbor_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stats: dict[str, object],
    expected: str,
) -> None:
    _write_job_result(tmp_path, stats)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert expected in detail


def test_run_harbor_rejects_reward_coverage_shortfall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stats = _complete_stats()
    stats["n_trials"] = 2
    stats["evals"] = {
        "eval": {
            "n_trials": 2,
            "n_errors": 0,
            "reward_stats": {"reward": {"1.0": ["case-001__abc"]}},
        }
    }
    _write_job_result(tmp_path, stats, total=2)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert "cover 1/2" in detail


def test_run_harbor_rejects_duplicate_rewarded_trial_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stats = _complete_stats()
    stats["evals"] = {
        "eval": {
            "n_trials": 1,
            "n_errors": 0,
            "reward_stats": {"reward": {"1.0": ["case-001__abc", "case-001__abc"]}},
        }
    }
    _write_job_result(tmp_path, stats)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert "duplicate rewarded trial names" in detail
