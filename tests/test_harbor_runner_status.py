# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor subprocess completion is not sufficient proof of successful trials."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import runner


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
