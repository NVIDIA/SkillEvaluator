# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor subprocess completion is not sufficient proof of successful trials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import runner


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
