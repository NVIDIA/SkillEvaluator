# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from skillevaluator.tier3.harbor.artifact_retention import HarborArtifactLifecycle


def test_default_cleanup_removes_jobs_and_tasks(tmp_path: Path) -> None:
    jobs = tmp_path / "_harbor-jobs"
    tasks = tmp_path / "_harbor-tasks"
    jobs.mkdir()
    tasks.mkdir()

    outcome = HarborArtifactLifecycle([jobs, tasks], keep_requested=False).finalize()

    assert outcome.retained is False
    assert outcome.reason == "not_retained"
    assert outcome.warning == ""
    assert not jobs.exists()
    assert not tasks.exists()


def test_explicit_keep_retains_jobs_and_tasks(tmp_path: Path) -> None:
    jobs = tmp_path / "_harbor-jobs"
    tasks = tmp_path / "_harbor-tasks"
    jobs.mkdir()
    tasks.mkdir()

    outcome = HarborArtifactLifecycle([jobs, tasks], keep_requested=True).finalize()

    assert outcome.retained is True
    assert outcome.reason == "explicit_keep"
    assert jobs.is_dir()
    assert tasks.is_dir()


def test_cleanup_failure_reports_actual_retention(monkeypatch, tmp_path: Path) -> None:
    jobs = tmp_path / "_harbor-jobs"
    jobs.mkdir()
    monkeypatch.setattr(
        "skillevaluator.tier3.harbor.artifact_retention.shutil.rmtree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")),
    )

    outcome = HarborArtifactLifecycle([jobs], keep_requested=False).finalize()

    assert outcome.retained is True
    assert outcome.reason == "cleanup_failed"
    assert "busy" in outcome.warning
    assert str(jobs) in outcome.warning


def test_finalize_is_idempotent(tmp_path: Path) -> None:
    jobs = tmp_path / "_harbor-jobs"
    jobs.mkdir()
    lifecycle = HarborArtifactLifecycle([jobs], keep_requested=True)

    first = lifecycle.finalize()
    jobs.rmdir()
    second = lifecycle.finalize()

    assert second is first
    assert second.retained is True


def test_mixed_cleanup_failure_reports_only_remaining_path(monkeypatch, tmp_path: Path) -> None:
    jobs = tmp_path / "_harbor-jobs"
    tasks = tmp_path / "_harbor-tasks"
    jobs.mkdir()
    tasks.mkdir()

    def selective_rmtree(path: Path) -> None:
        if path == jobs:
            raise OSError("jobs busy")
        path.rmdir()

    monkeypatch.setattr("skillevaluator.tier3.harbor.artifact_retention.shutil.rmtree", selective_rmtree)

    outcome = HarborArtifactLifecycle([jobs, tasks], keep_requested=False).finalize()

    assert outcome.retained is True
    assert outcome.reason == "cleanup_failed"
    assert jobs.exists()
    assert not tasks.exists()
    assert str(jobs) in outcome.warning
    assert str(tasks) not in outcome.warning
