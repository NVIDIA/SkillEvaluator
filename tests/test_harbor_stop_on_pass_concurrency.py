# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop-on-pass runs cases concurrently while each case's attempts stay sequential."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("harbor")

from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.result import TrialResult

from skillevaluator.tier3.harbor import runner

_CASES = ("a", "b", "c", "d")
# a and c pass first time, b passes on its second attempt, d never passes.
_EXPECTED_ATTEMPTS = {"a": 1, "b": 2, "c": 1, "d": 3}


def _write_attempt_job(job_dir: Path, *, agent: str, case: str, score: float) -> None:
    trial_name = f"{case}__attempt"
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    task_path = str(job_dir.parent / "tasks" / case)
    trial = TrialResult.model_validate(
        {
            "id": uuid4(),
            "task_name": case,
            "trial_name": trial_name,
            "trial_uri": trial_dir.as_uri(),
            "task_id": {"path": task_path},
            "task_checksum": "stop-on-pass-concurrency",
            "config": {"task": {"path": task_path}, "trial_name": trial_name, "trials_dir": str(job_dir)},
            "agent_info": {"name": agent, "version": "test", "model_info": {"name": "test-model"}},
            "agent_result": {},
            "verifier_result": {"rewards": {"overall": score}},
            "started_at": now,
            "finished_at": now,
        }
    )
    (trial_dir / "result.json").write_text(trial.model_dump_json(indent=2), encoding="utf-8")
    job = JobResult(
        id=uuid4(),
        started_at=now,
        updated_at=now,
        finished_at=now,
        n_total_trials=1,
        stats=JobStats.from_trial_results([trial], n_total_trials=1),
        trial_results=[trial],
    )
    (job_dir / "result.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")


@pytest.mark.parametrize("concurrency", [1, 2, 4])
def test_stop_on_pass_runs_cases_concurrently_with_serial_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    concurrency: int,
) -> None:
    # Fails if cases are serialized (the rendezvous times out), if the pool
    # exceeds n_concurrent, or if a case retries in parallel or after a pass.
    rendezvous = threading.Barrier(concurrency)
    lock = threading.Lock()
    active: set[tuple[str, str]] = set()
    peak = 0
    attempts: dict[tuple[str, str], int] = {}
    launches: list[tuple[str, str, str]] = []
    rendezvous_failed: list[tuple[str, str]] = []
    overlapping_attempts: list[tuple[str, str]] = []
    jobs_dir = tmp_path / "jobs"

    def fake_run_harbor(**kwargs: object) -> tuple[bool, str]:
        nonlocal peak
        dataset = kwargs["dataset"]
        assert isinstance(dataset, Path)
        assert kwargs["n_attempts"] == kwargs["n_concurrent"] == kwargs["expected_trials"] == 1
        include = kwargs["include_task_names"]
        assert isinstance(include, list) and len(include) == 1
        case = include[0]
        key = (dataset.name, case)
        with lock:
            if key in active:
                overlapping_attempts.append(key)
            active.add(key)
            peak = max(peak, len(active))
            attempts[key] = attempts.get(key, 0) + 1
            attempt = attempts[key]
            launches.append((dataset.name, case, str(kwargs["job_name"])))
        if attempt == 1:
            try:
                rendezvous.wait(timeout=10)
            except threading.BrokenBarrierError:
                rendezvous_failed.append(key)
            # Finish later cases first so completion order differs from dataset order.
            time.sleep(0.02 * (len(_CASES) - _CASES.index(case)))
        passed = case in ("a", "c") or (case == "b" and attempt == 2)
        _write_attempt_job(
            jobs_dir / str(kwargs["job_name"]),
            agent=str(kwargs["agent"]),
            case=case,
            score=1.0 if passed else 0.0,
        )
        with lock:
            active.discard(key)
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_run_harbor)

    errors = runner._run_agent_pair(
        skill_name="demo",
        agent="opencode",
        model="test-model",
        env_mode="docker",
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        jobs_dir=jobs_dir,
        run_env={},
        n_attempts=3,
        n_concurrent=concurrency,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=len(_CASES) * 3,
        stop_on_pass=True,
        pass_threshold=0.5,
        task_names=list(_CASES),
    )

    assert errors == []
    assert rendezvous_failed == []
    assert overlapping_attempts == []
    assert peak == concurrency
    assert not active
    # Conditions stay sequential: every with-skill launch precedes the baseline.
    variants = [variant for variant, _case, _job in launches]
    assert variants == ["with"] * 7 + ["without"] * 7
    for variant in ("with", "without"):
        assert {case: attempts[(variant, case)] for case in _CASES} == _EXPECTED_ATTEMPTS
        for case in _CASES:
            case_jobs = [
                job for launched, launched_case, job in launches if (launched, launched_case) == (variant, case)
            ]
            assert case_jobs == [
                f"demo-opencode-{variant}-{case}-attempt{attempt:03d}"
                for attempt in range(1, _EXPECTED_ATTEMPTS[case] + 1)
            ]
        aggregate = json.loads((jobs_dir / f"demo-opencode-{variant}" / "result.json").read_text(encoding="utf-8"))
        assert aggregate["n_total_trials"] == 7
        # Merged in dataset order, independent of which case finished first.
        assert [trial["task_name"] for trial in aggregate["trial_results"]] == [
            case for case in _CASES for _attempt in range(_EXPECTED_ATTEMPTS[case])
        ]


def test_stop_on_pass_failed_attempts_retry_only_their_case_in_dataset_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    launched: list[str] = []
    lock = threading.Lock()

    def fake_run_harbor(**kwargs: object) -> tuple[bool, str]:
        job_name = str(kwargs["job_name"])
        include = kwargs["include_task_names"]
        assert isinstance(include, list)
        case = include[0]
        with lock:
            launched.append(job_name)
        if case in ("a", "c") and job_name.endswith("-attempt001"):
            if case == "a":
                # Report the later case's failure first.
                time.sleep(0.05)
            return False, "harbor run exited 1"
        _write_attempt_job(jobs_dir / job_name, agent="opencode", case=case, score=1.0)
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_run_harbor)

    errors = runner._run_stop_on_pass_variant(
        skill_name="demo",
        agent="opencode",
        variant="with",
        dataset=tmp_path / "with",
        task_names=["a", "b", "c"],
        env_mode="docker",
        model="test-model",
        jobs_dir=jobs_dir,
        run_env={},
        n_attempts=3,
        pass_threshold=0.5,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        n_concurrent=3,
    )

    assert errors == [
        "opencode with-skill Harbor run failed: a attempt 1: harbor run exited 1",
        "opencode with-skill Harbor run failed: c attempt 1: harbor run exited 1",
    ]
    assert sorted(launched) == [
        "demo-opencode-with-a-attempt001",
        "demo-opencode-with-a-attempt002",
        "demo-opencode-with-b-attempt001",
        "demo-opencode-with-c-attempt001",
        "demo-opencode-with-c-attempt002",
    ]


def test_stop_on_pass_unexpected_case_error_stops_queued_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launched: list[str] = []
    merged: list[object] = []

    def fake_run_harbor(**kwargs: object) -> tuple[bool, str]:
        include = kwargs["include_task_names"]
        assert isinstance(include, list)
        launched.append(include[0])
        if include[0] == "a":
            raise KeyError("unexpected launcher failure")
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_run_harbor)
    monkeypatch.setattr(runner, "_job_passed", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", lambda *args, **_kwargs: merged.append(args))

    with pytest.raises(KeyError, match="unexpected launcher failure"):
        runner._run_stop_on_pass_variant(
            skill_name="demo",
            agent="opencode",
            variant="with",
            dataset=tmp_path / "with",
            task_names=["a", "b", "c"],
            env_mode="docker",
            model="test-model",
            jobs_dir=tmp_path / "jobs",
            run_env={},
            n_attempts=3,
            pass_threshold=0.5,
            timeout_multiplier=1.0,
            override_cpus=None,
            override_memory_mb=None,
            override_storage_mb=None,
            n_concurrent=1,
        )

    # As in the former serial loop, no case launches after the failure and no
    # partial aggregate is merged.
    assert launched == ["a"]
    assert merged == []
