# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize NeMo Gym Harbor-environment rollouts into SkillEvaluator trial evidence.

The shared collector reads one job directory per agent and arm. Each Gym rollout
becomes one trial holding the verifier files Gym downloaded from ``/logs/verifier``
and the ATIF trajectory Gym wrote for the verifier, so scoring, lift, pass@k and
reports stay backend-neutral.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_VERIFIER_FILES = ("reward.json", "reward.txt", "skill_evaluator_reward.json", "test-stdout.txt")


def _safe(value: str) -> str:
    return _UNSAFE.sub("-", value).strip("-") or "task"


def read_rollouts(path: Path, errors: list[str] | None = None) -> list[dict[str, Any]]:
    """Read a Gym rollouts JSONL file, skipping blank lines.

    A line Gym was still writing when it was stopped (timeout, OOM, full disk) is
    skipped and reported through ``errors``, so it becomes a missing trial.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    skipped = 0
    # Gym ends each row with "\n"; str.splitlines() would also split on raw U+2028/U+0085 inside strings.
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
    if skipped and errors is not None:
        lost = (
            "those failure reasons were lost"
            if path.name.endswith("_failures.jsonl")
            else "those rollouts were not collected"
        )
        errors.append(f"NeMo Gym wrote {skipped} unreadable line(s) to {path.name}; {lost}")
    return rows


_MISSING_ROLLOUT = "_skillevaluator_missing_rollout"


def read_rollout_failures(rollouts_path: Path, errors: list[str] | None = None) -> dict[str, list[str]]:
    """Map each task folder name to the failure messages Gym recorded in ``<stem>_failures.jsonl``."""
    failures: dict[str, list[str]] = defaultdict(list)
    for row in read_rollouts(rollouts_path.with_name(f"{rollouts_path.stem}_failures.jsonl"), errors):
        task_id = row.get("_ng_task_id")
        if isinstance(task_id, str):
            with contextlib.suppress(ValueError):
                task_id = json.loads(task_id)
        task_name = rollout_task_name({"task_id": task_id})
        message = row.get("error") or row.get("_ng_failure_message") or row.get("_ng_failure_class")
        if task_name and message:
            failures[task_name].append(str(message))
    return dict(failures)


def rollout_task_name(row: dict[str, Any]) -> str | None:
    """Return the task folder name a Gym Harbor rollout ran."""
    task_id = row.get("_ng_task_id", row.get("task_id"))
    if isinstance(task_id, str) and task_id.startswith("{"):
        with contextlib.suppress(ValueError):
            task_id = json.loads(task_id)
    if isinstance(task_id, dict):
        task_id = task_id.get("task_id")
    return task_id if isinstance(task_id, str) and task_id else None


def read_verifier_rewards(verifier_dir: Path) -> dict[str, Any] | None:
    """Return a trial's rewards from ``reward.json``, else a numeric ``reward.txt``, as Harbor reads them."""
    reward_json = verifier_dir / "reward.json"
    if reward_json.is_file() and not reward_json.is_symlink():
        try:
            loaded = json.loads(reward_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return loaded if isinstance(loaded, dict) else None
    reward_txt = verifier_dir / "reward.txt"
    if reward_txt.is_file() and not reward_txt.is_symlink():
        try:
            value = float(reward_txt.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
        return {"reward": value} if math.isfinite(value) else None
    return None


def rollout_failure_text(row: dict[str, Any]) -> str:
    """Return why a Gym rollout failed, or an empty string when its verifier ran."""
    return _failure_text(row)


def _failure_text(row: dict[str, Any]) -> str:
    failure = row.get("failure") or row.get("failure_reason") or row.get("failure_kind")
    if isinstance(failure, dict):
        return str(failure.get("message") or failure.get("failure_kind") or "NeMo Gym episode failed")
    if failure:
        return str(failure)
    if row.get("mask_sample"):
        return "NeMo Gym masked the sample (Gym-side failure)"
    if row.get("verifier_return_code") is None:
        return "SkillEvaluator verifier did not run in the NeMo Gym sandbox"
    return ""


def _copy_verifier_logs(logs_dir: Path, trial_dir: Path) -> None:
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    for name in _VERIFIER_FILES:
        source = logs_dir / name
        if source.is_file() and not source.is_symlink():
            shutil.copyfile(source, verifier_dir / name)
    trajectory = logs_dir / "trajectory.json"
    if trajectory.is_file() and not trajectory.is_symlink():
        (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(trajectory, trial_dir / "agent" / "trajectory.json")


def write_job_from_rollouts(
    rows: list[dict[str, Any]],
    job_dir: Path,
    *,
    eval_name: str,
    task_case_ids: dict[str, str],
    n_attempts: int,
    failures: dict[str, list[str]] | None = None,
) -> list[str]:
    """Write one collector job directory from Gym rollouts and return the run errors.

    ``task_case_ids`` maps each staged task folder name to its evals.json case id.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_name = rollout_task_name(row)
        if task_name not in task_case_ids:
            errors.append(f"NeMo Gym returned a rollout for an unknown task: {task_name!r}")
            continue
        by_task[task_name].append(row)

    rewarded: dict[str, list[str]] = defaultdict(list)
    n_scored = 0
    n_written = 0
    n_errors = 0
    for task_name, case_id in task_case_ids.items():
        task_rows = by_task.get(task_name, [])
        if len(task_rows) < n_attempts:
            reasons = sorted(set((failures or {}).get(task_name, [])))
            detail = f": {'; '.join(reasons)}" if reasons else ""
            errors.append(f"NeMo Gym returned {len(task_rows)}/{n_attempts} rollouts for case {case_id}{detail}")
            # Every expected attempt gets a failed trial, so the collector keeps the arm's scored trials.
            missing = {"failure": f"NeMo Gym returned no rollout{detail}", _MISSING_ROLLOUT: True}
            task_rows = [*task_rows, *([missing] * (n_attempts - len(task_rows)))]
        elif len(task_rows) > n_attempts:
            # Only the planned attempts become trials; the excess is reported so the run is not shown as matching.
            errors.append(
                f"NeMo Gym returned {len(task_rows)}/{n_attempts} rollouts for case {case_id}; "
                f"the {len(task_rows) - n_attempts} extra rollout(s) were not collected"
            )
        for attempt, row in enumerate(task_rows[:n_attempts], start=1):
            trial_name = f"{_safe(task_name)}__gym{attempt:03d}"
            trial_dir = job_dir / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = row.get("verifier_logs_dir")
            if isinstance(logs_dir, str) and Path(logs_dir).is_dir():
                _copy_verifier_logs(Path(logs_dir), trial_dir)
            reward_path = trial_dir / "verifier" / "reward.json"
            failure = _failure_text(row)
            rewards: dict[str, Any] | None = None
            loaded = None if failure else read_verifier_rewards(trial_dir / "verifier")
            if loaded is not None:
                rewards = {"entry_id": case_id, **loaded}
                if reward_path.is_file():
                    reward_path.write_text(json.dumps(rewards, indent=2), encoding="utf-8")
            if rewards is None:
                failure = failure or "the task verifier wrote no reward.json or numeric reward.txt"
                n_errors += 1
                if not row.get(_MISSING_ROLLOUT):
                    errors.append(f"case {case_id} attempt {attempt}: {failure}")
            else:
                n_scored += 1
                score = rewards.get("overall", rewards.get("reward", row.get("reward", 0.0)))
                rewarded[str(score)].append(trial_name)
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "trial_name": trial_name,
                        "task_name": task_name,
                        "execution_backend": "nemo-gym",
                        "episode_id": row.get("episode_id"),
                        "verifier_result": {"rewards": rewards} if rewards else None,
                        "exception_info": None
                        if rewards
                        else {"exception_type": "NeMoGymEpisodeFailure", "exception_message": failure},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            n_written += 1

    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "execution_backend": "nemo-gym",
                "n_total_trials": len(task_case_ids) * n_attempts,
                "stats": {
                    "n_trials": n_written,
                    "n_errors": n_errors,
                    "evals": {
                        eval_name: {
                            "n_trials": n_scored,
                            "n_errors": n_errors,
                            "reward_stats": {"reward": dict(rewarded)},
                        }
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return errors
