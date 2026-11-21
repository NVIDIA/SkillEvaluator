# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize NeMo Gym Harbor-environment rollouts into SkillEvaluator trial evidence.

The shared collector reads one job directory per agent and arm. Each Gym rollout
becomes one trial holding the verifier files Gym downloaded from ``/logs/verifier``
and the ATIF trajectory Gym wrote for the verifier, so scoring, lift, pass@k and
reports stay backend-neutral.

A multi-step episode reports each step on its ordered ``turns`` and leaves the
top-level verifier fields unset. It becomes Harbor's multi-step trial layout:
``step_results`` with one ``steps/<name>/`` folder per step, and a root reward
rolled up from the step reward files by Gym's ``turn_reward_strategy``.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_VERIFIER_FILES = ("reward.json", "reward.txt", "skill_evaluator_reward.json", "custom_reward.json", "test-stdout.txt")
_TURN_REWARD_STRATEGIES = ("mean", "final")
_TURN_METADATA = ("turn_sessions", "turn_reward_strategy", "stopped_after_turn", "turns_without_verifier_result")
_NO_REWARDS = "the task verifier wrote no reward.json or numeric reward.txt"


def _safe(value: str) -> str:
    return _UNSAFE.sub("-", value) or "task"


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


def rollout_rewards(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return a Gym rollout's rewards read in place from its verifier logs, or ``None`` when it is unscored.

    A multi-step episode is rolled up from its turns with the rules the result normalizer applies.
    """
    turns = row.get("turns")
    if turns is None:
        return _logged_rewards(row)[0]
    names = _turn_names(turns)
    if names is None:
        return None
    return _episode_rewards(row, names, [_logged_rewards(turn) for turn in turns])[0]


def _logged_rewards(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    failure = _failure_text(row)
    if failure:
        return None, failure
    logs_dir = row.get("verifier_logs_dir")
    rewards = read_verifier_rewards(Path(logs_dir)) if isinstance(logs_dir, str) and logs_dir else None
    if rewards is None:
        return None, _NO_REWARDS
    return rewards, ""


def _failure_text(row: dict[str, Any]) -> str:
    failure = _reported_failure_text(row)
    if not failure and row.get("verifier_return_code") is None:
        return "SkillEvaluator verifier did not run in the NeMo Gym sandbox"
    return failure


def _reported_failure_text(row: dict[str, Any]) -> str:
    failure = row.get("failure") or row.get("failure_reason") or row.get("failure_kind")
    if isinstance(failure, dict):
        return str(failure.get("message") or failure.get("failure_kind") or "NeMo Gym episode failed")
    if failure:
        return str(failure)
    if row.get("mask_sample"):
        return "NeMo Gym masked the sample (Gym-side failure)"
    return ""


def _verified_rewards(verifier_dir: Path, case_id: str) -> tuple[dict[str, Any] | None, str]:
    """Return the verifier's rewards stamped with ``case_id``, or why they cannot be scored."""
    loaded = read_verifier_rewards(verifier_dir)
    if loaded is None:
        return None, _NO_REWARDS
    loaded_entry_id = loaded.get("entry_id")
    if loaded_entry_id is not None and str(loaded_entry_id) != str(case_id):
        return None, f"the task verifier reported case {str(loaded_entry_id)!r} for staged case {str(case_id)!r}"
    rewards = {**loaded, "entry_id": case_id}
    reward_path = verifier_dir / "reward.json"
    if reward_path.is_file():
        reward_path.write_text(json.dumps(rewards, indent=2), encoding="utf-8")
    return rewards, ""


def _turn_names(turns: Any) -> list[str] | None:
    """Return the turns' step names when each can be its own ``steps/<name>/`` folder, else ``None``."""
    from skillevaluator.tier3.harbor.collector import _safe_trial_path_component

    if not isinstance(turns, list) or not turns:
        return None
    names: list[str] = []
    keys: set[str] = set()
    for turn in turns:
        name = turn.get("name") if isinstance(turn, dict) else None
        if not isinstance(name, str) or _safe_trial_path_component(name) != name:
            return None
        # Case-insensitive file systems would otherwise merge two steps' evidence into one folder.
        key = unicodedata.normalize("NFC", name).casefold()
        if key in keys:
            return None
        names.append(name)
        keys.add(key)
    return names


def _reward_number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _rolled_up_rewards(step_rewards: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    """Roll up the steps that ran as Harbor does: the last step's rewards, or each key's mean, a missing key as 0."""
    if strategy == "final":
        return dict(step_rewards[-1])
    keys = sorted(
        {key for rewards in step_rewards for key, value in rewards.items() if _reward_number(value) is not None}
    )
    return {
        key: sum(_reward_number(rewards.get(key)) or 0.0 for rewards in step_rewards) / len(step_rewards)
        for key in keys
    }


def _episode_rewards(
    row: dict[str, Any], names: list[str], steps: list[tuple[dict[str, Any] | None, str]]
) -> tuple[dict[str, Any] | None, str]:
    """Return a multi-step episode's rolled-up rewards from each step's ``(rewards, failure)``, or why it is unscored.

    Every step must have run its verifier and written rewards for the episode to be scored. A masked or
    failed step, a step without a verifier result, or an episode Gym marked failed leaves the attempt
    unscored. Gym's ``mean`` counts a step without a verifier result as 0 and Harbor's leaves it out; the
    adapter applies neither, so an ungraded step never becomes a score. An episode that stopped after a
    step missed its ``min_reward`` is scored over the steps that ran, as on Harbor.
    """
    failure = next(
        (
            f"step {name!r}: {step_failure}"
            for name, (rewards, step_failure) in zip(names, steps, strict=True)
            if rewards is None
        ),
        "",
    )
    strategy = row.get("turn_reward_strategy")
    stopped_after = row.get("stopped_after_turn")
    failure = failure or _reported_failure_text(row)
    if not failure and strategy not in _TURN_REWARD_STRATEGIES:
        failure = f"NeMo Gym reported an unsupported turn_reward_strategy {strategy!r}"
    if not failure and stopped_after is not None and stopped_after != names[-1]:
        failure = f"NeMo Gym stopped after turn {stopped_after!r}, which is not its last turn"
    if failure:
        return None, failure
    return _rolled_up_rewards([rewards for rewards, _failure in steps if rewards is not None], strategy), ""


def _multi_step_outcome(
    row: dict[str, Any], trial_dir: Path, case_id: str
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Write a multi-step episode's steps and return its rewards, why it is unscored, and its result fields.

    An unscored episode keeps each step's evidence.
    """
    turns = row.get("turns")
    fields: dict[str, Any] = {"nemo_gym": {key: row.get(key) for key in _TURN_METADATA}}
    names = _turn_names(turns)
    if names is None:
        return None, "NeMo Gym returned malformed multi-step turns", fields
    # Gym exports each turn's own input and new output under the episode's one session id, even under
    # turn_sessions "continue", so no turn holds an earlier one's steps for Harbor's resume merge to strip.
    fields["config"] = {"agent": {"resume_trajectory": False}}
    steps: list[tuple[dict[str, Any] | None, str]] = []
    for name, turn in zip(names, turns, strict=True):
        step_dir = trial_dir / "steps" / name
        logs_dir = turn.get("verifier_logs_dir")
        if isinstance(logs_dir, str) and Path(logs_dir).is_dir():
            _copy_verifier_logs(Path(logs_dir), step_dir)
        step_failure = _failure_text(turn)
        steps.append(_verified_rewards(step_dir / "verifier", case_id) if not step_failure else (None, step_failure))
    fields["step_results"] = [
        {
            "step_name": name,
            "verifier_result": {"rewards": rewards} if rewards else None,
            "exception_info": {"exception_type": "NeMoGymStepFailure", "exception_message": step_failure}
            if step_failure
            else None,
        }
        for name, (rewards, step_failure) in zip(names, steps, strict=True)
    ]
    rewards, failure = _episode_rewards(row, names, steps)
    return ({**rewards, "entry_id": case_id} if rewards is not None else None), failure, fields


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
            rewards: dict[str, Any] | None = None
            step_fields: dict[str, Any] = {}
            if row.get("turns") is not None:
                rewards, failure, step_fields = _multi_step_outcome(row, trial_dir, case_id)
            else:
                logs_dir = row.get("verifier_logs_dir")
                if isinstance(logs_dir, str) and Path(logs_dir).is_dir():
                    _copy_verifier_logs(Path(logs_dir), trial_dir)
                failure = _failure_text(row)
                if not failure:
                    rewards, failure = _verified_rewards(trial_dir / "verifier", case_id)
            if rewards is None:
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
                        **step_fields,
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
