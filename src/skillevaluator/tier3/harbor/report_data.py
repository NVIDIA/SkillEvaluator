# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load Harbor artifacts for canonical Tier 3 reporting.

This module intentionally contains no HTML generation. It translates the
on-disk Harbor result layout into data consumed by the shared report adapters.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, LEGACY_METRICS

__all__ = (
    "load_agent_data",
    "load_dataset",
    "load_staged_harbor_dataset",
    "metrics_for_agents",
)


def _metrics_for_rewards(rewards: list[dict[str, Any]]) -> list[str]:
    if any(isinstance(reward.get("security"), int | float) for reward in rewards):
        return list(DEFAULT_METRICS)
    if any(any(isinstance(reward.get(metric), int | float) for metric in LEGACY_METRICS) for reward in rewards):
        return list(LEGACY_METRICS)
    return []


def _skill_evaluator_metrics_for_agent(agent_info: dict[str, Any]) -> list[str]:
    configured = agent_info.get("metrics_with_skill")
    if isinstance(configured, list):
        return [str(metric) for metric in configured]
    scores = agent_info.get("with_skill", {})
    if isinstance(scores, dict) and "security" in scores:
        return list(DEFAULT_METRICS)
    rewards = agent_info.get("rewards", [])
    return _metrics_for_rewards(rewards) if isinstance(rewards, list) else []


def metrics_for_agents(agents: dict[str, dict[str, Any]]) -> list[str]:
    """Return the canonical default or legacy metric set represented by agents."""
    saw_metrics = False
    for info in agents.values():
        metrics = _skill_evaluator_metrics_for_agent(info)
        if metrics:
            saw_metrics = True
        if "security" in metrics:
            return list(DEFAULT_METRICS)
    return list(LEGACY_METRICS) if saw_metrics else []


def _nonnegative_counter(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _condition_status(agent_info: dict[str, Any], condition: str) -> str:
    conditions = agent_info.get("conditions")
    data = conditions.get(condition) if isinstance(conditions, dict) else None
    return str(data.get("execution_status") or "unknown") if isinstance(data, dict) else "unknown"


def load_agent_data(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load per-agent summaries, rewards, lift, and execution coverage."""
    agents: dict[str, dict[str, Any]] = {}
    for agent_dir in sorted(results_dir.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue
        agent_name = agent_dir.name
        agent_info: dict[str, Any] = {"name": agent_name}
        condition_execution: dict[str, dict[str, Any]] = {}

        for variant in ("with-skill", "without-skill"):
            summary = agent_dir / variant / "summary.json"
            if summary.exists():
                try:
                    data = json.loads(summary.read_text(encoding="utf-8"))
                    key = "with_skill" if variant == "with-skill" else "without_skill"
                    agent_info[key] = data.get("scores", data)
                    metric_key = "metrics_with_skill" if variant == "with-skill" else "metrics_without_skill"
                    agent_info[metric_key] = data.get("metrics", [])
                    custom_key = "custom_with_skill" if variant == "with-skill" else "custom_without_skill"
                    if "custom_scores" in data:
                        agent_info[custom_key] = data.get("custom_scores", {})
                    dimension_key = "dimensions_with_skill" if variant == "with-skill" else "dimensions_without_skill"
                    if "dimensions" in data:
                        agent_info[dimension_key] = data.get("dimensions", {})
                    pass_key = "pass_with_skill" if variant == "with-skill" else "pass_without_skill"
                    if "pass_at_k" in data:
                        agent_info[pass_key] = data["pass_at_k"]
                    status = data.get("execution_status")
                    if status not in {"succeeded", "failed", "skipped"}:
                        status = "unknown"
                    errors = data.get("execution_errors")
                    condition_errors = [str(error) for error in errors] if isinstance(errors, list) else []
                    label = "With skill" if variant == "with-skill" else "Without skill"
                    job_failure = data.get("job_failure")
                    if job_failure:
                        condition_errors.append(f"{label} aggregate job: {job_failure}")
                    trial_failures = data.get("trial_failures")
                    if isinstance(trial_failures, list):
                        for failure in trial_failures:
                            if not isinstance(failure, dict):
                                continue
                            trial = failure.get("trial") or "unknown"
                            reason = failure.get("reason") or "Unknown Harbor trial failure"
                            condition_errors.append(f"{label} trial {trial}: {reason}")
                    condition_execution[key] = {
                        "execution_status": status,
                        "execution_errors": condition_errors,
                        "expected_attempts": _nonnegative_counter(data.get("expected_attempts")),
                        "scored_attempts": _nonnegative_counter(data.get("scored_attempts")),
                    }
                    if variant == "with-skill":
                        agent_info["num_trials"] = data.get("num_trials", 0)
                except (json.JSONDecodeError, OSError):
                    pass

        lift_file = agent_dir / "lift.json"
        if lift_file.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                agent_info["lift"] = json.loads(lift_file.read_text(encoding="utf-8"))

        pass_lift_file = agent_dir / "pass_at_k_lift.json"
        if pass_lift_file.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                agent_info["pass_lift"] = json.loads(pass_lift_file.read_text(encoding="utf-8"))

        custom_lift_file = agent_dir / "custom_lift.json"
        if custom_lift_file.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                agent_info["custom_lift"] = json.loads(custom_lift_file.read_text(encoding="utf-8"))

        for variant_key, variant_dir_name in (("rewards", "with-skill"), ("rewards_baseline", "without-skill")):
            trial_list: list[dict[str, Any]] = []
            trials_dir = agent_dir / variant_dir_name / "trials"
            if trials_dir.exists():
                for trial_dir in sorted(trials_dir.iterdir()):
                    if not trial_dir.is_dir():
                        continue
                    reward_file = trial_dir / "reward.json"
                    if not reward_file.exists():
                        continue
                    try:
                        reward = json.loads(reward_file.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    if not reward.get("entry_id"):
                        reward["entry_id"] = trial_dir.name.split("__", 1)[0] if trial_dir.name else "unknown"
                    trajectory_file = trial_dir / "trajectory.json"
                    if trajectory_file.exists():
                        try:
                            trajectory = json.loads(trajectory_file.read_text(encoding="utf-8"))
                            final_metrics = trajectory.get("final_metrics", {})
                            reward["_traj"] = {
                                "steps": len(trajectory.get("steps", [])),
                                "prompt_tokens": final_metrics.get("total_prompt_tokens", 0),
                                "completion_tokens": final_metrics.get("total_completion_tokens", 0),
                                "cached_tokens": final_metrics.get("total_cached_tokens", 0),
                            }
                        except (json.JSONDecodeError, OSError):
                            pass
                    trial_list.append(reward)
            agent_info[variant_key] = trial_list

        if "with_skill" not in agent_info:
            continue

        active_conditions = list(condition_execution.values())
        execution_errors = [
            error for condition in active_conditions for error in condition.get("execution_errors", []) if error
        ]
        if not active_conditions or any(
            condition.get("execution_status") in {"failed", "unknown"} for condition in active_conditions
        ):
            execution_status = "failed" if execution_errors else "unknown"
        elif all(condition.get("execution_status") == "skipped" for condition in active_conditions):
            execution_status = "skipped"
        else:
            execution_status = "succeeded"
        agent_info.update(
            {
                "conditions": condition_execution,
                "execution_status": execution_status,
                "execution_errors": list(dict.fromkeys(execution_errors)),
                "expected_attempts": sum(
                    _nonnegative_counter(condition.get("expected_attempts")) for condition in active_conditions
                ),
                "scored_attempts": sum(
                    _nonnegative_counter(condition.get("scored_attempts")) for condition in active_conditions
                ),
            }
        )

        condition_quality_fields = {
            "with_skill": (
                "with_skill",
                "custom_with_skill",
                "dimensions_with_skill",
                "pass_with_skill",
                "rewards",
            ),
            "without_skill": (
                "without_skill",
                "custom_without_skill",
                "dimensions_without_skill",
                "pass_without_skill",
                "rewards_baseline",
            ),
        }
        for condition, fields in condition_quality_fields.items():
            condition_status = _condition_status(agent_info, condition)
            if condition_status == "succeeded":
                continue
            condition_info = condition_execution.get(condition, {})
            for field in fields:
                if field.startswith("pass_") and condition_status in {"failed", "unknown"}:
                    agent_info[field] = {
                        "attempts_used": _nonnegative_counter(condition_info.get("scored_attempts")),
                        "max_attempts_possible": _nonnegative_counter(condition_info.get("expected_attempts")),
                    }
                else:
                    agent_info[field] = [] if field.startswith("rewards") else {}
        agents[agent_name] = agent_info
    return agents


def load_dataset(skill_path: Path | None) -> list[dict[str, Any]]:
    """Load the first supported Tier 3 dataset from a skill directory."""
    if not skill_path:
        return []
    evals_dir = skill_path / "evals"
    for name in ("evals.json", "evals.jsonl", "evals.yaml", "evals.yml", "dataset.json"):
        candidate = evals_dir / name
        if candidate.exists():
            try:
                from skillevaluator.tier3.dataset_utils import load_dataset_entries

                return load_dataset_entries(candidate)
            except (json.JSONDecodeError, OSError, ValueError):
                pass
    return []


def load_staged_harbor_dataset(results_dir: Path) -> list[dict[str, Any]]:
    """Load and deduplicate dataset entries staged into Harbor task trees."""
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    tasks_dir = results_dir / "_harbor-tasks"
    if not tasks_dir.exists():
        return entries
    for entry_file in sorted(tasks_dir.rglob("tests/entry.json")):
        try:
            entry = json.loads(entry_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        identity = f"id:{entry_id}" if entry_id is not None else f"payload:{json.dumps(entry, sort_keys=True)}"
        if identity in seen:
            continue
        seen.add(identity)
        entries.append(entry)
    return entries
