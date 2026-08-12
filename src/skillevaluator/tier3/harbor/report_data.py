# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load Harbor artifacts for canonical Tier 3 reporting.

This module intentionally contains no HTML generation. It translates the
on-disk Harbor result layout into data consumed by the shared report adapters.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import stat
from collections.abc import Callable, Iterable
from itertools import islice
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, LEGACY_METRICS

logger = logging.getLogger(__name__)

_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 50_000
_MAX_AGENTS = 64
_MAX_AGENT_PATHS_SCANNED = 512
_MAX_TRIALS_PER_CONDITION = 512
_MAX_TRIAL_PATHS_SCANNED = 4096
_MAX_STAGED_TASKS = 4096
_MAX_STAGED_PATHS_SCANNED = 32_768
_MAX_DATASET_RECORDS = 4096
_MAX_DIAGNOSTIC_REASONS = 8
_INVALID_JSON = object()

__all__ = (
    "load_agent_data",
    "load_dataset",
    "load_staged_harbor_dataset",
    "metrics_for_agents",
)


class _JSONLimitError(ValueError):
    def __init__(self, code: str, limit: int) -> None:
        super().__init__(code)
        self.code = code
        self.limit = limit


class _BoundedDataset(list[dict[str, Any]]):
    """List-compatible dataset carrying bounded loader diagnostics."""

    def __init__(self, entries: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
        super().__init__(entries)
        self._report_truncation = {"truncated": True, "reasons": [dict(reason) for reason in diagnostics]}


def _record_truncation(
    diagnostics: list[dict[str, Any]],
    *,
    code: str,
    artifact: str,
    limit: int,
) -> None:
    """Record one bounded, content-free diagnostic and emit it once per scope."""
    reason = {"code": code, "artifact": artifact, "limit": limit}
    if reason in diagnostics or len(diagnostics) >= _MAX_DIAGNOSTIC_REASONS:
        return
    diagnostics.append(reason)
    logger.warning("Tier 3 report loader bounded %s: %s (limit=%d)", artifact, code, limit)


def _attach_truncation(target: dict[str, Any], diagnostics: list[dict[str, Any]]) -> None:
    if not diagnostics:
        return
    existing = target.get("_report_truncation")
    reasons = existing.get("reasons", []) if isinstance(existing, dict) else []
    merged = [*reasons]
    for reason in diagnostics:
        if reason not in merged and len(merged) < _MAX_DIAGNOSTIC_REASONS:
            merged.append(reason)
    target["_report_truncation"] = {"truncated": True, "reasons": merged}


def _dataset_result(entries: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _BoundedDataset(entries, diagnostics) if diagnostics else entries


def _bounded_smallest(
    paths: Iterable[Path],
    limit: int,
    *,
    scan_limit: int,
    predicate: Callable[[Path], bool] | None = None,
) -> tuple[list[Path], bool, bool]:
    """Select deterministic paths while bounding both visits and retained paths."""
    scanned = 0

    def candidates() -> Iterable[Path]:
        nonlocal scanned
        for path in islice(paths, scan_limit + 1):
            scanned += 1
            if scanned > scan_limit:
                break
            if predicate is None or predicate(path):
                yield path

    selected = heapq.nsmallest(limit + 1, candidates(), key=lambda path: path.as_posix())
    return selected[:limit], len(selected) > limit, scanned > scan_limit


def _bounded_staged_entry_files(tasks_dir: Path) -> tuple[list[Path], bool, bool]:
    """Find staged entry files without allowing ``rglob`` to hide unbounded visits."""
    pending = [tasks_dir]
    matches: list[Path] = []
    visited = 0
    scan_truncated = False
    while pending and not scan_truncated:
        directory = pending.pop()
        try:
            with os.scandir(directory) as scanner:
                children = []
                for child in scanner:
                    visited += 1
                    if visited > _MAX_STAGED_PATHS_SCANNED:
                        scan_truncated = True
                        break
                    children.append(child)
        except OSError:
            continue
        for child in sorted(children, key=lambda entry: entry.name, reverse=True):
            try:
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                elif child.name == "entry.json" and directory.name == "tests" and child.is_file(follow_symlinks=False):
                    matches.append(Path(child.path))
            except OSError:
                continue
    selected = heapq.nsmallest(_MAX_STAGED_TASKS + 1, matches, key=lambda path: path.as_posix())
    return selected[:_MAX_STAGED_TASKS], len(selected) > _MAX_STAGED_TASKS, scan_truncated


def _validate_json_tree(value: Any) -> None:
    """Validate decoded JSON iteratively so validation itself cannot recurse."""
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise _JSONLimitError("json_nodes", _MAX_JSON_NODES)
        if not isinstance(current, dict | list):
            continue
        if depth > _MAX_JSON_DEPTH:
            raise _JSONLimitError("json_depth", _MAX_JSON_DEPTH)
        if nodes + len(current) > _MAX_JSON_NODES:
            raise _JSONLimitError("json_nodes", _MAX_JSON_NODES)
        children = current.values() if isinstance(current, dict) else current
        stack.extend((child, depth + 1) for child in children)


def _read_bounded_bytes(
    path: Path,
    diagnostics: list[dict[str, Any]],
    *,
    artifact: str,
) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                _record_truncation(diagnostics, code="json_file_type", artifact=artifact, limit=0)
                return None
            if metadata.st_size > _MAX_JSON_BYTES:
                _record_truncation(
                    diagnostics,
                    code="json_bytes",
                    artifact=artifact,
                    limit=_MAX_JSON_BYTES,
                )
                return None
            raw = stream.read(_MAX_JSON_BYTES + 1)
    except OSError:
        return None
    if len(raw) > _MAX_JSON_BYTES:
        _record_truncation(
            diagnostics,
            code="json_bytes",
            artifact=artifact,
            limit=_MAX_JSON_BYTES,
        )
        return None
    return raw


def _load_bounded_json(
    path: Path,
    diagnostics: list[dict[str, Any]],
    *,
    artifact: str,
) -> Any:
    raw = _read_bounded_bytes(path, diagnostics, artifact=artifact)
    if raw is None:
        return _INVALID_JSON
    return _decode_bounded_json(raw, diagnostics, artifact=artifact)


def _decode_bounded_json(
    raw: bytes,
    diagnostics: list[dict[str, Any]],
    *,
    artifact: str,
    strict_syntax: bool = False,
) -> Any:
    try:
        value = json.loads(raw)
        _validate_json_tree(value)
    except RecursionError:
        _record_truncation(
            diagnostics,
            code="json_depth",
            artifact=artifact,
            limit=_MAX_JSON_DEPTH,
        )
        return _INVALID_JSON
    except _JSONLimitError as exc:
        _record_truncation(diagnostics, code=exc.code, artifact=artifact, limit=exc.limit)
        return _INVALID_JSON
    except json.JSONDecodeError:
        if strict_syntax:
            raise
        return _INVALID_JSON
    except UnicodeDecodeError:
        if strict_syntax:
            raise
        return _INVALID_JSON
    return value


def _bounded_dataset_payload(payload: Any, diagnostics: list[dict[str, Any]]) -> Any:
    """Slice list-like dataset shapes before normalization copies their records."""
    if isinstance(payload, list):
        if len(payload) > _MAX_DATASET_RECORDS:
            _record_truncation(
                diagnostics,
                code="dataset_record_limit",
                artifact="dataset",
                limit=_MAX_DATASET_RECORDS,
            )
            return payload[:_MAX_DATASET_RECORDS]
        return payload
    if not isinstance(payload, dict):
        return payload
    for key in ("evals", "cases"):
        records = payload.get(key)
        if not isinstance(records, list) or len(records) <= _MAX_DATASET_RECORDS:
            continue
        _record_truncation(
            diagnostics,
            code="dataset_record_limit",
            artifact="dataset",
            limit=_MAX_DATASET_RECORDS,
        )
        return {**payload, key: records[:_MAX_DATASET_RECORDS]}
    return payload


def _load_bounded_jsonl(raw: bytes, diagnostics: list[dict[str, Any]]) -> list[Any]:
    records: list[Any] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        value = _decode_bounded_json(
            line,
            diagnostics,
            artifact="dataset_record",
            strict_syntax=True,
        )
        if value is _INVALID_JSON:
            continue
        records.append(value)
        if len(records) > _MAX_DATASET_RECORDS:
            _record_truncation(
                diagnostics,
                code="dataset_record_limit",
                artifact="dataset",
                limit=_MAX_DATASET_RECORDS,
            )
            return records[:_MAX_DATASET_RECORDS]
    return records


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


def load_agent_data(
    results_dir: Path,
    *,
    allow_legacy_missing_status: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load per-agent summaries, rewards, lift, and execution coverage."""
    agents: dict[str, dict[str, Any]] = {}
    selection_diagnostics: list[dict[str, Any]] = []
    try:
        agent_dirs, agents_truncated, agent_scan_truncated = _bounded_smallest(
            results_dir.iterdir(),
            _MAX_AGENTS,
            scan_limit=_MAX_AGENT_PATHS_SCANNED,
            predicate=lambda path: path.is_dir() and not path.name.startswith("_"),
        )
    except OSError:
        return agents
    if agents_truncated:
        _record_truncation(
            selection_diagnostics,
            code="agent_limit",
            artifact="agents",
            limit=_MAX_AGENTS,
        )
    if agent_scan_truncated:
        _record_truncation(
            selection_diagnostics,
            code="agent_scan_limit",
            artifact="agents",
            limit=_MAX_AGENT_PATHS_SCANNED,
        )

    for agent_dir in agent_dirs:
        agent_name = agent_dir.name
        agent_info: dict[str, Any] = {"name": agent_name}
        agent_diagnostics: list[dict[str, Any]] = []
        condition_execution: dict[str, dict[str, Any]] = {}

        for variant in ("with-skill", "without-skill"):
            summary = agent_dir / variant / "summary.json"
            if summary.exists():
                data = _load_bounded_json(summary, agent_diagnostics, artifact="summary")
                if isinstance(data, dict):
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
                    if status is None and allow_legacy_missing_status:
                        status = "succeeded"
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

        lift_file = agent_dir / "lift.json"
        if lift_file.exists():
            lift = _load_bounded_json(lift_file, agent_diagnostics, artifact="lift")
            if lift is not _INVALID_JSON:
                agent_info["lift"] = lift

        pass_lift_file = agent_dir / "pass_at_k_lift.json"
        if pass_lift_file.exists():
            pass_lift = _load_bounded_json(pass_lift_file, agent_diagnostics, artifact="pass_lift")
            if pass_lift is not _INVALID_JSON:
                agent_info["pass_lift"] = pass_lift

        custom_lift_file = agent_dir / "custom_lift.json"
        if custom_lift_file.exists():
            custom_lift = _load_bounded_json(custom_lift_file, agent_diagnostics, artifact="custom_lift")
            if custom_lift is not _INVALID_JSON:
                agent_info["custom_lift"] = custom_lift

        for variant_key, variant_dir_name in (("rewards", "with-skill"), ("rewards_baseline", "without-skill")):
            trial_list: list[dict[str, Any]] = []
            trials_dir = agent_dir / variant_dir_name / "trials"
            if trials_dir.exists():
                try:
                    trial_dirs, trials_truncated, trial_scan_truncated = _bounded_smallest(
                        trials_dir.iterdir(),
                        _MAX_TRIALS_PER_CONDITION,
                        scan_limit=_MAX_TRIAL_PATHS_SCANNED,
                        predicate=lambda path: path.is_dir(),
                    )
                except OSError:
                    trial_dirs, trials_truncated, trial_scan_truncated = [], False, False
                if trials_truncated:
                    _record_truncation(
                        agent_diagnostics,
                        code="trial_limit",
                        artifact=variant_dir_name,
                        limit=_MAX_TRIALS_PER_CONDITION,
                    )
                if trial_scan_truncated:
                    _record_truncation(
                        agent_diagnostics,
                        code="trial_scan_limit",
                        artifact=variant_dir_name,
                        limit=_MAX_TRIAL_PATHS_SCANNED,
                    )
                for trial_dir in trial_dirs:
                    reward_file = trial_dir / "reward.json"
                    if not reward_file.exists():
                        continue
                    reward = _load_bounded_json(reward_file, agent_diagnostics, artifact="reward")
                    if not isinstance(reward, dict):
                        continue
                    if not reward.get("entry_id"):
                        reward["entry_id"] = trial_dir.name.split("__", 1)[0] if trial_dir.name else "unknown"
                    trajectory_file = trial_dir / "trajectory.json"
                    if trajectory_file.exists():
                        trajectory = _load_bounded_json(
                            trajectory_file,
                            agent_diagnostics,
                            artifact="trajectory",
                        )
                        if isinstance(trajectory, dict):
                            final_metrics = trajectory.get("final_metrics", {})
                            if not isinstance(final_metrics, dict):
                                final_metrics = {}
                            steps = trajectory.get("steps", [])
                            reward["_traj"] = {
                                "steps": len(steps) if isinstance(steps, list) else 0,
                                "prompt_tokens": final_metrics.get("total_prompt_tokens", 0),
                                "completion_tokens": final_metrics.get("total_completion_tokens", 0),
                                "cached_tokens": final_metrics.get("total_cached_tokens", 0),
                            }
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
        _attach_truncation(agent_info, agent_diagnostics)
        agents[agent_name] = agent_info
    if selection_diagnostics and agents:
        _attach_truncation(next(iter(agents.values())), selection_diagnostics)
    return agents


def load_dataset(skill_path: Path | None) -> list[dict[str, Any]]:
    """Load the first supported Tier 3 dataset from a skill directory."""
    if not skill_path:
        return []
    evals_dir = skill_path / "evals"
    diagnostics: list[dict[str, Any]] = []
    for name in ("evals.json", "evals.jsonl", "evals.yaml", "evals.yml", "dataset.json"):
        candidate = evals_dir / name
        if candidate.exists():
            try:
                from skillevaluator.tier3.dataset_utils import normalize_dataset_entries

                if candidate.suffix.lower() == ".json":
                    payload = _load_bounded_json(candidate, diagnostics, artifact="dataset")
                    if payload is _INVALID_JSON:
                        continue
                else:
                    raw = _read_bounded_bytes(candidate, diagnostics, artifact="dataset")
                    if raw is None:
                        continue
                    if candidate.suffix.lower() == ".jsonl":
                        payload = _load_bounded_jsonl(raw, diagnostics)
                    else:
                        import yaml

                        payload = yaml.safe_load(raw)
                        try:
                            _validate_json_tree(payload)
                        except _JSONLimitError as exc:
                            _record_truncation(
                                diagnostics,
                                code=exc.code,
                                artifact="dataset",
                                limit=exc.limit,
                            )
                            continue
                entries = normalize_dataset_entries(_bounded_dataset_payload(payload, diagnostics))
                return _dataset_result(entries, diagnostics)
            except (json.JSONDecodeError, OSError, RecursionError, UnicodeDecodeError, ValueError):
                pass
    return _dataset_result([], diagnostics)


def load_staged_harbor_dataset(results_dir: Path) -> list[dict[str, Any]]:
    """Load and deduplicate dataset entries staged into Harbor task trees."""
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    diagnostics: list[dict[str, Any]] = []
    tasks_dir = results_dir / "_harbor-tasks"
    if not tasks_dir.exists():
        return entries
    try:
        entry_files, tasks_truncated, task_scan_truncated = _bounded_staged_entry_files(tasks_dir)
    except OSError:
        return entries
    if tasks_truncated:
        _record_truncation(
            diagnostics,
            code="staged_task_limit",
            artifact="staged_tasks",
            limit=_MAX_STAGED_TASKS,
        )
    if task_scan_truncated:
        _record_truncation(
            diagnostics,
            code="staged_task_scan_limit",
            artifact="staged_tasks",
            limit=_MAX_STAGED_PATHS_SCANNED,
        )
    for entry_file in entry_files:
        entry = _load_bounded_json(entry_file, diagnostics, artifact="staged_entry")
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        identity = f"id:{entry_id}" if entry_id is not None else f"payload:{json.dumps(entry, sort_keys=True)}"
        if identity in seen:
            continue
        seen.add(identity)
        entries.append(entry)
        if len(entries) >= _MAX_DATASET_RECORDS:
            if len(entry_files) > len(entries):
                _record_truncation(
                    diagnostics,
                    code="dataset_record_limit",
                    artifact="staged_dataset",
                    limit=_MAX_DATASET_RECORDS,
                )
            break
    return _dataset_result(entries, diagnostics)
