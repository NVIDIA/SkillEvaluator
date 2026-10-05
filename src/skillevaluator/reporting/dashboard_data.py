# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only normalization of retained evaluations for the comparison dashboard.

Usage totals require a measurement for every trial. Missing measurements stay
unknown, and differences require identical case/attempt coverage in both arms.
Agent execution time excludes environment setup and verifier execution.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_USAGE = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "duration_seconds", "cost_usd")
_IDENTITY = ("source", "run_id", "skill", "model", "harness", "features")
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_FILES = 1000
_MAX_PATHS = 20_000


@dataclass
class DashboardData:
    rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _measurement(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _features(*owners: dict[str, Any]) -> str:
    for owner in owners:
        metadata = _dict(owner.get("dashboard_metadata"))
        features = metadata.get("harness_features")
        if isinstance(features, str) and features.strip():
            return features.strip()
        if isinstance(features, list) and all(isinstance(item, str) for item in features):
            return ", ".join(sorted(set(features))) or "none (recorded)"
        if isinstance(features, dict):
            try:
                return json.dumps(features, sort_keys=True, ensure_ascii=False, allow_nan=False)
            except (ValueError, TypeError, RecursionError):
                return "unknown"
    return "unknown"


def _payloads(value: Any, depth: int = 0):
    """Prefer authoritative per-validator results over a top-level Tier 3 mirror."""
    if depth > 32:
        return
    if isinstance(value, list):
        for item in value:
            yield from _payloads(item, depth + 1)
        return
    if not isinstance(value, dict):
        return
    if isinstance(value.get("results"), list):
        for item in value["results"]:
            yield from _payloads(item, depth + 1)
        return
    metadata = _dict(value.get("metadata"))
    for candidate in (metadata.get("agent_eval"), value.get("tier3"), value.get("agent_eval")):
        if isinstance(candidate, dict):
            yield candidate
            return
    if isinstance(value.get("agents"), dict):
        yield value
        return
    # Embedded catalog documents are supported; path references are resolved by
    # load_dashboard_path, never by the parser for uploaded JSON.
    if isinstance(value.get("skills"), list):
        for item in value["skills"]:
            yield from _payloads(item, depth + 1)


def _trial_keys(trials: list[dict[str, Any]]) -> tuple[tuple[str, str], ...] | None:
    counts: Counter[str] = Counter()
    keys = []
    implicit: set[str] = set()
    seen_trials: dict[str, str] = {}
    for trial in trials:
        entry = trial.get("entry_id")
        if not isinstance(entry, str) or not entry:
            return None
        trial_id = str(trial.get("trial_id") or "")
        if trial_id and trial_id in seen_trials:
            if seen_trials[trial_id] != entry:
                return None
            # Fallback multi-step rewards share one physical attempt identity.
            continue
        if trial_id:
            seen_trials[trial_id] = entry
        counts[entry] += 1
        attempt = trial.get("attempt") or trial.get("attempt_index")
        # Native canonical reports omit attempt numbers, but Harbor trial IDs
        # carry __attemptN when attempts are repeated.
        match = re.search(r"(?:__|_)attempt[_-]?(\d+)(?:__|$)", trial_id)
        if attempt is None and match:
            attempt = int(match.group(1))
        if attempt is None:
            implicit.add(entry)
        keys.append((entry, str(attempt) if attempt is not None else f"ordinal:{counts[entry]}"))
    if any(counts[entry] > 1 for entry in implicit):
        return None
    return tuple(sorted(keys))


def _usage(trial: dict[str, Any]) -> dict[str, float | None]:
    tokens = _dict(trial.get("tokens"))
    prompt = _measurement(tokens.get("prompt"))
    completion = _measurement(tokens.get("completion"))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if prompt is not None and completion is not None else None,
        "cached_tokens": _measurement(tokens.get("cached")),
        "duration_seconds": _measurement(trial.get("duration_seconds")),
        "cost_usd": _measurement(trial.get("cost_usd")),
    }


def _set_usage(row: dict[str, Any], usages: list[dict[str, Any]], expected: int | None = None) -> None:
    expected = row["trials"] if expected is None else expected
    row["_usage_coverage"] = {}
    for metric in _USAGE:
        values = [_measurement(usage.get(metric)) for usage in usages]
        available = sum(value is not None for value in values)
        row["_usage_coverage"][metric] = f"{available}/{expected}"
        row[metric] = (
            _measurement(sum(value for value in values if value is not None))
            if values and available == len(values) == expected
            else None
        )
    eligibility = "complete" if row["_complete"] else "incomplete or unknown trial coverage"
    row["coverage"] = (
        f"{row['trials']} trials ({eligibility}); tokens {row['_usage_coverage']['total_tokens']}, "
        f"time {row['_usage_coverage']['duration_seconds']}, cost {row['_usage_coverage']['cost_usd']}"
    )


def parse_dashboard_report(payload: Any, source: str) -> DashboardData:
    """Normalize canonical Tier 3, validate-report, and embedded catalog JSON."""
    output = DashboardData()
    for report in _payloads(payload):
        summary = _dict(report.get("summary"))
        provenance = _dict(report.get("provenance"))
        run_id = str(report.get("run_id") or summary.get("run_id") or "")
        if not run_id and provenance.get("run_dir"):
            run_id = Path(str(provenance["run_dir"])).name
        if not run_id:
            run_id = str(report.get("evaluated_at") or summary.get("evaluated_at") or "unknown")
        truncation = _dict(report.get("report_truncation")) or _dict(report.get("truncation"))
        if truncation.get("truncated"):
            output.warnings.append(f"{source}: report details are truncated; comparisons are unavailable.")
        for name, value in _dict(report.get("agents")).items():
            agent = _dict(value)
            if not agent:
                continue
            conditions = _dict(agent.get("conditions"))
            for condition, trial_field, score_field, count_field in (
                ("with_skill", "trials", "with_skill", "num_trials"),
                ("without_skill", "trials_baseline", "baseline", "num_trials_baseline"),
                ("adverse", "trials_adverse", "adverse", "num_trials_adverse"),
            ):
                details = _dict(conditions.get(condition))
                raw_trials = agent.get(trial_field)
                if condition != "with_skill" and not details and not raw_trials and agent.get(score_field) is None:
                    continue
                trials = (
                    [trial for trial in (raw_trials or []) if isinstance(trial, dict)]
                    if isinstance(raw_trials, list)
                    else []
                )
                status = details.get("execution_status") or agent.get("execution_status") or "unknown"
                if not isinstance(status, str) or status not in {"succeeded", "failed", "unknown", "skipped"}:
                    status = "unknown"
                declared = _count(agent.get(count_field))
                if declared is None:
                    declared = len(trials)
                expected = _count(details.get("expected_attempts"))
                scored = _count(details.get("scored_attempts"))
                keys = _trial_keys(trials)
                logical_count = len(keys) if keys is not None else declared
                complete = (
                    status == "succeeded"
                    and declared > 0
                    and len(trials) == declared
                    and keys is not None
                    and len(set(keys)) == len(keys)
                    and not truncation.get("truncated")
                    and (scored is None or scored == logical_count)
                    and (expected is None or scored == expected)
                )
                row = {
                    "source": source,
                    "run_id": run_id,
                    "skill": str(report.get("skill_name") or summary.get("skill_name") or "unknown"),
                    "model": str(agent.get("model") or "unknown"),
                    "harness": str(agent.get("harness") or agent.get("name") or name),
                    "features": _features(details, agent, report),
                    "condition": condition,
                    "status": status,
                    "trials": logical_count,
                    "score": _number(agent.get(score_field)) if complete else None,
                    "pass_rate": _number(_dict(_dict(agent.get("pass_at_k")).get(condition)).get("rate"))
                    if complete
                    else None,
                    "_complete": complete,
                    "_trial_keys": keys,
                    "_dataset_digest": report.get("dataset_digest") or summary.get("dataset_digest"),
                    "_attempt_policy": report.get("attempt_policy"),
                }
                _set_usage(row, [_usage(trial) for trial in trials], expected=declared)
                output.rows.append(row)
    if not output.rows:
        output.warnings.append(f"{source}: no supported Tier 3 agent results found.")
    return output


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute with-minus-without differences only within an identical run."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("condition") in {"with_skill", "without_skill"}:
            groups.setdefault(tuple(row.get(key) for key in _IDENTITY), []).append(row)
    comparisons = []
    for identity, arms in groups.items():
        with_rows = [row for row in arms if row["condition"] == "with_skill"]
        without_rows = [row for row in arms if row["condition"] == "without_skill"]
        result = dict(zip(_IDENTITY, identity, strict=True))
        status = "missing condition"
        paired = False
        if len(with_rows) > 1 or len(without_rows) > 1:
            status = "ambiguous duplicate conditions"
        elif with_rows and without_rows:
            with_row, without_row = with_rows[0], without_rows[0]
            paired = (
                with_row.get("_complete")
                and without_row.get("_complete")
                and with_row.get("_trial_keys") == without_row.get("_trial_keys")
                and with_row.get("_dataset_digest") == without_row.get("_dataset_digest")
                and with_row.get("_attempt_policy") == without_row.get("_attempt_policy")
            )
            status = "comparable" if paired else "incomplete or mismatched coverage"
        result["comparison_status"] = status
        result["coverage"] = (
            f"{with_rows[0]['trials']} with / {without_rows[0]['trials']} without"
            if with_rows and without_rows
            else "missing condition"
        )
        for metric in ("score", "pass_rate", *_USAGE):
            with_value = _number(with_rows[0].get(metric)) if len(with_rows) == 1 else None
            without_value = _number(without_rows[0].get(metric)) if len(without_rows) == 1 else None
            result[f"with_{metric}"] = with_value
            result[f"without_{metric}"] = without_value
            available = paired and with_value is not None and without_value is not None
            result[f"{metric}_delta"] = _number(with_value - without_value) if available else None
            if metric in _USAGE:
                result[f"{metric}_percent"] = (
                    _number(100 * (with_value - without_value) / without_value)
                    if available and without_value != 0
                    else None
                )
        comparisons.append(result)
    return comparisons


def _read_json(path: Path) -> Any:
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError("JSON must be a regular file no larger than 16 MiB")
    with path.open("rb") as stream:
        data = stream.read(_MAX_FILE_BYTES + 1)
    if len(data) > _MAX_FILE_BYTES:
        raise ValueError("JSON exceeds 16 MiB")
    return json.loads(data)


def _duration(result: dict[str, Any]) -> float | None:
    execution = _dict(result.get("agent_execution"))
    try:
        started = datetime.fromisoformat(str(execution["started_at"]))
        finished = datetime.fromisoformat(str(execution["finished_at"]))
        return _measurement((finished - started).total_seconds())
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _native_usage(trial_dir: Path) -> dict[str, Any]:
    trajectory = _dict(_read_json(trial_dir / "trajectory.json")) if (trial_dir / "trajectory.json").is_file() else {}
    final = _dict(trajectory.get("final_metrics"))
    result = _dict(_read_json(trial_dir / "result.json")) if (trial_dir / "result.json").is_file() else {}
    agent_result = _dict(result.get("agent_result"))
    prompt = _measurement(final.get("total_prompt_tokens"))
    completion = _measurement(final.get("total_completion_tokens"))
    cached = _measurement(final.get("total_cached_tokens"))
    if prompt is None:
        prompt = _measurement(agent_result.get("n_input_tokens"))
    if completion is None:
        completion = _measurement(agent_result.get("n_output_tokens"))
    if cached is None:
        cached = _measurement(agent_result.get("n_cache_tokens"))
    # Constituent steps own their measured cost and execution time. The root
    # may also mirror a step, so never add both the root and constituent values.
    steps = result.get("step_results")
    constituents = steps if isinstance(steps, list) and steps else [result]
    if isinstance(steps, list) and steps:

        def step_total(key: str) -> float | None:
            values = [_measurement(_dict(_dict(step).get("agent_result")).get(key)) for step in steps]
            return _measurement(sum(values)) if all(value is not None for value in values) else None

        prompt = step_total("n_input_tokens")
        completion = step_total("n_output_tokens")
        cached = step_total("n_cache_tokens")
    durations = [_duration(_dict(step)) for step in constituents]
    costs = [_measurement(_dict(_dict(step).get("agent_result")).get("cost_usd")) for step in constituents]
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if prompt is not None and completion is not None else None,
        "cached_tokens": cached,
        "duration_seconds": _measurement(sum(durations))
        if durations and all(value is not None for value in durations)
        else None,
        "cost_usd": _measurement(sum(costs)) if costs and all(value is not None for value in costs) else None,
    }


def _load_native(path: Path) -> DashboardData:
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory

    engine_result = _dict(_read_json(path / "result.json"))
    skill = str(engine_result.get("skill_name") or "unknown")
    result = agent_eval_result_from_directory(Path(skill), path, engine_result=engine_result, use_llm_judge=False)
    if result is None:
        return DashboardData(warnings=[f"{path}: no supported retained agent summaries found."])
    canonical = result.metadata["agent_eval"]
    canonical["run_id"] = engine_result.get("run_id") or path.name
    run_config = _dict(engine_result.get("run_config"))
    if not run_config and (path / "run_config.json").is_file():
        run_config = _dict(_read_json(path / "run_config.json"))
    canonical["dashboard_metadata"] = run_config.get("dashboard_metadata")
    output = parse_dashboard_report(canonical, str(path))
    for row in output.rows:
        # The report's agent keys identify repeated harness occurrences on disk.
        agents = _dict(canonical.get("agents"))
        name = next(
            (key for key, agent in agents.items() if _dict(agent).get("name") == row["harness"]), row["harness"]
        )
        agent_config = _dict(_dict(run_config.get("agents")).get(name))
        row["features"] = _features(agent_config, run_config)
        row["harness"] = str(agent_config.get("agent") or row["harness"])
        variant = {"with_skill": "with-skill", "without_skill": "without-skill", "adverse": "adverse"}[row["condition"]]
        summary_path = path / name / variant / "summary.json"
        if summary_path.is_file():
            condition_summary = _dict(_read_json(summary_path))
            row["model"] = str(condition_summary.get("model") or row["model"])
            row["features"] = _features(condition_summary, agent_config, run_config)
        trials_dir = path / name / variant / "trials"
        usages_by_trial: dict[str, dict[str, Any]] = {}
        if trials_dir.is_dir():
            for trial_dir in sorted(trials_dir.iterdir())[:513]:
                if not trial_dir.is_dir() or trial_dir.is_symlink():
                    continue
                try:
                    reward = _dict(_read_json(trial_dir / "reward.json"))
                    physical_id = str(reward.get("trial_id") or trial_dir.name)
                    if physical_id not in usages_by_trial:
                        usages_by_trial[physical_id] = _native_usage(trial_dir)
                except (OSError, ValueError, RecursionError) as exc:
                    output.warnings.append(f"{trial_dir}: cannot read usage: {exc}")
                    usages_by_trial[trial_dir.name] = {}
        _set_usage(row, list(usages_by_trial.values()))
    return output


def _load_report_file(path: Path) -> DashboardData:
    payload = _read_json(path)
    data = parse_dashboard_report(payload, str(path))
    if data.rows or not isinstance(payload, dict) or not isinstance(payload.get("skills"), list):
        return data
    output = DashboardData()
    for entry in payload["skills"][:_MAX_FILES]:
        entry = _dict(entry)
        report_dir, report_name = entry.get("report_dir"), entry.get("json_report")
        if not isinstance(report_dir, str) or not isinstance(report_name, str):
            continue
        try:
            report_path = (path.parent / report_dir / report_name).resolve()
            report_path.relative_to(path.parent.resolve())
            report = parse_dashboard_report(_read_json(report_path), str(report_path))
        except (OSError, ValueError, RecursionError) as exc:
            output.warnings.append(f"{path}: cannot load catalog report: {exc}")
            continue
        output.rows.extend(report.rows)
        output.warnings.extend(report.warnings)
    if len(payload["skills"]) > _MAX_FILES:
        output.warnings.append("Dashboard catalog limit reached; some inputs were omitted.")
    if not output.rows:
        output.warnings.extend(data.warnings)
    return output


def load_dashboard_path(path: Path) -> DashboardData:
    """Load one JSON report, a retained run, or a tree of reports and runs.

    Native run subtrees are consumed once and pruned, so their mirrored JSON
    reports and constituent trials cannot become independent dashboard rows.
    Directory symlinks are not traversed.
    """
    path = path.expanduser().resolve()
    output = DashboardData()
    if path.is_file():
        try:
            return _load_report_file(path)
        except (OSError, ValueError, RecursionError) as exc:
            return DashboardData(warnings=[f"{path}: cannot read report: {exc}"])
    if not path.is_dir():
        return DashboardData(warnings=[f"Dashboard input does not exist: {path}"])
    visited = 0
    files = 0
    for directory, subdirectories, filenames in os.walk(path, followlinks=False):
        root = Path(directory)
        subdirectories[:] = sorted(name for name in subdirectories if not (root / name).is_symlink())
        visited += len(subdirectories) + len(filenames)
        if visited > _MAX_PATHS or files >= _MAX_FILES:
            output.warnings.append("Dashboard discovery limit reached; some inputs were omitted.")
            break
        if "result.json" in filenames and "run_config.json" in filenames:
            try:
                data = _load_native(root)
            except (OSError, ValueError, RecursionError) as exc:
                data = DashboardData(warnings=[f"{root}: cannot load retained run: {exc}"])
            subdirectories.clear()
            output.rows.extend(data.rows)
            output.warnings.extend(data.warnings)
            files += 1
            continue
        for filename in sorted(filenames):
            if not filename.endswith(".json"):
                continue
            files += 1
            try:
                data = parse_dashboard_report(_read_json(root / filename), str(root / filename))
            except (OSError, ValueError, RecursionError) as exc:
                output.warnings.append(f"{root / filename}: cannot read report: {exc}")
                continue
            output.rows.extend(data.rows)
            # Avoid noisy warnings for unrelated config/dataset/summary files.
            if data.rows:
                output.warnings.extend(data.warnings)
            if files >= _MAX_FILES:
                output.warnings.append("Dashboard file limit reached; some inputs were omitted.")
                subdirectories.clear()
                break
    if not output.rows:
        output.warnings.append(f"{path}: no supported Tier 3 agent results found.")
    return output
