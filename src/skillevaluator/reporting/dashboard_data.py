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
from itertools import islice
from pathlib import Path
from typing import Any

from packaging.version import Version

from skillevaluator.tier3.harbor.resource_usage import agent_execution_seconds

_USAGE = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "duration_seconds", "cost_usd")
_IDENTITY = ("source", "run_id", "skill", "model", "harness", "features")
_PAIRING_IDENTITY = (*_IDENTITY, "_report_index", "_agent_key", "_feature_key")
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_FILES = 1000
_MAX_PATHS = 20_000
_MAX_COUNT = 1_000_000  # far above report_data's 512-trial bound; larger values are malformed
# Mirror report_data._MAX_TRIALS_PER_CONDITION / _MAX_TRIAL_PATHS_SCANNED so the
# dashboard totals the same bounded trial population as the canonical loader.
_MAX_NATIVE_TRIALS = 512
_MAX_NATIVE_TRIAL_ENTRIES = 4096
_CATALOG_LIMIT_NOTICE = "Dashboard catalog limit reached; some inputs were omitted."
_DISCOVERY_LIMIT_NOTICE = "Dashboard discovery limit reached; some inputs were omitted."
# Loader bounds on these artifacts drop only per-trial telemetry (tokens, steps).
# Canonical trials then omit those counters, so coverage counts keep totals blank.
_TELEMETRY_ARTIFACTS = frozenset({"trajectory"})
# Canonical exports from earlier evaluators wrote missing token counters as zero.
_ZERO_TOKEN_COUNTERS_FIXED_IN = Version("0.5.0")


@dataclass
class DashboardData:
    rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _CatalogBudget:
    seen: set[Path] = field(default_factory=set)
    files_remaining: int = field(default_factory=lambda: _MAX_FILES)
    references_remaining: int = field(default_factory=lambda: _MAX_PATHS)


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
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_COUNT else None


def _feature_config(*owners: dict[str, Any]) -> tuple[str, str]:
    for owner in owners:
        metadata = _dict(owner.get("dashboard_metadata"))
        features = metadata.get("harness_features")
        if isinstance(features, str) and features.strip():
            return features.strip(), json.dumps(["string", features.strip()], ensure_ascii=False)
        if isinstance(features, list) and all(isinstance(item, str) for item in features):
            values = sorted(set(features))
            return ", ".join(values) or "none (recorded)", json.dumps(["list", values], ensure_ascii=False)
        if isinstance(features, dict):
            try:
                value = json.dumps(features, sort_keys=True, ensure_ascii=False, allow_nan=False)
                return value, json.dumps(["dict", features], sort_keys=True, ensure_ascii=False, allow_nan=False)
            except (ValueError, TypeError, RecursionError):
                return "unknown", "unknown"
    return "unknown", "unknown"


def _set_features(row: dict[str, Any], *owners: dict[str, Any]) -> None:
    row["features"], row["_feature_key"] = _feature_config(*owners)


def _is_native_result(value: dict[str, Any]) -> bool:
    return any(
        isinstance(_dict(agent).get("with_skill"), dict) or isinstance(_dict(agent).get("without_skill"), dict)
        for agent in _dict(value.get("agents")).values()
    )


def _trial_coverage_truncated(marker: dict[str, Any]) -> bool:
    if not marker.get("truncated"):
        return False
    loading = marker.get("artifact_loading")
    if loading and (
        not isinstance(loading, list)
        or any(
            not isinstance(reason, dict)
            or not isinstance(reason.get("artifact"), str)
            or reason["artifact"] not in _TELEMETRY_ARTIFACTS
            for reason in loading
        )
    ):
        return True
    omitted = _dict(marker.get("omitted"))
    # These budgets affect diagnostics, not the normalized trial population.
    # Non-best detail pruning is checked per agent against its declared counts.
    diagnostics = {
        "raw_trial_rewards",
        "raw_detail_fields",
        "raw_reward_fields",
        "raw_metric_values",
        "unpaired_case_ids",
        "comparison_payloads",
        "evidence_entries",
        "evaluator_cards",
        "insight_items",
        "custom_metric_discovery_trials",
        "non_best_agent_details",
    }
    if not omitted:
        return not loading  # an unexplained truncation still fails closed
    return any(section not in diagnostics for section in omitted)


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
    seen_trials: dict[str, tuple[str, str | None]] = {}
    for trial in trials:
        entry = trial.get("entry_id")
        if not isinstance(entry, str) or not entry:
            return None
        trial_id = str(trial.get("trial_id") or "")
        attempts = [trial[field] for field in ("attempt", "attempt_index") if trial.get(field) is not None]
        normalized_attempts = []
        for attempt in attempts:
            if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0:
                normalized_attempts.append(str(attempt))
            elif isinstance(attempt, str) and re.fullmatch(r"[0-9]+", attempt):
                normalized_attempts.append(attempt.lstrip("0") or "0")
            else:
                return None
        if len(set(normalized_attempts)) > 1:
            return None
        attempt = normalized_attempts[0] if normalized_attempts else None
        if trial_id and trial_id in seen_trials:
            if seen_trials[trial_id] != (entry, attempt):
                return None
            # Fallback multi-step rewards share one consistent attempt identity.
            continue
        if trial_id:
            seen_trials[trial_id] = (entry, attempt)
        counts[entry] += 1
        if attempt is None:
            implicit.add(entry)
        keys.append((entry, str(attempt) if attempt is not None else f"ordinal:{counts[entry]}"))
    # Harbor names repeated attempts <task>__<random> without an ordinal, so
    # unnumbered attempts pair by case count, as in the HTML report, only when
    # each one has its own trial ID.
    implicit_ids: dict[str, set[str]] = {}
    for trial_id, (entry, attempt) in seen_trials.items():
        if attempt is None:
            implicit_ids.setdefault(entry, set()).add(trial_id)
    if any(counts[entry] > 1 and len(implicit_ids.get(entry, ())) != counts[entry] for entry in implicit):
        return None
    return tuple(sorted(keys))


def _zero_encodes_missing_tokens(report: dict[str, Any]) -> bool:
    # The version names the run's evaluator, so re-exports of older runs also fail closed.
    version = report.get("evaluator_version") or _dict(report.get("summary")).get("evaluator_version")
    if not isinstance(version, str):
        return False
    try:
        return Version(version) < _ZERO_TOKEN_COUNTERS_FIXED_IN
    except ValueError:  # InvalidVersion, or more digits than int() converts
        return False


def _has_zero_token_counter(trials: list[dict[str, Any]]) -> bool:
    return any(
        _measurement(_dict(trial.get("tokens")).get(key)) == 0
        for trial in trials
        for key in ("prompt", "completion", "cached")
    )


def _usage(trial: dict[str, Any], *, zero_tokens_missing: bool = False) -> dict[str, float | None]:
    tokens = _dict(trial.get("tokens"))
    prompt, completion, cached = (_measurement(tokens.get(key)) for key in ("prompt", "completion", "cached"))
    if zero_tokens_missing:
        prompt, completion, cached = (value or None for value in (prompt, completion, cached))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if prompt is not None and completion is not None else None,
        "cached_tokens": cached,
        # Canonical Tier 3 JSON records agent_seconds; duration_seconds is only
        # read from hand-built payloads and never replaces an invalid agent_seconds.
        "duration_seconds": _measurement(
            trial["agent_seconds"] if "agent_seconds" in trial else trial.get("duration_seconds")
        ),
        "cost_usd": _measurement(trial.get("cost_usd")),
    }


def _set_usage(row: dict[str, Any], usages: list[dict[str, Any]], *, expected: int) -> None:
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
    # A total exists only when every expected trial is measured, so this is an exact mean.
    for metric in ("total_tokens", "duration_seconds", "cost_usd"):
        total = row[metric]
        row[f"{metric}_per_trial"] = _number(total / expected) if total is not None and expected else None
    eligibility = "complete" if row["_complete"] else "incomplete or unknown trial coverage"
    row["coverage"] = (
        f"{row['trials']} trial{'' if row['trials'] == 1 else 's'} ({eligibility}); "
        f"tokens {row['_usage_coverage']['total_tokens']}, "
        f"time {row['_usage_coverage']['duration_seconds']}, cost {row['_usage_coverage']['cost_usd']}"
    )


def _ambiguous_resource_notice(source: str) -> str:
    return (
        f"{source}: canonical multi-step resource ownership is ambiguous; "
        "usage totals are unavailable. Load the retained run directory for authoritative step measurements."
    )


def _zero_token_notice(source: str) -> str:
    return (
        f"{source}: exports before v0.5.0 may record missing token counters as zero; "
        "zero token counters are shown as unavailable. Load the retained run directory for recorded counters."
    )


def parse_dashboard_report(payload: Any, source: str) -> DashboardData:
    """Normalize canonical Tier 3, validate-report, and embedded catalog JSON."""
    output = DashboardData()
    for report_index, report in enumerate(_payloads(payload)):
        if _is_native_result(report):
            output.warnings.append(
                f"{source}: native engine JSON needs its retained run directory for trial data; "
                "load the retained directory or a canonical Tier 3 report."
            )
            continue
        summary = _dict(report.get("summary"))
        provenance = _dict(report.get("provenance"))
        run_id = str(report.get("run_id") or summary.get("run_id") or "")
        if not run_id and provenance.get("run_dir"):
            run_id = Path(str(provenance["run_dir"])).name
        if not run_id:
            run_id = str(report.get("evaluated_at") or summary.get("evaluated_at") or "unknown")
        truncation = _dict(report.get("report_truncation")) or _dict(report.get("truncation"))
        coverage_truncated = _trial_coverage_truncated(truncation)
        zero_tokens_missing = _zero_encodes_missing_tokens(report)
        non_best_pruned = bool(truncation.get("truncated")) and "non_best_agent_details" in _dict(
            truncation.get("omitted")
        )
        if truncation.get("truncated"):
            message = (
                "trial data may be truncated; comparisons are unavailable."
                if coverage_truncated
                else "diagnostic details are truncated; complete trial coverage is checked separately."
            )
            output.warnings.append(f"{source}: {message}")
        for name, value in _dict(report.get("agents")).items():
            agent = _dict(value)
            if not agent:
                continue
            conditions = _dict(agent.get("conditions"))
            # The embedded-size budget empties a non-best agent's trial details
            # but keeps its declared counts. Older reports also dropped its
            # conditions; current ones keep a compacted copy.
            details_pruned = (
                non_best_pruned
                and str(name) != str(report.get("best_agent") or "")
                and not agent.get("trials")
                and not agent.get("trials_baseline")
            )
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
                invalid_counts = any(
                    owner.get(key) is not None and _count(owner[key]) is None
                    for owner, key in (
                        (agent, count_field),
                        (details, "expected_attempts"),
                        (details, "scored_attempts"),
                    )
                )
                if invalid_counts:
                    output.warnings.append(
                        f"{source}: {name}/{condition}: invalid trial coverage counts; "
                        "measurements and comparisons are unavailable."
                    )
                declared = _count(agent.get(count_field))
                if declared is None:
                    declared = len(trials)
                pruned = details_pruned and not trials
                expected = _count(details.get("expected_attempts"))
                scored = _count(details.get("scored_attempts"))
                keys = _trial_keys(trials)
                logical_count = len(keys) if keys is not None else declared
                physical_counts = Counter(str(trial["trial_id"]) for trial in trials if trial.get("trial_id"))
                ambiguous_ids = {trial_id for trial_id, count in physical_counts.items() if count > 1}
                if ambiguous_ids:
                    notice = _ambiguous_resource_notice(source)
                    if notice not in output.warnings:
                        output.warnings.append(notice)
                if zero_tokens_missing and _has_zero_token_counter(trials):
                    notice = _zero_token_notice(source)
                    if notice not in output.warnings:
                        output.warnings.append(notice)
                complete = (
                    status == "succeeded"
                    and not invalid_counts
                    and declared > 0
                    and len(trials) == declared
                    and keys is not None
                    and len(set(keys)) == len(keys)
                    and not coverage_truncated
                    and (scored is None or scored == logical_count)
                    and (expected is None or scored == expected)
                )
                row = {
                    "source": source,
                    "run_id": run_id,
                    "skill": str(report.get("skill_name") or summary.get("skill_name") or "unknown"),
                    "model": str(agent.get("model") or "unknown"),
                    "harness": str(agent.get("harness") or agent.get("name") or name),
                    "condition": condition,
                    "status": status,
                    "trials": declared if pruned else logical_count,
                    "score": _number(agent.get(score_field)) if complete else None,
                    "pass_rate": _number(_dict(_dict(agent.get("pass_at_k")).get(condition)).get("rate"))
                    if complete
                    else None,
                    "_complete": complete,
                    "_invalid_counts": invalid_counts,
                    "_coverage_truncated": coverage_truncated,
                    "_details_pruned": pruned,
                    "_report_index": report_index,
                    "_agent_key": str(name),
                    "_trial_keys": keys,
                    "_skill_activation_status": _dict(agent.get("skill_activation")).get("status", "unknown"),
                    "_dataset_digest": report.get("dataset_digest") or summary.get("dataset_digest"),
                    "_attempt_policy": report.get("attempt_policy"),
                    "_expected_usage_trials": max(
                        logical_count, expected or 0, scored or 0, declared if len(trials) != declared else 0
                    ),
                    "_ambiguous_trial_ids": tuple(sorted(ambiguous_ids)),
                }
                _set_features(row, details, agent, report)
                _set_usage(
                    row,
                    []
                    if invalid_counts
                    else [
                        {}
                        if str(trial.get("trial_id") or "") in ambiguous_ids
                        else _usage(trial, zero_tokens_missing=zero_tokens_missing)
                        for trial in trials
                    ],
                    # Trials missing from the declared count are already part of it.
                    expected=declared + max(0, row["_expected_usage_trials"] - max(logical_count, declared)),
                )
                output.rows.append(row)
    if not output.rows and not output.warnings:
        output.warnings.append(f"{source}: no supported Tier 3 agent results found.")
    return output


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop rows reloaded from the same resolved source; keys are unique within one parse."""
    seen: set[tuple[Any, ...]] = set()
    kept = []
    for row in rows:
        key = (row.get("source"), row.get("_report_index"), row.get("_agent_key"), row.get("condition"))
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept, len(rows) - len(kept)


def _ineligible_reason(with_row: dict[str, Any], without_row: dict[str, Any]) -> str:
    """Name the first check that keeps a with/without pair from being comparable."""
    arms = (("with-skill", with_row), ("without-skill", without_row))
    for label, row in arms:
        if row.get("status") != "succeeded":
            return f"{label} arm status {row.get('status') or 'unknown'}"
    if with_row.get("_invalid_counts") or without_row.get("_invalid_counts"):
        return "invalid trial counts"
    if with_row.get("_coverage_truncated") or without_row.get("_coverage_truncated"):
        return "trial data truncated"
    if with_row.get("_details_pruned") or without_row.get("_details_pruned"):
        return "trial details pruned from embedded report; load the retained run directory"
    for label, row in arms:
        if not row.get("_complete"):
            return f"{label} trial coverage incomplete"
    if with_row.get("_trial_keys") != without_row.get("_trial_keys"):
        return "different cases or attempts"
    if with_row.get("_dataset_digest") != without_row.get("_dataset_digest"):
        return "different dataset"
    if with_row.get("_attempt_policy") != without_row.get("_attempt_policy"):
        return "different attempt policy"
    return ""


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute with-minus-without differences only within an identical run."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("condition") in {"with_skill", "without_skill"}:
            groups.setdefault(tuple(row.get(key) for key in _PAIRING_IDENTITY), []).append(row)
    comparisons = []
    for identity, arms in groups.items():
        with_rows = [row for row in arms if row["condition"] == "with_skill"]
        without_rows = [row for row in arms if row["condition"] == "without_skill"]
        result = dict(zip(_PAIRING_IDENTITY, identity, strict=True))
        status = "missing condition"
        reason = f"no {'without' if with_rows else 'with'}-skill arm for this run and configuration"
        paired = False
        if len(with_rows) > 1 or len(without_rows) > 1:
            status = "ambiguous duplicate conditions"
            reason = "same arm loaded more than once"
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
            reason = "" if paired else _ineligible_reason(with_row, without_row)
            if paired and with_row.get("_skill_activation_status") == "never_activated":
                # Absolute scores and usage remain diagnostic, but a complete
                # run that never exercised its skill cannot establish lift.
                paired = False
                status = "inconclusive"
                reason = "Inconclusive: skill never activated in with-skill trials"
        result["comparison_status"] = status
        result["comparison_reason"] = reason
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
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_FILE_BYTES:
        raise ValueError("JSON must be a regular file no larger than 16 MiB")
    with path.open("rb") as stream:
        data = stream.read(_MAX_FILE_BYTES + 1)
    if len(data) > _MAX_FILE_BYTES:
        raise ValueError("JSON exceeds 16 MiB")
    return json.loads(data)


def _safe_native_path(path: Path, run_root: Path) -> bool:
    try:
        path.resolve().relative_to(run_root)
        relative = path.relative_to(run_root)
    except (ValueError, OSError, RuntimeError):
        return False
    current = run_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return False
    return True


def _read_native_json(path: Path, run_root: Path) -> Any:
    if not _safe_native_path(path, run_root):
        raise ValueError(f"symlink or external retained artifact skipped: {path}")
    return _read_json(path)


def _native_usage(
    trial_dir: Path, run_root: Path, warnings: list[str], *, require_steps: bool = False
) -> dict[str, Any]:
    trajectory: dict[str, Any] = {}
    if (trial_dir / "trajectory.json").is_file():
        try:
            trajectory = _dict(_read_native_json(trial_dir / "trajectory.json", run_root))
        except (OSError, ValueError, RecursionError) as exc:
            # As in canonical JSON, this loses only the trajectory's token totals;
            # result.json still records time, cost, and agent_result counters.
            warnings.append(f"{trial_dir}: cannot read trajectory: {exc}")
    final = _dict(trajectory.get("final_metrics"))
    result = (
        _dict(_read_native_json(trial_dir / "result.json", run_root)) if (trial_dir / "result.json").is_file() else {}
    )
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
    if require_steps and not (isinstance(steps, list) and steps):
        return {"_resource_scope_valid": False}
    constituents = steps if isinstance(steps, list) and steps else [result]
    if isinstance(steps, list) and steps:

        def step_total(key: str) -> float | None:
            values = [_measurement(_dict(_dict(step).get("agent_result")).get(key)) for step in steps]
            return _measurement(sum(values)) if all(value is not None for value in values) else None

        prompt = step_total("n_input_tokens")
        completion = step_total("n_output_tokens")
        cached = step_total("n_cache_tokens")
    durations = [agent_execution_seconds(_dict(step)) for step in constituents]
    costs = [_measurement(_dict(_dict(step).get("agent_result")).get("cost_usd")) for step in constituents]
    return {
        "_resource_scope_valid": True,
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

    engine_result = _dict(_read_native_json(path / "result.json", path))
    skill = str(engine_result.get("skill_name") or "unknown")
    # The payload is parsed in memory, never embedded, so keep every agent's trials.
    result = agent_eval_result_from_directory(
        Path(skill), path, engine_result=engine_result, use_llm_judge=False, enforce_embedded_size=False
    )
    if result is None:
        return DashboardData(warnings=[f"{path}: no supported retained agent summaries found."])
    canonical = result.metadata["agent_eval"]
    canonical["run_id"] = engine_result.get("run_id") or path.name
    # The runner writes the same config into result.json, but run_config.json is
    # the copy users edit to add dashboard_metadata, so it wins when readable.
    run_config = _dict(engine_result.get("run_config"))
    config_warning = None
    if (path / "run_config.json").is_file():
        try:
            run_config = _dict(_read_native_json(path / "run_config.json", path)) or run_config
        except (OSError, ValueError, RecursionError) as exc:
            config_warning = f"{path / 'run_config.json'}: cannot read run config: {exc}"
    canonical["dashboard_metadata"] = run_config.get("dashboard_metadata")
    output = parse_dashboard_report(canonical, str(path))
    # Retained usage is read from disk below, so the export notice does not apply.
    output.warnings = [warning for warning in output.warnings if warning != _zero_token_notice(str(path))]
    if config_warning:
        output.warnings.append(config_warning)
    unresolved_scope = False
    for row in output.rows:
        # The report's agent keys identify repeated harness occurrences on disk.
        name = row["_agent_key"]
        agent_config = _dict(_dict(run_config.get("agents")).get(name))
        _set_features(row, agent_config, run_config)
        row["harness"] = str(agent_config.get("agent") or row["harness"])
        variant = {"with_skill": "with-skill", "without_skill": "without-skill", "adverse": "adverse"}[row["condition"]]
        summary_path = path / name / variant / "summary.json"
        if summary_path.is_file():
            condition_summary = _dict(_read_native_json(summary_path, path))
            row["model"] = str(condition_summary.get("model") or row["model"])
            _set_features(row, condition_summary, agent_config, run_config)
            row["_invalid_counts"] |= any(
                condition_summary.get(key) is not None and _count(condition_summary[key]) is None
                for key in ("num_trials", "expected_attempts", "scored_attempts")
            )
        if row["_invalid_counts"]:
            row["_complete"] = False
            row["score"] = row["pass_rate"] = None
            _set_usage(row, [], expected=row["_expected_usage_trials"])
            output.warnings.append(f"{summary_path}: invalid trial coverage counts; measurements are unavailable.")
            continue
        trials_dir = path / name / variant / "trials"
        usages_by_trial: dict[str, dict[str, Any]] = {}
        if not _safe_native_path(trials_dir, path):
            output.warnings.append(f"{trials_dir}: symlink or external retained directory skipped.")
            _set_usage(row, [], expected=row["_expected_usage_trials"])
            continue
        if trials_dir.is_dir():
            try:
                trial_paths, truncated = _directory_entries(trials_dir, _MAX_NATIVE_TRIAL_ENTRIES)
                trial_dirs = [
                    candidate for candidate in trial_paths if candidate.is_dir() and not candidate.is_symlink()
                ]
            except (OSError, ValueError) as exc:
                output.warnings.append(f"{trials_dir}: cannot read retained trials: {exc}")
                _set_usage(row, [], expected=row["_expected_usage_trials"])
                continue
            truncated |= len(trial_dirs) > _MAX_NATIVE_TRIALS
            loaded: list[tuple[Path, dict[str, Any] | None]] = []
            for trial_dir in trial_dirs[:_MAX_NATIVE_TRIALS]:
                try:
                    loaded.append((trial_dir, _dict(_read_native_json(trial_dir / "reward.json", path))))
                except (OSError, ValueError, RecursionError) as exc:
                    output.warnings.append(f"{trial_dir}: cannot read usage: {exc}")
                    loaded.append((trial_dir, None))
            # Canonical reports collapse reward rows that share one physical
            # trial, so count shared identities in the retained rows themselves.
            physical_counts = Counter(
                str(reward.get("trial_id") or trial_dir.name) for trial_dir, reward in loaded if reward is not None
            )
            for trial_dir, reward in loaded:
                if reward is None:
                    usages_by_trial[trial_dir.name] = {}
                    continue
                physical_id = str(reward.get("trial_id") or trial_dir.name)
                if physical_id in usages_by_trial:
                    continue
                try:
                    usages_by_trial[physical_id] = _native_usage(
                        trial_dir,
                        path,
                        output.warnings,
                        require_steps=physical_counts[physical_id] > 1 or physical_id in row["_ambiguous_trial_ids"],
                    )
                    unresolved_scope |= usages_by_trial[physical_id].get("_resource_scope_valid") is False
                except (OSError, ValueError, RecursionError) as exc:
                    output.warnings.append(f"{trial_dir}: cannot read usage: {exc}")
                    usages_by_trial[trial_dir.name] = {}
            if truncated:
                output.warnings.append(f"{trials_dir}: trial discovery limit reached; usage totals are unavailable.")
                usages_by_trial["__omitted_trials__"] = {}
                row["_expected_usage_trials"] = max(row["_expected_usage_trials"], len(usages_by_trial))
        _set_usage(row, list(usages_by_trial.values()), expected=row["_expected_usage_trials"])
    notice = _ambiguous_resource_notice(str(path))
    if not unresolved_scope:
        output.warnings = [warning for warning in output.warnings if warning != notice]
    elif notice not in output.warnings:
        output.warnings.append(notice)
    return output


def _load_report_file(
    path: Path, ancestors: frozenset[Path] = frozenset(), budget: _CatalogBudget | None = None
) -> DashboardData:
    budget = budget if budget is not None else _CatalogBudget()
    path = path.resolve()
    if path in ancestors or len(ancestors) >= 16:
        return DashboardData(warnings=[f"{path}: cyclic or excessively nested catalog reference skipped."])
    if path in budget.seen:
        return DashboardData()
    if budget.files_remaining <= 0 or budget.references_remaining <= 0:
        return DashboardData(warnings=[_CATALOG_LIMIT_NOTICE])
    budget.seen.add(path)
    budget.files_remaining -= 1
    if path.name == "result.json" and (path.parent / "run_config.json").is_file():
        return _load_native(path.parent)
    payload = _read_json(path)
    data = parse_dashboard_report(payload, str(path))
    if not isinstance(payload, dict) or not isinstance(payload.get("skills"), list):
        return data
    output = DashboardData(rows=data.rows, warnings=data.warnings if data.rows else [])
    for entry in payload["skills"][:_MAX_FILES]:
        entry = _dict(entry)
        report_dir, report_name = entry.get("report_dir"), entry.get("json_report")
        if not isinstance(report_dir, str) or not isinstance(report_name, str):
            continue
        if budget.files_remaining <= 0 or budget.references_remaining <= 0:
            output.warnings.append(_CATALOG_LIMIT_NOTICE)
            break
        budget.references_remaining -= 1
        try:
            report_path = (path.parent / report_dir / report_name).resolve()
            report_path.relative_to(path.parent.resolve())
            report = _load_report_file(report_path, ancestors | {path}, budget)
        except (OSError, ValueError, RecursionError, RuntimeError, TypeError, AttributeError) as exc:
            output.warnings.append(f"{path}: cannot load catalog report: {exc}")
            continue
        output.rows.extend(report.rows)
        output.warnings.extend(report.warnings)
    if len(payload["skills"]) > _MAX_FILES:
        output.warnings.append(_CATALOG_LIMIT_NOTICE)
    if not output.rows:
        output.warnings.extend(data.warnings)
    return output


def _directory_entries(path: Path, limit: int) -> tuple[list[Path], bool]:
    """Bound directory enumeration before sorting or allocating the full tree."""
    with os.scandir(path) as entries:
        selected = [Path(entry.path) for entry in islice(entries, limit + 1)]
    return sorted(selected[:limit]), len(selected) > limit


def load_dashboard_path(path: Path) -> DashboardData:
    """Load one JSON report, a retained run, or a tree of reports and runs.

    Native run subtrees are consumed once and pruned, so their mirrored JSON
    reports and constituent trials cannot become independent dashboard rows.
    Directory symlinks are not traversed.
    """
    try:
        path = path.expanduser().resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        return DashboardData(warnings=[f"{path}: cannot resolve dashboard input: {exc}"])
    output = DashboardData()
    if path.is_file():
        try:
            return _load_report_file(path)
        except (OSError, ValueError, RecursionError, RuntimeError, TypeError, AttributeError) as exc:
            return DashboardData(warnings=[f"{path}: cannot read report: {exc}"])
    if not path.is_dir():
        return DashboardData(warnings=[f"Dashboard input does not exist: {path}"])
    visited = 0
    files = 0
    pending = [path]
    while pending:
        root = pending.pop()
        if visited >= _MAX_PATHS or files >= _MAX_FILES:
            output.warnings.append(_DISCOVERY_LIMIT_NOTICE)
            break
        try:
            # is_file() re-raises EACCES for a subdirectory without search permission.
            native_run = (root / "result.json").is_file() and (root / "run_config.json").is_file()
        except OSError as exc:
            output.warnings.append(f"{root}: cannot discover reports: {exc}")
            continue
        if native_run:
            try:
                data = _load_native(root)
            except (OSError, ValueError, RecursionError, RuntimeError, TypeError, AttributeError) as exc:
                data = DashboardData(warnings=[f"{root}: cannot load retained run: {exc}"])
            output.rows.extend(data.rows)
            output.warnings.extend(data.warnings)
            files += 1
            continue
        try:
            entries, truncated = _directory_entries(root, _MAX_PATHS - visited)
        except OSError as exc:
            output.warnings.append(f"{root}: cannot discover reports: {exc}")
            continue
        visited += len(entries)
        if truncated:
            output.warnings.append(_DISCOVERY_LIMIT_NOTICE)
        subdirectories = []
        for entry in entries:
            if files >= _MAX_FILES:
                output.warnings.append("Dashboard file limit reached; some inputs were omitted.")
                break
            if entry.is_symlink():
                continue
            if entry.is_dir():
                subdirectories.append(entry)
                continue
            if entry.suffix != ".json":
                continue
            files += 1
            try:
                data = parse_dashboard_report(_read_json(entry), str(entry))
            except (OSError, ValueError, RecursionError) as exc:
                output.warnings.append(f"{entry}: cannot read report: {exc}")
                continue
            output.rows.extend(data.rows)
            # Avoid noisy warnings for unrelated config/dataset/summary files.
            if data.rows:
                output.warnings.extend(data.warnings)
        pending.extend(reversed(subdirectories))
    if not output.rows:
        output.warnings.append(f"{path}: no supported Tier 3 agent results found.")
    return output
