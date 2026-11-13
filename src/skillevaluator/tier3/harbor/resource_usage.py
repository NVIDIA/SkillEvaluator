# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured agent resources, separate from routing/tool-productivity scores."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Any

from skillevaluator.tier3.harbor.trajectory_timing import TIMING_SOURCES

_MAX_SAFE_INTEGER = (1 << 53) - 1
_TIMING_FIELDS = ("llm_seconds", "tool_seconds")


def _is_measurement(value: Any) -> bool:
    """A JSON-safe, finite, non-negative counter; bools, strings and NaN are not measurements."""
    return type(value) in (int, float) and 0 <= value <= _MAX_SAFE_INTEGER and math.isfinite(value)


def normalized_trial_timing(raw: Any) -> dict[str, Any] | None:
    """Validate one trial's ``timing``; invalid or untagged values stay unavailable, not zero."""
    if not isinstance(raw, dict) or raw.get("source") not in TIMING_SOURCES:
        return None
    timing: dict[str, Any] = {}
    for field in _TIMING_FIELDS:
        value = raw.get(field)
        timing[field] = float(value) if _is_measurement(value) else None
    if all(timing[field] is None for field in _TIMING_FIELDS):
        return None
    timing["source"] = raw["source"]
    return timing


def agent_execution_seconds(result: dict[str, Any]) -> float | None:
    """Read Harbor's measured agent phase; never substitute total job/verifier time."""
    timing = result.get("agent_execution")
    if not isinstance(timing, dict):
        return None
    try:
        started = datetime.fromisoformat(timing["started_at"])
        finished = datetime.fromisoformat(timing["finished_at"])
        if started.utcoffset() is None or finished.utcoffset() is None:
            return None
        seconds = (finished - started).total_seconds()
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def resource_comparison(
    with_trials: list[dict[str, Any]],
    baseline_trials: list[dict[str, Any]],
    *,
    with_complete: bool,
    baseline_complete: bool,
) -> dict[str, Any]:
    """Compare complete matching batches; missing counters and baseline remain null."""

    def identity(trials: list[dict[str, Any]]) -> Counter | None:
        ids = [trial.get("entry_id") for trial in trials]
        return Counter(ids) if ids and all(isinstance(value, str) and value for value in ids) else None

    paired = with_complete and baseline_complete and identity(with_trials) is not None
    paired = paired and identity(with_trials) == identity(baseline_trials)

    def trial_value(trial: dict[str, Any], field: str) -> Any:
        if field in _TIMING_FIELDS:
            timing = normalized_trial_timing(trial.get("timing"))
            return timing[field] if timing is not None else None
        value = trial.get("agent_seconds")
        if field == "tokens":
            tokens = trial.get("tokens", {})
            if not isinstance(tokens, dict):
                tokens = {}
            prompt, completion = tokens.get("prompt"), tokens.get("completion")
            value = (
                prompt + completion
                if type(prompt) is int and type(completion) is int and min(prompt, completion) >= 0
                else None
            )
        return value

    def measure(trials: list[dict[str, Any]], field: str, complete: bool) -> dict[str, Any]:
        values = []
        for trial in trials:
            value = trial_value(trial, field)
            if _is_measurement(value):
                values.append(value)
        # Millisecond precision keeps float noise out of the timing totals.
        total = round(math.fsum(values), 3) if field in _TIMING_FIELDS else sum(values)
        available = (
            complete
            and bool(trials)
            and len(values) == len(trials)
            and math.isfinite(total)
            and total <= _MAX_SAFE_INTEGER
        )
        return {
            "status": "complete" if available else "unavailable",
            "measured_trials": len(values),
            "trials": len(trials),
            "total": total if available else None,
        }

    measurements = {}
    for field in ("tokens", "agent_seconds", *_TIMING_FIELDS):
        with_measure = measure(with_trials, field, with_complete)
        baseline_measure = measure(baseline_trials, field, baseline_complete)
        delta = None
        if paired and with_measure["total"] is not None and baseline_measure["total"] is not None:
            delta = with_measure["total"] - baseline_measure["total"]
            if field in _TIMING_FIELDS:
                delta = round(delta, 3)
        measurements[field] = {"with_skill": with_measure, "baseline": baseline_measure, "delta": delta}

    def per_token(trials: list[dict[str, Any]], arm: str) -> dict[str, Any]:
        """LLM milliseconds per prompt + completion token, from the arm's complete totals only."""
        llm_total = measurements["llm_seconds"][arm]["total"]
        token_total = measurements["tokens"][arm]["total"]
        value = round(llm_total * 1000 / token_total, 3) if llm_total is not None and token_total else None
        measured = sum(
            1
            for trial in trials
            if _is_measurement(trial_value(trial, "llm_seconds")) and _is_measurement(trial_value(trial, "tokens"))
        )
        return {
            "status": "complete" if value is not None else "unavailable",
            "measured_trials": measured,
            "trials": len(trials),
            "value": value,
        }

    with_per_token = per_token(with_trials, "with_skill")
    baseline_per_token = per_token(baseline_trials, "baseline")
    per_token_delta = None
    if paired and with_per_token["value"] is not None and baseline_per_token["value"] is not None:
        per_token_delta = round(with_per_token["value"] - baseline_per_token["value"], 3)
    measurements["llm_ms_per_token"] = {
        "with_skill": with_per_token,
        "baseline": baseline_per_token,
        "delta": per_token_delta,
    }

    def sources(trials: list[dict[str, Any]]) -> list[str]:
        found = {timing["source"] for trial in trials if (timing := normalized_trial_timing(trial.get("timing")))}
        return sorted(found)

    timing_sources = {"with_skill": sources(with_trials), "baseline": sources(baseline_trials)}
    mixed = len({*timing_sources["with_skill"], *timing_sources["baseline"]}) > 1
    return {
        "scope": (
            "observed agent-only trial totals, including failed task outcomes; "
            "prompt + completion tokens (cache included), excluding judge/setup time"
        ),
        "interpretation": (
            "Observed consumption only. Lower use alone does not establish improved task quality or Efficiency. "
            "Pairing does not certify matched concurrency or budget, or judge-model agreement. "
            "Reused baseline measurements are historical, not fresh or independent evidence."
        ),
        "pairing": "matching case counts" if paired else "unavailable",
        "delta_definition": "with skill minus baseline; negative means less observed resource use",
        "timing_definition": (
            "llm_seconds and tool_seconds split each trial's ATIF trajectory time; source atif reads native "
            "timing extras, source step_timestamps infers it from step timestamps and the wall time of Codex "
            "shell and apply_patch calls (LLM time then includes harness and queue time); llm_ms_per_token is "
            "1000 * llm_seconds / (prompt + completion tokens)"
        ),
        "timing_sources": timing_sources,
        "caveats": ["mixed_timing_sources"] if mixed else [],
        **measurements,
    }
