# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured agent resources, separate from routing/tool-productivity scores."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Any


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

    def measure(trials: list[dict[str, Any]], field: str, complete: bool) -> dict[str, Any]:
        values = []
        for trial in trials:
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
            if type(value) in (int, float) and 0 <= value <= (1 << 53) - 1 and math.isfinite(value):
                values.append(value)
        total = sum(values)
        available = (
            complete and bool(trials) and len(values) == len(trials) and math.isfinite(total) and total <= (1 << 53) - 1
        )
        return {
            "status": "complete" if available else "unavailable",
            "measured_trials": len(values),
            "trials": len(trials),
            "total": total if available else None,
        }

    measurements = {}
    for field in ("tokens", "agent_seconds"):
        with_measure = measure(with_trials, field, with_complete)
        baseline_measure = measure(baseline_trials, field, baseline_complete)
        delta = None
        if paired and with_measure["total"] is not None and baseline_measure["total"] is not None:
            delta = with_measure["total"] - baseline_measure["total"]
        measurements[field] = {"with_skill": with_measure, "baseline": baseline_measure, "delta": delta}
    return {
        "scope": "observed agent-only trial totals; prompt + completion tokens (cache included), excluding judge/setup time",
        "pairing": "matching case counts" if paired else "unavailable",
        "delta_definition": "with skill minus baseline; negative means less observed resource use",
        **measurements,
    }
