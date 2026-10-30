# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Condition-aware chart data derived from canonical Tier 3 trials."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

DEFAULT_AGENT_LABEL = "(default)"


def _step_value(trial: Mapping[str, Any]) -> float | None:
    value = trial.get("steps")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _entry_id(trial: Mapping[str, Any]) -> str:
    return str(trial.get("entry_id") or trial.get("trial_id") or "?")


def _trial_list(raw: object) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [trial for trial in raw if isinstance(trial, Mapping)]


def _finite_mean(values: list[float]) -> float | None:
    """Average non-negative values without overflowing to infinity."""
    if not values:
        return None
    scale = max(values)
    if scale == 0:
        return 0.0
    mean = scale * (math.fsum(value / scale for value in values) / len(values))
    return mean if math.isfinite(mean) else None


def _steps_by_case(trials: list[Mapping[str, Any]]) -> dict[str, float | None]:
    """Mean steps per case; a case with any unmeasured retry stays unavailable."""
    grouped: dict[str, list[float | None]] = {}
    for trial in trials:
        grouped.setdefault(_entry_id(trial), []).append(_step_value(trial))
    by_case: dict[str, float | None] = {}
    for entry_id, values in grouped.items():
        measured = [value for value in values if value is not None]
        by_case[entry_id] = _finite_mean(measured) if len(measured) == len(values) else None
    return by_case


def steps_by_case_chart(
    agents: object,
    trials: object,
    *,
    with_label: str = "With skill",
    without_label: str = "Without skill",
) -> dict[str, Any]:
    """Return step-chart labels and one dataset per agent and condition.

    With-skill rows come from the canonical top-level ``trials`` (annotated with
    ``agent``) and baseline rows from each agent's ``trials_baseline``. A series
    with no complete measurement is omitted. Cases absent from a series, or with
    any retry lacking step telemetry, are ``None`` so Chart.js renders a gap
    rather than a misleading zero or partial average.
    """
    agent_map = agents if isinstance(agents, Mapping) else {}
    with_skill: dict[str, list[Mapping[str, Any]]] = {str(name): [] for name in agent_map}
    for trial in _trial_list(trials):
        with_skill.setdefault(str(trial.get("agent") or DEFAULT_AGENT_LABEL), []).append(trial)

    series: list[tuple[str, list[Mapping[str, Any]], dict[str, float | None]]] = []
    for name, agent_trials in with_skill.items():
        raw_agent = agent_map.get(name)
        baseline_trials = _trial_list(raw_agent.get("trials_baseline")) if isinstance(raw_agent, Mapping) else []
        for condition, condition_trials in ((with_label, agent_trials), (without_label, baseline_trials)):
            values = _steps_by_case(condition_trials)
            if any(value is not None for value in values.values()):
                series.append((f"{name} · {condition}", condition_trials, values))

    labels: list[str] = []
    seen: set[str] = set()
    for _label, condition_trials, _values in series:
        for trial in condition_trials:
            entry_id = _entry_id(trial)
            if entry_id not in seen:
                seen.add(entry_id)
                labels.append(entry_id)

    return {
        "labels": labels,
        "datasets": [
            {"label": label, "data": [values.get(entry_id) for entry_id in labels]}
            for label, _condition_trials, values in series
        ],
    }
