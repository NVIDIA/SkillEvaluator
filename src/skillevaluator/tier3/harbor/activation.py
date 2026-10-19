# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize observed skill activation independently of execution scores."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    LEGACY_METRIC_SET,
    metric_set_for_reward,
    metric_value,
)


def _activation_state(reward: Mapping[str, Any]) -> str:
    details = reward.get("details")
    execution = details.get("skill_execution") if isinstance(details, Mapping) else None
    if not isinstance(execution, Mapping):
        return "unknown"
    message = execution.get("message")
    if execution.get("should_trigger") is False or (
        isinstance(message, str)
        and message
        in {
            "No expected_skill -- skipped",
            "Negative case, no skill identified",
        }
    ):
        return "not_applicable"
    standard_sets = {DEFAULT_METRIC_SET, LEGACY_METRIC_SET}
    declared = reward.get("metric_set") or reward.get("metric_set_version")
    metric_set, metrics = metric_set_for_reward(reward)
    if (
        (declared and str(declared) not in standard_sets)
        or metric_set not in standard_sets
        or "skill_execution" not in metrics
        or metric_value(reward, "skill_execution") is None
    ):
        return "unknown"
    activation = execution.get("activation")
    if not isinstance(activation, Mapping) or activation.get("supported") is False:
        return "unknown"
    score = activation.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float) or not 0 <= score <= 1 or not math.isfinite(score):
        return "unknown"
    if activation.get("passed") is True and score > 0:
        return "activated"
    if activation.get("passed") is False and score == 0:
        return "never_activated"
    return "unknown"


def summarize_skill_activation(rewards: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count logical trials, preserving uncertainty and excluding negative controls.

    A multi-step trial activated the skill if any step did. Missing or unsupported
    evidence prevents a claim that the skill was *never* activated. Historical
    and custom-only artifacts without activation evidence remain unknown.
    """
    trials: dict[str, set[str]] = {}
    for index, reward in enumerate(rewards):
        trial = str(reward.get("_trial_root_name") or reward.get("_trial_name") or f"__row_{index}")
        trials.setdefault(trial, set()).add(_activation_state(reward))

    positive = activated = unknown = 0
    for states in trials.values():
        if states & {"activated", "never_activated"}:
            positive += 1
        if "activated" in states:
            activated += 1
        elif "unknown" in states:
            unknown += 1

    if activated:
        status, reason = "activated", "Skill activation observed in with-skill trials"
    elif unknown or not trials:
        status, reason = "unknown", "Skill activation evidence is unavailable or incomplete"
    elif positive:
        status, reason = "never_activated", "Inconclusive: skill never activated in with-skill trials"
    else:
        status, reason = "not_applicable", "No positive skill-activation cases were evaluated"
    return {
        "status": status,
        "positive_trials": positive,
        "activated_trials": activated,
        "unknown_trials": unknown,
        "reason": reason,
    }
