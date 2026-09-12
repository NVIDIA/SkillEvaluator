# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical metric definitions for Harbor-backed SkillEvaluator evals."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from skillevaluator.constants import DEFAULT_SCORE_POLICY, DIMENSION_MAPPING

DEFAULT_METRIC_SET = "skill-evaluator-default-v2"
LEGACY_METRIC_SET = "skill-evaluator-default-v1"
CUSTOM_ONLY_METRIC_SET = "custom-only"

LEGACY_SCORE_POLICY = "skill-evaluator-metric-mean-v1"
CUSTOM_SCORE_POLICY = "custom-overall-v1"
PARTIAL_SCORE_POLICY = "legacy-partial-dimension-mean-v1"

DEFAULT_METRICS = (
    "security",
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
)

LEGACY_METRICS = (
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
)

METRIC_DISPLAY = {
    "security": "Security",
    "skill_execution": "Skill Execution",
    "skill_efficiency": "Efficiency",
    "accuracy": "Accuracy",
    "goal_accuracy": "Goal Accuracy",
    "behavior_check": "Behavior Check",
}

METRIC_DESCRIPTIONS = {
    "security": "Trace scan for unsafe operations, secret leakage, and unauthorized access",
    "skill_execution": "Activation, script run, workflow order, error recovery",
    "skill_efficiency": "Routing correctness, workspace-aware skill reads, tool call productivity",
    "accuracy": "Factual correctness (5-criterion LLM rubric)",
    "goal_accuracy": "Did the agent achieve the user's goal?",
    "behavior_check": "Adherence to expected workflow steps",
}

METRIC_QUESTIONS = {
    "security": "Is the run safe?",
    "skill_execution": "Was the target skill discovered and executed?",
    "skill_efficiency": "Did it use the skill efficiently?",
    "accuracy": "Was the final answer correct?",
    "goal_accuracy": "Did it satisfy the user's goal?",
    "behavior_check": "Did it follow the expected workflow?",
}

# Collection, deterministic judging, and publication share one canonical
# mapping. Legacy fallbacks remain in ``DIMENSION_MAPPING`` for old artifacts,
# but only the primary evaluators participate in newly collected scores.
DIMENSION_DEFINITIONS = {
    dimension: dict(zip(config["evaluators"], config["weights"], strict=True))
    for dimension, config in DIMENSION_MAPPING.items()
}

DIMENSION_DISPLAY = {
    "security": "Security",
    "correctness": "Correctness",
    "discoverability": "Discoverability",
    "effectiveness": "Effectiveness",
    "efficiency": "Efficiency",
}

DIMENSION_QUESTIONS = {
    "security": "Is the run safe?",
    "correctness": "Is the answer correct?",
    "discoverability": "Was the right skill loaded when needed?",
    "effectiveness": "Did the skill help complete the task?",
    "efficiency": "Did it avoid wasted tool or skill usage?",
}

_RESERVED_METADATA_KEYS = {
    "details",
    "entry_id",
    "error",
    "evaluation_errors",
    "evaluation_status",
    "has_skill",
    "metric_set",
    "metric_set_version",
    "metrics",
    "overall",
    "trajectory_detail",
    "trajectory_source",
}

RESERVED_METRIC_NAMES = frozenset(DEFAULT_METRICS) | _RESERVED_METADATA_KEYS


def _finite_number(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def metric_value(reward: dict[str, Any], metric: str) -> float | None:
    """Return a numeric metric value from a reward payload, if present."""
    val = _finite_number(reward.get(metric))
    if val is not None:
        return val

    metrics = reward.get("metrics")
    if isinstance(metrics, dict):
        raw = metrics.get(metric)
        if isinstance(raw, dict):
            raw = raw.get("score")
        numeric = _finite_number(raw)
        if numeric is not None:
            return numeric

    return None


def metric_set_for_reward(reward: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Return the metric-set label and metric order used by a reward payload."""
    metric_set = str(reward.get("metric_set") or reward.get("metric_set_version") or "")
    if metric_set == DEFAULT_METRIC_SET:
        return DEFAULT_METRIC_SET, DEFAULT_METRICS
    if metric_set == LEGACY_METRIC_SET:
        return LEGACY_METRIC_SET, LEGACY_METRICS

    if metric_value(reward, "security") is not None:
        return DEFAULT_METRIC_SET, DEFAULT_METRICS
    if any(metric_value(reward, m) is not None for m in LEGACY_METRICS):
        return LEGACY_METRIC_SET, LEGACY_METRICS
    if _finite_number(reward.get("overall")) is not None:
        return CUSTOM_ONLY_METRIC_SET, ()
    return DEFAULT_METRIC_SET, DEFAULT_METRICS


def metric_set_for_rewards(rewards: list[dict[str, Any]]) -> tuple[str, tuple[str, ...]]:
    """Return the metric set for a collection, preferring the new SkillEvaluator set."""
    declared = {str(reward.get("metric_set") or reward.get("metric_set_version") or "") for reward in rewards}
    if DEFAULT_METRIC_SET in declared:
        return DEFAULT_METRIC_SET, DEFAULT_METRICS
    if LEGACY_METRIC_SET in declared:
        return LEGACY_METRIC_SET, LEGACY_METRICS
    if any(metric_value(reward, "security") is not None for reward in rewards):
        return DEFAULT_METRIC_SET, DEFAULT_METRICS
    if any(any(metric_value(reward, m) is not None for m in LEGACY_METRICS) for reward in rewards):
        return LEGACY_METRIC_SET, LEGACY_METRICS
    if any(_finite_number(reward.get("overall")) is not None for reward in rewards):
        return CUSTOM_ONLY_METRIC_SET, ()
    return DEFAULT_METRIC_SET, DEFAULT_METRICS


def average_metrics(rewards: list[dict[str, Any]]) -> tuple[dict[str, float], str, tuple[str, ...]]:
    """Average SkillEvaluator metrics across rewards, with legacy artifact compatibility."""
    metric_set, metrics = metric_set_for_rewards(rewards)
    if not rewards:
        return {}, metric_set, metrics
    metric_sums: dict[str, float] = dict.fromkeys(metrics, 0.0)
    metric_counts: dict[str, int] = dict.fromkeys(metrics, 0)

    for reward in rewards:
        for metric in metrics:
            val = metric_value(reward, metric)
            if val is not None:
                metric_sums[metric] += val
                metric_counts[metric] += 1

    averages: dict[str, float] = {}
    for metric in metrics:
        count = metric_counts[metric]
        if count > 0:
            averages[metric] = round(metric_sums[metric] / count, 4)

    return averages, metric_set, metrics


def score_policy_for_metrics(metrics: tuple[str, ...]) -> str:
    """Return the versioned overall-score policy for one metric set."""
    if metrics == DEFAULT_METRICS:
        return DEFAULT_SCORE_POLICY
    if metrics == LEGACY_METRICS:
        return LEGACY_SCORE_POLICY
    if not metrics:
        return CUSTOM_SCORE_POLICY
    return PARTIAL_SCORE_POLICY


def canonical_dimension_mean(values: Iterable[object]) -> float | None:
    """Average one complete canonical dimension set, failing closed on gaps."""
    raw_values = tuple(values)
    if len(raw_values) != len(DIMENSION_DEFINITIONS):
        return None
    numeric = tuple(_finite_number(value) for value in raw_values)
    if any(value is None for value in numeric):
        return None
    return sum(value for value in numeric if value is not None) / len(numeric)


def overall_score_from_metrics(
    scores: Mapping[str, object],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> float | None:
    """Aggregate one metric set using its versioned score policy."""
    values = tuple(_finite_number(scores.get(metric)) for metric in metrics)
    if not values or any(value is None for value in values):
        return None

    if metrics != LEGACY_METRICS and all(metric in DEFAULT_METRICS for metric in metrics):
        dimensions = dimension_scores(dict(zip(metrics, values, strict=True)))
        dimension_values = tuple(
            dimensions[dimension]["score"] for dimension in DIMENSION_DEFINITIONS if dimension in dimensions
        )
        if metrics == DEFAULT_METRICS:
            return canonical_dimension_mean(dimension_values)
        return sum(dimension_values) / len(dimension_values) if dimension_values else None

    return sum(value for value in values if value is not None) / len(values)


def overall_score(reward: dict[str, Any]) -> float | None:
    """Compute pass@k/lift overall score for a reward payload.

    Default rewards use the equal-weight mean of the five canonical dimensions.
    Legacy metric sets retain their historical metric mean. Custom rewards without
    SkillEvaluator metrics can still pass through by emitting numeric ``overall``.
    """
    _, metrics = metric_set_for_reward(reward)
    if metrics:
        return overall_score_from_metrics({metric: metric_value(reward, metric) for metric in metrics}, metrics)

    return _finite_number(reward.get("overall"))


def score_definition(metrics: tuple[str, ...] = DEFAULT_METRICS) -> str:
    """Human-readable definition for the SkillEvaluator overall score."""
    if not metrics:
        return f"overall = user-provided reward overall [{CUSTOM_SCORE_POLICY}]"
    if metrics == DEFAULT_METRICS:
        return (
            "overall = mean(Security, Correctness, Discoverability, Effectiveness, Efficiency) "
            f"[{DEFAULT_SCORE_POLICY}]"
        )
    if metrics == LEGACY_METRICS:
        return "overall = mean(" + ", ".join(metrics) + f") [{LEGACY_SCORE_POLICY}]"
    return f"overall = mean(available canonical dimensions) [{PARTIAL_SCORE_POLICY}]"


def dimension_scores(scores: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Compute report-only SkillEvaluator dimension scores from default metric scores."""
    out: dict[str, dict[str, Any]] = {}
    for dimension, sources in DIMENSION_DEFINITIONS.items():
        if not all(_finite_number(scores.get(metric)) is not None for metric in sources):
            continue
        total_weight = sum(sources.values())
        if total_weight <= 0:
            continue
        score = sum(float(scores[metric]) * weight for metric, weight in sources.items()) / total_weight
        out[dimension] = {
            "score": round(score, 4),
            "sources": sources,
        }
    return out


def extract_custom_metrics(reward: dict[str, Any]) -> dict[str, float]:
    """Return user/custom numeric metrics that are separate from SkillEvaluator metrics."""
    custom: dict[str, float] = {}

    explicit = reward.get("custom_metrics")
    if isinstance(explicit, dict):
        for name, value in explicit.items():
            if name in RESERVED_METRIC_NAMES:
                continue
            if isinstance(value, dict):
                value = value.get("score")
            numeric = _finite_number(value)
            if numeric is not None:
                custom[str(name)] = numeric

    metrics = reward.get("metrics")
    if isinstance(metrics, dict):
        for name, value in metrics.items():
            if name in RESERVED_METRIC_NAMES:
                continue
            if isinstance(value, dict):
                value = value.get("score")
            numeric = _finite_number(value)
            if numeric is not None:
                custom[str(name)] = numeric

    for name, value in reward.items():
        if name in RESERVED_METRIC_NAMES or name.startswith("_"):
            continue
        numeric = _finite_number(value)
        if numeric is not None:
            custom[str(name)] = numeric

    return custom


def average_custom_metrics(rewards: list[dict[str, Any]]) -> dict[str, float]:
    """Average custom metrics across rewards."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for reward in rewards:
        for name, value in extract_custom_metrics(reward).items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    return {name: round(sums[name] / counts[name], 4) for name in sorted(sums)}
