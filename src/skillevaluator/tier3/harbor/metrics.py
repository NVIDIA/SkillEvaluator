# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical metric definitions for Harbor-backed SkillEvaluator evals."""

from __future__ import annotations

import math
import re
from typing import Any

from skillevaluator.constants import DIMENSION_MAPPING
from skillevaluator.utils.redaction import contains_credential_value, is_sensitive_key

DEFAULT_METRIC_SET = "skill-evaluator-default-v2"
LEGACY_METRIC_SET = "skill-evaluator-default-v1"
CUSTOM_ONLY_METRIC_SET = "custom-only"

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
    "custom_details",
    "custom_metrics",
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

# Keep collection, aggregation, generated JSON, and the canonical report on one
# explicit custom-metric envelope. The report visits at most 128 names per
# reward; enforcing the same limit before aggregation prevents summaries and
# paired lift artifacts from expanding past their browser-safe bounds.
MAX_CUSTOM_METRICS = 128
MAX_CUSTOM_METRIC_NAME_BYTES = 256

_SAFE_SENSITIVE_METRIC_PREFIXES = {"auth", "secret", "token"}
_SAFE_SENSITIVE_METRIC_SUFFIXES = {
    "accuracy",
    "compliance",
    "count",
    "coverage",
    "efficiency",
    "handling",
    "leakage",
    "precision",
    "quality",
    "rate",
    "ratio",
    "recall",
    "safety",
    "score",
    "usage",
}


class CustomMetricContractError(ValueError):
    """Raised when custom metrics exceed the bounded publication contract."""


def _custom_metric_name_shape_is_valid(name: str) -> bool:
    try:
        encoded = name.encode("utf-8")
    except UnicodeError:
        return False
    return bool(name) and name == name.strip() and name.isprintable() and len(encoded) <= MAX_CUSTOM_METRIC_NAME_BYTES


def _is_explicit_safe_sensitive_metric_name(name: str) -> bool:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", camel_split).strip("_").lower()
    parts = tuple(part for part in normalized.split("_") if part)
    return (
        len(parts) == 2 and parts[0] in _SAFE_SENSITIVE_METRIC_PREFIXES and parts[1] in _SAFE_SENSITIVE_METRIC_SUFFIXES
    )


def _custom_metric_name_contains_sensitive_data(name: str) -> bool:
    return bool(
        contains_credential_value(name)
        or (is_sensitive_key(name) and not _is_explicit_safe_sensitive_metric_name(name))
    )


def custom_metric_name_is_publishable(name: object) -> bool:
    """Return whether a custom metric name is bounded and safe to publish."""
    text = str(name)
    if not _custom_metric_name_shape_is_valid(text):
        return False
    # Credential values can themselves appear as keys. Never turn those into a
    # visible ``<redacted>`` alias because aliases can collide. Narrow,
    # explicitly metric-shaped names such as ``secret_handling`` remain valid.
    return not _custom_metric_name_contains_sensitive_data(text)


def _iter_custom_metric_candidates(reward: dict[str, Any]):
    explicit = reward.get("custom_metrics")
    if isinstance(explicit, dict):
        yield from explicit.items()

    metrics = reward.get("metrics")
    if isinstance(metrics, dict):
        yield from metrics.items()

    for name, value in reward.items():
        if str(name) in RESERVED_METRIC_NAMES or str(name).startswith("_"):
            continue
        yield name, value


def custom_metric_contract_error(reward: dict[str, Any]) -> str | None:
    """Return a fixed diagnostic when one reward exceeds the custom-metric contract."""
    explicit = reward.get("custom_metrics")
    if "custom_metrics" in reward and not isinstance(explicit, dict):
        return "Custom metrics container must be a JSON object"
    if isinstance(explicit, dict) and any(str(name) in RESERVED_METRIC_NAMES for name in explicit):
        return "Custom metric collides with reserved SkillEvaluator metric names"

    names: set[str] = set()
    for raw_name, raw_value in _iter_custom_metric_candidates(reward):
        name = str(raw_name)
        if name in RESERVED_METRIC_NAMES:
            continue
        value = raw_value.get("score") if isinstance(raw_value, dict) else raw_value
        if score_value(value) is None:
            continue
        # Secret-shaped names are deliberately omitted rather than reported or
        # aliased. Their content must not reach diagnostics either.
        if _custom_metric_name_contains_sensitive_data(name):
            continue
        if not _custom_metric_name_shape_is_valid(name):
            return "Custom metric name exceeds the bounded publication contract"
        names.add(name)
        if len(names) > MAX_CUSTOM_METRICS:
            return "Custom metric count exceeds the per reward publication limit"
    return None


def _finite_number(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def score_value(value: object) -> float | None:
    """Return a finite score inside SkillEvaluator's documented 0..1 range."""
    numeric = _finite_number(value)
    return numeric if numeric is not None and 0.0 <= numeric <= 1.0 else None


def _has_metric_field(reward: dict[str, Any], metric: str) -> bool:
    """Return whether a reward claims a metric, independently of score validity."""
    if metric in reward:
        return True
    metrics = reward.get("metrics")
    return isinstance(metrics, dict) and metric in metrics


def metric_value(reward: dict[str, Any], metric: str) -> float | None:
    """Return a numeric metric value from a reward payload, if present."""
    val = score_value(reward.get(metric))
    if val is not None:
        return val

    metrics = reward.get("metrics")
    if isinstance(metrics, dict):
        raw = metrics.get(metric)
        if isinstance(raw, dict):
            raw = raw.get("score")
        numeric = score_value(raw)
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
    if metric_set:
        # Explicit custom metric sets cannot claim SkillEvaluator's reserved
        # canonical names. Their only standard score surface is ``overall``.
        return CUSTOM_ONLY_METRIC_SET, ()

    if _has_metric_field(reward, "security"):
        return DEFAULT_METRIC_SET, DEFAULT_METRICS
    if any(_has_metric_field(reward, metric) for metric in LEGACY_METRICS):
        return LEGACY_METRIC_SET, LEGACY_METRICS
    if score_value(reward.get("overall")) is not None:
        return CUSTOM_ONLY_METRIC_SET, ()
    return DEFAULT_METRIC_SET, DEFAULT_METRICS


def metric_set_for_rewards(rewards: list[dict[str, Any]]) -> tuple[str, tuple[str, ...]]:
    """Return the metric set for a collection, preferring the new SkillEvaluator set."""
    declared = {str(reward.get("metric_set") or reward.get("metric_set_version") or "") for reward in rewards}
    if DEFAULT_METRIC_SET in declared:
        return DEFAULT_METRIC_SET, DEFAULT_METRICS
    if LEGACY_METRIC_SET in declared:
        return LEGACY_METRIC_SET, LEGACY_METRICS
    undeclared = [reward for reward in rewards if not (reward.get("metric_set") or reward.get("metric_set_version"))]
    if any(_has_metric_field(reward, "security") for reward in undeclared):
        return DEFAULT_METRIC_SET, DEFAULT_METRICS
    if any(any(_has_metric_field(reward, metric) for metric in LEGACY_METRICS) for reward in undeclared):
        return LEGACY_METRIC_SET, LEGACY_METRICS
    if any(score_value(reward.get("overall")) is not None for reward in rewards):
        return CUSTOM_ONLY_METRIC_SET, ()
    if any(value for value in declared):
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
        _, reward_metrics = metric_set_for_reward(reward)
        for metric in metrics:
            if metric not in reward_metrics:
                continue
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


def overall_score(reward: dict[str, Any]) -> float | None:
    """Compute pass@k/lift overall score for a reward payload.

    SkillEvaluator default rewards use the mean of their active SkillEvaluator metric set.  Custom
    rewards without SkillEvaluator metrics can still pass through by emitting numeric
    ``overall``.
    """
    _, metrics = metric_set_for_reward(reward)
    values = [metric_value(reward, m) for m in metrics]
    if metrics:
        if not values or any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None) / len(values)

    return score_value(reward.get("overall"))


def score_definition(metrics: tuple[str, ...] = DEFAULT_METRICS) -> str:
    """Human-readable definition for the SkillEvaluator overall score."""
    if not metrics:
        return "overall = user-provided reward overall"
    return "overall = mean(" + ", ".join(metrics) + ")"


def dimension_scores(scores: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Compute report-only SkillEvaluator dimension scores from default metric scores."""
    out: dict[str, dict[str, Any]] = {}
    for dimension, sources in DIMENSION_DEFINITIONS.items():
        if not all(score_value(scores.get(metric)) is not None for metric in sources):
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
            name = str(name)
            if name in RESERVED_METRIC_NAMES or not custom_metric_name_is_publishable(name):
                continue
            if isinstance(value, dict):
                value = value.get("score")
            numeric = score_value(value)
            if numeric is not None:
                custom[name] = numeric

    metrics = reward.get("metrics")
    if isinstance(metrics, dict):
        for name, value in metrics.items():
            name = str(name)
            if name in RESERVED_METRIC_NAMES or not custom_metric_name_is_publishable(name):
                continue
            if isinstance(value, dict):
                value = value.get("score")
            numeric = score_value(value)
            if numeric is not None:
                custom[name] = numeric

    for name, value in reward.items():
        name = str(name)
        if name in RESERVED_METRIC_NAMES or name.startswith("_") or not custom_metric_name_is_publishable(name):
            continue
        if isinstance(value, dict):
            value = value.get("score")
        numeric = score_value(value)
        if numeric is not None:
            custom[name] = numeric

    return custom


def average_custom_metrics(rewards: list[dict[str, Any]]) -> dict[str, float]:
    """Average custom metrics across rewards inside the publication envelope."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for reward in rewards:
        if reason := custom_metric_contract_error(reward):
            raise CustomMetricContractError(reason)
        for name, value in extract_custom_metrics(reward).items():
            if name not in sums and len(sums) >= MAX_CUSTOM_METRICS:
                raise CustomMetricContractError("Custom metric union exceeds the per condition publication limit")
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    return {name: round(sums[name] / counts[name], 4) for name in sorted(sums)}
