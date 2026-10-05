# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical metric definitions for Harbor-backed SkillEvaluator evals."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any

from skillevaluator.constants import DIMENSION_MAPPING

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

# A judged metric is "not applicable" when its dataset case has nothing to
# judge against (no ground_truth / expected_behavior). The verifier records the
# metric as ``null`` with ``details[metric].status == "not_applicable"``; N/A
# metrics are excluded from the trial overall, the arm averages, and lift.
# Deterministic metrics never become N/A, and an absent or non-finite metric
# without the explicit marker stays an incomplete (unscored) reward.
NOT_APPLICABLE_STATUS = "not_applicable"
NOT_APPLICABLE_ELIGIBLE_METRICS = ("accuracy", "goal_accuracy", "behavior_check")
NOT_APPLICABLE_REASONS = {
    "accuracy": "No ground_truth defined, so there is no reference answer to judge against",
    "goal_accuracy": "No ground_truth defined, so there is no expected outcome to judge against",
    "behavior_check": "No expected_behavior defined, so there is no workflow to check",
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


def finite_number(value: object, *, non_negative: bool = False) -> float | None:
    """Return *value* as a finite float, or ``None`` when it is not a usable number.

    Booleans, non-numbers, NaN, infinities and integers too large for a float
    (``json.loads`` turns any long digit string into one) are rejected, as are
    negative values when *non_negative* is set.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    if not math.isfinite(numeric) or (non_negative and numeric < 0):
        return None
    return numeric


def metric_value(reward: dict[str, Any], metric: str) -> float | None:
    """Return a numeric metric value from a reward payload, if present."""
    val = finite_number(reward.get(metric))
    if val is not None:
        return val

    metrics = reward.get("metrics")
    if isinstance(metrics, dict):
        raw = metrics.get(metric)
        if isinstance(raw, dict):
            raw = raw.get("score")
        numeric = finite_number(raw)
        if numeric is not None:
            return numeric

    return None


def metric_is_not_applicable(reward: dict[str, Any], metric: str) -> bool:
    """Return whether *reward* explicitly records a judged *metric* as not applicable.

    Only the LLM-judged metrics can be N/A, only when no numeric value is
    present, and only with the verifier's ``details[metric].status`` marker.
    Absence alone never means N/A.
    """
    if metric not in NOT_APPLICABLE_ELIGIBLE_METRICS:
        return False
    if reward.get(metric) is not None or metric_value(reward, metric) is not None:
        return False
    details = reward.get("details")
    detail = details.get(metric) if isinstance(details, dict) else None
    return (
        isinstance(detail, dict)
        and detail.get("status") == NOT_APPLICABLE_STATUS
        and finite_number(detail.get("score")) is None
    )


def not_applicable_metrics(
    rewards: list[dict[str, Any]],
    metrics: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Return metrics that every reward records as N/A, so no reward scores them."""
    if not rewards:
        return []
    if metrics is None:
        _, metrics = metric_set_for_rewards(rewards)
    return [
        metric
        for metric in metrics
        if metric in NOT_APPLICABLE_ELIGIBLE_METRICS
        and all(metric_is_not_applicable(reward, metric) for reward in rewards)
    ]


def not_applicable_list(raw: object) -> list[str]:
    """Return the judged metrics a stored ``not_applicable_metrics`` list names, in canonical order.

    A value that is not a list or tuple, and any name that cannot be N/A, is ignored.
    """
    if not isinstance(raw, list | tuple):
        return []
    return [metric for metric in NOT_APPLICABLE_ELIGIBLE_METRICS if metric in raw]


def not_applicable_counts(
    rewards: list[dict[str, Any]],
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    """Return how many rewards record each metric as N/A (metrics with none are omitted)."""
    if not rewards:
        return {}
    if metrics is None:
        _, metrics = metric_set_for_rewards(rewards)
    counts: dict[str, int] = {}
    for metric in metrics:
        count = sum(1 for reward in rewards if metric_is_not_applicable(reward, metric))
        if count:
            counts[metric] = count
    return counts


def mark_not_applicable(reward: dict[str, Any], metric: str) -> None:
    """Record *metric* as N/A on *reward* using the verifier's marker shape."""
    reward[metric] = None
    details = reward.get("details")
    if not isinstance(details, dict):
        details = {}
        reward["details"] = details
    existing = details.get(metric)
    detail = dict(existing) if isinstance(existing, dict) else {}
    detail.update(
        {
            "score": None,
            "status": NOT_APPLICABLE_STATUS,
            "reason": NOT_APPLICABLE_REASONS.get(metric, "Not applicable to this eval case"),
        }
    )
    details[metric] = detail


def dimension_is_not_applicable(dimension: str, not_applicable: tuple[str, ...] | list[str] | set[str]) -> bool:
    """Return whether every source metric of *dimension* is N/A."""
    sources = DIMENSION_DEFINITIONS.get(dimension)
    return bool(sources) and all(metric in not_applicable for metric in sources)


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
    if finite_number(reward.get("overall")) is not None:
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
    if any(finite_number(reward.get("overall")) is not None for reward in rewards):
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


def overall_score(reward: dict[str, Any]) -> float | None:
    """Compute pass@k/lift overall score for a reward payload.

    SkillEvaluator default rewards use the mean of their active SkillEvaluator metric set,
    excluding metrics explicitly recorded as not applicable.  Custom rewards
    without SkillEvaluator metrics can still pass through by emitting numeric
    ``overall``.
    """
    _, metrics = metric_set_for_reward(reward)
    if metrics:
        values: list[float] = []
        for metric in metrics:
            value = metric_value(reward, metric)
            if value is not None:
                values.append(value)
            elif not metric_is_not_applicable(reward, metric):
                return None
        if not values:
            return None
        return sum(values) / len(values)

    return finite_number(reward.get("overall"))


def score_definition(metrics: tuple[str, ...] = DEFAULT_METRICS) -> str:
    """Human-readable definition for the SkillEvaluator overall score."""
    if not metrics:
        return "overall = user-provided reward overall"
    definition = "overall = mean(" + ", ".join(metrics) + ")"
    if any(metric in NOT_APPLICABLE_ELIGIBLE_METRICS for metric in metrics):
        definition += ", excluding metrics that are not applicable to a case"
    return definition


def weighted_dimension_score(
    value_of: Callable[[str], object],
    config: Mapping[str, Any],
    *,
    active_metrics: Collection[str] | None = None,
) -> float | None:
    """Return one dimension's weighted evaluator mean, or ``None`` when no evaluator is scored.

    *config* is a ``DIMENSION_MAPPING`` entry and *value_of* returns an
    evaluator's score; anything other than a finite number counts as unscored.
    Unscored evaluators and non-finite weights are skipped, so the remaining
    weights renormalize.

    The ``fallback_evaluators`` (``behavior_check`` for a legacy reward without
    ``security``) replace the primary evaluators when none of those belongs to
    *active_metrics* (a reward's metric set). Without *active_metrics*, they
    replace them when no primary evaluator is scored. Both rules agree for a
    complete reward, which scores every metric of its set that is not N/A.
    """
    evaluators = config.get("evaluators") or ()
    weights = config.get("weights") or ()
    fallback_evaluators = config.get("fallback_evaluators") or ()
    fallback_weights = config.get("fallback_weights") or ()
    if active_metrics is not None:
        if fallback_evaluators and not any(evaluator in active_metrics for evaluator in evaluators):
            return _weighted_mean(value_of, fallback_evaluators, fallback_weights)
        return _weighted_mean(value_of, evaluators, weights)
    score = _weighted_mean(value_of, evaluators, weights)
    if score is None and fallback_evaluators:
        score = _weighted_mean(value_of, fallback_evaluators, fallback_weights)
    return score


def _weighted_mean(
    value_of: Callable[[str], object],
    evaluators: Iterable[str],
    weights: Iterable[object],
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for evaluator, weight in zip(evaluators, weights, strict=False):
        value = finite_number(value_of(evaluator))
        numeric_weight = finite_number(weight)
        if value is None or numeric_weight is None:
            continue
        numerator += value * numeric_weight
        denominator += numeric_weight
    return numerator / denominator if denominator > 0 else None


def dimension_scores(
    scores: dict[str, float],
    not_applicable: tuple[str, ...] | list[str] | set[str] = (),
) -> dict[str, dict[str, Any]]:
    """Compute report-only SkillEvaluator dimension scores from default metric scores.

    Source metrics that are N/A for the whole arm (and so absent from *scores*)
    drop out and the remaining weights are renormalized. A dimension whose
    sources are all N/A is omitted, as is one with any other missing source.
    """
    out: dict[str, dict[str, Any]] = {}
    for dimension, configured_sources in DIMENSION_DEFINITIONS.items():
        sources = {
            metric: weight
            for metric, weight in configured_sources.items()
            if not (metric in not_applicable and finite_number(scores.get(metric)) is None)
        }
        if not sources or not all(finite_number(scores.get(metric)) is not None for metric in sources):
            continue
        total_weight = sum(sources.values())
        if total_weight <= 0:
            continue
        score = sum(float(scores[metric]) * weight for metric, weight in sources.items()) / total_weight
        entry: dict[str, Any] = {
            "score": round(score, 4),
            "sources": sources,
        }
        skipped = [metric for metric in configured_sources if metric not in sources]
        if skipped:
            entry["not_applicable_sources"] = skipped
        out[dimension] = entry
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
            numeric = finite_number(value)
            if numeric is not None:
                custom[str(name)] = numeric

    metrics = reward.get("metrics")
    if isinstance(metrics, dict):
        for name, value in metrics.items():
            if name in RESERVED_METRIC_NAMES:
                continue
            if isinstance(value, dict):
                value = value.get("score")
            numeric = finite_number(value)
            if numeric is not None:
                custom[str(name)] = numeric

    for name, value in reward.items():
        if name in RESERVED_METRIC_NAMES or name.startswith("_"):
            continue
        numeric = finite_number(value)
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
