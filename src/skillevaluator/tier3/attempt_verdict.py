# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Tier 3 attempt verdict rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from skillevaluator.tier3.harbor.metrics import metric_value

DEFAULT_GOAL_ACCURACY_PASS_THRESHOLD = 1.0


@dataclass(frozen=True)
class AttemptVerdict:
    passed: bool
    reason: str | None
    correct_answer: bool | None


def scored_goal_accuracy(reward: dict[str, Any]) -> float | None:
    """Return a scored goal-accuracy value, excluding explicit N/A and errors."""
    details = reward.get("details")
    detail = details.get("goal_accuracy") if isinstance(details, dict) else None
    if isinstance(detail, dict) and detail.get("status") in {"not_applicable", "error"}:
        return None
    metrics = reward.get("metrics")
    metric = metrics.get("goal_accuracy") if isinstance(metrics, dict) else None
    if isinstance(metric, dict) and metric.get("status") in {"not_applicable", "error"}:
        return None
    errors = reward.get("evaluation_errors")
    if isinstance(errors, dict) and errors.get("goal_accuracy"):
        return None
    score = metric_value(reward, "goal_accuracy")
    if (
        not isinstance(score, int | float)
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        return None
    return float(score)


def attempt_verdict(
    overall_score: float | None,
    pass_threshold: float,
    goal_accuracy: float | None,
    *,
    goal_accuracy_pass_threshold: float = DEFAULT_GOAL_ACCURACY_PASS_THRESHOLD,
) -> AttemptVerdict:
    """Apply the goal-accuracy gate before the existing Overall threshold."""
    correct_answer = None
    if goal_accuracy is not None:
        correct_answer = goal_accuracy >= goal_accuracy_pass_threshold
        if not correct_answer:
            return AttemptVerdict(
                passed=False,
                reason=f"wrong answer (goal accuracy {goal_accuracy:.2f})",
                correct_answer=False,
            )
    return AttemptVerdict(
        passed=overall_score is not None and overall_score >= pass_threshold,
        reason=None,
        correct_answer=correct_answer,
    )
