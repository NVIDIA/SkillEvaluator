# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Tier 3 attempt verdict rules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from skillevaluator.tier3.harbor.metrics import metric_value

DEFAULT_GOAL_ACCURACY_PASS_THRESHOLD = 1.0
DEFAULT_SECURITY_PASS_THRESHOLD = 1.0
RUN_FAILS_ON_ANY_FAILED_CASE = True


@dataclass(frozen=True)
class AttemptVerdict:
    passed: bool
    reason: str | None
    warnings: tuple[str, ...]
    correct_answer: bool | None
    security_clean: bool | None
    required_failure: str | None
    security_score: float | None
    security_message: str | None
    security_exempt: bool


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


def _scored_metric(reward: Mapping[str, Any], metric_name: str) -> float | None:
    details = reward.get("details")
    detail = details.get(metric_name) if isinstance(details, Mapping) else None
    if isinstance(detail, Mapping) and detail.get("status") in {"not_applicable", "error"}:
        return None
    metrics = reward.get("metrics")
    metric = metrics.get(metric_name) if isinstance(metrics, Mapping) else None
    if isinstance(metric, Mapping) and metric.get("status") in {"not_applicable", "error"}:
        return None
    errors = reward.get("evaluation_errors")
    if isinstance(errors, Mapping) and errors.get(metric_name):
        return None
    score = metric_value(dict(reward), metric_name)
    if (
        not isinstance(score, int | float)
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        return None
    return float(score)


def _unsupported_security_result(result: Mapping[str, Any]) -> bool:
    if result.get("supported") is False:
        return True
    evidence = result.get("unsupported_evidence")
    if isinstance(evidence, list) and "unsupported_native_codex_exec_wrapper" in evidence:
        return True
    findings = result.get("findings")
    score_findings = (
        [finding for finding in findings if isinstance(finding, Mapping) and finding.get("score_impact") is True]
        if isinstance(findings, list)
        else []
    )
    return len(score_findings) == 1 and (
        score_findings[0].get("message") == "Unsupported native Codex exec wrapper could not be safely normalized"
    )


def _security_finding_message(result: Mapping[str, Any]) -> str | None:
    findings = result.get("findings")
    if isinstance(findings, list):
        for finding in findings:
            if isinstance(finding, Mapping) and finding.get("score_impact") is True:
                message = finding.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
    reason = result.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return None


def _behavior_warning(score: float, result: Mapping[str, Any]) -> str:
    results = result.get("results")
    missed_steps = (
        sorted(
            {
                step
                for item in results
                if isinstance(item, Mapping)
                if item.get("passed") is False
                and isinstance((step := item.get("step")), int)
                and not isinstance(step, bool)
                and step > 0
            }
        )
        if isinstance(results, list)
        else []
    )
    total = len(results) if isinstance(results, list) else 0
    if missed_steps and total:
        label = "step" if len(missed_steps) == 1 else "steps"
        steps = ", ".join(str(step) for step in missed_steps)
        return f"missed expected {label} {steps} of {total} (behavior {score:.2f})"
    return f"missed expected behavior (behavior {score:.2f})"


def attempt_verdict(
    overall_score: float | None,
    pass_threshold: float,
    goal_accuracy: float | None = None,
    *,
    reward: Mapping[str, Any] | None = None,
    goal_accuracy_pass_threshold: float = DEFAULT_GOAL_ACCURACY_PASS_THRESHOLD,
) -> AttemptVerdict:
    """Apply required checks before the existing Overall threshold."""
    if reward is not None:
        goal_accuracy = scored_goal_accuracy(dict(reward))

    security_score = _scored_metric(reward, "security") if reward is not None else None
    details = reward.get("details") if reward is not None else None
    security_result = details.get("security") if isinstance(details, Mapping) else None
    security_result = security_result if isinstance(security_result, Mapping) else {}
    unsupported_security = _unsupported_security_result(security_result)
    security_clean = None if security_score is None else security_score >= DEFAULT_SECURITY_PASS_THRESHOLD
    security_exempt = security_clean is False and unsupported_security
    security_message = _security_finding_message(security_result) if security_clean is False else None

    behavior_score = _scored_metric(reward, "behavior_check") if reward is not None else None
    behavior_result = details.get("behavior_check") if isinstance(details, Mapping) else None
    behavior_result = behavior_result if isinstance(behavior_result, Mapping) else {}
    warnings = (
        (_behavior_warning(behavior_score, behavior_result),)
        if behavior_score is not None and behavior_score < 1.0
        else ()
    )

    correct_answer = None
    if goal_accuracy is not None:
        correct_answer = goal_accuracy >= goal_accuracy_pass_threshold
        if not correct_answer:
            return AttemptVerdict(
                passed=False,
                reason=f"goal not met (goal accuracy {goal_accuracy:.2f})",
                warnings=warnings,
                correct_answer=False,
                security_clean=security_clean,
                required_failure="wrong_answer",
                security_score=security_score,
                security_message=security_message,
                security_exempt=security_exempt,
            )
    if security_clean is False and security_score is not None and not unsupported_security:
        message = _security_finding_message(security_result)
        reason = f"security issue (security {security_score:.2f})"
        if message:
            reason += f": {message}"
        return AttemptVerdict(
            passed=False,
            reason=reason,
            warnings=warnings,
            correct_answer=correct_answer,
            security_clean=False,
            required_failure="security",
            security_score=security_score,
            security_message=security_message,
            security_exempt=False,
        )
    passed = overall_score is not None and overall_score >= pass_threshold
    reason = None
    if not passed:
        reason = (
            f"overall score {overall_score:.2f} below pass threshold {pass_threshold:.2f}"
            if overall_score is not None
            else "overall score was not available"
        )
    return AttemptVerdict(
        passed=passed,
        reason=reason,
        warnings=warnings,
        correct_answer=correct_answer,
        security_clean=security_clean,
        required_failure=None,
        security_score=security_score,
        security_message=security_message,
        security_exempt=security_exempt,
    )


def required_check_failure_reason(summary: Mapping[str, Any] | None) -> str | None:
    """Return the run-level required-check failure reason for one arm."""
    if not RUN_FAILS_ON_ANY_FAILED_CASE or not isinstance(summary, Mapping):
        return None
    goal_failures = summary.get("wrong_answer")
    security_failures = summary.get("security")
    scored_cases = summary.get("scored")
    counts = (goal_failures, security_failures, scored_cases)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        return None
    failed_cases = goal_failures + security_failures
    if failed_cases == 0 or failed_cases > scored_cases:
        return None
    reasons = []
    if goal_failures:
        noun = "goal not met" if goal_failures == 1 else "goals not met"
        reasons.append(f"{goal_failures} {noun}")
    if security_failures:
        noun = "security issue" if security_failures == 1 else "security issues"
        reasons.append(f"{security_failures} {noun}")
    return f"{failed_cases} of {scored_cases} cases failed: {', '.join(reasons)}"
