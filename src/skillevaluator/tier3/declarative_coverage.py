# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded declarative-check coverage, independent of task success and skill lift."""

from __future__ import annotations

from typing import Any

from skillevaluator.evaluation.tier3_findings import sanitize_grader_text


def aggregate_declarative_checks(rewards: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep every check and each reward's denominator; never pool behavior scores."""
    totals = dict.fromkeys(
        ("checked_count", "passed_count", "failed_count", "judged_count", "not_checked_count", "score_denominator"), 0
    )
    results: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for reward in rewards:
        details = reward.get("details")
        behavior = details.get("behavior_check") if isinstance(details, dict) else None
        coverage = behavior.get("declarative_checks") if isinstance(behavior, dict) else None
        if not isinstance(coverage, dict) or not isinstance(coverage.get("results"), list):
            continue
        checks = [item for item in coverage["results"] if isinstance(item, dict)]
        identity = {
            "case_id": sanitize_grader_text(str(reward.get("entry_id") or "unknown")),
            "trial_id": sanitize_grader_text(str(reward.get("_trial_name") or reward.get("trial_id") or "")),
        }
        passed = sum(item.get("passed") is True for item in checks)
        failed = sum(item.get("passed") is False for item in checks)
        judged = coverage.get("judged_count")
        judged = judged if type(judged) is int and judged >= 0 else None
        denominator = passed + failed + judged if judged is not None else None
        counts = {
            "checked_count": passed + failed,
            "passed_count": passed,
            "failed_count": failed,
            "judged_count": judged,
            "not_checked_count": len(checks) - passed - failed,
            "score_denominator": denominator,
        }
        for key, value in counts.items():
            totals[key] = totals[key] + value if totals[key] is not None and value is not None else None
        attempts.append(
            {**identity, **counts, "score_status": sanitize_grader_text(behavior.get("status") or "unknown")}
        )
        for item in checks:
            verdict = item.get("passed")
            results.append(
                {
                    **identity,
                    "type": sanitize_grader_text(str(item.get("type") or "unknown")),
                    "expected": sanitize_grader_text(str(item.get("expected") or "")),
                    "passed": verdict if type(verdict) is bool else None,
                    "reason": sanitize_grader_text(str(item.get("reason") or "")),
                }
            )
    return {**totals, "attempts": attempts, "results": results} if results else {}


def agent_declarative_checks(
    with_rewards: list[dict[str, Any]], without_rewards: list[dict[str, Any]]
) -> dict[str, Any]:
    """Omit absent arms and omit the entire section for non-declarative runs."""
    return {
        arm: coverage
        for arm, rewards in (("with_skill", with_rewards), ("without_skill", without_rewards))
        if (coverage := aggregate_declarative_checks(rewards))
    }
