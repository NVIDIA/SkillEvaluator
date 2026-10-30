# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display completed-agent evidence separately from the publication decision."""

from __future__ import annotations

import math
from typing import Any


def validate_report_scope(scope: str, results: list) -> None:
    if scope not in {"publication", "tier3"}:
        raise ValueError("report_scope must be publication or tier3")
    if scope == "tier3" and any(result.validator_name != "AGENT_EVAL" for result in results):
        raise ValueError("Tier 3 report scope cannot hide results from other tiers")


def completed_agent_result(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a display-only summary of the canonical best completed agent.

    This never changes the payload, execution status, or publication assessment.
    Do not fill gaps using another agent, incompatible metric sets, or failed trials.
    """
    if not isinstance(payload, dict):
        return None
    agents = payload.get("agents")
    name = payload.get("best_agent")
    if not isinstance(agents, dict) or not isinstance(name, str):
        return None
    agent = agents.get(name)
    if not isinstance(agent, dict) or agent.get("execution_status") != "succeeded":
        return None
    if agent.get("execution_errors") or agent.get("execution_error_details_total"):
        return None
    score = agent.get("with_skill")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        return None
    expected, scored = agent.get("expected_attempts"), agent.get("scored_attempts")
    if type(expected) is not int or type(scored) is not int or expected <= 0 or expected != scored:
        return None
    conditions = agent.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) not in ({"with_skill"}, {"with_skill", "without_skill"}):
        return None
    if "without_skill" not in conditions:
        # A standalone --skip-baseline run writes only the with-skill summary.
        # Permit its completed score, but never manufacture a baseline or lift.
        conditions = {
            **conditions,
            "without_skill": {
                "execution_status": "skipped",
                "execution_errors": [],
                "expected_attempts": 0,
                "scored_attempts": 0,
            },
        }
    for arm, condition in conditions.items():
        if (
            not isinstance(condition, dict)
            or condition.get("execution_errors")
            or condition.get("execution_error_details_total")
        ):
            return None
        n, actual = condition.get("expected_attempts"), condition.get("scored_attempts")
        if type(n) is not int or type(actual) is not int or n != actual:
            return None
        if arm == "without_skill" and condition.get("execution_status") == "skipped" and n == 0:
            continue
        if condition.get("execution_status") != "succeeded" or n <= 0:
            return None
    if sum(c["scored_attempts"] for c in conditions.values()) != scored:
        return None
    baseline, lift = agent.get("baseline"), agent.get("lift")
    comparable = (
        conditions["without_skill"].get("execution_status") == "succeeded"
        and bool(agent.get("metrics_with_skill"))
        and agent.get("metrics_with_skill") == agent.get("metrics_without_skill")
    )
    if (
        not comparable
        or isinstance(baseline, bool)
        or not isinstance(baseline, (int, float))
        or not math.isfinite(baseline)
        or not 0 <= baseline <= 1
    ):
        baseline, lift = None, None
    if isinstance(lift, bool) or not isinstance(lift, (int, float)) or not math.isfinite(lift) or not -1 <= lift <= 1:
        lift = None
    from skillevaluator.reporting.base import (
        _agent_eval_condition_attempt_coverage_complete,
        _agent_eval_dataset_count,
    )

    policy = payload.get("attempt_policy")
    if not isinstance(policy, dict):
        return None
    max_attempts, stop_on_pass = policy.get("max_attempts"), policy.get("stop_on_pass")
    cases = _agent_eval_dataset_count(payload)
    if type(max_attempts) is not int or max_attempts <= 0 or type(stop_on_pass) is not bool or cases <= 0:
        return None
    if not _agent_eval_condition_attempt_coverage_complete(
        {**agent, "conditions": conditions},
        expected_attempts=expected,
        scored_attempts=scored,
        dataset_tasks=cases,
        max_attempts=max_attempts,
        stop_on_pass=stop_on_pass,
        require_baseline=False,
    ):
        return None
    from skillevaluator.evaluation.tier3_report import _agent_quality_verdict

    return {
        "name": name,
        "score": score,
        "baseline": baseline,
        "lift": lift,
        "verdict": _agent_quality_verdict(agent),
        "needs_attention": payload.get("execution_status") != "succeeded",
        "recovered_attempts": sum(
            c.get("recovered_attempts_total", 0)
            for c in conditions.values()
            if type(c.get("recovered_attempts_total", 0)) is int and c.get("recovered_attempts_total", 0) >= 0
        ),
    }
