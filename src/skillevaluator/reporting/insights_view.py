# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only, provider-free presentation of persisted Tier 3 insights."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from skillevaluator.reporting.tier3_coverage import incomplete_arms

__all__ = ["build_insights_view"]

_FIELDS = ("conclusions", "recommendations", "next_steps", "execution_errors")


def _records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _coverage(record: dict[str, Any]) -> tuple[int, int] | None:
    scored, expected = record.get("scored_attempts"), record.get("expected_attempts")
    if type(scored) is int and type(expected) is int and 0 <= scored <= expected and expected > 0:
        return scored, expected
    return None


def _run_coverage(payload: dict, summary: dict, agents: dict) -> tuple[int, int] | None:
    fallback = _coverage(payload) or _coverage(summary)
    if not agents:
        return fallback
    totals = []
    mode = payload.get("mode")
    arms = {
        "with_skill_only": ("with_skill",),
        "baseline_only": ("without_skill",),
    }.get(mode if isinstance(mode, str) else None, ("with_skill", "without_skill"))
    for agent in agents.values():
        if not isinstance(agent, dict):
            return fallback
        conditions = agent.get("conditions")
        condition_counts = []
        if isinstance(conditions, dict):
            # Explicit retained arm evidence takes precedence over an older mode.
            agent_arms = [arm for arm in ("with_skill", "without_skill") if arm in conditions or arm in arms]
            for arm in agent_arms:
                record = conditions.get(arm)
                if isinstance(record, dict) and record.get("execution_status") == "skipped":
                    continue
                counts = _coverage(record) if isinstance(record, dict) else None
                if counts is None:
                    condition_counts = []
                    break
                condition_counts.append(counts)
        if condition_counts:
            totals.append((sum(s for s, _ in condition_counts), sum(e for _, e in condition_counts)))
        elif counts := _coverage(agent):
            totals.append(counts)
        else:
            return fallback
    expected = sum(expected for _, expected in totals)
    return (sum(scored for scored, _ in totals), expected) if expected else fallback


def _unique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Compare the whole record: identical prose is not identical evidence.
    result: list[dict[str, Any]] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _matches_error_sequence(message: object, errors: object) -> bool:
    """Match whole structured errors, including a prefix saved before later errors.

    Delimiters inside an error are its original prose, never new error records.
    """
    if not isinstance(message, str) or not isinstance(errors, list) or not errors:
        return False
    offset = 0
    for index, error in enumerate(errors):
        if not isinstance(error, str):
            return False
        piece = ("; " if index else "") + error
        if not message.startswith(piece, offset):
            return False
        offset += len(piece)
        if offset == len(message):
            return True
    return False


def build_insights_view(payload: dict) -> dict:
    """Project concise failed-run cards, retaining original diagnostics separately.

    Healthy feedback is copied verbatim. Failed runs replace only recognized
    deterministic error-blob copies; authored prose is never split or rewritten.
    This function reads only the supplied object and never opens artifacts,
    invokes an insights provider, or changes canonical evaluation evidence.
    """
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    view = {key: deepcopy(payload.get(key, [])) for key in _FIELDS}
    if not isinstance(view["execution_errors"], list):
        view["execution_errors"] = []
    if "execution_errors" not in payload:
        view["execution_errors"] = deepcopy(summary.get("execution_errors") or [])
    if payload.get("execution_status", summary.get("execution_status")) != "failed":
        return view
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    if (
        payload.get("reason_code", summary.get("reason_code")) == "eval_dataset_required"
        or provenance.get("source") == "tier3_source_preflight"
    ):
        # A blocking source check fails publication before any agent runs.
        return view

    agents = payload.get("agents") if isinstance(payload.get("agents"), dict) else {}
    error_sources = [
        view["execution_errors"],
        *[agent.get("execution_errors") for agent in agents.values() if isinstance(agent, dict)],
    ]

    def is_blob(item: dict[str, Any], *, conclusion: bool) -> bool:
        if any(item.get(key) for key in ("agent", "case", "case_id", "condition", "metric")):
            return False
        if item.get("source") not in (None, "deterministic"):
            return False
        if not any(_matches_error_sequence(item.get("message"), errors) for errors in error_sources):
            return False
        return not conclusion or item.get("title") == "Evaluation incomplete"

    old_conclusions = _records(view["conclusions"])
    failed_arms = [(name, arm) for name, agent in agents.items() for arm in incomplete_arms(agent)]

    def refreshed_coverage(item: dict[str, Any]) -> bool:
        return item.get("insight_kind") == "execution_coverage" and any(
            item.get("agent") == name and item.get("condition") == arm.condition for name, arm in failed_arms
        )

    retained_conclusions = [
        item
        for item in old_conclusions
        if not is_blob(item, conclusion=True)
        and item.get("insight_kind") != "execution_summary"
        and not refreshed_coverage(item)
    ]
    counts = _run_coverage(payload, summary, agents)
    count_text = (
        f"{counts[0]} of {counts[1]} attempts scored; {counts[1] - counts[0]} of {counts[1]} attempts remain unscored."
        if counts
        else "Attempt coverage is unavailable."
    )
    headline = {
        "title": "Evaluation incomplete",
        "message": f"Tier 3 execution failed. {count_text}",
        "severity": "fail",
        "source": "deterministic",
        "category": "Execution",
        "metric": "agent_eval",
        "insight_kind": "execution_summary",
    }
    # Retain every distinct trace association, including on repeated projection.
    conclusions = []
    for item in old_conclusions:
        if (is_blob(item, conclusion=True) or item.get("insight_kind") == "execution_summary") and item.get("evidence"):
            conclusions.append({**headline, "evidence": item["evidence"]})
    if not conclusions:
        conclusions.append(headline)
    for name, arm in failed_arms:
        coverage = {
            "title": f"{name} · {'With skill' if arm.condition == 'with_skill' else 'Baseline'}",
            "message": f"{name}: {arm}.",
            "severity": "fail",
            "source": "deterministic",
            "category": "Execution",
            "metric": "agent_eval",
            "agent": name,
            "condition": arm.condition,
            "insight_kind": "execution_coverage",
        }
        matches = [
            item
            for item in old_conclusions
            if refreshed_coverage(item) and item.get("agent") == name and item.get("condition") == arm.condition
        ]
        if matches:
            conclusions.extend(
                {**coverage, **item, "title": coverage["title"], "message": coverage["message"]} for item in matches
            )
        else:
            conclusions.append(coverage)
    view["conclusions"] = _unique([*conclusions, *retained_conclusions])
    steps = [
        step for step in _records(view["next_steps"]) if isinstance(step.get("action"), str) and step["action"].strip()
    ]
    fallback_action = "Check the agent and judge logs, fix execution failures, then rerun the evaluation."
    recommendations = []
    for item in _records(view["recommendations"]):
        if is_blob(item, conclusion=False):
            if item.get("evidence"):
                recommendations.append(
                    {
                        **item,
                        "title": "Restore complete execution",
                        "message": fallback_action,
                        "insight_kind": "execution_action",
                    }
                )
        elif not (steps and item.get("insight_kind") == "execution_fallback"):
            recommendations.append(item)
    for step in steps:
        recommendation = {
            "title": step.get("kind") or "Next step",
            "message": step["action"],
            "severity": "warn",
            "source": "deterministic",
            "category": "Action",
            "metric": step.get("target") or "agent_eval",
            "insight_kind": "execution_action",
        }
        for key in ("agent", "case", "case_id", "condition", "check", "strength"):
            if key in step:
                recommendation[key] = step[key]
        if step.get("evidence"):
            key = "evidence" if isinstance(step["evidence"], dict) else "evidence_text"
            recommendation[key] = step["evidence"]
        recommendations.append(recommendation)
    if not recommendations:
        recommendations.append(
            {
                "title": "Restore complete execution",
                "category": "Fix",
                "severity": "warn",
                "source": "deterministic",
                "message": fallback_action,
                "metric": "agent_eval",
                "insight_kind": "execution_fallback",
            }
        )
    view["recommendations"] = _unique(recommendations)
    return view
