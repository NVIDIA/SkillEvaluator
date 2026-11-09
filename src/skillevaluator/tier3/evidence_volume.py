from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from skillevaluator.tier3.dataset_sufficiency import MIN_DISTINCT_CASES


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def evidence_volume(
    *,
    cases: int,
    attempts_planned: int,
    attempts_scored: int,
    execution_failures: int,
    judge_failures: int,
    by_arm: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Build a bounded evidence summary without changing any scores."""
    cases = _count(cases)
    attempts_planned = _count(attempts_planned)
    attempts_scored = _count(attempts_scored)
    execution_failures = _count(execution_failures)
    judge_failures = _count(judge_failures)
    result = {
        "cases": cases,
        "attempts_planned": attempts_planned,
        "attempts_scored": attempts_scored,
        "execution_failures": execution_failures,
        "judge_failures": judge_failures,
        "by_arm": by_arm or {},
        "limited": cases < MIN_DISTINCT_CASES or attempts_scored < attempts_planned,
    }
    result["display"] = format_evidence_volume(result)
    return result


def evidence_volume_for_conditions(
    conditions: object,
    *,
    trial_failures: object = None,
    baseline_only: bool = False,
) -> dict[str, Any]:
    if not isinstance(conditions, Mapping):
        conditions = {}
    failures = trial_failures if isinstance(trial_failures, Mapping) else {}
    arms: dict[str, dict[str, int]] = {}
    for arm in ("with_skill", "without_skill"):
        condition = conditions.get(arm)
        if not isinstance(condition, Mapping) or condition.get("execution_status") == "skipped":
            continue
        arm_failures = failures.get(arm)
        judge_trials = (
            {
                str(item.get("trial"))
                for item in arm_failures
                if isinstance(item, Mapping)
                and str(item.get("reason") or "").startswith("Required judge evaluation failed:")
            }
            if isinstance(arm_failures, list)
            else set()
        )
        cases = max(
            _count(condition.get("expected_cases")),
            _count(condition.get("scored_cases")),
            _count(condition.get("failure_cases")),
        )
        raw_judge_failures = condition.get("judge_failures")
        judge_count = (
            _count(raw_judge_failures)
            if isinstance(raw_judge_failures, int) and not isinstance(raw_judge_failures, bool)
            else len(judge_trials)
        )
        planned = _count(condition.get("expected_attempts"))
        scored = _count(condition.get("scored_attempts"))
        arms[arm] = {
            "cases": cases,
            "attempts_planned": planned,
            "attempts_scored": scored,
            "execution_failures": max(0, planned - scored - judge_count),
            "judge_failures": judge_count,
        }
    primary = "without_skill" if baseline_only else "with_skill"
    if primary not in arms and arms:
        primary = next(iter(arms))
    summary = arms.get(primary, {})
    return evidence_volume(
        cases=summary.get("cases", 0),
        attempts_planned=summary.get("attempts_planned", 0),
        attempts_scored=summary.get("attempts_scored", 0),
        execution_failures=summary.get("execution_failures", 0),
        judge_failures=summary.get("judge_failures", 0),
        by_arm=arms,
    )


def format_evidence_volume(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    cases = _count(value.get("cases"))
    planned = _count(value.get("attempts_planned"))
    scored = _count(value.get("attempts_scored"))
    execution_failures = _count(value.get("execution_failures"))
    judge_failures = _count(value.get("judge_failures"))
    case_word = "case" if cases == 1 else "cases"
    attempt_word = "attempt" if scored == 1 else "attempts"
    if cases and planned % cases == 0:
        per_case = planned // cases
        planned_text = f"{cases} {case_word} \u00d7 {per_case} {'attempt' if per_case == 1 else 'attempts'}"
    else:
        planned_text = f"{cases} {case_word}"
    if scored == planned:
        result = f"Based on {planned_text} = {scored} scored {attempt_word}"
    else:
        result = f"Based on {planned_text} = {scored} of {planned} {'attempt' if planned == 1 else 'attempts'} scored"
    result += (
        f" ({execution_failures} execution {'failure' if execution_failures == 1 else 'failures'}, "
        f"{judge_failures} judge {'failure' if judge_failures == 1 else 'failures'})"
    )
    arms = value.get("by_arm")
    if isinstance(arms, Mapping) and "without_skill" in arms:
        arm_parts = []
        for arm in ("with_skill", "without_skill"):
            details = arms.get(arm)
            if not isinstance(details, Mapping):
                continue
            arm_cases = _count(details.get("cases"))
            arm_planned = _count(details.get("attempts_planned"))
            arm_scored = _count(details.get("attempts_scored"))
            arm_execution_failures = _count(details.get("execution_failures"))
            arm_judge_failures = _count(details.get("judge_failures"))
            label = "with skill" if arm == "with_skill" else "without skill"
            scored_text = (
                f"{arm_scored} scored {'attempt' if arm_scored == 1 else 'attempts'}"
                if arm_scored == arm_planned
                else f"{arm_scored} of {arm_planned} {'attempt' if arm_planned == 1 else 'attempts'} scored"
            )
            arm_parts.append(
                f"{label}: {arm_cases} {'case' if arm_cases == 1 else 'cases'}, "
                f"{scored_text} "
                f"({arm_execution_failures} execution "
                f"{'failure' if arm_execution_failures == 1 else 'failures'}, "
                f"{arm_judge_failures} judge {'failure' if arm_judge_failures == 1 else 'failures'})"
            )
        if arm_parts:
            result += "; by arm: " + "; ".join(arm_parts)
    if value.get("limited") is True or cases < MIN_DISTINCT_CASES or scored < planned:
        return f"Limited evidence: {result}"
    return result
