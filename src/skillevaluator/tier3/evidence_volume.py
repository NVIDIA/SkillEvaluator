from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from skillevaluator.tier3.dataset_sufficiency import MIN_DISTINCT_CASES


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def evidence_volume(
    *,
    cases: int | None,
    attempts_planned: int,
    attempts_scored: int,
    execution_failures: int,
    judge_failures: int,
    by_arm: dict[str, dict[str, Any]] | None = None,
    uniform_attempts: bool = True,
    attempts_per_case: int | None = None,
) -> dict[str, Any]:
    """Build a bounded evidence summary without changing any scores."""
    cases = _count(cases) or None if cases is not None else None
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
        "limited": (
            (cases is not None and cases < MIN_DISTINCT_CASES)
            or attempts_scored < attempts_planned
            or any(
                _count(arm.get("attempts_scored")) < _count(arm.get("attempts_planned"))
                for arm in (by_arm or {}).values()
            )
        ),
        "uniform_attempts": uniform_attempts,
        "attempts_per_case": _count(attempts_per_case) if attempts_per_case is not None else None,
    }
    result["display"] = format_evidence_volume(result)
    return result


def evidence_volume_for_conditions(
    conditions: object,
    *,
    trial_failures: object = None,
    baseline_only: bool = False,
    dataset_summary: object = None,
    rewards_by_arm: object = None,
    uniform_attempts: bool = True,
    attempts_per_case: int | None = None,
) -> dict[str, Any]:
    if not isinstance(conditions, Mapping):
        conditions = {}
    failures = trial_failures if isinstance(trial_failures, Mapping) else {}
    arms: dict[str, dict[str, Any]] = {}
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
        expected_cases = condition.get("expected_cases")
        scored_cases = condition.get("scored_cases")
        recorded_cases = any(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (expected_cases, scored_cases)
        )
        if recorded_cases:
            cases = max(_count(expected_cases), _count(scored_cases)) or None
        else:
            summary = dataset_summary if isinstance(dataset_summary, Mapping) else {}
            summary_count = _count(summary.get("total_tasks"))
            rewards = rewards_by_arm.get(arm) if isinstance(rewards_by_arm, Mapping) else None
            reward_cases = (
                {
                    str(reward.get("entry_id") or reward.get("case_id") or reward.get("task_id"))
                    for reward in rewards
                    if isinstance(reward, Mapping)
                    if reward.get("entry_id") or reward.get("case_id") or reward.get("task_id")
                }
                if isinstance(rewards, list)
                else set()
            )
            cases = summary_count or len(reward_cases) or None
        raw_judge_failures = condition.get("judge_failures")
        planned = _count(condition.get("expected_attempts"))
        scored = _count(condition.get("scored_attempts"))
        judge_count = (
            _count(raw_judge_failures)
            if isinstance(raw_judge_failures, int) and not isinstance(raw_judge_failures, bool)
            else len(judge_trials)
        )
        judge_count = min(judge_count, max(0, planned - scored))
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
        cases=summary.get("cases"),
        attempts_planned=summary.get("attempts_planned", 0),
        attempts_scored=summary.get("attempts_scored", 0),
        execution_failures=summary.get("execution_failures", 0),
        judge_failures=summary.get("judge_failures", 0),
        by_arm=arms,
        uniform_attempts=uniform_attempts,
        attempts_per_case=attempts_per_case,
    )


def format_evidence_volume(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    raw_cases = value.get("cases")
    cases = _count(raw_cases) or None if raw_cases is not None else None
    planned = _count(value.get("attempts_planned"))
    scored = _count(value.get("attempts_scored"))
    execution_failures = _count(value.get("execution_failures"))
    judge_failures = _count(value.get("judge_failures"))
    case_word = "case" if cases == 1 else "cases"
    attempt_word = "attempt" if scored == 1 else "attempts"
    if cases and not value.get("uniform_attempts", True):
        maximum = _count(value.get("attempts_per_case"))
        planned_text = f"{cases} {case_word}, {planned} attempts (up to {maximum} per case)"
    elif cases and planned % cases == 0:
        per_case = planned // cases
        planned_text = f"{cases} {case_word} \u00d7 {per_case} {'attempt' if per_case == 1 else 'attempts'}"
    else:
        planned_text = f"{cases} {case_word}"
    if cases is None:
        result = (
            f"Based on {scored} scored {'attempt' if scored == 1 else 'attempts'}"
            if scored == planned
            else f"Based on {scored} of {planned} {'attempt' if planned == 1 else 'attempts'} scored"
        )
    elif scored == planned:
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
            raw_arm_cases = details.get("cases")
            arm_cases = _count(raw_arm_cases) or None if raw_arm_cases is not None else None
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
            case_summary = f"{arm_cases} {'case' if arm_cases == 1 else 'cases'}, " if arm_cases is not None else ""
            arm_parts.append(
                f"{label}: {case_summary}{scored_text} "
                f"({arm_execution_failures} execution "
                f"{'failure' if arm_execution_failures == 1 else 'failures'}, "
                f"{arm_judge_failures} judge {'failure' if arm_judge_failures == 1 else 'failures'})"
            )
        if arm_parts:
            result += "; by arm: " + "; ".join(arm_parts)
    arm_limited = (
        any(
            isinstance(details, Mapping)
            and _count(details.get("attempts_scored")) < _count(details.get("attempts_planned"))
            for details in arms.values()
        )
        if isinstance(arms, Mapping)
        else False
    )
    if (
        value.get("limited") is True
        or (cases is not None and cases < MIN_DISTINCT_CASES)
        or scored < planned
        or arm_limited
    ):
        return f"Limited evidence: {result}"
    return result
