# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join authored requirements to recorded verdicts, never to aggregate scores."""

from __future__ import annotations

from collections import Counter
from typing import Any

from skillevaluator.evaluation.tier3_findings import sanitize_grader_text

_MAX_CASES = 512
_MAX_ATTEMPTS = 512
_MAX_EXPECTATIONS = 128
_MAX_JUDGMENTS = 8192
_NEXT_STEPS = {
    "met": "No correction indicated by this check.",
    "unmet": (
        "Compare the recorded evidence with this requirement. If the requirement is valid, inspect the skill or "
        "agent execution; if it is inappropriate for this prompt, correct the test. Ownership is not established by the score."
    ),
    "not_checked": "Inspect execution and judge artifacts, then rerun this check before diagnosing a skill defect.",
    "unscored": "Inspect the grader configuration and score output; no scored verdict supports a skill defect.",
    "not_applicable": "Confirm this check is intentionally inapplicable to this condition; do not count it as a failure.",
}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _requirement_text(value: Any) -> str:
    # Keep authored whitespace, unlike compact grader reasons.
    from skillevaluator.tier3.eval_core.llm_judge import _redact_configured_credentials
    from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line
    from skillevaluator.utils.redaction import redact_sensitive_text

    return redact_sensitive_text(redact_secrets_in_log_line(_redact_configured_credentials(str(value))))


def _requirements(case: dict[str, Any]) -> list[dict[str, Any]]:
    requirements = []
    for field, fallback, metric in (
        ("expected_output", "ground_truth", "goal_accuracy"),
        ("assertions", "expected_behavior", "behavior_check"),
        ("expected_skill", "expected_skill", "skill_execution"),
        ("expected_script", "expected_script", "skill_execution"),
    ):
        source = field if field in case else fallback
        value = case.get(source)
        if field == "expected_skill" and source in case:
            text = str(value) if value is not None else "Do not activate the evaluated skill (expected_skill: null)"
            requirements.append({"expectation_id": source, "requirement": _requirement_text(text), "metric": metric})
        elif field == "assertions" and isinstance(value, list):
            for index, text in enumerate(value):
                requirements.append(
                    {
                        "expectation_id": f"{source}/{index}",
                        "requirement": str(text),
                        "metric": metric,
                        "step": index + 1,
                        "source_conflict": fallback in case and case[fallback] != value,
                    }
                )
        elif value is not None and str(value).strip():
            requirements.append(
                {
                    "expectation_id": source,
                    "requirement": str(value),
                    "metric": metric,
                    "source_conflict": fallback in case and case[fallback] != value,
                }
            )
            if metric == "goal_accuracy":
                requirements.append(
                    {
                        "expectation_id": f"{source}/accuracy",
                        "requirement": str(value),
                        "metric": "accuracy",
                        "source_conflict": fallback in case and case[fallback] != value,
                    }
                )
    return requirements


def _judgment(
    requirement: dict[str, Any], reward: dict[str, Any], *, ambiguous: bool = False, missing: bool = False
) -> dict[str, Any]:
    from skillevaluator.tier3.harbor.metrics import metric_set_for_reward, metric_value

    metric = requirement["metric"]
    detail = _mapping(_mapping(reward.get("details")).get(metric))
    check = detail
    if metric == "behavior_check":
        results = detail.get("results")
        matches = (
            [
                item
                for item in results
                if isinstance(item, dict) and type(item.get("step")) is int and item["step"] == requirement.get("step")
            ]
            if isinstance(results, list)
            else []
        )
        check = matches[0] if len(matches) == 1 else {}
    elif metric == "skill_execution":
        key = (
            "script_execution"
            if requirement["expectation_id"] == "expected_script"
            else ("negative_check" if detail.get("should_trigger") is False else "activation")
        )
        check = _mapping(detail.get(key))

    reason = sanitize_grader_text(check.get("reason") or detail.get("reason") or detail.get("message"), 512)
    criteria = _mapping(detail.get("criteria")) if metric == "accuracy" else {}
    excluded = detail.get("excluded_criteria")
    excluded = excluded if isinstance(excluded, list) else []
    applicable_criteria = {key: value for key, value in criteria.items() if key not in excluded}
    metric_error = _mapping(reward.get("evaluation_errors")).get(metric)
    score = metric_value(reward, metric)
    if requirement.get("source_conflict"):
        status, reason = (
            "not_checked",
            "The dataset's current and legacy requirement fields disagree; verify the test input",
        )
    elif missing:
        status, reason = "not_checked", "No retained trial result could be matched to this case"
    elif ambiguous:
        status, reason = "not_checked", "Multiple reward rows cannot be matched to this requirement unambiguously"
    elif reward.get("error"):
        status, reason = "not_checked", sanitize_grader_text(reward["error"], 512)
    elif detail.get("status") == "error" or metric_error:
        status = "not_checked"
        reason = sanitize_grader_text(metric_error, 512) or reason or "Judge failed"
    elif detail.get("status") == "not_applicable" and score is None:
        status = "not_applicable"
    elif metric not in metric_set_for_reward(reward)[1] or score is None:
        status = "unscored"
        reason = reason or "This grader did not record a score for the requirement's metric"
    else:
        passed = check.get("achieved") if metric == "goal_accuracy" else check.get("passed")
        if metric == "accuracy" and applicable_criteria and all(type(v) is bool for v in applicable_criteria.values()):
            passed = all(applicable_criteria.values())
        if type(passed) is bool:
            status = "met" if passed else "unmet"
        else:
            status = "not_checked"
            reason = reason or "No unique explicit verdict was recorded for this expectation"

    evidence = []
    refs = detail.get("evidence_refs")
    if isinstance(refs, list) and status in {"met", "unmet"}:
        for ref in refs[:4]:
            if not isinstance(ref, dict) or ref.get("kind") in {"expected_behavior", "ground_truth"}:
                continue
            text = sanitize_grader_text(ref.get("excerpt"), 300)
            if text:
                evidence.append(text)
    return {
        "expectation_id": requirement["expectation_id"],
        "status": status,
        "reason": reason or "No reason recorded",
        "evidence": evidence,
        "unmet_checks": [sanitize_grader_text(key, 128) for key, value in applicable_criteria.items() if value is False]
        if status == "unmet"
        else [],
        "inapplicable_checks": [sanitize_grader_text(key, 128) for key in excluded[:16]],
        "next_step": _NEXT_STEPS[status],
    }


def build_case_diagnostics(
    dataset: list[dict[str, Any]], agents: dict[str, dict[str, Any]], *, evaluation_mode: str = "both"
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from skillevaluator.tier3.harbor.report_data import logical_trial_reward_groups

    condition_trials = []
    for agent, info in agents.items():
        for condition, key in (("with_skill", "rewards"), ("without_skill", "rewards_baseline")):
            if evaluation_mode == "baseline_only" and condition == "with_skill":
                continue
            if evaluation_mode == "with_skill_only" and condition == "without_skill":
                continue
            conditions = _mapping(info.get("conditions"))
            if _mapping(conditions.get(condition)).get("execution_status") == "skipped":
                continue
            if not info.get(key) and condition not in conditions:
                continue
            rows = info.get(key)
            groups_by_case: dict[str, list[list[dict[str, Any]]]] = {}
            groups = (
                logical_trial_reward_groups([row for row in rows if isinstance(row, dict)])
                if isinstance(rows, list)
                else []
            )
            for group in groups:
                for entry_id in {row["entry_id"] for row in group if isinstance(row.get("entry_id"), str)}:
                    groups_by_case.setdefault(entry_id, []).append(group)
            errors = _mapping(conditions.get(condition)).get("execution_errors")
            condition_trials.append((agent, condition, groups_by_case, errors))

    omitted: Counter[str] = Counter()
    cases = []
    attempts_left, judgments_left = _MAX_ATTEMPTS, _MAX_JUDGMENTS
    for case in dataset[:_MAX_CASES]:
        entry_id = case.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            continue
        requirements = _requirements(case)
        omitted["expectations"] += max(0, len(requirements) - _MAX_EXPECTATIONS)
        requirements = requirements[:_MAX_EXPECTATIONS]
        for requirement in requirements:
            requirement["requirement"] = _requirement_text(requirement["requirement"])
        attempts = []
        for agent, condition, groups_by_case, errors in condition_trials:
            for group in groups_by_case.get(entry_id) or [[]]:
                if attempts_left <= 0 or judgments_left < len(requirements):
                    omitted["expectation_attempts"] += 1
                    continue
                attempts_left -= 1
                judgments_left -= len(requirements)
                reward = group[0] if group else {}
                attempts.append(
                    {
                        "agent": agent,
                        "condition": condition,
                        "trial_id": reward.get("trial_id"),
                        "errors": [sanitize_grader_text(error, 512) for error in errors[:4]]
                        if not group and isinstance(errors, list)
                        else [],
                        "judge_note": sanitize_grader_text(
                            _mapping(_mapping(reward.get("details")).get("behavior_check")).get("reason"), 512
                        ),
                        "judgments": [
                            _judgment(requirement, reward, ambiguous=len(group) > 1, missing=not group)
                            for requirement in requirements
                        ],
                    }
                )
        cases.append({"entry_id": entry_id, "requirements": requirements, "attempts": attempts})
    omitted["expectation_cases"] += max(0, len(dataset) - _MAX_CASES)
    return cases, {key: value for key, value in omitted.items() if value}
