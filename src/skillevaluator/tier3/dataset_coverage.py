# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptive dataset coverage, independent of validity and run gates."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from skillevaluator.tier3.dataset_utils import find_eval_file, load_dataset_entries
from skillevaluator.tier3.eval_core.checks import resolve_should_trigger
from skillevaluator.tier3.eval_core.llm_judge import _has_judge_reference
from skillevaluator.tier3.evals_config import load_evals_config


def summarize_dataset(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Count normalized cases without interpreting any count as sufficiency."""
    count = len(entries)
    positive = sum(resolve_should_trigger(entry) is True for entry in entries)
    negative = sum(resolve_should_trigger(entry) is False for entry in entries)
    outcomes = sum(_has_judge_reference(entry.get("ground_truth")) for entry in entries)
    behaviors = sum(_has_judge_reference(entry.get("expected_behavior")) for entry in entries)
    gaps = []
    if not count:
        gaps.append("no evaluation cases")
    if not positive:
        gaps.append("no positive case that should trigger the skill")
    if not negative:
        gaps.append("no negative case that should not trigger the skill")
    unknown = count - positive - negative
    if unknown:
        gaps.append(f"{unknown} cases have no resolved should_trigger label")
    if count - outcomes:
        gaps.append(f"{count - outcomes} cases have no expected output, so accuracy and goal_accuracy are N/A for them")
    if count - behaviors:
        prefix = "no assertions; " if not behaviors else ""
        gaps.append(
            f"{prefix}{count - behaviors} cases have no assertions or expected behaviors, so behavior_check is N/A for them"
        )
    return {
        "case_count": count,
        "starter": count == 1,
        "positive_cases": positive,
        "negative_cases": negative,
        "unlabeled_trigger_cases": unknown,
        "cases_with_expected_outcome": outcomes,
        "cases_with_expected_behaviors": behaviors,
        "gaps": gaps,
    }


def coverage_text(coverage: dict[str, Any]) -> str:
    count = coverage["case_count"]
    lines = [
        f"Dataset coverage: {count} cases; {coverage['positive_cases']} positive, "
        f"{coverage['negative_cases']} negative/non-trigger, "
        f"{coverage['unlabeled_trigger_cases']} unlabeled trigger.",
        f"Expected outcome: {coverage['cases_with_expected_outcome']}/{count}; "
        f"assertions/expected behaviors: {coverage['cases_with_expected_behaviors']}/{count}.",
    ]
    if coverage["starter"]:
        lines.append("One-case starter: a starting point, not a representative evaluation.")
    lines.extend(f"Gap: {gap}." for gap in coverage["gaps"])
    return "\n".join(lines)


def dataset_coverage_result(skill_path: Path) -> dict[str, Any] | None:
    """Load authored entries; leave invalid and native sources to their validators."""
    try:
        config, _ = load_evals_config(skill_path)
        if (config.get("harbor") or {}).get("task_source") == "native_harbor":
            return None
        dataset = find_eval_file(skill_path)
        if dataset is None or dataset.parent.name != "evals":
            return None
        coverage = summarize_dataset(load_dataset_entries(dataset))
    except Exception:
        return None
    return {
        "path": f"{dataset.relative_to(skill_path)}#coverage",
        "status": "ok",
        "message": coverage_text(coverage),
        "coverage": coverage,
    }


def starter_guidance(skill_path: Path) -> str | None:
    result = dataset_coverage_result(skill_path)
    if result is None:
        return None
    path = shlex.quote(str(skill_path))
    return (
        result["message"] + "\n"
        "Add author guidance in evals/EVAL.md or pass --prompt before replacing the starter.\n"
        f"Replace the starter with a full dataset using: skillevaluator create-eval-dataset {path} --full --force\n"
        "`--force` overwrites the starter and discards any edits to it.\n"
        f"Inspect coverage with: skillevaluator tier3 validate {path}"
    )
