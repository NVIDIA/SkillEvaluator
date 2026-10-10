# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared payloads for the p01 gate report tests: synthetic Tier 3 scores built by the shipped payload builder."""

from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING, Any

from rich.console import Console

from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, build_agent_eval_payload
from skillevaluator.reporting.cli import CLIReporter

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult

METRICS = ("security", "skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check")
EDGE05_WITH = {
    "security": 1.0,
    "skill_execution": 0.9769,
    "skill_efficiency": 0.697,
    "accuracy": 0.5,
    "goal_accuracy": 0.5,
    "behavior_check": 0.5,
}
EDGE05_WITHOUT = {
    "security": 1.0,
    "skill_execution": 0.9769,
    "skill_efficiency": 0.6784,
    "accuracy": 1.0,
    "goal_accuracy": 1.0,
    "behavior_check": 1.0,
}
EDGE05_INTERVAL = {
    "effectiveness": {
        "estimate": -0.1963,
        "ci_low": -0.2,
        "ci_high": -0.1889,
        "confidence": 0.95,
        "n_cases": 9,
        "precision": "adequate",
        "ci_includes_zero": False,
    }
}


def scores(**overrides: float) -> dict[str, float]:
    scores = dict.fromkeys(METRICS, 1.0)
    scores.update(overrides)
    return scores


def build_payload(
    with_scores: dict[str, float],
    without_scores: dict[str, float] | None = None,
    **info_overrides: Any,
) -> dict[str, Any]:
    plugin_provenance = info_overrides.pop("plugin_provenance", None)
    info: dict[str, Any] = {
        "with_skill": dict(with_scores),
        "without_skill": dict(without_scores or {}),
        "execution_status": "succeeded",
        "rewards": [],
        "num_trials": 2,
        "model": "test-model",
        **info_overrides,
    }
    payload = build_agent_eval_payload(
        "gate-demo",
        {"codex": info},
        use_llm_judge=False,
        run_config={"eval_target": {"kind": "plugin"}},
        plugin_provenance=plugin_provenance,
    )
    assert payload is not None
    return payload


def tier3(payload: dict[str, Any], *, blocking: bool | None = None) -> ValidationResult:
    result = _validation_result_from_payload(payload)
    assert result is not None
    if blocking is not None:
        result.metadata["gating"] = {"tier": 3, "blocking": blocking}
    return result


def section(markdown: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", markdown, flags=re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


def terminal(results: list[ValidationResult]) -> str:
    stream = io.StringIO()
    CLIReporter(Console(file=stream, width=200, emoji=False, color_system=None)).print_all(results)
    return stream.getvalue()
