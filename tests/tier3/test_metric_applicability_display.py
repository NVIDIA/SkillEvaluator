# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import re

from rich.console import Console

from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, dimension_scores
from skillevaluator.tier3.result_display import _default_with_skill_overall, _render_agent_scores, render_result


def test_terminal_keeps_not_applicable_distinct_from_missing_evidence():
    data = {
        "execution_status": "succeeded",
        "with_skill": {"accuracy": None},
        "not_applicable_metrics": {"with_skill": ["accuracy"]},
    }
    console = Console(record=True, width=140)
    _render_agent_scores(
        console=console, result={}, agent="test", data=data, metrics=["accuracy", "security"], safe=str
    )
    output = console.export_text()
    assert "N/A" in output
    assert "NO SCORE" in output


def test_terminal_overall_renormalizes_only_explicit_na_metrics():
    values = {"security": 1.0}
    na = ["accuracy", "goal_accuracy", "behavior_check", "skill_execution", "skill_efficiency"]
    assert (
        _default_with_skill_overall(
            {"with_skill": values, "execution_status": "succeeded", "not_applicable_metrics": {"with_skill": na}}
        )
        == 1.0
    )
    assert _default_with_skill_overall({"with_skill": values, "execution_status": "succeeded"}) is None


def _rows(output: str) -> dict[str, list[str]]:
    """Map each table row label to its cells, dropping borders and score bars."""
    rows: dict[str, list[str]] = {}
    for line in output.splitlines():
        cells = re.sub(r"[─-╿█░]", " ", line).split("  ")
        cells = [cell.strip() for cell in cells if cell.strip()]
        if cells:
            rows.setdefault(cells[0], cells[1:])
    return rows


def test_terminal_lift_reads_na_when_either_arm_is_na():
    with_scores = {"security": 1.0, "skill_execution": 0.8, "skill_efficiency": 0.6, "accuracy": 0.9}
    baseline_scores = {"security": 1.0, "accuracy": 0.5}
    with_na = ["behavior_check"]
    baseline_na = ["skill_execution", "skill_efficiency", "behavior_check"]
    result = {
        "execution_status": "succeeded",
        "metrics": list(DEFAULT_METRICS),
        "agents": {
            "codex": {
                "execution_status": "succeeded",
                "conditions": {
                    "with_skill": {"execution_status": "succeeded"},
                    "without_skill": {"execution_status": "succeeded"},
                },
                "with_skill": with_scores,
                "without_skill": baseline_scores,
                "not_applicable_metrics": {"with_skill": with_na, "without_skill": baseline_na},
                "lift": {"security": {"delta": 0.0}, "accuracy": {"delta": 0.4}},
                "dimensions_with_skill": dimension_scores(with_scores, with_na),
                "dimensions_without_skill": dimension_scores(baseline_scores, baseline_na),
            }
        },
    }

    rows = _rows(render_result(result))

    # N/A in one arm or both: no lift, and N/A says why instead of NO SCORE.
    assert rows["Skill Execution"] == ["0.80", "N/A", "N/A"]
    assert rows["Behavior Check"] == ["N/A", "N/A", "N/A"]
    assert rows["Discoverability"] == ["0.80", "N/A", "N/A"]
    # goal_accuracy is missing, not N/A: its lift still reads NO SCORE.
    assert rows["Goal Accuracy"] == ["NO SCORE", "NO SCORE", "NO SCORE"]
    assert rows["Effectiveness"] == ["NO SCORE", "NO SCORE", "NO SCORE"]
    # Scored rows keep their numeric lift.
    assert rows["Accuracy"] == ["0.90", "0.50", "+0.40"]
    assert rows["Correctness"] == ["0.90", "0.50", "+0.40"]


def test_terminal_lift_reads_skipped_in_a_baseline_only_run_unless_the_row_is_na():
    baseline_scores = {"security": 1.0, "accuracy": 0.5, "goal_accuracy": 0.5, "behavior_check": 0.5}
    baseline_na = ["skill_execution", "skill_efficiency"]
    result = {
        "execution_status": "succeeded",
        "metrics": list(DEFAULT_METRICS),
        "agents": {
            "codex": {
                "execution_status": "succeeded",
                "conditions": {
                    "with_skill": {"execution_status": "skipped"},
                    "without_skill": {"execution_status": "succeeded"},
                },
                "with_skill": {},
                "without_skill": baseline_scores,
                "not_applicable_metrics": {"with_skill": [], "without_skill": baseline_na},
                "lift": {},
                "dimensions_with_skill": {},
                "dimensions_without_skill": dimension_scores(baseline_scores, baseline_na),
            }
        },
    }

    rows = _rows(render_result(result))

    # --skip-with-skill: the arm never ran, so its cells and the lift read skipped, as in the other reports.
    assert rows["Accuracy"] == ["skipped", "0.50", "skipped"]
    assert rows["Correctness"] == ["skipped", "0.50", "skipped"]
    # N/A wins over skipped: the baseline has no skill to activate.
    assert rows["Skill Execution"] == ["skipped", "N/A", "N/A"]
    assert rows["Discoverability"] == ["skipped", "N/A", "N/A"]
