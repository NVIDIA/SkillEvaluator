# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from rich.console import Console

from skillevaluator.tier3.result_display import _default_with_skill_overall, _render_agent_scores


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
