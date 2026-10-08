# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M13: in ``--lift-mode integration`` the members' lift is Integration, never Skill Lift.

The legacy two-arm mode stages the plugin's member skills in its only
baseline, so that comparison is plugin vs. member skills. Before the fix the
statistics filed its interval as "effectiveness" (live run L1:
``lift_uncertainty.effectiveness 0.2363`` while BENCHMARK said "Not
measured"), and the HTML card, the payload's ``overall_lift`` and the
Markdown/CLI "composite lift" showed it as Skill Lift (+0.21, proof example
check-14 pos-06 and the check-13 integration replay).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from _f90_lift_fixtures import MEMBERS, benchmark, cli, collect, html, metrics, payload

from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context

CASES = [f"case-{index}" for index in range(1, 7)]


def _legacy_run(tmp_path: Path) -> dict:
    arms = {
        "with": {case: [metrics(0.85)] for case in CASES},
        # The only baseline stages the member skills, so it keeps its skill metrics.
        "without": {case: [metrics(0.64)] for case in CASES},
    }
    context = build_plugin_signals_context(member_skills=MEMBERS, baseline_has_members=True)
    return collect(tmp_path, {"codex": arms}, n_attempts=1, plugin_signals=context)


def test_legacy_members_interval_is_filed_under_integration(tmp_path: Path) -> None:
    collected = _legacy_run(tmp_path)
    uncertainty = collected["agents"]["codex"]["lift_uncertainty"]

    assert uncertainty["effectiveness"] is None
    assert uncertainty["integration"]["estimate"] == 0.21
    assert uncertainty["integration"]["control_arm"] == "without_skill"
    persisted = json.loads((tmp_path / "results" / "codex" / "statistics.json").read_text(encoding="utf-8"))
    assert persisted["lift_uncertainty"]["effectiveness"] is None


def test_legacy_members_lift_is_never_reported_as_skill_lift(tmp_path: Path) -> None:
    _legacy_run(tmp_path)
    built = payload(tmp_path, "integration")

    assert built["overall_lift"] is None
    assert built["composite_lift"] is None
    agent = built["agents"]["codex"]
    assert agent["lift"] is None
    assert agent["integration_lift"] == 0.21
    assert agent["sum_of_parts"] == 0.64
    integration = built["integration"]
    assert integration["measured"] is True
    assert integration["integration_lift"] == 0.21
    assert integration["lift_uncertainty"]["estimate"] == 0.21
    assert "Skill Lift not distinguishable from zero" not in [c["title"] for c in built["conclusions"]]

    page = html(built)
    card = re.search(r'<h3 class="dashboard-card-title">Effectiveness Lift</h3>\s*<p[^>]*>([^<]*)</p>', page)
    assert card is not None and card.group(1) == "Not measured"
    assert "+0.21" not in re.sub(r"\s+", " ", page).split("Effectiveness Lift", 1)[1][:200]
    assert "Lift vs. Member Skills" in page
    assert "Dimension Scores and Lift vs. Member Skills" in page

    output = cli(built)
    assert "Evaluator Scores (plugin vs. member skills)" in output
    assert "Evaluator Scores (Skill Lift)" not in output

    card = benchmark(built)
    assert "| Overall | 64% → 85% (+21 points) |" in card
    assert "| Plugin lift (plugin vs. no plugin) | Not measured" in card
    assert "Real integration, +21 points" in card


def test_older_legacy_runs_are_refiled_under_integration(tmp_path: Path) -> None:
    _legacy_run(tmp_path)
    statistics_path = tmp_path / "results" / "codex" / "statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    # A run saved before the fix: the members interval sits under "effectiveness" and spans zero.
    older = dict(statistics["lift_uncertainty"]["integration"], ci_low=-0.05, ci_includes_zero=True)
    statistics["lift_uncertainty"] = {"effectiveness": older, "integration": None}
    statistics_path.write_text(json.dumps(statistics), encoding="utf-8")

    built = payload(tmp_path, "integration")

    assert built["lift_uncertainty"]["effectiveness"] is None
    assert built["lift_uncertainty"]["integration"]["ci_low"] == -0.05
    assert built["overall_lift"] is None
    assert "Skill Lift not distinguishable from zero" not in [c["title"] for c in built["conclusions"]]
    assert built["integration"]["verdict"] == "inconclusive"
