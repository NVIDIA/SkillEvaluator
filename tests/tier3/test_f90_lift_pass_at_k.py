# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H5 follow-up: the pass@k lift does not reward skill activation alone.

Proof example check-13 edge-04 (activation only): both arms got the same judge
verdicts. Each arm's pass flag used its own overall score, and only the plugin
arm's overall has ``skill_execution`` and ``skill_efficiency``. So the plugin
"passed" cases the no-plugin arm failed on identical answers:
``pass_at_k_lift.json`` said +33%, the HTML said "skill-only passes 3", and the
CLI Reliability table put 78% next to 44% as if they were comparable.

The cross-arm pass comparison now scores both arms on the metrics both scored.
Each arm's own pass@k is unchanged, and the Reliability table says when the
arms' rates use different metrics.
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path

from _f90_lift_fixtures import cli, collect, html, metrics, payload

CASES = [f"case-{index}" for index in range(1, 7)]


def _activation_only(tmp_path: Path) -> dict:
    # Judged 0.4 in both arms. The plugin arm's own overall is (4 x 0.4 + 1.0 + 0.75) / 6 = 0.56, a pass
    # at the 0.5 threshold; the no-plugin arm's is 0.4, a fail. On the shared metrics both are 0.4.
    plugin = metrics(0.4, skill_execution=1.0, skill_efficiency=0.75)
    no_plugin = metrics(0.4, skill=False)
    arms = {
        "with": {case: [plugin, plugin] for case in CASES},
        "without": {case: [no_plugin, no_plugin] for case in CASES},
    }
    return collect(tmp_path, {"claude-code": arms}, n_attempts=2)


def test_pass_at_k_lift_compares_both_arms_on_the_metrics_both_scored(tmp_path: Path) -> None:
    collected = _activation_only(tmp_path)
    agent = collected["agents"]["claude-code"]

    # Each arm's own pass@k keeps its meaning.
    assert agent["pass_at_k"]["with_skill"]["rate"] == 1.0
    assert agent["pass_at_k"]["without_skill"]["rate"] == 0.0

    on_disk = json.loads((tmp_path / "results" / "claude-code" / "pass_at_k_lift.json").read_text())
    for pass_lift in (agent["pass_at_k"]["lift"], on_disk):
        assert pass_lift["basis"] == "shared_metrics"
        assert pass_lift["excluded_metrics"] == ["skill_execution", "skill_efficiency"]
        assert pass_lift["with_skill"] == pass_lift["without_skill"] == 0.0
        assert pass_lift["delta"] == 0.0
        assert pass_lift["count_derived_delta"] == 0.0
        assert pass_lift["passed_cases_delta"] == 0
        paired = pass_lift["paired_comparison"]
        assert paired["with_skill_only_pass"] == 0
        assert paired["neither_pass"] == len(CASES)
        assert paired["paired_rate_delta"] == 0.0


def test_pass_at_k_lift_still_measures_a_real_difference(tmp_path: Path) -> None:
    plugin = metrics(0.7, skill_execution=1.0, skill_efficiency=1.0)
    no_plugin = metrics(0.3, skill=False)
    arms = {
        "with": {case: [plugin] for case in CASES},
        "without": {case: [no_plugin] for case in CASES},
    }
    pass_lift = collect(tmp_path, {"codex": arms}, n_attempts=1)["agents"]["codex"]["pass_at_k"]["lift"]

    assert pass_lift["delta"] == 1.0
    assert pass_lift["paired_comparison"]["with_skill_only_pass"] == len(CASES)


def test_pass_at_k_lift_is_unchanged_when_both_arms_score_the_same_metrics(tmp_path: Path) -> None:
    # A sum-of-parts style baseline that stages the skill scores every metric: nothing is left out.
    arms = {
        "with": {case: [metrics(0.6)] for case in CASES},
        "without": {case: [metrics(0.45)] for case in CASES},
    }
    agent = collect(tmp_path, {"codex": arms}, n_attempts=1)["agents"]["codex"]
    pass_lift = agent["pass_at_k"]["lift"]

    assert pass_lift["excluded_metrics"] == []
    assert pass_lift["with_skill"] == agent["pass_at_k"]["with_skill"]["rate"] == 1.0
    assert pass_lift["without_skill"] == agent["pass_at_k"]["without_skill"]["rate"] == 0.0
    assert pass_lift["delta"] == 1.0


def test_reports_say_the_arm_pass_rates_use_different_metrics(tmp_path: Path) -> None:
    _activation_only(tmp_path)
    built = payload(tmp_path)

    page = html_lib.unescape(html(built))
    pass_table = " ".join(page.split('id="tier3-pass-at-k"', 1)[1].split("</table>", 1)[0].split())
    assert "+100%" not in pass_table
    assert "+0%" in pass_table
    assert "skill-only passes 0" in pass_table
    assert "leaving out skill_execution, skill_efficiency: with skill 0%, without skill 0%" in pass_table

    note = "pass@k and pass^k use each arm's own metrics, so they are not a like-for-like comparison"
    assert note in page
    assert "Baseline (no plugin) has no skill_execution or skill_efficiency score" in page
    assert note in " ".join(cli(built).split())
