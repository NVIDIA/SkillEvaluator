# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H5 and M10: one lift, on one basis, everywhere.

H5 (proof example check-13 edge-04): both arms got identical judge verdicts
and the plugin arm only activated its skill, yet the lift was +0.16 because the
no-plugin arm was scored on Discoverability and Efficiency. With those N/A in
that arm the lift compares only the dimensions both arms share, so it is about
zero, and BENCHMARK no longer shows "27% -> 27%" next to "+16 points".

M10 (proof examples check-14 edge-04 and check-20 edge-08): the headline lift
was a trial-weighted difference of dimension means while its interval was a
case-weighted paired bootstrap, so the headline could fall outside its own
interval and the Integration verdict could contradict it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _f90_lift_fixtures import NA, benchmark, collect, html, metrics, payload

from skillevaluator.tier3.harbor import stats

CASES = [f"case-{index}" for index in range(1, 7)]


def _activation_only(case_index: int) -> tuple[dict, dict]:
    """Same judge verdicts in both arms; only the plugin arm can activate a skill."""
    judged = 0.2 + 0.1 * (case_index % 3)
    plugin = metrics(judged, security=1.0, skill_execution=1.0, skill_efficiency=0.75)
    no_plugin = metrics(judged, skill=False, security=1.0)
    return plugin, no_plugin


def test_activation_alone_earns_no_effectiveness_lift(tmp_path: Path) -> None:
    arms = {"with": {}, "without": {}}
    for index, case in enumerate(CASES):
        plugin, no_plugin = _activation_only(index)
        arms["with"][case] = [plugin, plugin]
        arms["without"][case] = [no_plugin, no_plugin]
    collected = collect(tmp_path, {"claude-code": arms}, n_attempts=2)
    agent = collected["agents"]["claude-code"]

    # The no-plugin arm is complete: its N/A skill metrics are not a failure.
    assert agent["conditions"]["without_skill"]["execution_status"] == "succeeded"
    assert set(agent["not_applicable_metrics"]["without_skill"]) == {"skill_execution", "skill_efficiency"}
    effectiveness = agent["lift_uncertainty"]["effectiveness"]
    assert effectiveness["dimensions"] == ["security", "correctness", "effectiveness"]
    assert effectiveness["estimate"] == 0.0
    assert effectiveness["treatment_score"] == effectiveness["control_score"]

    built = payload(tmp_path)
    best = built["agents"]["claude-code"]
    assert built["overall_lift"] == 0.0
    assert best["lift"] == 0.0
    assert best["baseline"] == effectiveness["control_score"]
    # The verdict still uses every with-plugin dimension.
    assert best["with_skill"] > best["lift_basis"]["effectiveness"]["with_skill"]
    discoverability = next(row for row in best["dimensions"] if row["id"] == "discoverability")
    assert discoverability["baseline"] is None and discoverability["baseline_not_applicable"] is True

    card = benchmark(built)
    assert "| Effectiveness | 30% → 30% (±0 points) |" in card
    assert "| Discoverability | 100% — not applicable without the plugin |" in card
    assert "| Overall | 67%; lift basis 53% → 53% (±0 points) |" in card
    lift_row = next(line for line in card.splitlines() if "Plugin lift (plugin vs. no plugin)" in line)
    assert "| ±0 points |" in lift_row

    page = html(built)
    assert "lift basis 0.53 vs baseline 0.53" in page
    assert 'title="Not applicable: the run without it has no skill to discover or route to"' in page


def test_effectiveness_headline_is_the_interval_estimate_with_unequal_attempts(tmp_path: Path) -> None:
    # Proof check-14 edge-04 shape: --stop-on-pass leaves 1 attempt where the plugin passed
    # at once and 3 where it never passed; the no-plugin arm never passes.
    with_arm = {case: [metrics(0.9)] for case in CASES[1:]}
    with_arm[CASES[0]] = [metrics(0.1), metrics(0.1), metrics(0.1)]
    without_arm = {case: [metrics(0.3)] * 3 for case in CASES}
    collect(tmp_path, {"codex": {"with": with_arm, "without": without_arm}}, n_attempts=3, stop_on_pass=True)

    built = payload(tmp_path)
    interval = built["agents"]["codex"]["lift_uncertainty"]["effectiveness"]

    # Case-weighted: (0.1 + 5 x 0.9) / 6 - 0.3. The trial-weighted headline was +0.30.
    assert interval["estimate"] == pytest.approx(0.4667, abs=1e-4)
    assert built["overall_lift"] == interval["estimate"]
    assert interval["ci_low"] <= built["overall_lift"] <= interval["ci_high"]
    # BENCHMARK's Overall row shows the full score, then the uplift on the lift's own basis.
    card = benchmark(built)
    assert "| Overall | 60%; lift basis 30% → 77% (+47 points) |" in card
    assert "each case counts once" in card


def test_integration_headline_and_verdict_agree_with_the_interval(tmp_path: Path) -> None:
    # Proof check-20 edge-08: 20 cases; accuracy has a reference in one case only, where the
    # plugin scores 0 and the parts 1. In the other 19 cases the plugin is +0.05 better.
    cases = [f"flip-{index:03d}" for index in range(1, 21)]
    arms: dict[str, dict] = {"with": {}, "without": {}, "sumofparts": {}}
    for index, case in enumerate(cases):
        if index == 0:
            arms["with"][case] = [metrics(0.6, accuracy=0.0)] * 2
            arms["sumofparts"][case] = [metrics(0.6, accuracy=1.0)] * 2
            arms["without"][case] = [metrics(0.3, accuracy=0.0)] * 2
        else:
            arms["with"][case] = [metrics(0.6, accuracy=NA)] * 2
            arms["sumofparts"][case] = [metrics(0.55, accuracy=NA)] * 2
            arms["without"][case] = [metrics(0.3, accuracy=NA)] * 2
    collect(tmp_path, {"claude-code": arms}, n_attempts=2)

    integration = payload(tmp_path, "both")["integration"]

    interval = integration["lift_uncertainty"]
    assert integration["integration_lift"] == interval["estimate"]
    assert interval["ci_low"] <= integration["integration_lift"] <= interval["ci_high"]
    assert integration["integration_lift"] > 0
    assert integration["with_plugin"] - integration["sum_of_parts"] == pytest.approx(
        integration["integration_lift"], abs=2e-4
    )
    assert integration["verdict"] != "negative_integration"
    assert integration["point_verdict"] != "negative_integration"


def test_shared_case_scores_compare_each_case_on_its_common_dimensions() -> None:
    def trial(case: str, scores: dict) -> stats.TrialObservation:
        return stats.TrialObservation(case_id=case, reward={"metric_set": "skill-evaluator-default-v2", **scores})

    plugin = [trial("a", metrics(0.8)), trial("b", metrics(0.6))]
    no_plugin = [trial("a", metrics(0.5, skill=False)), trial("b", metrics(0.6, skill=False, skill_execution=0.0))]

    treatment, control, dimensions = stats.shared_case_scores(plugin, no_plugin)

    assert dimensions == ["security", "correctness", "discoverability", "effectiveness"]
    assert treatment == pytest.approx({"a": 0.8, "b": 0.6})
    # Case b's baseline has Discoverability (an alternate was staged): 0.0 there, so (3 x 0.6 + 0) / 4.
    assert control == pytest.approx({"a": 0.5, "b": 0.45})
