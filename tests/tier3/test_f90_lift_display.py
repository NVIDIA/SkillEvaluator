# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof L18 (PR #28 half): one lift, shown once, and never as final when it is not.

- ``evaluate-plugin`` printed "OVERALL SKILL LIFT +0.42" (lift.json's metric
  mean) above "Effectiveness lift +0.40" (the reports' basis).
- A partial (INCOMPLETE) plugin run still printed "OVERALL SKILL LIFT +0.31".
- A 1-case interval printed as a final-looking "[+0.27, +0.27]".
- An all-null interval printed as "n/a n/a (95% CI)".
- BENCHMARK showed two different "Effectiveness" numbers.
- A baseline that ran and failed was described as "baseline not run".
"""

from __future__ import annotations

import re
from pathlib import Path

from _f90_lift_fixtures import benchmark, cli, collect, markdown, metrics, payload, provenance, run_config

from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.result_display import render_result

CASES = [f"case-{index}" for index in range(1, 7)]

# The no-plugin arm still has skill metrics here (an alternate skill was staged), so
# lift.json's six-metric mean (+0.37) differs from the reports' five-dimension basis (+0.40).
PLUGIN = metrics(1.0, accuracy=0.9, goal_accuracy=0.5, behavior_check=0.5)
NO_PLUGIN = metrics(1.0, skill_execution=0.5, skill_efficiency=0.4, accuracy=0.2, goal_accuracy=0.3, behavior_check=0.3)


def _engine_result(tmp_path: Path, *, partial: bool = False) -> dict:
    arms = {"with": {case: [PLUGIN] for case in CASES}, "without": {case: [NO_PLUGIN] for case in CASES}}
    collected = collect(tmp_path, {"codex": arms}, n_attempts=1)
    return {
        **collected,
        "skill_name": "demo",
        "run_config": run_config("effectiveness"),
        "plugin_provenance": provenance("effectiveness", partial=partial),
    }


def test_evaluate_plugin_shows_the_reports_lift_only(tmp_path: Path) -> None:
    result = _engine_result(tmp_path)
    assert result["agents"]["codex"]["lift"]["overall"]["delta"] == 0.3667  # lift.json's metric mean

    output = render_result(result)

    assert re.search(r"OVERALL SKILL LIFT\s+\+0\.40", output)
    assert "+0.37" not in output
    assert payload(tmp_path)["overall_lift"] == 0.4


def test_partial_plugin_run_prints_no_final_looking_headline(tmp_path: Path) -> None:
    output = render_result(_engine_result(tmp_path, partial=True))

    assert not re.search(r"OVERALL SKILL LIFT\s+\+", output)
    assert "OVERALL SKILL LIFT not shown: partial plugin run (INCOMPLETE), not final" in output


def test_one_case_interval_is_insufficient_not_final(tmp_path: Path) -> None:
    single = stats.paired_case_bootstrap({"only": 0.6}, {"only": 0.33})
    assert (single["estimate"], single["ci_low"], single["ci_high"]) == (0.27, None, None)
    assert single["precision"] == "insufficient"

    collect(tmp_path, {"codex": {"with": {"only": [PLUGIN]}, "without": {"only": [NO_PLUGIN]}}}, n_attempts=1)
    built = payload(tmp_path)

    assert "[+0.40, +0.40]" not in markdown(built)
    assert "(no interval: 1 paired case; too few to resample)" in markdown(built)
    assert "n/a (too few cases)" in cli(built)
    assert "No interval: 1 paired case(s), too few; precision insufficient" in benchmark(built)


def test_unmeasured_interval_never_prints_n_a_n_a(tmp_path: Path) -> None:
    collect(
        tmp_path,
        {"codex": {"with": {c: [PLUGIN] for c in CASES}, "without": {c: [NO_PLUGIN] for c in CASES}}},
        n_attempts=1,
    )
    built = payload(tmp_path)
    empty = stats.paired_case_bootstrap({}, {})
    built["lift_uncertainty"] = {"effectiveness": empty, "integration": None}
    built["agents"]["codex"]["lift_uncertainty"] = {"effectiveness": empty, "integration": None}

    report = markdown(built)

    assert "n/a n/a" not in report
    assert "- Effectiveness lift: not measured" in report


def test_benchmark_shows_one_effectiveness_and_says_when_the_baseline_failed(tmp_path: Path) -> None:
    arms = {
        "with": {case: [PLUGIN] for case in CASES},
        "without": {case: [NO_PLUGIN] for case in CASES},
    }
    arms["without"]["case-3"] = [None]  # the no-plugin trial ran and timed out
    collect(tmp_path, {"codex": arms}, n_attempts=1)

    card = benchmark(payload(tmp_path))

    glance, lift_section = card.split("## Results at a Glance", 1)[1].split("\n## ", 2)[:2]
    results = glance + lift_section
    # Never two "Effectiveness" numbers. The run is INCOMPLETE (one baseline trial timed out), so the gate
    # package's rule hides its dimension rows: no Effectiveness score row at all, and the lift row is partial.
    assert len(re.findall(r"^\| Effectiveness", results, flags=re.MULTILINE)) == 0
    assert "| Plugin lift (plugin vs. no plugin) | Partial run, not final:" in results
    assert "baseline not run" not in results
    # It still says the baseline arm ran and failed, not that it was not run.
    assert "baseline 5 of 6 attempts scored" in results
    assert "did not complete: Baseline (no plugin)" in results


def test_partial_plugin_run_marks_the_skill_lift_row_not_final(tmp_path: Path) -> None:
    # Proof check-13 cc-incomplete: the headline was hidden, but the table still showed
    # "Skill Lift 0.83 ... 0.47 ... +0.37" a few lines below it.
    output = render_result(_engine_result(tmp_path, partial=True))

    lift_row = next(line for line in output.splitlines() if "Skill Lift" in line and "not shown" not in line)
    assert "(partial run, not final)" in lift_row
    assert "partial" in lift_row.split("(partial run, not final)", 1)[1]
    assert not re.search(r"[+-]\d\.\d\d\s*$", lift_row.rstrip(" │"))

    complete = render_result(_engine_result(tmp_path / "complete"))
    complete_row = next(line for line in complete.splitlines() if "Skill Lift" in line)
    assert "+0.40" in complete_row
    assert "not final" not in complete_row
