# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H5 and L18 follow-ups: every screen shows the lift with its own two arm scores.

Once the no-skill arm records Discoverability and Efficiency as not applicable,
the lift compares only the dimensions both arms scored. The full with-skill
score still has all five, so "full score minus lift" is not the baseline, a
dimension the baseline cannot score has no "missing baseline", and an
INCOMPLETE plugin run or a failed baseline never reads as a final lift.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from _f90_lift_fixtures import collect, html, metrics, payload, result_for

CASES = [f"case-{index}" for index in range(1, 7)]


def _skill_arms(with_value: float = 0.6, without_value: float = 0.4) -> dict[str, dict]:
    """The plugin arm activates its skill (Discoverability and Efficiency 1.0); the no-plugin arm cannot."""
    plugin = metrics(with_value, skill_execution=1.0, skill_efficiency=1.0)
    no_plugin = metrics(without_value, skill=False)
    return {
        "with": {case: [plugin, plugin] for case in CASES},
        "without": {case: [no_plugin, no_plugin] for case in CASES},
    }


def _row_text(row: Any) -> str:
    return "".join(text for text, _style in row.segments)


def test_validate_panel_shows_the_baseline_on_the_lift_basis(tmp_path: Path) -> None:
    # Proof check-13 edge-04 and check-14 edge-04: the panel printed "with-skill 0.65 baseline 0.64"
    # (full score minus lift) while the real baseline on the lift basis was 0.52.
    from skillevaluator.reporting.console_ui import summarize_tier3

    collect(tmp_path, {"claude-code": _skill_arms()}, n_attempts=2)
    built = payload(tmp_path)
    agent = built["agents"]["claude-code"]
    assert agent["with_skill"] == 0.76  # all five dimensions
    assert agent["lift_basis"]["effectiveness"]["baseline"] == 0.4

    result = result_for(built)
    result.passed = True
    ran, _ok, rows, _reason = summarize_tier3(result)

    assert ran
    lift_row = next(row for row in rows if row.label == "lift")
    text = _row_text(lift_row)
    assert "+0.20" in text
    assert "with-skill 0.60" in text
    assert "baseline 0.40" in text
    assert "baseline 0.56" not in text


def test_validate_panel_keeps_older_payloads_without_a_lift_basis() -> None:
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting.console_ui import summarize_tier3

    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Run live agent evaluation")
    result.passed = True
    result.metadata["agent_eval"] = {
        "summary": {"overall_score": 0.7, "overall_lift": 0.2, "best_agent": "codex", "agents_run": ["codex"]},
        "agents": {"codex": {"with_skill": 0.7, "baseline": 0.5, "lift": 0.2}},
    }
    _ran, _ok, rows, _reason = summarize_tier3(result)

    text = _row_text(next(row for row in rows if row.label == "lift"))
    assert "with-skill 0.70" in text and "baseline 0.50" in text


def _partial_plugin_run(payload: dict[str, Any]) -> Any:
    """A Tier 3 result stamped INCOMPLETE the way ``validate`` stamps a partial plugin run."""
    from skillevaluator.models import ValidationResult

    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Run live agent evaluation")
    result.metadata["agent_eval"] = payload
    result.passed = False
    result.metadata["execution_status"] = "skipped"
    result.metadata["skip_reason"] = "INCOMPLETE: Tier 3 plugin evaluation did not complete"
    return result


def test_validate_panel_shows_a_partial_plugin_run_that_scored_trials_as_run() -> None:
    # H-13: the INCOMPLETE stamp made the panel say "skipped" for a run that scored 22 of 24 attempts.
    from skillevaluator.reporting.console_ui import summarize_tier3

    summary = {"agents_run": ["codex"], "expected_attempts": 24, "scored_attempts": 22, "execution_status": "failed"}
    result = _partial_plugin_run(
        {"execution_status": "failed", "execution_errors": ["codex: 2 trials timed out"], "summary": summary}
    )

    ran, ok, rows, reason = summarize_tier3(result)

    assert ran
    assert not ok
    assert reason == ""
    assert _row_text(rows[0]) == "codex"
    incomplete = next(row for row in rows if row.label == "incomplete")
    assert _row_text(incomplete) == "INCOMPLETE: Tier 3 plugin evaluation did not complete"


def test_validate_panel_still_skips_a_plugin_run_that_evaluated_nothing() -> None:
    from skillevaluator.evaluation.tier3_report import advisory_skip_result
    from skillevaluator.reporting.console_ui import summarize_tier3

    skipped = advisory_skip_result("Tier 3 plugin evaluation is INCOMPLETE: nothing was evaluated.", skill_name="demo")
    result = _partial_plugin_run(skipped.metadata["agent_eval"])

    ran, _ok, rows, reason = summarize_tier3(result)

    assert not ran
    assert rows == []
    assert reason == "INCOMPLETE: Tier 3 plugin evaluation did not complete"


def test_dimension_reasoning_says_not_applicable_when_the_baseline_ran(tmp_path: Path) -> None:
    # Proof check-13 cc-evalplugin-B: "Efficiency is lowest at 0.70 ... No baseline run available;
    # lift cannot be computed." The baseline did run; Efficiency just does not apply to it.
    plugin = metrics(0.9, skill_execution=1.0, skill_efficiency=0.7)
    no_plugin = metrics(0.6, skill=False)
    arms = {"with": {case: [plugin] for case in CASES}, "without": {case: [no_plugin] for case in CASES}}
    collect(tmp_path, {"claude-code": arms}, n_attempts=1)
    built = payload(tmp_path)

    dimensions = {row["id"]: row for row in built["agents"]["claude-code"]["dimensions"]}
    for dim_id in ("discoverability", "efficiency"):
        row = dimensions[dim_id]
        assert row["baseline_not_applicable"] is True
        assert "Not applicable without the skill or plugin, so there is no lift." in row["reasoning_bullets"]
        assert "No baseline run available" not in row["explanation"]
    assert "baseline_not_applicable" not in dimensions["security"]
    assert "+0.30 lift over baseline 0.60." in dimensions["security"]["explanation"]

    weakest = next(item for item in built["conclusions"] if item["title"] == "Weakest dimension")
    assert weakest["message"].startswith("Efficiency is lowest at 0.70.")
    assert "Not applicable without the skill or plugin" in weakest["message"]
    assert "No baseline run available" not in html(built)


def test_dimension_reasoning_still_says_no_baseline_when_none_ran(tmp_path: Path) -> None:
    plugin = metrics(0.9, skill_execution=1.0, skill_efficiency=0.7)
    collect(tmp_path, {"claude-code": {"with": {case: [plugin] for case in CASES}}}, n_attempts=1)
    built = payload(tmp_path)

    efficiency = next(row for row in built["agents"]["claude-code"]["dimensions"] if row["id"] == "efficiency")
    assert "No baseline run available; lift cannot be computed." in efficiency["reasoning_bullets"]
    assert "baseline_not_applicable" not in efficiency


def _kpi_and_scorecard(page: str) -> tuple[str, str]:
    import html as html_lib

    text = " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", page)).split())
    kpi = text.split("Overall Score", 2)[-1].split("Trials", 1)[0]
    scorecard = text.split("Agent Scorecards", 1)[1].split("trials", 1)[0]
    return kpi, scorecard


def test_html_says_the_baseline_did_not_complete_not_no_baseline(tmp_path: Path) -> None:
    # Proof check-14 neg-03 (failed baseline arm): the HTML said "Skill Lift N/A no baseline" and
    # "claude-code 0.90 no baseline" while BENCHMARK said the baseline did not complete.
    arms = _skill_arms(0.9, 0.5)
    arms["without"] = {case: [None, None] for case in CASES}
    collect(tmp_path, {"claude-code": arms}, n_attempts=2)
    built = payload(tmp_path)
    agent = built["agents"]["claude-code"]
    assert agent["lift"] is None and agent["lift_note"] == "baseline did not complete"

    kpi, scorecard = _kpi_and_scorecard(html(built))
    assert "baseline did not complete" in kpi
    assert "Skill Lift N/A baseline did not complete" in kpi
    assert "no baseline" not in kpi
    assert "claude-code 0.94 baseline did not complete" in scorecard


def test_html_marks_a_partial_comparison_as_not_final(tmp_path: Path) -> None:
    arms = _skill_arms(0.9, 0.5)
    arms["without"]["case-3"] = [None, None]  # one no-plugin case timed out in both attempts
    collect(tmp_path, {"claude-code": arms}, n_attempts=2)
    built = payload(tmp_path)
    agent = built["agents"]["claude-code"]
    assert agent["lift"] is None
    assert agent["lift_note"] == "baseline did not complete; partial: 5 of 6 cases, not final"

    kpi, scorecard = _kpi_and_scorecard(html(built))
    assert "Skill Lift N/A baseline did not complete; partial: 5 of 6 cases, not final" in kpi
    assert "baseline did not complete; partial: 5 of 6 cases, not final" in scorecard


def test_html_still_says_no_baseline_with_skip_baseline(tmp_path: Path) -> None:
    collect(tmp_path, {"claude-code": {"with": _skill_arms()["with"]}}, n_attempts=2)
    built = payload(tmp_path)

    assert built["agents"]["claude-code"]["lift_note"] == "no baseline"
    kpi, scorecard = _kpi_and_scorecard(html(built))
    assert "Skill Lift N/A no baseline" in kpi
    assert "no baseline" in scorecard
