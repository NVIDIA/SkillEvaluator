# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proofs M15, H1, L5 and L32 (verifier follow-up): a partial run that failed is FAIL in every report, a gated Tier 3
run that never ran is FAIL in the HTML report too, dimension scores use one threshold set in every view, and an
INCOMPLETE run prints no Tier 3 scores as results."""

from __future__ import annotations

import io
import re

from _f90_gate_payloads import (
    EDGE05_INTERVAL,
    EDGE05_WITH,
    EDGE05_WITHOUT,
    build_payload,
    scores,
    section,
    terminal,
    tier3,
)
from rich.console import Console

from skillevaluator.evaluation.tier3_report import (
    _validation_result_from_payload,
    advisory_skip_result,
    build_agent_eval_payload,
)
from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting.benchmark import BenchmarkReporter
from skillevaluator.tier3.result_display import render_evaluation_result

PARTIAL = {"partial": True, "unresolved_skill_refs": ["github::example-org/example-repo::skills::missing"]}

# ---------------------------------------------------------------------------
# M15: a confirmed regression fails a partial plugin run, as the legend says
# ---------------------------------------------------------------------------


def test_partial_run_with_a_confirmed_regression_has_a_fail_card() -> None:
    payload = build_payload(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL, plugin_provenance=PARTIAL)
    assert payload["lift_band"]["regression_confirmed"] is True

    card = BenchmarkReporter(content_type="plugin", skill_name="gate-demo").render_all([tier3(payload, blocking=False)])
    head = card.split("## Evaluation Metadata")[0]

    assert "Overall verdict: FAIL" in head
    assert "INCOMPLETE — Required evidence is missing" not in head
    assert "Skill Lift regression" in head
    # The legend's rule and the card agree.
    assert "makes this card FAIL" in card
    tier3_row = next(line for line in card.splitlines() if line.startswith("| Tier 3 |"))
    assert "**FAIL**" in tier3_row
    assert "partial plugin run" in tier3_row


def test_partial_run_with_a_confirmed_regression_is_a_failure_in_the_terminal() -> None:
    payload = build_payload(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL, plugin_provenance=PARTIAL)

    output = terminal([tier3(payload, blocking=True)])

    assert re.search(r"AGENT_EVAL\s*│\s*FAIL\s*│\s*Tier 3 Skill Lift regression", output)
    assert "[FAIL] Validation failed" in output
    assert "[INCOMPLETE] Validation evidence is incomplete" not in output


def _html_tier3_status(result: object) -> tuple[str, str, str]:
    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all([result])
    card = re.search(r'<span class="tier-card-verdict">([^<]+)</span>', html)
    verdict = re.search(
        r'<h3 class="dashboard-card-title">Verdict</h3>\s*<p class="dashboard-card-value">([^<]+)</p>', html
    )
    lead = re.search(r'<div class="t3-overview-lead">(.*?)</div>', html, re.DOTALL)
    assert card is not None and verdict is not None and lead is not None
    return card.group(1), verdict.group(1), " ".join(re.sub(r"<[^>]+>", "", lead.group(1)).split())


def test_html_partial_run_that_failed_is_fail() -> None:
    payload = build_payload(scores(security=0.0), plugin_provenance=PARTIAL)

    card, verdict, lead = _html_tier3_status(tier3(payload, blocking=False))

    assert (card, verdict) == ("FAIL", "FAIL")
    assert "INCOMPLETE (partial)" in lead


# ---------------------------------------------------------------------------
# L5: a gated Tier 3 run that never ran is FAIL in the HTML report too
# ---------------------------------------------------------------------------


def test_html_gated_skip_is_fail_and_never_says_advisory() -> None:
    skipped = advisory_skip_result("Tier 3 live evaluation skipped: docker missing", skill_name="gate-demo")
    skipped.metadata["gating"] = {"tier": 3, "blocking": True}

    card, verdict, lead = _html_tier3_status(skipped)

    assert (card, verdict) == ("FAIL", "FAIL")
    assert "docker missing" in lead
    assert "does not block required validation" not in lead
    assert "--block-on-agent-eval made Tier 3 gate this run" in lead


def test_html_advisory_skip_stays_skipped() -> None:
    skipped = advisory_skip_result("Tier 3 live evaluation skipped: docker missing", skill_name="gate-demo")
    skipped.metadata["gating"] = {"tier": 3, "blocking": False}

    card, verdict, lead = _html_tier3_status(skipped)

    assert (card, verdict) == ("SKIPPED", "SKIPPED")
    assert "does not block required validation" in lead


# ---------------------------------------------------------------------------
# L32: one threshold set for dimension rows, pills and the standalone display
# ---------------------------------------------------------------------------


def test_html_dimension_row_and_pills_use_the_dimension_thresholds() -> None:
    # check-28 collector attr: a 0.56 security score is a PASS dimension, so every view colors it as a pass.
    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all(
        [tier3(build_payload(scores(security=0.56)))]
    )

    row = re.findall(r'<span class="metric-value" style="color:([^"]+)">0\.56</span>', html)
    pills = re.findall(r'<span class="t3-dim-score" style="color:([^;"]+);">0\.56</span>', html)
    assert row == ["var(--success-color)"]
    assert pills and set(pills) == {"var(--success-color)"}


def test_html_multi_agent_dimension_cells_use_the_dimension_thresholds() -> None:
    def info() -> dict[str, object]:
        return {
            "with_skill": scores(security=0.56),
            "without_skill": {},
            "execution_status": "succeeded",
            "rewards": [],
            "num_trials": 2,
            "model": "test-model",
        }

    payload = build_agent_eval_payload(
        "gate-demo",
        {"codex": info(), "claude-code": info()},
        use_llm_judge=False,
        run_config={"eval_target": {"kind": "plugin"}},
    )
    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all(
        [_validation_result_from_payload(payload)]
    )

    cells = re.findall(r'<span class="t3-mad-score" style="color:([^;"]+);">0\.56</span>', html)
    assert cells == ["var(--success-color)", "var(--success-color)"]


def test_standalone_result_display_colors_a_passing_dimension_score_green() -> None:
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system="standard", width=200)
    render_evaluation_result(
        {
            "execution_status": "succeeded",
            "agents": {"codex": {"execution_status": "succeeded", "with_skill": scores(security=0.56)}},
        },
        console=console,
    )

    security = next(line for line in stream.getvalue().splitlines() if "Security" in line and "0.56" in line)
    # Bold green (32), the PASS color; 0.56 is at or above the 0.50 dimension pass threshold.
    assert re.search(r"\x1b\[1;32m\s*0\.56", security)


# ---------------------------------------------------------------------------
# L32: an INCOMPLETE run prints no with-plugin percentages as results
# ---------------------------------------------------------------------------


def test_incomplete_run_prints_no_tier3_scores_in_results_at_a_glance() -> None:
    # check-13 pos-03 replay shape: one baseline trial failed, so the run is INCOMPLETE (44 of 45 baseline attempts).
    conditions = {
        "with_skill": {"execution_status": "succeeded", "expected_attempts": 45, "scored_attempts": 45},
        "without_skill": {"execution_status": "failed", "expected_attempts": 45, "scored_attempts": 44},
    }
    payload = build_payload(
        scores(security=0.9, accuracy=0.8),
        scores(),
        execution_status="failed",
        execution_errors=["Scored attempt coverage is 44/45"],
        conditions=conditions,
        num_trials_baseline=44,
        plugin_provenance={"partial": True, "execution_incomplete": "Tier 3 plugin evaluation did not complete"},
    )

    card = BenchmarkReporter(content_type="plugin", skill_name="gate-demo").render_all([tier3(payload)])
    glance = section(card, "Results at a Glance")

    assert "INCOMPLETE" in glance
    assert "with-plugin 45 of 45 attempts scored" in glance
    assert "baseline 44 of 45 attempts scored" in glance
    assert not re.search(r"\d+%", glance)
    assert "baseline not run" not in glance


def test_complete_run_still_prints_its_scores() -> None:
    card = BenchmarkReporter(content_type="plugin", skill_name="gate-demo").render_all(
        [tier3(build_payload(scores(security=0.9)))]
    )
    glance = section(card, "Results at a Glance")

    assert "| Security | 90% — baseline not run; uplift unavailable |" in glance
    assert "INCOMPLETE run" not in glance
