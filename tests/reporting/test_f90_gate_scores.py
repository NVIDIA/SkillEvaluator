# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof L32 (report half): one threshold set for the evaluator card and the dimension verdict, and no
\"baseline not run\" claim for an INCOMPLETE run whose baseline did run (check-28 notes 6 and 11)."""

from __future__ import annotations

import re

import pytest
from _f90_gate_payloads import build_payload, scores, section, tier3

from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting.benchmark import BenchmarkReporter

# ---------------------------------------------------------------------------
# L32: one threshold set for the evaluator card and the dimension verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("security", "verdict"), [(0.56, "PASS"), (0.45, "NEUTRAL"), (0.3, "FAIL")])
def test_evaluator_card_status_matches_the_dimension_verdict(security: float, verdict: str) -> None:
    payload = build_payload(scores(security=security))

    dimension = next(item for item in payload["dimensions"] if item["id"] == "security")
    card = next(item for item in payload["evaluator_cards"] if item["id"] == "security")

    assert dimension["verdict"] == verdict
    assert card["status"] == {"PASS": "pass", "NEUTRAL": "warn", "FAIL": "fail"}[verdict]


def test_html_security_card_pill_agrees_with_the_dimension_row() -> None:
    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all(
        [tier3(build_payload(scores(security=0.56)))]
    )
    card = re.search(
        r'<div class="t3-eval-title">Security</div>.*?<span class="t3-pill ([a-z]+)">([a-z]+)</span>', html, re.DOTALL
    )

    assert card is not None
    assert card.group(2) == "pass"
    assert card.group(1) in {"ok", "pass"}


def test_incomplete_run_does_not_claim_the_baseline_was_not_run() -> None:
    # check-13 pos-03 / check-28 note 11: one baseline trial timed out, so the run is INCOMPLETE and the payload has
    # no comparable baseline. The baseline arm did run (44 of 45 attempts scored).
    conditions = {
        "with_skill": {"execution_status": "succeeded", "expected_attempts": 45, "scored_attempts": 45},
        "without_skill": {"execution_status": "failed", "expected_attempts": 45, "scored_attempts": 44},
    }
    payload = build_payload(
        scores(security=0.9),
        scores(),
        execution_status="failed",
        execution_errors=["Scored attempt coverage is 44/45"],
        conditions=conditions,
        num_trials_baseline=44,
        plugin_provenance={"partial": True, "execution_incomplete": "Tier 3 plugin evaluation did not complete"},
    )
    assert payload["agents"]["codex"]["baseline"] is None

    card = BenchmarkReporter(content_type="plugin", skill_name="gate-demo").render_all([tier3(payload)])
    glance = section(card, "Results at a Glance")

    assert "baseline not run" not in glance
    assert "44 of 45" in glance
    assert "INCOMPLETE" in glance


def test_skip_baseline_run_still_says_baseline_not_run() -> None:
    card = BenchmarkReporter(content_type="plugin", skill_name="gate-demo").render_all([tier3(build_payload(scores()))])

    assert "baseline not run" in section(card, "Results at a Glance")
