# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M15 (report half): the lift band is shown, and the BENCHMARK legend says what the code applies (check-13 edge-05)."""

from __future__ import annotations

import re

from _f90_gate_payloads import EDGE05_INTERVAL, EDGE05_WITH, EDGE05_WITHOUT, build_payload, terminal, tier3

from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting.benchmark import BenchmarkReporter

# ---------------------------------------------------------------------------
# M15: the lift band is real, and the legend says what it does
# ---------------------------------------------------------------------------


def test_methodology_legend_describes_the_applied_lift_rule() -> None:
    payload = build_payload(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL)
    card = BenchmarkReporter(content_type="plugin", skill_name="gate-demo").render_all([tier3(payload)])
    legend = card.split("## Scoring Methodology", 1)[1]

    assert "Lift is reported as diagnostic evidence and does not override this gate." not in legend
    assert "confirmed regression" in legend
    assert "--block-on-agent-eval" in legend


def test_negative_lift_banner_in_benchmark_and_html() -> None:
    payload = build_payload(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL)
    result = tier3(payload, blocking=False)

    card = BenchmarkReporter(content_type="plugin", skill_name="gate-demo").render_all([result])
    head = card.split("## Evaluation Metadata")[0]
    assert "Skill Lift" in head and "-20 points" in head and "[-20, -19]" in head

    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all([result])
    lead = re.search(r'<div class="t3-overview-lead">(.*?)</div>', html, re.DOTALL)
    assert lead is not None
    assert "Skill Lift band" in lead.group(1) and "FAIL" in lead.group(1)
    assert "advisory" in lead.group(1)


def test_terminal_verdict_line_shows_the_lift_band() -> None:
    payload = build_payload(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL)
    output = terminal([tier3(payload, blocking=True)])

    assert "Verdict: PASS (composite lift = -0.20; Skill Lift band FAIL, confirmed regression)" in output
