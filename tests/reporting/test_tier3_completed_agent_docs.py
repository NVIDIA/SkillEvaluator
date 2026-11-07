# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the completed-agent docs aligned with what Tier 3 reports render."""

from __future__ import annotations

import re
from pathlib import Path

from tests.reporting.test_tier3_scoped_reports import result_with_completed_peer

from skillevaluator.reporting import BenchmarkReporter, HTMLReporter

REPO = Path(__file__).resolve().parents[2]
HEADING = "### Completed agents in an incomplete run"
ANCHOR = "completed-agents-in-an-incomplete-run"


def _plain(text: str) -> str:
    """Drop HTML tags and Markdown emphasis, and collapse whitespace."""
    return " ".join(re.sub(r"<[^>]+>", " ", text).replace("*", "").split())


def _reports_page() -> str:
    return (REPO / "docs" / "reports.mdx").read_text(encoding="utf-8")


def _section() -> str:
    page = _reports_page()
    start = page.index(HEADING)
    return _plain(page[start : page.index("\n## ", start)])


def test_documented_labels_match_a_run_with_a_failed_peer() -> None:
    result = result_with_completed_peer()
    payload = result.metadata["agent_eval"]
    payload["agents"]["opencode"]["conditions"]["with_skill"]["recovered_attempts_total"] = 1
    section = _section()
    html = _plain(HTMLReporter(report_scope="tier3").render(result))
    scoped = _plain(BenchmarkReporter(report_scope="tier3").render(result))
    card = _plain(BenchmarkReporter().render(result))

    assert payload["execution_status"] == "failed"
    assert payload["summary"]["overall_score"] is None
    for label in (
        "Completed-agent result. Other execution failures need attention",
        "Completed Agent Score",
        "Quality Verdict",
    ):
        assert label in section
        assert label in html
    assert "1 timeout attempt(s) recovered" in html
    needs_attention = "Needs attention: other execution failures remain"
    assert needs_attention in section
    assert needs_attention in scoped
    assert "Recovered timeout attempts: 1" in scoped
    # A validate card shows the same lines under its unchanged verdict.
    assert needs_attention in card
    assert "Overall verdict: FAIL" in card


def test_documented_line_when_no_agent_qualifies() -> None:
    result = result_with_completed_peer()
    # The best agent's coverage is incomplete; the report does not fall back.
    result.metadata["agent_eval"]["agents"]["opencode"]["scored_attempts"] = 1
    line = "Tier 3: No complete agent result"

    assert line in _section()
    assert line in _plain(BenchmarkReporter(report_scope="tier3").render(result))


def test_results_tree_and_tier3_guide_point_to_the_section() -> None:
    page = _reports_page()
    guide = (REPO / "docs" / "tier3-live-evaluation.mdx").read_text(encoding="utf-8")

    assert "      - tier3-benchmark.md\n" in page
    assert f"(#{ANCHOR})" in page
    assert f"(reports.mdx#{ANCHOR})" in guide
