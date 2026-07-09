# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advisory Tier 3 skips stay visible without failing required validation."""

from __future__ import annotations

import json
import re

from skillevaluator.evaluation.tier3_report import advisory_skip_result
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, HTMLReporter, JSONReporter, MarkdownReporter


def _html_report_data(output: str) -> dict:
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', output, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _html_tab(output: str, tab_id: str) -> str:
    marker = f'id="tab-{tab_id}"'
    start = output.index(marker)
    next_tab = output.find('id="tab-', start + len(marker))
    report_data = output.find('id="report-data"', start + len(marker))
    candidates = [position for position in (next_tab, report_data) if position >= 0]
    end = min(candidates) if candidates else len(output)
    return output[start:end]


def test_advisory_skip_is_non_blocking_in_json() -> None:
    payload = json.loads(
        JSONReporter(include_timestamp=False).render_all(
            [advisory_skip_result("No public provider key", skill_name="demo")]
        )
    )

    assert payload["overall_status"] == "passed"
    assert payload["overall_passed"] is True
    assert payload["total_advisory_skipped"] == 1
    assert payload["results"][0]["passed"] is False
    assert payload["results"][0]["status"] == "skipped"


def test_advisory_skip_is_non_blocking_in_markdown() -> None:
    output = MarkdownReporter(include_timestamp=False).render_all(
        [advisory_skip_result("No public provider key", skill_name="demo")]
    )

    assert "**Status:** ✅ PASSED" in output
    assert "| Validator Results | 1 |" in output
    assert "| Validators Run |" not in output
    assert "| ⏭️ Advisory skips | 1 |" in output
    assert "### ⏭️ SKIPPED AGENT_EVAL" in output
    assert "❌ FAILED" not in output


def test_advisory_skip_is_non_blocking_in_html_and_benchmark() -> None:
    result = advisory_skip_result("No public provider key", skill_name="demo")
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])

    assert html_data["summary"]["status"] == "passed"
    assert html_data["summary"]["all_passed"] is True
    assert html_data["summary"]["passed_count"] == 0
    assert html_data["summary"]["failed_count"] == 0
    assert html_data["summary"]["advisory_skipped_count"] == 1
    assert html_data["results"][0]["status"] == "skipped"
    assert html_data["gating"]["would_block"] is False
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert re.search(r'tier-card-verdict">\s*SKIPPED\s*</span>', html)
    assert "Live evaluation skipped:</strong> No public provider key" in html
    assert not re.search(r'tier-card-verdict">\s*NEUTRAL\s*</span>', html)
    assert "Overall verdict: PASS" in benchmark
    assert "Overall verdict: FAIL" not in benchmark
    assert "Tier 3 live evaluation: SKIPPED — No public provider key" in benchmark
    assert "based on the completed required-tier results" in benchmark


def test_advisory_skip_does_not_appear_as_tier1_failure_in_html() -> None:
    schema = ValidationResult(validator_name="SCHEMA", validator_description="Schema validation")
    skipped = advisory_skip_result("No public provider key", skill_name="demo")

    output = HTMLReporter(include_timestamp=False).render_all([schema, skipped])
    html_data = _html_report_data(output)
    tier1 = _html_tab(output, "tier1")

    assert html_data["summary"]["passed_count"] == 1
    assert html_data["summary"]["failed_count"] == 0
    assert html_data["summary"]["advisory_skipped_count"] == 1
    assert re.search(r"Validators Run</h3>\s*<p[^>]*>1</p>", tier1)
    assert re.search(r"Failed</h3>\s*<p[^>]*>0</p>", tier1)
    assert "AGENT_EVAL" not in tier1


def test_real_agent_eval_failure_still_fails_required_validation() -> None:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.add_error("Harbor execution failed")

    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))

    assert payload["overall_status"] == "failed"
    assert payload["overall_passed"] is False
    assert payload["results"][0]["status"] == "failed"

    html = HTMLReporter(include_timestamp=False).render_all([result])
    tier3 = _html_tab(html, "tier3")
    assert re.search(r'tier-card-verdict">\s*FAIL\s*</span>', html)
    assert "Harbor execution failed" in tier3
