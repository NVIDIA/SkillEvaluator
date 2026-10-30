# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incomplete scanner evidence is non-green in every report format."""

import json
import re

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import (
    BenchmarkReporter,
    CLIReporter,
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
)


def _plain(output: str) -> str:
    """Strip ANSI styling and table box-drawing, collapse wrapping whitespace."""
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    output = re.sub(r"[─-╿]", " ", output)
    return " ".join(output.split())


def _assert_html_verdict_is_incomplete(output: str) -> None:
    """The header verdict and the summary verdict both say Incomplete, and nothing says Passed."""
    header = output.split('<header class="main-header">', 1)[1].split("</header>", 1)[0]
    summary = output.split('<section class="exec-summary', 1)[1].split("</section>", 1)[0]

    assert 'data-verdict="incomplete"' in header
    assert '<span class="sr-only">Verdict: </span>Incomplete</span>' in header
    assert '<span class="verdict-chip warn">Incomplete</span>' in summary
    assert 'data-verdict="pass"' not in output
    assert "verdict-chip pass" not in output


def _incomplete_result() -> ValidationResult:
    result = ValidationResult(
        validator_name="Security Scan",
        validator_description="Detect security vulnerabilities",
    )
    result.mark_scan_incomplete("skillspector")
    result.add_warning("skillspector timed out after 300 seconds")
    return result


class TestIncompleteScanReporting:
    def test_incomplete_scan_is_a_non_green_result(self):
        result = _incomplete_result()

        assert result.passed is False
        assert result.status == "incomplete"

    def test_cli_uses_incomplete_status_and_never_pass(self):
        output = _plain(CLIReporter().render_all([_incomplete_result()]))

        assert "INCOMPLETE" in output
        assert "skillspector did not complete" in output
        assert "[PASS]" not in output

    def test_cli_report_shows_incomplete_warning(self):
        output = _plain(CLIReporter().render_all([_incomplete_result()]))

        assert "skillspector timed out after 300 seconds" in output

    def test_findings_do_not_hide_incomplete_status(self):
        result = _incomplete_result()
        result.add_error("Real finding elsewhere")
        output = _plain(CLIReporter().render_all([result]))

        assert "INCOMPLETE" in output
        assert "skillspector did not complete" in output

    def test_json_marks_overall_and_result_incomplete(self):
        payload = json.loads(JSONReporter(include_timestamp=False).render_all([_incomplete_result()]))

        assert payload["overall_passed"] is False
        assert payload["overall_status"] == "incomplete"
        assert payload["incomplete_scans"] == ["skillspector"]
        assert payload["results"][0]["passed"] is False
        assert payload["results"][0]["status"] == "incomplete"
        assert payload["results"][0]["incomplete_scans"] == ["skillspector"]

    def test_markdown_marks_incomplete_and_shows_diagnostic(self):
        output = MarkdownReporter(include_timestamp=False).render_all([_incomplete_result()])

        assert "**Status:** ⚠️ INCOMPLETE" in output
        assert "skillspector timed out after 300 seconds" in output
        assert "✅ PASSED" not in output

    def test_html_marks_incomplete_and_never_claims_a_pass(self):
        output = HTMLReporter(include_timestamp=False).render_all([_incomplete_result()])

        _assert_html_verdict_is_incomplete(output)
        assert "skillspector" in output

    def test_html_incomplete_check_among_passing_checks_is_incomplete(self):
        passing = []
        for name in ("License Compliance", "PII Scan"):
            result = ValidationResult(validator_name=name, validator_description="d")
            result.add_success(check_name=name, message="No issues")
            passing.append(result)

        output = HTMLReporter(include_timestamp=False, explicit_skill_name="demo").render_all(
            [passing[0], _incomplete_result(), passing[1]]
        )

        _assert_html_verdict_is_incomplete(output)

    def test_benchmark_blocks_publication_on_incomplete_evidence(self):
        output = BenchmarkReporter(include_timestamp=False).render_all([_incomplete_result()])

        assert "Overall verdict: INCOMPLETE" in output
        assert "skillspector" in output
        assert "incomplete" in output.lower()
        assert "suitable to proceed" not in output

    def test_clean_pass_is_not_flagged(self):
        result = ValidationResult(validator_name="Security Scan", validator_description="d")
        result.add_success(check_name="skillspector", message="No issues")
        output = _plain(CLIReporter().render_all([result]))
        assert "did not complete" not in output
        assert "INCOMPLETE" not in output
