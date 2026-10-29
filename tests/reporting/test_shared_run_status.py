# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting.run_status import Status, resolve_run_status


def failure_and_missing():
    result = ValidationResult(validator_name="Security")
    result.add_finding(
        Finding(
            category="SECURITY",
            severity=Severity.HIGH,
            check_name="blocked",
            message="Unsafe declaration",
            file_path="manifest.json",
        )
    )
    result.mark_scan_incomplete("scanner")
    return result


def test_real_failure_outranks_missing_evidence_in_same_result():
    result = failure_and_missing()
    status = resolve_run_status([result])
    assert status.headline == Status.FAILED
    assert status.of(result) == Status.FAILED
    assert status.missing == ("scanner",)


def test_failure_outranks_missing_evidence_in_another_result():
    missing = ValidationResult(validator_name="Missing")
    missing.mark_scan_incomplete("missing-tool")
    failed = ValidationResult(validator_name="Failed")
    failed.add_error("failed check")
    assert resolve_run_status([missing, failed]).headline == Status.FAILED
    assert resolve_run_status([failed, missing]).headline == Status.FAILED


def test_advisory_missing_evidence_does_not_change_exit_gate():
    missing = ValidationResult(validator_name="Advisory")
    missing.mark_scan_incomplete("optional-tool")
    missing.metadata["gating"] = {"tier": 2, "blocking": False}
    status = resolve_run_status([missing])
    assert status.headline == Status.PASSED
    assert status.of(missing) == Status.INCOMPLETE
    assert status.advisory
    assert status.publication == Status.INCOMPLETE


def test_clean_skip_is_distinct_from_pass():
    skipped = ValidationResult(validator_name="Skipped", metadata={"skipped": True})
    status = resolve_run_status([skipped])
    assert status.of(skipped) == Status.SKIPPED
    assert status.tiers[1].status == Status.SKIPPED


def test_report_surfaces_share_failure_precedence():
    from skillevaluator.reporting.cli import CLIReporter
    from skillevaluator.reporting.html import HTMLReporter
    from skillevaluator.reporting.json_reporter import JSONReporter
    from skillevaluator.reporting.markdown import MarkdownReporter
    from skillevaluator.reporting.sarif_reporter import SARIFReporter

    result = failure_and_missing()
    data = json.loads(JSONReporter().render_all([result]))
    assert data["overall_status"] == "failed"
    assert data["results"][0]["status"] == "failed"
    assert "[FAIL]" in CLIReporter().render_all([result])
    assert "**Status:** ❌ FAILED" in MarkdownReporter().render_all([result])
    assert '"status": "failed"' in HTMLReporter().render_all([result]) or "FAILED" in HTMLReporter().render_all(
        [result]
    )
    sarif = json.loads(SARIFReporter().render_all([result]))
    assert sarif["runs"][0]["properties"]["status"] == "failed"


def test_inconclusive_tier3_cannot_hide_a_confirmed_security_failure():
    result = failure_and_missing()
    result.validator_name = "AGENT_EVAL"
    result.metadata["agent_eval"] = {"verdict": "inconclusive", "summary": {"verdict": "inconclusive"}}
    status = resolve_run_status([result])
    assert status.of(result) == Status.FAILED
    assert status.headline == Status.FAILED
    assert status.missing == ("scanner",)


def test_confirmed_legacy_failure_survives_separate_missing_evidence():
    result = ValidationResult(validator_name="Code Integrity & Hygiene")
    result.add_error("Banned dependency found in one audited member")
    result.metadata["confirmed_failure"] = True
    result.mark_scan_incomplete("another-member")
    status = resolve_run_status([result])
    assert status.headline == Status.FAILED
    assert status.of(result) == Status.FAILED


def test_missing_scanner_process_error_alone_remains_incomplete():
    result = ValidationResult(validator_name="Scanner")
    result.add_error("The scanner produced no trustworthy report")
    result.mark_scan_incomplete("scanner")
    assert resolve_run_status([result]).headline == Status.INCOMPLETE


def test_gate_status_does_not_repeat_publication_payload_analysis(monkeypatch):
    from types import SimpleNamespace

    from skillevaluator.reporting import run_status

    calls = []

    def assess(*_args, **_kwargs):
        calls.append(True)
        return SimpleNamespace(status="incomplete")

    monkeypatch.setattr(run_status, "assess_publication", assess)
    status = resolve_run_status([ValidationResult(validator_name="Static")])
    assert status.headline == Status.PASSED
    assert calls == []
    assert status.publication == Status.INCOMPLETE
    assert status.publication == Status.INCOMPLETE
    assert calls == [True]


def test_cli_and_markdown_distinguish_explicit_na_from_missing_scores():
    from skillevaluator.reporting.cli import CLIReporter
    from skillevaluator.reporting.markdown import MarkdownReporter

    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = {
        "evaluators": {
            "accuracy": {
                "with_skill": None,
                "baseline": None,
                "lift": None,
                "with_skill_status": "not_applicable",
                "baseline_status": "missing",
            }
        }
    }
    terminal = CLIReporter().render(result)
    assert "N/A" in terminal and "NO SCORE" in terminal
    markdown = MarkdownReporter().render_all([result])
    assert "| Accuracy | N/A | NO SCORE | N/A |" in markdown
