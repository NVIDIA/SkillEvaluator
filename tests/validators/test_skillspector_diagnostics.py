# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completeness evidence survives validation, merging, and report rendering."""

import json
from pathlib import Path

import pytest
from rich.text import Text

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting import CLIReporter, HTMLReporter, JSONReporter, MarkdownReporter
from skillevaluator.reporting.retained import retained_results
from skillevaluator.skillspector_diagnostics import (
    completeness_record,
    merge_completeness,
    sanitize_completeness,
)
from skillevaluator.validators.policy import apply_policy, default_policy, load_policy_file
from skillevaluator.validators.security import SecurityValidator


def partial_result() -> ValidationResult:
    payload = json.loads((Path(__file__).parents[1] / "fixtures" / "skillspector-2.13.3-safe-no-llm.json").read_text())
    payload["risk_assessment"]["recommendation"] = "CAUTION"
    payload["analysis_completeness"].update(
        is_complete=False,
        status="partial",
        ledger_exceptions=[
            {
                "fatal": False,
                "path": "SKILL.md",
                "start_line": 8,
                "phase": "reference_resolution",
                "reason_code": "reference_missing",
                "analyzers": [],
            }
        ],
    )
    result = ValidationResult(validator_name="Security Scan")
    assert SecurityValidator._validate_skillspector_report(payload, 0, False, "skillspector", result)
    return result


@pytest.mark.parametrize("terminal", ["dumb", "xterm-256color"])
def test_partial_evidence_is_retained_without_turning_it_into_a_pass(monkeypatch, terminal):
    monkeypatch.setenv("TERM", terminal)
    monkeypatch.delenv("NO_COLOR", raising=False)
    result = partial_result()
    apply_policy([result], default_policy())
    assert result.status == "incomplete"
    assert not result.passed
    assert result.findings == []  # Coverage diagnostics are not security defects.
    for reporter in (CLIReporter(), HTMLReporter(), MarkdownReporter(), JSONReporter()):
        output = reporter.render(result)
        if isinstance(reporter, CLIReporter):
            output = Text.from_ansi(output).plain
        assert "reference_missing" in output
        assert "SKILL.md:8" in output
    data = json.loads(JSONReporter().render(result))
    row = data["skillspector_completeness"]["records"][0]["ledger_exceptions"][0]
    assert row["path"] == "SKILL.md"
    assert row["start_line"] == 8
    assert row["phase"] == "reference_resolution"


def test_merge_preserves_both_stages_and_skill_identity():
    static, llm = partial_result(), partial_result()
    llm.metadata["skillspector_completeness"]["records"][0]["stage"] = "skillspector-llm"
    static.merge(llm)
    folder = ValidationResult()
    folder.merge_with_prefix(static, "contact-skill")
    records = json.loads(JSONReporter().render(folder))["skillspector_completeness"]["records"]
    assert [record["stage"] for record in records] == ["skillspector", "skillspector-llm"]
    assert {record["skill"] for record in records} == {"contact-skill"}
    assert folder.is_incomplete


def test_policy_recalculation_keeps_completeness_error_and_retained_report_evidence(tmp_path):
    result = partial_result()
    result.add_structured_finding(
        Finding(
            category="PII",
            check_name="ownership_emails",
            severity=Severity.HIGH,
            message="contact",
            file_path="SKILL.md",
        )
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("severity_overrides:\n  PII.ownership_emails: info\n")
    apply_policy([result], load_policy_file(policy_path))
    assert result.status == "incomplete"
    assert any("reference_missing" in error for error in result.errors)
    retained = retained_results(json.loads(JSONReporter().render_all([result])))[0]
    assert retained.metadata["skillspector_completeness"] == result.metadata["skillspector_completeness"]
    assert "reference_missing" in HTMLReporter().render(retained)


def test_diagnostic_text_and_lists_are_bounded_redacted_and_idempotent():
    secret = "ghp_" + "a" * 36
    raw = {
        "ledger_exceptions": [
            {
                "path": f"references/{secret}/file.md\n\x1b[31mextra",
                "phase": "static",
                "reason_code": "opaque_content",
                "analyzers": ["analyzer"] * 12,
                "start_line": True,
                "unexpected": "must not persist",
            }
        ]
        * 25,
        "limitations": ["password=fictional-password " + "x" * 800] * 10,
    }
    record = completeness_record(raw, "skillspector")
    assert record["omitted_exceptions"] == 5
    assert record["omitted_limitations"] == 2
    assert record["ledger_exceptions"][0]["omitted_analyzers"] == 4
    assert "start_line" not in record["ledger_exceptions"][0]
    encoded = json.dumps(record)
    assert secret not in encoded and "fictional-password" not in encoded
    assert "must not persist" not in encoded and "\\u001b" not in encoded and "\\n" not in encoded
    wrapped = {"records": [record], "omitted_records": 0}
    assert sanitize_completeness(wrapped) == wrapped
    merged = merge_completeness(wrapped, {"records": [record] * 70})
    assert len(merged["records"]) == 64
    assert merged["omitted_records"] == 7
    assert sanitize_completeness(merged) == merged


def test_saved_diagnostic_metadata_is_revalidated_at_json_boundary():
    result = partial_result()
    record = result.metadata["skillspector_completeness"]["records"][0]
    record["unexpected_secret"] = "not part of contract"
    record["ledger_exceptions"][0]["path"] = "password=fictional-password"
    output = JSONReporter().render(result)
    assert "fictional-password" not in output
    assert "unexpected_secret" not in output


@pytest.mark.parametrize("stage", [[], {}, None, True, 3, "unsupported"])
def test_malformed_saved_stage_cannot_crash_json_or_retained_reports(stage):
    result = partial_result()
    result.metadata["skillspector_completeness"]["records"][0]["stage"] = stage
    data = json.loads(JSONReporter().render_all([result]))
    assert data["results"][0]["skillspector_completeness"]["records"][0]["stage"] == "unknown"
    # Exercise the independent retained-report input boundary as well.
    data["results"][0]["skillspector_completeness"]["records"][0]["stage"] = stage
    retained = retained_results(data)[0]
    assert retained.is_incomplete
    assert retained.metadata["skillspector_completeness"]["records"][0]["stage"] == "unknown"
    assert "reference_missing" in HTMLReporter().render(retained)


def test_limitation_only_partial_report_keeps_its_reason_in_the_error():
    payload = json.loads((Path(__file__).parents[1] / "fixtures" / "skillspector-2.13.3-safe-no-llm.json").read_text())
    payload["risk_assessment"]["recommendation"] = "CAUTION"
    payload["analysis_completeness"].update(
        is_complete=False,
        status="partial",
        ledger_exceptions=[],
        limitations=["Binary assets were not inspected"],
    )
    result = ValidationResult(validator_name="Security Scan")

    SecurityValidator._validate_skillspector_report(payload, 0, False, "skillspector", result)

    assert result.status == "incomplete"
    assert any("Limitation: Binary assets were not inspected" in error for error in result.errors)
