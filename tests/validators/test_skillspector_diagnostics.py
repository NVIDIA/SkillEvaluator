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
        * 505,
        "limitations": ["password=fictional-password " + "x" * 800] * 10,
    }
    record = completeness_record(raw, "skillspector")
    assert 0 < record["ledger_rows_retained"] <= 500
    assert record["omitted_exceptions"] == 505 - record["ledger_rows_retained"]
    assert record["ledger_rows_total"] == 505
    assert record["omitted_limitations"] == 2
    assert record["ledger_exceptions"][0]["omitted_analyzers"] == 4
    assert "start_line" not in record["ledger_exceptions"][0]
    encoded = json.dumps(record)
    assert secret not in encoded and "fictional-password" not in encoded
    assert "must not persist" not in encoded and "\\u001b" not in encoded and "\\n" not in encoded
    wrapped = {"records": [record], "omitted_records": 0}
    assert sanitize_completeness(wrapped) == wrapped
    small_record = completeness_record({"ledger_exceptions": [{"path": "guide.md"}]}, "skillspector")
    merged = merge_completeness(wrapped, {"records": [small_record] * 520})
    assert len(merged["records"]) == 512
    assert merged["omitted_records"] == 9
    assert merged["records"][0]["ledger_rows_total"] == 505
    assert sanitize_completeness(merged) == merged


def test_parser_diagnostic_is_projected_without_source_offsets_and_survives_merge():
    raw = {
        "ledger_exceptions": [
            {
                "path": "scripts/admission.mjs",
                "phase": "static",
                "reason_code": "static_parse_limit",
                "analyzers": ["static_patterns_tool_misuse"],
                "parser_diagnostic": {
                    "cause": "runtime_dependent",
                    "check": "command_string_runtime",
                    "start_line": 74,
                    "end_line": 170,
                    "unchecked_scope": "remaining_input",
                    "full_input_checked": False,
                    "source_start_offset": 3958,
                    "source_end_offset": 8650,
                    "location_precision": "source_span",
                    "observed_is_lower_bound": False,
                },
            }
        ]
    }
    record = completeness_record(raw, "skillspector")
    diagnostic = record["ledger_exceptions"][0]["parser_diagnostic"]
    assert diagnostic == {
        "cause": "runtime_dependent",
        "check": "command_string_runtime",
        "start_line": 74,
        "end_line": 170,
        "unchecked_scope": "remaining_input",
        "full_input_checked": False,
    }
    encoded = json.dumps(record)
    assert "source_start_offset" not in encoded
    assert "3958" not in encoded
    assert "location_precision" not in encoded
    wrapped = {"records": [record], "omitted_records": 0}
    assert sanitize_completeness(wrapped) == wrapped
    merged = merge_completeness({}, wrapped, prefix="spawn-helper")
    assert merged["records"][0]["skill"] == "spawn-helper"
    assert merged["records"][0]["ledger_exceptions"][0]["parser_diagnostic"] == diagnostic
    assert sanitize_completeness(merged) == merged
    record["ledger_exceptions"][0]["parser_diagnostic"]["source_start_offset"] = 3958
    saved = sanitize_completeness({"records": [record]})
    reprojected = saved["records"][0]["ledger_exceptions"][0]["parser_diagnostic"]
    assert "source_start_offset" not in reprojected


@pytest.mark.parametrize(
    "diagnostic",
    [
        "not-a-dict",
        True,
        {"cause": "runtime_dependent", "start_line": "74", "end_line": 170},
        {"cause": True, "check": False, "start_line": -1, "end_line": 10, "full_input_checked": "no"},
        {"cause": "runtime_dependent", "start_line": 170, "end_line": 74},
    ],
    ids=["string", "bool", "string-line", "bool-fields", "end-before-start"],
)
def test_malformed_parser_diagnostic_is_dropped_or_stripped_to_safe_fields(diagnostic: object):
    record = completeness_record(
        {"ledger_exceptions": [{"path": "scripts/helper.mjs", "parser_diagnostic": diagnostic}]},
        "skillspector",
    )
    projected = record["ledger_exceptions"][0].get("parser_diagnostic")
    if projected is None:
        return
    assert "start_line" not in projected
    assert "end_line" not in projected
    assert projected.get("full_input_checked") is not True
    text_keys = {key for key in projected if key not in {"start_line", "end_line", "full_input_checked"}}
    assert all(isinstance(projected[key], str) for key in text_keys)


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


def test_ledger_keeps_every_row_of_a_realistic_library_and_states_the_totals():
    """A skill with dozens of incomplete files keeps every row; the record says N of N."""
    rows = [
        {"path": f"scripts/file_{i}.sh", "phase": "static", "reason_code": "static_parse_limit", "analyzers": ["a"]}
        for i in range(47)
    ]
    record = completeness_record({"ledger_exceptions": rows, "limitations": []}, "skillspector")
    assert len(record["ledger_exceptions"]) == 47
    assert record["omitted_exceptions"] == 0
    assert record["ledger_rows_retained"] == record["ledger_rows_total"] == 47

    # Re-projecting a saved record that already carried omissions keeps the total honest.
    saved = {**record, "omitted_exceptions": 3}
    again = sanitize_completeness({"records": [saved], "omitted_records": 0})["records"][0]
    assert again["ledger_rows_retained"] == 47
    assert again["omitted_exceptions"] == 3
    assert again["ledger_rows_total"] == 50


def test_ledger_row_byte_limit_matches_escaped_json_and_preserves_totals(monkeypatch):
    import skillevaluator.skillspector_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_MAX_LEDGER_ROW_BYTES", 1_024)
    short = {"path": "scripts/one.py", "phase": "static", "reason_code": "static_parse_limit"}
    wide = {**short, "path": f"scripts/{'é' * 130}.py"}
    assert completeness_record({"ledger_exceptions": [wide]}, "skillspector")["ledger_exceptions"] == [wide]

    record = completeness_record({"ledger_exceptions": [short, wide, short]}, "skillspector")

    assert record["ledger_exceptions"] == [short]
    assert record["ledger_rows_retained"] == 1
    assert record["ledger_rows_total"] == 3
    assert record["omitted_exceptions"] == 2
    assert sanitize_completeness({"records": [record]})["records"] == [record]


def test_gate_message_names_gap_ledger_rows_and_cap(monkeypatch):
    import skillevaluator.skillspector_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_MAX_ROWS", 2)
    payload = json.loads((Path(__file__).parents[1] / "fixtures" / "skillspector-2.13.3-safe-no-llm.json").read_text())
    payload["risk_assessment"]["recommendation"] = "CAUTION"
    payload["analysis_completeness"].update(
        is_complete=False,
        status="partial",
        ledger_exceptions=[
            {
                "fatal": False,
                "path": f"scripts/file_{i}.sh",
                "phase": "static",
                "reason_code": "static_parse_limit",
                "analyzers": ["static_patterns_shell"],
            }
            for i in range(5)
        ],
    )
    result = ValidationResult(validator_name="Security Scan")
    assert SecurityValidator._validate_skillspector_report(payload, 0, False, "skillspector", result)
    record = result.metadata["skillspector_completeness"]["records"][0]
    assert record["ledger_rows_retained"] == 2 and record["ledger_rows_total"] == 5
    assert any(
        "The completeness ledger for these gaps has 5 rows, with 3 beyond the per-record limit" in error
        for error in result.errors
    )


def test_folder_merge_projects_each_incoming_record_once(monkeypatch):
    import skillevaluator.skillspector_diagnostics as diagnostics

    projected = 0
    original = diagnostics.completeness_record

    def record(payload, stage):
        nonlocal projected
        projected += 1
        return original(payload, stage)

    monkeypatch.setattr(diagnostics, "completeness_record", record)
    combined = ValidationResult(validator_name="Security Scan")
    for index in range(20):
        member = ValidationResult(validator_name="Security Scan")
        member.metadata["skillspector_completeness"] = {
            "records": [{"stage": "skillspector", "ledger_exceptions": [{"path": "guide.md"}]}]
        }
        combined.merge_with_prefix(member, f"skill-{index}")
        member.metadata["skillspector_completeness"]["records"][0]["ledger_exceptions"][0]["path"] = "modified.md"

    assert projected == 20
    saved = diagnostics.sanitize_completeness(combined.metadata["skillspector_completeness"])
    assert projected == 40
    assert [record["skill"] for record in saved["records"]] == [f"skill-{index}" for index in range(20)]
    assert all(item["ledger_exceptions"][0]["path"] == "guide.md" for item in saved["records"])

    aggregate = combined.metadata["skillspector_completeness"]
    standalone = merge_completeness(aggregate, None)
    standalone["records"][0]["ledger_exceptions"][0]["path"] = "modified.md"
    assert aggregate["records"][0]["ledger_exceptions"][0]["path"] == "guide.md"
