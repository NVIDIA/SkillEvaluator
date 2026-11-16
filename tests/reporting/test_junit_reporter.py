# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JUnit XML reporter."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.junit_reporter import JUnitReporter
from skillevaluator.tier1.commands import emit_reports

_TEXT_ANALYZER = (
    Path(__file__).resolve().parents[2] / "src/skillevaluator/tier3/reference_skills/text-analyzer"
)


def _cases(root: ET.Element) -> list[ET.Element]:
    return list(root.iter("testcase"))


def _count_children(cases: list[ET.Element], tag: str) -> int:
    return sum(1 for case in cases if case.find(tag) is not None)


def _assert_counts_match(root: ET.Element) -> None:
    cases = _cases(root)
    failures = _count_children(cases, "failure")
    skipped = _count_children(cases, "skipped")
    assert int(root.attrib["tests"]) == len(cases)
    assert int(root.attrib["failures"]) == failures
    assert int(root.attrib["skipped"]) == skipped
    assert int(root.attrib["errors"]) == 0
    for suite in root.findall("testsuite"):
        suite_cases = suite.findall("testcase")
        assert int(suite.attrib["tests"]) == len(suite_cases)
        assert int(suite.attrib["failures"]) == _count_children(suite_cases, "failure")
        assert int(suite.attrib["skipped"]) == _count_children(suite_cases, "skipped")
        assert int(suite.attrib["errors"]) == 0


def test_name_and_extension() -> None:
    reporter = JUnitReporter()
    assert reporter.name == "junit"
    assert reporter.get_file_extension() == ".junit.xml"


def test_pass_emits_testcase_without_failure() -> None:
    result = ValidationResult(validator_name="SCHEMA", validator_description="Schema checks")
    result.add_success("manifest_found", "Found SKILL.md in skill directory")
    result.add_success("frontmatter_valid", "Parsed successfully")

    document = JUnitReporter().render_all([result])
    root = ET.fromstring(document)

    assert root.tag == "testsuites"
    assert len(root.findall("testsuite")) == 1
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.attrib["name"] == "SCHEMA"
    cases = suite.findall("testcase")
    assert [case.attrib["name"] for case in cases] == ["manifest_found", "frontmatter_valid"]
    assert all(case.find("failure") is None and case.find("skipped") is None for case in cases)
    _assert_counts_match(root)


def test_fail_includes_file_and_line() -> None:
    result = ValidationResult(validator_name="SECRETS", validator_description="Hardcoded secrets")
    result.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.CRITICAL,
            check_name="aws-access-key-id",
            message="AWS Access Key ID detected",
            file_path="config/settings.py",
            line_number=15,
            suggestion="Remove hardcoded key. Use environment variables.",
        )
    )

    root = ET.fromstring(JUnitReporter().render_all([result]))
    case = root.find("testsuite/testcase")
    assert case is not None
    assert case.attrib["name"] == "aws-access-key-id"
    assert case.attrib["file"] == "config/settings.py"
    assert case.attrib["line"] == "15"
    failure = case.find("failure")
    assert failure is not None
    assert failure.attrib["message"] == "AWS Access Key ID detected"
    assert "config/settings.py:15" in (failure.text or "")
    assert "AWS Access Key ID detected" in (failure.text or "")
    _assert_counts_match(root)
    assert int(root.attrib["failures"]) == 1
    assert int(root.attrib["skipped"]) == 0


def test_skipped_and_incomplete_are_not_all_passed() -> None:
    skipped = ValidationResult(validator_name="SCHEMA", validator_description="Schema checks")
    skipped.metadata["skipped"] = True
    skipped.metadata["skip_reason"] = "skipped with --skip"

    incomplete = ValidationResult(validator_name="BANDIT", validator_description="Bandit")
    incomplete.add_error("Bandit did not return valid JSON; scan did not complete")
    incomplete.mark_scan_incomplete("bandit")

    root = ET.fromstring(JUnitReporter().render_all([skipped, incomplete]))
    suites = {suite.attrib["name"]: suite for suite in root.findall("testsuite")}

    skipped_case = suites["SCHEMA"].find("testcase")
    assert skipped_case is not None
    skipped_el = skipped_case.find("skipped")
    assert skipped_el is not None
    assert "skipped with --skip" in (skipped_el.attrib.get("message") or "")
    assert skipped_case.find("failure") is None

    incomplete_suite = suites["BANDIT"]
    incomplete_cases = incomplete_suite.findall("testcase")
    assert any(case.find("skipped") is not None for case in incomplete_cases)
    skipped_incomplete = next(case for case in incomplete_cases if case.find("skipped") is not None)
    skipped_el = skipped_incomplete.find("skipped")
    assert skipped_el is not None
    assert "bandit scan did not complete" in (skipped_el.attrib.get("message") or "")
    assert int(incomplete_suite.attrib["skipped"]) >= 1
    assert int(root.attrib["skipped"]) >= 1
    passing = [
        case
        for case in _cases(root)
        if case.find("failure") is None and case.find("skipped") is None
    ]
    assert passing == []
    _assert_counts_match(root)


def test_incomplete_warning_only_is_skipped_not_passed() -> None:
    result = ValidationResult(validator_name="Security Scan", validator_description="Security")
    result.mark_scan_incomplete("skillspector")
    result.add_warning("skillspector timed out after 300 seconds")
    result.add_finding(
        Finding(
            category="SECURITY",
            severity=Severity.LOW,
            check_name="skillspector_timeout",
            message="skillspector timed out after 300 seconds",
            file_path="SKILL.md",
        )
    )

    root = ET.fromstring(JUnitReporter().render_all([result]))
    cases = _cases(root)
    assert cases
    assert all(case.find("skipped") is not None for case in cases)
    assert all(case.find("failure") is None for case in cases)
    assert int(root.attrib["tests"]) == int(root.attrib["skipped"])
    assert int(root.attrib["failures"]) == 0
    _assert_counts_match(root)


def test_xml_forbidden_characters_are_stripped() -> None:
    result = ValidationResult(validator_name="SCHEMA\x00", validator_description="schema")
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="bad\x01name",
            message="line\x00with\x08bell\x07 and \x0bvt",
            file_path="SKILL\x00.md",
            line_number=2,
        )
    )

    document = JUnitReporter().render_all([result])
    for forbidden in ("\x00", "\x01", "\x07", "\x08", "\x0b"):
        assert forbidden not in document
    root = ET.fromstring(document)
    case = root.find("testsuite/testcase")
    assert case is not None
    assert "\x00" not in case.attrib["name"]
    failure = case.find("failure")
    assert failure is not None
    assert "line" in (failure.attrib.get("message") or "")
    _assert_counts_match(root)


def test_markup_in_messages_is_escaped_and_parseable() -> None:
    result = ValidationResult(validator_name="QUALITY", validator_description="quality")
    result.add_finding(
        Finding(
            category="QUALITY",
            severity=Severity.HIGH,
            check_name="unsafe",
            message="use <script> & 'quotes'",
            file_path="SKILL.md",
            line_number=3,
        )
    )

    document = JUnitReporter().render_all([result])
    root = ET.fromstring(document)
    failure = root.find("testsuite/testcase/failure")
    assert failure is not None
    assert failure.attrib["message"] == "use <script> & 'quotes'"
    assert "<script>" in (failure.text or "")
    assert "&lt;script&gt;" in document
    _assert_counts_match(root)


def test_counts_consistency_across_suites() -> None:
    passed = ValidationResult(validator_name="SCHEMA")
    passed.add_success("ok", "ok")
    failed = ValidationResult(validator_name="SECRETS")
    failed.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.HIGH,
            check_name="generic-api-key",
            message="key",
            file_path="src/api.py",
            line_number=8,
        )
    )
    skipped = ValidationResult(validator_name="LICENSE")
    skipped.metadata["skipped"] = True
    skipped.metadata["skip_reason"] = "not applicable"

    root = ET.fromstring(JUnitReporter().render_all([passed, failed, skipped]))
    assert len(root.findall("testsuite")) == 3
    assert int(root.attrib["tests"]) == 3
    assert int(root.attrib["failures"]) == 1
    assert int(root.attrib["skipped"]) == 1
    _assert_counts_match(root)


def test_quality_score_and_threshold_in_system_out() -> None:
    result = ValidationResult(validator_name="Quality Score (4-Dimension Analysis)")
    result.metadata["quality_scores"] = {"overall_score": 88.8, "grade": "B"}
    result.metadata["quality_config"] = {"min_score": 70}
    result.add_success("quality_score", "Score: 88.8/100 (Grade: B)")

    root = ET.fromstring(JUnitReporter().render_all([result]))
    suite_out = root.find("testsuite/system-out")
    assert suite_out is not None
    assert "Score: 88.8/100 (Grade: B)" in (suite_out.text or "")
    assert "Threshold: 70/100" in (suite_out.text or "")
    _assert_counts_match(root)


def test_warning_findings_are_not_failures() -> None:
    result = ValidationResult(validator_name="QUALITY")
    result.add_finding(
        Finding(
            category="QUALITY",
            severity=Severity.MEDIUM,
            check_name="quality_reliability",
            message="No troubleshooting section documented",
            file_path="SKILL.md",
        )
    )
    result.fail_gate("Quality score 71.0/100 is below the required 80/100 in SKILL.md")

    root = ET.fromstring(JUnitReporter().render_all([result]))
    cases = {case.attrib["name"]: case for case in _cases(root)}
    warning = cases["quality_reliability"]
    assert warning.find("failure") is None
    warning_out = warning.find("system-out")
    assert warning_out is not None
    assert "No troubleshooting section documented" in (warning_out.text or "")
    gate = cases["gate_failure"]
    assert gate.find("failure") is not None
    assert "71.0/100" in (gate.find("failure").attrib.get("message") or "")
    assert int(root.attrib["failures"]) == 1
    _assert_counts_match(root)


def test_paths_are_relative_to_scanned_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skill = repo / "skills" / "demo"
    skill.mkdir(parents=True)
    result = ValidationResult(validator_name="QUALITY")
    result.add_finding(
        Finding(
            category="QUALITY",
            severity=Severity.HIGH,
            check_name="spacing",
            message="issue",
            file_path="SKILL.md",
            line_number=2,
        )
    )

    root = ET.fromstring(
        JUnitReporter(workspace_root=repo, scan_root=skill).render_all([result])
    )
    case = root.find("testsuite/testcase")
    assert case is not None
    assert case.attrib["file"] == "skills/demo/SKILL.md"
    failure = case.find("failure")
    assert failure is not None
    assert "skills/demo/SKILL.md:2" in (failure.text or "")


def test_emit_reports_writes_junit_file(tmp_path: Path) -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_success("manifest_found", "Found SKILL.md")
    output_dir = tmp_path / "reports"
    emit_reports(
        [result],
        report_formats=("junit",),
        output_dir=output_dir,
        basename="skillevaluator-quality",
        announce_paths=False,
    )
    path = output_dir / "skillevaluator-quality.junit.xml"
    assert path.is_file()
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    assert root.tag == "testsuites"
    _assert_counts_match(root)


def test_quality_check_cli_writes_junit_xml(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    result = CliRunner().invoke(
        cli,
        ["quality-check", str(_TEXT_ANALYZER), "-r", "junit", "-o", str(reports)],
    )
    assert result.exit_code == 0, result.output
    matches = list(reports.glob("*.junit.xml"))
    assert matches == [reports / "skillevaluator-quality.junit.xml"]
    root = ET.fromstring(matches[0].read_text(encoding="utf-8"))
    assert root.tag == "testsuites"
    assert int(root.attrib["tests"]) >= 1
    _assert_counts_match(root)
    ET.fromstring(matches[0].read_text(encoding="utf-8"))
