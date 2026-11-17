# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JUnit XML reporter."""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.junit_reporter import JUnitReporter, junit_target_identity
from skillevaluator.tier1.commands import emit_reports

_TEXT_ANALYZER = Path(__file__).resolve().parents[2] / "src/skillevaluator/tier3/reference_skills/text-analyzer"


def _cases(root: ET.Element) -> list[ET.Element]:
    return list(root.iter("testcase"))


def _count_children(cases: list[ET.Element], tag: str) -> int:
    return sum(1 for case in cases if case.find(tag) is not None)


def _assert_counts_match(root: ET.Element) -> None:
    cases = _cases(root)
    failures = _count_children(cases, "failure")
    errors = _count_children(cases, "error")
    skipped = _count_children(cases, "skipped")
    assert int(root.attrib["tests"]) == len(cases)
    assert int(root.attrib["failures"]) == failures
    assert int(root.attrib["errors"]) == errors
    assert int(root.attrib["skipped"]) == skipped
    for suite in root.findall("testsuite"):
        suite_cases = suite.findall("testcase")
        assert int(suite.attrib["tests"]) == len(suite_cases)
        assert int(suite.attrib["failures"]) == _count_children(suite_cases, "failure")
        assert int(suite.attrib["errors"]) == _count_children(suite_cases, "error")
        assert int(suite.attrib["skipped"]) == _count_children(suite_cases, "skipped")


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
    assert case.attrib["name"] == "aws-access-key-id (config/settings.py:15)"
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
    assert all(case.find("skipped") is None for case in incomplete_cases)
    incomplete_case = next(case for case in incomplete_cases if case.attrib["name"] == "incomplete/bandit")
    error_el = incomplete_case.find("error")
    assert error_el is not None
    assert error_el.attrib["type"] == "incomplete"
    assert "scan did not complete" in (error_el.attrib.get("message") or "")
    execution = next(case for case in incomplete_cases if case.attrib["name"] == "error").find("error")
    assert execution is not None
    assert execution.attrib["type"] == "execution"
    assert int(incomplete_suite.attrib["failures"]) == 0
    assert int(incomplete_suite.attrib["errors"]) == 2
    assert int(root.attrib["errors"]) == 2
    assert int(root.attrib["skipped"]) == 1
    passing = [
        case
        for case in _cases(root)
        if case.find("failure") is None and case.find("error") is None and case.find("skipped") is None
    ]
    assert passing == []
    _assert_counts_match(root)


def test_incomplete_scan_does_not_skip_other_cases() -> None:
    result = ValidationResult(validator_name="Security Scan", validator_description="Security")
    for name in ("check_a", "check_b", "check_c"):
        result.add_success(name, f"{name} passed")
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
    cases = {case.attrib["name"]: case for case in _cases(root)}
    for name in ("check_a", "check_b", "check_c", "skillspector_timeout (SKILL.md)"):
        assert cases[name].find("skipped") is None
        assert cases[name].find("failure") is None
    incomplete = cases["incomplete/skillspector"]
    assert incomplete.find("skipped") is None
    error = incomplete.find("error")
    assert error is not None
    assert error.attrib["type"] == "incomplete"
    assert error.attrib["message"] == "skillspector timed out after 300 seconds"
    assert int(root.attrib["tests"]) == 5
    assert int(root.attrib["failures"]) == 0
    assert int(root.attrib["errors"]) == 1
    assert int(root.attrib["skipped"]) == 0
    _assert_counts_match(root)


def test_incomplete_scan_is_an_error_with_the_scanner_reason() -> None:
    result = ValidationResult(validator_name="Security Scan", validator_description="Security")
    result.mark_scan_incomplete("skillspector")
    result.add_warning("skillspector not installed; install it to run the security scan")

    root = ET.fromstring(JUnitReporter().render_all([result]))
    (case,) = _cases(root)
    assert case.attrib["name"] == "incomplete/skillspector"
    assert case.find("skipped") is None
    error = case.find("error")
    assert error is not None
    assert error.attrib["type"] == "incomplete"
    assert error.attrib["message"] == "skillspector not installed; install it to run the security scan"
    assert int(root.attrib["failures"]) + int(root.attrib["errors"]) >= 1
    assert int(root.attrib["errors"]) == 1
    assert int(root.attrib["skipped"]) == 0
    _assert_counts_match(root)


def test_incomplete_scan_without_reason_uses_generic_message() -> None:
    result = ValidationResult(validator_name="Audit", validator_description="Dependency audit")
    result.mark_scan_incomplete("pip-audit")
    result.add_finding(
        Finding(
            category="DEPENDENCIES",
            severity=Severity.INFO,
            check_name="pip_audit",
            message="Dependencies could not be audited",
            file_path="requirements.txt",
        )
    )

    root = ET.fromstring(JUnitReporter().render_all([result]))
    error = next(case for case in _cases(root) if case.attrib["name"] == "incomplete/pip-audit").find("error")
    assert error is not None
    assert error.attrib["message"] == "pip-audit scan did not complete"
    assert int(root.attrib["errors"]) == 1
    assert int(root.attrib["skipped"]) == 0
    _assert_counts_match(root)


def test_execution_errors_beside_an_incomplete_scan_are_junit_errors() -> None:
    result = ValidationResult(validator_name="Code Risk", validator_description="Code risk")
    result.add_error("Semgrep did not return valid JSON; scan did not complete")
    result.mark_scan_incomplete("semgrep")
    result.fail_gate("Code risk score 40 is below the minimum 70")
    result.add_finding(
        Finding(
            category="BANDIT",
            severity=Severity.HIGH,
            check_name="B602",
            message="subprocess call with shell=True",
            file_path="scripts/run.py",
            line_number=3,
        )
    )

    root = ET.fromstring(JUnitReporter().render_all([result]))
    cases = {case.attrib["name"]: case for case in _cases(root)}
    execution = cases["error"].find("error")
    assert execution is not None
    assert execution.attrib["type"] == "execution"
    assert execution.attrib["message"] == "Semgrep did not return valid JSON; scan did not complete"
    assert cases["error"].find("failure") is None
    gate = cases["gate_failure"].find("failure")
    assert gate is not None
    assert gate.attrib["message"] == "Code risk score 40 is below the minimum 70"
    assert cases["B602 (scripts/run.py:3)"].find("failure") is not None
    assert cases["incomplete/semgrep"].find("error").attrib["type"] == "incomplete"
    assert int(root.attrib["failures"]) == 2
    assert int(root.attrib["errors"]) == 2
    _assert_counts_match(root)


def test_independent_error_without_an_incomplete_scan_stays_a_failure() -> None:
    result = ValidationResult(validator_name="SCHEMA", validator_description="Schema checks")
    result.add_error("No SKILL.md found in demo")

    root = ET.fromstring(JUnitReporter().render_all([result]))
    (case,) = _cases(root)
    assert case.attrib["name"] == "error"
    assert case.find("error") is None
    assert case.find("failure") is not None
    assert int(root.attrib["failures"]) == 1
    assert int(root.attrib["errors"]) == 0
    _assert_counts_match(root)


def test_advisory_execution_errors_are_not_errors() -> None:
    result = ValidationResult(validator_name="CONTENT_DEDUP")
    result.add_error("Embedding provider error: connection refused")
    result.mark_scan_incomplete("embedding-provider")
    result.metadata["gating"] = {"tier": 2, "blocking": False}

    root = ET.fromstring(JUnitReporter().render_all([result]))
    cases = {case.attrib["name"]: case for case in _cases(root)}
    assert all(case.find("failure") is None and case.find("error") is None for case in cases.values())
    out = cases["error"].find("system-out")
    assert out is not None
    assert out.text == "Advisory, does not gate: Embedding provider error: connection refused"
    assert cases["error"].find("skipped") is None
    assert cases["incomplete/embedding-provider"].find("skipped") is not None
    assert int(root.attrib["errors"]) == 0
    assert int(root.attrib["skipped"]) == 1
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
    warning = cases["quality_reliability (SKILL.md)"]
    assert warning.find("failure") is None
    warning_out = warning.find("system-out")
    assert warning_out is not None
    assert "No troubleshooting section documented" in (warning_out.text or "")
    gate = cases["gate_failure"]
    assert gate.find("failure") is not None
    assert "71.0/100" in (gate.find("failure").attrib.get("message") or "")
    assert int(root.attrib["failures"]) == 1
    _assert_counts_match(root)


def test_medium_finding_recorded_as_blocking_is_a_failure() -> None:
    result = ValidationResult(validator_name="Code Risk")
    result.add_structured_finding(
        Finding(
            category="BANDIT",
            severity=Severity.HIGH,
            check_name="B602:subprocess_popen_with_shell_equals_true",
            message="subprocess call with shell=True",
            file_path="scripts/run.py",
            line_number=3,
        ),
        is_error=True,
    )
    result.add_structured_finding(
        Finding(
            category="BANDIT",
            severity=Severity.MEDIUM,
            check_name="B301:blacklist",
            message="Pickle can be unsafe when used to deserialize untrusted data",
            file_path="scripts/load.py",
            line_number=7,
        ),
        is_error=True,
    )
    result.add_structured_finding(
        Finding(
            category="BANDIT",
            severity=Severity.MEDIUM,
            check_name="B108:hardcoded_tmp_directory",
            message="Probable insecure usage of temp file",
            file_path="scripts/tmp.py",
            line_number=2,
        ),
        is_error=False,
    )

    root = ET.fromstring(JUnitReporter().render_all([result]))
    cases = {case.attrib["name"]: case for case in _cases(root)}
    pickle_case = cases["B301:blacklist (scripts/load.py:7)"]
    failure = pickle_case.find("failure")
    assert failure is not None
    assert "scripts/load.py:7" in (failure.text or "")
    assert pickle_case.attrib["file"] == "scripts/load.py"
    assert pickle_case.attrib["line"] == "7"
    assert cases["B108:hardcoded_tmp_directory (scripts/tmp.py:2)"].find("failure") is None
    assert "error" not in cases
    assert int(root.attrib["failures"]) == 2
    _assert_counts_match(root)


def _scored_tier3(verdict: str, *, blocking: bool) -> ValidationResult:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = {"execution_status": "succeeded", "verdict": verdict, "overall_score": 41.0}
    result.add_success("agent_eval", f"Tier 3 evaluation complete: verdict {verdict.upper()}; best agent demo")
    result.passed = True
    result.metadata["gating"] = {"tier": 3, "blocking": blocking}
    return result


def test_blocking_tier3_verdict_that_fails_the_gate_is_a_failure() -> None:
    for verdict in ("fail", "neutral"):
        root = ET.fromstring(JUnitReporter().render_all([_scored_tier3(verdict, blocking=True)]))
        cases = {case.attrib["name"]: case for case in _cases(root)}
        assert cases["agent_eval"].find("failure") is None
        failure = cases["gate_failure"].find("failure")
        assert failure is not None
        assert failure.attrib["message"] == f"Tier 3 verdict {verdict.upper()} does not pass the required gate"
        assert int(root.attrib["failures"]) == 1
        _assert_counts_match(root)


def test_advisory_tier3_verdict_is_not_a_failure() -> None:
    root = ET.fromstring(JUnitReporter().render_all([_scored_tier3("fail", blocking=False)]))
    assert all(case.find("failure") is None for case in _cases(root))
    assert int(root.attrib["failures"]) == 0
    _assert_counts_match(root)


def test_blocking_tier3_that_could_not_run_reports_the_reason() -> None:
    from skillevaluator.evaluation.tier3_report import advisory_skip_result

    result = advisory_skip_result("ANTHROPIC_API_KEY is not set; Tier 3 did not run")
    result.metadata["gating"] = {"tier": 3, "blocking": True}

    root = ET.fromstring(JUnitReporter().render_all([result]))
    failures = [case.find("failure") for case in _cases(root) if case.find("failure") is not None]
    assert len(failures) == 1
    assert failures[0].attrib["message"] == "ANTHROPIC_API_KEY is not set; Tier 3 did not run"
    _assert_counts_match(root)

    result.metadata["gating"] = {"tier": 3, "blocking": False}
    root = ET.fromstring(JUnitReporter().render_all([result]))
    assert all(case.find("failure") is None for case in _cases(root))
    skipped = root.find("testsuite/testcase/skipped")
    assert skipped is not None
    assert skipped.attrib["message"] == "ANTHROPIC_API_KEY is not set; Tier 3 did not run"


def test_blocking_skip_that_fails_run_status_is_a_failure() -> None:
    from skillevaluator.evaluation.tier3_report import advisory_skip_result
    from skillevaluator.reporting.run_status import Status, result_status

    result = advisory_skip_result("Tier 3 skipped with --skip")
    result.passed = True
    result.metadata["skipped"] = True
    result.metadata["skip_reason"] = "Tier 3 skipped with --skip"
    result.metadata["gating"] = {"tier": 3, "blocking": True}
    assert result_status(result) == Status.FAILED

    root = ET.fromstring(JUnitReporter().render_all([result]))
    cases = _cases(root)
    assert all(case.find("skipped") is None for case in cases)
    failure = root.find("testsuite/testcase[@name='gate_failure']/failure")
    assert failure is not None
    assert failure.attrib["message"] == "Tier 3 skipped with --skip"
    assert int(root.attrib["failures"]) == 1
    assert int(root.attrib["skipped"]) == 0
    _assert_counts_match(root)

    result.metadata["gating"] = {"tier": 3, "blocking": False}
    assert result_status(result) == Status.SKIPPED
    root = ET.fromstring(JUnitReporter().render_all([result]))
    skipped = root.find("testsuite/testcase/skipped")
    assert skipped is not None
    assert skipped.attrib["message"] == "Tier 3 skipped with --skip"
    assert int(root.attrib["failures"]) == 0
    _assert_counts_match(root)


def test_advisory_tier2_findings_and_errors_are_not_failures() -> None:
    result = ValidationResult(validator_name="CONTENT_DEDUP")
    result.add_finding(
        Finding(
            category="CONTENT_DEDUP",
            severity=Severity.HIGH,
            check_name="duplicate_section",
            message="Section duplicates references/guide.md",
            file_path="SKILL.md",
            line_number=12,
        )
    )
    result.add_error("Similarity model returned no scores")
    result.metadata["gating"] = {"tier": 2, "blocking": False}

    root = ET.fromstring(JUnitReporter().render_all([result]))
    cases = {case.attrib["name"]: case for case in _cases(root)}
    assert all(case.find("failure") is None for case in cases.values())
    finding_out = cases["duplicate_section (SKILL.md:12)"].find("system-out")
    assert finding_out is not None
    assert (finding_out.text or "").startswith("Advisory, does not gate: SKILL.md:12: Section duplicates")
    assert int(root.attrib["failures"]) == 0
    _assert_counts_match(root)

    result.metadata["gating"] = {"tier": 2, "blocking": True}
    root = ET.fromstring(JUnitReporter().render_all([result]))
    assert int(root.attrib["failures"]) == 2
    assert "gate_failure" not in {case.attrib["name"] for case in _cases(root)}
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

    root = ET.fromstring(JUnitReporter(workspace_root=repo, scan_root=skill).render_all([result]))
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


def test_single_check_command_paths_are_relative_to_git_root(tmp_path: Path, monkeypatch) -> None:
    """A single-check command reports the same repository-relative file as ``validate``, in JUnit and SARIF."""
    repo = tmp_path / "repo"
    skill = repo / "skills" / "demo"
    shutil.copytree(_TEXT_ANALYZER, skill)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    reports = tmp_path / "reports"

    def lint(_path: Path) -> list[ValidationResult]:
        result = ValidationResult(validator_name="SCRIPT_LINT")
        result.add_finding(
            Finding(
                category="SCRIPT_LINT",
                severity=Severity.HIGH,
                check_name="script_lint",
                message="Unsafe call",
                file_path="SKILL.md",
                line_number=2,
            )
        )
        return [result]

    monkeypatch.setattr("skillevaluator.cli.run_lint_scripts", lint)

    CliRunner().invoke(cli, ["lint-scripts", str(skill), "-r", "junit", "-r", "sarif", "-o", str(reports)])

    (junit,) = reports.glob("*.junit.xml")
    files = {case.attrib.get("file") for case in _cases(ET.fromstring(junit.read_text(encoding="utf-8")))}
    assert "skills/demo/SKILL.md" in files
    (sarif,) = reports.glob("*.sarif.json")
    sarif_text = sarif.read_text(encoding="utf-8")
    assert '"uri": "skills/demo/SKILL.md"' in sarif_text


def test_tier1_workflow_with_a_relative_path_reports_git_root_relative_files(tmp_path: Path, monkeypatch) -> None:
    """``tier1 skills/demo`` from the repository root names ``skills/demo/SKILL.md`` once, in JUnit and SARIF."""
    repo = tmp_path / "repo"
    skill = repo / "skills" / "demo"
    shutil.copytree(_TEXT_ANALYZER, skill)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    reports = tmp_path / "reports"

    def tier1(skill_path: Path, **_: object) -> list[ValidationResult]:
        # Validators build finding paths from the root they are handed.
        result = ValidationResult(validator_name="QUALITY")
        result.add_finding(
            Finding(
                category="QUALITY",
                severity=Severity.HIGH,
                check_name="quality_correctness",
                message="issue",
                file_path=str(skill_path / "SKILL.md"),
                line_number=2,
            )
        )
        return [result]

    monkeypatch.setattr("skillevaluator.tier_workflows.run_tier1_workflow", tier1)
    monkeypatch.chdir(repo)

    CliRunner().invoke(cli, ["tier1", "skills/demo", "-r", "junit", "-r", "sarif", "-o", str(reports)])

    (junit,) = reports.glob("*.junit.xml")
    files = {case.attrib.get("file") for case in _cases(ET.fromstring(junit.read_text(encoding="utf-8")))}
    assert files == {"skills/demo/SKILL.md"}
    (sarif,) = reports.glob("*.sarif.json")
    assert '"uri": "skills/demo/SKILL.md"' in sarif.read_text(encoding="utf-8")


def test_target_identity_qualifies_suite_and_classname() -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_success("manifest_exists", "Found SKILL.md")

    root = ET.fromstring(JUnitReporter(target_identity="group-a/demo").render_all([result]))
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.attrib["name"] == "group-a/demo::SCHEMA"
    case = suite.find("testcase")
    assert case is not None
    assert case.attrib["classname"] == "group-a/demo::SCHEMA"
    assert case.attrib["name"] == "manifest_exists"


def test_target_identity_prefers_repository_then_catalog_then_name(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skill = repo / "skills" / "group-a" / "demo"
    skill.mkdir(parents=True)

    assert junit_target_identity(skill, repo, repo / "skills") == "skills/group-a/demo"
    # A non-Git catalog validates each skill with the skill itself as its root.
    assert junit_target_identity(skill, skill, repo / "skills") == "group-a/demo"
    assert junit_target_identity(skill, skill) == "demo"
    assert junit_target_identity(None) == ""


def test_catalog_with_repeated_leaf_names_keeps_identities_distinct(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    for group in ("group-a", "group-b"):
        shutil.copytree(_TEXT_ANALYZER, catalog / group / "demo")
    reports = tmp_path / "reports"

    CliRunner().invoke(
        cli,
        [
            "validate",
            str(catalog),
            "--tiers",
            "1",
            "--checks",
            "schema",
            "--workers",
            "1",
            "--no-autopilot",
            "-r",
            "junit",
            "-o",
            str(reports),
        ],
    )

    files = sorted(reports.rglob("*.junit.xml"))
    assert [path.relative_to(reports).parent.as_posix() for path in files] == ["group-a/demo", "group-b/demo"]
    identities: list[tuple[str, str]] = []
    classnames: set[str] = set()
    for path in files:
        for case in _cases(ET.fromstring(path.read_text(encoding="utf-8"))):
            identities.append((case.attrib["classname"], case.attrib["name"]))
            classnames.add(case.attrib["classname"].split("::", 1)[0])
    assert identities
    assert len(identities) == len(set(identities))
    assert classnames == {"group-a/demo", "group-b/demo"}


def _findings_result(order: list[tuple[str, str, int | None]]) -> ValidationResult:
    result = ValidationResult(validator_name="QUALITY", validator_description="Quality")
    for check_name, file_path, line in order:
        result.add_finding(
            Finding(
                category="QUALITY",
                severity=Severity.LOW,
                check_name=check_name,
                message=f"{check_name} in {file_path}",
                file_path=file_path,
                line_number=line,
            )
        )
    return result


def test_finding_case_names_do_not_depend_on_finding_order() -> None:
    findings = [
        ("quality_correctness", "SKILL.md", 12),
        ("quality_correctness", "SKILL.md", 40),
        ("quality_correctness", "references/api.md", None),
    ]

    def names(order: list[tuple[str, str, int | None]]) -> set[str]:
        root = ET.fromstring(JUnitReporter().render_all([_findings_result(order)]))
        return {case.attrib["name"] for case in _cases(root)}

    first = names(findings)
    assert first == {
        "quality_correctness (SKILL.md:12)",
        "quality_correctness (SKILL.md:40)",
        "quality_correctness (references/api.md)",
    }
    assert names(list(reversed(findings))) == first
    added = names([("quality_correctness", "SKILL.md", 5), *findings])
    assert first <= added
