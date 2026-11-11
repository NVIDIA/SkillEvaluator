# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate behavior for actual SkillSpector missing-reference reports."""

import json
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.reporting import HTMLReporter, JSONReporter, MarkdownReporter, SARIFReporter
from skillevaluator.reporting.retained import retained_results
from skillevaluator.validators.base import Finding, Severity, ValidationResult
from skillevaluator.validators.security import SecurityValidator

FIXTURE = Path(__file__).parents[1] / "fixtures" / "skillspector-2.13.3-reference-caveat"


class ReportText(HTMLParser):
    """Rendered body text, excluding JSON/script payloads that users cannot read."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, _attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def captured_report() -> dict:
    return json.loads((FIXTURE / "report.json").read_text())


def validate_report(report: dict, *, use_llm: bool = False) -> ValidationResult:
    result = ValidationResult(validator_name="Security Scan")
    valid = SecurityValidator._validate_skillspector_report(
        report,
        0,
        use_llm,
        "skillspector-llm" if use_llm else "skillspector",
        result,
        scanned_root=FIXTURE,
    )
    if not valid:
        result.mark_scan_incomplete("skillspector-llm" if use_llm else "skillspector")
    return result


def test_fully_inspected_missing_reference_report_is_a_visible_warning():
    """A generic partial gate must not discard complete bundled-file evidence."""
    report = captured_report()
    result = validate_report(report)

    assert not result.is_incomplete, result.errors
    assert not result.errors
    assert any("Referenced targets were not verified" in warning for warning in result.warnings)
    assert any("SKILL.md:11" in warning and "SKILL.md:12" in warning for warning in result.warnings)
    assert result.metadata["skillspector_reference_caveats"] == {
        "status": "partial",
        "recommendation": "CAUTION",
        "reason_code": "reference_missing",
        "count": 2,
        "fully_inspected_files": 1,
        "missing_targets_verified": False,
    }
    assert report["analysis_completeness"]["is_complete"] is False
    assert report["risk_assessment"]["recommendation"] == "CAUTION"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason_code", "reference_unresolved"),
        ("reason_code", "runtime_limit"),
        ("reason_code", "opaque_content"),
        ("fatal", True),
        ("phase", "cache"),
        ("outcome", "failed"),
        ("path", "missing-script.py"),
        ("start_line", 0),
        ("end_line", 9999),
        ("analyzers", ["static_yara"]),
        ("unknown_limit", True),
    ],
)
def test_other_or_unverifiable_ledger_evidence_still_blocks(field, value):
    report = captured_report()
    report["analysis_completeness"]["ledger_exceptions"][0][field] = value

    result = validate_report(report)

    assert result.is_incomplete or result.errors
    assert "skillspector_reference_caveats" not in result.metadata


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("limitations", ["Analyzer unavailable"]),
        ("coverage_percent", 99),
        ("fully_inspected_files", 0),
        ("partially_inspected_files", 1),
        ("entirely_uninspected_files", 1),
        ("execution_successful", False),
        ("unknown_limit", True),
        ("scope_exclusions", [{"fatal": True}]),
    ],
)
def test_missing_reference_exception_does_not_hide_other_coverage_gaps(field, value):
    report = captured_report()
    report["analysis_completeness"][field] = value

    result = validate_report(report)

    assert result.is_incomplete or result.errors
    assert "skillspector_reference_caveats" not in result.metadata


@pytest.mark.parametrize("mutation", ["absent", "zero_work", "duplicate", "disabled", "partial"])
def test_required_universal_analyzer_must_inspect_every_component(mutation):
    report = captured_report()
    statuses = report["analysis_completeness"]["analyzer_statuses"]
    yara = next(row for row in statuses if row["analyzer_id"] == "static_yara")
    if mutation == "absent":
        statuses.remove(yara)
    elif mutation == "zero_work":
        yara.update(planned_work=0, completed=0)
    elif mutation == "duplicate":
        statuses.append(dict(yara))
    elif mutation == "disabled":
        yara.update(status="disabled", planned_work=0, completed=0, reason_code="disabled_by_configuration")
    else:
        yara.update(status="degraded", completed=0, partial=1)

    result = validate_report(report)

    assert result.is_incomplete or result.errors
    assert "skillspector_reference_caveats" not in result.metadata


def test_analyzer_not_applicable_requires_real_applicability_evidence():
    report = captured_report()
    ast = next(
        row for row in report["analysis_completeness"]["analyzer_statuses"] if row["analyzer_id"] == "behavioral_ast"
    )
    ast.pop("reason_code")

    assert validate_report(report).is_incomplete


@pytest.mark.parametrize("version", ["2.13.2", "3.0.0"])
def test_unverified_scanner_contract_does_not_gain_the_exception(version):
    report = captured_report()
    report["metadata"]["skillspector_version"] = version

    assert validate_report(report).is_incomplete


def test_requested_llm_analysis_is_still_required():
    assert validate_report(captured_report(), use_llm=True).is_incomplete


@pytest.mark.parametrize("field", ["risk_assessment", "components"])
def test_malformed_report_sections_stay_incomplete_without_crashing(field):
    report = captured_report()
    report[field] = None

    assert validate_report(report).is_incomplete


@pytest.mark.parametrize("mutation", ["missing_ledger_match", "ambiguous", "invalid_resolved_target"])
def test_reference_inventory_must_agree_with_the_missing_only_ledger(mutation):
    report = captured_report()
    references = report["analysis_completeness"]["references"]
    if mutation == "missing_ledger_match":
        references.pop()
    elif mutation == "ambiguous":
        references[0]["status"] = "ambiguous"
    else:
        references[0].update(status="resolved", disposition="analyzed", target_path={})

    assert validate_report(report).is_incomplete


def test_claiming_safe_or_hiding_an_inventory_entry_still_blocks():
    report = captured_report()
    report["risk_assessment"]["recommendation"] = "SAFE"
    result = validate_report(report)
    assert result.is_incomplete
    assert "skillspector_reference_caveats" not in result.metadata

    report = captured_report()
    report["components"] = []
    assert validate_report(report).is_incomplete


def test_reference_warning_and_original_status_survive_report_serialization():
    result = validate_report(captured_report())

    payload = json.loads(JSONReporter(include_timestamp=False).render(result))
    assert payload["skillspector_reference_caveats"]["status"] == "partial"
    assert payload["skillspector_reference_caveats"]["missing_targets_verified"] is False
    html = HTMLReporter().render(result)
    assert "Referenced targets were not verified" in html
    assert "SKILL.md:11" in html
    reloaded = retained_results(json.loads(JSONReporter(include_timestamp=False).render_all([result])))[0]
    assert reloaded.metadata["skillspector_reference_caveats"] == result.metadata["skillspector_reference_caveats"]
    assert "Referenced targets were not verified" in HTMLReporter().render(reloaded)


def test_reference_caveat_is_visible_in_actual_ci_render_all_outputs(tmp_path):
    from ci.secret_redaction import sanitized_outputs

    result = validate_report(captured_report())
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    restored = retained_results(payload)
    (tmp_path / "skillevaluator-tier1.json").write_text(json.dumps(payload))
    for extension, reporter in (
        ("html", HTMLReporter()),
        ("md", MarkdownReporter()),
        ("sarif.json", SARIFReporter()),
    ):
        (tmp_path / f"skillevaluator-tier1.{extension}").write_text(reporter.render_all(restored))
    sanitized = sanitized_outputs(tmp_path, set())
    parsed = ReportText()
    parsed.feed(sanitized["skillevaluator-tier1.html"].decode())

    assert "Referenced targets were not verified" in " ".join(parsed.parts)
    assert "SKILL.md:11" in " ".join(parsed.parts)
    assert b"Referenced targets were not verified" in sanitized["skillevaluator-tier1.md"]
    sarif = json.loads(sanitized["skillevaluator-tier1.sarif.json"])
    findings = sarif["runs"][0]["results"]
    assert len(findings) == 1
    assert "Referenced targets were not verified" in findings[0]["message"]["text"]
    assert findings[0]["level"] == "note"
    assert result.summary.warnings == 1
    assert result.passed


def test_reference_caveat_is_not_counted_as_an_active_baseline_finding(tmp_path):
    from skillevaluator.validators.security import _inspect_skillspector_baseline

    result = validate_report(captured_report())
    SecurityValidator()._process_skillspector_cli_result(captured_report(), result)
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text("version: 2\nrules: []\nfingerprints: []\n")
    baseline, error = _inspect_skillspector_baseline(baseline_path)
    assert error is None

    SecurityValidator._report_skillspector_baseline(result, baseline, static_suppressed=0)

    assert baseline["active_findings"] == 0
    assert baseline["active_blocking_findings"] == 0
    assert result.passed


@pytest.mark.parametrize("with_issue", [False, True])
def test_reference_evidence_advisory_is_not_sent_for_llm_finding_verification(tmp_path, with_issue):
    result = validate_report(captured_report())
    advisory = result.findings[0]
    if with_issue:
        issue = Finding(
            category="SECURITY",
            severity=Severity.HIGH,
            check_name="Unreviewed execution",
            message="A genuine issue still needs verification.",
            file_path="SKILL.md",
        )
        result.add_finding(issue)

    with patch("skillevaluator.inference.FindingVerifier") as verifier_cls:
        verifier_cls.return_value.verify.return_value = {
            0: {"verdict": "true_positive", "confidence": "high", "rationale": "Confirmed issue."}
        }
        SecurityValidator(verify_llm=True)._verify_findings_with_llm(result, tmp_path)

    if with_issue:
        verifier_cls.return_value.verify.assert_called_once_with([issue], tmp_path)
        assert issue.metadata["llm_verdict"] == "true_positive"
        assert result.errors
    else:
        verifier_cls.assert_not_called()
        assert result.passed
    assert "llm_verdict" not in advisory.metadata
    assert not result.is_incomplete


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "complete"),
        ("recommendation", "SAFE"),
        ("missing_targets_verified", True),
        ("count", True),
        ("fully_inspected_files", 0),
        ("untrusted_message", "arbitrary text"),
    ],
)
def test_retained_caveat_metadata_cannot_claim_unverified_targets_are_safe(field, value):
    result = validate_report(captured_report())
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    payload["results"][0]["skillspector_reference_caveats"][field] = value

    with pytest.raises(ValueError, match="Invalid skillspector_reference_caveats"):
        retained_results(payload)


def test_existing_security_findings_remain_blocking_with_reference_caveats():
    path = FIXTURE.parent / "skillspector-2.13.3-passwd-mount" / "separate-read.json"
    report = json.loads(path.read_text())
    report["analysis_completeness"].update(
        status="partial",
        is_complete=False,
        ledger_exceptions=[
            {
                "outcome": "partial",
                "phase": "reference_resolution",
                "reason_code": "reference_missing",
                "message": "Unverified reference.",
                "path": "SKILL.md",
                "start_line": 1,
                "end_line": 1,
                "fatal": False,
            }
        ],
    )
    report["analysis_completeness"]["references"].append(
        {
            "source_path": "SKILL.md",
            "line": 1,
            "column": 1,
            "evidence": "Unverified reference.",
            "target_path": None,
            "status": "missing",
            "disposition": "partial",
            "reference_kind": "quoted_or_code",
        }
    )
    result = validate_report(report)
    assert not result.is_incomplete

    SecurityValidator()._process_skillspector_cli_result(report, result)

    assert result.errors
    assert not result.passed
    assert any(finding.severity.value == "high" for finding in result.findings)
