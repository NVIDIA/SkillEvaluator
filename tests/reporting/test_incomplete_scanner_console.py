# SPDX-License-Identifier: Apache-2.0
"""Incomplete scanner evidence must not be diagnosed as a missing installation."""

import pytest

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting.console_ui import summarize_tier1


def _text(rows):
    return "\n".join("".join(chunk for chunk, _ in row.segments) for row in rows)


@pytest.mark.parametrize(
    ("diagnostic", "gate", "runtime_version"),
    [
        (
            "skillspector JSON field 'analysis_completeness.analyzer_statuses' has a status that contradicts "
            "its work accounting; security scan did not complete",
            False,
            None,
        ),
        (
            "skillspector reports incomplete analysis (status 'partial'): SKILL.md:4 "
            "[reference_resolution; reference_unresolved]; and 11 more. Missing references require "
            "a standalone target with complete, matching reference and scan evidence.",
            True,
            "2.11.2",
        ),
    ],
)
def test_partial_analysis_keeps_its_diagnostic_and_version(diagnostic, gate, runtime_version):
    # An incomplete marker is not evidence that the installed scanner never ran.
    result = ValidationResult(validator_name="Security Scan")
    result.record_scanner("skillspector", None if runtime_version else "2.11.2")
    if runtime_version:
        result.metadata["skillspector_version"] = runtime_version
    result.mark_scan_incomplete("skillspector")
    if gate:
        result.fail_gate(diagnostic)
    else:
        result.add_error(diagnostic)
    result.add_finding(Finding("SECURITY", Severity.MEDIUM, "rug-pull", "Advisory issue", "SKILL.md"))

    passed, rows = summarize_tier1([result], lineup=["security"])
    text = _text(rows)

    assert not passed
    assert diagnostic in text
    assert "SkillSpector 2.11.2, incomplete" in text
    assert "0/1 checks passed  ·  1 incomplete" in text
    assert "security incomplete:" in text
    assert "rerun" in text
    assert "not installed" not in text
    assert "didn't run" not in text
    assert "--skip" not in text


@pytest.mark.parametrize("diagnostic", [None, "skillspector timed out after 300 seconds; partial results only"])
def test_unrecorded_scanner_version_does_not_imply_an_installation_failure(diagnostic):
    result = ValidationResult(validator_name="Security Scan")
    result.mark_scan_incomplete("skillspector")
    if diagnostic:
        result.add_error(diagnostic)

    _passed, rows = summarize_tier1([result])
    text = _text(rows)

    assert "SkillSpector incomplete" in text
    assert (diagnostic or "skillspector did not produce complete evidence") in text
    assert "not installed" not in text
    assert "didn't run" not in text
    assert "--skip" not in text


def test_partial_analysis_and_missing_scanner_keep_their_separate_recoveries():
    security = ValidationResult(validator_name="Security Scan")
    security.record_scanner("skillspector", "2.11.2")
    security.mark_scan_incomplete("skillspector")
    security.fail_gate("skillspector reports incomplete analysis: reference_unresolved at SKILL.md:4")
    secrets = ValidationResult(validator_name="Secrets Detection")
    secrets.mark_scan_incomplete("gitleaks")
    secrets.add_warning("couldn't download Gitleaks 8.30.0 (connection refused). Install with: brew install gitleaks")

    _passed, rows = summarize_tier1([security, secrets])
    text = _text(rows)

    assert "security incomplete: skillspector reports incomplete analysis: reference_unresolved at SKILL.md:4" in text
    assert "secrets didn't run: couldn't download Gitleaks 8.30.0 (connection refused)" in text
    assert "check network access to github.com, then rerun" in text
    assert "brew install gitleaks" in text
    assert "--skip security" not in text


def test_an_advisory_about_a_missing_package_cannot_become_a_scanner_installation_error():
    result = ValidationResult(validator_name="Security Scan")
    result.record_scanner("skillspector", "2.11.2")
    result.mark_scan_incomplete("skillspector")
    result.add_error("skillspector reports partial analysis: reference_unresolved")
    result.add_finding(Finding("SECURITY", Severity.MEDIUM, "dependency", "Optional client not installed", "run.py"))

    _passed, rows = summarize_tier1([result])
    text = _text(rows)

    assert "security incomplete: skillspector reports partial analysis: reference_unresolved" in text
    assert "skillspector not installed" not in text
    assert "--skip security" not in text


@pytest.mark.parametrize("reverse", [False, True])
def test_completed_member_cannot_hide_an_incomplete_scanner_run(reverse):
    partial = ValidationResult(validator_name="Security Scan")
    partial.record_scanner("skillspector", None)
    partial.mark_scan_incomplete("skillspector")
    partial.add_error("skillspector returned malformed analysis accounting")
    complete = ValidationResult(validator_name="Security Scan")
    complete.record_scanner("skillspector", "2.11.2")
    results = [partial, complete]

    _passed, rows = summarize_tier1(results[::-1] if reverse else results)
    scanner = next(row for row in rows if row.label == "scanners")

    assert _text([scanner]) == "SkillSpector 2.11.2, incomplete"


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("blocking", [False, True])
def test_final_verdict_distinguishes_incomplete_scans_from_known_failures(tmp_path, count, blocking):
    import io

    from rich.console import Console

    from skillevaluator.cli import _finish_pipeline_view
    from skillevaluator.reporting.console_ui import ValidateView

    results = []
    for _ in range(count):
        result = ValidationResult(validator_name="Security Scan", metadata={"gating": {"tier": 1, "blocking": True}})
        result.record_scanner("skillspector", "2.11.2")
        result.mark_scan_incomplete("skillspector")
        result.fail_gate("skillspector reports incomplete analysis: reference_unresolved")
        results.append(result)
    if blocking:
        results[0].add_finding(Finding("SECURITY", Severity.HIGH, "known-risk", "Known risk", "run.py"))
    stream = io.StringIO()
    view = ValidateView("demo", [(1, "Static", "static")], console=Console(file=stream, width=180, color_system=None))

    _finish_pipeline_view(
        view,
        tier_gate_results=results,
        tier3_result=None,
        gate_failed=True,
        output_dir=tmp_path,
        basename="partial",
        report_formats=(),
        target_path=tmp_path,
        agent_eval=False,
    )
    text = stream.getvalue()

    if count == 1:
        assert f"Security Scan {'failed' if blocking else 'incomplete'} in Tier 1" in text
    else:
        assert ("Tier 1: 1 failed, 1 incomplete" if blocking else "Tier 1: 2 incomplete") in text
    assert "didn't run" not in text
