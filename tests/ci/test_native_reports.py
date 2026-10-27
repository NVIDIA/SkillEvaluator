# SPDX-License-Identifier: Apache-2.0
"""Native CI rendering preserves retained evidence and fails closed."""

import json
from pathlib import Path

import pytest
from ci.native_reports import render_native_reports
from ci.run_tier1 import _summary

from skillevaluator.models import Finding, ValidationResult
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.reporting.retained import RetainedReportError


def _write_fixture(tmp_path: Path, failed=False):
    result = ValidationResult(validator_name="Schema", validator_description="Validate skill metadata")
    if failed:
        result.add_finding(
            Finding(
                category="SCHEMA",
                severity="high",
                check_name="name",
                message="Bad name </script><img src=x onerror=alert(1)>",
                file_path="/workspace/skills/example/SKILL.md",
                suggestion="Use kebab-case",
            )
        )
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    directory = tmp_path / "skill-001"
    directory.mkdir()
    raw = json.dumps(payload, indent=3) + "\n"
    (directory / "skillevaluator-tier1.json").write_text(raw)
    (directory / "skillevaluator-tier1.html").write_text('<script>alert("ARTIFACT HTML")</script>')
    row = {
        "path": "skills/example",
        "report_dir": "skill-001",
        "exit_code": 1 if failed else 0,
        "passed": not failed,
        **{
            key: payload[key]
            for key in (
                "overall_status",
                "incomplete_scans",
                "total_validators",
                "total_errors",
                "total_warnings",
                "severity_counts",
            )
        },
    }
    return _summary([row]), raw


@pytest.mark.parametrize("failed", [False, True])
def test_native_reports_overwrite_artifact_html_keep_json_and_link_details(tmp_path, failed):
    summary, raw = _write_fixture(tmp_path, failed)
    render_native_reports(tmp_path, summary, {"report_url": "https://example.com/run", "llm_enabled": False})
    native = (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()
    index = (tmp_path / "index.html").read_text()
    assert "ARTIFACT HTML" not in native
    assert 'href="../index.html"' in native
    assert 'href="skillevaluator-tier1.json"' in native
    assert 'href="skill-001/skillevaluator-tier1.html"' in index
    assert (tmp_path / "skill-001/skillevaluator-tier1.json").read_text() == raw
    assert "Content-Security-Policy" in native and "sha256-" in native and "unsafe-eval" not in native
    assert "<img src=x onerror=alert(1)>" not in native
    assert "https://example.com/run/skill-001/skillevaluator-tier1.html" in (tmp_path / "details.md").read_text()


def test_malformed_retained_evidence_is_not_silently_published(tmp_path):
    summary, _ = _write_fixture(tmp_path)
    path = tmp_path / "skill-001/skillevaluator-tier1.json"
    payload = json.loads(path.read_text())
    payload["results"][0]["passed"] = "true"
    path.write_text(json.dumps(payload))
    with pytest.raises(RetainedReportError):
        render_native_reports(tmp_path, summary, {})
    assert not (tmp_path / "index.html").exists()


def test_execution_error_overwrites_stale_html_with_honest_diagnostic(tmp_path):
    summary, _ = _write_fixture(tmp_path)
    summary["skills"][0].update(passed=False, overall_status="error", error="Process failed")
    render_native_reports(tmp_path, _summary(summary["skills"]), {})
    native = (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()
    assert "Report unavailable" in native and "Process failed" in native
    assert "No passing result is claimed" in native and "ARTIFACT HTML" not in native
    assert "Incomplete" in (tmp_path / "index.html").read_text()


def _no_changes_summary():
    summary = _summary([])
    summary.update(
        overall_status="skipped",
        overall_passed=False,
        selection={
            "mode": "changed",
            "base": "a" * 40,
            "head": "b" * 40,
            "discovered": 2,
            "selected": [],
            "deleted": ["skills/removed"],
            "changed_files": 1,
        },
    )
    return summary


def test_changed_selection_without_work_is_neutral_and_preserves_deleted_directories(tmp_path):
    summary = _no_changes_summary()
    raw = json.dumps(summary, indent=2)
    (tmp_path / "summary.json").write_text(raw)
    render_native_reports(tmp_path, summary, {"llm_enabled": True})
    page = (tmp_path / "index.html").read_text()
    assert 'class="status ">No skills changed</span>' in page
    assert "Validation did not run." in page
    assert "Tier 1: Not run (no skills selected)" in page
    assert "LLM: Not run" in page
    assert "Deleted skill directories (not scanned)" in page
    assert "skills/removed" in page
    assert "No skill results were recorded" not in page
    assert (tmp_path / "summary.json").read_text() == raw
    assert not list(tmp_path.glob("skill-*"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda summary: summary.update(overall_passed=True),
        lambda summary: summary["selection"].update(selected=["skills/example"]),
        lambda summary: summary.update(error="Selection failed"),
    ],
)
def test_skipped_marker_with_inconsistent_evidence_cannot_claim_no_changes(tmp_path, mutate):
    summary = _no_changes_summary()
    mutate(summary)
    render_native_reports(tmp_path, summary, {})
    page = (tmp_path / "index.html").read_text()
    assert 'class="status ">No skills changed</span>' not in page
    assert "Validation did not run." not in page
