# SPDX-License-Identifier: Apache-2.0
"""Native CI rendering preserves retained evidence and fails closed."""

import json
import shutil
from pathlib import Path

import pytest
from ci.native_reports import render_native_reports
from ci.run_tier1 import _summary

from skillevaluator.models import Finding, ValidationResult
from skillevaluator.reporting.json_reporter import JSONReporter


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
    render_native_reports(tmp_path, summary, {})
    assert "Incomplete" in (tmp_path / "index.html").read_text()
    assert "Report unavailable" in (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()
    assert "INCOMPLETE" in (tmp_path / "summary.md").read_text()
    assert path.read_text() == json.dumps(payload)


def test_execution_error_overwrites_stale_html_with_honest_diagnostic(tmp_path):
    summary, _ = _write_fixture(tmp_path)
    summary["skills"][0].update(passed=False, overall_status="error", error="Process failed")
    render_native_reports(tmp_path, _summary(summary["skills"]), {})
    native = (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()
    assert "Report unavailable" in native and "Process failed" in native
    assert "No passing result is claimed" in native and "ARTIFACT HTML" not in native
    assert "Incomplete" in (tmp_path / "index.html").read_text()


def test_published_execution_error_without_raw_json_gets_a_diagnostic_page(tmp_path):
    summary, _ = _write_fixture(tmp_path)
    shutil.rmtree(tmp_path / "skill-001")
    row = summary["skills"][0]
    row.update(passed=False, overall_status="error", error="Report evidence rejected")
    errors = render_native_reports(tmp_path, _summary([row]), {})
    assert errors == {"skill-001": "Report evidence rejected"}
    native = (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()
    assert "Report unavailable" in native and "Report evidence rejected" in native
    assert not (tmp_path / "skill-001/skillevaluator-tier1.json").exists()
    assert "Report evidence rejected" in (tmp_path / "pr-comment.md").read_text()


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


@pytest.mark.parametrize("path", ["skills/" + "nested/" * 50 + "example", "."])
def test_long_display_paths_and_repository_root_keep_native_reports(tmp_path, path):
    summary, _ = _write_fixture(tmp_path)
    summary["skills"][0]["path"] = path
    render_native_reports(tmp_path, summary, {})
    html = (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()
    assert "Report unavailable" not in html
    assert path in html
    assert 'class="status pass">Passed</span>' in (tmp_path / "index.html").read_text()


def test_one_unrenderable_skill_does_not_erase_other_reports(tmp_path):
    summary, _ = _write_fixture(tmp_path)
    shutil.copytree(tmp_path / "skill-001", tmp_path / "skill-002")
    second = {**summary["skills"][0], "report_dir": "skill-002", "path": "skills/second"}
    summary["skills"][0]["path"] = "skills/control\tpath"
    summary = _summary([summary["skills"][0], second])
    render_native_reports(tmp_path, summary, {})
    assert "Report unavailable" in (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()
    assert "Report unavailable" not in (tmp_path / "skill-002/skillevaluator-tier1.html").read_text()
    for filename in ("summary.md", "details.md", "pr-comment.md"):
        contents = (tmp_path / filename).read_text()
        assert "2 skills · 1 passed · 0 failed · 1 incomplete" in contents
        assert "Invalid explicit skill display path" in contents
    assert "Incomplete" in (tmp_path / "index.html").read_text()


@pytest.mark.parametrize(
    "injection",
    [
        "<script>alert('unexpected')</script>",
        '<img src="x" onerror="alert(1)">',
        '<script src="https://evil.example/a.js"></script>',
    ],
)
def test_csp_does_not_trust_code_found_in_final_render(tmp_path, injection):
    from ci.native_reports import constrain_native_html

    summary, _ = _write_fixture(tmp_path)
    render_native_reports(tmp_path, summary, {})
    document = (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()
    with pytest.raises(ValueError, match="Unexpected executable markup"):
        constrain_native_html(document.replace("</body>", injection + "</body>"))


def test_failed_and_incomplete_counts_agree_across_all_surfaces(tmp_path):
    summary, _ = _write_fixture(tmp_path, failed=True)
    shutil.copytree(tmp_path / "skill-001", tmp_path / "skill-002")
    payload_path = tmp_path / "skill-002/skillevaluator-tier1.json"
    payload = json.loads(payload_path.read_text())
    payload["results"][0]["incomplete_scans"] = ["scanner"]
    payload["results"][0]["status"] = "incomplete"
    payload.update(overall_status="incomplete", incomplete_scans=["scanner"])
    payload_path.write_text(json.dumps(payload))
    second = {
        **summary["skills"][0],
        "report_dir": "skill-002",
        "path": "skills/incomplete",
        "overall_status": "incomplete",
        "incomplete_scans": ["scanner"],
    }
    summary = _summary([summary["skills"][0], second])
    render_native_reports(tmp_path, summary, {})
    for filename in ("summary.md", "details.md", "pr-comment.md"):
        assert "2 skills · 0 passed · 1 failed · 1 incomplete" in (tmp_path / filename).read_text()
    page = (tmp_path / "index.html").read_text()
    assert "<strong>1</strong><span>Failed</span>" in page
    assert "<strong>1</strong><span>Incomplete</span>" in page
