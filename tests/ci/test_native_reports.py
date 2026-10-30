# SPDX-License-Identifier: Apache-2.0
"""Native CI rendering preserves retained evidence and fails closed."""

import base64
import hashlib
import json
import re
import shutil
from html import unescape
from pathlib import Path

import pytest
from ci.native_reports import _trusted_inline_code, constrain_native_html, render_native_reports
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


def _csp_hashes(document: str) -> set[str]:
    policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', document)[1]
    return set(re.findall(r"'(sha256-[^']+)'", unescape(policy)))


def _sha256(code: str) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(code.encode()).digest()).decode()


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
    # The page links the canonical JSON instead of embedding an unused copy.
    assert 'id="report-data"' not in native
    assert "<img src=x onerror=alert(1)>" not in native
    assert "https://example.com/run/skill-001/skillevaluator-tier1.html" in (tmp_path / "details.md").read_text()


def test_omitting_embedded_report_json_keeps_the_csp_script_hashes(tmp_path):
    summary, raw = _write_fixture(tmp_path, failed=True)
    render_native_reports(tmp_path, summary, {})
    native = (tmp_path / "skill-001/skillevaluator-tier1.html").read_text(encoding="utf-8")
    page = re.sub(r'<meta http-equiv="Content-Security-Policy" content="[^"]+">', "", native, count=1)
    # Put back the block that standalone reports embed, escaped the way HTMLReporter escapes it.
    embedded = json.dumps(json.loads(raw)).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    block = f'<script id="report-data" type="application/json">{embedded}</script>'
    assert page.count("</body>") == 1
    standalone = page.replace("</body>", block + "</body>")
    hashes = _csp_hashes(native)
    assert _csp_hashes(constrain_native_html(standalone)) == _csp_hashes(constrain_native_html(page)) == hashes
    assert hashes and hashes <= {_sha256(code) for code in _trusted_inline_code()}


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
    # A run with nothing to scan uses the report vocabulary and explains itself once.
    assert '<span class="status neutral"><span class="sr-only">Verdict: </span>Skipped</span>' in page
    assert page.count("Validation did not run.") == 1
    assert "Quality is a content score" not in page
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
    assert "</span>Skipped</span>" not in page
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


def test_each_skill_report_is_read_once_for_all_views(tmp_path, monkeypatch):
    from ci import native_reports

    summary, _ = _write_fixture(tmp_path, failed=True)
    shutil.copytree(tmp_path / "skill-001", tmp_path / "skill-002")
    summary = _summary([summary["skills"][0], {**summary["skills"][0], "report_dir": "skill-002", "path": "skills/b"}])
    reads = []
    original = native_reports.views._read
    monkeypatch.setattr(
        native_reports.views, "_read", lambda directory: reads.append(directory.name) or original(directory)
    )
    render_native_reports(tmp_path, summary, {})
    assert sorted(reads) == ["skill-001", "skill-002"]
    assert "Bad name" in (tmp_path / "skill-002/skillevaluator-tier1.html").read_text()
    assert "2 skills · 0 passed · 2 failed · 0 incomplete" in (tmp_path / "details.md").read_text()


def test_failed_and_incomplete_counts_agree_across_all_surfaces(tmp_path):
    summary, _ = _write_fixture(tmp_path, failed=True)
    shutil.copytree(tmp_path / "skill-001", tmp_path / "skill-002")
    payload_path = tmp_path / "skill-002/skillevaluator-tier1.json"
    incomplete = ValidationResult(validator_name="Scanner", validator_description="Scanner execution")
    incomplete.add_error("Scanner execution did not complete")
    incomplete.mark_scan_incomplete("scanner")
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([incomplete]))
    payload_path.write_text(json.dumps(payload))
    second = {
        **summary["skills"][0],
        "report_dir": "skill-002",
        "path": "skills/incomplete",
        "overall_status": "incomplete",
        "incomplete_scans": ["scanner"],
        "total_errors": payload["total_errors"],
        "total_warnings": payload["total_warnings"],
        "severity_counts": payload["severity_counts"],
    }
    summary = _summary([summary["skills"][0], second])
    render_native_reports(tmp_path, summary, {})
    incomplete_report = (tmp_path / "skill-002/skillevaluator-tier1.html").read_text()
    assert "Scanner execution did not complete" in incomplete_report
    assert "Report unavailable" not in incomplete_report
    for filename in ("summary.md", "details.md", "pr-comment.md"):
        assert "2 skills · 0 passed · 1 failed · 1 incomplete" in (tmp_path / filename).read_text()
    page = (tmp_path / "index.html").read_text()
    assert "<strong>1</strong><span>Changes required</span>" in page
    assert "<strong>1</strong><span>Incomplete</span>" in page


def test_native_overview_displays_quality_threshold_failure_not_an_advisory(tmp_path):
    result = ValidationResult(validator_name="QUALITY", validator_description="Quality score")
    result.add_finding(Finding(category="QUALITY", severity="medium", check_name="version", message="Optional version missing", file_path="skills/example/SKILL.md"))
    result.fail_gate("Quality score 98.2/100 is below the required 100/100")
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    directory = tmp_path / "skill-001"
    directory.mkdir()
    (directory / "skillevaluator-tier1.json").write_text(json.dumps(payload))
    row = {"path": "skills/example", "report_dir": "skill-001", "exit_code": 1, "passed": False}
    row.update({key: payload[key] for key in ("overall_status", "incomplete_scans", "total_validators", "total_errors", "total_warnings", "severity_counts")})
    render_native_reports(tmp_path, _summary([row]), {})
    index = (tmp_path / "index.html").read_text()
    attention = index.split('<section class="attention">', 1)[1].split("</section>", 1)[0]
    assert "below the required 100/100" in attention
    assert "Optional version missing" not in attention


def test_index_uses_the_report_vocabulary_logo_and_first_fix(tmp_path):
    summary, _ = _write_fixture(tmp_path, failed=True)
    render_native_reports(tmp_path, summary, {})
    page = (tmp_path / "index.html").read_text()
    report = (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()

    assert '<span class="status fail"><span class="sr-only">Verdict: </span>Changes required</span>' in page
    assert '<span class="status fail">Changes required</span>' in page
    assert 'class="mark"' not in page
    assert re.search(r'<aside class="rail"><div class="brand"><svg class="se-logo"[^>]*aria-hidden="true"', page)
    assert "most severe first" not in page
    # "Needs attention" names the skill report's first fix and links to its card.
    anchor = re.search(r'href="#(f-[^"]+)" class="fix-link"', report).group(1)
    assert f'href="skill-001/skillevaluator-tier1.html#{anchor}"' in page
    assert f'id="{anchor}"' in report


def test_repository_root_skill_lists_each_fix_first_location_once(tmp_path):
    """Code Risk Analysis records absolute scan paths; other scanners record skill-relative ones."""
    risk = ValidationResult(validator_name="Code Risk Analysis", validator_description="Static analysis")
    risk.add_finding(
        Finding(
            category="BANDIT",
            severity="high",
            check_name="B602",
            message="subprocess call with shell=True",
            file_path="/workspace/scripts/fetch.py",
            line_number=21,
        )
    )
    scan = ValidationResult(validator_name="Security Scan", validator_description="Detect vulnerabilities")
    scan.add_finding(
        Finding(
            category="SECURITY",
            severity="high",
            check_name="tool_misuse",
            message="Tool Misuse: subprocess.run(command, shell=True",
            file_path="scripts/fetch.py",
            line_number=21,
        )
    )
    summary, _ = _write_fixture(tmp_path)
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([risk, scan]))
    (tmp_path / "skill-001/skillevaluator-tier1.json").write_text(json.dumps(payload))
    summary["skills"][0].update(path=".", passed=False, overall_status="failed")

    render_native_reports(tmp_path, summary, {})
    report = (tmp_path / "skill-001/skillevaluator-tier1.html").read_text()
    fix_first = report.split('class="fix-first"', 1)[1].split("</section>", 1)[0]

    assert re.findall(r"<code>([^<]+)</code>", fix_first) == ["scripts/fetch.py:21"]
    assert "/workspace/" not in fix_first


def test_index_dark_theme_never_applies_to_print():
    from ci.native_reports import _STYLE

    assert "@media screen and (prefers-color-scheme:dark)" in _STYLE
    assert "@media(prefers-color-scheme:dark)" not in _STYLE
    assert all(int(size) >= 11 for size in re.findall(r"font-size:(\d+)px", _STYLE))
