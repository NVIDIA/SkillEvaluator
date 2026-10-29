# SPDX-License-Identifier: Apache-2.0
"""Report presentation must preserve evidence, scope, and untrusted boundaries."""

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "ci" / "report_views.py"
SPEC = importlib.util.spec_from_file_location("ci_report_views", MODULE_PATH)
views = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(views)


def example(tmp_path, *, result=None):
    directory = tmp_path / "skill-001"
    directory.mkdir()
    result = (
        result
        if result is not None
        else {
            "validator": "Code risk",
            "passed": True,
            "status": "passed",
            "summary": {"checks_performed": 2, "errors": 0, "warnings": 0},
            "success_details": [{"check": "bandit", "message": "No issues"}],
        }
    )
    status = (
        "incomplete"
        if result.get("incomplete_scans") or result.get("status") in {"incomplete", "error"}
        else ("failed" if result.get("status") == "failed" or result.get("passed") is False else "passed")
    )
    errors = result.get("summary", {}).get("errors", 0)
    warnings = result.get("summary", {}).get("warnings", 0)
    data = {
        "overall_status": status,
        "overall_passed": status == "passed",
        "total_errors": errors,
        "total_warnings": warnings,
        "results": [result],
        "quality_summary": [
            {
                "overall_score": 98.2,
                "grade": "A",
                "dimensions": {"correctness": {"score": 95, "weight": 0.35, "issues_count": 1}},
            }
        ],
        "workflow": {"stages": {"llm": {"status": "skipped", "reason": "Disabled by --no-llm"}}},
    }
    (directory / "skillevaluator-tier1.json").write_text(json.dumps(data))
    summary = {
        "total": 1,
        "passed": int(status == "passed"),
        "failed": int(status != "passed"),
        "incomplete": int(status == "incomplete"),
        "overall_status": status,
        "total_errors": errors,
        "total_warnings": warnings,
        "skills": [
            {"path": "skills/sample", "report_dir": "skill-001", "overall_status": status, "passed": status == "passed"}
        ],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    return summary, data


def align_verdict(summary, data, status):
    """Make explicitly mutated fixtures internally consistent; mismatch tests opt out."""
    data.update(overall_status=status, overall_passed=status == "passed")
    summary.update(
        overall_status=status,
        passed=int(status == "passed"),
        failed=int(status != "passed"),
        incomplete=int(status == "incomplete"),
    )
    summary["skills"][0].update(overall_status=status, passed=status == "passed")


def compact_header(markdown):
    return next(line for line in markdown.splitlines() if line.startswith("**Tier 1 scan result:"))


def visible_markdown(markdown):
    """Assert what a GitHub-style Markdown reader sees, including escaped punctuation."""
    from html.parser import HTMLParser

    from markdown_it import MarkdownIt

    class Visible(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    parser = Visible()
    parser.feed(MarkdownIt("commonmark").enable("table").render(markdown))
    return "".join(parser.parts)


def test_scores_scope_and_json_only_publish(tmp_path):
    summary, _ = example(tmp_path)
    original = (tmp_path / "summary.json").read_bytes()
    views.render_reports(
        tmp_path,
        summary,
        {"report_url": "https://example.com/reports/run-1", "run_url": "https://github.com/o/r/actions/runs/1"},
    )
    page = (tmp_path / "details.md").read_text()
    markdown = (tmp_path / "summary.md").read_text()
    comment = (tmp_path / "pr-comment.md").read_text()
    assert "index.html#skill-001" in page
    assert not (tmp_path / "index.html").exists()
    assert "98.2" in visible_markdown(page) and "95" in page and "bandit" in page
    assert "Tier 2: Not run" in markdown and "Tier 3: Not run" in comment
    assert "LLM stages: skipped" in page and "Disabled by" in comment
    assert "https://example.com/reports/run-1/index.html" in comment
    assert "skillevaluator-tier1.html" not in page
    assert (tmp_path / "summary.json").read_bytes() == original


def test_failure_finding_location_and_suggestion_are_visible(tmp_path):
    result = {
        "validator": "Schema",
        "status": "failed",
        "passed": False,
        "summary": {"errors": 1, "warnings": 0, "checks_performed": 1},
        "findings": [
            {
                "check_name": "frontmatter_field",
                "severity": "high",
                "message": "Bad name",
                "file_path": "skills/sample/SKILL.md",
                "line_number": 2,
                "line_content": "name: BAD",
                "suggestion": "Use kebab-case",
            }
        ],
        "legacy": {"errors": ["Schema rejected the name"], "warnings": ["Additional advisory"]},
    }
    summary, _ = example(tmp_path, result=result)
    views.render_reports(tmp_path, summary, {})
    for file in ["summary.md", "details.md", "pr-comment.md"]:
        text = (tmp_path / file).read_text()
        assert "Bad name" in text and "Use kebab" in text and "SKILL" in text
    page = (tmp_path / "details.md").read_text()
    assert page.index("Needs attention") < page.index("Skills and scores")
    assert page.index("Bad name") < page.index("Skills and scores")
    assert "name: BAD" in page
    assert "Schema rejected the name" in (tmp_path / "details.md").read_text()
    assert "Additional advisory" in (tmp_path / "details.md").read_text()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"status": "passed", "passed": True, "success_details": [{"message": "No code - not applicable"}]},
            "PASS",
        ),
        (
            {
                "status": "passed",
                "passed": True,
                "success_details": [{"message": "No code - not applicable"}, {"message": "Ran another check"}],
            },
            "PASS",
        ),
        ({"status": "incomplete", "passed": True}, "INCOMPLETE"),
        ({"status": "passed", "passed": True, "incomplete_scans": ["semgrep"]}, "INCOMPLETE"),
        ({"status": "skipped", "passed": False}, "NOT RUN"),
        ({"status": "skipped", "passed": True, "skip_reason": "Not applicable: no dependency files"}, "NOT APPLICABLE"),
        ({"status": "failed", "passed": True}, "FAIL"),
        ({"status": "passed"}, "UNKNOWN"),
        ({"status": {"malformed": True}}, "UNKNOWN"),
    ],
)
def test_outcomes_never_invent_success(payload, expected):
    assert views._status(payload) == expected


def test_untrusted_strings_and_urls_cannot_create_markup(tmp_path):
    summary, data = example(tmp_path)
    attack = '<script>alert("x")</script> [escape](javascript:alert(1)) |\n## forged'
    summary["skills"][0]["path"] = attack
    data["results"][0]["validator"] = attack
    data["results"][0]["success_details"][0]["message"] = attack
    (tmp_path / "skill-001/skillevaluator-tier1.json").write_text(json.dumps(data))
    views.render_reports(
        tmp_path,
        summary,
        {"run_url": "javascript:alert(1)", "report_url": "data:text/html,x", "image": attack, "commit": attack},
    )
    page = (tmp_path / "details.md").read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert 'href="javascript:' not in page and 'href="data:' not in page
    assert "<script>" not in (tmp_path / "summary.md").read_text()
    assert "\\[escape\\]" in (tmp_path / "details.md").read_text()
    assert "<script" not in page.lower() and "<iframe" not in page.lower()


@pytest.mark.parametrize("kind", ["missing", "malformed", "traversal", "symlink"])
def test_missing_or_unsafe_evidence_never_gets_green_score(tmp_path, kind):
    summary, _ = example(tmp_path)
    path = tmp_path / "skill-001/skillevaluator-tier1.json"
    if kind == "missing":
        path.unlink()
    elif kind == "malformed":
        path.write_text("[]")
    elif kind == "traversal":
        summary["skills"][0]["report_dir"] = "../outside"
    else:
        path.unlink()
        path.symlink_to(tmp_path / "summary.json")
    views.render_reports(tmp_path, summary, {})
    page = (tmp_path / "details.md").read_text()
    assert "INCOMPLETE" in page and "98.2" not in page
    assert "Not recorded" in page
    assert "Tier 1: executed for 0/1" in page


def test_summary_and_comment_limits_keep_complete_source_evidence(tmp_path):
    summary, data = example(tmp_path)
    data["results"][0]["legacy"] = {"warnings": ["Long finding " + "x" * (600 * 1024)]}
    data["results"][0]["status"] = "failed"
    data["results"][0]["passed"] = False
    align_verdict(summary, data, "failed")
    (tmp_path / "skill-001/skillevaluator-tier1.json").write_text(json.dumps(data))
    views.render_reports(tmp_path, summary, {"report_url": "https://example.com/full"})
    assert (tmp_path / "summary.md").stat().st_size <= 20 * 1024
    assert (tmp_path / "pr-comment.md").stat().st_size <= views.COMMENT_LIMIT
    assert "Truncated for CI display limits" in (tmp_path / "details.md").read_text()
    assert "x" * (600 * 1024) in (tmp_path / "skill-001/skillevaluator-tier1.json").read_text()
    assert "https://example.com/full/index.html" in (tmp_path / "pr-comment.md").read_text()


def test_output_symlink_is_rejected(tmp_path):
    summary, _ = example(tmp_path)
    target = tmp_path / "original.txt"
    target.write_text("preserve")
    (tmp_path / "summary.md").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        views.render_reports(tmp_path, summary, {})
    assert target.read_text() == "preserve"


def test_github_mentions_are_neutralized_in_all_markdown_views(tmp_path):
    summary, data = example(tmp_path)
    summary["skills"][0]["path"] = "skills/@some-user/@some-org/team"
    data["results"][0]["status"] = "failed"
    data["results"][0]["passed"] = False
    align_verdict(summary, data, "failed")
    data["results"][0]["findings"] = [
        {"message": "Ask @some-user and @some-org/team", "suggestion": "Contact @some-user"}
    ]
    (tmp_path / "skill-001/skillevaluator-tier1.json").write_text(json.dumps(data))
    views.render_reports(tmp_path, summary, {"commit": "@some-user", "image": "@some-org/team"})
    for filename in ["summary.md", "details.md", "pr-comment.md"]:
        text = (tmp_path / filename).read_text()
        assert "@some-user" not in text
        assert "@some-org" not in text
        assert "@\u200bsome" in text
    # Original retained evidence is not rewritten by the presentation adapter.
    assert "@some-user" in (tmp_path / "skill-001/skillevaluator-tier1.json").read_text()


@pytest.mark.parametrize(
    "result",
    [
        {
            "validator": "SCRIPT_LINT",
            "success_details": [{"check": "lint", "message": "No scripts/ or tools/ directory found"}],
        },
        {
            "validator": "Semantic Version Validation",
            "success_details": [
                {"check": "version_optional", "message": "No version supplied; version validation is optional"}
            ],
        },
    ],
)
def test_optional_or_absent_targets_are_not_reported_as_executed_passes(result):
    result.update({"status": "passed", "passed": True})
    assert views._status(result) == "NOT APPLICABLE"
    assert views._evidence(result)[0][1] == result["success_details"][0]["message"]


@pytest.mark.parametrize("change", ["missing_boolean", "opposite_boolean", "opposite_status", "summary_count"])
def test_inconsistent_verdicts_never_display_a_passing_skill(tmp_path, change):
    summary, data = example(tmp_path)
    if change == "missing_boolean":
        data.pop("overall_passed")
    elif change == "opposite_boolean":
        summary["skills"][0]["passed"] = False
    elif change == "opposite_status":
        summary["skills"][0]["overall_status"] = "failed"
    else:
        summary["passed"] = 0
    (tmp_path / "skill-001/skillevaluator-tier1.json").write_text(json.dumps(data))
    views.render_reports(tmp_path, summary, {})
    page = (tmp_path / "details.md").read_text()
    assert "INCOMPLETE" in page
    if change != "summary_count":
        assert "Summary and child report verdicts" in page


def test_ci_run_conclusion_is_separate_from_scan_result(tmp_path):
    summary, _ = example(tmp_path)
    views.render_reports(tmp_path, summary, {"ci_conclusion": "failure"})
    for filename in ["summary.md", "details.md", "pr-comment.md"]:
        text = (tmp_path / filename).read_text()
        assert "CI run conclusion" in text and "failure" in text
        assert "PASSED" in text


def test_generated_markdown_tables_parse_with_github_table_rules(tmp_path):
    from markdown_it import MarkdownIt

    summary, _ = example(tmp_path)
    views.render_reports(tmp_path, summary, {})
    parser = MarkdownIt("commonmark").enable("table")
    markdown = (tmp_path / "details.md").read_text()
    comment = (tmp_path / "pr-comment.md").read_text()
    summary_html = parser.render(markdown)
    comment_html = parser.render(comment)
    # Overview, real quality dimensions, and validator outcomes are actual tables.
    assert summary_html.count("<table>") == 3
    assert comment_html.count("<table>") == 1
    assert "<td>98.2</td>" in summary_html
    assert "<td>95</td>" in summary_html
    assert "<td>skills/sample</td>" in comment_html
    assert "<td>98.2 / A</td>" in comment_html
    assert "<td>Code risk</td>" in summary_html


def test_hosted_markdown_uses_published_native_html_index_and_json(tmp_path):
    from markdown_it import MarkdownIt

    summary, _ = example(tmp_path)
    directory = tmp_path / "skill-001"
    for filename in [
        "skillevaluator-tier1.html",
        "skillevaluator-tier1.md",
        "skillevaluator-tier1.sarif.json",
        "stdout.log",
        "stderr.log",
    ]:
        (directory / filename).write_text("Original artifact evidence")
    base = "https://example.com/reports/run-1"
    views.render_reports(tmp_path, summary, {"report_url": base})
    markdown = (tmp_path / "details.md").read_text()
    tokens = MarkdownIt("commonmark").enable("table").parse(markdown)
    urls = [child.attrGet("href") for token in tokens for child in token.children or [] if child.type == "link_open"]
    assert base + "/index.html#skill-001" in urls
    assert base + "/skill-001/skillevaluator-tier1.json" in urls
    assert base + "/skill-001/skillevaluator-tier1.html" in urls
    assert all(
        url
        in {
            base + "/index.html",
            base + "/index.html#skill-001",
            base + "/skill-001/skillevaluator-tier1.json",
            base + "/skill-001/skillevaluator-tier1.html",
        }
        for url in urls
    )
    for filename in [
        "skillevaluator-tier1.md",
        "skillevaluator-tier1.sarif.json",
        "stdout.log",
        "stderr.log",
    ]:
        assert filename not in markdown
        # Detailed local Markdown links all original report evidence.
        views.render_reports(tmp_path, summary, {})
        assert f"skill-001/{filename}" in (tmp_path / "details.md").read_text()


def test_truncated_markdown_keeps_table_boundaries_intact():
    from markdown_it import MarkdownIt

    lines = ["# Results", "| Name | Score |", "| --- | --- |"]
    lines.extend(f"| skill-{index} | 98.2 |" for index in range(100))
    text = views._bounded(lines, 1024, "Full report: index.html")
    assert len(text.encode()) <= 1024
    assert "Truncated" in text
    assert "<table>" not in MarkdownIt("commonmark").enable("table").render(text)
    assert "| --- |" not in text


def test_markdown_findings_render_readable_quotes_and_preserve_entity_text(tmp_path):
    from html.parser import HTMLParser

    from markdown_it import MarkdownIt

    class VisibleText(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    summary, data = example(tmp_path)
    message = "Field 'name': invalid \"BAD NAME\"; use kebab-case & preserve <literal> text."
    data["results"][0].update(
        {
            "status": "failed",
            "passed": False,
            "findings": [{"message": message, "check_name": "frontmatter_field", "severity": "high"}],
        }
    )
    align_verdict(summary, data, "failed")
    (tmp_path / "skill-001/skillevaluator-tier1.json").write_text(json.dumps(data))
    views.render_reports(tmp_path, summary, {})
    for filename in ["summary.md", "details.md", "pr-comment.md"]:
        markdown = (tmp_path / filename).read_text()
        parser = VisibleText()
        parser.feed(MarkdownIt("commonmark").enable("table").render(markdown))
        assert message in "".join(parser.parts)
        assert "&\\#x27;" not in markdown
        assert "&#x27;" not in "".join(parser.parts)
    parser = VisibleText()
    parser.feed(MarkdownIt("commonmark").render(views._md("A & B > C < D; entity text &#x27;")))
    assert "A & B > C < D; entity text &#x27;" in "".join(parser.parts)


def failed_result(
    *, message="First blocking problem", suggestion="Apply the recorded fix", path="skills/sample/SKILL.md"
):
    return {
        "validator": "Schema",
        "status": "failed",
        "passed": False,
        "summary": {"errors": 1, "warnings": 0, "checks_performed": 2},
        "findings": [
            {
                "category": "SCHEMA",
                "check_name": "frontmatter_field",
                "severity": "high",
                "message": message,
                "file_path": path,
                "line_number": 2,
                "suggestion": suggestion,
            }
        ],
    }


def test_compact_all_pass_is_concise_and_rich_evidence_stays_in_details(tmp_path):
    from markdown_it import MarkdownIt

    summary, _ = example(tmp_path)
    views.render_reports(tmp_path, summary, {"report_url": "https://example.com/run"})
    compact = (tmp_path / "summary.md").read_text()
    rich = (tmp_path / "details.md").read_text()
    assert compact_header(compact) == "**Tier 1 scan result: PASSED**"
    assert "1 skills · 1 passed · 0 failed · 0 incomplete" in compact
    assert len(compact.encode()) <= 20 * 1024
    assert len(compact) < 2000
    for dump in ("98.2", "Code risk", "bandit", "skills/sample", "Validator outcomes", "Quality dimensions"):
        assert dump not in compact
    assert "<table>" not in MarkdownIt("commonmark").enable("table").render(compact)
    assert "98.2" in rich and "Code risk" in rich and "bandit" in rich
    assert "Tier 2: Not run" in compact and "Tier 3: Not run" in compact
    assert "skipped" in compact and "--no-llm" in visible_markdown(compact)
    assert "https://example.com/run/index.html" in compact


def test_compact_failure_has_one_action_with_recorded_fix_and_location(tmp_path):
    result = failed_result()
    result["findings"].append(
        {
            "severity": "high",
            "message": "Secondary finding belongs in details",
            "file_path": "skills/sample/SECOND.md",
            "suggestion": "Secondary fix",
        }
    )
    summary, _ = example(tmp_path, result=result)
    views.render_reports(tmp_path, summary, {})
    compact = (tmp_path / "summary.md").read_text()
    assert compact_header(compact) == "**Tier 1 scan result: FAILED**"
    assert "1 skills · 0 passed · 1 failed · 0 incomplete" in compact
    assert "First blocking problem" in compact and "Apply the recorded fix" in compact
    assert "skills/sample/SKILL.md:2" in visible_markdown(compact)
    assert "Secondary finding belongs in details" not in compact
    assert "Secondary finding belongs in details" in (tmp_path / "details.md").read_text()
    assert "download" in compact.lower() and "index.html" in compact and "details.md" in compact


@pytest.mark.parametrize("with_diagnostic", [True, False])
def test_compact_incomplete_has_reason_even_without_findings(tmp_path, with_diagnostic):
    result = {
        "validator": "Code risk",
        "status": "incomplete",
        "passed": True,
        "summary": {"errors": 0, "warnings": 1, "checks_performed": 1},
    }
    if with_diagnostic:
        result.update(incomplete_scans=["semgrep"], legacy={"errors": ["Semgrep timed out after 30 seconds"]})
    summary, _ = example(tmp_path, result=result)
    views.render_reports(tmp_path, summary, {})
    compact = (tmp_path / "summary.md").read_text()
    assert compact_header(compact) == "**Tier 1 scan result: INCOMPLETE**"
    assert "1 skills · 0 passed · 0 failed · 1 incomplete" in compact
    assert summary["failed"] == 1  # Runtime failed count includes its incomplete subset.
    assert "Report evidence is missing or inconsistent" not in compact
    assert "skills/sample" in compact and "Problem:" in compact and "Suggested fix:" in compact
    if with_diagnostic:
        assert "Semgrep timed out after 30 seconds" in visible_markdown(compact)
    else:
        assert "incomplete" in compact.lower()


def test_compact_failed_validator_without_recorded_findings_still_has_action(tmp_path):
    summary, _ = example(tmp_path, result={"validator": "Scanner", "status": "failed", "passed": False})
    views.render_reports(tmp_path, summary, {})
    text = (tmp_path / "summary.md").read_text()
    assert "FAILED" in compact_header(text) and "skills/sample" in text
    assert "without a recorded reason" in text and "Suggested fix:" in text


def test_compact_configuration_error_is_not_green_and_has_artifact_fallback(tmp_path):
    summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "incomplete": 0,
        "overall_status": "error",
        "error": "No skill directories configured",
        "skills": [],
    }
    views.render_reports(tmp_path, summary, {"run_url": "https://github.com/o/r/actions/runs/1"})
    compact = (tmp_path / "summary.md").read_text()
    assert compact_header(compact) == "**Tier 1 scan result: ERROR**"
    assert "No skill directories configured" in compact and "configuration" in compact
    assert "artifact" in compact and "index.html" in compact
    assert "https://github.com/o/r/actions/runs/1" in compact
    assert "No blocking issues found" not in compact


def test_comment_retains_run_error_with_all_optional_header_fields(tmp_path):
    summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "incomplete": 0,
        "overall_status": "error",
        "error": "Deleted SKILL.md left tracked content",
        "skills": [],
    }
    context = {
        "run_url": "https://github.com/o/r/actions/runs/1",
        "report_url": "https://example.com/reports/1",
        "ci_conclusion": "failure",
        "commit": "a" * 40,
        "image": "ghcr.io/o/image@sha256:" + "b" * 64,
    }
    views.render_reports(tmp_path, summary, context)
    comment = visible_markdown((tmp_path / "pr-comment.md").read_text())
    assert summary["error"] in comment
    assert context["commit"] in comment and context["image"] in comment.replace("\u200b", "")
    assert "Run error" in comment


@pytest.mark.parametrize(
    "recorded,expected",
    [
        ("tools/check.py", "skills/sample/tools/check.py"),
        ("skills/sample/tools/check.py", "skills/sample/tools/check.py"),
        ("/workspace/skills/sample/tools/check.py", "skills/sample/tools/check.py"),
        ("../../outside", None),
        ("/tmp/foreign", None),
        ("file\nforged", None),
    ],
)
def test_relative_finding_locations_are_resolved_without_escaping_checkout(tmp_path, recorded, expected):
    assert views.finding_repository_path("skills/sample", recorded) == expected
    summary, _ = example(tmp_path, result=failed_result(path=recorded))
    views.render_reports(tmp_path, summary, {})
    markdown = visible_markdown((tmp_path / "summary.md").read_text())
    if expected:
        assert f"File: {expected}:2" in markdown
    else:
        assert "File:" not in markdown


def test_compact_lists_at_most_five_affected_skills_with_explicit_remainder(tmp_path):
    summary, data = example(tmp_path, result=failed_result())
    rows = []
    for index in range(1, 9):
        directory = tmp_path / f"skill-{index:03d}"
        directory.mkdir(exist_ok=True)
        child = copy.deepcopy(data)
        child["results"][0]["findings"][0]["file_path"] = f"skills/broken-{index}/SKILL.md"
        (directory / "skillevaluator-tier1.json").write_text(json.dumps(child))
        rows.append(
            {
                "path": f"skills/broken-{index}",
                "report_dir": directory.name,
                "overall_status": "failed",
                "passed": False,
            }
        )
    summary.update(total=8, passed=0, failed=8, incomplete=0, skills=rows)
    views.render_reports(tmp_path, summary, {})
    text = visible_markdown((tmp_path / "summary.md").read_text())
    for index in range(1, 6):
        assert f"skills/broken-{index}" in text
    for index in range(6, 9):
        assert f"skills/broken-{index}" not in text
    assert re.search(r"3 (?:more|additional) affected skills", text)
    assert "8 skills · 0 passed · 8 failed · 0 incomplete" in text


@pytest.mark.parametrize("kind", ["missing", "contradictory_child", "rollup_counts"])
def test_compact_unproven_evidence_never_claims_pass(tmp_path, kind):
    summary, data = example(tmp_path)
    if kind == "missing":
        (tmp_path / "skill-001/skillevaluator-tier1.json").unlink()
    elif kind == "contradictory_child":
        data["results"][0].update(status="failed", passed=False)
        (tmp_path / "skill-001/skillevaluator-tier1.json").write_text(json.dumps(data))
    else:
        summary.update(total=999, passed=999)
    views.render_reports(tmp_path, summary, {})
    text = (tmp_path / "summary.md").read_text()
    assert "PASSED" not in compact_header(text)
    assert "missing or inconsistent" in text
    assert "1 skills" in text
    if kind != "rollup_counts":
        assert "skills/sample" in text and "Problem:" in text and "Suggested fix:" in text


def test_compact_large_untrusted_fields_stay_bounded_and_useful(tmp_path):
    result = failed_result(
        message="Priority finding " + "界" * 50000,
        suggestion="Recorded fix " + "界" * 50000,
        path="skills/sample/SKILL.md" + "界" * 50000,
    )
    summary, data = example(tmp_path, result=result)
    data["workflow"]["stages"]["llm"]["reason"] = "Large reason " + "界" * 50000
    (tmp_path / "skill-001/skillevaluator-tier1.json").write_text(json.dumps(data))
    views.render_reports(tmp_path, summary, {"report_url": "https://example.com/full"})
    text = (tmp_path / "summary.md").read_text()
    assert len(text.encode()) <= 20 * 1024
    assert "FAILED" in compact_header(text) and "skills/sample" in text
    assert "Priority finding" in text and "Recorded fix" in text
    assert "https://example.com/full/index.html" in text
    assert "界" * 50000 not in text


def test_compact_adversarial_failure_fields_render_as_text(tmp_path):
    from markdown_it import MarkdownIt

    attack = "<script>alert(1)</script> [escape](javascript:alert(1)) @some-org/team |\n## forged"
    summary, _ = example(tmp_path, result=failed_result(message=attack, suggestion=attack, path=attack))
    views.render_reports(tmp_path, summary, {"run_url": "javascript:alert(1)", "report_url": "data:text/html,x"})
    text = (tmp_path / "summary.md").read_text()
    rendered = MarkdownIt("commonmark").enable("table").render(text)
    assert "<script" not in rendered and 'href="javascript:' not in rendered and 'href="data:' not in rendered
    assert "<h2>forged" not in rendered
    assert "@some-org" not in visible_markdown(text) and "@\u200bsome-org" in visible_markdown(text)
    assert "&lt;script&gt;" in text


@pytest.mark.parametrize(
    "field,folder,normalize",
    [("name", "valid-folder", True), ("description", "valid-folder", False), ("name", "INVALID FOLDER", False)],
)
def test_compact_concrete_name_replacement_requires_exact_known_evidence(tmp_path, field, folder, normalize):
    canonical = "Field 'name': Value error, Skill name 'INVALID NAME' must be in kebab-case format (lowercase letters, numbers, and hyphens only, starting with a letter)"
    result = failed_result(message=canonical, suggestion="Recorded suggestion")
    result["findings"][0]["metadata"] = {"field": field}
    summary, _ = example(tmp_path, result=result)
    summary["skills"][0]["path"] = "skills/" + folder
    views.render_reports(tmp_path, summary, {})
    text = visible_markdown((tmp_path / "summary.md").read_text())
    if normalize:
        assert "name: valid-folder" in text and "match the skill directory" in text
    else:
        assert "Recorded suggestion" in text and "Set name:" not in text


def test_compact_near_matching_name_error_keeps_recorded_fix(tmp_path):
    canonical = "Field 'name': Value error, Skill name 'INVALID NAME' must be in kebab-case format (lowercase letters, numbers, and hyphens only, starting with a letter)"
    result = failed_result(message=canonical + " Additional diagnostic context.", suggestion="Recorded suggestion")
    result["findings"][0]["metadata"] = {"field": "name"}
    summary, _ = example(tmp_path, result=result)
    views.render_reports(tmp_path, summary, {})
    text = visible_markdown((tmp_path / "summary.md").read_text())
    assert "Recorded suggestion" in text and "Set name:" not in text


def test_compact_percent_encoded_huge_hosted_url_preserves_headline_and_bound(tmp_path):
    summary, _ = example(tmp_path)
    hosted = "https://example.com/reports/" + "€" * 3000
    views.render_reports(tmp_path, summary, {"report_url": hosted})
    compact = (tmp_path / "summary.md").read_text()
    assert len(compact.encode()) <= 20 * 1024
    assert compact_header(compact) == "**Tier 1 scan result: PASSED**"
    assert "1 skills · 1 passed · 0 failed · 0 incomplete" in compact
    assert "Tier 2: Not run" in compact and "Tier 3: Not run" in compact
    assert "artifact" in compact.lower() and "index.html" in compact
    assert "%E2%82%AC" * 3000 not in compact


def test_compact_without_hosted_url_has_one_artifact_instruction(tmp_path):
    summary, _ = example(tmp_path)
    views.render_reports(tmp_path, summary, {})
    compact = (tmp_path / "summary.md").read_text()
    assert compact.lower().count("artifact") == 1
    assert "scan" in compact.lower() and "download" in compact.lower() and "index.html" in compact


def neutral_selection_summary():
    return {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "incomplete": 0,
        "total_validators": 0,
        "total_errors": 0,
        "total_warnings": 0,
        "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "overall_status": "skipped",
        "overall_passed": False,
        "skills": [],
        "selection": {
            "mode": "changed",
            "base": "a" * 40,
            "base_ref": "c" * 40,
            "head": "b" * 40,
            "discovered": 2,
            "selected": [],
            "deleted": ["skills/removed"],
            "changed_files": 1,
        },
    }


def test_real_noop_marker_reports_no_scan_and_discloses_deleted_skills(tmp_path):
    summary = neutral_selection_summary()
    assert views.is_noop_summary(summary)
    views.render_reports(tmp_path, summary, {"llm_enabled": True})
    for name in ("summary.md", "details.md", "pr-comment.md"):
        text = (tmp_path / name).read_text()
        assert "NO SKILLS CHANGED" in text
        assert "Validation did not run" in text
        assert "Tier 1: Not run" in text and "LLM: Not run" in text
        assert "skills/removed" in text
        assert "PASSED" not in text
    assert not list(tmp_path.glob("skill-*"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(overall_passed=True),
        lambda value: value.update(error="Selection unavailable"),
        lambda value: value.update(total=1),
        lambda value: value.update(total_errors=False),
        lambda value: value["selection"].update(base="unknown"),
        lambda value: value["selection"].update(base_ref="unknown"),
        lambda value: value["selection"].update(selected=["skills/not-scanned"]),
        lambda value: value["selection"].update(deleted=["../outside"]),
        lambda value: value["selection"].update(changed_files=0),
        lambda value: value["selection"].update(discovered=0, deleted=[]),
    ],
)
def test_noop_requires_complete_consistent_selection_evidence(tmp_path, mutate):
    summary = neutral_selection_summary()
    mutate(summary)
    assert views.is_noop_summary(summary) is False
    views.render_reports(tmp_path, summary, {})
    assert "NO SKILLS CHANGED" not in (tmp_path / "summary.md").read_text()
    assert "PASSED" not in compact_header((tmp_path / "summary.md").read_text())
