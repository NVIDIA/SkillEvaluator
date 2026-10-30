# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed Tier 3 source checks are listed in every report, not only in JSON."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from jinja2 import make_logging_undefined
from markupsafe import escape

from skillevaluator.cli import cli
from skillevaluator.constants import SOURCE_VALIDATION_CLI_LIMIT
from skillevaluator.evaluation.tier3_report import advisory_skip_result, dataset_required_result
from skillevaluator.models import Finding, ValidationResult
from skillevaluator.reporting import CLIReporter, HTMLReporter, JSONReporter
from skillevaluator.reporting.base import source_validation_errors
from skillevaluator.reporting.console_ui import summarize_tier3
from skillevaluator.reporting.html import _sanitize_tier3_display_payload
from skillevaluator.tier3.evals_spec import CheckResult

_MESSAGE = "Missing required fields: question"
_OVERFLOW = "see Full Insights in the HTML report."


def _plain(output: str) -> str:
    """Strip ANSI styling and Rich's line wrapping."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).split())


def _source_result(tmp_path: Path, checks: list[CheckResult], *, blocking: bool = True) -> ValidationResult:
    return dataset_required_result(
        tmp_path / "evals" / "evals.json",
        checks,
        blocking=blocking,
        skill_name="example-skill",
        source_kind="evals_json",
    )


def _entry_errors(count: int, message: str = _MESSAGE) -> list[CheckResult]:
    return [CheckResult(f"entry case-{index}", "error", message) for index in range(1, count + 1)]


def _html_section(html: str, section_id: str) -> str:
    start = html.index(f'id="{section_id}"')
    return html[start : html.index("</section>", start)]


@pytest.mark.parametrize("blocking", [False, True])
def test_dataset_required_result_records_only_failed_checks(tmp_path: Path, blocking: bool) -> None:
    result = _source_result(
        tmp_path,
        [
            CheckResult("evals/evals.json", "ok", "Tier 3 eval dataset found."),
            CheckResult("entry case-1", "error", _MESSAGE),
            CheckResult("entry case-1", "warning", "Missing recommended fields: expected_skill"),
            CheckResult("evals/config.yml", "missing", "Required config is absent."),
        ],
        blocking=blocking,
    )

    expected = [
        {"path": "entry case-1", "status": "error", "message": _MESSAGE},
        {"path": "evals/config.yml", "status": "missing", "message": "Required config is absent."},
    ]
    assert result.findings[0].metadata["source_validation"] == expected
    assert result.metadata["agent_eval"]["provenance"]["source_validation"] == expected
    json_report = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    assert json_report["tier3"]["provenance"]["source_validation"] == expected


def test_source_errors_stay_out_of_the_finding_text(tmp_path: Path) -> None:
    # Report grouping and SARIF results key on the finding text, so fixing one
    # entry at a time must not turn the remaining errors into a "new" finding.
    before = _source_result(tmp_path, _entry_errors(13))
    after = _source_result(tmp_path, _entry_errors(13)[3:])
    suggestion = f"Create or fix the Tier 3 source under {tmp_path / 'evals'} and rerun validation."

    assert before.findings[0].suggestion == after.findings[0].suggestion == suggestion
    assert before.findings[0].to_legacy_string() == after.findings[0].to_legacy_string()
    assert before.metadata["agent_eval"]["suggestions"] == after.metadata["agent_eval"]["suggestions"] == [suggestion]
    assert len(after.findings[0].metadata["source_validation"]) == 10


@pytest.mark.parametrize("error_count", [0, 1, 5, 6, 13, 100])
def test_cli_report_previews_source_errors_in_order(tmp_path: Path, error_count: int) -> None:
    output = _plain(CLIReporter().render(_source_result(tmp_path, _entry_errors(error_count))))

    shown = min(error_count, SOURCE_VALIDATION_CLI_LIMIT)
    assert re.findall(r"entry case-\d+:", output) == [f"entry case-{index}:" for index in range(1, shown + 1)]
    assert "Create or fix the Tier 3 source" in output
    assert ("Source validation:" in output) == (error_count > 0)
    if error_count > SOURCE_VALIDATION_CLI_LIMIT:
        hidden = error_count - SOURCE_VALIDATION_CLI_LIMIT
        assert f"{hidden} more {'error' if hidden == 1 else 'errors'}: {_OVERFLOW}" in output
        assert output.index("Source validation:") > output.index("Create or fix the Tier 3 source")
    else:
        assert _OVERFLOW not in output


def test_cli_report_prints_source_errors_as_inert_text(tmp_path: Path) -> None:
    result = _source_result(
        tmp_path,
        [
            CheckResult(
                "entry [bold]x[/bold]\x1b[2J\x1b]8;;https://evil.test/\x07", "error", "Expected <field>\n& more."
            ),
            CheckResult("", "error", "Cannot parse dataset: [/x] at C:\\evals\\"),
        ],
    )

    text = re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render(result))

    # Rich's own styling aside, no escape sequence from the eval file reaches the terminal.
    assert re.search(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]", text) is None, repr(text)
    output = " ".join(text.split())
    assert "Source validation: entry [bold]x[/bold]: Expected <field> & more." in output
    assert "more. Cannot parse dataset: [/x] at C:\\evals\\" in output
    assert "evil.test" not in output


def test_cli_report_ignores_malformed_source_validation_metadata() -> None:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.add_finding(
        Finding(
            category="AGENT_EVAL",
            severity="high",
            check_name="eval_dataset_required",
            message="No valid Tier 3 source.",
            file_path="evals/evals.json",
            suggestion="Keep the existing recommendation.",
            metadata={
                "source_validation": [
                    "entry text-only",
                    {"path": "entry ok", "status": "ok", "message": "Loaded."},
                    {"path": "entry warn", "status": "warning", "message": "Optional field."},
                    {"path": 7, "status": "error", "message": "Non-string path."},
                    {"path": "entry blank", "status": "error", "message": "  "},
                    {"path": "entry kept", "status": "error", "message": "Invalid entry."},
                ]
            },
        )
    )

    output = _plain(CLIReporter().render(result))

    assert "Keep the existing recommendation." in output
    assert "Source validation: entry kept: Invalid entry." in output
    for dropped in ("text-only", "Loaded.", "Optional field.", "Non-string path.", "entry blank"):
        assert dropped not in output


@pytest.mark.parametrize("metadata", [{}, {"source_validation": []}, {"source_validation": "entry x: bad"}])
def test_cli_report_omits_source_section_without_errors(metadata: dict) -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity="high",
            check_name="schema-check",
            message="A validation finding.",
            file_path="SKILL.md",
            suggestion="Keep the existing recommendation.",
            metadata=metadata,
        )
    )

    output = _plain(CLIReporter().render(result))

    assert "Keep the existing recommendation." in output
    assert "Source validation:" not in output


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("error_count", [0, 1, 5, 6, 13, 100])
@pytest.mark.parametrize("show_source_details", [False, True])
def test_compact_tier3_panel_counts_and_previews_source_errors(
    tmp_path: Path,
    blocking: bool,
    error_count: int,
    show_source_details: bool,
) -> None:
    result = _source_result(tmp_path, _entry_errors(error_count), blocking=blocking)
    finding = result.findings[0]
    original_suggestion = finding.suggestion
    original_metadata = json.dumps(result.metadata, sort_keys=True)

    ran, _passed, rows, skip_reason = summarize_tier3(result, show_source_details=show_source_details)

    assert ran is True
    assert skip_reason == ""
    shown = min(error_count, SOURCE_VALIDATION_CLI_LIMIT) if show_source_details else 0
    assert [row.segments[0][0] for row in rows if row.label == "source"] == [
        f"entry case-{index}: {_MESSAGE}" for index in range(1, shown + 1)
    ]
    counts = [row.segments[0][0] for row in rows if row.label == "validation"]
    truncated = show_source_details and error_count > SOURCE_VALIDATION_CLI_LIMIT
    if error_count:
        expected = f"{error_count} source validation {'error' if error_count == 1 else 'errors'}"
        if truncated:
            expected += f" (showing first {SOURCE_VALIDATION_CLI_LIMIT})"
        assert counts == [expected]
    else:
        assert counts == []
    rendered = " ".join(text for row in rows for text, _style in row.segments)
    assert (_OVERFLOW in rendered) == truncated
    assert finding.suggestion == original_suggestion
    assert json.dumps(result.metadata, sort_keys=True) == original_metadata


def test_compact_tier3_panel_projects_source_errors_to_one_safe_line(tmp_path: Path) -> None:
    result = _source_result(
        tmp_path,
        [CheckResult("entry \x1b[2Jcase\x07-1", "error", "Expected\na value [bold]here[/bold]. " + "x" * 300)],
    )

    _ran, _passed, rows, _skip_reason = summarize_tier3(result)

    (source_row,) = [row for row in rows if row.label == "source"]
    text = source_row.segments[0][0]
    assert text.startswith("entry [2Jcase -1: Expected a value [bold]here[/bold]. x")
    assert text.endswith("…")
    assert len(text) <= 110
    assert not any(ord(character) < 32 for character in text)


@pytest.mark.parametrize("error_count", [0, 1, 2, 13, 100])
def test_html_report_shows_source_errors_in_preview_and_full_insights(tmp_path: Path, error_count: int) -> None:
    result = _source_result(tmp_path, _entry_errors(error_count))
    html = HTMLReporter(include_timestamp=False).render_all([result])
    preview = _html_section(html, "tier3-next-steps-preview")
    insights = _html_section(html, "tier3-page-suggestions")

    assert "Open Full Insights</button>" in preview
    assert "Create or fix the Tier 3 source" in preview
    assert "Create or fix the Tier 3 source" in insights
    if error_count == 1:
        assert f"Source validation error: entry case-1: {_MESSAGE}" in preview
    elif error_count > 1:
        assert f"{error_count} source validation errors found. Open Full Insights to review the details." in preview
        assert "entry case-" not in preview
    else:
        assert "Source validation error" not in preview
        assert "source validation errors found" not in preview

    if error_count:
        assert f"Source Validation Errors ({error_count})" in insights
        items = re.findall(r"<li>(entry case-\d+): ([^<]*)</li>", insights)
        assert items == [(f"entry case-{index}", _MESSAGE) for index in range(1, error_count + 1)]
    else:
        assert 'id="tier3-source-validation-errors"' not in insights


def test_html_report_escapes_source_errors(tmp_path: Path) -> None:
    result = _source_result(tmp_path, [CheckResult("entry <b>&", "error", "Expected <script>alert(1)</script>.")])
    html = HTMLReporter(include_timestamp=False).render_all([result])
    preview = _html_section(html, "tier3-next-steps-preview")
    insights = _html_section(html, "tier3-page-suggestions")
    error_text = escape("entry <b>&: Expected <script>alert(1)</script>.")

    assert f"Source validation error: {error_text}" in preview
    assert f"<li>{error_text}</li>" in insights
    assert "<script>alert(1)" not in html


@pytest.mark.parametrize("has_source_errors", [False, True])
def test_html_source_errors_keep_existing_insights(tmp_path: Path, has_source_errors: bool) -> None:
    result = _source_result(tmp_path, _entry_errors(1 if has_source_errors else 0))
    payload = result.metadata["agent_eval"]
    payload["conclusions"] = [{"title": "Coverage gap", "message": "Error handling needs more coverage."}]
    payload["recommendations"] = [{"title": "Improve coverage", "message": "Add an error-handling test."}]

    insights = _html_section(HTMLReporter(include_timestamp=False).render_all([result]), "tier3-page-suggestions")

    assert "<summary>Conclusions</summary>" in insights
    assert "Error handling needs more coverage." in insights
    assert "<summary>Recommendations</summary>" in insights
    assert "Add an error-handling test." in insights
    assert ('id="tier3-source-validation-errors"' in insights) == has_source_errors


@pytest.mark.parametrize("has_source_errors", [False, True])
def test_html_empty_insights_message_requires_no_source_errors(tmp_path: Path, has_source_errors: bool) -> None:
    payload = _source_result(tmp_path, _entry_errors(1 if has_source_errors else 0)).metadata["agent_eval"]
    payload["suggestions"] = []
    # A bare payload carries no finding diagnostics that would become recommendations.
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.passed = False
    result.metadata["agent_eval"] = payload

    insights = _html_section(HTMLReporter(include_timestamp=False).render_all([result]), "tier3-page-suggestions")

    assert ("No evaluation insights were generated." in insights) == (not has_source_errors)
    assert ('id="tier3-source-validation-errors"' in insights) == has_source_errors


def test_skipped_tier3_html_defines_source_errors(caplog: pytest.LogCaptureFixture) -> None:
    reporter = HTMLReporter(include_timestamp=False, expected_skill_name="example-skill")
    reporter._env.undefined = make_logging_undefined(logger=logging.getLogger(__name__))

    with caplog.at_level(logging.WARNING):
        html = reporter.render_all([advisory_skip_result("Harbor runtime unavailable", skill_name="example-skill")])

    assert "Harbor runtime unavailable" in html
    assert 'id="tier3-source-validation-errors"' not in html
    assert "t3_source_errors" not in caplog.text


def test_html_payload_keeps_only_well_formed_source_errors() -> None:
    payload = _sanitize_tier3_display_payload(
        {
            "provenance": {
                "source_validation": [
                    {"path": "entry a", "status": "error", "message": "Invalid entry.", "extra": "dropped"},
                    {"path": "entry b", "status": "ok", "message": "Loaded."},
                    {"path": None, "status": "error", "message": "Bad path."},
                    ["entry c", "error", "Not a mapping."],
                    {"message": "No path or status recorded."},
                ]
            }
        }
    )

    assert payload["provenance"]["source_validation"] == [
        {"path": "entry a", "status": "error", "message": "Invalid entry."},
        {"path": "", "status": "error", "message": "No path or status recorded."},
    ]
    assert source_validation_errors({"source_validation": {"path": "entry a"}}) == []


def _write_invalid_dataset(skill: Path, count: int) -> None:
    (skill / "evals").mkdir()
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": f"case-{index}", "expected_behavior": ["Done."]} for index in range(1, count + 1)]),
        encoding="utf-8",
    )


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("report_formats", [None, "cli,json,html", "json,html"])
def test_validate_shows_source_errors_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    blocking: bool,
    report_formats: str | None,
) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.evaluation import EvaluationService

    # A forced terminal would animate the pipeline view and repeat each frame.
    for name in ("FORCE_COLOR", "TTY_COMPATIBLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "120")
    skill = tmp_path / "example-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: example-skill\ndescription: test\n---\n", encoding="utf-8")
    _write_invalid_dataset(skill, 13)
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "ok")
    monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [tier1])
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda *_args, **_kwargs: pytest.fail("source preflight must run before live evaluation"),
    )
    output_dir = tmp_path / "reports"
    args = ["validate", "--no-autopilot", str(skill), "--no-tier2", "--tier3", "--output-dir", str(output_dir)]
    if report_formats:
        args.extend(["--report", report_formats])
    if blocking:
        args.append("--block-on-agent-eval")

    command = CliRunner().invoke(cli, args)

    assert command.exit_code == (1 if blocking else 0), command.output
    output = _plain(command.output)
    full_cli_report = report_formats is not None and "cli" in report_formats.split(",")
    assert "13 source validation errors" in output
    assert ("(showing first 5)" in output) == (not full_cli_report)
    assert ("Source validation:" in output) == full_cli_report
    assert re.findall(r"entry case-\d+:", output) == [f"entry case-{index}:" for index in range(1, 6)]
    assert output.count(f"8 more errors: {_OVERFLOW}") == 1
    report = json.loads(next(output_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert len(report["tier3"]["provenance"]["source_validation"]) == 13
    html = next(output_dir.glob("*.html")).read_text(encoding="utf-8")
    assert "Source Validation Errors (13)" in html
    assert f"<li>entry case-13: {_MESSAGE}</li>" in html
