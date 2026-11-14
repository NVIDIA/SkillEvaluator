# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed result counts the files SkillSpector did not completely inspect in every report.

The counts come from the AE1 findings' evidence (see
``tests/validators/test_skillspector_reference_diagnostics.py``). The real
SkillSpector v2.11.2 report names ``scripts/helper.pl`` at two lines of
``SKILL.md``.
"""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.validators.test_security import _jpeg_bytes, _png_bytes

from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.console_ui import summarize_tier1
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.reporting.retained import render_retained_report, retained_results
from skillevaluator.skillspector_references import (
    combine_reference_summaries,
    reference_summary_message,
    reference_summary_record,
)
from skillevaluator.utils.tool_runner import ToolResult
from skillevaluator.validators.base import Severity, ValidationResult
from skillevaluator.validators.policy import ValidationPolicy, apply_policy
from skillevaluator.validators.security import SecurityValidator

_REPORT = Path(__file__).parents[1] / "fixtures" / "skillspector-2.11.2-perl-help-text-ae1-no-llm.json"
# A complete v2.12.0 scan whose two AE1 findings name image files it does not analyze.
_IMAGE_REPORT = Path(__file__).parents[1] / "fixtures" / "skillspector-2.12.0-image-path-reference-no-llm.json"
# A failed v2.13.3 scan whose first AE1 finding gives a structured reason.
_PARSE_LIMIT_REPORT = Path(__file__).parents[1] / "fixtures" / "skillspector-2.13.3-perl-parse-limit-ae1-no-llm.json"
_AE1_OVERRIDE = "SECURITY.Incomplete referenced artifact analysis (AE1)"


def _write_skill(root: Path) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: perl-test\ndescription: Inspect a Perl helper.\n---\n\nRun `scripts/helper.pl`.\n\n"
        "See `scripts/helper.pl`.\n",
        encoding="utf-8",
    )
    (root / "scripts" / "helper.pl").write_text('print "Use rm to remove a project\\n";\n', encoding="utf-8")


def _validate(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, skills: tuple[str, ...], lines: int = 0
) -> ValidationResult:
    """Scan *skills* under *root* with the captured report; *lines* lists that many reference lines in one AE1."""
    for name in skills:
        _write_skill(root / name)
    validator = SecurityValidator(use_llm=False)

    def scan(path: Path) -> ValidationResult:
        report = json.loads(_REPORT.read_text(encoding="utf-8"))
        if lines:
            report["issues"][0]["occurrences"] = [
                {"file": "SKILL.md", "start_line": line} for line in range(1, lines + 1)
            ]
        result = ValidationResult(validator_name="Security Scan")
        validator._process_skillspector_cli_result(report, result, source_root=path)
        return result

    monkeypatch.setattr(validator, "_run_skillspector", scan)
    return validator.validate_security_only(root)


def _rendered(result: ValidationResult) -> dict[str, str]:
    passed, rows = summarize_tier1([result], lineup=["security"])
    assert passed == result.passed
    return {
        "CLI": re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render(result)),
        "terminal summary": "\n".join("".join(text for text, _style in row.segments) for row in rows),
        "Markdown": MarkdownReporter(include_timestamp=False).render(result),
        # The line above the Tier 1 results.
        "HTML": "\n".join(
            re.findall(
                r'<p class="reference-analysis-summary"[^>]*>([^<]*)</p>',
                HTMLReporter(include_timestamp=False).render(result),
            )
        ),
    }


def _phrase(artifacts: int, occurrences: str) -> str:
    return (
        f"skillspector could not completely inspect {artifacts} referenced artifact(s) across {occurrences} "
        "reference occurrence(s); each AE1 finding names the artifact and any reasons SkillSpector gave"
    )


@pytest.mark.parametrize("skills", [(".",), ("alpha", "beta")], ids=["skill", "folder"])
def test_failed_reports_show_the_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skills: tuple) -> None:
    result = _validate(tmp_path, monkeypatch, skills=skills)

    expected = {
        "affected_artifacts": len(skills),
        "reference_occurrences": 2 * len(skills),
        "reference_occurrences_truncated": False,
    }
    assert not result.passed
    assert result.metadata["skillspector_reference_summary"] == expected
    assert sum(finding.check_name.endswith("(AE1)") for finding in result.findings) == 2 * len(skills)
    phrase = _phrase(len(skills), str(2 * len(skills)))
    for reporter, rendered in _rendered(result).items():
        assert rendered.count(phrase) == 1, reporter
    exported = json.loads(JSONReporter(include_timestamp=False).render(result))
    assert exported["skillspector_reference_summary"] == expected
    combined = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    assert combined["results"][0]["skillspector_reference_summary"] == expected


def test_copies_that_differ_only_in_run_time_are_one_unique_finding() -> None:
    """Two copies of one reference line whose reasons differ only in the run time and time budget group as one finding.

    With ``--llm`` both stages report the line, and they are not merged when
    the second stage is incomplete.
    """
    result = ValidationResult(validator_name="Security Scan")
    for limit, observed in ((29.99847808299819, 30.0012), (29.998489167017397, 30.0441)):
        issue = json.loads(_PARSE_LIMIT_REPORT.read_text(encoding="utf-8"))["issues"][0]
        issue["evidence"]["reasons"] = [
            {
                "reason_code": "runtime_limit",
                "phase": "exclusion_audit",
                "message": "Inspection reached its configured runtime limit.",
                "analyzers": [],
                "limits": {"limit_seconds": limit, "observed_seconds": observed},
            }
        ]
        finding, _is_error = SecurityValidator._convert_skillspector_issue(issue)
        result.add_finding(finding)

    exported = json.loads(JSONReporter(include_timestamp=False).render_all([result]))

    assert exported["finding_counts"]["raw_occurrences"] == 2
    assert exported["finding_counts"]["unique_findings"] == 1
    assert len(exported["finding_groups"]) == 1


def test_reports_rendered_again_from_the_json_show_the_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CI renders its HTML and Markdown reports again from the JSON report; a failed result keeps the line."""
    result = _validate(tmp_path, monkeypatch, skills=(".",))
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))

    [retained] = retained_results(payload)

    assert not retained.passed
    assert retained.metadata["skillspector_reference_summary"] == {
        "affected_artifacts": 1,
        "reference_occurrences": 2,
        "reference_occurrences_truncated": False,
    }
    phrase = _phrase(1, "2")
    assert MarkdownReporter(include_timestamp=False).render_all([retained]).count(phrase) == 1
    html = render_retained_report(payload, skill_name="perl-test")
    assert re.findall(r'<p class="reference-analysis-summary"[^>]*>([^<]*)</p>', html) == [phrase]


@pytest.mark.parametrize(
    "summary",
    [
        [],
        {"affected_artifacts": 0, "reference_occurrences": 2},
        {"affected_artifacts": 1, "reference_occurrences": 10**400},
        {"affected_artifacts": "<script>alert(1)</script>", "reference_occurrences": 2},
    ],
)
def test_reports_rendered_again_from_the_json_drop_malformed_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, summary: object
) -> None:
    result = _validate(tmp_path, monkeypatch, skills=(".",))
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    payload["results"][0]["skillspector_reference_summary"] = summary

    [retained] = retained_results(payload)

    assert "skillspector_reference_summary" not in retained.metadata
    html = render_retained_report(payload, skill_name="perl-test")
    assert "referenced artifact(s)" not in html
    assert "<script>alert(1)" not in html
    # The findings themselves still render.
    assert "Incomplete referenced artifact analysis" in html


def test_folder_reference_lines_name_their_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In a folder, each exported reference line carries its skill's prefix, like the finding's own path."""
    result = _validate(tmp_path, monkeypatch, skills=("alpha", "beta"))

    exported = json.loads(JSONReporter(include_timestamp=False).render(result))
    located = sorted(
        (finding["file_path"], finding["line_number"], finding["metadata"]["skillspector_occurrences"])
        for finding in exported["findings"]
        if finding["check_name"].endswith("(AE1)")
    )
    assert located == [
        (f"[{skill}] SKILL.md", line, [{"file": f"[{skill}] SKILL.md", "start_line": line}])
        for skill in ("alpha", "beta")
        for line in (6, 8)
    ]
    # The referenced file stays relative to the skill that the prefix names.
    assert {
        finding["metadata"]["skillspector_evidence"]["target_path"]
        for finding in exported["findings"]
        if finding["check_name"].endswith("(AE1)")
    } == {"scripts/helper.pl"}


def test_a_cut_occurrence_list_is_a_lower_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _validate(tmp_path, monkeypatch, skills=("alpha", "beta"), lines=129)

    # Each skill: 128 kept lines of the first AE1, of which line 8 is the second AE1's.
    assert result.metadata["skillspector_reference_summary"] == {
        "affected_artifacts": 2,
        "reference_occurrences": 256,
        "reference_occurrences_truncated": True,
    }
    for reporter, rendered in _rendered(result).items():
        assert _phrase(2, "at least 256") in rendered, reporter


@patch("skillevaluator.validators.security.Tools")
def _scan_images(root: Path, mock_tools, *, pattern: object = "keep") -> ValidationResult:
    """Scan a skill that names two images with the captured v2.12.0 report; *pattern* replaces each AE1's."""
    (root / "assets").mkdir(parents=True)
    (root / "SKILL.md").write_text("See `assets/logo.png` and `assets/photo.jpg`.\n", encoding="utf-8")
    (root / "assets" / "logo.png").write_bytes(_png_bytes())
    (root / "assets" / "photo.jpg").write_bytes(_jpeg_bytes())
    report = json.loads(_IMAGE_REPORT.read_text(encoding="utf-8"))
    if pattern != "keep":
        for issue in report["issues"]:
            issue["pattern"] = pattern
    mock_tools.skillspector.is_available = True
    mock_tools.skillspector.run.return_value = ToolResult(
        success=True, stdout=json.dumps(report), stderr="", exit_code=0
    )
    return SecurityValidator(use_llm=False).validate_security_only(root)


def test_a_downgraded_passing_scan_does_not_show_the_counts(tmp_path: Path) -> None:
    """A policy that lowers the AE1 findings of a complete scan passes it; no report lists the count line."""
    result = _scan_images(tmp_path / "raster-demo")
    assert not result.passed
    assert result.incomplete_scans == []

    apply_policy([result], ValidationPolicy(severity_overrides={_AE1_OVERRIDE: Severity.LOW}))

    assert result.passed
    for reporter, rendered in _rendered(result).items():
        assert "could not completely inspect" not in rendered, reporter
    # A passing result lists its messages as passed checks in the full page.
    assert "could not completely inspect" not in HTMLReporter(include_timestamp=False).render(result)
    # JSON records the counts for every result, but not as a message.
    exported = JSONReporter(include_timestamp=False).render(result)
    assert json.loads(exported)["skillspector_reference_summary"] == {
        "affected_artifacts": 2,
        "reference_occurrences": 2,
        "reference_occurrences_truncated": False,
    }
    assert "could not completely inspect" not in exported


@pytest.mark.parametrize(("override", "passes"), [(_AE1_OVERRIDE, True), ("SECURITY.Unknown (AE1)", False)])
def test_overrides_match_the_rule_name_of_reports_before_v2_12(tmp_path: Path, override: str, passes: bool) -> None:
    """Before v2.12.0 an AE1 has no pattern; a severity override uses the rule's name, not ``Unknown``."""
    result = _scan_images(tmp_path / "raster-demo", pattern=None)
    assert {finding.check_name for finding in result.findings} == {"Incomplete referenced artifact analysis (AE1)"}

    apply_policy([result], ValidationPolicy(severity_overrides={override: Severity.LOW}))

    assert result.passed is passes


def test_passing_results_do_not_show_the_counts() -> None:
    """Only a failed result gets the count line; JSON still records it."""
    result = ValidationResult(validator_name="Security Scan")
    result.metadata["skillspector_reference_summary"] = {
        "affected_artifacts": 1,
        "reference_occurrences": 2,
        "reference_occurrences_truncated": False,
    }

    assert result.passed
    for reporter, rendered in _rendered(result).items():
        assert "referenced artifact(s)" not in rendered, reporter
    exported = json.loads(JSONReporter(include_timestamp=False).render(result))
    assert exported["skillspector_reference_summary"]["affected_artifacts"] == 1


@pytest.mark.parametrize(
    "summary",
    [
        None,
        [],
        {"affected_artifacts": 0, "reference_occurrences": 2},
        {"affected_artifacts": True, "reference_occurrences": 2},
        {"affected_artifacts": 1, "reference_occurrences": -1},
        {"affected_artifacts": 1, "reference_occurrences": 1 << 63},
        {"affected_artifacts": "1", "reference_occurrences": 2},
        {"affected_artifacts": "<script>alert(1)</script>", "reference_occurrences": 2},
    ],
)
def test_malformed_counts_are_neither_shown_nor_exported(summary: object) -> None:
    result = ValidationResult(validator_name="Security Scan")
    result.add_error("failed")
    result.metadata["skillspector_reference_summary"] = summary

    assert reference_summary_message(summary) is None
    for reporter, rendered in _rendered(result).items():
        assert "referenced artifact(s)" not in rendered, reporter
    assert "skillspector_reference_summary" not in json.loads(JSONReporter(include_timestamp=False).render(result))


def test_export_keeps_only_the_count_fields() -> None:
    summary = {
        "affected_artifacts": 2,
        "reference_occurrences": 3,
        "reference_occurrences_truncated": "yes",
        "paths": ["scripts/helper.pl"],
    }

    assert reference_summary_record(summary) == {
        "affected_artifacts": 2,
        "reference_occurrences": 3,
        "reference_occurrences_truncated": False,
    }


def test_folder_totals_skip_malformed_children() -> None:
    assert combine_reference_summaries([]) is None
    assert combine_reference_summaries(
        [
            {"affected_artifacts": 1, "reference_occurrences": 2, "reference_occurrences_truncated": False},
            {"affected_artifacts": "1", "reference_occurrences": 2},
            {"affected_artifacts": 3, "reference_occurrences": 0, "reference_occurrences_truncated": True},
        ]
    ) == {"affected_artifacts": 4, "reference_occurrences": 2, "reference_occurrences_truncated": True}
