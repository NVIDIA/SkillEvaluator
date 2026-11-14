# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SkillSpector AE1 findings name the file it did not completely inspect and why.

SkillSpector reports a High ``AE1`` (incomplete referenced artifact analysis)
finding at each line that names such a file. The findings keep their severity,
and the scan's completeness and gates are unchanged; only the finding's name,
suggestion and metadata explain it, and the result counts the files and lines.

The reports are real SkillSpector ``--no-llm`` captures with ``skill.source``
set to ``/fixture/<name>``:

- v2.11.2 for a skill whose ``SKILL.md`` names ``scripts/helper.pl`` twice.
  That release leaves ``pattern`` and ``evidence`` empty, so the reason is
  only in the ledger row for the helper.
- v2.12.0 for a skill that names three helpers, one of them twice, with a
  ``static_parse_limit`` or ``obfuscated_instruction_text`` reason each.
- v2.13.3 for a Perl helper with a string of 2,000 words, which SkillSpector
  cannot parse within its span limit.
"""

import copy
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.validators.test_security import _jpeg_bytes, _png_bytes

from skillevaluator import skillspector_references
from skillevaluator.skillspector_references import (
    AE1_PATTERN,
    index_legacy_reference_reasons,
    issue_diagnostics,
    reference_summary_message,
    report_path,
)
from skillevaluator.utils.tool_runner import ToolResult
from skillevaluator.validators.base import Severity, ValidationResult
from skillevaluator.validators.security import SecurityValidator, _record_skillspector_reference_summary

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_LEGACY_REPORT = "skillspector-2.11.2-perl-help-text-ae1-no-llm.json"
_HOST_LANGUAGE_REPORT = "skillspector-2.12.0-host-language-parse-limits-ae1-no-llm.json"
_PARSE_LIMIT_REPORT = "skillspector-2.13.3-perl-parse-limit-ae1-no-llm.json"
_IMAGE_PATH_REPORT = "skillspector-2.12.0-image-path-reference-no-llm.json"
_CHECK_NAME = f"{AE1_PATTERN} (AE1)"
_INCOMPLETE = "skillspector JSON field 'analysis_completeness' reports incomplete analysis (status 'partial')"
_OBFUSCATION_ANALYZERS = [
    f"static_patterns_{name}"
    for name in (
        "agent_snooping",
        "anti_refusal",
        "deserialization",
        "excessive_agency",
        "harmful_content",
        "memory_poisoning",
        "privilege_escalation",
        "prompt_injection",
        "rogue_agent",
        "ssrf",
        "supply_chain",
        "system_prompt_leakage",
        "tool_misuse",
    )
]


def _report(name: str = _LEGACY_REPORT) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _ae1(result: ValidationResult) -> list:
    return [finding for finding in result.findings if finding.check_name.endswith("(AE1)")]


def _count_messages(result: ValidationResult) -> list[str]:
    """Result messages with the count line, which reporters would list as passed checks."""
    return [message for message in result.messages if "could not completely inspect" in message]


def _process(report: dict, source_root: Path | None = None) -> ValidationResult:
    result = ValidationResult(validator_name="Security Scan")
    SecurityValidator(use_llm=False)._process_skillspector_cli_result(report, result, source_root=source_root)
    return result


def _helper_skill(root: Path) -> Path:
    """The skill of the v2.11.2 report: SKILL.md names scripts/helper.pl on lines 6 and 8."""
    (root / "scripts").mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: perl-test\ndescription: Inspect a Perl helper.\n---\n\nRun `scripts/helper.pl`.\n\n"
        "See `scripts/helper.pl`.\n",
        encoding="utf-8",
    )
    (root / "scripts" / "helper.pl").write_text('print "Use rm to remove a project\\n";\n', encoding="utf-8")
    return root


def _parse_limit_skill(root: Path) -> Path:
    """The skill of the v2.13.3 report, whose helper SkillSpector cannot parse within its span limit."""
    (root / "scripts").mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: parser-limit\ndescription: Inspect a Perl helper.\n---\n\nRun `scripts/helper.pl`.\n\n"
        "See `scripts/helper.pl`.\n",
        encoding="utf-8",
    )
    (root / "scripts" / "helper.pl").write_text(
        '#!/usr/bin/env perl\nuse strict;\nuse warnings;\nprint "Use rm ' + "safe " * 2000 + '-rf *";\n',
        encoding="utf-8",
    )
    return root


class TestCapturedReports:
    """Real reports from three SkillSpector releases."""

    def test_legacy_report_is_named_and_explained_from_the_ledger(self) -> None:
        report = _report()
        assert {issue["pattern"] for issue in report["issues"]} == {None}
        assert {json.dumps(issue["evidence"]) for issue in report["issues"]} == {"{}"}

        result = _process(report)

        findings = _ae1(result)
        assert [finding.line_number for finding in findings] == [6, 8]
        for finding in findings:
            assert finding.check_name == _CHECK_NAME
            assert finding.severity == Severity.HIGH
            assert finding.message == "analysis-evasion: scripts/helper.pl (partial)"
            assert finding.suggestion.startswith(
                "Referenced artifact scripts/helper.pl was not completely inspected (partial). "
                "static_parse_limit (static_patterns_tool_misuse): A security-relevant expression exceeded "
                "a bounded static parser's span limit. "
            )
            # The generic remediation of this release suggests the file is absent.
            assert "locally available" not in finding.suggestion
            assert "minimal reproducer" in finding.suggestion
            assert finding.metadata["skillspector_evidence"] == {
                "target_path": "scripts/helper.pl",
                "target_disposition": "partial",
                "reasons": [
                    {
                        "reason_code": "static_parse_limit",
                        "phase": "static",
                        "message": "A security-relevant expression exceeded a bounded static parser's span limit.",
                        "analyzers": ["static_patterns_tool_misuse"],
                    }
                ],
            }
            assert finding.metadata["skillspector_occurrences"] == [
                {"file": "SKILL.md", "start_line": finding.line_number}
            ]
            assert finding.metadata["skillspector_confidence"] == 1.0
        assert result.metadata["skillspector_reference_summary"] == {
            "affected_artifacts": 1,
            "reference_occurrences": 2,
            "reference_occurrences_truncated": False,
        }
        # Reporters show the count from the metadata, and only for a failed result.
        assert _count_messages(result) == []

    def test_structured_evidence_names_every_reason_and_keeps_the_scanner_remedy(self) -> None:
        report = _report(_HOST_LANGUAGE_REPORT)

        result = _process(report)

        by_target = {}
        for finding in _ae1(result):
            assert finding.check_name == _CHECK_NAME
            by_target.setdefault(finding.metadata["skillspector_evidence"]["target_path"], []).append(finding)
        assert {target: [f.line_number for f in found] for target, found in by_target.items()} == {
            "scripts/render_notes.py": [8, 14],
            "scripts/check_labels.py": [10],
            "scripts/load_plugins.pl": [12],
        }
        render_notes = by_target["scripts/render_notes.py"][0].suggestion
        assert render_notes.startswith(
            "Referenced artifact scripts/render_notes.py was not completely inspected (partial). "
            "static_parse_limit (static_patterns_tool_misuse): "
        )
        assert render_notes.endswith(report["issues"][0]["remediation"])
        # Thirteen reasons that differ only in the analyzer read as one.
        check_labels = by_target["scripts/check_labels.py"][0]
        assert len(check_labels.metadata["skillspector_evidence"]["reasons"]) == 13
        assert check_labels.suggestion.count("obfuscated_instruction_text") == 1
        assert (
            f"obfuscated_instruction_text ({', '.join(_OBFUSCATION_ANALYZERS)}): Obfuscated instruction text "
            "could not be fully evaluated by the deterministic layer."
        ) in check_labels.suggestion
        assert result.metadata["skillspector_reference_summary"] == {
            "affected_artifacts": 3,
            "reference_occurrences": 4,
            "reference_occurrences_truncated": False,
        }

    @patch("skillevaluator.validators.security.Tools")
    def test_scan_stays_incomplete_and_failed(self, mock_tools, tmp_path: Path) -> None:
        """The diagnostics change no gate: the partial report keeps the scan INCOMPLETE."""
        root = _parse_limit_skill(tmp_path / "perl-parse-limit")
        mock_tools.skillspector.is_available = True
        mock_tools.skillspector.run.return_value = ToolResult(
            success=True, stdout=json.dumps(_report(_PARSE_LIMIT_REPORT)), stderr="", exit_code=0
        )

        result = SecurityValidator(use_llm=False).validate_security_only(root)

        assert not result.passed
        assert result.incomplete_scans == ["skillspector"]
        assert any(error.startswith(_INCOMPLETE) for error in result.errors)
        findings = _ae1(result)
        assert [(finding.line_number, finding.severity) for finding in findings] == [
            (6, Severity.HIGH),
            (8, Severity.HIGH),
        ]
        assert all("static_parse_limit (static_patterns_tool_misuse)" in finding.suggestion for finding in findings)
        assert result.summary.high_count == 2
        assert result.metadata["skillspector_reference_summary"]["reference_occurrences"] == 2

    @patch("skillevaluator.validators.security.Tools")
    def test_llm_stage_counts_each_reference_once(self, mock_tools, tmp_path: Path) -> None:
        """Both stages report the AE1 findings of an accepted image gap; each reference line counts once."""
        root = tmp_path / "raster-demo"
        (root / "assets").mkdir(parents=True)
        (root / "SKILL.md").write_text("See `assets/logo.png` and `assets/photo.jpg`.\n", encoding="utf-8")
        (root / "assets" / "logo.png").write_bytes(_png_bytes())
        (root / "assets" / "photo.jpg").write_bytes(_jpeg_bytes())
        static = _report(_IMAGE_PATH_REPORT)
        enriched = _report(_IMAGE_PATH_REPORT)
        enriched["metadata"].update({"llm_requested": True, "llm_available": True})
        for status in enriched["analysis_completeness"]["analyzer_statuses"]:
            if status["status"] == "disabled":
                status.update({"status": "completed", "planned_work": 1, "completed": 1, "reason_code": None})
        mock_tools.skillspector.is_available = True
        mock_tools.skillspector.run.side_effect = [
            ToolResult(success=True, stdout=json.dumps(report), stderr="", exit_code=0) for report in (static, enriched)
        ]

        result = SecurityValidator(use_llm=True).validate_security_only(root)

        assert mock_tools.skillspector.run.call_count == 2
        assert result.incomplete_scans == []
        assert sorted(finding.metadata["skillspector_evidence"]["target_path"] for finding in _ae1(result)) == [
            "assets/logo.png",
            "assets/photo.jpg",
        ]
        # One line names both images: two references, one per artifact.
        assert result.metadata["skillspector_reference_summary"] == {
            "affected_artifacts": 2,
            "reference_occurrences": 2,
            "reference_occurrences_truncated": False,
        }
        assert _count_messages(result) == []


class TestRuleName:
    @pytest.mark.parametrize("pattern", [None, "", "   ", [], {"unexpected": "object"}])
    def test_ae1_without_a_pattern_is_named_by_its_rule(self, pattern: object) -> None:
        issue = _report()["issues"][0]
        issue["pattern"] = pattern
        finding, is_error = SecurityValidator._convert_skillspector_issue(issue)
        assert finding.check_name == _CHECK_NAME
        assert is_error

    def test_ae1_keeps_the_scanner_pattern(self) -> None:
        issue = _report()["issues"][0]
        issue["pattern"] = "Referenced helper not inspected"
        finding, _is_error = SecurityValidator._convert_skillspector_issue(issue)
        assert finding.check_name == "Referenced helper not inspected (AE1)"

    def test_other_rules_keep_their_fallback_name_and_metadata(self) -> None:
        issue = {
            "id": "SQP1",
            "category": "Quality Policy",
            "pattern": None,
            "severity": "LOW",
            "finding": "Trigger description",
            "evidence": {"target_path": "scripts/helper.pl", "target_disposition": "partial", "reasons": []},
            "occurrences": [{"file": "SKILL.md", "start_line": 1}],
        }
        finding, _is_error = SecurityValidator._convert_skillspector_issue(issue)
        assert finding.check_name == "Unknown (SQP1)"
        assert finding.metadata == {}


class TestEvidenceProjection:
    def test_evidence_and_limits_are_bounded_and_sanitized(self) -> None:
        issue = _report()["issues"][0]
        issue["evidence"] = {
            "target_path": "scripts/helper.pl",
            "target_disposition": "partial",
            "reasons": [
                {
                    "reason_code": "static_parse_limit",
                    "phase": "static",
                    "message": "Expression exceeded parser limit.",
                    "analyzers": ["static_patterns_tool_misuse", {"arbitrary": "object"}, "bad\x1b[31m"],
                    "start_line": 3,
                    "end_line": True,
                    "limits": {
                        "observed_characters": 9000,
                        "limit_characters": 8000,
                        "observed_seconds": 1.25,
                        "limit_seconds": float("inf"),
                        "observed_depth": True,
                        "limit_bytes": -1,
                        "token": "not-for-export",
                        "arbitrary": {"nested": "not-for-export"},
                    },
                    "unexpected": "not-for-export",
                }
            ]
            * 17,
            "arbitrary": {"nested": "not-for-export"},
        }
        issue["occurrences"] += [{"file": "../outside", "start_line": 2}, {"file": "SKILL.md", "start_line": True}]

        finding, is_error = SecurityValidator._convert_skillspector_issue(issue)

        evidence = finding.metadata["skillspector_evidence"]
        assert is_error
        assert evidence["reasons_truncated"] is True
        assert evidence["reasons"] == [
            {
                "reason_code": "static_parse_limit",
                "phase": "static",
                "message": "Expression exceeded parser limit.",
                "analyzers": ["static_patterns_tool_misuse"],
                "start_line": 3,
                "limits": {"observed_characters": 9000, "limit_characters": 8000, "observed_seconds": 1.25},
            }
        ]
        assert "at target line 3" in finding.suggestion
        # The measured run time stays in the metadata, not in the suggestion.
        assert "limit_characters=8000, observed_characters=9000." in finding.suggestion
        assert "observed_seconds" not in finding.suggestion
        assert "not-for-export" not in json.dumps(finding.metadata)
        assert "bad" not in json.dumps(finding.metadata)
        assert finding.metadata["skillspector_occurrences"] == [
            {"file": "SKILL.md", "start_line": 6},
            {"file": "SKILL.md"},
        ]

    @pytest.mark.parametrize("field", ["observed_characters", "observed_seconds"])
    @pytest.mark.parametrize(
        "value",
        [10**400, -(10**400), 1 << 63, float("nan"), -1.0],
        ids=["beyond-float", "below-float", "beyond-count", "nan", "negative"],
    )
    def test_out_of_range_limits_are_dropped(self, field: str, value: object) -> None:
        """A limit beyond the count range, even an integer too large for a float, is left out."""
        issue = _report(_PARSE_LIMIT_REPORT)["issues"][0]
        issue["evidence"]["reasons"][0]["limits"] = {field: value, "limit_characters": 8000}

        finding, is_error = SecurityValidator._convert_skillspector_issue(issue)

        assert is_error
        assert finding.metadata["skillspector_evidence"]["reasons"][0]["limits"] == {"limit_characters": 8000}

    @staticmethod
    def _runtime_limit_finding(*runs: tuple[float, float]):
        """The v2.13.3 AE1 finding with one runtime_limit reason per ``(limit_seconds, observed_seconds)`` in *runs*."""
        issue = _report(_PARSE_LIMIT_REPORT)["issues"][0]
        issue["evidence"]["reasons"] = [
            {
                "reason_code": "runtime_limit",
                "phase": "exclusion_audit",
                "message": "Inspection reached its configured runtime limit.",
                "analyzers": [],
                "limits": {"limit_seconds": limit, "observed_seconds": observed},
            }
            for limit, observed in runs
        ]
        finding, is_error = SecurityValidator._convert_skillspector_issue(issue)
        assert is_error
        return finding

    @pytest.mark.parametrize(
        ("first_run", "second_run"),
        [
            ((30.0, 30.0412), (30.0, 30.1877)),
            ((29.99847808299819, 30.0012), (29.998489167017397, 30.0441)),
            ((12.5, 0.0), (7.25, 0.0)),
        ],
        ids=["run-time", "dependency-budget-left", "yara-budget-left"],
    )
    def test_run_time_and_time_budget_do_not_change_the_finding_id(
        self, first_run: tuple[float, float], second_run: tuple[float, float]
    ) -> None:
        """Two runs that differ only in the run time and the time budget give the same suggestion and ``finding_id``.

        SkillSpector sets ``limit_seconds`` to the part of the scan's time
        budget left when a step began, such as the dependency analysis of a
        lockfile, or each file that YARA did not reach (with an
        ``observed_seconds`` of 0), so it differs from run to run as well.
        """
        first = self._runtime_limit_finding(first_run)
        second = self._runtime_limit_finding(second_run)

        assert first.suggestion == second.suggestion
        assert first.finding_id == second.finding_id
        assert "runtime_limit: Inspection reached its configured runtime limit. " in first.suggestion
        assert "_seconds" not in first.suggestion
        # The metadata, which the ID does not hash, keeps what each run recorded.
        assert [finding.metadata["skillspector_evidence"]["reasons"][0]["limits"] for finding in (first, second)] == [
            {"limit_seconds": limit, "observed_seconds": observed} for limit, observed in (first_run, second_run)
        ]

    def test_a_time_limit_is_left_out_beside_a_size_limit(self) -> None:
        """Only the time limits leave the suggestion; a limit that the file and configuration fix stays."""
        issue = _report(_PARSE_LIMIT_REPORT)["issues"][0]
        issue["evidence"]["reasons"][0]["limits"] = {
            "observed_characters": 9000,
            "limit_characters": 8000,
            "observed_seconds": 2.5,
            "limit_seconds": 17.75,
        }

        finding, is_error = SecurityValidator._convert_skillspector_issue(issue)

        assert is_error
        assert "; limit_characters=8000, observed_characters=9000." in finding.suggestion
        assert "_seconds" not in finding.suggestion
        assert finding.metadata["skillspector_evidence"]["reasons"][0]["limits"] == {
            "observed_characters": 9000,
            "limit_characters": 8000,
            "observed_seconds": 2.5,
            "limit_seconds": 17.75,
        }

    def test_reasons_that_differ_only_in_run_time_are_described_once(self) -> None:
        finding = self._runtime_limit_finding((30.0, 1.5), (12.0, 2.5))

        assert len(finding.metadata["skillspector_evidence"]["reasons"]) == 2
        assert finding.suggestion.count("runtime_limit") == 1
        assert finding.finding_id == self._runtime_limit_finding((5.0, 3.5)).finding_id

    @patch("skillevaluator.validators.security.Tools")
    def test_an_oversized_limit_does_not_stop_the_scan(self, mock_tools, tmp_path: Path) -> None:
        root = _parse_limit_skill(tmp_path / "perl-parse-limit")
        report = _report(_PARSE_LIMIT_REPORT)
        report["issues"][0]["evidence"]["reasons"][0]["limits"] = {"observed_characters": 10**400}
        mock_tools.skillspector.is_available = True
        mock_tools.skillspector.run.return_value = ToolResult(
            success=True, stdout=json.dumps(report), stderr="", exit_code=0
        )

        result = SecurityValidator(use_llm=False).validate_security_only(root)

        assert not result.passed
        assert result.incomplete_scans == ["skillspector"]
        assert result.summary.high_count == 2
        assert result.metadata["skillspector_reference_summary"]["affected_artifacts"] == 1

    def test_occurrence_list_is_bounded(self) -> None:
        issue = _report(_PARSE_LIMIT_REPORT)["issues"][0]
        issue["occurrences"] = [{"file": "SKILL.md", "start_line": line} for line in range(1, 130)]

        finding, _is_error = SecurityValidator._convert_skillspector_issue(issue)

        assert len(finding.metadata["skillspector_occurrences"]) == 128
        assert finding.metadata["skillspector_occurrences_truncated"] is True

    def test_secret_shaped_text_is_redacted(self) -> None:
        issue = _report()["issues"][0]
        secret = "nvapi-" + "examplecredential" * 3
        issue["evidence"] = {
            "target_path": "scripts/helper.pl",
            "target_disposition": "partial",
            "reasons": [{"reason_code": "static_parse_limit", "message": secret, "analyzers": [secret]}],
        }

        finding, is_error = SecurityValidator._convert_skillspector_issue(issue)

        assert is_error
        assert secret not in json.dumps(finding.metadata)
        assert secret not in finding.suggestion
        assert "nvapi-<redacted>" in json.dumps(finding.metadata)

    @pytest.mark.parametrize(
        "target", ["/outside.pl", "../outside.pl", "scripts/../../outside.pl", "C:/outside.pl", "scripts/\x00helper.pl"]
    )
    def test_structured_target_outside_the_scan_is_dropped(self, target: str) -> None:
        issue = _report()["issues"][0]
        issue["finding"] = "malformed finding"
        issue["evidence"] = {"target_path": target, "target_disposition": "partial", "reasons": []}

        finding, is_error = SecurityValidator._convert_skillspector_issue(issue)

        assert is_error
        assert "skillspector_evidence" not in finding.metadata

    @pytest.mark.parametrize("disposition", [None, "analyzed", "PARTIAL", ["partial"]])
    def test_unknown_disposition_is_dropped(self, disposition: object) -> None:
        issue = _report(_PARSE_LIMIT_REPORT)["issues"][0]
        issue["evidence"]["target_disposition"] = disposition

        finding, _is_error = SecurityValidator._convert_skillspector_issue(issue)

        assert "skillspector_evidence" not in finding.metadata

    def test_absolute_paths_count_only_under_the_scan_root(self, tmp_path: Path) -> None:
        root = _helper_skill(tmp_path / "skill")
        issue = _report(_PARSE_LIMIT_REPORT)["issues"][0]
        issue["evidence"]["target_path"] = f"{root.as_posix()}/scripts/helper.pl"
        issue["occurrences"] = [
            {"file": f"{root.as_posix()}/SKILL.md", "start_line": 6},
            {"file": f"{tmp_path.as_posix()}/SKILL.md", "start_line": 8},
        ]

        finding, _is_error = SecurityValidator._convert_skillspector_issue(issue, scan_root=root)

        assert finding.metadata["skillspector_evidence"]["target_path"] == "scripts/helper.pl"
        assert finding.metadata["skillspector_occurrences"] == [{"file": "SKILL.md", "start_line": 6}]
        assert report_path(f"{root.as_posix()}/scripts/helper.pl") is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("scripts/helper.pl", "scripts/helper.pl"),
            ("./scripts//helper.pl", "scripts/helper.pl"),
            ("scripts\\helper.pl", "scripts/helper.pl"),
            ("scripts/../helper.pl", "helper.pl"),
            ("..", None),
            (".", None),
            ("", None),
            (["scripts/helper.pl"], None),
        ],
    )
    def test_report_path_is_lexical_and_contained(self, value: object, expected: str | None) -> None:
        assert report_path(value) == expected


class TestLegacyLedgerReasons:
    """Before v2.12.0 the reasons are only in the target's ledger rows."""

    @pytest.mark.parametrize("target", ["../outside.pl", "/outside.pl", "scripts/\x00helper.pl", "scripts/other.pl"])
    def test_ledger_rows_must_match_an_in_scope_target(self, target: str) -> None:
        report = _report()
        issue = report["issues"][0]
        issue["finding"] = f"{target} (partial)"

        finding, is_error = SecurityValidator._convert_skillspector_issue(
            issue, completeness=report["analysis_completeness"]
        )

        assert is_error
        assert "skillspector_evidence" not in finding.metadata
        assert finding.suggestion == f"{issue['explanation']} {issue['remediation']}"

    def test_source_location_must_be_in_scope(self) -> None:
        report = _report()
        issue = report["issues"][0]
        issue["location"] = {"file": "../outside.md", "start_line": 5}

        finding, _is_error = SecurityValidator._convert_skillspector_issue(
            issue, completeness=report["analysis_completeness"]
        )

        assert "skillspector_evidence" not in finding.metadata

    def test_without_the_ledger_the_legacy_finding_is_unchanged(self) -> None:
        issue = _report()["issues"][0]

        finding, _is_error = SecurityValidator._convert_skillspector_issue(issue)

        assert finding.check_name == _CHECK_NAME
        assert "skillspector_evidence" not in finding.metadata
        assert finding.suggestion == f"{issue['explanation']} {issue['remediation']}"

    @pytest.mark.parametrize("structured", [False, True])
    @pytest.mark.parametrize("reason_code", ["read_error", "file_disappeared"])
    def test_read_failures_get_a_read_remedy(self, structured: bool, reason_code: str) -> None:
        report = _report()
        issue = report["issues"][0]
        row = report["analysis_completeness"]["ledger_exceptions"][0]
        row.update({"reason_code": reason_code, "message": "Artifact could not be read."})
        if structured:
            issue["evidence"] = {"target_path": "scripts/helper.pl", "target_disposition": "partial", "reasons": [row]}

        finding, is_error = SecurityValidator._convert_skillspector_issue(
            issue, completeness=report["analysis_completeness"]
        )

        assert is_error
        assert finding.severity == Severity.HIGH
        assert f"{reason_code} (static_patterns_tool_misuse): Artifact could not be read." in finding.suggestion
        assert "readable where the scan runs" in finding.suggestion
        assert "minimal reproducer" not in finding.suggestion

    @pytest.mark.parametrize("target", [["not a string"], {"unexpected": "object"}, True, "bad\npath"])
    @pytest.mark.parametrize("reasons", [[], [{"reason_code": "unverified_reason"}]])
    def test_unreadable_structured_target_needs_the_ledger(self, target: object, reasons: list) -> None:
        issue = _report()["issues"][0]
        issue["evidence"] = {"target_path": target, "target_disposition": "partial", "reasons": reasons}

        finding, is_error = SecurityValidator._convert_skillspector_issue(issue)

        assert is_error
        assert "skillspector_evidence" not in finding.metadata

    def test_unreadable_structured_target_uses_only_the_ledger_reasons(self) -> None:
        report = _report()
        issue = report["issues"][0]
        issue["evidence"] = {
            "target_path": ["not a string"],
            "target_disposition": "partial",
            "reasons": [{"reason_code": "unverified_reason"}],
        }

        finding, _is_error = SecurityValidator._convert_skillspector_issue(
            issue, completeness=report["analysis_completeness"]
        )

        assert "static_parse_limit" in finding.suggestion
        assert "unverified_reason" not in finding.suggestion

    def test_ledger_is_read_once_and_bounded(self) -> None:
        issue = _report()["issues"][0]
        issues = [issue] * 100
        completeness = {
            "ledger_exceptions": [
                {"path": "scripts/helper.pl", "reason_code": "static_parse_limit"},
                {"path": "scripts/unrelated.pl", "reason_code": "read_error"},
            ]
            * 1000
        }
        calls = 0

        def normalize(path: object) -> str | None:
            nonlocal calls
            calls += 1
            return report_path(path)

        index = index_legacy_reference_reasons(issues, completeness=completeness, normalize_path=normalize)

        assert calls == 3  # The source, the target and the unrelated ledger path.
        assert set(index) == {"scripts/helper.pl"}
        assert len(index["scripts/helper.pl"]) == 17
        for repeated in issues:
            evidence = issue_diagnostics(
                repeated, completeness=completeness, normalize_path=normalize, legacy_reason_index=index
            )["skillspector_evidence"]
            assert evidence["reasons"] == [{"reason_code": "static_parse_limit"}]
            assert evidence["reasons_truncated"] is True
        # Each finding normalizes its own paths, never the ledger again.
        assert calls <= 4 * len(issues) + 3


class TestLargeReports:
    def test_each_distinct_path_and_text_is_normalized_once_per_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Findings that repeat the same absolute reference lines and reasons redact each text once."""
        root = _parse_limit_skill(tmp_path / "skill")
        report = _report(_PARSE_LIMIT_REPORT)
        template = report["issues"][0]
        template["occurrences"] = [
            {"file": f"{root.as_posix()}/SKILL.md", "start_line": line} for line in range(1, 129)
        ]
        reason = template["evidence"]["reasons"][0]
        template["evidence"]["reasons"] = [{**reason, "start_line": line} for line in range(1, 17)]
        report["issues"] = []
        for index in range(50):
            issue = copy.deepcopy(template)
            issue["evidence"]["target_path"] = f"scripts/helper{index}.pl"
            report["issues"].append(issue)
        redacted: list[str] = []
        prefix_lookups = 0
        redact, root_prefixes = skillspector_references.redact_sensitive_text, skillspector_references._root_prefixes

        def counting_redact(value: str) -> str:
            redacted.append(value)
            return redact(value)

        def counting_root_prefixes(scan_root: Path | None) -> list:
            nonlocal prefix_lookups
            prefix_lookups += 1
            return root_prefixes(scan_root)

        monkeypatch.setattr(skillspector_references, "redact_sensitive_text", counting_redact)
        monkeypatch.setattr(skillspector_references, "_root_prefixes", counting_root_prefixes)

        result = _process(report, source_root=root)

        findings = _ae1(result)
        assert len(findings) == 50
        for finding in findings:
            assert len(finding.metadata["skillspector_occurrences"]) == 128
            assert len(finding.metadata["skillspector_evidence"]["reasons"]) == 16
        assert result.metadata["skillspector_reference_summary"]["affected_artifacts"] == 50
        assert [text for text, count in Counter(redacted).items() if count > 1] == []
        assert prefix_lookups == 1


class TestTruncatedReasons:
    @pytest.mark.parametrize("structured", [False, True])
    @pytest.mark.parametrize(
        ("visible_reason", "hidden_reason"),
        [("static_parse_limit", "read_error"), ("read_error", "static_parse_limit")],
    )
    def test_no_single_remedy_is_claimed(self, structured: bool, visible_reason: str, hidden_reason: str) -> None:
        report = _report()
        row = report["analysis_completeness"]["ledger_exceptions"][0]
        rows = [
            {**row, "reason_code": visible_reason, "message": "Visible analysis limitation.", "start_line": line}
            for line in range(1, 17)
        ]
        rows.append({**row, "reason_code": hidden_reason, "message": "Hidden analysis limitation.", "start_line": 17})
        report["analysis_completeness"]["ledger_exceptions"] = rows
        if structured:
            for issue in report["issues"]:
                issue["evidence"] = {
                    "target_path": "scripts/helper.pl",
                    "target_disposition": "partial",
                    "reasons": rows[:16],
                    "reasons_truncated": True,
                }

        result = _process(report)

        findings = _ae1(result)
        assert len(findings) == 2
        for finding in findings:
            assert finding.severity == Severity.HIGH
            assert finding.metadata["skillspector_evidence"]["reasons_truncated"] is True
            assert visible_reason in finding.suggestion
            assert hidden_reason not in finding.suggestion
            assert "SkillSpector reported more reasons than are shown here" in finding.suggestion
            assert "`analysis_completeness.ledger_exceptions`" in finding.suggestion
            assert finding.suggestion.endswith("Resolve the reported analysis limitation and rerun the scan.")

    def test_the_scanner_remedy_is_kept(self) -> None:
        issue = _report()["issues"][0]
        issue["evidence"] = {
            "target_path": "scripts/helper.pl",
            "target_disposition": "partial",
            "reasons": [{"reason_code": "static_parse_limit"}],
            "reasons_truncated": True,
        }
        issue["remediation"] = "Review the expression boundary and rerun after addressing every ledger exception."

        finding, _is_error = SecurityValidator._convert_skillspector_issue(issue)

        assert finding.suggestion.endswith(issue["remediation"])
        assert "SkillSpector reported more reasons than are shown here" in finding.suggestion


class TestSummaryRecord:
    def test_a_later_record_replaces_the_count(self) -> None:
        result = _process(_report(_HOST_LANGUAGE_REPORT))
        result.findings = [
            finding
            for finding in result.findings
            if finding.metadata.get("skillspector_evidence", {}).get("target_path") == "scripts/render_notes.py"
        ]

        _record_skillspector_reference_summary(result)

        assert result.metadata["skillspector_reference_summary"] == {
            "affected_artifacts": 1,
            "reference_occurrences": 2,
            "reference_occurrences_truncated": False,
        }

        result.findings = []
        _record_skillspector_reference_summary(result)

        assert "skillspector_reference_summary" not in result.metadata
        assert _count_messages(result) == []

    def test_a_finding_without_reasons_counts_without_claiming_one(self) -> None:
        """A named file with no reasons and no ledger rows is counted; the line claims no reason for it."""
        report = _report(_PARSE_LIMIT_REPORT)
        for issue in report["issues"]:
            if issue["id"] == "AE1":
                issue["evidence"] = {
                    "target_path": "scripts/helper.pl",
                    "target_disposition": "uninspected",
                    "reasons": [],
                }
        report["analysis_completeness"]["ledger_exceptions"] = []

        result = _process(report)

        findings = _ae1(result)
        assert len(findings) == 2
        for finding in findings:
            assert finding.metadata["skillspector_evidence"]["reasons"] == []
            assert finding.suggestion.startswith(
                "Referenced artifact scripts/helper.pl was not completely inspected (uninspected). Review "
            )
        assert reference_summary_message(result.metadata["skillspector_reference_summary"]) == (
            "skillspector could not completely inspect 1 referenced artifact(s) across 2 reference occurrence(s); "
            "each AE1 finding names the artifact and any reasons SkillSpector gave"
        )

    def test_reports_without_ae1_evidence_have_no_summary(self) -> None:
        report = _report()
        for issue in report["issues"]:
            issue["finding"] = "malformed finding"

        result = _process(report)

        assert len(_ae1(result)) == 2
        assert "skillspector_reference_summary" not in result.metadata
        assert _count_messages(result) == []
