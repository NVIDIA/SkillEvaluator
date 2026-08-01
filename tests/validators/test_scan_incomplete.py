# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External scanner failures are explicit, non-green incomplete scans."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.models import ValidationResult
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.secrets import SecretsValidator
from skillevaluator.validators.security import SecurityValidator

TIMEOUT_RESULT = ToolResult(success=False, stdout="", stderr="", exit_code=-1, error_message="tool timed out")
CRASH_RESULT = ToolResult(success=True, stdout="", stderr="usage error", exit_code=2)


@pytest.fixture
def skill_dir(tmp_path) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: skill\ndescription: d\n---\n")
    (skill / "code.py").write_text("x = 1\n")
    return skill


class TestScanIncompleteMarking:
    @patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector")
    @patch.object(Tools.skillspector, "run", return_value=TIMEOUT_RESULT)
    def test_skillspector_timeout_marks_incomplete(self, _mock_run, skill_dir):
        result = SecurityValidator()._run_skillspector(skill_dir)
        assert result.metadata["incomplete_scans"] == ["skillspector"]
        assert result.status == "incomplete"
        assert not result.passed

    @patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
    @patch.object(Tools.bandit, "run", return_value=TIMEOUT_RESULT)
    def test_bandit_timeout_marks_incomplete(self, _mock_run, skill_dir):
        result = CodeRiskValidator()._run_bandit(skill_dir)
        assert result.metadata["incomplete_scans"] == ["bandit"]
        assert result.status == "incomplete"
        assert not result.passed

    @patch.object(Tools.semgrep, "_path", "/usr/bin/semgrep")
    @patch.object(Tools.semgrep, "run", return_value=CRASH_RESULT)
    def test_semgrep_crash_marks_incomplete(self, _mock_run, skill_dir):
        result = CodeRiskValidator()._run_semgrep(skill_dir)
        assert result.metadata["incomplete_scans"] == ["semgrep"]
        assert result.status == "incomplete"
        assert not result.passed

    @patch.object(Tools.gitleaks, "_path", "/usr/bin/gitleaks")
    @patch.object(Tools.gitleaks, "run", return_value=TIMEOUT_RESULT)
    def test_gitleaks_timeout_marks_incomplete(self, _mock_run, skill_dir):
        result = SecretsValidator()._validate_single_skill(skill_dir)
        assert result.metadata["incomplete_scans"] == ["gitleaks"]
        assert result.status == "incomplete"
        assert not result.passed

    @pytest.mark.parametrize(
        ("tool_name", "run_scan"),
        [
            ("skillspector", lambda path: SecurityValidator()._run_skillspector(path)),
            ("bandit", lambda path: CodeRiskValidator()._run_bandit(path)),
            ("semgrep", lambda path: CodeRiskValidator()._run_semgrep(path)),
            ("gitleaks", lambda path: SecretsValidator()._validate_single_skill(path)),
        ],
    )
    def test_missing_scanner_is_incomplete(self, tool_name, run_scan, skill_dir):
        tool = getattr(Tools, tool_name)
        with patch.object(tool, "_path", None):
            result = run_scan(skill_dir)

        assert result.metadata["incomplete_scans"] == [tool_name]
        assert result.status == "incomplete"
        assert not result.passed

    @patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
    @patch.object(
        Tools.bandit,
        "run",
        return_value=ToolResult(success=True, stdout="not-json", stderr="", exit_code=0),
    )
    def test_bandit_malformed_report_is_incomplete(self, _mock_run, skill_dir):
        result = CodeRiskValidator()._run_bandit(skill_dir)

        assert result.metadata["incomplete_scans"] == ["bandit"]
        assert result.status == "incomplete"
        assert not any("No security issues" in message for message in result.messages)

    @patch.object(Tools.semgrep, "_path", "/usr/bin/semgrep")
    @patch.object(
        Tools.semgrep,
        "run",
        return_value=ToolResult(success=True, stdout="{}", stderr="", exit_code=0),
    )
    def test_semgrep_malformed_report_is_incomplete(self, _mock_run, skill_dir):
        result = CodeRiskValidator()._run_semgrep(skill_dir)

        assert result.metadata["incomplete_scans"] == ["semgrep"]
        assert result.status == "incomplete"

    @patch.object(Tools.gitleaks, "_path", "/usr/bin/gitleaks")
    @patch.object(
        Tools.gitleaks,
        "run",
        return_value=ToolResult(success=True, stdout="not-json", stderr="", exit_code=10),
    )
    def test_gitleaks_malformed_findings_report_is_incomplete(self, _mock_run, skill_dir):
        result = SecretsValidator()._validate_single_skill(skill_dir)

        assert result.metadata["incomplete_scans"] == ["gitleaks"]
        assert result.status == "incomplete"
        assert not any("No secrets detected" in message for message in result.messages)

    @patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector")
    @patch.object(
        Tools.skillspector,
        "run",
        return_value=ToolResult(success=True, stdout=json.dumps({"issues": []}), stderr="", exit_code=0),
    )
    def test_skillspector_missing_required_report_fields_is_incomplete(self, _mock_run, skill_dir):
        result = SecurityValidator()._run_skillspector(skill_dir)

        assert result.metadata["incomplete_scans"] == ["skillspector"]
        assert result.status == "incomplete"

    @patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector")
    @patch.object(Tools.skillspector, "run")
    def test_clean_scan_is_not_marked(self, mock_run, skill_dir):
        mock_run.return_value = ToolResult(
            success=True,
            stdout=json.dumps(
                {
                    "risk_assessment": {"score": 0, "severity": "LOW", "recommendation": "SAFE"},
                    "issues": [],
                    "metadata": {},
                }
            ),
            stderr="",
            exit_code=0,
        )
        result = SecurityValidator()._run_skillspector(skill_dir)
        assert "incomplete_scans" not in result.metadata


class TestMergeCombinesIncompleteScans:
    def test_merge_concatenates_tool_lists(self):
        first = ValidationResult()
        first.mark_scan_incomplete("bandit")
        second = ValidationResult()
        second.mark_scan_incomplete("semgrep")

        first.merge(second)

        assert first.metadata["incomplete_scans"] == ["bandit", "semgrep"]
        assert first.status == "incomplete"
        assert not first.passed

    def test_merge_with_prefix_preserves_incomplete_status(self):
        parent = ValidationResult()
        child = ValidationResult()
        child.mark_scan_incomplete("gitleaks")

        parent.merge_with_prefix(child, "example-skill")

        assert parent.metadata["incomplete_scans"] == ["gitleaks"]
        assert parent.status == "incomplete"
        assert not parent.passed

    def test_duplicate_marks_are_deduplicated(self):
        result = ValidationResult()

        result.mark_scan_incomplete("semgrep")
        result.mark_scan_incomplete("semgrep")

        assert result.metadata["incomplete_scans"] == ["semgrep"]

    def test_merge_without_marks_adds_no_key(self):
        first = ValidationResult()
        first.merge(ValidationResult())
        assert "incomplete_scans" not in first.metadata
