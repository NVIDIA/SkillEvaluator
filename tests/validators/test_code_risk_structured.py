# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CodeRiskValidator structured findings output.

Validates that _report_bandit_issue and _report_semgrep_finding use
add_structured_finding to populate result.findings with Finding objects.
"""

import json
from pathlib import Path
from unittest.mock import patch

from skillevaluator.utils.tool_runner import Severity, ToolResult, Tools
from skillevaluator.validators.code_risk import CodeRiskValidator

# =============================================================================
# TEST DATA
# =============================================================================

BANDIT_OUTPUT = json.dumps(
    {
        "results": [
            {
                "issue_severity": "HIGH",
                "issue_confidence": "HIGH",
                "issue_cwe": {"id": 78},
                "issue_text": "subprocess call with shell=True identified, security issue.",
                "test_id": "B602",
                "test_name": "subprocess_popen_with_shell_equals_true",
                "filename": "/tmp/test/bad.py",
                "line_number": 6,
            },
            {
                "issue_severity": "LOW",
                "issue_confidence": "MEDIUM",
                "issue_cwe": {"id": 259},
                "issue_text": "Possible hardcoded password: 'SuperSecret123'",
                "test_id": "B105",
                "test_name": "hardcoded_password_string",
                "filename": "/tmp/test/bad.py",
                "line_number": 9,
            },
        ]
    }
)

SEMGREP_OUTPUT = json.dumps(
    {
        "results": [
            {
                "check_id": "python.lang.security.audit.subprocess-shell-true",
                "path": "/tmp/test/bad.py",
                "start": {"line": 6},
                "extra": {
                    "severity": "ERROR",
                    "message": "subprocess call with shell=True",
                    "metadata": {
                        "cwe": ["CWE-78"],
                        "owasp": ["A1:2017"],
                    },
                },
            },
        ],
        "errors": [],
    }
)


# =============================================================================
# BANDIT STRUCTURED FINDINGS
# =============================================================================


class TestBanditStructuredFindings:
    @patch.object(Tools.bandit, "_path", "/usr/bin/bandit")
    @patch.object(Tools.bandit, "run")
    def test_run_bandit_end_to_end(self, mock_run):
        mock_run.return_value = ToolResult(
            success=True,
            stdout=BANDIT_OUTPUT,
            stderr="",
            exit_code=1,
        )

        v = CodeRiskValidator(fail_on_low=True)
        result = v._run_bandit(Path("/tmp/test"))

        assert len(result.findings) == 2
        assert result.findings[0].file_path == "/tmp/test/bad.py"
        assert result.findings[0].severity == Severity.HIGH
        assert result.findings[0].check_name == "B602:subprocess_popen_with_shell_equals_true"
        assert result.findings[1].file_path == "/tmp/test/bad.py"
        assert result.findings[1].severity == Severity.LOW
        assert result.findings[1].check_name == "B105:hardcoded_password_string"


# =============================================================================
# SEMGREP STRUCTURED FINDINGS
# =============================================================================


class TestSemgrepStructuredFindings:
    @patch.object(Tools.semgrep, "_path", "/usr/bin/semgrep")
    @patch.object(Tools.semgrep, "run")
    def test_run_semgrep_end_to_end(self, mock_run):
        mock_run.return_value = ToolResult(
            success=True,
            stdout=SEMGREP_OUTPUT,
            stderr="",
            exit_code=0,
        )

        v = CodeRiskValidator()
        result = v._run_semgrep(Path("/tmp/test"))

        assert len(result.findings) == 1
        assert result.findings[0].category == "SEMGREP"
        assert result.findings[0].file_path == "/tmp/test/bad.py"
        assert result.findings[0].severity == Severity.HIGH
        assert result.findings[0].check_name == "python.lang.security.audit.subprocess-shell-true"
        assert result.findings[0].line_number == 6

    @patch.object(Tools.semgrep, "_path", "/usr/bin/semgrep")
    @patch.object(Tools.semgrep, "run")
    def test_run_semgrep_excludes_uppercase_generated_benchmark(self, mock_run):
        mock_run.return_value = ToolResult(
            success=True,
            stdout=json.dumps({"results": [], "errors": []}),
            stderr="",
            exit_code=0,
        )

        v = CodeRiskValidator()
        v._run_semgrep(Path("/tmp/test"))

        args = mock_run.call_args.args[0]
        exclude_values = [args[i + 1] for i, arg in enumerate(args) if arg == "--exclude"]
        assert "benchmark.md" in exclude_values
        assert "BENCHMARK.md" in exclude_values
        assert "benchmarks.md" not in exclude_values
