# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SecretsValidator structured findings output.

Validates that _process_gitleaks_output uses add_structured_finding
to populate result.findings with Finding objects.
"""

import json
from pathlib import Path
from unittest.mock import patch

from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter, JSONReporter, MarkdownReporter
from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.secrets import SecretsValidator

# =============================================================================
# TEST DATA
# =============================================================================

AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
NVIDIA_KEY = "nvapi-" + "1234567890abcdef"

GITLEAKS_OUTPUT = json.dumps(
    [
        {
            "Description": "Generic API Key",
            "File": "/tmp/test/config.py",
            "StartLine": 5,
            "EndLine": 5,
            "Match": f"api_key = '{AWS_KEY}'",
            "Secret": AWS_KEY,
            "RuleID": "generic-api-key",
            "Tags": ["key", "api"],
        },
        {
            "Description": "NVIDIA NGC API Key",
            "File": "/tmp/test/config.py",
            "StartLine": 8,
            "EndLine": 8,
            "Match": f"ngc_key = '{NVIDIA_KEY}'",
            "Secret": NVIDIA_KEY,
            "RuleID": "nvidia-ngc-api-key",
            "Tags": ["key", "nvidia"],
        },
    ]
)


# =============================================================================
# SECRETS STRUCTURED FINDINGS
# =============================================================================


class TestSecretsStructuredFindings:
    @patch.object(Tools.gitleaks, "_path", "/usr/bin/gitleaks")
    @patch.object(Tools.gitleaks, "run")
    def test_run_gitleaks_end_to_end(self, mock_run):
        mock_run.return_value = ToolResult(
            success=True,
            stdout=GITLEAKS_OUTPUT,
            stderr="",
            exit_code=10,
        )

        v = SecretsValidator()
        result = v._validate_single_skill(Path("/tmp/test"))

        assert len(result.findings) == 2
        assert result.findings[0].check_name == "generic-api-key"
        assert result.findings[0].file_path == "/tmp/test/config.py"
        assert result.findings[0].line_number == 5
        assert result.findings[0].category == "SECRET"
        assert result.findings[1].check_name == "nvidia-ngc-api-key"
        assert result.findings[1].file_path == "/tmp/test/config.py"
        assert result.findings[1].line_number == 8
        assert result.findings[1].category == "SECRET"

        rendered = [
            *result.errors,
            *result.warnings,
            *result.messages,
            CLIReporter().render_all([result]),
            JSONReporter(include_timestamp=False).render_all([result]),
            MarkdownReporter(include_timestamp=False).render_all([result]),
            HTMLReporter(include_timestamp=False).render_all([result]),
            BenchmarkReporter(include_timestamp=False).render_all([result]),
        ]
        assert AWS_KEY not in "\n".join(rendered)
        assert NVIDIA_KEY not in "\n".join(rendered)

    @patch.object(Tools.gitleaks, "_path", "/usr/bin/gitleaks")
    @patch.object(Tools.gitleaks, "run")
    def test_unexpected_exit_does_not_echo_scanner_stdout(self, mock_run):
        mock_run.return_value = ToolResult(
            success=True,
            stdout=GITLEAKS_OUTPUT,
            stderr="",
            exit_code=2,
        )

        result = SecretsValidator()._validate_single_skill(Path("/tmp/test"))

        rendered = "\n".join([*result.errors, *result.warnings, *result.messages])
        assert AWS_KEY not in rendered
        assert NVIDIA_KEY not in rendered
        assert "redacted" in rendered.lower()
