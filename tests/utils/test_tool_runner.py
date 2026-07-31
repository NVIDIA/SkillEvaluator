# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock unit tests for skillevaluator.utils.tool_runner module.

Tests ExternalTool, parse_json_output, create_temp_config, cvss_to_severity,
and Tools registry with mocked subprocess and filesystem where appropriate.
"""

import os
import sys
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import patch

import pytest

from skillevaluator.models.result import Severity
from skillevaluator.utils import tool_runner
from skillevaluator.utils.tool_runner import (
    CVSS_THRESHOLDS,
    ExternalTool,
    ToolResult,
    Tools,
    create_temp_config,
    cvss_to_severity,
    parse_json_output,
)


class TestCvssToSeverity:
    """Tests for cvss_to_severity function."""

    def test_critical_severity(self):
        """CVSS >= 9.0 maps to CRITICAL."""
        assert cvss_to_severity(9.0) == Severity.CRITICAL
        assert cvss_to_severity(10.0) == Severity.CRITICAL

    def test_high_severity(self):
        """CVSS 7.0-8.9 maps to HIGH."""
        assert cvss_to_severity(7.0) == Severity.HIGH
        assert cvss_to_severity(8.9) == Severity.HIGH

    def test_medium_severity(self):
        """CVSS 4.0-6.9 maps to MEDIUM."""
        assert cvss_to_severity(4.0) == Severity.MEDIUM
        assert cvss_to_severity(5.5) == Severity.MEDIUM

    def test_low_severity(self):
        """CVSS 0.1-3.9 maps to LOW."""
        assert cvss_to_severity(0.1) == Severity.LOW
        assert cvss_to_severity(2.0) == Severity.LOW


class TestParseJsonOutput:
    """Tests for parse_json_output function."""

    def test_valid_json_object(self):
        """Valid JSON object is parsed correctly."""
        output = '{"key": "value", "count": 42}'
        result = parse_json_output(output)
        assert result == {"key": "value", "count": 42}

    def test_valid_json_array(self):
        """Valid JSON array is parsed correctly."""
        output = "[1, 2, 3]"
        result = parse_json_output(output)
        assert result == [1, 2, 3]

    def test_empty_string_returns_none(self):
        """Empty or whitespace-only output returns None."""
        assert parse_json_output("") is None
        assert parse_json_output("   ") is None

    def test_invalid_json_returns_none(self):
        """Invalid JSON returns None."""
        assert parse_json_output("not json") is None
        assert parse_json_output("{invalid") is None

    def test_no_json_in_mixed_output_returns_none(self):
        """Mixed output with no valid JSON still returns None."""
        assert parse_json_output("line1\nline2\nline3") is None

    def test_on_error_fallback(self):
        """When on_error string is in output, returns empty dict for non-JSON."""
        output = "Error: something went wrong"
        result = parse_json_output(output, on_error="Error:")
        assert result == {}


class TestCreateTempConfig:
    """Tests for create_temp_config function."""

    def test_creates_file_with_content(self):
        """Creates temp file with correct content."""
        content = "[section]\nkey = value"
        path = create_temp_config(content)
        try:
            assert path.exists()
            assert path.read_text() == content
            assert path.suffix == ".toml"
        finally:
            path.unlink(missing_ok=True)

    def test_custom_suffix(self):
        """Custom suffix is applied."""
        path = create_temp_config("content", suffix=".yaml")
        try:
            assert path.suffix == ".yaml"
        finally:
            path.unlink(missing_ok=True)

    def test_file_is_in_temp_directory(self):
        """File is created in system temp directory."""
        path = create_temp_config("x")
        try:
            assert "skillevaluator_" in path.name
            assert path.parent == Path(path).parent
        finally:
            path.unlink(missing_ok=True)


class TestResolveToolPath:
    """Tests for resolve_tool_path (PATH + interpreter bin-dir fallback)."""

    def test_prefers_path_lookup(self):
        """A tool found on PATH is returned without consulting the fallback."""
        with patch("shutil.which", return_value="/usr/bin/tool") as mock_which:
            assert tool_runner.resolve_tool_path("tool") == "/usr/bin/tool"
            mock_which.assert_called_once_with("tool")

    def test_bundled_tool_prefers_interpreter_sibling_over_path(self, tmp_path):
        """Bundled scanners must come from the SkillEvaluator environment."""
        fake_interpreter = tmp_path / "bin" / "python"
        sibling = tmp_path / "bin" / "semgrep"

        def fake_which(command, *, path=None):
            if path == str(fake_interpreter.parent):
                return str(sibling)
            return "/usr/local/bin/semgrep"

        with patch.object(sys, "executable", str(fake_interpreter)), patch("shutil.which", side_effect=fake_which):
            resolved = tool_runner.resolve_tool_path("semgrep", prefer_interpreter_sibling=True)

        assert resolved == str(sibling)

    def test_external_tool_keeps_path_precedence(self, tmp_path):
        """External Gitleaks must remain user/PATH supplied, not sibling bound."""
        fake_interpreter = tmp_path / "bin" / "python"

        def fake_which(command, *, path=None):
            if path == str(fake_interpreter.parent):
                return str(tmp_path / "bin" / "gitleaks")
            return "/opt/homebrew/bin/gitleaks"

        with patch.object(sys, "executable", str(fake_interpreter)), patch("shutil.which", side_effect=fake_which):
            tool = ExternalTool("Gitleaks", "gitleaks")

        assert tool.path == "/opt/homebrew/bin/gitleaks"

    def test_bundled_external_tool_requests_sibling_first(self, tmp_path):
        fake_interpreter = tmp_path / "bin" / "python"
        sibling = tmp_path / "bin" / "bandit"

        def fake_which(command, *, path=None):
            if path == str(fake_interpreter.parent):
                return str(sibling)
            return "/usr/local/bin/bandit"

        with patch.object(sys, "executable", str(fake_interpreter)), patch("shutil.which", side_effect=fake_which):
            tool = ExternalTool("Bandit", "bandit", prefer_interpreter_sibling=True)

        assert tool.path == str(sibling)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable override")
    def test_bundled_tool_explicit_absolute_override_is_auditable(self, tmp_path, monkeypatch):
        override = tmp_path / "approved-semgrep"
        override.write_text("#!/bin/sh\nexit 0\n")
        override.chmod(0o755)
        monkeypatch.setenv("SKILLEVALUATOR_SEMGREP_PATH", str(override))

        tool = ExternalTool(
            "Semgrep",
            "semgrep",
            prefer_interpreter_sibling=True,
            override_env="SKILLEVALUATOR_SEMGREP_PATH",
        )

        assert tool.path == str(override.resolve())
        assert tool.resolution_source == "override:SKILLEVALUATOR_SEMGREP_PATH"

    def test_bundled_tool_rejects_relative_override(self, monkeypatch):
        monkeypatch.setenv("SKILLEVALUATOR_BANDIT_PATH", "./bandit")

        tool = ExternalTool(
            "Bandit",
            "bandit",
            prefer_interpreter_sibling=True,
            override_env="SKILLEVALUATOR_BANDIT_PATH",
        )

        assert not tool.is_available
        assert "absolute executable file" in tool.get_install_hint()

    def test_external_gitleaks_has_no_bundled_override(self):
        assert Tools.gitleaks.override_env is None

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake console script")
    def test_falls_back_to_interpreter_bin_dir(self, tmp_path):
        """A tool missing from PATH is found next to the running interpreter.

        This is the ``uv tool install`` / ``pipx`` layout: console scripts of
        bundled extras such as Bandit land in the tool venv's bin directory,
        which is not on the user's PATH.
        """
        script = tmp_path / "sk-test-scanner"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        fake_interpreter = tmp_path / "python"

        with patch.object(sys, "executable", str(fake_interpreter)):
            assert tool_runner.resolve_tool_path("sk-test-scanner") == str(script)

    def test_returns_none_when_missing_everywhere(self, tmp_path):
        """No PATH hit and no sibling script means None."""
        fake_interpreter = tmp_path / "python"
        with patch.object(sys, "executable", str(fake_interpreter)):
            assert tool_runner.resolve_tool_path("sk-test-absent") is None

    def test_handles_empty_sys_executable(self):
        """Embedded interpreters may report an empty sys.executable."""
        with patch.object(sys, "executable", ""):
            assert tool_runner.resolve_tool_path("sk-test-absent") is None

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake console script")
    def test_external_tool_uses_fallback(self, tmp_path):
        """ExternalTool resolves bundled console scripts via the fallback."""
        script = tmp_path / "sk-test-scanner"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        fake_interpreter = tmp_path / "python"

        with patch.object(sys, "executable", str(fake_interpreter)):
            tool = ExternalTool("Test Scanner", "sk-test-scanner")

        assert tool.is_available is True
        assert tool.path == str(script)


class TestExternalTool:
    """Tests for ExternalTool class."""

    def test_is_available_when_found(self):
        """is_available True when tool is in PATH."""
        with patch("shutil.which", return_value="/usr/bin/echo"):
            tool = ExternalTool("Echo", "echo")
            assert tool.is_available is True
            assert tool.path == "/usr/bin/echo"

    def test_is_available_when_not_found(self):
        """is_available False when tool not in PATH."""
        with patch("shutil.which", return_value=None):
            tool = ExternalTool("Nonexistent", "nonexistent")
            assert tool.is_available is False
            assert tool.path is None

    def test_run_returns_error_when_unavailable(self):
        """run returns ToolResult with error when tool not installed."""
        with patch("shutil.which", return_value=None):
            tool = ExternalTool("Missing", "missing")
            result = tool.run(["arg"])
            assert result.success is False
            assert result.exit_code == -1
            assert "not installed" in (result.error_message or "")

    def test_run_executes_when_available(self):
        """run executes command when tool is available."""
        mock_proc = CompletedProcess(args=["true"], returncode=0, stdout="", stderr="")
        with (
            patch("shutil.which", return_value="/usr/bin/true"),
            patch("skillevaluator.utils.tool_runner.subprocess.run", return_value=mock_proc),
        ):
            tool = ExternalTool("True", "true")
            result = tool.run([], log_command=False)
            assert result.success is True
            assert result.exit_code == 0
            assert result.error_message is None

    def test_timeout_preserves_partial_output(self):
        timeout = TimeoutExpired(
            cmd=["scanner"],
            timeout=1,
            output=b'{"partial": true}',
            stderr=b"still working",
        )
        with (
            patch("shutil.which", return_value="/usr/bin/scanner"),
            patch("skillevaluator.utils.tool_runner.subprocess.run", side_effect=timeout),
        ):
            result = ExternalTool("Scanner", "scanner").run([], log_command=False, timeout=1)

        assert result.success is False
        assert result.stdout == '{"partial": true}'
        assert result.stderr == "still working"
        assert "timed out" in (result.error_message or "")

    def test_run_merges_extra_env_over_ambient(self):
        """env additions are merged onto os.environ, not substituted for it."""
        mock_proc = CompletedProcess(args=["true"], returncode=0, stdout="", stderr="")
        with (
            patch("shutil.which", return_value="/usr/bin/true"),
            patch("skillevaluator.utils.tool_runner.subprocess.run", return_value=mock_proc) as mock_run,
        ):
            tool = ExternalTool("True", "true")
            tool.run([], log_command=False, env={"SKILLEVALUATOR_TEST_EXTRA": "1"})

        passed_env = mock_run.call_args.kwargs["env"]
        assert passed_env["SKILLEVALUATOR_TEST_EXTRA"] == "1"
        assert "PATH" in passed_env

    def test_run_can_replace_ambient_environment(self):
        """Sensitive subprocesses can opt into an exact invocation environment."""
        mock_proc = CompletedProcess(args=["true"], returncode=0, stdout="", stderr="")
        with (
            patch.dict(os.environ, {"UNRELATED_PARENT_SECRET": "must-not-leak"}, clear=False),
            patch("shutil.which", return_value="/usr/bin/true"),
            patch("skillevaluator.utils.tool_runner.subprocess.run", return_value=mock_proc) as mock_run,
        ):
            tool = ExternalTool("True", "true")
            tool.run([], log_command=False, env={"PATH": "/usr/bin", "SELECTED_KEY": "selected"}, replace_env=True)

        assert mock_run.call_args.kwargs["env"] == {"PATH": "/usr/bin", "SELECTED_KEY": "selected"}

    def test_run_without_env_leaves_subprocess_default(self):
        """No env argument means subprocess inherits the parent environment."""
        mock_proc = CompletedProcess(args=["true"], returncode=0, stdout="", stderr="")
        with (
            patch("shutil.which", return_value="/usr/bin/true"),
            patch("skillevaluator.utils.tool_runner.subprocess.run", return_value=mock_proc) as mock_run,
        ):
            ExternalTool("True", "true").run([], log_command=False)

        assert mock_run.call_args.kwargs.get("env") is None

    def test_get_install_hint_for_known_tools(self):
        """Known tools return specific install hints."""
        with patch("shutil.which", return_value=None):
            tool = ExternalTool("Gitleaks", "gitleaks")
            hint = tool.get_install_hint()
            assert "gitleaks" in hint.lower()

    def test_skillspector_install_hint_uses_public_source_distribution(self):
        with patch("shutil.which", return_value=None):
            tool = ExternalTool("SkillSpector", "skillspector")

        assert tool.get_install_hint() == "Install with: uv tool install git+https://github.com/NVIDIA/SkillSpector.git"
        assert "[security]" not in tool.get_install_hint()

    def test_bundled_scanner_hints_point_at_security_extra(self):
        """Scanners shipped via the security extra hint at reinstalling with extras.

        Bare ``pip install <scanner>`` is misleading for the documented
        install paths (uv tool / pipx), where the fix is reinstalling
        skillevaluator with the extras enabled.
        """
        with patch("shutil.which", return_value=None):
            for command in ("bandit", "pip-audit"):
                hint = ExternalTool(command, command).get_install_hint()
                assert "[all]" in hint or "[security]" in hint, f"{command}: {hint}"

    def test_semgrep_hint_uses_a_separate_tool_install(self):
        with patch("shutil.which", return_value=None):
            hint = ExternalTool("Semgrep", "semgrep").get_install_hint()

        assert "uv tool install semgrep" in hint
        assert "[security]" not in hint

    def test_optional_safety_hint_uses_a_separate_public_install(self):
        with patch("shutil.which", return_value=None):
            hint = ExternalTool("Safety", "safety").get_install_hint()

        assert hint == "Optional secondary scanner; install separately with: uv tool install safety"

    def test_skillevaluator_hint_does_not_reference_pypi(self):
        """skillevaluator is not published to PyPI; the hint must use the git source."""
        with patch("shutil.which", return_value=None):
            hint = ExternalTool("skillevaluator", "skillevaluator").get_install_hint()
        assert "git+https://github.com/NVIDIA/SkillEvaluator.git" in hint

    def test_get_install_hint_for_unknown_tools(self):
        """Unknown tools return generic hint."""
        with patch("shutil.which", return_value=None):
            tool = ExternalTool("CustomTool", "custom-tool")
            hint = tool.get_install_hint()
            assert "Install" in hint
            assert "PATH" in hint


class TestToolsRegistry:
    """Tests for Tools class (tool registry)."""

    def test_cvss_thresholds_match_severity_levels(self):
        """CVSS_THRESHOLDS contains all severity levels used by cvss_to_severity."""
        assert Severity.CRITICAL in CVSS_THRESHOLDS
        assert Severity.HIGH in CVSS_THRESHOLDS
        assert Severity.MEDIUM in CVSS_THRESHOLDS
        assert Severity.LOW in CVSS_THRESHOLDS


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """ToolResult captures successful execution."""
        result = ToolResult(
            success=True,
            stdout="output",
            stderr="",
            exit_code=0,
        )
        assert result.success is True
        assert result.stdout == "output"
        assert result.exit_code == 0
        assert result.error_message is None

    def test_failure_result_with_error_message(self):
        """ToolResult captures failed execution with error message."""
        result = ToolResult(
            success=False,
            stdout="",
            stderr="error",
            exit_code=1,
            error_message="Tool failed",
        )
        assert result.success is False
        assert result.error_message == "Tool failed"
