# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock unit tests for skillevaluator.utils.tool_runner module.

Tests ExternalTool, parse_json_output, create_temp_config, cvss_to_severity,
and Tools registry with mocked subprocess and filesystem where appropriate.
"""

import importlib.util
import locale
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

    @staticmethod
    def _python_child(monkeypatch: pytest.MonkeyPatch, codec: str, stdout: bytes, stderr: bytes) -> ToolResult:
        """Run a real Python child that writes raw bytes, with ``codec`` as the selected text codec."""
        if sys.flags.utf8_mode and codec != "utf-8":
            pytest.skip("UTF-8 mode fixes the subprocess text codec to UTF-8")
        monkeypatch.setattr(locale, "getencoding", lambda: codec)
        monkeypatch.setenv("SKILLEVAL_TEST_PYTHON", sys.executable)
        code = (
            "import sys;"
            f"sys.stdout.buffer.write(bytes.fromhex('{stdout.hex()}'));"
            f"sys.stderr.buffer.write(bytes.fromhex('{stderr.hex()}'))"
        )
        tool = ExternalTool("Python", "python", override_env="SKILLEVAL_TEST_PYTHON")
        return tool.run(["-I", "-c", code], log_command=False)

    def test_run_decodes_non_utf8_output_tolerantly(self, monkeypatch: pytest.MonkeyPatch):
        """A child emitting bytes outside the selected codec must not lose its captured streams.

        A strict decode raises UnicodeDecodeError inside subprocess (in the
        reader thread on Windows, leaving ``proc.stdout``/``proc.stderr`` as
        None). Callers such as the SkillSpector wrapper then crash on
        ``stderr.casefold()`` instead of reading the scanner's diagnostics.
        """
        result = self._python_child(
            monkeypatch, "utf-8", stdout=b'{"ok": true}', stderr=b"WARNING path \xa1 not utf-8\r\n"
        )

        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == '{"ok": true}'
        assert result.stderr == "WARNING path � not utf-8\n"

    def test_run_keeps_the_selected_text_codec(self, monkeypatch: pytest.MonkeyPatch):
        """Output in the parent's selected codec (a non-UTF-8 locale) keeps its non-ASCII paths."""
        path = "café/中文.py"
        result = self._python_child(
            monkeypatch,
            "gb18030",
            stdout=f'{{"file": "{path}"}}'.encode("gb18030"),
            stderr=f"warning: {path}\n".encode("gb18030"),
        )

        assert result.stdout == f'{{"file": "{path}"}}'
        assert result.stderr == f"warning: {path}\n"

    def test_run_decodes_text_with_replacement(self):
        """subprocess.run keeps text mode but replaces undecodable bytes."""
        mock_proc = CompletedProcess(args=["scanner"], returncode=0, stdout="out", stderr="err")
        with (
            patch("shutil.which", return_value="/usr/bin/scanner"),
            patch("skillevaluator.utils.tool_runner.subprocess.run", return_value=mock_proc) as mock_run,
        ):
            result = ExternalTool("Scanner", "scanner").run([], log_command=False)

        assert mock_run.call_args.kwargs.get("text") is True
        assert mock_run.call_args.kwargs.get("errors") == "replace"
        assert (result.stdout, result.stderr) == ("out", "err")

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

    def test_skillspector_install_hint_uses_release_wheel(self):
        with patch("shutil.which", return_value=None):
            tool = ExternalTool("SkillSpector", "skillspector")

        assert (
            tool.get_install_hint() == "Install with: uv tool install "
            "https://github.com/NVIDIA/SkillSpector/releases/download/v2.13.4/skillspector-2.13.4-py3-none-any.whl"
        )
        assert "[security]" not in tool.get_install_hint()

    def test_bundled_scanner_hints_point_at_reinstalling_skillevaluator(self):
        """Bundled scanners hint at reinstalling SkillEvaluator, not a bare pip install.

        Bandit is a core dependency, so the fix needs no extra; pip-audit
        ships with the security extra.
        """
        with patch("shutil.which", return_value=None):
            bandit_hint = ExternalTool("bandit", "bandit").get_install_hint()
            pip_audit_hint = ExternalTool("pip-audit", "pip-audit").get_install_hint()

        assert "uv tool install --reinstall" in bandit_hint
        assert "[security]" not in bandit_hint and "[all]" not in bandit_hint
        assert "[all]" in pip_audit_hint or "[security]" in pip_audit_hint

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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake console scripts")
class TestAutomaticSkillSpectorVersion:
    """Automatic resolution must not silently change the pinned scanner release."""

    @pytest.fixture
    def scanner_bin(self, tmp_path, monkeypatch):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        monkeypatch.setenv("PATH", str(bindir))
        monkeypatch.setenv("SKILLEVALUATOR_RUNTIME_DIR", str(tmp_path / "runtime"))
        monkeypatch.delenv("SKILLEVALUATOR_SKILLSPECTOR_PATH", raising=False)
        monkeypatch.setattr(sys, "executable", str(bindir / "python"))
        # Provisioning has its own checksum/download tests. Keep selection offline.
        monkeypatch.setattr(tool_runner, "ensure_skillspector_wheel", lambda: None)
        monkeypatch.setattr(tool_runner, "_packaged_uv", lambda: None)
        uv = bindir / "uv"
        uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        uv.chmod(0o755)
        return bindir

    @staticmethod
    def _scanner(bindir, version_output, version_exit=0):
        scanner = bindir / "skillspector"
        scanner.write_text(
            '#!/bin/sh\nif [ "$1" = "--version" ]; then\n'
            f"  printf '%s\\n' '{version_output}'\n  exit {version_exit}\n"
            "fi\nprintf 'direct scanner\\n'\n"
        )
        scanner.chmod(0o755)
        return scanner

    @pytest.mark.parametrize(
        ("version_output", "version_exit"),
        [
            ("SkillSpector v2.11.2", 0),
            ("SkillSpector v2.14.0", 0),
            ("version unavailable", 0),
            ("SkillSpector v2.13.4", 2),
            ("SkillSpector v2.13.4rc1", 0),
            ("SkillSpector v2.13.4_rc1", 0),
            ("Dependency foo 2.13.4 initialized; scanner version unavailable", 0),
            ("SkillSpector v2.13.4\nSkillSpector v2.11.2", 0),
        ],
    )
    def test_mismatched_or_unverified_version_runs_pinned_wheel(
        self, scanner_bin, version_output, version_exit, caplog, monkeypatch
    ):
        scanner = self._scanner(scanner_bin, version_output, version_exit)
        monkeypatch.setattr(tool_runner.logger, "propagate", True)

        tool = ExternalTool("SkillSpector", "skillspector", managed=True)
        result = tool.run(["scan", "target"], log_command=False)

        assert result.success
        assert result.exit_code == 0
        assert result.stdout.splitlines() == [
            "tool",
            "run",
            "--quiet",
            "--from",
            str(scanner_bin.parent / "runtime/skillspector/2.13.4/skillspector-2.13.4-py3-none-any.whl"),
            "skillspector",
            "scan",
            "target",
        ]
        assert tool.path == str(scanner_bin / "uv")
        assert tool.version == "2.13.4"
        assert tool.resolution_source == "uv tool run"
        assert str(scanner) in caplog.text
        assert "2.13.4" in caplog.text

    def test_constructor_refresh_and_path_lookup_do_not_launch_scanner(self, scanner_bin):
        scanner = self._scanner(scanner_bin, "SkillSpector v2.13.4")
        marker = scanner_bin / "launched"
        scanner.write_text(scanner.read_text().replace("#!/bin/sh\n", f"#!/bin/sh\n: > '{marker}'\n"))

        tool = ExternalTool("SkillSpector", "skillspector", managed=True)
        tool.refresh()

        assert tool.path == str(scanner)
        assert tool.is_available
        assert not marker.exists()
        assert tool.run(["scan", "target"], log_command=False).success
        assert marker.exists()

    def test_fresh_module_import_does_not_launch_scanner(self, scanner_bin, monkeypatch):
        self._scanner(scanner_bin, "SkillSpector v2.13.4")
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            return CompletedProcess(argv, 0, "SkillSpector v2.13.4", "")

        monkeypatch.setattr(tool_runner.subprocess, "run", run)
        name = "skillevaluator.utils._tool_runner_import_test"
        spec = importlib.util.spec_from_file_location(name, tool_runner.__file__)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)

        spec.loader.exec_module(module)

        assert module.Tools.skillspector.path == str(scanner_bin / "skillspector")
        assert calls == []

    def test_lazy_probe_uses_sanitized_environment_and_caller_timeout(self, scanner_bin, monkeypatch):
        self._scanner(scanner_bin, "SkillSpector v2.13.4")
        monkeypatch.setenv("NVIDIA_API_KEY", "ambient-secret")
        real_run = tool_runner.subprocess.run
        calls = []

        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            return real_run(argv, **kwargs)

        monkeypatch.setattr(tool_runner.subprocess, "run", run)
        tool = ExternalTool("SkillSpector", "skillspector", managed=True)
        result = tool.run(
            ["scan", "target"],
            timeout=5,
            env={"PATH": str(scanner_bin), "NVIDIA_API_KEY": "requested-scan-secret"},
            replace_env=True,
            log_command=False,
        )

        assert result.success
        assert calls[0][0] == [str(scanner_bin / "skillspector"), "--version"]
        assert 0 < calls[0][1]["timeout"] <= 5
        assert calls[0][1]["env"] == {"PATH": str(scanner_bin)}
        assert calls[-1][1]["env"]["NVIDIA_API_KEY"] == "requested-scan-secret"

    def test_matching_pinned_version_uses_installed_scanner(self, scanner_bin):
        scanner = self._scanner(scanner_bin, "SkillSpector v2.13.4")

        tool = ExternalTool("SkillSpector", "skillspector", managed=True)

        assert tool.run(["scan", "target"], log_command=False).stdout == "direct scanner\n"
        assert tool.path == str(scanner)
        assert tool.version == "2.13.4"
        assert tool.resolution_source == "automatic"

    def test_stale_scanner_falls_back_to_packaged_uv(self, scanner_bin, monkeypatch):
        self._scanner(scanner_bin, "SkillSpector v2.11.2")
        packaged_uv = scanner_bin / "packaged-uv"
        (scanner_bin / "uv").rename(packaged_uv)
        monkeypatch.setattr(tool_runner, "_packaged_uv", lambda: str(packaged_uv))

        tool = ExternalTool("SkillSpector", "skillspector", managed=True)
        result = tool.run(["scan", "target"], log_command=False)

        assert result.success
        assert result.exit_code == 0
        assert result.stdout.splitlines()[:4] == ["tool", "run", "--quiet", "--from"]
        assert result.stdout.splitlines()[4].endswith("/2.13.4/skillspector-2.13.4-py3-none-any.whl")
        assert tool.path == str(packaged_uv)
        assert tool.resolution_source == "uv tool run"

    def test_explicit_older_override_is_preserved(self, scanner_bin, monkeypatch):
        scanner = self._scanner(scanner_bin, "SkillSpector v2.11.2")
        monkeypatch.setenv("SKILLEVALUATOR_SKILLSPECTOR_PATH", str(scanner))

        tool = ExternalTool(
            "SkillSpector", "skillspector", override_env="SKILLEVALUATOR_SKILLSPECTOR_PATH", managed=True
        )

        assert tool.run(["scan", "target"], log_command=False).stdout == "direct scanner\n"
        assert tool.path == str(scanner.resolve())
        assert tool.version == "2.11.2"
        assert tool.resolution_source == "override:SKILLEVALUATOR_SKILLSPECTOR_PATH"

    @pytest.mark.parametrize("unavailable", ["uv", "runtime"])
    def test_mismatched_version_cannot_run_when_managed_runtime_unavailable(
        self, scanner_bin, monkeypatch, unavailable
    ):
        scanner = self._scanner(scanner_bin, "SkillSpector v2.11.2")
        if unavailable == "uv":
            (scanner_bin / "uv").unlink()
        else:
            monkeypatch.setattr(tool_runner, "managed_skillspector_wheel", lambda: None)

        tool = ExternalTool(
            "SkillSpector", "skillspector", override_env="SKILLEVALUATOR_SKILLSPECTOR_PATH", managed=True
        )
        result = tool.run(["scan", "target"], log_command=False)

        assert not tool.is_available
        assert not result.success
        assert result.exit_code == -1
        assert result.stdout == ""
        assert tool.configuration_error
        assert result.error_message == tool.configuration_error
        assert str(scanner) in tool.configuration_error
        assert "2.11.2" in tool.configuration_error
        assert "2.13.4" in tool.configuration_error
        assert "uv" in tool.get_install_hint()
        assert "SKILLEVALUATOR_SKILLSPECTOR_PATH" in tool.get_install_hint()


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


def test_version_reports_pinned_release_for_managed_tools(tmp_path, monkeypatch):
    from skillevaluator.utils.scanner_install import GITLEAKS_VERSION
    from skillevaluator.utils.tool_runner import ExternalTool

    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("SKILLEVALUATOR_RUNTIME_DIR", str(tmp_path / "rt"))
    assert ExternalTool("Gitleaks", "gitleaks", managed=True).version == GITLEAKS_VERSION


class TestPackagedUv:
    """SkillSpector provisioning after a plain ``pip install``: no uv on PATH."""

    @staticmethod
    def _skillspector() -> ExternalTool:
        return ExternalTool("skillspector", "skillspector", managed=True)

    def test_uv_dependency_ships_an_executable(self):
        """The ``uv`` base dependency installs a binary that the fallback resolves."""
        path = tool_runner._packaged_uv()

        assert path is not None
        assert Path(path).name in {"uv", "uv.exe"}
        assert os.access(path, os.X_OK)

    def test_missing_uv_package_or_binary_is_not_an_error(self, monkeypatch):
        import uv

        def not_found() -> str:
            raise uv._find_uv.UvNotFound("no uv binary")

        monkeypatch.setattr(uv, "find_uv_bin", not_found)
        assert tool_runner._packaged_uv() is None

        monkeypatch.setitem(sys.modules, "uv", None)
        assert tool_runner._packaged_uv() is None

    def test_skillspector_runs_through_packaged_uv_when_path_has_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SKILLEVALUATOR_RUNTIME_DIR", str(tmp_path / "rt"))
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.delenv("SKILLEVALUATOR_SKILLSPECTOR_PATH", raising=False)
        # The interpreter's own bin dir holds neither tool, as with ``pip install --user``.
        monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
        monkeypatch.setattr(tool_runner, "_packaged_uv", lambda: "/env/bin/uv")

        tool = self._skillspector()

        assert tool.is_available
        assert tool.path == "/env/bin/uv"
        assert tool.resolution_source == "uv tool run"
        assert tool._launch_args()[:2] == ("tool", "run")

    def test_uv_on_path_keeps_precedence_over_packaged_uv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SKILLEVALUATOR_RUNTIME_DIR", str(tmp_path / "rt"))
        path_uv = tmp_path / "uv"
        path_uv.write_text("#!/bin/sh\n")
        path_uv.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
        monkeypatch.setattr(tool_runner, "_packaged_uv", lambda: "/env/bin/uv")

        assert self._skillspector().path == str(path_uv)

    def test_installed_skillspector_keeps_precedence_over_packaged_uv(self, tmp_path, monkeypatch):
        installed = tmp_path / "skillspector"
        installed.write_text("#!/bin/sh\n")
        installed.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
        monkeypatch.setattr(tool_runner, "_packaged_uv", lambda: "/env/bin/uv")

        tool = self._skillspector()

        assert tool.path == str(installed)
        assert tool.resolution_source == "automatic"

    def test_unavailable_when_no_uv_exists_anywhere(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
        monkeypatch.setattr(tool_runner, "_packaged_uv", lambda: None)

        tool = self._skillspector()

        assert not tool.is_available
        assert tool.get_install_hint().startswith("Install with: uv tool install ")


def test_version_parses_tool_output(tmp_path, monkeypatch):
    from skillevaluator.utils.tool_runner import ExternalTool

    tool = tmp_path / "bandit"
    tool.write_text("#!/bin/sh\necho 'bandit 1.9.4'\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ExternalTool("Bandit", "bandit").version == "1.9.4"
