# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optional dashboard launcher preserves the lightweight CLI contract."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.reporting import dashboard


def _capture_launch(monkeypatch, *args: str):
    """Invoke the dashboard command with Streamlit present and the child process faked."""
    calls = []
    monkeypatch.setattr(dashboard.importlib.util, "find_spec", lambda _name: object())

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return 7

    monkeypatch.setattr(dashboard.subprocess, "call", capture)
    result = CliRunner().invoke(cli, ["dashboard", *args])
    command, kwargs = calls[0]
    return result, command, kwargs


def test_dashboard_help_and_input_validation() -> None:
    runner = CliRunner()
    help_result = runner.invoke(cli, ["dashboard", "--help"])
    assert help_result.exit_code == 0
    assert "retained Tier 3" in help_result.output
    assert runner.invoke(cli, ["dashboard", "--port", "0"]).exit_code == 2
    assert runner.invoke(cli, ["dashboard", "missing-report-170.json"]).exit_code == 2


def test_missing_extra_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setattr(dashboard.importlib.util, "find_spec", lambda _name: None)
    result = CliRunner().invoke(cli, ["dashboard"])
    assert result.exit_code == 1
    assert "skillevaluator[all,dashboard]" in result.output
    assert "uv run --extra dashboard" in result.output
    assert 'pip install "skillevaluator[dashboard]"' not in result.output


def test_launch_uses_loopback_and_literal_absolute_paths(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "report with spaces; literal.json"
    report.write_text("{}", encoding="utf-8")
    result, command, kwargs = _capture_launch(monkeypatch, str(report), "--port", "8517", "--no-browser")
    assert result.exit_code == 7
    assert command[:4] == [sys.executable, "-P", "-m", "streamlit"]
    app = Path(command[command.index("run") + 1])
    assert app.name == "skillevaluator_dashboard.py"
    assert app.parent.name == "dashboard_ui"
    assert Path(kwargs["cwd"]) == app.parent
    assert kwargs["env"]["STREAMLIT_SERVER_ALLOWED_HOSTS"] == "127.0.0.1 localhost"
    assert "--server.address=127.0.0.1" in command
    assert "--server.port=8517" in command
    assert "--server.maxUploadSize=16" in command
    for flag in (
        "--server.showEmailPrompt=false",
        "--server.fileWatcherType=none",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=true",
        "--client.toolbarMode=viewer",
        "--server.headless=true",
    ):
        assert flag in command
    assert "--browser.gatherUsageStats=false" in command
    assert command[-2:] == ["--", str(report.resolve())]


def test_default_launch_does_not_force_headless(monkeypatch) -> None:
    # An explicit headless=false makes Streamlit prompt for an email and exit without a TTY.
    _result, command, _kwargs = _capture_launch(monkeypatch)
    assert not any(arg.startswith("--server.headless") for arg in command)


def test_streamlit_script_directory_cannot_shadow_modules() -> None:
    # Streamlit puts the script directory first on sys.path in the server.
    app = dashboard._APP
    assert app.is_file()
    stems = {path.stem for path in app.parent.glob("*.py")}
    assert stems <= {"__init__", app.stem}
    assert not stems & set(sys.stdlib_module_names)
    assert importlib.util.find_spec(app.stem) is None


def test_streamlit_is_not_shadowed_by_cwd(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("streamlit")
    _result, command, _kwargs = _capture_launch(monkeypatch)
    (tmp_path / "streamlit.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    completed = subprocess.run(
        [*command[:4], "version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Streamlit" in completed.stdout
