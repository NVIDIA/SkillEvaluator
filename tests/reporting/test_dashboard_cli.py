# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optional dashboard launcher preserves the lightweight CLI contract."""

from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.reporting import dashboard


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
    assert 'pip install "skillevaluator[dashboard]"' in result.output


def test_launch_uses_loopback_and_literal_absolute_paths(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "report with spaces; literal.json"
    report.write_text("{}", encoding="utf-8")
    commands = []
    monkeypatch.setattr(dashboard.importlib.util, "find_spec", lambda _name: object())

    def capture(command):
        commands.append(command)
        return 7

    monkeypatch.setattr(dashboard.subprocess, "call", capture)
    result = CliRunner().invoke(cli, ["dashboard", str(report), "--port", "8517", "--no-browser"])
    assert result.exit_code == 7
    command = commands[0]
    assert command[:3] == [sys.executable, "-m", "streamlit"]
    assert "--server.address=127.0.0.1" in command
    assert "--server.port=8517" in command
    assert "--server.headless=true" in command
    assert "--browser.gatherUsageStats=false" in command
    assert command[-2:] == ["--", str(report.resolve())]
