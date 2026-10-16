# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optional dashboard launcher preserves the lightweight CLI contract."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.reporting import dashboard


class FakePopen:
    """Stand in for the Streamlit child: record the launch, waits, and forwarded signals."""

    def __init__(self, command: list[str], **kwargs) -> None:
        self.command = command
        self.kwargs = kwargs
        self.sent: list[int] = []
        self.wait_timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        return self.exit_code()

    def exit_code(self) -> int:
        return 7

    def send_signal(self, signum: int) -> None:
        self.sent.append(signum)

    def terminate(self) -> None:
        self.send_signal(signal.SIGTERM)


def _fake_streamlit_spec(monkeypatch, spec: importlib.machinery.ModuleSpec | None) -> None:
    """Fake only Streamlit's lookup; coverage and pytest call find_spec for other modules."""
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        dashboard.importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: spec if name == "streamlit" else find_spec(name, *args, **kwargs),
    )


def _capture_launch(monkeypatch, *args: str, popen: type[FakePopen] = FakePopen):
    """Invoke the dashboard command with Streamlit present and the child process faked."""
    launched: list[FakePopen] = []
    _fake_streamlit_spec(monkeypatch, importlib.machinery.ModuleSpec("streamlit", None))

    def start(command, **kwargs):
        launched.append(popen(command, **kwargs))
        return launched[-1]

    monkeypatch.setattr(dashboard.subprocess, "Popen", start)
    result = CliRunner().invoke(cli, ["dashboard", *args])
    return result, launched[0]


def test_dashboard_help_and_input_validation() -> None:
    runner = CliRunner()
    help_result = runner.invoke(cli, ["dashboard", "--help"])
    assert help_result.exit_code == 0
    assert "retained Tier 3" in help_result.output
    assert "next free port" in " ".join(help_result.output.split())
    assert runner.invoke(cli, ["dashboard", "--port", "0"]).exit_code == 2
    assert runner.invoke(cli, ["dashboard", "missing-report-170.json"]).exit_code == 2


def test_missing_extra_has_actionable_error(monkeypatch) -> None:
    _fake_streamlit_spec(monkeypatch, None)
    result = CliRunner().invoke(cli, ["dashboard"])
    assert result.exit_code == 1
    assert "skillevaluator[all,dashboard]" in result.output
    assert "uv run --extra dashboard" in result.output
    assert 'pip install "skillevaluator[dashboard]"' not in result.output


def test_launch_uses_loopback_and_literal_absolute_paths(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "report with spaces; literal.json"
    report.write_text("{}", encoding="utf-8")
    result, fake = _capture_launch(monkeypatch, str(report), "--port", "8517", "--no-browser")
    assert result.exit_code == 7
    command, kwargs = fake.command, fake.kwargs
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


def test_launch_settings_exist_in_installed_streamlit(monkeypatch) -> None:
    # Streamlit silently ignores settings it does not define; server.allowedHosts needs 1.61.
    config = pytest.importorskip("streamlit.config")
    # Filled on import; config._config_options stays None until Streamlit loads its config.
    options = config._config_options_template
    monkeypatch.delenv("STREAMLIT_SERVER_ALLOWED_HOSTS", raising=False)
    _result, fake = _capture_launch(monkeypatch, "--port", "8517", "--no-browser")
    command, env = fake.command, fake.kwargs["env"]
    flags = command[command.index("run") + 2 : command.index("--")]
    assert flags
    for flag in flags:
        name, separator, _value = flag.removeprefix("--").partition("=")
        assert flag.startswith("--") and separator, flag
        assert name in options and not options[name].deprecated, flag
    env_options = {option.env_var: name for name, option in options.items()}
    added = {name for name, value in env.items() if os.environ.get(name) != value}
    assert {env_options.get(name) for name in added} == {"server.allowedHosts"}


def test_default_launch_lets_streamlit_choose_port(monkeypatch) -> None:
    # Streamlit moves to the next free port only when the port was not set explicitly.
    _result, fake = _capture_launch(monkeypatch)
    assert not any(arg.startswith("--server.port") for arg in fake.command)


def test_explicit_port_failure_has_hint(monkeypatch) -> None:
    class BusyPortPopen(FakePopen):
        def exit_code(self) -> int:
            return 1

    result, _fake = _capture_launch(monkeypatch, "--port", "8517", popen=BusyPortPopen)
    assert result.exit_code == 1
    assert "choose another with --port" in result.output


def test_default_launch_does_not_force_headless(monkeypatch) -> None:
    # An explicit headless=false makes Streamlit prompt for an email and exit without a TTY.
    _result, fake = _capture_launch(monkeypatch)
    assert not any(arg.startswith("--server.headless") for arg in fake.command)


def test_sigterm_is_forwarded_to_streamlit(monkeypatch) -> None:
    original = signal.getsignal(signal.SIGTERM)

    class TerminatedPopen(FakePopen):
        def exit_code(self) -> int:
            # A process manager sends SIGTERM to the launcher alone.
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return -15

    result, fake = _capture_launch(monkeypatch, popen=TerminatedPopen)
    assert fake.sent == [signal.SIGTERM]
    assert signal.getsignal(signal.SIGTERM) is original
    assert result.exit_code == 128 + signal.SIGTERM


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP is POSIX-only")
def test_streamlit_killed_by_signal_exits_like_a_shell(monkeypatch) -> None:
    class HungUpPopen(FakePopen):
        def exit_code(self) -> int:
            # Streamlit has no SIGHUP handler, so the forwarded signal kills it.
            signal.getsignal(signal.SIGHUP)(signal.SIGHUP, None)
            return -signal.SIGHUP

    result, fake = _capture_launch(monkeypatch, "--port", "8517", popen=HungUpPopen)
    assert fake.sent == [signal.SIGHUP]
    assert result.exit_code == 128 + signal.SIGHUP
    # The busy-port hint is only for Streamlit's own failures.
    assert "choose another with --port" not in result.output


def test_ctrl_c_returns_zero(monkeypatch) -> None:
    class InterruptedPopen(FakePopen):
        def exit_code(self) -> int:
            if len(self.wait_timeouts) == 1:
                raise KeyboardInterrupt
            return 0

    result, fake = _capture_launch(monkeypatch, popen=InterruptedPopen)
    assert result.exit_code == 0
    # The launcher waits for Streamlit to finish its own Ctrl+C shutdown, and
    # passes SIGINT on in case it was sent to the launcher alone.
    assert fake.wait_timeouts == [None, 10]
    assert fake.sent == ([] if sys.platform == "win32" else [signal.SIGINT])


def test_ctrl_c_terminates_streamlit_that_does_not_stop(monkeypatch) -> None:
    class StuckPopen(FakePopen):
        def exit_code(self) -> int:
            if len(self.wait_timeouts) == 1:
                raise KeyboardInterrupt
            if self.wait_timeouts[-1] is not None:
                raise subprocess.TimeoutExpired(self.command, self.wait_timeouts[-1])
            return -signal.SIGTERM

    result, fake = _capture_launch(monkeypatch, popen=StuckPopen)
    assert result.exit_code == 0
    assert fake.wait_timeouts == [None, 10, None]
    assert fake.sent == ([] if sys.platform == "win32" else [signal.SIGINT]) + [signal.SIGTERM]


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
    _result, fake = _capture_launch(monkeypatch)
    command = fake.command
    # Restore the real subprocess.Popen before starting Python below.
    monkeypatch.undo()
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
