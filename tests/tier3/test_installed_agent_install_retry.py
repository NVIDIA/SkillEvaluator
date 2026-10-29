# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry a container agent CLI install that a transient failure stopped."""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.base import NetworkConnectionError, NonZeroAgentExitCodeError
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.environments.base import ExecResult

from skillevaluator.tier3.harbor import installed_agents
from skillevaluator.tier3.harbor.installed_agents import SkillEvaluatorClaudeCode, SkillEvaluatorCodex
from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorClaudeCode as ContainerClaudeCode
from skillevaluator.tier3.harbor.local_agents import (
    SkillEvaluatorGatewayCodex,
    SkillEvaluatorLocalClaudeCode,
    SkillEvaluatorLocalCodex,
    SkillEvaluatorNvidiaBuildClaudeCode,
    SkillEvaluatorNvidiaBuildCodex,
)

_INSTALLER_MARKERS = ("nvm install 22", "claude-code-releases/bootstrap.sh")
_PACKAGE_INSTALL_MARKERS = ("apt-get install", "apk add")


class _FlakyEnvironment:
    """Fail the vendor installer or the package install a set number of times."""

    default_user = None

    def __init__(
        self,
        *,
        installer_failures: int = 0,
        package_failures: int = 0,
        system_npm: bool = False,
        probe_result: ExecResult | None = None,
        probe_error: Exception | None = None,
    ) -> None:
        self.commands: list[str] = []
        self.installer_failures = installer_failures
        self.package_failures = package_failures
        self.system_npm = system_npm
        self.probe_result = probe_result
        self.probe_error = probe_error
        self.installed = False

    def _count(self, markers: tuple[str, ...]) -> int:
        return sum(any(marker in command for marker in markers) for command in self.commands)

    @property
    def installer_runs(self) -> int:
        return self._count(_INSTALLER_MARKERS)

    @property
    def package_installs(self) -> int:
        return self._count(_PACKAGE_INSTALL_MARKERS)

    @property
    def probes(self) -> int:
        return sum(command.startswith("if ") and "printf" in command for command in self.commands)

    async def exec(self, command: str, **_kwargs: object) -> ExecResult:
        self.commands.append(command)
        if command in (
            Codex._INSTALL_CHECK_COMMAND,
            ClaudeCode._INSTALL_CHECK_COMMAND,
            Codex._INSTALL_VERSION_COMMAND,
            ClaudeCode._INSTALL_VERSION_COMMAND,
        ):
            return ExecResult(return_code=0 if self.installed else 1, stdout="", stderr="")
        if command.startswith("command -v ") and command.endswith(" >/dev/null 2>&1"):
            missing = command.split()[2] in {"node", "npm"}
            return ExecResult(return_code=1 if missing else 0, stdout="", stderr="")
        if command.startswith("for manager in "):
            return ExecResult(return_code=0, stdout="apk" if self.system_npm else "apt-get", stderr="")
        if command.startswith("if ") and "printf" in command:
            if self.probe_error is not None:
                raise self.probe_error
            if self.probe_result is not None:
                return self.probe_result
            return ExecResult(return_code=0, stdout="system" if self.system_npm else "bundled", stderr="")
        if any(marker in command for marker in _PACKAGE_INSTALL_MARKERS):
            if self.package_installs <= self.package_failures:
                return ExecResult(
                    return_code=1,
                    stdout="",
                    stderr="WARNING: fetching https://dl-cdn.alpinelinux.org/alpine/APKINDEX.tar.gz: temporary error",
                )
        elif any(marker in command for marker in _INSTALLER_MARKERS):
            if self.installer_runs <= self.installer_failures:
                return ExecResult(return_code=6, stdout="", stderr="curl: (6) Could not resolve host: example.invalid")
            self.installed = True
        return ExecResult(return_code=0, stdout="", stderr="")


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(installed_agents.asyncio, "sleep", record_sleep)
    return delays


_CONTAINER_AGENT_CLASSES = [
    SkillEvaluatorCodex,
    SkillEvaluatorClaudeCode,
    SkillEvaluatorGatewayCodex,
    SkillEvaluatorNvidiaBuildCodex,
    SkillEvaluatorNvidiaBuildClaudeCode,
    pytest.param(ContainerClaudeCode, id="ContainerClaudeCode"),
]


@pytest.mark.parametrize("agent_class", _CONTAINER_AGENT_CLASSES)
def test_transient_installer_failure_is_retried(tmp_path: Path, agent_class: type, sleeps: list[float]) -> None:
    environment = _FlakyEnvironment(installer_failures=1)
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert environment.installer_runs == 2
    assert environment.installed
    assert sleeps == [installed_agents._INSTALL_RETRY_BASE_DELAY_SEC]


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
def test_persistent_installer_failure_stops_after_bounded_attempts(
    tmp_path: Path, agent_class: type, sleeps: list[float]
) -> None:
    environment = _FlakyEnvironment(installer_failures=installed_agents._INSTALL_ATTEMPTS)
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(NetworkConnectionError, match="Could not resolve host"):
        asyncio.run(agent.install(environment))

    assert environment.installer_runs == installed_agents._INSTALL_ATTEMPTS
    base_delay = installed_agents._INSTALL_RETRY_BASE_DELAY_SEC
    assert sleeps == [base_delay, 2 * base_delay]


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
def test_transient_system_package_failure_is_retried(tmp_path: Path, agent_class: type, sleeps: list[float]) -> None:
    # The npm installer branch still needs distribution Node/npm from a mirror.
    environment = _FlakyEnvironment(package_failures=1, system_npm=True)
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert environment.package_installs == 2
    assert any("npm install -g" in command for command in environment.commands)
    assert len(sleeps) == 1


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
@pytest.mark.parametrize(
    ("probe_result", "probe_error", "expected"),
    [
        (ExecResult(return_code=0, stdout="unrecognized", stderr=""), None, RuntimeError),
        (None, TimeoutError("probe transport timeout"), TimeoutError),
    ],
    ids=["unknown-platform", "probe-timeout"],
)
def test_platform_and_transport_errors_are_not_retried(
    tmp_path: Path,
    agent_class: type,
    probe_result: ExecResult | None,
    probe_error: Exception | None,
    expected: type[Exception],
    sleeps: list[float],
) -> None:
    environment = _FlakyEnvironment(probe_result=probe_result, probe_error=probe_error)
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(expected) as raised:
        asyncio.run(agent.install(environment))

    assert not isinstance(raised.value, NonZeroAgentExitCodeError)
    assert environment.probes == 1
    assert environment.installer_runs == 0
    assert sleeps == []


@pytest.mark.parametrize("agent_class", [SkillEvaluatorLocalCodex, SkillEvaluatorLocalClaudeCode])
def test_local_agents_do_not_retry_a_missing_host_cli(tmp_path: Path, agent_class: type, sleeps: list[float]) -> None:
    class _MissingCli:
        default_user = None

        def __init__(self) -> None:
            self.commands: list[str] = []

        async def exec(self, command: str, **_kwargs: object) -> ExecResult:
            self.commands.append(command)
            return ExecResult(return_code=127, stdout="", stderr="command not found")

    environment = _MissingCli()
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(NonZeroAgentExitCodeError):
        asyncio.run(agent.install(environment))

    assert len(environment.commands) == 1
    assert sleeps == []
