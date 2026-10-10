# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the inherited Harbor bootstrap, including package selection."""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.environments.base import ExecResult
from harbor.models.trial.config import AgentConfig

from skillevaluator.tier3.harbor import DOCKER_AGENT_IMPORT_PATHS
from skillevaluator.tier3.harbor.installed_agents import SkillEvaluatorClaudeCode, SkillEvaluatorCodex
from skillevaluator.tier3.harbor.local_agents import (
    SkillEvaluatorGatewayCodex,
    SkillEvaluatorLocalClaudeCode,
    SkillEvaluatorLocalCodex,
    SkillEvaluatorNvidiaBuildClaudeCode,
    SkillEvaluatorNvidiaBuildCodex,
)


class _BootstrapEnvironment:
    default_user = None

    def __init__(
        self,
        *,
        manager: str = "apt-get",
        system_node: bool = False,
        missing_commands: tuple[str, ...] = ("node", "npm"),
        installed: bool = False,
        installed_version: str = "1.2.3",
        probe_result: ExecResult | None = None,
        probe_error: Exception | None = None,
    ) -> None:
        self.commands: list[str] = []
        self.exec_kwargs: list[dict[str, object]] = []
        self.manager = manager
        self.system_node = system_node
        self.missing_commands = missing_commands
        self.installed = installed
        self.installed_version = installed_version
        self.probe_result = probe_result
        self.probe_error = probe_error

    @property
    def probes(self) -> list[str]:
        return [command for command in self.commands if command.startswith("if ") and "printf" in command]

    @property
    def package_installs(self) -> list[str]:
        return [
            command
            for command in self.commands
            if any(text in command for text in ("apt-get install", "dnf install", "yum install", "apk add"))
        ]

    async def exec(self, command: str, **_kwargs: object) -> ExecResult:
        self.commands.append(command)
        self.exec_kwargs.append(_kwargs)
        if command in (Codex._INSTALL_CHECK_COMMAND, ClaudeCode._INSTALL_CHECK_COMMAND):
            return ExecResult(return_code=0 if self.installed else 1, stdout="", stderr="")
        if command in (Codex._INSTALL_VERSION_COMMAND, ClaudeCode._INSTALL_VERSION_COMMAND):
            version = (
                f"codex-cli {self.installed_version}"
                if command == Codex._INSTALL_VERSION_COMMAND
                else f"{self.installed_version} (Claude Code)"
            )
            return ExecResult(return_code=0 if self.installed else 1, stdout=version, stderr="")
        if command.startswith("command -v ") and command.endswith(" >/dev/null 2>&1"):
            executable = command.split()[2]
            return ExecResult(return_code=1 if executable in self.missing_commands else 0, stdout="", stderr="")
        if command.startswith("for manager in "):
            return ExecResult(return_code=0, stdout=self.manager, stderr="")
        if command.startswith("if ") and "printf" in command:
            if self.probe_error is not None:
                raise self.probe_error
            if self.probe_result is not None:
                return self.probe_result
            return ExecResult(return_code=0, stdout="system" if self.system_node else "bundled", stderr="")
        return ExecResult(return_code=0, stdout="", stderr="")


def test_gateway_codex_debian_bootstrap_does_not_install_system_node(tmp_path: Path) -> None:
    environment = _BootstrapEnvironment()
    agent = SkillEvaluatorGatewayCodex(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert environment.package_installs == [], f"Unexpected system package install: {environment.package_installs}"
    assert any("nvm install 22" in command for command in environment.commands)


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
@pytest.mark.parametrize("manager", ["apt-get", "dnf"])
def test_native_bootstrap_avoids_system_node_and_keeps_vendor_installer(
    tmp_path: Path, agent_class: type, manager: str
) -> None:
    environment = _BootstrapEnvironment(manager=manager)
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert environment.package_installs == []
    assert len(environment.probes) == 1
    probe_index = environment.commands.index(environment.probes[0])
    assert environment.exec_kwargs[probe_index]["user"] == "root"
    if agent_class is SkillEvaluatorCodex:
        assert "ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]" in environment.probes[0]
        assert any("nvm install 22" in command for command in environment.commands)
        assert any("ln -sf" in command for command in environment.commands)
    else:
        assert "command -v apk" in environment.probes[0]
        assert any("https://downloads.claude.ai/claude-code-releases/bootstrap.sh" in c for c in environment.commands)


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
@pytest.mark.parametrize("manager", ["apt-get", "apk"])
def test_system_npm_installer_still_provisions_node_and_npm(tmp_path: Path, agent_class: type, manager: str) -> None:
    environment = _BootstrapEnvironment(manager=manager, system_node=True)
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert len(environment.package_installs) == 1
    assert "nodejs npm" in environment.package_installs[0]
    assert environment.package_installs[0].startswith(
        "set -o pipefail; apt-get update" if manager == "apt-get" else "set -o pipefail; apk add"
    )
    assert any("npm install -g" in command for command in environment.commands)


@pytest.mark.parametrize(
    ("agent_class", "missing_command", "expected_package"),
    [(SkillEvaluatorCodex, "rg", "ripgrep"), (SkillEvaluatorClaudeCode, "pgrep", "procps")],
)
def test_native_bootstrap_preserves_other_system_dependencies(
    tmp_path: Path, agent_class: type, missing_command: str, expected_package: str
) -> None:
    environment = _BootstrapEnvironment(missing_commands=("node", "npm", missing_command))
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert environment.package_installs == [f"set -o pipefail; apt-get update && apt-get install -y {expected_package}"]


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
@pytest.mark.parametrize("version", [None, "1.2.3"])
def test_existing_matching_cli_skips_probe_and_bootstrap(
    tmp_path: Path, agent_class: type, version: str | None
) -> None:
    environment = _BootstrapEnvironment(installed=True)
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model", version=version)

    asyncio.run(agent.install(environment))

    assert len(environment.commands) == 1
    assert environment.probes == []
    assert environment.package_installs == []


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
def test_mismatching_cli_version_runs_bootstrap_with_requested_version(tmp_path: Path, agent_class: type) -> None:
    environment = _BootstrapEnvironment(installed=True, installed_version="1.2.2")
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model", version="1.2.3")

    asyncio.run(agent.install(environment))

    assert environment.package_installs == []
    assert len(environment.probes) == 1
    expected = "@openai/codex@1.2.3" if agent_class is SkillEvaluatorCodex else "bash -s -- 1.2.3"
    assert any(expected in command for command in environment.commands)


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
@pytest.mark.parametrize(
    ("return_code", "stdout"),
    [(1, "bundled"), (137, ""), (0, ""), (0, "unrecognized")],
)
def test_unknown_installer_platform_stops_without_installing(
    tmp_path: Path, agent_class: type, return_code: int, stdout: str
) -> None:
    environment = _BootstrapEnvironment(probe_result=ExecResult(return_code=return_code, stdout=stdout, stderr=""))
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(RuntimeError, match="Could not determine"):
        asyncio.run(agent.install(environment))

    assert environment.package_installs == []
    assert not any("npm install -g" in command for command in environment.commands)


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
def test_platform_probe_transport_errors_propagate(tmp_path: Path, agent_class: type) -> None:
    environment = _BootstrapEnvironment(probe_error=TimeoutError("probe transport timeout"))
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(TimeoutError, match="probe transport timeout"):
        asyncio.run(agent.install(environment))

    assert environment.package_installs == []


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
def test_non_node_dependency_requests_skip_platform_detection(tmp_path: Path, agent_class: type) -> None:
    environment = _BootstrapEnvironment(missing_commands=("git",), probe_error=AssertionError("unneeded probe"))
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.ensure_system_dependencies(environment, ("git",)))

    assert environment.package_installs == ["set -o pipefail; apt-get update && apt-get install -y git"]
    assert environment.probes == []


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode])
def test_native_bootstrap_preserves_unknown_dependency_validation(tmp_path: Path, agent_class: type) -> None:
    environment = _BootstrapEnvironment()
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(ValueError, match="Unknown system dependencies: unknown-package"):
        asyncio.run(agent.ensure_system_dependencies(environment, ("nodejs", "npm", "unknown-package")))

    assert len(environment.probes) == 1
    assert environment.package_installs == []


@pytest.mark.parametrize(
    ("agent_name", "agent_class", "model_name"),
    [
        ("codex", SkillEvaluatorCodex, "openai/test-model"),
        ("claude-code", SkillEvaluatorClaudeCode, "anthropic/test-model"),
    ],
)
@pytest.mark.parametrize("config_field", ["name", "import_path"])
def test_docker_import_routes_resolve_and_preserve_report_identity(
    tmp_path: Path, agent_name: str, agent_class: type, model_name: str, config_field: str
) -> None:
    config = AgentConfig(
        **{config_field: DOCKER_AGENT_IMPORT_PATHS[agent_name]},
        model_name=model_name,
        kwargs={"version": "1.2.3"},
    )

    assert AgentFactory.get_agent_class_from_config(config) is agent_class
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)

    assert type(agent) is agent_class
    assert agent.name() == agent_name
    info = agent.to_agent_info()
    assert info.name == agent_name
    assert info.version == "1.2.3"
    assert info.model_info is not None
    assert info.model_info.name == "test-model"
    assert info.model_info.provider == model_name.split("/", maxsplit=1)[0]


@pytest.mark.parametrize(
    "agent_class",
    [SkillEvaluatorGatewayCodex, SkillEvaluatorNvidiaBuildCodex, SkillEvaluatorNvidiaBuildClaudeCode],
)
def test_gateway_and_nvidia_wrappers_use_repaired_bootstrap(tmp_path: Path, agent_class: type) -> None:
    environment = _BootstrapEnvironment()
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert environment.package_installs == []
    assert len(environment.probes) == 1
    assert any("npm install -g" in command for command in environment.commands)


@pytest.mark.parametrize("agent_class", [SkillEvaluatorLocalCodex, SkillEvaluatorLocalClaudeCode])
def test_local_agents_keep_existing_version_check_without_install(tmp_path: Path, agent_class: type) -> None:
    environment = _BootstrapEnvironment(installed=True)
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert len(environment.commands) == 1
    assert environment.commands[0].endswith(
        "codex --version" if agent_class is SkillEvaluatorLocalCodex else "claude --version"
    )
    assert environment.package_installs == []
    assert environment.probes == []
