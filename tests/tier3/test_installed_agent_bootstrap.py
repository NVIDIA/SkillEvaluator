# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the inherited Harbor bootstrap, including package selection."""

import asyncio
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import ExecResult
from harbor.models.trial.config import AgentConfig
from harbor.utils.import_path import import_symbol

from skillevaluator.tier3 import commands
from skillevaluator.tier3.harbor import DOCKER_AGENT_IMPORT_PATHS, adapter
from skillevaluator.tier3.harbor.installed_agents import (
    SkillEvaluatorClaudeCode,
    SkillEvaluatorCodex,
    SkillEvaluatorOpenCode,
)
from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorClaudeCode as ContainerClaudeCode
from skillevaluator.tier3.harbor.local_agents import (
    SkillEvaluatorGatewayCodex,
    SkillEvaluatorGatewayOpenCode,
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
        ca_bundle: bool = True,
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
        self.ca_bundle = ca_bundle
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
            if self.system_node:
                return ExecResult(return_code=0, stdout="system", stderr="")
            return ExecResult(return_code=0, stdout="bundled" if self.ca_bundle else "bundled-without-ca", stderr="")
        return ExecResult(return_code=0, stdout="", stderr="")


def _vendor_install(agent_class: type) -> str:
    """The text of the agent CLI install in Harbor's installer command."""
    return "npm i -g opencode-ai" if issubclass(agent_class, OpenCode) else "npm install -g"


def test_gateway_codex_debian_bootstrap_does_not_install_system_node(tmp_path: Path) -> None:
    environment = _BootstrapEnvironment()
    agent = SkillEvaluatorGatewayCodex(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert environment.package_installs == [], f"Unexpected system package install: {environment.package_installs}"
    assert any("nvm install 22" in command for command in environment.commands)


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode, SkillEvaluatorOpenCode])
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
    elif agent_class is SkillEvaluatorOpenCode:
        assert "ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]" in environment.probes[0]
        assert any("nvm install 22" in command for command in environment.commands)
        assert any("npm i -g opencode-ai" in command for command in environment.commands)
    else:
        assert "command -v apk" in environment.probes[0]
        assert any("https://downloads.claude.ai/claude-code-releases/bootstrap.sh" in c for c in environment.commands)


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode, SkillEvaluatorOpenCode])
@pytest.mark.parametrize(
    ("manager", "expected_install"),
    [
        ("apt-get", "set -o pipefail; apt-get update && apt-get install -y ca-certificates"),
        ("dnf", "set -o pipefail; dnf install -y ca-certificates"),
    ],
)
def test_bundled_installer_without_ca_bundle_installs_ca_certificates(
    tmp_path: Path, agent_class: type, manager: str, expected_install: str
) -> None:
    # Without npm, nothing else pulls in the CA bundle the vendor downloads need.
    environment = _BootstrapEnvironment(manager=manager, ca_bundle=False)
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert len(environment.probes) == 1
    assert "[ -s /etc/ssl/certs/ca-certificates.crt ]" in environment.probes[0]
    assert "[ -s /etc/pki/tls/certs/ca-bundle.crt ]" in environment.probes[0]
    assert environment.package_installs == [expected_install]


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode, SkillEvaluatorOpenCode])
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
    assert any(_vendor_install(agent_class) in command for command in environment.commands)


@pytest.mark.parametrize(
    ("agent_class", "missing_command", "expected_package"),
    [
        (SkillEvaluatorCodex, "rg", "ripgrep"),
        (SkillEvaluatorClaudeCode, "pgrep", "procps"),
        (SkillEvaluatorOpenCode, "stdbuf", "coreutils"),
    ],
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


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode, SkillEvaluatorOpenCode])
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
    assert not any(_vendor_install(agent_class) in command for command in environment.commands)


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode, SkillEvaluatorOpenCode])
def test_platform_probe_transport_errors_propagate(tmp_path: Path, agent_class: type) -> None:
    environment = _BootstrapEnvironment(probe_error=TimeoutError("probe transport timeout"))
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(TimeoutError, match="probe transport timeout"):
        asyncio.run(agent.install(environment))

    assert environment.package_installs == []


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode, SkillEvaluatorOpenCode])
def test_non_node_dependency_requests_skip_platform_detection(tmp_path: Path, agent_class: type) -> None:
    environment = _BootstrapEnvironment(missing_commands=("git",), probe_error=AssertionError("unneeded probe"))
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.ensure_system_dependencies(environment, ("git",)))

    assert environment.package_installs == ["set -o pipefail; apt-get update && apt-get install -y git"]
    assert environment.probes == []


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorClaudeCode, SkillEvaluatorOpenCode])
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
        ("claude-code", ContainerClaudeCode, "anthropic/test-model"),
        ("opencode", SkillEvaluatorOpenCode, "nvidia/test-model"),
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
    [
        SkillEvaluatorGatewayCodex,
        SkillEvaluatorNvidiaBuildCodex,
        SkillEvaluatorNvidiaBuildClaudeCode,
        ContainerClaudeCode,
        SkillEvaluatorGatewayOpenCode,
    ],
)
def test_gateway_and_nvidia_wrappers_use_repaired_bootstrap(tmp_path: Path, agent_class: type) -> None:
    environment = _BootstrapEnvironment()
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    asyncio.run(agent.install(environment))

    assert environment.package_installs == []
    assert len(environment.probes) == 1
    assert any(_vendor_install(agent_class) in command for command in environment.commands)


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


# Debian marks coreutils Essential and the python:3.12-slim base installs
# ca-certificates, so every generated image has both without listing them.
_SLIM_BASE_APT_PACKAGES = frozenset({"coreutils", "ca-certificates"})


def _skill(root: Path) -> Path:
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: skill\ndescription: Test fixture.\n---\n", encoding="utf-8")
    return root


def _apt_packages(dockerfile: str) -> set[str]:
    """The packages named by a generated Dockerfile's ``apt-get install`` command."""
    flags = "install -y --no-install-recommends"
    command = next(line for line in dockerfile.replace("\\\n", " ").splitlines() if flags in line)
    return set(command.split(flags, 1)[1].split("&&", 1)[0].replace("\\", " ").split())


def _generated_dockerfiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Every Dockerfile SkillEvaluator writes on the python:3.12-slim base, by its writer."""
    built: list[str] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        built.append((Path(cmd[-1]) / "Dockerfile").read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    assert adapter.build_eval_base_image(_skill(tmp_path / "base-skill")).startswith("skillevaluator-base:")

    task = tmp_path / "task"
    adapter._write_dockerfile(
        task_dir=task,
        skill_path=_skill(tmp_path / "task-skill"),
        reference_skills_dir=None,
        workspace_skill_paths=None,
        has_skill=True,
    )

    environment = tmp_path / "environment"
    environment.mkdir()
    adapter._write_default_environment_dockerfile(
        environment, base_image="", agent_config_lines=[], include_input=False
    )

    starter_skill = _skill(tmp_path / "starter-skill")
    assert (
        commands.init_harbor_task(
            starter_skill, force=False, case_id="case-001", mode="default", language="python", with_config=False
        )
        == 0
    )

    return {
        "build_eval_base_image": built[0],
        "_write_dockerfile": (task / "environment" / "Dockerfile").read_text(encoding="utf-8"),
        "_write_default_environment_dockerfile": (environment / "Dockerfile").read_text(encoding="utf-8"),
        "init-harbor-task": (starter_skill / "evals/harbor/case-001/environment/Dockerfile").read_text(
            encoding="utf-8"
        ),
    }


@pytest.mark.parametrize("agent_name", ["claude-code", "codex", "opencode"])
def test_generated_images_satisfy_docker_agent_setup_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_name: str
) -> None:
    """Harbor's agent setup apt-installs, in every trial, each prerequisite the image lacks."""
    requested: list[tuple[str, ...]] = []

    async def record(self: BaseInstalledAgent, environment: object, dependencies: tuple[str, ...]) -> None:
        requested.append(tuple(dependencies))

    # Record what reaches Harbor's package step after the wrapper's installer-branch filter.
    monkeypatch.setattr(BaseInstalledAgent, "ensure_system_dependencies", record)
    agent_class = import_symbol(DOCKER_AGENT_IMPORT_PATHS[agent_name])
    agent = agent_class(logs_dir=tmp_path / "logs", model_name="openai/test-model")
    asyncio.run(agent.install(_BootstrapEnvironment()))
    assert requested
    needed = {
        package
        for dependencies in requested
        for dependency in dependencies
        for package in BaseInstalledAgent.SYSTEM_PACKAGES[dependency].packages["apt-get"]
    }
    assert not {"nodejs", "npm"} & needed

    for writer, dockerfile in _generated_dockerfiles(tmp_path, monkeypatch).items():
        assert dockerfile.startswith("FROM python:3.12-slim\n"), writer
        image = _apt_packages(dockerfile) | _SLIM_BASE_APT_PACKAGES
        assert needed - image == set(), writer


def _docker_setup_note() -> str:
    """The eval-datasets note on Docker agent setup packages, with whitespace collapsed."""
    docs = (Path(__file__).resolve().parents[2] / "docs" / "eval-datasets.mdx").read_text(encoding="utf-8")
    notes = [block.split("</Note>", 1)[0] for block in docs.split("<Note>")[1:]]
    (note,) = [note for note in notes if "`procps`" in note]
    return " ".join(note.split())


def test_custom_dockerfiles_reach_procps_only_through_the_shared_base_image_rebase(tmp_path: Path) -> None:
    """A custom Dockerfile keeps its own base, and so needs procps, unless it is rebased."""
    custom_from = "FROM debian:bookworm-slim"
    first_lines: dict[str, str] = {}
    for name, kwargs in {
        # The runner passes an empty base_image unless harbor.base_image_mode is reuse or rebuild.
        "default": {},
        "preserve": {"base_image": "skillevaluator-base:test", "custom_dockerfile_mode": "preserve"},
        "rebase": {"base_image": "skillevaluator-base:test", "custom_dockerfile_mode": "rebase"},
    }.items():
        skill = _skill(tmp_path / f"{name}-skill")
        (skill / "evals" / "environment").mkdir(parents=True)
        (skill / "evals" / "environment" / "Dockerfile").write_text(f"{custom_from}\nRUN true\n", encoding="utf-8")
        task = tmp_path / f"{name}-task"
        adapter._write_dockerfile(
            task_dir=task,
            skill_path=skill,
            reference_skills_dir=None,
            workspace_skill_paths=None,
            has_skill=True,
            **kwargs,
        )
        first_lines[name] = (task / "environment" / "Dockerfile").read_text(encoding="utf-8").splitlines()[0]

    assert first_lines == {
        "default": custom_from,
        "preserve": custom_from,
        "rebase": "FROM skillevaluator-base:test",
    }
    note = _docker_setup_note()
    assert "A custom `evals/environment/Dockerfile` keeps its own base image, so it should install `procps` too" in note
    assert (
        "unless `harbor.base_image_mode` is `reuse` or `rebuild` and `harbor.custom_dockerfile_mode` is `rebase`"
        in note
    )


def test_docker_setup_note_scopes_the_npx_mcp_example_to_claude_code() -> None:
    """Codex and OpenCode stdio MCP servers inherit nvm's npx; only Claude Code has no Node.js."""
    note = _docker_setup_note()

    assert "including stdio MCP servers" in note
    assert 'Node.js with Claude Code, for example `command = "npx ..."`' in note
    assert "or `gcc`, `g++`, or `make` with any agent" in note
