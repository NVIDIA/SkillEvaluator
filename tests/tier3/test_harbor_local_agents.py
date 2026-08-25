# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import EnvironmentConfig, MCPServerConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

from skillevaluator.tier3.harbor.local_agents import (
    SkillEvaluatorLocalClaudeCode,
    SkillEvaluatorLocalCodex,
    SkillEvaluatorLocalOpenCode,
    SkillEvaluatorNvidiaBuildClaudeCode,
    SkillEvaluatorNvidiaBuildCodex,
)


class _RecordingEnvironment:
    default_user = None

    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []

    async def upload_file(self, source: object, destination: object) -> None:
        self.uploads.append((str(destination), Path(source).read_text(encoding="utf-8")))

    @contextlib.contextmanager
    def scoped_exec_env(self, _env: dict[str, str]):
        yield

    def last_codex_config(self, remote_home: object) -> dict[str, object]:
        remote_path = f"{remote_home}/config.toml"
        configs = [content for destination, content in self.uploads if destination == remote_path]
        assert configs, f"no Codex config was uploaded to {remote_path}"
        return tomllib.loads(configs[-1])


class _MergingRecordingEnvironment(BaseEnvironment):
    """Real Harbor env scoping with a local child process for final-env proof."""

    def __init__(self, tmp_path: Path) -> None:
        self.commands: list[str] = []
        self.merged_exec_envs: list[dict[str, str]] = []
        self.actual_child_envs: list[dict[str, str]] = []
        super().__init__(
            environment_dir=tmp_path,
            environment_name="recording",
            session_id="recording-session",
            trial_paths=TrialPaths(tmp_path / "trial"),
            task_env_config=EnvironmentConfig(),
            persistent_env={"PERSISTENT_VALUE": "persistent"},
        )

    @staticmethod
    def type() -> str:
        return "recording"

    def _validate_definition(self) -> None:
        return None

    async def start(self, force_build: bool) -> None:
        _ = force_build

    async def stop(self, delete: bool) -> None:
        _ = delete

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        _ = (source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        _ = (source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        _ = (source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        _ = (source_dir, target_dir)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = (timeout_sec, user)
        merged = self._merge_env(env) or {}
        self.commands.append(command)
        self.merged_exec_envs.append(dict(merged))

        process_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **merged}
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            command,
            cwd=cwd,
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode()
        child_env: dict[str, str] = {}
        for line in stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                child_env[key] = value
        self.actual_child_envs.append(child_env)
        return ExecResult(
            stdout=stdout,
            stderr=stderr_bytes.decode(),
            return_code=process.returncode,
        )


def _record_installed_agent_exec(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, str]]]:
    calls: list[tuple[str, dict[str, str]]] = []

    async def fake_parent_exec(
        _self: object,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        calls.append((command, dict(env or {})))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )
    return calls


def test_local_codex_uses_per_trial_codex_home() -> None:
    assert str(SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME).startswith(EnvironmentPaths.agent_dir.as_posix())
    assert str(SkillEvaluatorLocalCodex._REMOTE_CODEX_SECRETS_DIR).startswith(EnvironmentPaths.agent_dir.as_posix())
    assert str(SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME) != "/tmp/codex-home"
    assert str(SkillEvaluatorLocalCodex._REMOTE_CODEX_SECRETS_DIR) != "/tmp/codex-secrets"


def test_local_codex_uploads_openai_responses_config_and_creates_per_trial_state_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _record_installed_agent_exec(monkeypatch)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    environment = _RecordingEnvironment()
    agent = SkillEvaluatorLocalCodex(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.4",
        extra_env={"OPENAI_API_KEY": "test-key"},
    )

    asyncio.run(agent.run("do the thing", environment=environment, context=object()))

    setup_command, setup_env = calls[0]
    assert 'mkdir -p "$CODEX_HOME"' in setup_command
    assert SkillEvaluatorLocalCodex._REMOTE_CODEX_SECRETS_DIR.as_posix() in setup_command
    assert setup_env["CODEX_HOME"] == SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME.as_posix()
    config = environment.last_codex_config(SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME)
    assert config["model_provider"] == "openai_compatible"
    assert config["model_providers"] == {
        "openai_compatible": {
            "name": "OpenAI-compatible provider",
            "base_url": "https://api.openai.com/v1",
            "env_key": "OPENAI_API_KEY",
            "wire_api": "responses",
        }
    }


def test_local_codex_final_config_and_launcher_preserve_explicit_gateway_routing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _record_installed_agent_exec(monkeypatch)
    environment = _RecordingEnvironment()
    gateway_url = "https://gateway.example/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", gateway_url)
    model_name = "openai/openai/gpt-5.4"
    agent = SkillEvaluatorLocalCodex(
        logs_dir=tmp_path,
        model_name=model_name,
        extra_env={"OPENAI_API_KEY": "test-key"},
    )

    asyncio.run(agent.run("do the thing", environment=environment, context=object()))

    config = environment.last_codex_config(SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME)
    assert config["model_provider"] == "openai_compatible"
    assert config["openai_base_url"] == gateway_url
    assert config["model_providers"]["openai_compatible"]["base_url"] == gateway_url
    run_command = next(command for command, _env in calls if "codex exec" in command)
    assert f"--model {model_name} " in run_command
    assert "--model gpt-5.4 " not in run_command


def test_local_codex_final_config_preserves_user_and_harbor_mcp_servers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _record_installed_agent_exec(monkeypatch)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    environment = _RecordingEnvironment()
    agent = SkillEvaluatorLocalCodex(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.4",
        extra_env={"OPENAI_API_KEY": "test-key"},
        config={
            "mcp_servers": {
                "user-tools": {"url": "https://user-tools.example/mcp"},
            }
        },
        mcp_servers=[
            MCPServerConfig(
                name="task-tools",
                transport="stdio",
                command="python3",
                args=["-m", "task_tools"],
            )
        ],
    )

    asyncio.run(agent.run("do the thing", environment=environment, context=object()))

    config = environment.last_codex_config(SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME)
    assert config["mcp_servers"] == {
        "user-tools": {"url": "https://user-tools.example/mcp"},
        "task-tools": {"command": "python3", "args": ["-m", "task_tools"]},
    }
    assert config["model_provider"] == "openai_compatible"


def test_local_codex_runtime_provider_fields_win_without_losing_unrelated_user_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _record_installed_agent_exec(monkeypatch)
    environment = _RecordingEnvironment()
    gateway_url = "https://runtime.example/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient-must-not-win.example/v1")
    agent = SkillEvaluatorLocalCodex(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.4",
        extra_env={"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": gateway_url},
        config={
            "approval_policy": "never",
            "model_provider": "user-provider",
            "openai_base_url": "https://user.example/v1",
            "model_providers": {
                "openai_compatible": {
                    "name": "User provider",
                    "base_url": "https://user.example/v1",
                    "env_key": "USER_API_KEY",
                    "wire_api": "chat_completions",
                    "http_headers": {"X-User": "preserved"},
                },
                "user-provider": {"base_url": "https://other.example/v1"},
            },
        },
    )

    asyncio.run(agent.run("do the thing", environment=environment, context=object()))

    config = environment.last_codex_config(SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME)
    assert config["approval_policy"] == "never"
    assert config["model_provider"] == "openai_compatible"
    assert config["openai_base_url"] == gateway_url
    assert config["model_providers"]["user-provider"] == {"base_url": "https://other.example/v1"}
    assert config["model_providers"]["openai_compatible"] == {
        "name": "OpenAI-compatible provider",
        "base_url": gateway_url,
        "env_key": "OPENAI_API_KEY",
        "wire_api": "responses",
        "http_headers": {"X-User": "preserved"},
    }


def test_local_codex_rewrites_upstream_tmp_secrets_dir(monkeypatch, tmp_path) -> None:
    captured: dict[str, str] = {}

    async def fake_parent_exec(self, environment, command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env") or {}

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )

    agent = SkillEvaluatorLocalCodex(logs_dir=tmp_path, model_name="openai/test")
    asyncio.run(
        agent.exec_as_agent(
            object(),
            "mkdir -p /tmp/codex-secrets && rm -rf /tmp/codex-secrets",
            env={"OPENAI_API_KEY": "test"},
        )
    )

    assert "/tmp/codex-secrets" not in captured["command"]
    assert SkillEvaluatorLocalCodex._REMOTE_CODEX_SECRETS_DIR.as_posix() in captured["command"]
    assert captured["env"] == {"OPENAI_API_KEY": "test"}


def test_local_claude_uses_managed_policy_permission_mode(monkeypatch, tmp_path) -> None:
    commands: list[str] = []

    async def fake_parent_exec(self, environment, command, **kwargs):
        commands.append(command)

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://provider.example/v1")

    agent = SkillEvaluatorLocalClaudeCode(logs_dir=tmp_path, model_name="aws/anthropic/bedrock-claude-opus-4-6")
    asyncio.run(agent.run("do the thing", environment=object(), context=object()))

    run_command = next(command for command in commands if "claude --verbose" in command)
    assert "--permission-mode=auto" in run_command
    assert "--permission-mode=bypassPermissions" not in run_command


def test_local_claude_does_not_rewrite_instruction_permission_text(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    async def fake_parent_exec(self, environment, command, env=None, **kwargs):
        calls.append((command, dict(env or {})))

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://provider.example/v1")

    instruction = "quote --permission-mode=bypassPermissions literally\nwithout rewriting 'bytes'"
    agent = SkillEvaluatorLocalClaudeCode(logs_dir=tmp_path, model_name="aws/anthropic/bedrock-claude-opus-4-6")
    asyncio.run(
        agent.run(
            instruction,
            environment=object(),
            context=object(),
        )
    )

    run_command, run_env = next((command, env) for command, env in calls if "claude --verbose" in command)
    instruction_vars = {
        key: value for key, value in run_env.items() if key.startswith("HARBOR_") and "_INSTRUCTION_" in key
    }
    assert instruction_vars and list(instruction_vars.values()) == [instruction]
    assert next(iter(instruction_vars.values())).encode() == instruction.encode()
    assert "--permission-mode=auto" in run_command
    assert "--permission-mode=bypassPermissions" not in run_command
    assert instruction not in run_command


def test_nvidia_build_codex_final_config_keeps_dynamic_bridge_over_user_and_runtime_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _record_installed_agent_exec(monkeypatch)
    environment = _RecordingEnvironment()
    bridge_origin = "http://127.0.0.1:43123"
    agent = SkillEvaluatorNvidiaBuildCodex(
        logs_dir=tmp_path,
        model_name="nvidia/meta/llama-3.1-8b-instruct",
        extra_env={
            "OPENAI_API_KEY": "external-key",
            "OPENAI_BASE_URL": "https://runtime-must-not-win.example/v1",
        },
        config={
            "approval_policy": "never",
            "openai_base_url": "https://user-must-not-win.example/v1",
            "model_provider": "user-provider",
            "model_providers": {
                "openai_compatible": {
                    "base_url": "https://user-must-not-win.example/v1",
                    "http_headers": {"X-User": "preserved"},
                }
            },
            "mcp_servers": {"user-tools": {"url": "https://user-tools.example/mcp"}},
        },
    )

    async def fake_start_bridge(_environment: object) -> None:
        agent._nvidia_build_bridge_started = True
        agent._nvidia_build_bridge_origin = bridge_origin
        agent._nvidia_build_bridge_client_token = "bridge-client-token"

    async def fake_cleanup_bridge(_environment: object) -> None:
        agent._nvidia_build_bridge_started = False
        agent._nvidia_build_bridge_origin = None
        agent._nvidia_build_bridge_client_token = None

    monkeypatch.setattr(agent, "_start_bridge", fake_start_bridge)
    monkeypatch.setattr(agent, "_cleanup_bridge", fake_cleanup_bridge)

    asyncio.run(agent.run("do the thing", environment=environment, context=object()))

    config = environment.last_codex_config(Codex._REMOTE_CODEX_HOME)
    assert config["approval_policy"] == "never"
    assert config["mcp_servers"] == {"user-tools": {"url": "https://user-tools.example/mcp"}}
    assert config["model_provider"] == "openai_compatible"
    assert config["openai_base_url"] == f"{bridge_origin}/v1"
    assert config["model_providers"]["openai_compatible"] == {
        "name": "OpenAI-compatible provider",
        "base_url": f"{bridge_origin}/v1",
        "env_key": "OPENAI_API_KEY",
        "wire_api": "responses",
        "http_headers": {"X-User": "preserved"},
    }
    run_command, run_env = next((command, env) for command, env in calls if "codex exec" in command)
    assert "--model nvidia/meta/llama-3.1-8b-instruct " in run_command
    assert run_env["OPENAI_API_KEY"] == "bridge-client-token"
    assert "OPENAI_BASE_URL" not in run_env
    assert "external-key" not in run_command


def test_nvidia_build_codex_get_env_delegates_alternatives_and_protects_bridge_values(
    tmp_path: Path,
) -> None:
    agent = SkillEvaluatorNvidiaBuildCodex(
        logs_dir=tmp_path,
        model_name="nvidia/meta/llama-3.1-8b-instruct",
        extra_env={
            "FALLBACK_TOKEN": "fallback-value",
            "OPENAI_API_KEY": "external-key",
            "OPENAI_BASE_URL": "https://external.example/v1",
        },
    )

    assert agent._get_env("PRIMARY_TOKEN", "FALLBACK_TOKEN") == "fallback-value"

    agent._nvidia_build_bridge_client_env = {"OPENAI_API_KEY": "bridge-client-token"}
    assert agent._get_env("PRIMARY_TOKEN", "FALLBACK_TOKEN") == "fallback-value"
    assert agent._get_env("PRIMARY_TOKEN", "OPENAI_API_KEY") == "bridge-client-token"
    assert agent._get_env("OPENAI_BASE_URL", "OPENAI_API_BASE") is None


def test_nvidia_build_codex_bridge_scope_wins_real_harbor_env_merge_and_resets(
    tmp_path: Path,
) -> None:
    environment = _MergingRecordingEnvironment(tmp_path)
    external_values = {
        "OPENAI_API_KEY": "external-openai-key",
        "OPENAI_BASE_URL": "https://external-base.example/v1",
        "OPENAI_API_BASE": "https://external-alias.example/v1",
        "NVIDIA_API_KEY": "external-nvidia-key",
        "OUTER_ONLY": "outer-value",
    }
    agent = SkillEvaluatorNvidiaBuildCodex(
        logs_dir=tmp_path,
        model_name="nvidia/meta/llama-3.1-8b-instruct",
        extra_env=external_values,
    )
    agent._nvidia_build_bridge_origin = "http://127.0.0.1:43123"
    agent._nvidia_build_bridge_client_env = {"OPENAI_API_KEY": "bridge-client-token"}
    snapshots: dict[str, dict[str, str] | None] = {}

    async def exercise() -> None:
        snapshots["before"] = environment._merge_env(None)
        with environment.scoped_exec_env(agent.extra_env):
            snapshots["outer_before"] = environment._merge_env(None)
            await agent.exec_as_agent(environment, command="env")
            snapshots["outer_after"] = environment._merge_env(None)
        snapshots["after"] = environment._merge_env(None)

    asyncio.run(exercise())

    assert environment.merged_exec_envs[-1]["OPENAI_API_KEY"] == "bridge-client-token"
    child_env = environment.actual_child_envs[-1]
    assert child_env["OPENAI_API_KEY"] == "bridge-client-token"
    assert child_env["OUTER_ONLY"] == "outer-value"
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "NVIDIA_API_KEY"):
        assert name not in child_env
    external_secret_values = {
        external_values[name] for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "NVIDIA_API_KEY")
    }
    assert not external_secret_values.intersection(child_env.values())

    command = environment.commands[-1]
    assert "env -u NVIDIA_API_KEY -u OPENAI_BASE_URL -u OPENAI_API_BASE bash -c env" in command
    assert "bridge-client-token" not in command
    assert not any(value in command for value in external_values.values())

    assert snapshots["before"] == {"PERSISTENT_VALUE": "persistent"}
    assert snapshots["outer_before"] == {"PERSISTENT_VALUE": "persistent", **external_values}
    assert snapshots["outer_after"] == snapshots["outer_before"]
    assert snapshots["after"] == snapshots["before"]
    assert environment._exec_env_overlays.get() == ()


def test_nvidia_build_claude_client_scope_wins_outer_agent_env(
    tmp_path: Path,
) -> None:
    environment = _MergingRecordingEnvironment(tmp_path)
    agent = SkillEvaluatorNvidiaBuildClaudeCode(
        logs_dir=tmp_path,
        model_name="nvidia/nemotron-3-super-120b-a12b",
        extra_env={
            "ANTHROPIC_API_KEY": "external-anthropic-key",
            "ANTHROPIC_BASE_URL": "https://external-anthropic.example",
            "ANTHROPIC_MODEL": "external-model",
            "NVIDIA_API_KEY": "external-nvidia-key",
        },
    )
    agent._nvidia_build_bridge_client_token = "bridge-client-token"
    agent._nvidia_build_bridge_origin = "http://127.0.0.1:43123"
    client_env = agent._bridge_client_environment()
    agent._nvidia_build_bridge_client_env = client_env

    async def exercise() -> None:
        with environment.scoped_exec_env(agent.extra_env):
            await agent.exec_as_agent(environment, command="env")

    asyncio.run(exercise())

    child_env = environment.actual_child_envs[-1]
    for name, value in client_env.items():
        assert child_env[name] == value
    assert "NVIDIA_API_KEY" not in child_env
    assert "external-anthropic-key" not in child_env.values()
    assert "external-model" not in child_env.values()
    assert "bridge-client-token" not in environment.commands[-1]


def test_local_codex_preserves_full_gateway_model_name(monkeypatch, tmp_path) -> None:
    captured: dict[str, str] = {}

    async def fake_parent_exec(self, environment, command, **kwargs):
        captured["command"] = command

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )

    agent = SkillEvaluatorLocalCodex(logs_dir=tmp_path, model_name="openai/openai/gpt-5.4")
    asyncio.run(
        agent.exec_as_agent(
            object(),
            "codex exec --model gpt-5.4 --json -- 'hello'",
            env={"OPENAI_BASE_URL": "https://provider.example/v1"},
        )
    )

    assert "--model openai/openai/gpt-5.4 " in captured["command"]
    assert "--model gpt-5.4 " not in captured["command"]


def test_local_codex_keeps_short_model_without_gateway(monkeypatch, tmp_path) -> None:
    captured: dict[str, str] = {}

    async def fake_parent_exec(self, environment, command, **kwargs):
        captured["command"] = command

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )

    agent = SkillEvaluatorLocalCodex(logs_dir=tmp_path, model_name="openai/openai/gpt-5.4")
    asyncio.run(
        agent.exec_as_agent(
            object(),
            "codex exec --model gpt-5.4 --json -- 'hello'",
            env={},
        )
    )

    assert "--model gpt-5.4 " in captured["command"]


def test_local_codex_preserves_instruction_text_during_launcher_rewrites(monkeypatch, tmp_path) -> None:
    captured: dict[str, str] = {}

    async def fake_parent_exec(self, environment, command, **kwargs):
        captured["command"] = command

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )

    agent = SkillEvaluatorLocalCodex(logs_dir=tmp_path, model_name="openai/openai/gpt-5.4")
    asyncio.run(
        agent.exec_as_agent(
            object(),
            "codex exec --model gpt-5.4 --json -- 'say --model gpt-5.4 and /tmp/codex-secrets literally'",
            env={"OPENAI_BASE_URL": "https://provider.example/v1"},
        )
    )

    launcher, _separator, prompt = captured["command"].partition(" -- ")
    assert "--model openai/openai/gpt-5.4 " in launcher
    assert "--model gpt-5.4 " not in launcher
    assert "--model gpt-5.4" in prompt
    assert "/tmp/codex-secrets" in prompt


def test_local_opencode_supports_nvidia_provider_without_harbor_patch(monkeypatch, tmp_path) -> None:
    commands: list[str] = []
    envs: list[dict[str, str]] = []

    async def fake_parent_exec(self, environment, command, **kwargs):
        commands.append(command)
        envs.append(kwargs.get("env") or {})

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")

    agent = SkillEvaluatorLocalOpenCode(logs_dir=tmp_path, model_name="nvidia/openai/openai/gpt-5.4")
    asyncio.run(agent.run("do the thing", environment=object(), context=object()))

    config_command = next(command for command in commands if "opencode.json" in command)
    run_command = next(command for command in commands if "opencode run" in command)
    assert "@ai-sdk/openai-compatible" in config_command
    assert "{env:OPENAI_API_KEY}" in config_command
    assert "opencode run --model=nvidia/openai/openai/gpt-5.4 --format=json" in run_command
    assert "stdbuf" not in run_command
    assert "--dangerously-skip-permissions" not in run_command
    assert any(env.get("OPENAI_API_KEY") == "sk-test" for env in envs)
    assert any(env.get("OPENAI_BASE_URL") == "https://provider.example/v1" for env in envs)


def test_local_opencode_non_nvidia_fallback_renders_prompt_once(monkeypatch, tmp_path) -> None:
    commands: list[str] = []
    render_count = 0

    async def fake_exec_as_agent(_environment, command, **_kwargs):
        commands.append(command)

    def fake_render(instruction: str) -> str:
        nonlocal render_count
        render_count += 1
        return f"rendered:{instruction}"

    agent = SkillEvaluatorLocalOpenCode(logs_dir=tmp_path, model_name="openai/openai/gpt-5.4")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)
    monkeypatch.setattr(agent, "render_instruction", fake_render)

    asyncio.run(agent.run("do the thing", environment=object(), context=object()))

    assert render_count == 1
    assert any("rendered:do the thing" in command for command in commands)


def test_local_opencode_removes_docker_only_stdbuf(monkeypatch, tmp_path) -> None:
    captured: dict[str, str] = {}

    async def fake_parent_exec(self, environment, command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env") or {}

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )

    agent = SkillEvaluatorLocalOpenCode(logs_dir=tmp_path, model_name="nvidia/test")
    asyncio.run(
        agent.exec_as_agent(
            object(),
            "opencode run --dangerously-skip-permissions 2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt",
            env={"OPENAI_API_KEY": "test"},
        )
    )

    assert "stdbuf" not in captured["command"]
    assert "--dangerously-skip-permissions" not in captured["command"]
    assert captured["command"].endswith("| tee /logs/agent/opencode.txt")
    assert captured["env"]["OPENAI_API_KEY"] == "test"
