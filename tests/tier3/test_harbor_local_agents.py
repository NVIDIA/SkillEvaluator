# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shlex
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.codex import Codex
from harbor.models.trial.paths import EnvironmentPaths

from skillevaluator.tier3.harbor.local_agents import (
    SkillEvaluatorClaudeCode,
    SkillEvaluatorLocalClaudeCode,
    SkillEvaluatorLocalCodex,
    SkillEvaluatorLocalOpenCode,
)


def test_local_codex_uses_per_trial_codex_home() -> None:
    assert str(SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME).startswith(EnvironmentPaths.agent_dir.as_posix())
    assert str(SkillEvaluatorLocalCodex._REMOTE_CODEX_SECRETS_DIR).startswith(EnvironmentPaths.agent_dir.as_posix())
    assert str(SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME) != "/tmp/codex-home"
    assert str(SkillEvaluatorLocalCodex._REMOTE_CODEX_SECRETS_DIR) != "/tmp/codex-secrets"


def test_local_codex_creates_per_trial_state_dirs(monkeypatch, tmp_path) -> None:
    commands: list[str] = []

    async def fake_exec_as_agent(_environment, command, **_kwargs):
        commands.append(command)

    async def fake_parent_run(self, instruction, environment, context):
        return None

    agent = SkillEvaluatorLocalCodex(logs_dir=tmp_path, model_name="openai/test")
    monkeypatch.setattr(agent, "exec_as_agent", fake_exec_as_agent)
    monkeypatch.setattr(Codex, "run", fake_parent_run)

    asyncio.run(agent.run("do the thing", environment=object(), context=object()))

    setup_command = commands[0]
    assert f"mkdir -p {SkillEvaluatorLocalCodex._REMOTE_CODEX_HOME.as_posix()}" in setup_command
    assert SkillEvaluatorLocalCodex._REMOTE_CODEX_SECRETS_DIR.as_posix() in setup_command


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
    asyncio.run(
        agent.run(
            "quote --permission-mode=bypassPermissions literally",
            environment=object(),
            context=object(),
        )
    )

    run_command = next(command for command in commands if "claude --verbose" in command)
    launcher, _separator, prompt = run_command.partition(" -- ")
    assert "--permission-mode=auto" in launcher
    assert "--permission-mode=bypassPermissions" not in launcher
    assert "--permission-mode=bypassPermissions" in prompt


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


def test_claude_mcp_servers_command_with_mcp_json(tmp_path: Path) -> None:
    """Build MCP configuration command with streamable-http and headers from task mcp_servers.json."""
    agent_logs = tmp_path / "agent"
    agent_logs.mkdir(parents=True)
    task_dir = tmp_path / "task-001"
    task_dir.mkdir(parents=True)

    mcp_servers = [
        {
            "name": "developer-knowledge",
            "transport": "streamable-http",
            "url": "https://developerknowledge.googleapis.com/mcp",
            "headers": {"X-Goog-Api-Key": "${DEVELOPERKNOWLEDGE_API_KEY}"},
        }
    ]
    (task_dir / "mcp_servers.json").write_text(json.dumps(mcp_servers), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"task": {"path": str(task_dir)}}), encoding="utf-8")

    agent = SkillEvaluatorClaudeCode(logs_dir=agent_logs, model_name="claude-sonnet-5")
    command = agent._build_register_mcp_servers_command()
    assert command is not None
    assert command.startswith('CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME}" && mkdir -p "$CLAUDE_DIR"')
    assert ' > "$CLAUDE_DIR/.claude.json"' in command

    # Extract and parse JSON payload
    _, _, suffix = command.partition("printf '%s\\n' ")
    payload_part, _, _ = suffix.partition(" > ")
    json_str = shlex.split(payload_part)[0]
    parsed = json.loads(json_str)

    assert "developer-knowledge" in parsed["mcpServers"]
    dk = parsed["mcpServers"]["developer-knowledge"]
    assert dk["type"] == "http"
    assert dk["url"] == "https://developerknowledge.googleapis.com/mcp"
    assert dk["headers"] == {"X-Goog-Api-Key": "${DEVELOPERKNOWLEDGE_API_KEY}"}


def test_claude_mcp_servers_command_stdio(tmp_path: Path) -> None:
    """Build MCP configuration command with stdio transport, env, and command arguments."""
    agent_logs = tmp_path / "agent"
    agent_logs.mkdir(parents=True)
    task_dir = tmp_path / "task-001"
    task_dir.mkdir(parents=True)

    mcp_servers = [
        {
            "name": "local-tool",
            "transport": "stdio",
            "command": "python3",
            "args": ["-m", "my_mcp_server"],
            "env": {"DATABASE_URL": "sqlite:///local.db"},
        }
    ]
    (task_dir / "mcp_servers.json").write_text(json.dumps(mcp_servers), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"task": {"path": str(task_dir)}}), encoding="utf-8")

    agent = SkillEvaluatorClaudeCode(logs_dir=agent_logs, model_name="claude-sonnet-5")
    command = agent._build_register_mcp_servers_command()
    assert command is not None
    assert command.startswith('CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME}" && mkdir -p "$CLAUDE_DIR"')
    assert ' > "$CLAUDE_DIR/.claude.json"' in command

    _, _, suffix = command.partition("printf '%s\\n' ")
    payload_part, _, _ = suffix.partition(" > ")
    json_str = shlex.split(payload_part)[0]
    parsed = json.loads(json_str)

    assert "local-tool" in parsed["mcpServers"]
    lt = parsed["mcpServers"]["local-tool"]
    assert lt["type"] == "stdio"
    assert lt["command"] == "python3"
    assert lt["args"] == ["-m", "my_mcp_server"]
    assert lt["env"] == {"DATABASE_URL": "sqlite:///local.db"}


def test_claude_mcp_servers_relative_task_path(tmp_path: Path) -> None:
    """Anchor relative task path to trial directory instead of host process CWD."""
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir(parents=True)
    agent_logs = trial_dir / "agent"
    agent_logs.mkdir(parents=True)
    task_dir = trial_dir / "relative-task"
    task_dir.mkdir(parents=True)

    mcp_servers = [
        {
            "name": "relative-mcp",
            "transport": "stdio",
            "command": "node",
            "args": ["server.js"],
        }
    ]
    (task_dir / "mcp_servers.json").write_text(json.dumps(mcp_servers), encoding="utf-8")
    (trial_dir / "config.json").write_text(json.dumps({"task": {"path": "relative-task"}}), encoding="utf-8")

    agent = SkillEvaluatorClaudeCode(logs_dir=agent_logs, model_name="claude-sonnet-5")
    command = agent._build_register_mcp_servers_command()
    assert command is not None
    assert "relative-mcp" in command


def test_claude_mcp_servers_command_fallback_when_no_mcp_json(tmp_path: Path) -> None:
    """Fall back to base MCP registration command when no task mcp_servers.json exists."""
    agent_logs = tmp_path / "agent"
    agent_logs.mkdir(parents=True)
    task_dir = tmp_path / "task-001"
    task_dir.mkdir(parents=True)
    (tmp_path / "config.json").write_text(json.dumps({"task": {"path": str(task_dir)}}), encoding="utf-8")

    agent = SkillEvaluatorClaudeCode(logs_dir=agent_logs, model_name="claude-sonnet-5")
    assert agent._build_register_mcp_servers_command() is None


def test_claude_mcp_servers_command_resilient_to_missing_config_json(tmp_path: Path) -> None:
    """Handle missing or malformed config.json without raising errors."""
    agent_logs = tmp_path / "agent"
    agent_logs.mkdir(parents=True)

    agent = SkillEvaluatorClaudeCode(logs_dir=agent_logs, model_name="claude-sonnet-5")
    assert agent._build_register_mcp_servers_command() is None


def test_claude_mcp_servers_command_nested_logs_dir_layout(tmp_path: Path) -> None:
    """Resolve task config.json from trial root when logs_dir is nested as <trial>/logs/agent."""
    trial_dir = tmp_path / "trial_001"
    agent_logs = trial_dir / "logs" / "agent"
    agent_logs.mkdir(parents=True)
    task_dir = trial_dir / "task"
    task_dir.mkdir(parents=True)

    mcp_servers = [{"name": "nested-mcp", "transport": "stdio", "command": "echo", "args": ["hi"]}]
    (task_dir / "mcp_servers.json").write_text(json.dumps(mcp_servers), encoding="utf-8")
    (trial_dir / "config.json").write_text(json.dumps({"task": {"path": str(task_dir)}}), encoding="utf-8")

    agent = SkillEvaluatorClaudeCode(logs_dir=agent_logs, model_name="claude-sonnet-5")
    command = agent._build_register_mcp_servers_command()
    assert command is not None
    assert "nested-mcp" in command
