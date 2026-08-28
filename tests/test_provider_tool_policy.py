# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-aware Claude Code server-tool policy contracts."""

from __future__ import annotations

import asyncio

import pytest

from skillevaluator.evaluation import EvaluationOptions, EvaluationService
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import local_agents, runner
from skillevaluator.tier3.harbor.provider_tool_policy import (
    CLAUDE_SERVER_TOOL_POLICY_BRIDGE_V1,
    CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1,
    CLAUDE_SERVER_TOOL_POLICY_ENV,
    CLAUDE_SERVER_TOOL_POLICY_NATIVE_V1,
    SERVER_TOOL_POLICY_PROVIDER_COMPATIBLE_V1,
    resolve_claude_server_tool_policy,
    validate_server_tool_policy,
)


def _provider(name: str, *, base_url: str | None) -> ProviderConfig:
    return ProviderConfig(
        provider=name,
        model="test-model",
        api_key="provider-key",
        base_url=base_url,
        litellm_model=("anthropic/test-model" if name == "anthropic" else "openai/test-model"),
    )


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (_provider("anthropic", base_url=None), CLAUDE_SERVER_TOOL_POLICY_NATIVE_V1),
        (_provider("anthropic", base_url="https://api.anthropic.com"), CLAUDE_SERVER_TOOL_POLICY_NATIVE_V1),
        (
            _provider("anthropic", base_url="https://inference-api.nvidia.com"),
            CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1,
        ),
        (
            _provider("openai-compatible", base_url="https://gateway.example/v1"),
            CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1,
        ),
        (
            _provider("nv_build", base_url="https://integrate.api.nvidia.com/v1"),
            CLAUDE_SERVER_TOOL_POLICY_BRIDGE_V1,
        ),
        (_provider("bedrock", base_url=None), CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1),
    ],
)
def test_provider_compatible_policy_resolves_from_actual_route(
    provider: ProviderConfig,
    expected: str,
) -> None:
    assert resolve_claude_server_tool_policy(provider, SERVER_TOOL_POLICY_PROVIDER_COMPATIBLE_V1) == expected


def test_unknown_public_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="server_tool_policy"):
        validate_server_tool_policy("skill-controlled")


def test_gateway_claude_runtime_plan_carries_operator_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "ANTHROPIC_API_KEY": "agent-key",
            "ANTHROPIC_BASE_URL": "https://inference-api.nvidia.com",
        },
    )

    plan = runner._resolve_agent_runtime_plan(
        provider=_provider("openai-compatible", base_url="https://inference-api.nvidia.com"),
        agents=["claude-code"],
        models={"claude-code": "aws/anthropic/bedrock-claude-opus-4-6"},
        configured_runtime_env={},
        env_mode="local",
        model_sources={"claude-code": "CLI"},
        server_tool_policy=SERVER_TOOL_POLICY_PROVIDER_COMPATIBLE_V1,
    )["claude-code"]

    assert plan.server_tool_policy == CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1
    assert plan.subprocess_env[CLAUDE_SERVER_TOOL_POLICY_ENV] == CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1
    assert CLAUDE_SERVER_TOOL_POLICY_ENV not in plan.staged_env


def test_disabled_policy_injects_web_tools_before_prompt_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CLAUDE_SERVER_TOOL_POLICY_ENV, CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1)
    command = "claude --verbose --output-format=stream-json --print -- 'search for evidence'"

    rewritten = local_agents._apply_claude_server_tool_policy(command)

    launcher, separator, prompt = rewritten.partition(" -- ")
    assert "--disallowedTools WebSearch,WebFetch" in launcher
    assert separator == " -- "
    assert prompt == "'search for evidence'"


def test_gateway_local_claude_wrapper_enforces_policy_on_real_run_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    async def fake_parent_exec(self, environment, command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env") or {}

    monkeypatch.setattr(
        "harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent",
        fake_parent_exec,
    )
    monkeypatch.setenv(CLAUDE_SERVER_TOOL_POLICY_ENV, CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1)

    agent = local_agents.SkillEvaluatorLocalClaudeCode(
        logs_dir=tmp_path,
        model_name="aws/anthropic/bedrock-claude-opus-4-6",
    )
    asyncio.run(
        agent.exec_as_agent(
            object(),
            "claude --verbose --permission-mode=bypassPermissions --print -- 'search for evidence'",
            env={"ANTHROPIC_API_KEY": "test-key"},
        )
    )

    launcher, separator, prompt = str(captured["command"]).partition(" -- ")
    assert "--permission-mode=auto" in launcher
    assert "--permission-mode=bypassPermissions" not in launcher
    assert "--disallowedTools WebSearch,WebFetch" in launcher
    assert separator == " -- "
    assert prompt == "'search for evidence'"


def test_native_policy_preserves_claude_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CLAUDE_SERVER_TOOL_POLICY_ENV, CLAUDE_SERVER_TOOL_POLICY_NATIVE_V1)
    command = "claude --verbose --output-format=stream-json --print -- 'use native search'"

    assert local_agents._apply_claude_server_tool_policy(command) == command


def test_policy_does_not_modify_setup_or_version_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CLAUDE_SERVER_TOOL_POLICY_ENV, CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1)

    assert local_agents._apply_claude_server_tool_policy("claude --version") == "claude --version"


def test_docker_policy_selects_enforcing_wrapper() -> None:
    provider = _provider("anthropic", base_url="https://gateway.example")

    assert (
        runner._agent_import_path(
            provider,
            "claude-code",
            "docker",
            server_tool_policy=SERVER_TOOL_POLICY_PROVIDER_COMPATIBLE_V1,
        )
        == "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorClaudeCode"
    )
    assert runner._agent_import_path(provider, "claude-code", "docker", server_tool_policy=None) is None


def test_evaluation_api_forwards_server_tool_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from skillevaluator.tier3 import commands

    captured: dict[str, object] = {}

    def fake_evaluate(skill_path, **kwargs):
        captured["skill_path"] = skill_path
        captured.update(kwargs)
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(commands, "evaluate", fake_evaluate)
    options = EvaluationOptions(
        skill_path=tmp_path,
        agents="claude-code",
        env_mode="local",
        server_tool_policy=SERVER_TOOL_POLICY_PROVIDER_COMPATIBLE_V1,
    )

    EvaluationService().evaluate(options)

    assert captured["server_tool_policy"] == SERVER_TOOL_POLICY_PROVIDER_COMPATIBLE_V1
