# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise model routing through the installed Harbor Claude launcher."""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from harbor.agents.factory import AgentFactory
from harbor.models.trial.config import AgentConfig
from harbor.utils.import_path import import_symbol

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorClaudeCode, SkillEvaluatorLocalClaudeCode
from skillevaluator.tier3.harbor.runner import _provider_agent_import_path


@pytest.mark.parametrize("env_mode", sorted(runner.HARBOR_ENV_MODES))
@pytest.mark.parametrize("model", ["azure/anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"])
def test_gateway_claude_launcher_preserves_catalog_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env_mode: str, model: str
) -> None:
    for name in (
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_FORCE_OAUTH",
    ):
        monkeypatch.delenv(name, raising=False)
    provider = ProviderConfig(
        provider="openai-compatible",
        model=model,
        api_key="test-key",
        base_url="https://gateway.example/v1",
        litellm_model=f"openai/{model}",
    )
    monkeypatch.setattr(runner, "_harbor_supports_yes", lambda: False)
    selected_model, _ = runner._model_for_agent("claude-code", cli_model=model, config_agents={}, provider=provider)
    command = runner.build_harbor_run_command(
        dataset_path=tmp_path / "tasks",
        agent="claude-code",
        job_name="gateway-claude",
        env_mode=env_mode,
        model=selected_model,
        agent_import_path=runner._agent_import_path(provider, "claude-code", env_mode),
    )
    selected_agent = command[command.index("--agent") + 1]
    selected_model = command[command.index("--model") + 1]
    calls = []

    async def capture_exec(self, environment, command, **kwargs):
        calls.append((command, kwargs.get("env", {})))

    # Only replace the external process boundary; Harbor builds the real launch environment.
    monkeypatch.setattr("harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent", capture_exec)
    agent = AgentFactory.create_agent_from_config(
        AgentConfig(
            import_path=selected_agent,
            model_name=selected_model,
            env={"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_BASE_URL": "https://gateway.example"},
        ),
        logs_dir=tmp_path,
    )
    if env_mode == "local":
        assert isinstance(agent, SkillEvaluatorLocalClaudeCode)
    asyncio.run(agent.run("Return OK", environment=object(), context=object()))
    env = next(env for command, env in calls if "claude --verbose" in command)
    assert env["ANTHROPIC_MODEL"] == model
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == model
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == model
    assert env["ANTHROPIC_BASE_URL"] == "https://gateway.example"


@pytest.mark.parametrize(
    ("model", "extra_env", "expected"),
    [
        ("anthropic/claude-sonnet-4-6", {"ANTHROPIC_API_KEY": "test-key"}, "claude-sonnet-4-6"),
        ("vertex_ai/claude-sonnet-4-6", {"CLAUDE_CODE_USE_VERTEX": "1"}, "claude-sonnet-4-6"),
        ("bedrock/us.anthropic.claude-sonnet-4-6", {}, "us.anthropic.claude-sonnet-4-6"),
    ],
)
def test_native_claude_model_resolution_is_unchanged(tmp_path, model, extra_env, expected):
    agent = SkillEvaluatorClaudeCode(logs_dir=tmp_path, model_name=model, extra_env=extra_env)
    assert agent._resolved_model_name() == expected


@pytest.mark.parametrize("env_mode", ["docker", "kata", "gke", "e2b", "local"])
def test_native_anthropic_does_not_select_gateway_wrapper(env_mode):
    provider = ProviderConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        litellm_model="anthropic/claude-sonnet-4-6",
    )
    assert _provider_agent_import_path(provider, "claude-code", env_mode) is None


@pytest.mark.parametrize("env_mode", ["docker", "local"])
@pytest.mark.parametrize(
    ("model", "extra_env", "expected"),
    [
        ("anthropic/claude-sonnet-4-6", {"ANTHROPIC_API_KEY": "test-key"}, "claude-sonnet-4-6"),
        (
            "anthropic/claude-sonnet-4-6",
            {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
            "claude-sonnet-4-6",
        ),
        ("vertex_ai/claude-sonnet-4-6", {"CLAUDE_CODE_USE_VERTEX": "1"}, "claude-sonnet-4-6"),
        ("bedrock/us.anthropic.claude-sonnet-4-6", {}, "us.anthropic.claude-sonnet-4-6"),
    ],
)
def test_gateway_judge_keeps_independent_claude_route(monkeypatch, tmp_path, env_mode, model, extra_env, expected):
    for name in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_FORCE_OAUTH",
    ):
        monkeypatch.delenv(name, raising=False)
    provider = ProviderConfig(
        provider="openai-compatible",
        model="gateway-judge",
        api_key="gateway-key",
        base_url="https://gateway.example/v1",
        litellm_model="openai/gateway-judge",
    )
    agent_type = import_symbol(_provider_agent_import_path(provider, "claude-code", env_mode))
    calls = []

    async def capture_exec(self, environment, command, **kwargs):
        calls.append((command, kwargs.get("env", {})))

    monkeypatch.setattr("harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent", capture_exec)
    agent = agent_type(logs_dir=tmp_path, model_name=model, extra_env=extra_env)
    asyncio.run(agent.run("Return OK", environment=object(), context=object()))
    env = next(env for command, env in calls if "claude --verbose" in command)
    assert env["ANTHROPIC_MODEL"] == expected
