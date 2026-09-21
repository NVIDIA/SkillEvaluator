# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import local_agents, runner


@pytest.mark.parametrize("provider", ["openai", "anthropic", "nv_build", "openai-compatible"])
@pytest.mark.parametrize("mode", ["local", "docker", "e2b"])
@pytest.mark.parametrize("agent", ["codex", "claude-code", "opencode"])
def test_gateway_adapter_selection_preserves_other_routes(provider, mode, agent):
    config = ProviderConfig(
        provider=provider,
        model="vendor/model",
        api_key="test-key",
        base_url="https://example.com/v1",
        litellm_model="openai/vendor/model",
    )
    expected = runner._nvidia_build_agent_import_path(config, agent, mode)
    if provider == "openai-compatible" and mode == "docker" and agent == "codex":
        expected = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorGatewayCodex"
    if provider == "openai-compatible" and mode == "docker" and agent == "opencode":
        expected = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorGatewayOpenCode"
    assert runner._agent_import_path(config, agent, mode) == expected


@pytest.mark.parametrize("model", ["azure/openai/gpt-5.6-sol", "gpt-5.6-sol"])
def test_gateway_codex_preserves_model_and_prompt(monkeypatch, tmp_path, model):
    commands = []

    async def capture(self, environment, command, **kwargs):
        commands.append(command)

    monkeypatch.setattr("harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent", capture)
    agent = local_agents.SkillEvaluatorGatewayCodex(logs_dir=tmp_path, model_name=model)
    command = "codex exec --model gpt-5.6-sol --json -- 'quote --model gpt-5.6-sol literally'"
    asyncio.run(agent.exec_as_agent(object(), command, env={"OPENAI_BASE_URL": "https://example.com/v1"}))
    launcher, _, prompt = commands[0].partition(" -- ")
    assert f"--model {model} " in launcher
    assert prompt == "'quote --model gpt-5.6-sol literally'"
