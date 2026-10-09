# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import shlex
from pathlib import Path

import pytest
import toml
from harbor.agents.factory import AgentFactory
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import local_agents, runner


@pytest.mark.parametrize("provider", ["openai", "anthropic", "nv_build", "openai-compatible"])
@pytest.mark.parametrize("mode", sorted(runner.HARBOR_ENV_MODES))
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
    if provider == "openai-compatible" and mode != "local" and agent == "codex":
        expected = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorGatewayCodex"
    if provider == "openai-compatible" and mode in {"docker", "kata"} and agent == "opencode":
        expected = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorGatewayOpenCode"
    assert runner._agent_import_path(config, agent, mode) == expected


@pytest.mark.parametrize("mode", ["local", "docker", "e2b", "daytona"])
@pytest.mark.parametrize(
    ("provider", "configured_model", "cli_model", "expected_model", "expected_source"),
    [
        ("openai-compatible", None, None, "openai/openai/gpt-5.6-sol", "openai-compatible agent default"),
        ("openai-compatible", "openai/openai/gpt-5.6-sol", None, "openai/openai/gpt-5.6-sol", "evals/config.yml"),
        ("openai-compatible", "openai/openai/gpt-5.6-sol", "azure/custom/deployment", "azure/custom/deployment", "CLI"),
        ("openai-compatible", None, "custom-deployment", "custom-deployment", "CLI"),
        ("openai", None, "openai/gpt-5.6-sol", "gpt-5.6-sol", "CLI"),
    ],
)
def test_selected_codex_launcher_preserves_provider_model_contract(
    monkeypatch, tmp_path, mode, provider, configured_model, cli_model, expected_model, expected_source
):
    """Exercise the built command, Harbor factory, and real Codex launcher together."""
    base_url = "https://gateway.example/v1" if provider == "openai-compatible" else None
    config = ProviderConfig(
        provider=provider,
        model="judge-model",
        api_key="test-key",
        base_url=base_url,
        litellm_model="openai/judge-model",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_FORCE_AUTH_JSON", raising=False)
    if base_url:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(runner, "_harbor_supports_yes", lambda: False)
    model, source = runner._model_for_agent(
        "codex",
        cli_model=cli_model,
        config_agents={"codex": {"model": configured_model}} if configured_model else {},
        provider=config,
    )
    assert source == expected_source
    command = runner.build_harbor_run_command(
        dataset_path=tmp_path / "tasks",
        agent="codex",
        job_name="gateway-launcher",
        env_mode=mode,
        model=model,
        agent_import_path=runner._agent_import_path(config, "codex", mode),
    )

    def option_value(flag):
        return command[command.index(flag) + 1] if flag in command else None

    if mode in {"e2b", "daytona"}:
        assert option_value("--env") == mode
        assert "--environment-import-path" not in command
    # Harbor's unified --agent selector carries either a built-in name or an import path.
    selected_agent = option_value("--agent")
    assert selected_agent is not None
    assert "-a" not in command
    assert "--agent-import-path" not in command
    agent = AgentFactory.create_agent_from_config(
        AgentConfig(
            name=None if ":" in selected_agent else selected_agent,
            import_path=selected_agent if ":" in selected_agent else None,
            model_name=option_value("--model"),
        ),
        logs_dir=tmp_path / "logs",
    )

    class RecordingEnvironment:
        default_user = None

        def __init__(self):
            self.commands = []
            self.uploads = {}

        async def exec(self, command, **kwargs):
            self.commands.append((command, kwargs.get("env") or {}))
            return ExecResult(return_code=0, stdout="", stderr="")

        async def upload_file(self, source_path, target_path):
            self.uploads[str(target_path)] = Path(source_path).read_text(encoding="utf-8")

    environment = RecordingEnvironment()
    instruction = "quote --model gpt-5.6-sol and /tmp/codex-secrets literally"
    asyncio.run(agent.run(instruction, environment=environment, context=AgentContext()))

    # Harbor uploads the effective Codex config; gateway routing must survive it.
    configs = [toml.loads(text) for path, text in environment.uploads.items() if path.endswith("config.toml")]
    if base_url:
        assert len(configs) == 1
        assert configs[0]["model_provider"] == "openai_compatible"
        assert configs[0]["model_providers"]["openai_compatible"]["base_url"] == base_url

    launchers = [(text, env) for text, env in environment.commands if "codex exec " in text]
    assert len(launchers) == 1
    launcher, launch_env = launchers[0]
    argv = shlex.split(launcher)
    assert argv[argv.index("--model") + 1] == expected_model
    assert argv[argv.index("--") + 1] == instruction
    assert launch_env.get("OPENAI_BASE_URL") == base_url


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
