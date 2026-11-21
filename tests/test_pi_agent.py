# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi Coding Agent compatibility at the public routing and Harbor boundaries."""

from __future__ import annotations

import asyncio
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.commands import validate_agents
from skillevaluator.tier3.harbor import (
    DOCKER_AGENT_IMPORT_PATHS,
    HARBOR_AGENTS_EXPERIMENTAL,
    LOCAL_AGENT_IMPORT_PATHS,
    LOCAL_HARBOR_AGENTS,
)
from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorLocalPi
from skillevaluator.tier3.harbor.local_runtime import PINNED_RUNTIME_VERSIONS
from skillevaluator.tier3.harbor.pi_agent import SkillEvaluatorPi
from skillevaluator.tier3.harbor.runner import (
    PI_VERSION,
    _agent_import_path,
    _harbor_agent_kwargs,
    _validate_agent_provider_credentials,
    build_harbor_run_command,
)


def _provider(name="openai", base_url="https://api.openai-compatible.example/v1"):
    return ProviderConfig(
        provider=name,
        model="glm-4.7",
        api_key="selected-provider-key",
        base_url=base_url,
        litellm_model="openai/glm-4.7",
    )


def _agent(tmp_path, **kwargs):
    agent = SkillEvaluatorPi(logs_dir=tmp_path, model_name=_provider().model, **kwargs)
    agent.ensure_system_dependencies = AsyncMock()
    agent.exec_as_agent = AsyncMock()
    return agent


def test_public_registration_and_pinned_docker_command(tmp_path):
    assert validate_agents(["pi"]) == []
    assert "pi" in HARBOR_AGENTS_EXPERIMENTAL
    assert _harbor_agent_kwargs("pi", _provider().model, "docker") == {"version": PI_VERSION}
    assert PINNED_RUNTIME_VERSIONS["pi"] == PI_VERSION
    # No provider-specific wrapper is required; the docker command resolves the pinned wrapper.
    assert _agent_import_path(_provider(), "pi", "docker") is None
    command = build_harbor_run_command(
        dataset_path=tmp_path, agent="pi", job_name="test", env_mode="docker", model=_provider().model
    )
    assert command[command.index("--agent") + 1] == DOCKER_AGENT_IMPORT_PATHS["pi"]
    assert f'version="{PI_VERSION}"' in command


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai-compatible"])
def test_public_providers_accept_pi(provider):
    # The Anthropic provider reaches pi on the public endpoint only.
    assert _validate_agent_provider_credentials(_provider(provider, base_url=None if provider == "anthropic" else "https://gw.example/v1"), ["pi"], {}) == []


def test_anthropic_base_url_rejects_pi():
    errors = _validate_agent_provider_credentials(_provider("anthropic"), ["pi"], {})
    assert errors and "no custom-endpoint path" in errors[0]


def test_bedrock_rejects_pi():
    errors = _validate_agent_provider_credentials(_provider("bedrock"), ["pi"], {})
    assert errors and "does not support live agent" in errors[0]


def test_install_pins_the_pinned_package(tmp_path):
    agent = _agent(tmp_path)
    asyncio.run(agent.install(SimpleNamespace()))
    command = agent.exec_as_agent.call_args.kwargs["command"]
    assert f"@earendil-works/pi-coding-agent@{PI_VERSION}" in command
    assert "pi --version" in command


def test_install_rejects_a_foreign_version(tmp_path):
    with pytest.raises(ValueError, match="requires"):
        asyncio.run(_agent(tmp_path, version="0.85.1").install(SimpleNamespace()))


def test_install_allows_mcp_servers_through_the_native_config(tmp_path):
    from harbor.models.task.config import MCPServerConfig

    server = MCPServerConfig(name="context7", transport="stdio", command="npx", args=["-y", "@upstash/context7-mcp"])
    agent = _agent(tmp_path, mcp_servers=[server])
    agent._upload_config_text = AsyncMock()
    asyncio.run(agent.install(SimpleNamespace()))
    commands = [call.kwargs["command"] for call in agent.exec_as_agent.call_args_list]
    npm = next(command for command in commands if "npm install -g" in command)
    assert "pi-mcp-adapter" not in npm  # pi 1.5.2 reads mcp.json natively
    assert any("mkdir -p" in command and "pi-mcp-adapter" in command for command in commands)
    kwargs = agent._upload_config_text.call_args.kwargs
    assert kwargs["remote_path"] == "/tmp/harbor-pi-mcp-adapter/node_modules/pi-mcp-adapter/index.ts"
    assert kwargs["content"].startswith("export default function")
    assert agent.mcp_servers == [server]  # restored for Harbor's mcp.json upload



def test_mcp_headers_are_preserved_as_env_references(tmp_path):
    from harbor.models.task.config import MCPServerConfig

    assert SkillEvaluatorPi.capabilities.mcp_http_headers, "Harbor's trial gate rejects header servers otherwise"
    server = MCPServerConfig(
        name="docs",
        transport="streamable-http",
        url="https://docs.example/mcp",
        headers={"Authorization": "Bearer ${DOCS_TOKEN}"},
    )
    agent = _agent(tmp_path, mcp_servers=[server])
    agent.mcp_header_env = SimpleNamespace(
        refs={"docs": {"Authorization": "DOCS_TOKEN"}}, env={"DOCS_TOKEN": "secret-token"}
    )

    config = agent._build_mcp_config()

    assert config["mcpServers"]["docs"]["url"] == "https://docs.example/mcp"
    assert config["mcpServers"]["docs"]["headers"] == {"Authorization": "${DOCS_TOKEN}"}


def test_mcp_config_verification_fails_when_a_server_is_not_reported(tmp_path):
    from harbor.models.task.config import MCPServerConfig

    server = MCPServerConfig(name="docs", transport="stdio", command="npx", args=["-y", "@upstash/context7-mcp"])
    agent = _agent(tmp_path, mcp_servers=[server])
    agent._upload_config_text = AsyncMock()
    agent.mcp_header_env = SimpleNamespace(refs={}, env={})
    agent.exec_as_agent = AsyncMock(
        return_value=SimpleNamespace(stdout='{"servers": [{"name": "other", "state": "connected"}]}')
    )

    with pytest.raises(RuntimeError, match="did not report"):
        asyncio.run(agent._write_mcp_config(object()))


def test_mcp_config_verification_tolerates_unconnected_servers(tmp_path):
    from harbor.models.task.config import MCPServerConfig

    server = MCPServerConfig(name="docs", transport="stdio", command="npx", args=["-y", "@upstash/context7-mcp"])
    agent = _agent(tmp_path, mcp_servers=[server])
    agent._upload_config_text = AsyncMock()
    agent.mcp_header_env = SimpleNamespace(refs={}, env={})
    agent.exec_as_agent = AsyncMock(
        return_value=SimpleNamespace(stdout='{"servers": [{"name": "docs", "state": "failed"}]}')
    )

    asyncio.run(agent._write_mcp_config(object()))  # a broken task server is the task's own concern
    # pi exits 1 with this census; the probe swallows the code and reads it.
    assert agent.exec_as_agent.call_args.kwargs["command"].endswith("pi mcp list --json; true")


def test_mcp_config_verification_fails_when_the_probe_itself_fails(tmp_path):
    from harbor.models.task.config import MCPServerConfig

    server = MCPServerConfig(name="docs", transport="stdio", command="npx", args=["-y", "@upstash/context7-mcp"])
    agent = _agent(tmp_path, mcp_servers=[server])
    agent.mcp_header_env = SimpleNamespace(refs={}, env={})
    agent.exec_as_agent = AsyncMock(side_effect=RuntimeError("connection lost"))

    with pytest.raises(RuntimeError, match="pi mcp list failed"):
        asyncio.run(agent._verify_mcp_config(object()))


def test_header_variables_reach_the_launch_command(monkeypatch, tmp_path):
    from harbor.agents.installed.pi import Pi
    from harbor.models.task.config import MCPServerConfig

    agent = object.__new__(SkillEvaluatorPi)
    agent.mcp_servers = [MCPServerConfig(name="docs", transport="streamable-http", url="https://docs.example/mcp")]
    agent.mcp_header_env = SimpleNamespace(refs={}, env={"DOCS_TOKEN": "secret-token"})
    captured = {}

    async def raw_exec(_self, _environment, command, env=None, **_kwargs):
        captured["env"] = env

    monkeypatch.setattr(Pi, "exec_as_agent", raw_exec)

    asyncio.run(agent.exec_as_agent(object(), "pi --print x", env={"OPENAI_API_KEY": "k"}))

    assert captured["env"] == {"DOCS_TOKEN": "secret-token", "OPENAI_API_KEY": "k"}


def test_mcp_probe_carries_the_header_variables(monkeypatch, tmp_path):
    from harbor.agents.installed.pi import Pi
    from harbor.models.task.config import MCPServerConfig

    server = MCPServerConfig(name="docs", transport="streamable-http", url="https://docs.example/mcp")
    agent = _agent(tmp_path, mcp_servers=[server])
    agent.mcp_header_env = SimpleNamespace(refs={"DOCS_TOKEN": "x"}, env={"DOCS_TOKEN": "secret-token"})
    agent.__delattr__("exec_as_agent")  # run the real override, not the helper's mock
    captured = {}

    async def raw_exec(_self, _environment, command, env=None, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(stdout='{"servers": [{"name": "docs", "state": "connected"}]}')

    monkeypatch.setattr(Pi, "exec_as_agent", raw_exec)

    asyncio.run(agent._verify_mcp_config(object()))

    # Without the header variables pi cannot resolve the staged mcp.json
    # headers and reports every authed server failed at startup.
    assert captured["command"].endswith("pi mcp list --json; true")
    assert captured["env"] == {"DOCS_TOKEN": "secret-token"}


def test_local_version_probe_with_empty_output_stays_a_warning(tmp_path):
    agent = _local_agent(tmp_path)
    agent.exec_as_agent = AsyncMock(return_value=SimpleNamespace(stdout=""))

    asyncio.run(agent.install(object()))  # unparsable output must not fail the trial

    assert agent._version is None


def _local_agent(tmp_path, **kwargs):
    agent = SkillEvaluatorLocalPi(logs_dir=tmp_path, model_name=_provider().model, **kwargs)
    agent.ensure_system_dependencies = AsyncMock()
    agent.exec_as_agent = AsyncMock(return_value=SimpleNamespace(stdout="1.6.0\n"))
    return agent


def test_local_registration_and_command(tmp_path):
    assert "pi" in LOCAL_HARBOR_AGENTS
    assert LOCAL_AGENT_IMPORT_PATHS["pi"].endswith("SkillEvaluatorLocalPi")
    command = build_harbor_run_command(dataset_path=tmp_path, agent="pi", job_name="j", env_mode="local")
    assert command[command.index("--agent") + 1] == LOCAL_AGENT_IMPORT_PATHS["pi"]
    joined = " ".join(command)
    assert "runtime_agent=pi" in joined
    assert "version=" not in joined  # local mode runs the operator-installed CLI


def test_local_install_records_the_installed_cli(tmp_path):
    agent = _local_agent(tmp_path)
    agent.exec_as_agent = AsyncMock(return_value=SimpleNamespace(stdout="1.4.0\n"))
    asyncio.run(agent.install(object()))  # soft check: drift from the pin warns, never raises
    assert agent._version == "1.4.0"


def test_local_install_stages_the_mcp_shim_in_the_agent_dir(tmp_path):
    from harbor.models.task.config import MCPServerConfig

    server = MCPServerConfig(name="context7", transport="stdio", command="npx", args=["-y", "@upstash/context7-mcp"])
    agent = _local_agent(tmp_path, mcp_servers=[server])
    agent._upload_config_text = AsyncMock()
    asyncio.run(agent.install(object()))
    commands = [call.kwargs["command"] for call in agent.exec_as_agent.call_args_list]
    assert any("mkdir -p /logs/agent/pi-mcp-adapter" in command for command in commands)
    kwargs = agent._upload_config_text.call_args.kwargs
    assert kwargs["remote_path"] == "/logs/agent/pi-mcp-adapter/node_modules/pi-mcp-adapter/index.ts"
    assert kwargs["content"].startswith("export default function")


def test_local_rewrite_moves_the_task_text_into_an_instruction_env_var(monkeypatch):
    from harbor.agents.installed.pi import Pi

    from skillevaluator.tier3.harbor.local_environment import _AGENT_INSTRUCTION_ENV_RE

    agent = object.__new__(SkillEvaluatorLocalPi)
    agent.mcp_servers = []
    instruction = "Print the literal path /tmp/harbor-pi-agent"
    agent._instruction = instruction
    agent._quoted_instruction = shlex.quote(instruction)
    captured = {}

    async def raw_exec(_self, _environment, command, env=None, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(Pi, "exec_as_agent", raw_exec)

    asyncio.run(
        agent.exec_as_agent(
            object(),
            "PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent pi --print "
            + shlex.quote(instruction)
            + " 2>&1 </dev/null | stdbuf -oL tee /logs/agent/pi.txt",
        )
    )

    command = str(captured["command"])
    env = captured["env"]
    assert "PI_CODING_AGENT_DIR=/logs/agent/pi-agent" in command  # the config dir moved
    assert instruction not in command  # the task text left the command string
    (env_var,) = [key for key in env if key.startswith("PI_INSTRUCTION_")]
    assert env[env_var] == instruction  # byte-identical in the launch env
    assert _AGENT_INSTRUCTION_ENV_RE.fullmatch(env_var)  # the local env keeps it verbatim
    shell_var = env_var.lower()
    assert f'{shell_var}="${env_var}"; unset {env_var};' in command
    assert f'"{shell_var}" 2>&1 </dev/null' in command  # pi's prompt argument


def test_local_rewrite_keeps_a_bare_config_path_prompt(monkeypatch):
    from harbor.agents.installed.pi import Pi

    agent = object.__new__(SkillEvaluatorLocalPi)
    agent.mcp_servers = []
    instruction = "/tmp/harbor-pi-agent"  # a bare path: shlex.quote() adds no quotes
    agent._instruction = instruction
    agent._quoted_instruction = shlex.quote(instruction)
    captured = {}

    async def raw_exec(_self, _environment, command, env=None, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(Pi, "exec_as_agent", raw_exec)

    asyncio.run(
        agent.exec_as_agent(
            object(),
            "PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent pi --print --mode json --model m "
            + instruction
            + " 2>&1 </dev/null | stdbuf -oL tee /logs/agent/pi.txt",
        )
    )

    command = str(captured["command"])
    env = captured["env"]
    # The launcher's config path moved; the prompt argument rides the env var.
    assert "PI_CODING_AGENT_DIR=/logs/agent/pi-agent pi --print --mode json --model m" in command
    (env_var,) = [key for key in env if key.startswith("PI_INSTRUCTION_")]
    assert env[env_var] == instruction
    assert f'"{env_var.lower()}" 2>&1 </dev/null' in command


def test_local_host_substitutions_stay_off_the_task_text(monkeypatch):
    from harbor.agents.installed.pi import Pi

    agent = object.__new__(SkillEvaluatorLocalPi)
    agent.mcp_servers = []
    instruction = "Explain the literal text . ~/.nvm/nvm.sh; and | stdbuf -oL tee "
    agent._instruction = instruction
    agent._quoted_instruction = shlex.quote(instruction)
    captured = {}

    async def raw_exec(_self, _environment, command, env=None, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(Pi, "exec_as_agent", raw_exec)

    asyncio.run(
        agent.exec_as_agent(
            object(),
            ". ~/.nvm/nvm.sh; pi --print --mode json --model m "
            + shlex.quote(instruction)
            + " 2>&1 </dev/null | stdbuf -oL tee /logs/agent/pi.txt",
        )
    )

    command = str(captured["command"])
    env = captured["env"]
    # The task text quotes the host preamble and pipeline and stays byte-identical
    # in the env var; only the launcher's own copies are rewritten for the host.
    (env_var,) = [key for key in env if key.startswith("PI_INSTRUCTION_")]
    assert env[env_var] == instruction
    assert instruction not in command
    assert command.startswith("pi_instruction_")
    assert 'pi --print --mode json --model m "' in command
    assert command.endswith("| tee /logs/agent/pi.txt")


def test_local_task_text_survives_the_environments_final_path_rewrite(monkeypatch, tmp_path):
    """The local environment rewrites container paths in the assembled command after
    the wrapper ran; the task text travels in an env var that stage keeps verbatim."""
    from harbor.agents.installed.pi import Pi
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment

    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    environment = SkillEvaluatorLocalEnvironment(
        environment_dir=environment_dir,
        environment_name="pi-prompt",
        session_id="pi-prompt",
        trial_paths=paths,
        task_env_config=EnvironmentConfig(),
        runtime_agent="codex",
        runtime_root=str(runtime_root),
        sandbox_mode="off",
    )

    agent = object.__new__(SkillEvaluatorLocalPi)
    agent.mcp_servers = []
    instruction = "Quote the literal paths /workspace/input and /logs/agent/pi.txt\nin the answer."
    agent._instruction = instruction
    agent._quoted_instruction = shlex.quote(instruction)
    captured = {}

    async def raw_exec(_self, _environment, command, env=None, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(Pi, "exec_as_agent", raw_exec)

    asyncio.run(
        agent.exec_as_agent(
            object(),
            "PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent pi --print "
            + shlex.quote(instruction)
            + " 2>&1 </dev/null | stdbuf -oL tee /logs/agent/pi.txt",
        )
    )

    # The final local stage: container paths in the command map to host paths;
    # env values rewrite the same way except instruction vars, which stay verbatim.
    command = environment._rewrite_command(str(captured["command"]))
    env = environment._rewrite_env_values(environment._filter_command_env(captured["env"], protected=set()))

    (env_var,) = [key for key in env if key.startswith("PI_INSTRUCTION_")]
    assert env[env_var] == instruction  # the multiline task text is byte-identical
    assert instruction not in command and "/workspace" not in command
    assert str(paths.agent_dir) in command  # the launcher's own log path did map to the host
    assert subprocess.run(["sh", "-n", "-c", command], check=False).returncode == 0


def test_built_in_cloud_modes_route_pi_through_the_pinned_wrapper(tmp_path):
    """e2b, daytona and the other built-in cloud modes reach the stock-agent branch;
    pi still needs the pinned wrapper there (stock pi-mcp-adapter, model setup)."""
    from skillevaluator.tier3.harbor import CONTAINER_AGENT_IMPORT_PATHS

    for env_mode in ("e2b", "daytona", "modal"):
        command = build_harbor_run_command(
            dataset_path=tmp_path, agent="pi", job_name="j", env_mode=env_mode, model="openai/gpt-5.5"
        )
        assert command[command.index("--agent") + 1] == CONTAINER_AGENT_IMPORT_PATHS["pi"], env_mode

    # Agents whose stock Harbor class works keep the stock route.
    command = build_harbor_run_command(
        dataset_path=tmp_path, agent="claude-code", job_name="j", env_mode="e2b", model="claude-sonnet-5"
    )
    assert command[command.index("--agent") + 1] == "claude-code"


def test_local_exec_rewrites_container_assumptions(monkeypatch):
    from harbor.agents.installed.pi import Pi

    agent = object.__new__(SkillEvaluatorLocalPi)
    captured = {}

    async def raw_exec(_self, _environment, command, env=None, **_kwargs):
        captured["command"] = command
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(Pi, "exec_as_agent", raw_exec)

    asyncio.run(
        agent.exec_as_agent(
            object(),
            ". ~/.nvm/nvm.sh; PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent pi --print "
            "--extension /tmp/harbor-pi-mcp-adapter/node_modules/pi-mcp-adapter/index.ts "
            "--mode json 2>&1 </dev/null | grep -v '\"type\":\"message_update\"' "
            "| stdbuf -oL tee /logs/agent/pi.txt",
        )
    )

    command = str(captured["command"])
    assert ". ~/.nvm/nvm.sh" not in command
    assert "/logs/agent/pi-mcp-adapter/node_modules/pi-mcp-adapter/index.ts" in command
    assert "/tmp/harbor-pi-mcp-adapter" not in command  # host CLI resolved from PATH
    assert "PI_CODING_AGENT_DIR=/logs/agent/pi-agent" in command  # inside the sandbox roots
    assert "/tmp/harbor-pi-agent" not in command
    assert "stdbuf" not in command  # hosts may lack GNU coreutils
    assert "grep -v" in command and "tee /logs/agent/pi.txt" in command


def test_nvidia_build_rejects_pi_in_every_mode():
    from skillevaluator.tier3.harbor.runner import _validate_agent_provider_credentials

    provider = ProviderConfig(
        provider="nv_build",
        model="openai/gpt-oss-120b",
        api_key="nvidia-build-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/openai/gpt-oss-120b",
    )

    for env_mode in ("docker", "local"):
        errors = _validate_agent_provider_credentials(provider, ["pi"], {}, env_mode=env_mode)
        assert errors and "nv_build does not support live agent(s): pi" in errors[0], env_mode


def test_custom_endpoints_wire_the_model_api_agent_option(tmp_path):
    from skillevaluator.tier3.harbor.runner import _pi_custom_endpoint_agent_kwargs

    # Without a base URL pi runs on the provider defaults and needs no model_api.
    assert _pi_custom_endpoint_agent_kwargs({}) == {}
    assert _pi_custom_endpoint_agent_kwargs({"OPENAI_BASE_URL": ""}) == {}
    overrides = _pi_custom_endpoint_agent_kwargs({"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://gw.example/v1"})
    assert overrides == {"pi": {"model_api": "openai-completions"}}
    # The OpenAI provider exports its default endpoint as a base URL too; the
    # public endpoint stays on pi's built-in provider (no models.json entry).
    assert _pi_custom_endpoint_agent_kwargs({"OPENAI_BASE_URL": "https://api.openai.com/v1"}) == {}
    command = build_harbor_run_command(
        dataset_path=tmp_path,
        agent="pi",
        job_name="j",
        env_mode="docker",
        model=_provider().model,
        agent_kwargs_overrides=overrides,
    )
    assert 'model_api="openai-completions"' in " ".join(command)


def test_public_openai_endpoint_keeps_pi_on_the_builtin_provider():
    """The wrapper hides the public OpenAI base URL from pi's own resolution.

    Harbor's process env keeps OPENAI_BASE_URL (the staged [verifier.env]
    ${VAR} placeholders resolve against it), so the stock resolution would
    build a models.json custom endpoint that needs the model_api agent
    argument; pi's built-in openai provider already targets that endpoint.
    """

    def resolve_env(table):
        def _resolve(*names):
            for name in names:
                if name in table:
                    return name, table[name]
            return None

        return _resolve

    public = {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://api.openai.com/v1"}
    connection = SkillEvaluatorPi.resolve_model_connection_config("openai/gpt-5.5", resolve_env(public))
    assert connection.user_supplied_base_url is None
    assert "OPENAI_BASE_URL" not in connection.env
    assert connection.api_key == "k"  # the key still reaches the launch env

    # Real gateways keep the custom-endpoint path (models.json + model_api).
    gateway = {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://gw.example/v1"}
    connection = SkillEvaluatorPi.resolve_model_connection_config("openai/glm-4.7", resolve_env(gateway))
    assert connection.user_supplied_base_url == "https://gw.example/v1"
    assert connection.candidate_base_url == "https://gw.example/v1"


def test_run_harbor_env_resolves_the_staged_verifier_placeholders(monkeypatch, tmp_path):
    """Every ${VAR} in the staged [verifier.env] resolves against the Harbor process env.

    The pi arm used to strip OPENAI_BASE_URL from the whole Harbor env; the
    wrapper now hides the public endpoint from pi alone, so the verifier's
    placeholder resolution must keep working for the default OpenAI provider.
    """
    from skillevaluator.tier3.harbor import runner

    captured: dict[str, object] = {}

    def fake_process(_command, *, env, **_kwargs):
        captured["env"] = dict(env)
        return runner._BoundedHarborProcessResult(returncode=1, output_tail="", output_exceeded=False)

    monkeypatch.setattr(runner, "_run_bounded_harbor_process", fake_process)

    run_env = {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://api.openai.com/v1"}
    staged_verifier_env = {name: f"${{{name}}}" for name in run_env}
    ok, _detail = runner._run_harbor(
        dataset=tmp_path / "dataset",
        agent="pi",
        job_name="j",
        env_mode="docker",
        model="openai/gpt-5.5",
        jobs_dir=tmp_path,
        run_env=run_env,
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        verifier_env=staged_verifier_env,
    )
    assert ok is False  # the stubbed process exited 1; only the env matters here
    harbor_env = captured["env"]
    for placeholder in staged_verifier_env.values():
        name = placeholder[2:-1]
        assert harbor_env.get(name) == run_env[name], name


def test_runtime_preflight_for_pi_with_the_default_openai_base_url(monkeypatch, tmp_path):
    """Preflight hands Harbor the provider env and asks for no model_api on the public endpoint."""
    from skillevaluator.tier3.harbor import runtime_preflight

    task_dir = tmp_path / "dataset" / "task-1"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('name = "task-1"\n', encoding="utf-8")

    captured: dict[str, object] = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime_preflight.subprocess, "run", run)
    monkeypatch.setattr(
        runtime_preflight, "validate_harbor_agent_only_job_result", lambda *_args, **_kwargs: (True, "ok")
    )

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=task_dir.parent,
        agent="pi",
        model="openai/gpt-5.5",
        env_mode="docker",
        jobs_dir=tmp_path / "jobs",
        run_env={"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
    )

    assert result.ok is True
    command = captured["command"]
    assert isinstance(command, list)
    # No custom endpoint on the public endpoint: Harbor's Pi must not demand model_api.
    assert "model_api" not in " ".join(command)
    assert command[command.index("--agent") + 1] == DOCKER_AGENT_IMPORT_PATHS["pi"]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("OPENAI_BASE_URL") == "https://api.openai.com/v1"


def test_custom_backends_route_pi_through_the_pinned_wrapper(monkeypatch, tmp_path):
    """e2b/daytona run stock Harbor agents for the ones they do not wrap, but pi needs the
    pinned wrapper: Harbor 0.25.0's stock pi-mcp-adapter rejects pi >= 0.86 before any setup."""
    from skillevaluator.tier3.harbor import CONTAINER_AGENT_IMPORT_PATHS, runner

    spec = runner.EnvironmentBackendSpec(
        import_path="skillevaluator.tier3.harbor.local_environment:SkillEvaluatorLocalEnvironment",
        agent_import_paths={"claude-code": "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorClaudeCode"},
    )
    monkeypatch.setattr(runner, "is_env_mode", lambda env_mode: env_mode == "acme")
    monkeypatch.setattr(runner, "_custom_backend", lambda env_mode: spec if env_mode == "acme" else None)

    command = build_harbor_run_command(
        dataset_path=tmp_path, agent="pi", job_name="j", env_mode="acme", model="openai/gpt-5.5"
    )

    assert command[command.index("--agent") + 1] == CONTAINER_AGENT_IMPORT_PATHS["pi"]
    # An agent the backend wraps itself keeps the backend's route.
    command = build_harbor_run_command(
        dataset_path=tmp_path, agent="claude-code", job_name="j", env_mode="acme", model="claude-sonnet-5"
    )
    assert command[command.index("--agent") + 1] == spec.agent_import_paths["claude-code"]


def test_anthropic_default_model_is_qualified_for_pi():
    from skillevaluator.tier3.harbor.runner import _model_for_agent

    selected, source = _model_for_agent(
        "pi", cli_model=None, config_agents={}, provider=_provider("anthropic", base_url=None)
    )
    assert (selected, source) == (f"anthropic/{_provider('anthropic', base_url=None).model}", "public provider default")
    # An operator override stays exactly as written.
    selected, _ = _model_for_agent(
        "pi", cli_model="claude-opus-5-5", config_agents={}, provider=_provider("anthropic", base_url=None)
    )
    assert selected == "claude-opus-5-5"


def test_gke_routes_pi_through_the_pinned_wrapper(tmp_path):
    command = build_harbor_run_command(
        dataset_path=tmp_path, agent="pi", job_name="j", env_mode="gke", model=_provider().model
    )
    assert command[command.index("--agent") + 1] == "skillevaluator.tier3.harbor.pi_agent:SkillEvaluatorPi"


def test_native_and_private_pi_are_registered_for_plugin_arms():
    from skillevaluator.tier3.plugin_native import PRIVATE_AGENT_IMPORT_PATHS, native_agent_import_path

    base = "skillevaluator.tier3.harbor.pi_agent:SkillEvaluatorPi"
    assert native_agent_import_path("pi", base) == "skillevaluator.tier3.harbor.native_agents:NativePi"
    assert PRIVATE_AGENT_IMPORT_PATHS[("pi", base)].endswith("PrivatePi")
    assert ("pi", None) not in PRIVATE_AGENT_IMPORT_PATHS


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("os").environ.get("SKILL_EVAL_PI_INTEGRATION") != "1",
    reason="Opt in with SKILL_EVAL_PI_INTEGRATION=1; installs pi and makes paid provider calls",
)
def test_real_pi_docker_matched_arms(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    provider = os.environ.get("SKILL_EVAL_PI_PROVIDER", "openai")
    api_key = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
    }.get(provider, f"{provider.upper()}_API_KEY")
    if not os.environ.get(api_key):
        pytest.skip(f"{api_key} is required")

    skill = Path(__file__).parent / "fixtures" / "skills" / "lookup-code"
    completed = subprocess.run(
        [
            str(Path(sys.executable).with_name("skillevaluator")),
            "tier3",
            "evaluate",
            str(skill),
            "--agents",
            "pi",
            "--env-mode",
            "docker",
            "--n-attempts",
            "1",
            "--n-concurrent",
            "2",
            "--max-agents",
            "1",
            "--no-stop-on-pass",
            "--harbor-keep-jobs",
            "--results-dir",
            str(tmp_path / "results"),
        ],
        env={**os.environ, "SKILL_EVAL_LLM_PROVIDER": provider},
        capture_output=True,
        timeout=3600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]

    results = tmp_path / "results"
    trials = list(results.glob("**/_harbor-jobs/*/*/result.json"))
    assert len(trials) == 4
    for result_path in trials:
        result = json.loads(result_path.read_text())
        assert result.get("exception_info") is None
        agent_dir = result_path.parent / "agent"
        assert (agent_dir / "pi.txt").is_file()
        assert (agent_dir / "trajectory.json").is_file()
