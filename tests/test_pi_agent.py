# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi Coding Agent compatibility at the public routing and Harbor boundaries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.commands import validate_agents
from skillevaluator.tier3.harbor import DOCKER_AGENT_IMPORT_PATHS, HARBOR_AGENTS_EXPERIMENTAL
from skillevaluator.tier3.harbor.local_runtime import PINNED_RUNTIME_VERSIONS
from skillevaluator.tier3.harbor.pi_agent import SkillEvaluatorPi
from skillevaluator.tier3.harbor.runner import (
    PI_VERSION,
    _agent_import_path,
    _harbor_agent_kwargs,
    _validate_agent_provider_credentials,
    build_harbor_run_command,
)


def _provider(name="openai"):
    return ProviderConfig(
        provider=name,
        model="glm-4.7",
        api_key="selected-provider-key",
        base_url="https://api.openai-compatible.example/v1",
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
    assert _validate_agent_provider_credentials(_provider(provider), ["pi"], {}) == []


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


def test_install_rejects_mcp_servers_on_the_pinned_version(tmp_path):
    from harbor.models.task.config import MCPServerConfig

    server = MCPServerConfig(name="context7", transport="stdio", command="npx", args=["-y", "@upstash/context7-mcp"])
    with pytest.raises(ValueError, match="pi-mcp-adapter"):
        asyncio.run(_agent(tmp_path, mcp_servers=[server]).install(SimpleNamespace()))

