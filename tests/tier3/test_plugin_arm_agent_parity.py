# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every arm of a plugin run builds on the agent wrapper the baseline arm runs.

The baseline, skills-only and sum-of-parts arms of Claude Code run the container wrapper on the
Docker-engine backends and on GKE (Vertex AI routing, the installer that skips the Node/npm packages
the native installer does not need, setup network failures renamed). The native plugin arm and the
loose-parts arm add their launch rewrite on top of that same wrapper, so a ladder run compares arms
that differ only in what they load.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import installed_agents, local_agents
from skillevaluator.tier3.harbor import runner as harbor_runner
from skillevaluator.tier3.loose_parts import loose_parts_agent_import_path
from skillevaluator.tier3.plugin_native import native_agent_import_path

ANTHROPIC = ProviderConfig("anthropic", "claude-sonnet-4-6", "test-key", None, "anthropic/claude-sonnet-4-6")
VERTEX_ENV = {
    "CLAUDE_CODE_USE_VERTEX": "1",
    "CLOUD_ML_REGION": "global",
    "ANTHROPIC_VERTEX_PROJECT_ID": "demo-project",
}


def _load(import_path: str) -> type:
    module_name, _, class_name = import_path.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def _baseline_agent(env_mode: str) -> str:
    """The Harbor agent the baseline arm's command runs (no routing wrapper for Anthropic)."""
    command = harbor_runner.build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="claude-code",
        job_name="baseline",
        env_mode=env_mode,
        agent_import_path=harbor_runner._agent_import_path(ANTHROPIC, "claude-code", env_mode),
    )
    return command[command.index("--agent") + 1]


def _arm_classes(env_mode: str) -> tuple[type, type, type]:
    base = harbor_runner._arm_base_import_paths(ANTHROPIC, ["claude-code"], env_mode)["claude-code"]
    native = _load(native_agent_import_path("claude-code", base))
    loose = _load(loose_parts_agent_import_path("claude-code", base))
    baseline = _baseline_agent(env_mode)
    return (_load(baseline) if ":" in baseline else type(None)), native, loose


@pytest.mark.parametrize("env_mode", ["docker", "kata"])
def test_docker_plugin_arms_build_on_the_baseline_container_wrapper(env_mode: str) -> None:
    baseline, native, loose = _arm_classes(env_mode)

    assert baseline is local_agents.SkillEvaluatorClaudeCode
    for arm in (native, loose):
        assert issubclass(arm, baseline), arm
        # The same installer (the #193 Node/npm skip), Vertex AI routing and setup-failure rename.
        assert issubclass(arm, installed_agents.SkillEvaluatorClaudeCode), arm
        assert arm.install is baseline.install
        assert arm._resolve_auth_env is baseline._resolve_auth_env
        assert arm.setup is baseline.setup is local_agents.SetupNetworkErrorAgent.setup


def test_gke_plugin_arms_build_on_the_container_wrapper() -> None:
    base = harbor_runner._arm_base_import_paths(ANTHROPIC, ["claude-code"], "gke")["claude-code"]

    assert base == "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorClaudeCode"
    for import_path in (
        native_agent_import_path("claude-code", base),
        loose_parts_agent_import_path("claude-code", base),
    ):
        assert issubclass(_load(import_path), local_agents.SkillEvaluatorClaudeCode)
        # A custom agent must be allowed on GKE, a Harbor-native backend.
        assert import_path in harbor_runner._native_backend_agent_import_paths()


def test_harbor_native_backends_keep_the_stock_agent_on_every_arm() -> None:
    """Where the baseline runs Harbor's stock Claude Code, the plugin arms add only their launch rewrite."""
    from harbor.agents.installed.claude_code import ClaudeCode

    assert _baseline_agent("daytona") == "claude-code"
    _baseline, native, loose = _arm_classes("daytona")
    for arm in (native, loose):
        assert issubclass(arm, ClaudeCode)
        assert not issubclass(arm, local_agents.SkillEvaluatorClaudeCode)


def test_nvidia_build_routing_reaches_every_arm() -> None:
    nv_build = ProviderConfig("nv_build", "m", "k", "https://example.invalid/v1", "nvidia/m")
    base = harbor_runner._arm_base_import_paths(nv_build, ["claude-code"], "docker")["claude-code"]

    assert base == "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode"
    for import_path in (
        native_agent_import_path("claude-code", base),
        loose_parts_agent_import_path("claude-code", base),
    ):
        assert issubclass(_load(import_path), local_agents.SkillEvaluatorNvidiaBuildClaudeCode)


@pytest.mark.parametrize("arm", ["native", "loose"])
def test_vertex_settings_reach_the_plugin_arms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _baseline, native, loose = _arm_classes("docker")
    agent_class = native if arm == "native" else loose
    agent = agent_class(logs_dir=tmp_path, model_name="vertex_ai/claude-sonnet-4-6", extra_env=dict(VERTEX_ENV))

    assert agent._resolve_auth_env() == VERTEX_ENV
    assert agent.model_connection.api_key is None

    captured: list[str] = []

    async def fake_parent_exec(self, environment, command, **kwargs):
        captured.append(command)

    monkeypatch.setattr("harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent", fake_parent_exec)
    asyncio.run(agent.exec_as_agent(object(), "claude --version"))
    # The direct Anthropic credentials are cleared on the plugin arm too, as on the baseline.
    assert captured[-1].startswith("unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL ")
