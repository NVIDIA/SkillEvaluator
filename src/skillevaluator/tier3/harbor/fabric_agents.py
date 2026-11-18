# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in NeMo Fabric launchers for Tier 3 Harbor runs (POC).

Harbor's ``FabricAgent`` runs the Fabric runner inside the task environment.
These subclasses keep SkillEvaluator's in-trial NVIDIA Build bridge in front of
the Fabric adapter, so Claude Code and Codex can reach NVIDIA Build models.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from harbor.agents.installed.fabric import FabricAgent

from skillevaluator.tier3.harbor.local_agents import _NvidiaBuildBridgeAgent

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

FABRIC_LAUNCHER_ENV = "SKILLEVALUATOR_FABRIC_LAUNCHER"
FABRIC_TELEMETRY_ENV = "SKILLEVALUATOR_FABRIC_TELEMETRY"
FABRIC_VERSION = "0.5.0"

FABRIC_AGENT_IMPORT_PATHS = {
    "claude-code": "skillevaluator.tier3.harbor.fabric_agents:SkillEvaluatorFabricClaudeCode",
    "codex": "skillevaluator.tier3.harbor.fabric_agents:SkillEvaluatorFabricCodex",
}

# Construction-time placeholder: Fabric validates the model schema in __init__,
# but the in-trial bridge URL is only known once the bridge starts in run().
_BRIDGE_URL_PLACEHOLDER = "http://127.0.0.1:1"
_MODEL_API_KEY_ENVS = {"claude-code": "ANTHROPIC_API_KEY", "codex": "OPENAI_API_KEY"}

_ADAPTERS = {
    "claude-code": ("nvidia.fabric.claude", "claude", {"permission_mode": "bypassPermissions"}),
    "codex": ("nvidia.fabric.codex", "codex", {"sandbox": "danger-full-access", "approval_mode": "deny_all"}),
}


def fabric_launcher_enabled() -> bool:
    return os.environ.get(FABRIC_LAUNCHER_ENV, "").strip().lower() in {"1", "true", "yes"}


def fabric_agent_kwargs(agent: str) -> dict[str, Any]:
    """Harbor ``--ak`` values for the Fabric launcher of ``agent``."""

    adapter_id, extra, settings = _ADAPTERS[agent]
    telemetry = os.environ.get(FABRIC_TELEMETRY_ENV, "relay").strip() or "relay"
    extras = f"{extra},relay" if telemetry == "relay" else extra
    return {
        "fabric_adapter_id": adapter_id,
        "fabric_workspace": "/workspace",
        "fabric_package": f"nemo-fabric[{extras}]=={FABRIC_VERSION}",
        "fabric_telemetry": telemetry,
        "fabric_harness_settings": settings,
        "fabric_runtime_timeout_seconds": 600,
        "fabric_model_base_url": _BRIDGE_URL_PLACEHOLDER,
        "fabric_model_api_key_env": _MODEL_API_KEY_ENVS[agent],
        # Task containers run as root; Claude refuses bypassPermissions as root outside a sandbox.
        "fabric_environment_env": {"IS_SANDBOX": "1"},
    }


class _FabricBridgeAgent(_NvidiaBuildBridgeAgent, FabricAgent):
    """FabricAgent whose model endpoint is the in-trial NVIDIA Build bridge."""

    _MODEL_API_KEY_ENV = ""

    def _bridge_model_base_url(self) -> str:
        raise NotImplementedError

    async def _prepare_bridge_client(self, environment: BaseEnvironment) -> None:  # noqa: ARG002
        self.options.fabric_model_base_url = self._bridge_model_base_url()
        self.options.fabric_model_api_key_env = self._MODEL_API_KEY_ENV

    @property
    def _runner_env(self) -> dict[str, str]:
        env = dict(super()._runner_env)
        client_env = getattr(self, "_nvidia_build_bridge_client_env", None) or {}
        env.update(client_env)
        for name in self._BRIDGE_CLIENT_ENV_UNSET:
            env.pop(name, None)
        return env

    async def run(self, instruction: str, environment: BaseEnvironment, context: Any) -> None:
        await super().run(instruction=instruction, environment=environment, context=context)


class SkillEvaluatorFabricClaudeCode(_FabricBridgeAgent):
    """Claude Code through NeMo Fabric's Claude adapter and the NVIDIA Build bridge."""

    _MODEL_API_KEY_ENV = "ANTHROPIC_API_KEY"

    def _bridge_model_base_url(self) -> str:
        return self._bridge_origin()

    def _bridge_client_environment(self) -> dict[str, str]:
        model = str(self.model_name or "")
        env = {
            "ANTHROPIC_API_KEY": self._bridge_client_token(),
            "ANTHROPIC_BASE_URL": self._bridge_origin(),
            "IS_SANDBOX": "1",
        }
        if model:
            for name in (
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL",
            ):
                env[name] = model
        return env


class SkillEvaluatorFabricCodex(_FabricBridgeAgent):
    """Codex through NeMo Fabric's Codex adapter and the NVIDIA Build bridge."""

    _BRIDGE_CLIENT_ENV_UNSET = ("NVIDIA_API_KEY", "OPENAI_API_BASE")
    _MODEL_API_KEY_ENV = "OPENAI_API_KEY"

    def _bridge_model_base_url(self) -> str:
        return self._bridge_api_url()

    def _bridge_client_environment(self) -> dict[str, str]:
        return {
            "OPENAI_API_KEY": self._bridge_client_token(),
            "OPENAI_BASE_URL": self._bridge_api_url(),
        }


FABRIC_HOST_PYTHON_ENV = "SKILLEVALUATOR_FABRIC_HOST_PYTHON"

FABRIC_LOCAL_AGENT_IMPORT_PATHS = {
    "claude-code": "skillevaluator.tier3.harbor.fabric_agents:SkillEvaluatorLocalFabricClaudeCode",
    "codex": "skillevaluator.tier3.harbor.fabric_agents:SkillEvaluatorLocalFabricCodex",
}


def fabric_local_agent_kwargs(agent: str) -> dict[str, Any]:
    python = os.environ.get(FABRIC_HOST_PYTHON_ENV, "").strip()
    if not python:
        raise ValueError(f"{FABRIC_HOST_PYTHON_ENV} must name a host Python with nemo-fabric installed")
    kwargs = fabric_agent_kwargs(agent)
    kwargs.pop("fabric_package")
    kwargs["fabric_python"] = python
    return kwargs


class _LocalFabricEnvironment:
    """Map Fabric's fixed /tmp spec paths and container paths onto the local run root."""

    _TMP_PREFIX = "/tmp/fabric-"

    def __init__(self, environment: Any) -> None:
        self._environment = environment

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def _map(self, value: str) -> str:
        import re

        tmp = str(self._environment._tmp)
        value = value.replace(self._TMP_PREFIX, f"{tmp}/fabric-")
        return re.sub(r"(?<=\s)'?/tmp'?(?=\s|$)", tmp, value)

    async def exec(self, command: str, *args: Any, **kwargs: Any) -> Any:
        return await self._environment.exec(self._map(command), *args, **kwargs)

    async def upload_file(self, source_path: Any, target_path: str) -> None:
        from pathlib import Path

        source = Path(source_path)
        if source.name.startswith("fabric-run-") and source.suffix == ".json":
            text = self._environment._rewrite_raw_paths(source.read_text(encoding="utf-8"))
            source.write_text(text, encoding="utf-8")
        await self._environment.upload_file(source, self._map(target_path))

    async def download_file(self, source_path: str, target_path: Any) -> None:
        await self._environment.download_file(self._map(source_path), target_path)


class _LocalFabricMixin:
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(_LocalFabricEnvironment(environment))  # type: ignore[misc]

    async def run(self, instruction: str, environment: BaseEnvironment, context: Any) -> None:
        await super().run(  # type: ignore[misc]
            instruction=instruction,
            environment=_LocalFabricEnvironment(environment),
            context=context,
        )
        self._promote_local_relay_atif()

    def _promote_local_relay_atif(self) -> None:
        """Fabric's in-runner validator only accepts literal /logs/agent paths.

        Local mode rewrites those to host paths, so promote the finalized Relay
        ATIF to Harbor's canonical trajectory here.
        """
        import json
        import shutil

        logs_dir = self.logs_dir  # type: ignore[attr-defined]
        canonical = logs_dir / "trajectory.json"
        if canonical.exists():
            return
        candidates = sorted((logs_dir / "fabric-artifacts").glob("*/relay/*/trajectory-*.atif.json"))
        if len(candidates) != 1:
            return
        shutil.copyfile(candidates[0], canonical)
        summary = logs_dir / "telemetry-validation.json"
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))
            data["local_promotion"] = {"status": "promoted", "source": candidates[0].name}
            summary.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _local_bridge_base() -> type:
    from skillevaluator.tier3.harbor.local_agents import _LocalHostNvidiaBuildBridgeAgent

    return _LocalHostNvidiaBuildBridgeAgent


class SkillEvaluatorLocalFabricClaudeCode(_LocalFabricMixin, _local_bridge_base(), SkillEvaluatorFabricClaudeCode):  # type: ignore[misc]
    """Fabric Claude adapter on the confined host, using the in-process bridge."""


class SkillEvaluatorLocalFabricCodex(_LocalFabricMixin, _local_bridge_base(), SkillEvaluatorFabricCodex):  # type: ignore[misc]
    """Fabric Codex adapter on the confined host, using the in-process bridge."""
