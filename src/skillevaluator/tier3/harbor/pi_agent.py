# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned pi Coding Agent execution for the Docker arm."""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

from harbor.agents.installed.pi import (
    _PI_CONFIG_DIR_ENV,
    _REMOTE_PI_CONFIG_DIR,
    _REMOTE_PI_MCP_ADAPTER_EXTENSION,
    Pi,
)

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

_PI_MCP_SHIM = "export default function () {}\n"
_PI_MCP_SHIM_FILENAME = "index.ts"


class SkillEvaluatorPi(Pi):
    # pi resolves staged mcp.json headers from its own environment; the header
    # variables are forwarded to the launch command by exec_as_agent below.
    capabilities = Pi.capabilities.model_copy(update={"mcp_http_headers": True})

    """Pin the pi CLI Harbor installs so Docker trials stay reproducible.

    Harbor's stock agent would install @latest; the pin keeps scores
    comparable across releases. pi 1.6.0 loads MCP servers natively from the
    config-dir mcp.json, so Harbor 0.25.0's pi-mcp-adapter path (which targets
    pi 0.85.x) is skipped: the adapter install is dropped, a no-op extension
    is staged at the path Harbor's launch line references, and the staged
    config is verified with ``pi mcp list`` before the trial launches.
    """

    async def install(self, environment: BaseEnvironment) -> None:
        from skillevaluator.tier3.harbor.local_runtime import PINNED_RUNTIME_VERSIONS

        pinned = PINNED_RUNTIME_VERSIONS["pi"]
        if self._version not in {None, pinned}:
            raise ValueError(f"SkillEvaluator pi requires {pinned}")
        self._version = pinned

        servers = self.mcp_servers
        if not servers:
            await super().install(environment)
            return
        # Harbor's stock install gates the 0.85.x-only pi-mcp-adapter; pi 1.6.0
        # needs no adapter, so stage the config dir directly instead.
        self.mcp_servers = []
        try:
            await super().install(environment)
        finally:
            self.mcp_servers = servers
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {shlex.quote(_REMOTE_PI_MCP_ADAPTER_EXTENSION.parent.as_posix())}",
        )
        await self._upload_config_text(
            environment,
            content=_PI_MCP_SHIM,
            remote_path=_REMOTE_PI_MCP_ADAPTER_EXTENSION.as_posix(),
            filename=_PI_MCP_SHIM_FILENAME,
        )

    def _build_mcp_config(self) -> dict[str, Any]:
        """Serialize servers with their HTTP headers kept as env references.

        Harbor's stock serializer drops headers; pi resolves them from its own
        environment, and the referenced variables are forwarded to the launch
        command by ``exec_as_agent`` below.
        """
        config = super()._build_mcp_config()
        header_refs = self.mcp_header_env.refs
        for server in self.mcp_servers:
            refs = header_refs.get(server.name)
            if refs:
                config["mcpServers"][server.name]["headers"] = {
                    name: "${" + variable + "}" for name, variable in refs.items()
                }
        return config

    async def _write_mcp_config(self, environment: BaseEnvironment) -> None:
        await super()._write_mcp_config(environment)
        await self._verify_mcp_config(environment)

    async def _verify_mcp_config(self, environment: BaseEnvironment) -> None:
        """Fail the trial when pi does not report the staged servers.

        A missing server name in ``pi mcp list`` output means pi never read the
        staged config-dir mcp.json, and the with-MCP arm would silently run
        without its servers. Connection failures are only logged: task servers
        may be broken on purpose.
        """
        command = (
            ". ~/.nvm/nvm.sh; "
            f"{_PI_CONFIG_DIR_ENV}={shlex.quote(_REMOTE_PI_CONFIG_DIR.as_posix())} pi mcp list --json"
        )
        try:
            result = await self.exec_as_agent(environment, command=command)
        except RuntimeError as exc:
            raise RuntimeError(f"pi mcp list failed after staging mcp.json: {exc}") from exc
        try:
            entries = json.loads(result.stdout or "").get("servers", [])
            reported = {str(entry.get("name")): str(entry.get("state")) for entry in entries}
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("pi mcp list returned output that is not a server list") from exc
        missing = sorted(str(server.name) for server in self.mcp_servers if str(server.name) not in reported)
        if missing:
            raise RuntimeError(f"pi did not report the staged MCP server(s): {', '.join(missing)}")
        unconnected = sorted(name for name, state in reported.items() if state != "connected")
        if unconnected:
            self.logger.warning(f"pi MCP server(s) not connected at startup: {', '.join(unconnected)}")

    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ):
        if env is not None and self.mcp_servers:
            # Header variables (approved env references) must reach pi so it can
            # resolve the staged mcp.json headers.
            env = {**self.mcp_header_env.env, **env}
        return await super().exec_as_agent(
            environment, command=command, env=env, cwd=cwd, timeout_sec=timeout_sec
        )

