# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned pi Coding Agent execution for the Docker arm."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from harbor.agents.installed.pi import _REMOTE_PI_MCP_ADAPTER_EXTENSION, Pi

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

_PI_MCP_SHIM = "export default function () {}\n"
_PI_MCP_SHIM_FILENAME = "index.ts"


class SkillEvaluatorPi(Pi):
    """Pin the pi CLI Harbor installs so Docker trials stay reproducible.

    Harbor's stock agent would install @latest; the pin keeps scores
    comparable across releases. Harbor 0.25.0 also wires MCP through
    pi-mcp-adapter, which supports pi 0.85.x only; pi 1.5.2 reads the
    config-dir mcp.json natively, so the adapter install is skipped and a
    no-op extension is staged at the adapter path Harbor's launch line
    references.
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
        self.mcp_servers = {}
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

