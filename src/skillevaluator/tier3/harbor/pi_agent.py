# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned pi Coding Agent execution for the Docker arm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor.agents.installed.pi import Pi

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment


class SkillEvaluatorPi(Pi):
    """Pin the pi CLI Harbor installs so Docker trials stay reproducible.

    Harbor's stock agent would install @latest; the pin keeps scores
    comparable across releases. Harbor's own install() also refuses MCP
    servers on this version range (pi-mcp-adapter supports pi 0.85.x only).
    """

    async def install(self, environment: BaseEnvironment) -> None:
        from skillevaluator.tier3.harbor.local_runtime import PINNED_RUNTIME_VERSIONS

        pinned = PINNED_RUNTIME_VERSIONS["pi"]
        if self._version not in {None, pinned}:
            raise ValueError(f"SkillEvaluator pi requires {pinned}")
        self._version = pinned
        await super().install(environment)

