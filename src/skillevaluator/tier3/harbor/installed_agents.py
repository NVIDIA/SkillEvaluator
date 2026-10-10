# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Harbor's bootstrap dependencies aligned with its installer branches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment


async def _installer_dependencies(
    environment: BaseEnvironment,
    dependencies: tuple[str, ...],
    *,
    system_npm_predicate: str,
) -> tuple[str, ...]:
    if not {"nodejs", "npm"}.intersection(dependencies):
        return dependencies

    # Match the pinned Harbor installer's branch before removing prerequisites.
    # Codex supplies Node via NVM on glibc; Claude's native installer needs no
    # Node. Both still need system npm on their respective fallback branches.
    result = await environment.exec(
        command=f"if {system_npm_predicate}; then printf '%s' system; else printf '%s' bundled; fi",
        user="root",
    )
    installer = (result.stdout or "").strip()
    if result.return_code != 0 or installer not in {"system", "bundled"}:
        raise RuntimeError("Could not determine agent installer system dependency requirements")
    if installer == "system":
        return dependencies
    return tuple(dependency for dependency in dependencies if dependency not in {"nodejs", "npm"})


class SkillEvaluatorCodex(Codex):
    """Avoid a redundant distribution Node/npm install before Harbor's NVM."""

    async def ensure_system_dependencies(
        self,
        environment: BaseEnvironment,
        dependencies: tuple[str, ...],
    ) -> None:
        dependencies = await _installer_dependencies(
            environment,
            dependencies,
            system_npm_predicate="ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]",
        )
        await super().ensure_system_dependencies(environment, dependencies)


class SkillEvaluatorClaudeCode(ClaudeCode):
    """Avoid distribution Node/npm for Harbor's native Claude installer."""

    async def ensure_system_dependencies(
        self,
        environment: BaseEnvironment,
        dependencies: tuple[str, ...],
    ) -> None:
        dependencies = await _installer_dependencies(
            environment,
            dependencies,
            system_npm_predicate="command -v apk >/dev/null 2>&1",
        )
        await super().ensure_system_dependencies(environment, dependencies)
