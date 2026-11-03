# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Harbor's bootstrap dependencies aligned with its installer branches.

The wrappers also retry a failed agent CLI install, since a dropped download or
a package mirror error otherwise fails the trial before the agent starts.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex

if TYPE_CHECKING:
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.environments.base import BaseEnvironment

# Harbor's installers download over HTTPS but never request a CA bundle; apt
# images used to get one only as a dependency of the skipped npm package.
_CA_BUNDLE_PREDICATE = "[ -s /etc/ssl/certs/ca-certificates.crt ] || [ -s /etc/pki/tls/certs/ca-bundle.crt ]"
_INSTALLER_BRANCHES = frozenset({"system", "bundled", "bundled-without-ca"})
# Harbor runs each installer once. Its install() skips a CLI that an earlier
# attempt already installed, so rerunning it after a failure is safe.
_INSTALL_ATTEMPTS = 3
_INSTALL_RETRY_BASE_DELAY_SEC = 2.0
# Harbor's default agent setup timeout covers every attempt. An attempt that
# already used a third of it leaves too little budget to retry without turning
# the original error into AgentSetupTimeoutError.
_SETUP_TIMEOUT_SEC = 360.0
_MAX_RETRYABLE_ATTEMPT_SEC = _SETUP_TIMEOUT_SEC / 3
_FAILURE_DETAIL_CHARS = 200


async def _install_with_retries(
    agent: BaseInstalledAgent,
    install: Callable[[BaseEnvironment], Awaitable[None]],
    environment: BaseEnvironment,
) -> None:
    # Retry only a Harbor command that exited non-zero. Probe, transport and
    # cancellation errors propagate, and all attempts share Harbor's agent
    # setup timeout.
    for attempt in range(1, _INSTALL_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            await install(environment)
            return
        except NonZeroAgentExitCodeError as exc:
            elapsed = time.monotonic() - started
            if attempt == _INSTALL_ATTEMPTS or elapsed > _MAX_RETRYABLE_ATTEMPT_SEC:
                raise
            delay = _INSTALL_RETRY_BASE_DELAY_SEC * attempt
            agent.logger.warning(
                "%s install attempt %d of %d failed with %s (%s); retrying in %.0fs",
                agent.name(),
                attempt,
                _INSTALL_ATTEMPTS,
                type(exc).__name__,
                str(exc)[-_FAILURE_DETAIL_CHARS:],
                delay,
            )
            await asyncio.sleep(delay)


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
        command=(
            f"if {system_npm_predicate}; then printf '%s' system; "
            f"elif {_CA_BUNDLE_PREDICATE}; then printf '%s' bundled; "
            "else printf '%s' bundled-without-ca; fi"
        ),
        user="root",
    )
    installer = (result.stdout or "").strip()
    if result.return_code != 0 or installer not in _INSTALLER_BRANCHES:
        raise RuntimeError("Could not determine agent installer system dependency requirements")
    if installer == "system":
        return dependencies
    bundled = tuple(dependency for dependency in dependencies if dependency not in {"nodejs", "npm"})
    if installer == "bundled-without-ca":
        return (*bundled, "ca_certificates")
    return bundled


class SkillEvaluatorCodex(Codex):
    """Avoid a redundant distribution Node/npm install before Harbor's NVM."""

    async def install(self, environment: BaseEnvironment) -> None:
        await _install_with_retries(self, super().install, environment)

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

    async def install(self, environment: BaseEnvironment) -> None:
        await _install_with_retries(self, super().install, environment)

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
