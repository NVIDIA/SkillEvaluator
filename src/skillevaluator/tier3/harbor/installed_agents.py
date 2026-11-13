# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Harbor's bootstrap dependencies aligned with its installer branches.

The Claude Code, Codex and OpenCode wrappers request distribution Node/npm only
on the installer branch that uses it. The Claude Code and Codex wrappers also
retry a failed agent CLI install, since a dropped download or a package mirror
error otherwise fails the trial before the agent starts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import TYPE_CHECKING

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.opencode import OpenCode

from skillevaluator.utils.redaction import redact_sensitive_text

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
# A slow failure (a connect that hangs, say) is not retried when it used more
# than this share of the agent setup timeout, so the trial still ends with the
# install error instead of Harbor's setup timeout.
_INSTALL_RETRY_MAX_ATTEMPT_SHARE = 1 / 3
# The retry warning keeps the end of Harbor's error, which holds the failed
# command's stderr; Harbor logs the full output only at debug level.
_INSTALL_FAILURE_DETAIL_CHARS = 200


def _install_failure_detail(exc: BaseException) -> str:
    # Redact before truncating so a credential cut at the boundary is still
    # recognized, and fold Harbor's stdout/stderr lines into one log line.
    detail = " ".join(redact_sensitive_text(str(exc)).split())
    if len(detail) > _INSTALL_FAILURE_DETAIL_CHARS:
        return "..." + detail[-_INSTALL_FAILURE_DETAIL_CHARS:]
    return detail


async def _install_with_retries(
    agent: BaseInstalledAgent,
    install: Callable[[BaseEnvironment], Awaitable[None]],
    environment: BaseEnvironment,
) -> None:
    # Retry only a Harbor command that exited non-zero. Probe, transport and
    # cancellation errors propagate, and all attempts share Harbor's agent
    # setup timeout. The agent cannot see Harbor's timeout multiplier, so the
    # budget is the agent's unscaled default; a larger multiplier only makes
    # the guard skip a retry that would have fit.
    budget_sec = agent.DEFAULT_SETUP_TIMEOUT_SEC
    started = monotonic()
    for attempt in range(1, _INSTALL_ATTEMPTS + 1):
        attempt_started = monotonic()
        try:
            await install(environment)
            return
        except NonZeroAgentExitCodeError as exc:
            if attempt == _INSTALL_ATTEMPTS:
                raise
            now = monotonic()
            attempt_sec = now - attempt_started
            delay = _INSTALL_RETRY_BASE_DELAY_SEC * attempt
            detail = _install_failure_detail(exc)
            # Assume another attempt takes as long as this one.
            if (
                attempt_sec > budget_sec * _INSTALL_RETRY_MAX_ATTEMPT_SHARE
                or now - started + delay + attempt_sec > budget_sec
            ):
                agent.logger.warning(
                    "%s install attempt %d of %d failed with %s after %.0fs; not retrying, another "
                    "attempt would not fit the %.0fs agent setup timeout: %s",
                    agent.name(),
                    attempt,
                    _INSTALL_ATTEMPTS,
                    type(exc).__name__,
                    attempt_sec,
                    budget_sec,
                    detail,
                )
                raise
            agent.logger.warning(
                "%s install attempt %d of %d failed with %s; retrying in %.0fs: %s",
                agent.name(),
                attempt,
                _INSTALL_ATTEMPTS,
                type(exc).__name__,
                delay,
                detail,
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
    # Codex and OpenCode supply Node via NVM on glibc; Claude's native installer
    # needs no Node. All three still need system npm on their fallback branches.
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


class SkillEvaluatorOpenCode(OpenCode):
    """Avoid a redundant distribution Node/npm install before Harbor's NVM."""

    async def ensure_system_dependencies(
        self,
        environment: BaseEnvironment,
        dependencies: tuple[str, ...],
    ) -> None:
        # The predicate is the one Harbor's OpenCode install uses to choose
        # between distribution Node (musl) and NVM (glibc).
        dependencies = await _installer_dependencies(
            environment,
            dependencies,
            system_npm_predicate="ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]",
        )
        await super().ensure_system_dependencies(environment, dependencies)
