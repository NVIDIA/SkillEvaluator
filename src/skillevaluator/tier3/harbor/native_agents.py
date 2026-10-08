# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor agent wrappers for native plugin loading (with-plugin arm only).

Each wrapper keeps the Harbor agent (and any SkillEvaluator routing wrapper it
extends) unchanged, except for its launch command:

* ``/bin/sh /skilleval/native/setup.sh`` runs first, after Harbor's own agent
  setup and in the same environment (``CLAUDE_CONFIG_DIR``, ``CODEX_HOME``,
  ``HERMES_HOME``). It stages per-run config and writes the load census to
  ``/logs/agent/skilleval-load-census.json``. It never fails the launch; a
  failed step is reported in the census instead.
* Claude Code gets the documented ``--plugin-dir`` flag.
* OpenCode gets ``OPENCODE_CONFIG`` and ``OPENCODE_CONFIG_DIR``.

No wrapper adds a permission-bypass flag; the launch flags are Harbor's own.

Only Harbor's launch command is rewritten. It is recognized on its launcher
line (the command up to the `` -- `` prompt separator, or the whole command
when the agent reads its prompt from an env var; it must be a single line) at a
shell-command boundary. Harbor's setup commands carry task and
plugin text (MCP JSON, YAML heredocs, skill paths) after their first line, so
that text can never take the launch's place. Every launch pattern is linear
(no nested or overlapping quantifiers). A run whose launch never appears, or
that shows a second launch, fails instead of silently running without the
plugin.
"""

from __future__ import annotations

import logging
import re
import shlex
from typing import TYPE_CHECKING, Any, ClassVar

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.hermes import Hermes
from harbor.agents.installed.opencode import OpenCode

from skillevaluator.tier3.harbor.local_agents import (
    SkillEvaluatorCodex,
    SkillEvaluatorGatewayCodex,
    SkillEvaluatorGatewayOpenCode,
    SkillEvaluatorNvidiaBuildClaudeCode,
    SkillEvaluatorNvidiaBuildCodex,
)
from skillevaluator.tier3.plugin_native import SETUP_SCRIPT, ClaudeCodeAdapter, OpenCodeAdapter

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

#: A shell-command boundary: the start of the line or a ``;``, ``&``, or ``|`` separator.
_COMMAND_START = r"(?:^|[;&|])[ \t]*"


class NativeLaunchError(RuntimeError):
    """Native plugin loading did not see exactly one Harbor launch command in an agent run."""


class _NativePluginLoadMixin:
    """Run the native setup script right before the agent's launch command."""

    #: Harbor's launch, matched on the single launcher line only (see the module docstring).
    _SKILLEVAL_LAUNCH_RE: ClassVar[re.Pattern[str]] = re.compile(r"(?!)")
    #: Harbor passes the prompt after `` -- `` (Codex, OpenCode); Claude Code and Hermes read it from an env var.
    _SKILLEVAL_PROMPT_SEPARATOR: ClassVar[bool] = True

    def _skilleval_rewrite_launcher(self, launcher: str, match: re.Match[str]) -> str:  # noqa: ARG002
        return launcher

    def _skilleval_launch_env(self) -> dict[str, str]:
        return {}

    def _skilleval_launch_shape(self, launcher: str) -> bool:  # noqa: ARG002
        """Extra checks on the launcher line beyond the launch pattern (Harbor's launch shape)."""
        return True

    def _skilleval_launch_match(self, command: str) -> tuple[str, str, str, re.Match[str]] | None:
        """Split Harbor's launch command, or return ``None`` for any other command."""
        launcher, separator, prompt = command.partition(" -- ")
        if self._SKILLEVAL_PROMPT_SEPARATOR and not separator:
            return None
        # Harbor's launcher is one line; a heredoc or multi-line setup command
        # carries task and plugin text after its first line and is never the launch.
        if "\n" in launcher or "\r" in launcher:
            return None
        match = self._SKILLEVAL_LAUNCH_RE.search(launcher)
        if match is None or not self._skilleval_launch_shape(launcher):
            return None
        return launcher, separator, prompt, match

    def skilleval_native_command(self, command: str, env: dict[str, str] | None) -> tuple[str, dict[str, str] | None]:
        """Return the launch command and env with native setup applied (once per agent run).

        Raises :class:`NativeLaunchError` when a second launch-shaped command
        arrives in the same run: the first one then was not the real launch,
        and the real one would run without the plugin.
        """
        found = self._skilleval_launch_match(command)
        if found is None:
            return command, env
        if getattr(self, "_skilleval_native_started", False):
            previous = getattr(self, "_skilleval_native_launch", "")
            logger.error("Native plugin setup already ran for an earlier launch-shaped command: %.200s", previous)
            raise NativeLaunchError(
                "native plugin loading saw a second launch command in one agent run; the plugin setup already "
                "ran for an earlier command, so this launch would run without the plugin"
            )
        launcher, separator, prompt, match = found
        self._skilleval_native_started = True
        self._skilleval_native_launch = launcher
        launcher = self._skilleval_rewrite_launcher(launcher, match)
        prefix = f"/bin/sh {shlex.quote(SETUP_SCRIPT)} </dev/null >/dev/null 2>&1 || true; "
        launch_env = self._skilleval_launch_env()
        return prefix + launcher + separator + prompt, ({**(env or {}), **launch_env} if launch_env else env)

    async def run(self, instruction: str, environment: BaseEnvironment, context: Any) -> None:
        """Run Harbor's agent; fail when its launch command never got the native setup."""
        self._skilleval_native_started = False
        self._skilleval_native_launch = ""
        await super().run(instruction=instruction, environment=environment, context=context)  # type: ignore[misc]
        if not self._skilleval_native_started:
            raise NativeLaunchError(
                "native plugin loading did not find the Harbor launch command for this agent, so the "
                "with-plugin arm ran without the plugin; the Harbor agent's launch command may have changed"
            )

    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        command, env = self.skilleval_native_command(command, env)
        return await super().exec_as_agent(  # type: ignore[misc]
            environment,
            command=command,
            env=env,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )


class _NativeClaudeCodeMixin(_NativePluginLoadMixin):
    # Harbor pipes the prompt in from an env var:
    # ``...; printf "%s" "$<var>" | claude --verbose --output-format=stream-json ... --print 2>&1 | tee ...``.
    # (Harbor 0.13 passed it after ``--print -- <prompt>``; that shape still matches.)
    _SKILLEVAL_LAUNCH_RE = re.compile(_COMMAND_START + r"(?P<cli>claude)[ \t]+--verbose\b")
    _SKILLEVAL_PROMPT_SEPARATOR = False
    _PRINT_FLAG_RE: ClassVar[re.Pattern[str]] = re.compile(r"[ \t]--print(?:[ \t]|$)")

    def _skilleval_launch_shape(self, launcher: str) -> bool:
        match = self._SKILLEVAL_LAUNCH_RE.search(launcher)
        if match is None:
            return False
        rest = launcher[match.end() :]
        return "--output-format=stream-json" in rest and self._PRINT_FLAG_RE.search(rest) is not None

    def _skilleval_rewrite_launcher(self, launcher: str, match: re.Match[str]) -> str:
        flag = f" --plugin-dir {shlex.quote(ClaudeCodeAdapter.plugin_dir)}"
        return launcher[: match.end("cli")] + flag + launcher[match.end("cli") :]


class _NativeCodexMixin(_NativePluginLoadMixin):
    # Harbor: ``...; codex exec --dangerously-bypass-approvals-and-sandbox ... -- <prompt>``.
    _SKILLEVAL_LAUNCH_RE = re.compile(_COMMAND_START + r"codex[ \t]+exec\b")


class _NativeOpenCodeMixin(_NativePluginLoadMixin):
    # Harbor: ``. ~/.nvm/nvm.sh; opencode --model=<model> run --format=json ... -- <prompt>``.
    _SKILLEVAL_LAUNCH_RE = re.compile(_COMMAND_START + r"opencode[ \t]+--model=[^ \t]+[ \t]+run\b")

    def _skilleval_launch_env(self) -> dict[str, str]:
        return OpenCodeAdapter().launch_env()


class _NativeHermesMixin(_NativePluginLoadMixin):
    # Harbor: ``export PATH=... && hermes --yolo chat -q "$HARBOR_INSTRUCTION" ...`` on one line.
    # Each flag token matches one way only (``-`` or ``--``, then a word character),
    # so a long run of dashes cannot backtrack exponentially (CodeQL py/redos).
    _SKILLEVAL_LAUNCH_RE = re.compile(_COMMAND_START + r"hermes(?:[ \t]+-{1,2}\w[\w-]*)*[ \t]+chat\b")
    _SKILLEVAL_PROMPT_SEPARATOR = False


class NativeClaudeCode(_NativeClaudeCodeMixin, ClaudeCode):
    """Claude Code with the plugin loaded through ``--plugin-dir``."""


class NativeNvidiaBuildClaudeCode(_NativeClaudeCodeMixin, SkillEvaluatorNvidiaBuildClaudeCode):
    """NVIDIA Build-routed Claude Code with the plugin loaded through ``--plugin-dir``."""


class NativeCodex(_NativeCodexMixin, SkillEvaluatorCodex):
    """Codex with plugin rules and MCP servers staged into ``$CODEX_HOME``."""


class NativeGatewayCodex(_NativeCodexMixin, SkillEvaluatorGatewayCodex):
    """Gateway-routed Codex with plugin rules and MCP servers staged into ``$CODEX_HOME``."""


class NativeNvidiaBuildCodex(_NativeCodexMixin, SkillEvaluatorNvidiaBuildCodex):
    """NVIDIA Build-routed Codex with plugin rules and MCP servers staged into ``$CODEX_HOME``."""


class NativeOpenCode(_NativeOpenCodeMixin, OpenCode):
    """OpenCode with ``OPENCODE_CONFIG`` and ``OPENCODE_CONFIG_DIR`` pointing at the staged plugin."""


class NativeGatewayOpenCode(_NativeOpenCodeMixin, SkillEvaluatorGatewayOpenCode):
    """Gateway-routed OpenCode with the staged plugin config."""


class NativeHermes(_NativeHermesMixin, Hermes):
    """Hermes with the load census written before launch."""
