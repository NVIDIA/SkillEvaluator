# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor Claude Code wrappers for the loose-parts arm (see :mod:`skillevaluator.tier3.loose_parts`).

The launch is rewritten like the native plugin arm's
(:mod:`skillevaluator.tier3.harbor.native_agents`): ``/skilleval/native/setup.sh``
runs first and writes the loose parts under ``$CLAUDE_CONFIG_DIR``, and the same
launch-shape checks apply (exactly one Harbor launch per run). The differences
are what make the parts loose: no ``--plugin-dir``; the plain MCP config is passed
with ``--mcp-config``; and the launch exports ``CLAUDE_PLUGIN_ROOT`` and
``CLAUDE_PLUGIN_DATA`` pointing at the loose directories, the values a plugin's
scripts see in the plugin arm. No wrapper adds a permission-bypass flag.
"""

from __future__ import annotations

import re
import shlex

from harbor.agents.installed.claude_code import ClaudeCode

from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorClaudeCode, SkillEvaluatorNvidiaBuildClaudeCode
from skillevaluator.tier3.harbor.native_agents import _NativeClaudeCodeMixin
from skillevaluator.tier3.loose_parts import LOOSE_MCP_CONFIG, loose_parts_launch_env


class _LoosePartsClaudeCodeMixin(_NativeClaudeCodeMixin):
    def _skilleval_rewrite_launcher(self, launcher: str, match: re.Match[str]) -> str:
        flag = f" --mcp-config {shlex.quote(LOOSE_MCP_CONFIG)}"
        return launcher[: match.end("cli")] + flag + launcher[match.end("cli") :]

    def _skilleval_launch_env(self) -> dict[str, str]:
        return loose_parts_launch_env()


class LoosePartsClaudeCode(_LoosePartsClaudeCodeMixin, ClaudeCode):
    """Claude Code with the plugin's parts staged loose: user-level pieces and ``--mcp-config``."""


class LoosePartsContainerClaudeCode(_LoosePartsClaudeCodeMixin, SkillEvaluatorClaudeCode):
    """The container Claude Code wrapper the baseline arms run, with the plugin's parts staged loose."""


class LoosePartsNvidiaBuildClaudeCode(_LoosePartsClaudeCodeMixin, SkillEvaluatorNvidiaBuildClaudeCode):
    """NVIDIA Build-routed Claude Code with the plugin's parts staged loose."""
