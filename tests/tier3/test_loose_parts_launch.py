# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loose-parts arm's Claude Code launch: setup first, no ``--plugin-dir``, plain MCP config."""

from __future__ import annotations

import shlex

import pytest

pytest.importorskip("harbor")

from skillevaluator.tier3.harbor import loose_agents
from skillevaluator.tier3.loose_parts import LOOSE_DATA, LOOSE_MCP_CONFIG, LOOSE_ROOT
from skillevaluator.tier3.plugin_native import SETUP_SCRIPT

_LAUNCH = (
    'export PATH="$HOME/.local/bin:$PATH"; '
    'harbor_claude_code_instruction_0123abcd="$HARBOR_CLAUDE_CODE_INSTRUCTION_0123ABCD"; '
    "unset HARBOR_CLAUDE_CODE_INSTRUCTION_0123ABCD; "
    'printf "%s" "$harbor_claude_code_instruction_0123abcd" | '
    "claude --verbose --output-format=stream-json --print 2>&1 | tee /logs/agent/claude-code.txt"
)


@pytest.mark.parametrize("cls", [loose_agents.LoosePartsClaudeCode, loose_agents.LoosePartsNvidiaBuildClaudeCode])
def test_loose_launch_runs_setup_and_loads_mcp_without_a_plugin(cls) -> None:
    agent = cls.__new__(cls)
    env = {"HARBOR_CLAUDE_CODE_INSTRUCTION_0123ABCD": "Do the task."}

    command, launch_env = agent.skilleval_native_command(_LAUNCH, env)

    assert command.startswith(f"/bin/sh {shlex.quote(SETUP_SCRIPT)} ")
    assert f"claude --mcp-config {shlex.quote(LOOSE_MCP_CONFIG)} --verbose" in command
    assert "--plugin-dir" not in command
    assert command.endswith(_LAUNCH[_LAUNCH.index(" --verbose") :])
    # Scripts that read the plugin-root variables find the loose root, as plugin scripts find theirs.
    assert launch_env == {**env, "CLAUDE_PLUGIN_ROOT": LOOSE_ROOT, "CLAUDE_PLUGIN_DATA": LOOSE_DATA}


def test_loose_launch_leaves_other_commands_alone() -> None:
    agent = loose_agents.LoosePartsClaudeCode.__new__(loose_agents.LoosePartsClaudeCode)
    command = "mkdir -p $CLAUDE_CONFIG_DIR/debug && echo claude --verbose"
    assert agent.skilleval_native_command(command, None) == (command, None)
