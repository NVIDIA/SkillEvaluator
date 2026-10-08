# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Claude Code plugin loading must find the launch of the pinned Harbor.

Harbor 0.13 passed the prompt after ``claude ... --print -- <prompt>``. Since
Harbor 0.22 Claude Code reads the prompt from an environment variable and the
launch is ``printf "%s" "$<var>" | claude --verbose ... --print 2>&1 | tee ...``
with no `` -- `` separator. Without recognizing that shape, every native Claude
Code run failed with "did not find the Harbor launch command" (part of the
Harbor upgrade for proof H4).
"""

from __future__ import annotations

import shlex

import pytest

pytest.importorskip("harbor")

from skillevaluator.tier3.harbor import native_agents
from skillevaluator.tier3.plugin_native import SETUP_SCRIPT, ClaudeCodeAdapter

# Harbor 0.24 ClaudeCode.run() launch, with the settings flag it adds for a config source.
_HARBOR_024_LAUNCH = (
    'export PATH="$HOME/.local/bin:$PATH"; '
    'harbor_claude_code_instruction_0123abcd="$HARBOR_CLAUDE_CODE_INSTRUCTION_0123ABCD"; '
    "unset HARBOR_CLAUDE_CODE_INSTRUCTION_0123ABCD; "
    'printf "%s" "$harbor_claude_code_instruction_0123abcd" | '
    "claude --verbose --output-format=stream-json --settings /logs/agent/settings.json --print 2>&1 | tee "
    "/logs/agent/claude-code.txt"
)
_HARBOR_013_LAUNCH = (
    'export PATH="$HOME/.local/bin:$PATH"; claude --verbose --output-format=stream-json --print -- "Do the task."'
)


def _agent() -> native_agents.NativeClaudeCode:
    return native_agents.NativeClaudeCode.__new__(native_agents.NativeClaudeCode)


@pytest.mark.parametrize("launch", [_HARBOR_024_LAUNCH, _HARBOR_013_LAUNCH], ids=["harbor-0.24", "harbor-0.13"])
def test_native_claude_code_rewrites_harbors_launch(launch: str) -> None:
    env = {"HARBOR_CLAUDE_CODE_INSTRUCTION_0123ABCD": "Do the task."}

    command, launch_env = _agent().skilleval_native_command(launch, env)

    assert command.startswith(f"/bin/sh {shlex.quote(SETUP_SCRIPT)} ")
    assert f"claude --plugin-dir {shlex.quote(ClaudeCodeAdapter.plugin_dir)} --verbose" in command
    assert command.count("--plugin-dir") == 1
    # The prompt still reaches Claude Code the way Harbor passed it.
    assert command.endswith(launch[launch.index(" --verbose") :])
    assert launch_env == env


@pytest.mark.parametrize(
    "command",
    [
        # Harbor's setup step before the launch.
        "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app && echo ok",
        # Task or plugin text that names Claude Code but is not Harbor's launch.
        "echo claude --verbose mcp serve",
        "sh -c 'x'; claude --verbose mcp serve",
        "printf x | claude --verbose --output-format=stream-json",
    ],
)
def test_native_claude_code_leaves_other_commands_alone(command: str) -> None:
    agent = _agent()

    assert agent.skilleval_native_command(command, None) == (command, None)
    assert not getattr(agent, "_skilleval_native_started", False)


def test_native_claude_code_refuses_a_second_launch_in_one_run() -> None:
    agent = _agent()
    agent.skilleval_native_command(_HARBOR_024_LAUNCH, None)

    with pytest.raises(native_agents.NativeLaunchError, match="second launch"):
        agent.skilleval_native_command(_HARBOR_024_LAUNCH, None)
