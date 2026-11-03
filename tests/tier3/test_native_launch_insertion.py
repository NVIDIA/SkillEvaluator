# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native setup is inserted at Harbor's launch pipeline without changing the launch line's shell structure."""

from __future__ import annotations

import subprocess

import pytest

from skillevaluator.tier3.harbor import native_agents
from skillevaluator.tier3.harbor.native_agents import NativeLaunchError, _pipeline_start

SETUP = native_agents._SETUP_GROUP


def _mixin(cls: type) -> object:
    agent = object.__new__(cls)
    agent._skilleval_native_started = False
    return agent


CLAUDE = (
    'export PATH="$HOME/.local/bin:$PATH"; P="$SKILLEVAL_PROMPT"; unset SKILLEVAL_PROMPT; '
    'printf "%s" "$P" | claude --verbose --output-format=stream-json --print 2>&1 | tee /logs/agent/claude-code.txt'
)
CODEX = (
    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; codex exec --dangerously-bypass-approvals-and-sandbox "
    "--model m --json -- 'do the task' 2>&1 </dev/null | tee /logs/agent/codex.txt"
)
OPENCODE = "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; opencode --model=p/m run --format=json -- 'task' 2>&1 | tee x"
HERMES = 'export PATH="$HOME/.local/bin:$PATH" && hermes --yolo chat -q "$HARBOR_INSTRUCTION" 2>&1 | tee x'


@pytest.mark.parametrize(
    ("cls", "command", "before"),
    [
        (native_agents._NativeClaudeCodeMixin, CLAUDE, "printf "),
        (native_agents._NativeCodexMixin, CODEX, "codex exec"),
        (native_agents._NativeOpenCodeMixin, OPENCODE, "opencode --model"),
        (native_agents._NativeHermesMixin, HERMES, "hermes --yolo"),
    ],
)
def test_setup_runs_after_launch_line_initialization(cls: type, command: str, before: str) -> None:
    rewritten, _ = _mixin(cls).skilleval_native_command(command, None)
    head, _, tail = rewritten.partition(SETUP)
    assert tail.startswith(before)
    assert command.startswith(head)
    assert subprocess.run(["sh", "-n", "-c", rewritten], check=False).returncode == 0


def test_claude_launch_keeps_pipe_and_gets_plugin_dir() -> None:
    rewritten, _ = _mixin(native_agents._NativeClaudeCodeMixin).skilleval_native_command(CLAUDE, None)
    assert "| claude --plugin-dir " in rewritten
    assert rewritten.endswith("| tee /logs/agent/claude-code.txt")


@pytest.mark.parametrize("status", [0, 3])
def test_launch_exit_status_and_preceding_conditionals_are_preserved(tmp_path, monkeypatch, status: int) -> None:
    monkeypatch.setattr(native_agents, "_SETUP_GROUP", "{ false || true; } && ")
    marker = tmp_path / "ran"
    command = f"false && echo skipped; [ -d / ] && touch {marker}; codex exec -- x"
    rewritten, _ = _mixin(native_agents._NativeCodexMixin).skilleval_native_command(command, None)
    script = f"codex() {{ return {status}; }}; {rewritten}"
    assert subprocess.run(["sh", "-c", script], check=False).returncode == status
    assert marker.exists()


@pytest.mark.parametrize("prefix", ["false || ", "sleep 1 & ", "( ", "echo '"])
def test_unsupported_launch_boundaries_fail_closed(prefix: str) -> None:
    command = prefix + "codex exec -- x"
    with pytest.raises(NativeLaunchError):
        _pipeline_start(command, len(prefix))


def test_quoted_separators_are_not_boundaries() -> None:
    command = "echo 'a; b || c' \"d && e\"; codex exec -- x"
    assert command[_pipeline_start(command, command.index("codex")) :].startswith("codex exec")


def test_second_launch_in_one_run_fails() -> None:
    agent = _mixin(native_agents._NativeCodexMixin)
    agent.skilleval_native_command(CODEX, None)
    with pytest.raises(NativeLaunchError):
        agent.skilleval_native_command(CODEX, None)
