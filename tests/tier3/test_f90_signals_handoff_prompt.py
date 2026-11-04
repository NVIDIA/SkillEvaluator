# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value handoffs: only the real task prompt counts as "prompt" (proof H15, L22 prompt rule).

The trajectories copy the Harbor ATIF shapes of Claude Code and Codex: after a
``Skill`` call Claude Code injects a ``user`` step that holds the skill body and
an ``ARGUMENTS:`` line with the agent's own Skill arguments, and a Claude
subagent's prompt is a sidechain ``user`` step written by the parent agent.
Neither is the task prompt.
"""

from __future__ import annotations

import json
from typing import Any

from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals, plugin_case_spec

TAG = "1.5.0+rk.23b59e"
DECLARED = {
    "skill": ["changelog-collect", "version-bump", "release-notes"],
    "mcp": ["reltools"],
    "plugin": ["release-kit"],
}
CASE = plugin_case_spec(
    {
        "id": "full-release",
        "handoffs": [{"producer": "Skill:version-bump", "consumer": "Skill:release-notes", "value": TAG}],
    }
)


def _agent(call_id: str, fn: str, args: dict[str, Any], content: str, **extra: Any) -> dict[str, Any]:
    return {
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": call_id, "function_name": fn, "arguments": args}],
        "observation": {"results": [{"source_call_id": call_id, "content": content}]},
        "extra": {"is_sidechain": False, **extra},
    }


def _user(text: str, *, sidechain: bool = False) -> dict[str, Any]:
    return {"source": "user", "message": text, "extra": {"is_sidechain": sidechain}}


def _skill_load(name: str, arguments: str = "") -> dict[str, Any]:
    body = f"Base directory for this skill: /tmp/skilleval/native/claude-code/plugin/skills/{name}\n\n# {name}\n"
    return _user(body + (f"\n\nARGUMENTS: {arguments}" if arguments else ""))


def _claude_echo_trajectory(prompt: str = "Prepare the next atlas release. The last tag is 1.4.2.") -> dict:
    """check-18 e01: Claude passes the computed tag to release-notes as Skill arguments."""
    compute = json.dumps({"build_tag": TAG, "bump": "minor"})
    return {
        "agent": {"name": "claude-code"},
        "steps": [
            _user(prompt),
            _agent("t1", "Skill", {"skill": "release-kit:version-bump"}, "Launching skill: release-kit:version-bump"),
            _skill_load("version-bump"),
            _agent(
                "t2",
                "mcp__plugin_release-kit_reltools__compute_version",
                {"current": "1.4.2", "bump": "minor"},
                compute,
            ),
            _agent(
                "t3",
                "Skill",
                {"skill": "release-kit:release-notes", "args": f"project=atlas build_tag={TAG}"},
                "Launching skill: release-kit:release-notes",
            ),
            _skill_load("release-notes", f"project=atlas build_tag={TAG}"),
            _agent(
                "t4",
                "mcp__plugin_release-kit_reltools__stage_release",
                {"project": "atlas", "version": TAG, "notes_path": "RELEASE_NOTES.md"},
                '{"staged": true}',
            ),
            {"source": "agent", "message": "Staged.", "extra": {"is_sidechain": False}},
        ],
    }


def _handoff(trajectory: dict[str, Any], declared: dict[str, Any] | None = None) -> dict[str, Any]:
    signals = compute_plugin_signals(trajectory, CASE, declared=declared or DECLARED)
    assert signals is not None
    return signals["handoff"]


def test_claude_skill_argument_echo_is_not_the_task_prompt() -> None:
    handoff = _handoff(_claude_echo_trajectory())

    assert (handoff["checked"], handoff["passed"]) == (1, 1), handoff["failures"]


def test_value_in_the_real_task_prompt_still_fails() -> None:
    handoff = _handoff(_claude_echo_trajectory(prompt=f"Release atlas as {TAG}."))

    assert handoff["passed"] == 0
    assert "task prompt" in handoff["failures"][0]["detail"]


def test_codex_leading_system_and_user_steps_are_the_prompt() -> None:
    """Codex puts AGENTS.md in a leading system/user step: a value there is still prompt text."""
    exec_read = "Process exited with code 0\nOutput:\n---\nname: {0}\n---\n# {0}\n"
    trajectory = {
        "agent": {"name": "codex"},
        "steps": [
            {"source": "system", "message": f"<permissions instructions> pinned tag {TAG}"},
            {"source": "user", "message": "Prepare the next atlas release."},
            _agent(
                "c1",
                "exec_command",
                {"cmd": "sed -n '1,220p' /tmp/agent-home/.agents/skills/version-bump/SKILL.md"},
                exec_read.format("version-bump"),
            ),
            _agent("c2", "mcp__reltools__compute_version", {"current": "1.4.2"}, f'{{"build_tag": "{TAG}"}}'),
            _agent(
                "c3",
                "exec_command",
                {"cmd": "sed -n '1,220p' /tmp/agent-home/.agents/skills/release-notes/SKILL.md"},
                exec_read.format("release-notes"),
            ),
            _agent("c4", "mcp__reltools__stage_release", {"version": TAG}, '{"staged": true}'),
        ],
    }

    handoff = _handoff(trajectory)

    assert handoff["passed"] == 0
    assert "task prompt" in handoff["failures"][0]["detail"]

    trajectory["steps"][0]["message"] = "<permissions instructions>"
    assert _handoff(trajectory)["passed"] == 1


def test_subagent_prompt_written_by_the_parent_is_not_the_task_prompt() -> None:
    """check-18 e04 (Harbor 0.20+ shape): a sidechain user step is the parent's own words."""
    trajectory = _claude_echo_trajectory()
    steps = trajectory["steps"]
    # Replace the echoed Skill arguments with a subagent that receives the tag in its prompt.
    steps[4] = _agent(
        "t3", "Skill", {"skill": "release-kit:release-notes"}, "Launching skill: release-kit:release-notes"
    )
    steps[5] = _skill_load("release-notes")
    steps[6:6] = [
        _agent(
            "t5",
            "Agent",
            {"subagent_type": "general-purpose", "prompt": f"Write the notes for {TAG}."},
            "Wrote the notes.",
        ),
        _user(f"Write the notes for {TAG}.", sidechain=True),
    ]

    handoff = _handoff(trajectory)

    assert (handoff["checked"], handoff["passed"]) == (1, 1), handoff["failures"]
