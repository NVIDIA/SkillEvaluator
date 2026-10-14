# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A member skill named after its plugin is a component, not the wrapper (proof M26, L19, M7).

Many plugins ship one skill with the plugin's own name (a ``csv-tidy`` plugin
with a ``csv-tidy`` skill). The runner passes the plugin name as a wrapper
alias, because the generated wrapper ``SKILL.md`` is named after the plugin.
That alias must not hide the member skill. The shapes follow the saved Harbor
ATIF trajectories: Claude Code native ``Skill(<plugin>:<skill>)``, a bare
``Skill(<skill>)``, and Codex ``exec_command`` reads of ``<skill>/SKILL.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.eval_core.plugin_signals import (
    ARM_WITH_SKILL,
    PluginSignalsContext,
    build_plugin_signals_context,
    compute_plugin_signals,
    plugin_case_spec,
)
from skillevaluator.tier3.harbor import runner

SKILLS = "/tmp/agent-home/.agents/skills"


def _step(call_id: str, fn: str, args: dict[str, Any], content: str = "ok") -> dict[str, Any]:
    return {
        "source": "agent",
        "tool_calls": [{"tool_call_id": call_id, "function_name": fn, "arguments": args}],
        "observation": {"results": [{"source_call_id": call_id, "content": content}]},
        "extra": {"cwd": "/workspace", "is_sidechain": False},
    }


def _exec(call_id: str, cmd: str, output: str = "") -> dict[str, Any]:
    content = f"Chunk ID: c0\nWall time: 0.0000 seconds\nProcess exited with code 0\nOutput:\n{output}"
    return _step(call_id, "exec_command", {"cmd": cmd, "workdir": "/workspace"}, content)


def _trajectory(agent: str, *steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": {"name": agent, "extra": {"cwd": "/workspace"}},
        "steps": [{"source": "user", "message": "Tidy input/data.csv and report on it."}, *steps],
    }


def _skill(call_id: str, name: str) -> dict[str, Any]:
    return _step(call_id, "Skill", {"skill": name}, f"Launching skill: {name}")


def _codex_read(call_id: str, member: str) -> dict[str, Any]:
    return _exec(call_id, f"sed -n '1,200p' {SKILLS}/{member}/SKILL.md", f"---\nname: {member}\n---\n")


def _runner_context(tmp_path: Path, members: list[str], package: str = "csv-tidy-plugin-eval") -> PluginSignalsContext:
    """The context the runner really builds for a ``<plugin>-plugin-eval`` package."""
    package_path = tmp_path / package
    (package_path / "evals" / "environment").mkdir(parents=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return runner._plugin_signals_context(
        skill_path=package_path,
        evaluator_skill_path=package_path,
        workspace_skills=[tmp_path / "skills" / member for member in members],
        run_dir=run_dir,
        baseline_has_members=False,
    )


def _signals(context: PluginSignalsContext, trajectory: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    signals = compute_plugin_signals(
        trajectory,
        plugin_case_spec({"id": "c", **case}),
        declared=context.declared_for(ARM_WITH_SKILL),
        wrapper_skills=context.wrapper_skills,
    )
    assert signals is not None
    return signals


# The same-named member skill reached three ways: Claude Code native, a bare Skill call, and a Codex read.
SAME_NAME_LOADS = {
    "claude-native": ("claude-code", lambda: _skill("t1", "csv-tidy:csv-tidy")),
    "claude-bare": ("claude-code", lambda: _skill("t1", "csv-tidy")),
    "claude-read": ("claude-code", lambda: _step("t1", "Read", {"file_path": f"{SKILLS}/csv-tidy/SKILL.md"}, "1\t---")),
    "codex-read": ("codex", lambda: _codex_read("c1", "csv-tidy")),
}


def test_runner_still_passes_the_plugin_name_as_a_wrapper_alias(tmp_path: Path) -> None:
    context = _runner_context(tmp_path, ["csv-tidy", "csv-report"])

    assert context.wrapper_skills == ("csv-tidy-plugin-eval", "csv-tidy")
    assert context.declared_for(ARM_WITH_SKILL)["plugin"] == ["csv-tidy"]


@pytest.mark.parametrize("shape", sorted(SAME_NAME_LOADS))
def test_member_named_after_the_plugin_counts_for_routing_and_coverage(tmp_path: Path, shape: str) -> None:
    context = _runner_context(tmp_path, ["csv-tidy", "csv-report"])
    agent, load = SAME_NAME_LOADS[shape]

    signals = _signals(context, _trajectory(agent, load()), {"expected_tools": ["Skill:csv-tidy"]})

    assert signals["activation_coverage"]["exercised"] == ["skill:csv-tidy"]
    assert signals["routing"]["recall"] == 1.0
    assert signals["routing"]["precision"] == 1.0


@pytest.mark.parametrize("shape", sorted(SAME_NAME_LOADS))
def test_member_named_after_the_plugin_satisfies_a_conflict_probe(tmp_path: Path, shape: str) -> None:
    context = _runner_context(tmp_path, ["csv-tidy", "csv-report"])
    agent, load = SAME_NAME_LOADS[shape]
    case = {"conflict_probes": [{"id": "p", "must_use": "Skill:csv-tidy", "must_not_use": "WebSearch"}]}

    conflict = _signals(context, _trajectory(agent, load()), case)["conflict"]

    assert (conflict["passed"], conflict["checked"]) == (1, 1)


def test_member_named_after_the_plugin_orders_before_another_member(tmp_path: Path) -> None:
    context = _runner_context(tmp_path, ["csv-tidy", "csv-report"])
    case = {"expected_order": [["Skill:csv-tidy", "Skill:csv-report"]]}
    claude = _trajectory("claude-code", _skill("t1", "csv-tidy:csv-tidy"), _skill("t2", "csv-tidy:csv-report"))
    codex = _trajectory("codex", _codex_read("c1", "csv-tidy"), _codex_read("c2", "csv-report"))

    for trajectory in (claude, codex):
        order = _signals(context, trajectory, case)["order"]
        assert (order["satisfied"], order["edges"]) == (1, 1)


def test_member_named_after_the_plugin_opens_a_handoff_window(tmp_path: Path) -> None:
    context = _runner_context(tmp_path, ["csv-tidy", "csv-report"])
    case = {"handoffs": [{"producer": "Skill:csv-tidy", "consumer": "Skill:csv-report", "artifact": "out/tidy.csv"}]}
    claude = _trajectory(
        "claude-code",
        _skill("t1", "csv-tidy:csv-tidy"),
        _step("t2", "Write", {"file_path": "/workspace/out/tidy.csv", "content": "a,b\n1,2\n"}),
        _skill("t3", "csv-tidy:csv-report"),
        _step("t4", "Read", {"file_path": "/workspace/out/tidy.csv"}, "1\ta,b"),
    )
    codex = _trajectory(
        "codex",
        _codex_read("c1", "csv-tidy"),
        _exec("c2", "mkdir -p out && printf 'a,b\\n1,2\\n' > out/tidy.csv"),
        _codex_read("c3", "csv-report"),
        _exec("c4", "cat out/tidy.csv", "a,b\n1,2\n"),
    )

    for trajectory in (claude, codex):
        handoff = _signals(context, trajectory, case)["handoff"]
        assert (handoff["passed"], handoff["checked"]) == (1, 1), handoff


# ---------------------------------------------------------------------------
# The wrapper itself still never counts
# ---------------------------------------------------------------------------


def test_plugin_name_is_still_the_wrapper_when_no_member_has_it(tmp_path: Path) -> None:
    context = _runner_context(tmp_path, ["changelog-collect"], package="release-kit-plugin-eval")
    trajectory = _trajectory("claude-code", _skill("t1", "release-kit"), _skill("t2", "changelog-collect"))

    signals = _signals(context, trajectory, {"expected_tools": ["Skill:release-*", "Skill:changelog-collect"]})

    assert signals["activation_coverage"]["exercised"] == ["skill:changelog-collect"]
    assert signals["routing"]["precision"] == 1.0


def test_wrapper_package_name_always_means_the_wrapper() -> None:
    context = build_plugin_signals_context(
        member_skills=["csv-tidy", "csv-tidy-plugin-eval"],
        wrapper_skills=["csv-tidy-plugin-eval", "csv-tidy"],
    )
    trajectory = _trajectory("claude-code", _skill("t1", "csv-tidy-plugin-eval"))

    signals = _signals(context, trajectory, {"expected_tools": ["Skill:csv-tidy"]})

    assert signals["activation_coverage"]["exercised"] == []
    assert signals["routing"]["called"] == []


def test_reading_the_wrapper_package_is_not_a_member_load(tmp_path: Path) -> None:
    context = _runner_context(tmp_path, ["csv-tidy", "csv-report"])
    trajectory = _trajectory("codex", _codex_read("c1", "csv-tidy-plugin-eval"))

    signals = _signals(context, trajectory, {"expected_tools": ["Skill:csv-tidy"]})

    assert signals["activation_coverage"]["exercised"] == []
    assert signals["routing"]["recall"] == 0.0
