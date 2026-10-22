# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 16 order: reasons, ties, parallel calls, failed calls, arms, validation, reports (proof M27, L20).

Shapes follow the check-16 replay examples: Claude Code native ``Skill`` and
``mcp__plugin_<plugin>_<server>__<tool>`` calls, Codex ``exec_command`` reads of
``<member>/SKILL.md`` and bare MCP names mapped through the session log.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting.cli import print_plugin_tier3
from skillevaluator.reporting.plugin_sections import signals_view
from skillevaluator.tier3.eval_core.plugin_signals import (
    ARM_SUM_OF_PARTS,
    ARM_WITH_SKILL,
    MAX_ORDER_EDGES,
    build_plugin_signals_context,
    compute_plugin_signals,
    plugin_case_spec,
    summarize_plugin_signals,
    validate_plugin_case_fields,
)

CONTEXT = build_plugin_signals_context(
    member_skills=["changelog-collect", "version-bump", "release-notes"],
    mcp_servers=["reltools"],
    wrapper_skills=["release-kit-plugin-eval", "release-kit"],
    subagents=["release-reviewer"],
    commands=["release-check"],
)
SKILLS = "/tmp/agent-home/.agents/skills"


def _calls_step(*calls: tuple[str, str, dict[str, Any], str]) -> dict[str, Any]:
    """One agent step; several calls in it were issued together (parallel)."""
    return {
        "source": "agent",
        "tool_calls": [{"tool_call_id": cid, "function_name": fn, "arguments": args} for cid, fn, args, _ in calls],
        "observation": {"results": [{"source_call_id": cid, "content": out} for cid, _, _, out in calls]},
    }


def _step(call_id: str, fn: str, args: dict[str, Any], content: str = "ok") -> dict[str, Any]:
    return _calls_step((call_id, fn, args, content))


def _trajectory(agent: str, *steps: dict[str, Any]) -> dict[str, Any]:
    return {"agent": {"name": agent}, "steps": [{"source": "user", "message": "Prepare the release."}, *steps]}


def _launch(call_id: str, name: str) -> dict[str, Any]:
    return _step(call_id, "Skill", {"skill": f"release-kit:{name}"}, f"Launching skill: release-kit:{name}")


def _codex_cat(call_id: str, *members: str) -> dict[str, Any]:
    cmd = " && printf '\\n---\\n' && ".join(f"cat {SKILLS}/{member}/SKILL.md" for member in members)
    body = "\n---\n".join(f"---\nname: {member}\n---\n" for member in members)
    return _step(
        call_id, "exec_command", {"cmd": cmd, "workdir": "/workspace"}, f"Process exited with code 0\nOutput:\n{body}"
    )


def _order(trajectory: dict[str, Any], edges: list[Any], *, arm: str = ARM_WITH_SKILL) -> dict[str, Any]:
    signals = compute_plugin_signals(
        trajectory,
        plugin_case_spec({"id": "c", "expected_order": edges}),
        declared=CONTEXT.declared_for(arm),
        wrapper_skills=CONTEXT.wrapper_skills,
    )
    assert signals is not None
    return signals["order"]


def _reasons(block: dict[str, Any]) -> list[str]:
    return [item["reason"] for item in block["violated"]]


# ---------------------------------------------------------------------------
# Why an edge failed (M27)
# ---------------------------------------------------------------------------


def test_batched_skill_reads_in_one_call_are_unordered_not_reversed() -> None:
    """check-16 edge-04: ``cat cc/SKILL.md && cat vb/SKILL.md`` read both at once."""
    codex = _trajectory("codex", _codex_cat("c1", "changelog-collect", "version-bump"))
    claude = _trajectory(
        "claude-code",
        _step(
            "t1",
            "Bash",
            {"command": f"cat {SKILLS}/changelog-collect/SKILL.md && cat {SKILLS}/version-bump/SKILL.md"},
            "---\nname: changelog-collect\n---\n---\nname: version-bump\n---\n",
        ),
    )
    for trajectory in (codex, claude):
        block = _order(trajectory, [["Skill:changelog-collect", "Skill:version-bump"]])
        assert (block["edges"], block["satisfied"], block["unordered"]) == (1, 0, 1)
        assert _reasons(block) == ["same_call"]


def test_never_called_and_reversed_are_told_apart() -> None:
    """check-16 pos-01 (reversed) and pos-02 (never called) gave the same violated entry."""
    reversed_trial = _trajectory("claude-code", _launch("t1", "version-bump"), _launch("t2", "changelog-collect"))
    missing_trial = _trajectory("claude-code", _launch("t1", "changelog-collect"))
    edge = [["Skill:changelog-collect", "Skill:version-bump"]]

    assert _reasons(_order(reversed_trial, edge)) == ["reversed"]
    assert _reasons(_order(missing_trial, edge)) == ["after_never_called"]
    assert _reasons(_order(_trajectory("claude-code", _launch("t1", "version-bump")), edge)) == ["before_never_called"]
    assert _reasons(_order(_trajectory("claude-code"), edge)) == ["never_called"]


def test_parallel_calls_in_one_step_have_no_order() -> None:
    """check-16 edge-11: two calls issued in one message are not ordered by list position."""
    trajectory = _trajectory(
        "claude-code",
        _calls_step(
            ("t1", "mcp__reltools__list_changes", {"project": "atlas"}, "[]"),
            ("t2", "mcp__reltools__compute_version", {"current": "1.4.2"}, '{"version": "1.5.0"}'),
        ),
        _calls_step(
            ("t3", "mcp__reltools__stage_release", {"project": "atlas"}, '{"staged": true}'),
            ("t4", "Skill", {"skill": "version-bump"}, "Launching skill: version-bump"),
        ),
    )
    block = _order(
        trajectory,
        [
            ["MCP:reltools/list_changes", "MCP:reltools/compute_version"],
            ["MCP:reltools/compute_version", "MCP:reltools/list_changes"],
            ["Skill:version-bump", "MCP:reltools/stage_release"],
            ["MCP:reltools/list_changes", "Skill:version-bump"],
        ],
    )
    assert (block["edges"], block["satisfied"], block["unordered"]) == (4, 1, 3)
    assert _reasons(block) == ["same_step", "same_step", "same_step"]


def test_failed_calls_are_not_the_first_use() -> None:
    """check-16 edge-05: a failed ``before`` call does not satisfy an edge."""
    trajectory = _trajectory(
        "claude-code",
        _step("t1", "mcp__plugin_release-kit_reltools__list_changes", {"limit": 200}, "MCP error -32000: closed"),
        _step("t2", "Skill", {"skill": "changelog-collect"}, "<tool_use_error>Unknown skill: x</tool_use_error>"),
        _launch("t3", "version-bump"),
    )
    block = _order(trajectory, [["MCP:reltools/list_changes", "Skill:version-bump"]])
    assert _reasons(block) == ["before_never_called"]


def test_an_early_look_at_an_after_alternative_does_not_flip_the_edge() -> None:
    """check-16 edge-03: the agent peeked at release-notes, then did the work in order."""
    trajectory = _trajectory(
        "codex",
        _codex_cat("c1", "release-notes"),
        _step("c2", "mcp__reltools__list_changes", {"project": "atlas"}, "[]"),
        _step("c3", "mcp__reltools__stage_release", {"project": "atlas"}, '{"staged": true}'),
    )
    block = _order(
        trajectory,
        [
            [
                ["Skill:changelog-collect", "MCP:reltools/list_changes"],
                ["Skill:release-notes", "MCP:reltools/stage_release"],
            ]
        ],
    )
    assert (block["edges"], block["satisfied"]) == (1, 1)


def test_edges_for_components_the_arm_cannot_have_are_skipped() -> None:
    """Codex stages no plugin agents or commands; the member-skills arm has no MCP server."""
    codex = _trajectory("codex", _codex_cat("c1", "release-notes"))
    block = _order(
        codex, [["Command:release-check", "Agent:release-reviewer"], ["Skill:release-notes", "Skill:version-bump"]]
    )
    assert block["edges"] == 1
    assert [item["before"] for item in block["skipped"]] == ["Command:release-check"]

    sum_of_parts = _trajectory("claude-code", _launch("t1", "changelog-collect"))
    block = _order(sum_of_parts, [["Skill:changelog-collect", "MCP:reltools/list_changes"]], arm=ARM_SUM_OF_PARTS)
    assert (block["edges"], block["status"]) == (0, "not_applicable")
    assert len(block["skipped"]) == 1


def test_every_violated_edge_is_listed() -> None:
    edges = [[f"Skill:missing-{index}", "Skill:version-bump"] for index in range(MAX_ORDER_EDGES)]
    block = _order(_trajectory("claude-code", _launch("t1", "version-bump")), edges)
    assert len(block["violated"]) == MAX_ORDER_EDGES


# ---------------------------------------------------------------------------
# Dataset validation for never-satisfiable edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edge", "fragment"),
    [
        (["Skill:changelog-collect", "Skill:changelog-collect"], "can never pass"),
        (["MCP:reltools/list_changes", "MCP:reltools"], "can never pass"),
        (["Skill:release-*", "Skill:release-notes"], ""),
        (["Skil:changelog-collect", "Skill:version-bump"], "did you mean 'Skill:'"),
        (["MPC:reltools", "Skill:version-bump"], "did you mean 'MCP:'"),
    ],
)
def test_never_satisfiable_edges_are_rejected(edge: list[str], fragment: str) -> None:
    problems = validate_plugin_case_fields({"id": "c", "expected_order": [edge]})
    if fragment:
        assert any(fragment in problem for problem in problems), problems
    else:
        assert problems == []


def test_satisfiable_overlapping_and_namespaced_edges_are_accepted() -> None:
    entry = {
        "id": "c",
        "expected_order": [["MCP:reltools", "MCP:reltools/list_changes"], ["release-kit:changelog-collect", "Bash"]],
    }
    assert validate_plugin_case_fields(entry) == []


# ---------------------------------------------------------------------------
# Summary and reports (L20 unit and violated edges)
# ---------------------------------------------------------------------------


def _view() -> dict[str, Any]:
    edge = [["Skill:changelog-collect", "Skill:version-bump"], ["Skill:version-bump", "Skill:release-notes"]]
    in_order = _trajectory(
        "claude-code", _launch("t1", "changelog-collect"), _launch("t2", "version-bump"), _launch("t3", "release-notes")
    )
    reversed_trial = _trajectory(
        "claude-code", _launch("t1", "version-bump"), _launch("t2", "changelog-collect"), _launch("t3", "release-notes")
    )
    signals = []
    for trajectory in (in_order, reversed_trial):
        result = compute_plugin_signals(
            trajectory,
            plugin_case_spec({"id": "c", "expected_order": edge}),
            declared=CONTEXT.declared_for(ARM_WITH_SKILL),
        )
        assert result is not None
        signals.append(result)
    summary = summarize_plugin_signals(signals)
    order = summary["order"]
    assert (order["edges"], order["satisfied"], order["trials_in_order"]) == (4, 3, 1)
    assert order["violated_edges"] == [
        {"before": "Skill:changelog-collect", "after": "Skill:version-bump", "reason": "reversed", "trials": 1}
    ]
    view = signals_view({"agents": {"claude-code": {"plugin_signals_summary": {"with_skill": summary}}}})
    assert view is not None
    return view


def test_reports_say_the_unit_is_edges_and_list_the_edges_not_in_order() -> None:
    view = _view()
    html = (
        HTMLReporter()
        ._env.from_string('{% from "plugin_sections.html.j2" import signals_section %}{{ signals_section(sig) }}')
        .render(sig=view)
    )
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    assert "Order checks 3/4 edges in order (75%) 2 trial(s) scored; 1 of 2 trial(s) fully in order" in text
    assert "Skill:changelog-collect Skill:version-bump &#39;after&#39; was used first 1" in text

    console = Console(file=StringIO(), width=200, color_system=None)
    print_plugin_tier3({"signals": view}, console)
    plain = " ".join(console.file.getvalue().split())
    assert "order 3/4 edges in order (75%); 1 of 2 trial(s) fully in order" in plain
    assert "not in order: Skill:changelog-collect -> Skill:version-bump: 'after' was used first (1 trial(s))" in plain
