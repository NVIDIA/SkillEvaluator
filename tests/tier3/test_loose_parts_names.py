# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One component, two spellings: the full native plugin arm and the loose-parts arm.

Native Claude Code namespaces a plugin's parts (``Skill(shipit:changelog-collect)``,
``Agent(shipit:release-reviewer)``, ``mcp__plugin_shipit_reltools__list_changes``);
the loose-parts arm stages the same parts with bare names
(``Skill(changelog-collect)``, ``mcp__reltools__list_changes``). Routing, tool
selection, order, handoff, MCP proof, activation coverage, and the collector's
``skill_invoked`` must credit the same declared component in both spellings, or
a naming artefact would show up as glue.
"""

from __future__ import annotations

from typing import Any

import pytest

from skillevaluator.tier3.eval_core.plugin_signals import (
    COMPONENT_COMMAND,
    COMPONENT_LSP,
    COMPONENT_MCP,
    COMPONENT_SKILL,
    COMPONENT_SUBAGENT,
    DECLARED_UNSTAGED,
    DECLARED_UNSTAGED_NAMES,
    build_plugin_signals_context,
    compute_plugin_signals,
    summarize_plugin_signals,
)
from skillevaluator.tier3.harbor.collector import _trajectory_skill_invoked
from skillevaluator.tier3.loose_parts import LOOSE_PARTS_ARM
from skillevaluator.tier3.mcp_proof import apply_in_agent_mcp_proof

CASE = {
    "id": "case-1",
    "prompt": "Prepare the release notes for v1.3.0 and have them reviewed.",
    "expected_skills": ["changelog-collect", "version-bump"],
    "cross_component": True,
    "expected_tools": ["Skill:changelog-collect", "MCP:reltools/list_changes", "Agent:release-reviewer"],
    "expected_order": [["MCP:reltools/list_changes", "Skill:version-bump"]],
    "handoffs": [
        {"producer": "MCP:reltools", "consumer": "Skill:version-bump", "value": "RK-417"},
    ],
}


def _context():
    return build_plugin_signals_context(
        member_skills=["changelog-collect", "version-bump"],
        mcp_servers=[],
        entries=[CASE],
        subagents=["release-reviewer"],
        commands=["release-check"],
        plugin_name="shipit",
        agent_mcp_servers={"claude-code": ["reltools"]},
        agent_lsp_servers={"claude-code": {"go": [".go"]}},
    )


def _trajectory(*calls: tuple[str, dict[str, Any], Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [{"step_id": 1, "source": "user", "message": CASE["prompt"]}]
    for index, (function, arguments, content) in enumerate(calls, start=1):
        steps.append(
            {
                "step_id": index + 1,
                "source": "agent",
                "message": "",
                "tool_calls": [{"tool_call_id": f"c{index}", "function_name": function, "arguments": arguments}],
                "observation": {"results": [{"source_call_id": f"c{index}", "content": content}]},
            }
        )
    return {"schema_version": "ATIF-v1.7", "agent": {"name": "claude-code", "version": "2.1"}, "steps": steps}


def _session(spell) -> dict[str, Any]:
    """One Claude Code session; ``spell(kind, name)`` gives the arm's spelling of a component."""
    return _trajectory(
        ("Skill", {"skill": spell("skill", "changelog-collect")}, "Launching skill"),
        (spell("mcp", "list_changes"), {"since": "v1.2.3"}, "Changes: RK-417 Fix release tag"),
        ("Skill", {"skill": spell("skill", "version-bump")}, "Launching skill"),
        ("Bash", {"command": "sh scripts/bump.sh RK-417"}, "1.3.0"),
        ("Agent", {"subagent_type": spell("agent", "release-reviewer"), "prompt": "Review"}, "PASS"),
        ("Skill", {"skill": spell("command", "release-check")}, "Launching skill"),
    )


def _full(kind: str, name: str) -> str:
    return f"mcp__plugin_shipit_reltools__{name}" if kind == "mcp" else f"shipit:{name}"


def _loose(kind: str, name: str) -> str:
    return f"mcp__reltools__{name}" if kind == "mcp" else name


_GRADED = ("routing", "tool_selection", "order", "handoff", "activation_coverage", "activations", "mcp_calls")


def test_loose_parts_arm_declares_every_part_it_stages() -> None:
    context = _context()
    declared = context.declared_for(LOOSE_PARTS_ARM, "claude-code")
    assert context.arm_enabled(LOOSE_PARTS_ARM)
    assert declared[COMPONENT_SKILL] == ["changelog-collect", "version-bump"]
    assert declared[COMPONENT_MCP] == ["reltools"]
    assert declared[COMPONENT_SUBAGENT] == ["release-reviewer"]
    assert declared[COMPONENT_COMMAND] == ["release-check"]
    # LSP servers load only from a plugin: glue, so unstaged in the loose arm.
    assert COMPONENT_LSP not in declared
    assert declared[DECLARED_UNSTAGED] == [COMPONENT_LSP]
    assert DECLARED_UNSTAGED_NAMES not in declared
    assert context.lsp_servers_for(LOOSE_PARTS_ARM, "claude-code") == {}


@pytest.mark.parametrize("piece", _GRADED)
def test_both_spellings_grade_the_same(piece: str) -> None:
    context = _context()
    full = compute_plugin_signals(_session(_full), CASE, declared=context.declared_for("with_skill", "claude-code"))
    loose = compute_plugin_signals(
        _session(_loose), CASE, declared=context.declared_for(LOOSE_PARTS_ARM, "claude-code")
    )
    assert full is not None and loose is not None
    if piece == "activation_coverage":
        # The full arm also declares the LSP server it stages; every other part is credited the same way.
        assert full[piece]["exercised"] == loose[piece]["exercised"]
        assert set(full[piece]["declared"]) - set(loose[piece]["declared"]) <= {"lsp:go"}
        return
    assert full[piece] == loose[piece]


def test_both_spellings_credit_every_declared_part() -> None:
    context = _context()
    loose = compute_plugin_signals(
        _session(_loose), CASE, declared=context.declared_for(LOOSE_PARTS_ARM, "claude-code")
    )
    assert loose is not None
    assert loose["routing"]["recall"] == 1.0
    assert loose["tool_selection"]["recall"] == 1.0
    assert loose["order"]["satisfied"] == loose["order"]["edges"] == 1
    assert loose["handoff"]["passed"] == loose["handoff"]["checked"] == 1
    assert sorted(loose["activation_coverage"]["exercised"]) == [
        "command:release-check",
        "mcp:reltools",
        "skill:changelog-collect",
        "skill:version-bump",
        "subagent:release-reviewer",
    ]
    assert list(loose["mcp_calls"]["by_server"]) == ["reltools"]


def test_mcp_proof_reads_both_spellings_as_the_declared_server() -> None:
    context = _context()
    for spell, arm in ((_full, "with_skill"), (_loose, LOOSE_PARTS_ARM)):
        signals = compute_plugin_signals(_session(spell), CASE, declared=context.declared_for(arm, "claude-code"))
        summary = summarize_plugin_signals([signals])
        engine = {"agents": {"claude-code": {"plugin_signals_summary": {"with_skill": summary}}}}
        proof = apply_in_agent_mcp_proof({"reltools": {"status": "declared", "tools": [], "detail": ""}}, engine)
        assert proof["reltools"]["status"] == "used-successfully", arm


def _skill_session(name: str) -> dict[str, Any]:
    return _trajectory(("Skill", {"skill": name}, "Launching skill"))


@pytest.mark.parametrize("spelled", ["changelog-collect", "shipit:changelog-collect"])
def test_collector_skill_invoked_credits_both_spellings_with_the_native_prefix(spelled: str) -> None:
    assert _trajectory_skill_invoked(_skill_session(spelled), "changelog-collect", native_prefix="shipit") is True


def test_collector_skill_invoked_without_a_prefix_keeps_todays_exact_match() -> None:
    assert _trajectory_skill_invoked(_skill_session("changelog-collect"), "changelog-collect") is True
    assert _trajectory_skill_invoked(_skill_session("shipit:changelog-collect"), "changelog-collect") is False
    # Another plugin's namespace never counts.
    assert (
        _trajectory_skill_invoked(
            _skill_session("other:changelog-collect"), "changelog-collect", native_prefix="shipit"
        )
        is False
    )


def test_probed_mcp_input_schemas_reach_the_loose_arm_too() -> None:
    from dataclasses import replace

    from skillevaluator.tier3.eval_core.plugin_signals import MCP_INPUT_SCHEMAS_KEY
    from skillevaluator.tier3.harbor.collector import _case_spec_with_input_schemas

    schemas = {"reltools": {"list_changes": {"type": "object"}}}
    context = replace(_context(), mcp_input_schemas=schemas)
    for arm in ("with_skill", LOOSE_PARTS_ARM):
        assert _case_spec_with_input_schemas(context, "case-1", arm)[MCP_INPUT_SCHEMAS_KEY] == schemas
    assert MCP_INPUT_SCHEMAS_KEY not in _case_spec_with_input_schemas(context, "case-1", "sum_of_parts")
