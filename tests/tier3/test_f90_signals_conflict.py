# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 19 conflict probes: failed calls, probe ids, side channels, refs, arms, reports (proof M29, L23).

Shapes follow the check-19 replay examples: Claude Code native and wrapper
calls, and Codex ``exec_command``/bare MCP names.
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
)
DOCS = build_plugin_signals_context(
    member_skills=["repo-docs", "cloudflare-docs"],
    mcp_servers=["deepwiki", "cfdocs"],
    wrapper_skills=["remote-docs-plugin-eval", "remote-docs"],
)
SKILLS = "/tmp/agent-home/.agents/skills"
STAGE_NOT_PUBLISH = {
    "id": "stage-not-publish",
    "must_use": "MCP:reltools/stage_release",
    "must_not_use": "MCP:reltools/publish_release",
}
JSONRPC = (
    "printf '%s\\n%s\\n' "
    '\'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}\' '
    '\'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"publish_release",'
    '"arguments":{"project":"atlas","version":"1.5.0"}}}\' | /tmp/release-kit/reltools'
)


def _step(call_id: str, fn: str, args: dict[str, Any], content: str = "ok", **result: Any) -> dict[str, Any]:
    return {
        "source": "agent",
        "tool_calls": [{"tool_call_id": call_id, "function_name": fn, "arguments": args}],
        "observation": {"results": [{"source_call_id": call_id, "content": content, **result}]},
    }


def _trajectory(agent: str, *steps: dict[str, Any]) -> dict[str, Any]:
    return {"agent": {"name": agent}, "steps": [{"source": "user", "message": "Prepare the release."}, *steps]}


def _exec(call_id: str, cmd: str, output: str = "") -> dict[str, Any]:
    content = f"Chunk ID: c0\nWall time: 0.0000 seconds\nProcess exited with code 0\nOutput:\n{output}"
    return _step(call_id, "exec_command", {"cmd": cmd, "workdir": "/workspace"}, content)


def _conflict(
    trajectory: dict[str, Any], *probes: dict[str, Any], context: Any = CONTEXT, arm: str = ARM_WITH_SKILL
) -> dict[str, Any]:
    signals = compute_plugin_signals(
        trajectory,
        plugin_case_spec({"id": "c", "conflict_probes": list(probes)}),
        declared=context.declared_for(arm),
        wrapper_skills=context.wrapper_skills,
    )
    assert signals is not None
    return signals["conflict"]


# ---------------------------------------------------------------------------
# A failed call does not satisfy must_use (M29; check-19 e01)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("agent", "fn", "result"),
    [
        (
            "claude-code",
            "mcp__reltools__stage_release",
            {"content": "<tool_use_error>Error: No such tool available: mcp__reltools__stage_release</tool_use_error>"},
        ),
        ("claude-code", "mcp__plugin_release-kit_reltools__stage_release", {"content": "MCP error -32000: closed"}),
        ("codex", "mcp__reltools__stage_release", {"content": "Output:\nfailed", "is_error": True}),
    ],
    ids=["claude-wrapper-missing-tool", "claude-native-mcp-error", "codex-flagged-error"],
)
def test_a_failed_call_does_not_satisfy_must_use(agent: str, fn: str, result: dict[str, Any]) -> None:
    trajectory = _trajectory(agent, _step("x1", fn, {"project": "atlas"}, **result))

    block = _conflict(trajectory, STAGE_NOT_PUBLISH)

    assert (block["checked"], block["passed"]) == (1, 0)
    assert block["failures"][0]["detail"] == "must_use MCP:reltools/stage_release was not activated"


def test_a_failed_forbidden_call_is_still_an_attempt() -> None:
    trajectory = _trajectory(
        "claude-code",
        _step("t1", "mcp__reltools__stage_release", {"project": "atlas"}, '{"staged": true}'),
        _step("t2", "mcp__reltools__publish_release", {"project": "atlas"}, "MCP error -32000: closed"),
    )
    block = _conflict(trajectory, STAGE_NOT_PUBLISH)
    assert block["failures"][0]["detail"] == "must_not_use MCP:reltools/publish_release was activated"


# ---------------------------------------------------------------------------
# Probe ids and contradictions (M29, L23; check-19 e08)
# ---------------------------------------------------------------------------


def test_probe_ids_that_differ_only_by_spaces_are_rejected_not_silently_dropped() -> None:
    entry = {
        "id": "c",
        "conflict_probes": [
            {"id": "dup", "must_use": "Skill:release-notes", "must_not_use": "Skill:version-bump"},
            {"id": " dup", "must_use": "Skill:release-notes", "must_not_use": "Skill:changelog-collect"},
        ],
    }
    problems = validate_plugin_case_fields(entry)
    assert problems == ["conflict_probes[1].id: must be unique (ids are compared without surrounding spaces)"]

    entry["conflict_probes"][1]["id"] = " other "
    assert validate_plugin_case_fields(entry) == []
    spec = plugin_case_spec(entry)
    trajectory = _trajectory("claude-code", _step("t1", "Skill", {"skill": "release-notes"}, "Launching skill"))
    signals = compute_plugin_signals(trajectory, spec, declared=CONTEXT.declared_for(ARM_WITH_SKILL))
    assert signals is not None
    assert (signals["conflict"]["checked"], signals["conflict"]["passed"]) == (2, 2)


@pytest.mark.parametrize(
    ("must_use", "must_not_use"),
    [
        ("MCP:reltools/stage_release", "MCP:reltools/stage_release"),
        ("MCP:reltools/stage_release", "MCP:reltools"),
        (["Skill:release-notes", "Skill:version-bump"], "Skill:*"),
    ],
)
def test_a_self_contradicting_probe_is_rejected(must_use: Any, must_not_use: Any) -> None:
    problems = validate_plugin_case_fields(
        {"id": "c", "conflict_probes": [{"id": "p", "must_use": must_use, "must_not_use": must_not_use}]}
    )
    assert problems == ["conflict_probes[0]: can never pass: every must_use ref is also matched by must_not_use"]


def test_a_probe_with_one_safe_alternative_is_accepted() -> None:
    probe = {
        "id": "p",
        "must_use": ["MCP:reltools/stage_release", "Skill:release-notes"],
        "must_not_use": "MCP:reltools",
    }
    assert validate_plugin_case_fields({"id": "c", "conflict_probes": [probe]}) == []


@pytest.mark.parametrize("ref", ["rule:release-policy", "hook:PreToolUse", "Hooks:pre-commit"])
def test_rule_and_hook_refs_are_rejected_with_a_clear_error(ref: str) -> None:
    """check-19 e03: these refs could never match a tool call."""
    problems = validate_plugin_case_fields(
        {"id": "c", "conflict_probes": [{"id": "p", "must_use": "Skill:release-notes", "must_not_use": ref}]}
    )
    assert len(problems) == 1
    assert "refs are not supported: rules and hooks are not tool calls" in problems[0]


# ---------------------------------------------------------------------------
# Side channels and batched reads (M29, M7; check-19 e06, e02)
# ---------------------------------------------------------------------------


def test_an_mcp_tool_called_through_the_shell_counts() -> None:
    """check-19 e06: publish_release sent as JSON-RPC to the server program through Bash/exec_command."""
    published = '{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\\"published\\": true}"}]}}'
    claude = _trajectory(
        "claude-code",
        _step("t1", "Skill", {"skill": "release-notes"}, "Launching skill: release-notes"),
        _step("t2", "Bash", {"command": JSONRPC}, published),
    )
    codex = _trajectory(
        "codex",
        _exec("c1", f"sed -n '1,220p' {SKILLS}/release-notes/SKILL.md", "---\nname: release-notes\n---\n"),
        _exec("c2", JSONRPC, published),
    )
    probe = {
        "id": "notes-not-publish",
        "must_use": "Skill:release-notes",
        "must_not_use": "MCP:reltools/publish_release",
    }
    for trajectory in (claude, codex):
        block = _conflict(trajectory, probe)
        assert block["passed"] == 0, trajectory["agent"]
        assert "must_not_use MCP:reltools/publish_release was activated" in block["failures"][0]["detail"]
    staged = _trajectory(
        "claude-code", _step("t1", "Bash", {"command": JSONRPC.replace("publish_release", "stage_release")})
    )
    assert _conflict(staged, STAGE_NOT_PUBLISH)["passed"] == 1


def test_running_a_member_skills_script_uses_that_skill() -> None:
    trajectory = _trajectory("codex", _exec("c1", f"python3 {SKILLS}/version-bump/scripts/bump.py --minor", "1.5.0"))
    probe = {"id": "no-bump", "must_use": "MCP:reltools", "must_not_use": "Skill:version-bump"}
    block = _conflict(trajectory, probe)
    assert "must_not_use Skill:version-bump was activated" in block["failures"][0]["detail"]


def test_one_read_of_several_skill_files_is_not_choosing_one_of_them() -> None:
    """check-19 e02 (Codex): the agent read three SKILL.md files at once and never computed a version."""
    paths = " ".join(f"{SKILLS}/{name}/SKILL.md" for name in ("changelog-collect", "release-notes", "version-bump"))
    trajectory = _trajectory(
        "codex",
        _exec("c1", f"cat {paths}", "---\nname: changelog-collect\n---\n---\nname: release-notes\n---\n"),
        _step("c2", "mcp__reltools__stage_release", {"project": "atlas"}, '{"staged": true}'),
    )
    probe = {
        "id": "given-tag-not-recomputed",
        "must_use": "Skill:release-notes",
        "must_not_use": ["MCP:reltools/compute_version", "Skill:version-bump"],
    }
    assert _conflict(trajectory, probe)["passed"] == 1
    # The `cp` shape of the same example is not a read at all.
    copied = _trajectory("codex", _exec("c1", f"cp {SKILLS}/version-bump/SKILL.md /tmp/vb.md"))
    copied["steps"].append(_exec("c2", f"cat {SKILLS}/release-notes/SKILL.md", "---\nname: release-notes\n---\n"))
    assert _conflict(copied, probe)["passed"] == 1


# ---------------------------------------------------------------------------
# Refs and arms (L23; check-19 e05)
# ---------------------------------------------------------------------------


def test_unprefixed_refs_mean_the_same_tool_on_both_harnesses() -> None:
    """check-19 e05: Codex web_search_call and exec_command answer to WebFetch and Bash."""
    claude = _trajectory(
        "claude-code",
        _step("t1", "Skill", {"skill": "remote-docs:cloudflare-docs"}, "Launching skill"),
        _step("t2", "mcp__plugin_remote-docs_deepwiki__ask_wiki_question", {"question": "alarms"}),
        _step("t3", "WebFetch", {"url": "https://example.com/alarms", "prompt": "alarms"}),
        _step("t4", "Bash", {"command": "ls /workspace"}),
    )
    codex = _trajectory(
        "codex",
        _exec("c1", f"sed -n '1,220p' {SKILLS}/cloudflare-docs/SKILL.md", "---\nname: cloudflare-docs\n---\n"),
        _step("c2", "mcp__deepwiki__ask_wiki_question", {"question": "alarms"}),
        {
            "source": "agent",
            "tool_calls": [
                {"tool_call_id": "", "function_name": "web_search_call", "arguments": {"action_type": "search"}}
            ],
        },
        _exec("c4", "ls /workspace"),
    )
    probes = (
        {"id": "no-web", "must_use": "MCP:deepwiki", "must_not_use": "WebFetch"},
        {"id": "no-shell", "must_use": "Skill:cloudflare-docs", "must_not_use": "Bash"},
    )
    for trajectory in (claude, codex):
        block = _conflict(trajectory, *probes, context=DOCS)
        assert (block["checked"], block["passed"]) == (2, 0), trajectory["agent"]


NOTES = "# cobalt 0.7.4\n\n## Bug Fixes\n\n- CB-0377: Handle empty config file\n"
STAGE_NO_EDIT = {"id": "stage-no-edit", "must_use": "MCP:reltools/stage_release", "must_not_use": ["Write", "Edit"]}


@pytest.mark.parametrize(
    "codex_write",
    [
        f"mkdir -p /workspace/out && cat > /workspace/out/RELEASE_NOTES.md <<'EOF'\n{NOTES}EOF",
        f"python3 - <<'PY'\nfrom pathlib import Path\nPath('out/RELEASE_NOTES.md').write_text({NOTES!r})\nPY",
        "sed -i 's/0.7.3/0.7.4/' RELEASE_NOTES.md",
    ],
    ids=["heredoc-redirect", "heredoc-python", "sed-in-place"],
)
def test_write_and_edit_refs_see_a_shell_write_on_both_harnesses(codex_write: str) -> None:
    """Live sk19 stage-no-edit: Claude wrote with ``Write``, Codex with ``exec_command``; both must be flagged."""
    claude = _trajectory(
        "claude-code",
        _step("t1", "mcp__plugin_release-kit_reltools__stage_release", {"project": "cobalt"}),
        _step("t2", "Write", {"file_path": "/workspace/RELEASE_NOTES.md", "content": NOTES}),
    )
    codex = _trajectory(
        "codex",
        _step("c1", "mcp__reltools__stage_release", {"project": "cobalt"}),
        _exec("c2", codex_write),
    )
    for trajectory in (claude, codex):
        block = _conflict(trajectory, STAGE_NO_EDIT)
        assert (block["checked"], block["passed"]) == (1, 0), trajectory["agent"]
        assert block["failures"][0]["detail"] == "must_not_use Write | Edit was activated"


def test_a_shell_call_that_writes_no_file_is_not_a_write() -> None:
    """Only real files count: ``2>/dev/null`` and a plain read are not writes."""
    codex = _trajectory(
        "codex",
        _step("c1", "mcp__reltools__stage_release", {"project": "cobalt"}),
        _exec("c2", "rg --files -g 'RELEASE_NOTES.md' /workspace 2>/dev/null"),
        _exec("c3", "cat RELEASE_NOTES.md > /dev/null && cat <<'EOF'\nnot a file\nEOF"),
    )
    block = _conflict(codex, STAGE_NO_EDIT)
    assert (block["checked"], block["passed"]) == (1, 1), block


def test_probes_the_arm_cannot_satisfy_are_skipped() -> None:
    trajectory = _trajectory("claude-code", _step("t1", "Skill", {"skill": "release-notes"}, "Launching skill"))
    block = _conflict(trajectory, STAGE_NOT_PUBLISH, arm=ARM_SUM_OF_PARTS)
    assert block["skipped"] == ["stage-not-publish"]
    assert (block["checked"], block["status"]) == (0, "not_applicable")


def test_reports_say_a_check_the_arm_skipped_was_skipped_not_unconfigured() -> None:
    """The member-skills arm stages no MCP server, so every MCP probe, edge, and handoff is skipped there."""
    trajectory = _trajectory("claude-code", _step("t1", "Skill", {"skill": "release-notes"}, "Launching skill"))
    case = {
        "id": "c",
        "conflict_probes": [STAGE_NOT_PUBLISH],
        "expected_order": [["Skill:release-notes", "MCP:reltools/stage_release"]],
        "handoffs": [{"producer": "MCP:reltools/list_changes", "consumer": "Skill:release-notes", "value": "AT-4"}],
    }
    signals = []
    for _ in range(2):
        result = compute_plugin_signals(
            trajectory, plugin_case_spec(case), declared=CONTEXT.declared_for(ARM_SUM_OF_PARTS)
        )
        assert result is not None
        signals.append(result)
    summary = summarize_plugin_signals(signals)
    assert [summary[key]["skipped"] for key in ("order", "handoff", "conflict")] == [2, 2, 2]
    view = signals_view({"agents": {"claude-code": {"plugin_signals_summary": {"sum_of_parts": summary}}}})
    assert view is not None
    skipped = "2 skipped (this arm cannot carry that component type)"
    assert {check["name"]: check["label"] for check in view["entries"][0]["checks"]} == {
        "Order": skipped,
        "Handoff": skipped,
        "Conflict": skipped,
    }


# ---------------------------------------------------------------------------
# Reports name the failed probes (L23 pooled rate)
# ---------------------------------------------------------------------------


def test_reports_name_the_probes_that_failed() -> None:
    probe = {
        "id": "stage-not-publish",
        "must_use": "MCP:reltools/stage_release",
        "must_not_use": "MCP:reltools/publish_release",
    }
    good = _trajectory("claude-code", _step("t1", "mcp__reltools__stage_release", {}, '{"staged": true}'))
    bad = _trajectory("claude-code", _step("t1", "mcp__reltools__publish_release", {}, '{"published": true}'))
    signals = []
    for trajectory in (good, bad, bad):
        result = compute_plugin_signals(
            trajectory,
            plugin_case_spec({"id": "c", "conflict_probes": [probe]}),
            declared=CONTEXT.declared_for(ARM_WITH_SKILL),
        )
        assert result is not None
        signals.append(result)
    summary = summarize_plugin_signals(signals)
    assert summary["conflict"]["failed_probes"] == [{"probe": "stage-not-publish", "trials": 2}]
    view = signals_view({"agents": {"claude-code": {"plugin_signals_summary": {"with_skill": summary}}}})
    assert view is not None

    html = (
        HTMLReporter()
        ._env.from_string('{% from "plugin_sections.html.j2" import signals_section %}{{ signals_section(sig) }}')
        .render(sig=view)
    )
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    assert "Conflict probes that failed: Probe Trials failed stage-not-publish 2" in text
    console = Console(file=StringIO(), width=200, color_system=None)
    print_plugin_tier3({"signals": view}, console)
    assert "failed probe: stage-not-publish (2 trial(s))" in " ".join(console.file.getvalue().split())
