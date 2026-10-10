# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component routing (check 15) and tool selection (check 22) (proof M26, L19, L23 refs, L26).

The trajectories copy the ATIF shapes of the saved Claude Code and Codex runs:
Claude names MCP tools ``mcp__[plugin_<plugin>_]<server>__<tool>``, Codex keeps
the bare tool name (its session log names the server) and records hosted web
search as ``web_search_call``.
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
    subagents=["release-reviewer"],
    commands=["release-check"],
)
DOCS = build_plugin_signals_context(
    member_skills=["repo-docs", "cloudflare-docs"],
    mcp_servers=["deepwiki", "cfdocs"],
    wrapper_skills=["remote-docs-plugin-eval", "remote-docs"],
)


def _step(call_id: str, fn: str, args: dict[str, Any], content: str | None = "ok") -> dict[str, Any]:
    step: dict[str, Any] = {
        "source": "agent",
        "tool_calls": [{"tool_call_id": call_id, "function_name": fn, "arguments": args}],
    }
    if content is not None:
        step["observation"] = {"results": [{"source_call_id": call_id, "content": content}]}
    return step


def _trajectory(agent: str, *steps: dict[str, Any]) -> dict[str, Any]:
    return {"agent": {"name": agent}, "steps": [{"source": "user", "message": "Do the task."}, *steps]}


def _codex_read(call_id: str, member: str) -> dict[str, Any]:
    return _step(
        call_id,
        "exec_command",
        {"cmd": f"cat /tmp/agent-home/.agents/skills/{member}/SKILL.md", "workdir": "/workspace"},
        f"Process exited with code 0\nOutput:\n---\nname: {member}\n---\n",
    )


def _signals(
    trajectory: dict[str, Any],
    case: dict[str, Any],
    *,
    context: Any = CONTEXT,
    arm: str = ARM_WITH_SKILL,
    servers: dict[str, str] | None = None,
) -> dict[str, Any]:
    signals = compute_plugin_signals(
        trajectory,
        plugin_case_spec({"id": "c", **case}),
        declared=context.declared_for(arm),
        wrapper_skills=context.wrapper_skills,
        mcp_call_servers=servers,
    )
    assert signals is not None
    return signals


def _brief(block: dict[str, Any]) -> tuple[Any, ...]:
    return block["precision"], block["recall"], block["decoy_calls"]


# ---------------------------------------------------------------------------
# Checks 15 and 22 are separate numbers (M26)
# ---------------------------------------------------------------------------


def test_mcp_calls_do_not_dilute_skill_routing_and_skills_do_not_dilute_mcp_selection() -> None:
    """check-15 e06 (skill-only refs) and check-22 l02 (MCP-only refs) on one trace."""
    trajectory = _trajectory(
        "claude-code",
        _step("t1", "Skill", {"skill": "remote-docs:repo-docs"}, "Launching skill: remote-docs:repo-docs"),
        _step("t2", "mcp__plugin_remote-docs_deepwiki__ask_wiki_question", {"question": "q"}),
        _step("t3", "Skill", {"skill": "remote-docs:cloudflare-docs"}, "Launching skill: remote-docs:cloudflare-docs"),
    )

    skills_only = _signals(trajectory, {"expected_tools": ["Skill:repo-docs"]}, context=DOCS)
    assert skills_only["routing"]["called"] == ["Skill:repo-docs", "Skill:cloudflare-docs"]
    assert _brief(skills_only["routing"]) == (0.5, 1.0, 0)
    assert skills_only["tool_selection"]["status"] == "not_applicable"

    mcp_only = _signals(trajectory, {"expected_tools": ["MCP:deepwiki/ask_wiki_question"]}, context=DOCS)
    assert mcp_only["tool_selection"]["called"] == ["mcp__deepwiki__ask_wiki_question"]
    assert _brief(mcp_only["tool_selection"]) == (1.0, 1.0, 0)
    assert mcp_only["routing"]["status"] == "not_applicable"

    summary = summarize_plugin_signals([skills_only, mcp_only])
    assert summary["routing"]["n_scored"] == 1
    assert summary["tool_selection"]["n_scored"] == 1


def test_codex_hosted_web_search_is_a_web_search_decoy() -> None:
    """check-15 e09 / check-22 e06: Codex ``web_search_call`` and Claude ``WebSearch`` count the same."""
    case = {"expected_tools": ["Skill:repo-docs", "MCP:deepwiki/ask_wiki_question"], "decoy_tools": ["WebSearch"]}
    codex = _trajectory(
        "codex",
        _codex_read("c1", "repo-docs"),
        _step("", "web_search_call", {"action_type": "search", "query": "flask app context"}, None),
        _step("c3", "ask_wiki_question", {"question": "app context"}, "Output:\nThe app context..."),
    )
    claude = _trajectory(
        "claude-code",
        _step("t1", "Skill", {"skill": "repo-docs"}, "Launching skill: repo-docs"),
        _step("t2", "WebSearch", {"query": "flask app context"}, "results"),
        _step("t3", "mcp__deepwiki__ask_wiki_question", {"question": "app context"}, "The app context..."),
    )

    for trajectory, servers in ((codex, {"c3": "deepwiki"}), (claude, None)):
        selection = _signals(trajectory, case, context=DOCS, servers=servers)["tool_selection"]
        assert selection["decoy_calls"] == 1, trajectory["agent"]
        assert selection["precision"] == 0.5, trajectory["agent"]
        # A search is not a ``WebFetch`` decoy on either harness.
        fetch = _signals(trajectory, {"decoy_tools": ["WebFetch"]}, context=DOCS, servers=servers)
        assert fetch["tool_selection"]["decoy_calls"] == 0, trajectory["agent"]


def test_a_codex_web_call_answers_to_the_web_tool_its_action_names() -> None:
    """A Codex page open is a right ``WebFetch``, not a ``WebSearch`` decoy, as Claude Code's ``WebFetch`` is."""
    url = "https://flask.palletsprojects.com/en/stable/appcontext/"
    fetches = (
        _trajectory("codex", _step("", "web_search_call", {"action_type": "open_page", "url": url}, None)),
        _trajectory("codex", _step("", "web_search_call", {"action_type": "find_in_page", "url": url}, None)),
        _trajectory("claude-code", _step("t1", "WebFetch", {"url": url, "prompt": "app context"})),
    )
    searches = (
        _trajectory("codex", _step("", "web_search_call", {"action_type": "search", "query": "flask"}, None)),
        _trajectory("claude-code", _step("t1", "WebSearch", {"query": "flask"})),
    )
    for trajectories, right, wrong in ((fetches, "WebFetch", "WebSearch"), (searches, "WebSearch", "WebFetch")):
        for trajectory in trajectories:
            chosen = _signals(trajectory, {"expected_tools": [right], "decoy_tools": [wrong]}, context=DOCS)
            assert _brief(chosen["tool_selection"]) == (1.0, 1.0, 0), (trajectory["agent"], right)
            # The same call is still the decoy when the case names it so.
            decoyed = _signals(trajectory, {"expected_tools": [wrong], "decoy_tools": [right]}, context=DOCS)
            assert _brief(decoyed["tool_selection"]) == (0.0, 0.0, 1), (trajectory["agent"], right)
    # A call that records no action could be either, so it still answers to both.
    unknown = _trajectory("codex", _step("", "web_search_call", {"action_type": ""}, None))
    for decoy in ("WebSearch", "WebFetch"):
        assert _signals(unknown, {"decoy_tools": [decoy]}, context=DOCS)["tool_selection"]["decoy_calls"] == 1, decoy


def test_a_decoy_write_through_the_shell_is_not_a_precise_choice() -> None:
    """An acceptable ``Bash`` does not make a shell call that is also a ``Write`` decoy a right choice."""
    case = {"expected_tools": ["Skill:release-notes"], "acceptable_tools": ["Bash", "Read"], "decoy_tools": ["Write"]}
    codex = _trajectory("codex", _step("c1", "exec_command", {"cmd": "cat > /workspace/x.md <<'EOF'\nx\nEOF"}))
    claude = _trajectory("claude-code", _step("t1", "Write", {"file_path": "/workspace/x.md", "content": "x"}))
    for trajectory in (codex, claude):
        selection = _signals(trajectory, case)["tool_selection"]
        assert (selection["precision"], selection["decoy_calls"]) == (0.0, 1), trajectory["agent"]


@pytest.mark.parametrize(
    ("ref", "codex_call", "claude_call"),
    [
        ("Bash", ("exec_command", {"cmd": "ls /workspace"}), ("Bash", {"command": "ls /workspace"})),
        ("Write", ("apply_patch", {"input": "*** Begin Patch\n*** Add File: a.txt\n+x\n*** End Patch"}), None),
        ("exec_command", None, ("Bash", {"command": "ls"})),
        # Codex writes most files through the shell (sk19 live run); so may Claude Code.
        (
            "Write",
            ("exec_command", {"cmd": "cat > out/a.txt <<'EOF'\nx\nEOF"}),
            ("Bash", {"command": "python3 - <<'PY'\nopen('out/a.txt', 'w').write('x')\nPY"}),
        ),
        ("Edit", ("exec_command", {"cmd": "sed -i 's/a/b/' out/a.txt"}), ("Bash", {"command": "echo y >> out/a.txt"})),
    ],
)
def test_unprefixed_tool_refs_mean_the_same_tool_on_both_harnesses(ref: str, codex_call: Any, claude_call: Any) -> None:
    for agent, call in (("codex", codex_call), ("claude-code", claude_call)):
        if call is None:
            continue
        signals = _signals(_trajectory(agent, _step("x1", call[0], call[1])), {"decoy_tools": [ref]})
        assert signals["tool_selection"]["decoy_calls"] == 1, (agent, ref)


def test_a_later_shell_write_counts_for_write_after_a_plain_shell_call() -> None:
    """Every shell call has one label; the write that comes second still answers to ``Write``."""
    case = {"expected_tools": ["Bash", "Write"]}
    codex = _trajectory(
        "codex",
        _step("c1", "exec_command", {"cmd": "ls /workspace"}),
        _step("c2", "exec_command", {"cmd": "cat > out/notes.md <<'EOF'\n# notes\nEOF"}),
    )
    claude = _trajectory(
        "claude-code",
        _step("t1", "Bash", {"command": "ls /workspace"}),
        _step("t2", "Bash", {"command": "echo '# notes' > out/notes.md"}),
    )
    for trajectory in (codex, claude):
        selection = _signals(trajectory, case)["tool_selection"]
        assert selection["recall"] == 1.0, trajectory["agent"]
        assert selection["precision"] == 1.0, trajectory["agent"]


@pytest.mark.parametrize(
    ("cmd", "recall"),
    [
        (
            "apply_patch <<'PATCH'\n*** Begin Patch\n*** Update File: /workspace/a.py\n@@\n-x\n+y\n*** End Patch\nPATCH",
            1.0,
        ),
        ("cd /workspace && apply_patch <<'EOF'\n*** Begin Patch\n*** Add File: out/a.txt\n+x\n*** End Patch\nEOF", 1.0),
        ("cat > out/a.txt <<'EOF'\nx\nEOF", 0.0),
    ],
)
def test_apply_patch_matches_a_shell_apply_patch_but_no_other_shell_write(cmd: str, recall: float) -> None:
    """Codex runs ``apply_patch`` as a shell here-document; ``apply_patch`` still means the file tools only."""
    trajectory = _trajectory("codex", _step("c1", "exec_command", {"cmd": cmd, "workdir": "/workspace"}))
    assert _signals(trajectory, {"expected_tools": ["apply_patch"]})["tool_selection"]["recall"] == recall


# ---------------------------------------------------------------------------
# What counts as a right choice
# ---------------------------------------------------------------------------


def test_a_tool_that_does_not_exist_is_never_a_correct_selection() -> None:
    """check-22 e05: a made-up tool on a real server, and a server the arm does not have."""
    missing = "<tool_use_error>Error: No such tool available: mcp__reltools__check_release</tool_use_error>"
    made_up = _trajectory("claude-code", _step("t1", "mcp__reltools__check_release", {"project": "cobalt"}, missing))
    selection = _signals(made_up, {"expected_tools": ["MCP:reltools"]})["tool_selection"]
    assert selection["called"] == ["mcp__reltools__check_release"]
    assert (selection["precision"], selection["recall"]) == (0.0, 0.0)

    no_server = _trajectory(
        "claude-code",
        _step("t1", "Skill", {"skill": "changelog-collect"}, "Launching skill: changelog-collect"),
        _step(
            "t2",
            "mcp__reltools__list_changes",
            {"project": "borealis"},
            "<tool_use_error>Error: No such tool available: mcp__reltools__list_changes</tool_use_error>",
        ),
    )
    case = {
        "expected_tools": ["Skill:changelog-collect", "MCP:reltools/list_changes"],
        "acceptable_tools": ["MCP:reltools/compute_version"],
    }
    signals = _signals(no_server, case, arm=ARM_SUM_OF_PARTS)
    assert _brief(signals["routing"]) == (1.0, 1.0, 0)
    tools = signals["tool_selection"]
    # The sum-of-parts arm has no MCP server: the MCP ref is skipped, the hallucinated call is a wrong choice.
    assert tools["skipped"] == ["MCP:reltools/list_changes"]
    assert (tools["precision"], tools["recall"]) == (0.0, None)


def test_unknown_skill_load_is_not_a_routing_hit() -> None:
    trajectory = _trajectory(
        "claude-code",
        _step(
            "t1", "Skill", {"skill": "release-kit:version-bump"}, "<tool_use_error>Unknown skill: x</tool_use_error>"
        ),
    )
    routing = _signals(trajectory, {"expected_tools": ["Skill:version-bump"]})["routing"]
    assert (routing["precision"], routing["recall"]) == (0.0, 0.0)


def test_untyped_bare_tool_names_match_mcp_calls_on_both_harnesses() -> None:
    """check-22 e08: ``list_changes`` expected and ``publish_release`` decoy, as the docs promise."""
    case = {"expected_tools": ["Skill:changelog-collect", "list_changes"], "decoy_tools": ["publish_release"]}
    claude = _trajectory(
        "claude-code",
        _step("t1", "Skill", {"skill": "changelog-collect"}, "Launching skill: changelog-collect"),
        _step("t2", "mcp__reltools__list_changes", {"project": "cobalt"}),
        _step("t3", "mcp__reltools__publish_release", {"project": "cobalt"}),
    )
    codex = _trajectory(
        "codex",
        _codex_read("c1", "changelog-collect"),
        _step("c2", "list_changes", {"project": "cobalt"}),
        _step("c3", "publish_release", {"project": "cobalt"}),
    )
    for trajectory, servers in ((claude, None), (codex, {"c2": "reltools", "c3": "reltools"})):
        signals = _signals(trajectory, case, servers=servers)
        assert _brief(signals["tool_selection"]) == (0.5, 1.0, 1), trajectory["agent"]
        assert _brief(signals["routing"]) == (1.0, 1.0, 0), trajectory["agent"]


def test_claude_server_spelling_matches_with_a_tool_too() -> None:
    """check-22 e04: declared ``docs.v2`` is ``docs_v2`` in Claude's tool name."""
    context = build_plugin_signals_context(mcp_servers=["docs.v2", "kb"])
    for fn in ("mcp__docs_v2__search", "mcp__plugin_my_plugin_docs_v2__search"):
        trajectory = _trajectory("claude-code", _step("t1", fn, {"q": "x"}))
        for ref in ("MCP:docs.v2/search", "MCP:docs_v2/search", "MCP:docs_v2"):
            selection = _signals(trajectory, {"expected_tools": [ref]}, context=context)["tool_selection"]
            assert selection["recall"] == 1.0, (fn, ref)


def test_a_plugin_name_glob_never_credits_the_generated_wrapper() -> None:
    """check-15 e02: ``Skill:release-*`` must not be satisfied by loading ``release-kit-plugin-eval``."""
    trajectory = _trajectory(
        "claude-code",
        _step("t1", "Skill", {"skill": "release-kit-plugin-eval"}, "Launching skill: release-kit-plugin-eval"),
        _step("t2", "mcp__reltools__list_changes", {"project": "atlas"}),
    )
    routing = _signals(trajectory, {"expected_tools": ["Skill:release-*"]})["routing"]
    assert routing["called"] == []
    assert routing["recall"] == 0.0


def test_precision_is_undefined_when_nothing_was_called() -> None:
    """check-15 e10: a miss is not a wrong choice."""
    trajectory = _trajectory("codex")
    signals = _signals(trajectory, {"expected_tools": ["Skill:changelog-collect", "MCP:reltools/list_changes"]})
    for key in ("routing", "tool_selection"):
        assert (signals[key]["precision"], signals[key]["recall"], signals[key]["f1"]) == (None, 0.0, 0.0), key


def test_untyped_refs_do_not_match_a_component_through_the_tool_that_carried_it() -> None:
    """A ``Read`` of a SKILL.md is the skill (routing) plus a separate ``Read`` tool identity."""
    trajectory = _trajectory(
        "claude-code",
        _step("t1", "Read", {"file_path": "/workspace/.claude/skills/version-bump/SKILL.md"}, "# version-bump"),
    )
    signals = _signals(trajectory, {"expected_tools": ["Skill:changelog-collect"], "acceptable_tools": ["Read"]})
    assert signals["routing"]["called"] == ["Skill:version-bump"]
    assert signals["routing"]["precision"] == 0.0
    assert signals["tool_selection"]["called"] == ["Read"]
    assert signals["tool_selection"]["precision"] == 1.0


def test_codex_cannot_have_plugin_agents_or_commands_so_those_refs_are_skipped() -> None:
    trajectory = _trajectory("codex", _codex_read("c1", "release-notes"))
    routing = _signals(
        trajectory, {"expected_tools": ["Skill:release-notes", "Agent:release-reviewer", "Command:release-check"]}
    )["routing"]
    assert routing["skipped"] == ["Agent:release-reviewer", "Command:release-check"]
    assert routing["recall"] == 1.0


def test_untyped_refs_to_components_the_member_skills_arm_does_not_stage_are_skipped_like_typed_ones() -> None:
    """The sum-of-parts arm stages no subagents or commands, so ``release-check`` is still a command ref there."""
    trajectory = _trajectory("claude-code", _step("t1", "Skill", {"skill": "release-notes"}, "Launching skill"))
    results = []
    for refs in (["Agent:release-reviewer", "Command:release-check"], ["release-reviewer", "release-check"]):
        case = {
            "expected_tools": ["Skill:release-notes", *refs],
            "expected_order": [["Skill:release-notes", refs[1]]],
            "conflict_probes": [{"id": "p1", "must_use": refs[0], "must_not_use": "WebSearch"}],
        }
        signals = _signals(trajectory, case, arm=ARM_SUM_OF_PARTS)
        routing, tools = signals["routing"], signals["tool_selection"]
        assert routing["skipped"] == refs
        assert tools["expected"] == []
        results.append((routing["recall"], tools["status"], signals["order"]["edges"], signals["conflict"]["skipped"]))
    assert results == [(1.0, "not_applicable", 0, ["p1"])] * 2


# ---------------------------------------------------------------------------
# Dataset validation (L26)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "decoy"),
    [
        (["MCP:reltools/*"], ["MCP:reltools/publish_release"]),
        (["MCP:reltools/list_changes"], ["MCP:reltools"]),
        (["WebSearch"], ["websearch"]),
    ],
)
def test_overlapping_expected_and_decoy_refs_are_rejected(expected: list[str], decoy: list[str]) -> None:
    problems = validate_plugin_case_fields({"id": "c", "expected_tools": expected, "decoy_tools": decoy})
    assert any(problem.startswith("decoy_tools[0]: overlaps expected_tools") for problem in problems), problems


def test_disjoint_expected_and_decoy_refs_are_accepted() -> None:
    entry = {"id": "c", "expected_tools": ["MCP:reltools/list_changes"], "decoy_tools": ["MCP:reltools/publish_*"]}
    assert validate_plugin_case_fields(entry) == []


# ---------------------------------------------------------------------------
# Reports (L19 decoy-rate label)
# ---------------------------------------------------------------------------


def _summary_view() -> dict[str, Any]:
    case = {"expected_tools": ["Skill:changelog-collect", "MCP:reltools/list_changes"], "decoy_tools": ["WebSearch"]}
    with_decoy = _trajectory(
        "claude-code",
        _step("t1", "Skill", {"skill": "changelog-collect"}, "Launching skill: changelog-collect"),
        _step("t2", "WebSearch", {"query": "q"}),
        _step("t3", "WebSearch", {"query": "q2"}),
        _step("t4", "mcp__reltools__list_changes", {"project": "atlas"}),
    )
    clean = _trajectory("claude-code", _step("t4", "mcp__reltools__list_changes", {"project": "atlas"}))
    summary = summarize_plugin_signals([_signals(with_decoy, case), _signals(clean, case)])
    view = signals_view({"agents": {"claude-code": {"plugin_signals_summary": {"with_skill": summary}}}})
    assert view is not None
    return view


def test_html_and_cli_show_routing_and_tool_selection_with_a_per_trial_decoy_rate() -> None:
    view = _summary_view()
    environment = HTMLReporter()._env
    html = environment.from_string(
        '{% from "plugin_sections.html.j2" import signals_section %}{{ signals_section(sig) }}'
    ).render(sig=view)
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    assert "Component routing (precision / recall / F1) 100% / 50% / 50% 0 decoy call(s), in 0% of trials" in text
    assert "Tool selection (precision / recall / F1) 75% / 100% / 83% 2 decoy call(s), in 50% of trials" in text
    assert "of calls" not in text

    console = Console(file=StringIO(), width=200, color_system=None)
    print_plugin_tier3({"signals": view}, console)
    plain = " ".join(console.file.getvalue().split())
    assert "component routing P/R/F1 100% / 50% / 50%, 0 decoy call(s), in 0% of trials" in plain
    assert "tool selection P/R/F1 75% / 100% / 83%, 2 decoy call(s), in 50% of trials" in plain
