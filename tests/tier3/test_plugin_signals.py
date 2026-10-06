# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only plugin component signals computed from ATIF trajectories."""

from __future__ import annotations

import json
from typing import Any

import pytest

from skillevaluator.tier3.eval_core import checks, plugin_signals
from skillevaluator.tier3.eval_core.plugin_signals import (
    MAX_TOOL_PATTERNS,
    build_plugin_signals_context,
    compute_plugin_signals,
    match_declared_mcp_server,
    plugin_case_spec,
    summarize_plugin_signals,
    validate_plugin_case_fields,
)

DECLARED = {"skill": ["alpha", "beta"], "mcp": ["github", "jira"]}


def _tc(fn: str, args: dict[str, Any] | None = None, call_id: str = "c1") -> dict[str, Any]:
    return {"tool_call_id": call_id, "function_name": fn, "arguments": args or {}}


def _res(call_id: str, content: Any, **extra: Any) -> dict[str, Any]:
    return {"source_call_id": call_id, "content": content, **extra}


def _step(tool_calls: list[dict[str, Any]], results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"source": "agent", "tool_calls": tool_calls, "observation": {"results": results or []}}


def _traj(*steps: dict[str, Any], prompt: str = "Please help.") -> dict[str, Any]:
    return {"schema_version": "ATIF-v1.2", "steps": [{"source": "user", "message": prompt}, *steps]}


def _one(fn: str, args: dict[str, Any] | None = None, content: Any = "ok", call_id: str = "c1", **extra: Any):
    return _step([_tc(fn, args, call_id)], [_res(call_id, content, **extra)])


def _signals(traj: dict[str, Any], case: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("declared", DECLARED)
    signals = compute_plugin_signals(traj, case or {}, **kwargs)
    assert signals is not None
    return signals


def _activations(traj: dict[str, Any]) -> list[dict[str, Any]]:
    return _signals(traj)["activations"]


def _record_regex_timeouts(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the deadline of every dataset-pattern search (a search without one fails)."""
    timeouts: list[float] = []
    search = plugin_signals.regex.search

    def search_with_deadline(pattern: str, text: str, *, timeout: float) -> Any:
        timeouts.append(timeout)
        return search(pattern, text, timeout=timeout)

    monkeypatch.setattr(plugin_signals.regex, "search", search_with_deadline)
    return timeouts


def _codex_exec(source: str, call_id: str = "exec-1") -> dict[str, Any]:
    return {"tool_call_id": call_id, "function_name": "exec", "arguments": {"input": source}}


# ---------------------------------------------------------------------------
# Component activation classifier
# ---------------------------------------------------------------------------


class TestClassifierClaudeStyle:
    def test_skill_mcp_subagent_and_command_keep_identity_order_and_outcome(self) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}, "Launching skill: alpha"),
            _one("mcp__github__list_issues", {"repo": "o/r"}, [{"type": "text", "text": "[]"}], call_id="c2"),
            _one("Task", {"subagent_type": "reviewer", "prompt": "review"}, "looks good", call_id="c3"),
            _one("SlashCommand", {"command": "/deploy prod"}, "deployed", call_id="c4"),
            _one("Read", {"file_path": "/workspace/data.csv"}, "a,b", call_id="c5"),
        )

        activations = _activations(traj)

        assert activations == [
            {"type": "skill", "name": "alpha", "tool": "Skill", "server": None, "step_index": 1, "succeeded": True},
            {
                "type": "mcp",
                "name": "github",
                "tool": "mcp__github__list_issues",
                "server": "github",
                "step_index": 2,
                "succeeded": True,
            },
            {
                "type": "subagent",
                "name": "reviewer",
                "tool": "Task",
                "server": None,
                "step_index": 3,
                "succeeded": True,
            },
            {
                "type": "command",
                "name": "deploy",
                "tool": "SlashCommand",
                "server": None,
                "step_index": 4,
                "succeeded": True,
            },
        ]

    def test_read_of_declared_member_manifest_counts_as_skill_activation(self) -> None:
        traj = _traj(
            _one("Read", {"file_path": "/workspace/.claude/skills/beta/SKILL.md"}, "# beta"),
            _one("Read", {"file_path": "/workspace/.claude/skills/other/SKILL.md"}, "# other", call_id="c2"),
            _one("Read", {"file_path": "/workspace/skills/beta/SKILL.md.bak"}, "# old", call_id="c3"),
        )

        activations = _activations(traj)

        assert [(a["type"], a["name"], a["tool"]) for a in activations] == [("skill", "beta", "Read:skill-md-read")]

    def test_ordinary_tools_are_not_activations(self) -> None:
        traj = _traj(_one("Bash", {"command": "ls"}), _one("Write", {"file_path": "x"}, call_id="c2"))
        assert _activations(traj) == []

    def test_mcp_filesystem_read_of_member_manifest_is_both_mcp_and_skill(self) -> None:
        traj = _traj(_one("mcp__filesystem__read_file", {"path": "/workspace/skills/alpha/SKILL.md"}, "# alpha"))

        activations = _activations(traj)

        assert [(a["type"], a["name"]) for a in activations] == [("mcp", "filesystem"), ("skill", "alpha")]


class TestClassifierCodexStyle:
    def test_shell_read_of_declared_manifest_counts(self) -> None:
        traj = _traj(
            _one("exec_command", {"cmd": ["bash", "-lc", "cd /workspace && sed -n '1,80p' skills/alpha/SKILL.md"]}),
            _one("shell", {"command": "FOO=1 cat skills/beta/SKILL.md | head"}, call_id="c2"),
        )

        activations = _activations(traj)

        assert [(a["name"], a["tool"]) for a in activations] == [
            ("alpha", "exec_command:skill-md-read"),
            ("beta", "shell:skill-md-read"),
        ]

    @pytest.mark.parametrize(
        "command",
        [
            "rm skills/alpha/SKILL.md",
            "ls skills/alpha/SKILL.md",
            "grep name skills/alpha/SKILL.md",
            "printf skills/alpha/SKILL.md",
            "cat skills/alpha/SKILL.md.bak",
            "cat skills/wrapper/SKILL.md",
            "echo hi > skills/alpha/SKILL.md",
        ],
    )
    def test_non_read_or_undeclared_shell_mentions_do_not_count(self, command: str) -> None:
        traj = _traj(_one("exec_command", {"cmd": command}))
        assert _activations(traj) == []

    @pytest.mark.parametrize(
        "command",
        [
            # The apostrophe sits in a here-document body, so the script is valid bash.
            "ls skills\ncat skills/alpha/SKILL.md\ncat > notes.md <<'EOF'\nIt's fine\nEOF",
            # A genuinely unbalanced quote on a later line must not merge the lines.
            "cd /workspace\nsed -n '1,120p' skills/alpha/SKILL.md\necho Don't forget",
        ],
    )
    def test_multiline_script_with_a_stray_quote_still_credits_the_read(self, command: str) -> None:
        (activation,) = _activations(_traj(_one("exec_command", {"cmd": command})))
        assert (activation["type"], activation["name"]) == ("skill", "alpha")

    def test_heredoc_body_is_data_not_a_manifest_read(self) -> None:
        command = "cat > notes.md <<EOF\ncat skills/alpha/SKILL.md\nEOF\nls"
        assert _activations(_traj(_one("exec_command", {"cmd": command}))) == []

    def test_native_exec_wrapper_is_normalized_and_mapped_observation_is_used(self) -> None:
        source = 'const r = await tools.mcp__github__list_issues({repo: "o/r"});\ntext(JSON.stringify(r));'
        traj = _traj(_step([_codex_exec(source)], [_res("exec-1", "[1, 2]")]))

        (activation,) = _activations(traj)

        assert activation["tool"] == "mcp__github__list_issues"
        assert activation["succeeded"] is True

    def test_native_exec_inner_calls_without_proven_observation_are_unknown(self) -> None:
        source = (
            'const a = await tools.mcp__github__list_issues({repo: "o/r"});\n'
            'const b = await tools.mcp__jira__create_ticket({title: "x"});\n'
            "text(JSON.stringify(a));\ntext(JSON.stringify(b));"
        )
        traj = _traj(_step([_codex_exec(source)], [_res("exec-1", "403: forbidden")]))

        activations = _activations(traj)

        assert [(a["tool"], a["succeeded"]) for a in activations] == [
            ("mcp__github__list_issues", None),
            ("mcp__jira__create_ticket", None),
        ]

    @pytest.mark.parametrize(
        "fn", ["mcp__fs__get_document", "fs__get_document", "fs.get_document", "mcp_fs_get_document", "fs_get_document"]
    )
    def test_every_spelling_of_a_declared_server_read_credits_the_member(self, fn: str) -> None:
        traj = _traj(_one(fn, {"path": "/workspace/skills/alpha/SKILL.md"}, "# alpha"))

        activations = _signals(traj, declared={**DECLARED, "mcp": ["fs"]})["activations"]

        assert [(a["type"], a["name"], a["tool"]) for a in activations] == [
            ("mcp", "fs", "mcp__fs__get_document"),
            ("skill", "alpha", f"{fn}:skill-md-read"),
        ]

    @pytest.mark.parametrize("fn", ["mcp__fs__bash", "fs__bash"])
    def test_mcp_tool_named_like_a_shell_is_not_parsed_as_one(self, fn: str) -> None:
        traj = _traj(_one(fn, {"command": "cat skills/alpha/SKILL.md"}))

        activations = _signals(traj, declared={**DECLARED, "mcp": ["fs"]})["activations"]

        assert [(a["type"], a["name"]) for a in activations] == [("mcp", "fs")]

    @pytest.mark.parametrize("fn", ["mcp__fs__edit_file", "mcp_fs_edit_file"])
    def test_every_spelling_of_an_mcp_edit_tool_writes_the_artifact(self, fn: str) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one(fn, {"path": "out/report.json", "edits": []}, call_id="c2"),
            _one("Skill", {"skill": "beta"}, call_id="c3"),
            _one("Read", {"file_path": "out/report.json"}, call_id="c4"),
        )
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "artifact": "out/report.json"}]}

        signals = _signals(traj, case, declared={**DECLARED, "mcp": ["fs"]})

        assert signals["handoff"]["passed"] == 1

    @pytest.mark.parametrize(
        ("agent", "fn", "key"),
        [
            ("claude-code", "mcp__fs__write_file", "destination"),
            ("hermes", "mcp_fs_write_file", "destination"),
            ("hermes", "mcp_fs_save_file", "output"),
            ("opencode", "fs_write_file", "destination"),
            ("opencode", "fs_write_file", "file"),
            ("opencode", "fs_edit_file", "target"),
        ],
    )
    def test_mcp_write_tools_without_a_path_key_still_write_the_artifact(self, agent: str, fn: str, key: str) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one(fn, {key: "out/report.json", "content": "{}"}, call_id="c2"),
            _one("Skill", {"skill": "beta"}, call_id="c3"),
            _one("Read", {"file_path": "out/report.json"}, call_id="c4"),
        )
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "artifact": "out/report.json"}]}

        signals = _signals({**traj, "agent": {"name": agent}}, case, declared={**DECLARED, "mcp": ["fs"]})

        assert (signals["handoff"]["passed"], signals["handoff"]["failures"]) == (1, [])

    def test_an_mcp_editor_view_does_not_write_the_artifact(self) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one("mcp__fs__str_replace_editor", {"command": "view", "target": "out/report.json"}, call_id="c2"),
            _one("Skill", {"skill": "beta"}, call_id="c3"),
            _one("Read", {"file_path": "out/report.json"}, call_id="c4"),
        )
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "artifact": "out/report.json"}]}

        signals = _signals(traj, case, declared={**DECLARED, "mcp": ["fs"]})

        assert [failure["detail"] for failure in signals["handoff"]["failures"]] == [
            "artifact was not written by the producer"
        ]

    def test_declared_server_alternate_spellings_are_canonicalized(self) -> None:
        traj = _traj(
            _one("github.search_code", {"q": "x"}),
            _one("jira__create_ticket", {"title": "x"}, call_id="c2"),
            _one("slack.post", {"text": "x"}, call_id="c3"),
        )

        activations = _activations(traj)

        assert [a["tool"] for a in activations] == ["mcp__github__search_code", "mcp__jira__create_ticket"]


class TestMcpServerNames:
    @pytest.mark.parametrize(
        ("observed", "declared", "expected"),
        [
            ("GitHub", ["github", "jira"], "github"),
            ("my_docs", ["my.docs", "my_docs"], "my_docs"),  # an exact name beats another server's spelling
            ("my_docs", ["my.docs"], "my.docs"),
            ("my_docs", ["my.docs", "my-docs"], None),  # a spelling two servers share credits neither
            ("plugin_demo-plugin_docs", ["docs"], "docs"),
            ("plugin_demo_my_docs", ["my.docs"], "my.docs"),
            ("plugin_a_b_team_docs", ["team_docs", "docs"], "team_docs"),  # the longest declared suffix
            ("plugin_x_y_my_docs", ["my.docs", "my-docs"], None),
            ("plugin_", ["docs"], None),
            ("slack", ["github"], None),
            ("github", [], None),
        ],
    )
    def test_observed_server_names_map_to_one_declared_server(
        self, observed: str, declared: list[str], expected: str | None
    ) -> None:
        assert match_declared_mcp_server(observed, declared) == expected

    @pytest.mark.parametrize(
        ("agent", "fn", "servers", "tool"),
        [
            ("claude-code", "my.docs__search", ["my.docs", "my"], "mcp__my.docs__search"),  # the longest prefix wins
            ("claude-code", "my_docs__search", ["my.docs", "my-docs"], "mcp__my_docs__search"),
            ("hermes", "mcp_my_docs_search", ["my.docs", "my-docs"], "mcp__my_docs__search"),
            ("opencode", "github_team.list", ["github", "github_team"], "mcp__github_team__list"),
            ("opencode", "docs_search", ["docs"], "mcp__docs__search"),
            ("codex", "docs_search", ["docs"], None),  # only OpenCode names MCP tools <server>_<tool>
            ("opencode", "web_search", ["web"], None),  # a built-in tool, not the web server's
            ("claude-code", "mcp__plugin_demo_my_docs__get", ["my.docs"], "mcp__my.docs__get"),
        ],
    )
    def test_tool_names_map_to_declared_servers(
        self, agent: str, fn: str, servers: list[str], tool: str | None
    ) -> None:
        traj = {**_traj(_one(fn)), "agent": {"name": agent}}

        activations = _signals(traj, declared={"skill": [], "mcp": servers})["activations"]

        assert [a["tool"] for a in activations if a["type"] == "mcp"] == ([tool] if tool else [])


class TestShellReadsAgainstTheScoredCheck:
    """The report-only SKILL.md read detector is deliberately more lenient than the scored check."""

    @staticmethod
    def _credited(command: str) -> list[str]:
        activations = _signals(_traj(_one("exec_command", {"cmd": command})))["activations"]
        return [a["name"] for a in activations if a["type"] == "skill"]

    @pytest.mark.parametrize("verb", sorted(checks._FILE_READ_VERBS))
    def test_every_scored_reader_verb_credits_the_member(self, verb: str) -> None:
        command = f"{verb} skills/alpha/SKILL.md"
        assert checks._cmd_reads_skill_md(command)
        assert self._credited(command) == ["alpha"]

    @pytest.mark.parametrize(
        "command",
        ["sudo cat skills/alpha/SKILL.md", "bash -lc 'cat skills/alpha/SKILL.md'", "tac skills/alpha/SKILL.md"],
    )
    def test_lenient_reads_credit_the_member_but_not_the_score(self, command: str) -> None:
        assert not checks._cmd_reads_skill_md(command)
        assert self._credited(command) == ["alpha"]

    def test_shell_variables_are_expanded_only_by_the_scored_check(self) -> None:
        command = "D=skills/alpha; cat $D/SKILL.md"
        assert checks._cmd_reads_skill_md(command)
        assert self._credited(command) == []


class TestOutcomeTriState:
    def test_structured_error_flag_marks_failure(self) -> None:
        traj = _traj(_one("mcp__github__x", content="{}", is_error=True))
        assert _activations(traj)[0]["succeeded"] is False

    @pytest.mark.parametrize(
        "content",
        [
            "403: Failed to decrypt access token",
            "<tool_use_error>No such tool</tool_use_error>",
            "MCP error -32602",
            "500: boom",
            "HTTP/1.1 404 - Not Found",
            "Done.\nstatus code 502: bad gateway",
            # A status after an error word or a request line, not at the start of the line.
            "Request failed with status 404: Not Found",
            "Error calling tool: 404: not found",
            "GET /repos/x: 404 - Not Found",
        ],
    )
    def test_failure_markers_mark_failure(self, content: str) -> None:
        traj = _traj(_one("mcp__github__x", content=content))
        assert _activations(traj)[0]["succeeded"] is False

    @pytest.mark.parametrize(
        "content",
        [
            "Open issues:\n#451 - Crash",
            "Found in src/app.py:404: def handler()",
            "Total: 500 - all good",
            "The error handler is in src/app.py:404: def handler()",
        ],
    )
    def test_status_like_numbers_inside_a_successful_answer_are_not_failures(self, content: str) -> None:
        traj = _traj(_one("mcp__github__x", content=content))
        assert _activations(traj)[0]["succeeded"] is True

    def test_failure_marker_mid_way_through_a_long_successful_body_is_ignored(self) -> None:
        # Scanning is bounded: each block's head and the result's tail, not the middle.
        traj = _traj(_one("mcp__github__x", content="x" * 5000 + " permission denied " + "y" * 5000))
        assert _activations(traj)[0]["succeeded"] is True

    def test_failure_marker_at_the_end_of_a_long_body_marks_failure(self) -> None:
        traj = _traj(_one("mcp__github__x", content="x" * 2100 + "\n401 - Unauthorized"))
        assert _activations(traj)[0]["succeeded"] is False

    def test_error_block_after_a_long_first_block_marks_failure(self) -> None:
        content = [
            {"type": "text", "text": "Search results preamble " + "." * 2100},
            {"type": "text", "text": "Error: 403: Failed to decrypt access token"},
            {"type": "text", "text": "z" * 3000},
        ]
        traj = _traj(_one("mcp__github__x", content=content))
        assert _activations(traj)[0]["succeeded"] is False

    def test_missing_sibling_file_does_not_fail_a_shell_manifest_read(self) -> None:
        output = "cat: skills/alpha/REFERENCE.md: No such file or directory\n---\nname: alpha\n---\n"
        traj = _traj(_one("exec_command", {"cmd": "cat skills/alpha/SKILL.md skills/alpha/REFERENCE.md"}, output))
        assert _activations(traj)[0]["succeeded"] is True

    @pytest.mark.parametrize("fn", ["Read", "mcp__github__read_file"])
    def test_missing_file_still_fails_mcp_and_file_read_tools(self, fn: str) -> None:
        traj = _traj(_one(fn, {"file_path": "skills/beta/SKILL.md"}, "Error: file does not exist"))
        assert {a["succeeded"] for a in _activations(traj)} == {False}

    def test_sibling_result_never_proves_success(self) -> None:
        traj = _traj(
            _step(
                [_tc("mcp__github__x", call_id="c1"), _tc("Bash", {"command": "ls"}, call_id="c2")],
                [_res("c2", "files")],
            )
        )
        assert _activations(traj)[0]["succeeded"] is None

    def test_idless_result_is_used_only_for_single_call_steps(self) -> None:
        single = _traj(_step([_tc("mcp__github__x", call_id="")], [{"content": "ok"}]))
        multi = _traj(
            _step([_tc("mcp__github__x", call_id=""), _tc("mcp__jira__y", call_id="")], [{"content": "ok"}]),
        )
        assert _activations(single)[0]["succeeded"] is True
        assert [a["succeeded"] for a in _activations(multi)] == [None, None]

    def test_empty_correlated_body_is_unknown(self) -> None:
        traj = _traj(_one("mcp__github__x", content="  "))
        assert _activations(traj)[0]["succeeded"] is None

    def test_string_arguments_are_decoded(self) -> None:
        traj = _traj(_step([{"tool_call_id": "c1", "function_name": "Skill", "arguments": '{"skill": "beta"}'}]))
        assert _activations(traj)[0]["name"] == "beta"


def test_unreadable_trajectory_yields_none() -> None:
    assert compute_plugin_signals(None, {}) is None
    assert compute_plugin_signals({"steps": "nope"}, {}) is None


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

EMPTY_CASE_STATUSES = ("tool_selection", "arguments", "order", "handoff", "conflict")


def test_case_without_fields_is_not_applicable_everywhere() -> None:
    signals = _signals(_traj(_one("Skill", {"skill": "alpha"})))
    for key in EMPTY_CASE_STATUSES:
        assert signals[key]["status"] == "not_applicable", key
    assert signals["tool_selection"]["precision"] is None
    assert signals["tool_selection"]["called"] == ["Skill:alpha"]
    assert set(signals) == {
        "activations",
        "tool_selection",
        "arguments",
        "mcp_calls",
        "order",
        "handoff",
        "conflict",
        "activation_coverage",
    }


class TestToolSelection:
    def test_exact_glob_skill_form_acceptable_and_decoys(self) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one("mcp__github__list_issues", call_id="c2"),
            _one("mcp__github__get_issue", call_id="c3"),
            _one("mcp__slack__post", call_id="c4"),
            _one("mcp__slack__post", call_id="c5"),
            _one("Bash", {"command": "ls"}, call_id="c6"),
        )
        case = {
            "expected_tools": ["Skill:alpha", "mcp__github__list_*", "mcp__jira__create_ticket"],
            "acceptable_tools": ["mcp__github__get_issue"],
            "decoy_tools": ["mcp__slack__*"],
        }

        block = _signals(traj, case)["tool_selection"]

        assert block["called"] == [
            "Skill:alpha",
            "mcp__github__list_issues",
            "mcp__github__get_issue",
            "mcp__slack__post",
        ]
        assert block["precision"] == 0.75
        assert block["recall"] == pytest.approx(0.6667)
        assert block["f1"] == pytest.approx(0.7059, abs=1e-4)
        assert block["decoy_calls"] == 2
        assert block["status"] == "scored"

    def test_listed_plain_tool_enters_scope_but_unlisted_ones_do_not(self) -> None:
        traj = _traj(_one("Bash", {"command": "ls"}), _one("Read", {"file_path": "x"}, call_id="c2"))
        block = _signals(traj, {"acceptable_tools": ["bash"]})["tool_selection"]
        assert block["called"] == ["Bash"]
        assert block["precision"] == 1.0
        assert block["recall"] is None
        assert block["f1"] is None

    def test_nothing_called_with_expectations_scores_zero(self) -> None:
        block = _signals(_traj(), {"expected_tools": ["Skill:alpha"]})["tool_selection"]
        assert (block["precision"], block["recall"], block["f1"]) == (0.0, 0.0, 0.0)

    def test_decoy_only_case_with_nothing_in_scope_has_undefined_precision(self) -> None:
        block = _signals(_traj(_one("Bash")), {"decoy_tools": ["mcp__slack__*"]})["tool_selection"]
        assert block["status"] == "scored"
        assert block["precision"] is None
        assert block["decoy_calls"] == 0

    def test_generated_wrapper_skill_is_not_a_selection(self) -> None:
        traj = _traj(_one("Skill", {"skill": "my-plugin"}), _one("Skill", {"skill": "alpha"}, call_id="c2"))
        block = _signals(traj, {"expected_tools": ["Skill:alpha"]}, wrapper_skills=["my-plugin"])["tool_selection"]
        assert block["called"] == ["Skill:alpha"]
        assert block["precision"] == 1.0

    def test_codex_manifest_read_satisfies_skill_ref(self) -> None:
        traj = _traj(_one("exec_command", {"cmd": "cat skills/alpha/SKILL.md"}))
        block = _signals(traj, {"expected_tools": ["skill:ALPHA"]})["tool_selection"]
        assert block["recall"] == 1.0


class TestArguments:
    def test_required_schema_and_semantic_constraints(self) -> None:
        traj = _traj(
            _one("mcp__github__search", {"query": "bug label:p1", "limit": 50, "opts": {"sort": "new"}}),
            _one("mcp__github__search", {"query": "feature", "limit": 5, "opts": {"sort": "old"}}, call_id="c2"),
        )
        case = {
            "tool_arguments": [
                {
                    "tool": "mcp__github__search",
                    "required": ["query", "opts.sort"],
                    "schema": {
                        "type": "object",
                        "required": ["limit"],
                        "properties": {
                            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                            "opts": {"type": "object", "properties": {"sort": {"enum": ["new", "top"]}}},
                        },
                    },
                    "contains": {"query": "label:"},
                    "pattern": {"query": "^bug\\b"},
                }
            ]
        }

        block = _signals(traj, case)["arguments"]

        assert block["checked"] == 2
        assert block["passed"] == 0
        rules = {(f["arg"], f["rule"]) for f in block["failures"]}
        assert ("limit", "schema") in rules
        assert ("opts.sort", "schema") in rules
        assert ("query", "contains") in rules
        assert ("query", "pattern") in rules

    def test_passing_call_and_equals_type_strictness(self) -> None:
        traj = _traj(
            _one("mcp__jira__create_ticket", {"project": "OPS", "priority": 2, "dry_run": False}),
            _one("mcp__jira__create_ticket", {"project": "OPS", "priority": 2.0, "dry_run": 0}, call_id="c2"),
        )
        case = {"tool_arguments": [{"tool": "mcp:jira", "equals": {"project": "OPS", "priority": 2, "dry_run": False}}]}

        block = _signals(traj, case)["arguments"]

        assert (block["checked"], block["passed"]) == (2, 1)
        assert block["failures"] == [
            {
                "tool": "mcp__jira__create_ticket",
                "arg": "dry_run",
                "rule": "equals",
                "detail": "expected false, got integer",
            }
        ]

    def test_rule_without_matching_call_is_reported_but_not_checked(self) -> None:
        block = _signals(_traj(_one("Bash")), {"tool_arguments": [{"tool": "mcp__x__*", "required": ["a"]}]})[
            "arguments"
        ]
        assert block["checked"] == 0
        assert block["failures"][0]["rule"] == "not_called"
        assert block["status"] == "scored"

    @pytest.mark.parametrize(
        "rule",
        [
            {"equals": {"command": "make test"}},
            {"contains": {"command": "pytest"}},
            {"pattern": {"command": "^pytest"}},
            {"schema": {"properties": {"command": {"enum": ["ls"], "pattern": "^ls"}}}},
        ],
    )
    @pytest.mark.parametrize(
        ("command", "secret"),
        [
            ("git clone https://deploy:S3cr3tP4ss@gitlab.example.com/x.git", "S3cr3tP4ss"),
            ("psql postgresql://admin:hunter2pw@db:5432/app", "hunter2pw"),
            ("mysql -u root -phunter2pw -h db", "hunter2pw"),
            ("printf '-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0'", "MIIEvQIBADANBgkqhkiG9w0"),
            ("curl -H 'X-Key: sk-" + "Ab1" * 8 + "'", "Ab1Ab1Ab1"),
        ],
    )
    def test_failure_details_never_echo_argument_values(self, rule: dict[str, Any], command: str, secret: str) -> None:
        traj = _traj(_one("Bash", {"command": command}))
        case = {"tool_arguments": [{"tool": "Bash", **rule}]}

        signals = _signals(traj, case)

        assert secret not in json.dumps(signals)
        assert signals["arguments"]["failures"]
        for failure in signals["arguments"]["failures"]:
            assert f"string(len={len(command)})" in failure["detail"]

    def test_failure_detail_is_truncated(self) -> None:
        traj = _traj(_one("mcp__github__call", {"body": "x" * 5000}))
        case = {"tool_arguments": [{"tool": "mcp__github__call", "contains": {"body": "y" * 1000}}]}

        (failure,) = _signals(traj, case)["arguments"]["failures"]

        assert len(failure["detail"]) <= 240

    def test_oversized_value_is_not_regex_scanned(self) -> None:
        traj = _traj(_one("mcp__github__call", {"body": "a" * 10_000}))
        case = {"tool_arguments": [{"tool": "mcp__github__call", "pattern": {"body": "^(a+)+$"}}]}
        (failure,) = _signals(traj, case)["arguments"]["failures"]
        assert failure["detail"] == "value exceeds the pattern-check size limit"

    @pytest.mark.parametrize(
        ("pattern", "subject"),
        [
            (r"^(\w+\s?)+$", "a" * 39 + "!"),
            (r"(a|aa)+$", "a" * 60 + "!"),
            (r"[a-z]*[a-z0-9]*!", "a" * 4096),
        ],
        ids=["nested-quantifier", "overlapping-alternation", "adjacent-quantifiers-at-subject-cap"],
    )
    def test_backtracking_pattern_fails_fast_instead_of_hanging(
        self, monkeypatch: pytest.MonkeyPatch, pattern: str, subject: str
    ) -> None:
        timeouts = _record_regex_timeouts(monkeypatch)
        traj = _traj(_one("Bash", {"command": subject}))
        case = {"tool_arguments": [{"tool": "Bash", "pattern": {"command": pattern}}]}

        block = _signals(traj, case)["arguments"]

        # The search ran under the per-check deadline, so it could not hang.
        assert timeouts
        assert all(0 < timeout <= plugin_signals._PATTERN_TIMEOUT_SECONDS for timeout in timeouts)
        assert (block["checked"], block["passed"]) == (1, 0)
        (failure,) = block["failures"]
        assert failure["detail"] in {
            "pattern check timed out",
            f"value string(len={len(subject)}) does not match {pattern}",
        }

    def test_pattern_checks_share_one_time_budget_per_trial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(plugin_signals, "_PATTERN_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(plugin_signals, "_PATTERN_BUDGET_SECONDS", 0.2)
        timeouts = _record_regex_timeouts(monkeypatch)
        steps = [_one("Bash", {"command": "a" * 60 + "!"}, call_id=f"c{index}") for index in range(20)]
        case = {"tool_arguments": [{"tool": "Bash", "pattern": {"command": "(a|aa)+$"}}]}

        block = _signals(_traj(*steps), case)["arguments"]

        # Each search that timed out spent its share of the 0.2 s budget, so the
        # later checks failed without searching at all.
        assert 0 < len(timeouts) < 20
        assert (block["checked"], block["passed"]) == (20, 0)
        assert {failure["detail"] for failure in block["failures"]} == {"pattern check timed out"}

    def test_huge_integers_and_non_ascii_digits_never_crash_grading(self) -> None:
        huge = 10**400
        traj = _traj(_one("Bash", {"n": huge, "items": [1, 2, 3]}))
        case = {
            "tool_arguments": [
                {"tool": "Bash", "equals": {"n": huge}},
                {"tool": "Bash", "equals": {"n": huge + 1}},
                {"tool": "Bash", "required": ["items.²"]},
                {"tool": "Bash", "schema": {"properties": {"n": {"maximum": huge - 1}}}},
            ]
        }
        assert validate_plugin_case_fields(case) == []

        block = _signals(traj, case)["arguments"]

        assert (block["checked"], block["passed"]) == (4, 1)
        assert [(failure["arg"], failure["detail"]) for failure in block["failures"]] == [
            ("n", "expected " + str(huge + 1)[:66] + "...<truncated>, got integer"),
            ("items.²", "argument is missing"),
            ("n", "value is above maximum " + str(huge - 1)[:66] + "...<truncated>"),
        ]

    @pytest.mark.parametrize("bound", ["minimum", "maximum"])
    def test_huge_schema_bounds_are_validated_not_raised(self, bound: str) -> None:
        entry = {"tool_arguments": [{"tool": "Bash", "schema": {"properties": {"n": {bound: 10**400}}}}]}
        assert validate_plugin_case_fields(entry) == []
        inverted = {"tool_arguments": [{"tool": "Bash", "schema": {"minimum": 10**400, "maximum": 10**399}}]}
        assert any("minimum must not exceed" in problem for problem in validate_plugin_case_fields(inverted))


class TestMcpCalls:
    def test_success_rate_excludes_unknown_and_covers_every_server(self) -> None:
        traj = _traj(
            _one("mcp__github__a", content="ok"),
            _one("mcp__github__b", content="500: boom", call_id="c2"),
            _step([_tc("mcp__undeclared__c", call_id="c3")]),
            _one("Bash", call_id="c4"),
        )

        block = _signals(traj)["mcp_calls"]

        assert {k: block[k] for k in ("total", "succeeded", "failed", "unknown", "success_rate")} == {
            "total": 3,
            "succeeded": 1,
            "failed": 1,
            "unknown": 1,
            "success_rate": 0.5,
        }
        assert block["by_server"]["github"]["tools"] == ["mcp__github__a", "mcp__github__b"]
        assert block["by_server"]["undeclared"]["unknown"] == 1

    def test_no_mcp_calls_has_undefined_rate(self) -> None:
        assert _signals(_traj())["mcp_calls"]["success_rate"] is None


class TestOrder:
    TRAJ = _traj(
        _one("Skill", {"skill": "alpha"}),
        _one("mcp__github__search", call_id="c2"),
        _one("Skill", {"skill": "beta"}, call_id="c3"),
    )

    def test_alternatives_and_violations(self) -> None:
        case = {
            "expected_order": [
                ["Skill:alpha", ["mcp__github__list", "mcp__github__search"]],
                [["Skill:gamma", "Skill:beta"], "Skill:alpha"],
                ["Skill:alpha", "Skill:missing"],
            ]
        }

        block = _signals(self.TRAJ, case)["order"]

        assert (block["edges"], block["satisfied"]) == (3, 1)
        assert block["violated"] == [
            {"before": "Skill:gamma | Skill:beta", "after": "Skill:alpha"},
            {"before": "Skill:alpha", "after": "Skill:missing"},
        ]

    def test_globs_in_refs(self) -> None:
        block = _signals(self.TRAJ, {"expected_order": [["mcp:git*", "skill:b*"]]})["order"]
        assert block["satisfied"] == 1


class TestHandoff:
    def test_value_flows_from_producer_output_to_later_consumer_input(self) -> None:
        traj = _traj(
            _one("mcp__github__get_issue", {"n": 1}, "Issue ISSUE-4242 is open"),
            _one("mcp__jira__create_ticket", {"title": "Track ISSUE-4242"}, call_id="c2"),
        )
        case = {"handoffs": [{"producer": "mcp:github", "consumer": "mcp:jira", "value": "ISSUE-4242"}]}
        block = _signals(traj, case)["handoff"]
        assert (block["checked"], block["passed"], block["failures"]) == (1, 1, [])

    def test_value_already_in_prompt_is_not_attributable(self) -> None:
        traj = _traj(
            _one("mcp__github__get_issue", {}, "ISSUE-4242"),
            _one("mcp__jira__create_ticket", {"title": "ISSUE-4242"}, call_id="c2"),
            prompt="File ISSUE-4242 in jira",
        )
        case = {"handoffs": [{"producer": "mcp:github", "consumer": "mcp:jira", "value": "ISSUE-4242"}]}
        (failure,) = _signals(traj, case)["handoff"]["failures"]
        assert "task prompt" in failure["detail"]

    def test_consumer_in_the_same_step_cannot_have_seen_the_output(self) -> None:
        traj = _traj(
            _step(
                [
                    _tc("mcp__github__get_issue", call_id="c1"),
                    _tc("mcp__jira__create_ticket", {"title": "ISSUE-4242"}, call_id="c2"),
                ],
                [_res("c1", "ISSUE-4242"), _res("c2", "ok")],
            )
        )
        case = {"handoffs": [{"producer": "mcp:github", "consumer": "mcp:jira", "value": "ISSUE-4242"}]}
        (failure,) = _signals(traj, case)["handoff"]["failures"]
        assert "did not reach consumer" in failure["detail"]

    def test_artifact_written_in_producer_skill_window_and_read_by_consumer(self) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one("Write", {"file_path": "/workspace/out/report.json", "content": "{}"}, call_id="c2"),
            _one("Skill", {"skill": "beta"}, call_id="c3"),
            _one("Bash", {"command": "jq . out/report.json"}, call_id="c4"),
        )
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "artifact": "out/report.json"}]}
        assert _signals(traj, case)["handoff"]["passed"] == 1

    def test_artifact_read_outside_the_consumer_window_does_not_count(self) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one("Bash", {"command": "echo '{}' > out/report.json"}, call_id="c2"),
            _one("Read", {"file_path": "out/report.json"}, call_id="c3"),
            _one("Skill", {"skill": "beta"}, call_id="c4"),
        )
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "artifact": "out/report.json"}]}
        (failure,) = _signals(traj, case)["handoff"]["failures"]
        assert failure == {
            "producer": "Skill:alpha",
            "consumer": "Skill:beta",
            "detail": "artifact was not read by the consumer after the producer wrote it",
        }

    @pytest.mark.parametrize(
        ("fn", "args"),
        [
            ("apply_patch", {"input": "*** Begin Patch\n  *** Add File: out/report.json\n+{}\n*** End Patch"}),
            (
                "apply_patch",
                {"input": "*** Begin Patch\n*** Update File: draft.json\n*** Move to: out/report.json\n*** End Patch"},
            ),
            ("apply_patch", {"patchText": "*** Begin Patch\n*** Add File: out/report.json\n+{}\n*** End Patch"}),
            ("apply_patch", {"raw": "*** Begin Patch\n*** Add File: out/report.json\n+{}\n*** End Patch"}),
            ("applypatch", {"input": "*** Begin Patch\n*** Add File: out/report.json\n+{}\n*** End Patch"}),
            ("patch", {"mode": "patch", "patch": "*** Begin Patch\n*** Update File: out/report.json\n*** End Patch"}),
            ("patch", {"mode": "replace", "path": "out/report.json", "old_string": "a", "new_string": "b"}),
        ],
        ids=[
            "indented-header",
            "move-to",
            "opencode-patch-text",
            "raw-input",
            "applypatch",
            "hermes-patch",
            "hermes-replace",
        ],
    )
    def test_patch_tool_writes_count_as_artifact_writes(self, fn: str, args: dict[str, Any]) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one(fn, args, call_id="c2"),
            _one("Skill", {"skill": "beta"}, call_id="c3"),
            _one("Read", {"file_path": "out/report.json"}, call_id="c4"),
        )
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "artifact": "out/report.json"}]}
        assert _signals(traj, case)["handoff"]["passed"] == 1

    @pytest.mark.parametrize(
        ("headers", "passed"),
        [
            ("*** Add File: out/report.json\n+{}", 1),
            ("*** Update File: out/report.json\n@@\n-a\n+b", 1),
            ("*** Update File: draft.json\n*** Move to: out/report.json", 1),
            ("*** Delete File: out/report.json", 0),
            ("*** Update File: out/other.json\n@@\n-a\n+b\n*** Delete File: out/report.json", 0),
            ("*** Delete File: out/report.json\n*** Add File: out/report.json\n+{}", 1),
            ("*** Add File: out/report.json\n+{}\n*** Delete File: out/report.json", 1),
        ],
        ids=["add", "update", "move-to", "delete", "delete-beside-update", "delete-then-add", "add-then-delete"],
    )
    @pytest.mark.parametrize("form", ["tool", "shell"])
    def test_a_patch_that_only_deletes_the_artifact_does_not_write_it(
        self, headers: str, passed: int, form: str
    ) -> None:
        patch = f"*** Begin Patch\n{headers}\n*** End Patch"
        call = (
            _one("apply_patch", {"input": patch}, call_id="c2")
            if form == "tool"
            else _one("exec_command", {"cmd": f"apply_patch <<'EOF'\n{patch}\nEOF"}, call_id="c2")
        )
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            call,
            _one("Skill", {"skill": "beta"}, call_id="c3"),
            _one("Read", {"file_path": "out/report.json"}, call_id="c4"),
        )
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "artifact": "out/report.json"}]}
        handoff = _signals(traj, case)["handoff"]
        assert handoff["passed"] == passed
        if not passed:
            assert [failure["detail"] for failure in handoff["failures"]] == [
                "artifact was not written by the producer"
            ]

    @pytest.mark.parametrize(
        ("command", "passed"),
        [
            ("cd /workspace && apply_patch <<'EOF'\n*** Begin Patch\n*** Add File: out/report.json\n+{}\nEOF", 1),
            # A header-like line in a file that is not a patch names no write.
            ("cat > notes.md <<'EOF'\n*** Add File: out/report.json\nEOF", 0),
        ],
        ids=["shell-apply-patch", "plain-heredoc"],
    )
    def test_shell_apply_patch_headers_count_as_artifact_writes(self, command: str, passed: int) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one("exec_command", {"cmd": command}, call_id="c2"),
            _one("Skill", {"skill": "beta"}, call_id="c3"),
            _one("Read", {"file_path": "out/report.json"}, call_id="c4"),
        )
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "artifact": "out/report.json"}]}
        assert _signals(traj, case)["handoff"]["passed"] == passed

    def test_mcp_consumer_receiving_the_artifact_path_counts_as_a_read(self) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one("Bash", {"command": "python gen.py | tee /workspace/out.csv"}, call_id="c2"),
            _one("mcp__github__upload", {"file": "/workspace/out.csv"}, call_id="c3"),
        )
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "mcp:github", "artifact": "out.csv"}]}
        assert _signals(traj, case)["handoff"]["passed"] == 1

    def test_inactive_components_fail_fast(self) -> None:
        case = {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "value": "x"}]}
        (failure,) = _signals(_traj(), case)["handoff"]["failures"]
        assert failure["detail"] == "producer was not activated; consumer was not activated"

    def test_prompts_are_read_only_when_a_handoff_checks_a_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reads: list[int] = []
        prompt_texts = plugin_signals._prompt_texts
        monkeypatch.setattr(plugin_signals, "_prompt_texts", lambda traj: reads.append(1) or prompt_texts(traj))
        traj = _traj(_one("Skill", {"skill": "alpha"}))

        _signals(traj, {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "artifact": "a.json"}]})
        assert reads == []
        _signals(traj, {"handoffs": [{"producer": "Skill:alpha", "consumer": "Skill:beta", "value": "v"}]})
        assert reads == [1]


class TestConflict:
    def test_pass_fail_and_alternatives(self) -> None:
        traj = _traj(_one("Skill", {"skill": "alpha"}), _one("mcp__jira__create", call_id="c2"))
        case = {
            "conflict_probes": [
                {"id": "ok", "must_use": "Skill:alpha", "must_not_use": ["Skill:beta", "mcp:slack"]},
                {"id": "bad", "must_use": "Skill:beta", "must_not_use": "mcp:jira", "description": "d"},
            ]
        }

        block = _signals(traj, case)["conflict"]

        assert (block["checked"], block["passed"]) == (2, 1)
        assert block["failures"] == [
            {"probe": "bad", "detail": "must_use Skill:beta was not activated; must_not_use mcp:jira was activated"}
        ]


class TestActivationCoverage:
    def test_declared_exercised_unverified_unavailable(self) -> None:
        traj = _traj(
            _one("Skill", {"skill": "alpha"}),
            _one("mcp__jira__create", content="connection refused", call_id="c2"),
            _one("mcp__jira__create", content="failed to connect", call_id="c3"),
            _one("mcp__github__list", content="403: nope", call_id="c4"),
            _step([_tc("mcp__github__list", call_id="c5")]),
        )

        block = _signals(traj)["activation_coverage"]

        assert block == {
            "declared": ["skill:alpha", "skill:beta", "mcp:github", "mcp:jira"],
            "exercised": ["skill:alpha", "mcp:github", "mcp:jira"],
            "unverified": ["skill:beta"],
            "unavailable": ["mcp:jira"],
        }

    def test_every_spelling_and_namespace_credits_the_declared_component(self) -> None:
        traj = _traj(
            _one("Skill", {"skill": "ALPHA"}),
            _one("Skill", {"skill": "demo:deploy"}, call_id="c2"),
            _one("Task", {"subagent_type": "Reviewer"}, call_id="c3"),
            _one("my_docs__search", content="connection refused", call_id="c4"),
            _one("mcp__plugin_demo_my_docs__search", content="403: forbidden", call_id="c5"),
            _one("mcp__tracker__list", content="500: down", call_id="c6"),
            _one("tracker.get", content="ok", call_id="c7"),
        )
        declared = {
            "skill": ["Alpha", "alpha", "beta"],
            "mcp": ["my.docs", "tracker"],
            "subagent": ["reviewer"],
            "command": ["deploy"],
        }

        block = _signals(traj, declared=declared)["activation_coverage"]

        assert block["exercised"] == [
            "skill:Alpha",
            "skill:alpha",
            "mcp:my.docs",
            "mcp:tracker",
            "subagent:reviewer",
            "command:deploy",
        ]
        assert block["unverified"] == ["skill:beta"]
        # Every my.docs call failed; one tracker call worked.
        assert block["unavailable"] == ["mcp:my.docs"]

    def test_no_declared_components(self) -> None:
        block = _signals(_traj(_one("Skill", {"skill": "alpha"})), declared=None)["activation_coverage"]
        assert block == {"declared": [], "exercised": [], "unverified": [], "unavailable": []}


# ---------------------------------------------------------------------------
# Per-arm summary
# ---------------------------------------------------------------------------


def test_summary_aggregates_rates_means_and_coverage_union() -> None:
    case = {
        "expected_tools": ["Skill:alpha"],
        "conflict_probes": [{"id": "p", "must_use": "Skill:alpha", "must_not_use": "Skill:beta"}],
    }
    first = _signals(
        _traj(_one("Skill", {"skill": "alpha"}), _one("mcp__jira__x", content="500: down", call_id="c2")), case
    )
    second = _signals(
        _traj(_one("Skill", {"skill": "beta"}), _one("mcp__jira__x", content="ok", call_id="c2")),
        case,
    )
    third = _signals(_traj(_one("mcp__github__y", content="403: denied")), {})

    summary = summarize_plugin_signals([first, second, third, None])

    assert summary["n_trials"] == 3
    assert summary["n_missing_trajectory"] == 1
    assert summary["activations"] == {
        "total": 5,
        "mean_per_trial": pytest.approx(1.6667),
        "by_type": {"skill": 2, "mcp": 3},
    }
    assert summary["tool_selection"]["n_scored"] == 2
    assert summary["tool_selection"]["recall"] == 0.5
    assert summary["conflict"] == {"n_scored": 2, "checked": 2, "passed": 1, "pass_rate": 0.5, "status": "scored"}
    assert summary["arguments"]["status"] == "not_applicable"
    assert summary["mcp_calls"]["by_server"]["jira"]["success_rate"] == 0.5
    coverage = summary["activation_coverage"]
    assert coverage["exercised"] == ["skill:alpha", "skill:beta", "mcp:github", "mcp:jira"]
    assert coverage["unverified"] == []
    # jira worked in one trial, so only github (failed everywhere it ran) is unavailable.
    assert coverage["unavailable"] == ["mcp:github"]
    assert coverage["exercise_rate"]["mcp:jira"] == pytest.approx(0.6667)


def test_summary_of_no_trials_is_empty_but_well_formed() -> None:
    summary = summarize_plugin_signals([])
    assert summary["n_trials"] == 0
    assert summary["tool_selection"]["status"] == "not_applicable"
    assert summary["mcp_calls"]["success_rate"] is None


# ---------------------------------------------------------------------------
# Dataset case fields
# ---------------------------------------------------------------------------

GOOD_CASE: dict[str, Any] = {
    "id": "case-1",
    "prompt": "p",
    "expected_tools": ["mcp__github__*", "Skill:alpha"],
    "acceptable_tools": ["Read"],
    "decoy_tools": ["mcp__slack__*"],
    "tool_arguments": [
        {
            "tool": "mcp__github__search",
            "required": ["query"],
            "schema": {"type": "object", "properties": {"limit": {"type": ["integer", "null"], "maximum": 10}}},
            "equals": {"sort": "new"},
            "contains": {"query": "bug"},
            "pattern": {"query": "^bug"},
        }
    ],
    "expected_order": [["Skill:alpha", ["mcp:github", "mcp:jira"]]],
    "handoffs": [{"producer": "mcp:github", "consumer": "mcp:jira", "value": "ISSUE-1", "artifact": "out.json"}],
    "conflict_probes": [{"id": "p1", "must_use": "Skill:alpha", "must_not_use": "Skill:beta", "description": "d"}],
}


def test_valid_case_fields_are_accepted_and_spec_is_idempotent() -> None:
    assert validate_plugin_case_fields(GOOD_CASE) == []
    spec = plugin_case_spec(GOOD_CASE)
    assert set(spec) == {
        "expected_tools",
        "acceptable_tools",
        "decoy_tools",
        "tool_arguments",
        "expected_order",
        "handoffs",
        "conflict_probes",
    }
    assert plugin_case_spec(spec) == spec
    assert validate_plugin_case_fields(spec) == []
    assert json.loads(json.dumps(spec)) == spec


def test_absent_and_null_fields_are_fine() -> None:
    assert validate_plugin_case_fields({"id": "x", "expected_tools": None}) == []
    assert plugin_case_spec({"id": "x"}) == {}


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("expected_tools", "mcp__x__y", "must be a list"),
        ("expected_tools", [""], "non-empty string"),
        ("expected_tools", ["Skill:"], "must be followed by a name"),
        ("acceptable_tools", ["x" * 300], "at most 256"),
        ("decoy_tools", [f"t{i}" for i in range(MAX_TOOL_PATTERNS + 1)], "at most 64"),
        ("tool_arguments", [{"tool": "x"}], "at least one of"),
        ("tool_arguments", [{"tool": "x", "requred": ["a"]}], "unsupported keys"),
        ("tool_arguments", [{"tool": "x", "schema": {"items": {}}}], "unsupported JSON-Schema keyword"),
        ("tool_arguments", [{"tool": "x", "schema": {"type": "float"}}], "must be one of"),
        ("tool_arguments", [{"tool": "x", "schema": {"minimum": 5, "maximum": 1}}], "minimum must not exceed"),
        ("tool_arguments", [{"tool": "x", "pattern": {"a": "("}}], "not a valid regular expression"),
        ("tool_arguments", [{"tool": "x", "contains": {"a": 3}}], "non-empty string"),
        ("tool_arguments", [{"tool": "x", "equals": {}}], "non-empty object"),
        ("expected_order", [["a"]], "two-item"),
        ("expected_order", [["a", []]], "non-empty list"),
        ("handoffs", [{"producer": "a", "consumer": "b"}], "value or an artifact"),
        ("handoffs", [{"producer": "a", "consumer": "b", "value": ""}], "non-empty string"),
        ("conflict_probes", [{"id": "p", "must_use": "a"}], "must_not_use"),
        ("conflict_probes", [{"id": "p", "must_use": "a", "must_not_use": "b", "description": 3}], "description"),
        ("handoffs", [{"producer": "a", "consumer": "b", "value": "v", "description": ["x"]}], "description"),
        ("tool_arguments", [{"tool": "x", "required": ["a"], "description": "d" * 2000}], "description"),
        (
            "conflict_probes",
            [{"id": "p", "must_use": "a", "must_not_use": "b"}, {"id": "p", "must_use": "a", "must_not_use": "b"}],
            "must be unique",
        ),
    ],
)
def test_invalid_case_fields_are_reported_and_dropped(field: str, value: Any, fragment: str) -> None:
    entry = {"id": "case-1", field: value}
    problems = validate_plugin_case_fields(entry)
    assert problems, field
    assert any(fragment in problem for problem in problems), problems
    assert field not in plugin_case_spec(entry)


@pytest.mark.parametrize(
    ("pattern", "valid"),
    [
        (r"^\p{Lu}\w+$", True),  # a Unicode property: the regex engine runs it, ``re`` cannot compile it
        ("[[:alpha:]", False),  # ``re`` reads a set of characters; the regex engine an unclosed POSIX class
        ("(?a)a(?u)", False),  # the regex engine raises ValueError, not regex.error
    ],
)
def test_patterns_are_validated_by_the_engine_that_runs_them(pattern: str, valid: bool) -> None:
    entry = {"tool_arguments": [{"tool": "mcp__jira__create", "pattern": {"title": pattern}}]}
    schema_entry = {"tool_arguments": [{"tool": "t", "schema": {"properties": {"title": {"pattern": pattern}}}}]}

    for problems in (validate_plugin_case_fields(entry), validate_plugin_case_fields(schema_entry)):
        if valid:
            assert problems == []
        else:
            assert any("not a valid regular expression" in problem for problem in problems), problems
    if valid:
        traj = _traj(_one("mcp__jira__create", {"title": "Track"}))
        assert _signals(traj, entry)["arguments"]["passed"] == 1


def test_deeply_nested_schema_is_rejected() -> None:
    schema: dict[str, Any] = {"type": "object"}
    for _ in range(10):
        schema = {"type": "object", "properties": {"x": schema}}
    problems = validate_plugin_case_fields({"tool_arguments": [{"tool": "t", "schema": schema}]})
    assert any("nesting" in problem for problem in problems)


def test_invalid_fields_are_ignored_by_graders() -> None:
    signals = _signals(_traj(_one("Skill", {"skill": "alpha"})), {"expected_tools": "Skill:alpha"})
    assert signals["tool_selection"]["status"] == "not_applicable"


def test_context_builder_bounds_and_scopes_declared_components() -> None:
    context = build_plugin_signals_context(
        member_skills=["alpha", "alpha", "", 3, "beta"],
        mcp_servers=["github", None],
        wrapper_skills=["demo-plugin-eval", "demo"],
        entries=[GOOD_CASE, {"id": "plain", "prompt": "p"}, {"id": "bad", "expected_tools": "x"}, "junk"],
    )

    assert context.member_skills == ("alpha", "beta")
    assert context.declared_for("with_skill") == {"skill": ["alpha", "beta"], "mcp": ["github"]}
    assert context.declared_for("sum_of_parts") == {"skill": ["alpha", "beta"], "mcp": []}
    assert set(context.cases) == {"case-1"}
    assert context.case_spec("missing") == {}
    assert context.arm_enabled("with_skill") and context.arm_enabled("sum_of_parts")
    assert not context.arm_enabled("without_skill")


class TestUntrustedTrajectoryContent:
    def test_argument_derived_names_persist_only_when_identifier_shaped_or_declared(self) -> None:
        traj = _traj(
            _one("Skill", {"skill": "https://u:SuperSecretPW@host/x"}),
            _one("Task", {"subagent_type": "-----BEGIN RSA PRIVATE KEY-----MIIEowIBAAKCAQEA"}, call_id="c2"),
            _one("SlashCommand", {"command": "postgresql://admin:hunter2pw@db/app --now"}, call_id="c3"),
            _one("Skill", {"skill": "Release Notes"}, call_id="c4"),
            _one("Task", {"subagent_type": "code-reviewer"}, call_id="c5"),
            _one("SlashCommand", {"command": "/plugin:deploy prod"}, call_id="c6"),
        )
        declared = {**DECLARED, "skill": [*DECLARED["skill"], "release notes"]}
        case = {"tool_arguments": [{"tool": "Skill:*", "required": ["missing"]}]}

        signals = _signals(traj, case, declared=declared)

        dumped = json.dumps(signals)
        for secret in ("SuperSecretPW", "MIIEowIBAAKCAQEA", "hunter2pw"):
            assert secret not in dumped
        assert [a["name"] for a in signals["activations"]] == [
            "<non-name>",
            "<non-name>",
            "<non-name>",
            "Release Notes",
            "code-reviewer",
            "plugin:deploy",
        ]
        assert signals["tool_selection"]["called"] == [
            "Skill:<non-name>",
            "Agent:<non-name>",
            "Command:<non-name>",
            "Skill:Release Notes",
            "Agent:code-reviewer",
            "Command:plugin:deploy",
        ]
        assert [f["tool"] for f in signals["arguments"]["failures"]] == ["Skill:<non-name>", "Skill:Release Notes"]

    def test_adversarial_text_for_the_redactor_does_not_stall_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = "a." * 32_768  # quadratic for the credential redactor's assignment patterns
        traj = _traj(
            _one("Skill", {"skill": payload}),
            _one("Task", {"subagent_type": payload}, call_id="c2"),
            _one("Bash", {"command": payload}, call_id="c3"),
        )
        case = {
            "tool_arguments": [
                {"tool": "Bash", "contains": {"command": "pytest"}},
                {"tool": "Bash", "equals": {"command": "make test"}},
                {"tool": "Bash", "pattern": {"command": "^pytest"}},
                {"tool": "Bash", "schema": {"properties": {"command": {"enum": ["ls"]}}}},
            ]
        }

        redacted: list[int] = []
        redact = plugin_signals.redact_sensitive_text

        def recording_redact(text: str, **kwargs: Any) -> str:
            redacted.append(len(text))
            return redact(text, **kwargs)

        monkeypatch.setattr(plugin_signals, "redact_sensitive_text", recording_redact)

        signals = _signals(traj, case)

        # The redactor only ever sees a bounded window (4 x 256 + 256 characters), never the payload.
        assert redacted
        assert max(redacted) <= 4 * plugin_signals._MAX_LABEL_CHARS + 256
        assert len(signals["arguments"]["failures"]) == 4

    def test_unhashable_step_source_is_ignored(self) -> None:
        traj = _traj(_one("Skill", {"skill": "alpha"}))
        traj["steps"].insert(0, {"source": ["user"], "message": "hi"})
        assert _signals(traj)["activations"][0]["name"] == "alpha"
