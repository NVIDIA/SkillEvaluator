# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static always-on context cost follows what each harness really loads (checks 7 and 25).

Each fixture is a small copy of a verification example: the forced output styles
Claude Code 2.1.284 applies (``force-for-plugin`` probe, two forced styles, the
forced-vs-selectable capture), MCP servers and SessionStart hooks the old rule
counted as 0, ``disable-model-invocation`` items Claude Code does not list, a
Cursor ``alwaysApply`` rule, CJK text, and a Latin-1 agent Claude Code still
lists. Every test runs the real Tier 1 plugin validator or the inventory builder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.plugin_components import build_plugin_inventory
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_STYLE_BODY = "Keep every reply short. Skip preambles and summaries. Lead with the answer."
_SKILL = "---\nname: notes\ndescription: Take short notes\n---\n# Notes\nWrite them down.\n"
# Claude Code 2.1.284 dropped 3,233-3,262 characters of its default system text when a forced style applied.
_REPLACED_DEFAULT_CHARS = 3_250


def _plugin(root: Path, files: dict[str, Any], manifest: dict[str, Any] | None = None) -> Path:
    manifest_rel = ".claude-plugin/plugin.json"
    target = root / manifest_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"name": "demo", **(manifest or {})}), encoding="utf-8")
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _cost(root: Path) -> dict[str, Any]:
    return PluginSchemaValidator().validate(root).metadata["plugin"]["context_cost"]


def _rows(cost: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["type"], row["name"]): row for row in cost["by_component"]}


def _style(name: str, force: str | None, body: str = _STYLE_BODY, extra: str = "") -> str:
    flag = f"force-for-plugin: {force}\n" if force is not None else ""
    return f"---\nname: {name}\ndescription: {name} style\n{flag}{extra}---\n\n{body}\n"


def _tokens(chars: int) -> int:
    return math.ceil(chars / 4) if chars >= 0 else -math.ceil(-chars / 4)


def _view(cost: dict[str, Any], harness: str, load_mode: str) -> dict[str, Any]:
    return next(v for v in cost["by_harness"] if (v["harness"], v["load_mode"]) == (harness, load_mode))


# --------------------------------------------------------------------------- #
# M21: forced output styles                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "forced"),
    [("true", True), ('"true"', True), ("yes", True), ("1", True), ('"false"', False), ("false", False)],
)
def test_m21_force_for_plugin_values_match_what_claude_code_forces(tmp_path: Path, value: str, forced: bool) -> None:
    """check-07 tier3-02 probe: Claude Code forces true, "true", yes and 1, and not "false" or false."""
    cost = _cost(_plugin(tmp_path, {"skills/notes/SKILL.md": _SKILL, "output-styles/terse.md": _style("terse", value)}))

    row = _rows(cost)[("output_style", "terse")]
    if forced:
        # A forced style replaces Claude Code's default coding instructions, so it shrinks the prompt.
        assert row["always_on_tokens"] < 0
        assert row["on_demand_tokens"] == 0
        assert "forced" in row["basis"]
    else:
        assert row["always_on_tokens"] == 0
        assert row["on_demand_tokens"] == _tokens(len(_STYLE_BODY))


def test_m21_only_one_forced_style_applies(tmp_path: Path) -> None:
    """check-07 edge-11: two forced styles; Claude Code applies terse and warns, so verbose is not always-on."""
    verbose = "Explain every step in detail. Add background, alternatives, and a summary at the end of every reply."
    root = _plugin(
        tmp_path,
        {
            "skills/notes/SKILL.md": _SKILL,
            "output-styles/terse.md": _style("terse", "true"),
            "output-styles/verbose.md": _style("verbose", "true", verbose),
        },
    )
    cost = _cost(root)
    rows = _rows(cost)

    terse, other = rows[("output_style", "terse")], rows[("output_style", "verbose")]
    assert terse["always_on_tokens"] < 0
    assert other["always_on_tokens"] == 0
    assert other["on_demand_tokens"] == _tokens(len(verbose))
    assert "only one forced output style" in other["basis"]
    assert "terse" in other["basis"]
    assert cost["always_on_tokens"] == sum(row["always_on_tokens"] for row in cost["by_component"])


def test_m21_forced_vs_selectable_matches_the_claude_code_capture(tmp_path: Path) -> None:
    """check-25 pos-03: Claude applied forced-string ("true"), not forced (true); the prompt shrank by ~664 tokens."""
    filler = (
        "The helper reads the named input, checks every field against the stated rules, writes the result to the "
        "out folder, and reports what it changed in one short paragraph for the reviewer. "
    )
    forced_string_body = ("Answer in bullet points.\n\n" + filler * 3)[:426]
    root = _plugin(
        tmp_path,
        {
            "skills/helper/SKILL.md": "---\nname: helper\ndescription: Tidy the named input file quickly\n---\nBody\n",
            "output-styles/forced.md": _style("forced", "true", ("Answer briefly.\n\n" + filler * 4)[:617]),
            "output-styles/forced-string.md": _style("forced-string", '"true"', forced_string_body),
            "output-styles/selectable.md": _style("selectable", None, ("Answer with a table.\n\n" + filler * 4)[:622]),
        },
    )
    cost = _cost(root)
    rows = _rows(cost)

    assert rows[("output_style", "forced-string")]["always_on_tokens"] < 0
    assert rows[("output_style", "forced")]["always_on_tokens"] == 0
    assert rows[("output_style", "selectable")]["always_on_tokens"] == 0
    # The local Claude Code capture measured -2,654 characters (-664 tokens) for this plugin.
    assert cost["always_on_tokens"] < 0
    assert abs(cost["always_on_tokens"] - -664) <= 0.15 * 664


def test_m21_keep_coding_instructions_keeps_the_default_prompt(tmp_path: Path) -> None:
    style = _style("terse", "true", extra="keep-coding-instructions: true\n")
    rows = _rows(_cost(_plugin(tmp_path, {"output-styles/terse.md": style})))

    row = rows[("output_style", "terse")]
    assert row["always_on_tokens"] > 0
    assert row["always_on_tokens"] < _tokens(_REPLACED_DEFAULT_CHARS)


# --------------------------------------------------------------------------- #
# M33: MCP schemas, hook output, harness and load mode, hidden items          #
# --------------------------------------------------------------------------- #
def test_m33_mcp_tool_schemas_are_reported_as_not_counted(tmp_path: Path) -> None:
    """check-25 neg-02 / tier3-08: a stdio server adds ~1,600 characters of tool schemas; say so, do not print 0."""
    root = _plugin(
        tmp_path,
        {
            "skills/notes/SKILL.md": _SKILL,
            ".mcp.json": {"mcpServers": {"reltools": {"command": "python3", "args": ["server.py"]}}},
        },
    )
    cost = _cost(root)

    row = _rows(cost)[("mcp", "reltools")]
    assert row["not_counted"]
    assert "tool schemas" in row["not_counted"]
    assert cost["lower_bound"] is True
    assert any("reltools" in item for item in cost["not_counted"])


def test_m33_session_start_hook_output_is_counted(tmp_path: Path) -> None:
    """check-25 edge-06: SessionStart `cat brief.md` put 4,197 characters into Claude Code's first request."""
    brief = ("Project brief line with the house rules for every reply. " * 74)[:4_197]
    hooks = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": 'cat "${CLAUDE_PLUGIN_ROOT}/context/brief.md"'}]}
            ]
        }
    }
    root = _plugin(
        tmp_path,
        {"skills/notes/SKILL.md": _SKILL, "hooks/hooks.json": hooks, "context/brief.md": brief},
    )
    cost = _cost(root)

    row = _rows(cost)[("hook", "hooks/hooks.json")]
    assert row["always_on_tokens"] == _tokens(len(brief))
    assert not row.get("not_counted")
    assert cost["always_on_tokens"] >= _tokens(len(brief))
    # Neither wrapper nor Codex native loading stages plugin hooks.
    assert _view(cost, "codex", "native")["always_on_tokens"] < _tokens(len(brief))


def test_m33_user_prompt_submit_echo_output_is_counted(tmp_path: Path) -> None:
    message = "Always answer in English and cite the house style guide."
    hooks = {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": f'echo "{message}"'}]}]}}
    cost = _cost(_plugin(tmp_path, {"hooks/hooks.json": hooks}))

    row = _rows(cost)[("hook", "hooks/hooks.json")]
    assert row["always_on_tokens"] == _tokens(len(message) + 1)
    assert "UserPromptSubmit" in row["basis"]
    assert cost["lower_bound"] is False


def test_m33_hook_output_that_cannot_be_read_statically_is_not_counted(tmp_path: Path) -> None:
    hooks = {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "./scripts/brief.sh"}]}]}}
    root = _plugin(tmp_path, {"hooks/hooks.json": hooks, "scripts/brief.sh": "#!/bin/sh\necho hi\n"})
    cost = _cost(root)

    row = _rows(cost)[("hook", "hooks/hooks.json")]
    assert row["not_counted"]
    assert cost["lower_bound"] is True


def test_m33_disable_model_invocation_items_are_not_listed_by_claude_code(tmp_path: Path) -> None:
    """skeptic x-dmi: Claude Code keeps hidden skills and commands out of the model's listing; Codex ignores the flag."""
    description = (
        "Converts quarterly ledger exports into reconciled vendor tables with currency normalisation, owner tags "
        "and audit notes. Use only when explicitly asked for the ledger reconcile flow."
    )
    root = _plugin(
        tmp_path,
        {
            "skills/hidden/SKILL.md": (
                f'---\nname: hidden\ndescription: "{description}"\ndisable-model-invocation: true\n---\n# Hidden\nBody\n'
            ),
            "skills/shown/SKILL.md": (
                '---\nname: shown\ndescription: "Tidies CSV files into typed JSON."\n---\n# Shown\nBody\n'
            ),
            "commands/manual.md": f'---\ndescription: "{description}"\ndisable-model-invocation: "true"\n---\nRun it.\n',
        },
    )
    cost = _cost(root)
    rows = _rows(cost)

    assert rows[("skill", "hidden")]["always_on_tokens"] == 0
    assert rows[("command", "manual")]["always_on_tokens"] == 0
    assert rows[("skill", "shown")]["always_on_tokens"] > 0
    assert _view(cost, "codex", "native")["always_on_tokens"] > _view(cost, "claude-code", "native")["always_on_tokens"]


def _noisy_like(root: Path) -> Path:
    return _plugin(
        root,
        {
            "skills/csv-tidy/SKILL.md": "---\nname: csv-tidy\ndescription: Convert CSV into typed JSON\n---\nBody\n",
            "agents/steward.md": "---\nname: steward\ndescription: Reviews data ownership\n---\nPrompt body.\n",
            "commands/tidy-all.md": "---\ndescription: Tidy every CSV file\n---\nRun csv-tidy on each file.\n",
            "rules/naming.md": "Use snake_case keys. " * 40,
            ".mcp.json": {"mcpServers": {"reltools": {"command": "python3", "args": ["server.py"]}}},
        },
    )


def test_m33_estimate_is_given_per_harness_and_load_mode(tmp_path: Path) -> None:
    """check-25 tier3-02/09: one static 3,086 sat next to 729-4,287 measured; each load mode needs its own number."""
    root = _noisy_like(tmp_path)
    inventory = build_plugin_inventory(
        root, {"name": "demo"}, contained=True, manifest_rel=".claude-plugin/plugin.json"
    )
    cost = inventory.context_cost()
    skill = _tokens(len("csv-tidy" + "Convert CSV into typed JSON"))
    agent = _tokens(len("Reviews data ownership"))
    command = _tokens(len("Tidy every CSV file"))
    rule = _tokens(len(("Use snake_case keys. " * 40).strip()))

    assert (cost["harness"], cost["load_mode"]) == ("claude-code", "native")
    assert _view(cost, "claude-code", "native")["always_on_tokens"] == skill + agent + command + rule
    assert _view(cost, "claude-code", "wrapper")["always_on_tokens"] == skill
    assert _view(cost, "codex", "native")["always_on_tokens"] == skill + rule
    assert _view(cost, "codex", "wrapper")["always_on_tokens"] == skill
    assert all(view["lower_bound"] for view in cost["by_harness"])  # the MCP server is in every view

    codex = inventory.context_cost(harness="codex", load_mode="wrapper")
    assert (codex["harness"], codex["load_mode"]) == ("codex", "wrapper")
    assert codex["always_on_tokens"] == skill
    rows = _rows(codex)
    assert rows[("rule", "naming.md")]["on_demand_tokens"] == rule
    assert "wrapper" in rows[("rule", "naming.md")]["basis"]
    assert rows[("agent", "steward")]["always_on_tokens"] == 0
    assert "not loaded" in rows[("agent", "steward")]["basis"]


def test_m33_codex_plugin_is_estimated_for_codex(tmp_path: Path) -> None:
    root = tmp_path
    (root / ".codex-plugin").mkdir(parents=True)
    (root / ".codex-plugin" / "plugin.json").write_text(json.dumps({"name": "demo", "skills": "./skills/"}))
    (root / "skills" / "notes").mkdir(parents=True)
    (root / "skills" / "notes" / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    (root / "commands").mkdir()
    (root / "commands" / "go.md").write_text("---\ndescription: Go now\n---\nGo.\n", encoding="utf-8")

    cost = PluginSchemaValidator().validate(root).metadata["plugin"]["context_cost"]

    assert (cost["harness"], cost["load_mode"]) == ("codex", "native")
    assert cost["always_on_tokens"] == _tokens(len("notes" + "Take short notes"))


# --------------------------------------------------------------------------- #
# L29: Cursor alwaysApply, CJK, Latin-1 agent                                 #
# --------------------------------------------------------------------------- #
def test_l29_cursor_always_apply_rule_is_always_on_without_its_frontmatter(tmp_path: Path) -> None:
    """check-25 edge-04/cursor: alwaysApply rules go into every Cursor request; the frontmatter is not body."""
    body = "# Style rule\n\n" + "Write short sentences and name every owner. " * 18
    rule = f"---\ndescription: House style\nalwaysApply: true\n---\n\n{body}"
    (tmp_path / ".cursor-plugin").mkdir(parents=True)
    (tmp_path / ".cursor-plugin" / "plugin.json").write_text(json.dumps({"name": "demo"}), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "style.mdc").write_text(rule, encoding="utf-8")
    (tmp_path / "rules" / "manual.mdc").write_text("---\ndescription: Manual\n---\nOnly when asked.\n")

    cost = PluginSchemaValidator().validate(tmp_path).metadata["plugin"]["context_cost"]
    rows = _rows(cost)

    assert (cost["harness"], cost["load_mode"]) == ("cursor", "native")
    assert rows[("rule", "style.mdc")]["always_on_tokens"] == _tokens(len(body.strip()))
    assert rows[("rule", "style.mdc")]["on_demand_tokens"] == 0
    assert "alwaysApply" in rows[("rule", "style.mdc")]["basis"]
    assert "Tier 3 wrapper" not in rows[("rule", "style.mdc")]["basis"]
    assert rows[("rule", "manual.mdc")]["always_on_tokens"] == 0


def test_l29_cjk_text_is_not_estimated_as_four_characters_per_token(tmp_path: Path) -> None:
    """check-25 edge-05: a 225-character Chinese description is 172 o200k tokens; chars/4 said 59."""
    chinese = ("把逗号分隔的数据文件整理成带类型的结构化数据\uff0c键名使用小写下划线格式\u3002" * 7)[:225]
    english = ("The helper reads the named input, checks every field against the stated rules. " * 3)[:225]
    root = _plugin(
        tmp_path,
        {
            "skills/cjk-tidy/SKILL.md": f'---\nname: cjk-tidy\ndescription: "{chinese}"\n---\n# CJK\nBody\n',
            "skills/ascii-tidy/SKILL.md": f'---\nname: ascii-tidy\ndescription: "{english}"\n---\n# ASCII\nBody\n',
        },
    )
    rows = _rows(_cost(root))

    assert rows[("skill", "ascii-tidy")]["always_on_tokens"] == _tokens(len("ascii-tidy" + english))
    # About one token per CJK character: at least the o200k count, never the chars/4 figure.
    assert rows[("skill", "cjk-tidy")]["always_on_tokens"] >= 172


def test_l29_latin1_agent_is_counted_like_claude_code_lists_it(tmp_path: Path) -> None:
    """skeptic e01: Claude Code lists a Latin-1 agent (with replacement characters); it must cost tokens, not vanish."""
    legacy = (
        b'---\nname: legacy\ndescription: "R\xe9sum\xe9 helper for legacy files"\n---\n\nReads r\xe9sum\xe9 files.\n'
    )
    cost = _cost(_plugin(tmp_path, {"agents/legacy.md": legacy}))

    row = _rows(cost)[("agent", "legacy")]
    assert row["always_on_tokens"] == _tokens(len("R�sum� helper for legacy files"))
    assert not any("could not be read" in note for note in cost["notes"])


# --------------------------------------------------------------------------- #
# M33 (follow-up): rules the native adapters do not stage                     #
# --------------------------------------------------------------------------- #
_RULE_BODY = "Write short sentences and name the owner of every dataset. " * 18


def test_m33_native_rule_view_leaves_out_rules_the_adapters_do_not_stage(tmp_path: Path) -> None:
    """Verifier repro: an agent-requested .mdc rule and a paths rule were counted always-on (550 for about 280)."""
    root = _plugin(
        tmp_path,
        {
            "rules/plain.md": _RULE_BODY,
            "rules/requested.mdc": f"---\ndescription: Data rules\nalwaysApply: false\n---\n{_RULE_BODY}",
            "rules/scoped.md": f"---\npaths: ['**/*.py']\n---\n{_RULE_BODY}",
        },
    )
    cost = _cost(root)
    body = _tokens(len(_RULE_BODY.strip()))

    claude = _rows(cost)
    assert (cost["harness"], cost["load_mode"]) == ("claude-code", "native")
    assert claude[("rule", "plain.md")]["always_on_tokens"] == body
    assert claude[("rule", "requested.mdc")]["always_on_tokens"] == 0
    assert claude[("rule", "requested.mdc")]["basis"].startswith("not loaded:")
    assert "agent-requested" in claude[("rule", "requested.mdc")]["basis"]
    # Claude Code native stages a paths rule; it loads with matching files.
    assert claude[("rule", "scoped.md")]["always_on_tokens"] == 0
    assert claude[("rule", "scoped.md")]["on_demand_tokens"] == body
    assert cost["always_on_tokens"] == body

    inventory = build_plugin_inventory(
        root, {"name": "demo"}, contained=True, manifest_rel=".claude-plugin/plugin.json"
    )
    codex_cost = inventory.context_cost(harness="codex", load_mode="native")
    codex = _rows(codex_cost)
    assert codex[("rule", "plain.md")]["always_on_tokens"] == body
    assert codex[("rule", "requested.mdc")]["always_on_tokens"] == 0
    # Codex native has only an always-on rules channel, so a scoped rule is not staged at all.
    assert (codex[("rule", "scoped.md")]["always_on_tokens"], codex[("rule", "scoped.md")]["on_demand_tokens"]) == (
        0,
        0,
    )
    assert "not staged" in codex[("rule", "scoped.md")]["basis"]
    assert codex_cost["always_on_tokens"] == body
    assert _view(cost, "codex", "native")["always_on_tokens"] == body
    # The wrapper embeds every rule body, so wrapper views still count all three on demand.
    assert _view(cost, "codex", "wrapper")["on_demand_tokens"] == 3 * body


@pytest.mark.parametrize(
    ("value", "always_on"),
    [("true", True), ('"true"', True), ("yes", True), ("1", False), ('"yes"', False), ("false", False)],
)
def test_m33_always_apply_is_read_the_way_the_native_adapters_read_it(
    tmp_path: Path, value: str, always_on: bool
) -> None:
    """The adapters stage a rule only for YAML true or "true"; `alwaysApply: 1` was counted always-on."""
    cost = _cost(_plugin(tmp_path, {"rules/style.md": f"---\nalwaysApply: {value}\n---\n{_RULE_BODY}"}))

    row = _rows(cost)[("rule", "style.md")]
    assert (row["always_on_tokens"] > 0) is always_on
    if not always_on:
        assert row["basis"].startswith("not loaded:")


# --------------------------------------------------------------------------- #
# M33 (follow-up): SessionStart hooks that do not run at startup              #
# --------------------------------------------------------------------------- #
def _session_start(matchers: list[str | None]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for matcher in matchers:
        group: dict[str, Any] = {"hooks": [{"type": "command", "command": "cat ${CLAUDE_PLUGIN_ROOT}/brief.md"}]}
        if matcher is not None:
            group["matcher"] = matcher
        groups.append(group)
    return {"hooks": {"SessionStart": groups}}


_BRIEF = ("Project brief line with the house rules for every reply. " * 80)[:4_400]


def test_m33_compact_only_session_start_hook_is_not_first_turn_cost(tmp_path: Path) -> None:
    """Verifier repro: a `compact` SessionStart hook counted 1,100 always-on tokens it never adds at startup."""
    root = _plugin(tmp_path, {"hooks/hooks.json": _session_start(["compact"]), "brief.md": _BRIEF})
    cost = _cost(root)

    row = _rows(cost)[("hook", "hooks/hooks.json")]
    assert row["always_on_tokens"] == 0
    assert row["on_demand_tokens"] == _tokens(len(_BRIEF))
    assert "startup" in row["basis"]
    assert cost["always_on_tokens"] == 0
    assert cost["lower_bound"] is False


@pytest.mark.parametrize(
    ("matcher", "counted"),
    [
        (None, True),
        ("", True),
        ("*", True),
        ("startup", True),
        ("startup|resume", True),
        ("^start", True),
        ("compact", False),
        ("resume|clear", False),
        ("comp.*", False),
    ],
)
def test_m33_session_start_matcher_decides_first_turn_cost(tmp_path: Path, matcher: str | None, counted: bool) -> None:
    root = _plugin(tmp_path, {"hooks/hooks.json": _session_start([matcher]), "brief.md": _BRIEF})

    row = _rows(_cost(root))[("hook", "hooks/hooks.json")]
    assert row["always_on_tokens"] == (_tokens(len(_BRIEF)) if counted else 0)


def test_m33_only_the_startup_session_start_group_is_counted(tmp_path: Path) -> None:
    root = _plugin(tmp_path, {"hooks/hooks.json": _session_start(["startup", "compact"]), "brief.md": _BRIEF})

    row = _rows(_cost(root))[("hook", "hooks/hooks.json")]
    assert row["always_on_tokens"] == _tokens(len(_BRIEF))
    assert row["on_demand_tokens"] == _tokens(len(_BRIEF))


def test_m33_user_prompt_submit_ignores_a_matcher(tmp_path: Path) -> None:
    """Claude Code has no matcher for UserPromptSubmit: the handler runs for every prompt."""
    message = "Always answer in English."
    hooks = {
        "hooks": {
            "UserPromptSubmit": [{"matcher": "compact", "hooks": [{"type": "command", "command": f'echo "{message}"'}]}]
        }
    }
    row = _rows(_cost(_plugin(tmp_path, {"hooks/hooks.json": hooks})))[("hook", "hooks/hooks.json")]

    assert row["always_on_tokens"] == _tokens(len(message) + 1)
