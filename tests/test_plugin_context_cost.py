# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context estimates preserve native client semantics without loading a runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def inventory(root: Path, manifest: str, data: dict, files: dict):
    from skillevaluator.plugin_inventory import plugin_inventory_for_root

    for name, value in {manifest: json.dumps(data), **files}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    result = plugin_inventory_for_root(root)
    assert result is not None
    return result


def test_english_and_wide_character_estimation():
    from skillevaluator.plugin_context_cost import cost_chars, estimate_tokens

    assert cost_chars("abcd") == 4
    assert cost_chars("中文") == 8
    assert estimate_tokens(5) == 2
    assert estimate_tokens(-5) == -2


def test_skill_listing_body_and_mcp_unknown_schema_are_separate(tmp_path):
    result = inventory(
        tmp_path,
        ".claude-plugin/plugin.json",
        {"name": "demo", "mcpServers": {"s": {"command": "python"}}},
        {
            "skills/demo/SKILL.md": "---\nname: demo\ndescription: abcd\n---\n" + "body" * 10,
        },
    ).context_cost()
    skill = next(row for row in result["by_component"] if row["type"] == "skill")
    assert skill["always_on_tokens"] == 2
    assert skill["on_demand_tokens"] == 10
    assert result["lower_bound"] and result["not_counted"] == ["mcp s"]
    assert {(v["harness"], v["load_mode"]) for v in result["by_harness"]} >= {
        ("codex", "native"),
        ("claude-code", "wrapper"),
    }


def test_model_hidden_is_client_specific(tmp_path):
    inv = inventory(
        tmp_path,
        ".claude-plugin/plugin.json",
        {"name": "demo"},
        {
            "skills/demo/SKILL.md": "---\nname: demo\ndescription: abcd\ndisable-model-invocation: true\n---\nbody",
        },
    )
    claude = inv.context_cost(harness="claude-code")
    codex = inv.context_cost(harness="codex")
    assert claude["always_on_tokens"] == 0 and claude["on_demand_tokens"] == 3
    assert codex["always_on_tokens"] == 2 and codex["on_demand_tokens"] == 1


@pytest.mark.parametrize("frontmatter", ["paths: ['src/**']", "globs: 'src/**'", "alwaysApply: false"])
def test_scoped_or_requested_rules_follow_native_and_wrapper_views(tmp_path, frontmatter):
    inv = inventory(
        tmp_path,
        ".claude-plugin/plugin.json",
        {"name": "demo"},
        {"rules/team.md": f"---\n{frontmatter}\n---\nabcdefgh"},
    )
    assert inv.context_cost(harness="codex")["always_on_tokens"] == 0
    assert inv.context_cost(harness="codex")["on_demand_tokens"] == 0
    assert inv.context_cost(harness="codex", load_mode="wrapper")["on_demand_tokens"] == 2


def test_forced_styles_replace_coding_instructions_and_only_first_applies(tmp_path):
    inv = inventory(
        tmp_path,
        ".claude-plugin/plugin.json",
        {"name": "demo"},
        {
            "output-styles/a.md": "---\nname: a\nforce-for-plugin: true\n---\nbe brief",
            "output-styles/b.md": "---\nname: b\nforce-for-plugin: true\n---\nbe long",
        },
    )
    rows = inv.context_cost()["by_component"]
    assert rows[0]["always_on_tokens"] < 0
    assert rows[1]["always_on_tokens"] == 0 and rows[1]["on_demand_tokens"] > 0
    assert inv.context_cost(harness="codex")["always_on_tokens"] == 0


def test_first_turn_hook_literals_and_unmeasured_dynamic_output(tmp_path):
    inv = inventory(
        tmp_path,
        ".claude-plugin/plugin.json",
        {"name": "demo"},
        {
            "hooks/hooks.json": json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {"matcher": "startup", "hooks": [{"type": "command", "command": "echo abcd"}]}
                        ],
                        "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python dynamic.py"}]}],
                    }
                }
            ),
        },
    )
    cost = inv.context_cost()
    assert cost["always_on_tokens"] == 2
    assert cost["lower_bound"] and cost["not_counted"]
    assert inv.context_cost(harness="codex")["always_on_tokens"] == 0


def test_late_session_start_output_is_on_demand(tmp_path):
    inv = inventory(
        tmp_path,
        ".claude-plugin/plugin.json",
        {"name": "demo"},
        {
            "hooks/hooks.json": json.dumps(
                {
                    "hooks": {
                        "SessionStart": [{"matcher": "resume", "hooks": [{"type": "command", "command": "echo abcd"}]}]
                    }
                }
            ),
        },
    )
    assert inv.context_cost()["always_on_tokens"] == 0
    assert inv.context_cost()["on_demand_tokens"] == 2


def test_cursor_selected_native_cost_and_unsupported_inventory_facts(tmp_path):
    inv = inventory(
        tmp_path,
        ".cursor-plugin/plugin.json",
        {"name": "demo"},
        {"rules/team.mdc": "---\nalwaysApply: false\n---\nabcdefgh"},
    )
    cost = inv.context_cost()
    assert cost["harness"] == "cursor" and cost["on_demand_tokens"] == 2
    assert any(v["harness"] == "cursor" for v in cost["by_harness"])


def test_inline_command_content_keeps_its_literal_frontmatter_text(tmp_path):
    content = "---\nname: inside\ndescription: literal\n---\nbody"
    inv = inventory(
        tmp_path,
        ".claude-plugin/plugin.json",
        {"name": "demo", "commands": {"inline": {"description": "abcd", "content": content}}},
        {},
    )
    row = next(row for row in inv.context_cost()["by_component"] if row["type"] == "command")
    assert row["always_on_tokens"] == 1
    assert row["on_demand_tokens"] == (len(content) + 3) // 4


def test_known_hook_output_is_captured_with_inventory_instead_of_reread(tmp_path):
    inv = inventory(
        tmp_path,
        ".claude-plugin/plugin.json",
        {"name": "demo"},
        {
            "hooks/hooks.json": json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {"hooks": [{"type": "command", "command": 'cat "${CLAUDE_PLUGIN_ROOT}/context.txt"'}]}
                        ]
                    }
                }
            ),
            "context.txt": "abcd",
        },
    )
    original = inv.context_cost()
    (tmp_path / "context.txt").write_text("x" * 400)
    assert inv.context_cost() == original
