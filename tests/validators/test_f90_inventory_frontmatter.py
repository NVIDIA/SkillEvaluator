# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontmatter that strict YAML rejects is read the way Claude Code reads it.

Claude Code retries a failed frontmatter parse after quoting plain values that
hold a YAML indicator or ``": "``. A command whose description reads
``Deploy: runs the deploy script.`` still gets ``allowed-tools: Bash``
pre-approved (proof M17, skeptic live probe ``zz-lenient-cmd``), an agent with
``tools: *`` still gets every tool (check-06 edge-08), and such descriptions
still cost always-on context (proof L29, check-25 edge-02).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.plugin_components import parse_markdown, plugin_inventory_for_root
from skillevaluator.validators.plugin_schema import PluginSchemaValidator


def _write(root: Path, files: dict[str, str | dict]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _findings(result: ValidationResult, check_name: str) -> list:
    return [finding for finding in result.findings if finding.check_name == check_name]


@pytest.mark.parametrize(
    ("frontmatter", "expected"),
    [
        ("description: Deploy: runs the deploy script.\nallowed-tools: Bash", {"allowed-tools": "Bash"}),
        ("name: broken-yaml\ntools: *", {"tools": "*"}),
        (
            "description: Use for questions: retention, owners, labels",
            {"description": "Use for questions: retention, owners, labels"},
        ),
        ('description: say "hi": now', {"description": 'say "hi": now'}),
    ],
    ids=["colon-in-description", "star-tools", "colon-cost", "quotes-escaped"],
)
def test_frontmatter_rejected_by_strict_yaml_is_read_like_claude_code(frontmatter: str, expected: dict) -> None:
    parsed = parse_markdown(f"---\n{frontmatter}\n---\nBody.\n")
    for key, value in expected.items():
        assert parsed.frontmatter[key] == value


def test_valid_frontmatter_and_flow_lists_are_unchanged() -> None:
    parsed = parse_markdown("---\nname: x\ntools: [Read, Grep]\ndescription: 'quoted: text'\n---\nBody.\n")
    assert parsed.frontmatter == {"name": "x", "tools": ["Read", "Grep"], "description": "quoted: text"}


def test_frontmatter_that_still_fails_gives_no_fields() -> None:
    parsed = parse_markdown("---\nname: x\n  - bad: [\n---\nBody.\n")
    assert parsed.frontmatter == {}
    assert parsed.body == "Body."


def test_lenient_command_grant_is_flagged(tmp_path: Path) -> None:
    """Skeptic zz-lenient-cmd: Claude Code pre-approved Bash live; Tier 1 passed with no finding."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "c06-lencmd", "version": "1.0.0"},
            "commands/lencmd.md": "---\ndescription: Deploy: runs the deploy script.\nallowed-tools: Bash\n---\nRun.\n",
            "agents/lenagent.md": (
                "---\nname: lenagent\ndescription: Review: checks code and fixes it.\ntools: Bash\n"
                "permissionMode: bypassPermissions\n---\nDo the task.\n"
            ),
        },
    )

    result = PluginSchemaValidator().validate(root)
    [command] = _findings(result, "plugin_command_unrestricted_bash")
    assert command.severity == Severity.HIGH
    assert _findings(result, "plugin_agent_bypass_permissions")
    assert not result.passed


def test_unquoted_star_tools_agent_is_a_wildcard_grant(tmp_path: Path) -> None:
    """Check-06 edge-08: 'tools: *' was read as no frontmatter at all."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "edge08", "version": "1.0.0"},
            "agents/broken-yaml.md": "---\nname: broken-yaml\ndescription: Edits the database.\ntools: *\n---\nx\n",
        },
    )

    result = PluginSchemaValidator().validate(root)
    [wildcard] = _findings(result, "plugin_agent_wildcard_tools")
    assert "broken-yaml" in wildcard.message


def test_description_with_colon_counts_toward_always_on_cost(tmp_path: Path) -> None:
    """Check-25 edge-02: 0 always-on, while Claude Code 2.1.284 lists both descriptions."""
    description = "Answers data-catalog questions: retention, owners, labels, and lineage for each table"
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "edge02", "version": "1.0.0"},
            "skills/broken-skill/SKILL.md": f"---\nname: broken-skill\ndescription: {description}\n---\nBody.\n",
            "agents/steward.md": f"---\nname: steward\ndescription: {description}\n---\nBody.\n",
        },
    )

    inventory = plugin_inventory_for_root(root)
    assert inventory is not None
    rows = {(row["type"], row["name"]): row for row in inventory.context_cost()["by_component"]}
    assert rows[("skill", "broken-skill")]["always_on_tokens"] == -(-(len("broken-skill") + len(description)) // 4)
    assert rows[("agent", "steward")]["always_on_tokens"] == -(-len(description) // 4)
