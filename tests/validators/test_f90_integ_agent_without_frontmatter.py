# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M17 (check-06 edge-08): an agent without frontmatter inherits every tool, so it gets the same findings.

Claude Code loads an agent with no frontmatter under its file name, with
every field at its default, so it inherits every tool like an agent that
omits ``tools``. The inventory package fixed the broken-YAML agent of the
same example; the agent with no frontmatter still got no finding because the
privilege analysis returned early.
"""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.models.result import Severity
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_AGENTS = {
    "omits.md": "---\nname: omits\ndescription: Control. Frontmatter parses and has no tools line.\n---\nDo the task.\n",
    "broken-yaml.md": (
        "---\nname: broken-yaml\ndescription: The unquoted star below is a YAML alias.\ntools: *\n---\nHelp.\n"
    ),
    "plain.md": "You help with anything. This agent file has no frontmatter.\n",
}


def _plugin(root: Path) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "c06-unparsed", "version": "1.0.0", "description": "Check 6 example."}), encoding="utf-8"
    )
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"db": {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/server.js"]}}}),
        encoding="utf-8",
    )
    (root / "server.js").write_text("console.log('ok');\n", encoding="utf-8")
    (root / "agents").mkdir()
    for name, text in _AGENTS.items():
        (root / "agents" / name).write_text(text, encoding="utf-8")
    return root


def test_every_agent_that_inherits_all_tools_is_flagged(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "plugin"))

    flagged: dict[str, set[tuple[str, Severity]]] = {}
    for finding in result.findings:
        if finding.check_name.startswith("plugin_agent_"):
            agent = finding.message.split("'")[1]
            flagged.setdefault(agent, set()).add((finding.check_name, finding.severity))

    expected = {("plugin_agent_unrestricted_bash", Severity.LOW), ("plugin_agent_inherits_all_tools", Severity.MEDIUM)}
    assert flagged["omits"] == expected
    assert flagged["plain"] == expected
    # Claude Code retries broken frontmatter with special values quoted, so 'tools: *' reads as a wildcard grant.
    assert flagged["broken-yaml"] == {("plugin_agent_wildcard_tools", Severity.MEDIUM)}
    plain = next(f for f in result.findings if "'plain'" in f.message)
    assert "has no frontmatter" in plain.message
