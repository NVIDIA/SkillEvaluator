# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Tier 1 static context-cost block says what it covers (check 25).

A plugin with a skill, an MCP server, and a SessionStart hook whose output is
not readable statically (the check-25 tier3-08 blind spots): HTML and Markdown
must name the harness and load mode, say the always-on total is a lower bound,
and show the per-harness estimates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.reporting.sarif_reporter import SARIFReporter
from skillevaluator.validators.plugin_schema import PluginSchemaValidator


def _plugin(root: Path) -> Path:
    files = {
        ".claude-plugin/plugin.json": {"name": "demo"},
        "skills/csv-tidy/SKILL.md": "---\nname: csv-tidy\ndescription: Convert CSV into typed JSON\n---\nBody\n",
        ".mcp.json": {"mcpServers": {"reltools": {"command": "python3", "args": ["server.py"]}}},
        "hooks/hooks.json": {
            "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "./scripts/brief.sh"}]}]}
        },
        "scripts/brief.sh": "#!/bin/sh\ncat brief.md\n",
    }
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _section(html: str) -> str:
    start = html.index('id="plugin-context-cost"')
    end = html.find('<div class="plugin-subsection" id=', start)
    return " ".join(re.sub(r"<[^>]+>", " ", html[start : end if end > 0 else len(html)]).split())


def test_m33_static_block_names_its_scope_and_lower_bound(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(_plugin(tmp_path))

    html = HTMLReporter(include_timestamp=False).render_all([result])
    markdown = MarkdownReporter(include_timestamp=False).render_all([result])

    section = _section(html)
    assert "Claude Code (native)" in section
    assert "lower bound" in section.lower()
    assert "reltools" in section
    for scope in ("Claude Code (wrapper)", "Codex (native)", "Codex (wrapper)"):
        assert scope in section
    md_block = markdown[markdown.index("### Context cost") :]
    md_block = md_block[: md_block.index("\n### ", 4) if "\n### " in md_block[4:] else len(md_block)]
    assert "Claude Code (native)" in md_block
    assert "lower bound" in md_block.lower()


def test_m33_sarif_context_cost_names_its_scope_and_lower_bound(tmp_path: Path) -> None:
    """Verifier: SARIF carried only method, estimator and the two totals, so a lower bound read as exact."""
    result = PluginSchemaValidator().validate(_plugin(tmp_path))

    document = json.loads(SARIFReporter(include_timestamp=False).render_all([result]))
    cost = document["runs"][0]["properties"]["plugin"]["contextCost"]

    assert cost["harness"] == "claude-code"
    assert cost["loadMode"] == "native"
    assert cost["lowerBound"] is True
    assert "mcp reltools" in cost["notCounted"]
    assert cost["alwaysOnTokens"] == result.metadata["plugin"]["context_cost"]["always_on_tokens"]
