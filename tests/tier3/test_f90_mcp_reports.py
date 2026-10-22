# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 24 reports: failed and unknown MCP calls are visible, with per-tool counts.

Shapes follow the proof's check-24 p03 case (``MCP calls 100% succeeded (1/3)``
with two unknown calls) and G1 (tools listed by name only).
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from skillevaluator.reporting.cli import print_plugin_tier3
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.reporting.plugin_sections import tier3_plugin_view
from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals, summarize_plugin_signals

DECLARED = {"skill": [], "mcp": ["reltools"]}


def _call(call_id: str, tool: str, content: str | None, **extra: Any) -> dict[str, Any]:
    step: dict[str, Any] = {
        "source": "agent",
        "tool_calls": [{"tool_call_id": call_id, "function_name": f"mcp__reltools__{tool}", "arguments": {}}],
    }
    if content is not None:
        result: dict[str, Any] = {"source_call_id": call_id, "content": content}
        if extra:
            result["extra"] = extra
        step["observation"] = {"results": [result]}
    return step


def _summary() -> dict[str, Any]:
    trajectory = {
        "agent": {"name": "claude-code"},
        "steps": [
            {"source": "user", "message": "Release atlas."},
            _call("t1", "list_changes", '{"changes": []}'),
            _call("t2", "list_changes", "unknown tag", tool_result_is_error=True),
            _call("t3", "stage_release", None),
            _call("t4", "compute_version", '{"next": "1.5.0"}'),
        ],
    }
    signals = compute_plugin_signals(trajectory, {}, declared=DECLARED)
    assert signals is not None
    return summarize_plugin_signals([signals])


def test_summary_keeps_per_tool_counts_and_server_rates() -> None:
    summary = _summary()
    server = summary["mcp_calls"]["by_server"]["reltools"]
    assert (server["total"], server["succeeded"], server["failed"], server["unknown"]) == (4, 2, 1, 1)
    assert server["success_rate"] == round(2 / 3, 4) or abs(server["success_rate"] - 2 / 3) < 1e-3
    by_tool = server["by_tool"]
    assert by_tool["mcp__reltools__list_changes"]["succeeded"] == 1
    assert by_tool["mcp__reltools__list_changes"]["failed"] == 1
    assert by_tool["mcp__reltools__stage_release"]["unknown"] == 1


def _view() -> dict[str, Any]:
    payload = {
        "eval_target": {"kind": "plugin"},
        "agents": {"codex": {"plugin_signals_summary": {"with_skill": _summary()}}},
    }
    view = tier3_plugin_view(payload)
    assert view is not None
    return view


def test_cli_shows_failed_and_unknown_calls_and_per_tool_counts() -> None:
    console = Console(record=True, width=240, color_system=None)
    print_plugin_tier3(_view(), console)
    plain = " ".join(console.export_text().split())
    assert "MCP calls 67% succeeded (2/4; 1 failed, 1 unknown)" in plain
    assert "mcp__reltools__list_changes (1/2 succeeded, 1 failed)" in plain
    assert "mcp__reltools__stage_release (0/1 succeeded, 1 unknown)" in plain


def test_markdown_shows_mcp_call_outcomes() -> None:
    lines: list[str] = []
    MarkdownReporter._render_tier3_plugin(_view(), lines)
    markdown = "\n".join(lines)
    assert "### MCP Calls (advisory)" in markdown
    assert "| codex · Plugin | all servers | 4 | 2 | 1 | 1 | 67% |" in markdown
    # Markdown escapes the double underscores of the tool label.
    assert r"mcp\_\_reltools\_\_list_changes (1/2 succeeded, 1 failed)" in markdown
