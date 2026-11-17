# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain native Codex MCP server identity for downstream checks."""

import json

import pytest

from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts
from skillevaluator.tier3.eval_core.checks import check_tool_efficiency
from skillevaluator.tier3.eval_core.log_converters import (
    load_trajectory_with_fallback,
    synthetic_trajectory_from_codex_txt,
)
from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals


def _call(server, tool="lookup", arguments=None):
    event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "id": "mcp-1",
            "server": server,
            "tool": tool,
            "arguments": arguments or {},
            "result": {"content": [{"type": "text", "text": "recorded result"}]},
            "status": "completed",
        },
    }
    trajectory = synthetic_trajectory_from_codex_txt(json.dumps(event))
    assert trajectory is not None
    return extract_tool_calls_as_dicts(trajectory)[0]


@pytest.mark.parametrize("server", ["reference", "project.docs"])
@pytest.mark.parametrize("tool", ["lookup", "task"])
def test_codex_mcp_server_and_tool_are_preserved_and_counted(server, tool):
    call = _call(server, tool, {"query": "requirements"})

    assert call["action"] == f"mcp__{server}__{tool}"
    assert call["action_input"] == {"query": "requirements"}
    assert "recorded result" in call["observation"]
    result = check_tool_efficiency([call])
    assert result["score"] == 1.0
    assert result["details"]["productive"] == 1
    assert result["details"]["neutral"] == 0


@pytest.mark.parametrize(
    ("server", "tool", "path", "expected"),
    [
        ("filesystem", "inspect", "/", 0.0),
        ("files", "list_directory", "/", 0.0),
        ("filesystem", "inspect", "/workspace/project", 1.0),
        ("github", "get_file_contents", "/", 1.0),
    ],
)
def test_codex_mcp_identity_retains_existing_root_search_policy(server, tool, path, expected):
    assert check_tool_efficiency([_call(server, tool, {"path": path})])["score"] == expected


@pytest.mark.parametrize("server", [None, "", 5])
def test_missing_or_invalid_server_does_not_invent_identity(server):
    assert _call(server)["action"] == "lookup"


def test_already_qualified_tool_name_is_not_prefixed_twice():
    assert _call("docs", "mcp__docs__lookup")["action"] == "mcp__docs__lookup"


def test_tool_name_cannot_replace_its_recorded_server_identity():
    call = _call("filesystem", "mcp__docs__lookup", {"path": "/"})

    assert call["action"] == "mcp__filesystem__mcp__docs__lookup"
    assert check_tool_efficiency([call])["score"] == 0.0


def _load_native_mcp_log(tmp_path, server, tool, path):
    logs = tmp_path / "agent"
    logs.mkdir()
    event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "id": "mcp-1",
            "server": server,
            "tool": tool,
            "arguments": {"path": path},
            "result": {"content": [{"type": "text", "text": "recorded result"}]},
            "status": "completed",
        },
    }
    (logs / "codex.txt").write_text(json.dumps(event) + "\n", encoding="utf-8")
    trajectory, metadata = load_trajectory_with_fallback(logs / "trajectory.json", logs)
    assert trajectory is not None
    assert metadata["source"] == "codex.txt"
    return trajectory


@pytest.mark.parametrize(
    ("server", "tool", "path", "expected"),
    [
        ("project__filesystem", "inspect", "/", 0.0),
        ("project.filesystem", "inspect", "/", 0.0),
        ("project__filesystem", "mcp__project__filesystem__inspect", "/", 0.0),
        ("project__filesystem", "mcp__docs__inspect", "/", 0.0),
        ("project__filesystem", "inspect", "/workspace/project", 1.0),
        ("project", "filesystem__inspect", "/", 1.0),
        ("project__docs", "inspect__list_directory", "/", 0.0),
    ],
)
def test_native_log_keeps_server_boundaries_for_efficiency(tmp_path, server, tool, path, expected):
    trajectory = _load_native_mcp_log(tmp_path, server, tool, path)
    calls = extract_tool_calls_as_dicts(trajectory)
    result = check_tool_efficiency(calls)

    assert result["score"] == expected
    assert result["details"]["wasted"] == int(expected == 0)
    assert result["details"]["productive"] == int(expected == 1)


@pytest.mark.parametrize(
    ("server", "tool"), [("project__filesystem", "inspect"), ("project", "filesystem__inspect")]
)
def test_native_log_keeps_ambiguous_labels_attributed_to_recorded_server(tmp_path, server, tool):
    trajectory = _load_native_mcp_log(tmp_path, server, tool, "/")
    calls = extract_tool_calls_as_dicts(trajectory)
    assert calls[0]["action"] == "mcp__project__filesystem__inspect"

    signals = compute_plugin_signals(trajectory, {}, declared={"mcp": ["project", "project__filesystem"]})
    assert signals["activations"][0]["server"] == server
    assert signals["activations"][0]["name"] == server
    assert calls[0]["extra"] == {"mcp_server": server, "mcp_tool": tool}


@pytest.mark.parametrize(
    "extra",
    [None, [], {}, {"mcp_server": 7, "mcp_tool": "inspect"}, {"mcp_server": "docs", "mcp_tool": "inspect"}],
)
def test_invalid_or_mismatched_mcp_metadata_preserves_legacy_name_policy(extra):
    call = {"action": "mcp__filesystem__inspect", "action_input": {"path": "/"}, "extra": extra}

    assert check_tool_efficiency([call])["score"] == 0.0
