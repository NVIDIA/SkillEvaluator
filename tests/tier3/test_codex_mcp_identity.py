# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain native Codex MCP server identity for downstream checks."""

import json

import pytest

from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts
from skillevaluator.tier3.eval_core.checks import check_tool_efficiency
from skillevaluator.tier3.eval_core.log_converters import synthetic_trajectory_from_codex_txt


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
