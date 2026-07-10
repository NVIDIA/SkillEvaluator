# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for synthetic ATIF reconstruction from agent logs."""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts
from skillevaluator.tier3.eval_core.log_converters import (
    load_trajectory_with_fallback,
    synthetic_trajectory_from_claude_stream_jsonl,
    synthetic_trajectory_from_cursor_cli,
)


def test_claude_stream_jsonl_tool_use_and_result():
    log = (
        '{"type":"assistant","message":{"content":['
        '{"type":"text","text":"Reading skill"},'
        '{"type":"tool_use","id":"tu1","name":"Read","input":{"file_path":"/workspace/skills/calculator/SKILL.md"}}'
        "]}}\n"
        '{"type":"user","message":{"content":['
        '{"type":"tool_result","tool_use_id":"tu1","content":"# Calculator skill"}'
        "]}}\n"
    )
    traj = synthetic_trajectory_from_claude_stream_jsonl(log)
    assert traj is not None
    assert len(traj["steps"]) == 1
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    assert tcs[0]["action"] == "Read"
    assert "/calculator/" in json.dumps(tcs[0]["action_input"])
    assert "Calculator" in tcs[0]["observation"]


def test_cursor_cli_heuristic_extracts_cat_and_read_path():
    text = """
Here is the plan.
`read_file('/workspace/skills/calculator/SKILL.md')`
$ cat /workspace/skills/calculator/SKILL.md
"""
    traj = synthetic_trajectory_from_cursor_cli(text)
    assert traj is not None
    steps = traj["steps"]
    assert len(steps) == 1
    tcs = extract_tool_calls_as_dicts(traj)
    actions = {tc["action"].lower() for tc in tcs}
    assert "read" in actions or "bash" in actions


def test_load_prefers_trajectory_json(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    traj_path = logs / "trajectory.json"
    traj_path.write_text(
        json.dumps({"steps": [{"source": "agent", "message": "p", "tool_calls": []}]}),
        encoding="utf-8",
    )
    data, meta = load_trajectory_with_fallback(traj_path, logs)
    assert data is not None
    assert meta["source"] == "trajectory.json"


def test_load_falls_back_to_cursor_txt(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    (logs / "cursor-cli.txt").write_text("Ran: cat skills/foo/SKILL.md\n", encoding="utf-8")
    traj_path = logs / "trajectory.json"
    data, meta = load_trajectory_with_fallback(traj_path, logs)
    assert data is not None
    assert meta["source"] == "cursor-cli.txt"


def test_load_prefers_claude_log_over_cursor_when_both(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    claude = (
        '{"type":"assistant","message":{"content":['
        '{"type":"tool_use","id":"a","name":"Bash","input":{"command":"echo hi"}}'
        "]}}\n"
    )
    (logs / "claude-code.txt").write_text(claude, encoding="utf-8")
    (logs / "cursor-cli.txt").write_text("cursor only\n", encoding="utf-8")
    traj_path = logs / "trajectory.json"
    data, meta = load_trajectory_with_fallback(traj_path, logs)
    assert meta["source"] == "claude-code.txt"
    tcs = extract_tool_calls_as_dicts(data)
    assert any(tc["action"] == "Bash" for tc in tcs)
