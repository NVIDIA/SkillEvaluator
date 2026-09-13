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
    synthetic_trajectory_from_codex_txt,
    synthetic_trajectory_from_cursor_cli,
    synthetic_trajectory_from_opencode_json,
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


def test_opencode_json_tool_use_and_read():
    log = (
        '{"type":"text","part":{"type":"text","text":"Reading skill"}}\n'
        '{"type":"tool_use","part":{"type":"tool","tool":"read","callID":"call-1",'
        '"state":{"status":"completed","input":{"filePath":"/workspace/skills/calculator/SKILL.md"},'
        '"output":"# Calculator skill"}}}\n'
        '{"type":"tool_use","part":{"type":"tool","tool":"bash","callID":"call-2",'
        '"state":{"status":"completed","input":{"command":"cat skills/foo/SKILL.md"},'
        '"output":"ok","metadata":{"exit":0}}}}\n'
    )
    traj = synthetic_trajectory_from_opencode_json(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 2
    read_call = next(tc for tc in tcs if tc["action"] == "read")
    assert "/calculator/" in json.dumps(read_call["action_input"])
    assert "Calculator" in read_call["observation"]
    bash_call = next(tc for tc in tcs if tc["action"] == "bash")
    assert "cat" in json.dumps(bash_call["action_input"]).lower()


def test_opencode_error_only_log_returns_none():
    log = json.dumps(
        {
            "type": "error",
            "error": {
                "name": "UnknownError",
                "data": {"message": "ResourceExhausted: Worker local total request limit reached"},
            },
        }
    )
    assert synthetic_trajectory_from_opencode_json(log + "\n") is None


def test_opencode_tool_error_uses_state_error_observation():
    log = (
        '{"type":"tool_use","part":{"type":"tool","tool":"bash","callID":"call-err",'
        '"state":{"status":"error","input":{"command":"missing-cmd"},'
        '"error":"command not found: missing-cmd"}}}\n'
    )
    traj = synthetic_trajectory_from_opencode_json(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    assert "missing-cmd" in tcs[0]["observation"]


def test_codex_thread_event_agent_message_and_command_execution():
    log = (
        '{"type":"item","item":{"type":"agent_message","id":"msg-1",'
        '"content":[{"type":"text","text":"Reading skill file"}],'
        '"function_call":{"name":"shell","arguments":{"command":"ls"}},'
        '"output":"demo"},"item.completed":true}\n'
        '{"type":"item","item":{"type":"agent_message","id":"msg-2",'
        '"content":[{"type":"text","text":"Done"}]},'
        '"item.completed":true}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    assert traj["schema_version"] == "ATIF-v1.2-synthetic-codex-log"
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    assert tcs[0]["action"] == "bash"


def test_codex_exec_json_tolerates_stderr_noise():
    log = (
        "ERROR responses_websocket: HTTP error: 405 Method Not Allowed\n"
        '{"type":"item","item":{"type":"agent_message","id":"msg-1",'
        '"content":[{"type":"text","text":"hello"}]},'
        '"item.completed":true}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    assert len(traj["steps"]) == 1


def test_codex_opencode_shaped_jsonl_fallback():
    log = (
        '{"type":"tool_use","part":{"type":"tool","tool":"read","callID":"c1",'
        '"state":{"status":"completed","input":{"path":"/workspace/skills/demo/SKILL.md"},'
        '"output":"# Demo"}}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    assert traj["schema_version"] == "ATIF-v1.2-synthetic-codex-log"
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    assert tcs[0]["action"] == "read"


def test_codex_file_change_emits_write_calls():
    log = (
        '{"type":"item.completed","item":{"type":"file_change","id":"fc-1",'
        '"changes":[{"path":"/workspace/output/real.py","kind":"add"}],'
        '"status":"completed"}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    assert tcs[0]["action"] == "write"
    assert tcs[0]["action_input"]["path"] == "/workspace/output/real.py"


def test_codex_mcp_tool_call_preserves_result():
    log = (
        '{"type":"item.completed","item":{"type":"mcp_tool_call","id":"mcp-1",'
        '"server":"demo","tool":"lookup","arguments":{"query":"skill"},'
        '"result":{"content":[{"type":"text","text":"found skill"}]},'
        '"status":"completed"}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    assert tcs[0]["action"] == "lookup"
    assert "found skill" in tcs[0]["observation"]


def test_codex_command_execution_preserves_failed_terminal_evidence():
    log = (
        '{"type":"item.completed","item":{"type":"command_execution","id":"cmd-1",'
        '"command":false,"exit_code":1,"status":"failed","aggregated_output":""}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    obs = tcs[0]["observation"]
    assert "status=failed" in obs
    assert "exit_code=1" in obs


def test_codex_file_change_preserves_status_and_error():
    log = (
        '{"type":"item.completed","item":{"type":"file_change","id":"fc-1",'
        '"changes":[{"path":"/workspace/output/real.py","kind":"add"}],'
        '"status":"failed","error":{"message":"disk full"}}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    obs = tcs[0]["observation"]
    assert "status=failed" in obs
    assert "disk full" in obs


def test_codex_mcp_tool_call_string_error():
    log = (
        '{"type":"item.completed","item":{"type":"mcp_tool_call","id":"mcp-2",'
        '"tool":"lookup","arguments":{},"error":"server unavailable","status":"failed"}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    obs = tcs[0]["observation"]
    assert "server unavailable" in obs
    assert "status=failed" in obs


def test_opencode_tool_error_retained_when_output_fills_budget():
    long_output = "o" * 9000
    log = (
        '{"type":"tool_use","part":{"type":"tool","tool":"bash","callID":"c-long",'
        f'"state":{{"status":"error","output":"{long_output}",'
        '"error":{"message":"command aborted"}}}}\n'
    )
    traj = synthetic_trajectory_from_opencode_json(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    obs = tcs[0]["observation"]
    assert "status=error" in obs
    assert "command aborted" in obs
    assert len(obs) <= 8000


def test_codex_multi_file_change_terminal_evidence_on_each_write():
    log = (
        '{"type":"item.completed","item":{"type":"file_change","id":"fc-2",'
        '"changes":[{"path":"/workspace/a.py","kind":"add"},'
        '{"path":"/workspace/b.py","kind":"add"}],'
        '"status":"failed","error":{"message":"disk full"}}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 2
    for call in tcs:
        assert "status=failed" in call["observation"]
        assert "disk full" in call["observation"]


def test_opencode_tool_error_retains_status_with_partial_output():
    log = (
        '{"type":"tool_use","part":{"type":"tool","tool":"bash","callID":"c-err",'
        '"state":{"status":"error","output":"partial log line",'
        '"error":{"message":"command aborted"}}}}\n'
    )
    traj = synthetic_trajectory_from_opencode_json(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    obs = tcs[0]["observation"]
    assert "status=error" in obs
    assert "partial log line" in obs
    assert "command aborted" in obs


def test_codex_plain_text_errors_return_none():
    text = "ERROR responses_websocket: HTTP error: 405 Method Not Allowed\n"
    assert synthetic_trajectory_from_codex_txt(text) is None


def test_load_falls_back_to_opencode_txt(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    (logs / "opencode.txt").write_text(
        '{"type":"tool_use","part":{"type":"tool","tool":"bash","callID":"c1",'
        '"state":{"status":"completed","input":{"command":"echo hi"},"output":"hi"}}}\n',
        encoding="utf-8",
    )
    traj_path = logs / "trajectory.json"
    data, meta = load_trajectory_with_fallback(traj_path, logs)
    assert data is not None
    assert meta["source"] == "opencode.txt"


def test_opencode_non_dict_tool_input_is_not_treated_as_shell_command():
    log = (
        '{"type":"tool_use","part":{"type":"tool","tool":"bash","callID":"c1",'
        '"state":{"status":"completed","input":"cat /etc/passwd","output":"blocked"}}}\n'
    )
    traj = synthetic_trajectory_from_opencode_json(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert tcs[0]["action_input"] == {}
