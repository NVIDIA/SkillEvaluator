# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for synthetic ATIF reconstruction from agent logs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts
from skillevaluator.tier3.eval_core.checks import check_error_recovery
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import normalized_tool_call_error_flag
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
    assert "exit_code=0" in bash_call["observation"]


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


def test_codex_web_search_multi_query_preserves_action_payload():
    # Schema-valid WebSearchItem: required top-level query, action.search with queries.
    log = (
        '{"type":"item.completed","item":{"type":"web_search","id":"ws-1",'
        '"query":"parallel query batch",'
        '"action":{"type":"search","queries":["alpha","beta"]}}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    assert tcs[0]["action"] == "web_search"
    assert tcs[0]["action_input"] == {
        "query": "parallel query batch",
        "action": {"type": "search", "queries": ["alpha", "beta"]},
    }


def test_codex_web_search_open_page_action_preserves_url():
    log = (
        '{"type":"item.completed","item":{"type":"web_search","id":"ws-2",'
        '"query":"https://example.com/docs",'
        '"action":{"type":"open_page","url":"https://example.com/docs"}}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert tcs[0]["action_input"] == {
        "query": "https://example.com/docs",
        "action": {"type": "open_page", "url": "https://example.com/docs"},
    }


def test_codex_web_search_find_in_page_preserves_pattern():
    log = (
        '{"type":"item.completed","item":{"type":"web_search","id":"ws-3",'
        '"query":"\'install\' in https://example.com/docs",'
        '"action":{"type":"find_in_page","url":"https://example.com/docs","pattern":"install"}}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert tcs[0]["action_input"]["action"] == {
        "type": "find_in_page",
        "url": "https://example.com/docs",
        "pattern": "install",
    }


def test_codex_collab_tool_call_spawn_agent_preserves_wire_fields():
    log = (
        '{"type":"item.completed","item":{"type":"collab_tool_call","id":"col-1",'
        '"tool":"spawn_agent","sender_thread_id":"thread-parent",'
        '"receiver_thread_ids":["thread-child"],"prompt":"review logs",'
        '"agents_states":{"thread-child":{"status":"completed","message":null}},'
        '"status":"completed"}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert len(tcs) == 1
    assert tcs[0]["action"] == "spawn_agent"
    assert tcs[0]["action_input"] == {
        "sender_thread_id": "thread-parent",
        "receiver_thread_ids": ["thread-child"],
        "prompt": "review logs",
        "agents_states": {"thread-child": {"status": "completed", "message": None}},
        "status": "completed",
    }
    assert "status=completed" in tcs[0]["observation"]


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


def test_error_recovery_detects_codex_status_failed_observations():
    log = (
        '{"type":"item.completed","item":{"type":"command_execution","id":"cmd-1",'
        '"command":"python run.py","exit_code":1,"status":"failed","aggregated_output":""}}\n'
        '{"type":"item.completed","item":{"type":"command_execution","id":"cmd-2",'
        '"command":"python run.py","exit_code":0,"status":"completed","aggregated_output":"ok"}}\n'
    )
    traj = synthetic_trajectory_from_codex_txt(log)
    assert traj is not None
    result = check_error_recovery(extract_tool_calls_as_dicts(traj))
    assert result["first_attempt_clean"] is False
    assert result["score"] < 1.0


def test_error_recovery_detects_opencode_completed_nonzero_exit():
    # OpenCode can mark bash as status=completed while metadata.exit is nonzero.
    log = (
        '{"type":"tool_use","part":{"type":"tool","tool":"bash","callID":"call-fail",'
        '"state":{"status":"completed","input":{"command":"python run.py"},'
        '"output":"","metadata":{"exit":1}}}}\n'
        '{"type":"tool_use","part":{"type":"tool","tool":"bash","callID":"call-ok",'
        '"state":{"status":"completed","input":{"command":"python run.py"},'
        '"output":"ok","metadata":{"exit":0}}}}\n'
    )
    traj = synthetic_trajectory_from_opencode_json(log)
    assert traj is not None
    tcs = extract_tool_calls_as_dicts(traj)
    assert "exit_code=1" in tcs[0]["observation"]
    assert "status=completed" in tcs[0]["observation"]
    assert "exit_code=0" in tcs[1]["observation"]
    result = check_error_recovery(tcs)
    assert result["first_attempt_clean"] is False
    assert result["score"] < 1.0
    assert "All commands succeeded on first attempt" not in result["reason"]


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


def _native_opencode_pair():
    call = {"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": "probe", "n": 1}}
    native = {
        "session_id": "ses_actual",
        "agent": {"name": "opencode", "version": "1.18.33"},
        "schema_version": "ATIF-v1.2",
        "final_metrics": {"total_tokens": 31},
        "steps": [
            {
                "source": "agent",
                "message": "Run the probe",
                "timestamp": "2026-10-27T00:00:00Z",
                "metrics": {"prompt_tokens": 17},
                "tool_calls": [call],
                "observation": {"results": [{"source_call_id": "c1", "content": ""}]},
            }
        ],
    }
    event = {
        "type": "tool_use",
        "sessionID": "ses_actual",
        "part": {
            "type": "tool",
            "tool": "bash",
            "callID": "c1",
            "sessionID": "ses_actual",
            "state": {
                "status": "completed",
                "input": deepcopy(call["arguments"]),
                "output": "",
                "metadata": {"exit": 1},
            },
        },
    }
    return native, event


def _load_native_opencode(tmp_path, native, events):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(native), encoding="utf-8")
    (tmp_path / "opencode.txt").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return load_trajectory_with_fallback(path)


def test_native_opencode_supplement_preserves_trace_and_reaches_consumers(tmp_path):
    native, event = _native_opencode_pair()
    data, meta = _load_native_opencode(tmp_path, native, [event])
    assert meta["source"] == "trajectory.json" and "added for 1 call" in meta["note"]
    calls = extract_tool_calls_as_dicts(data)
    assert calls[0]["observation"] == "status=completed\nexit_code=1\n"
    assert check_error_recovery(calls)["first_attempt_clean"] is False
    assert check_error_recovery(calls)["score"] == 0
    data["steps"][0]["observation"]["results"].pop(0)
    assert data == native  # Includes original output, input, turns, timestamps and token metrics.


def test_native_opencode_error_only_and_large_output_are_preserved(tmp_path):
    native, event = _native_opencode_pair()
    step, state = native["steps"][0], event["part"]["state"]
    state.update(status="error", error="File not found: example.txt")
    state.pop("output")
    state.pop("metadata")
    step["observation"]["results"] = []
    data, _ = _load_native_opencode(tmp_path, native, [event])
    call = data["steps"][0]["tool_calls"][0]
    assert normalized_tool_call_error_flag(data["steps"][0], call) is True
    assert "File not found: example.txt" in extract_tool_calls_as_dicts(data)[0]["observation"]
    state["output"] = "original output" * 1000
    step["observation"]["results"] = [{"source_call_id": "c1", "content": state["output"]}]
    data, _ = _load_native_opencode(tmp_path, native, [event])
    assert data["steps"][0]["observation"]["results"][-1]["content"] == state["output"]
    assert "File not found: example.txt" in extract_tool_calls_as_dicts(data)[0]["observation"]


@pytest.mark.parametrize(
    "boundary",
    [
        "session",
        "part_session",
        "call_id",
        "tool",
        "bool_input",
        "float_input",
        "missing_input",
        "duplicate_raw",
        "duplicate_native",
        "duplicate_result",
        "output",
        "native_flag",
        "native_call_flag",
        "pending",
        "user",
        "malformed_results",
        "missing_session",
        "partial_log",
        "output_type",
        "unscoped_result",
        "empty_result_id",
    ],
)
def test_native_opencode_supplement_refuses_uncertain_identity(tmp_path, boundary):
    native, event = _native_opencode_pair()
    step, part = native["steps"][0], event["part"]
    events = [event]
    if boundary == "session":
        event["sessionID"] = "stale"
    elif boundary == "part_session":
        part["sessionID"] = "stale"
    elif boundary == "call_id":
        part["callID"] = "different"
    elif boundary == "tool":
        part["tool"] = "read"
    elif boundary in ("bool_input", "float_input"):
        part["state"]["input"]["n"] = True if boundary == "bool_input" else 1.0
    elif boundary == "missing_input":
        part["state"].pop("input")
    elif boundary == "duplicate_raw":
        events.append(deepcopy(event))
    elif boundary == "duplicate_native":
        native["steps"].append(deepcopy(step))
    elif boundary == "duplicate_result":
        step["observation"]["results"].append(deepcopy(step["observation"]["results"][0]))
    elif boundary == "output":
        part["state"]["output"] = "different"
    elif boundary == "native_flag":
        step["observation"]["results"][0]["extra"] = {"tool_result_is_error": False}
    elif boundary == "native_call_flag":
        step["tool_calls"][0]["is_error"] = False
    elif boundary == "pending":
        part["state"]["status"] = "running"
    elif boundary == "user":
        step["source"] = "user"
    elif boundary == "malformed_results":
        step["observation"]["results"] = 42
    elif boundary == "missing_session":
        native.pop("session_id")
    elif boundary == "partial_log":
        events = [{"type": "tool_use", "part": {"callID": "c1"}}]
    elif boundary == "output_type":
        step["observation"]["results"][0]["content"] = True
        part["state"]["output"] = 1
    elif boundary in ("unscoped_result", "empty_result_id"):
        result = {"content": "exit_code=0"}
        if boundary == "empty_result_id":
            result["source_call_id"] = ""
        step["observation"]["results"].append(result)
    data, meta = _load_native_opencode(tmp_path, native, events)
    assert data == native
    assert meta["source"] == "trajectory.json" and meta["warning"] is None
    assert "skipped" in meta["note"] or "added for 0 call" in meta["note"]


@pytest.mark.parametrize("exit_value", [None, True, False, 1.0, "1", float("inf")])
def test_native_opencode_never_coerces_or_infers_exit_status(tmp_path, exit_value):
    native, event = _native_opencode_pair()
    event["part"]["state"]["metadata"]["exit"] = exit_value
    data, _ = _load_native_opencode(tmp_path, native, [event])
    observation = extract_tool_calls_as_dicts(data)[0]["observation"]
    assert "status=completed" in observation and "exit_code=" not in observation


@pytest.mark.parametrize(("actual_exit", "quoted_exit"), [(17, 0), (0, 17)])
def test_native_opencode_actual_exit_precedes_status_quoted_in_stdout(tmp_path, actual_exit, quoted_exit):
    native, event = _native_opencode_pair()
    output = f"exit_code={quoted_exit}\nThe diagnostic text above is an example."
    native["steps"][0]["observation"]["results"][0]["content"] = output
    event["part"]["state"].update(output=output, metadata={"exit": actual_exit})
    data, _ = _load_native_opencode(tmp_path, native, [event])
    calls = extract_tool_calls_as_dicts(data)
    assert calls[0]["observation"].startswith(f"status=completed\nexit_code={actual_exit}\n")
    assert check_error_recovery(calls)["first_attempt_clean"] is (actual_exit == 0)
    assert data["steps"][0]["observation"]["results"][-1]["content"] == output
