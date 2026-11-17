# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON string content is not a JSONL record separator."""

import json

import pytest

from skillevaluator.tier3.eval_core.log_converters import (
    synthetic_trajectory_from_claude_stream_jsonl,
    synthetic_trajectory_from_cline_cli,
    synthetic_trajectory_from_codex_json,
    synthetic_trajectory_from_codex_txt,
    synthetic_trajectory_from_opencode_json,
)


def _events(format_name: str, content: str) -> list[dict]:
    if format_name == "claude":
        return [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": content}}]
                },
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}]},
            },
            {"type": "assistant", "message": {"content": "finished"}},
        ]
    if format_name == "cline":
        return [
            {"say": "command", "text": content},
            {"ask": "command_output", "text": content},
            {"say": "completion_result", "text": "finished"},
        ]
    if format_name == "codex":
        return [
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "id": "t1",
                    "command": content,
                    "aggregated_output": content,
                    "exit_code": 0,
                },
            },
            {"type": "item.completed", "item": {"type": "agent_message", "text": "finished"}},
        ]
    return [
        {
            "type": "tool_use",
            "part": {
                "type": "tool",
                "tool": "bash",
                "callID": "t1",
                "state": {"status": "completed", "input": {"command": content}, "output": content},
            },
        },
        {"type": "text", "part": {"type": "text", "text": "finished"}},
    ]


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
@pytest.mark.parametrize(
    ("format_name", "parser"),
    [
        ("claude", synthetic_trajectory_from_claude_stream_jsonl),
        ("cline", synthetic_trajectory_from_cline_cli),
        ("opencode", synthetic_trajectory_from_opencode_json),
        ("codex", synthetic_trajectory_from_codex_json),
        ("opencode", synthetic_trajectory_from_codex_txt),
    ],
)
def test_literal_unicode_separators_preserve_command_and_output(format_name, parser, line_ending, separator):
    content = f"echo before{separator}after"
    events = _events(format_name, content)
    escaped_log = line_ending.join(json.dumps(event, ensure_ascii=True) for event in events) + line_ending
    literal_log = line_ending.join(json.dumps(event, ensure_ascii=False) for event in events) + line_ending
    expected = parser(escaped_log)
    assert expected is not None
    assert separator in expected["steps"][0]["tool_calls"][0]["arguments"]["command"]

    assert parser(literal_log) == expected
