# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON string content is not a JSONL record separator."""

import json

import pytest

from skillevaluator.tier3.eval_core.log_converters import (
    load_trajectory_with_fallback,
    synthetic_trajectory_from_claude_stream_jsonl,
    synthetic_trajectory_from_cline_cli,
    synthetic_trajectory_from_codex_json,
    synthetic_trajectory_from_codex_txt,
    synthetic_trajectory_from_opencode_json,
)
from skillevaluator.tier3.trace_import import build_candidates


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


# Parsers that read a trial-directory log file, with the file name that loader looks for.
_FILE_PARSERS = [
    ("claude", synthetic_trajectory_from_claude_stream_jsonl, "claude-code.txt"),
    ("cline", synthetic_trajectory_from_cline_cli, "cline.txt"),
    ("opencode", synthetic_trajectory_from_opencode_json, "opencode.txt"),
    ("opencode", synthetic_trajectory_from_codex_txt, "codex.txt"),
]


def _joined(format_name: str, line_ending: str) -> str:
    return line_ending.join(json.dumps(event) for event in _events(format_name, "echo hi")) + line_ending


@pytest.mark.parametrize(
    ("format_name", "parser"),
    [(name, parser) for name, parser, _ in _FILE_PARSERS] + [("codex", synthetic_trajectory_from_codex_json)],
)
def test_parsers_do_not_split_a_cr_only_string_into_records(format_name, parser):
    assert parser(_joined(format_name, "\r")) is None


@pytest.mark.parametrize("wrapper", [True, False])
def test_cr_only_agent_log_in_an_import_traces_export_is_unsupported(wrapper):
    def candidate_warnings(line_ending: str) -> list[str]:
        session = {
            "session_id": "s1",
            "prompt": "do it",
            "agent": "claude-code",
            "agent_log": _joined("claude", line_ending),
        }
        if wrapper:
            export = {"format": "skillevaluator.trace-export", "version": 1, "sessions": [session]}
            raw = json.dumps(export).encode()
        else:
            raw = (json.dumps(session) + "\n").encode()
        review, _ = build_candidates(raw, "export.json", "demo-skill")
        return review["candidates"][0]["warnings"]

    assert candidate_warnings("\n") == []
    assert candidate_warnings("\r\n") == []
    assert candidate_warnings("\r") == ["unsupported agent log format"]


@pytest.mark.parametrize(("format_name", "parser", "filename"), _FILE_PARSERS)
def test_cr_only_trial_directory_log_is_still_loaded(tmp_path, format_name, parser, filename):
    (tmp_path / filename).write_bytes(_joined(format_name, "\r").encode())
    expected = parser(_joined(format_name, "\n"))
    assert expected is not None

    trajectory, meta = load_trajectory_with_fallback(tmp_path / "trajectory.json", tmp_path)

    assert meta["source"] == filename
    assert trajectory == expected
