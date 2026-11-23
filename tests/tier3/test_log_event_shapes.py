# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-event JSON values must not discard valid surrounding agent evidence."""

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core.log_converters import (
    load_trajectory_with_fallback,
    synthetic_trajectory_from_claude_stream_jsonl,
    synthetic_trajectory_from_cline_cli,
)

CLAUDE_EVENTS = [
    {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "echo kept"}}]},
    },
    {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "kept"}]}},
    {"type": "assistant", "message": {"content": "finished"}},
]
CLINE_EVENTS = [
    {"say": "command", "text": "echo kept"},
    {"ask": "command_output", "text": "kept"},
    {"say": "completion_result", "text": "finished"},
]
NON_EVENTS = [None, [], False, 7, "noise"]


def _jsonl(events: list[object]) -> str:
    return "\n".join(json.dumps(event) for event in events)


@pytest.mark.parametrize("noise", NON_EVENTS)
@pytest.mark.parametrize(
    ("parser", "events"),
    [
        (synthetic_trajectory_from_claude_stream_jsonl, CLAUDE_EVENTS),
        (synthetic_trajectory_from_cline_cli, CLINE_EVENTS),
    ],
)
def test_non_object_events_preserve_surrounding_tool_evidence(parser, events, noise) -> None:
    clean = parser(_jsonl(events))
    assert clean is not None

    assert parser(_jsonl([noise, events[0], noise, *events[1:], noise])) == clean
    assert parser(_jsonl([noise])) is None


@pytest.mark.parametrize("payload", NON_EVENTS)
def test_cline_non_object_tool_payload_preserves_surrounding_events(payload) -> None:
    malformed_tool = {"say": "tool", "text": json.dumps(payload)}

    assert synthetic_trajectory_from_cline_cli(
        _jsonl([CLINE_EVENTS[0], malformed_tool, *CLINE_EVENTS[1:]])
    ) == synthetic_trajectory_from_cline_cli(_jsonl(CLINE_EVENTS))


@pytest.mark.parametrize(
    ("filename", "parser", "events"),
    [
        ("claude-code.txt", synthetic_trajectory_from_claude_stream_jsonl, CLAUDE_EVENTS),
        ("cline.txt", synthetic_trajectory_from_cline_cli, CLINE_EVENTS),
    ],
)
def test_fallback_recovers_valid_evidence_around_non_events(tmp_path: Path, filename, parser, events) -> None:
    (tmp_path / filename).write_text(_jsonl([*NON_EVENTS, *events, *NON_EVENTS]), encoding="utf-8")

    trajectory, metadata = load_trajectory_with_fallback(tmp_path / "trajectory.json", tmp_path)

    assert trajectory == parser(_jsonl(events))
    assert metadata["source"] == filename
