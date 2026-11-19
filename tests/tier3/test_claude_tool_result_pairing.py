# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude result lookup must preserve chronological pairing and evidence ordering."""

import json

import pytest

from skillevaluator.tier3.eval_core.log_converters import synthetic_trajectory_from_claude_stream_jsonl


def _message(role, content):
    return {"type": role, "message": {"content": content}}


def _call(call_id, path="file.md"):
    return {"type": "tool_use", "id": call_id, "name": "Read", "input": {"file_path": path}}


def _result(call_id, content, **extra):
    return {"type": "tool_result", "tool_use_id": call_id, "content": content, **extra}


def _trajectory(*events):
    return synthetic_trajectory_from_claude_stream_jsonl("\n".join(json.dumps(event) for event in events))


def test_delayed_out_of_order_and_repeated_results_keep_their_steps():
    trajectory = _trajectory(
        _message("assistant", [_call("a"), _call("b")]),
        _message("assistant", "progress update"),
        _message("assistant", [_call("c")]),
        _message(
            "user",
            [
                _result("b", "failed", is_error=True),
                _result("c", [{"type": "text", "text": "third"}]),
                _result("a", "first"),
                _result("b", "retry result"),
            ],
        ),
        {"type": "result", "result": "finished"},
    )

    steps = trajectory["steps"]
    assert len(steps) == 4
    assert steps[0]["observation"]["results"] == [
        {"source_call_id": "b", "content": "failed", "extra": {"tool_result_is_error": True}},
        {"source_call_id": "a", "content": "first"},
        {"source_call_id": "b", "content": "retry result"},
    ]
    assert steps[1]["message"] == "progress update"
    assert steps[1]["observation"]["results"] == []
    assert steps[2]["observation"]["results"] == [{"source_call_id": "c", "content": "third"}]
    assert steps[3]["message"] == "finished"
    assert steps[3]["observation"]["results"] == []


def test_reused_ids_attach_once_to_the_latest_preceding_step():
    trajectory = _trajectory(
        _message("user", [_result("same", "too early")]),
        _message("assistant", [_call("same", "old.md"), _call("same", "duplicate.md")]),
        _message("user", [_result("same", "old result")]),
        _message("assistant", [_call("same", "new.md"), _call("")]),
        _message("user", [_result("same", "new result"), _result("unknown", "orphan"), _result("", "empty")]),
    )

    first, second = trajectory["steps"]
    assert len(first["tool_calls"]) == 2
    assert first["observation"]["results"] == [{"source_call_id": "same", "content": "old result"}]
    assert second["tool_calls"][0]["arguments"]["path"] == "new.md"
    assert second["observation"]["results"] == [{"source_call_id": "same", "content": "new result"}]


@pytest.mark.parametrize("reverse", [False, True])
def test_batched_calls_keep_result_order(reverse):
    ids = [f"call-{index}" for index in range(256)]
    result_ids = list(reversed(ids)) if reverse else ids
    trajectory = _trajectory(
        _message("assistant", [_call(call_id) for call_id in ids]),
        _message("user", [_result(call_id, f"result {call_id}") for call_id in result_ids]),
    )

    assert trajectory["steps"][0]["observation"]["results"] == [
        {"source_call_id": call_id, "content": f"result {call_id}"} for call_id in result_ids
    ]
