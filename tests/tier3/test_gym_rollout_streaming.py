# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym rollout ingestion should retain rows without copying the entire JSONL file."""

from __future__ import annotations

import json
import tracemalloc

from skillevaluator.tier3.nemo_gym.results import read_rollouts


def test_rollout_loading_avoids_whole_file_text_copies(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    count = 512
    payload = "x" * 16_384
    with path.open("w", encoding="utf-8") as stream:
        for index in range(count):
            stream.write(json.dumps({"task_id": str(index), "payload": payload}) + "\n")
    file_size = path.stat().st_size

    tracemalloc.start()
    try:
        rows = read_rollouts(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(rows) == count
    assert [row["task_id"] for row in rows] == [str(index) for index in range(count)]
    assert all(row["payload"] == payload for row in rows)
    # Parsed rows must remain in memory; avoid retaining the whole list of
    # split input lines alongside those parsed rows.
    assert peak < 1.5 * file_size, f"{file_size:,}-byte JSONL required {peak:,} traced bytes"


def test_unreadable_nested_row_keeps_healthy_rows_on_both_sides(tmp_path, monkeypatch):
    path = tmp_path / "rollouts.jsonl"
    nested = '{"nested":[]}'
    path.write_text('{"task_id":"first"}\n' + nested + '\n{"task_id":"last"}\n', encoding="utf-8")
    errors = []
    # Decoder nesting limits differ between supported Python versions.
    original_loads = json.loads

    def decode_row(line):
        if line.strip() == nested:
            raise RecursionError("JSON decoder nesting limit")
        return original_loads(line)

    monkeypatch.setattr(json, "loads", decode_row)

    assert read_rollouts(path, errors) == [{"task_id": "first"}, {"task_id": "last"}]
    assert errors == ["NeMo Gym wrote 1 unreadable line(s) to rollouts.jsonl; those rollouts were not collected"]


def test_streamed_rows_preserve_unicode_replacement_and_final_line(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    first = json.dumps({"text": "before\u2028between\u0085after"}, ensure_ascii=False).encode("utf-8")
    path.write_bytes(b"\r\n" + first + b'\r\n{"text":"bad \xff byte"}\r\n{"task_id":"last"}')
    errors = []

    assert read_rollouts(path, errors) == [
        {"text": "before\u2028between\u0085after"},
        {"text": "bad \ufffd byte"},
        {"task_id": "last"},
    ]
    assert errors == []
