# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The finding prompt's 100-line preview must not allocate the rest of the file."""

from __future__ import annotations

import tracemalloc

import pytest

from skillevaluator.inference.finding_verifier import FindingVerifier


@pytest.mark.parametrize("count", [0, 1, 99, 100, 101])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
@pytest.mark.parametrize("terminated", [False, True])
def test_context_keeps_the_existing_line_boundary(tmp_path, count, newline, terminated):
    lines = [f"line {index}: café" for index in range(count)]
    content = newline.join(line.encode("utf-8") for line in lines)
    if terminated and lines:
        content += newline
    (tmp_path / "source.txt").write_bytes(content)

    expected_lines = lines[:100]
    if terminated and 0 < count < 100:
        expected_lines.append("")
    assert FindingVerifier._read_file_context(tmp_path, "source.txt") == "\n".join(expected_lines)


def test_context_keeps_ignore_policy_for_invalid_utf8(tmp_path):
    (tmp_path / "source.txt").write_bytes(b"first\xff line\r\nsecond\n")
    assert FindingVerifier._read_file_context(tmp_path, "source.txt") == "first line\nsecond\n"


def test_context_allocation_does_not_scale_with_unread_tail(tmp_path):
    path = tmp_path / "large.txt"
    prefix = "preview line\n" * 100
    with path.open("w", encoding="utf-8") as stream:
        stream.write(prefix)
        tail = "irrelevant tail\n" * 4096
        for _ in range(128):
            stream.write(tail)

    tracemalloc.start()
    try:
        context = FindingVerifier._read_file_context(tmp_path, "large.txt")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert context == prefix.removesuffix("\n")
    assert peak < 1024 * 1024, f"100-line context allocated {peak:,} bytes for the unused tail"
