# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding windows must add new document content, including a short unique tail."""

import string

import pytest

from skillevaluator.embedding.client import EmbeddingClient, _fixed_size_chunks


@pytest.mark.parametrize(
    ("length", "overlap", "starts"),
    [
        (19, 4, [0, 8]),
        (20, 4, [0, 8]),
        (21, 4, [0, 8, 16]),
        (27, 4, [0, 8, 16]),
        (28, 4, [0, 8, 16]),
        (29, 4, [0, 8, 16, 24]),
        (20, 8, [0, 4, 8]),
        (24, 0, [0, 12]),
        (25, 0, [0, 12, 24]),
    ],
)
def test_windows_stop_once_the_document_is_covered(length, overlap, starts):
    text = string.ascii_letters[:length]

    assert _fixed_size_chunks(text, max_chars=12, overlap_chars=overlap) == [
        text[start : start + 12] for start in starts
    ]


@pytest.mark.parametrize("last_character", ["", "u"])
def test_embed_chunked_pools_only_windows_with_new_content(monkeypatch, last_character):
    client = EmbeddingClient(model="test-model", api_key="test-key", base_url="http://127.0.0.1:12345/v1")
    text = "abcdefghijklmnopqrst" + last_character
    expected_chunks = [text[:12], text[8:20]]
    if last_character:
        expected_chunks.append(text[16:])
    observed = []

    def embed(chunks):
        observed.extend(chunks)
        return [[1.0, 0.0] if chunk == text[:12] else [0.0, 1.0] for chunk in chunks]

    monkeypatch.setattr(client, "embed", embed)

    vector = client.embed_chunked(text, chunk_size=3, overlap=1)

    assert observed == expected_chunks
    count = len(expected_chunks)
    assert vector == pytest.approx([1 / count, (count - 1) / count])
