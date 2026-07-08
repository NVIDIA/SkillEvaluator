# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for EmbeddingClient -- public OpenAI-compatible API wrapper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from skillevaluator.embedding.client import (
    EmbeddingClient,
    SimilarityConfigError,
    _average_pool,
    _fixed_size_chunks,
    _split_by_headings,
    _split_into_chunks,
)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vec = [1.0, 2.0, 3.0]
        assert EmbeddingClient.cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert EmbeddingClient.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert EmbeddingClient.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert EmbeddingClient.cosine_similarity(a, b) == 0.0

    def test_similar_vectors(self) -> None:
        a = [1.0, 1.0]
        b = [1.0, 0.9]
        score = EmbeddingClient.cosine_similarity(a, b)
        assert 0.99 < score < 1.0


@dataclass
class _FakeEmbeddingItem:
    embedding: list[float]
    index: int


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingItem]


def _make_fake_response(vectors: list[list[float]]) -> _FakeEmbeddingResponse:
    return _FakeEmbeddingResponse(
        data=[_FakeEmbeddingItem(embedding=vector, index=index) for index, vector in enumerate(vectors)]
    )


class TestEmbed:
    def test_openai_provider_uses_public_openai_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
        monkeypatch.setenv("SKILL_EVAL_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_openai = MagicMock()

        with patch("openai.OpenAI", return_value=mock_openai) as mock_cls:
            EmbeddingClient()._get_client()

        mock_cls.assert_called_once_with(api_key="test-key", base_url="https://api.openai.com/v1")

    def test_embed_returns_vectors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        client = EmbeddingClient()
        expected = [[0.1, 0.2], [0.3, 0.4]]

        mock_openai = MagicMock()
        mock_openai.embeddings.create.return_value = _make_fake_response(expected)

        with patch("openai.OpenAI", return_value=mock_openai):
            result = client.embed(["hello", "world"])

        assert result == expected
        mock_openai.embeddings.create.assert_called_once_with(
            model="nvidia/nv-embed-v1",
            input=["hello", "world"],
            encoding_format="float",
        )

    def test_embed_reorders_out_of_order_provider_response_by_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        client = EmbeddingClient()
        first = [1.0, 0.0]
        second = [0.0, 1.0]
        mock_openai = MagicMock()
        mock_openai.embeddings.create.return_value = _FakeEmbeddingResponse(
            data=[
                _FakeEmbeddingItem(embedding=second, index=1),
                _FakeEmbeddingItem(embedding=first, index=0),
            ]
        )

        with patch("openai.OpenAI", return_value=mock_openai):
            result = client.embed(["first", "second"])

        assert result == [first, second]

    @pytest.mark.parametrize(
        "items",
        [
            [
                _FakeEmbeddingItem(embedding=[1.0, 0.0], index=0),
                _FakeEmbeddingItem(embedding=[0.0, 1.0], index=0),
            ],
            [_FakeEmbeddingItem(embedding=[1.0, 0.0], index=2)],
        ],
        ids=["duplicate-index", "out-of-range-index"],
    )
    def test_embed_rejects_invalid_provider_response_indices(
        self,
        monkeypatch: pytest.MonkeyPatch,
        items: list[_FakeEmbeddingItem],
    ) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        mock_openai.embeddings.create.return_value = _FakeEmbeddingResponse(data=items)

        with (
            patch("openai.OpenAI", return_value=mock_openai),
            pytest.raises(SimilarityConfigError, match="index"),
        ):
            EmbeddingClient().embed(["first", "second"])

    @pytest.mark.parametrize("vector", [[math.nan, 0.0], [1e308, 1e308]])
    def test_embed_rejects_unsafe_provider_vectors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vector: list[float],
    ) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_openai = MagicMock()
        mock_openai.embeddings.create.return_value = _make_fake_response([vector])

        with (
            patch("openai.OpenAI", return_value=mock_openai),
            pytest.raises(SimilarityConfigError, match=r"finite|magnitude|stable"),
        ):
            EmbeddingClient().embed(["unsafe"])

    def test_embed_single_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        client = EmbeddingClient()
        expected = [0.5, 0.6, 0.7]

        mock_openai = MagicMock()
        mock_openai.embeddings.create.return_value = _make_fake_response([expected])

        with patch("openai.OpenAI", return_value=mock_openai):
            result = client.embed_single("test text")

        assert result == expected

    def test_embed_empty_list_returns_empty(self) -> None:
        client = EmbeddingClient(api_key="unused")
        assert client.embed([]) == []

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("SKILL_EVAL_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("SKILL_EVAL_EMBEDDING_PROVIDER", raising=False)

        with pytest.raises(SimilarityConfigError, match="SKILL_EVAL_EMBEDDING_PROVIDER"):
            EmbeddingClient().embed(["test"])

    def test_env_embedding_model_is_honored_without_explicit_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SKILL_EVAL_EMBEDDING_MODEL wins when no explicit model is given.

        This is the documented contract in docs/configuration.mdx and what
        makes custom OpenAI-compatible endpoints (including local servers)
        usable for Tier 2.
        """
        monkeypatch.setenv("SKILL_EVAL_EMBEDDING_PROVIDER", "openai-compatible")
        monkeypatch.setenv("SKILL_EVAL_EMBEDDING_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("SKILL_EVAL_EMBEDDING_MODEL", "nomic-embed-text")
        monkeypatch.setenv("SKILL_EVAL_EMBEDDING_API_KEY", "local-no-key")

        assert EmbeddingClient().model == "nomic-embed-text"


class TestEmbedChunked:
    def test_embed_chunked_averages_vectors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        client = EmbeddingClient()

        mock_openai = MagicMock()
        mock_openai.embeddings.create.return_value = _make_fake_response([[1.0, 0.0], [0.0, 1.0]])

        with patch("openai.OpenAI", return_value=mock_openai):
            result = client.embed_chunked("## Section 1\nfoo\n## Section 2\nbar")

        assert result == pytest.approx([0.5, 0.5])


class TestSplitIntoChunks:
    def test_short_text_single_chunk(self) -> None:
        text = "A short sentence."
        chunks = _split_into_chunks(text, chunk_size=100, overlap=10)
        assert chunks == [text]

    def test_split_by_headings(self) -> None:
        text = "Preamble text\n## Section A\nContent A\n## Section B\nContent B"
        sections = _split_by_headings(text)
        assert len(sections) == 3
        assert "Preamble" in sections[0]
        assert "Section A" in sections[1]
        assert "Section B" in sections[2]

    def test_fallback_fixed_size(self) -> None:
        text = "x" * 100
        chunks = _fixed_size_chunks(text, max_chars=30, overlap_chars=5)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 30

    def test_empty_text_returns_original(self) -> None:
        chunks = _split_into_chunks("", chunk_size=100, overlap=10)
        assert chunks == [""]

    def test_heading_split_preserves_heading_with_content(self) -> None:
        text = "# Title\nSome intro\n## Part 1\nDetails"
        sections = _split_by_headings(text)
        assert any("# Title" in s for s in sections)
        assert any("## Part 1" in s for s in sections)


class TestAveragePool:
    def test_single_vector(self) -> None:
        assert _average_pool([[1.0, 2.0]]) == [1.0, 2.0]

    def test_two_vectors(self) -> None:
        result = _average_pool([[1.0, 0.0], [0.0, 1.0]])
        assert result == pytest.approx([0.5, 0.5])

    def test_empty_returns_empty(self) -> None:
        assert _average_pool([]) == []
