# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding client for public OpenAI-compatible providers.

Wraps the OpenAI-compatible API to generate vector embeddings for
similarity comparison. Supports both single-string and chunked
full-body embedding modes.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import replace
from typing import Any

from skillevaluator.constants import (
    SIMILARITY_CHUNK_OVERLAP,
    SIMILARITY_CHUNK_SIZE,
    SIMILARITY_DEFAULT_MODEL,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.provider_config import ProviderConfig, ProviderConfigurationError, resolve_embedding_provider

logger = get_logger(__name__)


class SimilarityConfigError(Exception):
    """Raised when embedding configuration is missing or invalid."""


MAX_EMBEDDING_VECTOR_DIMENSION = 65_536


def validate_embedding_vector(
    vector: object,
    expected_dimension: int | None = None,
    *,
    context: str = "Embedding provider",
) -> int:
    """Validate a provider vector before it reaches similarity arithmetic."""
    values, _norm = _validated_vector_values(
        vector,
        expected_dimension,
        context=context,
        allow_zero=False,
    )
    return len(values)


class EmbeddingClient:
    """Public OpenAI-compatible client for generating text embeddings.

    Uses the OpenAI SDK because NVIDIA Build exposes an OpenAI-compatible API.
    Client construction is deferred until the first embed() call so that
    importing this module never requires an API key.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client: Any = None
        self._provider_config: ProviderConfig | None = None

    @property
    def model(self) -> str:
        """The embedding model identifier."""
        return self._model or self._resolved_config().model

    def _resolved_config(self) -> ProviderConfig:
        if self._provider_config is not None:
            return self._provider_config
        if self._api_key or self._base_url:
            model = self._model or SIMILARITY_DEFAULT_MODEL
            self._provider_config = ProviderConfig(
                provider="openai-compatible",
                model=model,
                api_key=self._api_key,
                base_url=self._base_url,
                litellm_model=f"openai/{model}",
            )
            return self._provider_config
        try:
            config = resolve_embedding_provider()
        except ProviderConfigurationError as exc:
            raise SimilarityConfigError(str(exc)) from exc
        if self._model:
            config = replace(config, model=self._model, litellm_model=f"openai/{self._model}")
        self._provider_config = config
        return config

    def _get_client(self) -> Any:
        """Lazily construct the OpenAI client on first use."""
        if self._client is not None:
            return self._client

        config = self._resolved_config()
        if not config.api_key or not config.base_url:
            raise SimilarityConfigError(f"No embedding API key or base URL resolved for {config.provider}.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SimilarityConfigError(
                "The 'openai' package is required for similarity checks. Install it with: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Strings to embed. Empty list returns empty list.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        data = list(response.data)
        if len(data) != len(texts):
            raise SimilarityConfigError(
                f"Embedding response index set is incomplete: received {len(data)} vectors for {len(texts)} inputs."
            )

        ordered: list[list[float] | None] = [None] * len(texts)
        expected_dimension: int | None = None
        for item in data:
            index = getattr(item, "index", None)
            if type(index) is not int or not 0 <= index < len(texts):
                raise SimilarityConfigError(f"Embedding response index is invalid: {index!r}.")
            if ordered[index] is not None:
                raise SimilarityConfigError(f"Embedding response contains duplicate index {index}.")
            vector = getattr(item, "embedding", None)
            dimension = validate_embedding_vector(
                vector,
                expected_dimension,
                context=f"Embedding response at index {index}",
            )
            if expected_dimension is None:
                expected_dimension = dimension
            ordered[index] = vector

        if any(vector is None for vector in ordered):
            raise SimilarityConfigError("Embedding response indices are incomplete.")
        return [vector for vector in ordered if vector is not None]

    def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""
        return self.embed([text])[0]

    def embed_chunked(
        self,
        text: str,
        chunk_size: int = SIMILARITY_CHUNK_SIZE,
        overlap: int = SIMILARITY_CHUNK_OVERLAP,
    ) -> list[float]:
        """Split text into chunks, embed each, and average-pool.

        Used by --full-body mode for documents that may exceed the
        model's context window.

        Args:
            text: Full document text.
            chunk_size: Approximate token count per chunk.
            overlap: Token overlap between adjacent chunks.

        Returns:
            Single averaged embedding vector.
        """
        chunks = _split_into_chunks(text, chunk_size, overlap)
        vectors = self.embed(chunks)
        return _average_pool(vectors)

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Uses stdlib math only -- no numpy dependency required.

        Returns:
            Similarity score in [-1.0, 1.0]. Returns 0.0 for zero vectors.

        Raises:
            SimilarityConfigError: If vectors have different dimensionality.
        """
        if len(vec_a) != len(vec_b):
            raise SimilarityConfigError(
                f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}. "
                "This usually means the embeddings were produced by different models "
                "or a stale cache is being used."
            )
        values_a, norm_a = _validated_vector_values(
            vec_a,
            len(vec_a),
            context="First similarity",
            allow_zero=True,
        )
        values_b, norm_b = _validated_vector_values(
            vec_b,
            len(vec_a),
            context="Second similarity",
            allow_zero=True,
        )
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        score = math.fsum((a / norm_a) * (b / norm_b) for a, b in zip(values_a, values_b, strict=True))
        if not math.isfinite(score):
            raise SimilarityConfigError("Cosine similarity produced a non-finite result.")
        return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_HEADING_PATTERN = re.compile(r"^#{1,6}\s", re.MULTILINE)

# Rough chars-per-token estimate for English markdown
_CHARS_PER_TOKEN = 4


def _split_into_chunks(
    text: str,
    chunk_size: int = SIMILARITY_CHUNK_SIZE,
    overlap: int = SIMILARITY_CHUNK_OVERLAP,
) -> list[str]:
    """Split markdown text into chunks for embedding.

    Strategy:
    1. Split on markdown headings (## ) as natural boundaries.
    2. If any section still exceeds chunk_size tokens, fall back to
       fixed-size overlapping windows.
    3. Guarantee at least one chunk is returned.
    """
    max_chars = chunk_size * _CHARS_PER_TOKEN
    overlap_chars = overlap * _CHARS_PER_TOKEN

    sections = _split_by_headings(text)

    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            chunks.extend(_fixed_size_chunks(section, max_chars, overlap_chars))

    return chunks or [text]


def _split_by_headings(text: str) -> list[str]:
    """Split text at markdown heading boundaries, keeping headings with their content."""
    positions = [m.start() for m in _HEADING_PATTERN.finditer(text)]

    if not positions:
        return [text] if text.strip() else []

    sections: list[str] = []
    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            sections.append(preamble)

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    return sections


def _fixed_size_chunks(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Fall back to fixed-size overlapping windows."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max_chars - overlap_chars
    return chunks


def _average_pool(vectors: list[list[float]]) -> list[float]:
    """Average-pool a list of vectors into a single vector."""
    if not vectors:
        return []
    dimension: int | None = None
    for index, vector in enumerate(vectors):
        dimension = validate_embedding_vector(
            vector,
            dimension,
            context=f"Embedding vector {index}",
        )
    assert dimension is not None
    count = len(vectors)
    pooled = [math.fsum(vector[index] for vector in vectors) / count for index in range(dimension)]
    validate_embedding_vector(pooled, dimension, context="Pooled embedding")
    return pooled


def _validated_vector_values(
    vector: object,
    expected_dimension: int | None,
    *,
    context: str,
    allow_zero: bool,
) -> tuple[list[float], float]:
    if not isinstance(vector, list) or not vector:
        raise SimilarityConfigError(f"{context} vector must be a non-empty list.")
    if len(vector) > MAX_EMBEDDING_VECTOR_DIMENSION:
        raise SimilarityConfigError(f"{context} vector dimension exceeds {MAX_EMBEDDING_VECTOR_DIMENSION}.")
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise SimilarityConfigError(
            f"{context} vector dimension mismatch: {len(vector)}; expected {expected_dimension}."
        )

    maximum_component = math.sqrt(sys.float_info.max / len(vector))
    values: list[float] = []
    norm = 0.0
    for value in vector:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SimilarityConfigError(f"{context} vector values must be finite numbers.")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise SimilarityConfigError(f"{context} vector values must be finite numbers.") from exc
        if not math.isfinite(numeric):
            raise SimilarityConfigError(f"{context} vector values must be finite numbers.")
        if abs(numeric) > maximum_component:
            raise SimilarityConfigError(f"{context} vector magnitude is too large for numerically stable similarity.")
        values.append(numeric)
        norm = math.hypot(norm, numeric)

    if not math.isfinite(norm):
        raise SimilarityConfigError(f"{context} vector magnitude is not numerically stable.")
    if norm == 0.0 and not allow_zero:
        raise SimilarityConfigError(f"{context} vector must not be a zero vector.")
    return values, norm
