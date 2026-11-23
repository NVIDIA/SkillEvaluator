# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosine similarity must preserve direction when accepted vectors are tiny."""

import math
from unittest.mock import MagicMock

import pytest

from skillevaluator.embedding.client import EmbeddingClient, normalize_embedding_vector
from skillevaluator.embedding.registry import EmbeddingRegistry, RegistryEntry


@pytest.mark.parametrize("scale", [math.ulp(0.0), 1e-320, 1e-310])
@pytest.mark.parametrize("direction", [[1.0, 1.0], [-1.0, 1.0], [1.0, 2.0, -3.0]])
def test_subnormal_vector_normalization_preserves_direction(scale: float, direction: list[float]) -> None:
    vector = [scale * component for component in direction]
    original = list(vector)
    expected = normalize_embedding_vector(direction)

    normalized = normalize_embedding_vector(vector)

    assert normalized == pytest.approx(expected, abs=1e-14)
    assert math.hypot(*normalized) == pytest.approx(1.0, abs=1e-14)
    assert vector == original
    axis = [1.0] + [0.0] * (len(vector) - 1)
    assert EmbeddingClient.cosine_similarity(vector, axis) == pytest.approx(expected[0], abs=1e-14)


def test_tiny_vector_does_not_create_a_false_high_similarity_match() -> None:
    client = MagicMock(spec=EmbeddingClient)
    client.embed_single.return_value = [math.ulp(0.0), math.ulp(0.0)]
    registry = EmbeddingRegistry(client)
    registry._entries["axis"] = RegistryEntry(
        name="axis",
        description="A distinct direction",
        path="axis",
        content_type="skill",
        embedding=[1.0, 0.0],
    )

    assert registry.query("query", threshold=0.9) == []
