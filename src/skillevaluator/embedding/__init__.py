# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding-based similarity detection for SkillEvaluator content."""

from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError
from skillevaluator.embedding.extractor import ContentEntry, discover_and_extract
from skillevaluator.embedding.registry import EmbeddingRegistry, RegistryEntry, SimilarityMatch

__all__ = [
    "ContentEntry",
    "EmbeddingClient",
    "EmbeddingRegistry",
    "RegistryEntry",
    "SimilarityConfigError",
    "SimilarityMatch",
    "discover_and_extract",
]
