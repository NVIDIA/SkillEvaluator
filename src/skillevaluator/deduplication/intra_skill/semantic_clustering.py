# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic clustering via pairwise cosine similarity + Union-Find.

Groups ContentChunk objects into clusters where all members have
similarity >= threshold to at least one other member in the group.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import TYPE_CHECKING

from skillevaluator.constants import CONTENT_DEDUP_MAX_CHUNKS, CONTENT_DEDUP_SIMILARITY_THRESHOLD
from skillevaluator.embedding.client import normalize_embedding_vector, unit_vector_similarity

CONTENT_DEDUP_MAX_SCALAR_COMPARISONS = 25_000_000

if TYPE_CHECKING:
    from skillevaluator.deduplication.utils.chunker import ContentChunk


@dataclass
class ContentCluster:
    """A group of semantically similar chunks."""

    members: list[ContentChunk]
    max_similarity: float
    avg_similarity: float
    cross_file: bool
    source_formats: set[str] = field(default_factory=set)


class UnionFind:
    """Disjoint Set Union with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


def _merge_split_pieces(members: list[ContentChunk]) -> list[ContentChunk]:
    """Report pieces of one split source chunk once, with their merged range."""
    merged: list[ContentChunk] = []
    positions: dict[tuple[str, int, int], int] = {}
    for chunk in members:
        group = chunk.split_group
        if group is None:
            merged.append(chunk)
        elif group in positions:
            first = merged[positions[group]]
            merged[positions[group]] = replace(
                first,
                text=f"{first.text}\n\n{chunk.text}",
                start_line=min(first.start_line, chunk.start_line),
                end_line=max(first.end_line, chunk.end_line),
            )
        else:
            positions[group] = len(merged)
            merged.append(chunk)
    return merged


def build_clusters(
    chunks: list[ContentChunk],
    threshold: float = CONTENT_DEDUP_SIMILARITY_THRESHOLD,
) -> list[ContentCluster]:
    """Cluster chunks by pairwise cosine similarity using Union-Find."""
    n = len(chunks)
    if n < 2:
        return []

    if n > CONTENT_DEDUP_MAX_CHUNKS:
        raise ValueError("Content chunk count exceeds the configured limit")
    dimension = len(chunks[0].embedding)
    if n * (n - 1) // 2 * dimension > CONTENT_DEDUP_MAX_SCALAR_COMPARISONS:
        raise ValueError("Content scalar comparison work exceeds the configured limit")
    units = [normalize_embedding_vector(chunk.embedding, dimension, allow_zero=True) for chunk in chunks]

    def siblings(i: int, j: int) -> bool:
        group = chunks[i].split_group
        return group is not None and group == chunks[j].split_group

    uf = UnionFind(n)
    pair_scores: dict[tuple[int, int], float] = {}

    for i, j in combinations(range(n), 2):
        score = unit_vector_similarity(units[i], units[j])
        pair_scores[(i, j)] = score
        if score >= threshold and not siblings(i, j):
            uf.union(i, j)

    clusters: list[ContentCluster] = []
    for indices in uf.components().values():
        if len(indices) < 2:
            continue

        members = _merge_split_pieces([chunks[i] for i in indices])
        scores = [pair_scores[(i, j)] for i, j in combinations(indices, 2) if not siblings(i, j)]

        clusters.append(
            ContentCluster(
                members=members,
                max_similarity=max(scores) if scores else 0.0,
                avg_similarity=sum(scores) / len(scores) if scores else 0.0,
                cross_file=len({c.source_file for c in members}) > 1,
                source_formats={c.source_format for c in members},
            )
        )

    clusters.sort(key=lambda c: c.max_similarity, reverse=True)
    return clusters
