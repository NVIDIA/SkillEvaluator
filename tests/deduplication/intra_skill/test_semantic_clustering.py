# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.intra_skill.semantic_clustering."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from skillevaluator.deduplication.intra_skill import semantic_clustering
from skillevaluator.deduplication.intra_skill.semantic_clustering import (
    UnionFind,
    build_clusters,
)
from skillevaluator.embedding.client import EmbeddingClient


class TestUnionFind:
    def test_initial_find_returns_self(self) -> None:
        uf = UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_connects_two_elements(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_union_is_transitive(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_components_initial(self) -> None:
        uf = UnionFind(3)
        comps = uf.components()
        assert len(comps) == 3

    def test_components_after_unions(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        comps = uf.components()
        assert len(comps) == 3  # {0,1}, {2,3}, {4}

    def test_path_compression(self) -> None:
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        root = uf.find(3)
        # After find with path compression, parent should point directly to root
        assert uf.parent[3] == root


class TestBuildClusters:
    @pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True])
    def test_rejects_invalid_threshold(self, make_chunk, threshold: object) -> None:
        chunks = [make_chunk(embedding=[1.0, 0.0]), make_chunk(embedding=[1.0, 0.0])]
        with pytest.raises(ValueError, match=r"threshold|finite|\[0, 1\]"):
            build_clusters(chunks, threshold=threshold)  # type: ignore[arg-type]

    def test_rejects_scalar_work_before_cosine_loop(self, make_chunk, monkeypatch) -> None:
        chunks = [make_chunk(embedding=[1.0, 0.0]), make_chunk(embedding=[1.0, 0.0])]
        monkeypatch.setattr(
            semantic_clustering,
            "unit_vector_similarity",
            lambda *_args: (_ for _ in ()).throw(AssertionError("cosine must not run")),
        )
        monkeypatch.setattr(semantic_clustering, "CONTENT_DEDUP_MAX_SCALAR_COMPARISONS", 1)

        with pytest.raises(ValueError, match=r"scalar.*limit|scalar.*exceeds"):
            build_clusters(chunks)

    def test_fewer_than_2_chunks_returns_empty(self, make_chunk) -> None:
        assert build_clusters([make_chunk(embedding=[1.0, 0.0])]) == []
        assert build_clusters([]) == []

    def test_dissimilar_chunks_no_cluster(self, make_chunk) -> None:
        # Orthogonal vectors → cosine similarity ~0
        a = make_chunk(embedding=[1.0, 0.0])
        b = make_chunk(embedding=[0.0, 1.0], source_file="other.md")
        clusters = build_clusters([a, b], threshold=0.80)
        assert len(clusters) == 0

    def test_similar_chunks_form_cluster(self, make_chunk) -> None:
        # Nearly identical vectors → high cosine similarity
        a = make_chunk(embedding=[1.0, 0.0])
        b = make_chunk(embedding=[0.99, 0.14], source_file="other.md")
        clusters = build_clusters([a, b], threshold=0.80)
        assert len(clusters) == 1
        assert len(clusters[0].members) == 2

    def test_max_similarity_computed(self, make_chunk) -> None:
        a = make_chunk(embedding=[1.0, 0.0])
        b = make_chunk(embedding=[1.0, 0.0], source_file="other.md")
        clusters = build_clusters([a, b], threshold=0.80)
        assert clusters[0].max_similarity == pytest.approx(1.0)

    def test_cross_file_true_when_different_sources(self, make_chunk) -> None:
        a = make_chunk(source_file="a.md", embedding=[1.0, 0.0])
        b = make_chunk(source_file="b.md", embedding=[1.0, 0.0])
        clusters = build_clusters([a, b], threshold=0.80)
        assert clusters[0].cross_file is True

    def test_cross_file_false_when_same_source(self, make_chunk) -> None:
        a = make_chunk(source_file="same.md", heading="## A", embedding=[1.0, 0.0])
        b = make_chunk(source_file="same.md", heading="## B", embedding=[1.0, 0.0])
        clusters = build_clusters([a, b], threshold=0.80)
        assert clusters[0].cross_file is False

    def test_singletons_excluded(self, make_chunk) -> None:
        a = make_chunk(embedding=[1.0, 0.0])
        b = make_chunk(embedding=[0.0, 1.0], source_file="b.md")
        c = make_chunk(embedding=[1.0, 0.01], source_file="c.md")
        clusters = build_clusters([a, b, c], threshold=0.80)
        # a and c are similar, b is different → one cluster of {a, c}
        assert len(clusters) == 1
        members = {m.source_file for m in clusters[0].members}
        assert "b.md" not in members

    def test_sorted_by_max_similarity_descending(self, make_chunk) -> None:
        # Create two clusters with different similarities
        a = make_chunk(source_file="a.md", embedding=[1.0, 0.0, 0.0])
        b = make_chunk(source_file="b.md", embedding=[0.99, 0.14, 0.0])
        c = make_chunk(source_file="c.md", embedding=[0.0, 0.0, 1.0])
        d = make_chunk(source_file="d.md", embedding=[0.0, 0.0, 0.99])
        clusters = build_clusters([a, b, c, d], threshold=0.80)
        if len(clusters) >= 2:
            assert clusters[0].max_similarity >= clusters[1].max_similarity

    def test_source_formats_collected(self, make_chunk) -> None:
        a = make_chunk(source_format="markdown", embedding=[1.0, 0.0])
        b = make_chunk(source_format="python", source_file="b.py", embedding=[1.0, 0.0])
        clusters = build_clusters([a, b], threshold=0.80)
        assert clusters[0].source_formats == {"markdown", "python"}

    def test_scores_match_cosine_similarity_and_each_vector_is_normalized_once(self, make_chunk, monkeypatch) -> None:
        rng = random.Random(28)
        vectors = [[rng.uniform(-1.0, 1.0) for _ in range(16)] for _ in range(12)]
        vectors[3] = list(vectors[0])
        vectors[5] = [value * 3.0 for value in vectors[1]]
        vectors[7] = [0.0] * 16
        chunks = [make_chunk(source_file=f"file-{index}.md", embedding=vector) for index, vector in enumerate(vectors)]
        normalized: list[int] = []
        real_normalize = semantic_clustering.normalize_embedding_vector

        def counting_normalize(vector, *args, **kwargs):
            normalized.append(len(vector))
            return real_normalize(vector, *args, **kwargs)

        monkeypatch.setattr(semantic_clustering, "normalize_embedding_vector", counting_normalize)

        clusters = build_clusters(chunks, threshold=0.2)

        assert normalized == [16] * len(chunks)
        assert clusters
        index_of = {id(chunk): index for index, chunk in enumerate(chunks)}
        for cluster in clusters:
            indices = sorted(index_of[id(member)] for member in cluster.members)
            expected = [EmbeddingClient.cosine_similarity(vectors[i], vectors[j]) for i, j in combinations(indices, 2)]
            assert cluster.max_similarity == max(expected)
            assert cluster.avg_similarity == sum(expected) / len(expected)
