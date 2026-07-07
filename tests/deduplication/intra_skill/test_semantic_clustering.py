# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.intra_skill.semantic_clustering."""

from __future__ import annotations

import pytest

from skillevaluator.deduplication.intra_skill.semantic_clustering import (
    UnionFind,
    build_clusters,
)


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
