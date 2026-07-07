# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for deduplication tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster
from skillevaluator.deduplication.utils.chunker import ContentChunk
from skillevaluator.deduplication.utils.skill_collector import CollectedFile


@pytest.fixture
def skill_root(tmp_path: Path) -> Path:
    """Create a minimal skill root directory."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    return skill_dir


@pytest.fixture
def make_collected_file():
    """Factory for CollectedFile objects with sensible defaults."""

    def _make(
        path: Path = Path("/fake/test.md"),
        rel_path: str = "test.md",
        extension: str = ".md",
        content: str = "# Heading\n\nSome content here that is long enough to pass the minimum character filter.\n",
        line_count: int = 4,
    ) -> CollectedFile:
        return CollectedFile(
            path=path,
            rel_path=rel_path,
            extension=extension,
            content=content,
            line_count=line_count,
        )

    return _make


@pytest.fixture
def make_chunk():
    """Factory for ContentChunk objects with sensible defaults."""

    def _make(
        source_file: str = "README.md",
        heading: str = "## Overview",
        text: str = "x" * 100,
        start_line: int = 1,
        end_line: int = 5,
        source_format: str = "markdown",
        embedding: list[float] | None = None,
    ) -> ContentChunk:
        chunk = ContentChunk(
            source_file=source_file,
            heading=heading,
            start_line=start_line,
            end_line=end_line,
            text=text,
            source_format=source_format,
        )
        if embedding is not None:
            chunk.embedding = embedding
        return chunk

    return _make


@pytest.fixture
def make_cluster(make_chunk):
    """Factory for ContentCluster objects."""

    def _make(
        members: list[ContentChunk] | None = None,
        max_similarity: float = 0.90,
        avg_similarity: float = 0.85,
        cross_file: bool = True,
    ) -> ContentCluster:
        if members is None:
            members = [
                make_chunk(source_file="SKILL.md", heading="## Overview"),
                make_chunk(source_file="README.md", heading="## About"),
            ]
        return ContentCluster(
            members=members,
            max_similarity=max_similarity,
            avg_similarity=avg_similarity,
            cross_file=cross_file,
            source_formats={m.source_format for m in members},
        )

    return _make
