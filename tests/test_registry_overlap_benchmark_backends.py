# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification tests for the optional registry benchmark engines."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_pgvector_rejects_dimensions_outside_half_precision_hnsw_limit() -> None:
    from scripts.registry_overlap_benchmark_backends import PgVectorBackend

    with pytest.raises(ValueError, match="dimension"):
        PgVectorBackend("", 4001)


@pytest.fixture(params=["postgres", "qdrant"])
def prepared_backend(request):
    if request.param == "postgres":
        endpoint = os.getenv("SKILLEVALUATOR_BENCHMARK_POSTGRES_DSN")
    else:
        endpoint = os.getenv("SKILLEVALUATOR_BENCHMARK_QDRANT_URL")
    if not endpoint:
        pytest.skip(f"Set the optional {request.param} benchmark endpoint to run this test")

    numpy = pytest.importorskip("numpy")
    from scripts.registry_overlap_benchmark_backends import PgVectorBackend, QdrantBackend

    # 64 KiB is above the Qdrant planner threshold even when split into two segments.
    vectors = numpy.eye(128, dtype=numpy.float32)
    backend_cls = PgVectorBackend if request.param == "postgres" else QdrantBackend
    backend = backend_cls(endpoint, vectors.shape[1])
    try:
        preparation = backend.prepare(vectors)
        assert preparation["planner_uses_ann"] is True
        assert preparation["ann_indexed_count"] > 0
        yield backend, vectors
    finally:
        backend.close()


def test_backend_search_returns_bounded_ids_and_excludes_only_requested_id(prepared_backend) -> None:
    backend, vectors = prepared_backend
    assert backend.search(vectors[3], 1) == [3]
    candidates = backend.search(vectors[3], 3, exclude=3)

    assert len(candidates) == len(set(candidates)) == 3
    assert 3 not in candidates
    assert backend.visible(3, 0)


def test_backend_rejects_late_write_after_removal_and_allows_newer_restore(prepared_backend) -> None:
    backend, vectors = prepared_backend
    skill_id = len(vectors)
    with ThreadPoolExecutor(max_workers=2) as workers:
        removal = workers.submit(backend.remove, skill_id, 2)
        stale_update = workers.submit(backend.upsert, skill_id, 1, vectors[3])
        removal.result()
        stale_update.result()

    assert not backend.visible(skill_id, 1)
    assert skill_id not in backend.search(vectors[3], 10)

    backend.upsert(skill_id, 3, vectors[3])
    backend.remove(skill_id, 2)
    assert backend.visible(skill_id, 3)
    assert backend.search(vectors[3], 1, exclude=3) == [skill_id]

    backend.remove(skill_id, 4)
    backend.upsert(skill_id, 4, vectors[3])
    assert not backend.visible(skill_id, 4)
    assert skill_id not in backend.search(vectors[3], 10)
