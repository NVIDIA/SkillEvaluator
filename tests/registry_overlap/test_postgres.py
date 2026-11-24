# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from skillevaluator.registry_overlap.events import parse_batch

pytest.importorskip("psycopg")
from skillevaluator.registry_overlap.postgres import EmbeddingProfile, RegistryStore

pytestmark = pytest.mark.integration

PROFILE = EmbeddingProfile("fake", "model-v1", "sha256:" + "1" * 64, 4, "v1")
PROFILE_NEXT = EmbeddingProfile("fake", "model-v2", "sha256:" + "1" * 64, 4, "v2")
VECTOR = [1.0, 0.0, 0.0, 0.0]


class Embedder:
    def __init__(self):
        self.inputs = []

    def __call__(self, texts):
        self.inputs.extend(texts)
        return [VECTOR for _ in texts]


def _event(skill_id, sequence, description="Work", *, body=""):
    event = {"id": skill_id, "sequence": sequence, "revision": str(sequence)}
    if description is None:
        return {**event, "kind": "remove"}
    manifest = f"---\nname: skill\ndescription: {description}\n---\n{body}"
    return {**event, "kind": "upsert", "manifest": manifest, "artifact": f"https://example.com/{skill_id}/{sequence}"}


def _batch(cursor, *events):
    raw = json.dumps({"schema_version": 1, "source": "registry", "cursor": cursor, "events": events}).encode()
    return parse_batch(raw, hashlib.sha256(raw).hexdigest())


@pytest.fixture
def database():
    from psycopg import connect, sql

    dsn = os.environ.get("SKILL_EVAL_TEST_REGISTRY_DSN")
    if not dsn:
        pytest.skip("SKILL_EVAL_TEST_REGISTRY_DSN absent")
    schema = f"overlap_test_{uuid.uuid4().hex}"
    with RegistryStore(dsn, schema=schema) as store:
        store.initialize("registry", PROFILE)
        yield store, dsn, schema
    with connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _hits(store, profile=PROFILE, exclude=None):
    result = store.search(profile, lambda: VECTOR, top_k=10, exclude=exclude)
    return [(hit["skill_id"], hit["revision"]) for hit in result["candidates"]]


def test_resume_reuse_revision_and_exclude_only_self(database):
    store, dsn, schema = database
    embedder = Embedder()
    original = _batch(1, _event("b", 1), _event("a", 1))
    assert store.accept(original) == 2
    assert store.accept(original) == 0
    assert _hits(store) == []
    with RegistryStore(dsn, schema=schema) as resumed:
        assert resumed.drain(PROFILE, embedder) == 2
    assert embedder.inputs == ["skill: Work"]
    assert _hits(store) == [("a", "1"), ("b", "1")]
    assert _hits(store, exclude="a") == [("b", "1")]
    store.accept(_batch(2, _event("b", 2, body="Updated body")))
    assert _hits(store) == [("a", "1"), ("b", "2")]
    store.drain(PROFILE, embedder)
    assert _hits(store) == [("a", "1"), ("b", "2")]
    assert embedder.inputs == ["skill: Work"]


def test_conflict_rolls_back_batch_and_stale_event_cannot_revive_delete(database):
    store, _, _ = database
    store.accept(_batch(1, _event("a", 2)))
    store.drain(PROFILE, Embedder())
    with pytest.raises(ValueError, match="conflict"):
        store.accept(_batch(2, _event("b", 1), _event("a", 2, "Conflicting content")))
    assert store.status()["accepted_cursor"] == 1
    store.accept(_batch(2, _event("a", 5, None)))
    assert _hits(store) == []
    store.accept(_batch(3, _event("a", 4)))
    store.drain(PROFILE, Embedder())
    assert _hits(store) == []
    store.accept(_batch(4, _event("a", 6)))
    store.drain(PROFILE, Embedder())
    assert _hits(store) == [("a", "6")]


def test_delete_during_embedding_fences_late_worker(database):
    store, dsn, schema = database
    store.accept(_batch(1, _event("a", 1)))
    started, release = threading.Event(), threading.Event()

    def embed(texts):
        started.set()
        assert release.wait(10)
        return [VECTOR for _ in texts]

    def drain():
        with RegistryStore(dsn, schema=schema) as worker:
            return worker.drain(PROFILE, embed)

    with ThreadPoolExecutor() as executor:
        worker = executor.submit(drain)
        try:
            assert started.wait(10)
            store.accept(_batch(2, _event("a", 2, None)))
            assert _hits(store) == []
        finally:
            release.set()
        worker.result(timeout=10)
    store.drain(PROFILE, Embedder())
    assert _hits(store) == []


def test_failed_embedding_and_unready_generation_are_explicit(database):
    store, _, _ = database
    store.accept(_batch(1, _event("a", 1)))

    def fail(_texts):
        raise TimeoutError

    with pytest.raises(ValueError, match="Embedding"):
        store.drain(PROFILE, fail)
    result = store.search(PROFILE, lambda: VECTOR, top_k=10)
    assert result["status"] == "degraded"
    generation = store.reindex(PROFILE_NEXT)
    with pytest.raises(ValueError, match="caught up"):
        store.activate(generation)
    with pytest.raises(ValueError, match="profile"):
        store.search(PROFILE_NEXT, lambda: pytest.fail("incompatible query embedded"))
    with pytest.raises(ValueError, match="Embedding"):
        store.search(PROFILE, lambda: fail([]))


def test_cutover_fences_query_and_rollback_preserves_delete(database):
    store, dsn, schema = database
    store.accept(_batch(1, _event("a", 1), _event("b", 1)))
    store.drain(PROFILE, Embedder())
    old = store.status()["active_generation"]
    next_generation = store.reindex(PROFILE_NEXT)
    store.drain(PROFILE_NEXT, Embedder(), generation=next_generation)

    def cutover_after_embedding_starts():
        with RegistryStore(dsn, schema=schema) as activator:
            activator.activate(next_generation)
        return VECTOR

    with pytest.raises(ValueError, match="generation changed"):
        store.search(PROFILE, cutover_after_embedding_starts)
    assert _hits(store, PROFILE_NEXT) == [("a", "1"), ("b", "1")]
    store.accept(_batch(2, _event("a", 2, None)))
    store.drain(PROFILE_NEXT, Embedder())
    with pytest.raises(ValueError, match="caught up"):
        store.activate(old)
    store.drain(PROFILE, Embedder(), generation=old)
    store.activate(old)
    assert _hits(store) == [("b", "1")]


def test_invalid_limits_vectors_and_missing_index(database):
    store, _, schema = database
    with pytest.raises(ValueError, match="top_k"):
        store.search(PROFILE, lambda: pytest.fail("invalid query embedded"), top_k=51)
    store.accept(_batch(1, _event("a", 1)))
    for vector in ([0.0] * 4, [float("nan"), 0.0, 0.0, 0.0], [1.0, 0.0]):
        with pytest.raises(ValueError, match="Embedding"):
            store.drain(PROFILE, lambda _texts, vector=vector: [vector])
    store.drain(PROFILE, Embedder())
    from psycopg import sql

    active = store.status()["active_generation"]
    store.connection.execute(
        sql.SQL("DROP INDEX {}.{}").format(sql.Identifier(schema), sql.Identifier(f"g_{active}_ann"))
    )
    store.connection.execute(
        sql.SQL("CREATE INDEX {} ON {} (skill_id)").format(
            sql.Identifier(f"g_{active}_ann"), sql.Identifier(f"g_{active}")
        )
    )
    with pytest.raises(ValueError, match="index is unavailable"):
        store.search(PROFILE, lambda: pytest.fail("unavailable query embedded"))


def test_outdated_vectors_cannot_displace_current_candidates(database):
    store, _, _ = database
    outdated = [_event(f"old-{number}", 1, f"Nearest {number}") for number in range(240)]
    eligible = [_event(f"live-{number}", 1, "Still available") for number in range(10)]
    store.accept(_batch(1, *outdated, *eligible))
    store.drain(
        PROFILE,
        lambda texts: [
            [1.0, int(text.rsplit(" ", 1)[1]) / 2000, 0.0, 0.0] if "Nearest" in text else [0.8, 0.6, 0.0, 0.0]
            for text in texts
        ],
    )
    store.accept(_batch(2, *[_event(f"old-{number}", 2, "Pending replacement") for number in range(240)]))
    results = store.search(PROFILE, lambda: VECTOR)
    assert results["status"] == "degraded"
    assert [hit["skill_id"] for hit in results["candidates"]] == [f"live-{number}" for number in range(10)]


def test_lookup_sql_uses_hnsw_and_bounds_metadata_joins(database):
    from psycopg import sql

    store, _, _ = database
    store.accept(_batch(1, *[_event(f"skill-{number}", 1) for number in range(250)]))
    store.drain(PROFILE, Embedder())
    generation = store.status()["active_generation"]
    with store.connection.transaction():
        store.connection.execute("SET LOCAL enable_seqscan TO off")
        plan = store.connection.execute(
            sql.SQL("EXPLAIN (FORMAT JSON) ") + store._candidate_sql(generation, PROFILE.dimension),
            ("[1,0,0,0]", 40, "[1,0,0,0]", "", 10),
        ).fetchone()["QUERY PLAN"]
    encoded = json.dumps(plan)
    assert f'"Index Name": "g_{generation}_ann"' in encoded
    assert '"Node Type": "Seq Scan"' not in encoded


def test_acceptance_lock_is_bounded_and_retriable(database):
    from psycopg import sql
    from psycopg.errors import LockNotAvailable

    store, dsn, schema = database
    store.accept(_batch(1, _event("a", 1)))
    store.drain(PROFILE, Embedder())
    removing = _batch(2, _event("a", 2, None))
    with RegistryStore(dsn, schema=schema) as busy, busy.connection.transaction():
        busy.connection.execute(sql.SQL("SELECT 1 FROM {} FOR UPDATE").format(sql.Identifier(schema, "registry_state")))
        store.connection.execute("SET lock_timeout TO '100ms'")
        with pytest.raises(LockNotAvailable):
            store.accept(removing)
    assert _hits(store) == [("a", "1")]
    store.accept(removing)
    assert _hits(store) == []


def test_conflicting_stale_observation_rejects_entire_batch(database):
    store, _, _ = database
    store.accept(_batch(1, _event("a", 5, None)))
    store.accept(_batch(2, _event("a", 4, "Old")))
    with pytest.raises(ValueError, match="sequence conflict"):
        store.accept(_batch(3, _event("a", 4, "Conflicting"), _event("b", 1)))
    assert store.status()["accepted_cursor"] == 2
    store.accept(_batch(3, _event("a", 4, "Old")))
    store.drain(PROFILE, Embedder())
    assert _hits(store) == []


def test_status_reports_oldest_pending_reindex_seed(database):
    store, _, _ = database
    store.accept(_batch(1, _event("a", 1)))
    oldest = store.status()["generations"][0]["oldest_pending"]
    store.drain(PROFILE, Embedder())
    assert store.status()["generations"][0]["oldest_pending"] is None
    generation = store.reindex(PROFILE_NEXT)
    row = next(row for row in store.status()["generations"] if row["id"] == generation)
    assert row["pending_snapshot"]
    assert row["oldest_pending"] == oldest
