# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disposable database adapters for the registry overlap qualification benchmark."""

from __future__ import annotations

import os
from itertools import repeat
from time import monotonic, sleep
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    import numpy as np


_MAX_TOP_K = 50
_SEARCH_BUDGET = 200
_INITIAL_SEQUENCE = 0
_QDRANT_FULL_SCAN_KB = 10


def _check_matrix(vectors: np.ndarray, dimension: int) -> None:
    import numpy as np

    if vectors.ndim != 2 or vectors.shape[1] != dimension or vectors.dtype != np.float32:
        raise ValueError("Expected a float32 vector matrix with the configured dimension")


def _check_top_k(k: int) -> None:
    if type(k) is not int or not 1 <= k <= _MAX_TOP_K:
        raise ValueError(f"top-k must be an integer from 1 to {_MAX_TOP_K}")


def _index_nodes(node):
    if name := node.get("Index Name"):
        yield name
    for child in node.get("Plans", []):
        yield from _index_nodes(child)


def _copy_vectors(copy, vectors: np.ndarray) -> None:
    """Write PostgreSQL binary COPY frames in bounded NumPy batches."""
    import numpy as np

    dimension = vectors.shape[1]
    row_type = np.dtype(
        [
            ("columns", ">i2"),
            ("id_length", ">i4"),
            ("id", ">i8"),
            ("vector_length", ">i4"),
            ("dimension", ">i2"),
            ("reserved", ">i2"),
            ("vector", ">f4", (dimension,)),
        ]
    )
    copy.write(b"PGCOPY\n\xff\r\n\x00" + b"\x00" * 8)
    for start in range(0, len(vectors), 1024):
        batch = vectors[start : start + 1024]
        rows = np.zeros(len(batch), dtype=row_type)
        rows["columns"] = 2
        rows["id_length"] = 8
        rows["id"] = np.arange(start, start + len(batch))
        rows["vector_length"] = 4 + dimension * 4
        rows["dimension"] = dimension
        rows["vector"] = batch
        copy.write(rows.tobytes())
    copy.write(b"\xff\xff")


class PgVectorBackend:
    """Search a dedicated half-precision HNSW index and rerank float32 vectors."""

    def __init__(self, dsn: str, dimension: int):
        if type(dimension) is not int or not 1 <= dimension <= 4000:
            raise ValueError("pgvector half-precision HNSW dimension must be between 1 and 4000")

        from psycopg import sql
        from psycopg_pool import ConnectionPool

        self.dimension = dimension
        self.resource_name = f"registry_overlap_bench_{uuid4().hex}"
        self._dsn = dsn
        self._schema = sql.Identifier(self.resource_name)
        self._table = sql.Identifier(self.resource_name, "skills")
        self._halfvec = sql.SQL(f"halfvec({dimension})")
        self._vector = sql.SQL(f"vector({dimension})")
        self._search = {
            exclude: sql.SQL(
                """
                WITH candidate_vectors AS MATERIALIZED (
                    SELECT skill_id, embedding
                    FROM {table}
                    WHERE embedding IS NOT NULL {exclude}
                    ORDER BY embedding::{halfvec} <=> %s::{halfvec}
                    LIMIT %s
                )
                SELECT skill_id FROM candidate_vectors
                ORDER BY embedding <=> %s::{vector}, skill_id
                LIMIT %s
                """
            ).format(
                table=self._table,
                exclude=sql.SQL("AND skill_id <> %s" if exclude else ""),
                halfvec=self._halfvec,
                vector=self._vector,
            )
            for exclude in (False, True)
        }
        self._write = sql.SQL(
            """
            INSERT INTO {table} AS current (skill_id, sequence, embedding) VALUES (%s, %s, %s::{vector})
            ON CONFLICT (skill_id) DO UPDATE SET sequence = EXCLUDED.sequence, embedding = EXCLUDED.embedding
            WHERE current.sequence < EXCLUDED.sequence
            """
        ).format(table=self._table, vector=self._vector)
        self._visible = sql.SQL(
            "SELECT 1 FROM {} WHERE skill_id = %s AND sequence = %s AND embedding IS NOT NULL"
        ).format(self._table)
        self._pool = ConnectionPool(
            dsn,
            kwargs={"autocommit": True, "connect_timeout": 5},
            open=False,
            min_size=2,
            max_size=4,
            timeout=5,
            configure=self._configure,
        )

    @staticmethod
    def _configure(connection) -> None:
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = '5s'")
            cursor.execute("SET lock_timeout = '5s'")
            cursor.execute("SET enable_seqscan = off")
            cursor.execute(f"SET hnsw.ef_search = {_SEARCH_BUDGET}")
            cursor.execute("SET hnsw.iterative_scan = 'strict_order'")
            cursor.execute("SET hnsw.max_scan_tuples = 20000")

    def prepare(self, vectors: np.ndarray) -> dict[str, object]:
        from psycopg import connect, sql

        _check_matrix(vectors, self.dimension)
        workers = int(os.environ.get("SKILLEVALUATOR_BENCHMARK_POSTGRES_BUILD_WORKERS", "0"))
        if not 0 <= workers <= 4:
            raise ValueError("PostgreSQL benchmark build workers must be between 0 and 4")

        with connect(self._dsn, connect_timeout=5) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_setting('server_version'), extversion FROM pg_extension WHERE extname='vector'"
            )
            versions = cursor.fetchone()
            if versions is None:
                raise RuntimeError("Install the pgvector extension in the benchmark database before the trial")

            cursor.execute(sql.SQL("CREATE SCHEMA {}").format(self._schema))
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE {} (skill_id bigint PRIMARY KEY, sequence bigint NOT NULL DEFAULT 0, embedding {})"
                ).format(self._table, self._vector)
            )
            with cursor.copy(
                sql.SQL("COPY {} (skill_id, embedding) FROM STDIN (FORMAT BINARY)").format(self._table)
            ) as copy:
                _copy_vectors(copy, vectors)
            # Parallel builds need a Docker shared-memory mount larger than maintenance_work_mem.
            cursor.execute(f"SET LOCAL max_parallel_maintenance_workers = {workers}")
            cursor.execute("SET LOCAL maintenance_work_mem = '1536MB'")
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX skills_embedding_hnsw ON {} USING hnsw ((embedding::{}) halfvec_cosine_ops) "
                    "WHERE embedding IS NOT NULL"
                ).format(self._table, self._halfvec)
            )
            cursor.execute(sql.SQL("ANALYZE {}").format(self._table))
            cursor.execute("SELECT pg_total_relation_size(%s::regclass)", (f"{self.resource_name}.skills",))
            storage_bytes = cursor.fetchone()[0]

        self._pool.open(wait=True, timeout=10)
        planner_uses_ann = False
        if len(vectors):
            with self._pool.connection() as connection:
                plan = connection.execute(
                    sql.SQL("EXPLAIN (FORMAT JSON) ") + self._search[True],
                    self._search_params(vectors[0], 10, 0),
                ).fetchone()[0][0]["Plan"]
            planner_uses_ann = "skills_embedding_hnsw" in _index_nodes(plan)
            if not planner_uses_ann:
                raise RuntimeError("PostgreSQL benchmark search did not plan an HNSW index scan")

        return {
            "resource_name": self.resource_name,
            "engine_version": versions[0],
            "extension_version": versions[1],
            "dimension": self.dimension,
            "ann_indexed_count": len(vectors),
            "planner_uses_ann": planner_uses_ann,
            "hnsw_ef_search": _SEARCH_BUDGET,
            "hnsw_max_scan_tuples": 20000,
            "table_and_index_bytes": storage_bytes,
            "maintenance_workers": workers,
            "precision": "halfvec HNSW with stored float32 reranking",
        }

    @staticmethod
    def _search_params(vector: np.ndarray, k: int, exclude: int | None) -> tuple:
        literal = str(vector.tolist())
        candidate_count = min(_SEARCH_BUDGET, max(40, 4 * k))
        params = (literal, candidate_count, literal, k)
        return params if exclude is None else (exclude, *params)

    def search(self, vector: np.ndarray, k: int, exclude: int | None = None) -> list[int]:
        _check_top_k(k)
        with self._pool.connection() as connection:
            rows = connection.execute(
                self._search[exclude is not None], self._search_params(vector, k, exclude)
            ).fetchall()
        return [row[0] for row in rows]

    def upsert(self, skill_id: int, sequence: int, vector: np.ndarray) -> None:
        with self._pool.connection() as connection:
            connection.execute(self._write, (skill_id, sequence, str(vector.tolist())))

    def remove(self, skill_id: int, sequence: int) -> None:
        with self._pool.connection() as connection:
            connection.execute(self._write, (skill_id, sequence, None))

    def visible(self, skill_id: int, sequence: int) -> bool:
        with self._pool.connection() as connection:
            return connection.execute(self._visible, (skill_id, sequence)).fetchone() is not None

    def close(self) -> None:
        self._pool.close()


class QdrantBackend:
    """Search a dedicated collection of full-precision, revisioned Qdrant points."""

    def __init__(self, url: str, dimension: int):
        if type(dimension) is not int or dimension < 1:
            raise ValueError("Qdrant vector dimension must be a positive integer")

        from qdrant_client import QdrantClient

        grpc_port = os.environ.get("SKILLEVALUATOR_BENCHMARK_QDRANT_GRPC_PORT")
        self.dimension = dimension
        self.resource_name = f"registry_overlap_bench_{uuid4().hex}"
        self._url = url
        self._transport = "gRPC" if grpc_port else "HTTP"
        self._client = QdrantClient(
            url=url,
            prefer_grpc=bool(grpc_port),
            grpc_port=int(grpc_port) if grpc_port else 6334,
            timeout=15,
            check_compatibility=False,
        )

    def prepare(self, vectors: np.ndarray) -> dict[str, object]:
        from qdrant_client import models

        _check_matrix(vectors, self.dimension)
        self._client.create_collection(
            self.resource_name,
            vectors_config={"embedding": models.VectorParams(size=self.dimension, distance=models.Distance.COSINE)},
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=64, full_scan_threshold=_QDRANT_FULL_SCAN_KB),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0, default_segment_number=2),
        )
        if len(vectors):
            self._client.upload_collection(
                self.resource_name,
                vectors={"embedding": vectors},
                payload=repeat({"sequence": _INITIAL_SEQUENCE}, len(vectors)),
                ids=range(len(vectors)),
                batch_size=256,
                wait=True,
            )
        threshold_kb = min(10000, max(_QDRANT_FULL_SCAN_KB, vectors.nbytes // 4096))
        self._client.update_collection(
            self.resource_name,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=threshold_kb),
        )
        index_expected = vectors.nbytes > 2 * threshold_kb * 1024
        deadline = monotonic() + 900
        while True:
            info = self._client.get_collection(self.resource_name)
            indexed_count = info.indexed_vectors_count or 0
            if info.status == models.CollectionStatus.GREEN and (indexed_count > 0 or not index_expected):
                break
            if info.status == models.CollectionStatus.RED or monotonic() >= deadline:
                raise RuntimeError(f"Qdrant collection {self.resource_name} did not finish building its HNSW index")
            sleep(0.25)

        if self._client.count(self.resource_name, exact=True).count != len(vectors):
            raise RuntimeError(f"Qdrant collection {self.resource_name} did not receive the complete corpus")

        probes = {}
        planner_uses_ann = False
        if indexed_count:
            self.search(vectors[0], 1)
            self.search(vectors[0], 1, exclude=0)
            probes = self._search_probes()
            planner_uses_ann = probes.get("unfiltered_hnsw", 0) > 0 and probes.get("filtered_large_cardinality", 0) > 0
            if index_expected and not planner_uses_ann:
                raise RuntimeError("Qdrant benchmark search did not confirm HNSW use with and without self-exclusion")

        return {
            "resource_name": self.resource_name,
            "engine_version": self._client.info().version,
            "dimension": self.dimension,
            "ann_indexed_count": indexed_count,
            "planner_uses_ann": planner_uses_ann,
            "search_probe_counts": probes,
            "hnsw_ef_search": _SEARCH_BUDGET,
            "indexing_threshold_kb": threshold_kb,
            "full_scan_threshold_kb": _QDRANT_FULL_SCAN_KB,
            "transport": self._transport,
            "precision": "float32 HNSW",
        }

    def _search_probes(self) -> dict[str, int]:
        import httpx

        response = httpx.get(f"{self._url.rstrip('/')}/telemetry", params={"details_level": 4}, timeout=5)
        response.raise_for_status()
        counts: dict[str, int] = {}
        collections = response.json()["result"]["collections"]["collections"]
        collection = next(collection for collection in collections if collection["id"] == self.resource_name)
        for shard in collection["shards"]:
            for segment in (shard.get("local") or {}).get("segments", []):
                for index in segment.get("vector_index_searches", []):
                    for name, counter in index.items():
                        if isinstance(counter, dict) and "count" in counter:
                            counts[name] = counts.get(name, 0) + counter["count"]
        return counts

    def search(self, vector: np.ndarray, k: int, exclude: int | None = None) -> list[int]:
        from qdrant_client import models

        _check_top_k(k)
        query_filter = None if exclude is None else models.Filter(must_not=[models.HasIdCondition(has_id=[exclude])])
        results = self._client.query_points(
            self.resource_name,
            using="embedding",
            query=vector,
            query_filter=query_filter,
            search_params=models.SearchParams(hnsw_ef=_SEARCH_BUDGET, exact=False),
            timeout=5,
            limit=k,
            with_vectors=False,
            with_payload=False,
        ).points
        return [int(result.id) for result in sorted(results, key=lambda point: (-point.score, int(point.id)))]

    def _write(self, skill_id: int, sequence: int, vector: np.ndarray | None) -> None:
        from qdrant_client import models

        # A vectorless point retains its sequence guard without entering vector search.
        result = self._client.upsert(
            self.resource_name,
            points=[
                models.PointStruct(
                    id=skill_id,
                    vector={} if vector is None else {"embedding": vector.tolist()},
                    payload={"sequence": sequence, "deleted": vector is None},
                )
            ],
            update_filter=models.Filter(must=[models.FieldCondition(key="sequence", range=models.Range(lt=sequence))]),
            ordering=models.WriteOrdering.STRONG,
            wait=True,
        )
        if result.status != models.UpdateStatus.COMPLETED:
            raise RuntimeError("Qdrant mutation did not complete within the request budget")

    def upsert(self, skill_id: int, sequence: int, vector: np.ndarray) -> None:
        self._write(skill_id, sequence, vector)

    def remove(self, skill_id: int, sequence: int) -> None:
        self._write(skill_id, sequence, None)

    def visible(self, skill_id: int, sequence: int) -> bool:
        points = self._client.retrieve(self.resource_name, ids=[skill_id], with_payload=True, with_vectors=False)
        if not points or not (payload := points[0].payload):
            return False
        return payload.get("sequence") == sequence and not payload.get("deleted", False)

    def close(self) -> None:
        self._client.close()
