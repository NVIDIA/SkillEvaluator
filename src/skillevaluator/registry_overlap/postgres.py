# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional event replay and bounded overlap retrieval over pgvector."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

from psycopg import connect, sql
from psycopg.errors import QueryCanceled, UndefinedTable
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from skillevaluator.embedding.client import SimilarityConfigError, normalize_embedding_vector
from skillevaluator.registry_overlap.events import MAX_BATCH_EVENTS, RegistryBatch, identifier

_MAX_GENERATIONS = 3
_EMBEDDING_BATCH_SIZE = 64
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmbeddingProfile:
    provider: str
    model: str
    endpoint_fingerprint: str
    dimension: int
    revision: str
    mode: str = "description-v1"

    def __post_init__(self):
        if type(self.dimension) is not int or not 1 <= self.dimension <= 4000:
            raise ValueError("Registry profile dimension must be in 1..4000")
        identifier(self.provider, "provider")
        identifier(self.model, "model")
        identifier(self.revision, "profile revision")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.endpoint_fingerprint):
            raise ValueError("Registry profile endpoint fingerprint is invalid")
        if self.mode != "description-v1":
            raise ValueError("Unsupported registry embedding mode")


def _vector_text(vector: list[float], dimension: int) -> str:
    try:
        unit = normalize_embedding_vector(vector, dimension, context="Registry embedding")
    except SimilarityConfigError:
        raise ValueError("Embedding is invalid for the registry profile") from None
    return "[" + ",".join(map(str, unit)) + "]"


class RegistryStore:
    """One operator session. Embedding calls run outside database transactions."""

    def __init__(self, dsn: str, *, schema: str = "skillevaluator_overlap"):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema) or schema == "public" or schema.startswith("pg_"):
            raise ValueError("Registry database schema name is invalid")
        self.schema = schema
        self.connection = connect(dsn, autocommit=True, row_factory=dict_row, connect_timeout=5)
        try:
            self.connection.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
            self.connection.execute("SET statement_timeout TO '30s'")
            self.connection.execute("SET lock_timeout TO '5s'")
        except BaseException:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def _row(self, query, parameters=()):
        return self.connection.execute(query, parameters).fetchone()

    def _control(self, *, lock=False):
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(self.schema, "registry_state"))
        if lock:
            query += sql.SQL(" FOR UPDATE")
        try:
            state = self._row(query)
        except UndefinedTable:
            raise ValueError("Registry is unavailable. Initialize it before use.") from None
        if state is None or state["version"] != _SCHEMA_VERSION:
            raise ValueError("Registry is unavailable for this schema version")
        return state

    def initialize(self, source: str, profile: EmbeddingProfile) -> int:
        """Create one source registry and its first empty active generation."""
        identifier(source, "source")
        with self.connection.transaction():
            self.connection.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    profile jsonb NOT NULL, checkpoint bigint NOT NULL
                );
                CREATE TABLE IF NOT EXISTS registry_state (
                    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
                    version integer NOT NULL, source text NOT NULL,
                    accepted_cursor bigint NOT NULL DEFAULT 0, tail bigint NOT NULL DEFAULT 0,
                    active_generation bigint REFERENCES generations(id)
                );
                CREATE TABLE IF NOT EXISTS batches (cursor bigint PRIMARY KEY, digest text NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    number bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    skill_id text NOT NULL, sequence bigint NOT NULL, revision text NOT NULL,
                    digest text NOT NULL, text text, artifact text, manifest_sha256 text, input_sha256 text,
                    observed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
                    UNIQUE (skill_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS heads (
                    skill_id text PRIMARY KEY, number bigint NOT NULL REFERENCES events(number)
                );
                CREATE TABLE IF NOT EXISTS seeds (
                    generation bigint REFERENCES generations(id) ON DELETE CASCADE,
                    number bigint REFERENCES events(number), PRIMARY KEY (generation, number)
                );
            """)
            self.connection.execute(
                "INSERT INTO registry_state(version, source) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (_SCHEMA_VERSION, source),
            )
            state = self._control(lock=True)
            if state["source"] != source:
                raise ValueError("Registry source differs from this database")
            if state["active_generation"] is not None:
                generation = state["active_generation"]
                self._require_profile(generation, profile)
                return generation
            generation = self._create_generation(profile, 0)
            self.connection.execute("UPDATE registry_state SET active_generation = %s", (generation,))
            return generation

    @staticmethod
    def _table(generation: int) -> sql.Identifier:
        return sql.Identifier(f"g_{generation}")

    def _create_generation(self, profile: EmbeddingProfile, checkpoint: int) -> int:
        generation = self._row(
            "INSERT INTO generations(profile, checkpoint) VALUES (%s, %s) RETURNING id",
            (Jsonb(asdict(profile)), checkpoint),
        )["id"]
        table = self._table(generation)
        dimension = sql.Literal(profile.dimension)
        self.connection.execute(
            sql.SQL("""
            CREATE TABLE {table} (
                skill_id text PRIMARY KEY, number bigint NOT NULL REFERENCES events(number),
                input_sha256 text NOT NULL, embedding vector({dimension}) NOT NULL
            )
        """).format(table=table, dimension=dimension)
        )
        self.connection.execute(
            sql.SQL("""
            CREATE INDEX {index} ON {table}
            USING hnsw ((embedding::halfvec({dimension})) halfvec_cosine_ops)
        """).format(index=sql.Identifier(f"g_{generation}_ann"), table=table, dimension=dimension)
        )
        self.connection.execute(sql.SQL("CREATE INDEX ON {} (input_sha256)").format(table))
        return generation

    def _require_profile(self, generation, profile):
        row = self._row("SELECT * FROM generations WHERE id = %s", (generation,))
        if row is None:
            raise ValueError("Registry generation is unavailable")
        if row["profile"] != asdict(profile):
            raise ValueError("Embedding profile differs from the registry generation")
        return row

    def accept(self, batch: RegistryBatch) -> int:
        """Commit an entire cursor and its current events before doing provider work."""
        with self.connection.transaction():
            state = self._control(lock=True)
            if batch.source != state["source"]:
                raise ValueError("Registry batch source differs from this database")
            if batch.cursor <= state["accepted_cursor"]:
                previous = self._row("SELECT digest FROM batches WHERE cursor = %s", (batch.cursor,))
                if previous is not None and previous["digest"] == batch.digest:
                    return 0
                raise ValueError("Registry batch cursor conflict")
            if batch.cursor != state["accepted_cursor"] + 1:
                raise ValueError("Registry batch cursor skips an accepted cursor")

            count, tail = 0, state["tail"]
            changed = set()
            for event in batch.events:
                previous = self._row(
                    "SELECT digest FROM events WHERE skill_id = %s AND sequence = %s",
                    (event.skill_id, event.sequence),
                )
                if previous is not None:
                    if previous["digest"] != event.digest:
                        raise ValueError("Registry event sequence conflict")
                    continue
                head = self._row(
                    """
                    SELECT e.sequence FROM heads h JOIN events e USING (number) WHERE h.skill_id = %s
                """,
                    (event.skill_id,),
                )
                if head is not None and head["sequence"] > event.sequence:
                    continue
                tail = self._row(
                    """
                    INSERT INTO events(skill_id, sequence, revision, digest, text, artifact, manifest_sha256, input_sha256)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING number
                """,
                    (
                        event.skill_id,
                        event.sequence,
                        event.revision,
                        event.digest,
                        event.text,
                        event.artifact,
                        event.manifest_sha256,
                        event.input_sha256,
                    ),
                )["number"]
                self.connection.execute(
                    """
                    INSERT INTO heads VALUES (%s, %s)
                    ON CONFLICT (skill_id) DO UPDATE SET number = excluded.number
                """,
                    (event.skill_id, tail),
                )
                changed.add(event.skill_id)
                count += 1

            if changed:
                for generation in self.connection.execute("SELECT id FROM generations").fetchall():
                    table = self._table(generation["id"])
                    # Remove outdated vectors before committing the new head. An
                    # unchanged embedding input can keep its vector and publish
                    # the new revision in the same transaction.
                    self.connection.execute(
                        sql.SQL("""
                        DELETE FROM {} g USING heads h, events e
                        WHERE g.skill_id = h.skill_id AND h.number = e.number
                        AND g.skill_id = ANY(%s) AND g.input_sha256 IS DISTINCT FROM e.input_sha256
                    """).format(table),
                        (list(changed),),
                    )
                    self.connection.execute(
                        sql.SQL("""
                        UPDATE {} g SET number = h.number FROM heads h, events e
                        WHERE g.skill_id = h.skill_id AND h.number = e.number
                        AND g.skill_id = ANY(%s) AND g.input_sha256 = e.input_sha256 AND g.number <> h.number
                    """).format(table),
                        (list(changed),),
                    )
            self.connection.execute("INSERT INTO batches VALUES (%s, %s)", (batch.cursor, batch.digest))
            self.connection.execute("UPDATE registry_state SET accepted_cursor = %s, tail = %s", (batch.cursor, tail))
            return count

    def _pending(self, generation, checkpoint, limit):
        seeded = self.connection.execute(
            """
            SELECT e.*, h.number = e.number AS current
            FROM seeds s JOIN events e USING (number) JOIN heads h ON h.skill_id = e.skill_id
            WHERE s.generation = %s ORDER BY e.number LIMIT %s
        """,
            (generation, limit),
        ).fetchall()
        if seeded:
            return seeded, True
        return self.connection.execute(
            """
            SELECT e.*, h.number = e.number AS current
            FROM events e JOIN heads h ON h.skill_id = e.skill_id
            WHERE e.number > %s ORDER BY e.number LIMIT %s
        """,
            (checkpoint, limit),
        ).fetchall(), False

    def _embeddings(self, pending, generation, profile, embed):
        vectors, needed = {}, {}
        table = self._table(generation)
        for event in pending:
            key = event["input_sha256"]
            if not event["current"] or key is None or key in vectors or key in needed:
                continue
            reuse = self._row(
                sql.SQL("SELECT embedding::text AS vector FROM {} WHERE input_sha256 = %s LIMIT 1").format(table),
                (key,),
            )
            if reuse is None:
                needed[key] = event["text"]
            else:
                vectors[key] = reuse["vector"]
        if needed:
            try:
                result = embed(list(needed.values()))
                if len(result) != len(needed):
                    raise ValueError("Unexpected vector count")
                vectors.update(
                    (key, _vector_text(vector, profile.dimension)) for key, vector in zip(needed, result, strict=True)
                )
            except Exception:
                raise ValueError("Embedding failed. The pending registry checkpoint can be retried.") from None
        return vectors

    def _apply_pending(self, pending, generation, vectors, seeded):
        with self.connection.transaction():
            self._control(lock=True)
            for event in pending:
                key = event["input_sha256"]
                if key not in vectors:
                    continue
                # A newer event (including a removal) can commit while embedding runs.
                self.connection.execute(
                    sql.SQL("""
                    INSERT INTO {} (skill_id, number, input_sha256, embedding)
                    SELECT skill_id, number, %s, %s::vector FROM heads
                    WHERE skill_id = %s AND number = %s
                    ON CONFLICT (skill_id) DO UPDATE SET
                        number = excluded.number, input_sha256 = excluded.input_sha256, embedding = excluded.embedding
                """).format(self._table(generation)),
                    (key, vectors[key], event["skill_id"], event["number"]),
                )
            if seeded:
                self.connection.execute(
                    "DELETE FROM seeds WHERE generation = %s AND number = ANY(%s)",
                    (generation, [event["number"] for event in pending]),
                )
            else:
                self.connection.execute(
                    "UPDATE generations SET checkpoint = %s WHERE id = %s", (pending[-1]["number"], generation)
                )

    def drain(
        self,
        profile: EmbeddingProfile,
        embed: Callable[[list[str]], list[list[float]]],
        *,
        generation: int | None = None,
        limit: int = MAX_BATCH_EVENTS,
    ) -> int:
        """Apply one bounded work cycle. A session lock prevents concurrent workers per generation."""
        if not 1 <= limit <= MAX_BATCH_EVENTS:
            raise ValueError("Registry drain limit is outside the supported range")
        generation = self._control()["active_generation"] if generation is None else generation
        if not self._row("SELECT pg_try_advisory_lock(hashtext(%s), %s::integer) AS locked", (self.schema, generation))[
            "locked"
        ]:
            raise ValueError("Registry generation already has an active worker")
        try:
            self._require_profile(generation, profile)
            count = 0
            while count < limit:
                row = self._require_profile(generation, profile)
                pending, seeded = self._pending(
                    generation, row["checkpoint"], min(_EMBEDDING_BATCH_SIZE, limit - count)
                )
                if not pending:
                    break
                vectors = self._embeddings(pending, generation, profile, embed)
                self._apply_pending(pending, generation, vectors, seeded)
                count += len(pending)
            return count
        finally:
            self.connection.execute("SELECT pg_advisory_unlock(hashtext(%s), %s::integer)", (self.schema, generation))

    def reindex(self, profile: EmbeddingProfile) -> int:
        """Record a stable source snapshot, then drain it before replaying later events."""
        with self.connection.transaction():
            state = self._control(lock=True)
            if self._row("SELECT count(*) AS count FROM generations")["count"] >= _MAX_GENERATIONS:
                raise ValueError("Registry generation limit reached. Retire an inactive generation.")
            generation = self._create_generation(profile, state["tail"])
            self.connection.execute(
                """
                INSERT INTO seeds SELECT %s, h.number FROM heads h JOIN events e USING (number) WHERE e.text IS NOT NULL
            """,
                (generation,),
            )
            return generation

    def _index_ready(self, generation):
        return self._row(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                JOIN pg_am am ON am.oid = c.relam JOIN pg_opclass op ON op.oid = i.indclass[0]
                WHERE i.indexrelid = to_regclass(%s) AND i.indrelid = to_regclass(%s)
                    AND am.amname = 'hnsw' AND op.opcname = 'halfvec_cosine_ops'
                    AND i.indisready AND i.indisvalid AND i.indpred IS NULL
                    AND pg_get_indexdef(i.indexrelid, 1, true) = (
                        SELECT '(embedding::halfvec(' || (profile->>'dimension') || '))'
                        FROM generations WHERE id = %s
                    )
            ) AS ready
        """,
            (f"{self.schema}.g_{generation}_ann", f"{self.schema}.g_{generation}", generation),
        )["ready"]

    def activate(self, generation: int) -> None:
        """Activate a caught-up generation. The same precondition protects rollback."""
        with self.connection.transaction():
            state = self._control(lock=True)
            row = self._row("SELECT checkpoint FROM generations WHERE id = %s", (generation,))
            pending = self._row("SELECT 1 FROM seeds WHERE generation = %s LIMIT 1", (generation,))
            if row is None or pending or row["checkpoint"] < state["tail"] or not self._index_ready(generation):
                raise ValueError("Registry generation is unavailable or has not caught up")
            self.connection.execute("UPDATE registry_state SET active_generation = %s", (generation,))

    def retire(self, generation: int) -> None:
        """Remove an explicitly selected inactive and idle generation."""
        with self.connection.transaction():
            state = self._control(lock=True)
            if generation == state["active_generation"]:
                raise ValueError("The active registry generation cannot be retired")
            locked = self._row(
                "SELECT pg_try_advisory_xact_lock(hashtext(%s), %s::integer) AS locked", (self.schema, generation)
            )["locked"]
            if not locked:
                raise ValueError("Registry generation has an active worker")
            row = self._row("DELETE FROM generations WHERE id = %s RETURNING id", (generation,))
            if row is None:
                raise ValueError("Registry generation is unavailable")
            self.connection.execute(sql.SQL("DROP TABLE {}").format(self._table(generation)))

    def status(self):
        with self.connection.transaction():
            self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            state = self._control()
            generations = self.connection.execute("""
                SELECT g.id, g.profile, g.checkpoint,
                    EXISTS (SELECT 1 FROM seeds s WHERE s.generation = g.id) AS pending_snapshot,
                    (SELECT observed_at FROM events
                        WHERE number > g.checkpoint ORDER BY number LIMIT 1) AS oldest_pending
                FROM generations g ORDER BY g.id
            """).fetchall()
            return {**state, "generations": generations, "index_ready": self._index_ready(state["active_generation"])}

    def _candidate_sql(self, generation: int, dimension: int):
        return sql.SQL("""
            WITH candidates AS MATERIALIZED (
                SELECT skill_id, number, embedding FROM {table}
                ORDER BY embedding::halfvec({dimension}) <=> %s::halfvec({dimension}) LIMIT %s
            )
            SELECT c.skill_id, e.revision, e.sequence, e.artifact, e.manifest_sha256, e.input_sha256,
                greatest(-1, least(1, 1 - (c.embedding <=> %s::vector))) AS score
            FROM candidates c JOIN heads h USING (skill_id) JOIN events e ON e.number = c.number
            WHERE h.number = c.number AND c.skill_id <> %s
            ORDER BY score DESC, c.skill_id COLLATE "C" LIMIT %s
                """).format(table=self._table(generation), dimension=sql.Literal(dimension))

    def search(
        self,
        profile: EmbeddingProfile,
        embed: Callable[[], list[float]],
        *,
        top_k: int = 10,
        exclude: str | None = None,
    ):
        """Look up and rerank a bounded shortlist in one consistent read snapshot."""
        if type(top_k) is not int or not 1 <= top_k <= 50:
            raise ValueError("Registry top_k must be an integer from 1 to 50")
        if exclude is not None:
            identifier(exclude, "skill ID")
        generation = self._control()["active_generation"]
        self._require_profile(generation, profile)
        if not self._index_ready(generation):
            raise ValueError("Registry search index is unavailable")
        try:
            vector = _vector_text(embed(), profile.dimension)
        except Exception:
            raise ValueError("Embedding failed for the registry query") from None
        try:
            with self.connection.transaction():
                self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                self.connection.execute("SET LOCAL statement_timeout TO '250ms'")
                state = self._control()
                if state["active_generation"] != generation:
                    raise ValueError("Registry generation changed during search. Retry the query.")
                row = self._require_profile(generation, profile)
                if not self._index_ready(generation):
                    raise ValueError("Registry search index is unavailable")
                self.connection.execute("SET LOCAL enable_seqscan TO off")
                self.connection.execute("SET LOCAL hnsw.ef_search TO 200")
                candidates = self.connection.execute(
                    self._candidate_sql(generation, profile.dimension),
                    (
                        vector,
                        min(200, max(40, 4 * top_k)),
                        vector,
                        exclude or "",
                        top_k,
                    ),
                ).fetchall()
        except (QueryCanceled, UndefinedTable):
            raise ValueError("Registry index is unavailable or search timed out") from None
        return {
            "status": "degraded" if row["checkpoint"] < state["tail"] else "ok",
            "source": state["source"],
            "generation": generation,
            "accepted_cursor": state["accepted_cursor"],
            "applied_event": row["checkpoint"],
            "candidates": candidates,
        }
