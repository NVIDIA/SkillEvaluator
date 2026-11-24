# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare vector-index implementations using precomputed Tier 2 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    VectorCorpus = NDArray[np.float32]

TIE_TOLERANCE = 2e-6
FORMAT_VERSION = 1
MAX_WORKLOAD_BYTES = 1024 * 1024
MAX_RELATED_IDS = 100
MIN_LABELED_QUERIES = 100
CORPUS_BLOCK_ROWS = 1024
ORACLE_BATCH = 16
UPSERT_DEADLINE = 60.0
DELETE_DEADLINE = 5.0
VISIBILITY_INTERVAL = 0.02


@dataclass(frozen=True)
class Workload:
    """Queries and exhaustive similarity cutoffs for one normalized corpus."""

    corpus_sha256: str
    size: int
    dimension: int
    k: int
    query_ids: list[int]
    cutoffs: list[float]
    strict_counts: list[int]


@dataclass(frozen=True)
class MutationTiming:
    kind: str
    ack_ms: float
    visibility_ms: float


@dataclass(frozen=True)
class Sample:
    lateness_ms: float
    latency_ms: float | None = None
    error: str | None = None
    mutation: MutationTiming | None = None


class VisibilityTimeout(TimeoutError):
    """The expected revision was still invisible when the time budget expired."""


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _normalize(block: VectorCorpus) -> VectorCorpus:
    import numpy as np

    if not np.isfinite(block).all():
        raise ValueError("Corpus vectors must contain finite values")
    wide = block.astype(np.float64)
    norms = np.sqrt(np.einsum("ij,ij->i", wide, wide))
    if np.any(norms == 0):
        raise ValueError("Corpus vectors must have nonzero norms")
    # Preserve the exact float32 representation of precomputed unit vectors.
    needs_normalization = np.abs(norms - 1) > 1e-7
    if np.any(needs_normalization):
        block = block.copy()
        block[needs_normalization] = wide[needs_normalization] / norms[needs_normalization, None]
    return block


def _blocks(size: int):
    for start in range(0, size, CORPUS_BLOCK_ROWS):
        yield slice(start, min(start + CORPUS_BLOCK_ROWS, size))


def load_corpus(path: Path, *, size: int, dimension: int) -> VectorCorpus:
    """Load a bounded prefix of a .npy float32 corpus, normalizing as needed."""
    import numpy as np

    if path.suffix != ".npy":
        raise ValueError("Vector corpus must be a .npy file")
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(raw, np.ndarray):
        raw.close()
        raise ValueError("Vector corpus must be a single .npy array")
    if raw.dtype != np.dtype("<f4") or raw.ndim != 2:
        raise ValueError("Vector corpus must be a two-dimensional little-endian float32 array")
    if not raw.flags.c_contiguous:
        raise ValueError("Vector corpus must be stored in row-major order")
    if raw.shape[0] < size or raw.shape[1] != dimension:
        raise ValueError(f"Vector corpus must contain at least {size} rows of dimension {dimension}")
    vectors = raw[:size]
    normalized = None
    for block in _blocks(size):
        original = vectors[block]
        checked = _normalize(original)
        if normalized is None and checked is not original:
            normalized = vectors.copy()
        if normalized is not None:
            normalized[block] = checked
    return normalized if normalized is not None else vectors


def synthetic_corpus(size: int, dimension: int, seed: int) -> VectorCorpus:
    """Build a deterministic clustered corpus with some exactly repeated vectors."""
    import numpy as np

    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((min(1024, max(2, size // 32)), dimension), dtype=np.float32)
    vectors = np.empty((size, dimension), dtype=np.float32)
    for block in _blocks(size):
        count = block.stop - block.start
        groups = rng.integers(len(centers), size=count)
        noise = rng.standard_normal((count, dimension), dtype=np.float32)
        vectors[block] = _normalize(centers[groups] + np.float32(0.35) * noise)
    vectors[97::97] = vectors[96:-1:97]
    return vectors


def corpus_sha256(corpus: VectorCorpus) -> str:
    digest = hashlib.sha256()
    for block in _blocks(len(corpus)):
        digest.update(memoryview(corpus[block]).cast("B"))
    return digest.hexdigest()


def exact_workload(
    corpus: VectorCorpus, query_ids: list[int], *, k: int, digest: str | None = None, progress: bool = False
) -> Workload:
    """Batch exhaustive float32 cosine ranking via BLAS and retain tie cutoffs."""
    import numpy as np

    size, dimension = corpus.shape
    cutoffs = []
    strict_counts = []
    for start in range(0, len(query_ids), ORACLE_BATCH):
        selected = query_ids[start : start + ORACLE_BATCH]
        scores = corpus @ corpus[selected].T
        scores[selected, range(len(selected))] = -np.inf
        thresholds = np.partition(scores, size - k, axis=0)[size - k]
        required = np.count_nonzero(scores > thresholds + TIE_TOLERANCE, axis=0)
        cutoffs.extend(thresholds.tolist())
        strict_counts.extend(required.tolist())
        if progress and (start // ORACLE_BATCH + 1) % 16 == 0:
            _progress(f"Exact oracle {min(start + ORACLE_BATCH, len(query_ids))}/{len(query_ids)} queries")
    return Workload(digest or corpus_sha256(corpus), size, dimension, k, query_ids, cutoffs, strict_counts)


def recall_for_results(corpus: VectorCorpus, workload: Workload, query: int, found: list[int]) -> float:
    """Give credit for a tied boundary candidate up to the boundary's remaining slots."""
    import numpy as np

    own_id = workload.query_ids[query]
    _validate_results(found, workload.k, own_id)
    if any(type(entry_id) is not int or not 0 <= entry_id < workload.size for entry_id in found):
        raise ValueError("Backend returned an ID outside the prepared corpus")
    scores = corpus[found] @ corpus[own_id]
    threshold = workload.cutoffs[query]
    strict_hits = int(np.count_nonzero(scores > threshold + TIE_TOLERANCE))
    tie_hits = int(np.count_nonzero(np.abs(scores - threshold) <= TIE_TOLERANCE))
    strict_count = workload.strict_counts[query]
    return (min(strict_hits, strict_count) + min(tie_hits, workload.k - strict_count)) / workload.k


def _validate_results(found: list[int], k: int, own_id: int) -> None:
    if len(found) > k:
        raise ValueError("Backend returned more than k candidates")
    if len(set(found)) != len(found):
        raise ValueError("Backend returned a duplicate candidate")
    if own_id in found:
        raise ValueError("Backend returned the excluded candidate")


def _write_json(path: Path | None, data: object) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        with path.open("w", encoding="utf-8") as output:
            output.write(text)


def _read_json(path: Path) -> object:
    with path.open("rb") as source:
        data = source.read(MAX_WORKLOAD_BYTES + 1)
    if len(data) > MAX_WORKLOAD_BYTES:
        raise ValueError("Workload JSON exceeds its size limit")
    return json.loads(data)


def _query_ids(raw: object, size: int, count: int) -> list[int]:
    if (
        not isinstance(raw, list)
        or len(raw) != count
        or any(type(entry_id) is not int or not 0 <= entry_id < size for entry_id in raw)
        or len(set(raw)) != count
    ):
        raise ValueError("Query IDs must be unique in-range row numbers matching --queries")
    return raw


def _read_workload(path: Path, args: argparse.Namespace, digest: str) -> Workload:
    raw = _read_json(path)
    if (
        not isinstance(raw, dict)
        or type(raw.get("version")) is not int
        or raw["version"] != FORMAT_VERSION
        or raw.get("tie_tolerance") != TIE_TOLERANCE
    ):
        raise ValueError("Unsupported workload format")
    if raw.get("corpus_sha256") != digest:
        raise ValueError("Workload corpus fingerprint does not match the supplied vectors")
    if (raw.get("size"), raw.get("dimension"), raw.get("k")) != (args.size, args.dimension, args.k):
        raise ValueError("Workload size, dimension or k differs from the command arguments")
    query_ids = _query_ids(raw.get("query_ids"), args.size, args.queries)
    cutoffs = raw.get("cutoffs")
    strict = raw.get("strict_counts")
    if (
        not isinstance(cutoffs, list)
        or len(cutoffs) != args.queries
        or any(type(value) not in (int, float) or not -1.00001 <= value <= 1.00001 for value in cutoffs)
        or not isinstance(strict, list)
        or len(strict) != args.queries
        or any(type(value) is not int or not 0 <= value < args.k for value in strict)
    ):
        raise ValueError("Workload contains invalid exact-ranking cutoffs")
    return Workload(digest, args.size, args.dimension, args.k, query_ids, cutoffs, strict)


def _read_labels(path: Path, workload: Workload) -> dict[int, set[int]]:
    def unique_keys(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Label query IDs must be unique")
        return result

    with path.open("rb") as source:
        encoded = source.read(MAX_WORKLOAD_BYTES + 1)
    if len(encoded) > MAX_WORKLOAD_BYTES:
        raise ValueError("Labels JSON exceeds its size limit")
    try:
        raw = json.loads(encoded, object_pairs_hook=unique_keys)
    except RecursionError as exc:
        raise ValueError("Labels JSON exceeds its nesting limit") from exc
    if not isinstance(raw, dict) or not raw or len(raw) > len(workload.query_ids):
        raise ValueError("Labels must map selected query IDs to lists of related row IDs")

    selected = {str(query): query for query in workload.query_ids}
    labels = {}
    for key, related in raw.items():
        if key not in selected:
            raise ValueError("Labels must refer to selected query IDs")
        query = selected[key]
        if (
            not isinstance(related, list)
            or not 1 <= len(related) <= MAX_RELATED_IDS
            or any(
                type(entry_id) is not int or not 0 <= entry_id < workload.size or entry_id == query
                for entry_id in related
            )
            or len(set(related)) != len(related)
        ):
            raise ValueError("Each label must list distinct in-range related row IDs excluding the query itself")
        labels[query] = set(related)
    return labels


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    data = sorted(values)
    return {
        "count": len(data),
        "p50": round(data[math.ceil(len(data) * 0.50) - 1], 3),
        "p95": round(data[math.ceil(len(data) * 0.95) - 1], 3),
        "p99": round(data[math.ceil(len(data) * 0.99) - 1], 3),
        "max": round(data[-1], 3),
    }


def run_stream(
    operation: Callable[[int], MutationTiming | None],
    *,
    start: float,
    rate: int,
    duration: float,
    workers: int,
    lateness_limit: float = 1.0,
) -> dict[str, object]:
    """Schedule an open-loop request stream with bounded workers and pending work."""
    count = math.ceil(rate * duration)
    max_pending = workers + math.ceil(rate * lateness_limit)
    pending: set[Future[Sample]] = set()
    samples: list[Sample] = []

    def invoke(index: int, due: float) -> Sample:
        began = time.monotonic()
        lateness_ms = 1000 * max(0.0, began - due)
        if began - due > lateness_limit:
            return Sample(lateness_ms)
        try:
            outcome = operation(index)
        except Exception as exc:
            return Sample(lateness_ms, error=type(exc).__name__)
        return Sample(lateness_ms, 1000 * (time.monotonic() - began), mutation=outcome)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index in range(count):
            due = start + index / rate
            remaining = due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            finished = {job for job in pending if job.done()}
            pending -= finished
            samples.extend(job.result() for job in finished)
            if len(pending) >= max_pending:
                samples.append(Sample(1000 * max(0.0, time.monotonic() - due)))
            else:
                pending.add(executor.submit(invoke, index, due))
        samples.extend(job.result() for job in pending)
    succeeded = [sample for sample in samples if sample.latency_ms is not None]
    errors = Counter(sample.error for sample in samples if sample.error is not None)
    mutations = [sample.mutation for sample in succeeded if sample.mutation is not None]
    return {
        "rate_per_second": rate,
        "duration_seconds": duration,
        "workers": workers,
        "scheduled": count,
        "completed": len(succeeded),
        "dropped": count - len(succeeded) - sum(errors.values()),
        "errors": dict(errors),
        "late_start": sum(sample.lateness_ms >= 1000 / rate for sample in samples),
        "start_lateness_limit_ms": 1000 * lateness_limit,
        "dispatch_lateness_ms": _distribution([sample.lateness_ms for sample in samples]),
        "latency_ms": _distribution([sample.latency_ms for sample in succeeded]),
        "response_from_schedule_ms": _distribution([sample.lateness_ms + sample.latency_ms for sample in succeeded]),
        "mutation_ack_ms": {
            kind: _distribution([timing.ack_ms for timing in mutations if timing.kind == kind])
            for kind in ("upsert", "delete")
        },
        "mutation_visibility_ms": {
            kind: _distribution([timing.visibility_ms for timing in mutations if timing.kind == kind])
            for kind in ("upsert", "delete")
        },
    }


def _wait_for_visibility(check: Callable[[], bool], began: float, deadline: float) -> float:
    while time.monotonic() - began <= deadline:
        if check():
            elapsed = time.monotonic() - began
            if elapsed <= deadline:
                return 1000 * elapsed
            break
        time.sleep(VISIBILITY_INTERVAL)
    raise VisibilityTimeout


class MutationLoad:
    """Exercise insert, replace and delete on a bounded set of dedicated IDs."""

    def __init__(self, backend, corpus: VectorCorpus):
        self.backend = backend
        self.corpus = corpus
        self.sequence = 1
        self.live: dict[int, int] = {}

    def __call__(self, _index: int) -> MutationTiming:
        # Cycle through insert, replacement and delete while rotating dedicated IDs.
        cycle, step = divmod(self.sequence - 1, 3)
        slot = cycle % 128
        entry_id = len(self.corpus) + slot
        self.sequence += 1
        sequence = self.sequence
        began = time.monotonic()
        if step == 2 and entry_id in self.live:
            prior = self.live[entry_id]
            self.backend.remove(entry_id, sequence)
            ack = 1000 * (time.monotonic() - began)
            elapsed = _wait_for_visibility(lambda: not self.backend.visible(entry_id, prior), began, DELETE_DEADLINE)
            del self.live[entry_id]
            return MutationTiming("delete", ack, elapsed)
        self.backend.upsert(entry_id, sequence, self.corpus[(slot + step) % len(self.corpus)])
        ack = 1000 * (time.monotonic() - began)
        elapsed = _wait_for_visibility(lambda: self.backend.visible(entry_id, sequence), began, UPSERT_DEADLINE)
        self.live[entry_id] = sequence
        return MutationTiming("upsert", ack, elapsed)


def validate_stale_replay(backend, first_id: int, vector: VectorCorpus) -> dict[str, bool]:
    """Probe stale upserts and deletes against independently established live rows."""
    backend.upsert(first_id, 10, vector)
    backend.remove(first_id, 11)
    backend.upsert(first_id, 10, vector)
    stale_upsert_after_delete = not backend.visible(first_id, 10)

    replacement = first_id + 1
    backend.upsert(replacement, 10, vector)
    backend.upsert(replacement, 12, vector)
    backend.upsert(replacement, 11, vector)
    stale_upsert_after_replacement = backend.visible(replacement, 12)

    stale_delete = first_id + 2
    backend.upsert(stale_delete, 12, vector)
    backend.remove(stale_delete, 11)
    stale_delete_after_replacement = backend.visible(stale_delete, 12)

    duplicate = first_id + 3
    backend.upsert(duplicate, 10, vector)
    backend.remove(duplicate, 11)
    backend.remove(duplicate, 11)
    duplicate_delete = not backend.visible(duplicate, 10)
    return {
        "stale_upsert_after_delete": stale_upsert_after_delete,
        "stale_upsert_after_replacement": stale_upsert_after_replacement,
        "stale_delete_after_replacement": stale_delete_after_replacement,
        "duplicate_delete": duplicate_delete,
    }


def _measure_recall(
    backend, corpus: VectorCorpus, workload: Workload, labels: dict[int, set[int]] | None = None
) -> dict[str, object]:
    recalls = []
    errors: Counter[str] = Counter()
    timings = []
    labels = labels or {}
    hits = 0
    for query, own_id in enumerate(workload.query_ids):
        began = time.monotonic()
        try:
            found = backend.search(corpus[own_id], workload.k, exclude=own_id)
            timings.append(1000 * (time.monotonic() - began))
            recalls.append(recall_for_results(corpus, workload, query, found))
            if own_id in labels and not labels[own_id].isdisjoint(found):
                hits += 1
        except Exception as exc:
            recalls.append(0.0)
            errors[type(exc).__name__] += 1
        if (query + 1) % 250 == 0:
            _progress(f"Retrieved {query + 1}/{len(workload.query_ids)} recall queries")
    return {
        "queries": len(recalls),
        "mean": math.fsum(recalls) / len(recalls),
        "minimum": min(recalls),
        "perfect_queries": sum(recall == 1.0 for recall in recalls),
        "errors": dict(errors),
        "latency_ms": _distribution(timings),
        "labeled_overlap": {"queries": len(labels), "hits": hits, "hit_rate": hits / len(labels) if labels else None},
    }


def _measure_load(backend, corpus: VectorCorpus, workload: Workload, args: argparse.Namespace):
    def search(index: int) -> None:
        own_id = workload.query_ids[index % len(workload.query_ids)]
        _validate_results(backend.search(corpus[own_id], workload.k, exclude=own_id), workload.k, own_id)

    mutate = MutationLoad(backend, corpus)
    phases = {}
    for phase, duration, mutation_rate in (("steady", args.duration, 5), ("burst", args.burst_duration, 25)):
        if duration <= 0:
            continue
        _progress(f"{phase.capitalize()} load for {duration:g} seconds")
        started = time.monotonic() + 0.10
        with ThreadPoolExecutor(max_workers=2) as harness:
            searches = harness.submit(run_stream, search, start=started, rate=20, duration=duration, workers=4)
            mutations = harness.submit(
                run_stream, mutate, start=started, rate=mutation_rate, duration=duration, workers=1
            )
            while wait((searches, mutations), timeout=60).not_done:
                _progress(f"{phase.capitalize()} load elapsed {time.monotonic() - started:.0f}/{duration:g} seconds")
            phases[phase] = {"search": searches.result(), "mutations": mutations.result()}
    return phases


def _run_backend(
    name: str, corpus: VectorCorpus, workload: Workload, labels: dict[int, set[int]], args: argparse.Namespace
) -> dict[str, object]:
    try:
        from scripts.registry_overlap_benchmark_backends import PgVectorBackend, QdrantBackend
    except ModuleNotFoundError as exc:
        if exc.name != "scripts":
            raise
        from registry_overlap_benchmark_backends import PgVectorBackend, QdrantBackend

    if name == "pgvector":
        backend = PgVectorBackend(args.postgres_dsn, args.dimension)
    else:
        backend = QdrantBackend(args.qdrant_url, args.dimension)
    issued = time.monotonic()
    try:
        _progress(f"{name}: preparing {len(corpus):,} indexed vectors")
        backend_metadata = backend.prepare(corpus)
        setup_seconds = time.monotonic() - issued
        _progress(f"{name}: measuring exact-reference recall")
        recall = _measure_recall(backend, corpus, workload, labels)
        load = _measure_load(backend, corpus, workload, args)
        _progress(f"{name}: verifying stale mutation replay")
        stale = validate_stale_replay(backend, len(corpus) + 128, corpus[0])
    finally:
        backend.close()
    result = {
        "setup_seconds": setup_seconds,
        "backend": backend_metadata,
        "recall": recall,
        "load": load,
        "stale_replay": stale,
    }
    result["qualification"] = qualify_backend(result, args)
    result["qualification"]["labeled_overlap_gate"] = qualify_labeled_overlap(
        recall["labeled_overlap"], k=workload.k, synthetic=not args.vectors
    )
    return result


def qualify_labeled_overlap(measurement: dict, *, k: int, synthetic: bool) -> dict[str, object]:
    """Assess published Hit@10 evidence for the supplied vectors."""
    checks = {
        "supplied_corpus": not synthetic,
        "hit_at_10_measured": k == 10 and measurement["hit_rate"] is not None,
        "minimum_labeled_queries": measurement["queries"] >= MIN_LABELED_QUERIES,
        "hit_rate": measurement["hit_rate"] is not None and measurement["hit_rate"] >= 0.9,
    }
    return {"passes_labeled_overlap_gate": all(checks.values()), "checks": checks}


def qualify_backend(result: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    """Evaluate the technical gate against measured end-to-end schedule timings."""
    metadata = result["backend"]
    using_ann = metadata.get("planner_uses_ann") is True and metadata.get("ann_indexed_count", 0) >= args.size
    checks = {
        "ann_index": using_ann,
        "full_operating_envelope": (
            args.size >= 100_000
            and args.dimension == 2048
            and args.queries >= 1000
            and args.k == 10
            and args.duration >= 1800
            and args.burst_duration >= 60
        ),
        "recall_at_10": args.k == 10 and result["recall"]["mean"] >= 0.95 and not result["recall"]["errors"],
        "stale_replay": bool(result["stale_replay"]) and all(result["stale_replay"].values()),
    }
    search_failure_fraction = {}
    for phase in ("steady", "burst"):
        phase_data = result["load"].get(phase)
        if phase_data is None:
            checks[f"{phase}_load_measured"] = False
            continue
        searches = phase_data["search"]
        mutations = phase_data["mutations"]
        percentile = searches["response_from_schedule_ms"]
        visibility = mutations["mutation_visibility_ms"]
        search_failures = searches["dropped"] + sum(searches["errors"].values())
        failed_fraction = search_failures / searches["scheduled"]
        search_failure_fraction[phase] = failed_fraction
        checks.update(
            {
                f"{phase}_search_p95": percentile["p95"] is not None and percentile["p95"] <= 100,
                f"{phase}_search_p99": percentile["p99"] is not None and percentile["p99"] <= 250,
                f"{phase}_search_timeout_error_rate": failed_fraction < 0.005,
                f"{phase}_upsert_visibility": (
                    visibility["upsert"]["p95"] is not None and visibility["upsert"]["p95"] <= 60_000
                ),
                f"{phase}_delete_visibility": (
                    visibility["delete"]["max"] is not None and visibility["delete"]["max"] <= 5_000
                ),
                f"{phase}_mutation_load_completed": not mutations["dropped"] and not mutations["errors"],
            }
        )
    return {
        "passes_technical_gate": all(checks.values()),
        "checks": checks,
        "search_failure_fraction": search_failure_fraction,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pgvector", "qdrant", "both"), default="both")
    parser.add_argument("--postgres-dsn", default=os.getenv("SKILLEVALUATOR_BENCHMARK_POSTGRES_DSN"))
    parser.add_argument("--qdrant-url", default=os.getenv("SKILLEVALUATOR_BENCHMARK_QDRANT_URL"))
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--duration", type=float, default=1800, help="Steady mixed-load seconds (0 skips).")
    parser.add_argument("--burst-duration", type=float, default=60, help="Mutation-burst seconds (0 skips).")
    parser.add_argument("--vectors", type=Path, help="Precomputed little-endian float32 .npy corpus or corpus prefix.")
    parser.add_argument("--write-vectors", type=Path, help="Save the normalized corpus as .npy for reuse.")
    parser.add_argument("--query-ids", type=Path, help="JSON list of distinct zero-based corpus row IDs.")
    parser.add_argument("--labels", type=Path, help="JSON map from selected query IDs to related corpus row IDs.")
    parser.add_argument("--workload", type=Path, help="Reuse query IDs and exact ranking from a prepared run.")
    parser.add_argument("--write-workload", type=Path, help="Save exact rankings for future runs on the same vectors.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare artifacts without contacting any backend.")
    parser.add_argument("--output", type=Path, help="Write the JSON report here (default stdout).")
    args = parser.parse_args(argv)
    if not 2 <= args.size <= 500_000 or not 1 <= args.dimension <= 4096:
        parser.error("size must be within [2, 500000], dimension within [1, 4096]")
    if args.size * args.dimension * 4 > 4 * 1024**3:
        parser.error("normalized corpus exceeds 4 GiB")
    if not 1 <= args.queries <= min(args.size, 10_000) or not 1 <= args.k <= min(100, args.size - 1):
        parser.error("queries must fit the corpus and be <= 10000; k must be below size and <= 100")
    if not 0 <= args.seed < 2**32 or not 0 <= args.duration <= 3600 or not 0 <= args.burst_duration <= 600:
        parser.error("seed must fit uint32; steady and burst periods must be within [0, 3600] and [0, 600]")
    if args.workload and args.query_ids:
        parser.error("--workload already specifies query IDs")
    if args.write_vectors and args.write_vectors.suffix != ".npy":
        parser.error("--write-vectors must name a .npy file")
    if not args.prepare_only:
        if args.backend in ("pgvector", "both") and not args.postgres_dsn:
            parser.error("supply --postgres-dsn or SKILLEVALUATOR_BENCHMARK_POSTGRES_DSN")
        if args.backend in ("qdrant", "both") and not args.qdrant_url:
            parser.error("supply --qdrant-url or SKILLEVALUATOR_BENCHMARK_QDRANT_URL")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import numpy as np
    except ImportError:
        _progress("Benchmark requires NumPy. Install it in an isolated benchmark environment.")
        return 2

    try:
        _progress(f"Preparing {args.size:,} vectors of dimension {args.dimension}")
        corpus = (
            load_corpus(args.vectors, size=args.size, dimension=args.dimension)
            if args.vectors
            else synthetic_corpus(args.size, args.dimension, args.seed)
        )
        digest = corpus_sha256(corpus)
        if args.write_vectors:
            np.save(args.write_vectors, corpus, allow_pickle=False)
        if args.workload:
            _progress("Reading recorded query IDs and exact ranking cutoffs")
            workload = _read_workload(args.workload, args, digest)
        else:
            _progress(f"Computing exhaustive cosine rankings for {args.queries:,} queries")
            if args.query_ids:
                query_ids = _query_ids(_read_json(args.query_ids), args.size, args.queries)
            else:
                rng = np.random.default_rng(args.seed)
                query_ids = rng.choice(args.size, args.queries, replace=False).tolist()
            workload = exact_workload(corpus, query_ids, k=args.k, digest=digest, progress=True)
        labels = _read_labels(args.labels, workload) if args.labels else {}
        workload_document = {"version": FORMAT_VERSION, "tie_tolerance": TIE_TOLERANCE, **asdict(workload)}
        if args.write_workload:
            _write_json(args.write_workload, workload_document)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        _progress(f"Invalid benchmark input or artifact: {exc}")
        return 2

    document = {
        "version": FORMAT_VERSION,
        "corpus": {
            "origin": "supplied .npy" if args.vectors else "synthetic clustered",
            "size": args.size,
            "dimension": args.dimension,
            "sha256": digest,
            "seed": args.seed,
        },
        "workload": {
            "k": args.k,
            "queries": args.queries,
            "labeled_queries": len(labels),
            "labels_sha256": (
                hashlib.sha256(json.dumps({key: sorted(labels[key]) for key in sorted(labels)}).encode()).hexdigest()
                if labels
                else None
            ),
            "reuse": bool(args.workload),
            "sha256": hashlib.sha256(json.dumps(workload_document, sort_keys=True).encode("utf-8")).hexdigest(),
            "steady_seconds": args.duration,
            "burst_seconds": args.burst_duration,
            "search_rate_per_second": 20,
            "steady_mutation_rate_per_second": 5,
            "burst_mutation_rate_per_second": 25,
            "tie_tolerance": TIE_TOLERANCE,
            "provider_time_included": False,
            "mutation_visibility_probe": "separate keyed live-revision read",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__},
        "qualification": {
            "embedding_cost": {"status": "unmeasured", "requires": "measured provider workload and pricing"},
            "embedding_profile": {
                "status": "unverified",
                "requires": "representative real-skill examples embedded with the intended provider profile",
            },
        },
        "results": {},
    }
    if not args.prepare_only:
        selected = ("pgvector", "qdrant") if args.backend == "both" else (args.backend,)
        for name in selected:
            try:
                document["results"][name] = _run_backend(name, corpus, workload, labels, args)
            except Exception as exc:
                # Driver errors may embed DSNs or other connection credentials.
                _progress(f"{name}: benchmark failed ({type(exc).__name__})")
                document["results"][name] = {"error": type(exc).__name__}
            if args.output:
                try:
                    _write_json(args.output, document)
                except OSError as exc:
                    _progress(f"Could not write report: {exc}")
                    return 2
    try:
        _write_json(args.output, document)
    except OSError as exc:
        _progress(f"Could not write report: {exc}")
        return 2
    return int(any("error" in result for result in document["results"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
