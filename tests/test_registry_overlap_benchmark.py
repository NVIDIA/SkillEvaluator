# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the standalone registry-overlap benchmark."""

import json
import subprocess
import sys
import time

import pytest


def test_help_runs_without_optional_benchmark_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", "scripts/benchmark_registry_overlap.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--vectors" in result.stdout
    assert "--workload" in result.stdout
    assert "--labels" in result.stdout


def test_oracle_accepts_boundary_ties_without_crediting_a_missing_strict_hit():
    np = pytest.importorskip("numpy")
    from scripts.benchmark_registry_overlap import exact_workload, recall_for_results

    corpus = np.array([[1, 0], [1, 0], [0, 1], [0, -1], [-1, 0]], dtype=np.float32)
    workload = exact_workload(corpus, [0], k=2)

    assert recall_for_results(corpus, workload, 0, [1, 3]) == 1.0
    assert recall_for_results(corpus, workload, 0, [2, 3]) == 0.5
    with pytest.raises(ValueError, match="duplicate"):
        recall_for_results(corpus, workload, 0, [1, 1])
    with pytest.raises(ValueError, match="excluded"):
        recall_for_results(corpus, workload, 0, [0, 1])


def test_external_corpus_checks_dtype_and_rejects_zero_norms(tmp_path):
    np = pytest.importorskip("numpy")
    from scripts.benchmark_registry_overlap import load_corpus

    corpus_file = tmp_path / "vectors.npy"
    np.save(corpus_file, np.array([[3, 4], [4, 3]], dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        load_corpus(corpus_file, size=2, dimension=2)

    np.save(corpus_file, np.array([[3, 4], [0, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="nonzero"):
        load_corpus(corpus_file, size=2, dimension=2)

    np.save(corpus_file, np.array([[3, 4], [0, 2], [3, 1]], dtype=np.float32))
    vectors = load_corpus(corpus_file, size=2, dimension=2)
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0, 1]])


def test_prepared_workload_can_be_reused_and_refuses_corpus_drift(tmp_path, capsys):
    np = pytest.importorskip("numpy")
    from scripts.benchmark_registry_overlap import main

    vectors = tmp_path / "vectors.npy"
    workload = tmp_path / "queries.json"
    summary = tmp_path / "prepare.json"
    common_args = ["--size", "16", "--dimension", "8", "--queries", "4", "--k", "3", "--prepare-only"]

    assert (
        main(
            [*common_args, "--write-vectors", str(vectors), "--write-workload", str(workload), "--output", str(summary)]
        )
        == 0
    )
    prepared = json.loads(summary.read_text(encoding="utf-8"))
    assert prepared["corpus"]["sha256"] == json.loads(workload.read_text(encoding="utf-8"))["corpus_sha256"]
    assert main([*common_args, "--vectors", str(vectors), "--workload", str(workload)]) == 0

    changed = np.load(vectors)
    changed[0] = changed[1]
    np.save(vectors, changed)
    assert main([*common_args, "--vectors", str(vectors), "--workload", str(workload)]) != 0
    assert "fingerprint" in capsys.readouterr().err


def test_rate_scheduler_records_requests_that_expire_in_its_bounded_queue():
    from scripts.benchmark_registry_overlap import run_stream

    def slow_request(_index):
        time.sleep(0.08)

    report = run_stream(
        slow_request, start=time.monotonic() + 0.01, rate=100, duration=0.10, workers=1, lateness_limit=0.04
    )

    assert report["scheduled"] == 10
    assert 0 < report["completed"] < 10
    assert report["dropped"] + report["completed"] == 10
    assert report["dispatch_lateness_ms"]["p95"] >= 0


def test_stale_replay_cannot_restore_a_deleted_id_or_replace_a_newer_version():
    np = pytest.importorskip("numpy")
    from scripts.benchmark_registry_overlap import validate_stale_replay

    class Backend:
        def __init__(self, ignores_order):
            self.entries = {}
            self.ignores_order = ignores_order

        def _set(self, entry_id, sequence, live):
            if self.ignores_order or sequence > self.entries.get(entry_id, (0, False))[0]:
                self.entries[entry_id] = sequence, live

        def upsert(self, entry_id, sequence, _vector):
            self._set(entry_id, sequence, True)

        def remove(self, entry_id, sequence):
            self._set(entry_id, sequence, False)

        def visible(self, entry_id, sequence):
            return self.entries.get(entry_id) == (sequence, True)

    vector = np.array([0, 1], dtype=np.float32)
    assert all(validate_stale_replay(Backend(False), 100, vector).values())
    assert validate_stale_replay(Backend(True), 100, vector) == {
        "stale_upsert_after_delete": False,
        "stale_upsert_after_replacement": False,
        "stale_delete_after_replacement": False,
        "duplicate_delete": True,
    }


def test_qualification_requires_ann_evidence_and_a_complete_operating_envelope():
    from types import SimpleNamespace

    from scripts.benchmark_registry_overlap import qualify_backend

    search = {
        "scheduled": 36_000,
        "completed": 36_000,
        "dropped": 0,
        "errors": {},
        "response_from_schedule_ms": {"p95": 90.0, "p99": 200.0},
    }
    mutations = {
        "dropped": 0,
        "errors": {},
        "mutation_visibility_ms": {"upsert": {"p95": 800.0}, "delete": {"max": 10.0}},
    }
    result = {
        "backend": {"planner_uses_ann": True, "ann_indexed_count": 100_000},
        "recall": {"mean": 0.97, "errors": {}},
        "load": {
            "steady": {"search": search, "mutations": mutations},
            "burst": {"search": search, "mutations": mutations},
        },
        "stale_replay": {"stale_upsert_after_delete": True},
    }
    args = SimpleNamespace(size=100_000, dimension=2048, queries=1000, k=10, duration=1800, burst_duration=60)

    report = qualify_backend(result, args)
    assert report["passes_technical_gate"] is True

    result["backend"]["planner_uses_ann"] = False
    report = qualify_backend(result, args)
    assert report["checks"]["ann_index"] is False
    assert report["passes_technical_gate"] is False

    result["backend"]["planner_uses_ann"] = True
    args.duration = 2
    report = qualify_backend(result, args)
    assert report["checks"]["full_operating_envelope"] is False
    assert report["passes_technical_gate"] is False

    mutations["mutation_visibility_ms"]["delete"]["max"] = 5000.01
    report = qualify_backend(result, args)
    assert report["checks"]["burst_delete_visibility"] is False


def test_labeled_overlap_hit_is_separate_from_exact_recall(tmp_path):
    np = pytest.importorskip("numpy")
    from scripts.benchmark_registry_overlap import _measure_recall, _read_labels, exact_workload

    class ExactBackend:
        def search(self, vector, k, exclude):
            scores = corpus @ vector
            scores[exclude] = -np.inf
            return np.argsort(-scores)[:k].tolist()

    corpus = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    workload = exact_workload(corpus, [0, 2], k=1)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text('{"0":[1],"2":[0]}', encoding="utf-8")

    labels = _read_labels(labels_path, workload)
    report = _measure_recall(ExactBackend(), corpus, workload, labels)

    assert report["mean"] == 1.0
    assert report["labeled_overlap"] == {"queries": 2, "hits": 1, "hit_rate": 0.5}


@pytest.mark.parametrize("label_json", ('{"1":[0]}', '{"0":[0]}', '{"0":[2,2]}', '{"0":[4]}', '{"0":[2],"0":[3]}'))
def test_labels_refuse_unselected_queries_and_invalid_related_ids(tmp_path, label_json):
    from scripts.benchmark_registry_overlap import Workload, _read_labels

    workload = Workload("corpus-hash", 4, 2, 1, [0], [1.0], [0])
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(label_json, encoding="utf-8")

    with pytest.raises(ValueError):
        _read_labels(labels_path, workload)


@pytest.mark.parametrize(
    ("queries", "hit_rate", "k", "synthetic", "passes"),
    [
        (100, 0.9, 10, False, True),
        (99, 1.0, 10, False, False),
        (100, 0.89, 10, False, False),
        (100, 0.9, 9, False, False),
        (100, 1.0, 10, True, False),
        (0, None, 10, False, False),
    ],
)
def test_labeled_gate_requires_evidence_for_supplied_vectors(queries, hit_rate, k, synthetic, passes):
    from scripts.benchmark_registry_overlap import qualify_labeled_overlap

    report = qualify_labeled_overlap({"queries": queries, "hit_rate": hit_rate}, k=k, synthetic=synthetic)

    assert report["passes_labeled_overlap_gate"] is passes


@pytest.mark.parametrize(("flag", "value"), [("--k", "51"), ("--dimension", "4001")])
def test_pgvector_runner_refuses_unsupported_parameters(flag, value):
    from scripts.benchmark_registry_overlap import _parse_args

    with pytest.raises(SystemExit) as exc:
        _parse_args(["--prepare-only", flag, value])

    assert exc.value.code == 2


@pytest.mark.parametrize("alias", ["same-path", "symlink", "hardlink"])
def test_runner_cannot_overwrite_vector_input(tmp_path, alias):
    np = pytest.importorskip("numpy")
    from scripts.benchmark_registry_overlap import main

    vectors = tmp_path / "vectors.npy"
    np.save(vectors, np.eye(2, dtype=np.float32))
    original = vectors.read_bytes()
    output = vectors if alias == "same-path" else tmp_path / "report.json"
    if alias == "symlink":
        output.symlink_to(vectors)
    if alias == "hardlink":
        output.hardlink_to(vectors)

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--prepare-only",
                "--vectors",
                str(vectors),
                "--output",
                str(output),
                "--size",
                "2",
                "--dimension",
                "2",
                "--queries",
                "1",
                "--k",
                "1",
            ]
        )

    assert exc.value.code == 2
    assert vectors.read_bytes() == original


@pytest.mark.parametrize("found", [[], [2], [2, 7], [True, 2], [-1, 2]])
def test_load_requires_full_in_range_candidate_lists(found):
    from scripts.benchmark_registry_overlap import _validate_results

    with pytest.raises(ValueError):
        _validate_results(found, k=2, own_id=0, id_limit=7, require_full=True)

    _validate_results([1, 6], k=2, own_id=0, id_limit=7, require_full=True)


def test_mutation_probe_requires_ann_suppression_after_keyed_removal(monkeypatch):
    np = pytest.importorskip("numpy")
    from scripts import benchmark_registry_overlap as benchmark

    class Backend:
        def __init__(self):
            self.indexed = None
            self.live = None

        def upsert(self, entry_id, sequence, vector):
            self.indexed = entry_id, vector.copy()
            self.live = entry_id, sequence

        def remove(self, _entry_id, _sequence):
            self.live = None

        def visible(self, entry_id, sequence):
            return self.live == (entry_id, sequence)

        def search(self, vector, _k):
            return [self.indexed[0]] if np.array_equal(vector, self.indexed[1]) else [0]

    backend = Backend()
    mutate = benchmark.MutationLoad(backend, np.eye(4, dtype=np.float32), seed=7)

    assert mutate(0).kind == "upsert"
    inserted = backend.indexed[1].copy()
    assert mutate(1).kind == "upsert"
    assert not np.array_equal(inserted, backend.indexed[1])
    monkeypatch.setattr(benchmark, "DELETE_DEADLINE", 0.02)

    with pytest.raises(benchmark.VisibilityTimeout):
        mutate(2)
