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
        def __init__(self, *, ignores_order=False, index_ignores_order=False):
            self.entries = {}
            self.vectors = {0: np.eye(16, dtype=np.float32)[0]}
            self.ignores_order = ignores_order
            self.index_ignores_order = index_ignores_order

        def _set(self, entry_id, sequence, vector):
            advances = self.ignores_order or sequence > self.entries.get(entry_id, (0, False))[0]
            if advances:
                self.entries[entry_id] = sequence, vector is not None
            if advances or self.index_ignores_order:
                if vector is None:
                    self.vectors.pop(entry_id, None)
                else:
                    self.vectors[entry_id] = vector

        def upsert(self, entry_id, sequence, vector):
            self._set(entry_id, sequence, vector)

        def remove(self, entry_id, sequence):
            self._set(entry_id, sequence, None)

        def visible(self, entry_id, sequence):
            return self.entries.get(entry_id) == (sequence, True)

        def search(self, vector, k):
            return sorted(self.vectors, key=lambda entry_id: (-self.vectors[entry_id] @ vector, entry_id))[:k]

    assert all(validate_stale_replay(Backend(), 100, dimension=16, seed=41).values())
    assert validate_stale_replay(Backend(ignores_order=True), 100, dimension=16, seed=41) == {
        "stale_upsert_after_delete": False,
        "stale_upsert_after_replacement": False,
        "stale_delete_after_replacement": False,
        "duplicate_delete": True,
    }
    assert validate_stale_replay(Backend(index_ignores_order=True), 100, dimension=16, seed=41) == {
        "stale_upsert_after_delete": False,
        "stale_upsert_after_replacement": False,
        "stale_delete_after_replacement": False,
        "duplicate_delete": True,
    }

    class MissingResultsBackend(Backend):
        def search(self, vector, k):
            return super().search(vector, k) if len(self.vectors) > 1 else []

    report = validate_stale_replay(MissingResultsBackend(), 100, dimension=16, seed=41)
    assert report["stale_upsert_after_delete"] is False


def test_one_dimensional_corpus_cannot_qualify_stale_replay():
    from scripts.benchmark_registry_overlap import validate_stale_replay

    assert validate_stale_replay(object(), 100, dimension=1, seed=41) == {"distinguishable_probe_vectors": False}


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


@pytest.mark.parametrize("phase", ["--duration", "--burst-duration"])
def test_one_dimensional_corpus_requires_recall_only_or_preparation(phase, capsys):
    from scripts.benchmark_registry_overlap import _parse_args

    args = [
        "--backend",
        "pgvector",
        "--postgres-dsn",
        "test",
        "--dimension",
        "1",
        "--duration",
        "0",
        "--burst-duration",
        "0",
    ]
    assert _parse_args(args).dimension == 1
    assert _parse_args([*args, phase, "1", "--prepare-only"]).dimension == 1

    with pytest.raises(SystemExit) as exc:
        _parse_args([*args, phase, "1"])

    assert exc.value.code == 2
    assert "at least two dimensions" in capsys.readouterr().err


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


@pytest.mark.parametrize(
    ("first_flag", "first_name", "second_name"),
    [
        ("--write-workload", "workload.json", "WORKLOAD.json"),
        ("--write-workload", "caf\u00e9.json", "cafe\u0301.json"),
        ("--write-vectors", "caf\u00e9.npy", "CAFE\u0301.NPY"),
    ],
)
def test_fresh_artifact_names_follow_destination_filesystem(tmp_path, capsys, first_flag, first_name, second_name):
    pytest.importorskip("numpy")
    from scripts.benchmark_registry_overlap import main

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / first_name).touch()
    first, second = tmp_path / first_name, tmp_path / second_name
    args = ["--prepare-only", "--size", "4", "--dimension", "2", "--queries", "2", "--k", "1"]
    args.extend([first_flag, str(first), "--output", str(second)])

    if (calibration / second_name).exists():
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 2
        assert "separate files" in capsys.readouterr().err
        assert list(tmp_path.iterdir()) == [calibration]
    else:
        assert main(args) == 0
        assert first.is_file() and second.is_file() and not first.samefile(second)


@pytest.mark.parametrize("first_output", ["vectors", "workload"])
def test_output_alias_that_appears_during_preparation_preserves_first_artifact(
    tmp_path, monkeypatch, capsys, first_output
):
    numpy = pytest.importorskip("numpy")
    from scripts import benchmark_registry_overlap as benchmark

    report = tmp_path / "report.json"
    args = ["--size", "4", "--dimension", "2", "--queries", "2", "--k", "1", "--prepare-only", "--output", str(report)]
    if first_output == "vectors":
        artifact = tmp_path / "vectors.npy"
        write_vectors = numpy.save

        def save_then_alias(path, corpus, **kwargs):
            write_vectors(path, corpus, **kwargs)
            report.hardlink_to(path)

        monkeypatch.setattr(numpy, "save", save_then_alias)
        args.extend(["--write-vectors", str(artifact)])
    else:
        artifact = tmp_path / "workload.json"
        write_json = benchmark._write_json

        def write_then_alias(path, data):
            write_json(path, data)
            if path == artifact:
                report.hardlink_to(path)

        monkeypatch.setattr(benchmark, "_write_json", write_then_alias)
        args.extend(["--write-workload", str(artifact)])

    assert benchmark.main(args) == 2
    assert "separate files" in capsys.readouterr().err
    if first_output == "vectors":
        assert numpy.load(artifact).shape == (4, 2)
    else:
        assert json.loads(artifact.read_text(encoding="utf-8"))["query_ids"]


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


def test_dropped_mutations_do_not_shift_later_writes(monkeypatch):
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    from scripts import benchmark_registry_overlap as benchmark

    class Backend:
        def __init__(self):
            self.writes = []
            self.live = None

        def upsert(self, entry_id, sequence, vector):
            self.writes.append((entry_id, sequence, vector.tobytes()))
            self.live = entry_id, sequence

        def remove(self, entry_id, sequence):
            self.writes.append((entry_id, sequence, None))
            self.live = None

        def visible(self, entry_id, sequence):
            return self.live == (entry_id, sequence)

        def search(self, _vector, _k):
            return [self.live[0] if self.live else 0]

    corpus = np.eye(4, dtype=np.float32)

    def run(indices):
        backend = Backend()
        mutate = benchmark.MutationLoad(backend, corpus, seed=41)
        for index in indices:
            mutate(index)
        return backend.writes

    full = run([0, 1, 2, 3])
    assert run([0, 2, 3]) == [full[0], full[2], full[3]]

    # The steady phase schedules three mutations and drops index 1. The burst starts at index 3.
    def run_stream(operation, *, rate, **_):
        for index in {5: [0, 2], 25: [0]}.get(rate, []):
            operation(index)
        return {}

    monkeypatch.setattr(benchmark, "run_stream", run_stream)
    backend = Backend()
    workload = benchmark.exact_workload(corpus, [0], k=1)
    benchmark._measure_load(backend, corpus, workload, SimpleNamespace(seed=41, duration=0.6, burst_duration=0.04))
    assert backend.writes == [full[0], full[2], full[3]]

    # A scheduled delete still sends its tombstone when both preceding writes were dropped.
    backend = Backend()
    with pytest.raises(benchmark.VisibilityTimeout, match="before deletion"):
        benchmark.MutationLoad(backend, corpus, seed=41)(2)
    assert backend.writes == [full[2]]
