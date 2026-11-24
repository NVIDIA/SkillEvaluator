# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from scripts import registry_overlap_benchmark_corpus as corpus


def test_projects_only_bounded_yaml_name_and_description() -> None:
    identity = {"id": "id", "skill_slug": "publisher/skill", "skill_version": "1.0.0"}
    good = "---\nname: A skill\ndescription: >-\n  First line.\n  Second line.\n---\nBody"
    assert corpus.project_row({**identity, "skill_md_content": good}).embedding_text == (
        "A skill: First line. Second line."
    )
    unsafe = "---\nname: A skill\ndescription: !!python/object:bad {}\n---\n"
    assert corpus.project_row({**identity, "skill_md_content": unsafe}) is None
    assert corpus.project_row({**identity, "skill_slug": "", "skill_md_content": good}) is None
    assert corpus.project_row({**identity, "skill_md_content": good.replace("name:", "missing:")}) is None


@pytest.fixture
def source_shards(tmp_path: Path) -> dict[str, str]:
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    good = "---\nname: A skill\ndescription: A description.\n---\nBody"
    table = arrow.table(
        {"id": ["id"], "skill_slug": ["publisher/skill"], "skill_version": ["1"], "skill_md_content": [good]}
    )
    for shard in corpus.PARQUET_SHARDS:
        parquet.write_table(table, tmp_path / shard)
    return {shard: hashlib.sha256((tmp_path / shard).read_bytes()).hexdigest() for shard in corpus.PARQUET_SHARDS}


def test_partial_projection_requires_explicit_sampling(tmp_path: Path, source_shards: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="checksum"):
        corpus._load_rows(tmp_path, allow_source_sample=False)
    rows, counts, files = corpus._load_rows(tmp_path, allow_source_sample=True)
    assert len(rows) == 1
    assert counts == {"source": 6, "retained": 1, "malformed": 0, "duplicate_slug": 5}
    assert files == source_shards


def test_full_source_validates_names_hashes_and_row_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_shards: dict[str, str]
) -> None:
    monkeypatch.setattr(corpus, "PARQUET_ROWS", len(source_shards))
    monkeypatch.setattr(corpus, "PARQUET_SHA256", source_shards)
    _, counts, files = corpus._load_rows(tmp_path, allow_source_sample=False)
    assert counts["source"] == len(source_shards)
    assert files == source_shards

    first_shard = corpus.PARQUET_SHARDS[0]
    original = tmp_path / first_shard
    unexpected = tmp_path / "unexpected.parquet"
    original.rename(unexpected)
    with pytest.raises(ValueError, match="shard names"):
        corpus._load_rows(tmp_path, allow_source_sample=False)

    unexpected.rename(original)
    unexpected.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="shard names"):
        corpus._load_rows(tmp_path, allow_source_sample=False)

    unexpected.unlink()
    monkeypatch.setattr(corpus, "PARQUET_ROWS", len(source_shards) + 1)
    with pytest.raises(ValueError, match="pinned source rows"):
        corpus._load_rows(tmp_path, allow_source_sample=False)


def test_full_row_count_in_sample_mode_does_not_claim_verified_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_shards: dict[str, str]
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for name in source_shards:
        (tmp_path / name).rename(source_dir / name)
    monkeypatch.setattr(corpus, "PARQUET_ROWS", len(source_shards))
    first_shard = corpus.PARQUET_SHARDS[0]
    monkeypatch.setattr(corpus, "PARQUET_SHA256", {**source_shards, first_shard: "0" * 64})
    for name, constant in [(corpus.MODEL_FILE, "MODEL_SHA256"), ("tokenizer.json", "TOKENIZER_SHA256")]:
        (tmp_path / name).write_bytes(b"model fixture")
        monkeypatch.setattr(corpus, constant, hashlib.sha256(b"model fixture").hexdigest())
    monkeypatch.setattr(corpus, "_embed", lambda *_args, **_kwargs: np.ones((1, corpus.MODEL_DIMENSION)))
    path = tmp_path / "matrix.npy"
    corpus.main(
        [
            "--parquet-dir",
            str(source_dir),
            "--model-dir",
            str(tmp_path),
            "--output",
            str(path),
            "--allow-source-sample",
            "--size",
            str(len(source_shards)),
            "--dimension",
            str(corpus.MODEL_DIMENSION),
        ]
    )
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["dataset_revision"] == corpus.DATASET_REVISION
    assert metadata["parquet_conversion_revision"] == corpus.PARQUET_REVISION
    assert metadata["source_selection"] == "local sample or projection"
    assert metadata["source_shards"] == [
        {"name": name, "sha256": digest, "verified_upstream": name != first_shard}
        for name, digest in source_shards.items()
    ]


def test_projection_preserves_geometry_and_padding_preserves_prefix(tmp_path: Path) -> None:
    real = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    corpus.write_matrix(real, first, size=15, dimension=8, seed=41)
    corpus.write_matrix(real, second, size=20, dimension=8, seed=41)
    left = np.load(first, allow_pickle=False)
    right = np.load(second, allow_pickle=False)
    assert left.dtype == np.dtype("<f4")
    assert left.shape == (15, 8)
    np.testing.assert_allclose(np.linalg.norm(right, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(left, right[: len(left)])
    n = real / np.linalg.norm(real, axis=1, keepdims=True)
    np.testing.assert_allclose(left[:3] @ left[:3].T, n @ n.T, atol=1e-6)
    assert np.unique(right, axis=0).shape[0] == len(right)


@pytest.mark.parametrize("invalid", [[[0.0, 0.0]], [[1.0, float("nan")]], [[float("inf"), 1.0]]])
def test_rejects_malformed_vectors_without_publishing_matrix(tmp_path: Path, invalid: list[list[float]]) -> None:
    path = tmp_path / "bad.npy"
    with pytest.raises(ValueError, match="finite nonzero"):
        corpus.write_matrix(np.array(invalid), path, size=3, dimension=8, seed=1)
    assert not path.exists()


@pytest.mark.parametrize("suffix", [".json", ".out"])
def test_output_extension_cannot_collide_with_provenance_sidecar(tmp_path, capsys, suffix):
    with pytest.raises(SystemExit) as exc:
        corpus.main(
            ["--parquet-dir", str(tmp_path), "--model-dir", str(tmp_path), "--output", str(tmp_path / f"data{suffix}")]
        )
    assert exc.value.code == 2
    assert "--output must use the .npy extension" in capsys.readouterr().err
