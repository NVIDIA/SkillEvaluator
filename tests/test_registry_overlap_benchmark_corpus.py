# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_partial_projection_requires_explicit_sampling(tmp_path: Path) -> None:
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    good = "---\nname: A skill\ndescription: A description.\n---\nBody"
    table = arrow.table(
        {"id": ["id"], "skill_slug": ["publisher/skill"], "skill_version": ["1"], "skill_md_content": [good]}
    )
    for shard in corpus.PARQUET_SHARDS:
        parquet.write_table(table, tmp_path / shard)
    with pytest.raises(ValueError, match="pinned source rows"):
        corpus._load_rows(tmp_path, allow_source_sample=False)
    rows, counts = corpus._load_rows(tmp_path, allow_source_sample=True)
    assert len(rows) == 1
    assert counts == {"source": 6, "retained": 1, "malformed": 0, "duplicate_slug": 5}


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
