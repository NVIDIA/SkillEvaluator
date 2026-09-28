# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection size and comparison work are independent, configurable limits."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import skillevaluator.embedding.registry as registry_module
from skillevaluator.cli import cli
from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.extractor import ContentEntry, discover_and_extract
from skillevaluator.embedding.registry import EmbeddingRegistry
from skillevaluator.tier2.commands import run_similarity_check
from skillevaluator.validators.similarity import SimilarityValidator


def _collection(root: Path, count: int) -> None:
    for index in range(count):
        directory = root / f"skill-{index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(f"---\nname: skill-{index}\ndescription: Description {index}\n---\n")


def _client(dimension: int = 3, duplicate_last: bool = False) -> MagicMock:
    client = MagicMock(spec=EmbeddingClient)
    client.model = "test-model"
    client._resolved_config.return_value = SimpleNamespace(provider="openai", base_url="https://api.openai.com/v1")

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            index = int(text.split(":", 1)[0].removeprefix("skill-"))
            vector = [0.0] * dimension
            vector[0 if duplicate_last and index == 342 else index] = 1.0
            vectors.append(vector)
        return vectors

    client.embed.side_effect = embed
    client.embed_single.return_value = [1.0] + [0.0] * (dimension - 1)
    return client


@pytest.mark.parametrize("value", [0, -1, 5_001, True, False, 1.5, "343", None])
def test_invalid_entry_limits_fail_at_python_boundaries(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="--max-entries"):
        discover_and_extract(tmp_path, "skill", max_entries=value)
    with pytest.raises(ValueError, match="--max-entries"):
        EmbeddingRegistry(_client(), max_entries=value)
    with pytest.raises(ValueError, match="--max-entries"):
        SimilarityValidator(max_entries=value)


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "128000000", None])
def test_invalid_scalar_limits_fail_at_python_boundaries(value: object) -> None:
    with pytest.raises(ValueError, match="--max-scalar-comparisons"):
        EmbeddingRegistry(_client(), max_scalar_comparisons=value)
    with pytest.raises(ValueError, match="--max-scalar-comparisons"):
        SimilarityValidator(max_scalar_comparisons=value)


def test_default_entry_boundary_and_explicit_increase(tmp_path: Path) -> None:
    _collection(tmp_path, 1_024)
    assert len(discover_and_extract(tmp_path, "skill")) == 1_024
    _collection(tmp_path, 1_025)
    with pytest.raises(ValueError, match=r"entry limit exceeded \(1024\).*--max-entries"):
        discover_and_extract(tmp_path, "skill")
    assert len(discover_and_extract(tmp_path, "skill", max_entries=1_025)) == 1_025
    assert EmbeddingRegistry(_client(), max_entries=5_000).size == 0


def test_maximum_entry_budget_extracts_all_5000_skills(tmp_path: Path) -> None:
    _collection(tmp_path, 5_000)

    entries = discover_and_extract(tmp_path, "skill", max_entries=5_000)

    assert len(entries) == 5_000
    assert {entry.name for entry in entries} == {f"skill-{index}" for index in range(5_000)}


@pytest.mark.parametrize("content_type", ["skill", "rules", "workflows"])
def test_entry_budget_counts_selected_invalid_manifests(tmp_path: Path, content_type: str) -> None:
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        filename = {"skill": "SKILL.md", "rules": "rule.mdc", "workflows": "workflow-rules.mdc"}[content_type]
        (directory / filename).write_text("No valid frontmatter")
    with pytest.raises(ValueError, match="--max-entries"):
        discover_and_extract(tmp_path, content_type, max_entries=1)


def test_default_scan_compares_all_343_skills_at_2048_dimensions(tmp_path: Path, monkeypatch) -> None:
    _collection(tmp_path, 343)
    client = _client(2_048, duplicate_last=True)
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: client)

    (result,) = run_similarity_check(tmp_path, content_type="skill")

    assert not result.is_incomplete
    assert len(result.errors) == 1
    assert result.errors[0].startswith("[SIMILARITY-CRITICAL]")
    indexed = next(detail for detail in result.success_details if detail.check_name == "index_built")
    assert indexed.metadata["entry_count"] == 343
    assert sum(len(call.args[0]) for call in client.embed.call_args_list) == 343
    assert len(result.findings) == 1
    match = result.findings[0]
    assert {match.metadata["entry_a"], match.metadata["entry_b"]} == {"skill-0", "skill-342"}
    assert match.metadata["score"] == 1.0


@pytest.mark.parametrize("limit, succeeds", [(8, False), (9, True)])
def test_cli_passes_comparison_budget_through_real_pipeline(tmp_path: Path, monkeypatch, limit: int, succeeds: bool):
    _collection(tmp_path, 3)
    client = _client()
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: client)
    cosine = MagicMock(wraps=registry_module.unit_vector_similarity)
    monkeypatch.setattr(registry_module, "unit_vector_similarity", cosine)

    result = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(tmp_path),
            "--type",
            "skill",
            "--max-entries",
            "3",
            "--max-scalar-comparisons",
            str(limit),
            "-r",
            "cli",
        ],
    )

    assert result.exit_code == (0 if succeeds else 1), result.output
    assert cosine.call_count == (3 if succeeds else 0)
    if not succeeds:
        assert "--max-scalar-comparisons" in result.output


def test_cli_entry_limit_fails_without_embedding_or_partial_catalog(tmp_path: Path, monkeypatch) -> None:
    _collection(tmp_path, 3)
    client = _client()
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: client)
    catalog = tmp_path / "catalog.json"

    result = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(tmp_path),
            "--type",
            "skill",
            "--max-entries",
            "2",
            "--save-catalog",
            str(catalog),
            "-r",
            "cli",
        ],
    )

    assert result.exit_code == 1
    assert "--max-entries" in result.output
    client.embed.assert_not_called()
    client.embed_chunked.assert_not_called()
    assert not catalog.exists()


@pytest.mark.parametrize(
    "flag,value",
    [("--max-entries", "0"), ("--max-entries", "5001"), ("--max-scalar-comparisons", "0"), ("--max-entries", "1.5")],
)
def test_cli_rejects_invalid_limits(tmp_path: Path, monkeypatch, flag: str, value: str) -> None:
    client = _client()
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: client)
    result = CliRunner().invoke(cli, ["similarity-check", str(tmp_path), flag, value])
    assert result.exit_code == 2
    assert flag in result.output
    client.embed.assert_not_called()


@pytest.mark.parametrize("query_method", ["query", "query_entry"])
@pytest.mark.parametrize("limit, succeeds", [(8, False), (9, True)])
def test_catalog_queries_keep_catalog_capacity_and_apply_work_budget(
    tmp_path: Path, monkeypatch, query_method: str, limit: int, succeeds: bool
) -> None:
    _collection(tmp_path, 3)
    client = _client()
    source = EmbeddingRegistry(client)
    source.build_from_directory(tmp_path, "skill")
    catalog = tmp_path / "catalog.json"
    source.save_catalog(catalog)
    registry = EmbeddingRegistry(client, max_entries=1, max_scalar_comparisons=limit)
    registry.load_catalog(catalog)
    assert registry.size == 3
    cosine = MagicMock(wraps=registry_module.unit_vector_similarity)
    monkeypatch.setattr(registry_module, "unit_vector_similarity", cosine)
    target = ContentEntry("target", "Target skill", "target", "skill") if query_method == "query_entry" else "target"
    if succeeds:
        matches = getattr(registry, query_method)(target, 0.75)
        assert len(matches) == 1
        assert matches[0].entry_b == "skill-0"
        assert cosine.call_count == 3
    else:
        with pytest.raises(ValueError, match="--max-scalar-comparisons"):
            getattr(registry, query_method)(target, 0.75)
        client.embed_single.assert_not_called()
        cosine.assert_not_called()


# Distinct skills with this embedding previously scored 0.9999999999999999 after normalization.
_EXACT_DUPLICATE_VECTOR = [-5.479683767155414, -6.166558151604646]


def _identical_client() -> MagicMock:
    client = _client()
    client.embed.side_effect = lambda texts: [list(_EXACT_DUPLICATE_VECTOR) for _ in texts]
    client.embed_single.return_value = list(_EXACT_DUPLICATE_VECTOR)
    return client


def test_cli_reports_identical_embeddings_at_threshold_one(tmp_path: Path, monkeypatch) -> None:
    _collection(tmp_path / "skills", 2)
    client = _identical_client()
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: client)
    reports = tmp_path / "reports"

    result = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(tmp_path / "skills"),
            "--type",
            "skill",
            "--threshold",
            "1",
            "-r",
            "json",
            "sarif",
            "-o",
            str(reports),
        ],
    )

    assert result.exit_code == 1, result.output
    (finding,) = json.loads((reports / "skillevaluator-similarity.json").read_text())["results"][0]["findings"]
    assert finding["check_name"] == "EXACT_DUPLICATE"
    assert finding["metadata"]["score"] == 1.0
    sarif = json.loads((reports / "skillevaluator-similarity.sarif.json").read_text())
    assert [item["ruleId"] for item in sarif["runs"][0]["results"]] == ["Similarity-Check/EXACT_DUPLICATE"]


@pytest.mark.parametrize("query_method", ["query", "query_entry"])
def test_catalog_queries_report_identical_embeddings_at_threshold_one(tmp_path: Path, query_method: str) -> None:
    _collection(tmp_path, 2)
    registry = EmbeddingRegistry(_identical_client())
    registry.build_from_directory(tmp_path, "skill")
    target = ContentEntry("target", "Target skill", "target", "skill") if query_method == "query_entry" else "target"

    assert [match.score for match in getattr(registry, query_method)(target, 1.0)] == [1.0, 1.0]
    assert [match.score for match in registry.find_duplicates(1.0)] == [1.0]


@pytest.mark.parametrize("operation", ["find_duplicates", "query", "query_entry"])
def test_loaded_catalog_rejects_over_budget_work_before_normalizing(
    tmp_path: Path, monkeypatch, operation: str
) -> None:
    _collection(tmp_path, 3)
    client = _client()
    source = EmbeddingRegistry(client)
    source.build_from_directory(tmp_path, "skill")
    catalog = tmp_path / "catalog.json"
    source.save_catalog(catalog)
    registry = EmbeddingRegistry(client, max_scalar_comparisons=8)
    registry.load_catalog(catalog)
    normalize = MagicMock(side_effect=AssertionError("the work budget must fail before normalizing the catalog"))
    monkeypatch.setattr(registry_module, "normalize_embedding_vector", normalize)
    target = ContentEntry("target", "Target skill", "target", "skill") if operation == "query_entry" else "target"

    with pytest.raises(ValueError, match="--max-scalar-comparisons"):
        if operation == "find_duplicates":
            registry.find_duplicates(0.75)
        else:
            getattr(registry, operation)(target, 0.75)

    normalize.assert_not_called()
    client.embed_single.assert_not_called()
