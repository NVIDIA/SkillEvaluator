# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A valid subset must never stand in for collection coverage."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.embedding import extractor
from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError
from skillevaluator.embedding.registry import EmbeddingRegistry
from skillevaluator.validators.similarity import SimilarityValidator


def _collection(root: Path, malformed: int = 1) -> None:
    for name, description in [("alpha", "Analyze logs"), ("beta", "Build packages")]:
        path = root / name
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n")
    for index in range(malformed):
        path = root / f"broken-{index:03d}"
        path.mkdir()
        (path / "SKILL.md").write_text("---\nname: [invalid YAML\n---\n")


def _client() -> MagicMock:
    client = MagicMock(spec=EmbeddingClient)
    client.model = "test-model"
    client._resolved_config.return_value = SimpleNamespace(provider="openai", base_url="https://api.openai.com/v1")
    client.embed.side_effect = lambda texts: [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]
    return client


def test_detailed_extraction_counts_selected_and_bounds_relative_reasons(tmp_path: Path) -> None:
    _collection(tmp_path, malformed=103)
    detailed = extractor.discover_and_extract_detailed(tmp_path, "skill")
    assert detailed.selected_count == 105
    assert detailed.indexed_count == 2
    assert detailed.skipped_count == 103
    metadata = detailed.to_metadata()
    assert len(metadata["exclusions"]) == 100
    assert metadata["omitted_exclusions"] == 3
    assert metadata["exclusions"][0] == {"path": "broken-000/SKILL.md", "reason": "invalid_yaml"}
    assert str(tmp_path) not in json.dumps(metadata)
    assert extractor.discover_and_extract(tmp_path, "skill") == detailed.entries
    with pytest.raises(ValueError, match="entry limit"):
        extractor.discover_and_extract_detailed(tmp_path, "skill", max_entries=104)


@pytest.mark.parametrize("malformed", [0, 1])
def test_validator_distinguishes_complete_from_incomplete_no_matches(
    tmp_path: Path, monkeypatch, malformed: int
) -> None:
    _collection(tmp_path, malformed)
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: _client())
    result = SimilarityValidator(content_type="skill").validate(tmp_path)
    assert result.is_incomplete is bool(malformed)
    assert result.passed is (not malformed)
    details = [detail for detail in result.success_details if detail.check_name == "collection_coverage"]
    assert len(details) == 1
    assert details[0].metadata["selected_count"] == 2 + malformed
    assert details[0].metadata["indexed_count"] == 2
    assert details[0].metadata["skipped_count"] == malformed
    assert any("No duplicates detected" in detail.message for detail in result.success_details) is (not malformed)


def test_incomplete_coverage_is_saved_in_report_without_replacing_catalog(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "skills"
    _collection(root)
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: _client())
    catalog = tmp_path / "catalog.json"
    catalog.write_text("original catalog")
    reports = tmp_path / "reports"
    result = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(root),
            "--type",
            "skill",
            "--save-catalog",
            str(catalog),
            "-r",
            "json",
            "-o",
            str(reports),
        ],
    )
    assert result.exit_code == 1, result.output
    assert catalog.read_text() == "original catalog"
    report = json.loads((reports / "skillevaluator-similarity.json").read_text())["results"][0]
    coverage = next(
        detail["metadata"] for detail in report["success_details"] if detail["check"] == "collection_coverage"
    )
    assert coverage["exclusions"] == [{"path": "broken-000/SKILL.md", "reason": "invalid_yaml"}]
    assert report["status"] == "incomplete"


def test_registry_rejects_partial_catalog_save(tmp_path: Path) -> None:
    _collection(tmp_path)
    registry = EmbeddingRegistry(_client())
    assert registry.build_from_directory(tmp_path, "skill") == 2
    with pytest.raises(ValueError, match="incomplete"):
        registry.save_catalog(tmp_path / "catalog.json")
    assert not (tmp_path / "catalog.json").exists()


@pytest.mark.parametrize(
    "content_type,filename,name_field",
    [("skill", "SKILL.md", "name"), ("rules", "rule.mdc", "title"), ("workflows", "workflow-rules.mdc", "title")],
)
def test_exclusion_reasons_account_for_every_selected_manifest(
    tmp_path: Path, content_type: str, filename: str, name_field: str
) -> None:
    documents = ["No frontmatter", "---\n[]\n---\n", f"---\n{name_field}: example\n---\n"]
    for index, document in enumerate(documents):
        path = tmp_path / str(index)
        path.mkdir()
        (path / filename).write_text(document)
    result = extractor.discover_and_extract_detailed(tmp_path, content_type)
    assert result.selected_count == 3
    assert result.indexed_count == 0
    assert result.skipped_count == 3
    assert [item["reason"] for item in result.exclusions] == [
        "missing_frontmatter",
        "invalid_frontmatter_mapping",
        "missing_name_or_description",
    ]


def test_incomplete_coverage_preserves_valid_subset_findings(tmp_path: Path, monkeypatch) -> None:
    _collection(tmp_path)
    client = _client()
    client.embed.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: client)
    result = SimilarityValidator(content_type="skill").validate(tmp_path)
    assert result.is_incomplete
    assert len(result.findings) == 1
    assert result.findings[0].check_name == "EXACT_DUPLICATE"
    assert not any("No duplicates detected" in detail.message for detail in result.success_details)


def test_too_few_valid_entries_still_reports_exclusions(tmp_path: Path, monkeypatch) -> None:
    _collection(tmp_path)
    (tmp_path / "beta" / "SKILL.md").unlink()
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: _client())
    result = SimilarityValidator(content_type="skill").validate(tmp_path)
    assert result.is_incomplete
    assert any("broken-000/SKILL.md: invalid_yaml" in error for error in result.errors)
    assert any("at least 2" in error for error in result.errors)


def test_successful_explicit_rebuild_replaces_incomplete_selection(tmp_path: Path) -> None:
    _collection(tmp_path)
    registry = EmbeddingRegistry(_client())
    registry.build_from_directory(tmp_path, "skill")
    registry.build_from_paths(tmp_path, [tmp_path / "alpha", tmp_path / "beta"])
    registry.save_catalog(tmp_path / "explicit.json")
    assert (tmp_path / "explicit.json").is_file()


def test_provider_failure_retains_selection_coverage(tmp_path: Path, monkeypatch) -> None:
    _collection(tmp_path)
    client = _client()
    client.embed.side_effect = SimilarityConfigError("synthetic unavailable")
    monkeypatch.setattr("skillevaluator.validators.similarity.EmbeddingClient", lambda **_kwargs: client)
    result = SimilarityValidator(content_type="skill").validate(tmp_path)
    assert result.is_incomplete
    assert any(detail.check_name == "collection_coverage" for detail in result.success_details)
    assert any("broken-000/SKILL.md: invalid_yaml" in error for error in result.errors)


def _single_skill(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Task {name}\n---\n")
    return path


@pytest.mark.parametrize("initial_malformed", [0, 1])
@pytest.mark.parametrize("failure", ["provider", "preflight", "minimum", "vector", "input"])
def test_failed_build_preserves_committed_index_and_coverage(
    tmp_path: Path, initial_malformed: int, failure: str
) -> None:
    first = tmp_path / "first"
    _collection(first, initial_malformed)
    second = tmp_path / "second"
    _single_skill(second, "gamma")
    _single_skill(second, "delta")
    if not initial_malformed:
        invalid = _single_skill(second, "broken")
        (invalid / "SKILL.md").write_text("No valid frontmatter")
    client = _client()
    client.embed_single.return_value = [1.0, 0.0]
    registry = EmbeddingRegistry(client, max_scalar_comparisons=1)
    registry.build_from_directory(first, "skill")
    committed_coverage = registry.coverage
    original = {key: entry.embedding[:] for key, entry in registry._entries.items()}
    if failure == "provider":
        client.embed.side_effect = SimilarityConfigError("synthetic failure")
    elif failure == "vector":
        client.embed.side_effect = lambda texts: [[float("nan"), 0.0] for _ in texts]
    elif failure == "input":
        # Each field fits the 16,384-character frontmatter field limit, so discovery
        # succeeds; together they exceed the description-mode embedding-text limit.
        (second / "gamma" / "SKILL.md").write_text(
            "---\nname: gamma-" + "g" * 9000 + "\ndescription: " + "x" * 9000 + "\n---\n"
        )
    with pytest.raises((ValueError, SimilarityConfigError)):
        registry.build_from_directory(
            second, "skill", minimum_entries=3 if failure == "minimum" else 1, for_pairwise_scan=failure == "preflight"
        )
    assert registry.coverage is committed_coverage
    assert registry.latest_attempt_coverage.skipped_count == (not initial_malformed)
    assert registry._vector_dimension == 2
    assert {key: entry.embedding for key, entry in registry._entries.items()} == original
    # Query behavior must continue to describe the committed index.
    registry._max_scalar_comparisons = 128000000
    assert [match.entry_b for match in registry.query("target", 0.75)] == ["alpha"]
    destination = tmp_path / "retained.json"
    if initial_malformed:
        with pytest.raises(ValueError, match="incomplete"):
            registry.save_catalog(destination)
        assert not destination.exists()
    else:
        registry.save_catalog(destination)
        assert {entry["name"] for entry in json.loads(destination.read_text())["entries"]} == {"alpha", "beta"}


@pytest.mark.parametrize("selection", ["strict_subset", "disjoint_explicit", "disjoint_directory", "clean_refresh"])
def test_successful_build_replaces_partial_index_completely(tmp_path: Path, selection: str) -> None:
    first = tmp_path / "first"
    _collection(first)
    client = _client()
    client.embed_single.return_value = [1.0, 0.0]
    registry = EmbeddingRegistry(client)
    registry.build_from_directory(first, "skill")
    expected = "alpha" if selection in {"strict_subset", "clean_refresh"} else "gamma"
    if selection == "strict_subset":
        count = registry.build_from_paths(first, [first / "alpha"])
    elif selection == "clean_refresh":
        (first / "beta" / "SKILL.md").unlink()
        (first / "broken-000" / "SKILL.md").unlink()
        count = registry.build_from_directory(first, "skill")
    else:
        second = tmp_path / "second"
        path = _single_skill(second, "gamma")
        count = (
            registry.build_from_paths(second, [path])
            if selection == "disjoint_explicit"
            else registry.build_from_directory(second, "skill")
        )
    assert count == 1
    assert registry.size == 1
    client.embed_single.return_value = [1.0, 0.0] if expected == "alpha" else [0.0, 1.0]
    assert [match.entry_b for match in registry.query("target", 0.75)] == [expected]
    destination = tmp_path / "replacement.json"
    registry.save_catalog(destination)
    assert [entry["name"] for entry in json.loads(destination.read_text())["entries"]] == [expected]
    if selection in {"strict_subset", "disjoint_explicit"}:
        assert registry.coverage is None
    else:
        assert registry.coverage.selected_count == 1
        assert registry.coverage.indexed_count == 1
        assert registry.coverage.skipped_count == 0
        assert registry.latest_attempt_coverage is registry.coverage


def test_empty_directory_replacement_does_not_retain_previous_entries(tmp_path: Path) -> None:
    first = tmp_path / "first"
    _collection(first)
    registry = EmbeddingRegistry(_client())
    registry.build_from_directory(first, "skill")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert registry.build_from_directory(empty, "skill") == 0
    assert registry.size == 0
    assert registry._vector_dimension is None
    with pytest.raises(ValueError, match="empty"):
        registry.save_catalog(tmp_path / "empty.json")


def test_complete_replacement_can_change_vector_width(tmp_path: Path) -> None:
    first = tmp_path / "first"
    _collection(first)
    client = _client()
    registry = EmbeddingRegistry(client)
    registry.build_from_directory(first, "skill")
    second = tmp_path / "second"
    _single_skill(second, "gamma")
    client.embed.side_effect = lambda texts: [[0.0, 0.0, 1.0] for _ in texts]
    assert registry.build_from_directory(second, "skill") == 1
    assert registry._vector_dimension == 3
    destination = tmp_path / "three-dimensional.json"
    registry.save_catalog(destination)
    saved = json.loads(destination.read_text())
    assert saved["vector_dimension"] == 3
    assert [entry["name"] for entry in saved["entries"]] == ["gamma"]
