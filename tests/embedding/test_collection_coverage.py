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
