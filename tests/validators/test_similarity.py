# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SimilarityValidator -- duplicate content detection."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skillevaluator.embedding.client import SimilarityConfigError
from skillevaluator.embedding.extractor import ContentEntry
from skillevaluator.embedding.registry import SimilarityMatch
from skillevaluator.models.result import Severity
from skillevaluator.validators.similarity import SimilarityValidator


class TestSimilarityValidatorProperties:
    def test_name(self) -> None:
        assert SimilarityValidator().name == "Similarity Check"

    def test_description(self) -> None:
        assert SimilarityValidator().description

    def test_no_model_flag_defers_to_provider_resolution(self) -> None:
        """Without --model, the validator must not pin a default model.

        Eagerly substituting SIMILARITY_DEFAULT_MODEL here overrides
        SKILL_EVAL_EMBEDDING_MODEL inside EmbeddingClient and breaks every
        non-NVIDIA embedding endpoint (docs/configuration.mdx contract).
        """
        assert SimilarityValidator()._model is None
        assert SimilarityValidator(model="custom-model")._model == "custom-model"


class TestValidateUnknownContentType:
    @patch("skillevaluator.cli_core.detect_content_type", return_value="unknown")
    def test_auto_detect_returns_error(self, _mock_detect, tmp_path: Path) -> None:
        """When auto-detection returns CONTENT_TYPE_UNKNOWN, validation fails."""
        validator = SimilarityValidator(content_type=None)
        result = validator.validate(tmp_path)

        assert not result.passed
        assert any("auto-detect" in e.lower() for e in result.errors)

    @patch("skillevaluator.cli_core.detect_content_type", return_value="unknown")
    def test_auto_detect_error_uses_target_label_without_host_path(self, _mock_detect, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "external-skill"
        target.mkdir(parents=True)

        result = SimilarityValidator(content_type=None).validate(target)

        error_text = " ".join(result.errors)
        assert str(target) not in error_text
        assert target.name in error_text


class TestValidateNoContent:
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_empty_directory_fails_actionably(self, _mock_client_cls, mock_registry_cls, tmp_path: Path):
        mock_registry = MagicMock()
        mock_registry.build_from_directory.return_value = 0
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill")
        result = validator.validate(tmp_path)

        assert not result.passed
        assert any("no skill content" in error.lower() for error in result.errors)

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_manifest_file_input_is_rejected(self, _mock_client_cls, mock_registry_cls, tmp_path: Path) -> None:
        manifest = tmp_path / "SKILL.md"
        manifest.write_text("---\nname: sample\ndescription: Sample skill\n---\n")

        result = SimilarityValidator(content_type="skill").validate(manifest)

        assert not result.passed
        assert any("directory" in error.lower() for error in result.errors)
        mock_registry_cls.return_value.build_from_directory.assert_not_called()

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_direct_rule_scan_requires_two_items(self, _mock_client_cls, mock_registry_cls, tmp_path: Path) -> None:
        registry = mock_registry_cls.return_value
        registry.build_from_directory.side_effect = ValueError("Collection similarity requires at least 2 items")

        result = SimilarityValidator(content_type="rules").validate(tmp_path)

        assert not result.passed
        registry.build_from_directory.assert_called_once_with(
            tmp_path,
            "rules",
            minimum_entries=2,
        )


class TestValidateNoDuplicates:
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_unique_content_passes(self, _mock_client_cls, mock_registry_cls, tmp_path: Path):
        mock_registry = MagicMock()
        mock_registry.build_from_directory.return_value = 3
        mock_registry.find_duplicates.return_value = []
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill")
        result = validator.validate(tmp_path)

        assert result.passed
        assert any("No duplicates" in d.message for d in result.success_details)


class TestDirectCollectionSemantics:
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_one_skill_fails_before_embedding_with_intra_skill_guidance(self, mock_client_cls, tmp_path: Path) -> None:
        skill_dir = tmp_path / "only-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: only-skill\ndescription: Only skill in collection\n---\n")
        client = mock_client_cls.return_value
        client.embed.return_value = [[1.0, 0.0]]

        result = SimilarityValidator(content_type="skill").validate(skill_dir)

        assert not result.passed
        assert any("context-optimization-check" in error for error in result.errors)
        client.embed.assert_not_called()


class TestValidateDuplicatesFound:
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_critical_duplicate_fails(self, _mock_client_cls, mock_registry_cls, tmp_path: Path):
        match = SimilarityMatch(
            entry_a="skill-a",
            entry_b="skill-b",
            score=0.97,
            path_a="skills/a",
            path_b="skills/b",
            classification="EXACT_DUPLICATE",
            severity=Severity.CRITICAL,
        )
        mock_registry = MagicMock()
        mock_registry.build_from_directory.return_value = 2
        mock_registry.find_duplicates.return_value = [match]
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill")
        result = validator.validate(tmp_path)

        assert not result.passed
        assert len(result.findings) == 1
        assert result.findings[0].category == "SIMILARITY"
        assert result.findings[0].severity == Severity.CRITICAL

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_multiple_matches_all_recorded(self, _mock_client_cls, mock_registry_cls, tmp_path: Path):
        matches = [
            SimilarityMatch(
                entry_a="a",
                entry_b="b",
                score=0.96,
                path_a="p/a",
                path_b="p/b",
                classification="EXACT_DUPLICATE",
                severity=Severity.CRITICAL,
            ),
            SimilarityMatch(
                entry_a="a",
                entry_b="c",
                score=0.85,
                path_a="p/a",
                path_b="p/c",
                classification="SIMILAR",
                severity=Severity.MEDIUM,
            ),
        ]
        mock_registry = MagicMock()
        mock_registry.build_from_directory.return_value = 3
        mock_registry.find_duplicates.return_value = matches
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill")
        result = validator.validate(tmp_path)

        assert not result.passed
        assert len(result.findings) == 2

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_finding_metadata_populated(self, _mock_client_cls, mock_registry_cls, tmp_path: Path):
        match = SimilarityMatch(
            entry_a="skill-x",
            entry_b="skill-y",
            score=0.91,
            path_a="skills/x",
            path_b="skills/y",
            classification="HIGH_SIMILARITY",
            severity=Severity.HIGH,
        )
        mock_registry = MagicMock()
        mock_registry.build_from_directory.return_value = 2
        mock_registry.find_duplicates.return_value = [match]
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill")
        result = validator.validate(tmp_path)

        finding = result.findings[0]
        assert finding.metadata["entry_a"] == "skill-x"
        assert finding.metadata["entry_b"] == "skill-y"
        assert finding.metadata["path_b"] == "skills/y"
        assert finding.metadata["score"] == 0.91


class TestValidateConfigError:
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_missing_api_key_during_build(self, _mock_client_cls, mock_registry_cls, tmp_path: Path):
        """SimilarityConfigError raised during build_from_directory is caught."""
        mock_registry = MagicMock()
        mock_registry.build_from_directory.side_effect = SimilarityConfigError("No API key")
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill")
        result = validator.validate(tmp_path)

        assert not result.passed
        assert any("API key" in e for e in result.errors)


class TestValidateCacheWorkflow:
    @patch("skillevaluator.validators.similarity.extract_from_skill")
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_deprecated_cache_alias_loads_catalog(
        self, _mock_client_cls, mock_registry_cls, mock_extract, tmp_path: Path
    ):
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("{}")

        mock_registry = MagicMock()
        mock_registry.size = 2
        mock_registry.query_entry.return_value = []
        mock_registry_cls.return_value = mock_registry
        target = ContentEntry(
            name="candidate",
            description="Candidate skill",
            path=str(tmp_path / "candidate"),
            content_type="skill",
        )
        (tmp_path / "candidate").mkdir()
        mock_extract.return_value = target

        validator = SimilarityValidator(content_type="skill", cache_path=cache_file)
        result = validator.validate(tmp_path)

        mock_registry.load_catalog.assert_called_once_with(cache_file)
        mock_registry.query_entry.assert_called_once_with(target, 0.75)
        assert result.passed

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_deprecated_save_cache_alias_saves_catalog(self, _mock_client_cls, mock_registry_cls, tmp_path: Path):
        save_path = tmp_path / "output.json"

        mock_registry = MagicMock()
        mock_registry.build_from_directory.return_value = 2
        mock_registry.find_duplicates.return_value = []
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill", save_cache_path=save_path)
        result = validator.validate(tmp_path)

        mock_registry.save_catalog.assert_called_once_with(save_path)
        assert result.passed

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_deprecated_cache_alias_missing_file_fails_closed(
        self, _mock_client_cls, mock_registry_cls, tmp_path: Path
    ):
        nonexistent_cache = tmp_path / "missing.json"

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill", cache_path=nonexistent_cache)
        result = validator.validate(tmp_path)

        mock_registry.load_catalog.assert_not_called()
        mock_registry.build_from_directory.assert_not_called()
        assert not result.passed
        assert any("does not exist" in error for error in result.errors)


class TestValidateFullBodyMode:
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_full_body_flag_passed_to_registry(self, _mock_client_cls, mock_registry_cls, tmp_path: Path):
        mock_registry = MagicMock()
        mock_registry.build_from_directory.return_value = 1
        mock_registry.find_duplicates.return_value = []
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill", full_body=True)
        validator.validate(tmp_path)

        mock_registry_cls.assert_called_once()
        _, kwargs = mock_registry_cls.call_args
        assert kwargs["full_body"] is True


class TestValidateCustomThreshold:
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_threshold_passed_to_find_duplicates(self, _mock_client_cls, mock_registry_cls, tmp_path: Path):
        mock_registry = MagicMock()
        mock_registry.build_from_directory.return_value = 2
        mock_registry.find_duplicates.return_value = []
        mock_registry_cls.return_value = mock_registry

        validator = SimilarityValidator(content_type="skill", threshold=0.90)
        validator.validate(tmp_path)

        mock_registry.find_duplicates.assert_called_once_with(0.90)


class TestLocalCatalogWorkflow:
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_missing_catalog_fails_closed(self, _mock_client_cls, mock_registry_cls, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"

        result = SimilarityValidator(content_type="skill", catalog_path=missing).validate(tmp_path)

        assert not result.passed
        assert any("does not exist" in error for error in result.errors)
        assert str(tmp_path) not in " ".join(result.errors)
        mock_registry_cls.return_value.build_from_directory.assert_not_called()

    @patch("skillevaluator.validators.similarity.extract_from_skill")
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_catalog_compares_exactly_one_target_against_loaded_entries(
        self, _mock_client_cls, mock_registry_cls, mock_extract, tmp_path: Path
    ) -> None:
        catalog = tmp_path / "catalog.json"
        catalog.write_text("{}")
        target = ContentEntry(
            name="candidate",
            description="Candidate skill",
            path=str(tmp_path / "candidate"),
            content_type="skill",
        )
        (tmp_path / "candidate").mkdir()
        mock_extract.return_value = target
        registry = mock_registry_cls.return_value
        registry.size = 3
        registry.query_entry.return_value = []

        result = SimilarityValidator(content_type="skill", catalog_path=catalog).validate(tmp_path / "candidate")

        registry.load_catalog.assert_called_once_with(catalog)
        registry.query_entry.assert_called_once_with(target, 0.75)
        registry.find_duplicates.assert_not_called()
        assert result.passed
        assert any("3 catalog entries" in detail.message for detail in result.success_details)

    @patch("skillevaluator.validators.similarity.extract_from_skill")
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_catalog_rejects_target_without_root_skill(
        self, _mock_client_cls, mock_registry_cls, mock_extract, tmp_path: Path
    ) -> None:
        catalog = tmp_path / "catalog.json"
        catalog.write_text("{}")
        mock_extract.return_value = None

        result = SimilarityValidator(content_type="skill", catalog_path=catalog).validate(tmp_path)

        assert not result.passed
        assert any("root SKILL.md" in error for error in result.errors)
        mock_registry_cls.return_value.query_entry.assert_not_called()

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_catalog_extracts_only_the_root_skill(self, _mock_client_cls, mock_registry_cls, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog.json"
        catalog.write_text("{}")
        target = tmp_path / "target"
        target.mkdir()
        (target / "SKILL.md").write_text("---\nname: root-skill\ndescription: Root target skill\n---\n")
        nested = target / "references" / "nested"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("---\nname: nested-skill\ndescription: Nested reference skill\n---\n")
        registry = mock_registry_cls.return_value
        registry.size = 1
        registry.query_entry.return_value = []

        result = SimilarityValidator(content_type="skill", catalog_path=catalog).validate(target)

        assert result.passed
        queried_entry = registry.query_entry.call_args.args[0]
        assert queried_entry.name == "root-skill"

    @patch("skillevaluator.validators.similarity.extract_from_skill")
    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_catalog_finding_labels_target_and_catalog_roles(
        self, _mock_client_cls, mock_registry_cls, mock_extract, tmp_path: Path
    ) -> None:
        catalog = tmp_path / "catalog.json"
        catalog.write_text("{}")
        target = ContentEntry(
            name="shared-name",
            description="Candidate skill",
            path=str(tmp_path / "candidate"),
            content_type="skill",
        )
        (tmp_path / "candidate").mkdir()
        mock_extract.return_value = target
        mock_registry_cls.return_value.size = 1
        mock_registry_cls.return_value.query_entry.return_value = [
            SimilarityMatch(
                entry_a="shared-name",
                entry_b="shared-name",
                score=1.0,
                path_a="candidate",
                path_b="catalog/shared-name",
                classification="EXACT_DUPLICATE",
                severity=Severity.CRITICAL,
            )
        ]

        result = SimilarityValidator(content_type="skill", catalog_path=catalog).validate(tmp_path / "candidate")

        finding = result.findings[0]
        assert "Target skill 'shared-name'" in finding.message
        assert "catalog skill 'shared-name'" in finding.message
        assert finding.metadata["comparison_mode"] == "target-vs-catalog"

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_save_catalog_keeps_direct_pairwise_scan(self, _mock_client_cls, mock_registry_cls, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog.json"
        registry = mock_registry_cls.return_value
        registry.build_from_directory.return_value = 1
        registry.find_duplicates.return_value = []

        result = SimilarityValidator(content_type="skill", save_catalog_path=catalog).validate(tmp_path)

        registry.find_duplicates.assert_called_once_with(0.75)
        registry.save_catalog.assert_called_once_with(catalog)
        assert result.passed
        success_text = " ".join(detail.message for detail in result.success_details)
        assert catalog.name in success_text
        assert str(tmp_path) not in success_text

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_save_catalog_rejects_empty_collection(self, _mock_client_cls, mock_registry_cls, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog.json"
        mock_registry_cls.return_value.build_from_directory.return_value = 0

        result = SimilarityValidator(content_type="skill", save_catalog_path=catalog).validate(tmp_path)

        assert not result.passed
        assert any("empty" in error.lower() for error in result.errors)
        mock_registry_cls.return_value.save_catalog.assert_not_called()

    @patch("skillevaluator.validators.similarity.EmbeddingRegistry")
    @patch("skillevaluator.validators.similarity.EmbeddingClient")
    def test_malformed_catalog_is_an_actionable_failure(
        self, _mock_client_cls, mock_registry_cls, tmp_path: Path
    ) -> None:
        catalog = tmp_path / "catalog.json"
        catalog.write_text("{}")
        mock_registry_cls.return_value.load_catalog.side_effect = ValueError(
            f"Unsupported catalog schema version in {catalog}"
        )

        result = SimilarityValidator(content_type="skill", catalog_path=catalog).validate(tmp_path)

        assert not result.passed
        assert "Unsupported catalog schema version" in result.errors[0]
        assert catalog.name in result.errors[0]
        assert str(tmp_path) not in result.errors[0]

    def test_catalog_options_are_mutually_exclusive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cannot be used together"):
            SimilarityValidator(
                catalog_path=tmp_path / "in.json",
                save_catalog_path=tmp_path / "out.json",
            )


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -0.1, 1.1])
def test_similarity_validator_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"finite.*\[0, 1\]"):
        SimilarityValidator(threshold=threshold)
