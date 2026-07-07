# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for EmbeddingRegistry -- index building, caching, and duplicate detection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import skillevaluator.embedding.extractor as extractor_module
import skillevaluator.embedding.registry as registry_module
from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.extractor import ContentEntry
from skillevaluator.embedding.registry import (
    EmbeddingRegistry,
    RegistryEntry,
    classify,
)
from skillevaluator.models.result import Severity


class TestClassify:
    def test_exact_duplicate(self) -> None:
        classification, severity = classify(0.97)
        assert classification == "EXACT_DUPLICATE"
        assert severity == Severity.CRITICAL

    def test_high_similarity(self) -> None:
        classification, severity = classify(0.92)
        assert classification == "HIGH_SIMILARITY"
        assert severity == Severity.HIGH

    def test_medium_similarity(self) -> None:
        classification, severity = classify(0.80)
        assert classification == "SIMILAR"
        assert severity == Severity.MEDIUM

    def test_low_similarity(self) -> None:
        classification, severity = classify(0.55)
        assert classification == "LOOSELY_RELATED"
        assert severity == Severity.LOW

    def test_distinct(self) -> None:
        classification, severity = classify(0.30)
        assert classification == "DISTINCT"
        assert severity == Severity.INFO

    def test_boundary_critical(self) -> None:
        classification, _ = classify(0.95)
        assert classification == "EXACT_DUPLICATE"

    def test_boundary_high(self) -> None:
        classification, _ = classify(0.90)
        assert classification == "HIGH_SIMILARITY"

    def test_boundary_medium(self) -> None:
        classification, _ = classify(0.75)
        assert classification == "SIMILAR"

    def test_boundary_low(self) -> None:
        classification, _ = classify(0.50)
        assert classification == "LOOSELY_RELATED"


def _make_mock_client(vectors: list[list[float]]) -> EmbeddingClient:
    """Create a mock EmbeddingClient that returns predetermined vectors."""
    client = MagicMock(spec=EmbeddingClient)
    client.embed.return_value = vectors
    client.embed_single.side_effect = lambda _text: vectors[0]
    client.embed_chunked.side_effect = lambda _text, **_kw: vectors[0]
    client.model = "test-model"
    client._resolved_config.return_value = SimpleNamespace(
        provider="nv_build",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    client.cosine_similarity = EmbeddingClient.cosine_similarity
    return client


class TestBuildFromDirectory:
    def test_empty_collection_debug_logs_use_label_without_host_path(self, tmp_path: Path, monkeypatch) -> None:
        collection = tmp_path / "nested" / "external-skills"
        collection.mkdir(parents=True)
        debug = MagicMock()
        monkeypatch.setattr(registry_module, "discover_and_extract", lambda *_args: [])
        monkeypatch.setattr(registry_module.logger, "debug", debug)

        count = EmbeddingRegistry(_make_mock_client([])).build_from_directory(collection, "skill")

        assert count == 0
        rendered = "\n".join(call.args[0] % call.args[1:] for call in debug.call_args_list)
        assert str(collection) not in rendered
        assert collection.name in rendered

    def test_build_from_directory_skills(self, tmp_path: Path, write_skill) -> None:
        write_skill(tmp_path, "skill-a", "First skill")
        write_skill(tmp_path, "skill-b", "Second skill")

        mock_client = _make_mock_client([[0.1, 0.2], [0.3, 0.4]])
        registry = EmbeddingRegistry(mock_client)

        count = registry.build_from_directory(tmp_path, "skill")

        assert count == 2
        assert registry.size == 2
        mock_client.embed.assert_called_once()

    def test_build_from_directory_rules(self, tmp_path: Path) -> None:
        rule_file = tmp_path / "lint.mdc"
        rule_file.write_text("---\nalwaysApply: false\ntitle: Lint\ndescription: Lint rules\n---\n")

        mock_client = _make_mock_client([[0.5, 0.6]])
        registry = EmbeddingRegistry(mock_client)
        count = registry.build_from_directory(tmp_path, "rules")

        assert count == 1

    def test_build_full_body_mode(self, tmp_path: Path, write_skill) -> None:
        write_skill(tmp_path, "body-skill", "Full body test")

        mock_client = _make_mock_client([[0.1, 0.2]])
        registry = EmbeddingRegistry(mock_client, full_body=True)
        count = registry.build_from_directory(tmp_path, "skill")

        assert count == 1
        mock_client.embed_chunked.assert_called_once()

    def test_empty_directory_returns_zero(self, tmp_path: Path) -> None:
        mock_client = _make_mock_client([])
        registry = EmbeddingRegistry(mock_client)
        count = registry.build_from_directory(tmp_path, "skill")

        assert count == 0
        assert registry.size == 0

    def test_duplicate_display_names_keep_distinct_relative_path_ids(self, tmp_path: Path) -> None:
        for directory in ("team-a", "team-b"):
            skill_dir = tmp_path / directory
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: shared-name\ndescription: Shared display name\n---\n")

        mock_client = _make_mock_client([[1.0, 0.0], [0.0, 1.0]])
        registry = EmbeddingRegistry(mock_client)

        count = registry.build_from_directory(tmp_path, "skill")

        assert count == 2
        assert set(registry._entries) == {"skill:team-a", "skill:team-b"}
        assert {entry.name for entry in registry._entries.values()} == {"shared-name"}
        assert {entry.path for entry in registry._entries.values()} == {"team-a", "team-b"}

    def test_entry_limit_is_enforced_before_embedding(self, tmp_path: Path, write_skill, monkeypatch) -> None:
        write_skill(tmp_path, "skill-a", "First skill")
        write_skill(tmp_path, "skill-b", "Second skill")
        monkeypatch.setattr(extractor_module, "MAX_COLLECTION_ENTRIES", 1, raising=False)
        client = _make_mock_client([[1.0, 0.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="entry limit"):
            EmbeddingRegistry(client).build_from_directory(tmp_path, "skill")

        client.embed.assert_not_called()

    def test_per_file_limit_is_enforced_before_embedding(self, tmp_path: Path, write_skill, monkeypatch) -> None:
        skill_dir = write_skill(tmp_path, "large", "Large skill")
        (skill_dir / "SKILL.md").write_text("---\nname: large\ndescription: Large skill\n---\n" + "x" * 128)
        monkeypatch.setattr(extractor_module, "MAX_MANIFEST_BYTES", 64, raising=False)
        client = _make_mock_client([[1.0, 0.0]])

        with pytest.raises(ValueError, match="per-file byte limit"):
            EmbeddingRegistry(client).build_from_directory(tmp_path, "skill")

        client.embed.assert_not_called()

    def test_total_byte_limit_is_enforced_before_embedding(self, tmp_path: Path, write_skill, monkeypatch) -> None:
        write_skill(tmp_path, "skill-a", "First skill with enough content")
        write_skill(tmp_path, "skill-b", "Second skill with enough content")
        monkeypatch.setattr(extractor_module, "MAX_MANIFEST_BYTES", 1_024, raising=False)
        monkeypatch.setattr(extractor_module, "MAX_COLLECTION_BYTES", 100, raising=False)
        client = _make_mock_client([[1.0, 0.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="total byte limit"):
            EmbeddingRegistry(client).build_from_directory(tmp_path, "skill")

        client.embed.assert_not_called()

    def test_description_embeddings_are_sent_in_bounded_batches(self, tmp_path: Path, write_skill, monkeypatch) -> None:
        for index in range(5):
            write_skill(tmp_path, f"skill-{index}", f"Description {index}")
        monkeypatch.setattr(registry_module, "EMBEDDING_BATCH_SIZE", 2, raising=False)
        client = _make_mock_client([])
        client.embed.side_effect = lambda texts: [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

        count = EmbeddingRegistry(client).build_from_directory(tmp_path, "skill")

        assert count == 5
        assert [len(call.args[0]) for call in client.embed.call_args_list] == [2, 2, 1]

    def test_description_text_limit_fails_before_embedding(self, tmp_path: Path, write_skill, monkeypatch) -> None:
        write_skill(tmp_path, "large-description", "x" * 128)
        monkeypatch.setattr(registry_module, "MAX_DESCRIPTION_EMBEDDING_TEXT_CHARS", 64, raising=False)
        client = _make_mock_client([[1.0, 0.0]])

        with pytest.raises(ValueError, match=r"(?i)description.*character limit"):
            EmbeddingRegistry(client).build_from_directory(tmp_path, "skill")

        client.embed.assert_not_called()
        client.embed_chunked.assert_not_called()

    @pytest.mark.parametrize("vector", [[math.nan, 0.0], [1e308, 1e308]])
    def test_build_rejects_unsafe_provider_vectors(self, tmp_path: Path, write_skill, vector: list[float]) -> None:
        write_skill(tmp_path, "skill-a", "First skill")
        client = _make_mock_client([vector])

        with pytest.raises(ValueError, match=r"finite|magnitude|stable"):
            EmbeddingRegistry(client).build_from_directory(tmp_path, "skill")


class TestFindDuplicates:
    def _build_registry_with_entries(
        self, client: EmbeddingClient, entries: dict[str, list[float]]
    ) -> EmbeddingRegistry:
        registry = EmbeddingRegistry(client)
        for name, embedding in entries.items():
            registry._entries[name] = RegistryEntry(
                name=name,
                description=f"Description of {name}",
                path=f"skills/{name}",
                content_type="skill",
                embedding=embedding,
            )
        return registry

    def test_find_duplicates_exact(self) -> None:
        mock_client = _make_mock_client([])
        registry = self._build_registry_with_entries(
            mock_client,
            {"skill-a": [1.0, 0.0, 0.0], "skill-b": [1.0, 0.001, 0.0]},
        )

        matches = registry.find_duplicates(threshold=0.50)
        assert len(matches) == 1
        assert matches[0].classification == "EXACT_DUPLICATE"
        assert matches[0].score > 0.95

    def test_find_duplicates_high(self) -> None:
        mock_client = _make_mock_client([])
        registry = self._build_registry_with_entries(
            mock_client,
            {"skill-a": [1.0, 0.0, 0.0], "skill-b": [0.92, 0.4, 0.0]},
        )

        matches = registry.find_duplicates(threshold=0.50)
        assert len(matches) == 1
        assert matches[0].classification == "HIGH_SIMILARITY"

    def test_find_duplicates_medium(self) -> None:
        mock_client = _make_mock_client([])
        registry = self._build_registry_with_entries(
            mock_client,
            {"skill-a": [1.0, 0.0, 0.0], "skill-b": [0.8, 0.6, 0.0]},
        )

        matches = registry.find_duplicates(threshold=0.50)
        assert len(matches) == 1
        assert matches[0].classification == "SIMILAR"

    def test_find_duplicates_low(self) -> None:
        mock_client = _make_mock_client([])
        registry = self._build_registry_with_entries(
            mock_client,
            {"skill-a": [1.0, 0.0, 0.0], "skill-b": [0.6, 0.8, 0.0]},
        )

        matches = registry.find_duplicates(threshold=0.50)
        assert len(matches) == 1
        assert matches[0].classification == "LOOSELY_RELATED"

    def test_no_duplicates_below_threshold(self) -> None:
        mock_client = _make_mock_client([])
        registry = self._build_registry_with_entries(
            mock_client,
            {"skill-a": [1.0, 0.0, 0.0], "skill-b": [0.0, 1.0, 0.0]},
        )

        matches = registry.find_duplicates(threshold=0.75)
        assert matches == []

    def test_sorted_by_score_descending(self) -> None:
        mock_client = _make_mock_client([])
        registry = self._build_registry_with_entries(
            mock_client,
            {
                "a": [1.0, 0.0],
                "b": [0.9, 0.4],
                "c": [1.0, 0.01],
            },
        )

        matches = registry.find_duplicates(threshold=0.50)
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_pairwise_comparison_limit_fails_before_comparing(self, monkeypatch) -> None:
        client = _make_mock_client([])
        registry = self._build_registry_with_entries(
            client,
            {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [1.0, 0.0]},
        )
        monkeypatch.setattr(registry_module, "MAX_PAIRWISE_COMPARISONS", 2, raising=False)

        with pytest.raises(ValueError, match="comparison limit"):
            registry.find_duplicates(0.75)

    def test_pairwise_match_limit_fails_closed(self, monkeypatch) -> None:
        client = _make_mock_client([])
        registry = self._build_registry_with_entries(
            client,
            {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [1.0, 0.0]},
        )
        monkeypatch.setattr(registry_module, "MAX_PAIRWISE_COMPARISONS", 10, raising=False)
        monkeypatch.setattr(registry_module, "MAX_SIMILARITY_MATCHES", 1, raising=False)

        with pytest.raises(ValueError, match="match limit"):
            registry.find_duplicates(0.75)

    def test_scalar_comparison_work_limit_fails_before_comparing(self, monkeypatch) -> None:
        client = _make_mock_client([])
        registry = self._build_registry_with_entries(
            client,
            {"a": [1.0, 0.0], "b": [1.0, 0.0]},
        )
        monkeypatch.setattr(registry_module, "MAX_SCALAR_COMPARISONS", 1, raising=False)

        with pytest.raises(ValueError, match=r"scalar.*limit|work limit"):
            registry.find_duplicates(0.75)


class TestCachePersistence:
    def test_load_save_cache_roundtrip(self, tmp_path: Path) -> None:
        mock_client = _make_mock_client([])
        registry = EmbeddingRegistry(mock_client)
        registry._entries["test-skill"] = RegistryEntry(
            name="test-skill",
            description="A test skill",
            path="skills/test-skill",
            content_type="skill",
            embedding=[0.1, 0.2, 0.3],
            entry_id="skill:skills/test-skill",
            content_fingerprint="a" * 64,
        )

        cache_file = tmp_path / "cache.json"
        registry.save_cache(cache_file)

        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert "entries" in data
        assert data["entries"][0]["id"] == "skill:skills/test-skill"
        assert data["entries"][0]["embedding"] == [0.1, 0.2, 0.3]

        new_registry = EmbeddingRegistry(mock_client)
        new_registry.load_cache(cache_file)

        assert new_registry.size == 1
        assert new_registry._entries["skill:skills/test-skill"].embedding == [0.1, 0.2, 0.3]

    def test_cache_includes_metadata(self, tmp_path: Path) -> None:
        mock_client = _make_mock_client([])
        registry = EmbeddingRegistry(mock_client)
        registry._entries["s"] = RegistryEntry(
            name="s",
            description="d",
            path="p",
            content_type="skill",
            embedding=[1.0],
            entry_id="skill:p",
            content_fingerprint="b" * 64,
        )

        cache_file = tmp_path / "meta.json"
        registry.save_cache(cache_file)

        data = json.loads(cache_file.read_text())
        assert "model" in data
        assert "provider" in data
        assert "schema_version" in data
        assert "vector_dimension" in data
        assert "created_at" in data
        assert "mode" in data


class TestQuery:
    def test_query_returns_matches_above_threshold(self) -> None:
        mock_client = _make_mock_client([[1.0, 0.0]])
        mock_client.embed_single.side_effect = None
        mock_client.embed_single.return_value = [1.0, 0.0]

        registry = EmbeddingRegistry(mock_client)
        registry._entries["similar"] = RegistryEntry(
            name="similar",
            description="d",
            path="p",
            content_type="skill",
            embedding=[0.99, 0.1],
        )
        registry._entries["different"] = RegistryEntry(
            name="different",
            description="d",
            path="p",
            content_type="skill",
            embedding=[0.0, 1.0],
        )

        matches = registry.query("test query", threshold=0.75)
        assert len(matches) == 1
        assert matches[0].entry_b == "similar"

    @pytest.mark.parametrize("query_method", ["query", "query_entry"])
    def test_description_query_text_limit_fails_before_embedding(self, monkeypatch, query_method: str) -> None:
        client = _make_mock_client([[1.0, 0.0]])
        registry = EmbeddingRegistry(client)
        registry._entries["skill:catalog"] = RegistryEntry(
            name="catalog",
            description="Catalog skill",
            path="catalog",
            content_type="skill",
            embedding=[1.0, 0.0],
        )
        registry._vector_dimension = 2
        monkeypatch.setattr(registry_module, "MAX_DESCRIPTION_EMBEDDING_TEXT_CHARS", 32)
        text = "x" * 33

        with pytest.raises(ValueError, match=r"(?i)description.*character limit"):
            if query_method == "query":
                registry.query(text, threshold=0.75)
            else:
                registry.query_entry(
                    ContentEntry(
                        name="candidate",
                        description=text,
                        path="candidate",
                        content_type="skill",
                    ),
                    threshold=0.75,
                )

        client.embed_single.assert_not_called()
        client.embed_chunked.assert_not_called()

    @pytest.mark.parametrize("query_method", ["query", "query_entry"])
    def test_full_body_query_text_limit_fails_before_embedding(self, monkeypatch, query_method: str) -> None:
        client = _make_mock_client([[1.0, 0.0]])
        registry = EmbeddingRegistry(client, full_body=True)
        registry._entries["skill:catalog"] = RegistryEntry(
            name="catalog",
            description="Catalog skill",
            path="catalog",
            content_type="skill",
            embedding=[1.0, 0.0],
        )
        registry._vector_dimension = 2
        monkeypatch.setattr(registry_module, "MAX_FULL_BODY_EMBEDDING_TEXT_BYTES", 32)
        text = "x" * 33

        with pytest.raises(ValueError, match=r"(?i)full-body.*byte limit"):
            if query_method == "query":
                registry.query(text, threshold=0.75)
            else:
                registry.query_entry(
                    ContentEntry(
                        name="candidate",
                        description="Candidate skill",
                        path="candidate",
                        content_type="skill",
                        full_text=text,
                    ),
                    threshold=0.75,
                )

        client.embed_single.assert_not_called()
        client.embed_chunked.assert_not_called()
