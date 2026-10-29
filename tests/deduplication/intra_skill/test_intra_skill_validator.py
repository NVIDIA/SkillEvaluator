# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.intra_skill.intra_skill_validator."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator import constants
from skillevaluator.deduplication.intra_skill import intra_skill_validator
from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator, effective_chunk_limit
from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster
from skillevaluator.deduplication.utils.chunker import ContentChunk
from skillevaluator.embedding.client import MAX_EMBEDDING_VECTOR_DIMENSION, SimilarityConfigError
from skillevaluator.inference import LLMClientError, LLMVerdict
from skillevaluator.models.result import Severity


def _numbered_sections(count: int) -> str:
    """Return Markdown that chunks into exactly ``count`` sections."""
    return "\n".join(f"## Section {index}\n" + ("x" * 100) for index in range(count))


def _unit_embeddings(vector_dimension: int):
    """Return an ``embed`` side effect producing one valid vector per input."""
    embedding = [1.0] + [0.0] * (vector_dimension - 1)
    return lambda batch: [list(embedding) for _text in batch]


class TestIntraSkillValidatorProperties:
    def test_name(self) -> None:
        v = IntraSkillValidator()
        assert v.name == "Context Deduplication"

    def test_description(self) -> None:
        v = IntraSkillValidator()
        assert len(v.description) > 0

    def test_no_model_flag_defers_to_provider_resolution(self) -> None:
        """Without --model, the validator must not pin a default model.

        Eagerly substituting SIMILARITY_DEFAULT_MODEL here overrides
        SKILL_EVAL_EMBEDDING_MODEL inside EmbeddingClient and breaks every
        non-NVIDIA embedding endpoint (docs/configuration.mdx contract).
        """
        assert IntraSkillValidator()._embedding_model is None
        assert IntraSkillValidator(embedding_model="custom-model")._embedding_model == "custom-model"

    def test_default_embedding_model_comes_from_provider_resolution(self, monkeypatch) -> None:
        """The public NVIDIA Build default applies at the provider layer, not here."""
        from skillevaluator.provider_config import resolve_embedding_provider

        monkeypatch.setenv("SKILL_EVAL_EMBEDDING_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.delenv("SKILL_EVAL_EMBEDDING_MODEL", raising=False)

        assert resolve_embedding_provider().model == "nvidia/nemotron-3-embed-1b"


class TestEffectiveChunkLimit:
    def test_scalar_budget_lets_2048_dimensional_embeddings_reach_the_chunk_ceiling(self) -> None:
        assert constants.CONTENT_DEDUP_MAX_CHUNKS == 512
        assert constants.CONTENT_DEDUP_MAX_PAIR_COMPARISONS == 512 * 511 // 2
        assert constants.CONTENT_DEDUP_MAX_SCALAR_COMPARISONS == 267_911_168
        assert (
            constants.CONTENT_DEDUP_MAX_SCALAR_COMPARISONS
            == constants.CONTENT_DEDUP_MAX_PAIR_COMPARISONS * constants.CONTENT_DEDUP_SCALAR_BUDGET_DIMENSION
        )

    @pytest.mark.parametrize(
        ("vector_dimension", "expected_limit"),
        [(1, 512), (1024, 512), (1536, 512), (2048, 512), (2049, 511), (3072, 418), (4096, 362)],
    )
    def test_limit_depends_on_vector_dimension(self, vector_dimension: int, expected_limit: int) -> None:
        assert effective_chunk_limit(vector_dimension) == expected_limit

    def test_limit_is_the_largest_chunk_count_within_budget_for_every_width(self) -> None:
        budget = constants.CONTENT_DEDUP_MAX_SCALAR_COMPARISONS
        for vector_dimension in range(1, MAX_EMBEDDING_VECTOR_DIMENSION + 1):
            limit = effective_chunk_limit(vector_dimension)
            assert 2 <= limit <= constants.CONTENT_DEDUP_MAX_CHUNKS
            assert limit * (limit - 1) // 2 * vector_dimension <= budget
            if limit < constants.CONTENT_DEDUP_MAX_CHUNKS:
                assert (limit + 1) * limit // 2 * vector_dimension > budget

    @pytest.mark.parametrize("vector_dimension", [0, -1, True, 2048.0, "2048", None])
    def test_rejects_invalid_vector_dimension(self, vector_dimension: object) -> None:
        with pytest.raises(ValueError, match=r"vector dimension must be a positive integer"):
            effective_chunk_limit(vector_dimension)  # type: ignore[arg-type]

    def test_limit_uses_an_explicit_budget(self) -> None:
        assert effective_chunk_limit(2048, 25_000_000) == 156
        assert effective_chunk_limit(2, 1) == 1

    @pytest.mark.parametrize("budget", [0, -1, True, 1.0, "1"])
    def test_rejects_invalid_budget(self, budget: object) -> None:
        with pytest.raises(ValueError, match=r"scalar comparison budget must be a positive integer"):
            effective_chunk_limit(2048, budget)  # type: ignore[arg-type]


class TestIntraSkillValidatorValidate:
    def test_collection_log_uses_target_label_without_host_path(self, tmp_path: Path, caplog) -> None:
        skill_dir = tmp_path / "nested" / "external-skill"
        skill_dir.mkdir(parents=True)

        with caplog.at_level(logging.INFO, logger=intra_skill_validator.__name__):
            result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is True
        assert str(skill_dir) not in caplog.text
        assert skill_dir.name in caplog.text

    def test_empty_directory_passes(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "empty-skill"
        skill_dir.mkdir()
        v = IntraSkillValidator()
        result = v.validate(skill_dir)
        assert result.passed is True

    def test_single_file_not_enough_to_compare(self, tmp_path: Path) -> None:
        """A skill with only one very short file should pass — not enough chunks."""
        skill_dir = tmp_path / "tiny-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Title\nShort.")
        v = IntraSkillValidator()
        result = v.validate(skill_dir)
        assert result.passed is True

    def test_unsafe_collected_path_fails_closed_with_actionable_finding(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("private host content")
        link = skill_dir / "references" / "outside.md"
        link.parent.mkdir()
        link.symlink_to(outside)

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is False
        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.severity == Severity.CRITICAL
        assert finding.check_name == "unsafe_path"
        assert finding.file_path == "references/outside.md"
        assert finding.suggestion is not None
        assert "replace" in finding.suggestion.lower()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_chunk_limit_fails_before_embedding(self, mock_embed, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(intra_skill_validator, "CONTENT_DEDUP_MAX_CHUNKS", 1)
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is False
        assert result.findings[0].check_name == "chunk_count_limit"
        assert result.findings[0].metadata == {"actual": 2, "limit": 1}
        mock_embed.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_scalar_work_limit_fails_before_clustering(
        self,
        mock_embed,
        mock_build_clusters,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr(intra_skill_validator, "CONTENT_DEDUP_MAX_SCALAR_COMPARISONS", 1)
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]
        mock_build_clusters.return_value = []

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is False
        finding = result.findings[0]
        assert finding.check_name == "scalar_comparison_limit"
        assert finding.file_path == skill_dir.name
        assert finding.metadata == {
            "pair_count": 1,
            "vector_dimension": 2,
            "scalar_work": 2,
            "limit": 1,
            "effective_chunk_limit": 1,
        }
        mock_build_clusters.assert_not_called()

    @pytest.mark.parametrize("vector_dimension", [1024, 1536, 2048])
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_valid_363_chunk_skill_fits_scalar_budget(
        self,
        mock_embed,
        mock_build_clusters,
        vector_dimension: int,
        tmp_path: Path,
    ) -> None:
        """A skill well under the chunk ceiling must not trip a lower scalar-work limit."""
        skill_dir = tmp_path / "valid-large-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_numbered_sections(363))
        mock_embed.return_value.embed.side_effect = _unit_embeddings(vector_dimension)
        mock_build_clusters.return_value = []

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is True
        assert result.findings == []
        assert mock_embed.return_value.embed.call_count == 6
        assert len(mock_build_clusters.call_args.args[0]) == 363

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_chunk_ceiling_is_reachable_with_2048_dimensional_embeddings(
        self, mock_embed, mock_build_clusters, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / "boundary-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_numbered_sections(512))
        mock_embed.return_value.embed.side_effect = _unit_embeddings(2048)
        mock_build_clusters.return_value = []

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is True
        assert result.findings == []
        assert len(mock_build_clusters.call_args.args[0]) == 512

    @pytest.mark.parametrize(("vector_dimension", "chunk_limit"), [(3072, 418), (4096, 362)])
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_wider_embeddings_report_their_effective_chunk_limit(
        self,
        mock_embed,
        mock_build_clusters,
        vector_dimension: int,
        chunk_limit: int,
        tmp_path: Path,
    ) -> None:
        skill_dir = tmp_path / "wide-embedding-skill"
        skill_dir.mkdir()
        mock_embed.return_value.embed.side_effect = _unit_embeddings(vector_dimension)
        mock_build_clusters.return_value = []

        (skill_dir / "SKILL.md").write_text(_numbered_sections(chunk_limit))
        assert IntraSkillValidator().validate(skill_dir).passed is True

        chunk_count = chunk_limit + 1
        pair_count = chunk_count * (chunk_count - 1) // 2
        (skill_dir / "SKILL.md").write_text(_numbered_sections(chunk_count))
        mock_build_clusters.reset_mock()
        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is False
        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.severity == Severity.CRITICAL
        assert finding.check_name == "scalar_comparison_limit"
        assert finding.file_path == skill_dir.name
        assert finding.metadata == {
            "pair_count": pair_count,
            "vector_dimension": vector_dimension,
            "scalar_work": pair_count * vector_dimension,
            "limit": 267_911_168,
            "effective_chunk_limit": chunk_limit,
        }
        assert f"effective chunk limit is {chunk_limit} for {vector_dimension}-dimensional" in finding.message
        mock_build_clusters.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_explicit_budget_reports_its_own_effective_chunk_limit(
        self, mock_embed, mock_build_clusters, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / "bounded-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_numbered_sections(157))
        mock_embed.return_value.embed.side_effect = _unit_embeddings(2048)

        result = IntraSkillValidator(_max_scalar_comparisons=25_000_000).validate(skill_dir)

        assert result.passed is False
        finding = result.findings[0]
        assert finding.check_name == "scalar_comparison_limit"
        assert finding.metadata["limit"] == 25_000_000
        assert finding.metadata["effective_chunk_limit"] == 156
        assert "effective chunk limit is 156 for 2048-dimensional" in finding.message
        assert mock_embed.return_value.embed.call_count == 1
        mock_build_clusters.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_absolute_chunk_ceiling_fails_before_embedding(self, mock_embed, tmp_path: Path) -> None:
        skill_dir = tmp_path / "many-headings"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_numbered_sections(513))

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is False
        assert result.findings[0].check_name == "chunk_count_limit"
        assert result.findings[0].metadata == {"actual": 513, "limit": 512}
        mock_embed.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_llm_cluster_limit_fails_before_llm_analysis(
        self, mock_embed, mock_build_clusters, mock_llm, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(intra_skill_validator, "CONTENT_DEDUP_MAX_LLM_CLUSTERS", 1)
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        members = [
            ContentChunk("SKILL.md", "## A", 1, 2, "a" * 100, "markdown"),
            ContentChunk("SKILL.md", "## B", 3, 4, "b" * 100, "markdown"),
        ]
        cluster = ContentCluster(members, 1.0, 1.0, False, {"markdown"})
        mock_build_clusters.return_value = [cluster, cluster]

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is False
        assert result.findings[0].check_name == "llm_cluster_count_limit"
        assert result.findings[0].metadata == {"actual": 2, "limit": 1}
        assert result.findings[0].file_path == skill_dir.name
        mock_llm.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_embedding_error_is_incomplete_without_skill_finding(self, mock_embed, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.side_effect = SimilarityConfigError("No API key")

        v = IntraSkillValidator()
        result = v.validate(skill_dir)
        assert result.passed is False
        assert result.status == "incomplete"
        assert result.incomplete_scans == ["embedding-provider"]
        assert result.findings == []
        assert "No API key" in result.errors[0]

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_nonfinite_provider_vector_fails_closed_before_clustering(
        self,
        mock_embed,
        mock_llm,
        tmp_path: Path,
    ) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.return_value = [[math.nan, 0.0], [math.nan, 0.0]]

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is False
        assert result.status == "incomplete"
        assert result.findings == []
        assert "Embedding provider error" in result.errors[0]
        mock_llm.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_no_clusters_above_threshold_passes(self, mock_embed, _mock_llm, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)

        # Return orthogonal embeddings → no clusters
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]

        v = IntraSkillValidator()
        result = v.validate(skill_dir)
        assert result.passed is True

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_duplicate_verdict_produces_finding(self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)

        # Return identical embeddings → cluster formed
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]

        mock_analyze.return_value = LLMVerdict(
            verdict="DUPLICATE", confidence=0.9, reasoning="Same content", suggestion="Remove one"
        )

        v = IntraSkillValidator()
        result = v.validate(skill_dir)
        assert result.passed is False
        assert result.status == "failed"
        assert len(result.findings) >= 1
        assert result.findings[0].category == "DUPLICATE"

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_short_config_intra_file_duplicate_downgraded_to_low(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        """Repeated short config snippets inside one file are LOW, not a HIGH blocker.

        Regression test for the report that a legitimately-repeated config
        block (the same ``pts-tolerance``/``sync`` settings recurring inside a
        single reference file) produced a HIGH ``DUPLICATE`` finding that failed
        the skill. It should be capped at LOW and not fail validation.
        """
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        config_block = "pts-tolerance = 60000\nsync = true\nqos = false\nmax-latency = 100000\nbuffer-size = 4096\n"
        (skill_dir / "metamux_config.md").write_text(f"## Pipeline A\n{config_block}\n## Pipeline B\n{config_block}")

        # Identical embeddings → single-file cluster of the two config blocks.
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_analyze.return_value = LLMVerdict(
            verdict="DUPLICATE",
            confidence=0.9,  # would normally map to HIGH
            reasoning="Same config block repeated",
            suggestion="Consolidate config",
        )

        v = IntraSkillValidator()
        result = v.validate(skill_dir)

        dup_findings = [f for f in result.findings if f.category == "DUPLICATE"]
        assert len(dup_findings) == 1
        assert dup_findings[0].severity == Severity.LOW
        # LOW is advisory only — it must not fail the skill.
        assert result.passed is True

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_generated_skill_card_duplicate_does_not_produce_finding(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        duplicate_text = "Helps developers build and debug cuOpt integrations. " * 5
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: cuopt-developer\ndescription: {duplicate_text}\n---\n## Instructions\n{duplicate_text}\n"
        )
        (skill_dir / "skill-card.md").write_text(
            f"## Description:\n{duplicate_text}\n\n## Use Case:\n{duplicate_text}\n"
        )

        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        mock_analyze.return_value = LLMVerdict(
            verdict="DUPLICATE",
            confidence=0.9,
            reasoning="Generated card mirrors manifest content",
            suggestion="Do not edit generated card",
        )

        v = IntraSkillValidator()
        result = v.validate(skill_dir)

        assert result.passed is True
        assert result.findings == []

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_generated_benchmark_report_duplicate_does_not_produce_finding(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        duplicate_text = "Helps developers build and debug cuOpt integrations. " * 5
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: cuopt-developer\ndescription: {duplicate_text}\n---\n## Instructions\n{duplicate_text}\n"
        )
        (skill_dir / "BENCHMARK.md").write_text(
            "# Evaluation Report\n\n"
            "This benchmark summarizes validation and Tier 3 live agent results.\n\n"
            f"{duplicate_text}\n"
        )

        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_analyze.return_value = LLMVerdict(
            verdict="DUPLICATE",
            confidence=0.9,
            reasoning="Generated benchmark mirrors skill content",
            suggestion="Do not edit generated benchmark",
        )

        v = IntraSkillValidator()
        result = v.validate(skill_dir)

        assert result.passed is True
        assert result.findings == []

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_intentional_detail_produces_no_finding(self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)

        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]

        mock_analyze.return_value = LLMVerdict(
            verdict="INTENTIONAL_DETAIL",
            confidence=0.85,
            reasoning="Summary vs detail",
            suggestion="Keep both",
        )

        v = IntraSkillValidator()
        result = v.validate(skill_dir)
        # INTENTIONAL_DETAIL is skipped — no findings added
        assert len(result.findings) == 0

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_llm_error_is_incomplete_without_skill_finding(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path, caplog
    ) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)

        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]

        mock_analyze.side_effect = LLMClientError("API failed: private provider data")

        v = IntraSkillValidator()
        result = v.validate(skill_dir)
        assert result.passed is False
        assert result.status == "incomplete"
        assert result.incomplete_scans == ["deduplication-llm"]
        assert result.findings == []
        assert "LLM" in result.errors[0]
        assert "private provider data" not in str(result.errors)
        assert "private provider data" not in caplog.text

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_partial_llm_failure_preserves_valid_duplicate_finding(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("\n".join(f"## Section {letter}\n{letter.lower() * 200}" for letter in "ABCD"))
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

        def analyze(_client, cluster):
            if any("Section A" in member.heading for member in cluster.members):
                return LLMVerdict("DUPLICATE", 0.9, "Same content", "Remove one")
            raise RuntimeError("private provider response")

        mock_analyze.side_effect = analyze
        result = IntraSkillValidator().validate(skill)

        assert result.status == "incomplete"
        assert result.incomplete_scans == ["deduplication-llm"]
        assert len(result.findings) == 1
        assert result.findings[0].category == "DUPLICATE"
        assert result.findings[0].severity == Severity.HIGH
        assert result.summary.critical_count == 0
        assert result.summary.high_count == 1
        assert result.metadata["llm_analysis"]["clusters_completed"] == 1
        assert result.metadata["llm_analysis"]["clusters_failed"] == 1
        assert "1 of 2 content clusters" in result.errors[0]

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_llm_initialization_error_preserves_collection_evidence(self, mock_embed, mock_llm, tmp_path: Path) -> None:
        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_llm.side_effect = LLMClientError("missing credential")

        result = IntraSkillValidator().validate(skill)

        assert result.status == "incomplete"
        assert result.findings == []
        assert {detail.check_name for detail in result.success_details} == {"file_collection", "chunking"}

    def test_custom_threshold(self) -> None:
        v = IntraSkillValidator(threshold=0.95)
        assert v._threshold == 0.95

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_distinct_cluster_errors_are_grouped_in_deterministic_order(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "\n".join(f"## Section {letter}\n{letter.lower() * 200}" for letter in "ABCDEF")
        )
        mock_embed.return_value.embed.return_value = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]

        class ServiceError(Exception):
            def __init__(self, status):
                self.status_code = status
                super().__init__("private provider response")

        def analyze(_client, cluster):
            raise ServiceError(429 if any("Section A" in member.heading for member in cluster.members) else 400)

        mock_analyze.side_effect = analyze
        result = IntraSkillValidator().validate(skill)

        assert result.status == "incomplete"
        assert len(result.errors) == 1
        assert "3 of 3 content clusters" in result.errors[0]
        failures = result.metadata["llm_analysis"]["failures"]
        assert [failure["count"] for failure in failures] == [2, 1]
        assert failures[0]["diagnostic"].startswith("HTTP 400")
        assert failures[1]["diagnostic"].startswith("HTTP 429")
        assert result.errors[0].index("HTTP 400") < result.errors[0].index("HTTP 429")
        assert "private provider response" not in result.errors[0]

    def test_custom_models(self) -> None:
        v = IntraSkillValidator(embedding_model="custom/embed", llm_model="custom/llm")
        assert v._embedding_model == "custom/embed"
        assert v._llm_model == "custom/llm"
