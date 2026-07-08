# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.intra_skill.intra_skill_validator."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from unittest.mock import patch

from skillevaluator.deduplication.intra_skill import intra_skill_validator
from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator
from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster
from skillevaluator.deduplication.utils.chunker import ContentChunk
from skillevaluator.embedding.client import SimilarityConfigError
from skillevaluator.inference import LLMClientError, LLMVerdict
from skillevaluator.models.result import Severity


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

        assert resolve_embedding_provider().model == "nvidia/nv-embed-v1"


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
        monkeypatch.setattr(intra_skill_validator, "CONTENT_DEDUP_MAX_SCALAR_COMPARISONS", 1, raising=False)
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
        }
        mock_build_clusters.assert_not_called()

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
    def test_embedding_error_produces_critical_finding(self, mock_embed, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.side_effect = SimilarityConfigError("No API key")

        v = IntraSkillValidator()
        result = v.validate(skill_dir)
        assert result.passed is False
        assert any(f.severity == Severity.CRITICAL for f in result.findings)
        assert all(f.file_path == skill_dir.name for f in result.findings)

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
        assert result.findings[0].check_name == "embedding_error"
        assert result.findings[0].file_path == skill_dir.name
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
    def test_llm_error_produces_critical_finding(self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)

        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]

        mock_analyze.side_effect = LLMClientError("API failed")

        v = IntraSkillValidator()
        result = v.validate(skill_dir)
        assert result.passed is False
        assert any(f.severity == Severity.CRITICAL for f in result.findings)
        assert any("LLM" in f.message for f in result.findings)
        assert all(f.file_path == skill_dir.name for f in result.findings)

    def test_custom_threshold(self) -> None:
        v = IntraSkillValidator(threshold=0.95)
        assert v._threshold == 0.95

    def test_custom_models(self) -> None:
        v = IntraSkillValidator(embedding_model="custom/embed", llm_model="custom/llm")
        assert v._embedding_model == "custom/embed"
        assert v._llm_model == "custom/llm"
