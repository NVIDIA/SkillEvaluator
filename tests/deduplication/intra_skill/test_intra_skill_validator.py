# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.intra_skill.intra_skill_validator."""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from skillevaluator import constants
from skillevaluator.deduplication.intra_skill import intra_skill_validator, llm_analyzer
from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator, effective_chunk_limit
from skillevaluator.deduplication.intra_skill.llm_analyzer import SYSTEM_PROMPT, build_user_prompt
from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster
from skillevaluator.deduplication.utils.chunker import ContentChunk
from skillevaluator.embedding.client import MAX_EMBEDDING_VECTOR_DIMENSION, SimilarityConfigError
from skillevaluator.inference import LLMClient, LLMClientError, LLMClientRefusedError, LLMVerdict
from skillevaluator.models.result import Severity
from skillevaluator.reporting.json_reporter import JSONReporter


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
    def test_scalar_budget_lets_1024_dimensional_embeddings_reach_the_chunk_ceiling(self) -> None:
        assert constants.CONTENT_DEDUP_MAX_CHUNKS == 1024
        assert constants.CONTENT_DEDUP_MAX_PAIR_COMPARISONS == 1024 * 1023 // 2
        assert constants.CONTENT_DEDUP_SCALAR_BUDGET_DIMENSION == 1024
        assert constants.CONTENT_DEDUP_MAX_SCALAR_COMPARISONS == 536_346_624
        assert (
            constants.CONTENT_DEDUP_MAX_SCALAR_COMPARISONS
            == constants.CONTENT_DEDUP_MAX_PAIR_COMPARISONS * constants.CONTENT_DEDUP_SCALAR_BUDGET_DIMENSION
        )

    @pytest.mark.parametrize(
        ("vector_dimension", "expected_limit"),
        [
            (1, 1024),
            (768, 1024),
            (1024, 1024),
            (1025, 1023),
            (1536, 836),
            (2048, 724),
            (3072, 591),
            (4096, 512),
        ],
    )
    def test_limit_depends_on_vector_dimension(self, vector_dimension: int, expected_limit: int) -> None:
        assert effective_chunk_limit(vector_dimension) == expected_limit

    @pytest.mark.parametrize(
        "environ",
        [
            pytest.param({"SKILL_EVAL_EMBEDDING_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}, id="openai"),
            pytest.param({"SKILL_EVAL_EMBEDDING_PROVIDER": "nv_build", "NVIDIA_API_KEY": "nvapi-test"}, id="nv-build"),
            pytest.param(
                {
                    "SKILL_EVAL_EMBEDDING_PROVIDER": "openai-compatible",
                    "SKILL_EVAL_EMBEDDING_API_KEY": "test-key",
                    "SKILL_EVAL_EMBEDDING_BASE_URL": "https://gateway.example/v1",
                },
                id="openai-compatible",
            ),
        ],
    )
    def test_docs_give_the_chunk_limit_of_each_default_embedding_model(self, environ: dict[str, str]) -> None:
        """The limits table names the limit a user of a default model gets, not only the ceiling."""
        from skillevaluator.provider_config import resolve_embedding_provider

        # Native output width of each default model; the embedding client requests no other width.
        default_model_widths = {
            "text-embedding-3-small": 1536,
            "nvidia/nemotron-3-embed-1b": 2048,
            "nvidia/nvidia/nemotron-3-embed-1b": 2048,
        }
        model = resolve_embedding_provider(environ).model
        vector_dimension = default_model_widths[model]
        guide = Path(__file__).resolve().parents[3] / "docs" / "tier2-deduplication.mdx"
        (row,) = [
            line
            for line in guide.read_text(encoding="utf-8").splitlines()
            if line.startswith("| Intra-skill comparison work |")
        ]

        (width_entry,) = [entry for entry in row.split(", ") if f"`{model}`" in entry]
        assert width_entry.startswith(f"{effective_chunk_limit(vector_dimension):,} ")
        assert f" at {vector_dimension:,} (" in width_entry

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
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_duplicate_shell_comments_at_eof_are_compared(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        comments = (
            "# Document the required setup before invoking this helper script.\n"
            "# Use the configured workspace for every generated output file.\n"
            "# Keep the source input unchanged while the helper processes it.\n"
        )
        (tmp_path / "first.sh").write_text("#!/bin/sh\necho ready\n" + comments, encoding="utf-8")
        (tmp_path / "second.sh").write_text("#!/bin/sh\necho ready\n" + comments, encoding="utf-8")
        mock_embed.return_value.embed.side_effect = _unit_embeddings(2)
        mock_analyze.return_value = LLMVerdict(
            verdict="DUPLICATE", confidence=0.9, reasoning="Repeated instructions", suggestion="Consolidate comments"
        )

        result = IntraSkillValidator().validate(tmp_path)

        mock_embed.return_value.embed.assert_called_once()
        assert len(mock_embed.return_value.embed.call_args.args[0]) == 2
        mock_analyze.assert_called_once()
        assert result.passed is False
        assert len(result.findings) == 1
        assert "first.sh" in result.findings[0].message
        assert "second.sh" in result.findings[0].message

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
    def test_frontmatter_merge_chain_fails_closed_before_embedding(self, mock_embed, tmp_path: Path) -> None:
        # Each level merges the previous mapping ten times, so unbounded YAML
        # builds 10**(level + 1) key-value pairs, about 180 MB at six levels.
        # Four levels keep this test cheap if the bound ever regresses.
        lines = ["m0: &m0 {" + ", ".join(f"k{index}: {index}" for index in range(10)) + "}"]
        lines += [f"m{level}: &m{level} {{<<: [{', '.join([f'*m{level - 1}'] * 10)}]}}" for level in range(1, 5)]
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: skill\n" + "\n".join(lines) + "\n---\n" + _numbered_sections(3), encoding="utf-8"
        )

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is False
        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.severity == Severity.CRITICAL
        assert finding.check_name == "manifest_complexity_limit"
        assert finding.file_path == "SKILL.md"
        mock_embed.assert_not_called()

    @staticmethod
    def _assert_work_limit_incomplete(result, work_limit: dict) -> None:
        """A tripped work limit is missing coverage, never a duplicate-content finding."""
        assert result.passed is False
        assert result.status == "incomplete"
        assert result.incomplete_scans == ["context-deduplication"]
        assert result.findings == []
        assert result.summary.critical_count == 0
        assert result.metadata["work_limit"] == work_limit
        assert len(result.errors) == 1
        assert result.errors[0].startswith("Tier 2 scan incomplete: ")

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_chunk_limit_is_incomplete_before_embedding(self, mock_embed, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(intra_skill_validator, "CONTENT_DEDUP_MAX_CHUNKS", 1)
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]

        result = IntraSkillValidator().validate(skill_dir)

        self._assert_work_limit_incomplete(
            result,
            {"check_name": "chunk_count_limit", "file_path": "SKILL.md", "actual": 2, "limit": 1},
        )
        assert "more than 1 content chunks" in result.errors[0]
        assert "reached in SKILL.md" in result.errors[0]
        assert "Reduce or split the skill content" in result.errors[0]
        mock_embed.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_scalar_work_limit_is_incomplete_before_clustering(
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

        self._assert_work_limit_incomplete(
            result,
            {
                "check_name": "scalar_comparison_limit",
                "pair_count": 1,
                "vector_dimension": 2,
                "scalar_work": 2,
                "limit": 1,
                "effective_chunk_limit": 1,
                "chunk_count": 2,
            },
        )
        assert "scalar comparison work of 2 exceeds the limit of 1" in result.errors[0]
        assert "effective chunk limit is 1 for 2-dimensional embeddings; this skill has 2 chunks." in result.errors[0]
        assert str(tmp_path) not in result.errors[0]
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
    def test_1024_chunks_complete_in_sixteen_embedding_batches(
        self, mock_embed, mock_build_clusters, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / "boundary-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_numbered_sections(1024), encoding="utf-8")
        mock_embed.return_value.embed.side_effect = _unit_embeddings(1024)
        mock_build_clusters.return_value = []

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is True
        assert result.status == "passed"
        assert result.findings == []
        assert [len(call.args[0]) for call in mock_embed.return_value.embed.call_args_list] == [64] * 16
        assert len(mock_build_clusters.call_args.args[0]) == 1024

    @pytest.mark.parametrize(("vector_dimension", "chunk_limit"), [(1536, 836), (2048, 724), (3072, 591), (4096, 512)])
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

        (skill_dir / "SKILL.md").write_text(_numbered_sections(chunk_limit), encoding="utf-8")
        assert IntraSkillValidator().validate(skill_dir).passed is True
        assert len(mock_build_clusters.call_args.args[0]) == chunk_limit

        chunk_count = chunk_limit + 1
        pair_count = chunk_count * (chunk_count - 1) // 2
        (skill_dir / "SKILL.md").write_text(_numbered_sections(chunk_count), encoding="utf-8")
        mock_build_clusters.reset_mock()
        mock_embed.return_value.embed.reset_mock()
        result = IntraSkillValidator().validate(skill_dir)

        self._assert_work_limit_incomplete(
            result,
            {
                "check_name": "scalar_comparison_limit",
                "pair_count": pair_count,
                "vector_dimension": vector_dimension,
                "scalar_work": pair_count * vector_dimension,
                "limit": 536_346_624,
                "effective_chunk_limit": chunk_limit,
                "chunk_count": chunk_count,
            },
        )
        assert (
            f"effective chunk limit is {chunk_limit} for {vector_dimension}-dimensional embeddings; "
            f"this skill has {chunk_count} chunks."
        ) in result.errors[0]
        # The first batch sets the width; no further batch is requested.
        assert mock_embed.return_value.embed.call_count == 1
        mock_build_clusters.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_2048_dimensional_skill_at_the_chunk_ceiling_stops_after_the_first_batch(
        self, mock_embed, mock_build_clusters, tmp_path: Path
    ) -> None:
        """Between the width's limit and the ceiling, the scan stops on comparison work, not chunk count."""
        skill_dir = tmp_path / "default-width-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_numbered_sections(1024), encoding="utf-8")
        mock_embed.return_value.embed.side_effect = _unit_embeddings(2048)

        result = IntraSkillValidator().validate(skill_dir)

        pair_count = 1024 * 1023 // 2
        self._assert_work_limit_incomplete(
            result,
            {
                "check_name": "scalar_comparison_limit",
                "pair_count": pair_count,
                "vector_dimension": 2048,
                "scalar_work": pair_count * 2048,
                "limit": 536_346_624,
                "effective_chunk_limit": 724,
                "chunk_count": 1024,
            },
        )
        assert [len(call.args[0]) for call in mock_embed.return_value.embed.call_args_list] == [64]
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
        assert result.status == "incomplete"
        assert result.findings == []
        work_limit = result.metadata["work_limit"]
        assert work_limit["check_name"] == "scalar_comparison_limit"
        assert work_limit["limit"] == 25_000_000
        assert work_limit["effective_chunk_limit"] == 156
        assert work_limit["chunk_count"] == 157
        assert (
            "effective chunk limit is 156 for 2048-dimensional embeddings; this skill has 157 chunks."
            in result.errors[0]
        )
        assert mock_embed.return_value.embed.call_count == 1
        mock_build_clusters.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_absolute_chunk_ceiling_fails_before_embedding(self, mock_embed, tmp_path: Path) -> None:
        skill_dir = tmp_path / "many-headings"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_numbered_sections(1025), encoding="utf-8")

        result = IntraSkillValidator().validate(skill_dir)

        self._assert_work_limit_incomplete(
            result,
            {"check_name": "chunk_count_limit", "file_path": "SKILL.md", "actual": 1025, "limit": 1024},
        )
        assert "more than 1024 content chunks" in result.errors[0]
        mock_embed.assert_not_called()

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_llm_cluster_limit_is_incomplete_before_llm_analysis(
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

        self._assert_work_limit_incomplete(
            result,
            {"check_name": "llm_cluster_count_limit", "actual": 2, "limit": 1},
        )
        assert "2 content clusters require LLM review, more than the limit of 1" in result.errors[0]
        mock_llm.assert_not_called()

    @pytest.mark.parametrize(
        ("limit_name", "check_name", "expected_message"),
        [
            (
                "intra_skill_validator.CONTENT_DEDUP_MAX_CLUSTER_MEMBERS",
                "llm_cluster_member_limit",
                "a content cluster has 2 members, more than the limit of 1 for LLM review.",
            ),
            (
                "llm_analyzer.MAX_PROMPT_CHARS",
                "llm_prompt_size_limit",
                "an LLM review prompt of {actual} characters exceeds the limit of 1.",
            ),
            (
                "intra_skill_validator.CONTENT_DEDUP_MAX_TOTAL_LLM_PROMPT_CHARS",
                "llm_total_prompt_size_limit",
                "LLM review prompts total {actual} characters, more than the limit of 1.",
            ),
        ],
    )
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_llm_review_limits_are_incomplete_before_llm_analysis(
        self,
        mock_embed,
        mock_build_clusters,
        mock_llm,
        limit_name: str,
        check_name: str,
        expected_message: str,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr(f"skillevaluator.deduplication.intra_skill.{limit_name}", 1)
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "a" * 200)
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        members = [
            ContentChunk("SKILL.md", "## A", 1, 2, "a" * 100, "markdown"),
            ContentChunk("SKILL.md", "## B", 3, 4, "a" * 100, "markdown"),
        ]
        cluster = ContentCluster(members, 1.0, 1.0, False, {"markdown"})
        mock_build_clusters.return_value = [cluster]
        actual = 2 if check_name == "llm_cluster_member_limit" else len(SYSTEM_PROMPT) + len(build_user_prompt(cluster))

        result = IntraSkillValidator().validate(skill_dir)

        self._assert_work_limit_incomplete(result, {"check_name": check_name, "actual": actual, "limit": 1})
        assert expected_message.format(actual=actual) in result.errors[0]
        assert "Reduce duplicated content or split the skill" in result.errors[0]
        assert str(tmp_path) not in result.errors[0]
        mock_llm.assert_not_called()

    def test_cluster_member_limit_is_the_chunk_ceiling(self) -> None:
        assert intra_skill_validator.CONTENT_DEDUP_MAX_CLUSTER_MEMBERS == constants.CONTENT_DEDUP_MAX_CHUNKS == 1024
        assert llm_analyzer.MAX_PROMPT_CHARS == 64_000
        # The validator reads llm_analyzer.MAX_PROMPT_CHARS; a stale copy here
        # would let a patched budget move one threshold and not the other.
        assert not hasattr(intra_skill_validator, "CONTENT_DEDUP_MAX_LLM_PROMPT_CHARS")
        assert intra_skill_validator.CONTENT_DEDUP_MAX_TOTAL_LLM_PROMPT_CHARS == 1_000_000

    @pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
    @pytest.mark.parametrize(
        ("headings", "body"),
        [
            pytest.param(["## Repeated"] * 500, "Describe the supported workflow. " * 4, id="500-same-heading"),
            pytest.param(
                [f"## Step {index}" for index in range(500)],
                "Describe the supported workflow. " * 4,
                id="500-distinct-headings",
            ),
            pytest.param(
                [f"## Tool {index}" for index in range(30)],
                "\n\n".join(["Boilerplate paragraph about safety and usage. " * 20] * 3),
                id="30-distinct-headings-long-body",
            ),
        ],
    )
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_repeated_section_bodies_complete_through_real_chunking_and_clustering(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path, headings: list[str], body: str, newline: str
    ) -> None:
        skill_dir = tmp_path / "repeated-sections"
        skill_dir.mkdir()
        markdown = "".join(f"{heading}\n{body}\n" for heading in headings)
        (skill_dir / "SKILL.md").write_bytes(markdown.replace("\n", newline).encode("utf-8"))
        mock_embed.return_value.embed.side_effect = _unit_embeddings(2)
        mock_analyze.return_value = LLMVerdict(
            verdict="DUPLICATE", confidence=0.9, reasoning="Repeated instructions", suggestion="Consolidate sections"
        )

        result = IntraSkillValidator().validate(skill_dir)

        assert result.status == "failed"
        assert result.incomplete_scans == []
        assert "work_limit" not in result.metadata
        mock_analyze.assert_called_once()
        cluster = mock_analyze.call_args.args[1]
        assert len(cluster.members) == len(headings)
        assert [member.heading for member in cluster.members] == headings
        assert all(len(member.text) >= constants.CONTENT_DEDUP_MIN_CHUNK_CHARS for member in cluster.members)
        prompt = build_user_prompt(cluster)
        assert len(SYSTEM_PROMPT) + len(prompt) <= llm_analyzer.MAX_PROMPT_CHARS
        assert prompt.count("---CHUNK ") == len(headings)
        # Each section's text is its own heading line plus the shared body.
        shared_body = body.strip().replace("\n", newline)
        assert prompt.count("---SHARED CONTENT ") == 1
        assert prompt.count(shared_body) == 1
        for member in cluster.members:
            assert member.text == f"{member.heading}{newline}{shared_body}"
            assert f"Content:\n{member.heading}{newline}[SHARED CONTENT 1]" in prompt
        [finding] = result.findings
        assert finding.severity == Severity.HIGH
        assert finding.message.startswith("Duplicate content found within SKILL.md:\n")
        assert finding.message.count(" in SKILL.md (lines ") == min(len(headings), 64)
        if len(headings) > 64:
            assert finding.message.endswith(f"\n  and {len(headings) - 64} more locations ({len(headings)} in total)")

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_short_repeated_bodies_under_long_headings_stay_inline_and_the_cluster_completes(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / "long-headings"
        skill_dir.mkdir()
        short_body = "See guide."
        long_body = "Boilerplate paragraph about safety and usage. " * 20
        headings = [
            f"## Configuration option for the workflow runner, number {index:04d} of the set" for index in range(270)
        ]
        headings += [f"## Big {index}" for index in range(14)]
        bodies = [short_body] * 270 + [long_body] * 14
        markdown = "".join(f"{heading}\n{body}\n" for heading, body in zip(headings, bodies, strict=True))
        (skill_dir / "SKILL.md").write_bytes(markdown.encode("utf-8"))
        mock_embed.return_value.embed.side_effect = _unit_embeddings(2)
        mock_analyze.return_value = LLMVerdict(
            verdict="DUPLICATE", confidence=0.9, reasoning="Repeated instructions", suggestion="Consolidate sections"
        )

        result = IntraSkillValidator().validate(skill_dir)

        assert result.incomplete_scans == []
        assert "work_limit" not in result.metadata
        mock_analyze.assert_called_once()
        cluster = mock_analyze.call_args.args[1]
        assert [member.heading for member in cluster.members] == headings
        budget = llm_analyzer.MAX_PROMPT_CHARS
        shared_body = long_body.strip()
        # Sent in full, or with both repeated bodies shared, the request is over
        # budget: a "[SHARED CONTENT n]" reference is longer than "See guide.".
        assert len(SYSTEM_PROMPT) + len(llm_analyzer._render_user_prompt(cluster)) > budget
        all_shared = llm_analyzer._render_user_prompt(cluster, {short_body: 1, shared_body: 2})
        assert all_shared.count("[SHARED CONTENT 1]") == 270
        assert len(SYSTEM_PROMPT) + len(all_shared) > budget
        prompt = build_user_prompt(cluster)
        assert len(SYSTEM_PROMPT) + len(prompt) <= budget
        assert prompt.count("---SHARED CONTENT ") == 1
        assert prompt.count(shared_body) == 1
        assert prompt.count("[SHARED CONTENT 1]") == 14
        for member in cluster.members[:270]:
            assert member.text == f"{member.heading}\n{short_body}"
            assert f"Content:\n{member.text}\n\n" in prompt
        [finding] = result.findings
        assert finding.severity == Severity.HIGH

    @pytest.mark.parametrize("verdict", ["DUPLICATE", "RELATED_BUT_DISTINCT"])
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_large_member_cluster_is_reviewed_in_full_when_prompt_fits(
        self, mock_embed, mock_clusters, _mock_llm, mock_analyze, tmp_path: Path, verdict: str
    ) -> None:
        skill_dir = tmp_path / "large-cluster"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## A\n" + "a" * 100 + "\n## B\n" + "b" * 100, encoding="utf-8")
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        members = [
            ContentChunk("SKILL.md", f"Member {index}", index + 1, index + 1, "content " * 10, "markdown")
            for index in range(500)
        ]
        cluster = ContentCluster(members, 1.0, 1.0, False, {"markdown"})
        mock_clusters.return_value = [cluster]
        mock_analyze.return_value = LLMVerdict(
            verdict=verdict, confidence=0.9, reasoning="Reviewed all members", suggestion="Consolidate duplicates"
        )

        result = IntraSkillValidator().validate(skill_dir)

        assert result.incomplete_scans == []
        assert "work_limit" not in result.metadata
        mock_analyze.assert_called_once()
        assert mock_analyze.call_args.args[1] is cluster
        prompt = build_user_prompt(cluster)
        assert len(SYSTEM_PROMPT) + len(prompt) <= llm_analyzer.MAX_PROMPT_CHARS
        assert prompt.count("---CHUNK ") == 500
        for member in members:
            assert f"Section: {member.heading}\n" in prompt
        if verdict == "DUPLICATE":
            assert result.passed is False
            [finding] = result.findings
            assert finding.severity == Severity.HIGH
            assert '"Member 0" in SKILL.md (lines 1-1)' in finding.message
            assert '"Member 63" in SKILL.md (lines 64-64)' in finding.message
            assert '"Member 64"' not in finding.message
            assert finding.message.endswith("\n  and 436 more locations (500 in total)")
            assert finding.metadata["source_files"] == ["SKILL.md"]
        else:
            assert result.passed is True
            assert result.findings == []

    @pytest.mark.parametrize(
        ("member_count", "text", "check_name"),
        [
            pytest.param(1025, "content " * 10, "llm_cluster_member_limit", id="member-limit"),
            pytest.param(500, "x" * 80, "llm_prompt_size_limit", id="prompt-limit"),
        ],
    )
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_large_cluster_still_respects_member_and_prompt_limits(
        self,
        mock_embed,
        mock_clusters,
        mock_llm,
        tmp_path: Path,
        member_count: int,
        text: str,
        check_name: str,
    ) -> None:
        skill_dir = tmp_path / "cluster-budget-limit"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## A\n" + "a" * 100 + "\n## B\n" + "b" * 100, encoding="utf-8")
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        # Members differ in their index prefix, so distinct text is never shared.
        members = [
            ContentChunk("SKILL.md", f"Member {index}", index + 1, index + 1, f"{index}: {text}", "markdown")
            for index in range(member_count)
        ]
        cluster = ContentCluster(members, 1.0, 1.0, False, {"markdown"})
        mock_clusters.return_value = [cluster]
        if check_name == "llm_cluster_member_limit":
            actual, limit = member_count, 1024
        else:
            actual, limit = len(SYSTEM_PROMPT) + len(build_user_prompt(cluster)), 64_000
            assert "SHARED CONTENT" not in build_user_prompt(cluster)

        result = IntraSkillValidator().validate(skill_dir)

        assert actual > limit
        self._assert_work_limit_incomplete(result, {"check_name": check_name, "actual": actual, "limit": limit})
        mock_llm.assert_not_called()

    @pytest.mark.parametrize("repeated", [True, False], ids=["shared-fits", "distinct-refused"])
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_one_prompt_budget_sets_sharing_and_refusal(
        self, mock_embed, mock_clusters, mock_llm, mock_analyze, tmp_path: Path, monkeypatch, repeated: bool
    ) -> None:
        budget = 8_000
        monkeypatch.setattr(llm_analyzer, "MAX_PROMPT_CHARS", budget)
        skill_dir = tmp_path / "lowered-budget"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## A\n" + "a" * 100 + "\n## B\n" + "b" * 100, encoding="utf-8")
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        members = [
            ContentChunk(
                "SKILL.md",
                f"Member {index}",
                index + 1,
                index + 1,
                ("" if repeated else f"{index}: ") + "z" * 300,
                "markdown",
            )
            for index in range(40)
        ]
        cluster = ContentCluster(members, 1.0, 1.0, False, {"markdown"})
        mock_clusters.return_value = [cluster]
        mock_analyze.return_value = LLMVerdict("DUPLICATE", 0.9, "Same content", "Keep one copy")
        prompt = build_user_prompt(cluster)
        prompt_chars = len(SYSTEM_PROMPT) + len(prompt)

        result = IntraSkillValidator().validate(skill_dir)

        if repeated:
            # Shared under the lowered budget, then accepted by that same budget.
            assert "SHARED CONTENT" in prompt
            assert prompt_chars <= budget
            assert result.incomplete_scans == []
            assert "work_limit" not in result.metadata
            mock_analyze.assert_called_once()
        else:
            # Distinct text cannot be shared, so the lowered budget refuses it.
            assert "SHARED CONTENT" not in prompt
            assert prompt_chars > budget
            self._assert_work_limit_incomplete(
                result, {"check_name": "llm_prompt_size_limit", "actual": prompt_chars, "limit": budget}
            )
            mock_llm.assert_not_called()
            mock_analyze.assert_not_called()

    @pytest.mark.parametrize(
        ("member_count", "message_tail"),
        [
            pytest.param(64, '"Section 63" in file-063.md (lines 64-64)', id="64-members"),
            pytest.param(65, "\n  and 1 more location (65 in total)", id="65-members"),
            pytest.param(100, "\n  and 36 more locations (100 in total)", id="100-members"),
        ],
    )
    @pytest.mark.parametrize("refused", [False, True], ids=["duplicate", "refused"])
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.build_clusters")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_finding_message_names_at_most_64_locations_and_files(
        self,
        mock_embed,
        mock_clusters,
        _mock_llm,
        mock_analyze,
        tmp_path: Path,
        refused: bool,
        member_count: int,
        message_tail: str,
    ) -> None:
        skill_dir = tmp_path / "many-files"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## A\n" + "a" * 100 + "\n## B\n" + "b" * 100, encoding="utf-8")
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        members = [
            ContentChunk(f"file-{index:03}.md", f"Section {index}", index + 1, index + 1, "y" * 100, "markdown")
            for index in range(member_count)
        ]
        mock_clusters.return_value = [ContentCluster(members, 1.0, 1.0, True, {"markdown"})]
        if refused:
            mock_analyze.side_effect = LLMClientRefusedError("declined")
        else:
            mock_analyze.return_value = LLMVerdict("DUPLICATE", 0.9, "Same content", "Keep one copy")

        result = IntraSkillValidator().validate(skill_dir)

        [finding] = result.findings
        assert finding.check_name == ("llm_refused" if refused else "duplicate")
        assert finding.metadata["source_files"] == [member.source_file for member in members]
        scope, _, locations = finding.message.partition(":\n  ")
        named_files = scope.split("across ", 1)[1].split(" and ")
        assert named_files[:64] == [member.source_file for member in members[:64]]
        assert locations.count(" (lines ") == min(member_count, 64)
        assert locations.endswith(message_tail)
        if member_count > 64:
            omitted = member_count - 64
            assert scope.endswith(f" and {omitted} more file{'s' if omitted != 1 else ''}")
            assert '"Section 64"' not in locations
        else:
            assert "more file" not in scope
            assert "more location" not in locations

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_policy_override_cannot_waive_work_limit(self, mock_embed, tmp_path: Path, monkeypatch) -> None:
        """Severity overrides remap findings; they must not turn missing analysis into a pass."""
        from skillevaluator.validators.policy import ValidationPolicy, apply_policy

        monkeypatch.setattr(intra_skill_validator, "CONTENT_DEDUP_MAX_CHUNKS", 1)
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        policy = ValidationPolicy(
            severity_overrides={
                "CONTENT_DEDUP.*": Severity.LOW,
                "CONTENT_DEDUP.chunk_count_limit": Severity.LOW,
            }
        )

        (result,) = apply_policy([IntraSkillValidator().validate(skill_dir)], policy)

        assert result.passed is False
        assert result.status == "incomplete"
        assert result.findings == []
        assert result.errors[0].startswith("Tier 2 scan incomplete: ")
        mock_embed.assert_not_called()

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

    @pytest.mark.parametrize("confidence", ["NaN", "Infinity", 85, True])
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_out_of_range_confidence_is_incomplete_without_skill_finding(
        self, mock_embed, mock_llm, tmp_path: Path, confidence: object
    ) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200, encoding="utf-8"
        )
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_llm.return_value.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "confidence": confidence,
            "rationale": "Same content",
            "suggestion": "Remove one",
        }

        result = IntraSkillValidator().validate(skill_dir)

        assert mock_llm.return_value.extract_json_from_response.call_count == 1
        assert result.status == "incomplete"
        assert result.incomplete_scans == ["deduplication-llm"]
        assert result.findings == []
        assert result.metadata["llm_analysis"]["clusters_failed"] == 1
        assert "LLM configuration or response was invalid" in result.errors[0]
        # Strict JSON (no NaN or Infinity) still renders for the incomplete result.
        json.loads(JSONReporter().render(result))

    @pytest.mark.parametrize(
        "reply",
        [
            {"verdict": "INTENTIONAL_DETAIL", "confidence": 0.9, "rationale": "Summary vs detail", "suggestion": None},
            {"verdict": "RELATED_BUT_DISTINCT", "confidence": "0.9", "rationale": "Different", "suggestion": "Keep"},
            {"verdict": "DUPLICATE", "confidence": "0.9", "rationale": "Same content", "suggestion": "Remove one"},
        ],
        ids=["intentional-null-suggestion", "related-string-confidence", "duplicate-string-confidence"],
    )
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_malformed_reply_field_is_incomplete_whatever_the_verdict(
        self, mock_embed, mock_llm, tmp_path: Path, reply: dict[str, object]
    ) -> None:
        # A non-finding verdict passed before these checks, and a "0.9"
        # DUPLICATE confidence made a HIGH finding; both now fail the analysis.
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200, encoding="utf-8"
        )
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_llm.return_value.extract_json_from_response.return_value = reply

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is False
        assert result.status == "incomplete"
        assert result.incomplete_scans == ["deduplication-llm"]
        assert result.findings == []
        assert result.metadata["llm_analysis"]["clusters_failed"] == 1
        assert "LLM configuration or response was invalid" in result.errors[0]

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_duplicate_without_confidence_is_medium_advisory_finding(
        self, mock_embed, mock_llm, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200, encoding="utf-8"
        )
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_llm.return_value.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "rationale": "Same content",
            "suggestion": "Remove one",
        }

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is True
        assert result.incomplete_scans == []
        assert result.errors == []
        assert [(f.check_name, f.severity) for f in result.findings] == [("duplicate", Severity.MEDIUM)]
        assert result.findings[0].metadata["confidence"] == 0.0

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

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_llm_refusal_is_an_advisory_finding(self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path) -> None:
        """A safety refusal says nothing about duplication, so it must not block the skill."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_analyze.side_effect = LLMClientRefusedError("private provider refusal text")

        result = IntraSkillValidator().validate(skill_dir)

        assert result.passed is True
        assert result.status != "incomplete"
        assert result.incomplete_scans == []
        assert result.errors == []
        [finding] = result.findings
        assert (finding.category, finding.check_name, finding.severity) == (
            "CONTENT_DEDUP",
            "llm_refused",
            Severity.MEDIUM,
        )
        # The author can find the refused overlap without rerunning the scan.
        assert (finding.file_path, finding.line_number) == ("SKILL.md", 1)
        assert finding.metadata["source_files"] == ["SKILL.md"]
        assert finding.message == (
            "The LLM declined to classify overlapping content within SKILL.md:\n"
            '  "## Section A" in SKILL.md (lines 1-2)\n'
            '  vs "## Section B" in SKILL.md (lines 3-4)'
        )
        assert "SKILL_EVAL_LLM_MODEL" in finding.suggestion
        assert "private provider refusal text" not in str(finding.to_dict())
        analysis = result.metadata["llm_analysis"]
        assert (analysis["clusters_completed"], analysis["clusters_refused"], analysis["clusters_failed"]) == (0, 1, 0)
        assert analysis["failures"] == []

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_llm_refusal_beside_a_failed_cluster_keeps_the_scan_incomplete(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("\n".join(f"## Section {letter}\n{letter.lower() * 200}" for letter in "ABCD"))
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

        def analyze(_client, cluster):
            if any("Section A" in member.heading for member in cluster.members):
                raise LLMClientRefusedError("declined")
            raise RuntimeError("private provider response")

        mock_analyze.side_effect = analyze
        result = IntraSkillValidator().validate(skill)

        # Only the provider failure is missing coverage; the refusal stays advisory.
        assert result.status == "incomplete"
        assert result.incomplete_scans == ["deduplication-llm"]
        assert [finding.check_name for finding in result.findings] == ["llm_refused"]
        assert result.summary.medium_count == 1
        analysis = result.metadata["llm_analysis"]
        assert (analysis["clusters_completed"], analysis["clusters_refused"], analysis["clusters_failed"]) == (0, 1, 1)
        assert "1 of 2 content clusters" in result.errors[0]

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_llm_refusal_beside_a_duplicate_keeps_the_duplicate_blocking(
        self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        (skill / "guide.md").write_text("## Section C\n" + "c" * 200 + "\n## Section D\n" + "d" * 200)
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]

        def analyze(_client, cluster):
            if any("Section A" in member.heading for member in cluster.members):
                return LLMVerdict("DUPLICATE", 0.9, "Same content", "Remove one")
            raise LLMClientRefusedError("declined")

        mock_analyze.side_effect = analyze
        result = IntraSkillValidator().validate(skill)

        assert result.status == "failed"
        assert result.incomplete_scans == []
        assert {(finding.check_name, finding.severity) for finding in result.findings} == {
            ("duplicate", Severity.HIGH),
            ("llm_refused", Severity.MEDIUM),
        }
        refusal = next(finding for finding in result.findings if finding.check_name == "llm_refused")
        assert refusal.message.startswith(
            "The LLM declined to classify overlapping content across SKILL.md and guide.md:"
        )
        assert refusal.metadata["source_files"] == ["SKILL.md", "guide.md"]

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_policy_can_make_an_llm_refusal_blocking(self, mock_embed, _mock_llm, mock_analyze, tmp_path: Path) -> None:
        from skillevaluator.validators.policy import ValidationPolicy, apply_policy

        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_analyze.side_effect = LLMClientRefusedError("declined")
        result = IntraSkillValidator().validate(skill_dir)

        apply_policy([result], ValidationPolicy(severity_overrides={"CONTENT_DEDUP.llm_refused": Severity.HIGH}))

        assert result.passed is False
        assert result.findings[0].severity == Severity.HIGH

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


def _section_skill(tmp_path: Path, letters: str) -> Path:
    """A skill with one 200-character section per letter, in SKILL.md."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "\n".join(f"## Section {letter}\n{letter.lower() * 200}" for letter in letters), encoding="utf-8"
    )
    return skill_dir


# Sections A/B and C/D of a four-section skill form two clusters.
_TWO_CLUSTER_VECTORS = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]


def _configure_llm(mock_llm, provider: str = "anthropic", model: str = "claude-opus-5-5") -> None:
    mock_llm.return_value._resolved_config.return_value = SimpleNamespace(provider=provider, model=model)


class TestAnsweringModelProvenance:
    """A verdict names the model that answered, which can be a server-side fallback model."""

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_duplicate_finding_names_the_model_that_answered(
        self, mock_embed, mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill_dir = _section_skill(tmp_path, "AB")
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        _configure_llm(mock_llm)
        mock_analyze.return_value = LLMVerdict("DUPLICATE", 0.9, "Same content", "Remove one", model="claude-opus-4-8")

        result = IntraSkillValidator().validate(skill_dir)

        assert result.status == "failed"
        [finding] = result.findings
        assert (finding.check_name, finding.severity) == ("duplicate", Severity.HIGH)
        assert finding.metadata["llm_provider"] == "anthropic"
        assert finding.metadata["llm_model_used"] == "claude-opus-4-8"
        analysis = result.metadata["llm_analysis"]
        # "model" stays the configured model; "models_used" names the models that answered.
        assert analysis["model"] == "claude-opus-5-5"
        assert analysis["models_used"] == ["claude-opus-4-8"]
        data = json.loads(JSONReporter(include_timestamp=False).render(result))
        assert data["findings"][0]["metadata"]["llm_model_used"] == "claude-opus-4-8"
        assert data["llm_analysis"]["model"] == "claude-opus-5-5"
        assert data["llm_analysis"]["models_used"] == ["claude-opus-4-8"]

    @pytest.mark.parametrize(
        "answered_by",
        ["https://proxy.example.test/v1", "sk-test", "model with spaces"],
        ids=["url", "key-like", "spaces"],
    )
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_unsafe_answering_model_is_reported_as_unresolved(
        self, mock_embed, mock_llm, mock_analyze, tmp_path: Path, answered_by: str
    ) -> None:
        skill_dir = _section_skill(tmp_path, "AB")
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        _configure_llm(mock_llm)
        mock_analyze.return_value = LLMVerdict("DUPLICATE", 0.9, "Same content", "Remove one", model=answered_by)

        result = IntraSkillValidator().validate(skill_dir)

        [finding] = result.findings
        assert finding.metadata["llm_model_used"] == "unresolved"
        assert result.metadata["llm_analysis"]["models_used"] == ["unresolved"]
        assert answered_by not in JSONReporter(include_timestamp=False).render(result)

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_verdict_without_a_model_adds_no_model_fields(
        self, mock_embed, mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill_dir = _section_skill(tmp_path, "AB")
        mock_embed.return_value.embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
        _configure_llm(mock_llm)
        mock_analyze.return_value = LLMVerdict("DUPLICATE", 0.9, "Same content", "Remove one")

        result = IntraSkillValidator().validate(skill_dir)

        [finding] = result.findings
        assert set(finding.metadata) == {"reasoning", "confidence", "source_files"}
        assert "models_used" not in result.metadata["llm_analysis"]
        assert result.metadata["llm_analysis"]["model"] == "claude-opus-5-5"

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_refusal_advisory_keeps_the_configured_model_beside_a_fallback_verdict(
        self, mock_embed, mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill_dir = _section_skill(tmp_path, "ABCD")
        mock_embed.return_value.embed.return_value = _TWO_CLUSTER_VECTORS
        _configure_llm(mock_llm)

        def analyze(_client, cluster):
            if any("Section A" in member.heading for member in cluster.members):
                return LLMVerdict("DUPLICATE", 0.9, "Same content", "Remove one", model="claude-opus-4-8")
            raise LLMClientRefusedError("private provider refusal text")

        mock_analyze.side_effect = analyze
        result = IntraSkillValidator().validate(skill_dir)

        by_check = {finding.check_name: finding for finding in result.findings}
        assert set(by_check) == {"duplicate", "llm_refused"}
        assert by_check["duplicate"].metadata["llm_model_used"] == "claude-opus-4-8"
        # A refusal has no answering model; its advisory still names the configured one.
        assert by_check["llm_refused"].metadata == {
            "source_files": ["SKILL.md"],
            "llm_provider": "anthropic",
            "llm_model": "claude-opus-5-5",
        }
        analysis = result.metadata["llm_analysis"]
        assert (analysis["clusters_completed"], analysis["clusters_refused"]) == (1, 1)
        assert analysis["models_used"] == ["claude-opus-4-8"]

    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.analyze_cluster")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.LLMClient")
    @patch("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient")
    def test_failed_cluster_adds_no_model_beside_a_fallback_verdict(
        self, mock_embed, mock_llm, mock_analyze, tmp_path: Path
    ) -> None:
        skill_dir = _section_skill(tmp_path, "ABCD")
        mock_embed.return_value.embed.return_value = _TWO_CLUSTER_VECTORS
        _configure_llm(mock_llm)

        def analyze(_client, cluster):
            if any("Section A" in member.heading for member in cluster.members):
                return LLMVerdict("DUPLICATE", 0.9, "Same content", "Remove one", model="claude-opus-4-8")
            raise RuntimeError("private provider response")

        mock_analyze.side_effect = analyze
        result = IntraSkillValidator().validate(skill_dir)

        assert result.status == "incomplete"
        [finding] = result.findings
        assert finding.metadata["llm_model_used"] == "claude-opus-4-8"
        analysis = result.metadata["llm_analysis"]
        assert (analysis["clusters_completed"], analysis["clusters_failed"]) == (1, 1)
        assert analysis["models_used"] == ["claude-opus-4-8"]
        assert "(provider: anthropic; model: claude-opus-5-5)" in result.errors[0]

    def test_concurrent_clusters_keep_their_own_answering_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Through the real client: each worker thread reads the model that answered its own call."""
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "claude-opus-5-5")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        for name in ("SKILL_EVAL_LLM_BASE_URL", "ANTHROPIC_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        skill_dir = _section_skill(tmp_path, "ABCD")
        reply = json.dumps(
            {"verdict": "DUPLICATE", "confidence": 0.9, "rationale": "Same content", "suggestion": "Remove one"}
        )

        def handler(request):
            prompt = json.loads(request.content)["messages"][0]["content"]
            content = [{"type": "text", "text": reply}]
            answered_by = "claude-opus-5-5"
            if "Section A" in prompt:
                # The configured model refused and the fallback answered in the same call.
                answered_by = "claude-opus-4-8"
                content.insert(
                    0, {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": answered_by}}
                )
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": answered_by,
                    "content": content,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        # Both calls finish before either cluster reads its answering model, so
        # one label shared by the threads would give both clusters the same model.
        both_answered = threading.Barrier(2, timeout=30)

        class BarrierClient(LLMClient):
            def extract_json_from_response(self, *args, **kwargs):
                data = super().extract_json_from_response(*args, **kwargs)
                both_answered.wait()
                return data

        with (
            httpx.Client(transport=httpx.MockTransport(handler)) as http_client,
            patch.object(
                intra_skill_validator, "LLMClient", lambda **kwargs: BarrierClient(http_client=http_client, **kwargs)
            ),
            patch.object(intra_skill_validator, "EmbeddingClient") as mock_embed,
        ):
            mock_embed.return_value.embed.return_value = _TWO_CLUSTER_VECTORS
            result = IntraSkillValidator().validate(skill_dir)

        assert result.status == "failed"
        assert result.incomplete_scans == []
        models = {"Section A" in finding.message: finding.metadata["llm_model_used"] for finding in result.findings}
        assert models == {True: "claude-opus-4-8", False: "claude-opus-5-5"}
        assert {finding.metadata["llm_provider"] for finding in result.findings} == {"anthropic"}
        analysis = result.metadata["llm_analysis"]
        assert analysis["clusters_completed"] == 2
        assert (analysis["provider"], analysis["model"]) == ("anthropic", "claude-opus-5-5")
        assert analysis["models_used"] == ["claude-opus-4-8", "claude-opus-5-5"]
