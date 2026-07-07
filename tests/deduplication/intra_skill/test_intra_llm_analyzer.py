# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.intra_skill.llm_analyzer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skillevaluator.deduplication.intra_skill.llm_analyzer import (
    analyze_cluster,
    build_user_prompt,
    verdict_to_severity,
)
from skillevaluator.inference import LLMClient, LLMClientError, LLMVerdict
from skillevaluator.models.result import Severity


class TestBuildUserPrompt:
    def test_contains_chunk_details(self, make_cluster) -> None:
        cluster = make_cluster()
        prompt = build_user_prompt(cluster)
        assert "---CHUNK 1---" in prompt
        assert "---CHUNK 2---" in prompt
        assert "File:" in prompt
        assert "Section:" in prompt
        assert "Lines:" in prompt

    def test_contains_similarity_score(self, make_cluster) -> None:
        cluster = make_cluster(max_similarity=0.925)
        prompt = build_user_prompt(cluster)
        assert "0.925" in prompt

    def test_contains_classify_instruction(self, make_cluster) -> None:
        cluster = make_cluster()
        prompt = build_user_prompt(cluster)
        assert "Classify this content overlap" in prompt

    def test_includes_chunk_content(self, make_chunk) -> None:
        chunk = make_chunk(text="This is the actual content of the chunk.")
        cluster_members = [chunk, make_chunk(source_file="other.md")]
        from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster

        cluster = ContentCluster(
            members=cluster_members,
            max_similarity=0.9,
            avg_similarity=0.85,
            cross_file=True,
            source_formats={"markdown"},
        )
        prompt = build_user_prompt(cluster)
        assert "This is the actual content" in prompt


class TestAnalyzeCluster:
    def test_duplicate_verdict(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "confidence": 0.9,
            "reasoning": "Same content",
            "suggestion": "Remove one",
        }
        cluster = make_cluster()
        result = analyze_cluster(mock_client, cluster)
        assert result.verdict == "DUPLICATE"
        assert result.confidence == 0.9

    def test_intentional_detail_verdict(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "INTENTIONAL_DETAIL",
            "confidence": 0.85,
            "reasoning": "Summary vs detail",
            "suggestion": "Keep both",
        }
        result = analyze_cluster(mock_client, make_cluster())
        assert result.verdict == "INTENTIONAL_DETAIL"

    def test_related_but_distinct_verdict(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "RELATED_BUT_DISTINCT",
            "confidence": 0.80,
            "reasoning": "Different purpose",
            "suggestion": "Keep both",
        }
        result = analyze_cluster(mock_client, make_cluster())
        assert result.verdict == "RELATED_BUT_DISTINCT"

    def test_unknown_verdict_raises(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "SOMETHING_ELSE",
            "confidence": 0.9,
            "reasoning": "Bad",
            "suggestion": "Bad",
        }
        with pytest.raises(LLMClientError, match="unknown verdict"):
            analyze_cluster(mock_client, make_cluster())

    def test_missing_verdict_raises(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "confidence": 0.9,
            "reasoning": "No verdict key",
        }
        with pytest.raises(LLMClientError, match="unknown verdict"):
            analyze_cluster(mock_client, make_cluster())

    def test_extracts_all_fields(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "confidence": 0.77,
            "reasoning": "The reasoning text",
            "suggestion": "The suggestion text",
        }
        result = analyze_cluster(mock_client, make_cluster())
        assert result.reasoning == "The reasoning text"
        assert result.suggestion == "The suggestion text"
        assert result.confidence == 0.77


class TestVerdictToSeverity:
    def test_duplicate_high_confidence(self) -> None:
        v = LLMVerdict("DUPLICATE", 0.9, "r", "s")
        assert verdict_to_severity(v) == Severity.HIGH

    def test_duplicate_low_confidence(self) -> None:
        v = LLMVerdict("DUPLICATE", 0.5, "r", "s")
        assert verdict_to_severity(v) == Severity.MEDIUM

    def test_duplicate_boundary_confidence(self) -> None:
        v = LLMVerdict("DUPLICATE", 0.7, "r", "s")
        assert verdict_to_severity(v) == Severity.HIGH

    def test_intentional_detail(self) -> None:
        v = LLMVerdict("INTENTIONAL_DETAIL", 0.95, "r", "s")
        assert verdict_to_severity(v) == Severity.INFO

    def test_related_but_distinct(self) -> None:
        v = LLMVerdict("RELATED_BUT_DISTINCT", 0.88, "r", "s")
        assert verdict_to_severity(v) == Severity.INFO
