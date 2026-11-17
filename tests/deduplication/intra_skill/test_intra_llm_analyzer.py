# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.intra_skill.llm_analyzer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skillevaluator.deduplication.intra_skill import llm_analyzer
from skillevaluator.deduplication.intra_skill.llm_analyzer import (
    SYSTEM_PROMPT,
    VERDICT_JSON_SCHEMA,
    analyze_cluster,
    build_user_prompt,
    verdict_to_severity,
)
from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster
from skillevaluator.inference import LLMClient, LLMClientError, LLMVerdict
from skillevaluator.models.result import Severity


def _request_chars(prompt: str) -> int:
    """Characters of one review request, measured as the validator measures it."""
    return len(SYSTEM_PROMPT) + len(prompt)


class TestBuildUserPrompt:
    def test_large_repeated_content_preserves_each_occurrence(self, make_chunk) -> None:
        repeated = "Exact repeated instructions. " * 10
        unique = "Unique instructions for the final section. " * 3
        members = [
            make_chunk(source_file=f"file-{index // 4}.md", heading=f"Section {index}", text=repeated)
            for index in range(500)
        ]
        members.append(make_chunk(source_file="unique.md", heading="Unique", text=unique))
        cluster = ContentCluster(members, 0.95, 0.85, True, {"markdown"})

        prompt = build_user_prompt(cluster)

        assert _request_chars(prompt) <= llm_analyzer.MAX_PROMPT_CHARS
        assert prompt.count(repeated) == 1
        assert prompt.count(unique) == 1
        assert prompt.count("---SHARED CONTENT ") == 1
        assert prompt.count("---CHUNK ") == 501
        assert prompt.count("Content:\n[SHARED CONTENT 1]") == 500
        assert "No chunks were omitted." in prompt
        for index, member in enumerate(members, start=1):
            assert (
                f"---CHUNK {index}---\nFile: {member.source_file}\nSection: {member.heading}\n"
                f"Lines: {member.start_line}-{member.end_line}\n"
            ) in prompt
        assert f"Section: Unique\nLines: 1-5\nContent:\n{unique}" in prompt
        assert prompt.endswith("Classify this content overlap.")

    def test_equally_repeated_texts_are_numbered_in_first_seen_order(self, make_chunk) -> None:
        first = "f" * 1_500
        second = "s" * 1_500
        members = [make_chunk(heading=f"Section {index}", text=first if index % 2 else second) for index in range(60)]
        cluster = ContentCluster(members, 0.95, 0.85, False, {"markdown"})

        prompt = build_user_prompt(cluster)

        assert _request_chars(prompt) <= llm_analyzer.MAX_PROMPT_CHARS
        assert prompt.count(first) == prompt.count(second) == 1
        assert prompt.index(f"---SHARED CONTENT 1---\n{second}") < prompt.index(f"---SHARED CONTENT 2---\n{first}")
        assert prompt.count("Content:\n[SHARED CONTENT 1]") == 30
        assert prompt.count("Content:\n[SHARED CONTENT 2]") == 30
        assert "Section: Section 1\nLines: 1-5\nContent:\n[SHARED CONTENT 2]" in prompt

    def test_most_repeated_text_gets_the_shortest_reference(self, make_chunk) -> None:
        # Nine 62-character texts seen first, each twice, save one character
        # each as blocks 1-9. Numbered after them, a 40-character body repeated
        # 400 times would be block 10, and its 19-character references would
        # cost 400 characters more in all than references to blocks 1-9.
        pairs = [letter * 62 for letter in "acdefghij"]
        body = "b" * 40
        members = [make_chunk(heading=f"Pair {k}.{j}", text=text) for k, text in enumerate(pairs) for j in range(2)]
        members += [make_chunk(heading=f"Body {j}", text=body) for j in range(400)]
        filler = make_chunk(source_file="unique.md", heading="Filler", text="u")
        cluster = ContentCluster([*members, filler], 0.95, 0.85, False, {"markdown"})
        budget = llm_analyzer.MAX_PROMPT_CHARS
        first_seen = {text: index for index, text in enumerate([*pairs, body], start=1)}
        filler.text += "u" * (budget + 200 - _request_chars(llm_analyzer._render_user_prompt(cluster, first_seen)))
        assert _request_chars(llm_analyzer._render_user_prompt(cluster, first_seen)) == budget + 200

        prompt = build_user_prompt(cluster)

        assert _request_chars(prompt) == budget - 200
        assert prompt.count(f"---SHARED CONTENT 1---\n{body}\n\n") == 1
        assert prompt.count("Content:\n[SHARED CONTENT 1]") == 400
        # Equally repeated, the pairs keep first-seen order; the ninth would be
        # block 10, where sharing it costs characters, so it stays inline.
        for index, text in enumerate(pairs[:8], start=2):
            assert prompt.count(f"---SHARED CONTENT {index}---\n{text}\n\n") == 1
        assert "SHARED CONTENT 10" not in prompt
        assert prompt.count(f"Content:\n{pairs[8]}\n\n") == 2

    def test_text_that_would_push_a_more_repeated_text_past_block_9_stays_inline(self, make_chunk) -> None:
        # The most repeated text saves 14 characters as block 1 and none past
        # block 9. Sharing it moves the ninth other text to block 10, which
        # costs that text's 19 references and its header one character each.
        marginal = "m" * 21
        others = [letter * 200 for letter in "abcdefghi"]
        members = [make_chunk(heading=f"Marginal {j}", text=marginal) for j in range(20)]
        members += [make_chunk(heading=f"Other {k}.{j}", text=text) for k, text in enumerate(others) for j in range(19)]
        members.append(make_chunk(source_file="unique.md", heading="Filler", text="u" * 20_000))
        cluster = ContentCluster(members, 0.95, 0.85, False, {"markdown"})
        greedy = {marginal: 1, **{text: index for index, text in enumerate(others, start=2)}}
        assert _request_chars(llm_analyzer._render_user_prompt(cluster)) > llm_analyzer.MAX_PROMPT_CHARS

        prompt = build_user_prompt(cluster)

        assert len(llm_analyzer._render_user_prompt(cluster, greedy)) - len(prompt) == 20 - 14
        assert prompt.count(f"Content:\n{marginal}\n\n") == 20
        for index, text in enumerate(others, start=1):
            assert prompt.count(f"---SHARED CONTENT {index}---\n{text}\n\n") == 1
            assert prompt.count(f"Content:\n[SHARED CONTENT {index}]") == 19
        assert "SHARED CONTENT 10" not in prompt

    def test_request_budget_includes_the_system_prompt(self, make_chunk, monkeypatch) -> None:
        repeated = "r" * 2_000
        filler = make_chunk(source_file="unique.md", heading="Unique", text="u" * 80)
        members = [make_chunk(heading=f"Section {index}", text=repeated) for index in range(30)]
        cluster = ContentCluster([*members, filler], 0.95, 0.85, True, {"markdown"})
        budget = llm_analyzer.MAX_PROMPT_CHARS
        with monkeypatch.context() as patched:
            patched.setattr(llm_analyzer, "MAX_PROMPT_CHARS", 10**9)
            filler.text += "u" * (budget - len(SYSTEM_PROMPT) // 2 - len(build_user_prompt(cluster)))
            full_prompt = build_user_prompt(cluster)
        # The user prompt alone fits; the request with the system prompt does not.
        assert len(full_prompt) <= budget < _request_chars(full_prompt)

        prompt = build_user_prompt(cluster)

        assert _request_chars(prompt) <= budget
        assert prompt.count(repeated) == 1
        assert prompt.count(filler.text) == 1
        assert prompt.count("---CHUNK ") == 31

    def test_prompt_within_budget_keeps_every_occurrence_inline(self, make_chunk) -> None:
        repeated = "Exact repeated instructions. " * 10
        members = [make_chunk(heading=f"Section {index}", text=repeated) for index in range(3)]

        prompt = build_user_prompt(ContentCluster(members, 0.95, 0.85, False, {"markdown"}))

        assert prompt.count(f"Content:\n{repeated}") == 3
        assert "SHARED CONTENT" not in prompt

    def test_distinct_text_over_budget_is_never_dropped_or_shared(self, make_chunk) -> None:
        members = [make_chunk(heading=f"Member {index}", text=f"{index}: " + "x" * 80) for index in range(500)]

        prompt = build_user_prompt(ContentCluster(members, 0.95, 0.85, False, {"markdown"}))

        assert _request_chars(prompt) > llm_analyzer.MAX_PROMPT_CHARS
        assert "SHARED CONTENT" not in prompt
        for member in members:
            assert f"Section: {member.heading}\nLines: 1-5\nContent:\n{member.text}" in prompt

    @pytest.mark.parametrize(
        "repeated",
        [
            pytest.param("short repeated line", id="block-longer-than-copies"),
            # Sharing this text alone saves 39 characters, fewer than the
            # shared-content notice adds.
            pytest.param("n" * 100, id="saving-smaller-than-notice"),
        ],
    )
    def test_sharing_is_skipped_when_it_does_not_shorten_the_prompt(self, make_chunk, repeated: str) -> None:
        members = [make_chunk(heading=f"Distinct {index}", text=f"{index}: " + "d" * 2_500) for index in range(30)]
        members += [make_chunk(heading=f"Short {index}", text=repeated) for index in range(2)]

        prompt = build_user_prompt(ContentCluster(members, 0.95, 0.85, False, {"markdown"}))

        assert _request_chars(prompt) > llm_analyzer.MAX_PROMPT_CHARS
        assert "SHARED CONTENT" not in prompt
        assert prompt.count(f"Content:\n{repeated}") == 2

    @pytest.mark.parametrize(
        ("length", "shared"),
        [pytest.param(61, False, id="saves-nothing"), pytest.param(62, True, id="saves-one-char")],
    )
    def test_repeated_text_is_shared_only_when_that_saves_characters(
        self, make_chunk, length: int, shared: bool
    ) -> None:
        # Two inline copies cost 2 * length characters. As block 2 they cost two
        # 18-character "[SHARED CONTENT 2]" references, the 23-character block
        # header, one copy and a 2-character separator: length + 61.
        big = "f" * 1_500
        text = "t" * length
        members = [make_chunk(heading=f"Big {index}", text=big) for index in range(60)]
        members += [make_chunk(heading=f"Small {index}", text=text) for index in range(2)]

        prompt = build_user_prompt(ContentCluster(members, 0.95, 0.85, False, {"markdown"}))

        assert _request_chars(prompt) <= llm_analyzer.MAX_PROMPT_CHARS
        assert prompt.count(big) == 1
        assert prompt.count("[SHARED CONTENT 1]") == 60
        if shared:
            assert prompt.count(f"---SHARED CONTENT 2---\n{text}\n\n") == 1
            assert prompt.count("Content:\n[SHARED CONTENT 2]") == 2
        else:
            assert "SHARED CONTENT 2" not in prompt
            assert prompt.count(f"Content:\n{text}\n\n") == 2

    @pytest.mark.parametrize(
        ("length", "shared"),
        [pytest.param(64, False, id="saves-nothing-as-block-10"), pytest.param(65, True, id="saves-one-char")],
    )
    def test_two_digit_block_numbers_cost_one_more_character(self, make_chunk, length: int, shared: bool) -> None:
        # As block 10, two copies cost two 19-character "[SHARED CONTENT 10]"
        # references, the 24-character block header, one copy and a
        # 2-character separator: length + 64. As block 1-9 they would cost
        # length + 61, so a 64-character text would be shared there.
        bigs = [letter * 1_500 for letter in "abcdefghi"]
        text = "t" * length
        members = [make_chunk(heading=f"Big {k}.{j}", text=big) for k, big in enumerate(bigs) for j in range(5)]
        members += [make_chunk(heading=f"Small {index}", text=text) for index in range(2)]

        prompt = build_user_prompt(ContentCluster(members, 0.95, 0.85, False, {"markdown"}))

        assert _request_chars(prompt) <= llm_analyzer.MAX_PROMPT_CHARS
        for index, big in enumerate(bigs, start=1):
            assert prompt.count(f"---SHARED CONTENT {index}---\n{big}\n\n") == 1
        if shared:
            assert prompt.count(f"---SHARED CONTENT 10---\n{text}\n\n") == 1
            assert prompt.count("Content:\n[SHARED CONTENT 10]") == 2
        else:
            assert "SHARED CONTENT 10" not in prompt
            assert prompt.count(f"Content:\n{text}\n\n") == 2

    @pytest.mark.parametrize(
        "heading",
        [
            pytest.param("## H{}", id="short-headings"),
            pytest.param("## Configuration option for the workflow runner, number {:04d}", id="long-headings"),
        ],
    )
    def test_short_repeated_body_stays_inline_whatever_its_heading_length(self, make_chunk, heading: str) -> None:
        # The heading line stays with every occurrence whether or not the body
        # is shared, so only the body's length and repeat count decide.
        short_body = "See guide."
        long_body = "Boilerplate paragraph about safety and usage. " * 20
        short_members = [
            make_chunk(heading=heading.format(index), text=f"{heading.format(index)}\n{short_body}")
            for index in range(200)
        ]
        long_members = [
            make_chunk(heading=f"## Big {index}", text=f"## Big {index}\n{long_body}") for index in range(60)
        ]
        cluster = ContentCluster([*short_members, *long_members], 0.95, 0.85, False, {"markdown"})
        assert _request_chars(llm_analyzer._render_user_prompt(cluster)) > llm_analyzer.MAX_PROMPT_CHARS

        prompt = build_user_prompt(cluster)

        assert prompt.count("---SHARED CONTENT ") == 1
        assert prompt.count(f"---SHARED CONTENT 1---\n{long_body}\n\n") == 1
        for member in short_members:
            assert f"Section: {member.heading}\nLines: 1-5\nContent:\n{member.text}\n\n" in prompt

    def test_short_repeated_bodies_under_long_headings_stay_inline(self, make_chunk) -> None:
        short_body = "See guide."
        long_body = "Boilerplate paragraph about safety and usage. " * 20
        short_members = [
            make_chunk(source_file="SKILL.md", heading=heading, text=f"{heading}\n{short_body}")
            for heading in (
                f"## Configuration option for the workflow runner, number {index:04d} of the set"
                for index in range(270)
            )
        ]
        long_members = [
            make_chunk(source_file="SKILL.md", heading=f"## Big {index}", text=f"## Big {index}\n{long_body}")
            for index in range(25)
        ]
        cluster = ContentCluster([*short_members, *long_members], 0.95, 0.85, False, {"markdown"})
        budget = llm_analyzer.MAX_PROMPT_CHARS
        # Sent in full, or with both repeated bodies shared, the request is over
        # budget: a "[SHARED CONTENT n]" reference is longer than "See guide.".
        assert _request_chars(llm_analyzer._render_user_prompt(cluster)) > budget
        assert _request_chars(llm_analyzer._render_user_prompt(cluster, {short_body: 1, long_body: 2})) > budget

        prompt = build_user_prompt(cluster)

        assert _request_chars(prompt) <= budget
        assert prompt.count("---SHARED CONTENT ") == 1
        assert prompt.count(f"---SHARED CONTENT 1---\n{long_body}\n\n") == 1
        assert prompt.count(long_body) == 1
        assert prompt.count("Content:\n## Big ") == prompt.count("[SHARED CONTENT 1]") == 25
        for member in short_members:
            assert f"Section: {member.heading}\nLines: 1-5\nContent:\n{member.text}\n\n" in prompt

    def test_same_body_under_different_heading_lines_is_shared(self, make_chunk) -> None:
        body = "Describe the supported workflow. " * 4
        members = [make_chunk(heading=f"## Step {index}", text=f"## Step {index}\n{body}") for index in range(500)]
        cluster = ContentCluster(members, 0.95, 0.85, False, {"markdown"})

        prompt = build_user_prompt(cluster)

        assert _request_chars(prompt) <= llm_analyzer.MAX_PROMPT_CHARS
        assert prompt.count(body) == 1
        assert f"---SHARED CONTENT 1---\n{body}\n\n---CHUNK 1---" in prompt
        assert prompt.count("---CHUNK ") == 500
        for index, member in enumerate(members, start=1):
            # The heading line plus the shared body is exactly the member's text.
            assert (
                f"---CHUNK {index}---\nFile: README.md\nSection: {member.heading}\nLines: 1-5\n"
                f"Content:\n{member.heading}\n[SHARED CONTENT 1]\n\n"
            ) in prompt

    def test_heading_line_with_a_carriage_return_is_kept_with_each_occurrence(self, make_chunk) -> None:
        body = "Describe the supported workflow.\r\n" * 4
        members = [
            make_chunk(heading=f"## Step {index}", text=f"## Step {index}\r\n{body}".strip()) for index in range(500)
        ]
        cluster = ContentCluster(members, 0.95, 0.85, False, {"markdown"})

        prompt = build_user_prompt(cluster)

        assert _request_chars(prompt) <= llm_analyzer.MAX_PROMPT_CHARS
        assert prompt.count(body.strip()) == 1
        assert prompt.count("---SHARED CONTENT ") == 1
        for member in members:
            assert f"Content:\n{member.heading}\r\n[SHARED CONTENT 1]" in prompt

    def test_whole_text_matching_another_chunks_body_shares_it(self, make_chunk) -> None:
        body = "Describe the supported workflow. " * 4
        sections = [make_chunk(heading=f"## Step {index}", text=f"## Step {index}\n{body}") for index in range(250)]
        docstrings = [
            make_chunk(source_file="tool.py", heading=f"step_{index}()", text=body, source_format="python")
            for index in range(250)
        ]
        cluster = ContentCluster([*sections, *docstrings], 0.95, 0.85, True, {"markdown", "python"})

        prompt = build_user_prompt(cluster)

        assert _request_chars(prompt) <= llm_analyzer.MAX_PROMPT_CHARS
        assert prompt.count(body) == 1
        assert prompt.count("---SHARED CONTENT ") == 1
        assert prompt.count("[SHARED CONTENT 1]") == 500
        for member in docstrings:
            assert f"Section: {member.heading}\nLines: 1-5\nContent:\n[SHARED CONTENT 1]" in prompt

    def test_text_that_does_not_open_with_its_heading_line_is_compared_whole(self, make_chunk) -> None:
        body = "Describe the supported workflow. " * 4
        # The first line differs per chunk but is not the chunk's heading.
        members = [make_chunk(heading=f"## Step {index}", text=f"Step {index}\n{body}") for index in range(500)]
        cluster = ContentCluster(members, 0.95, 0.85, False, {"markdown"})

        prompt = build_user_prompt(cluster)

        assert _request_chars(prompt) > llm_analyzer.MAX_PROMPT_CHARS
        assert "SHARED CONTENT" not in prompt
        assert prompt.count(body) == 500

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

    def test_sends_the_shared_content_prompt_that_the_validator_measures(self, make_chunk) -> None:
        repeated = "Exact repeated instructions. " * 10
        members = [make_chunk(heading=f"Section {index}", text=repeated) for index in range(500)]
        cluster = ContentCluster(members, 0.95, 0.85, False, {"markdown"})
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "confidence": 0.9,
            "rationale": "Same instructions in every section",
            "suggestion": "Keep one copy",
        }

        analyze_cluster(mock_client, cluster)

        system_prompt, user_prompt = mock_client.extract_json_from_response.call_args.args
        assert system_prompt == SYSTEM_PROMPT
        assert user_prompt == build_user_prompt(cluster)
        assert user_prompt.count(repeated) == 1
        assert _request_chars(user_prompt) <= llm_analyzer.MAX_PROMPT_CHARS

    def test_requests_the_verdict_schema_and_reads_rationale(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "confidence": 0.9,
            "rationale": "Same steps in both files",
            "suggestion": "Keep one copy",
        }

        result = analyze_cluster(mock_client, make_cluster())

        assert result.reasoning == "Same steps in both files"
        assert mock_client.extract_json_from_response.call_args.kwargs["response_schema"] == VERDICT_JSON_SCHEMA
        assert '"rationale"' in SYSTEM_PROMPT

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

    @pytest.mark.parametrize(
        "confidence",
        [float("nan"), float("inf"), -0.1, 1.1, 85, True, "0.5", "NaN", None, 10**400],
        ids=[
            "nan",
            "inf",
            "negative",
            "above-one",
            "percent",
            "bool",
            "numeric-string",
            "nan-string",
            "null",
            "huge-int",
        ],
    )
    def test_confidence_outside_zero_to_one_is_rejected(self, make_cluster, confidence: object) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "confidence": confidence,
            "rationale": "Reason",
            "suggestion": "Suggestion",
        }

        with pytest.raises(LLMClientError, match=r"confidence must be a finite number within \[0, 1\]"):
            analyze_cluster(mock_client, make_cluster())

    @pytest.mark.parametrize(("confidence", "expected"), [(0, 0.0), (1, 1.0), (0.0, 0.0), (1.0, 1.0)])
    def test_confidence_bounds_are_accepted_as_floats(self, make_cluster, confidence: object, expected: float) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "confidence": confidence,
            "rationale": "Reason",
            "suggestion": "Suggestion",
        }

        result = analyze_cluster(mock_client, make_cluster())

        assert type(result.confidence) is float
        assert result.confidence == expected

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("verdict", ["DUPLICATE"], "verdict must be a string"),
            ("verdict", None, "verdict must be a string"),
            ("rationale", ["Reason"], "rationale must be a string"),
            ("rationale", None, "rationale must be a string"),
            ("suggestion", {"text": "Suggestion"}, "suggestion must be a string"),
            ("suggestion", 3, "suggestion must be a string"),
            ("rationale", "x" * 8_193, "rationale exceeds the 8192-character limit"),
            ("suggestion", "x" * 8_193, "suggestion exceeds the 8192-character limit"),
        ],
        ids=[
            "verdict-list",
            "verdict-null",
            "rationale-list",
            "rationale-null",
            "suggestion-object",
            "suggestion-number",
            "rationale-too-long",
            "suggestion-too-long",
        ],
    )
    def test_non_string_or_oversized_reply_fields_are_rejected(
        self, make_cluster, field: str, value: object, message: str
    ) -> None:
        mock_client = MagicMock(spec=LLMClient)
        reply: dict[str, object] = {
            "verdict": "DUPLICATE",
            "confidence": 0.9,
            "rationale": "Reason",
            "suggestion": "Suggestion",
        }
        reply[field] = value
        mock_client.extract_json_from_response.return_value = reply

        with pytest.raises(LLMClientError, match=message):
            analyze_cluster(mock_client, make_cluster())

    def test_legacy_reasoning_field_is_bounded_too(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "confidence": 0.9,
            "reasoning": {"text": "Reason"},
            "suggestion": "Suggestion",
        }

        with pytest.raises(LLMClientError, match="rationale must be a string"):
            analyze_cluster(mock_client, make_cluster())

    def test_reply_fields_at_the_length_limit_are_kept_whole(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {
            "verdict": "DUPLICATE",
            "confidence": 0.9,
            "rationale": "r" * 8_192,
            "suggestion": "s" * 8_192,
        }

        result = analyze_cluster(mock_client, make_cluster())

        assert result.reasoning == "r" * 8_192
        assert result.suggestion == "s" * 8_192

    def test_missing_confidence_and_text_fields_keep_their_defaults(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = {"verdict": "RELATED_BUT_DISTINCT"}

        result = analyze_cluster(mock_client, make_cluster())

        assert result == LLMVerdict("RELATED_BUT_DISTINCT", 0.0, "", "")

    @pytest.mark.parametrize("reply", [["DUPLICATE"], "DUPLICATE", 0.9, None], ids=["list", "string", "number", "null"])
    def test_reply_that_is_not_a_json_object_is_rejected(self, make_cluster, reply: object) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = reply

        with pytest.raises(LLMClientError, match="response must be a JSON object"):
            analyze_cluster(mock_client, make_cluster())


_DUPLICATE_REPLY = {"verdict": "DUPLICATE", "confidence": 0.9, "rationale": "Same content", "suggestion": "Remove one"}


class TestAnsweringModel:
    """The verdict names the model that answered, which can be a server-side fallback."""

    def test_verdict_records_the_model_that_answered(self, make_cluster) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = dict(_DUPLICATE_REPLY)
        mock_client.model_used = "claude-opus-4-8"

        result = analyze_cluster(mock_client, make_cluster())

        assert result.model == "claude-opus-4-8"
        assert (result.verdict, result.confidence) == ("DUPLICATE", 0.9)

    def test_model_is_kept_as_the_client_named_it(self, make_cluster) -> None:
        # Reports sanitize the model; the verdict keeps what the client said.
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = dict(_DUPLICATE_REPLY)
        mock_client.model_used = "https://proxy.example.test/v1"

        assert analyze_cluster(mock_client, make_cluster()).model == "https://proxy.example.test/v1"

    @pytest.mark.parametrize("model_used", [None, 42, ["claude-opus-4-8"]], ids=["none", "number", "list"])
    def test_verdict_names_no_model_unless_the_client_gives_a_string(self, make_cluster, model_used: object) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.extract_json_from_response.return_value = dict(_DUPLICATE_REPLY)
        mock_client.model_used = model_used

        assert analyze_cluster(mock_client, make_cluster()).model is None

    def test_client_without_model_used_gives_no_model(self, make_cluster) -> None:
        class Client:
            def extract_json_from_response(self, *_args, **_kwargs):
                return {"verdict": "INTENTIONAL_DETAIL", "confidence": 0.8}

        result = analyze_cluster(Client(), make_cluster())

        assert result == LLMVerdict("INTENTIONAL_DETAIL", 0.8, "", "")
        assert result.model is None

    def test_positional_verdict_has_no_model(self) -> None:
        assert LLMVerdict("DUPLICATE", 0.9, "r", "s").model is None


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
