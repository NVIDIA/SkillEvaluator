# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-based content analysis for intra-skill deduplication.

Builds prompts from content clusters, calls LLMClient, maps verdicts to severity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from skillevaluator.inference import LLMClient, LLMClientError, LLMVerdict
from skillevaluator.models.result import Severity

if TYPE_CHECKING:
    from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster

VALID_VERDICTS = {"DUPLICATE", "INTENTIONAL_DETAIL", "RELATED_BUT_DISTINCT"}

SYSTEM_PROMPT = """You are a technical content analyst for AI agent skill packages.
You analyze groups of text chunks from different files within the same skill directory.

Classify the content overlap as:

1. DUPLICATE — The chunks contain substantially the same information repeated.
   One should be consolidated into the other or removed to reduce context bloat.

2. INTENTIONAL_DETAIL — One chunk provides a summary/overview while another
   provides detailed implementation or reference. This is intentional and desirable
   (e.g., SKILL.md has a quick reference, references/*.md has the full guide).

3. RELATED_BUT_DISTINCT — The chunks are related in topic but cover different
   aspects, use cases, or serve different purposes (e.g., prose guide vs shell script).

Respond with ONLY a JSON object:
{
  "verdict": "DUPLICATE" | "INTENTIONAL_DETAIL" | "RELATED_BUT_DISTINCT",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentence explanation>",
  "suggestion": "<actionable recommendation for the content author>"
}"""


def build_user_prompt(cluster: ContentCluster) -> str:
    """Build the user prompt from cluster member chunks."""
    parts: list[str] = []
    for i, chunk in enumerate(cluster.members, start=1):
        parts.append(
            f"---CHUNK {i}---\n"
            f"File: {chunk.source_file}\n"
            f"Section: {chunk.heading}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"Content:\n{chunk.text}"
        )

    parts.append(f"\nEmbedding similarity: {cluster.max_similarity:.3f}")
    parts.append("\nClassify this content overlap.")
    return "\n\n".join(parts)


def analyze_cluster(client: LLMClient, cluster: ContentCluster) -> LLMVerdict:
    """Run LLM analysis on a single content cluster."""
    user_prompt = build_user_prompt(cluster)
    data = client.extract_json_from_response(SYSTEM_PROMPT, user_prompt)

    verdict = data.get("verdict", "")
    if verdict not in VALID_VERDICTS:
        raise LLMClientError(f"LLM returned unknown verdict '{verdict}'. Expected one of: {VALID_VERDICTS}")

    return LLMVerdict(
        verdict=verdict,
        confidence=float(data.get("confidence", 0.0)),
        reasoning=data.get("reasoning", ""),
        suggestion=data.get("suggestion", ""),
    )


def verdict_to_severity(verdict: LLMVerdict) -> Severity:
    """Map an LLM verdict to a Finding severity level."""
    if verdict.verdict == "DUPLICATE":
        return Severity.HIGH if verdict.confidence >= 0.7 else Severity.MEDIUM
    return Severity.INFO
