# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-based content analysis for intra-skill deduplication.

Builds prompts from content clusters, calls LLMClient, maps verdicts to severity.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from skillevaluator.inference import LLMClient, LLMClientError, LLMVerdict
from skillevaluator.models.result import Severity

if TYPE_CHECKING:
    from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster

VALID_VERDICTS = {"DUPLICATE", "INTENTIONAL_DETAIL", "RELATED_BUT_DISTINCT"}

# Per-call budget for one review request: SYSTEM_PROMPT plus the user prompt.
# The validator refuses a cluster whose request is still over it.
MAX_PROMPT_CHARS = 64_000

VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": sorted(VALID_VERDICTS)},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "suggestion": {"type": "string"},
    },
    "required": ["verdict", "confidence", "rationale", "suggestion"],
    "additionalProperties": False,
}

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
  "rationale": "<2-3 sentence explanation>",
  "suggestion": "<actionable recommendation for the content author>"
}"""


def build_user_prompt(cluster: ContentCluster) -> str:
    """Build the user prompt from cluster member chunks.

    Every member is rendered with its full text. When the request
    (``SYSTEM_PROMPT`` plus this prompt) would exceed ``MAX_PROMPT_CHARS``,
    each exactly repeated chunk text is listed once as ``SHARED CONTENT n``.
    Every occurrence keeps its own ``CHUNK`` entry with file, section and
    lines, plus a reference to the shared text. Distinct text is never
    dropped or shortened, so a request still over budget stays over budget.
    """
    prompt = _render_user_prompt(cluster)
    if len(SYSTEM_PROMPT) + len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    counts = Counter(chunk.text for chunk in cluster.members)
    repeated = (text for text, count in counts.items() if count > 1)
    shared = {text: index for index, text in enumerate(repeated, start=1)}
    if not shared:
        return prompt
    compact = _render_user_prompt(cluster, shared)
    return compact if len(compact) < len(prompt) else prompt


def _render_user_prompt(cluster: ContentCluster, shared: dict[str, int] | None = None) -> str:
    """Render every member; texts in ``shared`` are listed once and referenced."""
    parts: list[str] = []
    if shared:
        parts.append(
            "Identical content is listed once below. Each content reference stands for the exact "
            "shared text. Review every occurrence in its own file, section and line context. "
            "No chunks were omitted."
        )
        parts.extend(f"---SHARED CONTENT {index}---\n{text}" for text, index in shared.items())
    for i, chunk in enumerate(cluster.members, start=1):
        if shared and chunk.text in shared:
            content = f"Content reference: SHARED CONTENT {shared[chunk.text]}"
        else:
            content = f"Content:\n{chunk.text}"
        parts.append(
            f"---CHUNK {i}---\n"
            f"File: {chunk.source_file}\n"
            f"Section: {chunk.heading}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"{content}"
        )

    parts.append(f"\nEmbedding similarity: {cluster.max_similarity:.3f}")
    parts.append("\nClassify this content overlap.")
    return "\n\n".join(parts)


def analyze_cluster(client: LLMClient, cluster: ContentCluster) -> LLMVerdict:
    """Run LLM analysis on a single content cluster."""
    user_prompt = build_user_prompt(cluster)
    data = client.extract_json_from_response(
        SYSTEM_PROMPT,
        user_prompt,
        response_schema=VERDICT_JSON_SCHEMA,
        schema_name="dedup_verdict",
    )

    verdict = data.get("verdict", "")
    if verdict not in VALID_VERDICTS:
        raise LLMClientError(f"LLM returned unknown verdict '{verdict}'. Expected one of: {VALID_VERDICTS}")

    return LLMVerdict(
        verdict=verdict,
        confidence=float(data.get("confidence", 0.0)),
        reasoning=data.get("rationale", data.get("reasoning", "")),
        suggestion=data.get("suggestion", ""),
    )


def verdict_to_severity(verdict: LLMVerdict) -> Severity:
    """Map an LLM verdict to a Finding severity level."""
    if verdict.verdict == "DUPLICATE":
        return Severity.HIGH if verdict.confidence >= 0.7 else Severity.MEDIUM
    return Severity.INFO
