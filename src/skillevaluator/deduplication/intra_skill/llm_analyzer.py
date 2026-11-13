# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-based content analysis for intra-skill deduplication.

Builds prompts from content clusters, calls LLMClient, maps verdicts to severity.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from skillevaluator.inference import LLMClient, LLMVerdict, parse_bounded_llm_verdict
from skillevaluator.models.result import Severity

if TYPE_CHECKING:
    from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster
    from skillevaluator.deduplication.utils.chunker import ContentChunk

VALID_VERDICTS = {"DUPLICATE", "INTENTIONAL_DETAIL", "RELATED_BUT_DISTINCT"}

# Per-call budget for one review request: SYSTEM_PROMPT plus the user prompt.
# The validator refuses a cluster whose request is still over it.
MAX_PROMPT_CHARS = 64_000

# Joins the parts of a user prompt (notice, shared blocks, chunks, footer).
_PART_SEPARATOR = "\n\n"

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
    each exactly repeated text is listed once as ``SHARED CONTENT n``. A
    Markdown section's text opens with its own heading line, so the text
    compared is the body after that line: the same body under different
    headings is listed once too. Every occurrence keeps its own ``CHUNK``
    entry with file, section and lines, its own heading line, and a
    ``[SHARED CONTENT n]`` reference in place of the shared text. A repeated
    text is shared only when its block and references are shorter than its
    inline copies. A short text, such as a one-line body, stays inline
    because its reference would be longer than the text; the heading line
    stays with every occurrence either way, so its length does not matter.
    Blocks are numbered and chosen as described in ``_choose_shared_texts``.
    Distinct text is never dropped or shortened, so a request still over
    budget stays over budget.
    """
    prompt = _render_user_prompt(cluster)
    if len(SYSTEM_PROMPT) + len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    shared = _choose_shared_texts(Counter(_split_own_heading(chunk)[1] for chunk in cluster.members))
    if not shared:
        return prompt
    compact = _render_user_prompt(cluster, shared)
    return compact if len(compact) < len(prompt) else prompt


def _choose_shared_texts(counts: Counter[str]) -> dict[str, int]:
    """Return the repeated texts to share, mapped to their block numbers in order.

    A reference to block 10 or later is one character longer than one to
    blocks 1-9, so the numbering matters. For any set of shared texts, the
    prompt is shortest when the most repeated texts get the lowest numbers,
    so candidates are numbered in descending repeat count, ties in first-seen
    order. Which candidates to share is then chosen exactly, by dynamic
    programming over that order, to save the most characters: sharing a text
    that saves only a little can push a more repeated text past block 9 and
    cost it one character in every reference.
    """
    candidates = [(text, count) for text, count in counts.most_common() if count > 1]
    # best[k]: most characters saved by sharing k of the candidates seen so
    # far as blocks 1..k, or None when no such k of them all save characters.
    best: list[int | None] = [0] + [None] * len(candidates)
    taken: list[list[bool]] = []
    for seen, (text, count) in enumerate(candidates, start=1):
        row = [False] * (len(candidates) + 1)
        for index in range(seen, 0, -1):
            before = best[index - 1]
            saving = _sharing_saving(text, count, index)
            current = best[index]
            if before is not None and saving > 0 and (current is None or before + saving > current):
                best[index] = before + saving
                row[index] = True
        taken.append(row)
    # The fewest blocks among the largest savings.
    blocks = max(range(len(best)), key=lambda k: -1 if best[k] is None else best[k])
    shared: dict[str, int] = {}
    for (text, _count), row in zip(reversed(candidates), reversed(taken), strict=True):
        if row[blocks]:
            shared[text] = blocks
            blocks -= 1
    return dict(sorted(shared.items(), key=lambda item: item[1]))


def _split_own_heading(chunk: ContentChunk) -> tuple[str, str]:
    """Return the chunk's own heading line and the text after it.

    A Markdown section chunk opens with its heading line, so the same body
    under different headings differs only in that line. Text that does not
    open with the chunk's heading line, or has nothing after it, is returned
    whole with an empty heading line. When a heading line is returned,
    ``heading_line + "\n" + body`` is exactly the chunk text.
    """
    first_line, newline, body = chunk.text.partition("\n")
    if newline and body.strip() and first_line.rstrip() == chunk.heading:
        return first_line, body
    return "", chunk.text


def _shared_block(text: str, index: int) -> str:
    return f"---SHARED CONTENT {index}---\n{text}"


def _shared_reference(index: int) -> str:
    return f"[SHARED CONTENT {index}]"


def _sharing_saving(text: str, count: int, index: int) -> int:
    """Characters saved by listing ``text`` once as shared block ``index``.

    Inline, each of the ``count`` occurrences carries the text. Shared, each
    carries a reference instead, and the text appears once in its own block,
    joined to the other prompt parts by ``_PART_SEPARATOR``. A negative
    result means sharing makes the prompt longer.
    """
    inline_chars = count * len(text)
    block_chars = len(_shared_block("", index)) + len(text)
    shared_chars = count * len(_shared_reference(index)) + block_chars + len(_PART_SEPARATOR)
    return inline_chars - shared_chars


def _render_user_prompt(cluster: ContentCluster, shared: dict[str, int] | None = None) -> str:
    """Render every member; texts in ``shared`` are listed once and referenced."""
    parts: list[str] = []
    if shared:
        parts.append(
            "Identical content is listed once below. In a chunk's content, [SHARED CONTENT n] stands for "
            "the exact text of that shared block. Review every occurrence in its own file, section and "
            "line context. No chunks were omitted."
        )
        parts.extend(_shared_block(text, index) for text, index in shared.items())
    for i, chunk in enumerate(cluster.members, start=1):
        content = chunk.text
        if shared:
            heading_line, body = _split_own_heading(chunk)
            if body in shared:
                reference = _shared_reference(shared[body])
                content = f"{heading_line}\n{reference}" if heading_line else reference
        parts.append(
            f"---CHUNK {i}---\n"
            f"File: {chunk.source_file}\n"
            f"Section: {chunk.heading}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"Content:\n{content}"
        )

    parts.append(f"\nEmbedding similarity: {cluster.max_similarity:.3f}")
    parts.append("\nClassify this content overlap.")
    return _PART_SEPARATOR.join(parts)


def analyze_cluster(client: LLMClient, cluster: ContentCluster) -> LLMVerdict:
    """Run LLM analysis on a single content cluster.

    The reply is checked before it can become a finding: a verdict outside
    ``VALID_VERDICTS``, a confidence that is not a number from 0 to 1, or a
    rationale or suggestion that is not a bounded string raises
    :class:`~skillevaluator.inference.LLMClientError`.

    The verdict's ``model`` is the client's ``model_used`` right after the
    call. The client keeps it per thread, so concurrent clusters keep their
    own answering model.
    """
    user_prompt = build_user_prompt(cluster)
    data = client.extract_json_from_response(
        SYSTEM_PROMPT,
        user_prompt,
        response_schema=VERDICT_JSON_SCHEMA,
        schema_name="dedup_verdict",
    )
    return parse_bounded_llm_verdict(
        data,
        valid_verdicts=VALID_VERDICTS,
        context="Content deduplication LLM",
        model=getattr(client, "model_used", None),
    )


def verdict_to_severity(verdict: LLMVerdict) -> Severity:
    """Map an LLM verdict to a Finding severity level."""
    if verdict.verdict == "DUPLICATE":
        return Severity.HIGH if verdict.confidence >= 0.7 else Severity.MEDIUM
    return Severity.INFO
