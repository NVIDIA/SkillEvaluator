# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context deduplication validator.

Orchestrates the full pipeline: collect → chunk → embed → cluster → LLM analyze → report.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from skillevaluator.constants import (
    CONTENT_DEDUP_EMBEDDING_BATCH_SIZE,
    CONTENT_DEDUP_MAX_CHUNKS,
    CONTENT_DEDUP_MAX_CLUSTER_MEMBERS,
    CONTENT_DEDUP_MAX_LLM_CLUSTERS,
    CONTENT_DEDUP_MAX_LLM_PROMPT_CHARS,
    CONTENT_DEDUP_MAX_SCALAR_COMPARISONS,
    CONTENT_DEDUP_MAX_TOTAL_LLM_PROMPT_CHARS,
    CONTENT_DEDUP_SIMILARITY_THRESHOLD,
    CONTENT_DEDUP_TRIVIAL_DUP_MAX_CHARS,
    LLM_VERIFY_MAX_TOKENS,
)
from skillevaluator.deduplication.intra_skill.llm_analyzer import (
    analyze_cluster,
    build_user_prompt,
    verdict_to_severity,
)
from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster, build_clusters
from skillevaluator.deduplication.utils.chunker import ContentChunk, chunk_file
from skillevaluator.deduplication.utils.skill_collector import SkillCollectionError, collect_files
from skillevaluator.embedding.client import (
    EmbeddingClient,
    SimilarityConfigError,
    validate_embedding_vector,
    validate_similarity_threshold,
)
from skillevaluator.inference import LLMClient, LLMClientError
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.utils.tier2_paths import safe_path_label
from skillevaluator.validators.base import ValidatorBase

logger = logging.getLogger(__name__)


# A line that is a comment (#, ;, //, --, !) or a `key = value` / `key: value`
# config assignment. Used to recognize repeated config/comment snippets.
_COMMENT_PREFIXES = ("#", ";", "//", "--", "!")
_CONFIG_KV_RE = re.compile(r"^[\w.\-/]+\s*[=:]\s*\S")


def _is_comment_or_config_line(line: str) -> bool:
    """Return True for comment lines or simple ``key=value``/``key: value`` config lines."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(_COMMENT_PREFIXES):
        return True
    return bool(_CONFIG_KV_RE.match(stripped))


def _chunk_is_short_config(chunk: ContentChunk) -> bool:
    """Return True if a chunk is a short block dominated by comment/config-style lines.

    The chunk's own heading line and code-fence markers are ignored; a chunk
    qualifies when at least 60% of the remaining content lines look like
    comments or config assignments (or there are none beyond the heading).
    """
    if chunk.char_count > CONTENT_DEDUP_TRIVIAL_DUP_MAX_CHARS:
        return False
    content_lines = [
        stripped
        for raw in chunk.text.splitlines()
        if (stripped := raw.strip()) and stripped != chunk.heading and not stripped.startswith(("```", "~~~"))
    ]
    if not content_lines:
        return True
    config = sum(1 for line in content_lines if _is_comment_or_config_line(line))
    return config / len(content_lines) >= 0.6


def _is_trivial_intra_file_duplicate(cluster: ContentCluster) -> bool:
    """Return True for single-file duplicates made only of short comment/config chunks.

    These are legitimately-repeated config snippets (the same comment line
    recurring inside one reference file), not the cross-file context bloat that
    deduplication is meant to surface, so they should not be a HIGH finding.
    """
    if cluster.cross_file:
        return False
    return all(_chunk_is_short_config(member) for member in cluster.members)


class IntraSkillValidator(ValidatorBase):
    """Detect redundant content within a skill directory."""

    def __init__(
        self,
        threshold: float = CONTENT_DEDUP_SIMILARITY_THRESHOLD,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        max_llm_clusters: int | None = None,
    ) -> None:
        self._threshold = validate_similarity_threshold(threshold, context="Content deduplication")
        max_llm_clusters = CONTENT_DEDUP_MAX_LLM_CLUSTERS if max_llm_clusters is None else max_llm_clusters
        if type(max_llm_clusters) is not int or not 1 <= max_llm_clusters <= CONTENT_DEDUP_MAX_LLM_CLUSTERS:
            raise ValueError(f"max_llm_clusters must be within [1, {CONTENT_DEDUP_MAX_LLM_CLUSTERS}]")
        # None defers to provider resolution (SKILL_EVAL_EMBEDDING_MODEL);
        # pinning SIMILARITY_DEFAULT_MODEL here would override the env var.
        self._embedding_model = embedding_model
        self._llm_model = llm_model
        self._max_llm_clusters = max_llm_clusters

    @property
    def name(self) -> str:
        return "Context Deduplication"

    @property
    def description(self) -> str:
        return "Detect redundant content within a skill directory"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Run context deduplication on a single skill directory."""
        report_path = skill_path.name or "."
        result = ValidationResult(
            validator_name=self.name,
            validator_description=self.description,
        )

        # Step 1: Collect files
        logger.info("Collecting files from %s...", safe_path_label(skill_path))
        try:
            collected = collect_files(skill_path)
        except SkillCollectionError as e:
            result.add_finding(
                Finding(
                    category="CONTENT_DEDUP",
                    severity=Severity.CRITICAL,
                    check_name=e.check_name,
                    message=str(e),
                    file_path=e.rel_path,
                    suggestion=e.suggestion,
                    metadata=e.metadata,
                )
            )
            return result
        logger.info("Collected %d file(s)", len(collected))
        result.add_success(
            "file_collection",
            f"Collected {len(collected)} file(s)",
            file_count=len(collected),
        )

        if not collected:
            result.add_success("context_dedup", "No text files found in skill directory")
            return result

        # Step 2: Chunk all files
        logger.info("Chunking %d file(s)...", len(collected))
        all_chunks: list[ContentChunk] = []
        for f in collected:
            file_chunks = chunk_file(f)
            prospective_count = len(all_chunks) + len(file_chunks)
            if prospective_count > CONTENT_DEDUP_MAX_CHUNKS:
                result.add_finding(
                    Finding(
                        category="CONTENT_DEDUP",
                        severity=Severity.CRITICAL,
                        check_name="chunk_count_limit",
                        message=f"Tier 2 produced more than {CONTENT_DEDUP_MAX_CHUNKS} content chunks.",
                        file_path=f.rel_path,
                        suggestion="Reduce or split the skill content before running Tier 2.",
                        metadata={"actual": prospective_count, "limit": CONTENT_DEDUP_MAX_CHUNKS},
                    )
                )
                return result
            all_chunks.extend(file_chunks)

        logger.info("Extracted %d chunk(s)", len(all_chunks))
        result.add_success(
            "chunking",
            f"Extracted {len(all_chunks)} chunk(s)",
            chunk_count=len(all_chunks),
        )

        if len(all_chunks) < 2:
            result.add_success("context_dedup", "Not enough content to compare")
            return result

        # Step 3: Batch embed all chunks
        logger.info("Embedding %d chunk(s) via the configured public provider...", len(all_chunks))
        try:
            client = EmbeddingClient(model=self._embedding_model)
            texts = [c.text for c in all_chunks]

            pair_count = len(all_chunks) * (len(all_chunks) - 1) // 2
            vector_dimension: int | None = None
            for i in range(0, len(texts), CONTENT_DEDUP_EMBEDDING_BATCH_SIZE):
                batch = texts[i : i + CONTENT_DEDUP_EMBEDDING_BATCH_SIZE]
                logger.info(
                    "  Embedding batch %d/%d (%d chunks)...",
                    i // CONTENT_DEDUP_EMBEDDING_BATCH_SIZE + 1,
                    (len(texts) - 1) // CONTENT_DEDUP_EMBEDDING_BATCH_SIZE + 1,
                    len(batch),
                )
                batch_embeddings = client.embed(batch)
                if len(batch_embeddings) != len(batch):
                    raise SimilarityConfigError(
                        f"Embedding provider returned {len(batch_embeddings)} vectors for a batch of "
                        f"{len(batch)} chunks."
                    )
                for chunk, emb in zip(all_chunks[i : i + len(batch)], batch_embeddings, strict=True):
                    vector_dimension = validate_embedding_vector(
                        emb,
                        vector_dimension,
                        context="Embedding provider",
                    )
                    chunk.embedding = emb

                scalar_work = pair_count * (vector_dimension or 0)
                if scalar_work > CONTENT_DEDUP_MAX_SCALAR_COMPARISONS:
                    result.add_finding(
                        Finding(
                            category="CONTENT_DEDUP",
                            severity=Severity.CRITICAL,
                            check_name="scalar_comparison_limit",
                            message=(
                                "Tier 2 scalar comparison work exceeds the configured limit "
                                f"({CONTENT_DEDUP_MAX_SCALAR_COMPARISONS})."
                            ),
                            file_path=report_path,
                            suggestion="Reduce or split the skill content before running Tier 2.",
                            metadata={
                                "pair_count": pair_count,
                                "vector_dimension": vector_dimension,
                                "scalar_work": scalar_work,
                                "limit": CONTENT_DEDUP_MAX_SCALAR_COMPARISONS,
                            },
                        )
                    )
                    return result

            logger.info("Embedding complete")

        except SimilarityConfigError as e:
            result.add_finding(
                Finding(
                    category="CONTENT_DEDUP",
                    severity=Severity.CRITICAL,
                    check_name="embedding_error",
                    message=f"Embedding provider error: {e}",
                    file_path=report_path,
                )
            )
            return result

        # Step 4: Cluster by similarity
        logger.info("Clustering %d chunks (threshold: %.2f)...", len(all_chunks), self._threshold)
        clusters = build_clusters(all_chunks, self._threshold)
        logger.info("Found %d cluster(s)", len(clusters))

        if len(clusters) > self._max_llm_clusters:
            result.add_finding(
                Finding(
                    category="CONTENT_DEDUP",
                    severity=Severity.CRITICAL,
                    check_name="llm_cluster_count_limit",
                    message=f"Tier 2 found more than {self._max_llm_clusters} clusters requiring LLM review.",
                    file_path=report_path,
                    suggestion="Reduce duplicated content or split the skill before rerunning Tier 2.",
                    metadata={"actual": len(clusters), "limit": self._max_llm_clusters},
                )
            )
            return result

        cluster_prompts: dict[int, str] = {}
        total_prompt_chars = 0
        for cluster in clusters:
            if len(cluster.members) > CONTENT_DEDUP_MAX_CLUSTER_MEMBERS:
                result.add_finding(
                    Finding(
                        category="CONTENT_DEDUP",
                        severity=Severity.CRITICAL,
                        check_name="llm_cluster_member_limit",
                        message="A Tier 2 cluster exceeds the LLM member limit.",
                        file_path=report_path,
                        metadata={"actual": len(cluster.members), "limit": CONTENT_DEDUP_MAX_CLUSTER_MEMBERS},
                    )
                )
                return result
            prompt = build_user_prompt(cluster)
            if len(prompt) > CONTENT_DEDUP_MAX_LLM_PROMPT_CHARS:
                result.add_finding(
                    Finding(
                        category="CONTENT_DEDUP",
                        severity=Severity.CRITICAL,
                        check_name="llm_prompt_size_limit",
                        message="A Tier 2 cluster exceeds the LLM prompt character limit.",
                        file_path=report_path,
                        metadata={"actual": len(prompt), "limit": CONTENT_DEDUP_MAX_LLM_PROMPT_CHARS},
                    )
                )
                return result
            total_prompt_chars += len(prompt)
            if total_prompt_chars > CONTENT_DEDUP_MAX_TOTAL_LLM_PROMPT_CHARS:
                result.add_finding(
                    Finding(
                        category="CONTENT_DEDUP",
                        severity=Severity.CRITICAL,
                        check_name="llm_total_prompt_size_limit",
                        message="Tier 2 aggregate LLM prompt characters exceed the configured limit.",
                        file_path=report_path,
                        metadata={
                            "actual": total_prompt_chars,
                            "limit": CONTENT_DEDUP_MAX_TOTAL_LLM_PROMPT_CHARS,
                        },
                    )
                )
                return result
            cluster_prompts[id(cluster)] = prompt

        if not clusters:
            result.add_success(
                "context_dedup",
                f"No redundant content detected (threshold: {self._threshold})",
            )
            return result

        # Step 5: LLM analysis for each cluster (concurrent)
        logger.info("Running LLM analysis on %d cluster(s) concurrently...", len(clusters))
        llm = LLMClient(model=self._llm_model, max_tokens=LLM_VERIFY_MAX_TOKENS)

        def analyze_one(cluster):
            logger.info(
                "  [thread] Analyzing cluster (%d chunks, max similarity: %.3f)...",
                len(cluster.members),
                cluster.max_similarity,
            )
            try:
                verdict = analyze_cluster(llm, cluster, user_prompt=cluster_prompts[id(cluster)])
                logger.info("  [thread] Verdict: %s (confidence: %.2f)", verdict.verdict, verdict.confidence)
                return (cluster, verdict)
            except LLMClientError as e:
                logger.error("  [thread] LLM failed: %s", e)
                return (cluster, None)

        cluster_results = []
        max_workers = min(len(clusters), 5)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_one, c): c for c in clusters}
            for future in as_completed(futures):
                cluster_results.append(future.result())

        # Process results
        for cluster, verdict in cluster_results:
            if verdict is None:
                result.add_finding(
                    Finding(
                        category="CONTENT_DEDUP",
                        severity=Severity.CRITICAL,
                        check_name="llm_error",
                        message="LLM analysis failed for a content cluster",
                        file_path=report_path,
                    )
                )
                continue

            severity = verdict_to_severity(verdict)

            # Only report actionable findings (DUPLICATE)
            if verdict.verdict != "DUPLICATE":
                continue

            # Short, legitimately-repeated comment/config snippets inside a
            # single file (e.g. a recurring `# default pts-tolerance is 60 ms.`
            # config line) are advisory at most: cap them at LOW so they no
            # longer fail the skill, while genuine large-block or cross-file
            # duplication keeps its HIGH/MEDIUM severity.
            if severity in (Severity.HIGH, Severity.MEDIUM) and _is_trivial_intra_file_duplicate(cluster):
                severity = Severity.LOW

            distinct_files = sorted({c.source_file for c in cluster.members})

            locations = []
            for c in cluster.members:
                locations.append(f'"{c.heading}" in {c.source_file} (lines {c.start_line}-{c.end_line})')
            location_text = "\n  vs ".join(locations)

            if len(distinct_files) == 1:
                message = f"Duplicate content found within {distinct_files[0]}:\n  {location_text}"
            else:
                files_str = " and ".join(distinct_files)
                message = f"Duplicate content found across {files_str}:\n  {location_text}"

            result.add_finding(
                Finding(
                    category=verdict.verdict,
                    severity=severity,
                    check_name=verdict.verdict.lower(),
                    message=message,
                    file_path=cluster.members[0].source_file,
                    line_number=cluster.members[0].start_line,
                    suggestion=verdict.suggestion,
                    metadata={
                        "reasoning": verdict.reasoning,
                        "confidence": verdict.confidence,
                        "source_files": distinct_files,
                    },
                )
            )

        return result
