# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context deduplication validator.

Orchestrates the full pipeline: collect → chunk → embed → cluster → LLM analyze → report.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from skillevaluator.constants import (
    CONTENT_DEDUP_EMBEDDING_BATCH_SIZE,
    CONTENT_DEDUP_LLM_REASONING_EFFORT,
    CONTENT_DEDUP_MAX_CHUNKS,
    CONTENT_DEDUP_MAX_LLM_CLUSTERS,
    CONTENT_DEDUP_SIMILARITY_THRESHOLD,
    CONTENT_DEDUP_TRIVIAL_DUP_MAX_CHARS,
)
from skillevaluator.deduplication.intra_skill.llm_analyzer import (
    SYSTEM_PROMPT,
    analyze_cluster,
    build_user_prompt,
    verdict_to_severity,
)
from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster, build_clusters
from skillevaluator.deduplication.utils.chunker import ContentChunk, chunk_file
from skillevaluator.deduplication.utils.skill_collector import CollectedFile, SkillCollectionError, collect_files
from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError, validate_embedding_vector
from skillevaluator.inference import LLMClient, LLMClientError
from skillevaluator.inference.diagnostics import llm_failure_diagnostic, safe_llm_labels
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.utils.tier2_paths import safe_path_label
from skillevaluator.validators.base import ValidatorBase

logger = logging.getLogger(__name__)


# A line that is a comment (#, ;, //, --, !) or a `key = value` / `key: value`
# config assignment. Used to recognize repeated config/comment snippets.
_COMMENT_PREFIXES = ("#", ";", "//", "--", "!")
_CONFIG_KV_RE = re.compile(r"^[\w.\-/]+\s*[=:]\s*\S")
CONTENT_DEDUP_MAX_SCALAR_COMPARISONS = 25_000_000
CONTENT_DEDUP_MAX_CLUSTER_MEMBERS = 64
CONTENT_DEDUP_MAX_LLM_PROMPT_CHARS = 64_000
CONTENT_DEDUP_MAX_TOTAL_LLM_PROMPT_CHARS = 1_000_000

# Scan label recorded when a work limit stops duplicate analysis early.
_WORK_LIMIT_SCAN = "context-deduplication"


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


def _mark_work_limit_incomplete(
    result: ValidationResult,
    check_name: str,
    message: str,
    suggestion: str,
    **details: object,
) -> None:
    """Record a tripped work limit as incomplete coverage, not a duplicate finding.

    The scan stopped before duplicate analysis finished, so it can neither
    report nor rule out duplicated content. A finding would misreport the skill
    and could be waived by a policy severity override; the incomplete status
    cannot. ``metadata["work_limit"]`` keeps the limit for JSON reports.
    """
    result.mark_scan_incomplete(_WORK_LIMIT_SCAN)
    result.metadata["work_limit"] = {"check_name": check_name, **details}
    result.add_error(f"Tier 2 scan incomplete: {message} {suggestion}")


class IntraSkillValidator(ValidatorBase):
    """Detect redundant content within a skill directory."""

    def __init__(
        self,
        threshold: float = CONTENT_DEDUP_SIMILARITY_THRESHOLD,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        max_llm_clusters: int | None = None,
        _max_scalar_comparisons: int | None = None,
        _max_total_prompt_chars: int | None = None,
    ) -> None:
        self._threshold = threshold
        # None defers to provider resolution (SKILL_EVAL_EMBEDDING_MODEL);
        # pinning SIMILARITY_DEFAULT_MODEL here would override the env var.
        self._embedding_model = embedding_model
        self._llm_model = llm_model
        max_llm_clusters = CONTENT_DEDUP_MAX_LLM_CLUSTERS if max_llm_clusters is None else max_llm_clusters
        if type(max_llm_clusters) is not int or not 1 <= max_llm_clusters <= CONTENT_DEDUP_MAX_LLM_CLUSTERS:
            raise ValueError(f"max_llm_clusters must be within [1, {CONTENT_DEDUP_MAX_LLM_CLUSTERS}]")
        self._max_llm_clusters = max_llm_clusters
        self._max_scalar_comparisons = (
            CONTENT_DEDUP_MAX_SCALAR_COMPARISONS if _max_scalar_comparisons is None else _max_scalar_comparisons
        )
        self._max_total_prompt_chars = (
            CONTENT_DEDUP_MAX_TOTAL_LLM_PROMPT_CHARS if _max_total_prompt_chars is None else _max_total_prompt_chars
        )
        for value in (self._max_scalar_comparisons, self._max_total_prompt_chars):
            if type(value) is not int or value < 1:
                raise ValueError("Private comparison budgets must be positive integers")

    @property
    def name(self) -> str:
        return "Context Deduplication"

    @property
    def description(self) -> str:
        return "Detect redundant content within a skill directory"

    def validate(self, skill_path: Path, *, _collected_files: list[CollectedFile] | None = None) -> ValidationResult:
        """Run context deduplication on a single skill directory."""
        result = ValidationResult(
            validator_name=self.name,
            validator_description=self.description,
        )

        # Step 1: Collect files
        logger.info("Collecting files from %s...", safe_path_label(skill_path))
        try:
            collected = collect_files(skill_path) if _collected_files is None else _collected_files
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
                _mark_work_limit_incomplete(
                    result,
                    "chunk_count_limit",
                    f"the skill produced more than {CONTENT_DEDUP_MAX_CHUNKS} content chunks "
                    f"(the limit was reached in {f.rel_path}).",
                    "Reduce or split the skill content before running Tier 2.",
                    file_path=f.rel_path,
                    actual=prospective_count,
                    limit=CONTENT_DEDUP_MAX_CHUNKS,
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

            vector_dimension: int | None = None
            pair_count = len(all_chunks) * (len(all_chunks) - 1) // 2
            for i in range(0, len(texts), CONTENT_DEDUP_EMBEDDING_BATCH_SIZE):
                batch = texts[i : i + CONTENT_DEDUP_EMBEDDING_BATCH_SIZE]
                vectors = client.embed(batch)
                if len(vectors) != len(batch):
                    raise SimilarityConfigError(
                        f"Embedding provider returned {len(vectors)} vectors for {len(batch)} chunks in a batch."
                    )
                for chunk, vector in zip(all_chunks[i : i + len(batch)], vectors, strict=True):
                    vector_dimension = validate_embedding_vector(vector, vector_dimension, context="Embedding provider")
                    chunk.embedding = vector
                # The first validated response establishes the dimension. Refuse
                # excess pairwise work before requesting another provider batch.
                scalar_work = pair_count * (vector_dimension or 0)
                if scalar_work > self._max_scalar_comparisons:
                    _mark_work_limit_incomplete(
                        result,
                        "scalar_comparison_limit",
                        f"scalar comparison work of {scalar_work} exceeds the limit of {self._max_scalar_comparisons}.",
                        "Reduce or split the skill content before running Tier 2.",
                        pair_count=pair_count,
                        vector_dimension=vector_dimension,
                        scalar_work=scalar_work,
                        limit=self._max_scalar_comparisons,
                    )
                    return result

            logger.info("Embedding complete")

        except SimilarityConfigError as e:
            result.mark_scan_incomplete("embedding-provider")
            result.add_error(f"Embedding provider error: {e}")
            return result

        # Step 4: Cluster by similarity
        logger.info("Clustering %d chunks (threshold: %.2f)...", len(all_chunks), self._threshold)
        clusters = build_clusters(all_chunks, self._threshold)
        logger.info("Found %d cluster(s)", len(clusters))

        if len(clusters) > self._max_llm_clusters:
            _mark_work_limit_incomplete(
                result,
                "llm_cluster_count_limit",
                f"{len(clusters)} content clusters require LLM review, more than the limit of "
                f"{self._max_llm_clusters}.",
                "Reduce duplicated content or split the skill before rerunning Tier 2.",
                actual=len(clusters),
                limit=self._max_llm_clusters,
            )
            return result

        if not clusters:
            result.add_success(
                "context_dedup",
                f"No redundant content detected (threshold: {self._threshold})",
            )
            return result

        total_prompt_chars = 0
        for cluster in clusters:
            prompt_chars = len(SYSTEM_PROMPT) + len(build_user_prompt(cluster))
            total_prompt_chars += prompt_chars
            limits = (
                (
                    len(cluster.members),
                    CONTENT_DEDUP_MAX_CLUSTER_MEMBERS,
                    "llm_cluster_member_limit",
                    "a content cluster has {actual} members, more than the limit of {limit} for LLM review.",
                ),
                (
                    prompt_chars,
                    CONTENT_DEDUP_MAX_LLM_PROMPT_CHARS,
                    "llm_prompt_size_limit",
                    "an LLM review prompt of {actual} characters exceeds the limit of {limit}.",
                ),
                (
                    total_prompt_chars,
                    self._max_total_prompt_chars,
                    "llm_total_prompt_size_limit",
                    "LLM review prompts total {actual} characters, more than the limit of {limit}.",
                ),
            )
            for actual, limit, check, message in limits:
                if actual > limit:
                    _mark_work_limit_incomplete(
                        result,
                        check,
                        message.format(actual=actual, limit=limit),
                        "Reduce duplicated content or split the skill before rerunning Tier 2.",
                        actual=actual,
                        limit=limit,
                    )
                    return result

        # Step 5: LLM analysis for each cluster (concurrent)
        logger.info("Running LLM analysis on %d cluster(s) concurrently...", len(clusters))
        try:
            llm = LLMClient(model=self._llm_model, reasoning_effort=CONTENT_DEDUP_LLM_REASONING_EFFORT)
            config = llm._resolved_config()
            provider, model = safe_llm_labels(config.provider, config.model)
        except LLMClientError:
            result.mark_scan_incomplete("deduplication-llm")
            result.add_error("LLM analysis could not start. Check the LLM provider configuration, then rerun Tier 2.")
            return result

        def analyze_one(cluster):
            logger.info(
                "  [thread] Analyzing cluster (%d chunks, max similarity: %.3f)...",
                len(cluster.members),
                cluster.max_similarity,
            )
            try:
                verdict = analyze_cluster(llm, cluster)
                logger.info("  [thread] Verdict: %s (confidence: %.2f)", verdict.verdict, verdict.confidence)
                return (cluster, verdict, None)
            except Exception as exc:
                # A provider exception or malformed model response is missing
                # evidence, not a duplicate-content finding. Never log raw bodies.
                return (cluster, None, llm_failure_diagnostic(exc))

        cluster_results = []
        max_workers = min(len(clusters), 5)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_one, c): c for c in clusters}
            for future in as_completed(futures):
                cluster_results.append(future.result())

        failures = Counter(diagnostic for _, _, diagnostic in cluster_results if diagnostic is not None)
        failed_count = sum(failures.values())
        result.metadata["llm_analysis"] = {
            "provider": provider,
            "model": model,
            "clusters_total": len(clusters),
            "clusters_completed": len(clusters) - failed_count,
            "clusters_failed": failed_count,
            "failures": [{"count": count, "diagnostic": diagnostic} for diagnostic, count in sorted(failures.items())],
        }
        if failures:
            result.mark_scan_incomplete("deduplication-llm")
            summary = (
                f"LLM analysis did not complete for {failed_count} of {len(clusters)} content clusters "
                f"(provider: {provider}; model: {model})."
            )
            details = " ".join(f"{count} cluster(s): {diagnostic}" for diagnostic, count in sorted(failures.items()))
            result.add_error(f"{summary} {details}")
            logger.error("%s %s", summary, details)

        # Process results
        for cluster, verdict, _ in cluster_results:
            if verdict is None:
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
