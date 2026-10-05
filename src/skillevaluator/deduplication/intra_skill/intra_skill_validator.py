# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context deduplication validator.

Orchestrates the full pipeline: collect → chunk → embed → cluster → LLM analyze → report.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
from skillevaluator.deduplication.result_status import mark_security_failure
from skillevaluator.deduplication.utils.chunker import ContentChunk, chunk_file
from skillevaluator.deduplication.utils.skill_collector import CollectedFile, SkillCollectionError, collect_files
from skillevaluator.embedding.client import (
    EmbeddingClient,
    SimilarityConfigError,
    validate_embedding_vector,
    validate_similarity_threshold,
)
from skillevaluator.inference import LLMClient, LLMClientError, LLMVerdict
from skillevaluator.inference.diagnostics import llm_failure_diagnostic, safe_llm_labels
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.utils.tier2_paths import safe_path_label
from skillevaluator.validators.base import ValidatorBase

logger = logging.getLogger(__name__)


# A line that is a comment (#, ;, //, --, !) or a `key = value` / `key: value`
# config assignment. Used to recognize repeated config/comment snippets.
_COMMENT_PREFIXES = ("#", ";", "//", "--", "!")
_CONFIG_KV_RE = re.compile(r"^[\w.\-/]+\s*[=:]\s*\S")

_REDUCE_SKILL_CONTENT = "Reduce or split the skill content before running Tier 2."

# Collection errors raised for links, hard links, special files, or roots that
# cannot be read safely. Size, count, and encoding limits are not included.
_UNSAFE_INPUT_CHECKS = frozenset(
    {
        "invalid_root",
        "path_access_error",
        "secure_open_unavailable",
        "unsafe_hardlink",
        "unsafe_path",
        "unsafe_root",
    }
)


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
        """Run context deduplication on a single skill directory.

        The phases run in order and stop at the first one that cannot continue:
        collect, chunk, embed, cluster, prepare the LLM prompts, review.
        """
        report_path = skill_path.name or "."
        result = ValidationResult(
            validator_name=self.name,
            validator_description=self.description,
        )
        collected = self._collect(skill_path, result)
        if collected is None:
            return result
        chunks = self._chunk(collected, result)
        if chunks is None or not self._embed(chunks, result, report_path):
            return result

        logger.info("Clustering %d chunks (threshold: %.2f)...", len(chunks), self._threshold)
        clusters = build_clusters(chunks, self._threshold)
        logger.info("Found %d cluster(s)", len(clusters))
        reviews = self._prepare_prompts(clusters, result, report_path)
        if reviews is None:
            return result
        if not reviews:
            result.add_success(
                "context_dedup",
                f"No redundant content detected (threshold: {self._threshold})",
            )
            return result
        self._review_clusters(reviews, result)
        return result

    @staticmethod
    def _collect(skill_path: Path, result: ValidationResult) -> list[CollectedFile] | None:
        """Collect the skill's text files; ``None`` when there is nothing to compare."""
        logger.info("Collecting files from %s...", safe_path_label(skill_path))
        try:
            collected = collect_files(skill_path)
        except SkillCollectionError as e:
            _add_stop_finding(
                result,
                e.check_name,
                str(e),
                file_path=e.rel_path,
                metadata=e.metadata,
                suggestion=e.suggestion,
            )
            if e.check_name in _UNSAFE_INPUT_CHECKS:
                # Unsafe input means the check could not run safely. Callers
                # that cap deduplication findings as advisory (plugin Tier 2)
                # must keep this result blocking.
                mark_security_failure(result)
            return None
        logger.info("Collected %d file(s)", len(collected))
        result.add_success(
            "file_collection",
            f"Collected {len(collected)} file(s)",
            file_count=len(collected),
        )
        if not collected:
            result.add_success("context_dedup", "No text files found in skill directory")
            return None
        return collected

    @staticmethod
    def _chunk(collected: list[CollectedFile], result: ValidationResult) -> list[ContentChunk] | None:
        """Split the files into chunks; ``None`` above the chunk limit or below two chunks."""
        logger.info("Chunking %d file(s)...", len(collected))
        chunks: list[ContentChunk] = []
        for collected_file in collected:
            file_chunks = chunk_file(collected_file)
            prospective_count = len(chunks) + len(file_chunks)
            if prospective_count > CONTENT_DEDUP_MAX_CHUNKS:
                _add_stop_finding(
                    result,
                    "chunk_count_limit",
                    f"Tier 2 produced more than {CONTENT_DEDUP_MAX_CHUNKS} content chunks.",
                    file_path=collected_file.rel_path,
                    metadata={"actual": prospective_count, "limit": CONTENT_DEDUP_MAX_CHUNKS},
                    suggestion=_REDUCE_SKILL_CONTENT,
                )
                return None
            chunks.extend(file_chunks)

        logger.info("Extracted %d chunk(s)", len(chunks))
        result.add_success(
            "chunking",
            f"Extracted {len(chunks)} chunk(s)",
            chunk_count=len(chunks),
        )
        if len(chunks) < 2:
            result.add_success("context_dedup", "Not enough content to compare")
            return None
        return chunks

    def _embed(self, chunks: list[ContentChunk], result: ValidationResult, report_path: str) -> bool:
        """Embed every chunk in batches; ``False`` on a provider error or above the scalar work limit."""
        logger.info("Embedding %d chunk(s) via the configured public provider...", len(chunks))
        pair_count = len(chunks) * (len(chunks) - 1) // 2
        batch_count = (len(chunks) - 1) // CONTENT_DEDUP_EMBEDDING_BATCH_SIZE + 1
        try:
            client = EmbeddingClient(model=self._embedding_model)
            vector_dimension: int | None = None
            for start in range(0, len(chunks), CONTENT_DEDUP_EMBEDDING_BATCH_SIZE):
                batch = chunks[start : start + CONTENT_DEDUP_EMBEDDING_BATCH_SIZE]
                logger.info(
                    "  Embedding batch %d/%d (%d chunks)...",
                    start // CONTENT_DEDUP_EMBEDDING_BATCH_SIZE + 1,
                    batch_count,
                    len(batch),
                )
                batch_embeddings = client.embed([chunk.text for chunk in batch])
                if len(batch_embeddings) != len(batch):
                    raise SimilarityConfigError(
                        f"Embedding provider returned {len(batch_embeddings)} vectors for a batch of "
                        f"{len(batch)} chunks."
                    )
                for chunk, embedding in zip(batch, batch_embeddings, strict=True):
                    vector_dimension = validate_embedding_vector(
                        embedding,
                        vector_dimension,
                        context="Embedding provider",
                    )
                    chunk.embedding = embedding

                if start == 0:
                    # The first batch fixes the vector width, so the pairwise
                    # work is known before any further batch is requested.
                    scalar_work = pair_count * (vector_dimension or 0)
                    if scalar_work > CONTENT_DEDUP_MAX_SCALAR_COMPARISONS:
                        _add_stop_finding(
                            result,
                            "scalar_comparison_limit",
                            (
                                "Tier 2 scalar comparison work exceeds the configured limit "
                                f"({CONTENT_DEDUP_MAX_SCALAR_COMPARISONS})."
                            ),
                            file_path=report_path,
                            metadata={
                                "pair_count": pair_count,
                                "vector_dimension": vector_dimension,
                                "scalar_work": scalar_work,
                                "limit": CONTENT_DEDUP_MAX_SCALAR_COMPARISONS,
                            },
                            suggestion=_REDUCE_SKILL_CONTENT,
                        )
                        return False

            logger.info("Embedding complete")
        except SimilarityConfigError as e:
            result.mark_scan_incomplete("embedding-provider")
            result.add_error(f"Embedding provider error: {e}")
            return False
        return True

    def _prepare_prompts(
        self,
        clusters: list[ContentCluster],
        result: ValidationResult,
        report_path: str,
    ) -> list[tuple[ContentCluster, str]] | None:
        """Pair each cluster with its LLM prompt; ``None`` when a review limit is exceeded."""
        if len(clusters) > self._max_llm_clusters:
            _add_stop_finding(
                result,
                "llm_cluster_count_limit",
                f"Tier 2 found more than {self._max_llm_clusters} clusters requiring LLM review.",
                file_path=report_path,
                metadata={"actual": len(clusters), "limit": self._max_llm_clusters},
                suggestion="Reduce duplicated content or split the skill before rerunning Tier 2.",
            )
            return None

        reviews: list[tuple[ContentCluster, str]] = []
        total_prompt_chars = 0
        for cluster in clusters:
            if len(cluster.members) > CONTENT_DEDUP_MAX_CLUSTER_MEMBERS:
                _add_stop_finding(
                    result,
                    "llm_cluster_member_limit",
                    "A Tier 2 cluster exceeds the LLM member limit.",
                    file_path=report_path,
                    metadata={"actual": len(cluster.members), "limit": CONTENT_DEDUP_MAX_CLUSTER_MEMBERS},
                )
                return None
            prompt = build_user_prompt(cluster)
            if len(prompt) > CONTENT_DEDUP_MAX_LLM_PROMPT_CHARS:
                _add_stop_finding(
                    result,
                    "llm_prompt_size_limit",
                    "A Tier 2 cluster exceeds the LLM prompt character limit.",
                    file_path=report_path,
                    metadata={"actual": len(prompt), "limit": CONTENT_DEDUP_MAX_LLM_PROMPT_CHARS},
                )
                return None
            total_prompt_chars += len(prompt)
            if total_prompt_chars > CONTENT_DEDUP_MAX_TOTAL_LLM_PROMPT_CHARS:
                _add_stop_finding(
                    result,
                    "llm_total_prompt_size_limit",
                    "Tier 2 aggregate LLM prompt characters exceed the configured limit.",
                    file_path=report_path,
                    metadata={"actual": total_prompt_chars, "limit": CONTENT_DEDUP_MAX_TOTAL_LLM_PROMPT_CHARS},
                )
                return None
            reviews.append((cluster, prompt))
        return reviews

    def _review_clusters(self, reviews: list[tuple[ContentCluster, str]], result: ValidationResult) -> None:
        """Ask the LLM about each cluster concurrently and report the duplicates."""
        logger.info("Running LLM analysis on %d cluster(s) concurrently...", len(reviews))
        try:
            llm = LLMClient(model=self._llm_model, max_tokens=LLM_VERIFY_MAX_TOKENS)
            config = llm._resolved_config()
            provider, model = safe_llm_labels(config.provider, config.model)
        except LLMClientError:
            result.mark_scan_incomplete("deduplication-llm")
            result.add_error("LLM analysis could not start. Check the LLM provider configuration, then rerun Tier 2.")
            return

        def analyze_one(cluster: ContentCluster, prompt: str) -> tuple[ContentCluster, LLMVerdict | None, str | None]:
            logger.info(
                "  [thread] Analyzing cluster (%d chunks, max similarity: %.3f)...",
                len(cluster.members),
                cluster.max_similarity,
            )
            try:
                verdict = analyze_cluster(llm, cluster, user_prompt=prompt)
                logger.info("  [thread] Verdict: %s (confidence: %.2f)", verdict.verdict, verdict.confidence)
                return (cluster, verdict, None)
            except Exception as exc:
                # A provider exception or malformed model response is missing
                # evidence, not a duplicate-content finding. Never log raw bodies.
                return (cluster, None, llm_failure_diagnostic(exc))

        with ThreadPoolExecutor(max_workers=min(len(reviews), 5)) as executor:
            futures = [executor.submit(analyze_one, cluster, prompt) for cluster, prompt in reviews]
            # Report in cluster order (most similar first), not completion order.
            cluster_results = [future.result() for future in futures]

        failures = Counter(diagnostic for _, _, diagnostic in cluster_results if diagnostic is not None)
        failed_count = sum(failures.values())
        result.metadata["llm_analysis"] = {
            "provider": provider,
            "model": model,
            "clusters_total": len(reviews),
            "clusters_completed": len(reviews) - failed_count,
            "clusters_failed": failed_count,
            "failures": [{"count": count, "diagnostic": diagnostic} for diagnostic, count in sorted(failures.items())],
        }
        if failures:
            result.mark_scan_incomplete("deduplication-llm")
            summary = (
                f"LLM analysis did not complete for {failed_count} of {len(reviews)} content clusters "
                f"(provider: {provider}; model: {model})."
            )
            details = " ".join(f"{count} cluster(s): {diagnostic}" for diagnostic, count in sorted(failures.items()))
            result.add_error(f"{summary} {details}")
            logger.error("%s %s", summary, details)

        for cluster, verdict, _ in cluster_results:
            # Only DUPLICATE verdicts are actionable findings.
            if verdict is not None and verdict.verdict == "DUPLICATE":
                result.add_finding(_duplicate_finding(cluster, verdict))


def _add_stop_finding(
    result: ValidationResult,
    check_name: str,
    message: str,
    *,
    file_path: str,
    metadata: dict[str, Any],
    suggestion: str | None = None,
) -> None:
    """Record the CRITICAL finding for unsafe input or a work limit that stopped the check."""
    result.add_finding(
        Finding(
            category="CONTENT_DEDUP",
            severity=Severity.CRITICAL,
            check_name=check_name,
            message=message,
            file_path=file_path,
            suggestion=suggestion,
            metadata=metadata,
        )
    )


def _duplicate_finding(cluster: ContentCluster, verdict: LLMVerdict) -> Finding:
    """Build the finding for one cluster the LLM judged a duplicate."""
    severity = verdict_to_severity(verdict)
    # Short, legitimately-repeated comment/config snippets inside a single
    # file (e.g. a recurring `# default pts-tolerance is 60 ms.` config line)
    # are advisory at most: cap them at LOW so they no longer fail the skill,
    # while genuine large-block or cross-file duplication keeps its
    # HIGH/MEDIUM severity.
    if severity in (Severity.HIGH, Severity.MEDIUM) and _is_trivial_intra_file_duplicate(cluster):
        severity = Severity.LOW

    distinct_files = sorted({c.source_file for c in cluster.members})
    location_text = "\n  vs ".join(
        f'"{c.heading}" in {c.source_file} (lines {c.start_line}-{c.end_line})' for c in cluster.members
    )
    if len(distinct_files) == 1:
        message = f"Duplicate content found within {distinct_files[0]}:\n  {location_text}"
    else:
        files_str = " and ".join(distinct_files)
        message = f"Duplicate content found across {files_str}:\n  {location_text}"

    return Finding(
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
