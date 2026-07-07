# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Similarity validator for detecting duplicate content.

Compares Skills, Rules, or Workflows via vector embedding cosine
similarity using public OpenAI-compatible embedding APIs.
"""

from __future__ import annotations

import math
from pathlib import Path

from skillevaluator.constants import (
    CONTENT_TYPE_UNKNOWN,
    SIMILARITY_DEFAULT_THRESHOLD,
)
from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError
from skillevaluator.embedding.extractor import extract_from_skill
from skillevaluator.embedding.registry import EmbeddingRegistry, SimilarityMatch
from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Finding
from skillevaluator.utils.tier2_paths import safe_path_label, sanitize_path_text
from skillevaluator.validators.base import ValidationResult, ValidatorBase

logger = get_logger(__name__)


class SimilarityValidator(ValidatorBase):
    """Detect duplicate content via embedding similarity.

    Supports all three SkillEvaluator content types (skill, rules, workflows).
    Produces structured Finding objects classified into four severity tiers.
    """

    def __init__(
        self,
        threshold: float = SIMILARITY_DEFAULT_THRESHOLD,
        model: str | None = None,
        catalog_path: Path | None = None,
        save_catalog_path: Path | None = None,
        cache_path: Path | None = None,
        save_cache_path: Path | None = None,
        content_type: str | None = None,
        full_body: bool = False,
    ) -> None:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Similarity threshold must be finite and within [0, 1]")
        if catalog_path and cache_path and catalog_path != cache_path:
            raise ValueError("--catalog and deprecated --cache cannot be used together")
        if save_catalog_path and save_cache_path and save_catalog_path != save_cache_path:
            raise ValueError("--save-catalog and deprecated --save-cache cannot be used together")
        resolved_catalog = catalog_path or cache_path
        resolved_save_catalog = save_catalog_path or save_cache_path
        if resolved_catalog and resolved_save_catalog:
            raise ValueError("--catalog and --save-catalog cannot be used together")
        self._threshold = threshold
        # None defers to provider resolution (SKILL_EVAL_EMBEDDING_MODEL);
        # pinning SIMILARITY_DEFAULT_MODEL here would override the env var.
        self._model = model
        self._catalog_path = resolved_catalog
        self._save_catalog_path = resolved_save_catalog
        self._content_type = content_type
        self._full_body = full_body

    @property
    def name(self) -> str:
        return "Similarity Check"

    @property
    def description(self) -> str:
        return "Detect duplicate content via embedding similarity"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Run similarity detection on a path.

        Without a catalog, performs pairwise comparison of discovered items.
        With a catalog, compares exactly one supplied skill against it.
        """
        result = ValidationResult(
            validator_name=self.name,
            validator_description=self.description,
        )
        if not skill_path.is_dir():
            result.add_error(
                "Similarity checks require a directory containing a skill collection "
                "or one root skill for --catalog comparison"
            )
            return result

        content_type = self._resolve_content_type(skill_path)
        if content_type == CONTENT_TYPE_UNKNOWN:
            result.add_error(
                f"Cannot auto-detect content type for {safe_path_label(skill_path)}. "
                "Use --type to specify skill, rules, or workflows."
            )
            return result
        if (self._catalog_path or self._save_catalog_path) and content_type != "skill":
            result.add_error("Local catalog workflows support skill content only")
            return result

        client = EmbeddingClient(model=self._model)
        registry = EmbeddingRegistry(client, full_body=self._full_body)

        try:
            if self._catalog_path:
                if not self._catalog_path.exists():
                    result.add_error(f"Catalog does not exist: {self._catalog_display_name(self._catalog_path)}")
                    return result
                registry.load_catalog(self._catalog_path)
                result.add_success(
                    "catalog_loaded",
                    f"Loaded {registry.size} entries from catalog",
                )
                target = extract_from_skill(skill_path)
                if target is None:
                    result.add_error("Catalog comparison requires a skill directory with a root SKILL.md")
                    return result
                matches = registry.query_entry(target, self._threshold)
                result.add_success(
                    "catalog_compared",
                    f"Compared '{target.name}' against {registry.size} catalog entries",
                    catalog_entry_count=registry.size,
                    target_name=target.name,
                )
            else:
                minimum_entries = 1 if self._save_catalog_path is not None else 2
                count = registry.build_from_directory(
                    skill_path,
                    content_type,
                    minimum_entries=minimum_entries,
                )
                if count == 0:
                    if self._save_catalog_path:
                        result.add_error(f"Cannot save a catalog from an empty {content_type} collection")
                    else:
                        result.add_error(f"No {content_type} content found to compare")
                    return result
                result.add_success(
                    "index_built",
                    f"Indexed {count} {content_type} entries",
                    content_type=content_type,
                    entry_count=count,
                )
                matches = registry.find_duplicates(self._threshold)
                if self._save_catalog_path:
                    registry.save_catalog(self._save_catalog_path)
                    result.add_success(
                        "catalog_saved",
                        f"Saved local catalog to {self._catalog_display_name(self._save_catalog_path)}",
                    )
        except (SimilarityConfigError, ValueError, OSError) as exc:
            result.add_error(
                sanitize_path_text(
                    str(exc),
                    (skill_path, self._catalog_path, self._save_catalog_path),
                )
            )
            return result

        self._record_matches(
            matches,
            result,
            catalog_comparison=self._catalog_path is not None,
        )

        if not matches:
            result.add_success(
                "similarity_check",
                f"No duplicates detected (threshold: {self._threshold})",
            )

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_content_type(self, path: Path) -> str:
        """Determine content type from explicit flag or auto-detection."""
        if self._content_type and self._content_type != "auto":
            return self._content_type

        from skillevaluator.cli_core import detect_content_type

        return detect_content_type(path)

    @staticmethod
    def _catalog_display_name(path: Path) -> str:
        """Return a result-safe catalog label without host directory details."""
        label = safe_path_label(path)
        return "catalog" if label == "." else label

    @staticmethod
    def _record_matches(
        matches: list[SimilarityMatch],
        result: ValidationResult,
        *,
        catalog_comparison: bool = False,
    ) -> None:
        """Convert SimilarityMatch objects into structured Findings."""
        for match in matches:
            if catalog_comparison:
                message = (
                    f"Target skill '{match.entry_a}' matches catalog skill "
                    f"'{match.entry_b}' as {match.classification} (score: {match.score:.3f})"
                )
            else:
                message = (
                    f"'{match.entry_a}' and '{match.entry_b}' are {match.classification} (score: {match.score:.3f})"
                )
            metadata = {
                "score": round(match.score, 4),
                "classification": match.classification,
                "entry_a": match.entry_a,
                "entry_b": match.entry_b,
                "path_a": match.path_a,
                "path_b": match.path_b,
            }
            if catalog_comparison:
                metadata["comparison_mode"] = "target-vs-catalog"
            result.add_finding(
                Finding(
                    category="SIMILARITY",
                    severity=match.severity,
                    check_name=match.classification,
                    message=message,
                    file_path=match.path_a,
                    metadata=metadata,
                )
            )
