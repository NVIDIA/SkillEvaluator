# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 command implementations."""

from __future__ import annotations

from pathlib import Path

from skillevaluator.constants import SIMILARITY_DEFAULT_MAX_ENTRIES, SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS
from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator
from skillevaluator.models.result import ValidationResult
from skillevaluator.tier1.commands import emit_reports
from skillevaluator.validators.similarity import SimilarityValidator


def _guarded_result(title: str, target_path: Path, callback) -> list[ValidationResult]:
    try:
        result = callback()
    except Exception as exc:  # validators convert expected failures; this protects CLI UX
        result = ValidationResult(validator_name=title, validator_description="Tier 2 check")
        result.mark_scan_incomplete(title)
        # SDK exceptions may contain response bodies, request data, or credentials.
        # Expected provider failures are explained by the validators themselves.
        result.add_error(
            f"{title} could not complete because of an unexpected error ({type(exc).__name__}). "
            "Check the provider configuration and connectivity, then rerun Tier 2. "
            "If the problem persists, report this error type to the maintainers."
        )
    if not result.validator_name:
        result.validator_name = title
    if not result.validator_description:
        result.validator_description = f"Tier 2 check for {target_path}"
    return [result]


def run_similarity_check(
    content_path: Path,
    *,
    content_type: str = "auto",
    threshold: float = 0.75,
    full_body: bool = False,
    model: str | None = None,
    catalog: Path | None = None,
    save_catalog: Path | None = None,
    cache: Path | None = None,
    save_cache: Path | None = None,
    max_entries: int = SIMILARITY_DEFAULT_MAX_ENTRIES,
    max_scalar_comparisons: int = SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
) -> list[ValidationResult]:
    def _run() -> ValidationResult:
        validator = SimilarityValidator(
            threshold=threshold,
            model=model,
            catalog_path=catalog,
            save_catalog_path=save_catalog,
            cache_path=cache,
            save_cache_path=save_cache,
            content_type=None if content_type == "auto" else content_type,
            full_body=full_body,
            max_entries=max_entries,
            max_scalar_comparisons=max_scalar_comparisons,
        )
        return validator.validate(content_path)

    return _guarded_result("Similarity Check", content_path, _run)


def run_context_optimization_check(
    skill_path: Path,
    *,
    threshold: float = 0.80,
    model: str | None = None,
    llm_model: str | None = None,
) -> list[ValidationResult]:
    def _run() -> ValidationResult:
        validator = IntraSkillValidator(
            threshold=threshold,
            embedding_model=model,
            llm_model=llm_model,
        )
        return validator.validate(skill_path)

    return _guarded_result("Context Deduplication", skill_path, _run)


def run_dedup_scan(
    skill_path: Path,
    *,
    threshold: float = 0.80,
    llm_model: str | None = None,
    model: str | None = None,
) -> list[ValidationResult]:
    return run_context_optimization_check(
        skill_path,
        threshold=threshold,
        model=model,
        llm_model=llm_model,
    )


__all__ = [
    "emit_reports",
    "run_context_optimization_check",
    "run_dedup_scan",
    "run_similarity_check",
]
