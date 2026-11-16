# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.constants import SIMILARITY_DEFAULT_MAX_ENTRIES, SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS
from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator
from skillevaluator.embedding.limits import validate_max_entries, validate_max_scalar_comparisons
from skillevaluator.tier1.commands import emit_reports
from skillevaluator.tier2.results import guarded_results
from skillevaluator.validators.similarity import SimilarityValidator

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult


def _guarded_result(title: str, target_path: Path, callback, *, check_id: str) -> list[ValidationResult]:
    return guarded_results(title, target_path, lambda: [callback()], check_id=check_id)


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
    llm_verdict: bool = False,
    llm_model: str | None = None,
) -> list[ValidationResult]:
    # Caller option errors have safe diagnostics and should not become provider failures.
    validate_max_entries(max_entries)
    validate_max_scalar_comparisons(max_scalar_comparisons)
    from skillevaluator.cli_core import resolve_plugin_path
    from skillevaluator.tier2.plugin_commands import plugin_similarity_results, plugin_target

    if (target := plugin_target(content_path, content_type)) is not None:

        def compare_plugins():
            if catalog is not None and cache is not None:
                raise ValueError("--catalog and deprecated --cache cannot be used together")
            if save_catalog is not None and save_cache is not None:
                raise ValueError("--save-catalog and deprecated --save-cache cannot be used together")
            return plugin_similarity_results(
                target,
                threshold=threshold,
                full_body=full_body,
                model=model,
                catalog=catalog or cache,
                save_catalog=save_catalog or save_cache,
                max_entries=max_entries,
                max_scalar_comparisons=max_scalar_comparisons,
                llm_verdict=llm_verdict,
                llm_model=llm_model,
            )

        return guarded_results(
            "Plugin Similarity Check", resolve_plugin_path(target), compare_plugins, check_id="similarity"
        )

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

    return _guarded_result("Similarity Check", content_path, _run, check_id="similarity")


def run_context_optimization_check(
    skill_path: Path,
    *,
    threshold: float = 0.80,
    model: str | None = None,
    llm_model: str | None = None,
    catalog: Path | None = None,
    llm_verdict: bool = False,
) -> list[ValidationResult]:
    from skillevaluator.tier2.plugin_commands import plugin_target, run_public_plugin_dedup

    if (target := plugin_target(skill_path)) is not None:
        return run_public_plugin_dedup(
            target, threshold=threshold, model=model, llm_model=llm_model, catalog=catalog, llm_verdict=llm_verdict
        )

    def _run() -> ValidationResult:
        validator = IntraSkillValidator(
            threshold=threshold,
            embedding_model=model,
            llm_model=llm_model,
        )
        return validator.validate(skill_path)

    return _guarded_result("Context Deduplication", skill_path, _run, check_id="context-optimization")


def run_dedup_scan(
    skill_path: Path,
    *,
    threshold: float = 0.80,
    llm_model: str | None = None,
    model: str | None = None,
    catalog: Path | None = None,
    llm_verdict: bool = False,
) -> list[ValidationResult]:
    return run_context_optimization_check(
        skill_path,
        threshold=threshold,
        model=model,
        llm_model=llm_model,
        catalog=catalog,
        llm_verdict=llm_verdict,
    )


__all__ = [
    "emit_reports",
    "run_context_optimization_check",
    "run_dedup_scan",
    "run_similarity_check",
]
