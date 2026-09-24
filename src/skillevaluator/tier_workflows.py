# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose complete, independently scoped Tier 1 and Tier 2 workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import click

from skillevaluator.provider_config import ProviderConfigurationError, resolve_embedding_provider, resolve_llm_provider
from skillevaluator.tier1 import commands as tier1
from skillevaluator.utils.tier2_paths import is_link_or_reparse, sanitize_path_text
from skillevaluator.validators.policy import apply_policy

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.validators.policy import ValidationPolicy


def require_skill(path: Path) -> None:
    if not path.is_dir() or not (path / "SKILL.md").is_file():
        raise click.UsageError("PATH must be one skill directory containing SKILL.md.")
    if is_link_or_reparse(path) or is_link_or_reparse(path / "SKILL.md"):
        raise click.UsageError("The skill directory and SKILL.md must not be symlinks or reparse points.")


def _chat_enabled(requested: bool | None) -> bool:
    if requested is False:
        return False
    configured = any(
        os.environ.get(name)
        for name in ("SKILL_EVAL_LLM_PROVIDER", "NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if requested is None and not configured:
        return False
    try:
        resolve_llm_provider()
    except ProviderConfigurationError as exc:
        raise click.ClickException(f"{exc}\n\nTo run static Tier 1 checks only, add --no-llm.") from exc
    return True


def _stage(results: list[ValidationResult]) -> dict[str, str]:
    status = "incomplete" if any(result.is_incomplete for result in results) else "passed"
    if status != "incomplete" and (not results or any(not result.passed for result in results)):
        status = "failed"
    return {"status": status}


def _record_scope(results: list[ValidationResult], tier: int, stages: dict) -> list[ValidationResult]:
    for result in results:
        result.metadata["workflow"] = {"tier": tier, "stages": stages}
        result.metadata["gating"] = {"tier": tier, "blocking": True}
    for stage, details in stages.items():
        if details["status"] == "skipped":
            click.echo(f"{stage.replace('_', ' ')} skipped: {details['reason']}", err=True)
            if results:
                results[0].add_warning(f"Workflow scope: {stage.replace('_', ' ')} skipped. {details['reason']}")
    return results


def run_tier1_workflow(
    skill_path: Path,
    *,
    checks: str | None,
    llm: bool | None,
    min_score: int,
    profile: ValidationPolicy,
    previous_version: str | None,
) -> list[ValidationResult]:
    require_skill(skill_path)
    if checks is not None and not checks.strip(", \t\n"):
        raise click.UsageError("--checks must name at least one static check.")
    selected = checks if checks is not None else ",".join((*tier1.DEFAULT_CHECKS, *tier1.OPTIONAL_CHECKS))
    lineup = tier1.enabled_check_lineup(selected)
    unknown = set(lineup) - tier1.RECOGNIZED_CHECKS
    if unknown:
        raise click.UsageError(f"Unknown Tier 1 check(s): {', '.join(sorted(unknown))}")
    use_llm = _chat_enabled(llm)
    if use_llm:
        try:
            import openai  # noqa: F401 -- fail before scanning if the LLM extra is absent
        except ImportError as exc:
            raise click.ClickException("Install skillevaluator[llm] to run LLM checks.") from exc
    results = tier1.run_validation(
        skill_path,
        checks=selected,
        use_llm=use_llm,
        llm_verify=use_llm,
        min_score=min_score,
        previous_version=previous_version,
        policy=profile,
        content_type="skill",
        continue_on_failure=True,
    )
    apply_policy(results, profile)
    stages = {"validation": _stage(results)}
    if use_llm:
        rubric_results = tier1.run_rubric_eval(skill_path, min_score=min_score)
        apply_policy(rubric_results, profile)
        results.extend(rubric_results)
        stages["rubric"] = _stage(rubric_results)
        # Static and LLM security findings share a validator result. Describe
        # enabled scope here; do not invent an independent LLM pass/fail.
        stages["llm"] = {"status": "enabled"}
    else:
        stages["llm"] = {
            "status": "skipped",
            "reason": "LLM checks disabled by --no-llm."
            if llm is False
            else ("No LLM provider configured; set SKILL_EVAL_LLM_PROVIDER and its API key to include LLM checks."),
        }
    return _record_scope(results, 1, stages)


def run_tier2_workflow(
    skill_path: Path,
    *,
    catalog: Path | None,
    threshold: float,
    similarity_threshold: float,
    full_body: bool,
) -> list[ValidationResult]:
    require_skill(skill_path)
    if full_body and catalog is None:
        raise click.UsageError("--full-body requires --catalog for inter-skill comparison.")
    try:
        resolve_embedding_provider()
        resolve_llm_provider()
    except ProviderConfigurationError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        from skillevaluator.tier2 import commands as tier2
    except ImportError as exc:
        raise click.ClickException("Install skillevaluator[tier2] to run Tier 2.") from exc
    if catalog is not None:
        from skillevaluator.embedding.client import EmbeddingClient
        from skillevaluator.embedding.registry import EmbeddingRegistry

        try:
            # Parsing and compatibility checks are local: reject invalid,
            # linked, or mismatched catalogs before any paid model calls.
            EmbeddingRegistry(EmbeddingClient(), full_body=full_body).load_catalog(catalog)
        except (OSError, ValueError) as exc:
            raise click.ClickException(sanitize_path_text(str(exc), (skill_path, catalog))) from exc
    results = tier2.run_context_optimization_check(skill_path, threshold=threshold)
    stages = {"intra_skill": _stage(results)}
    if catalog is not None:
        inter_results = tier2.run_similarity_check(
            skill_path,
            catalog=catalog,
            threshold=similarity_threshold,
            full_body=full_body,
        )
        results.extend(inter_results)
        stages["catalog_comparison"] = _stage(inter_results)
    else:
        stages["catalog_comparison"] = {
            "status": "skipped",
            "reason": "No comparison catalog supplied; add --catalog skill-catalog.json to compare other skills.",
        }
    return _record_scope(results, 2, stages)
