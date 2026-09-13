# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 command implementations."""

from __future__ import annotations

from pathlib import Path

from skillevaluator.constants import MAX_PLUGIN_DEDUP_LLM_CALLS, MAX_PLUGIN_DEDUP_SKILLS
from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator
from skillevaluator.deduplication.utils.skill_collector import SkillCollectionError
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.tier1.commands import emit_reports
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.validators.similarity import SimilarityValidator


def _guarded_result(title: str, target_path: Path, callback) -> list[ValidationResult]:
    try:
        result = callback()
    except Exception as exc:  # validators convert expected failures; this protects CLI UX
        result = ValidationResult(validator_name=title, validator_description="Tier 2 check")
        result.add_error(f"{title} failed: {exc}")
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
    validator = IntraSkillValidator(
        threshold=threshold,
        embedding_model=model,
        llm_model=llm_model,
    )
    return _guarded_result(
        "Context Deduplication",
        skill_path,
        lambda: validator.validate(skill_path),
    )


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


def _make_advisory(result: ValidationResult) -> ValidationResult:
    """Cap plugin Tier 2 findings and legacy errors at advisory severity."""
    if result.metadata.get("security_failure"):
        # Filesystem-integrity failures mean the requested check could not be
        # executed safely. Keep them blocking instead of disguising them as an
        # ordinary advisory deduplication finding.
        result.passed = False
        result.metadata.update({"execution_status": "failed", "optional": False})
        return result

    legacy_errors = list(result.errors)
    for finding in result.findings:
        if finding.severity in (Severity.CRITICAL, Severity.HIGH):
            finding.severity = Severity.MEDIUM
    if result.findings:
        result.recalculate_from_findings()
    else:
        result.errors.clear()
        result.summary.errors = 0
    for error in legacy_errors:
        if error not in result.warnings:
            result.warnings.append(error)
            result.summary.warnings += 1
    result.passed = True
    result.metadata["advisory_tier2"] = True
    return result


def _unsafe_plugin_result(reason: Exception | str) -> ValidationResult:
    """Return a blocking result when plugin content cannot be read safely."""
    result = ValidationResult(
        validator_name="Context Deduplication",
        validator_description="Detect redundant content within each bundled plugin skill",
    )
    result.add_finding(
        Finding(
            category="PLUGIN_SECURITY",
            severity=Severity.HIGH,
            check_name="unsafe_plugin_filesystem",
            message=f"Unsafe plugin filesystem input refused: {reason}",
            file_path="<plugin-filesystem>",
            suggestion="Replace links, hardlinks, and special selected files with regular files inside the plugin root.",
        )
    )
    result.metadata.update(
        {
            "security_failure": True,
            "execution_status": "failed",
            "optional": False,
        }
    )
    return result


def _plugin_work_limit_result(actual_skills: int) -> ValidationResult:
    """Return an advisory skip before an oversized plugin triggers paid work."""
    reason = (
        f"Plugin bundles {actual_skills} skills, exceeding the automatic Tier 2 "
        f"limit of {MAX_PLUGIN_DEDUP_SKILLS}; no embedding or LLM calls were made."
    )
    result = ValidationResult(
        validator_name="Context Deduplication",
        validator_description="Detect redundant content within each bundled plugin skill",
    )
    result.add_warning(reason)
    result.metadata.update(
        {
            "advisory_tier2": True,
            "execution_status": "skipped",
            "optional": True,
            "skip_reason": reason,
            "work_limit_exceeded": True,
            "actual_skills": actual_skills,
            "skill_limit": MAX_PLUGIN_DEDUP_SKILLS,
        }
    )
    return result


def run_plugin_skill_context_dedup(
    plugin_root: Path,
    *,
    threshold: float = 0.80,
    model: str | None = None,
    llm_model: str | None = None,
) -> list[ValidationResult]:
    """Run C-intra over each safely discovered bundled skill."""
    from skillevaluator.utils.helpers import find_bundled_plugin_skills

    aggregate = ValidationResult(
        validator_name="Context Deduplication",
        validator_description="Detect redundant content within each bundled plugin skill",
    )
    aggregate.metadata["advisory_tier2"] = True
    try:
        skill_dirs = find_bundled_plugin_skills(plugin_root)
    except ValueError as exc:
        return [_unsafe_plugin_result(exc)]
    if not skill_dirs:
        aggregate.add_success("context_dedup", "No bundled skills to deduplicate")
        return [aggregate]
    if len(skill_dirs) > MAX_PLUGIN_DEDUP_SKILLS:
        return [_plugin_work_limit_result(len(skill_dirs))]

    skills_root = plugin_root / "skills"
    per_skill_llm_budget = max(1, MAX_PLUGIN_DEDUP_LLM_CALLS // len(skill_dirs))
    validator = IntraSkillValidator(
        threshold=threshold,
        embedding_model=model,
        llm_model=llm_model,
        max_llm_clusters=per_skill_llm_budget,
    )
    aggregate.metadata["max_llm_calls"] = per_skill_llm_budget * len(skill_dirs)
    for skill_dir in skill_dirs:
        skill_name = skill_dir.relative_to(skills_root).as_posix()
        try:
            skill_result = validator.validate(skill_dir)
        except (SecurePathError, SkillCollectionError) as exc:
            skill_result = _unsafe_plugin_result(exc)
        except Exception as exc:
            skill_result = ValidationResult(
                validator_name="Context Deduplication",
                validator_description="Detect redundant content within a bundled plugin skill",
            )
            skill_result.add_finding(
                Finding(
                    category="CONTENT_DEDUP",
                    severity=Severity.MEDIUM,
                    check_name="context_dedup_error",
                    message=f"Context deduplication could not run for bundled skill: {exc}",
                    file_path=str(skill_dir),
                )
            )
        aggregate.merge_with_prefix(_make_advisory(skill_result), skill_name)
        aggregate.summary.files_scanned += skill_result.summary.files_scanned
        aggregate.summary.checks_performed += skill_result.summary.checks_performed
        aggregate.summary.critical_count += skill_result.summary.critical_count
        aggregate.summary.high_count += skill_result.summary.high_count
        aggregate.summary.medium_count += skill_result.summary.medium_count
        aggregate.summary.low_count += skill_result.summary.low_count
        if skill_result.metadata.get("security_failure"):
            aggregate.metadata.update(
                {
                    "security_failure": True,
                    "execution_status": "failed",
                    "optional": False,
                }
            )
    aggregate.passed = not aggregate.metadata.get("security_failure", False)
    aggregate.metadata["advisory_tier2"] = True
    return [aggregate]


def run_plugin_dedup_scan(
    plugin_root: Path,
    *,
    run_context: bool = True,
    threshold: float = 0.80,
    model: str | None = None,
    llm_model: str | None = None,
) -> list[ValidationResult]:
    """Run the public plugin Tier 2 contract: offline Check A and C-intra."""
    from skillevaluator.deduplication.plugin import IntraPluginValidator
    from skillevaluator.utils.helpers import find_bundled_plugin_skills

    results = [_make_advisory(IntraPluginValidator().validate(plugin_root))]
    try:
        find_bundled_plugin_skills(plugin_root)
    except ValueError as exc:
        results.append(_unsafe_plugin_result(exc))
        return results
    if run_context:
        results.extend(
            run_plugin_skill_context_dedup(
                plugin_root,
                threshold=threshold,
                model=model,
                llm_model=llm_model,
            )
        )
    else:
        skipped = ValidationResult(
            validator_name="Context Deduplication",
            validator_description="Detect redundant content within each bundled plugin skill",
        )
        reason = "Skipped: configure a public embedding provider or install the Tier 2 extra."
        skipped.add_warning(reason)
        skipped.metadata.update(
            {"execution_status": "skipped", "skip_reason": reason, "optional": True, "advisory_tier2": True}
        )
        results.append(skipped)
    return results


__all__ = [
    "emit_reports",
    "run_context_optimization_check",
    "run_dedup_scan",
    "run_plugin_dedup_scan",
    "run_plugin_skill_context_dedup",
    "run_similarity_check",
]
