# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 command implementations."""

from __future__ import annotations

from pathlib import Path

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_LLM_CLUSTERS,
    MAX_PLUGIN_DEDUP_LLM_CALLS,
    MAX_PLUGIN_DEDUP_SKILLS,
    SIMILARITY_DEFAULT_MAX_ENTRIES,
    SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
    SIMILARITY_DEFAULT_THRESHOLD,
)
from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator
from skillevaluator.deduplication.result_status import mark_advisory_skip, mark_security_failure
from skillevaluator.embedding.limits import validate_max_entries, validate_max_scalar_comparisons
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.tier1.commands import emit_reports
from skillevaluator.utils.secure_fs import SecurePathError
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
    # Caller option errors have safe diagnostics and should not become provider failures.
    validate_max_entries(max_entries)
    validate_max_scalar_comparisons(max_scalar_comparisons)

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


def _make_advisory(result: ValidationResult) -> ValidationResult:
    """Cap plugin Tier 2 findings at advisory severity and keep other notes as warnings."""
    from skillevaluator.deduplication.plugin.catalog_checks import advisory_severity

    if result.metadata.get("security_failure"):
        # Filesystem-integrity failures mean the requested check could not be
        # executed safely. Keep them blocking instead of disguising them as an
        # ordinary advisory deduplication finding.
        return mark_security_failure(result)

    # Errors and warnings that are not a finding's legacy string (provider
    # failures, skip reasons) are notes. Finding strings are rebuilt from the
    # capped severities, then each note is kept once as a warning.
    finding_strings = {finding.to_legacy_string() for finding in result.findings}
    notes = [message for message in (*result.warnings, *result.errors) if message not in finding_strings]
    for finding in result.findings:
        finding.severity = advisory_severity(finding.severity)
    result.recalculate_from_findings()
    for note in notes:
        if note not in result.warnings:
            result.add_warning(note)
    result.passed = True
    result.metadata["advisory_tier2"] = True
    return result


def _unsafe_plugin_result(
    reason: Exception | str,
    *,
    name: str = "Context Deduplication",
    description: str = "Detect redundant content within each bundled plugin skill",
) -> ValidationResult:
    """Return a blocking result when plugin content cannot be read safely."""
    result = ValidationResult(
        validator_name=name,
        validator_description=description,
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
    return mark_security_failure(result)


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
    return mark_advisory_skip(
        result,
        reason,
        work_limit_exceeded=True,
        actual_skills=actual_skills,
        skill_limit=MAX_PLUGIN_DEDUP_SKILLS,
    )


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
    # Share the plugin-wide LLM allowance across skills without exceeding the
    # per-skill cluster limit the validator accepts; a plugin with few skills
    # would otherwise hand one skill more than that limit.
    per_skill_llm_budget = min(
        CONTENT_DEDUP_MAX_LLM_CLUSTERS,
        max(1, MAX_PLUGIN_DEDUP_LLM_CALLS // len(skill_dirs)),
    )
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
            # Unsafe skill content is returned, not raised, as a result marked
            # security_failure; _make_advisory keeps that result blocking.
            skill_result = validator.validate(skill_dir)
        except SecurePathError as exc:
            # An unsafe path that still escapes validate() stays blocking too.
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
            mark_security_failure(aggregate)
    aggregate.passed = not aggregate.metadata.get("security_failure", False)
    aggregate.metadata["advisory_tier2"] = True
    return [aggregate]


_NO_CATALOG_REASON = (
    "No local catalog supplied; run `skillevaluator tier2 PLUGIN --catalog FILE` to compare this plugin "
    "and its bundled skills with a saved local catalog."
)


def _catalog_check_skips(reason: str, **metadata: object) -> list[ValidationResult]:
    from skillevaluator.deduplication.plugin.catalog_checks import (
        INTER_PLUGIN_DESCRIPTION,
        INTER_PLUGIN_KEY,
        INTER_PLUGIN_NAME,
        INTER_SKILL_DESCRIPTION,
        INTER_SKILL_KEY,
        INTER_SKILL_NAME,
        skipped_result,
    )

    results = [
        skipped_result(INTER_SKILL_NAME, INTER_SKILL_DESCRIPTION, INTER_SKILL_KEY, reason),
        skipped_result(INTER_PLUGIN_NAME, INTER_PLUGIN_DESCRIPTION, INTER_PLUGIN_KEY, reason),
    ]
    for result in results:
        result.metadata.update(metadata)
    return results


def _finalize_catalog_result(result: ValidationResult) -> ValidationResult:
    """Keep local catalog results advisory without discarding non-finding warnings."""
    if result.metadata.get("security_failure"):
        return _make_advisory(result)
    for error in list(result.errors):
        if error not in result.warnings:
            result.warnings.append(error)
            result.summary.warnings += 1
    result.errors.clear()
    result.summary.errors = 0
    result.passed = True
    result.metadata["advisory_tier2"] = True
    return result


def run_plugin_catalog_checks(
    plugin_root: Path,
    *,
    catalog: Path | None,
    threshold: float = SIMILARITY_DEFAULT_THRESHOLD,
    model: str | None = None,
    llm_verdict: bool = False,
    llm_model: str | None = None,
    max_scalar_comparisons: int = SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
) -> list[ValidationResult]:
    """Run advisory Check C-inter and Check B against a local JSON catalog.

    Without a catalog both checks are recorded as skipped, not failed. The
    catalog is read with the bounded no-follow loader; plugin and bundled skill
    reads use the secure plugin helpers. Only the configured embedding provider
    (and the chat LLM when ``llm_verdict`` is set) is contacted.
    """
    from skillevaluator.deduplication.plugin.catalog_checks import (
        INTER_PLUGIN_DESCRIPTION,
        INTER_PLUGIN_KEY,
        INTER_PLUGIN_NAME,
        INTER_SKILL_DESCRIPTION,
        INTER_SKILL_KEY,
        INTER_SKILL_NAME,
        check_catalog_plugins,
        check_catalog_skills,
        skipped_result,
    )
    from skillevaluator.deduplication.plugin.profile import (
        PluginProfileError,
        PluginSkillLimitError,
        load_plugin_profile,
    )
    from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError, validate_similarity_threshold
    from skillevaluator.embedding.registry import EmbeddingRegistry
    from skillevaluator.utils.tier2_paths import sanitize_path_text

    threshold = validate_similarity_threshold(threshold, context="Plugin catalog similarity")
    validate_max_scalar_comparisons(max_scalar_comparisons)
    if catalog is None:
        return _catalog_check_skips(_NO_CATALOG_REASON)

    try:
        profile = load_plugin_profile(plugin_root, max_skills=MAX_PLUGIN_DEDUP_SKILLS)
    except PluginSkillLimitError:
        from skillevaluator.utils.helpers import find_bundled_plugin_skills

        actual = len(find_bundled_plugin_skills(plugin_root))
        reason = (
            f"Plugin bundles {actual} skills, exceeding the automatic Tier 2 limit of "
            f"{MAX_PLUGIN_DEDUP_SKILLS}; no embedding, catalog, or LLM calls were made."
        )
        return _catalog_check_skips(
            reason,
            work_limit_exceeded=True,
            actual_skills=actual,
            skill_limit=MAX_PLUGIN_DEDUP_SKILLS,
        )
    except PluginProfileError as exc:
        return _catalog_check_skips(f"Plugin manifest could not supply a comparable profile: {exc}")
    except (SecurePathError, ValueError) as exc:
        # The profile reads the manifest and the bundled skills, so neither
        # check can run; both report the refusal so Check B never vanishes.
        reason = sanitize_path_text(str(exc), (plugin_root, catalog))
        return [
            _unsafe_plugin_result(reason, name=INTER_SKILL_NAME, description=INTER_SKILL_DESCRIPTION),
            _unsafe_plugin_result(reason, name=INTER_PLUGIN_NAME, description=INTER_PLUGIN_DESCRIPTION),
        ]

    safe_paths = (plugin_root, catalog)
    registry = EmbeddingRegistry(EmbeddingClient(model=model), max_scalar_comparisons=max_scalar_comparisons)
    try:
        registry.load_catalog(catalog)
    except (ValueError, OSError, SimilarityConfigError) as exc:
        return _catalog_check_skips(
            sanitize_path_text(f"Local catalog could not be loaded; comparison did not run: {exc}", safe_paths)
        )

    checks = (
        (
            INTER_SKILL_NAME,
            INTER_SKILL_DESCRIPTION,
            INTER_SKILL_KEY,
            len(registry.skill_entries),
            lambda: check_catalog_skills(profile, registry, threshold=threshold),
        ),
        (
            INTER_PLUGIN_NAME,
            INTER_PLUGIN_DESCRIPTION,
            INTER_PLUGIN_KEY,
            registry.plugin_size,
            lambda: check_catalog_plugins(
                profile,
                registry,
                threshold=threshold,
                llm_verdict=llm_verdict,
                llm_model=llm_model,
            ),
        ),
    )
    results: list[ValidationResult] = []
    for name, description, key, catalog_entries, run_check in checks:
        try:
            result = run_check()
        except (SimilarityConfigError, ValueError, OSError) as exc:
            # Provider failures and exceeded work limits mean the comparison
            # did not run; record an advisory skip with a path-free reason.
            prefix = "Embedding provider error" if isinstance(exc, SimilarityConfigError) else "Comparison did not run"
            reason = sanitize_path_text(f"{prefix}: {exc}", safe_paths)
            result = skipped_result(name, description, key, reason, catalog_entries=catalog_entries)
        results.append(_finalize_catalog_result(result))
    return results


def run_plugin_dedup_scan(
    plugin_root: Path,
    *,
    run_context: bool = True,
    threshold: float = 0.80,
    model: str | None = None,
    llm_model: str | None = None,
    catalog: Path | None = None,
    similarity_threshold: float = SIMILARITY_DEFAULT_THRESHOLD,
    llm_verdict: bool = False,
) -> list[ValidationResult]:
    """Run the public plugin Tier 2 contract.

    Check A and C-intra always run offline or against the configured embedding
    provider. Check C-inter and Check B compare against an optional local JSON
    catalog and are recorded as skipped when no catalog is supplied.
    """
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
        results.append(mark_advisory_skip(skipped, reason))
    if catalog is not None and not run_context:
        results.extend(
            _catalog_check_skips(
                "Local catalog comparison needs a configured public embedding provider and the Tier 2 extra."
            )
        )
    else:
        results.extend(
            run_plugin_catalog_checks(
                plugin_root,
                catalog=catalog,
                threshold=similarity_threshold,
                model=model,
                llm_verdict=llm_verdict,
                llm_model=llm_model,
            )
        )
    return results


__all__ = [
    "emit_reports",
    "run_context_optimization_check",
    "run_dedup_scan",
    "run_plugin_catalog_checks",
    "run_plugin_dedup_scan",
    "run_plugin_skill_context_dedup",
    "run_similarity_check",
]
