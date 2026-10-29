# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private composition of bundle deduplication; deliberately absent from CLI dispatch."""

from __future__ import annotations

import copy
from pathlib import Path

from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator
from skillevaluator.deduplication.utils.chunker import chunk_file
from skillevaluator.deduplication.utils.skill_collector import SkillCollectionError, collect_files
from skillevaluator.embedding.client import EmbeddingClient, SimilarityConfigError
from skillevaluator.inference.diagnostics import llm_failure_diagnostic
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.utils.secure_fs import SecurePathError

from .catalog import LocalCatalog
from .catalog_checks import (
    INTER_PLUGIN_DESCRIPTION,
    INTER_PLUGIN_KEY,
    INTER_PLUGIN_NAME,
    INTER_SKILL_DESCRIPTION,
    INTER_SKILL_KEY,
    INTER_SKILL_NAME,
    advisory_severity,
    check_catalog_plugins,
    check_catalog_skills,
    skipped_result,
)
from .intra_plugin_validator import IntraPluginValidator
from .profile import load_plugin_profile

MAX_PLUGIN_DEDUP_SKILLS = 32
MAX_PLUGIN_DEDUP_LLM_CALLS = 100
MAX_PLUGIN_DEDUP_CHUNKS = 512
MAX_PLUGIN_DEDUP_TEXT_BYTES = 8 * 1024 * 1024
MAX_PLUGIN_DEDUP_FILES = 256
MAX_PLUGIN_DEDUP_SCALAR_COMPARISONS = 25_000_000
MAX_PLUGIN_DEDUP_PROMPT_CHARS = 1_000_000
_UNSAFE = frozenset(
    {
        "invalid_skill_root",
        "path_access_error",
        "unsafe_path",
        "unsafe_hardlink",
        "unsafe_root",
        "secure_open_unavailable",
    }
)


def _skip_context(reason, **metadata):
    result = ValidationResult(
        validator_name="Context Deduplication",
        validator_description="Compare content within each effective bundle skill",
    )
    result.metadata.update(advisory_tier2=True, scan_status="skipped", optional_skip=True, **metadata)
    result.add_warning(reason)
    return result


def _advisory(result):
    result = copy.deepcopy(result)
    if result.metadata.get("security_failure") or any(f.check_name in _UNSAFE for f in result.findings):
        result.metadata["security_failure"] = True
        result.passed = False
        return result
    legacy = {finding.to_legacy_string() for finding in result.findings}
    notes = [note for note in (*result.warnings, *result.errors) if note not in legacy]
    for finding in result.findings:
        finding.metadata.setdefault("native_severity", finding.severity.value)
        finding.severity = advisory_severity(finding.severity)
    result.recalculate_from_findings()
    for note in dict.fromkeys(notes):
        result.add_warning(note)
    result.metadata["advisory_tier2"] = True
    return result


def run_plugin_skill_context_dedup(profile, *, threshold=0.8, model=None, llm_model=None):
    """Run C-intra for unique effective skill directories with an aggregate call allowance."""
    skills = profile.bundled_skills
    if len(skills) > MAX_PLUGIN_DEDUP_SKILLS:
        return [
            _skip_context(
                "Bundle skill limit exceeded before provider work",
                work_limit_exceeded=True,
                actual_skills=len(skills),
                skill_limit=MAX_PLUGIN_DEDUP_SKILLS,
            )
        ]
    if not skills:
        return [_skip_context("No effective skill directories to compare")]
    # Capture all skill text before any provider work. The same bounded
    # snapshots feed the existing validator, avoiding a second filesystem read.
    snapshots = []
    total_chunks = total_text_bytes = total_files = 0
    for skill in skills:
        try:
            collected = collect_files(skill.path)
        except SkillCollectionError as exc:
            result = ValidationResult(validator_name="Context Deduplication")
            result.add_finding(
                Finding(
                    "CONTENT_DEDUP",
                    Severity.CRITICAL,
                    exc.check_name,
                    str(exc),
                    file_path=(Path(skill.root_relative) / exc.rel_path).as_posix(),
                    suggestion=exc.suggestion,
                    metadata=exc.metadata,
                )
            )
            return [_advisory(result)]
        total_files += len(collected)
        total_text_bytes += sum(len(file.content.encode("utf-8")) for file in collected)
        total_chunks += sum(len(chunk_file(file)) for file in collected)
        if (
            total_chunks > MAX_PLUGIN_DEDUP_CHUNKS
            or total_text_bytes > MAX_PLUGIN_DEDUP_TEXT_BYTES
            or total_files > MAX_PLUGIN_DEDUP_FILES
        ):
            return [
                _skip_context(
                    "Aggregate bundle content limit exceeded before provider work",
                    work_limit_exceeded=True,
                    actual_chunks=total_chunks,
                    actual_text_bytes=total_text_bytes,
                    actual_files=total_files,
                )
            ]
        snapshots.append(collected)
    per_skill = min(50, MAX_PLUGIN_DEDUP_LLM_CALLS // len(skills))
    validator = IntraSkillValidator(
        threshold=threshold,
        embedding_model=model,
        llm_model=llm_model,
        max_llm_clusters=per_skill,
        _max_scalar_comparisons=MAX_PLUGIN_DEDUP_SCALAR_COMPARISONS // len(skills),
        _max_total_prompt_chars=MAX_PLUGIN_DEDUP_PROMPT_CHARS // len(skills),
    )
    results = []
    for skill, collected in zip(skills, snapshots, strict=True):
        try:
            result = validator.validate(skill.path, _collected_files=collected)
        except SecurePathError as exc:
            result = ValidationResult(validator_name="Context Deduplication")
            result.add_error(str(exc))
            result.metadata["security_failure"] = True
        result = _advisory(result)
        for finding in result.findings:
            relative = finding.file_path or "."
            if relative == skill.path.name and finding.check_name.endswith("limit"):
                relative = "."
            finding.file_path = (Path(skill.root_relative) / relative).as_posix()
        result.metadata.update(
            bundled_skill=skill.root_relative, max_llm_calls=per_skill, aggregate_max_llm_calls=per_skill * len(skills)
        )
        results.append(result)
    return results


def _security_result(error):
    result = ValidationResult(validator_name="Context Deduplication")
    result.add_finding(
        Finding(
            "PLUGIN_SECURITY",
            Severity.HIGH,
            "unsafe_plugin_filesystem",
            "Bundle source could not be read safely",
            file_path=error.relative_path,
            metadata={"source_error": error.code},
        )
    )
    result.mark_scan_incomplete("bundle-filesystem")
    result.metadata["security_failure"] = True
    return result


def _catalog_skips(reason):
    return [
        skipped_result(INTER_SKILL_NAME, INTER_SKILL_DESCRIPTION, INTER_SKILL_KEY, reason),
        skipped_result(INTER_PLUGIN_NAME, INTER_PLUGIN_DESCRIPTION, INTER_PLUGIN_KEY, reason),
    ]


def run_plugin_dedup_scan(
    plugin_root: Path,
    *,
    run_context=True,
    threshold=0.8,
    model=None,
    llm_model=None,
    catalog=None,
    similarity_threshold=0.9,
    llm_verdict=False,
):
    """A, C-intra, then fresh local-catalog C-inter and B; no public registration."""
    results = [IntraPluginValidator().validate(plugin_root)]
    if results[0].metadata.get("security_failure"):
        return results
    try:
        profile = load_plugin_profile(plugin_root)
    except SecurePathError as exc:
        results.append(_security_result(exc))
        results.extend(_catalog_skips("Unsafe bundle source; catalog comparisons did not run"))
        return results
    except ValueError as exc:
        failed = ValidationResult(validator_name="Context Deduplication")
        failed.mark_scan_incomplete("bundle-inventory")
        failed.add_error(str(exc))
        results.append(failed)
        results.extend(_catalog_skips(str(exc)))
        return results
    results.extend(
        run_plugin_skill_context_dedup(profile, threshold=threshold, model=model, llm_model=llm_model)
        if run_context
        else [_skip_context("Context comparison disabled")]
    )
    if any(result.metadata.get("security_failure") for result in results):
        results.extend(_catalog_skips("Unsafe bundle source; catalog comparisons did not run"))
        return results
    if catalog is None:
        results.extend(_catalog_skips("No local catalog supplied"))
        return results
    try:
        # C-intra can take minutes. Use the current declarations and contents.
        profile = load_plugin_profile(plugin_root)
        local = LocalCatalog.load(catalog, client=EmbeddingClient(model=model))
        results.append(check_catalog_skills(profile, local, threshold=similarity_threshold))
        results.append(
            check_catalog_plugins(
                profile, local, threshold=similarity_threshold, llm_verdict=llm_verdict, llm_model=llm_model
            )
        )
    except SecurePathError as exc:
        results.append(_security_result(exc))
        results.extend(_catalog_skips("Unsafe bundle source; catalog comparisons did not run"))
    except (ValueError, SimilarityConfigError) as exc:
        reason = llm_failure_diagnostic(exc) if isinstance(exc, SimilarityConfigError) else str(exc)
        results.extend(_catalog_skips(f"Local catalog comparison unavailable: {reason}"))
    return results
