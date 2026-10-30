# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private advisory member/catalog and bundle/catalog comparisons."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from skillevaluator.embedding.registry import classify
from skillevaluator.inference.diagnostics import llm_failure_diagnostic
from skillevaluator.models.result import Finding, Severity, ValidationResult

from .profile import member_overlap

INTER_SKILL_KEY = "catalog_skill_similarity"
INTER_PLUGIN_KEY = "inter_plugin_similarity"
INTER_SKILL_NAME = "Inter-Skill Deduplication"
INTER_PLUGIN_NAME = "Inter-Plugin Deduplication"
INTER_SKILL_DESCRIPTION = "Compare effective bundle skills with a local skills catalog"
INTER_PLUGIN_DESCRIPTION = "Compare the bundle with other bundles in a local catalog"
INTER_PLUGIN_TOP_K = 10
INTER_SKILL_CATALOG_TOP_K = 10
INTER_PLUGIN_MEMBER_OVERLAP_THRESHOLD = 0.5


def advisory_severity(severity):
    return Severity.MEDIUM if severity in (Severity.HIGH, Severity.CRITICAL) else severity


def record_similarity(result, key, *, status, catalog_entries, matches, reason=None):
    result.metadata.setdefault("plugin", {})[key] = {
        "status": status,
        "catalog_entries": catalog_entries,
        "matches": matches,
        "reason": reason,
    }
    result.metadata["advisory_tier2"] = True
    if status == "skipped":
        result.metadata.update(scan_status="skipped", optional_skip=True)
        result.add_warning(reason or "Comparison did not run")


def skipped_result(name, description, key, reason, *, catalog_entries=0):
    result = ValidationResult(validator_name=name, validator_description=description)
    record_similarity(result, key, status="skipped", catalog_entries=catalog_entries, matches=[], reason=reason)
    return result


def self_plugin_entries(profile, catalog):
    return [e for e in catalog.plugins if e.source_fingerprint == profile.source_fingerprint]


def _top_k(value):
    if type(value) is not int or not 1 <= value <= INTER_PLUGIN_TOP_K:
        raise ValueError(f"top_k must be within [1, {INTER_PLUGIN_TOP_K}]")


def _threshold(value):
    from skillevaluator.embedding.registry import _validate_threshold

    _validate_threshold(value)


def check_catalog_skills(profile, catalog, *, threshold, top_k=INTER_SKILL_CATALOG_TOP_K):
    """C-inter compares every comparable effective skill; names never identify self."""
    _threshold(threshold)
    _top_k(top_k)
    comparable = [s for s in profile.bundled_skills if s.entry is not None]
    if not comparable or not catalog.skills:
        return skipped_result(
            INTER_SKILL_NAME,
            INTER_SKILL_DESCRIPTION,
            INTER_SKILL_KEY,
            "No comparable bundle or catalog skills",
            catalog_entries=len(catalog.skills),
        )
    result = ValidationResult(validator_name=INTER_SKILL_NAME, validator_description=INTER_SKILL_DESCRIPTION)
    for skill in profile.bundled_skills:
        if skill.entry is None:
            result.add_warning(f"Skill {skill.root_relative} was not compared: {skill.skip_reason}")
    prefixes = [entry.path for entry in self_plugin_entries(profile, catalog)]
    scores = catalog.score([s.entry.embedding_text for s in comparable], catalog.skills)
    matches = []
    excluded_paths = set()
    excluded = 0
    for skill, scored in zip(comparable, scores, strict=True):
        if catalog.plugins:
            self_paths = {(PurePosixPath(prefix) / skill.root_relative).as_posix() for prefix in prefixes}
            fingerprint = None
        else:
            # A v1 catalog can have been rooted at this bundle or its skills folder.
            self_paths = {skill.root_relative, skill.rel}
            fingerprint = hashlib.sha256(skill.entry.embedding_text.encode("utf-8")).hexdigest()
        candidates = []
        for entry, score in scored:
            if entry.path in self_paths and (fingerprint is None or entry.content_fingerprint == fingerprint):
                excluded += 1
                excluded_paths.add(entry.path)
            elif score >= threshold:
                candidates.append((entry, score))
        candidates.sort(key=lambda pair: (-pair[1], pair[0].path))
        for entry, score in candidates[:top_k]:
            classification, native = classify(score)
            matches.append({"skill": skill.root_relative, "match": entry.name, "similarity": round(score, 4)})
            result.add_finding(
                Finding(
                    "INTER_SKILL",
                    advisory_severity(native),
                    classification,
                    f"Skill '{skill.entry.name}' overlaps catalog skill '{entry.name}' ({entry.path}); score {score:.3f}",
                    file_path=skill.root_relative,
                    suggestion="Review whether the existing skill can be reused.",
                    metadata={
                        "score": round(score, 4),
                        "classification": classification,
                        "native_severity": native.value,
                        "bundled_skill": skill.root_relative,
                        "catalog_skill": entry.name,
                        "catalog_path": entry.path,
                        "comparison_mode": "bundled-skill-vs-catalog",
                    },
                )
            )
        if len(candidates) > top_k:
            result.add_warning(f"Skill {skill.root_relative}: reported {top_k} of {len(candidates)} catalog matches")
    result.add_success(
        "skill_catalog_comparison",
        "Compared bundle skills with the local catalog",
        bundled_skills=len(comparable),
        catalog_entries=len(catalog.skills),
        self_matches_excluded=excluded,
        excluded_catalog_paths=sorted(excluded_paths),
    )
    record_similarity(result, INTER_SKILL_KEY, status="compared", catalog_entries=len(catalog.skills), matches=matches)
    return result


def _llm_verdicts(profile, candidates, *, llm_model, result):
    from skillevaluator.inference import LLMClient

    from .llm_analyzer import analyze_inter_plugin, build_inter_plugin_prompt, validate_prompt_batch

    verdicts = [None] * len(candidates)
    try:
        prompts = [
            build_inter_plugin_prompt(profile=profile, existing=e, similarity_score=s, member_overlap=o)
            for e, s, o in candidates
        ]
        validate_prompt_batch(prompts)
        client = LLMClient(model=llm_model, max_tokens=2048)
    except Exception as exc:
        result.add_warning(f"Optional bundle verdict skipped: {llm_failure_diagnostic(exc)}")
        result.metadata["llm_analysis"] = {"status": "skipped", "candidates_total": len(candidates)}
        return verdicts
    failed = 0
    for index, prompt in enumerate(prompts):
        try:
            verdicts[index] = analyze_inter_plugin(client, prompt)
        except Exception as exc:
            failed += 1
            result.add_warning(f"Optional bundle verdict unavailable: {llm_failure_diagnostic(exc)}")
    result.metadata["llm_analysis"] = {
        "candidates_total": len(candidates),
        "candidates_completed": len(candidates) - failed,
        "candidates_failed": failed,
    }
    return verdicts


def check_catalog_plugins(profile, catalog, *, threshold, llm_verdict=False, llm_model=None, top_k=INTER_PLUGIN_TOP_K):
    """B retains name collisions even when an optional verdict finds distinct functionality."""
    _threshold(threshold)
    _top_k(top_k)
    if not catalog.plugins or profile.description is None:
        return skipped_result(
            INTER_PLUGIN_NAME,
            INTER_PLUGIN_DESCRIPTION,
            INTER_PLUGIN_KEY,
            "No comparable bundle descriptions in the local catalog",
            catalog_entries=len(catalog.plugins),
        )
    result = ValidationResult(validator_name=INTER_PLUGIN_NAME, validator_description=INTER_PLUGIN_DESCRIPTION)
    self_ids = {e.entry_id for e in self_plugin_entries(profile, catalog)}
    candidates = []
    for entry, similarity in catalog.score([profile.embedding_text], catalog.plugins)[0]:
        if entry.entry_id in self_ids:
            continue
        overlap = member_overlap(profile.members, entry.members)
        collision = entry.name.strip().casefold() == profile.name.casefold()
        if similarity >= threshold or overlap >= INTER_PLUGIN_MEMBER_OVERLAP_THRESHOLD or collision:
            candidates.append((entry, similarity, overlap))
    candidates.sort(
        key=lambda item: (item[0].name.strip().casefold() != profile.name.casefold(), -item[1], -item[2], item[0].path)
    )
    total = len(candidates)
    candidates = candidates[:top_k]
    verdicts = (
        _llm_verdicts(profile, candidates, llm_model=llm_model, result=result)
        if llm_verdict and candidates
        else [None] * len(candidates)
    )
    matches = []
    for (entry, score, overlap), verdict in zip(candidates, verdicts, strict=True):
        collision = entry.name.strip().casefold() == profile.name.casefold()
        verdict_name = verdict.verdict if verdict else None
        matches.append(
            {
                "name": entry.name,
                "similarity": round(score, 4),
                "member_overlap": round(overlap, 4),
                "verdict": verdict_name,
            }
        )
        if verdict_name == "UNIQUE" and not collision:
            continue
        classification, native = (
            classify(score)
            if score >= threshold
            else ("NAME_COLLISION" if collision else "MEMBER_OVERLAP", Severity.LOW)
        )
        if verdict_name == "UNIQUE":
            classification, native = "NAME_COLLISION", Severity.LOW
        elif verdict is not None:
            from .llm_analyzer import inter_plugin_verdict_to_severity

            classification, native = verdict_name, inter_plugin_verdict_to_severity(verdict)
        result.add_finding(
            Finding(
                "INTER_PLUGIN",
                advisory_severity(native),
                classification,
                f"Bundle '{profile.name}' overlaps catalog bundle '{entry.name}' ({entry.path}); score {score:.3f}, member overlap {overlap:.3f}"
                + ("; same name with a different manifest" if collision else ""),
                file_path=profile.manifest,
                suggestion=verdict.suggestion
                if verdict and verdict.suggestion
                else "Review the overlap or rename a colliding bundle.",
                metadata={
                    "catalog_plugin": entry.name,
                    "catalog_path": entry.path,
                    "similarity": round(score, 4),
                    "member_overlap": round(overlap, 4),
                    "name_collision": collision,
                    "native_severity": native.value,
                    **(
                        {"verdict": verdict_name, "confidence": verdict.confidence, "reasoning": verdict.reasoning}
                        if verdict
                        else {}
                    ),
                },
            )
        )
    if total > top_k:
        result.add_warning(f"Reported {top_k} of {total} matching catalog bundles")
    result.add_success(
        "plugin_catalog_comparison",
        "Compared bundle with the local catalog",
        catalog_entries=len(catalog.plugins),
        self_matches_excluded=len(self_ids),
        excluded_catalog_paths=sorted(e.path for e in self_plugin_entries(profile, catalog)),
    )
    record_similarity(
        result, INTER_PLUGIN_KEY, status="compared", catalog_entries=len(catalog.plugins), matches=matches
    )
    return result
