# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advisory local-catalog plugin checks: Check C-inter and Check B.

Both checks compare against a local JSON catalog loaded through the bounded,
no-follow catalog reader. No vector database or catalog service is contacted;
the only network calls are the configured embedding provider (and, only when
explicitly requested, the configured chat LLM for Check B verdicts).

- **Check C-inter** embeds each bundled skill's name and description and
  compares it with every catalog skill entry, excluding only catalog entries
  that positively identify one of the plugin's own bundled skills (see
  :class:`_SelfSkillIdentity`).
- **Check B** embeds the plugin name and description, compares it with every
  catalog plugin entry, and scores member-skill overlap (Jaccard on member
  skill names). The plugin itself is excluded only by its manifest source
  fingerprint; a same-name catalog plugin with a different manifest is a
  distinct plugin and is always reported as a name collision.

Both checks record the catalog paths they excluded as self matches, so every
exclusion can be audited. Results record ``metadata["plugin"]["catalog_skill_similarity"]`` and
``metadata["plugin"]["inter_plugin_similarity"]``. Findings are advisory; the
Tier 2 plugin orchestration caps their severity at MEDIUM.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import (
    INTER_PLUGIN_MEMBER_OVERLAP_THRESHOLD,
    INTER_PLUGIN_TOP_K,
    INTER_SKILL_CATALOG_TOP_K,
    LLM_VERIFY_MAX_TOKENS,
    PLUGIN_CATALOG_PLUGIN_SIMILARITY_KEY,
    PLUGIN_CATALOG_SKILL_SIMILARITY_KEY,
    TIER2_LLM_MAX_CALLS,
    TIER2_LLM_MAX_TOTAL_PROMPT_CHARS,
)
from skillevaluator.deduplication.plugin.profile import BundledSkill, PluginProfile, member_overlap
from skillevaluator.deduplication.result_status import mark_advisory_skip
from skillevaluator.embedding.registry import classify
from skillevaluator.models.result import Finding, Severity, ValidationResult

if TYPE_CHECKING:
    from skillevaluator.embedding.registry import EmbeddingRegistry, PluginRegistryEntry, RegistryEntry

INTER_SKILL_NAME = "Inter-Skill Deduplication"
INTER_SKILL_DESCRIPTION = "Compare each bundled plugin skill with a local skills catalog"
INTER_SKILL_KEY = PLUGIN_CATALOG_SKILL_SIMILARITY_KEY
INTER_PLUGIN_NAME = "Inter-Plugin Deduplication"
INTER_PLUGIN_DESCRIPTION = "Compare the plugin with other plugins in a local catalog"
INTER_PLUGIN_KEY = PLUGIN_CATALOG_PLUGIN_SIMILARITY_KEY

STATUS_COMPARED = "compared"
STATUS_SKIPPED = "skipped"
_ADVISORY_CEILING = Severity.MEDIUM


def advisory_severity(severity: Severity) -> Severity:
    """Cap a native severity at MEDIUM: local catalog findings never gate."""
    return _ADVISORY_CEILING if severity in (Severity.CRITICAL, Severity.HIGH) else severity


def record_similarity(
    result: ValidationResult,
    key: str,
    *,
    status: str,
    catalog_entries: int,
    matches: list[dict[str, Any]],
    reason: str | None,
) -> None:
    """Record one local-catalog check summary under ``metadata["plugin"][key]``.

    A skipped check also gets its ``reason`` as a warning and the advisory skip metadata.
    """
    plugin_meta = result.metadata.setdefault("plugin", {})
    plugin_meta[key] = {
        "status": status,
        "catalog_entries": catalog_entries,
        "matches": matches,
        "reason": reason,
    }
    result.metadata["advisory_tier2"] = True
    if status == STATUS_SKIPPED:
        mark_advisory_skip(result, reason or "The local catalog check did not run.")


def skipped_result(name: str, description: str, key: str, reason: str, *, catalog_entries: int = 0) -> ValidationResult:
    """Return an advisory, optional skip that says the check did not run (not that it failed)."""
    result = ValidationResult(validator_name=name, validator_description=description)
    record_similarity(
        result,
        key,
        status=STATUS_SKIPPED,
        catalog_entries=catalog_entries,
        matches=[],
        reason=reason,
    )
    return result


def self_plugin_entries(profile: PluginProfile, registry: EmbeddingRegistry) -> list[PluginRegistryEntry]:
    """Catalog plugin entries that positively identify the plugin under test.

    Only identical manifest bytes (the source fingerprint) identify the plugin.
    Catalog names are not unique, so a same-name entry with a different
    manifest is a distinct plugin: Check B compares it and reports the name
    collision.
    """
    return [entry for entry in registry.plugin_entries if entry.source_fingerprint == profile.source_fingerprint]


def _same_name(profile: PluginProfile, entry: PluginRegistryEntry) -> bool:
    return entry.name.strip().casefold() == profile.name.casefold()


@dataclass(frozen=True)
class _SelfSkillIdentity:
    """Catalog entries that positively identify one bundled skill of the plugin under test.

    In a catalog with plugin entries, a skill entry is this skill only at
    ``<prefix>/skills/<rel>`` under a catalog plugin entry that identifies this
    plugin (see :func:`self_plugin_entries`). A skill-only catalog carries no
    plugin identity, so an entry is this skill only when it has the same
    content fingerprint at a path a catalog built from this plugin records:
    ``<rel>`` (built from ``skills/``) or ``skills/<rel>`` (built from the
    plugin root). A same-named or identical skill anywhere else is a distinct
    skill and is compared.
    """

    paths: frozenset[str]
    fingerprint: str | None = None

    @classmethod
    def build(
        cls,
        skill: BundledSkill,
        *,
        self_prefixes: list[str],
        catalog_has_plugins: bool,
    ) -> _SelfSkillIdentity:
        relative = skill.root_relative
        if catalog_has_plugins:
            return cls(
                paths=frozenset(relative if prefix == "." else f"{prefix}/{relative}" for prefix in self_prefixes)
            )
        text = skill.entry.embedding_text if skill.entry is not None else ""
        return cls(
            paths=frozenset({skill.rel, relative}),
            fingerprint=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def matches(self, entry: RegistryEntry) -> bool:
        if entry.path not in self.paths:
            return False
        return self.fingerprint is None or entry.content_fingerprint == self.fingerprint


def check_catalog_skills(
    profile: PluginProfile,
    registry: EmbeddingRegistry,
    *,
    threshold: float,
    top_k: int = INTER_SKILL_CATALOG_TOP_K,
) -> ValidationResult:
    """Check C-inter: compare every bundled skill with the catalog's skill entries."""
    catalog_skills = registry.skill_entries
    if not profile.bundled_skills:
        return skipped_result(
            INTER_SKILL_NAME,
            INTER_SKILL_DESCRIPTION,
            INTER_SKILL_KEY,
            "Plugin bundles no skills under skills/; inter-skill catalog comparison is not applicable.",
            catalog_entries=len(catalog_skills),
        )
    if not catalog_skills:
        return skipped_result(
            INTER_SKILL_NAME,
            INTER_SKILL_DESCRIPTION,
            INTER_SKILL_KEY,
            "The local catalog has no skill entries; inter-skill comparison did not run.",
            catalog_entries=0,
        )

    result = ValidationResult(validator_name=INTER_SKILL_NAME, validator_description=INTER_SKILL_DESCRIPTION)
    comparable = [skill for skill in profile.bundled_skills if skill.entry is not None]
    for skill in profile.bundled_skills:
        if skill.entry is None:
            result.add_warning(f"Bundled skill {skill.root_relative} was not compared: {skill.skip_reason}")
    if not comparable:
        reason = "No bundled skill has a SKILL.md name and description to compare."
        record_similarity(
            result,
            INTER_SKILL_KEY,
            status=STATUS_SKIPPED,
            catalog_entries=len(catalog_skills),
            matches=[],
            reason=reason,
        )
        return result

    self_prefixes = [entry.path for entry in self_plugin_entries(profile, registry)]
    catalog_has_plugins = registry.plugin_size > 0
    scored = registry.score_skill_entries([skill.entry for skill in comparable if skill.entry is not None])
    matches: list[dict[str, Any]] = []
    excluded = 0
    excluded_paths: set[str] = set()
    for skill, scores in zip(comparable, scored, strict=True):
        identity = _SelfSkillIdentity.build(
            skill,
            self_prefixes=self_prefixes,
            catalog_has_plugins=catalog_has_plugins,
        )
        candidates: list[tuple[RegistryEntry, float]] = []
        for catalog_entry, score in scores:
            if identity.matches(catalog_entry):
                excluded += 1
                excluded_paths.add(catalog_entry.path)
                continue
            if score >= threshold:
                candidates.append((catalog_entry, score))
        candidates.sort(key=lambda item: (-item[1], item[0].path))
        for catalog_entry, score in candidates[:top_k]:
            classification, severity = classify(score)
            skill_name = skill.entry.name if skill.entry is not None else skill.rel
            matches.append({"skill": skill.root_relative, "match": catalog_entry.name, "similarity": round(score, 4)})
            result.add_finding(
                Finding(
                    category="INTER_SKILL",
                    severity=advisory_severity(severity),
                    check_name=classification,
                    message=(
                        f"Bundled skill '{skill_name}' ({skill.root_relative}) matches catalog skill "
                        f"'{catalog_entry.name}' ({catalog_entry.path}) as {classification} (score: {score:.3f})"
                    ),
                    file_path=skill.root_relative,
                    suggestion=(
                        "Review whether this bundled skill duplicates the catalog skill; reuse or reference "
                        "the existing skill instead of bundling a near-identical copy."
                    ),
                    metadata={
                        "score": round(score, 4),
                        "classification": classification,
                        "native_severity": severity.value,
                        "bundled_skill": skill.root_relative,
                        "catalog_skill": catalog_entry.name,
                        "catalog_path": catalog_entry.path,
                        "comparison_mode": "bundled-skill-vs-catalog",
                    },
                )
            )
        if len(candidates) > top_k:
            result.add_warning(
                f"Bundled skill {skill.root_relative} has {len(candidates)} catalog matches; reported the top {top_k}."
            )

    result.add_success(
        "skill_catalog_comparison",
        f"Compared {len(comparable)} bundled skill(s) against {len(catalog_skills)} catalog skill entries",
        bundled_skills=len(comparable),
        catalog_entries=len(catalog_skills),
        self_matches_excluded=excluded,
        excluded_catalog_paths=sorted(excluded_paths),
    )
    if not matches:
        result.add_success("skill_catalog_check", f"No near-duplicate catalog skills (threshold: {threshold})")
    record_similarity(
        result,
        INTER_SKILL_KEY,
        status=STATUS_COMPARED,
        catalog_entries=len(catalog_skills),
        matches=matches,
        reason=None,
    )
    return result


def _plugin_severity(similarity: float, threshold: float, *, name_collision: bool) -> tuple[str, Severity]:
    """Classify a description match; name-collision-only and member-overlap-only matches are LOW."""
    if similarity < threshold:
        return ("NAME_COLLISION" if name_collision else "MEMBER_OVERLAP"), Severity.LOW
    return classify(similarity)


def _llm_verdicts(
    profile: PluginProfile,
    candidates: list[tuple[PluginRegistryEntry, float, float]],
    *,
    llm_model: str | None,
    result: ValidationResult,
) -> list[Any | None]:
    """Return one parsed verdict (or ``None``) per candidate; failures never block."""
    from skillevaluator.deduplication.plugin.llm_analyzer import analyze_inter_plugin, build_inter_plugin_prompt
    from skillevaluator.inference import LLMClient, LLMClientError, validate_tier2_llm_prompt_batch
    from skillevaluator.inference.diagnostics import llm_failure_diagnostic, safe_llm_labels

    verdicts: list[Any | None] = [None] * len(candidates)
    try:
        prompts = [
            build_inter_plugin_prompt(
                new_plugin_name=profile.name,
                new_plugin_description=profile.description or "",
                new_plugin_members=list(profile.members),
                existing=entry,
                similarity_score=similarity,
                member_overlap=overlap,
            )
            for entry, similarity, overlap in candidates
        ]
        validate_tier2_llm_prompt_batch(
            prompts,
            context="Inter-plugin deduplication",
            max_calls=min(TIER2_LLM_MAX_CALLS, INTER_PLUGIN_TOP_K),
            max_total_chars=TIER2_LLM_MAX_TOTAL_PROMPT_CHARS,
        )
        llm = LLMClient(model=llm_model, max_tokens=LLM_VERIFY_MAX_TOKENS)
        config = llm._resolved_config()
        provider, model = safe_llm_labels(config.provider, config.model)
    except LLMClientError as exc:
        result.add_warning(f"Optional inter-plugin LLM verdict skipped: {llm_failure_diagnostic(exc)}")
        result.metadata["llm_analysis"] = {"status": "skipped", "candidates_total": len(candidates)}
        return verdicts

    failures: Counter[str] = Counter()
    for index, prompt in enumerate(prompts):
        try:
            verdicts[index] = analyze_inter_plugin(llm, prompt)
        except Exception as exc:  # provider/parse failures are missing evidence, not findings
            failures[llm_failure_diagnostic(exc)] += 1
    failed = sum(failures.values())
    result.metadata["llm_analysis"] = {
        "provider": provider,
        "model": model,
        "candidates_total": len(candidates),
        "candidates_completed": len(candidates) - failed,
        "candidates_failed": failed,
        "failures": [{"count": count, "diagnostic": diagnostic} for diagnostic, count in sorted(failures.items())],
    }
    if failed:
        result.add_warning(
            f"Optional inter-plugin LLM verdict unavailable for {failed} of {len(candidates)} match(es) "
            f"(provider: {provider}; model: {model}); similarity findings are reported without a verdict."
        )
    return verdicts


def check_catalog_plugins(
    profile: PluginProfile,
    registry: EmbeddingRegistry,
    *,
    threshold: float,
    llm_verdict: bool = False,
    llm_model: str | None = None,
    top_k: int = INTER_PLUGIN_TOP_K,
) -> ValidationResult:
    """Check B: compare the plugin description and members with catalog plugin entries.

    A catalog plugin with the same name but a different manifest is always a
    candidate: it is either an unrelated plugin that collides with this name
    or an older catalog snapshot of this plugin, and either way it is reported.
    """
    catalog_plugins = registry.plugin_entries
    if not catalog_plugins:
        return skipped_result(
            INTER_PLUGIN_NAME,
            INTER_PLUGIN_DESCRIPTION,
            INTER_PLUGIN_KEY,
            "The local catalog has no plugin entries; rebuild it with "
            "`skillevaluator similarity-check PLUGINS --type plugin --save-catalog FILE` to compare plugins.",
            catalog_entries=0,
        )
    if profile.description is None:
        return skipped_result(
            INTER_PLUGIN_NAME,
            INTER_PLUGIN_DESCRIPTION,
            INTER_PLUGIN_KEY,
            "Plugin manifest has no description; inter-plugin similarity did not run.",
            catalog_entries=len(catalog_plugins),
        )

    result = ValidationResult(validator_name=INTER_PLUGIN_NAME, validator_description=INTER_PLUGIN_DESCRIPTION)
    self_entries = self_plugin_entries(profile, registry)
    self_ids = {entry.entry_id for entry in self_entries}
    candidates: list[tuple[PluginRegistryEntry, float, float]] = []
    for entry, similarity in registry.score_plugin_text(profile.embedding_text):
        if entry.entry_id in self_ids:
            continue
        overlap = member_overlap(profile.members, entry.members)
        shared = bool(set(profile.members) & set(entry.members))
        if (
            similarity >= threshold
            or (shared and overlap >= INTER_PLUGIN_MEMBER_OVERLAP_THRESHOLD)
            or _same_name(profile, entry)
        ):
            candidates.append((entry, similarity, overlap))
    # Name collisions first, so the top-k cut never drops one.
    candidates.sort(key=lambda item: (not _same_name(profile, item[0]), -item[1], -item[2], item[0].path))
    total_candidates = len(candidates)
    candidates = candidates[:top_k]

    verdicts: list[Any | None] = [None] * len(candidates)
    if llm_verdict and candidates:
        verdicts = _llm_verdicts(profile, candidates, llm_model=llm_model, result=result)

    matches: list[dict[str, Any]] = []
    for (entry, similarity, overlap), verdict in zip(candidates, verdicts, strict=True):
        verdict_name = getattr(verdict, "verdict", None)
        matches.append(
            {
                "name": entry.name,
                "similarity": round(similarity, 4),
                "member_overlap": round(overlap, 4),
                "verdict": verdict_name,
            }
        )
        name_collision = _same_name(profile, entry)
        # A UNIQUE verdict clears a functional overlap, not a shared plugin name.
        if verdict_name == "UNIQUE" and not name_collision:
            continue
        check_name, severity = _plugin_severity(similarity, threshold, name_collision=name_collision)
        suggestion = (
            "Review whether this plugin duplicates the catalog plugin; consider extending the existing "
            "plugin instead of creating a near-identical bundle."
        )
        message = (
            f"Plugin '{profile.name}' overlaps catalog plugin '{entry.name}' ({entry.path}): "
            f"description similarity {similarity:.3f}, member skill overlap {overlap:.3f}"
        )
        if name_collision:
            message += "; same plugin name with a different manifest"
            suggestion = (
                "A catalog plugin with this name has a different manifest. Rename this plugin, or rebuild the "
                "catalog if that entry is an earlier version of this plugin."
            )
        verdict_meta: dict[str, Any] = {}
        if verdict is not None:
            from skillevaluator.deduplication.plugin.llm_analyzer import inter_plugin_verdict_to_severity

            if verdict_name == "UNIQUE":
                check_name, severity = "NAME_COLLISION", Severity.LOW
            else:
                severity = inter_plugin_verdict_to_severity(verdict)
                check_name = verdict.verdict
                suggestion = verdict.suggestion or suggestion
            verdict_meta = {
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "reasoning": verdict.reasoning,
            }
        result.add_finding(
            Finding(
                category="INTER_PLUGIN",
                severity=advisory_severity(severity),
                check_name=check_name,
                message=message,
                file_path=profile.manifest,
                suggestion=suggestion,
                metadata={
                    "catalog_plugin": entry.name,
                    "catalog_path": entry.path,
                    "similarity": round(similarity, 4),
                    "member_overlap": round(overlap, 4),
                    "name_collision": name_collision,
                    "native_severity": severity.value,
                    **verdict_meta,
                },
            )
        )
    if total_candidates > len(candidates):
        result.add_warning(f"{total_candidates} catalog plugins matched; reported the top {len(candidates)}.")
    result.add_success(
        "plugin_catalog_comparison",
        f"Compared plugin '{profile.name}' against {len(catalog_plugins)} catalog plugin entries",
        catalog_entries=len(catalog_plugins),
        self_matches_excluded=len(self_ids),
        excluded_catalog_paths=sorted(entry.path for entry in self_entries),
        llm_verdict="enabled" if llm_verdict else "disabled",
    )
    if not matches:
        result.add_success(
            "plugin_catalog_check",
            f"No overlapping catalog plugins (similarity threshold: {threshold}; "
            f"member overlap threshold: {INTER_PLUGIN_MEMBER_OVERLAP_THRESHOLD})",
        )
    record_similarity(
        result,
        INTER_PLUGIN_KEY,
        status=STATUS_COMPARED,
        catalog_entries=len(catalog_plugins),
        matches=matches,
        reason=None,
    )
    return result


__all__ = [
    "INTER_PLUGIN_DESCRIPTION",
    "INTER_PLUGIN_KEY",
    "INTER_PLUGIN_NAME",
    "INTER_SKILL_DESCRIPTION",
    "INTER_SKILL_KEY",
    "INTER_SKILL_NAME",
    "advisory_severity",
    "check_catalog_plugins",
    "check_catalog_skills",
    "record_similarity",
    "self_plugin_entries",
    "skipped_result",
]
