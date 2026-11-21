# SPDX-License-Identifier: Apache-2.0
"""Read-only views of recorded Tier 2 scope and duplicate evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillevaluator.reporting.base import additional_errors, get_skip_reason, is_cleanly_skipped

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult


_CANONICAL_CATEGORIES = {
    category.casefold(): category for category in ("DUPLICATE", "SIMILARITY", "INTER_SKILL", "INTER_PLUGIN")
}


@dataclass(frozen=True)
class Tier2SkipView:
    reason: str
    results: tuple[ValidationResult, ...]


@dataclass(frozen=True)
class Tier2ScopeView:
    id: str
    label: str
    results: tuple[ValidationResult, ...]
    completed: int | None
    duplicates: int | None
    affected: int | None
    catalog_entries: int | None
    notes: tuple[str, ...]
    catalog_not_applicable: bool = False
    not_run: bool = False
    skips: tuple[Tier2SkipView, ...] = ()


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


# The bundle catalog setup record (build, load or save) that precedes its comparisons.
_CATALOG_SETUP_NAME = "plugin catalog similarity"


def _normalized_name(result: ValidationResult) -> str:
    return " ".join(result.validator_name.casefold().replace("_", " ").replace("-", " ").split())


def _scope(result: ValidationResult) -> str | None:
    name = _normalized_name(result)
    if name == "plugin dependency deduplication":
        return "other"
    if name in {"inter skill deduplication", "inter plugin deduplication"} or "similarity" in name:
        return "across"
    if "dedup" in name or "context optimization" in name:
        return "within"
    return None


def _skip_views(results: tuple[ValidationResult, ...]) -> tuple[Tier2SkipView, ...]:
    groups: dict[str, list[ValidationResult]] = {}
    for result in results:
        if is_cleanly_skipped(result):
            groups.setdefault(get_skip_reason(result), []).append(result)
    return tuple(Tier2SkipView(reason, tuple(group)) for reason, group in groups.items())


def tier2_scope_views(results: list[ValidationResult]) -> tuple[Tier2ScopeView, ...]:
    """Keep missing completion/count metadata unknown; never rewrite results.

    Workflow status describes scope, while a failed gate may still have finished
    comparison. Completion requires recorded comparison evidence and no incomplete
    scanner or execution error. Gate-derived workflow passes and findings alone
    do not prove it.
    """
    views = []
    for scope, label, stage in (
        ("within", "Within this skill", "intra_skill"),
        ("across", "Across skills", "catalog_comparison"),
    ):
        members = tuple(result for result in results if _scope(result) == scope)
        notes: list[str] = []
        stages = []
        for result in results:
            workflow = result.metadata.get("workflow")
            if not isinstance(workflow, dict) or workflow.get("tier") != 2:
                continue
            raw = workflow.get("stages")
            evidence = raw.get(stage) if isinstance(raw, dict) else None
            if isinstance(evidence, dict):
                stages.append(evidence)
                if evidence.get("status") == "skipped" and isinstance(evidence.get("reason"), str):
                    notes.append(evidence["reason"])
        completed = 0
        unfinished = False
        known = bool(members) or any(item.get("status") == "skipped" for item in stages)
        all_complete = bool(members)
        catalog_counts: list[int] = []
        catalog_known = bool(members) and scope == "across"
        pairwise_only = bool(members) and scope == "across"
        for result in members:
            # A catalog setup record completes by building or loading the
            # catalog; its index size is not a pairwise collection scan.
            setup = scope == "across" and _normalized_name(result) == _CATALOG_SETUP_NAME
            budget = result.metadata.get("plugin_context_budget")
            if isinstance(budget, dict):
                limits, consumed = budget.get("limits"), budget.get("consumed")
                if (
                    isinstance(limits, dict)
                    and isinstance(consumed, dict)
                    and all(
                        _count(source.get(key)) is not None
                        for source, key in (
                            (limits, "chunks"),
                            (limits, "scalar_comparisons"),
                            (consumed, "chunks_collected"),
                            (consumed, "scalar_comparisons"),
                        )
                    )
                ):
                    notes.append(
                        f"Plugin context budget: {consumed['chunks_collected']}/{limits['chunks']} chunks collected; "
                        f"{consumed['scalar_comparisons']}/{limits['scalar_comparisons']} scalar comparisons consumed."
                    )
            details = {detail.check_name: detail.metadata for detail in result.success_details}
            compared_check = next(
                (
                    check
                    for check in ("catalog_compared", "skill_catalog_comparison", "plugin_catalog_comparison")
                    if check in details
                ),
                None,
            )
            indexed = details.get("index_built")
            entries = _count(indexed.get("entry_count")) if isinstance(indexed, dict) and not setup else None
            analysis = result.metadata.get("llm_analysis")
            analysis_complete = (
                isinstance(analysis, dict)
                and _count(analysis.get("clusters_total")) is not None
                and _count(analysis.get("clusters_completed")) == analysis.get("clusters_total")
                and _count(analysis.get("clusters_failed")) == 0
                and _count(analysis.get("clusters_refused")) == 0
            )
            recorded_completion = bool(
                {"index_built", "catalog_loaded"} & details.keys()
                if setup
                else compared_check or {"index_built", "similarity_check"} & details.keys()
                if scope == "across"
                else "context_dedup" in details or analysis_complete
            )
            # Bundle verdicts are optional after the catalog comparison; their
            # candidate-review status is not mandatory within-skill clustering.
            optional_catalog_analysis = (
                compared_check == "plugin_catalog_comparison"
                and isinstance(analysis, dict)
                and _count(analysis.get("candidates_total")) is not None
            )
            insufficient = any(
                detail.check_name == "context_dedup"
                and detail.message in {"No text files found in skill directory", "Not enough content to compare"}
                for detail in result.success_details
            )
            if insufficient:
                notes.append(
                    "Comparison not applicable: "
                    + next(detail.message for detail in result.success_details if detail.check_name == "context_dedup")
                )
            # A collection index of fewer than two entries has nothing to compare pairwise.
            single_entry = scope == "across" and entries is not None and entries < 2 and compared_check is None
            if single_entry:
                notes.append(f"Comparison not applicable: the collection index has {entries} entry; pairs need two.")
            clean_skip = is_cleanly_skipped(result) or insufficient or single_entry
            finished = (
                recorded_completion
                and (not isinstance(analysis, dict) or analysis_complete or optional_catalog_analysis)
                and not result.is_incomplete
                and not additional_errors(result)
                and not clean_skip
            )
            completed += bool(finished)
            all_complete = all_complete and bool(finished)
            if not finished and not clean_skip:
                known = False
                unfinished = True
            compared = details.get(compared_check)
            compared = compared if isinstance(compared, dict) else {}
            catalog_count_key = "catalog_entry_count" if compared_check == "catalog_compared" else "catalog_entries"
            total, excluded = _count(compared.get(catalog_count_key)), _count(compared.get("self_matches_excluded"))
            if total is not None and excluded is not None and excluded <= total and finished:
                catalog_counts.append(total - excluded)
                bundled = _count(compared.get("bundled_skills"))
                target = (
                    f"{bundled} bundled skill{'s' if bundled != 1 else ''}"
                    if compared_check == "skill_catalog_comparison" and bundled is not None
                    else "one target"
                )
                notes.append(
                    f"Compared {target} against {total - excluded} catalog entries; {excluded} self entries excluded."
                )
            elif not (setup and finished):
                catalog_known = False
            pairwise_only = pairwise_only and entries is not None and compared_check is None
            if scope == "across" and entries is not None and not single_entry and compared_check is None:
                notes.append(f"Pairwise comparison of {entries} collection entries; no catalog used.")
        duplicates = []
        paths: set[tuple[str, str]] = set()
        paths_known = True
        for result in members:
            for finding in result.findings:
                category = _CANONICAL_CATEGORIES.get(finding.category.casefold())
                if category not in (
                    {"DUPLICATE"} if scope == "within" else {"SIMILARITY", "INTER_SKILL", "INTER_PLUGIN"}
                ):
                    continue
                duplicates.append(finding)
                metadata = finding.metadata if isinstance(finding.metadata, dict) else {}
                if scope == "within":
                    recorded = metadata.get("source_files")
                elif category in {"INTER_SKILL", "INTER_PLUGIN"}:
                    recorded = [finding.file_path, metadata.get("catalog_path")]
                else:
                    recorded = [metadata.get("path_a"), metadata.get("path_b")]
                if (
                    not isinstance(recorded, list)
                    or not recorded
                    or any(not isinstance(p, str) or not p for p in recorded)
                ):
                    paths_known = False
                elif category in {"INTER_SKILL", "INTER_PLUGIN"}:
                    check = "skill_catalog_comparison" if category == "INTER_SKILL" else "plugin_catalog_comparison"
                    comparison = next(
                        (detail.metadata for detail in result.success_details if detail.check_name == check), None
                    )
                    comparison = comparison if isinstance(comparison, dict) else {}
                    excluded = comparison.get("excluded_catalog_paths")
                    excluded_count = _count(comparison.get("self_matches_excluded"))
                    target = result.metadata.get("plugin_catalog_target")
                    indexed_target = (
                        finding.file_path
                        if category == "INTER_SKILL"
                        else (target.get("path") if isinstance(target, dict) else None)
                    )
                    catalog_namespace = f"{category}:catalog"
                    # Reconcile collection targets only through recorded self
                    # exclusions; identical relative labels alone prove nothing.
                    if (
                        isinstance(indexed_target, str)
                        and indexed_target
                        and excluded_count is not None
                        and excluded_count > 0
                        and isinstance(excluded, list)
                        and indexed_target in excluded
                    ):
                        target_identity = (catalog_namespace, indexed_target)
                    elif excluded_count == 0 and excluded == []:
                        target_identity = (f"{category}:target", finding.file_path)
                    else:
                        paths_known = False
                        continue
                    paths.update((target_identity, (catalog_namespace, metadata["catalog_path"])))
                else:
                    paths.update((scope, path) for path in recorded)
        duplicate_count = len(duplicates) if duplicates or all_complete else None
        affected = len(paths) if paths_known and duplicate_count is not None else None
        # No member and no recorded stage: this command never runs the scope.
        not_run = not members and not stages
        if not_run:
            notes.append("Not run by this command.")
        elif not members and not notes:
            notes.append("Comparison evidence unavailable; this scope was not recorded.")
        elif members and unfinished:
            notes.append("Comparison coverage is incomplete or unavailable; counts describe recorded findings only.")
        views.append(
            Tier2ScopeView(
                scope,
                label,
                members,
                completed if known else None,
                duplicate_count,
                affected,
                sum(catalog_counts) if catalog_known else None,
                tuple(dict.fromkeys(notes)),
                scope == "within" or pairwise_only,
                not_run,
                _skip_views(members),
            )
        )
    other = tuple(result for result in results if _scope(result) in {None, "other"})
    if other:
        notes = []
        if any(_scope(result) == "other" for result in other):
            notes.append("Dependency-reference checks are separate from content comparisons.")
        if any(_scope(result) is None for result in other):
            notes.append("Comparison scope unavailable.")
        views.append(
            Tier2ScopeView(
                "other",
                "Other recorded checks",
                other,
                None,
                None,
                None,
                None,
                tuple(notes),
                skips=_skip_views(other),
            )
        )
    return tuple(views)
