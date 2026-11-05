# SPDX-License-Identifier: Apache-2.0
"""Read-only views of recorded Tier 2 scope and duplicate evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillevaluator.reporting.base import additional_errors, is_cleanly_skipped

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult


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


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def _scope(result: ValidationResult) -> str | None:
    name = " ".join(result.validator_name.casefold().replace("_", " ").replace("-", " ").split())
    if "similarity" in name:
        return "across"
    if "dedup" in name or "context optimization" in name:
        return "within"
    return None


def tier2_scope_views(results: list[ValidationResult]) -> tuple[Tier2ScopeView, ...]:
    """Keep missing completion/count metadata unknown; never rewrite results.

    Workflow status describes scope, while a failed gate may still have finished
    comparison. Completion requires a recorded stage pass or comparison evidence,
    and no incomplete scanner or execution error. Findings alone do not prove it.
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
        known = bool(members) or any(item.get("status") == "skipped" for item in stages)
        all_complete = bool(members)
        catalog_counts: list[int] = []
        catalog_known = bool(members) and scope == "across"
        for result in members:
            details = {detail.check_name: detail.metadata for detail in result.success_details}
            workflow = result.metadata.get("workflow")
            raw = workflow.get("stages") if isinstance(workflow, dict) else None
            evidence = raw.get(stage) if isinstance(raw, dict) else None
            stage_pass = isinstance(evidence, dict) and evidence.get("status") == "passed"
            analysis = result.metadata.get("llm_analysis")
            analysis_complete = (
                isinstance(analysis, dict)
                and _count(analysis.get("clusters_total")) is not None
                and _count(analysis.get("clusters_completed")) == analysis.get("clusters_total")
                and _count(analysis.get("clusters_failed")) == 0
                and _count(analysis.get("clusters_refused")) == 0
            )
            recorded_completion = stage_pass or bool(
                {"catalog_compared", "index_built", "similarity_check"} & details.keys()
                if scope == "across"
                else "context_dedup" in details or analysis_complete
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
            clean_skip = is_cleanly_skipped(result) or insufficient
            finished = (
                recorded_completion
                and (not isinstance(analysis, dict) or analysis_complete)
                and not result.is_incomplete
                and not additional_errors(result)
                and not clean_skip
            )
            completed += bool(finished)
            all_complete = all_complete and bool(finished)
            if not finished and not clean_skip:
                known = False
            compared = details.get("catalog_compared")
            compared = compared if isinstance(compared, dict) else {}
            total, excluded = _count(compared.get("catalog_entry_count")), _count(compared.get("self_matches_excluded"))
            if total is not None and excluded is not None and excluded <= total and finished:
                catalog_counts.append(total - excluded)
                notes.append(
                    f"Compared one target against {total - excluded} catalog entries; {excluded} self entries excluded."
                )
            else:
                catalog_known = False
            indexed = details.get("index_built")
            entries = _count(indexed.get("entry_count")) if isinstance(indexed, dict) else None
            if scope == "across" and entries is not None:
                notes.append(f"Pairwise comparison of {entries} collection entries; no catalog used.")
        duplicates = []
        paths: set[str] = set()
        paths_known = True
        for result in members:
            for finding in result.findings:
                category = finding.category.casefold()
                if category not in ({"duplicate"} if scope == "within" else {"similarity"}):
                    continue
                duplicates.append(finding)
                metadata = finding.metadata if isinstance(finding.metadata, dict) else {}
                recorded = (
                    metadata.get("source_files")
                    if scope == "within"
                    else [metadata.get("path_a"), metadata.get("path_b")]
                )
                if (
                    not isinstance(recorded, list)
                    or not recorded
                    or any(not isinstance(p, str) or not p for p in recorded)
                ):
                    paths_known = False
                else:
                    paths.update(recorded)
        duplicate_count = len(duplicates) if duplicates or all_complete else None
        affected = len(paths) if paths_known and duplicate_count is not None else None
        if not members and not notes:
            notes.append("Comparison evidence unavailable; this scope was not recorded.")
        elif members and not all_complete:
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
            )
        )
    unknown = tuple(result for result in results if _scope(result) is None)
    if unknown:
        views.append(
            Tier2ScopeView(
                "other", "Other recorded checks", unknown, None, None, None, None, ("Comparison scope unavailable.",)
            )
        )
    return tuple(views)
