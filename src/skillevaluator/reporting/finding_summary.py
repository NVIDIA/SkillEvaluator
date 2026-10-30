# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical finding reconciliation shared by every report format.

Validators can report the same problem more than once: one check fires on
several lines, or two validators flag the same line. Each reporter used to
derive its own totals from raw validator counts, so one run could show
different issue counts in JSON, HTML, and Markdown. This module groups raw
findings by their stable ``finding_id``, keeps every distinct
``occurrence_id`` as a location, and returns one set of counts that every
reporter emits.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.base import additional_errors, additional_warnings

if TYPE_CHECKING:
    from collections.abc import Mapping


_SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
# ValidationResult.add_success records each passed check as "[OK] check: message".
_SUCCESS_PREFIX = "[OK] "


def _occurrence_sort_key(item: dict[str, Any]) -> tuple[str, int, str, str]:
    """Order occurrences by location without failing on malformed values."""
    line_number = item["line_number"]
    return (
        str(item["file_path"] or ""),
        line_number if isinstance(line_number, int) else 0,
        str(item["validator"] or ""),
        item["occurrence_id"],
    )


def _finalize_summary(groups: dict[str, dict[str, Any]], raw_by_severity: dict[str, int]) -> dict[str, Any]:
    """Finalize counts and stable ordering for a set of finding groups."""
    unique_by_severity = dict.fromkeys(_SEVERITY_ORDER, 0)
    for group in groups.values():
        unique_by_severity.setdefault(group["severity"], 0)
        unique_by_severity[group["severity"]] += 1
        group["validators"].sort()
        group["occurrences"].sort(key=_occurrence_sort_key)
        group["occurrence_count"] = len(group["occurrences"])

    raw_count = sum(raw_by_severity.values())
    distinct_occurrences = sum(group["occurrence_count"] for group in groups.values())
    unique_count = len(groups)
    return {
        "raw_occurrences": raw_count,
        "distinct_occurrences": distinct_occurrences,
        "unique_findings": unique_count,
        "duplicates_removed": raw_count - distinct_occurrences,
        "occurrences_grouped": distinct_occurrences - unique_count,
        "raw_by_severity": raw_by_severity,
        "unique_by_severity": unique_by_severity,
        "groups": sorted(groups.values(), key=lambda group: group["finding_id"]),
    }


def summarize_findings(results: list[ValidationResult]) -> dict[str, Any]:
    """Group raw findings by stable identity and retain every occurrence."""
    groups: dict[str, dict[str, Any]] = {}
    seen_occurrences: dict[str, set[str]] = {}
    raw_by_severity = dict.fromkeys(_SEVERITY_ORDER, 0)

    for result in results:
        for finding in result.findings:
            severity = (
                finding.severity.value if isinstance(finding.severity, Severity) else str(finding.severity).lower()
            )
            raw_by_severity.setdefault(severity, 0)
            raw_by_severity[severity] += 1
            finding_id = finding.finding_id
            occurrence_id = finding.occurrence_id
            group = groups.get(finding_id)
            if group is None:
                group = {
                    "finding_id": finding_id,
                    "category": finding.category,
                    "severity": severity,
                    "check_name": finding.check_name,
                    "message": finding.message,
                    "suggestion": finding.suggestion,
                    "metadata": finding.metadata or None,
                    "validators": [],
                    "occurrences": [],
                }
                groups[finding_id] = group
                seen_occurrences[finding_id] = set()
            elif _SEVERITY_ORDER.get(severity, 0) > _SEVERITY_ORDER.get(group["severity"], 0):
                group["severity"] = severity

            if result.validator_name not in group["validators"]:
                group["validators"].append(result.validator_name)
            if occurrence_id not in seen_occurrences[finding_id]:
                seen_occurrences[finding_id].add(occurrence_id)
                group["occurrences"].append(
                    {
                        "occurrence_id": occurrence_id,
                        "validator": result.validator_name,
                        "file_path": finding.file_path,
                        "line_number": finding.line_number,
                        "line_content": finding.line_content,
                    }
                )

    return _finalize_summary(groups, raw_by_severity)


def merge_finding_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-result summaries without hashing the same findings again."""
    groups: dict[str, dict[str, Any]] = {}
    raw_by_severity = dict.fromkeys(_SEVERITY_ORDER, 0)
    for summary in summaries:
        for severity, count in summary["raw_by_severity"].items():
            raw_by_severity.setdefault(severity, 0)
            raw_by_severity[severity] += count
        for incoming in summary["groups"]:
            finding_id = incoming["finding_id"]
            group = groups.get(finding_id)
            if group is None:
                groups[finding_id] = deepcopy(incoming)
                continue
            if _SEVERITY_ORDER.get(incoming["severity"], 0) > _SEVERITY_ORDER.get(group["severity"], 0):
                group["severity"] = incoming["severity"]
            group["validators"] = sorted(set(group["validators"]) | set(incoming["validators"]))
            existing_occurrences = {item["occurrence_id"] for item in group["occurrences"]}
            group["occurrences"].extend(
                deepcopy(item) for item in incoming["occurrences"] if item["occurrence_id"] not in existing_occurrences
            )

    return _finalize_summary(groups, raw_by_severity)


def groups_by_display_priority(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a summary's groups with the most severe and most repeated first.

    ``groups`` stays ordered by ``finding_id`` so machine output is stable.
    A capped human-readable list uses this order instead, so a critical
    finding is never hidden behind low-severity ones.
    """
    return sorted(
        summary["groups"],
        key=lambda group: (
            -_SEVERITY_ORDER.get(group["severity"], 0),
            -group["occurrence_count"],
            group["finding_id"],
        ),
    )


def finding_counts(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a summary's reconciliation counts without its finding groups."""
    return {key: value for key, value in summary.items() if key != "groups"}


_IDENTITY_FIELDS = ("finding_id", "occurrence_id")


def refresh_finding_identities(document: dict[str, Any]) -> None:
    """Recompute a JSON report's finding IDs and groups from its current finding text.

    CI removes detected credentials from a report after it is written. IDs
    hashed before that would no longer match the redacted text, and a short
    redacted value could be recovered by hashing guesses against them. Call
    this after redacting: it replaces only the ID and reconciliation fields
    the report already has, so a report written before finding IDs existed
    keeps its shape. Message counts do not depend on text and are kept.
    """
    rows = document.get("results")
    if not isinstance(rows, list):
        return
    summaries = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("findings"), list):
            continue
        result = ValidationResult(validator_name=str(row.get("validator") or ""))
        for item in row["findings"]:
            if not isinstance(item, dict):
                continue
            finding = Finding(
                category=item.get("category"),
                severity=item.get("severity"),
                check_name=item.get("check_name"),
                message=item.get("message"),
                file_path=item.get("file_path"),
                line_number=item.get("line_number"),
                line_content=item.get("line_content"),
                suggestion=item.get("suggestion"),
                metadata=item.get("metadata") or {},
            )
            for key in _IDENTITY_FIELDS:
                if key in item:
                    item[key] = getattr(finding, key)
            result.findings.append(finding)
        summary = summarize_findings([result])
        if "finding_counts" in row:
            row["finding_counts"] = finding_counts(summary)
        summaries.append(summary)
    merged = merge_finding_summaries(summaries)
    if "finding_counts" in document:
        document["finding_counts"] = finding_counts(merged)
    if "finding_groups" in document:
        document["finding_groups"] = merged["groups"]


def summarize_messages(results: list[ValidationResult]) -> dict[str, int]:
    """Count string messages that no structured finding represents.

    Findings also add a mirrored string to a result's ``errors`` or
    ``warnings`` list. ``independent_errors`` and ``independent_warnings``
    leave those mirrors out, so they never count a finding twice. They cover
    problems with no ``Finding`` behind them: failed gates such as a minimum
    score, scanner and execution errors, and a missing or failed scanner.
    ``informational`` counts informational messages. It leaves out the
    ``[OK]`` line that ``add_success`` records for every passed check; those
    checks are listed in ``success_details``.
    """
    return {
        "informational": sum(
            1 for result in results for message in result.messages if not str(message).startswith(_SUCCESS_PREFIX)
        ),
        "independent_errors": sum(len(additional_errors(result)) for result in results),
        "independent_warnings": sum(len(additional_warnings(result)) for result in results),
    }


def total_issue_count(counts: Mapping[str, Any], messages: Mapping[str, int]) -> int:
    """Return the headline issue count: unique findings plus independent errors and warnings."""
    return counts["unique_findings"] + messages["independent_errors"] + messages["independent_warnings"]


def count_phrase(count: int, singular: str, plural: str | None = None) -> str:
    """Return ``count`` followed by the singular or plural noun."""
    return f"{count} {singular if count == 1 else plural or f'{singular}s'}"


def describe_issue_counts(
    counts: Mapping[str, Any],
    messages: Mapping[str, int],
    *,
    unique_suffix: str = "",
    raw_records: bool = False,
    duplicates: bool = False,
) -> str:
    """Describe a report's issue counts in the shared HTML and Markdown wording.

    Unique findings and their distinct occurrences come first; ``raw_records``
    and ``duplicates`` add the raw record and reconciled duplicate totals.
    Independent errors and warnings follow, so a run that fails a gate or
    leaves a scan incomplete without any finding never reads as zero issues.
    ``unique_suffix`` is appended to the unique count, for example a severity
    breakdown.
    """
    parts = [
        count_phrase(counts["unique_findings"], "unique finding") + unique_suffix,
        count_phrase(counts["distinct_occurrences"], "occurrence"),
    ]
    if raw_records:
        parts.append(count_phrase(counts["raw_occurrences"], "raw record"))
    if duplicates and counts["duplicates_removed"]:
        parts.append(f"{count_phrase(counts['duplicates_removed'], 'duplicate')} reconciled")
    finding_text = " / ".join(parts)

    independent = [
        count_phrase(count, noun)
        for count, noun in (
            (messages["independent_errors"], "error"),
            (messages["independent_warnings"], "warning"),
        )
        if count
    ]
    if not independent:
        return finding_text
    independent_text = f"{' and '.join(independent)} without a finding"
    if not counts["unique_findings"]:
        return independent_text
    return f"{finding_text}; {independent_text}"
