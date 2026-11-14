# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight evidence checks for fully inspected bundles with reference caveats.

SkillSpector 2.13.3's MCP gate allows this narrow case while retaining the
scanner's partial/CAUTION verdict. Missing targets are not certified. The
adapter must still validate findings, risk, inventory and requested LLM work.
"""

from collections.abc import Mapping, Set
from pathlib import PurePosixPath

REFERENCE_COMPLETENESS_FIELDS = frozenset(
    {
        "total_components",
        "scanned_components",
        "coverage_percent",
        "is_complete",
        "status",
        "execution_successful",
        "fully_inspected_files",
        "partially_inspected_files",
        "entirely_uninspected_files",
        "ledger_exceptions",
        "scope_exclusions",
        "analyzer_statuses",
        "references",
        "limitations",
        "findings_before_filtering",
        "findings_after_filtering",
    }
)
_EXCEPTION_FIELDS = frozenset(
    {
        "outcome",
        "phase",
        "reason_code",
        "message",
        "path",
        "start_line",
        "end_line",
        "fatal",
        "analyzers",
    }
)
_WORK_FIELDS = ("planned_work", "completed", "partial", "skipped", "failed", "unaccounted")
_STATUS_FIELDS = frozenset({"analyzer_id", "status", "reason_code", "message", *_WORK_FIELDS})


def reference_caveat_metadata(
    data: Mapping,
    *,
    scanner_version: tuple[int, int, int] | None,
    required_analyzers: Set[str],
    universal_analyzers: Set[str],
    optional_analyzers: Set[str],
    use_llm: bool,
) -> dict[str, object] | None:
    """Return honest warning metadata only for the verified 2.x report contract."""
    if scanner_version is None or not (2, 13, 3) <= scanner_version < (3, 0, 0):
        return None
    risk = data.get("risk_assessment")
    if not isinstance(risk, dict):
        return None
    completeness = data.get("analysis_completeness")
    if (
        not isinstance(completeness, dict)
        or set(completeness) != REFERENCE_COMPLETENESS_FIELDS
        or data.get("execution_successful") is not True
        or completeness.get("execution_successful") is not True
        or completeness.get("status") != "partial"
        or completeness.get("is_complete") is not False
        or completeness.get("limitations") != []
        # Keep unfamiliar exclusion evidence outside this narrow exception.
        or completeness.get("scope_exclusions") != []
        or completeness.get("coverage_percent") != 100
        or completeness.get("partially_inspected_files") != 0
        or completeness.get("entirely_uninspected_files") != 0
    ):
        return None
    total = completeness.get("total_components")
    if (
        type(total) is not int
        or total <= 0
        or completeness.get("scanned_components") != total
        or completeness.get("fully_inspected_files") != total
    ):
        return None

    components = data.get("components")
    if not isinstance(components, list) or len(components) != total:
        return None
    component_lines: dict[str, int] = {}
    for component in components:
        if not isinstance(component, dict):
            return None
        path, lines = component.get("path"), component.get("lines")
        if (
            not isinstance(path, str)
            or not path
            or path in component_lines
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or "\\" in path
            or "\x00" in path
            or type(lines) is not int
            or lines < 0
        ):
            return None
        component_lines[path] = lines

    exceptions = completeness.get("ledger_exceptions")
    if not isinstance(exceptions, list) or not exceptions:
        return None
    exception_locations: set[tuple[str, int]] = set()
    for row in exceptions:
        if not isinstance(row, dict) or not set(row) <= _EXCEPTION_FIELDS:
            return None
        path, start, end = row.get("path"), row.get("start_line"), row.get("end_line")
        if (
            row.get("reason_code") != "reference_missing"
            or row.get("phase") != "reference_resolution"
            or row.get("outcome") != "partial"
            or row.get("fatal") is not False
            or row.get("analyzers", []) != []
            or not isinstance(path, str)
            or path not in component_lines
            or type(start) is not int
            or type(end) is not int
            or not 0 < start <= end <= component_lines[path]
        ):
            return None
        exception_locations.add((path, start))

    references = completeness.get("references")
    if not isinstance(references, list):
        return None
    missing_locations: set[tuple[str, int]] = set()
    for reference in references:
        if not isinstance(reference, dict):
            return None
        status = reference.get("status")
        if status == "missing":
            path, line = reference.get("source_path"), reference.get("line")
            if (
                not isinstance(path, str)
                or path not in component_lines
                or type(line) is not int
                or not 0 < line <= component_lines[path]
                or reference.get("target_path") is not None
                or reference.get("disposition") != "partial"
            ):
                return None
            missing_locations.add((path, line))
        elif status == "resolved":
            if (
                not isinstance(reference.get("target_path"), str)
                or reference["target_path"] not in component_lines
                or reference.get("disposition") != "analyzed"
            ):
                return None
        elif status != "rejected" or reference.get("disposition") != "out_of_scope":
            # Ambiguity, unavailable target bytes, and unknown states still block.
            return None
    if missing_locations != exception_locations:
        return None

    statuses = completeness.get("analyzer_statuses")
    if not isinstance(statuses, list):
        return None
    observed: set[str] = set()
    for row in statuses:
        if not isinstance(row, dict) or not set(row) <= _STATUS_FIELDS:
            return None
        analyzer = row.get("analyzer_id")
        if not isinstance(analyzer, str) or analyzer in observed:
            return None
        observed.add(analyzer)
        if any(type(row.get(field)) is not int or row[field] < 0 for field in _WORK_FIELDS):
            return None
        if any(row[field] != 0 for field in ("partial", "skipped", "failed", "unaccounted")):
            return None
        state = row.get("status")
        if state == "completed":
            if row["planned_work"] != row["completed"]:
                return None
        elif state == "not_applicable":
            if row.get("reason_code") != "no_applicable_files" or any(row[field] for field in _WORK_FIELDS):
                return None
        elif state == "disabled":
            if (
                use_llm
                or analyzer not in optional_analyzers
                or row.get("reason_code") != "disabled_by_configuration"
                or any(row[field] for field in _WORK_FIELDS)
            ):
                return None
        else:
            return None
        if analyzer in universal_analyzers and (state != "completed" or row["completed"] != total):
            return None
    if not (required_analyzers | universal_analyzers) <= observed:
        return None

    return {
        "status": "partial",
        "recommendation": risk.get("recommendation"),
        "reason_code": "reference_missing",
        "count": len(exceptions),
        "fully_inspected_files": total,
        "missing_targets_verified": False,
    }


def sanitize_reference_caveat_metadata(value: object) -> dict[str, object] | None:
    """Retain only bounded, canonical caveat metadata at report boundaries."""
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "status",
            "recommendation",
            "reason_code",
            "count",
            "fully_inspected_files",
            "missing_targets_verified",
        }
        or value.get("status") != "partial"
        or value.get("recommendation") not in ("CAUTION", "DO_NOT_INSTALL")
        or value.get("reason_code") != "reference_missing"
        or value.get("missing_targets_verified") is not False
        or any(
            type(value.get(key)) is not int or not 0 < value[key] < 2**31 for key in ("count", "fully_inspected_files")
        )
    ):
        return None
    return dict(value)
