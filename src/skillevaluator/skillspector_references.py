# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded projections of SkillSpector's incomplete-reference (AE1) evidence.

SkillSpector reports a High ``AE1`` finding, *Incomplete referenced artifact
analysis*, at each line that names a file it did not completely inspect. From
v2.12.0 the issue's ``evidence`` names the file (``target_path``), its
``target_disposition`` and the ``reasons`` the analysis stopped. Earlier
releases leave ``pattern`` and ``evidence`` empty and name the file only in
the ``finding`` text, such as ``scripts/helper.pl (partial)``; the reasons are
then only in the ``analysis_completeness.ledger_exceptions`` rows for that
file.

These projections explain a finding to its reader. They never change its
severity, the scan's completeness or a gate. Each text is bounded, free of
control characters and redacted with
:func:`~skillevaluator.utils.redaction.redact_sensitive_text`, and a path is
kept only when it stays inside the scanned artifact.
"""

from __future__ import annotations

import math
import os
import posixpath
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from skillevaluator.utils.redaction import redact_sensitive_text

__all__ = [
    "AE1_PATTERN",
    "AE1_RULE_ID",
    "REFERENCE_SUMMARY_KEY",
    "ReportNormalizer",
    "combine_reference_summaries",
    "diagnostic_text",
    "index_legacy_reference_reasons",
    "issue_diagnostics",
    "reference_suggestion",
    "reference_summary",
    "reference_summary_message",
    "reference_summary_record",
    "report_path",
]

AE1_RULE_ID = "AE1"
AE1_PATTERN = "Incomplete referenced artifact analysis"
REFERENCE_SUMMARY_KEY = "skillspector_reference_summary"
# The generic remediation of SkillSpector releases before v2.12.0. It suggests
# that the file is absent, which is wrong for a file the scan read in part.
_LEGACY_REMEDIATION = "Make the referenced artifact locally available and fully analyzable, or remove the reference."
_DISPOSITIONS = frozenset({"partial", "failed", "out_of_scope", "uninspected"})
_LEGACY_FINDING_RE = re.compile(r"(.+) \((partial|failed|out_of_scope|uninspected)\)")
_READ_REASONS = frozenset({"read_error", "stat_error", "file_disappeared", "missing_file_cache"})
_MAX_REASONS = 16
_MAX_OCCURRENCES = 128
_MAX_ANALYZERS = 32
_MAX_COUNT = (1 << 63) - 1
_LIMIT_FIELDS = tuple(
    (f"{prefix}_{unit}", unit == "seconds")
    for prefix in ("observed", "limit")
    for unit in ("artifacts", "bytes", "characters", "depth", "findings", "records", "seconds")
)
# Every time limit differs from run to run: SkillSpector records both the time
# a step took and, as its limit, the part of the scan's time budget left when
# the step began. A suggestion is part of its finding's ``finding_id``, which
# must stay the same across runs, so it shows only the limits that the file and
# the configuration fix; the finding's metadata keeps every limit.
_RUN_DEPENDENT_LIMITS = frozenset(field for field, seconds in _LIMIT_FIELDS if seconds)


def diagnostic_text(value: object, *, max_length: int = 4096) -> str | None:
    """*value* stripped and redacted when it is short text without control characters, else ``None``.

    Nothing else is coerced to text, so a list or an object is ``None``.
    """
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    return redact_sensitive_text(value.strip())


def _root_prefixes(scan_root: Path | None) -> list[PurePosixPath]:
    """The spellings of *scan_root*'s directory that an absolute report path may start with."""
    if scan_root is None:
        return []
    prefixes: list[PurePosixPath] = []
    try:
        roots = (Path(os.path.abspath(scan_root)), scan_root.resolve(strict=False))  # noqa: PTH100
        directories = [root.parent if root.is_file() else root for root in roots]
    except (OSError, RuntimeError):
        return prefixes
    for directory in directories:
        prefix = PurePosixPath(str(directory).replace("\\", "/"))
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


class ReportNormalizer:
    """:func:`diagnostic_text` and :func:`report_path` for one report, each distinct value computed once.

    A report repeats the same paths and reason texts across its AE1 findings
    and their reference lines. One normalizer per report redacts each distinct
    text once and resolves the scan root at most once, however many findings
    repeat them. Its caches hold only the values of that report.
    """

    def __init__(self, scan_root: Path | None = None) -> None:
        self._scan_root = scan_root
        self._prefixes: list[PurePosixPath] | None = None
        self._texts: dict[tuple[str, int], str | None] = {}
        self._paths: dict[str, str | None] = {}

    def text(self, value: object, *, max_length: int = 4096) -> str | None:
        """:func:`diagnostic_text` of *value*, computed once per distinct text."""
        if not isinstance(value, str):
            return None
        key = (value, max_length)
        if key not in self._texts:
            self._texts[key] = diagnostic_text(value, max_length=max_length)
        return self._texts[key]

    def path(self, value: object) -> str | None:
        """:func:`report_path` of *value* under this report's scan root, computed once per distinct path."""
        if not isinstance(value, str):
            return None
        if value not in self._paths:
            self._paths[value] = self._contained_path(self.text(value))
        return self._paths[value]

    def _contained_path(self, text: str | None) -> str | None:
        if text is None:
            return None
        slash_path = text.replace("\\", "/")
        if PurePosixPath(slash_path).is_absolute() or PureWindowsPath(slash_path).drive:
            if self._prefixes is None:
                self._prefixes = _root_prefixes(self._scan_root)
            for prefix in self._prefixes:
                try:
                    slash_path = PurePosixPath(slash_path).relative_to(prefix).as_posix()
                    break
                except ValueError:
                    continue
            else:
                return None
        normalized = posixpath.normpath(slash_path)
        if normalized in {".", ".."} or normalized.startswith(("../", "/")):
            return None
        return normalized


def report_path(value: object, *, scan_root: Path | None = None) -> str | None:
    """A SkillSpector report path as POSIX text relative to the scan, else ``None``.

    The path is checked lexically and never resolved, because the file an AE1
    finding names may be missing or unreadable. An absolute path counts only
    under *scan_root*. A path with control characters, a Windows drive, or a
    ``..`` that leaves the scan, and the scan root itself, are ``None``. To
    normalize the paths of a whole report, use one :class:`ReportNormalizer`.
    """
    return ReportNormalizer(scan_root).path(value)


def _line(value: object) -> int | None:
    return value if type(value) is int and 0 < value <= _MAX_COUNT else None


def _reasons(rows: object, text: Callable[..., str | None]) -> list[dict]:
    """The first reasons of *rows* with only known, valid fields, without repeats."""
    if not isinstance(rows, list):
        return []
    reasons: list[dict] = []
    for row in rows[:_MAX_REASONS]:
        if not isinstance(row, dict):
            continue
        reason_code = text(row.get("reason_code"), max_length=128)
        if not reason_code:
            continue
        reason: dict[str, Any] = {"reason_code": reason_code}
        for field, max_length in (("phase", 128), ("message", 4096)):
            if value := text(row.get(field), max_length=max_length):
                reason[field] = value
        analyzers = row.get("analyzers")
        if isinstance(analyzers, list):
            reason["analyzers"] = [name for item in analyzers[:_MAX_ANALYZERS] if (name := text(item, max_length=256))]
        for field in ("start_line", "end_line"):
            if (line := _line(row.get(field))) is not None:
                reason[field] = line
        metrics = row.get("limits")
        if isinstance(metrics, dict):
            limits = {}
            for key, accepts_float in _LIMIT_FIELDS:
                value = metrics.get(key)
                # The range is checked before any float conversion, so an
                # integer too large for a float is dropped, not raised.
                if (type(value) is int and 0 <= value <= _MAX_COUNT) or (
                    accepts_float and type(value) is float and math.isfinite(value) and 0 <= value <= _MAX_COUNT
                ):
                    limits[key] = value
            if limits:
                reason["limits"] = limits
        if reason not in reasons:
            reasons.append(reason)
    return reasons


def _reference_target(
    issue: Mapping[str, object], normalize_path: Callable[[object], str | None], text: Callable[..., str | None]
):
    """``(target, disposition, structured)`` of an AE1 issue, else ``None``.

    *structured* says whether the target came from ``evidence`` (v2.12.0 and
    later) rather than from the ``finding`` text of an earlier release, which
    counts only when the issue's own location is inside the scan.
    """
    evidence = issue.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    target = text(evidence.get("target_path"))
    structured = target is not None
    disposition = evidence.get("target_disposition")
    if not structured:
        match = _LEGACY_FINDING_RE.fullmatch(text(issue.get("finding")) or "")
        location = issue.get("location")
        source = location.get("file") if isinstance(location, dict) else None
        if match and normalize_path(source) is not None:
            target, disposition = match.groups()
    if not target or not isinstance(disposition, str) or disposition not in _DISPOSITIONS:
        return None
    target = normalize_path(target)
    return None if target is None else (target, disposition, structured)


def index_legacy_reference_reasons(
    issues: Iterable[object],
    *,
    normalize_path: Callable[[object], str | None],
    completeness: object = None,
    text: Callable[..., str | None] = diagnostic_text,
) -> dict[str, list[dict]]:
    """The ledger rows of each AE1 target that has no structured reasons, read once.

    At most one row more than an AE1 finding keeps is indexed per target, so
    the extra row still shows that the reasons were cut short. Each path is
    normalized once, however often it repeats. *normalize_path* and *text*
    are usually the methods of the report's :class:`ReportNormalizer`.
    """
    if not isinstance(completeness, dict) or not isinstance(completeness.get("ledger_exceptions"), list):
        return {}
    normalized: dict[str, str | None] = {}

    def normalize_once(path: object) -> str | None:
        if not isinstance(path, str):
            return normalize_path(path)
        if path not in normalized:
            normalized[path] = normalize_path(path)
        return normalized[path]

    by_target: dict[str, list[dict]] = {}
    for issue in issues:
        if not isinstance(issue, dict) or issue.get("id") != AE1_RULE_ID:
            continue
        target_info = _reference_target(issue, normalize_once, text)
        if target_info is None:
            continue
        target, _disposition, structured = target_info
        evidence = issue.get("evidence")
        if structured and isinstance(evidence, dict) and evidence.get("reasons"):
            continue
        by_target.setdefault(target, [])
    if not by_target:
        return {}
    for row in completeness["ledger_exceptions"]:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            continue
        target = normalize_once(row["path"])
        if target in by_target and len(by_target[target]) <= _MAX_REASONS:
            by_target[target].append(row)
    return by_target


def issue_diagnostics(
    issue: Mapping[str, object],
    *,
    normalize_path: Callable[[object], str | None],
    completeness: object = None,
    legacy_reason_index: Mapping[str, list[dict]] | None = None,
    text: Callable[..., str | None] = diagnostic_text,
) -> dict:
    """Finding metadata that explains an AE1 *issue*: the target, its reasons and the reference lines.

    ``skillspector_evidence`` holds ``target_path``, ``target_disposition``
    and up to 16 ``reasons``, with ``reasons_truncated`` when SkillSpector or
    this projection left some out. Before v2.12.0 the reasons come from the
    target's ledger rows in *completeness* (or *legacy_reason_index*), and the
    finding gets no evidence without them. ``skillspector_occurrences`` lists
    up to 128 of the issue's ``occurrences``. Any other issue gets ``{}``.
    Paths go through *normalize_path* and other text through *text*.
    """
    if issue.get("id") != AE1_RULE_ID:
        return {}
    target_info = _reference_target(issue, normalize_path, text)
    if target_info is None:
        return {}
    target, disposition, structured = target_info
    evidence = issue.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    rows = evidence.get("reasons") if structured else None
    if not rows:
        if legacy_reason_index is None:
            legacy_reason_index = index_legacy_reference_reasons(
                [issue], completeness=completeness, normalize_path=normalize_path, text=text
            )
        rows = legacy_reason_index.get(target, [])
    reasons = _reasons(rows, text)
    # A target named only in the finding text needs matching ledger rows; an
    # unreadable evidence target must not pass for a structured one.
    if not reasons and not structured:
        return {}
    safe_evidence: dict[str, Any] = {"target_path": target, "target_disposition": disposition, "reasons": reasons}
    if evidence.get("reasons_truncated") is True or (isinstance(rows, list) and len(rows) > _MAX_REASONS):
        safe_evidence["reasons_truncated"] = True
    metadata: dict[str, Any] = {"skillspector_evidence": safe_evidence}
    occurrences = issue.get("occurrences")
    if isinstance(occurrences, list):
        safe_occurrences = []
        for occurrence in occurrences[:_MAX_OCCURRENCES]:
            if not isinstance(occurrence, dict) or (path := normalize_path(occurrence.get("file"))) is None:
                continue
            safe_occurrence: dict[str, Any] = {"file": path}
            for field in ("start_line", "end_line"):
                if (line := _line(occurrence.get(field))) is not None:
                    safe_occurrence[field] = line
            safe_occurrences.append(safe_occurrence)
        metadata["skillspector_occurrences"] = safe_occurrences
        if len(occurrences) > _MAX_OCCURRENCES:
            metadata["skillspector_occurrences_truncated"] = True
    return metadata


def _reason_details(reasons: list[dict]) -> list[str]:
    """One text per distinct reason, with the analyzers of rows that differ only in them joined.

    The run time and the time budget (``observed_seconds``, ``limit_seconds``)
    are left out, so the text, and the ``finding_id`` that hashes it, is the
    same in every run of one scan.
    """
    grouped: dict[tuple, list[str]] = {}
    for reason in reasons:
        limits = sorted(
            (key, value) for key, value in (reason.get("limits") or {}).items() if key not in _RUN_DEPENDENT_LIMITS
        )
        key = (reason["reason_code"], reason.get("start_line"), reason.get("message"), tuple(limits))
        analyzers = grouped.setdefault(key, [])
        analyzers.extend(name for name in reason.get("analyzers", []) if name not in analyzers)
    details = []
    for (reason_code, start_line, message, limits), analyzers in grouped.items():
        detail = reason_code
        if analyzers:
            detail += f" ({', '.join(analyzers)})"
        if start_line:
            detail += f" at target line {start_line}"
        if message:
            detail += f": {message.rstrip('.')}"
        if limits:
            detail += "; " + ", ".join(f"{key}={value}" for key, value in limits)
        details.append(detail + ".")
    return details


def reference_suggestion(
    evidence: Mapping[str, Any], remediation: object, *, text: Callable[..., str | None] = diagnostic_text
) -> str:
    """The suggestion of an AE1 finding: what was not inspected, why, and what to do.

    SkillSpector's own remediation is kept unless it is missing or the generic
    text of releases before v2.12.0. A specific remedy is then chosen only from
    a complete list of reasons, because the visible part of a truncated list
    cannot show that one remedy covers every cause.
    """
    reasons = evidence["reasons"]
    truncated = evidence.get("reasons_truncated") is True
    action = text(remediation)
    if not action or action == _LEGACY_REMEDIATION:
        codes = set() if truncated else {reason["reason_code"] for reason in reasons}
        if codes == {"static_parse_limit"}:
            action = (
                "Check that SkillSpector is current (`skillevaluator doctor --tier 1` shows its version) and rerun "
                "the scan. If the limit remains and the source is valid, report a minimal reproducer to SkillSpector."
            )
        elif codes and codes <= _READ_REASONS:
            action = "Make sure the referenced file is readable where the scan runs, then rerun the scan."
        else:
            action = "Resolve the reported analysis limitation and rerun the scan."
    target, disposition = evidence["target_path"], evidence["target_disposition"]
    summary = f"Referenced artifact {target} was not completely inspected ({disposition})."
    if truncated:
        summary += (
            " SkillSpector reported more reasons than are shown here: run `skillspector scan` with "
            "`--format json` and review the rows for this file in "
            "`analysis_completeness.ledger_exceptions`, then address every limitation before rerunning the scan."
        )
    return " ".join([summary, *_reason_details(reasons), action])


def reference_summary(findings: Iterable[object]) -> dict | None:
    """Count the referenced artifacts and distinct reference lines behind *findings*' AE1 evidence.

    A finding counts when its metadata holds ``skillspector_evidence`` from
    :func:`issue_diagnostics`. Its reference lines are its
    ``skillspector_occurrences``, else its own location, and the count is a
    lower bound when an occurrence list was cut short.
    """
    targets: dict[str, set[tuple[object, object]]] = {}
    truncated = False
    for finding in findings:
        metadata = getattr(finding, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        evidence = metadata.get("skillspector_evidence")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("target_path"), str):
            continue
        occurrences = metadata.get("skillspector_occurrences") or [
            {"file": getattr(finding, "file_path", None), "start_line": getattr(finding, "line_number", None)}
        ]
        truncated |= metadata.get("skillspector_occurrences_truncated") is True
        targets.setdefault(evidence["target_path"], set()).update(
            (occurrence.get("file"), occurrence.get("start_line"))
            for occurrence in occurrences
            if isinstance(occurrence, dict)
        )
    if not targets:
        return None
    return {
        "affected_artifacts": len(targets),
        "reference_occurrences": sum(len(lines) for lines in targets.values()),
        "reference_occurrences_truncated": truncated,
    }


def reference_summary_record(summary: object) -> dict | None:
    """*summary* with only its count fields when they are valid, else ``None``."""
    if not isinstance(summary, dict):
        return None
    targets = summary.get("affected_artifacts")
    occurrences = summary.get("reference_occurrences")
    if type(targets) is not int or not 0 < targets <= _MAX_COUNT:
        return None
    if type(occurrences) is not int or not 0 <= occurrences <= _MAX_COUNT:
        return None
    return {
        "affected_artifacts": targets,
        "reference_occurrences": occurrences,
        "reference_occurrences_truncated": summary.get("reference_occurrences_truncated") is True,
    }


def combine_reference_summaries(summaries: Iterable[object]) -> dict | None:
    """The totals of the valid *summaries*, such as those of each skill in a folder."""
    records = [record for summary in summaries if (record := reference_summary_record(summary)) is not None]
    if not records:
        return None
    return {
        "affected_artifacts": min(sum(record["affected_artifacts"] for record in records), _MAX_COUNT),
        "reference_occurrences": min(sum(record["reference_occurrences"] for record in records), _MAX_COUNT),
        "reference_occurrences_truncated": any(record["reference_occurrences_truncated"] for record in records),
    }


def reference_summary_message(summary: object) -> str | None:
    """The one-line text of a valid *summary*, the same in every report, else ``None``.

    It says only that each finding names its file and any reasons, because a
    finding counts even when SkillSpector gave no reason for its file.
    """
    record = reference_summary_record(summary)
    if record is None:
        return None
    occurrences = record["reference_occurrences"]
    count = f"at least {occurrences}" if record["reference_occurrences_truncated"] else str(occurrences)
    return (
        f"skillspector could not completely inspect {record['affected_artifacts']} referenced artifact(s) across "
        f"{count} reference occurrence(s); each AE1 finding names the artifact and any reasons SkillSpector gave"
    )
