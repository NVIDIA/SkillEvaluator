# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- shared trusted CI comment helper.
"""Summarize already-validated native results without trusting producer metrics."""

from __future__ import annotations

import math
import unicodedata
from typing import TYPE_CHECKING

from skillevaluator.models import Severity
from skillevaluator.reporting.finding_summary import summarize_messages
from skillevaluator.reporting.run_status import Status, result_status

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult

_ISSUE_SEVERITIES = (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW)
_PLAIN_TEXT = str.maketrans(dict.fromkeys("<>[]`*_~|\\", " ") | {"@": "(at)"})
_TOP_ISSUE_LIMIT = 160


def _plain_text(message: str) -> str | None:
    # Keep only retained finding/gate text. In particular, do not copy source
    # lines, legacy provider diagnostics, metadata or suggestions into a comment.
    printable = "".join(" " if unicodedata.category(character).startswith("C") else character for character in message)
    text = " ".join(printable.translate(_PLAIN_TEXT).split())
    if len(text) > _TOP_ISSUE_LIMIT:
        text = text[: _TOP_ISSUE_LIMIT - 1].rstrip() + "…"
    return text or None


def _number(value: object, *, minimum: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= 1


def _tier3_metrics(results: list[ValidationResult]) -> tuple[dict, bool]:
    payloads = [
        result.metadata["agent_eval"] for result in results if isinstance(result.metadata.get("agent_eval"), dict)
    ]
    if len(payloads) != 1:
        return {}, False
    payload = payloads[0]
    summary = payload.get("summary") or payload
    expected, scored = summary.get("expected_attempts"), summary.get("scored_attempts")
    coverage_known = type(expected) is int and type(scored) is int and expected > 0 and 0 <= scored <= expected
    metrics = {"coverage": f"{scored}/{expected}"} if coverage_known else {}
    complete = (
        coverage_known
        and scored == expected
        and payload.get("execution_status", summary.get("execution_status")) == "succeeded"
        and not payload.get("execution_errors")
        and not summary.get("execution_errors")
    )
    if complete:
        for name, key, minimum in (("score", "overall_score", 0), ("skill_lift", "overall_lift", -1)):
            value = summary.get(key)
            if _number(value, minimum=minimum):
                metrics[name] = value
    return metrics, complete


def summarize_results(results: list[ValidationResult], *, tier: int, native: dict | None = None) -> dict:
    """Return observed issue counts, completeness and one safe highest-severity issue.

    Call only after ``retained_results`` validates the native document. Issue
    counts include critical/high/medium/low findings, deduplicated by canonical
    finding ID with their highest severity, plus independent errors and warnings.
    Canonical message reconciliation excludes finding mirrors and workflow-scope
    warnings; INFO and success records require no action. Stored summary counters
    are ignored. Counts remain observed lower bounds when ``issue_count_known``
    is false; an incomplete zero must not imply clean. With no finding, a retained
    gate failure or generic diagnostic notice explains the issue without copying
    legacy execution/provider text into the comment.

    ``native`` is accepted for the caller's uniform interface but is intentionally
    unused: optional Tier 3 values come from validated result metadata instead.
    The renderer must still escape all text for its output format.
    """
    unique = {}
    for result in results:
        for finding in result.findings:
            if finding.severity not in _ISSUE_SEVERITIES:
                continue
            identity = finding.finding_id
            previous = unique.get(identity)
            if previous is None or _ISSUE_SEVERITIES.index(finding.severity) < _ISSUE_SEVERITIES.index(
                previous.severity
            ):
                unique[identity] = finding
    findings = list(unique.values())
    known = bool(results) and all(
        not result.is_incomplete
        and not result.metadata.get("skipped")
        and result_status(result) not in {Status.INCOMPLETE, Status.SKIPPED, Status.NOT_RUN}
        for result in results
    )
    highest = min(findings, key=lambda finding: _ISSUE_SEVERITIES.index(finding.severity)) if findings else None
    messages = summarize_messages(results)
    independent = messages["independent_errors"] + messages["independent_warnings"]
    message = highest.message if highest else next((gate for result in results for gate in result.gate_failures), None)
    if not message and independent:
        message = "Additional validation errors/warnings; see report"
    metrics = {
        "issue_count": len(findings) + independent,
        "issue_count_known": known,
        "critical_count": sum(finding.severity == Severity.CRITICAL for finding in findings),
        "high_count": sum(finding.severity == Severity.HIGH for finding in findings),
        "top_issue": _plain_text(message) if message else None,
    }
    if tier == 3:
        execution, complete = _tier3_metrics(results)
        metrics.update(execution)
        metrics["issue_count_known"] = known and complete
    return metrics
