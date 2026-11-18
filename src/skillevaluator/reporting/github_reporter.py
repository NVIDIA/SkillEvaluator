# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GitHub Actions workflow-command reporter.

Prints ``::error``, ``::warning``, and ``::notice`` lines for Tier 1 findings so a
workflow that runs the CLI directly can annotate the pull request without uploading
SARIF. Output goes to stdout only; this format is not a report file.

GitHub shows at most 10 error and 10 warning annotations per step. Extra findings
are counted on the closing notice rather than omitted silently.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.reporting.base import ReporterBase, get_skip_reason, is_cleanly_skipped, passes_required_gate
from skillevaluator.reporting.run_status import Status, result_status
from skillevaluator.reporting.sarif_reporter import (
    _finding_severity_value,
    _positive_start_line,
    _quote_uri_path,
    _resolve_artifact_path,
    _workspace_relative_path,
)
from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.rich_markup import strip_terminal_controls

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult

_ERROR_CAP = 10
_WARNING_CAP = 10
_MESSAGE_MAX = 4000
_TITLE_MAX = 256
_LEVEL_RANK = {"error": 0, "warning": 1, "notice": 2}
_KIND_RANK = {"incomplete": 0, "gate": 1, "error": 2, "finding": 3}
_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4, "": 5}


def annotation_text(value: str, *, property_value: bool = False) -> str:
    """Escape GitHub workflow-command syntax; finding text must never create commands."""
    escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return escaped.replace(":", "%3A").replace(",", "%2C") if property_value else escaped


def _utf8_text(value: object) -> str:
    """Return *value* as text, writing lone surrogates as backslash escapes."""
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    try:
        return text.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")
    except UnicodeEncodeError:
        return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _clean_text(value: object, *, max_len: int = _MESSAGE_MAX) -> str:
    """Strip controls, mask credentials, and bound length before command escaping."""
    return redact_sensitive_text(strip_terminal_controls(_utf8_text(value)), max_len=max_len)


def _github_path_text(posix: str) -> str:
    """Return a stdout-safe POSIX path; non-UTF-8 names use the SARIF/JUnit encoding."""
    try:
        posix.encode("utf-8")
    except UnicodeEncodeError:
        return _quote_uri_path(posix)
    return posix


def _github_file_path(
    file_path: object,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> str | None:
    """Return a repository-relative POSIX path, or None when the path is outside the repo."""
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    raw = file_path.strip()
    path = _resolve_artifact_path(raw, scan_root)
    if workspace_root is not None and (path.is_absolute() or path.root):
        relative = _workspace_relative_path(path, raw, workspace_root, scan_root)
        if relative is None:
            return None
        return _github_path_text(relative.as_posix())
    if scan_root is not None:
        try:
            return _github_path_text(path.resolve().relative_to(scan_root.resolve()).as_posix())
        except ValueError:
            if path.is_absolute() or path.root:
                return None
    if path.is_absolute() or path.root:
        return None
    return _github_path_text(raw.replace("\\", "/"))


def _end_line(finding: Finding) -> int | None:
    metadata = finding.metadata if isinstance(finding.metadata, dict) else {}
    for key in ("end_line", "endLine"):
        line = _positive_start_line(metadata.get(key))
        if line is not None:
            return line
    return None


def _level_for_severity(severity: str) -> str:
    if severity in {"critical", "high"}:
        return "error"
    if severity in {"medium", "low"}:
        return "warning"
    return "notice"


def _gating(result: ValidationResult) -> dict | None:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    gating = metadata.get("gating")
    return gating if isinstance(gating, dict) else None


def _advisory(result: ValidationResult) -> bool:
    gating = _gating(result)
    return gating is not None and not gating.get("blocking", True)


def _incomplete_reason(result: ValidationResult, tool: str) -> str:
    needle = tool.casefold()
    for text in (*result.warnings, *result.errors):
        if isinstance(text, str) and needle in text.casefold():
            return text
    return f"{tool} scan did not complete"


def _title(validator: str, check: str) -> str:
    validator = _clean_text(validator, max_len=_TITLE_MAX) or "check"
    check = _clean_text(check, max_len=_TITLE_MAX)
    if check and check != validator:
        return f"{validator}/{check}"[:_TITLE_MAX]
    return validator[:_TITLE_MAX]


def _failure_reason(result: ValidationResult) -> str:
    if result.errors:
        return result.errors[0]
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    payload = metadata.get("agent_eval")
    if isinstance(payload, dict) and payload.get("execution_status") == "succeeded" and payload.get("verdict"):
        return f"Tier 3 verdict {str(payload['verdict']).upper()} does not pass the required gate"
    return get_skip_reason(result) or "failed"


@dataclass(slots=True, frozen=True)
class _Annotation:
    level: str
    kind: str
    severity: str
    path: str
    line: int | None
    end_line: int | None
    check: str
    title: str
    message: str

    def sort_key(self) -> tuple:
        return (
            _LEVEL_RANK[self.level],
            _KIND_RANK.get(self.kind, 9),
            _SEVERITY_RANK.get(self.severity, 9),
            self.path,
            self.line if self.line is not None else 0,
            self.check,
            self.message,
        )

    def format_line(self) -> str:
        properties: list[str] = [f"title={annotation_text(self.title, property_value=True)}"]
        if self.path:
            properties.append(f"file={annotation_text(self.path, property_value=True)}")
            if self.line is not None:
                properties.append(f"line={self.line}")
                if self.end_line is not None and self.end_line >= self.line:
                    properties.append(f"endLine={self.end_line}")
        body = annotation_text(_clean_text(self.message))
        return f"::{self.level} {','.join(properties)}::{body}"


def _finding_annotation(
    finding: Finding,
    validator: str,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> _Annotation:
    severity = _finding_severity_value(finding)
    path = _github_file_path(finding.file_path, workspace_root, scan_root) or ""
    line = _positive_start_line(finding.line_number)
    suggestion = _clean_text(finding.suggestion) if finding.suggestion else ""
    message = _clean_text(finding.message)
    if suggestion:
        message = f"{message} Suggestion: {suggestion}"
    return _Annotation(
        level=_level_for_severity(severity),
        kind="finding",
        severity=severity,
        path=path,
        line=line,
        end_line=_end_line(finding),
        check=_clean_text(finding.check_name, max_len=_TITLE_MAX),
        title=_title(validator, finding.check_name),
        message=message,
    )


def _annotations_for_result(
    result: ValidationResult,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> list[_Annotation]:
    validator = result.validator_name or "check"
    advisory = _advisory(result)
    annotations: list[_Annotation] = []

    if is_cleanly_skipped(result) and result_status(result) == Status.SKIPPED:
        return annotations

    for finding in result.findings:
        annotations.append(_finding_annotation(finding, validator, workspace_root, scan_root))

    gate_failures = set(result.gate_failures)
    for error in result.independent_errors():
        is_gate = error in gate_failures
        if advisory:
            continue
        annotations.append(
            _Annotation(
                level="error",
                kind="gate" if is_gate else "error",
                severity="high",
                path="",
                line=None,
                end_line=None,
                check="gate_failure" if is_gate else "error",
                title=_title(validator, "gate_failure" if is_gate else "error"),
                message=_clean_text(error),
            )
        )

    for tool in result.incomplete_scans:
        annotations.append(
            _Annotation(
                level="warning",
                kind="incomplete",
                severity="medium",
                path="",
                line=None,
                end_line=None,
                check=f"incomplete/{tool}",
                title=_title(validator, f"incomplete/{tool}"),
                message=_clean_text(_incomplete_reason(result, tool)),
            )
        )

    if (
        not advisory
        and not passes_required_gate(result)
        and not any(item.kind in {"gate", "error", "incomplete"} or item.level == "error" for item in annotations)
    ):
        annotations.append(
            _Annotation(
                level="error",
                kind="gate",
                severity="high",
                path="",
                line=None,
                end_line=None,
                check="gate_failure",
                title=_title(validator, "gate_failure"),
                message=_clean_text(_failure_reason(result)),
            )
        )
    return annotations


def _skill_counts(results: list[ValidationResult]) -> tuple[int, int, int, int]:
    """Return (checked, passed, failed, incomplete) for this render."""
    if not results:
        return 0, 0, 0, 0
    statuses = [result_status(result) for result in results]
    if any(status == Status.FAILED for status in statuses):
        return 1, 0, 1, 0
    if any(status == Status.INCOMPLETE for status in statuses):
        return 1, 0, 0, 1
    return 1, 1, 0, 0


def _summary_message(counts: tuple[int, int, int, int], omitted: int) -> str:
    checked, passed, failed, incomplete = counts
    skill_word = "skill" if checked == 1 else "skills"
    text = f"{checked} {skill_word} checked, {passed} passed, {failed} failed, {incomplete} incomplete"
    if omitted:
        text += (
            f". {omitted} more were not annotated "
            "(GitHub limits 10 errors and 10 warnings per step); see the other reports"
        )
    return text


def _select(annotations: list[_Annotation]) -> tuple[list[_Annotation], int]:
    chosen: list[_Annotation] = []
    errors = 0
    warnings = 0
    omitted = 0
    for item in sorted(annotations, key=lambda row: row.sort_key()):
        if item.level == "error":
            if errors < _ERROR_CAP:
                chosen.append(item)
                errors += 1
            else:
                omitted += 1
        elif item.level == "warning":
            if warnings < _WARNING_CAP:
                chosen.append(item)
                warnings += 1
            else:
                omitted += 1
        else:
            chosen.append(item)
    return chosen, omitted


def _render(annotations: list[_Annotation], counts: tuple[int, int, int, int]) -> str:
    chosen, omitted = _select(annotations)
    lines = [item.format_line() for item in chosen]
    summary = _Annotation(
        level="notice",
        kind="finding",
        severity="info",
        path="",
        line=None,
        end_line=None,
        check="summary",
        title="SkillEvaluator",
        message=_summary_message(counts, omitted),
    )
    lines.append(summary.format_line())
    # Each finding is one command line. A leftover newline would let untrusted
    # text start ::add-mask / ::stop-commands / ::set-output / ::add-path.
    sanitized: list[str] = []
    for line in lines:
        compact = line.replace("\r", "").replace("\n", "")
        if compact.startswith(("::add-mask", "::stop-commands", "::set-output", "::add-path")):
            continue
        sanitized.append(compact)
    return "\n".join(sanitized) + "\n"


def _write_stdout(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8", "backslashreplace"))
    sys.stdout.buffer.flush()


# A catalog validates each skill as its own run. While a collection is open,
# print_all records each skill's annotations here instead of printing them, so
# the catalog can print one capped stream and one summary for the whole job.
_collector: ContextVar[list[dict] | None] = ContextVar("github_annotation_collector", default=None)


@contextmanager
def collect_annotations() -> Iterator[list[dict]]:
    """Collect GitHub annotations from every render in this context instead of printing them.

    Each entry is a plain, picklable dict, so a catalog worker process can return
    its entries to the parent.
    """
    collected: list[dict] = []
    token = _collector.set(collected)
    try:
        yield collected
    finally:
        _collector.reset(token)


def print_collected_annotations(entries: list[dict]) -> None:
    """Print one capped annotation stream and one summary for collected per-skill entries."""
    annotations = [_Annotation(**item) for entry in entries for item in entry["annotations"]]
    totals = [0, 0, 0, 0]
    for entry in entries:
        for index, value in enumerate(entry["counts"]):
            totals[index] += value
    _write_stdout(_render(annotations, (totals[0], totals[1], totals[2], totals[3])))


class GitHubReporter(ReporterBase):
    """GitHub Actions workflow commands for inline pull-request annotations."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        scan_root: Path | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.scan_root = scan_root

    @property
    def name(self) -> str:
        return "github"

    @property
    def description(self) -> str:
        return "GitHub Actions annotations for pull-request diffs and the run summary"

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def annotations(self, results: list[ValidationResult]) -> list[_Annotation]:
        annotations: list[_Annotation] = []
        for result in results:
            annotations.extend(_annotations_for_result(result, self.workspace_root, self.scan_root))
        return annotations

    def render_all(self, results: list[ValidationResult]) -> str:
        return _render(self.annotations(results), _skill_counts(results))

    def print_all(self, results: list[ValidationResult]) -> None:
        collected = _collector.get()
        if collected is not None:
            collected.append(
                {
                    "annotations": [asdict(item) for item in self.annotations(results)],
                    "counts": list(_skill_counts(results)),
                }
            )
            return
        _write_stdout(self.render_all(results))

    def get_file_extension(self) -> str:
        return ""
