# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GitHub Actions workflow-command reporter.

Prints ``::error``, ``::warning``, and ``::notice`` lines for Tier 1 findings so a
workflow that runs the CLI directly can annotate the pull request without uploading
SARIF. Output goes to stdout only; this format is not a report file.

GitHub shows at most 10 error, 10 warning, and 10 notice annotations per step.
Finding notices stop at 9 so the closing summary notice is always shown, and
extra findings are counted on that summary rather than omitted silently.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Collection, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.constants import SKILL_MANIFEST_VARIANTS
from skillevaluator.reporting.base import ReporterBase, get_skip_reason, is_cleanly_skipped, passes_required_gate
from skillevaluator.reporting.run_status import Status, resolve_run_status, result_status
from skillevaluator.reporting.sarif_reporter import (
    _finding_severity_value,
    _positive_start_line,
    _resolve_artifact_path,
    _resolve_for_reporting,
    _workspace_relative_path,
)
from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.rich_markup import strip_terminal_controls

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult

_ERROR_CAP = 10
_WARNING_CAP = 10
# GitHub's notice cap is 10 per step; one slot is kept for the closing summary.
_NOTICE_CAP = 9
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


def _is_valid_utf8(text: str) -> bool:
    """Return whether *text* encodes cleanly as UTF-8 (no surrogate escapes)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _native_posix_path(file_path: str, scan_root: Path | None) -> Path | None:
    """Return *file_path* as given when its backslash is part of an existing POSIX file name.

    On POSIX a backslash is an ordinary file name character, so a tracked
    ``design\\notes.md`` must not be rewritten to ``design/notes.md``.
    """
    if os.sep != "/" or "\\" not in file_path:
        return None
    path = Path(file_path)
    if not path.is_absolute():
        if scan_root is None:
            return None
        candidate = scan_root / path
    else:
        candidate = path
    try:
        exists = os.path.lexists(candidate)
    except (OSError, ValueError):
        return None
    return path if exists else None


def _raw_github_file_path(
    file_path: object,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> str | None:
    """Return a repository-relative POSIX path, or None when the path is outside the repo."""
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    native = _native_posix_path(file_path, scan_root)
    if native is None:
        path = _resolve_artifact_path(file_path, scan_root)
    else:
        path = _resolve_for_reporting(native if native.is_absolute() or scan_root is None else scan_root / native)
    if workspace_root is not None and (path.is_absolute() or path.root):
        relative = _workspace_relative_path(path, file_path, workspace_root, scan_root)
        if relative is None:
            return None
        return relative.as_posix()
    if scan_root is not None:
        try:
            return path.resolve().relative_to(scan_root.resolve()).as_posix()
        except (OSError, ValueError):
            if path.is_absolute() or path.root:
                return None
    if path.is_absolute() or path.root:
        return None
    return file_path.replace("\\", "/")


def _github_file_path(
    file_path: object,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> str | None:
    """Return a valid UTF-8 repository-relative POSIX path, or None."""
    posix = _raw_github_file_path(file_path, workspace_root, scan_root)
    if posix is None or not _is_valid_utf8(posix):
        return None
    return posix


def _raw_link_github_file_path(
    file_path: Path,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> str | None:
    """Return the repository-relative path of *file_path* itself, or None.

    Only the parent directory is resolved, so a symlink keeps its own path
    even when its target is inside the repository.
    """
    if workspace_root is None or scan_root is None or ".." in file_path.parts:
        return None
    path = file_path if file_path.is_absolute() else scan_root / file_path
    try:
        relative = (path.parent.resolve() / path.name).relative_to(workspace_root.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return relative.as_posix()


def _link_github_file_path(
    file_path: Path,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> str | None:
    """Return a valid UTF-8 repository-relative path of *file_path* itself, or None."""
    posix = _raw_link_github_file_path(file_path, workspace_root, scan_root)
    if posix is None or not _is_valid_utf8(posix):
        return None
    return posix


def _same_path(first: Path, second: Path) -> bool:
    try:
        return first.resolve() == second.resolve()
    except (OSError, RuntimeError):
        return first == second


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return _same_path(path, root)
    return True


def _effective_workspace_root(
    workspace_root: Path | None,
    scan_root: Path | None,
    catalog_root: Path | None,
    *,
    catalog_workspace_root: Path | None = None,
) -> Path | None:
    """Return the workspace root that member annotations resolve against.

    In a catalog run, an explicit *catalog_workspace_root* from the parent takes
    precedence so nested Git roots inside the catalog still resolve against the
    outer checkout. Without it, a per-skill *workspace_root* that is absent,
    equal to *scan_root*, or nested inside *catalog_root* falls back to
    *catalog_root*.
    """
    if catalog_workspace_root is not None:
        return catalog_workspace_root
    if catalog_root is not None and (
        workspace_root is None
        or (scan_root is not None and _same_path(workspace_root, scan_root))
        or _path_is_within(workspace_root, catalog_root)
    ):
        return catalog_root
    return workspace_root


def _skill_manifest_path(
    workspace_root: Path | None,
    scan_root: Path | None,
) -> str | None:
    """Return the workspace-relative SKILL.md path for *scan_root*, or None."""
    if workspace_root is None or scan_root is None:
        return None
    if scan_root.is_symlink() or not scan_root.is_dir():
        if scan_root.name in SKILL_MANIFEST_VARIANTS:
            if scan_root.is_symlink():
                return _link_github_file_path(Path(scan_root.name), workspace_root, scan_root.parent)
            if scan_root.is_file():
                return _github_file_path(str(scan_root), workspace_root, scan_root.parent)
        return None
    try:
        entries = {entry.name for entry in scan_root.iterdir()}
    except OSError:
        return None
    for variant in SKILL_MANIFEST_VARIANTS:
        if variant not in entries:
            continue
        candidate = scan_root / variant
        if candidate.is_symlink():
            return _link_github_file_path(Path(variant), workspace_root, scan_root)
        if candidate.is_file():
            return _github_file_path(variant, workspace_root, scan_root)
    return None


def _relative_skill_path(scan_root: Path, root: Path) -> str | None:
    candidates: list[tuple[Path, Path]] = []
    if ".." not in scan_root.parts and ".." not in root.parts:
        candidates.append((scan_root, root))
    try:
        resolved_root = root.resolve()
    except (OSError, RuntimeError):
        resolved_root = None
    if resolved_root is not None:
        with suppress(OSError, RuntimeError):
            candidates.append((scan_root.parent.resolve() / scan_root.name, resolved_root))
        with suppress(OSError, RuntimeError):
            candidates.append((scan_root.resolve(), resolved_root))
    for target, base in candidates:
        try:
            relative = target.relative_to(base)
        except ValueError:
            continue
        text = relative.as_posix()
        if text not in {"", "."} and not text.startswith(".."):
            return _utf8_text(text)
    return None


def _catalog_skill_label(
    scan_root: Path | None,
    catalog_root: Path | None,
    workspace_root: Path | None,
) -> str:
    """Return *scan_root*'s catalog- or workspace-relative skill path, or an empty string."""
    if scan_root is None:
        return ""
    for root in (catalog_root, workspace_root):
        if root is None:
            continue
        relative = _relative_skill_path(scan_root, root)
        if relative is not None:
            return relative
    return ""


def _combine_labels(outer: str, inner: str) -> str:
    if not outer:
        return inner
    if not inner or inner == outer or inner.startswith(f"{outer}/"):
        return inner or outer
    return f"{outer}/{inner}"


def _raw_prefixed_artifact_github_path(
    artifact_path: object,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> str | None:
    """Resolve an unprefixed member artifact path when its repo location is unambiguous.

    A relative member path is relative to the child member rather than *scan_root*,
    so it cannot be placed on a repository file without guessing where the member lives.
    """
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return None
    raw = Path(artifact_path.replace("\\", "/"))
    if not (raw.is_absolute() or raw.root):
        return None
    return _raw_github_file_path(artifact_path, workspace_root, scan_root)


def _qualify_message(message: object, skill_label: str) -> str:
    cleaned = _clean_text(message)
    if not skill_label:
        return cleaned
    prefix = f"{_clean_text(skill_label, max_len=_TITLE_MAX)}: "
    if cleaned.startswith(prefix):
        return cleaned
    return _clean_text(f"{prefix}{cleaned}")


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
            self.title,
            self.message,
        )

    def format_line(self) -> str:
        # The Actions runner trims the whole property list before splitting it,
        # so the always-present title goes last: a file path ending in a space
        # (a tracked name may) then keeps that space.
        properties: list[str] = []
        if self.path:
            properties.append(f"file={annotation_text(self.path, property_value=True)}")
            if self.line is not None:
                properties.append(f"line={self.line}")
                if self.end_line is not None and self.end_line >= self.line:
                    properties.append(f"endLine={self.end_line}")
        properties.append(f"title={annotation_text(self.title, property_value=True)}")
        body = annotation_text(_clean_text(self.message))
        return f"::{self.level} {','.join(properties)}::{body}"


def _incomplete_diagnostic(
    result: ValidationResult,
    validator: str,
    error: str,
    *,
    default_path: str = "",
    skill_label: str = "",
) -> _Annotation:
    text = error.casefold()
    tool = next((tool for tool in result.incomplete_scans if tool.casefold() in text), None)
    check = f"incomplete/{tool}" if tool else "incomplete"
    return _Annotation(
        level="warning",
        kind="incomplete",
        severity="medium",
        path=default_path,
        line=None,
        end_line=None,
        check=check,
        title=_title(validator, check),
        message=_qualify_message(error, "" if default_path else skill_label),
    )


def _finding_annotation(
    finding: Finding,
    validator: str,
    workspace_root: Path | None,
    scan_root: Path | None,
    *,
    advisory: bool = False,
    default_path: str = "",
    skill_label: str = "",
    catalog_skill_label: str = "",
) -> _Annotation:
    severity = _finding_severity_value(finding)
    level = _level_for_severity(severity)
    if advisory and level == "error":
        level = "warning"
    metadata = finding.metadata if isinstance(finding.metadata, dict) else {}
    artifact_path = metadata.get("artifact_path")
    has_artifact_path = isinstance(artifact_path, str) and bool(artifact_path.strip())
    raw_member_label = metadata.get("member_label")
    member_label = raw_member_label.strip() if isinstance(raw_member_label, str) else ""
    has_file_path = isinstance(finding.file_path, str) and bool(finding.file_path.strip())

    fallback_to_anchor = False
    display_path = ""
    if has_artifact_path:
        resolved = _raw_prefixed_artifact_github_path(artifact_path, workspace_root, scan_root)
        if resolved:
            if _is_valid_utf8(resolved):
                path = resolved
                line = _positive_start_line(finding.line_number)
                end_line = _end_line(finding)
            else:
                path = ""
                line = None
                end_line = None
                display_path = resolved
        else:
            path = default_path
            line = None
            end_line = None
            fallback_to_anchor = True
    elif member_label or not has_file_path:
        path = default_path
        line = None
        end_line = None
        fallback_to_anchor = True
    else:
        raw_path = _raw_github_file_path(finding.file_path, workspace_root, scan_root) or ""
        if raw_path and _is_valid_utf8(raw_path):
            path = raw_path
            line = _positive_start_line(finding.line_number)
            end_line = _end_line(finding)
        elif raw_path:
            path = ""
            line = None
            end_line = None
            display_path = raw_path
        else:
            path = ""
            line = None
            end_line = None
            if catalog_skill_label:
                fallback_to_anchor = True

    suggestion = _clean_text(finding.suggestion) if finding.suggestion else ""
    message = _clean_text(finding.message)
    if suggestion:
        message = f"{message} Suggestion: {suggestion}"
    if display_path:
        message = _qualify_message(message, display_path)
    elif fallback_to_anchor:
        label = _combine_labels("" if path else skill_label, member_label)
        message = _qualify_message(message, label)
    return _Annotation(
        level=level,
        kind="finding",
        severity=severity,
        path=path,
        line=line,
        end_line=end_line,
        check=_clean_text(finding.check_name, max_len=_TITLE_MAX),
        title=_title(validator, finding.check_name),
        message=message,
    )


def _annotations_for_result(
    result: ValidationResult,
    workspace_root: Path | None,
    scan_root: Path | None,
    catalog_root: Path | None = None,
    *,
    catalog_workspace_root: Path | None = None,
) -> list[_Annotation]:
    validator = result.validator_name or "check"
    advisory = _advisory(result)
    annotations: list[_Annotation] = []

    if is_cleanly_skipped(result) and result_status(result) == Status.SKIPPED:
        return annotations

    effective_workspace = _effective_workspace_root(
        workspace_root,
        scan_root,
        catalog_root,
        catalog_workspace_root=catalog_workspace_root,
    )
    default_path = _skill_manifest_path(effective_workspace, scan_root) or ""
    catalog_label = (
        (_relative_skill_path(scan_root, catalog_root) or "")
        if scan_root is not None and catalog_root is not None
        else ""
    )
    skill_label = catalog_label or (
        "" if default_path else _catalog_skill_label(scan_root, catalog_root, effective_workspace)
    )
    unanchored_label = "" if default_path else skill_label

    for finding in result.findings:
        annotations.append(
            _finding_annotation(
                finding,
                validator,
                effective_workspace,
                scan_root,
                advisory=advisory,
                default_path=default_path,
                skill_label=skill_label,
                catalog_skill_label=catalog_label,
            )
        )

    gate_failures = set(result.gate_failures)
    # An incomplete scanner's reason is reported once, as its incomplete warning below.
    incomplete_reasons = {_incomplete_reason(result, tool) for tool in result.incomplete_scans}
    for error in result.independent_errors():
        is_gate = error in gate_failures
        if error in incomplete_reasons and not is_gate:
            continue
        if result.incomplete_scans and not is_gate:
            # Other diagnostics from an incomplete scan, such as a scanner's further
            # per-file errors, are warnings like the scan's own incomplete status.
            annotations.append(
                _incomplete_diagnostic(
                    result,
                    validator,
                    error,
                    default_path=default_path,
                    skill_label=unanchored_label,
                )
            )
            continue
        annotations.append(
            _Annotation(
                level="warning" if advisory else "error",
                kind="gate" if is_gate else "error",
                severity="high",
                path=default_path,
                line=None,
                end_line=None,
                check="gate_failure" if is_gate else "error",
                title=_title(validator, "gate_failure" if is_gate else "error"),
                message=_qualify_message(error, unanchored_label),
            )
        )

    for warning in result.independent_warnings():
        if warning in incomplete_reasons:
            continue
        annotations.append(
            _Annotation(
                level="warning",
                kind="finding",
                severity="medium",
                path=default_path,
                line=None,
                end_line=None,
                check="warning",
                title=_title(validator, "warning"),
                message=_qualify_message(warning, unanchored_label),
            )
        )

    for tool in result.incomplete_scans:
        annotations.append(
            _Annotation(
                level="warning",
                kind="incomplete",
                severity="medium",
                path=default_path,
                line=None,
                end_line=None,
                check=f"incomplete/{tool}",
                title=_title(validator, f"incomplete/{tool}"),
                message=_qualify_message(_incomplete_reason(result, tool), unanchored_label),
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
                path=default_path,
                line=None,
                end_line=None,
                check="gate_failure",
                title=_title(validator, "gate_failure"),
                message=_qualify_message(_failure_reason(result), unanchored_label),
            )
        )
    return annotations


def _skill_counts(results: list[ValidationResult]) -> tuple[int, int, int, int]:
    """Return (checked, passed, failed, incomplete) for this render.

    Counts follow the required validation gate, as the JSON ``overall_status``
    and the exit code do, so an advisory group's result cannot fail the skill.
    """
    if not results:
        return 0, 0, 0, 0
    headline = resolve_run_status(results).headline
    if headline == Status.FAILED:
        return 1, 0, 1, 0
    if headline == Status.INCOMPLETE:
        return 1, 0, 0, 1
    return 1, 1, 0, 0


def _summary_message(counts: tuple[int, int, int, int], omitted: int, report_formats: tuple[str, ...]) -> str:
    checked, passed, failed, incomplete = counts
    skill_word = "skill" if checked == 1 else "skills"
    text = f"{checked} {skill_word} checked, {passed} passed, {failed} failed, {incomplete} incomplete"
    if omitted:
        text += f". {omitted} more were not annotated (GitHub limits 10 errors, 10 warnings, and 10 notices per step)."
        file_reports = ", ".join(fmt for fmt in report_formats if fmt not in {"cli", "github"})
        if file_reports:
            text += f" See the requested file reports ({file_reports})."
        else:
            text += " Rerun with a file report such as -r github,json to see the omitted findings."
    return text


def _select(annotations: list[_Annotation]) -> tuple[list[_Annotation], int]:
    chosen: list[_Annotation] = []
    errors = 0
    warnings = 0
    notices = 0
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
        elif notices < _NOTICE_CAP:
            chosen.append(item)
            notices += 1
        else:
            omitted += 1
    return chosen, omitted


def _render(annotations: list[_Annotation], counts: tuple[int, int, int, int], report_formats: tuple[str, ...]) -> str:
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
        message=_summary_message(counts, omitted, report_formats),
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
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        sys.stdout.write(text)
        sys.stdout.flush()
    else:
        sys.stdout.flush()
        buffer.write(text.encode("utf-8", "backslashreplace"))
        buffer.flush()


# A catalog validates each skill as its own run. While a collection is open,
# print_all records each skill's annotations here instead of printing them, so
# the catalog can print one capped stream and one summary for the whole job.
_collector: ContextVar[list[dict] | None] = ContextVar("github_annotation_collector", default=None)
_collector_workspace_root: ContextVar[Path | None] = ContextVar(
    "github_annotation_collector_workspace_root", default=None
)


@contextmanager
def collect_annotations(*, workspace_root: Path | None = None) -> Iterator[list[dict]]:
    """Collect GitHub annotations from every render in this context instead of printing them.

    Each entry is a plain, picklable dict, so a catalog worker process can return
    its entries to the parent. When *workspace_root* is given, collected member
    annotations resolve paths against that shared catalog workspace root.
    """
    collected: list[dict] = []
    collector_token = _collector.set(collected)
    workspace_token = _collector_workspace_root.set(workspace_root)
    try:
        yield collected
    finally:
        _collector_workspace_root.reset(workspace_token)
        _collector.reset(collector_token)


def _collected_status(entries: list[dict]) -> str | None:
    """Return the status one skill's own collected renders reported, or None when it rendered nothing."""
    if not entries:
        return None
    totals = [sum(entry["counts"][index] for entry in entries) for index in range(4)]
    if totals[2]:
        return "failed"
    if totals[3]:
        return "incomplete"
    return "passed"


def catalog_annotation(
    *,
    check: str,
    message: str,
    severity: str,
    blocking: bool,
    file_path: Path | None = None,
    workspace_root: Path | None = None,
    scan_root: Path | None = None,
    follow_links: bool = True,
) -> dict:
    """Return a collected-annotation entry for a finding the catalog itself made.

    A catalog checks some policies, such as ``metadata.id`` uniqueness, across
    skills after each skill has rendered, so those findings exist in no skill's
    own stream. Blocking findings are errors; advisory ones follow severity but
    never error. With ``follow_links=False`` a symlinked *file_path* is
    annotated on the link, not on its target.
    """
    level = "error" if blocking else _level_for_severity(severity)
    if level == "error" and not blocking:
        level = "warning"
    if file_path is None:
        raw_path = None
    elif follow_links:
        raw_path = _raw_github_file_path(str(file_path), workspace_root, scan_root)
    else:
        raw_path = _raw_link_github_file_path(file_path, workspace_root, scan_root)
    if raw_path and _is_valid_utf8(raw_path):
        path = raw_path
        qualified = _clean_text(message)
    else:
        path = ""
        qualified = _qualify_message(message, raw_path or "")
    return asdict(
        _Annotation(
            level=level,
            kind="finding",
            severity=severity,
            path=path,
            line=None,
            end_line=None,
            check=_clean_text(check, max_len=_TITLE_MAX),
            title=_title("catalog", check),
            message=qualified,
        )
    )


def print_collected_annotations(
    skills: dict[str, list[dict]],
    *,
    failed: Collection[str] = (),
    catalog_failed: Collection[str] = (),
    catalog_annotations: list[dict] | None = None,
    report_formats: tuple[str, ...] = (),
) -> None:
    """Print one capped annotation stream and one summary for a whole catalog.

    *skills* maps every catalog skill to the entries its own renders collected.
    The summary counts follow the catalog's final outcomes, not the per-skill
    renders: *failed* names every skill the catalog failed and *catalog_failed*
    the ones failed by a catalog-level check, which no skill render saw. A skill
    the catalog failed only because its own scan did not complete stays
    incomplete. *catalog_annotations* join the per-skill ones before the cap.
    """
    annotations = [
        _Annotation(**item) for entries in skills.values() for entry in entries for item in entry["annotations"]
    ]
    annotations.extend(_Annotation(**item) for item in catalog_annotations or [])
    passed = failures = incomplete = 0
    for name, entries in skills.items():
        own = _collected_status(entries)
        if name in catalog_failed:
            failures += 1
        elif own == "incomplete":
            incomplete += 1
        elif name in failed:
            failures += 1
        else:
            passed += 1
    _write_stdout(_render(annotations, (len(skills), passed, failures, incomplete), report_formats))


class GitHubReporter(ReporterBase):
    """GitHub Actions workflow commands for inline pull-request annotations."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        scan_root: Path | None = None,
        catalog_root: Path | None = None,
        catalog_workspace_root: Path | None = None,
        report_formats: tuple[str, ...] = (),
    ) -> None:
        self.workspace_root = workspace_root
        self.scan_root = scan_root
        self.catalog_root = catalog_root
        self.catalog_workspace_root = catalog_workspace_root
        self.report_formats = report_formats

    @property
    def name(self) -> str:
        return "github"

    @property
    def description(self) -> str:
        return "GitHub Actions annotations for pull-request diffs and the run summary"

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def annotations(self, results: list[ValidationResult]) -> list[_Annotation]:
        shared_workspace = (
            self.catalog_workspace_root if self.catalog_workspace_root is not None else _collector_workspace_root.get()
        )
        annotations: list[_Annotation] = []
        for result in results:
            annotations.extend(
                _annotations_for_result(
                    result,
                    self.workspace_root,
                    self.scan_root,
                    self.catalog_root,
                    catalog_workspace_root=shared_workspace,
                )
            )
        return annotations

    def render_all(self, results: list[ValidationResult]) -> str:
        return _render(self.annotations(results), _skill_counts(results), self.report_formats)

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
