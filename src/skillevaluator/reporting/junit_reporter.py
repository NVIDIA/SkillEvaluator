# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JUnit XML reporter for CI test-report consumers.

Produces a JUnit XML document with one ``testsuite`` per check group
(``ValidationResult``) and one ``testcase`` per check, finding, gate failure, or
incomplete scanner. CI systems that ingest JUnit reports (GitLab merge-request
test reports, Jenkins, Azure Pipelines, CircleCI, Buildkite) read this format.

File paths are stored as repository- or scan-root-relative POSIX paths using the
same resolution rules as the SARIF reporter, percent-encoded as SARIF does so a
file name that is not valid UTF-8 stays distinct (``caf\\udce9.md`` becomes
``caf%E9.md``). In other text, lone surrogates are written as backslash escapes,
then characters XML 1.0 forbids are stripped before serialization.

When the reporter knows the evaluated target, suite names and testcase
classnames are qualified with its repository-relative path (for example
``group-a/demo::Schema & Repository Governance``), so reports from several
targets collected into one CI test report keep distinct test identities.
A finding case is named after its check and location (``check (path:line)``), so
its identity does not shift when other findings are added or reordered.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.reporting.base import (
    ReporterBase,
    get_skip_reason,
    is_cleanly_skipped,
    passes_required_gate,
)
from skillevaluator.reporting.run_status import Status, finding_blocks, result_status
from skillevaluator.reporting.sarif_reporter import (
    _positive_start_line,
    _quote_uri_path,
    _resolve_artifact_path,
    _workspace_relative_path,
)

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult

_XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>\n'


def _xml_char_allowed(character: str) -> bool:
    """Return whether *character* is legal in XML 1.0 text."""
    code = ord(character)
    if code in {0x9, 0xA, 0xD}:
        return True
    if 0x20 <= code <= 0xD7FF:
        return True
    if 0xE000 <= code <= 0xFFFD:
        return True
    return 0x10000 <= code <= 0x10FFFF


def _escape_surrogates(text: str) -> str:
    """Return *text* with lone surrogates written as backslash escapes.

    A file name that is not valid UTF-8 arrives with surrogate escapes, which
    XML cannot hold; escaping them keeps ``caf\\udce9.md`` distinct from
    ``caf.md`` (it becomes ``caf\\xe9.md``).
    """
    try:
        return text.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")
    except UnicodeEncodeError:
        return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _xml_text(value: object) -> str:
    """Return *value* as XML 1.0 text, dropping characters the spec forbids."""
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    text = _escape_surrogates(text).replace("\r\n", "\n").replace("\r", "\n")
    return "".join(character for character in text if _xml_char_allowed(character))


def _junit_file_path(
    file_path: object,
    workspace_root: Path | None,
    scan_root: Path | None,
) -> str:
    """Return a percent-encoded POSIX path relative to the scanned workspace, matching SARIF."""
    if not isinstance(file_path, str) or not file_path.strip():
        return ""
    raw = file_path.strip()
    path = _resolve_artifact_path(raw, scan_root)
    if workspace_root is not None and (path.is_absolute() or path.root):
        relative = _workspace_relative_path(path, raw, workspace_root, scan_root)
        if relative is not None:
            return _quote_uri_path(relative.as_posix())
    if scan_root is not None:
        try:
            return _quote_uri_path(path.resolve().relative_to(scan_root.resolve()).as_posix())
        except ValueError:
            pass
    if path.is_absolute() or path.root:
        return _quote_uri_path(path.as_posix())
    return _quote_uri_path(raw.replace("\\", "/"))


def _relative_posix(path: Path, root: Path | None) -> str | None:
    """Return *path* relative to *root* as POSIX text, or None when outside it."""
    if root is None:
        return None
    try:
        relative = path.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    text = relative.as_posix()
    return None if text in {"", "."} else text


def junit_target_identity(
    scan_root: Path | None,
    repository_root: Path | None = None,
    catalog_root: Path | None = None,
) -> str:
    """Return a stable identity for the evaluated target.

    The target's path relative to the repository root comes first, then its path
    relative to the catalog folder it was found in, then its own name. The same
    leaf name under two catalog groups (``group-a/demo`` and ``group-b/demo``)
    therefore yields two identities.
    """
    if scan_root is None:
        return ""
    try:
        target = scan_root.resolve()
    except OSError:
        target = scan_root
    for root in (repository_root, catalog_root):
        relative = _relative_posix(target, root)
        if relative is not None:
            return _xml_text(relative)
    return _xml_text(target.name)


def _qualified(target: str, name: str) -> str:
    return f"{target}::{name}" if target else name


def _failure_body(finding: Finding, file_path: str) -> str:
    """Return failure text with file:line when the finding records a location."""
    line = _positive_start_line(finding.line_number)
    if file_path and line is not None:
        location = f"{file_path}:{line}"
    elif file_path:
        location = file_path
    else:
        location = ""
    message = _xml_text(finding.message)
    body = f"{location}: {message}" if location else message
    suggestion = _xml_text(finding.suggestion) if finding.suggestion else ""
    if suggestion:
        return f"{body}\nSuggestion: {suggestion}"
    return body


def _finding_case_name(check_name: str, file_path: str, line: int | None) -> str:
    """Return a finding case name keyed on its location, not its position in the list.

    ``quality_correctness (SKILL.md:12)`` stays the same when other findings are
    added or reordered, so CI test history keyed on suite, class and name holds.
    """
    stem = check_name or "finding"
    if not file_path:
        return stem
    location = f"{file_path}:{line}" if line is not None else file_path
    return f"{stem} ({location})"


def _unique_name(base: str, used: set[str]) -> str:
    stem = base or "check"
    name = stem
    index = 2
    while name in used:
        name = f"{stem}-{index}"
        index += 1
    used.add(name)
    return name


def _suite_system_out(result: ValidationResult) -> str:
    """Return score and threshold text when the result records them."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    lines: list[str] = []
    quality = metadata.get("quality_scores")
    if isinstance(quality, dict) and quality.get("overall_score") is not None:
        score = quality["overall_score"]
        grade = quality.get("grade")
        if grade:
            lines.append(f"Score: {score}/100 (Grade: {grade})")
        else:
            lines.append(f"Score: {score}/100")
    config = metadata.get("quality_config")
    if isinstance(config, dict) and config.get("min_score") is not None:
        lines.append(f"Threshold: {config['min_score']}/100")
    rubric = metadata.get("rubric_eval")
    if isinstance(rubric, dict) and rubric.get("overall_score") is not None:
        lines.append(f"Rubric score: {rubric['overall_score']}/100")
        if rubric.get("min_score") is not None:
            lines.append(f"Rubric threshold: {rubric['min_score']}/100")
    return _xml_text("\n".join(lines))


def _set_counts(element: ET.Element, *, tests: int, failures: int, errors: int, skipped: int) -> None:
    element.set("tests", str(tests))
    element.set("failures", str(failures))
    element.set("errors", str(errors))
    element.set("skipped", str(skipped))
    element.set("time", "0")


@dataclass(slots=True, frozen=True)
class _Case:
    name: str
    classname: str
    failed: bool = False
    errored: bool = False
    error_type: str = ""
    skipped: bool = False
    message: str = ""
    body: str = ""
    file_path: str = ""
    line: int | None = None
    system_out: str = ""


def _gating(result: ValidationResult) -> dict | None:
    """Return the gate ``validate`` stamped on *result*, or None when it has none."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    gating = metadata.get("gating")
    return gating if isinstance(gating, dict) else None


def _failure_reason(result: ValidationResult) -> str:
    """Return the real cause of a failed result for a synthetic failure case."""
    if result.errors:
        return _xml_text(result.errors[0])
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    payload = metadata.get("agent_eval")
    if isinstance(payload, dict) and payload.get("execution_status") == "succeeded" and payload.get("verdict"):
        return _xml_text(f"Tier 3 verdict {str(payload['verdict']).upper()} does not pass the required gate")
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if metadata.get("skip_reason") or result.warnings or (isinstance(provenance, dict) and provenance.get("message")):
        return _xml_text(get_skip_reason(result))
    return "failed"


def _incomplete_reason(result: ValidationResult, tool: str) -> str:
    """Return the first warning or error that names *tool*, else a generic incomplete-scan reason."""
    needle = tool.casefold()
    for text in (*result.warnings, *result.errors):
        if isinstance(text, str) and needle in text.casefold():
            return _xml_text(text)
    return _xml_text(f"{tool} scan did not complete")


def _cases_for_result(
    result: ValidationResult,
    workspace_root: Path | None,
    scan_root: Path | None,
    target: str = "",
) -> list[_Case]:
    validator = _xml_text(result.validator_name) or "check"
    classname = _qualified(target, validator)
    used: set[str] = set()
    # A skip that run status reports as failed (a blocking gate that did not run)
    # falls through to the gate_failure case below rather than a <skipped> case.
    if is_cleanly_skipped(result) and result_status(result) == Status.SKIPPED:
        return [
            _Case(
                name=_unique_name(validator, used),
                classname=classname,
                skipped=True,
                message=_xml_text(get_skip_reason(result)),
                body=_xml_text(get_skip_reason(result)),
            )
        ]

    cases: list[_Case] = []
    for finding in result.findings:
        file_path = _junit_file_path(finding.file_path, workspace_root, scan_root)
        line = _positive_start_line(finding.line_number)
        failed = finding_blocks(result, finding)
        name = _unique_name(_finding_case_name(_xml_text(finding.check_name), file_path, line), used)
        detail = _failure_body(finding, file_path)
        cases.append(
            _Case(
                name=name,
                classname=classname,
                failed=failed,
                message=_xml_text(finding.message),
                body=detail if failed else "",
                file_path=file_path,
                line=line,
                system_out="" if failed else detail,
            )
        )
    for success in result.success_details:
        cases.append(
            _Case(
                name=_unique_name(_xml_text(success.check_name) or "check", used),
                classname=classname,
                system_out=_xml_text(success.message),
            )
        )
    gate_failures = set(result.gate_failures)
    # A group with an incomplete scan recorded that a scanner could not run or crashed,
    # so its independent errors are execution diagnostics: JUnit <error>, not <failure>.
    # Gate failures (a minimum score, for example) stay <failure> either way.
    scan_incomplete = bool(result.incomplete_scans)
    for error in result.independent_errors():
        is_gate = error in gate_failures
        execution = scan_incomplete and not is_gate
        cases.append(
            _Case(
                name=_unique_name("gate_failure" if is_gate else "error", used),
                classname=classname,
                failed=not execution,
                errored=execution,
                error_type="execution" if execution else "",
                message=_xml_text(error),
                body=_xml_text(error),
            )
        )
    for tool in result.incomplete_scans:
        # An incomplete scan fails the run, so it is an <error>, never a <skipped> case.
        reason = _incomplete_reason(result, tool)
        cases.append(
            _Case(
                name=_unique_name(_xml_text(f"incomplete/{tool}"), used),
                classname=classname,
                errored=True,
                error_type="incomplete",
                message=reason,
                body=reason,
            )
        )
    gating = _gating(result)
    if gating is not None and not gating.get("blocking", True):
        # An advisory result never fails the exit gate, so it fails or errors no case here:
        # incomplete scanners are skipped, and failures and execution errors move to system-out.
        cases = [
            _Case(
                name=case.name,
                classname=case.classname,
                skipped=case.skipped or case.error_type == "incomplete",
                message=case.message if case.skipped or case.error_type == "incomplete" else "",
                body=case.body if case.skipped or case.error_type == "incomplete" else "",
                file_path=case.file_path,
                line=case.line,
                system_out=(
                    f"Advisory, does not gate: {case.body}"
                    if case.failed or case.error_type == "execution"
                    else case.system_out
                ),
            )
            for case in cases
        ]
    needs_failure = not passes_required_gate(result) and not any(case.failed or case.errored for case in cases)
    if needs_failure:
        message = _failure_reason(result)
        cases.append(
            _Case(
                name=_unique_name("gate_failure" if gating is not None else "error", used),
                classname=classname,
                failed=True,
                message=message,
                body=message,
            )
        )
    elif not cases:
        note = _failure_reason(result) if not result.passed else "passed"
        cases.append(_Case(name=validator, classname=classname, system_out=note))
    return cases


def _append_case(suite: ET.Element, case: _Case) -> None:
    testcase = ET.SubElement(suite, "testcase")
    testcase.set("name", case.name)
    testcase.set("classname", case.classname)
    testcase.set("time", "0")
    if case.file_path:
        testcase.set("file", case.file_path)
    if case.line is not None:
        testcase.set("line", str(case.line))
    if case.skipped:
        skipped = ET.SubElement(testcase, "skipped")
        if case.message:
            skipped.set("message", case.message)
        if case.body:
            skipped.text = case.body
    elif case.errored:
        error = ET.SubElement(testcase, "error")
        if case.message:
            error.set("message", case.message)
        error.set("type", case.error_type or "error")
        if case.body:
            error.text = case.body
    elif case.failed:
        failure = ET.SubElement(testcase, "failure")
        if case.message:
            failure.set("message", case.message)
        failure.set("type", "failure")
        if case.body:
            failure.text = case.body
    if case.system_out:
        system_out = ET.SubElement(testcase, "system-out")
        system_out.text = case.system_out


class JUnitReporter(ReporterBase):
    """JUnit XML export for CI test-report integrations."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        scan_root: Path | None = None,
        target_identity: str | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.scan_root = scan_root
        self.target_identity = _xml_text(target_identity) if target_identity else ""

    @property
    def name(self) -> str:
        return "junit"

    @property
    def description(self) -> str:
        return "JUnit XML for GitLab, Jenkins, and other CI test-report consumers"

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def render_all(self, results: list[ValidationResult]) -> str:
        testsuites = ET.Element("testsuites")
        testsuites.set("name", "SkillEvaluator")
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        for result in results:
            cases = _cases_for_result(result, self.workspace_root, self.scan_root, self.target_identity)
            suite = ET.SubElement(testsuites, "testsuite")
            suite.set("name", _qualified(self.target_identity, _xml_text(result.validator_name) or "check"))
            failures = sum(1 for case in cases if case.failed)
            errors = sum(1 for case in cases if case.errored)
            skipped = sum(1 for case in cases if case.skipped)
            _set_counts(suite, tests=len(cases), failures=failures, errors=errors, skipped=skipped)
            for case in cases:
                _append_case(suite, case)
            system_out_text = _suite_system_out(result)
            if system_out_text:
                system_out = ET.SubElement(suite, "system-out")
                system_out.text = system_out_text
            total_tests += len(cases)
            total_failures += failures
            total_errors += errors
            total_skipped += skipped
        _set_counts(testsuites, tests=total_tests, failures=total_failures, errors=total_errors, skipped=total_skipped)
        ET.indent(testsuites, space="  ")
        body = ET.tostring(testsuites, encoding="unicode")
        if not body.endswith("\n"):
            body += "\n"
        return _XML_DECLARATION + body

    def get_file_extension(self) -> str:
        return ".junit.xml"
