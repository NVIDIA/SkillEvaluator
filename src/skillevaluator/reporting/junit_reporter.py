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
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.reporting.base import ReporterBase, get_skip_reason, is_cleanly_skipped
from skillevaluator.reporting.run_status import finding_blocks
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


def _set_counts(element: ET.Element, *, tests: int, failures: int, skipped: int) -> None:
    element.set("tests", str(tests))
    element.set("failures", str(failures))
    element.set("errors", "0")
    element.set("skipped", str(skipped))
    element.set("time", "0")


@dataclass(slots=True, frozen=True)
class _Case:
    name: str
    classname: str
    failed: bool = False
    skipped: bool = False
    message: str = ""
    body: str = ""
    file_path: str = ""
    line: int | None = None
    system_out: str = ""


def _cases_for_result(
    result: ValidationResult,
    workspace_root: Path | None,
    scan_root: Path | None,
    target: str = "",
) -> list[_Case]:
    validator = _xml_text(result.validator_name) or "check"
    classname = _qualified(target, validator)
    used: set[str] = set()
    if is_cleanly_skipped(result):
        return [
            _Case(
                name=_unique_name(validator, used),
                classname=classname,
                skipped=True,
                message=_xml_text(get_skip_reason(result)),
                body=_xml_text(get_skip_reason(result)),
            )
        ]

    incomplete = bool(result.incomplete_scans)
    cases: list[_Case] = []
    for finding in result.findings:
        file_path = _junit_file_path(finding.file_path, workspace_root, scan_root)
        line = _positive_start_line(finding.line_number)
        failed = finding_blocks(result, finding)
        skipped = incomplete and not failed
        name = _unique_name(_xml_text(finding.check_name) or "finding", used)
        detail = _failure_body(finding, file_path)
        cases.append(
            _Case(
                name=name,
                classname=classname,
                failed=failed,
                skipped=skipped,
                message=_xml_text(finding.message),
                body=detail if failed or skipped else "",
                file_path=file_path,
                line=line,
                system_out="" if failed or skipped else detail,
            )
        )
    for success in result.success_details:
        if incomplete:
            cases.append(
                _Case(
                    name=_unique_name(_xml_text(success.check_name) or "check", used),
                    classname=classname,
                    skipped=True,
                    message=_xml_text(success.message),
                    body=_xml_text(success.message),
                )
            )
        else:
            cases.append(
                _Case(
                    name=_unique_name(_xml_text(success.check_name) or "check", used),
                    classname=classname,
                    system_out=_xml_text(success.message),
                )
            )
    gate_failures = set(result.gate_failures)
    for error in result.independent_errors():
        base = "gate_failure" if error in gate_failures else "error"
        cases.append(
            _Case(
                name=_unique_name(base, used),
                classname=classname,
                failed=True,
                message=_xml_text(error),
                body=_xml_text(error),
            )
        )
    for tool in result.incomplete_scans:
        reason = f"{tool} scan did not complete"
        cases.append(
            _Case(
                name=_unique_name(_xml_text(f"incomplete/{tool}"), used),
                classname=classname,
                skipped=True,
                message=_xml_text(reason),
                body=_xml_text(reason),
            )
        )
    if not cases:
        if result.passed:
            cases.append(_Case(name=validator, classname=classname, system_out="passed"))
        else:
            message = _xml_text(result.errors[0] if result.errors else "failed")
            cases.append(
                _Case(name=validator, classname=classname, failed=True, message=message, body=message)
            )
    elif not result.passed and not any(case.failed or case.skipped for case in cases):
        message = _xml_text(result.errors[0] if result.errors else "failed")
        cases.append(
            _Case(
                name=_unique_name("error", used),
                classname=classname,
                failed=True,
                message=message,
                body=message,
            )
        )
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
        total_skipped = 0
        for result in results:
            cases = _cases_for_result(result, self.workspace_root, self.scan_root, self.target_identity)
            suite = ET.SubElement(testsuites, "testsuite")
            suite.set("name", _qualified(self.target_identity, _xml_text(result.validator_name) or "check"))
            failures = sum(1 for case in cases if case.failed)
            skipped = sum(1 for case in cases if case.skipped)
            _set_counts(suite, tests=len(cases), failures=failures, skipped=skipped)
            for case in cases:
                _append_case(suite, case)
            system_out_text = _suite_system_out(result)
            if system_out_text:
                system_out = ET.SubElement(suite, "system-out")
                system_out.text = system_out_text
            total_tests += len(cases)
            total_failures += failures
            total_skipped += skipped
        _set_counts(testsuites, tests=total_tests, failures=total_failures, skipped=total_skipped)
        ET.indent(testsuites, space="  ")
        body = ET.tostring(testsuites, encoding="unicode")
        if not body.endswith("\n"):
            body += "\n"
        return _XML_DECLARATION + body

    def get_file_extension(self) -> str:
        return ".junit.xml"
