# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SARIF reporter for CI security integrations.

Produces SARIF 2.1.0 output suitable for GitHub Code Scanning and other SARIF
consumers. Tier 1 findings with file locations are mapped to ``results``;
validators without structured findings are omitted.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from skillevaluator import __version__
from skillevaluator.reporting.base import ReporterBase
from skillevaluator.reporting.run_status import resolve_run_status

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult

_SARIF_SCHEMA = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
_TOOL_URI = "https://github.com/NVIDIA/SkillEvaluator"
_RULE_ID_PATTERN = re.compile(r"[^A-Za-z0-9._/-]+")


def _sanitize_rule_component(value: str) -> str:
    """Return a SARIF-safe rule identifier fragment."""
    cleaned = _RULE_ID_PATTERN.sub("-", value.strip())
    return cleaned.strip("-") or "finding"


def _rule_id(validator_name: str, check_name: str) -> str:
    validator = _sanitize_rule_component(validator_name)
    check = _sanitize_rule_component(check_name)
    if check and check != validator:
        return f"{validator}/{check}"
    return validator


def _severity_to_level(severity: str) -> str:
    if severity in {"critical", "high"}:
        return "error"
    if severity == "medium":
        return "warning"
    return "note"


def _finding_severity_value(finding: Finding) -> str:
    severity = finding.severity
    if hasattr(severity, "value"):
        return str(severity.value).lower()
    return str(severity).lower()


def _rule_descriptor(finding: Finding, validator_name: str) -> dict[str, Any]:
    rule_id = _rule_id(validator_name, finding.check_name)
    severity = _finding_severity_value(finding)
    descriptor: dict[str, Any] = {
        "id": rule_id,
        "name": finding.check_name,
        "shortDescription": {"text": finding.check_name},
        "defaultConfiguration": {"level": _severity_to_level(severity)},
    }
    if finding.message:
        descriptor["fullDescription"] = {"text": finding.message}
    return descriptor


def _positive_start_line(line_number: Any) -> int | None:
    """Return a SARIF-valid positive integer line number, else ``None``."""
    if isinstance(line_number, bool):
        return None
    if isinstance(line_number, str):
        stripped = line_number.strip()
        if not stripped:
            return None
        try:
            line_number = int(stripped)
        except ValueError:
            return None
    if isinstance(line_number, int) and line_number > 0:
        return line_number
    return None


def _resolve_artifact_path(file_path: str, scan_root: Path | None) -> Path:
    """Resolve a validator file path against the scanned skill directory."""
    normalized = file_path.replace("\\", "/")
    path = Path(normalized)
    if scan_root is not None and not path.is_absolute():
        return (scan_root / path).resolve()
    if path.is_absolute():
        return path.resolve()
    return path


def _quote_uri_path(text: str, safe: str = "/:@") -> str:
    """Percent-encode a POSIX-style path for use in a URI.

    A file name that is not valid UTF-8 arrives with surrogate escapes; its
    original bytes are encoded (``caf\\udce9.md`` becomes ``caf%E9.md``).
    """
    try:
        octets = text.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        octets = text.encode("utf-8", "surrogatepass")
    return quote(octets, safe=safe)


def _relative_reference(normalized: str) -> str:
    """Return a relative URI reference whose first segment cannot read as a scheme."""
    uri = _quote_uri_path(normalized)
    # RFC 3986 section 4.2: a colon in the first segment would make
    # ``notes:v2.md`` parse as scheme ``notes``; a dot segment keeps it a path.
    if ":" in uri.split("/", 1)[0]:
        return f"./{uri}"
    return uri


def _file_uri(path: Path) -> str:
    """Return an absolute ``file:`` URI for a path outside the workspace."""
    posix = path.as_posix()
    if posix.startswith("//"):  # UNC: //server/share/...
        host, _, rest = posix[2:].partition("/")
        return f"file://{_quote_uri_path(host, safe='')}/{_quote_uri_path(rest)}"
    if posix.startswith("/"):
        return f"file://{_quote_uri_path(posix)}"
    return f"file:///{_quote_uri_path(posix)}"  # drive letter: C:/...


def _unresolved_artifact_path(file_path: str, scan_root: Path | None) -> Path | None:
    """Return the absolute path a finding names, without following its symlinks.

    ``None`` when the path has a ``..`` segment: after a symlinked directory
    that names a different file than the lexically normalized path would.
    """
    path = Path(file_path.replace("\\", "/"))
    if ".." in path.parts:
        return None
    if not path.is_absolute():
        if scan_root is None:
            return None
        path = scan_root.resolve() / path
    return Path(os.path.normpath(path))


def _workspace_relative_path(
    path: Path,
    file_path: str,
    workspace_root: Path,
    scan_root: Path | None,
) -> Path | None:
    """Return the artifact path relative to the workspace root, or ``None``.

    The resolved path is tried first. A symlink inside the workspace whose
    target is outside it keeps the path of the link, which is what the
    finding is about.
    """
    root = workspace_root.resolve()
    candidates = [path.resolve()]
    unresolved = _unresolved_artifact_path(file_path, scan_root)
    if unresolved is not None:
        candidates.append(unresolved)
    for candidate in candidates:
        try:
            return candidate.relative_to(root)
        except ValueError:
            continue
    return None


def _normalize_artifact_uri(
    file_path: str,
    workspace_root: Path | None,
    scan_root: Path | None = None,
) -> str:
    """Return a URI-encoded SARIF artifact location.

    Paths inside the workspace become repository-relative references; other
    absolute paths become ``file:`` URIs.
    """
    path = _resolve_artifact_path(file_path, scan_root)
    # Windows paths can be rooted (``\\workspace\\...``) without a drive, in
    # which case ``is_absolute()`` is false until resolved.
    if workspace_root is not None and (path.is_absolute() or path.root):
        relative = _workspace_relative_path(path, file_path, workspace_root, scan_root)
        if relative is not None:
            return _relative_reference(relative.as_posix())
        return _file_uri(path.resolve())
    if path.is_absolute():
        return _file_uri(path)
    return _relative_reference(path.as_posix())


def _physical_location(
    finding: Finding,
    workspace_root: Path | None,
    scan_root: Path | None = None,
) -> dict[str, Any] | None:
    if not finding.file_path:
        return None
    location: dict[str, Any] = {
        "artifactLocation": {
            "uri": _normalize_artifact_uri(finding.file_path, workspace_root, scan_root),
        },
    }
    start_line = _positive_start_line(finding.line_number)
    if start_line is not None:
        region: dict[str, Any] = {"startLine": start_line}
        if finding.line_content:
            region["snippet"] = {"text": finding.line_content}
        location["region"] = region
    return {"physicalLocation": location}


def _result_from_finding(
    finding: Finding,
    validator_name: str,
    workspace_root: Path | None,
    scan_root: Path | None = None,
) -> dict[str, Any]:
    severity = _finding_severity_value(finding)
    result: dict[str, Any] = {
        "ruleId": _rule_id(validator_name, finding.check_name),
        "level": _severity_to_level(severity),
        "message": {"text": finding.message},
    }
    if finding.suggestion:
        result["message"]["markdown"] = f"{finding.message}\n\n**Suggestion:** {finding.suggestion}"
    location = _physical_location(finding, workspace_root, scan_root)
    if location is not None:
        result["locations"] = [location]
    properties: dict[str, Any] = {
        "category": finding.category,
        "validator": validator_name,
        "checkName": finding.check_name,
        "severity": severity,
    }
    if finding.metadata:
        properties["metadata"] = finding.metadata
    result["properties"] = properties
    return result


def _collect_incomplete_scans(results: list[ValidationResult]) -> list[str]:
    scans: list[str] = []
    for result in results:
        for tool in result.incomplete_scans:
            if tool not in scans:
                scans.append(tool)
    return scans


def _build_invocation(results: list[ValidationResult]) -> dict[str, Any]:
    incomplete_scans = _collect_incomplete_scans(results)
    invocation: dict[str, Any] = {
        "executionSuccessful": not incomplete_scans,
        "endTimeUtc": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    if incomplete_scans:
        invocation["toolExecutionNotifications"] = [
            {
                "descriptor": {"id": f"incomplete/{tool}"},
                "level": "error",
                "message": {"text": f"{tool} scan did not complete"},
            }
            for tool in incomplete_scans
        ]
    return invocation


def merge_catalog_sarif_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-skill SARIF payloads into one upload-ready document.

    GitHub Code Scanning rejects SARIF with more than 20 runs, so child reports
    are folded into a single run with combined rules and results.
    """
    rules: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    invocations: list[dict[str, Any]] = []
    driver: dict[str, Any] = {
        "name": "SkillEvaluator",
        "informationUri": _TOOL_URI,
    }

    for payload in documents:
        for run in payload.get("runs", []):
            run_driver = run.get("tool", {}).get("driver", {})
            if run_driver.get("name"):
                driver["name"] = run_driver["name"]
            if run_driver.get("informationUri"):
                driver["informationUri"] = run_driver["informationUri"]
            if run_driver.get("version"):
                driver["version"] = run_driver["version"]
            for rule in run_driver.get("rules", []):
                rules[rule["id"]] = rule
            results.extend(run.get("results", []))
            invocations.extend(run.get("invocations", []))

    merged_run: dict[str, Any] = {
        "tool": {
            "driver": {
                **driver,
                "rules": sorted(rules.values(), key=lambda item: item["id"]),
            }
        },
        "results": results,
        "automationDetails": {"id": "/skillevaluator/catalog"},
    }
    if invocations:
        merged_run["invocations"] = invocations
    return {
        "$schema": _SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [merged_run],
    }


class SARIFReporter(ReporterBase):
    """SARIF 2.1.0 export for security scanning integrations."""

    def __init__(
        self,
        *,
        indent: int | None = 2,
        include_timestamp: bool = True,
        workspace_root: Path | None = None,
        scan_root: Path | None = None,
    ) -> None:
        self.indent = indent
        self.include_timestamp = include_timestamp
        self.workspace_root = workspace_root
        self.scan_root = scan_root

    @property
    def name(self) -> str:
        return "sarif"

    @property
    def description(self) -> str:
        return "SARIF 2.1.0 for GitHub Code Scanning and SARIF consumers"

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def render_all(self, results: list[ValidationResult]) -> str:
        rules: dict[str, dict[str, Any]] = {}
        sarif_results: list[dict[str, Any]] = []
        workspace_root = self.workspace_root
        scan_root = self.scan_root

        for result in results:
            validator_name = result.validator_name or "UNKNOWN"
            for finding in result.findings:
                rule = _rule_descriptor(finding, validator_name)
                rules[rule["id"]] = rule
                sarif_results.append(_result_from_finding(finding, validator_name, workspace_root, scan_root))

        run: dict[str, Any] = {
            "tool": {
                "driver": {
                    "name": "SkillEvaluator",
                    "informationUri": _TOOL_URI,
                    "version": __version__,
                    "rules": sorted(rules.values(), key=lambda item: item["id"]),
                }
            },
            "results": sarif_results,
            "properties": {"status": resolve_run_status(results).headline.value},
        }
        if self.include_timestamp or _collect_incomplete_scans(results):
            run["invocations"] = [_build_invocation(results)]

        document: dict[str, Any] = {
            "$schema": _SARIF_SCHEMA,
            "version": "2.1.0",
            "runs": [run],
        }
        return json.dumps(document, indent=self.indent, default=str, allow_nan=False)

    def get_file_extension(self) -> str:
        return ".sarif.json"
