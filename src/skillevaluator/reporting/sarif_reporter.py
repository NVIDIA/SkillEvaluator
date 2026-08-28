# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SARIF reporter for CI security integrations.

Produces SARIF 2.1.0 output suitable for GitHub Code Scanning and other SARIF
consumers. Tier 1 findings with file locations are mapped to ``results``;
validators without structured findings are omitted.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from skillevaluator import __version__
from skillevaluator.reporting.base import ReporterBase

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult

_SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
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


def _physical_location(finding: Finding) -> dict[str, Any] | None:
    if not finding.file_path:
        return None
    location: dict[str, Any] = {
        "artifactLocation": {"uri": finding.file_path.replace("\\", "/")},
    }
    if finding.line_number is not None:
        region: dict[str, Any] = {"startLine": finding.line_number}
        if finding.line_content:
            region["snippet"] = {"text": finding.line_content}
        location["region"] = region
    return {"physicalLocation": location}


def _result_from_finding(finding: Finding, validator_name: str) -> dict[str, Any]:
    severity = _finding_severity_value(finding)
    result: dict[str, Any] = {
        "ruleId": _rule_id(validator_name, finding.check_name),
        "level": _severity_to_level(severity),
        "message": {"text": finding.message},
    }
    if finding.suggestion:
        result["message"]["markdown"] = f"{finding.message}\n\n**Suggestion:** {finding.suggestion}"
    location = _physical_location(finding)
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


class SARIFReporter(ReporterBase):
    """SARIF 2.1.0 export for security scanning integrations."""

    def __init__(self, *, indent: int | None = 2, include_timestamp: bool = True) -> None:
        self.indent = indent
        self.include_timestamp = include_timestamp

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

        for result in results:
            validator_name = result.validator_name or "UNKNOWN"
            for finding in result.findings:
                rule = _rule_descriptor(finding, validator_name)
                rules[rule["id"]] = rule
                sarif_results.append(_result_from_finding(finding, validator_name))

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
        }
        if self.include_timestamp:
            run["invocations"] = [
                {
                    "executionSuccessful": True,
                    "endTimeUtc": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                }
            ]

        document: dict[str, Any] = {
            "$schema": _SARIF_SCHEMA,
            "version": "2.1.0",
            "runs": [run],
        }
        return json.dumps(document, indent=self.indent, default=str, allow_nan=False)

    def get_file_extension(self) -> str:
        return ".sarif.json"
