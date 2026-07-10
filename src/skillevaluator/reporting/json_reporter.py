# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON reporter for CI/CD integration.

This reporter outputs machine-readable JSON suitable for:
- CI/CD pipeline parsing
- Integration with other tools
- Data analysis and aggregation
- API responses

The JSON schema is designed to be stable and backward-compatible.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from skillevaluator.reporting.base import ReporterBase, is_advisory_agent_eval_skip, passes_required_gate

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult


class JSONReporter(ReporterBase):
    """JSON export for machine-readable output.

    Produces structured JSON output suitable for programmatic consumption.
    Supports both compact and pretty-printed output formats.
    """

    def __init__(self, *, indent: int | None = 2, include_timestamp: bool = True) -> None:
        """Initialize JSON reporter.

        Args:
            indent: JSON indentation level (None for compact)
            include_timestamp: Whether to include generation timestamp
        """
        self.indent = indent
        self.include_timestamp = include_timestamp

    @property
    def name(self) -> str:
        return "json"

    @property
    def description(self) -> str:
        return "Machine-readable JSON for CI/CD integration"

    def render(self, result: ValidationResult) -> str:
        """Render single result to JSON."""
        data = self._result_to_dict(result)
        if self.include_timestamp:
            data["generated_at"] = datetime.now(tz=UTC).isoformat()
        return json.dumps(data, indent=self.indent, default=str, allow_nan=False)

    def render_all(self, results: list[ValidationResult]) -> str:
        """Render all results to JSON with overall summary."""
        from skillevaluator.reporting.html import HTMLReporter

        all_passed = all(passes_required_gate(r) for r in results)
        advisory_skip_count = sum(1 for r in results if is_advisory_agent_eval_skip(r))
        incomplete_scans = list(dict.fromkeys(tool for result in results for tool in result.incomplete_scans))
        overall_status = "incomplete" if incomplete_scans else "passed" if all_passed else "failed"
        total_errors = sum(r.summary.errors for r in results)
        total_warnings = sum(r.summary.warnings for r in results)

        # Reuse HTML reporter's skill reorganization and contributor extraction
        html_reporter = HTMLReporter()
        skills_by_name = html_reporter._reorganize_by_skill(results)
        contributors = HTMLReporter._extract_contributors(skills_by_name, results)

        # Build per-skill summary
        skills_summary = []
        for skill_name in sorted(skills_by_name):
            skill = skills_by_name[skill_name]
            issue_count = skill.get("issue_count", 0)
            skills_summary.append(
                {
                    "name": skill_name,
                    "passed": skill["passed"],
                    "issue_count": issue_count,
                }
            )

        data: dict[str, Any] = {
            "overall_passed": all_passed,
            "overall_status": overall_status,
            "incomplete_scans": incomplete_scans,
            "total_validators": len(results),
            "total_advisory_skipped": advisory_skip_count,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "severity_counts": {
                "critical": sum(r.summary.critical_count for r in results),
                "high": sum(r.summary.high_count for r in results),
                "medium": sum(r.summary.medium_count for r in results),
                "low": sum(r.summary.low_count for r in results),
            },
            "skills": skills_summary,
            "contributors": contributors,
            "results": [self._result_to_dict(r) for r in results],
        }

        # Quality summary from any QUALITY validator results
        quality_results = [r.metadata["quality_scores"] for r in results if r.metadata.get("quality_scores")]
        if quality_results:
            data["quality_summary"] = quality_results

        # Findings alone cannot distinguish a completed failing rubric from
        # an unavailable or malformed judge response. Persist the validator's
        # typed execution contract in the machine-readable report.
        rubric_results = [r.metadata["rubric_eval"] for r in results if r.metadata.get("rubric_eval")]
        if rubric_results:
            data["rubric_eval"] = rubric_results[0]

        # Tier 3: Agent evaluation summary
        tier3_results = [r.metadata["agent_eval"] for r in results if r.metadata.get("agent_eval")]
        if tier3_results:
            data["tier3"] = tier3_results[0]

        if self.include_timestamp:
            data["generated_at"] = datetime.now(tz=UTC).isoformat()

        return json.dumps(data, indent=self.indent, default=str, allow_nan=False)

    def _result_to_dict(self, result: ValidationResult) -> dict[str, Any]:
        """Convert ValidationResult to serializable dictionary."""
        data: dict[str, Any] = {
            "validator": result.validator_name,
            "description": result.validator_description,
            "passed": result.passed,
            "status": "skipped" if is_advisory_agent_eval_skip(result) else result.status,
            "incomplete_scans": result.incomplete_scans,
            "summary": {
                "files_scanned": result.summary.files_scanned,
                "checks_performed": result.summary.checks_performed,
                "errors": result.summary.errors,
                "warnings": result.summary.warnings,
                "critical_count": result.summary.critical_count,
                "high_count": result.summary.high_count,
                "medium_count": result.summary.medium_count,
                "low_count": result.summary.low_count,
            },
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity.value,
                    "check_name": f.check_name,
                    "message": f.message,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "line_content": f.line_content,
                    "suggestion": f.suggestion,
                    "metadata": f.metadata or None,
                }
                for f in result.findings
            ],
            "success_details": [
                {
                    "check": s.check_name,
                    "message": s.message,
                    "metadata": s.metadata or None,
                }
                for s in result.success_details
            ],
            # Include legacy fields for backward compatibility
            "legacy": {
                "errors": result.errors,
                "warnings": result.warnings,
                "messages": result.messages,
            },
        }

        # Quality scores (from QualityScoreValidator)
        qs = result.metadata.get("quality_scores")
        if qs:
            data["quality"] = qs

        # Tier 3: Agent evaluation data
        ae = result.metadata.get("agent_eval")
        if ae:
            data["tier3"] = ae

        rubric = result.metadata.get("rubric_eval")
        if rubric:
            data["rubric_eval"] = rubric

        return data

    def get_file_extension(self) -> str:
        return ".json"
