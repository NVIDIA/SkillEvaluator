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

from skillevaluator.publication_evidence import result_publication_evidence_dict
from skillevaluator.reporting.base import (
    ReporterBase,
    assess_publication,
    get_skip_reason,
    is_advisory_agent_eval_skip,
    is_cleanly_skipped,
    passes_required_gate,
    result_publication_target_conflict_marker,
    result_publication_target_dict,
    select_agent_eval_payload,
)

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult


def _json_safe_agent_eval(value: object) -> dict[str, Any]:
    """Bound and normalize Tier 3 metadata before machine-readable embedding."""
    from skillevaluator.reporting.html import _json_safe_tier3_payload

    return _json_safe_tier3_payload(value)


def _json_safe_benchmark_policy(value: object) -> dict[str, bool] | None:
    """Project untrusted per-result policy metadata onto its two typed keys."""
    if not isinstance(value, dict):
        return None
    policy = {
        key: candidate
        for key in ("tier2_required", "tier3_required")
        if isinstance((candidate := value.get(key)), bool)
    }
    return policy or None


class JSONReporter(ReporterBase):
    """JSON export for machine-readable output.

    Produces structured JSON output suitable for programmatic consumption.
    Supports both compact and pretty-printed output formats.
    """

    def __init__(
        self,
        *,
        indent: int | None = 2,
        include_timestamp: bool = True,
        expected_skill_name: str | None = None,
    ) -> None:
        """Initialize JSON reporter.

        Args:
            indent: JSON indentation level (None for compact)
            include_timestamp: Whether to include generation timestamp
        """
        self.indent = indent
        self.include_timestamp = include_timestamp
        self.expected_skill_name = expected_skill_name

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
        skip_count = sum(1 for r in results if is_cleanly_skipped(r))
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
            "total_skipped": skip_count,
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

        agent_eval = select_agent_eval_payload(results)
        emitted_agent_eval = _json_safe_agent_eval(agent_eval) if agent_eval else None
        # Publication claims must be supported by both the raw evidence and the
        # normalized evidence actually emitted to consumers. Normalization must
        # neither invent proof from malformed keys/values nor hide proof through
        # truncation, so retain the more conservative assessment.
        raw_publication = assess_publication(
            results,
            agent_eval,
            expected_skill_name=self.expected_skill_name,
        )
        emitted_publication = assess_publication(
            results,
            emitted_agent_eval,
            expected_skill_name=self.expected_skill_name,
        )
        status_rank = {"pass": 0, "neutral": 1, "incomplete": 2, "fail": 3}
        publication = (
            emitted_publication
            if status_rank[emitted_publication.status] > status_rank[raw_publication.status]
            else raw_publication
        )
        data["benchmark_policy"] = publication.benchmark_policy
        data["publication_status"] = publication.status
        data["publication"] = {
            "status": publication.status,
            "eligible": publication.status == "pass",
            "reasons": list(publication.reasons),
            "tier3": {
                "status": publication.tier3.status,
                "evidence_complete": publication.tier3.evidence_complete,
                "execution_status": publication.tier3.execution_status,
                "verdict": publication.tier3.verdict,
                "reason": publication.tier3.reason,
            },
        }

        policy = next(
            (result.metadata.get("policy") for result in results if isinstance(result.metadata.get("policy"), dict)),
            None,
        )
        if policy is not None:
            data["policy"] = policy

        gating_by_tier: dict[str, dict[str, Any]] = {}
        for result in results:
            gating = result.metadata.get("gating")
            if not isinstance(gating, dict) or gating.get("tier") is None:
                continue
            tier = str(gating["tier"])
            tier_entry = gating_by_tier.setdefault(tier, {"blocking": False, "validators": []})
            tier_entry["blocking"] = bool(tier_entry["blocking"] or gating.get("blocking"))
            tier_entry["validators"].append(result.validator_name)
        if gating_by_tier:
            data["gating"] = {"tiers": gating_by_tier}

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
        if emitted_agent_eval:
            data["tier3"] = emitted_agent_eval
            applicability = next(
                (
                    result.metadata.get("tier3_applicability")
                    for result in results
                    if isinstance(result.metadata.get("tier3_applicability"), dict)
                ),
                None,
            )
            if applicability is not None:
                data["tier3_applicability"] = applicability

        if self.include_timestamp:
            data["generated_at"] = datetime.now(tz=UTC).isoformat()

        return json.dumps(data, indent=self.indent, default=str, allow_nan=False)

    def _result_to_dict(self, result: ValidationResult) -> dict[str, Any]:
        """Convert ValidationResult to serializable dictionary."""
        skipped = is_cleanly_skipped(result)
        data: dict[str, Any] = {
            "validator": result.validator_name,
            "description": result.validator_description,
            "passed": result.passed,
            "status": "skipped" if skipped else result.status,
            "skipped": skipped,
            "skip_reason": get_skip_reason(result) if skipped else None,
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
            data["tier3"] = _json_safe_agent_eval(ae)

        rubric = result.metadata.get("rubric_eval")
        if rubric:
            data["rubric_eval"] = rubric

        gating = result.metadata.get("gating")
        if isinstance(gating, dict):
            data["gating"] = gating

        benchmark_policy = _json_safe_benchmark_policy(result.metadata.get("benchmark_policy"))
        if benchmark_policy is not None:
            data["benchmark_policy"] = benchmark_policy

        publication_target = result_publication_target_dict(result)
        if publication_target is not None:
            data["publication_target"] = publication_target
        publication_target_conflict = result_publication_target_conflict_marker(result)
        if publication_target_conflict is not None:
            data["publication_target_conflict"] = publication_target_conflict
        publication_evidence = result_publication_evidence_dict(result)
        if publication_evidence is not None:
            data["publication_evidence"] = publication_evidence

        return data

    def get_file_extension(self) -> str:
        return ".json"
