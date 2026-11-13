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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator import __version__
from skillevaluator.publication_evidence import (
    TIER1_REQUIRED_CHECKS_KEY,
    TIER1_UNSCANNED_PATHS_KEY,
    result_publication_evidence_dict,
)
from skillevaluator.reporting.base import (
    ReporterBase,
    assess_publication,
    get_skip_reason,
    is_advisory_agent_eval_skip,
    is_cleanly_skipped,
    result_publication_target_conflict_marker,
    result_publication_target_dict,
    select_agent_eval_payload,
)
from skillevaluator.reporting.finding_summary import (
    finding_counts,
    merge_finding_summaries,
    summarize_findings,
    summarize_messages,
)
from skillevaluator.reporting.run_status import Status, resolve_run_status, result_status
from skillevaluator.skillspector_diagnostics import sanitize_completeness
from skillevaluator.skillspector_provenance import sanitize_skillspector_provenance_metadata
from skillevaluator.skillspector_reference_caveats import sanitize_reference_caveat_metadata
from skillevaluator.skillspector_references import reference_summary_record
from skillevaluator.source_identity import recorded_evaluated_source

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult


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


_MAX_SEMGREP_UNPARSED_FILES = 1000


def _tier1_scope_strings(value: object, field: str) -> list[str]:
    """Copy bounded scope evidence without narrowing its publication meaning.

    Reject malformed declarations rather than dropping them: absence has legacy
    semantics and could otherwise turn incomplete evidence into a passing run.
    Semantic lineup validation remains the publication assessor's responsibility.
    """
    if (
        not isinstance(value, list)
        or len(value) > 20_000
        or any(not isinstance(item, str) or not 1 <= len(item) <= 4096 for item in value)
    ):
        raise ValueError(f"Invalid {field}: expected a bounded list of nonempty strings")
    return list(value)


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
        target_path: str | None = None,
        explicit_skill_name: str | None = None,
    ) -> None:
        """Initialize JSON reporter.

        Args:
            indent: JSON indentation level (None for compact)
            include_timestamp: Whether to include generation timestamp
            expected_skill_name: Publication target name the evidence must bind to
            target_path: Display path or Git remote URL of the validated target
            explicit_skill_name: Name of the one skill a single-skill run
                validated. It names the ``skills`` entry, which is otherwise
                inferred from finding paths and quality metadata and so falls
                back to one entry per check ID for a clean run without them.
        """
        self.indent = indent
        self.include_timestamp = include_timestamp
        self.expected_skill_name = expected_skill_name
        self.target_path = target_path
        self.explicit_skill_name = explicit_skill_name

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

        run_status = resolve_run_status(results)
        all_passed = run_status.headline == Status.PASSED
        skip_count = sum(1 for r in results if is_cleanly_skipped(r))
        advisory_skip_count = sum(1 for r in results if is_advisory_agent_eval_skip(r))
        incomplete_scans = list(dict.fromkeys(tool for result in results for tool in result.incomplete_scans))
        overall_status = run_status.headline.value
        total_errors = sum(r.summary.errors for r in results)
        total_warnings = sum(r.summary.warnings for r in results)
        result_finding_summaries = [summarize_findings([result]) for result in results]
        finding_summary = merge_finding_summaries(result_finding_summaries)

        # Reuse HTML reporter's skill reorganization and contributor extraction.
        # The explicit name keeps a single-skill run to one ``skills`` entry
        # when no finding path or quality score names the skill, and the
        # target path is the same fallback the HTML report uses.
        identity_target = self.target_path
        if identity_target and "://" not in identity_target:
            local_target = Path(identity_target)
            if (local_target / "SKILL.md").exists() or (local_target / "skill.md").exists():
                # Preserve the requested target in JSON while resolving aliases
                # such as "." for the shared single-skill identity fallback.
                identity_target = str(local_target.resolve())
        html_reporter = HTMLReporter(
            target_path=identity_target,
            explicit_skill_name=self.explicit_skill_name,
            expected_skill_name=self.expected_skill_name,
        )
        skills_by_name = html_reporter._reorganize_by_skill(results)
        contributors = HTMLReporter._extract_contributors(skills_by_name, results)
        # Top issues across every tier, the rows the HTML report data carries
        # (its summary panel lists the Tier 1 rows only), without ``issue_key``:
        # that is an HTML anchor hashed from the finding text, and CI redaction
        # masks the text in this report but cannot refresh a hash it never sees.
        top_issues = [
            {key: value for key, value in issue.items() if key != "issue_key"}
            for issue in html_reporter._compute_top_issues(skills_by_name)
        ]

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
            "version": __version__,
            "target_path": self.target_path,
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
            # Raw totals above count every validator record. These reconcile
            # them into unique findings and distinct locations by stable ID.
            "finding_counts": finding_counts(finding_summary),
            "finding_groups": finding_summary["groups"],
            "message_counts": summarize_messages(results),
            "skills": skills_summary,
            "contributors": contributors,
            "top_issues": top_issues,
            "results": [
                self._result_to_dict(result, finding_summary=result_summary)
                for result, result_summary in zip(results, result_finding_summaries, strict=True)
            ],
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

        quality_config = next(
            (
                result.metadata.get("quality_config")
                for result in results
                if isinstance(result.metadata.get("quality_config"), dict)
            ),
            None,
        )
        if quality_config is not None:
            data["quality_config"] = quality_config

        # BENCHMARK.md records which source tree was evaluated, so the report a
        # CI job actually parses has to record it too: a consumer cannot verify
        # a card it never reads. The key is always present, so ``null`` says the
        # orchestration input supplied no identity rather than leaving a caller
        # to guess whether an older reporter simply omitted it. Every carrier is
        # folded, so a run recording two source trees raises rather than
        # publishing one of them. The CLI resolves the identity before any
        # report is written, so this fails closed only on programmatic misuse.
        data["evaluated_source"] = recorded_evaluated_source(r.metadata for r in results)

        gating_by_tier: dict[str, dict[str, Any]] = {}
        for result in results:
            gating = result.metadata.get("gating")
            if not isinstance(gating, dict) or gating.get("tier") is None:
                continue
            tier = str(gating["tier"])
            tier_entry = gating_by_tier.setdefault(
                tier,
                {"blocking": False, "validators": [], "validator_policies": []},
            )
            tier_entry["blocking"] = bool(tier_entry["blocking"] or gating.get("blocking"))
            tier_entry["validators"].append(result.validator_name)
            # ``blocking`` above is true when any validator of the tier blocks;
            # this records each validator's own policy, so an advisory Tier 2
            # check is not read as blocking because another one is.
            tier_entry["validator_policies"].append(
                {"validator": result.validator_name, "blocking": bool(gating.get("blocking"))}
            )
        if gating_by_tier:
            data["gating"] = {"tiers": gating_by_tier}

        workflow = next((r.metadata["workflow"] for r in results if r.metadata.get("workflow")), None)
        if workflow is not None:
            data["workflow"] = workflow

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

        # Tier 1 applicability: ``--type auto`` found no supported artifact, so
        # the manifest checks were recorded as not applicable. Every result of
        # such a run carries the record, whatever --checks selected, so CI can
        # read this instead of parsing skip reasons.
        tier1_applicability = next(
            (
                result.metadata.get("tier1_applicability")
                for result in results
                if isinstance(result.metadata.get("tier1_applicability"), dict)
            ),
            None,
        )
        if tier1_applicability is not None:
            data["tier1_applicability"] = tier1_applicability

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

    def _result_to_dict(
        self,
        result: ValidationResult,
        *,
        finding_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert ValidationResult to serializable dictionary."""
        if finding_summary is None:
            finding_summary = summarize_findings([result])
        skipped = is_cleanly_skipped(result)
        data: dict[str, Any] = {
            "validator": result.validator_name,
            "description": result.validator_description,
            "passed": result.passed,
            "status": result_status(result).value,
            "skipped": skipped,
            "skip_reason": get_skip_reason(result) if skipped else None,
            "incomplete_scans": result.incomplete_scans,
            **({"scanners": result.scanners} if result.scanners else {}),
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
            "finding_counts": finding_counts(finding_summary),
            "findings": [self._finding_to_dict(finding) for finding in result.findings],
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

        if result.metadata.get("confirmed_failure") is True:
            data["confirmed_failure"] = True

        for key in (TIER1_REQUIRED_CHECKS_KEY, TIER1_UNSCANNED_PATHS_KEY):
            if key in result.metadata:
                data[key] = _tier1_scope_strings(result.metadata[key], key)

        skillspector_version = result.metadata.get("skillspector_version")
        if isinstance(skillspector_version, str):
            data["skillspector_version"] = skillspector_version
        # Tier 1 SkillSpector LLM provenance (``--llm``), revalidated here so
        # only the canonical record reaches the report.
        data.update(sanitize_skillspector_provenance_metadata(result.metadata))
        completeness = sanitize_completeness(result.metadata.get("skillspector_completeness"))
        if completeness:
            data["skillspector_completeness"] = completeness

        if result.gate_failures:
            data["gate_failures"] = result.gate_failures

        # Quality scores (from QualityScoreValidator)
        coverage = result.metadata.get("dependency_coverage")
        if isinstance(coverage, list):
            data["dependency_coverage"] = coverage

        # Files Semgrep could not fully parse. The error message names at most 20;
        # this keeps up to _MAX_SEMGREP_UNPARSED_FILES and counts the rest, so a
        # report that was cut short says so instead of looking complete.
        unparsed_files = result.metadata.get("semgrep_unparsed_files")
        if isinstance(unparsed_files, list):
            paths = [item for item in unparsed_files if isinstance(item, str) and 0 < len(item) <= 4096]
            if paths:
                data["semgrep_unparsed_files"] = paths[:_MAX_SEMGREP_UNPARSED_FILES]
                # A report restored from JSON already carries the count of files it left out.
                carried = result.metadata.get("semgrep_unparsed_files_omitted")
                omitted = max(0, len(paths) - _MAX_SEMGREP_UNPARSED_FILES)
                if type(carried) is int and carried > 0:
                    omitted += carried
                if omitted:
                    data["semgrep_unparsed_files_omitted"] = omitted

        # Tier 1 SkillSpector file-role scoping: retained and suppressed issue
        # counts, and the raw versus applicable aggregate risk score.
        skillspector_applicability = result.metadata.get("skillspector_applicability")
        if isinstance(skillspector_applicability, dict):
            data["skillspector_applicability"] = skillspector_applicability

        # Tier 1 SkillSpector AE1 counts: referenced artifacts it did not
        # completely inspect and the reference lines that name them.
        reference_summary = reference_summary_record(result.metadata.get("skillspector_reference_summary"))
        if reference_summary is not None:
            data["skillspector_reference_summary"] = reference_summary

        qs = result.metadata.get("quality_scores")
        if qs:
            data["quality"] = qs

        # Whether an explicitly requested SkillSpector baseline was applied, so
        # CI can tell an inert baseline from one that matched nothing.
        skillspector_baseline = result.metadata.get("skillspector_baseline")
        if isinstance(skillspector_baseline, dict):
            data["skillspector_baseline"] = dict(skillspector_baseline)

        reference_caveats = sanitize_reference_caveat_metadata(result.metadata.get("skillspector_reference_caveats"))
        if reference_caveats is not None:
            data["skillspector_reference_caveats"] = reference_caveats

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

        workflow = result.metadata.get("workflow")
        if isinstance(workflow, dict):
            data["workflow"] = workflow

        # Tier 2 records bounded provider diagnostics and completed/failed
        # cluster counts independently of duplicate-content findings.
        llm_analysis = result.metadata.get("llm_analysis")
        if isinstance(llm_analysis, dict):
            data["llm_analysis"] = llm_analysis
        # A Tier 2 work limit makes the scan incomplete; record which limit stopped it.
        work_limit = result.metadata.get("work_limit")
        if isinstance(work_limit, dict):
            data["work_limit"] = work_limit

        return data

    @staticmethod
    def _finding_to_dict(finding: Finding) -> dict[str, Any]:
        """Preserve the legacy JSON null for findings without metadata."""
        data = finding.to_dict()
        data["metadata"] = data["metadata"] or None
        return data

    def get_file_extension(self) -> str:
        return ".json"
