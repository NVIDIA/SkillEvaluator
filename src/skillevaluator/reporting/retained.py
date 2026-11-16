# SPDX-License-Identifier: Apache-2.0
"""Render retained public JSON with trusted templates, without reading skill paths.

The caller must bound the incoming JSON byte size. This adapter additionally
bounds its shape and rejects malformed typed evidence instead of inventing it.
"""

from __future__ import annotations

import copy
import math
from dataclasses import fields
from functools import partial
from typing import Any

from skillevaluator.models import Finding, SuccessDetail, ValidationResult, ValidationSummary
from skillevaluator.publication_evidence import TIER1_REQUIRED_CHECKS_KEY, TIER1_UNSCANNED_PATHS_KEY
from skillevaluator.reporting.base import is_cleanly_skipped, is_tier2_result, is_tier3_result, passes_required_gate
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.reporting.json_reporter import _tier1_scope_strings
from skillevaluator.reporting.run_status import resolve_run_status, result_status
from skillevaluator.skillspector_diagnostics import sanitize_completeness
from skillevaluator.skillspector_provenance import sanitize_skillspector_provenance_metadata
from skillevaluator.skillspector_reference_caveats import sanitize_reference_caveat_metadata
from skillevaluator.skillspector_references import REFERENCE_SUMMARY_KEY, reference_summary_record


class RetainedReportError(ValueError):
    """The retained document cannot be rendered faithfully."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RetainedReportError(message)


def _json(value: Any, depth: int = 0) -> None:
    _require(depth <= 24, "Metadata is too deeply nested")
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), "Non-finite metadata number")
    elif isinstance(value, str):
        _require(len(value) <= 1_000_000, "Metadata string too large")
    elif isinstance(value, list):
        _require(len(value) <= 20_000, "Metadata list too large")
        for item in value:
            _json(item, depth + 1)
    elif isinstance(value, dict):
        _require(len(value) <= 20_000, "Metadata object too large")
        for key, item in value.items():
            _require(isinstance(key, str), "Metadata keys must be strings")
            _json(item, depth + 1)
    else:
        raise RetainedReportError("Metadata must contain JSON values only")


def _text(value: Any, field: str, *, optional: bool = False) -> Any:
    _require(isinstance(value, str) or (optional and value is None), f"Invalid {field}")
    return value


def _mapping(value: Any, field: str, *, copy_value: bool = True) -> dict:
    _require(isinstance(value, dict), f"Invalid {field}: expected object")
    return copy.deepcopy(value) if copy_value else value


def _strings(value: Any, field: str) -> list[str]:
    _require(isinstance(value, list) and all(isinstance(v, str) for v in value), f"Invalid {field}")
    return list(value)


def _typed_metadata(metadata: dict) -> None:
    """Validate nested containers dereferenced by native report templates."""
    workflow = metadata.get("workflow")
    if workflow is not None:
        stages = workflow.get("stages", {})
        _require(isinstance(stages, dict), "Invalid workflow.stages")
        for stage in stages.values():
            _require(isinstance(stage, dict) and isinstance(stage.get("status"), str), "Invalid workflow stage")
            if "reason" in stage:
                _text(stage["reason"], "workflow reason")
    quality = metadata.get("quality_scores")
    if quality is not None:
        for key in ("skill_name", "grade"):
            _text(quality.get(key), f"quality.{key}")
        score = quality.get("overall_score")
        _require(type(score) in (int, float) and 0 <= score <= 100, "Invalid quality overall_score")
        for key in ("dimensions", "metrics"):
            _require(isinstance(quality.get(key), dict), f"Invalid quality.{key}")
        for dimension in quality["dimensions"].values():
            _require(isinstance(dimension, dict), "Invalid quality dimension")
            for key in ("score", "weight"):
                _require(type(dimension.get(key)) in (int, float), f"Invalid dimension.{key}")
            _require(
                0 <= dimension["score"] <= 100 and 0 <= dimension["weight"] <= 1, "Invalid dimension score or weight"
            )
            _require(
                type(dimension.get("issues_count")) is int and dimension["issues_count"] >= 0,
                "Invalid dimension issues_count",
            )
            _require(
                isinstance(dimension.get("issues"), list)
                and all(isinstance(issue, dict) for issue in dimension["issues"]),
                "Invalid dimension issues",
            )
    policy = metadata.get("policy")
    if policy is not None:
        for key in ("profile", "audience", "digest", "source"):
            if key in policy:
                _text(policy[key], f"policy.{key}")
        if "severity_overrides" in policy:
            _require(
                isinstance(policy["severity_overrides"], dict)
                and all(isinstance(value, str) for value in policy["severity_overrides"].values()),
                "Invalid severity overrides",
            )
    agent_eval = metadata.get("agent_eval")
    if agent_eval is not None:
        for key in (
            "summary",
            "agents",
            "dataset_summary",
            "attempt_policy",
            "evaluators",
            "insights",
            "metric_labels",
            "verdict_policy",
            "provenance",
        ):
            if key in agent_eval:
                _require(isinstance(agent_eval[key], dict), f"Invalid tier3.{key}")
        for key in (
            "dimensions",
            "dataset",
            "trials",
            "cases",
            "evaluator_cards",
            "timing",
            "agents_run",
            "execution_errors",
            "suggestions",
            "suggestions_v2",
            "metric_ids",
        ):
            if key in agent_eval:
                _require(isinstance(agent_eval[key], list), f"Invalid tier3.{key}")
        for key in ("verdict", "execution_status", "skill_name"):
            if key in agent_eval:
                _text(agent_eval[key], f"tier3.{key}")
        for agent in agent_eval.get("agents", {}).values():
            _require(isinstance(agent, dict), "Invalid tier3 agent")
            if "conditions" in agent:
                _require(isinstance(agent["conditions"], dict), "Invalid tier3 agent.conditions")
            for key in ("with_skill", "without_skill"):
                if agent.get(key) is not None:
                    _require(type(agent[key]) in (int, float) and 0 <= agent[key] <= 1, f"Invalid tier3 agent.{key}")
        for record in (agent_eval, agent_eval.get("summary", {})):
            for key in ("expected_attempts", "scored_attempts"):
                if key in record:
                    _require(type(record[key]) is int and record[key] >= 0, f"Invalid tier3.{key}")
            for key in ("overall_score", "overall_lift", "composite_lift"):
                if record.get(key) is not None:
                    minimum = 0 if key == "overall_score" else -1
                    _require(type(record[key]) in (int, float) and minimum <= record[key] <= 1, f"Invalid tier3.{key}")
    benchmark = metadata.get("benchmark_policy")
    if benchmark is not None:
        _require(all(type(value) is bool for value in benchmark.values()), "Invalid benchmark policy")


def retained_results(
    payload: dict, *, copy_metadata: bool = True, allowed_tiers: set[int] | None = None
) -> list[ValidationResult]:
    """Rehydrate the JSONReporter's public contract using explicit field maps.

    Results own deep copies of the payload's metadata by default. A caller that
    only validates and discards the results can pass copy_metadata=False; the
    results then share the payload's mappings, which neither side modifies.
    """
    _require(isinstance(payload, dict), "Report must be an object")
    _json(payload)
    mapping = partial(_mapping, copy_value=copy_metadata)
    rows = payload.get("results")
    _require(isinstance(rows, list) and 0 < len(rows) <= 1000, "Report requires nonempty results")
    results = []
    seen = set()
    mappings = {
        "quality": "quality_scores",
        "workflow": "workflow",
        "policy": "policy",
        "rubric_eval": "rubric_eval",
        "gating": "gating",
        "benchmark_policy": "benchmark_policy",
        "publication_target": "publication_target",
        "publication_evidence": "publication_evidence",
        "llm_analysis": "llm_analysis",
        "work_limit": "work_limit",
        "tier3": "agent_eval",
        "skillspector_applicability": "skillspector_applicability",
        "skillspector_baseline": "skillspector_baseline",
    }
    for row in rows:
        _require(isinstance(row, dict), "Result must be an object")
        name = _text(row.get("validator"), "validator")
        _require(bool(name) and name not in seen, "Validator names must be nonempty and unique")
        seen.add(name)
        _require(type(row.get("passed")) is bool, "Result passed must be boolean")
        summary = mapping(row.get("summary"), "summary")
        for field in fields(ValidationSummary):
            _require(type(summary.get(field.name)) is int and summary[field.name] >= 0, f"Invalid summary.{field.name}")
        _require(set(summary) == {f.name for f in fields(ValidationSummary)}, "Unexpected summary fields")
        findings = []
        _require(isinstance(row.get("findings"), list), "findings must be an array")
        for item in row["findings"]:
            _require(isinstance(item, dict), "Finding must be an object")
            severity = item.get("severity")
            _require(severity in ("critical", "high", "medium", "low", "info"), "Invalid finding severity")
            line = item.get("line_number")
            _require(line is None or (type(line) is int and line > 0), "Invalid finding line_number")
            findings.append(
                Finding(
                    **{
                        key: _text(item.get(key), f"finding.{key}")
                        for key in ("category", "check_name", "message", "file_path")
                    },
                    severity=severity,
                    line_number=line,
                    line_content=_text(item.get("line_content"), "line_content", optional=True),
                    suggestion=_text(item.get("suggestion"), "suggestion", optional=True),
                    metadata=mapping(
                        item.get("metadata") if item.get("metadata") is not None else {}, "finding.metadata"
                    ),
                )
            )
        details = []
        _require(isinstance(row.get("success_details"), list), "success_details must be an array")
        for item in row["success_details"]:
            _require(isinstance(item, dict), "Success detail must be an object")
            details.append(
                SuccessDetail(
                    check_name=_text(item.get("check"), "success.check"),
                    message=_text(item.get("message"), "success.message"),
                    metadata=mapping(
                        item.get("metadata") if item.get("metadata") is not None else {}, "success.metadata"
                    ),
                )
            )
        legacy = mapping(row.get("legacy"), "legacy")
        metadata = {}
        for source, dest in mappings.items():
            if source in row:
                metadata[dest] = mapping(row[source], source)
        if "skillspector_version" in row:
            metadata["skillspector_version"] = _text(row["skillspector_version"], "skillspector_version")
        # Revalidated like every report boundary; a non-canonical record is left out.
        metadata.update(sanitize_skillspector_provenance_metadata(row))
        completeness = sanitize_completeness(row.get("skillspector_completeness"))
        if completeness:
            metadata["skillspector_completeness"] = completeness
        if "skillspector_reference_caveats" in row:
            reference_caveats = sanitize_reference_caveat_metadata(row["skillspector_reference_caveats"])
            _require(reference_caveats is not None, "Invalid skillspector_reference_caveats")
            metadata["skillspector_reference_caveats"] = reference_caveats
        if "publication_target_conflict" in row:
            metadata["publication_target_conflict"] = _text(
                row["publication_target_conflict"], "publication_target_conflict"
            )
        for key in ("policy", "quality_config"):
            if key not in metadata and payload.get(key) is not None:
                metadata[key] = mapping(payload[key], key)
        # Retain source identity under the exact carrier consumed by reporters.
        if payload.get("evaluated_source") is not None:
            metadata["evaluated_source"] = mapping(payload["evaluated_source"], "evaluated_source")
        for key in ("incomplete_scans", "gate_failures", "semgrep_unparsed_files"):
            if key in row:
                metadata[key] = _strings(row[key], key)
        if "semgrep_unparsed_files_omitted" in row:
            omitted = row["semgrep_unparsed_files_omitted"]
            _require(type(omitted) is int and omitted >= 0, "Invalid semgrep_unparsed_files_omitted")
            metadata["semgrep_unparsed_files_omitted"] = omitted
        for key in (TIER1_REQUIRED_CHECKS_KEY, TIER1_UNSCANNED_PATHS_KEY):
            if key in row:
                try:
                    metadata[key] = _tier1_scope_strings(row[key], key)
                except ValueError as exc:
                    raise RetainedReportError(str(exc)) from exc
        if "scanners" in row:
            _require(
                isinstance(row["scanners"], dict)
                and all(isinstance(k, str) and isinstance(v, str) for k, v in row["scanners"].items()),
                "scanners must map scanner names to versions",
            )
            metadata["scanners"] = dict(row["scanners"])
        # Only the valid counts are kept; a failed result's reports show them.
        reference_summary = reference_summary_record(row.get(REFERENCE_SUMMARY_KEY))
        if reference_summary is not None:
            metadata[REFERENCE_SUMMARY_KEY] = reference_summary
        if "dependency_coverage" in row:
            _require(
                isinstance(row["dependency_coverage"], list)
                and all(isinstance(x, dict) for x in row["dependency_coverage"]),
                "Invalid dependency_coverage",
            )
            coverage = row["dependency_coverage"]
            metadata["dependency_coverage"] = copy.deepcopy(coverage) if copy_metadata else coverage
        if "skipped" in row:
            _require(type(row["skipped"]) is bool, "skipped must be boolean")
            metadata["skipped"] = row["skipped"]
        if "optional_skip" in row:
            _require(type(row["optional_skip"]) is bool, "optional_skip must be boolean")
            metadata["optional_skip"] = row["optional_skip"]
        if "confirmed_failure" in row:
            _require(type(row["confirmed_failure"]) is bool, "confirmed_failure must be boolean")
            metadata["confirmed_failure"] = row["confirmed_failure"]
        if row.get("skip_reason") is not None:
            metadata["skip_reason"] = _text(row["skip_reason"], "skip_reason")
        if "agent_eval" in metadata and payload.get("tier3_applicability") is not None:
            metadata["tier3_applicability"] = mapping(payload["tier3_applicability"], "tier3_applicability")
        _typed_metadata(metadata)
        result = ValidationResult(
            validator_name=name,
            validator_description=_text(row.get("description"), "description"),
            passed=row["passed"],
            findings=findings,
            success_details=details,
            summary=ValidationSummary(**summary),
            metadata=metadata,
            **{key: _strings(legacy.get(key), f"legacy.{key}") for key in ("errors", "warnings", "messages")},
        )
        if metadata.get("confirmed_failure"):
            _require(
                not result.passed and result.summary.errors > 0 and bool(result.errors),
                "Confirmed failure requires failing diagnostics",
            )
        tier = 3 if is_tier3_result(result) else 2 if is_tier2_result(result) else 1
        _require(
            allowed_tiers is None or tier in allowed_tiers,
            "Only retained Tier 1 reports are supported" if allowed_tiers == {1} else "Result tier is not allowed",
        )
        if "gating" in metadata:
            _require(
                type(metadata["gating"].get("tier")) is int
                and metadata["gating"]["tier"] == tier
                and type(metadata["gating"].get("blocking")) is bool,
                f"Invalid Tier {tier} gating",
            )
        if "quality_scores" in metadata:
            quality = metadata["quality_scores"]
            _require(isinstance(quality.get("skill_name"), str), "Invalid quality skill_name")
            for key in ("overall_score", "score"):
                if key in quality:
                    _require(type(quality[key]) in (int, float) and 0 <= quality[key] <= 100, "Invalid quality score")
        expected_status = result_status(result).value
        _require(row.get("status", expected_status) == expected_status, "Result status contradicts evidence")
        verdict_consistent = (
            is_cleanly_skipped(result)
            if expected_status == "skipped"
            else result.passed == (expected_status == "passed")
        )
        if tier == 3 and result.passed:
            # Native Tier 3's boolean records completed execution; quality can
            # still fail or be inconclusive after both arms produced scores.
            agent = metadata.get("agent_eval") or {}
            quality_verdict = agent.get("verdict") or (agent.get("summary") or {}).get("verdict")
            verdict_consistent = verdict_consistent or (
                agent.get("execution_status") == "succeeded"
                and (expected_status, quality_verdict) in {("failed", "fail"), ("incomplete", "inconclusive")}
            )
        _require(verdict_consistent, "Result verdict contradicts evidence")
        results.append(result)
    if "overall_passed" in payload:
        _require(
            type(payload["overall_passed"]) is bool
            and payload["overall_passed"] == all(passes_required_gate(r) for r in results),
            "Overall verdict contradicts retained results",
        )
    overall = resolve_run_status(results).headline.value
    if "overall_status" in payload:
        _require(payload["overall_status"] == overall, "Overall status contradicts retained results")
    if "total_validators" in payload:
        _require(
            type(payload["total_validators"]) is int and payload["total_validators"] == len(results),
            "Validator count contradicts retained results",
        )
    return results


def render_retained_report(
    payload: dict,
    *,
    skill_name: str,
    expected_skill_name: str | None = None,
    context: dict | None = None,
    scan_root: str | None = None,
) -> str:
    """Render a single skill from retained evidence without inspecting its paths.

    *scan_root* is the folder the scanners ran on. It is only compared with
    recorded finding paths, never opened.
    """
    _require(
        isinstance(skill_name, str)
        and 0 < len(skill_name.encode("utf-8")) <= 4096
        and not any(ord(c) < 32 or ord(c) == 127 for c in skill_name),
        "Invalid explicit skill display path",
    )
    canonical_name = expected_skill_name if expected_skill_name is not None else skill_name
    _require(
        isinstance(canonical_name, str)
        and 0 < len(canonical_name) <= 256
        and not any(ord(c) < 32 or ord(c) == 127 for c in canonical_name),
        "Invalid expected skill identity",
    )
    results = retained_results(payload)
    return HTMLReporter(
        include_timestamp=False,
        expected_skill_name=canonical_name,
        explicit_skill_name=skill_name,
        ci_context=context,
        scan_root=scan_root,
    ).render_all(results)
