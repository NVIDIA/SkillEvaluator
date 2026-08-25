# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map native Harbor (Tier 3) results into the canonical ``agent_eval`` payload.

SkillEvaluator folds Tier 3 into the *combined* ``validate`` report (HTML / JSON /
BENCHMARK.md) by attaching a canonical ``metadata["agent_eval"]`` payload to a
single ``AGENT_EVAL`` :class:`~skillevaluator.models.result.ValidationResult`. The
shared reporters (ported faithfully from SkillEvaluator) consume that payload.

SkillEvaluator runs Tier 3 through its own in-process Harbor engine, which writes
per-agent results to disk rather than returning a canonical payload. This module
reads those on-disk results and produces the same canonical ``agent_eval`` shape
so ``validate --agent-eval`` emits one combined report containing all three tiers
-- restoring parity with SkillEvaluator.
"""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from skillevaluator import __version__
from skillevaluator.constants import (
    AGENT_EVAL_EVALUATORS,
    AGENT_EVAL_SCORE_DEFINITION,
    DIMENSION_HINTS,
    DIMENSION_MAPPING,
    DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
    DIMENSION_VERDICT_PASS_THRESHOLD,
    TIER3_LIFT_FAIL_THRESHOLD,
    TIER3_LIFT_PASS_THRESHOLD,
)
from skillevaluator.models.result import Finding, Severity, ValidationResult

# Verdict labels mirror SkillEvaluator's AGENT_EVAL_VERDICT_* values so the ported
# reporters classify the overall outcome identically.
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_NEUTRAL = "neutral"

_AGENT_EVAL_VALIDATOR = "AGENT_EVAL"
_AGENT_EVAL_DESCRIPTION = "Tier 3: Live Agent Evaluation (Harbor)"

_DIMENSION_IDS = list(DIMENSION_MAPPING.keys())

_SCHEMA_VERSION = "2.0"
_TIER3_FEEDBACK_SCHEMA_VERSION = "1.0"
_TIER3_FEEDBACK_FIELDS = ("conclusions", "recommendations", "suggestions", "suggestions_v2")

# Canonical reports are self-contained HTML/JSON artifacts, so untrusted custom
# grader cardinality must not multiply metric-by-trial detail without bound. The
# full Harbor artifacts remain available under ``provenance.run_dir``.
_MAX_EVALUATOR_CARDS_TOTAL = 64
_MAX_EVIDENCE_PER_CARD = 16
_MAX_EVIDENCE_SCAN_PER_CARD = 64
_MAX_EVIDENCE_ENTRIES_TOTAL = 256
_MAX_RAW_TRIAL_REWARDS_TOTAL = 256
_MAX_RAW_METRICS_PER_REWARD = 64
_MAX_RAW_REWARD_FIELDS = 96
_MAX_CUSTOM_METRIC_NAME_VISITS_PER_REWARD = 128
_MAX_UNPAIRED_CASE_IDS_IN_REPORT = 64
_MAX_EMBEDDED_REPORT_BYTES = 2 * 1024 * 1024


def _finite_float(value: object) -> float | None:
    """Return a JSON-safe finite number, rejecting booleans and non-finite floats."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) else None


def _sanitize_json_numbers(value: Any) -> Any:
    """Copy a canonical payload while replacing non-finite floats with JSON null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_json_numbers(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize_json_numbers(item) for item in value]
    return value


@dataclass
class _ReportBudget:
    cards_remaining: int = _MAX_EVALUATOR_CARDS_TOTAL
    evidence_remaining: int = _MAX_EVIDENCE_ENTRIES_TOTAL
    raw_rewards_remaining: int = _MAX_RAW_TRIAL_REWARDS_TOTAL
    omitted: dict[str, int] = field(default_factory=dict)
    deduplicated_evidence: int = 0
    artifact_loading: list[dict[str, Any]] = field(default_factory=list)

    def omit(self, section: str, count: int = 1) -> None:
        if count > 0:
            self.omitted[section] = self.omitted.get(section, 0) + count

    @property
    def truncated(self) -> bool:
        return bool(self.omitted or self.artifact_loading)

    def signal(self) -> dict[str, Any]:
        signal: dict[str, Any] = {
            "truncated": True,
            "reason": (
                "Embedded report details were bounded; retained Harbor artifacts are referenced "
                "by provenance.run_dir when available."
            ),
            "payload_budget_bytes": _MAX_EMBEDDED_REPORT_BYTES,
            "limits": {
                "evaluator_cards": _MAX_EVALUATOR_CARDS_TOTAL,
                "evidence_per_card": _MAX_EVIDENCE_PER_CARD,
                "evidence_scanned_per_card": _MAX_EVIDENCE_SCAN_PER_CARD,
                "evidence_entries": _MAX_EVIDENCE_ENTRIES_TOTAL,
                "raw_trial_rewards": _MAX_RAW_TRIAL_REWARDS_TOTAL,
                "raw_metrics_per_reward": _MAX_RAW_METRICS_PER_REWARD,
                "custom_metric_name_visits_per_reward": _MAX_CUSTOM_METRIC_NAME_VISITS_PER_REWARD,
                "unpaired_case_id_samples": _MAX_UNPAIRED_CASE_IDS_IN_REPORT,
            },
            "omitted": dict(sorted(self.omitted.items())),
        }
        if self.deduplicated_evidence:
            signal["deduplicated_evidence"] = self.deduplicated_evidence
        if self.artifact_loading:
            signal["artifact_loading"] = list(self.artifact_loading)
        return signal


_MAX_ARTIFACT_LOADING_REASONS = 16


def _artifact_loading_reasons(
    agents: dict[str, dict[str, Any]],
    dataset: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Collect only bounded, schema-checked loader diagnostics for the report."""
    markers = [info.get("_report_truncation") for info in agents.values() if isinstance(info, dict)]
    markers.append(getattr(dataset, "_report_truncation", None))
    reasons: list[dict[str, Any]] = []
    for marker in markers:
        if not isinstance(marker, dict) or not isinstance(marker.get("reasons"), list):
            continue
        for candidate in marker["reasons"]:
            if not isinstance(candidate, dict):
                continue
            code = candidate.get("code")
            artifact = candidate.get("artifact")
            limit = candidate.get("limit")
            if (
                not isinstance(code, str)
                or not isinstance(artifact, str)
                or not isinstance(limit, int)
                or isinstance(limit, bool)
                or limit < 0
                or len(code) > 64
                or len(artifact) > 64
            ):
                continue
            reason = {"code": code, "artifact": artifact, "limit": limit}
            if reason not in reasons:
                reasons.append(reason)
            if len(reasons) >= _MAX_ARTIFACT_LOADING_REASONS:
                return reasons
    return reasons


def _advisory_agent_eval_payload(
    message: str,
    *,
    skill_name: str | None = None,
    n_attempts: int | None = None,
    pass_threshold: float | None = None,
    stop_on_pass: bool | None = None,
) -> dict[str, Any]:
    """Build the canonical (but empty) ``agent_eval`` payload for a skipped Tier 3 run.

    The combined HTML/JSON report only renders a Tier 3 section when some
    result carries ``metadata["agent_eval"]`` (``HTMLReporter`` keys off it and
    the template gates the Tier 3 tab/card on ``has_tier3``). Attaching this
    minimal payload — verdict ``neutral``, no agents/dimensions, and the skip
    reason surfaced via ``suggestions`` + ``provenance`` — guarantees an
    explicit ``--agent-eval`` request always produces a visible, self-explaining
    Tier 3 section instead of silently disappearing. Mirrors SkillEvaluator, which
    always emits an ``AGENT_EVAL`` result with a payload (e.g.
    ``_tier3_dataset_required_result`` / ``_invalid_skill_evaluator_result``) even when the
    dataset/runtime is unavailable.
    """
    attempt_policy = _default_attempt_policy()
    if n_attempts is not None:
        attempt_policy["max_attempts"] = n_attempts
    if pass_threshold is not None:
        attempt_policy["pass_threshold"] = pass_threshold
    if stop_on_pass is not None:
        attempt_policy["stop_on_pass"] = stop_on_pass
    dataset_summary = _dataset_summary([], [])
    verdict_policy = _verdict_policy(attempt_policy)
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "verdict": VERDICT_NEUTRAL,
        "skill_name": skill_name or "",
        "best_agent": "",
        "agents_run": [],
        "overall_score": None,
        "overall_lift": None,
        "environment": None,
        "runtime_seconds": 0.0,
        "evaluated_at": None,
        "evaluator_version": __version__,
        "dataset_summary": dataset_summary,
        "dataset_digest": None,
        "dataset_digest_algorithm": None,
        "verdict_policy": verdict_policy,
        "execution_status": "skipped",
        "execution_errors": [message],
        "expected_attempts": 0,
        "scored_attempts": 0,
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "summary": summary,
        "skill_name": skill_name or "",
        "verdict": VERDICT_NEUTRAL,
        "best_agent": "",
        "agents_run": [],
        "environment": None,
        "overall_score": None,
        "overall_lift": None,
        "composite_lift": None,
        "execution_status": "skipped",
        "execution_errors": [message],
        "expected_attempts": 0,
        "scored_attempts": 0,
        "runtime_seconds": 0.0,
        "evaluated_at": None,
        "evaluator_version": __version__,
        "agents": {},
        "dimensions": [],
        "evaluators": {},
        "evaluator_cards": [],
        "cases": [],
        "insights": {},
        "suggestions": [message],
        "suggestions_v2": [],
        "metric_ids": [],
        "metric_labels": {},
        "attempt_policy": attempt_policy,
        "dataset": [],
        "dataset_summary": dataset_summary,
        "dataset_digest": None,
        "dataset_digest_algorithm": None,
        "verdict_policy": verdict_policy,
        "provenance": {
            "source": "advisory",
            "reason": "skipped",
            "advisory": True,
            "message": message,
        },
    }


def advisory_skip_result(message: str, *, skill_name: str | None = None) -> ValidationResult:
    """Return a non-blocking Tier 3 result recording why Tier 3 did not produce data.

    Mirrors the advisory Tier 3 behavior for an explicitly requested
    explicitly-requested ``--agent-eval`` that cannot run (missing dataset,
    missing key, unavailable runtime, or an evaluation error) is surfaced as a
    non-blocking note rather than crashing the whole ``validate`` pipeline.

    The result carries an empty (but canonical) ``metadata["agent_eval"]``
    payload so the combined report still renders a Tier 3 section explaining
    *why* live evaluation produced no data. Without it, ``HTMLReporter`` finds
    no ``agent_eval`` metadata and drops the Tier 3 tab/card entirely, so an
    explicit ``--agent-eval`` request looks like it silently "didn't run".
    """
    result = ValidationResult(
        validator_name=_AGENT_EVAL_VALIDATOR,
        validator_description=_AGENT_EVAL_DESCRIPTION,
    )
    result.add_warning(message)
    result.metadata["agent_eval"] = _advisory_agent_eval_payload(message, skill_name=skill_name)
    # The caller keeps Tier 3 outside the CLI exit gate.  The result itself must
    # still tell reporters that no live evaluation succeeded.
    result.passed = False
    return result


def dataset_required_result(
    dataset_path: Path,
    checks: list[Any],
    *,
    blocking: bool,
    skill_name: str,
    source_kind: str,
) -> ValidationResult:
    """Return a policy-aware outcome for an unusable Tier 3 task source."""
    suggestion = f"Create or fix the Tier 3 source under {dataset_path.parent} and rerun validation."
    if blocking:
        message = "Tier 3 is configured as blocking, but no valid Tier 3 task source was found."
        severity = Severity.HIGH
        verdict = VERDICT_FAIL
        applicability = "required"
        execution_status = "failed"
    else:
        message = (
            "Tier 3 was requested but no valid task source was found. "
            "Tier 3 is advisory unless --block-on-agent-eval is enabled."
        )
        severity = Severity.LOW
        verdict = VERDICT_NEUTRAL
        applicability = "not_required"
        execution_status = "not_applicable"

    serialized_checks = [
        {
            "path": str(getattr(check, "path", "")),
            "status": str(getattr(check, "status", "error")),
            "message": str(getattr(check, "message", check)),
        }
        for check in checks
    ]
    result = ValidationResult(
        validator_name=_AGENT_EVAL_VALIDATOR,
        validator_description=_AGENT_EVAL_DESCRIPTION,
    )
    result.add_finding(
        Finding(
            category="AGENT_EVAL",
            severity=severity,
            check_name="eval_dataset_required",
            message=message,
            file_path=str(dataset_path),
            suggestion=suggestion,
            metadata={"source_kind": source_kind, "source_validation": serialized_checks},
        )
    )
    payload = _advisory_agent_eval_payload(message, skill_name=skill_name)
    payload.update(
        {
            "verdict": verdict,
            "execution_status": execution_status,
            "applicability": applicability,
            "reason_code": "eval_dataset_required",
            "source_kind": source_kind,
            "suggestions": [suggestion],
        }
    )
    payload["summary"].update(
        {
            "verdict": verdict,
            "execution_status": execution_status,
            "applicability": applicability,
            "reason_code": "eval_dataset_required",
        }
    )
    payload["provenance"].update(
        {
            "source": "tier3_source_preflight",
            "reason": "eval_dataset_required",
            "advisory": not blocking,
            "source_kind": source_kind,
            "source_validation": serialized_checks,
        }
    )
    result.metadata.update(
        {
            "agent_eval": payload,
            "tier3_applicability": {
                "applicability": applicability,
                "reason_code": "eval_dataset_required",
                "source_kind": source_kind,
            },
        }
    )
    return result


def agent_eval_result_from_run(
    skill_path: Path,
    *,
    results_dir: Path | None = None,
    env_mode: str | None = None,
    engine_result: dict[str, Any] | None = None,
    use_llm_judge: bool = True,
) -> ValidationResult | None:
    """Build an advisory ``AGENT_EVAL`` result from the latest on-disk Harbor run.

    Returns ``None`` when no usable run directory or agent data can be found, so
    the caller can fall back to :func:`advisory_skip_result`.
    """
    from skillevaluator.tier3.results_location import resolve_latest_results

    latest = resolve_latest_results(skill_path, results_dir)
    if not latest.exists():
        return None
    run_dir = latest.resolve() if latest.is_symlink() else latest
    return agent_eval_result_from_directory(
        skill_path,
        run_dir,
        env_mode=env_mode,
        engine_result=engine_result,
        use_llm_judge=use_llm_judge,
    )


def agent_eval_result_from_directory(
    skill_path: Path,
    run_dir: Path,
    *,
    env_mode: str | None = None,
    engine_result: dict[str, Any] | None = None,
    evaluated_at: str | None = None,
    evaluator_version: str | None = None,
    use_llm_judge: bool = True,
) -> ValidationResult | None:
    """Build the canonical ``AGENT_EVAL`` result for one explicit Harbor run."""
    # Imported lazily so base-only Tier 1 workflows do not load Tier 3 helpers.
    from skillevaluator.tier3.harbor.report_data import (
        load_agent_data,
        load_dataset_snapshot,
        load_staged_harbor_dataset,
    )

    run_dir = run_dir.expanduser().resolve()
    from skillevaluator.tier3.results_location import is_legacy_completed_run_dir

    agents = load_agent_data(
        run_dir,
        allow_legacy_missing_status=is_legacy_completed_run_dir(run_dir),
    )
    if not agents:
        return None

    run_truth = _run_truth_metadata(run_dir, engine_result, load_dataset_snapshot(run_dir))
    dataset = run_truth.get("dataset") or load_staged_harbor_dataset(run_dir)
    payload = build_agent_eval_payload(
        skill_path.name,
        agents,
        dataset=dataset,
        attempt_policy=_read_attempt_policy(run_dir),
        run_config=_read_run_config(run_dir),
        env_mode=env_mode,
        runtime_seconds=_runtime_seconds(engine_result),
        harbor_viewer=_harbor_viewer_from_engine_result(engine_result),
        suggestions_v2=_load_suggestions_v2(run_dir, agents),
        run_dir=run_dir,
        comparison=_read_comparison(run_dir),
        evaluated_at=evaluated_at or _evaluated_at_from_run(run_dir, engine_result),
        evaluator_version=evaluator_version or run_truth.get("evaluator_version"),
        persisted_dataset_summary=run_truth.get("dataset_summary"),
        dataset_digest=run_truth.get("dataset_digest"),
        dataset_digest_algorithm=run_truth.get("dataset_digest_algorithm"),
        use_llm_judge=use_llm_judge,
    )
    return _validation_result_from_payload(payload)


def _validation_result_from_payload(payload: dict[str, Any] | None) -> ValidationResult | None:
    """Wrap a canonical Tier 3 payload in the shared validation-result model."""
    if payload is None:
        return None

    result = ValidationResult(
        validator_name=_AGENT_EVAL_VALIDATOR,
        validator_description=_AGENT_EVAL_DESCRIPTION,
    )
    result.metadata["agent_eval"] = payload
    best = payload.get("best_agent") or "n/a"
    if payload.get("execution_status") == "succeeded" and _finite_float(payload.get("overall_score")) is not None:
        result.add_success(
            "agent_eval",
            f"Tier 3 evaluation complete: verdict {str(payload.get('verdict', 'neutral')).upper()}; best agent {best}",
        )
        result.passed = True
    else:
        errors = payload.get("execution_errors") or ["Tier 3 evaluation did not produce a complete scored run"]
        for error in errors:
            result.add_error(str(error))
    return result


def render_agent_eval_html_report(
    skill_path: Path,
    run_dir: Path,
    *,
    output_path: Path | None = None,
    env_mode: str | None = None,
    engine_result: dict[str, Any] | None = None,
    use_llm_judge: bool = True,
) -> Path:
    """Render one standalone Tier 3 run with the canonical HTML reporter."""
    from skillevaluator.reporting import HTMLReporter

    skill_path = skill_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    result = agent_eval_result_from_directory(
        skill_path,
        run_dir,
        env_mode=env_mode,
        engine_result=engine_result,
        use_llm_judge=use_llm_judge,
    )
    if result is None:
        raise ValueError(f"No agent results found in {run_dir}")

    canonical_payload = result.metadata.get("agent_eval")
    if engine_result is not None and isinstance(canonical_payload, dict):
        # Persist only the compact feedback contract needed by the CLI. The
        # complete canonical payload remains in the HTML report and can be much
        # larger because it duplicates trials, datasets, agents, and provenance.
        engine_result["tier3_feedback"] = {
            "schema_version": _TIER3_FEEDBACK_SCHEMA_VERSION,
            **{field: list(canonical_payload.get(field) or []) for field in _TIER3_FEEDBACK_FIELDS},
        }

    target = output_path.expanduser().resolve() if output_path is not None else run_dir / "report.html"
    reporter = HTMLReporter(
        target_path=str(skill_path),
        content_label="Skill",
        tabs=[{"id": "tier3", "label": "Tier 3: Live Agent Evaluation"}],
    )
    reporter.save([result], target)
    return target


def build_agent_eval_payload(
    skill_name: str,
    agents: dict[str, dict[str, Any]],
    *,
    dataset: list[dict[str, Any]] | None = None,
    attempt_policy: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
    env_mode: str | None = None,
    runtime_seconds: float = 0.0,
    harbor_viewer: dict[str, Any] | None = None,
    suggestions_v2: list[dict[str, Any]] | None = None,
    run_dir: Path | None = None,
    comparison: dict[str, Any] | None = None,
    evaluated_at: str | None = None,
    evaluator_version: str | None = __version__,
    persisted_dataset_summary: dict[str, Any] | None = None,
    dataset_digest: str | None = None,
    dataset_digest_algorithm: str | None = None,
    use_llm_judge: bool = True,
) -> dict[str, Any] | None:
    """Assemble the canonical Tier 3 ``agent_eval`` payload from loaded agent data.

    ``agents`` is the structure produced by
    :func:`skillevaluator.tier3.harbor.report_data.load_agent_data`.
    Returns ``None`` when no agent carries usable scores.

    The payload mirrors SkillEvaluator's canonical Tier 3 shape so the ported reporters
    render every Tier 3 sub-tab: per-trial data (``trials`` / per-agent
    ``trials`` + ``pass_at_k``) feeds the Trials tab, deterministic + LLM
    ``conclusions`` / ``recommendations`` / ``suggestions`` feed the Insights tab,
    and ``provenance`` (raw evaluators, raw lift, raw trial rewards) feeds the
    Diagnostics tab.
    """
    from skillevaluator.tier3.harbor.report_data import (
        build_dataset_snapshot,
        deduplicate_dataset_entries,
        metrics_for_agents,
    )

    metrics = metrics_for_agents(agents)
    report_budget = _ReportBudget(artifact_loading=_artifact_loading_reasons(agents, dataset))
    agent_payloads: dict[str, dict[str, Any]] = {}
    for name in sorted(agents):
        info = agents[name]
        model = _agent_model(name, info, run_config)
        agent_payloads[name] = _build_agent(name, info, metrics, model)

    if not agent_payloads:
        return None

    best_agent = _pick_best_agent(agent_payloads)
    detail_priority = ([best_agent] if best_agent else []) + [name for name in agent_payloads if name != best_agent]
    for name in detail_priority:
        _attach_agent_report_details(
            agent_payloads[name],
            agents.get(name, {}),
            report_budget,
        )
    best = agent_payloads.get(best_agent, {})

    execution_errors = list(
        dict.fromkeys(
            str(error) for agent in agent_payloads.values() for error in agent.get("execution_errors", []) if error
        )
    )
    statuses = [agent.get("execution_status") for agent in agent_payloads.values()]
    if statuses and all(status == "succeeded" for status in statuses):
        execution_status = "succeeded"
    elif any(status == "failed" for status in statuses):
        execution_status = "failed"
    elif any(status == "unknown" for status in statuses):
        execution_status = "unknown"
    else:
        execution_status = "skipped"

    raw_overall_score = best.get("with_skill")
    overall_score = _finite_float(raw_overall_score) if execution_status == "succeeded" else None
    overall_lift = _finite_float(best.get("lift"))
    verdict = _overall_verdict_from_agents(agent_payloads) if overall_score is not None else VERDICT_NEUTRAL

    metric_ids = list(best.get("evaluators", {}).keys())
    metric_labels = _metric_labels(metric_ids)

    policy = attempt_policy or _default_attempt_policy()
    canonical_trials = _flatten_trials(agent_payloads)
    public_dataset = deduplicate_dataset_entries([entry for entry in (dataset or []) if isinstance(entry, dict)])
    computed_dataset_truth = (
        build_dataset_snapshot(public_dataset, evaluator_version=evaluator_version or "") if public_dataset else None
    )
    dataset_summary = (
        dict(persisted_dataset_summary)
        if isinstance(persisted_dataset_summary, dict)
        else _dataset_summary(public_dataset, canonical_trials)
    )
    effective_dataset_digest = dataset_digest or (
        str(computed_dataset_truth["dataset_digest"]) if computed_dataset_truth else None
    )
    effective_dataset_digest_algorithm = dataset_digest_algorithm or (
        str(computed_dataset_truth["dataset_digest_algorithm"]) if computed_dataset_truth else None
    )
    verdict_policy = _verdict_policy(policy)
    harbor_summary = _merge_harbor_viewer_summaries(
        _harbor_viewer_summary(canonical_trials),
        harbor_viewer,
    )
    best_dimensions = best.get("dimensions", [])
    evidence_links = list(harbor_summary.get("evidence_links") or [])

    summary = {
        "schema_version": _SCHEMA_VERSION,
        "verdict": verdict,
        "skill_name": skill_name,
        "best_agent": best_agent,
        "agents_run": list(agent_payloads.keys()),
        "overall_score": round(overall_score, 4) if overall_score is not None else None,
        "overall_lift": round(overall_lift, 4) if overall_lift is not None else None,
        "environment": env_mode,
        "runtime_seconds": _finite_float(runtime_seconds) or 0.0,
        "evaluated_at": evaluated_at,
        "evaluator_version": evaluator_version,
        "dataset_summary": dataset_summary,
        "dataset_digest": effective_dataset_digest,
        "dataset_digest_algorithm": effective_dataset_digest_algorithm,
        "verdict_policy": verdict_policy,
        "execution_status": execution_status,
        "execution_errors": execution_errors,
        "expected_attempts": sum(
            _as_nonnegative_int(agent.get("expected_attempts")) for agent in agent_payloads.values()
        ),
        "scored_attempts": sum(_as_nonnegative_int(agent.get("scored_attempts")) for agent in agent_payloads.values()),
    }
    if harbor_summary:
        summary["harbor_viewer"] = {
            key: harbor_summary[key] for key in ("job_url", "analysis_url") if harbor_summary.get(key)
        }

    # Deterministic baselines render even when the LLM judge is unavailable, so
    # the Insights tab is never empty for a run that produced scores.
    if overall_score is None:
        failure_message = "; ".join(execution_errors) or "Tier 3 evaluation did not produce a complete scored run"
        deterministic_conclusions = [{"severity": "fail", "title": "Evaluation incomplete", "message": failure_message}]
        deterministic_suggestions = [failure_message]
    else:
        deterministic_conclusions = _build_conclusions(
            agent_payloads, best_dimensions, pass_threshold=_pass_threshold_from_policy(policy)
        )
        deterministic_suggestions = _suggestions_for_dimensions(best_dimensions)
    recommendations = _attach_harbor_evidence_to_recommendations(
        [
            {
                "title": _recommendation_title_from(text),
                "message": text,
                "category": _recommendation_category_from(text),
                "severity": "warn",
                "source": "deterministic",
            }
            for text in deterministic_suggestions
        ],
        evidence_links,
    )

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "summary": summary,
        "skill_name": skill_name,
        "verdict": verdict,
        "best_agent": best_agent,
        "agents_run": list(agent_payloads.keys()),
        "environment": env_mode,
        "overall_score": round(overall_score, 4) if overall_score is not None else None,
        "overall_lift": summary["overall_lift"],
        "composite_lift": round(overall_lift, 4) if overall_lift is not None else None,
        "execution_status": execution_status,
        "execution_errors": execution_errors,
        "expected_attempts": summary["expected_attempts"],
        "scored_attempts": summary["scored_attempts"],
        "runtime_seconds": _finite_float(runtime_seconds) or 0.0,
        "evaluated_at": evaluated_at,
        "evaluator_version": evaluator_version,
        "dataset_summary": dataset_summary,
        "dataset_digest": effective_dataset_digest,
        "dataset_digest_algorithm": effective_dataset_digest_algorithm,
        "verdict_policy": verdict_policy,
        "agents": agent_payloads,
        "dimensions": best_dimensions,
        "dimension_hints": dict(DIMENSION_HINTS),
        "evaluators": best.get("evaluators", {}),
        "evaluator_cards": best.get("evaluator_cards", []),
        "cases": best.get("cases", []),
        "trials": canonical_trials,
        "pass_at_k": best.get("pass_at_k", {}),
        "insights": _insights_from_dimensions(best_dimensions),
        "conclusions": list(deterministic_conclusions),
        "recommendations": recommendations,
        "suggestions": list(deterministic_suggestions),
        "suggestions_v2": _attach_harbor_evidence_to_suggestions_v2(suggestions_v2 or [], evidence_links),
        "metric_ids": metric_ids,
        "supported_metric_ids": list(AGENT_EVAL_EVALUATORS),
        "metric_labels": metric_labels,
        "attempt_policy": policy,
        "dataset": public_dataset,
        "provenance": _build_provenance(
            agent_payloads,
            agents,
            run_dir,
            comparison,
            report_budget,
            detail_priority=detail_priority,
        ),
    }
    if harbor_summary:
        payload["harbor_viewer"] = harbor_summary

    _layer_llm_insights(
        payload,
        deterministic_conclusions=deterministic_conclusions,
        deterministic_suggestions=deterministic_suggestions,
        use_llm_judge=use_llm_judge and overall_score is not None,
    )
    if evidence_links:
        payload["conclusions"] = _attach_harbor_evidence_to_conclusions(
            payload.get("conclusions") or [],
            evidence_links,
        )
        payload["recommendations"] = _attach_harbor_evidence_to_recommendations(
            payload.get("recommendations") or [],
            evidence_links,
        )
        payload["suggestions_v2"] = _attach_harbor_evidence_to_suggestions_v2(
            payload.get("suggestions_v2") or [],
            evidence_links,
        )
    payload = _sanitize_json_numbers(payload)
    _enforce_report_payload_budget(payload, report_budget)
    return payload


def _layer_llm_insights(
    payload: dict[str, Any],
    *,
    deterministic_conclusions: list[dict[str, Any]],
    deterministic_suggestions: list[str],
    use_llm_judge: bool,
) -> None:
    """Append LLM-as-Judge conclusions/recommendations on top of the deterministic
    baselines. The judge never raises; when the LLM is unavailable the
    deterministic content is preserved unchanged (SkillEvaluator parity).
    """
    if not use_llm_judge:
        return
    try:
        from skillevaluator.evaluation.insights_judge import build_insights

        extra = build_insights(
            payload,
            deterministic={
                "conclusions": deterministic_conclusions,
                "suggestions": deterministic_suggestions,
            },
            use_llm=True,
        )
    except Exception:  # pragma: no cover - judge already handles failures
        extra = {"conclusions": [], "recommendations": []}

    for item in extra.get("conclusions") or []:
        payload["conclusions"].append(item)
    for item in extra.get("recommendations") or []:
        payload["recommendations"].append(item)
        text = item.get("message") or item.get("title")
        if isinstance(text, str) and text and text not in payload["suggestions"]:
            payload["suggestions"].append(text)


def _build_provenance(
    agent_payloads: dict[str, dict[str, Any]],
    raw_agents: dict[str, dict[str, Any]],
    run_dir: Path | None,
    comparison: dict[str, Any] | None,
    report_budget: _ReportBudget,
    *,
    detail_priority: list[str],
) -> dict[str, Any]:
    """Assemble the Diagnostics ``provenance`` block.

    Mirrors SkillEvaluator's Harbor provenance: per-agent raw evaluator scores and lift
    feed the "Raw Evaluator Scores Per Agent" / "Raw Lift Per Agent" diagnostics
    panels, ``comparison`` feeds the "comparison.json" panel, and
    ``raw_trial_rewards`` preserves the underlying Harbor reward scores for deep
    dives. ``evaluator_paths`` stays empty for SkillEvaluator's in-process Harbor runs
    (no SkillEvaluator subprocess artifacts).
    """
    return {
        "source": "harbor",
        "run_dir": str(run_dir) if run_dir else None,
        "raw_evaluators": {name: ap.get("evaluators", {}) for name, ap in agent_payloads.items()},
        "raw_lift": {
            name: {m: e.get("lift") for m, e in ap.get("evaluators", {}).items()} for name, ap in agent_payloads.items()
        },
        "raw_trial_rewards": {
            name: _raw_trial_rewards(raw_agents.get(name, {}), report_budget)
            for name in detail_priority
            if name in agent_payloads
        },
        "evaluator_paths": {},
        "comparison": comparison if isinstance(comparison, dict) else {},
    }


# Verbose per-evaluator ``details`` / ``custom_details`` (evidence refs,
# per-check breakdowns) are
# dropped from the diagnostics payload: they are not rendered by any report
# panel and would multiply the embedded JSON size several-fold. The full
# details remain on disk under ``provenance.run_dir`` for deep dives.
_REWARD_HEAVY_KEYS = frozenset({"details", "custom_details"})


def _raw_trial_rewards(info: dict[str, Any], report_budget: _ReportBudget) -> list[dict[str, Any]]:
    """Return compact raw Harbor reward dicts (internal + verbose keys stripped)."""
    source_rewards = info.get("rewards") or []
    total_rewards = len(source_rewards)
    if report_budget.raw_rewards_remaining <= 0:
        report_budget.omit("raw_trial_rewards", total_rewards)
        return []

    rewards: list[dict[str, Any]] = []
    for reward_index, reward in enumerate(source_rewards):
        if report_budget.raw_rewards_remaining <= 0:
            report_budget.omit("raw_trial_rewards", total_rewards - reward_index)
            break
        if not isinstance(reward, dict):
            continue

        compact: dict[str, Any] = {}
        if "custom_details" in reward:
            report_budget.omit("raw_detail_fields")
        for field_index, (key, value) in enumerate(reward.items()):
            if key.startswith("_") or key in _REWARD_HEAVY_KEYS:
                continue
            if len(compact) >= _MAX_RAW_REWARD_FIELDS:
                report_budget.omit("raw_reward_fields", len(reward) - field_index)
                break
            if key in {"custom_metrics", "metrics"} and isinstance(value, dict):
                value = _bounded_raw_metric_mapping(value, report_budget)
            compact[key] = value

        rewards.append(compact)
        report_budget.raw_rewards_remaining -= 1
    return rewards


def _bounded_raw_metric_mapping(value: dict[Any, Any], report_budget: _ReportBudget) -> dict[str, Any]:
    """Keep a deterministic representative slice of raw custom metric maps."""
    bounded: dict[str, Any] = {}
    candidates = list(islice(value.items(), _MAX_RAW_METRICS_PER_REWARD + 1))
    for raw_name, raw_value in sorted(candidates, key=lambda item: str(item[0])):
        if len(bounded) >= _MAX_RAW_METRICS_PER_REWARD:
            break
        name = str(raw_name)
        if len(name) > 256 or name in bounded:
            continue
        bounded[name] = raw_value
    report_budget.omit("raw_metric_values", max(0, len(value) - len(bounded)))
    return bounded


def _prune_non_best_agent_details(payload: dict[str, Any], report_budget: _ReportBudget) -> None:
    """Drop duplicated lower-priority details before touching best-agent evidence."""
    best_agent = str(payload.get("best_agent") or "")
    agents = payload.get("agents")
    if not isinstance(agents, dict):
        return

    omitted = 0
    provenance = payload.get("provenance")
    raw_rewards = provenance.get("raw_trial_rewards") if isinstance(provenance, dict) else None
    for name, agent in agents.items():
        if name == best_agent or not isinstance(agent, dict):
            continue
        for key in ("evaluator_cards", "trials", "trials_baseline", "cases"):
            items = agent.get(key)
            if isinstance(items, list) and items:
                omitted += len(items)
                agent[key] = []
        if agent.get("conditions"):
            omitted += 1
            agent["conditions"] = {}
        if isinstance(raw_rewards, dict):
            items = raw_rewards.get(name)
            if isinstance(items, list) and items:
                omitted += len(items)
                raw_rewards[name] = []
    report_budget.omit("non_best_agent_details", omitted)


def _prune_pass_at_k_pairing_diagnostics(payload: dict[str, Any], report_budget: _ReportBudget) -> None:
    """Bound legacy full mismatch-ID arrays while preserving counts and samples."""
    pass_at_k_payloads = [payload.get("pass_at_k")]
    agents = payload.get("agents")
    if isinstance(agents, dict):
        pass_at_k_payloads.extend(agent.get("pass_at_k") for agent in agents.values() if isinstance(agent, dict))

    seen: set[int] = set()
    omitted = 0
    for pass_at_k in pass_at_k_payloads:
        if not isinstance(pass_at_k, dict):
            continue
        lift = pass_at_k.get("lift")
        paired = lift.get("paired_comparison") if isinstance(lift, dict) else None
        if not isinstance(paired, dict) or id(paired) in seen:
            continue
        seen.add(id(paired))

        for condition in ("with_skill", "without_skill"):
            ids_key = f"{condition}_unpaired_case_ids"
            count_key = f"{condition}_unpaired_case_count"
            truncated_key = f"{ids_key}_truncated"
            case_ids = paired.get(ids_key)
            if not isinstance(case_ids, list):
                continue

            declared_count = paired.get(count_key)
            if not isinstance(declared_count, int) or isinstance(declared_count, bool) or declared_count < 0:
                declared_count = 0
            paired[count_key] = max(declared_count, len(case_ids))
            if len(case_ids) > _MAX_UNPAIRED_CASE_IDS_IN_REPORT:
                omitted += len(case_ids) - _MAX_UNPAIRED_CASE_IDS_IN_REPORT
                paired[ids_key] = case_ids[:_MAX_UNPAIRED_CASE_IDS_IN_REPORT]
                paired[truncated_key] = True
            else:
                paired[truncated_key] = bool(paired.get(truncated_key))

    report_budget.omit("unpaired_case_ids", omitted)


def _enforce_report_payload_budget(payload: dict[str, Any], report_budget: _ReportBudget) -> None:
    """Keep the complete self-contained payload within a hard serialized budget.

    Cardinality limits normally keep the payload comfortably below the cap. The
    staged pruning below is a final fail-safe for unusually large diagnostics,
    datasets, or user-authored strings. Every lossy stage is surfaced through
    ``report_truncation`` and the original run artifacts remain on disk.
    """

    def refresh_signal() -> None:
        if report_budget.truncated:
            payload["report_truncation"] = report_budget.signal()

    # Older artifacts may contain every unmatched case identifier. Normalize
    # them before the size check so a report cannot discard all agents and
    # pass@k truth merely because diagnostic IDs were duplicated into the
    # best-agent and top-level projections.
    _prune_pass_at_k_pairing_diagnostics(payload, report_budget)
    refresh_signal()
    if _serialized_payload_size(payload) <= _MAX_EMBEDDED_REPORT_BYTES:
        return

    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        comparison = provenance.get("comparison")
        if comparison:
            provenance["comparison"] = {}
            report_budget.omit("comparison_payloads")
    refresh_signal()

    if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES:
        _prune_non_best_agent_details(payload, report_budget)
        refresh_signal()

    if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES:
        omitted_items = 0
        for key in ("dataset", "trials"):
            items = payload.get(key)
            if isinstance(items, list) and items:
                omitted_items += len(items)
                payload[key] = []
        for agent in (payload.get("agents") or {}).values():
            if not isinstance(agent, dict):
                continue
            for key in ("trials", "trials_baseline", "cases"):
                items = agent.get(key)
                if isinstance(items, list) and items:
                    omitted_items += len(items)
                    agent[key] = []
            if agent.get("conditions"):
                agent["conditions"] = {}
                omitted_items += 1
        report_budget.omit("dataset_and_trial_items", omitted_items)
        refresh_signal()

    if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES:
        raw_rewards = provenance.get("raw_trial_rewards") if isinstance(provenance, dict) else None
        if isinstance(raw_rewards, dict):
            omitted = sum(len(items) for items in raw_rewards.values() if isinstance(items, list))
            provenance["raw_trial_rewards"] = {name: [] for name in raw_rewards}
            report_budget.omit("raw_trial_rewards", omitted)
        refresh_signal()

    if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES:
        omitted_evidence = 0
        for agent in (payload.get("agents") or {}).values():
            if not isinstance(agent, dict):
                continue
            for card in agent.get("evaluator_cards") or []:
                if isinstance(card, dict) and isinstance(card.get("evidence"), list):
                    omitted_evidence += len(card["evidence"])
                    card["evidence"] = []
        for card in payload.get("evaluator_cards") or []:
            if isinstance(card, dict) and isinstance(card.get("evidence"), list):
                card["evidence"] = []
        report_budget.omit("evidence_entries", omitted_evidence)
        refresh_signal()

    if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES:
        omitted_cards = 0
        for agent in (payload.get("agents") or {}).values():
            if isinstance(agent, dict):
                omitted_cards += len(agent.get("evaluator_cards") or [])
                agent["evaluator_cards"] = []
        payload["evaluator_cards"] = []
        report_budget.omit("evaluator_cards", omitted_cards)
        for key in ("conclusions", "recommendations", "suggestions", "suggestions_v2"):
            items = payload.get(key)
            if isinstance(items, list) and items:
                report_budget.omit("insight_items", len(items))
                payload[key] = []
        refresh_signal()

    if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES:
        _replace_with_minimal_payload(payload, report_budget)

    refresh_signal()


def _serialized_payload_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _replace_with_minimal_payload(payload: dict[str, Any], report_budget: _ReportBudget) -> None:
    """Last-resort bounded shape for pathological single-field payloads."""
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    compact_summary = {
        key: value
        for key, value in summary.items()
        if key
        in {
            "schema_version",
            "verdict",
            "overall_score",
            "overall_lift",
            "environment",
            "runtime_seconds",
            "execution_status",
            "expected_attempts",
            "scored_attempts",
        }
    }
    compact_summary["skill_name"] = str(summary.get("skill_name") or payload.get("skill_name") or "")[:256]
    compact_summary["best_agent"] = str(summary.get("best_agent") or payload.get("best_agent") or "")[:256]
    compact_summary["agents_run"] = [str(name)[:256] for name in (summary.get("agents_run") or [])[:64]]
    compact_summary["execution_errors"] = [str(error)[:1024] for error in (summary.get("execution_errors") or [])[:16]]

    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    compact = {
        "schema_version": payload.get("schema_version", _SCHEMA_VERSION),
        "summary": compact_summary,
        "skill_name": compact_summary["skill_name"],
        "verdict": payload.get("verdict", VERDICT_NEUTRAL),
        "best_agent": compact_summary["best_agent"],
        "agents_run": compact_summary["agents_run"],
        "environment": payload.get("environment"),
        "overall_score": payload.get("overall_score"),
        "overall_lift": payload.get("overall_lift"),
        "composite_lift": payload.get("composite_lift"),
        "execution_status": payload.get("execution_status"),
        "execution_errors": compact_summary["execution_errors"],
        "expected_attempts": payload.get("expected_attempts", 0),
        "scored_attempts": payload.get("scored_attempts", 0),
        "runtime_seconds": payload.get("runtime_seconds", 0.0),
        "agents": {},
        "dimensions": [],
        "evaluators": {},
        "evaluator_cards": [],
        "cases": [],
        "trials": [],
        "insights": {},
        "conclusions": [],
        "recommendations": [],
        "suggestions": [],
        "suggestions_v2": [],
        "metric_ids": [],
        "metric_labels": {},
        "dataset": [],
        "provenance": {
            "source": provenance.get("source", "harbor"),
            "run_dir": str(provenance.get("run_dir") or "")[:1024] or None,
            "raw_trial_rewards": {},
            "evaluator_paths": {},
            "comparison": {},
        },
    }
    payload.clear()
    payload.update(compact)
    report_budget.omit("payload_sections")


# ---------------------------------------------------------------------------
# Per-agent assembly
# ---------------------------------------------------------------------------


def _condition_quality_available(info: dict[str, Any], condition: str) -> bool:
    """Return whether a condition may contribute score-bearing report fields."""
    conditions = info.get("conditions")
    condition_info = conditions.get(condition) if isinstance(conditions, dict) else None
    status = condition_info.get("execution_status") if isinstance(condition_info, dict) else None
    if status is None:
        status = info.get("execution_status")
    return status not in {"failed", "unknown", "skipped"}


def _build_agent(
    name: str,
    info: dict[str, Any],
    metrics: list[str],
    model: str | None,
) -> dict[str, Any]:
    with_scores = info.get("with_skill") or {}
    without_scores = info.get("without_skill") or {}
    lift_data = info.get("lift") or {}
    with_quality_available = _condition_quality_available(info, "with_skill")
    baseline_quality_available = _condition_quality_available(info, "without_skill")
    if not with_quality_available:
        with_scores = {}
    if not baseline_quality_available:
        without_scores = {}

    evaluators = _build_evaluators(metrics, with_scores, without_scores, lift_data)
    dimensions = _build_dimensions(
        with_scores,
        without_scores,
        info.get("dimensions_with_skill") or {},
        info.get("dimensions_without_skill") or {},
    )
    overall_ws = _mean([d["with_skill"] for d in dimensions])
    overall_bl = _mean([d["baseline"] for d in dimensions])
    if overall_ws is None and not metrics and with_quality_available:
        overall_ws = _finite_float(info.get("overall_with_skill"))
        if overall_ws is None and info.get("rewards_complete") is not False:
            overall_ws = _logical_reward_mean(info.get("rewards"), "overall")
    if overall_bl is None and not metrics and baseline_quality_available:
        overall_bl = _finite_float(info.get("overall_without_skill"))
        if overall_bl is None and info.get("rewards_baseline_complete") is not False:
            overall_bl = _logical_reward_mean(info.get("rewards_baseline"), "overall")
    overall_lift = round(overall_ws - overall_bl, 4) if overall_ws is not None and overall_bl is not None else None

    trials = _normalize_trials(info.get("rewards") or [], metrics)
    baseline_trials = _normalize_trials(info.get("rewards_baseline") or [], metrics)
    _attach_baseline_pairs(trials, baseline_trials, metrics)

    return {
        "name": name,
        "model": model,
        "execution_status": (
            info.get("execution_status")
            if info.get("execution_status") in {"succeeded", "failed", "skipped", "unknown"}
            else "unknown"
        ),
        "execution_errors": [str(error) for error in info.get("execution_errors", [])]
        if isinstance(info.get("execution_errors"), list)
        else [],
        "expected_attempts": _as_nonnegative_int(info.get("expected_attempts")),
        "scored_attempts": _as_nonnegative_int(info.get("scored_attempts")),
        "conditions": info.get("conditions", {}) if isinstance(info.get("conditions"), dict) else {},
        "evaluators": evaluators,
        "evaluator_cards": [],
        "dimensions": dimensions,
        "with_skill": overall_ws,
        "baseline": overall_bl,
        "lift": overall_lift,
        "num_trials": int(info.get("num_trials", 0) or 0),
        "num_trials_baseline": int(info.get("num_trials_baseline", len(baseline_trials)) or 0),
        "trials": trials,
        "trials_baseline": baseline_trials,
        "pass_at_k": {
            "with_skill": info.get("pass_with_skill") or {},
            "without_skill": info.get("pass_without_skill") or {},
            "lift": info.get("pass_lift") or {},
        },
        "cases": _cases(info),
    }


def _attach_agent_report_details(
    agent_payload: dict[str, Any],
    info: dict[str, Any],
    report_budget: _ReportBudget,
) -> None:
    """Populate bounded diagnostic details after the best agent is known.

    Global report limits intentionally prioritize the best-scoring agent. This
    keeps its top-level evaluator cards, evidence, and raw rewards useful even
    when an alphabetically earlier agent has adversarial custom-metric
    cardinality.
    """
    agent_payload["evaluator_cards"] = _evaluator_cards(
        agent_payload.get("evaluators", {}),
        rewards=info.get("rewards") or [],
        custom_with_skill=info.get("custom_with_skill") or {},
        custom_without_skill=info.get("custom_without_skill") or {},
        custom_lift=info.get("custom_lift") or {},
        report_budget=report_budget,
    )


def _build_evaluators(
    metrics: list[str],
    with_scores: dict[str, Any],
    without_scores: dict[str, Any],
    lift_data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    evaluators: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        ws = _finite_float(with_scores.get(metric))
        if ws is None:
            continue
        bl = _finite_float(without_scores.get(metric))
        lift = _lift_value(metric, lift_data)
        if lift is None and bl is not None:
            lift = round(ws - bl, 4)
        evaluators[metric] = {
            "with_skill": ws,
            "baseline": bl,
            "lift": lift if lift is not None else 0.0,
        }
    return evaluators


def _build_dimensions(
    with_scores: dict[str, Any],
    without_scores: dict[str, Any],
    precomputed_with: dict[str, Any],
    precomputed_without: dict[str, Any],
) -> list[dict[str, Any]]:
    dimensions: list[dict[str, Any]] = []
    for dim_id in _DIMENSION_IDS:
        cfg = DIMENSION_MAPPING[dim_id]
        ws = _precomputed_score(precomputed_with, dim_id)
        if ws is None:
            ws = _dimension_score(with_scores, cfg)
        bl = _precomputed_score(precomputed_without, dim_id)
        if bl is None:
            bl = _dimension_score(without_scores, cfg)
        if ws is None and bl is None:
            continue
        lift = round(ws - bl, 4) if ws is not None and bl is not None else None
        entry = precomputed_with.get(dim_id) if isinstance(precomputed_with.get(dim_id), dict) else {}
        # Signals (the evaluators that actually fed this dimension) populate the
        # "Signals" column; reasoning bullets and a deterministic verdict fill
        # the "Reasoning"/"Verdict" columns when the engine left them blank.
        signals = _dimension_signals(entry, with_scores, cfg)
        explanation = entry.get("explanation")
        reasoning_bullets = entry.get("reasoning_bullets")
        if not reasoning_bullets and not explanation:
            reasoning_bullets, explanation = _deterministic_reasoning(ws, bl, lift, signals, with_scores)
        verdict = entry.get("verdict") or _deterministic_verdict(ws)
        dimensions.append(
            {
                "id": dim_id,
                "with_skill": round(ws, 4) if ws is not None else None,
                "score": round(ws, 4) if ws is not None else None,
                "baseline": round(bl, 4) if bl is not None else None,
                "lift": lift,
                "explanation": explanation,
                "verdict": verdict,
                "evaluators": signals,
                "reasoning_bullets": reasoning_bullets or [],
            }
        )
    return dimensions


def _dimension_signals(entry: dict[str, Any], with_scores: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Return the evaluator signals that feed a dimension (Signals column).

    Prefers the engine's precomputed ``sources`` (the evaluators that actually
    contributed to the score), then the configured primary mapping, then the
    legacy fallback mapping — keeping only signals that carry data.
    """
    sources = entry.get("sources") if isinstance(entry, dict) else None
    if isinstance(sources, dict) and sources:
        return list(sources.keys())
    mapped = [e for e in cfg.get("evaluators", []) if e in with_scores]
    if mapped:
        return mapped
    fallback = [e for e in (cfg.get("fallback_evaluators") or []) if e in with_scores]
    return fallback or list(cfg.get("evaluators", []))


def _deterministic_reasoning(
    ws: float | None,
    bl: float | None,
    lift: float | None,
    signals: list[str],
    with_scores: dict[str, Any],
) -> tuple[list[str], str]:
    """Build deterministic reasoning bullets for a dimension (SkillEvaluator parity).

    Reuses the ported dimension-judge helper so the Reasoning column reads
    identically to SkillEvaluator when no LLM explanation is available.
    """
    from skillevaluator.evaluation.dimension_judge import _human_reasoning_bullets

    parts: list[str] = []
    for signal in signals:
        value = _finite_float(with_scores.get(signal))
        if value is not None:
            parts.append(f"{signal}={value:.2f}")
    numeric_with_skill = _finite_float(ws)
    numeric_baseline = _finite_float(bl)
    if numeric_with_skill is None:
        bullets = ["With-skill score unavailable; no verdict was computed."]
        if numeric_baseline is not None:
            bullets.append(
                f"Baseline score {numeric_baseline:.2f}; lift cannot be computed without a with-skill score."
            )
        else:
            bullets.append("No baseline run available; lift cannot be computed.")
        return bullets, " ".join(bullets)
    bullets = _human_reasoning_bullets(
        with_skill=numeric_with_skill,
        baseline=numeric_baseline,
        lift=lift,
        parts=parts,
    )
    return bullets, " ".join(bullets)


def _deterministic_verdict(ws: float | None) -> str | None:
    """Deterministic PASS/NEUTRAL/FAIL verdict for a dimension score."""
    numeric = _finite_float(ws)
    if numeric is None:
        return None
    from skillevaluator.evaluation.dimension_judge import _verdict_for_score

    return _verdict_for_score(numeric)


def _compact_evidence_refs(raw_refs: object) -> list[str]:
    if not isinstance(raw_refs, list):
        return []
    refs: list[str] = []
    for raw in raw_refs:
        if isinstance(raw, str):
            rendered = raw.strip()
        elif isinstance(raw, dict):
            source = str(raw.get("source") or "").strip()
            pointer = str(raw.get("json_pointer") or raw.get("path") or "").strip()
            rendered = f"{source}{pointer}" if source else pointer
        else:
            continue
        if rendered and rendered not in refs:
            refs.append(rendered[:512])
        if len(refs) == 3:
            break
    return refs


def _custom_metric_value(reward: dict[str, Any], metric: str) -> float | None:
    """Read one custom metric without materializing every custom metric in a reward."""
    from skillevaluator.tier3.harbor.metrics import RESERVED_METRIC_NAMES

    if metric in RESERVED_METRIC_NAMES:
        return None

    numeric: float | None = None
    sources = (reward.get("custom_metrics"), reward.get("metrics"), reward)
    for source in sources:
        if not isinstance(source, dict) or metric not in source:
            continue
        value = source.get(metric)
        if isinstance(value, dict):
            value = value.get("score")
        candidate = _finite_float(value)
        if candidate is not None:
            numeric = candidate
    return numeric


def _bounded_custom_metric_names(
    reward: dict[str, Any],
    *,
    excluded: set[str],
    limit: int,
) -> tuple[list[str], bool]:
    """Return a bounded custom-name sample and whether more names may exist."""
    from skillevaluator.tier3.harbor.metrics import RESERVED_METRIC_NAMES

    if limit <= 0:
        return [], False

    sources = [
        source for source in (reward.get("custom_metrics"), reward.get("metrics"), reward) if isinstance(source, dict)
    ]
    total_items = sum(len(source) for source in sources)
    names: list[str] = []
    seen: set[str] = set()
    visits = 0
    for source in sources:
        for raw_name, raw_value in source.items():
            visits += 1
            name = str(raw_name)
            value = raw_value.get("score") if isinstance(raw_value, dict) else raw_value
            if (
                name not in RESERVED_METRIC_NAMES
                and name not in excluded
                and name not in seen
                and _finite_float(value) is not None
            ):
                seen.add(name)
                names.append(name)
                if len(names) >= limit:
                    return names, visits < total_items
            if visits >= _MAX_CUSTOM_METRIC_NAME_VISITS_PER_REWARD:
                return names, visits < total_items
    return names, False


def _metric_evidence(
    metric: str,
    rewards: list[dict[str, Any]],
    report_budget: _ReportBudget,
    sampling: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if report_budget.evidence_remaining <= 0:
        report_budget.omit("evidence_entries", len(rewards))
        return []

    evidence: list[dict[str, Any]] = []
    by_fingerprint: dict[str, dict[str, Any]] = {}
    total_rewards = len(rewards)
    scan_limit = min(total_rewards, _MAX_EVIDENCE_SCAN_PER_CARD)
    scanned_trials = 0
    output_truncated = False
    for reward in islice(rewards, scan_limit):
        if report_budget.evidence_remaining <= 0:
            output_truncated = True
            break
        scanned_trials += 1
        if not isinstance(reward, dict):
            continue
        details = reward.get("details")
        detail = details.get(metric) if isinstance(details, dict) else None
        if not isinstance(detail, dict):
            custom_details = reward.get("custom_details")
            detail = custom_details.get(metric) if isinstance(custom_details, dict) else None
        if not isinstance(detail, dict):
            continue

        raw_score = _finite_float(reward.get(metric))
        if raw_score is None:
            raw_score = _custom_metric_value(reward, metric)

        notes: list[str] = []
        reason = detail.get("reason")
        if isinstance(reason, str) and reason.strip():
            notes.append(reason.strip()[:512])

        failures: list[str] = []
        results = detail.get("results")
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict) or result.get("passed") is not False:
                    continue
                failure = result.get("reason")
                if isinstance(failure, str) and failure.strip():
                    failures.append(failure.strip()[:512])
                if len(failures) == 3:
                    break

        checks: list[str] = []
        criteria = detail.get("criteria")
        if isinstance(criteria, dict):
            checks = [str(name)[:128] for name in criteria][:8]

        entry = {
            "entry_id": str(reward.get("entry_id") or "trial")[:256],
            "score": raw_score,
            "notes": notes,
            "failures": failures,
            "checks": checks,
            "evidence_refs": _compact_evidence_refs(detail.get("evidence_refs")),
        }
        fingerprint = json.dumps(
            entry,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        existing = by_fingerprint.get(fingerprint)
        if existing is not None:
            existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
            report_budget.deduplicated_evidence += 1
            continue

        if len(evidence) >= _MAX_EVIDENCE_PER_CARD or report_budget.evidence_remaining <= 0:
            report_budget.omit("evidence_entries")
            output_truncated = True
            continue
        evidence.append(entry)
        by_fingerprint[fingerprint] = entry
        report_budget.evidence_remaining -= 1
    unscanned_trials = max(0, total_rewards - scanned_trials)
    report_budget.omit("evidence_entries", unscanned_trials)
    if sampling is not None and (output_truncated or unscanned_trials):
        represented_trials = sum(int(item.get("occurrences", 1)) for item in evidence)
        sampling.update(
            {
                "truncated": True,
                "counts_are_lower_bounds": True,
                "scanned_trials": scanned_trials,
                "total_trials": total_rewards,
                "represented_cases": len({str(item.get("entry_id") or "trial") for item in evidence}),
                "represented_trials": represented_trials,
            }
        )
    return evidence


def _custom_metric_score(metric: str, configured: dict[str, Any], rewards: list[dict[str, Any]]) -> float | None:
    value = configured.get(metric)
    if isinstance(value, dict):
        value = value.get("score")
    configured_score = _finite_float(value)
    if configured_score is not None:
        return configured_score
    values = [
        score
        for reward in rewards
        if isinstance(reward, dict) and (score := _custom_metric_value(reward, metric)) is not None
    ]
    return _mean(values)


def _discover_custom_metric_scores(
    custom_with_skill: dict[str, Any],
    rewards: list[dict[str, Any]],
    excluded: set[str],
    limit: int,
    report_budget: _ReportBudget,
) -> dict[str, float]:
    """Discover at most ``limit`` custom names and aggregate reward scores once."""
    if limit <= 0:
        report_budget.omit("evaluator_cards", len(custom_with_skill))
        report_budget.omit("custom_metric_discovery_trials", len(rewards))
        return {}

    candidates: dict[str, None] = {}
    for raw_name in islice(iter(custom_with_skill), limit + 1):
        name = str(raw_name)
        if name not in excluded and name not in candidates:
            candidates[name] = None

    configured_total = len(custom_with_skill)
    if len(candidates) > limit:
        selected = sorted(candidates)[:limit]
        report_budget.omit("evaluator_cards", max(1, configured_total - len(selected)))
        report_budget.omit("custom_metric_discovery_trials", len(rewards))
        return {
            name: score for name in selected if (score := _custom_metric_score(name, custom_with_skill, [])) is not None
        }

    sums: dict[str, float] = dict.fromkeys(candidates, 0.0)
    counts: dict[str, int] = dict.fromkeys(candidates, 0)
    omitted_name_seen = configured_total > len(candidates)
    for reward in rewards:
        if not isinstance(reward, dict):
            continue
        for name in tuple(candidates):
            numeric = _custom_metric_value(reward, name)
            if numeric is not None:
                sums[name] += numeric
                counts[name] += 1

        remaining = limit - len(candidates)
        discovered, truncated = _bounded_custom_metric_names(
            reward,
            excluded=excluded | set(candidates),
            limit=remaining + 1 if remaining > 0 else 1,
        )
        if truncated or len(discovered) > remaining:
            omitted_name_seen = True
        for name in sorted(discovered)[:remaining]:
            candidates[name] = None
            sums[name] = 0.0
            counts[name] = 0
            numeric = _custom_metric_value(reward, name)
            if numeric is not None:
                sums[name] = numeric
                counts[name] = 1

    if omitted_name_seen:
        report_budget.omit("evaluator_cards", max(1, configured_total - len(candidates)))

    scores: dict[str, float] = {}
    for name in sorted(candidates):
        configured = _custom_metric_score(name, custom_with_skill, [])
        if configured is not None:
            scores[name] = configured
        elif counts.get(name, 0):
            scores[name] = round(sums[name] / counts[name], 4)
    return scores


def _evaluator_card(
    metric: str,
    scores: dict[str, Any],
    *,
    label: str,
    rewards: list[dict[str, Any]],
    report_budget: _ReportBudget,
) -> dict[str, Any]:
    ws = _as_float(scores.get("with_skill"))
    evidence_sampling: dict[str, Any] = {}
    card = {
        "id": metric,
        "label": label,
        "with_skill": ws,
        "baseline": scores.get("baseline"),
        "lift": scores.get("lift"),
        "status": "pass" if ws >= 0.8 else ("warn" if ws >= 0.6 else "fail"),
        "evidence": _metric_evidence(metric, rewards, report_budget, evidence_sampling),
    }
    if evidence_sampling:
        card["evidence_sampling"] = evidence_sampling
    return card


def _evaluator_cards(
    evaluators: dict[str, dict[str, Any]],
    *,
    rewards: list[dict[str, Any]],
    custom_with_skill: dict[str, Any],
    custom_without_skill: dict[str, Any],
    custom_lift: dict[str, Any],
    report_budget: _ReportBudget,
) -> list[dict[str, Any]]:
    from skillevaluator.tier3.harbor.metrics import METRIC_DISPLAY

    cards: list[dict[str, Any]] = []
    evaluator_items = list(evaluators.items())
    for evaluator_index, (metric, scores) in enumerate(evaluator_items):
        if report_budget.cards_remaining <= 0:
            report_budget.omit("evaluator_cards", len(evaluator_items) - evaluator_index)
            break
        report_budget.cards_remaining -= 1
        cards.append(
            _evaluator_card(
                metric,
                scores,
                label=METRIC_DISPLAY.get(metric, metric.replace("_", " ").title()),
                rewards=rewards,
                report_budget=report_budget,
            )
        )

    custom_scores = _discover_custom_metric_scores(
        custom_with_skill,
        rewards,
        set(evaluators),
        report_budget.cards_remaining,
        report_budget,
    )
    for metric, with_skill in custom_scores.items():
        report_budget.cards_remaining -= 1
        baseline = _custom_metric_score(metric, custom_without_skill, [])
        lift = _lift_value(metric, custom_lift)
        if lift is None and baseline is not None:
            lift = round(with_skill - baseline, 4)
        cards.append(
            _evaluator_card(
                metric,
                {"with_skill": with_skill, "baseline": baseline, "lift": lift},
                label=f"Custom: {metric}",
                rewards=rewards,
                report_budget=report_budget,
            )
        )
    return cards


def _cases(info: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for reward in info.get("rewards") or []:
        if not isinstance(reward, dict):
            continue
        cases.append(
            {
                "entry_id": reward.get("entry_id"),
                "overall": reward.get("overall"),
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Trials (Trials tab)
# ---------------------------------------------------------------------------


def _normalize_trials(rewards: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    """Project raw Harbor reward dicts into canonical per-trial entries.

    Each reward (loaded by ``_load_agent_data``) carries the per-evaluator
    scores at the top level, an ``overall`` score, and an internal ``_traj``
    annotation with step/token counters. The canonical shape mirrors SkillEvaluator's
    ``_normalize_harbor_trials`` so the ported Trials tab (per-evaluator
    drill-down, token/steps charts, warnings) renders identically.
    """
    # Import lazily: ``skillevaluator.tier3`` imports report construction through
    # its command module, so importing Harbor metrics at module load time creates
    # a collector-first circular import.
    from skillevaluator.tier3.harbor.metrics import (
        DEFAULT_METRIC_SET,
        LEGACY_METRIC_SET,
        metric_set_for_reward,
        metric_value,
    )

    out: list[dict[str, Any]] = []
    for reward in rewards:
        if not isinstance(reward, dict):
            continue
        declared_metric_set = reward.get("metric_set") or reward.get("metric_set_version")
        standard_metric_sets = {DEFAULT_METRIC_SET, LEGACY_METRIC_SET}
        metric_set, standard_metrics = metric_set_for_reward(reward)
        is_declared_custom = bool(declared_metric_set) and str(declared_metric_set) not in standard_metric_sets
        scores = {
            m: numeric
            for m in metrics
            if not (is_declared_custom and m in {"skill_execution", "skill_routing"})
            if (numeric := _finite_float(reward.get(m))) is not None
        }
        trial: dict[str, Any] = {
            "trial_id": reward.get("trial_id"),
            "entry_id": reward.get("entry_id"),
            "scores": scores,
            "overall": _finite_float(reward.get("overall")),
        }
        traj = reward.get("_traj")
        if isinstance(traj, dict):
            trial["steps"] = traj.get("steps")
            trial["tokens"] = {
                "prompt": traj.get("prompt_tokens", 0),
                "completion": traj.get("completion_tokens", 0),
                "cached": traj.get("cached_tokens", 0),
            }
        if reward.get("warnings"):
            trial["warnings"] = list(reward["warnings"])
        if reward.get("error_recovery"):
            trial["error_recovery"] = reward["error_recovery"]
        is_standard_reward = (
            (not declared_metric_set or str(declared_metric_set) in standard_metric_sets)
            and metric_set in standard_metric_sets
            and "skill_execution" in standard_metrics
            and metric_value(reward, "skill_execution") is not None
        )
        if is_standard_reward and reward.get("invocation_evidence_source") == "trajectory":
            for key in ("skill_invoked", "routing_passed"):
                if type(reward.get(key)) is bool:
                    trial[key] = reward[key]
            if "skill_invoked" in trial or "routing_passed" in trial:
                trial["invocation_evidence_source"] = "trajectory"
        harbor_viewer = _normalize_harbor_viewer_metadata(reward.get("harbor_viewer"))
        if harbor_viewer:
            trial["harbor_viewer"] = harbor_viewer
        out.append(trial)
    return out


def _attach_baseline_pairs(
    trials: list[dict[str, Any]],
    baseline_trials: list[dict[str, Any]],
    metrics: list[str],
) -> None:
    """Pair with-skill trials to their baseline counterparts by ``entry_id``.

    Adds ``baseline_overall`` / ``baseline_scores`` / ``lift_scores`` to each
    matched trial so the "Lift per Eval Case" panel can render the per-metric
    deltas (SkillEvaluator parity).
    """
    by_entry: dict[str, list[dict[str, Any]]] = {}
    for trial in baseline_trials:
        entry_id = trial.get("entry_id")
        if entry_id:
            by_entry.setdefault(str(entry_id), []).append(trial)

    for trial in trials:
        entry_id = trial.get("entry_id")
        if not entry_id:
            continue
        matches = by_entry.get(str(entry_id)) or []
        if not matches:
            continue
        baseline = matches.pop(0)
        trial["baseline_overall"] = baseline.get("overall")
        trial["baseline_scores"] = baseline.get("scores") or {}
        lift_scores: dict[str, float] = {}
        scores = trial.get("scores") or {}
        for metric in metrics:
            score = _finite_float(scores.get(metric))
            base = _finite_float(trial["baseline_scores"].get(metric))
            if score is not None and base is not None:
                lift_scores[metric] = round(score - base, 4)
        if lift_scores:
            trial["lift_scores"] = lift_scores


def _flatten_trials(agents: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-agent trials into a single list with ``agent`` annotated."""
    out: list[dict[str, Any]] = []
    for name, agent in agents.items():
        for trial in agent.get("trials", []):
            out.append({"agent": name, **trial})
    return out


# ---------------------------------------------------------------------------
# Harbor Log Viewer links
# ---------------------------------------------------------------------------


def _harbor_viewer_from_engine_result(engine_result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(engine_result, dict):
        return {}
    return _normalize_harbor_upload_summary(engine_result.get("harbor_viewer"))


def _normalize_harbor_upload_summary(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    jobs: list[dict[str, str]] = []
    seen: set[str] = set()
    for upload in raw.get("uploads") or []:
        if not isinstance(upload, dict):
            continue
        job_url = _safe_harbor_viewer_url(upload.get("viewer_url") or upload.get("job_url"))
        if not job_url or job_url in seen:
            continue
        seen.add(job_url)
        analysis_url = _safe_harbor_viewer_url(upload.get("analysis_url")) or _build_job_analysis_url(job_url)
        job: dict[str, str] = {"url": job_url, "analysis_url": analysis_url}
        name = upload.get("uploaded_job_name") or upload.get("job_name") or upload.get("original_job_name")
        if isinstance(name, str) and name.strip():
            job["name"] = name.strip()
        jobs.append(job)

    job_url = _safe_harbor_viewer_url(raw.get("job_url"))
    analysis_url = _safe_harbor_viewer_url(raw.get("analysis_url"))
    if job_url and job_url not in seen:
        jobs.insert(0, {"url": job_url, "analysis_url": analysis_url or _build_job_analysis_url(job_url)})

    summary: dict[str, Any] = {}
    if jobs:
        summary["jobs"] = jobs
        summary["job_url"] = jobs[0]["url"]
        summary["analysis_url"] = jobs[0].get("analysis_url") or _build_job_analysis_url(jobs[0]["url"])
    return summary


def _normalize_harbor_viewer_metadata(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    out: dict[str, Any] = {}
    for key in ("job_name", "job_url", "analysis_url", "trial_url"):
        value = raw.get(key)
        if key.endswith("_url"):
            cleaned = _safe_harbor_viewer_url(value)
            if cleaned:
                out[key] = cleaned
        elif isinstance(value, str) and value.strip():
            out[key] = value.strip()

    evidence_urls = _normalize_harbor_evidence_urls(raw.get("evidence_urls"))
    if evidence_urls:
        out["evidence_urls"] = evidence_urls
    return out or None


def _normalize_harbor_evidence_urls(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []

    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        label: str | None = None
        url: str | None = None
        if isinstance(item, dict):
            url = _safe_harbor_viewer_url(item.get("url") or item.get("href"))
            raw_label = item.get("label") or item.get("text") or item.get("metric")
            if isinstance(raw_label, str) and raw_label.strip():
                label = raw_label.strip()
        elif isinstance(item, str):
            url = _safe_harbor_viewer_url(item)

        if not url or url in seen:
            continue
        seen.add(url)
        step = _step_number_from_url(url)
        normalized: dict[str, Any] = {
            "url": url,
            "label": f"Step {step}" if step else (label or "Trajectory evidence"),
        }
        if label:
            normalized["metric"] = label
        if step:
            normalized["step"] = step
        evidence.append(normalized)
    return evidence


def _safe_harbor_viewer_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    return url


def _build_job_analysis_url(job_url: str) -> str:
    parts = urlsplit(job_url)
    query = [
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key not in {"tab", "view"}
    ]
    query.append(("tab", "analysis"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _step_number_from_url(url: str) -> int | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    for key, value in parse_qsl(parts.query, keep_blank_values=False):
        if key in {"step", "trajectory_step", "trajectoryStep"}:
            try:
                step = int(value)
            except (TypeError, ValueError):
                return None
            return step if step > 0 else None
    fragment = parts.fragment.strip().lower()
    for prefix in ("step-", "trajectory-step-"):
        if fragment.startswith(prefix):
            try:
                step = int(fragment[len(prefix) :])
            except ValueError:
                return None
            return step if step > 0 else None
    return None


def _harbor_viewer_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    jobs: list[dict[str, str]] = []
    evidence_links: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    seen_evidence: set[str] = set()

    for trial in sorted(trials, key=_trial_evidence_sort_key):
        harbor_viewer = trial.get("harbor_viewer")
        if not isinstance(harbor_viewer, dict):
            continue

        job_url = _safe_harbor_viewer_url(harbor_viewer.get("job_url"))
        analysis_url = _safe_harbor_viewer_url(harbor_viewer.get("analysis_url"))
        if job_url and job_url not in seen_jobs:
            seen_jobs.add(job_url)
            job: dict[str, str] = {"url": job_url, "analysis_url": analysis_url or _build_job_analysis_url(job_url)}
            if harbor_viewer.get("job_name"):
                job["name"] = str(harbor_viewer["job_name"])
            jobs.append(job)

        for evidence in harbor_viewer.get("evidence_urls") or []:
            if not isinstance(evidence, dict):
                continue
            url = _safe_harbor_viewer_url(evidence.get("url"))
            if not url or url in seen_evidence:
                continue
            seen_evidence.add(url)
            entry = {
                "url": url,
                "label": _display_label_for_harbor_evidence(evidence),
                "agent": str(trial.get("agent") or ""),
                "trial_id": str(trial.get("trial_id") or ""),
                "entry_id": str(trial.get("entry_id") or ""),
                "kind": "step" if evidence.get("step") else "trial",
            }
            if evidence.get("step"):
                entry["step"] = evidence["step"]
            evidence_links.append(entry)

        trial_url = _safe_harbor_viewer_url(harbor_viewer.get("trial_url"))
        if trial_url and trial_url not in seen_evidence:
            seen_evidence.add(trial_url)
            evidence_links.append(
                {
                    "url": trial_url,
                    "label": str(trial.get("entry_id") or trial.get("trial_id") or "Trial"),
                    "agent": str(trial.get("agent") or ""),
                    "trial_id": str(trial.get("trial_id") or ""),
                    "entry_id": str(trial.get("entry_id") or ""),
                    "kind": "trial",
                }
            )

    if not jobs and not evidence_links:
        return {}

    summary: dict[str, Any] = {"jobs": jobs, "evidence_links": evidence_links}
    if jobs:
        summary["job_url"] = jobs[0]["url"]
        summary["analysis_url"] = jobs[0].get("analysis_url") or _build_job_analysis_url(jobs[0]["url"])
    return summary


def _merge_harbor_viewer_summaries(*summaries: dict[str, Any] | None) -> dict[str, Any]:
    jobs: list[dict[str, str]] = []
    evidence: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    seen_evidence: set[str] = set()

    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        for job in summary.get("jobs") or []:
            if not isinstance(job, dict):
                continue
            url = _safe_harbor_viewer_url(job.get("url") or job.get("job_url"))
            if not url or url in seen_jobs:
                continue
            seen_jobs.add(url)
            normalized: dict[str, str] = {"url": url}
            analysis_url = _safe_harbor_viewer_url(job.get("analysis_url")) or _build_job_analysis_url(url)
            normalized["analysis_url"] = analysis_url
            if job.get("name"):
                normalized["name"] = str(job["name"])
            jobs.append(normalized)
        direct_job = _safe_harbor_viewer_url(summary.get("job_url"))
        if direct_job and direct_job not in seen_jobs:
            seen_jobs.add(direct_job)
            jobs.append(
                {
                    "url": direct_job,
                    "analysis_url": _safe_harbor_viewer_url(summary.get("analysis_url"))
                    or _build_job_analysis_url(direct_job),
                }
            )
        for item in summary.get("evidence_links") or []:
            if not isinstance(item, dict):
                continue
            url = _safe_harbor_viewer_url(item.get("url"))
            if not url or url in seen_evidence:
                continue
            seen_evidence.add(url)
            normalized_evidence = dict(item)
            normalized_evidence["url"] = url
            normalized_evidence["label"] = _display_label_for_harbor_evidence(normalized_evidence)
            step = _step_number_from_url(url)
            if step:
                normalized_evidence["step"] = step
                normalized_evidence["kind"] = "step"
            evidence.append(normalized_evidence)

    merged: dict[str, Any] = {}
    if jobs:
        merged["jobs"] = jobs
        merged["job_url"] = jobs[0]["url"]
        merged["analysis_url"] = jobs[0].get("analysis_url") or _build_job_analysis_url(jobs[0]["url"])
    if evidence:
        merged["evidence_links"] = evidence
    return merged


def _display_label_for_harbor_evidence(evidence: dict[str, Any]) -> str:
    step = evidence.get("step")
    if not step and evidence.get("url"):
        step = _step_number_from_url(str(evidence["url"]))
    if isinstance(step, int) and step > 0:
        return f"Step {step}"
    label = evidence.get("label") or evidence.get("metric") or evidence.get("entry_id") or evidence.get("trial_id")
    return str(label).strip() if label else "evidence"


def _trial_evidence_sort_key(trial: dict[str, Any]) -> tuple[int, str]:
    overall = _finite_float(trial.get("overall"))
    if overall is not None:
        return (0 if overall < 0.8 else 1, f"{overall:.4f}")
    return (2, str(trial.get("entry_id") or trial.get("trial_id") or ""))


def _attach_harbor_evidence_to_recommendations(
    recommendations: list[dict[str, Any]],
    evidence_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not recommendations or not evidence_links:
        return recommendations

    linked: list[dict[str, Any]] = []
    for index, recommendation in enumerate(recommendations):
        if not isinstance(recommendation, dict):
            linked.append(recommendation)
            continue
        entry = dict(recommendation)
        evidence = _grounded_or_positional_evidence(entry, entry.get("evidence"), evidence_links, index)
        if evidence is None:
            entry.pop("evidence", None)
        else:
            entry["evidence"] = evidence
        linked.append(entry)
    return linked


def _attach_harbor_evidence_to_conclusions(
    conclusions: list[dict[str, Any]],
    evidence_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not conclusions or not evidence_links:
        return conclusions

    linked: list[dict[str, Any]] = []
    for conclusion in conclusions:
        if not isinstance(conclusion, dict):
            linked.append(conclusion)
            continue
        entry = dict(conclusion)
        case_ids = _evidence_case_ids(entry)
        if case_ids:
            evidence = _case_matched_evidence(case_ids, evidence_links)
            if evidence is None:
                entry.pop("evidence", None)
            else:
                entry["evidence"] = evidence
        elif not _valid_evidence(entry.get("evidence")):
            entry.pop("evidence", None)
        linked.append(entry)
    return linked


def _attach_harbor_evidence_to_suggestions_v2(
    suggestions: list[dict[str, Any]],
    evidence_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not suggestions or not evidence_links:
        return suggestions

    linked: list[dict[str, Any]] = []
    for index, suggestion in enumerate(suggestions):
        if not isinstance(suggestion, dict):
            linked.append(suggestion)
            continue
        entry = dict(suggestion)
        existing = entry.get("harbor_evidence") or entry.get("evidence")
        evidence = _grounded_or_positional_evidence(entry, existing, evidence_links, index)
        if evidence is None:
            entry.pop("harbor_evidence", None)
        else:
            entry["harbor_evidence"] = evidence
        linked.append(entry)
    return linked


def _valid_evidence(evidence: object) -> bool:
    return isinstance(evidence, dict) and _safe_harbor_viewer_url(evidence.get("url")) is not None


def _evidence_case_ids(item: dict[str, Any]) -> list[str]:
    case_ids = item.get("evidence_case_ids")
    if not isinstance(case_ids, list):
        return []
    return [case_id.strip() for case_id in case_ids if isinstance(case_id, str) and case_id.strip()]


def _case_matched_evidence(
    case_ids: list[str],
    evidence_links: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for case_id in case_ids:
        for evidence in evidence_links:
            if (
                isinstance(evidence, dict)
                and str(evidence.get("entry_id") or "") == case_id
                and _safe_harbor_viewer_url(evidence.get("url"))
            ):
                return evidence
    return None


def _grounded_or_positional_evidence(
    item: dict[str, Any],
    existing: object,
    evidence_links: list[dict[str, Any]],
    index: int,
) -> dict[str, Any] | None:
    case_ids = _evidence_case_ids(item)
    if case_ids:
        return _case_matched_evidence(case_ids, evidence_links)
    if _valid_evidence(existing):
        return existing
    return evidence_links[min(index, len(evidence_links) - 1)]


# ---------------------------------------------------------------------------
# Insights (Insights tab): deterministic conclusions + recommendations
# ---------------------------------------------------------------------------


_RECO_CATEGORY_HINTS: dict[str, str] = {
    "update": "Update",
    "revise": "Update",
    "refactor": "Update",
    "rewrite": "Update",
    "rework": "Update",
    "add": "Add",
    "create": "Add",
    "introduce": "Add",
    "provide": "Add",
    "include": "Add",
    "implement": "Implement",
    "build": "Implement",
    "develop": "Implement",
    "design": "Implement",
    "enable": "Implement",
    "document": "Document",
    "clarify": "Document",
    "describe": "Document",
    "explain": "Document",
    "note": "Document",
    "fix": "Fix",
    "correct": "Fix",
    "resolve": "Fix",
    "address": "Fix",
    "repair": "Fix",
    "test": "Test",
    "verify": "Test",
    "validate": "Test",
    "check": "Test",
    "ensure": "Test",
    "improve": "Improve",
    "expand": "Improve",
    "broaden": "Improve",
    "tighten": "Improve",
}


def _recommendation_category_from(text: str) -> str:
    """Heuristic category derived from the imperative verb of a suggestion."""
    if not isinstance(text, str) or not text.strip():
        return "Action"
    first = text.strip().split()[0].lower().rstrip(",.;:")
    return _RECO_CATEGORY_HINTS.get(first, "Action")


def _recommendation_title_from(text: str) -> str:
    """Short title for a deterministic recommendation (first sentence, truncated)."""
    if not isinstance(text, str):
        return "Action"
    snippet = text.strip().split(".", 1)[0].strip()
    return snippet[:90] if snippet else "Action"


def _build_conclusions(
    agents: dict[str, dict[str, Any]],
    dimensions: list[dict[str, Any]],
    *,
    pass_threshold: float,
) -> list[dict[str, str]]:
    """Generate stable Insights conclusions from canonical scores (SkillEvaluator parity)."""
    conclusions: list[dict[str, str]] = []
    if agents:
        best_name = _pick_best_agent(agents)
        if best_name:
            best = agents[best_name]
            best_score = _finite_float(best.get("with_skill")) or 0.0
            lift = _finite_float(best.get("lift"))
            conclusions.append(
                {
                    "severity": "pass" if best_score >= pass_threshold else "warn",
                    "title": "Best performing agent",
                    "message": (
                        f"{best_name} leads with overall score {best_score:.2f}"
                        + (f" and lift {lift:+.2f}." if lift is not None else ".")
                    ),
                }
            )

    numeric_dims = [d for d in dimensions if _finite_float(d.get("score")) is not None]
    if numeric_dims:
        weakest = min(numeric_dims, key=lambda d: d.get("score", 0.0))
        conclusions.append(
            {
                "severity": "fail" if weakest.get("score", 0.0) < 0.4 else "warn",
                "title": "Weakest dimension",
                "message": (
                    f"{weakest.get('id', 'unknown').title()} is lowest at "
                    f"{weakest.get('score', 0.0):.2f}. {weakest.get('explanation') or ''}"
                ).strip(),
            }
        )

    failing_trials: list[str] = []
    for agent_name, agent in agents.items():
        for trial in agent.get("trials") or []:
            overall = _finite_float(trial.get("overall"))
            if overall is None or overall < pass_threshold:
                failing_trials.append(f"{agent_name}/{trial.get('entry_id') or trial.get('trial_id')}")
    if failing_trials:
        conclusions.append(
            {
                "severity": "warn",
                "title": "Cases needing review",
                "message": (
                    f"{len(failing_trials)} trial(s) missed the pass threshold; examples: "
                    + ", ".join(failing_trials[:5])
                ),
            }
        )
    return conclusions


def _suggestions_for_dimensions(dimensions: list[dict[str, Any]]) -> list[str]:
    """Default suggestions: target the weakest dimensions (SkillEvaluator parity)."""
    pending: list[tuple[float, str]] = []
    for dim in dimensions:
        score = _finite_float(dim.get("with_skill", dim.get("score", 0.0)))
        if score is not None and score < DIMENSION_VERDICT_PASS_THRESHOLD:
            pending.append((score, dim.get("id", "")))
    pending.sort()

    if not pending:
        return ["Skill performance is healthy across all evaluated dimensions; consider expanding eval coverage."]

    return [
        f"Improve {dim_id.title()} (current score {score:.2f}); add eval coverage and tighten skill instructions."
        for score, dim_id in pending[:3]
    ]


def _pass_threshold_from_policy(attempt_policy: dict[str, Any]) -> float:
    value = attempt_policy.get("pass_threshold", 0.50)
    if value is None:
        return 0.50
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.50
    return numeric if math.isfinite(numeric) else 0.50


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _dimension_score(scores: dict[str, Any], cfg: dict[str, Any]) -> float | None:
    value = _weighted(scores, cfg.get("evaluators", []), cfg.get("weights", []))
    if value is None and cfg.get("fallback_evaluators"):
        value = _weighted(scores, cfg["fallback_evaluators"], cfg.get("fallback_weights", []))
    return value


def _weighted(scores: dict[str, Any], evaluators: list[str], weights: list[float]) -> float | None:
    num = 0.0
    den = 0.0
    for evaluator, weight in zip(evaluators, weights, strict=False):
        value = _finite_float(scores.get(evaluator))
        finite_weight = _finite_float(weight)
        if value is not None and finite_weight is not None:
            num += value * finite_weight
            den += finite_weight
    return (num / den) if den > 0 else None


def _precomputed_score(precomputed: dict[str, Any], dim_id: str) -> float | None:
    entry = precomputed.get(dim_id)
    if isinstance(entry, dict):
        return _finite_float(entry.get("score"))
    return None


def _lift_value(metric: str, lift_data: dict[str, Any]) -> float | None:
    entry = lift_data.get(metric)
    if isinstance(entry, dict):
        candidate = entry.get("delta", entry.get("lift"))
        return _finite_float(candidate)
    return _finite_float(entry)


def _verdict_from_lift(lift: float | None) -> str:
    numeric = _finite_float(lift)
    if numeric is None:
        return VERDICT_NEUTRAL
    if numeric >= TIER3_LIFT_PASS_THRESHOLD:
        return VERDICT_PASS
    if numeric <= TIER3_LIFT_FAIL_THRESHOLD:
        return VERDICT_FAIL
    return VERDICT_NEUTRAL


def _agent_quality_verdict(agent: dict[str, Any]) -> str:
    """Classify one supported agent by the canonical dimension gate."""
    if agent.get("execution_status") != "succeeded":
        return VERDICT_NEUTRAL

    dimensions = {
        str(dimension.get("id")): dimension
        for dimension in agent.get("dimensions") or []
        if isinstance(dimension, dict)
    }
    scores: list[float] = []
    for dimension_id in _DIMENSION_IDS:
        dimension = dimensions.get(dimension_id)
        value = (dimension or {}).get("with_skill", (dimension or {}).get("score"))
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return VERDICT_NEUTRAL
        scores.append(float(value))

    if any(score < DIMENSION_VERDICT_NEUTRAL_THRESHOLD for score in scores):
        return VERDICT_FAIL
    if any(score < DIMENSION_VERDICT_PASS_THRESHOLD for score in scores):
        return VERDICT_NEUTRAL
    return VERDICT_PASS


def _overall_verdict_from_agents(agents: dict[str, dict[str, Any]]) -> str:
    """PASS only when one supported agent passes every required dimension."""
    verdicts = [
        _agent_quality_verdict(agent) for agent in agents.values() if agent.get("execution_status") == "succeeded"
    ]
    if any(verdict == VERDICT_PASS for verdict in verdicts):
        return VERDICT_PASS
    if any(verdict == VERDICT_NEUTRAL for verdict in verdicts) or not verdicts:
        return VERDICT_NEUTRAL
    return VERDICT_FAIL


def _pick_best_agent(agents: dict[str, dict[str, Any]]) -> str:
    eligible = {
        name: agent
        for name, agent in agents.items()
        if agent.get("execution_status") == "succeeded" and _finite_float(agent.get("with_skill")) is not None
    }
    if not eligible:
        return ""
    if len(eligible) == 1:
        return next(iter(eligible))

    def _key(item: tuple[str, dict[str, Any]]) -> tuple[float, float]:
        _name, agent = item
        return (_as_float(agent.get("with_skill")), _as_float(agent.get("lift")))

    return max(eligible.items(), key=_key)[0]


def _insights_from_dimensions(dimensions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    insights: dict[str, dict[str, Any]] = {}
    for dim in dimensions:
        dim_id = dim.get("id")
        if not dim_id:
            continue
        insights[dim_id] = {
            "score": dim.get("with_skill"),
            "explanation": dim.get("explanation"),
        }
    return insights


def _metric_labels(metric_ids: list[str]) -> dict[str, str]:
    from skillevaluator.tier3.harbor.metrics import METRIC_DISPLAY

    return {metric: METRIC_DISPLAY.get(metric, metric.replace("_", " ").title()) for metric in metric_ids}


# ---------------------------------------------------------------------------
# On-disk metadata loaders
# ---------------------------------------------------------------------------


def _agent_model(name: str, info: dict[str, Any], run_config: dict[str, Any] | None) -> str | None:
    if isinstance(run_config, dict):
        meta = (run_config.get("agents") or {}).get(name)
        if isinstance(meta, dict) and meta.get("model"):
            return str(meta["model"])
    model = info.get("model")
    return str(model) if model else None


def _read_attempt_policy(run_dir: Path) -> dict[str, Any]:
    policy = _default_attempt_policy()
    policy_file = run_dir / "attempt_policy.json"
    if policy_file.exists():
        with contextlib.suppress(OSError, ValueError):
            loaded = json.loads(policy_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                policy.update(loaded)
    return policy


def _read_run_config(run_dir: Path) -> dict[str, Any]:
    run_config_file = run_dir / "run_config.json"
    if run_config_file.exists():
        with contextlib.suppress(OSError, ValueError):
            loaded = json.loads(run_config_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
    return {}


def _read_comparison(run_dir: Path) -> dict[str, Any]:
    """Read the cross-agent ``comparison.json`` for the Diagnostics tab, if present."""
    comparison_file = run_dir / "comparison.json"
    if comparison_file.exists():
        with contextlib.suppress(OSError, ValueError):
            loaded = json.loads(comparison_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
    return {}


def _run_truth_metadata(
    run_dir: Path,
    engine_result: dict[str, Any] | None,
    persisted_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """Read dataset/evaluator truth owned by the evaluated run, never live source."""
    persisted_result: dict[str, Any] | None = None
    result_file = run_dir / "result.json"
    if result_file.exists():
        with contextlib.suppress(OSError, UnicodeError, ValueError):
            loaded = json.loads(result_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                persisted_result = loaded

    candidates: list[dict[str, Any]] = []
    if isinstance(persisted_snapshot, dict):
        candidates.append(persisted_snapshot)
    for candidate in (engine_result, persisted_result):
        if not isinstance(candidate, dict):
            continue
        nested = candidate.get("dataset_snapshot")
        if isinstance(nested, dict):
            candidates.append(nested)
        candidates.append(candidate)

    truth: dict[str, Any] = {}
    for candidate in candidates:
        if "dataset" not in truth and isinstance(candidate.get("dataset"), list):
            truth["dataset"] = [entry for entry in candidate["dataset"] if isinstance(entry, dict)]
        if "dataset_summary" not in truth and isinstance(candidate.get("dataset_summary"), dict):
            truth["dataset_summary"] = dict(candidate["dataset_summary"])
        for field_name in ("evaluator_version", "dataset_digest", "dataset_digest_algorithm"):
            value = candidate.get(field_name)
            if field_name not in truth and isinstance(value, str) and value.strip():
                truth[field_name] = value.strip()
    return truth


def _evaluated_at_from_run(run_dir: Path, engine_result: dict[str, Any] | None) -> str | None:
    """Return the persisted UTC evaluation time, never a guessed legacy date."""
    candidates: list[dict[str, Any]] = []
    if isinstance(engine_result, dict):
        candidates.append(engine_result)

    result_file = run_dir / "result.json"
    if result_file.exists():
        with contextlib.suppress(OSError, UnicodeError, ValueError):
            loaded = json.loads(result_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                candidates.append(loaded)

    for candidate in candidates:
        value = candidate.get("evaluated_at")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load_suggestions_v2(run_dir: Path, agents: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Read evidence-backed suggestions from the best agent's findings.json."""
    suggestions: list[dict[str, Any]] = []
    for agent_name in agents:
        findings_file = run_dir / agent_name / "findings.json"
        if not findings_file.exists():
            continue
        try:
            data = json.loads(findings_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for item in data.get("suggestions_v2") or []:
            if not isinstance(item, dict):
                continue
            suggestions.append(
                {
                    "metric": item.get("dimension") or item.get("metric") or "agent_eval",
                    "recommendation": item.get("suggestion") or item.get("recommendation") or "",
                    "evidence_refs": item.get("evidence_refs") or [],
                }
            )
        if suggestions:
            break
    return suggestions


def _runtime_seconds(engine_result: dict[str, Any] | None) -> float:
    if not isinstance(engine_result, dict):
        return 0.0
    for key in ("runtime_seconds", "elapsed", "duration_seconds", "total_runtime"):
        value = _finite_float(engine_result.get(key))
        if value is not None:
            return value
    return 0.0


def _default_attempt_policy() -> dict[str, Any]:
    return {
        "max_attempts": 1,
        "pass_threshold": 0.50,
        "stop_on_pass": False,
        "score_definition": AGENT_EVAL_SCORE_DEFINITION,
    }


def _dataset_summary(dataset: list[dict[str, Any]], trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a stable unique-task count and activation-intent summary."""
    if dataset:
        from skillevaluator.tier3.harbor.report_data import summarize_dataset_entries

        return summarize_dataset_entries(dataset)

    task_ids: set[str] = set()
    for trial in trials:
        for key in ("entry_id", "case_id", "task_id", "id"):
            value = trial.get(key)
            if value is not None and str(value).strip():
                task_ids.add(str(value).strip())
                break
    return {
        "total_tasks": len(task_ids),
        "positive_tasks": 0,
        "negative_tasks": 0,
        "unclassified_tasks": len(task_ids),
        "source": "trials" if task_ids else "unavailable",
    }


def _verdict_policy(attempt_policy: dict[str, Any]) -> dict[str, Any]:
    """Expose the distinct task-attempt, dimension, and overall-lift gates."""
    attempt_threshold = attempt_policy.get("pass_threshold")
    return {
        "attempt_pass_threshold": (
            float(attempt_threshold)
            if isinstance(attempt_threshold, (int, float)) and not isinstance(attempt_threshold, bool)
            else None
        ),
        "dimension_pass_threshold": DIMENSION_VERDICT_PASS_THRESHOLD,
        "dimension_neutral_threshold": DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
        "lift_pass_threshold": TIER3_LIFT_PASS_THRESHOLD,
        "lift_fail_threshold": TIER3_LIFT_FAIL_THRESHOLD,
        "overall_pass_rule": "one_supported_agent_all_dimensions_pass",
    }


def _mean(values: list[float]) -> float | None:
    numeric = [finite for value in values if (finite := _finite_float(value)) is not None]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


def _logical_reward_mean(rewards: Any, field: str) -> float | None:
    """Average a persisted reward field once per logical Harbor trial."""
    if not isinstance(rewards, list):
        return None
    from skillevaluator.tier3.harbor.report_data import logical_trial_reward_groups

    group_means = [
        group_mean
        for group in logical_trial_reward_groups([reward for reward in rewards if isinstance(reward, dict)])
        if (group_mean := _mean([reward.get(field) for reward in group])) is not None
    ]
    return _mean(group_means)


def _as_float(value: Any) -> float:
    return _finite_float(value) or 0.0


def _as_nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = [
    "advisory_skip_result",
    "agent_eval_result_from_run",
    "build_agent_eval_payload",
]
