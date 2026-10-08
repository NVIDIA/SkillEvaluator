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
import logging
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
    LIFT_CI_MIN_PAIRED_CASES,
    TIER3_LIFT_FAIL_THRESHOLD,
    TIER3_LIFT_PASS_THRESHOLD,
)
from skillevaluator.evidence import evidence_ref_identity
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.source_identity import resolve_evaluated_source

logger = logging.getLogger(__name__)

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

_INTEGRATION_SCHEMA_VERSION = "1.0"
INTEGRATION_VERDICT_REAL = "real_integration"
INTEGRATION_VERDICT_COSMETIC = "cosmetic_bundling"
INTEGRATION_VERDICT_NEGATIVE = "negative_integration"
INTEGRATION_VERDICT_INCONCLUSIVE = "inconclusive"
_INTEGRATION_REAL_THRESHOLD = 0.05
_INTEGRATION_NEGATIVE_THRESHOLD = -0.05
_INTEGRATION_INTERPRETATION = {
    INTEGRATION_VERDICT_REAL: "The coordinated plugin measurably outperforms its member components alone.",
    INTEGRATION_VERDICT_COSMETIC: "The plugin performs about the same as its member components alone.",
    INTEGRATION_VERDICT_NEGATIVE: "The plugin underperforms its member components alone; inspect coordination overhead.",
    INTEGRATION_VERDICT_INCONCLUSIVE: "The sum-of-parts comparison did not produce complete, conclusive evidence.",
}
_LIFT_MODES = ("effectiveness", "integration", "both")
_INTEGRATION_LIFT_MODES = ("integration", "both")
_INTEGRATION_REASON_NO_WORKSPACE = "The run recorded no plugin workspace, so no member-skills arm could be compared."
_INTEGRATION_REASON_NO_COMPONENTS = "The run staged no member components, so no member-skills arm could be compared."
_INTEGRATION_REASON_NO_ARM = "The member-skills (sum-of-parts) arm was not run, so Integration was not measured."
_INTEGRATION_REASON_NO_SCORE = "The member-skills (sum-of-parts) arm produced no comparable score."
# The per-trial rewards list of each arm in the loaded agent data.
_ARM_REWARDS_FIELDS = {
    "with_skill": "rewards",
    "without_skill": "rewards_baseline",
    "sum_of_parts": "rewards_sum_of_parts",
}

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
_MAX_JSON_SAFE_INTEGER = (1 << 53) - 1


def _finite_float(value: object) -> float | None:
    """Return a JSON-safe finite number, rejecting booleans and non-finite floats."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) else None


def _token_counter(value: object) -> int | None:
    """Return one browser-safe token count, preserving unavailable as null."""
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_JSON_SAFE_INTEGER
        else None
    )


def _sanitize_json_numbers(value: Any) -> Any:
    """Copy a payload while replacing numbers browsers cannot represent safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if -_MAX_JSON_SAFE_INTEGER <= value <= _MAX_JSON_SAFE_INTEGER else None
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
        "evaluated_source": None,
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
        "evaluated_source": None,
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
    dataset_source: Path | None = None,
    env_mode: str | None = None,
    engine_result: dict[str, Any] | None = None,
    plugin_provenance: dict[str, Any] | None = None,
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
        dataset_source=dataset_source,
        env_mode=env_mode,
        engine_result=engine_result,
        plugin_provenance=plugin_provenance,
        use_llm_judge=use_llm_judge,
    )


def agent_eval_result_from_directory(
    skill_path: Path,
    run_dir: Path,
    *,
    dataset_source: Path | None = None,
    env_mode: str | None = None,
    engine_result: dict[str, Any] | None = None,
    plugin_provenance: dict[str, Any] | None = None,
    evaluated_at: str | None = None,
    evaluator_version: str | None = None,
    evaluated_source: dict[str, Any] | None = None,
    use_llm_judge: bool = True,
) -> ValidationResult | None:
    """Build the canonical ``AGENT_EVAL`` result for one explicit Harbor run.

    When the caller passes no ``plugin_provenance``, the run-dir
    ``plugin_provenance.json`` sidecar is read instead, so re-rendering a plugin
    run (``view`` / ``render_agent_eval_html_report``) keeps its provenance and
    INCOMPLETE status instead of degrading to a skill-shaped report.
    """
    # Imported lazily so base-only Tier 1 workflows do not load Tier 3 helpers.
    from skillevaluator.tier3.harbor.report_data import (
        load_agent_data,
        load_dataset,
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

    if plugin_provenance is None:
        plugin_provenance = _read_plugin_provenance(run_dir) or None

    run_truth = _run_truth_metadata(run_dir, engine_result, load_dataset_snapshot(run_dir))
    dataset = (
        run_truth.get("dataset")
        or (load_dataset(dataset_source) if dataset_source is not None else None)
        or load_staged_harbor_dataset(run_dir)
    )
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
        plugin_provenance=plugin_provenance,
        evaluated_at=evaluated_at or _evaluated_at_from_run(run_dir, engine_result),
        evaluator_version=evaluator_version or run_truth.get("evaluator_version"),
        persisted_dataset_summary=run_truth.get("dataset_summary"),
        dataset_digest=run_truth.get("dataset_digest"),
        dataset_digest_algorithm=run_truth.get("dataset_digest_algorithm"),
        evaluated_source=evaluated_source,
        use_llm_judge=use_llm_judge,
    )
    return _validation_result_from_payload(payload)


def incomplete_reason(provenance: dict[str, Any]) -> str:
    """Return why a partial plugin run is INCOMPLETE, worded as every report words it.

    The text is the completeness view's reason: an unreadable provenance
    sidecar, otherwise why the run did not complete or its native plugin load
    was never confirmed, followed by the declared components it deferred.
    """
    return f"INCOMPLETE: {_plugin_completeness(provenance)['reason']}"


def _plugin_completeness(provenance: dict[str, Any]) -> dict[str, Any]:
    from skillevaluator.reporting.plugin_sections import completeness_view

    # An empty record still belongs to a partial run, for a reason nobody recorded.
    return completeness_view(provenance or {"partial": True}) or {}


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
    plugin_provenance = payload.get("plugin_provenance") or {}
    partial = bool(isinstance(plugin_provenance, dict) and plugin_provenance.get("partial"))
    if payload.get("execution_status") == "succeeded" and _finite_float(payload.get("overall_score")) is not None:
        result.add_success(
            "agent_eval",
            f"Tier 3 evaluation complete: verdict {str(payload.get('verdict', 'neutral')).upper()}; best agent {best}",
        )
        result.passed = True
        if partial:
            result.passed = False
            result.metadata["execution_status"] = "skipped"
            result.metadata["skip_reason"] = incomplete_reason(plugin_provenance)
        # A run fails its gate on a FAIL verdict or a confirmed Skill Lift regression,
        # even when it is partial: the parts it did evaluate already failed, so it is
        # FAIL rather than INCOMPLETE on every surface. ``validate`` counts this only
        # with --block-on-agent-eval; without it the result stays advisory.
        failures = _tier3_gate_failures(payload)
        if failures:
            result.metadata["tier3_gate_failures"] = [label for label, _detail in failures]
        for label, detail in failures:
            result.add_error(f"{label}: {detail}")
    else:
        errors = payload.get("execution_errors") or ["Tier 3 evaluation did not produce a complete scored run"]
        for error in errors:
            result.add_error(str(error))
    return result


def _tier3_gate_failures(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(label, detail)`` for each reason a complete, scored Tier 3 run fails its gate.

    A FAIL verdict fails it, and so does a confirmed Skill Lift regression (a
    lift in the FAIL band whose interval lies wholly below zero). NEUTRAL never
    does. The caller decides whether the gate counts toward the exit code
    (``validate --block-on-agent-eval``).
    """
    reasons: list[tuple[str, str]] = []
    if str(payload.get("verdict") or "").lower() == VERDICT_FAIL:
        reasons.append(("Tier 3 verdict FAIL", _failing_dimensions_text(payload)))
    band = payload.get("lift_band")
    if isinstance(band, dict) and band.get("regression_confirmed") is True:
        reasons.append(
            (
                "Tier 3 Skill Lift regression",
                f"lift {_lift_band_text(band)} is at or below "
                f"{_finite_float(band.get('fail_threshold')) or TIER3_LIFT_FAIL_THRESHOLD:+.2f} "
                "with the whole interval below zero",
            )
        )
    return reasons


def _failing_dimensions_text(payload: dict[str, Any]) -> str:
    """Name each scored agent's dimensions below the NEUTRAL threshold."""
    parts: list[str] = []
    agents = payload.get("agents") if isinstance(payload.get("agents"), dict) else {}
    for name in sorted(agents):
        agent = agents[name]
        if not isinstance(agent, dict) or agent.get("execution_status") != "succeeded":
            continue
        low: list[str] = []
        for dimension in agent.get("dimensions") or []:
            if not isinstance(dimension, dict):
                continue
            score = _finite_float(dimension.get("with_skill", dimension.get("score")))
            if score is not None and score < DIMENSION_VERDICT_NEUTRAL_THRESHOLD:
                low.append(f"{dimension.get('id')} {score:.2f}")
        if low:
            parts.append(f"{name}: {', '.join(low)}")
    rule = f"no scored agent kept every dimension at {DIMENSION_VERDICT_NEUTRAL_THRESHOLD:.2f} or above"
    return f"{rule} ({'; '.join(parts)})" if parts else rule


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
    _save_agent_eval_html(result, skill_path, target)
    return target


def _save_agent_eval_html(result: ValidationResult, skill_path: Path, target: Path) -> None:
    """Write the Tier 3-only canonical HTML report for one result."""
    from skillevaluator.reporting import HTMLReporter
    from skillevaluator.reporting.plugin_sections import is_plugin_payload

    payload = result.metadata.get("agent_eval")
    reporter = HTMLReporter(
        target_path=str(skill_path),
        content_label="Plugin" if is_plugin_payload(payload) else "Skill",
        tabs=[{"id": "tier3", "label": "Tier 3: Live Agent Evaluation"}],
    )
    reporter.save([result], target)


def refresh_plugin_run_report(
    skill_path: Path,
    run_dir: Path,
    *,
    result: ValidationResult | None = None,
    env_mode: str | None = None,
    engine_result: dict[str, Any] | None = None,
    use_llm_judge: bool = True,
    plugin_provenance: dict[str, Any] | None = None,
) -> Path | None:
    """Re-render a plugin run's ``report.html`` once its provenance sidecar exists.

    The Harbor runner writes ``report.html`` before the CLI persists
    ``plugin_provenance.json``, so the runner's copy cannot show plugin
    provenance or an INCOMPLETE status, and ``view`` opens that copy. Pass the
    already-built *result* to avoid rebuilding the payload; otherwise the run is
    re-read. The caller's in-memory *plugin_provenance* wins over the sidecar,
    so a sidecar write that failed cannot drop an INCOMPLETE status. Best
    effort: a failure keeps the runner's report.
    """
    try:
        run_dir = Path(run_dir).expanduser().resolve()
        target = run_dir / "report.html"
        if result is None:
            result = agent_eval_result_from_directory(
                Path(skill_path),
                run_dir,
                env_mode=env_mode,
                engine_result=dict(engine_result) if isinstance(engine_result, dict) else None,
                plugin_provenance=plugin_provenance,
                use_llm_judge=use_llm_judge,
            )
        if result is None or not isinstance(result.metadata.get("agent_eval"), dict):
            return None
        _save_agent_eval_html(result, Path(skill_path).expanduser().resolve(), target)
        return target
    except Exception as exc:  # best effort: the runner's report remains usable
        logger.warning("Plugin report refresh failed (%s); keeping the runner's report.html", type(exc).__name__)
        logger.debug("Plugin report refresh failure detail", exc_info=True)
        return None


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
    plugin_provenance: dict[str, Any] | None = None,
    evaluated_at: str | None = None,
    evaluator_version: str | None = __version__,
    persisted_dataset_summary: dict[str, Any] | None = None,
    dataset_digest: str | None = None,
    dataset_digest_algorithm: str | None = None,
    evaluated_source: dict[str, Any] | None = None,
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
    from skillevaluator.tier3.harbor.report_data import build_dataset_snapshot, deduplicate_dataset_entries

    report_budget = _ReportBudget(artifact_loading=_artifact_loading_reasons(agents, dataset))
    agent_payloads = _agent_payloads(agents, run_config, plugin_provenance)

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
    execution_error_details = _aggregate_execution_error_details(agent_payloads, len(execution_errors))
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
    effective_evaluated_source = resolve_evaluated_source(evaluated_source, (run_config or {}).get("evaluated_source"))
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
        "evaluated_source": effective_evaluated_source,
        "verdict_policy": verdict_policy,
        "execution_status": execution_status,
        "execution_errors": execution_errors,
        **execution_error_details,
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
    lift_band = _lift_band(best, run_config, plugin_provenance) if overall_score is not None else None
    if overall_score is None:
        failure_message = "; ".join(execution_errors) or "Tier 3 evaluation did not produce a complete scored run"
        deterministic_conclusions = [{"severity": "fail", "title": "Evaluation incomplete", "message": failure_message}]
        deterministic_suggestions = [failure_message]
    else:
        deterministic_conclusions = _build_conclusions(
            agent_payloads, best_dimensions, pass_threshold=_pass_threshold_from_policy(policy)
        )
        deterministic_suggestions = _suggestions_for_dimensions(best_dimensions)
        lift_uncertainty_warning = _effectiveness_uncertainty_conclusion(best)
        if lift_uncertainty_warning is not None:
            deterministic_conclusions.append(lift_uncertainty_warning)
        if (lift_band_warning := _lift_band_conclusion(lift_band)) is not None:
            deterministic_conclusions.append(lift_band_warning)
    if plugin_provenance and plugin_provenance.get("partial"):
        deterministic_conclusions = [
            _plugin_incompleteness_conclusion(plugin_provenance),
            *deterministic_conclusions,
        ]
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
        **execution_error_details,
        "expected_attempts": summary["expected_attempts"],
        "scored_attempts": summary["scored_attempts"],
        "runtime_seconds": _finite_float(runtime_seconds) or 0.0,
        "evaluated_at": evaluated_at,
        "evaluator_version": evaluator_version,
        "dataset_summary": dataset_summary,
        "dataset_digest": effective_dataset_digest,
        "dataset_digest_algorithm": effective_dataset_digest_algorithm,
        "evaluated_source": effective_evaluated_source,
        "verdict_policy": verdict_policy,
        "agents": agent_payloads,
        "dimensions": best_dimensions,
        "dimension_hints": dict(DIMENSION_HINTS),
        "evaluators": best.get("evaluators", {}),
        "evaluator_cards": best.get("evaluator_cards", []),
        "not_applicable_evaluators": best.get("not_applicable_evaluators", []),
        "not_applicable_dimensions": best.get("not_applicable_dimensions", []),
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
    if lift_band is not None:
        payload["lift_band"] = lift_band
    if plugin_provenance:
        payload["plugin_provenance"] = plugin_provenance
        summary["plugin_provenance"] = plugin_provenance
    for field_name in _statistics_fields():
        if isinstance(best.get(field_name), dict):
            payload[field_name] = best[field_name]
    if _is_plugin_target(run_config) or plugin_provenance:
        lift_modes = _plugin_lift_modes(run_config, plugin_provenance)
        payload["lift_mode_requested"] = lift_modes["requested"]
        payload["lift_mode_effective"] = lift_modes["effective"]
    _attach_integration_reports(payload, agent_payloads, best_agent, run_config, plugin_provenance)
    _attach_plugin_report_fields(payload, agents)

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


def integration_reports_for(
    agents: dict[str, dict[str, Any]],
    run_config: dict[str, Any] | None,
    plugin_provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Return the Integration blocks the report for this run carries, without building the report.

    The first is the run-level block (``payload["integration"]``), ``None``
    when Integration was neither measured nor requested; the second maps each
    agent with a block to its own named block
    (``payload["agents"][name]["integration"]``). *agents* is what
    :func:`build_agent_eval_payload` takes. A caller that needs only these
    blocks (the run summary) skips the payload's evaluator cards, evidence,
    insights and size budget.
    """
    agent_payloads = _agent_payloads(agents, run_config, plugin_provenance)
    integration, per_agent = _integration_reports(
        agent_payloads, _pick_best_agent(agent_payloads), run_config, plugin_provenance
    )
    return (
        _sanitize_json_numbers(integration) if integration is not None else None,
        {name: _sanitize_json_numbers(block) for name, block in per_agent.items()},
    )


def _agent_payloads(
    agents: dict[str, dict[str, Any]],
    run_config: dict[str, Any] | None,
    plugin_provenance: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Build each agent's scores, in name order."""
    from skillevaluator.tier3.harbor.report_data import metrics_for_condition

    sum_of_parts_baseline = _baseline_is_sum_of_parts(run_config, plugin_provenance)
    return {
        name: _build_agent(
            name,
            info,
            metrics_for_condition(info, "with_skill"),
            metrics_for_condition(info, "without_skill"),
            _agent_model(name, info, run_config),
            sum_of_parts_baseline=sum_of_parts_baseline,
        )
        for name, info in sorted(agents.items())
    }


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
    from skillevaluator.tier3.harbor.metrics import (
        RESERVED_METRIC_NAMES,
        custom_metric_name_is_publishable,
    )

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
            elif key not in RESERVED_METRIC_NAMES:
                candidate = value.get("score") if isinstance(value, dict) else value
                custom_score_shape = isinstance(candidate, int | float) and not isinstance(candidate, bool)
                if custom_score_shape and not custom_metric_name_is_publishable(key):
                    report_budget.omit("raw_reward_fields")
                    continue
            compact[key] = value

        rewards.append(compact)
        report_budget.raw_rewards_remaining -= 1
    return rewards


def _bounded_raw_metric_mapping(value: dict[Any, Any], report_budget: _ReportBudget) -> dict[str, Any]:
    """Keep a deterministic representative slice of raw custom metric maps."""
    from skillevaluator.tier3.harbor.metrics import (
        RESERVED_METRIC_NAMES,
        custom_metric_name_is_publishable,
    )

    bounded: dict[str, Any] = {}
    candidates = list(islice(value.items(), _MAX_RAW_METRICS_PER_REWARD + 1))
    for raw_name, raw_value in sorted(candidates, key=lambda item: str(item[0])):
        if len(bounded) >= _MAX_RAW_METRICS_PER_REWARD:
            break
        name = str(raw_name)
        if name in bounded or (name not in RESERVED_METRIC_NAMES and not custom_metric_name_is_publishable(name)):
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
    # Plugin text (a manifest name in the provenance sidecar) can carry a lone
    # surrogate, which strict UTF-8 cannot encode; it still has a size.
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return len(serialized.encode("utf-8", "surrogatepass"))


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
            "execution_error_details_total",
            "execution_error_details_shown",
            "execution_error_details_truncated",
            "expected_attempts",
            "scored_attempts",
        }
    }
    compact_summary["skill_name"] = str(summary.get("skill_name") or payload.get("skill_name") or "")[:256]
    compact_summary["best_agent"] = str(summary.get("best_agent") or payload.get("best_agent") or "")[:256]
    compact_summary["agents_run"] = [str(name)[:256] for name in (summary.get("agents_run") or [])[:64]]
    compact_summary["execution_errors"] = [str(error)[:1024] for error in (summary.get("execution_errors") or [])[:16]]
    compact_summary.update(
        _aggregate_execution_error_details(
            {"summary": summary},
            len(compact_summary["execution_errors"]),
        )
    )

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
        "execution_error_details_total": compact_summary.get("execution_error_details_total", 0),
        "execution_error_details_shown": compact_summary.get("execution_error_details_shown", 0),
        "execution_error_details_truncated": compact_summary.get("execution_error_details_truncated", False),
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


def _comparison_basis(
    entry: object,
    with_dimensions: dict[str, Any],
    control_dimensions: dict[str, Any],
    control_key: str,
) -> dict[str, Any] | None:
    """Return the two arm scores and the lift of one comparison on one shared basis.

    The collector's paired statistics score both arms per case on the
    dimensions both scored, so the headline, the arm scores next to it and the
    interval agree. An arm without the skill has no Discoverability or
    Efficiency, so comparing every dimension would credit skill activation
    alone. Runs without those statistics fall back to the arm means of the
    dimensions both arms scored.
    """
    if isinstance(entry, dict):
        lift = _finite_float(entry.get("estimate"))
        treatment = _finite_float(entry.get("treatment_score"))
        control = _finite_float(entry.get("control_score"))
        if lift is not None and treatment is not None and control is not None:
            return {
                "basis": str(entry.get("basis") or "shared_dimensions_case_weighted"),
                "dimensions": [str(dim) for dim in entry.get("dimensions") or []],
                "with_skill": round(treatment, 4),
                control_key: round(control, 4),
                "lift": round(lift, 4),
                "n_cases": _as_nonnegative_int(entry.get("n_cases")),
                "expected_cases": _as_nonnegative_int(entry.get("expected_cases")),
                "partial": entry.get("partial") is True,
            }
    shared = [
        dim_id
        for dim_id in (*_DIMENSION_IDS, "overall")
        if _finite_float(with_dimensions.get(dim_id)) is not None
        and _finite_float(control_dimensions.get(dim_id)) is not None
    ]
    if not shared:
        return None
    treatment = _mean([with_dimensions[dim_id] for dim_id in shared])
    control = _mean([control_dimensions[dim_id] for dim_id in shared])
    if treatment is None or control is None:
        return None
    return {
        "basis": "shared_dimensions_arm_means",
        "dimensions": shared,
        "with_skill": treatment,
        control_key: control,
        "lift": round(treatment - control, 4),
        "n_cases": None,
        "expected_cases": None,
        "partial": False,
    }


def _lift_note(
    info: dict[str, Any],
    with_skill: float | None,
    lift: float | None,
    interval: object,
    sum_of_parts_baseline: bool,
) -> str | None:
    """Why an agent with a with-skill score shows no Skill Lift, in a few words; ``None`` otherwise.

    A baseline that ran and failed is not "no baseline", and the partial
    interval over the cases both arms scored (*interval*) is not final.
    """
    if lift is not None or with_skill is None or sum_of_parts_baseline:
        return None
    notes: list[str] = []
    conditions = info.get("conditions")
    baseline = conditions.get("without_skill") if isinstance(conditions, dict) else None
    if isinstance(baseline, dict) and baseline.get("execution_status") in {"failed", "unknown"}:
        notes.append("baseline did not complete")
    if isinstance(interval, dict) and interval.get("partial") is True:
        paired = _as_nonnegative_int(interval.get("n_cases"))
        notes.append(f"partial: {paired} of {_as_nonnegative_int(interval.get('expected_cases'))} cases, not final")
    return "; ".join(notes) or "no baseline"


def _agent_statistics(info: dict[str, Any], *, sum_of_parts_baseline: bool) -> dict[str, Any]:
    """The collector's report-only statistics; a legacy members interval is filed under Integration."""
    fields = {field: info[field] for field in _statistics_fields() if isinstance(info.get(field), dict)}
    uncertainty = fields.get("lift_uncertainty")
    if sum_of_parts_baseline and isinstance(uncertainty, dict) and uncertainty.get("effectiveness"):
        # Older runs filed the plugin-vs-member-skills interval as "effectiveness".
        fields["lift_uncertainty"] = {
            **uncertainty,
            "effectiveness": None,
            "integration": uncertainty.get("integration") or uncertainty["effectiveness"],
        }
    return fields


def _baseline_is_sum_of_parts(run_config: dict[str, Any] | None, plugin_provenance: dict[str, Any] | None) -> bool:
    """Whether the only baseline arm staged the member skills (legacy ``--lift-mode integration``)."""
    if not _is_plugin_target(run_config) and not plugin_provenance:
        return False
    return _plugin_lift_modes(run_config, plugin_provenance)["effective"] == "integration"


def _build_agent(
    name: str,
    info: dict[str, Any],
    with_metrics: list[str],
    baseline_metrics: list[str],
    model: str | None,
    *,
    sum_of_parts_baseline: bool = False,
) -> dict[str, Any]:
    """Assemble one agent's scores, lifts and statistics for the report payload.

    With ``sum_of_parts_baseline`` (legacy ``--lift-mode integration``) the
    baseline arm staged the plugin's member skills, so its comparison is the
    Integration lift: ``lift`` (plugin vs. no plugin) stays ``None`` and the
    member-skills score and lift fill ``sum_of_parts`` and ``integration_lift``.
    """
    with_scores = info.get("with_skill") or {}
    without_scores = info.get("without_skill") or {}
    lift_data = info.get("lift") or {}
    raw_uncertainty = info.get("lift_uncertainty")
    uncertainty = raw_uncertainty if isinstance(raw_uncertainty, dict) else {}
    with_quality_available = _condition_quality_available(info, "with_skill")
    baseline_quality_available = _condition_quality_available(info, "without_skill")
    if not with_quality_available:
        with_scores = {}
    if not baseline_quality_available:
        without_scores = {}
        # A lift needs a usable baseline arm; an engine lift from a failed arm is not one.
        lift_data = {}

    evaluators = _build_evaluators(with_metrics, with_scores, without_scores, lift_data)
    dimensions = _build_dimensions(
        with_scores,
        without_scores,
        info.get("dimensions_with_skill") or {},
        info.get("dimensions_without_skill") or {},
        baseline_not_applicable=_arm_not_applicable(info, "without_skill") if baseline_quality_available else [],
    )
    with_not_applicable = _arm_not_applicable(info, "with_skill") if with_quality_available else []
    not_applicable_evaluators = _not_applicable_evaluators(with_metrics, with_scores, with_not_applicable)
    not_applicable_dimensions = _not_applicable_dimensions(dimensions, with_not_applicable)
    overall_ws = _mean([d["with_skill"] for d in dimensions])
    with_dimensions = {str(d["id"]): d["with_skill"] for d in dimensions}
    baseline_dimensions = {str(d["id"]): d["baseline"] for d in dimensions}
    with_mixed_contract = with_quality_available and _condition_has_mixed_metric_contracts(
        info,
        flag="mixed_metric_contracts_with_skill",
        rewards="rewards",
    )
    baseline_mixed_contract = baseline_quality_available and _condition_has_mixed_metric_contracts(
        info,
        flag="mixed_metric_contracts_without_skill",
        rewards="rewards_baseline",
    )
    if with_mixed_contract or (overall_ws is None and not with_metrics and with_quality_available):
        # Custom-only runs have no dimension mean. For mixed condition
        # contracts, the dimension mean covers only standard rows and can
        # overstate Harbor's logical attempt score used by pass@k. In both
        # cases, prefer the collector-owned logical overall.
        overall_ws = _arm_logical_overall(info, "with_skill")
        with_dimensions = {"overall": overall_ws}
    if baseline_mixed_contract or (
        not baseline_metrics and baseline_quality_available and _mean(list(baseline_dimensions.values())) is None
    ):
        baseline_dimensions = {"overall": _arm_logical_overall(info, "without_skill")}
    # A logical overall on one side compares only with the other side's overall.
    if set(with_dimensions) == {"overall"} and "overall" not in baseline_dimensions:
        baseline_dimensions = {"overall": _mean(list(baseline_dimensions.values()))}
    elif set(baseline_dimensions) == {"overall"} and "overall" not in with_dimensions:
        with_dimensions = {"overall": overall_ws}
    # One lift on one basis: the dimensions both arms scored, case-weighted when
    # the collector's paired statistics exist. The headline equals the interval
    # estimate, and ``baseline`` is the no-skill score on that same basis.
    effectiveness_basis = (
        _comparison_basis(uncertainty.get("effectiveness"), with_dimensions, baseline_dimensions, "baseline")
        if with_quality_available and baseline_quality_available and overall_ws is not None
        else None
    )
    members_basis = None
    if sum_of_parts_baseline:
        # Older runs filed the plugin-vs-members interval under "effectiveness".
        members_entry = uncertainty.get("integration") or uncertainty.get("effectiveness")
        members_basis = (
            _comparison_basis(members_entry, with_dimensions, baseline_dimensions, "sum_of_parts")
            if with_quality_available and baseline_quality_available and overall_ws is not None
            else None
        )
        effectiveness_basis = None
    # A partial comparison keeps its interval (marked partial) but no final-looking headline.
    overall_lift = effectiveness_basis["lift"] if effectiveness_basis and not effectiveness_basis["partial"] else None
    # Without a comparison (the with-skill arm is unavailable) the baseline arm's
    # own score still shows, so reports can tell the two failures apart.
    overall_bl = (
        effectiveness_basis["baseline"]
        if effectiveness_basis
        else (_mean(list(baseline_dimensions.values())) if baseline_quality_available else None)
    )

    sum_of_parts_quality_available = _condition_quality_available(info, "sum_of_parts")
    sum_of_parts_scores = (info.get("sum_of_parts") or {}) if sum_of_parts_quality_available else {}
    sum_of_parts_by_dimension = _dimension_scores(sum_of_parts_scores, info.get("dimensions_sum_of_parts") or {})
    sum_of_parts_overall = _mean(list(sum_of_parts_by_dimension.values()))
    if sum_of_parts_overall is None and not with_metrics and sum_of_parts_quality_available:
        sum_of_parts_overall = _arm_logical_overall(info, "sum_of_parts")
        sum_of_parts_by_dimension = {"overall": sum_of_parts_overall}
    if set(with_dimensions) == {"overall"} and "overall" not in sum_of_parts_by_dimension:
        sum_of_parts_by_dimension = {"overall": sum_of_parts_overall}
    integration_basis = (
        _comparison_basis(uncertainty.get("integration"), with_dimensions, sum_of_parts_by_dimension, "sum_of_parts")
        if with_quality_available and sum_of_parts_quality_available and overall_ws is not None
        else None
    )
    if sum_of_parts_baseline:
        integration_basis = members_basis
        sum_of_parts_overall = overall_bl
    if integration_basis is not None:
        sum_of_parts_overall = integration_basis["sum_of_parts"]
        if sum_of_parts_baseline:
            overall_bl = integration_basis["sum_of_parts"]
    integration_lift = integration_basis["lift"] if integration_basis else None

    trials = _normalize_trials(info.get("rewards") or [], with_metrics)
    baseline_trials = _normalize_trials(info.get("rewards_baseline") or [], baseline_metrics)
    _attach_baseline_pairs(trials, baseline_trials, with_metrics)

    execution_errors = (
        [str(error) for error in info.get("execution_errors", [])]
        if isinstance(info.get("execution_errors"), list)
        else []
    )
    execution_error_details = _aggregate_execution_error_details({name: info}, len(execution_errors))

    return {
        "name": name,
        "model": model,
        "execution_status": (
            info.get("execution_status")
            if info.get("execution_status") in {"succeeded", "failed", "skipped", "unknown"}
            else "unknown"
        ),
        "execution_errors": execution_errors,
        **execution_error_details,
        "expected_attempts": _as_nonnegative_int(info.get("expected_attempts")),
        "scored_attempts": _as_nonnegative_int(info.get("scored_attempts")),
        "conditions": info.get("conditions", {}) if isinstance(info.get("conditions"), dict) else {},
        "evaluators": evaluators,
        "evaluator_cards": [],
        "not_applicable_evaluators": not_applicable_evaluators,
        "dimensions": dimensions,
        "not_applicable_dimensions": not_applicable_dimensions,
        "with_skill": overall_ws,
        "baseline": overall_bl,
        "lift": overall_lift,
        "sum_of_parts": sum_of_parts_overall,
        "integration_lift": integration_lift,
        "lift_note": _lift_note(
            info, overall_ws, overall_lift, uncertainty.get("effectiveness"), sum_of_parts_baseline
        ),
        # The comparable arm scores behind each lift (same dimensions, same cases).
        "lift_basis": {"effectiveness": effectiveness_basis, "integration": integration_basis},
        "integration_completeness": info.get("integration_completeness") or {},
        **_agent_statistics(info, sum_of_parts_baseline=sum_of_parts_baseline),
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


# Per arm: the engine's overall-score field and the flag that says the arm's rewards list is complete.
_ARM_OVERALL_FIELDS = {
    "with_skill": ("overall_with_skill", "rewards_complete"),
    "without_skill": ("overall_without_skill", "rewards_baseline_complete"),
    "sum_of_parts": ("overall_sum_of_parts", "rewards_sum_of_parts_complete"),
}


def _arm_logical_overall(info: dict[str, Any], arm: str) -> float | None:
    """Return the collector's own overall score for an arm, else its mean logical-trial reward.

    The reward mean is used only when the arm's rewards list is complete.
    """
    overall_field, complete_field = _ARM_OVERALL_FIELDS[arm]
    overall = _finite_float(info.get(overall_field))
    if overall is None and info.get(complete_field) is not False:
        overall = _logical_reward_mean(info.get(_ARM_REWARDS_FIELDS[arm]), "overall")
    return overall


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
    custom_with_skill = info.get("custom_with_skill")
    custom_without_skill = info.get("custom_without_skill")
    custom_lift = info.get("custom_lift")
    agent_payload["evaluator_cards"] = _evaluator_cards(
        agent_payload.get("evaluators", {}),
        rewards=info.get("rewards") or [],
        custom_with_skill=custom_with_skill if isinstance(custom_with_skill, dict) else {},
        custom_without_skill=custom_without_skill if isinstance(custom_without_skill, dict) else {},
        custom_lift=custom_lift if isinstance(custom_lift, dict) else {},
        report_budget=report_budget,
    )


def _statistics_fields() -> tuple[str, ...]:
    """The report-only statistics blocks the collector stores for an agent (``STATISTICS_BLOCKS``), copied as is.

    Integration completeness is left out: an agent's payload sets it on its own,
    and the payload reports it in the Integration block, not at the top level.
    """
    from skillevaluator.tier3.harbor.stats import STATISTICS_BLOCKS

    return tuple(block for block in STATISTICS_BLOCKS if block != "integration_completeness")


def _arm_not_applicable(info: dict[str, Any], condition: str) -> list[str]:
    """Judged metrics an arm recorded as not applicable in every trial."""
    from skillevaluator.tier3.harbor.metrics import not_applicable_list

    return not_applicable_list(info.get(f"not_applicable_{condition}"))


def _not_applicable_evaluators(
    metrics: list[str],
    with_scores: dict[str, Any],
    not_applicable: list[str],
) -> list[dict[str, str]]:
    """Evaluators with no with-skill score because no eval case gave them a reference."""
    from skillevaluator.tier3.harbor.metrics import METRIC_DISPLAY, NOT_APPLICABLE_REASONS

    return [
        {
            "id": metric,
            "label": METRIC_DISPLAY.get(metric, metric.replace("_", " ").title()),
            "reason": NOT_APPLICABLE_REASONS.get(metric, "Not applicable to these eval cases"),
        }
        for metric in metrics
        if metric in not_applicable and _finite_float(with_scores.get(metric)) is None
    ]


def _not_applicable_dimensions(
    dimensions: list[dict[str, Any]],
    not_applicable: list[str],
) -> list[dict[str, str]]:
    """Dimensions left unscored because every source evaluator was not applicable."""
    from skillevaluator.tier3.harbor.metrics import METRIC_DISPLAY, dimension_is_not_applicable

    scored = {dimension.get("id") for dimension in dimensions}
    out: list[dict[str, str]] = []
    for dim_id in _DIMENSION_IDS:
        if dim_id in scored or not dimension_is_not_applicable(dim_id, not_applicable):
            continue
        labels = ", ".join(METRIC_DISPLAY.get(metric, metric) for metric in DIMENSION_MAPPING[dim_id]["evaluators"])
        out.append(
            {
                "id": dim_id,
                "reason": f"Not applicable: {labels} had nothing to judge against in any eval case",
            }
        )
    return out


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
        # ``lift`` stays None when there is no baseline to compare with: a 0.0
        # would read as "measured, no change" in every report.
        evaluators[metric] = {"with_skill": ws, "baseline": bl, "lift": lift}
    return evaluators


def _arm_dimension_score(scores: dict[str, Any], precomputed: dict[str, Any], dim_id: str) -> float | None:
    """Return one arm's score for a dimension: the engine's, else the weighted evaluator scores."""
    from skillevaluator.tier3.harbor.metrics import weighted_dimension_score

    score = _precomputed_score(precomputed, dim_id)
    return score if score is not None else weighted_dimension_score(scores.get, DIMENSION_MAPPING[dim_id])


def _dimension_scores(scores: dict[str, Any], precomputed: dict[str, Any]) -> dict[str, float]:
    """Return one arm's scored dimensions, rounded as the dimension rows round them."""
    return {
        dim_id: round(score, 4)
        for dim_id in _DIMENSION_IDS
        if (score := _arm_dimension_score(scores, precomputed, dim_id)) is not None
    }


def _build_dimensions(
    with_scores: dict[str, Any],
    without_scores: dict[str, Any],
    precomputed_with: dict[str, Any],
    precomputed_without: dict[str, Any],
    *,
    baseline_not_applicable: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Score each dimension in both arms.

    *baseline_not_applicable* lists the metrics the baseline recorded as not
    applicable. A dimension built only from them (Discoverability and
    Efficiency in an arm without the skill) is flagged
    ``baseline_not_applicable``, and its reasoning says so instead of "no
    baseline run": the baseline ran, it just has nothing to score there.
    """
    from skillevaluator.tier3.harbor.metrics import dimension_is_not_applicable

    dimensions: list[dict[str, Any]] = []
    for dim_id in _DIMENSION_IDS:
        cfg = DIMENSION_MAPPING[dim_id]
        ws = _arm_dimension_score(with_scores, precomputed_with, dim_id)
        bl = _arm_dimension_score(without_scores, precomputed_without, dim_id)
        if ws is None and bl is None:
            continue
        lift = round(ws - bl, 4) if ws is not None and bl is not None else None
        bl_not_applicable = bl is None and dimension_is_not_applicable(dim_id, baseline_not_applicable or [])
        entry = precomputed_with.get(dim_id) if isinstance(precomputed_with.get(dim_id), dict) else {}
        # Signals (the evaluators that actually fed this dimension) populate the
        # "Signals" column; reasoning bullets and a deterministic verdict fill
        # the "Reasoning"/"Verdict" columns when the engine left them blank.
        signals = _dimension_signals(entry, with_scores, cfg)
        explanation = entry.get("explanation")
        reasoning_bullets = entry.get("reasoning_bullets")
        if not reasoning_bullets and not explanation:
            reasoning_bullets, explanation = _deterministic_reasoning(
                ws, bl, lift, signals, with_scores, baseline_not_applicable=bl_not_applicable
            )
        verdict = entry.get("verdict") or _deterministic_verdict(ws)
        dimension = {
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
        if bl_not_applicable:
            # The baseline ran but has nothing to score here, so reports say N/A, not "not run".
            dimension["baseline_not_applicable"] = True
        dimensions.append(dimension)
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
    *,
    baseline_not_applicable: bool = False,
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
        baseline_not_applicable=baseline_not_applicable,
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
            identity = evidence_ref_identity(raw)
            rendered = f"{source}{identity}" if source else identity
        else:
            continue
        if rendered and rendered not in refs:
            refs.append(rendered[:512])
        if len(refs) == 3:
            break
    return refs


def _custom_metric_value(reward: dict[str, Any], metric: str) -> float | None:
    """Read one custom metric without materializing every custom metric in a reward."""
    from skillevaluator.tier3.harbor.metrics import (
        RESERVED_METRIC_NAMES,
        custom_metric_name_is_publishable,
        score_value,
    )

    if metric in RESERVED_METRIC_NAMES or not custom_metric_name_is_publishable(metric):
        return None

    numeric: float | None = None
    sources = (reward.get("custom_metrics"), reward.get("metrics"), reward)
    for source in sources:
        if not isinstance(source, dict) or metric not in source:
            continue
        value = source.get(metric)
        if isinstance(value, dict):
            value = value.get("score")
        candidate = score_value(value)
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
    from skillevaluator.tier3.harbor.metrics import (
        RESERVED_METRIC_NAMES,
        custom_metric_name_is_publishable,
        score_value,
    )

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
                and custom_metric_name_is_publishable(name)
                and name not in excluded
                and name not in seen
                and score_value(value) is not None
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
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, metric_set_for_reward

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
        if metric in DEFAULT_METRICS and metric not in metric_set_for_reward(reward)[1]:
            continue
        custom_details = reward.get("custom_details")
        custom_detail_is_authoritative = (
            metric not in DEFAULT_METRICS and isinstance(custom_details, dict) and metric in custom_details
        )
        if custom_detail_is_authoritative:
            detail = custom_details[metric]
        else:
            details = reward.get("details")
            detail = details.get(metric) if isinstance(details, dict) else None
            if not isinstance(detail, dict):
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
    from skillevaluator.tier3.harbor.metrics import score_value

    value = configured.get(metric)
    if isinstance(value, dict):
        value = value.get("score")
    configured_score = score_value(value)
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
    from skillevaluator.tier3.harbor.metrics import (
        RESERVED_METRIC_NAMES,
        custom_metric_name_is_publishable,
    )

    if limit <= 0:
        report_budget.omit("evaluator_cards", len(custom_with_skill))
        report_budget.omit("custom_metric_discovery_trials", len(rewards))
        return {}

    candidates: dict[str, None] = {}
    for raw_name in islice(iter(custom_with_skill), _MAX_CUSTOM_METRIC_NAME_VISITS_PER_REWARD):
        name = str(raw_name)
        if (
            name not in RESERVED_METRIC_NAMES
            and custom_metric_name_is_publishable(name)
            and name not in excluded
            and name not in candidates
        ):
            candidates[name] = None
            if len(candidates) > limit:
                break

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
        # The dimension verdict's thresholds, so a card never says FAIL beside a PASS dimension row.
        "status": (
            "pass"
            if ws >= DIMENSION_VERDICT_PASS_THRESHOLD
            else ("warn" if ws >= DIMENSION_VERDICT_NEUTRAL_THRESHOLD else "fail")
        ),
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
    from skillevaluator.tier3.harbor.metrics import overall_score
    from skillevaluator.tier3.harbor.report_data import logical_trial_reward_groups

    cases: list[dict[str, Any]] = []
    rewards = [reward for reward in (info.get("rewards") or []) if isinstance(reward, dict)]
    for group in logical_trial_reward_groups(rewards):
        if not group:
            continue
        reward = group[0]
        group_is_consistent = _logical_group_entry_identity_is_consistent(group)
        cases.append(
            {
                "entry_id": reward.get("entry_id"),
                "overall": _complete_mean([overall_score(item) for item in group]) if group_is_consistent else None,
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
        NOT_APPLICABLE_REASONS,
        metric_is_not_applicable,
        metric_set_for_reward,
        metric_set_for_rewards,
        metric_value,
        overall_score,
    )
    from skillevaluator.tier3.harbor.report_data import logical_trial_reward_groups

    out: list[dict[str, Any]] = []
    reward_groups = logical_trial_reward_groups([reward for reward in rewards if isinstance(reward, dict)])
    for group in reward_groups:
        if not group:
            continue
        reward = group[0]
        is_multi_row = len(group) > 1
        group_is_consistent = _logical_group_entry_identity_is_consistent(group)
        declared_metric_set = reward.get("metric_set") or reward.get("metric_set_version")
        standard_metric_sets = {DEFAULT_METRIC_SET, LEGACY_METRIC_SET}
        metric_set, standard_metrics = metric_set_for_rewards(group)
        declared_metric_sets = [
            str(value) for item in group if (value := item.get("metric_set") or item.get("metric_set_version"))
        ]
        is_declared_custom = len(declared_metric_sets) == len(group) and all(
            value not in standard_metric_sets for value in declared_metric_sets
        )
        standard_rows = [(item, metric_set_for_reward(item)[1]) for item in group]
        scores: dict[str, float] = {}
        for metric in metrics:
            if is_declared_custom and metric in {"skill_execution", "skill_routing"}:
                continue
            value = _complete_mean(
                [metric_value(item, metric) for item, item_metrics in standard_rows if metric in item_metrics]
            )
            if value is not None:
                scores[metric] = value
        trial: dict[str, Any] = {
            "trial_id": reward.get("trial_id"),
            "entry_id": reward.get("entry_id"),
            "scores": scores,
            "overall": _complete_mean([overall_score(item) for item in group]) if group_is_consistent else None,
        }
        not_applicable = {
            metric: NOT_APPLICABLE_REASONS[metric]
            for metric in metrics
            if metric in NOT_APPLICABLE_REASONS and metric_is_not_applicable(reward, metric)
        }
        if not_applicable:
            trial["not_applicable"] = not_applicable
        traj = reward.get("_traj")
        if not is_multi_row and isinstance(traj, dict):
            steps = _token_counter(traj.get("steps"))
            if steps is not None:
                trial["steps"] = steps
            prompt_tokens = _token_counter(traj.get("prompt_tokens"))
            completion_tokens = _token_counter(traj.get("completion_tokens"))
            cached_tokens = _token_counter(traj.get("cached_tokens"))
            if prompt_tokens is not None and completion_tokens is not None:
                trial["tokens"] = {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                }
                if cached_tokens is not None:
                    trial["tokens"]["cached"] = cached_tokens
        warnings = list(
            dict.fromkeys(
                str(warning) for item in group if isinstance(item.get("warnings"), list) for warning in item["warnings"]
            )
        )
        if warnings:
            trial["warnings"] = warnings
        if not is_multi_row and reward.get("error_recovery"):
            trial["error_recovery"] = reward["error_recovery"]
        is_standard_reward = (
            (not declared_metric_set or str(declared_metric_set) in standard_metric_sets)
            and metric_set in standard_metric_sets
            and "skill_execution" in standard_metrics
            and metric_value(reward, "skill_execution") is not None
        )
        if not is_multi_row and is_standard_reward and reward.get("invocation_evidence_source") == "trajectory":
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


def _plugin_incompleteness_conclusion(plugin_provenance: dict[str, Any]) -> dict[str, str]:
    """Build the leading deterministic conclusion for a partial plugin run.

    The message states the same reason as every report (see :func:`incomplete_reason`);
    the title names the main cause.
    """
    from skillevaluator.reporting.plugin_sections import text

    completeness = _plugin_completeness(plugin_provenance)
    if completeness["sidecar_error"]:
        title = "plugin provenance unreadable"
        consequence = "The score is not a full evaluation and must not be read as a pass."
    elif completeness["run_notes"] and not completeness["deferred"]:
        title = (
            "the run did not complete"
            if text(plugin_provenance.get("execution_incomplete"))
            else "native plugin load not confirmed"
        )
        consequence = (
            "No declared component was deferred, but the score is not a full evaluation and must not be read as a pass."
        )
    else:
        counts = completeness["counts"]
        title = "unresolved dependencies"
        consequence = (
            f"The score reflects only the resolved components ({counts['skills_resolved']} skill(s), "
            f"{counts['rules_resolved']} rule(s)) and must not be read as a full pass."
        )
    return {
        "severity": "fail",
        "title": f"Evaluation INCOMPLETE - {title}",
        "message": f"This plugin run is INCOMPLETE: {completeness['reason']}. {consequence}",
    }


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
    except (TypeError, ValueError, OverflowError):
        return 0.50
    return numeric if math.isfinite(numeric) else 0.50


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


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


def _integration_interval_verdict(uncertainty: dict[str, Any]) -> tuple[str, str | None]:
    """Classify the Integration lift by where its whole interval lies relative to the +/-0.05 band.

    "real" needs the interval to clear +0.05 and "negative" to stay below
    -0.05. "cosmetic" is an equivalence claim: the whole interval lies inside
    the band, so a tight tie around zero is cosmetic, not inconclusive.
    Otherwise the interval cannot tell the bands apart and the reason says why.
    """
    low = _finite_float(uncertainty.get("ci_low"))
    high = _finite_float(uncertainty.get("ci_high"))
    ci_text = _ci_text(uncertainty) or "interval"
    if low is None or high is None:
        return INTEGRATION_VERDICT_INCONCLUSIVE, "The Integration lift has no interval to classify it."
    if low >= _INTEGRATION_REAL_THRESHOLD:
        return INTEGRATION_VERDICT_REAL, None
    if high <= _INTEGRATION_NEGATIVE_THRESHOLD:
        return INTEGRATION_VERDICT_NEGATIVE, None
    if low > _INTEGRATION_NEGATIVE_THRESHOLD and high < _INTEGRATION_REAL_THRESHOLD:
        return INTEGRATION_VERDICT_COSMETIC, None
    band = (
        f"{_INTEGRATION_REAL_THRESHOLD:+.2f}"
        if high >= _INTEGRATION_REAL_THRESHOLD
        else f"{_INTEGRATION_NEGATIVE_THRESHOLD:+.2f}"
    )
    if low <= 0.0 <= high:
        return (
            INTEGRATION_VERDICT_INCONCLUSIVE,
            f"The paired case bootstrap {ci_text} for the Integration lift includes zero and reaches past {band}, "
            "so it cannot tell a real or negative effect from no effect.",
        )
    effect = "a real" if high >= _INTEGRATION_REAL_THRESHOLD else "a negative"
    return (
        INTEGRATION_VERDICT_INCONCLUSIVE,
        f"The paired case bootstrap {ci_text} for the Integration lift crosses the {band} band edge, "
        f"so it cannot tell {effect} effect from cosmetic bundling.",
    )


def _lift_band(
    agent: dict[str, Any],
    run_config: dict[str, Any] | None,
    plugin_provenance: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Place the headline Skill Lift (with versus without) in its PASS/NEUTRAL/FAIL band.

    A lift at or below the FAIL threshold whose paired-case interval lies wholly
    below zero is a confirmed regression: it adds a warning and fails
    ``validate --block-on-agent-eval``. An interval over fewer than
    ``LIFT_CI_MIN_PAIRED_CASES`` paired cases (precision ``insufficient``) is
    too unstable to confirm one, as for the Integration lift. The dimension
    verdict is unchanged. An
    Integration-only plugin run has no no-plugin arm (its lift compares the
    plugin with its own parts, which stays advisory), so it gets no band.
    """
    plugin_run = _is_plugin_target(run_config) or bool(plugin_provenance)
    if plugin_run and _plugin_lift_modes(run_config, plugin_provenance)["effective"] == "integration":
        return None
    lift = _finite_float(agent.get("lift"))
    if lift is None:
        return None
    entry = _lift_uncertainty_entry(agent, "effectiveness") or {}
    ci_low = _finite_float(entry.get("ci_low"))
    ci_high = _finite_float(entry.get("ci_high"))
    has_interval = ci_low is not None and ci_high is not None
    precision = entry.get("precision") if has_interval and isinstance(entry.get("precision"), str) else None
    verdict = _verdict_from_lift(lift)
    return {
        "verdict": verdict,
        "lift": round(lift, 4),
        "ci_low": ci_low if has_interval else None,
        "ci_high": ci_high if has_interval else None,
        "confidence": (_finite_float(entry.get("confidence")) or 0.95) if has_interval else None,
        "precision": precision,
        "pass_threshold": TIER3_LIFT_PASS_THRESHOLD,
        "fail_threshold": TIER3_LIFT_FAIL_THRESHOLD,
        "regression_confirmed": bool(
            verdict == VERDICT_FAIL and has_interval and ci_high < 0 and precision != "insufficient"
        ),
    }


def _lift_band_text(band: dict[str, Any]) -> str:
    """``-0.20 (95% CI [-0.20, -0.19])``, or the lift alone when no interval was computed."""
    text = f"{_as_float(band.get('lift')):+.2f}"
    low, high = _finite_float(band.get("ci_low")), _finite_float(band.get("ci_high"))
    if low is not None and high is not None:
        confidence = _finite_float(band.get("confidence")) or 0.95
        text += f" ({confidence:.0%} CI [{low:+.2f}, {high:+.2f}])"
    return text


def _lift_band_conclusion(band: dict[str, Any] | None) -> dict[str, str] | None:
    """Warn when the Skill Lift is in the FAIL band; say whether the regression is confirmed."""
    if not band or band.get("verdict") != VERDICT_FAIL:
        return None
    threshold = f"{_finite_float(band.get('fail_threshold')) or TIER3_LIFT_FAIL_THRESHOLD:+.2f}"
    if band.get("regression_confirmed"):
        return {
            "severity": "fail",
            "title": "Skill Lift regression",
            "message": (
                f"Skill Lift {_lift_band_text(band)} is in the FAIL band (at or below {threshold}) and the whole "
                "interval is below zero: results were worse with it than without it. The dimension verdict is "
                "unchanged; validate --block-on-agent-eval fails on this regression."
            ),
        }
    if band.get("ci_low") is None or band.get("ci_high") is None:
        reason = "no paired-case interval was computed"
    elif band.get("precision") == "insufficient":
        reason = f"its interval rests on fewer than {LIFT_CI_MIN_PAIRED_CASES} paired cases"
    else:
        reason = "its interval includes zero"
    return {
        "severity": "warn",
        "title": "Negative Skill Lift",
        "message": (
            f"Skill Lift {_lift_band_text(band)} is in the FAIL band (at or below {threshold}), but {reason}, "
            "so the regression is not confirmed and does not gate. Add cases or attempts to confirm it."
        ),
    }


def _integration_verdict(lift: float | None, *, complete: bool) -> str:
    numeric = _finite_float(lift)
    if not complete or numeric is None:
        return INTEGRATION_VERDICT_INCONCLUSIVE
    if numeric >= _INTEGRATION_REAL_THRESHOLD:
        return INTEGRATION_VERDICT_REAL
    if numeric <= _INTEGRATION_NEGATIVE_THRESHOLD:
        return INTEGRATION_VERDICT_NEGATIVE
    return INTEGRATION_VERDICT_COSMETIC


def _is_plugin_target(run_config: dict[str, Any] | None) -> bool:
    """Return whether ``run_config`` marks a plugin evaluation."""
    if not isinstance(run_config, dict):
        return False
    target = run_config.get("eval_target")
    return isinstance(target, dict) and target.get("kind") == "plugin"


def _lift_mode_value(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if text in _LIFT_MODES else None


def _plugin_lift_modes(
    run_config: dict[str, Any] | None,
    plugin_provenance: dict[str, Any] | None,
) -> dict[str, str | None]:
    """Resolve the requested and effective plugin lift modes.

    The CLI records both in the plugin provenance and the runner records them in
    ``run_config["lift_mode"]``. The runner's effective mode wins because it
    reflects the arms that actually ran. Older runs carry neither, so the
    effective mode falls back to the configured arms. A measured Integration
    mode was necessarily requested; an effectiveness run with no record leaves
    the request unknown (``None``).
    """
    provenance = plugin_provenance if isinstance(plugin_provenance, dict) else {}
    config = run_config if isinstance(run_config, dict) else {}
    recorded = config.get("lift_mode") if isinstance(config.get("lift_mode"), dict) else {}
    effective = _lift_mode_value(recorded.get("effective")) or _lift_mode_value(provenance.get("effective_lift_mode"))
    if effective is None and _is_plugin_target(config):
        workspace = config.get("skill_workspace") if isinstance(config.get("skill_workspace"), dict) else {}
        if workspace.get("sum_of_parts_arm"):
            effective = "both"
        elif workspace.get("baseline_includes_workspace_skills"):
            effective = "integration"
        else:
            effective = "effectiveness"
    requested = _lift_mode_value(provenance.get("requested_lift_mode")) or _lift_mode_value(recorded.get("requested"))
    if requested is None and effective in _INTEGRATION_LIFT_MODES:
        requested = effective
    skip_reason = str(provenance.get("integration_skip_reason") or recorded.get("integration_skip_reason") or "")
    return {"requested": requested, "effective": effective, "integration_skip_reason": skip_reason.strip() or None}


def _lift_uncertainty_entry(agent: dict[str, Any], comparison: str) -> dict[str, Any] | None:
    uncertainty = agent.get("lift_uncertainty")
    entry = uncertainty.get(comparison) if isinstance(uncertainty, dict) else None
    return entry if isinstance(entry, dict) else None


def _ci_text(entry: dict[str, Any]) -> str | None:
    low = _finite_float(entry.get("ci_low"))
    high = _finite_float(entry.get("ci_high"))
    if low is None or high is None:
        return None
    confidence = _finite_float(entry.get("confidence")) or 0.95
    return f"{confidence:.0%} CI [{low:+.2f}, {high:+.2f}]"


def _effectiveness_uncertainty_conclusion(best: dict[str, Any]) -> dict[str, str] | None:
    """Warn when the Skill Lift interval cannot rule out zero (gating is unchanged)."""
    entry = _lift_uncertainty_entry(best, "effectiveness")
    if entry is None or entry.get("ci_includes_zero") is not True:
        return None
    ci_text = _ci_text(entry)
    if ci_text is None:
        return None
    precision = str(entry.get("precision") or "unknown")
    n_cases = _as_nonnegative_int(entry.get("n_cases"))
    return {
        "severity": "warn",
        "title": "Skill Lift not distinguishable from zero",
        "message": (
            f"The paired case bootstrap {ci_text} for Skill Lift includes zero "
            f"({n_cases} paired case(s), precision {precision}). The lift band is unchanged; "
            "add cases or attempts before relying on the direction of the lift."
        ),
    }


_ARM_NAMES = {
    "with_skill": "with-plugin arm",
    "with_plugin": "with-plugin arm",
    "without_skill": "no-plugin arm",
    "sum_of_parts": "member-skills (sum-of-parts) arm",
}


# Legacy ``--lift-mode integration``: the only baseline arm stages the member skills.
_LEGACY_ARM_NAMES = {**_ARM_NAMES, "without_skill": "member-skills baseline arm"}


def _arm_name(arm: object, names: dict[str, str] | None = None) -> str:
    return (names or _ARM_NAMES).get(str(arm), str(arm).replace("_", " "))


def _paired_cases_note(uncertainty: dict[str, Any] | None) -> str:
    """``"8 of 9 cases"`` for a partial interval, or ``""``."""
    if not isinstance(uncertainty, dict) or uncertainty.get("partial") is not True:
        return ""
    paired = _as_nonnegative_int(uncertainty.get("n_cases"))
    expected = _as_nonnegative_int(uncertainty.get("expected_cases"))
    return f"{paired} of {expected} cases" if expected else f"{paired} cases"


def _integration_completeness_reason(
    completeness: dict[str, Any] | None,
    uncertainty: dict[str, Any] | None = None,
    *,
    names: dict[str, str] | None = None,
) -> str:
    """Explain why the per-case Integration completeness check failed.

    Names the arm that really failed, the missing cases and the attempt
    shortfall, and how many paired cases the lift still used.
    """
    base = "The plugin and member-skills arms did not cover the same expected cases with the configured attempts"
    details: list[str] = []
    completeness = completeness if isinstance(completeness, dict) else {}
    failed_arms = [str(arm) for arm in completeness.get("failed_arms") or []]
    if not failed_arms and isinstance(uncertainty, dict):
        failed_arms = [str(arm) for arm in uncertainty.get("failed_arms") or []]
    if failed_arms:
        details.append("did not complete: " + ", ".join(_arm_name(arm, names) for arm in failed_arms[:4]))
    missing = [str(case) for case in completeness.get("missing_cases") or []]
    if missing:
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        details.append("missing case(s): " + ", ".join(missing[:5]) + suffix)
    shortfall = [row for row in completeness.get("attempt_shortfall") or [] if isinstance(row, dict)]
    if shortfall:
        rows = [
            f"{row.get('case')} ({_arm_name(row.get('arm'), names)}) {row.get('observed')}/{row.get('expected')}"
            for row in shortfall[:5]
        ]
        suffix = f" (+{len(shortfall) - 5} more)" if len(shortfall) > 5 else ""
        details.append("attempt shortfall: " + ", ".join(rows) + suffix)
    reason = base + (": " + "; ".join(details) + "." if details else ".")
    pairs = _paired_cases_note(uncertainty)
    if pairs:
        reason += f" The lift shown uses the {pairs} both arms scored."
    return reason


def _integration_no_score_reason(
    agent: dict[str, Any],
    control_condition: str,
    *,
    names: dict[str, str] | None = None,
) -> str:
    """Name the arm that left the Integration comparison without a single scored pair."""
    conditions = agent.get("conditions") if isinstance(agent.get("conditions"), dict) else {}
    unscored = [
        condition
        for condition in ("with_skill", control_condition)
        if isinstance(conditions.get(condition), dict)
        and not _as_nonnegative_int(conditions[condition].get("scored_attempts"))
    ]
    if not unscored:
        return _INTEGRATION_REASON_NO_SCORE
    arms = " and the ".join(_arm_name(condition, names) for condition in unscored)
    errors = [str(error) for error in conditions[unscored[0]].get("execution_errors") or [] if str(error).strip()]
    cause = f" ({errors[0][:200]})" if errors else ""
    return f"The {arms} produced no usable scored trial{cause}, so no case could be paired for the Integration lift."


def _integration_block(
    *,
    components: list[str],
    with_plugin: float | None,
    sum_of_parts: float | None,
    lift: float | None,
    verdict: str,
    point_verdict: str | None,
    complete: bool,
    completeness: dict[str, Any] | None,
    reason: str | None,
    uncertainty: dict[str, Any] | None,
    lift_modes: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "schema_version": _INTEGRATION_SCHEMA_VERSION,
        "advisory": True,
        "report_only": True,
        # False when the block explains why no measurement exists.
        "measured": lift is not None,
        "basis": "compositional-lift-ablation",
        "baseline": "sum-of-parts",
        "components": list(dict.fromkeys(components)),
        "with_plugin": round(with_plugin, 4) if with_plugin is not None else None,
        "sum_of_parts": round(sum_of_parts, 4) if sum_of_parts is not None else None,
        "integration_lift": lift,
        "verdict": verdict,
        # The +/-0.05 band classification before any uncertainty or
        # completeness downgrade; None when no lift was measured.
        "point_verdict": point_verdict,
        "complete": complete,
        "completeness": completeness if isinstance(completeness, dict) else None,
        "lift_uncertainty": uncertainty,
        "interpretation": _INTEGRATION_INTERPRETATION[verdict],
        # Set whenever the verdict is inconclusive.
        "reason": reason,
        "lift_mode_requested": lift_modes.get("requested"),
        "lift_mode_effective": lift_modes.get("effective"),
    }


def _build_integration_report(
    best: dict[str, Any],
    run_config: dict[str, Any] | None,
    plugin_provenance: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the plugin-only, report-only compositional-lift result.

    Returns ``None`` only when Integration was neither measured nor requested.
    When ``--lift-mode integration|both`` was requested but no member-skills
    comparison exists (for example a ``both`` run that fell back to
    effectiveness for lack of cross-component evidence), an explicit
    ``inconclusive`` block with ``measured=False`` and a ``reason`` is returned
    instead of silently omitting the section.

    A measured lift is classified by the +/-0.05 bands (``point_verdict``).
    The advisory ``verdict`` uses the whole paired case bootstrap interval: real
    when it clears +0.05, negative when it stays below -0.05, cosmetic when it
    lies inside the band. It is ``inconclusive`` when the per-case completeness
    check fails, when fewer than the minimum paired cases exist, or when the
    interval crosses a band edge.
    """
    if not isinstance(run_config, dict) or not _is_plugin_target(run_config):
        return None
    lift_modes = _plugin_lift_modes(run_config, plugin_provenance)
    requested = lift_modes["requested"] in _INTEGRATION_LIFT_MODES

    def _unmeasured(reason: str, components: list[str]) -> dict[str, Any] | None:
        if not requested:
            return None
        return _integration_block(
            components=components,
            with_plugin=None,
            sum_of_parts=None,
            lift=None,
            verdict=INTEGRATION_VERDICT_INCONCLUSIVE,
            point_verdict=None,
            complete=False,
            completeness=None,
            reason=lift_modes["integration_skip_reason"] or reason,
            uncertainty=None,
            lift_modes=lift_modes,
        )

    workspace = run_config.get("skill_workspace")
    if not isinstance(workspace, dict):
        return _unmeasured(_INTEGRATION_REASON_NO_WORKSPACE, [])
    raw_components = workspace.get("staged_skills") or workspace.get("include") or []
    components = [Path(str(component)).name for component in raw_components if str(component).strip()]
    if not components:
        return _unmeasured(_INTEGRATION_REASON_NO_COMPONENTS, [])

    raw_bases = best.get("lift_basis")
    bases = raw_bases if isinstance(raw_bases, dict) else {}
    with_plugin = _finite_float(best.get("with_skill"))
    if workspace.get("sum_of_parts_arm"):
        control_condition = "sum_of_parts"
        basis = bases.get("integration") if isinstance(bases.get("integration"), dict) else None
        sum_of_parts = _finite_float(best.get("sum_of_parts"))
        completeness = best.get("integration_completeness")
        complete = bool(isinstance(completeness, dict) and completeness.get("complete"))
        uncertainty = _lift_uncertainty_entry(best, "integration")
    elif workspace.get("baseline_includes_workspace_skills"):
        # Legacy two-arm Integration: the single baseline is the member-skills arm.
        control_condition = "without_skill"
        basis = bases.get("integration") if isinstance(bases.get("integration"), dict) else None
        sum_of_parts = _finite_float(best.get("baseline"))
        if sum_of_parts is None:
            sum_of_parts = _finite_float(best.get("sum_of_parts"))
        completeness = None
        complete = sum_of_parts is not None and with_plugin is not None
        # Older runs filed this interval under "effectiveness".
        uncertainty = _lift_uncertainty_entry(best, "integration") or _lift_uncertainty_entry(best, "effectiveness")
    else:
        return _unmeasured(_INTEGRATION_REASON_NO_ARM, components)
    names = _LEGACY_ARM_NAMES if control_condition == "without_skill" else _ARM_NAMES

    # The headline, both arm scores and the interval share one basis: the
    # dimensions both arms scored, as case-weighted paired means. When one
    # arm lost a trial, the paired statistics still carry the cases both
    # arms scored; the lift is kept as a point estimate of a partial run.
    if basis is not None:
        with_plugin = _finite_float(basis.get("with_skill"))
        lift = _finite_float(basis.get("lift"))
    elif uncertainty is not None and _finite_float(uncertainty.get("treatment_score")) is not None:
        with_plugin = _finite_float(uncertainty.get("treatment_score"))
        sum_of_parts = _finite_float(uncertainty.get("control_score"))
        lift = _finite_float(uncertainty.get("estimate"))
    else:
        lift = round(with_plugin - sum_of_parts, 4) if with_plugin is not None and sum_of_parts is not None else None
    if uncertainty is not None and uncertainty.get("partial") is True:
        complete = False
    point_verdict = _integration_verdict(lift, complete=True) if lift is not None else None
    verdict = point_verdict or INTEGRATION_VERDICT_INCONCLUSIVE
    reason: str | None = None
    if lift is None:
        reason = _integration_no_score_reason(best, control_condition, names=names)
    elif not complete:
        reason = _integration_completeness_reason(
            completeness if isinstance(completeness, dict) else None,
            uncertainty,
            names=names,
        )
    elif uncertainty is not None and uncertainty.get("precision") == "insufficient":
        reason = (
            f"Only {_as_nonnegative_int(uncertainty.get('n_cases'))} paired case(s); at least "
            f"{LIFT_CI_MIN_PAIRED_CASES} are needed for a usable interval on the Integration lift."
        )
    elif uncertainty is not None:
        verdict, reason = _integration_interval_verdict(uncertainty)
    if reason is not None:
        verdict = INTEGRATION_VERDICT_INCONCLUSIVE
    return _integration_block(
        components=components,
        with_plugin=with_plugin,
        sum_of_parts=sum_of_parts,
        lift=lift,
        verdict=verdict,
        point_verdict=point_verdict,
        complete=complete,
        completeness=completeness if isinstance(completeness, dict) else None,
        reason=reason,
        uncertainty=uncertainty,
        lift_modes=lift_modes,
    )


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
        numeric = _finite_float(value)
        if numeric is None:
            return VERDICT_NEUTRAL
        scores.append(numeric)

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


def _canonical_agent_rank(agent: dict[str, Any]) -> tuple[float, float] | None:
    """Return the score/lift tuple used to rank one canonical agent payload."""
    if agent.get("execution_status") != "succeeded":
        return None
    score = _finite_float(agent.get("with_skill"))
    if score is None:
        return None
    return score, _as_float(agent.get("lift"))


def _canonical_agent_rank_from_info(info: dict[str, Any]) -> tuple[float, float] | None:
    """Build and rank raw Harbor agent data exactly as the canonical report does."""
    from skillevaluator.tier3.harbor.report_data import metrics_for_condition

    agent = _build_agent(
        "",
        info,
        metrics_for_condition(info, "with_skill"),
        metrics_for_condition(info, "without_skill"),
        None,
    )
    return _canonical_agent_rank(agent)


def _attach_integration_reports(
    payload: dict[str, Any],
    agents: dict[str, dict[str, Any]],
    best_agent: str,
    run_config: dict[str, Any] | None,
    plugin_provenance: dict[str, Any] | None,
) -> None:
    """Give every agent its own named Integration block, and the run one for its Integration agent.

    A multi-agent run compares each agent's plugin arm with that agent's
    member-skills arm; one unnamed block from the best agent hid the others.
    ``payload["integration"]`` stays the run-level block (the best agent's,
    or the first agent with an Integration comparison) and names its agent.
    """
    integration, per_agent = _integration_reports(agents, best_agent, run_config, plugin_provenance)
    for name, block in per_agent.items():
        agents[name]["integration"] = block
    if integration is not None:
        payload["integration"] = integration


def _integration_reports(
    agents: dict[str, dict[str, Any]],
    best_agent: str,
    run_config: dict[str, Any] | None,
    plugin_provenance: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Return the run-level Integration block and each agent's own block, each naming its agent.

    The run-level block is a copy of the Integration agent's block
    (:func:`_integration_agent`), or ``None`` when that agent has none.
    """
    primary = _integration_agent(agents, best_agent)
    integration: dict[str, Any] | None = None
    per_agent: dict[str, dict[str, Any]] = {}
    for name, agent in agents.items():
        block = _build_integration_report(agent, run_config, plugin_provenance)
        if block is None:
            continue
        block["agent"] = name
        per_agent[name] = block
        if agent is primary:
            integration = dict(block)
    return integration, per_agent


def _integration_agent(agents: dict[str, dict[str, Any]], best_agent: str) -> dict[str, Any]:
    """The agent whose Integration comparison the run-level block reports.

    The best agent when there is one. Integration compares the with-plugin and
    member-skills arms only, so an agent whose no-plugin baseline lost a trial
    (and so is not "succeeded") still has a valid Integration comparison.
    """
    if best_agent and best_agent in agents:
        return agents[best_agent]
    for name in sorted(agents):
        agent = agents[name]
        if _lift_uncertainty_entry(agent, "integration") or _finite_float(agent.get("sum_of_parts")) is not None:
            return agent
    return next((agents[name] for name in sorted(agents)), {})


def _pick_best_agent(agents: dict[str, dict[str, Any]]) -> str:
    eligible = {name: rank for name, agent in agents.items() if (rank := _canonical_agent_rank(agent)) is not None}
    if not eligible:
        return ""
    if len(eligible) == 1:
        return next(iter(eligible))
    return max(eligible.items(), key=lambda item: item[1])[0]


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


_PLUGIN_PROVENANCE_SIDECAR = "plugin_provenance.json"
_MAX_PLUGIN_PROVENANCE_BYTES = 1024 * 1024
_MAX_PLUGIN_PROVENANCE_LIST_ITEMS = 4096
_PLUGIN_PROVENANCE_TEXT_FIELDS = frozenset(
    {
        "plugin_name",
        "requested_lift_mode",
        "effective_lift_mode",
        "integration_skip_reason",
        "lift_mode_requested",
        "lift_mode_effective",
        "sidecar_error",
    }
)
_PLUGIN_PROVENANCE_DEFERRAL_FIELDS = (
    "unresolved_skill_refs",
    "unresolved_rule_refs",
    "provider_only_mcp_servers",
    "mcp_unsupported_config",
)
_PLUGIN_PROVENANCE_LIST_FIELDS = frozenset(
    {"evaluated_member_skills", "staged_rules", "runnable_mcp_servers", *_PLUGIN_PROVENANCE_DEFERRAL_FIELDS}
)
_PLUGIN_PROVENANCE_COUNT_FIELDS = frozenset({"dataset_case_count", "cross_component_case_count"})
_PLUGIN_PROVENANCE_FLAG_FIELDS = frozenset({"integration_evidence_ready", "partial"})
_PLUGIN_PROVENANCE_MAPPING_FIELDS = frozenset(
    {"component_coverage", "context_cost", "mcp_pinning", "dependency_status_counts"}
)


def _read_plugin_provenance(run_dir: Path) -> dict[str, Any]:
    """Read the durable ``plugin_provenance.json`` sidecar written by the CLI.

    Plugin CLI paths pass provenance in-process, but re-rendering an on-disk run
    (``view``, ``render_agent_eval_html_report``) has no such caller, and the
    sidecar is the only surviving record once the temporary staging directory
    is gone. The read is descriptor-anchored under the run directory, refuses
    symlinks, hard links, and non-regular files, is bounded in size, and only
    accepts a JSON object. Known fields are type-checked; the partial flag
    fails closed, so a damaged or unreadable sidecar can never turn an
    INCOMPLETE run into a complete one.

    Returns ``{}`` only when the sidecar is absent. A sidecar that exists but
    cannot be used returns a partial record whose ``sidecar_error`` names the
    reason, so a truncated, oversized, or linked sidecar keeps the run
    INCOMPLETE instead of reading as a run without plugin provenance.
    """
    from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, stat_is_link_or_reparse

    sidecar = Path(_PLUGIN_PROVENANCE_SIDECAR)
    try:
        metadata = (run_dir / sidecar).lstat()
    except FileNotFoundError:
        return {}
    except OSError:
        logger.debug("Plugin provenance sidecar could not be inspected", exc_info=True)
        return _unusable_plugin_provenance(run_dir, "not_inspectable")
    if stat_is_link_or_reparse(metadata):
        return _unusable_plugin_provenance(run_dir, "symlink_or_reparse_point")
    try:
        with SecureRoot(run_dir) as secure_root:
            raw, _opened = secure_root.read_bytes(sidecar, _MAX_PLUGIN_PROVENANCE_BYTES)
        loaded = json.loads(raw.decode("utf-8"))
    except SecurePathError as exc:
        return _unusable_plugin_provenance(run_dir, str(exc.code or "unsafe_path"))
    except UnicodeDecodeError:
        return _unusable_plugin_provenance(run_dir, "invalid_text_encoding")
    except (ValueError, RecursionError):
        return _unusable_plugin_provenance(run_dir, "invalid_json")
    except OSError:
        return _unusable_plugin_provenance(run_dir, "read_error")
    if not isinstance(loaded, dict):
        return _unusable_plugin_provenance(run_dir, "not_a_json_object")
    return _typed_plugin_provenance(loaded)


def _unusable_plugin_provenance(run_dir: Path, code: str) -> dict[str, Any]:
    """Return the fail-closed record for a sidecar that exists but cannot be used."""
    logger.warning(
        "Plugin provenance sidecar in %s is unusable (%s); reporting the run as INCOMPLETE", run_dir.name, code
    )
    return {"partial": True, "sidecar_error": code}


def _typed_plugin_provenance(loaded: dict[str, Any]) -> dict[str, Any]:
    """Drop mistyped known fields while keeping the partial verdict fail-closed.

    A mistyped deferral field or partial flag forces ``partial``, and so does a
    recorded ``sidecar_error``.
    """
    provenance: dict[str, Any] = {}
    damaged_deferral = False
    for key, value in loaded.items():
        if not isinstance(key, str):
            continue
        if key in _PLUGIN_PROVENANCE_TEXT_FIELDS:
            if isinstance(value, str):
                provenance[key] = value
        elif key in _PLUGIN_PROVENANCE_LIST_FIELDS:
            if isinstance(value, list):
                provenance[key] = [item for item in value if isinstance(item, str)][:_MAX_PLUGIN_PROVENANCE_LIST_ITEMS]
                damaged_deferral |= key in _PLUGIN_PROVENANCE_DEFERRAL_FIELDS and len(provenance[key]) != len(value)
            else:
                damaged_deferral |= key in _PLUGIN_PROVENANCE_DEFERRAL_FIELDS
        elif key in _PLUGIN_PROVENANCE_COUNT_FIELDS:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                provenance[key] = value
        elif key in _PLUGIN_PROVENANCE_FLAG_FIELDS:
            if isinstance(value, bool):
                provenance[key] = value
            elif key == "partial":
                damaged_deferral = True
        elif key in _PLUGIN_PROVENANCE_MAPPING_FIELDS:
            if isinstance(value, dict):
                provenance[key] = _sanitize_json_numbers(value)
        else:
            provenance[key] = _sanitize_json_numbers(value)
    deferred = any(provenance.get(key) for key in _PLUGIN_PROVENANCE_DEFERRAL_FIELDS)
    if provenance.get("partial") is True or deferred or damaged_deferral or provenance.get("sidecar_error"):
        provenance["partial"] = True
    elif provenance:
        provenance["partial"] = bool(provenance.get("partial", False))
    return provenance


_MAX_SIGNAL_LIST_ITEMS = 32


def _bounded_report_copy(value: Any, *, depth: int = 0) -> Any:
    """Copy advisory report data with bounded list lengths and nesting depth."""
    if depth > 8:
        return None
    if isinstance(value, dict):
        return {
            str(key): _bounded_report_copy(item, depth=depth + 1)
            for key, item in islice(value.items(), _MAX_RAW_REWARD_FIELDS)
        }
    if isinstance(value, list | tuple):
        return [_bounded_report_copy(item, depth=depth + 1) for item in value[:_MAX_SIGNAL_LIST_ITEMS]]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


_MAX_TOP_ARGUMENT_FAILURES = 5


def _top_argument_failures(rewards: object) -> list[dict[str, Any]]:
    """Aggregate per-trial ``plugin_signals.arguments.failures`` into the most frequent failures.

    Every recorded failure row of the first ``_MAX_RAW_TRIAL_REWARDS_TOTAL``
    rewards counts (the grader keeps at most 50 per trial). Only a fallback for
    summaries without the collector's exact ``top_failures`` (older runs).
    """
    from skillevaluator.tier3.eval_core.plugin_signals import top_argument_failures

    rows = rewards[:_MAX_RAW_TRIAL_REWARDS_TOTAL] if isinstance(rewards, list) else []
    return top_argument_failures(
        (reward.get("plugin_signals") for reward in rows if isinstance(reward, dict)),
        limit=_MAX_TOP_ARGUMENT_FAILURES,
    )


def _attach_plugin_report_fields(payload: dict[str, Any], agents: dict[str, dict[str, Any]]) -> None:
    """Carry advisory per-arm plugin signals from run artifacts into the payload.

    Report-only: nothing here feeds a score or verdict. Per-arm
    ``plugin_signals_summary`` blocks loaded from condition summaries are copied
    onto each agent (bounded, with the most frequent argument failures from the
    per-trial rewards), and the best agent's copy is repeated at the top level.
    An existing value wins (``setdefault``), so a producer that already placed
    the field keeps it.
    """
    agent_payloads = payload.get("agents")
    if not isinstance(agent_payloads, dict):
        return
    for name, agent_payload in agent_payloads.items():
        raw_agent = agents.get(name) or {}
        signals = raw_agent.get("plugin_signals_summary")
        if not isinstance(agent_payload, dict) or not isinstance(signals, dict) or not signals:
            continue
        summaries = _bounded_report_copy(signals)
        for arm, summary in summaries.items():
            arguments = summary.get("arguments") if isinstance(summary, dict) else None
            if (
                isinstance(arguments, dict)
                and "failures" not in arguments
                and not arguments.get("top_failures")
                and arm in _ARM_REWARDS_FIELDS
            ):
                # The collector's exact counts win; the bounded rewards are only a fallback for older runs.
                top = _top_argument_failures(raw_agent.get(_ARM_REWARDS_FIELDS[arm]))
                if top:
                    arguments["top_failures"] = top
        agent_payload.setdefault("plugin_signals_summary", summaries)
    best = agent_payloads.get(payload.get("best_agent"))
    if isinstance(best, dict) and isinstance(best.get("plugin_signals_summary"), dict):
        payload.setdefault("plugin_signals_summary", best["plugin_signals_summary"])
    # Per-arm canary exfiltration results (the verifier already scored each leak).
    for name, agent_payload in agent_payloads.items():
        canary = (agents.get(name) or {}).get("canary_summary")
        if isinstance(agent_payload, dict) and isinstance(canary, dict) and canary:
            agent_payload.setdefault("canary_summary", _bounded_report_copy(canary))
    if isinstance(best, dict) and isinstance(best.get("canary_summary"), dict):
        payload.setdefault("canary_summary", best["canary_summary"])


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
        "attempt_pass_threshold": _finite_float(attempt_threshold),
        "dimension_pass_threshold": DIMENSION_VERDICT_PASS_THRESHOLD,
        "dimension_neutral_threshold": DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
        "lift_pass_threshold": TIER3_LIFT_PASS_THRESHOLD,
        "lift_fail_threshold": TIER3_LIFT_FAIL_THRESHOLD,
        "overall_pass_rule": "one_supported_agent_all_dimensions_pass",
    }


def _mean(values: list[float]) -> float | None:
    numeric = [finite for value in values if (finite := _finite_float(value)) is not None]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


def _complete_mean(values: list[Any]) -> float | None:
    """Average values only when every expected constituent is finite."""
    if not values:
        return None
    numeric = [_finite_float(value) for value in values]
    if any(value is None for value in numeric):
        return None
    return round(sum(value for value in numeric if value is not None) / len(numeric), 4)


def _logical_group_entry_identity_is_consistent(group: list[dict[str, Any]]) -> bool:
    """Reject ambiguous multi-row trials whose physical rows claim different cases."""
    if len(group) <= 1:
        return True
    entry_id = group[0].get("entry_id")
    return isinstance(entry_id, str) and bool(entry_id) and all(item.get("entry_id") == entry_id for item in group)


def _condition_has_mixed_metric_contracts(
    info: dict[str, Any],
    *,
    flag: str,
    rewards: str,
) -> bool:
    """Prefer collector-owned contract truth while retaining legacy inference."""
    explicit = info.get(flag)
    if isinstance(explicit, bool):
        return explicit

    from skillevaluator.tier3.harbor.metrics import rewards_have_mixed_metric_contracts

    return rewards_have_mixed_metric_contracts(info.get(rewards))


def _logical_reward_mean(rewards: Any, field: str) -> float | None:
    """Average a persisted reward field once per logical Harbor trial."""
    if not isinstance(rewards, list):
        return None
    from skillevaluator.tier3.harbor.report_data import logical_trial_reward_groups

    groups = logical_trial_reward_groups([reward for reward in rewards if isinstance(reward, dict)])
    group_means = [
        _complete_mean([reward.get(field) for reward in group])
        if _logical_group_entry_identity_is_consistent(group)
        else None
        for group in groups
    ]
    return _complete_mean(group_means)


def _as_float(value: Any) -> float:
    return _finite_float(value) or 0.0


def _as_nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _aggregate_execution_error_details(
    summaries: dict[str, dict[str, Any]],
    displayed_count: int,
) -> dict[str, Any]:
    """Aggregate hidden diagnostic occurrences without duplicating display text."""
    from skillevaluator.tier3.harbor.report_data import aggregate_execution_error_details

    return aggregate_execution_error_details(summaries.values(), displayed_count)


__all__ = [
    "advisory_skip_result",
    "agent_eval_result_from_run",
    "build_agent_eval_payload",
    "incomplete_reason",
    "integration_reports_for",
    "refresh_plugin_run_report",
]
