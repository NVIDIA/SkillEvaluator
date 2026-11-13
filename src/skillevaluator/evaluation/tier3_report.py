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
from collections.abc import Collection
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from skillevaluator import __version__
from skillevaluator.constants import (
    AGENT_EVAL_EVALUATORS,
    AGENT_EVAL_SCORE_DEFINITION,
    DEFAULT_SCORE_POLICY,
    DIMENSION_HINTS,
    DIMENSION_MAPPING,
    DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
    DIMENSION_VERDICT_PASS_THRESHOLD,
    TIER3_LIFT_FAIL_THRESHOLD,
    TIER3_LIFT_PASS_THRESHOLD,
)
from skillevaluator.evaluation.tier3_findings import (
    TRIAL_PASS_THRESHOLD,
    failure_excerpt,
    failure_reason,
    failure_reason_text_slots,
    sanitize_grader_text,
    strip_failure_reason_text,
)
from skillevaluator.evidence import evidence_ref_identity
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.source_identity import resolve_evaluated_source
from skillevaluator.tier3.evidence_volume import evidence_volume_for_conditions

logger = logging.getLogger(__name__)

# Verdict labels mirror SkillEvaluator's AGENT_EVAL_VERDICT_* values so the ported
# reporters classify the overall outcome identically.
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_NEUTRAL = "neutral"
VERDICT_INCONCLUSIVE = "inconclusive"


def _never_activated(agent: dict[str, Any]) -> bool:
    activation = agent.get("skill_activation")
    return isinstance(activation, dict) and activation.get("status") == "never_activated"


def _activation_summary(info: dict[str, Any]) -> dict[str, Any]:
    from skillevaluator.tier3.harbor.activation import summarize_skill_activation

    # A retained summary covers all trials even when report loading is bounded.
    summary = info.get("skill_activation")
    if isinstance(summary, dict):
        return dict(summary)
    summary = summarize_skill_activation(info.get("rewards") or [])
    if info.get("rewards_complete") is False and summary.get("status") == "never_activated":
        summary = dict(summary, status="unknown", reason="Activation evidence is incomplete.")
    return summary


_AGENT_EVAL_VALIDATOR = "AGENT_EVAL"
_AGENT_EVAL_DESCRIPTION = "Tier 3: Live Agent Evaluation (Harbor)"
_PUBLICATION_TARGET_CONFLICT_MARKER = "source changed during evaluation"
_RUNTIME_CONTEXT_CONFLICT_REASON_CODE = "runtime_context_outside_publication_target"
_RUNTIME_CONTEXT_CONFLICT_MARKER = "runtime context is not bound to publication target"
_PUBLICATION_TARGET_CONFLICT_MARKERS = (
    _PUBLICATION_TARGET_CONFLICT_MARKER,
    _RUNTIME_CONTEXT_CONFLICT_MARKER,
)

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
# An evaluator card lists at most this many of the cases its lift left out; the
# card's ``excluded_cases`` count stays exact.
_MAX_LIFT_EXCLUDED_CASE_IDS_PER_CARD = 64
_MAX_LIFT_EXCLUDED_CASE_ID_CHARS = 256
# Lift pairing cannot compare a trial without a resolved case.
_UNRESOLVED_CASE_IDS = frozenset({"", "unknown"})
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


def _unit_interval_score(value: object) -> float | None:
    """Return a finite score in the documented 0.0 to 1.0 range."""
    numeric = _finite_float(value)
    return numeric if numeric is not None and 0.0 <= numeric <= 1.0 else None


def _skill_lift_value(value: object) -> float | None:
    """Return a finite Skill Lift value in its documented -1.0 to 1.0 range."""
    numeric = _finite_float(value)
    return numeric if numeric is not None and -1.0 <= numeric <= 1.0 else None


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
        "score_policy": attempt_policy["score_policy"],
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
        "score_policy": attempt_policy["score_policy"],
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

    # Reporters list these as source errors, so passing checks and warnings stay
    # out. The detail lives in metadata rather than the suggestion, which keeps
    # the finding text stable while the author fixes entries one by one.
    serialized_checks = [
        {
            "path": str(getattr(check, "path", "")),
            "status": str(getattr(check, "status", "error")),
            "message": str(getattr(check, "message", check)),
        }
        for check in checks
        if str(getattr(check, "status", "error")) in {"missing", "error"}
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
    evaluated_source: dict[str, Any] | None = None,
    use_llm_judge: bool = True,
    enforce_embedded_size: bool = True,
) -> ValidationResult | None:
    """Build the canonical ``AGENT_EVAL`` result for one explicit Harbor run.

    Read-only consumers that never embed the payload pass
    ``enforce_embedded_size=False`` to keep every agent's trial details.
    """
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
    attempt_policy = _read_attempt_policy(run_dir)
    publication_target = run_truth.get("publication_target")
    persisted_skill_name = publication_target.get("skill_name") if isinstance(publication_target, dict) else None
    skill_name = (
        persisted_skill_name if isinstance(persisted_skill_name, str) and persisted_skill_name else skill_path.name
    )
    payload = build_agent_eval_payload(
        skill_name,
        agents,
        dataset=dataset,
        attempt_policy=attempt_policy,
        historical_missing_score_policy="score_policy" not in attempt_policy,
        run_config=_read_run_config(run_dir),
        env_mode=env_mode,
        runtime_seconds=_runtime_seconds(engine_result),
        harbor_viewer=_harbor_viewer_from_engine_result(engine_result),
        run_dir=run_dir,
        comparison=_read_comparison(run_dir),
        evaluated_at=evaluated_at or _evaluated_at_from_run(run_dir, engine_result),
        evaluator_version=evaluator_version or run_truth.get("evaluator_version"),
        persisted_dataset_summary=run_truth.get("dataset_summary"),
        dataset_digest=run_truth.get("dataset_digest"),
        dataset_digest_algorithm=run_truth.get("dataset_digest_algorithm"),
        dataset_provenance=run_truth.get("dataset_provenance"),
        run_id=run_truth.get("run_id"),
        publication_target=publication_target,
        publication_target_conflict=run_truth.get("publication_target_conflict"),
        evaluated_source=evaluated_source,
        use_llm_judge=use_llm_judge,
        enforce_embedded_size=enforce_embedded_size,
    )
    _apply_engine_failure(payload, engine_result if isinstance(engine_result, dict) else _persisted_run_result(run_dir))
    with contextlib.suppress(Exception):
        source = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if isinstance(source, dict) and isinstance(source.get("agents"), dict):
            from skillevaluator.tier3.case_summary import load_case_findings
            from skillevaluator.tier3.next_steps import next_steps

            payload["next_steps"] = next_steps(source, run_dir=run_dir, fixes=load_case_findings(run_dir))
    return _validation_result_from_payload(payload)


def _persisted_run_result(run_dir: Path) -> dict[str, Any] | None:
    """Read the run's own ``result.json`` so re-rendered reports keep the engine status."""
    try:
        persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return persisted if isinstance(persisted, dict) else None


def _apply_engine_failure(payload: dict[str, Any] | None, engine_result: dict[str, Any] | None) -> None:
    """Carry a failed engine status (for example a backend launch failure) into the report payload.

    Scores from the trials that did complete are kept; only the status and errors change.
    """
    if not isinstance(payload, dict) or not isinstance(engine_result, dict):
        return
    if engine_result.get("execution_status") != "failed":
        return
    raw_errors = engine_result.get("execution_errors") or []
    engine_errors = [str(error) for error in (raw_errors if isinstance(raw_errors, list) else [raw_errors]) if error]
    agents = payload.get("agents")
    agent_payloads = (
        {name: agent for name, agent in agents.items() if isinstance(agent, dict)} if isinstance(agents, dict) else {}
    )
    attributed = {
        name: [error for error in engine_errors if error.startswith((f"{name} ", f"{name}:"))]
        for name in agent_payloads
    }
    # Only agents an error names are marked; the root and summary carry unattributed failures.
    for name, errors in attributed.items():
        if errors:
            _mark_failed(agent_payloads[name], errors)
    summary = payload.get("summary")
    for target in (payload, summary) if isinstance(summary, dict) else (payload,):
        _mark_failed(target, engine_errors)


def _mark_failed(target: dict[str, Any], errors: list[str]) -> None:
    target["execution_status"] = "failed"
    existing = [str(error) for error in target.get("execution_errors") or [] if error]
    target["execution_errors"] = list(dict.fromkeys([*existing, *errors])) or [
        "Tier 3 execution backend reported a failed run"
    ]


def _validation_result_from_payload(payload: dict[str, Any] | None) -> ValidationResult | None:
    """Wrap a canonical Tier 3 payload in the shared validation-result model."""
    if payload is None:
        return None

    result = ValidationResult(
        validator_name=_AGENT_EVAL_VALIDATOR,
        validator_description=_AGENT_EVAL_DESCRIPTION,
    )
    result.metadata["agent_eval"] = payload
    publication_target = payload.get("publication_target")
    if isinstance(publication_target, dict):
        result.metadata["publication_target"] = dict(publication_target)
    publication_target_conflict = payload.get("publication_target_conflict")
    if isinstance(publication_target_conflict, str):
        result.metadata["publication_target_conflict"] = publication_target_conflict
    run_id = payload.get("run_id")
    if isinstance(run_id, str):
        result.metadata["run_id"] = run_id
    best = payload.get("best_agent") or "n/a"
    if payload.get("execution_status") == "succeeded" and _finite_float(payload.get("overall_score")) is not None:
        if payload.get("mode") == "baseline_only":
            result.add_success(
                "agent_eval",
                f"Tier 3 baseline calibration complete: best agent {best} "
                "(no with-skill arm run; no skill lift measured)",
            )
        else:
            result.add_success(
                "agent_eval",
                f"Tier 3 evaluation complete: verdict {str(payload.get('verdict', 'neutral')).upper()}; "
                f"best agent {best}",
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
    include_benchmark: bool = False,
    saved_feedback: dict[str, list[Any]] | None = None,
) -> Path:
    """Render a standalone Tier 3 HTML report and optional standard benchmark.

    ``saved_feedback`` is the report feedback a completed run saved in
    ``result.json``. It replaces the rebuilt feedback fields and nothing calls
    an LLM; see ``_apply_saved_feedback``.
    """
    from skillevaluator.reporting import BenchmarkReporter, HTMLReporter

    skill_path = skill_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    result = agent_eval_result_from_directory(
        skill_path,
        run_dir,
        env_mode=env_mode,
        engine_result=engine_result,
        use_llm_judge=use_llm_judge and not saved_feedback,
    )
    if result is None:
        raise ValueError(f"No agent results found in {run_dir}")

    canonical_payload = result.metadata.get("agent_eval")
    if isinstance(canonical_payload, dict):
        if saved_feedback:
            _apply_saved_feedback(canonical_payload, saved_feedback, engine_result, run_dir, skill_path)
        else:
            _apply_case_findings(canonical_payload, engine_result, run_dir, skill_path, use_llm=use_llm_judge)
        # Restored feedback and fresh findings both join after the initial size budget.
        fit_token_usage(canonical_payload)
    if engine_result is not None and isinstance(canonical_payload, dict):
        # Persist only the compact feedback contract needed by the CLI. The
        # complete canonical payload remains in the HTML report and can be much
        # larger because it duplicates trials, datasets, agents, and provenance.
        engine_result["tier3_feedback"] = {
            "schema_version": _TIER3_FEEDBACK_SCHEMA_VERSION,
            **{field: list(canonical_payload.get(field) or []) for field in _TIER3_FEEDBACK_FIELDS},
        }
        from skillevaluator.tier3.token_usage import per_agent_token_usage

        # The terminal summary prints token usage per agent; per-case rows stay in the report files.
        if token_usage := per_agent_token_usage(canonical_payload.get("token_usage")):
            engine_result["token_usage"] = token_usage
        else:
            engine_result.pop("token_usage", None)

    target = output_path.expanduser().resolve() if output_path is not None else run_dir / "report.html"
    from skillevaluator.publication_identity import publication_target_name

    reporter = HTMLReporter(
        target_path=str(skill_path),
        content_label="Skill",
        report_scope="tier3",
        tabs=[{"id": "tier3", "label": "Tier 3: Live Agent Evaluation"}],
        # The run records the filesystem entry name, which a lexical alias
        # on a case- or normalization-insensitive filesystem need not match.
        expected_skill_name=publication_target_name(skill_path) or skill_path.name,
    )
    written: list[Path] = []
    try:
        reporter.save([result], target)
        written.append(target)
        if include_benchmark:
            BenchmarkReporter(
                skill_name=publication_target_name(skill_path) or skill_path.name, report_scope="tier3"
            ).save([result], target.with_name("tier3-benchmark.md"))
    except Exception:
        # Do not advertise half of a newly generated report pair. An unsafe
        # destination that the writer rejected is never followed or removed.
        for path in written:
            path.unlink(missing_ok=True)
        raise
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
    dataset_provenance: dict[str, Any] | None = None,
    historical_missing_score_policy: bool = False,
    run_id: str | None = None,
    publication_target: dict[str, str] | None = None,
    publication_target_conflict: str | None = None,
    evaluated_source: dict[str, Any] | None = None,
    use_llm_judge: bool = True,
    enforce_embedded_size: bool = True,
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
    Diagnostics tab. ``enforce_embedded_size=False`` skips the embedded-report
    size pruning; loader truncation is still recorded in ``report_truncation``.
    """
    from skillevaluator.tier3.harbor.report_data import (
        build_dataset_snapshot,
        deduplicate_dataset_entries,
        metrics_for_agents,
        metrics_for_condition,
    )

    mode = _evaluation_mode(agents)
    baseline_only = mode == "baseline_only"
    # A baseline-only run (--skip-with-skill) scores only its without-skill arm,
    # which then supplies the metric contract, ranking and overall score.
    scored_condition = "without_skill" if baseline_only else "with_skill"
    score_field = "baseline" if baseline_only else "with_skill"
    # Custom-only agents publish their grader's own overall under the custom
    # policy and rank against standard agents. The run-level policy, its
    # conflict checks and the metric-set comparison cover standard agents.
    custom_only_agents = {name for name, info in agents.items() if _is_custom_only_agent(info)}
    policy_agents = {name: info for name, info in agents.items() if name not in custom_only_agents} or agents
    metrics = metrics_for_agents(policy_agents, condition=scored_condition)
    policy_metrics = _policy_metrics_from_agents(policy_agents, metrics, condition=scored_condition)

    from skillevaluator.tier3.harbor.metrics import (
        CUSTOM_SCORE_POLICY,
        DEFAULT_METRICS,
        LEGACY_SCORE_POLICY,
        SUPPORTED_SCORE_POLICIES,
        score_policy_for_metrics,
    )

    policy = dict(attempt_policy) if attempt_policy else _default_attempt_policy()
    if not attempt_policy:
        policy.pop("score_definition", None)
        policy.pop("score_policy", None)
    recorded_policies = _recorded_score_policies(policy_agents)
    stored_policy = policy.get("score_policy")
    if isinstance(stored_policy, str) and stored_policy.strip():
        policy["score_policy"] = stored_policy.strip()
    elif len(recorded_policies) == 1:
        policy["score_policy"] = next(iter(recorded_policies))
    elif historical_missing_score_policy and policy_metrics == DEFAULT_METRICS:
        policy["score_policy"] = LEGACY_SCORE_POLICY
    else:
        policy["score_policy"] = score_policy_for_metrics(policy_metrics)
    conflicting_policies = len(recorded_policies) > 1 or bool(
        recorded_policies and policy["score_policy"] not in recorded_policies
    )
    if conflicting_policies:
        policy["score_policy"] = "mixed-score-policies"
        policy.pop("score_definition", None)
    if policy["score_policy"] != CUSTOM_SCORE_POLICY or not isinstance(policy.get("score_definition"), str):
        policy["score_definition"] = _score_definition_for_policy(policy_metrics, policy["score_policy"])

    report_budget = _ReportBudget(artifact_loading=_artifact_loading_reasons(agents, dataset))
    agent_payloads: dict[str, dict[str, Any]] = {}
    for name in sorted(agents):
        info = agents[name]
        model = _agent_model(name, info, run_config)
        agent_payloads[name] = _build_agent(
            name,
            info,
            metrics_for_condition(info, "with_skill"),
            metrics_for_condition(info, "without_skill"),
            model,
            CUSTOM_SCORE_POLICY if name in custom_only_agents else policy["score_policy"],
            baseline_only=baseline_only,
        )

    if not agent_payloads:
        return None

    scored_metric_sets = {
        frozenset(agent[f"metrics_{scored_condition}"])
        for name, agent in agent_payloads.items()
        if name not in custom_only_agents
        and agent["execution_status"] == "succeeded"
        and agent[score_field] is not None
    }
    conflicting_metric_sets = len(scored_metric_sets) > 1
    if conflicting_metric_sets and not conflicting_policies:
        policy["score_definition"] = "overall unavailable: incomparable metric sets across agents"
    best_agent = (
        ""
        if conflicting_policies or conflicting_metric_sets
        else _pick_best_agent(agent_payloads, score_field=score_field)
    )
    detail_priority = ([best_agent] if best_agent else []) + [name for name in agent_payloads if name != best_agent]
    for name in detail_priority:
        _attach_agent_report_details(
            agent_payloads[name],
            agents.get(name, {}),
            report_budget,
            baseline_only=baseline_only,
        )
    _attach_activation_expectations(agent_payloads, dataset or [], skill_name)
    best = agent_payloads.get(best_agent, {})
    if suggestions_v2 is None and run_dir is not None:
        # Findings are per-agent evidence. Resolve the canonical best agent
        # before loading them, so stale advice from a different agent cannot
        # become the run's recommendations.
        suggestions_v2 = _load_suggestions_v2(run_dir, {best_agent: agents[best_agent]} if best_agent else {})

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
    if conflicting_policies:
        execution_status = "failed"
        execution_errors.append(
            "Conflicting score policies across scored conditions: " + ", ".join(sorted(recorded_policies))
        )
    elif conflicting_metric_sets:
        execution_status = "failed"
        execution_errors.append("Incomparable metric sets across scored agents; overall ranking is unavailable")
    elif policy["score_policy"] not in SUPPORTED_SCORE_POLICIES and any(
        agent.get("execution_status") == "succeeded" and agent.get(score_field) is None
        for agent in agent_payloads.values()
    ):
        execution_status = "failed"
        execution_errors.append("Unsupported score policy has no persisted overall score")
    # Count score-policy errors appended above as displayed diagnostics.
    execution_error_details = _aggregate_execution_error_details(agent_payloads, len(execution_errors))

    raw_overall_score = best.get(score_field)
    overall_score = _finite_float(raw_overall_score) if execution_status == "succeeded" else None
    overall_lift = _finite_float(best.get("lift"))
    # Baseline calibration scores measure no skill, so they never set a skill verdict.
    # The verdict reads the agents the best agent is picked from.
    verdict_agents = _headline_agents(agent_payloads, score_field=score_field)
    verdict = (
        _overall_verdict_from_agents(verdict_agents)
        if overall_score is not None and not baseline_only
        else VERDICT_NEUTRAL
    )

    metric_ids = list(best.get("evaluators", {}).keys())
    metric_labels = _metric_labels(metric_ids)
    canonical_trials = _flatten_trials(agent_payloads)
    # The drill-down reads with-skill reasons from the flattened trials and
    # baseline reasons from ``trials_baseline``, so keep one copy of each.
    for agent in agent_payloads.values():
        for trial in agent.get("trials", []):
            trial.pop("score_reasons", None)
    public_dataset = deduplicate_dataset_entries([entry for entry in (dataset or []) if isinstance(entry, dict)])
    from skillevaluator.evaluation.tier3_expectations import build_case_diagnostics

    case_diagnostics, expectation_omissions = build_case_diagnostics(public_dataset, agents, evaluation_mode=mode)
    for section, count in expectation_omissions.items():
        report_budget.omit(section, count)
    computed_dataset_truth = (
        build_dataset_snapshot(public_dataset, evaluator_version=evaluator_version or "") if public_dataset else None
    )
    dataset_summary = (
        dict(persisted_dataset_summary)
        if isinstance(persisted_dataset_summary, dict)
        else _dataset_summary(public_dataset, canonical_trials)
    )
    from skillevaluator.tier3.dataset_sufficiency import MIN_NEGATIVE_CASES, assess_dataset_sufficiency

    for agent_name, agent_payload in agent_payloads.items():
        agent_info = agents.get(agent_name, {})
        agent_payload["evidence_volume"] = evidence_volume_for_conditions(
            agent_info.get("conditions"),
            trial_failures=agent_info.get("trial_failures"),
            baseline_only=baseline_only,
            dataset_summary=dataset_summary,
            rewards_by_arm={
                "with_skill": agent_info.get("rewards"),
                "without_skill": agent_info.get("rewards_baseline"),
            },
            uniform_attempts=not bool(policy.get("stop_on_pass")),
            attempts_per_case=_as_nonnegative_int(policy.get("max_attempts")) or None,
        )
        pass_summary = agent_payload.get("pass_at_k") if isinstance(agent_payload.get("pass_at_k"), dict) else {}
        with_pass = pass_summary.get("with_skill") if isinstance(pass_summary.get("with_skill"), dict) else {}
        without_pass = pass_summary.get("without_skill")
        lift = pass_summary.get("lift") if isinstance(pass_summary.get("lift"), dict) else {}
        paired = lift.get("paired_comparison") if isinstance(lift.get("paired_comparison"), dict) else {}
        raw_paired_cases = paired.get("paired_cases")
        paired_cases = (
            _as_nonnegative_int(raw_paired_cases)
            if isinstance(raw_paired_cases, int) and not isinstance(raw_paired_cases, bool) and raw_paired_cases >= 0
            else None
        )
        unclassified_cases = _as_nonnegative_int(dataset_summary.get("unclassified_tasks"))
        negative_cases = _as_nonnegative_int(dataset_summary.get("negative_tasks"))
        if unclassified_cases > 0 and negative_cases < MIN_NEGATIVE_CASES:
            negative_cases = None
        agent_payload["dataset_sufficiency"] = assess_dataset_sufficiency(
            distinct_cases=_as_nonnegative_int(dataset_summary.get("total_tasks")),
            negative_cases=negative_cases,
            pass_rate_interval=with_pass.get("rate_interval"),
            paired_cases=paired_cases,
            baseline_skipped=not bool(without_pass),
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
    # The full viewer summary remains diagnostic. Advice must only link to
    # agents whose trials were eligible for its quality assessment; identical
    # case IDs across agents do not make their traces interchangeable.
    advice_agents = {name for name, agent in agent_payloads.items() if not _never_activated(agent)}
    evidence_links = [link for link in harbor_summary.get("evidence_links") or [] if link.get("agent") in advice_agents]
    best_evidence_links = [link for link in evidence_links if link.get("agent") == best_agent]
    verdict_source: dict[str, Any] | None = None
    if best_agent and _agent_quality_verdict(agent_payloads.get(best_agent, {})) == verdict:
        verdict_source = agent_payloads[best_agent]
    else:
        # The reason and counts come from an agent the verdict was read from.
        verdict_source = next(
            (agent for agent in verdict_agents.values() if _agent_quality_verdict(agent) == verdict),
            None,
        )

    from skillevaluator.tier3.dataset_provenance import sanitize_generation_record

    safe_dataset_provenance = sanitize_generation_record(dataset_provenance)
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "verdict": verdict,
        "verdict_reason": None if verdict == VERDICT_PASS else _agent_quality_verdict_reason(verdict_source or {}),
        "skill_activation": best.get("skill_activation", {}),
        "skill_name": skill_name,
        "mode": mode,
        "best_agent": best_agent,
        "agents_run": list(agent_payloads.keys()),
        "overall_score": round(overall_score, 4) if overall_score is not None else None,
        "overall_lift": round(overall_lift, 4) if overall_lift is not None else None,
        "correct_answers": verdict_source.get("correct_answers") if verdict_source else None,
        "required_checks": verdict_source.get("required_checks") if verdict_source else None,
        "security_clean": verdict_source.get("security_clean") if verdict_source else None,
        "environment": env_mode,
        "runtime_seconds": _finite_float(runtime_seconds) or 0.0,
        "evaluated_at": evaluated_at,
        "evaluator_version": evaluator_version,
        "dataset_summary": dataset_summary,
        "dataset_sufficiency": best.get("dataset_sufficiency"),
        "evidence_volume": best.get("evidence_volume"),
        "dataset_digest": effective_dataset_digest,
        "dataset_digest_algorithm": effective_dataset_digest_algorithm,
        "score_policy": policy["score_policy"],
        "run_id": run_id,
        "publication_target": dict(publication_target) if isinstance(publication_target, dict) else None,
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
    if safe_dataset_provenance is not None:
        summary["dataset_provenance"] = safe_dataset_provenance
    if publication_target_conflict in _PUBLICATION_TARGET_CONFLICT_MARKERS:
        summary["publication_target_conflict"] = publication_target_conflict
    case_selection = _case_selection(run_config)
    if case_selection is not None:
        summary["case_selection"] = case_selection
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
    elif baseline_only:
        # A deliberate baseline-only run is complete, not failed: state the
        # calibration result so it is never read as a skill-lift measurement.
        deterministic_conclusions = _build_baseline_only_conclusions(best_agent, overall_score, best_dimensions)
        deterministic_suggestions = [
            "Run without --skip-with-skill to evaluate the with-skill arm and measure skill lift against this baseline."
        ]
    elif verdict == VERDICT_INCONCLUSIVE:
        deterministic_conclusions = [
            {
                "severity": "warn",
                "title": "Inconclusive: skill never activated",
                "message": (
                    "The skill was never activated in any positive with-skill trial"
                    # An agent kept with a failed baseline did not decide this verdict.
                    + ("" if len(verdict_agents) == len(agent_payloads) else " of an agent with a complete baseline")
                    + ". Scores are diagnostic only; no lift verdict was computed."
                ),
            }
        ]
        deterministic_suggestions = ["Check skill loading and agent support, then rerun the evaluation."]
    else:
        deterministic_conclusions = _build_conclusions(
            {name: agent for name, agent in agent_payloads.items() if not _never_activated(agent)},
            best_dimensions,
            pass_threshold=_pass_threshold_from_policy(policy),
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
        best_evidence_links,
    )

    budget_metadata = (run_config or {}).get("budget")
    from skillevaluator.tier3.harbor.run_budget import budget_line

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "baseline": (run_config or {}).get("baseline", {}),
        "prior_version_comparison": (run_config or {}).get("prior_version_comparison", {}),
        "summary": summary,
        "skill_name": skill_name,
        "mode": mode,
        "verdict": verdict,
        "verdict_reason": summary["verdict_reason"],
        "skill_activation": summary["skill_activation"],
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
        "dataset_sufficiency": best.get("dataset_sufficiency"),
        "evidence_volume": best.get("evidence_volume"),
        "dataset_digest": effective_dataset_digest,
        "dataset_digest_algorithm": effective_dataset_digest_algorithm,
        "dataset_provenance": safe_dataset_provenance,
        "score_policy": policy["score_policy"],
        "run_id": run_id,
        "publication_target": dict(publication_target) if isinstance(publication_target, dict) else None,
        "evaluated_source": effective_evaluated_source,
        "verdict_policy": verdict_policy,
        "agents": agent_payloads,
        "dimensions": best_dimensions,
        "dimension_hints": dict(DIMENSION_HINTS),
        "evaluators": best.get("evaluators", {}),
        "not_applicable_evaluators": best.get("not_applicable_evaluators", []),
        "not_applicable_dimensions": best.get("not_applicable_dimensions", []),
        "evaluator_cards": best.get("evaluator_cards", []),
        "cases": best.get("cases", []),
        "trials": canonical_trials,
        "pass_at_k": best.get("pass_at_k", {}),
        "correct_answers": best.get("correct_answers"),
        "required_checks": best.get("required_checks"),
        "security_clean": best.get("security_clean"),
        "insights": _insights_from_dimensions(best_dimensions),
        "conclusions": list(deterministic_conclusions),
        "recommendations": recommendations,
        "suggestions": list(deterministic_suggestions),
        "suggestions_v2": _attach_harbor_evidence_to_suggestions_v2(
            [] if _never_activated(best) or baseline_only else (suggestions_v2 or []),
            best_evidence_links,
        ),
        "metric_ids": metric_ids,
        "supported_metric_ids": list(AGENT_EVAL_EVALUATORS),
        "metric_labels": metric_labels,
        "attempt_policy": policy,
        "budget": budget_metadata,
        "budget_summary": budget_line(budget_metadata) if isinstance(budget_metadata, dict) else None,
        "dataset": public_dataset,
        "case_diagnostics": case_diagnostics,
        "provenance": _build_provenance(
            agent_payloads,
            agents,
            run_dir,
            comparison,
            report_budget,
            detail_priority=detail_priority,
        ),
    }
    if publication_target_conflict in _PUBLICATION_TARGET_CONFLICT_MARKERS:
        payload["publication_target_conflict"] = publication_target_conflict
    if case_selection is not None:
        payload["case_selection"] = dict(case_selection)
    if harbor_summary:
        payload["harbor_viewer"] = harbor_summary
    from skillevaluator.reporting.tier3_coverage import coverage_notices

    # Name each arm that ran but failed, so no report reads its missing scores
    # as an arm that never ran.
    if notices := coverage_notices(agent_payloads):
        payload["coverage_notices"] = notices
    from skillevaluator.tier3.token_usage import USAGE_ONLY_ATTEMPT_KEYS, build_token_usage

    # Measured, never scored: token usage per case and agent, counting attempts
    # that failed before scoring. Scores and ``resources`` do not read it. It
    # joins the payload after the budget, so it never displaces score evidence.
    baseline_record = (run_config or {}).get("baseline")
    token_usage = build_token_usage(
        {
            name: {**agent, **{key: agents.get(name, {}).get(key) for key in USAGE_ONLY_ATTEMPT_KEYS}}
            for name, agent in agent_payloads.items()
        },
        mode=mode,
        # A reused baseline's copied trajectories hold the tokens of the run it came from.
        baseline_reused=isinstance(baseline_record, dict) and baseline_record.get("mode") == "reused",
        # The dataset's own case IDs are the only canonical ones, so every case
        # row is checked against them and none is invented from a trial name.
        dataset_case_ids=[entry["id"] for entry in public_dataset if isinstance(entry.get("id"), str) and entry["id"]],
    )

    _layer_llm_insights(
        payload,
        deterministic_conclusions=deterministic_conclusions,
        deterministic_suggestions=deterministic_suggestions,
        # The insights judge reasons about with-skill quality, which a
        # baseline-only run never measured.
        use_llm_judge=use_llm_judge and overall_score is not None and not _never_activated(best) and not baseline_only,
    )
    if evidence_links:
        payload["conclusions"] = _attach_harbor_evidence_to_conclusions(
            payload.get("conclusions") or [],
            evidence_links,
        )
        linked_recommendations = []
        for index, recommendation in enumerate(payload.get("recommendations") or []):
            links = (
                best_evidence_links
                if isinstance(recommendation, dict) and recommendation.get("source") == "deterministic"
                else evidence_links
            )
            linked_recommendations.extend(
                _attach_harbor_evidence_to_recommendations([recommendation], links, start=index)
            )
        payload["recommendations"] = linked_recommendations
        payload["suggestions_v2"] = _attach_harbor_evidence_to_suggestions_v2(
            payload.get("suggestions_v2") or [],
            best_evidence_links,
        )
    payload = _sanitize_json_numbers(payload)
    _enforce_report_payload_budget(payload, report_budget, enforce_size=enforce_embedded_size)
    if token_usage:
        payload["token_usage"] = _sanitize_json_numbers(token_usage)
        if enforce_embedded_size:
            fit_token_usage(payload)
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

        # In mixed runs, keep never-activated trials available as diagnostics but
        # exclude them from skill-quality advice and its grounding evidence.
        eligible_agents = {
            name: agent
            for name, agent in (payload.get("agents") or {}).items()
            if isinstance(agent, dict) and not _never_activated(agent)
        }
        insight_payload = {
            **payload,
            "agents": eligible_agents,
            "agents_run": list(eligible_agents),
            "trials": [
                trial
                for trial in payload.get("trials") or []
                if isinstance(trial, dict) and trial.get("agent") in eligible_agents
            ],
        }
        extra = build_insights(
            insight_payload,
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


def _compact_agent_conditions(agent: dict[str, Any]) -> bool:
    """Bound condition diagnostics while preserving publication truth fields."""
    conditions = agent.get("conditions")
    if not isinstance(conditions, dict):
        return False

    def bounded_count(condition: dict[str, Any], key: str) -> int:
        value = condition.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**63 - 1 else 0

    compact: dict[str, dict[str, Any]] = {}
    canonical_names = {"with_skill", "without_skill"}
    for condition_name in ("with_skill", "without_skill"):
        condition = conditions.get(condition_name)
        if not isinstance(condition, dict):
            continue
        raw_errors = condition.get("execution_errors")
        if isinstance(raw_errors, list) and not raw_errors:
            bounded_errors: list[str] | None = []
        elif isinstance(raw_errors, list):
            bounded_errors = [
                error[:1024] if isinstance(error, str) else "Malformed condition execution error"
                for error in raw_errors[:16]
            ] or ["Condition execution errors were omitted"]
        else:
            bounded_errors = None
        raw_status = condition.get("execution_status")
        compact[condition_name] = {
            "execution_status": raw_status if isinstance(raw_status, str) and len(raw_status) <= 16 else None,
            "execution_errors": bounded_errors,
            "expected_attempts": bounded_count(condition, "expected_attempts"),
            "scored_attempts": bounded_count(condition, "scored_attempts"),
        }
    if set(conditions) - canonical_names:
        # Preserve invalidity without retaining attacker-controlled keys or
        # values. Publication validation requires exactly the two canonical
        # condition names, so compaction must never repair malformed evidence.
        compact["invalid_condition_shape"] = {
            "execution_status": None,
            "execution_errors": None,
            "expected_attempts": 0,
            "scored_attempts": 0,
        }
    changed = compact != conditions
    agent["conditions"] = compact
    return changed


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
        if _compact_agent_conditions(agent):
            omitted += 1
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


def _enforce_report_payload_budget(
    payload: dict[str, Any], report_budget: _ReportBudget, *, enforce_size: bool = True
) -> None:
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
    if not enforce_size or _serialized_payload_size(payload) <= _MAX_EMBEDDED_REPORT_BYTES:
        return

    # Built failure reasons and excerpts only explain scores the payload already
    # carries, so they go before any recorded evidence.
    deferred_reasons = [(container, key, container[key]) for container, key in failure_reason_text_slots(payload)]
    omitted_reasons = strip_failure_reason_text(payload)
    if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES:
        report_budget.omit("case_diagnostics", len(payload.pop("case_diagnostics", [])))
        for container, key, value in deferred_reasons:
            container[key] = value
        refresh_signal()
        omitted_reasons = (
            strip_failure_reason_text(payload) if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES else 0
        )
    report_budget.omit("failure_reasons", omitted_reasons)
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
        compacted_attempt_verdicts = 0
        trial_lists = []
        trial_lists.extend(
            agent.get(key)
            for agent in (payload.get("agents") or {}).values()
            if isinstance(agent, dict)
            for key in ("trials", "trials_baseline")
        )
        for trials in trial_lists:
            if not isinstance(trials, list):
                continue
            for trial in trials:
                attempt = trial.get("attempt_verdict") if isinstance(trial, dict) else None
                if not isinstance(attempt, dict):
                    continue
                if "reason" in attempt or "warnings" in attempt:
                    attempt.pop("reason", None)
                    attempt.pop("warnings", None)
                    compacted_attempt_verdicts += 1
        flattened_trials = payload.get("trials")
        if isinstance(flattened_trials, list):
            payload["trials"] = [
                {key: value for key, value in trial.items() if key != "attempt_verdict"}
                if isinstance(trial, dict) and "attempt_verdict" in trial
                else trial
                for trial in flattened_trials
            ]
            compacted_attempt_verdicts += sum(
                isinstance(trial, dict) and "attempt_verdict" in trial for trial in flattened_trials
            )
        report_budget.omit("attempt_verdict_details", compacted_attempt_verdicts)
        refresh_signal()

    if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES:
        raw_rewards = provenance.get("raw_trial_rewards") if isinstance(provenance, dict) else None
        if isinstance(raw_rewards, dict):
            omitted = sum(len(items) for items in raw_rewards.values() if isinstance(items, list))
            provenance["raw_trial_rewards"] = {name: [] for name in raw_rewards}
            report_budget.omit("raw_trial_rewards", omitted)
        refresh_signal()

    if _serialized_payload_size(payload) > _MAX_EMBEDDED_REPORT_BYTES:
        dataset = payload.get("dataset")
        if isinstance(dataset, list) and dataset:
            payload["dataset"] = []
            report_budget.omit("dataset_items", len(dataset))
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
            if _compact_agent_conditions(agent):
                omitted_items += 1
        report_budget.omit("dataset_and_trial_items", omitted_items)
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


# ``report_truncation.omitted`` keys that only token usage adds.
_TOKEN_USAGE_OMISSIONS = ("token_usage", "token_usage_case_rows")


def fit_token_usage(payload: dict[str, Any]) -> None:
    """Keep measured token usage from costing a bounded payload its report limits.

    Token usage is unscored, so it may never push a payload past the embedded
    size budget or make it fail the report serialization limits (nodes, depth
    and text) that every reporter checks. It joins the payload last, after the
    size budget, so it gives way first: its per-case rows, then its sums, each
    drop recorded in ``report_truncation``. If not even that record fits, the
    token usage and its records go, which leaves the payload as it is without
    token usage. Code that adds fields to a built payload, such as case findings
    or ``validate``'s applicability keys, calls this again.
    """
    omitted = _report_omissions(payload)
    if "token_usage" not in payload and not (omitted and any(key in omitted for key in _TOKEN_USAGE_OMISSIONS)):
        return
    if _token_usage_fits(payload):
        return
    if "token_usage" in payload:
        from skillevaluator.tier3.token_usage import drop_token_usage_cases

        dropped = drop_token_usage_cases(payload["token_usage"])
        if (
            dropped
            and _record_token_usage_omission(payload, "token_usage_case_rows", dropped)
            and _token_usage_fits(payload)
        ):
            return
        del payload["token_usage"]
        if _record_token_usage_omission(payload, "token_usage", 1) and _token_usage_fits(payload):
            return
    _remove_token_usage_omissions(payload)


def _report_omissions(payload: dict[str, Any]) -> dict[str, Any] | None:
    truncation = payload.get("report_truncation")
    omitted = truncation.get("omitted") if isinstance(truncation, dict) else None
    return omitted if isinstance(omitted, dict) else None


def _token_usage_fits(payload: dict[str, Any]) -> bool:
    from skillevaluator.reporting.base import agent_eval_report_serialization_issue

    # The minimal last-resort payload carries no details.
    if "payload_sections" in (_report_omissions(payload) or {}):
        return False
    try:
        size = _serialized_payload_size(payload)
    except (TypeError, ValueError):
        return False
    return size <= _MAX_EMBEDDED_REPORT_BYTES and agent_eval_report_serialization_issue(payload) is None


def _record_token_usage_omission(payload: dict[str, Any], section: str, count: int) -> bool:
    """Count a token usage drop in ``report_truncation``; ``False`` when its record cannot take it."""
    if "report_truncation" not in payload:
        payload["report_truncation"] = _ReportBudget(omitted={section: count}).signal()
        return True
    omitted = _report_omissions(payload)
    if omitted is None:
        return False
    previous = omitted.get(section)
    total = count + (previous if isinstance(previous, int) and not isinstance(previous, bool) else 0)
    payload["report_truncation"]["omitted"] = dict(sorted({**omitted, section: total}.items()))
    return True


def _remove_token_usage_omissions(payload: dict[str, Any]) -> None:
    """Take token usage's keys out of ``report_truncation``, and the record when only they made it."""
    omitted = _report_omissions(payload)
    if omitted is None or not any(key in omitted for key in _TOKEN_USAGE_OMISSIONS):
        return
    remaining = {key: value for key, value in omitted.items() if key not in _TOKEN_USAGE_OMISSIONS}
    truncation = payload["report_truncation"]
    if remaining or truncation.get("artifact_loading"):
        truncation["omitted"] = remaining
    else:
        del payload["report_truncation"]


def _serialized_payload_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _replace_with_minimal_payload(payload: dict[str, Any], report_budget: _ReportBudget) -> None:
    """Last-resort bounded shape for pathological single-field payloads."""
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    attempt_policy = payload.get("attempt_policy") if isinstance(payload.get("attempt_policy"), dict) else {}
    compact_attempt_policy = {
        key: attempt_policy[key] for key in ("max_attempts", "pass_threshold", "stop_on_pass") if key in attempt_policy
    }
    for key, limit in (("score_definition", 1024), ("score_policy", 256)):
        value = attempt_policy.get(key)
        if isinstance(value, str):
            compact_attempt_policy[key] = value[:limit]
    compact_summary = {
        key: value
        for key, value in summary.items()
        if key
        in {
            "schema_version",
            "verdict",
            "overall_score",
            "overall_lift",
            "correct_answers",
            "required_checks",
            "security_clean",
            "verdict_reason",
            "environment",
            "runtime_seconds",
            "score_policy",
            "execution_status",
            "execution_error_details_total",
            "execution_error_details_shown",
            "execution_error_details_truncated",
            "expected_attempts",
            "scored_attempts",
            "evidence_volume",
        }
    }
    compact_summary["skill_name"] = str(summary.get("skill_name") or payload.get("skill_name") or "")[:256]
    compact_summary["best_agent"] = str(summary.get("best_agent") or payload.get("best_agent") or "")[:256]
    raw_score_policy = payload.get("score_policy", summary.get("score_policy"))
    compact_summary["score_policy"] = raw_score_policy[:256] if isinstance(raw_score_policy, str) else None
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
        "correct_answers": summary.get("correct_answers"),
        "required_checks": summary.get("required_checks"),
        "security_clean": summary.get("security_clean"),
        "verdict_reason": summary.get("verdict_reason"),
        "composite_lift": payload.get("composite_lift"),
        "evidence_volume": payload.get("evidence_volume"),
        "execution_status": payload.get("execution_status"),
        "execution_errors": compact_summary["execution_errors"],
        "execution_error_details_total": compact_summary.get("execution_error_details_total", 0),
        "execution_error_details_shown": compact_summary.get("execution_error_details_shown", 0),
        "execution_error_details_truncated": compact_summary.get("execution_error_details_truncated", False),
        "expected_attempts": payload.get("expected_attempts", 0),
        "scored_attempts": payload.get("scored_attempts", 0),
        "runtime_seconds": payload.get("runtime_seconds", 0.0),
        "score_policy": compact_summary["score_policy"],
        "attempt_policy": compact_attempt_policy,
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


def _policy_metrics_from_agents(
    agents: dict[str, dict[str, Any]],
    metrics: list[str],
    *,
    condition: str = "with_skill",
) -> tuple[str, ...]:
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

    available = tuple(
        metric
        for metric in (metrics or DEFAULT_METRICS)
        if any(
            _condition_quality_available(info, condition)
            and (_finite_float(scores.get(metric)) is not None or metric in _arm_not_applicable(info, condition))
            for info in agents.values()
            for scores in (info.get(condition),)
            if isinstance(scores, dict)
        )
    )
    return available or tuple(metrics)


def _is_custom_only_agent(info: dict[str, Any]) -> bool:
    """Return whether every condition of an agent is scored only by its custom grader."""
    from skillevaluator.tier3.harbor.metrics import CUSTOM_SCORE_POLICY
    from skillevaluator.tier3.harbor.report_data import metrics_for_condition

    recorded = {
        policy.strip()
        for condition in ("with_skill", "without_skill")
        if isinstance(policy := info.get(f"score_policy_{condition}"), str) and policy.strip()
    }
    if recorded:
        return recorded == {CUSTOM_SCORE_POLICY}
    return not metrics_for_condition(info, "with_skill") and not metrics_for_condition(info, "without_skill")


def _recorded_score_policies(agents: dict[str, dict[str, Any]]) -> set[str]:
    recorded: set[str] = set()
    for info in agents.values():
        for condition, scores_key, policy_key in (
            ("with_skill", "with_skill", "score_policy_with_skill"),
            ("without_skill", "without_skill", "score_policy_without_skill"),
        ):
            scores = info.get(scores_key)
            policy = info.get(policy_key)
            if (
                _condition_quality_available(info, condition)
                and isinstance(scores, dict)
                and (scores or _unit_interval_score(info.get(f"overall_{condition}")) is not None)
                and isinstance(policy, str)
                and policy.strip()
            ):
                recorded.add(policy.strip())
    return recorded


def _score_definition_for_policy(metrics: tuple[str, ...], score_policy: str) -> str:
    from skillevaluator.tier3.harbor.metrics import (
        CUSTOM_SCORE_POLICY,
        DEFAULT_METRICS,
        DEFAULT_SCORE_POLICY,
        LEGACY_SCORE_POLICY,
        PARTIAL_SCORE_POLICY,
        score_definition,
    )

    if score_policy == LEGACY_SCORE_POLICY and metrics:
        return f"overall = mean({', '.join(metrics)}) [{LEGACY_SCORE_POLICY}]"
    if score_policy == CUSTOM_SCORE_POLICY:
        return score_definition(())
    if score_policy == DEFAULT_SCORE_POLICY:
        return score_definition(DEFAULT_METRICS)
    if score_policy == PARTIAL_SCORE_POLICY:
        return score_definition(metrics)
    return f"overall = persisted score; formula unavailable [{score_policy}]"


def _complete_metric_mean(scores: dict[str, Any], metrics: list[str]) -> float | None:
    values = tuple(_finite_float(scores.get(metric)) for metric in metrics)
    if not values or any(value is None or not 0.0 <= value <= 1.0 for value in values):
        return None
    return round(sum(value for value in values if value is not None) / len(values), 4)


def _condition_quality_available(info: dict[str, Any], condition: str) -> bool:
    """Return whether a condition may contribute score-bearing report fields."""
    conditions = info.get("conditions")
    condition_info = conditions.get(condition) if isinstance(conditions, dict) else None
    status = condition_info.get("execution_status") if isinstance(condition_info, dict) else None
    if status is None:
        status = info.get("execution_status")
    return status not in {"failed", "unknown", "skipped"}


def _arm_not_applicable(info: dict[str, Any], condition: str) -> list[str]:
    """Return only all-N/A facts validated against complete scored reward groups."""
    from skillevaluator.tier3.harbor.metrics import not_applicable_list, not_applicable_metrics, score_value

    rewards_key = "rewards" if condition == "with_skill" else "rewards_baseline"
    scores = info.get(condition) or {}
    if info.get(f"not_applicable_validated_{condition}") is True:
        return [
            metric
            for metric in not_applicable_list(info.get(f"not_applicable_{condition}"))
            if score_value(scores.get(metric)) is None
        ]
    if info.get(f"{rewards_key}_complete") is True:
        return [
            metric
            for metric in not_applicable_metrics(info.get(rewards_key) or [])
            if score_value(scores.get(metric)) is None
        ]
    return []


def _not_applicable_evaluators(
    metrics: list[str],
    with_scores: dict[str, Any],
    not_applicable: list[str],
    *,
    with_skill: bool = True,
) -> list[dict[str, str]]:
    """Evaluators with no score in the reported arm because every trial recorded them as N/A."""
    from skillevaluator.tier3.harbor.metrics import METRIC_DISPLAY, not_applicable_reason

    return [
        {
            "id": metric,
            "label": METRIC_DISPLAY.get(metric, metric.replace("_", " ").title()),
            "reason": not_applicable_reason(
                metric, with_skill=with_skill, default="Not applicable to these eval cases"
            ),
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


def _logical_condition_rewards(info: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], bool]:
    """Return each arm's logical attempts as the lift rebuild compares them, and whether both loads are complete.

    The first list holds the with-skill attempts, the second the baseline's. A
    logical trial whose rows disagree on the case is left out and makes the
    result incomplete.
    """
    from skillevaluator.tier3.harbor.collector import _logical_attempt_rewards
    from skillevaluator.tier3.harbor.report_data import logical_trial_reward_groups

    conditions: list[list[dict[str, Any]]] = []
    complete = True
    for key in ("rewards", "rewards_baseline"):
        rows = info.get(key)
        complete = complete and info.get(f"{key}_complete") is True
        groups = logical_trial_reward_groups(
            [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        )
        logical_rows: list[dict[str, Any]] = []
        for index, group in enumerate(groups):
            if not _logical_group_entry_identity_is_consistent(group):
                complete = False
                continue
            logical_rows.extend(
                {
                    **row,
                    "_trial_root_name": str(index),
                    **({"_step_name": str(step)} if len(group) > 1 else {}),
                }
                for step, row in enumerate(group)
            )
        conditions.append(_logical_attempt_rewards(logical_rows))
    return conditions, complete


def _judge_mixed_lift_exclusions(info: dict[str, Any], metrics: Collection[str]) -> dict[str, dict[str, Any]]:
    """Name the cases each judge metric's lift left out because the arms' judge models differed.

    Reads the logical attempts the lift rebuild compares, so a case listed here
    is one that ``_compute_lift`` left out of that metric. Per metric it returns
    ``case_ids``, ``excluded_attempts``, the with-skill attempts that scored the
    metric on those cases, and ``complete``, whether both arms' rewards loaded
    completely with every logical trial naming one case. A saved run without
    loaded rewards yields no entry; its recorded per-metric counts still
    describe the exclusion.
    """
    from skillevaluator.tier3.harbor.collector import _entry_id, _judge_mixed_cases
    from skillevaluator.tier3.harbor.metrics import metric_set_for_reward, metric_value

    (with_rows, without_rows), complete = _logical_condition_rewards(info)
    if not with_rows or not without_rows:
        return {}
    exclusions: dict[str, dict[str, Any]] = {}
    for metric, cases in _judge_mixed_cases(with_rows, without_rows, None).items():
        case_ids = {case for case in cases if case not in _UNRESOLVED_CASE_IDS}
        if metric not in metrics or not case_ids:
            continue
        exclusions[metric] = {
            "case_ids": case_ids,
            "excluded_attempts": sum(
                1
                for row in with_rows
                if _entry_id(row) in case_ids
                and metric in metric_set_for_reward(row)[1]
                and metric_value(row, metric) is not None
            ),
            "complete": complete,
        }
    return exclusions


def _judge_aware_lift(
    info: dict[str, Any],
    with_scores: dict[str, Any],
    without_scores: dict[str, Any],
    score_policy: str,
) -> dict[str, Any] | None:
    """Rebuild judge exclusions, falling back to recorded lift for incomplete loads."""
    from skillevaluator.tier3.harbor.collector import (
        _LLM_JUDGE_METRICS,
        _compute_lift,
        _reward_judge_models,
    )

    recorded = info.get("lift")
    recorded = recorded if isinstance(recorded, dict) else {}
    from skillevaluator.tier3.harbor.metrics import not_applicable_list

    basis = recorded.get("basis")
    aggregate_only = all(
        key not in info for key in ("rewards", "rewards_baseline", "rewards_complete", "rewards_baseline_complete")
    )
    unchanged_basis = (
        isinstance(basis, dict)
        and type(basis.get("excluded_case_entries")) is int
        and basis["excluded_case_entries"] == 0
        and basis.get("excluded_cases") == []
        and basis.get("excluded_metrics") == []
        and basis.get("excluded_cases_truncated") is False
    )
    # Live collector results can contain only aggregate metric maps. With no
    # exclusions their comparison is recomputed from those maps below. Saved
    # runs always carry explicit reward-completeness fields, so missing saved
    # evidence never takes this aggregate-only path.
    paired_contract = (isinstance(basis, dict) and not (aggregate_only and unchanged_basis)) or any(
        not_applicable_list(info.get(f"_not_applicable_hint_{arm}")) or _arm_not_applicable(info, arm)
        for arm in ("with_skill", "without_skill")
    )
    recorded_mixed = recorded.get("judge_mixed_cases")
    mixed = (
        {
            metric: count
            for metric, count in recorded_mixed.items()
            if metric in _LLM_JUDGE_METRICS and isinstance(count, int) and not isinstance(count, bool) and count > 0
        }
        if isinstance(recorded_mixed, dict)
        else {}
    )
    recorded_counts = mixed.copy()
    if (
        not paired_contract
        and not mixed
        and not any(
            _reward_judge_models(row, metric)
            for rows in (info.get("rewards"), info.get("rewards_baseline"))
            if isinstance(rows, list)
            for row in rows
            if isinstance(row, dict)
            for metric in _LLM_JUDGE_METRICS
        )
    ):
        return None
    conditions, complete = _logical_condition_rewards(info)
    clean_scores = [
        {metric: value for metric, raw in scores.items() if (value := _unit_interval_score(raw)) is not None}
        for scores in (with_scores, without_scores)
    ]
    if paired_contract and not complete:
        return {"comparison_unavailable": True}
    rebuilt = _compute_lift(
        *clean_scores,
        *conditions,
        score_policy=score_policy,
        with_not_applicable=_arm_not_applicable(info, "with_skill"),
        without_not_applicable=_arm_not_applicable(info, "without_skill"),
    )
    if paired_contract and complete:
        unsupported = {
            metric for metric, count in mixed.items() if rebuilt.get("judge_mixed_cases", {}).get(metric, 0) < count
        }
        for metric in unsupported:
            rebuilt.pop(metric, None)
        if unsupported:
            rebuilt.pop("overall", None)
            rebuilt["judge_mixed_cases"] = {
                **rebuilt.get("judge_mixed_cases", {}),
                **{metric: mixed[metric] for metric in unsupported},
            }
        if any(
            _condition_has_mixed_metric_contracts(info, flag=flag, rewards=rewards)
            for flag, rewards in (
                ("mixed_metric_contracts_with_skill", "rewards"),
                ("mixed_metric_contracts_without_skill", "rewards_baseline"),
            )
        ):
            rebuilt.pop("overall", None)
            if not rebuilt.get("judge_mixed_cases"):
                logical_overall = _mixed_logical_overall_lift(conditions)
                if logical_overall is not None:
                    rebuilt["overall"] = logical_overall
        return rebuilt
    rebuilt_mixed = rebuilt.get("judge_mixed_cases", {})
    unsupported_exclusions = {metric for metric, count in mixed.items() if rebuilt_mixed.get(metric, 0) < count}
    for metric, count in rebuilt_mixed.items():
        mixed[metric] = max(count, mixed.get(metric, 0))
    if not mixed:
        return None
    if not complete:
        # The collector already excluded mixed judges from the full cohort;
        # a partial load must not replace its recorded comparison.
        covered = {metric for metric, count in mixed.items() if recorded_counts.get(metric, 0) >= count}
        if covered == mixed.keys():
            covered.add("overall")
        for metric in (*mixed, "overall"):
            rebuilt.pop(metric, None)
            if all(clean_scores) and metric in covered and metric in recorded:
                rebuilt[metric] = recorded[metric]
        rebuilt["judge_mixed_cases"] = mixed
        return rebuilt
    rebuilt["judge_mixed_cases"] = mixed
    for metric in unsupported_exclusions:
        rebuilt.pop(metric, None)
    if unsupported_exclusions:
        rebuilt.pop("overall", None)
    return rebuilt


def _mixed_logical_overall_lift(conditions: list[list[dict[str, Any]]]) -> dict[str, float] | None:
    """Compare complete mixed rewards only when both arms scored the same case contracts."""
    from skillevaluator.tier3.harbor.collector import _average_overall, _entry_id
    from skillevaluator.tier3.harbor.metrics import metric_is_not_applicable, metric_set_for_reward

    cohorts = []
    for rewards in conditions:
        by_case: dict[str, set[tuple[Any, ...]]] = {}
        for reward in rewards:
            case = _entry_id(reward)
            if case in {"", "unknown"}:
                return None
            _, metrics = metric_set_for_reward(reward)
            contract = (
                metrics,
                tuple(metric for metric in metrics if metric_is_not_applicable(reward, metric)),
                bool(reward.get("_logical_mixed_contracts")),
            )
            by_case.setdefault(case, set()).add(contract)
        cohorts.append(by_case)
    if not cohorts[0] or cohorts[0] != cohorts[1]:
        return None
    with_score, baseline_score = (_average_overall(rewards) for rewards in conditions)
    if with_score is None or baseline_score is None:
        return None
    return {"with_skill": with_score, "without_skill": baseline_score, "delta": round(with_score - baseline_score, 4)}


def _build_agent(
    name: str,
    info: dict[str, Any],
    with_metrics: list[str],
    baseline_metrics: list[str],
    model: str | None,
    score_policy: str,
    *,
    baseline_only: bool = False,
) -> dict[str, Any]:
    from skillevaluator.tier3.harbor.metrics import (
        CUSTOM_SCORE_POLICY,
        DEFAULT_METRICS,
        DEFAULT_SCORE_POLICY,
        LEGACY_SCORE_POLICY,
        PARTIAL_SCORE_POLICY,
        overall_score_from_metrics,
    )

    with_scores = info.get("with_skill") or {}
    without_scores = info.get("without_skill") or {}
    skill_activation = _activation_summary(info)
    never_activated = skill_activation.get("status") == "never_activated"
    condition_policies = {
        value.strip() if isinstance(value, str) and value.strip() else score_policy
        for value in (info.get("score_policy_with_skill"), info.get("score_policy_without_skill"))
    }
    comparable_scores = (
        not never_activated and set(with_metrics) == set(baseline_metrics) and len(condition_policies) == 1
    )
    lift_data = info.get("lift") if isinstance(info.get("lift"), dict) else {}
    comparison_unavailable = (
        info.get("comparison_unavailable") is True
        or info.get("comparison_complete") is False
        or lift_data.get("comparison_unavailable") is True
    )
    with_quality_available = _condition_quality_available(info, "with_skill")
    baseline_quality_available = _condition_quality_available(info, "without_skill")
    if not with_quality_available:
        with_scores = {}
    if not baseline_quality_available:
        without_scores = {}

    judge_lift = _judge_aware_lift(info, with_scores, without_scores, score_policy)
    if judge_lift is not None:
        lift_data = judge_lift
    with_not_applicable = _arm_not_applicable(info, "with_skill") if with_quality_available else []
    baseline_not_applicable = _arm_not_applicable(info, "without_skill") if baseline_quality_available else []
    evaluators = _build_evaluators(
        baseline_metrics if baseline_only else with_metrics,
        with_scores,
        without_scores,
        lift_data,
        with_not_applicable=with_not_applicable,
        baseline_not_applicable=baseline_not_applicable,
        baseline_only=baseline_only,
    )
    dimensions = _build_dimensions(
        with_scores,
        without_scores,
        info.get("dimensions_with_skill") or {},
        info.get("dimensions_without_skill") or {},
        judge_lift,
        with_metrics=with_metrics,
        baseline_metrics=baseline_metrics,
        with_not_applicable=with_not_applicable,
        baseline_not_applicable=baseline_not_applicable,
        baseline_only=baseline_only,
    )
    # N/A notes describe the scored arm: the baseline in a baseline-only run.
    if baseline_only:
        not_applicable_evaluators = _not_applicable_evaluators(
            baseline_metrics, without_scores, baseline_not_applicable, with_skill=False
        )
        not_applicable_dimensions = _not_applicable_dimensions(dimensions, baseline_not_applicable)
    else:
        not_applicable_evaluators = _not_applicable_evaluators(with_metrics, with_scores, with_not_applicable)
        not_applicable_dimensions = _not_applicable_dimensions(dimensions, with_not_applicable)
    if never_activated:
        for evaluator in evaluators.values():
            evaluator["lift"] = None
            evaluator.pop("lift_excluded_cases", None)
        for dimension in dimensions:
            dimension.update(
                lift=None,
                verdict=VERDICT_INCONCLUSIVE,
                explanation="Skill never activated; scores are diagnostic only.",
                reasoning_bullets=["Skill never activated; no skill-effectiveness verdict was computed."],
            )
    with_dimension_values = [d["with_skill"] for d in dimensions]
    baseline_dimension_values = [d["baseline"] for d in dimensions]
    overall_ws = None
    overall_bl = None
    if score_policy == LEGACY_SCORE_POLICY:
        overall_ws = (
            _complete_metric_mean(with_scores, [m for m in with_metrics if m not in with_not_applicable])
            if with_quality_available
            else None
        )
        overall_bl = (
            _complete_metric_mean(without_scores, [m for m in baseline_metrics if m not in baseline_not_applicable])
            if baseline_quality_available
            else None
        )
        if overall_ws is None and not with_scores and with_quality_available:
            overall_ws = _unit_interval_score(info.get("overall_with_skill"))
        if overall_bl is None and not without_scores and baseline_quality_available:
            overall_bl = _unit_interval_score(info.get("overall_without_skill"))
    elif score_policy == DEFAULT_SCORE_POLICY:
        overall_ws = (
            overall_score_from_metrics(with_scores, DEFAULT_METRICS, not_applicable=with_not_applicable)
            if with_quality_available
            else None
        )
        overall_bl = (
            overall_score_from_metrics(without_scores, DEFAULT_METRICS, not_applicable=baseline_not_applicable)
            if baseline_quality_available
            else None
        )
    elif score_policy == PARTIAL_SCORE_POLICY:
        overall_ws = (
            overall_score_from_metrics(with_scores, tuple(with_metrics), not_applicable=with_not_applicable)
            if with_quality_available
            else None
        )
        overall_bl = (
            overall_score_from_metrics(without_scores, tuple(baseline_metrics), not_applicable=baseline_not_applicable)
            if baseline_quality_available
            else None
        )
        if overall_ws is None:
            overall_ws = _mean(with_dimension_values)
        if overall_bl is None:
            overall_bl = _mean(baseline_dimension_values)
    elif score_policy == CUSTOM_SCORE_POLICY or score_policy not in {
        DEFAULT_SCORE_POLICY,
        LEGACY_SCORE_POLICY,
        PARTIAL_SCORE_POLICY,
    }:
        overall_ws = _unit_interval_score(info.get("overall_with_skill")) if with_quality_available else None
        overall_bl = _unit_interval_score(info.get("overall_without_skill")) if baseline_quality_available else None
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
    custom_with = score_policy == CUSTOM_SCORE_POLICY and overall_ws is None and not with_metrics
    custom_baseline = score_policy == CUSTOM_SCORE_POLICY and overall_bl is None and not baseline_metrics
    if with_mixed_contract or (custom_with and with_quality_available):
        # Custom-only runs have no dimension mean. For mixed condition
        # contracts, the standard-metric overall covers only standard rows and
        # can overstate Harbor's logical attempt score used by pass@k. In both
        # cases, prefer the collector-owned logical overall.
        overall_ws = _unit_interval_score(info.get("overall_with_skill"))
        if overall_ws is None and info.get("rewards_complete") is not False:
            overall_ws = _logical_reward_mean(info.get("rewards"), "overall")
    if baseline_mixed_contract or (custom_baseline and baseline_quality_available):
        overall_bl = _unit_interval_score(info.get("overall_without_skill"))
        if overall_bl is None and info.get("rewards_baseline_complete") is not False:
            overall_bl = _logical_reward_mean(info.get("rewards_baseline"), "overall")
    overall_lift = (
        round(overall_ws - overall_bl, 4)
        if comparable_scores and overall_ws is not None and overall_bl is not None
        else None
    )
    if score_policy == LEGACY_SCORE_POLICY and overall_lift is not None and with_metrics:
        with_values = tuple(_unit_interval_score(with_scores.get(metric)) for metric in with_metrics)
        baseline_values = tuple(_unit_interval_score(without_scores.get(metric)) for metric in with_metrics)
        if all(value is not None for value in (*with_values, *baseline_values)):
            # Historical lift subtracted the unrounded metric means. Recompute
            # that value from trusted metrics instead of a mutable lift artifact.
            overall_lift = round(
                sum(w - b for w, b in zip(with_values, baseline_values, strict=True)) / len(with_metrics),
                4,
            )
    if judge_lift is not None:
        overall_lift = _lift_value("overall", judge_lift) if comparable_scores else None
    if comparison_unavailable:
        overall_lift = None
    lift_overall = _lift_overall_scores(overall_lift, judge_lift, overall_ws, overall_bl)

    pass_with_skill = info.get("pass_with_skill")
    pass_threshold = _finite_float(pass_with_skill.get("pass_threshold")) if isinstance(pass_with_skill, dict) else None
    pass_threshold = pass_threshold if pass_threshold is not None else 0.5
    trials = _normalize_trials(info.get("rewards") or [], with_metrics, pass_threshold=pass_threshold)
    baseline_trials = _normalize_trials(
        info.get("rewards_baseline") or [],
        baseline_metrics,
        pass_threshold=pass_threshold,
    )
    if not never_activated:
        _attach_baseline_pairs(trials, baseline_trials, with_metrics, comparable_overall=comparable_scores)
    from skillevaluator.tier3.harbor.resource_usage import resource_comparison

    resources = resource_comparison(
        trials,
        baseline_trials,
        with_complete=with_quality_available and info.get("rewards_complete") is True,
        baseline_complete=baseline_quality_available and info.get("rewards_baseline_complete") is True,
    )

    execution_errors = (
        [str(error) for error in info.get("execution_errors", [])]
        if isinstance(info.get("execution_errors"), list)
        else []
    )
    execution_error_details = _aggregate_execution_error_details({name: info}, len(execution_errors))

    return {
        "name": name,
        "skill_activation": skill_activation,
        "verdict": VERDICT_INCONCLUSIVE if never_activated else None,
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
        "not_applicable_evaluators": not_applicable_evaluators,
        "not_applicable_dimensions": not_applicable_dimensions,
        "not_applicable_metrics": {"with_skill": with_not_applicable, "without_skill": baseline_not_applicable},
        "lift_basis": judge_lift.get("basis")
        if judge_lift
        and not never_activated
        and info.get("rewards_complete") is True
        and info.get("rewards_baseline_complete") is True
        else None,
        "evaluator_cards": [],
        "dimensions": dimensions,
        "with_skill": overall_ws,
        "baseline": overall_bl,
        "lift": overall_lift,
        "lift_overall": lift_overall,
        "metrics_with_skill": with_metrics,
        "metrics_without_skill": baseline_metrics,
        "num_trials": int(info.get("num_trials", 0) or 0),
        "num_trials_baseline": int(info.get("num_trials_baseline", len(baseline_trials)) or 0),
        "trials": trials,
        "trials_baseline": baseline_trials,
        "resources": resources,
        "correct_answers": _correct_answer_comparison(info),
        "required_checks": _required_check_comparison(info),
        "security_clean": _security_clean_comparison(info),
        "pass_at_k": {
            "with_skill": info.get("pass_with_skill") or {},
            "without_skill": info.get("pass_without_skill") or {},
            "lift": (info.get("pass_lift") or {}) if comparable_scores else {},
        },
        "evidence_volume": evidence_volume_for_conditions(
            info.get("conditions"),
            trial_failures=info.get("trial_failures"),
            baseline_only=baseline_only,
        ),
        "cases": _cases(info),
    }


def _attach_agent_report_details(
    agent_payload: dict[str, Any],
    info: dict[str, Any],
    report_budget: _ReportBudget,
    *,
    baseline_only: bool = False,
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
    evaluators = agent_payload.get("evaluators", {})
    lift_excluded_metrics = [metric for metric, values in evaluators.items() if values.get("lift_excluded_cases")]
    agent_payload["evaluator_cards"] = _evaluator_cards(
        evaluators,
        # Baseline-only cards draw their evidence from the only arm that ran.
        rewards=(info.get("rewards_baseline") if baseline_only else info.get("rewards")) or [],
        custom_with_skill=custom_with_skill if isinstance(custom_with_skill, dict) else {},
        custom_without_skill=custom_without_skill if isinstance(custom_without_skill, dict) else {},
        custom_lift=custom_lift if isinstance(custom_lift, dict) else {},
        report_budget=report_budget,
        baseline_only=baseline_only,
        lift_exclusions=_judge_mixed_lift_exclusions(info, lift_excluded_metrics) if lift_excluded_metrics else {},
        lift_basis=agent_payload.get("lift_basis"),
    )
    if _never_activated(agent_payload):
        for card in agent_payload["evaluator_cards"]:
            card["lift"] = None


def _build_evaluators(
    metrics: list[str],
    with_scores: dict[str, Any],
    without_scores: dict[str, Any],
    lift_data: dict[str, Any],
    *,
    with_not_applicable: list[str] = (),
    baseline_not_applicable: list[str] = (),
    baseline_only: bool = False,
) -> dict[str, dict[str, Any]]:
    evaluators = {}
    mixed = lift_data.get("judge_mixed_cases") or {}
    paired = isinstance(lift_data.get("basis"), dict) or lift_data.get("comparison_unavailable") is True
    for metric in metrics:
        ws, bl = _finite_float(with_scores.get(metric)), _finite_float(without_scores.get(metric))
        baseline_status = (
            "scored" if bl is not None else ("not_applicable" if metric in baseline_not_applicable else "missing")
        )
        if baseline_only:
            # No with-skill arm ran: the baseline score is the calibration row,
            # and with-skill/lift stay unavailable rather than becoming 0.0.
            evaluators[metric] = {
                "with_skill": None,
                "baseline": bl,
                "lift": None,
                "with_skill_status": "skipped",
                "baseline_status": baseline_status,
            }
            continue
        lift = _lift_value(metric, lift_data)
        if lift is None and ws is not None and bl is not None and metric not in mixed and not paired:
            lift = round(ws - bl, 4)
        evaluators[metric] = {
            "with_skill": ws,
            "baseline": bl,
            "lift": lift,
            "with_skill_status": "scored"
            if ws is not None
            else ("not_applicable" if metric in with_not_applicable else "missing"),
            "baseline_status": baseline_status,
        }
        if excluded_cases := _lift_excluded_case_count(mixed, metric):
            # The arm scores keep these cases; only this metric's lift left them out.
            evaluators[metric]["lift_excluded_cases"] = excluded_cases
    return evaluators


def _lift_excluded_case_count(mixed: object, metric: str) -> int:
    """Return how many cases a judge metric's lift left out because the arms' judge models differed."""
    from skillevaluator.tier3.harbor.collector import _LLM_JUDGE_METRICS

    count = mixed.get(metric) if isinstance(mixed, dict) and metric in _LLM_JUDGE_METRICS else None
    # A count JSON cannot carry exactly is left out rather than published as ``null``.
    return count if type(count) is int and 0 < count <= _MAX_JSON_SAFE_INTEGER else 0


def _build_dimensions(
    with_scores: dict[str, Any],
    without_scores: dict[str, Any],
    precomputed_with: dict[str, Any],
    precomputed_without: dict[str, Any],
    judge_lift: dict[str, Any] | None = None,
    *,
    with_metrics: list[str] | None = None,
    baseline_metrics: list[str] | None = None,
    with_not_applicable: list[str] = (),
    baseline_not_applicable: list[str] = (),
    baseline_only: bool = False,
) -> list[dict[str, Any]]:
    dimensions: list[dict[str, Any]] = []
    for dim_id in _DIMENSION_IDS:
        cfg = DIMENSION_MAPPING[dim_id]
        # Cached rows may predate applicability checks. Raw evaluator inputs
        # determine whether a dimension is complete and what its score is.
        ws = _dimension_score(with_scores, cfg, active_metrics=with_metrics, not_applicable=with_not_applicable)
        bl = _dimension_score(
            without_scores, cfg, active_metrics=baseline_metrics, not_applicable=baseline_not_applicable
        )
        if ws is None and bl is None:
            continue
        if baseline_only:
            if bl is not None:
                dimensions.append(_baseline_only_dimension(dim_id, cfg, bl, without_scores, precomputed_without))
            continue
        lift = round(ws - bl, 4) if ws is not None and bl is not None else None
        entry = (
            precomputed_with.get(dim_id) if ws is not None and isinstance(precomputed_with.get(dim_id), dict) else {}
        )
        # Signals (the evaluators that actually fed this dimension) populate the
        # "Signals" column; reasoning bullets and a deterministic verdict fill
        # the "Reasoning"/"Verdict" columns when the engine left them blank.
        signals = _dimension_signals(entry, with_scores, cfg)
        explanation = entry.get("explanation")
        reasoning_bullets = entry.get("reasoning_bullets")
        comparison_signals = _dimension_signals({}, with_scores, cfg)
        excluded = judge_lift is not None and (
            isinstance(judge_lift.get("basis"), dict)
            or judge_lift.get("comparison_unavailable") is True
            or any(metric in judge_lift.get("judge_mixed_cases", {}) for metric in comparison_signals)
        )
        reasoning_baseline = bl
        if excluded and judge_lift is not None:
            pairs = {metric: judge_lift.get(metric) for metric in comparison_signals}
            paired_with = {metric: pair.get("with_skill") for metric, pair in pairs.items() if isinstance(pair, dict)}
            paired_without = {
                metric: pair.get("without_skill") for metric, pair in pairs.items() if isinstance(pair, dict)
            }
            comparison_not_applicable = [*with_not_applicable, *baseline_not_applicable]
            comparison_ws = _dimension_score(
                paired_with, cfg, active_metrics=with_metrics, not_applicable=comparison_not_applicable
            )
            reasoning_baseline = _dimension_score(
                paired_without, cfg, active_metrics=baseline_metrics, not_applicable=comparison_not_applicable
            )
            lift = (
                round(comparison_ws - reasoning_baseline, 4)
                if comparison_ws is not None and reasoning_baseline is not None
                else None
            )
            reasoning_bullets = explanation = None
        if not reasoning_bullets and not explanation:
            reasoning_bullets, explanation = _deterministic_reasoning(
                ws, reasoning_baseline, lift, signals, with_scores
            )
        if excluded:
            judge_exclusion = any(metric in judge_lift.get("judge_mixed_cases", {}) for metric in comparison_signals)
            note = (
                "Cases scored by different judge models are excluded from lift."
                if judge_exclusion
                else "Lift compares only the cases both arms scored on the same metrics."
            )
            if lift is None:
                reasoning_bullets[1] = (
                    "Lift unavailable: no complete comparison with matching judge models."
                    if judge_exclusion
                    else "Lift unavailable: no complete shared comparison for this dimension."
                )
            reasoning_bullets.append(note)
            explanation = " ".join(reasoning_bullets)
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


def _baseline_only_dimension(
    dim_id: str,
    cfg: dict[str, Any],
    bl: float,
    without_scores: dict[str, Any],
    precomputed_without: dict[str, Any],
) -> dict[str, Any]:
    """Describe one dimension of a baseline-only run (--skip-with-skill).

    The baseline is the measurement here, not a comparison point: it drives the
    reasoning and band, while with-skill and lift stay unavailable.
    """
    entry = precomputed_without.get(dim_id) if isinstance(precomputed_without.get(dim_id), dict) else {}
    signals = _dimension_signals(entry, without_scores, cfg)
    explanation = entry.get("explanation")
    reasoning_bullets = entry.get("reasoning_bullets")
    if not reasoning_bullets and not explanation:
        reasoning_bullets, explanation = _deterministic_reasoning(
            bl, None, None, signals, without_scores, baseline_only=True
        )
    return {
        "id": dim_id,
        "with_skill": None,
        "score": None,
        "baseline": round(bl, 4),
        "lift": None,
        "explanation": explanation,
        "verdict": entry.get("verdict") or _deterministic_verdict(bl),
        "evaluators": signals,
        "reasoning_bullets": reasoning_bullets or [],
    }


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
    baseline_only: bool = False,
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
    if baseline_only:
        # ``ws`` carries the baseline score; replace the "no baseline" lift note.
        bullets[1] = "Baseline-only run: no with-skill arm ran, so no skill lift was measured."
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


def _metric_detail(reward: dict[str, Any], metric: str) -> dict[str, Any] | None:
    """Return the grader detail that explains one metric score in ``reward``."""
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

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
    return detail if isinstance(detail, dict) else None


def _metric_evidence(
    metric: str,
    rewards: list[dict[str, Any]],
    report_budget: _ReportBudget,
    sampling: dict[str, Any] | None = None,
    *,
    lift_excluded_case_ids: Collection[str] = (),
) -> list[dict[str, Any]]:
    from skillevaluator.tier3.eval_core.atif_helpers import _METRIC_EVIDENCE_REF_METRICS
    from skillevaluator.tier3.harbor.collector import _entry_id
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
        detail = _metric_detail(reward, metric)
        if detail is None:
            continue

        raw_score = _finite_float(reward.get(metric))
        if raw_score is None:
            raw_score = _custom_metric_value(reward, metric)
        score_failed = raw_score is not None and raw_score < TRIAL_PASS_THRESHOLD

        notes: list[str] = []
        raw_grader_reason = detail.get("reason")
        reason = sanitize_grader_text(raw_grader_reason)
        if reason:
            notes.append(reason[:512])

        failures: list[str] = []
        results = detail.get("results")
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict) or result.get("passed") is not False:
                    continue
                failure = sanitize_grader_text(result.get("reason"))
                if failure:
                    failures.append(failure[:512])
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
        if lift_excluded_case_ids and _entry_id(reward) in lift_excluded_case_ids:
            # The case's arms were scored by different judge models: its score
            # counts in the arm score but not in this metric's lift.
            entry["comparison_excluded"] = True
        if score_failed and metric in _METRIC_EVIDENCE_REF_METRICS and not entry["evidence_refs"]:
            from skillevaluator.evaluation.tier3_findings import evidence_unavailable_reason

            entry["evidence_unavailable_reason"] = evidence_unavailable_reason(reward)
        # The built reason and excerpt are not part of the fingerprint: attempts
        # of one case often quote different commands under the same grader
        # reason, and they still collapse into one entry with ``occurrences``.
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
        if score_failed:
            # Graders that record only findings or failed criteria still explain
            # a failing score. Notes already show the grader's own reason, so the
            # separate built reason omits that text and leaves out failing
            # results, which the card already lists as ``failures``.
            omitted_reasons = (raw_grader_reason, *notes) if isinstance(raw_grader_reason, str) else tuple(notes)
            if fallback_reason := failure_reason(
                detail,
                include_results=False,
                metric=metric,
                reward=reward,
                omit=omitted_reasons,
            ):
                entry["reason"] = fallback_reason
            if excerpt := failure_excerpt(detail):
                entry["excerpt"] = excerpt
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
    baseline_only: bool = False,
    lift_exclusion: dict[str, Any] | None = None,
    lift_basis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if baseline_only:
        # No with-skill arm ran: the baseline score sets the card's status band.
        ws = None
        headline = _as_float(scores.get("baseline"))
    else:
        ws = headline = _as_float(scores.get("with_skill"))
    excluded_cases = scores.get("lift_excluded_cases")
    if baseline_only or type(excluded_cases) is not int or excluded_cases <= 0:
        excluded_cases = 0
    excluded_case_ids: set[str] = set((lift_exclusion or {}).get("case_ids") or ()) if excluded_cases else set()
    evidence_sampling: dict[str, Any] = {}
    card = {
        "id": metric,
        "label": label,
        "with_skill": ws,
        "baseline": scores.get("baseline"),
        "lift": scores.get("lift"),
        "status": "pass" if headline >= 0.8 else ("warn" if headline >= 0.6 else "fail"),
        "evidence": _metric_evidence(
            metric, rewards, report_budget, evidence_sampling, lift_excluded_case_ids=excluded_case_ids
        ),
    }
    if evidence_sampling:
        card["evidence_sampling"] = evidence_sampling
    if excluded_cases:
        card["comparison_basis"] = _card_comparison_basis(
            metric,
            excluded_cases,
            excluded_case_ids,
            lift_exclusion,
            lift_basis,
            arm_scores=scores.get("with_skill") is not None and scores.get("baseline") is not None,
        )
    return card


def _card_comparison_basis(
    metric: str,
    excluded_cases: int,
    case_ids: set[str],
    lift_exclusion: dict[str, Any] | None,
    lift_basis: dict[str, Any] | None,
    *,
    arm_scores: bool,
) -> dict[str, Any]:
    """Explain which cases an evaluator card's lift left out while its arm scores keep them.

    ``excluded_cases`` is the comparison's own count. ``case_ids`` lists the
    lift's keys for the cases the loaded rewards identify (case IDs without an
    attempt suffix), which can be fewer, for example for a saved run whose
    rewards were not loaded. The with-skill attempt counts are both ``None``
    unless both arms' rewards loaded completely and identify every left-out
    case: ``excluded_attempts`` scored the metric on those cases, and
    ``contributing_attempts`` fed the metric's lift. ``arm_scores`` says whether
    both arm scores exist for the reason to point at.
    """
    exclusion = lift_exclusion or {}
    # The lift basis is published whenever both arms' reward loads report complete,
    # even when a logical trial named two cases and left the comparison short.
    complete = exclusion.get("complete") is True and isinstance(lift_basis, dict) and len(case_ids) == excluded_cases
    metrics_basis = lift_basis.get("metrics") if complete else None
    metric_basis = metrics_basis.get(metric) if isinstance(metrics_basis, dict) else None
    trials = metric_basis.get("trials") if isinstance(metric_basis, dict) else None
    contributing = trials.get("with_skill") if isinstance(trials, dict) else None
    excluded_attempts = exclusion.get("excluded_attempts") if complete else None
    if type(contributing) is not int or type(excluded_attempts) is not int:
        contributing = excluded_attempts = None
    reason = "Cases whose arms were scored by different judge models are left out of this metric's lift"
    return {
        "excluded_cases": excluded_cases,
        # Cutting can make two long keys equal; each is listed once.
        "case_ids": sorted({case_id[:_MAX_LIFT_EXCLUDED_CASE_ID_CHARS] for case_id in case_ids})[
            :_MAX_LIFT_EXCLUDED_CASE_IDS_PER_CARD
        ],
        "excluded_attempts": excluded_attempts,
        "contributing_attempts": contributing,
        "reason": f"{reason}; the arm scores include them." if arm_scores else f"{reason}.",
    }


def _evaluator_cards(
    evaluators: dict[str, dict[str, Any]],
    *,
    rewards: list[dict[str, Any]],
    custom_with_skill: dict[str, Any],
    custom_without_skill: dict[str, Any],
    custom_lift: dict[str, Any],
    report_budget: _ReportBudget,
    baseline_only: bool = False,
    lift_exclusions: dict[str, dict[str, Any]] | None = None,
    lift_basis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from skillevaluator.tier3.harbor.metrics import METRIC_DISPLAY

    cards: list[dict[str, Any]] = []
    # Cards cover scored evaluators; a baseline-only run is scored by its baseline arm.
    headline_field = "baseline" if baseline_only else "with_skill"
    evaluator_items = [
        (metric, values)
        for metric, values in evaluators.items()
        if _finite_float(values.get(headline_field)) is not None
    ]
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
                baseline_only=baseline_only,
                lift_exclusion=(lift_exclusions or {}).get(metric),
                lift_basis=lift_basis,
            )
        )

    custom_scores = _discover_custom_metric_scores(
        custom_without_skill if baseline_only else custom_with_skill,
        rewards,
        set(evaluators),
        report_budget.cards_remaining,
        report_budget,
    )
    for metric, score in custom_scores.items():
        report_budget.cards_remaining -= 1
        if baseline_only:
            cards.append(
                _evaluator_card(
                    metric,
                    {"with_skill": None, "baseline": score, "lift": None},
                    label=f"Custom: {metric}",
                    rewards=rewards,
                    report_budget=report_budget,
                    baseline_only=True,
                )
            )
            continue
        with_skill = score
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


def _correct_answer_arm(value: object) -> dict[str, int | None]:
    summary = value.get("correct_answers") if isinstance(value, dict) else None
    summary = summary if isinstance(summary, dict) else {}

    def count(name: str) -> int | None:
        number = summary.get(name)
        return number if isinstance(number, int) and not isinstance(number, bool) and number >= 0 else None

    return {"correct": count("correct"), "scored": count("scored")}


def _correct_answer_comparison(info: dict[str, Any]) -> dict[str, Any]:
    return _count_comparison(info, "correct_answers", numerator="correct")


def _count_arm(value: object, key: str, *, numerator: str) -> dict[str, int | None]:
    summary = value.get(key) if isinstance(value, dict) else None
    summary = summary if isinstance(summary, dict) else {}

    def count(name: str) -> int | None:
        number = summary.get(name)
        return number if isinstance(number, int) and not isinstance(number, bool) and number >= 0 else None

    return {numerator: count(numerator), "scored": count("scored")}


def _count_comparison(info: dict[str, Any], key: str, *, numerator: str) -> dict[str, Any]:
    with_skill = _count_arm(info.get("pass_with_skill"), key, numerator=numerator)
    without_skill = _count_arm(info.get("pass_without_skill"), key, numerator=numerator)
    difference = None
    if (
        with_skill[numerator] is not None
        and with_skill["scored"]
        and without_skill[numerator] is not None
        and without_skill["scored"]
    ):
        if with_skill["scored"] == without_skill["scored"]:
            difference = with_skill[numerator] - without_skill[numerator]
        else:
            difference = round(
                100.0
                * (with_skill[numerator] / with_skill["scored"] - without_skill[numerator] / without_skill["scored"]),
                1,
            )
    return {
        "with_skill": with_skill,
        "without_skill": without_skill,
        "difference": difference,
        "difference_unit": (
            "percentage_points"
            if difference is not None and with_skill["scored"] != without_skill["scored"]
            else "count"
        ),
    }


def _required_check_comparison(info: dict[str, Any]) -> dict[str, Any]:
    comparison = _count_comparison(info, "required_checks", numerator="passed")
    for arm_name, source_name in (("with_skill", "pass_with_skill"), ("without_skill", "pass_without_skill")):
        arm = info.get(source_name)
        checks = arm.get("required_checks") if isinstance(arm, dict) else None
        checks = checks if isinstance(checks, dict) else {}
        comparison[arm_name].update(
            wrong_answer=checks.get("wrong_answer", 0),
            security=checks.get("security", 0),
        )
    return comparison


def _security_clean_comparison(info: dict[str, Any]) -> dict[str, Any]:
    return _count_comparison(info, "security_clean", numerator="clean")


# ---------------------------------------------------------------------------
# Trials (Trials tab)
# ---------------------------------------------------------------------------


def _normalize_trials(
    rewards: list[dict[str, Any]],
    metrics: list[str],
    *,
    pass_threshold: float = 0.5,
) -> list[dict[str, Any]]:
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
    from skillevaluator.tier3.attempt_verdict import attempt_verdict
    from skillevaluator.tier3.harbor.metrics import (
        DEFAULT_METRIC_SET,
        LEGACY_METRIC_SET,
        metric_is_not_applicable,
        metric_set_for_reward,
        metric_set_for_rewards,
        metric_value,
        overall_score,
    )
    from skillevaluator.tier3.harbor.report_data import _token_counter_names, logical_trial_reward_groups
    from skillevaluator.tier3.harbor.resource_usage import normalized_trial_timing

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
        verdict = attempt_verdict(trial["overall"], pass_threshold, reward=reward)
        trial["attempt_verdict"] = {
            "passed": verdict.passed,
            "reason": verdict.reason,
            "warnings": list(verdict.warnings),
        }
        from skillevaluator.tier3.harbor.metrics import not_applicable_metrics

        trial["not_applicable_metrics"] = not_applicable_metrics(group, metrics)
        # One extraction point for the drill-down tooltips of both arms.
        score_reasons = _trial_score_reasons(standard_rows, scores)
        if score_reasons:
            trial["score_reasons"] = score_reasons
        seconds = _finite_float(reward.get("_agent_seconds"))
        if not is_multi_row and seconds is not None and seconds >= 0:
            trial["agent_seconds"] = seconds
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
            if invalid_counters := _token_counter_names(reward.get("_invalid_token_counters")):
                # Names only: token usage counts the attempt as unmeasured, not as a zero cache.
                trial["invalid_token_counters"] = invalid_counters
            timing = normalized_trial_timing(traj.get("timing"))
            if timing is not None:
                trial["timing"] = timing
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
            and (
                metric_value(reward, "skill_execution") is not None
                or metric_is_not_applicable(reward, "skill_execution")
            )
        )
        if (
            not is_multi_row
            and reward.get("invocation_evidence_scope") != "step"
            and is_standard_reward
            and reward.get("invocation_evidence_source") == "trajectory"
        ):
            for key in ("skill_invoked", "routing_passed"):
                if type(reward.get(key)) is bool:
                    trial[key] = reward[key]
            if "skill_invoked" in trial or "routing_passed" in trial:
                trial["invocation_evidence_source"] = "trajectory"
        harbor_viewer = _normalize_harbor_viewer_metadata(reward.get("harbor_viewer"))
        if harbor_viewer:
            trial["harbor_viewer"] = harbor_viewer
        trial["activation_validation"] = _trial_activation_validation(group, trial)
        out.append(trial)
    return out


def _trusted_observed_activation(reward: dict[str, Any]) -> str:
    """Project collector-issued invocation evidence; never inspect trajectory or scores as proof."""
    from skillevaluator.tier3.harbor.metrics import (
        DEFAULT_METRIC_SET,
        LEGACY_METRIC_SET,
        metric_is_not_applicable,
        metric_set_for_reward,
        metric_value,
    )

    standard_sets = {DEFAULT_METRIC_SET, LEGACY_METRIC_SET}
    declared = reward.get("metric_set") or reward.get("metric_set_version")
    metric_set, metrics = metric_set_for_reward(reward)
    if (
        (declared and str(declared) not in standard_sets)
        or metric_set not in standard_sets
        or "skill_execution" not in metrics
        or (metric_value(reward, "skill_execution") is None and not metric_is_not_applicable(reward, "skill_execution"))
        or reward.get("invocation_evidence_source") != "trajectory"
    ):
        return "unknown"
    invoked = reward.get("skill_invoked")
    return ("invoked" if invoked else "not_invoked") if type(invoked) is bool else "unknown"


def _trial_activation_validation(group: list[dict[str, Any]], trial: dict[str, Any]) -> dict[str, Any]:
    """Keep routing, task quality, workflow and resources at their observed scopes.

    Fallback reward rows do not prove a complete logical trajectory. Even when
    every retained row says non-use, omitted steps may have used the skill.
    Only a collector-produced single/merged root can certify logical non-use.
    """
    scopes: list[dict[str, Any]] = []
    for index, reward in enumerate(group[:_MAX_EVIDENCE_PER_CARD]):
        scope: dict[str, Any] = {"reward_row": index, "observed_activation": _trusted_observed_activation(reward)}
        artifacts = reward.get("_activation_artifacts")
        if isinstance(artifacts, dict):
            scope["artifacts"] = {
                key: value
                for key, value in artifacts.items()
                if key in {"reward", "trajectory"}
                and isinstance(value, str)
                and len(value) <= 4096
                and not Path(value).is_absolute()
                and ".." not in Path(value).parts
            }
        details = reward.get("details")
        execution = details.get("skill_execution") if isinstance(details, dict) else None
        workflow: dict[str, Any] = {}
        if isinstance(execution, dict):
            for key in ("workflow_order", "script_execution", "error_recovery"):
                check = execution.get(key)
                if isinstance(check, dict):
                    workflow[key] = {
                        "passed": check.get("passed") if type(check.get("passed")) is bool else None,
                        "score": _unit_interval_score(check.get("score")),
                    }
        scope["workflow"] = workflow
        scopes.append(scope)
    logical_scope = len(group) == 1 and group[0].get("invocation_evidence_scope") == "logical_trial"
    return {
        "expected_activation": None,
        "observed_activation": _trusted_observed_activation(group[0]) if logical_scope else "unknown",
        "expectation_status": "unknown",
        "observation_scope": "logical_trial" if logical_scope else "reward_rows_only",
        "evidence_scopes": scopes,
        "evidence_scopes_complete": len(group) <= _MAX_EVIDENCE_PER_CARD,
        "task_outcome": {key: trial["scores"].get(key) for key in ("accuracy", "goal_accuracy")},
        "resources": {key: trial[key] for key in ("steps", "tokens", "agent_seconds", "timing") if key in trial},
    }


def _attach_activation_expectations(
    agents: dict[str, dict[str, Any]],
    dataset: list[dict[str, Any]],
    skill_name: str,
) -> None:
    """Compare authored expectations with observed use, independently of correctness."""
    from skillevaluator.tier3.eval_core.checks import resolve_acceptable_skills, resolve_should_trigger

    entries: dict[str, list[dict[str, Any]]] = {}
    for entry in dataset:
        if isinstance(entry, dict) and entry.get("id") is not None:
            entries.setdefault(str(entry["id"]), []).append(entry)
    for agent in agents.values():
        for arm, baseline in (("trials", False), ("trials_baseline", True)):
            for trial in agent.get(arm, []):
                expected: bool | None = False if baseline else None
                matches = entries.get(str(trial.get("entry_id")), [])
                if not baseline and len(matches) == 1:
                    entry = matches[0]
                    should_trigger = resolve_should_trigger(entry)
                    if should_trigger is False:
                        expected = False
                    elif should_trigger is True:
                        expected_skill = entry.get("expected_skill")
                        if isinstance(expected_skill, str) and expected_skill.strip():
                            target = skill_name.strip().lower()
                            if expected_skill.strip().lower() == target:
                                expected = True
                            elif target not in {name.lower() for name in resolve_acceptable_skills(entry)}:
                                expected = False
                            # Allowed alternate use is optional, so neither use nor non-use is required.
                validation = trial["activation_validation"]
                validation["expected_activation"] = expected
                observed = validation["observed_activation"]
                if expected is not None and observed != "unknown":
                    validation["expectation_status"] = (
                        "matched" if (observed == "invoked") == expected else "mismatched"
                    )


def _trial_score_reasons(
    rows: list[tuple[dict[str, Any], tuple[str, ...]]],
    scores: dict[str, float],
) -> dict[str, str]:
    """Return one bounded grader reason per failing metric of a logical trial.

    A logical trial can span several reward rows. The reason comes from the
    lowest-scoring row that explains the metric, so a passing row cannot hide a
    failure behind the averaged score.
    """
    from skillevaluator.tier3.harbor.metrics import metric_value

    reasons: dict[str, str] = {}
    for metric in scores:
        lowest: tuple[float, str] | None = None
        for row, row_metrics in rows:
            if metric not in row_metrics:
                continue
            row_score = metric_value(row, metric)
            if row_score is None or row_score >= TRIAL_PASS_THRESHOLD:
                continue
            if lowest is not None and row_score >= lowest[0]:
                continue
            reason = failure_reason(_metric_detail(row, metric), metric=metric, reward=row)
            if reason:
                lowest = (row_score, reason)
        if lowest is not None:
            reasons[metric] = lowest[1]
    return reasons


def _attach_baseline_pairs(
    trials: list[dict[str, Any]],
    baseline_trials: list[dict[str, Any]],
    metrics: list[str],
    *,
    comparable_overall: bool = True,
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
        if comparable_overall:
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
    *,
    start: int = 0,
) -> list[dict[str, Any]]:
    if not recommendations or not evidence_links:
        return recommendations

    linked: list[dict[str, Any]] = []
    for index, recommendation in enumerate(recommendations, start=start):
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


def _is_case_verdict(conclusion: object) -> bool:
    """Whether a conclusion is the per-case verdict that ``_apply_case_findings`` adds (the report headline)."""
    return isinstance(conclusion, dict) and conclusion.get("title") == "Skill effect"


def _saved_case_fixes(run_dir: Path) -> list[dict[str, Any]] | None:
    """The run's ``what_to_fix.json`` when the report can show it as is, else ``None``."""
    from skillevaluator.tier3.case_summary import load_case_findings

    fixes = load_case_findings(run_dir)
    if fixes is None or any(
        not all(isinstance(fix.get(key), str) for key in ("case", "scope", "kind", "what"))
        or not all(isinstance(fix.get(key) or "", str) for key in ("action", "evidence"))
        for fix in fixes
    ):
        return None
    return fixes


def _apply_saved_feedback(
    payload: dict[str, Any],
    saved_feedback: dict[str, list[Any]],
    engine_result: dict[str, Any] | None,
    run_dir: Path,
    skill_path: Path,
) -> None:
    """Show the report feedback a completed run saved, with no LLM call.

    The case-findings pass still runs without the LLM, on the run's own
    ``what_to_fix.json`` when it is usable (otherwise it writes the rule-based
    fixes), so what it adds beyond the saved fields stays. Saved conclusions
    that carry the per-case verdict replace the pass's verdict and fixes, so
    rule-based fixes written then can differ from the saved recommendations;
    feedback saved before the verdict existed gets both added, as a fresh
    render shows them.
    """
    saved = {field: list(saved_feedback[field]) for field in _TIER3_FEEDBACK_FIELDS if field in saved_feedback}
    fixes = _saved_case_fixes(run_dir)
    if any(_is_case_verdict(item) for item in saved.get("conclusions", [])):
        _apply_case_findings(payload, engine_result, run_dir, skill_path, use_llm=False, fixes=fixes)
        payload.update(saved)
    else:
        payload.update(saved)
        _apply_case_findings(payload, engine_result, run_dir, skill_path, use_llm=False, fixes=fixes)


def _apply_case_findings(
    payload: dict[str, Any],
    engine_result: dict[str, Any] | None,
    run_dir: Path,
    skill_path: Path,
    *,
    use_llm: bool,
    fixes: list[dict[str, Any]] | None = None,
) -> None:
    """Lead the report with the case-based verdict and fixes; also saves them as what_to_fix.json for the CLI.

    ``fixes`` already saved for the run are reused as they are and not written again.
    """
    from skillevaluator.tier3.case_summary import goal_not_met_count, save_case_findings, what_to_fix
    from skillevaluator.tier3.case_summary import verdict as case_verdict

    source = engine_result
    if not isinstance(source, dict):
        with contextlib.suppress(OSError, ValueError):
            source = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("agents"), dict):
        return
    try:
        if fixes is None:
            fixes = what_to_fix(source, run_dir=run_dir, skill_path=skill_path, use_llm=use_llm)
            save_case_findings(run_dir, fixes)
    except Exception:  # advisory: the report keeps its metric-based insights
        logger.debug("Case findings skipped", exc_info=True)
        return
    label, detail = case_verdict(source["agents"])
    if not label:
        return
    message = f"per-case result: {label} ({detail})."
    agents = payload.get("agents") if isinstance(payload.get("agents"), dict) else {}
    scores = {
        n: v for n, a in agents.items() if isinstance(a, dict) and (v := _finite_float(a.get("with_skill"))) is not None
    }
    lifts = {n: v for n, a in agents.items() if isinstance(a, dict) and (v := _finite_float(a.get("lift"))) is not None}
    if len(scores) > 1 and lifts:
        top, lifted = max(scores, key=scores.__getitem__), max(lifts, key=lifts.__getitem__)
        message += f" Highest score: {top} {scores[top]:.2f}"
        message += (
            f" · biggest skill lift: {lifted} {lifts[lifted]:+.2f}." if lifted != top else " with the biggest lift."
        )
    if missed := goal_not_met_count(source, run_dir=run_dir):
        message += f" {missed} passing with-skill attempt(s) did not meet the goal (goal accuracy 0.00)."
    severity = "pass" if label == "skill helps" and not fixes else "fail" if label == "skill hurts" else "warn"
    effect = {"severity": severity, "title": "Skill effect", "message": message, "source": "deterministic"}
    payload["conclusions"] = [effect, *(payload.get("conclusions") or [])]
    # Evidence stays plain text: report links are limited to validated Harbor viewer URLs.
    payload["recommendations"] = [
        {
            "title": f"{fix['case']} · {fix['scope']}",
            "category": fix["kind"],
            "message": " ".join(
                part
                for part in (
                    f"{fix['what'].rstrip('.')}.",
                    f"Fix: {fix['action']}" if fix.get("action") else "",
                    f"Evidence: {fix['evidence']}" if fix.get("evidence") else "",
                )
                if part
            ),
            "severity": "fail" if fix["kind"] == "skill hurts" else "warn",
            "source": "deterministic",
        }
        for fix in fixes
    ]
    payload["suggestions"] = [fix["action"] for fix in fixes if fix.get("action")]
    from skillevaluator.tier3.next_steps import next_steps

    with contextlib.suppress(Exception):  # advisory, like the fixes above
        payload["next_steps"] = next_steps(source, run_dir=run_dir, fixes=fixes)


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
            message = f"{best_name} leads with overall score {best_score:.2f}" + (
                f" and lift {lift:+.2f}." if lift is not None else "."
            )
            lifts = {name: value for name, a in agents.items() if (value := _finite_float(a.get("lift"))) is not None}
            top_lift = max(lifts, key=lifts.__getitem__) if lifts else None
            if top_lift and top_lift != best_name and lift is not None and lifts[top_lift] > lift:
                message += f" Biggest skill lift: {top_lift} {lifts[top_lift]:+.2f}."
            conclusions.append(
                {
                    "severity": "pass" if best_score >= pass_threshold else "warn",
                    "title": "Best performing agent",
                    "message": message,
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
    unscored_trials: list[str] = []
    for agent_name, agent in agents.items():
        for trial in agent.get("trials") or []:
            overall = _finite_float(trial.get("overall"))
            trial_label = f"{agent_name}/{trial.get('entry_id') or trial.get('trial_id')}"
            attempt = trial.get("attempt_verdict")
            attempt_passed = attempt.get("passed") if isinstance(attempt, dict) else None
            attempt_reason = attempt.get("reason") if isinstance(attempt, dict) else None
            if attempt_passed is False:
                if overall is None and attempt_reason == "overall score was not available":
                    unscored_trials.append(trial_label)
                else:
                    failing_trials.append(trial_label)
            elif overall is None:
                unscored_trials.append(trial_label)
            elif overall < pass_threshold:
                failing_trials.append(trial_label)
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
    if unscored_trials:
        conclusions.append(
            {
                "severity": "warn",
                "title": "Unscored trials",
                "message": (
                    f"{len(unscored_trials)} trial(s) have unavailable overall scores; examples: "
                    + ", ".join(unscored_trials[:5])
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
    except (TypeError, ValueError, OverflowError):
        return 0.50
    return numeric if math.isfinite(numeric) else 0.50


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _dimension_score(
    scores: dict[str, Any],
    cfg: dict[str, Any],
    *,
    active_metrics: list[str] | None = None,
    not_applicable: list[str] = (),
) -> float | None:
    """Require every applicable source; only a legacy contract selects fallback sources."""
    evaluators, weights = cfg.get("evaluators", []), cfg.get("weights", [])
    if (
        cfg.get("fallback_evaluators")
        and active_metrics is not None
        and not any(evaluator in active_metrics for evaluator in evaluators)
    ):
        evaluators, weights = cfg["fallback_evaluators"], cfg.get("fallback_weights", [])
    if active_metrics is not None and any(evaluator not in active_metrics for evaluator in evaluators):
        return None
    return _weighted(scores, evaluators, weights, not_applicable=not_applicable)


def _weighted(
    scores: dict[str, Any],
    evaluators: list[str],
    weights: list[float],
    *,
    not_applicable: list[str] = (),
) -> float | None:
    numerator = denominator = 0.0
    if len(evaluators) != len(weights):
        return None
    for evaluator, weight in zip(evaluators, weights, strict=True):
        value = _unit_interval_score(scores.get(evaluator))
        if value is None and evaluator in not_applicable:
            continue
        finite_weight = _finite_float(weight)
        if value is None or finite_weight is None or finite_weight < 0:
            return None
        numerator += value * finite_weight
        denominator += finite_weight
    return numerator / denominator if denominator > 0 else None


def _precomputed_score(precomputed: dict[str, Any], dim_id: str) -> float | None:
    entry = precomputed.get(dim_id)
    if isinstance(entry, dict):
        return _finite_float(entry.get("score"))
    return None


def _lift_overall_scores(
    overall_lift: float | None,
    judge_lift: dict[str, Any] | None,
    overall_ws: float | None,
    overall_bl: float | None,
) -> dict[str, float] | None:
    """Return the two overall scores that the published overall lift subtracts.

    When a metric is N/A in one arm it drops out of both arms for lift, so these
    scores can differ from the arm overall scores.
    """
    if overall_lift is None:
        return None
    entry = judge_lift.get("overall") if isinstance(judge_lift, dict) else None
    candidates = []
    if isinstance(entry, dict):
        candidates.append(
            (_unit_interval_score(entry.get("with_skill")), _unit_interval_score(entry.get("without_skill")))
        )
    candidates.append((overall_ws, overall_bl))
    for with_score, baseline_score in candidates:
        if (
            with_score is not None
            and baseline_score is not None
            and abs((with_score - baseline_score) - overall_lift) <= 1e-4 + 1e-9
        ):
            return {"with_skill": round(with_score, 4), "baseline": round(baseline_score, 4)}
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
    """Classify one supported agent by required checks, then dimensions.

    A dimension below the neutral threshold fails the agent even when another
    dimension has no score (for example Correctness without ``ground_truth``).
    A missing or non-finite dimension still blocks PASS.
    """
    if agent.get("execution_status") != "succeeded":
        return VERDICT_NEUTRAL
    if _never_activated(agent):
        return VERDICT_INCONCLUSIVE
    from skillevaluator.tier3.attempt_verdict import required_check_failure_reason

    required_checks = agent.get("required_checks")
    required_check_arm = required_checks.get("with_skill") if isinstance(required_checks, dict) else None
    if required_check_failure_reason(required_check_arm):
        return VERDICT_FAIL
    dimensions = {
        str(dimension.get("id")): dimension
        for dimension in agent.get("dimensions") or []
        if isinstance(dimension, dict)
    }
    scores: list[float] = []
    incomplete = False
    for dimension_id in _DIMENSION_IDS:
        dimension = dimensions.get(dimension_id)
        value = (dimension or {}).get("with_skill", (dimension or {}).get("score"))
        numeric = _finite_float(value)
        if numeric is None:
            incomplete = True
        else:
            scores.append(numeric)

    if any(score < DIMENSION_VERDICT_NEUTRAL_THRESHOLD for score in scores):
        return VERDICT_FAIL
    if incomplete or any(score < DIMENSION_VERDICT_PASS_THRESHOLD for score in scores):
        return VERDICT_NEUTRAL
    return VERDICT_PASS


def _agent_quality_verdict_reason(agent: dict[str, Any]) -> str | None:
    if _never_activated(agent):
        return None
    from skillevaluator.tier3.attempt_verdict import required_check_failure_reason

    required_checks = agent.get("required_checks")
    arm = required_checks.get("with_skill") if isinstance(required_checks, dict) else None
    return required_check_failure_reason(arm)


def _overall_verdict_from_agents(agents: dict[str, dict[str, Any]]) -> str:
    """PASS only when one supported agent passes its quality requirements."""
    verdicts = [
        _agent_quality_verdict(agent) for agent in agents.values() if agent.get("execution_status") == "succeeded"
    ]
    if any(verdict == VERDICT_PASS for verdict in verdicts):
        return VERDICT_PASS
    if any(verdict == VERDICT_NEUTRAL for verdict in verdicts) or not verdicts:
        return VERDICT_NEUTRAL
    if all(verdict == VERDICT_INCONCLUSIVE for verdict in verdicts):
        return VERDICT_INCONCLUSIVE
    return VERDICT_FAIL


def _canonical_agent_rank(agent: dict[str, Any], *, score_field: str = "with_skill") -> tuple[float, float] | None:
    """Return the score/lift tuple used to rank one canonical agent payload."""
    if agent.get("execution_status") != "succeeded":
        return None
    score = _finite_float(agent.get(score_field))
    if score is None:
        return None
    return score, _as_float(agent.get("lift"))


def _agent_score_policy(info: dict[str, Any]) -> str:
    """Return the with-skill score policy used to rank one agent's raw Harbor data.

    A recorded policy wins. Artifacts without one use the policy for their
    with-skill metric set, as ``build_agent_eval_payload`` does.
    """
    from skillevaluator.tier3.harbor.metrics import score_policy_for_metrics
    from skillevaluator.tier3.harbor.report_data import metrics_for_condition

    recorded = info.get("score_policy_with_skill")
    if isinstance(recorded, str) and recorded.strip():
        return recorded.strip()
    metrics = metrics_for_condition(info, "with_skill")
    return score_policy_for_metrics(_policy_metrics_from_agents({"agent": info}, metrics))


def _canonical_agent_rank_from_info(
    info: dict[str, Any],
    score_policy: str | None = None,
) -> tuple[float, float] | None:
    """Build and rank raw Harbor agent data exactly as the canonical report does."""
    from skillevaluator.tier3.harbor.report_data import metrics_for_condition

    agent = _build_agent(
        "",
        info,
        metrics_for_condition(info, "with_skill"),
        metrics_for_condition(info, "without_skill"),
        None,
        score_policy or _agent_score_policy(info),
    )
    return _canonical_agent_rank(agent)


def _evaluation_mode(agents: dict[str, dict[str, Any]]) -> str:
    """Classify the arms a run executed: ``both``, ``with_skill_only`` or ``baseline_only``.

    Read from the collector's deliberate per-arm ``execution_status="skipped"``
    markers, never inferred from empty scores, which can also mean a failed arm.
    """

    def arm_statuses(condition: str) -> list[object]:
        statuses: list[object] = []
        for info in agents.values():
            conditions = info.get("conditions")
            arm = conditions.get(condition) if isinstance(conditions, dict) else None
            statuses.append(arm.get("execution_status") if isinstance(arm, dict) else None)
        return statuses

    with_statuses = arm_statuses("with_skill")
    without_statuses = arm_statuses("without_skill")
    if with_statuses and all(status == "skipped" for status in with_statuses):
        return "baseline_only"
    if without_statuses and all(status in {None, "skipped"} for status in without_statuses):
        return "with_skill_only"
    return "both"


def _build_baseline_only_conclusions(
    best_agent: str,
    overall_score: float,
    dimensions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Insights for a baseline-only run: the calibration result, never a lift claim."""
    conclusions = [
        {
            "severity": "pass",
            "title": "Baseline calibration complete",
            "message": (
                f"Baseline-only run (--skip-with-skill): {best_agent} scores {overall_score:.2f} without the "
                "skill. No with-skill arm was run, so these are baseline calibration scores and no skill lift "
                "was measured."
            ),
        }
    ]
    numeric_dims = [d for d in dimensions if _finite_float(d.get("baseline")) is not None]
    if numeric_dims:
        weakest = min(numeric_dims, key=lambda d: d["baseline"])
        conclusions.append(
            {
                "severity": "fail" if weakest["baseline"] < 0.4 else "warn",
                "title": "Weakest baseline dimension",
                "message": (
                    f"{weakest.get('id', 'unknown').title()} is lowest at {weakest['baseline']:.2f} without the skill."
                ),
            }
        )
    return conclusions


def _headline_agents(
    agents: dict[str, dict[str, Any]], *, score_field: str = "with_skill"
) -> dict[str, dict[str, Any]]:
    """Return the agents that set the run's best agent and verdict.

    An agent kept by --tolerate-baseline-timeouts is set aside whenever an
    agent with a complete baseline ranks, so the headline score and verdict
    both come from complete evidence.
    """
    from skillevaluator.tier3.harbor.baseline_timeouts import kept_with_failed_baseline

    complete = {name: agent for name, agent in agents.items() if not kept_with_failed_baseline(agent)}
    if any(_canonical_agent_rank(agent, score_field=score_field) is not None for agent in complete.values()):
        return complete
    return agents


def _pick_best_agent(agents: dict[str, dict[str, Any]], *, score_field: str = "with_skill") -> str:
    eligible = {
        name: rank
        for name, agent in _headline_agents(agents, score_field=score_field).items()
        if (rank := _canonical_agent_rank(agent, score_field=score_field)) is not None
    }
    activated = {name: rank for name, rank in eligible.items() if not _never_activated(agents[name])}
    eligible = activated or eligible
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


def _case_selection(run_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the recorded ``max_cases`` counts so reports can flag a partial run."""
    if not isinstance(run_config, dict) or "case_selection" not in run_config:
        return None
    selection = run_config["case_selection"]
    if not isinstance(selection, dict):
        selection = {}
    return {key: selection.get(key) for key in ("policy", "limit", "total", "selected", "skipped")}


def _read_attempt_policy(run_dir: Path) -> dict[str, Any]:
    policy = _default_attempt_policy()
    policy_file = run_dir / "attempt_policy.json"
    if policy_file.exists():
        with contextlib.suppress(OSError, ValueError):
            loaded = json.loads(policy_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                policy.update(loaded)
                if "score_policy" not in loaded:
                    policy.pop("score_policy", None)
                if "score_definition" not in loaded:
                    policy.pop("score_definition", None)
                return policy
    policy.pop("score_definition", None)
    policy.pop("score_policy", None)
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
    result_file_present = result_file.exists()
    if result_file_present:
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
        if "dataset_provenance" not in truth:
            from skillevaluator.tier3.dataset_provenance import sanitize_generation_record

            provenance = sanitize_generation_record(candidate.get("dataset_provenance"))
            if provenance is not None:
                truth["dataset_provenance"] = provenance
        for field_name in ("evaluator_version", "dataset_digest", "dataset_digest_algorithm"):
            value = candidate.get(field_name)
            if field_name not in truth and isinstance(value, str) and value.strip():
                truth[field_name] = value.strip()

    # Publication identity is one run-owned pair. A completed ``result.json``
    # is authoritative for rerenders; the in-memory engine result is used only
    # while the runner is rendering before that final artifact exists.
    identity_source = persisted_result if result_file_present else engine_result
    if isinstance(identity_source, dict):
        run_id = identity_source.get("run_id")
        publication_target = identity_source.get("publication_target")
        if isinstance(run_id, str) and run_id == run_dir.name:
            truth["run_id"] = run_id
            if "publication_target_conflict" in identity_source:
                conflict = identity_source.get("publication_target_conflict")
                marker = (
                    _RUNTIME_CONTEXT_CONFLICT_MARKER
                    if isinstance(conflict, dict)
                    and conflict.get("reason_code") == _RUNTIME_CONTEXT_CONFLICT_REASON_CODE
                    else _PUBLICATION_TARGET_CONFLICT_MARKER
                )
                truth["publication_target_conflict"] = marker
            elif isinstance(publication_target, dict):
                truth["publication_target"] = dict(publication_target)
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
    for agent_name, info in agents.items():
        if _activation_summary(info).get("status") == "never_activated":
            continue
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
        "score_policy": DEFAULT_SCORE_POLICY,
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
    from skillevaluator.tier3.attempt_verdict import (
        DEFAULT_GOAL_ACCURACY_PASS_THRESHOLD,
        DEFAULT_SECURITY_PASS_THRESHOLD,
        RUN_FAILS_ON_ANY_FAILED_CASE,
    )

    attempt_threshold = attempt_policy.get("pass_threshold")
    return {
        "attempt_pass_threshold": _finite_float(attempt_threshold),
        "goal_accuracy_pass_threshold": DEFAULT_GOAL_ACCURACY_PASS_THRESHOLD,
        "security_pass_threshold": DEFAULT_SECURITY_PASS_THRESHOLD,
        "attempt_pass_rule": "overall_threshold_and_goal_accuracy_and_security_threshold_when_scored",
        "run_fail_policy": "any_with_skill_required_check_failure"
        if RUN_FAILS_ON_ANY_FAILED_CASE
        else "dimension_gate",
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
]
