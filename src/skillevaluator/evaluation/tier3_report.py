# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map native Harbor (Tier 3) results into the canonical ``agent_eval`` payload.

Skill Evaluator folds Tier 3 into the *combined* ``validate`` report (HTML / JSON /
BENCHMARK.md) by attaching a canonical ``metadata["agent_eval"]`` payload to a
single ``AGENT_EVAL`` :class:`~skillevaluator.models.result.ValidationResult`. The
shared reporters (ported faithfully from Skill Evaluator) consume that payload.

SkillEvaluator runs Tier 3 through its own in-process Harbor engine, which writes
per-agent results to disk rather than returning a canonical payload. This module
reads those on-disk results and produces the same canonical ``agent_eval`` shape
so ``validate --agent-eval`` emits one combined report containing all three tiers
-- restoring parity with Skill Evaluator.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from skillevaluator.constants import (
    AGENT_EVAL_EVALUATORS,
    AGENT_EVAL_SCORE_DEFINITION,
    DIMENSION_HINTS,
    DIMENSION_MAPPING,
)
from skillevaluator.models.result import ValidationResult

# Verdict labels mirror Skill Evaluator's AGENT_EVAL_VERDICT_* values so the ported
# reporters classify the overall outcome identically.
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_NEUTRAL = "neutral"

# Lift thresholds mirror Skill Evaluator TIER3_LIFT_PASS_THRESHOLD / _FAIL_THRESHOLD.
_VERDICT_PASS_THRESHOLD = 0.05
_VERDICT_FAIL_THRESHOLD = -0.05

_AGENT_EVAL_VALIDATOR = "AGENT_EVAL"
_AGENT_EVAL_DESCRIPTION = "Tier 3: Live Agent Evaluation (Harbor)"

_DIMENSION_IDS = list(DIMENSION_MAPPING.keys())

_SCHEMA_VERSION = "2.0"
_TIER3_FEEDBACK_SCHEMA_VERSION = "1.0"
_TIER3_FEEDBACK_FIELDS = ("conclusions", "recommendations", "suggestions", "suggestions_v2")


def _advisory_agent_eval_payload(message: str, *, skill_name: str | None = None) -> dict[str, Any]:
    """Build the canonical (but empty) ``agent_eval`` payload for a skipped Tier 3 run.

    The combined HTML/JSON report only renders a Tier 3 section when some
    result carries ``metadata["agent_eval"]`` (``HTMLReporter`` keys off it and
    the template gates the Tier 3 tab/card on ``has_tier3``). Attaching this
    minimal payload — verdict ``neutral``, no agents/dimensions, and the skip
    reason surfaced via ``suggestions`` + ``provenance`` — guarantees an
    explicit ``--agent-eval`` request always produces a visible, self-explaining
    Tier 3 section instead of silently disappearing. Mirrors Skill Evaluator, which
    always emits an ``AGENT_EVAL`` result with a payload (e.g.
    ``_tier3_dataset_required_result`` / ``_invalid_skill_evaluator_result``) even when the
    dataset/runtime is unavailable.
    """
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
        "attempt_policy": _default_attempt_policy(),
        "dataset": [],
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
    use_llm_judge: bool = True,
) -> ValidationResult | None:
    """Build the canonical ``AGENT_EVAL`` result for one explicit Harbor run."""
    # Imported lazily so base-only Tier 1 workflows do not load Tier 3 helpers.
    from skillevaluator.tier3.harbor.report_data import (
        load_agent_data,
        load_dataset,
        load_staged_harbor_dataset,
    )

    run_dir = run_dir.expanduser().resolve()
    agents = load_agent_data(run_dir)
    if not agents:
        return None

    dataset = load_dataset(skill_path) or load_staged_harbor_dataset(run_dir)
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
    if payload.get("execution_status") == "succeeded" and isinstance(payload.get("overall_score"), int | float):
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
    use_llm_judge: bool = True,
) -> dict[str, Any] | None:
    """Assemble the canonical Tier 3 ``agent_eval`` payload from loaded agent data.

    ``agents`` is the structure produced by
    :func:`skillevaluator.tier3.harbor.report_data.load_agent_data`.
    Returns ``None`` when no agent carries usable scores.

    The payload mirrors Skill Evaluator's canonical Tier 3 shape so the ported reporters
    render every Tier 3 sub-tab: per-trial data (``trials`` / per-agent
    ``trials`` + ``pass_at_k``) feeds the Trials tab, deterministic + LLM
    ``conclusions`` / ``recommendations`` / ``suggestions`` feed the Insights tab,
    and ``provenance`` (raw evaluators, raw lift, raw trial rewards) feeds the
    Diagnostics tab.
    """
    from skillevaluator.tier3.harbor.report_data import metrics_for_agents

    metrics = metrics_for_agents(agents)
    agent_payloads: dict[str, dict[str, Any]] = {}
    for name in sorted(agents):
        info = agents[name]
        model = _agent_model(name, info, run_config)
        agent_payloads[name] = _build_agent(name, info, metrics, model)

    if not agent_payloads:
        return None

    best_agent = _pick_best_agent(agent_payloads)
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
    overall_score = (
        float(raw_overall_score)
        if execution_status == "succeeded"
        and isinstance(raw_overall_score, int | float)
        and not isinstance(raw_overall_score, bool)
        else None
    )
    overall_lift = best.get("lift")
    verdict = _verdict_from_lift(overall_lift) if overall_score is not None else VERDICT_NEUTRAL

    metric_ids = list(best.get("evaluators", {}).keys())
    metric_labels = _metric_labels(metric_ids)

    policy = attempt_policy or _default_attempt_policy()
    canonical_trials = _flatten_trials(agent_payloads)
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
        "overall_lift": (round(overall_lift, 4) if isinstance(overall_lift, (int, float)) else None),
        "environment": env_mode,
        "runtime_seconds": float(runtime_seconds or 0.0),
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
        "composite_lift": round(overall_lift, 4) if isinstance(overall_lift, (int, float)) else None,
        "execution_status": execution_status,
        "execution_errors": execution_errors,
        "expected_attempts": summary["expected_attempts"],
        "scored_attempts": summary["scored_attempts"],
        "runtime_seconds": float(runtime_seconds or 0.0),
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
        "dataset": [d for d in (dataset or []) if isinstance(d, dict)],
        "provenance": _build_provenance(agent_payloads, agents, run_dir, comparison),
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
        payload["recommendations"] = _attach_harbor_evidence_to_recommendations(
            payload.get("recommendations") or [],
            evidence_links,
        )
        payload["suggestions_v2"] = _attach_harbor_evidence_to_suggestions_v2(
            payload.get("suggestions_v2") or [],
            evidence_links,
        )
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
    deterministic content is preserved unchanged (Skill Evaluator parity).
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
) -> dict[str, Any]:
    """Assemble the Diagnostics ``provenance`` block.

    Mirrors Skill Evaluator's Harbor provenance: per-agent raw evaluator scores and lift
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
        "raw_trial_rewards": {name: _raw_trial_rewards(raw_agents.get(name, {})) for name in agent_payloads},
        "evaluator_paths": {},
        "comparison": comparison if isinstance(comparison, dict) else {},
    }


# Verbose per-evaluator ``details`` (evidence refs, per-check breakdowns) are
# dropped from the diagnostics payload: they are not rendered by any report
# panel and would multiply the embedded JSON size several-fold. The full
# details remain on disk under ``provenance.run_dir`` for deep dives.
_REWARD_HEAVY_KEYS = frozenset({"details"})


def _raw_trial_rewards(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact raw Harbor reward dicts (internal + verbose keys stripped)."""
    rewards: list[dict[str, Any]] = []
    for reward in info.get("rewards") or []:
        if isinstance(reward, dict):
            rewards.append({k: v for k, v in reward.items() if not k.startswith("_") and k not in _REWARD_HEAVY_KEYS})
    return rewards


# ---------------------------------------------------------------------------
# Per-agent assembly
# ---------------------------------------------------------------------------


def _build_agent(
    name: str,
    info: dict[str, Any],
    metrics: list[str],
    model: str | None,
) -> dict[str, Any]:
    with_scores = info.get("with_skill") or {}
    without_scores = info.get("without_skill") or {}
    lift_data = info.get("lift") or {}

    evaluators = _build_evaluators(metrics, with_scores, without_scores, lift_data)
    dimensions = _build_dimensions(
        with_scores,
        without_scores,
        info.get("dimensions_with_skill") or {},
        info.get("dimensions_without_skill") or {},
    )
    overall_ws = _mean([d["with_skill"] for d in dimensions if isinstance(d.get("with_skill"), (int, float))])
    overall_bl = _mean([d["baseline"] for d in dimensions if isinstance(d.get("baseline"), (int, float))])
    if overall_ws is None and not metrics:
        overall_ws = _mean([reward.get("overall") for reward in info.get("rewards", []) if isinstance(reward, dict)])
    if overall_bl is None and not metrics:
        overall_bl = _mean(
            [reward.get("overall") for reward in info.get("rewards_baseline", []) if isinstance(reward, dict)]
        )
    overall_lift = (
        round(overall_ws - overall_bl, 4)
        if isinstance(overall_ws, (int, float)) and isinstance(overall_bl, (int, float))
        else None
    )

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
        "evaluator_cards": _evaluator_cards(evaluators),
        "dimensions": dimensions,
        "with_skill": overall_ws,
        "baseline": overall_bl,
        "lift": overall_lift,
        "num_trials": int(info.get("num_trials", 0) or 0),
        "num_trials_baseline": len(baseline_trials),
        "trials": trials,
        "trials_baseline": baseline_trials,
        "pass_at_k": {
            "with_skill": info.get("pass_with_skill") or {},
            "without_skill": info.get("pass_without_skill") or {},
            "lift": info.get("pass_lift") or {},
        },
        "cases": _cases(info),
    }


def _build_evaluators(
    metrics: list[str],
    with_scores: dict[str, Any],
    without_scores: dict[str, Any],
    lift_data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    evaluators: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        ws = with_scores.get(metric)
        if not isinstance(ws, (int, float)) or isinstance(ws, bool):
            continue
        bl = without_scores.get(metric)
        bl = float(bl) if isinstance(bl, (int, float)) and not isinstance(bl, bool) else None
        lift = _lift_value(metric, lift_data)
        if lift is None and bl is not None:
            lift = round(float(ws) - bl, 4)
        evaluators[metric] = {
            "with_skill": float(ws),
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
        lift = round(ws - bl, 4) if isinstance(ws, (int, float)) and isinstance(bl, (int, float)) else None
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
                "with_skill": round(ws, 4) if isinstance(ws, (int, float)) else None,
                "score": round(ws, 4) if isinstance(ws, (int, float)) else None,
                "baseline": round(bl, 4) if isinstance(bl, (int, float)) else None,
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
    """Build deterministic reasoning bullets for a dimension (Skill Evaluator parity).

    Reuses the ported dimension-judge helper so the Reasoning column reads
    identically to Skill Evaluator when no LLM explanation is available.
    """
    from skillevaluator.evaluation.dimension_judge import _human_reasoning_bullets

    parts: list[str] = []
    for signal in signals:
        value = with_scores.get(signal)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{signal}={float(value):.2f}")
    bullets = _human_reasoning_bullets(
        with_skill=float(ws) if isinstance(ws, (int, float)) else 0.0,
        baseline=bl if isinstance(bl, (int, float)) else None,
        lift=lift,
        parts=parts,
    )
    return bullets, " ".join(bullets)


def _deterministic_verdict(ws: float | None) -> str | None:
    """Deterministic PASS/NEUTRAL/FAIL verdict for a dimension score."""
    if not isinstance(ws, (int, float)):
        return None
    from skillevaluator.evaluation.dimension_judge import _verdict_for_score

    return _verdict_for_score(float(ws))


def _evaluator_cards(evaluators: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    from skillevaluator.tier3.harbor.metrics import METRIC_DISPLAY

    cards: list[dict[str, Any]] = []
    for metric, scores in evaluators.items():
        ws = _as_float(scores.get("with_skill"))
        status = "pass" if ws >= 0.8 else ("warn" if ws >= 0.6 else "fail")
        cards.append(
            {
                "id": metric,
                "label": METRIC_DISPLAY.get(metric, metric.replace("_", " ").title()),
                "with_skill": ws,
                "baseline": scores.get("baseline"),
                "lift": scores.get("lift"),
                "status": status,
                "evidence": [],
            }
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
    annotation with step/token counters. The canonical shape mirrors Skill Evaluator's
    ``_normalize_harbor_trials`` so the ported Trials tab (per-evaluator
    drill-down, token/steps charts, warnings) renders identically.
    """
    out: list[dict[str, Any]] = []
    for reward in rewards:
        if not isinstance(reward, dict):
            continue
        scores = {
            m: reward.get(m)
            for m in metrics
            if isinstance(reward.get(m), (int, float)) and not isinstance(reward.get(m), bool)
        }
        trial: dict[str, Any] = {
            "trial_id": reward.get("trial_id"),
            "entry_id": reward.get("entry_id"),
            "scores": scores,
            "overall": reward.get("overall"),
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
    deltas (Skill Evaluator parity).
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
            score = scores.get(metric)
            base = trial["baseline_scores"].get(metric)
            if isinstance(score, (int, float)) and isinstance(base, (int, float)):
                lift_scores[metric] = round(float(score) - float(base), 4)
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
    overall = trial.get("overall")
    if isinstance(overall, int | float) and not isinstance(overall, bool):
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
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict) or not _safe_harbor_viewer_url(evidence.get("url")):
            entry["evidence"] = evidence_links[min(index, len(evidence_links) - 1)]
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
        evidence = entry.get("harbor_evidence") or entry.get("evidence")
        if not isinstance(evidence, dict) or not _safe_harbor_viewer_url(evidence.get("url")):
            entry["harbor_evidence"] = evidence_links[min(index, len(evidence_links) - 1)]
        linked.append(entry)
    return linked


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
    """Generate stable Insights conclusions from canonical scores (Skill Evaluator parity)."""
    conclusions: list[dict[str, str]] = []
    if agents:
        best_name = _pick_best_agent(agents)
        if best_name:
            best = agents[best_name]
            best_score = best.get("with_skill", 0.0)
            if not isinstance(best_score, (int, float)):
                best_score = 0.0
            lift = best.get("lift")
            conclusions.append(
                {
                    "severity": "pass" if best_score >= 0.7 else "warn",
                    "title": "Best performing agent",
                    "message": (
                        f"{best_name} leads with overall score {best_score:.2f}"
                        + (f" and lift {lift:+.2f}." if isinstance(lift, (int, float)) else ".")
                    ),
                }
            )

    numeric_dims = [d for d in dimensions if isinstance(d.get("score"), (int, float))]
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
            overall = trial.get("overall")
            if overall is None or (isinstance(overall, (int, float)) and overall < pass_threshold):
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
    """Default suggestions: target the weakest dimensions (Skill Evaluator parity)."""
    pending: list[tuple[float, str]] = []
    for dim in dimensions:
        score = dim.get("with_skill", dim.get("score", 0.0))
        if isinstance(score, (int, float)) and score < 0.7:
            pending.append((float(score), dim.get("id", "")))
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
        return float(value)
    except (TypeError, ValueError):
        return 0.50


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
        value = scores.get(evaluator)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            num += float(value) * float(weight)
            den += float(weight)
    return (num / den) if den > 0 else None


def _precomputed_score(precomputed: dict[str, Any], dim_id: str) -> float | None:
    entry = precomputed.get(dim_id)
    if isinstance(entry, dict):
        score = entry.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
    return None


def _lift_value(metric: str, lift_data: dict[str, Any]) -> float | None:
    entry = lift_data.get(metric)
    if isinstance(entry, dict):
        candidate = entry.get("delta", entry.get("lift"))
        return float(candidate) if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) else None
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        return float(entry)
    return None


def _verdict_from_lift(lift: float | None) -> str:
    if not isinstance(lift, (int, float)):
        return VERDICT_NEUTRAL
    if lift >= _VERDICT_PASS_THRESHOLD:
        return VERDICT_PASS
    if lift <= _VERDICT_FAIL_THRESHOLD:
        return VERDICT_FAIL
    return VERDICT_NEUTRAL


def _pick_best_agent(agents: dict[str, dict[str, Any]]) -> str:
    eligible = {
        name: agent
        for name, agent in agents.items()
        if agent.get("execution_status") == "succeeded"
        and isinstance(agent.get("with_skill"), int | float)
        and not isinstance(agent.get("with_skill"), bool)
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
        value = engine_result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _default_attempt_policy() -> dict[str, Any]:
    return {
        "max_attempts": 1,
        "pass_threshold": 0.50,
        "stop_on_pass": False,
        "score_definition": AGENT_EVAL_SCORE_DEFINITION,
    }


def _mean(values: list[float]) -> float | None:
    numeric = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


def _as_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _as_nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = [
    "advisory_skip_result",
    "agent_eval_result_from_run",
    "build_agent_eval_payload",
]
