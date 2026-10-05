# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only Tier 3 statistics computed from collected Harbor trials.

Everything in this module is advisory. None of it changes a metric, a
dimension, ``overall_score``, pass@k, or a pass/fail gate. The collector does
all file I/O and hands this module plain observations, so the functions here
are pure and deterministic.

The blocks produced here (``lift_uncertainty``, ``reliability``, ``cost``,
``token_efficiency``, ``context_cost_measured`` and
``integration_completeness``) follow the shared C4 statistics contract.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from skillevaluator.constants import (
    DIMENSION_MAPPING,
    LIFT_BOOTSTRAP_CONFIDENCE,
    LIFT_BOOTSTRAP_RESAMPLES,
    LIFT_BOOTSTRAP_SEED,
    LIFT_CI_LOW_PRECISION_WIDTH,
    LIFT_CI_MIN_PAIRED_CASES,
    TOKEN_EFFICIENCY_HALF_LIFE,
)
from skillevaluator.tier3.harbor.metrics import (
    finite_number,
    metric_set_for_reward,
    metric_value,
    overall_score,
    weighted_dimension_score,
)

LIFT_UNCERTAINTY_METHOD = "paired_case_bootstrap"
CONTEXT_COST_METHOD = "paired_first_turn_prompt_tokens"

PRECISION_ADEQUATE = "adequate"
PRECISION_LOW = "low"
PRECISION_INSUFFICIENT = "insufficient"

ARM_WITH = "with_skill"
ARM_WITHOUT = "without_skill"
ARM_SUM_OF_PARTS = "sum_of_parts"

_SUCCEEDED = "succeeded"


@dataclass(frozen=True)
class TrialObservation:
    """One logical Harbor attempt: its case, reward payload and resource usage.

    ``usage`` holds optional non-negative counters read from the trial's ATIF
    trajectory or Harbor ``result.json``: ``prompt_tokens`` (including cached
    tokens), ``cached_tokens``, ``completion_tokens``, ``cost_usd`` and
    ``first_turn_prompt_tokens``. Missing counters are simply absent.
    """

    case_id: str
    reward: Mapping[str, Any]
    usage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmObservations:
    """All logical attempts of one evaluation arm, in attempt order per case."""

    execution_status: str
    trials: Sequence[TrialObservation] = ()
    pass_summary: Mapping[str, Any] = field(default_factory=dict)
    job_failure: str = ""

    @property
    def succeeded(self) -> bool:
        return self.execution_status == _SUCCEEDED


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


# ---------------------------------------------------------------------------
# Per-trial quality score (the basis of the reported lift)
# ---------------------------------------------------------------------------


def trial_quality_score(reward: Mapping[str, Any]) -> float | None:
    """Return one trial's score on the same basis as the reported lift.

    The Tier 3 report's lift is the mean of the dimension scores. Dimension
    scores are linear in the metrics, so each trial is scored as the mean of its
    own available dimensions. A legacy reward without the ``security`` metric
    uses the same ``behavior_check`` fallback as the report. Custom-only rewards
    use their ``overall`` value. Metrics that are missing, non-finite or N/A are
    skipped; ``None`` means the trial carries no usable score.
    """
    payload = dict(reward)
    _, active_metrics = metric_set_for_reward(payload)
    if not active_metrics:
        return overall_score(payload)
    value_of = partial(metric_value, payload)
    dimension_values: list[float] = []
    for config in DIMENSION_MAPPING.values():
        value = weighted_dimension_score(value_of, config, active_metrics=active_metrics)
        if value is not None:
            dimension_values.append(value)
    if dimension_values:
        return _mean(dimension_values)
    return overall_score(payload)


def case_mean_scores(trials: Sequence[TrialObservation]) -> dict[str, float]:
    """Average usable trial scores within each case (attempts nest in cases)."""
    by_case: dict[str, list[float]] = {}
    for trial in trials:
        score = trial_quality_score(trial.reward)
        if score is not None:
            by_case.setdefault(trial.case_id, []).append(score)
    return {case_id: math.fsum(scores) / len(scores) for case_id, scores in by_case.items() if scores}


# ---------------------------------------------------------------------------
# Lift uncertainty
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Linearly interpolated percentile of an already sorted sequence."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def lift_precision(n_cases: int, ci_low: float | None, ci_high: float | None) -> str:
    """Classify how informative a lift interval is."""
    if n_cases < LIFT_CI_MIN_PAIRED_CASES or ci_low is None or ci_high is None:
        return PRECISION_INSUFFICIENT
    if ci_high - ci_low > LIFT_CI_LOW_PRECISION_WIDTH:
        return PRECISION_LOW
    return PRECISION_ADEQUATE


def paired_case_bootstrap(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
    *,
    resamples: int = LIFT_BOOTSTRAP_RESAMPLES,
    seed: int = LIFT_BOOTSTRAP_SEED,
    confidence: float = LIFT_BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the mean paired per-case lift.

    ``treatment`` and ``control`` map case ids to case-level mean scores. Only
    cases present in both arms are paired. Whole cases are resampled with
    replacement from a ``random.Random(seed)`` stream, so the interval is fully
    deterministic for a given input.
    """
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    paired = sorted(set(treatment) & set(control))
    deltas = [float(treatment[case_id]) - float(control[case_id]) for case_id in paired]
    n_cases = len(deltas)
    result: dict[str, Any] = {
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "confidence": confidence,
        "method": LIFT_UNCERTAINTY_METHOD,
        "resamples": resamples,
        "seed": seed,
        "n_cases": n_cases,
        "precision": PRECISION_INSUFFICIENT,
        "ci_includes_zero": None,
    }
    if not n_cases:
        return result

    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(deltas, k=n_cases)) / n_cases for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    ci_low = _percentile(means, tail)
    ci_high = _percentile(means, 1.0 - tail)
    rounded_low = round(ci_low, 4)
    rounded_high = round(ci_high, 4)
    result.update(
        {
            "estimate": round(math.fsum(deltas) / n_cases, 4),
            "ci_low": rounded_low,
            "ci_high": rounded_high,
            "precision": lift_precision(n_cases, ci_low, ci_high),
            "ci_includes_zero": rounded_low <= 0.0 <= rounded_high,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Reliability (pass^k)
# ---------------------------------------------------------------------------


def reliability_summary(pass_summary: Mapping[str, Any], *, stop_on_pass: bool) -> dict[str, Any] | None:
    """Report pass@k next to pass^k (every one of the k attempts passed).

    pass@k keeps its existing meaning. pass^k needs all k attempts to be
    observed, so it is ``None`` when ``stop_on_pass`` truncated the attempts.
    Unscored attempts never count as passes.
    """
    if not pass_summary:
        return None
    k = int(pass_summary.get("k") or 0)
    n_cases = int(pass_summary.get("total_cases") or 0)
    cases = pass_summary.get("cases")
    pass_hat_k: float | None = None
    if k > 0 and n_cases > 0 and isinstance(cases, Mapping) and not (stop_on_pass and k > 1):
        all_passed = 0
        for case in cases.values():
            if not isinstance(case, Mapping) or case.get("extra_case"):
                continue
            attempts = [row for row in case.get("attempts") or [] if isinstance(row, Mapping)]
            if len(attempts) >= k and all(bool(row.get("passed")) for row in attempts[:k]):
                all_passed += 1
        pass_hat_k = round(all_passed / n_cases, 4)
    return {
        "pass_at_k": finite_number(pass_summary.get("rate")),
        "pass_hat_k": pass_hat_k,
        "k": k,
        "n_cases": n_cases,
    }


# ---------------------------------------------------------------------------
# Tokens, cost and token efficiency
# ---------------------------------------------------------------------------


def uncached_tokens(usage: Mapping[str, Any]) -> float | None:
    """Return uncached prompt plus completion tokens, or ``None`` if unknown.

    Prompt counters include cached tokens (ATIF ``total_prompt_tokens`` and
    Harbor ``n_input_tokens`` both do), so cached tokens are subtracted.
    """
    prompt = finite_number(usage.get("prompt_tokens"), non_negative=True)
    completion = finite_number(usage.get("completion_tokens"), non_negative=True)
    if prompt is None or completion is None:
        return None
    cached = finite_number(usage.get("cached_tokens"), non_negative=True) or 0.0
    return max(prompt - cached, 0.0) + completion


def token_efficiency_score(tokens: float | None) -> float | None:
    """Soft-asymptote token score: ``1 / (1 + tokens / TOKEN_EFFICIENCY_HALF_LIFE)``."""
    if tokens is None or tokens <= 0:
        return None
    return 1.0 / (1.0 + tokens / TOKEN_EFFICIENCY_HALF_LIFE)


def arm_token_efficiency(trials: Sequence[TrialObservation]) -> float | None:
    """Mean per-trial token efficiency; ``None`` unless every trial has token counts."""
    scores = [token_efficiency_score(uncached_tokens(trial.usage)) for trial in trials]
    if not scores or any(score is None for score in scores):
        return None
    return _round(_mean([score for score in scores if score is not None]))


def arm_cost(trials: Sequence[TrialObservation], *, successes: int) -> dict[str, Any]:
    """Token and USD cost per passed case, counting every attempt.

    Totals are only reported when every attempt carries the counter, so a
    partial sum is never presented as the arm's cost. USD comes only from
    costs reported by Harbor or the agent trajectory; nothing is priced here.
    """
    token_counts = [uncached_tokens(trial.usage) for trial in trials]
    total_tokens = (
        round(math.fsum(value for value in token_counts if value is not None))
        if token_counts and all(value is not None for value in token_counts)
        else None
    )
    costs = [finite_number(trial.usage.get("cost_usd"), non_negative=True) for trial in trials]
    total_usd = (
        round(math.fsum(value for value in costs if value is not None), 6)
        if costs and all(value is not None for value in costs)
        else None
    )
    return {
        "tokens_per_success": round(total_tokens / successes, 2) if total_tokens is not None and successes else None,
        "usd_per_success": round(total_usd / successes, 6) if total_usd is not None and successes else None,
        "total_tokens": total_tokens,
        "total_usd": total_usd,
        "successes": successes,
    }


# ---------------------------------------------------------------------------
# Measured always-on context cost
# ---------------------------------------------------------------------------


def _case_first_turn_tokens(trials: Sequence[TrialObservation]) -> tuple[dict[str, float], bool]:
    by_case: dict[str, list[float]] = {}
    complete = True
    for trial in trials:
        tokens = finite_number(trial.usage.get("first_turn_prompt_tokens"), non_negative=True)
        if tokens is None:
            complete = False
            continue
        by_case.setdefault(trial.case_id, []).append(tokens)
    return {case_id: math.fsum(values) / len(values) for case_id, values in by_case.items()}, complete


def context_cost_measured(
    with_arm: ArmObservations | None,
    without_arm: ArmObservations | None,
) -> dict[str, Any]:
    """Paired mean first-turn prompt-token delta (with minus without), by case.

    The first LLM turn carries the always-on context (system prompt, skill and
    rule preambles, tool schemas) before task work starts, so the paired delta
    measures what the plugin adds to every request.
    """
    result: dict[str, Any] = {
        "method": CONTEXT_COST_METHOD,
        "delta_tokens_mean": None,
        "n_pairs": 0,
        "status": "unavailable",
        "reason": None,
    }
    if with_arm is None or without_arm is None or without_arm.execution_status == "skipped":
        result["reason"] = "No baseline (without) arm was run."
        return result
    if not with_arm.succeeded or not without_arm.succeeded:
        result["reason"] = "The with and without arms did not both complete."
        return result
    with_tokens, with_complete = _case_first_turn_tokens(with_arm.trials)
    without_tokens, without_complete = _case_first_turn_tokens(without_arm.trials)
    if not with_complete or not without_complete:
        result["reason"] = "Trajectories do not expose per-step prompt token counts for every attempt."
        return result
    paired = sorted(set(with_tokens) & set(without_tokens))
    if not paired:
        result["reason"] = "No case has first-turn prompt tokens in both arms."
        return result
    deltas = [with_tokens[case_id] - without_tokens[case_id] for case_id in paired]
    result.update(
        {
            "delta_tokens_mean": round(math.fsum(deltas) / len(deltas), 2),
            "n_pairs": len(paired),
            "status": "measured",
        }
    )
    return result


# ---------------------------------------------------------------------------
# Per-case Integration completeness
# ---------------------------------------------------------------------------


def _scored_pass_flags(trials: Sequence[TrialObservation], pass_threshold: float) -> dict[str, list[bool]]:
    flags: dict[str, list[bool]] = {}
    for trial in trials:
        score = overall_score(dict(trial.reward))
        if score is None:
            continue
        flags.setdefault(trial.case_id, []).append(round(score, 4) >= pass_threshold)
    return flags


def _required_attempts(flags: Sequence[bool], *, n_attempts: int, stop_on_pass: bool) -> int:
    if stop_on_pass:
        first_pass = next((index for index, passed in enumerate(flags, start=1) if passed), None)
        if first_pass is not None:
            return first_pass
    return n_attempts


def integration_completeness(
    with_arm: ArmObservations | None,
    sum_of_parts_arm: ArmObservations | None,
    *,
    expected_case_ids: Sequence[str] | None,
    n_attempts: int,
    stop_on_pass: bool,
    pass_threshold: float,
    requested: bool = True,
) -> dict[str, Any]:
    """Check that the Integration arms cover the same cases with enough attempts.

    Coverage counts only scoreable attempts. ``complete`` requires every
    compared case to be present in both arms, no failed arm (including a failed
    sum-of-parts Harbor job), and no case below its required attempt count:
    ``n_attempts``, or the first passing attempt when ``stop_on_pass`` legitimately
    ended a case early. ``complete`` is ``None`` when no sum-of-parts arm was
    ``requested`` (effectiveness-only lift, ``--skip-baseline``): there is
    nothing to compare, so nothing is incomplete. A requested arm without
    observations is listed in ``failed_arms``.
    """
    if not requested:
        return {"complete": None, "missing_cases": [], "failed_arms": [], "attempt_shortfall": []}
    if with_arm is None or sum_of_parts_arm is None:
        missing_arms = [
            arm for arm, observed in ((ARM_WITH, with_arm), (ARM_SUM_OF_PARTS, sum_of_parts_arm)) if observed is None
        ]
        return {"complete": False, "missing_cases": [], "failed_arms": missing_arms, "attempt_shortfall": []}

    failed_arms: list[str] = []
    if not with_arm.succeeded:
        failed_arms.append(ARM_WITH)
    if not sum_of_parts_arm.succeeded or sum_of_parts_arm.job_failure:
        failed_arms.append(ARM_SUM_OF_PARTS)

    with_flags = _scored_pass_flags(with_arm.trials, pass_threshold)
    sop_flags = _scored_pass_flags(sum_of_parts_arm.trials, pass_threshold)
    expected = list(dict.fromkeys(str(case_id) for case_id in (expected_case_ids or []) if str(case_id)))
    compared = expected or sorted(set(with_flags) | set(sop_flags))

    missing_cases = [case_id for case_id in compared if not with_flags.get(case_id) or not sop_flags.get(case_id)]
    attempt_shortfall: list[dict[str, Any]] = []
    for case_id in compared:
        for arm, flags_by_case in ((ARM_WITH, with_flags), (ARM_SUM_OF_PARTS, sop_flags)):
            flags = flags_by_case.get(case_id, [])
            required = _required_attempts(flags, n_attempts=n_attempts, stop_on_pass=stop_on_pass)
            if len(flags) < required:
                attempt_shortfall.append({"case": case_id, "arm": arm, "expected": required, "observed": len(flags)})

    return {
        "complete": bool(compared) and not missing_cases and not failed_arms and not attempt_shortfall,
        "missing_cases": missing_cases,
        "failed_arms": failed_arms,
        "attempt_shortfall": attempt_shortfall,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_agent_statistics(
    arms: Mapping[str, ArmObservations],
    *,
    expected_case_ids: Sequence[str] | None,
    n_attempts: int,
    stop_on_pass: bool,
    pass_threshold: float,
    sum_of_parts_requested: bool,
) -> dict[str, Any]:
    """Assemble every C4 statistics block for one agent.

    ``arms`` is keyed by ``with_skill``, ``without_skill`` and ``sum_of_parts``;
    absent keys mean the arm was not run. Per-arm blocks cover only arms whose
    execution succeeded, mirroring how lift and pass@k are gated.
    """
    with_arm = arms.get(ARM_WITH)
    without_arm = arms.get(ARM_WITHOUT)
    sop_arm = arms.get(ARM_SUM_OF_PARTS) if sum_of_parts_requested else None

    effectiveness = None
    if with_arm is not None and without_arm is not None and with_arm.succeeded and without_arm.succeeded:
        effectiveness = paired_case_bootstrap(case_mean_scores(with_arm.trials), case_mean_scores(without_arm.trials))
    integration = None
    if with_arm is not None and sop_arm is not None and with_arm.succeeded and sop_arm.succeeded:
        integration = paired_case_bootstrap(case_mean_scores(with_arm.trials), case_mean_scores(sop_arm.trials))

    reliability: dict[str, Any] = {}
    cost: dict[str, Any] = {}
    token_efficiency: dict[str, Any] = {}
    for name, arm in arms.items():
        if name == ARM_SUM_OF_PARTS and not sum_of_parts_requested:
            continue
        if not arm.succeeded:
            continue
        summary = reliability_summary(arm.pass_summary, stop_on_pass=stop_on_pass)
        if summary is not None:
            reliability[name] = summary
        successes = int(arm.pass_summary.get("passed_cases") or 0) if arm.pass_summary else 0
        cost[name] = arm_cost(arm.trials, successes=successes)
        token_efficiency[name] = arm_token_efficiency(arm.trials)

    return {
        "lift_uncertainty": {"effectiveness": effectiveness, "integration": integration},
        "reliability": reliability,
        "cost": cost,
        "token_efficiency": token_efficiency,
        "context_cost_measured": context_cost_measured(with_arm, without_arm),
        "integration_completeness": integration_completeness(
            with_arm,
            sop_arm,
            expected_case_ids=expected_case_ids,
            n_attempts=n_attempts,
            stop_on_pass=stop_on_pass,
            pass_threshold=pass_threshold,
            requested=sum_of_parts_requested,
        ),
    }
