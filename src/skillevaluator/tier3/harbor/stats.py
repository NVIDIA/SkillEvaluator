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
    LIFT_BOOTSTRAP_TAIL_RESAMPLES,
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

# The statistics blocks of one agent, in the order build_agent_statistics()
# returns them. Each is persisted in statistics.json and in the agent's results.
STATISTICS_BLOCKS = (
    "lift_uncertainty",
    "reliability",
    "cost",
    "token_efficiency",
    "context_cost_measured",
    "integration_completeness",
)

_SUCCEEDED = "succeeded"


@dataclass(frozen=True)
class TrialObservation:
    """One logical Harbor attempt: its case, reward payload and resource usage.

    ``usage`` holds optional non-negative counters read from the trial's ATIF
    trajectory or Harbor ``result.json``: ``prompt_tokens`` (including cached
    tokens), ``cached_tokens``, ``completion_tokens``, ``cost_usd`` and
    ``first_turn_prompt_tokens``. Missing counters are simply absent.

    ``logical_overall`` is set for a multi-step attempt whose steps use
    different metric contracts (standard and custom-only rows). Its dimension
    scores cover only the standard rows, so the attempt is scored by its
    logical overall instead, the score pass@k and the report headline use.
    """

    case_id: str
    reward: Mapping[str, Any]
    usage: Mapping[str, Any] = field(default_factory=dict)
    logical_overall: float | None = None


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


OVERALL_DIMENSION = "overall"


def trial_dimension_scores(reward: Mapping[str, Any]) -> dict[str, float]:
    """Return one trial's score per dimension, skipping dimensions it cannot score.

    Dimension scores are linear in the metrics. A legacy reward without the
    ``security`` metric uses the same ``behavior_check`` fallback as the
    report. Metrics that are missing, non-finite or N/A are skipped, so a
    dimension whose metrics are all N/A (for example Discoverability in an arm
    that has no skill to discover) is left out. A custom-only reward, or one
    without any dimension, is scored by its ``overall`` value alone.
    """
    payload = dict(reward)
    _, active_metrics = metric_set_for_reward(payload)
    scores: dict[str, float] = {}
    if active_metrics:
        value_of = partial(metric_value, payload)
        for dimension, config in DIMENSION_MAPPING.items():
            value = weighted_dimension_score(value_of, config, active_metrics=active_metrics)
            if value is not None:
                scores[dimension] = value
    if not scores:
        overall = overall_score(payload)
        if overall is not None:
            scores[OVERALL_DIMENSION] = overall
    return scores


def trial_quality_score(reward: Mapping[str, Any], dimensions: Sequence[str] | None = None) -> float | None:
    """Return one trial's score on the same basis as the reported lift.

    The score is the mean of the trial's dimension scores. With *dimensions*,
    only those dimensions count, which is how two arms are compared on the
    dimensions both can score. ``None`` means the trial carries no usable score.
    """
    scores = trial_dimension_scores(reward)
    if dimensions is not None:
        wanted = set(dimensions)
        scores = {dimension: value for dimension, value in scores.items() if dimension in wanted}
    return _mean(list(scores.values())) if scores else None


def case_dimension_means(trials: Sequence[TrialObservation]) -> dict[str, dict[str, float]]:
    """Average each dimension over the scored attempts of each case (attempts nest in cases)."""
    by_case: dict[str, dict[str, list[float]]] = {}
    for trial in trials:
        scores = (
            {OVERALL_DIMENSION: trial.logical_overall}
            if trial.logical_overall is not None
            else trial_dimension_scores(trial.reward)
        )
        for dimension, value in scores.items():
            by_case.setdefault(trial.case_id, {}).setdefault(dimension, []).append(value)
    return {
        case_id: {dimension: math.fsum(values) / len(values) for dimension, values in dimensions.items()}
        for case_id, dimensions in by_case.items()
        if dimensions
    }


def case_mean_scores(trials: Sequence[TrialObservation]) -> dict[str, float]:
    """Average usable trial scores within each case (attempts nest in cases)."""
    return {
        case_id: math.fsum(dimensions.values()) / len(dimensions)
        for case_id, dimensions in case_dimension_means(trials).items()
    }


_DIMENSION_ORDER = (*DIMENSION_MAPPING, OVERALL_DIMENSION)


def _overall_aligned(
    treatment: dict[str, float], control: dict[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    """A case scored only by its overall on one side compares with the other side's overall (dimension mean)."""
    if set(treatment) == {OVERALL_DIMENSION} and OVERALL_DIMENSION not in control and control:
        return treatment, {OVERALL_DIMENSION: math.fsum(control.values()) / len(control)}
    if set(control) == {OVERALL_DIMENSION} and OVERALL_DIMENSION not in treatment and treatment:
        return {OVERALL_DIMENSION: math.fsum(treatment.values()) / len(treatment)}, control
    return treatment, control


def shared_case_scores(
    treatment: Sequence[TrialObservation],
    control: Sequence[TrialObservation],
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Score two arms per case on the dimensions both arms scored for that case.

    An arm without the skill under test has no Discoverability or Efficiency
    (the verifier records them as N/A), so a lift over every dimension would
    credit skill activation alone. Each paired case is scored on the
    dimensions both arms have for it, the same set on both sides. Returns the
    treatment and control case scores and the dimensions that were compared.
    """
    treatment_cases = case_dimension_means(treatment)
    control_cases = case_dimension_means(control)
    treatment_scores: dict[str, float] = {}
    control_scores: dict[str, float] = {}
    compared: set[str] = set()
    for case_id in sorted(set(treatment_cases) & set(control_cases)):
        treatment_case, control_case = _overall_aligned(treatment_cases[case_id], control_cases[case_id])
        shared = sorted(set(treatment_case) & set(control_case))
        if not shared:
            continue
        treatment_scores[case_id] = math.fsum(treatment_case[dim] for dim in shared) / len(shared)
        control_scores[case_id] = math.fsum(control_case[dim] for dim in shared) / len(shared)
        compared.update(shared)
    return treatment_scores, control_scores, [dim for dim in _DIMENSION_ORDER if dim in compared]


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


# A percentile bootstrap over a handful of cases is too narrow (about 82%
# coverage at 5 cases for a nominal 95%), and at small n the narrow intervals
# are the overconfident ones. "adequate" therefore also needs this many cases.
LIFT_CI_ADEQUATE_MIN_CASES = 10
LIFT_INTERVAL_METHOD = "expanded_percentile"


def lift_precision(n_cases: int, ci_low: float | None, ci_high: float | None) -> str:
    """Classify how informative a lift interval is, by case count and width."""
    if n_cases < LIFT_CI_MIN_PAIRED_CASES or ci_low is None or ci_high is None:
        return PRECISION_INSUFFICIENT
    if n_cases < LIFT_CI_ADEQUATE_MIN_CASES or ci_high - ci_low > LIFT_CI_LOW_PRECISION_WIDTH:
        return PRECISION_LOW
    return PRECISION_ADEQUATE


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _incomplete_beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    fraction = d
    for m in range(1, 300):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            fraction *= c * d
        if abs(c * d - 1.0) < 1e-15:
            break
    return fraction


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    """Student's t cumulative distribution, via the regularized incomplete beta function."""
    df = float(degrees_of_freedom)
    x = df / (df + value * value)
    a, b = df / 2.0, 0.5
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        incomplete = math.exp(log_front) * _incomplete_beta_fraction(a, b, x) / a
    else:
        incomplete = 1.0 - math.exp(log_front) * _incomplete_beta_fraction(b, a, 1.0 - x) / b
    tail = 0.5 * incomplete
    return 1.0 - tail if value > 0 else tail


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Upper quantile (probability above 0.5) of Student's t, by bisection."""
    high = 1.0
    while _student_t_cdf(high, degrees_of_freedom) < probability and high < 1e6:
        high *= 2.0
    low = 0.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if _student_t_cdf(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def expanded_tail(n_cases: int, confidence: float) -> float:
    """Lower-tail probability of an expanded percentile bootstrap interval.

    A plain percentile bootstrap undercovers at small n. Hesterberg's expanded
    percentile interval uses the percentiles at ``Phi(-sqrt(n / (n - 1)) *
    t_{alpha/2, n-1})`` instead of ``alpha / 2``, which widens the interval at
    small n and converges to the plain interval as n grows.
    """
    alpha = 1.0 - confidence
    if n_cases < 2:
        return alpha / 2.0
    t_value = _student_t_quantile(1.0 - alpha / 2.0, n_cases - 1)
    return _normal_cdf(-math.sqrt(n_cases / (n_cases - 1)) * t_value)


def paired_case_bootstrap(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
    *,
    resamples: int = LIFT_BOOTSTRAP_RESAMPLES,
    seed: int = LIFT_BOOTSTRAP_SEED,
    confidence: float = LIFT_BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Expanded percentile bootstrap CI for the mean paired per-case lift.

    ``treatment`` and ``control`` map case ids to case-level mean scores. Only
    cases present in both arms are paired. Whole cases are resampled with
    replacement from a ``random.Random(seed)`` stream, so the interval is fully
    deterministic for a given input. The percentiles are widened for small n
    (:func:`expanded_tail`), because a plain percentile bootstrap over 5-10
    cases is overconfident. ``resamples`` is a minimum: with at least
    ``LIFT_CI_MIN_PAIRED_CASES`` cases, enough are drawn that each bound rests on
    ``LIFT_BOOTSTRAP_TAIL_RESAMPLES`` resampled means, so the bounds do not move
    with the seed. With fewer cases the bounds are the extreme resampled means,
    which the minimum already reaches.
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
        "interval": LIFT_INTERVAL_METHOD,
        "resamples": resamples,
        "seed": seed,
        "n_cases": n_cases,
        "precision": PRECISION_INSUFFICIENT,
        "ci_includes_zero": None,
    }
    if not n_cases:
        return result
    if n_cases < 2:
        # One case cannot be resampled: a zero-width interval would look final.
        result["estimate"] = round(deltas[0], 4)
        return result

    tail = expanded_tail(n_cases, confidence)
    if n_cases >= LIFT_CI_MIN_PAIRED_CASES:
        resamples = max(resamples, math.ceil(LIFT_BOOTSTRAP_TAIL_RESAMPLES / tail))
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(deltas, k=n_cases)) / n_cases for _ in range(resamples))
    ci_low = _percentile(means, tail)
    ci_high = _percentile(means, 1.0 - tail)
    rounded_low = round(ci_low, 4)
    rounded_high = round(ci_high, 4)
    result.update(
        {
            "estimate": round(math.fsum(deltas) / n_cases, 4),
            "ci_low": rounded_low,
            "ci_high": rounded_high,
            "resamples": resamples,
            "precision": lift_precision(n_cases, ci_low, ci_high),
            "ci_includes_zero": rounded_low <= 0.0 <= rounded_high,
        }
    )
    return result


LIFT_BASIS_SHARED_DIMENSIONS = "shared_dimensions_case_weighted"


def paired_lift(
    treatment: ArmObservations | None,
    control: ArmObservations | None,
    *,
    treatment_arm: str,
    control_arm: str,
    expected_case_ids: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Paired lift of *treatment* over *control*, with its interval, on one basis.

    Both arms are scored per case on the dimensions both scored for that case
    (see :func:`shared_case_scores`), so the headline lift, the arm scores
    shown next to it and the interval are all the same case-weighted paired
    mean. ``treatment_score - control_score`` equals ``estimate`` up to rounding.
    Returns ``None`` when no case was scored in both arms.
    """
    if treatment is None or control is None:
        return None
    treatment_scores, control_scores, dimensions = shared_case_scores(treatment.trials, control.trials)
    if not treatment_scores:
        return None
    result = paired_case_bootstrap(treatment_scores, control_scores)
    expected_ids = [str(case_id) for case_id in (expected_case_ids or []) if str(case_id)]
    observed_ids = {trial.case_id for trial in (*treatment.trials, *control.trials)}
    expected_cases = len(set(expected_ids)) if expected_ids else len(observed_ids)
    failed_arms = [
        arm
        for arm, observed in ((treatment_arm, treatment), (control_arm, control))
        if not observed.succeeded or observed.job_failure
    ]
    result.update(
        {
            "basis": LIFT_BASIS_SHARED_DIMENSIONS,
            "dimensions": dimensions,
            "treatment_arm": treatment_arm,
            "control_arm": control_arm,
            "treatment_score": _round(_mean(list(treatment_scores.values()))),
            "control_score": _round(_mean(list(control_scores.values()))),
            "expected_cases": expected_cases,
            "partial": bool(failed_arms) or int(result["n_cases"]) < expected_cases,
            "failed_arms": failed_arms,
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
    Unscored attempts never count as passes. The published ``cases`` block is
    capped for size, so the collector's ``_pairing_cases`` (every case and
    attempt) is read when present.
    """
    if not pass_summary:
        return None
    k = int(pass_summary.get("k") or 0)
    n_cases = int(pass_summary.get("total_cases") or 0)
    cases = pass_summary.get("_pairing_cases", pass_summary.get("cases"))
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


def arm_not_applicable_metrics(trials: Sequence[TrialObservation]) -> list[str]:
    """Metrics that no trial of an arm scored, in metric-set order.

    In an arm that succeeded these are not applicable, for example the skill
    metrics of an arm without the skill. Each arm's pass@k uses only the
    metrics it scored, so arms with different lists are not a like-for-like
    pass@k comparison.
    """
    active: list[str] = []
    scored: set[str] = set()
    for trial in trials:
        reward = dict(trial.reward)
        _, metrics = metric_set_for_reward(reward)
        for metric in metrics:
            if metric not in active:
                active.append(metric)
            if metric_value(reward, metric) is not None:
                scored.add(metric)
    return [metric for metric in active if metric not in scored]


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


#: Fewer paired cases than this give a delta with status ``insufficient``: shown, but not a measurement.
CONTEXT_COST_MIN_PAIRED_CASES = 3


@dataclass
class _FirstTurns:
    """One arm's per-case mean first-turn prompt tokens, and the trials left out."""

    by_case: dict[str, float]
    used: int = 0
    hosted_search: int = 0
    missing: int = 0  # usage was read, but it has no first-turn count
    no_usage: int = 0  # no usage was collected for the trial at all


def _case_first_turn_tokens(trials: Sequence[TrialObservation]) -> _FirstTurns:
    by_case: dict[str, list[float]] = {}
    turns = _FirstTurns({})
    for trial in trials:
        tokens = finite_number(trial.usage.get("first_turn_prompt_tokens"), non_negative=True)
        if tokens is not None:
            turns.used += 1
            by_case.setdefault(trial.case_id, []).append(tokens)
        elif finite_number(trial.usage.get("first_turn_hosted_search"), non_negative=True):
            turns.hosted_search += 1
        elif trial.usage:
            turns.missing += 1
        else:
            turns.no_usage += 1
    turns.by_case = {case_id: math.fsum(values) / len(values) for case_id, values in by_case.items()}
    return turns


def _left_out(arm_name: str, turns: _FirstTurns) -> list[str]:
    parts = []
    if turns.no_usage:
        parts.append(f"no usage was collected for {turns.no_usage} trial(s) of the {arm_name}")
    if turns.missing:
        parts.append(f"{turns.missing} trial(s) of the {arm_name} have no first-turn prompt token count")
    if turns.hosted_search:
        parts.append(
            f"{turns.hosted_search} trial(s) of the {arm_name} ran a hosted web search inside the first model call, "
            "whose results count as that call's input"
        )
    return parts


def context_cost_measured(
    with_arm: ArmObservations | None,
    without_arm: ArmObservations | None,
) -> dict[str, Any]:
    """Paired mean first-turn prompt-token delta (with minus without), by case.

    The first LLM turn carries the always-on context (system prompt, skill and
    rule preambles, tool schemas) before task work starts, so the paired delta
    measures what the plugin adds to every request. Every trial with a count
    is used, also in an arm that failed elsewhere (a judge error does not
    change a token count). Trials without a count, and trials whose first call
    ran a hosted web search, are left out and the result says ``partial``.
    Fewer than :data:`CONTEXT_COST_MIN_PAIRED_CASES` pairs give ``insufficient``.
    """
    result: dict[str, Any] = {
        "method": CONTEXT_COST_METHOD,
        "delta_tokens_mean": None,
        "n_pairs": 0,
        "min_pairs": CONTEXT_COST_MIN_PAIRED_CASES,
        "status": "unavailable",
        "partial": False,
        "excluded": {"hosted_search": 0, "missing_tokens": 0},
        "reason": None,
    }
    if with_arm is None or without_arm is None or without_arm.execution_status == "skipped":
        result["reason"] = "No baseline (without) arm was run."
        return result
    with_turns = _case_first_turn_tokens(with_arm.trials)
    without_turns = _case_first_turn_tokens(without_arm.trials)
    result["excluded"] = {
        "hosted_search": with_turns.hosted_search + without_turns.hosted_search,
        "missing_tokens": with_turns.missing + with_turns.no_usage + without_turns.missing + without_turns.no_usage,
    }
    left_out = [*_left_out("with-plugin arm", with_turns), *_left_out("baseline arm", without_turns)]
    for arm_name, turns in (("with-plugin arm", with_turns), ("baseline arm", without_turns)):
        if not turns.by_case:
            result["reason"] = f"The {arm_name} has no trial with a usable first-turn prompt token count" + (
                f" ({'; '.join(left_out)})." if left_out else "."
            )
            return result
    paired = sorted(set(with_turns.by_case) & set(without_turns.by_case))
    if not paired:
        result["reason"] = "No case has first-turn prompt tokens in both arms."
        return result
    deltas = [with_turns.by_case[case_id] - without_turns.by_case[case_id] for case_id in paired]
    unpaired = len(set(with_turns.by_case) ^ set(without_turns.by_case))
    partial = bool(left_out or unpaired or not with_arm.succeeded or not without_arm.succeeded)
    status = "measured" if len(paired) >= CONTEXT_COST_MIN_PAIRED_CASES else "insufficient"
    reasons: list[str] = []
    if status == "insufficient":
        reasons.append(
            f"Only {len(paired)} case(s) have first-turn prompt tokens in both arms; at least "
            f"{CONTEXT_COST_MIN_PAIRED_CASES} are needed."
        )
    if partial:
        detail = [*left_out]
        if unpaired:
            detail.append(f"{unpaired} case(s) have a count in one arm only")
        if not detail:
            failed = [name for name, arm in (("with-plugin", with_arm), ("baseline", without_arm)) if not arm.succeeded]
            detail.append(f"the {' and '.join(failed)} arm did not complete, but every collected trial has its count")
        reasons.append("Partial: " + "; ".join(detail) + ".")
    result.update(
        {
            "delta_tokens_mean": round(math.fsum(deltas) / len(deltas), 2),
            "n_pairs": len(paired),
            "status": status,
            "partial": partial,
            "reason": " ".join(reasons) or None,
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
    baseline_is_sum_of_parts: bool = False,
) -> dict[str, Any]:
    """Assemble every C4 statistics block for one agent.

    ``arms`` is keyed by ``with_skill``, ``without_skill`` and ``sum_of_parts``;
    absent keys mean the arm was not run. A lift interval uses every case
    scored in both of its arms: one failed trial (a judge error or a timeout)
    marks it ``partial`` with ``n_cases`` of ``expected_cases`` pairs instead of
    removing it. Per-arm reliability, cost and token blocks cover only arms
    whose execution succeeded, mirroring how pass@k is gated.

    ``baseline_is_sum_of_parts`` marks the legacy two-arm ``--lift-mode
    integration`` run, whose only baseline stages the member skills: its
    interval compares the plugin with its parts, so it is filed under
    ``integration`` and no effectiveness interval exists.
    """
    with_arm = arms.get(ARM_WITH)
    without_arm = arms.get(ARM_WITHOUT)
    sop_arm = arms.get(ARM_SUM_OF_PARTS) if sum_of_parts_requested else None

    effectiveness = None
    if baseline_is_sum_of_parts:
        integration = paired_lift(
            with_arm,
            without_arm,
            treatment_arm=ARM_WITH,
            control_arm=ARM_WITHOUT,
            expected_case_ids=expected_case_ids,
        )
    else:
        effectiveness = paired_lift(
            with_arm,
            without_arm,
            treatment_arm=ARM_WITH,
            control_arm=ARM_WITHOUT,
            expected_case_ids=expected_case_ids,
        )
        integration = paired_lift(
            with_arm,
            sop_arm,
            treatment_arm=ARM_WITH,
            control_arm=ARM_SUM_OF_PARTS,
            expected_case_ids=expected_case_ids,
        )

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
            summary["not_applicable_metrics"] = arm_not_applicable_metrics(arm.trials)
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
