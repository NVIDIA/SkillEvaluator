# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pair two arms per case before a per-metric lift file compares them.

``lift.json``, ``integration_lift.json`` and ``loose_parts_lift.json`` compare
two arms metric by metric. An arm's own average covers only the trials that arm
scored, so the two averages covered different cases whenever one arm recorded a
metric as not applicable on some cases (an arm without a case's skill), missed
a case, or a case was dropped for load evidence. Here each metric keeps only the
cases both arms scored it on; each arm's value is the mean of its scored trials
on those cases (the arm average itself when nothing is left out). A case whose
two arms were judged by different models (a judge fallback in one arm) is left
out of that judge metric. The ``basis`` block says what each metric covers and
what was left out, and why.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from skillevaluator.tier3.harbor.metrics import metric_is_not_applicable, metric_set_for_reward, metric_value
from skillevaluator.tier3.harbor.stats import EXCLUDED_LOAD_FAILED, EXCLUDED_NOT_SCORED

PAIRING_SAME_CASES = "same_cases_per_metric"
# Why a case (or a whole metric) is left out of a comparison.
EXCLUDED_NOT_APPLICABLE = "not_applicable"  # the arm's verifier recorded the metric as N/A
EXCLUDED_NO_VALUE = "no_value"  # the arm scored the case but has no number for the metric
EXCLUDED_NO_SHARED_CASE = "no_shared_case"  # both arms scored the metric, never on the same case
EXCLUDED_UNIDENTIFIED = "unidentified_case"  # trials whose case could not be resolved
EXCLUDED_JUDGE_MIXED = "judge_mixed"  # the two arms were judged by different models on this case
_UNIDENTIFIED_CASES = frozenset({"", "unknown"})
# Reasons that leave a whole case out, whatever the metric.
_CASE_LEVEL_REASONS = frozenset({EXCLUDED_LOAD_FAILED, EXCLUDED_NOT_SCORED, EXCLUDED_UNIDENTIFIED})
# The report loader caps a JSON file's size, so a long dataset lists only this many excluded-case entries (the
# per-metric counts stay exact; ``excluded_case_entries`` and ``excluded_cases_truncated`` say how many there were).
MAX_EXCLUDED_CASE_ENTRIES = 64

TrialRows = Sequence[tuple[str, dict[str, Any]]]


@dataclass(frozen=True)
class PairedScores:
    """Both arms' per-metric means on the shared cases, plus the ``basis`` record for the lift file.

    ``not_comparable`` names the metrics left out only because of N/A (an arm
    never scores them, or both arms do but never on the same case): the overall
    lift treats them as N/A, as it does an arm's all-N/A metric.
    """

    treatment: dict[str, float]
    control: dict[str, float]
    not_comparable: tuple[str, ...]
    basis: dict[str, Any]
    # Judge metric -> number of cases left out because the arms' judge models differed.
    judge_mixed: dict[str, int] = field(default_factory=dict)


def _scored(reward: dict[str, Any], metric: str) -> float | None:
    _, metrics = metric_set_for_reward(reward)
    return metric_value(reward, metric) if metric in metrics else None


def _status(rewards: Sequence[dict[str, Any]], metric: str) -> str | None:
    """``None`` when some trial scored *metric*, else why none did."""
    if any(_scored(reward, metric) is not None for reward in rewards):
        return None
    if all(metric_is_not_applicable(reward, metric) for reward in rewards):
        return EXCLUDED_NOT_APPLICABLE
    return EXCLUDED_NO_VALUE


def pair_metric_scores(
    treatment: TrialRows,
    control: TrialRows,
    *,
    arms: tuple[str, str],
    metrics: Sequence[str],
    expected_case_ids: Sequence[str] = (),
    dropped_cases: Mapping[str, Sequence[str]] | None = None,
    judge_mixed_cases: Mapping[str, Collection[str]] | None = None,
) -> PairedScores:
    """Score both arms per metric on the cases both scored it on.

    *treatment* and *control* are ``(case id, logical reward)`` rows; *arms*
    names them in that order. *dropped_cases* maps a case to the arms whose load
    evidence failed: the case is left out of every metric. A case one arm never
    scored, an expected case no arm scored, and a trial without a case id are
    left out too. A metric one arm scored on no case is not compared at all.
    *judge_mixed_cases* maps a judge metric to the cases whose two arms were
    judged by different models: each is left out of that metric only.
    """
    rows = dict(zip(arms, (treatment, control), strict=True))
    by_case: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in arms}
    unidentified: dict[str, int] = dict.fromkeys(arms, 0)
    for arm, arm_rows in rows.items():
        for case_id, reward in arm_rows:
            if str(case_id) in _UNIDENTIFIED_CASES:
                unidentified[arm] += 1
            else:
                by_case[arm].setdefault(str(case_id), []).append(reward)
    dropped = {str(case): sorted({str(arm) for arm in failed}) for case, failed in (dropped_cases or {}).items()}
    judge_mixed = {str(metric): {str(case) for case in cases} for metric, cases in (judge_mixed_cases or {}).items()}
    judge_mixed_counts: dict[str, int] = {}
    universe = sorted({*by_case[arms[0]], *by_case[arms[1]], *(str(case) for case in expected_case_ids if case)})

    excluded_metrics: list[dict[str, Any]] = []
    compared: list[str] = []
    not_comparable: list[str] = []
    for metric in metrics:
        reasons: dict[str, list[str]] = {}
        for arm in arms:
            statuses = [_status(rewards, metric) for rewards in by_case[arm].values()]
            if statuses and None in statuses:
                continue
            reason = (
                EXCLUDED_NOT_APPLICABLE
                if statuses and all(status == EXCLUDED_NOT_APPLICABLE for status in statuses)
                else EXCLUDED_NO_VALUE
            )
            reasons.setdefault(reason, []).append(arm)
        if reasons:
            excluded_metrics.extend({"metric": metric, "reason": r, "arms": a} for r, a in reasons.items())
            if set(reasons) == {EXCLUDED_NOT_APPLICABLE}:
                not_comparable.append(metric)
        else:
            compared.append(metric)

    excluded: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
    shared: dict[str, set[str]] = {metric: set() for metric in compared}
    for case_id in universe:
        missing = tuple(arm for arm in arms if case_id not in by_case[arm])
        if case_id in dropped or missing:
            key = (case_id, EXCLUDED_LOAD_FAILED, tuple(dropped[case_id])) if case_id in dropped else None
            excluded[key or (case_id, EXCLUDED_NOT_SCORED, missing)] = list(compared)
            continue
        for metric in compared:
            if case_id in judge_mixed.get(metric, ()):
                excluded.setdefault((case_id, EXCLUDED_JUDGE_MIXED, tuple(arms)), []).append(metric)
                judge_mixed_counts[metric] = judge_mixed_counts.get(metric, 0) + 1
                continue
            statuses = {arm: _status(by_case[arm][case_id], metric) for arm in arms}
            if all(status is None for status in statuses.values()):
                shared[metric].add(case_id)
                continue
            for reason in dict.fromkeys(status for status in statuses.values() if status is not None):
                key = (case_id, reason, tuple(arm for arm in arms if statuses[arm] == reason))
                excluded.setdefault(key, []).append(metric)
    if any(unidentified.values()):
        key = ("", EXCLUDED_UNIDENTIFIED, tuple(arm for arm in arms if unidentified[arm]))
        excluded[key] = list(compared)

    means: dict[str, dict[str, float]] = {arm: {} for arm in arms}
    counts: dict[str, dict[str, Any]] = {}
    for metric in compared:
        trials: dict[str, int] = {}
        for arm in arms:
            # In trial order, as the arm average sums them, so an unchanged row keeps every digit.
            values = [
                value
                for case_id, reward in rows[arm]
                if str(case_id) in shared[metric] and (value := _scored(reward, metric)) is not None
            ]
            trials[arm] = len(values)
            if values:
                means[arm][metric] = round(sum(values) / len(values), 4)
        counts[metric] = {"cases": len(shared[metric]), "trials": trials}
        if not shared[metric]:
            excluded_metrics.append({"metric": metric, "reason": EXCLUDED_NO_SHARED_CASE, "arms": list(arms)})
            if all(
                reason == EXCLUDED_NOT_APPLICABLE
                for (_, reason, _), names in excluded.items()
                if metric in names and reason not in _CASE_LEVEL_REASONS
            ):
                not_comparable.append(metric)

    entries = [
        {
            "case": case_id or None,
            "reason": reason,
            "arms": list(names),
            "metrics": metric_names,
            **({"trials": {arm: unidentified[arm] for arm in names}} if reason == EXCLUDED_UNIDENTIFIED else {}),
        }
        for (case_id, reason, names), metric_names in excluded.items()
    ]
    basis = {
        "pairing": PAIRING_SAME_CASES,
        "arms": list(arms),
        "metrics": counts,
        "excluded_metrics": excluded_metrics,
        "excluded_cases": entries[:MAX_EXCLUDED_CASE_ENTRIES],
        "excluded_case_entries": len(entries),
        "excluded_cases_truncated": len(entries) > MAX_EXCLUDED_CASE_ENTRIES,
    }
    return PairedScores(
        treatment=means[arms[0]],
        control=means[arms[1]],
        not_comparable=tuple(not_comparable),
        basis=basis,
        judge_mixed=judge_mixed_counts,
    )
