# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M12: lift intervals were overconfident at small n, and "adequate" said so.

Proof example check-14 edge-05 drew cases from 39 real per-case lift deltas
and asked for the 95% interval: it held the true lift only 82% of the time at
5 cases, and 48% of the intervals labelled "adequate" at 5 cases missed it.
The interval is now an expanded percentile bootstrap (wider at small n), and
"adequate" needs at least 10 paired cases as well as a narrow interval.
"""

from __future__ import annotations

import random
import statistics

import pytest

from skillevaluator.tier3.harbor import stats

# The 39 per-case lift deltas of proof example check-14 edge-05 (population.json).
POPULATION = [
    0.5067, 0.6742, 0.57, 0.3905, 0.324, 0.3891, 0.3752, 0.3876, 0.0, 0.2945, 0.4719, 0.2325, 0.5522,
    0.5289, 0.3061, 0.2702, 0.1871, 0.0333, 0.2187, 0.4362, 0.1843, 0.4144, 0.3971, 0.0428, 0.3301,
    0.338, -0.0173, 0.1923, 0.34, 0.1539, 0.4773, 0.3624, 0.0, 0.1567, 0.0, 0.0333, 0.15, 0.2867, 0.0,
]  # fmt: skip


def _simulate(n_cases: int, sims: int, seed: int) -> tuple[float, list[str]]:
    truth = statistics.fmean(POPULATION)
    rng = random.Random(seed)
    covered = 0
    labels: list[str] = []
    for _ in range(sims):
        deltas = [rng.choice(POPULATION) for _ in range(n_cases)]
        treatment = {f"c{index}": delta for index, delta in enumerate(deltas)}
        interval = stats.paired_case_bootstrap(treatment, dict.fromkeys(treatment, 0.0))
        covered += interval["ci_low"] <= truth <= interval["ci_high"]
        labels.append(interval["precision"])
    return covered / sims, labels


@pytest.mark.parametrize(("n_cases", "minimum_coverage"), [(5, 0.87), (9, 0.90)])
def test_small_n_intervals_cover_the_true_lift(n_cases: int, minimum_coverage: float) -> None:
    coverage, labels = _simulate(n_cases, sims=400, seed=2026)

    # Before the fix: 81.8% at 5 cases and 89.6% at 9 (proof edge-05, 1,000 runs).
    assert coverage >= minimum_coverage
    # A handful of cases is never "adequate", however narrow the interval looks.
    assert "adequate" not in labels


def test_adequate_needs_enough_cases_as_well_as_a_narrow_interval() -> None:
    assert stats.lift_precision(9, 0.30, 0.35) == "low"
    assert stats.lift_precision(10, 0.30, 0.35) == "adequate"
    assert stats.lift_precision(30, 0.10, 0.35) == "low"
    assert stats.lift_precision(4, 0.30, 0.31) == "insufficient"


def test_expanded_interval_is_wider_at_small_n_and_converges() -> None:
    assert stats.expanded_tail(5, 0.95) < stats.expanded_tail(9, 0.95) < stats.expanded_tail(100, 0.95) < 0.025
    assert stats.expanded_tail(100_000, 0.95) == pytest.approx(0.025, abs=1e-4)
    # The t quantiles behind the expansion match the textbook values.
    assert stats._student_t_quantile(0.975, 4) == pytest.approx(2.7764, abs=1e-4)
    assert stats._student_t_quantile(0.975, 8) == pytest.approx(2.3060, abs=1e-4)

    deltas = {f"c{index}": value for index, value in enumerate(POPULATION[:6])}
    interval = stats.paired_case_bootstrap(deltas, dict.fromkeys(deltas, 0.0))
    assert interval["interval"] == "expanded_percentile"
    assert interval["ci_low"] <= interval["estimate"] <= interval["ci_high"]
    assert interval["precision"] == "low"
