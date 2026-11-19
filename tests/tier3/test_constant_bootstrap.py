# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constant case deltas need no random draws and keep the same bootstrap output."""

from __future__ import annotations

import math
import random

import pytest

from skillevaluator.tier3.harbor import stats


@pytest.mark.parametrize("value", [0.0, 0.1, -0.3, 1.0])
def test_constant_resamples_match_explicit_draws_without_sampling(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    deltas = [value] * 13
    rng = random.Random(42)
    expected = [math.fsum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(30)]

    def unexpected_draw(*args, **kwargs):
        pytest.fail("a constant population does not need random choices")

    monkeypatch.setattr(random.Random, "choices", unexpected_draw)

    assert stats._resample_means(deltas, resamples=30, seed=42) == expected


def test_constant_ladder_keeps_bootstrap_results_without_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [f"case-{index}" for index in range(10)]
    scores = {
        stats.ARM_WITHOUT: dict.fromkeys(cases, 0.0),
        stats.ARM_SUM_OF_PARTS: dict.fromkeys(cases, 0.25),
        stats.ARM_LOOSE_PARTS: dict.fromkeys(cases, 0.5),
        stats.ARM_WITH: dict.fromkeys(cases, 0.75),
    }
    expected = {
        name: stats.paired_case_bootstrap(scores[treatment], scores[control])
        for name, (treatment, control) in {**stats.LADDER_STEPS, "bundle": stats.LADDER_BUNDLE}.items()
    }

    def unexpected_draw(*args, **kwargs):
        pytest.fail("constant ladder steps do not need random choices")

    monkeypatch.setattr(random.Random, "choices", unexpected_draw)
    steps, bundle, skill = stats._ladder_rungs(scores, cases)

    for name, step in steps.items():
        assert step == stats._step_fields(expected[name], split=False)
    assert bundle == stats._step_fields(expected["bundle"], split=False)
    assert skill == expected["skill"]


def test_mixed_ladder_steps_keep_the_seeded_case_indices() -> None:
    cases = [f"case-{index}" for index in range(10)]
    scores = {
        stats.ARM_WITHOUT: dict.fromkeys(cases, 0.0),
        stats.ARM_SUM_OF_PARTS: dict.fromkeys(cases, 0.25),
        stats.ARM_LOOSE_PARTS: {case: 0.25 + index / 100 for index, case in enumerate(cases)},
        stats.ARM_WITH: {case: 0.5 + index / 100 for index, case in enumerate(cases)},
    }

    steps, bundle, _ = stats._ladder_rungs(scores, cases)

    for name, (treatment, control) in {**stats.LADDER_STEPS, "bundle": stats.LADDER_BUNDLE}.items():
        expected = stats.paired_case_bootstrap(scores[treatment], scores[control])
        actual = bundle if name == "bundle" else steps[name]
        assert actual == stats._step_fields(expected, split=False)
