# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Tier 3 LLM-as-judge modules (deterministic paths).

The dimension judge and insights judge fall back to deterministic / empty
output when no LLM is available, so these tests exercise that path without any
network access.
"""

from __future__ import annotations

import pytest

from skillevaluator.constants import DIMENSION_MAPPING
from skillevaluator.evaluation import (
    build_insights,
    compute_dimensions,
    compute_dimensions_deterministic,
)


@pytest.fixture
def evaluators() -> dict:
    return {
        "security": {"with_skill": 0.9, "baseline": 0.8},
        "skill_execution": {"with_skill": 0.7, "baseline": 0.5},
        "skill_efficiency": {"with_skill": 0.6, "baseline": 0.6},
        "accuracy": {"with_skill": 0.8, "baseline": 0.7},
        "goal_accuracy": {"with_skill": 0.75, "baseline": 0.6},
        "behavior_check": {"with_skill": 0.85, "baseline": 0.8},
        "token_efficiency": {"with_skill": 0.5, "baseline": 0.4},
    }


class TestDimensionJudgeDeterministic:
    def test_produces_all_five_dimensions(self, evaluators: dict) -> None:
        dims = compute_dimensions_deterministic(evaluators)
        assert {d["id"] for d in dims} == set(DIMENSION_MAPPING)

    def test_scores_and_lift_present(self, evaluators: dict) -> None:
        dims = {d["id"]: d for d in compute_dimensions_deterministic(evaluators)}
        sec = dims["security"]
        assert sec["score"] == pytest.approx(0.9)
        assert sec["with_skill"] == pytest.approx(0.9)
        assert sec["baseline"] == pytest.approx(0.8)
        assert sec["lift"] == pytest.approx(0.1)
        assert sec["verdict"] == "PASS"
        assert sec["reasoning_bullets"]
        assert sec["explanation"]

    def test_verdict_thresholds(self, evaluators: dict) -> None:
        dims = {d["id"]: d for d in compute_dimensions_deterministic(evaluators)}
        # efficiency = 0.7*0.6 + 0.3*0.5 = 0.57 -> below 0.7 pass threshold
        assert dims["efficiency"]["verdict"] == "NEUTRAL"

    def test_baseline_absent_yields_none_lift(self) -> None:
        evaluators = {"security": {"with_skill": 0.9}}
        dims = {d["id"]: d for d in compute_dimensions_deterministic(evaluators)}
        assert dims["security"]["baseline"] is None
        assert dims["security"]["lift"] is None

    def test_compute_dimensions_no_llm_uses_deterministic(self, evaluators: dict) -> None:
        dims = compute_dimensions(evaluators, [], 0.1, use_llm=False)
        assert {d["id"] for d in dims} == set(DIMENSION_MAPPING)


class TestInsightsJudgeFallback:
    def test_build_insights_no_llm_returns_empty(self, evaluators: dict) -> None:
        deterministic = {"conclusions": [], "recommendations": []}
        canonical = {"dimensions": compute_dimensions_deterministic(evaluators)}
        out = build_insights(canonical, deterministic, use_llm=False)
        assert out == {"conclusions": [], "recommendations": []}
