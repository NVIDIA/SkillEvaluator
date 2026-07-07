# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden contract for Tier 3 report/schema anchors (Phase 4).

Pins the stable, namespace-independent schema identifiers and the pure
metric-set classification logic. Full live-run report goldens (pass_at_k,
attempt_policy, run_config, infrastructure_errors, artifact_manifest.json,
failure.json, harbor_multi_step, final_metrics.extra token counters) require a
recorded Harbor run fixture and are tracked as a CI-recorded follow-up.
"""

from __future__ import annotations

from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    metric_set_for_reward,
    score_definition,
)


def test_metric_set_identifier_is_stable() -> None:
    assert DEFAULT_METRIC_SET == "skill-evaluator-default-v2"


def test_default_metrics_include_core_dimensions() -> None:
    assert DEFAULT_METRICS == (
        "security",
        "skill_execution",
        "skill_efficiency",
        "accuracy",
        "goal_accuracy",
        "behavior_check",
    )


def test_metric_set_for_reward_classifies_default_v2() -> None:
    metric_set, metrics = metric_set_for_reward({"metric_set": "skill-evaluator-default-v2", "scores": {}})
    assert metric_set == DEFAULT_METRIC_SET
    assert metrics == DEFAULT_METRICS


def test_metric_set_for_reward_defaults_to_skill_evaluator() -> None:
    metric_set, metrics = metric_set_for_reward({})
    assert metric_set == DEFAULT_METRIC_SET
    assert metrics == DEFAULT_METRICS


def test_score_definition_mentions_metrics() -> None:
    definition = score_definition(DEFAULT_METRICS)
    assert isinstance(definition, str)
    assert "security" in definition
