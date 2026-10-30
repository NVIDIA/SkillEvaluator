# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reload explicit applicability and recompute comparisons from complete rewards."""

from __future__ import annotations

import json

import pytest

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    DEFAULT_SCORE_POLICY,
    average_metrics,
    dimension_scores,
    not_applicable_metrics,
    overall_score,
)
from skillevaluator.tier3.harbor.report_data import load_agent_data


def _reward(case, *, accuracy, baseline=False):
    row = dict.fromkeys(DEFAULT_METRICS, 1.0)
    row.update(
        entry_id=case,
        trial_id=f"{case}-trial",
        metric_set=DEFAULT_METRIC_SET,
        score_policy=DEFAULT_SCORE_POLICY,
        details={},
    )
    row["accuracy"] = accuracy
    row["behavior_check"] = None
    if baseline:
        row["skill_execution"] = row["skill_efficiency"] = None
    for metric in DEFAULT_METRICS:
        if row[metric] is None:
            row["details"][metric] = {"score": None, "status": "not_applicable", "reason": "No reference"}
    return row


def _run(root):
    for condition, rows in {
        "with-skill": [_reward("a", accuracy=0.9), _reward("b", accuracy=None)],
        "without-skill": [_reward("a", accuracy=0.6, baseline=True), _reward("b", accuracy=0.0, baseline=True)],
    }.items():
        scores = average_metrics(rows)[0]
        na = not_applicable_metrics(rows)
        directory = root / "codex" / condition
        directory.mkdir(parents=True)
        summary = {
            "scores": scores,
            "metrics": list(DEFAULT_METRICS),
            "metric_set": DEFAULT_METRIC_SET,
            "score_policy": DEFAULT_SCORE_POLICY,
            "not_applicable_metrics": na,
            "dimensions": dimension_scores(scores, na),
            "num_trials": 2,
            "num_reward_rows": 2,
            "execution_status": "succeeded",
            "overall_score": sum(overall_score(row) for row in rows) / 2,
        }
        (directory / "summary.json").write_text(json.dumps(summary))
        for row in rows:
            trial = directory / "trials" / row["entry_id"]
            trial.mkdir(parents=True)
            (trial / "reward.json").write_text(json.dumps(row))
    (root / "codex/lift.json").write_text(
        json.dumps(
            {
                "accuracy": {"with_skill": 0.9, "without_skill": 0.3, "delta": 0.6},
                "overall": {"delta": 0.99},
                "basis": {"pairing": "same_cases_per_metric", "metrics": {"accuracy": {"cases": 999}}},
            }
        )
    )
    return root


def _report(root):
    payload = build_agent_eval_payload("demo", load_agent_data(root), use_llm_judge=False)
    assert payload is not None
    return payload


def test_saved_run_retains_validated_na_and_recomputes_tampered_basis(tmp_path):
    root = _run(tmp_path)
    loaded = load_agent_data(root)["codex"]
    assert loaded["not_applicable_with_skill"] == ["behavior_check"]
    assert loaded["not_applicable_without_skill"] == ["skill_execution", "skill_efficiency", "behavior_check"]
    payload = _report(root)
    agent = payload["agents"]["codex"]
    assert agent["with_skill"] == 0.98
    assert agent["evaluators"]["behavior_check"]["with_skill_status"] == "not_applicable"
    assert agent["evaluators"]["behavior_check"]["with_skill"] is None
    assert agent["evaluators"]["skill_execution"]["baseline_status"] == "not_applicable"
    assert agent["evaluators"]["accuracy"]["lift"] == 0.3
    assert agent["lift_basis"]["metrics"]["accuracy"]["cases"] == 1
    assert not any(card["id"] == "behavior_check" for card in agent["evaluator_cards"])
    assert any(item["id"] == "behavior_check" for item in agent["not_applicable_evaluators"])
    assert agent["lift"] == 0.1


def test_matching_saved_basis_and_metric_lifts_cannot_authorize_a_tampered_overall(tmp_path):
    from skillevaluator.tier3.harbor.collector import _compute_lift

    root = _run(tmp_path)
    loaded = load_agent_data(root)["codex"]
    assert loaded["rewards_complete"] is loaded["rewards_baseline_complete"] is True
    cached = _compute_lift(
        loaded["with_skill"],
        loaded["without_skill"],
        loaded["rewards"],
        loaded["rewards_baseline"],
        score_policy=DEFAULT_SCORE_POLICY,
    )
    assert cached["overall"]["delta"] == 0.1
    # The basis and every per-metric value remain exactly as rebuilt from the
    # complete rewards. Only the cached headline delta has been changed.
    cached["overall"]["delta"] = 0.99
    (root / "codex/lift.json").write_text(json.dumps(cached))
    assert _report(root)["agents"]["codex"]["lift"] == 0.1


@pytest.mark.parametrize(
    "unavailable",
    [
        {"comparison_unavailable": True},
        {"comparison_complete": False},
        {"lift": {"comparison_unavailable": True}},
    ],
)
def test_explicit_unavailable_comparison_suppresses_aggregate_overall_lift(unavailable):
    from skillevaluator.evaluation.tier3_report import _build_agent

    agent = _build_agent(
        "codex",
        {
            "execution_status": "succeeded",
            "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
            "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.6),
            **unavailable,
        },
        list(DEFAULT_METRICS),
        list(DEFAULT_METRICS),
        None,
        DEFAULT_SCORE_POLICY,
    )
    assert agent["lift"] is None


def test_numeric_contradictions_override_tampered_aggregate_na_markers(tmp_path):
    root = _run(tmp_path)
    path = root / "codex/with-skill/summary.json"
    summary = json.loads(path.read_text())
    summary["not_applicable_metrics"] = ["accuracy", "security", "behavior_check"]
    path.write_text(json.dumps(summary))
    loaded = load_agent_data(root)["codex"]
    assert loaded["not_applicable_with_skill"] == ["behavior_check"]
    payload = _report(root)
    row = payload["agents"]["codex"]["evaluators"]["accuracy"]
    assert row["with_skill_status"] == "scored" and row["with_skill"] == 0.9


def test_new_basis_comparison_is_unavailable_when_rewards_are_incomplete(tmp_path):
    root = _run(tmp_path)
    (root / "codex/with-skill/trials/b/reward.json").unlink()
    payload = _report(root)
    agent = payload["agents"]["codex"]
    assert agent["lift"] is None
    assert agent["evaluators"]["accuracy"]["lift"] is None
    assert not agent.get("lift_basis")


def test_incomplete_saved_full_metric_basis_cannot_fall_back_to_aggregate_scores(tmp_path):
    root = _run(tmp_path)
    for condition in ("with-skill", "without-skill"):
        directory = root / "codex" / condition
        scores = dict.fromkeys(DEFAULT_METRICS, 1.0)
        summary = json.loads((directory / "summary.json").read_text())
        summary.update(scores=scores, not_applicable_metrics=[], dimensions=dimension_scores(scores))
        (directory / "summary.json").write_text(json.dumps(summary))
        for path in (directory / "trials").glob("*/reward.json"):
            row = json.loads(path.read_text())
            row.update(scores, details={})
            path.write_text(json.dumps(row))
    (root / "codex/lift.json").write_text(
        json.dumps(
            {
                "overall": {"delta": 0.99},
                "basis": {
                    "excluded_case_entries": 0,
                    "excluded_cases": [],
                    "excluded_metrics": [],
                    "excluded_cases_truncated": False,
                },
            }
        )
    )
    (root / "codex/with-skill/trials/b/reward.json").unlink()
    assert _report(root)["agents"]["codex"]["lift"] is None


def test_complete_mixed_saved_run_rebuilds_logical_lift_and_rejects_different_case_cohorts(tmp_path):
    from skillevaluator.tier3.harbor.metrics import CUSTOM_ONLY_METRIC_SET

    for condition, standard_score in (("with-skill", 1.0), ("without-skill", 0.6)):
        directory = tmp_path / "codex" / condition
        directory.mkdir(parents=True)
        scores = dict.fromkeys(DEFAULT_METRICS, standard_score)
        summary = {
            "scores": scores,
            "metrics": list(DEFAULT_METRICS),
            "score_policy": DEFAULT_SCORE_POLICY,
            "num_trials": 2,
            "num_reward_rows": 2,
            "mixed_metric_contracts": True,
            "execution_status": "succeeded",
            "overall_score": standard_score / 2,
        }
        (directory / "summary.json").write_text(json.dumps(summary))
        rows = [
            {**scores, "entry_id": "a", "trial_id": "a", "metric_set": DEFAULT_METRIC_SET},
            {"entry_id": "b", "trial_id": "b", "metric_set": CUSTOM_ONLY_METRIC_SET, "overall": 0.0},
        ]
        for row in rows:
            trial = directory / "trials" / row["entry_id"]
            trial.mkdir(parents=True)
            (trial / "reward.json").write_text(json.dumps(row))
    (tmp_path / "codex/lift.json").write_text(
        json.dumps(
            {
                "overall": {"delta": 0.99},
                "basis": {"pairing": "same_cases_per_metric"},
            }
        )
    )
    assert _report(tmp_path)["agents"]["codex"]["lift"] == 0.2
    path = tmp_path / "codex/without-skill/trials/b/reward.json"
    row = json.loads(path.read_text())
    row["entry_id"] = "different-case"
    path.write_text(json.dumps(row))
    assert _report(tmp_path)["agents"]["codex"]["lift"] is None


def test_default_dimensions_require_every_applicable_metric_even_with_cached_scores():
    from skillevaluator.evaluation.tier3_report import _build_agent

    incomplete = dict.fromkeys(DEFAULT_METRICS, 1.0)
    incomplete.pop("behavior_check")
    assert "effectiveness" not in dimension_scores(incomplete)
    for cached in ({}, {"effectiveness": {"score": 1.0, "verdict": "PASS"}}):
        agent = _build_agent(
            "codex",
            {
                "execution_status": "succeeded",
                "with_skill": incomplete,
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.4),
                "dimensions_with_skill": cached,
            },
            list(DEFAULT_METRICS),
            list(DEFAULT_METRICS),
            None,
            DEFAULT_SCORE_POLICY,
        )
        effectiveness = next(row for row in agent["dimensions"] if row["id"] == "effectiveness")
        assert effectiveness["with_skill"] is None
        assert effectiveness["baseline"] == 0.4
        assert effectiveness["verdict"] is None
        assert effectiveness["lift"] is None
        assert agent["with_skill"] is None


def test_security_fallback_requires_a_legacy_metric_contract():
    from skillevaluator.evaluation.tier3_report import _build_agent
    from skillevaluator.tier3.harbor.metrics import LEGACY_METRICS, LEGACY_SCORE_POLICY, PARTIAL_SCORE_POLICY

    scores = dict.fromkeys(LEGACY_METRICS, 0.7)
    for contract, policy, expected in (
        (DEFAULT_METRICS, DEFAULT_SCORE_POLICY, None),
        (LEGACY_METRICS, LEGACY_SCORE_POLICY, 0.7),
        (("accuracy",), PARTIAL_SCORE_POLICY, None),
    ):
        agent = _build_agent(
            "codex",
            {
                "execution_status": "succeeded",
                "with_skill": scores,
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.4),
            },
            list(contract),
            list(DEFAULT_METRICS),
            None,
            policy,
        )
        safety = next(row for row in agent["dimensions"] if row["id"] == "security")
        assert safety["with_skill"] == expected


def test_saved_run_does_not_renormalize_a_missing_metric_without_validated_na(tmp_path):
    root = _run(tmp_path)
    for condition in ("with-skill", "without-skill"):
        for path in (root / "codex" / condition / "trials").glob("*/reward.json"):
            reward = json.loads(path.read_text())
            reward["details"].pop("behavior_check")
            path.write_text(json.dumps(reward))
    payload = _report(root)
    agent = payload["agents"]["codex"]
    assert "behavior_check" not in load_agent_data(root)["codex"]["not_applicable_with_skill"]
    assert agent["evaluators"]["behavior_check"]["with_skill_status"] == "missing"
    assert agent["with_skill"] is None
    assert not any(row["id"] == "effectiveness" and row["with_skill"] is not None for row in agent["dimensions"])
