# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reload explicit applicability and recompute comparisons from complete rewards."""

from __future__ import annotations

import json

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
