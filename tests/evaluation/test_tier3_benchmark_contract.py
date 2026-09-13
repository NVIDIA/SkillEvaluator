# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for BENCHMARK.md's canonical Tier 3 data contract."""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator import __version__
from skillevaluator.constants import (
    DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
    DIMENSION_VERDICT_PASS_THRESHOLD,
    TIER3_LIFT_FAIL_THRESHOLD,
    TIER3_LIFT_PASS_THRESHOLD,
)
from skillevaluator.evaluation.tier3_report import (
    VERDICT_FAIL,
    VERDICT_NEUTRAL,
    VERDICT_PASS,
    _advisory_agent_eval_payload,
    _verdict_from_lift,
    agent_eval_result_from_run,
    build_agent_eval_payload,
)
from skillevaluator.tier3.harbor.report_data import build_dataset_snapshot, load_dataset_snapshot


def _agents() -> dict:
    return {
        "codex": {
            "with_skill": {
                "security": 1.0,
                "skill_execution": 0.9,
                "skill_efficiency": 0.8,
                "accuracy": 0.8,
                "goal_accuracy": 0.8,
                "behavior_check": 0.8,
            },
            "without_skill": {
                "security": 0.9,
                "skill_execution": 0.6,
                "skill_efficiency": 0.6,
                "accuracy": 0.5,
                "goal_accuracy": 0.5,
                "behavior_check": 0.5,
            },
            "execution_status": "succeeded",
            "rewards": [],
            "num_trials": 2,
        }
    }


def test_lift_verdict_uses_shared_asymmetric_policy() -> None:
    assert _verdict_from_lift(TIER3_LIFT_PASS_THRESHOLD) == VERDICT_PASS
    assert _verdict_from_lift(TIER3_LIFT_FAIL_THRESHOLD) == VERDICT_FAIL
    assert _verdict_from_lift(-0.05) == VERDICT_NEUTRAL


def test_payload_exposes_report_truth_metadata() -> None:
    dataset = [
        {"id": "positive", "expected_skill": "demo"},
        {"id": "negative", "expected_skill": None},
        {"id": "legacy"},
    ]
    attempt_policy = {"max_attempts": 3, "pass_threshold": 0.6, "stop_on_pass": True}

    payload = build_agent_eval_payload(
        "demo",
        _agents(),
        dataset=dataset,
        attempt_policy=attempt_policy,
        evaluated_at="2026-07-24T12:30:00+00:00",
        use_llm_judge=False,
    )

    assert payload is not None
    assert payload["evaluated_at"] == "2026-07-24T12:30:00+00:00"
    assert payload["evaluator_version"] == __version__
    assert payload["dataset_digest"].startswith("sha256:")
    assert payload["dataset_digest_algorithm"] == "skill-evaluator-dataset-snapshot/1"
    assert payload["dataset_summary"] == {
        "total_tasks": 3,
        "positive_tasks": 1,
        "negative_tasks": 1,
        "unclassified_tasks": 1,
        "source": "dataset",
    }
    assert payload["verdict_policy"] == {
        "attempt_pass_threshold": 0.6,
        "dimension_pass_threshold": DIMENSION_VERDICT_PASS_THRESHOLD,
        "dimension_neutral_threshold": DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
        "lift_pass_threshold": TIER3_LIFT_PASS_THRESHOLD,
        "lift_fail_threshold": TIER3_LIFT_FAIL_THRESHOLD,
        "overall_pass_rule": "one_supported_agent_all_dimensions_pass",
    }
    assert payload["summary"]["dataset_summary"] == payload["dataset_summary"]
    assert payload["summary"]["verdict_policy"] == payload["verdict_policy"]


def test_advisory_payload_exposes_same_report_truth_metadata() -> None:
    payload = _advisory_agent_eval_payload(
        "Tier 3 was skipped",
        n_attempts=3,
        pass_threshold=0.6,
        stop_on_pass=True,
    )

    assert payload["evaluated_at"] is None
    assert payload["evaluator_version"] == __version__
    assert payload["dataset_summary"] == {
        "total_tasks": 0,
        "positive_tasks": 0,
        "negative_tasks": 0,
        "unclassified_tasks": 0,
        "source": "unavailable",
    }
    assert payload["verdict_policy"]["attempt_pass_threshold"] == 0.6
    for field in (
        "evaluated_at",
        "evaluator_version",
        "dataset_summary",
        "dataset_digest",
        "dataset_digest_algorithm",
        "verdict_policy",
    ):
        assert payload["summary"][field] == payload[field]


def test_dataset_truth_counts_unique_task_identities() -> None:
    duplicate_without_id = {"prompt": "legacy case", "expected_skill": None}
    payload = build_agent_eval_payload(
        "demo",
        _agents(),
        dataset=[
            {"id": "same-task", "expected_skill": "demo"},
            {"id": "same-task", "expected_skill": "demo", "prompt": "duplicate"},
            duplicate_without_id,
            dict(duplicate_without_id),
        ],
        use_llm_judge=False,
    )

    assert payload is not None
    assert payload["dataset_summary"] == {
        "total_tasks": 2,
        "positive_tasks": 1,
        "negative_tasks": 1,
        "unclassified_tasks": 0,
        "source": "dataset",
    }
    assert len(payload["dataset"]) == 2


def test_dataset_snapshot_rejects_duplicate_or_tampered_dataset(tmp_path: Path) -> None:
    entry = {"id": "case-1", "prompt": "Run the skill", "expected_skill": "demo"}
    snapshot = build_dataset_snapshot([entry], evaluator_version="0.8.2")
    snapshot["dataset"].append(dict(entry))
    (tmp_path / "dataset_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    assert load_dataset_snapshot(tmp_path) is None


def test_overall_verdict_requires_every_dimension_even_with_passing_lift() -> None:
    agents = _agents()
    agents["codex"]["with_skill"]["goal_accuracy"] = 0.3
    agents["codex"]["with_skill"]["behavior_check"] = 0.3
    agents["codex"]["without_skill"]["goal_accuracy"] = 0.0
    agents["codex"]["without_skill"]["behavior_check"] = 0.0

    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)

    assert payload is not None
    assert payload["overall_lift"] >= TIER3_LIFT_PASS_THRESHOLD
    effectiveness = next(dimension for dimension in payload["dimensions"] if dimension["id"] == "effectiveness")
    assert effectiveness["verdict"] == "FAIL"
    assert payload["verdict"] == VERDICT_FAIL


def test_overall_passes_when_one_supported_agent_passes_every_dimension() -> None:
    agents = _agents()
    agents["codex"]["with_skill"]["goal_accuracy"] = 0.3
    agents["codex"]["with_skill"]["behavior_check"] = 0.3
    agents["opencode"] = _agents()["codex"]

    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)

    assert payload is not None
    assert payload["verdict"] == VERDICT_PASS


def test_overall_dimension_gate_is_not_overridden_by_neutral_lift() -> None:
    agents = _agents()
    agents["codex"]["without_skill"] = dict(agents["codex"]["with_skill"])

    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)

    assert payload is not None
    assert payload["overall_lift"] == 0.0
    assert payload["verdict"] == VERDICT_PASS


def test_run_wrapper_defers_timestamp_lookup_to_directory_loader(monkeypatch, tmp_path: Path) -> None:
    from skillevaluator.evaluation import tier3_report
    from skillevaluator.tier3 import results_location

    skill = tmp_path / "demo"
    run_dir = tmp_path / "results" / "20260724_120000"
    skill.mkdir()
    run_dir.mkdir(parents=True)
    expected = object()
    captured: dict = {}

    monkeypatch.setattr(results_location, "resolve_latest_results", lambda *_args, **_kwargs: run_dir)
    monkeypatch.setattr(
        tier3_report,
        "_evaluated_at_from_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("timestamp was read eagerly")),
    )

    def load_directory(*_args, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(tier3_report, "agent_eval_result_from_directory", load_directory)

    result = agent_eval_result_from_run(skill)

    assert result is expected
    assert "evaluated_at" not in captured
