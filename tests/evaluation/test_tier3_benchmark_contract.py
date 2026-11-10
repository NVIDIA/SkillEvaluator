# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for BENCHMARK.md's canonical Tier 3 data contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator import __version__
from skillevaluator.constants import (
    DEFAULT_SCORE_POLICY,
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
from skillevaluator.tier3.harbor.metrics import LEGACY_METRICS, LEGACY_SCORE_POLICY
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
    assert payload["score_policy"] == DEFAULT_SCORE_POLICY
    assert payload["attempt_policy"]["score_policy"] == DEFAULT_SCORE_POLICY
    assert payload["dataset_summary"] == {
        "total_tasks": 3,
        "positive_tasks": 1,
        "negative_tasks": 1,
        "unclassified_tasks": 1,
        "source": "dataset",
    }
    assert payload["verdict_policy"] == {
        "attempt_pass_threshold": 0.6,
        "goal_accuracy_pass_threshold": 1.0,
        "security_pass_threshold": 1.0,
        "attempt_pass_rule": "overall_threshold_and_goal_accuracy_and_security_threshold_when_scored",
        "run_fail_policy": "any_with_skill_required_check_failure",
        "dimension_pass_threshold": DIMENSION_VERDICT_PASS_THRESHOLD,
        "dimension_neutral_threshold": DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
        "lift_pass_threshold": TIER3_LIFT_PASS_THRESHOLD,
        "lift_fail_threshold": TIER3_LIFT_FAIL_THRESHOLD,
        "overall_pass_rule": "one_supported_agent_all_dimensions_pass",
    }
    assert {"required_checks", "correct_answers", "security_clean"} <= payload.keys()
    assert payload["summary"]["dataset_summary"] == payload["dataset_summary"]
    assert payload["summary"]["score_policy"] == payload["score_policy"]
    assert payload["summary"]["verdict_policy"] == payload["verdict_policy"]


def test_legacy_payload_preserves_the_versioned_metric_mean() -> None:
    scores = {
        "skill_execution": 0.2,
        "skill_efficiency": 0.4,
        "accuracy": 0.6,
        "goal_accuracy": 0.8,
        "behavior_check": 1.0,
    }

    payload = build_agent_eval_payload(
        "legacy-demo",
        {
            "opencode": {
                "execution_status": "succeeded",
                "metrics_with_skill": list(LEGACY_METRICS),
                "with_skill": scores,
            }
        },
        use_llm_judge=False,
    )

    assert payload is not None
    assert payload["overall_score"] == 0.6
    assert payload["score_policy"] == LEGACY_SCORE_POLICY


def test_advisory_payload_exposes_same_report_truth_metadata() -> None:
    payload = _advisory_agent_eval_payload(
        "Tier 3 was skipped",
        n_attempts=3,
        pass_threshold=0.6,
        stop_on_pass=True,
    )

    assert payload["evaluated_at"] is None
    assert payload["evaluator_version"] == __version__
    assert payload["score_policy"] == DEFAULT_SCORE_POLICY
    assert payload["evaluated_source"] is None
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
        "score_policy",
        "evaluated_source",
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


@pytest.mark.parametrize(
    ("routing_fields", "positive", "negative", "unclassified"),
    [
        pytest.param({"expected_skill": "demo"}, 1, 0, 0, id="legacy-positive"),
        pytest.param({"expected_skill": "demo", "should_trigger": False}, 0, 1, 0, id="explicit-negative"),
        pytest.param({"expected_skill": None, "should_trigger": True}, 1, 0, 0, id="explicit-positive"),
        pytest.param({"should_trigger": True}, 1, 0, 0, id="positive-label-only"),
        pytest.param({"should_trigger": False}, 0, 1, 0, id="negative-label-only"),
        pytest.param({"expected_skill": None}, 0, 1, 0, id="legacy-negative"),
        pytest.param({}, 0, 0, 1, id="unlabeled"),
    ],
)
def test_dataset_truth_respects_explicit_routing_intent(
    routing_fields: dict, positive: int, negative: int, unclassified: int
) -> None:
    dataset = [{"id": "case-1", **routing_fields}]
    expected_summary = {
        "total_tasks": 1,
        "positive_tasks": positive,
        "negative_tasks": negative,
        "unclassified_tasks": unclassified,
        "source": "dataset",
    }

    snapshot = build_dataset_snapshot(dataset, evaluator_version=__version__)
    payload = build_agent_eval_payload("demo", _agents(), dataset=dataset, use_llm_judge=False)

    assert snapshot["dataset_summary"] == expected_summary
    assert payload is not None
    assert payload["dataset_summary"] == expected_summary
    assert payload["summary"]["dataset_summary"] == expected_summary


def test_dataset_snapshot_rejects_duplicate_or_tampered_dataset(tmp_path: Path) -> None:
    entry = {"id": "case-1", "prompt": "Run the skill", "expected_skill": "demo"}
    snapshot = build_dataset_snapshot([entry], evaluator_version="0.8.2")
    snapshot["dataset"].append(dict(entry))
    (tmp_path / "dataset_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    assert load_dataset_snapshot(tmp_path) is None


def test_dataset_snapshot_migrates_valid_legacy_routing_summary_in_memory(tmp_path: Path) -> None:
    # This fixture was emitted by the base revision's schema 1.0 builder.
    original = (Path(__file__).parents[1] / "fixtures" / "legacy-routing-dataset-snapshot-1.0.json").read_bytes()
    persisted = tmp_path / "dataset_snapshot.json"
    persisted.write_bytes(original)
    historical = json.loads(original)

    loaded = load_dataset_snapshot(tmp_path)

    assert loaded is not None
    assert loaded["schema_version"] == "1.1"
    assert loaded["dataset"] == historical["dataset"]
    assert loaded["dataset_digest"] == historical["dataset_digest"]
    assert loaded["evaluator_version"] == "0.8.0"
    assert loaded["dataset_summary"] == {
        "total_tasks": 3,
        "positive_tasks": 1,
        "negative_tasks": 2,
        "unclassified_tasks": 0,
        "source": "dataset",
    }
    assert persisted.read_bytes() == original


@pytest.mark.parametrize("schema_version", ["1.0", "1.1"])
@pytest.mark.parametrize("tamper", ["summary-rule", "dataset", "digest", "algorithm"])
def test_dataset_snapshot_rejects_tampered_versioned_truth(tmp_path: Path, schema_version: str, tamper: str) -> None:
    snapshot = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "legacy-routing-dataset-snapshot-1.0.json").read_text(
            encoding="utf-8"
        )
    )
    legacy_summary = dict(snapshot["dataset_summary"])
    current_summary = {
        "total_tasks": 3,
        "positive_tasks": 1,
        "negative_tasks": 2,
        "unclassified_tasks": 0,
        "source": "dataset",
    }
    snapshot["schema_version"] = schema_version
    if schema_version == "1.1":
        snapshot["dataset_summary"] = current_summary
    if tamper == "summary-rule":
        snapshot["dataset_summary"] = current_summary if schema_version == "1.0" else legacy_summary
    elif tamper == "dataset":
        snapshot["dataset"][0]["prompt"] = "Changed after evaluation"
    elif tamper == "digest":
        snapshot["dataset_digest"] = "sha256:" + "0" * 64
    else:
        snapshot["dataset_digest_algorithm"] = "unknown-digest"
    (tmp_path / "dataset_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    assert load_dataset_snapshot(tmp_path) is None


def test_new_dataset_snapshot_versions_corrected_routing_summary() -> None:
    snapshot = build_dataset_snapshot(
        [{"id": "negative", "expected_skill": "demo", "should_trigger": False}],
        evaluator_version=__version__,
    )

    assert snapshot["schema_version"] == "1.1"
    assert snapshot["dataset_summary"]["positive_tasks"] == 0
    assert snapshot["dataset_summary"]["negative_tasks"] == 1


@pytest.mark.parametrize("schema_version", [[], {}, "9.9"], ids=["array", "object", "unknown"])
def test_dataset_snapshot_rejects_invalid_schema(tmp_path: Path, schema_version: object) -> None:
    snapshot = build_dataset_snapshot([{"id": "case-1", "expected_skill": "demo"}], evaluator_version=__version__)
    snapshot["schema_version"] = schema_version
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
