# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unscored Tier 3 conditions keep the requested grading policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.tier3.harbor import report_data
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.metrics import (
    CUSTOM_ONLY_METRIC_SET,
    CUSTOM_SCORE_POLICY,
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    DEFAULT_SCORE_POLICY,
    score_definition,
)


def _expected_contract(grading_mode: str | None) -> tuple[str, tuple[str, ...], str]:
    if grading_mode == "custom_only":
        return CUSTOM_ONLY_METRIC_SET, (), CUSTOM_SCORE_POLICY
    return DEFAULT_METRIC_SET, DEFAULT_METRICS, DEFAULT_SCORE_POLICY


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_failed_job(job_dir: Path) -> None:
    """Persist a Harbor job whose only trial failed before producing a reward."""
    _write_json(
        job_dir / "result.json",
        {
            "n_total_trials": 1,
            "stats": {
                "n_trials": 1,
                "n_errors": 1,
                "evals": {"eval": {"n_trials": 0, "n_errors": 1, "reward_stats": {}}},
            },
        },
    )
    _write_json(
        job_dir / "case-001__attempt" / "result.json",
        {
            "task_name": "case-001",
            "trial_name": "case-001__attempt",
            "exception_info": {"exception_type": "RuntimeError", "exception_message": "sandbox connection lost"},
        },
    )


def _write_scored_job(job_dir: Path, reward: dict[str, object]) -> None:
    _write_json(
        job_dir / "result.json",
        {
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 1,
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
                "n_retries": 0,
                "evals": {
                    "eval": {"n_trials": 1, "n_errors": 0, "reward_stats": {"reward": {"0.75": ["case-001__attempt"]}}}
                },
            },
        },
    )
    _write_json(job_dir / "case-001__attempt" / "result.json", {"trial_name": "case-001__attempt"})
    _write_json(job_dir / "case-001__attempt" / "verifier" / "reward.json", reward)


@pytest.mark.parametrize("grading_mode", [None, "default", "default_plus_custom", "custom_only"])
@pytest.mark.parametrize("skip_baseline", [True, False])
def test_failed_conditions_keep_requested_grading_policy(
    tmp_path: Path, grading_mode: str | None, skip_baseline: bool
) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_failed_job(jobs_dir / "demo-codex-with")
    if not skip_baseline:
        _write_failed_job(jobs_dir / "demo-codex-without")

    result = collect_harbor_results(
        "demo",
        ["codex"],
        tmp_path / "results",
        jobs_dir,
        skip_baseline=skip_baseline,
        expected_cases=1,
        expected_case_ids=["case-001"],
        grading_mode=grading_mode,
    )

    metric_set, metrics, policy = _expected_contract(grading_mode)
    assert result["execution_status"] == "failed"
    assert result["scored_attempts"] == 0
    assert result["metric_set"] == metric_set
    assert result["metrics"] == list(metrics)
    assert result["score_policy"] == result["attempt_policy"]["score_policy"] == policy
    assert result["attempt_policy"]["score_definition"] == score_definition(metrics)
    persisted_policy = json.loads((tmp_path / "results" / "attempt_policy.json").read_text(encoding="utf-8"))
    assert persisted_policy["score_policy"] == policy
    agent = result["agents"]["codex"]
    conditions = ["with_skill"] if skip_baseline else ["with_skill", "without_skill"]
    for condition in conditions:
        assert agent[f"metrics_{condition}"] == list(metrics)
        assert agent[f"score_policy_{condition}"] == policy
        assert agent[condition] == {}
        summary_path = tmp_path / "results" / "codex" / condition.replace("_", "-") / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["metric_set"] == metric_set
        assert summary["metrics"] == list(metrics)
        assert summary["score_policy"] == policy
        assert summary["execution_status"] == "failed"
        assert not summary["scores"] and not summary["custom_scores"] and not summary["dimensions"]
    assert not list((tmp_path / "results").rglob("reward.json"))

    # The report header must describe the requested policy, not the default fallback.
    payload = build_agent_eval_payload(
        "demo",
        report_data.load_agent_data(tmp_path / "results"),
        attempt_policy=persisted_policy,
        use_llm_judge=False,
    )
    assert payload is not None
    assert payload["score_policy"] == payload["attempt_policy"]["score_policy"] == policy
    assert payload["overall_score"] is None


@pytest.mark.parametrize("grading_mode", ["default", "custom_only"])
def test_missing_jobs_keep_requested_grading_policy(tmp_path: Path, grading_mode: str) -> None:
    result = collect_harbor_results(
        "demo",
        ["codex"],
        tmp_path / "results",
        tmp_path / "jobs",
        expected_cases=1,
        expected_case_ids=["case-001"],
        launch_errors=["codex with-skill Harbor run failed: sandbox unavailable"],
        grading_mode=grading_mode,
    )

    metric_set, metrics, policy = _expected_contract(grading_mode)
    assert result["execution_status"] == "failed"
    assert result["metric_set"] == metric_set
    assert result["score_policy"] == result["attempt_policy"]["score_policy"] == policy
    assert result["attempt_policy"]["score_definition"] == score_definition(metrics)
    agent = result["agents"]["codex"]
    assert agent["score_policy_with_skill"] == agent["score_policy_without_skill"] == policy
    baseline_summary = json.loads(
        (tmp_path / "results" / "codex" / "without-skill" / "summary.json").read_text(encoding="utf-8")
    )
    assert baseline_summary["score_policy"] == policy


@pytest.mark.parametrize("grading_mode", ["default", "default_plus_custom", "custom_only"])
@pytest.mark.parametrize("default_reward", [False, True])
def test_observed_rewards_still_determine_metric_set(tmp_path: Path, grading_mode: str, default_reward: bool) -> None:
    reward: dict[str, object] = {"entry_id": "case-001", "overall": 0.75}
    if default_reward:
        reward.update({"metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, 0.75)})
    _write_scored_job(tmp_path / "jobs" / "demo-codex-with", reward)

    result = collect_harbor_results(
        "demo",
        ["codex"],
        tmp_path / "results",
        tmp_path / "jobs",
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-001"],
        grading_mode=grading_mode,
    )

    assert result["execution_status"] == "succeeded"
    assert result["scored_attempts"] == 1
    assert result["metric_set"] == (DEFAULT_METRIC_SET if default_reward else CUSTOM_ONLY_METRIC_SET)
    assert result["metrics"] == (list(DEFAULT_METRICS) if default_reward else [])
    assert result["score_policy"] == (DEFAULT_SCORE_POLICY if default_reward else CUSTOM_SCORE_POLICY)
