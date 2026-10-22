# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activation evidence must precede a causal skill-lift verdict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.activation import summarize_skill_activation
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.metrics import CUSTOM_ONLY_METRIC_SET, CUSTOM_SCORE_POLICY, DEFAULT_METRICS
from skillevaluator.tier3.harbor.report import _findings_eligible


def _reward(score: float = 0.0, *, passed: bool | None = False, **activation: object) -> dict:
    return {
        **dict.fromkeys(DEFAULT_METRICS, 0.8),
        "details": {"skill_execution": {"activation": {"score": score, "passed": passed, **activation}}},
    }


def test_zero_activation_is_separate_from_execution_score() -> None:
    summary = summarize_skill_activation([_reward(), _reward()])
    assert summary["status"] == "never_activated"
    assert summary["positive_trials"] == 2
    assert summary["activated_trials"] == 0
    assert summary["unknown_trials"] == 0


def test_unactivated_skill_does_not_generate_quality_advice() -> None:
    assert not _findings_eligible(
        {"execution_status": "succeeded", "skill_activation": summarize_skill_activation([_reward()])}
    )


def test_custom_grader_details_do_not_imply_standard_activation() -> None:
    reward = {
        "metric_set": CUSTOM_ONLY_METRIC_SET,
        "score_policy": CUSTOM_SCORE_POLICY,
        "custom_metrics": {"quality": 0.8},
        "details": _reward()["details"],
    }
    assert summarize_skill_activation([reward])["status"] == "unknown"


def test_malformed_message_is_not_an_activation_skip() -> None:
    reward = _reward()
    reward["details"]["skill_execution"] = {"message": {"malformed": True}}
    assert summarize_skill_activation([reward])["status"] == "unknown"


@pytest.mark.parametrize("score", [0.5, 0.75, 1.0])
def test_any_positive_activation_preserves_lift(score: float) -> None:
    summary = summarize_skill_activation([_reward(), _reward(score, passed=True)])
    assert summary["status"] == "activated"
    assert summary["activated_trials"] == 1


@pytest.mark.parametrize(
    "unknown",
    [{}, _reward(0.5, passed=None, supported=False), _reward(float("nan")), _reward(True), _reward(-1.0)],
)
def test_unknown_or_malformed_evidence_does_not_assert_never_activated(unknown: dict) -> None:
    summary = summarize_skill_activation([_reward(), unknown])
    assert summary["status"] == "unknown"
    assert summary["unknown_trials"] == 1


@pytest.mark.parametrize(
    "detail",
    [{"should_trigger": False}, {"message": "No expected_skill -- skipped"}],
)
def test_negative_and_unlabelled_cases_do_not_count_as_activation_opportunities(detail: dict) -> None:
    negative = {"details": {"skill_execution": detail}}
    assert summarize_skill_activation([negative])["status"] == "not_applicable"
    summary = summarize_skill_activation([_reward(), negative])
    assert summary["status"] == "never_activated"
    assert summary["positive_trials"] == 1


def test_multistep_trial_counts_once_and_later_activation_is_preserved() -> None:
    rows = [
        {**_reward(), "_trial_root_name": "trial-a", "_step_name": "one"},
        {**_reward(1.0, passed=True), "_trial_root_name": "trial-a", "_step_name": "two"},
        {**_reward(), "_trial_root_name": "trial-b"},
    ]
    summary = summarize_skill_activation(rows)
    assert summary["status"] == "activated"
    assert summary["positive_trials"] == 2
    assert summary["activated_trials"] == 1


def _write_job(jobs: Path, variant: str, rewards: list[dict]) -> None:
    job = jobs / f"demo-opencode-{variant}"
    for index, reward in enumerate(rewards):
        verifier = job / f"case-{index}__trial" / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.json").write_text(json.dumps({"entry_id": f"case-{index}", **reward}))
    (job / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(rewards),
                "stats": {
                    "n_trials": len(rewards),
                    "n_errors": 0,
                    "evals": {
                        "opencode__model": {
                            "n_trials": len(rewards),
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.8": [f"case-{i}__trial" for i in range(len(rewards))]}},
                        }
                    },
                },
            }
        )
    )


@pytest.mark.parametrize("skip_baseline", [False, True])
def test_collector_preserves_scores_but_suppresses_unactivated_lift(tmp_path: Path, skip_baseline: bool) -> None:
    jobs, output = tmp_path / "jobs", tmp_path / "output"
    _write_job(jobs, "with", [_reward()])
    if not skip_baseline:
        _write_job(jobs, "without", [_reward()])
    result = collect_harbor_results(
        "demo", ["opencode"], output, jobs, skip_baseline=skip_baseline, expected_case_ids=["case-0"]
    )
    agent = result["agents"]["opencode"]
    assert agent["execution_status"] == "succeeded"
    assert agent["with_skill"]["accuracy"] == 0.8
    assert agent["skill_activation"]["status"] == "never_activated"
    assert agent["lift"] == {}
    assert agent["custom_lift"] == {}
    assert agent["pass_at_k"]["lift"] == {}
    summary = json.loads((output / "opencode/with-skill/summary.json").read_text())
    assert summary["skill_activation"] == agent["skill_activation"]
    assert not (output / "opencode/lift.json").exists()


def test_collector_keeps_lift_when_one_trial_activates(tmp_path: Path) -> None:
    jobs, output = tmp_path / "jobs", tmp_path / "output"
    _write_job(jobs, "with", [_reward(), _reward(0.75, passed=True)])
    _write_job(jobs, "without", [_reward(), _reward()])
    result = collect_harbor_results("demo", ["opencode"], output, jobs, expected_case_ids=["case-0", "case-1"])
    assert result["agents"]["opencode"]["skill_activation"]["activated_trials"] == 1
    assert result["agents"]["opencode"]["lift"]["overall"]["delta"] == 0.0
