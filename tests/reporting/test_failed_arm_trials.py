# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed arms retain trial diagnostics without restoring aggregate scores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import (
    _validation_result_from_payload,
    agent_eval_result_from_directory,
)
from skillevaluator.reporting.benchmark import BenchmarkReporter
from skillevaluator.tier3.harbor import report_data
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.token_usage import build_token_usage


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _failed_run(
    tmp_path: Path,
    arm: str,
    *,
    malformed: bool = False,
    empty: bool = False,
    failed_status: str | None = None,
    scored_trials: tuple[str, ...] = ("case__scored",),
    legacy_ids: bool = False,
) -> dict:
    """Write a run whose ``arm`` failed; ``legacy_ids`` leaves ``trial_id`` out of the failed arm's rewards.

    ``scored_trials`` are the failed arm's scored attempts. The other arm keeps one.
    """
    skill = tmp_path / "demo"
    skill.mkdir()
    run = tmp_path / "results"
    for variant in ("with-skill", "without-skill"):
        failed = variant == arm
        _write(
            run / "codex" / variant / "summary.json",
            {
                "scores": dict.fromkeys(DEFAULT_METRICS, 1.0),
                "metrics": list(DEFAULT_METRICS),
                "num_trials": 1,
                "execution_status": "failed" if failed else "succeeded",
                "execution_errors": ["AgentTimeoutError: timed out"] if failed else [],
                "expected_attempts": 2 if failed else 1,
                "scored_attempts": 1,
            },
        )
        if failed and empty:
            continue
        trials_dir = run / "codex" / variant / "trials"
        for name in scored_trials if failed else ("case__scored",):
            trial = trials_dir / name
            reward = {
                **dict.fromkeys(DEFAULT_METRICS, 1.0),
                "metric_set": DEFAULT_METRIC_SET,
                "entry_id": "case",
                "trial_id": name,
                "skill_invoked": variant == "with-skill",
                "invocation_evidence_source": "trajectory",
                "invocation_evidence_scope": "logical_trial",
            }
            if failed and legacy_ids:
                # Older artifacts carry no collector-issued trial identity.
                del reward["trial_id"]
            _write(trial / "reward.json", reward)
            _write(
                trial / "trajectory.json", {"final_metrics": {"total_prompt_tokens": 10, "total_completion_tokens": 2}}
            )
        if failed:
            error = trials_dir / "case__errored"
            _write(error / "failure.json", {"entry_id": "case", "reason": "AgentTimeoutError: timed out"})
            _write(
                error / "trajectory.json", {"final_metrics": {"total_prompt_tokens": 20, "total_completion_tokens": 3}}
            )
            if malformed:
                (error / "reward.json").write_text("{")
            if failed_status:
                _write(
                    error / "reward.json",
                    {
                        **dict.fromkeys(DEFAULT_METRICS, 1.0),
                        "metric_set": DEFAULT_METRIC_SET,
                        "entry_id": "case",
                        "trial_id": "case__errored",
                        "evaluation_status": failed_status,
                        "skill_invoked": False,
                        "invocation_evidence_source": "trajectory",
                        "invocation_evidence_scope": "logical_trial",
                    },
                )
    result = agent_eval_result_from_directory(skill, run, use_llm_judge=False)
    assert result is not None
    return result.metadata["agent_eval"]


@pytest.mark.parametrize("arm,key", [("with-skill", "trials"), ("without-skill", "trials_baseline")])
def test_failed_arm_keeps_scored_trial_activation(tmp_path: Path, arm: str, key: str) -> None:
    payload = _failed_run(tmp_path, arm)
    agent = payload["agents"]["codex"]
    scored = next((row for row in agent[key] if row["trial_id"] == "case__scored"), None)
    assert scored is not None
    assert scored["scores"]["goal_accuracy"] == 1.0
    assert scored["activation_validation"]["observed_activation"] == (
        "invoked" if arm == "with-skill" else "not_invoked"
    )
    assert agent["with_skill" if arm == "with-skill" else "baseline"] is None
    assert agent["lift"] is None
    usage = payload["token_usage"]["agents"]["codex"]["with_skill" if arm == "with-skill" else "baseline"]
    assert usage["coverage"]["observed_attempts"] == 2
    assert usage["coverage"]["unscored_attempts"] == 1
    assert usage["totals"]["total_tokens"] == 35


@pytest.mark.parametrize("arm,key", [("with-skill", "trials"), ("without-skill", "trials_baseline")])
def test_failed_arm_rewards_without_trial_id_are_counted_once(tmp_path: Path, arm: str, key: str) -> None:
    """Older rewards carry no ``trial_id``: the diagnostic trial and the usage-only copy are one attempt."""
    (tmp_path / "with-ids").mkdir()
    (tmp_path / "legacy").mkdir()
    with_ids = _failed_run(tmp_path / "with-ids", arm)
    legacy = _failed_run(tmp_path / "legacy", arm, legacy_ids=True)
    scored = [row for row in legacy["agents"]["codex"][key] if row.get("scored") is not False]
    assert len(scored) == 1
    assert scored[0]["scores"]["goal_accuracy"] == 1.0
    usage = legacy["token_usage"]["agents"]["codex"]["with_skill" if arm == "with-skill" else "baseline"]
    assert usage["coverage"]["observed_attempts"] == 2
    assert usage["coverage"]["measured_attempts"] == 2
    assert usage["coverage"]["unscored_attempts"] == 1
    totals = usage["totals"]
    assert (totals["prompt_tokens"], totals["completion_tokens"], totals["total_tokens"]) == (30, 5, 35)
    # The same run with trial IDs reports the same usage.
    assert legacy["token_usage"] == with_ids["token_usage"]


@pytest.mark.parametrize("legacy_ids", [True, False])
def test_failed_arm_distinct_attempts_with_identical_usage_each_count(tmp_path: Path, legacy_ids: bool) -> None:
    """Retries share a case ID, and these two also share their token counts. Each is still one attempt.

    Attempts are told apart by the trial directory they ran in, never by case ID or usage.
    """
    payload = _failed_run(tmp_path, "with-skill", legacy_ids=legacy_ids, scored_trials=("case__first", "case__second"))
    agent = payload["agents"]["codex"]
    scored = [row for row in agent["trials"] if row.get("scored") is not False]
    assert [row["entry_id"] for row in scored] == ["case", "case"]
    usage = payload["token_usage"]["agents"]["codex"]["with_skill"]
    assert usage["coverage"]["observed_attempts"] == 3
    assert usage["coverage"]["unscored_attempts"] == 1
    # Two scored attempts of 12 tokens each, and the errored attempt's 23.
    assert usage["totals"]["total_tokens"] == 47


@pytest.mark.parametrize("identity", ["absent", "", 7, "x" * 513])
def test_withheld_rewards_without_a_valid_trial_id_share_their_directory_identity(
    tmp_path: Path, identity: object
) -> None:
    _failed_run(tmp_path, "with-skill", legacy_ids=True, scored_trials=("case__a", "case__b"))
    reward_path = tmp_path / "results" / "codex" / "with-skill" / "trials" / "case__a" / "reward.json"
    reward = json.loads(reward_path.read_text())
    if identity != "absent":
        reward["trial_id"] = identity
    _write(reward_path, reward)
    # A reward that does carry a collector-issued identity keeps it.
    other_path = reward_path.parents[1] / "case__b" / "reward.json"
    _write(other_path, {**json.loads(other_path.read_text()), "trial_id": "collector-issued"})

    agent = report_data.load_agent_data(tmp_path / "results")["codex"]

    diagnostic = [row.get("trial_id") for row in agent["diagnostic_rewards"] if row.get("_unscored") is not True]
    usage = [row["trial_id"] for row in agent["usage_only_attempts"] if row["scored"] is True]
    assert diagnostic == usage == ["case__a", "collector-issued"]
    assert agent["rewards"] == []


def test_succeeded_arm_rewards_without_trial_id_load_unchanged(tmp_path: Path) -> None:
    """Only a withheld arm needs a stand-in identity; a succeeded arm's legacy rewards stay identity-less."""
    _failed_run(tmp_path, "")
    trials = tmp_path / "results" / "codex" / "with-skill" / "trials"
    for name in ("case__first", "case__second"):
        reward = {**dict.fromkeys(DEFAULT_METRICS, 1.0), "metric_set": DEFAULT_METRIC_SET, "entry_id": "case"}
        _write(trials / name / "reward.json", reward)
    agent = report_data.load_agent_data(tmp_path / "results")["codex"]
    assert [row.get("trial_id") for row in agent["rewards"]] == [None, "case__scored", None]
    assert len(report_data.logical_trial_reward_groups(agent["rewards"])) == 3


@pytest.mark.parametrize("artifact", ["missing-reward", "malformed-reward", "failed", "error"])
def test_errored_trial_has_unknown_activation(tmp_path: Path, artifact: str) -> None:
    agent = _failed_run(
        tmp_path,
        "with-skill",
        malformed=artifact == "malformed-reward",
        failed_status=artifact if artifact in {"failed", "error"} else None,
    )["agents"]["codex"]
    error = next((row for row in agent["trials"] if row["trial_id"] == "case__errored"), None)
    assert error is not None
    assert error["activation_validation"]["observed_activation"] == "unknown"
    assert error["activation_validation"]["expectation_status"] == "unknown"
    assert error["activation_validation"]["diagnostics"][0]["code"] == "attempt_unscored"
    assert error["scored"] is False
    assert error["scores"] == {}
    assert error["overall"] is None


def test_empty_failed_arm_does_not_invent_trials(tmp_path: Path) -> None:
    assert _failed_run(tmp_path, "with-skill", empty=True)["agents"]["codex"]["trials"] == []


def test_diagnostic_trials_do_not_double_count_usage() -> None:
    agent = {
        "conditions": {"with_skill": {"execution_status": "failed", "expected_attempts": 2, "scored_attempts": 1}},
        "trials": [
            {"entry_id": "case", "trial_id": "case__scored", "tokens": {"prompt": 10, "completion": 2}},
            {"entry_id": "case", "trial_id": "case__errored", "scored": False},
        ],
        "usage_only_attempts": [
            {
                "entry_id": "case",
                "trial_id": "case__scored",
                "scored": True,
                "tokens": {"prompt_tokens": 10, "completion_tokens": 2},
            },
            {
                "entry_id": "case",
                "trial_id": "case__errored",
                "scored": False,
                "tokens": {"prompt_tokens": 20, "completion_tokens": 3},
            },
        ],
    }
    usage = build_token_usage({"codex": agent}, mode="with_skill_only", dataset_case_ids=["case"])
    arm = usage["agents"]["codex"]["with_skill"]
    assert arm["coverage"]["observed_attempts"] == 2
    assert arm["coverage"]["unscored_attempts"] == 1
    assert arm["totals"]["total_tokens"] == 35


@pytest.mark.parametrize("arm,key", [("with-skill", "trials"), ("without-skill", "trials_baseline")])
def test_failed_arm_score_bearing_outputs_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str, key: str
) -> None:
    payload = _failed_run(tmp_path, arm)
    load = report_data.load_agent_data

    def without_diagnostics(*args, **kwargs):
        agents = load(*args, **kwargs)
        for info in agents.values():
            info.pop("diagnostic_rewards", None)
            info.pop("diagnostic_rewards_baseline", None)
        return agents

    monkeypatch.setattr(report_data, "load_agent_data", without_diagnostics)
    reference = agent_eval_result_from_directory(tmp_path / "demo", tmp_path / "results", use_llm_judge=False)
    assert reference is not None
    expected = reference.metadata["agent_eval"]
    agent, before = payload["agents"]["codex"], expected["agents"]["codex"]
    assert agent["with_skill" if arm == "with-skill" else "baseline"] is None
    assert agent["lift"] is None
    for field in ("with_skill", "baseline", "lift", "dimensions", "evaluators", "cases", "resources"):
        assert agent[field] == before[field]
    for field in (
        "cases",
        "verdict",
        "recommendations",
        "suggestions",
        "suggestions_v2",
        "conclusions",
        "case_diagnostics",
        "dataset_summary",
        "dataset_sufficiency",
        "overall_score",
        "overall_lift",
    ):
        assert payload[field] == expected[field]
    assert payload["dataset_summary"]["total_tasks"] == (0 if arm == "with-skill" else 1)
    other_key = "trials_baseline" if key == "trials" else "trials"
    assert all(row.get("diagnostic_only") is True for row in agent[key])
    assert all("diagnostic_only" not in row for row in agent[other_key])
    reporter = BenchmarkReporter(include_timestamp=False, report_scope="tier3")
    result = _validation_result_from_payload(payload)
    assert result is not None
    benchmark = reporter.render_all([result])
    assert benchmark == reporter.render_all([reference])
    overall = next(line for line in benchmark.splitlines() if line.startswith("| Overall |"))
    assert "uplift unavailable" in overall
    assert ("Not scored with the skill" if arm == "with-skill" else "baseline incomplete") in overall
    assert len(payload["agents"]["codex"][key]) == 2


@pytest.mark.parametrize("status", ["failed", "error"])
def test_succeeded_arm_aggregates_are_unchanged(tmp_path: Path, status: str) -> None:
    expected = _failed_run(tmp_path, "")
    reward_path = tmp_path / "results" / "codex" / "with-skill" / "trials" / "case__scored" / "reward.json"
    reward = json.loads(reward_path.read_text())
    reward["evaluation_status"] = status
    _write(reward_path, reward)
    result = agent_eval_result_from_directory(tmp_path / "demo", tmp_path / "results", use_llm_judge=False)
    assert result is not None
    agent = result.metadata["agent_eval"]["agents"]["codex"]
    assert agent["conditions"]["with_skill"]["execution_status"] == "succeeded"
    assert len(agent["trials"]) == 1
    for field in ("with_skill", "baseline", "lift", "dimensions", "evaluators", "cases", "resources"):
        assert agent[field] == expected["agents"]["codex"][field]
