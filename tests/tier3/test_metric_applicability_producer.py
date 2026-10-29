# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Case applicability survives judging, numeric sidecars and collection."""

from __future__ import annotations

import json

import pytest

from skillevaluator.constants import DEFAULT_SCORE_POLICY
from skillevaluator.tier3.harbor import metrics


def reward(case="a", **overrides):
    result = dict.fromkeys(metrics.DEFAULT_METRICS, 1.0)
    result.update(metric_set=metrics.DEFAULT_METRIC_SET, score_policy=DEFAULT_SCORE_POLICY, entry_id=case, details={})
    for metric, value in overrides.items():
        result[metric] = value
        if value is None:
            result["details"][metric] = {"score": None, "status": "not_applicable", "reason": "No reference"}
    return result


@pytest.mark.parametrize("value", [None, "", "  ", [], ["", None]])
def test_missing_references_return_na_without_calling_a_model(monkeypatch, value):
    from skillevaluator.tier3.eval_core import llm_judge

    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: pytest.fail("Absent reference called a judge"))
    for function, args in (
        (llm_judge.judge_accuracy, ("q", value, "response")),
        (llm_judge.judge_goal_accuracy, ("q", value, "response")),
        (llm_judge.judge_behavior_check, ("response", value)),
    ):
        result = function(*args)
        assert result["score"] is None
        assert result["status"] == "not_applicable"


@pytest.mark.parametrize("value", [0, 0.0, False, [False], "0"])
def test_false_and_zero_are_real_reference_answers(value):
    from skillevaluator.tier3.eval_core.llm_judge import _has_judge_reference

    assert _has_judge_reference(value)


def test_only_explicit_markers_skip_eligible_metrics():
    payload = reward(accuracy=None, goal_accuracy=None, behavior_check=None)
    assert metrics.overall_score(payload) == 1.0
    assert metrics.not_applicable_metrics([payload]) == ["accuracy", "goal_accuracy", "behavior_check"]
    del payload["details"]["accuracy"]
    assert metrics.overall_score(payload) is None
    payload = reward(security=None)
    assert not metrics.metric_is_not_applicable(payload, "security")
    assert metrics.overall_score(payload) is None


def test_mixed_trial_averages_keep_na_counts_and_real_zeros():
    first = reward("a", accuracy=0.0, behavior_check=None)
    second = reward("b", accuracy=None, behavior_check=None)
    scores, _, _ = metrics.average_metrics([first, second])
    assert scores["accuracy"] == 0.0 and "behavior_check" not in scores
    assert metrics.not_applicable_counts([first, second]) == {"accuracy": 1, "behavior_check": 2}
    assert metrics.dimension_scores(scores, ["behavior_check"])["effectiveness"]["score"] == 1.0


def test_pairing_covers_same_cases_and_excludes_judge_mismatch_per_metric():
    from skillevaluator.tier3.harbor.lift_basis import pair_metric_scores

    paired = pair_metric_scores(
        [("a", reward(accuracy=0.9)), ("b", reward("b", accuracy=None))],
        [("a", reward(accuracy=0.4)), ("b", reward("b", accuracy=0.0))],
        arms=("with_skill", "without_skill"),
        metrics=("security", "accuracy"),
        expected_case_ids=("a", "b", "missing"),
        judge_mixed_cases={"accuracy": {"a"}},
    )
    assert paired.treatment == {"security": 1.0}
    assert paired.control == {"security": 1.0}
    assert paired.judge_mixed == {"accuracy": 1}
    assert paired.basis["metrics"]["security"]["cases"] == 2
    assert any(row["case"] == "missing" and row["reason"] == "not_scored" for row in paired.basis["excluded_cases"])


def test_collector_lift_pairs_metrics_before_subtracting_arm_averages():
    from skillevaluator.tier3.harbor.collector import _compute_lift

    treatment = [reward("a", accuracy=0.9), reward("b", accuracy=None)]
    control = [reward("a", accuracy=0.6), reward("b", accuracy=0.0)]
    lift = _compute_lift(
        metrics.average_metrics(treatment)[0], metrics.average_metrics(control)[0], treatment, control, ["a", "b"]
    )
    assert lift["accuracy"]["without_skill"] == 0.6
    assert lift["accuracy"]["delta"] == 0.3
    assert lift["basis"]["metrics"]["accuracy"]["cases"] == 1


def test_collector_merges_numeric_reward_with_explicit_na_sidecar(tmp_path):
    from skillevaluator.tier3.harbor.collector import _merge_reward_sidecars, _partition_scoreable_rewards

    payload = reward(accuracy=None, goal_accuracy=None, behavior_check=None)
    (tmp_path / "skill_evaluator_reward.json").write_text(json.dumps(payload))
    numeric = {key: value for key, value in payload.items() if isinstance(value, float)}
    _merge_reward_sidecars(numeric, tmp_path)
    scored, failed = _partition_scoreable_rewards([numeric])
    assert len(scored) == 1 and failed == []


def test_logical_multistep_reward_preserves_all_na_metrics():
    from skillevaluator.tier3.harbor.collector import _logical_attempt_rewards

    rows = [
        dict(
            reward(accuracy=None, behavior_check=None),
            _trial_root_name="trial",
            _trial_name=f"trial/{name}",
            _step_name=name,
        )
        for name in ("first", "second")
    ]
    [logical] = _logical_attempt_rewards(rows)
    assert metrics.metric_is_not_applicable(logical, "accuracy")
    assert metrics.metric_is_not_applicable(logical, "behavior_check")
    assert metrics.overall_score(logical) == 1.0


def test_shared_metric_pass_comparison_does_not_credit_activation_alone():
    from skillevaluator.tier3.harbor.collector import _shared_metric_pass_lift

    treatment = [reward(accuracy=0.0, goal_accuracy=0.0, behavior_check=0.0)]
    control = [reward(accuracy=0.0, goal_accuracy=0.0, behavior_check=0.0, skill_execution=None, skill_efficiency=None)]
    comparison = _shared_metric_pass_lift(
        treatment,
        control,
        n_attempts=1,
        pass_threshold=0.5,
        stop_on_pass=False,
        expected_cases=1,
        expected_case_ids=["a"],
    )
    assert comparison["basis"] == "shared_metrics"
    assert comparison["delta"] == 0.0
    assert comparison["excluded_metrics"] == ["skill_execution", "skill_efficiency"]


def test_template_required_judge_accepts_only_verifier_authorized_na():
    from skillevaluator.tier3.harbor.templates import eval as verifier

    na = {"score": None, "status": "not_applicable", "reason": "No reference"}
    assert (
        verifier._normalize_required_judge_result("accuracy", na, allow_not_applicable=True)["status"]
        == "not_applicable"
    )
    assert verifier._normalize_required_judge_result("accuracy", na)["status"] == "error"
    assert not verifier.skill_metrics_applicable(False, True, "demo")
    assert verifier.skill_metrics_applicable(False, True, "demo", workspace_skill_names=["demo"])
    payload = reward(
        accuracy=None, goal_accuracy=None, behavior_check=None, skill_execution=None, skill_efficiency=None
    )
    assert verifier._reward_overall(payload, payload["details"]) == 1.0


def test_template_no_reference_main_writes_numeric_reward_and_rich_na(tmp_path, monkeypatch):
    from skillevaluator.tier3.harbor.templates import eval as verifier

    entry = tmp_path / "entry.json"
    entry.write_text(json.dumps({"id": "a", "question": "q", "has_skill": False}))
    monkeypatch.setattr(verifier, "ENTRY_PATH", entry)
    monkeypatch.setattr(
        verifier,
        "load_trajectory_with_fallback",
        lambda *_args: ({"steps": [{"source": "agent", "message": "done"}]}, {"source": "trajectory.json"}),
    )
    monkeypatch.setattr(verifier, "check_security", lambda *_a: {"score": 1.0})
    monkeypatch.setattr(verifier, "REWARD_JSON", tmp_path / "reward.json")
    monkeypatch.setattr(verifier, "REWARD_TXT", tmp_path / "reward.txt")
    monkeypatch.setattr(verifier, "SKILL_EVALUATOR_REWARD_JSON", tmp_path / "skill_evaluator_reward.json")
    monkeypatch.setattr(verifier, "call_public_llm", lambda *_a, **_k: pytest.fail("No-reference case called an LLM"))
    verifier.main()
    numeric = json.loads((tmp_path / "reward.json").read_text())
    rich = json.loads((tmp_path / "skill_evaluator_reward.json").read_text())
    assert numeric == {"security": 1.0, "overall": 1.0}
    assert all(rich[metric] is None for metric in metrics.NOT_APPLICABLE_ELIGIBLE_METRICS)


def test_authoritative_multistep_na_requires_every_constituent_sidecar(tmp_path):
    from skillevaluator.tier3.harbor.collector import _merge_constituent_default_reward_failure

    rich = reward(accuracy=None, behavior_check=None)
    numeric = {key: value for key, value in rich.items() if isinstance(value, float)}
    result = {"verifier_result": {"rewards": dict(numeric)}, "step_results": []}
    for name in ("first", "second"):
        directory = tmp_path / "steps" / name / "verifier"
        directory.mkdir(parents=True)
        (directory / "skill_evaluator_reward.json").write_text(json.dumps(rich))
        result["step_results"].append({"step_name": name, "verifier_result": {"rewards": dict(numeric)}})
    data = dict(numeric)
    _merge_constituent_default_reward_failure(data, result, tmp_path)
    assert data.get("evaluation_status") != "failed"
    assert metrics.metric_is_not_applicable(data, "accuracy")
    (tmp_path / "steps/second/verifier/skill_evaluator_reward.json").unlink()
    data = dict(numeric)
    _merge_constituent_default_reward_failure(data, result, tmp_path)
    assert data["evaluation_status"] == "failed"


def test_collected_summaries_and_lift_share_na_case_basis(tmp_path):
    from skillevaluator.tier3.harbor.collector import collect_harbor_results

    arms = {
        "with": [reward("a", accuracy=0.9, behavior_check=None), reward("b", accuracy=None, behavior_check=None)],
        "without": [
            reward("a", accuracy=0.6, behavior_check=None, skill_execution=None, skill_efficiency=None),
            reward("b", accuracy=0.0, behavior_check=None, skill_execution=None, skill_efficiency=None),
        ],
    }
    for arm, rows in arms.items():
        job = tmp_path / "jobs" / f"demo-opencode-{arm}"
        trial_names = []
        for row in rows:
            trial = f"{row['entry_id']}__attempt1"
            trial_names.append(trial)
            verifier = job / trial / "verifier"
            verifier.mkdir(parents=True)
            numeric = {key: value for key, value in row.items() if isinstance(value, float)}
            numeric["overall"] = metrics.overall_score(row)
            (verifier / "reward.json").write_text(json.dumps(numeric))
            (verifier / "skill_evaluator_reward.json").write_text(json.dumps(row))
        (job / "result.json").write_text(
            json.dumps(
                {
                    "n_total_trials": len(rows),
                    "stats": {
                        "n_completed_trials": len(rows),
                        "n_errored_trials": 0,
                        "n_running_trials": 0,
                        "n_pending_trials": 0,
                        "n_cancelled_trials": 0,
                        "n_retries": 0,
                        "evals": {
                            "agent__model___harbor-tasks": {
                                "n_trials": len(rows),
                                "n_errors": 0,
                                "reward_stats": {"reward": {"0.1": trial_names}},
                            }
                        },
                    },
                }
            )
        )
    result = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        expected_cases=2,
        expected_case_ids=["a", "b"],
        expected_trials=2,
    )
    agent = result["agents"]["opencode"]
    assert agent["execution_status"] == "succeeded"
    assert agent["not_applicable_metrics"]["without_skill"] == ["skill_execution", "skill_efficiency", "behavior_check"]
    summary = json.loads((tmp_path / "results/opencode/with-skill/summary.json").read_text())
    assert summary["not_applicable_counts"] == {"accuracy": 1, "behavior_check": 2}
    assert summary["dimensions"]["effectiveness"]["score"] == 1.0
    assert agent["lift"]["accuracy"]["without_skill"] == 0.6
    assert agent["lift"]["accuracy"]["delta"] == 0.3
    assert agent["lift"]["basis"]["metrics"]["accuracy"]["cases"] == 1
    assert result["metrics"] == list(metrics.DEFAULT_METRICS)
