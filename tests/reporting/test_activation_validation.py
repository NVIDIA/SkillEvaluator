# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeatable activation acceptance cases through the trusted collector and report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import _normalize_trials, build_agent_eval_payload
from skillevaluator.tier3.harbor.collector import _save_trials
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS


def _trajectory(skill: str | None = None) -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "root",
        "trajectory_id": "root",
        "agent": {"name": "codex", "version": "fixture"},
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "message": "done",
                "tool_calls": (
                    [
                        {
                            "tool_call_id": "read",
                            "function_name": "Read",
                            "arguments": {"file_path": f"/workspace/skills/{skill}/SKILL.md"},
                        }
                    ]
                    if skill
                    else []
                ),
            }
        ],
    }


def _collected(tmp_path: Path, trajectory: dict | None) -> dict:
    trial = tmp_path / "job" / "case__attempt1"
    (trial / "agent").mkdir(parents=True)
    if trajectory is not None:
        (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory))
    reward = {
        "entry_id": "case",
        "_trial_name": "case__attempt1",
        "_trial_root_name": "case__attempt1",
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "details": {
            "skill_execution": {
                "workflow_order": {"score": 0.0, "passed": False},
                "script_execution": {"score": 1.0, "passed": True},
            }
        },
        # A verifier's invented invocation must be replaced, including when no trace exists.
        "skill_invoked": True,
        "invocation_evidence_source": "verifier",
    }
    _save_trials([reward], tmp_path / "saved", tmp_path / "job", skill_name="demo", agent="codex", variant="with")
    return json.loads(next((tmp_path / "saved").glob("*/reward.json")).read_text())


def _payload(rewards: list[dict], expected_skill: object = "demo", *, baseline: list[dict] | None = None) -> dict:
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 1.0),
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 1.0) if baseline else {},
                "rewards": rewards,
                "rewards_baseline": baseline or [],
                "model": "fixture-model",
            }
        },
        dataset=[{"id": "case", "expected_skill": expected_skill}],
        env_mode="docker",
        run_id="fixture-run",
        use_llm_judge=False,
    )
    assert payload is not None
    return payload


@pytest.mark.parametrize(
    ("skill", "expected_skill", "observed", "status"),
    [
        ("demo", "demo", "invoked", "matched"),
        (None, "demo", "not_invoked", "mismatched"),
        ("unrelated", "demo", "not_invoked", "mismatched"),
        (None, None, "not_invoked", "matched"),
        ("demo", None, "invoked", "mismatched"),
    ],
    ids=["explicit-use", "missed-implicit-use", "wrong-skill", "negative-non-use", "irrelevant-use"],
)
def test_activation_expectation_is_independent_of_perfect_task_score(
    tmp_path: Path,
    skill: str | None,
    expected_skill: str | None,
    observed: str,
    status: str,
) -> None:
    reward = _collected(tmp_path, _trajectory(skill))
    trial = _payload([reward], expected_skill)["trials"][0]
    validation = trial["activation_validation"]
    assert validation["expected_activation"] is (expected_skill == "demo")
    assert validation["observed_activation"] == observed
    assert validation["expectation_status"] == status
    assert validation["task_outcome"] == {"accuracy": 1.0, "goal_accuracy": 1.0}
    assert trial["overall"] == 1.0
    assert validation["evidence_scopes"][0]["workflow"]["workflow_order"] == {"passed": False, "score": 0.0}


def test_correct_baseline_is_non_use_without_becoming_skill_success(tmp_path: Path) -> None:
    baseline = _collected(tmp_path, _trajectory())
    payload = _payload([baseline], baseline=[baseline])
    validation = payload["agents"]["codex"]["trials_baseline"][0]["activation_validation"]
    assert validation["expected_activation"] is False
    assert validation["observed_activation"] == "not_invoked"
    assert validation["expectation_status"] == "matched"
    assert validation["task_outcome"]["accuracy"] == 1.0


def test_missing_trace_stays_unknown_even_with_perfect_scores(tmp_path: Path) -> None:
    trial = _payload([_collected(tmp_path, None)])["trials"][0]
    assert trial["activation_validation"]["observed_activation"] == "unknown"
    assert trial["activation_validation"]["expectation_status"] == "unknown"


@pytest.mark.parametrize("referenced", [True, False])
def test_only_referenced_subagent_proves_activation(tmp_path: Path, referenced: bool) -> None:
    root = _trajectory()
    child = _trajectory("demo")
    child["trajectory_id"] = child["session_id"] = "child"
    root["subagent_trajectories"] = [child]
    if referenced:
        root["steps"][0]["observation"] = {
            "results": [
                {
                    "content": "done",
                    "subagent_trajectory_ref": [{"trajectory_id": "child"}],
                }
            ]
        }
    validation = _payload([_collected(tmp_path, root)])["trials"][0]["activation_validation"]
    assert validation["observed_activation"] == ("invoked" if referenced else "not_invoked")


def test_copied_context_does_not_prove_activation(tmp_path: Path) -> None:
    root = _trajectory("demo")
    root["steps"][0]["is_copied_context"] = True
    root["steps"].append({"step_id": 2, "source": "agent", "message": "done", "tool_calls": []})
    validation = _payload([_collected(tmp_path, root)])["trials"][0]["activation_validation"]
    assert validation["observed_activation"] == "not_invoked"


def test_multistep_projection_retains_scoped_evidence_without_claiming_complete_trace() -> None:
    rows = [
        {
            "trial_id": "logical",
            "entry_id": "case",
            "metric_set": DEFAULT_METRIC_SET,
            **dict.fromkeys(DEFAULT_METRICS, 1.0),
            "skill_invoked": invoked,
            "invocation_evidence_source": "trajectory",
            "_activation_artifacts": {"reward": f"codex/with-skill/trials/step-{index}/reward.json"},
        }
        for index, invoked in enumerate((True, False))
    ]
    [trial] = _normalize_trials(rows, list(DEFAULT_METRICS))
    validation = trial["activation_validation"]
    assert validation["observed_activation"] == "unknown"
    assert [scope["observed_activation"] for scope in validation["evidence_scopes"]] == ["invoked", "not_invoked"]
    assert validation["evidence_scopes"][1]["artifacts"]["reward"] == "codex/with-skill/trials/step-1/reward.json"
    assert "skill_invoked" not in trial


def test_custom_only_and_untrusted_flags_stay_unknown() -> None:
    for contract, source in (("custom-only", "trajectory"), (DEFAULT_METRIC_SET, "verifier")):
        [trial] = _normalize_trials(
            [
                {
                    "metric_set": contract,
                    "skill_execution": 1.0,
                    "skill_invoked": True,
                    "invocation_evidence_source": source,
                }
            ],
            ["skill_execution"],
        )
        assert trial["activation_validation"]["observed_activation"] == "unknown"


def test_dangling_subagent_reference_cannot_prove_non_use(tmp_path: Path) -> None:
    root = _trajectory()
    root["steps"][0]["observation"] = {
        "results": [
            {
                "content": "missing child",
                "subagent_trajectory_ref": [{"trajectory_id": "missing"}],
            }
        ]
    }
    validation = _payload([_collected(tmp_path, root)])["trials"][0]["activation_validation"]
    assert validation["observed_activation"] == "unknown"


def test_saved_run_links_each_observation_to_retained_artifacts(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.report_data import load_agent_data

    reward = _collected(tmp_path, _trajectory("demo"))
    condition = tmp_path / "run" / "codex" / "with-skill"
    trial = condition / "trials" / "case"
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text(json.dumps(reward))
    (trial / "trajectory.json").write_text(json.dumps(_trajectory("demo")))
    (condition / "summary.json").write_text(
        json.dumps(
            {
                "scores": dict.fromkeys(DEFAULT_METRICS, 1.0),
                "metrics": list(DEFAULT_METRICS),
                "execution_status": "succeeded",
                "num_trials": 1,
            }
        )
    )
    agents = load_agent_data(tmp_path / "run")
    payload = build_agent_eval_payload(
        "demo", agents, dataset=[{"id": "case", "expected_skill": "demo"}], use_llm_judge=False
    )
    assert payload is not None
    scope = payload["trials"][0]["activation_validation"]["evidence_scopes"][0]
    assert scope["artifacts"] == {
        "reward": "codex/with-skill/trials/case/reward.json",
        "trajectory": "codex/with-skill/trials/case/trajectory.json",
    }
    assert scope["observed_activation"] == "invoked"


def test_missing_expectation_is_unknown_rather_than_a_negative_case() -> None:
    reward = {
        "entry_id": "case",
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "skill_invoked": False,
        "invocation_evidence_source": "trajectory",
    }
    payload = build_agent_eval_payload(
        "demo",
        {"codex": {"with_skill": dict.fromkeys(DEFAULT_METRICS, 1.0), "rewards": [reward]}},
        dataset=[{"id": "case"}],
        use_llm_judge=False,
    )
    assert payload is not None
    validation = payload["trials"][0]["activation_validation"]
    assert validation["expected_activation"] is None
    assert validation["expectation_status"] == "unknown"


def test_resource_diagnostics_do_not_imply_activation() -> None:
    [trial] = _normalize_trials(
        [
            {
                "accuracy": 1.0,
                "_traj": {"steps": 2, "prompt_tokens": 10, "completion_tokens": 4},
                "_agent_seconds": 3.0,
            }
        ],
        ["accuracy"],
    )
    validation = trial["activation_validation"]
    assert validation["observed_activation"] == "unknown"
    assert validation["resources"] == {"steps": 2, "tokens": {"prompt": 10, "completion": 4}, "agent_seconds": 3.0}


def test_omitted_multistep_rows_do_not_turn_one_retained_step_into_logical_non_use() -> None:
    [trial] = _normalize_trials(
        [
            {
                "trial_id": "logical",
                "entry_id": "case",
                "metric_set": DEFAULT_METRIC_SET,
                **dict.fromkeys(DEFAULT_METRICS, 1.0),
                "skill_invoked": False,
                "invocation_evidence_source": "trajectory",
                "invocation_evidence_scope": "step",
            }
        ],
        list(DEFAULT_METRICS),
    )
    validation = trial["activation_validation"]
    assert validation["observed_activation"] == "unknown"
    assert validation["observation_scope"] == "reward_rows_only"
    assert validation["evidence_scopes"][0]["observed_activation"] == "not_invoked"


def test_collector_marks_logical_invocation_scope(tmp_path: Path) -> None:
    reward = _collected(tmp_path, _trajectory("demo"))
    assert reward["invocation_evidence_scope"] == "logical_trial"


@pytest.mark.parametrize("complete", [True, False])
def test_native_multistep_root_requires_every_authoritative_trajectory(tmp_path: Path, complete: bool) -> None:
    job = tmp_path / "job"
    trial = job / "case__attempt1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "config": {"agent": {"resume_trajectory": False}},
                "step_results": [{"step_name": "prepare"}, {"step_name": "finish"}],
            }
        )
    )
    for step_name in ("prepare", "finish") if complete else ("prepare",):
        agent = trial / "steps" / step_name / "agent"
        agent.mkdir(parents=True)
        trajectory = _trajectory("demo" if step_name == "finish" else None)
        trajectory["session_id"] = trajectory["trajectory_id"] = step_name
        (agent / "trajectory.json").write_text(json.dumps(trajectory))
    reward = {
        "entry_id": "case",
        "_trial_name": "case__attempt1",
        "_trial_root_name": "case__attempt1",
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
    }
    _save_trials([reward], tmp_path / "saved", job, skill_name="demo", agent="codex", variant="with")
    saved = json.loads(next((tmp_path / "saved").glob("*/reward.json")).read_text())
    validation = _payload([saved])["trials"][0]["activation_validation"]
    assert validation["observed_activation"] == ("invoked" if complete else "unknown")
    if complete:
        assert saved["invocation_evidence_scope"] == "logical_trial"


def test_collector_retains_step_scope_for_fallback_reward_rows(tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = job / "case__attempt1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"step_results": [{"step_name": "one"}]}))
    agent = trial / "steps" / "one" / "agent"
    agent.mkdir(parents=True)
    (agent / "trajectory.json").write_text(json.dumps(_trajectory()))
    reward = {
        "entry_id": "case",
        "_trial_name": "case__attempt1__one",
        "_trial_root_name": "case__attempt1",
        "_step_name": "one",
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
    }
    _save_trials([reward], tmp_path / "saved", job, skill_name="demo", agent="codex", variant="with")
    saved = json.loads(next((tmp_path / "saved").glob("*/reward.json")).read_text())
    assert saved["invocation_evidence_scope"] == "step"
    validation = _payload([saved])["trials"][0]["activation_validation"]
    assert validation["observed_activation"] == "unknown"
    assert validation["evidence_scopes"][0]["observed_activation"] == "not_invoked"


def test_historical_unspecified_scope_cannot_certify_a_complete_logical_trial(tmp_path: Path) -> None:
    reward = _collected(tmp_path, _trajectory("demo"))
    reward.pop("invocation_evidence_scope")
    validation = _payload([reward])["trials"][0]["activation_validation"]
    assert validation["observed_activation"] == "unknown"
    assert validation["evidence_scopes"][0]["observed_activation"] == "invoked"
