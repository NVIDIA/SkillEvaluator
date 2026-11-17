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


def _collected(tmp_path: Path, trajectory: dict | None, *, variant: str = "with") -> dict:
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
    if variant == "without":
        for metric in ("skill_execution", "skill_efficiency"):
            reward[metric] = None
            reward["details"][metric] = {
                "score": None,
                "status": "not_applicable",
                "reason": "N/A: this arm has no skill under test installed, so there is no skill to discover, run or route to",
            }
    _save_trials([reward], tmp_path / "saved", tmp_path / "job", skill_name="demo", agent="codex", variant=variant)
    return json.loads(next((tmp_path / "saved").glob("*/reward.json")).read_text())


def _payload(
    rewards: list[dict],
    expected_skill: object = "demo",
    *,
    baseline: list[dict] | None = None,
    entry: dict | None = None,
) -> dict:
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 1.0),
                "without_skill": {"security": 1.0, "accuracy": 1.0, "goal_accuracy": 1.0, "behavior_check": 1.0}
                if baseline
                else {},
                "rewards": rewards,
                "rewards_baseline": baseline or [],
                "model": "fixture-model",
            }
        },
        dataset=[entry if entry is not None else {"id": "case", "expected_skill": expected_skill}],
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


@pytest.mark.parametrize("expected_skill", ["DEMO", " Demo "])
@pytest.mark.parametrize(("skill", "status"), [("demo", "matched"), (None, "mismatched")])
def test_primary_activation_identity_is_case_insensitive(
    tmp_path: Path, expected_skill: str, skill: str | None, status: str
) -> None:
    trial = _payload([_collected(tmp_path, _trajectory(skill))], expected_skill)["trials"][0]
    validation = trial["activation_validation"]
    assert validation["expected_activation"] is True
    assert validation["expectation_status"] == status
    assert trial["overall"] == 1.0


@pytest.mark.parametrize("alias", ["acceptable_skills", "acceptable_alternates"])
@pytest.mark.parametrize("skill", ["demo", None])
def test_allowed_alternate_activation_is_optional(tmp_path: Path, alias: str, skill: str | None) -> None:
    entry = {"id": "case", "expected_skill": "other", alias: [" DEMO "]}
    trial = _payload([_collected(tmp_path, _trajectory(skill))], entry=entry)["trials"][0]
    validation = trial["activation_validation"]
    assert validation["expected_activation"] is None
    assert validation["expectation_status"] == "unknown"
    assert validation["observed_activation"] == ("invoked" if skill else "not_invoked")
    assert trial["overall"] == 1.0


@pytest.mark.parametrize("alias", ["acceptable_skills", "acceptable_alternates"])
@pytest.mark.parametrize(("skill", "status"), [("demo", "mismatched"), (None, "matched")])
def test_explicit_negative_overrides_allowed_alternate(
    tmp_path: Path, alias: str, skill: str | None, status: str
) -> None:
    entry = {"id": "case", "expected_skill": "other", alias: ["DEMO"], "should_trigger": False}
    validation = _payload([_collected(tmp_path, _trajectory(skill))], entry=entry)["trials"][0][
        "activation_validation"
    ]
    assert validation["expected_activation"] is False
    assert validation["expectation_status"] == status


@pytest.mark.parametrize(
    ("skill", "trace_present", "observed", "status"),
    [
        (None, True, "not_invoked", "matched"),
        ("demo", True, "invoked", "mismatched"),
        (None, False, "unknown", "unknown"),
    ],
    ids=["correct-baseline-non-use", "baseline-invocation", "missing-baseline-trace"],
)
def test_standard_baseline_retains_na_scores_and_independent_activation_evidence(
    tmp_path: Path,
    skill: str | None,
    trace_present: bool,
    observed: str,
    status: str,
) -> None:
    with_skill = _collected(tmp_path / "with", _trajectory("demo"))
    baseline = _collected(tmp_path / "without", _trajectory(skill) if trace_present else None, variant="without")
    payload = _payload([with_skill], baseline=[baseline])
    trial = payload["agents"]["codex"]["trials_baseline"][0]
    validation = trial["activation_validation"]
    assert validation["expected_activation"] is False
    assert validation["observed_activation"] == observed
    assert validation["expectation_status"] == status
    assert validation["task_outcome"] == {"accuracy": 1.0, "goal_accuracy": 1.0}
    for metric in ("skill_execution", "skill_efficiency"):
        assert baseline[metric] is None
        assert baseline["details"][metric]["score"] is None
        assert baseline["details"][metric]["status"] == "not_applicable"
        assert metric not in trial["scores"]
        assert metric in trial["not_applicable_metrics"]
    if trace_present:
        assert baseline["skill_invoked"] is (skill == "demo")
        assert baseline["invocation_evidence_scope"] == "logical_trial"
    else:
        assert "skill_invoked" not in baseline
        assert "invocation_evidence_scope" not in baseline


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


@pytest.mark.parametrize(
    "entry",
    [
        {"id": "case", "expected_skill": "demo", "should_trigger": False},
        {"id": "case", "should_trigger": False},
    ],
    ids=["named-negative", "nameless-negative"],
)
@pytest.mark.parametrize(("skill", "status"), [(None, "matched"), ("demo", "mismatched")])
def test_explicit_negative_precedence_matches_runner_contract(
    tmp_path: Path,
    entry: dict,
    skill: str | None,
    status: str,
) -> None:
    reward = _collected(tmp_path, _trajectory(skill))
    validation = _payload([reward], entry=entry)["trials"][0]["activation_validation"]
    assert validation["expected_activation"] is False
    assert validation["expectation_status"] == status


@pytest.mark.parametrize(
    "entry",
    [
        {"id": "case"},
        {"id": "case", "should_trigger": True},
        {"id": "case", "expected_skill": None, "should_trigger": True},
    ],
    ids=["unlabeled", "positive-without-target", "positive-with-null-target"],
)
def test_activation_expectation_needs_a_target_for_positive_routing(tmp_path: Path, entry: dict) -> None:
    reward = _collected(tmp_path, _trajectory("demo"))
    validation = _payload([reward], entry=entry)["trials"][0]["activation_validation"]
    assert validation["expected_activation"] is None
    assert validation["expectation_status"] == "unknown"


def test_custom_only_na_skill_metrics_do_not_create_supported_activation(tmp_path: Path) -> None:
    reward = _collected(tmp_path, _trajectory(), variant="without")
    reward["metric_set"] = "custom-only"
    reward["skill_invoked"] = False
    reward["invocation_evidence_source"] = "trajectory"
    reward["invocation_evidence_scope"] = "logical_trial"
    [trial] = _normalize_trials([reward], list(DEFAULT_METRICS))
    assert trial["activation_validation"]["observed_activation"] == "unknown"


def test_null_skill_score_without_explicit_na_provenance_stays_unsupported() -> None:
    [trial] = _normalize_trials(
        [
            {
                "metric_set": DEFAULT_METRIC_SET,
                **dict.fromkeys(DEFAULT_METRICS, 1.0),
                "skill_execution": None,
                "skill_invoked": False,
                "invocation_evidence_source": "trajectory",
                "invocation_evidence_scope": "logical_trial",
            }
        ],
        list(DEFAULT_METRICS),
    )
    assert trial["activation_validation"]["observed_activation"] == "unknown"
