# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skill metrics an eval case gives nothing to score are N/A in the with-skill arm, not 1.0."""

from __future__ import annotations

import json
from typing import Any

import pytest

from skillevaluator.constants import DEFAULT_SCORE_POLICY
from skillevaluator.tier3.eval_core import checks
from skillevaluator.tier3.harbor import metrics
from skillevaluator.tier3.harbor.activation import summarize_skill_activation

UNLABELED = "N/A: no expected_skill or should_trigger defined for this eval case"
NO_EXPECTED_SKILL = "N/A: no expected_skill defined for this eval case"
NEGATIVE = "N/A: negative case, so there is no skill use to measure"

READ_SKILL = {"action": "Read", "action_input": {"file_path": "/skills/demo/SKILL.md"}, "observation": "ok"}
READ_OTHER = {"action": "Read", "action_input": {"file_path": "/workspace/README.md"}, "observation": "ok"}
READ_ALT = {"action": "Read", "action_input": {"file_path": "/skills/alt/SKILL.md"}, "observation": "ok"}


def _judge(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {"score": 0.8, "reason": "ok", "results": []}


def _verifier(tmp_path, monkeypatch, entry, *, tool_calls=None, trajectory=True):
    """Point the Harbor verifier at tmp_path with stubbed trajectory, tool calls, security and judges."""
    from skillevaluator.tier3.harbor.templates import eval as verifier

    entry_path = tmp_path / "entry.json"
    entry_path.write_text(json.dumps(entry), encoding="utf-8")
    loaded = (
        ({"steps": [{"source": "agent", "message": "done"}]}, {"source": "trajectory.json"})
        if trajectory
        else (None, {"source": None})
    )
    monkeypatch.setattr(verifier, "ENTRY_PATH", entry_path)
    monkeypatch.setattr(verifier, "load_trajectory_with_fallback", lambda *_args: loaded)
    monkeypatch.setattr(verifier, "extract_tool_calls_as_dicts", lambda _traj: list(tool_calls or []))
    monkeypatch.setattr(verifier, "get_skill_tool_calls", lambda _traj: [])
    monkeypatch.setattr(verifier, "check_security", lambda *_a, **_k: {"score": 1.0})
    monkeypatch.setattr(verifier, "REWARD_JSON", tmp_path / "reward.json")
    monkeypatch.setattr(verifier, "REWARD_TXT", tmp_path / "reward.txt")
    monkeypatch.setattr(verifier, "SKILL_EVALUATOR_REWARD_JSON", tmp_path / "skill_evaluator_reward.json")
    monkeypatch.setattr(verifier, "call_public_llm", lambda *_a, **_k: pytest.fail("Unexpected LLM call"))
    for name in ("judge_accuracy", "judge_goal_accuracy", "judge_behavior_check"):
        monkeypatch.setattr(verifier, name, _judge)
    return verifier


def _outputs(tmp_path):
    numeric = json.loads((tmp_path / "reward.json").read_text(encoding="utf-8"))
    rich = json.loads((tmp_path / "skill_evaluator_reward.json").read_text(encoding="utf-8"))
    return numeric, rich


def _case(**fields: Any) -> dict[str, Any]:
    return {
        "id": "case-1",
        "question": "q",
        "ground_truth": "42",
        "expected_behavior": ["Answers"],
        "evaluated_skill": "demo",
        "has_skill": True,
        **fields,
    }


# ── Verifier: judged trials ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("tool_calls", "skill_execution"),
    [([READ_OTHER], 1.0), ([READ_SKILL], 0.0)],
    ids=["skill-left-alone", "skill-wrongly-read"],
)
@pytest.mark.parametrize(
    "labels",
    # create-eval-dataset writes a negative case with expected_skill null and no should_trigger.
    [{"expected_skill": None}, {"expected_skill": ""}, {"expected_skill": "demo", "should_trigger": False}],
    ids=["generated", "empty-expected-skill", "explicit"],
)
def test_negative_case_leaves_efficiency_na_and_keeps_its_negative_check(
    tmp_path, monkeypatch, labels, tool_calls, skill_execution
):
    verifier = _verifier(tmp_path, monkeypatch, _case(**labels), tool_calls=tool_calls)

    verifier.main()

    numeric, rich = _outputs(tmp_path)
    assert "skill_efficiency" not in numeric
    assert rich["skill_efficiency"] is None
    assert rich["details"]["skill_efficiency"] == {"score": None, "status": "not_applicable", "reason": NEGATIVE}
    assert metrics.metric_is_not_applicable(rich, "skill_efficiency")
    # Discoverability still checks that the agent left the skill under test alone.
    assert numeric["skill_execution"] == skill_execution
    assert rich["details"]["skill_execution"]["should_trigger"] is False
    assert "negative_check" in rich["details"]["skill_execution"]
    # Security, Correctness, Discoverability and Effectiveness; Efficiency drops out.
    expected = round((1.0 + 0.8 + skill_execution + 0.8) / 4, 4)
    assert numeric["overall"] == expected
    assert (tmp_path / "reward.txt").read_text(encoding="utf-8") == str(expected)
    assert metrics.overall_score(rich) == expected


@pytest.mark.parametrize(
    ("labels", "reason"),
    [
        ({}, UNLABELED),
        ({"should_trigger": True}, NO_EXPECTED_SKILL),
        ({"should_trigger": True, "expected_skill": ""}, NO_EXPECTED_SKILL),
    ],
    ids=["legacy-unlabeled", "positive-without-expected-skill", "positive-with-empty-expected-skill"],
)
def test_case_without_expected_skill_leaves_both_skill_metrics_na(tmp_path, monkeypatch, labels, reason):
    verifier = _verifier(tmp_path, monkeypatch, _case(**labels), tool_calls=[READ_SKILL])

    verifier.main()

    numeric, rich = _outputs(tmp_path)
    for metric in ("skill_execution", "skill_efficiency"):
        assert metric not in numeric
        assert rich[metric] is None
        assert rich["details"][metric] == {"score": None, "status": "not_applicable", "reason": reason}
    # Security, Correctness and Effectiveness only.
    assert numeric["overall"] == round((1.0 + 0.8 + 0.8) / 3, 4)
    # An N/A Discoverability is not an activation opportunity.
    assert summarize_skill_activation([rich])["status"] == "not_applicable"


def test_positive_case_without_tool_calls_keeps_the_zero_efficiency_penalty(tmp_path, monkeypatch):
    verifier = _verifier(tmp_path, monkeypatch, _case(expected_skill="demo"), tool_calls=[])

    verifier.main()

    numeric, rich = _outputs(tmp_path)
    assert numeric["skill_execution"] == 0.0
    assert numeric["skill_efficiency"] == 0.0
    assert rich["details"]["skill_efficiency"] == {"message": "No tool calls in trajectory"}
    assert numeric["overall"] == round((1.0 + 0.8 + 0.0 + 0.8 + 0.0) / 5, 4)


def test_positive_case_scores_both_skill_metrics(tmp_path, monkeypatch):
    verifier = _verifier(tmp_path, monkeypatch, _case(expected_skill="demo"), tool_calls=[READ_SKILL])

    verifier.main()

    numeric, rich = _outputs(tmp_path)
    assert numeric["skill_execution"] == 1.0
    assert numeric["skill_efficiency"] == 1.0
    assert set(rich["details"]["skill_efficiency"]) == {"routing", "tool_efficiency"}
    assert not metrics.not_applicable_metrics([rich])


@pytest.mark.parametrize(
    ("labels", "tool_calls", "na_reasons"),
    [
        ({"expected_skill": None}, [READ_OTHER], {"skill_efficiency": NEGATIVE}),
        (
            {"should_trigger": True, "expected_skill": "", "acceptable_skills": ["alt"]},
            [READ_ALT],
            {"skill_execution": NO_EXPECTED_SKILL, "skill_efficiency": NO_EXPECTED_SKILL},
        ),
        ({"expected_skill": "demo", "acceptable_skills": ["alt"]}, [READ_ALT], {}),
    ],
    ids=["negative", "positive-without-expected-skill", "positive-with-alternate"],
)
def test_baseline_staging_a_case_candidate_follows_the_case_rules(
    tmp_path, monkeypatch, labels, tool_calls, na_reasons
):
    # A no-skill arm that stages a skill the case could activate is scored on the
    # skill metrics, but a metric the case gives nothing to score stays N/A there too.
    entry = _case(has_skill=False, workspace_skill_names=["demo", "alt"], skill_workspace_mode="group", **labels)
    verifier = _verifier(tmp_path, monkeypatch, entry, tool_calls=tool_calls)

    verifier.main()

    numeric, rich = _outputs(tmp_path)
    for metric in ("skill_execution", "skill_efficiency"):
        if metric in na_reasons:
            assert metric not in numeric
            assert rich["details"][metric] == {"score": None, "status": "not_applicable", "reason": na_reasons[metric]}
        else:
            # Scored, not the baseline's N/A: the staged skill matches a case candidate.
            assert rich["details"][metric].get("status") != "not_applicable"
            assert numeric[metric] > 0
    assert set(metrics.not_applicable_metrics([rich])) == set(na_reasons)
    assert numeric["overall"] == metrics.overall_score(rich)


# ── Verifier: trials without a trajectory ───────────────────────────────


@pytest.mark.parametrize(
    ("labels", "na_reasons"),
    [
        ({"expected_skill": None}, {"skill_efficiency": NEGATIVE}),
        ({}, {"skill_execution": UNLABELED, "skill_efficiency": UNLABELED}),
        ({"expected_skill": "demo"}, {}),
    ],
    ids=["negative", "legacy-unlabeled", "positive"],
)
def test_trial_without_trajectory_keeps_case_level_skill_na(tmp_path, monkeypatch, labels, na_reasons):
    verifier = _verifier(tmp_path, monkeypatch, _case(**labels), trajectory=False)

    verifier.main()

    numeric, rich = _outputs(tmp_path)
    for metric in ("skill_execution", "skill_efficiency"):
        if metric in na_reasons:
            # A crash cannot turn a metric the case gives nothing to score into 0.0.
            assert metric not in numeric
            assert rich["details"][metric] == {
                "score": None,
                "status": "not_applicable",
                "reason": na_reasons[metric],
            }
        else:
            assert numeric[metric] == 0.0
    assert numeric["overall"] == 0.0


# ── Library scorers ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("should_trigger", "expected_skill", "reason"),
    [(False, "demo", NEGATIVE), (None, "demo", UNLABELED), (None, None, UNLABELED), (True, "", NO_EXPECTED_SKILL)],
    ids=["negative", "unlabeled-with-skill-name", "unlabeled", "positive-without-expected-skill"],
)
def test_library_skill_efficiency_is_na_without_a_skill_use_to_measure(should_trigger, expected_skill, reason):
    result = checks.score_skill_efficiency([READ_SKILL], expected_skill, should_trigger=should_trigger)

    assert result == {"score": None, "details": {"score": None, "status": "not_applicable", "reason": reason}}


def test_library_skill_efficiency_keeps_scores_for_positive_cases():
    assert checks.score_skill_efficiency([], "demo")["score"] == 0.0
    assert checks.score_skill_efficiency([READ_SKILL], "demo")["score"] == 1.0


@pytest.mark.parametrize("should_trigger", [None, False, True])
@pytest.mark.parametrize("expected_skill", [None, "", "demo"])
def test_case_reasons_match_the_library_scorers(should_trigger, expected_skill):
    # The verifier decides N/A for trials without a trajectory with the same helpers.
    execution = checks.score_skill_execution(
        [READ_OTHER], expected_skill, should_trigger=should_trigger, evaluated_skill="demo"
    )
    efficiency = checks.score_skill_efficiency([READ_OTHER], expected_skill, should_trigger=should_trigger)

    for result, reason in (
        (execution, checks.skill_execution_not_applicable_reason(should_trigger, expected_skill)),
        (efficiency, checks.skill_efficiency_not_applicable_reason(should_trigger, expected_skill)),
    ):
        if reason is None:
            assert isinstance(result["score"], float)
        else:
            assert result["score"] is None
            assert result["details"]["reason"] == reason


# ── Collection: some trials N/A ─────────────────────────────────────────


def _reward(case: str, *, has_skill: bool = True, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(metrics.DEFAULT_METRICS, 0.8)
    row.update(
        metric_set=metrics.DEFAULT_METRIC_SET,
        score_policy=DEFAULT_SCORE_POLICY,
        entry_id=case,
        has_skill=has_skill,
        details={},
    )
    for metric, value in overrides.items():
        row[metric] = value
        if value is None:
            row["details"][metric] = {"score": None, "status": "not_applicable", "reason": "No skill use"}
    return row


def _write_arm(job, rows):
    trial_names = []
    for row in rows:
        trial = f"{row['entry_id']}__attempt1"
        trial_names.append(trial)
        verifier = job / trial / "verifier"
        verifier.mkdir(parents=True)
        numeric = {key: value for key, value in row.items() if isinstance(value, float)}
        numeric["overall"] = metrics.overall_score(row)
        (verifier / "reward.json").write_text(json.dumps(numeric), encoding="utf-8")
        (verifier / "skill_evaluator_reward.json").write_text(json.dumps(row), encoding="utf-8")
    stats = {
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
    }
    (job / "result.json").write_text(json.dumps({"n_total_trials": len(rows), "stats": stats}), encoding="utf-8")


def test_arm_with_some_na_efficiency_trials_stays_scored(tmp_path):
    from skillevaluator.tier3.harbor.collector import collect_harbor_results

    baseline_na = {"skill_execution": None, "skill_efficiency": None}
    arms = {
        # A positive case scores Efficiency; the negative case leaves it N/A. Its agent read the
        # skill under test, so its negative check scores 0.0. Counting the N/A Efficiency as 0.0
        # would fail the negative case (0.44) instead of passing it (0.55).
        "with": [
            _reward("pos", skill_efficiency=0.6),
            _reward(
                "neg",
                security=1.0,
                accuracy=0.6,
                skill_execution=0.0,
                goal_accuracy=1.0,
                behavior_check=0.2,
                skill_efficiency=None,
            ),
        ],
        "without": [
            _reward("pos", has_skill=False, accuracy=0.5, **baseline_na),
            _reward("neg", has_skill=False, accuracy=0.5, **baseline_na),
        ],
    }
    for arm, rows in arms.items():
        _write_arm(tmp_path / "jobs" / f"demo-opencode-{arm}", rows)

    result = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        expected_cases=2,
        expected_case_ids=["pos", "neg"],
        expected_trials=2,
    )

    agent = result["agents"]["opencode"]
    assert agent["execution_status"] == "succeeded"
    assert "skill_efficiency" not in agent["not_applicable_metrics"]["with_skill"]
    summary = json.loads((tmp_path / "results/opencode/with-skill/summary.json").read_text(encoding="utf-8"))
    assert summary["not_applicable_counts"] == {"skill_efficiency": 1}
    # The arm mean is the positive case's score alone.
    assert summary["scores"]["skill_efficiency"] == 0.6
    assert summary["dimensions"]["efficiency"]["score"] == 0.6
    # pass@k scores each attempt without its N/A metric: (1.0 + 0.6 + 0.0 + 0.6) / 4.
    neg = summary["pass_at_k"]["cases"]["neg"]
    assert neg["best_score"] == pytest.approx(0.55)
    assert neg["passed"] is True
    # Lift pairs Efficiency on no case: the baseline never has a skill to use.
    assert "skill_efficiency" not in agent["lift"]
    assert agent["lift"]["accuracy"]["delta"] == pytest.approx(0.2)
    assert agent["lift"]["basis"]["metrics"]["accuracy"]["cases"] == 2


# ── Collection: the N/A reason the verifier recorded survives ──────────


def _negative_step_row(name: str, reason: str = NEGATIVE) -> dict[str, Any]:
    row = _reward("neg", skill_efficiency=None)
    row["details"]["skill_efficiency"]["reason"] = reason
    return dict(row, _trial_root_name="trial", _trial_name=f"trial/{name}", _step_name=name)


def test_mark_not_applicable_keeps_the_reason_the_verifier_recorded():
    recorded = _reward("neg", skill_efficiency=None)
    recorded["details"]["skill_efficiency"]["reason"] = NEGATIVE
    metrics.mark_not_applicable(recorded, "skill_efficiency", "Some other reason")
    assert recorded["details"]["skill_efficiency"]["reason"] == NEGATIVE

    given: dict[str, Any] = {"skill_efficiency": 0.5}
    metrics.mark_not_applicable(given, "skill_efficiency", NEGATIVE)
    assert given["skill_efficiency"] is None
    assert given["details"]["skill_efficiency"] == {"score": None, "status": "not_applicable", "reason": NEGATIVE}

    # A scored detail's text is no N/A reason; without a given one the metric's own reason applies.
    scored: dict[str, Any] = {"details": {"skill_efficiency": {"score": 0.5, "reason": "Two redundant reads"}}}
    metrics.mark_not_applicable(scored, "skill_efficiency")
    assert scored["details"]["skill_efficiency"]["reason"] == metrics.NOT_APPLICABLE_REASONS["skill_efficiency"]

    long: dict[str, Any] = {}
    metrics.mark_not_applicable(long, "skill_efficiency", "x" * 5000)
    assert len(long["details"]["skill_efficiency"]["reason"]) == metrics.NOT_APPLICABLE_REASON_MAX_CHARS


def test_logical_multistep_reward_keeps_the_negative_case_reason():
    from skillevaluator.tier3.harbor.collector import _logical_attempt_rewards

    [logical] = _logical_attempt_rewards([_negative_step_row("first"), _negative_step_row("second")])

    assert metrics.metric_is_not_applicable(logical, "skill_efficiency")
    assert logical["details"]["skill_efficiency"]["reason"] == NEGATIVE

    # Steps that record different reasons keep each of them.
    [mixed] = _logical_attempt_rewards([_negative_step_row("first"), _negative_step_row("second", UNLABELED)])
    assert mixed["details"]["skill_efficiency"]["reason"] == f"{NEGATIVE}; {UNLABELED}"


def test_restored_multistep_na_keeps_the_reason_each_step_recorded(tmp_path):
    from skillevaluator.tier3.harbor.collector import _restore_constituent_not_applicable

    rich = _negative_step_row("step")
    numeric = {key: value for key, value in rich.items() if isinstance(value, float)}
    result: dict[str, Any] = {"verifier_result": {"rewards": dict(numeric)}, "step_results": []}
    for name in ("first", "second"):
        directory = tmp_path / "steps" / name / "verifier"
        directory.mkdir(parents=True)
        (directory / "skill_evaluator_reward.json").write_text(json.dumps(rich), encoding="utf-8")
        result["step_results"].append({"step_name": name, "verifier_result": {"rewards": dict(numeric)}})
    data = dict(numeric)

    restored = _restore_constituent_not_applicable(data, result, tmp_path)

    rewards = [
        data,
        restored["verifier_result"]["rewards"],
        *(step["verifier_result"]["rewards"] for step in restored["step_results"]),
    ]
    for reward in rewards:
        assert metrics.metric_is_not_applicable(reward, "skill_efficiency")
        assert reward["details"]["skill_efficiency"]["reason"] == NEGATIVE
