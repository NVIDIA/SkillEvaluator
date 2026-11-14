# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Never-activated trials must stay inconclusive across saved report surfaces."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from skillevaluator.evaluation.tier3_report import (
    _load_suggestions_v2,
    _validation_result_from_payload,
    agent_eval_result_from_directory,
    build_agent_eval_payload,
)
from skillevaluator.reporting import HTMLReporter, MarkdownReporter
from skillevaluator.reporting.base import assess_tier3_evidence
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.console_ui import summarize_tier3
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.harbor.report_data import load_agent_data
from skillevaluator.tier3.result_display import render_result


def _agent(*, activated: bool = False, score: float = 0.4) -> dict:
    reward = {
        **dict.fromkeys(DEFAULT_METRICS, score),
        "entry_id": "case",
        "trial_id": "case__attempt1",
        "details": {"skill_execution": {"activation": {"passed": activated, "score": float(activated)}}},
    }
    return {
        "execution_status": "succeeded",
        "expected_attempts": 2,
        "scored_attempts": 2,
        "num_trials": 1,
        "num_trials_baseline": 1,
        "with_skill": dict.fromkeys(DEFAULT_METRICS, score),
        "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
        "rewards": [reward],
        "rewards_baseline": [{**reward, "trial_id": "baseline__attempt1"}],
        "lift": {"overall": {"delta": -0.4}},
        "pass_lift": {"delta": -1.0},
        "custom_with_skill": {"custom_grade": 0.2},
        "custom_without_skill": {"custom_grade": 0.9},
        "custom_lift": {"custom_grade": {"delta": -0.7}},
    }


@pytest.mark.parametrize("score", [0.2, 0.9])
def test_no_activation_suppresses_all_causal_lift_without_erasing_scores(score: float) -> None:
    payload = build_agent_eval_payload("demo", {"opencode": _agent(score=score)}, use_llm_judge=False)
    assert payload is not None
    assert payload["execution_status"] == "succeeded"
    assert payload["verdict"] == "inconclusive"
    assert payload["overall_score"] == score
    assert payload["overall_lift"] is None
    assert payload["composite_lift"] is None
    assert payload["summary"]["skill_activation"]["status"] == "never_activated"
    assert all(item["lift"] is None for item in payload["evaluators"].values())
    assert all(item["lift"] is None and item["verdict"] == "inconclusive" for item in payload["dimensions"])
    assert all(item["lift"] is None for item in payload["evaluator_cards"])
    assert payload["pass_at_k"]["lift"] == {}
    assert all("lift_scores" not in trial for trial in payload["trials"])
    assert all(item["severity"] != "fail" for item in payload["conclusions"])


def test_inconclusive_renders_consistently_in_html_markdown_cli_and_evidence_assessment() -> None:
    payload = build_agent_eval_payload("demo", {"opencode": _agent()}, use_llm_judge=False)
    result = _validation_result_from_payload(payload)
    assert result is not None
    assessment = assess_tier3_evidence([result])
    assert assessment.status == "inconclusive"
    assert not assessment.evidence_complete
    assert assessment.execution_status == "succeeded"
    assert any("INCONCLUSIVE" in text for row in summarize_tier3(result)[2] for text, _ in row.segments)
    html = HTMLReporter(include_timestamp=False).render_all([result])
    markdown = MarkdownReporter(include_timestamp=False).render_all([result])
    assert "Inconclusive: skill never activated." in html
    assert "**Verdict:** INCONCLUSIVE" in markdown
    assert "Skill never activated." in markdown
    output = io.StringIO()
    CLIReporter._print_agent_eval_tables(payload, Console(file=output, width=140, color_system=None))
    assert "Verdict: INCONCLUSIVE" in output.getvalue()


def test_mixed_agents_select_activated_evidence_even_when_unactivated_score_is_higher() -> None:
    payload = build_agent_eval_payload(
        "demo", {"unactivated": _agent(score=1.0), "activated": _agent(activated=True, score=0.8)}, use_llm_judge=False
    )
    assert payload is not None
    assert payload["best_agent"] == "activated"
    assert payload["verdict"] == "pass"
    assert payload["agents"]["unactivated"]["verdict"] == "inconclusive"
    assert payload["agents"]["unactivated"]["lift"] is None
    assert payload["overall_lift"] == 0.0


def test_mixed_agents_discard_unattributed_stale_skill_advice(tmp_path) -> None:
    agents = {"unactivated": _agent(score=1.0), "activated": _agent(activated=True, score=0.8)}
    stale_advice = [{"suggestion": "Remove the skill because it regressed quality."}]
    agent_dir = tmp_path / "unactivated"
    agent_dir.mkdir()
    (agent_dir / "findings.json").write_text(json.dumps({"suggestions_v2": stale_advice}))
    suggestions = _load_suggestions_v2(tmp_path, agents)
    assert suggestions == []
    payload = build_agent_eval_payload("demo", agents, suggestions_v2=suggestions, use_llm_judge=False)
    assert payload is not None
    assert payload["suggestions_v2"] == []
    assert payload["verdict"] == "pass"


@pytest.mark.parametrize("with_best_advice", [True, False])
def test_mixed_agents_load_advice_only_from_selected_best_agent(tmp_path, with_best_advice: bool) -> None:
    agents = {
        "a-unactivated": _agent(score=1.0),
        "b-activated": _agent(activated=True, score=0.7),
        "z-best": _agent(activated=True, score=0.8),
    }
    for name, info in agents.items():
        for condition, score_key, reward_key in (
            ("with-skill", "with_skill", "rewards"),
            ("without-skill", "without_skill", "rewards_baseline"),
        ):
            condition_dir = tmp_path / name / condition
            trial_dir = condition_dir / "trials" / "case__attempt1"
            trial_dir.mkdir(parents=True)
            (condition_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "scores": info[score_key],
                        "execution_status": "succeeded",
                        "num_trials": 1,
                        "expected_attempts": 1,
                        "scored_attempts": 1,
                    }
                )
            )
            (trial_dir / "reward.json").write_text(json.dumps(info[reward_key][0]))
        if name != "z-best" or with_best_advice:
            (tmp_path / name / "findings.json").write_text(
                json.dumps({"suggestions_v2": [{"suggestion": f"Advice from {name}", "dimension": "accuracy"}]})
            )

    result = agent_eval_result_from_directory(tmp_path / "demo", tmp_path, use_llm_judge=False)
    assert result is not None
    payload = result.metadata["agent_eval"]
    assert payload["best_agent"] == "z-best"
    recommendations = [item["recommendation"] for item in payload["suggestions_v2"]]
    assert recommendations == (["Advice from z-best"] if with_best_advice else [])
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert ("Advice from z-best" in html) is with_best_advice
    assert "Advice from a-unactivated" not in html
    assert "Advice from b-activated" not in html


@pytest.mark.parametrize("activated", [True, False])
def test_llm_insights_use_eligible_trials_without_hiding_diagnostics(monkeypatch, activated: bool) -> None:
    calls = []
    prompts = []

    def build_insights(payload, **kwargs):
        from skillevaluator.evaluation.insights_judge import InsightsJudge

        calls.append(payload)
        prompt = InsightsJudge().create_user_prompt(canonical=payload, deterministic=kwargs["deterministic"])
        prompts.append(json.loads(prompt.split("\n\n", 1)[1]))
        return {"conclusions": [], "recommendations": [{"message": "Advice from activated evidence."}]}

    monkeypatch.setattr("skillevaluator.evaluation.insights_judge.build_insights", build_insights)
    payload = build_agent_eval_payload(
        "demo",
        {"unactivated": _agent(score=1.0), "activated": _agent(activated=activated, score=0.8)},
        use_llm_judge=True,
    )
    assert payload is not None
    assert {trial["agent"] for trial in payload["trials"]} == {"activated", "unactivated"}
    if activated:
        assert len(calls) == 1
        assert calls[0]["agents_run"] == ["activated"]
        assert set(calls[0]["agents"]) == {"activated"}
        assert {trial["agent"] for trial in calls[0]["trials"]} == {"activated"}
        assert prompts[0]["skill"]["agents_run"] == ["activated"]
        assert {trial["agent"] for trial in prompts[0]["trials"]} == {"activated"}
        assert "Advice from activated evidence." in payload["suggestions"]
    else:
        assert calls == []
        assert "Advice from activated evidence." not in payload["suggestions"]


@pytest.mark.parametrize("best_has_trace", [True, False])
def test_mixed_agent_advice_links_only_eligible_traces_and_keeps_full_diagnostics(monkeypatch, best_has_trace) -> None:
    agents = {
        "never": _agent(score=0.1),
        "activated": _agent(activated=True, score=0.4),
        "best": _agent(activated=True, score=0.8),
    }
    for name, agent in agents.items():
        if name == "best" and not best_has_trace:
            continue
        agent["rewards"][0]["harbor_viewer"] = {
            "job_url": "https://harbor.example.test/jobs/run",
            "trial_url": f"https://harbor.example.test/jobs/run/trials/{name}",
        }

    def build_insights(_payload, **_kwargs):
        return {
            "conclusions": [{"title": "Eligible case evidence", "source": "llm", "evidence_case_ids": ["case"]}],
            "recommendations": [
                {"message": "Improve case using activated evidence.", "source": "llm", "evidence_case_ids": ["case"]}
            ],
        }

    monkeypatch.setattr("skillevaluator.evaluation.insights_judge.build_insights", build_insights)
    payload = build_agent_eval_payload(
        "demo",
        agents,
        suggestions_v2=[{"recommendation": "Advice from best", "evidence_case_ids": ["case"]}],
        use_llm_judge=True,
    )
    assert payload is not None
    assert payload["best_agent"] == "best"
    linked_agents = set(agents) if best_has_trace else {"never", "activated"}
    assert {link["agent"] for link in payload["harbor_viewer"]["evidence_links"]} == linked_agents
    for field in ("conclusions", "recommendations"):
        llm_advice = [item for item in payload[field] if item.get("source") == "llm"]
        assert len(llm_advice) == 1
        assert llm_advice[0]["evidence"]["agent"] in {"activated", "best"}
        assert "/trials/never" not in llm_advice[0]["evidence"]["url"]
        assert all(item["evidence"]["agent"] != "never" for item in payload[field] if item.get("evidence"))
    deterministic_advice = [item for item in payload["recommendations"] if item.get("source") == "deterministic"]
    assert deterministic_advice
    if best_has_trace:
        assert payload["suggestions_v2"][0]["harbor_evidence"]["agent"] == "best"
        assert payload["suggestions_v2"][0]["harbor_evidence"]["url"].endswith("/trials/best")
        assert all(item["evidence"]["agent"] == "best" for item in deterministic_advice)
    else:
        assert "harbor_evidence" not in payload["suggestions_v2"][0]
        assert all("evidence" not in item for item in deterministic_advice)


def test_ungrounded_llm_recommendations_keep_positional_trace_links(monkeypatch) -> None:
    agent = _agent(activated=True, score=0.6)
    template = agent["rewards"][0]
    agent["rewards"] = [
        {
            **template,
            "entry_id": f"case-{index}",
            "trial_id": f"case-{index}__attempt1",
            "overall": 0.1 * (index + 1),
            "harbor_viewer": {"trial_url": f"https://harbor.example.test/jobs/run/trials/{index}"},
        }
        for index in range(4)
    ]
    agent["num_trials"] = 4

    def build_insights(_payload, **_kwargs):
        return {
            "conclusions": [],
            "recommendations": [
                {"message": "First ungrounded advice.", "source": "llm"},
                {"message": "Second ungrounded advice.", "source": "llm"},
            ],
        }

    monkeypatch.setattr("skillevaluator.evaluation.insights_judge.build_insights", build_insights)
    payload = build_agent_eval_payload("demo", {"claude-code": agent}, use_llm_judge=True)
    assert payload is not None
    urls = [item["evidence"]["url"] for item in payload["recommendations"]]
    # Each recommendation keeps its own positional trace, as before per-source link selection.
    assert len(urls) == 3
    assert len(set(urls)) == 3


def test_missing_or_incomplete_activation_evidence_does_not_assert_never_activated() -> None:
    for change in ({"rewards": []}, {"rewards_complete": False}):
        payload = build_agent_eval_payload("demo", {"opencode": {**_agent(), **change}}, use_llm_judge=False)
        assert payload is not None
        assert payload["verdict"] != "inconclusive"


def test_negative_only_activation_cases_are_not_inconclusive() -> None:
    agent = _agent()
    agent["rewards"][0]["details"]["skill_execution"] = {"should_trigger": False}
    payload = build_agent_eval_payload("demo", {"opencode": agent}, use_llm_judge=False)
    assert payload is not None
    assert payload["skill_activation"]["status"] == "not_applicable"
    assert payload["verdict"] != "inconclusive"


def test_no_baseline_does_not_turn_never_activated_scores_into_quality_verdict() -> None:
    agent = _agent(score=1.0)
    agent.pop("without_skill")
    agent.pop("rewards_baseline")
    agent["conditions"] = {"without_skill": {"execution_status": "skipped"}}
    payload = build_agent_eval_payload("demo", {"opencode": agent}, use_llm_judge=False)
    assert payload is not None
    assert payload["verdict"] == "inconclusive"
    assert payload["overall_score"] == 1.0
    assert payload["overall_lift"] is None


def test_saved_summary_preserves_census_when_trial_loading_is_incomplete(tmp_path) -> None:
    condition = tmp_path / "opencode" / "with-skill"
    condition.mkdir(parents=True)
    summary = {
        "scores": dict.fromkeys(DEFAULT_METRICS, 0.4),
        "execution_status": "succeeded",
        "num_trials": 12,
        "expected_attempts": 12,
        "scored_attempts": 12,
        "skill_activation": {
            "status": "never_activated",
            "positive_trials": 12,
            "activated_trials": 0,
            "unknown_trials": 0,
            "reason": "Skill never activated.",
        },
    }
    (condition / "summary.json").write_text(json.dumps(summary))
    agents = load_agent_data(tmp_path)
    assert agents["opencode"]["skill_activation"]["positive_trials"] == 12
    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)
    assert payload is not None and payload["verdict"] == "inconclusive"


def test_historical_saved_trials_rebuild_activation_census_for_all_report_consumers(tmp_path) -> None:
    condition = tmp_path / "opencode" / "with-skill"
    trial = condition / "trials" / "case__attempt1"
    trial.mkdir(parents=True)
    agent = _agent()
    (condition / "summary.json").write_text(
        json.dumps({"scores": agent["with_skill"], "execution_status": "succeeded", "num_trials": 1})
    )
    (trial / "reward.json").write_text(json.dumps(agent["rewards"][0]))
    agents = load_agent_data(tmp_path)
    assert agents["opencode"]["rewards_complete"]
    assert agents["opencode"]["skill_activation"]["status"] == "never_activated"


def test_terminal_does_not_restore_dimension_lift_from_raw_scores() -> None:
    agent = _agent()
    agent.update(
        skill_activation={"status": "never_activated"},
        dimensions_with_skill={"effectiveness": {"score": 0.4}},
        dimensions_without_skill={"effectiveness": {"score": 0.8}},
    )
    output = render_result({"execution_status": "succeeded", "agents": {"opencode": agent}})
    assert "INCONCLUSIVE: skill never activated" in output
    assert "OVERALL SKILL LIFT" not in output
    assert "-0.40" not in output


@pytest.mark.parametrize("activation", [None, [], "unknown", {"status": None}, {"status": []}, {"status": "unknown"}])
def test_html_tolerates_malformed_persisted_activation_summary(activation) -> None:
    payload = build_agent_eval_payload("demo", {"opencode": _agent(activated=True)}, use_llm_judge=False)
    assert payload is not None
    payload["agents"]["opencode"]["skill_activation"] = activation
    result = _validation_result_from_payload(payload)
    assert result is not None
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert "Agent Scorecards" in html
