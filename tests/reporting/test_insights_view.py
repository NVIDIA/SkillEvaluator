# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concise execution insights preserve canonical errors and scoped evidence."""

import copy

import pytest

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload


def _project(payload):
    from skillevaluator.reporting.insights_view import build_insights_view

    return build_insights_view(payload)


def _failed_payload():
    errors = ["AgentTimeoutError: runtime failed; diagnostic contains a semicolon", "Judge failed for case-b"]
    blob = "; ".join(errors)
    return {
        "execution_status": "failed",
        "execution_errors": errors,
        "expected_attempts": 6,
        "scored_attempts": 3,
        "agents": {
            "agent-a": {
                "execution_status": "failed",
                "conditions": {
                    "with_skill": {"execution_status": "failed", "scored_attempts": 2, "expected_attempts": 3},
                    "without_skill": {"execution_status": "failed", "scored_attempts": 1, "expected_attempts": 3},
                },
            },
        },
        "conclusions": [{"title": "Evaluation incomplete", "severity": "fail", "message": blob}],
        "recommendations": [{"title": "AgentTimeoutError", "message": blob, "source": "deterministic"}],
        "next_steps": [
            {
                "kind": "incomplete run",
                "target": "evaluation",
                "problem": "3 attempts missing",
                "evidence": "agent-a/case-a (with-skill)",
                "action": "Rerun the failed attempts after checking logs.",
                "check": "Every planned attempt is scored.",
                "strength": "supported",
            }
        ],
    }


def test_joined_error_copies_become_coverage_and_actions_without_splitting_error_prose():
    payload = _failed_payload()
    view = _project(payload)

    assert set(view) == {"conclusions", "recommendations", "next_steps", "execution_errors"}
    text = " ".join(card["message"] for key in ("conclusions", "recommendations") for card in view[key])
    assert "3 of 6 attempts scored" in text
    assert "2 of 3 attempts scored" in text and "1 of 3 attempts scored" in text
    assert "agent-a" in text and "baseline" in text and "with-skill" in text
    assert "diagnostic contains a semicolon" not in text
    assert view["execution_errors"] == payload["execution_errors"]
    assert [item["message"] for item in view["recommendations"]] == ["Rerun the failed attempts after checking logs."]
    assert view["next_steps"] == payload["next_steps"]


def test_projection_does_not_mutate_or_alias_retained_feedback_and_errors():
    payload = _failed_payload()
    before = copy.deepcopy(payload)
    view = _project(payload)
    view["conclusions"][0]["message"] = "changed"
    view["execution_errors"].append("changed")
    view["next_steps"][0]["action"] = "changed"
    assert payload == before


def test_healthy_feedback_with_semicolons_and_duplicates_stays_verbatim():
    payload = {
        "execution_status": "succeeded",
        "conclusions": [{"title": "Author's conclusion", "message": "Keep this; exact wording.", "source": "llm"}] * 2,
        "recommendations": [{"message": "Explain one; then two.", "evidence": {"url": "https://example.test/case"}}],
        "next_steps": [{"kind": "more cases", "action": "Add cases; keep existing ones."}],
        "execution_errors": [],
    }
    assert _project(payload) == {
        key: payload[key] for key in ("conclusions", "recommendations", "next_steps", "execution_errors")
    }


def test_same_failure_at_different_cases_agents_or_evidence_is_not_deduplicated():
    payload = _failed_payload()
    cards = [
        {
            "title": "case-a",
            "message": "Timeout; inspect trace.",
            "agent": "agent-a",
            "case": "case-a",
            "evidence": {"url": "https://example.test/a"},
        },
        {
            "title": "case-b",
            "message": "Timeout; inspect trace.",
            "agent": "agent-a",
            "case": "case-b",
            "evidence": {"url": "https://example.test/b"},
        },
        {
            "title": "case-a",
            "message": "Timeout; inspect trace.",
            "agent": "agent-b",
            "case": "case-a",
            "evidence": {"url": "https://example.test/c"},
        },
    ]
    payload["recommendations"].extend([cards[0], *cards])
    view = _project(payload)
    scoped = [card for card in view["recommendations"] if card.get("case")]
    assert scoped == cards


@pytest.mark.parametrize(
    "counts",
    [{}, {"scored_attempts": None, "expected_attempts": None}, {"scored_attempts": True, "expected_attempts": 2}],
)
def test_missing_or_invalid_coverage_is_unavailable_never_invented_zero(counts):
    payload = {"execution_status": "failed", "execution_errors": ["setup failed"], **counts}
    view = _project(payload)
    text = " ".join(card["message"] for card in view["conclusions"])
    assert "unavailable" in text.lower()
    assert "0 of" not in text
    assert view["recommendations"] and "setup failed" not in view["recommendations"][0]["message"]


def test_unrecognized_authored_semicolon_feedback_is_preserved_on_failed_runs():
    payload = _failed_payload()
    authored = {"title": "case-a", "message": "Check authorization; then retry.", "source": "llm"}
    payload["conclusions"].append(authored)
    assert authored in _project(payload)["conclusions"]


def test_new_failed_evaluation_emits_actions_and_keeps_complete_errors_without_provider_calls(monkeypatch):
    from tests.reporting.test_tier3_incomplete_arms import _agent, _condition

    from skillevaluator.evaluation import insights_judge

    def forbidden(*args, **kwargs):
        pytest.fail("failed evaluation attempted an insights provider call")

    monkeypatch.setattr(insights_judge, "build_insights", forbidden)
    errors = ["AgentTimeoutError: agent stopped; keep this full detail", "JudgeError: required check failed"]
    agent = _agent(baseline=_condition("failed", 1, errors=errors), with_skill=_condition("failed", 1, errors=errors))
    payload = build_agent_eval_payload("demo", {"agent-a": agent}, use_llm_judge=True)
    assert payload["execution_status"] == "failed" and payload["overall_score"] is None
    assert payload["execution_errors"] == errors
    assert "2 of 4 attempts scored" in " ".join(item["message"] for item in payload["conclusions"])
    assert all(error not in card["message"] for error in errors for card in payload["recommendations"])
    assert payload["recommendations"]


def test_error_blob_replacement_keeps_its_distinct_trace_associations():
    payload = _failed_payload()
    blob = payload["recommendations"][0]
    payload["recommendations"] = [
        {**blob, "evidence": {"url": "https://example.test/trace-a", "agent": "agent-a"}},
        {**blob, "evidence": {"url": "https://example.test/trace-b", "agent": "agent-b"}},
    ]
    view = _project(payload)
    assert [card["evidence"]["url"] for card in view["recommendations"] if card.get("evidence")] == [
        "https://example.test/trace-a",
        "https://example.test/trace-b",
    ]
    assert all("diagnostic contains a semicolon" not in card["message"] for card in view["recommendations"])


def test_rendering_retained_failed_evidence_does_not_rebuild_or_call_a_provider(monkeypatch):
    from skillevaluator.evaluation import insights_judge, tier3_report
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting import HTMLReporter

    def forbidden(*args, **kwargs):
        pytest.fail("rendering retained evidence attempted evaluation or an insights provider call")

    monkeypatch.setattr(insights_judge, "build_insights", forbidden)
    monkeypatch.setattr(tier3_report, "build_agent_eval_payload", forbidden)
    payload = _failed_payload()
    payload.update({"skill_name": "demo", "verdict": "fail", "best_agent": "agent-a"})
    result = ValidationResult(validator_name="AGENT_EVAL", metadata={"agent_eval": payload})
    before = copy.deepcopy(payload)
    report = HTMLReporter(include_timestamp=False).render_all([result])
    assert "Evaluation incomplete" in report
    assert result.metadata["agent_eval"] == before


@pytest.mark.parametrize("level", ["agent", "condition"])
def test_retained_agent_or_condition_counters_supply_missing_run_coverage(level):
    payload = _failed_payload()
    payload.pop("scored_attempts")
    payload.pop("expected_attempts")
    if level == "agent":
        payload["agents"]["agent-a"].update(scored_attempts=3, expected_attempts=6)
        payload["agents"]["agent-a"].pop("conditions")
    view = _project(payload)
    assert "3 of 6 attempts scored" in view["conclusions"][0]["message"]
    assert "3 of 6 attempts remain unscored" in view["conclusions"][0]["message"]


def test_unknown_peer_coverage_prevents_partial_totals_being_reported_as_whole_run():
    payload = _failed_payload()
    payload.pop("scored_attempts")
    payload.pop("expected_attempts")
    payload["agents"]["unknown-peer"] = {"execution_status": "failed"}
    assert "unavailable" in _project(payload)["conclusions"][0]["message"].lower()


def test_structured_action_keeps_its_evidence_and_verification_check():
    payload = _failed_payload()
    [card] = _project(payload)["recommendations"]
    assert card["evidence_text"] == "agent-a/case-a (with-skill)"
    assert card["check"] == "Every planned attempt is scored."


def test_render_time_error_append_does_not_hide_a_recognizable_legacy_blob_copy():
    payload = _failed_payload()
    payload["execution_errors"].append("Later retained report diagnostic")
    view = _project(payload)
    assert len(view["conclusions"]) == 3
    assert len(view["recommendations"]) == 1
    assert view["execution_errors"][-1] == "Later retained report diagnostic"
    assert all("diagnostic contains a semicolon" not in card["message"] for card in view["conclusions"])


def test_partial_error_prose_is_not_misidentified_as_a_complete_legacy_blob():
    payload = _failed_payload()
    authored = {"title": "Evaluation incomplete", "message": "AgentTimeoutError: runtime failed"}
    payload["conclusions"] = [authored]
    assert authored in _project(payload)["conclusions"]


@pytest.mark.parametrize("mode,arm", [("with_skill_only", "with_skill"), ("baseline_only", "without_skill")])
def test_deliberately_skipped_arm_does_not_make_active_arm_coverage_unknown(mode, arm):
    payload = {
        "execution_status": "failed",
        "mode": mode,
        "agents": {
            "agent-a": {
                "conditions": {
                    arm: {"execution_status": "failed", "scored_attempts": 0, "expected_attempts": 2},
                }
            }
        },
    }
    text = _project(payload)["conclusions"][0]["message"]
    assert "0 of 2 attempts scored" in text and "2 of 2 attempts remain unscored" in text


def test_projecting_a_new_failed_producer_payload_twice_does_not_duplicate_cards():
    payload = _failed_payload()
    first = _project(payload)
    second = _project({**payload, **first})
    assert second == first


def test_authored_feedback_with_malformed_optional_metadata_remains_renderable():
    payload = _failed_payload()
    authored = {"title": "Author note", "message": "Inspect the case.", "insight_kind": ["legacy"]}
    payload["conclusions"].append(authored)
    payload["mode"] = {"legacy": True}
    payload.pop("scored_attempts")
    payload.pop("expected_attempts")
    assert authored in _project(payload)["conclusions"]


def test_error_summary_keeps_distinct_trace_associations_across_repeated_projection():
    payload = _failed_payload()
    blob = payload["conclusions"][0]
    payload["conclusions"] = [
        {**blob, "evidence": {"url": "https://example.test/trace-a"}},
        {**blob, "evidence": {"url": "https://example.test/trace-b"}},
    ]
    first = _project(payload)
    assert [card["evidence"]["url"] for card in first["conclusions"] if card.get("evidence")] == [
        "https://example.test/trace-a",
        "https://example.test/trace-b",
    ]
    assert _project({**payload, **first}) == first


@pytest.mark.parametrize("level", ["agent", "condition"])
def test_complete_structured_coverage_takes_precedence_over_stale_run_summary(level):
    payload = _failed_payload()
    payload.pop("scored_attempts")
    payload.pop("expected_attempts")
    payload["summary"] = {"scored_attempts": 1, "expected_attempts": 1}
    if level == "agent":
        payload["agents"]["agent-a"].update(scored_attempts=3, expected_attempts=6)
        payload["agents"]["agent-a"].pop("conditions")
    assert "3 of 6 attempts scored" in _project(payload)["conclusions"][0]["message"]
