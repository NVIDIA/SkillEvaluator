# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re

import pytest

from skillevaluator.evaluation import tier3_expectations, tier3_report
from skillevaluator.evaluation.tier3_expectations import build_case_diagnostics
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting.html import _bounded_tier3_preview
from skillevaluator.tier3.eval_core import llm_judge
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.harbor.report_data import load_agent_data


def _reward(**overrides):
    return {
        "entry_id": "case-1",
        "trial_id": "trial-1",
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "details": {},
        **overrides,
    }


def _diagnostic(case, reward, baseline=None):
    agents = {"codex": {"rewards": [reward]}}
    if baseline is not None:
        agents["codex"]["rewards_baseline"] = [baseline]
    cases, omitted = build_case_diagnostics([{"id": "case-1", **case}], agents)
    assert not omitted
    return cases[0]


def _statuses(diagnostic, attempt=0):
    return {item["expectation_id"]: item["status"] for item in diagnostic["attempts"][attempt]["judgments"]}


def test_full_and_partial_judge_results_reach_case_report(monkeypatch):
    requirements = ["Read the input\nwithout changing it.", "Write <output.txt>", "Verify the result"]
    response = json.dumps(
        {
            "results": [
                {"step": 3, "passed": True, "reason": "result checked"},
                {"step": 1, "passed": True, "reason": "input read"},
                {"step": 2, "passed": False, "reason": "output file absent"},
            ]
        }
    )
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_kw: (response, None))
    full = llm_judge.judge_behavior_check("Agent read the input and checked, but did not write output.", requirements)
    truncated = '{"results":[{"step":2,"passed":false,"reason":"output file absent"},{"step":3'
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_kw: (truncated, None))
    partial = llm_judge.judge_behavior_check("same trace", requirements)

    full["evidence_refs"] = [{"kind": "tool_observation", "excerpt": "ls: output.txt: No such file"}]
    rewards = [
        _reward(behavior_check=full["score"], details={"behavior_check": full}),
        _reward(trial_id="trial-2", behavior_check=partial["score"], details={"behavior_check": partial}),
    ]
    payload = tier3_report.build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 2,
                "scored_attempts": 2,
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.5),
                "rewards": rewards,
            }
        },
        dataset=[{"id": "case-1", "expected_behavior": requirements}],
        use_llm_judge=False,
    )
    diagnostic = payload["case_diagnostics"][0]
    assert [item["requirement"] for item in diagnostic["requirements"]] == requirements
    assert list(_statuses(diagnostic).values()) == ["met", "unmet", "met"]
    assert list(_statuses(diagnostic, 1).values()) == ["not_checked", "unmet", "not_checked"]
    assert "Salvaged 1/3" in diagnostic["attempts"][1]["judge_note"]

    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload
    html = HTMLReporter(include_timestamp=False).render_all([result])
    dataset_html = html.split('id="tier3-page-dataset"')[1].split('id="tier3-page-suggestions"')[0]
    for expected in ("Unmet (1)", "Not Checked (2)", "ls: output.txt: No such file", "Next diagnostic step:"):
        assert expected in dataset_html
    assert "Write &lt;output.txt&gt;" in dataset_html
    assert "Ownership is not established by the score" in dataset_html
    assert "trial-1" in dataset_html and "trial-2" in dataset_html
    assert '<ul class="t3-ds-diagnostics">' in dataset_html
    assert '<ul class="t3-ds-behaviors">' not in dataset_html
    assert "<div><b>expected_behavior/1</b>:" in dataset_html
    diagnostic_styles = re.findall(r"\.t3-ds-diagnostics[^{}]*\{([^{}]*)\}", html)
    assert diagnostic_styles
    assert not any("grid" in style or "counter" in style for style in diagnostic_styles)


@pytest.mark.parametrize(
    "detail,score,status",
    [
        (
            {"status": "error", "reason": "judge timed out", "results": [{"step": 1, "passed": False}]},
            None,
            "not_checked",
        ),
        ({"status": "not_applicable", "reason": "excluded by grader"}, None, "not_applicable"),
        ({"results": []}, None, "unscored"),
        ({"results": []}, 0.0, "not_checked"),
        ({"results": [{"step": 1, "passed": False}, {"step": 1, "passed": True}]}, 0.5, "not_checked"),
        ({"results": [{"step": True, "passed": False}]}, 0.0, "not_checked"),
        ({"results": [{"step": 1, "passed": "false"}]}, 0.0, "not_checked"),
        ({"results": [{"step": 1, "passed": None, "reason": "missing artifact"}]}, 0.0, "not_checked"),
    ],
)
def test_missing_or_invalid_verdicts_are_never_skill_defects(detail, score, status):
    diagnostic = _diagnostic(
        {"assertions": ["write a file"]}, _reward(behavior_check=score, details={"behavior_check": detail})
    )
    assert _statuses(diagnostic) == {"assertions/0": status}


def test_execution_error_overrides_nominal_scores_and_verdicts():
    diagnostic = _diagnostic(
        {"expected_behavior": ["write a file"]},
        _reward(error="No trajectory", details={"behavior_check": {"results": [{"step": 1, "passed": False}]}}),
    )
    judgment = diagnostic["attempts"][0]["judgments"][0]
    assert judgment["status"] == "not_checked"
    assert judgment["reason"] == "No trajectory"


def test_negative_control_routes_correctly_while_task_outcome_fails():
    reward = _reward(
        goal_accuracy=0.0,
        details={
            "skill_execution": {"should_trigger": False, "negative_check": {"passed": True, "reason": "skill avoided"}},
            "goal_accuracy": {"achieved": False, "reason": "wrong answer"},
            "accuracy": {
                "criteria": {"TASK_ADDRESSED": False, "SKILL_IDENTIFIED": False},
                "excluded_criteria": ["SKILL_IDENTIFIED"],
            },
        },
    )
    baseline = _reward(
        skill_execution=None,
        details={
            "skill_execution": {"status": "not_applicable", "reason": "skill not installed"},
            "goal_accuracy": {"achieved": True, "reason": "right answer"},
        },
    )
    diagnostic = _diagnostic({"expected_skill": None, "expected_output": "42"}, reward, baseline)
    assert _statuses(diagnostic) == {
        "expected_output": "unmet",
        "expected_output/accuracy": "unmet",
        "expected_skill": "met",
    }
    assert _statuses(diagnostic, 1)["expected_skill"] == "not_applicable"
    assert _statuses(diagnostic, 1)["expected_output"] == "met"
    accuracy = diagnostic["attempts"][0]["judgments"][1]
    assert accuracy["unmet_checks"] == ["TASK_ADDRESSED"]
    assert accuracy["inapplicable_checks"] == ["SKILL_IDENTIFIED"]
    assert "remove" not in json.dumps(diagnostic).lower()


def test_successful_positive_routing_does_not_hide_unmet_script_requirement():
    diagnostic = _diagnostic(
        {"expected_skill": "demo", "expected_script": "run.py"},
        _reward(
            details={
                "skill_execution": {
                    "activation": {"passed": True},
                    "script_execution": {"passed": False, "reason": "not invoked"},
                },
            }
        ),
    )
    assert _statuses(diagnostic) == {"expected_skill": "met", "expected_script": "unmet"}


def test_custom_grader_does_not_impersonate_standard_judgments():
    diagnostic = _diagnostic(
        {"assertions": ["write the file"]},
        _reward(
            metric_set="custom-only",
            details={
                "behavior_check": {"results": [{"step": 1, "passed": False}]},
            },
        ),
    )
    assert _statuses(diagnostic) == {"assertions/0": "unscored"}


def test_case_identity_and_condition_are_not_matched_by_position():
    cases, _ = build_case_diagnostics(
        [{"id": "case-1", "assertions": ["first"]}, {"id": "case-2", "assertions": ["second"]}],
        {
            "codex": {
                "rewards": [
                    _reward(entry_id="case-2", details={"behavior_check": {"results": [{"step": 1, "passed": False}]}})
                ],
                "conditions": {"with_skill": {"execution_errors": ["first trial crashed"]}},
            }
        },
    )
    assert _statuses(cases[0]) == {"assertions/0": "not_checked"}
    assert cases[0]["attempts"][0]["errors"] == ["first trial crashed"]
    assert _statuses(cases[1]) == {"assertions/0": "unmet"}


def test_conflicting_aliases_and_multirow_trials_fail_closed():
    diagnostic = _diagnostic({"assertions": ["current"], "expected_behavior": ["older"]}, _reward())
    assert _statuses(diagnostic) == {"assertions/0": "not_checked"}
    assert diagnostic["requirements"][0]["requirement"] == "current"
    cases, _ = build_case_diagnostics(
        [{"id": "case-1", "assertions": ["first"]}], {"codex": {"rewards": [_reward(), _reward()]}}
    )
    assert len(cases[0]["attempts"]) == 1
    assert _statuses(cases[0]) == {"assertions/0": "not_checked"}


def test_diagnostics_redact_credentials_and_bound_cardinality(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-credential")
    diagnostic = _diagnostic(
        {"assertions": ["Do not echo private-test-credential"]},
        _reward(
            details={
                "behavior_check": {
                    "results": [{"step": 1, "passed": False, "reason": "echoed private-test-credential"}],
                    "evidence_refs": [{"kind": "tool_observation", "excerpt": "private-test-credential"}],
                },
            }
        ),
    )
    assert "private-test-credential" not in json.dumps(diagnostic)
    monkeypatch.setattr(tier3_expectations, "_MAX_EXPECTATIONS", 1)
    monkeypatch.setattr(tier3_expectations, "_MAX_ATTEMPTS", 1)
    _, omitted = build_case_diagnostics(
        [{"id": "case-1", "assertions": ["a", "b"]}],
        {
            "codex": {"rewards": [_reward(), _reward(trial_id="trial-2")]},
        },
    )
    assert omitted == {"expectations": 1, "expectation_attempts": 1}


def test_baseline_only_omits_unrun_with_skill_condition():
    cases, _ = build_case_diagnostics(
        [{"id": "case-1", "assertions": ["first"]}],
        {
            "codex": {"rewards": [], "rewards_baseline": [_reward()]},
        },
        evaluation_mode="baseline_only",
    )
    assert [attempt["condition"] for attempt in cases[0]["attempts"]] == ["without_skill"]


@pytest.mark.parametrize("baseline_status", [None, "skipped"])
def test_loaded_with_skill_only_report_has_no_phantom_baseline(tmp_path, monkeypatch, baseline_status):
    monkeypatch.setattr(tier3_expectations, "_MAX_ATTEMPTS", 2)
    active = tmp_path / "codex" / "with-skill"
    active.mkdir(parents=True)
    (active / "summary.json").write_text(
        json.dumps(
            {
                "scores": dict.fromkeys(DEFAULT_METRICS, 1.0),
                "metrics": list(DEFAULT_METRICS),
                "execution_status": "succeeded",
                "num_trials": 2,
                "expected_attempts": 2,
                "scored_attempts": 2,
            }
        )
    )
    for entry_id in ("case-1", "case-2"):
        trial = active / "trials" / entry_id
        trial.mkdir(parents=True)
        (trial / "reward.json").write_text(json.dumps(_reward(entry_id=entry_id, trial_id=f"trial-{entry_id}")))
    if baseline_status:
        baseline = tmp_path / "codex" / "without-skill"
        baseline.mkdir()
        (baseline / "summary.json").write_text(json.dumps({"scores": {}, "execution_status": baseline_status}))
    agents = load_agent_data(tmp_path)
    assert agents["codex"]["rewards_baseline"] == []
    payload = tier3_report.build_agent_eval_payload(
        "demo",
        agents,
        dataset=[{"id": entry_id, "assertions": ["first"]} for entry_id in ("case-1", "case-2")],
        use_llm_judge=False,
    )
    assert payload["mode"] == "with_skill_only"
    assert len(payload["case_diagnostics"]) == 2
    for case in payload["case_diagnostics"]:
        assert [(attempt["condition"], attempt["trial_id"]) for attempt in case["attempts"]] == [
            ("with_skill", f"trial-{case['entry_id']}")
        ]
    assert "expectation_attempts" not in payload.get("report_truncation", {}).get("omitted", {})


def test_skipped_conditions_and_absent_empty_arms_do_not_consume_attempt_budget(monkeypatch):
    monkeypatch.setattr(tier3_expectations, "_MAX_ATTEMPTS", 3)
    cases, omitted = build_case_diagnostics(
        [{"id": "case-1", "assertions": ["first"]}],
        {
            "codex": {
                "rewards": [_reward()],
                "rewards_baseline": [],
                "conditions": {"without_skill": {"execution_status": "skipped"}},
            },
            "other": {"rewards": [_reward()], "rewards_baseline": [_reward()]},
            "absent": {"rewards": [], "rewards_baseline": []},
        },
    )
    assert not omitted
    assert [(a["agent"], a["condition"]) for a in cases[0]["attempts"]] == [
        ("codex", "with_skill"),
        ("other", "with_skill"),
        ("other", "without_skill"),
    ]


@pytest.mark.parametrize("status", ["failed", "unknown"])
def test_unscored_failed_or_unknown_arm_remains_visible(status):
    cases, omitted = build_case_diagnostics(
        [{"id": "case-1", "assertions": ["first"]}],
        {
            "codex": {
                "rewards": [_reward()],
                "rewards_baseline": [],
                "conditions": {"without_skill": {"execution_status": status, "execution_errors": ["No result"]}},
            }
        },
    )
    assert not omitted
    assert [attempt["condition"] for attempt in cases[0]["attempts"]] == ["with_skill", "without_skill"]
    assert _statuses(cases[0], 1) == {"assertions/0": "not_checked"}
    assert cases[0]["attempts"][1]["errors"] == ["No result"]


def test_large_preview_omits_whole_diagnostics_without_corrupting_identities():
    payload = {
        "summary": {"overall_score": 0.8},
        "case_diagnostics": [
            {"entry_id": "case-1", "requirements": [{"requirement": "x" * 300_000}]},
        ],
    }
    preview, notice = _bounded_tier3_preview(payload)
    assert "case_diagnostics" not in preview
    assert notice["case_diagnostics"] == 1
    assert preview["summary"] == payload["summary"]
    assert payload["case_diagnostics"][0]["entry_id"] == "case-1"


def test_canonical_budget_records_dropped_diagnostics(monkeypatch):
    monkeypatch.setattr(tier3_report, "_MAX_EMBEDDED_REPORT_BYTES", 1000)
    payload = {"summary": {"overall_score": 0.8}, "case_diagnostics": [{"entry_id": "case-1", "large": "x" * 2000}]}
    budget = tier3_report._ReportBudget()
    tier3_report._enforce_report_payload_budget(payload, budget)
    assert "case_diagnostics" not in payload
    assert payload["report_truncation"]["omitted"]["case_diagnostics"] == 1
    assert payload["summary"]["overall_score"] == 0.8


@pytest.mark.parametrize("oversized_reasons", [False, True])
def test_canonical_budget_restores_reasons_after_dropping_diagnostics(monkeypatch, oversized_reasons):
    monkeypatch.setattr(tier3_report, "_MAX_EMBEDDED_REPORT_BYTES", 1500)
    reason = "x" * (2000 if oversized_reasons else 100)
    evidence = {"reason": reason, "excerpt": "observed result"}
    cards = [{"evidence": [evidence]}]
    payload = {
        "summary": {"overall_score": 0.2},
        "agents": {"codex": {"evaluator_cards": cards, "trials_baseline": [{"score_reasons": {"accuracy": reason}}]}},
        "evaluator_cards": cards,
        "trials": [{"agent": "codex", "score_reasons": {"accuracy": reason}}],
        "case_diagnostics": [{"entry_id": "case-1", "large": "x" * 2000}],
    }
    budget = tier3_report._ReportBudget()
    tier3_report._enforce_report_payload_budget(payload, budget)
    assert "case_diagnostics" not in payload
    assert budget.omitted.get("case_diagnostics") == 1
    assert tier3_report._serialized_payload_size(payload) <= 1500
    if oversized_reasons:
        assert budget.omitted["failure_reasons"] == 4
        assert "reason" not in evidence and "excerpt" not in evidence
        assert "score_reasons" not in payload["trials"][0]
        assert "score_reasons" not in payload["agents"]["codex"]["trials_baseline"][0]
    else:
        assert "failure_reasons" not in budget.omitted
        assert evidence == {"reason": reason, "excerpt": "observed result"}
        assert payload["trials"][0]["score_reasons"] == {"accuracy": reason}
        assert payload["agents"]["codex"]["trials_baseline"][0]["score_reasons"] == {"accuracy": reason}
