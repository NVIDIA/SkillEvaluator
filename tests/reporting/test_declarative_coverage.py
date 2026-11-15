# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative coverage survives aggregation and every supported report surface."""

from __future__ import annotations

import copy
import io
import json
from html.parser import HTMLParser

import pytest
from rich.console import Console

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory, build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter, JSONReporter
from skillevaluator.reporting.base import agent_eval_report_serialization_issue
from skillevaluator.reporting.declarative_coverage import coverage_summary
from skillevaluator.tier3.declarative_coverage import (
    COMPACT_CHECK_ROWS_PER_ARM,
    agent_declarative_checks,
    aggregate_declarative_checks,
    compact_declarative_checks,
)
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.result_display import render_evaluation_result


class _VisibleHTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = False
        self.parts = []

    def handle_starttag(self, tag, _attrs):
        if tag in {"script", "style"}:
            self.hidden = True

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = False

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


@pytest.mark.parametrize("tag", ["script", "SCRIPT", "script ", "style"])
def test_visible_html_text_excludes_embedded_data(tag):
    parser = _VisibleHTMLText()
    parser.feed(f"<p>Visible evidence</p><{tag}>Embedded data</{tag}><p>More evidence</p>")
    parser.close()
    assert " ".join(parser.parts) == "Visible evidence More evidence"


def _reward(*, baseline=False):
    checks = [
        {"type": "script_invoked", "expected": "scripts/build.py", "passed": True, "reason": "Invoked build script"},
        {
            "type": "command_succeeded",
            "expected": "scripts/build.py",
            "passed": False,
            "reason": "Invoked but exited 2",
        },
        {"type": "file_exists", "expected": "out/report.md", "passed": None, "reason": "File evidence unavailable"},
    ]
    if baseline:
        for check in checks:
            check.update(passed=None, reason="Not applicable without the skill")
    return {
        **dict.fromkeys(DEFAULT_METRICS, 0.8),
        "overall": 0.8,
        "entry_id": "build-case",
        "_trial_name": "build-case__baseline" if baseline else "build-case__attempt",
        "details": {
            "behavior_check": {
                "score": 0.5,
                "status": "accepted",
                "declarative_checks": {
                    "results": checks,
                    "judged_count": 2,
                    "checked_count": 0 if baseline else 2,
                    "not_checked_count": 3 if baseline else 1,
                },
            }
        },
    }


def _info(*, checks=True):
    rewards = [_reward()]
    baseline = [_reward(baseline=True)]
    if not checks:
        for reward in rewards + baseline:
            reward["details"] = {}
    return {
        "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
        "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
        "execution_status": "succeeded",
        "expected_attempts": 2,
        "scored_attempts": 2,
        "conditions": {
            arm: {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}
            for arm in ("with_skill", "without_skill")
        },
        "rewards": rewards,
        "rewards_baseline": baseline,
    }


def _payload(*, checks=True):
    info = _info(checks=checks)
    return build_agent_eval_payload("demo", {"agent-a": info, "agent-b": copy.deepcopy(info)}, use_llm_judge=False)


def _validation(payload):
    return ValidationResult(validator_name="AGENT_EVAL", metadata={"agent_eval": payload})


def test_coverage_counts_exclude_not_checked_and_do_not_mutate_rewards():
    rewards = [_reward(), _reward(baseline=True)]
    original = copy.deepcopy(rewards)
    coverage = aggregate_declarative_checks(rewards)
    assert {
        key: coverage[key]
        for key in (
            "checked_count",
            "passed_count",
            "failed_count",
            "judged_count",
            "not_checked_count",
            "score_denominator",
        )
    } == {
        "checked_count": 2,
        "passed_count": 1,
        "failed_count": 1,
        "judged_count": 4,
        "not_checked_count": 4,
        "score_denominator": 6,
    }
    assert [item["score_denominator"] for item in coverage["attempts"]] == [4, 2]
    assert len(coverage["results"]) == 6
    assert rewards == original


@pytest.mark.parametrize(
    "rewards",
    [
        [],
        [{"details": {}}],
        [{"details": {"behavior_check": {"declarative_checks": {"results": [], "judged_count": 2}}}}],
    ],
)
def test_no_declarative_checks_omit_section(rewards):
    assert aggregate_declarative_checks(rewards) == {}
    assert agent_declarative_checks(rewards, []) == {}


def test_error_and_unknown_judgment_counts_do_not_invent_a_score():
    reward = _reward()
    reward["details"]["behavior_check"].update(status="error", score=None)
    reward["details"]["behavior_check"]["declarative_checks"].pop("judged_count")
    coverage = aggregate_declarative_checks([reward])
    assert coverage["judged_count"] is None
    assert coverage["score_denominator"] is None
    assert coverage["attempts"][0]["score_status"] == "error"
    assert coverage["checked_count"] == 2


def test_canonical_payload_keeps_both_agents_and_arms():
    payload = _payload()
    for agent in payload["agents"].values():
        coverage = agent["declarative_checks"]
        assert coverage["with_skill"]["checked_count"] == 2
        assert coverage["with_skill"]["score_denominator"] == 4
        assert coverage["without_skill"]["not_checked_count"] == 3
        assert coverage["without_skill"]["score_denominator"] == 2
        assert coverage["with_skill"]["results"][1]["reason"] == "Invoked but exited 2"


def _terminal(payload, *, verbose):
    stream = io.StringIO()
    console = Console(file=stream, width=220, force_terminal=False, color_system=None)
    render_evaluation_result({"agents": payload["agents"]}, console=console, verbose=verbose)
    return stream.getvalue()


@pytest.mark.parametrize("verbose", [True, False])
def test_evaluate_terminal_shows_coverage_even_in_compact_mode(verbose):
    text = _terminal(_payload(), verbose=verbose)
    assert "Declarative-check coverage" in text
    assert "Checked 2 (1 passed / 1 failed); judged 2; not checked 1; behavior denominator 4" in text
    for value in (
        "agent-a",
        "agent-b",
        "with skill",
        "without skill",
        "build-case",
        "command_succeeded",
        "scripts/build.py",
        "Invoked but exited 2",
        "File evidence unavailable",
        "Not applicable without the skill",
    ):
        assert value in text


def test_validate_terminal_shows_coverage():
    stream = io.StringIO()
    console = Console(file=stream, width=220, force_terminal=False, color_system=None)
    CLIReporter._print_agent_eval_tables(_payload(), console)
    text = stream.getvalue()
    assert "behavior denominator 4" in text
    assert "File evidence unavailable" in text


@pytest.mark.parametrize(
    "reporter",
    [
        HTMLReporter(report_scope="tier3"),
        BenchmarkReporter(skill_name="demo", report_scope="tier3"),
    ],
)
def test_html_and_both_benchmark_cards_show_all_checks(reporter):
    output = reporter.render(_validation(_payload()))
    visible = output
    if isinstance(reporter, HTMLReporter):
        parser = _VisibleHTMLText()
        parser.feed(output)
        parser.close()
        visible = " ".join(parser.parts)
    if isinstance(reporter, BenchmarkReporter):
        visible = visible.replace("\\_", "_")
    for value in (
        "Declarative-check coverage",
        "Checked 2 (1 passed / 1 failed)",
        "judged 2; not checked 1; behavior denominator 4",
        "agent-a",
        "agent-b",
        "build-case",
        "script_invoked",
        "command_succeeded",
        "scripts/build.py",
        "Invoked but exited 2",
        "File evidence unavailable",
        "Not applicable without the skill",
    ):
        assert value in visible


def test_exported_report_json_keeps_structured_counts_and_evidence():
    output = json.loads(JSONReporter().render(_validation(_payload())))
    assert "declarative_checks" in json.dumps(output)
    encoded = json.dumps(output)
    for value in (
        '"score_denominator": 4',
        '"passed_count": 1',
        '"failed_count": 1',
        '"judged_count": 2',
        '"not_checked_count": 3',
        '"passed": null',
        "File evidence unavailable",
        "Not applicable without the skill",
    ):
        assert value in encoded


def test_html_escapes_untrusted_check_evidence():
    payload = _payload()
    for agent in payload["agents"].values():
        agent["declarative_checks"]["with_skill"]["results"][0]["reason"] = '<img src=x onerror="alert(1)">'
    output = HTMLReporter(report_scope="tier3").render(_validation(payload))
    assert '<img src=x onerror="alert(1)">' not in output
    assert "&lt;img" in output


def test_coverage_redacts_configured_credentials(monkeypatch):
    credential = "example-private-credential-value"
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    reward = _reward()
    reward["details"]["behavior_check"]["declarative_checks"]["results"][0]["reason"] = f"Error: {credential}"
    assert credential not in json.dumps(aggregate_declarative_checks([reward]))


@pytest.mark.parametrize("format_name", ["terminal", "cli", "html", "benchmark", "json"])
def test_non_declarative_reports_add_no_coverage(format_name):
    payload = _payload(checks=False)
    assert all("declarative_checks" not in agent for agent in payload["agents"].values())
    result = _validation(payload)
    if format_name == "terminal":
        output = _terminal(payload, verbose=False)
    else:
        reporter = {
            "cli": CLIReporter(),
            "html": HTMLReporter(report_scope="tier3"),
            "benchmark": BenchmarkReporter(skill_name="demo", report_scope="tier3"),
            "json": JSONReporter(),
        }[format_name]
        output = reporter.render(result)
    assert "Declarative-check coverage" not in output
    assert '"declarative_checks"' not in output


def test_collector_result_json_carries_coverage(tmp_path):
    jobs = tmp_path / "jobs"
    for suffix, baseline in (("with", False), ("without", True)):
        job = jobs / f"demo-opencode-{suffix}"
        verifier = job / "build-case__attempt" / "verifier"
        verifier.mkdir(parents=True)
        reward = _reward(baseline=baseline)
        reward.pop("_trial_name")
        (verifier / "reward.json").write_text(json.dumps(reward))
        (job / "build-case__attempt" / "result.json").write_text(json.dumps({"trial_name": "build-case__attempt"}))
        (job / "result.json").write_text(
            json.dumps(
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
                            "eval": {
                                "n_trials": 1,
                                "n_errors": 0,
                                "reward_stats": {"reward": {"0.8": ["build-case__attempt"]}},
                            }
                        },
                    },
                }
            )
        )
    result = collect_harbor_results(
        "demo", ["opencode"], tmp_path / "output", jobs, expected_cases=1, expected_case_ids=["build-case"]
    )
    persisted = json.loads(json.dumps(result))
    coverage = persisted["agents"]["opencode"]["declarative_checks"]
    assert coverage["with_skill"]["passed_count"] == 1
    assert coverage["with_skill"]["score_denominator"] == 4
    assert coverage["without_skill"]["not_checked_count"] == 3
    assert coverage["with_skill"]["results"][1]["reason"] == "Invoked but exited 2"


def test_failed_arm_retains_coverage_without_restoring_withheld_quality(tmp_path):
    arm = tmp_path / "run" / "agent-a" / "with-skill"
    trial = arm / "trials" / "build-case__attempt"
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text(json.dumps(_reward()))
    (arm / "summary.json").write_text(
        json.dumps(
            {
                "scores": dict.fromkeys(DEFAULT_METRICS, 0.8),
                "metrics": list(DEFAULT_METRICS),
                "num_trials": 1,
                "execution_status": "failed",
                "execution_errors": ["Another attempt failed"],
                "expected_attempts": 2,
                "scored_attempts": 1,
            }
        )
    )
    skill = tmp_path / "demo"
    skill.mkdir()
    result = agent_eval_result_from_directory(skill, tmp_path / "run", use_llm_judge=False)
    agent = result.metadata["agent_eval"]["agents"]["agent-a"]
    assert agent["declarative_checks"]["with_skill"]["score_denominator"] == 4
    assert agent["declarative_checks"]["with_skill"]["results"][2]["passed"] is None
    assert agent["with_skill"] is None
    assert agent["conditions"]["with_skill"]["execution_status"] == "failed"
    html = HTMLReporter(report_scope="tier3").render(result)
    assert "behavior denominator 4" in html
    assert "File evidence unavailable" in html


def test_compact_coverage_keeps_exact_counts_and_prefers_failing_rows():
    full = aggregate_declarative_checks([_reward() for _ in range(4)])
    original = copy.deepcopy(full)
    assert len(full["results"]) == 12

    compact, dropped = compact_declarative_checks({"with_skill": full}, max_rows=5)

    arm = compact["with_skill"]
    assert dropped == 7
    assert arm["omitted_checks"] == 7
    assert [row["passed"] for row in arm["results"]] == [False, None, False, None, False]
    assert {key: value for key, value in arm.items() if key not in {"results", "omitted_checks"}} == {
        key: value for key, value in full.items() if key != "results"
    }
    assert full == original
    assert compact_declarative_checks(compact, max_rows=5) == (compact, 0)
    assert compact_declarative_checks(compact, max_rows=3)[0]["with_skill"]["omitted_checks"] == 9
    assert compact_declarative_checks({"with_skill": full}, max_rows=12) == ({"with_skill": full}, 0)


def test_coverage_summary_says_how_many_check_rows_were_omitted():
    group = {"checked_count": 2, "passed_count": 1, "failed_count": 1, "judged_count": 0}
    assert "omitted" not in coverage_summary(group)
    assert coverage_summary({**group, "omitted_checks": 7}).endswith("; 7 more check rows omitted to fit report limits")


def _large_run_info(*, cases, attempts, checks):
    def rewards(arm):
        return [
            {
                **dict.fromkeys(DEFAULT_METRICS, 0.8),
                "entry_id": f"case-{case:03d}",
                "_trial_name": f"case-{case:03d}__{arm}{attempt}",
                "details": {
                    "behavior_check": {
                        "status": "accepted",
                        "declarative_checks": {
                            "judged_count": 0,
                            "results": [
                                {
                                    "type": "script_invoked",
                                    "expected": f"scripts/s{check}.py",
                                    "passed": check != 0,
                                    "reason": f"check {check}",
                                }
                                for check in range(checks)
                            ],
                        },
                    }
                },
            }
            for case in range(cases)
            for attempt in range(attempts)
        ]

    info = _info(checks=False)
    info.update(rewards=rewards("a"), rewards_baseline=rewards("b"), without_skill=dict.fromkeys(DEFAULT_METRICS, 0.6))
    return info


def test_large_declarative_run_bounds_coverage_before_scored_report_details():
    info = _large_run_info(cases=120, attempts=3, checks=10)
    full = agent_declarative_checks(info["rewards"], info["rewards_baseline"])
    assert all(len(arm["results"]) == 3600 for arm in full.values())

    payload = build_agent_eval_payload("demo", {"agent-a": info}, use_llm_judge=False)

    assert agent_eval_report_serialization_issue(payload) is None
    agent = payload["agents"]["agent-a"]
    for arm, expected in full.items():
        coverage = agent["declarative_checks"][arm]
        assert len(coverage["results"]) == COMPACT_CHECK_ROWS_PER_ARM
        assert coverage["omitted_checks"] == 3600 - COMPACT_CHECK_ROWS_PER_ARM
        assert {key: value for key, value in coverage.items() if key not in {"results", "omitted_checks"}} == {
            key: value for key, value in expected.items() if key != "results"
        }
    omitted = payload["report_truncation"]["omitted"]
    assert omitted["declarative_check_rows"] == 2 * (3600 - COMPACT_CHECK_ROWS_PER_ARM)
    assert not {"non_best_agent_details", "dataset_and_trial_items", "attempt_verdict_details"} & set(omitted)
    assert payload["overall_score"] == pytest.approx(0.8)
    assert agent["with_skill"] == pytest.approx(0.8)
    assert agent["baseline"] == pytest.approx(0.6)
    assert agent["lift"] == pytest.approx(0.2)
    assert len(agent["dimensions"]) == 5
    assert len(agent["evaluator_cards"]) == 6
    assert len(agent["trials"]) == len(agent["trials_baseline"]) == len(agent["cases"]) == 360
    html = HTMLReporter(report_scope="tier3").render(_validation(payload))
    assert "3400 more check rows omitted to fit report limits" in html


def test_declarative_run_within_report_limits_keeps_every_check_row():
    info = _large_run_info(cases=10, attempts=3, checks=10)
    payload = build_agent_eval_payload("demo", {"agent-a": info}, use_llm_judge=False)
    for arm in payload["agents"]["agent-a"]["declarative_checks"].values():
        assert len(arm["results"]) == 300
        assert "omitted_checks" not in arm
    assert "declarative_check_rows" not in payload.get("report_truncation", {}).get("omitted", {})
