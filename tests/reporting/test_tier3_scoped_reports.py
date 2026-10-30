# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completed-agent display never changes whole-run publication evidence."""

from __future__ import annotations

import copy

import pytest

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, HTMLReporter
from skillevaluator.reporting.base import UnsafeReportPathError, assess_publication
from skillevaluator.reporting.tier3_scope import completed_agent_result
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def result_with_completed_peer(*, baseline=0.4, peer_failed=True):
    conditions = {
        arm: {"execution_status": "succeeded", "execution_errors": [], "expected_attempts": 1, "scored_attempts": 1}
        for arm in ("with_skill", "without_skill")
    }
    agents = {
        "opencode": {
            "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
            "without_skill": dict.fromkeys(DEFAULT_METRICS, baseline),
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 2,
            "scored_attempts": 2,
            "conditions": conditions,
            "num_trials": 1,
            "num_trials_baseline": 1,
        }
    }
    if peer_failed:
        agents["codex"] = {
            "execution_status": "failed",
            "execution_errors": ["Agent setup failed"],
            "with_skill": {},
            "without_skill": {},
        }
    payload = build_agent_eval_payload(
        "demo",
        agents,
        dataset=[{"id": "case-a"}],
        attempt_policy={"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = payload
    if peer_failed:
        result.add_error("Agent setup failed")
    return result


@pytest.mark.parametrize("baseline,lift", [(0.4, "+0.4000"), (0.9, "-0.1000"), (0.8, "+0.0000")])
def test_scoped_reports_show_completed_agent_and_signed_lift(baseline, lift):
    result = result_with_completed_peer(baseline=baseline)
    original = copy.deepcopy(result)
    assessment = assess_publication([result])
    benchmark = BenchmarkReporter(report_scope="tier3").render(result)
    html = HTMLReporter(report_scope="tier3").render(result)
    assert "score **0.8000** (80.00%)" in benchmark
    assert lift in benchmark
    assert "Needs attention" in benchmark
    assert "Publication: **NOT ASSESSED**" in benchmark
    assert "Publication blocked" not in benchmark
    assert "Completed-agent result" in html
    assert ">0.80</div>" in html
    assert 'data-publication-status="not-assessed"' in html
    assert "Agent setup failed" in html
    assert result == original
    assert assess_publication([result]) == assessment
    assert assessment.status != "pass"


def test_default_report_keeps_publication_gate_and_format():
    result = result_with_completed_peer()
    benchmark = BenchmarkReporter().render(result)
    html = HTMLReporter().render(result)
    assert "Overall verdict: FAIL" in benchmark
    assert 'data-publication-status="fail"' in html
    assert "## Results at a Glance" in benchmark
    assert "## Tier Status" in benchmark


@pytest.mark.parametrize("corruption", ["bool", "nan", "missing", "errors", "coverage", "conditions"])
def test_no_completed_summary_for_invalid_agent_evidence(corruption):
    result = result_with_completed_peer()
    payload = result.metadata["agent_eval"]
    agent = payload["agents"]["opencode"]
    assert completed_agent_result(payload)
    if corruption == "bool":
        agent["with_skill"] = True
    elif corruption == "nan":
        agent["with_skill"] = float("nan")
    elif corruption == "missing":
        agent["with_skill"] = None
    elif corruption == "errors":
        agent["execution_errors"] = ["failed"]
    elif corruption == "coverage":
        agent["scored_attempts"] = 1
    elif corruption == "conditions":
        agent["conditions"]["with_skill"]["execution_status"] = "failed"
    assert completed_agent_result(payload) is None


def test_incompatible_basis_has_no_invented_lift():
    result = result_with_completed_peer()
    payload = result.metadata["agent_eval"]
    payload["agents"]["opencode"]["metrics_without_skill"] = ["accuracy"]
    summary = completed_agent_result(payload)
    assert summary["score"] == 0.8
    assert summary["lift"] is None


@pytest.mark.parametrize("reporter", [BenchmarkReporter, HTMLReporter])
def test_scope_cannot_hide_other_tier_results(reporter):
    with pytest.raises(ValueError, match="cannot hide"):
        reporter(report_scope="tier3").render_all([ValidationResult(validator_name="MANIFEST")])
    with pytest.raises(ValueError, match="report_scope"):
        reporter(report_scope="anything")


def test_scope_does_not_accept_partial_dataset_coverage():
    result = result_with_completed_peer()
    payload = result.metadata["agent_eval"]
    payload["dataset_summary"]["total_tasks"] = 32
    assert completed_agent_result(payload) is None


def test_standard_report_pair_and_unsafe_companion_destination(tmp_path):
    import json

    from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report

    skill = tmp_path / "demo"
    skill.mkdir()
    run = tmp_path / "run"
    for arm in ("with-skill", "without-skill"):
        directory = run / "opencode" / arm
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "scores": dict.fromkeys(DEFAULT_METRICS, 0.8),
                    "metrics": list(DEFAULT_METRICS),
                    "num_trials": 1,
                    "execution_status": "succeeded",
                    "execution_errors": [],
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                }
            )
        )
    report = render_agent_eval_html_report(skill, run, include_benchmark=True, use_llm_judge=False)
    assert report.is_file()
    benchmark = run / "tier3-benchmark.md"
    assert "Publication: **NOT ASSESSED**" in benchmark.read_text()
    assert "## Results at a Glance" in benchmark.read_text()
    benchmark.unlink()
    private = tmp_path / "untouched.md"
    private.write_text("preserve this file")
    benchmark.symlink_to(private)
    with pytest.raises(UnsafeReportPathError, match=r"(?i)symlink|unsafe|regular"):
        render_agent_eval_html_report(skill, run, include_benchmark=True, use_llm_judge=False)
    assert private.read_text() == "preserve this file"
    assert benchmark.is_symlink()
    assert not report.exists()


def test_without_baseline_preserves_score_without_inventing_comparison():
    result = result_with_completed_peer(peer_failed=False)
    payload = result.metadata["agent_eval"]
    agent = payload["agents"]["opencode"]
    del agent["conditions"]["without_skill"]
    agent["expected_attempts"] = agent["scored_attempts"] = 1
    summary = completed_agent_result(payload)
    assert summary["score"] == 0.8
    assert summary["baseline"] is None
    assert summary["lift"] is None
