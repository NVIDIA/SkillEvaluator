# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intentionally omitted baseline arms do not make scored execution incomplete."""

import json
import shutil

import pytest
from tests.reporting.test_case_findings_report import _run
from tests.test_tier3_case_summary import _agent, _case

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory, render_agent_eval_html_report
from skillevaluator.reporting import BenchmarkReporter
from skillevaluator.tier3.case_summary import _contract_mismatch, _execution_unavailable, verdict
from skillevaluator.tier3.next_steps import next_steps
from skillevaluator.tier3.result_display import render_result


@pytest.fixture
def skipped_agent():
    agent = _agent({"case": _case(True)}, {})
    agent.update(
        metrics_with_skill=["security", "skill_execution", "accuracy", "goal_accuracy"],
        metrics_without_skill=[],
        conditions={
            "with_skill": {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1},
            "without_skill": {"execution_status": "skipped", "expected_attempts": 0, "scored_attempts": 0},
        },
        lift={},
        dimensions_without_skill={},
    )
    return agent


def test_skipped_baseline_is_not_an_execution_or_contract_failure(skipped_agent, tmp_path):
    assert not _contract_mismatch(skipped_agent)
    assert not _execution_unavailable(skipped_agent)
    assert verdict({"opencode": skipped_agent}) == (
        "lift not measured",
        "opencode: no baseline was run; lift not measured",
    )
    result = {"execution_status": "succeeded", "agents": {"opencode": skipped_agent}}
    assert all(step["kind"] != "incomplete run" for step in next_steps(result, run_dir=tmp_path))
    for agents in ({"opencode": skipped_agent}, {"opencode": skipped_agent, "codex": skipped_agent}):
        terminal = render_result({**result, "agents": agents})
        assert "no baseline was run; lift not measured" in terminal
        assert "contracts differ" not in terminal
        assert "execution is incomplete" not in terminal
        assert "  ?" not in "\n".join(line for line in terminal.splitlines() if "? incomplete" not in line)


@pytest.mark.parametrize("status,expected", [("failed", 1), ("missing", 1), ("skipped", 1), ("skipped", None)])
def test_unscored_baseline_is_not_treated_as_intentionally_omitted(skipped_agent, tmp_path, status, expected):
    skipped_agent["conditions"]["without_skill"].update(execution_status=status, expected_attempts=expected)
    assert _execution_unavailable(skipped_agent)
    assert _contract_mismatch(skipped_agent)
    assert verdict({"opencode": skipped_agent})[0] == "inconclusive"
    steps = next_steps({"agents": {"opencode": skipped_agent}}, run_dir=tmp_path)
    assert any(step["kind"] == "incomplete run" for step in steps)


@pytest.mark.parametrize("status", ["failed", "succeeded"])
def test_skipped_baseline_does_not_hide_with_skill_failure(skipped_agent, tmp_path, status):
    skipped_agent["execution_status"] = status
    skipped_agent["conditions"]["with_skill"].update(execution_status=status, scored_attempts=0)
    skipped_agent["pass_at_k"]["with_skill"]["cases"]["case"] = _case(missing=1)
    assert _execution_unavailable(skipped_agent) is (status == "failed")
    assert verdict({"opencode": skipped_agent})[0] == "inconclusive"
    assert any(
        step["kind"] == "incomplete run"
        for step in next_steps({"agents": {"opencode": skipped_agent}}, run_dir=tmp_path)
    )


def test_empty_agents_do_not_claim_a_skipped_baseline():
    assert verdict({}) == ("", "")
    assert not _contract_mismatch({})


def test_saved_run_html_and_benchmark_keep_no_baseline_wording(skipped_agent, tmp_path):
    run = _run(tmp_path, goal_accuracy=1.0)
    shutil.rmtree(run / "opencode" / "without-skill")
    source = {"execution_status": "succeeded", "agents": {"opencode": skipped_agent}}
    (run / "result.json").write_text(json.dumps(source))
    html = render_agent_eval_html_report(tmp_path, run, use_llm_judge=False, include_benchmark=True).read_text()
    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)
    assert result is not None
    publication = BenchmarkReporter(skill_name=tmp_path.name).render_all([result])
    tier3 = (run / "tier3-benchmark.md").read_text()
    for text in (html, publication, tier3):
        assert "contracts differ" not in text
        assert "execution is incomplete" not in text
    assert "no baseline was run; lift not measured" in html
    assert "no baseline" in tier3.lower()
    assert "INCOMPLETE" in publication
