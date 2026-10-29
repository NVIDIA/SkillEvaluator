# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""report.html leads with the case-based verdict and fixes shared with the CLI."""

import json

from tests.reporting.test_resource_comparison import write_arm

from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def _run(tmp_path, *, goal_accuracy: float):
    run = tmp_path / "run"
    trial = write_arm(run, "with-skill", 10, 3)
    write_arm(run, "without-skill", 20, 5)
    reward = {
        "entry_id": "case",
        "trial_id": "case__1",
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "goal_accuracy": goal_accuracy,
    }
    reward["details"] = {"goal_accuracy": {"reason": "no report", "end_state": "asked for the data"}}
    (trial / "reward.json").write_text(json.dumps(reward))

    def condition(passed):
        attempt = {"attempt": 1, "trial": "case__1", "score": 0.9 if passed else 0.2, "passed": passed}
        return {"k": 1, "cases": {"case": {"passed": passed, "attempts": [attempt]}}}

    agent = {"pass_at_k": {"with_skill": condition(True), "without_skill": condition(False)}}
    for arm, passed in (("with-skill", True), ("without-skill", False)):
        summary = json.loads((run / "opencode" / arm / "summary.json").read_text())
        (run / "opencode" / arm / "summary.json").write_text(json.dumps({**summary, "pass_at_k": condition(passed)}))
    (run / "result.json").write_text(json.dumps({"agents": {"opencode": agent}}))
    return run


def test_report_leads_with_skill_effect_and_goal_not_met_fix(tmp_path) -> None:
    run = _run(tmp_path, goal_accuracy=0.0)

    html = render_agent_eval_html_report(tmp_path, run, use_llm_judge=False).read_text(encoding="utf-8")

    [fix] = json.loads((run / "what_to_fix.json").read_text())
    assert fix["kind"] == "goal not met"
    assert (
        "· per-case result: skill helps (opencode +100 pp). 1 passing with-skill attempt(s) did not meet the goal"
        in html
    )
    assert "1 issue to fix, most attempts lost first:" in html
    assert "[goal not met]: passed on the other metrics but scored goal accuracy 0.00: asked for the data." in html
    assert "Evidence: opencode/with-skill/trials/case__1/trajectory.json" in html
    assert "goal not met (0.00)</span>" in html


def test_report_says_nothing_to_fix_when_every_case_passes_and_meets_its_goal(tmp_path) -> None:
    run = _run(tmp_path, goal_accuracy=1.0)

    html = render_agent_eval_html_report(tmp_path, run, use_llm_judge=False).read_text(encoding="utf-8")

    assert json.loads((run / "what_to_fix.json").read_text()) == []
    assert "Nothing to fix: every with-skill attempt passed and met its goal" in html
    assert "goal not met (0.00)" not in html
