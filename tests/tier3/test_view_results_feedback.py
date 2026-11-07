# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tier3 view-results`` rebuilds a missing report from the feedback its run saved."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.evaluation import insights_judge
from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.tier3 import case_summary, commands
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

RUN_ID = "20261027_120000"
SAVED_FEEDBACK = {
    "schema_version": "1.0",
    "conclusions": [{"severity": "warn", "title": "Saved conclusion", "message": "From this run"}],
    "recommendations": [{"severity": "warn", "title": "Saved recommendation", "message": "Kept from the run"}],
    "suggestions": ["Saved suggestion"],
    "suggestions_v2": [],
}
SAVED_FIXES = [
    {
        "case": "case-1",
        "scope": "opencode",
        "kind": "skill hurts",
        "what": "passes 1/1 without the skill but 0/1 with it",
        "attempts_lost": 1,
        "action": "Saved fix from the run",
    }
]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_arm(run: Path, arm: str, overall: float) -> None:
    scores = dict.fromkeys(DEFAULT_METRICS, overall)
    _write_json(
        run / "opencode" / arm / "summary.json",
        {
            "scores": scores,
            "metrics": list(DEFAULT_METRICS),
            "num_trials": 1,
            "expected_attempts": 1,
            "scored_attempts": 1,
            "execution_status": "succeeded",
        },
    )
    _write_json(run / "opencode" / arm / "trials" / "case-1__1" / "reward.json", {"entry_id": "case-1", **scores})


def _completed_run(tmp_path: Path, **result: Any) -> tuple[Path, Path, Path]:
    """A completed run whose case passes without the skill and fails with it; no report.html."""
    skill = tmp_path / "demo-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo.\n---\n# Demo\n", encoding="utf-8")
    results_root = tmp_path / "results"
    run = results_root / skill.name / RUN_ID
    run.mkdir(parents=True)
    _write_json(run / "run_config.json", {})
    _write_arm(run, "with-skill", 0.4)
    _write_arm(run, "without-skill", 0.6)
    cases = {
        "with_skill": {"cases": {"case-1": {"attempts": [{"passed": False, "trial": "case-1__1", "score": 0.4}]}}},
        "without_skill": {"cases": {"case-1": {"attempts": [{"passed": True, "trial": "case-1__1", "score": 0.6}]}}},
    }
    _write_json(
        run / "result.json",
        {"run_id": RUN_ID, "skill_name": skill.name, "agents": {"opencode": {"pass_at_k": cases}}, **result},
    )
    _write_json(run / "what_to_fix.json", SAVED_FIXES)
    return skill, results_root, run


@pytest.fixture
def llm_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every judge and what-to-fix LLM call; each one answers with fresh text."""
    calls: list[str] = []

    def judge(*_args: object, **_kwargs: object) -> dict[str, Any]:
        calls.append("insights")
        return {"conclusions": [{"title": "Fresh judge conclusion", "message": "Second pass"}], "recommendations": []}

    def actions(issues: list[dict[str, Any]], *_args: object) -> list[str]:
        calls.append("what_to_fix")
        return ["Fresh fix from a second pass"] * len(issues)

    monkeypatch.setattr(insights_judge, "build_insights", judge)
    monkeypatch.setattr(case_summary, "_llm_actions", actions)
    monkeypatch.setattr(commands.webbrowser, "open", lambda _uri: True)
    return calls


def test_view_reuses_saved_feedback_when_report_is_missing(tmp_path: Path, llm_calls: list[str]) -> None:
    skill, results_root, run = _completed_run(tmp_path, tier3_feedback=SAVED_FEEDBACK)
    saved_result = (run / "result.json").read_bytes()
    saved_fixes = (run / "what_to_fix.json").read_bytes()

    report = commands.view_results(skill, results_dir=results_root)

    html = report.read_text(encoding="utf-8")
    assert report == run / "report.html"
    assert llm_calls == []
    for text in ("Saved conclusion", "Saved recommendation", "Saved suggestion"):
        assert text in html
    assert "Fresh judge conclusion" not in html
    assert "Fresh fix from a second pass" not in html
    # Rebuilding the report writes nothing else into the run.
    assert (run / "result.json").read_bytes() == saved_result
    assert (run / "what_to_fix.json").read_bytes() == saved_fixes


@pytest.mark.parametrize(
    "result",
    [
        pytest.param({}, id="no_saved_feedback"),
        pytest.param({"tier3_feedback": {**SAVED_FEEDBACK, "schema_version": "0.9"}}, id="other_schema"),
        pytest.param({"tier3_feedback": {"schema_version": "1.0", "conclusions": "Saved conclusion"}}, id="no_lists"),
    ],
)
def test_view_rebuilds_feedback_when_none_is_saved_for_the_run(
    tmp_path: Path, llm_calls: list[str], result: dict[str, Any]
) -> None:
    skill, results_root, _run = _completed_run(tmp_path, **result)

    report = commands.view_results(skill, results_dir=results_root)

    html = report.read_text(encoding="utf-8")
    assert sorted(llm_calls) == ["insights", "what_to_fix"]
    assert "Fresh judge conclusion" in html
    assert "Saved conclusion" not in html


def test_view_ignores_feedback_saved_for_another_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill, results_root, run = _completed_run(tmp_path, tier3_feedback=SAVED_FEEDBACK)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    _write_json(run / "result.json", {**result, "run_id": "20261026_120000"})
    calls: list[dict[str, Any]] = []

    def render(_skill_path: Path, target: Path, **kwargs: Any) -> Path:
        calls.append(kwargs)
        report = target / "report.html"
        report.write_text("<html>rebuilt</html>", encoding="utf-8")
        return report

    # Run selection already rejects such a run; the feedback check must too.
    monkeypatch.setattr(commands, "resolve_latest_results", lambda *_args: run)
    monkeypatch.setattr(commands, "render_agent_eval_html_report", render)
    monkeypatch.setattr(commands.webbrowser, "open", lambda _uri: True)

    commands.view_results(skill, results_dir=results_root)

    assert calls == [{}]


def test_view_opens_an_existing_report_without_rebuilding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill, results_root, run = _completed_run(tmp_path, tier3_feedback=SAVED_FEEDBACK)
    (run / "report.html").write_text("<html>original</html>", encoding="utf-8")
    opened: list[str] = []

    def render(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("an existing report must not be rebuilt")

    monkeypatch.setattr(commands, "render_agent_eval_html_report", render)
    monkeypatch.setattr(commands.webbrowser, "open", opened.append)

    report = commands.view_results(skill, results_dir=results_root)

    assert report == run / "report.html"
    assert opened == [report.as_uri()]
    assert report.read_text(encoding="utf-8") == "<html>original</html>"


def test_render_with_saved_feedback_skips_the_judge(tmp_path: Path, llm_calls: list[str]) -> None:
    skill, _results_root, run = _completed_run(tmp_path)
    engine_result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    saved = {name: value for name, value in SAVED_FEEDBACK.items() if name != "schema_version"}

    report = render_agent_eval_html_report(skill, run, engine_result=engine_result, saved_feedback=saved)

    html = report.read_text(encoding="utf-8")
    assert llm_calls == []
    assert "Saved conclusion" in html
    assert "Fresh judge conclusion" not in html
    assert engine_result["tier3_feedback"] == SAVED_FEEDBACK
