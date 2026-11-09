# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``skillevaluator view`` rebuilds a missing report from the feedback its run saved."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.evaluation import insights_judge, tier3_report
from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
from skillevaluator.tier3 import case_summary, commands
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

RUN_ID = "20261027_120000"
HARBOR_JOB_URL = "https://harbor.example.test/jobs/run"
SAVED_VERDICT = "per-case result: skill hurts (saved by the run)."
# What a run saves since the case findings exist: the per-case verdict leads the conclusions.
SAVED_FEEDBACK = {
    "schema_version": "1.0",
    "conclusions": [
        {"severity": "fail", "title": "Skill effect", "message": SAVED_VERDICT, "source": "deterministic"},
        {"severity": "warn", "title": "Saved conclusion", "message": "From this run"},
    ],
    "recommendations": [{"severity": "warn", "title": "Saved recommendation", "message": "Kept from the run"}],
    "suggestions": ["Saved suggestion"],
    "suggestions_v2": [],
}
# What earlier runs saved: no per-case verdict.
LEGACY_FEEDBACK = {
    **SAVED_FEEDBACK,
    "conclusions": [{"severity": "warn", "title": "Saved conclusion", "message": "From this run"}],
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


def _completed_run(tmp_path: Path, *, fixes: Any = SAVED_FIXES, **result: Any) -> tuple[Path, Path, Path]:
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
    if fixes is not None:
        _write_json(run / "what_to_fix.json", fixes)
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
    skill, results_root, run = _completed_run(
        tmp_path,
        tier3_feedback=SAVED_FEEDBACK,
        duration_seconds=4321.5,
        harbor_viewer={"job_url": HARBOR_JOB_URL},
    )
    saved_result = (run / "result.json").read_bytes()
    saved_fixes = (run / "what_to_fix.json").read_bytes()

    report = commands.view_results(skill, results_dir=results_root)

    html = report.read_text(encoding="utf-8")
    assert report == run / "report.html"
    assert llm_calls == []
    for text in (SAVED_VERDICT, "Saved conclusion", "Saved recommendation", "Saved suggestion"):
        assert text in html
    # The saved verdict leads the report; the rebuilt one is not added next to it.
    assert html.count("per-case result:") == html.count(SAVED_VERDICT)
    assert "Fresh judge conclusion" not in html
    assert "Fresh fix from a second pass" not in html
    # result.json is the report's engine result, so its runtime and Harbor links come back too.
    assert re.search(r'"runtime_seconds":\s*4321\.5', html)
    assert f'href="{HARBOR_JOB_URL}"' in html
    # Rebuilding the report writes nothing else into the run.
    assert (run / "result.json").read_bytes() == saved_result
    assert (run / "what_to_fix.json").read_bytes() == saved_fixes


def test_view_writes_rule_based_fixes_when_a_verdict_run_lost_them(tmp_path: Path, llm_calls: list[str]) -> None:
    """The saved feedback still leads the report; only the missing fixes file gets the rule-based text."""
    skill, results_root, run = _completed_run(tmp_path, fixes=None, tier3_feedback=SAVED_FEEDBACK)
    saved_result = (run / "result.json").read_bytes()

    report = commands.view_results(skill, results_dir=results_root)

    html = report.read_text(encoding="utf-8")
    assert llm_calls == []
    for text in (SAVED_VERDICT, "Saved conclusion", "Saved recommendation", "Saved suggestion"):
        assert text in html
    assert "Check whether SKILL.md steers the agent" not in html
    assert (run / "result.json").read_bytes() == saved_result
    written = json.loads((run / "what_to_fix.json").read_text(encoding="utf-8"))
    assert written[0]["kind"] == "skill hurts"
    assert "Check whether SKILL.md steers the agent" in written[0]["action"]


@pytest.mark.parametrize(
    ("fixes", "shown_fix"),
    [
        pytest.param(SAVED_FIXES, "Saved fix from the run", id="fixes_saved"),
        pytest.param(None, "Check whether SKILL.md steers the agent", id="no_fixes_saved"),
        pytest.param([{**SAVED_FIXES[0], "what": 5}], "Check whether SKILL.md steers the agent", id="unusable_fixes"),
    ],
)
def test_view_adds_the_case_verdict_to_feedback_saved_before_it(
    tmp_path: Path, llm_calls: list[str], fixes: Any, shown_fix: str
) -> None:
    skill, results_root, run = _completed_run(tmp_path, fixes=fixes, tier3_feedback=LEGACY_FEEDBACK)
    saved_result = (run / "result.json").read_bytes()

    report = commands.view_results(skill, results_dir=results_root)

    html = report.read_text(encoding="utf-8")
    assert llm_calls == []
    # The verdict needs no LLM: a run saved before it existed gets it, next to its own conclusions.
    assert "per-case result: skill hurts" in html
    assert "Saved conclusion" in html
    assert shown_fix in html
    assert "Fresh judge conclusion" not in html
    assert (run / "result.json").read_bytes() == saved_result
    # A usable saved what_to_fix.json is kept; otherwise the rule-based fixes are written.
    written = json.loads((run / "what_to_fix.json").read_text(encoding="utf-8"))
    assert written[0]["kind"] == "skill hurts"
    assert shown_fix in written[0]["action"]
    assert (written == SAVED_FIXES) is (fixes is SAVED_FIXES)


@pytest.mark.parametrize(
    "result",
    [
        pytest.param({}, id="no_saved_feedback"),
        pytest.param({"tier3_feedback": {**SAVED_FEEDBACK, "schema_version": "0.9"}}, id="other_schema"),
        pytest.param({"tier3_feedback": {"schema_version": "1.0", "conclusions": "Saved conclusion"}}, id="no_lists"),
        pytest.param(
            {"tier3_feedback": {"schema_version": "1.0", "conclusions": SAVED_FEEDBACK["conclusions"]}},
            id="partial_feedback",
        ),
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

    assert len(calls) == 1
    assert "saved_feedback" not in calls[0]
    assert calls[0].get("use_llm_judge", True)


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


def test_saved_feedback_keeps_what_the_case_pass_adds(
    tmp_path: Path, llm_calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report content the case-findings pass computes but the run does not save survives a rebuild."""
    skill, results_root, _run = _completed_run(tmp_path, tier3_feedback=SAVED_FEEDBACK)
    passes: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    apply_case_findings = tier3_report._apply_case_findings
    build_result = tier3_report.agent_eval_result_from_directory

    def case_pass(payload: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        passes.append(kwargs)
        apply_case_findings(payload, *args, **kwargs)
        payload["next_steps"] = ["Computed by the case pass"]

    def build(*args: Any, **kwargs: Any) -> Any:
        result = build_result(*args, **kwargs)
        payloads.append(result.metadata["agent_eval"])
        return result

    monkeypatch.setattr(tier3_report, "_apply_case_findings", case_pass)
    monkeypatch.setattr(tier3_report, "agent_eval_result_from_directory", build)

    commands.view_results(skill, results_dir=results_root)

    assert llm_calls == []
    assert len(passes) == 1
    assert passes[0]["use_llm"] is False
    assert passes[0]["fixes"] == SAVED_FIXES
    [payload] = payloads
    assert payload["next_steps"] == ["Computed by the case pass"]
    for field in ("conclusions", "recommendations", "suggestions", "suggestions_v2"):
        assert payload[field] == SAVED_FEEDBACK[field]
