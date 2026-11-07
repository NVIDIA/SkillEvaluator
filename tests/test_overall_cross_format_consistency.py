"""One run publishes one Overall score and one Skill Lift in every output format."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.reporting.benchmark import BenchmarkReporter
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS, DEFAULT_SCORE_POLICY
from skillevaluator.tier3.result_display import render_result

SKILL_METRICS = ("skill_execution", "skill_efficiency")
WITH_SKILL = {**dict.fromkeys(DEFAULT_METRICS, 0.5), "goal_accuracy": 1.0, "behavior_check": 1.0}
BASELINE = {**dict.fromkeys(DEFAULT_METRICS, 0.85), "goal_accuracy": 0.0, "behavior_check": 0.0}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _job(job_dir: Path, scores: dict[str, float], cases: int, not_applicable: tuple[str, ...] = ()) -> None:
    trials = []
    for index in range(1, cases + 1):
        trial = f"case-{index}__trial"
        trials.append(trial)
        reward = dict(scores)
        details: dict[str, dict] = {}
        for metric in not_applicable:
            reward[metric] = None
            details[metric] = {"score": None, "status": "not_applicable", "reason": "no skill under test"}
        if "skill_execution" not in not_applicable:
            details["skill_execution"] = {"activation": {"passed": True, "score": 1.0}}
        reward.update(
            entry_id=f"case-{index}",
            metric_set=DEFAULT_METRIC_SET,
            score_policy=DEFAULT_SCORE_POLICY,
            details=details,
        )
        _write(job_dir / trial / "verifier" / "reward.json", reward)
    stats = {
        "n_completed_trials": cases,
        "n_errored_trials": 0,
        "n_running_trials": 0,
        "n_pending_trials": 0,
        "n_cancelled_trials": 0,
        "n_retries": 0,
        "evals": {
            "agent__model___tasks": {
                "n_trials": cases,
                "n_errors": 0,
                "reward_stats": {"overall": {"0.5": trials}},
            }
        },
    }
    _write(job_dir / "result.json", {"n_total_trials": cases, "stats": stats})


def _publish(tmp_path: Path, baseline_not_applicable: tuple[str, ...]) -> dict:
    skill = tmp_path / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill.\n---\nBody.\n", encoding="utf-8")
    jobs = tmp_path / "jobs"
    _job(jobs / "demo-codex-with", WITH_SKILL, 2)
    _job(jobs / "demo-codex-without", BASELINE, 2, baseline_not_applicable)
    run = tmp_path / "results" / "20261106_120000"
    collected = collect_harbor_results(
        "demo", ["codex"], run, jobs, expected_case_ids=["case-1", "case-2"], expected_trials=2
    )
    result = agent_eval_result_from_directory(skill, run, use_llm_judge=False)
    return {
        "collected": collected,
        "summaries": {
            arm: json.loads((run / "codex" / arm / "summary.json").read_text(encoding="utf-8"))
            for arm in ("with-skill", "without-skill")
        },
        "lift_json": json.loads((run / "codex" / "lift.json").read_text(encoding="utf-8")),
        "agent_eval": result.metadata["agent_eval"],
        "json": json.loads(JSONReporter().render_all([result])),
        "html": HTMLReporter(include_timestamp=False).render_all([result]),
        "markdown": MarkdownReporter().render_all([result]),
        "benchmark": BenchmarkReporter(skill_name="demo", report_scope="tier3").render_all([result]),
        "terminal": render_result(collected),
    }


def _json_values(payload: object, key: str) -> list[object]:
    found: list[object] = []
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name == key:
                found.append(value)
            found.extend(_json_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_json_values(value, key))
    return found


@pytest.mark.parametrize(
    ("baseline_not_applicable", "with_overall", "baseline_overall", "lift_pair", "lift", "benchmark_row"),
    [
        # A six-metric mean would give 0.667 vs 0.567 (+0.10); the five-dimension
        # policy reverses the lift and drops the with-skill arm below 0.65.
        ((), 0.6, 0.68, (0.6, 0.68), -0.08, "68.0% → 60.0% (-8.0 points)"),
        # The baseline has no skill under test, so Discoverability and Efficiency
        # drop out of both arms for lift but stay in the with-skill arm score.
        (SKILL_METRICS, 0.6, 0.5667, (0.6667, 0.5667), 0.1, "56.7% → 66.7% (+10.0 points)"),
    ],
    ids=["five-dimension-reverses-six-metric-lift", "baseline-skill-metrics-not-applicable"],
)
def test_every_output_publishes_one_overall_and_lift(
    tmp_path: Path,
    baseline_not_applicable: tuple[str, ...],
    with_overall: float,
    baseline_overall: float,
    lift_pair: tuple[float, float],
    lift: float,
    benchmark_row: str,
) -> None:
    out = _publish(tmp_path, baseline_not_applicable)

    for summary in out["summaries"].values():
        assert summary["score_policy"] == DEFAULT_SCORE_POLICY
    assert out["summaries"]["with-skill"]["overall_score"] == pytest.approx(with_overall, abs=1e-4)
    assert out["summaries"]["without-skill"]["overall_score"] == pytest.approx(baseline_overall, abs=1e-4)

    recorded = out["lift_json"]["overall"]
    assert (recorded["with_skill"], recorded["without_skill"]) == pytest.approx(lift_pair, abs=1e-4)
    assert recorded["delta"] == pytest.approx(lift, abs=1e-4)
    assert out["collected"]["agents"]["codex"]["lift"]["overall"] == recorded

    agent_eval = out["agent_eval"]
    agent = agent_eval["agents"]["codex"]
    assert agent_eval["score_policy"] == DEFAULT_SCORE_POLICY
    assert (agent["with_skill"], agent["baseline"]) == pytest.approx((with_overall, baseline_overall), abs=1e-4)
    assert agent["lift"] == agent_eval["overall_lift"] == agent_eval["composite_lift"] == pytest.approx(lift)
    assert (agent["lift_overall"]["with_skill"], agent["lift_overall"]["baseline"]) == pytest.approx(lift_pair)

    json_lifts = _json_values(out["json"], "overall_lift")
    assert json_lifts
    assert json_lifts == pytest.approx([lift] * len(json_lifts))
    assert set(_json_values(out["json"], "score_policy")) == {DEFAULT_SCORE_POLICY}
    assert f"composite lift = {lift:+.2f}" in out["markdown"]
    assert f"| Overall | {benchmark_row} |" in out["benchmark"]
    assert re.search(rf"OVERALL SKILL LIFT\s+{re.escape(f'{lift:+.2f}')}", out["terminal"])
    assert re.search(
        rf"quality score \(0-1\)\s+{lift_pair[0]:.2f}\s+{lift_pair[1]:.2f}\s+{re.escape(f'{lift:+.2f}')}",
        out["terminal"],
    )
    assert f"Skill Lift {lift:+.2f}" in re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", out["html"]))
    assert ("compares only the dimensions both arms scored" in out["benchmark"]) is bool(baseline_not_applicable)
