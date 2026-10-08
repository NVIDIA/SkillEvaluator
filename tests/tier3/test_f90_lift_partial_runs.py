# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M4: one failed trial must not remove the lift interval or the Integration verdict.

Before the fix, one judge error or timeout in any arm voided both lift
intervals and the Integration verdict, and the reason blamed the
member-skills arm even when the no-plugin baseline or the plugin arm failed
(proof examples check-20 edge-03, edge-06, edge-07 and check-14 neg-03; live
run A-r2 lost Integration with 45 of 45 plugin and member-skills trials
scored). Now each interval pairs the cases both of its arms scored and says
"partial: n of m cases", Integration compares its own two arms, and the
reason names the arm that really failed.
"""

from __future__ import annotations

from pathlib import Path

from _f90_lift_fixtures import benchmark, cli, collect, markdown, metrics, payload

from skillevaluator.tier3.harbor import stats

CASES = [f"case-{index}" for index in range(1, 7)]


def _three_arms(*failed: tuple[str, str, int]) -> dict:
    """Plugin 0.85, no plugin 0.4, member skills 0.65; two attempts per case; *failed* attempts time out."""
    arms = {
        "with": {case: [metrics(0.85), metrics(0.85)] for case in CASES},
        "without": {case: [metrics(0.4), metrics(0.4)] for case in CASES},
        "sumofparts": {case: [metrics(0.65), metrics(0.65)] for case in CASES},
    }
    for variant, case, attempt in failed:
        arms[variant][case][attempt - 1] = None
    return arms


def test_failed_no_plugin_trial_keeps_integration_and_a_partial_effectiveness_interval(tmp_path: Path) -> None:
    # Proof check-20 edge-06 / live A-r2: plugin and member-skills arms complete, one baseline trial times out.
    collected = collect(tmp_path, {"claude-code": _three_arms(("without", "case-4", 2))}, n_attempts=2)
    agent = collected["agents"]["claude-code"]
    assert agent["conditions"]["without_skill"]["execution_status"] == "failed"

    effectiveness = agent["lift_uncertainty"]["effectiveness"]
    assert effectiveness["partial"] is True
    assert effectiveness["failed_arms"] == ["without_skill"]
    assert (effectiveness["n_cases"], effectiveness["expected_cases"]) == (6, 6)
    assert effectiveness["estimate"] == 0.45

    built = payload(tmp_path, "both")
    integration = built["integration"]
    assert integration["measured"] is True
    assert integration["integration_lift"] == 0.2
    assert integration["verdict"] == "real_integration"
    assert integration["reason"] is None

    card = benchmark(built)
    effectiveness_row = next(line for line in card.splitlines() if "Plugin lift (plugin vs. no plugin)" in line)
    assert "Partial run, not final: +45 points" in effectiveness_row
    assert "partial: 6 of 6 cases; did not complete: Baseline (no plugin)" in effectiveness_row
    assert "Integration (plugin vs. its own parts) | Real integration, +20 points" in card

    report = markdown(built)
    assert "**Real integration:** plugin 0.85 vs sum-of-parts 0.65 (lift +0.20" in report
    assert "produced no comparable score" not in report


def test_failed_plugin_trial_is_named_and_the_lift_is_kept_as_a_point_estimate(tmp_path: Path) -> None:
    # Proof check-20 edge-07: one plugin-arm trial times out.
    collect(tmp_path, {"codex": _three_arms(("with", "case-1", 1))}, n_attempts=2)

    integration = payload(tmp_path, "both")["integration"]

    assert integration["verdict"] == "inconclusive"
    assert integration["measured"] is True
    assert integration["point_verdict"] == "real_integration"
    assert integration["integration_lift"] == 0.2
    assert "with-plugin arm" in integration["reason"]
    assert "case-1 (with-plugin arm) 1/2" in integration["reason"]
    assert "member-skills (sum-of-parts) arm produced no comparable score" not in integration["reason"]
    assert integration["lift_uncertainty"]["failed_arms"] == ["with_skill"]


def test_missing_member_skills_case_names_the_case_and_counts_pairs(tmp_path: Path) -> None:
    # Proof check-20 edge-01: both member-skills attempts of one case time out.
    collect(
        tmp_path, {"claude-code": _three_arms(("sumofparts", "case-6", 1), ("sumofparts", "case-6", 2))}, n_attempts=2
    )

    built = payload(tmp_path, "both")
    integration = built["integration"]

    assert integration["verdict"] == "inconclusive"
    assert integration["integration_lift"] == 0.2
    assert "missing case(s): case-6" in integration["reason"]
    assert "5 of 6 cases" in integration["reason"]
    assert integration["lift_uncertainty"]["n_cases"] == 5
    output = cli(built)
    assert "partial: 5 of 6 cases" in output


def test_member_skills_job_that_never_ran_is_named(tmp_path: Path) -> None:
    # Proof check-20 edge-03: the sum-of-parts Harbor job never wrote its folder.
    arms = _three_arms()
    del arms["sumofparts"]
    collect(tmp_path, {"claude-code": arms}, n_attempts=2)
    # The runner asked for the arm; the collector looks for its job and finds none.
    from skillevaluator.tier3.harbor.collector import collect_harbor_results

    collect_harbor_results(
        skill_name="demo",
        agents=["claude-code"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        sum_of_parts_arm=True,
        n_attempts=2,
        expected_cases=len(CASES),
        expected_case_ids=CASES,
        expected_trials=2 * len(CASES),
    )

    integration = payload(tmp_path, "both")["integration"]

    assert integration["measured"] is False
    assert integration["reason"].startswith("The member-skills (sum-of-parts) arm produced no usable scored trial")
    assert "job directory was not created" in integration["reason"]


def test_statistics_pair_the_scored_cases_of_a_failed_arm() -> None:
    def arm(status: str, cases: dict[str, float]) -> stats.ArmObservations:
        trials = [
            stats.TrialObservation(case_id=case, reward={"metric_set": "skill-evaluator-default-v2", **metrics(value)})
            for case, value in cases.items()
        ]
        return stats.ArmObservations(execution_status=status, trials=tuple(trials))

    with_arm = arm("succeeded", dict.fromkeys(CASES, 0.8))
    without_arm = arm("failed", dict.fromkeys(CASES[:4], 0.3))

    block = stats.build_agent_statistics(
        {"with_skill": with_arm, "without_skill": without_arm},
        expected_case_ids=CASES,
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=False,
    )

    effectiveness = block["lift_uncertainty"]["effectiveness"]
    assert effectiveness["estimate"] == 0.5
    assert (effectiveness["n_cases"], effectiveness["expected_cases"], effectiveness["partial"]) == (4, 6, True)
    assert effectiveness["failed_arms"] == ["without_skill"]
    # Per-arm cost and reliability blocks stay limited to completed arms.
    assert set(block["reliability"]) <= {"with_skill"}


def test_a_partial_interval_never_becomes_a_final_headline() -> None:
    from skillevaluator.evaluation.tier3_report import _build_agent

    scores = dict.fromkeys(("security", "skill_execution", "skill_efficiency", "accuracy"), 0.8)
    scores.update(goal_accuracy=0.8, behavior_check=0.8)
    info = {
        "with_skill": scores,
        "without_skill": dict.fromkeys(scores, 0.4),
        "conditions": {
            "with_skill": {"execution_status": "succeeded"},
            "without_skill": {"execution_status": "succeeded"},
        },
        "execution_status": "succeeded",
        "lift_uncertainty": {
            "effectiveness": {
                "estimate": 0.4,
                "ci_low": 0.3,
                "ci_high": 0.5,
                "n_cases": 5,
                "expected_cases": 6,
                "partial": True,
                "treatment_score": 0.8,
                "control_score": 0.4,
            },
            "integration": None,
        },
    }

    agent = _build_agent("codex", info, list(scores), list(scores), None)

    assert agent["lift"] is None
    assert agent["lift_basis"]["effectiveness"]["partial"] is True
    assert agent["lift_uncertainty"]["effectiveness"]["partial"] is True
