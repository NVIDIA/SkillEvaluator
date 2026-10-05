# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-only Tier 3 statistics: lift CIs, pass^k, cost, efficiency, completeness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.cli import _plugin_lift_fallback_metadata
from skillevaluator.constants import LIFT_BOOTSTRAP_RESAMPLES, LIFT_BOOTSTRAP_SEED
from skillevaluator.evaluation.tier3_report import (
    _build_integration_report,
    _plugin_lift_modes,
    build_agent_eval_payload,
)
from skillevaluator.tier3.harbor import stats
from skillevaluator.tier3.harbor.collector import _trial_usage, collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS, LEGACY_METRIC_SET
from skillevaluator.tier3.harbor.report_data import load_agent_data
from skillevaluator.tier3.harbor.stats import ArmObservations, TrialObservation

CASES = [f"case-{index}" for index in range(1, 7)]
PLUGIN_CONFIG = {
    "eval_target": {"kind": "plugin"},
    "skill_workspace": {
        "staged_skills": ["loader", "summarizer"],
        "baseline_includes_workspace_skills": False,
        "sum_of_parts_arm": True,
    },
}


def _reward(case_id: str, score: float | None) -> dict[str, Any]:
    return {"entry_id": case_id, "metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, score)}


def _trial(case_id: str, score: float | None, **usage: float) -> TrialObservation:
    return TrialObservation(case_id=case_id, reward=_reward(case_id, score), usage=usage)


def _arm(trials: list[TrialObservation], *, status: str = "succeeded", **kwargs: Any) -> ArmObservations:
    return ArmObservations(execution_status=status, trials=tuple(trials), **kwargs)


# ---------------------------------------------------------------------------
# Paired, case-clustered bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    treatment = {case: 0.5 + 0.05 * index for index, case in enumerate(CASES)}
    control = dict.fromkeys(CASES, 0.5)

    first = stats.paired_case_bootstrap(treatment, control)
    second = stats.paired_case_bootstrap(treatment, control)

    assert first == second
    assert first["seed"] == LIFT_BOOTSTRAP_SEED
    assert first["resamples"] == LIFT_BOOTSTRAP_RESAMPLES
    assert first["method"] == "paired_case_bootstrap"
    assert first["confidence"] == 0.95
    assert first["n_cases"] == 6
    assert first["estimate"] == 0.125
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]
    assert first["ci_includes_zero"] is False
    other_seed = stats.paired_case_bootstrap(treatment, control, seed=LIFT_BOOTSTRAP_SEED + 1)
    assert other_seed["estimate"] == first["estimate"]


def test_bootstrap_known_datasets_give_expected_interval_shape() -> None:
    constant = stats.paired_case_bootstrap(dict.fromkeys(CASES, 0.9), dict.fromkeys(CASES, 0.8))
    assert (constant["estimate"], constant["ci_low"], constant["ci_high"]) == (0.1, 0.1, 0.1)
    assert constant["precision"] == "adequate"
    assert constant["ci_includes_zero"] is False

    # Alternating +/-0.2 deltas centre on zero; resampled means stay inside the
    # observed range and straddle zero roughly symmetrically.
    alternating = {case: (0.7 if index % 2 else 0.3) for index, case in enumerate(CASES)}
    symmetric = stats.paired_case_bootstrap(alternating, dict.fromkeys(CASES, 0.5))
    assert symmetric["estimate"] == 0.0
    assert -0.2 <= symmetric["ci_low"] < 0.0 < symmetric["ci_high"] <= 0.2
    assert abs(symmetric["ci_low"] + symmetric["ci_high"]) <= 0.07
    assert symmetric["ci_includes_zero"] is True
    assert symmetric["precision"] == "low"


def test_bootstrap_pairs_by_case_and_nests_attempts_within_cases() -> None:
    # case-1 has three attempts in one arm; its case mean (0.6) is what pairs.
    treatment = stats.case_mean_scores(
        [_trial("case-1", 0.3), _trial("case-1", 0.6), _trial("case-1", 0.9), _trial("case-2", 1.0)]
    )
    assert treatment == pytest.approx({"case-1": 0.6, "case-2": 1.0})
    control = {"case-1": 0.5, "case-3": 0.0}

    result = stats.paired_case_bootstrap(treatment, control)

    assert result["n_cases"] == 1
    assert result["estimate"] == 0.1
    assert result["precision"] == "insufficient"


def test_bootstrap_without_pairs_reports_insufficient_nulls() -> None:
    result = stats.paired_case_bootstrap({"a": 1.0}, {"b": 0.0})
    assert result["n_cases"] == 0
    assert result["estimate"] is None
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["ci_includes_zero"] is None
    assert result["precision"] == "insufficient"


def test_trial_quality_score_matches_dimension_mean_and_skips_na_metrics() -> None:
    reward = {
        "metric_set": DEFAULT_METRIC_SET,
        "security": 1.0,
        "accuracy": 0.5,
        "skill_execution": 1.0,
        "goal_accuracy": 0.0,
        "behavior_check": 1.0,
        "skill_efficiency": 0.5,
    }
    # Dimensions: security 1.0, correctness 0.5, discoverability 1.0,
    # effectiveness 0.5, efficiency 0.5.
    assert stats.trial_quality_score(reward) == pytest.approx(0.7)

    reward["security"] = "N/A"
    assert stats.trial_quality_score(reward) == pytest.approx(0.625)
    assert stats.trial_quality_score(_reward("case-1", None)) is None
    assert stats.trial_quality_score({"metric_set": "custom-only", "overall": 0.4}) == 0.4


def test_trial_quality_score_scores_legacy_security_from_behavior_check() -> None:
    reward = {
        "metric_set": LEGACY_METRIC_SET,
        "accuracy": 0.5,
        "skill_execution": 1.0,
        "goal_accuracy": 0.0,
        "behavior_check": 1.0,
        "skill_efficiency": 0.5,
    }
    # Dimensions: security 1.0 (behavior_check), correctness 0.5, discoverability 1.0,
    # effectiveness 0.5, efficiency 0.5.
    assert stats.trial_quality_score(reward) == pytest.approx(0.7)

    # A legacy reward's stray security value is not part of its metric set.
    reward["security"] = 0.0
    assert stats.trial_quality_score(reward) == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("n_cases", "low", "high", "expected"),
    [
        (4, 0.0, 0.05, "insufficient"),
        (5, 0.0, 0.2, "adequate"),
        (5, 0.0, 0.2001, "low"),
        (50, -0.3, 0.3, "low"),
        (5, None, None, "insufficient"),
    ],
)
def test_precision_thresholds(n_cases: int, low: float | None, high: float | None, expected: str) -> None:
    assert stats.lift_precision(n_cases, low, high) == expected


# ---------------------------------------------------------------------------
# Reliability, cost and token efficiency
# ---------------------------------------------------------------------------


def _pass_summary(cases: dict[str, list[bool]], *, k: int) -> dict[str, Any]:
    passed = sum(any(flags) for flags in cases.values())
    return {
        "k": k,
        "total_cases": len(cases),
        "passed_cases": passed,
        "rate": round(passed / len(cases), 4),
        "cases": {
            case: {
                "passed": any(flags),
                "attempts": [{"attempt": i + 1, "passed": flag} for i, flag in enumerate(flags)],
            }
            for case, flags in cases.items()
        },
    }


def test_pass_hat_k_requires_every_attempt_to_pass() -> None:
    summary = _pass_summary({"a": [True, True, True], "b": [True, False, True], "c": [False, False, False]}, k=3)
    summary["cases"]["extra"] = {"extra_case": True, "attempts": [{"passed": True}] * 3}

    reliability = stats.reliability_summary(summary, stop_on_pass=False)

    assert reliability == {"pass_at_k": 0.6667, "pass_hat_k": 0.3333, "k": 3, "n_cases": 3}


def test_pass_hat_k_counts_missing_or_unscored_attempts_as_failures() -> None:
    summary = _pass_summary({"a": [True, True], "b": [True]}, k=2)
    assert stats.reliability_summary(summary, stop_on_pass=False)["pass_hat_k"] == 0.5


def test_pass_hat_k_is_unobservable_when_stop_on_pass_truncates_attempts() -> None:
    summary = _pass_summary({"a": [True], "b": [False, True]}, k=2)
    assert stats.reliability_summary(summary, stop_on_pass=True)["pass_hat_k"] is None
    single = _pass_summary({"a": [True], "b": [False]}, k=1)
    assert stats.reliability_summary(single, stop_on_pass=True)["pass_hat_k"] == 0.5
    assert stats.reliability_summary({}, stop_on_pass=False) is None


def test_cost_per_success_with_reported_usd() -> None:
    trials = [
        _trial("a", 0.9, prompt_tokens=12_000, cached_tokens=2_000, completion_tokens=1_000, cost_usd=0.03),
        _trial("a", 0.2, prompt_tokens=8_000, cached_tokens=0, completion_tokens=1_000, cost_usd=0.01),
        _trial("b", 0.9, prompt_tokens=4_000, completion_tokens=500, cost_usd=0.02),
    ]

    cost = stats.arm_cost(trials, successes=2)

    assert cost == {
        "tokens_per_success": 12_250.0,
        "usd_per_success": 0.03,
        "total_tokens": 24_500,
        "total_usd": 0.06,
        "successes": 2,
    }


def test_cost_without_usd_never_invents_a_price() -> None:
    trials = [
        _trial("a", 0.9, prompt_tokens=1_000, completion_tokens=100, cost_usd=0.01),
        _trial("b", 0.9, prompt_tokens=1_000, completion_tokens=100),
    ]
    cost = stats.arm_cost(trials, successes=2)
    assert cost["total_tokens"] == 2_200
    assert cost["tokens_per_success"] == 1_100.0
    assert cost["total_usd"] is None
    assert cost["usd_per_success"] is None

    no_success = stats.arm_cost(trials, successes=0)
    assert no_success["tokens_per_success"] is None and no_success["usd_per_success"] is None

    partial_tokens = stats.arm_cost([*trials, _trial("c", 0.9)], successes=3)
    assert partial_tokens["total_tokens"] is None and partial_tokens["tokens_per_success"] is None


def test_token_efficiency_uses_uncached_prompt_plus_completion() -> None:
    half_life = [_trial("a", 0.9, prompt_tokens=250_000, cached_tokens=100_000, completion_tokens=50_000)]
    assert stats.arm_token_efficiency(half_life) == 0.5
    mixed = [*half_life, _trial("b", 0.9, prompt_tokens=0, completion_tokens=0)]
    assert stats.arm_token_efficiency(mixed) is None
    assert stats.arm_token_efficiency([_trial("a", 0.9, prompt_tokens=10)]) is None
    assert stats.token_efficiency_score(600_000) == 0.25


# ---------------------------------------------------------------------------
# Integration completeness
# ---------------------------------------------------------------------------


def _completeness(
    with_trials: list[TrialObservation],
    sop_trials: list[TrialObservation],
    *,
    n_attempts: int = 2,
    stop_on_pass: bool = False,
    sop_status: str = "succeeded",
    sop_job_failure: str = "",
) -> dict[str, Any]:
    return stats.integration_completeness(
        _arm(with_trials),
        _arm(sop_trials, status=sop_status, job_failure=sop_job_failure),
        expected_case_ids=["a", "b"],
        n_attempts=n_attempts,
        stop_on_pass=stop_on_pass,
        pass_threshold=0.5,
    )


def _full(case_ids: list[str], score: float = 0.9, attempts: int = 2) -> list[TrialObservation]:
    return [_trial(case_id, score) for case_id in case_ids for _ in range(attempts)]


def test_completeness_is_complete_when_every_case_has_every_attempt() -> None:
    result = _completeness(_full(["a", "b"]), _full(["a", "b"]))
    assert result == {"complete": True, "missing_cases": [], "failed_arms": [], "attempt_shortfall": []}


def test_completeness_flags_missing_cases_and_unscoreable_attempts() -> None:
    sop = [*_full(["a"]), _trial("b", None), _trial("b", None)]
    result = _completeness(_full(["a", "b"]), sop)
    assert result["complete"] is False
    assert result["missing_cases"] == ["b"]
    assert {"case": "b", "arm": "sum_of_parts", "expected": 2, "observed": 0} in result["attempt_shortfall"]


def test_completeness_flags_failed_sum_of_parts_job_even_with_residual_rewards() -> None:
    result = _completeness(_full(["a", "b"]), _full(["a", "b"]), sop_job_failure="status=failed")
    assert result["complete"] is False
    assert result["failed_arms"] == ["sum_of_parts"]
    failed_status = _completeness(_full(["a", "b"]), _full(["a", "b"]), sop_status="failed")
    assert failed_status["failed_arms"] == ["sum_of_parts"]


def test_completeness_flags_attempt_shortfall() -> None:
    result = _completeness(_full(["a", "b"]), [*_full(["a"]), _trial("b", 0.9)])
    assert result["complete"] is False
    assert result["missing_cases"] == []
    assert result["attempt_shortfall"] == [{"case": "b", "arm": "sum_of_parts", "expected": 2, "observed": 1}]


def test_completeness_accepts_legitimate_stop_on_pass_truncation() -> None:
    with_trials = [_trial("a", 0.9), _trial("b", 0.9)]
    sop_trials = [_trial("a", 0.1), _trial("a", 0.9), _trial("b", 0.1), _trial("b", 0.1), _trial("b", 0.1)]
    assert _completeness(with_trials, sop_trials, n_attempts=3, stop_on_pass=True)["complete"] is True
    short = _completeness(with_trials, sop_trials[:4], n_attempts=3, stop_on_pass=True)
    assert short["attempt_shortfall"] == [{"case": "b", "arm": "sum_of_parts", "expected": 3, "observed": 2}]


def test_completeness_without_a_sum_of_parts_arm_is_not_complete() -> None:
    result = stats.integration_completeness(
        _arm(_full(["a"])), None, expected_case_ids=["a"], n_attempts=1, stop_on_pass=False, pass_threshold=0.5
    )
    assert result["complete"] is False


# ---------------------------------------------------------------------------
# Measured always-on context
# ---------------------------------------------------------------------------


def test_context_cost_is_the_paired_first_turn_prompt_delta() -> None:
    with_arm = _arm(
        [_trial("a", 0.9, first_turn_prompt_tokens=1_500), _trial("b", 0.9, first_turn_prompt_tokens=1_700)]
    )
    without_arm = _arm(
        [_trial("a", 0.5, first_turn_prompt_tokens=1_000), _trial("b", 0.5, first_turn_prompt_tokens=1_000)]
    )

    result = stats.context_cost_measured(with_arm, without_arm)

    assert result == {
        "method": "paired_first_turn_prompt_tokens",
        "delta_tokens_mean": 600.0,
        "n_pairs": 2,
        "status": "measured",
        "reason": None,
    }


def test_context_cost_is_unavailable_without_per_step_prompt_tokens() -> None:
    with_arm = _arm([_trial("a", 0.9, first_turn_prompt_tokens=1_500), _trial("b", 0.9)])
    without_arm = _arm([_trial("a", 0.5, first_turn_prompt_tokens=1_000), _trial("b", 0.5)])
    result = stats.context_cost_measured(with_arm, without_arm)
    assert result["status"] == "unavailable"
    assert result["delta_tokens_mean"] is None
    assert "per-step prompt token counts" in result["reason"]

    skipped = stats.context_cost_measured(with_arm, _arm([], status="skipped"))
    assert skipped["status"] == "unavailable"
    assert "baseline" in skipped["reason"]


# ---------------------------------------------------------------------------
# Collector wiring (real Harbor-shaped job directories)
# ---------------------------------------------------------------------------


def _write_job(
    jobs_dir: Path,
    variant: str,
    scores: dict[str, float],
    *,
    attempts: int = 2,
    first_turn_prompt: int = 1_000,
    per_step_metrics: bool = True,
    trajectory_cost: bool = True,
) -> None:
    job_dir = jobs_dir / f"demo-opencode-{variant}"
    trial_names: list[str] = []
    for case_id, score in scores.items():
        for attempt in range(1, attempts + 1):
            trial_name = f"{case_id}__attempt{attempt:03d}"
            trial_names.append(trial_name)
            trial_dir = job_dir / trial_name
            (trial_dir / "verifier").mkdir(parents=True)
            (trial_dir / "agent").mkdir()
            (trial_dir / "verifier" / "reward.json").write_text(json.dumps(_reward(case_id, score)), encoding="utf-8")
            agent_step: dict[str, Any] = {"step_id": 2, "source": "agent", "message": "done"}
            if per_step_metrics:
                agent_step["metrics"] = {"prompt_tokens": first_turn_prompt, "completion_tokens": 40}
            final_metrics: dict[str, Any] = {
                "total_prompt_tokens": 20_000,
                "total_cached_tokens": 5_000,
                "total_completion_tokens": 1_000,
            }
            if trajectory_cost:
                final_metrics["total_cost_usd"] = 0.01
            trajectory = {
                "schema_version": "ATIF-v1.4",
                "steps": [{"step_id": 1, "source": "user", "message": "task"}, agent_step],
                "final_metrics": final_metrics,
            }
            (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
            (trial_dir / "result.json").write_text(
                json.dumps({"trial_name": trial_name, "task_name": case_id}), encoding="utf-8"
            )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(trial_names),
                "stats": {
                    "n_trials": len(trial_names),
                    "n_errors": 0,
                    "evals": {
                        "agent__model": {
                            "n_trials": len(trial_names),
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.1": trial_names}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _collect_three_arms(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    return collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        sum_of_parts_arm=True,
        n_attempts=2,
        expected_cases=len(CASES),
        expected_case_ids=CASES,
        expected_trials=2 * len(CASES),
        **kwargs,
    )


def test_collector_emits_c4_statistics_for_every_arm(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    _write_job(jobs, "with", dict.fromkeys(CASES, 0.9), first_turn_prompt=1_500)
    _write_job(jobs, "without", dict.fromkeys(CASES, 0.4), trajectory_cost=False)
    _write_job(jobs, "sumofparts", dict.fromkeys(CASES, 0.7))

    result = _collect_three_arms(tmp_path)
    agent = result["agents"]["opencode"]

    effectiveness = agent["lift_uncertainty"]["effectiveness"]
    assert effectiveness["n_cases"] == 6
    assert (effectiveness["estimate"], effectiveness["ci_low"], effectiveness["ci_high"]) == (0.5, 0.5, 0.5)
    assert effectiveness["precision"] == "adequate"
    assert effectiveness["ci_includes_zero"] is False
    assert agent["lift_uncertainty"]["integration"]["estimate"] == 0.2

    assert agent["reliability"]["with_skill"] == {"pass_at_k": 1.0, "pass_hat_k": 1.0, "k": 2, "n_cases": 6}
    assert agent["reliability"]["without_skill"]["pass_hat_k"] == 0.0
    assert set(agent["reliability"]) == {"with_skill", "without_skill", "sum_of_parts"}

    # 12 attempts x (20000 prompt - 5000 cached + 1000 completion), 6 passing cases.
    assert agent["cost"]["with_skill"] == {
        "tokens_per_success": 32_000.0,
        "usd_per_success": 0.02,
        "total_tokens": 192_000,
        "total_usd": 0.12,
        "successes": 6,
    }
    assert agent["cost"]["without_skill"]["successes"] == 0
    assert agent["cost"]["without_skill"]["total_usd"] is None
    assert agent["token_efficiency"]["with_skill"] == round(1 / (1 + 16_000 / 200_000), 4)

    assert agent["context_cost_measured"]["status"] == "measured"
    assert agent["context_cost_measured"]["delta_tokens_mean"] == 500.0
    assert agent["context_cost_measured"]["n_pairs"] == 6

    completeness = agent["integration_completeness"]
    assert completeness["complete"] is True
    assert completeness["missing_cases"] == [] and completeness["attempt_shortfall"] == []
    assert completeness["with_plugin"]["execution_status"] == "succeeded"

    persisted = json.loads((tmp_path / "results" / "opencode" / "statistics.json").read_text(encoding="utf-8"))
    assert tuple(persisted) == stats.STATISTICS_BLOCKS
    assert {block: agent[block] for block in stats.STATISTICS_BLOCKS} == persisted
    assert persisted["lift_uncertainty"] == agent["lift_uncertainty"]
    assert persisted["integration_completeness"]["complete"] is True
    # Private usage counters never leak into persisted trial rewards.
    reward_file = next((tmp_path / "results" / "opencode" / "with-skill" / "trials").glob("*/reward.json"))
    assert "_usage" not in reward_file.read_text(encoding="utf-8")

    loaded = load_agent_data(tmp_path / "results")["opencode"]
    assert loaded["reliability"] == agent["reliability"]
    assert {block: loaded[block] for block in stats.STATISTICS_BLOCKS} == persisted
    payload = build_agent_eval_payload(
        "demo",
        {"opencode": loaded},
        run_config=PLUGIN_CONFIG,
        plugin_provenance={"requested_lift_mode": "both", "effective_lift_mode": "both"},
        use_llm_judge=False,
    )
    assert payload is not None
    assert payload["lift_uncertainty"]["effectiveness"]["estimate"] == 0.5
    assert payload["agents"]["opencode"]["cost"]["with_skill"]["successes"] == 6
    assert payload["lift_mode_requested"] == "both"
    assert payload["lift_mode_effective"] == "both"
    integration = payload["integration"]
    assert integration["verdict"] == "real_integration"
    assert integration["point_verdict"] == "real_integration"
    assert integration["measured"] is True
    assert integration["lift_uncertainty"]["ci_includes_zero"] is False
    assert integration["reason"] is None


def test_collector_marks_integration_incomplete_on_attempt_shortfall(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    _write_job(jobs, "with", dict.fromkeys(CASES, 0.9))
    _write_job(jobs, "without", dict.fromkeys(CASES, 0.4))
    _write_job(jobs, "sumofparts", dict.fromkeys(CASES, 0.7))
    (jobs / "demo-opencode-sumofparts" / "case-6__attempt002" / "verifier" / "reward.json").unlink()

    agent = _collect_three_arms(tmp_path)["agents"]["opencode"]

    completeness = agent["integration_completeness"]
    assert completeness["complete"] is False
    assert completeness["failed_arms"] == ["sum_of_parts"]
    assert {"case": "case-6", "arm": "sum_of_parts", "expected": 2, "observed": 1} in completeness["attempt_shortfall"]
    assert agent["lift_uncertainty"]["integration"] is None
    assert "sum_of_parts" not in agent["reliability"]


def test_collector_reports_context_cost_unavailable_without_step_metrics(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    _write_job(jobs, "with", dict.fromkeys(CASES, 0.9), per_step_metrics=False)
    _write_job(jobs, "without", dict.fromkeys(CASES, 0.4), per_step_metrics=False)

    result = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs,
        n_attempts=2,
        expected_cases=len(CASES),
        expected_case_ids=CASES,
        expected_trials=2 * len(CASES),
    )
    agent = result["agents"]["opencode"]

    assert agent["context_cost_measured"]["status"] == "unavailable"
    assert agent["lift_uncertainty"]["integration"] is None
    # No sum-of-parts arm was requested, so nothing is incomplete.
    assert agent["integration_completeness"]["complete"] is None


def test_trial_usage_falls_back_to_harbor_agent_result(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "case-1__attempt001"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "agent_result": {
                    "n_input_tokens": 9_000,
                    "n_cache_tokens": 4_000,
                    "n_output_tokens": 500,
                    "cost_usd": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )

    usage = _trial_usage(job_dir, {"_trial_root_name": "case-1__attempt001"})

    assert usage == {"prompt_tokens": 9_000, "completion_tokens": 500, "cached_tokens": 4_000, "cost_usd": 0.5}
    assert stats.uncached_tokens(usage) == 5_500
    assert _trial_usage(job_dir, {"_trial_root_name": "../escape"}) == {}
    assert _trial_usage(None, {"_trial_root_name": "case-1__attempt001"}) == {}


def test_trial_usage_never_follows_a_symlinked_trajectory(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "trajectory.json").write_text(
        json.dumps({"final_metrics": {"total_prompt_tokens": 1, "total_completion_tokens": 1}}), encoding="utf-8"
    )
    trial_dir = job_dir / "case-1__attempt001"
    trial_dir.mkdir(parents=True)
    (trial_dir / "agent").symlink_to(outside, target_is_directory=True)

    assert _trial_usage(job_dir, {"_trial_root_name": "case-1__attempt001"}) == {}


def test_trial_usage_sums_native_multistep_fragments(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "case-1__attempt001"
    for index, step in enumerate(("prepare", "finish"), start=1):
        agent_dir = trial_dir / "steps" / step / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "steps": [{"step_id": 1, "source": "agent", "metrics": {"prompt_tokens": 100 * index}}],
                    "final_metrics": {
                        "total_prompt_tokens": 1_000 * index,
                        "total_completion_tokens": 10 * index,
                        "total_cost_usd": 0.1,
                    },
                }
            ),
            encoding="utf-8",
        )
    (trial_dir / "result.json").write_text(
        json.dumps({"step_results": [{"step_name": "prepare"}, {"step_name": "finish"}]}), encoding="utf-8"
    )

    usage = _trial_usage(job_dir, {"_trial_root_name": "case-1__attempt001"})

    assert usage["prompt_tokens"] == 3_000
    assert usage["completion_tokens"] == 30
    assert usage["cost_usd"] == pytest.approx(0.2)
    assert usage["first_turn_prompt_tokens"] == 100


def test_oversized_usage_counter_is_missing_instead_of_aborting_collection(tmp_path: Path) -> None:
    # json.loads turns a 400-digit counter into an int that float() cannot convert.
    oversized = 10**400
    jobs = tmp_path / "jobs"
    _write_job(jobs, "with", dict.fromkeys(CASES, 0.9))
    _write_job(jobs, "without", dict.fromkeys(CASES, 0.4))
    job_dir = jobs / "demo-opencode-with"
    trial_dir = job_dir / "case-1__attempt001"
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["final_metrics"]["total_prompt_tokens"] = oversized
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    # The result.json fallback carries an oversized counter too.
    (trial_dir / "result.json").write_text(
        json.dumps({"agent_result": {"n_input_tokens": oversized, "n_output_tokens": 500}}), encoding="utf-8"
    )

    result = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs,
        n_attempts=2,
        expected_cases=len(CASES),
        expected_case_ids=CASES,
        expected_trials=2 * len(CASES),
    )

    assert _trial_usage(job_dir, {"_trial_root_name": "case-1__attempt001"}) == {
        "cost_usd": 0.01,
        "first_turn_prompt_tokens": 1_000,
    }
    cost = result["agents"]["opencode"]["cost"]["with_skill"]
    assert cost["total_tokens"] is None
    assert cost["total_usd"] == 0.12
    assert stats.uncached_tokens({"prompt_tokens": oversized, "completion_tokens": 1}) is None


# ---------------------------------------------------------------------------
# Integration verdict and lift modes
# ---------------------------------------------------------------------------


def _uncertainty(low: float, high: float, *, n_cases: int = 10, precision: str = "adequate") -> dict[str, Any]:
    return {
        "estimate": round((low + high) / 2, 4),
        "ci_low": low,
        "ci_high": high,
        "confidence": 0.95,
        "method": "paired_case_bootstrap",
        "resamples": 2000,
        "seed": 0,
        "n_cases": n_cases,
        "precision": precision,
        "ci_includes_zero": low <= 0 <= high,
    }


def _best(integration: dict[str, Any] | None, *, sum_of_parts: float = 0.65) -> dict[str, Any]:
    return {
        "with_skill": 0.80,
        "baseline": 0.30,
        "sum_of_parts": sum_of_parts,
        "integration_completeness": {"complete": True},
        "lift_uncertainty": {"effectiveness": _uncertainty(0.4, 0.6), "integration": integration},
    }


def test_integration_ci_including_zero_downgrades_verdict_to_inconclusive() -> None:
    report = _build_integration_report(_best(_uncertainty(-0.05, 0.35, precision="low")), PLUGIN_CONFIG)

    assert report is not None
    assert report["integration_lift"] == 0.15
    assert report["point_verdict"] == "real_integration"
    assert report["verdict"] == "inconclusive"
    assert "includes zero" in report["reason"]
    assert "[-0.05, +0.35]" in report["reason"]
    assert report["measured"] is True


def test_integration_with_too_few_paired_cases_is_inconclusive() -> None:
    report = _build_integration_report(
        _best(_uncertainty(0.1, 0.2, n_cases=3, precision="insufficient")), PLUGIN_CONFIG
    )
    assert report is not None
    assert report["verdict"] == "inconclusive"
    assert report["point_verdict"] == "real_integration"
    assert "3 paired case(s)" in report["reason"]


def test_integration_ci_excluding_zero_keeps_the_point_verdict() -> None:
    negative = _build_integration_report(_best(_uncertainty(-0.3, -0.1), sum_of_parts=0.95), PLUGIN_CONFIG)
    assert negative is not None
    assert negative["verdict"] == negative["point_verdict"] == "negative_integration"
    assert negative["reason"] is None
    low_precision = _build_integration_report(_best(_uncertainty(0.01, 0.4, precision="low")), PLUGIN_CONFIG)
    assert low_precision is not None and low_precision["verdict"] == "real_integration"


def test_integration_incomplete_arms_explain_the_inconclusive_verdict() -> None:
    best = _best(_uncertainty(0.1, 0.2))
    best["integration_completeness"] = {
        "complete": False,
        "missing_cases": ["case-9"],
        "failed_arms": ["sum_of_parts"],
        "attempt_shortfall": [{"case": "case-9", "arm": "sum_of_parts", "expected": 2, "observed": 0}],
    }
    report = _build_integration_report(best, PLUGIN_CONFIG)
    assert report is not None
    assert report["verdict"] == "inconclusive"
    assert report["point_verdict"] == "real_integration"
    assert "missing case(s): case-9" in report["reason"]
    assert "failed arm(s): sum_of_parts" in report["reason"]


def test_legacy_integration_mode_uses_the_member_skills_baseline_interval() -> None:
    config = {
        "eval_target": {"kind": "plugin"},
        "skill_workspace": {"staged_skills": ["member"], "baseline_includes_workspace_skills": True},
    }
    best = _best(None)
    best["lift_uncertainty"]["effectiveness"] = _uncertainty(-0.1, 0.9)
    report = _build_integration_report(best, config)
    assert report is not None
    assert report["integration_lift"] == 0.5
    assert report["verdict"] == "inconclusive"
    assert report["lift_mode_effective"] == "integration"
    assert report["lift_mode_requested"] == "integration"


def test_requested_but_unmeasured_integration_emits_explicit_inconclusive_block() -> None:
    config = {
        "eval_target": {"kind": "plugin"},
        "skill_workspace": {"staged_skills": ["member"], "sum_of_parts_arm": False},
    }
    provenance = {
        "requested_lift_mode": "both",
        "effective_lift_mode": "effectiveness",
        "integration_skip_reason": "no dataset case sets cross_component=true",
    }

    report = _build_integration_report(_best(None), config, provenance)

    assert report is not None
    assert report["verdict"] == "inconclusive"
    assert report["measured"] is False
    assert report["reason"] == "no dataset case sets cross_component=true"
    assert report["integration_lift"] is None and report["point_verdict"] is None
    assert report["lift_mode_requested"] == "both"
    assert report["lift_mode_effective"] == "effectiveness"

    # Recorded in run_config by the runner when no provenance is passed.
    recorded = {**config, "lift_mode": {"requested": "both", "effective": "effectiveness"}}
    from_run_config = _build_integration_report(_best(None), recorded)
    assert from_run_config is not None and from_run_config["measured"] is False
    assert "not run" in from_run_config["reason"]

    # Integration was never requested: no block at all.
    assert _build_integration_report(_best(None), config, {"requested_lift_mode": "effectiveness"}) is None
    assert _build_integration_report(_best(None), config) is None


def test_payload_always_records_requested_and_effective_lift_modes() -> None:
    assert _plugin_lift_modes(PLUGIN_CONFIG, None) == {
        "requested": "both",
        "effective": "both",
        "integration_skip_reason": None,
    }
    effectiveness_only = {**PLUGIN_CONFIG, "skill_workspace": {"staged_skills": ["m"]}}
    assert _plugin_lift_modes(effectiveness_only, None)["requested"] is None
    assert _plugin_lift_modes(effectiveness_only, None)["effective"] == "effectiveness"
    assert _plugin_lift_modes(effectiveness_only, {"requested_lift_mode": "bogus"})["requested"] is None

    assert _plugin_lift_fallback_metadata("effectiveness", "effectiveness", None) == {
        "requested_lift_mode": "effectiveness",
        "effective_lift_mode": "effectiveness",
    }
    assert _plugin_lift_fallback_metadata("both", "effectiveness", "no evidence")["integration_skip_reason"] == (
        "no evidence"
    )


def _payload_agent(uncertainty: dict[str, Any] | None) -> dict[str, Any]:
    scores = dict.fromkeys(DEFAULT_METRICS, 0.8)
    baseline = dict.fromkeys(DEFAULT_METRICS, 0.78)
    return {
        "execution_status": "succeeded",
        "conditions": {
            "with_skill": {"execution_status": "succeeded"},
            "without_skill": {"execution_status": "succeeded"},
        },
        "with_skill": scores,
        "without_skill": baseline,
        "lift_uncertainty": {"effectiveness": uncertainty, "integration": None},
    }


def test_effectiveness_ci_including_zero_warns_without_changing_the_verdict() -> None:
    warned = build_agent_eval_payload(
        "demo", {"opencode": _payload_agent(_uncertainty(-0.05, 0.09, precision="low"))}, use_llm_judge=False
    )
    quiet = build_agent_eval_payload(
        "demo", {"opencode": _payload_agent(_uncertainty(0.01, 0.03))}, use_llm_judge=False
    )
    assert warned is not None and quiet is not None
    assert warned["verdict"] == quiet["verdict"]
    assert warned["overall_lift"] == quiet["overall_lift"] == 0.02
    titles = [conclusion["title"] for conclusion in warned["conclusions"]]
    assert "Skill Lift not distinguishable from zero" in titles
    assert "Skill Lift not distinguishable from zero" not in [c["title"] for c in quiet["conclusions"]]
    assert warned["lift_uncertainty"]["effectiveness"]["precision"] == "low"
    # Skill (non-plugin) payloads carry no lift-mode fields.
    assert "lift_mode_requested" not in warned


def test_cli_plugin_summary_prints_integration_interval_and_reason() -> None:
    from io import StringIO

    from rich.console import Console

    from skillevaluator.reporting.cli import CLIReporter

    buffer = StringIO()
    report = _build_integration_report(_best(_uncertainty(-0.05, 0.35, precision="low")), PLUGIN_CONFIG)
    CLIReporter._print_agent_eval_tables(
        {"verdict": "pass", "integration": report},
        Console(file=buffer, width=200, color_system=None),
    )

    output = buffer.getvalue()
    assert "Integration: INCONCLUSIVE (lift +0.15)" in output
    assert "95% CI [-0.05, +0.35], precision low" in output
    assert "includes zero" in output


def test_completeness_without_a_requested_sum_of_parts_arm_is_not_applicable() -> None:
    result = stats.integration_completeness(
        _arm(_full(["a"])),
        None,
        expected_case_ids=["a"],
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        requested=False,
    )
    assert result == {"complete": None, "missing_cases": [], "failed_arms": [], "attempt_shortfall": []}


def test_a_requested_sum_of_parts_arm_without_observations_is_a_failed_arm() -> None:
    result = stats.integration_completeness(
        _arm(_full(["a"])), None, expected_case_ids=["a"], n_attempts=1, stop_on_pass=False, pass_threshold=0.5
    )
    assert result["failed_arms"] == ["sum_of_parts"]


def test_agent_statistics_of_an_effectiveness_only_run_have_nothing_incomplete() -> None:
    arms = {"with_skill": _arm(_full(["a", "b"])), "without_skill": _arm(_full(["a", "b"], score=0.4))}
    result = stats.build_agent_statistics(
        arms,
        expected_case_ids=["a", "b"],
        n_attempts=2,
        stop_on_pass=False,
        pass_threshold=0.5,
        sum_of_parts_requested=False,
    )
    assert tuple(result) == stats.STATISTICS_BLOCKS
    assert result["integration_completeness"]["complete"] is None
