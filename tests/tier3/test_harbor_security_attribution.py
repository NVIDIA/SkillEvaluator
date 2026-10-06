# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from skillevaluator.tier3.harbor.collector import _annotate_security_attribution, collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS


def _reward(entry_id, findings):
    return {
        "entry_id": entry_id,
        "details": {
            "security": {
                "score": 0.0 if any(f.get("score_impact") for f in findings) else 1.0,
                "findings": findings,
            }
        },
    }


def _finding(finding_type="destructive_command", *, target_skill_used_before=True, evidence=None):
    evidence = evidence or ("rm -rf cpp/build" if finding_type == "destructive_command" else "~/.bashrc")
    return {
        "type": finding_type,
        "severity": "critical",
        "message": f"Agent executed unsafe action: {evidence}",
        "evidence": evidence,
        "source": "agent_tool_call",
        "score_impact": True,
        "target_skill_used_before": target_skill_used_before,
    }


def test_security_attribution_marks_with_skill_only_after_skill_use_as_skill_related():
    with_rewards = [_reward("case-1", [_finding(target_skill_used_before=True)])]
    without_rewards = [_reward("case-1", [])]

    summary = _annotate_security_attribution(with_rewards, without_rewards)

    finding = with_rewards[0]["details"]["security"]["findings"][0]
    assert finding["attribution"] == "likely_skill_related"
    assert summary["likely_skill_related"] == 1


def test_security_attribution_marks_shared_unsafe_behavior_as_baseline_related():
    with_rewards = [_reward("case-1", [_finding()])]
    without_rewards = [_reward("case-1", [_finding()])]

    summary = _annotate_security_attribution(with_rewards, without_rewards)

    finding = with_rewards[0]["details"]["security"]["findings"][0]
    assert finding["attribution"] == "likely_baseline_prompt_or_environment"
    assert summary["likely_baseline_prompt_or_environment"] == 1


def test_security_attribution_keeps_unrelated_baseline_findings_separate():
    with_rewards = [_reward("case-1", [_finding("destructive_command", target_skill_used_before=True)])]
    without_rewards = [_reward("case-1", [_finding("sensitive_file_write")])]

    summary = _annotate_security_attribution(with_rewards, without_rewards)

    finding = with_rewards[0]["details"]["security"]["findings"][0]
    assert finding["attribution"] == "likely_skill_related"
    assert summary["likely_skill_related"] == 1
    assert summary["likely_baseline_prompt_or_environment"] == 0


def test_security_attribution_notes_when_skill_may_have_improved_safety():
    with_rewards = [_reward("case-1", [])]
    without_rewards = [_reward("case-1", [_finding()])]

    summary = _annotate_security_attribution(with_rewards, without_rewards)

    security = with_rewards[0]["details"]["security"]
    assert security["attribution"] == "skill_may_have_improved_safety"
    assert security["findings"][0]["type"] == "skill_reduced_unsafe_behavior"
    assert summary["skill_may_have_improved_safety"] == 1


def test_security_attribution_avoids_skill_blame_without_baseline():
    with_rewards = [_reward("case-1", [_finding(target_skill_used_before=True)])]

    summary = _annotate_security_attribution(with_rewards, [], baseline_run=False)

    finding = with_rewards[0]["details"]["security"]["findings"][0]
    assert finding["attribution"] == "unknown_no_baseline"
    assert summary["unknown_no_baseline"] == 1


def test_security_attribution_preserves_distinct_authored_ids_without_expected_set():
    """Verify authored IDs like case-1-with-skill and skillevaluator-case-1 do not collapse onto case-1."""
    with_rewards = [
        _reward("case-1", []),
        _reward("case-1-with-skill", [_finding(target_skill_used_before=True)]),
        _reward("skillevaluator-case-1", [_finding(target_skill_used_before=True)]),
    ]
    without_rewards = [
        _reward("case-1", [_finding()]),
        _reward("case-1-with-skill", []),
        _reward("skillevaluator-case-1", []),
    ]

    summary = _annotate_security_attribution(with_rewards, without_rewards)

    assert with_rewards[1]["details"]["security"]["findings"][0]["attribution"] == "likely_skill_related"
    assert with_rewards[2]["details"]["security"]["findings"][0]["attribution"] == "likely_skill_related"
    assert summary["likely_skill_related"] == 2
    assert summary["likely_baseline_prompt_or_environment"] == 0
    assert set(summary["cases"]) == {"case-1", "case-1-with-skill", "skillevaluator-case-1"}


def _write_harbor_job(jobs_dir: Path, variant: str, rewards: dict[str, dict]) -> None:
    job_dir = jobs_dir / f"demo-opencode-{variant}"
    for trial_name, reward in rewards.items():
        verifier = job_dir / trial_name / "verifier"
        verifier.mkdir(parents=True)
        scored = {"metric_set": DEFAULT_METRIC_SET, **dict.fromkeys(DEFAULT_METRICS, 0.5), "overall": 0.5, **reward}
        (verifier / "reward.json").write_text(json.dumps(scored), encoding="utf-8")
    stats = {
        "n_completed_trials": len(rewards),
        "n_errored_trials": 0,
        "n_running_trials": 0,
        "n_pending_trials": 0,
        "n_cancelled_trials": 0,
        "n_retries": 0,
        "evals": {
            "opencode": {"n_trials": len(rewards), "n_errors": 0, "reward_stats": {"reward": {"0.5": list(rewards)}}}
        },
    }
    (job_dir / "result.json").write_text(json.dumps({"n_total_trials": len(rewards), "stats": stats}), encoding="utf-8")


def test_the_saved_with_skill_rewards_carry_the_security_attribution(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_harbor_job(
        jobs_dir,
        "with",
        {
            "case-1__with": _reward("case-1", [_finding(target_skill_used_before=True)]),
            "case-2__with": _reward("case-2", []),
        },
    )
    _write_harbor_job(
        jobs_dir,
        "without",
        {"case-1__without": _reward("case-1", []), "case-2__without": _reward("case-2", [_finding()])},
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=False,
        expected_cases=2,
        expected_case_ids=["case-1", "case-2"],
        expected_trials=2,
    )

    assert results["agents"]["opencode"]["security_attribution"]["likely_skill_related"] == 1
    trials_dir = tmp_path / "results" / "opencode" / "with-skill" / "trials"
    flagged = json.loads((trials_dir / "case-1__with" / "reward.json").read_text(encoding="utf-8"))
    assert flagged["details"]["security"]["findings"][0]["attribution"] == "likely_skill_related"
    improved = json.loads((trials_dir / "case-2__with" / "reward.json").read_text(encoding="utf-8"))
    assert improved["details"]["security"]["attribution"] == "skill_may_have_improved_safety"
    assert improved["details"]["security"]["findings"][0]["type"] == "skill_reduced_unsafe_behavior"
