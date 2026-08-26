# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from skillevaluator.tier3.harbor.collector import (
    _annotate_security_attribution,
    _reward_publication_projection_is_safe,
)


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
    assert "target skill" in finding["attribution_explanation"]
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


def test_security_attribution_omits_per_finding_expansion_when_projection_would_overflow():
    findings = [{"score_impact": True} for _ in range(17_000)]
    with_rewards = [_reward("case-1", findings)]

    summary = _annotate_security_attribution(with_rewards, [], baseline_run=False)

    security = with_rewards[0]["details"]["security"]
    assert summary["unknown_no_baseline"] == len(findings)
    assert security["attribution"] == "unknown_no_baseline"
    assert "omitted" in security["attribution_completeness"].casefold()
    assert all("attribution" not in finding for finding in findings)


def test_security_attribution_retains_labels_when_only_repeated_explanations_overflow():
    findings = [{"score_impact": True} for _ in range(9_000)]
    with_rewards = [_reward("case-1", findings)]

    _annotate_security_attribution(with_rewards, [], baseline_run=False)

    security = with_rewards[0]["details"]["security"]
    assert "repeated explanations were omitted" in security["attribution_completeness"]
    projected_findings = security["findings"]
    assert all(finding["attribution"] == "unknown_no_baseline" for finding in projected_findings)
    assert all("attribution_explanation" not in finding for finding in projected_findings)
    assert _reward_publication_projection_is_safe(with_rewards[0])


def test_security_improvement_normalizes_malformed_findings_without_crashing():
    with_rewards = [
        {
            "entry_id": "case-1",
            "details": {"security": {"score": 1.0, "findings": {}}},
        }
    ]
    without_rewards = [_reward("case-1", [_finding()])]

    summary = _annotate_security_attribution(with_rewards, without_rewards)

    security = with_rewards[0]["details"]["security"]
    assert summary["skill_may_have_improved_safety"] == 1
    assert isinstance(security["findings"], list)
    assert security["findings"][0]["type"] == "skill_reduced_unsafe_behavior"


def test_security_improvement_does_not_copy_unbounded_baseline_messages():
    padding = "x" * 1_500_000
    with_reward = {
        "entry_id": "case-1",
        "metric_set": "custom-only",
        "overall": 1.0,
        "details": {"padding": padding, "security": {"score": 1.0, "findings": []}},
    }
    baseline_finding = _finding()
    baseline_finding["message"] = padding

    summary = _annotate_security_attribution(
        [with_reward],
        [_reward("case-1", [baseline_finding])],
    )

    security = with_reward["details"]["security"]
    assert summary["skill_may_have_improved_safety"] == 1
    assert security["findings"][0]["evidence"].startswith("Without-skill baseline contained 1")
    assert padding not in security["findings"][0]["evidence"]
    assert _reward_publication_projection_is_safe(with_reward)


def test_security_attribution_bounds_case_details_with_exact_metadata():
    with_rewards = [_reward(f"case-{index:04d}", []) for index in range(300)]

    summary = _annotate_security_attribution(with_rewards, [], baseline_run=False)

    assert summary["case_details_total"] == 300
    assert summary["case_details_shown"] == 256
    assert summary["case_details_truncated"] is True
    assert len(summary["cases"]) == 256
