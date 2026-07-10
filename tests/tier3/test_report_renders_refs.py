# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from skillevaluator.tier3.harbor import report


def _reward():
    return {
        "entry_id": "evaluator-plugin-002",
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
        "goal_accuracy": 0.1,
        "behavior_check": 1.0,
        "details": {
            "goal_accuracy": {
                "reason": "results file not produced",
                "evidence_refs": [
                    {
                        "source": "trajectory.json",
                        "json_pointer": "/steps/14",
                        "kind": "tool_call",
                        "label": "bash: nemo evaluator submit",
                    }
                ],
            }
        },
    }


def _reward_with_dict_refs():
    """Reward whose evidence_refs are the new dict shape."""
    return {
        "entry_id": "evaluator-plugin-099",
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
        "goal_accuracy": 0.2,
        "behavior_check": 1.0,
        "details": {
            "goal_accuracy": {
                "reason": "results file not produced",
                "evidence_refs": [
                    {
                        "source": "trajectory.json",
                        "json_pointer": "/steps/14",
                        "kind": "tool_call",
                        "path": "steps[14].tool_use",
                        "excerpt": "submit command",
                    }
                ],
            }
        },
    }


def _reward_with_custom_metric():
    return {
        "entry_id": "custom-001",
        "overall": 0.9,
        "domain_quality": 0.9,
        "custom_metrics": {"domain_quality": 0.9},
        "details": {
            "domain_quality": {
                "score": 0.9,
                "reason": "custom domain matched",
                "evidence_refs": [
                    {
                        "source": "custom_reward.json",
                        "json_pointer": "/details/domain_quality",
                        "kind": "custom_metric",
                        "label": "domain quality",
                    }
                ],
            }
        },
    }


def test_cli_findings_body_renders_evidence_pointer():
    findings = report._extract_findings([_reward()])
    text = report._render_findings_body(findings).plain
    assert "/steps/14" in text and "evidence:" in text


def test_cli_findings_include_custom_metric_details():
    findings = report._extract_findings([_reward_with_custom_metric()])
    custom = next(f for f in findings if f["metric"] == "domain_quality")
    assert custom["label"] == "custom: domain_quality"
    assert custom["reasons"] == ["custom domain matched"]
    assert custom["evidence_refs"][0]["source"] == "custom_reward.json"


# --- New renderer backward-compat tests ---


def test_cli_renders_dict_ref_as_compact_string(monkeypatch):
    """CLI renderer must show source+json_pointer when fed a dict ref from suggestions_v2."""
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "Fix evaluator", "dimension": "goal_accuracy",'
            ' "evidence_refs": ["trajectory.json#/steps/14"]}]',
            None,
        ),
    )
    reward = _reward_with_dict_refs()
    findings = report._extract_findings([reward])
    objs = report._generate_suggestions_structured("demo", findings, [reward])
    # The resolved ref should be a dict
    assert objs and isinstance(objs[0]["evidence_refs"][0], dict)
    # _render_findings_body handles dicts in finding.evidence_refs (already dict-shaped from _extract_findings)
    text = report._render_findings_body(findings).plain
    assert "trajectory.json" in text


def test_cli_renders_legacy_string_ref_without_error():
    """CLI renderer must tolerate legacy string refs in findings (backward compat)."""
    # Simulate a legacy artifact where evidence_refs is a list of strings
    findings_with_string_refs = [
        {
            "metric": "goal_accuracy",
            "label": "GOAL ACCURACY",
            "severity": "warning",
            "score": 0.5,
            "reasons": ["something failed"],
            "evidence_refs": ["trajectory.json#/steps/14"],  # legacy string format
        }
    ]
    # Should not raise; string refs are rendered defensively
    text = report._render_findings_body(findings_with_string_refs).plain
    # The ref won't be rendered as a dict, but the function should not crash
    assert text  # just checking it renders without exception
