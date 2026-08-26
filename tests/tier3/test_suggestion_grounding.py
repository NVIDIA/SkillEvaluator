# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from skillevaluator.tier3.harbor import report


def _reward(metric_score=0.1):
    return {
        "entry_id": "evaluator-plugin-002",
        "goal_accuracy": metric_score,
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
        "behavior_check": 1.0,
        "details": {
            "goal_accuracy": {
                "reason": "job submitted but results file not produced",
                "evidence_refs": [
                    {
                        "source": "trajectory.json",
                        "json_pointer": "/steps/14",
                        "kind": "tool_call",
                        "label": "bash: nemo evaluator submit",
                        "excerpt": "submit ...",
                    },
                ],
                "omitted": {"count": 3, "truncated": True, "reason": "older results dropped"},
            },
        },
    }


def _reward_multi_metric(metric_score=0.1):
    """Reward with evidence_refs on multiple metrics for lookup testing."""
    return {
        "entry_id": "evaluator-plugin-003",
        "goal_accuracy": metric_score,
        "security": 0.3,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
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
                        "excerpt": "submit ...",
                    },
                ],
            },
            "security": {
                "reason": "unsafe operation",
                "evidence_refs": [
                    {
                        "source": "trajectory.json",
                        "json_pointer": "/steps/5",
                        "kind": "tool_call",
                        "path": "steps[5].tool_use",
                        "excerpt": "rm -rf /",
                    },
                ],
            },
        },
    }


def test_findings_carry_evidence_refs():
    findings = report._extract_findings([_reward(0.1)])
    goal = next(f for f in findings if f["metric"] == "goal_accuracy")
    assert goal["evidence_refs"], "finding must carry the metric's evidence_refs"
    assert goal["evidence_refs"][0]["json_pointer"] == "/steps/14"


def test_findings_normalize_legacy_string_evidence_refs_without_crashing():
    reward = _reward(0.1)
    reward["details"]["goal_accuracy"]["evidence_refs"] = ["trajectory.json#/steps/14"]

    findings = report._extract_findings([reward])

    goal = next(finding for finding in findings if finding["metric"] == "goal_accuracy")
    assert goal["evidence_refs"] == [
        {
            "source": "trajectory.json",
            "json_pointer": "/steps/14",
            "kind": "evidence",
        }
    ]


@pytest.mark.parametrize(
    ("metric", "malformed_detail"),
    [
        ("behavior_check", {"results": ["malformed"]}),
        ("accuracy", []),
        ("accuracy", {"criteria": [], "reason": 42}),
        ("goal_accuracy", {"findings": "malformed", "reason": {"nested": "value"}}),
    ],
)
@pytest.mark.parametrize("metric_score", [0.1, 0.9], ids=["failing", "passing"])
def test_findings_tolerate_malformed_nested_detail_shapes(metric, malformed_detail, metric_score):
    reward = _reward(metric_score)
    reward[metric] = metric_score
    reward["details"][metric] = malformed_detail

    findings = report._extract_findings([reward])

    finding = next(item for item in findings if item["metric"] == metric)
    assert finding["score"] == pytest.approx(metric_score)
    assert all(isinstance(reason, str) and len(reason) <= 512 for reason in finding["reasons"])


def test_findings_report_rejects_overflowing_reward_numbers_without_crashing():
    reward = _reward(10**400)
    reward["details"]["goal_accuracy"]["score"] = 10**400

    assert report._extract_findings([reward]) == []
    assert report._prioritized_evidence_rewards([reward]) == [reward]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), 10**400])
def test_findings_best_agent_ignores_invalid_legacy_summary_numbers(invalid: float | int):
    invalid_scores = dict.fromkeys(report.DISPLAY_METRICS, 1.0)
    invalid_scores[report.DISPLAY_METRICS[0]] = invalid
    agents = {
        "invalid": {
            "execution_status": "succeeded",
            "with_skill": invalid_scores,
        },
        "valid": {
            "execution_status": "succeeded",
            "with_skill": dict.fromkeys(report.DISPLAY_METRICS, 0.5),
        },
    }

    assert report._pick_best_agent(agents) == "valid"


def test_passing_suggestions_count_mixed_current_and_legacy_logical_trials():
    suggestions = report._passing_skill_suggestions(
        [],
        [
            {"trial_id": "current-attempt"},
            {"trial_id": "current-attempt"},
            {"entry_id": "legacy-one"},
            {"entry_id": "legacy-two"},
        ],
    )

    assert any("currently 3" in suggestion for suggestion in suggestions)


def test_generate_suggestions_prompt_includes_refs_and_uses_larger_budget(monkeypatch):
    captured = {}

    def fake_hub(prompt, **_kw):
        captured["prompt"] = prompt
        captured["max_tokens"] = _kw.get("max_tokens")
        return (
            '[{"suggestion": "Wait for the evaluator job and save results.", '
            '"dimension": "goal_accuracy", "evidence_refs": ["trajectory.json#/steps/14"]}]',
            None,
        )

    monkeypatch.setattr("skillevaluator.tier3.eval_core.llm_judge.call_public_llm", fake_hub)
    findings = report._extract_findings([_reward(0.1)])
    out = report._generate_suggestions("demo-skill", findings, [_reward(0.1)])
    assert "/steps/14" in captured["prompt"]  # refs reached the prompt
    assert captured["max_tokens"] and captured["max_tokens"] >= 1024  # raised from 512
    assert out and isinstance(out[0], str)  # back-compat: returns display strings


def test_generate_suggestions_structured_returns_objects(monkeypatch):
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "X", "dimension": "goal_accuracy", "evidence_refs": ["trajectory.json#/steps/14"]}]',
            None,
        ),
    )
    findings = report._extract_findings([_reward(0.1)])
    objs = report._generate_suggestions_structured("demo", findings, [_reward(0.1)])
    assert objs and objs[0]["dimension"] == "goal_accuracy" and "suggestion" in objs[0]


def test_findings_artifact_includes_suggestions_v2(tmp_path):
    import json

    findings = report._extract_findings([_reward(0.1)])
    art = report._write_findings_artifact(
        results_dir=tmp_path,
        skill_name="demo",
        agent="codex",
        findings=findings,
        suggestions=["do X"],
        suggestion_mode="remediation",
        suggestions_v2=[
            {
                "suggestion": "do X",
                "dimension": "goal_accuracy",
                "trial_id": "evaluator-plugin-002",
                "evidence_refs": [],
            }
        ],
    )
    payload = json.loads(art.read_text())
    assert "suggestions_v2" in payload and payload["suggestions_v2"][0]["dimension"] == "goal_accuracy"


def test_findings_artifact_bounds_evidence_across_multiple_large_rewards(tmp_path):
    rewards = []
    for index in range(8):
        reward = _reward(0.1)
        reward["entry_id"] = f"case-{index}"
        reward["details"]["goal_accuracy"]["evidence_refs"] = [
            {
                "source": f"trajectory-{index}.json",
                "json_pointer": f"/steps/{index}",
                "kind": "tool_call",
                "excerpt": "x" * 1_000_000,
            }
        ]
        rewards.append(reward)

    findings = report._extract_findings(rewards)
    art = report._write_findings_artifact(
        results_dir=tmp_path,
        skill_name="demo",
        agent="codex",
        findings=findings,
        suggestions=[],
        suggestion_mode="not_generated",
    )
    payload = json.loads(art.read_text(encoding="utf-8"))
    goal = next(finding for finding in payload["findings"] if finding["metric"] == "goal_accuracy")

    assert art.stat().st_size <= report.report_data._MAX_JSON_BYTES
    assert len(goal["evidence_refs"]) == report._MAX_FINDING_EVIDENCE_REFS
    assert all(len(ref["excerpt"]) < 1_000 for ref in goal["evidence_refs"])


# --- New tests for dict-shaped evidence_refs in suggestions_v2 ---


def test_suggestions_evidence_refs_resolved_to_full_dict(monkeypatch):
    """LLM returns string ref; function resolves it to the full dict from rewards."""
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "Fix evaluator job", "dimension": "goal_accuracy",'
            ' "evidence_refs": ["trajectory.json#/steps/14"]}]',
            None,
        ),
    )
    reward = _reward_multi_metric(0.1)
    findings = report._extract_findings([reward])
    objs = report._generate_suggestions_structured("demo", findings, [reward])
    assert objs, "expected at least one suggestion"
    refs = objs[0]["evidence_refs"]
    assert refs, "expected non-empty evidence_refs"
    # Must be a dict now, not a plain string
    assert isinstance(refs[0], dict), f"expected dict ref, got {type(refs[0])!r}: {refs[0]!r}"
    # Must have preserved the rich fields from the reward's evidence_refs
    assert refs[0]["kind"] == "tool_call", f"kind not preserved: {refs[0]}"
    assert refs[0]["json_pointer"] == "/steps/14"
    assert refs[0]["source"] == "trajectory.json"
    # Optional fields should be present if they were in the source dict
    assert "path" in refs[0]
    assert "excerpt" in refs[0]


def test_suggestions_evidence_refs_unresolvable_string_fallback(monkeypatch):
    """Unresolvable string ref is parsed into a minimal dict with kind='evidence'."""
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "Fix something", "dimension": "goal_accuracy",'
            ' "evidence_refs": ["trajectory.json#/steps/999"]}]',
            None,
        ),
    )
    reward = _reward_multi_metric(0.1)
    findings = report._extract_findings([reward])
    objs = report._generate_suggestions_structured("demo", findings, [reward])
    assert objs
    refs = objs[0]["evidence_refs"]
    assert refs and isinstance(refs[0], dict), f"expected dict fallback, got {refs!r}"
    assert refs[0]["source"] == "trajectory.json"
    assert refs[0]["json_pointer"] == "/steps/999"
    assert refs[0]["kind"] == "evidence"


def test_suggestions_evidence_refs_lookup_uses_all_metrics(monkeypatch):
    """Lookup is built from ALL metrics' evidence_refs, not just goal_accuracy."""
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "Fix security", "dimension": "security", "evidence_refs": ["trajectory.json#/steps/5"]}]',
            None,
        ),
    )
    reward = _reward_multi_metric(0.1)
    findings = report._extract_findings([reward])
    objs = report._generate_suggestions_structured("demo", findings, [reward])
    assert objs
    refs = objs[0]["evidence_refs"]
    assert refs and isinstance(refs[0], dict)
    assert refs[0]["json_pointer"] == "/steps/5"
    assert refs[0]["kind"] == "tool_call"


def test_suggestions_structured_evidence_refs_are_dicts_not_strings(monkeypatch):
    """End-to-end: suggestions_v2 artifacts must have dict refs, never plain strings."""
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "Wait for evaluator job", "dimension": "goal_accuracy",'
            ' "evidence_refs": ["trajectory.json#/steps/14"]}]',
            None,
        ),
    )
    reward = _reward(0.1)
    findings = report._extract_findings([reward])
    objs = report._generate_suggestions_structured("demo", findings, [reward])
    for obj in objs:
        for ref in obj.get("evidence_refs", []):
            assert isinstance(ref, dict), f"evidence_ref must be a dict in suggestions_v2, got {type(ref)!r}: {ref!r}"


def test_display_findings_report_writes_artifact(tmp_path, monkeypatch):
    """Smoke the real findings report display path over a temporary run directory."""
    import json

    condition_dir = tmp_path / "codex" / "with-skill"
    trial_dir = condition_dir / "trials" / "case-001"
    trial_dir.mkdir(parents=True)
    reward = _reward(0.1)
    (condition_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "codex",
                "scores": {
                    metric: reward[metric]
                    for metric in (
                        "security",
                        "skill_execution",
                        "skill_efficiency",
                        "accuracy",
                        "goal_accuracy",
                        "behavior_check",
                    )
                },
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    monkeypatch.setattr(
        report,
        "_generate_suggestions_structured",
        lambda _skill, _findings, _rewards: [
            {
                "suggestion": "Tighten the workflow around result-file creation.",
                "dimension": "goal_accuracy",
                "evidence_refs": [{"source": "trajectory.json", "json_pointer": "/steps/14", "kind": "tool_call"}],
            }
        ],
    )
    report.display_findings_report(
        {
            "env_mode": "local",
            "agents": {
                "codex": {
                    "execution_status": "succeeded",
                    "model": "gpt-test",
                    "model_source": "test",
                    "with_skill": {
                        "security": 1.0,
                        "skill_execution": 1.0,
                        "skill_efficiency": 1.0,
                        "accuracy": 1.0,
                        "goal_accuracy": 0.1,
                        "behavior_check": 1.0,
                    },
                }
            },
        },
        "demo-skill",
        ["codex"],
        tmp_path,
    )

    artifact = tmp_path / "codex" / "findings.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["skill_name"] == "demo-skill"
    assert payload["agent"] == "codex"
    assert payload["suggestion_mode"] == "remediation"
    assert payload["suggestions"] == ["Tighten the workflow around result-file creation."]
    assert payload["suggestions_v2"][0]["dimension"] == "goal_accuracy"
    assert any(finding["metric"] == "goal_accuracy" for finding in payload["findings"])
