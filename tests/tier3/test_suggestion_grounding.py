# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from skillevaluator.tier3.eval_core.atif_helpers import attach_metric_evidence_refs, build_metric_evidence_refs
from skillevaluator.tier3.harbor import report
from skillevaluator.tier3.harbor.metrics import CUSTOM_ONLY_METRIC_SET, DEFAULT_METRIC_SET, DEFAULT_METRICS


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


@pytest.mark.parametrize("same_excerpt", [False, True])
def test_cross_trial_compact_refs_are_rejected_in_both_orders(same_excerpt):
    first, second = _reward(), _reward()
    second["entry_id"] = "different-case"
    if not same_excerpt:
        second["details"]["goal_accuracy"]["evidence_refs"][0]["excerpt"] = "different case evidence"
    key = "trajectory.json#/steps/14"

    for rewards in ([first, second], [second, first]):
        lookup = report._build_evidence_ref_lookup(rewards)
        assert report._recorded_evidence_refs([key], lookup) == []


def test_repeated_ref_within_one_trial_remains_available():
    reward = _reward()
    reward["details"]["accuracy"] = reward["details"]["goal_accuracy"]
    lookup = report._build_evidence_ref_lookup([reward])

    [ref] = report._recorded_evidence_refs(["trajectory.json#/steps/14"], lookup)

    assert ref["excerpt"] == "submit ..."


def test_multi_trial_refs_are_qualified_from_reward_ownership():
    first, second = _reward(), _reward()
    first["trial_id"], second["trial_id"] = "case-a__one", "case-b__one"
    second["details"]["goal_accuracy"]["evidence_refs"][0]["json_pointer"] = "/steps/99"
    for rewards in ([first, second], [second, first]):
        lookup = report._build_evidence_ref_lookup(rewards)
        assert len(lookup) == 2
        assert "trials/case-a__one/trajectory.json#/steps/14" in lookup
        assert "trials/case-b__one/trajectory.json#/steps/99" in lookup
    for reward in (first, second):
        reward["details"]["goal_accuracy"]["evidence_refs"][0]["source"] = f"{reward['trial_id']}/trajectory.json"
    forward = report._build_evidence_ref_lookup([first, second])
    backward = report._build_evidence_ref_lookup([second, first])

    assert forward == backward
    assert len(forward) == 2
    assert report._recorded_evidence_refs(["case-b__one/trajectory.json#/steps/99"], forward)[0]["source"] == (
        "case-b__one/trajectory.json"
    )


def test_empty_evidence_lookup_rejects_unrecorded_refs():
    assert report._recorded_evidence_refs(["trajectory.json#/steps/14"], report._build_evidence_ref_lookup([])) == []


@pytest.mark.parametrize("reverse", [False, True])
def test_prompt_exposes_citable_trial_qualified_evidence(monkeypatch, reverse):
    first, second = _reward(), _reward()
    first["trial_id"], second["trial_id"] = "case-a__one", "case-b__one"
    second["entry_id"] = "case-b"
    second["details"]["goal_accuracy"]["evidence_refs"][0]["excerpt"] = "case-b failed output"
    rewards = [second, first] if reverse else [first, second]
    key = "trials/case-b__one/trajectory.json#/steps/14"

    def fake_hub(prompt, **_kw):
        assert key in prompt
        assert "trials/case-a__one/trajectory.json#/steps/14" in prompt
        return json.dumps([{"suggestion": "Fix case-b", "dimension": "goal_accuracy", "evidence_refs": [key]}]), None

    monkeypatch.setattr("skillevaluator.tier3.eval_core.llm_judge.call_public_llm", fake_hub)
    findings = report._extract_findings(rewards)
    [suggestion] = report._generate_suggestions_structured("demo", findings, rewards)
    [ref] = suggestion["evidence_refs"]
    assert ref["source"] == "trials/case-b__one/trajectory.json"
    assert ref["excerpt"] == "case-b failed output"


@pytest.mark.parametrize("reverse", [False, True])
def test_failing_finding_only_cites_failing_cases(reverse):
    passing, failing = _reward(1.0), _reward(0.0)
    passing["entry_id"], failing["entry_id"] = "arithmetic-negative", "syllable-positive"
    passing["details"]["goal_accuracy"]["evidence_refs"][0]["excerpt"] = "Answers 56"
    failing["details"]["goal_accuracy"]["evidence_refs"][0]["excerpt"] = "Missing syllable_count"
    rewards = [failing, passing] if reverse else [passing, failing]

    finding = next(f for f in report._extract_findings(rewards) if f["metric"] == "goal_accuracy")

    assert finding["severity"] != "ok"
    assert [(ref["source"], ref["excerpt"]) for ref in finding["evidence_refs"]] == [
        ("trials/syllable-positive/trajectory.json", "Missing syllable_count")
    ]


def test_multi_trial_evidence_without_an_owner_is_rejected():
    first, second = _reward(), _reward()
    del first["entry_id"]
    del second["entry_id"]
    assert report._build_evidence_ref_lookup([first, second]) == {}
    assert all(not finding["evidence_refs"] for finding in report._extract_findings([first, second]))


def test_conflicting_refs_with_the_same_owner_are_rejected():
    first, second = _reward(), _reward()
    second["details"]["goal_accuracy"]["evidence_refs"][0]["excerpt"] = "conflicting evidence"
    for rewards in ([first, second], [second, first]):
        assert report._build_evidence_ref_lookup(rewards) == {}
        assert all(not finding["evidence_refs"] for finding in report._extract_findings(rewards))


_ARTIFACT_PATH = "/workspace/output/results.json"
_GROUND_TRUTH_KEY = "evals.json#/ground_truth"
_EXPECTED_BEHAVIOR_KEY = "evals.json#/expected_behavior/0"
_WRITE_CALL_KEY = "trajectory.json#/steps/0/tool_calls/0"


def _builder_reward(entry_id, trajectory, *, ground_truth="", expected_behavior=None, score=0.1):
    """A failing reward whose evidence refs come from the production builder."""
    refs = build_metric_evidence_refs(
        trajectory, "question", ground_truth=ground_truth, expected_behavior=expected_behavior
    )
    details = {
        metric: {"reason": f"{metric} not met", "evidence_refs": refs[metric]}
        for metric in ("accuracy", "goal_accuracy", "behavior_check")
    }
    attach_metric_evidence_refs(details, refs)
    reward = {
        "entry_id": entry_id,
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": score,
        "goal_accuracy": score,
        "behavior_check": score,
        "details": details,
    }
    return json.loads(json.dumps(reward))  # as read back from reward.json: refs are not shared objects


def _artifact_expectation_reward(entry_id="case-a"):
    return _builder_reward(
        entry_id,
        {"steps": []},
        ground_truth=f"Write {_ARTIFACT_PATH}",
        expected_behavior=[f"Writes {_ARTIFACT_PATH}"],
    )


def _file_write_reward(entry_id="case-a"):
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "write-1",
                        "function_name": "Write",
                        "arguments": {"file_path": _ARTIFACT_PATH, "content": "{}"},
                    }
                ],
            }
        ]
    }
    return _builder_reward(entry_id, trajectory, ground_truth="Produce the results file")


def _capture_prompt(monkeypatch, suggestion_refs):
    captured = {}

    def fake_hub(prompt, **_kw):
        captured["prompt"] = prompt
        return json.dumps(
            [{"suggestion": "Write the results", "dimension": "goal_accuracy", "evidence_refs": suggestion_refs}]
        ), None

    monkeypatch.setattr("skillevaluator.tier3.eval_core.llm_judge.call_public_llm", fake_hub)
    return captured


def test_builder_emits_the_variants_the_lookup_must_tolerate():
    reward = _artifact_expectation_reward()
    kinds_at = {}
    for metric_refs in reward["details"].values():
        for ref in metric_refs["evidence_refs"]:
            kinds_at.setdefault(ref["json_pointer"], set()).add(ref["kind"])

    assert kinds_at["/ground_truth"] == {"ground_truth", "expected_artifact"}
    assert kinds_at["/expected_behavior/0"] == {"expected_behavior", "expected_artifact"}
    file_kinds = {
        ref["kind"]
        for metric_refs in _file_write_reward()["details"].values()
        for ref in metric_refs["evidence_refs"]
        if ref["json_pointer"] == "/steps/0/tool_calls/0"
    }
    assert file_kinds == {"tool_call", "file_change"}


def test_artifact_expectations_keep_one_canonical_ref_per_location():
    reward = _artifact_expectation_reward()
    forward = report._build_evidence_ref_lookup([reward])

    assert set(forward) == {_GROUND_TRUTH_KEY, _EXPECTED_BEHAVIOR_KEY}
    assert forward[_GROUND_TRUTH_KEY]["kind"] == "ground_truth"
    assert forward[_EXPECTED_BEHAVIOR_KEY]["kind"] == "expected_artifact"
    assert forward[_EXPECTED_BEHAVIOR_KEY]["path"] == _ARTIFACT_PATH
    assert report._build_evidence_ref_lookup([_artifact_expectation_reward()]) == forward


def test_file_write_tool_call_keeps_one_canonical_ref():
    lookup = report._build_evidence_ref_lookup([_file_write_reward()])

    assert lookup[_WRITE_CALL_KEY]["kind"] == "tool_call"
    assert lookup[_WRITE_CALL_KEY]["path"] == _ARTIFACT_PATH


@pytest.mark.parametrize(
    ("make_reward", "expected_keys"),
    [
        (_artifact_expectation_reward, {_GROUND_TRUTH_KEY, _EXPECTED_BEHAVIOR_KEY}),
        (_file_write_reward, {_WRITE_CALL_KEY, _GROUND_TRUTH_KEY}),
    ],
    ids=["artifact_expectation", "file_write"],
)
def test_findings_keep_evidence_for_same_case_role_variants(make_reward, expected_keys):
    findings = {f["metric"]: f for f in report._extract_findings([make_reward()])}

    for metric in ("goal_accuracy", "behavior_check"):
        refs = findings[metric]["evidence_refs"]
        assert refs, metric
        assert "evidence_unavailable_reason" not in findings[metric]
        assert {report._compact_evidence_ref(ref) for ref in refs} <= expected_keys
    assert expected_keys <= {
        report._compact_evidence_ref(ref) for finding in findings.values() for ref in finding["evidence_refs"]
    }


@pytest.mark.parametrize(
    ("make_reward", "cited"),
    [
        (_artifact_expectation_reward, _EXPECTED_BEHAVIOR_KEY),
        (_file_write_reward, _WRITE_CALL_KEY),
    ],
    ids=["artifact_expectation", "file_write"],
)
def test_suggestion_prompt_and_citation_keep_same_case_role_variants(monkeypatch, make_reward, cited):
    captured = _capture_prompt(monkeypatch, [cited])
    reward = make_reward()
    findings = report._extract_findings([reward])

    [suggestion] = report._generate_suggestions_structured("demo", findings, [reward])

    assert cited in captured["prompt"]
    assert "(no evidence refs)" not in captured["prompt"]
    [ref] = suggestion["evidence_refs"]
    assert report._compact_evidence_ref(ref) == cited
    assert ref["excerpt"]


def test_role_variants_stay_scoped_to_their_own_case(monkeypatch):
    first, second = _artifact_expectation_reward("case-a"), _artifact_expectation_reward("case-b")
    lookup = report._build_evidence_ref_lookup([first, second])
    scoped = {
        f"trials/{case}/{key}" for case in ("case-a", "case-b") for key in (_GROUND_TRUTH_KEY, _EXPECTED_BEHAVIOR_KEY)
    }

    assert set(lookup) == scoped
    captured = _capture_prompt(monkeypatch, [f"trials/case-b/{_EXPECTED_BEHAVIOR_KEY}"])
    findings = report._extract_findings([first, second])
    [suggestion] = report._generate_suggestions_structured("demo", findings, [first, second])

    assert all(key in captured["prompt"] for key in scoped)
    assert [ref["source"] for ref in suggestion["evidence_refs"]] == ["trials/case-b/evals.json"]


def test_role_variants_claimed_by_different_owners_are_still_rejected():
    first, second = _artifact_expectation_reward("same-case"), _artifact_expectation_reward("same-case")

    for rewards in ([first, second], [second, first]):
        assert report._build_evidence_ref_lookup(rewards) == {}
        assert all(not finding["evidence_refs"] for finding in report._extract_findings(rewards))


def test_role_variants_with_different_content_are_still_rejected():
    reward = _artifact_expectation_reward()
    clashing = next(
        ref
        for ref in reward["details"]["behavior_check"]["evidence_refs"]
        if ref["kind"] == "expected_artifact" and ref["json_pointer"] == "/ground_truth"
    )
    clashing["excerpt"] = "different text at the same location"

    lookup = report._build_evidence_ref_lookup([reward])

    assert _GROUND_TRUTH_KEY not in lookup
    assert _EXPECTED_BEHAVIOR_KEY in lookup


def test_same_kind_conflicts_with_one_owner_are_still_rejected():
    reward = _artifact_expectation_reward()
    for ref in reward["details"]["behavior_check"]["evidence_refs"]:
        if ref["kind"] == "expected_artifact" and ref["json_pointer"] == "/expected_behavior/0":
            ref["label"] = "Expected artifact: somewhere else"

    assert _EXPECTED_BEHAVIOR_KEY not in report._build_evidence_ref_lookup([reward])


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


def _reward_with_normalized_tool_refs():
    return {
        "entry_id": "evaluator-plugin-004",
        "goal_accuracy": 0.1,
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
        "behavior_check": 1.0,
        "details": {
            "goal_accuracy": {
                "reason": "two commands need distinct remediation",
                "evidence_refs": [
                    {
                        "source": "trajectory.json",
                        "json_pointer": "/steps/0/tool_calls/0",
                        "evidence_id": "/steps/0/tool_calls/0/normalized/0",
                        "kind": "tool_call",
                        "excerpt": "first command",
                    },
                    {
                        "source": "trajectory.json",
                        "json_pointer": "/steps/0/tool_calls/0",
                        "evidence_id": "/steps/0/tool_calls/0/normalized/1",
                        "kind": "tool_call",
                        "excerpt": "second command",
                    },
                ],
            },
        },
    }


def _reward_with_path_only_refs():
    return {
        "entry_id": "evaluator-plugin-005",
        "goal_accuracy": 0.1,
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
        "behavior_check": 1.0,
        "details": {
            "goal_accuracy": {
                "reason": "two artifacts need distinct remediation",
                "evidence_refs": [
                    {
                        "source": "artifact.txt",
                        "path": f"results/{name}.json",
                        "kind": "artifact",
                        "excerpt": f"{name} artifact",
                    }
                    for name in ("first", "second")
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


def test_findings_best_agent_matches_canonical_dimension_ranking_and_suggestion_identity(tmp_path, monkeypatch):
    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload

    standard_scores = dict(zip(DEFAULT_METRICS, (1.0, 1.0, 1.0, 1.0, 0.0, 0.0), strict=True))
    standard_reward = _reward(0.0)
    standard_reward.update(
        {
            "entry_id": "standard-case",
            "metric_set": DEFAULT_METRIC_SET,
            "behavior_check": 0.0,
        }
    )
    custom_reward = {
        "entry_id": "custom-case",
        "metric_set": CUSTOM_ONLY_METRIC_SET,
        "overall": 0.75,
        "custom_metrics": {"domain_quality": 0.75},
        "custom_details": {"domain_quality": {"reason": "custom evidence"}},
    }
    specs = {
        "standard-agent": (standard_scores, {}, list(DEFAULT_METRICS), 4 / 6, standard_reward),
        "custom-agent": ({}, {"domain_quality": 0.75}, [], 0.75, custom_reward),
    }
    live_agents = {}
    for agent, (scores, custom_scores, metrics, overall, reward) in specs.items():
        condition_dir = tmp_path / agent / "with-skill"
        trial_dir = condition_dir / "trials" / f"{reward['entry_id']}__attempt"
        trial_dir.mkdir(parents=True)
        (condition_dir / "summary.json").write_text(
            json.dumps(
                {
                    "agent": agent,
                    "scores": scores,
                    "custom_scores": custom_scores,
                    "overall_score": overall,
                    "metrics": metrics,
                    "execution_status": "succeeded",
                    "execution_errors": [],
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                    "num_trials": 1,
                    "num_reward_rows": 1,
                }
            ),
            encoding="utf-8",
        )
        (trial_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
        live_agents[agent] = {
            "execution_status": "succeeded",
            "with_skill": scores,
            "custom_with_skill": custom_scores,
            "conditions": {"with_skill": {"execution_status": "succeeded"}},
        }

    loaded_agents = report.report_data.load_agent_data(tmp_path)
    canonical = build_agent_eval_payload(
        "demo",
        loaded_agents,
        use_llm_judge=False,
    )
    assert canonical is not None
    assert canonical["agents"]["standard-agent"]["with_skill"] == 0.8
    assert canonical["agents"]["custom-agent"]["with_skill"] == 0.75
    assert canonical["best_agent"] == "standard-agent"
    assert report._pick_best_agent(live_agents, loaded_agents) == canonical["best_agent"]

    selected_reward_ids = []

    def fake_suggestions(_skill, _findings, rewards):
        selected_reward_ids.extend(reward["entry_id"] for reward in rewards)
        return [{"suggestion": "Fix the standard case.", "dimension": "effectiveness", "evidence_refs": []}]

    monkeypatch.setattr(report, "_generate_suggestions_structured", fake_suggestions)
    report.display_findings_report(
        {"agents": live_agents},
        "demo",
        ["custom-agent", "standard-agent"],
        tmp_path,
    )

    assert selected_reward_ids == ["standard-case"]
    standard_artifact = json.loads((tmp_path / "standard-agent" / "findings.json").read_text(encoding="utf-8"))
    custom_artifact = json.loads((tmp_path / "custom-agent" / "findings.json").read_text(encoding="utf-8"))
    assert standard_artifact["suggestions_v2"][0]["suggestion"] == "Fix the standard case."
    assert custom_artifact["suggestions_v2"] == []
    assert custom_artifact["suggestion_mode"] == "not_generated"


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


def test_suggestions_drop_evidence_refs_the_run_did_not_record(monkeypatch):
    """A cited ref that the run did not record is dropped instead of rendered as evidence."""
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "Fix something", "dimension": "goal_accuracy",'
            ' "evidence_refs": ["trajectory.json#/steps/999", "trajectory.json#/steps/14",'
            ' "trajectory.json#/steps/14", {"source": "made-up.json", "json_pointer": "/x"}]}]',
            None,
        ),
    )
    reward = _reward_multi_metric(0.1)
    findings = report._extract_findings([reward])
    [obj] = report._generate_suggestions_structured("demo", findings, [reward])
    assert [ref["json_pointer"] for ref in obj["evidence_refs"]] == ["/steps/14"]
    assert obj["evidence_refs"][0]["kind"] == "tool_call"


@pytest.mark.parametrize(
    "raw_refs",
    [
        ["trajectory.json#/steps/14"],
        {"source": "trajectory.json", "json_pointer": "/steps/14", "kind": "tool_call", "excerpt": "submit ..."},
        "trajectory.json#/steps/14",
    ],
)
def test_suggestions_resolve_legacy_evidence_ref_shapes(monkeypatch, raw_refs):
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "Fix evaluator job", "dimension": "goal_accuracy",'
            ' "evidence_refs": ["trajectory.json#/steps/14"]}]',
            None,
        ),
    )
    reward = _reward_multi_metric(0.1)
    reward["details"]["goal_accuracy"]["evidence_refs"] = raw_refs
    findings = report._extract_findings([reward])
    [suggestion] = report._generate_suggestions_structured("demo", findings, [reward])
    assert suggestion["evidence_refs"][0]["source"] == "trajectory.json"
    assert suggestion["evidence_refs"][0]["json_pointer"] == "/steps/14"
    if isinstance(raw_refs, dict):
        assert suggestion["evidence_refs"][0]["kind"] == "tool_call"
        assert suggestion["evidence_refs"][0]["excerpt"] == "submit ..."


def test_suggestions_keep_one_copy_of_a_repeated_suggestion(monkeypatch):
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "Install jq in evals/environment/Dockerfile.", "dimension": "goal_accuracy"},'
            ' {"suggestion": "install jq in  evals/environment/Dockerfile.", "dimension": "security"},'
            ' "Document the output format in SKILL.md."]',
            None,
        ),
    )
    reward = _reward_multi_metric(0.1)
    objs = report._generate_suggestions_structured("demo", report._extract_findings([reward]), [reward])
    assert [obj["suggestion"] for obj in objs] == [
        "Install jq in evals/environment/Dockerfile.",
        "Document the output format in SKILL.md.",
    ]


def test_suggestion_prompt_guards_requirements_and_labels_hypotheses(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda prompt, **_kw: (prompts.append(prompt), ("[]", None))[1],
    )
    reward = _reward_multi_metric(0.1)
    report._generate_suggestions_structured("demo", report._extract_findings([reward]), [reward])
    [prompt] = prompts
    assert "Do not suggest loosening, removing, or rewriting an eval expectation just so the agent passes" in prompt
    assert 'start the suggestion with "Hypothesis:"' in prompt
    assert "Give distinct suggestions" in prompt


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


@pytest.mark.parametrize("metric_set", [DEFAULT_METRIC_SET, CUSTOM_ONLY_METRIC_SET])
def test_suggestions_evidence_lookup_prefers_custom_detail_for_custom_metric_collision(monkeypatch, metric_set):
    compact_ref = "trajectory.json#/steps/8"
    reward = {
        "entry_id": "custom-case",
        "metric_set": metric_set,
        "overall": 0.2,
        "custom_metrics": {"domain_quality": 0.2},
        "details": {
            "domain_quality": {
                "reason": "stale ordinary detail",
                "evidence_refs": [
                    {
                        "source": "trajectory.json",
                        "json_pointer": "/steps/8",
                        "kind": "stale",
                        "excerpt": "stale evidence",
                    }
                ],
            }
        },
        "custom_details": {
            "domain_quality": {
                "reason": "authoritative custom detail",
                "evidence_refs": [
                    {
                        "source": "trajectory.json",
                        "json_pointer": "/steps/8",
                        "kind": "custom_evidence",
                        "excerpt": "authoritative evidence",
                    }
                ],
            }
        },
    }
    if metric_set == DEFAULT_METRIC_SET:
        reward.update(dict.fromkeys(DEFAULT_METRICS, 1.0))
    monkeypatch.setattr(
        "skillevaluator.tier3.eval_core.llm_judge.call_public_llm",
        lambda _prompt, **_kw: (
            '[{"suggestion": "Fix custom scoring", "dimension": "domain_quality", '
            f'"evidence_refs": ["{compact_ref}"]}}]',
            None,
        ),
    )

    findings = report._extract_findings([reward])
    suggestion = report._generate_suggestions_structured("demo", findings, [reward])[0]

    assert next(item for item in findings if item["metric"] == "domain_quality")["reasons"] == [
        "authoritative custom detail"
    ]
    assert suggestion["evidence_refs"] == [
        {
            "source": "trajectory.json",
            "json_pointer": "/steps/8",
            "kind": "custom_evidence",
            "excerpt": "authoritative evidence",
        }
    ]


def test_normalized_tool_evidence_refs_remain_distinct_and_resolve_by_compact_identity(monkeypatch):
    compact_ref = "trajectory.json#/steps/0/tool_calls/0/normalized/1"
    captured = {}

    def fake_hub(prompt, **_kw):
        captured["prompt"] = prompt
        return (
            f'[{{"suggestion": "Fix the second command", "dimension": "goal_accuracy", "evidence_refs": ["{compact_ref}"]}}]',
            None,
        )

    monkeypatch.setattr("skillevaluator.tier3.eval_core.llm_judge.call_public_llm", fake_hub)
    reward = _reward_with_normalized_tool_refs()
    findings = report._extract_findings([reward])

    assert len(findings[0]["evidence_refs"]) == 2
    result = report._generate_suggestions_structured("demo", findings, [reward])

    assert compact_ref in captured["prompt"]
    assert result[0]["evidence_refs"] == [
        {
            "source": "trajectory.json",
            "json_pointer": "/steps/0/tool_calls/0",
            "evidence_id": "/steps/0/tool_calls/0/normalized/1",
            "kind": "tool_call",
            "excerpt": "second command",
        }
    ]


def test_whitespace_evidence_ids_fall_back_before_deduplication():
    reward = _reward(0.1)
    reward["details"]["goal_accuracy"]["evidence_refs"] = [
        {
            "source": "trajectory.json",
            "evidence_id": whitespace,
            "json_pointer": pointer,
            "kind": "tool_call",
        }
        for whitespace, pointer in (("   ", "/steps/1"), ("\t", "/steps/2"))
    ]

    findings = report._extract_findings([reward])

    assert [ref["json_pointer"] for ref in findings[0]["evidence_refs"]] == ["/steps/1", "/steps/2"]


def test_path_only_evidence_refs_remain_distinct_in_prompt_and_lookup(monkeypatch):
    compact_ref = "artifact.txt#results/second.json"
    captured = {}

    def fake_hub(prompt, **_kw):
        captured["prompt"] = prompt
        return (
            f'[{{"suggestion": "Fix the second artifact", "dimension": "goal_accuracy", '
            f'"evidence_refs": ["{compact_ref}"]}}]',
            None,
        )

    monkeypatch.setattr("skillevaluator.tier3.eval_core.llm_judge.call_public_llm", fake_hub)
    reward = _reward_with_path_only_refs()
    findings = report._extract_findings([reward])

    assert len(findings[0]["evidence_refs"]) == 2
    result = report._generate_suggestions_structured("demo", findings, [reward])

    assert "artifact.txt#results/first.json" in captured["prompt"]
    assert compact_ref in captured["prompt"]
    assert result[0]["evidence_refs"] == [
        {
            "source": "artifact.txt",
            "path": "results/second.json",
            "kind": "artifact",
            "excerpt": "second artifact",
        }
    ]


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
