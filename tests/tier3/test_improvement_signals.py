# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from skillevaluator.tier3.improvement_signals import (
    _attempt,
    _runtime_context,
    _verification_context,
    compare_signals,
    comparison_markdown,
    load_signals,
)


def signals(score=0.0):
    return {
        "run_dir": "run",
        "dataset_digest": "frozen",
        "contract": {"grader": "frozen"},
        "execution_status": "succeeded",
        "agents": {
            "opencode": {
                "model": "fixed",
                "execution_status": "succeeded",
                "aggregate": {},
                "cases": [
                    {
                        "entry_id": "repair",
                        "split": "selection",
                        "baseline_attempts": [],
                        "attempts": [
                            {
                                "scores": {"artifact_correct": score, "accuracy": score},
                                "findings": [],
                                "judges": {"accuracy": "fixed-judge"},
                            }
                        ],
                    }
                ],
            }
        },
    }


def test_actual_outcomes_and_proxy_scores_have_distinct_verdicts():
    before, after = signals(0), signals(1)
    assert compare_signals(before, after)["verdict"] == "signal_improvement"
    result = compare_signals(before, after, outcome_metrics=("artifact_correct",))
    assert result["verdict"] == "outcome_improvement"
    assert result["agents"]["opencode"]["cases"][0]["metrics"]["artifact_correct"]["delta"] == 1
    assert compare_signals(after, before)["verdict"] == "regression"
    assert compare_signals(after, after)["verdict"] == "no_demonstrated_benefit"


def test_outcome_gain_remains_visible_alongside_proxy_regression():
    before, after = signals(0), signals(1)
    before["agents"]["opencode"]["cases"][0]["attempts"][0]["scores"]["accuracy"] = 1
    after["agents"]["opencode"]["cases"][0]["attempts"][0]["scores"]["accuracy"] = 0.9
    result = compare_signals(before, after, outcome_metrics=("artifact_correct",))
    assert result["verdict"] == "regression"
    case = result["agents"]["opencode"]["cases"][0]
    assert case["outcome_gains"] == ["artifact_correct"]
    assert case["outcome_regressions"] == []
    assert case["other_metric_declines"] == ["accuracy"]
    assert "Outcome gains: artifact_correct; outcome regressions: none. Other metric declines: accuracy." in comparison_markdown(result)
    assert "Outcome assessment unavailable" in comparison_markdown(compare_signals(before, after))


@pytest.mark.parametrize("change", ["dataset", "grader", "missing_case", "incomplete", "judge", "attempts", "metric"])
def test_incompatible_or_incomplete_evidence_abstains(change):
    before, after = signals(), signals(1)
    agent = after["agents"]["opencode"]
    if change == "dataset":
        after["dataset_digest"] = "changed"
    elif change == "grader":
        after["contract"]["grader"] = "easier"
    elif change == "missing_case":
        agent["cases"] = []
    elif change == "incomplete":
        agent["execution_status"] = "failed"
    elif change == "judge":
        agent["cases"][0]["attempts"][0]["judges"]["accuracy"] = "other"
    elif change == "attempts":
        agent["cases"][0]["attempts"] *= 2
    else:
        del agent["cases"][0]["attempts"][0]["scores"]["artifact_correct"]
    result = compare_signals(before, after, outcome_metrics=("artifact_correct",))
    assert result["verdict"] == "abstain"
    assert result["issues"]
    if change == "metric":
        assert result["agents"]["opencode"]["cases"][0]["outcome_gains"] is None


def test_one_regression_cannot_be_hidden_by_improved_average():
    before, after = signals(), signals(1)
    second = deepcopy(before["agents"]["opencode"]["cases"][0])
    second["entry_id"] = "preserve-behavior"
    before["agents"]["opencode"]["cases"].append(deepcopy(second))
    after["agents"]["opencode"]["cases"].append(deepcopy(second))
    before["agents"]["opencode"]["cases"][1]["attempts"][0]["scores"]["artifact_correct"] = 0.1
    assert compare_signals(before, after)["verdict"] == "regression"


def test_unscored_metadata_and_nonfinite_values_never_become_edit_signals():
    result = _attempt(
        {
            "entry_id": "case",
            "trial_id": "trial",
            "accuracy": float("nan"),
            "custom_details": {"audit_evidence": {"path": "output.json"}},
            "custom_metrics": {"artifact_correct": 0},
            "details": {"accuracy": {"reason": "unscored"}},
        },
        "opencode",
    )
    assert result["scores"] == {"artifact_correct": 0}
    assert not result["findings"]


def test_partial_failures_keep_evidence_above_cli_summary_cutoff():
    from skillevaluator.tier3.harbor.report import _extract_findings

    reward = {
        "entry_id": "case", "trial_id": "trial", "accuracy": 0.8,
        "skill_execution": 0.95,
        "details": {
            "accuracy": {"score": 0.8, "criteria": {"FACTUALLY_ACCURATE": False},
                         "reason": "One claimed result contradicts the artifact."},
            "skill_execution": {"score": 0.95, "error_recovery": {
                "passed": False, "reason": "Helper failed before recovery.",
                "corrections": [{"fault": "unattributed", "error": "UnicodeEncodeError"}],
            }},
        },
    }
    assert all(f["severity"] == "ok" for f in _extract_findings([reward]))
    findings = {f["metric"]: f for f in _attempt(reward, "agent")["findings"]}
    assert findings["accuracy"]["score"] == 0.8
    assert "FACTUALLY_ACCURATE failed" in findings["accuracy"]["reasons"]
    assert any("UnicodeEncodeError" in r for r in findings["skill_execution"]["reasons"])


@pytest.mark.parametrize("padding", ["", "界" * 20000])
@pytest.mark.parametrize("credential", ["configured-private-value", 'configured-\\private-"value-界'])
def test_custom_failure_diagnostic_preserves_cause_redacts_and_bounds(padding, credential, monkeypatch):
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", credential)
    cause = "fixture 'pairs' not found"
    reward = {
        "entry_id": "case", "trial_id": "trial", "custom_metrics": {"objective_outcome": 0},
        "custom_details": {"objective_outcome": {
            "reason": "Command failed: " + "header " * 100,
            "checks": [{"passed": False, "evidence": {
                "stdout": "header " * 100 + cause,
                "api_key": "private-nested-value", "stderr": credential,
                "extra": padding,
            }}],
        }},
    }
    finding = _attempt(reward, "agent")["findings"][0]
    assert cause not in finding["reasons"][0]
    diagnostic = finding["diagnostic"]
    assert cause in diagnostic["text"]
    assert "private-nested-value" not in diagnostic["text"]
    assert credential not in diagnostic["text"] and json.dumps(credential)[1:-1] not in diagnostic["text"]
    assert diagnostic["truncated"] is bool(padding)
    assert len(json.dumps(diagnostic["text"]).encode()) - 2 <= 16 * 1024
    reward["evaluation_status"] = "failed"
    assert _attempt(reward, "agent")["findings"] == []


def test_context_freezes_nested_public_tasks_and_custom_environment(tmp_path):
    task = tmp_path / "_harbor-tasks/opencode/with/repair"
    for relative in (
        "tests/test.sh",
        "environment/helper.py",
        "environment/skills/target/SKILL.md",
        "environment/input/project/task.toml",
        "environment/input/project/environment/skills/target/fixture.txt",
        "task.toml",
        "instruction.md",
    ):
        path = task / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    original = _verification_context(tmp_path, "target")
    (task / "environment/skills/target/SKILL.md").write_text("candidate")
    assert _verification_context(tmp_path, "target") == original
    (task / "environment/helper.py").write_text("easier task")
    assert _verification_context(tmp_path, "target") != original
    assert any(name.endswith("tests/test.sh") for name in original)
    assert any(name.endswith("task.toml") for name in original)
    fixture = task / "environment/input/project/environment/skills/target/fixture.txt"
    before_fixture_change = _verification_context(tmp_path, "target")
    fixture.write_text("changed input")
    assert _verification_context(tmp_path, "target") != before_fixture_change
    cache_fixture = task / "environment/input/project/__pycache__/helper.pyc"
    cache_fixture.parent.mkdir()
    cache_fixture.write_bytes(b"frozen fixture")
    before_cache_change = _verification_context(tmp_path, "target")
    assert any(name.endswith("__pycache__/helper.pyc") for name in before_cache_change)
    cache_fixture.write_bytes(b"different fixture")
    assert _verification_context(tmp_path, "target") != before_cache_change


def test_malformed_metric_detail_remains_missing_not_not_applicable():
    attempt = _attempt({"entry_id": "case", "details": {"accuracy": None}}, "agent")
    assert "accuracy" in attempt["missing_scores"]


@pytest.mark.parametrize("diagnostic", [
    {"error": "custom_only requires numeric overall"},
    {"evaluation_errors": {"accuracy": "judge unavailable"}},
    {"evaluation_status": "failed"},
    {},
])
def test_grader_errors_are_unavailable_but_real_zero_scores_remain_scored(diagnostic):
    reward = {
        "entry_id": "case", "metric_set": "custom-only", "score_policy": "custom-overall-v1",
        "overall": 0, "custom_metrics": {"artifact_correct": 0},
        "custom_details": {"artifact_correct": {"reason": "Incorrect artifact"}},
        **diagnostic,
    }
    attempt = _attempt(reward, "agent")
    if diagnostic:
        assert attempt["evaluation_status"] == "failed"
        assert attempt["scores"] == {}
        assert attempt["findings"] == []
        assert "overall" in attempt["missing_scores"]
        for key in ("error", "evaluation_errors"):
            if key in diagnostic:
                assert attempt["evaluation_errors"][key] == diagnostic[key]
    else:
        assert attempt["scores"] == {"overall": 0, "artifact_correct": 0}
        assert attempt["findings"][0]["metric"] == "artifact_correct"


def test_custom_scores_and_unknown_overall_policies_obey_public_contract():
    reward = {
        "entry_id": "case",
        "metric_set": "custom-only",
        "accuracy": 0.25,
        "overall": 1,
        "custom_metrics": {"objective_outcome": 1},
    }
    assert _attempt(reward, "agent")["scores"] == {"overall": 1, "objective_outcome": 1}
    reward["score_policy"] = "unknown"
    attempt = _attempt(reward, "agent")
    assert attempt["scores"] == {"objective_outcome": 1}
    assert attempt["missing_scores"] == ["overall"]


def test_retained_runtime_detects_backend_values_resources_and_agent_version(tmp_path):
    job = tmp_path / "_harbor-jobs/skill-agent-with"
    trial = job / "case__attempt/result.json"
    trial.parent.mkdir(parents=True)
    trial.write_text(json.dumps({"agent_info": {"name": "opencode", "version": "1.18.33", "model_info": {"name": "original", "provider": "hub"}}}))
    config = job / "config.json"
    config.write_text(json.dumps({"environment": {"override_memory_mb": 2048, "kwargs": {"image": "pinned-a"}}}))
    original = _runtime_context(tmp_path)
    config.write_text(json.dumps({"environment": {"override_memory_mb": 4096, "kwargs": {"image": "pinned-b"}}}))
    assert _runtime_context(tmp_path) != original
    assert "pinned-a" not in json.dumps(original)
    trial.write_text(json.dumps({"agent_info": {"name": "opencode", "version": "1.18.33", "model_info": {"name": "fallback", "provider": "hub"}}}))
    assert _runtime_context(tmp_path)[job.name]["agents"] != original[job.name]["agents"]


def _write_custom_signal_run(tmp_path, authoring_status=None):
    from skillevaluator.tier3.dataset_utils import normalize_dataset_entries
    from skillevaluator.tier3.harbor.report_data import build_dataset_snapshot

    case = {"id": "case", "prompt": "Check the artifact.", "unresolved_requirements": ["Verify the expected value"]}
    if authoring_status is not None:
        case["authoring_status"] = authoring_status
    entries = normalize_dataset_entries({"skill_name": "skill", "evals": [case]})
    documents = {
        "result.json": {"skill_name": "skill", "execution_status": "succeeded"},
        "run_config.json": {"agents": {"opencode": {"model": "fixed"}}},
        "dataset_snapshot.json": build_dataset_snapshot(entries, evaluator_version="test"),
        "opencode/with-skill/summary.json": {
            "scores": {}, "execution_status": "succeeded", "num_trials": 1,
            "expected_attempts": 1, "scored_attempts": 1, "metric_set": "custom-only",
        },
        "opencode/with-skill/trials/case__one/reward.json": {
            "entry_id": "case", "metric_set": "custom-only", "score_policy": "custom-overall-v1",
            "overall": 1, "custom_metrics": {"objective_outcome": 1},
        },
        "_harbor-jobs/skill-opencode-with/config.json": {"agents": [{"model_name": "fixed"}]},
        "_harbor-jobs/skill-opencode-with/case__one/result.json": {
            "agent_info": {"name": "opencode", "version": "1.18.33"},
        },
    }
    for relative, value in documents.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    task = tmp_path / "_harbor-tasks/opencode/with/case"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text("[metadata]\n")
    (task / "tests/test.sh").write_text("exit 0\n")
    return case


@pytest.mark.parametrize("authoring_status", [None, "draft_requires_oracle_review", "qualified"])
def test_load_signals_preserves_oracle_metadata_and_blocks_explicit_drafts(tmp_path, authoring_status):
    from skillevaluator.tier3.improve import discovery_packet

    case = _write_custom_signal_run(tmp_path, authoring_status)
    loaded = load_signals(tmp_path)
    observed = discovery_packet(loaded)["agents"]["opencode"]["cases"][0]
    assert observed.get("authoring_status") == authoring_status
    assert observed["unresolved_requirements"] == case["unresolved_requirements"]
    assert observed["attempts"][0]["scores"]["objective_outcome"] == 1
    draft = authoring_status == "draft_requires_oracle_review"
    assert bool(loaded["evidence_issues"]) is draft
    if draft:
        assert "independent oracle qualification" in loaded["evidence_issues"][0]
    assert compare_signals(loaded, loaded)["verdict"] == ("abstain" if draft else "no_demonstrated_benefit")


def test_computed_token_measurements_survive_redaction_without_exposing_credentials(tmp_path):
    loaded = []
    for name, prompt in (("before", 66533), ("after", 51107)):
        run = tmp_path / name
        _write_custom_signal_run(run)
        config = run / "run_config.json"
        config.write_text(json.dumps({
            "agents": {"opencode": {"model": "fixed"}},
            "provider": {"api_key": "provider-secret", "tokens": {"access": "opaque-secret"}},
        }))
        trajectory = run / "opencode/with-skill/trials/case__one/trajectory.json"
        trajectory.write_text(json.dumps({"final_metrics": {
            "total_prompt_tokens": prompt, "total_completion_tokens": 1960,
        }}))
        loaded.append(load_signals(run))

    before, after = loaded
    aggregate = before["agents"]["opencode"]["aggregate"]
    assert "token_efficiency" not in aggregate  # The public report loader supplies no such aggregate.
    assert aggregate["resources"]["tokens"]["with_skill"] == {
        "status": "complete", "measured_trials": 1, "trials": 1, "total": 68493,
    }
    assert aggregate["resources"]["tokens"]["baseline"]["total"] is None
    assert aggregate["resources"]["tokens"]["delta"] is None
    assert before["contract"]["provider"] == {"api_key": "<redacted>", "tokens": "<redacted>"}
    assert "provider-secret" not in json.dumps(before) and "opaque-secret" not in json.dumps(before)
    comparison = compare_signals(before, after)
    assert comparison["verdict"] == "no_demonstrated_benefit"
    assert comparison["agents"]["opencode"]["aggregate"]["resources.tokens.with_skill.total"] == {
        "before": 68493, "after": 53067, "delta": -15426,
    }
    assert "resources.tokens.with_skill.total | 68493 | 53067 | -15426" in comparison_markdown(comparison)

    trajectory.write_text(json.dumps({"final_metrics": {"total_prompt_tokens": "sk-not-a-valid-counter"}}))
    missing = load_signals(run)
    assert missing["agents"]["opencode"]["aggregate"]["resources"]["tokens"]["with_skill"]["total"] is None
    assert "sk-not-a-valid-counter" not in json.dumps(missing)


@pytest.mark.parametrize("state", ["activated", "never_activated", "custom", "error"])
def test_activation_evidence_survives_export_and_comparison_without_changing_verdict(tmp_path, state):
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
    from skillevaluator.tier3.improve import discovery_packet

    _write_custom_signal_run(tmp_path)
    reward_path = tmp_path / "opencode/with-skill/trials/case__one/reward.json"
    reward = json.loads(reward_path.read_text())
    reward["details"] = {
        "skill_execution": {
            "activation": {"passed": state != "never_activated", "score": int(state != "never_activated")}
        }
    }
    if state != "custom":
        reward.update(dict.fromkeys(DEFAULT_METRICS, 1))
        reward["metric_set"] = DEFAULT_METRIC_SET
        reward.pop("score_policy")
    if state == "error":
        reward["error"] = "judge failed despite retained activation detail"
    reward_path.write_text(json.dumps(reward))
    baseline = tmp_path / "opencode/without-skill"
    baseline_reward = baseline / "trials/case__baseline/reward.json"
    baseline_reward.parent.mkdir(parents=True)
    baseline_reward.write_text(json.dumps(reward))
    (baseline / "summary.json").write_text((tmp_path / "opencode/with-skill/summary.json").read_text())

    loaded = load_signals(tmp_path)
    agent = loaded["agents"]["opencode"]
    expected = "unknown" if state in {"custom", "error"} else state
    assert agent["skill_activation"]["status"] == expected
    assert agent["cases"][0]["attempts"][0]["skill_activation"]["status"] == expected
    assert "skill_activation" not in agent["cases"][0]["baseline_attempts"][0]
    comparison = compare_signals(loaded, loaded)
    assert comparison["verdict"] == ("abstain" if state == "error" else "no_demonstrated_benefit")
    assert comparison["agents"]["opencode"]["skill_activation"] == {
        "before": agent["skill_activation"],
        "after": agent["skill_activation"],
    }
    assert f"Skill activation: **{expected}** → **{expected}**" in comparison_markdown(comparison)
    packet_agent = discovery_packet(loaded)["agents"]["opencode"]
    assert "skill_activation" not in packet_agent  # Full-run summaries may include held-out cases.
    assert packet_agent["cases"][0]["attempts"][0]["skill_activation"]["status"] == expected
    assert "skill_activation" not in packet_agent["cases"][0]["baseline_attempts"][0]


@pytest.mark.parametrize("baseline_status", ["skipped", "succeeded"])
@pytest.mark.parametrize("summary_retained", [False, True])
def test_custom_only_requires_retained_baseline_evidence_unless_skipped(tmp_path, baseline_status, summary_retained):
    _write_custom_signal_run(tmp_path)
    condition = {
        "execution_status": baseline_status,
        "expected_attempts": int(baseline_status != "skipped"),
        "scored_attempts": int(baseline_status != "skipped"),
    }
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text())
    result["agents"] = {"opencode": {"conditions": {"without_skill": condition}}}
    result_path.write_text(json.dumps(result))
    if summary_retained:
        summary = tmp_path / "opencode/without-skill/summary.json"
        summary.parent.mkdir()
        summary.write_text(json.dumps({**condition, "scores": {}, "num_trials": condition["scored_attempts"]}))

    loaded = load_signals(tmp_path)

    assert bool(loaded["evidence_issues"]) is (baseline_status != "skipped")
    if baseline_status != "skipped":
        assert "retained rewards_baseline coverage is incomplete" in loaded["evidence_issues"][0]
    assert compare_signals(loaded, loaded)["verdict"] == (
        "no_demonstrated_benefit" if baseline_status == "skipped" else "abstain"
    )


def test_agent_observations_are_opt_in_bounded_redacted_and_do_not_change_scores(tmp_path):
    from skillevaluator.tier3.harbor.report_data import load_agent_data

    _write_custom_signal_run(tmp_path)
    trial = tmp_path / "opencode/with-skill/trials/case__one"
    reward_path = trial / "reward.json"
    reward = json.loads(reward_path.read_text())
    reward["_agent_observations"] = "forged reward field"
    reward_path.write_text(json.dumps(reward))
    trajectory = trial / "trajectory.json"
    trajectory.write_text(json.dumps({"steps": [None, {"source": "agent", "step_id": 4,
        "message": "api_key=observation-secret\nI worked around a helper defect. " + "é" * 9000}, 9]}))
    baseline = tmp_path / "opencode/without-skill/trials/case__baseline"
    baseline.mkdir(parents=True)
    (baseline.parent.parent / "summary.json").write_bytes((trial.parent.parent / "summary.json").read_bytes())
    (baseline / "reward.json").write_bytes((trial / "reward.json").read_bytes())
    (baseline / "trajectory.json").write_bytes(trajectory.read_bytes())
    default = load_signals(tmp_path)
    enabled = load_signals(tmp_path, include_agent_observations=True)
    attempt = enabled["agents"]["opencode"]["cases"][0]["attempts"][0]
    observation = attempt["agent_observations"][0]
    assert "agent_observations" not in default["agents"]["opencode"]["cases"][0]["attempts"][0]
    assert "observation-secret" not in json.dumps(enabled)
    assert len(observation["text"].encode()) <= 8192 and observation["truncated"] is True
    assert observation["source"].endswith("case__one/trajectory.json")
    assert observation["step_index"] == 1 and observation["step_id"] == 4
    assert observation["status"] == "unverified_source_check_hypothesis"
    assert attempt["scores"] == default["agents"]["opencode"]["cases"][0]["attempts"][0]["scores"]
    assert not attempt["findings"]
    loaded = load_agent_data(tmp_path, include_agent_observations=True)
    assert "_agent_observations" not in loaded["opencode"]["rewards_baseline"][0]
    for malformed in (3, [None, {"source": "agent", "message": {"invalid": "shape"}}]):
        trajectory.write_text(json.dumps({"steps": malformed}))
        assert "agent_observations" not in load_signals(tmp_path, include_agent_observations=True)["agents"]["opencode"]["cases"][0]["attempts"][0]


@pytest.mark.parametrize("padding", [0, 8186])
def test_opt_in_observations_redact_configured_values_before_byte_clipping(tmp_path, monkeypatch, padding):
    from skillevaluator.tier3.improve import discovery_packet

    secret = 'opaqueConfiguredCredentialWith"Quote\\Slash'
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    _write_custom_signal_run(tmp_path)
    trajectory = tmp_path / "opencode/with-skill/trials/case__one/trajectory.json"
    trajectory.write_text(json.dumps({"steps": [{"source": "agent", "step_id": secret,
        "message": "x" * padding + secret + " observed in output."}]}))
    signals = load_signals(tmp_path, include_agent_observations=True)
    observation = signals["agents"]["opencode"]["cases"][0]["attempts"][0]["agent_observations"][0]
    assert observation["step_id"] is None
    assert "opaque" not in observation["text"]
    assert len(observation["text"].encode()) <= 8192
    packet = discovery_packet(signals, include_agent_observations=True)
    assert "opaque" not in json.dumps(packet)


@pytest.mark.parametrize("split", ["selection", "holdout", "heldout", "held-out", "test", "validation", "val"])
def test_reserved_case_observations_are_not_exported(tmp_path, split):
    from skillevaluator.tier3.harbor.report_data import build_dataset_snapshot

    _write_custom_signal_run(tmp_path)
    snapshot = tmp_path / "dataset_snapshot.json"
    content = json.loads(snapshot.read_text())
    content["dataset"][0]["split"] = split
    snapshot.write_text(json.dumps(build_dataset_snapshot(content["dataset"], evaluator_version="test")))
    trajectory = tmp_path / "opencode/with-skill/trials/case__one/trajectory.json"
    trajectory.write_text(json.dumps({"steps": [{"source": "agent", "message": "Private held-out diagnosis"}]}))
    loaded = load_signals(tmp_path, include_agent_observations=True)
    assert "Private held-out diagnosis" not in json.dumps(loaded)


def test_optional_observation_step_ids_are_bounded_safe_scalars(tmp_path):
    _write_custom_signal_run(tmp_path)
    trajectory = tmp_path / "opencode/with-skill/trials/case__one/trajectory.json"
    for step_id, expected in [(4, 4), ("step-4", "step-4"), (True, None), ({"nested": "id"}, None),
                              ("é" * 100, None), ("\ud800", None)]:
        trajectory.write_text(json.dumps({"steps": [{"source": "agent", "step_id": step_id,
                                                      "message": "An unverified agent report."}]}))
        loaded = load_signals(tmp_path, include_agent_observations=True)
        observation = loaded["agents"]["opencode"]["cases"][0]["attempts"][0]["agent_observations"][0]
        assert observation["step_id"] == expected and observation["step_index"] == 0
        json.dumps(loaded, ensure_ascii=False).encode("utf-8")


def test_discovery_execution_context_uses_retained_source_without_creating_findings(tmp_path):
    from skillevaluator.tier3.improve import discovery_packet

    _write_custom_signal_run(tmp_path)
    trial = tmp_path / "opencode/with-skill/trials/case__one"
    reward_path = trial / "reward.json"
    reward = json.loads(reward_path.read_text())
    reward.update(trial_id="case__one", execution_evidence={"summary": "forged evidence"})
    reward_path.write_text(json.dumps(reward))
    trajectory = {"steps": [{"source": "agent", "step_id": 2,
        "tool_calls": [{"tool_call_id": "test", "function_name": "bash", "arguments": {"command": "pytest"}}],
        "observation": {"results": [{"source_call_id": "test", "content": "FAIL original test"}]}},
        {"source": "agent", "message": "FINAL SELF REPORT MUST BE OPT IN"}]}
    (trial / "trajectory.json").write_text(json.dumps(trajectory))
    attempt = discovery_packet(load_signals(tmp_path))["agents"]["opencode"]["cases"][0]["attempts"][0]
    evidence = attempt["execution_evidence"]
    assert evidence["status"] == "available" and len(evidence["sha256"]) == 64
    assert evidence["source"] == "opencode/with-skill/trials/case__one/trajectory.json"
    assert "FAIL original test" in evidence["summary"] and "forged evidence" not in str(attempt)
    assert "FINAL SELF REPORT MUST BE OPT IN" not in str(attempt)
    assert attempt["findings"] == [] and attempt["scores"]["objective_outcome"] == 1


@pytest.mark.parametrize("source", ["missing", "malformed", "symlink", "parent_link", "traversal"])
def test_execution_evidence_rejects_untrusted_source_paths(tmp_path, source):
    from skillevaluator.tier3.improvement_signals import _execution_evidence

    root = tmp_path / "run"
    trial = root / "agent/with-skill/trials/case"
    trial.mkdir(parents=True)
    private = tmp_path / "private.json"
    private.write_text(json.dumps({"steps": [{"source": "agent", "message": "PRIVATE HELDOUT ANSWER"}]}))
    path = trial / "trajectory.json"
    if source == "malformed":
        path.write_text("{not json}")
    elif source == "symlink":
        path.symlink_to(private)
    elif source == "parent_link":
        trial.rmdir()
        trial.symlink_to(tmp_path, target_is_directory=True)
        (tmp_path / "trajectory.json").write_bytes(private.read_bytes())
    evidence = _execution_evidence(root, "agent", "../../../../private.json" if source == "traversal" else "case")
    assert evidence["status"] == "unavailable" and "PRIVATE HELDOUT ANSWER" not in str(evidence)


def test_discovery_cannot_redirect_trajectory_identity_to_holdout(tmp_path):
    _write_custom_signal_run(tmp_path)
    trial = tmp_path / "opencode/with-skill/trials/case__one"
    path = trial / "reward.json"
    reward = json.loads(path.read_text())
    reward["trial_id"] = "hidden__trial"
    path.write_text(json.dumps(reward))
    hidden = trial.parent / "hidden__trial"
    hidden.mkdir()
    (hidden / "reward.json").write_text(json.dumps({**reward, "entry_id": "hidden"}))
    (hidden / "trajectory.json").write_text(json.dumps({"steps": [{"source": "agent",
        "tool_calls": [{"tool_call_id": "read", "function_name": "read", "arguments": {"path": "oracle"}}],
        "observation": {"results": [{"source_call_id": "read", "content": "PRIVATE HOLDOUT RESULT"}]}}]}))
    case = next(case for case in load_signals(tmp_path)["agents"]["opencode"]["cases"] if case["entry_id"] == "case")
    assert case["attempts"][0]["execution_evidence"]["status"] == "unavailable"
    assert "PRIVATE HOLDOUT RESULT" not in str(case)
