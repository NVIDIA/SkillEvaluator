# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.studio.adapter import (
    is_native_custom_only,
    json_safe,
    normalize_evaluation,
    normalize_trial,
    read_json,
)


def test_failed_trial_never_exposes_diagnostic_scores_as_scored() -> None:
    case_id, trial = normalize_trial(
        {"entry_id": "case-a", "trial_id": "physical-1", "overall": 1.0, "accuracy": 1.0},
        agent="opencode",
        arm="with_skill",
        failure={"error": {"message": "agent timed out"}},
    )
    assert case_id == "case-a"
    assert trial["id"] == "physical-1"
    assert trial["status"] == "failed"
    assert trial["metrics"] == {}
    assert trial["diagnostic_metrics"]["overall"] == 1.0


def test_unknown_case_identity_and_numeric_predicates_remain_unknown() -> None:
    case_id, trial = normalize_trial(
        {
            "trial_id": "case-a-attempt1__random",
            "overall": 0.6,
            "details": {
                "accuracy": {"score": 0.6, "reason": "some criteria failed", "criteria": {"FORMAT_FOLLOWED": False}},
            },
        },
        agent="opencode",
        arm="without_skill",
    )
    assert case_id is None
    assert trial["identity_status"] == "unknown"
    assert trial["checks"][0]["status"] == "scored"
    assert trial["checks"][1]["status"] == "failed"


def test_native_flat_execution_and_efficiency_checks_remain_actionable() -> None:
    reward = {
        "overall": 0.7,
        "details": {
            "skill_execution": {
                "activation": {"passed": True, "score": 1, "reason": "Read the target skill"},
                "script_execution": {"passed": False, "score": 0, "reason": "Expected script was not executed"},
            },
            "skill_efficiency": {"error_recovery": {"passed": False, "score": 0, "reason": "Unrecovered failure"}},
        },
    }
    _, trial = normalize_trial(
        reward, agent="opencode", arm="with_skill", evidence=[{"kind": "reward", "path": "reward.json"}]
    )
    checks = {check["name"]: check for check in trial["checks"]}
    assert checks["skill_execution: script_execution"]["status"] == "failed"
    assert checks["skill_efficiency: error_recovery"]["reason"] == "Unrecovered failure"
    assert checks["skill_execution: activation"]["evidence"][0]["json_pointer"] == "/details/skill_execution/activation"


def test_native_nested_checks_have_resolvable_escaped_json_pointers() -> None:
    reward = {
        "overall": 0,
        "details": {"metric": {"checks": {"path/name~id": {"passed": False, "reason": "Failed predicate"}}}},
    }
    _, trial = normalize_trial(
        reward, agent="opencode", arm="with_skill", evidence=[{"kind": "reward", "path": "reward.json"}]
    )
    pointer = trial["checks"][0]["evidence"][0]["json_pointer"]
    assert pointer == "/details/metric/checks/path~1name~0id"
    target = reward
    for part in pointer.lstrip("/").split("/"):
        target = target[part.replace("~1", "/").replace("~0", "~")]
    assert target["passed"] is False


def test_native_lift_is_preserved_instead_of_recomputed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    raw = {
        "execution_status": "succeeded",
        "expected_attempts": 2,
        "scored_attempts": 2,
        "agents": {
            "opencode": {
                "overall_with_skill": 0.8,
                "overall_without_skill": 0.6,
                "lift": {"overall": {"delta": -0.1, "reason": "paired judge-aware comparison"}},
                "trials": [{"entry_id": "case-a", "trial_id": "run-a", "overall": 0.8}],
                "trials_baseline": [{"entry_id": "case-a", "trial_id": "run-b", "overall": 0.6}],
            }
        },
    }
    result = normalize_evaluation(raw, job_dir=tmp_path, skill_path=source)
    assert result["summary"]["lift"] == -0.1
    assert result["summary"]["precision"] is None
    assert result["suggestions"] == []
    assert result["suggestions_status"] == "not_generated"
    assert result["optimizer"]["available"] is True
    assert [trial["id"] for trial in result["cases"][0]["arms"]["without_skill"]] == ["run-b"]


def test_missing_comparison_stays_missing_and_failed_aggregate_is_unavailable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    result = normalize_evaluation(
        {
            "execution_status": "failed",
            "agents": {
                "codex": {
                    "overall_with_skill": 1.0,
                    "overall_without_skill": 1.0,
                    "lift": {"overall": {"delta": 0.0}},
                }
            },
        },
        job_dir=tmp_path,
        skill_path=source,
    )
    assert result["summary"]["with_skill"] is None
    assert result["summary"]["lift"] is None
    assert result["summary"]["expected_attempts"] is None


def test_contained_native_trials_keep_failures_final_response_and_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    run = tmp_path / "results" / "skill" / "run"
    good = run / "opencode" / "with-skill" / "trials" / "case-a__good"
    failed = run / "opencode" / "without-skill" / "trials" / "case-a__failed"
    good.mkdir(parents=True)
    failed.mkdir(parents=True)
    (good / "reward.json").write_text(
        json.dumps(
            {
                "entry_id": "case-a",
                "trial_id": "physical-a",
                "overall": 1,
                "accuracy": 1,
                "details": {"accuracy": {"score": 1, "reason": "valid", "criteria": {"TASK_ADDRESSED": True}}},
            }
        )
    )
    (good / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "assistant", "message": "Answer", "tool_calls": []},
                ]
            }
        )
    )
    (failed / "failure.json").write_text(
        json.dumps({"status": "unscored", "trial": "failure-record", "error": {"message": "timeout"}})
    )
    (run / "result.json").write_text("{}")
    (run / "opencode" / "findings.json").write_text(
        json.dumps(
            {
                "suggestion_mode": "passing_next_steps",
                "suggestions_v2": [{"suggestion": "Expand coverage", "evidence_refs": []}],
            }
        )
    )
    raw = {"execution_status": "failed", "run_dir": str(run), "agents": {"opencode": {}}}
    result = normalize_evaluation(raw, job_dir=tmp_path, skill_path=source)
    known = next(case for case in result["cases"] if case["id"] == "case-a")
    trial = known["arms"]["with_skill"][0]
    assert trial["final_response"] == "Answer"
    assert trial["id"] == "physical-a"
    assert trial["checks"][0]["evidence"][0]["path"].endswith("reward.json")
    unknown = next(case for case in result["cases"] if case["id"] is None)
    assert unknown["arms"]["without_skill"][0]["status"] == "failed"
    assert result["suggestions"][0]["mode"] == "passing_next_steps"
    assert all(not Path(item["path"]).is_absolute() for item in result["artifacts"])
    assert result["improvement_signals"] is None
    assert len(result["notes"]) == 1 and result["notes"][0].startswith("Improvement evidence is unavailable:")


def test_artifact_reader_refuses_symlink_and_nonfinite_values(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked.json").symlink_to(outside)
    with pytest.raises(ValueError):
        read_json(root / "linked.json", root=root)
    (root / "nan.json").write_text('{"score":NaN}')
    with pytest.raises(ValueError, match="Non-finite"):
        read_json(root / "nan.json", root=root)


def test_browser_json_masks_actual_key_and_nonfinite_measurements() -> None:
    result = json_safe(
        {"message": "arbitrary-key-value", "score": float("nan"), "valid": False},
        secret_values=("arbitrary-key-value",),
    )
    assert result == {"message": "<redacted>", "score": None, "valid": False}


def test_native_custom_only_detection_requires_the_authored_contract(tmp_path: Path) -> None:
    evals = tmp_path / "evals"
    (evals / "harbor").mkdir(parents=True)
    assert is_native_custom_only(tmp_path) is False
    config = "schema_version: 1\nharbor:\n  task_source: native_harbor\ngrading:\n  mode: custom_only\n"
    (evals / "config.yml").write_text(config)
    assert is_native_custom_only(tmp_path) is True
    (evals / "config.yml").write_text(config.replace("custom_only", "default"))
    assert is_native_custom_only(tmp_path) is False


def test_unknown_native_score_policy_is_not_presented_as_a_scored_trial() -> None:
    _, trial = normalize_trial(
        {"overall": 1, "score_policy": "unknown-future-policy"}, agent="opencode", arm="with_skill"
    )
    assert trial["status"] == "unscored"
    assert trial["metrics"] == {}
    assert trial["diagnostic_metrics"]["overall"] == 1


def test_zero_custom_task_score_is_preserved_with_a_grader_explanation() -> None:
    _, trial = normalize_trial(
        {"overall": 0, "error": "The requested artifact was missing"}, agent="opencode", arm="with_skill"
    )
    assert trial["status"] == "scored"
    assert trial["metrics"]["overall"] == 0
    assert trial["error"] == "The requested artifact was missing"


@pytest.mark.parametrize("source", ["agent", "assistant"])
def test_atif_final_response_supports_native_agent_and_assistant_sources(source: str) -> None:
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {"step_id": 1, "source": "user", "message": "Merge the PDF files."},
            {
                "step_id": 2,
                "source": source,
                "message": "Running the merge.",
                "tool_calls": [{"function_name": "bash", "arguments": {"command": "python merge.py"}}],
            },
            {
                "step_id": 3,
                "source": source,
                "message": [
                    {"type": "text", "text": "Created merged.pdf."},
                    {"type": "text", "text": "The output has three pages."},
                ],
                "tool_calls": [],
            },
            {"step_id": 4, "source": "user", "message": "This is not an agent answer."},
        ],
    }
    _, trial = normalize_trial({"overall": 1}, agent="opencode", arm="with_skill", trajectory=trajectory)
    assert trial["final_response"] == "Created merged.pdf.\nThe output has three pages."


def test_atif_tool_call_message_is_not_presented_as_a_final_response() -> None:
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "source": "agent",
                "message": "I will run the command.",
                "tool_calls": [{"function_name": "bash", "arguments": {"command": "python merge.py"}}],
            },
        ],
    }
    _, trial = normalize_trial({"overall": 1}, agent="opencode", arm="with_skill", trajectory=trajectory)
    assert trial["final_response"] is None


def test_native_resource_measurements_survive_redaction_without_exposing_tokens() -> None:
    from skillevaluator.tier3.harbor.resource_usage import resource_comparison

    measured = resource_comparison(
        [{"entry_id": "case", "tokens": {"prompt": 10, "completion": 2}}],
        [{"entry_id": "case", "tokens": {"prompt": 20, "completion": 3}}],
        with_complete=True,
        baseline_complete=True,
    )
    projected = json_safe(
        {
            "resources": measured,
            "aggregate": {"resources.tokens.delta": {"before": 2, "after": -11, "delta": -13}},
            "tokens": {"opaque": "do-not-display"},
        }
    )
    assert projected["resources"]["tokens"] == measured["tokens"]
    assert projected["aggregate"]["resources.tokens.delta"]["delta"] == -13
    assert projected["tokens"] == "<redacted>"
