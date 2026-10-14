# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import skillevaluator.tier3.harbor.runner as harbor_runner
from skillevaluator import __version__
from skillevaluator.tier3.harbor import collector as harbor_collector
from skillevaluator.tier3.harbor import report_data

_SNAPSHOT_LIMIT_ERROR = (
    "Dataset snapshot exceeds the 2 MiB, depth-64, or 50,000-node publication limit; "
    "reduce dataset size or structural complexity."
)


def _write_staged_entry(run_dir: Path, task: str, entry: dict[str, object]) -> Path:
    entry_path = run_dir / "_harbor-tasks" / task / "tests" / "entry.json"
    entry_path.parent.mkdir(parents=True)
    entry_path.write_text(json.dumps(entry, separators=(",", ":")), encoding="utf-8")
    return entry_path


def test_runner_persists_deduplicated_dataset_truth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for task, entry in (
        ("first", {"id": "same-task", "expected_skill": "demo"}),
        ("duplicate", {"id": "same-task", "expected_skill": "demo"}),
        ("negative", {"id": "negative", "expected_skill": None}),
    ):
        entry_path = run_dir / "_harbor-tasks" / task / "tests" / "entry.json"
        entry_path.parent.mkdir(parents=True)
        entry_path.write_text(json.dumps(entry), encoding="utf-8")

    snapshot = harbor_runner._persist_dataset_truth(run_dir, fallback_task_ids=[])

    assert snapshot["evaluator_version"] == __version__
    assert snapshot["dataset_summary"] == {
        "total_tasks": 2,
        "positive_tasks": 1,
        "negative_tasks": 1,
        "unclassified_tasks": 0,
        "source": "dataset",
    }
    assert snapshot["dataset_digest"].startswith("sha256:")
    assert json.loads((run_dir / "dataset_snapshot.json").read_text(encoding="utf-8")) == snapshot


def test_runner_rejects_snapshot_whose_combined_entries_exceed_publication_bytes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_staged_entry(run_dir, "first", {"id": "first", "prompt": "a" * 1_100_000})
    _write_staged_entry(run_dir, "second", {"id": "second", "prompt": "b" * 1_100_000})

    with pytest.raises(ValueError, match=rf"^{_SNAPSHOT_LIMIT_ERROR}$"):
        harbor_runner._persist_dataset_truth(run_dir, fallback_task_ids=[])

    assert not (run_dir / "dataset_snapshot.json").exists()
    assert not list(run_dir.glob(".dataset_snapshot.*"))


def test_runner_rejects_snapshot_when_wrapper_pushes_entry_over_node_limit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_staged_entry(run_dir, "case", {"id": "case", "values": [0] * 49_990})

    with pytest.raises(ValueError, match=rf"^{_SNAPSHOT_LIMIT_ERROR}$"):
        harbor_runner._persist_dataset_truth(run_dir, fallback_task_ids=[])

    assert not (run_dir / "dataset_snapshot.json").exists()


def test_runner_rejects_snapshot_when_wrapper_pushes_entry_over_depth_limit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    nested: object = "leaf"
    for _ in range(62):
        nested = {"value": nested}
    _write_staged_entry(run_dir, "case", {"id": "case", "metadata": nested})

    with pytest.raises(ValueError, match=rf"^{_SNAPSHOT_LIMIT_ERROR}$"):
        harbor_runner._persist_dataset_truth(run_dir, fallback_task_ids=[])

    assert not (run_dir / "dataset_snapshot.json").exists()


def test_runner_round_trips_near_byte_limit_snapshot_and_embeds_only_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_staged_entry(run_dir, "case", {"id": "case", "prompt": "p" * 1_900_000})

    snapshot = harbor_runner._persist_dataset_truth(run_dir, fallback_task_ids=[])
    persisted = run_dir / "dataset_snapshot.json"
    manifest = report_data.dataset_snapshot_manifest(snapshot)

    assert persisted.stat().st_size <= report_data.DATASET_SNAPSHOT_MAX_BYTES
    assert report_data.load_dataset_snapshot(run_dir) == snapshot
    assert "dataset" not in manifest
    assert manifest == {
        "schema_version": snapshot["schema_version"],
        "evaluator_version": snapshot["evaluator_version"],
        "dataset_summary": snapshot["dataset_summary"],
        "dataset_digest": snapshot["dataset_digest"],
        "dataset_digest_algorithm": snapshot["dataset_digest_algorithm"],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _max_detail_pass_summary() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for index in range(256):
        case_id = f"case-{index:03d}-" + ("c" * 500)
        cases[case_id] = {
            "passed": True,
            "first_pass_attempt": 1,
            "attempts_used": 2,
            "attempts_skipped": 0,
            "attempts_missing": 0,
            "best_score": 1.0,
            "attempts": [
                {
                    "attempt": attempt,
                    "trial": f"trial-{index:03d}-{attempt}-" + ("t" * 490),
                    "score": 1.0,
                    "passed": True,
                }
                for attempt in (1, 2)
            ],
            "attempt_details_total": 2,
            "attempt_details_shown": 2,
            "attempt_details_truncated": False,
        }
    return {
        "k": 2,
        "pass_threshold": 0.5,
        "stop_on_pass": False,
        "passed_cases": 256,
        "failed_cases": 0,
        "total_cases": 256,
        "rate": 1.0,
        "rate_interval": {"lower": 0.98, "upper": 1.0},
        "attempts_used": 512,
        "max_attempts_possible": 512,
        "avg_attempts_used": 2.0,
        "extra_case_count": 0,
        "extra_cases": [],
        "extra_cases_truncated": False,
        "case_details_total": 256,
        "case_details_shown": 256,
        "case_details_truncated": False,
        "case_details_limit": 256,
        "cases": cases,
    }


def _max_detail_result(run_dir: Path) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    pass_summary = _max_detail_pass_summary()
    metric_names = [f"metric-{index:03d}-" + ("m" * 244) for index in range(128)]
    custom_scores = dict.fromkeys(metric_names, 0.75)
    custom_lift = {name: {"with_skill": 0.75, "without_skill": 0.25, "delta": 0.5} for name in metric_names}
    failures = [
        {"trial": f"trial-{index:03d}", "reason": f"failure-{index:03d}: " + ("r" * 2_000)} for index in range(32)
    ]
    condition_errors = [failure["reason"] for failure in failures]
    security_cases = {
        f"case-{index:03d}-" + ("s" * 500): {
            "status": "with_skill_unsafe",
            "with_skill_findings": 1,
            "baseline_findings": 0,
        }
        for index in range(256)
    }
    security_attribution = {
        "likely_skill_related": 256,
        "likely_baseline_prompt_or_environment": 0,
        "skill_may_have_improved_safety": 0,
        "ambiguous_with_skill_only": 0,
        "unknown_no_baseline": 0,
        "case_details_total": 256,
        "case_details_shown": 256,
        "case_details_truncated": False,
        "case_details_limit": 256,
        "cases": security_cases,
    }
    condition = {
        "execution_status": "failed",
        "execution_errors": condition_errors,
        "execution_error_details_total": 32,
        "execution_error_details_shown": 32,
        "execution_error_details_truncated": False,
        "expected_attempts": 512,
        "scored_attempts": 480,
        "runtime_failure_details_total": 32,
        "runtime_failure_details_shown": 32,
        "runtime_failure_details_truncated": False,
        "reward_failure_details_total": 32,
        "reward_failure_details_shown": 32,
        "reward_failure_details_truncated": False,
    }
    pass_lift = {
        "with_skill": 1.0,
        "without_skill": 1.0,
        "delta": 0.0,
        "passed_cases_delta": 0,
    }
    run_config_agents: dict[str, Any] = {}
    for index in range(3):
        agent = f"agent-{index}"
        agent_dir = run_dir / agent
        model = f"model-{index}"
        source = "CLI"
        summary = {
            "agent": agent,
            "model": model,
            "model_source": source,
            "scores": {},
            "custom_scores": custom_scores,
            "overall_score": 0.75,
            "metric_set": "custom-only",
            "metrics": [],
            "dimensions": {},
            "num_trials": 480,
            "num_reward_rows": 480,
            "pass_at_k": pass_summary,
            **condition,
            "job_failure": "aggregate failure: " + ("j" * 4_000),
            "trial_failures": failures,
            "trial_failure_details_total": 32,
            "trial_failure_details_shown": 32,
            "trial_failure_details_truncated": False,
        }
        _write_json(agent_dir / "with-skill" / "summary.json", summary)
        _write_json(agent_dir / "without-skill" / "summary.json", summary)
        _write_json(agent_dir / "lift.json", {})
        _write_json(agent_dir / "custom_lift.json", custom_lift)
        _write_json(agent_dir / "pass_at_k_lift.json", pass_lift)
        _write_json(agent_dir / "security_attribution.json", security_attribution)
        agents[agent] = {
            "model": model,
            "model_source": source,
            "model_resolution": {"model": model, "source": source},
            "with_skill": {},
            "without_skill": {},
            "custom_with_skill": custom_scores,
            "custom_without_skill": custom_scores,
            "dimensions_with_skill": {},
            "dimensions_without_skill": {},
            "lift": {},
            "custom_lift": custom_lift,
            "pass_at_k": {
                "with_skill": pass_summary,
                "without_skill": pass_summary,
                "lift": pass_lift,
            },
            "security_attribution": security_attribution,
            "agent_runtime_failures": {"with_skill": failures, "without_skill": failures},
            "trial_failures": {"with_skill": failures, "without_skill": failures},
            "failure_detail_metadata": {
                key: {"details_total": 32, "details_shown": 32, "details_truncated": False}
                for key in (
                    "with_skill_runtime",
                    "without_skill_runtime",
                    "with_skill_trials",
                    "without_skill_trials",
                )
            },
            "job_failures": {"with_skill": summary["job_failure"], "without_skill": summary["job_failure"]},
            "conditions": {"with_skill": condition, "without_skill": condition},
            "execution_status": "failed",
            "execution_errors": condition_errors,
            "execution_error_details_total": 32,
            "execution_error_details_shown": 32,
            "execution_error_details_truncated": False,
            "expected_attempts": 1_024,
            "scored_attempts": 960,
            "num_trials_with": 480,
            "num_trials_without": 480,
            "output_dir": str(agent_dir.resolve()),
        }
        run_config_agents[agent] = {"agent": agent, "model": model, "source": source}

    comparison = {
        "metrics": {
            name: {agent: {"with_skill": 0.75, "without_skill": 0.25, "delta": 0.5} for agent in agents}
            for name in metric_names
        }
    }
    _write_json(run_dir / "comparison.json", comparison)
    run_config = {
        "config_file": "none",
        "harbor": {
            "environment": {"value": "docker", "source": "CLI"},
            "n_attempts": 2,
            "stop_on_pass": False,
            "n_concurrent": 3,
            "timeout_multiplier": 1.0,
            "base_image_mode": "disabled",
            "jobs_retained": True,
        },
        "provider": {"name": "openai", "model": "test-model"},
        "task_source": "evals_json",
        "grading": {"mode": "custom_only"},
        "agents": run_config_agents,
    }
    return {
        "skill_name": "demo",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "result_path": str(run_dir / "result.json"),
        "run_config": run_config,
        "agents": agents,
        "comparison": comparison,
        "metric_set": "custom-only",
        "metrics": [],
        "attempt_policy": {
            "max_attempts": 2,
            "pass_threshold": 0.5,
            "stop_on_pass": False,
            "score_definition": "mean custom metrics",
        },
        "execution_status": "failed",
        "execution_errors": condition_errors,
        "execution_error_details_total": 32,
        "execution_error_details_shown": 32,
        "execution_error_details_truncated": False,
        "error": condition_errors[:1],
        "evaluator_version": __version__,
        "dataset_snapshot": {
            "schema_version": "1.0",
            "evaluator_version": __version__,
            "dataset_summary": {"total_tasks": 256},
            "dataset_digest": "sha256:" + ("d" * 64),
            "dataset_digest_algorithm": "skill-evaluator-dataset-snapshot/1",
        },
        "dataset_snapshot_path": str(run_dir / "dataset_snapshot.json"),
        "dataset_summary": {"total_tasks": 256},
        "dataset_digest": "sha256:" + ("d" * 64),
        "dataset_digest_algorithm": "skill-evaluator-dataset-snapshot/1",
        "report_status": "complete",
        "duration_seconds": 1.0,
    }


def test_final_result_projects_three_max_detail_agents_to_bounded_artifact_references(tmp_path: Path) -> None:
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory

    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "20260825_120000_123_aaaaaaaaaaaa"
    run_dir.mkdir()
    (run_dir / "_harbor-jobs").mkdir()
    (run_dir / "_harbor-tasks").mkdir()
    result = _max_detail_result(run_dir)
    _write_json(run_dir / "run_config.json", result["run_config"])
    _write_json(run_dir / "result.json", {})
    assert len(json.dumps(result, indent=2).encode("utf-8")) > 2 * 1024 * 1024

    harbor_runner._finalize_harbor_artifacts(
        run_dir_value=run_dir,
        keep_requested=True,
        result=result,
    )

    result_path = run_dir / "result.json"
    encoded = result_path.read_bytes()
    persisted = json.loads(encoded)
    assert len(encoded) <= 2 * 1024 * 1024
    assert result["agents"]["agent-0"]["pass_at_k"]["with_skill"]["cases"]
    assert persisted["agents"]["agent-0"]["pass_at_k"]["with_skill"]["cases"] == {}
    assert len(persisted["agents"]["agent-0"]["custom_with_skill"]) == 128
    assert persisted["agents"]["agent-0"]["expected_attempts"] == 1_024
    assert persisted["agents"]["agent-0"]["conditions"]["with_skill"]["execution_error_details_total"] == 32
    assert persisted["agents"]["agent-0"]["security_attribution"]["case_details_total"] == 256
    assert persisted["execution_errors"]
    assert persisted["error"] == persisted["execution_errors"][:1]
    assert persisted["dataset_digest"] == result["dataset_digest"]
    assert persisted["result_path"] == result["result_path"]
    projection = persisted["result_projection"]
    assert projection == result["result_projection"]
    assert projection["schema_version"] == "1.0"
    assert projection["mode"] == "artifact_referenced"
    reference = projection["agents"]["agent-0"]["with_skill_summary"]
    referenced = run_dir / reference["path"]
    assert reference["bytes"] == referenced.stat().st_size
    assert reference["sha256"] == "sha256:" + hashlib.sha256(referenced.read_bytes()).hexdigest()
    diagnostics: list[dict[str, Any]] = []
    assert report_data._load_bounded_json(result_path, diagnostics, artifact="result") == persisted
    assert diagnostics == []
    regenerated = agent_eval_result_from_directory(
        skill,
        run_dir,
        engine_result=None,
        use_llm_judge=False,
    )
    assert regenerated is not None
    assert regenerated.metadata["agent_eval"]["execution_status"] == "failed"


def test_inline_final_result_round_trips_normal_agent_artifacts_for_disk_report(
    tmp_path: Path,
) -> None:
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory

    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "20260825_120000_123_bbbbbbbbbbbb"
    run_dir.mkdir()
    agent_dir = run_dir / "opencode"
    summary = {
        "agent": "opencode",
        "model": "test-model",
        "model_source": "CLI",
        "scores": {"overall": 1.0},
        "custom_scores": {"quality": 0.75},
        "overall_score": 1.0,
        "metric_set": "skill-evaluator-default-v2",
        "metrics": ["overall"],
        "dimensions": {},
        "num_trials": 1,
        "num_reward_rows": 1,
        "pass_at_k": {
            "k": 1,
            "pass_threshold": 0.5,
            "stop_on_pass": False,
            "passed_cases": 1,
            "failed_cases": 0,
            "total_cases": 1,
            "rate": 1.0,
            "attempts_used": 1,
            "max_attempts_possible": 1,
            "cases": {"case-1": {"passed": True, "attempts": []}},
        },
        "execution_status": "succeeded",
        "execution_errors": [],
        "execution_error_details_total": 0,
        "execution_error_details_shown": 0,
        "execution_error_details_truncated": False,
        "expected_attempts": 1,
        "scored_attempts": 1,
        "job_failure": "",
        "trial_failures": [],
    }
    _write_json(agent_dir / "with-skill" / "summary.json", summary)
    baseline = {**summary, "execution_status": "skipped", "expected_attempts": 0, "scored_attempts": 0}
    _write_json(agent_dir / "without-skill" / "summary.json", baseline)
    reward_dir = agent_dir / "with-skill" / "trials" / "case-1_attempt001"
    _write_json(
        reward_dir / "reward.json",
        {
            "entry_id": "case-1",
            "overall": 1.0,
            "metric_set": "skill-evaluator-default-v2",
            "metrics": ["overall"],
        },
    )
    run_config = {
        "config_file": "none",
        "harbor": {
            "environment": {"value": "docker", "source": "CLI"},
            "n_attempts": 1,
            "stop_on_pass": False,
            "n_concurrent": 1,
            "timeout_multiplier": 1.0,
            "base_image_mode": "disabled",
            "jobs_retained": True,
        },
        "provider": {"name": "openai", "model": "test-model"},
        "task_source": "evals_json",
        "grading": {"mode": "default_plus_custom"},
        "agents": {"opencode": {"agent": "opencode", "model": "test-model", "source": "CLI"}},
    }
    result = {
        "skill_name": skill.name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "result_path": str(run_dir / "result.json"),
        "run_config": run_config,
        "agents": {
            "opencode": {
                "model": "test-model",
                "model_source": "CLI",
                "model_resolution": {"model": "test-model", "source": "CLI"},
                "with_skill": summary["scores"],
                "without_skill": {},
                "custom_with_skill": summary["custom_scores"],
                "custom_without_skill": {},
                "dimensions_with_skill": {},
                "dimensions_without_skill": {},
                "lift": {},
                "custom_lift": {},
                "pass_at_k": {"with_skill": summary["pass_at_k"], "without_skill": {}, "lift": {}},
                "security_attribution": {},
                "agent_runtime_failures": {"with_skill": [], "without_skill": []},
                "trial_failures": {"with_skill": [], "without_skill": []},
                "failure_detail_metadata": {},
                "job_failures": {"with_skill": "", "without_skill": ""},
                "conditions": {
                    "with_skill": {
                        "execution_status": "succeeded",
                        "execution_errors": [],
                        "execution_error_details_total": 0,
                        "expected_attempts": 1,
                        "scored_attempts": 1,
                    },
                    "without_skill": {
                        "execution_status": "skipped",
                        "execution_errors": [],
                        "execution_error_details_total": 0,
                        "expected_attempts": 0,
                        "scored_attempts": 0,
                    },
                },
                "output_dir": str(agent_dir.resolve()),
                "execution_status": "succeeded",
                "execution_errors": [],
                "execution_error_details_total": 0,
                "execution_error_details_shown": 0,
                "execution_error_details_truncated": False,
                "expected_attempts": 1,
                "scored_attempts": 1,
                "num_trials_with": 1,
                "num_trials_without": 0,
            }
        },
        "metric_set": "skill-evaluator-default-v2",
        "metrics": ["overall"],
        "attempt_policy": {
            "max_attempts": 1,
            "pass_threshold": 0.5,
            "stop_on_pass": False,
            "score_definition": "overall",
        },
        "execution_status": "succeeded",
        "execution_errors": [],
        "error": [],
        "report_status": "complete",
        "duration_seconds": 1.0,
    }
    _write_json(run_dir / "run_config.json", run_config)
    _write_json(run_dir / "result.json", {})

    harbor_runner._finalize_harbor_artifacts(
        run_dir_value=run_dir,
        keep_requested=True,
        result=result,
    )

    persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    loaded_agents = report_data.load_agent_data(run_dir)
    report_result = agent_eval_result_from_directory(
        skill,
        run_dir,
        engine_result=None,
        use_llm_judge=False,
    )
    assert persisted == result
    assert persisted["result_projection"]["mode"] == "inline"
    assert loaded_agents["opencode"]["pass_with_skill"]["cases"] == summary["pass_at_k"]["cases"]
    assert report_result is not None
    assert report_result.metadata["agent_eval"]["execution_status"] == "succeeded"


def test_final_result_fails_closed_before_publishing_unreferenced_oversize_detail(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result = {"execution_status": "failed", "execution_errors": [], "unreferenced": "x" * 2_100_000}

    with pytest.raises(ValueError, match="Final Tier 3 result exceeds"):
        harbor_runner._write_final_result(result_path, result)

    assert not result_path.exists()
    assert not list(tmp_path.glob(".result.json.*"))


def test_final_result_projects_real_multi_agent_aggregate_errors_without_losing_exact_total(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agents: dict[str, Any] = {}
    for agent_index in range(3):
        agent = f"agent-{agent_index}"
        conditions: dict[str, Any] = {}
        for condition, directory in (("with_skill", "with-skill"), ("without_skill", "without-skill")):
            failures = [
                {
                    "trial": f"trial-{agent_index}-{condition}-{failure_index}",
                    "reason": f"reason-{agent_index}-{condition}-{failure_index}:" + ("🚀" * 1_800),
                }
                for failure_index in range(32)
            ]
            summary = harbor_collector._condition_execution_summary(
                [],
                expected_case_ids=[],
                expected_cases=0,
                n_attempts=1,
                job_failure="",
                runtime_failures=failures,
            )
            conditions[condition] = summary
            _write_json(
                run_dir / agent / directory / "summary.json",
                {
                    "agent": agent,
                    "scores": {},
                    "custom_scores": {},
                    "pass_at_k": {},
                    "trial_failures": [],
                    "job_failure": "",
                    **summary,
                },
            )
        agents[agent] = {
            "pass_at_k": {"with_skill": {}, "without_skill": {}, "lift": {}},
            "conditions": conditions,
            "agent_runtime_failures": {"with_skill": [], "without_skill": []},
            "trial_failures": {"with_skill": [], "without_skill": []},
            "job_failures": {"with_skill": "", "without_skill": ""},
            **harbor_collector._aggregate_execution(list(conditions.values())),
        }

    result: dict[str, Any] = {
        "agents": agents,
        **harbor_collector._aggregate_execution(list(agents.values())),
    }
    result["error"] = list(result["execution_errors"])
    assert len(result["execution_errors"]) == 192
    assert len(json.dumps(result, indent=2).encode("utf-8")) > 2 * 1024 * 1024

    result_path = run_dir / "result.json"
    harbor_runner._write_final_result(result_path, result)

    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["execution_error_details_total"] == 192
    assert len(result["execution_errors"]) == 192
    assert persisted["execution_error_details_total"] == 192
    assert persisted["execution_error_details_shown"] == 1
    assert persisted["execution_error_details_truncated"] is True
    assert persisted["error"] == persisted["execution_errors"][:1]
    assert persisted["result_projection"]["omitted_root_detail_fields"] == ["error", "execution_errors"]
    assert result_path.stat().st_size <= harbor_runner.FINAL_RESULT_MAX_BYTES
    assert set(report_data.load_agent_data(run_dir)) == set(agents)


def test_final_result_projection_preserves_aggregate_hidden_child_error_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "opencode" / "with-skill" / "summary.json", {})
    _write_json(run_dir / "opencode" / "without-skill" / "summary.json", {})
    child_summaries = [
        {
            "execution_status": "failed",
            "execution_errors": ["shared visible error"],
            "execution_error_details_total": hidden_total,
            "execution_error_details_shown": 1,
            "execution_error_details_truncated": True,
            "expected_attempts": 1,
            "scored_attempts": 0,
        }
        for hidden_total in (300, 2)
    ]
    agent = harbor_collector._aggregate_execution(child_summaries)
    agent["conditions"] = {
        "with_skill": child_summaries[0],
        "without_skill": child_summaries[1],
    }
    result: dict[str, Any] = {
        "agents": {"opencode": agent},
        **harbor_collector._aggregate_execution([agent]),
    }
    result["error"] = list(result["execution_errors"])

    result_path = run_dir / "result.json"
    harbor_runner._write_final_result(result_path, result)

    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["execution_error_details_total"] == 302
    assert result["execution_error_details_truncated"] is True
    assert persisted["execution_error_details_total"] == 302
    assert persisted["execution_error_details_shown"] == 1
    assert persisted["execution_error_details_truncated"] is True
    assert persisted["agents"]["opencode"]["execution_error_details_total"] == 302
    assert persisted["agents"]["opencode"]["execution_error_details_truncated"] is True


def test_result_artifact_reference_rejects_intermediate_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "inside" / "summary.json", {"scores": {}})
    (run_dir / "agent").mkdir()
    try:
        (run_dir / "agent" / "with-skill").symlink_to("../inside", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert (
        harbor_runner._result_artifact_reference(
            run_dir,
            Path("agent/with-skill/summary.json"),
        )
        is None
    )


def test_result_artifact_reference_rejects_final_symlink_and_hardlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    source = run_dir / "source.json"
    _write_json(source, {"scores": {}})
    target_dir = run_dir / "agent" / "with-skill"
    target_dir.mkdir(parents=True)
    linked = target_dir / "summary.json"
    try:
        linked.symlink_to(source)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    assert harbor_runner._result_artifact_reference(run_dir, Path("agent/with-skill/summary.json")) is None

    linked.unlink()
    try:
        os.link(source, linked)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    assert harbor_runner._result_artifact_reference(run_dir, Path("agent/with-skill/summary.json")) is None


def test_result_artifact_reference_rejects_oversize_or_invalid_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "agent" / "with-skill" / "summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"value": "x" * harbor_runner.FINAL_RESULT_MAX_BYTES}), encoding="utf-8")
    relative = Path("agent/with-skill/summary.json")
    assert harbor_runner._result_artifact_reference(run_dir, relative) is None

    artifact.write_text("{not-json", encoding="utf-8")
    assert harbor_runner._result_artifact_reference(run_dir, relative) is None
