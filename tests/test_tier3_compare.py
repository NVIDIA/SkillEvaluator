# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-run comparison must consume only successful evaluation summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillevaluator.tier3.commands import compare_results
from skillevaluator.tier3.output_provenance import mark_generated_output_root


def _complete_current_run(run_dir: Path) -> None:
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps({"run_id": run_dir.name, "agents": {}}),
        encoding="utf-8",
    )


def _write_authentic_pre_status_run(root: Path, run_id: str, *, score: float = 0.8) -> Path:
    run_dir = root / run_id
    summary_dir = run_dir / "opencode" / "with-skill"
    summary_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "harbor": {"n_attempts": {"value": 1, "source": "ACES default"}},
                "task_source": "evals_json",
                "agents": {
                    "opencode": {
                        "agent": "opencode",
                        "model": "test-model",
                        "source": "test default",
                        "occurrence": "1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "agent": "opencode",
                "model": "test-model",
                "model_source": "test default",
                "scores": {"security": score},
                "custom_scores": {},
                "metric_set": "aces-default-v2",
                "metrics": ["security"],
                "dimensions": {"safety": {"score": score, "sources": {"security": 1.0}}},
                "num_trials": 1,
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
                    "avg_attempts_used": 1.0,
                    "extra_cases": [],
                    "cases": {
                        "case": {
                            "passed": True,
                            "first_pass_attempt": 1,
                            "attempts_used": 1,
                            "attempts_skipped": 0,
                            "attempts_missing": 0,
                            "best_score": score,
                            "attempts": [
                                {
                                    "attempt": 1,
                                    "trial": "case__attempt1",
                                    "score": score,
                                    "passed": True,
                                }
                            ],
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_compare_rejects_partial_scores_from_failed_summary(tmp_path: Path) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    results_root = tmp_path / "results"
    summary_dir = results_root / "demo" / "20260709_010000" / "opencode" / "with-skill"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "execution_status": "failed",
                "execution_errors": ["Scored attempt coverage is 3/4"],
                "scored_attempts": 3,
                "expected_attempts": 4,
                "scores": {"security": 1.0, "accuracy": 0.8},
            }
        ),
        encoding="utf-8",
    )
    _complete_current_run(summary_dir.parents[1])

    assert compare_results(skill_path, results_dir=results_root) == 1


def test_compare_ignores_failed_baseline_scores(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    results_root = tmp_path / "results"
    agent_dir = results_root / "demo" / "20260709_010000" / "opencode"
    for variant, status, score in (
        ("with-skill", "succeeded", 1.0),
        ("without-skill", "failed", 0.1),
    ):
        summary_dir = agent_dir / variant
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(
            json.dumps({"execution_status": status, "scores": {"security": score}}),
            encoding="utf-8",
        )
    _complete_current_run(agent_dir.parent)

    assert compare_results(skill_path, results_dir=results_root) == 0
    assert "lift" not in capsys.readouterr().out.lower()


def test_compare_never_pairs_baseline_from_a_different_timestamp(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    results_root = tmp_path / "results"
    skill_results = results_root / "demo"

    newest = skill_results / "20260709_020000" / "opencode"
    for variant, status, score in (
        ("with-skill", "failed", 0.9),
        ("without-skill", "succeeded", 0.2),
    ):
        summary_dir = newest / variant
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(
            json.dumps({"execution_status": status, "scores": {"security": score}}),
            encoding="utf-8",
        )
    _complete_current_run(newest.parent)

    older = skill_results / "20260709_010000" / "opencode" / "with-skill"
    older.mkdir(parents=True)
    (older / "summary.json").write_text(
        json.dumps({"execution_status": "succeeded", "scores": {"security": 1.0}}),
        encoding="utf-8",
    )
    _complete_current_run(older.parents[1])

    assert compare_results(skill_path, results_dir=results_root) == 0
    assert "lift" not in capsys.readouterr().out.lower()


def test_compare_same_timestamp_unique_runs_uses_result_completion_time(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    skill_results = tmp_path / "results" / "demo"
    lexically_later = "20260709_010000_999_ffffffffffff"
    completed_later = "20260709_010000_111_aaaaaaaaaaaa"

    for run_id, score, completed_ns in (
        (lexically_later, 0.1, 100),
        (completed_later, 0.9, 200),
    ):
        run_dir = skill_results / run_id
        run_dir.mkdir(parents=True)
        mark_generated_output_root(run_dir)
        summary_dir = run_dir / "opencode" / "with-skill"
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(
            json.dumps({"execution_status": "succeeded", "scores": {"security": score}}),
            encoding="utf-8",
        )
        result_path = run_dir / "result.json"
        run_config: dict[str, object] = {
            "config_file": "none",
            "harbor": {
                "environment": {"value": "local", "source": "test"},
                "n_attempts": 1,
                "stop_on_pass": False,
                "n_concurrent": 1,
                "timeout_multiplier": 1.0,
                "base_image_mode": "disabled",
                "jobs_retained": False,
            },
            "provider": {"name": "nvidia", "model": "test-model"},
            "task_source": "evals_json",
            "grading": {"mode": "default"},
            "agents": {"opencode": {"agent": "opencode", "model": "test-model", "source": "test default"}},
        }
        agent_result = {
            "model": "test-model",
            "model_source": "test default",
            "model_resolution": {"model": "test-model", "source": "test default"},
            "with_skill": {"security": score},
            "without_skill": {},
            "custom_with_skill": {},
            "custom_without_skill": {},
            "dimensions_with_skill": {},
            "dimensions_without_skill": {},
            "lift": {},
            "custom_lift": {},
            "pass_at_k": {"with_skill": {}, "without_skill": {}, "lift": {}},
            "security_attribution": {},
            "agent_runtime_failures": {"with_skill": [], "without_skill": []},
            "trial_failures": {"with_skill": [], "without_skill": []},
            "job_failures": {"with_skill": "", "without_skill": ""},
            "conditions": {"with_skill": {}, "without_skill": {}},
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 1,
            "scored_attempts": 1,
            "num_trials_with": 1,
            "num_trials_without": 0,
            "output_dir": str((run_dir / "opencode").resolve()),
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
        result_path.write_text(
            json.dumps(
                {
                    "skill_name": skill_path.name,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "result_path": str(result_path.resolve()),
                    "run_config": run_config,
                    "agents": {"opencode": agent_result},
                    "attempt_policy": {
                        "max_attempts": 1,
                        "pass_threshold": 0.5,
                        "stop_on_pass": False,
                        "score_definition": "test",
                    },
                    "execution_status": "succeeded",
                    "execution_errors": [],
                    "report_status": "complete",
                    "duration_seconds": 1.0,
                }
            ),
            encoding="utf-8",
        )
        os.utime(result_path, ns=(completed_ns, completed_ns))

    assert compare_results(skill_path, results_dir=tmp_path / "results") == 0
    output = capsys.readouterr().out
    assert completed_later in output
    assert lexically_later not in output


@pytest.mark.parametrize("partial_kind", ["missing", "malformed", "mismatched"])
def test_compare_ignores_newer_partial_timestamp_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    partial_kind: str,
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    skill_results = tmp_path / "results" / "demo"
    older = skill_results / "20260709_010000"
    older_summary = older / "opencode" / "with-skill" / "summary.json"
    older_summary.parent.mkdir(parents=True)
    older_summary.write_text(
        json.dumps({"execution_status": "succeeded", "scores": {"security": 0.8}}),
        encoding="utf-8",
    )
    _complete_current_run(older)

    newer = skill_results / "20260709_020000"
    newer_summary = newer / "opencode" / "with-skill" / "summary.json"
    newer_summary.parent.mkdir(parents=True)
    newer_summary.write_text(
        json.dumps({"execution_status": "succeeded", "scores": {"security": 0.1}}),
        encoding="utf-8",
    )
    if partial_kind == "malformed":
        (newer / "result.json").write_text("{not json", encoding="utf-8")
    elif partial_kind == "mismatched":
        (newer / "result.json").write_text(json.dumps({"run_id": "different"}), encoding="utf-8")

    assert compare_results(skill_path, results_dir=tmp_path / "results") == 0
    output = capsys.readouterr().out
    assert older.name in output
    assert newer.name not in output


def test_compare_retains_non_timestamp_legacy_summary_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    legacy = tmp_path / "results" / "demo" / "legacy-run"
    summary = legacy / "opencode" / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"execution_status": "succeeded", "scores": {"security": 0.7}}),
        encoding="utf-8",
    )

    assert compare_results(skill_path, results_dir=tmp_path / "results") == 0
    assert legacy.name in capsys.readouterr().out


def test_compare_consumes_authenticated_timestamped_pre_status_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    run_id = "20260709_010000"
    _write_authentic_pre_status_run(tmp_path / "results" / skill_path.name, run_id)

    assert compare_results(skill_path, results_dir=tmp_path / "results") == 0
    output = capsys.readouterr().out
    assert run_id in output
    assert "opencode" in output


@pytest.mark.parametrize("primary_kind", ["empty", "partial", "non-object"])
def test_compare_falls_back_from_unusable_primary_root_to_valid_legacy_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    primary_kind: str,
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    primary = tmp_path / "results" / skill_path.name
    primary.mkdir(parents=True)
    if primary_kind == "partial":
        partial = primary / "20260709_020000" / "codex" / "with-skill"
        partial.mkdir(parents=True)
        (partial / "summary.json").write_text(
            json.dumps({"execution_status": "succeeded", "scores": {"security": 0.1}}),
            encoding="utf-8",
        )
    elif primary_kind == "non-object":
        completed = primary / "20260709_020000"
        summary = completed / "codex" / "with-skill" / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text("[]\n", encoding="utf-8")
        _complete_current_run(completed)

    legacy = skill_path / "evals" / "results" / "legacy-run" / "opencode" / "with-skill"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text(
        json.dumps({"execution_status": "succeeded", "scores": {"security": 0.9}}),
        encoding="utf-8",
    )

    assert compare_results(skill_path, results_dir=tmp_path / "results") == 0
    output = capsys.readouterr().out
    assert "opencode" in output
    assert "legacy-run" in output


def test_compare_never_mixes_agents_across_candidate_roots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    primary_run = tmp_path / "results" / "demo" / "20260709_020000"
    primary = primary_run / "codex" / "with-skill"
    primary.mkdir(parents=True)
    (primary / "summary.json").write_text(
        json.dumps({"execution_status": "succeeded", "scores": {"security": 0.8}}),
        encoding="utf-8",
    )
    _complete_current_run(primary_run)
    legacy = skill_path / "evals" / "results" / "legacy-run" / "opencode" / "with-skill"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text(
        json.dumps({"execution_status": "succeeded", "scores": {"security": 0.9}}),
        encoding="utf-8",
    )

    assert compare_results(skill_path, results_dir=tmp_path / "results") == 0
    output = capsys.readouterr().out
    assert "codex" in output
    assert "opencode" not in output
