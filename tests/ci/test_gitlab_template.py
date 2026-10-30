# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the reusable GitLab job and consumer overrides."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_user_can_override_included_global_defaults():
    config = yaml.safe_load((ROOT / "ci/gitlab/skill-evaluation.yml").read_text())
    job = config["skill-evaluation"]
    for name in ("SKILL_EVAL_PATHS", "SKILL_EVAL_MIN_SCORE", "SKILL_EVAL_SCAN_MODE"):
        assert name in config["variables"]
        assert name not in job.get("variables", {})
    assert config["variables"]["SKILL_EVAL_MIN_SCORE"] == "70"
    assert "LLM_ENABLED" not in config["variables"]


def _run_template(tmp_path, extra_env, *, scanner_exit=0, existing_report=False):
    import os
    import subprocess

    config = yaml.safe_load((ROOT / "ci/gitlab/skill-evaluation.yml").read_text())
    script = config["skill-evaluation"]["script"][0]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if existing_report:
        (workspace / "skill-evaluator-report").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = --reports ]; then shift; report_dir=$1; fi\n'
        "  shift\n"
        "done\n"
        'printf \'%s\\n%s\\n%s\\n%s\\n\' "$SCAN_MODE" "${DIFF_BASE_SHA:-}" "${DIFF_HEAD_SHA:-}" "$MIN_SCORE" > "$report_dir/evidence.txt"\n'
        'printf \'%s\' "${REPORT_RUN_URL:-}" > "$report_dir/run_url.txt"\n'
        f"exit {scanner_exit}\n"
    )
    stub.chmod(0o755)
    # Start from a clean environment so a suite run inside GitLab CI cannot leak its own job variables.
    outer = {key: value for key, value in os.environ.items() if not key.startswith(("CI_", "GITLAB_", "SKILL_EVAL_"))}
    env = {
        **outer,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "CI_PROJECT_DIR": str(workspace),
        "CI_PIPELINE_SOURCE": "push",
        "CI_COMMIT_SHA": "b" * 40,
        "SKILL_EVAL_IMAGE": "registry.example/test@sha256:" + "a" * 64,
        "SKILL_EVAL_PATHS": "skills",
        "SKILL_EVAL_MIN_SCORE": "80",
        "SKILL_EVAL_SCAN_MODE": "auto",
        **extra_env,
    }
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=False)
    report = workspace / "skill-evaluator-report/evidence.txt"
    return result, report.read_text().splitlines() if report.exists() else None


def test_gitlab_job_runs_all_skills_on_push_and_keeps_output(tmp_path):
    result, evidence = _run_template(tmp_path, {})
    assert result.returncode == 0
    assert evidence == ["all", "", "", "80"]


def test_gitlab_job_uses_mr_diff_and_retains_failed_results(tmp_path):
    result, evidence = _run_template(
        tmp_path,
        {"CI_PIPELINE_SOURCE": "merge_request_event", "CI_MERGE_REQUEST_DIFF_BASE_SHA": "a" * 40},
        scanner_exit=1,
    )
    assert result.returncode == 1
    assert evidence == ["changed", "a" * 40, "b" * 40, "80"]


def test_gitlab_job_rejects_merged_results_for_changed_selection(tmp_path):
    result, evidence = _run_template(
        tmp_path,
        {"CI_PIPELINE_SOURCE": "merge_request_event", "CI_MERGE_REQUEST_EVENT_TYPE": "merged_result"},
    )
    assert result.returncode != 0
    assert evidence is None
    assert "detached merge-request" in result.stderr


def test_gitlab_job_all_mode_supports_merged_result_pipelines(tmp_path):
    result, evidence = _run_template(
        tmp_path,
        {
            "CI_PIPELINE_SOURCE": "merge_request_event",
            "CI_MERGE_REQUEST_EVENT_TYPE": "merged_result",
            "SKILL_EVAL_SCAN_MODE": "all",
        },
    )
    assert result.returncode == 0
    assert evidence == ["all", "", "", "80"]


def test_gitlab_job_does_not_overwrite_existing_report_directory(tmp_path):
    result, evidence = _run_template(tmp_path, {}, existing_report=True)
    assert result.returncode != 0
    assert evidence is None
    assert "already exists" in result.stderr


def test_gitlab_job_passes_only_https_job_links(tmp_path):
    for job_url, expected in (
        ("http://gitlab.example.com/g/p/-/jobs/1", ""),
        ("https://gitlab.example.com/g/p/-/jobs/1", "https://gitlab.example.com/g/p/-/jobs/1"),
    ):
        case = tmp_path / ("https" if expected else "http")
        case.mkdir()
        result, evidence = _run_template(case, {"CI_JOB_URL": job_url})
        assert result.returncode == 0
        assert evidence == ["all", "", "", "80"]
        assert (case / "workspace/skill-evaluator-report/run_url.txt").read_text() == expected


def test_gitlab_template_ignores_the_outer_ci_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_MERGE_REQUEST_EVENT_TYPE", "merged_result")
    monkeypatch.setenv("SKILL_EVAL_SCAN_MODE", "all")
    result, evidence = _run_template(
        tmp_path,
        {"CI_PIPELINE_SOURCE": "merge_request_event", "CI_MERGE_REQUEST_DIFF_BASE_SHA": "a" * 40},
    )
    assert result.returncode == 0
    assert evidence == ["changed", "a" * 40, "b" * 40, "80"]
