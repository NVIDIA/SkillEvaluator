# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the stable wrapper with the reusable GitLab scan script."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "ci/bundles/1.0.1"
RELEASE = json.loads((BUNDLE / "release.json").read_text(encoding="utf-8"))
EVIDENCE = "skill-evaluator-ci.json"
REPORT = "skill-evaluator-report"


def _stable_config():
    from ci.stable_channel import render

    assert (ROOT / "ci/templates/stable/stable-tier1.yml").is_file(), "The GitLab stable template is missing."
    return yaml.safe_load(render(BUNDLE)["stable-tier1.yml"])


def test_stable_wrapper_pairs_the_nested_evaluator_with_its_literal_image():
    config = _stable_config()
    assert config["include"] == [
        {
            "remote": "https://raw.githubusercontent.com/NVIDIA/SkillEvaluator/"
            f"{RELEASE['workflow_ref']}/ci/gitlab/skill-evaluation.yml"
        }
    ]
    job = config["skill-evaluation"]
    assert set(config) == {"include", "skill-evaluation"}
    assert set(job) == {"image", "before_script", "artifacts"}
    assert job["image"] == {"name": RELEASE["image"]}
    assert job["artifacts"] == {"paths": [f"{REPORT}/", EVIDENCE]}


def _run_stable_job(workspace, *, image=RELEASE["image"], scanner_exit=0, pipeline_source="push"):
    stable_job = _stable_config()["skill-evaluation"]
    upstream = yaml.safe_load((ROOT / "ci/gitlab/skill-evaluation.yml").read_text(encoding="utf-8"))
    script = "\n".join(stable_job["before_script"] + upstream["skill-evaluation"]["script"])
    bin_dir = workspace.parent / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        """#!/bin/sh
if [ "$1" != /opt/skillevaluator/ci/run_tier1.py ]; then
    exec "$TEST_PYTHON" "$@"
fi
while [ "$#" -gt 0 ]; do
    if [ "$1" = --reports ]; then shift; report_dir=$1; fi
    shift
done
printf '%s\\n' "$REPORT_IMAGE" "$REPORT_COMMIT" "$SCAN_MODE" > "$report_dir/execution.txt"
exit "$TEST_SCANNER_EXIT"
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=workspace,
        env={
            "PATH": str(bin_dir) + os.pathsep + os.defpath,
            "TEST_PYTHON": sys.executable,
            "TEST_SCANNER_EXIT": str(scanner_exit),
            "CI_PROJECT_DIR": str(workspace),
            "CI_JOB_IMAGE": image,
            "CI_JOB_ID": "9001",
            "CI_PIPELINE_ID": "321",
            "CI_PIPELINE_SOURCE": pipeline_source,
            "CI_COMMIT_SHA": "c" * 40,
            "CI_MERGE_REQUEST_DIFF_BASE_SHA": "d" * 40,
            "SKILL_EVAL_PATHS": "skills",
            "SKILL_EVAL_MIN_SCORE": "70",
            "SKILL_EVAL_SCAN_MODE": "auto",
            "SKILL_EVAL_IMAGE": "override.example/other:latest",
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("pipeline_source", "scanner_exit", "scan_mode"),
    [("push", 0, "all"), ("merge_request_event", 1, "changed")],
)
def test_stable_job_records_resolved_release_and_preserves_the_scan_result(
    tmp_path, pipeline_source, scanner_exit, scan_mode
):
    workspace = tmp_path / "project checkout"
    workspace.mkdir()
    result = _run_stable_job(workspace, scanner_exit=scanner_exit, pipeline_source=pipeline_source)

    assert result.returncode == scanner_exit, result.stderr
    assert json.loads((workspace / EVIDENCE).read_text(encoding="utf-8")) == {
        "channel": "stable",
        "release": "v1.0.1",
        "evaluator_sha": RELEASE["workflow_ref"],
        "image": RELEASE["image"],
        "consumer_commit": "c" * 40,
        "pipeline_id": "321",
        "job_id": "9001",
    }
    assert (workspace / REPORT / "execution.txt").read_text(encoding="utf-8").splitlines() == [
        RELEASE["image"],
        "c" * 40,
        scan_mode,
    ]


@pytest.mark.parametrize("image", ["", "other.example/scanner:latest"])
def test_stable_job_rejects_an_unverified_execution_image(tmp_path, image):
    workspace = tmp_path / "checkout"
    workspace.mkdir()
    result = _run_stable_job(workspace, image=image)

    assert result.returncode != 0
    assert "does not match" in result.stderr
    assert not (workspace / EVIDENCE).exists()
    assert not (workspace / REPORT).exists()


@pytest.mark.parametrize("existing", ["file", "directory", "symlink", "dangling symlink"])
def test_stable_job_keeps_existing_metadata_paths_untouched(tmp_path, existing):
    workspace = tmp_path / "checkout"
    workspace.mkdir()
    evidence = workspace / EVIDENCE
    target = tmp_path / "target"
    if existing == "file":
        evidence.write_text("checkout data", encoding="utf-8")
    elif existing == "directory":
        evidence.mkdir()
    else:
        if existing == "symlink":
            target.write_text("outside data", encoding="utf-8")
        evidence.symlink_to(target)
    result = _run_stable_job(workspace)

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert not (workspace / REPORT).exists()
    if existing == "file":
        assert evidence.read_text(encoding="utf-8") == "checkout data"
    elif existing == "directory":
        assert evidence.is_dir()
    else:
        assert evidence.is_symlink()
        if existing == "symlink":
            assert target.read_text(encoding="utf-8") == "outside data"
        else:
            assert not target.exists()
