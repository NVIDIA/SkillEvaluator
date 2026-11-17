# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve GitLab policy authority using real commits and a local Git remote."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
HELPER = ROOT / "ci/prepare_security_baselines.py"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), "-c", "commit.gpgSign=false", *args], text=True).strip()


def repository(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init", "-q", "-b", "main")
    git(remote, "config", "user.name", "Baseline test")
    git(remote, "config", "user.email", "baseline@example.invalid")
    (remote / "policy").write_text("approved\n")
    git(remote, "add", ".")
    git(remote, "commit", "-qm", "approved")
    base = git(remote, "rev-parse", "HEAD")
    git(remote, "checkout", "-qb", "feature")
    (remote / "policy").write_text("candidate\n")
    git(remote, "commit", "-qam", "candidate")
    head = git(remote, "rev-parse", "HEAD")
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", str(remote), str(checkout)], check=True)
    return checkout, base, head


def run(workspace, **extra):
    env = {key: value for key, value in os.environ.items() if key in ("PATH", "LANG", "LC_ALL")}
    env.update(
        {
            "CI_PROJECT_DIR": str(workspace),
            "CI_PIPELINE_SOURCE": "merge_request_event",
            "CI_PROJECT_ID": "42",
            "CI_MERGE_REQUEST_TARGET_PROJECT_ID": "42",
            "CI_MERGE_REQUEST_TARGET_BRANCH_NAME": "main",
            **extra,
        }
    )
    return subprocess.run([sys.executable, "-I", str(HELPER)], env=env, capture_output=True, text=True)


def test_detached_gitlab_mr_resolves_target_branch_instead_of_candidate_or_merge_base(tmp_path):
    workspace, base, head = repository(tmp_path)
    result = run(workspace, CI_COMMIT_SHA=head, CI_MERGE_REQUEST_DIFF_BASE_SHA="0" * 40)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == base


def test_gitlab_target_fetch_preserves_existing_diff_base_ancestry(tmp_path):
    workspace, base, intermediate = repository(tmp_path)
    git(workspace, "config", "user.name", "Baseline test")
    git(workspace, "config", "user.email", "baseline@example.invalid")
    (workspace / "policy").write_text("candidate follow-up\n")
    git(workspace, "commit", "-qam", "candidate follow-up")
    head = git(workspace, "rev-parse", "HEAD")
    git(tmp_path / "remote", "update-ref", "refs/heads/main", intermediate)
    assert git(workspace, "merge-base", base, head) == base
    result = run(workspace, CI_COMMIT_SHA=head, CI_MERGE_REQUEST_DIFF_BASE_SHA=base)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == intermediate
    assert git(workspace, "merge-base", base, head) == base
    assert git(workspace, "rev-parse", "--is-shallow-repository") == "false"


def test_gitlab_push_uses_platform_commit(tmp_path):
    workspace, _, head = repository(tmp_path)
    result = run(workspace, CI_PIPELINE_SOURCE="push", CI_COMMIT_SHA=head)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == head


def test_gitlab_fork_without_available_exact_target_sha_fails_closed(tmp_path):
    workspace, _, head = repository(tmp_path)
    result = run(workspace, CI_COMMIT_SHA=head, CI_MERGE_REQUEST_TARGET_PROJECT_ID="99")
    assert result.returncode != 0
    assert "target" in result.stderr.lower()


def test_gitlab_fork_can_use_available_platform_target_commit(tmp_path):
    workspace, base, head = repository(tmp_path)
    result = run(
        workspace, CI_COMMIT_SHA=head, CI_MERGE_REQUEST_TARGET_PROJECT_ID="99", CI_MERGE_REQUEST_TARGET_BRANCH_SHA=base
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == base
