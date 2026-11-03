# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real repository discovery and path-boundary checks."""

import subprocess

import pytest

from skillevaluator import _ci_onboarding_git as git_helpers
from test_ci_onboarding import repository as repository  # noqa: PLC0414 -- export shared pytest fixture


def raw_git(repo, *args):
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-C", str(repo), *args], text=True
    )


@pytest.mark.parametrize(
    "value",
    [
        "https://other.example/example-org/demo.git",
        "https://token@github.com/example-org/demo.git",
        "https://github.com:443/example-org/demo.git",
        "https://github.com/example-org/demo?token=secret",
        "git@github.com:../demo.git",
        "file:///tmp/repository",
    ],
)
def test_remote_rejects_non_github_or_credential_bearing_inputs(value):
    with pytest.raises(git_helpers.SetupError):
        git_helpers.parse_remote(value)


@pytest.mark.parametrize("value", ["../skills", "/tmp/skills", "skills/../x", "skills\\x", "a\nb", "a//b", ""])
def test_paths_reject_unsafe_inputs(value):
    with pytest.raises(git_helpers.SetupError):
        git_helpers.safe_relative(value)


def test_both_supported_remote_forms():
    assert git_helpers.parse_remote("git@github.com:example-org/demo.git") == "example-org/demo"
    assert git_helpers.parse_remote("https://github.com/example-org/demo") == "example-org/demo"


def test_discovery_groups_hidden_roots_excludes_fixtures_and_untracked(repository):
    new = repository / "untracked/new/SKILL.md"
    new.parent.mkdir(parents=True)
    new.write_text("not committed")
    assert git_helpers.discover(repository) == [
        {"root": ".agents/skills", "count": 1, "default": True},
        {"root": "skills", "count": 2, "default": True},
        {"root": "tests/fixtures", "count": 1, "default": False},
    ]


def test_tracked_symlink_manifest_is_rejected(repository, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("source outside repository")
    link = repository / "skills/a/SKILL.md"
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(git_helpers.SetupError, match="Symlink"):
        git_helpers.discover(repository)


@pytest.mark.parametrize("operation", ["fetch", "push", "ls-remote"])
def test_network_git_uses_existing_gh_login_without_global_config(monkeypatch, tmp_path, operation):
    captured = {}
    monkeypatch.setenv("GH_HOST", "github.example.internal")
    monkeypatch.setenv("GH_REPO", "unrelated/repository")

    def run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(git_helpers.subprocess, "run", run)
    name = git_helpers.parse_remote("git@github.com:example-org/skills.git")
    assert git_helpers.git(tmp_path, operation, f"https://github.com/{name}.git", network=True) == "ok\n"
    args = captured["args"]
    assert "credential.helper=" in args
    assert "credential.https://github.com.helper=!gh auth git-credential" in args
    assert "--global" not in args
    assert captured["env"]["GH_HOST"] == "github.com"
    assert "GH_REPO" not in captured["env"]


def test_local_git_operations_do_not_enable_network_or_credentials(monkeypatch, tmp_path):
    captured = []

    def run(args, **kwargs):
        captured.extend(args)
        return ""

    monkeypatch.setattr(git_helpers, "run", run)
    git_helpers.git(tmp_path, "status", "--porcelain=v1")
    assert "protocol.allow=never" in captured
    assert not any("credential" in value for value in captured)


@pytest.mark.parametrize("missing_key", ["user.name", "user.email"])
def test_identity_preflight_has_local_configuration_guidance(monkeypatch, tmp_path, missing_key):
    def git(_repo, *args):
        if args == ("config", "--get", missing_key):
            raise git_helpers.SetupError("missing")
        return "set"

    monkeypatch.setattr(git_helpers, "git", git)
    with pytest.raises(git_helpers.SetupError, match=r"git config user\.name"):
        git_helpers.require_identity(tmp_path)


def test_push_error_suggests_workflow_permission_without_exposing_stderr(monkeypatch, tmp_path):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="secret-credential")

    monkeypatch.setattr(git_helpers.subprocess, "run", run)
    with pytest.raises(git_helpers.SetupError, match="gh auth refresh") as error:
        git_helpers.git(tmp_path, "push", "https://github.com/example-org/consumer.git", network=True)
    assert "secret-credential" not in str(error.value)
