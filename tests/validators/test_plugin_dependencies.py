# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 plugin dependency classification and the missing-dependency gate."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator import plugin_dependencies
from skillevaluator.cli import cli
from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.models.result import Severity
from skillevaluator.plugin_dependencies import (
    DEPENDENCY_STATES,
    RepositoryIdentity,
    classify_ref,
    local_repo_slug,
    resolve_repository_identity,
)
from skillevaluator.tier1.commands import run_validation
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

REPO = "example-org/example-repo"
ORIGIN = "https://github.com/Example-Org/example-repo.git"

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")


def _git_repo(root: Path, origin: str | None = ORIGIN) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    if origin:
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", origin], check=True)
    return root


def _skill(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {directory.name}\ndescription: Dependency fixture skill.\n"
        "metadata:\n  author: Test Author <test@example.com>\n---\n"
        f"# {directory.name}\n\n## Instructions\nFollow it.\n\n## Examples\nRun it.\n",
        encoding="utf-8",
    )
    return directory


def _manifest(plugin: Path, skills: list, rules: list | None = None) -> Path:
    plugin.mkdir(parents=True, exist_ok=True)
    body: dict = {"name": "dep-plugin", "author": {"email": "dev@example.com"}, "skills": {"refs": skills}}
    if rules is not None:
        body["rules"] = {"refs": rules}
    (plugin / "agent_plugin.yaml").write_text(json.dumps(body), encoding="utf-8")
    return plugin


def _rows(result) -> dict[str, dict]:
    resolution = result.metadata["plugin"]["dependency_resolution"]
    return {row["ref"]: row for row in [*resolution["skills"], *resolution["rules"]]}


@requires_git
def test_catalog_layout_classifies_each_state_and_gates_missing(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    _skill(repo / "skills" / "shared")
    (repo / "rules").mkdir()
    (repo / "rules" / "style.md").write_text("Use short sentences.\n", encoding="utf-8")
    plugin = _manifest(
        repo / "plugins" / "p",
        skills=[
            f"github::{REPO}::skills::shared",
            f"github::{REPO}::skills::ghost",
            "github::other-org/other-repo::skills::bundled",
            {"source": "github", "repo": REPO, "path": "plugins/p/skills/bundled"},
            f"github::{REPO}::docs::guide",
            # Repo-relative skills/bundled does not exist; the plugin's own
            # skills/bundled copy is noted but never assumed to satisfy it.
            f"github::{REPO}::skills::bundled",
        ],
        rules=[f"github::{REPO}::rules::style.md", f"git::{REPO}::rules::gone.md"],
    )
    _skill(plugin / "skills" / "bundled")

    result = PluginSchemaValidator().validate(plugin)

    rows = _rows(result)
    assert rows[f"github::{REPO}::skills::shared"] == {
        "ref": f"github::{REPO}::skills::shared",
        "state": "referenced",
        "path": "skills/shared",
        "reason": rows[f"github::{REPO}::skills::shared"]["reason"],
    }
    assert rows[f"github::{REPO}::skills::ghost"]["state"] == "missing"
    assert rows[f"github::{REPO}::skills::ghost"]["path"] is None
    external = rows["github::other-org/other-repo::skills::bundled"]
    assert external["state"] == "external"
    assert "skills/bundled" in external["reason"]  # same-name bundled copy is noted, not assumed
    provided = rows[f"github::{REPO}::plugins::p/skills/bundled"]
    assert (provided["state"], provided["path"]) == ("provided", "skills/bundled")
    # Outside the plugin root only recognized content roots are eligible.
    assert rows[f"github::{REPO}::docs::guide"]["state"] == "unresolved"
    shadowed = rows[f"github::{REPO}::skills::bundled"]
    assert shadowed["state"] == "missing"
    assert "bundled skill 'skills/bundled'" in shadowed["reason"]
    assert rows[f"github::{REPO}::rules::style.md"]["state"] == "referenced"
    assert rows[f"git::{REPO}::rules::gone.md"]["state"] == "missing"

    plugin_meta = result.metadata["plugin"]
    assert plugin_meta["dependency_status_counts"] == {
        "provided": 1,
        "referenced": 2,
        "missing": 3,
        "external": 1,
        "unresolved": 1,
    }
    assert plugin_meta["bundled_skills"] == ["skills/bundled"]
    assert plugin_meta["in_plugin_skills"] == 1

    missing = [f for f in result.findings if f.check_name == "plugin_dependency_missing"]
    assert {f.metadata["ref"] for f in missing} == {
        f"github::{REPO}::skills::ghost",
        f"github::{REPO}::skills::bundled",
        f"git::{REPO}::rules::gone.md",
    }
    assert all(f.severity == Severity.HIGH for f in missing)
    assert not result.passed
    # Only the missing refs produce findings; the other states are advisory.
    assert {f.check_name for f in result.findings} == {"plugin_dependency_missing"}


@requires_git
def test_standalone_plugin_repo_bundled_ref_is_provided(tmp_path: Path) -> None:
    plugin = _git_repo(tmp_path / "plugin")
    _skill(plugin / "skills" / "bundled")
    _manifest(plugin, skills=[f"github::{REPO}::skills::bundled"])

    result = PluginSchemaValidator().validate(plugin)

    row = _rows(result)[f"github::{REPO}::skills::bundled"]
    assert (row["state"], row["path"]) == ("provided", "skills/bundled")
    assert result.passed
    assert result.metadata["plugin"]["dependency_status_counts"]["provided"] == 1
    assert any(detail.check_name == "plugin_dependencies" for detail in result.success_details)


def _without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a checkout with no git metadata, regardless of the temp dir's parents."""
    monkeypatch.setattr(plugin_dependencies, "resolve_git_root", lambda _path: None)


def test_unknown_slug_is_unresolved_never_referenced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed: path existence alone never proves repository identity."""
    _without_git(monkeypatch)
    repo = tmp_path / "catalog"
    _skill(repo / "skills" / "shared")
    plugin = _manifest(repo / "plugins" / "p", skills=[f"github::{REPO}::skills::shared", f"github::{REPO}::skills::x"])

    result = PluginSchemaValidator().validate(plugin)

    states = {row["state"] for row in _rows(result).values()}
    assert states == {"unresolved"}
    assert all("repository identity unknown" in row["reason"] for row in _rows(result).values())
    assert result.passed
    assert not [f for f in result.findings if f.check_name == "plugin_dependency_missing"]
    assert result.metadata["plugin"]["dependency_status_counts"] == {
        "provided": 0,
        "referenced": 0,
        "missing": 0,
        "external": 0,
        "unresolved": 2,
    }


@requires_git
def test_git_repo_without_origin_is_unresolved(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo", origin=None)
    _skill(repo / "skills" / "shared")
    plugin = _manifest(repo / "plugins" / "p", skills=[f"github::{REPO}::skills::shared"])

    result = PluginSchemaValidator().validate(plugin)

    assert _rows(result)[f"github::{REPO}::skills::shared"]["state"] == "unresolved"
    assert result.passed


@requires_git
def test_symlinks_are_never_followed_to_a_referenced_target(tmp_path: Path) -> None:
    outside = _skill(tmp_path / "outside" / "escape")
    repo = _git_repo(tmp_path / "repo")
    (repo / "skills").mkdir()
    real = _skill(repo / "skills" / "real")
    try:
        (repo / "skills" / "escape").symlink_to(outside, target_is_directory=True)
        (repo / "skills" / "linked-parent").symlink_to(outside.parent, target_is_directory=True)
        (real / "SKILL.md").unlink()
        (real / "SKILL.md").symlink_to(outside / "SKILL.md")
    except OSError:
        pytest.skip("symlinks are unavailable")
    plugin = _manifest(
        repo / "plugins" / "p",
        skills=[
            f"github::{REPO}::skills::escape",
            f"github::{REPO}::skills::linked-parent/escape",
            f"github::{REPO}::skills::real",
        ],
    )

    result = PluginSchemaValidator().validate(plugin)

    rows = _rows(result)
    assert {row["state"] for row in rows.values()} == {"unresolved"}
    assert all("symlink" in row["reason"] for row in rows.values())
    assert all(row["path"] is None for row in rows.values())
    assert result.passed


@requires_git
def test_repo_root_option_overrides_detected_repository(tmp_path: Path) -> None:
    outer = _git_repo(tmp_path / "outer")
    _skill(outer / "skills" / "shared")
    inner = _git_repo(outer / "vendor" / "inner", origin="https://github.com/other-org/inner.git")
    plugin = _manifest(inner / "plugins" / "p", skills=[f"github::{REPO}::skills::shared"])
    ref = f"github::{REPO}::skills::shared"

    detected = PluginSchemaValidator().validate(plugin)
    overridden = PluginSchemaValidator(repo_root=outer).validate(plugin)
    ignored = PluginSchemaValidator(repo_root=tmp_path / "outer" / "skills").validate(plugin)

    assert _rows(detected)[ref]["state"] == "external"
    assert (_rows(overridden)[ref]["state"], _rows(overridden)[ref]["path"]) == ("referenced", "skills/shared")
    assert _rows(ignored)[ref]["state"] == "external"
    assert any("--repo-root ignored" in message for message in ignored.messages)


@requires_git
def test_repo_root_that_is_not_the_git_top_level_fails_closed(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    sub = repo / "sub"
    _skill(sub / "skills" / "shared")
    plugin = _manifest(sub / "plugins" / "p", skills=[f"github::{REPO}::skills::shared"])

    assert local_repo_slug(repo) == REPO
    assert local_repo_slug(sub) is None
    identity = resolve_repository_identity(plugin, sub)
    assert identity.local_slug is None
    result = PluginSchemaValidator(repo_root=sub).validate(plugin)
    assert _rows(result)[f"github::{REPO}::skills::shared"]["state"] == "unresolved"


@requires_git
def test_repository_identity_runs_two_git_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The top-level and the origin remote; the branch and HEAD a browse link needs are never asked for."""
    repo = _git_repo(tmp_path / "repo")
    plugin = _manifest(repo / "plugins" / "p", skills=[f"github::{REPO}::skills::shared"])
    commands: list[list[str]] = []
    real_check_output = subprocess.check_output

    def recording_check_output(args, *more, **kwargs):
        commands.append(list(args))
        return real_check_output(args, *more, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", recording_check_output)

    identity = resolve_repository_identity(plugin)

    assert identity.local_slug == REPO
    assert commands == [["git", "rev-parse", "--show-toplevel"], ["git", "remote", "get-url", "origin"]]


@requires_git
@pytest.mark.parametrize(
    ("origin", "slug"),
    [
        (ORIGIN, REPO),
        ("git@github.com:Example-Org/example-repo.git", REPO),
        ("ssh://git@github.com/Example-Org/example-repo.git", REPO),
        ("http://github.com/Example-Org/example-repo.git", None),
        # Read as an SCP-style "user@host:path" remote before the scheme was checked first.
        ("http://user@github.com:8080/Example-Org/example-repo.git", None),
        ("git://git@github.com:9418/Example-Org/example-repo.git", None),
        ("file:///srv/git/Example-Org/example-repo.git", None),
        # Regression: credentials plus a port gave the slug "8443/example-org/example-repo".
        ("https://gitlab-ci-token:example-token@gitlab.example.com:8443/Example-Org/example-repo.git", REPO),
        ("https://user@gitlab.example.com:8443/Example-Org/example-repo.git", REPO),
        ("ssh://git@gitlab.example.com:2222/Example-Org/example-repo.git", REPO),
    ],
)
def test_only_ssh_and_https_origins_establish_identity(tmp_path: Path, origin: str, slug: str | None) -> None:
    repo = _git_repo(tmp_path / "repo", origin=origin)

    assert local_repo_slug(repo) == slug


@requires_git
def test_https_origin_with_credentials_and_a_port_keeps_the_missing_dependency_gate(tmp_path: Path) -> None:
    """Regression: a mangled slug made same-repository refs 'external', so a missing one passed."""
    origin = "https://gitlab-ci-token:example-token@gitlab.example.com:8443/Example-Org/example-repo.git"
    repo = _git_repo(tmp_path / "repo", origin=origin)
    _skill(repo / "skills" / "shared")
    plugin = _manifest(repo / "plugins" / "p", skills=[f"git::{REPO}::skills::shared", f"git::{REPO}::skills::ghost"])

    result = PluginSchemaValidator().validate(plugin)

    rows = _rows(result)
    assert rows[f"git::{REPO}::skills::shared"]["state"] == "referenced"
    assert rows[f"git::{REPO}::skills::ghost"]["state"] == "missing"
    assert [f.check_name for f in result.findings if f.check_name == "plugin_dependency_missing"] == [
        "plugin_dependency_missing"
    ]
    assert not result.passed


@pytest.mark.parametrize(
    ("ref", "reason"),
    [
        (f"github::{REPO}::shared", "not a canonical"),
        ({"source": "github", "repo": REPO, "path": "skills/../secrets"}, "unsafe reference path"),
        (f"github::{REPO}::docs::guide", "not a recognized skills content root"),
        (42, "invalid reference"),
    ],
)
def test_unverifiable_refs_are_unresolved(tmp_path: Path, ref, reason: str) -> None:
    identity = RepositoryIdentity(clone_root=tmp_path, local_slug=REPO)

    row = classify_ref(ref, kind="skills", plugin_root=tmp_path / "plugins" / "p", identity=identity)

    assert row.state == "unresolved"
    assert reason in row.reason


def test_rule_ref_to_directory_or_hardlink_is_not_a_valid_rule(tmp_path: Path) -> None:
    (tmp_path / "rules" / "folder").mkdir(parents=True)
    rule = tmp_path / "rules" / "linked.md"
    rule.write_text("rule\n", encoding="utf-8")
    try:
        (tmp_path / "hardlink.md").hardlink_to(rule)
    except OSError:
        pytest.skip("hardlinks are unavailable")
    identity = RepositoryIdentity(clone_root=tmp_path, local_slug=REPO)
    plugin_root = tmp_path / "plugins" / "p"

    folder = classify_ref(f"github::{REPO}::rules::folder", kind="rules", plugin_root=plugin_root, identity=identity)
    linked = classify_ref(f"github::{REPO}::rules::linked.md", kind="rules", plugin_root=plugin_root, identity=identity)

    assert folder.state == "missing"
    assert linked.state == "unresolved"


def test_contained_manifest_has_no_dependency_resolution(tmp_path: Path) -> None:
    manifest = tmp_path / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir()
    manifest.write_text('{"name": "contained"}', encoding="utf-8")

    result = PluginSchemaValidator().validate(tmp_path)

    plugin_meta = result.metadata["plugin"]
    assert "dependency_resolution" not in plugin_meta
    assert plugin_meta["bundled_skills"] == []
    assert plugin_meta["in_plugin_skills"] == 0


@requires_git
def test_run_validation_threads_repo_root(tmp_path: Path) -> None:
    outer = _git_repo(tmp_path / "outer")
    inner = _git_repo(outer / "vendor" / "inner", origin="https://github.com/other-org/inner.git")
    plugin = _manifest(inner / "plugins" / "p", skills=[f"github::{REPO}::skills::ghost"])

    default = run_validation(plugin, checks="schema", content_type=CONTENT_TYPE_PLUGIN)
    overridden = run_validation(plugin, checks="schema", content_type=CONTENT_TYPE_PLUGIN, repo_root=outer)

    assert all(result.passed for result in default)
    assert not overridden[0].passed
    assert overridden[0].metadata["plugin"]["dependency_status_counts"]["missing"] == 1


@requires_git
def test_validate_cli_repo_root_flag_gates_missing_dependency(tmp_path: Path) -> None:
    outer = _git_repo(tmp_path / "outer")
    _skill(outer / "skills" / "shared")
    inner = _git_repo(outer / "vendor" / "inner", origin="https://github.com/other-org/inner.git")
    plugin = _manifest(inner / "plugins" / "p", skills=[f"github::{REPO}::skills::shared"])
    missing_plugin = _manifest(inner / "plugins" / "q", skills=[f"github::{REPO}::skills::ghost"])
    base_args = ["--no-llm", "--no-tier2", "--checks", "schema", "--type", "plugin", "-r", "json"]

    runner = CliRunner()
    ok = runner.invoke(
        cli, ["validate", str(plugin), *base_args, "-o", str(tmp_path / "ok"), "--repo-root", str(outer)]
    )
    missing = runner.invoke(
        cli, ["validate", str(missing_plugin), *base_args, "-o", str(tmp_path / "bad"), "--repo-root", str(outer)]
    )

    assert ok.exit_code == 0, ok.output
    assert missing.exit_code == 1, missing.output
    report = json.loads(next((tmp_path / "bad").glob("*.json")).read_text(encoding="utf-8"))
    assert "plugin_dependency_missing" in json.dumps(report)


def test_tier3_provenance_reports_dependency_status_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

    _without_git(monkeypatch)

    plugin = _manifest(tmp_path / "plugin", skills=[f"github::{REPO}::skills::remote"])

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert package.skipped
    counts = package.provenance()["dependency_status_counts"]
    assert set(counts) == set(DEPENDENCY_STATES)
    assert counts["unresolved"] == 1
    assert sum(counts.values()) == 1
