# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private dependency identity and classification; public missing-dependency gating is deferred."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from skillevaluator.plugin_dependencies import (
    CAUSE_SOURCE,
    CAUSE_UNSAFE,
    REMOTE_REF_SOURCES,
    RepositoryIdentity,
    classify_plugin_dependencies,
    classify_ref,
    local_repo_slug,
    resolve_repository_identity,
    unverified_groups,
)
from skillevaluator.plugin_refs import PLUGIN_REF_SOURCES, find_duplicate_refs

REPO = "example-org/example-repo"
ORIGIN = "https://github.com/Example-Org/example-repo.git"
GITLAB_REPO = "example-group/tools/agent-catalog"
GITLAB_ORIGIN = "https://gitlab.example.com/Example-Group/tools/agent-catalog.git"

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


@requires_git
@pytest.mark.parametrize("suffix", ["", ".git", ".GIT", ".Git"])
def test_same_repository_reference_forms_share_suffix_normalization(tmp_path: Path, suffix: str) -> None:
    repo = _git_repo(tmp_path / "repo")
    plugin = _manifest(repo / "plugins" / "p", skills=[])
    _skill(repo / "skills" / "shared")
    identity = resolve_repository_identity(plugin)
    repo_name = f"Example-Org/Example-Repo{suffix}"
    refs = [
        f"github::{repo_name}::skills::shared",
        {"source": "github", "repo": repo_name, "path": "skills/shared"},
    ]

    rows = [classify_ref(ref, kind="skills", plugin_root=plugin, identity=identity) for ref in refs]

    assert [row.state for row in rows] == ["referenced", "referenced"]
    assert [row.path for row in rows] == ["skills/shared", "skills/shared"]
    assert len(find_duplicate_refs(refs)) == 1
    missing_refs = [
        f"github::{repo_name}::skills::absent",
        {"source": "github", "repo": repo_name, "path": "skills/absent"},
    ]
    missing_rows = [classify_ref(ref, kind="skills", plugin_root=plugin, identity=identity) for ref in missing_refs]
    assert [row.state for row in missing_rows] == ["missing", "missing"]


@requires_git
@pytest.mark.parametrize("suffix", ["", ".git", ".GIT", ".Git"])
def test_repository_remote_suffix_matches_reference_identity(tmp_path: Path, suffix: str) -> None:
    repo = _git_repo(tmp_path / "repo", origin=f"https://github.com/Example-Org/Example-Repo{suffix}")
    plugin = _manifest(repo / "plugins" / "p", skills=[])
    _skill(repo / "skills" / "shared")
    identity = resolve_repository_identity(plugin)

    row = classify_ref(f"github::{REPO}::skills::shared", kind="skills", plugin_root=plugin, identity=identity)

    assert identity.local_slug == REPO
    assert (row.state, row.path) == ("referenced", "skills/shared")


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
        # An SCP-style remote without a user gave the slug "github.com:example-org/example-repo", so every
        # same-repository ref was "external" and a missing one passed.
        ("github.com:Example-Org/example-repo.git", REPO),
        # git reads any "host:path" with no slash before the colon as SSH, so an SSH config alias names a
        # hosted repository; it was reported as a local path, leaving every same-repository ref unresolved.
        ("work-github:Example-Org/example-repo.git", REPO),
        ("gh:Example-Org/example-repo.git", REPO),
        ("C:/srv/git/example-repo.git", None),
        ("C:\\srv\\git\\example-repo.git", None),
        ("ssh://git@github.com/Example-Org/example-repo.git", REPO),
        # http:// and git:// origins name their repository as well as https:// does.
        ("http://github.com/Example-Org/example-repo.git", REPO),
        # Once read as an SCP-style "user@host:path" remote, giving "host/8080/group/repo".
        ("http://user:example-secret@github.com:8080/Example-Org/example-repo.git", REPO),
        ("git://git@github.com:9418/Example-Org/example-repo.git", REPO),
        ("file:///srv/git/Example-Org/example-repo.git", None),
        ("/srv/git/example-repo.git", None),
        # Regression: credentials plus a port gave the slug "8443/example-org/example-repo".
        ("https://gitlab-ci-token:example-token@gitlab.example.com:8443/Example-Org/example-repo.git", REPO),
        ("https://user@gitlab.example.com:8443/Example-Org/example-repo.git", REPO),
        ("ssh://git@gitlab.example.com:2222/Example-Org/example-repo.git", REPO),
        # GitLab subgroup repositories keep every group in the slug, in each remote form.
        (GITLAB_ORIGIN, GITLAB_REPO),
        ("git@gitlab.example.com:example-group/tools/agent-catalog.git", GITLAB_REPO),
        ("ssh://git@gitlab.example.com:2222/example-group/tools/agent-catalog.git", GITLAB_REPO),
        ("https://gitlab.example.com/example-group/tools/agent-catalog/-/tree/main", GITLAB_REPO),
    ],
)
def test_hosted_origins_establish_identity_and_local_paths_do_not(
    tmp_path: Path, origin: str, slug: str | None
) -> None:
    repo = _git_repo(tmp_path / "repo", origin=origin)

    assert local_repo_slug(repo) == slug


@pytest.mark.parametrize(
    "ref",
    [
        f"bitbucket::{GITLAB_REPO}::skills::shared",
        {"source": "bitbucket", "repo": GITLAB_REPO, "path": "skills/shared"},
    ],
    ids=["canonical", "selector"],
)
def test_unknown_source_is_unresolved_and_the_advice_names_every_supported_source(tmp_path: Path, ref) -> None:
    """The classifier (which Tier 3 runs on raw refs) still refuses a source outside the schema's list."""
    _skill(tmp_path / "skills" / "shared")
    identity = RepositoryIdentity(clone_root=tmp_path, local_slug=GITLAB_REPO)

    resolution = classify_plugin_dependencies({"skills": {"refs": [ref]}}, tmp_path / "plugins" / "p", identity)

    [row] = resolution.skills
    assert (row.state, row.path, row.cause) == ("unresolved", None, CAUSE_SOURCE)
    assert "unsupported reference source 'bitbucket'" in row.reason
    [(cause, rows, suggestion)] = unverified_groups(resolution)
    assert (cause, rows) == (CAUSE_SOURCE, [row])
    assert suggestion == "Use a supported reference source (github, gitlab, or git)."
    assert set(PLUGIN_REF_SOURCES) == REMOTE_REF_SOURCES


@requires_git
def test_missing_origin_advises_adding_one(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo", origin=None)
    plugin = _manifest(repo / "plugins" / "p", skills=[f"git::{REPO}::skills::shared"])

    identity = resolve_repository_identity(plugin)

    assert "the git repository 'repo' has no 'origin' remote" in identity.reason
    assert "git remote add origin <url>" in identity.remedy


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


@pytest.mark.parametrize(
    "ref",
    [
        "github::example-org/example-repo::skills::bad\x00name",
        "github::example-org/example-repo::skills\x00::bad",
        {"source": "github", "repo": REPO, "path": "skills/bad\x00name"},
    ],
)
def test_nul_dependency_paths_are_unresolved_instead_of_raising(tmp_path: Path, ref) -> None:
    (tmp_path / "skills").mkdir()
    manifest = {"skills": {"refs": [ref]}}
    identity = RepositoryIdentity(clone_root=tmp_path, local_slug=REPO)

    resolution = classify_plugin_dependencies(manifest, tmp_path, identity)

    assert len(resolution.skills) == 1
    assert resolution.skills[0].state == "unresolved"
    assert resolution.skills[0].cause == CAUSE_UNSAFE
    assert resolution.skills[0].path is None


@requires_git
@pytest.mark.parametrize(
    ("origin", "slug", "reason"),
    [
        ("https://git.example.org/\u00e9quipe/catalog.git", "\u00e9quipe/catalog", None),
        # UTF-8 "\u041d" ends in 0x9D, which the cp1252 code page cannot decode.
        ("C:/Users/\u041d\u0430\u0442\u0430\u043b\u044c\u044f/src/catalog", None, "is a local path"),
    ],
)
def test_origin_url_is_decoded_as_utf8_whatever_the_locale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin: str, slug: str | None, reason: str | None
) -> None:
    """On Windows ``text=True`` decodes with the ANSI code page; git prints the URL as UTF-8."""
    repo = _git_repo(tmp_path / "repo", origin=origin)
    real_check_output = subprocess.check_output

    def windows_default_check_output(args, *more, **kwargs):
        if kwargs.get("text") and "encoding" not in kwargs:
            kwargs["encoding"] = "cp1252"
        return real_check_output(args, *more, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", windows_default_check_output)

    identity = resolve_repository_identity(repo, repo_root=repo)

    assert identity.local_slug == slug
    assert (reason is None) == (identity.reason is None)
    if reason is not None:
        assert reason in identity.reason


@pytest.mark.parametrize(
    "ref",
    [
        f"github::{REPO}::.git::config",
        {"source": "github", "repo": REPO, "path": ".git/config"},
        f"github::{REPO}::.git::HEAD",
        f"github::{REPO}::.GIT::config",
        f"github::{REPO}::node_modules::pkg/index.js",
        f"github::{REPO}::rules::.git/config",
    ],
)
def test_plugin_at_the_repository_root_cannot_provide_never_content_folders(tmp_path: Path, ref) -> None:
    """Inside a repository-root plugin every path is in the plugin, so .git/config was "provided"."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("x\n", encoding="utf-8")
    (tmp_path / "rules" / ".git").mkdir(parents=True)
    (tmp_path / "rules" / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "rules" / "ok.md").write_text("rule\n", encoding="utf-8")
    identity = RepositoryIdentity(clone_root=tmp_path, local_slug=REPO)

    row = classify_ref(ref, kind="rules", plugin_root=tmp_path, identity=identity)
    bundled = classify_ref(f"github::{REPO}::rules::ok.md", kind="rules", plugin_root=tmp_path, identity=identity)

    assert (row.state, row.path, row.cause) == ("unresolved", None, CAUSE_UNSAFE)
    assert "never holds plugin content" in row.reason
    assert (bundled.state, bundled.path) == ("provided", "rules/ok.md")
