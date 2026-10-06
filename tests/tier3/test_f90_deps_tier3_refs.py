# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 3 at Tier 3 (proof M8, M9 Tier 3 half, L25 provider-only sub-item).

Fixtures copy the shapes of check-03 ``edge-11-bundled-same-name-not-assumed``, ``t3-01``, ``t3-02``,
``edge-05-repo-root-ignored``, the skeptic's ``ext-leaf`` fixture and ``audit-retests/check-03-all-external``.
Nothing runs an agent: the decisions are made while the package is prepared, before any trial.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import _run_plugin_agent_eval, cli
from skillevaluator.reporting.base import is_advisory_agent_eval_skip
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

ORIGIN = "https://github.com/acme/agent-catalog.git"
REPO = "acme/agent-catalog"
EVALS = {
    "skill_name": "demo-plugin",
    "evals": [{"id": "c1", "prompt": "Write a changelog.", "expected_output": "A list."}],
}

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")


def _clone(root: Path, origin: str | None = ORIGIN) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    if origin:
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", origin], check=True)
    return root


def _skill(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {directory.name}\ndescription: Skill {directory.name}.\n---\n# {directory.name}\n",
        encoding="utf-8",
    )
    return directory


def _plugin(plugin: Path, *, skills: list | None = None, mcp: list | None = None, evals: bool = True) -> Path:
    plugin.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"name": "demo-plugin", "version": "1.0.0", "author": {"email": "dev@example.com"}}
    if skills is not None:
        manifest["skills"] = {"refs": skills}
    if mcp is not None:
        manifest["mcp"] = mcp
    (plugin / "agent_plugin.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    if evals:
        (plugin / "evals").mkdir(exist_ok=True)
        (plugin / "evals" / "evals.json").write_text(json.dumps(EVALS), encoding="utf-8")
    return plugin


@requires_git
@pytest.mark.parametrize(
    "ref",
    [f"github::{REPO}::skills::bundled", "github::other-org/other-repo::skills::bundled"],
    ids=["missing-edge11", "external-ext-leaf"],
)
def test_same_named_bundled_skill_does_not_cover_a_missing_or_external_ref(tmp_path: Path, ref: str) -> None:
    """Proof M8: the bundled skills/bundled 'covered' the ref, so the run was COMPLETE with exit 0."""
    repo = _clone(tmp_path / "repo")
    plugin = _plugin(repo / "plugins" / "demo", skills=[ref])
    _skill(plugin / "skills" / "bundled")

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert [path.name for path in package.include_skills] == ["bundled"]  # still evaluated as a bundled skill
    assert package.unresolved_skill_refs == (ref,)
    provenance = package.provenance()
    assert provenance["partial"] is True
    assert not package.skipped


@requires_git
def test_t3_02_tier3_resolves_from_the_git_top_level_like_tier1(tmp_path: Path) -> None:
    """Proof M9, check-03 t3-02: plugin at tools/demo; Tier 1 said referenced, Tier 3 skipped it as unresolved."""
    repo = _clone(tmp_path / "repo")
    _skill(repo / "skills" / "shared-skill")
    plugin = _plugin(repo / "tools" / "demo", skills=[f"github::{REPO}::skills::shared-skill"])

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert not package.skipped
    assert [path.name for path in package.include_skills] == ["shared-skill"]
    assert package.unresolved_skill_refs == ()
    provenance = package.provenance()
    assert provenance["partial"] is False
    assert provenance["dependency_status_counts"]["referenced"] == 1


@requires_git
def test_edge05_repo_root_that_does_not_contain_the_plugin_is_ignored_at_tier3(tmp_path: Path) -> None:
    """Proof M9, check-03 edge-05: Tier 1 ignores this --repo-root; Tier 3 now does the same."""
    repo = _clone(tmp_path / "repo")
    _skill(repo / "skills" / "shared-skill")
    plugin = _plugin(repo / "tools" / "demo", skills=[f"github::{REPO}::skills::shared-skill"])

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", repo_root=repo / "skills")

    assert [path.name for path in package.include_skills] == ["shared-skill"]
    assert package.unresolved_skill_refs == ()


@requires_git
def test_standalone_plugin_repo_ref_to_its_own_skill_is_covered(tmp_path: Path) -> None:
    """Tier 1 calls this ref provided; Tier 3 evaluates the bundled skill at that exact path."""
    plugin = _clone(tmp_path / "plugin", origin="https://github.com/acme/demo-plugin.git")
    _plugin(plugin, skills=["github::acme/demo-plugin::skills::helper"])
    _skill(plugin / "skills" / "helper")

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert [path.name for path in package.include_skills] == ["helper"]
    assert package.unresolved_skill_refs == ()
    assert package.provenance()["partial"] is False


@requires_git
def test_include_skills_still_supplies_an_external_ref(tmp_path: Path) -> None:
    repo = _clone(tmp_path / "repo")
    plugin = _plugin(repo / "plugins" / "demo", skills=["github::other-org/other-repo::skills::vendor-skill"])
    vendor = _skill(tmp_path / "vendored" / "vendor-skill")

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", include_skills=(vendor,))

    assert package.unresolved_skill_refs == ()
    assert package.provenance()["partial"] is False


@requires_git
def test_duplicate_external_refs_are_listed_once(tmp_path: Path) -> None:
    repo = _clone(tmp_path / "repo")
    _skill(repo / "skills" / "shared-skill")
    refs = [
        f"github::{REPO}::skills::shared-skill",
        "github::other-org/other-repo::skills::vendor-skill",
        {"source": "github", "repo": "Other-Org/Other-Repo.git", "path": "skills/vendor-skill"},
    ]
    package = prepare_plugin_eval_package(_plugin(repo / "plugins" / "demo", skills=refs), stage_root=tmp_path / "s")

    assert package.unresolved_skill_refs == ("github::other-org/other-repo::skills::vendor-skill",)
    assert package.provenance()["dependency_status_counts"]["external"] == 1


def _all_external(tmp_path: Path) -> Path:
    repo = _clone(tmp_path / "repo")
    _skill(repo / "skills" / "shared-skill")
    return _plugin(repo / "plugins" / "demo", skills=["github::other-org/other-repo::skills::vendor-skill"])


def _provider_only(tmp_path: Path) -> Path:
    return _plugin(tmp_path / "provider-only", mcp=[{"name": "search", "provider": "example-provider"}])


@requires_git
@pytest.mark.parametrize("build", [_all_external, _provider_only], ids=["all-external", "provider-only-mcp"])
def test_plugin_with_nothing_local_is_incomplete_not_skipped(tmp_path: Path, build) -> None:
    """Proof M8 and L25: 'Skipping plugin evaluation', exit 0, for a plugin whose components all failed to resolve."""
    plugin = build(tmp_path)

    run = CliRunner().invoke(cli, ["tier3", "evaluate-plugin", str(plugin), "--progress", "off"])
    assert run.exit_code == 1, run.output
    assert "INCOMPLETE" in run.output
    assert "Skipping plugin evaluation" not in run.output

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")
    assert package.skipped
    assert package.incomplete_skip

    result = _run_plugin_agent_eval(
        plugin, agents=None, env_mode="local", skip_baseline=False, n_concurrent=None, max_agents=None
    )
    assert not result.passed
    assert str(result.metadata["skip_reason"]).startswith("INCOMPLETE:")
    assert not is_advisory_agent_eval_skip(result)
    plugin_provenance = result.metadata["agent_eval"]["plugin_provenance"]
    assert plugin_provenance["partial"] is True


def test_plugin_that_declares_nothing_evaluable_is_still_an_optional_skip(tmp_path: Path) -> None:
    """A plugin with no unresolved component (only a rule that is staged nowhere) keeps the optional skip."""
    plugin = tmp_path / "empty"
    plugin.mkdir()
    (plugin / "agent_plugin.yaml").write_text(
        json.dumps({"name": "empty", "author": {"email": "dev@example.com"}, "skills": {"refs": []}}), encoding="utf-8"
    )

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert package.skipped
    assert not package.incomplete_skip
