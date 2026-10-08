# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 3 at Tier 3, review follow-ups (proof M9/L8 for rules, L25/M8 skip wording).

Rule refs are classified exactly as at Tier 1, and the reason a plugin evaluated nothing names only the
kinds of components it really has. Nothing runs an agent: every decision is made while the package is
prepared, before any trial.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.plugin_dependencies import classify_plugin_dependencies, resolve_repository_identity
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

ORIGIN = "https://github.com/acme/agent-catalog.git"
REPO = "acme/agent-catalog"
EVALS = {
    "skill_name": "demo-plugin",
    "evals": [{"id": "c1", "prompt": "Write a changelog.", "expected_output": "A list."}],
}
SKILL_ADVICE = "--include-skills"

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")


def _clone(root: Path, origin: str = ORIGIN) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin", origin], check=True)
    return root


def _bundle_plugin(plugin: Path, *, rules: list | None = None, mcp: list | None = None) -> tuple[Path, dict]:
    plugin.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"name": "demo-plugin", "version": "1.0.0", "author": {"email": "dev@example.com"}}
    if rules is not None:
        manifest["rules"] = {"refs": rules}
    if mcp is not None:
        manifest["mcp"] = mcp
    (plugin / "agent_plugin.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    (plugin / "evals").mkdir(exist_ok=True)
    (plugin / "evals" / "evals.json").write_text(json.dumps(EVALS), encoding="utf-8")
    return plugin, manifest


def _tier1_rule_states(plugin: Path, manifest: dict) -> list[tuple[str, str]]:
    resolution = classify_plugin_dependencies(manifest, plugin, resolve_repository_identity(plugin))
    return [(row.ref, row.state) for row in resolution.rules]


# --------------------------------------------------------------------------- #
# M9 / L8: rule refs use the Tier 1 classifier                                #
# --------------------------------------------------------------------------- #
@requires_git
def test_rule_ref_that_differs_only_in_case_is_unresolved_at_tier3_as_at_tier1(tmp_path: Path) -> None:
    """On a case-insensitive filesystem Tier 3 staged rules/style.md for '::rules::Style.md'; Tier 1 said missing."""
    repo = _clone(tmp_path / "repo")
    (repo / "rules").mkdir()
    (repo / "rules" / "style.md").write_text("# Style\nUse short lines.\n", encoding="utf-8")
    ref = f"github::{REPO}::rules::Style.md"
    plugin, manifest = _bundle_plugin(repo / "plugins" / "demo", rules=[ref])

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert _tier1_rule_states(plugin, manifest) == [(ref, "missing")]
    assert package.staged_rules == ()
    assert package.unresolved_rule_refs == (ref,)
    assert package.provenance()["partial"] is True


@requires_git
def test_same_repository_rule_ref_is_staged_as_at_tier1(tmp_path: Path) -> None:
    repo = _clone(tmp_path / "repo")
    (repo / "rules").mkdir()
    (repo / "rules" / "style.md").write_text("# Style\nUse short lines.\n", encoding="utf-8")
    ref = f"github::{REPO}::rules::style.md"
    plugin, manifest = _bundle_plugin(repo / "plugins" / "demo", rules=[ref, ref])

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert _tier1_rule_states(plugin, manifest) == [(ref, "referenced")]
    assert package.staged_rules == ("style.md",)
    assert package.unresolved_rule_refs == ()


@requires_git
@pytest.mark.parametrize(
    "ref",
    [
        "gitlab::example-group/tools/agent-catalog::rules::style.md",
        {"source": "gitlab", "repo": "example-group/tools/agent-catalog", "path": "rules/style.md"},
    ],
    ids=["canonical", "selector"],
)
def test_gitlab_subgroup_rule_ref_is_staged_as_at_tier1(tmp_path: Path, ref) -> None:
    """Regression: a ``source: gitlab`` rule ref was never classified or staged."""
    repo = _clone(tmp_path / "repo", origin="https://gitlab.example.com/example-group/tools/agent-catalog.git")
    (repo / "rules").mkdir()
    (repo / "rules" / "style.md").write_text("# Style\nUse short lines.\n", encoding="utf-8")
    plugin, manifest = _bundle_plugin(repo / "plugins" / "demo", rules=[ref])

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert _tier1_rule_states(plugin, manifest) == [
        ("gitlab::example-group/tools/agent-catalog::rules::style.md", "referenced")
    ]
    assert package.staged_rules == ("style.md",)
    assert package.unresolved_rule_refs == ()


@requires_git
@pytest.mark.parametrize("layout", ["plugin-at-repo-root", "plugin-in-subfolder"])
def test_rule_ref_inside_the_plugin_is_provided_and_staged(tmp_path: Path, layout: str) -> None:
    """A ref Tier 1 calls 'provided' (inside the plugin root) was unresolved at Tier 3, making the run partial."""
    repo = _clone(tmp_path / "repo")
    if layout == "plugin-at-repo-root":
        plugin_dir, ref = repo, f"github::{REPO}::rules::house.md"
    else:
        plugin_dir, ref = repo / "plugins" / "demo", f"github::{REPO}::plugins::demo/rules/house.md"
    (plugin_dir / "rules").mkdir(parents=True)
    (plugin_dir / "rules" / "house.md").write_text("# House rules\nCite sources.\n", encoding="utf-8")
    plugin, manifest = _bundle_plugin(plugin_dir, rules=[ref])

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert _tier1_rule_states(plugin, manifest) == [(ref, "provided")]
    assert package.staged_rules == ("house.md",)
    assert package.unresolved_rule_refs == ()
    assert package.provenance()["partial"] is False


# --------------------------------------------------------------------------- #
# L25 / M8: the skip reason gives advice only for what is present             #
# --------------------------------------------------------------------------- #
def test_provider_only_mcp_plugin_gets_mcp_advice_not_skill_advice(tmp_path: Path) -> None:
    plugin, _manifest = _bundle_plugin(tmp_path / "provider-only", mcp=[{"name": "search", "provider": "example"}])

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert package.incomplete_skip
    reason = str(package.skip_reason)
    assert "1 provider-only MCP server(s)" in reason
    assert "declare a command or url" in reason
    assert SKILL_ADVICE not in reason and "referenced skills" not in reason


@requires_git
def test_external_rule_only_plugin_gets_rule_advice(tmp_path: Path) -> None:
    repo = _clone(tmp_path / "repo")
    plugin, _manifest = _bundle_plugin(
        repo / "plugins" / "demo", rules=["github::other-org/other-repo::rules::style.md"]
    )

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert package.incomplete_skip
    reason = str(package.skip_reason)
    assert "1 unresolved rule ref(s)" in reason and "Add the referenced rules" in reason
    assert SKILL_ADVICE not in reason


def _local_mcp_plugin(root: Path, server: dict) -> Path:
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "local-mcp"}), encoding="utf-8")
    (root / ".mcp.json").write_text(json.dumps({"mcpServers": {"srv": server}}), encoding="utf-8")
    (root / "server").mkdir()
    (root / "server" / "index.js").write_text("console.log('hi')\n", encoding="utf-8")
    (root / "evals").mkdir()
    (root / "evals" / "evals.json").write_text(json.dumps({**EVALS, "skill_name": "local-mcp"}), encoding="utf-8")
    return root


def test_local_mcp_only_plugin_is_incomplete_with_native_load_advice(tmp_path: Path) -> None:
    """A Claude plugin whose only component is `node ${CLAUDE_PLUGIN_ROOT}/server/index.js` got skill-ref advice."""
    plugin = _local_mcp_plugin(
        tmp_path / "local-mcp", {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/server/index.js"]}
    )

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert package.skipped and package.incomplete_skip
    reason = str(package.skip_reason)
    assert "1 MCP server(s) launched from unstaged plugin files" in reason
    assert "--plugin-load native or auto" in reason
    assert SKILL_ADVICE not in reason and "Remote bundle-reference" not in reason

    run = CliRunner().invoke(cli, ["tier3", "evaluate-plugin", str(plugin), "--progress", "off"])
    assert run.exit_code == 1, run.output
    assert "INCOMPLETE" in run.output and "--plugin-load native or auto" in run.output
    assert SKILL_ADVICE not in run.output


def test_unrooted_local_mcp_server_gets_root_placeholder_advice(tmp_path: Path) -> None:
    plugin = _local_mcp_plugin(tmp_path / "local-mcp", {"command": "node", "args": ["./server/index.js"]})

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    reason = str(package.skip_reason)
    assert "${CLAUDE_PLUGIN_ROOT} paths instead of relative paths" in reason
    assert "--plugin-load native" not in reason
    assert SKILL_ADVICE not in reason
