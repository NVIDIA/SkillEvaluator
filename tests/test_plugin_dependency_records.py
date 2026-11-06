# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline composition of bundle refs and client marketplace dependencies."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from skillevaluator.models.result import Severity
from skillevaluator.plugin_inventory import plugin_inventory_for_root


def repository(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "remote", "add", "origin", "https://github.com/org/catalog.git"], check=True
    )
    return root


def manifest(root: Path, refs: list, *, name="agent_plugin.yaml"):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps({"name": "demo", "skills": {"refs": refs}}))
    result = plugin_inventory_for_root(root)
    assert result is not None
    return result


def records(inv, **kwargs):
    from skillevaluator.plugin_dependency_records import dependency_records

    return dependency_records(inv, **kwargs)


def test_same_repository_missing_dependency_fails_without_fetching(tmp_path, monkeypatch):
    repo = repository(tmp_path / "repo")
    inv = manifest(repo / "plugins/demo", ["github::org/catalog::skills::missing"])
    real_run = subprocess.run

    def no_network(command, *args, **kwargs):
        assert not set(command) & {"fetch", "clone", "pull"}
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", no_network)
    result = records(inv, repo_root=repo)
    assert not result.passed
    assert any(f.check_name == "plugin_dependency_missing" for f in result.findings)
    row = result.metadata["plugin"]["dependency_resolution"]["skills"][0]
    assert row["state"] == "missing"
    assert next(m for m in inv.components if m.type == "skill").problem == "missing"


def test_external_and_unknown_identity_remain_advisory(tmp_path):
    repo = repository(tmp_path / "repo")
    inv = manifest(repo / "plugins/demo", ["github::other/catalog::skills::outside"])
    result = records(inv, repo_root=repo)
    assert result.passed
    assert result.metadata["plugin"]["dependency_status_counts"]["external"] == 1
    assert any(f.severity == Severity.MEDIUM for f in result.findings)
    unknown = manifest(tmp_path / "unverified", ["github::org/catalog::skills::missing"])
    result = records(unknown)
    assert result.passed
    assert result.metadata["plugin"]["dependency_status_counts"]["unresolved"] == 1
    assert any(f.check_name == "plugin_dependency_unverified" for f in result.findings)


def test_provided_and_referenced_targets_keep_their_own_scope(tmp_path):
    repo = repository(tmp_path / "repo")
    (repo / "skills/shared").mkdir(parents=True)
    (repo / "skills/shared/SKILL.md").write_text("# Shared")
    (repo / "plugins/demo/skills/local").mkdir(parents=True)
    (repo / "plugins/demo/skills/local/SKILL.md").write_text("# Local")
    inv = manifest(
        repo / "plugins/demo",
        ["github::org/catalog::plugins::demo/skills/local", "github::org/catalog::skills::shared"],
    )
    result = records(inv, repo_root=repo)
    assert result.passed
    counts = result.metadata["plugin"]["dependency_status_counts"]
    assert counts["provided"] == counts["referenced"] == 1
    local = next(m for m in inv.components if m.path == "skills/local")
    assert local.origin == "declared+packaged"
    assert all(not m.path or m.path != "skills/shared" for m in inv.selected_skills)


def test_additional_manifest_dependencies_keep_provenance_and_missing_gate(tmp_path):
    repo = repository(tmp_path / "repo")
    root = repo / "plugins/demo"
    root.mkdir(parents=True)
    (root / "agent_plugin.yaml").write_text(
        json.dumps({"name": "demo", "skills": {"refs": ["github::other/catalog::skills::external"]}})
    )
    (root / "agent_plugin.yml").write_text(
        json.dumps({"name": "demo", "rules": {"refs": ["github::org/catalog::rules::missing"]}})
    )
    inv = plugin_inventory_for_root(root)
    assert inv is not None
    result = records(inv, repo_root=repo)
    missing = next(f for f in result.findings if f.check_name == "plugin_dependency_missing")
    assert missing.metadata["declared_by"] == "agent_plugin.yml"
    assert not result.passed
    assert len(result.metadata["plugin"]["dependency_manifests"]) == 2


def test_malformed_and_unsafe_references_are_bounded_advisories(tmp_path):
    repo = repository(tmp_path / "repo")
    inv = manifest(repo / "plugins/demo", [42, "x" * 20000, "github::org/catalog::skills::../secrets"])
    result = records(inv, repo_root=repo)
    assert result.passed
    rows = result.metadata["plugin"]["dependency_resolution"]["skills"]
    assert all(row["state"] == "unresolved" for row in rows)
    assert max(len(row["ref"]) for row in rows) <= 512
    assert any(f.metadata.get("cause") == "unsafe" for f in result.findings)


def test_claude_marketplace_missing_and_external_dependencies(tmp_path):
    repo = repository(tmp_path / "repo")
    root = repo / "plugins/demo"
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin/plugin.json").write_text(
        json.dumps({"name": "demo", "dependencies": ["missing", "remote@other-market"]})
    )
    (repo / ".claude-plugin").mkdir()
    (repo / ".claude-plugin/marketplace.json").write_text(
        json.dumps({"name": "local-market", "plugins": [{"name": "demo", "source": "./plugins/demo"}]})
    )
    inv = plugin_inventory_for_root(root)
    assert inv is not None
    result = records(inv, repo_root=repo)
    rows = result.metadata["plugin"]["dependency_resolution"]["plugins"]
    assert {row["state"] for row in rows} == {"missing", "external"}
    assert not result.passed
