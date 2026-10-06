# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code ``dependencies`` are resolved like ``agent_plugin.yaml`` refs (proof H7, dependency half).

Claude Code 2.1.284 refuses to load a plugin whose dependency is not
installed ("failed to load: Dependency ... is not installed", completeness
c7 oracle). Offline, a missing dependency can be proven through the plugin's
own marketplace manifest; anything else is reported as unverified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import Severity
from skillevaluator.reporting.plugin_sections import dependency_view
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

PLUGIN = {
    "name": "claude-demo",
    "version": "1.0.0",
    "description": "Claude Code plugin that declares a plugin dependency.",
    "author": {"name": "Demo Author", "email": "dev@example.com"},
    "keywords": ["release", "notes"],
}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _marketplace(root: Path, plugins: list[dict], **extra) -> Path:
    _write(root / ".claude-plugin" / "marketplace.json", {"name": "local-mkt", "plugins": plugins, **extra})
    return root


def _plugin(root: Path, dependencies: list) -> Path:
    _write(root / ".claude-plugin" / "plugin.json", {**PLUGIN, "dependencies": dependencies})
    return root


def _findings(result, check_name: str) -> list:
    return [finding for finding in result.findings if finding.check_name == check_name]


def test_dependency_outside_any_marketplace_is_unverified_neg03(tmp_path: Path) -> None:
    # check-03 neg-03: `dependencies: ["ghost-helper@acme-marketplace"]` with no marketplace in sight.
    root = _plugin(tmp_path / "repo" / "claude-demo", ["ghost-helper@acme-marketplace", "bare-helper"])
    result = PluginSchemaValidator(repo_root=tmp_path / "repo").validate(root)

    unverified = _findings(result, "plugin_dependency_unverified")
    assert [finding.metadata["state"] for finding in unverified] == ["external", "unresolved"]
    assert all(finding.severity == Severity.MEDIUM for finding in unverified)
    assert "installed and enabled" in unverified[0].message
    assert result.passed
    counts = result.metadata["plugin"]["dependency_status_counts"]
    assert counts["external"] == 1
    assert counts["unresolved"] == 1


def test_dependency_the_marketplace_does_not_list_is_missing(tmp_path: Path) -> None:
    repo = _marketplace(tmp_path / "repo", [{"name": "claude-demo", "source": "./plugins/claude-demo"}])
    root = _plugin(repo / "plugins" / "claude-demo", ["ghost-helper"])
    result = PluginSchemaValidator(repo_root=repo).validate(root)

    missing = _findings(result, "plugin_dependency_missing")
    assert len(missing) == 1
    assert missing[0].severity == Severity.HIGH
    assert "lists no plugin named 'ghost-helper'" in missing[0].message
    assert "refuses to load this plugin" in missing[0].message
    assert not result.passed
    view = dependency_view(result.metadata["plugin"])
    assert view is not None
    assert view["missing"] == 1
    assert view["rows"][0]["kind"] == "plugin"
    assert view["rows"][0]["ref"] == "ghost-helper"


def test_dependency_listed_with_a_missing_source_folder_is_missing(tmp_path: Path) -> None:
    repo = _marketplace(
        tmp_path / "repo",
        [
            {"name": "claude-demo", "source": "./plugins/claude-demo"},
            {"name": "helper", "source": "./plugins/helper"},
        ],
    )
    root = _plugin(repo / "plugins" / "claude-demo", ["helper@local-mkt"])
    result = PluginSchemaValidator(repo_root=repo).validate(root)

    missing = _findings(result, "plugin_dependency_missing")
    assert len(missing) == 1
    assert "does not exist" in missing[0].message


def test_dependency_listed_in_the_same_marketplace_is_referenced(tmp_path: Path) -> None:
    repo = _marketplace(
        tmp_path / "repo",
        [
            {"name": "claude-demo", "source": "./plugins/claude-demo"},
            {"name": "helper", "source": "./plugins/helper"},
        ],
    )
    (repo / "plugins" / "helper").mkdir(parents=True)
    root = _plugin(repo / "plugins" / "claude-demo", ["helper", {"name": "helper", "marketplace": "local-mkt"}])
    result = PluginSchemaValidator(repo_root=repo).validate(root)

    assert not _findings(result, "plugin_dependency_missing")
    assert not _findings(result, "plugin_dependency_unverified")
    assert result.passed
    rows = result.metadata["plugin"]["dependency_resolution"]["plugins"]
    assert [row["state"] for row in rows] == ["referenced", "referenced"]
    assert rows[0]["path"] == "plugins/helper"
    assert any(detail.check_name == "plugin_dependencies" for detail in result.success_details)


def test_cross_marketplace_dependency_is_external(tmp_path: Path) -> None:
    repo = _marketplace(
        tmp_path / "repo",
        [{"name": "claude-demo", "source": "./plugins/claude-demo"}],
        allowCrossMarketplaceDependenciesOn=["trusted-mkt"],
    )
    root = _plugin(repo / "plugins" / "claude-demo", ["vault@trusted-mkt", "other@unknown-mkt"])
    result = PluginSchemaValidator(repo_root=repo).validate(root)

    unverified = _findings(result, "plugin_dependency_unverified")
    assert [finding.metadata["state"] for finding in unverified] == ["external", "external"]
    assert "allows" in unverified[0].message
    assert "blocks installing it" in unverified[1].message
    assert result.passed


def test_marketplace_that_does_not_list_the_plugin_is_ignored(tmp_path: Path) -> None:
    repo = _marketplace(tmp_path / "repo", [{"name": "someone-else", "source": "./plugins/someone-else"}])
    root = _plugin(repo / "plugins" / "claude-demo", ["ghost-helper"])
    result = PluginSchemaValidator(repo_root=repo).validate(root)

    assert not _findings(result, "plugin_dependency_missing")
    assert [finding.metadata["state"] for finding in _findings(result, "plugin_dependency_unverified")] == [
        "unresolved"
    ]


def test_marketplace_above_the_repository_root_is_not_read(tmp_path: Path) -> None:
    _marketplace(tmp_path, [{"name": "claude-demo", "source": "./repo/plugins/claude-demo"}])
    repo = tmp_path / "repo"
    root = _plugin(repo / "plugins" / "claude-demo", ["ghost-helper"])
    result = PluginSchemaValidator(repo_root=repo).validate(root)

    assert not _findings(result, "plugin_dependency_missing")
    assert _findings(result, "plugin_dependency_unverified")[0].metadata["state"] == "unresolved"


@pytest.mark.skipif(os.name == "nt", reason="POSIX link fixture")
def test_linked_marketplace_manifest_is_not_followed(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.json"
    real.write_text(
        json.dumps({"name": "local-mkt", "plugins": [{"name": "claude-demo", "source": "./plugins/claude-demo"}]}),
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    (repo / ".claude-plugin").mkdir(parents=True)
    (repo / ".claude-plugin" / "marketplace.json").symlink_to(real)
    root = _plugin(repo / "plugins" / "claude-demo", ["ghost-helper"])
    result = PluginSchemaValidator(repo_root=repo).validate(root)

    assert not _findings(result, "plugin_dependency_missing")
    assert _findings(result, "plugin_dependency_unverified")[0].metadata["state"] == "unresolved"


def test_validate_cli_fails_on_a_missing_claude_dependency(tmp_path: Path) -> None:
    repo = _marketplace(tmp_path / "repo", [{"name": "claude-demo", "source": "./plugins/claude-demo"}])
    root = _plugin(repo / "plugins" / "claude-demo", ["ghost-helper"])
    args = ["--type", "plugin", "--tiers", "1", "--no-llm", "--checks", "schema", "-r", "json"]

    run = CliRunner().invoke(cli, ["validate", str(root), *args, "-o", str(tmp_path / "out"), "--repo-root", str(repo)])

    assert run.exit_code == 1, run.output
    report = json.loads(next((tmp_path / "out").glob("*.json")).read_text(encoding="utf-8"))
    checks = [finding["check_name"] for result in report["results"] for finding in result["findings"]]
    assert "plugin_dependency_missing" in checks
