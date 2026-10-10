# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``declared_dependencies`` counts only fields that name other plugins (proof L1, check-01 e18)."""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.reporting.plugin_sections import tier1_plugin_view
from skillevaluator.validators.plugin_schema import PluginSchemaValidator


def _manifest(root: Path, rel: str, manifest: dict) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_claude_keywords_are_not_declared_dependencies_e18(tmp_path: Path) -> None:
    root = _manifest(tmp_path, ".claude-plugin/plugin.json", {"name": "demo", "keywords": ["a", "b", "c"]})
    result = PluginSchemaValidator().validate(root)

    plugin = result.metadata["plugin"]
    assert "declared_dependencies" not in plugin
    view = tier1_plugin_view(plugin)
    assert view["declared_dependencies"] == []


def test_claude_component_and_metadata_lists_are_not_dependencies(tmp_path: Path) -> None:
    (tmp_path / "skills" / "a").mkdir(parents=True)
    manifest = {"name": "demo", "skills": ["./skills/a"], "commands": ["./commands/x.md"], "keywords": ["k"]}
    result = PluginSchemaValidator().validate(_manifest(tmp_path, ".claude-plugin/plugin.json", manifest))

    assert "declared_dependencies" not in result.metadata["plugin"]


def test_claude_dependencies_are_counted(tmp_path: Path) -> None:
    manifest = {"name": "demo", "keywords": ["k"], "dependencies": ["helper", {"name": "vault", "marketplace": "m"}]}
    result = PluginSchemaValidator().validate(_manifest(tmp_path, ".claude-plugin/plugin.json", manifest))

    assert result.metadata["plugin"]["declared_dependencies"] == {"plugins": 2}


def test_codex_components_are_not_dependencies(tmp_path: Path) -> None:
    manifest = {"name": "demo-codex", "version": "1.0.0", "mcpServers": "./.mcp.json", "hooks": "./hooks.json"}
    result = PluginSchemaValidator().validate(_manifest(tmp_path, ".codex-plugin/plugin.json", manifest))

    assert "declared_dependencies" not in result.metadata["plugin"]
