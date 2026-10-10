# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M38 and L6: a wrong-typed component value in a Codex manifest gets Codex's severity, once.

Codex 0.142.5 drops a ``skills``, ``hooks``, or ``mcpServers`` value of the
wrong type, loads the default location, and installs the plugin (the
check-01 client oracles ``cx-skills-num``, ``cx-hooks-num``, ``cx-mcp-num``).
The manifest check rated these MEDIUM, but the component path check and the
MCP check still added a HIGH for the same defect, so Tier 1 failed a plugin
Codex installs. A wrong-typed ``apps`` makes Codex refuse the manifest and
stays HIGH.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import Severity
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_MANIFEST = {
    "name": "demo-codex",
    "version": "1.0.0",
    "description": "Demo Codex plugin",
    "author": {"name": "Example Dev"},
    "interface": {
        "displayName": "Demo Codex",
        "shortDescription": "A demo plugin",
        "longDescription": "A demo plugin used to test manifest validation.",
        "developerName": "Example Dev",
        "category": "Productivity",
        "capabilities": ["Interactive"],
        "websiteURL": "https://example.com",
        "privacyPolicyURL": "https://example.com/privacy",
        "termsOfServiceURL": "https://example.com/terms",
        "defaultPrompt": ["Say hello"],
    },
}


def _plugin(root: Path, **changes: object) -> Path:
    path = root / ".codex-plugin" / "plugin.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**_MANIFEST, **changes}), encoding="utf-8")
    return root


@pytest.mark.parametrize("field", ["skills", "hooks", "mcpServers"])
def test_codex_ignored_scalar_component_is_one_medium_finding(tmp_path: Path, field: str) -> None:
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", **{field: 5}))

    about = [finding for finding in result.findings if f"'{field}'" in finding.message or field in finding.check_name]
    assert [finding.severity for finding in about] == [Severity.MEDIUM], [(f.check_name, f.severity) for f in about]
    assert "Codex ignores the value" in about[0].message
    assert result.passed


def test_codex_scalar_apps_stays_high(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(_plugin(tmp_path / "p", apps=5))

    assert not result.passed
    assert any(finding.severity == Severity.HIGH for finding in result.findings)


def test_claude_scalar_component_stays_high(tmp_path: Path) -> None:
    path = tmp_path / "p" / ".claude-plugin" / "plugin.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"name": "demo", "version": "1.0.0", "skills": 5}), encoding="utf-8")

    result = PluginSchemaValidator().validate(tmp_path / "p")

    high = [finding for finding in result.findings if finding.severity == Severity.HIGH]
    assert [finding.check_name for finding in high] == ["plugin_component_path_invalid"]


@pytest.mark.parametrize("field", ["skills", "hooks", "mcpServers"])
def test_validate_exits_zero_for_a_manifest_codex_installs(tmp_path: Path, field: str) -> None:
    root = _plugin(tmp_path / "p", **{field: 5})
    args = ["--type", "plugin", "--tiers", "1", "--no-llm", "--checks", "schema", "-r", "json"]

    outcome = CliRunner().invoke(cli, ["validate", str(root), *args, "-o", str(tmp_path / "out")])

    assert outcome.exit_code == 0, outcome.output
