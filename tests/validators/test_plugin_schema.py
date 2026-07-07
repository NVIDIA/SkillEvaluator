# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PluginSchemaValidator (bundle-reference plugin manifest validation)."""

from pathlib import Path

from skillevaluator.constants import PLUGIN_MANIFEST_TYPE, PLUGIN_MODE
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_VALID_MANIFEST = """
name: my-bundle
description: A test bundle
version: 1.0
author:
  email: dev@example.com
skills:
  refs:
    - "github::example-org/example-repo::skills::build-infra"
mcp:
  - name: filesystem
    provider: stdio
"""


def _write_manifest(dir_path: Path, body: str, name: str = "agent_plugin.yaml") -> Path:
    manifest = dir_path / name
    manifest.write_text(body)
    return manifest


class TestPluginSchemaValidator:
    def test_valid_manifest_passes_with_metadata(self, tmp_path: Path):
        _write_manifest(tmp_path, _VALID_MANIFEST)
        result = PluginSchemaValidator().validate(tmp_path)
        assert result.passed
        assert not result.findings
        assert result.metadata["manifest_type"] == PLUGIN_MANIFEST_TYPE
        assert result.metadata["plugin_mode"] == PLUGIN_MODE
        assert result.metadata["plugin"]["name"] == "my-bundle"

    def test_valid_manifest_file_path(self, tmp_path: Path):
        manifest = _write_manifest(tmp_path, _VALID_MANIFEST)
        result = PluginSchemaValidator().validate(manifest)
        assert result.passed

    def test_yml_extension_accepted(self, tmp_path: Path):
        _write_manifest(tmp_path, _VALID_MANIFEST, name="agent_plugin.yml")
        result = PluginSchemaValidator().validate(tmp_path)
        assert result.passed

    def test_missing_manifest_produces_finding(self, tmp_path: Path):
        result = PluginSchemaValidator().validate(tmp_path)
        assert not result.passed
        assert any(f.check_name == "manifest_missing" for f in result.findings)

    def test_invalid_yaml_produces_finding(self, tmp_path: Path):
        _write_manifest(tmp_path, "name: [unclosed\n")
        result = PluginSchemaValidator().validate(tmp_path)
        assert not result.passed
        assert any(f.check_name == "manifest_invalid_yaml" for f in result.findings)

    def test_non_mapping_manifest_produces_finding(self, tmp_path: Path):
        _write_manifest(tmp_path, "- just\n- a\n- list\n")
        result = PluginSchemaValidator().validate(tmp_path)
        assert not result.passed
        assert any(f.check_name == "manifest_not_mapping" for f in result.findings)

    def test_contract_violations_produce_schema_findings(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            """
name: bad-bundle
author:
  email: not-an-email
skills:
  refs:
    - "badsource::noslash::x"
""",
        )
        result = PluginSchemaValidator().validate(tmp_path)
        assert not result.passed
        assert all(f.category == "PLUGIN_SCHEMA" for f in result.findings)
        assert any(f.check_name.startswith("schema:") for f in result.findings)

    def test_contained_plugin_with_name_passes(self, tmp_path: Path):
        manifest = tmp_path / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        manifest.write_text('{"name": "contained-plugin", "skills": ["demo"]}', encoding="utf-8")

        result = PluginSchemaValidator().validate(tmp_path)

        assert result.passed
        assert result.metadata["manifest_type"] == "claude_plugin_json"
        assert result.metadata["plugin_mode"] == "contained"
        assert result.metadata["plugin"]["name"] == "contained-plugin"

    def test_contained_plugin_requires_non_empty_name(self, tmp_path: Path):
        manifest = tmp_path / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        manifest.write_text('{"name": ""}', encoding="utf-8")

        result = PluginSchemaValidator().validate(tmp_path)

        assert not result.passed
        assert any(f.check_name == "schema:name:missing" for f in result.findings)

    def test_bundle_manifest_wins_over_contained_manifest(self, tmp_path: Path):
        _write_manifest(tmp_path, _VALID_MANIFEST)
        manifest = tmp_path / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        manifest.write_text("not valid json", encoding="utf-8")

        result = PluginSchemaValidator().validate(tmp_path)

        assert result.passed
        assert result.metadata["manifest_type"] == PLUGIN_MANIFEST_TYPE

    def test_contained_plugin_merges_bundled_skill_schema_findings(self, tmp_path: Path):
        manifest = tmp_path / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        manifest.write_text('{"name": "contained-plugin"}', encoding="utf-8")
        skill = tmp_path / "skills" / "broken-skill" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("# Missing frontmatter", encoding="utf-8")

        result = PluginSchemaValidator().validate(tmp_path)

        assert not result.passed
        assert result.metadata["plugin"]["bundled_skills"] == ["broken-skill"]
        assert any("[broken-skill]" in finding.file_path for finding in result.findings)

    def test_contained_plugin_with_valid_bundled_skill_passes(self, tmp_path: Path):
        manifest = tmp_path / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        manifest.write_text('{"name": "contained-plugin"}', encoding="utf-8")
        skill = tmp_path / "skills" / "demo" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            "---\n"
            "name: demo\n"
            "description: Demo bundled skill\n"
            "metadata:\n"
            "  author: Demo Author <demo@example.com>\n"
            "---\n"
            "# Demo\n\n"
            "## Instructions\nUse this demo skill.\n\n"
            "## Examples\nRun the demo.\n",
            encoding="utf-8",
        )

        result = PluginSchemaValidator().validate(tmp_path)

        assert result.passed
        assert any(detail.check_name == "demo" for detail in result.success_details)
