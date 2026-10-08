# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PluginSchemaValidator (bundle-reference plugin manifest validation)."""

import json
import os
from pathlib import Path

import pytest

from skillevaluator.constants import PLUGIN_MANIFEST_TYPE, PLUGIN_MODE
from skillevaluator.models.result import Severity
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


def _write_valid_skill(skills_root: Path, name: str) -> Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: A valid bundled skill used for secure discovery tests.\n"
        "metadata:\n"
        "  author: Test Author <test@example.com>\n"
        "---\n\n"
        f"# {name}\n\n"
        "## Instructions\nFollow the request.\n\n"
        "## Examples\nRun the example.\n",
        encoding="utf-8",
    )
    return skill_dir


class TestPluginSchemaValidator:
    def test_bundled_manifest_swap_after_discovery_fails_closed(self, tmp_path: Path, monkeypatch):
        from skillevaluator.utils import helpers

        _write_manifest(tmp_path, _VALID_MANIFEST)
        skill_dir = _write_valid_skill(tmp_path / "skills", "safe-skill")
        manifest = skill_dir / "SKILL.md"
        outside = tmp_path / "outside.md"
        outside.write_text(
            "---\nname: outside-canary\ndescription: Content outside the plugin root.\n---\n\n# Outside\n",
            encoding="utf-8",
        )
        real_discover = helpers.discover_secure_files

        def discover_then_swap(*args, **kwargs):
            files = real_discover(*args, **kwargs)
            if Path(args[0]) == tmp_path / "skills":
                manifest.unlink()
                manifest.symlink_to(outside)
            return files

        monkeypatch.setattr(helpers, "discover_secure_files", discover_then_swap)

        result = PluginSchemaValidator().validate(tmp_path)

        assert result.metadata["security_failure"] is True
        assert any(finding.check_name == "bundled_skill_path_unsafe" for finding in result.findings)
        assert not any("outside-canary" in finding.message for finding in result.findings)

    def test_bundled_directory_swap_after_secure_read_does_not_inspect_outside(self, tmp_path: Path, monkeypatch):
        from skillevaluator.utils.secure_fs import SecureRoot

        _write_manifest(tmp_path, _VALID_MANIFEST)
        skill_dir = _write_valid_skill(tmp_path / "skills", "safe-skill")
        outside = tmp_path / "outside-skill"
        outside.mkdir()
        (outside / "OUTSIDE_CANARY").write_text("do not inspect", encoding="utf-8")
        original = tmp_path / "original-safe-skill"
        real_read = SecureRoot.read_file_text

        def read_then_swap(self, manifest, max_bytes):
            content = real_read(self, manifest, max_bytes)
            if manifest.relative_path.as_posix() == "safe-skill/SKILL.md":
                skill_dir.rename(original)
                skill_dir.symlink_to(outside, target_is_directory=True)
            return content

        monkeypatch.setattr(SecureRoot, "read_file_text", read_then_swap)

        result = PluginSchemaValidator().validate(tmp_path)

        rendered_findings = "\n".join(f"{finding.message}\n{finding.file_path or ''}" for finding in result.findings)
        assert "OUTSIDE_CANARY" not in rendered_findings

    def test_rejects_manifest_symlink_outside_plugin_root(self, tmp_path: Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        external_manifest = outside / "external.yaml"
        external_manifest.write_text(_VALID_MANIFEST, encoding="utf-8")
        plugin = tmp_path / "plugin"
        plugin.mkdir()
        try:
            (plugin / "agent_plugin.yaml").symlink_to(external_manifest)
        except OSError:
            pytest.skip("symlinks are unavailable")

        result = PluginSchemaValidator().validate(plugin)

        assert not result.passed
        assert {finding.check_name for finding in result.findings} == {"manifest_outside_root"}

    def test_rejects_hardlinked_manifest(self, tmp_path: Path):
        outside = tmp_path / "outside.yaml"
        outside.write_text(_VALID_MANIFEST, encoding="utf-8")
        plugin = tmp_path / "plugin"
        plugin.mkdir()
        try:
            os.link(outside, plugin / "agent_plugin.yaml")
        except OSError:
            pytest.skip("hardlinks are unavailable")

        result = PluginSchemaValidator().validate(plugin)

        assert not result.passed
        assert result.metadata["security_failure"] is True

    @pytest.mark.parametrize("key", ["123", "true", "false", "null", "2026-01-01"])
    def test_non_string_key_returns_validation_finding(self, tmp_path, key):
        manifest = _write_manifest(tmp_path, f"{key}: extra value\n{_VALID_MANIFEST}")

        result = PluginSchemaValidator().validate(tmp_path)

        assert not result.passed
        finding = next(finding for finding in result.findings if finding.check_name == f"schema:{key}:invalid_key")
        assert finding.message == f"Field '{key}': Keys should be strings"
        assert finding.file_path == str(manifest)

    def test_quoted_unknown_key_remains_an_extra_field_error(self, tmp_path):
        _write_manifest(tmp_path, f'"true": extra value\n{_VALID_MANIFEST}')

        result = PluginSchemaValidator().validate(tmp_path)

        assert not result.passed
        assert any(finding.check_name == "schema:true:extra_forbidden" for finding in result.findings)
        assert not any("invalid_key" in finding.check_name for finding in result.findings)

    def test_valid_manifest_passes_with_metadata(self, tmp_path: Path):
        _write_manifest(tmp_path, _VALID_MANIFEST)
        result = PluginSchemaValidator().validate(tmp_path)
        assert result.passed
        # Outside a git clone the missing-dependency gate cannot run; that is a non-blocking finding, not a pass.
        assert [(f.check_name, f.severity) for f in result.findings] == [
            ("plugin_dependency_unverified", Severity.MEDIUM)
        ]
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

    def test_deep_yaml_produces_bounded_complexity_finding(self, tmp_path: Path):
        nested = "[" * 1_500 + "safe" + "]" * 1_500
        _write_manifest(tmp_path, f"name: safe\ndescription: {nested}\n")

        result = PluginSchemaValidator().validate(tmp_path)

        assert not result.passed
        assert len(result.findings) == 1
        assert result.findings[0].check_name == "manifest_complexity_limit"

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
        manifest.write_text('{"name": "contained-plugin", "skills": ["./demo"]}', encoding="utf-8")
        (tmp_path / "demo").mkdir()

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
        # An empty name has its own check name and message, not the "missing" one.
        assert any(f.check_name == "schema:name:empty" for f in result.findings)

    @pytest.mark.parametrize("manifest_dir", [".claude-plugin", ".cursor-plugin"])
    def test_declared_dependencies_count_the_same_fields_on_both_paths(self, tmp_path: Path, manifest_dir: str):
        """Regression: the Claude Code path counted keywords and skipped an mcpServers path.

        Both manifest paths now apply one rule: only fields that name other plugins
        are dependencies (proof L1). Keywords and component fields (an
        ``mcpServers`` path, ``commands``) are not, and only Claude Code has a
        ``dependencies`` field.
        """
        manifest = tmp_path / manifest_dir / "plugin.json"
        manifest.parent.mkdir()
        declared = {"keywords": ["a", "b"], "mcpServers": "./mcp.json", "commands": ["./c.md"]}
        if manifest_dir == ".claude-plugin":
            declared["dependencies"] = ["helper"]
        manifest.write_text(json.dumps({"name": "demo", **declared}), encoding="utf-8")
        (tmp_path / "mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
        (tmp_path / "c.md").write_text("# C\n", encoding="utf-8")

        result = PluginSchemaValidator().validate(tmp_path)

        expected = {"plugins": 1} if manifest_dir == ".claude-plugin" else None
        assert result.metadata["plugin"].get("declared_dependencies") == expected

    def test_deep_contained_json_produces_bounded_complexity_finding(self, tmp_path: Path):
        manifest = tmp_path / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        nested = "[" * 1_500 + "0" + "]" * 1_500
        manifest.write_text('{"name":"deep","metadata":' + nested + "}", encoding="utf-8")

        result = PluginSchemaValidator().validate(tmp_path)

        assert not result.passed
        assert len(result.findings) == 1
        assert result.findings[0].check_name == "manifest_complexity_limit"

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
        assert result.metadata["plugin"]["bundled_skills"] == ["skills/broken-skill"]
        assert result.metadata["plugin"]["in_plugin_skills"] == 1
        assert any("[broken-skill]" in finding.file_path for finding in result.findings)

    def test_bundled_skill_name_mismatch_suggests_keeping_the_folder_name(self, tmp_path: Path):
        # Claude Code loads a plugin skill by its folder name, so the fix is the frontmatter name.
        manifest = tmp_path / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        manifest.write_text('{"name": "hookify"}', encoding="utf-8")
        skill = tmp_path / "skills" / "writing-rules" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("---\nname: writing-hookify-rules\ndescription: Writes hookify rules.\n---\n# Rules\n")

        result = PluginSchemaValidator().validate(tmp_path)

        [finding] = [finding for finding in result.findings if finding.check_name == "name_consistency"]
        assert finding.suggestion.startswith("Update the frontmatter name to 'writing-rules'.")
        assert "rename directory" not in finding.suggestion

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


def _bundle_manifest(dir_path: Path, *, skills=None, rules=None, **fields) -> Path:
    body: dict = {"name": "ref-bundle", "author": {"email": "dev@example.com"}, **fields}
    if skills is not None:
        body["skills"] = {"refs": skills}
    if rules is not None:
        body["rules"] = {"refs": rules}
    return _write_manifest(dir_path, json.dumps(body))


def _schema_checks(result) -> list[str]:
    return [f.check_name for f in result.findings if f.check_name.startswith("schema")]


class TestBundleRefSchemaFindings:
    """Regression: a bad selector also failed the ``str`` member of the ref union, giving two HIGH findings."""

    @pytest.mark.parametrize("section", ["skills", "rules"])
    def test_selector_with_unknown_source_is_one_finding(self, tmp_path: Path, section: str):
        selector = {"source": "bitbucket", "repo": "example-org/example-repo", "path": f"{section}/demo"}
        _bundle_manifest(tmp_path, **{section: [selector]})

        result = PluginSchemaValidator().validate(tmp_path)

        assert not result.passed
        [finding] = [f for f in result.findings if f.check_name.startswith("schema")]
        assert finding.check_name == f"schema:{section}.refs.0.PluginSelector.source:literal_error"
        assert finding.severity == Severity.HIGH
        assert "'github', 'gitlab' or 'git'" in finding.message
        assert "valid string" not in finding.message

    @pytest.mark.parametrize(
        ("selector", "check"),
        [
            ({"source": "github", "repo": "noslash", "path": "skills/demo"}, "PluginSelector.repo:value_error"),
            (
                {"source": "github", "repo": "a/b", "path": "skills/demo", "ref": "main"},
                "PluginSelector.ref:extra_forbidden",
            ),
            ({"source": "github", "repo": "a/b", "path": "demo"}, "PluginSelector.path:value_error"),
        ],
        ids=["repo", "extra-key", "path"],
    )
    def test_every_selector_error_names_the_selector_field_only(self, tmp_path: Path, selector: dict, check: str):
        _bundle_manifest(tmp_path, skills=[selector])

        result = PluginSchemaValidator().validate(tmp_path)

        assert _schema_checks(result) == [f"schema:skills.refs.0.{check}"]

    def test_malformed_canonical_string_is_one_finding_at_its_own_index(self, tmp_path: Path):
        _bundle_manifest(tmp_path, skills=["github::example-org/example-repo::skills::ok", "bitbucket::a/b::skills::x"])

        result = PluginSchemaValidator().validate(tmp_path)

        assert not result.passed
        [finding] = [f for f in result.findings if f.check_name.startswith("schema")]
        assert finding.check_name == "schema:skills.refs.1.str:value_error"
        assert finding.severity == Severity.HIGH
        assert "canonical ref must be" in finding.message
        assert "'bitbucket::a/b::skills::x'" in finding.message

    def test_each_bad_ref_gets_exactly_one_finding(self, tmp_path: Path):
        """A malformed string no longer hides the selector after it; neither ref fans out into two findings."""
        _bundle_manifest(
            tmp_path,
            skills=[
                "github::noslash::skills::x",
                {"source": "svn", "repo": "a/b", "path": "skills/y"},
                "github::example-org/example-repo::skills::ok",
            ],
            rules=[{"source": "github", "repo": "a/b", "path": "rules/style.md"}, "git::noslash::rules::style.md"],
        )

        result = PluginSchemaValidator().validate(tmp_path)

        assert _schema_checks(result) == [
            "schema:skills.refs.0.str:value_error",
            "schema:skills.refs.1.PluginSelector.source:literal_error",
            "schema:rules.refs.1.str:value_error",
        ]

    def test_non_string_non_mapping_ref_is_still_one_finding_for_the_list(self, tmp_path: Path):
        _bundle_manifest(tmp_path, skills=[42, {"source": "svn", "repo": "a/b", "path": "skills/y"}])

        result = PluginSchemaValidator().validate(tmp_path)

        [finding] = [f for f in result.findings if f.check_name.startswith("schema")]
        assert finding.check_name == "schema:skills.refs:value_error"
        assert "(got int)" in finding.message

    @pytest.mark.parametrize("source", ["github", "gitlab", "git"])
    def test_valid_refs_produce_no_schema_findings(self, tmp_path: Path, source: str):
        _bundle_manifest(
            tmp_path,
            skills=[
                f"{source}::example-org/example-repo::skills::one",
                {"source": source, "repo": "example-group/tools/example-repo", "path": "skills/two"},
            ],
            rules=[{"source": source, "repo": "example-org/example-repo", "path": "rules/style.md"}],
        )

        result = PluginSchemaValidator().validate(tmp_path)

        assert _schema_checks(result) == []
        assert result.passed
        assert result.metadata["plugin"]["declared_dependencies"] == {"skills": 2, "rules": 1, "mcp": 0}

    def test_other_fields_keep_one_finding_each(self, tmp_path: Path):
        _bundle_manifest(
            tmp_path,
            tags="not-a-list",
            metadata=5,
            mcp=[{"name": "bad name", "provider": "stdio"}, "filesystem"],
        )

        result = PluginSchemaValidator().validate(tmp_path)

        assert _schema_checks(result) == [
            "schema:tags:list_type",
            "schema:metadata:dict_type",
            "schema:mcp.0.name:value_error",
            "schema:mcp.1:model_type",
        ]

    def test_bad_selectors_past_the_reporting_cap_are_counted_once_each(self, tmp_path: Path):
        """105 bad selectors were 210 errors (each failed the str member too); now 105, and the cap still applies."""
        selectors = [{"source": "bitbucket", "repo": "a/b", "path": f"skills/s{index:03}"} for index in range(105)]
        _bundle_manifest(tmp_path, skills=selectors)

        result = PluginSchemaValidator().validate(tmp_path)

        *reported, truncated = [f for f in result.findings if f.check_name.startswith("schema")]
        assert [f.check_name for f in reported] == [
            f"schema:skills.refs.{index}.PluginSelector.source:literal_error" for index in range(100)
        ]
        assert truncated.check_name == "schema_errors_truncated"
        assert truncated.severity == Severity.HIGH
        assert truncated.metadata == {"actual": 105, "reported": 100, "highest_unreported_severity": "high"}
        assert not result.passed
