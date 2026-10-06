# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for content type detection functions in skillevaluator.cli_core (CLI infrastructure)."""

import os
from pathlib import Path

import pytest

from skillevaluator import cli_core, plugin_manifest
from skillevaluator.cli_core import (
    _detect_from_directory,
    _detect_from_file,
    _detect_from_nested_structure,
    _detect_from_path_parts,
    detect_content_type,
    resolve_plugin_path,
    resolve_rules_path,
    resolve_skill_path,
    resolve_workflows_path,
)
from skillevaluator.constants import (
    CONTENT_TYPE_PLUGIN,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_UNKNOWN,
    CONTENT_TYPE_WORKFLOWS,
)
from skillevaluator.plugin_manifest import agent_plugins_path_opt_in
from skillevaluator.utils.secure_fs import SecurePathError


class TestDetectFromFile:
    """Tests for _detect_from_file helper."""

    def test_detect_skill_md_uppercase(self, tmp_path: Path):
        """Test detection of SKILL.md file."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("content")
        assert _detect_from_file(skill_md) == CONTENT_TYPE_SKILL

    def test_detect_skill_md_lowercase(self, tmp_path: Path):
        """Test detection of skill.md file."""
        skill_md = tmp_path / "skill.md"
        skill_md.write_text("content")
        assert _detect_from_file(skill_md) == CONTENT_TYPE_SKILL

    def test_detect_mdc_rule_file(self, tmp_path: Path):
        """Test detection of .mdc rule file."""
        rule_file = tmp_path / "my-rule.mdc"
        rule_file.write_text("content")
        assert _detect_from_file(rule_file) == CONTENT_TYPE_RULES

    def test_detect_workflow_rules_mdc(self, tmp_path: Path):
        """Test detection of workflow-rules.mdc file."""
        workflow_rules = tmp_path / "workflow-rules.mdc"
        workflow_rules.write_text("content")
        assert _detect_from_file(workflow_rules) == CONTENT_TYPE_WORKFLOWS

    def test_detect_reference_mdc_as_workflow(self, tmp_path: Path):
        """Test detection of .mdc in references/ directory as workflow."""
        refs_dir = tmp_path / "references"
        refs_dir.mkdir()
        ref_file = refs_dir / "some-reference.mdc"
        ref_file.write_text("content")
        assert _detect_from_file(ref_file) == CONTENT_TYPE_WORKFLOWS

    def test_detect_unknown_file(self, tmp_path: Path):
        """Test detection returns None for unknown file types."""
        readme = tmp_path / "README.md"
        readme.write_text("content")
        assert _detect_from_file(readme) is None

    def test_detect_plugin_manifest_yaml(self, tmp_path: Path):
        """Test detection of an agent_plugin.yaml manifest file."""
        manifest = tmp_path / "agent_plugin.yaml"
        manifest.write_text("name: x")
        assert _detect_from_file(manifest) == CONTENT_TYPE_PLUGIN

    def test_detect_plugin_manifest_yml(self, tmp_path: Path):
        """Test detection of an agent_plugin.yml manifest file."""
        manifest = tmp_path / "agent_plugin.yml"
        manifest.write_text("name: x")
        assert _detect_from_file(manifest) == CONTENT_TYPE_PLUGIN

    def test_detect_contained_plugin_manifest_file(self, tmp_path: Path):
        """A .claude-plugin/plugin.json file is detected as a contained plugin."""
        claude_dir = tmp_path / ".claude-plugin"
        claude_dir.mkdir()
        manifest = claude_dir / "plugin.json"
        manifest.write_text('{"name": "x"}')
        assert _detect_from_file(manifest) == CONTENT_TYPE_PLUGIN

    def test_plain_plugin_json_not_detected(self, tmp_path: Path):
        """A plugin.json outside .claude-plugin/ is not a plugin manifest."""
        manifest = tmp_path / "plugin.json"
        manifest.write_text('{"name": "x"}')
        assert _detect_from_file(manifest) is None


class TestDetectFromDirectory:
    """Tests for _detect_from_directory helper."""

    def test_detect_skill_directory(self, tmp_path: Path):
        """Test detection of directory containing SKILL.md."""
        (tmp_path / "SKILL.md").write_text("content")
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_SKILL

    def test_detect_skill_directory_lowercase(self, tmp_path: Path):
        """Test detection of directory containing skill.md."""
        (tmp_path / "skill.md").write_text("content")
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_SKILL

    def test_detect_workflow_directory(self, tmp_path: Path):
        """Test detection of directory containing workflow-rules.mdc."""
        (tmp_path / "workflow-rules.mdc").write_text("content")
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_WORKFLOWS

    def test_detect_rules_directory(self, tmp_path: Path):
        """Test detection of directory containing .mdc files."""
        (tmp_path / "some-rule.mdc").write_text("content")
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_RULES

    def test_detect_plugin_directory(self, tmp_path: Path):
        """Test detection of directory containing an agent_plugin.yaml."""
        (tmp_path / "agent_plugin.yaml").write_text("name: x")
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_plugin_manifest_wins_over_nested_skill(self, tmp_path: Path):
        """A root agent_plugin.yaml must win over a nested skills/**/SKILL.md tree."""
        (tmp_path / "agent_plugin.yaml").write_text("name: x")
        nested = tmp_path / "skills" / "embedded"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("content")
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_PLUGIN
        assert detect_content_type(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_detect_contained_plugin_directory(self, tmp_path: Path):
        """A directory rooted by .claude-plugin/plugin.json is a contained plugin."""
        claude_dir = tmp_path / ".claude-plugin"
        claude_dir.mkdir()
        (claude_dir / "plugin.json").write_text('{"name": "x"}')
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_contained_plugin_wins_over_nested_skill(self, tmp_path: Path):
        """A contained manifest at the root wins over a nested skills tree."""
        claude_dir = tmp_path / ".claude-plugin"
        claude_dir.mkdir()
        (claude_dir / "plugin.json").write_text('{"name": "x"}')
        nested = tmp_path / "skills" / "embedded"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("---\nname: embedded\n---\n")
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_bundle_manifest_wins_over_contained(self, tmp_path: Path):
        """A bundle manifest takes precedence when both plugin models exist."""
        (tmp_path / "agent_plugin.yaml").write_text("name: x")
        claude_dir = tmp_path / ".claude-plugin"
        claude_dir.mkdir()
        (claude_dir / "plugin.json").write_text('{"name": "x"}')
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_detect_empty_directory(self, tmp_path: Path):
        """Test detection returns None for empty directory."""
        assert _detect_from_directory(tmp_path) is None

    def test_broken_selected_manifest_link_is_detected_lexically(self, tmp_path: Path) -> None:
        (tmp_path / "agent_plugin.yaml").symlink_to("missing-manifest")

        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_directory_named_like_manifest_is_not_detected_as_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "agent_plugin.yaml").mkdir()

        assert _detect_from_directory(tmp_path) is None

    def test_root_detection_stops_at_path_budget(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for index in range(20):
            (tmp_path / f"irrelevant-{index:02}.txt").write_text("x")
        real_scandir = os.scandir
        yielded = 0

        class TrackingScandir:
            def __init__(self, path) -> None:
                self._iterator = real_scandir(path)

            def __enter__(self):
                self._iterator.__enter__()
                return self

            def __exit__(self, *args):
                return self._iterator.__exit__(*args)

            def __iter__(self):
                return self

            def __next__(self):
                nonlocal yielded
                entry = next(self._iterator)
                yielded += 1
                return entry

        monkeypatch.setattr(cli_core, "CONTENT_DEDUP_MAX_DISCOVERED_PATHS", 2)
        monkeypatch.setattr(cli_core.os, "scandir", TrackingScandir)

        assert _detect_from_directory(tmp_path) is None
        assert yielded == 3


class TestDetectFromPathParts:
    """Tests for _detect_from_path_parts helper."""

    def test_detect_skills_in_path(self, tmp_path: Path):
        """Test detection from 'skills' in path."""
        skills_path = tmp_path / "skills" / "my-skill"
        skills_path.mkdir(parents=True)
        assert _detect_from_path_parts(skills_path) == CONTENT_TYPE_SKILL

    def test_detect_team_skills_in_path(self, tmp_path: Path):
        """Test detection from 'team-skills' in path."""
        team_skills_path = tmp_path / "team-skills" / "my-team" / "my-skill"
        team_skills_path.mkdir(parents=True)
        assert _detect_from_path_parts(team_skills_path) == CONTENT_TYPE_SKILL

    def test_detect_team_rules_in_path(self, tmp_path: Path):
        """Test detection from 'team-rules' in path."""
        rules_path = tmp_path / "team-rules" / "my-team"
        rules_path.mkdir(parents=True)
        assert _detect_from_path_parts(rules_path) == CONTENT_TYPE_RULES

    def test_detect_workflows_in_path(self, tmp_path: Path):
        """Test detection from 'workflows' in path."""
        workflows_path = tmp_path / "workflows" / "my-workflow"
        workflows_path.mkdir(parents=True)
        assert _detect_from_path_parts(workflows_path) == CONTENT_TYPE_WORKFLOWS

    def test_detect_team_workflows_in_path(self, tmp_path: Path):
        """Test detection from 'team-workflows' in path."""
        team_workflows_path = tmp_path / "team-workflows" / "my-team" / "my-workflow"
        team_workflows_path.mkdir(parents=True)
        assert _detect_from_path_parts(team_workflows_path) == CONTENT_TYPE_WORKFLOWS

    def test_detect_no_special_path(self, tmp_path: Path):
        """Test detection returns None for paths without special folders."""
        generic_path = tmp_path / "some" / "random" / "path"
        generic_path.mkdir(parents=True)
        assert _detect_from_path_parts(generic_path) is None


class TestDetectFromNestedStructure:
    """Tests for _detect_from_nested_structure helper."""

    def test_detect_nested_skills_directory(self, tmp_path: Path):
        """Test detection of nested skills/ directory."""
        skills_dir = tmp_path / "skills" / "my-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("content")
        assert _detect_from_nested_structure(tmp_path) == CONTENT_TYPE_SKILL

    def test_detect_nested_team_skills_directory(self, tmp_path: Path):
        """Test detection of nested team-skills/ directory."""
        team_skills_dir = tmp_path / "team-skills" / "my-team" / "my-skill"
        team_skills_dir.mkdir(parents=True)
        (team_skills_dir / "SKILL.md").write_text("content")
        assert _detect_from_nested_structure(tmp_path) == CONTENT_TYPE_SKILL

    def test_detect_nested_team_rules_directory(self, tmp_path: Path):
        """Test detection of nested team-rules/ directory."""
        rules_dir = tmp_path / "team-rules" / "my-team"
        rules_dir.mkdir(parents=True)
        (rules_dir / "my-rule.mdc").write_text("content")
        assert _detect_from_nested_structure(tmp_path) == CONTENT_TYPE_RULES

    def test_detect_nested_workflows_directory(self, tmp_path: Path):
        """Test detection of nested workflows/ directory."""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        assert _detect_from_nested_structure(tmp_path) == CONTENT_TYPE_WORKFLOWS

    def test_detect_nested_team_workflows_directory(self, tmp_path: Path):
        """Test detection of nested team-workflows/ directory."""
        team_workflows_dir = tmp_path / "team-workflows"
        team_workflows_dir.mkdir()
        assert _detect_from_nested_structure(tmp_path) == CONTENT_TYPE_WORKFLOWS

    def test_detect_empty_nested_structure(self, tmp_path: Path):
        """Test detection returns None for empty nested structure."""
        assert _detect_from_nested_structure(tmp_path) is None

    def test_detect_skills_dir_without_skill_md(self, tmp_path: Path):
        """Shallow detection classifies a regular skills/ marker without descent."""
        skills_dir = tmp_path / "skills" / "empty-skill"
        skills_dir.mkdir(parents=True)
        # No SKILL.md, but workflows exists
        (tmp_path / "workflows").mkdir()
        assert _detect_from_nested_structure(tmp_path) == CONTENT_TYPE_SKILL

    def test_nested_workflow_redirect_is_not_followed(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "workflows").symlink_to(outside, target_is_directory=True)

        assert _detect_from_nested_structure(tmp_path) is None


class TestDetectContentType:
    """Tests for the main detect_content_type function."""

    def test_detect_skill_from_file(self, tmp_path: Path):
        """Test full detection from SKILL.md file."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("content")
        assert detect_content_type(skill_md) == CONTENT_TYPE_SKILL

    def test_detect_skill_from_directory(self, tmp_path: Path):
        """Test full detection from directory with SKILL.md."""
        (tmp_path / "SKILL.md").write_text("content")
        assert detect_content_type(tmp_path) == CONTENT_TYPE_SKILL

    def test_detect_rules_from_file(self, tmp_path: Path):
        """Test full detection from .mdc file."""
        rule = tmp_path / "my-rule.mdc"
        rule.write_text("content")
        assert detect_content_type(rule) == CONTENT_TYPE_RULES

    def test_detect_workflows_from_directory(self, tmp_path: Path):
        """Test full detection from workflow directory."""
        (tmp_path / "workflow-rules.mdc").write_text("content")
        assert detect_content_type(tmp_path) == CONTENT_TYPE_WORKFLOWS

    def test_detect_plugin_from_file(self, tmp_path: Path):
        """Test full detection from an agent_plugin.yaml file."""
        manifest = tmp_path / "agent_plugin.yaml"
        manifest.write_text("name: x")
        assert detect_content_type(manifest) == CONTENT_TYPE_PLUGIN

    def test_detect_plugin_from_directory(self, tmp_path: Path):
        """Test full detection from a directory with agent_plugin.yaml."""
        (tmp_path / "agent_plugin.yaml").write_text("name: x")
        assert detect_content_type(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_detect_contained_plugin_from_manifest_file(self, tmp_path: Path):
        """Full detection supports a .claude-plugin/plugin.json file."""
        claude_dir = tmp_path / ".claude-plugin"
        claude_dir.mkdir()
        manifest = claude_dir / "plugin.json"
        manifest.write_text('{"name": "x"}')
        assert detect_content_type(manifest) == CONTENT_TYPE_PLUGIN

    def test_detect_contained_plugin_from_directory(self, tmp_path: Path):
        """Full detection supports a contained-plugin root directory."""
        claude_dir = tmp_path / ".claude-plugin"
        claude_dir.mkdir()
        (claude_dir / "plugin.json").write_text('{"name": "x"}')
        assert detect_content_type(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_detect_unknown(self, tmp_path: Path):
        """Test detection returns unknown for unrecognized path."""
        random_dir = tmp_path / "random"
        random_dir.mkdir()
        assert detect_content_type(random_dir) == CONTENT_TYPE_UNKNOWN

    def test_directory_is_listed_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The root-manifest and nested-structure checks share one bounded listing of the directory."""
        (tmp_path / "workflows").mkdir()
        real_scandir = os.scandir
        listed: list[str] = []

        def tracking_scandir(path):
            listed.append(os.fspath(path))
            return real_scandir(path)

        monkeypatch.setattr(cli_core.os, "scandir", tracking_scandir)

        assert detect_content_type(tmp_path) == CONTENT_TYPE_WORKFLOWS
        assert listed == [os.fspath(tmp_path)]

    @pytest.mark.parametrize("name", ["SKILL.md", "agent_plugin.yaml", "workflow-rules.mdc"])
    def test_detects_broken_selected_link_by_lexical_name(self, tmp_path: Path, name: str) -> None:
        selected = tmp_path / name
        selected.symlink_to("missing-target")

        assert detect_content_type(selected) != CONTENT_TYPE_UNKNOWN


class TestResolvePathFunctions:
    """Tests for path resolution functions."""

    def test_resolve_skill_path_from_directory(self, tmp_path: Path):
        """Test resolve_skill_path with directory."""
        result = resolve_skill_path(tmp_path)
        assert result == tmp_path

    def test_resolve_skill_path_from_file(self, tmp_path: Path):
        """Test resolve_skill_path with file."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("content")
        result = resolve_skill_path(skill_md)
        assert result == tmp_path

    def test_resolve_rules_path(self, tmp_path: Path):
        """Test resolve_rules_path returns path as-is."""
        rule = tmp_path / "my-rule.mdc"
        rule.write_text("content")
        result = resolve_rules_path(rule)
        assert result == rule

    def test_resolve_workflows_path_from_directory(self, tmp_path: Path):
        """Test resolve_workflows_path with directory."""
        result = resolve_workflows_path(tmp_path)
        assert result == tmp_path

    def test_resolve_workflows_path_from_file(self, tmp_path: Path):
        """Test resolve_workflows_path with workflow-rules.mdc file."""
        workflow_rules = tmp_path / "workflow-rules.mdc"
        workflow_rules.write_text("content")
        result = resolve_workflows_path(workflow_rules)
        assert result == tmp_path

    def test_resolve_plugin_path_from_file(self, tmp_path: Path):
        """Test resolve_plugin_path collapses an agent_plugin.yaml file to its dir."""
        manifest = tmp_path / "agent_plugin.yaml"
        manifest.write_text("name: x")
        assert resolve_plugin_path(manifest) == tmp_path

    def test_resolve_plugin_path_from_directory(self, tmp_path: Path):
        """Test resolve_plugin_path returns a directory unchanged."""
        assert resolve_plugin_path(tmp_path) == tmp_path

    def test_resolve_plugin_path_from_contained_manifest(self, tmp_path: Path):
        """A contained manifest resolves to the parent of .claude-plugin/."""
        claude_dir = tmp_path / ".claude-plugin"
        claude_dir.mkdir()
        manifest = claude_dir / "plugin.json"
        manifest.write_text('{"name": "x"}')
        assert resolve_plugin_path(manifest) == tmp_path

    @pytest.mark.parametrize(
        ("name", "resolver", "expected_parent_levels"),
        [
            ("SKILL.md", resolve_skill_path, 1),
            ("workflow-rules.mdc", resolve_workflows_path, 1),
            ("agent_plugin.yaml", resolve_plugin_path, 1),
            (".claude-plugin/plugin.json", resolve_plugin_path, 2),
        ],
    )
    def test_resolvers_use_lexical_manifest_shape_without_following_link(
        self, tmp_path: Path, name: str, resolver, expected_parent_levels: int
    ) -> None:
        selected = tmp_path / name
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.symlink_to("missing-target")
        expected = selected
        for _ in range(expected_parent_levels):
            expected = expected.parent

        assert resolver(selected) == expected


class TestAgentPluginsRootManifestDetection:
    """Auto-detection of a root plugin.json agrees with the secure plugin locator."""

    _SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"

    def _plugin_with_skills(self, root: Path, manifest: bytes) -> Path:
        root.mkdir()
        (root / "plugin.json").write_bytes(manifest)
        skill = root / "skills" / "a"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: a\ndescription: d\n---\nbody\n")
        return root

    def test_unparseable_manifest_naming_the_schema_is_a_plugin(self, tmp_path: Path) -> None:
        """Regression: a syntax error in an Agent Plugins manifest must not demote the plugin to a skill."""
        from skillevaluator.plugin_manifest import locate_plugin_manifest

        manifest = f'{{"$schema": "{self._SCHEMA}", "name": "x",}}'.encode()
        root = self._plugin_with_skills(tmp_path / "p", manifest)

        located = locate_plugin_manifest(root)
        assert located is not None
        assert located.manifest_type == "agent_plugins_v1"
        assert detect_content_type(root) == CONTENT_TYPE_PLUGIN
        assert _detect_from_file(root / "plugin.json") == CONTENT_TYPE_PLUGIN

    def test_unparseable_plugin_json_without_the_schema_is_not_a_plugin(self, tmp_path: Path) -> None:
        root = self._plugin_with_skills(tmp_path / "p", b'{"name": "legacy",}')

        assert detect_content_type(root) == CONTENT_TYPE_SKILL
        assert _detect_from_file(root / "plugin.json") is None

    @pytest.mark.parametrize("encoding", ["utf-16", "latin-1"])
    def test_manifest_that_is_not_utf8_is_detected_like_the_locator(self, tmp_path: Path, encoding: str) -> None:
        """Regression: a non-UTF-8 manifest naming the schema was a skill here but a manifest in the locator.

        The locator reports its encoding error; auto-detection used to demote the folder to a skill instead,
        so that error and the plugin's mcp.json were never checked.
        """
        from skillevaluator.plugin_manifest import locate_plugin_manifest

        text = f'{{"$schema": "{self._SCHEMA}", "name": "caf\xe9"}}'
        root = self._plugin_with_skills(tmp_path / "p", text.encode(encoding))

        located = locate_plugin_manifest(root)
        assert located is not None
        assert located.manifest_type == "agent_plugins_v1"
        assert _detect_from_file(root / "plugin.json") == CONTENT_TYPE_PLUGIN
        assert detect_content_type(root) == CONTENT_TYPE_PLUGIN

    def test_hard_linked_manifest_is_a_plugin_that_fails_closed(self, tmp_path: Path) -> None:
        """Regression: a hard-linked root plugin.json was detected as a skill, so no plugin check ran.

        The opt-in reads plugin.json like every other manifest read, so a hard
        link is refused, not parsed. The locator refuses it too, so detection
        counts it as a plugin manifest and validation fails closed.
        """
        from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest

        root = self._plugin_with_skills(tmp_path / "p", f'{{"$schema": "{self._SCHEMA}", "name": "x"}}'.encode())
        os.link(root / "plugin.json", tmp_path / "alias.json")

        with pytest.raises(SecurePathError):
            agent_plugins_path_opt_in(root / "plugin.json")
        assert _detect_from_file(root / "plugin.json") == CONTENT_TYPE_PLUGIN
        assert detect_content_type(root) == CONTENT_TYPE_PLUGIN
        with pytest.raises(PluginManifestPathError):
            locate_plugin_manifest(root)

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
    @pytest.mark.parametrize("name", ["plugin.json", "Plugin.json"])
    def test_symlinked_manifest_is_a_plugin_that_fails_closed(self, tmp_path: Path, name: str) -> None:
        """A linked root plugin.json is never read, opted in or not; the locator refuses it, so it marks a plugin."""
        from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest

        root = self._plugin_with_skills(tmp_path / "p", b'{"name": "legacy"}')
        (root / "plugin.json").unlink()
        (root / "real.json").write_text(f'{{"$schema": "{self._SCHEMA}", "name": "x"}}')
        (root / name).symlink_to("real.json")

        assert _detect_from_file(root / name) == CONTENT_TYPE_PLUGIN
        assert detect_content_type(root) == CONTENT_TYPE_PLUGIN
        with pytest.raises(PluginManifestPathError):
            locate_plugin_manifest(root)

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
    def test_special_file_manifest_is_a_plugin_that_fails_closed(self, tmp_path: Path) -> None:
        """A root plugin.json that is a named pipe is not opened; the locator refuses it, so it marks a plugin."""
        from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest

        root = self._plugin_with_skills(tmp_path / "p", b"{}")
        (root / "plugin.json").unlink()
        os.mkfifo(root / "plugin.json")

        assert _detect_from_file(root / "plugin.json") == CONTENT_TYPE_PLUGIN
        assert detect_content_type(root) == CONTENT_TYPE_PLUGIN
        with pytest.raises(PluginManifestPathError):
            locate_plugin_manifest(root)

    def test_hard_linked_manifest_fails_validation_as_a_plugin(self, tmp_path: Path) -> None:
        """End to end: the hard-linked manifest's hook used to pass unchecked as a skill; now the run fails closed."""
        from skillevaluator.tier1.commands import run_validation

        root = self._plugin_with_skills(tmp_path / "p", b"{}")
        (root / "plugin.json").unlink()
        manifest = f'{{"$schema": "{self._SCHEMA}", "name": "x", "hooks": "./hooks/hooks.json"}}'
        (root / "real").mkdir()
        (root / "real" / "manifest.json").write_text(manifest)
        os.link(root / "real" / "manifest.json", root / "plugin.json")

        content_type = detect_content_type(root)
        results = run_validation(root, checks="schema", content_type=content_type)

        assert content_type == CONTENT_TYPE_PLUGIN
        [schema] = results
        assert not schema.passed
        assert [finding.check_name for finding in schema.findings] == ["manifest_outside_root"]

    def test_manifest_over_the_opt_in_bound_opts_in_unread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clients read any size, so a root plugin.json over the bound opts in without being parsed."""
        root = self._plugin_with_skills(tmp_path / "p", b'{"name": "legacy-copilot"}')
        assert agent_plugins_path_opt_in(root / "plugin.json") is False

        def parse(raw: bytes) -> bool:
            raise AssertionError("an oversize manifest must not be parsed")

        monkeypatch.setattr(plugin_manifest, "AGENT_PLUGINS_OPT_IN_MAX_BYTES", 8)
        monkeypatch.setattr(plugin_manifest, "agent_plugins_opt_in", parse)
        assert agent_plugins_path_opt_in(root / "plugin.json") is True
        assert detect_content_type(root) == CONTENT_TYPE_PLUGIN
