# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for content type detection functions in skillevaluator.cli_core (CLI infrastructure)."""

from pathlib import Path

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

    def test_detect_contained_plugin_manifest(self, tmp_path: Path):
        """Test detection of a .claude-plugin/plugin.json manifest file."""
        manifest = tmp_path / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        manifest.write_text('{"name": "contained"}')
        assert _detect_from_file(manifest) == CONTENT_TYPE_PLUGIN


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

    def test_detect_contained_plugin_directory(self, tmp_path: Path):
        """Test detection of a directory containing .claude-plugin/plugin.json."""
        manifest = tmp_path / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        manifest.write_text('{"name": "contained"}')
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_plugin_manifest_wins_over_nested_skill(self, tmp_path: Path):
        """A root agent_plugin.yaml must win over a nested skills/**/SKILL.md tree."""
        (tmp_path / "agent_plugin.yaml").write_text("name: x")
        nested = tmp_path / "skills" / "embedded"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("content")
        assert _detect_from_directory(tmp_path) == CONTENT_TYPE_PLUGIN
        assert detect_content_type(tmp_path) == CONTENT_TYPE_PLUGIN

    def test_detect_empty_directory(self, tmp_path: Path):
        """Test detection returns None for empty directory."""
        assert _detect_from_directory(tmp_path) is None


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
        """Test that skills/ without SKILL.md returns None (continues checking)."""
        skills_dir = tmp_path / "skills" / "empty-skill"
        skills_dir.mkdir(parents=True)
        # No SKILL.md, but workflows exists
        (tmp_path / "workflows").mkdir()
        assert _detect_from_nested_structure(tmp_path) == CONTENT_TYPE_WORKFLOWS


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

    def test_detect_unknown(self, tmp_path: Path):
        """Test detection returns unknown for unrecognized path."""
        random_dir = tmp_path / "random"
        random_dir.mkdir()
        assert detect_content_type(random_dir) == CONTENT_TYPE_UNKNOWN


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
