# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-type detection and path resolution for the SkillEvaluator CLI.

Pure helpers with no Click/console/logging dependencies, shared by the
:mod:`skillevaluator.cli` entry point and :mod:`skillevaluator.validators`. The
Click command group itself lives in :mod:`skillevaluator.cli`.
"""

from pathlib import Path

from skillevaluator.constants import (
    CONTENT_TYPE_PLUGIN,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_UNKNOWN,
    CONTENT_TYPE_WORKFLOWS,
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    PLUGIN_MANIFEST_FILES,
    RULES_FILE_EXTENSION,
    SKILL_MANIFEST_FILE,
    WORKFLOWS_MANIFEST_FILE,
)

# ---------------------------------------------------------------------------
# Content-type detection
# ---------------------------------------------------------------------------


def _is_contained_plugin_manifest(path: Path) -> bool:
    """Return whether *path* is a contained-plugin manifest."""
    return path.name == PLUGIN_CONTAINED_MANIFEST_FILE and path.parent.name == PLUGIN_CONTAINED_MANIFEST_DIR


def _detect_from_file(path: Path) -> str | None:
    """Detect content type from a file path."""
    if path.name in PLUGIN_MANIFEST_FILES or _is_contained_plugin_manifest(path):
        return CONTENT_TYPE_PLUGIN
    if path.name.upper() == SKILL_MANIFEST_FILE.upper():
        return CONTENT_TYPE_SKILL
    if path.suffix == RULES_FILE_EXTENSION:
        parent = path.parent
        if parent.name == "references" or path.name == WORKFLOWS_MANIFEST_FILE:
            return CONTENT_TYPE_WORKFLOWS
        return CONTENT_TYPE_RULES
    return None


def _detect_from_directory(path: Path) -> str | None:
    """Detect content type from directory contents."""
    # Plugin detection must win before the SKILL.md / nested-structure checks:
    # a plugin dir may also contain skills/**/SKILL.md, but an agent_plugin.yaml
    # at the root makes it a plugin (prevents a false-green skill detection).
    if (
        any((path / manifest).exists() for manifest in PLUGIN_MANIFEST_FILES)
        or (path / PLUGIN_CONTAINED_MANIFEST_DIR / PLUGIN_CONTAINED_MANIFEST_FILE).exists()
    ):
        return CONTENT_TYPE_PLUGIN
    if (path / SKILL_MANIFEST_FILE).exists() or (path / SKILL_MANIFEST_FILE.lower()).exists():
        return CONTENT_TYPE_SKILL
    if (path / WORKFLOWS_MANIFEST_FILE).exists():
        return CONTENT_TYPE_WORKFLOWS
    if any(path.glob(f"*{RULES_FILE_EXTENSION}")):
        return CONTENT_TYPE_RULES
    return None


def _detect_from_path_parts(path: Path) -> str | None:
    """Detect content type from folder path patterns."""
    parts = path.parts
    if "skills" in parts or "team-skills" in parts:
        return CONTENT_TYPE_SKILL
    if "team-rules" in parts:
        return CONTENT_TYPE_RULES
    if "workflows" in parts or "team-workflows" in parts:
        return CONTENT_TYPE_WORKFLOWS
    return None


def _detect_from_nested_structure(path: Path) -> str | None:
    """Detect content type from nested directory structure."""
    skills_dir = path / "skills"
    team_skills_dir = path / "team-skills"
    if (skills_dir.exists() and any(skills_dir.rglob(SKILL_MANIFEST_FILE))) or (
        team_skills_dir.exists() and any(team_skills_dir.rglob(SKILL_MANIFEST_FILE))
    ):
        return CONTENT_TYPE_SKILL
    team_rules_dir = path / "team-rules"
    if team_rules_dir.exists() and any(team_rules_dir.rglob(f"*{RULES_FILE_EXTENSION}")):
        return CONTENT_TYPE_RULES
    if (path / "workflows").exists() or (path / "team-workflows").exists():
        return CONTENT_TYPE_WORKFLOWS
    return None


def detect_content_type(path: Path) -> str:
    """Auto-detect whether path contains a skill, rules, workflows, or plugin.

    Detection order: file type -> directory manifests -> path patterns -> nested structure.
    A plugin manifest (agent_plugin.yaml/.yml) at the root wins over a nested
    skills tree.
    """
    if path.is_file() and (detected := _detect_from_file(path)):
        return detected
    if path.is_dir() and (detected := _detect_from_directory(path)):
        return detected

    if detected := _detect_from_path_parts(path):
        return detected

    if path.is_dir() and (detected := _detect_from_nested_structure(path)):
        return detected

    return CONTENT_TYPE_UNKNOWN


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def resolve_skill_path(skill_path: Path) -> Path:
    """Convert SKILL.md file path to its parent directory."""
    return skill_path.parent if skill_path.is_file() else skill_path


def resolve_rules_path(rules_path: Path) -> Path:
    """Return path as-is for rules (can be file or directory)."""
    return rules_path


def resolve_workflows_path(workflows_path: Path) -> Path:
    """Convert workflow-rules.mdc path to its parent directory."""
    if workflows_path.is_file() and workflows_path.name == WORKFLOWS_MANIFEST_FILE:
        return workflows_path.parent
    return workflows_path


def resolve_plugin_path(path: Path) -> Path:
    """Convert a plugin manifest file path to its plugin root directory."""
    if path.is_file():
        if path.name in PLUGIN_MANIFEST_FILES:
            return path.parent
        if _is_contained_plugin_manifest(path):
            return path.parent.parent
    return path
