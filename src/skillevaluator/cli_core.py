# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-type detection and path resolution for the SkillEvaluator CLI.

Pure helpers with no Click/console/logging dependencies, shared by the
:mod:`skillevaluator.cli` entry point and :mod:`skillevaluator.validators`. The
Click command group itself lives in :mod:`skillevaluator.cli`.
"""

import os
import stat
from pathlib import Path

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
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
    SKILL_MANIFEST_VARIANTS,
    WORKFLOWS_MANIFEST_FILE,
)
from skillevaluator.utils.secure_fs import stat_is_link_or_reparse

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
    try:
        root_metadata = path.lstat()
    except OSError:
        return None
    if stat_is_link_or_reparse(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
        return None
    plugin = skill = workflows = rules = False
    contained = False
    try:
        with os.scandir(path) as iterator:
            for count, entry in enumerate(iterator, start=1):
                if count > CONTENT_DEDUP_MAX_DISCOVERED_PATHS:
                    return None
                interesting = (
                    entry.name in PLUGIN_MANIFEST_FILES
                    or entry.name in SKILL_MANIFEST_VARIANTS
                    or entry.name in {WORKFLOWS_MANIFEST_FILE, PLUGIN_CONTAINED_MANIFEST_DIR}
                    or entry.name.endswith(RULES_FILE_EXTENSION)
                )
                if not interesting:
                    continue
                metadata = entry.stat(follow_symlinks=False)
                non_directory = not stat.S_ISDIR(metadata.st_mode)
                if entry.name in PLUGIN_MANIFEST_FILES and non_directory:
                    plugin = True
                elif entry.name == PLUGIN_CONTAINED_MANIFEST_DIR:
                    # Presence is enough for auto-detection. The secure plugin
                    # locator later distinguishes a real contained manifest
                    # from an empty, linked, or malformed marker directory.
                    contained = True
                elif entry.name in SKILL_MANIFEST_VARIANTS and non_directory:
                    skill = True
                elif entry.name == WORKFLOWS_MANIFEST_FILE and non_directory:
                    workflows = True
                elif entry.name.endswith(RULES_FILE_EXTENSION) and non_directory:
                    rules = True
    except OSError:
        return None
    # Plugin detection must win before the SKILL.md / nested-structure checks:
    # a plugin dir may also contain skills/**/SKILL.md, but a plugin manifest at
    # the root -- either agent_plugin.yaml/.yml (bundle-reference) or
    # .claude-plugin/plugin.json (contained) -- makes it a plugin.
    if plugin or contained:
        return CONTENT_TYPE_PLUGIN
    if skill:
        return CONTENT_TYPE_SKILL
    if workflows:
        return CONTENT_TYPE_WORKFLOWS
    if rules:
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
    """Detect content type from a bounded, shallow structural marker scan."""
    try:
        root_metadata = path.lstat()
    except OSError:
        return None
    if stat_is_link_or_reparse(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
        return None

    markers: set[str] = set()
    try:
        with os.scandir(path) as iterator:
            for count, entry in enumerate(iterator, start=1):
                if count > CONTENT_DEDUP_MAX_DISCOVERED_PATHS:
                    return None
                if entry.name not in {"skills", "team-skills", "team-rules", "workflows", "team-workflows"}:
                    continue
                metadata = entry.stat(follow_symlinks=False)
                if not stat_is_link_or_reparse(metadata) and stat.S_ISDIR(metadata.st_mode):
                    markers.add(entry.name)
    except OSError:
        return None
    if markers & {"skills", "team-skills"}:
        return CONTENT_TYPE_SKILL
    if "team-rules" in markers:
        return CONTENT_TYPE_RULES
    if markers & {"workflows", "team-workflows"}:
        return CONTENT_TYPE_WORKFLOWS
    return None


def detect_content_type(path: Path) -> str:
    """Auto-detect whether path contains a skill, rules, workflows, or plugin.

    Detection order: file type -> directory manifests -> path patterns -> nested structure.
    A plugin manifest at the root -- agent_plugin.yaml/.yml (bundle-reference) or
    .claude-plugin/plugin.json (contained) -- wins over a nested skills tree.
    """
    try:
        metadata = path.lstat()
    except OSError:
        metadata = None
    if metadata is not None and not stat.S_ISDIR(metadata.st_mode) and (detected := _detect_from_file(path)):
        return detected
    if metadata is not None and stat.S_ISDIR(metadata.st_mode) and (detected := _detect_from_directory(path)):
        return detected

    if detected := _detect_from_path_parts(path):
        return detected

    if metadata is not None and stat.S_ISDIR(metadata.st_mode) and (detected := _detect_from_nested_structure(path)):
        return detected

    return CONTENT_TYPE_UNKNOWN


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def resolve_skill_path(skill_path: Path) -> Path:
    """Convert SKILL.md file path to its parent directory."""
    try:
        metadata = skill_path.lstat()
    except OSError:
        return skill_path
    is_manifest_link = stat_is_link_or_reparse(metadata) and skill_path.name in SKILL_MANIFEST_VARIANTS
    return skill_path.parent if stat.S_ISREG(metadata.st_mode) or is_manifest_link else skill_path


def resolve_rules_path(rules_path: Path) -> Path:
    """Return path as-is for rules (can be file or directory)."""
    return rules_path


def resolve_workflows_path(workflows_path: Path) -> Path:
    """Convert workflow-rules.mdc path to its parent directory."""
    try:
        metadata = workflows_path.lstat()
    except OSError:
        return workflows_path
    if not stat.S_ISDIR(metadata.st_mode) and workflows_path.name == WORKFLOWS_MANIFEST_FILE:
        return workflows_path.parent
    return workflows_path


def resolve_plugin_path(path: Path) -> Path:
    """Convert a plugin manifest file path to the plugin root directory."""
    try:
        metadata = path.lstat()
    except OSError:
        return path
    if not stat.S_ISDIR(metadata.st_mode):
        if path.name in PLUGIN_MANIFEST_FILES:
            return path.parent
        if _is_contained_plugin_manifest(path):
            return path.parent.parent
    return path


def resolve_content_path(path: Path, content_type: str) -> Path:
    """Normalize a direct manifest path for the selected content type."""
    resolvers = {
        CONTENT_TYPE_SKILL: resolve_skill_path,
        CONTENT_TYPE_RULES: resolve_rules_path,
        CONTENT_TYPE_WORKFLOWS: resolve_workflows_path,
        CONTENT_TYPE_PLUGIN: resolve_plugin_path,
    }
    resolver = resolvers.get(content_type)
    return resolver(path) if resolver else path
