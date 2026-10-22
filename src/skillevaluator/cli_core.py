# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-type detection and path resolution for the SkillEvaluator CLI.

Pure helpers with no Click/console/logging dependencies, shared by the
:mod:`skillevaluator.cli` entry point and :mod:`skillevaluator.validators`. The
Click command group itself lives in :mod:`skillevaluator.cli`.
"""

import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_TYPE_PLUGIN,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_UNKNOWN,
    CONTENT_TYPE_WORKFLOWS,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_MANIFEST_FILES,
    RULES_FILE_EXTENSION,
    SKILL_MANIFEST_FILE,
    SKILL_MANIFEST_VARIANTS,
    WORKFLOWS_MANIFEST_FILE,
)
from skillevaluator.plugin_manifest import (
    NATIVE_MANIFEST_DIRS_FOLDED,
    agent_plugins_path_opt_in,
    manifest_relative_path,
    manifest_root_for,
    manifest_type_for_relative_path,
)
from skillevaluator.utils.secure_fs import SecurePathError, stat_is_link_or_reparse

# ---------------------------------------------------------------------------
# Content-type detection
# ---------------------------------------------------------------------------


def _root_plugin_json_opts_in(path: Path) -> bool:
    """Return whether the root ``plugin.json`` at *path* marks a plugin.

    A root ``plugin.json`` is an Agent Plugins v1 manifest only when it opts in
    with that ``$schema``; any other ``plugin.json`` is not a plugin manifest.
    The decision and the bounded, no-follow read are the plugin locator's
    (:func:`~skillevaluator.plugin_manifest.agent_plugins_path_opt_in`), so
    detection and validation always agree: unparseable or non-UTF-8 JSON that
    names the Agent Plugins schema host still marks a plugin, so its error is
    reported rather than hidden, and a file over the manifest size bound is
    parsed whole up to the lenient bound. A file that cannot be read safely (a
    link, a hard-linked or special file, or one that changes while it is read)
    also marks a plugin: the locator refuses it, so validation fails closed
    instead of checking the folder as a skill without the plugin checks.
    """
    try:
        return agent_plugins_path_opt_in(path)
    except (OSError, SecurePathError, ValueError):
        return True


def _detect_from_file(path: Path) -> str | None:
    """Detect content type from a file path."""
    # Every plugin manifest name marks a plugin (the locator's lexical rule),
    # except that a root plugin.json must also opt into Agent Plugins.
    relative = manifest_relative_path(path)
    if relative is not None and (
        manifest_type_for_relative_path(relative) != PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE
        or _root_plugin_json_opts_in(path)
    ):
        return CONTENT_TYPE_PLUGIN
    if path.name.upper() == SKILL_MANIFEST_FILE.upper():
        return CONTENT_TYPE_SKILL
    if path.suffix == RULES_FILE_EXTENSION:
        parent = path.parent
        if parent.name == "references" or path.name == WORKFLOWS_MANIFEST_FILE:
            return CONTENT_TYPE_WORKFLOWS
        return CONTENT_TYPE_RULES
    return None


# Content folders in detection precedence order. A path that passes through one
# (``.../skills/<name>``), or a directory that holds one as a real subdirectory,
# is that content type.
_STRUCTURE_MARKERS: tuple[tuple[str, frozenset[str]], ...] = (
    (CONTENT_TYPE_SKILL, frozenset({"skills", "team-skills"})),
    (CONTENT_TYPE_RULES, frozenset({"team-rules"})),
    (CONTENT_TYPE_WORKFLOWS, frozenset({"workflows", "team-workflows"})),
)
_STRUCTURE_MARKER_NAMES = frozenset(name for _content_type, names in _STRUCTURE_MARKERS for name in names)


def _structure_type(names: Iterable[str]) -> str | None:
    """Return the content type of the first content folder, by precedence, among *names*."""
    present = frozenset(names)
    for content_type, markers in _STRUCTURE_MARKERS:
        if not markers.isdisjoint(present):
            return content_type
    return None


@dataclass(frozen=True)
class _RootMarkers:
    """What one listing of a directory's top level found (see :func:`_root_markers`)."""

    # A plugin manifest, or a vendor manifest directory (.claude-plugin/, ...).
    plugin: bool
    skill: bool
    workflows: bool
    rules: bool
    # Content folders (skills/, team-rules/, ...) that are real directories.
    structure_dirs: frozenset[str]

    @property
    def content_type(self) -> str | None:
        """The content type the directory's own manifests and files mark.

        A plugin manifest at the root -- agent_plugin.yaml/.yml
        (bundle-reference), a .claude-plugin/, .codex-plugin/, or
        .cursor-plugin/ plugin.json, or an Agent Plugins root plugin.json
        (contained) -- wins: a plugin may also contain skills/**/SKILL.md.
        """
        if self.plugin:
            return CONTENT_TYPE_PLUGIN
        if self.skill:
            return CONTENT_TYPE_SKILL
        if self.workflows:
            return CONTENT_TYPE_WORKFLOWS
        if self.rules:
            return CONTENT_TYPE_RULES
        return None


def _may_mark_content(name: str) -> bool:
    """Whether a top-level entry named *name* can mark a content type (only these are inspected)."""
    folded = name.casefold()
    return (
        name in PLUGIN_MANIFEST_FILES
        or name in SKILL_MANIFEST_VARIANTS
        or name == WORKFLOWS_MANIFEST_FILE
        or name in _STRUCTURE_MARKER_NAMES
        or name.endswith(RULES_FILE_EXTENSION)
        or folded == PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE
        or folded in NATIVE_MANIFEST_DIRS_FOLDED
    )


def _root_markers(path: Path) -> _RootMarkers | None:
    """List the top level of the directory *path* once, bounded and without following links.

    Only entries that can mark a content type are inspected, each with
    ``lstat``. ``None`` means *path* is not a regular directory, cannot be
    listed or inspected, or has more than ``CONTENT_DEDUP_MAX_DISCOVERED_PATHS``
    entries.
    """
    try:
        root_metadata = path.lstat()
    except OSError:
        return None
    if stat_is_link_or_reparse(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
        return None
    plugin = skill = workflows = rules = False
    structure_dirs: set[str] = set()
    try:
        with os.scandir(path) as iterator:
            for count, entry in enumerate(iterator, start=1):
                if count > CONTENT_DEDUP_MAX_DISCOVERED_PATHS:
                    return None
                name = entry.name
                if not _may_mark_content(name):
                    continue
                folded = name.casefold()
                metadata = entry.stat(follow_symlinks=False)
                non_directory = not stat.S_ISDIR(metadata.st_mode)
                if name in _STRUCTURE_MARKER_NAMES:
                    if not non_directory and not stat_is_link_or_reparse(metadata):
                        structure_dirs.add(name)
                elif name in PLUGIN_MANIFEST_FILES and non_directory:
                    plugin = True
                elif folded == PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE:
                    # A regular root plugin.json marks a plugin when it declares
                    # the Agent Plugins $schema (bounded, no-follow read). A link
                    # or special file is never read: the locator refuses it, so
                    # it marks a plugin that then fails closed.
                    if non_directory and (not stat.S_ISREG(metadata.st_mode) or _root_plugin_json_opts_in(path / name)):
                        plugin = True
                elif folded in NATIVE_MANIFEST_DIRS_FOLDED:
                    # Presence is enough for auto-detection (any spelling: a
                    # case-insensitive client opens it). The secure plugin
                    # locator later distinguishes a real contained manifest
                    # from an empty, linked, or malformed marker directory.
                    plugin = True
                elif name in SKILL_MANIFEST_VARIANTS and non_directory:
                    skill = True
                elif name == WORKFLOWS_MANIFEST_FILE and non_directory:
                    workflows = True
                elif name.endswith(RULES_FILE_EXTENSION) and non_directory:
                    rules = True
    except OSError:
        return None
    return _RootMarkers(plugin, skill, workflows, rules, frozenset(structure_dirs))


def _directory_content_type(path: Path, markers: _RootMarkers) -> str | None:
    """The content type of the directory *path*, whose top level holds *markers*.

    Its own manifests and files decide first (:attr:`_RootMarkers.content_type`).
    Without any, Claude Code plugin components in their default locations
    (:func:`manifestless_plugin_markers`) make it a plugin.
    """
    if detected := markers.content_type:
        return detected
    return CONTENT_TYPE_PLUGIN if manifestless_plugin_markers(path) else None


def _detect_from_directory(path: Path) -> str | None:
    """Detect content type from directory contents."""
    markers = _root_markers(path)
    return _directory_content_type(path, markers) if markers is not None else None


# Claude Code default plugin locations that only a plugin uses: a folder of
# Markdown subagents, commands, or output styles, or a hooks, monitors, or LSP
# config. A root .mcp.json or skills/ folder is common outside plugins too, so
# neither alone marks a plugin.
_MARKDOWN_PLUGIN_DIRS = ("agents", "commands", "output-styles")
_CONFIG_PLUGIN_FILES = ("hooks/hooks.json", "monitors/monitors.json", ".lsp.json")
_MARKER_SCAN_LIMIT = 256


def _has_markdown_file(directory: Path) -> bool:
    """Whether a real directory directly holds a regular ``.md`` file (bounded, no links followed)."""
    try:
        metadata = directory.lstat()
        if stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            return False
        with os.scandir(directory) as iterator:
            for count, entry in enumerate(iterator, start=1):
                if count > _MARKER_SCAN_LIMIT:
                    return False
                if entry.name.lower().endswith(".md") and stat.S_ISREG(entry.stat(follow_symlinks=False).st_mode):
                    return True
    except OSError:
        return False
    return False


def manifestless_plugin_markers(path: Path) -> list[str]:
    """Claude Code default plugin locations present in a folder that has no plugin manifest.

    Claude Code ``--plugin-dir`` loads any folder as a plugin: without a
    ``.claude-plugin/plugin.json`` it reads its default locations and takes the
    plugin name from the folder. A folder with ``agents/``, ``commands/``, or
    ``output-styles/`` Markdown files, or a ``hooks/hooks.json``,
    ``monitors/monitors.json``, or ``.lsp.json``, is such a plugin, so it is
    validated as one (its subagent and command privileges included) rather than
    as a skill collection. Nothing is followed through a link.
    """
    markers = [f"{name}/" for name in _MARKDOWN_PLUGIN_DIRS if _has_markdown_file(path / name)]
    for relative in _CONFIG_PLUGIN_FILES:
        current = path
        try:
            for part in relative.split("/"):
                current = current / part
                metadata = current.lstat()
                if stat_is_link_or_reparse(metadata):
                    break
            else:
                if stat.S_ISREG(metadata.st_mode):
                    markers.append(relative)
        except OSError:
            continue
    return markers


def _detect_from_path_parts(path: Path) -> str | None:
    """Detect content type from folder path patterns."""
    return _structure_type(path.parts)


def _detect_from_nested_structure(path: Path) -> str | None:
    """Detect content type from a bounded, shallow structural marker scan."""
    markers = _root_markers(path)
    return _structure_type(markers.structure_dirs) if markers is not None else None


def detect_content_type(path: Path) -> str:
    """Auto-detect whether path contains a skill, rules, workflows, or plugin.

    Detection order: file type -> directory manifests -> path patterns -> nested structure.
    A plugin manifest at the root -- agent_plugin.yaml/.yml (bundle-reference), a
    vendor plugin.json (.claude-plugin/, .codex-plugin/, .cursor-plugin/), or an
    Agent Plugins root plugin.json (contained) -- wins over a nested skills tree.
    A folder without any manifest or root ``SKILL.md`` that ships Claude Code
    plugin components in their default locations (:func:`manifestless_plugin_markers`)
    is a plugin too: Claude Code ``--plugin-dir`` loads it.
    """
    try:
        metadata = path.lstat()
    except OSError:
        metadata = None
    if metadata is not None and not stat.S_ISDIR(metadata.st_mode) and (detected := _detect_from_file(path)):
        return detected
    # One listing of a directory serves the manifest and the nested-structure checks.
    markers = _root_markers(path) if metadata is not None and stat.S_ISDIR(metadata.st_mode) else None
    if markers is not None and (detected := _directory_content_type(path, markers)):
        return detected

    if detected := _detect_from_path_parts(path):
        return detected

    if markers is not None and (detected := _structure_type(markers.structure_dirs)):
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
        root = manifest_root_for(path)
        if root is not None:
            return root
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
