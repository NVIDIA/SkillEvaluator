# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, local plugin profiles for local-catalog Tier 2 comparisons.

A profile holds only what the local catalog Checks B and C-inter need: the
manifest name and description, the member skill names, the bundled skill
manifests, and a credential-free source identity (the SHA-256 of the manifest
bytes). Every read goes through the no-follow, root-contained plugin manifest
locator and the bounded embedding extractor; nothing is fetched remotely.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from skillevaluator.constants import (
    DESCRIPTION_MAX_LENGTH,
    NAME_MAX_LENGTH,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE,
    PLUGIN_CATALOG_MAX_MEMBER_CHARS,
    PLUGIN_CATALOG_MAX_MEMBERS,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    PLUGIN_MANIFEST_FILES,
    PLUGIN_NATIVE_MANIFEST_DIRS,
    SCAN_EXCLUDED_DIRS,
    SIMILARITY_MAX_DISCOVERED_PATHS,
)
from skillevaluator.deduplication.plugin.ref_utils import normalize_ref
from skillevaluator.embedding.extractor import (
    CollectionLimitError,
    ContentEntry,
    ExtractionBudget,
    extract_skill_manifest,
)
from skillevaluator.plugin_formats import parse_manifest_text
from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest
from skillevaluator.utils.helpers import find_bundled_plugin_skill_manifests
from skillevaluator.utils.secure_fs import SecureFile, SecurePathError, SecureRoot, discover_secure_files
from skillevaluator.utils.structured_data import StructuredDataError, require_bounded_string

MAX_PLUGIN_MEMBERS = PLUGIN_CATALOG_MAX_MEMBERS
MAX_PLUGIN_MEMBER_CHARS = PLUGIN_CATALOG_MAX_MEMBER_CHARS


class PluginProfileError(ValueError):
    """Raised when a plugin manifest cannot supply a comparable profile."""


class PluginSkillLimitError(PluginProfileError):
    """Raised before reading more bundled skills than the caller allows."""

    def __init__(self, actual: int, limit: int) -> None:
        super().__init__(f"Plugin bundles {actual} skills, exceeding the limit of {limit}")
        self.actual = actual
        self.limit = limit


@dataclass(frozen=True)
class BundledSkill:
    """One live skill under ``<plugin>/skills`` and its bounded manifest fields."""

    rel: str
    path: Path
    entry: ContentEntry | None
    skip_reason: str | None = None

    @property
    def root_relative(self) -> str:
        """Plugin-root-relative identity, matching Tier 1 ``bundled_skills``."""
        return f"skills/{self.rel}"


@dataclass(frozen=True)
class PluginProfile:
    """Comparable, bounded plugin facts for local catalog checks."""

    root: Path
    name: str
    description: str | None
    manifest: str
    source_fingerprint: str
    members: tuple[str, ...]
    bundled_skills: tuple[BundledSkill, ...]

    @property
    def embedding_text(self) -> str:
        """Text embedded for plugin entries; mirrors ``ContentEntry.embedding_text``."""
        return plugin_embedding_text(self.name, self.description or "")


def plugin_embedding_text(name: str, description: str) -> str:
    return f"{name}: {description}"


def member_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    """Return the Jaccard overlap of two member-name collections (0.0 when both are empty)."""
    left_set = {item.casefold() for item in left}
    right_set = {item.casefold() for item in right}
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _member_name(value: str) -> str:
    member = value.strip().casefold()
    if len(member) > MAX_PLUGIN_MEMBER_CHARS:
        raise PluginProfileError(f"Plugin member skill name exceeds {MAX_PLUGIN_MEMBER_CHARS} characters")
    return member


def _ref_member_name(ref: Any) -> str | None:
    """Return the leaf skill name of a normalized ``skills.refs`` entry."""
    canonical = normalize_ref(ref)
    if canonical is None:
        return None
    segments = canonical.split("::")
    tail = segments[-1] if len(segments) == 4 else canonical
    leaf = tail.rstrip("/").rsplit("/", 1)[-1].strip()
    return leaf or None


def _load_manifest_data(plugin_root: Path, budget: ExtractionBudget | None) -> tuple[dict[str, Any], str, str]:
    located = locate_plugin_manifest(plugin_root)
    if located is None:
        raise PluginProfileError(
            "No plugin manifest (agent_plugin.yaml/.yml, a .claude-plugin/, .codex-plugin/, or .cursor-plugin/ "
            "plugin.json, or an Agent Plugins root plugin.json) found"
        )
    raw = located.read_text()
    if budget is not None:
        budget.consume_bytes(len(raw.encode("utf-8")))
    try:
        data: Any = parse_manifest_text(located.manifest_type, raw)
    except StructuredDataError as exc:
        raise PluginProfileError("Plugin manifest could not be parsed within structured-data limits") from exc
    if not isinstance(data, dict):
        raise PluginProfileError("Plugin manifest is not a mapping")
    fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return data, located.manifest_filename, fingerprint


def _bundled_skills(
    plugin_root: Path,
    *,
    max_skills: int | None,
    budget: ExtractionBudget | None,
    skill_manifests: Sequence[SecureFile] | None,
) -> tuple[BundledSkill, ...]:
    manifests = find_bundled_plugin_skill_manifests(plugin_root) if skill_manifests is None else skill_manifests
    if max_skills is not None and len(manifests) > max_skills:
        raise PluginSkillLimitError(len(manifests), max_skills)
    if not manifests:
        return ()
    skills_root = plugin_root / "skills"
    bundled: list[BundledSkill] = []
    # One anchored root reads every manifest, each against its discovery identity.
    with SecureRoot(skills_root) as secure_root:
        for manifest in manifests:
            folder = manifest.relative_path.parent
            rel = folder.as_posix()
            # Without a collection budget (one plugin, at most ``max_skills``
            # manifests) each manifest is bounded on its own.
            manifest_budget = ExtractionBudget() if budget is None else budget
            try:
                entry = extract_skill_manifest(secure_root, manifest, budget=manifest_budget, display_root=skills_root)
            except (SecurePathError, CollectionLimitError):
                raise
            except ValueError as exc:
                # Tier 1 owns malformed skill metadata; this comparison skips it.
                # Its bytes were already reserved in the budget before the read.
                bundled.append(BundledSkill(rel=rel, path=skills_root / folder, entry=None, skip_reason=str(exc)))
                continue
            reason = None if entry is not None else "SKILL.md lacks a name or description in its frontmatter"
            bundled.append(BundledSkill(rel=rel, path=skills_root / folder, entry=entry, skip_reason=reason))
    return tuple(bundled)


def load_plugin_profile(
    plugin_root: Path,
    *,
    max_skills: int | None = None,
    budget: ExtractionBudget | None = None,
    skill_manifests: Sequence[SecureFile] | None = None,
) -> PluginProfile:
    """Load one plugin's comparable profile through the secure bounded readers.

    ``budget`` bounds the bytes read across a collection of plugins: the
    manifest and every bundled skill manifest count, including one that cannot
    be parsed. ``skill_manifests`` are the plugin's bundled skill manifests
    (from :func:`find_bundled_plugin_skill_manifests`) when the caller has
    already discovered them. Raises :class:`PluginManifestPathError` (or
    ``SecurePathError``) for unsafe inputs, ``ValueError`` for unsafe bundled
    skill discovery or an exhausted budget, and :class:`PluginProfileError`
    when the manifest cannot supply a profile.
    """
    data, manifest, fingerprint = _load_manifest_data(plugin_root, budget)

    raw_name = data.get("name")
    if raw_name is None or (isinstance(raw_name, str) and not raw_name.strip()):
        raw_name = plugin_root.resolve().name
    try:
        name = require_bounded_string(raw_name, "Plugin name", max_chars=NAME_MAX_LENGTH).strip()
    except ValueError as exc:
        raise PluginProfileError(f"Invalid plugin name: {exc}") from exc

    raw_description = data.get("description")
    description: str | None = None
    if raw_description is not None and not (isinstance(raw_description, str) and not raw_description.strip()):
        try:
            description = require_bounded_string(
                raw_description,
                "Plugin description",
                max_chars=DESCRIPTION_MAX_LENGTH,
            ).strip()
        except ValueError as exc:
            raise PluginProfileError(f"Invalid plugin description: {exc}") from exc

    members: set[str] = set()
    skills_section = data.get("skills")
    refs = skills_section.get("refs") if isinstance(skills_section, dict) else None
    if isinstance(refs, list):
        if len(refs) > MAX_PLUGIN_MEMBERS:
            raise PluginProfileError(f"Plugin skills.refs exceeds the {MAX_PLUGIN_MEMBERS}-item limit")
        for ref in refs:
            leaf = _ref_member_name(ref)
            if leaf:
                members.add(_member_name(leaf))

    bundled = _bundled_skills(plugin_root, max_skills=max_skills, budget=budget, skill_manifests=skill_manifests)
    for skill in bundled:
        label = skill.entry.name if skill.entry is not None else PurePosixPath(skill.rel).name
        if label.strip():
            members.add(_member_name(label))
    if len(members) > MAX_PLUGIN_MEMBERS:
        raise PluginProfileError(f"Plugin member skills exceed the {MAX_PLUGIN_MEMBERS}-item limit")

    return PluginProfile(
        root=plugin_root,
        name=name,
        description=description,
        manifest=manifest,
        source_fingerprint=fingerprint,
        members=tuple(sorted(members)),
        bundled_skills=bundled,
    )


def discover_plugin_roots(root: Path, *, max_plugins: int) -> list[Path]:
    """Return plugin roots at or below ``root`` without following redirects.

    ``root`` itself is returned when it is a plugin. Otherwise every regular
    plugin manifest below it (bounded by the similarity path budget) marks a
    plugin root. Unsafe links fail closed through the secure discovery walk.
    """
    if locate_plugin_manifest(root) is not None:
        return [root]

    def _is_vendor_manifest(posix: PurePosixPath) -> bool:
        return (
            posix.name == PLUGIN_CONTAINED_MANIFEST_FILE
            and len(posix.parts) >= 2
            and posix.parts[-2] in PLUGIN_NATIVE_MANIFEST_DIRS
        )

    def _selected(relative: Path) -> bool:
        posix = PurePosixPath(relative.as_posix())
        return (
            posix.name in PLUGIN_MANIFEST_FILES
            or posix.name == PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE
            or _is_vendor_manifest(posix)
        )

    files = discover_secure_files(
        root,
        selected=_selected,
        excluded_dirs=SCAN_EXCLUDED_DIRS,
        max_paths=SIMILARITY_MAX_DISCOVERED_PATHS,
    )
    plugin_dirs: set[PurePosixPath] = set()
    for file in files:
        relative = PurePosixPath(file.relative_path.as_posix())
        plugin_dir = relative.parent.parent if _is_vendor_manifest(relative) else relative.parent
        # A root plugin.json roots a plugin only when it is an Agent Plugins manifest.
        if not _is_vendor_manifest(relative) and relative.name == PLUGIN_AGENT_PLUGINS_V1_MANIFEST_FILE:
            try:
                located = locate_plugin_manifest(root / Path(*plugin_dir.parts))
            except PluginManifestPathError:
                located = None
            if located is None:
                continue
        plugin_dirs.add(plugin_dir)
        if len(plugin_dirs) > max_plugins:
            raise ValueError(
                f"Collection entry limit exceeded ({max_plugins}) before embedding; "
                "increase --max-entries within its supported range to scan the complete collection"
            )
    return [root / Path(*plugin_dir.parts) for plugin_dir in sorted(plugin_dirs, key=lambda item: item.as_posix())]


__all__ = [
    "MAX_PLUGIN_MEMBERS",
    "MAX_PLUGIN_MEMBER_CHARS",
    "BundledSkill",
    "PluginManifestPathError",
    "PluginProfile",
    "PluginProfileError",
    "PluginSkillLimitError",
    "discover_plugin_roots",
    "load_plugin_profile",
    "member_overlap",
    "plugin_embedding_text",
]
