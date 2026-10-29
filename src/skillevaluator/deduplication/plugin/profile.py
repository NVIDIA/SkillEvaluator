# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded comparison facts projected from the shared private inventory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from skillevaluator.constants import DESCRIPTION_MAX_LENGTH
from skillevaluator.embedding.extractor import ContentEntry
from skillevaluator.plugin_formats import parse_manifest_text
from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest
from skillevaluator.plugin_refs import normalize_ref
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.utils.structured_data import require_bounded_string

PLUGIN_NAME_MAX_REPORT_CHARS = 256
MAX_PLUGIN_MEMBERS = 256
MAX_PLUGIN_MEMBER_CHARS = 256


def plugin_inventory_for_root(path: Path):
    # Keep the optional inventory composition outside ordinary skill imports.
    from skillevaluator.plugin_inventory import plugin_inventory_for_root as discover

    return discover(path)


@dataclass(frozen=True)
class BundledSkill:
    """A client-visible skill directory and its comparable manifest metadata."""

    root_relative: str
    path: Path
    entry: ContentEntry | None
    skip_reason: str | None = None

    @property
    def rel(self) -> str:
        """Legacy skills-root-relative identity when the skill lives under skills/."""
        return self.root_relative.removeprefix("skills/")


@dataclass(frozen=True)
class PluginProfile:
    root: Path
    name: str
    description: str | None
    manifest: str
    source_fingerprint: str
    members: tuple[str, ...]
    bundled_skills: tuple[BundledSkill, ...]

    @property
    def embedding_text(self) -> str:
        return f"{self.name}: {self.description or ''}"


def member_overlap(left, right) -> float:
    left_set, right_set = {s.casefold() for s in left}, {s.casefold() for s in right}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _member(value: str) -> str:
    return require_bounded_string(value.strip().casefold(), "Member name", max_chars=MAX_PLUGIN_MEMBER_CHARS)


def load_plugin_profile(plugin_root: Path) -> PluginProfile:
    """Discover fresh effective members and project only bounded comparison fields.

    Calling again after provider work intentionally obtains a new snapshot.
    An incomplete discovery never turns into a clean comparison result.
    """
    located = locate_plugin_manifest(plugin_root)
    if located is None:
        raise ValueError("No supported bundle manifest found")
    try:
        raw = located.read_text()
    except PluginManifestPathError as exc:
        if exc.content_error:
            raise ValueError("Bundle manifest exceeds readable comparison limits") from exc
        raise
    inventory = plugin_inventory_for_root(plugin_root)
    # Inventory composition converts read failures into findings. Recheck the
    # original secure identity so a source swap cannot become an advisory skip.
    try:
        located.read_text()
    except PluginManifestPathError as exc:
        raise SecurePathError("source_changed", "Bundle manifest changed during discovery") from exc
    if (
        inventory is None
        or not inventory.manifest_sha256
        or inventory.selected_manifest.manifest_filename != located.manifest_filename
        or inventory.manifest_sha256 != hashlib.sha256(raw.encode("utf-8")).hexdigest()
    ):
        raise SecurePathError("source_changed", "Bundle manifest identity changed during discovery")
    if not inventory.tree_complete:
        raise SecurePathError("source_scan_incomplete", "Bundle source tree discovery is incomplete")
    if getattr(inventory.discovery, "unread_files", 0) or any(
        member.problem in {"unsafe", "unreadable"} for member in inventory.selected_components
    ):
        raise SecurePathError("source_read_failed", "Bundle source paths could not be read safely")
    if not inventory.complete:
        raise ValueError("Bundle inventory is incomplete; comparisons did not run")
    data: Any = parse_manifest_text(located.manifest_type, raw.removeprefix("\ufeff"))
    if not isinstance(data, dict):
        raise ValueError("Bundle manifest could not be parsed")
    name = require_bounded_string(
        data.get("name") or inventory.root.name, "Bundle name", max_chars=PLUGIN_NAME_MAX_REPORT_CHARS
    ).strip()
    description = data.get("description")
    if description is not None:
        description = (
            require_bounded_string(
                description, "Bundle description", max_chars=DESCRIPTION_MAX_LENGTH, allow_empty=True
            ).strip()
            or None
        )
    bundled: list[BundledSkill] = []
    members: set[str] = set()
    seen: set[str] = set()
    for skill in inventory.selected_skills:
        if skill.path is None or skill.path in seen:
            continue
        relative = PurePosixPath(skill.path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in skill.path:
            raise ValueError("Unsafe skill directory in bundle inventory")
        seen.add(skill.path)
        metadata = skill.frontmatter
        skill_name, skill_description = metadata.get("name"), metadata.get("description")
        entry = None
        reason = "SKILL.md lacks a comparable name and description"
        if (
            isinstance(skill_name, str)
            and skill_name.strip()
            and isinstance(skill_description, str)
            and skill_description.strip()
        ):
            skill_name = require_bounded_string(skill_name, "Skill name", max_chars=MAX_PLUGIN_MEMBER_CHARS)
            skill_description = require_bounded_string(
                skill_description, "Skill description", max_chars=DESCRIPTION_MAX_LENGTH
            )
            entry = ContentEntry(
                skill_name, skill_description, str(inventory.root / skill.path), "skill", skill.text or ""
            )
            reason = None
        label = skill_name if isinstance(skill_name, str) and skill_name.strip() else relative.name or name
        members.add(_member(label))
        bundled.append(BundledSkill(relative.as_posix(), inventory.root / skill.path, entry, reason))
    section = data.get("skills")
    refs = section.get("refs") if isinstance(section, dict) else None
    if isinstance(refs, list):
        if len(refs) > MAX_PLUGIN_MEMBERS:
            raise ValueError("Bundle member reference limit exceeded")
        for ref in refs:
            canonical = normalize_ref(ref)
            if canonical:
                leaf = canonical.split("::")[-1].rstrip("/").rsplit("/", 1)[-1]
                if leaf.strip():
                    members.add(_member(leaf))
    if len(members) > MAX_PLUGIN_MEMBERS:
        raise ValueError("Bundle member count limit exceeded")
    return PluginProfile(
        inventory.root,
        name,
        description,
        located.manifest_filename,
        inventory.manifest_sha256,
        tuple(sorted(members)),
        tuple(bundled),
    )
