# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified content extraction for similarity detection.

Normalizes Skills (SKILL.md), Rules (.mdc), and Workflows
(workflow-rules.mdc) into a common ContentEntry dataclass so
the embedding pipeline can treat all three types identically.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

import yaml

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_FILES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_WORKFLOWS,
    RULES_FILE_EXTENSION,
    SCAN_EXCLUDED_DIRS,
    SKILL_MANIFEST_VARIANTS,
    WORKFLOWS_MANIFEST_FILE,
)
from skillevaluator.deduplication.utils.skill_collector import _SecureReadError, _SecureRoot
from skillevaluator.logging_config import get_logger
from skillevaluator.utils.tier2_paths import safe_path_label
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

logger = get_logger(__name__)

MAX_COLLECTION_ENTRIES = CONTENT_DEDUP_MAX_FILES
MAX_MANIFEST_BYTES = CONTENT_DEDUP_MAX_FILE_BYTES
MAX_COLLECTION_BYTES = CONTENT_DEDUP_MAX_TOTAL_BYTES
MAX_DISCOVERED_PATHS = CONTENT_DEDUP_MAX_DISCOVERED_PATHS
DISCOVERY_EXCLUDED_DIRS = SCAN_EXCLUDED_DIRS


@dataclass
class ContentEntry:
    """Unified representation of a content item for embedding.

    Attributes:
        name: Skill name or rule/workflow title.
        description: Description from frontmatter.
        path: File or directory path (as string for serialization).
        content_type: One of "skill", "rules", "workflows".
        full_text: Entire file content including frontmatter, used by --full-body.
    """

    name: str
    description: str
    path: str
    content_type: str
    full_text: str = ""

    @property
    def embedding_text(self) -> str:
        """Concatenated string used for description-only embedding."""
        return f"{self.name}: {self.description}"


@dataclass
class _ExtractionBudget:
    """Tracks collection bounds before any embedding request is made."""

    entry_count: int = 0
    total_bytes: int = 0

    def reserve(self, file_path: Path, declared_bytes: int) -> None:
        self.entry_count += 1
        if self.entry_count > MAX_COLLECTION_ENTRIES:
            raise ValueError(f"Collection entry limit exceeded ({MAX_COLLECTION_ENTRIES}) before embedding")
        if declared_bytes > MAX_MANIFEST_BYTES:
            raise ValueError(
                f"Manifest exceeds the Tier 2 per-file byte limit ({MAX_MANIFEST_BYTES}): {file_path.name}"
            )
        self.total_bytes += declared_bytes
        if self.total_bytes > MAX_COLLECTION_BYTES:
            raise ValueError(f"Collection total byte limit exceeded ({MAX_COLLECTION_BYTES}) before embedding")

    def reconcile(self, declared_bytes: int, actual_bytes: int) -> None:
        self.total_bytes += actual_bytes - declared_bytes
        if self.total_bytes > MAX_COLLECTION_BYTES:
            raise ValueError(f"Collection total byte limit exceeded ({MAX_COLLECTION_BYTES}) before embedding")


# ---------------------------------------------------------------------------
# Per-type extraction functions
# ---------------------------------------------------------------------------


def extract_from_skill(skill_dir: Path, *, budget: _ExtractionBudget | None = None) -> ContentEntry | None:
    """Extract name + description from a skill directory's SKILL.md."""
    try:
        with _SecureRoot(skill_dir) as secure_root:
            for variant in SKILL_MANIFEST_VARIANTS:
                manifest = skill_dir / variant
                if manifest.exists() or _is_symlink_or_reparse(manifest):
                    _ensure_safe_content_path(skill_dir, manifest)
                    return _extract_from_file(
                        manifest,
                        name_field="name",
                        description_field="description",
                        content_type=CONTENT_TYPE_SKILL,
                        budget=budget,
                        secure_root=secure_root,
                        relative_path=Path(variant),
                    )
    except _SecureReadError as exc:
        raise ValueError(f"Skill root is a symlink, reparse point, or cannot be securely opened: {skill_dir}") from exc
    logger.debug("No SKILL.md found in %s", skill_dir)
    return None


def extract_from_rule(rule_path: Path, *, budget: _ExtractionBudget | None = None) -> ContentEntry | None:
    """Extract title + description from a .mdc rules file."""
    if rule_path.suffix != RULES_FILE_EXTENSION:
        logger.debug("Not a valid .mdc file: %s", rule_path)
        return None
    try:
        with _SecureRoot(rule_path.parent) as secure_root:
            if not rule_path.exists() and not _is_symlink_or_reparse(rule_path):
                logger.debug("Not a valid .mdc file: %s", rule_path)
                return None
            _ensure_safe_content_path(rule_path.parent, rule_path)
            return _extract_from_file(
                rule_path,
                name_field="title",
                description_field="description",
                content_type=CONTENT_TYPE_RULES,
                budget=budget,
                secure_root=secure_root,
                relative_path=Path(rule_path.name),
            )
    except _SecureReadError as exc:
        raise ValueError(
            f"Rule root is a symlink, reparse point, or cannot be securely opened: {rule_path.parent}"
        ) from exc


def extract_from_workflow(workflow_dir: Path, *, budget: _ExtractionBudget | None = None) -> ContentEntry | None:
    """Extract title + description from a workflow's workflow-rules.mdc."""
    try:
        with _SecureRoot(workflow_dir) as secure_root:
            manifest = workflow_dir / WORKFLOWS_MANIFEST_FILE
            if not manifest.exists() and not _is_symlink_or_reparse(manifest):
                logger.debug("No %s found in %s", WORKFLOWS_MANIFEST_FILE, workflow_dir)
                return None
            _ensure_safe_content_path(workflow_dir, manifest)
            return _extract_from_file(
                manifest,
                name_field="title",
                description_field="description",
                content_type=CONTENT_TYPE_WORKFLOWS,
                budget=budget,
                secure_root=secure_root,
                relative_path=Path(WORKFLOWS_MANIFEST_FILE),
            )
    except _SecureReadError as exc:
        raise ValueError(
            f"Workflow root is a symlink, reparse point, or cannot be securely opened: {workflow_dir}"
        ) from exc


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_and_extract(root: Path, content_type: str) -> list[ContentEntry]:
    """Auto-discover content items under a directory and extract entries.

    Args:
        root: Root directory to scan.
        content_type: One of "skill", "rules", "workflows".

    Returns:
        List of successfully extracted ContentEntry objects.
    """
    strategy = _DISCOVERY_STRATEGIES.get(content_type)
    if strategy is None:
        logger.warning("Unknown content type '%s' for discovery", content_type)
        return []

    entries = strategy(root, _ExtractionBudget())
    logger.debug("Discovered %d %s entries in %s", len(entries), content_type, safe_path_label(root))
    return entries


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_symlink_or_reparse(path: Path) -> bool:
    """Return whether *path* redirects traversal through a link or reparse point."""
    try:
        metadata = path.lstat()
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_flag)


def _ensure_safe_content_path(root: Path, candidate: Path) -> None:
    """Reject linked manifests and any path that resolves outside *root*."""
    root_absolute = root.absolute()
    candidate_absolute = candidate.absolute()
    try:
        relative = candidate_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"Content path escapes scan root: {candidate}") from exc

    current = root_absolute
    for part in (None, *relative.parts):
        if part is not None:
            current /= part
        if _is_symlink_or_reparse(current):
            raise ValueError(f"Content path contains a symlink or reparse point: {current}")

    try:
        resolved_root = root_absolute.resolve(strict=True)
        resolved_candidate = candidate_absolute.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Content path resolves outside its scan root or does not exist: {candidate}") from exc


def _extract_from_file(
    file_path: Path,
    *,
    name_field: str,
    description_field: str,
    content_type: str,
    budget: _ExtractionBudget | None = None,
    secure_root: _SecureRoot | None = None,
    relative_path: Path | None = None,
) -> ContentEntry | None:
    """Read once, parse frontmatter, and build a bounded content entry."""
    full_text = _read_bounded_manifest(
        file_path,
        budget,
        secure_root=secure_root,
        relative_path=relative_path,
    )
    match = FRONTMATTER_PATTERN.match(full_text)
    if not match:
        logger.debug("Missing YAML frontmatter in %s", file_path)
        return None

    frontmatter_yaml, _markdown_content = match.groups()
    try:
        data = yaml.safe_load(frontmatter_yaml)
    except yaml.YAMLError as exc:
        logger.debug("Invalid YAML frontmatter in %s: %s", file_path, exc)
        return None
    if not data or not isinstance(data, dict):
        logger.debug("Frontmatter in %s is not a non-empty mapping", file_path)
        return None
    name = data.get(name_field)
    description = data.get(description_field)

    if not name or not description:
        logger.debug(
            "Missing %s or %s in %s",
            name_field,
            description_field,
            file_path,
        )
        return None

    return ContentEntry(
        name=str(name),
        description=str(description),
        path=str(file_path.parent if content_type == CONTENT_TYPE_SKILL else file_path),
        content_type=content_type,
        full_text=full_text,
    )


def _read_bounded_manifest(
    file_path: Path,
    budget: _ExtractionBudget | None,
    *,
    secure_root: _SecureRoot | None = None,
    relative_path: Path | None = None,
) -> str:
    """Read one regular UTF-8 manifest through its anchored parent directory."""
    try:
        if secure_root is None:
            with _SecureRoot(file_path.parent) as manifest_root:
                raw_bytes, opened_info = manifest_root.read_bounded(Path(file_path.name), MAX_MANIFEST_BYTES)
        else:
            if relative_path is None:
                raise _SecureReadError("unsafe_path", "Anchored manifest read requires a relative path.")
            raw_bytes, opened_info = secure_root.read_bounded(relative_path, MAX_MANIFEST_BYTES)
    except _SecureReadError as exc:
        raise ValueError(f"Cannot securely read manifest {file_path.name}: {exc}") from exc
    if budget is not None:
        budget.reserve(file_path, opened_info.st_size)
        budget.reconcile(opened_info.st_size, len(raw_bytes))
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Manifest is not valid UTF-8: {file_path.name}") from exc


def _iter_discovery_files(root: Path):
    """Yield bounded files from one pruned inter-skill traversal."""
    if _is_symlink_or_reparse(root):
        raise ValueError(f"Discovery root is a symlink or reparse point: {root.name}")
    if not root.is_dir():
        return

    discovered_paths = 0

    def _raise_walk_error(error: OSError) -> None:
        raise ValueError(f"Cannot safely traverse the Tier 2 collection: {error}") from error

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=_raise_walk_error, followlinks=False):
        dirnames.sort()
        filenames.sort()
        kept_dirs: list[str] = []
        for dirname in dirnames:
            discovered_paths += 1
            if discovered_paths > MAX_DISCOVERED_PATHS:
                raise ValueError(f"Collection path limit exceeded ({MAX_DISCOVERED_PATHS}) before embedding")
            if dirname in DISCOVERY_EXCLUDED_DIRS:
                continue
            directory = Path(dirpath) / dirname
            if _is_symlink_or_reparse(directory):
                raise ValueError(f"Discovery path contains a symlink or reparse point: {dirname}")
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            discovered_paths += 1
            if discovered_paths > MAX_DISCOVERED_PATHS:
                raise ValueError(f"Collection path limit exceeded ({MAX_DISCOVERED_PATHS}) before embedding")
            file_path = Path(dirpath) / filename
            if _is_symlink_or_reparse(file_path):
                raise ValueError(f"Discovery path contains a symlink or reparse point: {filename}")
            yield file_path


def _discover_skills(root: Path, budget: _ExtractionBudget) -> list[ContentEntry]:
    """Find skill directories by locating SKILL.md files."""
    entries: list[ContentEntry] = []
    seen_dirs: set[Path] = set()
    with _SecureRoot(root) as secure_root:
        for manifest in _iter_discovery_files(root):
            if manifest.name not in SKILL_MANIFEST_VARIANTS:
                continue
            skill_dir = manifest.parent
            if skill_dir in seen_dirs:
                continue
            seen_dirs.add(skill_dir)
            entry = _extract_from_file(
                manifest,
                name_field="name",
                description_field="description",
                content_type=CONTENT_TYPE_SKILL,
                budget=budget,
                secure_root=secure_root,
                relative_path=manifest.relative_to(root),
            )
            if entry:
                entries.append(entry)
    return entries


def _discover_rules(root: Path, budget: _ExtractionBudget) -> list[ContentEntry]:
    """Find .mdc rule files recursively, excluding workflow manifests."""
    entries: list[ContentEntry] = []
    with _SecureRoot(root) as secure_root:
        for mdc_file in _iter_discovery_files(root):
            if mdc_file.suffix != RULES_FILE_EXTENSION or mdc_file.name == WORKFLOWS_MANIFEST_FILE:
                continue
            entry = _extract_from_file(
                mdc_file,
                name_field="title",
                description_field="description",
                content_type=CONTENT_TYPE_RULES,
                budget=budget,
                secure_root=secure_root,
                relative_path=mdc_file.relative_to(root),
            )
            if entry:
                entries.append(entry)
    return entries


def _discover_workflows(root: Path, budget: _ExtractionBudget) -> list[ContentEntry]:
    """Find workflow directories by locating workflow-rules.mdc."""
    entries: list[ContentEntry] = []
    with _SecureRoot(root) as secure_root:
        for manifest in _iter_discovery_files(root):
            if manifest.name != WORKFLOWS_MANIFEST_FILE:
                continue
            entry = _extract_from_file(
                manifest,
                name_field="title",
                description_field="description",
                content_type=CONTENT_TYPE_WORKFLOWS,
                budget=budget,
                secure_root=secure_root,
                relative_path=manifest.relative_to(root),
            )
            if entry:
                entries.append(entry)
    return entries


_DISCOVERY_STRATEGIES: dict[str, callable] = {
    CONTENT_TYPE_SKILL: _discover_skills,
    CONTENT_TYPE_RULES: _discover_rules,
    CONTENT_TYPE_WORKFLOWS: _discover_workflows,
}
