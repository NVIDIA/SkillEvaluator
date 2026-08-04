# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded no-follow extraction for similarity and catalog embedding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_FILES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_WORKFLOWS,
    DESCRIPTION_MAX_LENGTH,
    NAME_MAX_LENGTH,
    RULES_FILE_EXTENSION,
    SCAN_EXCLUDED_DIRS,
    SKILL_MANIFEST_VARIANTS,
    TITLE_MAX_LENGTH,
    WORKFLOWS_MANIFEST_FILE,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.utils.secure_fs import SecureFile, SecureRoot, discover_secure_files
from skillevaluator.utils.structured_data import (
    StructuredDataLimitError,
    StructuredDataSyntaxError,
    load_bounded_yaml,
    require_bounded_string,
)
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
    """Unified representation of a content item for embedding."""

    name: str
    description: str
    path: str
    content_type: str
    full_text: str = ""

    @property
    def embedding_text(self) -> str:
        return f"{self.name}: {self.description}"


@dataclass
class _ExtractionBudget:
    entry_count: int = 0
    total_bytes: int = 0

    def reserve(self, file: SecureFile) -> None:
        self.entry_count += 1
        if self.entry_count > MAX_COLLECTION_ENTRIES:
            raise ValueError(f"Collection entry limit exceeded ({MAX_COLLECTION_ENTRIES}) before embedding")
        declared_bytes = file.metadata.st_size
        if declared_bytes > MAX_MANIFEST_BYTES:
            raise ValueError(f"Manifest exceeds the Tier 2 per-file byte limit ({MAX_MANIFEST_BYTES}): {file.rel_path}")
        self.total_bytes += declared_bytes
        if self.total_bytes > MAX_COLLECTION_BYTES:
            raise ValueError(f"Collection total byte limit exceeded ({MAX_COLLECTION_BYTES}) before embedding")

    def reconcile(self, declared_bytes: int, actual_bytes: int) -> None:
        self.total_bytes += actual_bytes - declared_bytes
        if self.total_bytes > MAX_COLLECTION_BYTES:
            raise ValueError(f"Collection total byte limit exceeded ({MAX_COLLECTION_BYTES}) before embedding")


def _parse_frontmatter_text(file: SecureFile, raw_text: str) -> tuple[dict, str] | None:
    match = FRONTMATTER_PATTERN.match(raw_text)
    if not match:
        logger.debug("Manifest lacks YAML frontmatter: %s", file.rel_path)
        return None
    raw_yaml, content = match.groups()
    try:
        data = load_bounded_yaml(raw_yaml)
    except StructuredDataSyntaxError:
        logger.debug("Manifest contains invalid YAML frontmatter: %s", file.rel_path)
        return None
    except StructuredDataLimitError as exc:
        raise ValueError(f"Manifest structured-data complexity limit exceeded: {file.rel_path}") from exc
    if not isinstance(data, dict) or not data:
        logger.debug("Manifest frontmatter is not a mapping: %s", file.rel_path)
        return None
    return data, content


def _extract_secure_file(
    secure_root: SecureRoot,
    file: SecureFile,
    *,
    name_field: str,
    description_field: str,
    content_type: str,
    budget: _ExtractionBudget,
    display_root: Path,
) -> ContentEntry | None:
    budget.reserve(file)
    raw_text = secure_root.read_file_text(file, MAX_MANIFEST_BYTES)
    budget.reconcile(file.metadata.st_size, len(raw_text.encode("utf-8")))
    parsed = _parse_frontmatter_text(file, raw_text)
    if parsed is None:
        return None
    data, _content = parsed
    name = data.get(name_field)
    description = data.get(description_field)
    if name is None or description is None:
        logger.debug("Missing %s or %s in %s", name_field, description_field, file.rel_path)
        return None
    name_limit = NAME_MAX_LENGTH if name_field == "name" else TITLE_MAX_LENGTH
    name = require_bounded_string(name, name_field, max_chars=name_limit)
    description = require_bounded_string(
        description,
        description_field,
        max_chars=DESCRIPTION_MAX_LENGTH,
    )
    lexical_path = display_root / file.relative_path
    display_path = lexical_path.parent if content_type == CONTENT_TYPE_SKILL else lexical_path
    return ContentEntry(
        name=name,
        description=description,
        path=str(display_path),
        content_type=content_type,
        full_text=raw_text,
    )


def _discover(
    root: Path,
    *,
    selected: Callable[[Path], bool],
    max_depth: int | None = None,
) -> list[SecureFile]:
    return discover_secure_files(
        root,
        selected=selected,
        excluded_dirs=DISCOVERY_EXCLUDED_DIRS,
        max_paths=MAX_DISCOVERED_PATHS,
        max_depth=max_depth,
    )


def extract_from_skill(skill_dir: Path) -> ContentEntry | None:
    """Extract one regular SKILL.md/skill.md without following redirects."""
    files = _discover(
        skill_dir,
        selected=lambda relative: len(relative.parts) == 1 and relative.name in SKILL_MANIFEST_VARIANTS,
        max_depth=1,
    )
    by_name = {file.relative_path.name: file for file in files}
    selected_file = next((by_name[name] for name in SKILL_MANIFEST_VARIANTS if name in by_name), None)
    if selected_file is None:
        logger.debug("No SKILL.md found in %s", skill_dir)
        return None
    with SecureRoot(skill_dir) as secure_root:
        return _extract_secure_file(
            secure_root,
            selected_file,
            name_field="name",
            description_field="description",
            content_type=CONTENT_TYPE_SKILL,
            budget=_ExtractionBudget(),
            display_root=skill_dir,
        )


def extract_from_rule(rule_path: Path) -> ContentEntry | None:
    """Extract one selected regular .mdc rule file."""
    if rule_path.suffix != RULES_FILE_EXTENSION:
        logger.debug("Not a valid .mdc file: %s", rule_path)
        return None
    files = _discover(
        rule_path.parent,
        selected=lambda relative: len(relative.parts) == 1 and relative.name == rule_path.name,
        max_depth=1,
    )
    if not files:
        logger.debug("Not a valid .mdc file: %s", rule_path)
        return None
    with SecureRoot(rule_path.parent) as secure_root:
        return _extract_secure_file(
            secure_root,
            files[0],
            name_field="title",
            description_field="description",
            content_type=CONTENT_TYPE_RULES,
            budget=_ExtractionBudget(),
            display_root=rule_path.parent,
        )


def extract_from_workflow(workflow_dir: Path) -> ContentEntry | None:
    """Extract one regular workflow-rules.mdc manifest."""
    files = _discover(
        workflow_dir,
        selected=lambda relative: len(relative.parts) == 1 and relative.name == WORKFLOWS_MANIFEST_FILE,
        max_depth=1,
    )
    if not files:
        logger.debug("No %s found in %s", WORKFLOWS_MANIFEST_FILE, workflow_dir)
        return None
    with SecureRoot(workflow_dir) as secure_root:
        return _extract_secure_file(
            secure_root,
            files[0],
            name_field="title",
            description_field="description",
            content_type=CONTENT_TYPE_WORKFLOWS,
            budget=_ExtractionBudget(),
            display_root=workflow_dir,
        )


def discover_and_extract(root: Path, content_type: str) -> list[ContentEntry]:
    """Discover and extract one bounded content collection before embedding."""
    selectors: dict[str, Callable[[Path], bool]] = {
        CONTENT_TYPE_SKILL: lambda relative: relative.name in SKILL_MANIFEST_VARIANTS,
        CONTENT_TYPE_RULES: lambda relative: (
            relative.suffix == RULES_FILE_EXTENSION and relative.name != WORKFLOWS_MANIFEST_FILE
        ),
        CONTENT_TYPE_WORKFLOWS: lambda relative: relative.name == WORKFLOWS_MANIFEST_FILE,
    }
    selector = selectors.get(content_type)
    if selector is None:
        logger.warning("Unknown content type '%s' for discovery", content_type)
        return []

    files = _discover(root, selected=selector)
    if content_type == CONTENT_TYPE_SKILL:
        grouped: dict[Path, dict[str, SecureFile]] = {}
        for file in files:
            grouped.setdefault(file.relative_path.parent, {})[file.relative_path.name] = file
        files = [
            next(variants[name] for name in SKILL_MANIFEST_VARIANTS if name in variants)
            for _directory, variants in sorted(grouped.items(), key=lambda item: item[0].as_posix())
        ]

    budget = _ExtractionBudget()
    entries: list[ContentEntry] = []
    with SecureRoot(root) as secure_root:
        for file in files:
            if content_type == CONTENT_TYPE_SKILL:
                fields = ("name", "description")
            else:
                fields = ("title", "description")
            entry = _extract_secure_file(
                secure_root,
                file,
                name_field=fields[0],
                description_field=fields[1],
                content_type=content_type,
                budget=budget,
                display_root=root,
            )
            if entry is not None:
                entries.append(entry)
    logger.debug("Discovered %d %s entries in %s", len(entries), content_type, safe_path_label(root))
    return entries
