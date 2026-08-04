# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, no-follow content collection for Tier 2 deduplication."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from skillevaluator.constants import (
    CONTENT_DEDUP_EXCLUDED_DIRS,
    CONTENT_DEDUP_EXCLUDED_FILES,
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_FILES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    CONTENT_DEDUP_SCANNABLE_EXTENSIONS,
)
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, discover_secure_files
from skillevaluator.utils.structured_data import (
    StructuredDataLimitError,
    StructuredDataSyntaxError,
    load_bounded_yaml,
)
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

logger = logging.getLogger(__name__)


class SkillCollectionError(ValueError):
    """Actionable fail-closed error for unsafe or unbounded skill content."""

    def __init__(
        self,
        check_name: str,
        message: str,
        *,
        rel_path: str,
        suggestion: str,
        metadata: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.check_name = check_name
        self.rel_path = rel_path
        self.suggestion = suggestion
        self.metadata = metadata or {}


@dataclass
class CollectedFile:
    """A bounded UTF-8 text file collected from a skill directory."""

    path: Path
    rel_path: str
    extension: str
    content: str
    line_count: int
    line_offset: int = 0


def _is_excluded(rel_parts: tuple[str, ...], excluded_dirs: frozenset[str]) -> bool:
    """Return True if any path component is in the excluded set."""
    return any(part in excluded_dirs for part in rel_parts)


def _strip_valid_frontmatter(raw_text: str) -> tuple[str, int]:
    """Strip valid mapping frontmatter and return its original line offset."""
    match = FRONTMATTER_PATTERN.match(raw_text)
    if not match:
        return raw_text, 0
    frontmatter_yaml, markdown_content = match.groups()
    try:
        data = load_bounded_yaml(frontmatter_yaml)
    except StructuredDataSyntaxError:
        return raw_text, 0
    if not isinstance(data, dict) or not data:
        return raw_text, 0
    return markdown_content, raw_text[: match.start(2)].count("\n")


def _collection_error(exc: SecurePathError) -> SkillCollectionError:
    message = str(exc)
    if exc.code == "path_count_limit":
        limit = exc.metadata.get("limit", CONTENT_DEDUP_MAX_DISCOVERED_PATHS)
        message = f"Skill tree contains more than {limit} paths."
    return SkillCollectionError(
        exc.code,
        message,
        rel_path=exc.relative_path,
        suggestion=(
            "Replace links, hardlinks, and special files with regular UTF-8 files stored inside the skill root; "
            "reduce authored content if a traversal or byte budget was exceeded."
        ),
        metadata=exc.metadata,
    )


def collect_files(
    skill_root: Path,
    excluded_dirs: Iterable[str] | None = None,
    excluded_files: Iterable[str] | None = None,
) -> list[CollectedFile]:
    """Collect all selected Tier 2 text through a verified root descriptor.

    Generated directories are pruned before the path budget. Other authored
    paths count even when irrelevant. File links that are irrelevant to the
    selected extension set are ignored without resolution; selected redirects,
    hardlinks, special files, or unbounded inputs fail closed.
    """
    excluded = CONTENT_DEDUP_EXCLUDED_DIRS if excluded_dirs is None else frozenset(excluded_dirs)
    excluded_basenames = (
        CONTENT_DEDUP_EXCLUDED_FILES if excluded_files is None else frozenset(name.lower() for name in excluded_files)
    )

    def selected(relative: Path) -> bool:
        return (
            relative.suffix.lower() in CONTENT_DEDUP_SCANNABLE_EXTENSIONS
            and relative.name.lower() not in excluded_basenames
            and not _is_excluded(relative.parts, excluded)
        )

    try:
        secure_files = discover_secure_files(
            skill_root,
            selected=selected,
            excluded_dirs=excluded,
            max_paths=CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
            allow_context_alias=True,
        )
        if len(secure_files) > CONTENT_DEDUP_MAX_FILES:
            raise SecurePathError(
                "file_count_limit",
                f"Skill contains more than {CONTENT_DEDUP_MAX_FILES} selected files.",
                metadata={"actual": len(secure_files), "limit": CONTENT_DEDUP_MAX_FILES},
            )

        declared_total = 0
        for secure_file in secure_files:
            size = secure_file.metadata.st_size
            if size > CONTENT_DEDUP_MAX_FILE_BYTES:
                raise SecurePathError(
                    "file_size_limit",
                    f"Selected file exceeds the Tier 2 per-file byte limit: {secure_file.rel_path}",
                    relative_path=secure_file.rel_path,
                    metadata={"actual_bytes": size, "limit_bytes": CONTENT_DEDUP_MAX_FILE_BYTES},
                )
            declared_total += size
            if declared_total > CONTENT_DEDUP_MAX_TOTAL_BYTES:
                raise SecurePathError(
                    "total_size_limit",
                    "Skill content exceeds the Tier 2 total byte limit.",
                    relative_path=secure_file.rel_path,
                    metadata={"actual_bytes": declared_total, "limit_bytes": CONTENT_DEDUP_MAX_TOTAL_BYTES},
                )

        collected: list[CollectedFile] = []
        actual_total = 0
        with SecureRoot(skill_root) as secure_root:
            for secure_file in secure_files:
                raw_text = secure_root.read_file_text(secure_file, CONTENT_DEDUP_MAX_FILE_BYTES)
                actual_total += len(raw_text.encode("utf-8"))
                if actual_total > CONTENT_DEDUP_MAX_TOTAL_BYTES:
                    raise SecurePathError(
                        "total_size_limit",
                        "Skill content exceeds the Tier 2 total byte limit.",
                        relative_path=secure_file.rel_path,
                        metadata={"actual_bytes": actual_total, "limit_bytes": CONTENT_DEDUP_MAX_TOTAL_BYTES},
                    )
                extension = secure_file.relative_path.suffix.lower()
                line_offset = 0
                try:
                    if extension in {".md", ".mdc"}:
                        content, line_offset = _strip_valid_frontmatter(raw_text)
                    else:
                        content = raw_text
                except StructuredDataLimitError as exc:
                    raise SkillCollectionError(
                        "manifest_complexity_limit",
                        f"Frontmatter structured-data complexity limit exceeded: {secure_file.rel_path}",
                        rel_path=secure_file.rel_path,
                        suggestion="Reduce frontmatter nesting, collection sizes, or YAML aliases.",
                    ) from exc
                collected.append(
                    CollectedFile(
                        path=skill_root / secure_file.relative_path,
                        rel_path=secure_file.rel_path,
                        extension=extension,
                        content=content,
                        line_count=len(raw_text.splitlines()),
                        line_offset=line_offset,
                    )
                )
        return collected
    except SecurePathError as exc:
        raise _collection_error(exc) from exc
