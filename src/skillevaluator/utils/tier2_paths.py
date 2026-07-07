# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-safety and host-path redaction helpers for Tier 2 commands."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult


def safe_path_label(path: Path) -> str:
    """Return a useful path label without exposing its host directory."""
    return path.name or "."


def is_link_or_reparse(path: Path) -> bool:
    """Return whether *path* itself is a symlink, junction, or reparse point."""
    try:
        metadata = path.lstat()
    except OSError:
        return False

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    is_junction = getattr(path, "is_junction", None)
    return bool(
        stat.S_ISLNK(metadata.st_mode) or file_attributes & reparse_flag or (callable(is_junction) and is_junction())
    )


def paths_refer_to_same_location(left: Path, right: Path) -> bool:
    """Compare output paths after resolving aliases, symlinks, and hard links."""
    with suppress(OSError):
        if left.exists() and right.exists() and left.samefile(right):
            return True

    def normalized(path: Path) -> str:
        expanded = path.expanduser()
        try:
            resolved = expanded.resolve(strict=False)
        except (OSError, RuntimeError):
            resolved = expanded.absolute()
        return os.path.normcase(os.path.normpath(str(resolved)))

    return normalized(left) == normalized(right)


def sanitize_path_text(message: str, paths: list[Path | None] | tuple[Path | None, ...]) -> str:
    """Replace absolute forms of supplied paths with stable, relative labels."""
    replacements: dict[str, str] = {}
    for path in paths:
        if path is None:
            continue
        label = safe_path_label(path)
        candidates = {str(path)}
        with suppress(OSError, RuntimeError):
            candidates.add(str(path.expanduser().absolute()))
        with suppress(OSError, RuntimeError):
            candidates.add(str(path.expanduser().resolve(strict=False)))
        for candidate in candidates:
            if candidate and Path(candidate).is_absolute():
                replacements[candidate] = label

    for candidate in sorted(replacements, key=len, reverse=True):
        message = message.replace(candidate, replacements[candidate])
    return message


def _sanitize_value(value, paths: tuple[Path | None, ...]):
    if isinstance(value, str):
        return sanitize_path_text(value, paths)
    if isinstance(value, Path):
        return sanitize_path_text(str(value), paths)
    if isinstance(value, dict):
        return {_sanitize_value(key, paths): _sanitize_value(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item, paths) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item, paths) for item in value)
    return value


def sanitize_tier2_results(results: list[ValidationResult], *paths: Path | None) -> list[ValidationResult]:
    """Remove supplied host paths from every user-visible result field in place."""
    sanitized_paths = tuple(paths)
    for result in results:
        result.validator_name = sanitize_path_text(result.validator_name, sanitized_paths)
        result.validator_description = sanitize_path_text(result.validator_description, sanitized_paths)
        result.errors[:] = [sanitize_path_text(value, sanitized_paths) for value in result.errors]
        result.warnings[:] = [sanitize_path_text(value, sanitized_paths) for value in result.warnings]
        result.messages[:] = [sanitize_path_text(value, sanitized_paths) for value in result.messages]
        result.metadata = _sanitize_value(result.metadata, sanitized_paths)

        for detail in result.success_details:
            detail.message = sanitize_path_text(detail.message, sanitized_paths)
            detail.metadata = _sanitize_value(detail.metadata, sanitized_paths)

        for finding in result.findings:
            finding.message = sanitize_path_text(finding.message, sanitized_paths)
            finding.file_path = sanitize_path_text(finding.file_path, sanitized_paths)
            if finding.line_content is not None:
                finding.line_content = sanitize_path_text(finding.line_content, sanitized_paths)
            if finding.suggestion is not None:
                finding.suggestion = sanitize_path_text(finding.suggestion, sanitized_paths)
            finding.metadata = _sanitize_value(finding.metadata, sanitized_paths)
    return results


__all__ = [
    "is_link_or_reparse",
    "paths_refer_to_same_location",
    "safe_path_label",
    "sanitize_path_text",
    "sanitize_tier2_results",
]
