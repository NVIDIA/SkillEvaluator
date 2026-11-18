# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-only exclusions; publication source identity has its own contract."""

from __future__ import annotations

import stat
from pathlib import Path

from skillevaluator.utils.path_security import matches_filesystem_name

_CACHE_DIRECTORIES = frozenset({".pytest_cache", ".mypy_cache", ".ruff_cache"})
_OS_METADATA_FILES = frozenset({".DS_Store", "Thumbs.db"})
_CACHE_NAMES_CASEFOLDED = frozenset(name.casefold() for name in _CACHE_DIRECTORIES | _OS_METADATA_FILES)


def runtime_cache_entry_is_excluded(source: Path, path: Path) -> bool:
    """Exclude known incidental entries, leaving unsafe node kinds for secure traversal."""
    if path.name.casefold() not in _CACHE_NAMES_CASEFOLDED:
        return False
    try:
        metadata = path.lstat()
        root_metadata = source.lstat()
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if metadata.st_dev != root_metadata.st_dev or getattr(metadata, "st_file_attributes", 0) & reparse_flag:
        return False
    if stat.S_ISDIR(metadata.st_mode):
        return matches_filesystem_name(path, _CACHE_DIRECTORIES)
    if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
        return matches_filesystem_name(path, _OS_METADATA_FILES)
    return False
