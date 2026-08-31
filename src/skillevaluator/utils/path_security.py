# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for secure lexical filesystem traversal."""

from __future__ import annotations

import os
import stat
from collections.abc import Collection
from pathlib import Path


def canonicalize_trusted_root_alias(path: Path) -> Path:
    """Expand a root-owned POSIX alias such as macOS ``/var`` or ``/tmp``.

    Only the first component is eligible, and only when both the filesystem
    root and alias are root-owned while the root is not group/world writable.
    Later components remain lexical so secure callers can reject their links.
    """
    if os.name != "posix" or len(path.parts) < 2:
        return path
    root = Path(path.anchor)
    alias = root / path.parts[1]
    try:
        root_metadata = root.lstat()
        alias_metadata = alias.lstat()
    except OSError:
        return path
    if (
        not stat.S_ISLNK(alias_metadata.st_mode)
        or root_metadata.st_uid != 0
        or alias_metadata.st_uid != 0
        or stat.S_IMODE(root_metadata.st_mode) & 0o022
    ):
        return path
    try:
        target = alias.readlink()
    except OSError:
        return path
    if not target.is_absolute():
        target = root / target
    normalized = Path(os.path.abspath(os.fspath(target)))  # noqa: PTH100 - lexical normalization is intentional
    return normalized.joinpath(*path.parts[2:])


def matches_filesystem_name(path: Path, canonical_names: Collection[str]) -> bool:
    """Match an entry name according to the host filesystem's case semantics.

    Exact spellings retain the caller's historical behavior. Differently
    cased spellings match only when the directory entry is the same physical
    node as the canonical spelling, preserving authored aliases on
    case-sensitive filesystems.
    """
    if path.name in canonical_names:
        return True
    possible_aliases = [name for name in canonical_names if name.casefold() == path.name.casefold()]
    if not possible_aliases:
        return False
    try:
        observed = path.lstat()
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & reparse_flag:
        return False
    for name in possible_aliases:
        try:
            canonical = path.with_name(name).lstat()
        except OSError:
            continue
        if (
            not stat.S_ISLNK(canonical.st_mode)
            and not (getattr(canonical, "st_file_attributes", 0) & reparse_flag)
            and os.path.samestat(observed, canonical)
        ):
            return True
    return False


__all__ = ["canonicalize_trusted_root_alias", "matches_filesystem_name"]
