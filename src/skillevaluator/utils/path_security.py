# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for secure lexical filesystem traversal."""

from __future__ import annotations

import os
import stat
import subprocess
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
    except (OSError, ValueError, UnicodeError):
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & reparse_flag:
        return False
    for name in possible_aliases:
        try:
            canonical = path.with_name(name).lstat()
        except (OSError, ValueError, UnicodeError):
            continue
        if (
            not stat.S_ISLNK(canonical.st_mode)
            and not (getattr(canonical, "st_file_attributes", 0) & reparse_flag)
            and os.path.samestat(observed, canonical)
        ):
            return True
    return False


def find_git_repo_root(path: Path) -> Path | None:
    """Return the containing Git root, with a metadata-search fallback."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip()).resolve()
    except Exception:
        pass

    current = path.resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    return None


def resolve_repo_context_root(path: Path) -> Path:
    """Return the exact source root used by Tier 3 repo-context staging."""
    try:
        repo_root = find_git_repo_root(path)
        resolved = path.resolve()
        return repo_root or resolved.parent
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve repository context root for: {path}") from exc


__all__ = [
    "canonicalize_trusted_root_alias",
    "find_git_repo_root",
    "matches_filesystem_name",
    "resolve_repo_context_root",
]
