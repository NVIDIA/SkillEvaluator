# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for secure lexical filesystem traversal."""

from __future__ import annotations

import os
import stat
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


__all__ = ["canonicalize_trusted_root_alias"]
