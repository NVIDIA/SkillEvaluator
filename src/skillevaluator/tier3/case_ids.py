# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation and path containment for Tier 3 case identifiers."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

MAX_CASE_ID_LENGTH = 128
_CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{index}" for index in range(1, 10)} | {f"LPT{index}" for index in range(1, 10)}
)


def validate_case_id(value: object) -> str:
    """Return the canonical Harbor-safe form of one authored eval ID.

    The public Agent Skills evaluation guide uses integer IDs, so integers are
    intentionally accepted and canonicalized to decimal strings. Booleans and
    other JSON scalar types are rejected to avoid ambiguous identities.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("case id must be a string or integer")

    case_id = str(value)
    if not case_id:
        raise ValueError("case id must not be empty")
    if len(case_id) > MAX_CASE_ID_LENGTH:
        raise ValueError(f"case id must be at most {MAX_CASE_ID_LENGTH} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in case_id):
        raise ValueError("case id must not contain control characters")
    if PurePosixPath(case_id).is_absolute() or PureWindowsPath(case_id).is_absolute():
        raise ValueError("case id must not be an absolute path")
    if "/" in case_id or "\\" in case_id:
        raise ValueError("case id must be one path component and must not contain '/' or '\\'")
    if ".." in case_id:
        raise ValueError("case id must not contain '..'")
    if not _CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(
            "case id must start with an ASCII letter or digit and contain only ASCII letters, digits, '.', '_', or '-'"
        )
    if case_id.endswith("."):
        raise ValueError("case id must not end with '.'")
    if case_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("case id must not use a reserved Windows device name")
    return case_id


def validate_case_ids(values: Iterable[object]) -> list[str]:
    """Validate and canonicalize a complete dataset's IDs before mutation."""
    validated: list[str] = []
    seen: dict[str, tuple[int, str]] = {}
    for index, value in enumerate(values):
        try:
            case_id = validate_case_id(value)
        except ValueError as exc:
            raise ValueError(f"invalid case id at entry {index}: {exc}") from exc

        collision_key = case_id.casefold()
        if collision_key in seen:
            previous_index, previous_id = seen[collision_key]
            raise ValueError(
                f"duplicate or cross-platform colliding case id at entry {index}: "
                f"{case_id!r} conflicts with entry {previous_index} ({previous_id!r})"
            )
        seen[collision_key] = (index, case_id)
        validated.append(case_id)
    return validated


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _canonicalize_platform_root_alias(path: Path) -> Path:
    """Resolve only a trusted root-owned POSIX alias such as macOS /tmp."""

    if os.name != "posix" or len(path.parts) < 2:
        return path
    root = Path(path.anchor)
    alias = root / path.parts[1]
    try:
        root_metadata = root.stat()
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
    target = alias.readlink()
    if not target.is_absolute():
        target = root / target
    normalized = Path(os.path.abspath(os.fspath(target)))  # noqa: PTH100
    return normalized.joinpath(*path.parts[2:])


def _validate_directory_components(path: Path, *, case_id: str) -> None:
    """Reject every existing link/reparse component in a lexical directory path."""

    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    if ".." in lexical.parts:
        raise ValueError(f"refusing case id {case_id!r}: output directory contains a parent traversal")
    lexical = _canonicalize_platform_root_alias(lexical)

    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        if component in {"", "."}:
            continue
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError(f"refusing case id {case_id!r}: cannot inspect output directory {current}") from exc
        is_junction = getattr(current, "is_junction", None)
        if _is_link_or_reparse(metadata) or (callable(is_junction) and is_junction()):
            raise ValueError(
                f"refusing case id {case_id!r}: output directory contains a symlink, reparse point, or junction"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"refusing case id {case_id!r}: output directory component is not a directory")


def safe_child(base: Path, component: object) -> Path:
    """Return a validated strict child of *base*, rejecting symlink escapes."""
    case_id = validate_case_id(component)
    _validate_directory_components(base, case_id=case_id)
    candidate = base / case_id
    _validate_directory_components(candidate, case_id=case_id)

    try:
        resolved_base = base.resolve(strict=False)
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(resolved_base)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"refusing case id {case_id!r}: task path escapes the output directory") from exc

    if resolved_candidate == resolved_base:
        raise ValueError(f"refusing case id {case_id!r}: task path is not a strict child")
    return candidate
