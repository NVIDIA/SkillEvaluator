# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, no-follow discovery of native plugin skill roots, shared by all tiers."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    PLUGIN_COMPONENT_MAX_ITEMS,
    SKILL_MANIFEST_VARIANTS,
)
from skillevaluator.plugin_paths import PluginRootReader, normalize_declared_path
from skillevaluator.utils.helpers import preferred_skill_manifests
from skillevaluator.utils.secure_fs import (
    MAX_SECURE_DIRECTORY_DEPTH,
    SecureFile,
    SecurePathError,
    discover_secure_files,
    lstat_walk,
)


def discover_openclaw_skill_manifests(
    plugin_root: Path,
    data: dict[str, Any],
    *,
    max_roots: int = PLUGIN_COMPONENT_MAX_ITEMS,
    max_paths: int = CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    max_depth: int = MAX_SECURE_DIRECTORY_DEPTH,
) -> list[SecureFile]:
    """Discover only declared OpenClaw roots, stopping recursively at skill boundaries.

    Dot-prefixed child directories and ``node_modules`` are not descended into,
    as in OpenClaw's skill-root discovery. These are child-descent rules only:
    an explicitly declared root may have either name. No implicit ``skills/``
    root is added when ``skills`` is omitted or empty.

    Shallow secure walks reject redirects before descent and retain discovery
    identities. The path allowance is shared across all declared roots, even
    when they overlap. Nothing below a directory containing ``SKILL.md`` is
    enumerated. Results are unique by skill directory, sorted, and relative to
    the plugin root, ready for anchored reads through ``SecureRoot(plugin_root)``.
    This function does not read skill content or import/execute plugin code.

    Invalid declarations or limit arguments raise ``ValueError``; unsafe paths
    and exhausted traversal limits raise ``SecurePathError``. Missing declared
    roots are skipped, matching OpenClaw; Tier 1 reports their missing paths.
    """
    if type(max_roots) is not int or max_roots < 0:
        raise ValueError("max_roots must be a non-negative integer")
    if type(max_paths) is not int or max_paths < 1:
        raise ValueError("max_paths must be a positive integer")
    if type(max_depth) is not int or not 1 <= max_depth <= MAX_SECURE_DIRECTORY_DEPTH:
        raise ValueError(f"max_depth must be an integer from 1 to {MAX_SECURE_DIRECTORY_DEPTH}")
    roots = data.get("skills", [])
    if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
        raise ValueError("OpenClaw skills must be a list of declared directory paths")
    if len(roots) > max_roots:
        raise ValueError(f"OpenClaw skills exceeds the {max_roots}-item limit")
    reader = PluginRootReader(plugin_root)
    declared_roots: set[PurePosixPath] = set()
    for raw in roots:
        declared = normalize_declared_path(raw, ())
        if declared.problem is not None or declared.rel is None:
            raise SecurePathError("unsafe_path", f"Unsafe OpenClaw skills directory: {raw!r}")
        declared_roots.add(declared.rel)

    manifests: dict[str, SecureFile] = {}
    visited: set[PurePosixPath] = set()
    discovered_paths = 0
    entries: list[Path] = []

    def selected(path: Path) -> bool:
        nonlocal discovered_paths
        discovered_paths += 1
        entries.append(path)
        return path.name in SKILL_MANIFEST_VARIANTS

    pending = sorted(declared_roots, reverse=True)
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        visited.add(relative)
        if len(relative.parts) > max_depth:
            raise SecurePathError("directory_depth_limit", "OpenClaw skills directory exceeds the depth limit")
        kind = reader.kind(relative)
        if kind == "missing" and relative in declared_roots and lstat_walk(reader.root, relative).outcome == "missing":
            # kind() also maps non-directory ancestors to "missing". Only a
            # genuinely absent declaration is optional, never a wrong-type path.
            continue
        if kind != "dir":
            raise SecurePathError("unsafe_path", f"OpenClaw skills directory changed or is unsafe: {relative}")
        remaining = max_paths - discovered_paths
        if remaining < 1:
            raise SecurePathError(
                "path_count_limit",
                f"OpenClaw skills exceed the discovery path limit of {max_paths} entries.",
                relative_path=relative.as_posix(),
                metadata={"actual": discovered_paths + 1, "limit": max_paths},
            )
        entries.clear()
        paths_before_walk = discovered_paths
        try:
            files = discover_secure_files(
                plugin_root / relative.as_posix(),
                selected=selected,
                max_paths=remaining,
                max_depth=1,
            )
        except SecurePathError as exc:
            if exc.code != "path_count_limit":
                raise
            raise SecurePathError(
                "path_count_limit",
                f"OpenClaw skills exceed the discovery path limit of {max_paths} entries.",
                relative_path=(relative / exc.relative_path).as_posix(),
                metadata={"actual": paths_before_walk + exc.metadata.get("actual", remaining + 1), "limit": max_paths},
            ) from exc
        preferred = preferred_skill_manifests(files)
        if preferred:
            file = preferred[0]
            manifest_relative = Path(relative.as_posix()) / file.relative_path
            manifests.setdefault(
                manifest_relative.parent.as_posix(),
                SecureFile(reader.root, reader.root / manifest_relative, manifest_relative, file.metadata),
            )
            continue
        for entry in reversed(entries):
            # Apply exclusions to children only, never to an explicitly declared
            # root. The shallow secure walk has not descended into any child.
            if entry.name.startswith(".") or entry.name == "node_modules":
                continue
            child = relative / entry.as_posix()
            kind = reader.kind(child)
            if kind == "dir":
                pending.append(child)
            elif kind in {"link", "special", "missing"}:
                raise SecurePathError("unsafe_path", f"OpenClaw skills path changed or is unsafe: {child}")
    return [manifests[key] for key in sorted(manifests)]


__all__ = ["discover_openclaw_skill_manifests"]
