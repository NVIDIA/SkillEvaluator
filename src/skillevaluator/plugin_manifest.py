# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-bounded discovery for supported plugin manifest forms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillevaluator.constants import (
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_MANIFEST_FILES,
    PLUGIN_MANIFEST_TYPE,
)


class PluginManifestPathError(ValueError):
    """Raised when a declared plugin manifest crosses its plugin root."""


@dataclass(frozen=True)
class PluginManifestLocation:
    """A plugin manifest and the root that is allowed to contain it."""

    path: Path
    declared_path: Path
    root: Path
    manifest_type: str

    @property
    def manifest_filename(self) -> str:
        if self.manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE:
            return f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}"
        return self.declared_path.name


def locate_plugin_manifest(path: Path) -> PluginManifestLocation | None:
    """Locate a plugin manifest without permitting a symlink escape."""
    target = path.expanduser()
    declared_path: Path | None = None
    manifest_type: str | None = None

    # Check directories first because ``Path.is_dir()`` follows symlinks. A
    # symlink to a standalone plugin directory is supported and must be searched
    # relative to that directory.
    if target.is_dir():
        root = target
        for manifest_name in PLUGIN_MANIFEST_FILES:
            candidate = root / manifest_name
            if candidate.exists() or candidate.is_symlink():
                declared_path = candidate
                manifest_type = PLUGIN_MANIFEST_TYPE
                break
        if declared_path is None:
            candidate = root / PLUGIN_CONTAINED_MANIFEST_DIR / PLUGIN_CONTAINED_MANIFEST_FILE
            if candidate.exists() or candidate.is_symlink():
                declared_path = candidate
                manifest_type = PLUGIN_CONTAINED_MANIFEST_TYPE
        if declared_path is None:
            return None
    elif target.is_file() or target.is_symlink():
        if target.name in PLUGIN_MANIFEST_FILES:
            root = target.parent
            declared_path = target
            manifest_type = PLUGIN_MANIFEST_TYPE
        elif target.name == PLUGIN_CONTAINED_MANIFEST_FILE and target.parent.name == PLUGIN_CONTAINED_MANIFEST_DIR:
            root = target.parent.parent
            declared_path = target
            manifest_type = PLUGIN_CONTAINED_MANIFEST_TYPE
        else:
            return None
    else:
        return None

    try:
        resolved_root = root.resolve(strict=True)
        resolved_manifest = declared_path.resolve(strict=True)
    except OSError as exc:
        raise PluginManifestPathError(f"Plugin manifest could not be resolved safely: {declared_path}") from exc

    if not resolved_manifest.is_file() or not resolved_manifest.is_relative_to(resolved_root):
        raise PluginManifestPathError(
            f"Plugin manifest resolves outside the plugin root; refusing to read it: {declared_path}"
        )

    return PluginManifestLocation(
        path=resolved_manifest,
        declared_path=declared_path,
        root=root,
        manifest_type=manifest_type,
    )
