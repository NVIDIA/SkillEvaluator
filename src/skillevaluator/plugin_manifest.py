# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, race-resistant discovery for supported plugin manifests."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_MANIFEST_FILES,
    PLUGIN_MANIFEST_TYPE,
)
from skillevaluator.utils.secure_fs import (
    SecureFile,
    SecurePathError,
    SecureRoot,
    discover_secure_files,
    stat_is_link_or_reparse,
)


class PluginManifestPathError(SecurePathError):
    """Raised when a plugin root/manifest cannot be trusted or read safely."""

    def __init__(self, message: str, *, relative_path: str = ".") -> None:
        super().__init__("unsafe_plugin_manifest", message, relative_path=relative_path)


@dataclass(frozen=True)
class PluginManifestLocation:
    """A manifest identity retained from no-follow discovery through reads."""

    declared_path: Path
    root: Path
    manifest_type: str
    secure_file: SecureFile

    @property
    def path(self) -> Path:
        """Return the declared lexical path for diagnostics and compatibility."""
        return self.declared_path

    @property
    def manifest_filename(self) -> str:
        if self.manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE:
            return f"{PLUGIN_CONTAINED_MANIFEST_DIR}/{PLUGIN_CONTAINED_MANIFEST_FILE}"
        return self.declared_path.name

    def read_text(self, *, encoding: str = "utf-8", max_bytes: int = CONTENT_DEDUP_MAX_FILE_BYTES) -> str:
        """Read the discovered inode through the anchored plugin root descriptor."""
        try:
            # ``SecureFile.root`` is the absolute lexical root captured during
            # discovery. ``self.root`` remains the caller's spelling for
            # diagnostics, and may be relative to a cwd that later changes.
            with SecureRoot(self.secure_file.root) as secure_root:
                raw, _metadata = secure_root.read_bytes(
                    self.secure_file.relative_path,
                    max_bytes,
                    expected=self.secure_file.metadata,
                )
            return raw.decode(encoding)
        except (SecurePathError, UnicodeError, LookupError) as exc:
            raise PluginManifestPathError(
                f"Plugin manifest changed, is unsafe, or cannot be decoded: {self.declared_path}: {exc}",
                relative_path=self.secure_file.rel_path,
            ) from exc


def _wrap_security_error(exc: SecurePathError) -> PluginManifestPathError:
    return PluginManifestPathError(str(exc), relative_path=exc.relative_path)


def locate_plugin_manifest(path: Path) -> PluginManifestLocation | None:
    """Locate one regular single-link manifest beneath a regular plugin root.

    All supported manifest variants are selected during discovery, so a linked,
    hard-linked, special, or reparse manifest fails even when another regular
    variant would otherwise win precedence. The returned object carries the
    discovered inode metadata and must perform the eventual bounded read.
    """
    target = path.expanduser()
    direct_relative: Path | None = None
    try:
        target_metadata = target.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PluginManifestPathError(f"Cannot inspect plugin path safely: {target}: {exc}") from exc

    # A regular directory remains a plugin root even when its basename happens
    # to equal a supported manifest filename. Non-directories with a manifest
    # lexical shape are treated as direct manifests so links/specials reach the
    # secure selected-file checks and fail explicitly.
    if not stat_is_link_or_reparse(target_metadata) and stat.S_ISDIR(target_metadata.st_mode):
        root = target
    elif target.name in PLUGIN_MANIFEST_FILES:
        root = target.parent
        direct_relative = Path(target.name)
    elif target.name == PLUGIN_CONTAINED_MANIFEST_FILE and target.parent.name == PLUGIN_CONTAINED_MANIFEST_DIR:
        root = target.parent.parent
        direct_relative = Path(PLUGIN_CONTAINED_MANIFEST_DIR) / PLUGIN_CONTAINED_MANIFEST_FILE
    else:
        if stat_is_link_or_reparse(target_metadata):
            raise PluginManifestPathError(f"Plugin root is a symlink, junction, or reparse point: {target}")
        if not stat.S_ISREG(target_metadata.st_mode):
            raise PluginManifestPathError(f"Plugin root is not a regular directory: {target}")
        return None

    manifest_paths = [Path(name) for name in PLUGIN_MANIFEST_FILES]
    contained_relative = Path(PLUGIN_CONTAINED_MANIFEST_DIR) / PLUGIN_CONTAINED_MANIFEST_FILE
    manifest_paths.append(contained_relative)
    selected_paths = frozenset(manifest_paths)
    try:
        files = discover_secure_files(
            root,
            selected=lambda relative: relative in selected_paths,
            max_paths=CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
            max_depth=2,
        )
    except SecurePathError as exc:
        raise _wrap_security_error(exc) from exc

    by_relative = {file.relative_path: file for file in files}
    if direct_relative is not None:
        selected_file = by_relative.get(direct_relative)
        if selected_file is None:
            raise PluginManifestPathError(
                f"Declared plugin manifest is missing or unsafe: {target}",
                relative_path=direct_relative.as_posix(),
            )
    else:
        selected_file = next((by_relative[relative] for relative in manifest_paths if relative in by_relative), None)
        if selected_file is None:
            return None

    manifest_type = (
        PLUGIN_CONTAINED_MANIFEST_TYPE if selected_file.relative_path == contained_relative else PLUGIN_MANIFEST_TYPE
    )
    declared_path = root / selected_file.relative_path
    return PluginManifestLocation(
        declared_path=declared_path,
        root=root,
        manifest_type=manifest_type,
        secure_file=selected_file,
    )
