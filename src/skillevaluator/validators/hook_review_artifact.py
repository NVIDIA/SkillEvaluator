# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the bounded packaged tree a reviewer approves for hook coverage."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from skillevaluator.constants import PLUGIN_TREE_MAX_DISCOVERED_PATHS
from skillevaluator.utils.path_security import canonicalize_trusted_root_alias
from skillevaluator.utils.secure_fs import SecureFile, SecurePathError, SecureRoot, discover_secure_files

_ARTIFACT_DIGEST_FORMAT = "hook-review-artifact-v1"
_MAX_PATH_BYTES = 1024
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class HookReviewArtifact:
    """The packaged tree digest and its file SHA-256 values by relative path."""

    artifact_sha256: str
    hashes: Mapping[str, str]


def _entry_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _inventory(
    root: Path, device: int
) -> tuple[list[SecureFile], dict[Path, os.stat_result], dict[Path, tuple[int, ...]]]:
    directories: dict[Path, os.stat_result] = {}
    observed: dict[Path, tuple[int, ...]] = {}
    portable_paths: set[str] = set()

    def record(relative: Path, metadata: os.stat_result) -> None:
        if any(part.casefold() == ".git" for part in relative.parts):
            raise ValueError("Hook review requires a packaged artifact without Git metadata.")
        path = relative.as_posix()
        if (
            len(path.encode("utf-8")) > _MAX_PATH_BYTES
            or unicodedata.normalize("NFC", path) != path
            or any(character in path for character in "\\:")
            or any(unicodedata.category(character)[0] == "C" for character in path)
            or any(part.endswith((" ", ".")) for part in relative.parts)
        ):
            raise ValueError("A hook review artifact path does not have a canonical portable spelling.")
        if path.casefold() in portable_paths:
            raise ValueError(f"Ambiguous hook review artifact path: {path}")
        if metadata.st_dev != device:
            raise ValueError(f"Hook review artifact crosses a filesystem mount: {path}")
        if stat.S_IMODE(metadata.st_mode) & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
            raise ValueError(f"Hook review artifact contains special permissions: {path}")
        portable_paths.add(path.casefold())
        observed[relative] = _entry_fingerprint(metadata)

    def add_directory(relative: Path, metadata: os.stat_result) -> None:
        record(relative, metadata)
        directories[relative] = metadata

    files = discover_secure_files(
        root,
        selected=lambda _: True,
        max_paths=PLUGIN_TREE_MAX_DISCOVERED_PATHS,
        allow_context_alias=False,
        refuse_selected_dirs=False,
        on_directory=add_directory,
    )
    for file in files:
        record(file.relative_path, file.metadata)
    return files, directories, observed


def _absolute_root(path: Path) -> Path:
    return canonicalize_trusted_root_alias(Path(os.path.abspath(path)))  # noqa: PTH100 - keep input paths lexical


def _snapshot_destination(path: Path, source: os.stat_result) -> Path:
    destination = _absolute_root(path)
    with SecureRoot(destination):
        if any(os.path.samestat(ancestor.lstat(), source) for ancestor in (destination, *destination.parents)):
            raise ValueError("The hook review snapshot destination is inside the package.")
        if any(destination.iterdir()):
            raise ValueError("The hook review snapshot destination must be empty.")
    return destination


def capture_hook_review_artifact(root: Path, destination: Path | None = None) -> HookReviewArtifact:
    """Hash one stable, regular packaged tree and optionally copy its captured bytes.

    The artifact hash is SHA-256 of compact UTF-8 JSON containing the
    literal format string and one list of records sorted by POSIX relative
    path. Each record is [path, kind, permission bits, file SHA-256 or null].
    Directories, including empty ones, are records. The enclosing root name
    and mode are outside the recipe, so a private snapshot has the same hash.

    The caller owns the optional empty private destination and must discard
    it if capture fails. Unsafe paths, unsupported metadata, size limits,
    and changes during discovery or reads raise ValueError.
    """
    try:
        root = _absolute_root(root)
        root_metadata = root.lstat()
        snapshot = _snapshot_destination(destination, root_metadata) if destination is not None else None
        files, directories, observed = _inventory(root, root_metadata.st_dev)
        records: list[tuple[str, str, int, str | None]] = []
        for directory, metadata in directories.items():
            records.append((directory.as_posix(), "directory", stat.S_IMODE(metadata.st_mode), None))
            if snapshot is not None:
                (snapshot / directory).mkdir()

        hashes: dict[str, str] = {}
        remaining_bytes = _MAX_ARTIFACT_BYTES
        with SecureRoot(root, expected=root_metadata) as reader:
            for file in files:
                relative = file.relative_path
                data, metadata = reader.read_bytes(
                    relative, min(_MAX_FILE_BYTES, remaining_bytes), expected=file.metadata
                )
                remaining_bytes -= len(data)
                path = relative.as_posix()
                file_sha256 = hashlib.sha256(data).hexdigest()
                mode = stat.S_IMODE(file.metadata.st_mode)
                if stat.S_IMODE(metadata.st_mode) != mode:
                    raise ValueError(f"Hook review artifact file mode changed while reading: {path}")
                hashes[path] = file_sha256
                records.append((path, "file", mode, file_sha256))
                if snapshot is not None:
                    output = snapshot / relative
                    with output.open("xb") as stream:
                        stream.write(data)
                    output.chmod(mode)

        _, _, confirmed = _inventory(root, root_metadata.st_dev)
        if confirmed != observed or _entry_fingerprint(root.lstat()) != _entry_fingerprint(root_metadata):
            raise ValueError("The hook review artifact changed during capture.")
        if snapshot is not None:
            # Leave directories writable until their children are copied.
            for directory in reversed(directories):
                (snapshot / directory).chmod(stat.S_IMODE(directories[directory].st_mode))
    except (OSError, SecurePathError, UnicodeError) as exc:
        raise ValueError(f"Could not capture hook review artifact: {exc}") from exc

    serialized = json.dumps(
        [_ARTIFACT_DIGEST_FORMAT, sorted(records)], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return HookReviewArtifact(hashlib.sha256(serialized).hexdigest(), MappingProxyType(hashes))
