# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-file content collector for deduplication.

Walks a skill directory, filters by extension, reads text content,
and strips YAML frontmatter from markdown files.
"""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import yaml

from skillevaluator.constants import (
    CONTENT_DEDUP_EXCLUDED_DIRS,
    CONTENT_DEDUP_EXCLUDED_FILES,
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_FILES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    CONTENT_DEDUP_SCANNABLE_EXTENSIONS,
)
from skillevaluator.utils.tier2_paths import is_contained_compatibility_alias
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

logger = logging.getLogger(__name__)

_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


class SkillCollectionError(ValueError):
    """A fail-closed error for unsafe or unbounded skill content."""

    def __init__(
        self,
        check_name: str,
        message: str,
        *,
        rel_path: str,
        suggestion: str,
        metadata: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.check_name = check_name
        self.rel_path = rel_path
        self.suggestion = suggestion
        self.metadata = metadata or {}


class _SecureReadError(ValueError):
    """An internal fail-closed error from descriptor-anchored reads."""

    def __init__(
        self,
        check_name: str,
        message: str,
        *,
        actual_bytes: int | None = None,
        limit_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.check_name = check_name
        self.actual_bytes = actual_bytes
        self.limit_bytes = limit_bytes


def _supports_descriptor_anchored_reads() -> bool:
    """Return whether this runtime can open every path component relative to a directory fd."""
    return bool(
        os.name == "posix" and hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW") and _OPEN_SUPPORTS_DIR_FD
    )


class _SecureRoot:
    """Read files relative to an anchored root without following path redirects."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._root_fd: int | None = None
        self._resolved_root: Path | None = None

    def __enter__(self) -> _SecureRoot:
        if os.name == "posix":
            if not _supports_descriptor_anchored_reads():
                raise _SecureReadError(
                    "secure_open_unavailable",
                    "This platform cannot guarantee secure descriptor-anchored Tier 2 reads.",
                )
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            try:
                root_fd = os.open(self.root, flags)
            except OSError as exc:
                raise _SecureReadError(
                    "unsafe_path",
                    f"Cannot securely open the Tier 2 root directory: {exc}",
                ) from exc
            try:
                root_info = os.fstat(root_fd)
            except OSError as exc:
                os.close(root_fd)
                raise _SecureReadError(
                    "unsafe_path",
                    "Cannot verify the opened Tier 2 root directory descriptor.",
                ) from exc
            if not stat.S_ISDIR(root_info.st_mode):
                os.close(root_fd)
                raise _SecureReadError("unsafe_path", "Tier 2 root is not a regular directory.")
            self._root_fd = root_fd
            return self

        if os.name == "nt":
            if _is_link_or_reparse(self.root):
                raise _SecureReadError("unsafe_path", "Tier 2 root is a symlink or reparse point.")
            try:
                resolved_root = self.root.resolve(strict=True)
            except OSError as exc:
                raise _SecureReadError("unsafe_path", f"Cannot resolve the Tier 2 root: {exc}") from exc
            if not resolved_root.is_dir():
                raise _SecureReadError("unsafe_path", "Tier 2 root is not a regular directory.")
            self._resolved_root = resolved_root
            return self

        raise _SecureReadError(
            "secure_open_unavailable",
            "This platform cannot guarantee secure descriptor-anchored Tier 2 reads.",
        )

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None

    def read_bounded(self, relative_path: Path, max_bytes: int) -> tuple[bytes, os.stat_result]:
        """Open one contained regular file and perform exactly one bounded read."""
        parts = relative_path.parts
        if relative_path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
            raise _SecureReadError("unsafe_path", f"Refusing unsafe relative path: {relative_path.as_posix()}")

        if os.name == "posix":
            fd = self._open_posix(parts)
        elif os.name == "nt":
            fd = self._open_windows(parts)
        else:
            raise _SecureReadError(
                "secure_open_unavailable",
                "This platform cannot guarantee secure descriptor-anchored Tier 2 reads.",
            )

        try:
            opened_info = os.fstat(fd)
            if not stat.S_ISREG(opened_info.st_mode):
                raise _SecureReadError("unsafe_path", f"Refusing non-regular file: {relative_path.as_posix()}")
            if getattr(opened_info, "st_nlink", 1) != 1:
                raise _SecureReadError("unsafe_path", f"Refusing hard-linked file: {relative_path.as_posix()}")
            if opened_info.st_size > max_bytes:
                raise _SecureReadError(
                    "file_size_limit",
                    f"File exceeds the Tier 2 per-file byte limit: {relative_path.as_posix()}",
                    actual_bytes=opened_info.st_size,
                    limit_bytes=max_bytes,
                )
            with os.fdopen(fd, "rb", closefd=True) as stream:
                fd = -1
                raw_bytes = stream.read(max_bytes + 1)
        finally:
            if fd >= 0:
                os.close(fd)

        if len(raw_bytes) > max_bytes:
            raise _SecureReadError(
                "file_size_limit",
                f"File exceeds the Tier 2 per-file byte limit: {relative_path.as_posix()}",
                actual_bytes=len(raw_bytes),
                limit_bytes=max_bytes,
            )
        return raw_bytes, opened_info

    def _open_posix(self, parts: tuple[str, ...]) -> int:
        if self._root_fd is None:
            raise _SecureReadError("secure_open_unavailable", "Tier 2 root descriptor is unavailable.")

        directory_fd = os.dup(self._root_fd)
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        file_flags = (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            for part in parts[:-1]:
                try:
                    next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise _SecureReadError(
                        "unsafe_path",
                        f"Cannot securely traverse Tier 2 path component {part!r}: {exc}",
                    ) from exc
                try:
                    next_info = os.fstat(next_fd)
                except OSError as exc:
                    os.close(next_fd)
                    raise _SecureReadError(
                        "unsafe_path",
                        f"Cannot verify opened Tier 2 path component: {part}",
                    ) from exc
                if not stat.S_ISDIR(next_info.st_mode):
                    os.close(next_fd)
                    raise _SecureReadError("unsafe_path", f"Tier 2 path component is not a directory: {part}")
                os.close(directory_fd)
                directory_fd = next_fd

            try:
                return os.open(parts[-1], file_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise _SecureReadError(
                    "unsafe_path",
                    f"Cannot securely open Tier 2 file {parts[-1]!r}: {exc}",
                ) from exc
        finally:
            os.close(directory_fd)

    def _open_windows(self, parts: tuple[str, ...]) -> int:
        if self._resolved_root is None:
            raise _SecureReadError("secure_open_unavailable", "Tier 2 root handle verification is unavailable.")

        candidate = self.root.joinpath(*parts)
        current = self.root
        for part in parts:
            current /= part
            if _is_link_or_reparse(current):
                raise _SecureReadError("unsafe_path", f"Tier 2 path contains a reparse point: {current.name}")

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
        try:
            fd = os.open(candidate, flags)
        except OSError as exc:
            raise _SecureReadError("unsafe_path", f"Cannot securely open Tier 2 file: {exc}") from exc

        try:
            final_path = _windows_final_path(fd).resolve(strict=True)
            final_path.relative_to(self._resolved_root)
            if _is_link_or_reparse(candidate):
                raise _SecureReadError("unsafe_path", f"Tier 2 path contains a reparse point: {candidate.name}")
        except (OSError, ValueError) as exc:
            os.close(fd)
            if isinstance(exc, _SecureReadError):
                raise
            raise _SecureReadError(
                "unsafe_path",
                "Opened Tier 2 file is not contained by the verified root handle.",
            ) from exc
        return fd


def _windows_final_path(fd: int) -> Path:
    """Return the kernel-resolved path for an open Windows file descriptor."""
    if os.name != "nt":
        raise OSError("Windows handle verification is unavailable on this platform")

    import ctypes
    import msvcrt
    from ctypes import wintypes

    get_final_path = ctypes.windll.kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_final_path(msvcrt.get_osfhandle(fd), buffer, len(buffer), 0)
    if length == 0 or length >= len(buffer):
        raise OSError(ctypes.get_last_error(), "Cannot resolve opened Windows file handle")
    path = buffer.value
    if path.startswith("\\\\?\\UNC\\"):
        path = "\\\\" + path[8:]
    elif path.startswith("\\\\?\\"):
        path = path[4:]
    return Path(path)


@dataclass
class CollectedFile:
    """A text file collected from a skill directory."""

    path: Path
    rel_path: str  # relative to skill root (for display)
    extension: str  # lowercase, e.g. ".md"
    content: str  # text content (frontmatter stripped for .md/.mdc)
    line_count: int  # total lines in original file
    line_offset: int = 0  # original lines removed before ``content``


@dataclass(frozen=True)
class _CandidateFile:
    """A validated regular file awaiting a bounded read."""

    source_path: Path
    resolved_path: Path
    rel_path: str
    extension: str


def _is_excluded(rel_parts: tuple[str, ...], excluded_dirs: frozenset[str]) -> bool:
    """Return True if any path component is in the excluded set.

    Matches at any depth so that ``evals/results/.../SKILL.md`` and
    ``.versions/1.0.0/SKILL.md`` are both filtered, including nested
    re-occurrences (e.g. ``references/evals/foo.md``).
    """
    return any(part in excluded_dirs for part in rel_parts)


def _is_link_or_reparse(path: Path, rel_path: str | None = None) -> bool:
    """Return whether *path* is a symbolic link or Windows reparse point."""
    try:
        info = path.lstat()
    except OSError as e:
        raise SkillCollectionError(
            "path_access_error",
            f"Cannot inspect skill path: {e}",
            rel_path=rel_path or path.name,
            suggestion="Make the path readable and rerun Tier 2.",
        ) from e

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(reparse_flag and file_attributes & reparse_flag)


def _strip_valid_frontmatter(raw_text: str) -> tuple[str, int]:
    """Strip valid mapping frontmatter and return its original line offset."""
    match = FRONTMATTER_PATTERN.match(raw_text)
    if not match:
        return raw_text, 0

    frontmatter_yaml, markdown_content = match.groups()
    try:
        data = yaml.safe_load(frontmatter_yaml)
    except yaml.YAMLError:
        return raw_text, 0
    if not data or not isinstance(data, dict):
        return raw_text, 0
    return markdown_content, raw_text[: match.start(2)].count("\n")


def _read_bounded(secure_root: _SecureRoot, candidate: _CandidateFile) -> bytes:
    """Read a validated regular file through the anchored skill root."""
    try:
        raw_bytes, _opened_info = secure_root.read_bounded(
            Path(candidate.rel_path),
            CONTENT_DEDUP_MAX_FILE_BYTES,
        )
    except _SecureReadError as e:
        metadata = {}
        if e.actual_bytes is not None and e.limit_bytes is not None:
            metadata = {"actual_bytes": e.actual_bytes, "limit_bytes": e.limit_bytes}
        raise SkillCollectionError(
            e.check_name,
            str(e),
            rel_path=candidate.rel_path,
            suggestion="Replace links with regular UTF-8 files and ensure the file is readable.",
            metadata=metadata,
        ) from e
    return raw_bytes


def collect_files(
    skill_root: Path,
    excluded_dirs: Iterable[str] | None = None,
    excluded_files: Iterable[str] | None = None,
) -> list[CollectedFile]:
    """Collect bounded text files through a descriptor-anchored root."""
    try:
        with _SecureRoot(skill_root) as secure_root:
            return _collect_files_anchored(
                skill_root,
                secure_root,
                excluded_dirs=excluded_dirs,
                excluded_files=excluded_files,
            )
    except _SecureReadError as exc:
        raise SkillCollectionError(
            exc.check_name,
            str(exc),
            rel_path=".",
            suggestion="Use a local regular directory on a platform with secure filesystem handle support.",
        ) from exc


def _collect_files_anchored(
    skill_root: Path,
    secure_root: _SecureRoot,
    excluded_dirs: Iterable[str] | None = None,
    excluded_files: Iterable[str] | None = None,
) -> list[CollectedFile]:
    """Walk a skill directory and collect all analyzable text files.

    Args:
        skill_root: Path to the skill directory.
        excluded_dirs: Directory names to skip at any depth. Defaults to
            :data:`CONTENT_DEDUP_EXCLUDED_DIRS` so that evaluation artifacts
            (``evals/``) and version snapshots (``.versions/``) never feed
            the deduplication pipeline. Pass an empty iterable to disable.
        excluded_files: Basenames to skip. Defaults to
            :data:`CONTENT_DEDUP_EXCLUDED_FILES` so generated metadata such as
            ``skill-card.md`` and ``BENCHMARK.md`` do not compare against
            their source content.

    Returns:
        Sorted list of CollectedFile objects for all accepted files.
    """
    excluded = CONTENT_DEDUP_EXCLUDED_DIRS if excluded_dirs is None else frozenset(excluded_dirs)
    excluded_basenames = CONTENT_DEDUP_EXCLUDED_FILES if excluded_files is None else frozenset(excluded_files)

    try:
        resolved_root = skill_root.resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise SkillCollectionError(
            "invalid_skill_root",
            f"Cannot resolve skill root {skill_root}: {e}",
            rel_path=str(skill_root),
            suggestion="Provide an existing, readable skill directory.",
        ) from e
    if not resolved_root.is_dir():
        raise SkillCollectionError(
            "invalid_skill_root",
            f"Skill root is not a directory: {skill_root}",
            rel_path=str(skill_root),
            suggestion="Provide a skill directory rather than a file.",
        )

    candidates: list[_CandidateFile] = []
    declared_total_bytes = 0

    def _raise_traversal_error(error: OSError) -> None:
        raise error

    def _iter_paths(root: Path) -> Iterator[Path]:
        # Prune the excluded dirs DURING traversal, not just when filtering
        # the discovered list: generated artifacts (evals/results snapshots)
        # must not consume the path budget, or a well-used skill fails the
        # path-count limit on content it was never going to scan.
        for dirpath, dirnames, filenames in os.walk(root, onerror=_raise_traversal_error):
            base = Path(dirpath)
            kept: list[str] = []
            for name in dirnames:
                directory = base / name
                rel_path = directory.relative_to(root).as_posix()
                if name in excluded:
                    logger.debug("Skipping excluded path: %s", rel_path)
                    continue
                # ``os.walk(..., followlinks=False)`` does not follow normal
                # symlinks, but Windows directory junctions are separate
                # reparse points and may still be traversed. Reject every
                # retained redirect before allowing the walk to recurse.
                if _is_link_or_reparse(directory, rel_path):
                    raise SkillCollectionError(
                        "unsafe_path",
                        f"Refusing symbolic link or reparse point: {rel_path}",
                        rel_path=rel_path,
                        suggestion="Replace the link with a regular directory stored inside the skill directory.",
                    )
                kept.append(name)
            dirnames[:] = kept
            for name in kept:
                yield base / name
            for name in filenames:
                yield base / name

    try:
        discovered_paths = list(islice(_iter_paths(skill_root), CONTENT_DEDUP_MAX_DISCOVERED_PATHS + 1))
    except OSError as e:
        raise SkillCollectionError(
            "path_access_error",
            f"Cannot safely traverse the skill directory: {e}",
            rel_path=".",
            suggestion="Make the skill directory readable and rerun Tier 2.",
        ) from e
    if len(discovered_paths) > CONTENT_DEDUP_MAX_DISCOVERED_PATHS:
        raise SkillCollectionError(
            "path_count_limit",
            f"Skill tree contains more than {CONTENT_DEDUP_MAX_DISCOVERED_PATHS} paths.",
            rel_path=".",
            suggestion="Remove generated content or split the skill before running Tier 2.",
            metadata={
                "actual": len(discovered_paths),
                "limit": CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
            },
        )
    discovered_paths.sort()
    discovered_names_by_parent: dict[Path, set[str]] = {}
    for discovered_path in discovered_paths:
        discovered_names_by_parent.setdefault(discovered_path.parent, set()).add(discovered_path.name)

    for file_path in discovered_paths:
        try:
            relative = file_path.relative_to(skill_root)
        except ValueError as e:
            raise SkillCollectionError(
                "unsafe_path",
                f"Path is not beneath the skill root: {file_path}",
                rel_path=file_path.as_posix(),
                suggestion="Keep all Tier 2 inputs inside the skill directory.",
            ) from e

        rel_parts = relative.parts
        rel_path = relative.as_posix()
        if _is_excluded(rel_parts, excluded):
            logger.debug("Skipping excluded path: %s", rel_path)
            continue

        if _is_link_or_reparse(file_path, rel_path):
            if is_contained_compatibility_alias(
                file_path,
                sibling_names=discovered_names_by_parent.get(file_path.parent, frozenset()),
            ):
                logger.debug("Skipping compatibility alias in favor of regular target: %s", rel_path)
                continue
            raise SkillCollectionError(
                "unsafe_path",
                f"Refusing symbolic link or reparse point: {rel_path}",
                rel_path=rel_path,
                suggestion="Replace the link with a regular file stored inside the skill directory.",
            )

        try:
            resolved_path = file_path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise SkillCollectionError(
                "path_access_error",
                f"Cannot resolve skill path {rel_path}: {e}",
                rel_path=rel_path,
                suggestion="Remove broken paths or make them readable before rerunning Tier 2.",
            ) from e
        if not resolved_path.is_relative_to(resolved_root):
            raise SkillCollectionError(
                "unsafe_path",
                f"Path resolves outside the skill root: {rel_path}",
                rel_path=rel_path,
                suggestion="Replace the path with a regular file stored inside the skill directory.",
            )

        if not resolved_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in CONTENT_DEDUP_SCANNABLE_EXTENSIONS:
            continue

        if file_path.name.lower() in excluded_basenames:
            logger.debug("Skipping excluded file: %s", file_path)
            continue

        candidate_count = len(candidates) + 1
        if candidate_count > CONTENT_DEDUP_MAX_FILES:
            raise SkillCollectionError(
                "file_count_limit",
                f"Skill contains more than {CONTENT_DEDUP_MAX_FILES} scannable files.",
                rel_path=rel_path,
                suggestion="Reduce or split the skill content before running Tier 2.",
                metadata={"actual": candidate_count, "limit": CONTENT_DEDUP_MAX_FILES},
            )

        try:
            size_bytes = resolved_path.stat().st_size
        except OSError as e:
            raise SkillCollectionError(
                "path_access_error",
                f"Cannot inspect file size for {rel_path}: {e}",
                rel_path=rel_path,
                suggestion="Make the file readable and rerun Tier 2.",
            ) from e
        if size_bytes > CONTENT_DEDUP_MAX_FILE_BYTES:
            raise SkillCollectionError(
                "file_size_limit",
                f"File exceeds the Tier 2 per-file byte limit: {rel_path}",
                rel_path=rel_path,
                suggestion="Split or reduce this file before running Tier 2.",
                metadata={"actual_bytes": size_bytes, "limit_bytes": CONTENT_DEDUP_MAX_FILE_BYTES},
            )

        declared_total_bytes += size_bytes
        if declared_total_bytes > CONTENT_DEDUP_MAX_TOTAL_BYTES:
            raise SkillCollectionError(
                "total_size_limit",
                "Skill content exceeds the Tier 2 total byte limit.",
                rel_path=rel_path,
                suggestion="Reduce the total scannable content or split it into separate skills.",
                metadata={"actual_bytes": declared_total_bytes, "limit_bytes": CONTENT_DEDUP_MAX_TOTAL_BYTES},
            )

        candidates.append(_CandidateFile(file_path, resolved_path, rel_path, ext))

    collected: list[CollectedFile] = []
    actual_total_bytes = 0
    for candidate in candidates:
        raw_bytes = _read_bounded(secure_root, candidate)
        actual_total_bytes += len(raw_bytes)
        if actual_total_bytes > CONTENT_DEDUP_MAX_TOTAL_BYTES:
            raise SkillCollectionError(
                "total_size_limit",
                "Skill content exceeds the Tier 2 total byte limit.",
                rel_path=candidate.rel_path,
                suggestion="Reduce the total scannable content or split it into separate skills.",
                metadata={"actual_bytes": actual_total_bytes, "limit_bytes": CONTENT_DEDUP_MAX_TOTAL_BYTES},
            )
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise SkillCollectionError(
                "invalid_text_encoding",
                f"Tier 2 requires UTF-8 text: {candidate.rel_path}",
                rel_path=candidate.rel_path,
                suggestion="Convert this file to UTF-8 or remove it from the scannable skill content.",
            ) from e

        # Strip YAML frontmatter from markdown files
        content = raw_text
        line_offset = 0
        if candidate.extension in {".md", ".mdc"}:
            content, line_offset = _strip_valid_frontmatter(raw_text)

        collected.append(
            CollectedFile(
                path=candidate.source_path,
                rel_path=candidate.rel_path,
                extension=candidate.extension,
                content=content,
                line_count=len(raw_text.splitlines()),
                line_offset=line_offset,
            )
        )

    return collected
