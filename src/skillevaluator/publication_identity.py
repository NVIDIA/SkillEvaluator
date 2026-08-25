# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical target identity for cross-job publication evidence.

The source-tree recipe is deliberately versioned independently from SHA-256.
It binds normalized relative paths, node kinds, executable bits, and regular
file contents while omitting only SkillEvaluator's generated artifact roots
and files. POSIX traversal is descriptor-anchored and no-follow; other
platforms use checked pre/post path metadata. Observed unsafe or unstable
snapshots do not receive an identity. Like the staging subsystem, this is not
a coherent filesystem snapshot or a guarantee against adversarial concurrent
mutation by another process with the same operating-system identity.
"""

from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.utils.path_security import canonicalize_trusted_root_alias

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult


PUBLICATION_TARGET_DIGEST_ALGORITHM = "skill-evaluator-source-tree/2"

# These path-aware exclusions are part of the version-2 digest contract. Do
# not extend or reinterpret them without introducing a new algorithm version.
# Authored Tier 3 inputs under ``evals/`` are deliberately included; only the
# generated ``evals/results/`` subtree is omitted.
_PUBLICATION_EXCLUDED_DIR_NAMES = frozenset({".git", ".venv", "__pycache__", "node_modules"})
_PUBLICATION_EXCLUDED_ROOT_DIRS = frozenset({".evals", ".results", ".versions", "results", "versions"})
_PUBLICATION_EXCLUDED_DIR_PATHS = frozenset({("evals", "results")})
_PUBLICATION_EXCLUDED_ROOT_FILES = frozenset({"BENCHMARK.md", "skill-card.md", "skill.oms.sig"})
_CHUNK_SIZE = 1024 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_BINARY_FLAG = getattr(os, "O_BINARY", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = (
    os.O_RDONLY
    | _BINARY_FLAG
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOCTTY", 0)
)
_DESCRIPTOR_BACKEND = (
    os.name == "posix"
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.scandir in os.supports_fd
)
_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE = os.name == "posix"


class PublicationTargetConflictError(ValueError):
    """Raised when a producer encounters a different persisted target claim."""


@dataclass(frozen=True, slots=True)
class _PublicationEntry:
    parts: tuple[str, ...]
    kind: str
    mode: int
    fingerprint: tuple[int, int, int, int, int, int, int]
    digest: str | None = None


def _digest_field(digest: Any, value: bytes) -> None:
    """Append one unambiguous, length-delimited field to a target digest."""
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _absolute_lexical(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100 - lexical normalization is intentional
    return canonicalize_trusted_root_alias(absolute)


def _is_link(metadata: os.stat_result) -> bool:
    return bool(stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_mode,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_entry(metadata: os.stat_result, path: Path, root_device: int | None) -> str:
    if _is_link(metadata):
        raise ValueError(f"publication source contains a symlink or reparse point: {path}")
    if stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
    elif stat.S_ISREG(metadata.st_mode):
        kind = "file"
        if metadata.st_nlink != 1:
            raise ValueError(f"publication source contains a hard-linked file: {path}")
    else:
        raise ValueError(f"publication source contains a special file: {path}")
    if root_device is not None and metadata.st_dev != root_device:
        raise ValueError(f"publication source crosses a filesystem mount: {path}")
    return kind


def _same_existing_path(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except (OSError, ValueError):
        return False


def _matches_generated_name(path: Path, canonical_name: str) -> bool:
    """Match an exclusion name using the filesystem's case semantics."""
    return path.name == canonical_name or _same_existing_path(path, path.with_name(canonical_name))


def _is_generated_artifact(source: Path, parts: tuple[str, ...], kind: str) -> bool:
    candidate = source.joinpath(*parts)
    if kind == "directory":
        return bool(
            any(_matches_generated_name(candidate, excluded) for excluded in _PUBLICATION_EXCLUDED_DIR_NAMES)
            or (
                len(parts) == 1
                and any(_matches_generated_name(candidate, excluded) for excluded in _PUBLICATION_EXCLUDED_ROOT_DIRS)
            )
            or parts in _PUBLICATION_EXCLUDED_DIR_PATHS
            or (len(parts) == 2 and _same_existing_path(candidate, source / "evals" / "results"))
        )
    return bool(
        kind == "file"
        and (
            _matches_generated_name(candidate, ".git")
            or (
                len(parts) == 1
                and any(_matches_generated_name(candidate, excluded) for excluded in _PUBLICATION_EXCLUDED_ROOT_FILES)
            )
        )
    )


def publication_source_entry_is_excluded(source: Path, candidate: Path) -> bool:
    """Return whether one existing source entry is omitted by the v2 recipe.

    Callers traversing a tree must invoke this for each entry before descending
    into directories. Unsafe nodes are not exclusions; the publication digest
    and runtime staging layers reject them through their own no-follow checks.
    """
    try:
        root = _absolute_lexical(source.expanduser())
        path = _absolute_lexical(candidate.expanduser())
        parts = path.relative_to(root).parts
        if not parts:
            return False
        metadata = path.lstat()
        if _is_link(metadata):
            return False
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
        else:
            return False
        return _is_generated_artifact(root, parts, kind)
    except (OSError, RuntimeError, ValueError):
        return False


def publication_source_path_is_excluded(source: Path, candidate: Path) -> bool:
    """Return whether a path is at or below an entry excluded by v2."""
    try:
        root = _absolute_lexical(source.expanduser())
        path = _absolute_lexical(candidate.expanduser())
        parts = path.relative_to(root).parts
    except (OSError, RuntimeError, ValueError):
        return False
    return any(
        publication_source_entry_is_excluded(root, root.joinpath(*parts[:index])) for index in range(1, len(parts) + 1)
    )


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, _CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _entry(parts: tuple[str, ...], kind: str, metadata: os.stat_result, digest: str | None = None) -> _PublicationEntry:
    return _PublicationEntry(
        parts=parts,
        kind=kind,
        mode=stat.S_IMODE(metadata.st_mode),
        fingerprint=_fingerprint(metadata),
        digest=digest,
    )


def _open_directory_no_follow(path: Path) -> int:
    descriptor = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for component in path.parts[1:]:
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if _is_link(before) or not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"publication source path contains a symlink or non-directory: {path}")
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            opened = os.fstat(child)
            if _is_link(opened) or not stat.S_ISDIR(opened.st_mode) or not os.path.samestat(before, opened):
                os.close(child)
                raise ValueError(f"publication source directory changed while opening: {path}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _scan_descriptor_tree(
    source: Path,
    descriptor: int,
    parts: tuple[str, ...],
    root_device: int,
    entries: list[_PublicationEntry],
    *,
    reverse: bool,
    known_digests: dict[tuple[str, ...], str] | None = None,
) -> None:
    with os.scandir(descriptor) as iterator:
        names = sorted((item.name for item in iterator), reverse=reverse)
    for name in names:
        child_parts = (*parts, name)
        child_path = source.joinpath(*child_parts)
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        kind = _validate_entry(before, child_path, root_device)
        if _is_generated_artifact(source, child_parts, kind):
            continue
        if kind == "directory":
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                if _fingerprint(opened) != _fingerprint(before):
                    raise ValueError(f"publication source directory was replaced: {child_path}")
                entries.append(_entry(child_parts, kind, opened))
                _scan_descriptor_tree(
                    source,
                    child,
                    child_parts,
                    root_device,
                    entries,
                    reverse=reverse,
                    known_digests=known_digests,
                )
                if _fingerprint(os.fstat(child)) != _fingerprint(opened):
                    raise ValueError(f"publication source directory changed while scanning: {child_path}")
            finally:
                os.close(child)
            continue
        child = os.open(name, _FILE_FLAGS, dir_fd=descriptor)
        try:
            opened = os.fstat(child)
            _validate_entry(opened, child_path, root_device)
            if _fingerprint(opened) != _fingerprint(before):
                raise ValueError(f"publication source file was replaced: {child_path}")
            if known_digests is None:
                digest = _hash_descriptor(child)
            else:
                expected_digest = known_digests.get(child_parts)
                if expected_digest is None:
                    raise ValueError(f"publication source file appeared while sealing: {child_path}")
                digest = _hash_descriptor(child)
                if digest != expected_digest:
                    raise ValueError(f"publication source file changed while sealing: {child_path}")
            after = os.fstat(child)
            named_after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if _fingerprint(after) != _fingerprint(opened) or _fingerprint(named_after) != _fingerprint(before):
                raise ValueError(f"publication source file changed while scanning: {child_path}")
            entries.append(_entry(child_parts, kind, after, digest))
        finally:
            os.close(child)


def _manifests_match(first: Iterable[_PublicationEntry], second: Iterable[_PublicationEntry]) -> bool:
    return sorted(first, key=lambda entry: entry.parts) == sorted(second, key=lambda entry: entry.parts)


def _descriptor_manifest(target: Path) -> tuple[_PublicationEntry, ...]:
    if target.is_dir():
        descriptor = _open_directory_no_follow(target)
        try:
            before = os.fstat(descriptor)
            root_kind = _validate_entry(before, target, before.st_dev)
            entries = [_entry((), root_kind, before)]
            _scan_descriptor_tree(target, descriptor, (), before.st_dev, entries, reverse=False)
            if _fingerprint(os.fstat(descriptor)) != _fingerprint(before):
                raise ValueError("publication source root changed while scanning")
            revalidated_root = os.fstat(descriptor)
            revalidated = [_entry((), root_kind, revalidated_root)]
            _scan_descriptor_tree(target, descriptor, (), before.st_dev, revalidated, reverse=True)
            if _fingerprint(os.fstat(descriptor)) != _fingerprint(revalidated_root) or not _manifests_match(
                entries, revalidated
            ):
                raise ValueError("publication source changed while revalidating")
            sealed_root = os.fstat(descriptor)
            sealed = [_entry((), root_kind, sealed_root)]
            known_digests = {
                entry.parts: entry.digest for entry in revalidated if entry.kind == "file" and entry.digest is not None
            }
            _scan_descriptor_tree(
                target,
                descriptor,
                (),
                before.st_dev,
                sealed,
                reverse=False,
                known_digests=known_digests,
            )
            if _fingerprint(os.fstat(descriptor)) != _fingerprint(sealed_root) or not _manifests_match(
                revalidated, sealed
            ):
                raise ValueError("publication source changed while sealing")
            return tuple(entries)
        finally:
            os.close(descriptor)

    parent = _open_directory_no_follow(target.parent)
    try:
        before = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
        if _validate_entry(before, target, before.st_dev) != "file":
            raise ValueError("publication source is not a regular file")
        descriptor = os.open(target.name, _FILE_FLAGS, dir_fd=parent)
        try:
            opened = os.fstat(descriptor)
            if _fingerprint(opened) != _fingerprint(before):
                raise ValueError("publication source file was replaced")
            digest = _hash_descriptor(descriptor)
            after = os.fstat(descriptor)
            named_after = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
            if _fingerprint(after) != _fingerprint(opened) or _fingerprint(named_after) != _fingerprint(before):
                raise ValueError("publication source file changed while scanning")
            revalidated_digest = _hash_descriptor(descriptor)
            revalidated = os.fstat(descriptor)
            named_revalidated = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
            if (
                digest != revalidated_digest
                or _fingerprint(revalidated) != _fingerprint(after)
                or _fingerprint(named_revalidated) != _fingerprint(named_after)
            ):
                raise ValueError("publication source file changed while revalidating")
            return (_entry((), "file", revalidated, revalidated_digest),)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _validate_fallback_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        metadata = current.lstat()
        if _is_link(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"publication source path contains a symlink or non-directory: {current}")


def _fallback_opened_matches_named(opened: os.stat_result, named: os.stat_result) -> bool:
    """Compare fallback metadata without assuming Windows CRT identities."""
    if _PATH_DESCRIPTOR_IDENTITIES_COMPARABLE:
        return _fingerprint(opened) == _fingerprint(named)
    return (
        stat.S_IFMT(opened.st_mode) == stat.S_IFMT(named.st_mode)
        and opened.st_nlink == named.st_nlink
        and opened.st_size == named.st_size
    )


def _hash_path_checked(path: Path, before: os.stat_result, root_device: int) -> str:
    descriptor = os.open(path, _FILE_FLAGS)
    try:
        opened = os.fstat(descriptor)
        _validate_entry(
            opened,
            path,
            root_device if _PATH_DESCRIPTOR_IDENTITIES_COMPARABLE else None,
        )
        if not _fallback_opened_matches_named(opened, before):
            raise ValueError(f"publication source file was replaced: {path}")
        digest = _hash_descriptor(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = path.lstat()
    if _fingerprint(after) != _fingerprint(opened) or _fingerprint(named_after) != _fingerprint(before):
        raise ValueError(f"publication source file changed while scanning: {path}")
    return digest


def _scan_fallback_tree(
    source: Path,
    parts: tuple[str, ...],
    root_device: int,
    entries: list[_PublicationEntry],
    *,
    reverse: bool,
    known_digests: dict[tuple[str, ...], str] | None = None,
) -> None:
    directory = source.joinpath(*parts)
    names = sorted((item.name for item in os.scandir(directory)), reverse=reverse)
    for name in names:
        child_parts = (*parts, name)
        child = source.joinpath(*child_parts)
        before = child.lstat()
        kind = _validate_entry(before, child, root_device)
        if _is_generated_artifact(source, child_parts, kind):
            continue
        if kind == "directory":
            entries.append(_entry(child_parts, kind, before))
            _scan_fallback_tree(
                source,
                child_parts,
                root_device,
                entries,
                reverse=reverse,
                known_digests=known_digests,
            )
            if _fingerprint(child.lstat()) != _fingerprint(before):
                raise ValueError(f"publication source directory changed while scanning: {child}")
        else:
            if known_digests is None:
                digest = _hash_path_checked(child, before, root_device)
                metadata = before
            else:
                expected_digest = known_digests.get(child_parts)
                if expected_digest is None:
                    raise ValueError(f"publication source file appeared while sealing: {child}")
                digest = _hash_path_checked(child, before, root_device)
                if digest != expected_digest:
                    raise ValueError(f"publication source file changed while sealing: {child}")
                metadata = before
            entries.append(_entry(child_parts, kind, metadata, digest))


def _fallback_manifest(target: Path) -> tuple[_PublicationEntry, ...]:
    if target.is_dir():
        _validate_fallback_components(target)
        before = target.lstat()
        root_kind = _validate_entry(before, target, before.st_dev)
        entries = [_entry((), root_kind, before)]
        _scan_fallback_tree(target, (), before.st_dev, entries, reverse=False)
        if _fingerprint(target.lstat()) != _fingerprint(before):
            raise ValueError("publication source root changed while scanning")
        revalidated_root = target.lstat()
        revalidated = [_entry((), root_kind, revalidated_root)]
        _scan_fallback_tree(target, (), before.st_dev, revalidated, reverse=True)
        if _fingerprint(target.lstat()) != _fingerprint(revalidated_root) or not _manifests_match(entries, revalidated):
            raise ValueError("publication source changed while revalidating")
        sealed_root = target.lstat()
        sealed = [_entry((), root_kind, sealed_root)]
        known_digests = {
            entry.parts: entry.digest for entry in revalidated if entry.kind == "file" and entry.digest is not None
        }
        _scan_fallback_tree(
            target,
            (),
            before.st_dev,
            sealed,
            reverse=False,
            known_digests=known_digests,
        )
        if _fingerprint(target.lstat()) != _fingerprint(sealed_root) or not _manifests_match(
            revalidated,
            sealed,
        ):
            raise ValueError("publication source changed while sealing")
        return tuple(entries)

    _validate_fallback_components(target.parent)
    before = target.lstat()
    if _validate_entry(before, target, before.st_dev) != "file":
        raise ValueError("publication source is not a regular file")
    digest = _hash_path_checked(target, before, before.st_dev)
    revalidated_before = target.lstat()
    revalidated_digest = _hash_path_checked(target, revalidated_before, revalidated_before.st_dev)
    if _fingerprint(revalidated_before) != _fingerprint(before) or revalidated_digest != digest:
        raise ValueError("publication source file changed while revalidating")
    return (_entry((), "file", revalidated_before, revalidated_digest),)


def _normalized_relative_path(parts: tuple[str, ...]) -> str:
    """Return the platform-independent NFC path used by the recipe."""
    return unicodedata.normalize("NFC", "/".join(parts))


def _manifest_digest(
    entries: Iterable[Any],
) -> str:
    """Digest securely captured manifest entries using recipe version 2."""
    captured_entries = tuple(entries)
    normalized_entries: list[tuple[str, Any]] = []
    normalized_paths: set[str] = set()
    for entry in captured_entries:
        relative = _normalized_relative_path(entry.parts)
        relative.encode("utf-8", errors="strict")
        if relative in normalized_paths:
            raise ValueError(f"normalized publication path collision: {relative}")
        normalized_paths.add(relative)
        normalized_entries.append((relative, entry))

    digest = hashlib.sha256()
    _digest_field(digest, PUBLICATION_TARGET_DIGEST_ALGORITHM.encode("ascii"))
    for relative, entry in sorted(normalized_entries, key=lambda item: item[0].encode("utf-8")):
        if entry.kind == "directory":
            node_kind = b"D"
        elif entry.kind == "file":
            node_kind = b"F"
        else:
            raise ValueError(f"unsupported publication node kind: {entry.kind}")
        if node_kind == b"D":
            if entry.digest is not None:
                raise ValueError("a publication directory cannot contain a file digest")
            content_digest = b""
        else:
            if not isinstance(entry.digest, str) or len(entry.digest) != 64:
                raise ValueError("a publication file requires a SHA-256 digest")
            content_digest = bytes.fromhex(entry.digest)
            if content_digest.hex() != entry.digest:
                raise ValueError("a publication file digest is not canonical lowercase hexadecimal")
        for field in (
            relative.encode("utf-8"),
            node_kind,
            (entry.mode & 0o111).to_bytes(1, "big"),
            content_digest,
        ):
            _digest_field(digest, field)
    return digest.hexdigest()


def publication_source_digest(target_path: Path) -> str | None:
    """Return the canonical digest of one stable, safe source snapshot.

    Directories use a descriptor-anchored manifest on supported POSIX systems
    and checked pre/post metadata elsewhere. Both detect incidental mid-scan
    mutation. Symlinks, reparse points, hard-linked regular files, special
    files, mount crossings, invalid Unicode paths, and NFC path collisions fail
    closed by returning ``None``.
    """
    try:
        target = _absolute_lexical(target_path.expanduser())
        metadata = target.lstat()
        _validate_entry(metadata, target, None)
        entries = _descriptor_manifest(target) if _DESCRIPTOR_BACKEND else _fallback_manifest(target)
        return _manifest_digest(entries)
    except (
        OSError,
        OverflowError,
        RuntimeError,
        UnicodeError,
        ValueError,
    ):
        return None
    return None


def _canonical_directory_entry_name(path: Path) -> str:
    """Return the spelling recorded by the parent directory for *path*."""
    metadata = path.lstat()
    matches: list[str] = []
    with os.scandir(path.parent) as entries:
        for entry in entries:
            try:
                # DirEntry.stat() exposes zero identity fields on Windows;
                # re-stat the named path before comparing filesystem identity.
                observed = path.parent.joinpath(entry.name).lstat()
            except OSError:
                continue
            if os.path.samestat(metadata, observed):
                matches.append(entry.name)
    if path.name in matches:
        return path.name
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"publication source directory entry is ambiguous: {path}")


def publication_target_from_path(target_path: Path) -> dict[str, str] | None:
    """Return the canonical publication target for one source snapshot."""
    try:
        target = _absolute_lexical(target_path.expanduser())
        entry_name = _canonical_directory_entry_name(target)
        skill_name = unicodedata.normalize("NFC", entry_name)
        if not skill_name:
            return None
        skill_name.encode("utf-8", errors="strict")
    except (OSError, RuntimeError, UnicodeError, ValueError):
        return None
    digest = publication_source_digest(target)
    if digest is None:
        return None
    try:
        if _canonical_directory_entry_name(target) != entry_name:
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return {
        "skill_name": skill_name,
        "skill_digest": f"sha256:{digest}",
        "skill_digest_algorithm": PUBLICATION_TARGET_DIGEST_ALGORITHM,
    }


def _identity_containers(result: ValidationResult) -> tuple[dict[str, Any], ...]:
    """Return all persisted identity containers owned by one result."""
    metadata = result.metadata
    if not isinstance(metadata, dict):
        raise PublicationTargetConflictError("validation result metadata is not a mapping")
    containers = [metadata]
    payload = metadata.get("agent_eval")
    if isinstance(payload, dict):
        containers.append(payload)
        summary = payload.get("summary")
        if isinstance(summary, dict):
            containers.append(summary)
    return tuple(containers)


def stamp_publication_identity(
    results: Iterable[ValidationResult],
    target: dict[str, str],
) -> dict[str, str]:
    """Atomically stamp an already captured, producer-owned target identity."""
    expected_keys = {"skill_name", "skill_digest", "skill_digest_algorithm"}
    if (
        set(target) != expected_keys
        or not isinstance(target.get("skill_name"), str)
        or not target["skill_name"]
        or unicodedata.normalize("NFC", target["skill_name"]) != target["skill_name"]
        or not isinstance(target.get("skill_digest"), str)
        or len(target["skill_digest"]) != 71
        or not target["skill_digest"].startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in target["skill_digest"][7:])
        or target.get("skill_digest_algorithm") != PUBLICATION_TARGET_DIGEST_ALGORITHM
    ):
        raise PublicationTargetConflictError("publication target identity is malformed")

    containers: list[dict[str, Any]] = []
    for result in tuple(results):
        containers.extend(_identity_containers(result))
    for container in containers:
        if "publication_target" in container and container["publication_target"] != target:
            raise PublicationTargetConflictError("conflicting publication target already persisted")
    for container in containers:
        if "publication_target" not in container:
            container["publication_target"] = dict(target)
    return target


def _mark_publication_target_conflict(
    results: Iterable[ValidationResult],
    message: str,
) -> None:
    """Remove contradictory target claims before persisting a conflict."""
    for result in results:
        if not isinstance(result.metadata, dict):
            continue
        for container in _identity_containers(result):
            container.pop("publication_target", None)
        result.metadata["publication_target_conflict"] = message


def finalize_publication_target(
    results: Iterable[ValidationResult],
    target_path: Path,
    initial_target: dict[str, str] | None,
) -> dict[str, str] | None:
    """Stamp results only when the producer's source stayed unchanged."""
    result_list = tuple(results)
    final_target = publication_target_from_path(target_path)
    if initial_target is None or final_target != initial_target:
        _mark_publication_target_conflict(result_list, "source changed during validation")
        return None
    try:
        return stamp_publication_identity(result_list, final_target)
    except PublicationTargetConflictError:
        _mark_publication_target_conflict(result_list, "conflicting producer identity")
        return None


def stamp_publication_target(
    results: Iterable[ValidationResult],
    target_path: Path,
) -> dict[str, str] | None:
    """Atomically stamp newly generated results with their source identity.

    Every preexisting claim is checked before any result is changed. A claim
    for another snapshot (including a malformed or explicit-null claim) raises
    :class:`PublicationTargetConflictError` instead of being retained beside
    newly stamped matching claims.
    """
    target = publication_target_from_path(target_path)
    if target is None:
        return None

    return stamp_publication_identity(results, target)


__all__ = [
    "PUBLICATION_TARGET_DIGEST_ALGORITHM",
    "PublicationTargetConflictError",
    "finalize_publication_target",
    "publication_source_digest",
    "publication_target_from_path",
    "stamp_publication_identity",
    "stamp_publication_target",
]
