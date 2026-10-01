# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

from skillevaluator.utils import secure_fs
from skillevaluator.utils.secure_fs import (
    SecurePathError,
    SecureRoot,
    discover_secure_files,
    secure_atomic_write_text,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor lifecycle tests are POSIX-specific")


def test_secure_root_closes_final_descriptor_when_post_open_fstat_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "unique-secure-root"
    root.mkdir()
    real_open = os.open
    real_fstat = os.fstat
    real_close = os.close
    root_fd: int | None = None
    fstat_calls = 0
    closed: list[int] = []

    def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal root_fd
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == root:
            root_fd = descriptor
        return descriptor

    def failing_fstat(descriptor: int):
        nonlocal fstat_calls
        if descriptor == root_fd:
            fstat_calls += 1
            if fstat_calls == 2:
                raise OSError("simulated post-open root fstat failure")
        return real_fstat(descriptor)

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(secure_fs.os, "open", tracked_open)
    monkeypatch.setattr(secure_fs.os, "fstat", failing_fstat)
    monkeypatch.setattr(secure_fs.os, "close", tracked_close)

    with pytest.raises(OSError, match="post-open"), SecureRoot(root):
        pass

    assert root_fd is not None
    assert root_fd in closed


def test_absolute_directory_open_closes_child_when_fstat_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "unique-fstat-root"
    root.mkdir()
    real_open = os.open
    real_fstat = os.fstat
    real_close = os.close
    child_fd: int | None = None
    closed: list[int] = []

    def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal child_fd
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == root:
            child_fd = descriptor
        return descriptor

    def failing_fstat(descriptor: int):
        if descriptor == child_fd:
            raise OSError("simulated child fstat failure")
        return real_fstat(descriptor)

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(secure_fs.os, "open", tracked_open)
    monkeypatch.setattr(secure_fs.os, "fstat", failing_fstat)
    monkeypatch.setattr(secure_fs.os, "close", tracked_close)

    with pytest.raises(OSError, match="child fstat"):
        secure_fs._open_absolute_directory_posix(root)

    assert child_fd is not None
    assert child_fd in closed


def test_secure_root_allows_symlinked_ancestor_when_declared_root_is_regular(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    root = real_parent / "root"
    root.mkdir(parents=True)
    (root / "guide.md").write_text("safe")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with SecureRoot(linked_parent / "root") as secure_root:
        assert secure_root.read_text(Path("guide.md"), 1024) == "safe"


def test_secure_root_rejects_symlinked_declared_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(SecurePathError, match=r"root|symlink|reparse"), SecureRoot(linked_root):
        pass


def test_secure_root_rejects_double_enter_without_replacing_descriptor(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secure_root = SecureRoot(root)

    secure_root.__enter__()
    original_descriptor = secure_root._root_fd
    try:
        with pytest.raises(SecurePathError, match=r"already|active|entered"):
            secure_root.__enter__()
        assert secure_root._root_fd == original_descriptor
    finally:
        secure_root.__exit__(None, None, None)


def test_secure_root_read_requires_active_context(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "guide.md").write_text("safe")

    with pytest.raises(SecurePathError, match=r"context|active|entered"):
        SecureRoot(root).read_text(Path("guide.md"), 1024)


def test_standalone_read_allows_symlinked_ancestor_of_regular_parent(tmp_path: Path) -> None:
    real_ancestor = tmp_path / "real-ancestor"
    real_parent = real_ancestor / "regular-parent"
    real_parent.mkdir(parents=True)
    (real_parent / "cache.json").write_text('{"safe": true}')
    linked_ancestor = tmp_path / "linked-ancestor"
    linked_ancestor.symlink_to(real_ancestor, target_is_directory=True)

    assert secure_fs.secure_read_path_text(linked_ancestor / "regular-parent" / "cache.json", 1024) == '{"safe": true}'


def test_component_open_closes_child_when_fstat_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    references = root / "references"
    references.mkdir(parents=True)
    (references / "guide.md").write_text("safe")
    real_open = os.open
    real_fstat = os.fstat
    real_close = os.close
    child_fd: int | None = None
    closed: list[int] = []

    with SecureRoot(root) as secure_root:

        def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal child_fd
            if dir_fd is None:
                descriptor = real_open(path, flags, mode)
            else:
                descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            if str(path) == "references":
                child_fd = descriptor
            return descriptor

        def failing_fstat(descriptor: int):
            if descriptor == child_fd:
                raise OSError("simulated component fstat failure")
            return real_fstat(descriptor)

        def tracked_close(descriptor: int) -> None:
            closed.append(descriptor)
            real_close(descriptor)

        monkeypatch.setattr(secure_fs.os, "open", tracked_open)
        monkeypatch.setattr(secure_fs.os, "fstat", failing_fstat)
        monkeypatch.setattr(secure_fs.os, "close", tracked_close)

        with pytest.raises(OSError, match="component fstat"):
            secure_root.read_text(Path("references/guide.md"), 1024)

    assert child_fd is not None
    assert child_fd in closed


def test_same_inode_mutation_during_read_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "guide.md"
    target.write_text("SAFE-CONTENT")
    files = discover_secure_files(root, selected=lambda path: path.suffix == ".md", max_paths=10)
    real_read = os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        if not mutated:
            target.write_text("EVIL-CONTENT")
            mutated = True
        return real_read(descriptor, count)

    monkeypatch.setattr(secure_fs.os, "read", mutating_read)

    with SecureRoot(root) as secure_root, pytest.raises(SecurePathError, match=r"changed|identity|mutation"):
        secure_root.read_file_text(files[0], 1024)
    assert mutated


def test_directory_swap_to_external_link_is_never_traversed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "safe.md").write_text("safe")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("SECRET_CANARY")
    moved = root / "moved-child"
    real_open = os.open
    swapped = False
    selected_paths: list[str] = []

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "child" and dir_fd is not None and flags & os.O_DIRECTORY and not swapped:
            child.rename(moved)
            child.symlink_to(outside, target_is_directory=True)
            swapped = True
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def selected(relative: Path) -> bool:
        selected_paths.append(relative.as_posix())
        return relative.suffix == ".md"

    monkeypatch.setattr(secure_fs.os, "open", swapping_open)

    with pytest.raises(SecurePathError, match=r"directory|link|changed|unsafe"):
        discover_secure_files(root, selected=selected, max_paths=20)

    assert swapped
    assert "child/secret.md" not in selected_paths


def test_path_budget_stops_scandir_at_limit_plus_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    for index in range(100):
        (root / f"irrelevant-{index:03}.bin").write_bytes(b"x")
    real_scandir = os.scandir
    yielded = 0

    class TrackingScandir:
        def __init__(self, path) -> None:
            self._iterator = real_scandir(path)

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, *args):
            return self._iterator.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal yielded
            entry = next(self._iterator)
            yielded += 1
            return entry

    monkeypatch.setattr(secure_fs.os, "scandir", TrackingScandir)

    with pytest.raises(SecurePathError, match=r"path.*limit"):
        discover_secure_files(root, selected=lambda _relative: False, max_paths=2)

    assert yielded == 3


def test_selected_link_never_queries_target_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "guide.md").symlink_to("missing-target")
    real_scandir = os.scandir
    target_queries: list[str] = []

    class EntryProxy:
        def __init__(self, entry) -> None:
            self._entry = entry
            self.name = entry.name

        def stat(self, *, follow_symlinks=True):
            return self._entry.stat(follow_symlinks=follow_symlinks)

        def is_dir(self, *, follow_symlinks=True):
            if follow_symlinks:
                target_queries.append(self.name)
                raise AssertionError("selected link target metadata must not be queried")
            return self._entry.is_dir(follow_symlinks=False)

    class ScandirProxy:
        def __init__(self, path) -> None:
            self._iterator = real_scandir(path)

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, *args):
            return self._iterator.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            return EntryProxy(next(self._iterator))

    monkeypatch.setattr(secure_fs.os, "scandir", ScandirProxy)

    with pytest.raises(SecurePathError, match=r"symlink|reparse|unsafe"):
        discover_secure_files(root, selected=lambda relative: relative.suffix == ".md", max_paths=10)

    assert target_queries == []


def test_selected_directory_is_rejected_as_non_regular(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "guide.md").mkdir(parents=True)

    with pytest.raises(SecurePathError, match=r"regular file|not a regular"):
        discover_secure_files(root, selected=lambda relative: relative.suffix == ".md", max_paths=10)


def test_path_budget_exact_boundary_and_limit_metadata(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "guide.md").write_text("safe")

    files = discover_secure_files(root, selected=lambda relative: relative.suffix == ".md", max_paths=2)
    assert [file.rel_path for file in files] == ["child/guide.md"]

    (root / "extra.bin").write_bytes(b"x")
    with pytest.raises(SecurePathError) as caught:
        discover_secure_files(root, selected=lambda relative: relative.suffix == ".md", max_paths=2)
    assert caught.value.metadata == {"actual": 3, "limit": 2}


def test_deep_tree_opens_each_directory_once_and_bounds_live_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    current = root
    directory_depth = 12
    for _index in range(directory_depth):
        current /= "d"
        current.mkdir()
    (current / "leaf.md").write_text("safe")
    real_open = os.open
    real_close = os.close
    directory_open_count = 0
    active_directory_descriptors: set[int] = set()
    max_active_directory_descriptors = 0

    def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal directory_open_count, max_active_directory_descriptors
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if flags & os.O_DIRECTORY:
            directory_open_count += 1
            active_directory_descriptors.add(descriptor)
            max_active_directory_descriptors = max(
                max_active_directory_descriptors,
                len(active_directory_descriptors),
            )
        return descriptor

    def tracked_close(descriptor: int) -> None:
        active_directory_descriptors.discard(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(secure_fs.os, "open", tracked_open)
    monkeypatch.setattr(secure_fs.os, "close", tracked_close)

    files = discover_secure_files(root, selected=lambda relative: relative.suffix == ".md", max_paths=30)

    assert [file.relative_path.name for file in files] == ["leaf.md"]
    assert directory_open_count == directory_depth + 1
    assert max_active_directory_descriptors == directory_depth + 1
    assert active_directory_descriptors == set()


def test_default_directory_depth_limit_fails_with_metadata(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    current = root
    for _index in range(secure_fs.MAX_SECURE_DIRECTORY_DEPTH + 1):
        current /= "d"
        current.mkdir()

    with pytest.raises(SecurePathError) as caught:
        discover_secure_files(root, selected=lambda _relative: False, max_paths=100)

    assert caught.value.code == "directory_depth_limit"
    assert caught.value.metadata == {
        "actual": secure_fs.MAX_SECURE_DIRECTORY_DEPTH + 1,
        "limit": secure_fs.MAX_SECURE_DIRECTORY_DEPTH,
    }


def test_child_entry_replacement_after_subtree_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "safe.md").write_text("safe")
    moved_child = root / "moved-child"
    real_stat = os.stat
    child_entry_stats = 0
    swapped = False

    def swapping_stat(path, *, dir_fd=None, follow_symlinks=True):
        nonlocal child_entry_stats, swapped
        if path == "child" and dir_fd is not None and not follow_symlinks:
            child_entry_stats += 1
            if child_entry_stats == 2:
                child.rename(moved_child)
                child.mkdir()
                swapped = True
        if dir_fd is None:
            return real_stat(path, follow_symlinks=follow_symlinks)
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(secure_fs.os, "stat", swapping_stat)

    with pytest.raises(SecurePathError, match=r"directory|changed|identity|unsafe"):
        discover_secure_files(root, selected=lambda relative: relative.suffix == ".md", max_paths=10)

    assert swapped


def test_directory_mutation_after_scandir_snapshot_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "safe.md").write_text("safe")
    real_scandir = os.scandir
    mutated = False

    class MutatingScandir:
        def __init__(self, path) -> None:
            self._iterator = real_scandir(path)

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, *args):
            return self._iterator.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal mutated
            try:
                return next(self._iterator)
            except StopIteration:
                if not mutated:
                    (root / "late.md").write_text("late")
                    mutated = True
                raise

    monkeypatch.setattr(secure_fs.os, "scandir", MutatingScandir)

    with pytest.raises(SecurePathError, match=r"directory.*changed|mutation|snapshot"):
        discover_secure_files(root, selected=lambda relative: relative.suffix == ".md", max_paths=10)

    assert mutated


def test_atomic_write_rejects_temporary_hardlink_race_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "cache.json"
    attacker_link = tmp_path / "attacker-link"
    real_fsync = os.fsync
    linked = False

    def hardlink_during_fsync(descriptor: int) -> None:
        nonlocal linked
        real_fsync(descriptor)
        temporary = next(tmp_path.glob(".cache.json.*.tmp"))
        os.link(temporary, attacker_link)
        linked = True

    monkeypatch.setattr(secure_fs.os, "fsync", hardlink_during_fsync)

    with pytest.raises(SecurePathError, match=r"hard.?link|single.?link|unsafe"):
        secure_atomic_write_text(destination, '{"safe": true}', 1024)

    assert linked
    assert not destination.exists()


def test_atomic_write_creates_and_replaces_regular_single_link_file(tmp_path: Path) -> None:
    destination = tmp_path / "cache.json"

    secure_atomic_write_text(destination, '{"version": 1}', 1024)
    assert destination.read_text() == '{"version": 1}'

    secure_atomic_write_text(destination, '{"version": 2}', 1024)
    assert destination.read_text() == '{"version": 2}'
    assert destination.stat().st_nlink == 1


def test_atomic_write_rejects_declared_parent_rename_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    destination = parent / "cache.json"
    real_fsync = os.fsync
    swapped = False

    def rename_parent_during_fsync(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        parent.rename(moved_parent)
        parent.mkdir()
        swapped = True

    monkeypatch.setattr(secure_fs.os, "fsync", rename_parent_during_fsync)

    with pytest.raises(SecurePathError, match=r"parent|directory|changed|unsafe"):
        secure_atomic_write_text(destination, '{"safe": true}', 1024)

    assert swapped
    assert not destination.exists()
    assert len(list(moved_parent.glob(".cache.json.*.tmp"))) == 1


def test_atomic_write_rejects_parent_rename_after_publish_without_touching_new_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    destination = parent / "cache.json"
    real_replace = os.replace
    swapped = False

    def rename_parent_after_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None) -> None:
        nonlocal swapped
        real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        parent.rename(moved_parent)
        parent.mkdir()
        swapped = True

    monkeypatch.setattr(secure_fs.os, "replace", rename_parent_after_replace)

    with pytest.raises(SecurePathError, match=r"parent|directory|changed|unsafe"):
        secure_atomic_write_text(destination, '{"safe": true}', 1024)

    assert swapped
    assert list(parent.iterdir()) == []
    assert (moved_parent / "cache.json").read_text() == '{"safe": true}'


def test_atomic_write_failure_never_unlinks_swapped_temporary_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "cache.json"
    saved_payload = tmp_path / "saved-payload"
    temporary_name: Path | None = None
    real_fsync = os.fsync

    def swap_temporary_during_fsync(descriptor: int) -> None:
        nonlocal temporary_name
        real_fsync(descriptor)
        temporary_name = next(tmp_path.glob(".cache.json.*.tmp"))
        temporary_name.rename(saved_payload)
        temporary_name.write_text("INNOCENT_CANARY")

    monkeypatch.setattr(secure_fs.os, "fsync", swap_temporary_during_fsync)

    with pytest.raises(SecurePathError, match=r"changed|identity|unsafe"):
        secure_atomic_write_text(destination, "PAYLOAD", 1024)

    assert temporary_name is not None
    assert temporary_name.read_text() == "INNOCENT_CANARY"
    assert saved_payload.read_text() == "PAYLOAD"
    assert not destination.exists()
