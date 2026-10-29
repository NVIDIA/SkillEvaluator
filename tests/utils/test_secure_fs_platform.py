# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from skillevaluator.utils import secure_fs

_NATIVE_FUNCTIONS = (
    "create_file",
    "close_handle",
    "get_file_information",
    "get_final_path",
    "set_file_information",
    "nt_create_file",
    "nt_set_information_file",
    "rtl_nt_status_to_dos_error",
)


def _admission(
    root: Path, *, max_paths: int = 10, excluded_dirs: tuple[str, ...] = ()
) -> secure_fs._DiscoveryAdmission:
    return secure_fs._DiscoveryAdmission(
        root,
        selected=lambda _relative: False,
        excluded_dirs=excluded_dirs,
        max_paths=max_paths,
        max_depth=None,
        allow_context_alias=True,
    )


@pytest.mark.parametrize("max_depth", [True, 0, -1, 65, 1.5])
def test_discovery_rejects_invalid_configured_depth(tmp_path: Path, max_depth: object) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(ValueError, match=r"max_depth|depth"):
        secure_fs.discover_secure_files(
            root,
            selected=lambda _relative: False,
            max_paths=10,
            max_depth=max_depth,  # type: ignore[arg-type]
        )


def test_windows_discovery_reparse_fallback_opens_object_without_follow(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, int | str]] = []
    reparse = secure_fs._WindowsHandleMetadata(
        attributes=0x400,
        volume_serial=7,
        file_id=11,
        size=9,
        link_count=1,
    )

    def open_relative(parent_handle: int, name: str, **kwargs: int) -> int:
        calls.append({"parent_handle": parent_handle, "name": name, **kwargs})
        if len(calls) == 1:
            raise OSError(4390, "reparse encountered")
        return 456

    monkeypatch.setattr(secure_fs, "_windows_open_relative_handle", open_relative)
    monkeypatch.setattr(secure_fs, "_windows_handle_metadata", lambda _handle: reparse)

    handle, metadata = secure_fs._windows_open_discovery_handle(123, "CLAUDE.md")

    assert handle == 456
    assert metadata == reparse
    assert calls[0]["share"] & 0x4 == 0
    assert calls[0]["object_attributes_flags"] & secure_fs._WINDOWS_OBJ_DONT_REPARSE
    assert calls[1]["share"] & 0x4 == 0
    assert calls[1]["object_attributes_flags"] == secure_fs._WINDOWS_OBJ_CASE_INSENSITIVE
    assert calls[1]["create_options"] & secure_fs._WINDOWS_FILE_OPEN_REPARSE_POINT


def test_windows_reparse_snapshot_ignores_cross_api_link_count_difference() -> None:
    path_metadata = os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 9, 0, 0, 0))
    handle_metadata = secure_fs._WindowsHandleMetadata(
        attributes=0x400,
        volume_serial=7,
        file_id=11,
        size=9,
        link_count=0,
    )

    secure_fs._validate_windows_entry_snapshot(path_metadata, handle_metadata, Path("CLAUDE.md"))


def test_windows_directory_name_enumeration_occurs_between_handle_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    snapshot = secure_fs._WindowsHandleMetadata(
        attributes=0x10,
        volume_serial=7,
        file_id=11,
        size=0,
        link_count=1,
        last_write_time=13,
    )
    events: list[str] = []

    class _Entry:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Scandir:
        def __enter__(self):
            return iter([_Entry("z.md"), _Entry("a.md")])

        def __exit__(self, *_args: object) -> None:
            return None

    def metadata(_handle: int) -> secure_fs._WindowsHandleMetadata:
        events.append("snapshot")
        return snapshot

    def scandir(path: Path) -> _Scandir:
        assert path == root
        events.append("scandir")
        return _Scandir()

    monkeypatch.setattr(secure_fs, "_windows_handle_metadata", metadata)
    monkeypatch.setattr(secure_fs.os, "scandir", scandir)

    names, stable = secure_fs._windows_enumerate_pinned_directory_names(
        root, 123, Path(), snapshot, admission=_admission(root)
    )

    assert names == ["a.md", "z.md"]
    assert stable == snapshot
    assert events == ["snapshot", "scandir", "snapshot"]


def test_windows_directory_name_enumeration_is_bounded_by_path_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    snapshot = secure_fs._WindowsHandleMetadata(
        attributes=0x10,
        volume_serial=7,
        file_id=11,
        size=0,
        link_count=1,
        last_write_time=13,
    )

    class _Entry:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Scandir:
        def __enter__(self):
            return iter(_Entry(f"entry-{index}") for index in range(4))

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(secure_fs, "_windows_handle_metadata", lambda _handle: snapshot)
    monkeypatch.setattr(secure_fs.os, "scandir", lambda _path: _Scandir())

    # Two budgeted paths plus one excluded directory name may be listed.
    with pytest.raises(secure_fs.SecurePathError) as caught:
        secure_fs._windows_enumerate_pinned_directory_names(
            root,
            123,
            Path(),
            snapshot,
            admission=_admission(root, max_paths=2, excluded_dirs=(".git",)),
        )

    assert caught.value.code == "path_count_limit"
    assert caught.value.metadata == {"actual": 4, "limit": 2}


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (
            secure_fs._WindowsHandleMetadata(
                attributes=0x400,
                volume_serial=7,
                file_id=11,
                size=12,
                link_count=1,
            ),
            r"reparse",
        ),
        (
            secure_fs._WindowsHandleMetadata(
                attributes=0,
                volume_serial=7,
                file_id=11,
                size=12,
                link_count=2,
            ),
            r"hard.?link|link count",
        ),
    ],
)
def test_windows_reader_handle_contract_rejects_redirects_and_hardlinks(
    monkeypatch: pytest.MonkeyPatch,
    metadata: secure_fs._WindowsHandleMetadata,
    match: str,
) -> None:
    monkeypatch.setattr(secure_fs, "_windows_handle_metadata", lambda _handle: metadata)

    with pytest.raises(secure_fs.SecurePathError, match=match):
        secure_fs._validate_windows_read_file_handle(123, Path("nested/SKILL.md"))


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows junction APIs")
def test_windows_discovery_rejects_existing_junction_before_descent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("outside", encoding="utf-8")
    junction = root / "linked"
    created = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"Windows junction creation unavailable: {created.stderr or created.stdout}")

    with pytest.raises(secure_fs.SecurePathError, match=r"linked directory|reparse"):
        secure_fs.discover_secure_files(root, selected=lambda _relative: False, max_paths=20)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows junction APIs")
def test_windows_discovery_rejects_directory_swapped_to_junction_before_descent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "guide.md").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "guide.md").write_text("outside", encoding="utf-8")
    moved = root / "original-child"
    original_open = secure_fs._windows_open_relative_handle
    swapped = False

    def race_open(parent_handle: int, name: str, **kwargs: int) -> int:
        nonlocal swapped
        is_directory_descent = bool(kwargs["create_options"] & secure_fs._WINDOWS_FILE_DIRECTORY_FILE)
        if name == "child" and is_directory_descent and not swapped:
            child.rename(moved)
            created = subprocess.run(
                ["cmd", "/d", "/c", "mklink", "/J", str(child), str(outside)],
                capture_output=True,
                text=True,
                check=False,
            )
            if created.returncode != 0:
                pytest.skip(f"Windows junction creation unavailable: {created.stderr or created.stdout}")
            swapped = True
        return original_open(parent_handle, name, **kwargs)

    monkeypatch.setattr(secure_fs, "_windows_open_relative_handle", race_open)

    with pytest.raises(secure_fs.SecurePathError, match=r"securely open|reparse|changed"):
        secure_fs.discover_secure_files(root, selected=lambda relative: relative.suffix == ".md", max_paths=20)
    assert swapped


# Run in a fresh interpreter, so the ctypes caches start cold. The patched
# namespace constructor holds the first builder until a second thread arrives
# (or one second passes), so two threads that both missed a cache would
# build two structure sets.
_RACE_PRELUDE = """
import threading
from skillevaluator.utils import secure_fs

barrier = threading.Barrier(2, timeout=1.0)
real_namespace = secure_fs.SimpleNamespace

def namespace_after_a_racer(**fields):
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    return real_namespace(**fields)

secure_fs.SimpleNamespace = namespace_after_a_racer

def race(function):
    results = []
    threads = [threading.Thread(target=lambda: results.append(function())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results
"""


class _TreeBackedWindowsHandles:
    """Native handle stand-ins backed by a real tree, so the Windows walker runs on any platform.

    Handles never follow links: a symlink is reported as a reparse point, with
    the directory attribute when it points at a directory, like a junction.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.paths: dict[int, Path] = {}
        self.closed: list[int] = []
        monkeypatch.setattr(secure_fs, "_windows_open_anchored_directory_chain", self.open_chain)
        monkeypatch.setattr(secure_fs, "_windows_open_relative_handle", self.open_relative)
        monkeypatch.setattr(secure_fs, "_windows_handle_metadata", self.metadata)
        monkeypatch.setattr(secure_fs, "_windows_close_handle", self.closed.append)

    def _open(self, path: Path) -> int:
        handle = 100 + len(self.paths)
        self.paths[handle] = path
        return handle

    def open_chain(self, path: Path, *, expected: os.stat_result) -> list[int]:
        assert os.path.samestat(path.lstat(), expected)
        return [self._open(path)]

    def open_relative(self, parent_handle: int, name: str, **kwargs: int) -> int:
        path = self.paths[parent_handle] / name
        # Like NtCreateFile, OBJ_DONT_REPARSE (part of the default flags) refuses a reparse point.
        flags = kwargs.get("object_attributes_flags", secure_fs._WINDOWS_OBJECT_ATTRIBUTES_FLAGS)
        if path.is_symlink() and flags & secure_fs._WINDOWS_OBJ_DONT_REPARSE:
            raise OSError(4395, "A reparse point was encountered while opening the object.")
        return self._open(path)

    def metadata(self, handle: int) -> secure_fs._WindowsHandleMetadata:
        path = self.paths[handle]
        metadata = path.lstat()
        attributes = 0x10 if stat.S_ISDIR(metadata.st_mode) else 0
        if path.is_symlink():
            attributes = 0x400 | (0x10 if path.is_dir() else 0)
        return secure_fs._WindowsHandleMetadata(
            attributes=attributes,
            volume_serial=metadata.st_dev,
            file_id=metadata.st_ino,
            size=metadata.st_size,
            link_count=metadata.st_nlink,
            last_write_time=metadata.st_mtime_ns,
        )

    def walk(self, root: Path, **options: object) -> list[str]:
        admission = secure_fs._DiscoveryAdmission(
            root,
            selected=lambda relative: relative.suffix == ".md",
            excluded_dirs=options.pop("excluded_dirs", ()),
            max_paths=options.pop("max_paths", 50),
            max_depth=None,
            allow_context_alias=True,
        )
        secure_fs._walk_windows(root, root.lstat(), admission)
        return [file.rel_path for file in admission.selected_files()]

    def all_closed(self) -> bool:
        return sorted(self.closed) == sorted(self.paths)


_SKIP_ON_NATIVE_WINDOWS = pytest.mark.skipif(
    os.name == "nt", reason="native Windows walks are covered by the real-handle tests"
)


@_SKIP_ON_NATIVE_WINDOWS
def test_windows_walker_selects_like_the_posix_walker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "skill"
    (root / "references" / "deep").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".git" / "ignored.md").write_text("pruned")
    (root / "SKILL.md").write_text("skill")
    (root / "references" / "guide.md").write_text("guide")
    (root / "references" / "deep" / "notes.md").write_text("notes")
    (root / "references" / "data.bin").write_bytes(b"x")
    (root / "AGENTS.md").write_text("agents")
    (root / "CLAUDE.md").symlink_to("AGENTS.md")
    posix = [
        file.rel_path
        for file in secure_fs.discover_secure_files(
            root, selected=lambda relative: relative.suffix == ".md", excluded_dirs=(".git",), max_paths=50
        )
    ]
    handles = _TreeBackedWindowsHandles(monkeypatch)

    selected = handles.walk(root, excluded_dirs=(".git",))

    assert selected == posix == ["AGENTS.md", "SKILL.md", "references/deep/notes.md", "references/guide.md"]
    assert handles.all_closed()


@_SKIP_ON_NATIVE_WINDOWS
@pytest.mark.parametrize(
    ("make_link", "match"),
    [
        (lambda root, outside: (root / "linked").symlink_to(outside, target_is_directory=True), "linked directory"),
        (lambda root, outside: (root / "guide.md").symlink_to(outside / "secret.md"), "symlink or reparse"),
        (lambda root, _outside: (root / "CLAUDE.md").symlink_to("SKILL.md"), "symlink or reparse"),
    ],
    ids=["junction", "file-link", "wrong-alias"],
)
def test_windows_walker_refuses_redirects_and_closes_every_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_link, match: str
) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("skill")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret")
    make_link(root, outside)
    handles = _TreeBackedWindowsHandles(monkeypatch)

    with pytest.raises(secure_fs.SecurePathError, match=match):
        handles.walk(root)

    assert handles.all_closed()
    assert all(not path.is_relative_to(outside) for path in handles.paths.values())


@_SKIP_ON_NATIVE_WINDOWS
def test_windows_walker_consumes_the_path_budget_like_the_posix_walker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "skill"
    (root / "b-dir").mkdir(parents=True)
    for name in ("c.md", "a.md", "b-dir/z.md"):
        (root / name).write_text("x")
    with pytest.raises(secure_fs.SecurePathError) as posix:
        secure_fs.discover_secure_files(
            root, selected=lambda relative: relative.suffix == ".md", excluded_dirs=("x", "y"), max_paths=3
        )
    handles = _TreeBackedWindowsHandles(monkeypatch)

    with pytest.raises(secure_fs.SecurePathError) as windows:
        handles.walk(root, excluded_dirs=("x", "y"), max_paths=3)

    assert (windows.value.relative_path, windows.value.metadata) == (posix.value.relative_path, posix.value.metadata)
    assert windows.value.relative_path == "b-dir/z.md"
    assert handles.all_closed()


@pytest.mark.parametrize(
    ("attributes", "is_directory", "is_reparse", "is_plain_directory"),
    [
        (0x10, True, False, True),
        (0x10 | 0x400, True, True, False),
        (0x400, False, True, False),
        (0x20, False, False, False),
    ],
    ids=["directory", "junction", "file-link", "regular-file"],
)
def test_windows_handle_metadata_names_its_attribute_bits(
    attributes: int, is_directory: bool, is_reparse: bool, is_plain_directory: bool
) -> None:
    metadata = secure_fs._WindowsHandleMetadata(
        attributes=attributes, volume_serial=7, file_id=11, size=0, link_count=1
    )

    assert (metadata.is_directory, metadata.is_reparse, metadata.is_plain_directory) == (
        is_directory,
        is_reparse,
        is_plain_directory,
    )
    assert metadata.same_identity(secure_fs._WindowsHandleMetadata(0, 7, 11, 99, 2, 5))
    assert not metadata.same_identity(secure_fs._WindowsHandleMetadata(attributes, 8, 11, 0, 1))
    assert not metadata.same_identity(secure_fs._WindowsHandleMetadata(attributes, 7, 12, 0, 1))


def _tree_with_link(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "skills" / "demo").mkdir(parents=True)
    (root / "skills" / "demo" / "SKILL.md").write_text("demo")
    (root / "README.md").write_text("readme")
    (tmp_path / "outside").mkdir()
    try:
        (root / "linked").symlink_to(tmp_path / "outside", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    return root


@pytest.mark.parametrize(
    ("relative", "outcome", "failing_index", "metadata_of", "error"),
    [
        ("skills/demo/SKILL.md", "ok", None, "skills/demo/SKILL.md", None),
        ("skills/demo", "ok", None, "skills/demo", None),
        ("", "ok", None, None, None),
        ("skills/missing/SKILL.md", "missing", 1, None, FileNotFoundError),
        ("linked/SKILL.md", "link", 0, "linked", None),
        ("README.md/SKILL.md", "not_dir", 0, "README.md", None),
    ],
)
def test_lstat_walk_stops_at_the_first_unusable_component(
    tmp_path: Path,
    relative: str,
    outcome: str,
    failing_index: int | None,
    metadata_of: str | None,
    error: type[OSError] | None,
) -> None:
    root = _tree_with_link(tmp_path)

    walk = secure_fs.lstat_walk(root, PurePosixPath(relative))

    assert (walk.outcome, walk.failing_index) == (outcome, failing_index)
    if metadata_of is None:
        assert walk.metadata is None
    else:
        assert os.path.samestat(walk.metadata, (root / metadata_of).lstat())
    if error is None:
        assert walk.error is None
    else:
        assert isinstance(walk.error, error)


def test_lstat_walk_inspects_each_component_in_order_and_never_below_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree_with_link(tmp_path)
    inspected: list[str] = []
    real_lstat = Path.lstat

    def recording_lstat(path: Path) -> os.stat_result:
        inspected.append(path.relative_to(root).as_posix())
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", recording_lstat)

    assert secure_fs.lstat_walk(root, PurePosixPath("skills/demo/SKILL.md")).outcome == "ok"
    assert secure_fs.lstat_walk(root, PurePosixPath("linked/deeper/SKILL.md")).outcome == "link"
    assert inspected == ["skills", "skills/demo", "skills/demo/SKILL.md", "linked"]


def test_lstat_walk_reports_an_unreadable_component_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    (root / "skills").mkdir(parents=True)
    real_lstat = Path.lstat

    def denied_lstat(path: Path) -> os.stat_result:
        if path.name == "skills":
            raise PermissionError(13, "Permission denied", str(path))
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", denied_lstat)

    walk = secure_fs.lstat_walk(root, PurePosixPath("skills/demo"))

    assert (walk.outcome, walk.metadata, walk.failing_index) == ("error", None, 0)
    assert isinstance(walk.error, PermissionError)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handles")
def test_windows_hardlink_read_requires_explicit_script_opt_in(tmp_path: Path) -> None:
    target = tmp_path / "script.sh"
    target.write_bytes(b"echo safe")
    os.link(target, tmp_path / "linked.sh")
    with secure_fs.SecureRoot(tmp_path) as root:
        with pytest.raises(secure_fs.SecurePathError, match=r"hard.link"):
            root.read_bytes(Path("linked.sh"), 1024)
        content, metadata = root.read_bytes(Path("linked.sh"), 1024, allow_hardlinks=True)
        assert content == b"echo safe"
        assert metadata.st_nlink == 2
