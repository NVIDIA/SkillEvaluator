# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

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


def _fake_windows_api(**functions: object) -> SimpleNamespace:
    """A stand-in for the bound native functions; any call not supplied fails the test."""

    def unexpected(*_args: object) -> object:
        raise AssertionError("unexpected native Windows call")

    return SimpleNamespace(
        invalid_handle=ctypes.c_void_p(-1).value,
        **{name: functions.get(name, unexpected) for name in _NATIVE_FUNCTIONS},
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


@pytest.mark.parametrize(
    "name",
    [
        "CON",
        "con.json",
        "con .json",
        "COM¹.json",
        "LPT³",
        "cache.json:stream",
        "cache?.json",
        "cache|.json",
        "cache\x00.json",
        "cache\x1f.json",
        "cache\ud800.json",
        "cache.json.",
        "a" * 256,
        "😀" * 128,
    ],
)
def test_windows_output_name_rejects_device_aliases_and_streams(name: str) -> None:
    with pytest.raises(secure_fs.SecurePathError, match=r"Destination has an unsafe Windows file name"):
        secure_fs._validate_windows_path_component(name, label="Destination")


@pytest.mark.parametrize("name", ["cache.json", "a" * 255, "😀" * 127])
def test_windows_output_name_accepts_valid_components(name: str) -> None:
    secure_fs._validate_windows_path_component(name, label="Destination")


@pytest.mark.parametrize(
    "name",
    [
        ".",
        "nested.",
        "nested ",
        "CON.txt",
        "COM¹.log",
        "cache.json:stream",
        "bad\\child",
        "bad|child",
        "bad\x1fchild",
        "bad\ud800child",
        "a" * 256,
    ],
)
def test_windows_relative_handle_rejects_normalization_hazards_before_native_open(name: str) -> None:
    with pytest.raises(secure_fs.SecurePathError, match=r"unsafe Windows file name"):
        secure_fs._windows_open_relative_handle(
            123,
            name,
            access=secure_fs._WINDOWS_FILE_READ_ACCESS,
            share=secure_fs._WINDOWS_SHARE_READ_WRITE,
            disposition=secure_fs._WINDOWS_FILE_OPEN,
            file_attributes=0,
            create_options=secure_fs._WINDOWS_FILE_OPEN_OPTIONS,
        )


def test_windows_handle_phase_allows_expected_payload_size_change(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = secure_fs._WindowsHandleMetadata(
        attributes=0,
        volume_serial=7,
        file_id=11,
        size=0,
        link_count=1,
    )
    prepared = secure_fs._WindowsHandleMetadata(
        attributes=0,
        volume_serial=7,
        file_id=11,
        size=12,
        link_count=1,
    )
    monkeypatch.setattr(secure_fs, "_windows_handle_metadata", lambda _handle: prepared)

    assert secure_fs._validate_windows_regular_handle(123, expected=empty, expected_size=12) == prepared


def test_windows_reader_contract_anchors_each_component_without_delete_sharing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "catalog" / "skill"
    root.mkdir(parents=True)
    relative_calls: list[dict[str, int | str]] = []
    absolute_calls: list[dict[str, int | Path]] = []
    next_handle = 100

    def open_absolute(path: Path, **kwargs: int) -> int:
        nonlocal next_handle
        next_handle += 1
        absolute_calls.append({"path": path, **kwargs})
        return next_handle

    def open_relative(parent_handle: int, name: str, **kwargs: int) -> int:
        nonlocal next_handle
        next_handle += 1
        relative_calls.append({"parent_handle": parent_handle, "name": name, **kwargs})
        return next_handle

    monkeypatch.setattr(secure_fs, "_windows_open_handle", open_absolute)
    monkeypatch.setattr(secure_fs, "_windows_open_relative_handle", open_relative)
    monkeypatch.setattr(secure_fs, "_validate_windows_read_directory_handle", lambda *_args: None)
    monkeypatch.setattr(
        secure_fs,
        "_verify_windows_handle_path",
        lambda *_args: pytest.fail("Reader authorization must not depend on final-path strings"),
    )

    handles = secure_fs._windows_open_anchored_directory_chain(root, expected=root.lstat())

    assert len(handles) == 1 + len(root.absolute().parts[1:])
    assert absolute_calls[0]["path"] == Path(root.absolute().anchor)
    assert absolute_calls[0]["share"] & 0x4 == 0  # FILE_SHARE_DELETE is absent
    assert absolute_calls[0]["flags"] & secure_fs._WINDOWS_FILE_OPEN_REPARSE_POINT
    assert secure_fs._WINDOWS_OBJECT_ATTRIBUTES_FLAGS & secure_fs._WINDOWS_OBJ_DONT_REPARSE
    assert relative_calls
    assert all(call["share"] & 0x4 == 0 for call in relative_calls)
    assert all(call["create_options"] & secure_fs._WINDOWS_FILE_OPEN_REPARSE_POINT for call in relative_calls)
    assert all(call["create_options"] & secure_fs._WINDOWS_FILE_DIRECTORY_FILE for call in relative_calls)


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


def test_windows_writer_stage_preserves_exclusive_relative_create_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int | str] = {}

    def open_relative(parent_handle: int, name: str, **kwargs: int) -> int:
        captured.update({"parent_handle": parent_handle, "name": name, **kwargs})
        return 456

    monkeypatch.setattr(secure_fs, "_windows_open_relative_handle", open_relative)

    assert secure_fs._windows_create_relative_file(123, ".skillevaluator-safe.tmp", access=789) == 456
    assert captured["parent_handle"] == 123
    assert captured["access"] == 789
    assert captured["share"] == 0
    assert captured["disposition"] == secure_fs._WINDOWS_FILE_CREATE
    assert captured["file_attributes"] == secure_fs._WINDOWS_FILE_ATTRIBUTE_NORMAL
    assert captured["create_options"] & secure_fs._WINDOWS_FILE_OPEN_REPARSE_POINT
    assert captured["create_options"] & secure_fs._WINDOWS_FILE_NON_DIRECTORY_FILE
    assert captured["create_options"] & 0x2  # FILE_WRITE_THROUGH


def test_windows_destination_snapshot_detects_concurrent_change(tmp_path: Path) -> None:
    destination = tmp_path / "cache.json"
    destination.write_text("one", encoding="utf-8")
    before = destination.lstat()

    secure_fs._validate_windows_destination_unchanged(before, before)
    destination.write_text("different-size", encoding="utf-8")

    with pytest.raises(secure_fs.SecurePathError, match=r"destination changed"):
        secure_fs._validate_windows_destination_unchanged(before, destination.lstat())
    with pytest.raises(secure_fs.SecurePathError, match=r"appeared or disappeared"):
        secure_fs._validate_windows_destination_unchanged(None, destination.lstat())


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handle APIs")
def test_windows_secure_root_reads_nested_selected_file_and_pins_root(tmp_path: Path) -> None:
    root = tmp_path / "skill"
    selected_path = root / "references" / "guide.md"
    selected_path.parent.mkdir(parents=True)
    selected_path.write_text("anchored content", encoding="utf-8")

    with secure_fs.SecureRoot(root) as secure_root:
        with pytest.raises(OSError):
            root.rename(tmp_path / "swapped-skill")
        content = secure_root.read_text(
            Path("references/guide.md"),
            1024,
            expected=selected_path.lstat(),
        )

    assert content == "anchored content"


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handle APIs")
def test_windows_secure_root_rejects_reparse_component(tmp_path: Path) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "guide.md").write_text("outside", encoding="utf-8")
    linked = root / "references"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Windows symlink privilege unavailable: {exc}")

    with (
        secure_fs.SecureRoot(root) as secure_root,
        pytest.raises(secure_fs.SecurePathError, match=r"reparse|unsafe"),
    ):
        secure_root.read_text(Path("references/guide.md"), 1024)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handle APIs")
def test_windows_secure_root_rejects_hardlinked_selected_file(tmp_path: Path) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    selected_path = root / "SKILL.md"
    selected_path.write_text("selected", encoding="utf-8")
    os.link(selected_path, tmp_path / "second-name.md")

    with (
        secure_fs.SecureRoot(root) as secure_root,
        pytest.raises(secure_fs.SecurePathError, match=r"hard.?link|link count"),
    ):
        secure_root.read_text(Path("SKILL.md"), 1024, expected=selected_path.lstat())


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


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handle APIs")
def test_windows_atomic_write_creates_and_replaces_cache(tmp_path: Path) -> None:
    destination = tmp_path / "cache.json"

    secure_fs._atomic_write_windows(destination, b'{"version": 1}')
    assert destination.read_bytes() == b'{"version": 1}'

    secure_fs._atomic_write_windows(destination, b'{"version": 2}')
    assert destination.read_bytes() == b'{"version": 2}'
    assert destination.stat().st_nlink == 1


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handle APIs")
def test_windows_atomic_write_rejects_hardlinked_destination(tmp_path: Path) -> None:
    destination = tmp_path / "cache.json"
    destination.write_text("original", encoding="utf-8")
    os.link(destination, tmp_path / "other-link.json")

    with pytest.raises(secure_fs.SecurePathError, match=r"hard.?link|link count"):
        secure_fs._atomic_write_windows(destination, b'{"safe": true}')

    assert destination.read_text(encoding="utf-8") == "original"


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handle APIs")
def test_windows_atomic_write_rejects_linked_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Windows symlink privilege unavailable: {exc}")

    with pytest.raises(secure_fs.SecurePathError, match=r"symlink|junction|reparse"):
        secure_fs._atomic_write_windows(linked_parent / "cache.json", b'{"safe": true}')

    assert not (real_parent / "cache.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handle APIs")
def test_windows_atomic_write_cleans_unpublished_stage_by_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "cache.json"

    def fail_write(_descriptor: int, _payload: bytes) -> int:
        raise OSError("injected write failure")

    monkeypatch.setattr(secure_fs.os, "write", fail_write)

    with pytest.raises(secure_fs.SecurePathError, match=r"injected write failure"):
        secure_fs._atomic_write_windows(destination, b'{"safe": true}')

    assert not destination.exists()
    assert list(tmp_path.glob(".skillevaluator-*.tmp")) == []


def test_windows_structures_are_defined_once_per_process() -> None:
    assert secure_fs._windows_types() is secure_fs._windows_types()


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


def _run_race(check: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", _RACE_PRELUDE + check], capture_output=True, text=True, check=True, timeout=120
    )
    return completed.stdout.strip()


def test_windows_structures_are_defined_once_when_two_threads_race() -> None:
    """Regression: two threads that both missed the cache defined two structure sets.

    The native API binds its prototypes to the set it sees, so a cache that kept
    the other set failed every later native call with ctypes.ArgumentError.
    """
    check = "results = race(secure_fs._windows_types)\nprint(all(r is secure_fs._windows_types() for r in results))\n"
    assert _run_race(check) == "True"


@pytest.mark.skipif(os.name != "nt", reason="the native API binds only on Windows")
def test_windows_api_is_bound_to_the_cached_structures_when_two_threads_race() -> None:
    check = (
        "results = race(secure_fs._windows_api)\n"
        "api, types = secure_fs._windows_api(), secure_fs._windows_types()\n"
        "bound = api.get_file_information.argtypes[1]._type_ is types.ByHandleFileInformation\n"
        "print(bound and all(r is api for r in results))\n"
    )
    assert _run_race(check) == "True"


@pytest.mark.skipif(os.name == "nt", reason="the native API binds only on Windows")
def test_windows_native_api_is_unavailable_off_windows() -> None:
    with pytest.raises(OSError, match="unavailable"):
        secure_fs._windows_api()
    with pytest.raises(OSError, match="unavailable"):
        secure_fs.windows_final_path(0)


def test_windows_relative_open_reuses_structures_and_keeps_its_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def nt_create_file(handle_ref, access, attributes_ref, io_status_ref, *arguments):
        _allocation, file_attributes, share, disposition, create_options, _ea_buffer, _ea_length = arguments
        attributes = attributes_ref._obj
        calls.append(
            {
                "access": access,
                "file_attributes": file_attributes,
                "share": share,
                "disposition": disposition,
                "create_options": create_options,
                "root": attributes.RootDirectory,
                "object_flags": attributes.Attributes,
                "name": attributes.ObjectName.contents.Buffer,
                "types": (type(attributes), type(attributes.ObjectName.contents), type(io_status_ref._obj)),
            }
        )
        handle_ref._obj.value = 456
        return 0

    monkeypatch.setattr(secure_fs, "_windows_api", lambda: _fake_windows_api(nt_create_file=nt_create_file))

    def open_component() -> int:
        return secure_fs._windows_open_relative_handle(
            123,
            "SKILL.md",
            access=secure_fs._WINDOWS_FILE_READ_ACCESS,
            share=secure_fs._WINDOWS_SHARE_READ,
            disposition=secure_fs._WINDOWS_FILE_OPEN,
            file_attributes=0,
            create_options=secure_fs._WINDOWS_FILE_OPEN_OPTIONS,
        )

    assert open_component() == 456
    # ctypes.POINTER caches each structure class for the life of the process,
    # so repeated opens must not define new classes.
    pointer_cache = getattr(ctypes, "_pointer_type_cache", None)
    cached_pointer_types = None if pointer_cache is None else len(pointer_cache)
    for _ in range(50):
        assert open_component() == 456

    if pointer_cache is not None:
        assert len(pointer_cache) == cached_pointer_types
    assert all(call == calls[0] for call in calls)
    types = secure_fs._windows_types()
    assert calls[0] == {
        "access": secure_fs._WINDOWS_FILE_READ_ACCESS,
        "file_attributes": 0,
        "share": secure_fs._WINDOWS_SHARE_READ,
        "disposition": secure_fs._WINDOWS_FILE_OPEN,
        "create_options": secure_fs._WINDOWS_FILE_OPEN_OPTIONS,
        "root": 123,
        "object_flags": secure_fs._WINDOWS_OBJECT_ATTRIBUTES_FLAGS,
        "name": "SKILL.md",
        "types": (types.ObjectAttributes, types.UnicodeString, types.IoStatusBlock),
    }


def test_windows_relative_open_reports_the_ntstatus_as_a_win32_error(monkeypatch: pytest.MonkeyPatch) -> None:
    status_object_name_not_found = -1073741772  # 0xC0000034 as a signed NTSTATUS
    monkeypatch.setattr(
        secure_fs,
        "_windows_api",
        lambda: _fake_windows_api(
            nt_create_file=lambda *_args: status_object_name_not_found,
            rtl_nt_status_to_dos_error=lambda status: 2 if status == status_object_name_not_found else 0,
        ),
    )

    with pytest.raises(OSError, match="missing") as caught:
        secure_fs._windows_open_relative_handle(
            123,
            "missing",
            access=secure_fs._WINDOWS_DIRECTORY_READ_ACCESS,
            share=secure_fs._WINDOWS_SHARE_READ_WRITE,
            disposition=secure_fs._WINDOWS_FILE_OPEN,
            file_attributes=0,
            create_options=secure_fs._WINDOWS_DIRECTORY_OPEN_OPTIONS,
        )

    assert caught.value.errno == 2


def test_windows_handle_metadata_reads_identity_size_links_and_write_time(monkeypatch: pytest.MonkeyPatch) -> None:
    def get_file_information(handle, information_ref) -> int:
        assert handle == 9
        information = information_ref._obj
        assert type(information) is secure_fs._windows_types().ByHandleFileInformation
        information.dwFileAttributes = 0x10
        information.dwVolumeSerialNumber = 7
        information.nFileIndexHigh, information.nFileIndexLow = 1, 2
        information.nFileSizeHigh, information.nFileSizeLow = 3, 4
        information.nNumberOfLinks = 1
        information.ftLastWriteTime.dwHighDateTime, information.ftLastWriteTime.dwLowDateTime = 5, 6
        return 1

    monkeypatch.setattr(secure_fs, "_windows_api", lambda: _fake_windows_api(get_file_information=get_file_information))

    assert secure_fs._windows_handle_metadata(9) == secure_fs._WindowsHandleMetadata(
        attributes=0x10,
        volume_serial=7,
        file_id=(1 << 32) | 2,
        size=(3 << 32) | 4,
        link_count=1,
        last_write_time=(5 << 32) | 6,
    )


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("\\\\?\\C:\\skills\\demo", "C:\\skills\\demo"),
        ("\\\\?\\UNC\\server\\share\\demo", "\\\\server\\share\\demo"),
        ("C:\\skills\\demo", "C:\\skills\\demo"),
    ],
)
def test_windows_final_path_drops_the_extended_length_prefix(
    monkeypatch: pytest.MonkeyPatch, native: str, expected: str
) -> None:
    def get_final_path(handle, buffer, size, flags) -> int:
        assert (handle, size, flags) == (5, len(buffer), 0)
        buffer.value = native
        return len(native)

    monkeypatch.setattr(secure_fs, "_windows_api", lambda: _fake_windows_api(get_final_path=get_final_path))

    assert str(secure_fs._windows_final_path_from_handle(5)) == expected


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


def test_windows_native_values_match_the_win32_definitions() -> None:
    # FILE_READ_ATTRIBUTES | FILE_TRAVERSE | SYNCHRONIZE for directory handles.
    assert secure_fs._WINDOWS_DIRECTORY_READ_ACCESS == 0x80 | 0x20 | 0x100000
    # FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT for CreateFileW.
    assert secure_fs._WINDOWS_DIRECTORY_HANDLE_FLAGS == 0x02000000 | 0x00200000
    assert secure_fs._WINDOWS_OPEN_EXISTING == 3
    # GENERIC_WRITE | FILE_READ_ATTRIBUTES | DELETE | SYNCHRONIZE for the writer's stage file.
    stage_access = (
        secure_fs._WINDOWS_GENERIC_WRITE
        | secure_fs._WINDOWS_FILE_READ_ATTRIBUTES
        | secure_fs._WINDOWS_DELETE
        | secure_fs._WINDOWS_SYNCHRONIZE
    )
    assert stage_access == 0x40000000 | 0x80 | 0x10000 | 0x100000
    # FILE_SHARE_DELETE (0x4) is never granted.
    assert secure_fs._WINDOWS_SHARE_READ_WRITE == 0x1 | 0x2
    assert secure_fs._WINDOWS_FILE_RENAME_INFORMATION == 10
    assert secure_fs._WINDOWS_FILE_DISPOSITION_INFO == 4
    assert sorted(secure_fs._WINDOWS_FILE_EXISTS_ERRORS) == [80, 183]


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


def test_windows_output_parents_must_be_real_directories(tmp_path: Path) -> None:
    root = _tree_with_link(tmp_path)

    secure_fs._validate_windows_parent_components(root / "skills" / "demo" / "cache.json")
    with pytest.raises(secure_fs.SecurePathError, match=r"non-directory component: linked"):
        secure_fs._validate_windows_parent_components(root / "linked" / "cache.json")
    with pytest.raises(secure_fs.SecurePathError, match=r"non-directory component: README.md"):
        secure_fs._validate_windows_parent_components(root / "README.md" / "nested" / "cache.json")
    with pytest.raises(secure_fs.SecurePathError, match=r"non-directory component: README.md"):
        secure_fs._validate_windows_parent_components(root / "README.md" / "cache.json")
    with pytest.raises(secure_fs.SecurePathError) as missing:
        secure_fs._validate_windows_parent_components(root / "absent" / "cache.json")
    assert missing.value.code == "path_access_error"
