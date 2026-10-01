# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import subprocess
from pathlib import Path

import pytest

from skillevaluator.utils import secure_fs


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
    with pytest.raises(secure_fs.SecurePathError, match=r"unsafe Windows file name"):
        secure_fs._validate_windows_output_name(name)


@pytest.mark.parametrize("name", ["cache.json", "a" * 255, "😀" * 127])
def test_windows_output_name_accepts_valid_components(name: str) -> None:
    secure_fs._validate_windows_output_name(name)


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

    names, stable = secure_fs._windows_enumerate_pinned_directory_names(root, 123, Path(), snapshot)

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

    with pytest.raises(secure_fs.SecurePathError) as caught:
        secure_fs._windows_enumerate_pinned_directory_names(
            root,
            123,
            Path(),
            snapshot,
            max_names=3,
            path_limit=2,
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
