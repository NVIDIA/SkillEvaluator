# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot, lstat_walk, read_bounded


def test_existing_reader_observes_bound_and_file_identity(tmp_path):
    root = tmp_path.resolve()
    file = root / "input.json"
    file.write_bytes(b'{"ok":true}')
    seen = file.stat()
    with SecureRoot(root) as reader:
        assert reader.read_bytes(Path("input.json"), 11, expected=seen)[0] == b'{"ok":true}'
        with pytest.raises(SecurePathError) as error:
            reader.read_bytes(Path("input.json"), 10)
        assert error.value.code == "file_size_limit"
        file.rename(root / "old.json")
        file.write_bytes(b'{"ok":true}')
        with pytest.raises(SecurePathError):
            reader.read_bytes(Path("input.json"), 11, expected=seen)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_existing_reader_refuses_files_that_are_not_single_link_regular(tmp_path, kind):
    root = tmp_path.resolve()
    source = root / "source"
    source.write_text("example")
    path = root / "selected"
    if kind == "symlink":
        path.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, path)
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO creation is unavailable on this platform")
        os.mkfifo(path)
    with SecureRoot(root) as reader, pytest.raises(SecurePathError):
        reader.read_bytes(Path("selected"), 100)


@pytest.mark.parametrize("limit", [0, 3, 8])
def test_shared_descriptor_reader_returns_only_requested_prefix(tmp_path, limit):
    file = tmp_path / "bytes"
    file.write_bytes(b"abcdefgh")
    with file.open("rb", buffering=0) as opened:
        assert read_bounded(opened.fileno(), limit, truncate=True) == b"abcdefgh"[:limit]
        assert opened.tell() == limit
    with file.open("rb", buffering=0) as opened:
        if limit < 8:
            with pytest.raises(SecurePathError) as error:
                read_bounded(opened.fileno(), limit)
            assert error.value.code == "file_size_limit"
            assert opened.tell() == limit + 1
        else:
            assert read_bounded(opened.fileno(), limit) == b"abcdefgh"


def test_no_follow_probe_stops_before_link_descendants(tmp_path):
    root = tmp_path.resolve()
    (root / "dir").mkdir()
    (root / "dir" / "file").write_text("example")
    (root / "link").symlink_to(root / "dir", target_is_directory=True)
    assert lstat_walk(root, Path("dir/file")).outcome == "ok"
    assert lstat_walk(root, Path("link/file")).outcome == "link"
    assert lstat_walk(root, Path("missing/file")).outcome == "missing"
    assert lstat_walk(root, Path("dir/file/child")).outcome == "not_dir"
