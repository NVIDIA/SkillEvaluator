# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from skillevaluator.utils import secure_fs
from skillevaluator.utils.secure_fs import SecurePathError


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


def _emulate_windows_final_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = os.open
    opened_paths: dict[int, Path] = {}

    def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened_paths[descriptor] = Path(path)
        return descriptor

    monkeypatch.setattr(secure_fs.os, "open", tracked_open)
    monkeypatch.setattr(secure_fs, "_windows_final_path", lambda descriptor: opened_paths[descriptor])


def test_windows_atomic_write_creates_and_replaces_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "cache.json"
    _emulate_windows_final_paths(monkeypatch)

    secure_fs._atomic_write_windows(destination, b'{"version": 1}')
    assert destination.read_bytes() == b'{"version": 1}'

    secure_fs._atomic_write_windows(destination, b'{"version": 2}')
    assert destination.read_bytes() == b'{"version": 2}'
    assert destination.stat().st_nlink == 1


def test_windows_atomic_write_rejects_linked_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "cache.json"
    target = tmp_path / "target.json"
    target.write_text("do not replace", encoding="utf-8")
    destination.symlink_to(target.name)
    _emulate_windows_final_paths(monkeypatch)

    with pytest.raises(SecurePathError, match=r"symlink|reparse|unsafe"):
        secure_fs._atomic_write_windows(destination, b'{"safe": true}')

    assert target.read_text(encoding="utf-8") == "do not replace"

