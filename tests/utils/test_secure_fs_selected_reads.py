# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for descriptor-anchored evaluator artifact reads."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator.utils import secure_fs
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot


def test_secure_root_binds_declared_root_to_expected_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    expected = root.lstat()
    root.rename(tmp_path / "original-root")
    root.mkdir()

    with pytest.raises(SecurePathError, match=r"changed|snapshot|identity"), SecureRoot(root, expected=expected):
        pass


def test_windows_selected_file_handle_denies_concurrent_write_and_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "summary.json"
    selected.write_text("{}", encoding="utf-8")
    root = SecureRoot(tmp_path)
    root._windows_root_handles = [100]
    captured: dict[str, int | str] = {}
    opened_flags: list[int] = []

    def open_relative(parent_handle: int, name: str, **kwargs: int) -> int:
        captured.update({"parent_handle": parent_handle, "name": name, **kwargs})
        return 200

    monkeypatch.setattr(secure_fs, "_windows_open_relative_handle", open_relative)
    monkeypatch.setattr(secure_fs, "_validate_windows_read_file_handle", lambda *_args: None)
    monkeypatch.setattr(secure_fs.os, "fstat", lambda _descriptor: selected.lstat())
    monkeypatch.setattr(secure_fs.os, "O_BINARY", 0x8000, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(open_osfhandle=lambda _handle, flags: opened_flags.append(flags) or 300),
    )

    assert root._open_windows(Path("summary.json"), None) == 300
    assert captured["share"] == 0x1  # FILE_SHARE_READ only
    assert int(opened_flags[0]) & 0x8000  # O_BINARY


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows sharing semantics")
def test_windows_selected_file_is_immutable_until_reader_handle_closes(tmp_path: Path) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    selected = root / "summary.json"
    selected.write_text("original", encoding="utf-8")

    with SecureRoot(root) as secure_root:
        descriptor = secure_root._open_windows(Path("summary.json"), selected.lstat())
        try:
            with pytest.raises(OSError):
                selected.write_text("changed", encoding="utf-8")
            with pytest.raises(OSError):
                selected.unlink()
        finally:
            os.close(descriptor)

    selected.write_text("changed", encoding="utf-8")
    selected.unlink()
    assert not selected.exists()
