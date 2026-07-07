# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace

from skillevaluator.utils.tier2_paths import is_link_or_reparse


def test_windows_reparse_attribute_is_rejected_without_symlink_mode(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "junction"
    target.mkdir()
    original_lstat = Path.lstat

    def fake_lstat(path: Path):
        if path == target:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    assert is_link_or_reparse(target)
