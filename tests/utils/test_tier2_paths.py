# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator.utils.tier2_paths import contained_directory_alias_target, is_link_or_reparse


def _directory_alias(root: Path, relative: str, target: str) -> Path:
    alias = root / relative
    alias.parent.mkdir(parents=True, exist_ok=True)
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    return alias


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


@pytest.mark.parametrize(
    ("relative", "target", "expected"),
    [
        pytest.param(".agents", ".claude", ".claude", id="sibling"),
        pytest.param(".claude/skills", "../shared-skills", "shared-skills", id="leading-parent"),
        pytest.param("nested/.agents", "./.claude/", "nested/.claude", id="current-directory"),
    ],
)
def test_contained_directory_alias_target_returns_lexical_in_root_claim(
    tmp_path: Path, relative: str, target: str, expected: str
) -> None:
    alias = _directory_alias(tmp_path, relative, target)

    assert contained_directory_alias_target(alias, Path(relative)) == expected


@pytest.mark.parametrize(
    ("relative", "target"),
    [
        pytest.param(".agents", ".", id="root"),
        pytest.param(".agents", "..", id="parent-of-root"),
        pytest.param(".agents", "../outside", id="escape"),
        pytest.param("nested/.agents", "../../outside", id="nested-escape"),
        pytest.param("nested/deeper/.agents", "..", id="own-parent"),
        pytest.param("nested/.agents", "../nested", id="own-parent-by-name"),
        pytest.param(".agents", "linked/../.claude", id="parent-after-named-component"),
    ],
)
def test_contained_directory_alias_target_rejects_root_escape_and_cycle_claims(
    tmp_path: Path, relative: str, target: str
) -> None:
    alias = _directory_alias(tmp_path, relative, target)

    assert contained_directory_alias_target(alias, Path(relative)) is None


def test_contained_directory_alias_target_rejects_absolute_link_text(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    alias = _directory_alias(tmp_path, ".agents", str(tmp_path / ".claude"))

    assert contained_directory_alias_target(alias, Path(".agents")) is None


def test_contained_directory_alias_target_rejects_unreadable_link_text(tmp_path: Path) -> None:
    regular = tmp_path / "regular"
    regular.mkdir()

    assert contained_directory_alias_target(regular, Path("regular")) is None


def test_contained_directory_alias_target_rejects_hard_linked_alias(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    alias = _directory_alias(tmp_path, ".agents", ".claude")
    try:
        os.link(alias, tmp_path / ".agents-copy", follow_symlinks=False)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"hard links to symlinks unavailable: {exc}")

    assert contained_directory_alias_target(alias, Path(".agents")) is None
