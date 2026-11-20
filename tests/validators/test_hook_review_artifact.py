# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review evidence binds the complete packaged tree to the scanner's snapshot."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from skillevaluator.validators import hook_review_artifact
from skillevaluator.validators.hook_review_artifact import capture_hook_review_artifact


def test_capture_copies_and_hashes_the_same_complete_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "results").mkdir()
    (source / "empty").mkdir()
    (source / "results" / "data.txt").write_bytes(b"data\x00\xff")
    (source / ".env.example").write_bytes(b"MODE=checked\n")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    captured = capture_hook_review_artifact(source, snapshot)

    assert captured.hashes == {
        ".env.example": hashlib.sha256(b"MODE=checked\n").hexdigest(),
        "results/data.txt": hashlib.sha256(b"data\x00\xff").hexdigest(),
    }
    assert (snapshot / "results" / "data.txt").read_bytes() == b"data\x00\xff"
    assert (snapshot / "empty").is_dir()
    assert capture_hook_review_artifact(snapshot) == captured


@pytest.mark.parametrize("change", ["contents", "name", "empty directory", "mode"])
def test_artifact_digest_changes_with_the_packaged_tree(tmp_path: Path, change: str) -> None:
    if change == "mode" and os.name != "posix":
        pytest.skip("executable-bit changes require POSIX mode semantics")
    file = tmp_path / "hook"
    file.write_bytes(b"original\n")
    before = capture_hook_review_artifact(tmp_path)

    if change == "contents":
        file.write_bytes(b"modified\n")
    elif change == "name":
        file.rename(tmp_path / "other")
    elif change == "empty directory":
        (tmp_path / "config").mkdir()
    else:
        file.chmod(stat.S_IMODE(file.stat().st_mode) | stat.S_IXUSR)

    assert capture_hook_review_artifact(tmp_path).artifact_sha256 != before.artifact_sha256


@pytest.mark.parametrize("link_kind", ["file", "directory", "hardlink"])
def test_capture_refuses_redirects_and_hardlinks(tmp_path: Path, link_kind: str) -> None:
    source = tmp_path / "package"
    source.mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    (external / "file").write_bytes(b"outside")
    alias = source / "alias"
    try:
        if link_kind == "hardlink":
            alias.hardlink_to(external / "file")
        elif link_kind == "directory":
            alias.symlink_to(external, target_is_directory=True)
        else:
            alias.symlink_to(external / "file")
    except OSError as exc:
        pytest.skip(f"this filesystem cannot create a {link_kind}: {exc}")

    with pytest.raises(ValueError):
        capture_hook_review_artifact(source)


@pytest.mark.parametrize("name", [".git", ".GIT"])
def test_capture_refuses_checkout_metadata(tmp_path: Path, name: str) -> None:
    (tmp_path / name).mkdir()

    with pytest.raises(ValueError, match="Git metadata"):
        capture_hook_review_artifact(tmp_path)


def test_capture_refuses_an_oversized_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_review_artifact, "_MAX_ARTIFACT_BYTES", 4)
    (tmp_path / "a").write_bytes(b"123")
    (tmp_path / "b").write_bytes(b"45")

    with pytest.raises(ValueError):
        capture_hook_review_artifact(tmp_path)


def test_capture_refuses_a_missing_package(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        capture_hook_review_artifact(tmp_path / "missing")


def test_capture_refuses_an_overlong_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "some-file").touch()
    monkeypatch.setattr(hook_review_artifact, "_MAX_PATH_BYTES", 5, raising=False)

    with pytest.raises(ValueError, match="canonical"):
        capture_hook_review_artifact(tmp_path)


def test_capture_detects_a_new_file_added_during_byte_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "hook").write_bytes(b"original")
    read_bytes = hook_review_artifact.SecureRoot.read_bytes

    def read_and_modify(reader, *args, **kwargs):
        result = read_bytes(reader, *args, **kwargs)
        (tmp_path / "new-runnable").write_bytes(b"new file")
        return result

    monkeypatch.setattr(hook_review_artifact.SecureRoot, "read_bytes", read_and_modify)

    with pytest.raises(ValueError, match="changed"):
        capture_hook_review_artifact(tmp_path)


def test_capture_refuses_destination_inside_the_package(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(ValueError):
        capture_hook_review_artifact(tmp_path, output)
