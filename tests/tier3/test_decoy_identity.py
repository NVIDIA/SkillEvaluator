# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime identity must match the exact decoy content copied to agents."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from skillevaluator.publication_identity import publication_source_digest
from skillevaluator.tier3.decoy_identity import decoy_source_copy_ignore, decoy_source_digest
from skillevaluator.tier3.harbor import secure_copy


def _skill(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"# {path.name}\nRuntime instructions.\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("filename", ["evals/config.yml", "evals/evals.json", "evals/decoys.json"])
def test_evaluator_edits_leave_runtime_identity_unchanged(tmp_path: Path, filename: str) -> None:
    source = _skill(tmp_path / "decoy")
    runtime_before = decoy_source_digest(source)
    publication_before = publication_source_digest(source)
    assert runtime_before is not None
    evaluator_file = source / filename
    evaluator_file.parent.mkdir(exist_ok=True)
    evaluator_file.write_text("new evaluator input\n", encoding="utf-8")

    assert decoy_source_digest(source) == runtime_before
    assert publication_source_digest(source) != publication_before
    evaluator_file.write_text("changed evaluator input\n", encoding="utf-8")
    assert decoy_source_digest(source) == runtime_before


def test_nested_skill_evals_are_excluded_but_reference_evals_are_runtime(tmp_path: Path) -> None:
    source = _skill(tmp_path / "decoy")
    nested = _skill(source / "references" / "nested-skill")
    before = decoy_source_digest(source)
    assert before is not None
    oracle = nested / "evals" / "oracle.txt"
    oracle.parent.mkdir()
    oracle.write_text("private evaluator data\n", encoding="utf-8")

    assert decoy_source_digest(source) == before
    ordinary = source / "references" / "evals" / "guide.txt"
    ordinary.parent.mkdir()
    ordinary.write_text("agent-visible reference\n", encoding="utf-8")
    assert decoy_source_digest(source) != before


@pytest.mark.parametrize("filename", ["SKILL.md", "scripts/helper.py", "references/guide.md"])
def test_runtime_edits_change_identity(tmp_path: Path, filename: str) -> None:
    source = _skill(tmp_path / "decoy")
    before = decoy_source_digest(source)
    assert before is not None
    runtime_file = source / filename
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text("changed runtime input\n", encoding="utf-8")

    assert decoy_source_digest(source) != before


@pytest.mark.skipif(os.name == "nt", reason="Windows does not preserve POSIX executable bits")
def test_executable_bits_change_runtime_identity(tmp_path: Path) -> None:
    source = _skill(tmp_path / "decoy")
    script = source / "helper.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o644)
    before = decoy_source_digest(source)
    assert before is not None
    script.chmod(0o755)

    assert decoy_source_digest(source) != before


@pytest.mark.skipif(os.name == "nt", reason="Windows does not preserve POSIX permission bits")
@pytest.mark.parametrize("umask", [0o027, 0o077])
def test_clone_umask_leaves_runtime_identity_unchanged(tmp_path: Path, umask: int) -> None:
    source = _skill(tmp_path / "decoy")
    scripts = source / "scripts"
    scripts.mkdir()
    script = scripts / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    modes = {source: 0o755, scripts: 0o755, script: 0o755, source / "SKILL.md": 0o644}
    for path, mode in modes.items():
        path.chmod(mode)
    before = decoy_source_digest(source)
    assert before is not None

    # A clone made under another umask differs only in group and other bits.
    for path, mode in modes.items():
        path.chmod(mode & ~umask)
    assert decoy_source_digest(source) == before


def test_secure_runtime_snapshot_has_the_same_identity(tmp_path: Path) -> None:
    source = _skill(tmp_path / "decoy")
    nested = _skill(source / "references" / "nested")
    for owner in (source, nested):
        (owner / "evals").mkdir()
        (owner / "evals" / "oracle.txt").write_text("private\n", encoding="utf-8")
    ordinary = source / "references" / "evals"
    ordinary.mkdir()
    (ordinary / "guide.txt").write_text("runtime guide\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    before = decoy_source_digest(source)
    assert before is not None

    secure_copy.copytree_secure(
        source,
        snapshot,
        allowed_root=source,
        ignore=decoy_source_copy_ignore(source),
    )

    assert decoy_source_digest(snapshot) == before
    assert not (snapshot / "evals").exists()
    assert not (snapshot / "references" / "nested" / "evals").exists()
    assert (snapshot / "references" / "evals" / "guide.txt").read_text() == "runtime guide\n"


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "root-symlink", "fifo"])
def test_unsafe_runtime_sources_fail_closed(tmp_path: Path, unsafe: str) -> None:
    source = _skill(tmp_path / "decoy")
    if unsafe == "symlink":
        (source / "linked.md").symlink_to(source / "SKILL.md")
    elif unsafe == "hardlink":
        (source / "linked.md").hardlink_to(source / "SKILL.md")
    elif unsafe == "root-symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        source = alias
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs are unavailable")
        os.mkfifo(source / "pipe")

    assert decoy_source_digest(source) is None


def test_normalized_path_collisions_fail_closed(tmp_path: Path) -> None:
    source = _skill(tmp_path / "decoy")
    first = source / "caf\u00e9.md"
    second = source / "cafe\u0301.md"
    first.write_text("first\n", encoding="utf-8")
    if second.exists():
        pytest.skip("Filesystem normalizes Unicode path names")
    second.write_text("second\n", encoding="utf-8")

    assert decoy_source_digest(source) is None


@pytest.mark.parametrize("after_capture", [1, 2])
def test_changes_between_stability_passes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_capture: int
) -> None:
    source = _skill(tmp_path / "decoy")
    capture = secure_copy._build_tree_manifest
    count = 0

    def changing_capture(*args, **kwargs):
        nonlocal count
        manifest = capture(*args, **kwargs)
        count += 1
        if count == after_capture:
            (source / "SKILL.md").write_text("changed after capture\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(secure_copy, "_build_tree_manifest", changing_capture)

    assert decoy_source_digest(source) is None


def test_checked_fallback_retains_runtime_projection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _skill(tmp_path / "decoy")
    (source / "evals").mkdir()
    (source / "evals" / "config.yml").write_text("private\n", encoding="utf-8")
    expected = decoy_source_digest(source)
    assert expected is not None
    monkeypatch.setattr(secure_copy, "_DESCRIPTOR_BACKEND", False)

    assert decoy_source_digest(source) == expected
    (source / "linked.md").symlink_to(source / "SKILL.md")
    assert decoy_source_digest(source) is None
