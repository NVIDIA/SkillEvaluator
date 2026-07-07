# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security regression tests for Harbor filesystem staging."""

from __future__ import annotations

import ctypes
import errno
import importlib
import os
import socket
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.adapter import _copy_skill_dirs, _write_dockerfile

_LINK_REJECTION = r"symlink|reparse|outside the staging root|without descriptor no-follow support"


def _copytree_secure(source: Path, destination: Path, **kwargs: object) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    module.copytree_secure(source, destination, **kwargs)


def _copy_file_secure(source: Path, destination: Path, **kwargs: object) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    module.copy_file_secure(source, destination, **kwargs)


def test_public_secure_copy_api_documents_same_uid_boundary() -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")

    assert "same UID" in module.__doc__
    assert "outside" in module.copytree_secure.__doc__
    assert "same UID" in module.copytree_secure.__doc__
    assert "outside" in module.copy_file_secure.__doc__
    assert "same UID" in module.copy_file_secure.__doc__


def test_external_file_symlink_is_rejected_before_copy(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("host secret", encoding="utf-8")
    source = tmp_path / "skill"
    source.mkdir()
    (source / "escape.txt").symlink_to(outside)
    destination = tmp_path / "staged"

    with pytest.raises(ValueError, match=_LINK_REJECTION):
        _copytree_secure(source, destination)

    assert not destination.exists()


def test_broken_symlink_is_rejected_before_copy(tmp_path: Path) -> None:
    source = tmp_path / "skill"
    source.mkdir()
    (source / "broken.txt").symlink_to(source / "missing.txt")

    with pytest.raises(ValueError, match=_LINK_REJECTION):
        _copytree_secure(source, tmp_path / "staged")


def test_internal_file_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "skill"
    source.mkdir()
    (source / "target.txt").write_text("public data", encoding="utf-8")
    (source / "alias.txt").symlink_to("target.txt")
    destination = tmp_path / "staged"

    with pytest.raises(ValueError, match=_LINK_REJECTION):
        _copytree_secure(source, destination)

    assert not destination.exists()


def test_internal_directory_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "skill"
    target = source / "references" / "shared"
    target.mkdir(parents=True)
    (target / "guide.md").write_text("guide", encoding="utf-8")
    (source / "linked-reference").symlink_to("references/shared", target_is_directory=True)
    destination = tmp_path / "staged"

    with pytest.raises(ValueError, match=_LINK_REJECTION):
        _copytree_secure(source, destination)

    assert not destination.exists()


def test_directory_symlink_cycle_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "skill"
    child = source / "child"
    child.mkdir(parents=True)
    (child / "back").symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match=_LINK_REJECTION):
        _copytree_secure(source, tmp_path / "staged")


def test_ignored_symlink_is_not_validated_or_copied(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("generated", encoding="utf-8")
    source = tmp_path / "skill"
    generated = source / "results"
    generated.mkdir(parents=True)
    (generated / "latest").symlink_to(outside)

    _copytree_secure(
        source,
        tmp_path / "staged",
        ignore=lambda _directory, names: [name for name in names if name == "results"],
    )

    assert not (tmp_path / "staged" / "results").exists()


def test_top_level_directory_link_must_stay_within_allowed_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    source_root = tmp_path / "skill"
    source_root.mkdir()
    linked = source_root / "sidecar"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=_LINK_REJECTION):
        _copytree_secure(linked, tmp_path / "staged", allowed_root=source_root)


def test_adapter_rejects_skill_symlink_to_outside_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("must not enter task context", encoding="utf-8")
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Safe skill\n", encoding="utf-8")
    (skill / "escape.txt").symlink_to(outside)

    with pytest.raises(ValueError, match=_LINK_REJECTION):
        _copy_skill_dirs(
            env_dir=tmp_path / "task" / "environment",
            skill_path=skill,
            reference_skills_dir=None,
            workspace_skill_paths=None,
            has_skill=True,
            exclude_skill_name=None,
        )

    staged_escape = tmp_path / "task" / "environment" / "skills" / "skill" / "escape.txt"
    assert not staged_escape.exists()


def test_baseline_rejects_external_custom_dockerfile_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.Dockerfile"
    outside.write_text("FROM python:3.12\nCOPY host-secret /tmp/\n", encoding="utf-8")
    skill = tmp_path / "skill"
    custom_environment = skill / "evals" / "environment"
    custom_environment.mkdir(parents=True)
    (custom_environment / "Dockerfile").symlink_to(outside)
    task = tmp_path / "task"

    with pytest.raises(ValueError, match="symlink"):
        _write_dockerfile(
            task_dir=task,
            skill_path=skill,
            reference_skills_dir=None,
            workspace_skill_paths=None,
            has_skill=False,
            exclude_skill_name=skill.name,
        )

    assert not (task / "environment" / "Dockerfile").exists()


def test_top_level_source_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "safe.txt").write_text("safe", encoding="utf-8")
    source = tmp_path / "source"
    source.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _copytree_secure(source, tmp_path / "destination", allowed_root=tmp_path)


def test_hard_linked_source_is_rejected_and_outside_unchanged(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside-canary")
    source = tmp_path / "source"
    source.mkdir()
    os.link(outside, source / "linked.txt")

    with pytest.raises(ValueError, match=r"hard.?link|multiple links"):
        _copytree_secure(source, tmp_path / "destination")

    assert outside.read_bytes() == b"outside-canary"
    assert not (tmp_path / "destination").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_special_source_entry_is_rejected_without_blocking(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "fifo")

    with pytest.raises(ValueError, match="regular file or directory"):
        _copytree_secure(source, tmp_path / "destination")


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
def test_socket_source_entry_is_rejected_without_blocking(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="secure-copy-", dir="/tmp") as temporary:
        source = Path(temporary) / "source"
        source.mkdir()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(source / "socket"))
            with pytest.raises(ValueError, match="regular file or directory"):
                _copytree_secure(source, tmp_path / "destination")
        finally:
            listener.close()


def test_ignored_symlink_subtree_is_opaque(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside-canary", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe", encoding="utf-8")
    (source / "results").symlink_to(outside, target_is_directory=True)
    destination = tmp_path / "destination"

    _copytree_secure(
        source,
        destination,
        ignore=lambda _directory, names: {"results"} & set(names),
    )

    assert (destination / "safe.txt").read_text(encoding="utf-8") == "safe"
    assert not (destination / "results").exists()
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "outside-canary"


def test_allowed_root_escape_and_symlink_are_rejected(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "safe.txt").write_text("safe", encoding="utf-8")

    with pytest.raises(ValueError, match="allowed root"):
        _copytree_secure(outside, tmp_path / "destination", allowed_root=allowed)

    alias = tmp_path / "allowed-alias"
    alias.symlink_to(allowed, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _copytree_secure(allowed, tmp_path / "destination", allowed_root=alias)


def test_file_swap_to_hardlink_after_manifest_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    victim = source / "victim.txt"
    victim.write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-canary", encoding="utf-8")
    original = module._build_tree_manifest

    def scan_then_swap(*args: object, **kwargs: object):
        manifest = original(*args, **kwargs)
        victim.unlink()
        os.link(outside, victim)
        return manifest

    monkeypatch.setattr(module, "_build_tree_manifest", scan_then_swap)

    with pytest.raises(ValueError, match=r"changed|hard.?link"):
        module.copytree_secure(source, tmp_path / "destination")

    assert outside.read_text(encoding="utf-8") == "outside-canary"
    assert not (tmp_path / "destination").exists()


def test_directory_swap_to_external_symlink_after_manifest_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "safe.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside-canary", encoding="utf-8")
    original = module._build_tree_manifest

    def scan_then_swap(*args: object, **kwargs: object):
        manifest = original(*args, **kwargs)
        nested.rename(source / "held")
        nested.symlink_to(outside, target_is_directory=True)
        return manifest

    monkeypatch.setattr(module, "_build_tree_manifest", scan_then_swap)

    with pytest.raises(ValueError, match=r"changed|symlink"):
        module.copytree_secure(source, tmp_path / "destination")

    assert (outside / "secret.txt").read_text(encoding="utf-8") == "outside-canary"
    assert not (tmp_path / "destination").exists()


def test_cross_device_source_entry_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe", encoding="utf-8")
    original = module._source_device

    def fake_device(metadata: os.stat_result, *, path: Path) -> int:
        if path.name == "safe.txt":
            return original(metadata, path=path) + 1
        return original(metadata, path=path)

    monkeypatch.setattr(module, "_source_device", fake_device)

    with pytest.raises(ValueError, match=r"device|mount"):
        module.copytree_secure(source, tmp_path / "destination")


@pytest.mark.parametrize("kind", ["root", "file", "directory", "unrelated"])
def test_merge_rejects_every_destination_symlink_without_mutation(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside-canary", encoding="utf-8")

    if kind == "root":
        destination.symlink_to(outside, target_is_directory=True)
    else:
        destination.mkdir()
        (destination / "old.txt").write_text("old", encoding="utf-8")
        if kind == "file":
            (destination / "new.txt").symlink_to(sentinel)
        elif kind == "directory":
            (destination / "nested").symlink_to(outside, target_is_directory=True)
        else:
            (destination / "unrelated").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=r"destination.*symlink"):
        _copytree_secure(source, destination, dirs_exist_ok=True)

    assert sentinel.read_text(encoding="utf-8") == "outside-canary"
    if kind != "root":
        assert (destination / "old.txt").read_text(encoding="utf-8") == "old"
        assert not (destination / "new.txt").is_file() or (destination / "new.txt").is_symlink()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_merge_rejects_unrelated_special_destination_entry(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    os.mkfifo(destination / "unrelated-fifo")

    with pytest.raises(ValueError, match=r"destination.*regular file or directory"):
        _copytree_secure(source, destination, dirs_exist_ok=True)


def test_merge_rejects_unrelated_destination_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-canary", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    os.link(outside, destination / "unrelated-hardlink.txt")

    with pytest.raises(ValueError, match=r"destination.*hard.?link|destination.*multiple links"):
        _copytree_secure(source, destination, dirs_exist_ok=True)

    assert outside.read_text(encoding="utf-8") == "outside-canary"
    assert not (destination / "new.txt").exists()


def test_merge_late_copy_failure_preserves_complete_old_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    (destination / "nested").mkdir(parents=True)
    (destination / "old.txt").write_bytes(b"old-root")
    (destination / "nested" / "old.txt").write_bytes(b"old-nested")
    original = module._copy_manifest_file

    def fail_after_copy(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        entry = kwargs.get("entry")
        if entry is not None and entry.parts == ("new.txt",):
            raise OSError("injected late copy failure")

    monkeypatch.setattr(module, "_copy_manifest_file", fail_after_copy)

    with pytest.raises(OSError, match="injected"):
        module.copytree_secure(source, destination, dirs_exist_ok=True)

    assert (destination / "old.txt").read_bytes() == b"old-root"
    assert (destination / "nested" / "old.txt").read_bytes() == b"old-nested"
    assert not (destination / "new.txt").exists()


def test_merge_publish_failure_rolls_back_complete_old_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    original = module._rename_no_replace
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        original(*args, **kwargs)

    monkeypatch.setattr(module, "_rename_no_replace", fail_second)

    with pytest.raises(OSError, match="injected"):
        module.copytree_secure(source, destination, dirs_exist_ok=True)

    assert (destination / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (destination / "new.txt").exists()


@pytest.mark.parametrize("force_fallback", [False, True])
def test_exact_replacement_removes_destination_only_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_fallback: bool,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    if force_fallback:
        monkeypatch.setattr(module, "_DESCRIPTOR_BACKEND", False)
        monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")

    module.copytree_secure(source, destination, replace_existing=True)

    assert sorted(path.name for path in destination.iterdir()) == ["new.txt"]
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"


def test_exact_replacement_publish_failure_restores_complete_old_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    original = module._rename_no_replace
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replacement publish failure")
        original(*args, **kwargs)

    monkeypatch.setattr(module, "_rename_no_replace", fail_second)

    with pytest.raises(OSError, match="replacement publish"):
        module.copytree_secure(source, destination, replace_existing=True)

    assert sorted(path.name for path in destination.iterdir()) == ["old.txt"]
    assert (destination / "old.txt").read_text(encoding="utf-8") == "old"


def test_exact_replacement_and_merge_modes_are_mutually_exclusive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="mutually exclusive"):
        _copytree_secure(source, tmp_path / "destination", dirs_exist_ok=True, replace_existing=True)


def test_copy_file_rejects_source_and_destination_links_and_hardlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-canary", encoding="utf-8")
    source_link = tmp_path / "source-link"
    source_link.symlink_to(outside)
    destination = tmp_path / "destination.txt"
    destination.write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        _copy_file_secure(source_link, destination, allowed_root=tmp_path)
    assert destination.read_text(encoding="utf-8") == "old"

    source_hardlink = tmp_path / "source-hardlink.txt"
    os.link(outside, source_hardlink)
    with pytest.raises(ValueError, match=r"hard.?link|multiple links"):
        _copy_file_secure(source_hardlink, destination, allowed_root=tmp_path)
    assert destination.read_text(encoding="utf-8") == "old"

    source = tmp_path / "source.txt"
    source.write_text("new", encoding="utf-8")
    destination.unlink()
    destination.symlink_to(outside)
    with pytest.raises(ValueError, match=r"destination.*symlink"):
        _copy_file_secure(source, destination, allowed_root=tmp_path)

    destination.unlink()
    os.link(outside, destination)
    with pytest.raises(ValueError, match=r"destination.*hard.?link|multiple links"):
        _copy_file_secure(source, destination, allowed_root=tmp_path)
    assert outside.read_text(encoding="utf-8") == "outside-canary"


def test_copy_file_late_verification_failure_restores_old_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source.txt"
    source.write_text("new", encoding="utf-8")
    destination = tmp_path / "destination.txt"
    destination.write_text("old", encoding="utf-8")
    original = module._verify_published_node

    def fail_after_publish(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        raise OSError("injected late verification failure")

    monkeypatch.setattr(module, "_verify_published_node", fail_after_publish)

    with pytest.raises(OSError, match="injected"):
        module.copy_file_secure(source, destination, allowed_root=tmp_path)

    assert destination.read_text(encoding="utf-8") == "old"


def test_regular_copy_preserves_modes_and_merges_transactionally(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    script = source / "nested" / "run.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o751)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "unrelated.txt").write_text("keep", encoding="utf-8")

    _copytree_secure(source, destination, dirs_exist_ok=True)

    assert (destination / "unrelated.txt").read_text(encoding="utf-8") == "keep"
    assert (destination / "nested" / "run.sh").read_text(encoding="utf-8") == "#!/bin/sh\n"
    assert stat.S_IMODE((destination / "nested" / "run.sh").stat().st_mode) == 0o751


def test_private_stage_injection_is_rejected_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe", encoding="utf-8")
    destination = tmp_path / "destination"
    original = module._stage_manifests

    def inject_extra(*args: object, **kwargs: object):
        staged = original(*args, **kwargs)
        descriptor = os.open(
            "injected.txt",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=staged.descriptor,
        )
        os.write(descriptor, b"injected")
        os.close(descriptor)
        return staged

    monkeypatch.setattr(module, "_stage_manifests", inject_extra)

    with pytest.raises(ValueError, match=r"unexpected|missing"):
        module.copytree_secure(source, destination)

    assert not destination.exists()


def test_checked_non_posix_fallback_copies_tree_and_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    monkeypatch.setattr(module, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe", encoding="utf-8")
    tree_destination = tmp_path / "tree-destination"

    module.copytree_secure(source, tree_destination, allowed_root=tmp_path)
    assert (tree_destination / "safe.txt").read_text(encoding="utf-8") == "safe"

    file_source = tmp_path / "file-source.txt"
    file_source.write_text("new", encoding="utf-8")
    file_source.chmod(0o640)
    file_destination = tmp_path / "file-destination.txt"
    file_destination.write_text("old", encoding="utf-8")
    module.copy_file_secure(file_source, file_destination, allowed_root=tmp_path)
    assert file_destination.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(file_destination.stat().st_mode) == 0o640


def test_checked_non_posix_fallback_still_rejects_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    monkeypatch.setattr(module, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-canary", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "escape.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        module.copytree_secure(source, tmp_path / "destination", allowed_root=tmp_path)

    assert outside.read_text(encoding="utf-8") == "outside-canary"


def test_missing_atomic_backend_routes_to_checked_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe", encoding="utf-8")
    destination = tmp_path / "destination"

    module.copytree_secure(source, destination)

    assert (destination / "safe.txt").read_text(encoding="utf-8") == "safe"


def test_post_finalize_tree_injection_restores_old_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    original = module._apply_final_tree_modes
    calls = 0

    def inject_after_modes(descriptor: int, manifests: object) -> None:
        nonlocal calls
        original(descriptor, manifests)
        calls += 1
        if calls == 1:
            injected = os.open(
                "extra.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=descriptor,
            )
            os.write(injected, b"injected")
            os.close(injected)

    monkeypatch.setattr(module, "_apply_final_tree_modes", inject_after_modes)

    with pytest.raises(ValueError, match=r"unexpected|missing"):
        module.copytree_secure(source, destination, dirs_exist_ok=True)

    assert (destination / "old.txt").read_text(encoding="utf-8") == "old"
    assert sorted(path.name for path in destination.iterdir()) == ["old.txt"]


def test_post_finalize_file_mutation_restores_old_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source.txt"
    source.write_text("new", encoding="utf-8")
    destination = tmp_path / "destination.txt"
    destination.write_text("old-original", encoding="utf-8")
    original = module._apply_final_file_mode
    calls = 0

    def mutate_after_mode(descriptor: int, mode: int) -> None:
        nonlocal calls
        original(descriptor, mode)
        calls += 1
        if calls == 1:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"tampered")
            os.ftruncate(descriptor, len(b"tampered"))

    monkeypatch.setattr(module, "_apply_final_file_mode", mutate_after_mode)

    with pytest.raises(ValueError, match=r"contents|size|digest"):
        module.copy_file_secure(source, destination, allowed_root=tmp_path)

    assert destination.read_text(encoding="utf-8") == "old-original"


def test_fallback_post_finalize_injection_restores_old_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    monkeypatch.setattr(module, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    original = module._apply_modes_fallback
    calls = 0

    def inject_after_modes(stage: Path, manifests: object) -> None:
        nonlocal calls
        original(stage, manifests)
        calls += 1
        if calls == 1:
            (stage / "new.txt").write_text("bad", encoding="utf-8")

    monkeypatch.setattr(module, "_apply_modes_fallback", inject_after_modes)

    with pytest.raises(ValueError, match=r"contents|digest"):
        module.copytree_secure(source, destination, dirs_exist_ok=True)

    assert (destination / "old.txt").read_text(encoding="utf-8") == "old"
    assert sorted(path.name for path in destination.iterdir()) == ["old.txt"]


def test_private_rollback_snapshot_ignores_mutated_moved_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "old.txt").write_text("old-original", encoding="utf-8")
    prepare = module._prepare_moved_backup
    expose_root = module._expose_tree_root_exact
    finalize_calls = 0

    def mutate_backup(*args: object, **kwargs: object) -> None:
        prepare(*args, **kwargs)
        backup_path = kwargs["backup_path"]
        (backup_path / "old.txt").write_text("mutated-backup", encoding="utf-8")

    def fail_new_finalize(node: object, manifests: object) -> None:
        nonlocal finalize_calls
        expose_root(node, manifests)
        finalize_calls += 1
        if finalize_calls == 1:
            raise OSError("injected post-backup failure")

    monkeypatch.setattr(module, "_prepare_moved_backup", mutate_backup)
    monkeypatch.setattr(module, "_expose_tree_root_exact", fail_new_finalize)

    with pytest.raises(OSError, match="injected post-backup failure"):
        module.copytree_secure(source, destination, dirs_exist_ok=True)

    assert (destination / "old.txt").read_text(encoding="utf-8") == "old-original"
    assert not (destination / "new.txt").exists()


def test_cleanup_failure_leaves_only_private_exact_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "old.txt").write_text("old-original", encoding="utf-8")
    original = module._remove_tree_at

    def fail_backup_cleanup(parent: int, name: str, **kwargs: object) -> None:
        if ".backup-" in name:
            raise OSError("injected cleanup failure")
        original(parent, name, **kwargs)

    monkeypatch.setattr(module, "_remove_tree_at", fail_backup_cleanup)

    module.copytree_secure(source, destination, dirs_exist_ok=True)

    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"
    backups = list(tmp_path.glob(".destination.backup-*"))
    assert len(backups) == 1
    backup = backups[0]
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    assert (backup / "old.txt").read_text(encoding="utf-8") == "old-original"


def test_fallback_regular_to_fifo_swap_never_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    monkeypatch.setattr(module, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    source = tmp_path / "source"
    source.mkdir()
    victim = source / "victim.txt"
    victim.write_text("safe", encoding="utf-8")
    original_open = module._open_fallback_regular
    swapped = threading.Event()

    def swap_at_open(path: Path, before: os.stat_result, **kwargs: object):
        if path == victim and not swapped.is_set():
            victim.unlink()
            os.mkfifo(victim)
            swapped.set()
        return original_open(path, before, **kwargs)

    monkeypatch.setattr(module, "_open_fallback_regular", swap_at_open)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(module.copytree_secure, source, tmp_path / "destination")
        try:
            with pytest.raises(ValueError, match=r"changed|regular|replaced"):
                future.result(timeout=2)
        except FutureTimeoutError:
            writer = os.open(victim, os.O_WRONLY | os.O_NONBLOCK)
            os.write(writer, b"unblock")
            os.close(writer)
            future.result(timeout=2)
            pytest.fail("fallback source open blocked on a swapped FIFO")

    assert time.monotonic() - started < 2
    assert not (tmp_path / "destination").exists()


def test_fallback_regular_to_symlink_swap_reads_no_outside_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    monkeypatch.setattr(module, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    source = tmp_path / "source"
    source.mkdir()
    victim = source / "victim.txt"
    victim.write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"OUTSIDE-CANARY")
    original_open = module._open_fallback_regular
    swapped = False

    def swap_at_open(path: Path, before: os.stat_result, **kwargs: object):
        nonlocal swapped
        if path == victim and not swapped:
            victim.unlink()
            victim.symlink_to(outside)
            swapped = True
        return original_open(path, before, **kwargs)

    monkeypatch.setattr(module, "_open_fallback_regular", swap_at_open)

    with pytest.raises(ValueError, match=r"changed|symlink|replaced"):
        module.copytree_secure(source, tmp_path / "destination")

    assert outside.read_bytes() == b"OUTSIDE-CANARY"
    assert not (tmp_path / "destination").exists()


def test_filesystem_atomic_unsupported_routes_to_fallback_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    calls = 0

    def unsupported(*_args: object) -> int:
        nonlocal calls
        calls += 1
        ctypes.set_errno(errno.ENOTSUP)
        return -1

    monkeypatch.setattr(module, "_ATOMIC_RENAME", unsupported)

    module.copytree_secure(source, destination, dirs_exist_ok=True)

    assert calls == 1
    assert (destination / "old.txt").read_text(encoding="utf-8") == "old"
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"


def test_posix_best_effort_pass_detects_sampled_prior_sibling_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("GOOD", encoding="utf-8")
    (source / "b.txt").write_text("BBBB", encoding="utf-8")
    destination = tmp_path / "destination"
    apply_modes = module._apply_final_tree_modes
    hash_exact = module._hash_exact_descriptor
    root_descriptor: int | None = None
    mutated = False

    def arm_after_modes(descriptor: int, manifests: object) -> None:
        nonlocal root_descriptor
        apply_modes(descriptor, manifests)
        root_descriptor = descriptor

    def mutate_a_while_b_is_hashed(descriptor: int, *, parts: tuple[str, ...]) -> str:
        nonlocal mutated
        digest = hash_exact(descriptor, parts=parts)
        if root_descriptor is not None and parts == ("b.txt",) and not mutated:
            victim = os.open("a.txt", os.O_WRONLY, dir_fd=root_descriptor)
            os.pwrite(victim, b"EVIL", 0)
            os.close(victim)
            mutated = True
        return digest

    monkeypatch.setattr(module, "_apply_final_tree_modes", arm_after_modes)
    monkeypatch.setattr(module, "_hash_exact_descriptor", mutate_a_while_b_is_hashed)

    with pytest.raises(ValueError, match=r"contents|digest|changed"):
        module.copytree_secure(source, destination)

    assert not destination.exists()


def test_fallback_best_effort_pass_detects_sampled_prior_sibling_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    monkeypatch.setattr(module, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("GOOD", encoding="utf-8")
    (source / "b.txt").write_text("BBBB", encoding="utf-8")
    destination = tmp_path / "destination"
    apply_modes = module._apply_modes_fallback
    hash_path = module._hash_path_checked
    staged_root: Path | None = None
    mutated = False

    def arm_after_modes(stage: Path, manifests: object) -> None:
        nonlocal staged_root
        apply_modes(stage, manifests)
        staged_root = stage

    def mutate_a_while_b_is_hashed(path: Path, before: os.stat_result, **kwargs: object) -> str:
        nonlocal mutated
        digest = hash_path(path, before, **kwargs)
        if staged_root is not None and path.parent == staged_root and path.name == "b.txt" and not mutated:
            (staged_root / "a.txt").write_text("EVIL", encoding="utf-8")
            mutated = True
        return digest

    monkeypatch.setattr(module, "_apply_modes_fallback", arm_after_modes)
    monkeypatch.setattr(module, "_hash_path_checked", mutate_a_while_b_is_hashed)

    with pytest.raises(ValueError, match=r"contents|digest|changed"):
        module.copytree_secure(source, destination)

    assert not destination.exists()


def test_posix_best_effort_post_order_detects_sampled_late_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("AAAA", encoding="utf-8")
    (source / "b.txt").write_text("BBBB", encoding="utf-8")
    destination = tmp_path / "destination"
    apply_modes = module._apply_final_tree_modes
    hash_exact = module._hash_exact_descriptor
    root_descriptor: int | None = None
    injected = False

    def arm_after_modes(descriptor: int, manifests: object) -> None:
        nonlocal root_descriptor
        apply_modes(descriptor, manifests)
        root_descriptor = descriptor

    def inject_after_last_hash(descriptor: int, *, parts: tuple[str, ...]) -> str:
        nonlocal injected
        digest = hash_exact(descriptor, parts=parts)
        if root_descriptor is not None and parts == ("b.txt",) and not injected:
            extra = os.open(
                "extra.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root_descriptor,
            )
            os.close(extra)
            injected = True
        return digest

    monkeypatch.setattr(module, "_apply_final_tree_modes", arm_after_modes)
    monkeypatch.setattr(module, "_hash_exact_descriptor", inject_after_last_hash)

    with pytest.raises(ValueError, match=r"names|unexpected|missing|changed"):
        module.copytree_secure(source, destination)

    assert not destination.exists()


def test_fallback_best_effort_post_order_detects_sampled_late_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    monkeypatch.setattr(module, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("AAAA", encoding="utf-8")
    (source / "b.txt").write_text("BBBB", encoding="utf-8")
    destination = tmp_path / "destination"
    apply_modes = module._apply_modes_fallback
    hash_path = module._hash_path_checked
    staged_root: Path | None = None
    injected = False

    def arm_after_modes(stage: Path, manifests: object) -> None:
        nonlocal staged_root
        apply_modes(stage, manifests)
        staged_root = stage

    def inject_after_last_hash(path: Path, before: os.stat_result, **kwargs: object) -> str:
        nonlocal injected
        digest = hash_path(path, before, **kwargs)
        if staged_root is not None and path.parent == staged_root and path.name == "b.txt" and not injected:
            (staged_root / "extra.txt").write_text("late", encoding="utf-8")
            injected = True
        return digest

    monkeypatch.setattr(module, "_apply_modes_fallback", arm_after_modes)
    monkeypatch.setattr(module, "_hash_path_checked", inject_after_last_hash)

    with pytest.raises(ValueError, match=r"names|unexpected|missing|changed"):
        module.copytree_secure(source, destination)

    assert not destination.exists()


def test_posix_sampled_corrupt_rollback_uses_exact_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "a.txt").write_text("OLDA", encoding="utf-8")
    (destination / "b.txt").write_text("OLDB", encoding="utf-8")
    prepare_backup = module._prepare_moved_backup
    prepare_hidden = module._prepare_hidden_final_tree
    hash_exact = module._hash_exact_descriptor
    known_roots: list[int] = []
    after_backup = False
    arm_rollback = False
    mutated = False

    def track_roots(node: object, manifests: object) -> None:
        nonlocal arm_rollback
        if node.descriptor not in known_roots:
            known_roots.append(node.descriptor)
        if after_backup and node.descriptor == known_roots[0]:
            prepare_hidden(node, manifests)
            arm_rollback = True
            return
        prepare_hidden(node, manifests)

    def mark_backup(*args: object, **kwargs: object) -> object:
        nonlocal after_backup
        result = prepare_backup(*args, **kwargs)
        after_backup = True
        return result

    def corrupt_a_while_b_is_checked(descriptor: int, *, parts: tuple[str, ...]) -> str:
        nonlocal mutated
        digest = hash_exact(descriptor, parts=parts)
        if arm_rollback and len(known_roots) > 1 and parts == ("b.txt",) and not mutated:
            victim = os.open("a.txt", os.O_WRONLY, dir_fd=known_roots[1])
            os.pwrite(victim, b"EVIL", 0)
            os.close(victim)
            mutated = True
        return digest

    monkeypatch.setattr(module, "_prepare_hidden_final_tree", track_roots)
    monkeypatch.setattr(module, "_prepare_moved_backup", mark_backup)
    monkeypatch.setattr(module, "_hash_exact_descriptor", corrupt_a_while_b_is_checked)

    with pytest.raises(ValueError, match=r"contents|digest|changed"):
        module.copytree_secure(source, destination, dirs_exist_ok=True)

    assert (destination / "a.txt").read_text(encoding="utf-8") == "OLDA"
    assert (destination / "b.txt").read_text(encoding="utf-8") == "OLDB"
    assert sorted(path.name for path in destination.iterdir()) == ["a.txt", "b.txt"]


def test_fallback_sampled_corrupt_rollback_uses_exact_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    monkeypatch.setattr(module, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(module, "_ATOMIC_RENAME", None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "a.txt").write_text("OLDA", encoding="utf-8")
    (destination / "b.txt").write_text("OLDB", encoding="utf-8")
    prepare_backup = module._prepare_fallback_moved_backup
    prepare_hidden = module._prepare_hidden_fallback_tree
    validate_hidden = module._validate_hidden_fallback_tree
    hash_path = module._hash_path_checked
    roots: list[Path] = []
    after_backup = False
    arm_rollback = False
    mutated = False

    def track_roots(stage: Path, manifests: object) -> None:
        prepare_hidden(stage, manifests)
        if stage not in roots:
            roots.append(stage)

    def mark_backup(*args: object, **kwargs: object) -> None:
        nonlocal after_backup
        prepare_backup(*args, **kwargs)
        after_backup = True

    def arm_during_rollback(stage: Path, manifests: object) -> None:
        nonlocal arm_rollback
        if after_backup and len(roots) > 1 and stage == roots[1]:
            arm_rollback = True
        validate_hidden(stage, manifests)

    def corrupt_a_while_b_is_checked(path: Path, before: os.stat_result, **kwargs: object) -> str:
        nonlocal mutated
        digest = hash_path(path, before, **kwargs)
        if arm_rollback and len(roots) > 1 and path.parent == roots[1] and path.name == "b.txt" and not mutated:
            (roots[1] / "a.txt").write_text("EVIL", encoding="utf-8")
            mutated = True
        return digest

    monkeypatch.setattr(module, "_prepare_hidden_fallback_tree", track_roots)
    monkeypatch.setattr(module, "_prepare_fallback_moved_backup", mark_backup)
    monkeypatch.setattr(module, "_validate_hidden_fallback_tree", arm_during_rollback)
    monkeypatch.setattr(module, "_hash_path_checked", corrupt_a_while_b_is_checked)

    with pytest.raises(ValueError, match=r"contents|digest|changed"):
        module.copytree_secure(source, destination, dirs_exist_ok=True)

    assert (destination / "a.txt").read_text(encoding="utf-8") == "OLDA"
    assert (destination / "b.txt").read_text(encoding="utf-8") == "OLDB"
    assert sorted(path.name for path in destination.iterdir()) == ["a.txt", "b.txt"]
