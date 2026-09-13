# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for path-bound generated-output provenance."""

from __future__ import annotations

import json
import os
import shutil
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator.tier3 import output_provenance
from skillevaluator.tier3.harbor import secure_copy
from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks, stage_native_harbor_tasks
from skillevaluator.tier3.output_provenance import (
    GENERATED_OUTPUT_MARKER,
    is_generated_output_root,
    mark_generated_output_root,
)


@pytest.fixture(autouse=True)
def _isolated_output_provenance_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE",
        str(tmp_path / ".skillevaluator-state" / "output-provenance.key"),
    )


def _write_skill(tmp_path: Path, *, native: bool = False) -> Path:
    skill = tmp_path / "skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Complete the task.", "files": []}]),
        encoding="utf-8",
    )
    if native:
        task = skill / "evals" / "harbor" / "case-001"
        task.mkdir(parents=True)
        (task / "instruction.md").write_text("Complete the native task.\n", encoding="utf-8")
        (task / "task.toml").write_text(
            'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
            '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
            encoding="utf-8",
        )
    return skill


def _in_skill_output(skill: Path, name: str) -> tuple[Path, Path]:
    declared_root = skill / name
    return declared_root, declared_root / "dataset"


def _simulate_windows_crt_fstat_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return valid descriptor metadata with a Windows-incompatible identity."""
    original_fstat = output_provenance.os.fstat

    def incompatible_fstat(descriptor: int) -> object:
        opened = original_fstat(descriptor)
        # The test runs on POSIX, where the newly pinned output root still
        # needs a real descriptor identity. Simulate Windows CRT identity only
        # for the output file descriptors whose compatibility is under test.
        if stat.S_ISDIR(opened.st_mode):
            return opened
        return SimpleNamespace(
            st_dev=opened.st_dev + 10_000,
            st_ino=opened.st_ino + 10_000,
            st_mode=opened.st_mode,
            st_nlink=opened.st_nlink,
            st_size=opened.st_size,
            st_uid=opened.st_uid,
            st_mtime_ns=opened.st_mtime_ns,
            st_ctime_ns=opened.st_ctime_ns,
        )

    monkeypatch.setattr(output_provenance, "_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE", False, raising=False)
    monkeypatch.setattr(output_provenance.os, "fstat", incompatible_fstat)


def test_atomic_output_accepts_windows_crt_descriptor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.json"
    destination.write_bytes(b"old")
    _simulate_windows_crt_fstat_identity(monkeypatch)

    output_provenance.write_output_file_atomically(destination, b"complete")

    assert destination.read_bytes() == b"complete"
    assert not list(tmp_path.glob(".result.json.*.tmp"))


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptor-relative rename support")
def test_atomic_output_parent_swap_cannot_publish_outside_validated_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    parked = tmp_path / "parked"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = trusted / "result.json"
    destination.write_bytes(b"old")
    original_replace = output_provenance.os.replace

    def swap_parent_then_replace(
        source: Path | str,
        target: Path | str,
        *args: object,
        **kwargs: object,
    ) -> None:
        trusted.rename(parked)
        trusted.symlink_to(outside, target_is_directory=True)

        # Model a same-UID attacker who can relink the already-written
        # temporary into the redirected parent after the writer's final path
        # validation.  An unanchored os.replace then publishes those bytes
        # outside the declared output directory.
        if kwargs.get("src_dir_fd") is None:
            parked_source = parked / Path(source).name
            os.link(parked_source, outside / Path(source).name)
        original_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(output_provenance.os, "replace", swap_parent_then_replace)

    with pytest.raises(ValueError, match="directory changed"):
        output_provenance.write_output_file_atomically(destination, b"new-secret")

    assert not list(outside.iterdir())
    assert (parked / "result.json").read_bytes() in {b"old", b"new-secret"}


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows sharing semantics")
def test_atomic_output_pins_windows_parent_against_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    parked = tmp_path / "parked"
    destination = trusted / "result.json"
    original_replace = output_provenance.os.replace
    rename_was_blocked = False

    def attempt_parent_rename(
        source: Path | str,
        target: Path | str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal rename_was_blocked
        try:
            trusted.rename(parked)
        except OSError:
            rename_was_blocked = True
        original_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(output_provenance.os, "replace", attempt_parent_rename)

    output_provenance.write_output_file_atomically(destination, b"complete")

    assert rename_was_blocked
    assert destination.read_bytes() == b"complete"
    assert not parked.exists()


def test_marker_failure_cleanup_accepts_windows_crt_descriptor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generated"
    root.mkdir()
    original_fdopen = output_provenance.os.fdopen

    class _FailingWriter:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def __enter__(self) -> _FailingWriter:
            return self

        def __exit__(self, *_args: object) -> None:
            self._handle.close()  # type: ignore[attr-defined]

        def write(self, payload: bytes) -> int:
            self._handle.write(payload[:1])  # type: ignore[attr-defined]
            self._handle.flush()  # type: ignore[attr-defined]
            raise OSError("injected marker write failure")

        def flush(self) -> None:
            self._handle.flush()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            return self._handle.fileno()  # type: ignore[attr-defined,no-any-return]

    monkeypatch.setattr(output_provenance, "_load_or_create_key", lambda: b"k" * 32)
    monkeypatch.setattr(
        output_provenance.os,
        "fdopen",
        lambda descriptor, *args, **kwargs: _FailingWriter(original_fdopen(descriptor, *args, **kwargs)),
    )
    _simulate_windows_crt_fstat_identity(monkeypatch)

    with pytest.raises(OSError, match="injected marker write failure"):
        output_provenance.write_generated_output_marker(root)

    assert list(root.iterdir()) == []


@pytest.mark.parametrize("failure_point", ["fstat", "lstat"])
def test_marker_metadata_failure_closes_descriptor_without_unproven_name_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    root = tmp_path / "generated"
    root.mkdir()
    marker = root / GENERATED_OUTPUT_MARKER
    original_close = output_provenance.os.close
    original_fstat = output_provenance.os.fstat
    original_lstat = output_provenance.Path.lstat
    original_unlink_if_same_file = output_provenance._unlink_if_same_file
    closed_descriptors: list[int] = []
    cleanup_attempts: list[Path] = []
    failure_injected = False

    def tracked_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        original_close(descriptor)

    def injected_fstat(descriptor: int) -> os.stat_result:
        nonlocal failure_injected
        if failure_point == "fstat" and not failure_injected:
            failure_injected = True
            raise OSError("injected marker fstat failure")
        return original_fstat(descriptor)

    def injected_lstat(path: Path) -> os.stat_result:
        nonlocal failure_injected
        if failure_point == "lstat" and path == marker and not failure_injected:
            failure_injected = True
            raise OSError("injected marker lstat failure")
        return original_lstat(path)

    def track_cleanup(path: Path, expected: os.stat_result) -> bool:
        cleanup_attempts.append(path)
        return original_unlink_if_same_file(path, expected)

    monkeypatch.setattr(output_provenance, "_load_or_create_key", lambda: b"k" * 32)
    monkeypatch.setattr(output_provenance.os, "close", tracked_close)
    monkeypatch.setattr(output_provenance.os, "fstat", injected_fstat)
    monkeypatch.setattr(output_provenance.Path, "lstat", injected_lstat)
    monkeypatch.setattr(output_provenance, "_unlink_if_same_file", track_cleanup)

    with pytest.raises(OSError, match=rf"injected marker {failure_point} failure"):
        output_provenance.write_generated_output_marker(root)

    assert failure_injected
    assert closed_descriptors
    assert cleanup_attempts == []
    assert marker.is_file()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX rename-while-open behavior")
def test_marker_lstat_failure_never_unlinks_a_substituted_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generated"
    root.mkdir()
    marker = root / GENERATED_OUTPUT_MARKER
    original_partial = root / ".original-partial-marker"
    original_lstat = output_provenance.Path.lstat
    substituted = False

    def substitute_then_fail(path: Path) -> os.stat_result:
        nonlocal substituted
        if path == marker and not substituted:
            substituted = True
            marker.rename(original_partial)
            marker.write_text("preserve replacement\n", encoding="utf-8")
            raise OSError("injected marker lstat failure after substitution")
        return original_lstat(path)

    monkeypatch.setattr(output_provenance, "_load_or_create_key", lambda: b"k" * 32)
    monkeypatch.setattr(output_provenance.Path, "lstat", substitute_then_fail)

    with pytest.raises(OSError, match="injected marker lstat failure after substitution"):
        output_provenance.write_generated_output_marker(root)

    assert substituted
    assert marker.read_text(encoding="utf-8") == "preserve replacement\n"
    assert original_partial.is_file()


def test_concurrent_first_key_creation_publishes_one_complete_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "generated"
    barrier = threading.Barrier(2)
    original_link = output_provenance.os.link

    def synchronized_link(source: Path, target: Path) -> None:
        barrier.wait(timeout=5)
        original_link(source, target)

    monkeypatch.setattr(output_provenance.os, "link", synchronized_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        payloads = list(
            executor.map(
                lambda _index: output_provenance.generated_output_marker_payload(destination),
                range(2),
            )
        )

    key_path = Path(os.environ["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"])
    assert payloads[0] == payloads[1]
    if os.name == "nt":
        assert 32 < key_path.stat().st_size <= 4096
    else:
        assert key_path.stat().st_size == 32
    assert key_path.stat().st_nlink == 1
    assert not list(key_path.parent.glob(".output-provenance.key.tmp-*"))


def test_interrupted_hardlink_publish_is_recovered(tmp_path: Path) -> None:
    key_path = Path(os.environ["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"])
    output_provenance.generated_output_marker_payload(tmp_path / "seed")
    temporary = key_path.parent / ".output-provenance.key.tmp-interrupted"
    os.link(key_path, temporary)

    payload = output_provenance.generated_output_marker_payload(tmp_path / "generated")

    assert payload.startswith(b"SkillEvaluator generated output v2\n")
    assert not temporary.exists()
    assert key_path.stat().st_nlink == 1


@pytest.mark.parametrize("failure_point", ["write", "fsync"])
def test_marker_publication_failure_removes_only_its_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """A failed marker claim must not leave a plausible owned reservation."""
    root = tmp_path / "generated"
    root.mkdir()
    output_provenance.generated_output_marker_payload(tmp_path / "seed")
    original_fdopen = output_provenance.os.fdopen
    original_fsync = output_provenance.os.fsync

    class _PartialWriter:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def __enter__(self) -> _PartialWriter:
            return self

        def __exit__(self, *_args: object) -> None:
            self._handle.close()  # type: ignore[attr-defined]

        def write(self, payload: bytes) -> int:
            self._handle.write(payload[: len(payload) // 2])  # type: ignore[attr-defined]
            self._handle.flush()  # type: ignore[attr-defined]
            raise OSError("injected marker write failure")

        def flush(self) -> None:
            self._handle.flush()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            return self._handle.fileno()  # type: ignore[attr-defined,no-any-return]

    if failure_point == "write":
        monkeypatch.setattr(
            output_provenance.os,
            "fdopen",
            lambda descriptor, *args, **kwargs: _PartialWriter(original_fdopen(descriptor, *args, **kwargs)),
        )
        expected_error = "injected marker write failure"
    else:
        monkeypatch.setattr(
            output_provenance.os,
            "fsync",
            lambda _descriptor: (_ for _ in ()).throw(OSError("injected marker fsync failure")),
        )
        expected_error = "injected marker fsync failure"

    with pytest.raises(OSError, match=expected_error):
        output_provenance.write_generated_output_marker(root)

    assert not (root / GENERATED_OUTPUT_MARKER).exists()
    assert list(root.iterdir()) == []
    monkeypatch.setattr(output_provenance.os, "fdopen", original_fdopen)
    monkeypatch.setattr(output_provenance.os, "fsync", original_fsync)


def test_owned_output_cleanup_refuses_a_path_substituted_after_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reserved-run"
    moved_root = tmp_path / "moved-reserved-run"
    mark_generated_output_root(root)
    original_identity = root.stat().st_dev, root.stat().st_ino
    original_fingerprint = output_provenance._node_fingerprint
    root_fingerprint_calls = 0

    def substitute_after_identity_check(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
        nonlocal root_fingerprint_calls
        fingerprint = original_fingerprint(metadata)
        if (metadata.st_dev, metadata.st_ino) == original_identity:
            root_fingerprint_calls += 1
            if root_fingerprint_calls == 2:
                root.rename(moved_root)
                root.mkdir()
                (root / "preserve.txt").write_text("replacement\n", encoding="utf-8")
        return fingerprint

    monkeypatch.setattr(output_provenance, "_node_fingerprint", substitute_after_identity_check)

    removed = output_provenance.remove_generated_output_root_if_owned(root)

    assert removed is False
    assert (root / "preserve.txt").read_text(encoding="utf-8") == "replacement\n"
    assert moved_root.is_dir()


def test_fallback_owned_output_cleanup_never_recursively_deletes_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reserved-run"
    mark_generated_output_root(root)
    sentinel = root / "preserve.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    identity = root.stat().st_dev, root.stat().st_ino
    monkeypatch.setattr(output_provenance, "_DESCRIPTOR_BACKEND", False)

    removed = output_provenance.remove_generated_output_root_if_owned(root, expected_identity=identity)

    assert removed is False
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert (root / GENERATED_OUTPUT_MARKER).is_file()


def test_fixed_marker_cannot_authorize_authored_source_replacement(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "fixed-marker-results")
    authored_task = output / "case-001"
    authored_task.mkdir(parents=True)
    (output / GENERATED_OUTPUT_MARKER).write_bytes(b"SkillEvaluator generated output v1\n")
    (authored_task / "SKILL.md").write_text("# Authored source\n", encoding="utf-8")
    sentinel = authored_task / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"runtime skill source|marker"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_generated_marker_is_bound_to_its_original_destination(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    original_root, original = _in_skill_output(skill, "original-results")
    copied_root, copied = _in_skill_output(skill, "copied-results")
    generate_harbor_tasks(skill, original, repo_context_exclude_paths=(original_root,))
    copied.mkdir(parents=True)
    shutil.copy2(original / GENERATED_OUTPUT_MARKER, copied / GENERATED_OUTPUT_MARKER)
    authored_task = copied / "case-001"
    authored_task.mkdir()
    (authored_task / "SKILL.md").write_text("# Authored source\n", encoding="utf-8")
    sentinel = authored_task / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"runtime skill source|marker"):
        generate_harbor_tasks(skill, copied, repo_context_exclude_paths=(copied_root,))

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


@pytest.mark.parametrize("native", [False, True])
def test_invalid_marker_blocks_in_skill_atomic_replacement(tmp_path: Path, native: bool) -> None:
    skill = _write_skill(tmp_path, native=native)
    declared_root, output = _in_skill_output(skill, "invalid-marker-results")
    output.mkdir(parents=True)
    (output / GENERATED_OUTPUT_MARKER).write_text("invalid\n", encoding="utf-8")
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    stager = stage_native_harbor_tasks if native else generate_harbor_tasks

    with pytest.raises(ValueError, match="marker"):
        stager(skill, output, repo_context_exclude_paths=(declared_root,))

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


@pytest.mark.parametrize(
    "force_fallback",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(os.name == "nt", reason="native Windows already exercises the fallback publisher"),
        ),
    ],
)
def test_partial_atomic_publication_preserves_signed_output_and_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_fallback: bool,
) -> None:
    if force_fallback:
        monkeypatch.setattr(secure_copy, "_DESCRIPTOR_BACKEND", False)
        monkeypatch.setattr(secure_copy, "_ATOMIC_RENAME", None)
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "partial-results")
    generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))
    old_dataset = (output / "dataset.toml").read_bytes()
    output_parent_metadata = output.parent.stat()
    rename_calls = 0

    using_fallback = not secure_copy._DESCRIPTOR_BACKEND or secure_copy._ATOMIC_RENAME is None
    if using_fallback:
        original_path_rename = Path.rename

        def _fail_second_path_rename(source: Path, target: Path) -> Path:
            nonlocal rename_calls
            target_path = Path(target)
            if output in (source, target_path):
                rename_calls += 1
                if rename_calls == 2:
                    raise OSError("injected generated-output publish failure after old destination moved")
            return original_path_rename(source, target_path)

        monkeypatch.setattr(Path, "rename", _fail_second_path_rename)
    else:
        original_atomic_rename = secure_copy._rename_no_replace

        def _fail_second_atomic_rename(
            source_name: str,
            destination_name: str,
            *,
            source_parent: int,
            destination_parent: int,
        ) -> None:
            nonlocal rename_calls
            in_output_parent = os.path.samestat(
                os.fstat(source_parent),
                output_parent_metadata,
            ) and os.path.samestat(os.fstat(destination_parent), output_parent_metadata)
            if in_output_parent and output.name in (source_name, destination_name):
                rename_calls += 1
                if rename_calls == 2:
                    raise OSError("injected generated-output publish failure after old destination moved")
            original_atomic_rename(
                source_name,
                destination_name,
                source_parent=source_parent,
                destination_parent=destination_parent,
            )

        monkeypatch.setattr(secure_copy, "_rename_no_replace", _fail_second_atomic_rename)
    with pytest.raises(OSError, match="generated-output publish"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))

    assert rename_calls == 3
    assert (output / "dataset.toml").read_bytes() == old_dataset
    assert is_generated_output_root(output)

    if using_fallback:
        monkeypatch.setattr(Path, "rename", original_path_rename)
    else:
        monkeypatch.setattr(secure_copy, "_rename_no_replace", original_atomic_rename)
    tasks = generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))
    assert tasks
    assert is_generated_output_root(output)


def test_external_output_retains_unmarked_overwrite_behavior(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    output = tmp_path / "external-output"
    output.mkdir()
    (output / GENERATED_OUTPUT_MARKER).write_text("invalid\n", encoding="utf-8")
    sentinel = output / "keep.txt"
    sentinel.write_text("replace\n", encoding="utf-8")

    tasks = generate_harbor_tasks(skill, output)

    assert tasks
    assert not sentinel.exists()
    assert not (output / GENERATED_OUTPUT_MARKER).exists()


def test_unsigned_v2_marker_does_not_create_private_key(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "unsigned-results")
    output.mkdir(parents=True)
    (output / GENERATED_OUTPUT_MARKER).write_bytes(b"SkillEvaluator generated output v2\n" + (b"A" * 43) + b"\n")
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    key_path = Path(os.environ["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"])

    with pytest.raises(ValueError, match="marker"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert not key_path.exists()


def test_key_override_inside_skill_is_rejected_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "key-location-results")
    key_path = skill / "private-output-key"
    monkeypatch.setenv("SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE", str(key_path))

    with pytest.raises(ValueError, match="outside evaluated and generated trees"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))

    assert not key_path.exists()
    assert not output.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_fifo_marker_and_key_are_rejected_without_blocking(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "fifo-results")
    output.mkdir(parents=True)
    os.mkfifo(output / GENERATED_OUTPUT_MARKER)
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match="marker"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"

    key_path = Path(os.environ["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"])
    key_path.parent.mkdir(parents=True, mode=0o700)
    os.mkfifo(key_path)
    with pytest.raises(ValueError, match="single-link regular file"):
        mark_generated_output_root(tmp_path / "key-fifo-output")


def test_symlinked_private_marker_root_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "marker-target"
    target.mkdir()
    linked_root = tmp_path / "marker-link"
    linked_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match=r"symlink|reparse|junction"):
        output_provenance.write_generated_output_marker(linked_root, destination=tmp_path / "destination")

    assert list(target.iterdir()) == []
