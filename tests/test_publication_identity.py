# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical publication target identities bind evidence to exact source."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator import publication_identity
from skillevaluator.models import ValidationResult
from skillevaluator.publication_identity import (
    PUBLICATION_TARGET_DIGEST_ALGORITHM,
    PublicationTargetConflictError,
    finalize_publication_target,
    publication_source_digest,
    publication_target_from_path,
    stamp_publication_target,
)


def _skill(root: Path, name: str = "demo") -> Path:
    skill = root / name
    script = skill / "scripts" / "run.sh"
    script.parent.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
    script.write_text("#!/bin/sh\necho demo\n", encoding="utf-8")
    script.chmod(0o755)
    return skill


def test_digest_is_deterministic_and_ignores_only_generated_artifacts(tmp_path: Path) -> None:
    first = _skill(tmp_path / "first")
    (first / "empty-author-dir").mkdir()
    second = tmp_path / "second" / "demo"
    second.parent.mkdir()
    shutil.copytree(first, second, copy_function=shutil.copy2)

    (first / "results").mkdir()
    (first / "results" / "run.json").write_text('{"run": 1}', encoding="utf-8")
    (first / "BENCHMARK.md").write_text("generated card A", encoding="utf-8")
    (second / "results").mkdir()
    (second / "results" / "run.json").write_text('{"run": 2}', encoding="utf-8")
    (second / "BENCHMARK.md").write_text("generated card B", encoding="utf-8")

    assert publication_source_digest(first) == publication_source_digest(second)
    identity = publication_target_from_path(first)
    assert identity is not None
    assert identity == {
        "skill_name": "demo",
        "skill_digest": f"sha256:{publication_source_digest(first)}",
        "skill_digest_algorithm": PUBLICATION_TARGET_DIGEST_ALGORITHM,
    }


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable-bit golden value")
def test_version_two_digest_has_a_stable_known_answer(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    empty = skill / "empty-author-dir"
    empty.mkdir()
    (skill / "SKILL.md").chmod(0o644)
    (skill / "scripts").chmod(0o755)
    (skill / "scripts" / "run.sh").chmod(0o755)
    empty.chmod(0o755)

    assert PUBLICATION_TARGET_DIGEST_ALGORITHM == "skill-evaluator-source-tree/2"
    assert publication_source_digest(skill) == "8b177d0281455ede748f8027aed3cc8124cff6578de603173f941be8f95e621a"


def test_checked_fallback_uses_the_same_digest_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _skill(tmp_path)
    (skill / "evals").mkdir()
    (skill / "evals" / "evals.json").write_text('{"evals": []}', encoding="utf-8")
    expected = publication_source_digest(skill)
    assert expected is not None

    monkeypatch.setattr(publication_identity, "_DESCRIPTOR_BACKEND", False)

    assert publication_source_digest(skill) == expected


def test_checked_windows_fallback_accepts_crt_descriptor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _skill(tmp_path)
    expected = publication_source_digest(skill)
    assert expected is not None
    original_fstat = publication_identity.os.fstat

    def windows_crt_fstat(descriptor: int) -> object:
        opened = original_fstat(descriptor)
        return SimpleNamespace(
            st_dev=opened.st_dev + 10_000,
            st_ino=opened.st_ino + 10_000,
            st_mode=opened.st_mode,
            st_nlink=opened.st_nlink,
            st_size=opened.st_size,
            st_mtime_ns=opened.st_mtime_ns,
            st_ctime_ns=opened.st_ctime_ns,
        )

    monkeypatch.setattr(publication_identity, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(publication_identity, "_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE", False, raising=False)
    monkeypatch.setattr(publication_identity.os, "fstat", windows_crt_fstat)

    assert publication_source_digest(skill) == expected


@pytest.mark.skipif(
    not publication_identity._DESCRIPTOR_BACKEND,
    reason="descriptor traversal is not available",
)
def test_descriptor_open_accepts_volatile_ancestor_metadata_for_the_same_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "stable-source"
    target.mkdir()
    canonical_target = publication_identity._absolute_lexical(target)
    original_stat = publication_identity.os.stat
    observed_target = False

    def volatile_stat(path: object, *args: object, **kwargs: object) -> object:
        nonlocal observed_target
        metadata = original_stat(path, *args, **kwargs)
        if path == target.name and kwargs.get("dir_fd") is not None:
            observed_target = True
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
                st_nlink=metadata.st_nlink + 1,
                st_size=metadata.st_size + 4096,
                st_mtime_ns=metadata.st_mtime_ns + 1,
                st_ctime_ns=metadata.st_ctime_ns + 1,
            )
        return metadata

    monkeypatch.setattr(publication_identity.os, "stat", volatile_stat)

    descriptor = publication_identity._open_directory_no_follow(canonical_target)
    try:
        assert os.path.samestat(publication_identity.os.fstat(descriptor), original_stat(canonical_target))
    finally:
        os.close(descriptor)
    assert observed_target is True


@pytest.mark.skipif(
    not publication_identity._DESCRIPTOR_BACKEND,
    reason="descriptor traversal is not available",
)
def test_descriptor_open_rejects_replacement_directory_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "stable-source"
    target.mkdir()
    replacement = tmp_path / "replacement-source"
    replacement.mkdir()
    canonical_target = publication_identity._absolute_lexical(target)
    original_stat = publication_identity.os.stat
    observed_target = False

    def replaced_stat(path: object, *args: object, **kwargs: object) -> object:
        nonlocal observed_target
        if path == target.name and kwargs.get("dir_fd") is not None:
            observed_target = True
            return original_stat(replacement)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(publication_identity.os, "stat", replaced_stat)

    with pytest.raises(ValueError, match="directory changed while opening"):
        publication_identity._open_directory_no_follow(canonical_target)
    assert observed_target is True


def test_digest_binds_empty_directories(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    original = publication_source_digest(skill)
    assert original is not None

    empty = skill / "author-owned-empty-dir"
    empty.mkdir()
    with_empty_directory = publication_source_digest(skill)
    assert with_empty_directory is not None
    assert with_empty_directory != original


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable-bit semantics")
def test_digest_binds_executable_bits(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    original = publication_source_digest(skill)
    assert original is not None

    script = skill / "scripts" / "run.sh"
    script.chmod(0o644)
    without_executable_bit = publication_source_digest(skill)
    assert without_executable_bit is not None
    assert without_executable_bit != original

    script.chmod(0o600)
    assert publication_source_digest(skill) == without_executable_bit


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory executable-bit semantics")
def test_digest_binds_the_root_directory_node(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    skill.chmod(0o755)
    original = publication_source_digest(skill)
    assert original is not None

    skill.chmod(0o700)

    changed = publication_source_digest(skill)
    assert changed is not None
    assert changed != original


def test_generated_exclusions_apply_only_to_the_expected_node_kind(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    original = publication_source_digest(skill)
    assert original is not None

    generated_file_name_used_as_directory = skill / "BENCHMARK.md"
    generated_file_name_used_as_directory.mkdir()
    (generated_file_name_used_as_directory / "authored.txt").write_text("content", encoding="utf-8")
    with_authored_directory = publication_source_digest(skill)
    assert with_authored_directory is not None and with_authored_directory != original

    (skill / "results").write_text("author-owned regular file", encoding="utf-8")
    with_authored_file = publication_source_digest(skill)
    assert with_authored_file is not None and with_authored_file != with_authored_directory


def test_digest_excludes_root_git_metadata_for_directory_and_file_checkouts(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    original = publication_source_digest(skill)
    assert original is not None

    git_metadata = skill / ".git"
    git_metadata.mkdir()
    (git_metadata / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    assert publication_source_digest(skill) == original

    shutil.rmtree(git_metadata)
    git_metadata.write_text("gitdir: /tmp/checkout-a/.git/worktrees/demo\n", encoding="utf-8")
    assert publication_source_digest(skill) == original

    git_metadata.write_text("gitdir: /tmp/checkout-b/.git/worktrees/demo\n", encoding="utf-8")
    assert publication_source_digest(skill) == original


@pytest.mark.parametrize(
    "descriptor_backend",
    [
        pytest.param(False, id="checked-fallback"),
        pytest.param(
            True,
            id="descriptor",
            marks=pytest.mark.skipif(
                not publication_identity._DESCRIPTOR_BACKEND,
                reason="descriptor traversal is not available",
            ),
        ),
    ],
)
def test_digest_excludes_nested_git_pointer_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptor_backend: bool,
) -> None:
    skill = _skill(tmp_path)
    nested = skill / "vendor"
    nested.mkdir()
    original = publication_source_digest(skill)
    assert original is not None
    monkeypatch.setattr(publication_identity, "_DESCRIPTOR_BACKEND", descriptor_backend)

    pointer = nested / ".git"
    pointer.write_text("gitdir: /tmp/checkout-a/.git/modules/vendor\n", encoding="utf-8")
    assert publication_source_digest(skill) == original

    pointer.write_text("gitdir: /tmp/checkout-b/.git/modules/vendor\n", encoding="utf-8")
    assert publication_source_digest(skill) == original


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_digest_excludes_filesystem_case_alias_of_git_metadata(tmp_path: Path, kind: str) -> None:
    skill = _skill(tmp_path)
    original = publication_source_digest(skill)
    assert original is not None

    alias = skill / ".GIT"
    if kind == "directory":
        alias.mkdir()
        (alias / "HEAD").write_text("first\n", encoding="utf-8")
    else:
        alias.write_text("gitdir: first\n", encoding="utf-8")
    canonical = skill / ".git"
    if not canonical.exists() or not alias.samefile(canonical):
        pytest.skip("filesystem preserves .GIT as an authored path distinct from .git")

    assert publication_source_digest(skill) == original
    if kind == "directory":
        (alias / "HEAD").write_text("second\n", encoding="utf-8")
    else:
        alias.write_text("gitdir: second\n", encoding="utf-8")
    assert publication_source_digest(skill) == original


def test_digest_binds_authored_eval_inputs_but_not_tier3_run_outputs(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    evals = skill / "evals"
    evals.mkdir()
    dataset = evals / "evals.json"
    dataset.write_text('{"evals": []}', encoding="utf-8")
    authored = publication_source_digest(skill)
    assert authored is not None

    dataset.write_text('{"evals": [{"id": "case-1"}]}', encoding="utf-8")
    changed_dataset = publication_source_digest(skill)
    assert changed_dataset is not None and changed_dataset != authored

    run_output = evals / "results" / "run-1" / "result.json"
    run_output.parent.mkdir(parents=True)
    run_output.write_text('{"score": 0}', encoding="utf-8")
    assert publication_source_digest(skill) == changed_dataset

    run_output.write_text('{"score": 1}', encoding="utf-8")
    assert publication_source_digest(skill) == changed_dataset


def test_digest_handles_case_alias_of_canonical_eval_results_by_filesystem_semantics(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    case_preserved_evals = skill / "EVALS"
    case_preserved_evals.mkdir()
    (case_preserved_evals / "evals.json").write_text('{"evals": []}', encoding="utf-8")
    authored = publication_source_digest(skill)
    assert authored is not None

    run_output = case_preserved_evals / "results" / "run-1" / "result.json"
    run_output.parent.mkdir(parents=True)
    run_output.write_text('{"score": 0}', encoding="utf-8")
    with_run_output = publication_source_digest(skill)
    assert with_run_output is not None

    canonical_evals = skill / "evals"
    case_alias_is_canonical = canonical_evals.exists() and case_preserved_evals.samefile(canonical_evals)
    if case_alias_is_canonical:
        assert with_run_output == authored
        run_output.write_text('{"score": 1}', encoding="utf-8")
        assert publication_source_digest(skill) == authored
    else:
        assert with_run_output != authored
        run_output.write_text('{"score": 1}', encoding="utf-8")
        assert publication_source_digest(skill) != with_run_output


def test_digest_excludes_only_root_generated_benchmark_card(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    root_card = skill / "BENCHMARK.md"
    root_card.write_text("generated card A", encoding="utf-8")
    generated = publication_source_digest(skill)
    assert generated is not None

    root_card.write_text("generated card B", encoding="utf-8")
    assert publication_source_digest(skill) == generated

    authored = skill / "references" / "BENCHMARK.md"
    authored.parent.mkdir()
    authored.write_text("author-owned reference A", encoding="utf-8")
    nested = publication_source_digest(skill)
    assert nested is not None and nested != generated

    authored.write_text("author-owned reference B", encoding="utf-8")
    assert publication_source_digest(skill) != nested


@pytest.mark.parametrize(
    ("canonical_name", "case_variant"),
    [
        ("BENCHMARK.md", "BENCHMARK.MD"),
        ("skill-card.md", "SKILL-CARD.MD"),
        ("skill.oms.sig", "SKILL.OMS.SIG"),
    ],
)
def test_root_generated_file_case_alias_follows_filesystem_semantics(
    tmp_path: Path,
    canonical_name: str,
    case_variant: str,
) -> None:
    skill = _skill(tmp_path)
    original = publication_source_digest(skill)
    assert original is not None
    alias = skill / case_variant
    alias.write_text("first\n", encoding="utf-8")
    canonical = skill / canonical_name
    first = publication_source_digest(skill)

    alias.write_text("second\n", encoding="utf-8")
    second = publication_source_digest(skill)
    if canonical.exists() and alias.samefile(canonical):
        assert first == original
        assert second == original
    else:
        assert first != original
        assert second != first


def test_digest_binds_normalized_relative_paths_and_file_bytes(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    original = publication_source_digest(skill)
    assert original is not None

    script = skill / "scripts" / "run.sh"
    script.write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
    changed_bytes = publication_source_digest(skill)
    assert changed_bytes is not None and changed_bytes != original

    renamed = skill / "scripts" / "renamed.sh"
    script.rename(renamed)
    changed_path = publication_source_digest(skill)
    assert changed_path is not None and changed_path != changed_bytes


def test_digest_supports_a_single_regular_file_target(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text("first", encoding="utf-8")
    original = publication_source_digest(target)
    assert original is not None

    target.write_text("second", encoding="utf-8")
    changed = publication_source_digest(target)
    assert changed is not None and changed != original

    alias = tmp_path / "alias.md"
    alias.symlink_to(target)
    assert publication_source_digest(alias) is None


def test_digest_binds_the_target_root_kind(tmp_path: Path) -> None:
    file_target = tmp_path / "file-parent" / "same-name"
    file_target.parent.mkdir()
    file_target.write_bytes(b"identical bytes\n")
    file_target.chmod(0o644)

    directory_target = tmp_path / "directory-parent" / "same-name"
    directory_target.mkdir(parents=True)
    child = directory_target / "same-name"
    child.write_bytes(b"identical bytes\n")
    child.chmod(0o644)

    assert publication_target_from_path(file_target) != publication_target_from_path(directory_target)


def test_digest_rejects_root_and_nested_symlinks(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(skill, target_is_directory=True)
    assert publication_source_digest(alias) is None

    nested = skill / "linked.sh"
    nested.symlink_to(skill / "scripts" / "run.sh")
    assert publication_source_digest(skill) is None


def test_digest_does_not_hide_symlinks_behind_generated_exclusions(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    (skill / "results").symlink_to(skill / "scripts", target_is_directory=True)

    assert publication_source_digest(skill) is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is not supported")
def test_digest_rejects_special_files(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    os.mkfifo(skill / "named-pipe")
    assert publication_source_digest(skill) is None


def test_digest_rejects_hard_linked_files(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    os.link(skill / "SKILL.md", skill / "SKILL-copy.md")
    assert publication_source_digest(skill) is None


def test_digest_rejects_normalized_path_collisions(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    composed = "caf\u00e9.txt"
    decomposed = "cafe\u0301.txt"
    if composed == decomposed:
        pytest.skip("filesystem cannot represent distinct Unicode normalization forms")
    (skill / composed).write_text("first", encoding="utf-8")
    try:
        (skill / decomposed).write_text("second", encoding="utf-8")
    except OSError:
        pytest.skip("filesystem normalizes Unicode filenames")
    if len([path for path in skill.iterdir() if unicodedata.normalize("NFC", path.name) == composed]) < 2:
        pytest.skip("filesystem normalizes Unicode filenames")

    assert publication_source_digest(skill) is None


def test_digest_rejects_source_mutation_while_hashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill = _skill(tmp_path)
    manifest = skill / "SKILL.md"
    original_hash_descriptor = publication_identity._hash_descriptor
    mutated = False

    def mutate_after_read(descriptor: int) -> str:
        nonlocal mutated
        value = original_hash_descriptor(descriptor)
        if not mutated:
            mutated = True
            manifest.write_text("---\nname: changed-during-read\n---\n", encoding="utf-8")
        return value

    monkeypatch.setattr(publication_identity, "_hash_descriptor", mutate_after_read)

    assert publication_source_digest(skill) is None
    assert mutated is True


@pytest.mark.parametrize(
    "descriptor_backend",
    [
        pytest.param(False, id="checked-fallback"),
        pytest.param(
            True,
            id="descriptor",
            marks=pytest.mark.skipif(
                not publication_identity._DESCRIPTOR_BACKEND,
                reason="descriptor traversal is not available",
            ),
        ),
    ],
)
def test_finalize_rejects_persistent_sibling_mutation_during_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptor_backend: bool,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    earlier = source / "a.txt"
    later = source / "z.txt"
    earlier.write_text("before\n", encoding="utf-8")
    later.write_text("stable\n", encoding="utf-8")
    initial = publication_target_from_path(source)
    assert initial is not None
    later_digest = hashlib.sha256(later.read_bytes()).hexdigest()
    original_hash_descriptor = publication_identity._hash_descriptor
    mutated = False

    def mutate_earlier_while_hashing_later(descriptor: int) -> str:
        nonlocal mutated
        value = original_hash_descriptor(descriptor)
        if value == later_digest and not mutated:
            earlier.write_text("after persistent mutation\n", encoding="utf-8")
            mutated = True
        return value

    monkeypatch.setattr(publication_identity, "_DESCRIPTOR_BACKEND", descriptor_backend)
    monkeypatch.setattr(publication_identity, "_hash_descriptor", mutate_earlier_while_hashing_later)
    result = ValidationResult(validator_name="SCHEMA")

    assert finalize_publication_target([result], source, initial) is None
    assert mutated is True
    assert "publication_target" not in result.metadata
    assert result.metadata["publication_target_conflict"] == "source changed during validation"


@pytest.mark.parametrize(
    "descriptor_backend",
    [
        pytest.param(False, id="checked-fallback"),
        pytest.param(
            True,
            id="descriptor",
            marks=pytest.mark.skipif(
                not publication_identity._DESCRIPTOR_BACKEND,
                reason="descriptor traversal is not available",
            ),
        ),
    ],
)
def test_finalize_rejects_sibling_mutation_after_its_last_reverse_pass_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptor_backend: bool,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    earlier = source / "a.txt"
    later = source / "z.txt"
    earlier.write_text("before\n", encoding="utf-8")
    later.write_text("stable\n", encoding="utf-8")
    initial = publication_target_from_path(source)
    assert initial is not None
    earlier_digest = hashlib.sha256(earlier.read_bytes()).hexdigest()
    original_hash_descriptor = publication_identity._hash_descriptor
    earlier_reads = 0
    mutated = False

    def mutate_later_after_its_last_reverse_pass_read(descriptor: int) -> str:
        nonlocal earlier_reads, mutated
        value = original_hash_descriptor(descriptor)
        if value == earlier_digest:
            earlier_reads += 1
            if earlier_reads == 2:
                later.write_text("change\n", encoding="utf-8")
                mutated = True
        return value

    monkeypatch.setattr(publication_identity, "_DESCRIPTOR_BACKEND", descriptor_backend)
    monkeypatch.setattr(
        publication_identity,
        "_hash_descriptor",
        mutate_later_after_its_last_reverse_pass_read,
    )
    result = ValidationResult(validator_name="SCHEMA")

    assert finalize_publication_target([result], source, initial) is None
    assert earlier_reads == 3
    assert mutated is True
    assert "publication_target" not in result.metadata
    assert result.metadata["publication_target_conflict"] == "source changed during validation"


def test_digest_rejects_a_manifest_without_a_canonical_file_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _skill(tmp_path)
    monkeypatch.setattr(publication_identity, "_hash_descriptor", lambda _descriptor: "")

    assert publication_source_digest(skill) is None


def test_publication_target_normalizes_the_visible_skill_name(tmp_path: Path) -> None:
    decomposed_name = "cafe\u0301"
    skill = _skill(tmp_path, decomposed_name)

    identity = publication_target_from_path(skill)

    assert identity is not None
    assert identity["skill_name"] == unicodedata.normalize("NFC", decomposed_name)


def test_publication_target_does_not_trust_scandir_identity_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _skill(tmp_path, "demo")
    original_scandir = publication_identity.os.scandir

    class EntryWithoutPortableIdentity:
        def __init__(self, entry: os.DirEntry[str]) -> None:
            self.name = entry.name
            self.path = entry.path

        def stat(self, *, follow_symlinks: bool = True) -> object:
            del follow_symlinks
            raise AssertionError("DirEntry.stat identity is not portable on Windows")

    class ParentEntries:
        def __init__(self, path: Path) -> None:
            with original_scandir(path) as entries:
                self.entries = [EntryWithoutPortableIdentity(entry) for entry in entries]

        def __enter__(self) -> object:
            return iter(self.entries)

        def __exit__(self, *args: object) -> None:
            del args

    def guarded_scandir(path: object) -> object:
        if path == skill.parent:
            return ParentEntries(skill.parent)
        return original_scandir(path)

    monkeypatch.setattr(publication_identity.os, "scandir", guarded_scandir)

    identity = publication_target_from_path(skill)

    assert identity is not None
    assert identity["skill_name"] == skill.name


@pytest.mark.skipif(os.name != "nt", reason="native Windows fallback diagnostics")
def test_native_windows_fallback_manifest_exposes_capture_errors(tmp_path: Path) -> None:
    skill = _skill(tmp_path)

    entries = publication_identity._fallback_manifest(skill)
    digest = publication_identity._manifest_digest(entries)

    assert publication_source_digest(skill) == digest
    assert publication_target_from_path(skill) is not None


def test_publication_target_uses_filesystem_entry_spelling_for_case_alias(tmp_path: Path) -> None:
    skill = _skill(tmp_path, "Demo")
    alias = skill.with_name("demo")
    if not alias.exists() or not alias.samefile(skill):
        pytest.skip("filesystem treats differently cased paths as distinct")

    assert publication_target_from_path(alias) == publication_target_from_path(skill)


@pytest.mark.parametrize("claim_location", ["metadata", "payload", "summary"])
@pytest.mark.parametrize(
    "existing_claim",
    [
        None,
        {
            "skill_name": "other",
            "skill_digest": "sha256:" + "0" * 64,
            "skill_digest_algorithm": PUBLICATION_TARGET_DIGEST_ALGORITHM,
        },
    ],
)
def test_stamp_is_atomic_and_rejects_any_preexisting_identity_conflict(
    tmp_path: Path,
    claim_location: str,
    existing_claim: object,
) -> None:
    skill = _skill(tmp_path)
    clean = ValidationResult(validator_name="SCHEMA")
    conflicting = ValidationResult(validator_name="AGENT_EVAL")
    conflicting.metadata["agent_eval"] = {"summary": {}}
    containers = {
        "metadata": conflicting.metadata,
        "payload": conflicting.metadata["agent_eval"],
        "summary": conflicting.metadata["agent_eval"]["summary"],
    }
    containers[claim_location]["publication_target"] = existing_claim
    before = copy.deepcopy(conflicting.metadata)

    with pytest.raises(PublicationTargetConflictError, match="conflicting publication target"):
        stamp_publication_target([clean, conflicting], skill)

    assert clean.metadata == {}
    assert conflicting.metadata == before


def test_stamp_fills_missing_claims_when_existing_claims_match(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    expected = publication_target_from_path(skill)
    assert expected is not None
    tier3 = ValidationResult(validator_name="AGENT_EVAL")
    tier3.metadata = {
        "publication_target": dict(expected),
        "agent_eval": {"publication_target": dict(expected), "summary": {}},
    }

    stamped = stamp_publication_target([tier3], skill)

    assert stamped == expected
    assert tier3.metadata["publication_target"] == expected
    assert tier3.metadata["agent_eval"]["publication_target"] == expected
    assert tier3.metadata["agent_eval"]["summary"]["publication_target"] == expected


def test_finalize_stamps_only_an_unchanged_source_snapshot(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    initial = publication_target_from_path(skill)
    assert initial is not None
    stable = ValidationResult(validator_name="SCHEMA")

    assert finalize_publication_target([stable], skill, initial) == initial
    assert stable.metadata["publication_target"] == initial

    changed = ValidationResult(validator_name="SCHEMA")
    (skill / "SKILL.md").write_text("---\nname: changed\n---\n", encoding="utf-8")
    assert finalize_publication_target([changed], skill, initial) is None
    assert "publication_target" not in changed.metadata
    assert changed.metadata["publication_target_conflict"] == "source changed during validation"


def test_finalize_scrubs_all_identity_claims_when_source_changes(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    initial = publication_target_from_path(skill)
    assert initial is not None
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata = {
        "publication_target": dict(initial),
        "agent_eval": {
            "publication_target": dict(initial),
            "summary": {"publication_target": dict(initial)},
        },
    }
    (skill / "SKILL.md").write_text("---\nname: changed\n---\n", encoding="utf-8")

    assert finalize_publication_target([result], skill, initial) is None
    assert "publication_target" not in result.metadata
    assert "publication_target" not in result.metadata["agent_eval"]
    assert "publication_target" not in result.metadata["agent_eval"]["summary"]
    assert result.metadata["publication_target_conflict"] == "source changed during validation"


def test_finalize_scrubs_all_identity_claims_when_producer_identity_conflicts(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    initial = publication_target_from_path(skill)
    assert initial is not None
    conflicting = dict(initial)
    conflicting["skill_digest"] = "sha256:" + "0" * 64
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata = {
        "publication_target": dict(conflicting),
        "agent_eval": {
            "publication_target": dict(conflicting),
            "summary": {"publication_target": dict(conflicting)},
        },
    }

    assert finalize_publication_target([result], skill, initial) is None
    assert "publication_target" not in result.metadata
    assert "publication_target" not in result.metadata["agent_eval"]
    assert "publication_target" not in result.metadata["agent_eval"]["summary"]
    assert result.metadata["publication_target_conflict"] == "conflicting producer identity"


def test_tier1_and_tier2_producers_stamp_the_same_stable_snapshot(tmp_path: Path) -> None:
    from skillevaluator.tier1.commands import run_validation
    from skillevaluator.tier2.commands import _guarded_result

    skill = _skill(tmp_path)
    expected = publication_target_from_path(skill)
    assert expected is not None

    tier1 = run_validation(skill, checks="schema")
    tier2_result = ValidationResult(validator_name="Similarity Check")
    tier2_result.add_success("similarity", "Similarity scan completed")
    tier2 = _guarded_result("Similarity Check", skill, lambda: tier2_result, check_id="similarity")

    assert tier1 and tier2
    assert all(result.metadata["publication_target"] == expected for result in [*tier1, *tier2])
    assert {result.metadata["publication_evidence"]["check_id"] for result in tier1} == {"schema"}
    assert {result.metadata["publication_evidence"]["check_id"] for result in tier2} == {"similarity"}


def test_tier1_source_mutation_is_marked_unbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier1 import commands

    skill = _skill(tmp_path)

    class MutatingValidator:
        name = "Mutating schema"
        description = "Mutate the target while validating"

        @staticmethod
        def validate(target: Path) -> ValidationResult:
            (target / "SKILL.md").write_text("---\nname: changed\n---\n", encoding="utf-8")
            result = ValidationResult(validator_name="Mutating schema")
            result.add_success("schema", "Schema validation completed")
            return result

    monkeypatch.setattr(commands, "_schema_validator_for", lambda *_args, **_kwargs: MutatingValidator())

    (result,) = commands.run_validation(skill, checks="schema")

    assert "publication_target" not in result.metadata
    assert result.metadata["publication_target_conflict"] == "source changed during validation"


def test_split_tier_aggregation_binds_every_result_to_one_source_snapshot(tmp_path: Path) -> None:
    from scripts.ci import check_public_benchmarks as benchmark_gate

    from skillevaluator.reporting import BenchmarkReporter, JSONReporter
    from skillevaluator.tier1.commands import _as_result
    from skillevaluator.tier2.commands import _guarded_result

    skill = _skill(tmp_path)

    def successful_result(name: str, check: str) -> ValidationResult:
        result = ValidationResult(validator_name=name)
        result.add_success(check, f"{name} completed")
        return result

    def tier3_result(target: dict[str, str]) -> ValidationResult:
        run_id = "split-tier-fixture-run"
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "Live evaluation completed")
        result.metadata["publication_target"] = dict(target)
        result.metadata["agent_eval"] = {
            "skill_name": "demo",
            "verdict": "pass",
            "execution_status": "succeeded",
            "evaluated_at": "2026-08-25T12:00:00+00:00",
            "evaluator_version": "0.9.0",
            "expected_attempts": 1,
            "scored_attempts": 1,
            "dataset_summary": {"total_tasks": 1},
            "dataset_digest": "sha256:" + "a" * 64,
            "dataset_digest_algorithm": "skill-evaluator-dataset-snapshot/1",
            "attempt_policy": {"max_attempts": 1, "pass_threshold": 0.5},
            "run_id": run_id,
            "publication_target": dict(target),
            "summary": {
                "skill_name": "demo",
                "verdict": "pass",
                "execution_status": "succeeded",
                "environment": "docker",
                "expected_attempts": 1,
                "scored_attempts": 1,
                "run_id": run_id,
                "publication_target": dict(target),
            },
            "agents": {
                "codex": {
                    "model": "gpt-codex",
                    "execution_status": "succeeded",
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                    "with_skill": 0.9,
                    "dimensions": [
                        {"id": dimension, "with_skill": 0.9}
                        for dimension in (
                            "security",
                            "correctness",
                            "discoverability",
                            "effectiveness",
                            "efficiency",
                        )
                    ],
                }
            },
        }
        return result

    tier1 = _as_result(
        "SCHEMA",
        "Schema validation",
        lambda _target: successful_result("SCHEMA", "schema"),
        skill,
        publication_check_id="schema",
    )
    tier2 = _guarded_result(
        "Similarity Check",
        skill,
        lambda: successful_result("Similarity Check", "similarity"),
        check_id="similarity",
    )[0]
    unchanged_target = publication_target_from_path(skill)
    assert unchanged_target is not None
    tier3 = tier3_result(unchanged_target)

    benchmark = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all([tier1, tier2, tier3])
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: PASS" in benchmark
    assert offenders == []
    machine_report = json.loads(JSONReporter(include_timestamp=False).render_all([tier1, tier2, tier3]))
    assert [result["publication_target"] for result in machine_report["results"]] == [
        unchanged_target,
        unchanged_target,
        unchanged_target,
    ]
    assert machine_report["tier3"]["run_id"] == "split-tier-fixture-run"

    (skill / "SKILL.md").write_text("---\nname: changed-between-jobs\n---\n", encoding="utf-8")
    changed_target = publication_target_from_path(skill)
    assert changed_target is not None and changed_target != unchanged_target
    changed_tier2 = _guarded_result(
        "Similarity Check",
        skill,
        lambda: successful_result("Similarity Check", "similarity"),
        check_id="similarity",
    )[0]
    changed_tier3 = tier3_result(changed_target)

    mixed = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(
        [tier1, changed_tier2, changed_tier3]
    )

    assert "Overall verdict: INCOMPLETE" in mixed
    assert "## Publication Recommendation" not in mixed
