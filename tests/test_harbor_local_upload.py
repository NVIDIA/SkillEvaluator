# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verifier uploads must publish a complete validated tree, never an empty stub."""

from __future__ import annotations

import asyncio
import errno
import os
import subprocess
from pathlib import Path

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from skillevaluator.tier3.harbor import local_sandbox, secure_copy
from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment
from skillevaluator.tier3.harbor.secure_copy import UnsafeStagingError


def _environment(tmp_path: Path) -> SkillEvaluatorLocalEnvironment:
    definition = tmp_path / "environment"
    definition.mkdir()
    return SkillEvaluatorLocalEnvironment(
        environment_dir=definition,
        environment_name="local-upload",
        session_id="local-upload",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        runtime_agent="codex",
        runtime_root=str(tmp_path / "runtime"),
        sandbox_mode="require",
    )


def test_rejected_upload_does_not_publish_an_empty_tests_directory(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not a verifier input", encoding="utf-8")
    try:
        (source / "linked.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(UnsafeStagingError):
        asyncio.run(environment.upload_dir(source, "/tests"))

    assert not environment._tests.exists()
    assert outside.read_text(encoding="utf-8") == "not a verifier input"


@pytest.mark.parametrize("preexisting", [False, True])
def test_upload_overlays_verifier_files_and_rewrites_only_the_copy(tmp_path: Path, preexisting: bool) -> None:
    environment = _environment(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    script = source / "test.sh"
    script.write_text("#!/bin/sh\ncat /tests/entry.json\n", encoding="utf-8")
    (source / "entry.json").write_text('{"id":"example"}\n', encoding="utf-8")
    if preexisting:
        environment._tests.mkdir(parents=True)
        (environment._tests / "shared.txt").write_text("shared input", encoding="utf-8")
        (environment._tests / "test.sh").write_text("old verifier", encoding="utf-8")

    asyncio.run(environment.upload_dir(source, "/tests"))

    assert (environment._tests / "entry.json").read_text(encoding="utf-8") == '{"id":"example"}\n'
    assert str(environment._tests / "entry.json") in (environment._tests / "test.sh").read_text(encoding="utf-8")
    assert script.read_text(encoding="utf-8") == "#!/bin/sh\ncat /tests/entry.json\n"
    if preexisting:
        assert (environment._tests / "shared.txt").read_text(encoding="utf-8") == "shared input"


@pytest.mark.parametrize("preexisting", [False, True])
def test_empty_directory_upload_preserves_existing_shared_files(tmp_path: Path, preexisting: bool) -> None:
    environment = _environment(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    if preexisting:
        environment._tests.mkdir(parents=True)
        (environment._tests / "shared.txt").write_text("shared input", encoding="utf-8")

    asyncio.run(environment.upload_dir(source, "/tests"))

    assert environment._tests.is_dir()
    if preexisting:
        assert (environment._tests / "shared.txt").read_text(encoding="utf-8") == "shared input"
    else:
        assert list(environment._tests.iterdir()) == []


@pytest.mark.parametrize("source_kind", ["missing", "broken-link", "linked-file", "regular-file"])
def test_invalid_replacement_preserves_previous_verifier(tmp_path: Path, source_kind: str) -> None:
    environment = _environment(tmp_path)
    environment._tests.mkdir(parents=True)
    previous = environment._tests / "test.sh"
    previous.write_text("previous verifier\n", encoding="utf-8")
    source = tmp_path / "source"
    if source_kind == "regular-file":
        source.write_text("not a directory", encoding="utf-8")
    elif source_kind in {"broken-link", "linked-file"}:
        try:
            if source_kind == "broken-link":
                source.symlink_to(tmp_path / "absent-directory", target_is_directory=True)
            else:
                source.mkdir()
                (source / "linked.txt").symlink_to(previous)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises((OSError, UnsafeStagingError)):
        asyncio.run(environment.upload_dir(source, "/tests"))

    assert previous.read_text(encoding="utf-8") == "previous verifier\n"
    assert [path.name for path in environment._tests.iterdir()] == ["test.sh"]


@pytest.mark.parametrize("broken_link", [False, True])
def test_missing_upload_source_fails_without_publishing_a_destination(tmp_path: Path, broken_link: bool) -> None:
    environment = _environment(tmp_path)
    source = tmp_path / "source"
    if broken_link:
        try:
            source.symlink_to(tmp_path / "absent-directory", target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises((OSError, UnsafeStagingError)):
        asyncio.run(environment.upload_dir(source, "/tests"))

    assert not environment._tests.exists()


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("copy_phase", [1, 2])
def test_late_copy_failure_preserves_previous_verifier_and_removes_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool, copy_phase: int
) -> None:
    if fallback:
        monkeypatch.setattr(secure_copy, "_DESCRIPTOR_BACKEND", False)
    elif not secure_copy._DESCRIPTOR_BACKEND:
        pytest.skip("descriptor backend unavailable")
    environment = _environment(tmp_path)
    environment._tests.mkdir(parents=True)
    (environment._tests / "test.sh").write_text("previous verifier\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "first.txt").write_bytes(b"first verifier input\n")
    (source / "second.txt").write_bytes(b"second verifier input\n")
    real_write = secure_copy.os.write
    phase = 0

    def disk_full(descriptor, data):
        nonlocal phase
        if bytes(data) == b"second verifier input\n":
            phase += 1
            if phase == copy_phase:
                raise OSError(errno.ENOSPC, "injected disk-full error")
        return real_write(descriptor, data)

    monkeypatch.setattr(secure_copy.os, "write", disk_full)
    with pytest.raises(OSError, match="disk-full"):
        asyncio.run(environment.upload_dir(source, "/tests"))

    assert (environment._tests / "test.sh").read_text(encoding="utf-8") == "previous verifier\n"
    assert [path.name for path in environment._tests.iterdir()] == ["test.sh"]
    assert [path.name for path in environment._root.iterdir()] == ["tests"]


def test_path_rewrite_failure_preserves_previous_verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment = _environment(tmp_path)
    environment._tests.mkdir(parents=True)
    previous = environment._tests / "test.sh"
    previous.write_text("previous verifier\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    script = source / "test.sh"
    script.write_text("cat /tests/entry.json\n", encoding="utf-8")
    (source / "entry.json").write_text("{}\n", encoding="utf-8")
    real_write = Path.write_text

    def disk_full_during_rewrite(path, data, *args, **kwargs):
        if path.name == "test.sh" and str(environment._tests / "entry.json") in data:
            real_write(path, "partial rewrite\n", *args, **kwargs)
            raise OSError(errno.ENOSPC, "injected rewrite disk-full error")
        return real_write(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", disk_full_during_rewrite)
    with pytest.raises(OSError, match="rewrite disk-full"):
        asyncio.run(environment.upload_dir(source, "/tests"))

    assert previous.read_text(encoding="utf-8") == "previous verifier\n"
    assert script.read_text(encoding="utf-8") == "cat /tests/entry.json\n"


def test_upload_rejects_a_target_outside_the_local_run(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("new input", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("existing data", encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        asyncio.run(environment.upload_dir(source, str(outside)))

    assert (outside / "keep.txt").read_text(encoding="utf-8") == "existing data"
    assert [path.name for path in outside.iterdir()] == ["keep.txt"]


def test_upload_rejects_a_linked_destination_without_touching_its_target(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    protected = environment._root / "protected"
    protected.mkdir(parents=True)
    (protected / "keep.txt").write_text("existing data", encoding="utf-8")
    try:
        environment._tests.symlink_to(protected, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("new input", encoding="utf-8")

    with pytest.raises(UnsafeStagingError):
        asyncio.run(environment.upload_dir(source, "/tests"))

    assert environment._tests.is_symlink()
    assert (protected / "keep.txt").read_text(encoding="utf-8") == "existing data"
    assert [path.name for path in protected.iterdir()] == ["keep.txt"]


@pytest.mark.skipif(os.name == "nt", reason="native Windows does not support the local Bash runtime")
def test_precreated_empty_target_and_read_only_script_in_a_path_with_spaces(tmp_path: Path) -> None:
    spaced = tmp_path / "path with spaces"
    spaced.mkdir()
    environment = _environment(spaced)
    environment._tests.mkdir(parents=True)
    source = spaced / "source"
    source.mkdir()
    script = source / "test.sh"
    script.write_text("cat /tests/entry.json\n", encoding="utf-8")
    script.chmod(0o444)
    (source / "entry.json").write_text("{}\n", encoding="utf-8")

    asyncio.run(environment.upload_dir(source, "/tests"))

    assert (environment._tests / "entry.json").read_text(encoding="utf-8") == "{}\n"
    result = subprocess.run(["bash", str(environment._tests / "test.sh")], capture_output=True, text=True, check=True)
    assert result.stdout == "{}\n"
    assert script.read_text(encoding="utf-8") == "cat /tests/entry.json\n"


@pytest.mark.integration
@pytest.mark.skipif(os.name == "nt", reason="native Windows local mode is unsupported")
@pytest.mark.parametrize("multi_step", [False, True])
@pytest.mark.parametrize("strict_reads", [False, True])
def test_real_harbor_verifier_reads_shared_and_step_inputs(
    tmp_path: Path, multi_step: bool, strict_reads: bool
) -> None:
    from harbor.models.task.task import Task
    from harbor.verifier.verifier import Verifier

    try:
        local_sandbox.detect("require")
    except local_sandbox.SandboxUnavailable as exc:
        pytest.skip(str(exc))
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    shared = task_dir / "tests"
    shared.mkdir()
    (shared / "shared.txt").write_text("shared\n", encoding="utf-8")
    script = (
        "#!/bin/sh\nset -eu\n"
        'test "$(cat /tests/shared.txt)" = shared\n'
        'test "$(cat /tests/step.txt)" = EXPECTED\n'
        "printf '1\\n' > /logs/verifier/reward.txt\n"
    )
    steps = ["first", "second"] if multi_step else [None]
    if multi_step:
        (task_dir / "task.toml").write_text('[[steps]]\nname="first"\n[[steps]]\nname="second"\n', encoding="utf-8")
        for name in steps:
            step = task_dir / "steps" / name
            (step / "tests").mkdir(parents=True)
            (step / "instruction.md").write_text("Read the supplied inputs.\n", encoding="utf-8")
            (step / "tests" / "step.txt").write_text(name + "\n", encoding="utf-8")
            (step / "tests" / "test.sh").write_text(script.replace("EXPECTED", name), encoding="utf-8")
    else:
        (task_dir / "task.toml").write_text("", encoding="utf-8")
        (task_dir / "instruction.md").write_text("Read the supplied inputs.\n", encoding="utf-8")
        (shared / "step.txt").write_text("single\n", encoding="utf-8")
        (shared / "test.sh").write_text(script.replace("EXPECTED", "single"), encoding="utf-8")
    task = Task(task_dir)
    paths = TrialPaths(tmp_path / "trial")
    environment = SkillEvaluatorLocalEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name="harbor-upload-integration",
        session_id="harbor-upload-integration",
        trial_paths=paths,
        task_env_config=task.config.environment,
        runtime_agent="codex",
        runtime_root=str(tmp_path / "runtime"),
        sandbox_mode="require",
        allow_net=False,
        strict_reads=strict_reads,
    )

    async def exercise():
        await environment.start()
        try:
            for name in steps:
                paths.reward_text_path.unlink(missing_ok=True)
                verifier = Verifier(task=task, trial_paths=paths, environment=environment, step_name=name)
                result = await verifier.verify()
                assert result.rewards == {"reward": 1.0}
        finally:
            await environment.stop(delete=True)

    asyncio.run(exercise())
