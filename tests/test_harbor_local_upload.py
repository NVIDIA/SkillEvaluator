# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verifier uploads must publish a complete validated tree, never an empty stub."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

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
def test_upload_replaces_verifier_files_and_rewrites_only_the_copy(tmp_path: Path, preexisting: bool) -> None:
    environment = _environment(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    script = source / "test.sh"
    script.write_text("#!/bin/sh\ncat /tests/entry.json\n", encoding="utf-8")
    (source / "entry.json").write_text('{"id":"example"}\n', encoding="utf-8")
    if preexisting:
        environment._tests.mkdir(parents=True)
        (environment._tests / "stale.txt").write_text("old verifier", encoding="utf-8")

    asyncio.run(environment.upload_dir(source, "/tests"))

    assert (environment._tests / "entry.json").read_text(encoding="utf-8") == '{"id":"example"}\n'
    assert str(environment._tests / "entry.json") in (environment._tests / "test.sh").read_text(encoding="utf-8")
    assert script.read_text(encoding="utf-8") == "#!/bin/sh\ncat /tests/entry.json\n"
    assert not (environment._tests / "stale.txt").exists()


def test_empty_directory_upload_still_creates_the_destination(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    source = tmp_path / "source"
    source.mkdir()

    asyncio.run(environment.upload_dir(source, "/tests"))

    assert environment._tests.is_dir()
    assert list(environment._tests.iterdir()) == []
