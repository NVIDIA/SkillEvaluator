# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated verifier scripts must keep LF line endings on every host.

``test.sh`` and the grader starters run inside a Linux container. On native
Windows, ``Path.write_text`` without ``newline=`` translates ``\\n`` to
``\\r\\n``, so ``#!/bin/bash`` becomes ``#!/bin/bash\\r`` and
``set -euo pipefail`` fails on ``pipefail\\r``. These tests emulate that
Windows default on any host so the regression is caught on POSIX CI too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.tier3 import commands
from skillevaluator.tier3.harbor.adapter import _write_test_sh


@pytest.fixture
def windows_text_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``Path.write_text`` translate newlines like it does on Windows."""
    original = Path.write_text

    def write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        return original(self, data, encoding=encoding, errors=errors, newline="\r\n" if newline is None else newline)

    monkeypatch.setattr(Path, "write_text", write_text)


def _assert_lf_script(path: Path, shebang: bytes) -> None:
    raw = path.read_bytes()
    assert b"\r" not in raw, f"{path.name} must not contain CR line endings"
    assert raw.startswith(shebang + b"\n")


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "sample-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_bytes(b"---\nname: sample-skill\ndescription: Test fixture.\n---\n# Sample\n")
    return skill


def test_windows_text_mode_emulation_translates_default_newlines(tmp_path: Path, windows_text_mode: None) -> None:
    probe = tmp_path / "probe.txt"
    probe.write_text("a\nb\n", encoding="utf-8")

    assert probe.read_bytes() == b"a\r\nb\r\n"


@pytest.mark.parametrize(
    ("grading_mode", "custom_grader"),
    [("default", False), ("default_plus_custom", True), ("custom_only", True)],
)
def test_harbor_task_test_sh_is_lf_only(
    tmp_path: Path, windows_text_mode: None, grading_mode: str, custom_grader: bool
) -> None:
    _write_test_sh(tmp_path, grading_mode=grading_mode, custom_grader=custom_grader)

    _assert_lf_script(tmp_path / "tests" / "test.sh", b"#!/bin/bash")


@pytest.mark.parametrize(
    ("language", "name", "shebang"),
    [("python", "grader.py", b"#!/usr/bin/env python3"), ("shell", "grader.sh", b"#!/usr/bin/env bash")],
)
def test_custom_grader_starter_is_lf_only(
    tmp_path: Path, windows_text_mode: None, language: str, name: str, shebang: bytes
) -> None:
    skill = _skill(tmp_path)

    assert (
        commands.init_custom_grader(
            skill,
            mode="default_plus_custom",
            language=language,
            force=False,
            no_config=True,
        )
        == 0
    )

    _assert_lf_script(skill / "evals" / name, shebang)


@pytest.mark.parametrize(
    ("language", "name", "shebang"),
    [("python", "grader.py", b"#!/usr/bin/env python3"), ("shell", "grader.sh", b"#!/usr/bin/env bash")],
)
def test_harbor_task_scaffold_scripts_are_lf_only(
    tmp_path: Path, windows_text_mode: None, language: str, name: str, shebang: bytes
) -> None:
    skill = _skill(tmp_path)

    assert (
        commands.init_harbor_task(
            skill,
            force=False,
            case_id="case-001",
            mode="custom_only",
            language=language,
            with_config=False,
        )
        == 0
    )

    tests_dir = skill / "evals" / "harbor" / "case-001" / "tests"
    _assert_lf_script(tests_dir / "test.sh", b"#!/bin/bash")
    _assert_lf_script(tests_dir / name, shebang)
