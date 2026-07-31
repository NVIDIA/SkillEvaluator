# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for per-entry eval input staging.

Ports SkillEvaluator 0.7.22 ``0d17f5e`` ("upload staged eval inputs to standard sandboxes")
into the in-process Tier 3 engine (``tier3/harbor/adapter.py``). The adapter now
stages both the shared ``evals/files/`` directory and each entry's ``files`` refs
into the task ``input/`` dir, with traversal protection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.adapter import (
    _entry_file_refs,
    _resolve_entry_file_ref,
    _stage_task_inputs,
)


def _make_skill(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill = tmp_path / "myskill"
    evals = skill / "evals"
    files = evals / "files"
    files.mkdir(parents=True)
    (files / "global.txt").write_text("global")
    data = evals / "data"
    data.mkdir()
    (data / "case1.txt").write_text("case1")
    env_dir = tmp_path / "task" / "environment"
    env_dir.mkdir(parents=True)
    return skill, evals, env_dir


class TestEntryFileRefs:
    def test_none_returns_empty(self):
        assert _entry_file_refs({"id": "t"}) == []

    def test_string_is_wrapped(self):
        assert _entry_file_refs({"files": "data/case1.txt"}) == ["data/case1.txt"]

    def test_list_is_passed_through(self):
        assert _entry_file_refs({"files": ["a/b.txt", " c/d.txt "]}) == ["a/b.txt", "c/d.txt"]

    def test_non_string_entry_rejected(self):
        with pytest.raises(ValueError, match="must be a string"):
            _entry_file_refs({"id": "t", "files": [123]})


class TestStageTaskInputs:
    def test_stages_global_files_and_entry_refs(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        entry = {"id": "t1", "files": ["data/case1.txt"]}
        staged = _stage_task_inputs(
            env_dir, input_files_dir=evals / "files", entry=entry, source_skill_path=skill, evals_dir=evals
        )
        assert staged is True
        names = sorted(p.name for p in (env_dir / "input").rglob("*") if p.is_file())
        assert names == ["case1.txt", "global.txt"]

    def test_no_inputs_returns_false(self, tmp_path: Path):
        skill = tmp_path / "myskill"
        evals = skill / "evals"
        evals.mkdir(parents=True)
        env_dir = tmp_path / "task" / "environment"
        env_dir.mkdir(parents=True)
        staged = _stage_task_inputs(
            env_dir, input_files_dir=None, entry={"id": "t"}, source_skill_path=skill, evals_dir=evals
        )
        assert staged is False


class TestResolveEntryFileRef:
    def test_traversal_outside_evals_blocked(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        with pytest.raises((ValueError, FileNotFoundError)):
            _resolve_entry_file_ref(
                "../../etc/passwd", skill_path=skill, evals_dir=evals, input_files_dir=evals / "files"
            )

    def test_absolute_path_rejected(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        absolute_ref = str(Path(tmp_path.anchor) / "outside.txt")
        with pytest.raises(ValueError, match="relative to evals/"):
            _resolve_entry_file_ref(absolute_ref, skill_path=skill, evals_dir=evals, input_files_dir=None)

    def test_uri_scheme_rejected(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        with pytest.raises(ValueError, match="unsupported URI scheme"):
            _resolve_entry_file_ref("https://example.com/x", skill_path=skill, evals_dir=evals, input_files_dir=None)
