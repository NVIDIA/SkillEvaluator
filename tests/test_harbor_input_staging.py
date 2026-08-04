# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for per-entry eval input staging.

The adapter stages an entry's declared ``files`` exclusively. The shared
``evals/files/`` corpus is copied only for entries that omit ``files`` so legacy
datasets keep working without leaking unrelated fixtures into explicit cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.adapter import (
    _entry_file_refs,
    _resolve_entry_file_ref,
    _stage_task_inputs,
    _write_dockerfile,
)


def _make_skill(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill = tmp_path / "myskill"
    evals = skill / "evals"
    files = evals / "files"
    files.mkdir(parents=True)
    (files / "global.txt").write_text("global")
    (files / "unrelated.txt").write_text("unrelated")
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
    def test_declared_files_exclude_unrelated_corpus(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        entry = {"id": "t1", "files": ["data/case1.txt"]}
        staged = _stage_task_inputs(
            env_dir, input_files_dir=evals / "files", entry=entry, source_skill_path=skill, evals_dir=evals
        )
        assert staged is True
        names = sorted(p.name for p in (env_dir / "input").rglob("*") if p.is_file())
        assert names == ["case1.txt"]

    def test_declared_files_stage_only_explicit_refs(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        entry = {"id": "t1", "files": ["evals/files/global.txt", "data/case1.txt"]}

        staged = _stage_task_inputs(
            env_dir, input_files_dir=evals / "files", entry=entry, source_skill_path=skill, evals_dir=evals
        )

        assert staged is True
        paths = sorted(
            path.relative_to(env_dir / "input").as_posix()
            for path in (env_dir / "input").rglob("*")
            if path.is_file()
        )
        assert paths == ["data/case1.txt", "global.txt"]

    def test_missing_files_key_preserves_legacy_corpus(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)

        staged = _stage_task_inputs(
            env_dir, input_files_dir=evals / "files", entry={"id": "t1"}, source_skill_path=skill, evals_dir=evals
        )

        assert staged is True
        names = sorted(path.name for path in (env_dir / "input").rglob("*") if path.is_file())
        assert names == ["global.txt", "unrelated.txt"]

    @pytest.mark.parametrize("declared_files", [None, [], "", ["  "]])
    def test_explicit_empty_files_stage_nothing(self, tmp_path: Path, declared_files: object):
        skill, evals, env_dir = _make_skill(tmp_path)
        input_dir = env_dir / "input"
        input_dir.mkdir()
        (input_dir / "stale.txt").write_text("stale")

        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": declared_files},
            source_skill_path=skill,
            evals_dir=evals,
        )

        assert staged is False
        assert not input_dir.exists()

    def test_default_dockerfile_omits_copy_for_explicit_empty_files(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        task_dir = tmp_path / "generated" / "task"

        _write_dockerfile(
            task_dir,
            skill,
            reference_skills_dir=None,
            workspace_skill_paths=None,
            has_skill=True,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": []},
            evals_dir=evals,
        )

        environment = task_dir / "environment"
        assert not (environment / "input").exists()
        assert "COPY input/ /workspace/input/" not in (environment / "Dockerfile").read_text(encoding="utf-8")

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
