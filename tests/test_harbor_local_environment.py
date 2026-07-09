# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment


def test_local_environment_rewrites_read_only_uploaded_script_once(tmp_path: Path) -> None:
    """A read-only template is made writable without rewriting its local path twice."""
    source = tmp_path / "template_eval.py"
    source.write_text('print("/tests/eval.py")\n', encoding="utf-8")
    source.chmod(0o444)

    local_tests = tmp_path / "trial" / "local-environment" / "tests"
    uploaded = local_tests / "eval.py"
    uploaded.parent.mkdir(parents=True)
    shutil.copy2(source, uploaded)

    should_restore_owner_write = not os.access(uploaded, os.W_OK)
    environment = SkillEvaluatorLocalEnvironment.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/tests", local_tests)]

    environment._rewrite_uploaded_script(uploaded)

    assert uploaded.read_text(encoding="utf-8") == f'print("{local_tests}{os.sep}eval.py")\n'
    mode = uploaded.stat().st_mode
    if should_restore_owner_write:
        assert mode & stat.S_IWUSR
    if os.name == "posix":
        assert not mode & stat.S_IWGRP
        assert not mode & stat.S_IWOTH


def test_raw_path_rewrite_respects_container_root_boundaries(tmp_path: Path) -> None:
    local_tests = tmp_path / "local" / "tests"
    environment = SkillEvaluatorLocalEnvironment.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/tests", local_tests)]

    rewritten = environment._rewrite_raw_paths('"/tests/eval.py" "/tests" "/testsuite" "/tests-v2"')

    assert rewritten == f'"{local_tests}{os.sep}eval.py" "{local_tests}" "/testsuite" "/tests-v2"'


def test_raw_path_rewrite_ignores_url_and_local_path_suffixes(tmp_path: Path) -> None:
    local_tests = tmp_path / "local" / "tests"
    environment = SkillEvaluatorLocalEnvironment.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/tests", local_tests)]
    value = f'"/tests/api" "https://example.invalid/tests/api" "{local_tests}/api"'

    rewritten = environment._rewrite_raw_paths(value)

    assert rewritten == f'"{local_tests}{os.sep}api" "https://example.invalid/tests/api" "{local_tests}/api"'
