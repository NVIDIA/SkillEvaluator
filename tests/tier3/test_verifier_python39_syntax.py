# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The staged Harbor verifier must run on the task image's Python 3.9."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.adapter import _copy_verifier


def _staged_modules(tmp_path: Path) -> list[Path]:
    _copy_verifier(tmp_path)
    modules = sorted((tmp_path / "tests/skill_evaluator").rglob("*.py"))
    assert modules, "the verifier staged no Python modules"
    return modules


def test_staged_verifier_parses_as_python39(tmp_path):
    for module in _staged_modules(tmp_path):
        source = module.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(module), feature_version=(3, 9))
        except SyntaxError as error:  # pragma: no cover - the failure message is the point
            pytest.fail(f"{module.name} does not parse as Python 3.9: {error}")


def test_staged_verifier_calls_no_zip_with_strict(tmp_path):
    # zip(..., strict=True) is new in Python 3.10; on 3.9 it raises
    # "TypeError: zip() takes no keyword arguments" when the call runs.
    offenders = []
    for module in _staged_modules(tmp_path):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "zip"
                and any(keyword.arg == "strict" for keyword in node.keywords)
            ):
                offenders.append(f"{module.name}:{node.lineno}")
    assert offenders == []
