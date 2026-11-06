# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The host and packaged verifier cross one shell-lexing seam."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core import checks, shell_lexing
from skillevaluator.tier3.harbor.adapter import _copy_verifier


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat 2>/dev/null", ["cat", "2>", "/dev/null"]),
        ("cat 2 > out", ["cat", "2", ">", "out"]),
        ("echo 'if'", ["echo", "\ue000if"]),
        ("echo one\necho two", ["echo", "one", ";", "echo", "two"]),
        ('echo "unterminated', []),
    ],
)
def test_existing_token_contract(command, expected):
    assert shell_lexing._shell_tokens(command) == expected


def test_both_callers_use_the_shared_implementation(monkeypatch):
    # Use a deliberate sibling alias; a previous template import must not make
    # this test dependent on sys.modules or sys.path ordering. The next test
    # verifies real file staging in an isolated interpreter.
    monkeypatch.setitem(sys.modules, "shell_lexing", shell_lexing)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    path = Path(__file__).parents[2] / "src/skillevaluator/tier3/harbor/templates/eval.py"
    spec = importlib.util.spec_from_file_location("shell_lexing_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert checks._shell_tokens is shell_lexing._shell_tokens
    assert module._shell_tokens is shell_lexing._shell_tokens
    assert checks._QUOTED_SYNTAX_MARK == module._QUOTED_SYNTAX_MARK


def test_real_staging_copies_a_standalone_lexer(tmp_path):
    _copy_verifier(tmp_path)
    staged = tmp_path / "tests" / "skill_evaluator" / "shell_lexing.py"
    assert staged.read_bytes() == Path(shell_lexing.__file__).read_bytes()
    # -I and -S exclude PYTHONPATH and site packages. Only the copied stdlib
    # module is loaded, so this cannot accidentally import the host package.
    program = (
        "import json,runpy,sys; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['_shell_tokens']('cat 2>/dev/null')))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, str(staged)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == ["cat", "2>", "/dev/null"]


def test_staged_invocation_analysis_uses_only_staged_helpers(tmp_path):
    from skillevaluator.tier3.eval_core import script_invocation

    _copy_verifier(tmp_path)
    staged = tmp_path / "tests" / "skill_evaluator"
    assert (staged / "script_invocation.py").read_bytes() == Path(script_invocation.__file__).read_bytes()
    assert checks._cmd_executes_script is script_invocation._cmd_executes_script
    program = (
        "import json,sys; sys.path.insert(0,sys.argv[1]); "
        "import script_invocation as module; "
        "commands=['python calc.py','cat calc.py',\"sh -c 'python calc.py'\"]; "
        "print(json.dumps([module._cmd_executes_script(c,'calc.py') for c in commands]))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, str(staged)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == [True, False, True]
