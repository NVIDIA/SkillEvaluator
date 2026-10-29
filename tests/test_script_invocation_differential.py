# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A historical baseline must use its own shared parser without import leaks."""

from __future__ import annotations

import subprocess
import sys

import pytest
from scripts import script_invocation_differential as differential


def test_current_git_baseline_loads_its_own_shared_modules(monkeypatch):
    if subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=differential.REPO_ROOT, capture_output=True
    ).returncode:
        pytest.skip("source archive has no git baseline")
    previous_path = sys.path.copy()
    previous_modules = sys.modules.copy()
    monkeypatch.setattr(
        differential.host_checks._script_invocation, "_cmd_executes_script", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(differential.host_checks._shell_lexing, "_shell_tokens", lambda _command: [])
    host, template = differential._load_baseline("HEAD")
    other_host, other_template = differential._load_baseline("HEAD")

    assert sys.path == previous_path
    for name in (
        "checks",
        "baseline_template",
        "script_invocation",
        "shell_lexing",
        "secret_redaction",
        "skillevaluator.tier3.eval_core.script_invocation",
        "skillevaluator.tier3.eval_core.shell_lexing",
    ):
        assert sys.modules.get(name) is previous_modules.get(name)
    assert host._cmd_executes_script is not differential.host_checks._cmd_executes_script
    assert template._cmd_executes_script is host._cmd_executes_script
    assert host._cmd_executes_script is not other_host._cmd_executes_script
    assert other_template._cmd_executes_script is other_host._cmd_executes_script
    for command, expected in [("python calc.py", True), ("cat calc.py", False), ("sh -c 'python calc.py'", True)]:
        assert host._cmd_executes_script(command, "calc.py") is expected
        assert template._cmd_executes_script(command, "calc.py") is expected
        assert other_host._cmd_executes_script(command, "calc.py") is expected


def test_invalid_git_baseline_does_not_change_imports():
    previous_path = sys.path.copy()
    previous_checker = sys.modules.get("checks")
    with pytest.raises(subprocess.CalledProcessError):
        differential._load_baseline("refs/invalid/missing-baseline")
    assert sys.path == previous_path
    assert sys.modules.get("checks") is previous_checker
