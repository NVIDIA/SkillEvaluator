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


def test_package_and_flat_import_eras_use_their_own_helpers(tmp_path, monkeypatch):
    """Two real refs cover both intermediate package imports and flat staging."""

    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "user.name=Baseline test", "-c", "user.email=baseline@example.invalid", *args],
            cwd=tmp_path,
            text=True,
        ).strip()

    git("init", "--quiet")
    core = tmp_path / "src/skillevaluator/tier3/eval_core"
    templates = tmp_path / "src/skillevaluator/tier3/harbor/templates"
    core.mkdir(parents=True)
    templates.mkdir(parents=True)
    (templates / "eval.py").write_text("from checks import _cmd_executes_script\n")
    (core / "checks.py").write_text(
        "from skillevaluator.tier3.eval_core import script_invocation\n"
        "_cmd_executes_script = script_invocation._cmd_executes_script\n"
    )
    parser = core / "script_invocation.py"
    parser.write_text("def _cmd_executes_script(command, script): return True\n")
    git("add", "src")
    git("commit", "--quiet", "-m", "package imports")
    package_ref = git("rev-parse", "HEAD")
    (core / "checks.py").write_text(
        "import script_invocation\n_cmd_executes_script = script_invocation._cmd_executes_script\n"
    )
    parser.write_text("def _cmd_executes_script(command, script): return False\n")
    git("add", "src")
    git("commit", "--quiet", "-m", "flat imports")
    flat_ref = git("rev-parse", "HEAD")
    monkeypatch.setattr(differential, "REPO_ROOT", tmp_path)
    candidate = differential.host_checks._script_invocation
    monkeypatch.setattr(candidate, "_cmd_executes_script", lambda *_args: None)
    from skillevaluator.tier3 import eval_core

    for ref, expected in [(package_ref, True), (flat_ref, False), (package_ref, True)]:
        host, template = differential._load_baseline(ref)
        assert host._cmd_executes_script("python calc.py", "calc.py") is expected
        assert template._cmd_executes_script("python calc.py", "calc.py") is expected
        assert eval_core.script_invocation is candidate
        assert candidate._cmd_executes_script("python calc.py", "calc.py") is None
