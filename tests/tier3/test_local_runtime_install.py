# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

from skillevaluator.tier3.harbor import local_runtime
from skillevaluator.tier3.harbor.local_runtime import install_local_runtime, managed_install_argv


def test_install_argv_pins_release_under_managed_root(tmp_path):
    argv = managed_install_argv("claude-code", runtime_root=tmp_path)
    assert argv[:5] == ["npm", "install", "--global", "--prefix", str(tmp_path.resolve() / "claude-code")]
    assert argv[-1] == "@anthropic-ai/claude-code@2.1.292"
    assert managed_install_argv("opencode", runtime_root=tmp_path)[-1] == "opencode-ai"


def test_install_without_npm(monkeypatch, tmp_path):
    monkeypatch.setattr(local_runtime.shutil, "which", lambda _name, **_kwargs: None)
    assert "npm not found" in install_local_runtime("codex", runtime_root=tmp_path)


def _fake_npm(monkeypatch, returncode, create_bin):
    calls = []

    def run(argv, **_kwargs):
        calls.append(argv)
        if create_bin:
            bin_dir = Path(argv[4]) / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "codex").write_text("#!/bin/sh\n")
            (bin_dir / "codex").chmod(0o755)
        return subprocess.CompletedProcess(argv, returncode, "", "E404 not found\n")

    real_which = local_runtime.shutil.which
    monkeypatch.setattr(
        local_runtime.shutil,
        "which",
        lambda name, **kwargs: "/usr/bin/npm" if name == "npm" else real_which(name, **kwargs),
    )
    monkeypatch.setattr(local_runtime.subprocess, "run", run)
    return calls


def test_install_success_finds_cli(monkeypatch, tmp_path):
    calls = _fake_npm(monkeypatch, 0, create_bin=True)
    assert install_local_runtime("codex", runtime_root=tmp_path) is None
    assert calls[0][0] == "/usr/bin/npm" and calls[0][-1] == "@openai/codex@0.161.0"


def test_install_failure_reports_npm_error(monkeypatch, tmp_path):
    _fake_npm(monkeypatch, 1, create_bin=False)
    assert install_local_runtime("codex", runtime_root=tmp_path) == "npm install failed (exit 1): E404 not found"


def test_install_reports_missing_cli_after_success(monkeypatch, tmp_path):
    _fake_npm(monkeypatch, 0, create_bin=False)
    assert "no codex command was found" in install_local_runtime("codex", runtime_root=tmp_path)
