# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the harbor CLI gate in Tier 3 dataset generation."""

import sys
from unittest.mock import patch

import pytest

from skillevaluator.tier3.generate_dataset import _run_agent_collect_trajectories


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake console script")
def test_harbor_gate_finds_tool_venv_harbor(tmp_path, capsys, monkeypatch):
    """A harbor console script next to the interpreter passes the gate.

    Under ``uv tool install`` / ``pipx``, harbor lands in the tool venv's
    bin directory, which is not on PATH; the gate must not skip the agent
    run in that layout.
    """
    harbor = tmp_path / "harbor"
    harbor.write_text("#!/bin/sh\nexit 0\n")
    harbor.chmod(0o755)
    empty_path_dir = tmp_path / "empty-path"
    empty_path_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_path_dir))

    with patch.object(sys, "executable", str(tmp_path / "python")):
        result = _run_agent_collect_trajectories(tmp_path, [])

    out = capsys.readouterr().out
    assert "harbor CLI not found" not in out
    # With an empty PATH the gate must advance to (and stop at) the Docker check.
    assert "Docker not found" in out
    assert result == {}
