# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unreadable artifacts do not prevent recovery from a later agent log."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core.log_converters import load_trajectory_with_fallback
from skillevaluator.tier3.harbor.adapter import _copy_verifier

LOG_NAMES = ("claude-code.txt", "cursor-cli.txt", "cline.txt", "opencode.txt", "codex.txt")
CODEX_LOG = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Recovered answer"}})


def test_invalid_utf8_trajectory_uses_separate_log_directory(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_bytes(b'{"steps": ["\xff"]}')
    logs = tmp_path / "agent"
    logs.mkdir()
    (logs / "cursor-cli.txt").write_text("Ran: cat skills/demo/SKILL.md\n", encoding="utf-8")

    data, meta = load_trajectory_with_fallback(trajectory, logs)

    assert data and data["steps"]
    assert meta["source"] == "cursor-cli.txt"
    assert meta["warning"].startswith("trajectory.json unreadable:")


@pytest.mark.parametrize("filename", LOG_NAMES)
def test_unreadable_log_without_an_alternative_reports_warning(tmp_path: Path, filename: str) -> None:
    (tmp_path / filename).mkdir()

    data, meta = load_trajectory_with_fallback(tmp_path / "trajectory.json")

    assert data is None
    assert meta["source"] is None
    assert meta["note"] is None
    assert meta["warning"].startswith(f"{filename} unreadable:")


@pytest.mark.parametrize("filename", LOG_NAMES[:-1])
def test_unreadable_preferred_log_does_not_hide_codex(tmp_path: Path, filename: str) -> None:
    (tmp_path / filename).mkdir()
    (tmp_path / "codex.txt").write_text(CODEX_LOG, encoding="utf-8")

    data, meta = load_trajectory_with_fallback(tmp_path / "trajectory.json")

    assert data and data["steps"][0]["message"] == "Recovered answer"
    assert meta["source"] == "codex.txt"
    assert meta["warning"].startswith(f"{filename} unreadable:")


def test_recovery_retains_warnings_for_all_unreadable_candidates(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text("{unfinished", encoding="utf-8")
    for filename in LOG_NAMES[:-1]:
        (tmp_path / filename).mkdir()
    (tmp_path / "codex.txt").write_text(CODEX_LOG, encoding="utf-8")

    data, meta = load_trajectory_with_fallback(trajectory)

    assert data and meta["source"] == "codex.txt"
    for filename in ("trajectory.json", *LOG_NAMES[:-1]):
        assert f"{filename} unreadable:" in meta["warning"]


def test_missing_artifacts_keep_empty_metadata(tmp_path: Path) -> None:
    assert load_trajectory_with_fallback(tmp_path / "trajectory.json") == (
        None,
        {"source": None, "warning": None, "note": None},
    )


def test_valid_trajectory_still_precedes_unreadable_logs(tmp_path: Path) -> None:
    trajectory = {"steps": [{"source": "agent", "message": "Canonical answer"}]}
    (tmp_path / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    for filename in LOG_NAMES:
        (tmp_path / filename).mkdir()

    assert load_trajectory_with_fallback(tmp_path / "trajectory.json") == (
        trajectory,
        {"source": "trajectory.json", "warning": None, "note": None},
    )


def test_staged_verifier_recovers_without_an_installed_package(tmp_path: Path) -> None:
    _copy_verifier(tmp_path)
    staged = tmp_path / "tests" / "skill_evaluator"
    logs = tmp_path / "agent"
    logs.mkdir()
    (logs / "trajectory.json").write_bytes(b"\xff")
    (logs / "claude-code.txt").mkdir()
    (logs / "codex.txt").write_text(CODEX_LOG, encoding="utf-8")
    script = """
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from log_converters import load_trajectory_with_fallback
print(json.dumps(load_trajectory_with_fallback(Path(sys.argv[2]))))
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(staged), str(logs / "trajectory.json")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    data, meta = json.loads(result.stdout)
    assert data["steps"][0]["message"] == "Recovered answer"
    assert meta["source"] == "codex.txt"
    assert "trajectory.json unreadable:" in meta["warning"]
    assert "claude-code.txt unreadable:" in meta["warning"]
