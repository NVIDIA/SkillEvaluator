# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable discovery for completed Tier 3 result directories."""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.tier3.results_location import external_results_root, resolve_latest_results


def _write_completed_run(root: Path, run_id: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps({"run_id": run_id, "agents": {}}),
        encoding="utf-8",
    )
    return run_dir


def test_latest_results_falls_back_to_newest_completed_run_without_symlink(tmp_path: Path) -> None:
    """Windows users can discover runs when creating ``latest`` is not permitted."""
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    _write_completed_run(root, "20260705_120000")
    newest = _write_completed_run(root, "20260705_130000")

    resolved = resolve_latest_results(skill_path, cli_results_dir, environ={})

    assert resolved == newest


def test_latest_results_fallback_ignores_hidden_partial_and_malformed_directories(tmp_path: Path) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    expected = _write_completed_run(root, "20260705_120000")

    _write_completed_run(root, ".20260705_180000")
    _write_completed_run(root, "_20260705_170000")
    _write_completed_run(root, "not-a-timestamp")
    (root / "20260705_160000").mkdir()
    malformed = root / "20260705_150000"
    malformed.mkdir()
    (malformed / "result.json").write_text("{not json", encoding="utf-8")
    mismatched = root / "20260705_140000"
    mismatched.mkdir()
    (mismatched / "result.json").write_text(
        json.dumps({"run_id": "20260705_999999", "agents": {}}),
        encoding="utf-8",
    )

    resolved = resolve_latest_results(skill_path, cli_results_dir, environ={})

    assert resolved == expected
