# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-run comparison must consume only successful evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from skillevaluator.tier3.commands import compare_results

if TYPE_CHECKING:
    import pytest


def test_compare_rejects_partial_scores_from_failed_summary(tmp_path: Path) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    results_root = tmp_path / "results"
    summary_dir = results_root / "demo" / "20260709_010000" / "opencode" / "with-skill"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "execution_status": "failed",
                "execution_errors": ["Scored attempt coverage is 3/4"],
                "scored_attempts": 3,
                "expected_attempts": 4,
                "scores": {"security": 1.0, "accuracy": 0.8},
            }
        ),
        encoding="utf-8",
    )

    assert compare_results(skill_path, results_dir=results_root) == 1


def test_compare_ignores_failed_baseline_scores(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    results_root = tmp_path / "results"
    agent_dir = results_root / "demo" / "20260709_010000" / "opencode"
    for variant, status, score in (
        ("with-skill", "succeeded", 1.0),
        ("without-skill", "failed", 0.1),
    ):
        summary_dir = agent_dir / variant
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(
            json.dumps({"execution_status": status, "scores": {"security": score}}),
            encoding="utf-8",
        )

    assert compare_results(skill_path, results_dir=results_root) == 0
    assert "lift" not in capsys.readouterr().out.lower()


def test_compare_never_pairs_baseline_from_a_different_timestamp(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    results_root = tmp_path / "results"
    skill_results = results_root / "demo"

    newest = skill_results / "20260709_020000" / "opencode"
    for variant, status, score in (
        ("with-skill", "failed", 0.9),
        ("without-skill", "succeeded", 0.2),
    ):
        summary_dir = newest / variant
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(
            json.dumps({"execution_status": status, "scores": {"security": score}}),
            encoding="utf-8",
        )

    older = skill_results / "20260709_010000" / "opencode" / "with-skill"
    older.mkdir(parents=True)
    (older / "summary.json").write_text(
        json.dumps({"execution_status": "succeeded", "scores": {"security": 1.0}}),
        encoding="utf-8",
    )

    assert compare_results(skill_path, results_dir=results_root) == 0
    assert "lift" not in capsys.readouterr().out.lower()
