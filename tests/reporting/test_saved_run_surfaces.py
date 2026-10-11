# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved real runs, re-rendered: every surface gives the run the same headline.

Set ``SKILLEVALUATOR_SAVED_RUNS`` to a folder of saved ``validate`` runs (each ``<run>/reports/`` holding the JSON
report the run wrote, as the plugin-evaluation verification keeps them). Each run is rebuilt from its JSON report
and rendered by every reporter. Without the variable the test is skipped: the saved runs are not in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from _saved_run import load_saved_results, render_surfaces, saved_report

_ROOT = Path(os.environ.get("SKILLEVALUATOR_SAVED_RUNS", "")).expanduser()
_RUNS = (
    sorted(path for path in _ROOT.iterdir() if (path / "reports").is_dir())
    if os.environ.get("SKILLEVALUATOR_SAVED_RUNS") and _ROOT.is_dir()
    else []
)


@pytest.mark.skipif(not _RUNS, reason="SKILLEVALUATOR_SAVED_RUNS does not name a folder of saved runs")
@pytest.mark.parametrize("run_dir", _RUNS, ids=[path.name for path in _RUNS])
def test_saved_run_has_one_headline_on_every_surface(run_dir: Path) -> None:
    results = load_saved_results(saved_report(run_dir))
    gate = [result for result in results if (result.metadata.get("gating") or {}).get("blocking", True)]
    exit_code = 0 if all(result.passed for result in gate) else 1
    recorded_exit = (
        (run_dir / "run-meta.txt").read_text(encoding="utf-8") if (run_dir / "run-meta.txt").is_file() else ""
    )

    surfaces = render_surfaces(results, name=run_dir.parent.name or "plugin")
    headlines = surfaces["headlines"]

    # The rebuilt results gate the way the saved run exited.
    if "exit=" in recorded_exit:
        assert f"exit={exit_code}" in recorded_exit
    expected = headlines["cli-footer"]
    assert headlines == dict.fromkeys(headlines, expected)
    assert (expected == "passed") == (exit_code == 0)
    if expected in {"failed", "incomplete"}:
        assert surfaces["benchmark"] == {"failed": "FAIL", "incomplete": "INCOMPLETE"}[expected]
    # A Tier 1 that failed reads FAIL on its card, whatever else did not complete.
    tier1 = [result for result in results if (result.metadata.get("gating") or {}).get("tier") == 1]
    tier1_failed = any(
        not result.passed and any(f.severity.value in {"critical", "high"} for f in result.findings) for result in tier1
    )
    if tier1_failed:
        assert surfaces["html_tier_cards"]["tier1"] == "FAIL"
