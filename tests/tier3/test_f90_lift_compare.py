# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H5 follow-up: ``tier3 compare`` does not score the no-skill arm's N/A skill metrics as 0.0.

Proof example check-13 edge-04 (activation only): both arms got the same judge
verdicts and only the with-skill arm could activate a skill. The no-skill
summary records ``skill_execution`` and ``skill_efficiency`` as not applicable,
with no score. ``compare`` read them as 0.0, so it showed a +0.98
skill_execution lift and a +0.29 Overall lift for activation alone.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from _f90_lift_fixtures import collect, metrics

from skillevaluator.tier3.commands import compare_results

if TYPE_CHECKING:
    import pytest

CASES = [f"case-{index}" for index in range(1, 7)]


def _run_with_real_summaries(tmp_path: Path) -> Path:
    """Collect an activation-only run, then lay its summaries out as a results directory."""
    plugin = metrics(0.4, skill_execution=1.0, skill_efficiency=0.75)
    no_plugin = metrics(0.4, skill=False)
    arms = {
        "with": {case: [plugin, plugin] for case in CASES},
        "without": {case: [no_plugin, no_plugin] for case in CASES},
    }
    collect(tmp_path, {"claude-code": arms}, n_attempts=2)
    without = json.loads((tmp_path / "results" / "claude-code" / "without-skill" / "summary.json").read_text())
    assert without["not_applicable_metrics"] == ["skill_execution", "skill_efficiency"]
    assert "skill_execution" not in without["scores"]

    run_dir = tmp_path / "compare" / "demo" / "20261004_120000"
    for variant in ("with-skill", "without-skill"):
        target = run_dir / "claude-code" / variant
        target.mkdir(parents=True)
        shutil.copy(tmp_path / "results" / "claude-code" / variant / "summary.json", target / "summary.json")
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_dir.name, "agents": {}}), encoding="utf-8")
    return tmp_path / "compare"


def _row(output: str, label: str) -> list[str]:
    line = next(line for line in output.splitlines() if re.search(rf"\b{label}\b", line))
    return line.replace("│", " ").split()


def test_compare_shows_no_lift_for_metrics_the_no_skill_arm_cannot_score(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = _run_with_real_summaries(tmp_path)
    skill_path = tmp_path / "demo"
    skill_path.mkdir()

    assert compare_results(skill_path, results_dir=results_root) == 0
    output = capsys.readouterr().out

    # The with-skill score still shows; the lift is N/A, not +1.00 against a 0.0 baseline.
    assert _row(output, "skill_execution")[1:] == ["1.00", "N/A"]
    assert _row(output, "skill_efficiency")[1:] == ["0.75", "N/A"]
    assert _row(output, "security")[1:] == ["0.40", "-"]
    # Overall: the with-skill score over every metric it scored, the lift only over the
    # metrics both arms scored (the same judge verdicts, so no lift).
    overall = _row(output, "Overall")
    assert overall[1] == "0.56"
    assert overall[2] == "0.00"
