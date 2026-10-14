# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the p01 gate tests: a small plugin, synthetic Tier 3 results, and a ``validate`` runner.

Only the Tier 3 engine call is replaced, with the canonical result the shipped payload builder makes from synthetic
agent scores (the same shape the check-28 gate probe injects). Everything else is the real ``validate`` command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from click.testing import CliRunner

import skillevaluator.cli as cli_module
from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, build_agent_eval_payload

if TYPE_CHECKING:
    import pytest

    from skillevaluator.models import ValidationResult

METRICS = ("security", "skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check")

# check-13 edge-05: the with-plugin arm answers worse (accuracy, goal and behavior 0.5 vs 1.0) but every dimension
# stays at or above 0.50, so the dimension verdict is PASS while the lift is -0.20 with a 95% CI of [-0.20, -0.19].
EDGE05_WITH = {
    "security": 1.0,
    "skill_execution": 0.9769,
    "skill_efficiency": 0.697,
    "accuracy": 0.5,
    "goal_accuracy": 0.5,
    "behavior_check": 0.5,
}
EDGE05_WITHOUT = {
    "security": 1.0,
    "skill_execution": 0.9769,
    "skill_efficiency": 0.6784,
    "accuracy": 1.0,
    "goal_accuracy": 1.0,
    "behavior_check": 1.0,
}
EDGE05_INTERVAL = {
    "effectiveness": {
        "estimate": -0.1963,
        "ci_low": -0.2,
        "ci_high": -0.1889,
        "confidence": 0.95,
        "method": "paired_case_bootstrap",
        "resamples": 2000,
        "seed": 0,
        "n_cases": 9,
        "precision": "adequate",
        "ci_includes_zero": False,
    },
    "integration": None,
}


def plugin_dir(tmp_path: Path) -> Path:
    plugin = tmp_path / "gate-demo"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "gate-demo", "version": "1.0.0", "description": "Gate demo plugin."}),
        encoding="utf-8",
    )
    skill = plugin / "skills" / "tidy"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: tidy\ndescription: Tidy CSV files. Use when asked to tidy a CSV.\nversion: 1.0.0\n"
        "metadata:\n  author: Test Author <test@example.com>\n---\n\n# Tidy\n\n## Instructions\nTidy it.\n",
        encoding="utf-8",
    )
    return plugin


def scores(**overrides: float) -> dict[str, float]:
    scores = dict.fromkeys(METRICS, 1.0)
    scores.update(overrides)
    return scores


def tier3_result(
    with_scores: dict[str, float],
    without_scores: dict[str, float] | None = None,
    *,
    lift_uncertainty: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
    plugin_provenance: dict[str, Any] | None = None,
) -> ValidationResult:
    info: dict[str, Any] = {
        "with_skill": dict(with_scores),
        "without_skill": dict(without_scores or {}),
        "execution_status": "succeeded",
        "rewards": [],
        "num_trials": 2,
        "model": "test-model",
    }
    if lift_uncertainty is not None:
        info["lift_uncertainty"] = lift_uncertainty
    payload = build_agent_eval_payload(
        "gate-demo",
        {"codex": info},
        use_llm_judge=False,
        run_config=run_config or {"eval_target": {"kind": "plugin"}},
        plugin_provenance=plugin_provenance,
    )
    result = _validation_result_from_payload(payload)
    assert result is not None
    return result


def run_validate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tier3: ValidationResult | None,
    *flags: str,
    tiers: str = "1,3",
    checks: str = "schema",
    reports: str = "json",
) -> tuple[int, dict[str, Any], str, str]:
    if tier3 is not None:
        monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", lambda *_a, **_k: tier3)
    out = tmp_path / "reports"
    plugin = plugin_dir(tmp_path)
    args = ["validate", str(plugin), "--type", "plugin", "--tiers", tiers, "--no-llm", "--checks", checks]
    result = CliRunner().invoke(cli_module.cli, [*args, "-r", reports, "-o", str(out), *flags])
    report = json.loads(next(out.glob("skillevaluator-output-*.json")).read_text(encoding="utf-8"))
    benchmark = (out / "BENCHMARK.md").read_text(encoding="utf-8")
    # Rich wraps the footer panel; join it into one line so phrases survive the wrap.
    footer = " ".join(result.output.split())
    return result.exit_code, report, benchmark, footer


def callout(benchmark: str) -> str:
    return next(line for line in benchmark.splitlines() if line.startswith("> "))
