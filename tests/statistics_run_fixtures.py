# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor job fixtures with scores, collected like a real run.

Each trial has a verifier reward, so the collector can score the arm and the run
statistics can pair its cases.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS, DEFAULT_SCORE_POLICY

SKILL = "demo"
AGENT = "codex"
CASES = [f"case-{index}" for index in range(1, 7)]


@dataclass(frozen=True)
class Trial:
    """One attempt: its quality, and whether it passes."""

    quality: float
    passed: bool = True
    activated: bool | None = None
    judge: str | None = None
    accuracy: float | None = None


def uniform(quality: float, count: int = 6, **trial: Any) -> dict[str, list[Trial]]:
    """``count`` single-attempt cases that all score ``quality``."""
    return {f"case-{index}": [Trial(quality, **trial)] for index in range(1, count + 1)}


def graded_baseline() -> dict[str, list[Trial]]:
    """A baseline that passes every other case of ``CASES``, so the two arms differ in score and in pass rate."""
    return {case: [Trial(0.6 if index % 2 else 0.2, passed=index % 2 == 1)] for index, case in enumerate(CASES)}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def reward(case: str, trial: Trial) -> dict[str, Any]:
    scores = dict.fromkeys(DEFAULT_METRICS, trial.quality)
    scores.update(security=1.0, goal_accuracy=1.0 if trial.passed else 0.0)
    if trial.accuracy is not None:
        scores["accuracy"] = trial.accuracy
    details: dict[str, Any] = {}
    if trial.judge:
        details["accuracy"] = {"score": scores["accuracy"], "judge_model_used": trial.judge}
    if trial.activated is not None:
        details["skill_execution"] = {"activation": {"passed": trial.activated, "score": float(trial.activated)}}
    return {
        **scores,
        "entry_id": case,
        "metric_set": DEFAULT_METRIC_SET,
        "score_policy": DEFAULT_SCORE_POLICY,
        "details": details,
    }


def write_job(job: Path, trials: Mapping[str, Sequence[Trial]]) -> None:
    """Write a plain (single-step) Harbor job: ``trials`` maps each case to its attempts in order."""
    names = []
    for case, attempts in trials.items():
        for ordinal, trial in enumerate(attempts, 1):
            name = f"{case}__attempt{ordinal}"
            names.append(name)
            root = job / name
            write_json(root / "verifier" / "reward.json", reward(case, trial))
    write_json(
        job / "result.json",
        {
            "n_total_trials": len(names),
            "stats": {
                "n_completed_trials": len(names),
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
                "n_retries": 0,
                "evals": {
                    "agent__model___tasks": {
                        "n_trials": len(names),
                        "n_errors": 0,
                        "reward_stats": {"overall": {"0.5": names}},
                    }
                },
            },
        },
    )


def collect(
    root: Path,
    with_skill: Mapping[str, Sequence[Trial]] | None,
    without_skill: Mapping[str, Sequence[Trial]] | None,
    *,
    cases: Sequence[str] | None = None,
    run_name: str = "run",
    **options: Any,
) -> tuple[dict[str, Any], Path]:
    """Write both arms' jobs under ``root/jobs``, collect them into ``root/<run_name>`` and return both."""
    jobs = root / "jobs"
    for suffix, arm in (("with", with_skill), ("without", without_skill)):
        if arm is not None:
            write_job(jobs / f"{SKILL}-{AGENT}-{suffix}", arm)
    expected = list(cases if cases is not None else (with_skill or without_skill or {}))
    attempts = max((len(rows) for rows in (with_skill or without_skill or {}).values()), default=1)
    options.setdefault("n_attempts", attempts)
    options.setdefault("expected_case_ids", expected)
    options.setdefault("expected_trials", len(expected) * attempts)
    output = root / run_name
    collected = collect_harbor_results(SKILL, [AGENT], output, jobs, **options)
    return collected, output
