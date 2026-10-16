# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract test: dashboard measurements must agree with the canonical Tier 3 ``resources`` block.

Builds the canonical payload with the producer's own loader/builder (agent_eval_result_from_directory),
so the test fails as soon as the producer and the dashboard disagree about field names or pairing rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.reporting.dashboard_data import comparison_rows, load_dashboard_path, parse_dashboard_report
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _retained_run(run: Path, *, attempts: int) -> Path:
    scores = dict.fromkeys(DEFAULT_METRICS, 1.0)
    _write(run / "run_config.json", {"agents": {"opencode": {"agent": "opencode", "model": "model-a"}}})
    _write(run / "result.json", {"skill_name": "demo", "run_id": run.name})
    for arm, prompt, seconds in (("with-skill", 20, 3), ("without-skill", 10, 5)):
        condition = run / "opencode" / arm
        count = 2 * attempts
        _write(
            condition / "summary.json",
            {
                "model": "model-a",
                "scores": scores,
                "metrics": list(DEFAULT_METRICS),
                "num_trials": count,
                "num_reward_rows": count,
                "execution_status": "succeeded",
                "expected_attempts": count,
                "scored_attempts": count,
                "pass_at_k": {"k": attempts, "rate": 1.0},
            },
        )
        for case in ("case-a", "case-b"):
            for attempt in range(attempts):
                # Harbor names repeated attempts <task>__<random>; no attempt ordinal is recorded.
                trial = condition / "trials" / f"{case}__trial{attempt}"
                _write(trial / "reward.json", {**scores, "entry_id": case, "trial_id": trial.name})
                _write(
                    trial / "trajectory.json",
                    {"steps": [{}], "final_metrics": {"total_prompt_tokens": prompt, "total_completion_tokens": 2}},
                )
                _write(
                    trial / "result.json",
                    {
                        "agent_execution": {
                            "started_at": "2026-10-11T00:00:00Z",
                            "finished_at": f"2026-10-11T00:00:0{seconds}Z",
                        }
                    },
                )
    return run


@pytest.mark.parametrize("attempts", [1, 3])
@pytest.mark.parametrize("source", ["canonical", "native"])
def test_dashboard_matches_main_resources(tmp_path: Path, attempts: int, source: str) -> None:
    run = _retained_run(tmp_path / "run", attempts=attempts)
    canonical = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False).metadata["agent_eval"]
    resources = canonical["agents"]["opencode"]["resources"]
    assert resources["pairing"] == "matching case counts"

    data = load_dashboard_path(run) if source == "native" else parse_dashboard_report(canonical, "report.json")
    by_condition = {row["condition"]: row for row in data.rows}
    for condition, arm in (("with_skill", "with_skill"), ("without_skill", "baseline")):
        assert by_condition[condition]["score"] is not None
        assert by_condition[condition]["total_tokens"] == resources["tokens"][arm]["total"]
        assert by_condition[condition]["duration_seconds"] == resources["agent_seconds"][arm]["total"]
    comparison = comparison_rows(data.rows)[0]
    assert comparison["comparison_status"] == "comparable"
    assert comparison["total_tokens_delta"] == resources["tokens"]["delta"]
    assert comparison["duration_seconds_delta"] == resources["agent_seconds"]["delta"]
