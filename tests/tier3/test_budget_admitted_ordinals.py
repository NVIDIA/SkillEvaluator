# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.tier3.test_baseline_timeouts import _attempt_job

from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.runner import _merge_attempt_jobs


@pytest.mark.parametrize("ordinals", [(1, 3), (1,), (3,), (), (1, 2), (1, 2, 3)])
@pytest.mark.parametrize("reason", ["budget_exhausted", None, "execution_failed"])
@pytest.mark.parametrize("stop_on_pass", [False, True])
def test_coverage_uses_admitted_slots(tmp_path, ordinals, reason, stop_on_pass):
    not_run = []
    for arm in ("with", "without"):
        jobs = [
            _attempt_job(tmp_path, "agent", arm, "selector", ordinal, "pass" if ordinal == 3 else "fail")
            for ordinal in ordinals
        ]
        _merge_attempt_jobs(jobs, tmp_path / "jobs" / f"demo-agent-{arm}")
        row = {"agent": "agent", "arm": arm, "case_id": "selector", "attempt": 2, "status": "not_run"}
        if reason is not None:
            row["reason"] = reason
        not_run.append(row)

    result = collect_harbor_results(
        skill_name="demo",
        agents=["agent"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        n_attempts=3,
        expected_trials=None if stop_on_pass else 3,
        expected_case_ids=["case"],
        expected_cases=1,
        case_id_by_task_selector={"selector": "case"},
        budget_not_run=not_run,
        stop_on_pass=stop_on_pass,
    )
    omitted = reason != "execution_failed"
    complete = ordinals == ((1, 3) if omitted else (1, 2, 3))
    assert result["execution_status"] == ("succeeded" if complete else "failed")
    for condition in result["agents"]["agent"]["conditions"].values():
        assert condition["expected_attempts"] == (2 if omitted else 3)
        if complete:
            assert condition["scored_attempts"] == condition["expected_attempts"]
            assert not condition["execution_errors"]
        else:
            assert any("scored attempts" in error for error in condition["execution_errors"])
