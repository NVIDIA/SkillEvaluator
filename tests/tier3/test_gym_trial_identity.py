# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distinct valid case IDs must not overwrite one another's Gym trial evidence."""

import json

import pytest

from skillevaluator.tier3.case_ids import validate_case_ids
from skillevaluator.tier3.nemo_gym.results import write_job_from_rollouts


@pytest.mark.parametrize("suffix", ["-", "--"])
@pytest.mark.parametrize("attempts", [1, 2])
def test_gym_trials_keep_distinct_trailing_hyphen_case_ids(tmp_path, suffix, attempts):
    task_names = ["case", "case" + suffix]
    assert validate_case_ids(task_names) == task_names
    rows = []
    for index, task_name in enumerate(task_names):
        for attempt in range(1, attempts + 1):
            logs = tmp_path / "logs" / task_name / str(attempt)
            logs.mkdir(parents=True)
            (logs / "reward.json").write_text(json.dumps({"overall": index / 2, "entry_id": task_name}))
            (logs / "trajectory.json").write_text(json.dumps({"steps": [{"message": f"{task_name}/{attempt}"}]}))
            rows.append({"task_id": task_name, "verifier_return_code": 0, "verifier_logs_dir": str(logs)})

    job = tmp_path / "job"
    errors = write_job_from_rollouts(
        rows, job, eval_name="agent", task_case_ids={name: name for name in task_names}, n_attempts=attempts
    )

    assert errors == []
    trials = {path.name: path for path in job.iterdir() if path.is_dir()}
    expected_names = {f"{name}__gym{attempt:03d}" for name in task_names for attempt in range(1, attempts + 1)}
    assert trials.keys() == expected_names
    for index, task_name in enumerate(task_names):
        for attempt in range(1, attempts + 1):
            trial = trials[f"{task_name}__gym{attempt:03d}"]
            result = json.loads((trial / "result.json").read_text())
            assert result["task_name"] == task_name
            assert result["verifier_result"]["rewards"] == {"overall": index / 2, "entry_id": task_name}
            assert (
                json.loads((trial / "agent/trajectory.json").read_text())["steps"][0]["message"]
                == f"{task_name}/{attempt}"
            )
    result = json.loads((job / "result.json").read_text())
    reward_groups = result["stats"]["evals"]["agent"]["reward_stats"]["reward"]
    assert {name for group in reward_groups.values() for name in group} == expected_names
