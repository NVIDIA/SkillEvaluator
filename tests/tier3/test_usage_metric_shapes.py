# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Malformed optional trajectory usage cannot abort completed result collection."""

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.baseline_store import measured_condition_usage


@pytest.mark.parametrize("metrics", [None, [], False, 42, "unavailable"])
@pytest.mark.parametrize("with_valid_trial", [False, True])
def test_nonobject_final_metrics_leave_condition_usage_unavailable(
    tmp_path: Path, metrics: object, with_valid_trial: bool
) -> None:
    records = [("bad", metrics)]
    if with_valid_trial:
        records.append(
            (
                "good",
                {
                    "total_prompt_tokens": 10,
                    "total_completion_tokens": 4,
                    "total_cached_tokens": 2,
                    "total_cost_usd": 0.05,
                },
            )
        )
    for trial_id, final_metrics in records:
        trial = tmp_path / "trials" / trial_id
        trial.mkdir(parents=True)
        (trial / "reward.json").write_text(json.dumps({"trial_id": trial_id}), encoding="utf-8")
        (trial / "trajectory.json").write_text(json.dumps({"final_metrics": final_metrics}), encoding="utf-8")

    assert measured_condition_usage(tmp_path) == {
        "total_prompt_tokens": None,
        "total_completion_tokens": None,
        "total_cached_tokens": None,
        "total_cost_usd": None,
    }
