# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery bookkeeping must scale with observations, not their cross product."""

from __future__ import annotations

import pytest

from skillevaluator.tier3.harbor.collector import PUBLISHED_CASE_DETAILS_MAX, _pass_summary
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


class _Recovery(dict):
    def __init__(self, case_id: str, attempt: int, reads: list[int]) -> None:
        super().__init__(case_id=case_id, attempt=attempt)
        self.reads = reads

    def __getitem__(self, key):
        if key == "case_id":
            self.reads[0] += 1
        return super().__getitem__(key)


@pytest.mark.parametrize("case_count", [128, 512])
def test_recovery_observations_are_not_rescanned_per_case(case_count: int) -> None:
    case_ids = [f"case-{index:04d}" for index in range(case_count)]
    reads = [0]
    recovered = [_Recovery(case, attempt, reads) for case in case_ids for attempt in range(1, 9)]
    rewards = [
        {
            "entry_id": case,
            "_attempt_ordinal": 9,
            "_trial_name": f"{case}__attempt9",
            **dict.fromkeys(DEFAULT_METRICS, 1.0),
        }
        for case in case_ids
    ]

    summary = _pass_summary(
        rewards,
        n_attempts=10,
        pass_threshold=0.8,
        stop_on_pass=True,
        expected_cases=case_count,
        expected_case_ids=case_ids,
        recovered_attempts=recovered,
    )

    assert summary["passed_cases"] == summary["total_cases"] == case_count
    assert summary["rate"] == 1.0
    assert len(summary["_pairing_cases"]) == case_count
    assert len(summary["cases"]) == min(case_count, PUBLISHED_CASE_DETAILS_MAX)
    for case in summary["cases"].values():
        assert case["first_pass_attempt"] == 9
        assert case["attempts_used"] == 1
        assert case["attempts_recovered"] == 8
        assert case["attempts_missing"] == 0
        assert case["attempts_skipped"] == 1
    assert reads[0] <= 4 * len(recovered), f"{reads[0]} recovery identity reads for {len(recovered)} observations"


def test_recovery_index_keeps_case_identity_and_ordinal_rules() -> None:
    reward = {"entry_id": "a", "_attempt_ordinal": 3, **dict.fromkeys(DEFAULT_METRICS, 1.0)}
    recovered = [
        {"case_id": "a", "attempt": 1},
        {"case_id": "a", "attempt": "2"},
        {"case_id": "b", "attempt": 2},
        {"case_id": ["a"], "attempt": 2},
    ]
    summary = _pass_summary(
        [reward],
        n_attempts=4,
        pass_threshold=0.8,
        stop_on_pass=True,
        expected_cases=2,
        expected_case_ids=["a", "b"],
        recovered_attempts=recovered,
    )

    # A string ordinal contributes to the count but does not establish which
    # physical attempt recovered. Another case cannot cover a missing slot.
    assert summary["cases"]["a"]["attempts_recovered"] == 2
    assert summary["cases"]["a"]["attempts_missing"] == 1
    assert summary["cases"]["a"]["attempts_skipped"] == 0
    assert summary["cases"]["b"]["attempts_recovered"] == 1
    assert summary["cases"]["b"]["attempts_missing"] == 3
    assert recovered[-1] == {"case_id": ["a"], "attempt": 2}
