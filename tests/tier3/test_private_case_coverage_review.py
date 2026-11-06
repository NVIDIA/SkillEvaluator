# SPDX-License-Identifier: Apache-2.0
"""Case coverage retains its own denominator alongside logical attempts."""

from __future__ import annotations

import pytest

from skillevaluator.tier3.harbor.collector import _condition_execution_summary
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS


def _reward(case: str, attempt: int = 1, value: float = 0.8) -> dict:
    return {
        "entry_id": case,
        "_trial_root_name": f"{case}__attempt{attempt}",
        "_attempt_ordinal": attempt,
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, value),
    }


@pytest.mark.parametrize(
    ("rewards", "options", "coverage", "status"),
    [
        (
            [_reward("a", 1), _reward("a", 2), _reward("b", 1), _reward("b", 2)],
            {},
            (2, 2, 4, 4),
            "succeeded",
        ),
        ([_reward("a", 1)], {}, (2, 1, 4, 1), "failed"),
        (
            [_reward("a", 1), _reward("b", 1, 0.2), _reward("b", 2)],
            {"n_attempts": 3, "stop_on_pass": True},
            (2, 2, 3, 3),
            "succeeded",
        ),
        ([], {"skipped": True}, (0, 0, 0, 0), "skipped"),
        (
            [_reward("a"), _reward("extra"), _reward("b", value=float("nan"))],
            {"n_attempts": 1},
            (2, 1, 2, 2),
            "failed",
        ),
        (
            [_reward("a"), _reward("b")],
            {"expected_case_ids": None, "expected_cases": 3, "n_attempts": 1},
            (3, 2, 3, 2),
            "failed",
        ),
        (
            [_reward("a", 2)],
            {
                "expected_case_ids": ["a"],
                "expected_cases": 1,
                "n_attempts": 3,
                "stop_on_pass": True,
                "recovered_attempts": [{"case_id": "a", "trial": "a__attempt1", "attempt": 1}],
            },
            (1, 1, 1, 1),
            "succeeded",
        ),
    ],
    ids=["multi-attempt", "missing-attempts", "stop-on-pass", "skipped", "extra-invalid", "count-only", "recovered"],
)
def test_condition_counts_cases_without_inferring_them_from_attempts(rewards, options, coverage, status):
    condition = _condition_execution_summary(
        rewards,
        job_failure="",
        **{"expected_case_ids": ["a", "b"], "expected_cases": 2, "n_attempts": 2, **options},
    )
    assert condition["execution_status"] == status
    assert (
        tuple(condition.get(key) for key in ("expected_cases", "scored_cases", "expected_attempts", "scored_attempts"))
        == coverage
    )
