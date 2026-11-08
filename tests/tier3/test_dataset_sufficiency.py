# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from skillevaluator.tier3.dataset_sufficiency import assess_dataset_sufficiency, generation_coverage_summary


def test_dataset_generation_coverage_summary():
    assert generation_coverage_summary(4, 0) == (
        "Dataset size: provisional — 4 of 10 cases, no negative case (policy dataset-sufficiency-v1)"
    )


def test_small_dataset_is_provisional():
    result = assess_dataset_sufficiency(
        distinct_cases=3,
        negative_cases=0,
        pass_rate_interval={"lower": 0.1, "upper": 0.9},
        paired_cases=3,
    )

    assert result["status"] == "provisional"
    assert [check["met"] for check in result["checks"]] == [False, False, False]


def test_dataset_meeting_policy_thresholds_is_sufficient():
    result = assess_dataset_sufficiency(
        distinct_cases=12,
        negative_cases=2,
        pass_rate_interval={"lower": 0.45, "upper": 0.55},
        paired_cases=12,
    )

    assert result["status"] == "sufficient"
    assert result["reason"] == "Enough cases for this policy"


def test_incomplete_pairing_is_provisional():
    result = assess_dataset_sufficiency(
        distinct_cases=12,
        negative_cases=2,
        pass_rate_interval={"lower": 0.45, "upper": 0.55},
        paired_cases=8,
    )

    assert result["checks"][2]["met"] is False
    assert "8 paired cases" in result["reason"]


def test_repeated_attempts_do_not_change_distinct_case_count():
    result = assess_dataset_sufficiency(
        distinct_cases=3,
        negative_cases=1,
        pass_rate_interval={"lower": 0.45, "upper": 0.55},
        paired_cases=3,
    )

    assert result["checks"][0]["observed"]["distinct_cases"] == 3
    assert result["checks"][0]["met"] is False


def test_skipped_baseline_makes_lift_comparison_unavailable():
    result = assess_dataset_sufficiency(
        distinct_cases=12,
        negative_cases=2,
        pass_rate_interval={"lower": 0.45, "upper": 0.55},
        paired_cases=None,
        baseline_skipped=True,
    )

    assert result["checks"][2]["met"] is None
    assert result["checks"][2]["reason"] == "baseline not run; lift comparison unavailable"
