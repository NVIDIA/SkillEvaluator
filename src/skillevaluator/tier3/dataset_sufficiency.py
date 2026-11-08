# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy-based dataset sufficiency checks for Tier 3 reports."""

from __future__ import annotations

MIN_DISTINCT_CASES = 10
MIN_NEGATIVE_CASES = 1
MAX_PASS_RATE_INTERVAL_HALF_WIDTH = 0.20
MIN_PAIRED_CASES = 10
POLICY = "dataset-sufficiency-v1"


def generation_coverage_summary(distinct_cases: int, negative_cases: int) -> str:
    sufficient = distinct_cases >= MIN_DISTINCT_CASES and negative_cases >= MIN_NEGATIVE_CASES
    status = "sufficient" if sufficient else "provisional"
    negative_label = "no negative case" if negative_cases == 0 else f"{negative_cases} negative case(s)"
    return (
        f"Dataset size: {status} — {distinct_cases} of {MIN_DISTINCT_CASES} cases, {negative_label} (policy {POLICY})"
    )


def assess_dataset_sufficiency(
    *,
    distinct_cases: int,
    negative_cases: int,
    pass_rate_interval: dict | None,
    paired_cases: int | None,
    baseline_skipped: bool = False,
) -> dict:
    """Assess dataset coverage, pass-rate reliability, and paired lift coverage."""
    coverage_met = distinct_cases >= MIN_DISTINCT_CASES and negative_cases >= MIN_NEGATIVE_CASES
    checks = [
        {
            "name": "coverage",
            "met": coverage_met,
            "observed": {"distinct_cases": distinct_cases, "negative_cases": negative_cases},
            "required": {"distinct_cases": MIN_DISTINCT_CASES, "negative_cases": MIN_NEGATIVE_CASES},
            "reason": (
                "Enough cases for this policy"
                if coverage_met
                else f"{distinct_cases} distinct cases and {negative_cases} negative cases; "
                f"requires {MIN_DISTINCT_CASES} and {MIN_NEGATIVE_CASES}"
            ),
        }
    ]
    if isinstance(pass_rate_interval, dict):
        lower, upper = pass_rate_interval.get("lower"), pass_rate_interval.get("upper")
        half_width = (upper - lower) / 2 if isinstance(lower, int | float) and isinstance(upper, int | float) else None
    else:
        half_width = None
    reliability_met = half_width is not None and half_width <= MAX_PASS_RATE_INTERVAL_HALF_WIDTH
    checks.append(
        {
            "name": "pass_rate_reliability",
            "met": reliability_met if half_width is not None else None,
            "observed": {"interval_half_width": half_width},
            "required": {"interval_half_width": MAX_PASS_RATE_INTERVAL_HALF_WIDTH},
            "reason": (
                "Enough cases for this policy"
                if reliability_met
                else "95% Wilson interval is unavailable"
                if half_width is None
                else f"95% Wilson interval half-width {half_width:.2f} exceeds {MAX_PASS_RATE_INTERVAL_HALF_WIDTH:.2f}"
            ),
        }
    )
    paired_met = paired_cases >= MIN_PAIRED_CASES if paired_cases is not None and not baseline_skipped else None
    checks.append(
        {
            "name": "lift_comparison",
            "met": paired_met,
            "observed": {"paired_cases": paired_cases},
            "required": {"paired_cases": MIN_PAIRED_CASES},
            "reason": (
                "baseline not run; lift comparison unavailable"
                if baseline_skipped
                else "Paired-case count is unavailable"
                if paired_cases is None
                else "Enough cases for this policy"
                if paired_met
                else f"{paired_cases} paired cases; requires {MIN_PAIRED_CASES}"
            ),
        }
    )
    unmet = [check["reason"] for check in checks if check["met"] is not True]
    return {
        "policy": POLICY,
        "status": "sufficient" if not unmet else "provisional",
        "checks": checks,
        "reason": "Enough cases for this policy" if not unmet else "provisional: " + "; ".join(unmet),
    }
