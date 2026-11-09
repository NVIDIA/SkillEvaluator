# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy-based dataset sufficiency checks for Tier 3 reports."""

from __future__ import annotations

MIN_DISTINCT_CASES = 10
MIN_NEGATIVE_CASES = 1
MAX_PASS_RATE_INTERVAL_HALF_WIDTH = 0.20
MIN_PAIRED_CASES = 10
POLICY = "dataset-sufficiency-v1"


def _format_dataset_size_summary(
    status: str,
    distinct_cases: int,
    negative_cases: int,
    reason: str | None = None,
) -> str:
    case_label = "case" if distinct_cases == 1 else "cases"
    negative_label = "negative case" if negative_cases == 1 else "negative cases"
    summary = (
        f"Dataset size: {status} — {distinct_cases} {case_label} (policy needs {MIN_DISTINCT_CASES}), "
        f"{negative_cases} {negative_label} (needs {MIN_NEGATIVE_CASES})"
    )
    if reason:
        summary += f"; {reason}"
    return f"{summary} (policy {POLICY})"


def generation_coverage_summary(distinct_cases: int, negative_cases: int) -> str:
    sufficient = distinct_cases >= MIN_DISTINCT_CASES and negative_cases >= MIN_NEGATIVE_CASES
    status = "sufficient" if sufficient else "provisional"
    return _format_dataset_size_summary(status, distinct_cases, negative_cases)


def _first_noncoverage_reason(checks: list[dict]) -> str | None:
    for check in checks:
        if check["name"] == "coverage" or check["met"] is True:
            continue
        if check["name"] == "pass_rate_reliability":
            half_width = check["observed"]["interval_half_width"]
            if half_width is None:
                return "pass-rate interval unavailable"
            return f"pass-rate interval ±{half_width:.2f} is wider than ±{MAX_PASS_RATE_INTERVAL_HALF_WIDTH:.2f}"
        if check["name"] == "lift_comparison":
            if check["reason"] == "baseline not run; lift comparison unavailable":
                return "baseline not run, so lift has no paired cases"
            paired_cases = check["observed"]["paired_cases"]
            if paired_cases is None:
                return "paired-case count unavailable, so lift cannot be compared"
            return f"lift has {paired_cases} paired cases (needs {MIN_PAIRED_CASES})"
    return None


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
    status = "sufficient" if not unmet else "provisional"
    return {
        "policy": POLICY,
        "status": status,
        "coverage_summary": _format_dataset_size_summary(
            status,
            distinct_cases,
            negative_cases,
            _first_noncoverage_reason(checks),
        ),
        "checks": checks,
        "reason": "Enough cases for this policy" if not unmet else "provisional: " + "; ".join(unmet),
    }
