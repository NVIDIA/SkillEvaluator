# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit operator-owned coverage and risk decisions for assessment receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path

from skillevaluator.utils.structured_data import load_bounded_json

SCHEMA = "skillevaluator.grader-qualification-policy/v1"
MAX_POLICY_BYTES = 256 * 1024
RATES = ("false_pass_rate", "false_fail_rate", "unjudgeable_rate")
CLASSES = ("pass", "fail")


def load_qualification_policy(path: Path) -> dict:
    """Read a bounded regular JSON file and reject incomplete/unknown policy criteria."""
    from skillevaluator.tier3.grader_validation import CATEGORIES, METRICS, _object

    # Nonblocking open protects against FIFOs where supported. Native Windows
    # has no O_NONBLOCK; check file type first there and still verify the fd.
    if not hasattr(os, "O_NONBLOCK") and not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("qualification policy must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_POLICY_BYTES:
            raise ValueError("qualification policy must be a regular file of at most 256 KiB")
        data = os.read(descriptor, MAX_POLICY_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(data) > MAX_POLICY_BYTES:
        raise ValueError("qualification policy exceeds 256 KiB")
    try:
        text = data.decode("utf-8")
        policy = load_bounded_json(text)
    except UnicodeError as exc:
        raise ValueError("qualification policy must be UTF-8 JSON") from exc

    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"qualification policy has duplicate field: {key}")
            result[key] = value
        return result

    # The existing parser establishes strict syntax/complexity first. Check all
    # object keys too: a last-wins limit would make a reviewed policy ambiguous.
    policy = json.loads(text, object_pairs_hook=unique_fields)
    required = {"schema", "policy_id", "required_metrics", "min_class_cases", "min_category_cases", "max_rates"}
    _object(policy, required | {"description"}, required, "qualification policy")
    if policy["schema"] != SCHEMA:
        raise ValueError(f"qualification policy schema must be {SCHEMA}")
    if not isinstance(policy["policy_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", policy["policy_id"]
    ):
        raise ValueError("policy_id must be a safe ASCII identifier of at most 128 characters")
    if "description" in policy and (
        not isinstance(policy["description"], str)
        or not policy["description"].strip()
        or len(policy["description"]) > 1024
    ):
        raise ValueError("description must be a nonempty string of at most 1024 characters")
    metrics = policy["required_metrics"]
    allowed_metrics = {*METRICS, "behavior_steps", "overall"}
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(not isinstance(m, str) or m not in allowed_metrics for m in metrics)
        or len(set(metrics)) != len(metrics)
    ):
        raise ValueError("required_metrics must be a nonempty unique list of known metrics")
    _object(policy["min_class_cases"], set(CLASSES), set(CLASSES), "min_class_cases")
    _object(policy["min_category_cases"], CATEGORIES, set(), "min_category_cases")
    for field in ("min_class_cases", "min_category_cases"):
        for key, value in policy[field].items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{field}.{key}: expected positive integer")
    _object(policy["max_rates"], set(RATES), set(RATES), "max_rates")
    for key, value in policy["max_rates"].items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
            or not math.isfinite(value)
        ):
            raise ValueError(f"max_rates.{key}: expected finite number in [0,1]")
    return policy


def _expected(cases, thresholds):
    from skillevaluator.tier3.grader_validation import observations

    if any(case["split"] != "assessment" for case in cases):
        raise ValueError("qualification policies require assessment cases only")
    return {case["id"]: observations(case, {}, thresholds) for case in cases}


def _decision(cases, thresholds, policy, report=None):
    expected = _expected(cases, thresholds)
    evaluated = report is not None
    reasons = []
    metrics = {}
    observed_categories = set()
    observed = defaultdict(set)
    for attempt in (report or {}).get("attempts", []):
        for row in attempt["observations"]:
            observed[attempt["case"], row["metric"], row["judgment"]].add(row["observed"])
    for metric in policy["required_metrics"]:
        classes = {label: set() for label in CLASSES}
        scored = {label: set() for label in CLASSES}
        labelled = False
        for case_id, rows in expected.items():
            for row in rows:
                if row["metric"] != metric:
                    continue
                labelled = True
                verdicts = observed[case_id, metric, row["judgment"]]
                binary_scored = bool(verdicts.intersection(CLASSES))
                if row["expected"] in CLASSES:
                    classes[row["expected"]].add(case_id)
                    if binary_scored:
                        scored[row["expected"]].add(case_id)
                        observed_categories.add(case_id)
                elif row["expected"] in {"not_applicable", "insufficient_evidence"} and row["expected"] in verdicts:
                    observed_categories.add(case_id)
        if not labelled:
            reasons.append(
                {"code": "missing_metric", "metric": metric, "message": f"{metric}: required metric has no labels"}
            )
        checks = {}
        for label, minimum in policy["min_class_cases"].items():
            count, scored_count = len(classes[label]), len(scored[label])
            checks[label] = {
                "minimum": minimum,
                "expected_met": count >= minimum,
                "scored_met": scored_count >= minimum if evaluated else None,
            }
            for phase, actual in (("expected", count), ("scored", scored_count)):
                if (phase == "expected" or evaluated) and actual < minimum:
                    reasons.append(
                        {
                            "code": f"{phase}_class_coverage",
                            "metric": metric,
                            "class": label,
                            "actual": actual,
                            "minimum": minimum,
                            "message": f"{metric}: {phase} {label} case coverage {actual} below {minimum}",
                        }
                    )
        rate_checks = None
        if evaluated:
            rate_checks = {}
            summary = report.get("metrics", {}).get(metric, {})
            if not summary:
                reasons.append(
                    {
                        "code": "missing_metric",
                        "metric": metric,
                        "message": f"{metric}: required metric has no observed summary",
                    }
                )
            for name, limit in policy["max_rates"].items():
                measurement = summary.get(name)
                value = measurement.get("rate") if isinstance(measurement, dict) else None
                available = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and 0 <= value <= 1
                    and measurement.get("denominator", 0) > 0
                )
                rate_checks[name] = {
                    "limit": limit,
                    "measurement": measurement,
                    "met": value <= limit if available else None,
                }
                if not available:
                    reasons.append(
                        {
                            "code": "missing_rate",
                            "metric": metric,
                            "rate": name,
                            "message": f"{metric}: {name} unavailable (no scored denominator or measurement)",
                        }
                    )
                elif value > limit:
                    reasons.append(
                        {
                            "code": "rate_exceeded",
                            "metric": metric,
                            "rate": name,
                            "actual": value,
                            "limit": limit,
                            "message": f"{metric}: {name} {value:g} exceeds {limit:g}",
                        }
                    )
        metrics[metric] = {
            "expected_class_counts": {label: len(values) for label, values in classes.items()},
            "scored_class_counts": {label: len(values) for label, values in scored.items()} if evaluated else None,
            "class_checks": checks,
            "rate_checks": rate_checks,
        }
    categories = {}
    counts = Counter(case["category"] for case in cases)
    observed_counts = Counter(case["category"] for case in cases if case["id"] in observed_categories)
    for category, minimum in policy["min_category_cases"].items():
        categories[category] = {
            "minimum": minimum,
            "expected_cases": counts[category],
            "observed_cases": observed_counts[category] if evaluated else None,
            "expected_met": counts[category] >= minimum,
            "observed_met": observed_counts[category] >= minimum if evaluated else None,
        }
        for phase, actual in (("expected", counts[category]), ("observed", observed_counts[category])):
            if (phase == "expected" or evaluated) and actual < minimum:
                reasons.append(
                    {
                        "code": f"{phase}_category_coverage",
                        "category": category,
                        "actual": actual,
                        "minimum": minimum,
                        "message": f"{category}: {phase} category case coverage {actual} below {minimum}",
                    }
                )
    insufficient = any(reason["code"] != "rate_exceeded" for reason in reasons)
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    return {
        "policy_id": policy["policy_id"],
        "policy_sha256": hashlib.sha256(canonical).hexdigest(),
        "policy": policy,
        "scope": {"split": "assessment", "case_ids": [case["id"] for case in cases]},
        "phase": "evaluated" if evaluated else "preflight",
        "status": "insufficient_evidence"
        if insufficient
        else "failed"
        if reasons
        else "passed"
        if evaluated
        else "ready",
        "metrics": metrics,
        "categories": categories,
        "reasons": reasons,
    }


def preflight_qualification(cases, thresholds, policy) -> dict:
    """Check label coverage only; scored counts and rate checks remain unobserved."""
    return _decision(cases, thresholds, policy)


def evaluate_qualification(report, cases, policy) -> dict:
    """Evaluate distinct-case coverage and existing attempt-level rates together."""
    from skillevaluator.tier3.grader_validation import DEFAULT_THRESHOLDS

    # The existing summaries describe all report attempts. Never attach them to
    # a subset's scope, where unselected cases could dilute error rates.
    selected = {case["id"] for case in cases}
    provenance = report.get("provenance", {})
    declared = provenance.get("case_ids", list(selected))
    if (
        not isinstance(declared, list)
        or any(not isinstance(case_id, str) for case_id in declared)
        or len(declared) != len(selected)
        or set(declared) != selected
        or {attempt["case"] for attempt in report.get("attempts", [])} != selected
        or provenance.get("split", "assessment") != "assessment"
    ):
        raise ValueError("qualification report scope must match the selected assessment cases")
    for attempt in report.get("attempts", []):
        if any(row["judgment"].split(":")[0] != attempt["case"] for row in attempt["observations"]):
            raise ValueError("qualification observation scope must match its attempt case")
    thresholds = DEFAULT_THRESHOLDS | provenance.get("thresholds", {})
    return _decision(cases, thresholds, policy, report)
