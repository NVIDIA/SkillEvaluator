# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection reconciliation retains distinct occurrences across overlapping summaries."""

from copy import deepcopy

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.finding_summary import merge_finding_summaries, summarize_findings


def test_merge_matches_reconciliation_across_overlapping_summaries() -> None:
    results = []
    for validator, paths, severity in (
        ("schema", ("a/SKILL.md", "b/SKILL.md"), Severity.LOW),
        ("quality", ("b/SKILL.md", "c/SKILL.md"), Severity.HIGH),
        ("schema", ("a/SKILL.md", "d/SKILL.md"), Severity.MEDIUM),
        ("quality", ("c/SKILL.md",), Severity.LOW),
    ):
        result = ValidationResult(validator_name=validator)
        for path in paths:
            result.add_finding(
                Finding(
                    category="QUALITY",
                    severity=severity,
                    check_name="description",
                    message="Description is missing",
                    file_path=path,
                    line_number=1,
                    metadata={"source": {"name": validator}},
                )
            )
        results.append(result)
    summaries = [summarize_findings([result]) for result in results]
    originals = deepcopy(summaries)

    merged = merge_finding_summaries(summaries)

    assert merged == summarize_findings(results)
    assert merged["distinct_occurrences"] == 4
    assert merged["duplicates_removed"] == 3
    assert merged["groups"][0]["severity"] == "high"
    assert merged["groups"][0]["validators"] == ["quality", "schema"]
    merged["groups"][0]["metadata"]["source"]["name"] = "changed"
    merged["groups"][0]["occurrences"][-1]["file_path"] = "changed"
    assert summaries == originals
