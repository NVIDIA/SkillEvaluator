# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection summary totals must describe the findings and checks that survive merging."""

import json
from collections import Counter
from copy import deepcopy

import pytest

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.validators.base import continue_on_failure_scope
from skillevaluator.validators.schema import SchemaValidator
from skillevaluator.validators.unicode_smuggle import UnicodeSmuggleValidator


@pytest.mark.parametrize("nested", [False, True])
def test_prefixed_merge_keeps_every_summary_counter(nested):
    child = ValidationResult()
    for severity in Severity:
        child.add_finding(Finding("TEST", severity, severity.value, "A finding", "SKILL.md"))
    child.add_error("An independent error")
    child.add_warning("An independent warning")
    child.add_success("content", "Content was inspected")
    child.summary.files_scanned = 3
    original = deepcopy(child)
    if nested:
        group = ValidationResult()
        group.merge_with_prefix(child, "inner")
        child = group
    parent = ValidationResult()
    parent.add_finding(Finding("TEST", Severity.HIGH, "parent", "Existing finding", "other.md"))
    parent.add_success("parent-check", "Parent check completed")
    parent.summary.files_scanned = 2
    expected = {key: value + original.summary.to_dict()[key] for key, value in parent.summary.to_dict().items()}

    parent.merge_with_prefix(child, "outer")

    assert parent.summary.to_dict() == expected
    assert parent.to_dict()["summary"] == expected
    assert not parent.passed
    assert len(parent.findings) == 6
    assert len(parent.success_details) == 2
    if not nested:
        assert child == original


@pytest.mark.parametrize("broken_count", [2, 1, 0])
def test_unicode_collection_json_keeps_scanned_files_and_raw_severity_counts(tmp_path, broken_count):
    for index, name in enumerate(("first-skill", "second-skill")):
        skill = tmp_path / name
        skill.mkdir()
        (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test\n---\n# Test\n")
        for filename, content in {
            "critical.md": "visible" + "".join(chr(0xE0000 + ord(char)) for char in "example"),
            "high.md": "normal\u202etext\n",
            "medium.md": "text\u206amore\n",
            "low.md": "hello\u200bworld\n",
        }.items():
            (skill / filename).write_text(content if index < broken_count else "Ordinary text\n", encoding="utf-8")

    with continue_on_failure_scope(True):
        result = UnicodeSmuggleValidator().validate(tmp_path)
    report = json.loads(JSONReporter(include_timestamp=False).render_all([result]))

    assert result.passed is (broken_count == 0)
    counts = Counter(finding.severity.value for finding in result.findings)
    expected_counts = dict.fromkeys(("critical", "high", "medium", "low"), broken_count)
    assert counts == Counter(expected_counts)
    assert report["severity_counts"] == expected_counts
    summary = report["results"][0]["summary"]
    assert summary["files_scanned"] == 10
    assert summary["errors"] == 2 * broken_count
    assert summary["warnings"] == 2 * broken_count
    for severity, count in expected_counts.items():
        assert summary[f"{severity}_count"] == count


def test_failed_schema_collection_counts_findings_once(tmp_path):
    skill = tmp_path / "skills" / "broken-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: broken-skill\ndescription: Test\nmetadata:\n  author: name@example.com\n---\n# Test\n"
    )

    result = SchemaValidator().validate(tmp_path)
    report = json.loads(JSONReporter(include_timestamp=False).render_all([result]))

    assert not result.passed
    counts = Counter(finding.severity.value for finding in result.findings)
    assert counts["high"] > 0
    assert report["severity_counts"] == {
        severity: counts[severity] for severity in ("critical", "high", "medium", "low")
    }
