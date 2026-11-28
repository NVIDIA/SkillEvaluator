# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact resolution scales with files without retaining stale filesystem state."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.sarif_reporter import SARIFReporter


def _finding(path: str, line: int) -> Finding:
    return Finding(
        category="STATIC",
        severity=Severity.MEDIUM,
        check_name="example",
        message=f"Diagnostic on line {line}",
        file_path=path,
        line_number=line,
        line_content=f"source line {line}",
    )


def _locations(reporter: SARIFReporter, results: list[ValidationResult]) -> list[dict]:
    document = json.loads(reporter.render_all(results))
    return [result["locations"][0]["physicalLocation"] for result in document["runs"][0]["results"]]


def test_repeated_findings_do_not_repeat_filesystem_resolution(tmp_path: Path) -> None:
    scan = tmp_path / "skill"
    scan.mkdir()
    paths = ["SKILL.md", "script name.py"]
    for path in paths:
        (scan / path).write_text("example\n", encoding="utf-8")
    reporter = SARIFReporter(include_timestamp=False, workspace_root=tmp_path, scan_root=scan)
    single = ValidationResult(validator_name="FIRST", findings=[_finding(path, 1) for path in paths])
    realpath = os.path.realpath
    with patch("os.path.realpath", wraps=realpath) as resolve:
        _locations(reporter, [single])
        unique_file_calls = resolve.call_count
    assert unique_file_calls > 0

    results = [
        ValidationResult(
            validator_name=name,
            findings=[_finding(path, line) for path in paths for line in range(1, 11)],
        )
        for name in ("FIRST", "SECOND")
    ]
    with patch("os.path.realpath", wraps=realpath) as resolve:
        locations = _locations(reporter, results)
        assert resolve.call_count == unique_file_calls
    assert locations == [
        {
            "artifactLocation": {"uri": uri},
            "region": {"startLine": line, "snippet": {"text": f"source line {line}"}},
        }
        for _ in results
        for uri in ("skill/SKILL.md", "skill/script%20name.py")
        for line in range(1, 11)
    ]


def test_reused_reporter_resolves_changed_scan_root(tmp_path: Path) -> None:
    for directory in ("first", "second"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "SKILL.md").write_text("example\n", encoding="utf-8")
    reporter = SARIFReporter(
        include_timestamp=False, workspace_root=tmp_path, scan_root=tmp_path / "first"
    )
    results = [ValidationResult(validator_name="TEST", findings=[_finding("SKILL.md", 1)])]
    assert _locations(reporter, results)[0]["artifactLocation"]["uri"] == "first/SKILL.md"
    reporter.scan_root = tmp_path / "second"
    assert _locations(reporter, results)[0]["artifactLocation"]["uri"] == "second/SKILL.md"


def test_reused_reporter_resolves_retargeted_symlink(tmp_path: Path) -> None:
    for name in ("first.md", "second.md"):
        (tmp_path / name).write_text("example\n", encoding="utf-8")
    link = tmp_path / "SKILL.md"
    try:
        link.symlink_to("first.md")
    except OSError:
        pytest.skip("symlinks unavailable")
    reporter = SARIFReporter(include_timestamp=False, workspace_root=tmp_path, scan_root=tmp_path)
    results = [ValidationResult(validator_name="TEST", findings=[_finding("SKILL.md", 1)])]
    assert _locations(reporter, results)[0]["artifactLocation"]["uri"] == "first.md"
    link.unlink()
    link.symlink_to("second.md")
    assert _locations(reporter, results)[0]["artifactLocation"]["uri"] == "second.md"
