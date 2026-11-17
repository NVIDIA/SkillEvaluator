# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Findings about cyclic links must remain exportable to SARIF."""

import errno
import json
from pathlib import Path

import pytest

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting import SARIFReporter
from skillevaluator.reporting.sarif_reporter import _resolve_for_reporting


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("two_links", [False, True])
@pytest.mark.parametrize("with_workspace", [False, True])
@pytest.mark.parametrize("expanding", [False, True])
def test_sarif_preserves_a_finding_on_a_symlink_loop(
    tmp_path: Path, relative: bool, two_links: bool, with_workspace: bool, expanding: bool
) -> None:
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    link = skill / "loop.md"
    try:
        target = "other.md" if two_links else "loop.md"
        link.symlink_to(f"{target}/child" if expanding else target)
        if two_links:
            (skill / "other.md").symlink_to("loop.md")
    except OSError:
        pytest.skip("symlinks are not available")
    result = ValidationResult(validator_name="security")
    result.add_finding(
        Finding(
            category="SECURITY",
            severity=Severity.HIGH,
            check_name="unsafe-link",
            message="The resource is a cyclic symbolic link",
            file_path="loop.md" if relative else str(link),
        )
    )

    output = SARIFReporter(workspace_root=tmp_path if with_workspace else None, scan_root=skill).render_all([result])

    findings = json.loads(output)["runs"][0]["results"]
    assert len(findings) == 1
    expected = "skills/demo/loop.md" if with_workspace else link.as_uri()
    assert findings[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == expected
    assert findings[0]["message"]["text"] == "The resource is a cyclic symbolic link"


def test_windows_loop_error_preserves_the_named_artifact(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "loop.md" / ".." / "named.md"
    error = OSError(errno.EINVAL, "The name of the file cannot be resolved by the system")
    error.winerror = 1921

    def realpath(value, *, strict=False):
        if strict:
            raise error
        return f"{value}/child"

    monkeypatch.setattr("skillevaluator.reporting.sarif_reporter.os.path.realpath", realpath)

    assert _resolve_for_reporting(path) == path.absolute()
    assert _resolve_for_reporting(_resolve_for_reporting(path)) == path.absolute()
