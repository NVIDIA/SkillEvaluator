# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Findings about cyclic links must remain exportable to SARIF."""

import errno
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import pytest

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting import SARIFReporter
from skillevaluator.reporting.junit_reporter import JUnitReporter
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


@pytest.mark.parametrize("location", ["outside-parent", "inside-parent", "symlink-parent", "cyclic-parent"])
@pytest.mark.parametrize("expanding", [False, True])
@pytest.mark.parametrize("reporter", ["sarif", "junit"])
def test_cyclic_artifact_parent_resolution_preserves_workspace_boundaries(
    tmp_path: Path, location: str, expanding: bool, reporter: str
) -> None:
    workspace = tmp_path / "workspace"
    skill = workspace / "skills" / "demo"
    skill.mkdir(parents=True)
    artifact = skill / "loop.md"
    try:
        if location == "outside-parent":
            outside = tmp_path / "outside docs"
            outside.mkdir()
            artifact = outside / "loop.md"
            file_path = os.path.relpath(artifact, skill)
        elif location == "inside-parent":
            file_path = "../demo/loop.md"
        elif location == "symlink-parent":
            (skill / "updir").symlink_to(skill.parent, target_is_directory=True)
            file_path = "updir/demo/loop.md"
        else:
            (skill / "loopdir").symlink_to("loopdir/child" if expanding else "loopdir", target_is_directory=True)
            artifact = skill / "loopdir" / "named.md"
            file_path = "loopdir/named.md"
        if location != "cyclic-parent":
            artifact.symlink_to("loop.md/child" if expanding else "loop.md")
    except OSError:
        pytest.skip("symlinks are not available")
    result = ValidationResult(validator_name="security")
    result.add_finding(
        Finding(
            category="SECURITY",
            severity=Severity.HIGH,
            check_name="unsafe-link",
            message="The resource is a cyclic symbolic link",
            file_path=file_path,
        )
    )
    outside_workspace = location == "outside-parent"
    relative = artifact.as_posix() if outside_workspace else artifact.relative_to(workspace).as_posix()

    if reporter == "sarif":
        document = json.loads(SARIFReporter(workspace_root=workspace, scan_root=skill).render_all([result]))
        finding = document["runs"][0]["results"][0]
        expected = artifact.as_uri() if outside_workspace else relative
        assert finding["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == expected
    else:
        document = ET.fromstring(JUnitReporter(workspace_root=workspace, scan_root=skill).render_all([result]))
        cases = [case for case in document.iter("testcase") if "file" in case.attrib]
        assert len(cases) == 1
        assert cases[0].attrib["file"] == quote(relative, safe="/:@")
    assert _resolve_for_reporting(skill / file_path) == artifact
    assert _resolve_for_reporting(_resolve_for_reporting(skill / file_path)) == artifact


def test_windows_loop_error_normalizes_the_parent_and_preserves_the_named_artifact(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "parent").mkdir()
    path = tmp_path / "parent" / ".." / "named.md"
    expected = tmp_path / "named.md"
    error = OSError(errno.EINVAL, "The name of the file cannot be resolved by the system")
    error.winerror = 1921
    actual_realpath = os.path.realpath

    def realpath(value, *, strict=False):
        if strict and Path(value).name == "named.md":
            raise error
        return actual_realpath(value, strict=strict)

    monkeypatch.setattr("skillevaluator.reporting.sarif_reporter.os.path.realpath", realpath)

    assert _resolve_for_reporting(path) == expected
    assert _resolve_for_reporting(_resolve_for_reporting(path)) == expected
