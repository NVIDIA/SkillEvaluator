# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security and formatting regressions for reporter output."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting import base
from skillevaluator.reporting.base import ReporterBase
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.tier1.commands import emit_reports


class _StaticReporter(ReporterBase):
    @property
    def name(self) -> str:
        return "static"

    def render(self, result: ValidationResult) -> str:
        del result
        return "new report"

    def render_all(self, results: list[ValidationResult]) -> str:
        del results
        return "new report"


def test_save_rejects_existing_output_symlink_without_touching_target(tmp_path: Path) -> None:
    external = tmp_path / "external.txt"
    external.write_text("keep me", encoding="utf-8")
    output = tmp_path / "report.txt"
    output.symlink_to(external)

    with pytest.raises(ValueError, match=r"report.*symlink|symlink.*report|reparse"):
        _StaticReporter().save([], output)

    assert output.is_symlink()
    assert external.read_text(encoding="utf-8") == "keep me"


def test_save_rejects_intermediate_symlink_without_writing_outside(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    linked_parent = tmp_path / "reports"
    linked_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match=r"report.*symlink|symlink.*report|reparse"):
        _StaticReporter().save([], linked_parent / "report.txt")

    assert not (external / "report.txt").exists()


def test_emit_reports_does_not_create_directories_through_symlinked_parent(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    linked_parent = tmp_path / "reports"
    linked_parent.symlink_to(external, target_is_directory=True)
    output_dir = linked_parent / "new" / "nested"

    with pytest.raises(ValueError, match=r"report.*symlink|symlink.*report|reparse"):
        emit_reports(
            [ValidationResult(validator_name="Static")],
            report_formats=("json",),
            output_dir=output_dir,
            basename="report",
        )

    assert not (external / "new").exists()


def test_save_keeps_existing_report_when_atomic_write_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.txt"
    output.write_text("previous report", encoding="utf-8")
    real_write = os.write
    calls = 0

    def interrupted_write(descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, payload[:3])
        raise OSError("simulated interrupted report write")

    monkeypatch.setattr(os, "write", interrupted_write)

    with pytest.raises(OSError, match="simulated interrupted report write"):
        _StaticReporter().save([], output)

    assert output.read_text(encoding="utf-8") == "previous report"
    assert list(tmp_path.glob(".report.txt.*.tmp")) == []


def test_checked_fallback_rejects_windows_reparse_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    real_lstat = Path.lstat

    def windows_lstat(path: Path):
        metadata = real_lstat(path)
        if path == reports:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
            )
        return metadata

    monkeypatch.setattr(base, "_USE_POSIX_DESCRIPTOR_WRITES", False, raising=False)
    monkeypatch.setattr(Path, "lstat", windows_lstat)

    with pytest.raises(ValueError, match=r"report.*reparse|reparse.*report"):
        _StaticReporter().save([], reports / "report.txt")

    assert not (reports / "report.txt").exists()


def test_checked_fallback_atomically_replaces_regular_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.txt"
    output.write_text("previous report", encoding="utf-8")
    monkeypatch.setattr(base, "_USE_POSIX_DESCRIPTOR_WRITES", False, raising=False)

    _StaticReporter().save([], output)

    assert output.read_text(encoding="utf-8") == "new report"
    assert list(tmp_path.glob(".report.txt.*.tmp")) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable to Windows")
def test_atomic_replace_preserves_existing_report_permissions(tmp_path: Path) -> None:
    output = tmp_path / "report.txt"
    output.write_text("previous report", encoding="utf-8")
    output.chmod(0o664)

    _StaticReporter().save([], output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o664


def test_markdown_finding_table_uses_one_safe_row_for_multiline_content() -> None:
    result = ValidationResult(validator_name="Context Optimization Check")
    result.add_finding(
        Finding(
            category="DUPLICATION",
            severity=Severity.MEDIUM,
            check_name="duplicate-content",
            message="Repeated block:\nalpha | `beta`\r\ngamma",
            file_path="references/a|b`c.md",
        )
    )

    output = MarkdownReporter(include_timestamp=False, include_details=False).render(result)
    finding_rows = [line for line in output.splitlines() if "MEDIUM" in line]

    assert finding_rows == [
        f"| {Severity.MEDIUM.emoji} MEDIUM | Repeated block:<br>alpha &#124; &#96;beta&#96;<br>gamma "
        "| <code>references/a&#124;b&#96;c.md</code> |"
    ]
