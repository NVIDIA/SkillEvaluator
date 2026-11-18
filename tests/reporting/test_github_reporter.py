# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GitHub Actions annotations reporter."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.github_reporter import GitHubReporter
from skillevaluator.tier1.commands import emit_reports

_TEXT_ANALYZER = Path(__file__).resolve().parents[2] / "src/skillevaluator/tier3/reference_skills/text-analyzer"


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line]


def _commands(text: str) -> list[str]:
    return [line for line in _lines(text) if line.startswith(("::error", "::warning", "::notice"))]


def _summary(text: str) -> str:
    notices = [line for line in _commands(text) if line.startswith("::notice title=SkillEvaluator::")]
    assert notices, text
    return notices[-1]


def test_name_and_stdout_only_extension() -> None:
    reporter = GitHubReporter()
    assert reporter.name == "github"
    assert reporter.get_file_extension() == ""


def test_severity_mapping() -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.CRITICAL,
            check_name="crit",
            message="c",
            file_path="SKILL.md",
            line_number=1,
        )
    )
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="high",
            message="h",
            file_path="SKILL.md",
            line_number=2,
        )
    )
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.MEDIUM,
            check_name="med",
            message="m",
            file_path="SKILL.md",
            line_number=3,
        )
    )
    result.add_finding(
        Finding(
            category="SCHEMA", severity=Severity.LOW, check_name="low", message="l", file_path="SKILL.md", line_number=4
        )
    )
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.INFO,
            check_name="info",
            message="i",
            file_path="SKILL.md",
            line_number=5,
        )
    )

    text = GitHubReporter().render_all([result])
    levels = []
    for line in _commands(text):
        if "title=SkillEvaluator::" in line:
            continue
        if line.startswith("::error"):
            levels.append("error")
        elif line.startswith("::warning"):
            levels.append("warning")
        elif line.startswith("::notice"):
            levels.append("notice")
    assert levels.count("error") == 2
    assert levels.count("warning") == 2
    assert levels.count("notice") >= 1
    assert "title=SCHEMA/crit" in text
    assert "::notice title=SkillEvaluator::" in text


def test_location_with_and_without_line() -> None:
    with_line = ValidationResult(validator_name="SECRETS")
    with_line.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.CRITICAL,
            check_name="aws-access-key-id",
            message="AWS Access Key ID detected",
            file_path="config/settings.py",
            line_number=15,
        )
    )
    without_line = ValidationResult(validator_name="QUALITY")
    without_line.add_finding(
        Finding(
            category="QUALITY",
            severity=Severity.MEDIUM,
            check_name="quality_reliability",
            message="No troubleshooting section documented",
            file_path="SKILL.md",
        )
    )

    with_text = GitHubReporter().render_all([with_line])
    assert "file=config/settings.py" in with_text
    assert "line=15" in with_text
    without_text = GitHubReporter().render_all([without_line])
    assert "file=SKILL.md" in without_text
    assert "line=" not in without_text.split("file=SKILL.md", 1)[1].split("::", 1)[0]


def test_end_line_when_recorded() -> None:
    result = ValidationResult(validator_name="QUALITY")
    result.add_finding(
        Finding(
            category="QUALITY",
            severity=Severity.HIGH,
            check_name="span",
            message="range",
            file_path="SKILL.md",
            line_number=4,
            metadata={"end_line": 8},
        )
    )
    text = GitHubReporter().render_all([result])
    assert "line=4" in text
    assert "endLine=8" in text


def test_path_outside_repo_omits_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skill = repo / "skills" / "demo"
    skill.mkdir(parents=True)
    outside = tmp_path / "other" / "secret.md"
    outside.parent.mkdir()
    outside.write_text("x\n", encoding="utf-8")
    result = ValidationResult(validator_name="SECRETS")
    result.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.HIGH,
            check_name="leak",
            message="secret outside",
            file_path=str(outside),
            line_number=1,
        )
    )
    text = GitHubReporter(workspace_root=repo, scan_root=skill).render_all([result])
    command = next(line for line in _commands(text) if line.startswith("::error"))
    assert "file=" not in command
    assert "secret outside" in command


def test_paths_are_relative_to_repository_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skill = repo / "skills" / "demo"
    skill.mkdir(parents=True)
    result = ValidationResult(validator_name="QUALITY")
    result.add_finding(
        Finding(
            category="QUALITY",
            severity=Severity.HIGH,
            check_name="spacing",
            message="issue",
            file_path="SKILL.md",
            line_number=2,
        )
    )
    text = GitHubReporter(workspace_root=repo, scan_root=skill).render_all([result])
    assert "file=skills/demo/SKILL.md" in text
    assert "line=2" in text


def test_gate_failure_is_error_when_finding_is_warning() -> None:
    result = ValidationResult(validator_name="QUALITY")
    result.add_finding(
        Finding(
            category="QUALITY",
            severity=Severity.MEDIUM,
            check_name="quality_reliability",
            message="No troubleshooting section documented",
            file_path="SKILL.md",
        )
    )
    result.fail_gate("Quality score 71.0/100 is below the required 80/100 in SKILL.md")

    text = GitHubReporter().render_all([result])
    assert any(line.startswith("::warning") and "quality_reliability" in line for line in _commands(text))
    assert any(line.startswith("::error") and "71.0/100" in line for line in _commands(text))


def test_incomplete_scan_is_warning() -> None:
    result = ValidationResult(validator_name="Security Scan")
    result.mark_scan_incomplete("skillspector")
    result.add_warning("skillspector timed out after 300 seconds")

    text = GitHubReporter().render_all([result])
    warning = next(line for line in _commands(text) if line.startswith("::warning"))
    assert "did not complete" in warning or "timed out" in warning
    assert "incomplete/skillspector" in warning
    assert "::error" not in text.split("::notice title=SkillEvaluator::")[0]
    assert "1 skill checked, 0 passed, 0 failed, 1 incomplete" in _summary(text)


def test_escaping_percent_cr_lf_colon_comma() -> None:
    result = ValidationResult(validator_name="SCHEMA:group,name")
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="percent",
            message="100%\r\nbad, path",
            file_path="docs/a,b%.md",
            line_number=1,
        )
    )
    text = GitHubReporter().render_all([result])
    command = next(line for line in _commands(text) if line.startswith("::error"))
    assert "%25" in command
    assert "%0D" in command
    assert "%0A" in command
    assert "file=docs/a%2Cb%25.md" in command
    assert "\r" not in command
    assert command.count("\n") == 0


def test_malicious_finding_stays_on_one_escaped_line() -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="inject",
            message="ok\n::add-mask::x and ::stop-commands::tok",
            file_path="SKILL.md",
            line_number=1,
        )
    )
    text = GitHubReporter().render_all([result])
    lines = _lines(text)
    assert all(not line.startswith(("::add-mask", "::stop-commands", "::set-output", "::add-path")) for line in lines)
    error = next(line for line in lines if line.startswith("::error"))
    assert error.count("\n") == 0
    assert "%0A::add-mask::x" in error
    assert "::stop-commands::tok" in error
    assert "\n::add-mask::x" not in text
    assert "\n::stop-commands::tok" not in text


def test_credential_masking() -> None:
    result = ValidationResult(validator_name="SECRETS")
    result.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.CRITICAL,
            check_name="aws-access-key-id",
            message="Found AKIATESTKEYTESTKEYTEST",
            file_path="config.py",
            line_number=3,
        )
    )
    text = GitHubReporter().render_all([result])
    assert "AKIATESTKEYTESTKEYTEST" not in text
    assert "aws-access-key-<redacted>" in text


def test_ordering_and_ten_ten_cap() -> None:
    result = ValidationResult(validator_name="QUALITY")
    for index in range(12):
        result.add_finding(
            Finding(
                category="QUALITY",
                severity=Severity.HIGH,
                check_name=f"high-{index:02d}",
                message=f"high {index}",
                file_path="b.md" if index % 2 else "a.md",
                line_number=20 - index,
            )
        )
    for index in range(12):
        result.add_finding(
            Finding(
                category="QUALITY",
                severity=Severity.LOW,
                check_name=f"low-{index:02d}",
                message=f"low {index}",
                file_path="z.md",
                line_number=index + 1,
            )
        )
    text = GitHubReporter().render_all([result])
    errors = [line for line in _commands(text) if line.startswith("::error")]
    warnings = [line for line in _commands(text) if line.startswith("::warning")]
    assert len(errors) == 10
    assert len(warnings) == 10
    paths = [line.split("file=", 1)[1].split(",", 1)[0] for line in errors]
    assert paths == sorted(paths) or paths[0] <= paths[-1]
    summary = _summary(text)
    assert "4 more were not annotated" in summary
    assert "10 errors and 10 warnings" in summary
    assert "see the other reports" in summary


def test_folder_of_skills_keeps_distinct_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    first = repo / "skills" / "alpha"
    second = repo / "skills" / "beta"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    alpha = ValidationResult(validator_name="SCHEMA")
    alpha.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="name",
            message="alpha bad",
            file_path="SKILL.md",
            line_number=2,
        )
    )
    beta = ValidationResult(validator_name="SCHEMA")
    beta.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.MEDIUM,
            check_name="name",
            message="beta bad",
            file_path="SKILL.md",
            line_number=4,
        )
    )
    text = GitHubReporter(workspace_root=repo, scan_root=first).render_all([alpha])
    text += GitHubReporter(workspace_root=repo, scan_root=second).render_all([beta])
    assert "file=skills/alpha/SKILL.md" in text
    assert "file=skills/beta/SKILL.md" in text


def test_emit_reports_prints_github_and_writes_sarif(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="frontmatter",
            message="missing name",
            file_path="SKILL.md",
            line_number=1,
        )
    )
    output_dir = tmp_path / "reports"
    emit_reports(
        [result],
        report_formats=("cli", "github", "sarif"),
        output_dir=output_dir,
        basename="skillevaluator-quality",
        announce_paths=False,
    )
    captured = capsys.readouterr()
    assert "::error" in captured.out
    assert "title=SkillEvaluator::" in captured.out
    matches = list(output_dir.glob("*.sarif.json"))
    assert matches == [output_dir / "skillevaluator-quality.sarif.json"]
    assert not list(output_dir.glob("*github*"))


def test_emit_reports_github_only_writes_no_report_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_success("ok", "ok")
    output_dir = tmp_path / "reports"
    emit_reports(
        [result],
        report_formats=("github",),
        output_dir=output_dir,
        basename="skillevaluator-quality",
        announce_paths=False,
    )
    captured = capsys.readouterr()
    assert "::notice title=SkillEvaluator::" in captured.out
    assert not output_dir.exists() or list(output_dir.iterdir()) == []


def test_quality_check_cli_prints_github(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    result = CliRunner().invoke(
        cli,
        ["quality-check", str(_TEXT_ANALYZER), "-r", "cli,github,sarif", "-o", str(reports)],
    )
    assert result.exit_code == 0, result.output
    assert "::notice title=SkillEvaluator::" in result.output
    assert list(reports.glob("*.sarif.json"))
    assert not list(reports.glob("*github*"))


def test_single_check_command_paths_are_relative_to_git_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    skill = repo / "skills" / "demo"
    shutil.copytree(_TEXT_ANALYZER, skill)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    def lint(_path: Path) -> list[ValidationResult]:
        result = ValidationResult(validator_name="SCRIPT_LINT")
        result.add_finding(
            Finding(
                category="SCRIPT_LINT",
                severity=Severity.HIGH,
                check_name="script_lint",
                message="Unsafe call",
                file_path="SKILL.md",
                line_number=2,
            )
        )
        return [result]

    monkeypatch.setattr("skillevaluator.cli.run_lint_scripts", lint)
    result = CliRunner().invoke(cli, ["lint-scripts", str(skill), "-r", "github"])
    assert "file=skills/demo/SKILL.md" in result.output
    assert "line=2" in result.output


def test_catalog_folder_emits_annotations_for_each_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = tmp_path / "skills"
    for name in ("group-a/demo", "group-b/demo"):
        skill = catalog / name
        shutil.copytree(_TEXT_ANALYZER, skill)

    def schema(_path: Path, **_kwargs) -> list[ValidationResult]:
        result = ValidationResult(validator_name="SCHEMA")
        result.add_finding(
            Finding(
                category="SCHEMA",
                severity=Severity.HIGH,
                check_name="name",
                message=f"bad {_path.name}",
                file_path="SKILL.md",
                line_number=2,
            )
        )
        return [result]

    monkeypatch.setattr("skillevaluator.cli.run_validation", schema)
    reports = tmp_path / "reports"
    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(catalog),
            "--tiers",
            "1",
            "--checks",
            "schema",
            "--workers",
            "1",
            "--no-autopilot",
            "-r",
            "github",
            "-o",
            str(reports),
        ],
    )
    assert (
        "file=group-a/demo/SKILL.md" in result.output
        or "file=skills/group-a/demo/SKILL.md" in result.output
        or "SKILL.md" in result.output
    )
    assert result.output.count("::error") >= 2
    assert result.output.count("::notice title=SkillEvaluator::") >= 2


@pytest.mark.skipif(os.name == "nt", reason="only POSIX file names carry undecodable bytes as surrogate escapes")
def test_non_utf8_file_name_is_encoded(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    name = "caf\udce9.md"
    result = ValidationResult(validator_name="SCHEMA")
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.LOW,
            check_name="unexpected_file",
            message=f"Unexpected '{name}' in skill root",
            file_path=str(skill / name),
        )
    )
    text = GitHubReporter(workspace_root=tmp_path, scan_root=skill).render_all([result])
    assert "file=skill/caf%25E9.md" in text or "file=skill/caf%E9.md" in text
    assert "Unexpected" in text
    text.encode("utf-8")
