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


@pytest.mark.parametrize("path_without_line", ["SKILL.md", "guide.txt "])
def test_location_with_and_without_line(path_without_line: str) -> None:
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
            file_path=path_without_line,
        )
    )

    with_text = GitHubReporter().render_all([with_line])
    assert "file=config/settings.py" in with_text
    assert "line=15" in with_text
    without_text = GitHubReporter().render_all([without_line])
    warning = next(line for line in _commands(without_text) if line.startswith("::warning "))
    properties = _runner_properties(warning)
    assert properties["file"] == path_without_line
    assert "line" not in properties


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


def _advisory_dedup_failure() -> ValidationResult:
    result = ValidationResult(validator_name="Tier 2 Dedup")
    result.add_error("SKILL.md duplicates an existing skill")
    result.metadata["gating"] = {"tier": 2, "blocking": False}
    return result


def test_advisory_failure_does_not_fail_the_skill() -> None:
    schema = ValidationResult(validator_name="SCHEMA")
    schema.metadata["gating"] = {"tier": 1, "blocking": True}

    text = GitHubReporter().render_all([schema, _advisory_dedup_failure()])

    assert "::error" not in text
    assert "1 skill checked, 1 passed, 0 failed, 0 incomplete" in _summary(text)


def test_advisory_failure_keeps_required_incomplete_scan() -> None:
    scan = ValidationResult(validator_name="Security Scan")
    scan.mark_scan_incomplete("skillspector")
    scan.metadata["gating"] = {"tier": 1, "blocking": True}

    text = GitHubReporter().render_all([scan, _advisory_dedup_failure()])

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
    assert "10 errors, 10 warnings, and 10 notices" in summary
    assert "Rerun with a file report such as -r github,json" in summary


def test_info_findings_leave_room_for_the_summary_notice() -> None:
    result = ValidationResult(validator_name="QUALITY")
    for index in range(12):
        result.add_finding(
            Finding(
                category="QUALITY",
                severity=Severity.INFO,
                check_name=f"info-{index:02d}",
                message=f"info {index}",
                file_path="SKILL.md",
                line_number=index + 1,
            )
        )
    text = GitHubReporter().render_all([result])
    notices = [line for line in _commands(text) if line.startswith("::notice")]
    assert len(notices) == 10
    assert notices[-1].startswith("::notice title=SkillEvaluator::")
    assert "3 more were not annotated" in notices[-1]
    assert "10 notices" in notices[-1]


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
    for line in range(1, 12):
        result.add_finding(
            Finding(
                category="SCHEMA",
                severity=Severity.HIGH,
                check_name="frontmatter",
                message="missing name",
                file_path="SKILL.md",
                line_number=line,
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
    assert "See the requested file reports (sarif)" in _summary(captured.out)
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


def _bad_schema(_path: Path, **_kwargs) -> list[ValidationResult]:
    result = ValidationResult(validator_name="SCHEMA")
    for check in ("name", "description"):
        result.add_finding(
            Finding(
                category="SCHEMA",
                severity=Severity.HIGH,
                check_name=check,
                message=f"bad {check} in {_path.name}",
                file_path="SKILL.md",
                line_number=2,
            )
        )
    return [result]


def _catalog(tmp_path: Path, count: int) -> Path:
    catalog = tmp_path / "skills"
    for index in range(count):
        shutil.copytree(_TEXT_ANALYZER, catalog / f"skill-{index:02d}")
    return catalog


def _validate_catalog_args(catalog: Path, reports: Path, workers: int) -> list[str]:
    return [
        "validate",
        str(catalog),
        "--tiers",
        "1",
        "--checks",
        "schema",
        "--workers",
        str(workers),
        "--no-autopilot",
        "-r",
        "github,json",
        "-o",
        str(reports),
    ]


def _assert_one_capped_catalog_stream(output: str, skills: int) -> None:
    commands = _commands(output)
    assert sum(line.startswith("::error") for line in commands) == 10
    summaries = [line for line in commands if line.startswith("::notice title=SkillEvaluator::")]
    assert len(summaries) == 1, output
    assert summaries[0].startswith(f"::notice title=SkillEvaluator::{skills} skills checked, 0 passed, {skills} failed")
    assert f"{skills * 2 - 10} more were not annotated" in summaries[0]


def test_catalog_emits_one_capped_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skillevaluator.cli.run_validation", _bad_schema)
    catalog = _catalog(tmp_path, 12)
    reports = tmp_path / "reports"
    result = CliRunner().invoke(cli, _validate_catalog_args(catalog, reports, workers=1))
    _assert_one_capped_catalog_stream(result.output, 12)
    assert "See the requested file reports (json)" in _summary(result.output)
    assert "file=" in result.output and "SKILL.md" in result.output
    assert len(list(reports.glob("skill-*/*.json"))) == 12


def test_parallel_catalog_returns_worker_annotations_to_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skillevaluator.cli.run_validation", _bad_schema)
    _inline_process_pool(monkeypatch)
    catalog = _catalog(tmp_path, 12)
    reports = tmp_path / "reports"
    result = CliRunner().invoke(cli, _validate_catalog_args(catalog, reports, workers=2))
    _assert_one_capped_catalog_stream(result.output, 12)
    assert len(list(reports.glob("skill-*/*.json"))) == 12


def _inline_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run parallel catalog jobs in-process so worker results reach the parent unchanged."""
    from concurrent.futures import Future

    import skillevaluator.cli as cli_module

    class InlineExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> InlineExecutor:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def submit(self, fn, job):
            future: Future = Future()
            future.set_result(fn(job))
            return future

    monkeypatch.setattr(cli_module, "ProcessPoolExecutor", InlineExecutor)


def _catalog_with_ids(tmp_path: Path, ids: list[str]) -> Path:
    catalog = tmp_path / "skills"
    for index, metadata_id in enumerate(ids):
        skill = catalog / f"skill-{index:02d}"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: skill-{index:02d}\n"
            "description: Use when a test needs a valid skill that declares a registry metadata id.\n"
            f"metadata:\n  author: Test Team <team@example.com>\n  id: {metadata_id}\n---\n\n# Skill\n",
            encoding="utf-8",
        )
    return catalog


@pytest.mark.parametrize("workers", [1, 2])
def test_catalog_summary_counts_duplicate_metadata_id_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    import json

    _inline_process_pool(monkeypatch)
    catalog = _catalog_with_ids(tmp_path, ["shared-id", "shared-id"])
    reports = tmp_path / "reports"
    args = [*_validate_catalog_args(catalog, reports, workers), "--profile", "registry"]
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    assert (summary["passed"], summary["failed"]) == (0, 2)
    commands = _commands(result.output)
    duplicates = [line for line in commands if "metadata_id_duplicate" in line]
    assert len(duplicates) == 2
    assert all(line.startswith("::error") for line in duplicates)
    assert any("file=skill-00/SKILL.md" in line and "skill-01" in line for line in duplicates)
    summaries = [line for line in commands if line.startswith("::notice title=SkillEvaluator::")]
    assert summaries == ["::notice title=SkillEvaluator::2 skills checked, 0 passed, 2 failed, 0 incomplete"]


def test_catalog_unique_metadata_ids_still_pass(tmp_path: Path) -> None:
    catalog = _catalog_with_ids(tmp_path, ["first-id", "second-id"])
    reports = tmp_path / "reports"
    args = [*_validate_catalog_args(catalog, reports, 1), "--profile", "registry"]
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert "metadata_id_duplicate" not in result.output
    assert _summary(result.output).endswith("2 skills checked, 2 passed, 0 failed, 0 incomplete")


def test_advisory_duplicate_metadata_id_is_a_warning(tmp_path: Path) -> None:
    catalog = _catalog_with_ids(tmp_path, ["shared-id", "shared-id"])
    reports = tmp_path / "reports"
    result = CliRunner().invoke(cli, _validate_catalog_args(catalog, reports, 1))

    assert result.exit_code == 0, result.output
    duplicates = [line for line in _commands(result.output) if "metadata_id_duplicate" in line]
    assert len(duplicates) == 2
    assert all(line.startswith("::warning") for line in duplicates)
    assert _summary(result.output).endswith("2 skills checked, 2 passed, 0 failed, 0 incomplete")


def _runner_properties(command: str) -> dict[str, str]:
    """Parse a workflow command's properties the way the Actions runner does.

    Mirrors ActionCommand.TryParseV2: the property list is trimmed as a whole,
    split on commas and then on the first ``=``, and each value is unescaped.
    """
    end = command.index("::", 2)
    info = command[2:end]
    properties: dict[str, str] = {}
    for pair in info[info.index(" ") + 1 :].strip().split(","):
        key, sep, value = pair.partition("=")
        if not pair or not sep or not key or not value:
            continue
        for escaped, raw in (("%0D", "\r"), ("%0A", "\n"), ("%3A", ":"), ("%2C", ","), ("%25", "%")):
            value = value.replace(escaped, raw)
        properties[key] = value
    return properties


@pytest.mark.skipif(os.name == "nt", reason="Windows does not distinguish files with trailing spaces")
def test_finding_annotations_preserve_trailing_spaces_in_tracked_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skill = repo / "skills" / "text-analyzer"
    shutil.copytree(_TEXT_ANALYZER, skill)
    (skill / "guide.txt").touch()
    (skill / "guide.txt ").touch()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)

    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(skill),
            "--checks",
            "schema",
            "--no-autopilot",
            "-r",
            "github",
            "-o",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.output
    warnings = [
        _runner_properties(line)["file"]
        for line in _commands(result.output)
        if line.startswith("::warning ") and "/unexpected_file" in line
    ]
    assert sorted(warnings) == ["skills/text-analyzer/guide.txt", "skills/text-analyzer/guide.txt "]


@pytest.mark.skipif(os.name == "nt", reason="creating a directory symlink needs privileges on Windows")
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("target", ["outside/external-skill", "repo/source"])
@pytest.mark.parametrize("link_name", ["linked", "linked "])
def test_linked_directory_discovery_error_points_at_the_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int, target: str, link_name: str
) -> None:
    _inline_process_pool(monkeypatch)
    linked = tmp_path / target
    shutil.copytree(_TEXT_ANALYZER, linked)
    repo = tmp_path / "repo"
    catalog = repo / "skills"
    shutil.copytree(_TEXT_ANALYZER, catalog / "skill-00")
    (catalog / link_name).symlink_to(linked, target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "-s"], check=True, capture_output=True, text=True
    ).stdout
    link_path = next(entry.split("\t", 1)[1] for entry in tracked.split("\0") if entry.startswith("120000 "))
    assert link_path == f"skills/{link_name}"
    reports = tmp_path / "reports"
    result = CliRunner().invoke(cli, _validate_catalog_args(catalog, reports, workers))

    assert result.exit_code == 1, result.output
    discovery = [line for line in _commands(result.output) if "catalog/discovery" in line]
    assert len(discovery) == 1, result.output
    assert discovery[0].startswith("::error "), discovery[0]
    parsed = _runner_properties(discovery[0])
    assert parsed["file"] == link_path, discovery[0]
    assert parsed["title"] == "catalog/discovery", discovery[0]
    assert "line" not in parsed, discovery[0]


def _collected(level: str, counts: list[int]) -> list[dict]:
    from skillevaluator.reporting.github_reporter import catalog_annotation

    item = catalog_annotation(check="c", message="m", severity="high", blocking=level == "error")
    return [{"annotations": [item], "counts": counts}]


def test_catalog_counts_keep_incomplete_and_apply_catalog_failures(capsys: pytest.CaptureFixture[str]) -> None:
    from skillevaluator.reporting.github_reporter import print_collected_annotations

    print_collected_annotations(
        {
            "incomplete": _collected("warning", [1, 0, 0, 1]),
            "incomplete-and-duplicate": _collected("warning", [1, 0, 0, 1]),
            "passed-then-duplicate": _collected("warning", [1, 1, 0, 0]),
            "own-failure": _collected("error", [1, 0, 1, 0]),
            "crashed": [],
            "passed": _collected("warning", [1, 1, 0, 0]),
        },
        failed={"incomplete", "incomplete-and-duplicate", "passed-then-duplicate", "own-failure", "crashed"},
        catalog_failed={"incomplete-and-duplicate", "passed-then-duplicate"},
    )
    assert _summary(capsys.readouterr().out).endswith("6 skills checked, 1 passed, 4 failed, 1 incomplete")


def test_catalog_findings_share_the_annotation_cap(capsys: pytest.CaptureFixture[str]) -> None:
    from skillevaluator.reporting.github_reporter import catalog_annotation, print_collected_annotations

    extra = [
        catalog_annotation(check="metadata_id_duplicate", message=f"dup {index}", severity="high", blocking=True)
        for index in range(4)
    ]
    print_collected_annotations(
        {f"skill-{index}": _collected("error", [1, 0, 1, 0]) for index in range(8)},
        failed={f"skill-{index}" for index in range(8)},
        catalog_failed={"skill-0", "skill-1"},
        catalog_annotations=extra,
    )
    out = capsys.readouterr().out
    assert sum(line.startswith("::error") for line in _commands(out)) == 10
    assert "2 more were not annotated" in _summary(out)


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


def _write_license_catalog_skill(skill: Path, *, with_license: bool) -> None:
    skill.mkdir(parents=True, exist_ok=True)
    license_line = "license: Apache-2.0\n" if with_license else ""
    (skill / "SKILL.md").write_text(
        f"---\nname: {skill.name}\n"
        "description: Use when a test checks GitHub annotations for catalog license findings.\n"
        f"{license_line}---\n\n# {skill.name}\n",
        encoding="utf-8",
    )


def _run_license_catalog(catalog: Path, reports: Path, workers: int) -> str:
    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(catalog),
            "--tiers",
            "1",
            "--checks",
            "license",
            "--workers",
            str(workers),
            "--no-autopilot",
            "-r",
            "github,json",
            "-o",
            str(reports),
        ],
    )
    assert result.exit_code == 0, result.output
    return result.output


@pytest.mark.parametrize("in_git_repo", [True, False], ids=["git-repo", "plain-dir"])
def test_catalog_missing_license_annotation_identifies_skill_across_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_git_repo: bool
) -> None:
    _inline_process_pool(monkeypatch)
    root = tmp_path / "repo" if in_git_repo else tmp_path
    catalog = root / "skills"
    if in_git_repo:
        catalog.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    else:
        monkeypatch.setattr("skillevaluator.cli.resolve_git_root", lambda _path: None)
    expected_prefix = "skills/" if in_git_repo else ""

    _write_license_catalog_skill(catalog / "alpha", with_license=False)
    _write_license_catalog_skill(catalog / "bravo", with_license=True)
    alpha_w1 = _run_license_catalog(catalog, tmp_path / "reports-alpha-1", workers=1)
    alpha_w2 = _run_license_catalog(catalog, tmp_path / "reports-alpha-2", workers=2)
    assert _commands(alpha_w1) == _commands(alpha_w2)
    alpha_warnings = [line for line in _commands(alpha_w1) if line.startswith("::warning ")]
    assert len(alpha_warnings) == 1, alpha_w1
    alpha_props = _runner_properties(alpha_warnings[0])
    assert alpha_props["file"] == f"{expected_prefix}alpha/SKILL.md"
    assert "line" not in alpha_props
    assert "bravo" not in alpha_warnings[0]

    _write_license_catalog_skill(catalog / "alpha", with_license=True)
    _write_license_catalog_skill(catalog / "bravo", with_license=False)
    bravo_w1 = _run_license_catalog(catalog, tmp_path / "reports-bravo-1", workers=1)
    bravo_w2 = _run_license_catalog(catalog, tmp_path / "reports-bravo-2", workers=2)
    assert _commands(bravo_w1) == _commands(bravo_w2)
    bravo_warnings = [line for line in _commands(bravo_w1) if line.startswith("::warning ")]
    assert len(bravo_warnings) == 1, bravo_w1
    bravo_props = _runner_properties(bravo_warnings[0])
    assert bravo_props["file"] == f"{expected_prefix}bravo/SKILL.md"
    assert "line" not in bravo_props
    assert "alpha" not in bravo_warnings[0]
    assert _commands(alpha_w1) != _commands(bravo_w1)


def test_pathless_finding_falls_back_to_catalog_skill_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    catalog = tmp_path / "external-skills"
    _write_license_catalog_skill(catalog / "group" / "alpha", with_license=False)
    _write_license_catalog_skill(catalog / "group" / "bravo", with_license=False)

    finding = Finding(
        category="LICENSE",
        severity=Severity.MEDIUM,
        check_name="missing_license",
        message="No license information found",
        file_path=None,
    )
    result = ValidationResult(validator_name="License Compliance")
    result.add_finding(finding)

    alpha_text = GitHubReporter(
        workspace_root=workspace,
        scan_root=catalog / "group" / "alpha",
        catalog_root=catalog,
    ).render_all([result])
    bravo_text = GitHubReporter(
        workspace_root=workspace,
        scan_root=catalog / "group" / "bravo",
        catalog_root=catalog,
    ).render_all([result])

    alpha_warning = next(line for line in _commands(alpha_text) if line.startswith("::warning "))
    bravo_warning = next(line for line in _commands(bravo_text) if line.startswith("::warning "))
    assert "file" not in _runner_properties(alpha_warning)
    assert "file" not in _runner_properties(bravo_warning)
    assert "::group/alpha: No license information found" in alpha_warning
    assert "::group/bravo: No license information found" in bravo_warning
    assert alpha_warning != bravo_warning


def test_catalog_nested_git_repos_anchor_annotations_to_catalog_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inline_process_pool(monkeypatch)
    outer = tmp_path / "outer"
    repo_a = outer / "repoA"
    repo_b = outer / "repoB"
    repo_a.mkdir(parents=True)
    repo_b.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    subprocess.run(["git", "init", "-q", str(repo_a)], check=True)
    subprocess.run(["git", "init", "-q", str(repo_b)], check=True)

    _write_license_catalog_skill(repo_a / "alpha", with_license=False)
    _write_license_catalog_skill(repo_b / "alpha", with_license=True)
    repo_a_w1 = _run_license_catalog(outer, tmp_path / "reports-a-1", workers=1)
    repo_a_w2 = _run_license_catalog(outer, tmp_path / "reports-a-2", workers=2)
    assert _commands(repo_a_w1) == _commands(repo_a_w2)
    repo_a_warnings = [line for line in _commands(repo_a_w1) if line.startswith("::warning ")]
    assert len(repo_a_warnings) == 1, repo_a_w1
    repo_a_props = _runner_properties(repo_a_warnings[0])
    assert repo_a_props["file"] == "repoA/alpha/SKILL.md"
    assert "line" not in repo_a_props
    assert "repoB" not in repo_a_warnings[0]

    _write_license_catalog_skill(repo_a / "alpha", with_license=True)
    _write_license_catalog_skill(repo_b / "alpha", with_license=False)
    repo_b_w1 = _run_license_catalog(outer, tmp_path / "reports-b-1", workers=1)
    repo_b_w2 = _run_license_catalog(outer, tmp_path / "reports-b-2", workers=2)
    assert _commands(repo_b_w1) == _commands(repo_b_w2)
    repo_b_warnings = [line for line in _commands(repo_b_w1) if line.startswith("::warning ")]
    assert len(repo_b_warnings) == 1, repo_b_w1
    repo_b_props = _runner_properties(repo_b_warnings[0])
    assert repo_b_props["file"] == "repoB/alpha/SKILL.md"
    assert "line" not in repo_b_props
    assert "repoA" not in repo_b_warnings[0]
    assert _commands(repo_a_w1) != _commands(repo_b_w1)

    # Regular finding paths across nested Git roots also anchor to the catalog workspace;
    # member paths outside that workspace omit file= and prefix the catalog-relative skill path.
    outside = tmp_path / "outside" / "secret.md"
    outside.parent.mkdir()
    outside.write_text("x\n", encoding="utf-8")
    regular = ValidationResult(validator_name="SCHEMA")
    regular.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="frontmatter",
            message="Invalid frontmatter",
            file_path="SKILL.md",
            line_number=2,
        )
    )
    regular.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.MEDIUM,
            check_name="external_ref",
            message="Referenced external file",
            file_path=str(outside),
            line_number=5,
        )
    )
    subcatalog = outer / "catalog"
    nested_repo = subcatalog / "repoA"
    nested_repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(nested_repo)], check=True)
    _write_license_catalog_skill(nested_repo / "alpha", with_license=True)
    nested_text = GitHubReporter(
        workspace_root=nested_repo,
        scan_root=nested_repo / "alpha",
        catalog_root=subcatalog,
        catalog_workspace_root=outer,
    ).render_all([regular])
    nested_error = next(line for line in _commands(nested_text) if line.startswith("::error "))
    nested_warning = next(line for line in _commands(nested_text) if line.startswith("::warning "))
    error_props = _runner_properties(nested_error)
    assert error_props["file"] == "catalog/repoA/alpha/SKILL.md"
    assert error_props["line"] == "2"
    warning_props = _runner_properties(nested_warning)
    assert "file" not in warning_props
    assert "line" not in warning_props
    assert "::repoA/alpha: Referenced external file" in nested_warning


def test_merged_plugin_finding_annotates_real_workspace_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    plugin = repo / "plugin"
    skill_md = plugin / "skills" / "demo" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text("# Missing frontmatter\n", encoding="utf-8")

    skill_result = ValidationResult(validator_name="SCHEMA")
    skill_result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="frontmatter_missing",
            message="Missing YAML frontmatter",
            file_path=str(skill_md),
            line_number=1,
        )
    )
    plugin_result = ValidationResult(validator_name="PLUGIN_SCHEMA")
    plugin_result.merge_with_prefix(skill_result, "demo")
    assert plugin_result.findings[0].file_path == f"[demo] {skill_md}"

    text = GitHubReporter(workspace_root=repo, scan_root=plugin).render_all([plugin_result])
    error = next(line for line in _commands(text) if line.startswith("::error "))
    props = _runner_properties(error)
    assert props["file"] == "plugin/skills/demo/SKILL.md"
    assert props["line"] == "1"
    assert "[demo]" not in error

    nested_result = ValidationResult(validator_name="PLUGIN_SCHEMA")
    nested_result.merge_with_prefix(plugin_result, "bundle")
    nested_text = GitHubReporter(workspace_root=repo, scan_root=repo).render_all([nested_result])
    nested_error = next(line for line in _commands(nested_text) if line.startswith("::error "))
    nested_props = _runner_properties(nested_error)
    assert nested_props["file"] == "plugin/skills/demo/SKILL.md"
    assert nested_props["line"] == "1"
    assert "[bundle]" not in nested_error
    assert "[demo]" not in nested_error


def test_merged_plugin_finding_outside_workspace_omits_bogus_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    plugin = repo / "plugin"
    plugin.mkdir(parents=True)
    outside = tmp_path / "outside" / "skills" / "demo" / "SKILL.md"
    outside.parent.mkdir(parents=True)
    outside.write_text("# Missing frontmatter\n", encoding="utf-8")

    skill_result = ValidationResult(validator_name="SCHEMA")
    skill_result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="frontmatter_missing",
            message="Missing YAML frontmatter",
            file_path=str(outside),
            line_number=4,
        )
    )
    skill_result.add_finding(
        Finding(
            category="SECRET",
            severity=Severity.MEDIUM,
            check_name="relative_member_path",
            message="Relative path inside member",
            file_path="SKILL.md",
            line_number=2,
        )
    )
    plugin_result = ValidationResult(validator_name="PLUGIN_SCHEMA")
    plugin_result.merge_with_prefix(skill_result, "demo")

    text = GitHubReporter(workspace_root=repo, scan_root=plugin).render_all([plugin_result])
    error = next(line for line in _commands(text) if line.startswith("::error "))
    warning = next(line for line in _commands(text) if line.startswith("::warning "))
    for command in (error, warning):
        props = _runner_properties(command)
        assert "file" not in props
        assert "line" not in props
        assert "[demo]" not in command
    assert "::plugin/demo: Missing YAML frontmatter" in error
    assert "::plugin/demo: Relative path inside member" in warning

    anchor_skill = repo / "skills" / "parent"
    anchor_skill.mkdir(parents=True)
    (anchor_skill / "SKILL.md").write_text("---\nname: parent\ndescription: parent\n---\n", encoding="utf-8")
    anchored_text = GitHubReporter(workspace_root=repo, scan_root=anchor_skill).render_all([plugin_result])
    anchored_error = next(line for line in _commands(anchored_text) if line.startswith("::error "))
    anchored_props = _runner_properties(anchored_error)
    assert anchored_props["file"] == "skills/parent/SKILL.md"
    assert "line" not in anchored_props
    assert "::demo: Missing YAML frontmatter" in anchored_error
