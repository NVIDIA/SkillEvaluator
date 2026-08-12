# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.cli_help import render_help
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.tier3.harbor.runner import format_harbor_view_command
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.secrets import SecretsValidator


def test_top_level_help_lists_primary_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    removed_command = "convert" + "-benchmark"
    tier3_result = CliRunner().invoke(cli, ["tier3", "--help"])

    assert result.exit_code == 0
    assert tier3_result.exit_code == 0
    assert "validate" in result.output
    assert "create-eval-dataset" in result.output
    # `evaluate` is intentionally hidden at top level: `tier3 evaluate` is the
    # advertised spelling (the top-level alias keeps working for scripts).
    assert "\n  evaluate" not in result.output
    assert "evaluate" in tier3_result.output
    assert "tier1" in result.output
    assert "tier2" in result.output
    assert "tier3" in result.output
    assert removed_command not in result.output
    assert removed_command not in tier3_result.output


def test_top_level_help_shows_branded_header() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "SKILLEVALUATOR: SkillEvaluator" in result.output
    assert "https://docs.nvidia.com/skills/skillevaluator/" in result.output
    # Plain text under a non-TTY runner so the output stays greppable.
    assert "\x1b[" not in result.output


def test_top_level_help_groups_commands_by_workflow() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    headings = (
        "Core workflows:",
        "Tier 1 · Static and security:",
        "Tier 2 · Deduplication:",
        "Tier 3 · Live evaluation:",
        "Expert aliases:",
    )
    positions = [result.output.index(heading) for heading in headings]
    assert positions == sorted(positions)

    core = result.output.split(headings[0], 1)[1].split(headings[1], 1)[0]
    assert all(command in core for command in ("validate", "health-check", "doctor", "models"))
    tier3 = result.output.split(headings[3], 1)[1].split(headings[4], 1)[0]
    assert all(command in tier3 for command in ("create-eval-dataset", "compare", "view", "harbor-view"))
    assert "Other commands:" not in result.output


def test_help_emits_color_when_enabled() -> None:
    ctx = click.Context(cli, info_name="skillevaluator", color=True)
    colored = render_help(cli, ctx, width=80)
    plain = render_help(cli, click.Context(cli, info_name="skillevaluator", color=False), width=80)

    assert "\x1b[" in colored
    assert "\x1b[" not in plain
    # No trailing whitespace in the rendered help (parity with Click).
    assert all(line == line.rstrip() for line in plain.splitlines())


def test_tier_alias_help() -> None:
    runner = CliRunner()

    for tier in ("tier1", "tier2", "tier3"):
        result = runner.invoke(cli, [tier, "--help"])
        assert result.exit_code == 0


def test_similarity_help_exposes_catalog_workflow_and_hides_legacy_cache_names() -> None:
    result = CliRunner().invoke(cli, ["similarity-check", "--help"])

    assert result.exit_code == 0
    assert "--catalog" in result.output
    assert "--save-catalog" in result.output
    assert "--cache" not in result.output
    assert "--save-cache" not in result.output


def test_similarity_cli_rejects_nonfinite_and_out_of_range_thresholds(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: sample\ndescription: Sample skill\n---\n")
    runner = CliRunner()

    for value in ("nan", "inf", "-0.1", "1.1"):
        result = runner.invoke(cli, ["similarity-check", str(skill), "--threshold", value])
        assert result.exit_code == 2
        assert "finite and within [0, 1]" in result.output


def test_similarity_cli_rejects_catalog_build_and_query_together(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: sample\ndescription: Sample skill\n---\n")
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")

    result = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(skill),
            "--catalog",
            str(catalog),
            "--save-catalog",
            str(tmp_path / "new.json"),
        ],
    )

    assert result.exit_code == 2
    assert "cannot be used together" in result.output


def test_similarity_cli_handles_pinned_openclaw_compatibility_layout(tmp_path: Path, monkeypatch) -> None:
    collection = tmp_path / "skills"
    autoreview = collection / "autoreview"
    autoreview.mkdir(parents=True)
    (autoreview / "SKILL.md").write_text(
        "---\nname: autoreview\ndescription: Review changes with multiple agents\n---\n"
    )
    (autoreview / "AGENTS.md").write_text("# Shared agent instructions\n")
    try:
        (autoreview / "CLAUDE.md").symlink_to("AGENTS.md")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    second = collection / "second-skill"
    second.mkdir()
    (second / "SKILL.md").write_text("---\nname: second-skill\ndescription: Exercise an unrelated workflow\n---\n")

    embedded_texts: list[str] = []

    def embed_without_network(_self, texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return [[1.0, float(index)] for index, _text in enumerate(texts)]

    monkeypatch.setattr("skillevaluator.embedding.client.EmbeddingClient.embed", embed_without_network)

    result = CliRunner().invoke(cli, ["similarity-check", str(collection), "--type", "skill"])

    assert result.exit_code == 0, result.output
    assert "All validations passed" in result.output
    assert sorted(embedded_texts) == [
        "autoreview: Review changes with multiple agents",
        "second-skill: Exercise an unrelated workflow",
    ]


@pytest.mark.parametrize(
    ("command", "runner_path"),
    [
        pytest.param("similarity-check", "skillevaluator.tier2.commands.run_similarity_check", id="similarity"),
        pytest.param(
            "context-optimization-check",
            "skillevaluator.tier2.commands.run_context_optimization_check",
            id="context",
        ),
        pytest.param("dedup-scan", "skillevaluator.tier2.commands.run_dedup_scan", id="dedup-alias"),
    ],
)
def test_tier2_commands_reject_unresolved_root_symlinks(
    tmp_path: Path,
    monkeypatch,
    command: str,
    runner_path: str,
) -> None:
    target = tmp_path / "real-skill"
    target.mkdir()
    linked_target = tmp_path / "linked-skill"
    try:
        linked_target.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("Tier 2 runner must not receive a linked root")

    monkeypatch.setattr(runner_path, must_not_run)

    result = CliRunner().invoke(cli, [command, str(linked_target)])

    assert result.exit_code == 2, result.output
    assert "symlink or reparse point" in result.output
    assert str(target) not in result.output


def test_validate_rejects_linked_root_before_default_tier2_run(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "real-skill"
    target.mkdir()
    (target / "SKILL.md").write_text(
        "---\nname: sample\ndescription: Sample skill\n---\n",
        encoding="utf-8",
    )
    linked_target = tmp_path / "linked-skill"
    try:
        linked_target.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("Validation must not receive a linked root when Tier 2 is enabled")

    monkeypatch.setattr("skillevaluator.cli.run_validation", must_not_run)

    result = CliRunner().invoke(cli, ["validate", str(linked_target)])

    assert result.exit_code == 2, result.output
    assert "symlink or reparse point" in result.output
    assert str(target) not in result.output


def test_validate_preserves_linked_root_support_when_tier2_is_disabled(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "real-skill"
    target.mkdir()
    (target / "SKILL.md").write_text(
        "---\nname: real-skill\ndescription: Validate the Tier 1-only linked-root compatibility path.\n---\n",
        encoding="utf-8",
    )
    linked_target = tmp_path / "linked-skill"
    try:
        linked_target.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    observed_targets: list[Path] = []

    def record_validation(path: Path, **_kwargs) -> list[ValidationResult]:
        observed_targets.append(path)
        return [ValidationResult(validator_name="schema", passed=True)]

    monkeypatch.setattr("skillevaluator.cli.run_validation", record_validation)

    result = CliRunner().invoke(
        cli,
        ["validate", str(linked_target), "--no-dedup", "--checks", "schema"],
    )

    assert result.exit_code == 0, result.output
    assert "symlink or reparse point" not in result.output
    assert observed_targets == [target.resolve()]


@pytest.mark.parametrize("extension", [".json", ".html", ".md"])
def test_similarity_cli_rejects_catalog_collision_with_every_selected_report(
    tmp_path: Path,
    monkeypatch,
    extension: str,
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    catalog = reports / f"skillevaluator-similarity{extension}"
    catalog.write_text("catalog must survive", encoding="utf-8")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("Similarity must not run when its catalog would be overwritten")

    monkeypatch.setattr("skillevaluator.tier2.commands.run_similarity_check", must_not_run)

    result = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(skill),
            "--save-catalog",
            str(catalog),
            "-r",
            "json,html,markdown",
            "-o",
            str(reports),
        ],
    )

    assert result.exit_code == 2, result.output
    assert "catalog" in result.output.lower()
    assert "report" in result.output.lower()
    assert catalog.read_text(encoding="utf-8") == "catalog must survive"


def test_similarity_cli_rejects_catalog_collision_through_output_directory_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    reports = tmp_path / "real-reports"
    reports.mkdir()
    reports_alias = tmp_path / "reports-alias"
    try:
        reports_alias.symlink_to(reports, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    catalog = reports / "skillevaluator-similarity.json"
    catalog.write_text("catalog must survive", encoding="utf-8")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("Similarity must not run when resolved output paths collide")

    monkeypatch.setattr("skillevaluator.tier2.commands.run_similarity_check", must_not_run)

    result = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(skill),
            "--save-catalog",
            str(catalog),
            "-r",
            "json",
            "-o",
            str(reports_alias),
        ],
    )

    assert result.exit_code == 2, result.output
    assert catalog.read_text(encoding="utf-8") == "catalog must survive"


def test_similarity_cli_rejects_query_catalog_collision_with_generated_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    catalog = reports / "skillevaluator-similarity.json"
    catalog.write_text("existing catalog must survive", encoding="utf-8")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("Similarity must not run when its input catalog would be overwritten")

    monkeypatch.setattr("skillevaluator.tier2.commands.run_similarity_check", must_not_run)

    result = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(skill),
            "--catalog",
            str(catalog),
            "-r",
            "json",
            "-o",
            str(reports),
        ],
    )

    assert result.exit_code == 2, result.output
    assert "catalog" in result.output.lower()
    assert "report" in result.output.lower()
    assert catalog.read_text(encoding="utf-8") == "existing catalog must survive"


def test_similarity_cli_allows_catalog_name_when_only_cli_report_is_selected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    reports = tmp_path / "reports"
    catalog = reports / "skillevaluator-similarity.json"
    called = False

    def run_without_file_report(*_args, **_kwargs):
        nonlocal called
        called = True
        return [ValidationResult(validator_name="Similarity Check")]

    monkeypatch.setattr("skillevaluator.tier2.commands.run_similarity_check", run_without_file_report)

    result = CliRunner().invoke(
        cli,
        [
            "similarity-check",
            str(skill),
            "--save-catalog",
            str(catalog),
            "-r",
            "cli",
            "-o",
            str(reports),
        ],
    )

    assert result.exit_code == 0, result.output
    assert called


def test_similarity_cli_reports_unsafe_output_path_without_traceback(tmp_path: Path, monkeypatch) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    reports = tmp_path / "reports"
    reports.symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(
        "skillevaluator.tier2.commands.run_similarity_check",
        lambda *_args, **_kwargs: [ValidationResult(validator_name="Similarity Check")],
    )

    result = CliRunner().invoke(
        cli,
        ["similarity-check", str(skill), "-r", "json", "-o", str(reports)],
    )

    assert result.exit_code == 1
    assert "Unsafe report output path" in result.output
    assert "Traceback" not in result.output
    assert list(external.iterdir()) == []


def _tier2_result_with_host_paths(target: Path) -> list[ValidationResult]:
    absolute_manifest = str(target / "SKILL.md")
    result = ValidationResult(
        validator_name="Tier 2 path safety",
        validator_description=f"Validation for {target}",
    )
    result.add_success(
        "collection",
        f"Collected {absolute_manifest}",
        path=absolute_manifest,
        nested={"paths": [absolute_manifest]},
    )
    result.add_finding(
        Finding(
            category="CONTENT_DEDUP",
            severity=Severity.HIGH,
            check_name="path_leak",
            message=f"Unsafe content at {absolute_manifest}",
            file_path=absolute_manifest,
            line_content=f"read from {absolute_manifest}",
            suggestion=f"Review {target}",
            metadata={"target": str(target), "nested": {"manifest": absolute_manifest}},
        )
    )
    result.add_warning(f"Warning for {absolute_manifest}")
    result.add_message(f"Message for {target}")
    result.metadata["target"] = str(target)
    return [result]


@pytest.mark.parametrize(
    ("command", "runner_path", "basename"),
    [
        pytest.param(
            "similarity-check",
            "skillevaluator.tier2.commands.run_similarity_check",
            "skillevaluator-similarity",
            id="similarity",
        ),
        pytest.param(
            "context-optimization-check",
            "skillevaluator.tier2.commands.run_context_optimization_check",
            "skillevaluator-context",
            id="context",
        ),
        pytest.param(
            "dedup-scan",
            "skillevaluator.tier2.commands.run_dedup_scan",
            "skillevaluator-dedup",
            id="dedup-alias",
        ),
    ],
)
def test_tier2_commands_redact_absolute_target_paths_from_all_reports(
    tmp_path: Path,
    monkeypatch,
    command: str,
    runner_path: str,
    basename: str,
) -> None:
    target = tmp_path / "external-skill"
    target.mkdir()
    reports = tmp_path / "reports"
    monkeypatch.setattr(runner_path, lambda *_args, **_kwargs: _tier2_result_with_host_paths(target))

    result = CliRunner().invoke(
        cli,
        [command, str(target), "-r", "cli,json,html,markdown", "-o", str(reports)],
    )

    assert result.exit_code == 1, result.output
    rendered = [result.output]
    for extension in (".json", ".html", ".md"):
        report_path = reports / f"{basename}{extension}"
        assert report_path.is_file()
        rendered.append(report_path.read_text(encoding="utf-8"))
    for output in rendered:
        assert str(target) not in output
        assert str(target / "SKILL.md") not in output
        assert target.name in output


def test_tier3_validate_help() -> None:
    result = CliRunner().invoke(cli, ["tier3", "validate", "--help"])

    assert result.exit_code == 0
    assert "--harbor-contract" in result.output


def test_removed_benchmark_authoring_command_is_unavailable() -> None:
    removed_command = "convert" + "-benchmark"

    for args in ([removed_command, "--help"], ["tier3", removed_command, "--help"]):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 2
        assert f"No such command '{removed_command}'" in result.output


def test_live_eval_help_uses_skill_evaluator_runtime_and_grading_names() -> None:
    runner = CliRunner()

    evaluate = runner.invoke(cli, ["evaluate", "--help"])
    assert evaluate.exit_code == 0
    assert "e2b" in evaluate.output
    assert "modal" in evaluate.output
    assert "default_plus_custom" in evaluate.output
    assert "harbor-environment" not in evaluate.output
    assert "k8s-sandbox" not in evaluate.output
    assert "local" not in evaluate.output
    assert "--autopilot" in evaluate.output
    assert "--progress [auto|rich|plain|off]" in evaluate.output

    grader = runner.invoke(cli, ["init-custom-grader", "--help"])
    assert grader.exit_code == 0
    assert "default_plus_custom" in grader.output


def test_live_eval_progress_is_presentation_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.evaluation import EvaluationService
    from skillevaluator.tier3.harbor.progress import PlainProgressReporter

    captured: dict[str, object] = {}

    def _fake_evaluate(self, options, *, progress_reporter=None):
        captured["options"] = options
        captured["reporter"] = progress_reporter
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(EvaluationService, "evaluate", _fake_evaluate, raising=True)
    result = CliRunner().invoke(
        cli,
        ["evaluate", str(Path(__file__).parent / "fixtures" / "skills" / "simple"), "--progress", "plain"],
    )

    assert result.exit_code == 0, result.output
    assert isinstance(captured["reporter"], PlainProgressReporter)
    assert "progress" not in captured["options"].engine_kwargs()


def test_harbor_view_command_uses_the_skillevaluator_wrapper() -> None:
    command = format_harbor_view_command("/tmp/jobs")

    assert command == "skillevaluator tier3 harbor-view /tmp/jobs"


def test_harbor_view_command_shell_quotes_paths() -> None:
    command = format_harbor_view_command("/tmp/jobs with spaces")

    assert command == "skillevaluator tier3 harbor-view '/tmp/jobs with spaces'"


def test_harbor_view_is_available_as_a_top_level_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3 import commands

    captured: list[Path] = []
    monkeypatch.setattr(commands, "harbor_view", lambda jobs_dir: captured.append(jobs_dir) or 0)

    result = CliRunner().invoke(cli, ["harbor-view", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert captured == [tmp_path]


def test_validate_help_is_detailed() -> None:
    result = CliRunner().invoke(cli, ["validate", "-h"])

    assert result.exit_code == 0
    # Grouped Tier 3 options under their own heading.
    assert "Tier 3 · Live Agent Evaluation:" in result.output
    # Sectioned epilog (parity with skill-evaluator validate -h).
    for section in ("Content types", "Report formats", "Tiers:", "LLM analysis", "Examples:"):
        assert section in result.output
    # Example commands are present.
    assert "skillevaluator validate ./my-skill" in result.output


def test_tier1_validate_alias_shares_detailed_help() -> None:
    # The tier1 alias is the same command object, so it carries the same help.
    result = CliRunner().invoke(cli, ["tier1", "validate", "-h"])

    assert result.exit_code == 0
    assert "Tier 3 · Live Agent Evaluation:" in result.output
    assert "Examples:" in result.output


def test_validate_code_integrity_reports_only_static_test_evidence(tmp_path: Path, monkeypatch) -> None:
    skill = tmp_path / "untrusted-skill"
    tests_dir = skill / "tests"
    tests_dir.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: untrusted-skill\n"
        "description: Exercise the static test-discovery boundary.\n"
        "---\n"
        "\n"
        "# Untrusted skill\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SEC001_TEST_SECRET", "must-not-be-read")
    markers = {name: tmp_path / f"{name}-executed" for name in ("pytest-module", "conftest", "plugin", "test-module")}
    (skill / "pytest.py").write_text(
        f"from pathlib import Path\nPath({str(markers['pytest-module'])!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (skill / "conftest.py").write_text(
        "import os\n"
        "import socket\n"
        "from pathlib import Path\n"
        "pytest_plugins = ('target_plugin',)\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 9), timeout=0.01)\n"
        "except OSError:\n"
        "    pass\n"
        f"Path({str(markers['conftest'])!r}).write_text(os.environ['SEC001_TEST_SECRET'], encoding='utf-8')\n",
        encoding="utf-8",
    )
    (skill / "target_plugin.py").write_text(
        f"from pathlib import Path\nPath({str(markers['plugin'])!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tests_dir / "test_payload.py").write_text(
        f"from pathlib import Path\nPath({str(markers['test-module'])!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tests_dir / "malformed_test.py").write_bytes(b"\xff\xfe\x00")

    outside_test = tmp_path / "outside_test.py"
    outside_test.write_text("raise RuntimeError('must not load')\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        (tests_dir / "test_symlink.py").symlink_to(outside_test)

    # Keep this test focused on the real HygieneValidator boundary; scanner
    # integrations have their own execution and status-contract coverage.
    monkeypatch.setattr(CodeRiskValidator, "validate", lambda _self, _path: ValidationResult())
    monkeypatch.setattr(SecretsValidator, "validate", lambda _self, _path: ValidationResult())

    reports = tmp_path / "reports"
    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(skill),
            "--verbose",
            "--checks",
            "code-integrity",
            "--no-dedup",
            "-r",
            "cli,json,markdown,html",
            "-o",
            str(reports),
        ],
    )

    assert result.exit_code == 0, result.output
    assert not [marker for marker in markers.values() if marker.exists()]
    assert "Tests passed" not in result.output
    assert "test coverage" not in result.output.lower()
    normalized_output = " ".join(result.output.lower().split())
    assert "target tests were not" in normalized_output
    assert "executed and coverage was not measured" in normalized_output

    report_paths = list(reports.glob("skillevaluator-output-*"))
    assert len(report_paths) == 3
    benchmark_path = reports / "BENCHMARK.md"
    assert benchmark_path.is_file()
    benchmark = benchmark_path.read_text(encoding="utf-8")
    assert "Evaluation of the `untrusted-skill` skill" in benchmark
    assert "- Skill: `untrusted-skill`" in benchmark
    data = json.loads(next(reports.glob("skillevaluator-output-*.json")).read_text(encoding="utf-8"))
    hygiene = next(item for item in data["results"] if item["validator"] == "Code Integrity & Hygiene")
    detail = next(item for item in hygiene["success_details"] if item["check"] == "test_discovery")
    assert detail["metadata"] == {
        "test_count": 2,
        "execution_performed": False,
        "coverage_measured": False,
        "patterns": ["test_*.py", "*_test.py"],
    }

    for report in [*report_paths, benchmark_path]:
        content = report.read_text(encoding="utf-8")
        assert "target tests were not executed" in content.lower()
        assert "coverage was not measured" in content.lower()
        assert "Tests passed" not in content
        assert "coverage_percent" not in content
        assert "test_coverage" not in content


def _medium_pii_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "pii-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: pii-skill\n"
        "description: Exercise PII verification status handling.\n"
        "---\n\n"
        "# PII example\n\n"
        "Contact Alice at 415-555-0123.\n"
        "Contact Bob at 212-555-0199.\n",
        encoding="utf-8",
    )
    return skill


def test_pii_scan_requested_llm_verification_without_verdicts_is_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "skillevaluator.inference.FindingVerifier.verify",
        lambda _self, _findings, _skill_path: {},
    )
    reports = tmp_path / "reports"
    skill = _medium_pii_skill(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "pii-scan",
            str(skill),
            "--llm-verify",
            "-r",
            "cli,json",
            "-o",
            str(reports),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "INCOMPLETE" in result.output
    payload = json.loads((reports / "skillevaluator-pii.json").read_text(encoding="utf-8"))
    assert payload["overall_status"] == "incomplete"
    assert payload["incomplete_scans"] == ["llm-verification"]
    pii_result = payload["results"][0]
    assert pii_result["status"] == "incomplete"
    assert pii_result["incomplete_scans"] == ["llm-verification"]
    assert pii_result["findings"]


def test_pii_scan_without_llm_verification_preserves_static_behavior(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("LLM verification must not run without --llm-verify")

    monkeypatch.setattr("skillevaluator.inference.FindingVerifier.verify", fail_if_called)
    reports = tmp_path / "reports"
    skill = _medium_pii_skill(tmp_path)

    result = CliRunner().invoke(
        cli,
        ["pii-scan", str(skill), "-r", "json", "-o", str(reports)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((reports / "skillevaluator-pii.json").read_text(encoding="utf-8"))
    assert payload["overall_status"] == "passed"
    assert payload["incomplete_scans"] == []
    assert payload["results"][0]["findings"]


def test_pii_scan_partial_llm_verification_is_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "skillevaluator.inference.FindingVerifier.verify",
        lambda _self, _findings, _skill_path: {
            0: {
                "verdict": "true_positive",
                "confidence": "high",
                "reasoning": "The first phone number is personal data.",
            }
        },
    )
    reports = tmp_path / "reports"
    skill = _medium_pii_skill(tmp_path)

    result = CliRunner().invoke(
        cli,
        ["pii-scan", str(skill), "--llm-verify", "-r", "json", "-o", str(reports)],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads((reports / "skillevaluator-pii.json").read_text(encoding="utf-8"))
    assert payload["overall_status"] == "incomplete"
    assert payload["incomplete_scans"] == ["llm-verification"]
    findings = payload["results"][0]["findings"]
    assert len(findings) == 2
    assert findings[0]["metadata"]["llm_verdict"] == "true_positive"
    assert "llm_verdict" not in findings[1]["metadata"]


@pytest.mark.parametrize(
    "malformed_verdict",
    [
        pytest.param(
            {
                "verdict": "invented_verdict",
                "confidence": "high",
                "reasoning": "This is not a supported verdict.",
            },
            id="unknown-verdict",
        ),
        pytest.param(
            {
                "verdict": [],
                "confidence": "high",
                "reasoning": "The verdict has the wrong type.",
            },
            id="non-string-verdict",
        ),
    ],
)
def test_pii_scan_malformed_llm_verdict_is_incomplete(
    tmp_path: Path,
    monkeypatch,
    malformed_verdict: dict,
) -> None:
    monkeypatch.setattr(
        "skillevaluator.inference.FindingVerifier.verify",
        lambda _self, _findings, _skill_path: {
            0: {
                "verdict": "true_positive",
                "confidence": "high",
                "reasoning": "The first phone number is personal data.",
            },
            1: malformed_verdict,
        },
    )
    reports = tmp_path / "reports"
    skill = _medium_pii_skill(tmp_path)

    result = CliRunner().invoke(
        cli,
        ["pii-scan", str(skill), "--llm-verify", "-r", "json", "-o", str(reports)],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads((reports / "skillevaluator-pii.json").read_text(encoding="utf-8"))
    assert payload["overall_status"] == "incomplete"
    assert payload["incomplete_scans"] == ["llm-verification"]
    findings = payload["results"][0]["findings"]
    assert findings[0]["metadata"]["llm_verdict"] == "true_positive"
    assert "llm_verdict" not in findings[1]["metadata"]
