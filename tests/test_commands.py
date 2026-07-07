# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from pathlib import Path

from click.testing import CliRunner

from skillevaluator.cli import cli

FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "simple"


def test_validate_fixture_no_llm() -> None:
    result = CliRunner().invoke(
        cli, ["validate", str(FIXTURE), "--no-llm", "--no-dedup", "--checks", "schema,quality,lint"]
    )

    assert result.exit_code == 0, result.output
    assert "All validations passed" in result.output


def test_validate_prints_tier1_section_banner() -> None:
    # The Tier 1 section is announced as it runs so it is visibly reported in
    # CI logs (Skill Evaluator parity), not only inside the final combined report.
    result = CliRunner().invoke(
        cli, ["validate", str(FIXTURE), "--no-llm", "--no-dedup", "--checks", "schema,quality,lint"]
    )

    assert result.exit_code == 0, result.output
    assert "Tier 1: Security and Static Validation" in result.output


def test_validate_tier1_banner_is_stable_in_narrow_terminal(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "20")

    result = CliRunner().invoke(
        cli, ["validate", str(FIXTURE), "--no-llm", "--no-dedup", "--checks", "schema,quality,lint"]
    )

    assert result.exit_code == 0, result.output
    assert "Tier 1: Security and Static Validation" in result.output


def test_validate_flushes_tier1_results_before_tier3(monkeypatch) -> None:
    # Tier 1 (and Tier 2) results must reach the terminal BEFORE the
    # long-running Tier 3 agent evaluation, so they remain visible in CI even
    # when Tier 3 is slow or interrupted. Tier 3 is stubbed to keep the test
    # fast and offline.
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    def _stub_agent_eval(*_args, **_kwargs) -> ValidationResult:
        result = ValidationResult(
            validator_name="AGENT_EVAL",
            validator_description="Tier 3: Live Agent Evaluation",
        )
        result.add_warning("stubbed Tier 3")
        # Tier 3 truth remains failed in the report, while validate's exit code
        # is gated only by the Tier 1/Tier 2 snapshot.
        result.passed = False
        return result

    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _stub_agent_eval)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["validate", str(FIXTURE), "--no-llm", "--no-dedup", "--agent-eval", "--checks", "schema"],
        )

    assert result.exit_code == 0, result.output
    out = result.output
    assert "Tier 1: Security and Static Validation" in out
    assert "Tier 3: Live Agent Evaluation" in out
    # The interim Tier 1 summary table is flushed ahead of the Tier 3 section.
    assert out.index("Validation Results") < out.index("Tier 3: Live Agent Evaluation")


def test_tier1_lint_scripts_fixture() -> None:
    result = CliRunner().invoke(cli, ["tier1", "lint-scripts", str(FIXTURE)])

    assert result.exit_code == 0, result.output
    assert "SCRIPT_LINT" in result.output


def test_create_dataset_dry_run_no_llm() -> None:
    result = CliRunner().invoke(cli, ["create-eval-dataset", str(FIXTURE), "--no-llm", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert '"prompt"' in result.output


def test_init_custom_grader_creates_valid_starter(tmp_path: Path) -> None:
    skill_path = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill_path)

    result = CliRunner().invoke(cli, ["init-custom-grader", str(skill_path)])

    assert result.exit_code == 0, result.output
    assert (skill_path / "evals" / "grader.py").exists()
    assert (skill_path / "evals" / "config.yml").exists()
    assert "mode: default_plus_custom" in (skill_path / "evals" / "config.yml").read_text(encoding="utf-8")


def test_init_harbor_task_creates_valid_contract(tmp_path: Path) -> None:
    skill_path = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill_path)

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill_path), "--with-config"])

    assert result.exit_code == 0, result.output
    assert (skill_path / "evals" / "harbor" / "case-001" / "task.toml").exists()

    validate = CliRunner().invoke(cli, ["tier3", "validate", str(skill_path), "--harbor-contract"])

    assert validate.exit_code == 0, validate.output
    assert "all checks passed" in validate.output
