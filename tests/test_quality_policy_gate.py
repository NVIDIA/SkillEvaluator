# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quality thresholds remain binding when policies reclassify findings."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting.console_ui import first_fix, summarize_tier1
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.utils.rich_markup import strip_terminal_controls
from skillevaluator.validators.policy import ValidationPolicy, apply_policy
from skillevaluator.validators.quality_score import QualityScoreValidator


@pytest.fixture
def below_threshold_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "minimal-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: minimal-skill\n"
        "description: I can handle various things and stuff for your data documents and projects. "
        "This general tool processes files and helps your workflow.\n"
        "---\n"
        "Short content.\n\n"
        "## Instructions\n\n"
        "The paradigm is synergy and awareness.\n",
        encoding="utf-8",
    )
    return skill


def test_raising_efficiency_severity_keeps_cli_minimum_score_failure(
    below_threshold_skill: Path, tmp_path: Path
) -> None:
    result = QualityScoreValidator(min_score=70).validate(below_threshold_skill)
    assert result.metadata["quality_scores"]["overall_score"] == 67.8
    assert not result.passed
    policy = tmp_path / "stricter-policy.yaml"
    policy.write_text("severity_overrides:\n  QUALITY.quality_efficiency: medium\n", encoding="utf-8")
    args = [
        "validate",
        str(below_threshold_skill),
        "--tiers",
        "1",
        "--checks",
        "quality",
        "--no-llm",
        "-r",
        "cli",
        "-o",
        str(tmp_path / "reports"),
    ]

    runner = CliRunner()
    default = runner.invoke(cli, args)
    stricter = runner.invoke(cli, [*args, "--policy", str(policy)])

    assert default.exit_code == 1, default.output
    assert stricter.exit_code == 1, stricter.output
    output = strip_terminal_controls(stricter.output)
    assert "[PASS] All validations passed" not in output
    assert "67.8/100 is below the required 70/100" in output


def test_pipeline_explains_minimum_score_failure(below_threshold_skill: Path) -> None:
    result = QualityScoreValidator(min_score=70).validate(below_threshold_skill)
    passed, rows = summarize_tier1([result])

    assert not passed
    failure_row = next(row for row in rows if row.label == "failed")
    assert result.gate_failures[0] in "".join(text for text, _style in failure_row.segments)
    fix = first_fix([result])
    assert fix is not None
    assert "67.8/100 is below the required 70/100" in fix[0][0]


def test_json_report_includes_minimum_score_failure(below_threshold_skill: Path) -> None:
    result = QualityScoreValidator(min_score=70).validate(below_threshold_skill)
    data = json.loads(JSONReporter().render_all([result]))

    assert data["results"][0]["gate_failures"] == result.gate_failures


def test_json_report_omits_gate_failures_when_minimum_is_met(below_threshold_skill: Path) -> None:
    result = QualityScoreValidator(min_score=67).validate(below_threshold_skill)
    data = json.loads(JSONReporter().render_all([result]))

    assert "gate_failures" not in data["results"][0]


def test_quality_minimum_survives_recalculation(below_threshold_skill: Path) -> None:
    result = QualityScoreValidator(min_score=70).validate(below_threshold_skill)
    errors = list(result.errors)

    result.recalculate_from_findings()
    result.recalculate_from_findings()

    assert not result.passed
    assert result.errors == errors
    assert result.summary.errors == 1
    assert result.to_dict()["gate_failures"] == result.gate_failures


def test_lowering_all_quality_severities_does_not_disable_minimum(below_threshold_skill: Path) -> None:
    result = QualityScoreValidator(min_score=70).validate(below_threshold_skill)

    apply_policy([result], ValidationPolicy(severity_overrides={"QUALITY.*": Severity.INFO}))

    assert not result.passed


@pytest.mark.parametrize("merge_method", ["merge", "merge_with_prefix"])
def test_quality_minimum_survives_merged_results(below_threshold_skill: Path, merge_method: str) -> None:
    parent = ValidationResult()
    for min_score in (70, 80):
        child = QualityScoreValidator(min_score=min_score).validate(below_threshold_skill)
        if merge_method == "merge_with_prefix":
            parent.merge_with_prefix(child, "minimal-skill")
        else:
            parent.merge(child)

    apply_policy([parent], ValidationPolicy(severity_overrides={"QUALITY.quality_efficiency": Severity.MEDIUM}))

    assert not parent.passed
    assert len(parent.gate_failures) == 2
    assert parent.summary.errors == 2
    assert "required 70/100" in parent.errors[0]
    assert "required 80/100" in parent.errors[1]


def test_folder_average_does_not_hide_a_skill_below_minimum(below_threshold_skill: Path) -> None:
    good_skill = below_threshold_skill.parent / "good-skill"
    good_skill.mkdir()
    (good_skill / "SKILL.md").write_text(
        "---\n"
        "name: good-skill\n"
        "description: Converts CSV documents to JSON. Use when converting tabular files.\n"
        "version: 1.0.0\n"
        "metadata:\n"
        "  author: Test User <test@example.com>\n"
        "  tags: [csv, json]\n"
        "---\n"
        "## Purpose\n\nConvert tabular data.\n\n"
        "## Instructions\n\n1. Read the input file.\n2. Write the JSON output.\n\n"
        "## Examples\n\n```text\ninput.csv -> output.json\n```\n\n"
        "## Prerequisites\n\nNone.\n\n"
        "## Limitations\n\nOnly CSV files are supported.\n\n"
        "## Troubleshooting\n\nReport invalid input errors.\n",
        encoding="utf-8",
    )
    result = QualityScoreValidator(min_score=70).validate(below_threshold_skill.parent)
    assert result.metadata["quality_scores"]["overall_score"] > 70

    apply_policy([result], ValidationPolicy(severity_overrides={"QUALITY.quality_efficiency": Severity.MEDIUM}))

    assert not result.passed
    assert len(result.gate_failures) == 1
    assert result.gate_failures[0].startswith("[minimal-skill]")


def test_minimum_below_score_still_passes_after_policy(below_threshold_skill: Path) -> None:
    result = QualityScoreValidator(min_score=67).validate(below_threshold_skill)

    apply_policy([result], ValidationPolicy(severity_overrides={"QUALITY.quality_efficiency": Severity.MEDIUM}))

    assert result.passed
    assert "gate_failures" not in result.to_dict()


def test_quality_severity_waiver_still_passes_when_score_is_above_minimum(below_threshold_skill: Path) -> None:
    manifest = below_threshold_skill / "SKILL.md"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("name: minimal-skill", "name: Bad_Name"))
    result = QualityScoreValidator(min_score=0).validate(below_threshold_skill)
    assert not result.passed
    assert result.metadata["quality_scores"]["overall_score"] > 0

    apply_policy([result], ValidationPolicy(severity_overrides={"QUALITY.*": Severity.INFO}))

    assert result.passed
    assert result.errors == []


def test_policy_can_still_clear_a_severity_based_failure() -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="author_missing",
            message="Missing author",
            file_path="SKILL.md",
        )
    )

    apply_policy([result], ValidationPolicy(severity_overrides={"SCHEMA.author_missing": Severity.INFO}))

    assert result.passed
    assert result.errors == []
