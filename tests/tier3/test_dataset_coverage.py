# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3.dataset_coverage import (
    coverage_text,
    dataset_coverage_result,
    summarize_dataset,
)


@pytest.fixture
def skill(tmp_path):
    root = tmp_path / "demo"
    (root / "evals").mkdir(parents=True)
    (root / "SKILL.md").write_text("---\nname: demo\ndescription: Compute values.\n---\n# Demo\n")
    return root


def write_dataset(skill, cases):
    (skill / "evals/evals.json").write_text(json.dumps({"skill_name": "demo", "evals": cases}))


def test_summary_counts_and_routing_resolution():
    result = summarize_dataset(
        [
            {"expected_skill": "demo", "ground_truth": "answer", "expected_behavior": ["act"]},
            {"expected_skill": "demo", "should_trigger": False, "ground_truth": 0, "expected_behavior": [""]},
            {"expected_skill": "", "ground_truth": False},
            {},
        ]
    )
    assert result["case_count"] == 4
    assert result["positive_cases"] == 1
    assert result["negative_cases"] == 2
    assert result["unlabeled_trigger_cases"] == 1
    assert result["cases_with_expected_outcome"] == 3
    assert result["cases_with_expected_behaviors"] == 1
    assert "1 cases have no resolved should_trigger label" in result["gaps"]
    assert any("1 cases have no expected output" in gap for gap in result["gaps"])
    assert any("3 cases have no assertions" in gap for gap in result["gaps"])
    assert result["starter"] is False


def test_starter_and_empty_summary():
    starter = summarize_dataset([{"expected_skill": "demo", "ground_truth": "  ", "expected_behavior": []}])
    assert starter["starter"] is True
    assert "a starting point, not a representative evaluation" in coverage_text(starter)
    assert "no negative case that should not trigger the skill" in starter["gaps"]
    assert any("no assertions" in gap for gap in starter["gaps"])
    empty = summarize_dataset([])
    assert empty["case_count"] == 0
    assert empty["starter"] is False
    assert "no evaluation cases" in empty["gaps"]


def test_more_cases_make_no_sufficiency_claim():
    summary = summarize_dataset(
        [
            {"should_trigger": value, "ground_truth": "result", "expected_behavior": ["act"]}
            for value in [True, False] * 20
        ]
    )
    assert summary["gaps"] == []
    text = coverage_text(summary)
    assert "40 cases" in text
    assert "representative" not in text
    assert "enough" not in text


def test_loader_normalizes_author_fields(skill):
    write_dataset(skill, [{"id": "one", "prompt": "Compute.", "expected_output": "0", "assertions": ["Return 0."]}])
    result = dataset_coverage_result(skill)
    assert result["coverage"]["positive_cases"] == 1
    assert result["coverage"]["cases_with_expected_outcome"] == 1
    assert result["coverage"]["cases_with_expected_behaviors"] == 1


def test_validate_text_json_and_strict_info(skill):
    write_dataset(skill, [{"id": "one", "prompt": "Compute.", "expected_output": ""}])
    runner = CliRunner()
    text = runner.invoke(cli, ["tier3", "validate", str(skill), "--strict"])
    assert text.exit_code == 0, text.output
    assert "Dataset coverage: 1 cases" in text.output
    assert "a starting point, not a representative evaluation" in text.output
    assert "no negative case" in text.output
    assert "accuracy and goal_accuracy are N/A" in text.output
    assert "no assertions" in text.output
    result = runner.invoke(cli, ["tier3", "validate", str(skill), "--json", "--strict"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    coverage = next(row for row in payload if row["path"] == "evals/evals.json#coverage")
    assert coverage["status"] == "ok"
    assert coverage["coverage"]["starter"] is True
    assert all({"path", "status", "message"} <= row.keys() for row in payload)
    assert not any(row["status"] == "warning" for row in payload)


@pytest.mark.parametrize("content", ["{invalid", '{"evals": []}', '["bad"]'])
def test_invalid_source_keeps_failure_exit(skill, content):
    (skill / "evals/evals.json").write_text(content)
    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--json"])
    assert result.exit_code == 1, result.output
    assert any(row["status"] == "error" for row in json.loads(result.output))


def test_missing_and_malformed_yaml_have_no_coverage(skill):
    assert dataset_coverage_result(skill) is None
    (skill / "evals/evals.yaml").write_text("cases: [")
    assert dataset_coverage_result(skill) is None


def test_native_source_skips_even_if_dataset_present(skill):
    write_dataset(skill, [{"id": "one", "prompt": "Compute."}])
    (skill / "evals/config.yml").write_text("harbor:\n  task_source: native_harbor\n")
    assert dataset_coverage_result(skill) is None
    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--json"])
    assert not any("#coverage" in row["path"] for row in json.loads(result.output))


def test_reference_starter_and_calculator():
    references = Path(__file__).parents[2] / "src/skillevaluator/tier3/reference_skills"
    assert dataset_coverage_result(references / "text-analyzer")["coverage"]["starter"] is True
    calculator = dataset_coverage_result(references / "calculator")["coverage"]
    assert calculator["case_count"] == 2
    assert calculator["starter"] is False
