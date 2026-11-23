# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared deterministic checks contribute authored behavior coverage."""

import json

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3.dataset_coverage import summarize_dataset


@pytest.mark.parametrize(
    "check",
    [
        {"type": "script_invoked", "script": "scripts/run.py"},
        {"type": "command_succeeded", "script": "scripts/run.py"},
        {"type": "file_exists", "path": "output.txt"},
        {"type": "output_contains", "text": "done"},
        {"type": "output_matches", "pattern": "done$"},
    ],
)
def test_checks_only_case_has_behavior_coverage(check):
    summary = summarize_dataset([{"checks": [check]}])

    assert summary["cases_with_expected_behaviors"] == 1
    assert not any("behavior_check is N/A" in gap for gap in summary["gaps"])


def test_behavior_coverage_counts_each_case_once():
    check = {"type": "output_contains", "text": "done"}
    summary = summarize_dataset(
        [
            {"expected_behavior": ["Return a result"], "checks": [check, check]},
            {"checks": [check]},
            {"expected_behavior": ["Explain the result"]},
            {},
        ]
    )

    assert summary["cases_with_expected_behaviors"] == 3
    assert any("1 cases have no assertions" in gap for gap in summary["gaps"])


@pytest.mark.parametrize(
    "checks", [[], None, {}, ["bad"], [{"type": "unknown"}], [{"type": "output_matches", "pattern": "["}]]
)
def test_invalid_or_empty_checks_do_not_claim_coverage(checks):
    assert summarize_dataset([{"checks": checks}])["cases_with_expected_behaviors"] == 0


def test_validate_reports_authored_declarative_coverage(tmp_path):
    root = tmp_path / "demo"
    (root / "evals").mkdir(parents=True)
    (root / "SKILL.md").write_text("---\nname: demo\ndescription: Compute values.\n---\n# Demo\n")
    (root / "evals/evals.json").write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "evals": [
                    {
                        "id": "one",
                        "prompt": "Compute.",
                        "expected_output": "done",
                        "checks": [{"type": "output_contains", "text": "done"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["tier3", "validate", str(root), "--json", "--strict"])
    assert result.exit_code == 0, result.output
    coverage = next(row for row in json.loads(result.output) if row["path"] == "evals/evals.json#coverage")
    assert coverage["coverage"]["cases_with_expected_behaviors"] == 1
    assert "behavior_check is N/A" not in coverage["message"]
