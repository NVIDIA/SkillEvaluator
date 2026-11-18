# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-skill Tier 2 reports must not turn internal check IDs into skills."""

import json

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import ValidationResult


@pytest.mark.parametrize(
    ("command", "provider_runner", "basename"),
    [
        ("dedup-scan", "run_dedup_scan", "skillevaluator-dedup"),
        ("context-optimization-check", "run_context_optimization_check", "skillevaluator-context"),
    ],
)
def test_one_skill_with_three_successful_checks_has_one_reported_skill(
    tmp_path, monkeypatch, command, provider_runner, basename
):
    skill = tmp_path / "ledger-distinct"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: ledger-distinct\ndescription: Reconcile a ledger.\n---\n")
    result = ValidationResult(validator_name="Content Deduplication")
    for name in ("chunking", "context_dedup", "cross_file_overlap"):
        result.add_success(name, "No redundant content detected")
    # Only the external provider operation is replaced. The CLI, sanitization,
    # skill grouping and report generation all run normally.
    monkeypatch.setattr(f"skillevaluator.tier2.commands.{provider_runner}", lambda *_a, **_kw: [result])
    output = tmp_path / "reports"

    invoked = CliRunner().invoke(cli, [command, str(skill), "-r", "json", "-r", "html", "-o", str(output)])

    assert invoked.exit_code == 0, invoked.output
    report = json.loads((output / f"{basename}.json").read_text())
    assert report["skills"] == [{"name": "ledger-distinct", "passed": True, "issue_count": 0}]
    html = (output / f"{basename}.html").read_text()
    assert "3 of 3 passed" not in html
