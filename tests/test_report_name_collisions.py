# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fast rerun must not replace the previous validation's report artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.models.result import ValidationResult
from skillevaluator.utils import helpers


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 7, 9, 14, 15, 30, 123456, tzinfo=UTC).astimezone(tz)


@pytest.mark.parametrize("suffix", ["", ".json", ".sarif.json"])
def test_report_names_stay_distinct_with_an_identical_clock(monkeypatch, suffix: str) -> None:
    monkeypatch.setattr(helpers, "datetime", _FrozenDatetime)

    names = [helpers.make_timestamped_basename("skillevaluator-output", suffix) for _ in range(32)]

    assert len(set(names)) == len(names)
    assert all(name.startswith("skillevaluator-output-20260709141530") for name in names)
    assert all(name.endswith(suffix) for name in names)
    assert all(Path(name).name == name for name in names)


def test_concurrent_validate_processes_keep_separate_reports(tmp_path: Path) -> None:
    script = """
import sys
from datetime import UTC, datetime
from unittest.mock import patch
from skillevaluator.cli import cli
from skillevaluator.utils import helpers
with patch.object(helpers, 'datetime') as clock:
    clock.now.return_value = datetime(2026, 7, 9, 14, 15, 30, 123456, tzinfo=UTC)
    cli.main(sys.argv[1:])
"""
    skill = tmp_path / "simple"
    shutil.copytree(Path(__file__).parent / "fixtures" / "skills" / "simple", skill)
    reports = tmp_path / "reports"
    command = [
        sys.executable,
        "-c",
        script,
        "validate",
        str(skill),
        "--tiers",
        "1",
        "--checks",
        "schema",
        "--no-llm",
        "--report",
        "json,html",
        "--output-dir",
        str(reports),
    ]

    def run(_index: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30, check=False)

    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = list(executor.map(run, range(2)))

    assert all(item.returncode == 0 for item in completed), [(item.stdout, item.stderr) for item in completed]
    json_reports = list(reports.glob("skillevaluator-output-*.json"))
    assert len(json_reports) == 2
    assert all(report.with_suffix(".html").is_file() for report in json_reports)
    assert all(json.loads(report.read_text(encoding="utf-8"))["overall_passed"] for report in json_reports)


@pytest.mark.parametrize("catalog", [False, True])
def test_validate_same_clock_preserves_both_runs_and_current_catalog_link(
    tmp_path: Path, monkeypatch, catalog: bool
) -> None:
    monkeypatch.setattr(helpers, "datetime", _FrozenDatetime)
    target = tmp_path / "catalog" if catalog else tmp_path / "simple"
    skill = target / "simple" if catalog else target
    shutil.copytree(Path(__file__).parent / "fixtures" / "skills" / "simple", skill)
    first = ValidationResult(validator_name="Example")
    second = ValidationResult(validator_name="Example")
    second.add_error("Second run's failure must not replace the first run")
    outcomes = iter([[first], [second]])
    monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: next(outcomes))
    reports = tmp_path / "reports"
    report_dir = reports / "simple" if catalog else reports
    command = [
        "validate",
        str(target),
        "--tiers",
        "1",
        "--checks",
        "schema",
        "--no-llm",
        "--report",
        "json,html",
        "--output-dir",
        str(reports),
    ]
    runner = CliRunner()

    completed = runner.invoke(cli_module.cli, command)
    assert completed.exit_code == 0, completed.output
    (first_json,) = report_dir.glob("skillevaluator-output-*.json")
    first_html = first_json.with_suffix(".html")
    saved_json, saved_html = first_json.read_bytes(), first_html.read_bytes()

    failed = runner.invoke(cli_module.cli, command)
    assert failed.exit_code != 0, failed.output
    json_reports = sorted(report_dir.glob("skillevaluator-output-*.json"))
    assert len(json_reports) == 2
    assert first_json.read_bytes() == saved_json
    assert first_html.read_bytes() == saved_html
    second_json = next(report for report in json_reports if report != first_json)
    assert second_json.with_suffix(".html").is_file()
    assert json.loads(saved_json)["overall_passed"] is True
    assert json.loads(second_json.read_text(encoding="utf-8"))["overall_passed"] is False
    if catalog:
        summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
        assert summary["skills"][0]["json_report"] == second_json.name
        assert summary["skills"][0]["overall_status"] == "failed"
