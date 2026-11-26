# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel catalogs retain a failed member when its worker cannot return a result."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import suppress
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module

_FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "simple"


class _MissingWorkdirExecutor(ProcessPoolExecutor):
    def submit(self, worker, job):
        if job["skill_name"] == "beta":
            job = {**job, "cwd": str(Path(job["output_dir"]) / "missing-workdir")}
        return super().submit(worker, job)


class _ExitedWorkerExecutor(ProcessPoolExecutor):
    def submit(self, worker, job):
        if job["skill_name"] == "beta":
            # Exit only a disposable pool child, after alpha's real validation
            # completed. The pool then rejects charlie's submission too.
            future = super().submit(os._exit, 17)
            with suppress(BrokenProcessPool):
                future.result(timeout=30)
            return future
        future = super().submit(worker, job)
        if job["skill_name"] == "alpha":
            future.result(timeout=30)
        return future


@pytest.mark.parametrize(
    ("executor", "failed", "error_type"),
    [
        (_MissingWorkdirExecutor, {"beta"}, "FileNotFoundError"),
        (_ExitedWorkerExecutor, {"beta", "charlie"}, "BrokenProcessPool"),
    ],
    ids=["worker-exception", "worker-exit-and-rejected-submission"],
)
def test_parallel_catalog_records_worker_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, executor, failed: set[str], error_type: str
) -> None:
    catalog = tmp_path / "catalog"
    for name in ("alpha", "beta", "charlie"):
        skill = catalog / name
        shutil.copytree(_FIXTURE, skill)
        manifest = skill / "SKILL.md"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace("name: simple", f"name: {name}"), encoding="utf-8"
        )
    reports = tmp_path / "reports"
    monkeypatch.setattr(cli_module, "ProcessPoolExecutor", executor)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "validate",
            str(catalog),
            "--tiers",
            "1",
            "--no-dedup",
            "--checks",
            "schema",
            "--workers",
            "2",
            "-r",
            "json",
            "-r",
            "github",
            "-o",
            str(reports),
        ],
    )

    assert result.exit_code == 1, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 3
    assert summary["failed"] == len(failed)
    assert summary["passed"] == 3 - len(failed)
    assert summary["overall_passed"] is False
    entries = {entry["name"]: entry for entry in summary["skills"]}
    assert set(entries) == {"alpha", "beta", "charlie"}
    for name, entry in entries.items():
        assert entry["passed"] is (name not in failed)
        if name in failed:
            assert error_type in entry["reason"]
            if error_type == "FileNotFoundError":
                assert "missing-workdir" in entry["reason"]
            else:
                assert "process" in entry["reason"].lower()
            assert "json_report" not in entry
        else:
            report = reports / entry["report_dir"] / entry["json_report"]
            assert json.loads(report.read_text(encoding="utf-8"))["overall_passed"] is True
    assert "skills failed validation" in result.output
    assert "::error" in result.output
    assert error_type in result.output
