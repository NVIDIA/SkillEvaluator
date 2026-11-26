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


def _raise_worker_error(detail: str) -> None:
    raise RuntimeError(detail)


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


@pytest.mark.parametrize("stage", ["submit", "result"])
@pytest.mark.parametrize("long_detail", [False, True], ids=["credentials", "long-message"])
def test_parallel_catalog_bounds_and_redacts_worker_details(tmp_path, monkeypatch, stage, long_detail):
    credential = "sk-" + "catalog-test-placeholder-123456789"
    detail = (
        "missing-workdir" + " padding" * 4000
        if long_detail
        else f"missing-workdir \x1b[31m{credential}\x1b[0m https://user:password@example.test/path"
    )

    class FailingExecutor(ProcessPoolExecutor):
        def submit(self, _worker, _job):
            if stage == "submit":
                raise RuntimeError(detail)
            return super().submit(_raise_worker_error, detail)

    catalog = tmp_path / "catalog"
    shutil.copytree(_FIXTURE, catalog / "simple")
    reports = tmp_path / "reports"
    monkeypatch.setattr(cli_module, "ProcessPoolExecutor", FailingExecutor)

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
    summary_text = (reports / "catalog-summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["total"] == summary["failed"] == 1
    assert summary["passed"] == 0
    (entry,) = summary["skills"]
    assert entry["passed"] is False
    assert "json_report" not in entry
    prefix = "worker could not start" if stage == "submit" else "worker returned no result"
    assert entry["reason"].startswith(f"{prefix} (RuntimeError: missing-workdir")
    if long_detail:
        assert len(entry["reason"]) <= 1100
        assert entry["reason"].endswith("...<truncated>)")
    else:
        assert entry["reason"] == (
            f"{prefix} (RuntimeError: missing-workdir sk-<redacted> https://<redacted>@example.test/path)"
        )
        for text in [summary_text, result.output]:
            assert credential not in text
            assert "password" not in text
            assert "catalog-test-placeholder" not in text
        assert "\x1b" not in entry["reason"]
    assert "::error" in result.output
    assert "RuntimeError" in result.output
