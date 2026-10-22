# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog children must keep separate Tier 3 result roots."""

from __future__ import annotations

import shutil
from concurrent.futures import Future
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.models.result import ValidationResult
from skillevaluator.tier3.results_location import ENV_RESULTS_DIR, resolve_results_root

FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "simple"


class _InlineExecutor:
    """Run the real catalog worker without spawning processes or live evals."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def submit(self, worker, job) -> Future:
        future: Future = Future()
        future.set_result(worker(job))
        return future


def _capture_results_roots(monkeypatch: pytest.MonkeyPatch) -> dict[Path, Path]:
    roots: dict[Path, Path] = {}

    def tier3(skill_path: Path, *, results_dir: Path | None, **_kwargs: object) -> ValidationResult:
        roots[skill_path] = resolve_results_root(skill_path, results_dir)
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "stubbed Tier 3")
        return result

    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", tier3)
    monkeypatch.setattr(cli_module, "ProcessPoolExecutor", _InlineExecutor)
    return roots


def _validate(target: Path, reports: Path, *, workers: int, extra_args: list[str]):
    return CliRunner().invoke(
        cli_module.cli,
        [
            "validate",
            str(target),
            "--tiers",
            "1,3",
            "--no-autopilot",
            "--no-llm",
            "--no-dedup",
            "--checks",
            "schema",
            "--workers",
            str(workers),
            "-r",
            "json",
            "-o",
            str(reports),
            *extra_args,
        ],
    )


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("configured_by", ["cli", "env", "both", "legacy"])
def test_catalog_children_keep_separate_tier3_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int, configured_by: str
) -> None:
    catalog = tmp_path / "catalog"
    skills = [catalog / "team-a" / "simple", catalog / "team-b" / "simple"]
    for skill in skills:
        shutil.copytree(FIXTURE, skill)
    roots = _capture_results_roots(monkeypatch)
    monkeypatch.delenv(ENV_RESULTS_DIR, raising=False)
    external = tmp_path / "results"
    extra_args: list[str] = []
    if configured_by in {"env", "both"}:
        monkeypatch.setenv(ENV_RESULTS_DIR, str(external if configured_by == "env" else tmp_path / "ignored"))
    if configured_by in {"cli", "both"}:
        extra_args = ["--results-dir", str(external)]

    result = _validate(catalog, tmp_path / "reports", workers=workers, extra_args=extra_args)

    assert result.exit_code == 0, result.output
    assert set(roots) == set(skills)
    for skill in skills:
        expected = skill / "evals" / "results" if configured_by == "legacy" else external / skill.relative_to(catalog)
        assert roots[skill] == expected
    assert len(set(roots.values())) == len(skills)


@pytest.mark.parametrize("configured_by", ["cli", "env", "both"])
def test_single_skill_preserves_results_root_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_by: str
) -> None:
    skill = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill)
    roots = _capture_results_roots(monkeypatch)
    monkeypatch.delenv(ENV_RESULTS_DIR, raising=False)
    external = tmp_path / "results"
    extra_args: list[str] = []
    if configured_by in {"env", "both"}:
        monkeypatch.setenv(ENV_RESULTS_DIR, str(external if configured_by == "env" else tmp_path / "ignored"))
    if configured_by in {"cli", "both"}:
        extra_args = ["--results-dir", str(external)]

    result = _validate(skill, tmp_path / "reports", workers=1, extra_args=extra_args)

    assert result.exit_code == 0, result.output
    assert roots == {skill: external / "simple"}
