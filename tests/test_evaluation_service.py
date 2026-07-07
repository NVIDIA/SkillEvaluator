# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process Tier 3 EvaluationService + CLI/API parity (Phase 4)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation import EvaluationOptions, EvaluationService

FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "simple"


def test_options_fields_match_engine_signature() -> None:
    """EvaluationOptions must not drift from the engine's evaluate() signature."""
    harbor = pytest.importorskip("harbor")
    assert harbor is not None
    from skillevaluator.tier3.commands import evaluate as engine_evaluate

    sig_params = set(inspect.signature(engine_evaluate).parameters) - {"skill_path"}
    option_fields = set(EvaluationOptions.__dataclass_fields__) - {"skill_path"}
    assert option_fields == sig_params


def test_engine_kwargs_excludes_skill_path() -> None:
    opts = EvaluationOptions(skill_path=Path("/tmp/x"), agents="codex", env_mode="docker")
    kwargs = opts.engine_kwargs()
    assert "skill_path" not in kwargs
    assert kwargs["agents"] == "codex"
    assert kwargs["env_mode"] == "docker"


def test_cli_evaluate_uses_shared_service(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_evaluate(self, options: EvaluationOptions) -> dict:
        captured["options"] = options
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(EvaluationService, "evaluate", _fake_evaluate, raising=True)
    result = CliRunner().invoke(
        cli, ["evaluate", str(FIXTURE), "-a", "codex", "--env-mode", "docker", "--skip-baseline"]
    )
    assert result.exit_code == 0, result.output
    opts = captured["options"]
    assert isinstance(opts, EvaluationOptions)
    assert opts.agents == "codex"
    assert opts.env_mode == "docker"
    assert opts.skip_baseline is True


def test_cli_evaluate_accepts_harbor_native_cloud_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_evaluate(self, options: EvaluationOptions) -> dict:
        captured["options"] = options
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(EvaluationService, "evaluate", _fake_evaluate, raising=True)
    result = CliRunner().invoke(
        cli,
        ["evaluate", str(FIXTURE), "-a", "codex", "--env-mode", "e2b"],
    )

    assert result.exit_code == 0, result.output
    opts = captured["options"]
    assert isinstance(opts, EvaluationOptions)
    assert opts.env_mode == "e2b"


@pytest.mark.parametrize(
    ("engine_result", "expected"),
    [
        ({}, "empty result"),
        ({"execution_status": "failed", "execution_errors": ["job incomplete"]}, "job incomplete"),
        ({"execution_status": "unknown"}, "status is unknown"),
        ({"execution_status": "succeeded", "error": ["late failure"]}, "late failure"),
        (None, "non-mapping result"),
    ],
)
def test_cli_evaluate_fails_closed_on_unusable_engine_result(
    monkeypatch: pytest.MonkeyPatch,
    engine_result: object,
    expected: str,
) -> None:
    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options: engine_result, raising=True)

    result = CliRunner().invoke(cli, ["evaluate", str(FIXTURE), "--skip-baseline"])

    assert result.exit_code != 0
    assert expected in result.output


def test_combined_evaluate_keeps_exit_advisory_but_result_false_for_empty_engine_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator import cli as cli_module

    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options: {}, raising=True)

    result = cli_module._run_agent_eval_or_skip(
        FIXTURE,
        agents="codex",
        env_mode="docker",
        skip_baseline=True,
        n_concurrent=1,
        max_agents=1,
    )

    assert result.passed is False
    assert result.metadata["agent_eval"]["execution_status"] == "skipped"
    assert "empty result" in result.metadata["agent_eval"]["execution_errors"][0]


def test_report_discovery_handles_timestamped_results(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    skill = tmp_path / "myskill"
    skill.mkdir()
    run_dir = results_root / "myskill" / "2026-06-18_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    (results_root / "myskill" / "latest").symlink_to(run_dir)

    service = EvaluationService()
    latest = service.discover_latest_results(skill, results_dir=results_root)
    assert latest is not None and latest.name == "2026-06-18_120000"
    report = service.discover_latest_report(skill, results_dir=results_root)
    assert report is not None and report.name == "report.html"


def test_report_discovery_returns_none_when_absent(tmp_path: Path) -> None:
    skill = tmp_path / "empty"
    skill.mkdir()
    service = EvaluationService()
    assert service.discover_latest_results(skill, results_dir=tmp_path / "nope") is None
    assert service.discover_latest_report(skill, results_dir=tmp_path / "nope") is None
