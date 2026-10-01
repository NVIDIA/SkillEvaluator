# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.evaluation import EvaluationService
from skillevaluator.models.result import ValidationResult
from skillevaluator.tier3.harbor import runner


def test_plugin_dispatch_configures_clean_effectiveness_and_sum_of_parts_arms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    package_path = tmp_path / "package"
    package_path.mkdir()
    member = tmp_path / "member"
    member.mkdir()
    captured = {}
    prepared = SimpleNamespace(
        skipped=False,
        package_path=package_path,
        include_skills=(member,),
        integration_evidence_error=lambda: None,
        provenance=lambda: {"plugin_name": "plugin", "partial": False},
    )
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: prepared)
    monkeypatch.setattr(
        EvaluationService, "evaluate", lambda _self, options, **_kwargs: captured.setdefault("options", options) or {}
    )
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    expected = ValidationResult(validator_name="AGENT_EVAL")
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.agent_eval_result_from_run", lambda *_a, **_k: expected)

    result = cli_module._run_agent_eval_or_skip(
        plugin,
        agents="codex",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=1,
        max_agents=1,
        kind="plugin",
        lift_mode="both",
    )

    assert result is expected
    options = captured["options"]
    assert options.eval_target_kind == "plugin"
    assert options.skill_workspace_mode == "group"
    assert options.include_skills == (member,)
    assert options.workspace_skills_baseline is False
    assert options.sum_of_parts_arm is True
    assert options.resolved_results_root == plugin / "evals" / "results"


def test_runner_launches_three_arms_and_sum_of_parts_is_report_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched: list[str] = []

    def fake_run(**kwargs):
        name = str(kwargs["job_name"])
        launched.append(name)
        return (False, "advisory failure") if name.endswith("sumofparts") else (True, "")

    monkeypatch.setattr(runner, "_run_harbor", fake_run)
    errors = runner._run_agent_pair(
        skill_name="plugin",
        agent="codex",
        model="model",
        env_mode="docker",
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        sum_of_parts=tmp_path / "parts",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=1,
        n_concurrent=3,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=1,
    )
    assert set(launched) == {"plugin-codex-with", "plugin-codex-without", "plugin-codex-sumofparts"}
    assert errors == []


def test_tier3_evaluate_plugin_command_uses_public_plugin_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    package_path = tmp_path / "package"
    package_path.mkdir()
    member = tmp_path / "member"
    member.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    captured = {}
    prepared = SimpleNamespace(
        skipped=False,
        skip_reason=None,
        package_path=package_path,
        include_skills=(member,),
        unresolved_skill_refs=(),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=(),
        integration_evidence_error=lambda: None,
        provenance=lambda: {"plugin_name": "plugin", "partial": False},
    )
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: prepared)
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: captured.setdefault("options", options) or {"run_dir": str(run_dir)},
    )
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.tier3.result_display.render_evaluation_result", lambda *_a, **_k: None)

    result = CliRunner().invoke(
        cli_module.cli,
        ["tier3", "evaluate-plugin", str(plugin), "--lift-mode", "both", "--progress", "off"],
    )

    assert result.exit_code == 0, result.output
    options = captured["options"]
    assert options.eval_target_kind == "plugin"
    assert options.workspace_skills_baseline is False
    assert options.sum_of_parts_arm is True
    assert options.resolved_results_root == plugin / "evals" / "results"


@pytest.mark.parametrize(
    ("arguments", "evidence_error", "expected"),
    [
        (["--lift-mode", "integration"], "composition evidence is missing", "Integration is inconclusive"),
        (["--lift-mode", "both", "--skip-baseline"], None, "Integration requires a baseline"),
    ],
)
def test_tier3_evaluate_plugin_rejects_invalid_integration_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: list[str],
    evidence_error: str | None,
    expected: str,
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    package_path = tmp_path / "package"
    package_path.mkdir()
    prepared = SimpleNamespace(
        skipped=False,
        skip_reason=None,
        package_path=package_path,
        include_skills=(),
        unresolved_skill_refs=(),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=(),
        integration_evidence_error=lambda: evidence_error,
        provenance=lambda: {"plugin_name": "plugin", "partial": False},
    )
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: prepared)
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda *_a, **_k: pytest.fail("evaluation must not start"),
    )

    result = CliRunner().invoke(
        cli_module.cli,
        ["tier3", "evaluate-plugin", str(plugin), *arguments, "--progress", "off"],
    )

    assert result.exit_code != 0
    assert expected in result.output
