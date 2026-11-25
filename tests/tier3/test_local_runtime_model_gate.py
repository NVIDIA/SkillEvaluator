# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-mode preflight and doctor reject an agent CLI that is too old for the selected model."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import local_runtime
from skillevaluator.tier3.harbor.local_runtime import ensure_local_runtimes, required_runtime_version

if TYPE_CHECKING:
    from pathlib import Path


def _stub_cli(monkeypatch: pytest.MonkeyPatch, command: str, stdout: str, *, returncode: int = 0) -> None:
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: command)
    monkeypatch.setattr(
        local_runtime,
        "subprocess",
        SimpleNamespace(
            run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")
        ),
        raising=False,
    )


@pytest.mark.parametrize(
    ("agent", "output", "model", "too_old"),
    [
        ("claude-code", "2.1.279 (Claude Code)", "claude-opus-5-5", "2.1.279"),
        ("claude-code", "2.1.280 (Claude Code)", "claude-opus-5-5", None),
        ("claude-code", "3.0.0 (Claude Code)", "claude-opus-5-5", None),
        ("claude-code", "2.1.279 (Claude Code)", "claude-opus-4-8", None),
        ("claude-code", "2.1.279 (Claude Code)", "gpt-6.1-sol", None),
        ("codex", "codex-cli 0.159.0", "gpt-6.1-sol", "0.159.0"),
        ("codex", "codex-cli 0.159.1", "gpt-6.1-sol", None),
        ("codex", "\n\n  codex-cli 0.159.1\n", "gpt-6.1-sol", None),
        ("codex", "codex-cli 0.159.0", "gpt-5.4", None),
        ("opencode", "0.0.1", "gpt-6.1-sol", None),
        ("opencode", "0.0.1", "claude-opus-5-5", None),
    ],
)
def test_installed_cli_is_checked_against_the_selected_model(
    monkeypatch: pytest.MonkeyPatch, agent: str, output: str, model: str, too_old: str | None
) -> None:
    command = f"/fake/{agent}"
    _stub_cli(monkeypatch, command, output)

    errors = ensure_local_runtimes([agent], agent_models={agent: [model]})

    if too_old is None:
        assert errors == []
        return
    minimum = {"claude-code": "2.1.280", "codex": "0.159.1"}[agent]
    assert errors == [
        f"Local mode agent '{agent}' at {command} is version {too_old}, but {model} needs {minimum} or later. "
        f"Update that CLI, use --env-mode docker, or choose an older model with --agent-model {agent}=<model>."
    ]


def test_only_the_selected_agents_models_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_cli(monkeypatch, "/fake/claude", "2.1.279")

    # The model is for another agent, so the old Claude Code CLI is not blocked.
    assert ensure_local_runtimes(["claude-code"], agent_models={"codex": ["gpt-6.1-sol"]}) == []


@pytest.mark.parametrize("agent_models", [None, {}, {"claude-code": []}])
def test_unknown_models_are_not_compared_with_the_cli_version(
    monkeypatch: pytest.MonkeyPatch, agent_models: Any
) -> None:
    _stub_cli(monkeypatch, "/fake/claude", "2.1.279 (Claude Code)")

    assert ensure_local_runtimes(["claude-code"], agent_models=agent_models) == []


def test_unparseable_version_output_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_cli(monkeypatch, "/fake/claude", "claude dev build")

    assert ensure_local_runtimes(["claude-code"], agent_models={"claude-code": ["claude-opus-5-5"]}) == []


def test_codex_banner_before_version_fails_preflight_for_gpt6(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_cli(monkeypatch, "/fake/codex", "\nLaunching Codex\ncodex-cli 0.159.1\n")

    (error,) = ensure_local_runtimes(["codex"], agent_models={"codex": ["gpt-6.1-sol"]})

    assert "did not report a recognizable version" in error
    assert "gpt-6.1-sol needs 0.159.1 or later" in error
    assert ensure_local_runtimes(["codex"], agent_models={"codex": ["gpt-5.4"]}) == []


def test_a_failing_version_command_keeps_its_own_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_cli(monkeypatch, "/fake/claude", "2.1.279", returncode=3)

    (error,) = ensure_local_runtimes(["claude-code"], agent_models={"claude-code": ["claude-opus-5-5"]})

    assert "exit 3" in error
    assert "needs" not in error


def test_each_agent_is_checked_against_its_own_models(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = {"/fake/claude": "2.1.279", "/fake/codex": "0.159.1"}
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda agent, **_: f"/fake/{agent.split('-')[0]}")
    monkeypatch.setattr(
        local_runtime,
        "subprocess",
        SimpleNamespace(
            run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, stdout=outputs[argv[0]], stderr="")
        ),
        raising=False,
    )

    errors = ensure_local_runtimes(
        ["claude-code", "codex"],
        agent_models={"claude-code": ["claude-opus-5-5"], "codex": ["gpt-6.1-sol"]},
    )

    assert len(errors) == 1
    assert "claude-code" in errors[0]


def test_required_runtime_version_only_names_a_minimum_for_the_newest_models() -> None:
    assert required_runtime_version("claude-code", "claude-opus-5-5") == (2, 1, 280)
    assert required_runtime_version("claude-code", "claude-opus-4-8") is None
    assert required_runtime_version("codex", "gpt-6.1-sol") == (0, 159, 1)
    assert required_runtime_version("codex", "gpt-5.4") is None
    assert required_runtime_version("opencode", "gpt-6.1-sol") is None
    assert required_runtime_version("unknown-agent", "claude-opus-5-5") is None


def test_run_preflight_passes_the_selected_models_and_stops_before_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("harbor")
    from tests.test_harbor_task_toml_retry_env import _stub_harbor_runner
    from tests.tier3.test_model_endpoint_failure import _generated_skill

    from skillevaluator.tier3.harbor import runner

    _stub_harbor_runner(monkeypatch, tmp_path, task_source="evals_json")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        runner, "_check_prerequisites", lambda **kwargs: captured.update(kwargs) or ["agent CLI is too old"]
    )

    result = runner.run_harbor_eval(
        _generated_skill(tmp_path),
        ["codex"],
        agent_models={"codex": "gpt-6.1-sol"},
        output_dir=tmp_path / "results",
        env_mode="local",
        agent_runtime_preflight=False,
    )

    assert captured["agent_models"] == {"codex": ["gpt-6.1-sol"]}
    assert result == {"error": ["agent CLI is too old"]}


def test_doctor_reports_an_outdated_cli_as_a_failed_prerequisite(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("harbor")
    from skillevaluator.tier3 import commands as tier3_commands
    from skillevaluator.tier3.harbor import local_sandbox, runner

    _stub_cli(monkeypatch, "/fake/claude", "2.1.279 (Claude Code)")
    monkeypatch.setattr(runner, "_harbor_bin", lambda: "/fake/harbor")
    monkeypatch.setattr(local_sandbox, "detect", lambda _mode: None)
    monkeypatch.setattr(
        tier3_commands,
        "resolve_llm_provider",
        lambda: ProviderConfig(
            provider="openai",
            model="gpt-5.4",
            api_key="test-key-never-sent",
            base_url="https://api.openai.com/v1",
            litellm_model="gpt-5.4",
            region=None,
        ),
    )

    rows = tier3_commands.readiness_rows(
        agents="claude-code",
        env_mode="local",
        agent_model=("claude-code=claude-opus-5-5",),
    )

    failed = [detail for check, status, detail in rows if check == "local prerequisite" and status == "fail"]
    assert len(failed) == 1
    assert "claude-opus-5-5 needs 2.1.280 or later" in failed[0]
