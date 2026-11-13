# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes compatibility at the public routing and Harbor artifact boundaries."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.agent.context import AgentContext
from harbor.models.task.task import Task
from harbor.models.trial.config import (
    AgentConfig as HarborAgentConfig,
)
from harbor.models.trial.config import (
    TaskConfig as HarborTaskConfig,
)
from harbor.models.trial.config import (
    TrialConfig,
    VerifierConfig,
)
from harbor.tasks.client import TaskDownloadResult
from harbor.trial.single_step import SingleStepTrial
from harbor.trial.trial import Trial

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.commands import validate_agents
from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts
from skillevaluator.tier3.eval_core.checks import check_activation
from skillevaluator.tier3.harbor import DOCKER_AGENT_IMPORT_PATHS, HARBOR_AGENTS_EXPERIMENTAL
from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
from skillevaluator.tier3.harbor.collector import AGENT_LOG_ARTIFACTS, _agent_log_runtime_failure_reason
from skillevaluator.tier3.harbor.hermes_agent import HERMES_COMMIT, HERMES_VERSION, SkillEvaluatorHermes
from skillevaluator.tier3.harbor.runner import (
    _agent_import_path,
    _harbor_agent_kwargs,
    _model_for_agent,
    _resolve_agent_runtime_plan,
    _validate_agent_provider_credentials,
    build_harbor_run_command,
)


def _provider(name="nv_build"):
    return ProviderConfig(
        provider=name,
        model="nvidia/nemotron-3.5-super-vl-120b-a12b",
        api_key="selected-provider-key",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="openai/nvidia/nemotron-3.5-super-vl-120b-a12b",
    )


def _agent(tmp_path):
    return SkillEvaluatorHermes(logs_dir=tmp_path, model_name=_provider().model)


def _session(*, tool="skill_view", arguments=None, observation=None):
    return json.dumps(
        {
            "id": "native-session-id",
            "messages": [
                {"role": "user", "content": "Look up cobalt"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": tool,
                                "arguments": json.dumps(arguments or {"name": "lookup-code"}),
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": json.dumps(observation or {"success": True, "content": "# Instructions"}),
                },
                {"role": "assistant", "content": "ZX-731"},
            ],
        }
    )


def test_public_registration_and_pinned_docker_command(tmp_path):
    assert validate_agents(["hermes"]) == []
    assert "hermes" in HARBOR_AGENTS_EXPERIMENTAL
    assert _harbor_agent_kwargs("hermes", _provider().model, "docker") == {"version": HERMES_VERSION}
    assert _agent_import_path(_provider(), "hermes", "docker") == DOCKER_AGENT_IMPORT_PATHS["hermes"]
    command = build_harbor_run_command(
        dataset_path=tmp_path, agent="hermes", job_name="test", env_mode="docker", model=_provider().model
    )
    assert command[command.index("--agent") + 1] == DOCKER_AGENT_IMPORT_PATHS["hermes"]
    assert f'version="{HERMES_VERSION}"' in command


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai-compatible", "bedrock"])
def test_unverified_providers_rejected(provider):
    errors = _validate_agent_provider_credentials(_provider(provider), ["hermes"], {"NVIDIA_API_KEY": "key"})
    assert "only NVIDIA Build" in errors[0]


@pytest.mark.parametrize("mode", ["local", "kata", "gke", "daytona", "mosaic"])
def test_unverified_environments_rejected(mode):
    errors = _validate_agent_provider_credentials(_provider(), ["hermes"], {"NVIDIA_API_KEY": "key"}, env_mode=mode)
    assert "--env-mode docker" in errors[0]


@pytest.mark.parametrize("model", ["nemotron", "openrouter/nvidia/model", "/model", "publisher/"])
def test_invalid_model_rejected(model):
    errors = _validate_agent_provider_credentials(
        _provider(), ["hermes"], {"NVIDIA_API_KEY": "key"}, agent_models={"hermes": model}
    )
    assert "publisher/model" in errors[0]


def test_missing_credential_does_not_fall_back_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unrelated-key")
    assert "NVIDIA_API_KEY" in _validate_agent_provider_credentials(_provider(), ["hermes"], {})[0]


def test_model_precedence_and_credential_isolation(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unrelated-key")
    monkeypatch.setenv("OPENAI_API_KEY", "another-key")
    for cli, config, expected, source in (
        (None, {}, _provider().model, "public provider default"),
        (None, {"hermes": {"model": "publisher/configured"}}, "publisher/configured", "evals/config.yml"),
        ("publisher/override", {"hermes": {"model": "ignored"}}, "publisher/override", "CLI"),
    ):
        assert _model_for_agent("hermes", cli_model=cli, config_agents=config, provider=_provider()) == (
            expected,
            source,
        )
    plans = _resolve_agent_runtime_plan(
        provider=_provider(),
        agents=["hermes"],
        models={"hermes": _provider().model},
        configured_runtime_env={},
        env_mode="docker",
    )
    plan = plans["hermes"]
    assert dict(plan.staged_env) == {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
    assert plan.subprocess_env["NVIDIA_API_KEY"] == "selected-provider-key"
    assert "OPENROUTER_API_KEY" not in plan.subprocess_env
    assert "OPENAI_API_KEY" not in plan.subprocess_env


def test_install_pins_script_checkout_and_disables_bundled_skills(tmp_path):
    agent = _agent(tmp_path)
    agent.ensure_system_dependencies = AsyncMock()
    agent.exec_as_agent = AsyncMock()
    asyncio.run(agent.install(SimpleNamespace()))
    command = agent.exec_as_agent.call_args.kwargs["command"]
    assert f"/{HERMES_COMMIT}/scripts/install.sh" in command
    assert f"--branch {HERMES_VERSION} --commit {HERMES_COMMIT}" in command
    assert "--no-skills" in command
    assert "main/" not in command
    assert "hermes --version" in command
    assert "hermes --version" in agent.get_version_command()
    assert agent.version() == HERMES_VERSION


@pytest.mark.parametrize(
    "failure",
    [
        None,
        NonZeroAgentExitCodeError("provider failed"),
        TimeoutError("timed out"),
        asyncio.CancelledError("cancelled"),
    ],
)
def test_native_invocation_exports_session_even_on_failure(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("NVIDIA_API_KEY", "selected-provider-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unrelated-key")
    agent = _agent(tmp_path)
    calls = []

    async def execute(_environment, **kwargs):
        calls.append(kwargs)
        if "hermes --yolo chat" in kwargs["command"] and failure:
            raise failure
        return SimpleNamespace(stdout=_session(), return_code=0)

    agent.exec_as_agent = execute
    if failure:
        with pytest.raises(type(failure), match=str(failure)):
            asyncio.run(agent.run("untrusted ' prompt\nEOF", SimpleNamespace(), AgentContext()))
    else:
        asyncio.run(agent.run("untrusted ' prompt\nEOF", SimpleNamespace(), AgentContext()))
    launch = next(call for call in calls if "hermes --yolo chat" in call["command"])
    assert "--provider nvidia --model nvidia/nemotron-3.5-super-vl-120b-a12b" in launch["command"]
    assert set(launch["env"]) == {
        "HERMES_HOME",
        "TERMINAL_ENV",
        "HERMES_BUNDLED_SKILLS",
        "NVIDIA_API_KEY",
        "HARBOR_INSTRUCTION",
    }
    assert launch["env"]["HERMES_BUNDLED_SKILLS"] == "/tmp/hermes/empty-bundled-skills"
    assert any("mkdir -p /tmp/hermes/empty-bundled-skills" in call["command"] for call in calls)
    assert "selected-provider-key" not in launch["command"]
    assert "untrusted" not in launch["command"]
    assert "sessions export" in calls[-1]["command"]
    assert "NVIDIA_API_KEY" not in calls[-1]["env"]
    assert agent._native_session_id == "native-session-id"
    assert any("cp -r /workspace/skills/" in call["command"] for call in calls)


@pytest.mark.parametrize(
    ("tool", "observation", "arguments", "activated"),
    [
        ("skill_view", {"success": True, "content": "# Instructions"}, {"name": "lookup-code"}, True),
        ("skill_view", {"success": False, "error": "not found"}, {"name": "lookup-code"}, False),
        (
            "skill_view",
            {"success": True, "content": "reference"},
            {"name": "lookup-code", "file_path": "references/help.md"},
            False,
        ),
        ("skill_view", {"success": True, "content": "other"}, {"name": "another-skill"}, False),
        (
            "read_file",
            {"error": "File not found: /tmp/hermes/skills/lookup-code/SKILL.md"},
            {"path": "/tmp/hermes/skills/lookup-code/SKILL.md"},
            False,
        ),
        (
            "read_file",
            {"success": False},
            {"path": "/tmp/hermes/skills/lookup-code/SKILL.md"},
            False,
        ),
        (
            "read_file",
            "The skill manifest contents",
            {"path": "/tmp/hermes/skills/lookup-code/SKILL.md"},
            True,
        ),
    ],
)
def test_atif_skill_activation_requires_successful_target_manifest(tmp_path, tool, observation, arguments, activated):
    trajectory = _agent(tmp_path)._convert_hermes_session_to_atif(
        _session(tool=tool, observation=observation, arguments=arguments), "session"
    )
    calls = extract_tool_calls_as_dicts(trajectory.to_json_dict())
    assert check_activation(calls, "lookup-code")["passed"] is activated
    assert calls[0]["observation"]


def test_terminal_conversion_and_token_context(tmp_path):
    agent = _agent(tmp_path)
    text = _session(tool="terminal", arguments={"command": "python lookup.py cobalt"})
    (tmp_path / "hermes-session.jsonl").write_text(text)
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert (context.n_input_tokens, context.n_output_tokens) == (100, 20)
    assert context.cost_usd is None
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    calls = extract_tool_calls_as_dicts(trajectory)
    assert calls[0]["action"] == "bash"
    assert calls[0]["action_input"]["command"] == "python lookup.py cobalt"


@pytest.mark.parametrize("result", [{"exit_code": 1}, {"error": "timed out"}, {"exit_code": 0}])
def test_terminal_failed_reads_cannot_prove_activation(tmp_path, result):
    trajectory = _agent(tmp_path)._convert_hermes_session_to_atif(
        _session(
            tool="terminal",
            arguments={"command": "cat /tmp/hermes/skills/lookup-code/SKILL.md"},
            observation={"output": "skill content or command error", **result},
        ),
        "session",
    )
    calls = extract_tool_calls_as_dicts(trajectory.to_json_dict())
    succeeded = result == {"exit_code": 0}
    assert calls[0]["is_error"] is not succeeded
    assert check_activation(calls, "lookup-code")["passed"] is succeeded


@pytest.mark.parametrize(
    ("text", "error_reason"),
    [
        (None, "missing_or_invalid_session"),
        ("", "missing_or_invalid_session"),
        ("malformed", "missing_or_invalid_session"),
        (
            '{"id":"session","messages":[{"role":"user","content":"Look up cobalt"}]}',
            "no_agent_response",
        ),
    ],
)
def test_missing_or_malformed_session_is_execution_failure(tmp_path, text, error_reason):
    if text is not None:
        (tmp_path / "hermes-session.jsonl").write_text(text)
    context = AgentContext()
    agent = _agent(tmp_path)
    agent._reported_session_metrics = ("stale-session", 100, 20)
    with pytest.raises(RuntimeError, match="Hermes"):
        agent.populate_context_post_run(context)
    assert context.metadata == {"hermes_session_error": error_reason}
    assert context.n_input_tokens is None
    assert context.n_output_tokens is None
    assert agent._reported_session_metrics is None


def test_harbor_trial_run_recovers_missing_hermes_session_as_execution_failure(tmp_path, monkeypatch):
    skill = Path(__file__).parent / "fixtures" / "skills" / "lookup-code"
    task_dir = generate_harbor_tasks(skill, tmp_path / "tasks", with_skill=True)[0]
    task = Task(task_dir, disable_verification=True)
    config = TrialConfig(
        task=HarborTaskConfig(path=task_dir),
        trial_name="hermes-missing-session",
        trials_dir=tmp_path / "trials",
        agent=HarborAgentConfig(
            import_path=DOCKER_AGENT_IMPORT_PATHS["hermes"],
            model_name=_provider().model,
            kwargs={"version": HERMES_VERSION},
        ),
        verifier=VerifierConfig(disable=True),
    )
    download = TaskDownloadResult(path=task_dir, download_time_sec=0, cached=True)

    def initialize_agent_environment(trial):
        trial.agent_environment = SimpleNamespace(
            capabilities=EnvironmentCapabilities(mounted=True, docker_compose=True),
        )

    async def no_prepare(_trial):
        pass

    async def no_agent_phase(_trial, *, target, **_kwargs):
        target.agent_result = AgentContext()

    async def no_download_agent_logs(_trial):
        pass

    async def no_collect_artifacts(_trial, **_kwargs):
        pass

    async def no_stop_agent_environment(_trial):
        pass

    monkeypatch.setattr(Trial, "_init_agent_environment", initialize_agent_environment)
    monkeypatch.setattr(Trial, "_init_artifact_handler", lambda _trial: None)
    monkeypatch.setattr(Trial, "_prepare", no_prepare)
    monkeypatch.setattr(Trial, "_run_agent_phase", no_agent_phase)
    monkeypatch.setattr(Trial, "_download_agent_logs", no_download_agent_logs)
    monkeypatch.setattr(SingleStepTrial, "_collect_artifacts", no_collect_artifacts)
    monkeypatch.setattr(Trial, "_stop_agent_environment", no_stop_agent_environment)

    trial = SingleStepTrial(config, _task=task, _task_download_result=download)
    result = asyncio.run(trial.run())

    assert result.exception_info is not None
    assert result.exception_info.exception_type == "RuntimeError"
    assert "Hermes produced no valid session trajectory" in result.exception_info.exception_message
    assert result.agent_result is not None
    assert result.agent_result.metadata == {"hermes_session_error": "missing_or_invalid_session"}


def test_resume_requires_exported_session_id(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "selected-provider-key")
    agent = _agent(tmp_path)
    agent._resume = True
    agent.exec_as_agent = AsyncMock()
    with pytest.raises(RuntimeError, match="previous run did not export"):
        asyncio.run(agent.run("resume", SimpleNamespace(), AgentContext()))
    assert not any("hermes --yolo chat" in call.kwargs["command"] for call in agent.exec_as_agent.call_args_list)


def test_diagnostic_retention_and_provider_failure_detection(tmp_path):
    assert "hermes.txt" in AGENT_LOG_ARTIFACTS
    (tmp_path / "hermes.txt").write_text("Error: 401 Unauthorized: invalid API key")
    assert _agent_log_runtime_failure_reason(tmp_path)


def test_unknown_usage_is_not_zero(tmp_path):
    record = json.loads(_session())
    del record["messages"][1]["usage"]
    (tmp_path / "hermes-session.jsonl").write_text(json.dumps(record))
    context = AgentContext()
    _agent(tmp_path).populate_context_post_run(context)
    assert context.n_input_tokens is None
    assert context.n_output_tokens is None
    assert context.cost_usd is None


def test_pinned_session_totals_and_resume_usage_are_not_double_counted(tmp_path):
    agent = _agent(tmp_path)
    record = json.loads(_session())
    del record["messages"][1]["usage"]
    for resume, input_tokens, output_tokens, expected in (
        (False, 500, 40, (500, 40)),
        (True, 700, 55, (200, 15)),
    ):
        record.update(input_tokens=input_tokens, output_tokens=output_tokens)
        (tmp_path / "hermes-session.jsonl").write_text(json.dumps(record))
        agent._last_run_was_resume = resume
        context = AgentContext()
        agent.populate_context_post_run(context)
        assert (context.n_input_tokens, context.n_output_tokens) == expected
    record["messages"][1]["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
    trajectory = agent._convert_hermes_session_to_atif(json.dumps(record), "session")
    assert trajectory.final_metrics.total_prompt_tokens == 100


def test_stale_trajectory_cannot_hide_missing_session(tmp_path):
    (tmp_path / "trajectory.json").write_text('{"steps": [{"source": "agent"}]}')
    with pytest.raises(RuntimeError, match="no valid session"):
        _agent(tmp_path).populate_context_post_run(AgentContext())


def test_matched_arms_preserve_cases_grading_inputs_and_skill_isolation(tmp_path):
    skill = Path(__file__).parent / "fixtures" / "skills" / "lookup-code"
    with_tasks = {task.name: task for task in generate_harbor_tasks(skill, tmp_path / "with", with_skill=True)}
    without_tasks = {task.name: task for task in generate_harbor_tasks(skill, tmp_path / "without", with_skill=False)}
    assert set(with_tasks) == set(without_tasks) == {"lookup-positive", "arithmetic-negative"}
    for case_id, task in with_tasks.items():
        baseline = without_tasks[case_id]
        assert (task / "instruction.md").read_bytes() == (baseline / "instruction.md").read_bytes()
        for filename in ("skill_evaluator/eval.py", "test.sh"):
            assert (task / "tests" / filename).read_bytes() == (baseline / "tests" / filename).read_bytes()
        with_entry = json.loads((task / "tests" / "entry.json").read_text())
        without_entry = json.loads((baseline / "tests" / "entry.json").read_text())
        assert with_entry.pop("has_skill") is True
        assert without_entry.pop("has_skill") is False
        assert with_entry == without_entry
        assert list((task / "environment").rglob("lookup.py"))
        assert not list((baseline / "environment").rglob("lookup.py"))


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("SKILL_EVAL_HERMES_INTEGRATION") != "1",
    reason="Opt in with SKILL_EVAL_HERMES_INTEGRATION=1; installs Hermes and makes paid NVIDIA Build calls",
)
def test_real_hermes_docker_matched_arms(tmp_path):
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY is required")
    skill = Path(__file__).parent / "fixtures" / "skills" / "lookup-code"
    completed = subprocess.run(
        [
            str(Path(sys.executable).with_name("skillevaluator")),
            "tier3",
            "evaluate",
            str(skill),
            "--agents",
            "hermes",
            "--env-mode",
            "docker",
            "--n-attempts",
            "1",
            "--n-concurrent",
            "2",
            "--max-agents",
            "1",
            "--no-stop-on-pass",
            "--harbor-keep-jobs",
            "--results-dir",
            str(tmp_path / "results"),
        ],
        env={
            **os.environ,
            "SKILL_EVAL_LLM_PROVIDER": "nv_build",
            "SKILL_EVAL_LLM_MODEL": _provider().model,
        },
        capture_output=True,
        timeout=3600,
        check=False,
    )
    assert completed.returncode == 0, "Hermes smoke failed; inspect the retained results directory"
    _assert_real_hermes_artifacts(tmp_path / "results")


def _assert_real_hermes_artifacts(results_dir):
    trials = list(results_dir.glob("**/_harbor-jobs/*/*/result.json"))
    assert len(trials) == 4
    for result_path in trials:
        result = json.loads(result_path.read_text())
        assert result.get("exception_info") is None
        assert result["agent_result"]["n_input_tokens"] > 0
        assert result["agent_result"]["n_output_tokens"] > 0
        agent_dir = result_path.parent / "agent"
        assert (agent_dir / "hermes.txt").is_file()
        trajectory = json.loads((agent_dir / "trajectory.json").read_text())
        calls = extract_tool_calls_as_dicts(trajectory)
        expected_activation = result_path.parent.name.startswith(
            "lookup-positive__"
        ) and result_path.parent.parent.name.endswith("-with")
        assert check_activation(calls, "lookup-code")["passed"] is expected_activation
        if expected_activation:
            assert any(
                call["action"] == "bash" and "lookup.py" in call["action_input"].get("command", "") for call in calls
            )
            assert result["verifier_result"]["rewards"]["goal_accuracy"] == 1.0
        if result_path.parent.name.startswith("arithmetic-negative__"):
            assert any(call["action"] == "bash" for call in calls)
            assert result["verifier_result"]["rewards"]["goal_accuracy"] == 1.0
