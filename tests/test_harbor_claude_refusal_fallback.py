# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code's refusal model fallback stays off in every staged Tier 3 task.

When the model provider's safety classifier stops the selected model, Claude
Code 2.1.292 retries the turn on another model from a fixed table (Claude Opus
5.5 -> ``claude-opus-4-8``) under that bare Anthropic id. Behind a gateway that
serves only the selected model the retry failed with HTTP 403 and Harbor
recorded an ``UnknownApiError``; where it succeeded, one scored trial mixed two
models. With the fallback off, the CLI stops with ``model_refusal_no_fallback``
and Harbor 0.24 raises ``AgentSafetyRefusalError``.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
from harbor.agents.installed.base import AgentSafetyRefusalError, ApiOverloadedError, UnknownApiError
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import ExecResult
from harbor.models.task.config import TaskConfig
from harbor.models.trial.paths import TrialPaths
from harbor.utils.env import is_sensitive_env_key
from tests.claude_refusal_fixtures import (
    NO_FALLBACK_TEXT,
    SELECTED_MODEL,
    gateway_fallback_403_stream,
    init_event,
    line,
    no_fallback_stream,
    result_event,
    tool_round,
)

from skillevaluator.provider_config import resolve_llm_provider
from skillevaluator.tier3.harbor import CONTAINER_AGENT_IMPORT_PATHS, LOCAL_AGENT_IMPORT_PATHS
from skillevaluator.tier3.harbor.adapter import (
    _write_task_toml,
    generate_harbor_tasks,
    stage_native_harbor_tasks,
)
from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment
from skillevaluator.tier3.harbor.runner import _model_for_agent, _resolve_agent_runtime_plan, _resolve_runtime_env

_SWITCH = "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK"
_FAKE_CREDENTIAL = "test-key"


def _write_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "demo-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Demo Skill\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "prompt": "Use the skill."}]),
        encoding="utf-8",
    )
    return skill


def _staged_environment_env(task: Path) -> dict[str, str]:
    config = TaskConfig.model_validate_toml((task / "task.toml").read_text(encoding="utf-8"))
    return dict(config.environment.env)


# ---------------------------------------------------------------------------
# The fallback switch in the task environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_skill", [True, False])
def test_generated_tasks_disable_claude_code_refusal_fallback(tmp_path: Path, with_skill: bool) -> None:
    skill = _write_skill(tmp_path)

    [task] = generate_harbor_tasks(skill, tmp_path / "tasks", with_skill=with_skill)

    assert _staged_environment_env(task)[_SWITCH] == "1"


_ROUTES = {
    "gateway": (
        {
            "SKILL_EVAL_LLM_PROVIDER": "openai-compatible",
            "SKILL_EVAL_LLM_BASE_URL": "https://gateway.example/team/v1",
            "SKILL_EVAL_LLM_API_KEY": _FAKE_CREDENTIAL,
        },
        {},
    ),
    "anthropic": (
        {"SKILL_EVAL_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": _FAKE_CREDENTIAL},
        {"ANTHROPIC_API_KEY": _FAKE_CREDENTIAL},
    ),
    "bedrock": (
        {"SKILL_EVAL_LLM_PROVIDER": "bedrock", "AWS_REGION": "us-east-1"},
        {
            "AWS_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": _FAKE_CREDENTIAL,
            "AWS_SECRET_ACCESS_KEY": _FAKE_CREDENTIAL,
        },
    ),
    "nv_build": (
        {"SKILL_EVAL_LLM_PROVIDER": "nv_build", "NVIDIA_API_KEY": _FAKE_CREDENTIAL},
        {"NVIDIA_API_KEY": _FAKE_CREDENTIAL},
    ),
}


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_claude_code_refusal_fallback_is_off_for_every_provider_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: str
) -> None:
    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "AWS_", "CLAUDE_CODE_", "NVIDIA_", "OPENAI_", "SKILL_EVAL_")):
            monkeypatch.delenv(name, raising=False)
    provider_env, host_env = _ROUTES[route]
    for name, value in host_env.items():
        monkeypatch.setenv(name, value)
    provider = resolve_llm_provider(provider_env)
    model, source = _model_for_agent("claude-code", cli_model=None, config_agents={}, provider=provider)
    plan = _resolve_agent_runtime_plan(
        provider=provider,
        agents=["claude-code"],
        models={"claude-code": model},
        model_sources={"claude-code": source},
        configured_runtime_env={},
        env_mode="docker",
    )["claude-code"]

    [task] = generate_harbor_tasks(
        _write_skill(tmp_path), tmp_path / "tasks", with_skill=True, runtime_env=dict(plan.staged_env)
    )

    # The switch is evaluator-managed task environment, not a provider credential:
    # every route stages it, and no route's credential set carries or replaces it.
    assert _staged_environment_env(task)[_SWITCH] == "1"
    assert _SWITCH not in plan.staged_env
    assert _SWITCH not in plan.subprocess_env


def test_refusal_fallback_switch_is_not_redacted_as_a_secret() -> None:
    # Harbor replaces each sensitive value across the trial directory; a value of
    # "1" must never be treated as one.
    assert not is_sensitive_env_key(_SWITCH)


def test_skill_runtime_env_cannot_reenable_refusal_fallback(tmp_path: Path) -> None:
    resolved, errors = _resolve_runtime_env({_SWITCH: "0"}, env_mode="docker")

    assert resolved == {}
    assert errors == [f"harbor.runtime_env.{_SWITCH} controls the host process and is not allowed"]

    _write_task_toml(
        tmp_path,
        {"id": "case-001", "expected_skill": "demo-skill"},
        has_skill=True,
        runtime_env={_SWITCH: "0"},
    )

    assert _staged_environment_env(tmp_path)[_SWITCH] == "1"


def test_native_tasks_cannot_reenable_refusal_fallback(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    native_task = skill / "evals" / "harbor" / "case-001"
    native_task.mkdir(parents=True)
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n'
        f'[environment]\n\n[environment.env]\n{_SWITCH} = "0"\n',
        encoding="utf-8",
    )

    [staged] = stage_native_harbor_tasks(skill, tmp_path / "native")

    assert _staged_environment_env(staged)[_SWITCH] == "1"


@pytest.mark.skipif(os.name == "nt", reason="local mode requires a POSIX host")
def test_local_mode_agent_commands_keep_refusal_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_SWITCH, "0")
    _write_task_toml(tmp_path, {"id": "case-001", "expected_skill": "demo-skill"}, has_skill=True)
    task_config = TaskConfig.model_validate_toml((tmp_path / "task.toml").read_text(encoding="utf-8"))
    (tmp_path / "environment").mkdir()
    (tmp_path / "runtime").mkdir()
    environment = SkillEvaluatorLocalEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="refusal-fallback",
        session_id="refusal-fallback",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=task_config.environment,
        runtime_agent="claude-code",
        runtime_root=str(tmp_path / "runtime"),
        sandbox_mode="off",
    )

    # Harbor's Claude Code agent passes its own per-command variables; the
    # staged task value still reaches the CLI over the host's setting.
    command_env = environment._exec_env({"IS_SANDBOX": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})

    assert command_env[_SWITCH] == "1"


# ---------------------------------------------------------------------------
# Harbor 0.24's own classification of the CLI's refusal (pins a future bump)
# ---------------------------------------------------------------------------


def _load(import_path: str) -> type[ClaudeCode]:
    module, class_name = import_path.split(":")
    return getattr(importlib.import_module(module), class_name)


_CLAUDE_CODE_CLASSES = {
    "harbor": lambda: ClaudeCode,
    "container": lambda: _load(CONTAINER_AGENT_IMPORT_PATHS["claude-code"]),
    "local": lambda: _load(LOCAL_AGENT_IMPORT_PATHS["claude-code"]),
}
# The shape of Harbor's own Claude Code launch, which selects its stream-json classifier.
_COMMAND = "claude --verbose --output-format=stream-json --print -- 'Use the skill.' 2>&1 | tee claude-code.txt"


def _classify(tmp_path: Path, lane: str, stdout: str) -> Exception:
    agent = _CLAUDE_CODE_CLASSES[lane]()(logs_dir=tmp_path, model_name=SELECTED_MODEL)
    return agent._classify_exec_error(_COMMAND, ExecResult(return_code=1, stdout=stdout, stderr=None))


@pytest.mark.parametrize("lane", sorted(_CLAUDE_CODE_CLASSES))
def test_disabled_fallback_refusal_is_harbors_safety_refusal_error(tmp_path: Path, lane: str) -> None:
    error = _classify(tmp_path, lane, no_fallback_stream())

    assert type(error) is AgentSafetyRefusalError
    assert "Command failed (exit 1)" in str(error)


@pytest.mark.parametrize("lane", sorted(_CLAUDE_CODE_CLASSES))
def test_enabled_fallback_gateway_403_reads_as_an_unknown_api_error(tmp_path: Path, lane: str) -> None:
    # Why the switch is needed: with the fallback on, the refusal never reaches
    # the CLI's terminal events, so Harbor sees only the fallback model's 403.
    assert type(_classify(tmp_path, lane, gateway_fallback_403_stream())) is UnknownApiError


@pytest.mark.parametrize("lane", sorted(_CLAUDE_CODE_CLASSES))
def test_refusal_text_in_tool_output_or_model_text_is_not_a_safety_refusal(tmp_path: Path, lane: str) -> None:
    quoted_event = no_fallback_stream().splitlines()[-3]
    model_text = line(
        {
            "type": "assistant",
            "message": {
                "model": SELECTED_MODEL,
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": f"The log says: {NO_FALLBACK_TEXT}"}],
            },
        }
    )
    stdout = "\n".join(
        [
            init_event(),
            *tool_round(f"{quoted_event}\nRequest blocked\n{NO_FALLBACK_TEXT}"),
            model_text,
            result_event("API Error: Overloaded", status=529),
        ]
    )

    error = _classify(tmp_path, lane, stdout)

    assert not isinstance(error, AgentSafetyRefusalError)
    assert type(error) is ApiOverloadedError
