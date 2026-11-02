# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code background tasks stay off in every staged Tier 3 task.

A headless Claude Code run waits for its background commands, so a command
that never exits held a trial that had already answered until the agent
timeout, and the case was recorded as an unscored agent timeout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from harbor.models.task.config import TaskConfig
from harbor.models.trial.paths import TrialPaths

from skillevaluator.tier3.harbor.adapter import (
    _write_task_toml,
    generate_harbor_tasks,
    stage_native_harbor_tasks,
)
from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment
from skillevaluator.tier3.harbor.runner import _resolve_runtime_env

_SWITCH = "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"


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


@pytest.mark.parametrize("with_skill", [True, False])
def test_generated_tasks_disable_claude_code_background_tasks(tmp_path: Path, with_skill: bool) -> None:
    skill = _write_skill(tmp_path)

    [task] = generate_harbor_tasks(skill, tmp_path / "tasks", with_skill=with_skill)

    assert _staged_environment_env(task)[_SWITCH] == "1"


def test_skill_runtime_env_cannot_reenable_background_tasks(tmp_path: Path) -> None:
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


def test_native_tasks_cannot_reenable_background_tasks(tmp_path: Path) -> None:
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
def test_local_mode_agent_commands_keep_background_tasks_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_SWITCH, "0")
    _write_task_toml(tmp_path, {"id": "case-001", "expected_skill": "demo-skill"}, has_skill=True)
    task_config = TaskConfig.model_validate_toml((tmp_path / "task.toml").read_text(encoding="utf-8"))
    (tmp_path / "environment").mkdir()
    (tmp_path / "runtime").mkdir()
    environment = SkillEvaluatorLocalEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="background-tasks",
        session_id="background-tasks",
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
