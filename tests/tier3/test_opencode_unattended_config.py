# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local OpenCode trials carry on after a denied permission instead of ending the turn."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor")

from skillevaluator.tier3.harbor.installed_agents import SkillEvaluatorOpenCode
from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorGatewayOpenCode, SkillEvaluatorLocalOpenCode


def _written_config(command: str | None) -> dict[str, Any]:
    assert command is not None
    tokens = shlex.split(command)
    return json.loads(tokens[tokens.index("echo") + 1])


def _local_configs(tmp_path: Path, **kwargs: Any) -> list[dict[str, Any]]:
    agent = SkillEvaluatorLocalOpenCode(logs_dir=tmp_path, model_name="openai/test-model", **kwargs)
    nvidia = SkillEvaluatorLocalOpenCode(logs_dir=tmp_path, model_name="nvidia/test-model", **kwargs)
    return [
        _written_config(agent._build_register_config_command()),
        _written_config(nvidia._nvidia_provider_config_command("test-model")),
    ]


def test_local_opencode_config_continues_after_a_denied_permission(tmp_path: Path) -> None:
    for config in _local_configs(tmp_path):
        assert config["experimental"]["continue_loop_on_deny"] is True
        # The inherited default stays.
        assert config["snapshot"] is False


def test_nvidia_provider_config_keeps_provider_entry(tmp_path: Path) -> None:
    agent = SkillEvaluatorLocalOpenCode(logs_dir=tmp_path, model_name="nvidia/test-model")

    config = _written_config(agent._nvidia_provider_config_command("test-model"))

    assert config["provider"]["nvidia"]["models"] == {"test-model": {"name": "test-model"}}


@pytest.mark.parametrize("agent_class", [SkillEvaluatorOpenCode, SkillEvaluatorGatewayOpenCode])
def test_container_wrappers_keep_the_default_turn_behavior(agent_class: type, tmp_path: Path) -> None:
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    config = _written_config(agent._build_register_config_command())

    assert config["snapshot"] is False
    assert "experimental" not in config


def test_job_opencode_config_overrides_the_default(tmp_path: Path) -> None:
    overrides = {"experimental": {"continue_loop_on_deny": False}}

    for config in _local_configs(tmp_path, opencode_config=overrides):
        assert config["experimental"]["continue_loop_on_deny"] is False
        assert config["snapshot"] is False


def test_one_agents_config_does_not_leak_into_the_class_default(tmp_path: Path) -> None:
    SkillEvaluatorLocalOpenCode(
        logs_dir=tmp_path,
        model_name="openai/test-model",
        opencode_config={"experimental": {"continue_loop_on_deny": False}},
    )._build_register_config_command()

    assert SkillEvaluatorLocalOpenCode._DEFAULT_CONFIG["experimental"] == {"continue_loop_on_deny": True}
    assert SkillEvaluatorOpenCode._DEFAULT_CONFIG == {"snapshot": False}
