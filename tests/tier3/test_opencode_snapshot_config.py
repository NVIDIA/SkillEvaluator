# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenCode trials run with snapshots off, so a git EPIPE cannot end a finished trial."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.opencode import OpenCode

from skillevaluator.tier3.harbor.installed_agents import SkillEvaluatorOpenCode
from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorGatewayOpenCode, SkillEvaluatorLocalOpenCode

_WRAPPERS = [SkillEvaluatorOpenCode, SkillEvaluatorLocalOpenCode, SkillEvaluatorGatewayOpenCode]


def _written_config(command: str | None) -> dict:
    assert command is not None
    tokens = shlex.split(command)
    return json.loads(tokens[tokens.index("echo") + 1])


@pytest.mark.parametrize("agent_class", _WRAPPERS)
def test_wrappers_write_config_with_snapshots_off(agent_class: type, tmp_path: Path) -> None:
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    assert _written_config(agent._build_register_config_command())["snapshot"] is False


def test_nvidia_build_config_has_snapshots_off(tmp_path: Path) -> None:
    agent = SkillEvaluatorLocalOpenCode(logs_dir=tmp_path, model_name="nvidia/test-model")

    assert _written_config(agent._nvidia_provider_config_command("test-model"))["snapshot"] is False


@pytest.mark.parametrize("agent_class", _WRAPPERS)
def test_job_opencode_config_can_turn_snapshots_back_on(agent_class: type, tmp_path: Path) -> None:
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model", opencode_config={"snapshot": True})

    assert _written_config(agent._build_register_config_command())["snapshot"] is True


def test_stock_harbor_opencode_is_unchanged(tmp_path: Path) -> None:
    command = OpenCode(logs_dir=tmp_path, model_name="openai/test-model")._build_register_config_command()

    assert command is None or "snapshot" not in _written_config(command)
