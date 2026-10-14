# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import shlex

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import local_agents, runner


def test_gateway_opencode_docker_uses_chat_completions_adapter():
    provider = ProviderConfig(
        provider="openai-compatible",
        model="vendor/team/model",
        api_key="test-key",
        base_url="https://gateway.example/v1",
        litellm_model="openai/vendor/team/model",
    )
    assert runner._agent_import_path(provider, "opencode", "docker") == (
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorGatewayOpenCode"
    )


def test_gateway_opencode_config_preserves_model_and_uses_env_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-must-not-be-serialized")
    assert hasattr(local_agents, "SkillEvaluatorGatewayOpenCode")
    agent = local_agents.SkillEvaluatorGatewayOpenCode(logs_dir=tmp_path, model_name="openai/vendor/team/model")
    command = agent._build_register_config_command()
    words = shlex.split(command)
    config = json.loads(words[words.index("echo") + 1])
    assert agent.model_name == "skillevaluator-gateway/vendor/team/model"
    route = config["provider"]["skillevaluator-gateway"]
    assert "vendor/team/model" in route["models"]
    assert route["npm"] == "@ai-sdk/openai-compatible"
    assert route["options"] == {"baseURL": "https://gateway.example/v1", "apiKey": "{env:OPENAI_API_KEY}"}
    assert "secret-must-not-be-serialized" not in command
    assert "openai" not in config["provider"]


def test_gateway_opencode_route_is_owned_and_does_not_mutate_caller_config(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    config = {
        "permission": {"bash": "ask"},
        "provider": {
            "skillevaluator-gateway": {
                "npm": "@ai-sdk/openai",
                "options": {"baseURL": "https://other.example", "apiKey": "wrong"},
            }
        },
    }
    before = copy.deepcopy(config)
    assert hasattr(local_agents, "SkillEvaluatorGatewayOpenCode")
    agent = local_agents.SkillEvaluatorGatewayOpenCode(
        logs_dir=tmp_path, model_name="openai/vendor/model", opencode_config=config
    )
    command = agent._build_register_config_command()
    words = shlex.split(command)
    emitted = json.loads(words[words.index("echo") + 1])
    assert config == before
    assert emitted["permission"] == {"bash": "ask"}
    route = emitted["provider"]["skillevaluator-gateway"]
    assert route["npm"] == "@ai-sdk/openai-compatible"
    assert route["options"]["baseURL"] == "https://gateway.example/v1"
    assert route["options"]["apiKey"] == "{env:OPENAI_API_KEY}"
