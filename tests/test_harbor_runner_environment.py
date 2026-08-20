# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security regressions for the environment passed to the Harbor subprocess."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from skillevaluator.provider_config import ProviderConfig, ProviderConfigurationError, resolve_llm_provider
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.adapter import _verifier_env_vars


def _provider(provider: str = "openai") -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model="test-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
        region="us-west-2" if provider == "bedrock" else None,
    )


def test_docker_opencode_nv_build_uses_operator_key_without_skill_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider("nv_build")
    monkeypatch.setattr(runner.os, "environ", {"PATH": "/usr/bin", "NVIDIA_API_KEY": "nvapi-test"})

    plans = runner._resolve_agent_runtime_plan(
        provider=provider,
        agents=["opencode"],
        models={"opencode": "nvidia/meta/llama-3.1-8b-instruct"},
        configured_runtime_env={},
        env_mode="docker",
    )

    plan = plans["opencode"]
    assert plan.staged_env == {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
    assert plan.subprocess_env["NVIDIA_API_KEY"] == "provider-key"
    assert "OPENAI_API_KEY" not in plan.subprocess_env


@pytest.mark.parametrize(
    ("provider_name", "runtime_model", "catalog_model", "litellm_prefix"),
    [
        ("nv_build", "nvidia/meta/llama-3.1-8b-instruct", "meta/llama-3.1-8b-instruct", "openai"),
        ("openai", "openai/gpt-4.1-mini", "gpt-4.1-mini", "openai"),
        ("openai-compatible", "openai/custom-model", "custom-model", "openai"),
        ("anthropic", "anthropic/claude-sonnet-test", "claude-sonnet-test", "anthropic"),
    ],
)
def test_opencode_runtime_plan_separates_runtime_and_catalog_model_names(
    provider_name: str,
    runtime_model: str,
    catalog_model: str,
    litellm_prefix: str,
) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model=catalog_model,
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model=f"{litellm_prefix}/{catalog_model}",
    )

    plan = runner._resolve_agent_runtime_plan(
        provider=provider,
        agents=["opencode"],
        models={"opencode": runtime_model},
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"opencode": "CLI"},
    )["opencode"]

    assert plan.model == runtime_model
    assert plan.provider.model == catalog_model
    assert plan.provider.litellm_model == f"{litellm_prefix}/{catalog_model}"


@pytest.mark.parametrize(
    ("provider_name", "agent", "model", "litellm_prefix"),
    [
        ("openai-compatible", "codex", "openai/gpt-oss", "openai"),
        ("anthropic", "claude-code", "anthropic/claude-custom", "anthropic"),
    ],
)
def test_non_opencode_runtime_plan_preserves_slash_prefixed_raw_model_ids(
    provider_name: str,
    agent: str,
    model: str,
    litellm_prefix: str,
) -> None:
    provider = ProviderConfig(
        provider=provider_name,
        model=model,
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model=f"{litellm_prefix}/{model}",
    )

    plan = runner._resolve_agent_runtime_plan(
        provider=provider,
        agents=[agent],
        models={agent: model},
        configured_runtime_env={},
        env_mode="docker",
    )[agent]

    assert plan.model == model
    assert plan.provider.model == model
    assert plan.provider.litellm_model == f"{litellm_prefix}/{model}"


@pytest.mark.parametrize("agent", ["codex", "claude-code"])
def test_nvidia_build_docker_bridge_plan_keeps_provider_key_out_of_task_env(agent: str) -> None:
    plan = runner._resolve_agent_runtime_plan(
        provider=_provider("nv_build"),
        agents=[agent],
        models={agent: "openai/gpt-oss-120b"},
        configured_runtime_env={},
        env_mode="docker",
    )[agent]

    assert plan.staged_env == {}
    assert plan.subprocess_env["NVIDIA_API_KEY"] == "provider-key"
    assert plan.provider.provider == "nv_build"
    assert plan.provider.model == "openai/gpt-oss-120b"


@pytest.mark.parametrize(
    "owned_name",
    [
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    ],
)
def test_skill_config_cannot_override_operator_owned_agent_credentials(owned_name: str) -> None:
    with pytest.raises(ValueError, match=rf"operator-owned.*{owned_name}|{owned_name}.*operator-owned"):
        runner._resolve_agent_runtime_plan(
            provider=_provider("nv_build"),
            agents=["opencode"],
            models={"opencode": "nvidia/model"},
            configured_runtime_env={owned_name: "attacker-value"},
            env_mode="docker",
        )


@pytest.mark.parametrize(
    "source_name",
    [
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "AWS_SECRET_ACCESS_KEY",
        "E2B_API_KEY",
        "DOCKER_HOST",
    ],
)
def test_skill_config_cannot_alias_operator_owned_credentials(
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
) -> None:
    monkeypatch.setenv(source_name, "operator-secret")

    resolved, errors = runner._resolve_runtime_env({"INNOCENT_NAME": f"${{{source_name}}}"})

    assert resolved == {}
    assert errors and "operator-owned" in errors[0]
    assert source_name in errors[0]


@pytest.mark.parametrize(
    "name",
    ["SKILL_EVAL_LLM_BASE_URL", "SKILL_EVAL_LLM_PROVIDER", "SKILL_EVAL_LLM_API_KEY"],
)
def test_skill_config_cannot_control_evaluator_provider_routing(name: str) -> None:
    resolved, errors = runner._resolve_runtime_env({name: "https://attacker.example/v1"})

    assert resolved == {}
    assert errors and "host process" in errors[0]


def test_mixed_agents_receive_disjoint_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "NVIDIA_API_KEY": "nvapi-test",
            "ANTHROPIC_API_KEY": "anthropic-test",
            "OPENAI_API_KEY": "openai-test",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
    )

    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("nv_build"),
        agents=["opencode", "claude-code", "codex"],
        models={
            "opencode": "nvidia/nvidia/nemotron-3-nano-30b-a3b",
            "claude-code": "nvidia/nemotron-3-super-120b-a12b",
            "codex": "nvidia/nemotron-3-nano-30b-a3b",
        },
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"opencode": "provider", "claude-code": "CLI", "codex": "CLI"},
    )

    assert plans["opencode"].staged_env == {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
    assert plans["claude-code"].staged_env == {}
    assert plans["codex"].staged_env == {}
    assert plans["opencode"].provider.provider == "nv_build"
    assert plans["claude-code"].provider.provider == "nv_build"
    assert plans["codex"].provider.provider == "nv_build"
    assert "ANTHROPIC_API_KEY" not in plans["opencode"].subprocess_env
    assert "OPENAI_API_KEY" not in plans["opencode"].subprocess_env
    assert "NVIDIA_API_KEY" in plans["claude-code"].subprocess_env  # evaluator verifier only
    assert "OPENAI_API_KEY" not in plans["claude-code"].subprocess_env
    assert "ANTHROPIC_API_KEY" not in plans["codex"].subprocess_env
    with pytest.raises(TypeError):
        plans["opencode"].staged_env["MUTATE"] = "forbidden"  # type: ignore[index]


@pytest.mark.parametrize(
    ("provider_name", "agent"),
    [
        ("bedrock", "codex"),
        ("bedrock", "opencode"),
    ],
)
def test_incompatible_provider_agent_pairs_fail_before_harbor(provider_name: str, agent: str) -> None:
    with pytest.raises(ValueError, match="does not support"):
        runner._resolve_agent_runtime_plan(
            provider=_provider(provider_name),
            agents=[agent],
            models={agent: "test-model"},
            configured_runtime_env={},
            env_mode="docker",
            model_sources={agent: "CLI"},
        )


def test_openai_mixed_agents_receive_isolated_native_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "ANTHROPIC_API_KEY": "anthropic-agent-key",
        },
    )

    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("openai"),
        agents=["codex", "claude-code"],
        models={"codex": "test-model", "claude-code": "anthropic/claude-sonnet-4-5"},
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"codex": "public provider default", "claude-code": "CLI"},
    )

    assert plans["codex"].staged_env == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
    }
    assert plans["claude-code"].staged_env == {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
    assert plans["codex"].provider.provider == "openai"
    assert plans["claude-code"].provider.provider == "anthropic"
    assert plans["claude-code"].provider.model == "claude-sonnet-4-5"
    assert "ANTHROPIC_API_KEY" not in plans["codex"].subprocess_env
    assert plans["claude-code"].subprocess_env["ANTHROPIC_API_KEY"] == "anthropic-agent-key"
    assert plans["claude-code"].subprocess_env["OPENAI_API_KEY"] == "provider-key"


@pytest.mark.parametrize(
    ("provider_name", "env_mode", "configured_base_url", "expected_base_url"),
    [
        ("openai", "docker", "https://agent-gateway.example", "https://agent-gateway.example"),
        ("openai", "e2b", "https://agent-gateway.example/v1", "https://agent-gateway.example"),
        (
            "openai-compatible",
            "docker",
            "https://agent-gateway.example/team/v1/",
            "https://agent-gateway.example/team",
        ),
        ("nv_build", "e2b", "https://agent-gateway.example", "https://agent-gateway.example"),
        ("nv_build", "e2b", "https://agent-gateway.example/v1", "https://agent-gateway.example"),
        (
            "nv_build",
            "e2b",
            "https://agent-gateway.example/team/v1/",
            "https://agent-gateway.example/team",
        ),
    ],
)
def test_independent_claude_base_url_is_canonical_across_the_runtime_plan(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    env_mode: str,
    configured_base_url: str,
    expected_base_url: str,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "ANTHROPIC_API_KEY": "anthropic-agent-key",
            "ANTHROPIC_BASE_URL": configured_base_url,
        },
    )
    evaluator_provider = _provider(provider_name)

    plan = runner._resolve_agent_runtime_plan(
        provider=evaluator_provider,
        agents=["claude-code"],
        models={"claude-code": "anthropic/claude-sonnet-4-5"},
        configured_runtime_env={},
        env_mode=env_mode,
        model_sources={"claude-code": "CLI"},
    )["claude-code"]

    assert plan.staged_env == {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL}",
    }
    assert plan.subprocess_env["ANTHROPIC_API_KEY"] == "anthropic-agent-key"
    assert plan.subprocess_env["ANTHROPIC_BASE_URL"] == expected_base_url
    assert plan.provider.provider == "anthropic"
    assert plan.provider.api_key == "anthropic-agent-key"
    assert plan.provider.base_url == expected_base_url
    assert plan.provider.base_url + "/v1/messages" == expected_base_url + "/v1/messages"
    assert "/v1/v1/" not in plan.provider.base_url + "/v1/messages"

    assert evaluator_provider.api_key == "provider-key"
    assert evaluator_provider.base_url == "https://provider.example/v1"
    if provider_name == "nv_build":
        assert plan.subprocess_env["NVIDIA_API_KEY"] == "provider-key"
        assert "OPENAI_API_KEY" not in plan.subprocess_env
    else:
        assert plan.subprocess_env["OPENAI_API_KEY"] == "provider-key"
        assert plan.subprocess_env["OPENAI_BASE_URL"] == "https://provider.example/v1"


@pytest.mark.parametrize(
    ("provider_name", "env_mode"),
    [
        ("openai", "docker"),
        ("openai-compatible", "docker"),
        ("nv_build", "e2b"),
    ],
)
@pytest.mark.parametrize(
    "configured_base_url",
    [
        "   ",
        "agent-gateway.example/v1",
        "https://agent-gateway.example/v1/messages",
        "https://url-user:url-secret@agent-gateway.example/v1",
        "https://agent-gateway.example/v1?token=url-secret",
        "https://agent-gateway.example/v1\n",
        "https://agent-gateway.example/%2576%2531/%256dessages",
    ],
)
def test_invalid_independent_claude_base_url_fails_before_harbor_without_echoing_secrets(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    env_mode: str,
    configured_base_url: str,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "ANTHROPIC_API_KEY": "anthropic-agent-key",
            "ANTHROPIC_BASE_URL": configured_base_url,
        },
    )

    with pytest.raises(ProviderConfigurationError) as exc_info:
        runner._resolve_agent_runtime_plan(
            provider=_provider(provider_name),
            agents=["claude-code"],
            models={"claude-code": "anthropic/claude-sonnet-4-5"},
            configured_runtime_env={},
            env_mode=env_mode,
            model_sources={"claude-code": "CLI"},
        )

    message = str(exc_info.value)
    assert "ANTHROPIC_BASE_URL" in message
    assert configured_base_url not in message
    assert "url-user" not in message
    assert "url-secret" not in message
    assert "anthropic-agent-key" not in message


@pytest.mark.parametrize("configured_base_url", [None, ""])
def test_unset_independent_claude_base_url_remains_absent(
    monkeypatch: pytest.MonkeyPatch,
    configured_base_url: str | None,
) -> None:
    host_env = {
        "PATH": "/usr/bin",
        "ANTHROPIC_API_KEY": "anthropic-agent-key",
    }
    if configured_base_url is not None:
        host_env["ANTHROPIC_BASE_URL"] = configured_base_url
    monkeypatch.setattr(runner.os, "environ", host_env)

    plan = runner._resolve_agent_runtime_plan(
        provider=_provider("openai"),
        agents=["claude-code"],
        models={"claude-code": "anthropic/claude-sonnet-4-5"},
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"claude-code": "CLI"},
    )["claude-code"]

    assert "ANTHROPIC_BASE_URL" not in plan.staged_env
    assert "ANTHROPIC_BASE_URL" not in plan.subprocess_env
    assert plan.provider.base_url is None


def test_independent_claude_base_url_optional_normalizer_result_is_not_staged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "ANTHROPIC_API_KEY": "anthropic-agent-key",
            "ANTHROPIC_BASE_URL": "https://agent-gateway.example/v1",
        },
    )

    def no_configured_base_url(_value: str, *, variable: str) -> None:
        assert variable == "ANTHROPIC_BASE_URL"

    monkeypatch.setattr(runner, "_normalize_anthropic_base_url", no_configured_base_url)

    plan = runner._resolve_agent_runtime_plan(
        provider=_provider("openai"),
        agents=["claude-code"],
        models={"claude-code": "anthropic/claude-sonnet-4-5"},
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"claude-code": "CLI"},
    )["claude-code"]

    assert "ANTHROPIC_BASE_URL" not in plan.staged_env
    assert "ANTHROPIC_BASE_URL" not in plan.subprocess_env
    assert plan.provider.base_url is None


@pytest.mark.live
@pytest.mark.skipif(shutil.which("claude") is None, reason="Installed Claude Code CLI is required")
def test_installed_claude_uses_canonical_independent_gateway_from_runtime_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request_paths: list[str] = []
    response_events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "gateway-ok"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    response_body = "".join(
        f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n" for event, payload in response_events
    ).encode()

    class RecordingHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:
            del args

        def do_POST(self) -> None:
            content_length = int(self.headers.get("content-length", "0"))
            self.rfile.read(content_length)
            request_paths.append(self.path.partition("?")[0])
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    claude_path = shutil.which("claude")
    assert claude_path is not None
    with ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler) as server:
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        configured_base_url = f"http://127.0.0.1:{server.server_port}/team/v1"
        monkeypatch.setattr(
            runner.os,
            "environ",
            {
                "PATH": os.environ["PATH"],
                "HOME": str(tmp_path),
                "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
                "ANTHROPIC_API_KEY": "test-agent-key",
                "ANTHROPIC_BASE_URL": configured_base_url,
            },
        )
        plan = runner._resolve_agent_runtime_plan(
            provider=_provider("openai"),
            agents=["claude-code"],
            models={"claude-code": "anthropic/claude-sonnet-4-5"},
            configured_runtime_env={},
            env_mode="docker",
            model_sources={"claude-code": "CLI"},
        )["claude-code"]

        try:
            result = subprocess.run(
                [
                    claude_path,
                    "--bare",
                    "--print",
                    "--no-session-persistence",
                    "--disable-slash-commands",
                    "--model",
                    plan.provider.model,
                    "Reply only with gateway-ok",
                ],
                cwd=tmp_path,
                env=dict(plan.subprocess_env),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        finally:
            server.shutdown()
            server_thread.join(timeout=5)

    expected_root = f"http://127.0.0.1:{server.server_port}/team"
    assert plan.provider.base_url == expected_root
    assert plan.subprocess_env["ANTHROPIC_BASE_URL"] == expected_root
    assert plan.subprocess_env["ANTHROPIC_API_KEY"] == "test-agent-key"
    assert plan.subprocess_env["OPENAI_API_KEY"] == "provider-key"
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gateway-ok"
    assert request_paths
    assert set(request_paths) == {"/team/v1/messages"}


def test_anthropic_mixed_agents_receive_isolated_native_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "openai-agent-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
    )

    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("anthropic"),
        agents=["claude-code", "codex"],
        models={"claude-code": "test-model", "codex": "openai/gpt-5.4-mini"},
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"claude-code": "public provider default", "codex": "CLI"},
    )

    assert plans["claude-code"].staged_env == {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL}",
    }
    assert plans["codex"].staged_env == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
    }
    assert plans["claude-code"].provider.provider == "anthropic"
    assert plans["codex"].provider.provider == "openai-compatible"
    assert plans["codex"].provider.model == "gpt-5.4-mini"
    assert "OPENAI_API_KEY" not in plans["claude-code"].subprocess_env
    assert plans["codex"].subprocess_env["OPENAI_API_KEY"] == "openai-agent-key"
    assert plans["codex"].subprocess_env["ANTHROPIC_API_KEY"] == "provider-key"


def test_anthropic_legacy_v1_base_is_canonical_for_verifier_and_native_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.os, "environ", {"PATH": "/usr/bin"})
    provider = resolve_llm_provider(
        {
            "SKILL_EVAL_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "anthropic-provider-key",
            "ANTHROPIC_BASE_URL": "https://gateway.example/team/v1/",
        }
    )

    verifier_env = runner._provider_environment(provider)
    plans = runner._resolve_agent_runtime_plan(
        provider=provider,
        agents=["claude-code", "opencode"],
        models={
            "claude-code": "claude-sonnet-4-5",
            "opencode": "anthropic/claude-sonnet-4-5",
        },
        configured_runtime_env={},
        env_mode="docker",
        model_sources={"claude-code": "public provider default", "opencode": "CLI"},
    )

    assert provider.base_url == "https://gateway.example/team"
    assert verifier_env["ANTHROPIC_BASE_URL"] == "https://gateway.example/team"
    for agent in ("claude-code", "opencode"):
        assert plans[agent].staged_env == {
            "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
            "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL}",
        }
        assert plans[agent].subprocess_env["ANTHROPIC_BASE_URL"] == "https://gateway.example/team"
        assert plans[agent].provider.base_url == "https://gateway.example/team"
        assert plans[agent].subprocess_env["ANTHROPIC_BASE_URL"] + "/v1/messages" == (
            "https://gateway.example/team/v1/messages"
        )
        assert "/v1/v1/" not in plans[agent].subprocess_env["ANTHROPIC_BASE_URL"] + "/v1/messages"


def test_openai_compatible_codex_stages_selected_provider_pair() -> None:
    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("openai-compatible"),
        agents=["codex"],
        models={"codex": "responses-model"},
        configured_runtime_env={},
        env_mode="docker",
    )

    assert plans["codex"].staged_env == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
    }


def test_local_anthropic_opencode_is_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="local mode"):
        runner._resolve_agent_runtime_plan(
            provider=_provider("anthropic"),
            agents=["opencode"],
            models={"opencode": "anthropic/claude-test"},
            configured_runtime_env={},
            env_mode="local",
        )


def test_local_bedrock_agent_is_rejected_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    with pytest.raises(ValueError, match="local mode"):
        runner._resolve_agent_runtime_plan(
            provider=_provider("bedrock"),
            agents=["claude-code"],
            models={"claude-code": "us.anthropic.model"},
            configured_runtime_env={},
            env_mode="local",
        )


def test_docker_bedrock_claude_activates_bedrock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "AWS_ACCESS_KEY_ID": "access",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_REGION": "us-west-2",
        },
    )

    plans = runner._resolve_agent_runtime_plan(
        provider=_provider("bedrock"),
        agents=["claude-code"],
        models={"claude-code": "us.anthropic.model"},
        configured_runtime_env={},
        env_mode="docker",
    )

    plan = plans["claude-code"]
    assert plan.staged_env["CLAUDE_CODE_USE_BEDROCK"] == "${CLAUDE_CODE_USE_BEDROCK}"
    assert plan.subprocess_env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert plan.staged_env["AWS_ACCESS_KEY_ID"] == "${AWS_ACCESS_KEY_ID}"


def test_run_harbor_eval_stages_per_agent_credential_trees(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text("[]\n", encoding="utf-8")
    provider = _provider("nv_build")
    emitted: list[tuple[str, bool, dict[str, str]]] = []
    launched: dict[str, tuple[str, str, dict[str, str]]] = {}

    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: ({"harbor": {"task_source": "evals_json"}}, None),
    )
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])

    def emit(_skill, target, *, with_skill, runtime_env, **_kwargs):
        task = target / "case-001"
        task.mkdir(parents=True)
        emitted.append((str(target.relative_to(tmp_path / "results")), with_skill, dict(runtime_env)))
        return [task]

    def launch(**kwargs):
        launched[kwargs["agent"]] = (
            str(kwargs["with_skill"].relative_to(tmp_path / "results")),
            str(kwargs["baseline"].relative_to(tmp_path / "results")),
            dict(kwargs["run_env"]),
        )
        return []

    monkeypatch.setattr(runner, "generate_harbor_tasks", emit)
    monkeypatch.setattr(runner, "_run_agent_pair", launch)
    monkeypatch.setattr(
        runner,
        "collect_harbor_results",
        lambda **_kwargs: {"execution_status": "complete", "execution_errors": [], "metrics": [], "agents": {}},
    )
    monkeypatch.setattr(runner, "render_agent_eval_html_report", lambda *_args, **_kwargs: tmp_path / "report.html")
    result = runner.run_harbor_eval(
        skill,
        ["opencode", "claude-code"],
        agent_models={
            "opencode": "nvidia/nvidia/nemotron-3-nano-30b-a3b",
            "claude-code": "nvidia/nemotron-3-super-120b-a12b",
        },
        output_dir=tmp_path / "results",
        env_mode="docker",
        keep_harbor_jobs=True,
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert {(path.split("/")[-2], path.split("/")[-1], with_skill) for path, with_skill, _env in emitted} == {
        ("opencode", "with", True),
        ("opencode", "without", False),
        ("claude-code", "with", True),
        ("claude-code", "without", False),
    }
    staged = {(path.split("/")[-2], path.split("/")[-1]): env for path, _with_skill, env in emitted}
    assert staged[("opencode", "with")] == {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
    assert staged[("claude-code", "with")] == {}
    assert launched["opencode"][0].endswith("_harbor-tasks/opencode/with")
    assert launched["claude-code"][1].endswith("_harbor-tasks/claude-code/without")
    assert "ANTHROPIC_API_KEY" not in launched["opencode"][2]
    assert "OPENAI_API_KEY" not in launched["opencode"][2]
    assert launched["claude-code"][2]["NVIDIA_API_KEY"] == "provider-key"


def test_run_harbor_eval_rejects_provenance_key_inside_skill_before_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text("[]\n", encoding="utf-8")
    key_path = skill / "private-output-provenance.key"
    monkeypatch.setenv("SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE", str(key_path))
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: _provider("nv_build"))
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: ({"harbor": {"task_source": "evals_json"}}, None),
    )
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])

    result = runner.run_harbor_eval(
        skill,
        ["opencode"],
        output_dir=tmp_path / "results",
        env_mode="docker",
        agent_runtime_preflight=False,
    )

    assert "error" in result
    assert "provenance key must be outside" in result["error"][0]
    assert not key_path.exists()
    assert not (tmp_path / "results").exists()


def test_harbor_subprocess_environment_excludes_arbitrary_host_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    host_environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path / "tmp"),
        "LANG": "C.UTF-8",
        "DOCKER_HOST": "unix:///safe/docker.sock",
        "HOST_DATABASE_PASSWORD": "must-not-leak",
        "SSH_AUTH_SOCK": "/private/agent.sock",
        "ANTHROPIC_API_KEY": "unrelated-provider-secret",
        "E2B_API_KEY": "unselected-backend-secret",
    }
    monkeypatch.setattr(runner.os, "environ", host_environment)
    provider = _provider()
    provider_env = runner._provider_environment(provider)

    environment = runner._harbor_subprocess_environment(
        env_mode="docker",
        provider=provider,
        configured_runtime_env={"SERVICE_TOKEN": "declared-runtime-secret"},
        provider_env=provider_env,
    )

    assert environment == {
        "PATH": host_environment["PATH"],
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path / "tmp"),
        "LANG": "C.UTF-8",
        "DOCKER_HOST": "unix:///safe/docker.sock",
        "SERVICE_TOKEN": "declared-runtime-secret",
        **provider_env,
    }


def test_harbor_subprocess_environment_includes_only_selected_backend_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "E2B_API_KEY": "selected-key",
            "MODAL_TOKEN_ID": "unselected-id",
            "MODAL_TOKEN_SECRET": "unselected-secret",
        },
    )
    provider = _provider()

    environment = runner._harbor_subprocess_environment(
        env_mode="e2b",
        provider=provider,
        configured_runtime_env={},
        provider_env=runner._provider_environment(provider),
    )

    assert environment["E2B_API_KEY"] == "selected-key"
    assert "MODAL_TOKEN_ID" not in environment
    assert "MODAL_TOKEN_SECRET" not in environment


def test_daytona_subprocess_environment_preserves_jwt_auth_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "DAYTONA_JWT_TOKEN": "jwt-token",
            "DAYTONA_ORGANIZATION_ID": "organization-id",
        },
    )
    provider = _provider()

    environment = runner._harbor_subprocess_environment(
        env_mode="daytona",
        provider=provider,
        configured_runtime_env={},
        provider_env=runner._provider_environment(provider),
    )

    assert environment["DAYTONA_JWT_TOKEN"] == "jwt-token"
    assert environment["DAYTONA_ORGANIZATION_ID"] == "organization-id"


def test_config_declared_substitution_is_resolved_without_leaking_source_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {"PATH": "/usr/bin", "HOME": "/home/test", "HOST_AGENT_TOKEN": "runtime-key", "UNRELATED_SECRET": "no"},
    )
    configured_runtime_env, errors = runner._resolve_runtime_env({"AGENT_TOKEN": "${HOST_AGENT_TOKEN}"})
    provider = _provider()

    environment = runner._harbor_subprocess_environment(
        env_mode="docker",
        provider=provider,
        configured_runtime_env=configured_runtime_env,
        provider_env=runner._provider_environment(provider),
    )

    assert errors == []
    assert environment["AGENT_TOKEN"] == "runtime-key"
    assert "HOST_AGENT_TOKEN" not in environment
    assert "UNRELATED_SECRET" not in environment


def test_runtime_env_rejects_host_process_control_names() -> None:
    unsafe_names = (
        "PATH",
        "PYTHONPATH",
        "HOME",
        "USERPROFILE",
        "XDG_CONFIG_HOME",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_DISABLE_POLICY_SKILLS",
        "CODEX_HOME",
        "GEMINI_CLI_HOME",
        "OPENCODE_CONFIG_DIR",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "BASH_ENV",
        "BASH_FUNC_hidden%%",
        "NODE_OPTIONS",
        "DOCKER_HOST",
        "COMPOSE_FILE",
        "HARBOR_HOME",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "HTTPS_PROXY",
        "E2B_API_KEY",
        "AWS_CONFIG_FILE",
        "AWS_PROFILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    )

    for name in unsafe_names:
        resolved, errors = runner._resolve_runtime_env({name: "attacker-controlled"})
        assert name not in resolved
        assert any(name in error and "host process" in error for error in errors)


def test_bedrock_subprocess_environment_keeps_only_explicit_aws_provider_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "AWS_ACCESS_KEY_ID": "access-key",
            "AWS_SECRET_ACCESS_KEY": "secret-key",
            "AWS_SESSION_TOKEN": "session-token",
            "AWS_PROFILE": "evaluation",
            "AWS_BEARER_TOKEN_BEDROCK": "bearer-token",
            "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/evaluator",
            "AWS_ROLE_SESSION_NAME": "evaluation-session",
            "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/secrets/token",
            "OTHER_CLOUD_SECRET": "must-not-leak",
        },
    )
    provider = _provider("bedrock")
    provider_env = runner._provider_environment(provider)

    environment = runner._harbor_subprocess_environment(
        env_mode="docker",
        provider=provider,
        configured_runtime_env={},
        provider_env=provider_env,
    )

    assert provider_env["AWS_ACCESS_KEY_ID"] == "access-key"
    assert provider_env["AWS_BEARER_TOKEN_BEDROCK"] == "bearer-token"
    assert provider_env["AWS_WEB_IDENTITY_TOKEN_FILE"] == "/var/run/secrets/token"
    assert {
        "AWS_ACCESS_KEY_ID",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    }.issubset(_verifier_env_vars(provider_env))
    assert environment["AWS_ACCESS_KEY_ID"] == "access-key"
    assert environment["AWS_SECRET_ACCESS_KEY"] == "secret-key"
    assert environment["AWS_SESSION_TOKEN"] == "session-token"
    assert environment["AWS_PROFILE"] == "evaluation"
    assert environment["AWS_BEARER_TOKEN_BEDROCK"] == "bearer-token"
    assert environment["AWS_ROLE_ARN"].endswith(":role/evaluator")
    assert environment["AWS_ROLE_SESSION_NAME"] == "evaluation-session"
    assert environment["AWS_WEB_IDENTITY_TOKEN_FILE"] == "/var/run/secrets/token"
    assert "OTHER_CLOUD_SECRET" not in environment


def test_local_nvidia_provider_preserves_explicit_codex_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": str(tmp_path),
            "SKILLEVALUATOR_RUNTIME_DIR": str(tmp_path / "runtimes"),
        },
    )
    provider = _provider("nv_build")

    environment = runner._harbor_subprocess_environment(
        env_mode="local",
        provider=provider,
        configured_runtime_env={
            "OPENAI_API_KEY": "codex-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
        provider_env=runner._provider_environment(provider),
    )

    assert environment["NVIDIA_API_KEY"] == "provider-key"
    assert environment["OPENAI_API_KEY"] == "codex-key"
    assert environment["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
