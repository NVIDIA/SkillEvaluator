# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local mode (`--env-mode local`) wiring."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import (
    ENV_MODE_LOCAL,
    HARBOR_ENV_MODES,
    LOCAL_AGENT_IMPORT_PATHS,
    LOCAL_ENV_IMPORT_PATH,
    local_sandbox,
)
from skillevaluator.tier3.harbor.local_agents import (
    NVIDIA_BUILD_AGENT_IMPORT_PATHS,
    SkillEvaluatorLocalOpenCode,
    SkillEvaluatorNvidiaBuildClaudeCode,
    SkillEvaluatorNvidiaBuildCodex,
)
from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment
from skillevaluator.tier3.harbor.local_runtime import ensure_local_runtimes, validate_local_agents
from skillevaluator.tier3.harbor.runner import (
    _check_prerequisites,
    _harbor_subprocess_environment,
    _local_agent_credentials,
    build_harbor_run_command,
)

_NATIVE_WINDOWS_LOCAL_REASON = "native Windows local mode requires WSL2; these checks exercise the POSIX backend"


def _local_environment(
    tmp_path: Path, *, persistent_env: dict[str, str] | None = None
) -> SkillEvaluatorLocalEnvironment:
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._runtime_root = tmp_path / "runtime"
    environment._runtime_agent = "opencode"
    environment._root = tmp_path / "run"
    environment._workspace = environment._root / "workspace"
    environment._tests = environment._root / "tests"
    environment._solution = environment._root / "solution"
    environment._installed_agent = environment._root / "installed-agent"
    environment._tmp = environment._root / "tmp"
    environment._home = environment._root / "home"
    environment._sandbox_mode = "off"
    environment._allow_net = False
    environment._inherit_agent_keys = False
    environment._strict_reads = False
    environment._active_processes = {}
    environment._persistent_env = persistent_env or {}
    environment._sandbox = local_sandbox.Sandbox(local_sandbox.SandboxPlan("none", "advisory-only", "test"))
    environment.trial_paths = type(
        "TrialPaths",
        (),
        {
            "trial_dir": tmp_path / "trial",
            "agent_dir": tmp_path / "trial" / "agent",
            "verifier_dir": tmp_path / "trial" / "verifier",
            "artifacts_dir": tmp_path / "trial" / "artifacts",
            "reward_json_path": tmp_path / "trial" / "verifier" / "reward.json",
            "reward_text_path": tmp_path / "trial" / "verifier" / "reward.txt",
        },
    )()
    environment.logger = logging.getLogger("test-local-environment")
    for path in (
        environment._workspace,
        environment._tests,
        environment._solution,
        environment._installed_agent,
        environment._tmp,
        environment._home,
        environment.trial_paths.agent_dir,
        environment.trial_paths.verifier_dir,
        environment.trial_paths.artifacts_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return environment


def _provider(name: str, *, api_key: str = "k", base_url: str | None = None) -> ProviderConfig:
    return ProviderConfig(provider=name, model="m", api_key=api_key, base_url=base_url, litellm_model="m", region=None)


def test_local_is_a_registered_env_mode() -> None:
    assert ENV_MODE_LOCAL == "local"
    assert "local" in HARBOR_ENV_MODES


def test_build_command_uses_import_paths_not_env_flag() -> None:
    cmd = build_harbor_run_command(dataset_path="/tmp/ds", agent="opencode", job_name="j", env_mode="local")
    joined = " ".join(cmd)
    assert "--environment-import-path" in cmd
    assert LOCAL_ENV_IMPORT_PATH in cmd
    assert "--agent-import-path" in cmd
    assert LOCAL_AGENT_IMPORT_PATHS["opencode"] in cmd
    # local mode must NOT pass Harbor's --env, and must NOT pass -a: harbor's
    # create_agent_from_config prefers the agent NAME over the import path when
    # both are set, which would run the stock (apt-get bootstrapping) agent.
    assert "--env" not in cmd
    assert "-a" not in cmd
    assert "sandbox_mode=require" in joined
    assert "allow_net=true" in joined  # egress on by default for the live agent
    assert "runtime_agent=opencode" in joined
    assert "strict_reads=false" in joined


def test_build_command_wires_strict_read_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLEVALUATOR_LOCAL_STRICT_READS", "1")

    cmd = build_harbor_run_command(dataset_path="/tmp/ds", agent="opencode", job_name="j", env_mode="local")

    assert "strict_reads=true" in " ".join(cmd)


def test_build_command_docker_mode_uses_secure_import_path() -> None:
    cmd = build_harbor_run_command(dataset_path="/tmp/ds", agent="codex", job_name="j", env_mode="docker")
    assert "-a" in cmd and cmd[cmd.index("-a") + 1] == "codex"
    assert "--env" not in cmd
    assert "--environment-import-path" in cmd


def test_docker_bridge_command_uses_custom_agent_import_without_native_agent_flag() -> None:
    import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex"

    cmd = build_harbor_run_command(
        dataset_path="/tmp/ds",
        agent="codex",
        job_name="nvidia-build-codex",
        env_mode="docker",
        agent_import_path=import_path,
    )

    assert "--env" not in cmd
    assert "--environment-import-path" in cmd
    assert "--agent-import-path" in cmd
    assert cmd[cmd.index("--agent-import-path") + 1] == import_path
    assert "-a" not in cmd


def test_nvidia_build_bridge_agents_are_not_local_mode_agents() -> None:
    assert NVIDIA_BUILD_AGENT_IMPORT_PATHS == {
        "codex": "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex",
        "claude-code": "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode",
    }
    assert "codex" in LOCAL_AGENT_IMPORT_PATHS
    assert LOCAL_AGENT_IMPORT_PATHS["codex"] != NVIDIA_BUILD_AGENT_IMPORT_PATHS["codex"]


def test_local_claude_uses_managed_permissions_and_trial_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.claude_code import ClaudeCode

    from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorLocalClaudeCode

    agent = object.__new__(SkillEvaluatorLocalClaudeCode)
    captured: dict[str, object] = {}

    async def raw_exec(
        _self: ClaudeCode,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        captured["command"] = command
        captured["env"] = dict(env or {})
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(ClaudeCode, "exec_as_agent", raw_exec)

    asyncio.run(
        agent.exec_as_agent(
            object(),
            "claude --permission-mode=bypassPermissions -- 'do not rewrite --permission-mode=bypassPermissions'",
            env={"CLAUDE_CODE_TMPDIR": "/private/tmp/unsafe"},
        )
    )

    command = str(captured["command"])
    launcher = command.partition(" -- ")[0]
    assert "--permission-mode=auto" in launcher
    assert "--permission-mode=bypassPermissions" not in launcher
    assert command.endswith("'do not rewrite --permission-mode=bypassPermissions'")
    assert command.startswith("mkdir -p /logs/agent/claude-tmp && ")
    assert captured["env"] == {"CLAUDE_CODE_TMPDIR": "/logs/agent/claude-tmp"}


def test_local_nvidia_build_codex_starts_authenticated_host_bridge_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from harbor.agents.installed.codex import Codex

    from skillevaluator.tier3.harbor import local_agents

    agent_class = getattr(local_agents, "SkillEvaluatorLocalNvidiaBuildCodex", None)
    assert agent_class is not None
    agent = object.__new__(agent_class)
    agent.model_name = "nvidia/nemotron-3-nano-30b-a3b"
    agent._extra_env = {}
    agent.render_instruction = lambda instruction: instruction
    agent._resolve_auth_json_path = lambda: None
    agent._build_register_skills_command = lambda: None
    agent._build_register_mcp_servers_command = lambda: None
    agent.build_cli_flags = lambda: ""
    agent.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    calls: list[tuple[str, dict[str, str]]] = []
    retained_logs: list[tuple[str, str]] = []
    origins: list[str] = []

    class Environment:
        async def upload_file(self, source: object, destination: object) -> None:
            retained_logs.append((str(destination), Path(source).read_text(encoding="utf-8")))

        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            calls.append((command, dict(env or {})))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def raw_exec(
        _self: Codex,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        calls.append((command, dict(env or {})))
        return SimpleNamespace(return_code=0)

    async def upstream_run(
        self: object,
        *,
        instruction: str,
        environment: object,
        context: object,
    ) -> None:
        _ = (instruction, context)
        origins.append(self._bridge_origin())
        await self.exec_as_agent(
            environment,
            command="codex exec --model nemotron-3-nano-30b-a3b -- test",
            env={"NVIDIA_API_KEY": "must-not-leak"},
        )

    monkeypatch.setattr(Codex, "exec_as_agent", raw_exec)
    monkeypatch.setattr(Codex, "run", upstream_run)
    monkeypatch.setenv("NVIDIA_API_KEY", "real-nvidia-key")

    asyncio.run(agent.run("test", Environment(), None))

    assert len(origins) == 1
    parsed = urlsplit(origins[0])
    assert parsed.hostname == "127.0.0.1"
    assert parsed.port is not None
    setup_commands = [(command, env) for command, env in calls if "model_provider" in command]
    assert len(setup_commands) == 1
    setup_command, setup_env = setup_commands[0]
    assert f'base_url = "{origins[0]}/v1"' in setup_command
    assert "api.openai.com" not in setup_command
    assert "real-nvidia-key" not in setup_command
    assert setup_env["OPENAI_API_KEY"] not in {"real-nvidia-key", "nvidia-build-loopback"}
    client_command, client_env = next((command, env) for command, env in calls if "codex exec" in command)
    assert "env -u NVIDIA_API_KEY" in client_command
    assert "NVIDIA_API_KEY" not in client_env
    assert client_env["OPENAI_API_KEY"] == setup_env["OPENAI_API_KEY"]
    assert retained_logs and retained_logs[0][0].endswith("nvidia-build-bridge.log")

    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", parsed.port))


def test_local_nvidia_build_bridge_closes_if_cancelled_during_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_agents

    agent = object.__new__(local_agents.SkillEvaluatorLocalNvidiaBuildClaudeCode)
    agent.model_name = "nvidia/nemotron-3-super-120b-a12b"
    agent._extra_env = {}
    worker_started = threading.Event()
    release_worker = threading.Event()
    closed = threading.Event()

    class Running:
        origin = "http://127.0.0.1:54321"
        client_token = "per-trial-capability"

        def close(self) -> None:
            closed.set()

    def delayed_start(**_kwargs: object) -> Running:
        worker_started.set()
        assert release_worker.wait(timeout=5)
        return Running()

    async def should_not_run(
        _self: object,
        *,
        instruction: str,
        environment: object,
        context: object,
    ) -> None:
        _ = (instruction, environment, context)
        pytest.fail("agent execution started after startup cancellation")

    class Environment:
        async def upload_file(self, _source: object, _destination: object) -> None:
            return None

    monkeypatch.setattr(local_agents, "start_in_process_bridge", delayed_start)
    monkeypatch.setattr(local_agents.SkillEvaluatorLocalClaudeCode, "run", should_not_run)
    monkeypatch.setenv("NVIDIA_API_KEY", "real-nvidia-key")

    async def exercise() -> None:
        task = asyncio.create_task(agent.run("test", Environment(), None))
        assert await asyncio.to_thread(worker_started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert closed.is_set()
    assert getattr(agent, "_nvidia_build_local_temp_dir", None) is None


def test_local_nvidia_build_bridge_cleanup_preserves_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_agents

    close_started = threading.Event()
    release_close = threading.Event()
    close_finished = threading.Event()

    class Running:
        def close(self) -> None:
            close_started.set()
            assert release_close.wait(timeout=5)
            close_finished.set()

    async def exercise() -> None:
        task = asyncio.create_task(local_agents._close_running_bridge(Running()))
        assert await asyncio.to_thread(close_started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert close_finished.is_set()


def test_start_in_process_bridge_waits_for_readiness_worker_to_finish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from skillevaluator.tier3.harbor import nvidia_build_bridge

    release_started = threading.Event()
    release_allowed = threading.Event()
    release_lock = threading.Lock()
    original_release_request = nvidia_build_bridge._BridgeHTTPServer.release_request
    first_release = True

    def delayed_first_release(server: object, request: object) -> None:
        nonlocal first_release
        with release_lock:
            delay_release = first_release
            first_release = False
        if delay_release:
            release_started.set()
            release_timer = threading.Timer(1.0, release_allowed.set)
            release_timer.daemon = True
            release_timer.start()
            assert release_allowed.wait(timeout=2)
        original_release_request(server, request)

    monkeypatch.setattr(nvidia_build_bridge._BridgeHTTPServer, "release_request", delayed_first_release)

    running = nvidia_build_bridge.start_in_process_bridge(
        api_key="test-readiness-worker-key",
        build_base_url="http://127.0.0.1:8080/v1",
        log_path=tmp_path / "nvidia-build-bridge.log",
        request_transport=lambda _endpoint, _body: b"{}",
    )
    try:
        assert release_started.wait(timeout=1)
        assert running._server.active_workers == 0
    finally:
        release_allowed.set()
        running.close()


def test_local_nvidia_build_bridge_repeated_cancellation_releases_slow_header_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from skillevaluator.tier3.harbor import local_agents, nvidia_build_bridge

    temp_dir = tempfile.TemporaryDirectory(prefix="bridge-repeated-cancel-")
    temp_path = Path(temp_dir.name)

    def transport(_endpoint: str, _body: bytes) -> bytes:
        return json.dumps(
            {
                "id": "chatcmpl-test",
                "model": "nvidia/model",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        ).encode("utf-8")

    running = nvidia_build_bridge.start_in_process_bridge(
        api_key="test-repeated-cancel-key",
        build_base_url="http://127.0.0.1:8080/v1",
        log_path=temp_path / "nvidia-build-bridge.log",
        request_transport=transport,
    )
    port = int(urlsplit(running.origin).port or 0)
    monkeypatch.setattr(nvidia_build_bridge, "REQUEST_HEADER_TIMEOUT_SECONDS", 60.0)
    monkeypatch.setattr(nvidia_build_bridge, "BACKEND_CONNECT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(nvidia_build_bridge, "IN_PROCESS_START_TIMEOUT_SECONDS", 0.05)
    client = socket.create_connection(("127.0.0.1", port), timeout=2)
    stop_sending = threading.Event()

    def drip_headers() -> None:
        while not stop_sending.is_set():
            try:
                client.sendall(b"P")
            except OSError:
                return
            time.sleep(0.015)

    sender = threading.Thread(target=drip_headers)
    sender.start()
    deadline = time.monotonic() + 1
    while running._server.active_workers == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert running._server.active_workers == 1

    close_started = threading.Event()
    release_close = threading.Event()
    original_close = running.close

    def delayed_close() -> None:
        close_started.set()
        assert release_close.wait(timeout=5)
        original_close()

    monkeypatch.setattr(running, "close", delayed_close)
    agent = object.__new__(local_agents.SkillEvaluatorLocalNvidiaBuildCodex)
    agent._nvidia_build_running_bridge = running
    agent._nvidia_build_local_log_path = temp_path / "nvidia-build-bridge.log"
    agent._nvidia_build_local_temp_dir = temp_dir
    agent._nvidia_build_bridge_client_token = running.client_token
    agent._nvidia_build_bridge_origin = running.origin
    agent._nvidia_build_bridge_started = True

    class Environment:
        async def upload_file(self, _source: object, _destination: object) -> None:
            pytest.fail("cancelled cleanup must not upload after cancellation")

    async def exercise() -> None:
        task = asyncio.create_task(agent._cleanup_bridge(Environment()))
        assert await asyncio.to_thread(close_started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise())
    finally:
        release_close.set()
        stop_sending.set()
        client.close()
        sender.join(timeout=1)
        if not running._closed:
            running.close()

    assert not sender.is_alive()
    assert running._closed is True
    assert agent._nvidia_build_running_bridge is None
    assert agent._nvidia_build_local_log_path is None
    assert agent._nvidia_build_local_temp_dir is None
    assert agent._nvidia_build_bridge_client_token is None
    assert agent._nvidia_build_bridge_origin is None
    assert agent._nvidia_build_bridge_started is False
    assert not temp_path.exists()


def test_nvidia_build_codex_bridge_isolated_from_client_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.codex import Codex

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/meta/llama-3.1-8b-instruct"
    calls: list[tuple[str, dict[str, str]]] = []
    root_calls: list[tuple[str, dict[str, str]]] = []
    uploads: list[tuple[str, str, int]] = []

    class Environment:
        async def upload_file(self, source: object, destination: object) -> None:
            source_path = Path(source)
            uploads.append(
                (str(destination), source_path.read_text(encoding="utf-8"), source_path.stat().st_mode & 0o777)
            )

        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            user: str | int | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            destination = root_calls if user == "root" else calls
            destination.append((command, dict(env or {})))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def raw_exec(
        _self: Codex,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        calls.append((command, dict(env or {})))
        return SimpleNamespace(return_code=0)

    async def root_exec(
        _self: Codex,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        root_calls.append((command, dict(env or {})))
        stdout = "http://127.0.0.1:43123\n" if "--check-ready-file" in command else ""
        return SimpleNamespace(return_code=0, stdout=stdout)

    monkeypatch.setattr(Codex, "exec_as_agent", raw_exec)
    monkeypatch.setattr(Codex, "exec_as_root", root_exec)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")
    agent._extra_env = {}
    agent.render_instruction = lambda instruction: instruction
    agent._resolve_auth_json_path = lambda: None
    agent._build_register_skills_command = lambda: None
    agent._build_register_mcp_servers_command = lambda: None
    agent.build_cli_flags = lambda: ""
    agent.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)

    asyncio.run(agent.run("test", Environment(), None))

    assert len(uploads) == 1
    assert uploads[0][0].endswith("nvidia-build-bridge.py")
    secret_handoff = next(
        (command, env) for command, env in root_calls if "SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY" in env
    )
    assert secret_handoff[1]["SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY"] == "nvidia-secret"
    client_token = secret_handoff[1]["SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_CLIENT_TOKEN"]
    assert client_token not in {"", "nvidia-secret", "nvidia-build-loopback"}
    assert len(client_token) >= 32
    assert "nvidia-secret" not in secret_handoff[0]
    assert client_token not in secret_handoff[0]
    assert ".key" in secret_handoff[0]
    assert ".token" in secret_handoff[0]
    bridge_start = next(
        (command, env) for command, env in root_calls if "nvidia-build-bridge.py" in command and "&" in command
    )
    assert bridge_start[1] == {}
    assert "--api-key-file" in bridge_start[0]
    assert "--client-token-file" in bridge_start[0]
    assert "chown 0:0" in bridge_start[0]
    assert bridge_start[0].index("chown 0:0") < bridge_start[0].index("chmod 600")
    assert "--allowed-model nvidia/meta/llama-3.1-8b-instruct" in bridge_start[0]
    assert "--max-requests" in bridge_start[0]
    assert "skillevaluator-nvidia-build-" in bridge_start[0]
    assert "nvidia-secret" not in bridge_start[0]
    assert client_token not in bridge_start[0]
    assert "--port 0" in bridge_start[0]
    assert "--ready-file" in bridge_start[0]
    assert "18080" not in bridge_start[0]
    health_command = next(command for command, _env in root_calls if "--check-ready-file" in command)
    assert "kill -0" in health_command
    assert "/healthz" not in health_command
    assert all("NVIDIA_API_KEY" not in env for _, env in [*calls, *root_calls])
    setup_command, setup_env = next((command, env) for command, env in calls if "model_provider" in command)
    assert 'model_provider = "openai_compatible"' in setup_command
    assert "[model_providers.openai_compatible]" in setup_command
    assert 'base_url = "http://127.0.0.1:43123/v1"' in setup_command
    assert 'wire_api = "responses"' in setup_command
    assert "openai_base_url" not in setup_command
    assert all("openai_base_url" not in command for command, _ in calls)
    assert "nvidia-secret" not in setup_command
    assert setup_env["OPENAI_API_KEY"] == client_token
    assert "OPENAI_BASE_URL" not in setup_env
    client_command, client_env = next((command, env) for command, env in calls if "codex exec" in command)
    assert "NVIDIA_API_KEY" not in client_env
    assert "OPENAI_BASE_URL" not in client_env
    assert client_env["OPENAI_API_KEY"] != "nvidia-secret"
    assert "env -u NVIDIA_API_KEY" in client_command
    assert "--model nvidia/meta/llama-3.1-8b-instruct" in client_command
    cleanup = next(command for command, _ in root_calls if 'kill "$(cat' in command)
    assert "skillevaluator-nvidia-build-" in cleanup
    assert "nvidia-build-bridge.ready" in cleanup
    assert "nvidia-build-bridge.py" in cleanup
    assert ".key" in cleanup
    assert ".token" in cleanup


def test_nvidia_build_container_bridge_streams_secrets_without_host_temp_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.codex import Codex

    from skillevaluator.tier3.harbor import local_agents

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    uploads: list[str] = []
    root_calls: list[tuple[str, dict[str, str]]] = []

    class Environment:
        async def upload_file(self, _source: object, destination: object) -> None:
            uploads.append(str(destination))

        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            root_calls.append((command, dict(env or {})))
            return SimpleNamespace(return_code=0, stdout="")

    def reject_host_secret_file(*_args, **_kwargs):
        raise AssertionError("bridge secrets must not be materialized in host temporary files")

    async def root_exec(
        _self: Codex,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        root_calls.append((command, dict(env or {})))
        stdout = "http://127.0.0.1:43123\n" if "--check-ready-file" in command else ""
        return SimpleNamespace(return_code=0, stdout=stdout)

    monkeypatch.setattr(local_agents.tempfile, "mkstemp", reject_host_secret_file)
    monkeypatch.setattr(Codex, "exec_as_root", root_exec)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")

    asyncio.run(agent._start_bridge(Environment()))
    asyncio.run(agent._cleanup_bridge(Environment()))

    assert len(uploads) == 1
    assert uploads[0].endswith("nvidia-build-bridge.py")
    secret_handoff = next((command, env) for command, env in root_calls if ".key" in command and ".token" in command)
    assert secret_handoff[1]["SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY"] == "nvidia-secret"
    client_token = secret_handoff[1]["SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_CLIENT_TOKEN"]
    assert len(client_token) >= 32
    assert "nvidia-secret" not in secret_handoff[0]
    assert client_token not in secret_handoff[0]
    assert "umask 077" in secret_handoff[0]


@pytest.mark.parametrize(
    "cancel_stage",
    ["bridge-script-upload", "secret-handoff", "startup", "health-check"],
)
def test_nvidia_build_container_bridge_cancellation_cleans_all_private_state_uninterruptibly(
    monkeypatch: pytest.MonkeyPatch,
    cancel_stage: str,
) -> None:
    from harbor.agents.installed.codex import Codex

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    stage_reached = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_commands: list[str] = []

    class Environment:
        async def upload_file(self, _source: object, destination: object) -> None:
            destination_text = str(destination)
            stage = (
                "bridge-script-upload"
                if destination_text.endswith("nvidia-build-bridge.py")
                else "api-key-upload"
                if destination_text.endswith(".key")
                else "client-token-upload"
                if destination_text.endswith(".token")
                else "unknown-upload"
            )
            if stage == cancel_stage:
                stage_reached.set()
                await asyncio.Future()

        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            del command
            if env and "SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY" in env and cancel_stage == "secret-handoff":
                stage_reached.set()
                await asyncio.Future()
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def root_exec(
        _self: Codex,
        _environment: object,
        command: str,
        **kwargs: object,
    ) -> SimpleNamespace:
        if 'kill "$(cat' in command:
            cleanup_commands.append(command)
            cleanup_started.set()
            await release_cleanup.wait()
            return SimpleNamespace(return_code=0, stdout="")
        environment = kwargs.get("env")
        stage = (
            "secret-handoff"
            if isinstance(environment, dict) and "SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY" in environment
            else "health-check"
            if "--check-ready-file" in command
            else "startup"
            if "nvidia-build-bridge.py" in command and "&" in command
            else "health-check"
        )
        if stage == cancel_stage:
            stage_reached.set()
            await asyncio.Future()
        stdout = "http://127.0.0.1:43123\n" if stage == "health-check" else ""
        return SimpleNamespace(return_code=0, stdout=stdout)

    monkeypatch.setattr(Codex, "exec_as_root", root_exec)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")

    async def exercise() -> None:
        nonlocal stage_reached, cleanup_started, release_cleanup
        stage_reached = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        task = asyncio.create_task(agent._start_bridge(Environment()))
        try:
            await asyncio.wait_for(stage_reached.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(cleanup_started.wait(), timeout=1)
            task.cancel()
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release_cleanup.set()
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    asyncio.run(exercise())

    assert len(cleanup_commands) == 1
    assert ".key" in cleanup_commands[0]
    assert ".token" in cleanup_commands[0]
    assert agent._nvidia_build_bridge_started is False
    assert agent._nvidia_build_bridge_key_file is None
    assert agent._nvidia_build_bridge_client_token_file is None
    assert agent._nvidia_build_bridge_client_token is None


def test_nvidia_build_bridge_prefers_file_backed_host_key_over_subprocess_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from harbor.agents.installed.codex import Codex

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    agent._nvidia_build_bridge_started = False
    agent._nvidia_build_bridge_key_file = None
    root_calls: list[tuple[str, dict[str, str]]] = []
    uploads: list[str] = []
    host_key_file = tmp_path / "nvidia-build-host-key"
    host_key_file.write_text("real-nvidia-secret", encoding="utf-8")

    class Environment:
        async def upload_file(self, source: object, destination: object) -> None:
            del source
            uploads.append(str(destination))

        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            root_calls.append((command, dict(env or {})))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def root_exec(
        _self: Codex,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        root_calls.append((command, dict(env or {})))
        stdout = "http://127.0.0.1:43123\n" if "--check-ready-file" in command else ""
        return SimpleNamespace(return_code=0, stdout=stdout)

    monkeypatch.setattr(Codex, "exec_as_root", root_exec)
    monkeypatch.setenv("NVIDIA_API_KEY", "skillevaluator-file-backed-nvidia-key")
    monkeypatch.setenv("SKILLEVALUATOR_NVIDIA_API_KEY_FILE", str(host_key_file))

    asyncio.run(agent._start_bridge(Environment()))
    asyncio.run(agent._cleanup_bridge(Environment()))

    assert len(uploads) == 1
    assert uploads[0].endswith("nvidia-build-bridge.py")
    secret_handoff = next(
        (command, env) for command, env in root_calls if "SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY" in env
    )
    assert secret_handoff[1]["SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY"] == "real-nvidia-secret"
    client_token = secret_handoff[1]["SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_CLIENT_TOKEN"]
    assert len(client_token) >= 32
    assert host_key_file.exists()
    assert all("real-nvidia-secret" not in command for command, _env in root_calls)
    assert all("skillevaluator-file-backed-nvidia-key" not in command for command, _env in root_calls)
    assert client_token not in secret_handoff[0]


def test_nvidia_build_bridge_rejects_file_backed_sentinel_without_host_key_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)

    class Environment:
        async def exec_with_sensitive_env(self, **_kwargs: object) -> SimpleNamespace:
            raise AssertionError("credential validation must fail before execution")

    monkeypatch.setenv("NVIDIA_API_KEY", "skillevaluator-file-backed-nvidia-key")
    monkeypatch.delenv("SKILLEVALUATOR_NVIDIA_API_KEY_FILE", raising=False)

    with pytest.raises(RuntimeError, match="SKILLEVALUATOR_NVIDIA_API_KEY_FILE"):
        asyncio.run(agent._start_bridge(Environment()))


def test_nvidia_build_bridge_rejects_unsafe_environment_before_reading_or_transferring_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_agents

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    uploads: list[str] = []
    executions: list[tuple[str, dict[str, str]]] = []
    key_reads: list[bool] = []

    class UnsafeEnvironment:
        async def upload_file(self, _source: object, destination: object) -> None:
            uploads.append(str(destination))

        async def exec(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            executions.append((command, dict(env or {})))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    def reject_key_read() -> str:
        key_reads.append(True)
        raise AssertionError("unsupported environments must be rejected before consuming stdin")

    monkeypatch.setattr(local_agents, "read_nvidia_build_key_from_stdin", reject_key_read)
    monkeypatch.setenv("NVIDIA_API_KEY", local_agents.NVIDIA_BUILD_STDIN_SENTINEL)

    with pytest.raises(RuntimeError, match="protected sensitive-value transport"):
        asyncio.run(agent._start_bridge(UnsafeEnvironment()))

    assert key_reads == []
    assert uploads == []
    assert executions == []

    agent._nvidia_build_bridge_client_env = {"OPENAI_API_KEY": "bridge-client-token-secret"}
    with pytest.raises(RuntimeError, match="protected sensitive-value transport"):
        asyncio.run(agent.exec_as_agent(UnsafeEnvironment(), command="codex exec -- test"))
    assert executions == []


def test_nvidia_build_bridge_health_failure_cleans_up_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.codex import Codex

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    commands: list[str] = []

    class Environment:
        async def upload_file(self, _source: object, _destination: object) -> None:
            return None

        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            commands.append(command)
            assert env and "SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY" in env
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def raw_exec(
        _self: Codex,
        _environment: object,
        command: str,
        **_kwargs: object,
    ) -> SimpleNamespace:
        commands.append(command)
        is_health = "/healthz" in command or "--check-ready-file" in command
        return SimpleNamespace(return_code=1 if is_health else 0, stdout="")

    monkeypatch.setattr(Codex, "exec_as_agent", raw_exec)
    monkeypatch.setattr(Codex, "exec_as_root", raw_exec)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")

    with pytest.raises(RuntimeError, match="health check"):
        asyncio.run(agent.run("test", Environment(), None))

    assert any("--check-ready-file" in command for command in commands)
    assert any('kill "$(cat' in command for command in commands)


def test_nvidia_build_bridge_start_failure_removes_uploaded_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.codex import Codex

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    commands: list[str] = []

    class Environment:
        async def upload_file(self, _source: object, _destination: object) -> None:
            return None

        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            commands.append(command)
            assert env and "SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY" in env
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def raw_exec(
        _self: Codex,
        _environment: object,
        command: str,
        **_kwargs: object,
    ) -> SimpleNamespace:
        commands.append(command)
        is_start = "nvidia-build-bridge.py" in command and "&" in command
        return SimpleNamespace(return_code=1 if is_start else 0)

    monkeypatch.setattr(Codex, "exec_as_agent", raw_exec)
    monkeypatch.setattr(Codex, "exec_as_root", raw_exec)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")

    with pytest.raises(RuntimeError, match="startup"):
        asyncio.run(agent.run("test", Environment(), None))

    cleanup = next(command for command in commands if 'kill "$(cat' in command)
    assert "skillevaluator-nvidia-build-" in cleanup
    assert "nvidia-build-bridge.ready" in cleanup


def test_nvidia_build_bridge_start_exception_removes_uploaded_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.codex import Codex

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    commands: list[str] = []

    class Environment:
        async def upload_file(self, _source: object, _destination: object) -> None:
            return None

        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            commands.append(command)
            assert env and "SKILLEVALUATOR_NVIDIA_BUILD_BRIDGE_API_KEY" in env
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def raw_exec(
        _self: Codex,
        _environment: object,
        command: str,
        **_kwargs: object,
    ) -> SimpleNamespace:
        commands.append(command)
        if "nvidia-build-bridge.py" in command and "&" in command:
            raise RuntimeError("docker exec failed")
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(Codex, "exec_as_agent", raw_exec)
    monkeypatch.setattr(Codex, "exec_as_root", raw_exec)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")

    with pytest.raises(RuntimeError, match="docker exec failed"):
        asyncio.run(agent.run("test", Environment(), None))

    cleanup = next(command for command in commands if 'kill "$(cat' in command)
    assert "skillevaluator-nvidia-build-" in cleanup
    assert "nvidia-build-bridge.ready" in cleanup


def test_nvidia_build_bridge_wraps_compound_codex_shell_commands_before_unsetting_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.codex import Codex

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    agent._nvidia_build_bridge_origin = "http://127.0.0.1:43123"
    agent._nvidia_build_bridge_client_env = {
        "OPENAI_API_KEY": "bridge-client-token-secret",
        "OPENAI_BASE_URL": "http://127.0.0.1:43123/v1",
    }
    agent._extra_env = {
        "NVIDIA_API_KEY": "extra-upstream-secret",
        "CODEX_HOME": "/logs/agent/codex-home",
    }
    log_records: list[tuple[object, object]] = []
    agent.logger = SimpleNamespace(debug=lambda message, *args, **kwargs: log_records.append((message, (args, kwargs))))
    captured: list[tuple[str, dict[str, str]]] = []
    simple_command = "codex exec --model model -- test"
    compound_command = 'if [ -d "$CODEX_HOME/sessions" ]; then cp -R "$CODEX_HOME/sessions" /logs/agent/sessions; fi'

    class Environment:
        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            captured.append((command, dict(env or {})))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def reject_logged_exec(
        _self: Codex,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        _ = (command, env)
        raise AssertionError("bridge credentials must bypass Harbor's logging wrapper")

    monkeypatch.setattr(Codex, "exec_as_agent", reject_logged_exec)

    for original_command in (simple_command, compound_command):
        asyncio.run(
            agent.exec_as_agent(
                Environment(),
                command=original_command,
                env={"NVIDIA_API_KEY": "per-call-upstream-secret"},
            )
        )

    assert [command for command, _ in captured] == [
        f"env -u NVIDIA_API_KEY bash -o pipefail -c {shlex.quote('codex exec --model nvidia/model -- test')}",
        f"env -u NVIDIA_API_KEY bash -o pipefail -c {shlex.quote(compound_command)}",
    ]
    assert all("NVIDIA_API_KEY" not in env for _, env in captured)
    assert all(env["OPENAI_API_KEY"] == "bridge-client-token-secret" for _, env in captured)
    assert all(env["CODEX_HOME"] == "/logs/agent/codex-home" for _, env in captured)
    assert log_records == []


def test_nvidia_build_bridge_client_preserves_pipefail_in_real_inner_shell() -> None:
    from harbor.agents.installed.base import NonZeroAgentExitCodeError

    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    agent._extra_env = {"NVIDIA_API_KEY": "upstream-key-must-be-unset"}
    agent._nvidia_build_bridge_origin = "http://127.0.0.1:43123"
    agent._nvidia_build_bridge_client_env = {"OPENAI_API_KEY": "bridge-client-token-secret"}

    class Environment:
        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                env={**os.environ, **(env or {})},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return SimpleNamespace(
                return_code=process.returncode,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
            )

    with pytest.raises(NonZeroAgentExitCodeError, match="exit code 1"):
        asyncio.run(agent.exec_as_agent(Environment(), command="false | true"))


def test_nvidia_build_bridge_client_failure_does_not_expose_sensitive_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.base import NonZeroAgentExitCodeError
    from harbor.agents.installed.codex import Codex

    bridge_token = "bridge-client-token-secret"
    upstream_key = "upstream-nvidia-key-secret"
    agent = object.__new__(SkillEvaluatorNvidiaBuildCodex)
    agent.model_name = "nvidia/model"
    agent._extra_env = {"NVIDIA_API_KEY": upstream_key}
    agent._nvidia_build_bridge_origin = "http://127.0.0.1:43123"
    agent._nvidia_build_bridge_client_env = {"OPENAI_API_KEY": bridge_token}

    class Environment:
        async def exec_with_sensitive_env(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                return_code=17,
                stdout=f"provider echoed {bridge_token}",
                stderr=f"provider echoed {upstream_key}",
            )

    async def reject_logged_exec(
        _self: Codex,
        _environment: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        raise AssertionError("bridge credentials must bypass Harbor's logging wrapper")

    monkeypatch.setattr(Codex, "exec_as_agent", reject_logged_exec)

    with pytest.raises(NonZeroAgentExitCodeError) as exc_info:
        asyncio.run(
            agent.exec_as_agent(
                Environment(),
                command="codex exec -- test",
                env={"NVIDIA_API_KEY": "per-call-upstream-key-secret"},
            )
        )

    message = str(exc_info.value)
    assert message == "NVIDIA Build bridge client command failed with exit code 17"
    assert bridge_token not in message
    assert upstream_key not in message


def test_nvidia_build_claude_bridge_configures_origin_and_full_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.claude_code import ClaudeCode

    agent = object.__new__(SkillEvaluatorNvidiaBuildClaudeCode)
    agent.model_name = "nvidia/meta/llama-3.1-8b-instruct"
    calls: list[tuple[str, dict[str, str]]] = []

    class Environment:
        async def upload_file(self, _source: object, _destination: object) -> None:
            return None

        async def exec_with_sensitive_env(
            self,
            command: str,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> SimpleNamespace:
            calls.append((command, dict(env or {})))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def raw_exec(
        _self: ClaudeCode,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        calls.append((command, dict(env or {})))
        stdout = "http://127.0.0.1:43123\n" if "--check-ready-file" in command else ""
        return SimpleNamespace(return_code=0, stdout=stdout)

    async def parent_run(
        self: ClaudeCode,
        *,
        instruction: str,
        environment: object,
        context: object,
    ) -> None:
        _ = (instruction, context)
        await self.exec_as_agent(environment, command="claude --print -- test", env={})

    monkeypatch.setattr(ClaudeCode, "exec_as_agent", raw_exec)
    monkeypatch.setattr(ClaudeCode, "exec_as_root", raw_exec)
    monkeypatch.setattr(ClaudeCode, "run", parent_run)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-secret")

    asyncio.run(agent.run("test", Environment(), None))

    client_command, client_env = next((command, env) for command, env in calls if "claude --print" in command)
    assert "env -u NVIDIA_API_KEY" in client_command
    assert "NVIDIA_API_KEY" not in client_env
    # This is configuration-only: a live Docker smoke test is required to
    # verify the installed Claude CLI's actual request construction.
    assert client_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:43123"
    assert client_env["ANTHROPIC_API_KEY"] != "nvidia-secret"
    assert client_env["ANTHROPIC_MODEL"] == "nvidia/meta/llama-3.1-8b-instruct"


@pytest.mark.live
@pytest.mark.skip(
    reason=(
        "Manual Docker E2E only: requires NVIDIA_API_KEY, Docker, and the installed Claude Code CLI; "
        "do not run in the unit suite."
    )
)
def test_nvidia_build_claude_bridge_live_smoke() -> None:
    """Manual scope: prove a Docker Claude CLI request reaches bridge /v1/messages."""
    pytest.fail("run the documented manual Docker E2E smoke with a real NVIDIA Build credential")


def test_local_agent_credentials_map_provider_to_agent_env() -> None:
    nv = _local_agent_credentials(
        _provider("nv_build", api_key="nvapi-x", base_url="https://integrate.api.nvidia.com/v1")
    )
    assert nv == {"OPENAI_API_KEY": "nvapi-x", "OPENAI_BASE_URL": "https://integrate.api.nvidia.com/v1"}
    anthropic = _local_agent_credentials(_provider("anthropic", api_key="sk-ant"))
    assert anthropic == {"ANTHROPIC_API_KEY": "sk-ant"}
    openai = _local_agent_credentials(_provider("openai", api_key="sk-o", base_url="https://api.openai.com/v1"))
    assert openai == {"OPENAI_API_KEY": "sk-o", "OPENAI_BASE_URL": "https://api.openai.com/v1"}


def test_local_subprocess_environment_keeps_only_the_trusted_nvidia_parent_key() -> None:
    provider = _provider("nv_build", api_key="nvapi-x", base_url="https://integrate.api.nvidia.com/v1")
    configured = {
        "OPENAI_API_KEY": "openai-key",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "ANTHROPIC_API_KEY": "anthropic-key",
    }
    provider_env = {"NVIDIA_API_KEY": "nvapi-x"}

    opencode = _harbor_subprocess_environment(
        env_mode="local",
        provider=provider,
        configured_runtime_env=configured,
        provider_env=provider_env,
        agent="opencode",
        agent_model="nvidia/meta/llama-3.1-8b-instruct",
    )
    codex = _harbor_subprocess_environment(
        env_mode="local",
        provider=provider,
        configured_runtime_env=configured,
        provider_env=provider_env,
        agent="codex",
        agent_model="nvidia/nemotron-3-nano-30b-a3b",
    )
    claude = _harbor_subprocess_environment(
        env_mode="local",
        provider=provider,
        configured_runtime_env=configured,
        provider_env=provider_env,
        agent="claude-code",
        agent_model="nvidia/nemotron-3-nano-30b-a3b",
    )

    assert opencode["OPENAI_API_KEY"] == "nvapi-x"
    assert opencode["OPENAI_BASE_URL"] == "https://integrate.api.nvidia.com/v1"
    assert "ANTHROPIC_API_KEY" not in opencode
    assert codex["NVIDIA_API_KEY"] == "nvapi-x"
    assert "OPENAI_API_KEY" not in codex
    assert "OPENAI_BASE_URL" not in codex
    assert "ANTHROPIC_API_KEY" not in codex
    assert claude["NVIDIA_API_KEY"] == "nvapi-x"
    assert "ANTHROPIC_API_KEY" not in claude
    assert "OPENAI_API_KEY" not in claude
    assert "OPENAI_BASE_URL" not in claude


def test_local_host_env_excludes_provider_keys_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    env = SkillEvaluatorLocalEnvironment._local_host_env(inherit_agent_keys=False)
    assert "NVIDIA_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "PATH" in env


def test_local_host_env_inherits_on_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret")
    env = SkillEvaluatorLocalEnvironment._local_host_env(inherit_agent_keys=True)
    assert env["NVIDIA_API_KEY"] == "nvapi-secret"


def test_default_runtime_root_rejects_host_home(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    monkeypatch.setenv("SKILLEVALUATOR_RUNTIME_DIR", str(Path.home()))

    with pytest.raises(ValueError, match="dedicated subdirectory"):
        local_runtime.default_runtime_root()


def test_runtime_read_binds_reject_host_home(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    environment._runtime_root = Path.home()

    with pytest.raises(ValueError, match="dedicated subdirectory"):
        environment._runtime_ro_binds()


def test_background_server_is_rejected_even_with_declared_ports(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    command = "python -m http.server 8000 &"

    reason = environment._local_command_guardrail_reason(
        command,
        command,
        {"HARBOR_DECLARED_PORTS": "8000"},
    )

    assert "unsupported in local mode" in reason
    assert "Docker" in reason


@pytest.mark.parametrize(
    "command",
    [
        "setsid sh -c 'curl https://example.com &'",
        "bash -c 'setsid sleep 60'",
        "bash -lc 'nohup sleep 60'",
        "env -i setsid sleep 60",
        "env -i SAFE=1 nohup sleep 60",
        "nice -n 5 setsid sleep 60",
        "nice nohup sleep 60",
    ],
)
def test_detached_setsid_process_is_rejected_before_launch(tmp_path: Path, command: str) -> None:
    environment = _local_environment(tmp_path)

    reason = environment._local_command_guardrail_reason(
        command,
        command,
        {"HARBOR_DECLARED_PORTS": "443"},
    )

    assert "detached" in reason
    assert "Docker" in reason


def test_detached_launcher_word_as_plain_argument_is_not_rejected(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    command = "printf '%s' daemon"

    assert environment._local_command_guardrail_reason(command, command, {}) == ""


def test_quoted_url_ampersand_is_not_treated_as_background_operator(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    command = 'printf %s "https://example.com/query?a=1&b=2"'

    reason = environment._local_command_guardrail_reason(command, command, {})

    assert reason == ""


@pytest.mark.parametrize(
    "command",
    (
        "sleep 30 & printf wait",
        "sleep 30 & wait",
        "bash -c '(sleep 30) >/dev/null 2>&1 &'",
        "sh -lc 'sleep 30 & wait'",
    ),
)
def test_background_command_cannot_bypass_guard_with_wait_argument(tmp_path: Path, command: str) -> None:
    environment = _local_environment(tmp_path)

    reason = environment._local_command_guardrail_reason(command, command, {})

    assert "unsupported in local mode" in reason


def test_nested_background_shell_is_blocked_before_process_survives(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    marker = environment._workspace / "nested-background-survived"
    command = "bash -c '(sleep .2; printf survived > nested-background-survived) >/dev/null 2>&1 &'"

    async def run_probe() -> object:
        result = await environment.exec(command)
        await asyncio.sleep(0.3)
        return result

    result = asyncio.run(run_probe())

    assert result.return_code == 126
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_task_env_is_hidden_from_launcher_but_reaches_inner_command(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    captured_env = tmp_path / "launcher-env.json"
    launcher = tmp_path / "capture-launcher.py"
    launcher.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "capture = {\n"
        "    'environment': dict(os.environ),\n"
        "    'payload_files': sorted(path.name for path in Path(sys.argv[2]).glob('.command-env-*')),\n"
        "}\n"
        "Path(sys.argv[1]).write_text(json.dumps(capture), encoding='utf-8')\n"
        "os.execvp(sys.argv[3], sys.argv[3:])\n",
        encoding="utf-8",
    )

    class CaptureLauncher:
        plan = local_sandbox.SandboxPlan("none", "advisory-only", "capture")

        @staticmethod
        def wrap(argv: list[str], **_kwargs: object) -> list[str]:
            return [sys.executable, str(launcher), str(captured_env), str(environment._tmp), *argv]

    environment._sandbox = CaptureLauncher()

    result = asyncio.run(environment.exec('printf %s "$SAFE_TASK_VALUE"', env={"SAFE_TASK_VALUE": "inner-only"}))

    assert result.return_code == 0, result.stderr
    assert result.stdout == "inner-only"
    capture = json.loads(captured_env.read_text(encoding="utf-8"))
    launcher_env = capture["environment"]
    assert "SAFE_TASK_VALUE" not in launcher_env
    assert capture["payload_files"] == []


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_exec_forwards_strict_read_policy_to_sandbox(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    environment._strict_reads = True
    captured: dict[str, object] = {}

    class CaptureSandbox:
        plan = local_sandbox.SandboxPlan("none", "advisory-only", "capture")

        @staticmethod
        def wrap(argv: list[str], **kwargs: object) -> list[str]:
            captured.update(kwargs)
            return argv

    environment._sandbox = CaptureSandbox()

    result = asyncio.run(environment.exec("printf ok"))

    assert result.return_code == 0
    assert captured["strict_reads"] is True


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("strict_reads", [False, True])
def test_seatbelt_exec_uses_canonical_interpreter_for_sandbox_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strict_reads: bool,
) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setattr(sys, "executable", str(venv_python))
    environment = _local_environment(tmp_path)
    environment._strict_reads = strict_reads
    captured: dict[str, object] = {}

    class CaptureSandbox:
        plan = local_sandbox.SandboxPlan("seatbelt", "kernel-macos", "capture")

        @staticmethod
        def wrap(argv: list[str], **_kwargs: object) -> list[str]:
            captured["argv"] = argv
            return argv

    environment._sandbox = CaptureSandbox()

    result = asyncio.run(environment.exec("printf ok"))

    assert result.return_code == 0, result.stderr
    wrapped_argv = captured["argv"]
    assert isinstance(wrapped_argv, list)
    assert wrapped_argv[0] == str(Path(sys.executable).resolve())


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize(
    ("backend", "strength", "strict_reads"),
    [
        ("bubblewrap", "kernel", True),
        ("none", "advisory-only", True),
    ],
)
def test_other_sandbox_modes_preserve_venv_interpreter_for_bootstrap(
    tmp_path: Path,
    backend: str,
    strength: str,
    strict_reads: bool,
) -> None:
    environment = _local_environment(tmp_path)
    environment._strict_reads = strict_reads
    captured: dict[str, object] = {}

    class CaptureSandbox:
        plan = local_sandbox.SandboxPlan(backend, strength, "capture")

        @staticmethod
        def wrap(argv: list[str], **_kwargs: object) -> list[str]:
            captured["argv"] = argv
            return argv

    environment._sandbox = CaptureSandbox()

    result = asyncio.run(environment.exec("printf ok"))

    assert result.return_code == 0, result.stderr
    wrapped_argv = captured["argv"]
    assert isinstance(wrapped_argv, list)
    assert wrapped_argv[0] == sys.executable


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists(),
    reason="requires macOS Seatbelt",
)
def test_strict_exec_bootstrap_runs_from_fresh_private_tmp_venv_under_real_seatbelt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_interpreter = next(
        (
            candidate
            for prefix in (Path("/opt/homebrew/bin"), Path("/usr/local/bin"))
            for version in ("3.13", "3.12")
            if (candidate := prefix / f"python{version}").exists()
        ),
        None,
    )
    if base_interpreter is None:
        pytest.skip("requires a Homebrew Python that creates a symlinked venv interpreter")

    with tempfile.TemporaryDirectory(prefix="skillevaluator-seatbelt-venv-", dir="/private/tmp") as temp_dir:
        venv_root = Path(temp_dir) / "venv"
        subprocess.run([str(base_interpreter), "-m", "venv", str(venv_root)], check=True, timeout=60)
        venv_python = venv_root / "bin" / "python"

        environment = _local_environment(tmp_path)
        environment._sandbox_mode = "require"
        environment._strict_reads = True
        environment._sandbox = local_sandbox.detect("require")

        with monkeypatch.context() as patch:
            patch.setattr(sys, "executable", str(venv_python))
            patch.setattr(sys, "prefix", str(venv_root))
            patch.setattr(sys, "exec_prefix", str(venv_root))
            result = asyncio.run(environment.exec("printf strict-bootstrap-ok"))

    assert result.return_code == 0, result.stderr
    assert result.stdout == "strict-bootstrap-ok"


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_exec_timeout_terminates_background_descendants(tmp_path: Path) -> None:
    # Deterministic under CPU load: the child sleeps far longer than the exec
    # timeout (so it can never legitimately write its marker), and instead of
    # trusting fixed sleeps the test polls until the recorded child PID is
    # gone. The previous 0.2s-timeout/0.5s-sleep pairing flaked under
    # pytest-xdist when scheduling latency ate the margins.
    environment = _local_environment(tmp_path)
    started = environment._workspace / "timeout-child-started"
    marker = environment._workspace / "timeout-child-survived"
    child_pid_path = environment._workspace / "timeout-child-pid"
    command = (
        "printf stdout-before-timeout; printf stderr-before-timeout >&2; "
        "printf started > timeout-child-started; "
        "(sleep 15; printf survived > timeout-child-survived) & "
        "printf '%s' \"$!\" > timeout-child-pid; wait"
    )
    environment._local_command_guardrail_reason = lambda *_args: ""  # type: ignore[method-assign]

    async def run_timeout() -> object:
        return await environment.exec(command, timeout_sec=1.0)

    result = asyncio.run(run_timeout())

    assert result.return_code == 124
    assert result.stdout == "stdout-before-timeout"
    assert "stderr-before-timeout" in (result.stderr or "")
    assert "Timed out" in (result.stderr or "")
    assert started.exists(), "the background descendant did not start before the timeout"

    def process_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 10
    while process_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not process_exists(child_pid), "a background descendant survived the timeout kill"
    assert not marker.exists(), "a background descendant wrote after the command timed out"


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_exec_cancellation_terminates_background_descendants(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    child_ready = environment._workspace / "cancel-child-ready"
    child_pid_path = environment._workspace / "cancel-child-pid"
    marker = environment._workspace / "cancel-child-survived"
    command = (
        "(printf ready > cancel-child-ready; sleep 30; "
        "printf survived > cancel-child-survived) & "
        "printf '%s' \"$!\" > cancel-child-pid; wait"
    )
    environment._local_command_guardrail_reason = lambda *_args: ""  # type: ignore[method-assign]

    def process_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    async def run_cancelled() -> None:
        task = asyncio.create_task(environment.exec(command))
        child_pid: int | None = None
        for _ in range(500):
            if child_ready.exists() and child_pid_path.exists():
                child_pid = int(child_pid_path.read_text(encoding="ascii"))
                break
            if task.done():
                pytest.fail(f"command exited before cancellation: {task.result()}")
            await asyncio.sleep(0.01)
        assert child_pid is not None, "the background descendant did not start before cancellation"

        try:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            for _ in range(200):
                if not process_exists(child_pid):
                    break
                await asyncio.sleep(0.01)
            assert not process_exists(child_pid), "the background descendant survived command cancellation"
        finally:
            if process_exists(child_pid):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child_pid, signal.SIGKILL)

    asyncio.run(run_cancelled())

    assert not marker.exists(), "a background descendant wrote after command cancellation"


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_process_tree_cleanup_is_bounded_and_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.harbor import local_environment

    signals: list[signal.Signals] = []
    monkeypatch.setattr(local_environment, "_REAP_TERM_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_KILL_SECONDS", 0.01)
    monkeypatch.setattr(local_environment.os, "killpg", lambda _pid, value: signals.append(value))

    class FakeProcess:
        pid = 4242
        returncode = None

    async def run_cleanup() -> tuple[bytes, bytes]:
        communication: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_running_loop().create_future()
        return await asyncio.wait_for(
            SkillEvaluatorLocalEnvironment._terminate_process_tree(FakeProcess(), communication),  # type: ignore[arg-type]
            timeout=0.2,
        )

    assert asyncio.run(run_cleanup()) == (b"", b"")
    assert signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_process_tree_cleanup_stays_bounded_when_communication_ignores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    monkeypatch.setattr(local_environment, "_REAP_TERM_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_KILL_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_CANCEL_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(local_environment.os, "killpg", lambda *_args: None)

    async def run_cleanup() -> bool:
        started = asyncio.Event()
        release = asyncio.Event()

        async def stubborn_communication() -> tuple[bytes, bytes]:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            return b"", b""

        communication = asyncio.create_task(stubborn_communication())
        await started.wait()

        class FakeProcess:
            pid = 4242
            returncode = None

        cleanup = asyncio.create_task(
            SkillEvaluatorLocalEnvironment._terminate_process_tree(  # type: ignore[arg-type]
                FakeProcess(),
                communication,
            )
        )
        done, _pending = await asyncio.wait({cleanup}, timeout=0.1)
        finished_within_bound = cleanup in done
        release.set()
        await communication
        await cleanup
        return finished_within_bound

    assert asyncio.run(run_cleanup()) is True


def test_stop_reaps_all_tracked_processes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)
    first = object()
    second = object()
    environment._active_processes.update({first: None, second: None})  # type: ignore[dict-item]
    reaped: list[object] = []

    async def reap(proc: object, _communication: object = None) -> tuple[bytes, bytes]:
        reaped.append(proc)
        return b"", b""

    monkeypatch.setattr(environment, "_terminate_process_tree", reap)

    asyncio.run(environment.stop(delete=False))

    assert set(reaped) == {first, second}
    assert environment._active_processes == {}


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_exec_cancellation_during_process_creation_terminates_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _local_environment(tmp_path)
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def run_cancelled_during_create() -> None:
        process_created = asyncio.Event()
        release_process = asyncio.Event()
        created: list[asyncio.subprocess.Process] = []

        async def delayed_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            proc = await create_subprocess_exec(*args, **kwargs)
            created.append(proc)
            process_created.set()
            await release_process.wait()
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_create)
        task = asyncio.create_task(environment.exec("sleep 30"))
        await asyncio.wait_for(process_created.wait(), timeout=5)
        task.cancel()
        release_process.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(100):
                if created[0].returncode is not None:
                    break
                await asyncio.sleep(0.01)
            assert created[0].returncode is not None, "spawned launcher survived cancellation during process creation"
        finally:
            if created and created[0].returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(created[0].pid, signal.SIGKILL)
                await created[0].communicate()

    asyncio.run(run_cancelled_during_create())


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_repeated_cancellation_during_process_creation_still_reaps_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _local_environment(tmp_path)
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def run_repeated_cancellation() -> None:
        process_created = asyncio.Event()
        release_process = asyncio.Event()
        created: list[asyncio.subprocess.Process] = []

        async def delayed_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            proc = await create_subprocess_exec(*args, **kwargs)
            created.append(proc)
            process_created.set()
            await release_process.wait()
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_create)
        task = asyncio.create_task(environment.exec("sleep 30"))
        await asyncio.wait_for(process_created.wait(), timeout=5)
        task.cancel()
        # Let exec enter its cancellation handler and start waiting for the
        # still-running creation task before delivering a second cancellation.
        await asyncio.sleep(0)
        task.cancel()
        release_process.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            for _ in range(100):
                if created[0].returncode is not None:
                    break
                await asyncio.sleep(0.01)
            assert created[0].returncode is not None, "repeated cancellation orphaned the spawned launcher"
        finally:
            if created and created[0].returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(created[0].pid, signal.SIGKILL)
                await created[0].communicate()

    asyncio.run(run_repeated_cancellation())


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_process_tree_cleanup_finishes_after_repeated_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.harbor import local_environment

    monkeypatch.setattr(local_environment, "_REAP_TERM_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_KILL_SECONDS", 0.01)

    async def run_cleanup() -> list[signal.Signals]:
        signals: list[signal.Signals] = []
        term_sent = asyncio.Event()
        communication: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_running_loop().create_future()

        def killpg(_pid: int, value: signal.Signals) -> None:
            signals.append(value)
            if value == signal.SIGTERM:
                term_sent.set()
            elif not communication.done():
                communication.set_result((b"", b""))

        monkeypatch.setattr(local_environment.os, "killpg", killpg)

        class FakeProcess:
            pid = 4242
            returncode = None

        task = asyncio.create_task(
            SkillEvaluatorLocalEnvironment._terminate_process_tree(FakeProcess(), communication)  # type: ignore[arg-type]
        )
        await asyncio.wait_for(term_sent.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        return signals

    assert asyncio.run(run_cleanup()) == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.parametrize(
    "name",
    [
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "PYTHONPATH",
        "PYTHONHOME",
        "BASH_ENV",
        "ENV",
        "ZDOTDIR",
        "RUBYOPT",
        "PERL5OPT",
        "NODE_OPTIONS",
    ],
)
@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_runtime_injection_env_is_blocked_before_launcher(name: str, tmp_path: Path) -> None:
    environment = _local_environment(tmp_path)

    async def unexpected_launch(*_args: object, **_kwargs: object) -> None:
        pytest.fail("loader-controlled task environment reached the host launcher")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(asyncio, "create_subprocess_exec", unexpected_launch)
        result = asyncio.run(environment.exec("true", env={name: "attacker-controlled"}))

    assert result.return_code == 126
    assert name in (result.stderr or "")


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_persistent_runtime_injection_env_is_blocked_before_launcher(tmp_path: Path) -> None:
    environment = _local_environment(tmp_path, persistent_env={"NODE_OPTIONS": "--require=/tmp/attack.js"})

    result = asyncio.run(environment.exec("true"))

    assert result.return_code == 126
    assert "NODE_OPTIONS" in (result.stderr or "")


def test_validate_local_agents_rejects_unsupported() -> None:
    assert validate_local_agents(["opencode", "gemini-cli", "aider"]) == ["aider", "gemini-cli"]
    assert validate_local_agents(["claude-code", "codex", "opencode"]) == []


def test_ensure_local_runtimes_reports_vendor_hint_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the CLI to look absent (it's installed on dev hosts) and confirm the
    # hint points to a vendor-supported install command.
    from skillevaluator.tier3.harbor import local_runtime

    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: None)
    errors = ensure_local_runtimes(["opencode"])
    assert len(errors) == 1
    assert "opencode" in errors[0]
    assert "npm install" in errors[0] or "brew install" in errors[0]


def test_ensure_local_runtimes_rejects_installed_cli_with_failing_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: "/fake/codex")
    monkeypatch.setattr(
        local_runtime,
        "subprocess",
        SimpleNamespace(
            run=lambda *args, **_kwargs: subprocess.CompletedProcess(args[0], 17, stdout="", stderr="broken")
        ),
        raising=False,
    )

    errors = ensure_local_runtimes(["codex"])

    assert len(errors) == 1
    assert "codex" in errors[0]
    assert "/fake/codex" in errors[0]
    assert "--version" in errors[0]
    assert "exit 17" in errors[0]
    assert "npm install" in errors[0] or "brew install" in errors[0]


def test_ensure_local_runtimes_rejects_installed_cli_when_version_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: "/fake/opencode")

    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(local_runtime, "subprocess", SimpleNamespace(run=time_out), raising=False)

    errors = ensure_local_runtimes(["opencode"])

    assert len(errors) == 1
    assert "opencode" in errors[0]
    assert "--version" in errors[0]
    assert "timed out" in errors[0]
    assert "npm install" in errors[0] or "brew install" in errors[0]


def test_ensure_local_runtimes_reports_sandbox_wrap_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime, local_sandbox

    command = tmp_path / "bin" / "opencode"
    command.parent.mkdir()
    command.write_text("binary", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: str(command))

    class RejectingSandbox:
        def wrap(self, *_args, **_kwargs):
            raise local_sandbox.SandboxUnavailable("cannot determine existing host HOME roots")

    errors = ensure_local_runtimes(["opencode"], sandbox=RejectingSandbox())

    assert len(errors) == 1
    assert "opencode" in errors[0]
    assert "sandboxed --version" in errors[0]
    assert "cannot determine existing host HOME roots" in errors[0]
    assert "--env-mode docker" in errors[0]


def test_ensure_local_runtimes_probes_with_effective_strict_read_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    command = tmp_path / "bin" / "opencode"
    command.parent.mkdir()
    command.write_text("binary", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: str(command))
    monkeypatch.setattr(local_runtime, "runtime_command_roots", lambda *_args, **_kwargs: [command])
    monkeypatch.setattr(
        local_runtime,
        "subprocess",
        SimpleNamespace(run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, stdout="1.0", stderr="")),
        raising=False,
    )
    captured: dict[str, object] = {}

    class CaptureSandbox:
        def wrap(self, argv, **kwargs):
            captured.update(kwargs)
            return argv

    assert ensure_local_runtimes(["opencode"], sandbox=CaptureSandbox(), strict_reads=True) == []
    assert captured["strict_reads"] is True


def test_ensure_local_runtimes_reports_disappearing_runtime_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    command = tmp_path / "bin" / "opencode"
    command.parent.mkdir()
    command.write_text("binary", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: str(command))

    def disappearing_roots(*_args, **_kwargs):
        raise FileNotFoundError("selected runtime disappeared before sandbox preparation")

    monkeypatch.setattr(local_runtime, "runtime_command_roots", disappearing_roots)

    class UnusedSandbox:
        def wrap(self, *_args, **_kwargs):
            raise AssertionError("wrap must not run after read-root discovery fails")

    errors = ensure_local_runtimes(["opencode"], sandbox=UnusedSandbox())

    assert len(errors) == 1
    assert "sandboxed --version" in errors[0]
    assert "FileNotFoundError" in errors[0]
    assert "selected runtime disappeared" in errors[0]
    assert "--env-mode docker" in errors[0]


def test_ensure_local_runtimes_reports_runtime_command_symlink_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    command = tmp_path / "bin" / "opencode"
    command.parent.mkdir()
    loop_target = command.parent / "opencode-loop"
    command.symlink_to(loop_target.name)
    loop_target.symlink_to(command.name)
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: str(command))

    class UnusedSandbox:
        def wrap(self, *_args, **_kwargs):
            raise AssertionError("wrap must not run after runtime path resolution fails")

    errors = ensure_local_runtimes(["opencode"], sandbox=UnusedSandbox())

    assert len(errors) == 1
    assert "sandboxed --version" in errors[0]
    assert "RuntimePathResolutionError" in errors[0]
    assert "symlink" in errors[0].lower()
    assert "--env-mode docker" in errors[0]


def test_ensure_local_runtimes_reports_shebang_interpreter_symlink_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    interpreter = tmp_path / "interpreters" / "python3"
    interpreter.parent.mkdir()
    loop_target = interpreter.parent / "python3-loop"
    interpreter.symlink_to(loop_target.name)
    loop_target.symlink_to(interpreter.name)
    command = tmp_path / "bin" / "opencode"
    command.parent.mkdir()
    command.write_text(f"#!{interpreter}\n", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: str(command))

    class UnusedSandbox:
        def wrap(self, *_args, **_kwargs):
            raise AssertionError("wrap must not run after interpreter path resolution fails")

    errors = ensure_local_runtimes(["opencode"], sandbox=UnusedSandbox())

    assert len(errors) == 1
    assert "sandboxed --version" in errors[0]
    assert "RuntimePathResolutionError" in errors[0]
    assert str(interpreter) in errors[0]
    assert "--env-mode docker" in errors[0]


@pytest.mark.parametrize(
    "error_type",
    [TypeError, AssertionError, RuntimeError, ValueError, OSError, subprocess.SubprocessError],
)
def test_ensure_local_runtimes_preserves_sandbox_programming_errors(
    error_type: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    command = tmp_path / "bin" / "opencode"
    command.parent.mkdir()
    command.write_text("binary", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: str(command))

    class BrokenSandbox:
        def wrap(self, *_args, **_kwargs):
            raise error_type("programming defect")

    with pytest.raises(error_type, match="programming defect"):
        ensure_local_runtimes(["opencode"], sandbox=BrokenSandbox())


@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, GeneratorExit])
def test_ensure_local_runtimes_preserves_sandbox_control_flow_signals(
    signal_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    command = tmp_path / "bin" / "opencode"
    command.parent.mkdir()
    command.write_text("binary", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: str(command))

    class InterruptingSandbox:
        def wrap(self, *_args, **_kwargs):
            raise signal_type()

    with pytest.raises(signal_type):
        ensure_local_runtimes(["opencode"], sandbox=InterruptingSandbox())


def test_ensure_local_runtimes_accepts_working_version_with_safe_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    captured: dict[str, object] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-probe")
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: "/fake/opencode")

    def capture_run(argv, **kwargs):
        captured.update({"argv": argv, **kwargs})
        probe_env = kwargs["env"]
        assert "OPENAI_API_KEY" not in probe_env
        assert Path(probe_env["HOME"]).is_dir()
        assert Path(probe_env["TMPDIR"]).is_dir()
        assert Path(kwargs["cwd"]).is_relative_to(Path(probe_env["HOME"]).parent)
        return subprocess.CompletedProcess(argv, 0, stdout="opencode 1.2.3\n", stderr="")

    monkeypatch.setattr(local_runtime, "subprocess", SimpleNamespace(run=capture_run), raising=False)

    assert ensure_local_runtimes(["opencode"], env={"PATH": "/custom/bin", "API_TOKEN": "secret"}) == []
    assert captured["argv"] == ["/fake/opencode", "--version"]
    assert isinstance(captured["timeout"], int | float) and captured["timeout"] > 0


def test_local_runtime_install_hint_has_no_internal_url() -> None:
    from skillevaluator.tier3.harbor.local_runtime import local_runtime_install_command

    hint = local_runtime_install_command(["claude-code", "codex", "opencode"])
    assert "npm install" in hint or "brew install" in hint


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_runtime_read_paths_include_symlinked_npm_package_without_sibling_bin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    npm_root = tmp_path / ".npm-global"
    command = npm_root / "lib" / "node_modules" / "opencode-ai" / "bin" / "opencode"
    command.parent.mkdir(parents=True)
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    bin_dir = npm_root / "bin"
    bin_dir.mkdir()
    (bin_dir / "opencode").symlink_to(command)
    monkeypatch.setenv("PATH", str(bin_dir))

    roots = local_runtime.runtime_command_roots(["opencode"], runtime_root=tmp_path / "managed")

    assert (bin_dir / "opencode").absolute() in roots
    assert command.parent.parent.resolve() in roots
    assert bin_dir.resolve() not in roots
    assert npm_root.resolve() not in roots


def test_runtime_read_paths_keep_direct_node_modules_bin_shim_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    command = bin_dir / "opencode"
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    sibling = bin_dir / "unselected-agent"
    sibling.write_text("must remain hidden", encoding="utf-8")
    monkeypatch.setenv("PATH", str(bin_dir))

    roots = local_runtime.runtime_command_roots(["opencode"], runtime_root=tmp_path / "managed")

    assert command.absolute() in roots
    assert bin_dir.resolve() not in roots
    assert sibling.resolve() not in roots


def test_runtime_read_paths_keep_direct_binary_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    command = tmp_path / "tools" / "opencode"
    command.parent.mkdir()
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setenv("PATH", str(command.parent))

    roots = local_runtime.runtime_command_roots(["opencode"], runtime_root=tmp_path / "managed")

    shell_paths = {Path("/bin").resolve() / "sh", Path("/bin/sh").resolve()}
    assert roots[0] == command.absolute()
    assert set(roots[1:]) == shell_paths
    assert command.parent.resolve() not in roots


def test_runtime_read_paths_keep_single_file_symlink_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    target = tmp_path / "tools" / "opencode"
    target.parent.mkdir()
    target.write_text("binary", encoding="utf-8")
    target.chmod(0o755)
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    command = bin_dir / "opencode"
    command.symlink_to(target)
    monkeypatch.setenv("PATH", str(bin_dir))

    roots = local_runtime.runtime_command_roots(["opencode"], runtime_root=tmp_path / "managed")

    assert command.absolute() in roots
    assert target.resolve() in roots
    assert tmp_path.resolve() not in roots
    assert bin_dir.resolve() not in roots
    assert target.parent.resolve() not in roots


def test_runtime_read_paths_preserve_claude_standalone_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    target = tmp_path / ".local" / "share" / "claude" / "versions" / "2.1.198"
    target.parent.mkdir(parents=True)
    target.write_text("binary", encoding="utf-8")
    target.chmod(0o755)
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    command = bin_dir / "claude"
    command.symlink_to(target)
    monkeypatch.setenv("PATH", str(bin_dir))

    roots = local_runtime.runtime_command_roots(["claude-code"], runtime_root=tmp_path / "managed")

    assert roots == [command.absolute(), target.resolve()]
    assert (tmp_path / ".local" / "share" / "claude").resolve() not in roots


def test_runtime_read_paths_resolve_env_shebang_helper_without_sibling_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    command = tmp_path / "agent-bin" / "opencode"
    command.parent.mkdir()
    command.write_text("#!/usr/bin/env helper\n", encoding="utf-8")
    command.chmod(0o755)
    helper = tmp_path / "helper-bin" / "helper"
    helper.parent.mkdir()
    helper.write_text("binary", encoding="utf-8")
    helper.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(command.parent), str(helper.parent), os.defpath)))

    roots = local_runtime.runtime_command_roots(["opencode"], runtime_root=tmp_path / "managed")

    assert command.absolute() in roots
    assert Path("/usr/bin/env") in roots
    assert helper.absolute() in roots
    assert command.parent.resolve() not in roots
    assert helper.parent.resolve() not in roots


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_runtime_read_paths_include_selected_homebrew_dependency_kegs_and_shipped_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    prefix = tmp_path / "homebrew"
    node_keg = prefix / "Cellar" / "node" / "25.0.0"
    node = node_keg / "bin" / "node"
    node.parent.mkdir(parents=True)
    node.write_text("binary", encoding="utf-8")
    node.chmod(0o755)
    (node_keg / "INSTALL_RECEIPT.json").write_text(
        json.dumps(
            {
                "runtime_dependencies": [
                    {"full_name": "openssl@3", "pkg_version": "3.0.0"},
                ]
            }
        ),
        encoding="utf-8",
    )

    openssl_keg = prefix / "Cellar" / "openssl@3" / "3.0.0"
    bottled_config = openssl_keg / ".bottle" / "etc" / "openssl@3" / "openssl.cnf"
    bottled_config.parent.mkdir(parents=True)
    bottled_config.write_text("shipped default", encoding="utf-8")
    openssl_opt = prefix / "opt" / "openssl@3"
    openssl_opt.parent.mkdir(parents=True)
    openssl_opt.symlink_to(openssl_keg)
    installed_config = prefix / "etc" / "openssl@3" / "openssl.cnf"
    installed_config.parent.mkdir(parents=True)
    installed_config.write_text("installed default", encoding="utf-8")
    shipped_link = bottled_config.parent / "outside.cnf"
    shipped_link.write_text("shipped default", encoding="utf-8")
    outside_target = tmp_path / "outside-homebrew.txt"
    outside_target.write_text("must remain hidden", encoding="utf-8")
    installed_link = installed_config.parent / "outside.cnf"
    installed_link.symlink_to(outside_target)
    shipped_prefix_link = bottled_config.parent / "prefix-secret.cnf"
    shipped_prefix_link.write_text("shipped default", encoding="utf-8")
    prefix_secret = prefix / "var" / "private-service" / "secret"
    prefix_secret.parent.mkdir(parents=True)
    prefix_secret.write_text("must remain hidden", encoding="utf-8")
    installed_prefix_link = installed_config.parent / "prefix-secret.cnf"
    installed_prefix_link.symlink_to(prefix_secret)
    shipped_etc_link = bottled_config.parent / "other-service.cnf"
    shipped_etc_link.write_text("shipped default", encoding="utf-8")
    other_service_secret = prefix / "etc" / "other-service" / "private.cnf"
    other_service_secret.parent.mkdir(parents=True)
    other_service_secret.write_text("must remain hidden", encoding="utf-8")
    installed_etc_link = installed_config.parent / "other-service.cnf"
    installed_etc_link.symlink_to(other_service_secret)
    unrelated_secret = prefix / "etc" / "openssl@3" / "private" / "host.key"
    unrelated_secret.parent.mkdir()
    unrelated_secret.write_text("must remain hidden", encoding="utf-8")

    helper_bin = prefix / "bin"
    helper_bin.mkdir()
    (helper_bin / "node").symlink_to(node)
    agent_bin = tmp_path / "agent-bin"
    agent_bin.mkdir()
    command = agent_bin / "codex"
    command.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(agent_bin), str(helper_bin), os.defpath)))

    roots = local_runtime.runtime_command_roots(["codex"], runtime_root=tmp_path / "managed")

    assert node_keg.resolve() in roots
    assert openssl_opt.absolute() in roots
    assert openssl_keg.resolve() in roots
    assert installed_config.resolve() in roots
    assert installed_link.absolute() not in roots
    assert outside_target.resolve() not in roots
    assert installed_prefix_link.absolute() not in roots
    assert prefix_secret.resolve() not in roots
    assert installed_etc_link.absolute() not in roots
    assert other_service_secret.resolve() not in roots
    assert unrelated_secret.resolve() not in roots
    assert installed_config.parent.resolve() not in roots
    assert (prefix / "Cellar").resolve() not in roots


def test_homebrew_runtime_roots_ignore_malformed_or_cyclic_receipt_dependencies(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    prefix = tmp_path / "homebrew"
    keg = prefix / "Cellar" / "node" / "25.0.0"
    binary = keg / "bin" / "node"
    binary.parent.mkdir(parents=True)
    binary.write_text("binary", encoding="utf-8")
    (keg / "INSTALL_RECEIPT.json").write_text(
        json.dumps(
            {
                "runtime_dependencies": [
                    {"full_name": "bad\u0000name", "pkg_version": "1"},
                    {"full_name": "cycle", "pkg_version": "1"},
                    {"full_name": "escape", "pkg_version": "1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    cycle = prefix / "opt" / "cycle"
    cycle.parent.mkdir(parents=True)
    cycle.symlink_to(cycle)
    outside_formula = tmp_path / "outside-formula"
    outside_keg = outside_formula / "1"
    outside_keg.mkdir(parents=True)
    cellar_formula = prefix / "Cellar" / "escape"
    cellar_formula.symlink_to(outside_formula)
    (prefix / "opt" / "escape").symlink_to(outside_keg)

    assert local_runtime._homebrew_runtime_roots(binary) == [keg.resolve()]


def test_homebrew_shipped_config_paths_reject_symlinked_layout_roots(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    prefix = tmp_path / "homebrew"
    keg = prefix / "Cellar" / "node" / "25.0.0"
    shipped = keg / ".bottle" / "etc" / "node" / "runtime.conf"
    shipped.parent.mkdir(parents=True)
    shipped.write_text("default", encoding="utf-8")
    outside_etc = tmp_path / "outside-etc"
    installed = outside_etc / "node" / "runtime.conf"
    installed.parent.mkdir(parents=True)
    installed.write_text("secret", encoding="utf-8")
    (prefix / "etc").symlink_to(outside_etc)

    assert local_runtime._homebrew_shipped_config_paths(prefix, keg) == []

    other_prefix = tmp_path / "other-homebrew"
    other_keg = other_prefix / "Cellar" / "node" / "25.0.0"
    outside_bottle = tmp_path / "outside-bottle"
    outside_shipped = outside_bottle / "node" / "runtime.conf"
    outside_shipped.parent.mkdir(parents=True)
    outside_shipped.write_text("default", encoding="utf-8")
    (other_keg / ".bottle").mkdir(parents=True)
    (other_keg / ".bottle" / "etc").symlink_to(outside_bottle)
    other_installed = other_prefix / "etc" / "node" / "runtime.conf"
    other_installed.parent.mkdir(parents=True)
    other_installed.write_text("secret", encoding="utf-8")

    assert local_runtime._homebrew_shipped_config_paths(other_prefix, other_keg) == []


def test_find_runtime_command_does_not_search_other_managed_agent_bins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    collision = tmp_path / "runtimes" / "opencode" / "bin" / "codex"
    collision.parent.mkdir(parents=True)
    collision.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    collision.chmod(0o755)
    monkeypatch.setenv("PATH", os.defpath)

    assert local_runtime.find_runtime_command("codex", runtime_root=tmp_path / "runtimes") is None


def test_version_probe_executes_canonical_parent_when_path_directory_is_symlinked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    actual_bin = tmp_path / "actual-bin"
    actual_bin.mkdir()
    command = actual_bin / "opencode"
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    path_alias = tmp_path / "path-alias"
    path_alias.symlink_to(actual_bin, target_is_directory=True)
    monkeypatch.setenv("PATH", str(path_alias))
    captured: dict[str, object] = {}

    def capture_run(argv, **_kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="opencode 1.0\n", stderr="")

    monkeypatch.setattr(local_runtime, "subprocess", SimpleNamespace(run=capture_run))

    assert ensure_local_runtimes(["opencode"], runtime_root=tmp_path / "managed") == []
    assert captured["argv"] == [str(command), "--version"]


def test_local_subprocess_env_excludes_sibling_managed_agent_bins(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    runtime_root = tmp_path / "runtimes"
    codex_bin = runtime_root / "codex" / "bin"
    opencode_bin = runtime_root / "opencode" / "bin"
    codex_bin.mkdir(parents=True)
    opencode_bin.mkdir(parents=True)
    inherited = os.pathsep.join((str(opencode_bin), str(codex_bin), os.defpath))

    env = local_runtime.local_subprocess_env(
        runtime_root=runtime_root,
        runtime_agents=["codex"],
        base_env={"PATH": inherited},
    )
    path_parts = env["PATH"].split(os.pathsep)

    assert str(codex_bin) in path_parts
    assert str(opencode_bin) not in path_parts


def test_runtime_path_excludes_symlink_alias_to_sibling_managed_bin(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    runtime_root = tmp_path / "runtimes"
    selected_bin = runtime_root / "codex" / "bin"
    sibling_bin = runtime_root / "opencode" / "bin"
    selected_bin.mkdir(parents=True)
    sibling_bin.mkdir(parents=True)
    collision = sibling_bin / "codex"
    collision.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    collision.chmod(0o755)
    sibling_alias = tmp_path / "sibling-alias"
    sibling_alias.symlink_to(sibling_bin, target_is_directory=True)
    host_bin = tmp_path / "host-bin"
    host_bin.mkdir()

    path = local_runtime.runtime_path(
        runtime_root,
        os.pathsep.join((str(sibling_alias), str(host_bin))),
        agents=["codex"],
    )
    path_parts = path.split(os.pathsep)

    assert path_parts[0] == str(selected_bin)
    assert str(sibling_alias) not in path_parts
    assert str(host_bin) in path_parts
    assert local_runtime.shutil.which("codex", path=path) is None


def test_runtime_path_canonicalizes_selected_bin_below_symlinked_runtime_root(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    real_runtime_root = tmp_path / "real-runtimes"
    selected_bin = real_runtime_root / "codex" / "bin"
    selected_bin.mkdir(parents=True)
    command = selected_bin / "codex"
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    runtime_alias = tmp_path / "runtime-alias"
    runtime_alias.symlink_to(real_runtime_root, target_is_directory=True)

    path = local_runtime.runtime_path(runtime_alias, "", agents=["codex"])
    path_parts = path.split(os.pathsep)

    assert path_parts[0] == str(selected_bin.resolve())
    assert str(runtime_alias / "codex" / "bin") not in path_parts
    assert local_runtime.shutil.which("codex", path=path) == str(command.resolve())


def test_runtime_path_canonicalizes_normal_host_bin_alias_and_preserves_unresolved(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    host_bin = tmp_path / "versions" / "current" / "bin"
    host_bin.mkdir(parents=True)
    helper = host_bin / "helper"
    helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    helper.chmod(0o755)
    host_alias = tmp_path / "host-bin-alias"
    host_alias.symlink_to(host_bin, target_is_directory=True)
    unresolved = tmp_path / "not-installed-yet"

    path = local_runtime.runtime_path(
        tmp_path / "managed",
        os.pathsep.join((str(host_alias), str(host_bin), str(unresolved), str(host_alias))),
        agents=["codex"],
    )
    path_parts = path.split(os.pathsep)

    assert path_parts == [str(host_bin.resolve()), str(unresolved)]
    assert local_runtime.shutil.which("helper", path=path) == str(helper.resolve())


def test_runtime_path_drops_non_absolute_inherited_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime

    workspace = tmp_path / "workspace"
    relative_bin = workspace / "evilbin"
    relative_bin.mkdir(parents=True)
    absolute_bin = tmp_path / "absolute-bin"
    absolute_bin.mkdir()
    absolute_file = tmp_path / "not-a-bin-directory"
    absolute_file.write_text("not a PATH directory", encoding="utf-8")
    unresolved_absolute = tmp_path / "not-installed-yet"
    monkeypatch.chdir(workspace)

    path = local_runtime.runtime_path(
        tmp_path / "managed",
        os.pathsep.join(
            (
                str(absolute_bin),
                "evilbin",
                ".",
                "..",
                "$HOME/bin",
                "relative/missing",
                "",
                str(absolute_file),
                str(unresolved_absolute),
            )
        ),
        agents=["codex"],
    )

    assert path.split(os.pathsep) == [str(absolute_bin.resolve()), str(unresolved_absolute)]


@pytest.mark.parametrize("runtime_agent", [None, "aider"])
def test_local_environment_rejects_missing_or_unsupported_runtime_agent(
    monkeypatch: pytest.MonkeyPatch,
    runtime_agent: str | None,
) -> None:
    monkeypatch.setattr(BaseEnvironment, "__init__", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="runtime_agent"):
        SkillEvaluatorLocalEnvironment(runtime_agent=runtime_agent)


def test_local_environment_requires_runtime_agent_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BaseEnvironment, "__init__", lambda *_args, **_kwargs: None)

    with pytest.raises(TypeError, match="runtime_agent"):
        SkillEvaluatorLocalEnvironment()


def test_runtime_ro_binds_include_only_selected_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    runtime_root = tmp_path / "runtimes"
    selected_root = runtime_root / "codex"
    sibling_root = runtime_root / "opencode"
    selected_root.mkdir(parents=True)
    sibling_root.mkdir()
    captured: list[tuple[str, ...]] = []

    def capture_runtime_roots(agents, **_kwargs):
        captured.append(tuple(agents))
        return []

    monkeypatch.setattr(local_environment, "runtime_command_roots", capture_runtime_roots)
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._runtime_root = runtime_root
    environment._runtime_agent = "codex"

    binds = environment._runtime_ro_binds()

    assert captured == [("codex",)]
    assert selected_root.resolve() not in binds
    assert sibling_root.resolve() not in binds
    assert runtime_root.resolve() not in binds


def test_runtime_ro_binds_preserve_exact_selected_agent_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    target = tmp_path / "tools" / "opencode"
    target.parent.mkdir()
    target.write_text("binary", encoding="utf-8")
    target.chmod(0o755)
    command = tmp_path / ".local" / "bin" / "opencode"
    command.parent.mkdir(parents=True)
    command.symlink_to(target)
    monkeypatch.setattr(
        local_environment,
        "runtime_command_roots",
        lambda *_args, **_kwargs: [command.absolute(), target.resolve()],
    )
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._runtime_root = tmp_path / "managed"
    environment._runtime_agent = "opencode"

    binds = environment._runtime_ro_binds()

    assert command.absolute() in binds
    assert target.resolve() in binds
    assert command.parent.resolve() not in binds


def test_strict_runtime_ro_binds_use_exact_interpreter_and_python_library_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sysconfig

    from skillevaluator.tier3.harbor import local_environment

    target = tmp_path / "python-install" / "bin" / "python3.12"
    target.parent.mkdir(parents=True)
    target.write_text("binary", encoding="utf-8")
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(target)
    library_roots = {
        name: tmp_path / "python-install" / "lib" / name for name in ("stdlib", "platstdlib", "purelib", "platlib")
    }
    for path in library_roots.values():
        path.mkdir(parents=True)

    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sys, "exec_prefix", "/usr/local")
    monkeypatch.setattr(sys, "base_prefix", "/opt")
    monkeypatch.setattr(sys, "base_exec_prefix", "/usr")
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {name: str(path) for name, path in library_roots.items()})
    monkeypatch.setattr(sysconfig, "get_config_var", lambda _name: None)
    monkeypatch.setattr(local_environment, "runtime_command_roots", lambda *_args, **_kwargs: [])
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._runtime_root = tmp_path / "managed"
    environment._runtime_agent = "opencode"
    environment._strict_reads = True

    binds = environment._runtime_ro_binds()

    assert interpreter.absolute() in binds
    assert target.resolve() in binds
    assert set(library_roots.values()).issubset(binds)
    assert interpreter.parent.resolve() not in binds
    assert Path(sys.prefix).resolve() not in binds
    assert Path(sys.exec_prefix).resolve() not in binds
    assert Path(sys.base_prefix).resolve() not in binds
    assert Path(sys.base_exec_prefix).resolve() not in binds


def test_non_strict_runtime_ro_binds_keep_prefix_and_interpreter_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    prefix = tmp_path / "venv"
    interpreter = prefix / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("binary", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "exec_prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(prefix))
    monkeypatch.setattr(sys, "base_exec_prefix", str(prefix))
    monkeypatch.setattr(local_environment, "runtime_command_roots", lambda *_args, **_kwargs: [])
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._runtime_root = tmp_path / "managed"
    environment._runtime_agent = "opencode"
    environment._strict_reads = False

    binds = environment._runtime_ro_binds()

    assert prefix.resolve() in binds
    assert interpreter.parent.resolve() in binds


def test_strict_exec_filters_broad_prefixes_and_publishes_selected_npm_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sysconfig

    from skillevaluator.tier3.harbor import local_runtime

    package = tmp_path / "host" / "usr" / "local" / "lib" / "node_modules" / "opencode-ai"
    target = package / "bin" / "opencode"
    target.parent.mkdir(parents=True)
    target.write_text("binary", encoding="utf-8")
    command = tmp_path / "host" / "usr" / "local" / "bin" / "opencode"
    command.parent.mkdir(parents=True)
    command.symlink_to(Path("../lib/node_modules/opencode-ai/bin/opencode"))
    narrow_site_packages = tmp_path / "host" / "opt" / "venv" / "lib" / "python" / "site-packages"
    narrow_site_packages.mkdir(parents=True)

    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda *_args, **_kwargs: str(command))
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "exec_prefix", "/usr/local")
    monkeypatch.setattr(sys, "base_prefix", "/opt")
    monkeypatch.setattr(sys, "base_exec_prefix", "/tmp")
    monkeypatch.setattr(
        sysconfig,
        "get_paths",
        lambda: {
            "stdlib": str(narrow_site_packages.parent / "stdlib"),
            "platstdlib": str(narrow_site_packages.parent / "platstdlib"),
            "purelib": str(narrow_site_packages),
            "platlib": str(narrow_site_packages),
        },
    )
    monkeypatch.setattr(sysconfig, "get_config_var", lambda _name: None)
    (narrow_site_packages.parent / "stdlib").mkdir()
    (narrow_site_packages.parent / "platstdlib").mkdir()
    monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: (Path.home().resolve(),))

    environment = _local_environment(tmp_path)
    environment._strict_reads = True
    captured: dict[str, object] = {}
    real_sandbox = local_sandbox.Sandbox(local_sandbox.SandboxPlan("bubblewrap", "kernel", "capture"))

    class CaptureSandbox:
        plan = real_sandbox.plan

        @staticmethod
        def wrap(argv: list[str], **kwargs: object) -> list[str]:
            captured["extra_ro"] = kwargs["extra_ro"]
            captured["deny_reads"] = kwargs["deny_reads"]
            captured["wrapped"] = real_sandbox.wrap(argv, **kwargs)  # type: ignore[arg-type]
            return argv

    environment._sandbox = CaptureSandbox()

    result = asyncio.run(environment.exec("printf ok"))

    assert result.return_code == 0, result.stderr
    assert result.stdout == "ok"
    wrapped = captured["wrapped"]
    extra_ro = captured["extra_ro"]
    assert isinstance(wrapped, list)
    assert isinstance(extra_ro, list)
    assert captured["deny_reads"] == (Path(os.environ["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"]),)
    ro_binds = {tuple(wrapped[index + 1 : index + 3]) for index, value in enumerate(wrapped) if value == "--ro-bind"}
    symlinks = {tuple(wrapped[index + 1 : index + 3]) for index, value in enumerate(wrapped) if value == "--symlink"}
    broad_roots = {"/", "/usr", "/usr/local", "/opt", "/tmp", "/private/tmp", "/var/tmp", str(Path.home())}
    assert not any(destination in broad_roots for _source, destination in ro_binds | symlinks)
    assert (str(target.resolve()), str(command.absolute())) in symlinks
    assert (str(package.resolve()), str(package.resolve())) in ro_binds
    assert (str(narrow_site_packages.resolve()), str(narrow_site_packages.resolve())) in ro_binds
    assert Path(sys.executable).parent.resolve() not in extra_ro


def test_runtime_ro_binds_do_not_expose_whole_managed_agent_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    managed_root = tmp_path / "managed" / "codex"
    command = managed_root / "bin" / "codex"
    command.parent.mkdir(parents=True)
    command.write_text("binary", encoding="utf-8")
    command.chmod(0o755)
    sibling = managed_root / "sibling-auth.txt"
    sibling.write_text("DENY", encoding="utf-8")
    monkeypatch.setattr(
        local_environment,
        "runtime_command_roots",
        lambda *_args, **_kwargs: [command.resolve()],
    )
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._runtime_root = tmp_path / "managed"
    environment._runtime_agent = "codex"

    binds = environment._runtime_ro_binds()

    assert command.resolve() in binds
    assert managed_root.resolve() not in binds
    assert sibling.resolve() not in binds


def test_evaluator_python_path_uses_only_selected_runtime_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    captured: dict[str, object] = {}

    def capture_bins(runtime_root, *, agents=None):
        captured.update({"runtime_root": runtime_root, "agents": agents})
        return []

    monkeypatch.setattr(local_environment, "runtime_bin_dirs", capture_bins)
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._runtime_root = tmp_path / "managed"
    environment._runtime_agent = "codex"

    environment._path_with_evaluator_python(os.defpath)

    assert captured["agents"] == ["codex"]


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_shell_path_rewrite_quotes_unquoted_path_with_spaces(tmp_path: Path) -> None:
    target = tmp_path / "result with spaces"
    target.mkdir()
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/logs", target)]

    rewritten = environment._rewrite_command("printf ok > /logs/output.txt")
    result = subprocess.run(["bash", "-c", rewritten], capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stderr
    assert (target / "output.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_shell_path_rewrite_preserves_existing_quotes(tmp_path: Path) -> None:
    target = tmp_path / "result with spaces"
    target.mkdir()
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/logs", target)]

    rewritten = environment._rewrite_command('printf ok > "/logs/output.txt"')
    result = subprocess.run(["bash", "-c", rewritten], capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stderr
    assert (target / "output.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_shell_path_rewrite_handles_exact_path_before_shell_separator(tmp_path: Path) -> None:
    target = tmp_path / "result with spaces"
    target.mkdir()
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/workspace", target)]

    rewritten = environment._rewrite_command("cd /workspace && printf ok > output.txt")
    result = subprocess.run(["bash", "-c", rewritten], capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stderr
    assert (target / "output.txt").read_text(encoding="utf-8") == "ok"


def test_rewrite_env_values_does_not_add_shell_quotes(tmp_path: Path) -> None:
    target = tmp_path / "result with spaces"
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/logs", target)]

    assert environment._rewrite_env_values({"OUTPUT": "/logs/output.txt"}) == {"OUTPUT": str(target / "output.txt")}


def test_local_opencode_confines_project_discovery_to_the_run_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(SkillEvaluatorLocalOpenCode)
    agent.model_name = "nvidia/openai/gpt-oss-120b"
    agent.mcp_servers = []
    agent._opencode_config = {}
    agent.render_instruction = lambda instruction: instruction
    agent._build_register_skills_command = lambda: "register-skills"
    agent.build_cli_flags = lambda: ""
    calls: list[tuple[str, dict[str, str]]] = []

    async def capture_exec(
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> None:
        calls.append((command, dict(env or {})))

    agent.exec_as_agent = capture_exec
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")

    asyncio.run(agent.run("test instruction", object(), None))

    assert calls[0][0] == "git -C /workspace init -q"
    assert all(env["OPENCODE_TEST_HOME"] == "/workspace" for _, env in calls)
    assert all(env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1" for _, env in calls)


def test_local_opencode_confinement_is_provider_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = object.__new__(SkillEvaluatorLocalOpenCode)
    agent.model_name = "openai/gpt-4.1-mini"
    calls: list[tuple[str, dict[str, str]]] = []

    async def fake_upstream_run(
        self: OpenCode,
        instruction: str,
        environment: object,
        context: object,
    ) -> None:
        _ = (instruction, context)
        await self.exec_as_agent(environment, command="opencode run", env={"OPENAI_API_KEY": "test-key"})

    async def capture_parent_exec(
        _self: OpenCode,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> None:
        calls.append((command, dict(env or {})))

    monkeypatch.setattr(OpenCode, "run", fake_upstream_run)
    monkeypatch.setattr(OpenCode, "exec_as_agent", capture_parent_exec)

    asyncio.run(agent.run("test instruction", object(), None))

    assert calls[0][0] == "git -C /workspace init -q"
    assert calls[1][0] == "opencode run"
    assert all(env["OPENCODE_FAKE_VCS"] == "git" for _, env in calls)
    assert all(env["OPENCODE_TEST_HOME"] == "/workspace" for _, env in calls)
    assert all(env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1" for _, env in calls)


def test_doctor_prerequisite_check_receives_selected_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import commands as tier3_commands

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        tier3_commands,
        "resolve_llm_provider",
        lambda: _provider("openai", api_key="key", base_url="https://api.openai.com/v1"),
    )
    monkeypatch.setattr(
        tier3_commands,
        "_check_prerequisites",
        lambda **kwargs: captured.update(kwargs) or [],
    )

    assert tier3_commands.doctor(agents="opencode", env_mode="local") == 0
    assert captured["agents"] == ["opencode"]


def test_local_prerequisite_probes_runtime_inside_detected_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.harbor import local_runtime, local_sandbox

    sandbox = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(local_sandbox, "detect", lambda _mode: sandbox)
    monkeypatch.setenv("SKILLEVALUATOR_LOCAL_STRICT_READS", "1")
    monkeypatch.setattr(
        local_runtime,
        "ensure_local_runtimes",
        lambda agents, **kwargs: captured.update({"agents": agents, **kwargs}) or [],
    )

    assert _check_prerequisites(env_mode="local", agents=["opencode"]) == []
    assert captured["agents"] == ["opencode"]
    assert captured["sandbox"] is sandbox
    assert captured["strict_reads"] is True


@pytest.mark.parametrize("mode", local_sandbox.SANDBOX_MODES)
def test_local_prerequisite_rejects_native_windows_for_every_sandbox_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime, runner

    monkeypatch.setattr(runner, "_harbor_bin", lambda: "/fake/harbor")
    monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Windows")
    monkeypatch.setenv(local_sandbox.SANDBOX_MODE_ENV, mode)
    monkeypatch.setattr(
        local_runtime,
        "ensure_local_runtimes",
        lambda *_args, **_kwargs: pytest.fail("native Windows must fail before runtime probing"),
    )

    errors = _check_prerequisites(env_mode="local", agents=["opencode"])

    assert len(errors) == 1
    assert "Native Windows local mode is unsupported" in errors[0]
    assert "WSL2" in errors[0]
    assert "--env-mode docker" in errors[0]


def test_run_harbor_eval_rejects_native_windows_before_provider_or_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import runner

    monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        runner,
        "resolve_llm_provider",
        lambda: pytest.fail("native Windows must fail before provider resolution"),
    )
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: pytest.fail("native Windows must fail before config loading"),
    )

    result = runner.run_harbor_eval(tmp_path, ["opencode"], env_mode="local")

    assert result == {
        "error": [
            "Native Windows local mode is unsupported, including with "
            "SKILLEVALUATOR_LOCAL_SANDBOX=prefer or off. "
            "Use WSL2 for Linux local mode or --env-mode docker."
        ]
    }


def test_local_prerequisite_reports_missing_host_home_as_readiness_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_runtime, local_sandbox, runner

    command = tmp_path / "bin" / "opencode"
    command.parent.mkdir()
    command.write_text("binary", encoding="utf-8")
    command.chmod(0o755)
    sandbox = local_sandbox.Sandbox(local_sandbox.SandboxPlan("seatbelt", "kernel-macos", "test"))
    monkeypatch.setattr(runner, "_harbor_bin", lambda: "/fake/harbor")
    monkeypatch.setattr(local_sandbox, "detect", lambda _mode: sandbox)
    monkeypatch.setattr(local_sandbox, "pwd", None)
    monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local_runtime, "find_runtime_command", lambda _agent, **_: str(command))
    monkeypatch.delenv("HOME", raising=False)

    errors = _check_prerequisites(env_mode="local", agents=["opencode"])

    assert len(errors) == 1
    assert "opencode" in errors[0]
    assert "sandboxed --version" in errors[0]
    assert "host HOME" in errors[0]


def test_doctor_uses_local_build_bridge_for_mixed_agents(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from skillevaluator.tier3 import commands as tier3_commands

    monkeypatch.setattr(
        tier3_commands,
        "resolve_llm_provider",
        lambda: _provider(
            "nv_build",
            api_key="nvidia-build-key",
            base_url="https://integrate.api.nvidia.com/v1",
        ),
    )
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setenv("OPENAI_API_KEY", "independent-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert tier3_commands.doctor(agents="codex,opencode", env_mode="local") == 0
    output = capsys.readouterr().out
    assert "Agent runtime credential" in output
    assert "Codex runtime credential" not in output
    assert "agent container" not in output


def test_doctor_accepts_nvidia_only_credentials_for_local_claude_bridge(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from skillevaluator.tier3 import commands as tier3_commands

    monkeypatch.setattr(
        tier3_commands,
        "resolve_llm_provider",
        lambda: _provider(
            "nv_build",
            api_key="nvidia-build-key",
            base_url="https://integrate.api.nvidia.com/v1",
        ),
    )
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert tier3_commands.doctor(agents="claude-code", env_mode="local") == 0
    assert "runtime credential" in capsys.readouterr().out


def test_doctor_local_codex_ignores_native_credentials_and_accepts_explicit_build_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from skillevaluator.tier3 import commands as tier3_commands

    monkeypatch.setattr(
        tier3_commands,
        "resolve_llm_provider",
        lambda: _provider(
            "nv_build",
            api_key="nvidia-build-key",
            base_url="https://integrate.api.nvidia.com/v1",
        ),
    )
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setenv("OPENAI_API_KEY", "independent-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    assert (
        tier3_commands.doctor(
            agents="codex",
            env_mode="local",
            agent_model=("codex=nvidia/nemotron-3-super-120b-a12b",),
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "operator credential and model plan resolved" in " ".join(output.split())


def test_doctor_accepts_isolated_docker_bridge_and_direct_agents(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from skillevaluator.tier3 import commands as tier3_commands

    monkeypatch.setattr(
        tier3_commands,
        "resolve_llm_provider",
        lambda: _provider(
            "nv_build",
            api_key="nvidia-build-key",
            base_url="https://integrate.api.nvidia.com/v1",
        ),
    )
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setenv("OPENAI_API_KEY", "independent-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    assert (
        tier3_commands.doctor(
            agents="codex,opencode",
            env_mode="docker",
            agent_model=("codex=nvidia/nemotron-3-super-120b-a12b",),
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Agent runtime credential" in output
    assert "operator credential and model plan resolved" in " ".join(output.split())


def test_harbor_preflight_system_exit_becomes_a_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    from harbor.environments.factory import EnvironmentFactory

    monkeypatch.setattr(
        "skillevaluator.tier3.harbor.runner.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="Docker Compose version v2", stderr=""),
    )
    monkeypatch.setattr(
        EnvironmentFactory,
        "run_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit("  daemon down\n")),
    )

    assert _check_prerequisites(env_mode="docker", agents=[]) == [
        "Harbor environment 'docker' is not ready: daemon down"
    ]


def test_harbor_preflight_does_not_swallow_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    from harbor.environments.factory import EnvironmentFactory

    monkeypatch.setattr(
        "skillevaluator.tier3.harbor.runner.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="Docker Compose version v2", stderr=""),
    )
    monkeypatch.setattr(
        EnvironmentFactory,
        "run_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        _check_prerequisites(env_mode="docker", agents=[])


def test_docker_prerequisite_rejects_missing_compose_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    from harbor.environments.factory import EnvironmentFactory

    monkeypatch.setattr(EnvironmentFactory, "run_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "skillevaluator.tier3.harbor.runner.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout="", stderr="unknown command: compose"),
    )

    errors = _check_prerequisites(env_mode="docker", agents=[])

    assert len(errors) == 1
    assert "Docker Compose v2" in errors[0]


def test_docker_prerequisite_accepts_compose_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    from harbor.environments.factory import EnvironmentFactory

    monkeypatch.setattr(EnvironmentFactory, "run_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "skillevaluator.tier3.harbor.runner.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout="Docker Compose version v2.40.0", stderr=""
        ),
    )

    assert _check_prerequisites(env_mode="docker", agents=[]) == []
