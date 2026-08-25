# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local mode (`--env-mode local`) wiring."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import errno
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
import tomllib
import traceback
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

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
from skillevaluator.tier3.harbor.secret_redaction import redact_secrets_in_log_line
from skillevaluator.tier3.harbor.secure_docker_environment import SECURE_DOCKER_ENV_IMPORT_PATH
from skillevaluator.tier3.harbor.stream_redaction import (
    StreamingLogRedactor,
    StreamingSecretRedactor,
    _StreamingKnownPatternRedactor,
)

_NATIVE_WINDOWS_LOCAL_REASON = "native Windows local mode requires WSL2; these checks exercise the POSIX backend"


class _NoopScopedExecEnvironment:
    @contextlib.contextmanager
    def scoped_exec_env(self, _env: dict[str, str]):
        yield


class _LocalCallbackBaseError(BaseException):
    pass


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
    environment._active_process_secret_values = {}
    environment._pending_creations = set()
    environment._creation_secret_values = {}
    environment._creation_cleanups = {}
    environment._creation_cleanup_errors = []
    environment._stop_requested = False
    environment._persistent_env = persistent_env or {}
    environment._output_callbacks = contextvars.ContextVar("test_local_output_callbacks", default=())
    environment._exec_env_overlays = contextvars.ContextVar("test_local_exec_env_overlays", default=())
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


def _initialized_local_environment(
    tmp_path: Path,
    *,
    persistent_env: dict[str, str] | None = None,
) -> SkillEvaluatorLocalEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    return SkillEvaluatorLocalEnvironment(
        environment_dir=environment_dir,
        environment_name="local-streaming-test",
        session_id="local-streaming-test",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        runtime_agent="opencode",
        runtime_root=str(runtime_root),
        sandbox_mode="off",
        allow_net=False,
        persistent_env=persistent_env,
    )


def _provider(name: str, *, api_key: str = "k", base_url: str | None = None) -> ProviderConfig:
    return ProviderConfig(provider=name, model="m", api_key=api_key, base_url=base_url, litellm_model="m", region=None)


def test_streaming_log_redactor_is_chunk_partition_invariant() -> None:
    multiline_secret = "FIRST-HALF\nSECOND-HALF"
    shorter_secret = "overlap-secret"
    longer_secret = "overlap-secret-tail"
    collision_secret = "redacted"
    known_secret = "".join(("nvapi-", "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8"))  # noqa: FLY002
    raw = (
        f"prefix {multiline_secret} {longer_secret} {shorter_secret} "
        f"{collision_secret} {known_secret} task-granularity suffix"
    )
    secrets = {multiline_secret, shorter_secret, longer_secret, collision_secret}

    baseline_redactor = StreamingLogRedactor(secrets)
    baseline = baseline_redactor.feed(raw) + baseline_redactor.finish()
    partitions = (
        [raw],
        list(raw),
        [raw[:1], raw[1:17], raw[17:43], raw[43:]],
        [raw[: len(raw) // 2], raw[len(raw) // 2 :]],
    )

    for chunks in partitions:
        redactor = StreamingLogRedactor(secrets)
        rendered = "".join(redactor.feed(chunk) for chunk in chunks) + redactor.finish()
        assert rendered == baseline

    assert "task-granularity" in baseline
    for secret in (*secrets, known_secret):
        assert secret not in baseline


def test_exact_stream_redactor_rejects_missing_compiled_prefix_matcher() -> None:
    redactor = StreamingSecretRedactor({"invariant-secret"})
    redactor._first_character_re = None

    with pytest.raises(RuntimeError, match="exact redactor invariant"):
        redactor.feed("ordinary output")


def test_known_pattern_redactor_rejects_unknown_candidate_kind() -> None:
    redactor = _StreamingKnownPatternRedactor()
    redactor._candidate_kind = "unexpected"
    redactor._candidate_prefix = "eyJ"
    redactor._candidate = ["eyJ"]

    with pytest.raises(RuntimeError, match="known-pattern redactor invariant"):
        redactor.feed("A")


@pytest.mark.parametrize("missing_stream", ("stdin", "stdout", "stderr"))
def test_local_stream_collector_rejects_missing_required_pipe(missing_stream: str) -> None:
    streams = {"stdin": object(), "stdout": object(), "stderr": object()}
    streams[missing_stream] = None
    process = SimpleNamespace(**streams)

    with pytest.raises(RuntimeError, match="local subprocess pipe invariant"):
        asyncio.run(
            SkillEvaluatorLocalEnvironment._collect_streamed_output(
                process,  # type: ignore[arg-type]
                b"",
                None,  # type: ignore[arg-type]
            )
        )


def test_streaming_log_redactor_preserves_nested_known_pattern_starts() -> None:
    jwt_secret = ".".join(("eyJ" + "A" * 20, "B" * 20, "C" * 20))
    raw = "ask-" + ("a" * 19 + "A1") + "--" + jwt_secret + "☃"
    expected = redact_secrets_in_log_line(raw)
    partitions = (
        [raw],
        list(raw),
        [raw[:5], raw[5:26], raw[26:51], raw[51:]],
    )

    for chunks in partitions:
        redactor = StreamingLogRedactor(())
        rendered = "".join(redactor.feed(chunk) for chunk in chunks) + redactor.finish()
        assert rendered == expected

    assert jwt_secret not in expected


@pytest.mark.parametrize(
    "raw",
    [
        " crsr_" + "0" * 16 + "sha256~" + "a" * 10 + "☃",
        "ordinary trailing xsk-abcdefgh",
        "ordinary trailing sk-short",
        "ordinary partial nvapi-",
        "ordinary partial crsr_0123",
        "ordinary partial eyJ" + "A" * 20,
        "_" + ".".join(("eyJ" + "A" * 20, "B" * 20, "C" * 20)),
        "é" + ".".join(("eyJ" + "A" * 20, "B" * 20, "C" * 20)),
    ],
)
def test_streaming_log_redactor_matches_batch_at_adjacent_patterns_and_eof(raw: str) -> None:
    expected = redact_secrets_in_log_line(raw)
    for chunks in ([raw], list(raw), [raw[: len(raw) // 2], raw[len(raw) // 2 :]]):
        redactor = StreamingLogRedactor(())
        rendered = "".join(redactor.feed(chunk) for chunk in chunks) + redactor.finish()
        assert rendered == expected


def test_streaming_log_redactor_emits_a_proven_long_key_before_its_terminator() -> None:
    redactor = StreamingLogRedactor(())

    emitted = redactor.feed("sk-" + "a" * 1_000_000)

    assert emitted == "sk-<redacted>"
    assert redactor.finish() == ""


@pytest.mark.parametrize(
    "prefix",
    [
        "eyJ",
        "eyJ" + "A" * 20 + ".",
    ],
)
def test_streaming_log_redactor_bounds_oversized_partial_jwt_candidates(prefix: str) -> None:
    redactor = StreamingLogRedactor(())
    raw = prefix + "A" * (512 * 1024)
    emitted: list[str] = []

    for offset in range(0, len(raw), 64 * 1024):
        emitted.append(redactor.feed(raw[offset : offset + 64 * 1024]))

    # Stage-zero/stage-one ambiguity must not retain attacker-sized output
    # until EOF. Once the conservative bound is reached, emit the marker and
    # discard the rest of that segment in bounded chunks.
    assert any(emitted)
    rendered = "".join(emitted) + redactor.finish()
    assert rendered == "jwt-<redacted>"
    assert len(rendered) < 256


@pytest.mark.parametrize(
    "raw",
    [
        "eyJ" + "A" * 256 + "." + "B" * 32 + "." + "C" * 32,
        "eyJ" + "A" * 20 + "." + "B" * 256 + "." + "C" * 32,
    ],
)
def test_oversized_partial_jwt_redaction_discards_later_segments(raw: str) -> None:
    redactor = StreamingLogRedactor(())

    rendered = redactor.feed(raw + " suffix") + redactor.finish()

    assert rendered == "jwt-<redacted> suffix"
    assert "B" * 32 not in rendered
    assert "C" * 32 not in rendered


@pytest.mark.parametrize(
    "known_secret",
    [
        "sk-abcdefgh",
        "nvapi-abcdefgh",
        "crsr_0123456789abcdef",
        "sha256~abcdefgh",
        ".".join(("eyJ" + "A" * 20, "B" * 20, "C" * 20)),
    ],
)
def test_exact_replacement_cannot_create_a_visible_known_secret(known_secret: str) -> None:
    exact_secret = "SECRET88"
    raw = exact_secret + known_secret
    for chunks in ([raw], list(raw), [raw[:7], raw[7:11], raw[11:]]):
        redactor = StreamingLogRedactor({exact_secret})
        rendered = "".join(redactor.feed(chunk) for chunk in chunks) + redactor.finish()
        assert exact_secret not in rendered
        assert known_secret not in rendered


def test_known_replacement_cannot_create_a_visible_exact_secret() -> None:
    exact_secret = "sk-<redacted>"
    raw = "sk-Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8"
    for chunks in ([raw], list(raw), [raw[:5], raw[5:19], raw[19:]]):
        redactor = StreamingLogRedactor({exact_secret})
        rendered = "".join(redactor.feed(chunk) for chunk in chunks) + redactor.finish()
        assert raw not in rendered
        assert exact_secret not in rendered


def test_local_is_a_registered_env_mode() -> None:
    assert ENV_MODE_LOCAL == "local"
    assert "local" in HARBOR_ENV_MODES


def test_build_command_uses_unified_flags_for_local_imports() -> None:
    cmd = build_harbor_run_command(dataset_path="/tmp/ds", agent="opencode", job_name="j", env_mode="local")
    joined = " ".join(cmd)
    assert "--agent-import-path" not in cmd
    assert "--environment-import-path" not in cmd
    assert "-a" not in cmd
    assert cmd.count("--agent") == 1
    assert cmd[cmd.index("--agent") + 1] == LOCAL_AGENT_IMPORT_PATHS["opencode"]
    assert cmd.count("--env") == 1
    assert cmd[cmd.index("--env") + 1] == LOCAL_ENV_IMPORT_PATH
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
    assert "--agent-import-path" not in cmd
    assert "--environment-import-path" not in cmd
    assert "-a" not in cmd
    assert cmd.count("--agent") == 1
    assert cmd[cmd.index("--agent") + 1] == "codex"
    assert cmd.count("--env") == 1
    assert cmd[cmd.index("--env") + 1] == SECURE_DOCKER_ENV_IMPORT_PATH


def test_docker_bridge_command_uses_unified_flags_for_custom_agent_and_secure_environment() -> None:
    import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex"

    cmd = build_harbor_run_command(
        dataset_path="/tmp/ds",
        agent="codex",
        job_name="nvidia-build-codex",
        env_mode="docker",
        agent_import_path=import_path,
    )

    assert "--agent-import-path" not in cmd
    assert "--environment-import-path" not in cmd
    assert "-a" not in cmd
    assert cmd.count("--agent") == 1
    assert cmd[cmd.index("--agent") + 1] == import_path
    assert cmd.count("--env") == 1
    assert cmd[cmd.index("--env") + 1] == SECURE_DOCKER_ENV_IMPORT_PATH


def test_nvidia_build_bridge_agents_are_not_local_mode_agents() -> None:
    assert NVIDIA_BUILD_AGENT_IMPORT_PATHS == {
        "codex": "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex",
        "claude-code": "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode",
    }
    assert "codex" in LOCAL_AGENT_IMPORT_PATHS
    assert LOCAL_AGENT_IMPORT_PATHS["codex"] != NVIDIA_BUILD_AGENT_IMPORT_PATHS["codex"]


def test_harbor_unified_specs_import_non_abstract_skill_evaluator_classes(tmp_path: Path) -> None:
    import inspect

    from harbor.agents.factory import AgentFactory
    from harbor.cli.utils import resolve_environment_spec
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    agent_specs = {
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalClaudeCode": ("SkillEvaluatorLocalClaudeCode"),
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalCodex": "SkillEvaluatorLocalCodex",
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalOpenCode": "SkillEvaluatorLocalOpenCode",
    }
    for import_path, expected_class_name in agent_specs.items():
        agent = AgentFactory.create_agent_from_config(
            AgentConfig(name=import_path, model_name="openai/gpt-4.1-mini"),
            logs_dir=tmp_path / "agent-logs",
        )

        assert agent.__class__.__name__ == expected_class_name
        assert not inspect.isabstract(agent.__class__)

    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    environment_specs = {
        LOCAL_ENV_IMPORT_PATH: (
            "SkillEvaluatorLocalEnvironment",
            {
                "runtime_agent": "codex",
                "runtime_root": str(tmp_path / "runtime"),
                "sandbox_mode": "off",
            },
        ),
        SECURE_DOCKER_ENV_IMPORT_PATH: ("SkillEvaluatorSecureDockerEnvironment", {}),
    }
    for index, (import_path, (expected_class_name, kwargs)) in enumerate(environment_specs.items()):
        environment_type, resolved_import_path = resolve_environment_spec(import_path)
        assert environment_type is None
        assert resolved_import_path == import_path
        environment = EnvironmentFactory.create_environment_from_config(
            EnvironmentConfig(
                type=environment_type,
                import_path=resolved_import_path,
                kwargs=kwargs,
            ),
            environment_dir=environment_dir,
            environment_name="unified-import-smoke",
            session_id=f"unified-import-smoke-{index}",
            trial_paths=TrialPaths(tmp_path / f"trial-{index}"),
            task_env_config=TaskEnvironmentConfig(),
        )

        assert environment.__class__.__name__ == expected_class_name
        assert not inspect.isabstract(environment.__class__)


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
    agent = agent_class(logs_dir=tmp_path, model_name="nvidia/nemotron-3-nano-30b-a3b")
    agent.render_instruction = lambda instruction: instruction
    agent._resolve_auth_json_path = lambda: None
    agent._build_register_skills_command = lambda: None
    agent._build_register_mcp_servers_command = lambda: None
    agent.build_cli_flags = lambda: ""
    agent.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    calls: list[tuple[str, dict[str, str]]] = []
    retained_logs: list[tuple[str, str]] = []
    origins: list[str] = []

    class Environment(_NoopScopedExecEnvironment):
        default_user = None

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
        effective_config = self._build_effective_config()
        await self._upload_effective_config(
            environment,
            effective_config,
            (self._REMOTE_CODEX_HOME / "config.toml").as_posix(),
        )
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
    config_upload = next(content for destination, content in retained_logs if destination.endswith("/config.toml"))
    config = tomllib.loads(config_upload)
    assert config["model_provider"] == "openai_compatible"
    assert config["openai_base_url"] == f"{origins[0]}/v1"
    assert config["model_providers"]["openai_compatible"]["base_url"] == f"{origins[0]}/v1"
    assert "api.openai.com" not in config_upload
    assert "real-nvidia-key" not in config_upload
    client_command, client_env = next((command, env) for command, env in calls if "codex exec" in command)
    assert "env -u NVIDIA_API_KEY" in client_command
    assert "NVIDIA_API_KEY" not in client_env
    assert client_env["OPENAI_API_KEY"] not in {"real-nvidia-key", "nvidia-build-loopback"}
    assert any(destination.endswith("nvidia-build-bridge.log") for destination, _content in retained_logs)

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
    tmp_path: Path,
) -> None:
    from harbor.agents.installed.codex import Codex

    agent = SkillEvaluatorNvidiaBuildCodex(
        logs_dir=tmp_path,
        model_name="nvidia/meta/llama-3.1-8b-instruct",
    )
    calls: list[tuple[str, dict[str, str]]] = []
    root_calls: list[tuple[str, dict[str, str]]] = []
    uploads: list[tuple[str, str, int]] = []

    class Environment(_NoopScopedExecEnvironment):
        default_user = None

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

    assert len(uploads) == 2
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
    config_upload = next(content for destination, content, _mode in uploads if destination.endswith("/config.toml"))
    config = tomllib.loads(config_upload)
    assert config["model_provider"] == "openai_compatible"
    assert config["openai_base_url"] == "http://127.0.0.1:43123/v1"
    assert config["model_providers"]["openai_compatible"] == {
        "name": "OpenAI-compatible provider",
        "base_url": "http://127.0.0.1:43123/v1",
        "env_key": "OPENAI_API_KEY",
        "wire_api": "responses",
    }
    assert all("openai_base_url" not in command for command, _ in calls)
    assert "nvidia-secret" not in config_upload
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
    agent._nvidia_build_bridge_client_env = {"OPENAI_API_KEY": "bridge-client-token-secret"}
    captured: list[tuple[str, dict[str, str]]] = []
    scoped: list[dict[str, str]] = []
    simple_command = "codex exec --model model -- test"
    compound_command = 'if [ -d "$CODEX_HOME/sessions" ]; then cp -R "$CODEX_HOME/sessions" /logs/agent/sessions; fi'

    class Environment:
        @contextlib.contextmanager
        def scoped_exec_env(self, values: dict[str, str]):
            scoped.append(dict(values))
            yield

    async def raw_exec(
        _self: Codex,
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        captured.append((command, dict(env or {})))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(Codex, "exec_as_agent", raw_exec)

    environment = Environment()
    for original_command in (simple_command, compound_command):
        asyncio.run(
            agent.exec_as_agent(
                environment,
                command=original_command,
                env={"NVIDIA_API_KEY": "must-not-leak"},
            )
        )

    unset_prefix = "env -u NVIDIA_API_KEY -u OPENAI_BASE_URL -u OPENAI_API_BASE bash -c"
    assert [command for command, _ in captured] == [
        f"{unset_prefix} {shlex.quote('codex exec --model nvidia/model -- test')}",
        f"{unset_prefix} {shlex.quote(compound_command)}",
    ]
    assert all("NVIDIA_API_KEY" not in env for _, env in captured)
    assert all(env["OPENAI_API_KEY"] == "bridge-client-token-secret" for _, env in captured)
    assert scoped == [
        {"OPENAI_API_KEY": "bridge-client-token-secret"},
        {"OPENAI_API_KEY": "bridge-client-token-secret"},
    ]


def test_nvidia_build_claude_bridge_configures_origin_and_full_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.agents.installed.claude_code import ClaudeCode

    agent = object.__new__(SkillEvaluatorNvidiaBuildClaudeCode)
    agent.model_name = "nvidia/meta/llama-3.1-8b-instruct"
    calls: list[tuple[str, dict[str, str]]] = []

    class Environment(_NoopScopedExecEnvironment):
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
def test_real_local_exec_streams_stdout_and_stderr_through_harbor_callback(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_chunks: list[tuple[str, str]] = []

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    async def exercise() -> object:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            return await environment.exec("printf stdout-value; printf stderr-value >&2")

    result = asyncio.run(exercise())

    assert result.stdout == "stdout-value"
    assert result.stderr == "stderr-value"
    assert "".join(text for text, stream in callback_chunks if stream == "stdout") == result.stdout
    assert "".join(text for text, stream in callback_chunks if stream == "stderr") == result.stderr


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_streamed_nonzero_exit_preserves_output_and_return_code(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_chunks: list[tuple[str, str]] = []
    secret = "nonzero-stream-secret"

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    async def exercise() -> object:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            return await environment.exec(
                'printf "stdout=%s\\n" "$NONZERO_TOKEN"; printf "stderr=%s\\n" "$NONZERO_TOKEN" >&2; exit 7',
                env={"NONZERO_TOKEN": secret},
            )

    result = asyncio.run(exercise())
    callback_stdout = "".join(text for text, stream in callback_chunks if stream == "stdout")
    callback_stderr = "".join(text for text, stream in callback_chunks if stream == "stderr")

    assert result.return_code == 7
    assert callback_stdout == result.stdout
    assert callback_stderr == result.stderr
    assert secret not in callback_stdout
    assert secret not in callback_stderr


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_real_local_exec_redacts_merged_secrets_across_byte_and_line_boundaries(
    tmp_path: Path,
) -> None:
    persistent_secret = "persistent-first\npersistent-second"
    scoped_secret = "scoped-secret-value"
    per_call_secret = "per-call-secret-value"
    environment = _initialized_local_environment(
        tmp_path,
        persistent_env={"PERSISTENT_TOKEN": persistent_secret},
    )
    callback_chunks: list[tuple[str, str]] = []
    script = """
import os
import time

values = (
    (1, os.environ["PERSISTENT_TOKEN"]),
    (2, os.environ["SCOPED_SECRET"]),
    (1, os.environ["PER_CALL_KEY"]),
    (2, "unicode-snowman-☃"),
)
for descriptor, value in values:
    payload = value.encode("utf-8")
    for byte in payload:
        os.write(descriptor, bytes((byte,)))
        time.sleep(0.001)
    os.write(descriptor, b"\\n")
"""
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    async def exercise() -> object:
        await environment.start()
        with (
            environment.scoped_exec_env({"SCOPED_SECRET": scoped_secret}),
            environment.scoped_output_callback(on_output),
        ):
            return await environment.exec(command, env={"PER_CALL_KEY": per_call_secret})

    result = asyncio.run(exercise())
    callback_stdout = "".join(text for text, stream in callback_chunks if stream == "stdout")
    callback_stderr = "".join(text for text, stream in callback_chunks if stream == "stderr")

    assert callback_stdout == result.stdout
    assert callback_stderr == result.stderr
    assert "unicode-snowman-☃" in callback_stderr
    for secret in (persistent_secret, scoped_secret, per_call_secret):
        assert secret not in callback_stdout
        assert secret not in callback_stderr
        assert secret not in (result.stdout or "")
        assert secret not in (result.stderr or "")


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_stream_redaction_handles_short_sensitive_and_marker_collision_values(
    tmp_path: Path,
) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_chunks: list[str] = []

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    async def exercise() -> object:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            return await environment.exec(
                'printf "%s|%s|%s" "$API_KEY" "$PUBLIC_LABEL" "$COLLISION_SECRET"',
                env={
                    "API_KEY": "x",
                    "PUBLIC_LABEL": "ok",
                    "COLLISION_SECRET": "redacted",
                },
            )

    result = asyncio.run(exercise())
    callback_output = "".join(callback_chunks)

    assert callback_output == result.stdout
    assert "|ok|" in callback_output
    assert "x" not in callback_output
    assert "redacted" not in callback_output.lower()


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_stream_redacts_known_secret_patterns_across_reader_chunks(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_chunks: list[tuple[str, str]] = []
    sk_secret = "".join(("sk-", "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8"))  # noqa: FLY002
    nvapi_secret = "".join(("nvapi-", "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8"))  # noqa: FLY002
    crsr_secret = "".join(("crsr_", "0123456789abcdef"))  # noqa: FLY002
    openshift_secret = "".join(("sha256~", "Abc.def_Ghi-jkl~mno"))  # noqa: FLY002
    jwt_secret = ".".join(("eyJ" + "A" * 20, "B" * 20, "C" * 20))
    script = f"""
import os
import time

values = (
    (1, {sk_secret!r}),
    (2, {nvapi_secret!r}),
    (1, {crsr_secret!r}),
    (2, {openshift_secret!r}),
    (1, {jwt_secret!r}),
    (2, "task-granularity"),
)
for descriptor, value in values:
    for byte in value.encode("utf-8"):
        os.write(descriptor, bytes((byte,)))
        time.sleep(0.001)
    os.write(descriptor, b"\\n")
"""
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    async def exercise() -> object:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            return await environment.exec(command)

    result = asyncio.run(exercise())
    callback_stdout = "".join(text for text, stream in callback_chunks if stream == "stdout")
    callback_stderr = "".join(text for text, stream in callback_chunks if stream == "stderr")

    assert callback_stdout == result.stdout
    assert callback_stderr == result.stderr
    assert "task-granularity" in callback_stderr
    for secret in (sk_secret, nvapi_secret, crsr_secret, openshift_secret, jwt_secret):
        assert secret not in callback_stdout
        assert secret not in callback_stderr


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_concurrent_callback_contexts_are_isolated(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    labels = tuple(f"callback-{index}" for index in range(10))
    callback_outputs: dict[str, list[str]] = {label: [] for label in labels}

    async def exercise() -> dict[str, object]:
        await environment.start()

        async def run(label: str) -> object:
            async def on_output(text: str, _stream: str) -> None:
                callback_outputs[label].append(text)

            with environment.scoped_output_callback(on_output):
                return await environment.exec(f"printf '{label}-first\\n'; sleep 0.05; printf '{label}-second\\n'")

        results = await asyncio.gather(*(run(label) for label in labels))
        return dict(zip(labels, results, strict=True))

    results = asyncio.run(exercise())

    for label in labels:
        rendered = "".join(callback_outputs[label])
        assert rendered == results[label].stdout
        for other_label in labels:
            if other_label != label:
                assert other_label not in rendered


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_nested_callbacks_run_in_scope_order(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    calls: list[tuple[str, str, str]] = []

    async def outer(text: str, stream: str) -> None:
        calls.append(("outer", text, stream))

    async def inner(text: str, stream: str) -> None:
        calls.append(("inner", text, stream))

    async def exercise() -> object:
        await environment.start()
        with (
            environment.scoped_output_callback(outer),
            environment.scoped_output_callback(inner),
        ):
            return await environment.exec("printf 'nested-output\\n'")

    result = asyncio.run(exercise())

    assert result.stdout == "nested-output\n"
    assert calls == [
        ("outer", "nested-output\n", "stdout"),
        ("inner", "nested-output\n", "stdout"),
    ]


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_callback_receives_complete_line_before_process_exit(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    first_line = asyncio.Event()

    async def on_output(text: str, stream: str) -> None:
        if text == "first-line\n" and stream == "stdout":
            first_line.set()

    async def exercise() -> object:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            task = asyncio.create_task(environment.exec("printf 'first-line\\n'; sleep 1; printf done"))
            await asyncio.wait_for(first_line.wait(), timeout=0.5)
            assert not task.done()
            return await task

    result = asyncio.run(exercise())

    assert result.stdout == "first-line\ndone"


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_callback_streams_safe_partial_line_before_process_exit(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    partial_output = asyncio.Event()
    callback_chunks: list[str] = []

    async def on_output(text: str, stream: str) -> None:
        if stream == "stdout":
            callback_chunks.append(text)
            if "safe-partial-output" in "".join(callback_chunks):
                partial_output.set()

    async def exercise() -> object:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            task = asyncio.create_task(environment.exec("printf safe-partial-output; sleep 1; printf done"))
            await asyncio.wait_for(partial_output.wait(), timeout=0.5)
            assert not task.done()
            return await task

    result = asyncio.run(exercise())

    assert "".join(callback_chunks) == result.stdout == "safe-partial-outputdone"


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_streaming_preserves_exact_json_stdin_bootstrap(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    captured_payloads: list[bytes] = []
    original_collect = environment._collect_streamed_output
    per_call_env = {"PER_CALL_VALUE": "☃"}

    async def capture_payload(
        proc: asyncio.subprocess.Process,
        stdin_data: bytes,
        callback_output: object,
    ) -> tuple[bytes, bytes]:
        captured_payloads.append(stdin_data)
        return await original_collect(
            proc,
            stdin_data,
            callback_output,  # type: ignore[arg-type]
        )

    async def on_output(_text: str, _stream: str) -> None:
        return None

    async def exercise() -> tuple[object, bytes]:
        await environment.start()
        with (
            environment.scoped_exec_env({"SCOPED_VALUE": "scoped"}),
            environment.scoped_output_callback(on_output),
        ):
            expected_payload = json.dumps(environment._exec_env(per_call_env)).encode("utf-8")
            environment._collect_streamed_output = capture_payload  # type: ignore[method-assign]
            result = await environment.exec('printf %s "$PER_CALL_VALUE"', env=per_call_env)
        return result, expected_payload

    result, expected_payload = asyncio.run(exercise())

    assert result.stdout == "☃"
    assert captured_payloads == [expected_payload]


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_exec_without_callback_keeps_buffered_communicate_path(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)

    async def unexpected_stream(*_args: object, **_kwargs: object) -> tuple[bytes, bytes]:
        pytest.fail("no-callback exec unexpectedly selected the streaming collector")

    async def exercise() -> object:
        await environment.start()
        environment._collect_streamed_output = unexpected_stream  # type: ignore[method-assign]
        return await environment.exec("printf buffered-only")

    result = asyncio.run(exercise())

    assert result.stdout == "buffered-only"
    assert result.stderr == ""
    assert result.return_code == 0


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_local_streaming_tolerates_child_closing_json_stdin_early(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)

    class EarlyExitSandbox:
        plan = local_sandbox.SandboxPlan("none", "advisory-only", "early-exit-test")

        @staticmethod
        def wrap(_argv: list[str], **_kwargs: object) -> list[str]:
            return [sys.executable, "-c", "import os; os._exit(7)"]

    environment._sandbox = EarlyExitSandbox()

    async def on_output(_text: str, _stream: str) -> None:
        return None

    async def exercise() -> object:
        await environment.start()
        environment._sandbox = EarlyExitSandbox()
        with environment.scoped_output_callback(on_output):
            return await environment.exec("ignored", env={"FILLER": "v" * (1024 * 1024)})

    result = asyncio.run(exercise())

    assert result.return_code == 7
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
@pytest.mark.parametrize("error_type", [TimeoutError, _LocalCallbackBaseError, asyncio.CancelledError])
def test_local_callback_exception_is_propagated_after_process_reap(
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_error = error_type("local callback failed")
    secret = "local-callback-base-error-secret"
    processes: list[asyncio.subprocess.Process] = []
    create_subprocess_exec = asyncio.create_subprocess_exec
    callback_chunks: list[str] = []
    reaped_by_exec: list[bool] = []
    caught_errors: list[BaseException] = []

    async def capture_process(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await create_subprocess_exec(*args, **kwargs)
        processes.append(process)
        return process

    async def failing_callback(text: str, _stream: str) -> None:
        callback_chunks.append(text)
        raise callback_error

    async def exercise() -> None:
        await environment.start()
        try:
            with (
                pytest.MonkeyPatch.context() as patch,
                environment.scoped_output_callback(failing_callback),
            ):
                patch.setattr(asyncio, "create_subprocess_exec", capture_process)
                for _ in range(3):
                    with pytest.raises(error_type) as caught:
                        await environment.exec(
                            'printf "%s\\n" "$CALLBACK_SECRET"; sleep 30',
                            env={"CALLBACK_SECRET": secret},
                        )
                    caught_errors.append(caught.value)
                    reaped_by_exec.append(processes[-1].returncode is not None)
        finally:
            for process in processes:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()

    asyncio.run(exercise())

    assert callback_chunks
    assert secret not in "".join(callback_chunks)
    assert secret not in str(callback_error)
    assert caught_errors == [callback_error] * 3
    assert len(processes) == 3
    assert reaped_by_exec == [True] * 3


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_local_callback_error_remains_primary_when_cleanup_reports_failure(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_error = _LocalCallbackBaseError("primary callback failure")
    secret = "synthetic cleanup report failure"
    original_terminate = environment._terminate_process_tree
    cleanup_calls = 0

    async def failing_cleanup(
        _proc: asyncio.subprocess.Process,
        _communication: asyncio.Task[tuple[bytes, bytes]] | None = None,
    ) -> tuple[bytes, bytes]:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise PermissionError(errno.EACCES, "Denied", secret)

    async def failing_callback(_text: str, _stream: str) -> None:
        raise callback_error

    async def exercise() -> tuple[BaseException, bool, bool]:
        await environment.start()
        environment._terminate_process_tree = failing_cleanup  # type: ignore[method-assign]
        with (
            environment.scoped_output_callback(failing_callback),
            pytest.raises(_LocalCallbackBaseError) as caught,
        ):
            await environment.exec(
                "printf 'callback-output\\n'; sleep 30",
                env={"API_KEY": secret},
            )
        retained_before_retry = bool(environment._active_processes)
        environment._terminate_process_tree = original_terminate  # type: ignore[method-assign]
        await environment.stop(delete=False)
        return caught.value, retained_before_retry, not environment._active_processes

    caught, retained_before_retry, released_after_retry = asyncio.run(exercise())

    assert caught is callback_error
    assert isinstance(caught.__cause__, RuntimeError)
    assert caught.__cause__.__context__ is None
    assert cleanup_calls == 1
    assert retained_before_retry
    assert released_after_retry
    assert any("cleanup" in note.lower() for note in caught.__notes__)
    assert secret not in "".join(caught.__notes__)
    assert secret not in str(caught.__cause__)
    assert secret not in "".join(traceback.format_exception(caught))


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_local_callback_error_message_receives_only_redacted_output(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    secret = "callback-error-message-secret"

    async def failing_callback(text: str, _stream: str) -> None:
        raise RuntimeError(f"consumer rejected: {text}")

    async def exercise() -> RuntimeError:
        await environment.start()
        with (
            environment.scoped_output_callback(failing_callback),
            pytest.raises(RuntimeError, match="consumer rejected") as caught,
        ):
            await environment.exec(
                'printf "%s\\n" "$ERROR_TOKEN"; sleep 30',
                env={"ERROR_TOKEN": secret},
            )
        return caught.value

    error = asyncio.run(exercise())

    assert secret not in str(error)
    assert "consumer rejected" in str(error)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_local_stream_collector_failure_is_propagated_after_process_reap(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    collector_error = RuntimeError("synthetic stream collector failure")
    processes: list[asyncio.subprocess.Process] = []
    create_subprocess_exec = asyncio.create_subprocess_exec
    reaped_by_exec: list[bool] = []

    async def capture_process(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await create_subprocess_exec(*args, **kwargs)
        processes.append(process)
        return process

    async def failing_collector(*_args: object, **_kwargs: object) -> tuple[bytes, bytes]:
        raise collector_error

    async def on_output(_text: str, _stream: str) -> None:
        return None

    async def exercise() -> BaseException:
        await environment.start()
        environment._collect_streamed_output = failing_collector  # type: ignore[method-assign]
        try:
            with (
                pytest.MonkeyPatch.context() as patch,
                environment.scoped_output_callback(on_output),
                pytest.raises(RuntimeError, match="synthetic stream collector failure") as caught,
            ):
                patch.setattr(asyncio, "create_subprocess_exec", capture_process)
                await environment.exec("sleep 30")
            reaped_by_exec.append(processes[0].returncode is not None)
            return caught.value
        finally:
            for process in processes:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()

    caught = asyncio.run(exercise())

    assert caught is collector_error
    assert caught.__cause__ is None
    assert reaped_by_exec == [True]


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
def test_streamed_timeout_callback_matches_result_and_contains_descendants(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    environment._local_command_guardrail_reason = lambda *_args: ""  # type: ignore[method-assign]
    secret = "stream-timeout-secret-value"
    child_pid_path = environment._workspace / "stream-timeout-child-pid"
    callback_chunks: list[tuple[str, str]] = []
    command = (
        'printf "%s\\n" "$TIMEOUT_TOKEN"; printf "stderr-before-timeout\\n" >&2; '
        "(sleep 30) & printf '%s' \"$!\" > stream-timeout-child-pid; wait"
    )

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    async def exercise() -> object:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            return await environment.exec(
                command,
                env={"TIMEOUT_TOKEN": secret},
                timeout_sec=1,
            )

    result = asyncio.run(exercise())
    callback_stdout = "".join(text for text, stream in callback_chunks if stream == "stdout")
    callback_stderr = "".join(text for text, stream in callback_chunks if stream == "stderr")

    assert result.return_code == 124
    assert callback_stdout == result.stdout
    assert callback_stderr == result.stderr
    assert callback_stderr == "stderr-before-timeout\nTimed out"
    assert secret not in callback_stdout
    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("with_callback", [False, True])
def test_timeout_diagnostic_cannot_synthesize_sensitive_value(
    tmp_path: Path,
    with_callback: bool,
) -> None:
    environment = _initialized_local_environment(tmp_path)
    secret = "Timed out"
    callback_chunks: list[tuple[str, str]] = []

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    async def exercise() -> object:
        await environment.start()
        callback_scope = environment.scoped_output_callback(on_output) if with_callback else contextlib.nullcontext()
        with callback_scope:
            return await environment.exec(
                "sleep 30",
                env={"TIMEOUT_SECRET": secret},
                timeout_sec=0.1,
            )

    result = asyncio.run(exercise())
    callback_stderr = "".join(text for text, stream in callback_chunks if stream == "stderr")

    assert result.return_code == 124
    assert secret not in (result.stderr or "")
    assert secret not in callback_stderr
    if with_callback:
        assert callback_stderr == result.stderr


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("with_callback", [False, True])
@pytest.mark.parametrize("secret", ["prefix\nTimed out", "\n"])
def test_timeout_diagnostic_uses_the_live_stderr_redactor_across_its_boundary(
    tmp_path: Path,
    with_callback: bool,
    secret: str,
) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_chunks: list[tuple[str, str]] = []

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    async def exercise() -> object:
        await environment.start()
        callback_scope = environment.scoped_output_callback(on_output) if with_callback else contextlib.nullcontext()
        with callback_scope:
            return await environment.exec(
                "printf prefix >&2; sleep 30",
                env={"TIMEOUT_SECRET": secret},
                timeout_sec=0.1,
            )

    result = asyncio.run(exercise())
    callback_stderr = "".join(text for text, stream in callback_chunks if stream == "stderr")

    assert result.return_code == 124
    assert secret not in (result.stderr or "")
    assert secret not in callback_stderr
    if with_callback:
        assert callback_stderr == result.stderr


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_blocked_callback_does_not_replace_command_timeout_with_cancelled_error(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    release_callback = asyncio.Event()

    async def blocked_callback(_text: str, _stream: str) -> None:
        await release_callback.wait()

    async def exercise() -> tuple[object, float]:
        await environment.start()
        started = time.monotonic()
        with environment.scoped_output_callback(blocked_callback):
            result = await environment.exec("printf callback-blocked; sleep 30", timeout_sec=0.1)
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(exercise())

    assert elapsed < 3
    assert result.return_code == 124
    assert "Timed out" in (result.stderr or "")


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_callback_suppressing_one_cleanup_cancellation_is_not_reentered_or_leaked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    environment = _initialized_local_environment(tmp_path)
    monkeypatch.setattr(local_environment, "_REAP_TERM_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_KILL_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_CANCEL_SECONDS", 0.05)
    callback_calls = 0

    async def cancellation_suppressing_callback(_text: str, _stream: str) -> None:
        nonlocal callback_calls
        callback_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A callback can perform async cleanup after its first
            # cancellation. A bounded second cancellation must finish that
            # cleanup path; exec must never invoke it concurrently again.
            await asyncio.Event().wait()

    async def exercise() -> tuple[object, list[str]]:
        await environment.start()
        with environment.scoped_output_callback(cancellation_suppressing_callback):
            result = await asyncio.wait_for(
                environment.exec("printf callback-blocked; sleep 30", timeout_sec=0.02),
                timeout=0.5,
            )
        await asyncio.sleep(0)
        leaked = [
            repr(task.get_coro())
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and any(
                name in repr(task.get_coro())
                for name in ("_collect_streamed_output", "invoke_callback", "_cancel_task_repeatedly")
            )
        ]
        return result, leaked

    result, leaked = asyncio.run(exercise())

    assert result.return_code == 124
    assert callback_calls == 1
    assert leaked == []


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("repeat_cancellation", [False, True])
def test_cancellation_during_timeout_diagnostic_callback_reaps_callback_tasks(
    tmp_path: Path,
    repeat_cancellation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    environment = _initialized_local_environment(tmp_path)
    monkeypatch.setattr(local_environment, "_REAP_TERM_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_KILL_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_CANCEL_SECONDS", 0.05)
    callback_started = asyncio.Event()
    callback_calls = 0

    async def blocked_callback(_text: str, _stream: str) -> None:
        nonlocal callback_calls
        callback_calls += 1
        callback_started.set()
        await asyncio.Event().wait()

    async def exercise() -> tuple[BaseException | None, list[str]]:
        await environment.start()
        with environment.scoped_output_callback(blocked_callback):
            task = asyncio.create_task(environment.exec("sleep 30", timeout_sec=0.02))
            await asyncio.wait_for(callback_started.wait(), timeout=2)
            task.cancel()
            if repeat_cancellation:
                await asyncio.sleep(0)
                task.cancel()
            outcome: BaseException | None = None
            try:
                await asyncio.wait_for(task, timeout=1)
            except BaseException as exc:
                outcome = exc
        await asyncio.sleep(0)
        leaked = [
            repr(pending.get_coro())
            for pending in asyncio.all_tasks()
            if pending is not asyncio.current_task()
            and not pending.done()
            and any(
                name in repr(pending.get_coro()) for name in ("finish", "invoke_callback", "_cancel_task_repeatedly")
            )
        ]
        return outcome, leaked

    outcome, leaked = asyncio.run(exercise())

    assert isinstance(outcome, asyncio.CancelledError)
    assert callback_calls == 1
    assert leaked == []
    assert not environment._active_processes


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("repeat_cancellation", [False, True])
def test_cancellation_during_timeout_callback_cleanup_is_not_swallowed(
    tmp_path: Path,
    repeat_cancellation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    environment = _initialized_local_environment(tmp_path)
    monkeypatch.setattr(local_environment, "_REAP_TERM_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_KILL_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_CANCEL_SECONDS", 0.05)
    callback_cleanup_started = asyncio.Event()

    async def cleanup_awaiting_callback(_text: str, _stream: str) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            callback_cleanup_started.set()
            await asyncio.Event().wait()

    async def exercise() -> tuple[BaseException | object, list[str]]:
        await environment.start()
        with environment.scoped_output_callback(cleanup_awaiting_callback):
            task = asyncio.create_task(environment.exec("sleep 30", timeout_sec=0.02))
            # This is set by the timeout finalizer's own first cancellation,
            # proving caller cancellation lands in its no-exception cleanup
            # path rather than the preceding bounded wait.
            await asyncio.wait_for(callback_cleanup_started.wait(), timeout=2)
            task.cancel()
            if repeat_cancellation:
                await asyncio.sleep(0)
                task.cancel()
            try:
                outcome: BaseException | object = await asyncio.wait_for(task, timeout=1)
            except BaseException as exc:
                outcome = exc
        await asyncio.sleep(0)
        leaked = [
            repr(pending.get_coro())
            for pending in asyncio.all_tasks()
            if pending is not asyncio.current_task()
            and not pending.done()
            and any(
                name in repr(pending.get_coro()) for name in ("finish", "invoke_callback", "_cancel_task_repeatedly")
            )
        ]
        return outcome, leaked

    outcome, leaked = asyncio.run(exercise())

    assert isinstance(outcome, asyncio.CancelledError)
    assert leaked == []
    assert not environment._active_processes


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_large_dense_short_secret_has_the_same_completed_outcome_with_callback(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    script = "import os; os.write(1, b'x' * 2_000_000)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    async def on_output(_text: str, _stream: str) -> None:
        return None

    async def exercise(with_callback: bool) -> object:
        await environment.start()
        callback_scope = environment.scoped_output_callback(on_output) if with_callback else contextlib.nullcontext()
        with callback_scope:
            return await environment.exec(command, env={"API_KEY": "x"}, timeout_sec=1.5)

    without_callback = asyncio.run(exercise(False))
    with_callback = asyncio.run(exercise(True))

    assert without_callback.return_code == with_callback.return_code == 0
    assert without_callback.stdout == with_callback.stdout
    assert "x" not in (with_callback.stdout or "")


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_streamed_exec_stays_bounded_when_output_closes_before_process_exit(tmp_path: Path) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_chunks: list[tuple[str, str]] = []

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    async def exercise() -> tuple[object, float]:
        await environment.start()
        started = time.monotonic()
        with environment.scoped_output_callback(on_output):
            result = await environment.exec("exec 1>&- 2>&-; sleep 30", timeout_sec=1)
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(exercise())

    assert elapsed < 3
    assert result.return_code == 124
    assert result.stdout == ""
    assert result.stderr == "Timed out"
    assert callback_chunks == [("Timed out", "stderr")]


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
@pytest.mark.parametrize("repeat_cancellation", [False, True])
def test_streamed_exec_cancellation_reaps_descendants(
    tmp_path: Path,
    repeat_cancellation: bool,
) -> None:
    environment = _initialized_local_environment(tmp_path)
    environment._local_command_guardrail_reason = lambda *_args: ""  # type: ignore[method-assign]
    secret = "stream-cancel-secret-value"
    child_pid_path = environment._workspace / "stream-cancel-child-pid"
    callback_started = asyncio.Event()
    callback_chunks: list[str] = []

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)
        callback_started.set()

    async def exercise() -> int:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            task = asyncio.create_task(
                environment.exec(
                    'printf "%s\\n" "$CANCEL_TOKEN"; (sleep 30) & printf \'%s\' "$!" > stream-cancel-child-pid; wait',
                    env={"CANCEL_TOKEN": secret},
                )
            )
            await asyncio.wait_for(callback_started.wait(), timeout=5)
            for _ in range(500):
                if child_pid_path.exists():
                    break
                await asyncio.sleep(0.01)
            assert child_pid_path.exists()
            task.cancel()
            if repeat_cancellation:
                await asyncio.sleep(0)
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        return int(child_pid_path.read_text(encoding="ascii"))

    child_pid = asyncio.run(exercise())

    assert callback_chunks
    assert secret not in "".join(callback_chunks)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("repeat_cancellation", [False, True])
def test_streamed_exec_cancellation_during_final_flush_reaps_descendants(
    tmp_path: Path,
    repeat_cancellation: bool,
) -> None:
    environment = _initialized_local_environment(tmp_path)
    environment._local_command_guardrail_reason = lambda *_args: ""  # type: ignore[method-assign]
    child_pid_path = environment._workspace / "flush-cancel-child-pid"
    callback_started = asyncio.Event()
    command = (
        "(trap '' TERM; exec >/dev/null 2>&1; sleep 30) & "
        f"printf '%s' \"$!\" > {shlex.quote(child_pid_path.name)}; "
        # A lone known-token prefix stays buffered until final redactor flush.
        "printf s"
    )

    async def blocked_callback(_text: str, _stream: str) -> None:
        callback_started.set()
        await asyncio.Event().wait()

    async def exercise() -> int:
        await environment.start()
        with environment.scoped_output_callback(blocked_callback):
            task = asyncio.create_task(environment.exec(command))
            await asyncio.wait_for(callback_started.wait(), timeout=5)
            assert child_pid_path.exists()
            assert not environment._active_processes
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
            task.cancel()
            if repeat_cancellation:
                await asyncio.sleep(0)
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        return int(child_pid_path.read_text(encoding="ascii"))

    child_pid: int | None = None
    try:
        child_pid = asyncio.run(exercise())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("repeat_cancellation", [False, True])
def test_cancellation_during_timeout_reap_preserves_cancellation_and_containment(
    tmp_path: Path,
    repeat_cancellation: bool,
) -> None:
    environment = _initialized_local_environment(tmp_path)
    environment._local_command_guardrail_reason = lambda *_args: ""  # type: ignore[method-assign]
    child_pid_path = environment._workspace / "timeout-reap-cancel-child-pid"
    reap_entered = asyncio.Event()
    original_terminate = environment._terminate_process_tree
    command = (
        "(trap '' TERM; exec >/dev/null 2>&1; sleep 30) & "
        f"printf '%s' \"$!\" > {shlex.quote(child_pid_path.name)}; wait"
    )

    async def observed_terminate(
        proc: asyncio.subprocess.Process,
        communication: asyncio.Task[tuple[bytes, bytes]] | None = None,
    ) -> tuple[bytes, bytes]:
        reap_entered.set()
        return await original_terminate(proc, communication)

    async def on_output(_text: str, _stream: str) -> None:
        return None

    async def exercise() -> tuple[BaseException | None, int, bool]:
        await environment.start()
        environment._terminate_process_tree = observed_terminate  # type: ignore[method-assign]
        with environment.scoped_output_callback(on_output):
            task = asyncio.create_task(environment.exec(command, timeout_sec=0.5))
            await asyncio.wait_for(reap_entered.wait(), timeout=5)
            assert child_pid_path.exists()
            task.cancel()
            if repeat_cancellation:
                await asyncio.sleep(0)
                task.cancel()
            outcome: BaseException | None = None
            try:
                await asyncio.wait_for(task, timeout=5)
            except BaseException as exc:
                outcome = exc
        retained_after_exec = bool(environment._active_processes)
        if retained_after_exec:
            await environment.stop(delete=False)
        return outcome, int(child_pid_path.read_text(encoding="ascii")), retained_after_exec

    child_pid: int | None = None
    try:
        outcome, child_pid, retained = asyncio.run(exercise())
        assert isinstance(outcome, asyncio.CancelledError)
        assert not retained
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("repeat_cancellation", [False, True])
def test_cancellation_remains_primary_when_timeout_reap_fails(
    tmp_path: Path,
    repeat_cancellation: bool,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    environment = _initialized_local_environment(tmp_path)
    original_terminate = environment._terminate_process_tree
    reap_entered = asyncio.Event()
    release_failure = asyncio.Event()
    secret = "timeout-reap-cleanup-secret"

    async def failing_terminate(
        _proc: asyncio.subprocess.Process,
        _communication: asyncio.Task[tuple[bytes, bytes]] | None = None,
    ) -> tuple[bytes, bytes]:
        async def fail_after_release() -> tuple[bytes, bytes]:
            await release_failure.wait()
            raise PermissionError(errno.EACCES, "Denied", secret)

        cleanup = asyncio.create_task(fail_after_release())
        reap_entered.set()
        return await local_environment._await_task_uninterruptibly(cleanup)

    async def on_output(_text: str, _stream: str) -> None:
        return None

    async def exercise() -> tuple[BaseException | None, bool]:
        await environment.start()
        environment._terminate_process_tree = failing_terminate  # type: ignore[method-assign]
        with environment.scoped_output_callback(on_output):
            task = asyncio.create_task(
                environment.exec(
                    "sleep 30",
                    env={"API_KEY": secret},
                    timeout_sec=0.02,
                )
            )
            await asyncio.wait_for(reap_entered.wait(), timeout=2)
            task.cancel()
            if repeat_cancellation:
                await asyncio.sleep(0)
                task.cancel()
            release_failure.set()
            outcome: BaseException | None = None
            try:
                await asyncio.wait_for(task, timeout=1)
            except BaseException as exc:
                outcome = exc
        retained = bool(environment._active_processes)
        environment._terminate_process_tree = original_terminate  # type: ignore[method-assign]
        await environment.stop(delete=False)
        return outcome, retained

    outcome, retained = asyncio.run(exercise())

    assert isinstance(outcome, asyncio.CancelledError)
    assert isinstance(outcome.__cause__, RuntimeError)
    assert outcome.__cause__.__context__ is None
    assert retained
    assert secret not in str(outcome.__cause__)
    assert secret not in "".join(traceback.format_exception(outcome))
    assert not environment._active_processes


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("lifecycle", ["timeout", "cancel", "repeat-cancel"])
def test_streamed_exec_escalates_after_launcher_exits_with_term_ignoring_descendant(
    tmp_path: Path,
    lifecycle: str,
) -> None:
    environment = _initialized_local_environment(tmp_path)
    environment._local_command_guardrail_reason = lambda *_args: ""  # type: ignore[method-assign]
    child_pid_path = environment._workspace / f"term-ignoring-{lifecycle}.pid"
    command = (
        "(trap '' TERM; exec >/dev/null 2>&1; sleep 30) & "
        f"printf '%s' \"$!\" > {shlex.quote(child_pid_path.name)}; "
        "printf ready; wait"
    )

    async def on_output(_text: str, _stream: str) -> None:
        return None

    async def exercise() -> object | None:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            task = asyncio.create_task(environment.exec(command, timeout_sec=0.2 if lifecycle == "timeout" else None))
            for _ in range(500):
                if child_pid_path.exists():
                    break
                await asyncio.sleep(0.01)
            assert child_pid_path.exists()
            if lifecycle == "timeout":
                result = await asyncio.wait_for(task, timeout=5)
                assert result.return_code == 124
                return result
            task.cancel()
            if lifecycle == "repeat-cancel":
                await asyncio.sleep(0)
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            return None

    child_pid: int | None = None
    try:
        asyncio.run(exercise())
        child_pid = int(child_pid_path.read_text(encoding="ascii"))
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
        if child_pid is not None:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                pass
            else:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child_pid, signal.SIGKILL)


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
def test_process_tree_cleanup_suppresses_permission_race_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 4242
        returncode = None

    async def communication() -> tuple[bytes, bytes]:
        return b"stdout", b"stderr"

    monkeypatch.setattr(
        os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError("leader exited")),
    )
    monkeypatch.setattr(
        os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(ProcessLookupError()),
    )

    async def exercise() -> tuple[bytes, bytes]:
        task = asyncio.create_task(communication())
        return await SkillEvaluatorLocalEnvironment._terminate_process_tree(  # type: ignore[arg-type]
            FakeProcess(),
            task,
        )

    result = asyncio.run(exercise())

    assert result == (b"stdout", b"stderr")


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_process_tree_cleanup_propagates_live_permission_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 4242
        returncode = None

    async def communication() -> tuple[bytes, bytes]:
        await asyncio.Event().wait()
        return b"", b""

    monkeypatch.setattr(
        os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError("live process denied")),
    )
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    async def exercise() -> None:
        task = asyncio.create_task(communication())
        try:
            with pytest.raises(PermissionError, match="live process denied"):
                await SkillEvaluatorLocalEnvironment._terminate_process_tree(  # type: ignore[arg-type]
                    FakeProcess(),
                    task,
                )
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(exercise())


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_process_tree_cleanup_stays_bounded_when_communication_ignores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    monkeypatch.setattr(local_environment, "_REAP_TERM_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_KILL_SECONDS", 0.01)
    monkeypatch.setattr(local_environment, "_REAP_CANCEL_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(local_environment.os, "killpg", lambda *_args: None)

    async def run_cleanup() -> tuple[bool, bool]:
        started = asyncio.Event()

        async def stubborn_communication() -> tuple[bytes, bytes]:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.Event().wait()
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
        await cleanup
        return finished_within_bound, communication.cancelled()

    assert asyncio.run(run_cleanup()) == (True, True)


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
def test_exec_cancellation_does_not_wait_forever_for_uncooperative_process_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    environment = _local_environment(tmp_path)
    create_subprocess_exec = asyncio.create_subprocess_exec
    monkeypatch.setattr(local_environment, "_CREATION_CANCEL_SECONDS", 0.02, raising=False)

    async def exercise() -> tuple[bool, asyncio.subprocess.Process]:
        process_created = asyncio.Event()
        release_creation = asyncio.Event()
        created: list[asyncio.subprocess.Process] = []

        async def uncooperative_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            process = await create_subprocess_exec(*args, **kwargs)
            created.append(process)
            process_created.set()
            try:
                await release_creation.wait()
            except asyncio.CancelledError:
                await release_creation.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", uncooperative_create)
        task = asyncio.create_task(environment.exec("sleep 30"))
        await asyncio.wait_for(process_created.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        done, _pending = await asyncio.wait({task}, timeout=0.2)
        returned_within_bound = task in done
        release_creation.set()
        if task not in done:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(asyncio.CancelledError):
                task.result()
        for _ in range(500):
            if created[0].returncode is not None:
                break
            await asyncio.sleep(0.01)
        return returned_within_bound, created[0]

    returned_within_bound, process = asyncio.run(exercise())

    assert returned_within_bound
    assert process.returncode is not None


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_stop_fails_closed_until_withheld_process_creation_is_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import local_environment

    environment = _local_environment(tmp_path)
    create_subprocess_exec = asyncio.create_subprocess_exec
    monkeypatch.setattr(local_environment, "_CREATION_CANCEL_SECONDS", 0.02)
    created: list[asyncio.subprocess.Process] = []

    async def exercise() -> tuple[asyncio.subprocess.Process, asyncio.subprocess.Process, bool, bool]:
        process_created = asyncio.Event()
        release_creation = asyncio.Event()

        active_process = await create_subprocess_exec(
            "bash",
            "-c",
            "sleep 30",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        active_communication = asyncio.create_task(active_process.communicate())
        environment._active_processes[active_process] = active_communication

        async def withheld_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            process = await create_subprocess_exec(*args, **kwargs)
            created.append(process)
            process_created.set()
            await release_creation.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", withheld_create)
        exec_task = asyncio.create_task(environment.exec("sleep 30"))
        await asyncio.wait_for(process_created.wait(), timeout=5)
        exec_task.cancel()
        await asyncio.sleep(0)
        exec_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(exec_task, timeout=0.2)

        with pytest.raises(RuntimeError, match="could not confirm process creation containment"):
            await asyncio.wait_for(environment.stop(delete=True), timeout=0.2)
        root_preserved_while_unresolved = environment._root.exists()
        process_alive_while_stop_failed = created[0].returncode is None
        assert active_process.returncode is not None
        assert active_process not in environment._active_processes

        release_creation.set()
        for _ in range(500):
            if created[0].returncode is not None and not environment._creation_cleanups:
                break
            await asyncio.sleep(0.01)
        await environment.stop(delete=True)
        return created[0], active_process, root_preserved_while_unresolved, process_alive_while_stop_failed

    process: asyncio.subprocess.Process | None = None
    active_process: asyncio.subprocess.Process | None = None
    try:
        process, active_process, root_preserved, process_was_alive = asyncio.run(exercise())
        assert root_preserved
        assert process_was_alive
        assert process.returncode is not None
        assert not environment._root.exists()
        assert not environment._pending_creations
        assert not environment._creation_cleanups
    finally:
        leaked_process = process or (created[0] if created else None)
        for process_to_reap in (leaked_process, active_process):
            if process_to_reap is not None and process_to_reap.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process_to_reap.pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize("secret", ["PermissionError", "Local process creation cleanup failed"])
def test_failed_late_creation_cleanup_retains_process_for_stop_retry(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    secret: str,
) -> None:
    environment = _local_environment(tmp_path)
    original_terminate = environment._terminate_process_tree
    process: asyncio.subprocess.Process | None = None
    caplog.set_level(logging.ERROR, logger="asyncio")

    async def exercise() -> tuple[bool, bool, bool]:
        nonlocal process
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            "sleep 30",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        creation = asyncio.create_task(asyncio.sleep(0, result=process))
        environment._pending_creations.add(creation)
        cleanup_attempts = 0

        async def fail_twice_then_reap(
            proc: asyncio.subprocess.Process,
            communication: asyncio.Task[tuple[bytes, bytes]] | None = None,
        ) -> tuple[bytes, bytes]:
            nonlocal cleanup_attempts
            cleanup_attempts += 1
            if cleanup_attempts <= 2:
                raise PermissionError(errno.EACCES, "Denied", secret)
            return await original_terminate(proc, communication)

        environment._terminate_process_tree = fail_twice_then_reap  # type: ignore[method-assign]
        cleanup = environment._schedule_creation_cleanup(creation, secret_values={secret})
        with pytest.raises(PermissionError):
            await cleanup
        await asyncio.sleep(0)
        retained_after_late_failure = process in environment._active_processes

        with pytest.raises(RuntimeError, match="cleanup is still pending"):
            await environment.start()
        with pytest.raises(RuntimeError, match="could not confirm process creation containment") as caught:
            await environment.stop(delete=True)
        first_stop_was_redacted = secret not in str(caught.value) and environment._root.exists()

        await environment.stop(delete=True)
        return retained_after_late_failure, first_stop_was_redacted, process.returncode is not None

    try:
        retained, first_stop_redacted, reaped = asyncio.run(exercise())
        assert retained
        assert first_stop_redacted
        assert secret not in caplog.text
        assert reaped
        assert not environment._active_processes
        assert not environment._creation_cleanup_errors
        assert not environment._root.exists()
    finally:
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason=_NATIVE_WINDOWS_LOCAL_REASON)
def test_creation_completing_during_stop_redacts_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _local_environment(tmp_path)
    create_subprocess_exec = asyncio.create_subprocess_exec
    original_terminate = environment._terminate_process_tree
    secret = "stop-race-cleanup-secret"

    async def stop_after_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await create_subprocess_exec(*args, **kwargs)
        environment._stop_requested = True
        return process

    async def fail_before_reap(
        _proc: asyncio.subprocess.Process,
        _communication: asyncio.Task[tuple[bytes, bytes]] | None = None,
    ) -> tuple[bytes, bytes]:
        raise PermissionError(errno.EACCES, "Denied", secret)

    async def exercise() -> tuple[BaseException, bool]:
        monkeypatch.setattr(asyncio, "create_subprocess_exec", stop_after_create)
        environment._terminate_process_tree = fail_before_reap  # type: ignore[method-assign]
        with pytest.raises(RuntimeError) as caught:
            await environment.exec("sleep 30", env={"API_KEY": secret})
        retained = bool(environment._active_processes)
        environment._terminate_process_tree = original_terminate  # type: ignore[method-assign]
        await environment.stop(delete=False)
        return caught.value, retained

    caught, retained = asyncio.run(exercise())

    assert retained
    assert caught.__context__ is None
    assert secret not in str(caught)
    assert secret not in "".join(traceback.format_exception(caught))
    assert not environment._active_processes


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
@pytest.mark.parametrize(
    ("command", "cwd", "env", "secret"),
    [
        ("true", "/tmp/ordinary-secret-path", {"API_KEY": "/tmp/ordinary-secret-path"}, "/tmp/ordinary-secret-path"),
        ("true", "/tmp/sk-Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8", None, "sk-Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8"),
        ('touch "$SECRET_PATH"', None, {"SECRET_PATH": "/tmp/guardrail-secret-path"}, "/tmp/guardrail-secret-path"),
        ("rm -rf /", None, {"API_KEY": "Local mode command blocked"}, "Local mode command blocked"),
        ('touch "$API_KEY"', None, {"API_KEY": "/x"}, "/x"),
    ],
)
def test_prelaunch_diagnostics_are_streamed_and_redacted_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    command: str,
    cwd: str | None,
    env: dict[str, str] | None,
    secret: str,
) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_chunks: list[tuple[str, str]] = []
    caplog.set_level(logging.WARNING)

    async def on_output(text: str, stream: str) -> None:
        callback_chunks.append((text, stream))

    async def exercise() -> object:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            return await environment.exec(command, cwd=cwd, env=env)

    result = asyncio.run(exercise())
    callback_stderr = "".join(text for text, stream in callback_chunks if stream == "stderr")

    assert result.return_code == 126
    assert callback_stderr == result.stderr
    assert secret not in (result.stderr or "")
    assert secret not in callback_stderr
    assert secret not in caplog.text


@pytest.mark.skipif(os.name == "nt", reason=_NATIVE_WINDOWS_LOCAL_REASON)
@pytest.mark.parametrize(
    ("name", "value", "is_sensitive"),
    [
        ("MONKEY", "banana", False),
        ("KEYBOARD", "clacky", False),
        ("API_KEY", "x", True),
    ],
)
def test_output_redaction_uses_component_aware_sensitive_environment_names(
    tmp_path: Path,
    name: str,
    value: str,
    is_sensitive: bool,
) -> None:
    environment = _initialized_local_environment(tmp_path)
    callback_chunks: list[str] = []

    async def on_output(text: str, _stream: str) -> None:
        callback_chunks.append(text)

    async def exercise() -> object:
        await environment.start()
        with environment.scoped_output_callback(on_output):
            return await environment.exec(f'printf %s "${name}"', env={name: value})

    result = asyncio.run(exercise())
    callback_output = "".join(callback_chunks)

    assert callback_output == result.stdout
    assert (value not in callback_output) is is_sensitive
    if not is_sensitive:
        assert callback_output == value


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
