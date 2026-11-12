# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import stat
import subprocess
import sys
import venv
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from harbor.models.task.config import ArtifactConfig, EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.artifact_handler import ArtifactHandler
from harbor.trial.trial import Trial
from harbor.verifier.verifier import Verifier

from skillevaluator.tier3.harbor import local_sandbox
from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment


@pytest.fixture
def local_output_environment(tmp_path: Path) -> SkillEvaluatorLocalEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    return SkillEvaluatorLocalEnvironment(
        environment_dir=environment_dir,
        environment_name="local-outputs",
        session_id="local-outputs",
        trial_paths=paths,
        task_env_config=EnvironmentConfig(),
        runtime_agent="codex",
        runtime_root=str(runtime_root),
        sandbox_mode="off",
    )


@pytest.mark.parametrize("role", ["agent", "user-agent"])
@pytest.mark.parametrize("direction", ["upload", "download"])
def test_harbor_role_log_lifecycle_preserves_local_files(
    local_output_environment: SkillEvaluatorLocalEnvironment,
    role: str,
    direction: str,
) -> None:
    """Harbor must recognize direct host mappings before copying role logs."""
    environment = local_output_environment
    host_logs = environment.trial_paths.trial_dir / role
    host_logs.mkdir(exist_ok=True)
    retained_log = host_logs / "session.jsonl"
    retained_log.write_text('{"message":"retained"}\n', encoding="utf-8")
    trial = SimpleNamespace(agent_environment=environment, logger=logging.getLogger(__name__))
    sandbox_logs = PurePosixPath("/logs") / role

    if direction == "upload":
        asyncio.run(Trial._upload_role_logs(trial, source_dir=host_logs, target_dir=sandbox_logs))
    else:
        asyncio.run(
            Trial._download_role_logs(
                trial,
                agent_config=AgentConfig(name="codex"),
                source_dir=sandbox_logs,
                target_dir=host_logs,
            )
        )

    assert retained_log.exists(), "Copying local logs onto themselves deleted the session"
    assert retained_log.read_text(encoding="utf-8") == '{"message":"retained"}\n'


def test_harbor_collects_local_convention_artifacts_without_recursive_copy(
    local_output_environment: SkillEvaluatorLocalEnvironment,
) -> None:
    environment = local_output_environment
    published_file = environment._resolve_path("/logs/artifacts/answer.txt")
    published_file.parent.mkdir(parents=True, exist_ok=True)
    published_file.write_text("agent answer", encoding="utf-8")
    assert Path(environment._exec_env({})["HARBOR_ARTIFACTS_DIR"]) == published_file.parent
    handler = ArtifactHandler(artifacts=[], logger=logging.getLogger(__name__))

    entry = asyncio.run(
        handler._download_artifact(
            source_env=environment,
            artifacts_dir=environment.trial_paths.artifacts_dir,
            artifact=ArtifactConfig(source="/logs/artifacts"),
            convention_source="/logs/artifacts",
        )
    )

    assert entry.status == "ok"
    recorded_file = environment.trial_paths.artifacts_dir / "logs" / "artifacts" / "answer.txt"
    assert recorded_file.read_text(encoding="utf-8") == "agent answer"
    assert published_file.read_text(encoding="utf-8") == "agent answer"


@pytest.mark.skipif(os.name == "nt", reason="Native Windows local mode is unsupported")
def test_harbor_verifier_preserves_local_reward_and_stdout(
    local_output_environment: SkillEvaluatorLocalEnvironment,
) -> None:
    environment = local_output_environment
    task_dir = environment.environment_dir.parent
    (task_dir / "task.toml").write_text('[verifier]\nenvironment_mode = "shared"\n', encoding="utf-8")
    (task_dir / "instruction.md").write_text("Evaluate the local reward lifecycle.", encoding="utf-8")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        '#!/bin/sh\nprintf \'{"reward":1}\' > /logs/verifier/reward.json\nprintf "verification complete\\n"\n',
        encoding="utf-8",
    )
    verifier = Verifier(Task(task_dir), environment.trial_paths, environment)

    async def verify():
        await environment.start()
        try:
            return await verifier.verify()
        finally:
            await environment.stop(delete=False)

    result = asyncio.run(verify())

    assert result.rewards == {"reward": 1}
    assert environment.trial_paths.reward_json_path.read_text(encoding="utf-8") == '{"reward":1}'
    assert "verification complete" in environment.trial_paths.test_stdout_path.read_text(encoding="utf-8")


def test_local_path_mount_detection_requires_matching_host_destination(
    local_output_environment: SkillEvaluatorLocalEnvironment,
    tmp_path: Path,
) -> None:
    environment = local_output_environment
    logs = environment.trial_paths.agent_dir

    assert environment.is_path_mounted("/logs/agent", logs)
    assert environment.is_path_mounted(PurePosixPath("/logs/agent/session.jsonl"), logs / "session.jsonl")
    assert not environment.is_path_mounted("/logs/agent", environment.trial_paths.verifier_dir)
    assert not environment.is_path_mounted("/logs", environment.trial_paths.trial_dir)
    assert not environment.is_path_mounted("/logs/agent-other", logs)
    assert not environment.is_path_mounted("/etc/passwd", tmp_path / "passwd")


@pytest.mark.skipif(os.name == "nt", reason="Native Windows local mode is unsupported")
def test_verifier_python_keeps_virtualenv_dependencies(monkeypatch, tmp_path: Path) -> None:
    """Resolving the interpreter symlink loses the venv's installed dependencies."""
    runtime = tmp_path / "evaluator venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
    site_dir = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip()
    (Path(site_dir) / "verifier_dependency.py").write_text("VALUE = 'available'\n")
    environment = SkillEvaluatorLocalEnvironment.__new__(SkillEvaluatorLocalEnvironment)
    environment._runtime_root = tmp_path / "agents"
    environment._runtime_agent = "codex"
    monkeypatch.setattr(sys, "executable", str(python))

    env = {**os.environ, "PATH": environment._path_with_evaluator_python(os.defpath)}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["python3", "-c", "import verifier_dependency; print(verifier_dependency.VALUE)"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "available"


def test_local_environment_rewrites_read_only_uploaded_script_once(tmp_path: Path) -> None:
    """A read-only template is made writable without rewriting its local path twice."""
    source = tmp_path / "template_eval.py"
    source.write_text('print("/tests/eval.py")\n', encoding="utf-8")
    source.chmod(0o444)

    local_tests = tmp_path / "trial" / "local-environment" / "tests"
    uploaded = local_tests / "eval.py"
    uploaded.parent.mkdir(parents=True)
    shutil.copy2(source, uploaded)

    should_restore_owner_write = not os.access(uploaded, os.W_OK)
    environment = SkillEvaluatorLocalEnvironment.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/tests", local_tests)]

    environment._rewrite_uploaded_script(uploaded)

    assert uploaded.read_text(encoding="utf-8") == f'print("{local_tests}{os.sep}eval.py")\n'
    mode = uploaded.stat().st_mode
    if should_restore_owner_write:
        assert mode & stat.S_IWUSR
    if os.name == "posix":
        assert not mode & stat.S_IWGRP
        assert not mode & stat.S_IWOTH


def test_raw_path_rewrite_respects_container_root_boundaries(tmp_path: Path) -> None:
    local_tests = tmp_path / "local" / "tests"
    environment = SkillEvaluatorLocalEnvironment.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/tests", local_tests)]

    rewritten = environment._rewrite_raw_paths('"/tests/eval.py" "/tests" "/testsuite" "/tests-v2"')

    assert rewritten == f'"{local_tests}{os.sep}eval.py" "{local_tests}" "/testsuite" "/tests-v2"'


def test_raw_path_rewrite_ignores_url_and_local_path_suffixes(tmp_path: Path) -> None:
    local_tests = tmp_path / "local" / "tests"
    environment = SkillEvaluatorLocalEnvironment.__new__(SkillEvaluatorLocalEnvironment)
    environment._path_map = lambda: [("/tests", local_tests)]
    value = f'"/tests/api" "https://example.invalid/tests/api" "{local_tests}/api"'

    rewritten = environment._rewrite_raw_paths(value)

    assert rewritten == f'"{local_tests}{os.sep}api" "https://example.invalid/tests/api" "{local_tests}/api"'


@pytest.mark.integration
@pytest.mark.skipif(os.name == "nt", reason="local subprocess backend requires POSIX")
@pytest.mark.parametrize("secret_name", ["SANDBOX_TOKEN", "MYSECRET", "SOMEKEY", "AUTHENTICATION"])
def test_real_local_exec_streams_redacted_stdout_and_stderr(
    tmp_path: Path,
    secret_name: str,
) -> None:
    try:
        detected_sandbox = local_sandbox.detect("require")
    except local_sandbox.SandboxUnavailable as exc:
        pytest.skip(str(exc))

    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    environment = SkillEvaluatorLocalEnvironment(
        environment_dir=environment_dir,
        environment_name="real-local-streaming",
        session_id="real-local-streaming",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        runtime_agent="opencode",
        runtime_root=str(runtime_root),
        sandbox_mode="require",
        allow_net=False,
        strict_reads=True,
    )
    secret = "real-sandbox-stream-secret"
    callbacks: list[tuple[str, str]] = []
    command = f"printf 'stdout=%s\\n' \"${secret_name}\"; printf 'stderr=%s\\n' \"${secret_name}\" >&2"

    async def on_output(text: str, stream: str) -> None:
        callbacks.append((text, stream))

    async def exercise() -> tuple[object, str]:
        await environment.start()
        try:
            with environment.scoped_output_callback(on_output):
                result = await environment.exec(command, env={secret_name: secret})
            assert environment._sandbox is not None
            return result, environment._sandbox.plan.backend
        finally:
            await environment.stop(delete=False)

    result, backend = asyncio.run(exercise())
    callback_stdout = "".join(text for text, stream in callbacks if stream == "stdout")
    callback_stderr = "".join(text for text, stream in callbacks if stream == "stderr")

    assert backend == detected_sandbox.plan.backend
    assert backend in {"bubblewrap", "seatbelt"}
    assert callback_stdout == result.stdout
    assert callback_stderr == result.stderr
    assert secret not in callback_stdout
    assert secret not in callback_stderr
    assert secret not in (result.stdout or "")
    assert secret not in (result.stderr or "")
    assert "stdout=" in callback_stdout
    assert "stderr=" in callback_stderr
