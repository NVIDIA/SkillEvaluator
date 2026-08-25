# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import shutil
import stat
from pathlib import Path

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from skillevaluator.tier3.harbor import local_sandbox
from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment


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
def test_real_local_exec_streams_redacted_stdout_and_stderr(tmp_path: Path) -> None:
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
    command = "printf 'stdout=%s\\n' \"$SANDBOX_TOKEN\"; printf 'stderr=%s\\n' \"$SANDBOX_TOKEN\" >&2"

    async def on_output(text: str, stream: str) -> None:
        callbacks.append((text, stream))

    async def exercise() -> tuple[object, str]:
        await environment.start()
        try:
            with environment.scoped_output_callback(on_output):
                result = await environment.exec(command, env={"SANDBOX_TOKEN": secret})
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
    assert "stdout=" in callback_stdout
    assert "stderr=" in callback_stderr
