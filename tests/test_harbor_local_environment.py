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


@pytest.mark.parametrize("destination", [None, "logs/artifacts"])
def test_harbor_collects_local_convention_artifacts_without_recursive_copy(
    local_output_environment: SkillEvaluatorLocalEnvironment,
    destination: str | None,
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
            artifact=ArtifactConfig(source="/logs/artifacts", destination=destination),
            convention_source="/logs/artifacts",
        )
    )

    assert entry.status == "ok"
    recorded_file = environment.trial_paths.artifacts_dir / "logs" / "artifacts" / "answer.txt"
    assert published_file != recorded_file
    assert not environment.is_path_mounted("/logs/artifacts", recorded_file.parent)
    assert recorded_file.read_text(encoding="utf-8") == "agent answer"
    assert published_file.read_text(encoding="utf-8") == "agent answer"
    asyncio.run(environment.stop(delete=True))
    assert not published_file.exists()
    assert recorded_file.read_text(encoding="utf-8") == "agent answer"


@pytest.mark.skipif(os.name == "nt", reason="Native Windows local mode is unsupported")
def test_local_exec_writes_and_reads_logical_artifact_paths(
    local_output_environment: SkillEvaluatorLocalEnvironment,
) -> None:
    environment = local_output_environment

    async def exercise():
        await environment.start()
        try:
            return await environment.exec(
                "printf agent-answer > /logs/artifacts/answer.txt && "
                'cat /logs/artifacts/answer.txt && cat "$HARBOR_ARTIFACTS_DIR/answer.txt"'
            )
        finally:
            await environment.stop(delete=False)

    result = asyncio.run(exercise())

    assert result.return_code == 0, result.stderr
    assert result.stdout == "agent-answeragent-answer"
    assert environment._resolve_path("/logs/artifacts/answer.txt").read_text(encoding="utf-8") == "agent-answer"


@pytest.mark.skipif(os.name == "nt", reason="Native Windows local mode is unsupported")
@pytest.mark.parametrize("destination", [None, "logs/artifacts"])
def test_local_artifact_parent_paths_publish_and_collect_step_snapshots(
    local_output_environment: SkillEvaluatorLocalEnvironment,
    destination: str | None,
) -> None:
    """Computing the artifact child from /logs must reach the publish directory."""
    environment = local_output_environment
    handler = ArtifactHandler(artifacts=[], logger=logging.getLogger(__name__))
    filenames = ("parent.txt", "relative.txt", "computed.txt")

    async def exercise() -> None:
        await environment.start()
        try:
            for step_name, answer in (("first", "first-answer"), ("second", "second-answer")):
                result = await environment.exec(
                    f'printf {answer} > "$HARBOR_LOGS_DIR/artifacts/parent.txt" && '
                    f"cd /logs && printf {answer} > artifacts/relative.txt && "
                    f'logs=/logs; child=artifacts; printf {answer} > "$logs/$child/computed.txt"; '
                    'cat "$HARBOR_ARTIFACTS_DIR/parent.txt" '
                    '"$HARBOR_ARTIFACTS_DIR/relative.txt" "$HARBOR_ARTIFACTS_DIR/computed.txt"'
                )
                assert result.return_code == 0, result.stderr
                assert result.stdout == answer * 3
                handler.begin_collection()
                entry = await handler._download_artifact(
                    source_env=environment,
                    artifacts_dir=environment.trial_paths.step_artifacts_dir(step_name),
                    artifact=ArtifactConfig(source="/logs/artifacts", destination=destination),
                    convention_source="/logs/artifacts",
                )
                assert entry.status == "ok"
                for filename in filenames:
                    assert environment._resolve_path(f"/logs/artifacts/{filename}").read_text() == answer
        finally:
            await environment.stop(delete=True)

    asyncio.run(exercise())

    assert not environment._root.exists()
    for step_name, answer in (("first", "first-answer"), ("second", "second-answer")):
        recorded_dir = environment.trial_paths.step_artifacts_dir(step_name) / "logs" / "artifacts"
        for filename in filenames:
            assert (recorded_dir / filename).read_text() == answer


@pytest.mark.skipif(os.name == "nt", reason="Native Windows local mode is unsupported")
def test_local_parent_logs_paths_preserve_retained_role_logs(
    local_output_environment: SkillEvaluatorLocalEnvironment,
) -> None:
    environment = local_output_environment
    roles = ("agent", "verifier", "user-agent")
    for role in roles:
        (environment.trial_paths.trial_dir / role).mkdir(exist_ok=True)

    async def exercise() -> None:
        await environment.start()
        try:
            result = await environment.exec(
                "logs=/logs; for role in agent verifier user-agent; do "
                'printf retained > "$logs/$role/session.jsonl" || exit; done'
            )
            assert result.return_code == 0, result.stderr
            for role in roles:
                host_logs = environment.trial_paths.trial_dir / role
                assert environment.is_path_mounted(f"/logs/{role}", host_logs)
                assert (host_logs / "session.jsonl").read_text() == "retained"
        finally:
            await environment.stop(delete=True)

    asyncio.run(exercise())

    for role in roles:
        assert (environment.trial_paths.trial_dir / role / "session.jsonl").read_text() == "retained"


@pytest.mark.skipif(os.name == "nt", reason="Native Windows local mode is unsupported")
@pytest.mark.parametrize("transfer", ["download", "excluded", "filtered", "upload"])
def test_local_parent_logs_transfers_fail_before_mutating_files(
    local_output_environment: SkillEvaluatorLocalEnvironment,
    tmp_path: Path,
    transfer: str,
) -> None:
    """The virtual parent must never copy its internal aliases as user artifacts."""
    environment = local_output_environment
    target = tmp_path / "downloaded-logs"
    target.mkdir()
    marker = target / "retained.txt"
    marker.write_text("previous download")
    role_log = environment.trial_paths.agent_dir / "session.jsonl"
    role_log.write_text("retained role log")

    async def exercise() -> None:
        await environment.start()
        try:
            with pytest.raises(ValueError, match="logs parent"):
                if transfer == "download":
                    await environment.download_dir("/logs", target)
                elif transfer == "excluded":
                    await environment.download_dir_with_exclusions(source_dir="/logs", target_dir=target, exclude=[])
                elif transfer == "filtered":
                    await environment.download_dir_filtered(source_dir="/logs", target_dir=target, include=["*"])
                else:
                    await environment.upload_dir(target, "/logs")
            assert marker.read_text() == "previous download"
            assert role_log.read_text() == "retained role log"
        finally:
            await environment.stop(delete=True)

    asyncio.run(exercise())
    assert marker.read_text() == "previous download"
    assert role_log.read_text() == "retained role log"


@pytest.mark.skipif(os.name == "nt", reason="Native Windows local mode is unsupported")
def test_local_role_transfer_normalizes_logical_path_spelling(
    local_output_environment: SkillEvaluatorLocalEnvironment,
    tmp_path: Path,
) -> None:
    environment = local_output_environment
    source = environment.trial_paths.agent_dir / "session.jsonl"
    source.write_text("retained role log", encoding="utf-8")
    target = tmp_path / "downloaded-role"

    async def exercise() -> None:
        await environment.start()
        try:
            await environment.download_dir("/logs/./agent", target)
        finally:
            await environment.stop(delete=True)

    asyncio.run(exercise())
    assert (target / "session.jsonl").read_text(encoding="utf-8") == "retained role log"
    assert source.read_text(encoding="utf-8") == "retained role log"


def test_local_artifact_collection_retains_source_and_separate_step_snapshots(
    local_output_environment: SkillEvaluatorLocalEnvironment,
) -> None:
    environment = local_output_environment
    source = environment._resolve_path("/logs/artifacts/answer.txt")
    source.parent.mkdir(parents=True, exist_ok=True)
    handler = ArtifactHandler(artifacts=[], logger=logging.getLogger(__name__))

    async def collect_steps() -> None:
        for step_name, answer in (("first", "first answer"), ("second", "second answer")):
            source.write_text(answer, encoding="utf-8")
            handler.begin_collection()
            entry = await handler._download_artifact(
                source_env=environment,
                artifacts_dir=environment.trial_paths.step_artifacts_dir(step_name),
                artifact=ArtifactConfig(source="/logs/artifacts", destination="logs/artifacts"),
                convention_source="/logs/artifacts",
            )
            assert entry.status == "ok"
            assert source.read_text(encoding="utf-8") == answer

    asyncio.run(collect_steps())

    for step_name, answer in (("first", "first answer"), ("second", "second answer")):
        recorded = environment.trial_paths.step_artifacts_dir(step_name) / "logs" / "artifacts" / "answer.txt"
        assert recorded.read_text(encoding="utf-8") == answer


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


def _contained_transfer_payload() -> dict[str, bytes]:
    return {
        "keep.txt": b"root contents\n",
        "omit.log": b"excluded root log\n",
        "protected.log": b"protected contents\n",
        "nested/keep.txt": b"nested contents\n",
        "nested/omit.log": b"excluded nested log\n",
        "nested/deeper/keep.txt": b"deeper contents\n",
        "odd [x] ' café.txt": b"literal odd filename\n",
        "--literal.txt": b"literal leading dash\n",
        "line\nbreak.txt": b"literal newline filename\n",
        "blob.bin": bytes(range(256)),
    }


def _contained_transfer_files(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.skipif(os.name == "nt", reason="real local subprocess backend requires POSIX")
@pytest.mark.parametrize(
    ("exclude", "expected"),
    [
        (
            ["*.log"],
            {
                "keep.txt",
                "nested/keep.txt",
                "nested/deeper/keep.txt",
                "odd [x] ' café.txt",
                "--literal.txt",
                "line\nbreak.txt",
                "blob.bin",
            },
        ),
        (
            ["nested"],
            {
                "keep.txt",
                "omit.log",
                "protected.log",
                "odd [x] ' café.txt",
                "--literal.txt",
                "line\nbreak.txt",
                "blob.bin",
            },
        ),
    ],
)
@pytest.mark.parametrize("service_wrapper", [False, True], ids=["direct", "main-service"])
def test_real_contained_exclusion_download_preserves_contents_and_destination(
    local_output_environment: SkillEvaluatorLocalEnvironment,
    tmp_path: Path,
    exclude: list[str],
    expected: set[str],
    service_wrapper: bool,
) -> None:
    environment = local_output_environment
    payload = _contained_transfer_payload()
    target = tmp_path / "downloaded"
    target.mkdir()
    (target / "retained.txt").write_bytes(b"existing unrelated output\n")
    (target / "keep.txt").write_bytes(b"stale selected output\n")

    async def exercise() -> None:
        await environment.start()
        try:
            source = environment._resolve_path("/workspace/export")
            for name, data in payload.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            before = set(environment._root.rglob("*"))
            if service_wrapper:
                await environment.service_download_dir_with_exclusions(
                    source_dir="/workspace/export", target_dir=target, exclude=exclude, service=None
                )
            else:
                await environment.download_dir_with_exclusions(
                    source_dir="/workspace/export", target_dir=target, exclude=exclude
                )
            assert _contained_transfer_files(target) == {
                **{name: payload[name] for name in expected},
                "retained.txt": b"existing unrelated output\n",
            }
            assert _contained_transfer_files(source) == payload
            assert set(environment._root.rglob("*")) == before
        finally:
            await environment.stop(delete=False)

    asyncio.run(exercise())


@pytest.mark.skipif(os.name == "nt", reason="real local subprocess backend requires POSIX")
@pytest.mark.parametrize("no_match", [False, True], ids=["include-exclude-protect", "no-match"])
def test_real_contained_filtered_download_preserves_literal_names_and_source(
    local_output_environment: SkillEvaluatorLocalEnvironment, tmp_path: Path, no_match: bool
) -> None:
    environment = local_output_environment
    payload = _contained_transfer_payload()
    target = tmp_path / "filtered"
    target.mkdir()
    (target / "retained.txt").write_bytes(b"existing unrelated output\n")

    async def exercise() -> None:
        await environment.start()
        try:
            source = environment._resolve_path("/workspace/export")
            for name, data in payload.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            before = set(environment._root.rglob("*"))
            await environment.download_dir_filtered(
                source_dir="/workspace/export",
                target_dir=target,
                include=["no-match/*.yaml"] if no_match else ["*.txt"],
                exclude=["nested/*"],
                protect=[] if no_match else ["nested/keep.txt", "protected.log"],
            )
            expected = (
                {}
                if no_match
                else {
                    name: payload[name]
                    for name in (
                        "keep.txt",
                        "nested/keep.txt",
                        "protected.log",
                        "odd [x] ' café.txt",
                        "--literal.txt",
                        "line\nbreak.txt",
                    )
                }
            )
            assert _contained_transfer_files(target) == {**expected, "retained.txt": b"existing unrelated output\n"}
            assert _contained_transfer_files(source) == payload
            assert set(environment._root.rglob("*")) == before
        finally:
            await environment.stop(delete=False)

    asyncio.run(exercise())


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS bsdtar generates AppleDouble metadata entries")
@pytest.mark.parametrize("method", ["download_dir_with_exclusions", "download_dir_filtered"])
def test_contained_download_keeps_authored_dot_underscore_files_without_generated_metadata(
    local_output_environment: SkillEvaluatorLocalEnvironment, tmp_path: Path, method: str
) -> None:
    environment = local_output_environment
    payload = {"keep.txt": b"selected contents\n", "._authored.txt": b"user-authored contents\n"}
    target = tmp_path / "downloaded"
    target.mkdir()
    (target / "retained.txt").write_bytes(b"existing unrelated output\n")

    async def exercise() -> None:
        await environment.start()
        try:
            source = environment._resolve_path("/workspace/export")
            source.mkdir()
            for name, data in payload.items():
                (source / name).write_bytes(data)
            # Directory metadata creates a synthetic ._. member; selected-file
            # metadata creates ._keep.txt even though no such source file exists.
            for path in (source, source / "keep.txt"):
                subprocess.run(
                    ["/usr/bin/xattr", "-w", "com.skillevaluator.transfer-test", "metadata", str(path)],
                    check=True,
                    capture_output=True,
                )
            before = set(environment._root.rglob("*"))
            options = {"exclude": []} if method == "download_dir_with_exclusions" else {"include": ["*.txt"]}

            await getattr(environment, method)(source_dir="/workspace/export", target_dir=target, **options)

            assert _contained_transfer_files(target) == {**payload, "retained.txt": b"existing unrelated output\n"}
            assert _contained_transfer_files(source) == payload
            assert set(environment._root.rglob("*")) == before
        finally:
            await environment.stop(delete=False)

    asyncio.run(exercise())


@pytest.mark.skipif(os.name == "nt", reason="real local subprocess backend requires POSIX")
@pytest.mark.parametrize("method", ["download_dir_with_exclusions", "download_dir_filtered"])
@pytest.mark.parametrize("spelling", ["absolute", "mapped-escape", "linked-source"])
def test_contained_transfer_refuses_outside_source_before_destination_or_staging_mutation(
    local_output_environment: SkillEvaluatorLocalEnvironment,
    tmp_path: Path,
    method: str,
    spelling: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = local_output_environment
    outside = tmp_path / "outside-source"
    outside.mkdir()
    (outside / "private.txt").write_bytes(b"outside sentinel\n")
    target = tmp_path / "destination"
    target.mkdir()
    (target / "retained.txt").write_bytes(b"destination sentinel\n")

    async def exercise() -> None:
        await environment.start()
        try:
            source = str(outside)
            if spelling == "mapped-escape":
                source = "/workspace/../../../outside-source"
            elif spelling == "linked-source":
                environment._resolve_path("/workspace/linked").symlink_to(outside, target_is_directory=True)
                source = "/workspace/linked"
            before = set(environment._root.rglob("*"))
            commands: list[str] = []
            original_exec = environment.exec

            async def observe_exec(command: str, **kwargs):
                commands.append(command)
                return await original_exec(command, **kwargs)

            monkeypatch.setattr(environment, "exec", observe_exec)
            options = {"exclude": []} if method == "download_dir_with_exclusions" else {"include": ["*.txt"]}
            with pytest.raises(ValueError, match="outside"):
                await getattr(environment, method)(source_dir=source, target_dir=target, **options)
            assert _contained_transfer_files(target) == {"retained.txt": b"destination sentinel\n"}
            assert commands == [], "invalid sources must fail before executing find/tar or cleanup"
            assert _contained_transfer_files(outside) == {"private.txt": b"outside sentinel\n"}
            assert set(environment._root.rglob("*")) == before
        finally:
            await environment.stop(delete=False)

    asyncio.run(exercise())


async def _download_directory(
    environment: SkillEvaluatorLocalEnvironment,
    transfer: str,
    source: str,
    target: Path,
) -> None:
    if transfer == "download":
        await environment.download_dir(source, target)
    elif transfer == "excluded":
        await environment.download_dir_with_exclusions(source_dir=source, target_dir=target, exclude=["*.log"])
    else:
        await environment.download_dir_filtered(source_dir=source, target_dir=target, include=["*.txt"])


@pytest.mark.skipif(os.name == "nt", reason="real local subprocess backend requires POSIX")
@pytest.mark.parametrize("transfer", ["download", "excluded", "filtered"])
@pytest.mark.parametrize("spelling", ["workspace", "logs", "home", "relative"])
def test_every_directory_download_refuses_dotdot_paths_that_leave_the_run_directory(
    local_output_environment: SkillEvaluatorLocalEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transfer: str,
    spelling: str,
) -> None:
    """download_dir and both tar/find downloads refuse the same sources, whichever root the path starts from."""
    environment = local_output_environment
    outside = tmp_path.resolve() / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"outside sentinel\n")
    prefix, host_base = {
        "workspace": ("/workspace", environment._workspace),
        "logs": ("/logs", environment._logs_dir),
        "home": ("~", environment._home),
        "relative": ("", environment._workspace),
    }[spelling]
    relative = os.path.relpath(outside, host_base)
    assert relative.startswith("..")
    source = f"{prefix}/{relative}" if prefix else relative
    target = tmp_path / "destination"
    target.mkdir()
    (target / "retained.txt").write_bytes(b"destination sentinel\n")

    async def exercise() -> None:
        await environment.start()
        try:
            before = set(environment._root.rglob("*"))
            commands: list[str] = []
            original_exec = environment.exec

            async def observe_exec(command: str, **kwargs):
                commands.append(command)
                return await original_exec(command, **kwargs)

            monkeypatch.setattr(environment, "exec", observe_exec)
            with pytest.raises(ValueError, match="outside"):
                await _download_directory(environment, transfer, source, target)
            assert commands == [], "a source outside the run directory must be refused before find or tar"
            assert _contained_transfer_files(target) == {"retained.txt": b"destination sentinel\n"}
            assert _contained_transfer_files(outside) == {"secret.txt": b"outside sentinel\n"}
            assert set(environment._root.rglob("*")) == before
        finally:
            await environment.stop(delete=False)

    asyncio.run(exercise())


@pytest.mark.skipif(os.name == "nt", reason="real local subprocess backend requires POSIX")
@pytest.mark.parametrize("transfer", ["download", "excluded", "filtered"])
def test_every_directory_download_accepts_dotdot_paths_that_stay_inside_the_run_directory(
    local_output_environment: SkillEvaluatorLocalEnvironment, tmp_path: Path, transfer: str
) -> None:
    """A ".." that resolves inside the run directory is fine, and the filters still apply."""
    environment = local_output_environment
    target = tmp_path / "destination"

    async def exercise() -> None:
        await environment.start()
        try:
            kept = environment._resolve_path("/workspace/kept")
            (kept / "nested").mkdir(parents=True)
            (kept / "answer.txt").write_bytes(b"inside\n")
            (kept / "notes.md").write_bytes(b"notes\n")
            (kept / "run.log").write_bytes(b"noise\n")
            (kept / "nested" / "summary.txt").write_bytes(b"nested\n")
            environment._resolve_path("/workspace/skills").mkdir(exist_ok=True)
            await _download_directory(environment, transfer, "/workspace/skills/../kept", target)
        finally:
            await environment.stop(delete=False)

    asyncio.run(exercise())

    expected = {
        "download": {
            "answer.txt": b"inside\n",
            "notes.md": b"notes\n",
            "run.log": b"noise\n",
            "nested/summary.txt": b"nested\n",
        },
        "excluded": {"answer.txt": b"inside\n", "notes.md": b"notes\n", "nested/summary.txt": b"nested\n"},
        "filtered": {"answer.txt": b"inside\n", "nested/summary.txt": b"nested\n"},
    }[transfer]
    assert _contained_transfer_files(target) == expected


@pytest.mark.skipif(os.name == "nt", reason="real local subprocess backend requires POSIX")
@pytest.mark.parametrize("method", ["download_dir_with_exclusions", "download_dir_filtered"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_contained_transfer_linked_destination_failure_cleans_staging_and_preserves_output(
    local_output_environment: SkillEvaluatorLocalEnvironment, tmp_path: Path, method: str, kind: str
) -> None:
    environment = local_output_environment
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside sentinel\n")
    target = tmp_path / "destination"
    target.mkdir()
    (target / "retained.txt").write_bytes(b"destination sentinel\n")
    if kind == "symlink":
        (target / "keep.txt").symlink_to(outside)
    else:
        os.link(outside, target / "keep.txt")

    async def exercise() -> None:
        await environment.start()
        try:
            source = environment._resolve_path("/workspace/export")
            source.mkdir()
            (source / "keep.txt").write_bytes(b"new source contents\n")
            before = set(environment._root.rglob("*"))

            def destination_identity():
                return {
                    path.name: (
                        metadata.st_mode,
                        metadata.st_ino,
                        metadata.st_dev,
                        metadata.st_nlink,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                    )
                    for path in target.iterdir()
                    for metadata in [path.lstat()]
                }

            before_destination = destination_identity()
            options = {"exclude": []} if method == "download_dir_with_exclusions" else {"include": ["*.txt"]}
            with pytest.raises(ValueError):
                await getattr(environment, method)(source_dir="/workspace/export", target_dir=target, **options)
            assert outside.read_bytes() == b"outside sentinel\n"
            assert (target / "retained.txt").read_bytes() == b"destination sentinel\n"
            assert (target / "keep.txt").read_bytes() == b"outside sentinel\n"
            assert destination_identity() == before_destination
            assert _contained_transfer_files(source) == {"keep.txt": b"new source contents\n"}
            assert set(environment._root.rglob("*")) == before
        finally:
            await environment.stop(delete=False)

    asyncio.run(exercise())


@pytest.mark.skipif(os.name == "nt", reason="real local subprocess backend requires POSIX")
def test_contained_filtered_transfer_propagates_directory_traversal_failure(
    local_output_environment: SkillEvaluatorLocalEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = local_output_environment
    target = tmp_path / "destination"
    target.mkdir()
    (target / "retained.txt").write_bytes(b"destination sentinel\n")

    async def exercise() -> None:
        await environment.start()
        try:
            source = environment._resolve_path("/workspace/export")
            source.mkdir()
            (source / "private").mkdir()
            (source / "private" / "keep.txt").write_bytes(b"source sentinel\n")
            before = set(environment._root.rglob("*"))
            original_walk = os.walk

            def failed_walk(top, *args, onerror=None, **kwargs):
                if Path(top) == source:
                    if onerror is not None:
                        onerror(PermissionError("cannot scan private child directory"))
                    return
                yield from original_walk(top, *args, onerror=onerror, **kwargs)

            monkeypatch.setattr(os, "walk", failed_walk)
            with pytest.raises(PermissionError, match="cannot scan"):
                await environment.download_dir_filtered(source_dir="/workspace/export", target_dir=target)
            assert _contained_transfer_files(target) == {"retained.txt": b"destination sentinel\n"}
            assert (source / "private" / "keep.txt").read_bytes() == b"source sentinel\n"
            assert set(environment._root.rglob("*")) == before
        finally:
            await environment.stop(delete=False)

    asyncio.run(exercise())
