# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor subprocess completion is not sufficient proof of successful trials."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import collector, runner


def test_published_execution_errors_are_redacted_bounded_and_counted() -> None:
    github_token = "ghp_" + ("A" * 36)
    errors = [f"launch failed with {github_token} " + ("x" * 65_536)]
    errors.extend(f"distinct launch error {index}" for index in range(300))

    published, total = runner._published_execution_errors(errors)

    assert total == 301
    assert len(published) == runner.PUBLISHED_EXECUTION_ERRORS_MAX
    assert all(len(error) <= runner.PUBLISHED_EXECUTION_ERROR_MAX_CHARS for error in published)
    assert github_token not in json.dumps(published)
    assert "<redacted>" in published[0]


def test_published_execution_errors_bound_serialized_multibyte_sample() -> None:
    github_token = "ghp_" + ("A" * 36)
    errors = [
        f"launch {index}: {github_token}\x1b[2J" + ("😀" * 2_048)
        for index in range(runner.PUBLISHED_EXECUTION_ERRORS_MAX)
    ]

    published, total = runner._published_execution_errors(errors)
    serialized = json.dumps(published, indent=2).encode("utf-8")

    assert total == runner.PUBLISHED_EXECUTION_ERRORS_MAX
    assert len(serialized) <= 64 * 1024
    assert len(published) < total
    assert github_token.encode() not in serialized
    assert b"\\u001b" not in serialized
    assert b"[2J" in serialized


def test_published_execution_errors_count_distinct_details_beyond_truncation() -> None:
    shared_prefix = "x" * (runner.PUBLISHED_EXECUTION_ERROR_MAX_CHARS * 2)

    published, total = runner._published_execution_errors([f"{shared_prefix}-first", f"{shared_prefix}-second"])

    assert total == 2
    assert len(published) == 1


@pytest.mark.parametrize("return_code", [0, 7])
def test_bounded_harbor_process_preserves_exit_and_combined_diagnostic_tail(
    return_code: int,
) -> None:
    script = (
        "import os; "
        "os.write(1, b'old-prefix-' + b'x' * 256); "
        "os.write(2, b'|useful-stderr-tail|'); "
        f"raise SystemExit({return_code})"
    )

    result = runner._run_bounded_harbor_process(
        [sys.executable, "-c", script],
        env=dict(os.environ),
        stdin_text=None,
        timeout_seconds=5,
        max_output_bytes=4096,
        diagnostic_tail_chars=64,
        secret_values=set(),
    )

    assert result.returncode == return_code
    assert result.output_exceeded is False
    assert len(result.output_tail) <= 64
    assert result.output_tail.endswith("|useful-stderr-tail|")


def test_bounded_harbor_process_redacts_before_retaining_diagnostic_tail() -> None:
    secret = "SYNTHETIC_SECRET_ABCDEF"
    script = f"import os; os.write(2, {('X' + secret + '|' + secret + '!' * 8).encode()!r})"

    result = runner._run_bounded_harbor_process(
        [sys.executable, "-c", script],
        env=dict(os.environ),
        stdin_text=None,
        timeout_seconds=5,
        max_output_bytes=4096,
        diagnostic_tail_chars=43,
        secret_values={secret},
    )

    assert result.returncode == 0
    assert result.output_exceeded is False
    assert secret not in result.output_tail
    assert all(secret[index:] not in result.output_tail for index in range(len(secret) - 3))
    assert result.output_tail.endswith("!" * 8)


def test_bounded_harbor_process_drops_partial_secret_at_output_limit() -> None:
    secret = "SYNTHETIC_SECRET_ABCDEF"
    prefix = "ordinary-prefix|"
    accepted_secret_prefix = secret[:-3]
    script = f"import os,time; os.write(2, {(prefix + secret + '|overflow').encode()!r}); time.sleep(30)"

    result = runner._run_bounded_harbor_process(
        [sys.executable, "-c", script],
        env=dict(os.environ),
        stdin_text=None,
        timeout_seconds=5,
        max_output_bytes=len((prefix + accepted_secret_prefix).encode()),
        diagnostic_tail_chars=128,
        secret_values={secret},
    )

    assert result.output_exceeded is True
    assert accepted_secret_prefix not in result.output_tail
    assert all(secret[index:-3] not in result.output_tail for index in range(len(secret) - 6))


def test_bounded_harbor_process_timeout_includes_blocked_stdin_delivery() -> None:
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="timed out"):
        runner._run_bounded_harbor_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=dict(os.environ),
            stdin_text="x" * (1024 * 1024),
            timeout_seconds=0.1,
            max_output_bytes=4096,
            diagnostic_tail_chars=128,
            secret_values=set(),
        )

    assert time.monotonic() - started < 3


def test_bounded_harbor_process_contains_tree_after_stdin_delivery_error() -> None:
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="stdin delivery failed"):
        runner._run_bounded_harbor_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=dict(os.environ),
            stdin_text="\udcff",
            timeout_seconds=5,
            max_output_bytes=4096,
            diagnostic_tail_chars=128,
            secret_values=set(),
        )

    assert time.monotonic() - started < 3


def test_bounded_harbor_process_timeout_preserves_redacted_diagnostic_tail() -> None:
    secret = "synthetic-timeout-secret"
    script = f"import os,time;os.write(2,{f'useful timeout diagnostic {secret}'.encode()!r});time.sleep(30)"

    with pytest.raises(RuntimeError, match="timed out") as raised:
        runner._run_bounded_harbor_process(
            [sys.executable, "-c", script],
            env=dict(os.environ),
            stdin_text=None,
            timeout_seconds=0.1,
            max_output_bytes=4096,
            diagnostic_tail_chars=128,
            secret_values={secret},
        )

    detail = str(raised.value)
    assert "useful timeout diagnostic" in detail
    assert secret not in detail
    assert "redacted" in detail.lower()


def test_bounded_harbor_process_redacts_secret_created_by_timeout_message() -> None:
    secret = "Harbor run timed out after 0.1 seconds"

    with pytest.raises(runner._HarborRunTimeoutError) as raised:
        runner._run_bounded_harbor_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=dict(os.environ),
            stdin_text=None,
            timeout_seconds=0.1,
            max_output_bytes=4096,
            diagnostic_tail_chars=128,
            secret_values={secret},
        )

    detail = str(raised.value)
    assert secret not in detail
    assert "redacted" in detail.lower()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX cleanup for the red-state fallback")
@pytest.mark.parametrize("failed_start", [1, 2])
def test_bounded_harbor_process_owns_cleanup_during_thread_start(
    monkeypatch: pytest.MonkeyPatch,
    failed_start: int,
) -> None:
    real_popen = runner.subprocess.Popen
    processes: list[object] = []
    started_threads: list[threading.Thread] = []
    start_count = 0

    def tracked_popen(*args: object, **kwargs: object) -> object:
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    real_thread_start = threading.Thread.start

    def failing_thread_start(thread: threading.Thread) -> None:
        nonlocal start_count
        start_count += 1
        if start_count == failed_start:
            raise RuntimeError("synthetic thread exhaustion")
        real_thread_start(thread)
        started_threads.append(thread)

    monkeypatch.setattr(runner.subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(threading.Thread, "start", failing_thread_start)

    try:
        with pytest.raises(RuntimeError, match="synthetic thread exhaustion"):
            runner._run_bounded_harbor_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                env=dict(os.environ),
                stdin_text="payload",
                timeout_seconds=5,
                max_output_bytes=4096,
                diagnostic_tail_chars=128,
                secret_values=set(),
            )

        assert len(processes) == 1
        assert processes[0].poll() is not None  # type: ignore[attr-defined]
        assert all(not thread.is_alive() for thread in started_threads)
    finally:
        for process in processes:
            if process.poll() is None:  # type: ignore[attr-defined]
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)  # type: ignore[attr-defined]
            with contextlib.suppress(Exception):
                process.wait(timeout=5)  # type: ignore[attr-defined]
            for stream_name in ("stdin", "stdout"):
                stream = getattr(process, stream_name, None)
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process-group verification")
@pytest.mark.parametrize("failure_mode", ["output", "timeout"])
def test_bounded_harbor_process_failure_reaps_descendants(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    child_pid_path = tmp_path / f"{failure_mode}-child-pid"
    child_marker = tmp_path / f"{failure_mode}-child-survived"
    child_script = (
        "import pathlib,signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(1);"
        f"pathlib.Path({str(child_marker)!r}).write_text('survived')"
    )
    parent_script = (
        "import os,pathlib,subprocess,sys,time;"
        f"child=subprocess.Popen([{sys.executable!r},'-c',{child_script!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid));"
        + ("os.write(1,b'a'*3000);os.write(2,b'b'*3000);" if failure_mode == "output" else "")
        + "time.sleep(30)"
    )

    started = time.monotonic()
    if failure_mode == "timeout":
        with pytest.raises(RuntimeError, match="timed out"):
            runner._run_bounded_harbor_process(
                [sys.executable, "-c", parent_script],
                env=dict(os.environ),
                stdin_text=None,
                timeout_seconds=1,
                max_output_bytes=4096,
                diagnostic_tail_chars=128,
                secret_values=set(),
            )
    else:
        result = runner._run_bounded_harbor_process(
            [sys.executable, "-c", parent_script],
            env=dict(os.environ),
            stdin_text=None,
            timeout_seconds=5,
            max_output_bytes=4096,
            diagnostic_tail_chars=128,
            secret_values=set(),
        )
        assert result.output_exceeded is True
    assert time.monotonic() - started < 3

    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
        pytest.fail("Harbor descendant survived process-tree containment")
    time.sleep(1.05)
    assert not child_marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows taskkill tree verification")
@pytest.mark.parametrize("failure_mode", ["output", "timeout"])
def test_bounded_harbor_process_failure_reaps_windows_descendants(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    child_pid_path = tmp_path / f"{failure_mode}-windows-child-pid"
    child_marker = tmp_path / f"{failure_mode}-windows-child-survived"
    child_script = f"import pathlib,time;time.sleep(1);pathlib.Path({str(child_marker)!r}).write_text('survived')"
    parent_script = (
        "import os,pathlib,subprocess,sys,time;"
        f"child=subprocess.Popen([{sys.executable!r},'-c',{child_script!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid));"
        + ("os.write(1,b'a'*3000);os.write(2,b'b'*3000);" if failure_mode == "output" else "")
        + "time.sleep(30)"
    )

    if failure_mode == "timeout":
        with pytest.raises(RuntimeError, match="timed out"):
            runner._run_bounded_harbor_process(
                [sys.executable, "-c", parent_script],
                env=dict(os.environ),
                stdin_text=None,
                timeout_seconds=1,
                max_output_bytes=4096,
                diagnostic_tail_chars=128,
                secret_values=set(),
            )
    else:
        result = runner._run_bounded_harbor_process(
            [sys.executable, "-c", parent_script],
            env=dict(os.environ),
            stdin_text=None,
            timeout_seconds=5,
            max_output_bytes=4096,
            diagnostic_tail_chars=128,
            secret_values=set(),
        )
        assert result.output_exceeded is True

    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    process_query_limited_information = 0x1000
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            child_pid,
        )
        if not handle:
            break
        kernel32.CloseHandle(handle)
        time.sleep(0.01)
    else:
        pytest.fail("Harbor descendant survived Windows process-tree containment")
    time.sleep(1.05)
    assert not child_marker.exists()


def test_windows_tree_cleanup_uses_verified_system32_taskkill_despite_path_and_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    system_directory = tmp_path / "trusted" / "System32"
    system_directory.mkdir(parents=True)
    trusted_taskkill = system_directory / "taskkill.exe"
    trusted_taskkill.write_bytes(b"trusted")
    decoy_directory = tmp_path / "decoy"
    decoy_directory.mkdir()
    decoy_taskkill = decoy_directory / "taskkill.exe"
    decoy_taskkill.write_bytes(b"decoy")
    monkeypatch.chdir(decoy_directory)
    monkeypatch.setenv("PATH", str(decoy_directory))
    monkeypatch.setattr(runner, "_windows_system_directory", lambda: system_directory, raising=False)

    class WindowsOs:
        name = "nt"

    class FakeProcess:
        def __init__(self, *, pid: int, returncode: int | None) -> None:
            self.pid = pid
            self.returncode = returncode
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = -9
            return self.returncode

    launched: list[list[str]] = []
    taskkill_process = FakeProcess(pid=99, returncode=0)

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        launched.append(command)
        return taskkill_process

    root_process = FakeProcess(pid=4242, returncode=None)
    monkeypatch.setattr(runner, "os", WindowsOs())
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    runner._terminate_harbor_process_tree(root_process)  # type: ignore[arg-type]

    assert len(launched) == 1
    selected_taskkill = Path(launched[0][0])
    assert selected_taskkill.is_absolute()
    assert selected_taskkill.resolve() == trusted_taskkill.resolve()
    assert selected_taskkill.resolve() != decoy_taskkill.resolve()


def test_windows_tree_cleanup_fails_closed_when_taskkill_misses_an_exited_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WindowsOs:
        name = "nt"

    class FakeProcess:
        def __init__(self, *, pid: int, returncode: int | None) -> None:
            self.pid = pid
            self.returncode = returncode
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return int(self.returncode or 0)

    taskkill_process = FakeProcess(pid=99, returncode=1)
    root_process = FakeProcess(pid=4242, returncode=0)
    monkeypatch.setattr(runner, "os", WindowsOs())
    monkeypatch.setattr(runner, "_verified_windows_taskkill_path", lambda: Path("/trusted/taskkill.exe"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: taskkill_process)

    with pytest.raises(RuntimeError, match="process-tree cleanup could not be confirmed"):
        runner._terminate_harbor_process_tree(root_process)  # type: ignore[arg-type]

    assert root_process.killed is False


@pytest.mark.parametrize("candidate_kind", ["missing", "directory"])
def test_windows_tree_cleanup_rejects_unverified_system32_taskkill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    candidate_kind: str,
) -> None:
    system_directory = tmp_path / "trusted" / "System32"
    system_directory.mkdir(parents=True)
    if candidate_kind == "directory":
        (system_directory / "taskkill.exe").mkdir()
    monkeypatch.setattr(runner, "_windows_system_directory", lambda: system_directory, raising=False)

    class WindowsOs:
        name = "nt"

    class FakeProcess:
        def __init__(self, *, pid: int, returncode: int | None) -> None:
            self.pid = pid
            self.returncode = returncode
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = -9
            return self.returncode

    launched: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        launched.append(command)
        return FakeProcess(pid=99, returncode=0)

    root_process = FakeProcess(pid=4242, returncode=None)
    monkeypatch.setattr(runner, "os", WindowsOs())
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="process-tree cleanup could not be confirmed"):
        runner._terminate_harbor_process_tree(root_process)  # type: ignore[arg-type]

    assert launched == []
    assert root_process.killed is True


@pytest.mark.parametrize("configured_concurrency", [1, 3, 4])
def test_agent_pair_treats_concurrency_as_a_global_condition_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_concurrency: int,
) -> None:
    lock = threading.Lock()
    active_budget = 0
    maximum_active_budget = 0
    launched: list[tuple[str, int]] = []

    def _run_harbor(**kwargs: object) -> tuple[bool, str]:
        nonlocal active_budget, maximum_active_budget
        budget = int(kwargs["n_concurrent"])
        with lock:
            active_budget += budget
            maximum_active_budget = max(maximum_active_budget, active_budget)
            launched.append((str(kwargs["job_name"]), budget))
        time.sleep(0.05)
        with lock:
            active_budget -= budget
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", _run_harbor)

    errors = runner._run_agent_pair(
        skill_name="demo",
        agent="opencode",
        model="nvidia/openai/gpt-oss-120b",
        env_mode="docker",
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=1,
        n_concurrent=configured_concurrency,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=4,
    )

    assert errors == []
    assert {name for name, _budget in launched} == {"demo-opencode-with", "demo-opencode-without"}
    assert maximum_active_budget <= configured_concurrency


def test_agent_pair_assigns_the_full_concurrency_budget_when_baseline_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    budgets: list[int] = []
    monkeypatch.setattr(
        runner,
        "_run_harbor",
        lambda **kwargs: budgets.append(int(kwargs["n_concurrent"])) or (True, ""),
    )

    errors = runner._run_agent_pair(
        skill_name="demo",
        agent="opencode",
        model="nvidia/openai/gpt-oss-120b",
        env_mode="docker",
        with_skill=tmp_path / "with",
        baseline=None,
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=1,
        n_concurrent=4,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=4,
    )

    assert errors == []
    assert budgets == [4]


@pytest.mark.parametrize(
    ("env_mode", "agent", "import_path"),
    [
        (
            "docker",
            "codex",
            "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex",
        ),
        (
            "docker",
            "claude-code",
            "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode",
        ),
        (
            "local",
            "codex",
            "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildCodex",
        ),
        (
            "local",
            "claude-code",
            "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalNvidiaBuildClaudeCode",
        ),
    ],
)
def test_stop_on_pass_preserves_nvidia_build_agent_import_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_mode: str,
    agent: str,
    import_path: str,
) -> None:
    launches: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "_run_harbor",
        lambda **kwargs: launches.append(kwargs) or (True, ""),
    )
    monkeypatch.setattr(runner, "_job_passed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", lambda *_args, **_kwargs: None)

    errors = runner._run_agent_pair(
        skill_name="demo",
        agent=agent,
        model="nvidia/nemotron-3-super-120b-a12b",
        env_mode=env_mode,
        with_skill=tmp_path / "with",
        baseline=None,
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=2,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=2,
        agent_import_path=import_path,
        stop_on_pass=True,
        task_names=["case-001"],
        verifier_env={"SKILL_EVAL_JUDGE_MODEL": "${SKILL_EVAL_JUDGE_MODEL}"},
    )

    assert errors == []
    assert len(launches) == 1
    assert launches[0]["agent_import_path"] == import_path
    assert launches[0]["include_task_names"] == ["case-001"]
    assert launches[0]["verifier_env"] == {
        "SKILL_EVAL_JUDGE_MODEL": "${SKILL_EVAL_JUDGE_MODEL}",
    }


_UNSAFE_LINK = r"symlink|reparse"


@pytest.mark.parametrize("link_kind", ["directory", "dangling"])
def test_merge_attempt_jobs_rejects_linked_whole_job_root(tmp_path: Path, link_kind: str) -> None:
    target = tmp_path / "real-job"
    if link_kind == "directory":
        target.mkdir()
    job_link = tmp_path / "attempt-001"
    job_link.symlink_to(target, target_is_directory=True)
    aggregate_dir = tmp_path / "aggregate"

    with pytest.raises(ValueError, match=r"non-linked|symlink|reparse"):
        runner._merge_attempt_jobs([job_link], aggregate_dir)

    assert not aggregate_dir.exists()


def test_merge_attempt_jobs_rejects_mocked_reparse_whole_job_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    detect_link = runner._job_path_is_link_or_reparse

    def mocked_reparse(path: Path, metadata: object) -> bool:
        return path == job_dir or detect_link(path, metadata)

    monkeypatch.setattr(runner, "_job_path_is_link_or_reparse", mocked_reparse)

    with pytest.raises(ValueError, match="non-linked"):
        runner._merge_attempt_jobs([job_dir], tmp_path / "aggregate")


def test_merge_attempt_jobs_rejects_non_directory_whole_job_root(tmp_path: Path) -> None:
    job_file = tmp_path / "attempt-001"
    job_file.write_text("not a job", encoding="utf-8")

    with pytest.raises(ValueError, match="non-linked directory"):
        runner._merge_attempt_jobs([job_file], tmp_path / "aggregate")


@pytest.mark.parametrize("artifact_kind", ["symlink", "hardlink"])
def test_merge_attempt_jobs_rejects_forged_root_result(tmp_path: Path, artifact_kind: str) -> None:
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    forged = tmp_path / "forged-result.json"
    forged.write_text('{"n_total_trials": 999, "stats": {}}', encoding="utf-8")
    result = job_dir / "result.json"
    try:
        if artifact_kind == "symlink":
            result.symlink_to(forged)
        else:
            result.hardlink_to(forged)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"{artifact_kind} creation unavailable: {exc}")

    with pytest.raises(ValueError, match=r"symlink|reparse|hard.?link|multiple links"):
        runner._merge_attempt_jobs([job_dir], tmp_path / "aggregate")


def test_merge_attempt_jobs_rejects_root_result_source_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secure_copy = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    result = job_dir / "result.json"
    result.write_text('{"n_total_trials": 1, "stats": {}}', encoding="utf-8")
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    marker = aggregate_dir / "keep.txt"
    marker.write_text("old aggregate", encoding="utf-8")
    original = secure_copy._build_tree_manifest

    def validate_then_replace(source: Path, *args: object, **kwargs: object):
        manifest = original(source, *args, **kwargs)
        if Path(source).resolve() == job_dir.resolve():
            result.unlink()
            result.write_text('{"n_total_trials": 999, "stats": {}}', encoding="utf-8")
        return manifest

    monkeypatch.setattr(secure_copy, "_build_tree_manifest", validate_then_replace)

    with pytest.raises(ValueError, match="source changed after validation"):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "old aggregate"


def test_merge_attempt_jobs_rejects_regular_root_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    (job_dir / "result.json").write_text('{"n_total_trials": 1, "stats": {}}', encoding="utf-8")
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    marker = aggregate_dir / "keep.txt"
    marker.write_text("old aggregate", encoding="utf-8")
    original_copy = runner.copytree_secure
    replaced = False

    def replace_root_then_copy(source: Path, destination: Path, **kwargs: object) -> None:
        nonlocal replaced
        if not replaced and Path(source) == job_dir:
            replaced = True
            job_dir.rename(tmp_path / "original-job")
            job_dir.mkdir()
            (job_dir / "result.json").write_text('{"n_total_trials": 999, "stats": {}}', encoding="utf-8")
        original_copy(source, destination, **kwargs)

    monkeypatch.setattr(runner, "copytree_secure", replace_root_then_copy)

    with pytest.raises(ValueError, match="root changed during snapshot"):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "old aggregate"


def _write_multistep_attempt_job(
    job_dir: Path,
    *,
    root_score: float,
    step_scores: tuple[float, ...],
    failed: bool = False,
) -> Path:
    trial = job_dir / "case-001_attempt001"
    for index, score in enumerate(step_scores, start=1):
        verifier = trial / "steps" / f"step-{index}" / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.json").write_text(json.dumps({"overall": score}), encoding="utf-8")
    result: dict[str, object] = {
        "trial_name": trial.name,
        "task_name": "case-001",
        "verifier_result": {"rewards": {"overall": root_score}},
        "step_results": [
            {"step_name": f"step-{index}", "verifier_result": {"rewards": {"overall": score}}}
            for index, score in enumerate(step_scores, start=1)
        ],
    }
    if failed:
        result["exception_info"] = {
            "exception_type": "TaskFailure",
            "exception_message": "attempt crashed",
        }
    (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
    _write_successful_job_result(job_dir, trial.name)
    return trial


def _write_successful_job_result(job_dir: Path, trial_name: str) -> None:
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_trials": 1,
                    "n_errors": 0,
                    "evals": {
                        "demo": {
                            "n_trials": 1,
                            "n_errors": 0,
                            "reward_stats": {"overall": {"1.0": [trial_name]}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_job_passed_uses_authoritative_multistep_root_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_multistep_attempt_job(job_dir, root_score=0.5, step_scores=(1.0, 0.0))

    assert runner._job_passed(job_dir, 0.75) is False
    rewards = collector._extract_rewards(job_dir)
    assert len(rewards) == 1
    assert collector._average_overall(rewards) == 0.5


def test_job_passed_accepts_passing_authoritative_multistep_root_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_multistep_attempt_job(job_dir, root_score=0.8, step_scores=(0.0, 0.0))

    assert runner._job_passed(job_dir, 0.75) is True


def test_job_passed_rejects_failed_trial_even_with_passing_rewards(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_multistep_attempt_job(job_dir, root_score=1.0, step_scores=(1.0,), failed=True)

    assert runner._job_passed(job_dir, 0.75) is False


def test_job_passed_rejects_failed_job_result_even_with_passing_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_multistep_attempt_job(job_dir, root_score=1.0, step_scores=(1.0,))
    job_result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    job_result["stats"]["n_errors"] = 1
    (job_dir / "result.json").write_text(json.dumps(job_result), encoding="utf-8")

    assert runner._job_passed(job_dir, 0.75) is False


def test_job_passed_preserves_legacy_single_step_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    trial_name = "case-001_attempt001"
    verifier = job_dir / trial_name / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(json.dumps({"overall": 0.9}), encoding="utf-8")
    _write_successful_job_result(job_dir, trial_name)

    assert runner._job_passed(job_dir, 0.75) is True


def test_merge_attempt_jobs_rejects_symlinked_trial_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside-trial"
    outside.mkdir()
    (outside / "host-secret.txt").write_text("secret", encoding="utf-8")
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    trial_link = job_dir / "case-001__trial"
    trial_link.symlink_to(outside, target_is_directory=True)
    aggregate_dir = tmp_path / "aggregate"

    with pytest.raises(ValueError, match=_UNSAFE_LINK):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert not (aggregate_dir / f"{job_dir.name}__{trial_link.name}" / "host-secret.txt").exists()
    assert not (aggregate_dir / "result.json").exists()


def test_merge_attempt_jobs_rejects_symlinked_trial_file(tmp_path: Path) -> None:
    outside = tmp_path / "host-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    job_dir = tmp_path / "attempt-001"
    trial_dir = job_dir / "case-001__trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "artifact.txt").symlink_to(outside)
    aggregate_dir = tmp_path / "aggregate"

    with pytest.raises(ValueError, match=_UNSAFE_LINK):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert not (aggregate_dir / f"{job_dir.name}__{trial_dir.name}" / "artifact.txt").exists()
    assert not (aggregate_dir / "result.json").exists()


def test_merge_attempt_jobs_rejects_nested_directory_link_like_reparse_point(tmp_path: Path) -> None:
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    (outside / "host-secret.txt").write_text("secret", encoding="utf-8")
    job_dir = tmp_path / "attempt-001"
    nested = job_dir / "case-001__trial" / "artifacts"
    nested.mkdir(parents=True)
    linked_dir = nested / "external"
    linked_dir.symlink_to(outside, target_is_directory=True)
    aggregate_dir = tmp_path / "aggregate"

    with pytest.raises(ValueError, match=_UNSAFE_LINK):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    copied_secret = aggregate_dir / f"{job_dir.name}__case-001__trial" / "artifacts" / "external" / "host-secret.txt"
    assert not copied_secret.exists()
    assert not (aggregate_dir / "result.json").exists()


def test_merge_attempt_jobs_preserves_regular_trial_artifacts(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from uuid import UUID

    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.result import TrialResult
    from harbor.viewer.scanner import JobScanner

    job_dir = tmp_path / "attempt-001"
    trial_dir = job_dir / "case-001__trial"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "output.txt").write_text("expected", encoding="utf-8")
    now = datetime(2026, 8, 25, tzinfo=UTC)
    trial_result = TrialResult.model_validate(
        {
            "id": UUID(int=2),
            "task_name": "nvidia/skillevaluator-case-001",
            "trial_name": trial_dir.name,
            "trial_uri": trial_dir.as_uri(),
            "task_id": {"path": str(job_dir / "task" / "case-001")},
            "source": "with",
            "task_checksum": "harbor-0.22-aggregate-fixture",
            "config": {
                "task": {"path": str(job_dir / "task" / "case-001"), "source": "with"},
                "trial_name": trial_dir.name,
                "trials_dir": str(job_dir),
            },
            "agent_info": {
                "name": "opencode",
                "version": "test",
                "model_info": {"name": "test-model"},
            },
            "agent_result": {"n_input_tokens": 7, "n_cache_tokens": 2, "n_output_tokens": 3},
            "verifier_result": {"rewards": {"reward": 1.0}},
            "started_at": now,
            "finished_at": now,
        }
    )
    source_job_result = JobResult(
        id=UUID(int=1),
        started_at=now,
        updated_at=now,
        finished_at=now,
        n_total_trials=1,
        stats=JobStats.from_trial_results([trial_result], n_total_trials=1),
        # Harbor 0.22 currently persists completed TrialResults in each trial
        # directory while this root list can remain empty.
        trial_results=[],
    )
    (job_dir / "result.json").write_text(source_job_result.model_dump_json(indent=2), encoding="utf-8")
    (trial_dir / "result.json").write_text(trial_result.model_dump_json(indent=2), encoding="utf-8")
    (trial_dir / "config.json").write_text(trial_result.config.model_dump_json(indent=2), encoding="utf-8")
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([job_dir], aggregate_dir)

    merged_trial = aggregate_dir / f"{job_dir.name}__{trial_dir.name}"
    assert (merged_trial / "artifacts" / "output.txt").read_text(encoding="utf-8") == "expected"
    merged_result = json.loads((aggregate_dir / "result.json").read_text(encoding="utf-8"))
    reward_names = [
        name
        for eval_stats in merged_result["stats"]["evals"].values()
        for name in eval_stats["reward_stats"]["reward"]["1.0"]
    ]
    assert reward_names == [merged_trial.name]
    scanner = JobScanner(tmp_path)
    parsed_result = scanner.get_job_result(aggregate_dir.name)
    parsed_config = scanner.get_job_config(aggregate_dir.name)
    assert parsed_result is not None
    assert parsed_result.n_total_trials == 1
    assert parsed_result.stats.n_completed_trials == 1
    assert len(parsed_result.trial_results) == parsed_result.stats.n_completed_trials
    assert scanner.list_trials(aggregate_dir.name) == [merged_trial.name]
    parsed_trial = scanner.get_trial_result(aggregate_dir.name, merged_trial.name)
    assert parsed_trial is not None
    assert parsed_trial.trial_name == merged_trial.name
    assert parsed_trial.trial_uri == merged_trial.as_uri()
    assert parsed_trial.config.trial_name == merged_trial.name
    assert parsed_trial.config.trials_dir == aggregate_dir
    assert parsed_result.trial_results[0] == parsed_trial
    assert parsed_config is not None
    assert parsed_config.job_name == aggregate_dir.name


def test_merge_attempt_jobs_normalizes_mixed_naive_and_aware_timestamps(tmp_path: Path) -> None:
    from harbor.viewer.scanner import JobScanner

    job_dir = tmp_path / "attempt-001"
    _write_current_harbor_attempt(
        job_dir,
        trial_name="case-001__trial",
        root_completed=1,
    )
    root_result_path = job_dir / "result.json"
    root_result = json.loads(root_result_path.read_text(encoding="utf-8"))
    root_result.update(
        {
            "started_at": "2026-08-25T19:00:00",
            "updated_at": "2026-08-25T19:01:00",
            "finished_at": "2026-08-25T19:01:00",
        }
    )
    root_result_path.write_text(json.dumps(root_result, indent=2), encoding="utf-8")
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([job_dir], aggregate_dir)

    merged = JobScanner(tmp_path).get_job_result(aggregate_dir.name)
    assert merged is not None
    assert merged.started_at.tzinfo is not None
    assert merged.updated_at.tzinfo is not None
    assert merged.finished_at is not None
    assert merged.finished_at.tzinfo is not None
    assert merged.started_at <= merged.updated_at


def _write_current_harbor_attempt(
    job_dir: Path,
    *,
    trial_name: str,
    root_completed: int,
    root_total: int = 1,
    result_id: int = 20,
) -> None:
    from datetime import UTC, datetime
    from uuid import UUID

    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.result import TrialResult

    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True)
    now = datetime(2026, 8, 25, tzinfo=UTC)
    trial_result = TrialResult.model_validate(
        {
            "id": UUID(int=result_id),
            "task_name": f"nvidia/{trial_name}",
            "trial_name": trial_name,
            "trial_uri": trial_dir.as_uri(),
            "task_id": {"path": str(job_dir / "task" / "case")},
            "source": "with",
            "task_checksum": "harbor-0.22-aggregate-fixture",
            "config": {
                "task": {"path": str(job_dir / "task" / "case"), "source": "with"},
                "trial_name": trial_name,
                "trials_dir": str(job_dir),
            },
            "agent_info": {
                "name": "opencode",
                "version": "test",
                "model_info": {"name": "test-model"},
            },
            "agent_result": {},
            "verifier_result": {"rewards": {"reward": 1.0}},
            "started_at": now,
            "finished_at": now,
        }
    )
    root_results = [trial_result] if root_completed else []
    root_stats = (
        JobStats.from_trial_results(root_results, n_total_trials=root_total)
        if root_completed
        else JobStats(n_pending_trials=root_total)
    )
    source_job_result = JobResult(
        id=UUID(int=result_id + 1),
        started_at=now,
        updated_at=now,
        finished_at=now if root_completed == root_total else None,
        n_total_trials=root_total,
        stats=root_stats,
        trial_results=[],
    )
    (job_dir / "result.json").write_text(source_job_result.model_dump_json(indent=2), encoding="utf-8")
    (trial_dir / "result.json").write_text(trial_result.model_dump_json(indent=2), encoding="utf-8")
    (trial_dir / "config.json").write_text(trial_result.config.model_dump_json(indent=2), encoding="utf-8")


def test_merge_attempt_jobs_accepts_completed_child_ahead_of_stale_root_stats(tmp_path: Path) -> None:
    from harbor.viewer.scanner import JobScanner

    job_dir = tmp_path / "demo-with-case-attempt001"
    _write_current_harbor_attempt(
        job_dir,
        trial_name="case-001__trial",
        root_completed=0,
    )
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([job_dir], aggregate_dir)

    merged = JobScanner(tmp_path).get_job_result(aggregate_dir.name)
    assert merged is not None
    assert merged.stats.n_completed_trials == 1
    assert merged.stats.n_pending_trials == 0
    assert merged.finished_at is not None


def test_merge_attempt_jobs_rejects_root_completed_count_without_valid_child(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from uuid import UUID

    from harbor.models.job.result import JobResult, JobStats

    job_dir = tmp_path / "demo-with-case-attempt001"
    job_dir.mkdir()
    now = datetime(2026, 8, 25, tzinfo=UTC)
    result = JobResult(
        id=UUID(int=30),
        started_at=now,
        updated_at=now,
        finished_at=now,
        n_total_trials=1,
        stats=JobStats(n_completed_trials=1),
        trial_results=[],
    )
    (job_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="completed 1 trials but retained 0"):
        runner._merge_attempt_jobs([job_dir], tmp_path / "aggregate")


@pytest.mark.parametrize(
    ("job_name", "trial_name"),
    [
        ("s" * 180 + "-attempt001", "case-" + "t" * 120),
        ("技" * 70 + "-attempt001", "例" * 70),
    ],
)
def test_merge_attempt_jobs_bounds_long_aggregate_trial_names_by_utf8_bytes(
    tmp_path: Path,
    job_name: str,
    trial_name: str,
) -> None:
    from harbor.viewer.scanner import JobScanner

    job_dir = tmp_path / job_name
    _write_current_harbor_attempt(job_dir, trial_name=trial_name, root_completed=1)
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([job_dir], aggregate_dir)

    merged_names = JobScanner(tmp_path).list_trials(aggregate_dir.name)
    assert len(merged_names) == 1
    assert len(merged_names[0].encode("utf-8")) <= 224
    assert "attempt001" in merged_names[0]


def test_merge_attempt_jobs_long_name_digest_avoids_cross_source_collisions(tmp_path: Path) -> None:
    job_name = "s" * 180 + "-attempt001"
    trial_name = "case-" + "t" * 120
    first = tmp_path / "first" / job_name
    second = tmp_path / "second" / job_name
    _write_current_harbor_attempt(first, trial_name=trial_name, root_completed=1, result_id=40)
    _write_current_harbor_attempt(second, trial_name=trial_name, root_completed=1, result_id=50)
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([first, second], aggregate_dir)

    merged_names = sorted(path.name for path in aggregate_dir.iterdir() if path.is_dir())
    assert len(merged_names) == 2
    assert len(set(merged_names)) == 2
    assert all(len(name.encode("utf-8")) <= 224 for name in merged_names)


def test_merge_attempt_jobs_rebuilds_cancel_retry_token_and_cost_stats(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from uuid import UUID

    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.result import TrialResult
    from harbor.viewer.scanner import JobScanner

    job_dir = tmp_path / "attempt-001"
    trial_dir = job_dir / "case-001__cancelled"
    trial_dir.mkdir(parents=True)
    now = datetime(2026, 8, 25, tzinfo=UTC)
    trial_result = TrialResult.model_validate(
        {
            "id": UUID(int=4),
            "task_name": "nvidia/skillevaluator-case-001",
            "trial_name": trial_dir.name,
            "trial_uri": trial_dir.as_uri(),
            "task_id": {"path": str(job_dir / "task" / "case-001")},
            "source": "with",
            "task_checksum": "harbor-0.22-cancelled-fixture",
            "config": {
                "task": {"path": str(job_dir / "task" / "case-001"), "source": "with"},
                "trial_name": trial_dir.name,
                "trials_dir": str(job_dir),
            },
            "agent_info": {
                "name": "opencode",
                "version": "test",
                "model_info": {"name": "test-model"},
            },
            "agent_result": {
                "n_input_tokens": 123,
                "n_cache_tokens": 4,
                "n_output_tokens": 5,
                "cost_usd": 0.25,
            },
            "verifier_result": None,
            "exception_info": {
                "exception_type": "CancelledError",
                "exception_message": "cancelled",
                "exception_traceback": "",
                "occurred_at": now,
            },
            "started_at": now,
            "finished_at": now,
        }
    )
    source_stats = JobStats.from_trial_results([trial_result], n_total_trials=1, n_retries=2)
    source_job_result = JobResult(
        id=UUID(int=3),
        started_at=now,
        updated_at=now,
        finished_at=now,
        n_total_trials=1,
        stats=source_stats,
        trial_results=[],
    )
    (job_dir / "result.json").write_text(source_job_result.model_dump_json(indent=2), encoding="utf-8")
    (trial_dir / "result.json").write_text(trial_result.model_dump_json(indent=2), encoding="utf-8")
    (trial_dir / "config.json").write_text(trial_result.config.model_dump_json(indent=2), encoding="utf-8")
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([job_dir], aggregate_dir)

    merged = JobScanner(tmp_path).get_job_result(aggregate_dir.name)
    assert merged is not None
    assert merged.stats.n_completed_trials == 1
    assert merged.stats.n_errored_trials == 1
    assert merged.stats.n_cancelled_trials == 1
    assert merged.stats.n_retries == 2
    assert merged.stats.n_input_tokens == 123
    assert merged.stats.n_cache_tokens == 4
    assert merged.stats.n_output_tokens == 5
    assert merged.stats.cost_usd == 0.25
    assert len(merged.trial_results) == 1


@pytest.mark.parametrize("with_job_result", [True, False])
def test_merge_attempt_jobs_preserves_config_only_interrupted_trials(
    tmp_path: Path,
    with_job_result: bool,
) -> None:
    from datetime import UTC, datetime
    from uuid import UUID

    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.config import TrialConfig
    from harbor.viewer.scanner import JobScanner

    job_dir = tmp_path / "attempt-001"
    trial_dir = job_dir / "case-001__interrupted"
    trial_dir.mkdir(parents=True)
    trial_config = TrialConfig.model_validate(
        {
            "task": {"path": str(job_dir / "task" / "case-001"), "source": "with"},
            "trial_name": trial_dir.name,
            "trials_dir": str(job_dir),
        }
    )
    (trial_dir / "config.json").write_text(trial_config.model_dump_json(indent=2), encoding="utf-8")
    if with_job_result:
        now = datetime(2026, 8, 25, tzinfo=UTC)
        source_job_result = JobResult(
            id=UUID(int=5),
            started_at=now,
            updated_at=now,
            finished_at=now,
            n_total_trials=1,
            stats=JobStats(n_pending_trials=1),
            trial_results=[],
        )
        (job_dir / "result.json").write_text(source_job_result.model_dump_json(indent=2), encoding="utf-8")
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([job_dir], aggregate_dir)

    scanner = JobScanner(tmp_path)
    merged = scanner.get_job_result(aggregate_dir.name)
    assert merged is not None
    assert merged.n_total_trials == 1
    assert merged.stats.n_completed_trials == 0
    assert merged.stats.n_pending_trials == 1
    assert merged.trial_results == []
    assert merged.finished_at is None
    merged_trial_name = f"{job_dir.name}__{trial_dir.name}"
    assert scanner.list_trials(aggregate_dir.name) == [merged_trial_name]
    assert scanner.get_trial_result(aggregate_dir.name, merged_trial_name) is None
    rewritten_config = scanner.get_trial_config(aggregate_dir.name, merged_trial_name)
    assert rewritten_config is not None
    assert rewritten_config.trial_name == merged_trial_name
    assert rewritten_config.trials_dir == aggregate_dir


def test_merge_attempt_jobs_ignores_tmpdir_inside_attempt_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "attempt-001"
    trial_dir = job_dir / "case-001__trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "artifact.txt").write_text("expected", encoding="utf-8")
    monkeypatch.setattr(runner.tempfile, "tempdir", str(job_dir))
    aggregate_dir = tmp_path / "aggregate"

    runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert (aggregate_dir / f"{job_dir.name}__{trial_dir.name}" / "artifact.txt").read_text() == "expected"
    assert not list(tmp_path.glob(".aggregate-merge-*"))


def test_merge_attempt_jobs_preserves_existing_aggregate_on_unsafe_source(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    safe_job = tmp_path / "attempt-001"
    safe_trial = safe_job / "case-001__trial"
    safe_trial.mkdir(parents=True)
    (safe_trial / "safe.txt").write_text("staged first", encoding="utf-8")
    unsafe_job = tmp_path / "attempt-002"
    unsafe_trial = unsafe_job / "case-001__trial"
    unsafe_trial.mkdir(parents=True)
    (unsafe_trial / "unsafe").symlink_to(outside, target_is_directory=True)
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    marker = aggregate_dir / "keep.txt"
    marker.write_text("old aggregate", encoding="utf-8")

    with pytest.raises(ValueError, match=_UNSAFE_LINK):
        runner._merge_attempt_jobs([safe_job, unsafe_job], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "old aggregate"
    assert not (aggregate_dir / f"{safe_job.name}__{safe_trial.name}").exists()
    assert not (aggregate_dir / f"{unsafe_job.name}__{unsafe_trial.name}").exists()
    assert not list(tmp_path.glob(".aggregate-merge-*"))


def test_merge_attempt_jobs_preserves_existing_aggregate_when_private_staging_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "attempt-001"
    job_dir.mkdir()
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    marker = aggregate_dir / "keep.txt"
    marker.write_text("old aggregate", encoding="utf-8")

    def fail_private_staging(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected temp failure")

    monkeypatch.setattr(runner.tempfile, "TemporaryDirectory", fail_private_staging)

    with pytest.raises(OSError, match="injected temp failure"):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "old aggregate"


@pytest.mark.parametrize("relationship", ["aggregate-in-job", "job-in-aggregate"])
def test_merge_attempt_jobs_rejects_source_destination_overlap(tmp_path: Path, relationship: str) -> None:
    if relationship == "aggregate-in-job":
        job_dir = tmp_path / "attempt-001"
        job_dir.mkdir()
        aggregate_dir = job_dir / "aggregate"
    else:
        aggregate_dir = tmp_path / "aggregate"
        job_dir = aggregate_dir / "attempt-001"
        job_dir.mkdir(parents=True)
    marker = job_dir / "keep.txt"
    marker.write_text("source", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overlap"):
        runner._merge_attempt_jobs([job_dir], aggregate_dir)

    assert marker.read_text(encoding="utf-8") == "source"


def _run(
    monkeypatch: pytest.MonkeyPatch,
    jobs_dir: Path,
    job_name: str = "demo-opencode-with",
    expected_total_trials: int | None = None,
) -> tuple[bool, str]:
    monkeypatch.setattr(
        runner,
        "build_harbor_run_command",
        lambda **_kwargs: [sys.executable, "-c", "pass"],
    )
    kwargs = {"expected_total_trials": expected_total_trials} if expected_total_trials is not None else {}
    return runner._run_harbor(
        dataset=jobs_dir / "dataset",
        agent="opencode",
        job_name=job_name,
        env_mode="docker",
        model="nvidia/openai/gpt-oss-120b",
        jobs_dir=jobs_dir,
        run_env={},
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        **kwargs,
    )


def _write_job_result(jobs_dir: Path, stats: dict[str, object], *, total: int = 1) -> None:
    job_dir = jobs_dir / "demo-opencode-with"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps({"n_total_trials": total, "stats": stats}),
        encoding="utf-8",
    )


def test_run_harbor_fails_closed_with_redacted_tail_on_output_overflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "synthetic-runner-overflow-secret"
    monkeypatch.setattr(
        runner,
        "_run_bounded_harbor_process",
        lambda *_args, **_kwargs: runner._BoundedHarborProcessResult(
            returncode=-9,
            output_tail=f"useful failure tail containing {secret}",
            output_exceeded=True,
        ),
    )

    ok, detail = runner._run_harbor(
        dataset=tmp_path / "dataset",
        agent="opencode",
        job_name="demo-opencode-with",
        env_mode="docker",
        model="nvidia/model",
        jobs_dir=tmp_path,
        run_env={"API_KEY": secret},
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert ok is False
    assert "output exceeded" in detail
    assert "useful failure tail" in detail
    assert secret not in detail
    assert "redacted" in detail.lower()


def test_run_harbor_reports_timeout_after_bounded_runner_contains_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner,
        "_run_bounded_harbor_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(runner._HarborRunTimeoutError("Harbor run timed out")),
    )

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert detail == "Harbor run timed out"


@pytest.mark.parametrize(
    ("secret", "bounded_result"),
    [
        (
            "safety",
            runner._BoundedHarborProcessResult(
                returncode=-9,
                output_tail="ordinary overflow tail",
                output_exceeded=True,
            ),
        ),
        (
            "harbor run exited 7",
            runner._BoundedHarborProcessResult(
                returncode=7,
                output_tail="",
                output_exceeded=False,
            ),
        ),
        (
            "result.json",
            runner._BoundedHarborProcessResult(
                returncode=0,
                output_tail="",
                output_exceeded=False,
            ),
        ),
    ],
)
def test_run_harbor_redacts_secrets_created_by_final_diagnostic_assembly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    secret: str,
    bounded_result: runner._BoundedHarborProcessResult,
) -> None:
    monkeypatch.setattr(
        runner,
        "build_harbor_run_command",
        lambda **_kwargs: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(runner, "_run_bounded_harbor_process", lambda *_args, **_kwargs: bounded_result)

    ok, detail = runner._run_harbor(
        dataset=tmp_path / "dataset",
        agent="opencode",
        job_name="demo-opencode-with",
        env_mode="docker",
        model="nvidia/model",
        jobs_dir=tmp_path,
        run_env={"API_KEY": secret},
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert ok is False
    assert secret not in detail
    assert "redacted" in detail.lower()


def test_run_harbor_uses_collision_safe_marker_for_synthesized_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    synthesized_secret = "<redacted>"
    monkeypatch.setattr(
        runner,
        "build_harbor_run_command",
        lambda **_kwargs: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(
        runner,
        "_run_bounded_harbor_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(runner._HarborRunTimeoutError("Harbor run timed out")),
    )

    ok, detail = runner._run_harbor(
        dataset=tmp_path / "dataset",
        agent="opencode",
        job_name="demo-opencode-with",
        env_mode="docker",
        model="nvidia/model",
        jobs_dir=tmp_path,
        run_env={
            "FIRST_API_KEY": "Harbor run timed out",
            "SECOND_API_KEY": synthesized_secret,
        },
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert ok is False
    assert "Harbor run timed out" not in detail
    assert synthesized_secret not in detail


def _complete_stats(**overrides: object) -> dict[str, object]:
    stats: dict[str, object] = {
        "n_trials": 1,
        "n_errors": 0,
        "evals": {
            "codex__model___harbor-tasks": {
                "n_trials": 1,
                "n_errors": 0,
                "reward_stats": {"reward": {"1.0": ["case-001__abc"]}},
            }
        },
    }
    stats.update(overrides)
    return stats


def test_run_harbor_rejects_missing_job_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert "result.json" in detail


def test_run_harbor_rejects_zero_trials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_job_result(tmp_path, _complete_stats(n_trials=0, evals={}), total=0)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert "zero trials" in detail


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"n_errors": 1}, "1 errored"),
        ({"n_trials": 0}, "completed 0/1"),
        ({"n_trials": 2}, "completed 2/1"),
    ],
)
def test_run_harbor_rejects_non_successful_trial_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, object],
    expected: str,
) -> None:
    _write_job_result(tmp_path, _complete_stats(**overrides))

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert expected in detail


def test_run_harbor_accepts_complete_successful_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_job_result(tmp_path, _complete_stats())

    assert _run(monkeypatch, tmp_path) == (True, "")


def test_run_harbor_accepts_pinned_harbor_0132_job_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stats = _complete_stats()
    stats.pop("n_trials")
    stats.pop("n_errors")
    stats.update(
        {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": 0,
        }
    )
    _write_job_result(tmp_path, stats)

    assert _run(monkeypatch, tmp_path) == (True, "")


@pytest.mark.parametrize(
    ("counter", "expected"),
    [
        ("n_errored_trials", "1 errored"),
        ("n_running_trials", "1 running"),
        ("n_pending_trials", "1 pending"),
        ("n_cancelled_trials", "1 cancelled"),
    ],
)
def test_run_harbor_rejects_incomplete_pinned_harbor_0132_job_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    counter: str,
    expected: str,
) -> None:
    stats = _complete_stats()
    stats.pop("n_trials")
    stats.pop("n_errors")
    stats.update(
        {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": 0,
        }
    )
    stats[counter] = 1
    _write_job_result(tmp_path, stats)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert expected in detail


def test_job_result_must_match_requested_trial_count(tmp_path: Path) -> None:
    _write_job_result(tmp_path, _complete_stats())

    ok, detail = runner._validate_harbor_job_result(
        tmp_path,
        "demo-opencode-with",
        expected_trials=2,
    )

    assert ok is False
    assert "declared 1 trials; expected 2" in detail


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({"n_trials": True, "n_errors": 0, "evals": {}}, "invalid n_trials"),
        ({"n_trials": 1, "n_errors": -1, "evals": {}}, "invalid n_errors"),
        ({"n_trials": 1, "n_errors": 0, "evals": {}}, "no evaluation statistics"),
        (
            {
                "n_trials": 1,
                "n_errors": 0,
                "evals": {"eval": {"n_trials": 0, "n_errors": 0, "reward_stats": {}}},
            },
            "account for 0/1",
        ),
        (
            {
                "n_trials": 1,
                "n_errors": 0,
                "evals": {"eval": {"n_trials": 1, "n_errors": 0, "reward_stats": {}}},
            },
            "no scored trial names",
        ),
    ],
)
def test_run_harbor_rejects_incomplete_real_harbor_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stats: dict[str, object],
    expected: str,
) -> None:
    _write_job_result(tmp_path, stats)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert expected in detail


def test_run_harbor_rejects_reward_coverage_shortfall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stats = _complete_stats()
    stats["n_trials"] = 2
    stats["evals"] = {
        "eval": {
            "n_trials": 2,
            "n_errors": 0,
            "reward_stats": {"reward": {"1.0": ["case-001__abc"]}},
        }
    }
    _write_job_result(tmp_path, stats, total=2)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert "cover 1/2" in detail


def test_run_harbor_rejects_duplicate_rewarded_trial_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stats = _complete_stats()
    stats["evals"] = {
        "eval": {
            "n_trials": 1,
            "n_errors": 0,
            "reward_stats": {"reward": {"1.0": ["case-001__abc", "case-001__abc"]}},
        }
    }
    _write_job_result(tmp_path, stats)

    ok, detail = _run(monkeypatch, tmp_path)

    assert ok is False
    assert "duplicate rewarded trial names" in detail
