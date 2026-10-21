# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for check_error_recovery unrecovered-failure scoring (issue #149).

An observed command failure that is never followed by a successful similar
command must NOT be scored as first-attempt clean.
"""

from __future__ import annotations

import pytest

from skillevaluator.tier3.eval_core.checks import check_error_recovery
from skillevaluator.tier3.harbor.report import _collect_fail_reasons
from skillevaluator.tier3.harbor.templates import eval as eval_template


@pytest.fixture(params=[check_error_recovery, eval_template.check_error_recovery], ids=["host", "harbor-verifier"])
def checker(request):
    def check(*args, **kwargs):
        result = request.param(*args, **kwargs)
        peer = eval_template.check_error_recovery if request.param is check_error_recovery else check_error_recovery
        assert result == peer(*args, **kwargs)
        return result

    return check


def _call(cmd: str, observation: str) -> dict:
    return {"action": "bash", "action_input": {"command": cmd}, "observation": observation}


def _codex_output(exit_code: int, output: str = "") -> str:
    return (
        "Chunk ID: a1b2c3\n"
        "Wall time: 0.125 seconds\n"
        f"Process exited with code {exit_code}\n"
        f"Original token count: {len(output.split())}\n"
        f"Output:\n{output}"
    )


def test_failure_with_no_retry_is_not_first_attempt_clean(checker):
    """A failure that is never retried must not score 1.0 / passed-clean."""
    result = checker([_call("pytest tests/test_api.py -x", "ERROR: test_create_user failed\nexit code 1")])
    assert result["first_attempt_clean"] is False
    assert result["score"] == 0.0
    assert len(result["unrecovered_failures"]) == 1
    assert result["unrecovered_failures"][0]["failed_cmd"] == "pytest tests/test_api.py -x"
    assert result["unrecovered_failures"][0]["recovered"] is False
    assert "1 unrecovered failure" in result["reason"]
    assert "pytest tests/test_api.py -x" in result["reason"]


def test_failed_retries_are_not_first_attempt_clean(checker):
    """Repeated failures of a similar command are all recorded as unrecovered."""
    result = checker(
        [
            _call("pytest tests/test_api.py -x", "ERROR: test_create_user failed\nexit code 1"),
            _call("pytest tests/test_api.py -x -v", "ERROR: test_create_user failed\nexit code 1"),
        ]
    )
    assert result["first_attempt_clean"] is False
    assert result["score"] == 0.0
    assert len(result["failures"]) == 2
    assert len(result["unrecovered_failures"]) == 2
    assert result["corrections"] == []


def test_recovered_failure_behavior_unchanged(checker):
    """A failure followed by a successful similar command keeps the old scoring."""
    result = checker(
        [
            _call("pytest tests/test_api.py", "pytest: error: unrecognized arguments: --bogus\nexit code 2"),
            _call("pytest tests/test_api.py -q", "12 passed"),
        ]
    )
    assert result["first_attempt_clean"] is False
    assert result["unrecovered_failures"] == []
    assert len(result["corrections"]) == 1
    assert result["corrections"][0]["failed_cmd"] == "pytest tests/test_api.py"
    assert result["corrections"][0]["retry_cmd"] == "pytest tests/test_api.py -q"
    # Unchanged pre-existing scoring: one corrected agent fault -> 0.9, passed.
    assert result["score"] == 0.9
    assert result["passed"] is True


def test_unrecovered_failure_after_earlier_success(checker):
    """An unrecovered failure later in the log still breaks first_attempt_clean."""
    result = checker(
        [
            _call("ls -la", "total 4\ndrwxr-xr-x  2 root root 4096 ."),
            _call("make build", "gcc: error: missing.c: No such file or directory"),
        ]
    )
    assert result["first_attempt_clean"] is False
    assert result["score"] == 0.0
    assert len(result["failures"]) == 1
    assert len(result["unrecovered_failures"]) == 1
    assert result["unrecovered_failures"][0]["fault"] == "skill"
    assert "make build" in result["reason"]


def test_unrecovered_failure_after_earlier_recovery(checker):
    result = checker(
        [
            _call("pytest tests/test_api.py", "pytest: error: unrecognized arguments: --bogus\nexit code 2"),
            _call("pytest tests/test_api.py -q", "12 passed"),
            _call("make build", "gcc: error: missing.c: No such file or directory"),
        ]
    )
    assert len(result["corrections"]) == 1
    assert result["first_attempt_clean"] is False
    assert [f["failed_cmd"] for f in result["unrecovered_failures"]] == ["make build"]
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["skill_faults"] == 1
    assert result["agent_faults"] == 1


@pytest.mark.parametrize(
    "observation",
    [
        _codex_output(0, "0 errors"),
        "exit_code=0\nERROR: example diagnostic",
        "exit code: 0\nTraceback (most recent call last):",
        "0 errors",
        "No errors found",
        "error handling example",
        "timeout configured to 30 seconds",
        "try:\n    open(p)\nexcept FileNotFoundError:\n    pass",
        "server:\n  timeout: 30",
    ],
)
def test_successful_output_is_not_failure(checker, observation):
    result = checker([_call("make build", observation)])
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["failures"] == []


@pytest.mark.parametrize(
    "observation",
    [
        _codex_output(7, "All done"),
        "exit_code=127",
        "exit code: 42",
        "exit code -9",
        "Timeout: command exceeded 30 seconds",
        "TimeoutError",
        "FileNotFoundError",
        "bash: foo: not found",
        "ConnectionRefused",
        "timeout",
        "ValueError: invalid value",
        "Traceback (most recent call last):",
        "status=failed",
        "command not found",
    ],
)
def test_failure_detection_agrees(checker, observation):
    result = checker([_call("run.py", observation)])
    assert result["score"] == 0.0
    assert result["first_attempt_clean"] is False
    assert len(result["unrecovered_failures"]) == 1
    assert result["skill_faults"] + result["agent_faults"] == 1


def test_unrecovered_skill_fault_report_reason(checker):
    result = checker([_call("make build", "gcc: error: missing.c: No such file or directory")])
    assert result["passed"] is False
    assert result["skill_faults"] == 1
    assert result["agent_faults"] == 0
    reasons = _collect_fail_reasons("skill_execution", [{"detail": {"error_recovery": result}}])
    assert result["reason"] in reasons
    assert "make build" in result["reason"]


def test_recovered_skill_fault_counted_once(checker):
    result = checker(
        [
            _call("make build", "No such file or directory"),
            _call("make build", "done"),
        ]
    )
    assert result["skill_faults"] == 1
    assert result["score"] == 0.75
    assert result["passed"] is False
    assert result["unrecovered_failures"] == []


@pytest.mark.parametrize(
    "observation",
    [
        "ERROR: test failed\nexit code 1",
        "Traceback (most recent call last):\nValueError: bad input",
    ],
)
def test_unrecovered_agent_fault_report_reason(checker, observation):
    result = checker([_call("pytest", observation)])
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["agent_faults"] == 1
    assert result["skill_faults"] == 0
    reasons = _collect_fail_reasons("skill_execution", [{"detail": {"error_recovery": result}}])
    assert result["reason"] in reasons


@pytest.mark.parametrize(
    "observation",
    [
        "def failure(stage, message, exit_code=1):",
        "    translate unexpected failures to exit code 2",
        "try:\n    open(p)\nexcept Exception:\n    pass",
        'print("Error: expected <spec_url>")',
        'f"{self.prog}: error: {message}"',
    ],
)
def test_displayed_source_is_not_execution_failure(checker, observation):
    result = checker([_call("cat loader.py", observation)])
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["failures"] == []


@pytest.mark.parametrize("inspections", [4, 5, 6, 8])
def test_recovery_after_intervening_inspections(checker, inspections):
    calls = [_call("pytest -q", _codex_output(1, "ERROR: test failed"))]
    calls.extend(_call(f"sed -n '1,20p' file{i}.py", _codex_output(0, "source")) for i in range(inspections))
    calls.append({"action": "apply_patch", "action_input": {}, "observation": "Done"})
    calls.append(_call("pytest -q", _codex_output(0, "12 passed")))
    result = checker(calls)
    assert result["passed"] is True
    assert result["score"] == 0.9
    assert len(result["corrections"]) == 1
    assert result["corrections"][0]["steps_to_fix"] == inspections + 2
    assert result["unrecovered_failures"] == []


@pytest.mark.parametrize(
    "command",
    [
        "rg TODO src",
        "grep needle file.txt",
        "test -f out.json",
        "[ -e out.json ]",
        "diff -q a b",
        "cmp -s a b",
        "/usr/bin/grep needle file.txt",
    ],
)
@pytest.mark.parametrize("observation", ["Exit code 1", "status=completed\nexit_code=1", _codex_output(1)])
def test_quiet_negative_probe_is_not_failure(checker, command, observation):
    result = checker([_call(command, observation)])
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["failures"] == []


@pytest.mark.parametrize(
    "command,observation",
    [
        ("pytest -q", "Exit code 1"),
        ("python run.py", "Exit code 1"),
        ("rg TODO src", "Exit code 2"),
        ("rg TODO src", "Exit code 1\nrg: src: Permission denied"),
        ("test -f out.json", "Exit code 1\ntest: invalid argument"),
        ("rg TODO src; python run.py", "Exit code 1"),
        ("rg TODO src && python run.py", "Exit code 1"),
        ("rg TODO src\npython run.py", "Exit code 1"),
        ("rg TODO src | python run.py", "Exit code 1"),
        ("git status", "Exit code 128\nfatal: not a git repository"),
    ],
)
def test_quiet_probe_exception_does_not_hide_real_failures(checker, command, observation):
    result = checker([_call(command, observation)])
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert len(result["unrecovered_failures"]) == 1


@pytest.mark.parametrize(
    "status", ["Exit code 2", "exit_code=2", "[exit_code=2]", "[exit_code] 2", "Process exited with code 2"]
)
def test_whole_status_lines_remain_failures(checker, status):
    result = checker([_call("python run.py", status)])
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_negative_probe_is_not_successful_recovery(checker):
    result = checker([_call("rg TODO src", "Exit code 2\nrg: Permission denied"), _call("rg TODO src", "Exit code 1")])
    assert result["passed"] is False
    assert result["corrections"] == []
    assert len(result["failures"]) == 1
    assert len(result["unrecovered_failures"]) == 1


@pytest.mark.parametrize(
    "command,retry,inspections,observation",
    [
        ("pytest -q", "test -f report.json", 6, "Exit code 1"),
        ("pytest tests/test_tools.py", "ls -la", 6, "Exit code 1"),
        ("pytest -q", "cat pytest.ini", 8, "Exit code 1"),
        (
            "python scripts/convert.py in.pdf out.json",
            'python3 -c "print(1)"',
            6,
            "Traceback (most recent call last):\nValueError: bad input",
        ),
        ("python scripts/convert.py in.pdf out.json", 'python -c "print(1)"', 6, "Exit code 1"),
        ("pytest tests/test_api.py", "pytest tests/test_other.py", 6, "Exit code 1"),
    ],
)
def test_unrelated_distant_success_does_not_recover(checker, command, retry, inspections, observation):
    calls = [_call(command, observation)]
    calls.extend(_call(f"head -n 5 source{i}.py", _codex_output(0, "source")) for i in range(inspections))
    calls.append(_call(retry, _codex_output(0)))
    result = checker(calls)
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["corrections"] == []
    assert [failure["failed_cmd"] for failure in result["unrecovered_failures"]] == [command]


@pytest.mark.parametrize("command", ["pytest -q", "python scripts/convert.py in.pdf out.json"])
def test_same_command_recovers_after_distant_edit(checker, command):
    calls = [_call(command, _codex_output(1, "ValueError: bad input"))]
    calls.extend(_call(f"head -n 5 source{i}.py", _codex_output(0, "source")) for i in range(8))
    calls.append({"action": "apply_patch", "action_input": {}, "observation": "Done"})
    calls.append(_call(command, _codex_output(0, "done")))
    result = checker(calls)
    assert result["score"] == 0.9
    assert result["passed"] is True
    assert result["corrections"][0]["steps_to_fix"] == 10
    assert result["unrecovered_failures"] == []


@pytest.mark.parametrize("command", ["rg TODO src", "test -f out.json"])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_codex_quiet_probe_header_is_not_diagnostic(checker, command, line_ending):
    result = checker([_call(command, _codex_output(1).replace("\n", line_ending))])
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["failures"] == []


@pytest.mark.parametrize(
    "output",
    [
        "Permission denied",
        "status=failed",
        "exit_code=2",
        "Output:",
        "Chunk ID: a1b2c3",
        "Wall time: 0.125 seconds",
        "Original token count: 0",
    ],
)
def test_codex_probe_output_is_never_stripped_as_metadata(checker, output):
    result = checker([_call("rg TODO src", _codex_output(1, output))])
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize(
    "command,exit_code,output",
    [
        ("rg TODO src; python run.py", 1, ""),
        ("rg TODO src | python run.py", 1, ""),
        ('rg "$(python run.py)" src', 1, ""),
        ("rg TODO src", 2, ""),
        ("python run.py", 1, ""),
        ("test -f out.json", 1, "test: invalid argument"),
    ],
)
def test_codex_probe_header_does_not_hide_real_failures(checker, command, exit_code, output):
    result = checker([_call(command, _codex_output(exit_code, output))])
    assert result["score"] == 0.0
    assert result["passed"] is False


def test_codex_quiet_probe_is_not_successful_retry(checker):
    result = checker(
        [_call("rg TODO src", _codex_output(2, "Permission denied")), _call("rg TODO src", _codex_output(1))]
    )
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert len(result["failures"]) == 1
    assert result["corrections"] == []
    assert len(result["unrecovered_failures"]) == 1


@pytest.mark.parametrize("command,retry", [("pytest  -q", "pytest -q"), ("pytest\t-q", "pytest -q")])
def test_distant_retry_ignores_only_whitespace(checker, command, retry):
    calls = [_call(command, "Exit code 1")]
    calls.extend(_call(f"head -n 5 source{i}.py", _codex_output(0, "source")) for i in range(6))
    calls.append(_call(retry, _codex_output(0, "passed")))
    result = checker(calls)
    assert result["score"] == 0.9
    assert result["passed"] is True
    assert result["corrections"][0]["steps_to_fix"] == 7
    assert result["unrecovered_failures"] == []


@pytest.mark.parametrize(
    "command,retry",
    [
        ("pytest tests/test_app.py -x", "pytest tests/test_app.py"),
        ("cd /app && pytest -q", "pytest -q"),
        ("pytest -q 2>&1 | tail -20", "pytest -q 2>&1 | tail -50"),
        ("", ""),
        ("", " \t "),
    ],
)
def test_distant_retry_keeps_tokens_strict_and_nonempty(checker, command, retry):
    calls = [_call(command, "Exit code 1")]
    calls.extend(_call(f"head -n 5 source{i}.py", _codex_output(0, "source")) for i in range(6))
    calls.append(_call(retry, _codex_output(0)))
    result = checker(calls)
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["corrections"] == []
    assert len(result["unrecovered_failures"]) == 1


@pytest.mark.parametrize("has_chunk_id,has_token_count", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
@pytest.mark.parametrize("output", ["", "Permission denied", "status=failed", "Original token count: 0"])
def test_codex_optional_headers_preserve_output(checker, has_chunk_id, has_token_count, line_ending, output):
    observation = _codex_output(1, output)
    if not has_chunk_id:
        observation = observation.replace("Chunk ID: a1b2c3\n", "", 1)
    if not has_token_count:
        observation = observation.replace(f"Original token count: {len(output.split())}\n", "", 1)
    observation = observation.replace("\n", line_ending)
    result = checker([_call("rg TODO src", observation)])
    assert result["passed"] is (not output)
    assert result["score"] == (0.0 if output else 1.0)
    assert len(result["failures"]) == (1 if output else 0)


@pytest.fixture(scope="module")
def claude_error_observation():
    from harbor.agents.installed.claude_code import ClaudeCode

    observation, metadata = ClaudeCode._format_tool_result(
        {"type": "tool_result", "content": "Exit code 1", "is_error": True},
        "Error: Exit code 1",
    )
    assert observation == "Exit code 1\n\n[error] tool reported failure"
    assert metadata["is_error"] is True
    return observation


@pytest.mark.parametrize(
    "command", ["rg TODO src", "grep needle file.txt", "test -f out.json", "[ -e out.json ]", "diff a b", "cmp -s a b"]
)
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_claude_error_marker_allows_quiet_negative_probe(checker, claude_error_observation, command, line_ending):
    observation = claude_error_observation.replace("\n", line_ending)
    result = checker([_call(command, observation)])
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["failures"] == []


@pytest.mark.parametrize(
    "command,exit_code,extra",
    [
        ("rg TODO src", 1, "rg: src: Permission denied"),
        ("rg TODO src", 1, "[error] tool reported failure\nrg: src: Permission denied"),
        ("rg TODO src", 1, "[stderr] Permission denied"),
        ("rg TODO src", 1, '[metadata] {"interrupted": true}'),
        ("rg TODO src", 1, "[error] tool reported failure: interrupted"),
        ("rg TODO src", 1, "[ERROR] tool reported failure"),
        ("rg TODO src", 1, " [error] tool reported failure"),
        ("rg TODO src", 2, ""),
        ("python run.py", 1, ""),
        ("rg TODO src; python run.py", 1, ""),
        ("rg TODO src && python run.py", 1, ""),
        ("rg TODO src | python run.py", 1, ""),
        ('rg "$(python run.py)" src', 1, ""),
        ("rg `python run.py` src", 1, ""),
    ],
)
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_claude_error_marker_does_not_hide_failure(
    checker, claude_error_observation, command, exit_code, extra, line_ending
):
    observation = claude_error_observation.replace("Exit code 1", f"Exit code {exit_code}", 1)
    if extra:
        observation += "\n" + extra
    result = checker([_call(command, observation.replace("\n", line_ending))])
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert len(result["unrecovered_failures"]) == 1


def test_claude_quiet_negative_probe_is_not_successful_retry(checker, claude_error_observation):
    result = checker(
        [
            _call("rg TODO src", claude_error_observation.replace("Exit code 1", "Exit code 2", 1)),
            _call("rg TODO src", claude_error_observation),
        ]
    )
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["corrections"] == []
    assert len(result["failures"]) == 1
    assert len(result["unrecovered_failures"]) == 1


def test_codex_output_does_not_strip_claude_marker(checker):
    result = checker([_call("rg TODO src", _codex_output(1, "[error] tool reported failure"))])
    assert result["passed"] is False
    assert result["score"] == 0.0
