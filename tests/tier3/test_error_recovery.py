# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for check_error_recovery unrecovered-failure scoring (issue #149).

An observed command failure that is never followed by a successful similar
command must NOT be scored as first-attempt clean.
"""

from __future__ import annotations

import pytest

from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts
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


@pytest.fixture(
    params=[
        (extract_tool_calls_as_dicts, check_error_recovery),
        (eval_template.extract_tool_calls_as_dicts, eval_template.check_error_recovery),
    ],
    ids=["host", "harbor-verifier"],
)
def trajectory_checker(request):
    """Extract tool calls from an ATIF trajectory and score them with one implementation."""
    extract, check = request.param

    def run(trajectory: dict) -> dict:
        return check(extract(trajectory))

    run.extract = extract
    return run


def _claude_code_trajectory(*commands: tuple[str, str, int]) -> dict:
    """Build Harbor Claude Code ATIF steps for (command, output, exit code) tuples."""
    from harbor.agents.installed.claude_code import ClaudeCode

    steps = []
    for index, (command, output, exit_code) in enumerate(commands):
        if exit_code:
            content = f"Exit code {exit_code}\n{output}".rstrip()
            block = {"type": "tool_result", "content": content, "is_error": True}
            observation, metadata = ClaudeCode._format_tool_result(block, f"Error: {content}")
        else:
            block = {"type": "tool_result", "content": output, "is_error": False}
            tool_use_result = {"stdout": output, "stderr": "", "interrupted": False, "isImage": False}
            observation, metadata = ClaudeCode._format_tool_result(block, tool_use_result)
        call_id = f"toolu_{index}"
        steps.append(
            {
                "step_id": index + 1,
                "source": "agent",
                "message": "",
                "tool_calls": [{"tool_call_id": call_id, "function_name": "Bash", "arguments": {"command": command}}],
                "observation": {
                    "results": [
                        {
                            "source_call_id": call_id,
                            "content": observation,
                            # Harbor stores the result's is_error flag here.
                            "extra": {"tool_result_metadata": metadata, "tool_result_is_error": block["is_error"]},
                        }
                    ]
                },
            }
        )
    return {"schema_version": "ATIF-v1.7", "agent": {"name": "claude-code"}, "steps": steps}


@pytest.mark.parametrize(
    "command,output",
    [
        (
            'python3 -m py_compile repository.py && echo OK; python3 -c "import sqlalchemy" 2>&1 | tail -1',
            "OK\nModuleNotFoundError: No module named 'sqlalchemy'",
        ),
        ("python scripts/fetch.py --url https://example.com/missing", '{"error": "HTTP 404: Not Found"}'),
        ("cat scripts/call_api.py", 'result["error"] = f"Request timed out after {timeout} seconds"'),
        ("cat scripts/parse.py", "    Raises:\n        ValueError: If the spec is malformed."),
        ("ls missing; ls skills", "ls: missing: No such file or directory\nreport-builder"),
        ("cat build.log", "step 3\nexit code 1\nretrying"),
    ],
)
def test_claude_code_success_flag_skips_text_check(trajectory_checker, command, output):
    trajectory = _claude_code_trajectory((command, output, 0))
    assert trajectory_checker.extract(trajectory)[0]["is_error"] is False
    result = trajectory_checker(trajectory)
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["failures"] == []


def test_claude_code_success_flag_counts_as_recovery(trajectory_checker):
    trajectory = _claude_code_trajectory(
        ("python scripts/build.py", "Traceback (most recent call last):\nKeyError: 'x'", 1),
        ("python scripts/build.py --in data.csv", "lint step failed, skipped\nexit code 1\nwrote report.html", 0),
    )
    result = trajectory_checker(trajectory)
    assert result["score"] == 0.9
    assert result["passed"] is True
    assert len(result["corrections"]) == 1
    assert result["unrecovered_failures"] == []


@pytest.mark.parametrize(
    "command,output,exit_code",
    [
        ("pytest -q", "F.\nFAILED tests/test_a.py::test_a - assert 1 == 2\n1 failed, 1 passed", 1),
        ("python run.py", "Traceback (most recent call last):\nKeyError: 'x'", 1),
        ("npx tsc --noEmit", "src/a.ts(3,7): error TS2322: Type 'string' is not assignable to type 'number'.", 2),
        ("python run.py", "", 1),
    ],
)
def test_claude_code_error_flag_still_fails(trajectory_checker, command, output, exit_code):
    trajectory = _claude_code_trajectory((command, output, exit_code))
    assert trajectory_checker.extract(trajectory)[0]["is_error"] is True
    result = trajectory_checker(trajectory)
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert len(result["unrecovered_failures"]) == 1


@pytest.mark.parametrize(
    "content,score",
    [
        ("Command timed out after 2m 0.0s", 0.0),
        ("Permission to use Bash with command rm -rf build has been denied.", 1.0),
    ],
)
def test_claude_code_error_flag_without_exit_status_uses_text(trajectory_checker, content, score):
    trajectory = _claude_code_trajectory(("rm -rf build", "", 0))
    result_entry = trajectory["steps"][0]["observation"]["results"][0]
    result_entry["content"] = f"{content}\n\n[error] tool reported failure"
    result_entry["extra"] = {"tool_result_is_error": True}
    assert trajectory_checker.extract(trajectory)[0]["is_error"] is True
    assert trajectory_checker(trajectory)["score"] == score


def test_unflagged_result_keeps_text_check(trajectory_checker):
    trajectory = _claude_code_trajectory(("python run.py", "", 0))
    result_entry = trajectory["steps"][0]["observation"]["results"][0]
    result_entry["content"] = "Traceback (most recent call last):\nKeyError: 'x'"
    del result_entry["extra"]
    assert "is_error" not in trajectory_checker.extract(trajectory)[0]
    result = trajectory_checker(trajectory)
    assert result["score"] == 0.0
    assert result["passed"] is False


def _opencode_trajectory(tmp_path, *commands: tuple[str, str, int | None]) -> dict:
    """Convert OpenCode bash events with Harbor's converter, which keeps only the output."""
    from harbor.agents.installed.opencode import OpenCode

    events = []
    for index, (command, output, exit_code) in enumerate(commands):
        state = {
            "status": "completed",
            "input": {"command": command, "description": "run"},
            "output": output,
            "metadata": {"exit": exit_code, "output": output, "description": "run"},
        }
        events.extend(
            [
                {"type": "step_start", "timestamp": 1000 + index, "sessionID": "s1", "part": {}},
                {
                    "type": "tool_use",
                    "timestamp": 1000 + index,
                    "sessionID": "s1",
                    "part": {"type": "tool", "tool": "bash", "callID": f"call_{index}", "state": state},
                },
                {
                    "type": "step_finish",
                    "timestamp": 1000 + index,
                    "sessionID": "s1",
                    "part": {"tokens": {}, "cost": 0},
                },
            ]
        )
    agent = OpenCode(logs_dir=tmp_path, model_name="openai/gpt-5")
    return agent._convert_events_to_trajectory(events).to_json_dict()


OPENCODE_PYTEST_FAIL = (
    "F.\n=================================== FAILURES ===================================\n"
    "___ test_total ___\n\n    def test_total():\n>       assert total([1, 2]) == 4\nE       assert 3 == 4\n\n"
    "tests/test_calc.py:4: AssertionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED tests/test_calc.py::test_total - assert 3 == 4\n1 failed, 1 passed in 0.03s\n"
)
OPENCODE_FAILURES = {
    "pytest": ("pytest -q", OPENCODE_PYTEST_FAIL, 1),
    "pytest-collection": (
        "pytest -q",
        "==== short test summary info ====\nERROR tests/test_calc.py\n!!! Interrupted: 1 error during collection !!!\n",
        2,
    ),
    "unittest": ("python -m unittest", "F\n----\nRan 1 test in 0.001s\n\nFAILED (failures=1)\n", 1),
    "tsc": (
        "npx tsc --noEmit",
        "src/a.ts(3,7): error TS2322: Type 'string' is not assignable to type 'number'.\n\nFound 1 error in src/a.ts:3\n",
        2,
    ),
    "jest": (
        "npm test",
        "> demo@1.0.0 test\n> jest\n\n FAIL  src/sum.test.js\n  x adds 1 + 2 to equal 3 (3 ms)\n\nTests:       1 failed, 1 total\n",
        1,
    ),
    "npm-err": ("npm install", "npm ERR! code E404\nnpm ERR! 404 Not Found - GET https://registry.npmjs.org/nope\n", 1),
    "npm-error": ("npm run build", "npm error Lifecycle script `build` failed with error:\nnpm error code 1\n", 1),
    "go-test": (
        "go test ./...",
        "--- FAIL: TestTotal (0.00s)\n    calc_test.go:8: got 3, want 4\nFAIL\nFAIL\tdemo\t0.01s\n",
        1,
    ),
    "timeout": (
        "python long_job.py",
        "partial output\n\n<bash_metadata>\nbash tool terminated command after exceeding timeout 120000 ms\n</bash_metadata>",
        None,
    ),
}


@pytest.mark.parametrize("name", sorted(OPENCODE_FAILURES))
def test_opencode_unrecovered_test_and_build_failures(trajectory_checker, tmp_path, name):
    trajectory = _opencode_trajectory(tmp_path, OPENCODE_FAILURES[name])
    assert "is_error" not in trajectory_checker.extract(trajectory)[0]
    result = trajectory_checker(trajectory)
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["first_attempt_clean"] is False
    assert len(result["unrecovered_failures"]) == 1


@pytest.mark.parametrize("name", ["pytest", "tsc"])
def test_opencode_recovered_test_and_build_failures(trajectory_checker, tmp_path, name):
    failure = OPENCODE_FAILURES[name]
    trajectory = _opencode_trajectory(tmp_path, failure, (failure[0], "..\n2 passed in 0.02s\n", 0))
    result = trajectory_checker(trajectory)
    assert result["score"] == 0.9
    assert result["passed"] is True
    assert result["first_attempt_clean"] is False
    assert len(result["corrections"]) == 1


@pytest.mark.parametrize(
    "output",
    [
        "..\n2 passed in 0.02s\n",
        "Tests:       0 failed, 4 passed, 4 total\n",
        "test result: ok. 3 passed; 0 failed; 0 ignored\n",
        "FAIL_FAST = True\nFAILED = 'failed'\nERROR = 2\n",
        "2 passed, 1 xfailed in 0.04s\n",
    ],
)
def test_opencode_passing_test_output_is_not_failure(trajectory_checker, tmp_path, output):
    result = trajectory_checker(_opencode_trajectory(tmp_path, ("pytest -q", output, 0)))
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["failures"] == []


@pytest.mark.parametrize(
    "observation",
    [
        'def call_api(url: str, method: str = "GET", timeout: int = 30) -> dict:',
        "    Args:\n        --timeout: Request timeout in seconds.",
        "class Settings(BaseModel):\n    retries: int = 3\n    timeout: float = 30.0",
        "@dataclass\nclass Result:\n    ok: bool\n    error: str | None = None",
        "export interface Result {\n  ok: boolean;\n  error: string | null;\n}",
        "def run(\n    cmd: list[str],\n    timeout: int,\n    last_error: Optional[str] = None,\n):",
    ],
)
def test_displayed_annotations_without_status_are_not_failures(checker, observation):
    result = checker([_call("cat scripts/call_api.py", observation)])
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["failures"] == []


@pytest.mark.parametrize(
    "observation",
    [
        "timeout: invalid time interval 'abc'",
        "error: externally-managed-environment",
        "error: unrecognized arguments: --bogus",
        "RuntimeError: boom",
        "ValueError: invalid literal for int() with base 10: 'x'",
        "requests.exceptions.Timeout: HTTPSConnectionPool(host='api.example.com', port=443)",
        "Error: Cannot find module 'jest'",
    ],
)
def test_diagnostics_without_status_still_fail(checker, observation):
    result = checker([_call("python run.py", observation)])
    assert result["score"] == 0.0
    assert result["passed"] is False


def _exit_one_observations(output: str, exit_code: int = 1) -> list[str]:
    """The same failed command as a plain status line, a Harbor Claude Code result and a Codex result."""
    from harbor.agents.installed.claude_code import ClaudeCode

    content = f"Exit code {exit_code}\n{output}".rstrip()
    claude_code, _ = ClaudeCode._format_tool_result(
        {"type": "tool_result", "content": content, "is_error": True}, f"Error: {content}"
    )
    return [content, claude_code, _codex_output(exit_code, output)]


@pytest.mark.parametrize(
    "command,output",
    [
        ("grep -c TODO a.py", "0"),
        ("grep -rc TODO src", "src/a.py:0\nsrc/b.py:0"),
        ("rg --count TODO src", ""),
        ("grep -rn TODO src 2>/dev/null", ""),
        ("grep -q foo a.txt >/dev/null 2>&1", ""),
        ("rg TODO src &>/dev/null", ""),
        ("cd /app && grep -rn TODO src", ""),
        ('cd "my app"; rg TODO', ""),
        ("pwd && rg --files -g AGENTS.md .", "/workspace"),
        ("cd /app && pwd; grep -rn TODO src", "/app"),
        ("LC_ALL=C grep -q foo a.txt", ""),
        ("ls | grep foo", ""),
        ("ls src | grep -c test_", "0"),
        ("test -f out.json && echo exists", ""),
        ("[[ -f out.json ]]", ""),
        ("command -v jq", ""),
        ("which jq", ""),
        ("git diff --quiet", ""),
        ("git diff --quiet -- src/app.py", ""),
    ],
)
def test_quiet_probe_variants_are_not_failures(checker, command, output):
    for observation in _exit_one_observations(output):
        result = checker([_call(command, observation)])
        assert result["score"] == 1.0, observation
        assert result["passed"] is True
        assert result["failures"] == []


@pytest.mark.parametrize(
    "command,output,exit_code",
    [
        ("grep TODO a.py", "0", 1),
        ("grep -c TODO a.py", "0\ngrep: b.py: Permission denied", 1),
        ("grep -c TODO a.py", "3", 1),
        ("grep -rn TODO src 2>/dev/null", "", 2),
        ("cd /app && rg TODO src", "rg: src: No such file or directory", 1),
        ("cd /app && python run.py", "", 1),
        ("cd /app && grep -rn TODO src; python run.py", "", 1),
        ("pwd && rg --files -g AGENTS.md .", "/workspace\nrg: ./private: Permission denied", 1),
        ("pwd; python run.py", "/workspace", 1),
        ("ls | python run.py", "", 1),
        ("test -f out.json && python run.py", "", 1),
        ("pytest -q 2>/dev/null", "", 1),
        ("command jq", "", 1),
        ("which jq", "which: no jq in (/usr/bin:/bin)", 1),
        ("git diff", "", 1),
        ("git diff --exit-code", "diff --git a/app.py b/app.py", 1),
        ("git status --short", "fatal: not a git repository (or any of the parent directories): .git", 128),
    ],
)
def test_quiet_probe_variants_keep_real_failures(checker, command, output, exit_code):
    for observation in _exit_one_observations(output, exit_code):
        result = checker([_call(command, observation)])
        assert result["score"] == 0.0, observation
        assert result["passed"] is False
        assert len(result["unrecovered_failures"]) == 1


@pytest.mark.parametrize(
    "command,retry",
    [
        ("python skills/report-builder/scripts/build.py --in data.csv", "ls /tmp"),
        ("cd /app && pytest -q", "cd /app && git status --short"),
        ("cd /app; python run.py", "cd /app; ls"),
        ("pytest tests/test_api.py", "cat pytest.ini"),
    ],
)
def test_unrelated_command_in_window_does_not_recover(checker, command, retry):
    calls = [
        _call(command, "Traceback (most recent call last):\nKeyError: 'x'\nexit code 1"),
        _call(retry, _codex_output(0, "skills")),
    ]
    result = checker(calls)
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["corrections"] == []
    assert [failure["failed_cmd"] for failure in result["unrecovered_failures"]] == [command]


@pytest.mark.parametrize(
    "command,retry",
    [
        ("cd /app && pytest -q", "pytest -q tests/test_api.py"),
        ("pytest -q", "cd /app && pytest -q"),
        ("python run.py", "python3 run.py --verbose"),
        ("/usr/bin/python3 run.py", "python run.py"),
        ("pytest -q", "python -m pytest -q"),
        ("bash scripts/build.sh", "scripts/build.sh --clean"),
    ],
)
def test_related_command_in_window_recovers(checker, command, retry):
    result = checker([_call(command, "exit code 1"), _call(retry, _codex_output(0, "done"))])
    assert result["score"] == 0.9
    assert result["passed"] is True
    assert len(result["corrections"]) == 1
    assert result["unrecovered_failures"] == []
