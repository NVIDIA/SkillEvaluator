# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A quoted script path must retain its recorded command exit status."""

import shlex
import shutil
import subprocess
import sys

import pytest

from skillevaluator.tier3.eval_core.declarative_checks import evaluate_declarative_checks


@pytest.mark.parametrize(
    "command",
    ["python 'scripts/my task.py'", 'python "scripts/my task.py"'],
)
@pytest.mark.parametrize("status", [0, 2])
def test_single_quoted_script_command_uses_its_recorded_status(command: str, status: int) -> None:
    (result,) = evaluate_declarative_checks(
        [{"type": "command_succeeded", "script": "scripts/my task.py"}],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": command},
                "observation": f"Process exited with code {status}",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("tail", [" | tail -1", " |& tail -1", " || true", "; echo done"])
def test_quoted_script_still_does_not_claim_a_shell_tail_status(tail: str) -> None:
    (result,) = evaluate_declarative_checks(
        [{"type": "command_succeeded", "script": "scripts/my task.py"}],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": "python 'scripts/my task.py'" + tail},
                "observation": "Process exited with code 0",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert result["passed"] is None


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="Requires POSIX bash to record actual negated shell statuses",
)
@pytest.mark.parametrize(
    ("script_operand", "relative_script"),
    [
        ("'scripts/my task.py'", "scripts/my task.py"),
        ('"scripts/my task.py"', "scripts/my task.py"),
        (r"scripts/my\ folder/task.py", "scripts/my folder/task.py"),
    ],
)
@pytest.mark.parametrize("script_status", [0, 2])
@pytest.mark.parametrize("negation", ["!", "time !", "time -p !"])
def test_negated_quoted_script_does_not_claim_the_shell_status(
    tmp_path, script_operand, relative_script, script_status, negation
):
    script = tmp_path / relative_script
    script.parent.mkdir(parents=True)
    script.write_text("import sys\nprint('script-ran')\nraise SystemExit(int(sys.argv[1]))\n", encoding="utf-8")
    command = f"{negation} {shlex.quote(sys.executable)} {script_operand} {script_status}"
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == int(script_status == 0), completed.stderr
    assert completed.stdout == "script-ran\n"
    if negation == "!":
        assert completed.stderr == ""
    else:
        assert "real" in completed.stderr
        assert "user" in completed.stderr
        assert "sys" in completed.stderr

    invoked, succeeded = evaluate_declarative_checks(
        [
            {"type": "script_invoked", "script": relative_script},
            {"type": "command_succeeded", "script": relative_script},
        ],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": command},
                "observation": f"Process exited with code {completed.returncode}",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert invoked["passed"] is True
    assert succeeded["passed"] is None
    assert "could not be attributed" in succeeded["reason"]


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="Requires POSIX bash to record an actual stderr pipeline status",
)
@pytest.mark.parametrize(
    ("script_operand", "relative_script"),
    [("scripts/run.py", "scripts/run.py"), ("'scripts/my task.py'", "scripts/my task.py")],
)
def test_stderr_pipeline_does_not_claim_the_last_process_status(tmp_path, script_operand, relative_script):
    script = tmp_path / relative_script
    script.parent.mkdir(parents=True)
    script.write_text("import sys\nprint('script-ran', file=sys.stderr)\nraise SystemExit(2)\n", encoding="utf-8")
    command = f"{shlex.quote(sys.executable)} {script_operand} |& tail -1"
    completed = subprocess.run(
        ["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "script-ran\n"
    assert completed.stderr == ""

    invoked, succeeded = evaluate_declarative_checks(
        [
            {"type": "script_invoked", "script": relative_script},
            {"type": "command_succeeded", "script": relative_script},
        ],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": command},
                "observation": f"Process exited with code {completed.returncode}",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert invoked["passed"] is True
    assert succeeded["passed"] is None
    assert "could not be attributed" in succeeded["reason"]


@pytest.mark.parametrize("status", [0, 2])
def test_literal_exclamation_argument_does_not_hide_script_status(status):
    (result,) = evaluate_declarative_checks(
        [{"type": "command_succeeded", "script": "scripts/my task.py"}],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": "python 'scripts/my task.py' '!'"},
                "observation": f"Process exited with code {status}",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert result["passed"] is (status == 0)
