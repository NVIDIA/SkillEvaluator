# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A quoted script path must retain its recorded command exit status."""

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


@pytest.mark.parametrize("tail", [" | tail -1", " || true", "; echo done"])
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
