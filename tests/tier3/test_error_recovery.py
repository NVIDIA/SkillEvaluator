# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for check_error_recovery unrecovered-failure scoring (issue #149).

An observed command failure that is never followed by a successful similar
command must NOT be scored as first-attempt clean.
"""

from __future__ import annotations

from skillevaluator.tier3.eval_core.checks import check_error_recovery


def _call(cmd: str, observation: str) -> dict:
    return {"action": "bash", "action_input": {"command": cmd}, "observation": observation}


def test_failure_with_no_retry_is_not_first_attempt_clean():
    """A failure that is never retried must not score 1.0 / passed-clean."""
    result = check_error_recovery(
        [_call("pytest tests/test_api.py -x", "ERROR: test_create_user failed\nexit code 1")]
    )
    assert result["first_attempt_clean"] is False
    assert result["score"] == 0.0
    assert len(result["unrecovered_failures"]) == 1
    assert result["unrecovered_failures"][0]["failed_cmd"] == "pytest tests/test_api.py -x"
    assert result["unrecovered_failures"][0]["recovered"] is False
    assert "1 unrecovered failure" in result["reason"]
    assert "pytest tests/test_api.py -x" in result["reason"]


def test_failed_retries_are_not_first_attempt_clean():
    """Repeated failures of a similar command are all recorded as unrecovered."""
    result = check_error_recovery(
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


def test_recovered_failure_behavior_unchanged():
    """A failure followed by a successful similar command keeps the old scoring."""
    result = check_error_recovery(
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


def test_unrecovered_failure_after_earlier_success():
    """An unrecovered failure later in the log still breaks first_attempt_clean."""
    result = check_error_recovery(
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
