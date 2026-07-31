# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the SKILL.md shell-read routing detection.

Ports SkillEvaluator 0.7.22 ``b656d5a`` ("recognize non-cat shell reads of SKILL.md in
routing checks") into the in-process Tier 3 engine (``tier3/eval_core/checks.py``).
"""

from __future__ import annotations

import pytest

from skillevaluator.tier3.eval_core.checks import (
    _cmd_reads_skill_md,
    check_routing,
    check_workflow_order,
)


@pytest.mark.parametrize(
    "cmd",
    [
        "cat SKILL.md",
        "cat skills/foo/SKILL.md",
        "head -20 SKILL.md",
        "tail -n 5 SKILL.md",
        "sed -n '1,50p' SKILL.md",
        "less skills/x/SKILL.md",
        "F=skills/x/SKILL.md; cat $F",
        "cat ./SKILL.md && echo done",
    ],
)
def test_recognizes_shell_reads_of_skill_md(cmd: str):
    assert _cmd_reads_skill_md(cmd) is True


@pytest.mark.parametrize(
    "cmd",
    [
        "grep SKILL config.json",  # search tool, not a viewer (the false positive this fixes)
        "sed -n '/SKILL/p' config.json",  # pattern match, never opens a SKILL.md
        "pgrep something",  # substring collision with 'grep '
        "cat config.json",  # reads a non-SKILL file
        "echo SKILL.md > out.txt",  # output redirect, not a read
        "rg SKILL .",
    ],
)
def test_ignores_non_reads_and_searches(cmd: str):
    assert _cmd_reads_skill_md(cmd) is False


def _exec(cmd: str) -> dict:
    return {"action": "bash", "action_input": {"command": cmd}, "observation": ""}


def test_check_routing_treats_sed_read_like_cat_read():
    """A sed read of the target SKILL.md must be credited identically to cat."""
    cat_result = check_routing(
        [_exec("cat skills/data-export/SKILL.md")],
        expected_skill="data-export",
        acceptable_skills=[],
    )
    sed_result = check_routing(
        [_exec("sed -n '1,40p' skills/data-export/SKILL.md")],
        expected_skill="data-export",
        acceptable_skills=[],
    )
    assert sed_result["passed"] == cat_result["passed"]


def test_workflow_order_flags_execution_before_read():
    calls = [_exec("python run.py")]
    result = check_workflow_order(calls, expected_skill="data-export")
    assert result["passed"] is False
    assert "before reading SKILL.md" in result["reason"]


def test_workflow_order_passes_read_then_execute():
    calls = [_exec("cat skills/data-export/SKILL.md"), _exec("python run.py")]
    result = check_workflow_order(calls, expected_skill="data-export")
    assert result["passed"] is True
