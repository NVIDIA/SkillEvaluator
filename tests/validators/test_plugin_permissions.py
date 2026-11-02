# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared permission rules: the bounded config walk and shell-tool allow-rule classification."""

from __future__ import annotations

from typing import Any

import pytest

from skillevaluator.models.result import Severity
from skillevaluator.plugin_permissions import (
    allowed_tools_flag_issues,
    bash_rule_risk,
    is_broad_allow_rule,
    iter_config_strings,
)


@pytest.mark.parametrize("name", ["notes", "help", "comment", "title", "description"])
def test_config_walk_reads_tables_under_documentation_key_names(name: str) -> None:
    """Regression: the walk skipped everything under a key such as 'notes', so a server the author named
    'notes' was never inspected."""
    config = {"mcp_servers": {name: {"command": "mcp", "args": ["serve", "--allowedTools", "Bash"]}}}
    [issue] = allowed_tools_flag_issues(config)
    assert issue.severity == Severity.HIGH
    assert issue.concept == "permission_allow_flag"


@pytest.mark.parametrize("prose", ["run with --allowedTools Bash", ["run with --allowedTools Bash"]])
def test_config_walk_skips_prose_under_documentation_keys(prose: Any) -> None:
    assert allowed_tools_flag_issues({"description": prose, "notes": prose}) == []
    assert list(iter_config_strings({"help": prose})) == []


@pytest.mark.parametrize(
    ("rule", "risk"),
    [
        ("PowerShell", "unrestricted"),
        ("PowerShell()", "unrestricted"),
        ("PowerShell(*)", "unrestricted"),
        ("PowerShell(python3:*)", "interpreter"),
        ("PowerShell(pwsh *)", "interpreter"),
        ("PowerShell(pwsh.exe -Command *)", "interpreter"),
        ("PowerShell(iex *)", "interpreter"),
        ("PowerShell(Invoke-Expression:*)", "interpreter"),
        ("PowerShell(cmd *)", "interpreter"),
        ("PowerShell(start-process *)", "interpreter"),
        ("PowerShell(npm.exe run *)", "interpreter"),
        ("PowerShell(Get-ChildItem *)", None),
        ("PowerShell(Start-Sleep *)", None),
        ("Bash(git status *)", None),
        ("Read", None),
    ],
)
def test_powershell_rules_are_classified_like_bash_rules(rule: str, risk: str | None) -> None:
    """Regression: only Bash rules were classified, so 'PowerShell' or 'PowerShell(iex *)' pre-approved any
    shell command with no finding."""
    assert bash_rule_risk(rule) == risk
    assert is_broad_allow_rule(rule) is (risk is not None)


def test_allowed_tools_flag_flags_powershell_grant() -> None:
    [issue] = allowed_tools_flag_issues(["claude", "-p", "x", "--allowedTools", "PowerShell"])
    assert issue.severity == Severity.HIGH
    assert "PowerShell" in issue.message
