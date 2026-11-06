# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Argument validity never supplies proof that a call executed successfully."""

from __future__ import annotations

import json
from typing import Any

import pytest

from skillevaluator.tier3.eval_core.plugin_signals import (
    compute_plugin_signals,
    summarize_plugin_signals,
    validate_plugin_case_fields,
)

DECLARED = {"skill": [], "mcp": ["reltools"]}

RULE = {"tool": "MCP:reltools/list_changes", "required": ["project", "since"], "equals": {"project": "atlas"}}


def _step(call_id: str, fn: str, args: dict[str, Any], content: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"source_call_id": call_id, "content": content}
    if extra:
        result["extra"] = extra
    return {
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": call_id, "function_name": fn, "arguments": args}],
        "observation": {"results": [result]},
    }


def _traj(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"agent": {"name": "claude-code"}, "steps": [{"source": "user", "message": "List atlas changes."}, *steps]}


def _arguments(trajectory: dict[str, Any], case: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    signals = compute_plugin_signals(trajectory, case, declared=DECLARED, **kwargs)
    assert signals is not None
    return signals["arguments"]


def test_a_call_the_harness_refused_does_not_pass_its_rules() -> None:
    # check-24/23 e05 Claude shape: the sum-of-parts arm has no server, Claude Code says "No such tool".
    refusal = "<tool_use_error>Error: No such tool available: mcp__reltools__list_changes</tool_use_error>"
    trajectory = _traj(
        _step(
            "toolu_1",
            "mcp__reltools__list_changes",
            {"project": "atlas", "since": "1.3.9"},
            refusal,
            tool_result_is_error=True,
        )
    )
    block = _arguments(trajectory, {"tool_arguments": [RULE]})
    assert (block["checked"], block["passed"]) == (1, 0)
    assert [row["rule"] for row in block["failures"]] == ["call_failed"]


def test_calls_the_server_rejected_do_not_pass_their_rules() -> None:
    # check-23 e05 Codex shape: codex.txt said "failed" for both calls (limit over 50, unknown tag).
    trajectory = _traj(
        _step(
            "call_1", "list_changes", {"project": "atlas", "since": "1.3.0", "limit": 200}, "x", harness_status="failed"
        ),
        _step("call_2", "list_changes", {"project": "atlas", "since": "1.3.9"}, "y", harness_status="failed"),
    )
    block = _arguments(
        trajectory, {"tool_arguments": [RULE]}, mcp_call_servers={"call_1": "reltools", "call_2": "reltools"}
    )
    assert (block["checked"], block["passed"]) == (2, 0)


def test_rules_whose_tool_was_never_called_count_against_the_rate() -> None:
    # check-23 e03: one trial calls list_changes but never stage_release; another calls nothing.
    rules = [
        {"tool": "MCP:reltools/list_changes", "required": ["project"]},
        {"tool": "MCP:reltools/stage_release", "required": ["version"]},
    ]
    one_called = compute_plugin_signals(
        _traj(_step("toolu_1", "mcp__reltools__list_changes", {"project": "atlas"}, '{"changes": []}')),
        {"tool_arguments": rules},
        declared=DECLARED,
    )
    none_called = compute_plugin_signals(
        _traj(),
        {"tool_arguments": [{"tool": "MCP:reltools/compute_version", "required": ["current"]}]},
        declared=DECLARED,
    )
    assert one_called is not None and none_called is not None
    assert (one_called["arguments"]["checked"], one_called["arguments"]["passed"]) == (2, 1)
    assert (none_called["arguments"]["checked"], none_called["arguments"]["passed"]) == (1, 0)
    arm = summarize_plugin_signals([one_called, none_called])["arguments"]
    assert (arm["checked"], arm["passed"]) == (3, 1)
    assert arm["pass_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_dataset_github_token_is_not_echoed_in_failure_details() -> None:
    # check-23 e06: the dataset's own expected value is shaped like a GitHub token.
    token = "ghp_" + "EXAMPLE" * 5 + "0"
    case = {"tool_arguments": [{"tool": "MCP:reltools/login", "equals": {"api_key": token}}]}
    block = _arguments(_traj(_step("toolu_1", "mcp__reltools__login", {"api_key": "x"}, "ok")), case)
    assert block["failures"] and token not in json.dumps(block)


def test_schema_error_cap_also_bounds_required() -> None:
    # check-23 e10: seven missing required properties gave seven rows; the cap is five per call.
    schema = {"type": "object", "required": [f"field_{index}" for index in range(7)]}
    case = {"tool_arguments": [{"tool": "MCP:reltools/stage", "schema": schema}]}
    block = _arguments(_traj(_step("toolu_1", "mcp__reltools__stage", {}, "ok")), case)
    assert len([row for row in block["failures"] if row["rule"] == "schema"]) == 5


def test_every_bad_rule_is_reported() -> None:
    # check-23 e04 "two bad rules": validation stopped at the first one.
    problems = validate_plugin_case_fields(
        {
            "tool_arguments": [
                {"tool": "MCP:reltools/a"},
                {"tool": "MCP:reltools/b", "required": ["x"]},
                {"tool": "MCP:reltools/c", "bogus": 1},
            ]
        }
    )
    joined = "\n".join(problems)
    assert "tool_arguments[0]" in joined and "tool_arguments[2]" in joined


def test_contains_on_an_object_matches_values_not_key_names() -> None:
    # check-23 e08: ``contains {"filters": "project"}`` passed on a key name only.
    case = {"tool_arguments": [{"tool": "MCP:reltools/search", "contains": {"filters": "atlas"}}]}
    key_only = _arguments(_traj(_step("t1", "mcp__reltools__search", {"filters": {"atlas": "x"}}, "ok")), case)
    value = _arguments(_traj(_step("t1", "mcp__reltools__search", {"filters": {"project": "atlas-2"}}, "ok")), case)
    assert (key_only["passed"], value["passed"]) == (0, 1)
