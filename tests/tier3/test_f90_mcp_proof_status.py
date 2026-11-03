# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reachability proof preserves failed, unknown, flaky and successful observations."""

from __future__ import annotations

from typing import Any

import pytest

from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals, summarize_plugin_signals
from skillevaluator.tier3.mcp_proof import NOT_REQUESTED_DETAIL, apply_in_agent_mcp_proof

HOST_OK = "initialize and tools/list succeeded over streamable-http"


def _probe(status: str, detail: str = HOST_OK, tools: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"status": status, "tools": list(tools), "detail": detail}


def _engine(by_server: dict[str, dict[str, int]], census: dict[str, Any] | None = None) -> dict[str, Any]:
    agent: dict[str, Any] = {"plugin_signals_summary": {"with_skill": {"mcp_calls": {"by_server": by_server}}}}
    if census is not None:
        agent["plugin_load_census"] = census
    return {"agents": {"claude-code": agent}}


def _census(loaded: list[str], not_loaded: dict[str, str]) -> dict[str, Any]:
    return {
        "agent": "claude-code",
        "mode": "native",
        "trials": 12,
        "harness": "claude-code system/init event",
        "loaded": [
            {"type": "mcp", "name": name, "evidence": f"claude-code system/init event: plugin:docs:{name} connected"}
            for name in loaded
        ],
        "not_loaded": [{"type": "mcp", "name": name, "reason": reason} for (name, reason) in not_loaded.items()],
    }


def _statuses(proof: dict[str, Any]) -> dict[str, str]:
    return {name: entry["status"] for (name, entry) in proof.items()}


def test_server_whose_init_failed_in_the_agent_is_not_proven() -> None:
    proof = {"deepwiki": _probe("reachable-host"), "cfdocs": _probe("reachable-host")}
    census = _census(
        ["deepwiki"], {"cfdocs": "claude-code init reported MCP server plugin:docs:cfdocs with status failed"}
    )
    final = apply_in_agent_mcp_proof(proof, _engine({"deepwiki": {"total": 10, "succeeded": 10}}, census))
    assert _statuses(final) == {"deepwiki": "used-successfully", "cfdocs": "not-loaded-in-agent"}
    assert "status failed" in final["cfdocs"]["detail"]
    assert "host probe: " + HOST_OK in final["cfdocs"]["detail"]


def test_server_the_agent_never_loaded_is_not_proven() -> None:
    proof = {
        "deepwiki": _probe("reachable-host"),
        "deepwiki-bad": _probe("reachable-host"),
        "gone": _probe("unreachable", "DNS resolution failed: gaierror"),
    }
    census = _census(
        ["deepwiki"],
        {
            "deepwiki-bad": "not reported by claude-code init (no plugin:docs:deepwiki-bad)",
            "gone": "claude-code init reported MCP server plugin:docs:gone with status failed",
        },
    )
    final = apply_in_agent_mcp_proof(proof, _engine({"deepwiki": {"total": 1, "succeeded": 1}}, census))
    assert _statuses(final) == {
        "deepwiki": "used-successfully",
        "deepwiki-bad": "not-loaded-in-agent",
        "gone": "unreachable",
    }


def test_called_but_nothing_succeeded_is_not_green_and_not_counted() -> None:
    proof = {
        "deepwiki": _probe("declared", NOT_REQUESTED_DETAIL),
        "deepwiki-bad": _probe("declared", NOT_REQUESTED_DETAIL),
        "gone": _probe("declared", NOT_REQUESTED_DETAIL),
    }
    final = apply_in_agent_mcp_proof(
        proof, _engine({"deepwiki-bad": {"total": 2, "succeeded": 0, "failed": 2, "unknown": 0}})
    )
    entry = final["deepwiki-bad"]
    assert entry["status"] == "called-no-success"
    assert entry["detail"].startswith("agent made 2 call(s): 0 succeeded, 2 failed, 0 unknown")


def test_a_failed_call_never_turns_an_unreachable_server_reachable() -> None:
    proof = {"gone": _probe("unreachable", "DNS resolution failed: gaierror")}
    final = apply_in_agent_mcp_proof(proof, _engine({"gone": {"total": 1, "succeeded": 0, "failed": 1}}))
    assert final["gone"]["status"] == "unreachable"


def test_one_successful_call_still_proves_the_server() -> None:
    proof = {"gone": _probe("unreachable", "DNS resolution failed: gaierror")}
    census = _census([], {"gone": "claude-code init reported MCP server plugin:docs:gone with status failed"})
    final = apply_in_agent_mcp_proof(proof, _engine({"gone": {"total": 2, "succeeded": 1, "failed": 1}}, census))
    assert final["gone"]["status"] == "used-successfully"


def test_failed_claude_calls_end_to_end_do_not_prove_the_server() -> None:
    trajectory = {
        "agent": {"name": "claude-code"},
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_01",
                        "function_name": "mcp__plugin_release-kit_reltools__list_changes",
                        "arguments": {"project": "atlas"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "toolu_01",
                            "content": "unknown tag 1.3.9\n\n[error] tool reported failure",
                            "extra": {"tool_result_is_error": True},
                        }
                    ]
                },
            }
        ],
    }
    signals = compute_plugin_signals(trajectory, {}, declared={"skill": [], "mcp": ["reltools"]})
    summary = summarize_plugin_signals([signals])
    engine = {"agents": {"claude-code": {"plugin_signals_summary": {"with_skill": summary}}}}
    final = apply_in_agent_mcp_proof({"reltools": _probe("declared", NOT_REQUESTED_DETAIL)}, engine)
    assert final["reltools"]["status"] == "called-no-success"


def test_applying_twice_is_stable() -> None:
    proof = {"docs": _probe("reachable-host"), "kb": _probe("declared", NOT_REQUESTED_DETAIL)}
    engine = _engine({"kb": {"total": 1, "succeeded": 0, "unknown": 1}})
    once = apply_in_agent_mcp_proof(proof, engine)
    assert apply_in_agent_mcp_proof(once, engine) == once
    assert apply_in_agent_mcp_proof(once, None) == once


@pytest.mark.parametrize(
    "loaded,failed_calls,expected",
    [(11, 0, "reachable-host"), (11, 2, "called-no-success"), (0, 2, "not-loaded-in-agent")],
)
def test_flaky_and_never_loaded_remain_distinct(loaded, failed_calls, expected):
    census = _census([], {"docs": "init reported unavailable"})
    engine = _engine({"docs": {"total": failed_calls, "succeeded": 0, "failed": failed_calls, "unknown": 0}}, census)
    engine["agents"]["claude-code"]["plugin_signals_summary"]["with_skill"]["mcp_load"] = {
        "docs": {"loaded": loaded, "trials": 12}
    }
    final = apply_in_agent_mcp_proof({"docs": _probe("reachable-host")}, engine)
    assert final["docs"]["status"] == expected
    if loaded:
        assert "not loaded in 1 of 12" in final["docs"]["detail"]
