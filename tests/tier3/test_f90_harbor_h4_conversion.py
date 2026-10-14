# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H4: the pinned Harbor must keep the raw agent data the plugin checks read.

Harbor converts each trial's raw agent log into ``agent/trajectory.json`` after the
run (``populate_context_post_run``). The collector and the plugin signals only see
that trajectory. Harbor 0.13.2 lost three things there:

* Codex ``token_count`` events, so no agent step carried per-step tokens and the
  measured first-turn context cost was always unavailable (check 25).
* Claude Code ``<session>/subagents/*.jsonl`` transcripts, so every subagent step
  and its MCP calls vanished (checks 18, 19, 22, 24, 28).
* The grouping of parallel Codex calls from one model request, so two parallel
  calls became two steps and check 18's same-step guard was lost.

Each test writes a small synthetic raw log shaped like the saved live ones, runs
Harbor's own post-run conversion through SkillEvaluator's Harbor agent classes,
and reads the result the way the collector and plugin signals do. No model,
network, Docker or sandbox is used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext

from skillevaluator.tier3.eval_core import plugin_signals as ps
from skillevaluator.tier3.eval_core.atif_helpers import extract_tool_calls_as_dicts
from skillevaluator.tier3.eval_core.log_converters import load_trajectory_with_fallback
from skillevaluator.tier3.harbor.collector import _trial_usage
from skillevaluator.tier3.harbor.local_agents import (
    SkillEvaluatorGatewayCodex,
    SkillEvaluatorLocalClaudeCode,
    SkillEvaluatorLocalCodex,
)
from skillevaluator.tier3.harbor.native_agents import NativeClaudeCode, NativeCodex

CODEX_CLASSES = [Codex, NativeCodex, SkillEvaluatorGatewayCodex, SkillEvaluatorLocalCodex]
CLAUDE_CLASSES = [ClaudeCode, NativeClaudeCode, SkillEvaluatorLocalClaudeCode]

_SESSION_ID = "0c0ffee0-0000-4000-8000-000000000004"
_ROLLOUT_NAME = f"rollout-2026-10-04T10-08-41-{_SESSION_ID}.jsonl"


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _convert(agent_cls: type, logs_dir: Path, model_name: str) -> dict[str, Any]:
    """Run Harbor's post-run conversion and return the trajectory it wrote."""
    agent = agent_cls(logs_dir=logs_dir, model_name=model_name)
    agent.logger = logging.getLogger("test-f90-harbor")
    context = AgentContext()
    agent.populate_context_post_run(context)
    trajectory_path = logs_dir / "trajectory.json"
    assert trajectory_path.is_file(), f"{agent_cls.__name__} wrote no trajectory"
    return json.loads(trajectory_path.read_text(encoding="utf-8"))


# --- Codex rollouts -------------------------------------------------------------------------


def _ts(second: int) -> str:
    return f"2026-10-04T10:08:{second:02d}.000Z"


def _token_count(second: int, total: tuple[int, int, int], last: tuple[int, int, int] | None) -> dict[str, Any]:
    if last is None:
        info = None
    else:
        info = {
            "total_token_usage": {
                "input_tokens": total[0],
                "cached_input_tokens": total[1],
                "output_tokens": total[2],
                "reasoning_output_tokens": 0,
                "total_tokens": total[0] + total[2],
            },
            "last_token_usage": {
                "input_tokens": last[0],
                "cached_input_tokens": last[1],
                "output_tokens": last[2],
                "reasoning_output_tokens": 0,
                "total_tokens": last[0] + last[2],
            },
            "model_context_window": 258400,
        }
    return {"timestamp": _ts(second), "type": "event_msg", "payload": {"type": "token_count", "info": info}}


def _codex_preamble(prompt: str) -> list[dict[str, Any]]:
    """Session start shaped like a Codex 0.124 ``codex exec`` rollout."""
    turn_id = "0c0ffee0-0000-4000-8000-0000000000t1"
    return [
        {
            "timestamp": _ts(0),
            "type": "session_meta",
            "payload": {
                "id": _SESSION_ID,
                "timestamp": _ts(0),
                "cwd": "/workspace",
                "originator": "codex_exec",
                "cli_version": "0.124.0",
                "source": "exec",
                "model_provider": "openai",
            },
        },
        {
            "timestamp": _ts(0),
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": turn_id, "model_context_window": 258400},
        },
        {
            "timestamp": _ts(0),
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "<permissions instructions>sandbox</permissions>"}],
            },
        },
        {
            "timestamp": _ts(0),
            "type": "turn_context",
            "payload": {"turn_id": turn_id, "cwd": "/workspace", "model": "gpt-test", "approval_policy": "never"},
        },
        {
            "timestamp": _ts(0),
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]},
        },
        {"timestamp": _ts(0), "type": "event_msg", "payload": {"type": "user_message", "message": prompt}},
        _token_count(0, (0, 0, 0), None),
    ]


def _assistant(second: int, text: str) -> list[dict[str, Any]]:
    return [
        {"timestamp": _ts(second), "type": "event_msg", "payload": {"type": "agent_message", "message": text}},
        {
            "timestamp": _ts(second),
            "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
        },
    ]


def _function_call(second: int, call_id: str, name: str, arguments: dict[str, Any], namespace: str | None = None):
    payload: dict[str, Any] = {
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(arguments),
        "call_id": call_id,
    }
    if namespace:
        payload["namespace"] = namespace
    return {"timestamp": _ts(second), "type": "response_item", "payload": payload}


def _function_output(second: int, call_id: str, output: str) -> dict[str, Any]:
    return {
        "timestamp": _ts(second),
        "type": "response_item",
        "payload": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def _write_rollout(logs_dir: Path, events: list[dict[str, Any]]) -> None:
    _write_jsonl(logs_dir / "sessions" / "2026" / "10" / "04" / _ROLLOUT_NAME, events)


def _codex_rollout_with_token_counts() -> list[dict[str, Any]]:
    """Shaped like the saved ``cx-with-ct-visits-005`` rollout: three model requests, each closed by ``token_count``."""
    events = _codex_preamble("Convert the clinic visits CSV (input/visits.csv) to JSON with our conventions.\n")
    events += _assistant(1, "I'll inspect the CSV and the local conversion rules first.")
    events.append(_function_call(1, "call_find", "exec_command", {"cmd": "find /workspace -name '*.csv'"}))
    events.append(_token_count(2, (13230, 12928, 96), (13230, 12928, 96)))
    events.append(_function_output(2, "call_find", "Process exited with code 0\nOutput:\n/workspace/input/visits.csv"))
    events.append(_token_count(2, (13230, 12928, 96), (13230, 12928, 96)))
    events += _assistant(3, "I've got the rules and data; now I'm writing the JSON output.")
    events.append(_function_call(3, "call_write", "exec_command", {"cmd": "mkdir -p /workspace/out"}))
    events.append(_token_count(4, (26616, 25856, 204), (13386, 12928, 108)))
    events.append(_function_output(4, "call_write", "Process exited with code 0\nOutput:\n"))
    events.append(_token_count(4, (26616, 25856, 204), (13386, 12928, 108)))
    events += _assistant(5, "Done: `out/visits.json` - 2 rows.")
    events.append(_token_count(6, (40207, 39296, 309), (13591, 13440, 105)))
    events.append(
        {
            "timestamp": _ts(6),
            "type": "event_msg",
            "payload": {"type": "task_complete", "last_agent_message": "Done: `out/visits.json` - 2 rows."},
        }
    )
    return events


def _write_job_trial(job_dir: Path, trajectory: dict[str, Any]) -> dict[str, Any]:
    trial = job_dir / "case-001__AbCdEfG"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    return {"_trial_root_name": trial.name}


@pytest.mark.parametrize("agent_cls", CODEX_CLASSES, ids=lambda cls: cls.__name__)
def test_codex_token_counts_reach_agent_steps_and_first_turn_tokens(tmp_path: Path, agent_cls: type) -> None:
    logs_dir = tmp_path / "job" / "case-001__AbCdEfG" / "agent"
    _write_rollout(logs_dir, _codex_rollout_with_token_counts())

    trajectory = _convert(agent_cls, logs_dir, "openai/gpt-test")

    agent_steps = [step for step in trajectory["steps"] if step.get("source") == "agent"]
    assert agent_steps
    # Every model request that produced output carries its own token counts.
    with_metrics = [step for step in agent_steps if (step.get("metrics") or {}).get("prompt_tokens")]
    assert len(with_metrics) == 3, [step.get("metrics") for step in agent_steps]
    assert with_metrics[0]["metrics"]["prompt_tokens"] == 13230
    assert with_metrics[0]["metrics"]["cached_tokens"] == 12928

    # The collector reads the first model turn's prompt tokens from the saved trajectory (check 25).
    usage = _trial_usage(tmp_path / "job", {"_trial_root_name": "case-001__AbCdEfG"})
    assert usage.get("first_turn_prompt_tokens") == 13230.0
    assert usage.get("prompt_tokens") == 40207.0


# --- Parallel Codex calls ------------------------------------------------------------------

_RELEASE_TAG = "1.5.0+rk.23b59e"


def _codex_rollout_with_parallel_calls() -> list[dict[str, Any]]:
    """One model request issues ``compute_version`` and ``stage_release`` in parallel (the ``par_grade`` spot check)."""
    events = _codex_preamble("Compute the next atlas version from 1.4.2 and stage the notes.\n")
    events += _assistant(1, "Computing the version and staging the notes.")
    events.append(
        _function_call(1, "call_compute", "compute_version", {"current": "1.4.2", "bump": "minor"}, "mcp__reltools__")
    )
    events.append(
        _function_call(
            1,
            "call_stage",
            "stage_release",
            {"project": "atlas", "version": _RELEASE_TAG, "notes_path": "RELEASE_NOTES.md"},
            "mcp__reltools__",
        )
    )
    events.append(_token_count(2, (13230, 12928, 96), (13230, 12928, 96)))
    events.append(
        _function_output(
            2, "call_compute", json.dumps([{"type": "text", "text": f'{{"build_tag": "{_RELEASE_TAG}"}}'}])
        )
    )
    events.append(_function_output(2, "call_stage", json.dumps([{"type": "text", "text": '{"receipt": "STG-1"}'}])))
    events += _assistant(3, f"Staged atlas {_RELEASE_TAG}.")
    events.append(_token_count(4, (26616, 25856, 204), (13386, 12928, 108)))
    return events


def _plugin_context(entry: dict[str, Any]) -> ps.PluginSignalsContext:
    return ps.build_plugin_signals_context(
        member_skills=["changelog-collect", "release-notes", "version-bump"],
        mcp_servers=["reltools"],
        entries=[entry],
        subagents=["release-reviewer"],
        commands=["release-check"],
    )


@pytest.mark.parametrize("agent_cls", CODEX_CLASSES, ids=lambda cls: cls.__name__)
def test_parallel_codex_calls_share_one_step_so_handoff_needs_a_later_step(tmp_path: Path, agent_cls: type) -> None:
    logs_dir = tmp_path / "agent"
    _write_rollout(logs_dir, _codex_rollout_with_parallel_calls())

    trajectory = _convert(agent_cls, logs_dir, "openai/gpt-test")

    call_steps = [
        [call["function_name"] for call in step.get("tool_calls") or []]
        for step in trajectory["steps"]
        if step.get("source") == "agent" and step.get("tool_calls")
    ]
    assert call_steps == [["compute_version", "stage_release"]]

    entry = {
        "id": "par",
        "prompt": "Compute the next atlas version from 1.4.2 and stage the notes.",
        "expected_output": "x",
        "handoffs": [
            {
                "producer": "MCP:reltools/compute_version",
                "consumer": "MCP:reltools/stage_release",
                "value": _RELEASE_TAG,
            }
        ],
    }
    context = _plugin_context(entry)
    servers = {"call_compute": "reltools", "call_stage": "reltools"}
    handoff = ps.compute_plugin_signals(
        trajectory,
        context.case_spec("par"),
        declared=context.declared_for(ps.ARM_WITH_SKILL),
        mcp_call_servers=servers,
        subagent_aliases=context.aliases_for(ps.ARM_WITH_SKILL),
    )["handoff"]
    # The consumer was issued with the producer, so it cannot have used the producer's output.
    assert (handoff["passed"], handoff["checked"]) == (0, 1)
    assert handoff["failures"][0]["detail"] == "value did not reach consumer input after the producer produced it"


# --- Claude Code sessions with subagents ---------------------------------------------------


def _claude_event(
    second: int,
    kind: str,
    uuid: str,
    parent: str | None,
    content: Any,
    *,
    message_id: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": kind, "content": content}
    if kind == "assistant":
        message.update(
            {
                "id": message_id,
                "model": "claude-test",
                "type": "message",
                "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100},
            }
        )
    event: dict[str, Any] = {
        "type": kind,
        "uuid": uuid,
        "parentUuid": parent,
        "isSidechain": agent_id is not None,
        "sessionId": _SESSION_ID,
        "timestamp": f"2026-10-04T10:09:{second:02d}.000Z",
        "cwd": "/workspace",
        "version": "2.1.0",
        "userType": "external",
        "message": message,
    }
    if agent_id:
        event["agentId"] = agent_id
    return event


def _tool_use(call_id: str, name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "id": call_id, "name": name, "input": arguments}]


def _tool_result(call_id: str, text: str) -> list[dict[str, Any]]:
    return [{"type": "tool_result", "tool_use_id": call_id, "content": text}]


def _write_claude_session(logs_dir: Path) -> None:
    """Main chain hands the review to a plugin subagent; the subagent's transcript is in ``subagents/``.

    Shaped like the saved ``rk-full-release-001`` session: Claude Code 2.x writes the subagent's own events,
    marked ``isSidechain`` with an ``agentId``, to ``<session>/subagents/agent-<id>.jsonl``.
    """
    project = logs_dir / "sessions" / "projects" / "-workspace"
    agent_id = "a0e77a0c7c00c39fc"
    main = [
        _claude_event(0, "user", "u-001", None, "Have the release reviewer check the draft and handle the release."),
        _claude_event(
            1,
            "assistant",
            "a-001",
            "u-001",
            _tool_use(
                "toolu_agent_1",
                "Agent",
                {
                    "description": "Review the draft",
                    "prompt": "Review RELEASE_NOTES.md for atlas.",
                    "subagent_type": "release-kit:release-reviewer",
                },
            ),
            message_id="msg_main_1",
        ),
        _claude_event(9, "user", "u-002", "a-001", _tool_result("toolu_agent_1", "Review done; release published.")),
        _claude_event(
            10, "assistant", "a-002", "u-002", [{"type": "text", "text": "The reviewer is done."}], message_id="msg_2"
        ),
    ]
    sidechain = [
        _claude_event(2, "user", "s-001", None, "Review RELEASE_NOTES.md for atlas.", agent_id=agent_id),
        _claude_event(
            3,
            "assistant",
            "s-002",
            "s-001",
            _tool_use("toolu_sub_read", "Read", {"file_path": "/workspace/RELEASE_NOTES.md"}),
            message_id="msg_sub_1",
            agent_id=agent_id,
        ),
        _claude_event(4, "user", "s-003", "s-002", _tool_result("toolu_sub_read", "# atlas 1.5.0"), agent_id=agent_id),
        _claude_event(
            5,
            "assistant",
            "s-004",
            "s-003",
            _tool_use("toolu_sub_publish", "mcp__reltools__publish_release", {"project": "atlas", "version": "1.5.0"}),
            message_id="msg_sub_2",
            agent_id=agent_id,
        ),
        _claude_event(
            6, "user", "s-005", "s-004", _tool_result("toolu_sub_publish", '{"published": true}'), agent_id=agent_id
        ),
        _claude_event(
            7,
            "assistant",
            "s-006",
            "s-005",
            [{"type": "text", "text": "Reviewed and published."}],
            message_id="msg_sub_3",
            agent_id=agent_id,
        ),
    ]
    _write_jsonl(project / f"{_SESSION_ID}.jsonl", main)
    _write_jsonl(project / _SESSION_ID / "subagents" / f"agent-{agent_id}.jsonl", sidechain)


@pytest.mark.parametrize("agent_cls", CLAUDE_CLASSES, ids=lambda cls: cls.__name__)
def test_claude_subagent_steps_and_mcp_calls_reach_the_trajectory(tmp_path: Path, agent_cls: type) -> None:
    logs_dir = tmp_path / "agent"
    _write_claude_session(logs_dir)

    trajectory = _convert(agent_cls, logs_dir, "anthropic/claude-test")

    sidechain = [step for step in trajectory["steps"] if (step.get("extra") or {}).get("is_sidechain")]
    assert sidechain, "the subagent's steps were dropped"
    sidechain_calls = [call["function_name"] for step in sidechain for call in step.get("tool_calls") or []]
    assert "Read" in sidechain_calls
    assert "mcp__reltools__publish_release" in sidechain_calls

    # The verifier reads the same trajectory (Harbor uploads it back to non-mounted environments).
    verifier_trajectory, meta = load_trajectory_with_fallback(logs_dir / "trajectory.json")
    assert meta["source"] == "trajectory.json"
    verifier_calls = [call["action"] for call in extract_tool_calls_as_dicts(verifier_trajectory or {})]
    assert "mcp__reltools__publish_release" in verifier_calls

    # Check 19: the forbidden MCP call happened inside the subagent, so the conflict probe fails.
    entry = {
        "id": "e07",
        "prompt": "Have the release reviewer check the draft and handle the release.",
        "expected_output": "replay",
        "conflict_probes": [
            {
                "id": "reviewer-never-publishes",
                "must_use": "Agent:release-reviewer",
                "must_not_use": "MCP:reltools/publish_release",
            }
        ],
    }
    context = _plugin_context(entry)
    signals = ps.compute_plugin_signals(
        trajectory,
        context.case_spec("e07"),
        declared=context.declared_for(ps.ARM_WITH_SKILL),
        subagent_aliases=context.aliases_for(ps.ARM_WITH_SKILL),
    )
    conflict = signals["conflict"]
    assert (conflict["passed"], conflict["checked"]) == (0, 1)
    assert signals["mcp_calls"]["total"] == 1
