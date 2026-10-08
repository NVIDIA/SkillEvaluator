# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 24 (execution success): real Claude Code, Codex, and OpenCode failure shapes.

Fixtures copy the shapes of the saved proof examples (check-24 e01-e04, the
check-22 codex.txt mapping case) with made-up values. Every MCP outcome is read
the way the collector reads it on the host.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context, compute_plugin_signals
from skillevaluator.tier3.harbor.collector import collect_harbor_results

DECLARED = {"skill": [], "mcp": ["reltools"]}
CASE_ID = "case-1"
REWARD = {
    "security": 0.9,
    "skill_execution": 0.8,
    "skill_efficiency": 0.7,
    "accuracy": 0.6,
    "goal_accuracy": 0.5,
    "behavior_check": 0.4,
}
WALL = "Wall time: 0.0029 seconds\nOutput:\n"


def _counts(block: dict[str, Any]) -> tuple[int, int, int, int]:
    return block["total"], block["succeeded"], block["failed"], block["unknown"]


def _claude_call(call_id: str, fn: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": call_id, "function_name": fn, "arguments": args}],
        "observation": {"results": [{"source_call_id": call_id, **result}]},
    }


def _traj(*steps: dict[str, Any], agent: str = "claude-code") -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": agent, "version": "1.0.0"},
        "steps": [{"source": "user", "message": "List what changed in atlas since 1.3.9."}, *steps],
    }


def _signals(trajectory: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    signals = compute_plugin_signals(trajectory, {}, declared=DECLARED, **kwargs)
    assert signals is not None
    return signals


# --------------------------------------------------------------------------- #
# Claude Code: Harbor keeps the flag under extra.tool_result_*                 #
# --------------------------------------------------------------------------- #
CLAUDE_NATIVE = "mcp__plugin_release-kit_reltools__list_changes"
TAG_ERROR = "Error: unknown tag '1.3.9' for atlas - known tags: 1.3.0, 1.4.2"


def _claude_error_result(text: str, **extra: Any) -> dict[str, Any]:
    return {"content": text, "extra": extra}


def test_claude_harbor_error_flags_mark_the_call_failed() -> None:
    # check-24 e01: the live trial shape, both Harbor keys and the appended line.
    raw = {"type": "tool_result", "content": TAG_ERROR, "is_error": True, "tool_use_id": "toolu_01"}
    e01 = _claude_error_result(
        f"{TAG_ERROR}\n\n[error] tool reported failure",
        tool_result_metadata={"is_error": True, "raw_tool_result": raw},
        tool_result_is_error=True,
    )
    signals = _signals(_traj(_claude_call("toolu_01", CLAUDE_NATIVE, {"project": "atlas", "since": "1.3.9"}, e01)))

    assert _counts(signals["mcp_calls"]) == (1, 0, 1, 0)
    assert signals["mcp_calls"]["success_rate"] == 0.0
    assert signals["activation_coverage"]["unavailable"] == ["mcp:reltools"]


@pytest.mark.parametrize(
    "extra",
    [
        {"tool_result_is_error": True},
        {"tool_result_metadata": {"is_error": True}},
        {"tool_result_metadata": {"raw_tool_result": {"type": "tool_result", "is_error": True}}},
    ],
    ids=["tool_result_is_error", "metadata.is_error", "raw_tool_result.is_error"],
)
def test_each_claude_flag_key_alone_marks_failure(extra: dict[str, Any]) -> None:
    # A plain body with no failure words: only the flag can say the call failed.
    result = _claude_error_result("unknown tag 1.3.9 for atlas", **extra)
    signals = _signals(_traj(_claude_call("toolu_02", CLAUDE_NATIVE, {}, result)))
    assert _counts(signals["mcp_calls"]) == (1, 0, 1, 0)


def test_claude_orphan_result_flag_on_the_step_extra_marks_failure() -> None:
    # Harbor's orphan tool_result path puts the flag on the single-call step's extra.
    step = {
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": "toolu_03", "function_name": "mcp__reltools__list_changes", "arguments": {}}],
        "observation": {"results": [{"source_call_id": "toolu_03", "content": "unknown tag 1.3.9"}]},
        "extra": {"tool_result_is_error": True, "metadata": {"is_error": True}},
    }
    assert _counts(_signals(_traj(step))["mcp_calls"]) == (1, 0, 1, 0)


@pytest.mark.parametrize("flag", ["true", "True", " TRUE ", "1", "yes", 1])
def test_string_and_numeric_error_flags_mark_failure(flag: Any) -> None:
    # check-24 e04 (L28): is_error spelled as a string must not read as success.
    result = {"content": "unknown tag 1.3.9 for atlas", "is_error": flag}
    claude = _signals(_traj(_claude_call("toolu_04", "mcp__reltools__list_changes", {}, result)))
    codex = _signals(
        _traj(_claude_call("call_04", "list_changes", {}, {**result, "content": WALL + "unknown tag"}), agent="codex"),
        mcp_call_servers={"call_04": "reltools"},
    )
    assert _counts(claude["mcp_calls"]) == (1, 0, 1, 0)
    assert _counts(codex["mcp_calls"]) == (1, 0, 1, 0)


@pytest.mark.parametrize("flag", [False, "false", 0, None])
def test_false_or_missing_flags_keep_a_good_result_succeeded(flag: Any) -> None:
    result = {"content": '{"changes": []}', "is_error": flag}
    assert _counts(_signals(_traj(_claude_call("toolu_05", "mcp__reltools__x", {}, result)))["mcp_calls"]) == (
        1,
        1,
        0,
        0,
    )


# --------------------------------------------------------------------------- #
# Codex: the failed status lives only in codex.txt                            #
# --------------------------------------------------------------------------- #
def _codex_step(call_id: str, tool: str, args: dict[str, Any], text: str) -> dict[str, Any]:
    body = json.dumps([{"type": "text", "text": text}], separators=(",", ":"))
    return _claude_call(call_id, tool, args, {"content": WALL + body})


def _session_line(call_id: str, server: str, tool: str, args: dict[str, Any]) -> str:
    payload = {
        "type": "function_call",
        "name": tool,
        "namespace": f"mcp__{server}__",
        "arguments": json.dumps(args, separators=(",", ":")),
        "call_id": call_id,
    }
    return json.dumps({"timestamp": "2026-10-04T10:13:38.842Z", "type": "response_item", "payload": payload})


def _codex_items(item_id: str, server: str, tool: str, args: dict[str, Any], text: str, status: str) -> list[str]:
    started = {"id": item_id, "type": "mcp_tool_call", "server": server, "tool": tool, "arguments": args}
    completed = {
        **started,
        "result": {"content": [{"type": "text", "text": text}], "structured_content": None},
        "error": None,
        "status": status,
    }
    return [
        json.dumps(
            {"type": "item.started", "item": {**started, "result": None, "error": None, "status": "in_progress"}}
        ),
        json.dumps({"type": "item.completed", "item": completed}),
    ]


def _write_codex_trial(
    job_dir: Path,
    trial: str,
    trajectory: dict[str, Any],
    *,
    session: list[str] | None,
    codex_txt: list[str] | None,
) -> None:
    trial_dir = job_dir / trial
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "verifier" / "reward.json").write_text(json.dumps(REWARD), encoding="utf-8")
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir()
    (agent_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    if session is not None:
        rollout = agent_dir / "sessions" / "2026" / "10" / "04"
        rollout.mkdir(parents=True)
        (rollout / "rollout-1.jsonl").write_text("\n".join(session) + "\n", encoding="utf-8")
    if codex_txt is not None:
        (agent_dir / "codex.txt").write_text("\n".join(codex_txt) + "\n", encoding="utf-8")


def _write_job(jobs_dir: Path, agent: str, variant: str, trials: dict[str, dict[str, Any]]) -> None:
    job_dir = jobs_dir / f"demo-{agent}-{variant}"
    for trial, spec in trials.items():
        _write_codex_trial(job_dir, trial, **spec)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(trials),
                "stats": {
                    "n_trials": len(trials),
                    "n_errors": 0,
                    "evals": {
                        "agent__model___harbor-tasks": {
                            "n_trials": len(trials),
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.65": list(trials)}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _collect_with_arm(tmp_path: Path, trials: dict[str, dict[str, Any]], *, servers: list[str]) -> dict[str, Any]:
    jobs_dir = tmp_path / "jobs"
    _write_job(jobs_dir, "codex", "with", trials)
    baseline = {name: {**spec, "session": None, "codex_txt": None} for name, spec in trials.items()}
    _write_job(jobs_dir, "codex", "without", baseline)
    context = build_plugin_signals_context(
        member_skills=[],
        mcp_servers=servers,
        wrapper_skills=["demo"],
        entries=[{"id": CASE_ID, "prompt": "p"}],
    )
    results = collect_harbor_results(
        skill_name="demo",
        agents=["codex"],
        output_dir=tmp_path / "out",
        jobs_dir=jobs_dir,
        expected_cases=1,
        expected_case_ids=[CASE_ID],
        expected_trials=len(trials),
        plugin_signals=context,
    )
    return results["agents"]["codex"]["plugin_signals_summary"]["with_skill"]


def test_codex_failed_status_in_codex_txt_marks_calls_failed(tmp_path: Path) -> None:
    # check-24 e02: three list_changes calls, codex.txt says failed, failed, completed;
    # a second trial has one failed call. Truth: 4 calls, 1 succeeded, 3 failed.
    calls = [
        ("call_a", {"project": "atlas", "since": "1.3.9", "limit": 100}, TAG_ERROR, "failed"),
        ("call_b", {"project": "atlas", "since": "1.3.0", "limit": 100}, "Error: limit must be 1 to 50", "failed"),
        ("call_c", {"project": "atlas", "since": "1.3.0", "limit": 50}, '{"changes": []}', "completed"),
    ]
    trial_1 = {
        "trajectory": _traj(*(_codex_step(cid, "list_changes", a, t) for cid, a, t, _ in calls), agent="codex"),
        "session": [_session_line(cid, "reltools", "list_changes", a) for cid, a, _, _ in calls],
        "codex_txt": [
            line
            for index, (_, a, t, status) in enumerate(calls)
            for line in _codex_items(f"item_{index}", "reltools", "list_changes", a, t, status)
        ],
    }
    one = {"project": "atlas", "since": "1.3.9"}
    trial_2 = {
        "trajectory": _traj(_codex_step("call_d", "list_changes", one, TAG_ERROR), agent="codex"),
        "session": [_session_line("call_d", "reltools", "list_changes", one)],
        "codex_txt": _codex_items("item_1", "reltools", "list_changes", one, TAG_ERROR, "failed"),
    }

    summary = _collect_with_arm(
        tmp_path,
        {f"{CASE_ID}__AbCd001": trial_1, f"{CASE_ID}__AbCd002": trial_2},
        servers=["reltools"],
    )

    assert _counts(summary["mcp_calls"]) == (4, 1, 3, 0)
    assert summary["mcp_calls"]["success_rate"] == 0.25
    assert summary["mcp_calls"]["by_server"]["reltools"]["failed"] == 3


def test_codex_status_is_read_even_when_the_body_has_no_error_words(tmp_path: Path) -> None:
    args = {"query": "release"}
    trial = {
        "trajectory": _traj(_codex_step("call_1", "search", args, "Repository not found"), agent="codex"),
        "session": [_session_line("call_1", "reltools", "search", args)],
        "codex_txt": _codex_items("item_1", "reltools", "search", args, "Repository not found", "failed"),
    }
    summary = _collect_with_arm(tmp_path, {f"{CASE_ID}__AbCd001": trial}, servers=["reltools"])
    assert _counts(summary["mcp_calls"]) == (1, 0, 1, 0)
    assert summary["activation_coverage"]["unavailable"] == ["mcp:reltools"]


def test_codex_txt_fallback_pairs_calls_by_arguments_not_name_and_order(tmp_path: Path) -> None:
    # check-22 e02 (L26): two servers expose ``search`` and codex.txt has an extra
    # item that no trajectory call made. Name-and-order pairing swapped the servers.
    docs_args, kb_args = {"q": "install"}, {"q": "pricing"}
    codex_txt = [
        *_codex_items("item_0", "kb", "search", {"q": "warm-up"}, "ok", "completed"),
        *_codex_items("item_1", "docs", "search", docs_args, "Install with pip.", "completed"),
        *_codex_items("item_2", "kb", "search", kb_args, "no such plan", "failed"),
    ]
    trial = {
        "trajectory": _traj(
            _codex_step("call_1", "search", docs_args, "Install with pip."),
            _codex_step("call_2", "search", kb_args, "no such plan"),
            agent="codex",
        ),
        "session": None,
        "codex_txt": codex_txt,
    }

    summary = _collect_with_arm(tmp_path, {f"{CASE_ID}__AbCd001": trial}, servers=["docs", "kb"])

    by_server = summary["mcp_calls"]["by_server"]
    assert _counts(by_server["docs"]) == (1, 1, 0, 0)
    assert _counts(by_server["kb"]) == (1, 0, 1, 0)
    assert by_server["docs"]["tools"] == ["mcp__docs__search"]


def test_codex_txt_fallback_uses_the_answer_when_arguments_tie(tmp_path: Path) -> None:
    # check-22 e02 trial 3 (L26): codex.txt holds a kb ``search`` with no trajectory call, then
    # the docs ``search`` the agent made, both with the same arguments. The docs answer decides.
    args = {"q": "rate limit"}
    codex_txt = [
        *_codex_items("item_1", "kb", "search", args, "kb hit (no trajectory call)", "completed"),
        *_codex_items("item_3", "docs", "search", args, "docs hit", "completed"),
    ]
    trial = {
        "trajectory": _traj(_codex_step("call_1", "search", args, "docs hit"), agent="codex"),
        "session": None,
        "codex_txt": codex_txt,
    }

    summary = _collect_with_arm(tmp_path, {f"{CASE_ID}__AbCd001": trial}, servers=["docs", "kb"])

    assert list(summary["mcp_calls"]["by_server"]) == ["docs"]


def test_harness_statuses_ignore_malformed_result_ids() -> None:
    # Trajectory content is untrusted: an unhashable source_call_id must not stop collection.
    from skillevaluator.tier3.harbor.collector import _with_harness_statuses

    trajectory = {"steps": [{"observation": {"results": [{"source_call_id": ["x"]}, {"source_call_id": "c1"}]}}]}
    marked = _with_harness_statuses(trajectory, {"c1": "failed"})
    assert marked is not None
    results = marked["steps"][0]["observation"]["results"]
    assert results[0] == {"source_call_id": ["x"]}
    assert results[1]["extra"]["harness_status"] == "failed"


def test_harness_errors_add_a_result_only_for_a_call_without_one() -> None:
    from skillevaluator.tier3.harbor.collector import _with_harness_statuses

    trajectory = {
        "steps": [
            {"tool_calls": [{"tool_call_id": ["x"]}, {"tool_call_id": "c1"}, {"tool_call_id": "c2"}]},
            {"observation": {"results": [{"source_call_id": "c2", "content": "partial"}]}},
        ]
    }
    marked = _with_harness_statuses(trajectory, {"c1": "error", "c2": "error"}, {"c1": "boom", "c2": "boom"})
    assert marked is not None
    assert marked["steps"][0]["observation"]["results"] == [
        {"source_call_id": "c1", "content": "boom", "extra": {"harness_status": "error"}}
    ]
    assert marked["steps"][1]["observation"]["results"] == [
        {"source_call_id": "c2", "content": "partial", "extra": {"harness_status": "error"}}
    ]


# --------------------------------------------------------------------------- #
# OpenCode: Harbor drops a failed call's result; opencode.txt keeps it        #
# --------------------------------------------------------------------------- #
UNKNOWN_AGENT = "Error: Unknown agent type: reviewer is not a valid agent type"


def _opencode_turn(call_id: str, tool: str, args: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    part = {"type": "tool", "tool": tool, "callID": call_id, "state": {"input": args, **state}}
    return [
        {"type": "step_start", "timestamp": 1, "sessionID": "ses_1", "part": {}},
        {"type": "tool_use", "timestamp": 2, "sessionID": "ses_1", "part": part},
        {"type": "step_finish", "timestamp": 3, "sessionID": "ses_1", "part": {"tokens": {"input": 1, "output": 1}}},
    ]


def test_opencode_error_state_in_opencode_txt_marks_calls_failed(tmp_path: Path) -> None:
    # A task call to an agent OpenCode does not know and an MCP call whose connection
    # closed. Harbor's trajectory keeps no result for either, so neither may read as
    # exercised; a completed call to another agent still is.
    pytest.importorskip("harbor")
    from harbor.agents.installed.opencode import OpenCode
    from harbor.models.agent.context import AgentContext

    events = [
        *_opencode_turn("c1", "task", {"subagent_type": "reviewer"}, {"status": "error", "error": UNKNOWN_AGENT}),
        *_opencode_turn("c2", "tracker_list_issues", {}, {"status": "error", "error": "MCP error -32000: closed"}),
        *_opencode_turn("c3", "task", {"subagent_type": "helper"}, {"status": "completed", "output": "Done."}),
    ]
    jobs_dir = tmp_path / "jobs"
    for variant in ("with", "without"):
        trial_dir = jobs_dir / f"demo-opencode-{variant}" / f"{CASE_ID}__AbCd001"
        (trial_dir / "verifier").mkdir(parents=True)
        (trial_dir / "verifier" / "reward.json").write_text(json.dumps(REWARD), encoding="utf-8")
        (trial_dir / "agent").mkdir()
        (trial_dir / "agent" / "opencode.txt").write_text("".join(json.dumps(e) + "\n" for e in events), "utf-8")
        OpenCode(logs_dir=trial_dir / "agent", model_name="openai/test-model").populate_context_post_run(AgentContext())
        (trial_dir.parent / "result.json").write_text(
            json.dumps(
                {
                    "n_total_trials": 1,
                    "stats": {
                        "n_trials": 1,
                        "n_errors": 0,
                        "evals": {
                            "agent__model___harbor-tasks": {
                                "n_trials": 1,
                                "n_errors": 0,
                                "reward_stats": {"reward": {"0.65": [trial_dir.name]}},
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    trajectory = json.loads((trial_dir / "agent" / "trajectory.json").read_text(encoding="utf-8"))
    assert ["observation" in step for step in trajectory["steps"]] == [False, False, True]
    case = {"id": CASE_ID, "prompt": "p", "expected_tools": ["Agent:reviewer"]}
    context = build_plugin_signals_context(
        mcp_servers=["tracker"], wrapper_skills=["demo"], entries=[case], subagents=["reviewer", "helper"]
    )

    collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "out",
        jobs_dir=jobs_dir,
        expected_cases=1,
        expected_case_ids=[CASE_ID],
        expected_trials=1,
        plugin_signals=context,
    )

    reward_path = tmp_path / "out" / "opencode" / "with-skill" / "trials" / f"{CASE_ID}__AbCd001" / "reward.json"
    signals = json.loads(reward_path.read_text(encoding="utf-8"))["plugin_signals"]
    assert [(a["type"], a["name"], a["succeeded"]) for a in signals["activations"]] == [
        ("subagent", "reviewer", False),
        ("mcp", "tracker", False),
        ("subagent", "helper", True),
    ]
    assert _counts(signals["mcp_calls"]) == (1, 0, 1, 0)
    coverage = signals["activation_coverage"]
    assert (coverage["exercised"], coverage["unavailable"]) == (
        ["subagent:helper"],
        ["mcp:tracker", "subagent:reviewer"],
    )
    # OpenCode said the agent does not exist, so the call is a wrong choice, not a hit.
    assert signals["routing"]["recall"] == 0.0


def test_opencode_error_state_fails_a_call_whose_result_the_trajectory_kept(tmp_path: Path) -> None:
    # Harbor drops an errored part's result today, but a trajectory that keeps one (here
    # partial output with no failure words) must still read the call as failed.
    trial = f"{CASE_ID}__AbCd001"
    step = _claude_call("c1", "tracker_list_issues", {}, {"content": "Listing open issues for example-org"})
    events = _opencode_turn("c1", "tracker_list_issues", {}, {"status": "error", "error": "MCP error -32000: closed"})
    jobs_dir = tmp_path / "jobs"
    for variant in ("with", "without"):
        spec = {"trajectory": _traj(step, agent="opencode"), "session": None, "codex_txt": None}
        _write_job(jobs_dir, "opencode", variant, {trial: spec})
        opencode_txt = jobs_dir / f"demo-opencode-{variant}" / trial / "agent" / "opencode.txt"
        opencode_txt.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    context = build_plugin_signals_context(
        mcp_servers=["tracker"], wrapper_skills=["demo"], entries=[{"id": CASE_ID, "prompt": "p"}]
    )

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "out",
        jobs_dir=jobs_dir,
        expected_cases=1,
        expected_case_ids=[CASE_ID],
        expected_trials=1,
        plugin_signals=context,
    )

    summary = results["agents"]["opencode"]["plugin_signals_summary"]["with_skill"]
    assert _counts(summary["mcp_calls"]) == (1, 0, 1, 0)
    assert summary["activation_coverage"]["unavailable"] == ["mcp:tracker"]
