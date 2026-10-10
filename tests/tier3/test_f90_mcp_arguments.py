# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 23 (argument correctness): failed calls, uncalled rules, server schemas, report details.

Shapes follow the proof's check-23 cases: e05 (a refused Claude call and
server-rejected Codex calls passed), e03 (an uncalled rule left out of the
rate), e06 (a ``ghp_`` dataset value echoed), e10 (the schema cap skipped for
``required``), e04 (only the first bad rule reported), e08 (``contains``
matched key names), and the report's top-failure counts.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.evaluation.tier3_report import _attach_plugin_report_fields, _top_argument_failures
from skillevaluator.tier3.eval_core.plugin_signals import (
    compute_plugin_signals,
    summarize_plugin_signals,
    validate_plugin_case_fields,
)
from skillevaluator.tier3.harbor import runner

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


# --------------------------------------------------------------------------- #
# M31: calls that failed never pass; uncalled rules count                       #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# M31: the server's own inputSchema, kept by --probe-mcp, is checked             #
# --------------------------------------------------------------------------- #
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "project": {"type": "string", "description": "lowercase project"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    },
    "required": ["project"],
    "additionalProperties": False,
}


class _McpHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        return

    def do_DELETE(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if "id" not in body:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if body["method"] == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": body.get("params", {}).get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1.0"},
            }
        else:
            result = {"tools": [{"name": "list_changes", "inputSchema": INPUT_SCHEMA}]}
        data = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", "s1")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_mcp() -> Iterator[str]:
    pytest.importorskip("mcp")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _McpHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    finally:
        server.shutdown()
        server.server_close()


def test_probed_input_schema_reaches_argument_checks(fake_mcp: str, tmp_path: Path) -> None:
    from skillevaluator.tier3.mcp_proof import probe_mcp_servers, write_mcp_input_schemas

    proof = probe_mcp_servers(
        [{"name": "reltools", "url": fake_mcp, "transport": "http"}], allowed_private_hosts=("127.0.0.1",)
    )
    assert proof["reltools"]["status"] == "reachable-host"
    package = tmp_path / "package"
    (package / "evals").mkdir(parents=True)
    assert write_mcp_input_schemas(package, proof)

    context = runner._plugin_signals_context(
        skill_path=package,
        evaluator_skill_path=package,
        workspace_skills=[],
        run_dir=tmp_path / "run",
        baseline_has_members=False,
    )
    assert context.mcp_input_schemas["reltools"]["list_changes"]["properties"]["limit"]["maximum"] == 50

    from skillevaluator.tier3.harbor.collector import _case_spec_with_input_schemas

    spec = _case_spec_with_input_schemas(context, "case-1", "with_skill")
    trajectory = _traj(
        _step("toolu_1", "mcp__reltools__list_changes", {"project": "atlas", "limit": 200}, '{"changes": []}'),
        _step("toolu_2", "mcp__reltools__list_changes", {"project": "atlas", "limit": 20}, '{"changes": []}'),
    )
    block = _arguments(trajectory, dict(spec))
    assert (block["checked"], block["passed"]) == (2, 1)
    [row] = block["failures"]
    assert (row["arg"], row["rule"]) == ("limit", "input_schema")
    assert "above maximum 50" in row["detail"]
    assert block["status"] == "scored"


def _nested_schema(depth: int) -> dict[str, Any]:
    if depth == 0:
        return {"type": "string", "enum": ["a", "b"]}
    return {
        "type": "object",
        "required": ["x", "y"],
        "properties": {"x": _nested_schema(depth - 1), "y": {"type": "integer", "minimum": 0}},
    }


def _large_tool_schema() -> dict[str, Any]:
    # About 1.5 KB of kept schema, under the 2 KiB per-tool cap.
    properties = {f"p{index}": _nested_schema(4) for index in range(3)}
    return {"type": "object", "required": [f"p{index}" for index in range(6)], "properties": properties}


@pytest.mark.parametrize("names_per_kind", [3, 240], ids=["few-names", "long-name-lists"])
def test_probed_schemas_near_the_size_cap_keep_subagents_and_commands(tmp_path: Path, names_per_kind: int) -> None:
    # Verifier probe for M31: 6 probed servers x 50 tools wrote a 457,784-byte
    # file (indent=2), over the 256 KiB read limit, so the run dropped the
    # declared subagents, commands, aliases and the schemas themselves.
    from skillevaluator.tier3.harbor.adapter import (
        _MAX_PLUGIN_RUNTIME_COMPONENTS_BYTES,
        load_plugin_runtime_components,
        load_plugin_subagent_aliases,
    )
    from skillevaluator.tier3.mcp_proof import load_mcp_input_schemas, write_mcp_input_schemas

    package = tmp_path / "package"
    env_dir = package / "evals" / "environment"
    env_dir.mkdir(parents=True)
    subagents = [f"release-reviewer-{index:03d}-" + "s" * 200 for index in range(names_per_kind)]
    commands = [f"ship-release-{index:03d}-" + "c" * 200 for index in range(names_per_kind)]
    aliases = {f"release-kit-{name}": name for name in subagents[:100]}
    target = env_dir / "plugin_runtime_components.json"
    target.write_text(
        json.dumps({"subagents": subagents, "commands": commands, "subagent_aliases": aliases}, indent=2),
        encoding="utf-8",
    )
    proof = {
        f"srv{server}": {
            "status": "reachable-host",
            "input_schemas": {f"tool{tool}": _large_tool_schema() for tool in range(50)},
        }
        for server in range(6)
    }

    assert write_mcp_input_schemas(package, proof)

    assert target.stat().st_size <= _MAX_PLUGIN_RUNTIME_COMPONENTS_BYTES
    assert load_plugin_runtime_components(package) == {"subagents": subagents, "commands": commands}
    assert load_plugin_subagent_aliases(package) == aliases
    schemas = load_mcp_input_schemas(package)
    assert sum(len(tools) for tools in schemas.values()) >= 40
    context = runner._plugin_signals_context(
        skill_path=package,
        evaluator_skill_path=package,
        workspace_skills=[],
        run_dir=tmp_path / "run",
        baseline_has_members=False,
    )
    assert list(context.subagents) == subagents and list(context.commands) == commands
    assert context.mcp_input_schemas == schemas


def test_probed_schemas_are_skipped_when_the_names_leave_no_room(tmp_path: Path) -> None:
    # A file already at the limit keeps its subagents and commands; no schema is added.
    from skillevaluator.tier3.harbor.adapter import _MAX_PLUGIN_RUNTIME_COMPONENTS_BYTES, load_plugin_runtime_components
    from skillevaluator.tier3.mcp_proof import write_mcp_input_schemas

    package = tmp_path / "package"
    env_dir = package / "evals" / "environment"
    env_dir.mkdir(parents=True)
    names = {"subagents": ["reviewer-" + "r" * (_MAX_PLUGIN_RUNTIME_COMPONENTS_BYTES - 64)], "commands": ["ship"]}
    target = env_dir / "plugin_runtime_components.json"
    target.write_text(json.dumps(names), encoding="utf-8")
    proof = {"srv": {"status": "reachable-host", "input_schemas": {"tool": _large_tool_schema()}}}

    assert not write_mcp_input_schemas(package, proof)
    assert json.loads(target.read_text(encoding="utf-8")) == names
    assert load_plugin_runtime_components(package)["commands"] == ["ship"]


# --------------------------------------------------------------------------- #
# L27: report details                                                          #
# --------------------------------------------------------------------------- #
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


def _failing_rows(count: int) -> list[dict[str, str]]:
    return [{"tool": "mcp__reltools__x", "arg": "limit", "rule": "schema", "detail": "too big"}] * count


def test_top_failure_counts_are_exact_beyond_32_rows() -> None:
    # check-23 replay "main": the report said 33 where the true count was 51.
    rewards = [
        {"plugin_signals": {"arguments": {"failures": _failing_rows(45)}}},
        {"plugin_signals": {"arguments": {"failures": _failing_rows(6)}}},
    ]
    [top] = _top_argument_failures(rewards)
    assert top["count"] == 51


def test_top_failures_survive_when_a_report_drops_the_rewards(tmp_path: Path) -> None:
    # check-23: an arm whose condition did not fully succeed loses its rewards in the report data.
    from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context
    from skillevaluator.tier3.harbor.collector import collect_harbor_results

    case = {"id": "case-1", "prompt": "p", "tool_arguments": [RULE]}
    trajectory = _traj(_step("toolu_1", "mcp__reltools__list_changes", {"project": "kb"}, '{"changes": []}'))
    reward = {
        "security": 0.9,
        "skill_execution": 0.8,
        "skill_efficiency": 0.7,
        "accuracy": 0.6,
        "goal_accuracy": 0.5,
        "behavior_check": 0.4,
    }
    jobs = tmp_path / "jobs"
    for variant in ("with", "without"):
        trial = jobs / f"demo-claude-code-{variant}" / "case-1__AbCd123"
        (trial / "verifier").mkdir(parents=True)
        (trial / "agent").mkdir()
        (trial / "verifier" / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
        (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
        (trial.parent / "result.json").write_text(
            json.dumps(
                {
                    "n_total_trials": 1,
                    "stats": {
                        "n_trials": 1,
                        "n_errors": 0,
                        "evals": {
                            "a__m___t": {
                                "n_trials": 1,
                                "n_errors": 0,
                                "reward_stats": {"reward": {"0.65": [trial.name]}},
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    results = collect_harbor_results(
        skill_name="demo",
        agents=["claude-code"],
        output_dir=tmp_path / "out",
        jobs_dir=jobs,
        expected_cases=1,
        expected_case_ids=["case-1"],
        expected_trials=1,
        plugin_signals=build_plugin_signals_context(mcp_servers=["reltools"], wrapper_skills=["demo"], entries=[case]),
    )
    summaries = results["agents"]["claude-code"]["plugin_signals_summary"]
    payload: dict[str, Any] = {"agents": {"claude-code": {}}, "best_agent": "claude-code"}
    _attach_plugin_report_fields(payload, {"claude-code": {"plugin_signals_summary": summaries, "rewards": []}})

    top = payload["plugin_signals_summary"]["with_skill"]["arguments"]["top_failures"]
    assert sorted((row["arg"], row["rule"], row["count"]) for row in top) == [
        ("project", "equals", 1),
        ("since", "required", 1),
    ]


def test_report_keeps_exact_top_failure_counts_beyond_256_rewards(tmp_path: Path) -> None:
    # Verifier probe for L27: the collector's exact counts were recomputed from at
    # most 256 rewards in the report, so 300 failing trials read as 256.
    from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context
    from skillevaluator.tier3.harbor.collector import collect_harbor_results

    case = {"id": "case-1", "prompt": "p", "tool_arguments": [RULE]}
    trajectory = _traj(
        _step("toolu_1", "mcp__reltools__list_changes", {"project": "kb", "since": "1.4.0"}, '{"changes": []}')
    )
    reward = {
        "security": 0.9,
        "skill_execution": 0.8,
        "skill_efficiency": 0.7,
        "accuracy": 0.6,
        "goal_accuracy": 0.5,
        "behavior_check": 0.4,
    }
    job = tmp_path / "jobs" / "demo-claude-code-with"
    names = [f"case-1__T{index:03d}xyz" for index in range(300)]
    for name in names:
        trial = job / name
        (trial / "verifier").mkdir(parents=True)
        (trial / "agent").mkdir()
        (trial / "verifier" / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
        (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    stats = {"n_trials": 300, "n_errors": 0, "reward_stats": {"reward": {"0.75": names}}}
    (job / "result.json").write_text(
        json.dumps({"n_total_trials": 300, "stats": {**stats, "evals": {"a__m___t": stats}}}), encoding="utf-8"
    )
    results = collect_harbor_results(
        skill_name="demo",
        agents=["claude-code"],
        output_dir=tmp_path / "out",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
        n_attempts=300,
        expected_cases=1,
        expected_case_ids=["case-1"],
        plugin_signals=build_plugin_signals_context(mcp_servers=["reltools"], wrapper_skills=["demo"], entries=[case]),
    )
    summaries = results["agents"]["claude-code"]["plugin_signals_summary"]
    rewards = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "out" / "claude-code" / "with-skill" / "trials").glob("*/reward.json"))
    ]
    assert len(rewards) == 300 and all(reward["plugin_signals"]["arguments"]["failures"] for reward in rewards)
    payload: dict[str, Any] = {"agents": {"claude-code": {}}, "best_agent": "claude-code"}

    _attach_plugin_report_fields(payload, {"claude-code": {"plugin_signals_summary": summaries, "rewards": rewards}})

    [top] = payload["plugin_signals_summary"]["with_skill"]["arguments"]["top_failures"]
    assert (top["arg"], top["rule"], top["count"]) == ("project", "equals", 300)


def test_report_still_counts_rewards_for_a_summary_without_top_failures() -> None:
    # Older runs: the summary has no top_failures, so the report counts the rewards it has.
    signals = {"arguments": {"failures": _failing_rows(2)}}
    summaries = {"with_skill": {"arguments": {"checked": 4, "passed": 2}}}
    payload: dict[str, Any] = {"agents": {"claude-code": {}}, "best_agent": "claude-code"}
    agent = {"plugin_signals_summary": summaries, "rewards": [{"plugin_signals": signals}] * 3}

    _attach_plugin_report_fields(payload, {"claude-code": agent})

    [top] = payload["plugin_signals_summary"]["with_skill"]["arguments"]["top_failures"]
    assert top["count"] == 6
