# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private proof bounds and harness evidence retain their source and uncertainty."""

from __future__ import annotations

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from tests.tier3.test_f90_mcp_execution import _codex_items, _codex_step, _session_line, _traj

from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals
from skillevaluator.tier3.mcp_proof import MAX_RESPONSE_BYTES, apply_in_agent_mcp_proof, probe_mcp_server
from skillevaluator.tier3.plugin_trace_artifacts import TrialTraceArtifacts


def test_compressed_reply_cannot_bypass_the_host_probe_body_limit():
    seen_encodings = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def empty(self, status):
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            self.empty(405)

        def do_DELETE(self):
            self.empty(200)

        def do_POST(self):
            seen_encodings.append(self.headers.get("Accept-Encoding"))
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if "id" not in request:
                self.empty(202)
                return
            compressed = request["method"] == "tools/list"
            result = (
                {
                    "tools": [
                        {
                            "name": "search",
                            "description": "x" * (2 * MAX_RESPONSE_BYTES),
                            "inputSchema": {"type": "object"},
                        }
                    ]
                }
                if compressed
                else {
                    "protocolVersion": request["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fixture", "version": "1"},
                }
            )
            data = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
            if compressed:
                data = gzip.compress(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        proof = probe_mcp_server(
            {
                "name": "docs",
                "url": f"http://127.0.0.1:{server.server_address[1]}/mcp",
                "headers": {"accept-encoding": "gzip"},
            },
            allowed_private_hosts=("127.0.0.1",),
            total_timeout=10,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert proof["status"] == "unreachable"
    assert "encoding" in proof["detail"].casefold()
    assert seen_encodings and set(seen_encodings) == {"identity"}


@pytest.mark.parametrize("session", [False, True])
def test_multistep_codex_logs_keep_reused_call_ids_in_their_source_step(tmp_path, session):
    for index, (server, status) in enumerate((("docs", "completed"), ("tracker", "failed"))):
        agent = tmp_path / "steps" / f"step-{index}" / "agent"
        agent.mkdir(parents=True)
        args, answer = {"q": "same query"}, "same answer"
        (agent / "trajectory.json").write_text(
            json.dumps(_traj(_codex_step("call", "search", args, answer), agent="codex"))
        )
        (agent / "codex.txt").write_text("\n".join(_codex_items("item_1", server, "search", args, answer, status)))
        if session:
            sessions = agent / "sessions"
            sessions.mkdir()
            (sessions / "rollout.jsonl").write_text(_session_line("call", server, "search", args))
    (tmp_path / "result.json").write_text(
        json.dumps({"step_results": [{"step_name": "step-0"}, {"step_name": "step-1"}]})
    )
    # A log outside the authoritative step set must contribute no evidence.
    unrelated = tmp_path / "agent"
    unrelated.mkdir()
    (unrelated / "codex.txt").write_text(
        "\n".join(_codex_items("item_1", "tracker", "search", args, answer, "completed"))
    )
    trace = TrialTraceArtifacts().trace(tmp_path)
    result = compute_plugin_signals(
        trace.trajectory,
        {"expected_tools": ["MCP:docs/search", "MCP:tracker/search"]},
        declared={"mcp": ["docs", "tracker"]},
        mcp_call_servers=trace.call_servers,
    )
    assert result["mcp_calls"]["total"] == 2
    assert result["mcp_calls"]["by_server"]["docs"]["succeeded"] == 1
    assert result["mcp_calls"]["by_server"]["tracker"]["failed"] == 1
    assert result["tool_selection"]["recall"] == 1.0
    assert result["evidence_complete"] is True


def test_multistep_opencode_dropped_results_keep_step_local_errors(tmp_path):
    for index in range(2):
        agent = tmp_path / "steps" / f"step-{index}" / "agent"
        agent.mkdir(parents=True)
        step = _codex_step("call", "docs_search", {}, "found")
        if index:
            step.pop("observation")
            (agent / "opencode.txt").write_text(
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "callID": "call",
                            "tool": "docs_search",
                            "state": {"status": "error", "error": "MCP error -32000: closed"},
                        },
                    }
                )
            )
        (agent / "trajectory.json").write_text(json.dumps(_traj(step, agent="opencode")))
    (tmp_path / "result.json").write_text(
        json.dumps({"step_results": [{"step_name": "step-0"}, {"step_name": "step-1"}]})
    )
    trace = TrialTraceArtifacts().trace(tmp_path)
    result = compute_plugin_signals(trace.trajectory, {}, declared={"mcp": ["docs"]})
    assert result["mcp_calls"]["total"] == 2
    assert result["mcp_calls"]["succeeded"] == result["mcp_calls"]["failed"] == 1
    assert result["mcp_calls"]["unknown"] == 0


@pytest.mark.parametrize("external", [False, True])
def test_multistep_call_scopes_preserve_child_ownership_and_handoffs(tmp_path, external):
    from tests.tier3.test_plugin_evidence_review import call, tree

    for index in range(2):
        agent = tmp_path / "steps" / f"step-{index}" / "agent"
        agent.mkdir(parents=True)
        root, child = tree(external=external)
        child["steps"][0]["observation"]["results"][0]["content"] = "carried-value"
        root["steps"].append(call("consume", "mcp__docs__consume", {"value": "carried-value"}))
        root["steps"][-1]["step_id"] = 2
        (agent / "trajectory.json").write_text(json.dumps(root))
        if external:
            (agent / "child.json").write_text(json.dumps(child))
    (tmp_path / "result.json").write_text(
        json.dumps({"step_results": [{"step_name": "step-0"}, {"step_name": "step-1"}]})
    )
    trace = TrialTraceArtifacts().trace(tmp_path)
    result = compute_plugin_signals(
        trace.trajectory,
        {"handoffs": [{"producer": "Agent:reviewer", "consumer": "MCP:docs/consume", "value": "carried-value"}]},
        declared={"mcp": ["docs"], "subagent": ["reviewer"]},
    )
    assert result["mcp_calls"]["total"] == 4
    assert result["handoff"]["passed"] == 1
    assert result["evidence_complete"] is True


def test_single_step_authored_multistep_extras_cannot_select_other_logs(tmp_path):
    trial, foreign = tmp_path / "trial", tmp_path / "foreign"
    for directory, server, status in ((trial, "docs", "failed"), (foreign, "tracker", "completed")):
        agent = directory / "agent"
        agent.mkdir(parents=True)
        args = {"q": "same query"}
        (agent / "codex.txt").write_text(
            "\n".join(_codex_items("item_1", server, "search", args, "same answer", status))
        )
    trajectory = _traj(_codex_step("call", "search", args, "same answer"), agent="codex")
    trajectory["extra"] = {"harbor_multi_step": {"step_names": ["../../foreign"]}}
    (trial / "agent/trajectory.json").write_text(json.dumps(trajectory))
    trace = TrialTraceArtifacts().trace(trial)
    result = compute_plugin_signals(
        trace.trajectory,
        {},
        declared={"mcp": ["docs", "tracker"]},
        mcp_call_servers=trace.call_servers,
    )
    assert result["mcp_calls"]["by_server"]["docs"]["failed"] == 1
    assert "tracker" not in result["mcp_calls"]["by_server"]


@pytest.mark.parametrize("unknown", [1, 2])
def test_unknown_trial_load_preserves_host_reachability(unknown):
    final = _load_proof({"loaded": 0, "trials": 1, "unknown": unknown})
    assert final["status"] == "reachable-host"
    assert "unknown" in final["detail"]


@pytest.mark.parametrize("tally", [{"loaded": 0, "trials": 2}, {"loaded": 0, "trials": 2, "unknown": 0}])
def test_known_never_loaded_classification_accepts_legacy_tallies(tally):
    assert _load_proof(tally)["status"] == "not-loaded-in-agent"


def _load_proof(tally):
    engine = {
        "agents": {
            "claude-code": {
                "plugin_load_census": {"not_loaded": [{"type": "mcp", "name": "docs", "reason": "init failed"}]},
                "plugin_signals_summary": {"with_skill": {"mcp_load": {"docs": tally}, "mcp_calls": {"by_server": {}}}},
            }
        }
    }
    return apply_in_agent_mcp_proof(
        {"docs": {"status": "reachable-host", "tools": [], "detail": "initialize and tools/list succeeded"}},
        engine,
    )["docs"]
