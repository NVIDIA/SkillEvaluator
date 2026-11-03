# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence attribution remains uncertain when identities or child artifacts are incomplete."""

from __future__ import annotations

import json

import pytest

from skillevaluator.tier3 import plugin_trace_artifacts as artifacts
from skillevaluator.tier3.eval_core import plugin_signals as signals


def call(call_id, name, args=None, *, refs=None):
    result = {"source_call_id": call_id, "content": "ok", "extra": {"tool_result_is_error": False}}
    if refs is not None:
        result["subagent_trajectory_ref"] = refs
    return {
        "step_id": 1,
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": call_id, "function_name": name, "arguments": args or {}}],
        "observation": {"results": [result]},
    }


def tree(*, external=False):
    child = {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": "child",
        "agent": {"name": "claude-code", "version": "test"},
        "steps": [call("c", "mcp__docs__search")],
    }
    ref = {"trajectory_path": "child.json"} if external else {"trajectory_id": "child"}
    root = {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": "root",
        "agent": {"name": "claude-code", "version": "test"},
        "steps": [call("spawn", "Agent", {"subagent_type": "reviewer"}, refs=[ref])],
    }
    if not external:
        root["subagent_trajectories"] = [child]
    return root, child


def grade(trajectory):
    return signals.compute_plugin_signals(
        trajectory, {"expected_tools": ["mcp:docs/search"]}, declared={"mcp": ["docs"], "subagent": ["reviewer"]}
    )


@pytest.mark.parametrize("server", [None, "docs"])
@pytest.mark.parametrize("output", ["same answer", "unmatched answer"])
def test_ambiguous_codex_items_do_not_borrow_identity_or_success(server, output):
    items = [
        {"tool": "search", "args": "{}", "server": "docs", "status": "completed", "text": "same answer"},
        {"tool": "search", "args": "{}", "server": server or "kb", "status": "failed", "text": "same answer"},
    ]
    assert artifacts._take_codex_item(items, tool="search", args="{}", server=server, output=output) is None
    assert len(items) == 2


def test_long_codex_argument_keys_cannot_match_by_a_shared_prefix():
    first, second = {"q": "x" * 5000 + "one"}, {"q": "x" * 5000 + "two"}
    key = artifacts._codex_args_key(first)
    other = artifacts._codex_args_key(second)
    assert key != other or key is None
    assert (
        artifacts._take_codex_item(
            [{"tool": "search", "args": other, "server": "docs", "status": "completed"}],
            tool="search",
            args=key,
            server="docs",
        )
        is None
    )


def test_embedded_executed_child_contributes_calls_once_without_mutating_source():
    root, _ = tree()
    root["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"] *= 2
    before = json.dumps(root, sort_keys=True)
    result = grade(root)
    assert result["mcp_calls"]["succeeded"] == result["mcp_calls"]["total"] == 1
    assert result["tool_selection"]["recall"] == 1.0 and result["evidence_complete"] is True
    assert json.dumps(root, sort_keys=True) == before


@pytest.mark.parametrize("kind", ["external", "missing", "cycle"])
def test_unresolved_or_cyclic_child_references_cannot_claim_complete_evidence(kind):
    root, child = tree(external=kind == "external")
    if kind == "missing":
        root.pop("subagent_trajectories")
    elif kind == "cycle":
        child["subagent_trajectories"] = [root]
        child["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"] = [{"trajectory_id": "root"}]
    assert grade(root)["evidence_complete"] is False


@pytest.mark.parametrize("external", [False, True])
def test_adapter_materializes_authorized_child_artifacts(tmp_path, external):
    root, child = tree(external=external)
    agent = tmp_path / "trial/agent"
    agent.mkdir(parents=True)
    (agent / "trajectory.json").write_text(json.dumps(root))
    if external:
        (agent / "child.json").write_text(json.dumps(child))
    trace = artifacts.TrialTraceArtifacts().trace(agent.parent)
    assert grade(trace.trajectory)["mcp_calls"]["succeeded"] == 1


def test_adapter_rejects_root_that_contradicts_authoritative_steps(tmp_path):
    root, _ = tree()
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent/trajectory.json").write_text(json.dumps(root))
    (trial / "result.json").write_text(json.dumps({"step_results": [{"step_name": "one"}]}))
    assert artifacts.TrialTraceArtifacts().trace(trial).trajectory is None


def test_redaction_drops_a_secret_cut_after_a_shrinking_assignment():
    secret, bound = "ghp_" + "B" * 36, 256 * 4 + 256
    prefix = "token=" + "a" * (bound - 39 - 7) + " "
    rendered = signals._safe_text(prefix + secret)
    assert "ghp_" not in rendered


@pytest.mark.parametrize("control", ["\x01", "\x9b", "\x1b[31m", "\x1b]0;title\x07"])
def test_diagnostics_redact_credentials_after_removing_controls(control):
    from skillevaluator.tier3 import mcp_proof, plugin_evidence

    value = "ghp_" + "B" * 33 + control + "C" * 3
    for render in (signals._safe_text, plugin_evidence._safe_text, mcp_proof._detail):
        text = render(value)
        assert "ghp_" not in text
        assert "B" * 12 not in text
        assert "redacted" in text


def test_malformed_codex_types_do_not_discard_valid_log_items(tmp_path):
    root = tmp_path / "trial"
    agent = root / "agent"
    sessions = agent / "sessions"
    sessions.mkdir(parents=True)
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "codex", "version": "test"},
        "steps": [call("c", "search")],
    }
    (agent / "trajectory.json").write_text(json.dumps(trajectory))
    (agent / "codex.txt").write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"type": [], "mcp": 1},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "mcp_tool_call",
                        "id": "i",
                        "server": "docs",
                        "tool": "search",
                        "arguments": {},
                        "status": "failed",
                    },
                },
            ]
        )
    )
    (sessions / "session.jsonl").write_text(
        "\n".join(
            json.dumps({"payload": payload})
            for payload in [
                {"item": {"type": [], "mcp": 1}},
                {"type": {}, "mcp": 1},
                {
                    "type": "function_call",
                    "namespace": "mcp__docs",
                    "name": "search",
                    "call_id": "c",
                    "arguments": "{}",
                },
            ]
        )
    )
    trace = artifacts.TrialTraceArtifacts().trace(root)
    result = signals.compute_plugin_signals(
        trace.trajectory, {}, declared={"mcp": ["docs"]}, mcp_call_servers=trace.call_servers
    )
    assert trace.call_servers == {"c": "docs"}
    assert result["mcp_calls"]["failed"] == result["mcp_calls"]["total"] == 1


@pytest.mark.parametrize("overlay", ["_codex_mcp_calls", "_opencode_call_errors"])
def test_harness_overlay_failure_keeps_plain_trajectory_and_other_trials(tmp_path, monkeypatch, overlay):
    original = getattr(artifacts, overlay)

    def failing_overlay(root, *args):
        if root.name == "bad":
            raise TypeError("malformed harness event")
        return original(root, *args)

    monkeypatch.setattr(artifacts, overlay, failing_overlay)
    cache = artifacts.TrialTraceArtifacts()
    for name in ("bad", "good"):
        agent = tmp_path / name / "agent"
        agent.mkdir(parents=True)
        (agent / "trajectory.json").write_text(
            json.dumps(
                {
                    "schema_version": "ATIF-v1.7",
                    "agent": {"name": "codex", "version": "test"},
                    "steps": [call("c", "mcp__docs__search")],
                }
            )
        )
        trace = cache.trace(agent.parent)
        assert grade(trace.trajectory)["mcp_calls"]["succeeded"] == 1
        assert cache.trace(agent.parent) is trace


def test_unstaged_file_server_is_skipped_without_hiding_a_portable_server(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    from tests.tier3.test_plugin_evidence import prepared

    from skillevaluator.plugin_inventory import plugin_inventory_for_root
    from skillevaluator.tier3.plugin_evidence import EvidenceOptions, RuntimeEvidenceObserver
    from skillevaluator.tier3.plugin_native import wrapper_decision
    from skillevaluator.tier3.plugin_prepare import prepare_runtime_package, stage_wrapper_skill

    prepared(tmp_path / "plugin")
    manifest = tmp_path / "plugin/.claude-plugin/plugin.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "demo",
                "mcpServers": {
                    "docs": {"url": "https://docs.example.invalid/mcp"},
                    "local": {"command": "python", "args": ["./server.py"]},
                },
            }
        )
    )
    (tmp_path / "plugin/server.py").write_text("print('fixture')")
    package = prepare_runtime_package(plugin_inventory_for_root(tmp_path / "plugin"))
    assert {server["name"] for server in package.native_source.plugin_file_mcp_servers} == {"local"}
    wrapper = stage_wrapper_skill(package, tmp_path / "wrapper")
    hooks = SimpleNamespace(decisions={"codex": wrapper_decision("codex", "fixture")}, stagings={})
    observer = RuntimeEvidenceObserver(package, wrapper, hooks, EvidenceOptions())
    observer._context_for_staging()
    declared = observer.context.declared_for("with_skill", "codex")
    assert declared["mcp"] == ["docs"]
    result = signals.compute_plugin_signals(
        {"steps": [], "agent": {"name": "codex"}},
        {"expected_tools": ["mcp:local/tool", "mcp:docs/search"]},
        declared=declared,
    )
    assert result["tool_selection"]["skipped"] == ["mcp:local/tool"]
    assert result["tool_selection"]["recall"] == 0.0
    native = replace(observer.context, agent_mcp_servers={"claude-code": ("local",)})
    assert set(native.declared_for("with_skill", "claude-code")["mcp"]) == {"docs", "local"}


@pytest.mark.parametrize("known_owner", [True, False])
def test_child_handoff_uses_its_referenced_parent_not_another_parallel_agent(known_owner):
    root, child = tree()
    parent = root["steps"][0]
    parent["tool_calls"].append(
        {"tool_call_id": "other", "function_name": "Agent", "arguments": {"subagent_type": "distractor"}}
    )
    result = parent["observation"]["results"][0]
    if not known_owner:
        result["source_call_id"] = "missing-parent"
    child["steps"][0]["observation"]["results"][0]["content"] = "carried-value"
    root["steps"].append(call("consume", "mcp__docs__consume", {"value": "carried-value"}))
    result = signals.compute_plugin_signals(
        root,
        {"handoffs": [{"producer": "Agent:reviewer", "consumer": "MCP:docs/consume", "value": "carried-value"}]},
        declared={"mcp": ["docs"], "subagent": ["reviewer", "distractor"]},
    )
    assert result["handoff"]["passed"] == int(known_owner)
    assert result["evidence_complete"] is known_owner


def test_child_work_shares_the_global_step_budget(monkeypatch):
    root, _ = tree()
    root["steps"].append(call("later", "mcp__docs__search"))
    monkeypatch.setattr(signals, "_MAX_STEPS", 2)
    result = grade(root)
    assert result["mcp_calls"]["total"] == 1
    assert result["evidence_complete"] is False and "steps" in result["limits"]


@pytest.mark.parametrize("native", [False, True])
def test_prepared_agent_components_keep_their_declared_names_and_staging_scope(tmp_path, native):
    from types import SimpleNamespace

    from tests.tier3.test_plugin_evidence import prepared

    from skillevaluator.plugin_inventory import plugin_inventory_for_root
    from skillevaluator.tier3.plugin_evidence import EvidenceOptions, RuntimeEvidenceObserver
    from skillevaluator.tier3.plugin_native import wrapper_decision
    from skillevaluator.tier3.plugin_prepare import prepare_runtime_package, stage_wrapper_skill

    prepared(tmp_path / "plugin")
    agents = tmp_path / "plugin/agents"
    agents.mkdir()
    (agents / "reviewer.md").write_text("---\nname: reviewer\ndescription: Review.\n---\nReview the work.\n")
    package = prepare_runtime_package(plugin_inventory_for_root(tmp_path / "plugin"))
    assert any(text.type == "agent" and text.name == "reviewer" for text in package.native_source.texts)
    wrapper = stage_wrapper_skill(package, tmp_path / "wrapper")
    decision = wrapper_decision("codex", "fixture")
    staging = SimpleNamespace(
        bundle=SimpleNamespace(declared=[{"type": "agent", "name": "reviewer"}]), adapter=SimpleNamespace()
    )
    hooks = SimpleNamespace(
        decisions={"codex": SimpleNamespace(native=True) if native else decision},
        stagings={"codex": staging} if native else {},
    )
    observer = RuntimeEvidenceObserver(package, wrapper, hooks, EvidenceOptions())
    observer._context_for_staging()
    assert observer.context.subagents == ("reviewer",)
    declared = observer.context.declared_for("with_skill", "codex")
    result = signals.compute_plugin_signals(
        {"steps": [], "agent": {"name": "claude-code"}}, {"expected_tools": ["Agent:reviewer"]}, declared=declared
    )
    assert result["routing"]["skipped"] == ([] if native else ["Agent:reviewer"])


def test_ambiguous_same_server_codex_outcomes_remain_unknown_after_overlay(tmp_path):
    from tests.tier3.test_f90_mcp_execution import _codex_items, _codex_step, _collect_with_arm, _traj

    args = {"q": "same"}
    trial = {
        "trajectory": _traj(_codex_step("call", "search", args, "same answer"), agent="codex"),
        "session": None,
        "codex_txt": [
            *_codex_items("one", "docs", "search", args, "same answer", "completed"),
            *_codex_items("two", "docs", "search", args, "same answer", "failed"),
        ],
    }
    summary = _collect_with_arm(tmp_path, {"case-1__attempt1": trial}, servers=["docs"])
    assert summary["mcp_calls"]["succeeded"] == 0
    assert summary["mcp_calls"]["unknown"] == 1
