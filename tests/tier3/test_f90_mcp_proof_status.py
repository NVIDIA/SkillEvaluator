# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 21: "proven reachable" needs a successful call or a real initialize.

Shapes follow the proof's check-21 cases: e05 (Claude init said cfdocs
failed in every trial), e06 (Claude never loaded a second server), p08 (every
call failed), a failed call to a host-unreachable server, and the verifier's
M30 probe (a server that failed to load in only one of 12 trials).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.reporting.plugin_sections import MCP_PROOF_LABELS, mcp_proof_view
from skillevaluator.tier3.eval_core.plugin_signals import (
    build_plugin_signals_context,
    compute_plugin_signals,
    summarize_plugin_signals,
)
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle
from skillevaluator.tier3.mcp_proof import NOT_REQUESTED_DETAIL, apply_in_agent_mcp_proof
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.tier3.plugin_native import HARNESS_ADAPTERS, LOAD_CENSUS_FILENAME, resolve_plugin_load

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
        "not_loaded": [{"type": "mcp", "name": name, "reason": reason} for name, reason in not_loaded.items()],
    }


def _statuses(proof: dict[str, Any]) -> dict[str, str]:
    return {name: entry["status"] for name, entry in proof.items()}


def test_server_whose_init_failed_in_the_agent_is_not_proven() -> None:
    # check-21 e05: the host probe reached cfdocs, but Claude's init said "failed" in 12 of 12 trials.
    proof = {"deepwiki": _probe("reachable-host"), "cfdocs": _probe("reachable-host")}
    census = _census(
        ["deepwiki"], {"cfdocs": "claude-code init reported MCP server plugin:docs:cfdocs with status failed"}
    )

    final = apply_in_agent_mcp_proof(proof, _engine({"deepwiki": {"total": 10, "succeeded": 10}}, census))

    assert _statuses(final) == {"deepwiki": "used-successfully", "cfdocs": "not-loaded-in-agent"}
    assert "status failed" in final["cfdocs"]["detail"]
    assert "host probe: " + HOST_OK in final["cfdocs"]["detail"]
    view = mcp_proof_view(final)
    assert view is not None
    assert view["headline"] == "1 of 2 URL MCP servers proven reachable"


def test_server_the_agent_never_loaded_is_not_proven() -> None:
    # check-21 e06: Claude Code dropped deepwiki-bad (same URL as deepwiki) and never listed it.
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
    assert mcp_proof_view(final)["headline"] == "1 of 3 URL MCP servers proven reachable"


def test_called_but_nothing_succeeded_is_not_green_and_not_counted() -> None:
    # check-21 p08: two calls to deepwiki-bad, both failed.
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
    view = mcp_proof_view(final)
    row = next(row for row in view["rows"] if row["server"] == "deepwiki-bad")
    assert row["status_class"] != "ok"
    assert view["headline"] == "0 of 3 URL MCP servers proven reachable"
    # Saved provenance from older runs still renders without a green pill.
    assert MCP_PROOF_LABELS["reachable-in-agent"][1] != "ok"


def test_a_failed_call_never_turns_an_unreachable_server_reachable() -> None:
    proof = {"gone": _probe("unreachable", "DNS resolution failed: gaierror")}
    final = apply_in_agent_mcp_proof(proof, _engine({"gone": {"total": 1, "succeeded": 0, "failed": 1}}))
    assert final["gone"]["status"] == "unreachable"
    assert mcp_proof_view(final)["headline"] == "0 of 1 URL MCP server proven reachable"


def test_one_successful_call_still_proves_the_server() -> None:
    proof = {"gone": _probe("unreachable", "DNS resolution failed: gaierror")}
    census = _census([], {"gone": "claude-code init reported MCP server plugin:docs:gone with status failed"})
    final = apply_in_agent_mcp_proof(proof, _engine({"gone": {"total": 2, "succeeded": 1, "failed": 1}}, census))
    assert final["gone"]["status"] == "used-successfully"


def test_failed_claude_calls_end_to_end_do_not_prove_the_server() -> None:
    # check-24 e01 into check 21: Harbor's Claude flag fails the call, so the proof is not "used-successfully".
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


# --------------------------------------------------------------------------- #
# M30: one flaky trial is not "never loaded"                                   #
# --------------------------------------------------------------------------- #
SLUG = "release-helper"
REWARD = {
    "security": 0.9,
    "skill_execution": 0.8,
    "skill_efficiency": 0.7,
    "accuracy": 0.6,
    "goal_accuracy": 0.5,
    "behavior_check": 0.4,
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def native_census(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """A native Claude Code plan and the census its real setup.sh writes, for a plugin with two MCP servers."""
    plugin = tmp_path / SLUG
    _write(plugin / ".claude-plugin" / "plugin.json", json.dumps({"name": SLUG, "description": "Release notes."}))
    _write(plugin / "skills" / "notes" / "SKILL.md", "---\nname: notes\ndescription: Draft notes.\n---\nBody.\n")
    servers = {
        "docs": {"command": "npx", "args": ["-y", "@example/docs-mcp@1.0.0"]},
        "kb": {"command": "npx", "args": ["-y", "@example/kb-mcp@1.0.0"]},
    }
    _write(plugin / ".mcp.json", json.dumps({"mcpServers": servers}))
    case = {"id": "case-1", "prompt": "Draft notes.", "expected_output": "Notes.", "assertions": ["Drafted"]}
    _write(plugin / "evals" / "evals.json", json.dumps({"skill_name": SLUG, "evals": [case]}))
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", plugin_load="native")
    assert package.native_source is not None
    staging = build_native_task_staging("claude-code", HARNESS_ADAPTERS["claude-code"], package.native_source)
    env_dir = tmp_path / "tasks" / "environment"
    env_dir.mkdir(parents=True)
    stage_native_bundle(env_dir, staging)
    # Run the generated setup.sh against a fake container to get a real census file.
    root = tmp_path / "container"
    container = root / "skilleval"
    shutil.copytree(env_dir / "skilleval", container)
    logs = root / "logs" / "agent"
    logs.mkdir(parents=True)
    script = (container / "native" / "setup.sh").read_text(encoding="utf-8")
    _write(root / "setup.sh", script.replace("/skilleval/", f"{container}/").replace("/logs/agent", str(logs)))
    env = {"PATH": "/usr/bin:/bin", "HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(root / "cfg")}
    completed = subprocess.run(
        ["/bin/sh", str(root / "setup.sh")], env=env, capture_output=True, text=True, check=False, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    decisions = resolve_plugin_load("native", ["claude-code"], env_mode="docker")
    plan = runner._plugin_load_census_plan("native", decisions, {"claude-code": staging}, package.native_source)
    assert plan is not None
    return plan["claude-code"], logs / LOAD_CENSUS_FILENAME


def _init(statuses: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "init",
        "plugins": [{"name": SLUG, "source": f"{SLUG}@inline"}],
        "skills": [f"{SLUG}:notes"],
        "mcp_servers": [{"name": f"plugin:{SLUG}:{name}", "status": status} for name, status in statuses.items()],
    }


def _failed_docs_call(call_id: str) -> dict[str, Any]:
    return {
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": call_id, "function_name": f"mcp__plugin_{SLUG}_docs__search", "arguments": {}}],
        "observation": {
            "results": [{"source_call_id": call_id, "content": "boom", "extra": {"tool_result_is_error": True}}]
        },
    }


def _collect(
    tmp_path: Path, native: tuple[dict[str, Any], Path], *, docs_failed_trials: int, docs_calls: int = 0
) -> dict[str, Any]:
    """Twelve scored with-plugin trials: kb fails to load in all, docs in ``docs_failed_trials`` of them."""
    plan, census = native
    job = tmp_path / "jobs" / f"{SLUG}-claude-code-with"
    names = [f"case-1__T{index:02d}abc" for index in range(12)]
    for index, name in enumerate(names):
        trial = job / name
        (trial / "agent").mkdir(parents=True)
        shutil.copyfile(census, trial / "agent" / LOAD_CENSUS_FILENAME)
        init = _init({"docs": "failed" if index < docs_failed_trials else "connected", "kb": "failed"})
        lines = [json.dumps(init), json.dumps({"type": "assistant", "message": {"content": []}})]
        _write(trial / "agent" / "claude-code.txt", "\n".join(lines) + "\n")
        steps = [_failed_docs_call(f"toolu_{call}") for call in range(docs_calls if index == 0 else 0)]
        trajectory = {"agent": {"name": "claude-code"}, "steps": [{"source": "user", "message": "Draft."}, *steps]}
        _write(trial / "agent" / "trajectory.json", json.dumps(trajectory))
        _write(trial / "verifier" / "reward.json", json.dumps(REWARD))
    stats = {"n_trials": 12, "n_errors": 0, "reward_stats": {"reward": {"0.65": names}}}
    _write(job / "result.json", json.dumps({"n_total_trials": 12, "stats": {**stats, "evals": {"a__m___t": stats}}}))
    return collect_harbor_results(
        skill_name=SLUG,
        agents=["claude-code"],
        output_dir=tmp_path / "out",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
        n_attempts=12,
        expected_cases=1,
        expected_case_ids=["case-1"],
        plugin_signals=build_plugin_signals_context(mcp_servers=["docs", "kb"], wrapper_skills=[SLUG]),
        plugin_load_census={"claude-code": plan},
    )


def test_one_flaky_load_keeps_a_reachable_server_out_of_not_loaded(
    tmp_path: Path, native_census: tuple[dict[str, Any], Path]
) -> None:
    # Verifier probe for M30: docs loaded in 11 of 12 trials (one init said "failed").
    engine = _collect(tmp_path, native_census, docs_failed_trials=1)
    assert engine["agents"]["claude-code"]["plugin_signals_summary"]["with_skill"]["mcp_load"] == {
        "docs": {"loaded": 11, "trials": 12},
        "kb": {"loaded": 0, "trials": 12},
    }
    proof = {"docs": _probe("reachable-host"), "kb": _probe("reachable-host")}

    final = apply_in_agent_mcp_proof(proof, engine)

    assert _statuses(final) == {"docs": "reachable-host", "kb": "not-loaded-in-agent"}
    assert final["docs"]["detail"].startswith("not loaded in 1 of 12 with-plugin trial(s) (claude-code: ")
    assert "host probe: " + HOST_OK in final["docs"]["detail"]
    assert final["kb"]["detail"].startswith("not loaded in the with-plugin arm (claude-code: ")


def test_one_flaky_load_with_failed_calls_is_called_no_success(
    tmp_path: Path, native_census: tuple[dict[str, Any], Path]
) -> None:
    engine = _collect(tmp_path, native_census, docs_failed_trials=1, docs_calls=2)

    final = apply_in_agent_mcp_proof({"docs": _probe("reachable-host")}, engine)

    assert final["docs"]["status"] == "called-no-success"
    detail = final["docs"]["detail"]
    assert detail.startswith("agent made 2 call(s): 0 succeeded, 2 failed, 0 unknown, in the with-plugin arm")
    assert "not loaded in 1 of 12 with-plugin trial(s)" in detail


def test_a_server_that_failed_to_load_in_every_trial_is_not_loaded(
    tmp_path: Path, native_census: tuple[dict[str, Any], Path]
) -> None:
    # check-21 e05 through the collector: init said "failed" in 12 of 12 trials.
    engine = _collect(tmp_path, native_census, docs_failed_trials=12, docs_calls=2)

    final = apply_in_agent_mcp_proof({"docs": _probe("reachable-host")}, engine)

    assert final["docs"]["status"] == "not-loaded-in-agent"
