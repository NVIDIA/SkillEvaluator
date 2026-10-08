# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime evidence folded into plugin coverage: hooks, subagents, and commands as ``exercised``."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from skillevaluator.plugin_components import summarize_coverage
from skillevaluator.tier3.eval_core.plugin_signals import (
    build_plugin_signals_context,
    compute_plugin_signals,
    summarize_plugin_signals,
)
from skillevaluator.tier3.harbor.adapter import PLUGIN_RUNTIME_COMPONENTS_FILENAME, load_plugin_runtime_components
from skillevaluator.tier3.plugin_eval import PLUGIN_RUNTIME_COMPONENTS_FILENAME as PLUGIN_EVAL_FILENAME
from skillevaluator.tier3.plugin_runtime import apply_runtime_coverage, apply_runtime_evidence


def _row(kind: str, name: str, state: str) -> dict[str, Any]:
    return {"type": kind, "name": name, "origin": "packaged", "path": None, "state": state, "reason": "base"}


def _provenance(rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"component_coverage": summarize_coverage(deepcopy(rows)), **extra}


def _engine(*, exercised=(), unavailable=(), hooks=(), by_server=None) -> dict[str, Any]:
    summary = {
        "activation_coverage": {
            "declared": list(exercised),
            "exercised": list(exercised),
            "unavailable": list(unavailable),
        },
        "hook_census": {"hooks": list(hooks), "total_runs": sum(hook["runs"] for hook in hooks)},
        "mcp_calls": {"by_server": by_server or {}},
    }
    return {"agents": {"claude-code": {"plugin_signals_summary": {"with_skill": summary}}}}


def _states(provenance: dict[str, Any]) -> dict[str, str]:
    return {row["name"]: row["state"] for row in provenance["component_coverage"]["components"]}


def _plugin_load(components: dict[str, str], agent: str = "claude-code") -> dict[str, Any]:
    return {"by_agent": {agent: {"mode": "native", "components": components}}}


def _staged_hooks(ids: dict[str, list[str]], agent: str = "claude-code") -> dict[str, Any]:
    """The load-census summary's exact staged hook ids (what native staging wrapped)."""
    return {agent: {"mode": "native", "staged_hook_ids": ids}}


_CENSUS_RUN = {"hook_id": "hooks/hooks.json#PreToolUse[0].hooks[0]", "event": "PreToolUse", "runs": 1, "failures": 0}


def test_hooks_that_ran_become_exercised_from_the_census() -> None:
    # Natively staged hooks are the ones wrapped with hook_census.sh.
    provenance = _provenance(
        [
            _row("hook", "hooks/hooks.json", "unsupported"),
            _row("hook", "inline", "loaded"),
            _row("hook", "bad", "invalid"),
        ],
        plugin_load=_plugin_load({"hook": "native"}),
        load_census=_staged_hooks(
            {"hooks/hooks.json": ["hooks/hooks.json#PreToolUse[0].hooks[0]"], "inline": ["inline#Stop[0].hooks[0]"]}
        ),
    )
    engine = _engine(
        hooks=[
            {"hook_id": "hooks/hooks.json#PreToolUse[0].hooks[0]", "event": "PreToolUse", "runs": 2, "failures": 0},
            {"hook_id": "bad", "event": "Stop", "runs": 1, "failures": 0},
            {"hook_id": "inline-other", "event": "Stop", "runs": 1, "failures": 0},
        ]
    )

    assert apply_runtime_coverage(provenance, engine) == 1
    assert _states(provenance) == {"hooks/hooks.json": "exercised", "inline": "loaded", "bad": "invalid"}
    row = provenance["component_coverage"]["components"][0]
    assert "hook census recorded 2 started run(s)" in row["reason"]
    assert "self-reported" in row["reason"]
    assert provenance["component_coverage"]["counts"]["exercised"] == 1
    # exercised and loaded count as evaluated; unsupported/invalid do not.
    assert provenance["component_coverage"]["not_evaluated"] == 1


def test_a_forged_hook_census_cannot_promote_an_unstaged_hook() -> None:
    # --plugin-load wrapper never stages or wraps hooks, so any census line was
    # written by code inside the sandbox (the census file is writable there).
    rows = [_row("hook", "hooks/hooks.json", "unsupported")]
    engine = _engine(hooks=[_CENSUS_RUN])
    for plugin_load in (
        None,
        {"by_agent": {"claude-code": {"mode": "wrapper", "components": {"hook": "wrapper"}}}},
        _plugin_load({"hook": "unsupported"}),
        _plugin_load({"hook": "native"}, agent="codex"),  # native for another agent only
    ):
        provenance = _provenance(rows, **({"plugin_load": plugin_load} if plugin_load else {}))
        before = deepcopy(provenance)

        assert apply_runtime_coverage(provenance, engine) == 0
        assert provenance == before
        assert _states(provenance) == {"hooks/hooks.json": "unsupported"}


def test_only_natively_staged_hooks_are_exercised_from_the_census() -> None:
    engine = _engine(hooks=[_CENSUS_RUN])
    staged_ids = _staged_hooks({"hooks/hooks.json": [_CENSUS_RUN["hook_id"]]})
    # A staged or loaded row alone is not enough: only native staging wraps hooks.
    for state in ("staged", "loaded"):
        provenance = _provenance([_row("hook", "hooks/hooks.json", state)], load_census=staged_ids)
        assert apply_runtime_coverage(provenance, engine) == 0

    native = _provenance(
        [_row("hook", "hooks/hooks.json", "unsupported")],
        plugin_load=_plugin_load({"hook": "native"}),
        load_census=staged_ids,
    )
    assert apply_runtime_coverage(native, engine) == 1
    assert _states(native) == {"hooks/hooks.json": "exercised"}


def test_subagents_and_commands_need_availability_to_be_exercised() -> None:
    rows = [
        _row("agent", "reviewer", "staged"),
        _row("agent", "planner", "unsupported"),
        _row("command", "deploy", "loaded"),
        _row("command", "flaky", "staged"),
        _row("skill", "alpha", "staged"),
        _row("mcp", "github", "staged"),
    ]
    engine = _engine(
        exercised=[
            "subagent:reviewer",
            "subagent:planner",
            "command:deploy",
            "command:flaky",
            "skill:alpha",
            "mcp:github",
        ],
        unavailable=["command:flaky"],
    )

    provenance = _provenance(rows)
    apply_runtime_coverage(provenance, engine)

    assert _states(provenance) == {
        "reviewer": "exercised",
        "planner": "unsupported",
        "deploy": "exercised",
        "flaky": "staged",
        "alpha": "exercised",
        "github": "exercised",
    }

    native = _provenance(rows, plugin_load=_plugin_load({"agent": "native"}))
    apply_runtime_coverage(native, engine)
    assert _states(native)["planner"] == "exercised"


def test_no_evidence_leaves_provenance_unchanged() -> None:
    provenance = _provenance([_row("hook", "hooks/hooks.json", "unsupported")])
    before = deepcopy(provenance)

    assert apply_runtime_coverage(provenance, None) == 0
    assert apply_runtime_coverage(provenance, _engine()) == 0
    assert provenance == before


def test_apply_runtime_evidence_upgrades_mcp_proof() -> None:
    provenance = _provenance(
        [_row("mcp", "docs", "staged")],
        mcp_proof={"docs": {"status": "declared", "tools": [], "detail": "not probed"}},
    )

    apply_runtime_evidence(
        provenance, _engine(exercised=["mcp:docs"], by_server={"docs": {"total": 2, "succeeded": 1}})
    )

    assert provenance["mcp_proof"]["docs"]["status"] == "used-successfully"
    assert _states(provenance) == {"docs": "exercised"}


def _call(step: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_id": step,
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": f"c{step}", "function_name": name, "arguments": args}],
        "observation": {"results": [{"source_call_id": f"c{step}", "content": "ok"}]},
    }


def test_declared_subagents_and_commands_count_in_with_plugin_activation_coverage() -> None:
    context = build_plugin_signals_context(
        member_skills=["alpha"], subagents=["reviewer", "planner"], commands=["/deploy"]
    )
    trajectory = {
        "steps": [
            _call(1, "Task", {"subagent_type": "demo-plugin:reviewer", "prompt": "review"}),
            _call(2, "SlashCommand", {"command": "/demo-plugin:deploy --dry-run"}),
        ]
    }

    with_plugin = compute_plugin_signals(trajectory, declared=context.declared_for("with_skill"))
    members = compute_plugin_signals(trajectory, declared=context.declared_for("sum_of_parts"))

    assert with_plugin is not None and members is not None
    assert with_plugin["activation_coverage"] == {
        "declared": ["skill:alpha", "subagent:reviewer", "subagent:planner", "command:deploy"],
        "exercised": ["subagent:reviewer", "command:deploy"],
        "unverified": ["skill:alpha", "subagent:planner"],
        "unavailable": [],
    }
    assert members["activation_coverage"]["declared"] == ["skill:alpha"]


def test_runtime_components_file_round_trips(tmp_path: Path) -> None:
    assert PLUGIN_RUNTIME_COMPONENTS_FILENAME == PLUGIN_EVAL_FILENAME
    skill = tmp_path / "pkg"
    environment = skill / "evals" / "environment"
    environment.mkdir(parents=True)
    assert load_plugin_runtime_components(skill) == {"subagents": [], "commands": []}
    (environment / PLUGIN_RUNTIME_COMPONENTS_FILENAME).write_text(
        '{"subagents": ["reviewer", 3], "commands": "not-a-list"}', encoding="utf-8"
    )
    assert load_plugin_runtime_components(skill) == {"subagents": ["reviewer"], "commands": []}
    (environment / PLUGIN_RUNTIME_COMPONENTS_FILENAME).write_text("not json", encoding="utf-8")
    assert load_plugin_runtime_components(skill) == {"subagents": [], "commands": []}


def _agent_plugin(root: Path) -> Path:
    files = {
        ".claude-plugin/plugin.json": json.dumps({"name": "Demo Kit", "description": "Builds and reviews."}),
        "skills/alpha/SKILL.md": "---\nname: alpha\ndescription: Alpha skill.\n---\nDo alpha.\n",
        "agents/build.md": "---\nname: build\ndescription: Builds the change\n---\nYou build.\n",
        "agents/reviewer.md": "---\nname: reviewer\ndescription: Reviews\n---\nYou review.\n",
        "evals/evals.json": json.dumps(
            {"skill_name": "demo", "evals": [{"id": "c1", "prompt": "Build it.", "expected_output": "Built."}]}
        ),
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def test_an_opencode_agent_staged_under_a_new_name_still_counts_as_the_declared_agent(tmp_path: Path) -> None:
    # OpenCode stages a plugin agent named like a built-in as <plugin>-<name>, so its
    # trajectory calls "demo-kit-build", never "build".
    from skillevaluator.tier3.harbor.adapter import load_plugin_subagent_aliases
    from skillevaluator.tier3.harbor.native_staging import build_native_task_staging
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
    from skillevaluator.tier3.plugin_native import HARNESS_ADAPTERS

    package = prepare_plugin_eval_package(
        _agent_plugin(tmp_path / "demo-kit"), stage_root=tmp_path / "stage", plugin_load="native"
    )
    assert package.native_source is not None
    staging = build_native_task_staging("opencode", HARNESS_ADAPTERS["opencode"], package.native_source)
    staged_agents = sorted(rel.rsplit("/", 1)[-1] for rel in staging.bundle.generated if "/config/agents/" in rel)
    assert staged_agents == ["demo-kit-build.md", "reviewer.md"]

    aliases = load_plugin_subagent_aliases(package.package_path)
    assert aliases == {"demo-kit-build": "build"}
    runtime = load_plugin_runtime_components(package.package_path)
    context = build_plugin_signals_context(
        member_skills=["alpha"], subagents=runtime["subagents"], subagent_aliases=aliases
    )
    trajectory = {
        "agent": {"name": "opencode"},
        "steps": [_call(1, "task", {"subagent_type": "demo-kit-build", "description": "build", "prompt": "go"})],
    }

    signals = compute_plugin_signals(
        trajectory, declared=context.declared_for("with_skill"), subagent_aliases=context.aliases_for("with_skill")
    )

    assert signals is not None
    coverage = signals["activation_coverage"]
    assert "subagent:build" in coverage["exercised"]
    assert "subagent:reviewer" in coverage["unverified"]
    # Aliases only ever map to a declared subagent, and only in the with-plugin arm.
    assert context.aliases_for("sum_of_parts") == {}
    stray = build_plugin_signals_context(subagents=["reviewer"], subagent_aliases={"x-build": "build"})
    assert stray.subagent_aliases == {}


def test_a_bare_builtin_agent_name_never_counts_as_the_plugin_agent_of_that_name() -> None:
    # OpenCode stages the plugin's "build" as "demo-kit-build" and Claude Code names its
    # "plan" "demo-kit:plan", so a bare "build" or "Plan" reaches the harness's own agent.
    context = build_plugin_signals_context(
        subagents=["build", "plan"], subagent_aliases={"demo-kit-build": "build"}, plugin_name="demo-kit"
    )

    def _signals(agent: str, tool: str, subagent: str, ref: str) -> dict[str, Any]:
        trajectory = {"agent": {"name": agent}, "steps": [_call(1, tool, {"subagent_type": subagent, "prompt": "go"})]}
        signals = compute_plugin_signals(
            trajectory,
            {"expected_tools": [ref]},
            declared=context.declared_for("with_skill"),
            subagent_aliases=context.aliases_for("with_skill"),
        )
        assert signals is not None
        return signals

    for agent, tool, subagent, ref in (
        ("opencode", "task", "build", "Agent:build"),
        ("opencode", "task", "plan", "Agent:plan"),
        ("claude-code", "Task", "Plan", "Agent:plan"),
    ):
        builtin = _signals(agent, tool, subagent, ref)
        assert builtin["activation_coverage"]["exercised"] == [], (agent, subagent)
        assert builtin["routing"]["recall"] == 0.0, (agent, subagent)
        assert builtin["activations"][0]["name"] == subagent
    for agent, tool, subagent, ref, declared in (
        ("opencode", "task", "demo-kit-build", "Agent:build", "subagent:build"),
        ("claude-code", "Task", "demo-kit:plan", "Agent:plan", "subagent:plan"),
        # Claude Code has no built-in "build", so the bare name is still the plugin's agent.
        ("claude-code", "Task", "build", "Agent:build", "subagent:build"),
    ):
        plugin_agent = _signals(agent, tool, subagent, ref)
        assert plugin_agent["activation_coverage"]["exercised"] == [declared], (agent, subagent)
        assert plugin_agent["routing"]["recall"] == 1.0, (agent, subagent)
    # A ref to a built-in the plugin does not declare still names it.
    assert _signals("claude-code", "Agent", "general-purpose", "Agent:general-purpose")["routing"]["recall"] == 1.0


def _failed_call(step: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
    call = _call(step, name, args)
    call["observation"]["results"][0].update(content="Agent type not found", is_error=True)
    return call


def test_a_claude_code_agent_call_that_names_the_subagent_by_type_exercises_it() -> None:
    # Claude Code 2.1.29x's Agent tool takes the subagent name as "type", not "subagent_type".
    context = build_plugin_signals_context(subagents=["lab-reviewer", "explore"], plugin_name="component-lab")
    args = {"type": "component-lab:lab-reviewer", "prompt": "Review the sentence for clarity."}

    def _summary(*steps: dict[str, Any]) -> dict[str, Any]:
        trajectory = {"agent": {"name": "claude-code"}, "steps": list(steps)}
        signals = compute_plugin_signals(
            trajectory, {"expected_tools": ["Agent:lab-reviewer"]}, declared=context.declared_for("with_skill")
        )
        assert signals is not None
        return signals

    signals = _summary(_call(1, "Agent", args))
    assert signals["activation_coverage"]["exercised"] == ["subagent:lab-reviewer"]
    assert signals["routing"]["recall"] == 1.0
    engine = {
        "agents": {"claude-code": {"plugin_signals_summary": {"with_skill": summarize_plugin_signals([signals])}}}
    }
    provenance = _provenance([_row("agent", "lab-reviewer", "loaded")])
    assert apply_runtime_coverage(provenance, engine) == 1
    assert _states(provenance) == {"lab-reviewer": "exercised"}

    # A failed call is not credited, the existing keys still come first, and a
    # bare built-in name still reaches Claude Code's own agent.
    failed = _summary(_failed_call(1, "Agent", args))["activation_coverage"]
    assert failed["exercised"] == [] and failed["unavailable"] == ["subagent:lab-reviewer"]
    first = _summary(_call(1, "Task", {"subagent_type": "lab-reviewer", "type": "general-purpose"}))
    assert first["activation_coverage"]["exercised"] == ["subagent:lab-reviewer"]
    builtin = _summary(_call(1, "Agent", {"type": "Explore", "prompt": "Find the notes."}))
    assert builtin["activation_coverage"]["exercised"] == []
