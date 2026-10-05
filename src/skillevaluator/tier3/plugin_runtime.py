# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fold runtime evidence from a finished plugin run into plugin provenance.

After the Harbor run, :func:`apply_runtime_evidence` promotes C2
``component_coverage`` rows to ``exercised`` when the with-plugin arm shows the
component ran, and upgrades ``mcp_proof`` from in-agent MCP calls:

* **hooks**: the hook census (``plugin_signals_summary.with_skill.hook_census``)
  recorded at least one run of a handler that SkillEvaluator wrapped. Census
  ids use the static hook-risk id format
  ``<source>#<event>[<group>].hooks[<handler>]`` (for example
  ``hooks/hooks.json#PreToolUse[0].hooks[0]``), and a hook's coverage row name
  is its ``<source>``. Only the exact ids staged for that source count; the
  staging plan records them in ``load_census.<agent>.staged_hook_ids``. A hook
  counts only when the agent loads hooks natively (``plugin_load``), because
  only native staging wraps hooks with ``hook_census.sh``. In
  ``--plugin-load wrapper`` mode hooks are inventoried but never staged, so a
  census line there can only have been written by code in the sandbox. Runs
  that could not start the hook command (exit 126/127) do not count. The
  census file is writable from inside the sandbox, so the evidence is labeled
  self-reported.
* **subagents and commands**: ``activation_coverage`` exercised a declared
  ``subagent:<name>`` or ``command:<name>`` and not every activation failed.
  Name matching is heuristic.
* **skills and MCP servers**: a staged or loaded member skill or runnable MCP
  server that ``activation_coverage`` exercised.

Subagents, commands, skills, and MCP servers are promoted only when the
component was actually available to the agent: its row is ``staged`` or
``loaded``, or ``plugin_load`` reports the type as natively loaded for that
agent.

Precedence is ``exercised`` > ``loaded`` > ``staged``; a row is never
downgraded and an ``invalid`` row never changes. Everything here is
report-only: coverage never changes the ``INCOMPLETE`` rule, a score, or a
verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from skillevaluator.plugin_components import EVALUATED_COVERAGE_STATES, summarize_coverage
from skillevaluator.tier3.mcp_proof import apply_in_agent_mcp_proof
from skillevaluator.tier3.plugin_native import native_types_by_agent

STATE_EXERCISED = "exercised"
_ACTIVATION_PREFIX = {"skill": "skill", "mcp": "mcp", "agent": "subagent", "command": "command"}


def _with_plugin_summaries(engine_result: Mapping[str, Any] | None) -> list[tuple[str, Mapping[str, Any]]]:
    agents = engine_result.get("agents") if isinstance(engine_result, Mapping) else None
    summaries: list[tuple[str, Mapping[str, Any]]] = []
    if not isinstance(agents, Mapping):
        return summaries
    for name, agent in agents.items():
        arms = agent.get("plugin_signals_summary") if isinstance(agent, Mapping) else None
        arm = arms.get("with_skill") if isinstance(arms, Mapping) else None
        if isinstance(arm, Mapping):
            summaries.append((str(name), arm))
    return summaries


def _staged_hook_ids(provenance: Mapping[str, Any], agent: str, source: str) -> frozenset[str]:
    """The exact hook ids SkillEvaluator wrapped for ``source`` in ``agent``'s with-plugin arm."""
    load_census = provenance.get("load_census")
    census = load_census.get(agent) if isinstance(load_census, Mapping) else None
    staged = census.get("staged_hook_ids") if isinstance(census, Mapping) else None
    ids = staged.get(source) if isinstance(staged, Mapping) else None
    if not isinstance(ids, list | tuple):
        return frozenset()
    return frozenset(identifier for identifier in ids if isinstance(identifier, str) and identifier)


def _hook_census_evidence(
    name: str, agent: str, summary: Mapping[str, Any], provenance: Mapping[str, Any], native_types: set[str]
) -> str | None:
    """Self-reported hook census evidence for one natively staged hook source, or ``None``.

    *native_types* are the component types ``agent`` loads natively.
    """
    if "hook" not in native_types:
        return None
    allowed = _staged_hook_ids(provenance, agent, name)
    if not allowed:
        return None
    census = summary.get("hook_census")
    hooks = census.get("hooks") if isinstance(census, Mapping) else None
    runs = not_started = 0
    for hook in hooks or ():
        if not isinstance(hook, Mapping) or str(hook.get("hook_id") or "") not in allowed:
            continue
        hook_runs = hook.get("runs")
        if not isinstance(hook_runs, int) or isinstance(hook_runs, bool) or hook_runs <= 0:
            continue
        runs += hook_runs
        skipped = hook.get("not_started")
        if isinstance(skipped, int) and not isinstance(skipped, bool) and skipped > 0:
            not_started += min(skipped, hook_runs)
    started = runs - not_started
    if started <= 0:
        return None
    return (
        f"hook census recorded {started} started run(s) in the {agent} with-plugin arm "
        "(self-reported by the hook wrapper; the census file is writable from the sandbox)"
    )


def _exercise_evidence(
    row: Mapping[str, Any],
    summaries: list[tuple[str, Mapping[str, Any]]],
    provenance: Mapping[str, Any],
    native_types: Mapping[str, set[str]],
) -> str | None:
    kind = str(row.get("type") or "")
    name = str(row.get("name") or "")
    state = str(row.get("state") or "")
    if not name or state == "invalid":
        return None
    for agent, summary in summaries:
        agent_native_types = native_types.get(agent, set())
        if kind == "hook":
            # Only natively staged hooks are wrapped, and only their exact
            # staged ids count: a census line is writable from the sandbox.
            if evidence := _hook_census_evidence(name, agent, summary, provenance, agent_native_types):
                return evidence
            continue
        # Available to this agent: staged or loaded, or its type natively loaded.
        # Unavailable components cannot have run, so an activation label alone
        # does not count.
        if state not in EVALUATED_COVERAGE_STATES and kind not in agent_native_types:
            continue
        prefix = _ACTIVATION_PREFIX.get(kind)
        if prefix is None:
            continue
        coverage = summary.get("activation_coverage")
        if not isinstance(coverage, Mapping):
            continue
        label = f"{prefix}:{name}"
        exercised = coverage.get("exercised") or ()
        unavailable = coverage.get("unavailable") or ()
        if label in exercised and label not in unavailable:
            return f"activated in the {agent} with-plugin arm"
    return None


def apply_runtime_coverage(provenance: dict[str, Any], engine_result: Mapping[str, Any] | None) -> int:
    """Promote coverage rows with runtime evidence to ``exercised``; return how many changed."""
    coverage = provenance.get("component_coverage")
    rows = coverage.get("components") if isinstance(coverage, Mapping) else None
    if not isinstance(rows, list):
        return 0
    summaries = _with_plugin_summaries(engine_result)
    if not summaries:
        return 0
    native_types = native_types_by_agent(provenance.get("plugin_load"))
    promoted = 0
    updated_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw) if isinstance(raw, Mapping) else raw
        if isinstance(row, dict) and row.get("state") != STATE_EXERCISED:
            evidence = _exercise_evidence(row, summaries, provenance, native_types)
            if evidence is not None:
                reason = str(row.get("reason") or "")
                row["state"] = STATE_EXERCISED
                row["reason"] = f"{reason}; runtime evidence: {evidence}" if reason else f"runtime evidence: {evidence}"
                promoted += 1
        updated_rows.append(row)
    if promoted:
        provenance["component_coverage"] = summarize_coverage(updated_rows)
    return promoted


def apply_runtime_evidence(provenance: dict[str, Any], engine_result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fold coverage and in-agent MCP proof into ``provenance`` in place and return it."""
    apply_runtime_coverage(provenance, engine_result)
    proof = provenance.get("mcp_proof")
    if isinstance(proof, Mapping) and proof:
        provenance["mcp_proof"] = apply_in_agent_mcp_proof(proof, engine_result)
    return provenance


__all__ = [
    "STATE_EXERCISED",
    "apply_runtime_coverage",
    "apply_runtime_evidence",
]
