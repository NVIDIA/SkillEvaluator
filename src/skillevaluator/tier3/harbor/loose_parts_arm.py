# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private loose-parts arm planning, emitted-task hashes, and cached load summaries.

The staging itself is :mod:`skillevaluator.tier3.loose_parts`. This module is
the one seam the runner and the collector call:

* :func:`plan_loose_parts` decides per agent whether the arm runs (native
  Claude Code only in v1) and builds its native task staging and Harbor
  wrapper, or records why it does not (``not_measured_reason``);
* :meth:`LoosePartsPlan.hash_tasks` hashes the Harbor tasks the runner
  emitted for the full-plugin, loose-parts and skills-only arms, so the
  capability check reads what the agents actually get;
* :meth:`LooseAgentArm.info` combines measured hashes with cached load
  evidence supplied by the comparison coordinator for durable provenance.

Nothing here assumes Docker: the loose arm uses the same native staging route
as the native full-plugin arm (the bundle is part of each task's
``environment/``), so it runs wherever that arm runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillevaluator.tier3.loose_parts import (
    LOOSE_AGENT,
    NOT_SUPPORTED_WRAPPER,
    LooseParts,
    build_loose_parts,
    load_failures_by_case,
    loose_parts_agent_import_path,
    not_supported_reason,
)
from skillevaluator.tier3.plugin_native import PluginLoadError

#: The loose arm's Harbor job-name suffix and staged task directory name.
LOOSE_PARTS_VARIANT = "looseparts"
#: Where the full per-trial load evidence is saved, next to the agent's statistics.json.
LOAD_EVIDENCE_FILENAME = "loose_parts_load_evidence.json"
#: Why the ladder cannot run at all without the no-plugin and skills-only arms beside it.
NEEDS_BASE_ARMS = (
    "the lift ladder needs the no-plugin and skills-only arms beside the loose-parts arm (a baseline and the "
    "plugin's member skills)"
)
LOAD_EVIDENCE_SOURCE = (
    "Claude Code's own system/init report; an MCP server init reported as pending (or not at all) counts as loaded "
    "only after a successful call of one of its tools in the trial's trajectory. Rules, hooks and scripts are not "
    "in that report (unverifiable, never a failure)."
)
# The summary's bounded copy of the failure records; the evidence file keeps every trial.
_MAX_FAILURE_CASES = 128
_MAX_FAILURE_RECORDS = 12


@dataclass
class LooseAgentArm:
    """One agent's measured loose-parts arm: the staged parts, the task staging, and the Harbor wrapper."""

    agent: str
    parts: LooseParts
    staging: Any  # NativeTaskStaging; the type lives in harbor.native_staging, imported lazily there
    import_path: str
    #: :meth:`LooseParts.content_report_for_tasks` of the emitted tasks, set by :meth:`LoosePartsPlan.hash_tasks`.
    content: dict[str, Any] = field(default_factory=dict)

    def info(
        self,
        evidence: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
        *,
        load_census: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The staging's ``loose_parts`` info block with the emitted tasks' hashes and a load-evidence summary."""
        content = (
            dict(self.content)
            if self.content
            else {
                "content_hashes": {},
                "equal_across_arms": None,
                "mismatches": [],
                "content_basis": "unavailable",
                "error": "Emitted task parity was not recorded",
            }
        )
        block = self.parts.info(content=content, load_evidence=load_evidence_summary(evidence))
        block["measured"] = True
        block["content_basis"] = content.get("content_basis")
        block["tasks_hashed"] = content.get("tasks_hashed")
        if content.get("error"):
            block["content_error"] = str(content["error"])[:500]
        if load_census is not None:
            block["load_census"] = dict(load_census)
        return block


@dataclass
class LoosePartsPlan:
    """The run's loose-parts arms: the measured agents, and why every other agent is not measured."""

    requested: bool
    order: tuple[str, ...] = ()
    arms: dict[str, LooseAgentArm] = field(default_factory=dict)
    not_measured: dict[str, str] = field(default_factory=dict)

    def arm_for(self, agent: str) -> LooseAgentArm | None:
        return self.arms.get(agent)

    def run_config(self) -> dict[str, Any]:
        """``run_config["loose_parts"]``: per agent, whether its loose arm runs and why not."""
        agents: dict[str, dict[str, Any]] = {}
        for agent in self.order:
            if agent in self.arms:
                agents[agent] = {"measured": True, "import_path": self.arms[agent].import_path}
            elif agent in self.not_measured:
                agents[agent] = {"measured": False, "not_measured_reason": self.not_measured[agent]}
        return {"requested": self.requested, "variant": LOOSE_PARTS_VARIANT, "agents": agents}

    def hash_tasks(self, agent: str, task_dirs_by_arm: Mapping[str, Sequence[Path]]) -> None:
        """Hash the tasks the runner emitted for ``agent`` (``{arm: [task dir, ...]}``) into its arm's ``content``."""
        arm = self.arms.get(agent)
        if arm is None:
            return
        try:
            report = arm.parts.content_report_for_tasks(task_dirs_by_arm)
        except (OSError, ValueError) as exc:
            # Report-only: an unreadable task leaves the capability unverified, never fails the run.
            report = {
                "content_hashes": {},
                "equal_across_arms": None,
                "mismatches": [],
                "tasks_hashed": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        arm.content = {**report, "content_basis": "emitted Harbor tasks"}


def plan_loose_parts(
    *,
    requested: bool,
    agents: Sequence[str],
    decisions: Mapping[str, Any],
    source: Any,
    agent_import_paths: Mapping[str, str],
    unavailable: str | None = None,
) -> LoosePartsPlan:
    """Decide per agent whether its loose-parts arm runs, and build its staging and Harbor wrapper.

    ``decisions`` are the run's private loading decisions (after any native
    fallback), ``source`` the native plugin snapshot (``None`` in wrapper
    mode), and ``agent_import_paths`` each agent's base Harbor import path.
    ``unavailable`` marks every agent not measured with that reason (the
    ladder cannot run in this run at all). Never raises for an agent: every
    refusal of the staging becomes that agent's ``not_measured_reason``.
    """
    plan = LoosePartsPlan(requested=requested, order=tuple(agents))
    if not requested:
        return plan
    for agent in agents:
        if unavailable:
            plan.not_measured[agent] = unavailable
            continue
        reason, arm = _agent_arm(agent, decisions.get(agent), source, agent_import_paths.get(agent))
        if arm is not None:
            plan.arms[agent] = arm
        else:
            plan.not_measured[agent] = reason or "not supported: the loose-parts arm could not be staged"
    return plan


def _agent_arm(
    agent: str, decision: Any, source: Any, base_import_path: str | None
) -> tuple[str | None, LooseAgentArm | None]:

    if agent != LOOSE_AGENT:
        return not_supported_reason(agent), None
    if source is None:
        return NOT_SUPPORTED_WRAPPER, None
    if decision is None or not getattr(decision, "native", False):
        return f"{NOT_SUPPORTED_WRAPPER} ({agent}'s plugin arm does not load the plugin natively in this run)", None
    try:
        parts = build_loose_parts(source, agent, plugin_load="native")
        if parts.error is not None:
            return parts.error, None
        staging = parts.task_staging()
        import_path = loose_parts_agent_import_path(agent, base_import_path)
    except PluginLoadError as exc:  # includes LoosePartsUnsupported ("not supported: ...")
        return str(exc), None
    except (ValueError, OSError) as exc:
        return f"loose parts cannot be staged: {exc}", None
    return None, LooseAgentArm(agent=agent, parts=parts, staging=staging, import_path=import_path)


def load_evidence_summary(evidence: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    """Per arm and piece, how many scored trials loaded it, and the cases dropped for a load failure.

    ``pieces[arm][piece]`` counts ``loaded``, ``not_loaded`` and
    ``unverifiable`` trials and lists the evidence ``sources`` seen
    (``system/init``, ``mcp-call``, ``unverifiable``).
    ``load_failures_by_case`` is :func:`load_failures_by_case` (bounded), the
    records the statistics drop cases by. The full per-trial evidence is saved
    in :data:`LOAD_EVIDENCE_FILENAME`.
    """
    failures = load_failures_by_case(evidence)
    pieces: dict[str, dict[str, dict[str, Any]]] = {}
    trials: dict[str, int] = {}
    for arm, cases in sorted(evidence.items()):
        rows = pieces.setdefault(arm, {})
        for attempts in cases.values():
            trials[arm] = trials.get(arm, 0) + len(attempts)
            for trial in attempts:
                for piece, row in trial.items():
                    counts = rows.setdefault(piece, {"loaded": 0, "not_loaded": 0, "unverifiable": 0, "sources": []})
                    loaded = row.get("loaded") if isinstance(row, Mapping) else None
                    counts["loaded" if loaded is True else "not_loaded" if loaded is False else "unverifiable"] += 1
                    source = str(row.get("source") or "") if isinstance(row, Mapping) else ""
                    if source and source not in counts["sources"]:
                        counts["sources"] = sorted([*counts["sources"], source])
    bounded = {case: records[:_MAX_FAILURE_RECORDS] for case, records in sorted(failures.items())[:_MAX_FAILURE_CASES]}
    return {
        "source": LOAD_EVIDENCE_SOURCE,
        "trials": trials,
        "pieces": pieces,
        "load_failures_by_case": bounded,
        "dropped_cases": sorted(failures),
        "file": LOAD_EVIDENCE_FILENAME,
    }
