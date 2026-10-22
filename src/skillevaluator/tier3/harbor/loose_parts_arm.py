# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run plumbing for the loose-parts arm (``--lift-mode ladder``): who runs it, its hashes, and its load evidence.

The staging itself is :mod:`skillevaluator.tier3.loose_parts`. This module is
the one seam the runner and the collector call:

* :func:`plan_loose_parts` decides per agent whether the arm runs (native
  Claude Code only in v1) and builds its native task staging and Harbor
  wrapper, or records why it does not (``not_measured_reason``);
* :meth:`LoosePartsPlan.hash_tasks` hashes the Harbor tasks the runner
  emitted for the full-plugin, loose-parts and skills-only arms, so the
  capability check reads what the agents actually get;
* :meth:`LooseAgentArm.load_evidence` reads Claude Code's own ``system/init``
  report (and, for a server init reported as pending, a successful MCP call)
  for every scored trial of those three arms;
* :meth:`LoosePartsPlan.record` builds the run's ``loose_parts`` block, which
  the plugin provenance carries to the report.

Nothing here assumes Docker: the loose arm uses the same native staging route
as the native full-plugin arm (the bundle is part of each task's
``environment/``), so it runs wherever that arm runs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.stats import ARM_LOOSE_PARTS
from skillevaluator.tier3.loose_parts import (
    LOOSE_AGENT,
    LOOSE_PARTS_LABEL,
    NOT_SUPPORTED_WRAPPER,
    LooseParts,
    build_loose_parts,
    ladder_skip_reason,
    load_failures_by_case,
    loose_parts_agent_import_path,
    loose_parts_load_evidence,
    not_supported_reason,
)
from skillevaluator.utils.redaction import redact_sensitive_text

logger = logging.getLogger(__name__)

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
_MAX_EVIDENCE_TEXT = 500

TrialRef = tuple[str, str, Path]
"""``(arm, case id, trial directory)`` of one scored trial, in attempt order per case."""


@dataclass
class LooseAgentArm:
    """One agent's measured loose-parts arm: the staged parts, the task staging, and the Harbor wrapper."""

    agent: str
    parts: LooseParts
    staging: Any  # NativeTaskStaging; the type lives in harbor.native_staging, imported lazily there
    import_path: str
    #: :meth:`LooseParts.content_report_for_tasks` of the emitted tasks, set by :meth:`LoosePartsPlan.hash_tasks`.
    content: dict[str, Any] = field(default_factory=dict)

    def census_plan(self) -> dict[str, Any]:
        """The collector's load-census plan for the loose arm (its own harness kind, never the plugin's)."""
        return self.parts.census_plan()

    def load_evidence(
        self,
        trials: Iterable[TrialRef],
        *,
        read_trajectory: Callable[[Path], Mapping[str, Any] | None],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """``{arm: {case id: [evidence per attempt]}}`` for the given scored trials.

        Each trial's evidence is :func:`loose_parts_load_evidence` over Claude
        Code's ``system/init`` event (from ``agent/claude-code.txt``) and the
        trial's trajectory. A trial with no init event has every init-verifiable
        piece ``loaded: False`` (the agent did not start as staged).
        """
        evidence: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for arm, case_id, trial_root in trials:
            init = _trial_init_event(trial_root)
            try:
                trajectory = read_trajectory(trial_root)
            except Exception:  # evidence is advisory; an unreadable trajectory only loses the MCP-call fallback
                logger.debug("Loose-parts evidence: trajectory of %s unreadable", trial_root.name, exc_info=True)
                trajectory = None
            row = loose_parts_load_evidence([init] if init else [], trajectory, self.parts, arm=arm)
            evidence.setdefault(arm, {}).setdefault(str(case_id), []).append(_redacted_evidence(row))
        return evidence

    def info(
        self,
        evidence: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
        *,
        load_census: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The staging's ``loose_parts`` info block with the emitted tasks' hashes and a load-evidence summary."""
        content = dict(self.content) if self.content else {**self.parts.content_report(), "content_basis": "staging"}
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

    @property
    def measured(self) -> bool:
        return bool(self.arms)

    @property
    def skip_reason(self) -> str | None:
        """Why no agent runs the arm (``None`` when one does, or when the ladder was not requested)."""
        if not self.requested or self.arms:
            return None
        return ladder_skip_reason(self.not_measured)

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

    def record(self, infos: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """The run's ``loose_parts`` block.

        The first measured agent's info block (``infos[agent]``, from
        :meth:`LooseAgentArm.info`) is the run-level record; ``agents`` holds
        every agent: ``{"measured": True}`` (plus its own info for a second
        measured agent) or ``{"measured": False, "not_measured_reason": ...}``.
        The report merges the two per agent and never gives a refused agent
        the measured agent's parts.
        """
        block: dict[str, Any] = {
            "arm": ARM_LOOSE_PARTS,
            "label": LOOSE_PARTS_LABEL,
            "requested": self.requested,
            "skip_reason": self.skip_reason,
        }
        agents: dict[str, dict[str, Any]] = {}
        primary: str | None = None
        for agent in self.order:
            arm = self.arms.get(agent)
            if arm is None:
                if agent in self.not_measured:
                    agents[agent] = {"measured": False, "not_measured_reason": self.not_measured[agent]}
                continue
            info = dict(infos.get(agent) or arm.info({}))
            if primary is None:
                primary = agent
                block.update(info)
                agents[agent] = {"measured": True}
            else:
                agents[agent] = {**info, "measured": True}
        block["agents"] = agents
        return block


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

    ``decisions`` are the run's ``--plugin-load`` decisions (after any native
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
    from skillevaluator.tier3.plugin_native import PluginLoadError

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


def _trial_init_event(trial_root: Path) -> dict[str, Any] | None:
    """Claude Code's first ``system/init`` event from the trial's ``agent/claude-code.txt``, or ``None``."""
    from skillevaluator.tier3.plugin_native import CLAUDE_CODE_LOG_FILENAME, claude_init_event, read_harness_log_prefix

    try:
        text = read_harness_log_prefix(trial_root / "agent" / CLAUDE_CODE_LOG_FILENAME)
    except Exception:  # a log that cannot be inspected safely gives no evidence
        return None
    return claude_init_event(text) if text else None


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


def write_load_evidence(path: Path, evidence: Mapping[str, Any]) -> bool:
    """Save the full per-trial load evidence (``{arm: {case id: [evidence per attempt]}}``) atomically.

    Report-only: a write that fails is logged and returns ``False``; it never
    fails the result collection.
    """
    from skillevaluator.tier3.output_provenance import write_output_file_atomically

    try:
        write_output_file_atomically(path, json.dumps(evidence, indent=2, sort_keys=True).encode("utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Loose-parts load evidence was not saved to %s: %s", path.name, exc)
        return False
    return True


def _redacted_evidence(row: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """One trial's evidence with every text that came from the agent's log redacted and bounded.

    Only the values are redacted (``evidence``, ``status``, ``tools``); the piece
    ids and ``loaded`` flags are the staging's own, so a failure is never hidden.
    """
    cleaned: dict[str, dict[str, Any]] = {}
    for piece, entry in row.items():
        item = dict(entry)
        for key in ("evidence", "status"):
            if isinstance(item.get(key), str):
                item[key] = redact_sensitive_text(item[key], max_len=_MAX_EVIDENCE_TEXT)
        if isinstance(item.get("tools"), list):
            item["tools"] = [redact_sensitive_text(str(tool), max_len=200) for tool in item["tools"][:64]]
        cleaned[piece] = item
    return cleaned
