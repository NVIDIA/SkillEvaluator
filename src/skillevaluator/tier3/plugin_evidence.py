# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private advisory evidence collected before the existing runtime releases its artifacts."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.tier3.eval_core.plugin_signals import (
    MCP_INPUT_SCHEMAS_KEY,
    PLUGIN_PART_ARMS,
    build_plugin_signals_context,
    compute_plugin_signals,
    summarize_plugin_signals,
    top_argument_failures,
    validate_plugin_case_fields,
)
from skillevaluator.tier3.harbor import collector
from skillevaluator.tier3.harbor.metrics import finite_number
from skillevaluator.tier3.mcp_proof import (
    MCP_LOAD_KEY,
    TOTAL_TIMEOUT_S,
    apply_in_agent_mcp_proof,
    declared_mcp_proof,
    mcp_input_schemas,
    parse_env_grant,
    probe_mcp_servers,
)
from skillevaluator.tier3.plugin_prepare import verify_source_identity
from skillevaluator.tier3.plugin_trace_artifacts import TrialTraceArtifacts
from skillevaluator.tier3.runtime_arms import ARM_LOOSE_PARTS, ARM_SUM_OF_PARTS, ARM_WITH, ARM_WITHOUT
from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.rich_markup import strip_terminal_controls
from skillevaluator.utils.secure_fs import SecureRoot, stat_is_link_or_reparse

if TYPE_CHECKING:
    from skillevaluator.tier3.plugin_execution import NativeRuntimeHooks
    from skillevaluator.tier3.plugin_prepare import PreparedRuntimePackage, WrapperSkill

RUNTIME_EVIDENCE_FILENAME = "plugin_runtime_evidence.json"
MAX_TRIALS_PER_ARM = 512
MAX_TRIAL_ENTRIES = 4096
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
ARM_SUFFIXES = {
    ARM_WITH: "with",
    ARM_WITHOUT: "without",
    ARM_SUM_OF_PARTS: "sumofparts",
    ARM_LOOSE_PARTS: "looseparts",
}


def _safe_text(value: Any, limit: int = 300) -> str:
    return strip_terminal_controls(
        redact_sensitive_text(str(value or ""), max_len=limit, max_input_chars=limit * 4 + 256)
    )


@dataclass(frozen=True)
class EvidenceOptions:
    """Private opt-ins; ordinary calls perform no host probe or environment expansion."""

    probe_mcp: bool = False
    env_grants: tuple[str, ...] = ()
    allowed_private_hosts: tuple[str, ...] = ()
    timeout: float = TOTAL_TIMEOUT_S

    def __post_init__(self) -> None:
        if type(self.probe_mcp) is not bool:
            raise ValueError("probe_mcp must be a boolean")
        if finite_number(self.timeout) is None or not 0 < self.timeout <= TOTAL_TIMEOUT_S:
            raise ValueError(f"probe timeout must be positive and at most {TOTAL_TIMEOUT_S:g}s")
        if len(self.env_grants) > 64 or len(self.allowed_private_hosts) > 64:
            raise ValueError("private evidence policy exceeds its collection bound")
        for grant in self.env_grants:
            parse_env_grant(grant)


@dataclass(frozen=True)
class LogicalTrialEvidence:
    """Cached logical observations for later private comparisons; raw rewards stay in memory."""

    trial_id: str
    case_id: str | None
    reward: Mapping[str, Any] | None
    signals: Mapping[str, Any] | None
    native_census: Mapping[str, Any] | None
    usage: Mapping[str, float | None] | None
    first_turn_usage: Mapping[str, float | None] | None
    evidence_status: str
    init_event: Mapping[str, Any] | None = None
    first_turn_excluded_reason: str | None = None

    def persisted(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "case_id": self.case_id,
            "signals": self.signals,
            "evidence_status": self.evidence_status,
        }


def _usage(raw: Any) -> dict[str, float | None] | None:
    if not isinstance(raw, Mapping):
        return None
    fields = (
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cost_usd",
    )
    kept = {key: finite_number(raw[key], non_negative=True) for key in fields if key in raw}
    return kept or None


def _first_usage(trajectory: Mapping[str, Any] | None) -> tuple[Mapping[str, float | None] | None, str | None]:
    steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else None
    for step in steps[:10_000] if isinstance(steps, list) else ():
        if isinstance(step, Mapping) and step.get("source") == "agent" and step.get("llm_call_count") != 0:
            calls = step.get("tool_calls")
            if isinstance(calls, list) and any(
                isinstance(call, Mapping) and call.get("function_name") == "web_search_call" for call in calls[:4000]
            ):
                return None, "hosted_search"
            usage = _usage(step.get("metrics"))
            return usage, None if usage and usage.get("prompt_tokens") is not None else "missing_tokens"
    return None, "missing_tokens"


def _trial_roots(job: Path) -> tuple[list[Path], bool]:
    """Bound directory enumeration and refuse links before opening any trial artifacts."""
    with SecureRoot(job):
        with os.scandir(job) as entries:
            found = list(islice(entries, MAX_TRIAL_ENTRIES + 1))
        limited = len(found) > MAX_TRIAL_ENTRIES
        roots = []
        for entry in sorted(found[:MAX_TRIAL_ENTRIES], key=lambda item: item.name):
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode) and not stat_is_link_or_reparse(metadata):
                roots.append(Path(entry.path))
        return roots[:MAX_TRIALS_PER_ARM], limited or len(roots) > MAX_TRIALS_PER_ARM


def _load_tally(records: tuple[LogicalTrialEvidence, ...]) -> dict[str, dict[str, int]]:
    found = []
    for record in records:
        census = record.native_census
        if not isinstance(census, Mapping) or census.get("fallback"):
            found.append({})
            continue
        servers: dict[str, bool | None] = {}
        for field in ("loaded", "listed", "staged", "not_loaded"):
            rows = census.get(field)
            for row in rows[:256] if isinstance(rows, list) else ():
                if isinstance(row, Mapping) and row.get("type") == "mcp" and isinstance(row.get("name"), str):
                    previous = servers.get(row["name"])
                    known = field == "loaded" or (field == "not_loaded" and record.init_event is not None)
                    servers[row["name"]] = (
                        True if field == "loaded" or previous is True else False if known else previous
                    )
        found.append(servers)
    names = sorted({name for row in found for name in row})[:64]
    return {
        name: {
            "loaded": sum(row.get(name) is True for row in found),
            "trials": sum(row.get(name) is not None for row in found),
            "unknown": sum(row.get(name) is None for row in found),
        }
        for name in names
    }


def _proof_census(records: tuple[LogicalTrialEvidence, ...]) -> dict[str, Any]:
    """Only confirmed harness states can contradict the result of a host probe."""
    tally = _load_tally(records)
    loaded = [
        {"type": "mcp", "name": name}
        for name, counts in tally.items()
        if counts["trials"] > 0 and counts["loaded"] == counts["trials"] and not counts["unknown"]
    ]
    not_loaded = {}
    for record in records:
        if record.init_event is None or not isinstance(record.native_census, Mapping):
            continue
        for row in record.native_census.get("not_loaded", ())[:256]:
            if isinstance(row, Mapping) and row.get("type") == "mcp" and isinstance(row.get("name"), str):
                not_loaded[row["name"]] = {
                    "type": "mcp",
                    "name": row["name"],
                    "reason": "harness reported the server unavailable",
                }
    return {"loaded": loaded, "not_loaded": list(not_loaded.values())}


class RuntimeEvidenceObserver:
    """One prepared declaration context and one artifact cache for a private run."""

    def __init__(
        self,
        package: PreparedRuntimePackage,
        wrapper: WrapperSkill,
        hooks: NativeRuntimeHooks,
        options: EvidenceOptions,
    ) -> None:
        self.package, self.wrapper, self.hooks, self.options = package, wrapper, hooks, options
        self.logical_trials: dict[str, dict[str, tuple[LogicalTrialEvidence, ...]]] = {}
        self.summaries: dict[str, dict[str, dict[str, Any]]] = {}
        self.errors: list[str] = []
        self.artifacts = TrialTraceArtifacts()
        self.arm_jobs: dict[str, dict[str, Path]] = {}
        self.selector_map: Mapping[str, str] | None = None
        self._rewards: dict[Path, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
        self._collected: set[Path] = set()
        self._jobs_dir: Path | None = None
        self._sidecar: Path | None = None
        verify_source_identity(package)

        entries = collector._read_json(wrapper.path / "evals" / "evals.json")
        if not isinstance(entries, list):
            raise ValueError("prepared runtime dataset is unreadable")
        for entry in entries:
            self.errors.extend(_safe_text(error) for error in validate_plugin_case_fields(entry))
        self.context = build_plugin_signals_context(
            member_skills=[Path(str(member.path)).name for member in package.skills],
            mcp_servers=[server["name"] for server in package.native_source.mcp_servers],
            wrapper_skills=[wrapper.name],
            entries=entries,
            plugin_name=package.plugin_name,
            subagents=[text.name for text in package.native_source.texts if text.type == "agent"],
            commands=[text.name for text in package.native_source.texts if text.type == "command"],
        )
        portable = {server["name"] for server in package.native_source.mcp_servers}
        self.context = replace(
            self.context,
            unstaged_mcp_servers=tuple(
                server["name"] for server in package.mcp_servers if server["name"] not in portable
            ),
        )
        targets = [
            {**server, "transport": server.get("transport") or server.get("type") or "http"}
            for server in package.mcp_servers
            if server.get("url")
        ]
        self.proof = (
            probe_mcp_servers(
                targets,
                allowed_private_hosts=options.allowed_private_hosts,
                expand_env=options.env_grants,
                total_timeout=options.timeout,
            )
            if options.probe_mcp
            else declared_mcp_proof(targets)
        )
        self.context = replace(self.context, mcp_input_schemas=mcp_input_schemas(self.proof))
        self._host_proof = self.proof
        self._declaration_errors = tuple(self.errors)
        verify_source_identity(package)

    def _begin_attempt(self, jobs_dir: Path) -> None:
        """Release the previous attempt before accepting this attempt's first handoff."""
        if self._jobs_dir == jobs_dir:
            return
        self._jobs_dir = jobs_dir
        self.logical_trials = {}
        self.summaries = {}
        self.errors = list(self._declaration_errors)
        self.proof = self._host_proof
        self.artifacts = TrialTraceArtifacts()
        self._rewards = {}
        self._collected = set()
        self.arm_jobs = {
            agent: {arm: job for arm, job in arms.items() if job.parent == jobs_dir}
            for agent, arms in self.arm_jobs.items()
        }
        self.hooks.trial_censuses = {
            root: row for root, row in self.hooks.trial_censuses.items() if root.parent.parent == jobs_dir
        }
        self.hooks.trial_init_events = {
            root: row for root, row in self.hooks.trial_init_events.items() if root.parent.parent == jobs_dir
        }

    def configure_run(
        self,
        *,
        case_id_by_task_selector: Mapping[str, str] | None = None,
        arm_jobs: Mapping[str, Mapping[str, Path]] | None = None,
    ) -> None:
        if case_id_by_task_selector is not None:
            selected = collector._validated_case_id_by_task_selector(
                dict(case_id_by_task_selector), list(case_id_by_task_selector.values())
            )
            if not set(selected.values()).issubset(self.wrapper.case_sources):
                raise ValueError("Task selector mapping contains unknown logical case identities")
            self.selector_map = selected
        if arm_jobs is not None:
            self.arm_jobs = {agent: dict(arms) for agent, arms in arm_jobs.items()}

    def observe_collected_trials(
        self,
        agent: str,
        arm: str,
        job_dir: Path,
        collected: list[dict[str, Any]],
        scoreable: list[dict[str, Any]],
        selector_map: Mapping[str, str] | None,
    ) -> None:
        self._begin_attempt(job_dir.parent)
        self.configure_run(case_id_by_task_selector=selector_map)
        self.arm_jobs.setdefault(agent, {})[arm] = job_dir
        self._rewards[job_dir] = (collected, scoreable)

    def _case_id(self, root: Path, row: Mapping[str, Any] | None) -> str | None:
        selector = row.get("_trusted_task_selector") if isinstance(row, Mapping) else None
        if self.selector_map is not None:
            if not selector:
                result = collector._read_json(root / "result.json")
                selector = collector._task_selector_from_harbor_result(result) if isinstance(result, dict) else None
            return self.selector_map.get(selector) if selector else None
        # Generated trial selectors are coordinator-owned. Reward entry_id is
        # deliberately never an identity source, even if it names a valid case.
        selector = root.name.split("__", 1)[0]
        for suffix in ("-with-skill", "-without-skill", "-sum-of-parts", "-loose-parts"):
            selector = selector.removesuffix(suffix)
        return selector if selector in self.wrapper.case_sources else None

    def _context_for_staging(self) -> None:
        extra_servers, unavailable, aliases, lsp = {}, {}, {}, {}
        for agent, decision in self.hooks.decisions.items():
            staging = self.hooks.stagings.get(agent)
            if staging is None or not decision.native:
                unavailable[agent] = ("subagent", "command", "lsp")
                continue
            declared = staging.bundle.declared
            kinds = {"subagent" if row.get("type") == "agent" else row.get("type") for row in declared}
            unavailable[agent] = tuple(kind for kind in ("subagent", "command", "lsp") if kind not in kinds)
            extra_servers[agent] = tuple(row["name"] for row in declared if row.get("type") == "mcp")
            adapter = staging.adapter
            if hasattr(adapter, "staged_name"):
                aliases.update(
                    {
                        adapter.staged_name(self.package.native_source, text).casefold(): text.name
                        for text in self.package.native_source.texts
                        if text.type == "agent"
                    }
                )
            if "lsp" in kinds:
                lsp[agent] = {
                    name: tuple(config.get("extensionToLanguage", {}))
                    for name, config in self.package.native_source.lsp_servers.items()
                    if isinstance(config, Mapping)
                }
        self.context = replace(
            self.context,
            agent_mcp_servers=extra_servers,
            agent_unstaged=unavailable,
            subagent_aliases=aliases,
            agent_lsp_servers=lsp,
        )

    def collect_arm(self, agent: str, arm: str, job: Path) -> tuple[LogicalTrialEvidence, ...]:
        """Cache one arm once; callers may register later private arms through ``arm_jobs``."""
        self._begin_attempt(job.parent)
        previous = self.logical_trials.get(agent, {}).get(arm)
        if previous is not None:
            return previous
        roots, limited = _trial_roots(job)
        cached = self._rewards.get(job)
        if cached is None:
            rewards = collector._extract_rewards(job, self.selector_map, selected_trial_roots=roots)
            scoreable, _failures = collector._partition_scoreable_rewards(rewards)
        else:
            rewards, scoreable = cached
        groups: dict[str, list[dict[str, Any]]] = {}
        raw_by_root: dict[str, Mapping[str, Any]] = {}
        for reward in rewards:
            root = collector._safe_trial_path_component(reward.get("_trial_root_name"))
            if root:
                raw_by_root.setdefault(root, reward)
        for reward in scoreable:
            root = collector._safe_trial_path_component(reward.get("_trial_root_name"))
            if root:
                groups.setdefault(root, []).append(reward)
        labels = collector._published_trial_labels_by_root([root.name for root in roots])
        records = []
        for root in roots:
            rows = groups.get(root.name, [])
            logical = collector._logical_attempt_rewards(rows)
            reward = logical[0] if len(logical) == 1 else None
            case_id = self._case_id(root, raw_by_root.get(root.name))
            if reward is not None:
                reward = {**reward, "entry_id": case_id, "_authoritative_entry_id": True}
            trace = self.artifacts.trace(root)
            spec = self.context.case_spec(case_id or "")
            if arm in PLUGIN_PART_ARMS and self.context.mcp_input_schemas:
                spec = {**spec, MCP_INPUT_SCHEMAS_KEY: self.context.mcp_input_schemas}
            signals = (
                compute_plugin_signals(
                    trace.trajectory,
                    spec,
                    declared=self.context.declared_for(arm, agent),
                    wrapper_skills=self.context.wrapper_skills,
                    mcp_call_servers=trace.call_servers,
                    subagent_aliases=self.context.aliases_for(arm),
                    lsp_servers=self.context.lsp_servers_for(arm, agent),
                )
                if self.context.arm_enabled(arm)
                else None
            )
            status = (
                ("recorded" if signals.get("evidence_complete") is not False else "incomplete")
                if signals is not None
                else "missing"
                if trace.trajectory is None
                else "not_applicable"
            )
            usage = _usage((trace.trajectory or {}).get("final_metrics"))
            first_usage, excluded = _first_usage(trace.trajectory)
            if excluded == "hosted_search":
                usage = {**(usage or {}), "first_turn_hosted_search": 1.0}
            elif first_usage and first_usage.get("prompt_tokens") is not None:
                usage = {**(usage or {}), "first_turn_prompt_tokens": first_usage["prompt_tokens"]}
            records.append(
                LogicalTrialEvidence(
                    labels[root.name],
                    case_id,
                    reward,
                    signals,
                    self.hooks.trial_censuses.get(root),
                    usage,
                    first_usage,
                    status,
                    self.hooks.trial_init_events.get(root),
                    excluded,
                )
            )
        result = tuple(records)
        self.logical_trials.setdefault(agent, {})[arm] = result
        summary = summarize_plugin_signals([record.signals for record in result])
        summary.update(
            evidence_complete=bool(result)
            and not limited
            and all(r.evidence_status not in {"missing", "incomplete"} and r.case_id is not None for r in result),
            trials_truncated=limited,
            n_missing_trials=0,
        )
        summary["n_missing_trajectory"] = sum(record.evidence_status == "missing" for record in result)
        tally = _load_tally(result)
        if tally:
            summary[MCP_LOAD_KEY] = tally
        failures = top_argument_failures(record.signals for record in result)
        if failures:
            summary["arguments"]["top_failures"] = failures
        self.summaries.setdefault(agent, {})[arm] = summary
        return result

    def collect(self, jobs_dir: Path, result: dict[str, Any] | None) -> None:
        self._begin_attempt(jobs_dir)
        if jobs_dir in self._collected:
            return
        self._collected.add(jobs_dir)
        self._sidecar = jobs_dir.parent / RUNTIME_EVIDENCE_FILENAME
        self._context_for_staging()
        agents = (result or {}).get("agents", {})
        names = sorted(set(self.hooks.decisions) | (set(agents) if isinstance(agents, Mapping) else set()))[:32]
        for agent in names:
            conditions = agents.get(agent, {}).get("conditions", {}) if isinstance(agents, Mapping) else {}
            requested = {
                arm
                for arm, condition in conditions.items()
                if arm in ARM_SUFFIXES
                and isinstance(condition, Mapping)
                and condition.get("execution_status") != "skipped"
            }
            if not conditions:
                requested = {ARM_WITH}
            registered = self.arm_jobs.get(agent, {})
            requested.update(registered)
            jobs = {
                arm: registered.get(arm, jobs_dir / f"{self.wrapper.name}-{agent}-{ARM_SUFFIXES[arm]}")
                for arm in requested
                if arm in ARM_SUFFIXES
            }
            for arm, job in jobs.items():
                condition = conditions.get(arm, {})
                expected = condition.get("expected_attempts")
                expected = expected if type(expected) is int and expected >= 0 else 1
                try:
                    if job.exists():
                        records = self.collect_arm(agent, arm, job)
                    else:
                        records = ()
                        self.logical_trials.setdefault(agent, {})[arm] = records
                        self.summaries.setdefault(agent, {})[arm] = summarize_plugin_signals([])
                    missing = max(0, expected - len(records))
                    summary = self.summaries[agent][arm]
                    summary["n_missing_trials"] = missing
                    summary["evidence_complete"] = (
                        bool(records) and not missing and summary.get("evidence_complete", False)
                    )
                except Exception as exc:
                    self.errors.append(_safe_text(f"{agent}/{arm}: {type(exc).__name__}: {exc}"))
                    self.summaries.setdefault(agent, {})[arm] = {
                        **summarize_plugin_signals([]),
                        "evidence_complete": False,
                        "n_missing_trials": expected,
                    }
            if result is not None and isinstance(agents, dict) and isinstance(agents.get(agent), dict):
                agents[agent]["plugin_signals_summary"] = self.summaries.get(agent, {})
        proof_agents = {
            agent: {
                "plugin_signals_summary": self.summaries.get(agent, {}),
                "plugin_load_census": _proof_census(self.logical_trials.get(agent, {}).get(ARM_WITH, ())),
            }
            for agent in names
        }
        self.proof = apply_in_agent_mcp_proof(self._host_proof, {"agents": proof_agents})
        if result is not None:
            result["runtime_evidence"] = self.provenance()
        self.persist()

    def record_failure(self, exc: Exception) -> None:
        """Keep observer failures advisory and redact diagnostics before publication."""
        self.errors.append(_safe_text(f"{type(exc).__name__}: {exc}"))

    def provenance(self) -> dict[str, Any]:
        return {
            "source": vars(self.package.identity),
            "mcp_proof": {
                name: {key: value for key, value in row.items() if key in {"status", "tools", "detail"}}
                for name, row in self.proof.items()
            },
            "errors": self.errors[:64],
            "evidence_complete": bool(self.summaries)
            and not self.errors
            and all(summary.get("evidence_complete") for arms in self.summaries.values() for summary in arms.values()),
        }

    def persist(self) -> None:
        if self._sidecar is None:
            return
        data = {
            **self.provenance(),
            "agents": {
                agent: {
                    arm: {
                        "summary": self.summaries[agent][arm],
                        "trials": [row.persisted() for row in rows],
                        "trial_details_total": len(rows),
                    }
                    for arm, rows in arms.items()
                }
                for agent, arms in self.logical_trials.items()
            },
        }
        while len(json.dumps(data, ensure_ascii=True, allow_nan=False)) > MAX_EVIDENCE_BYTES:
            groups = [arm for arms in data["agents"].values() for arm in arms.values() if arm["trials"]]
            if not groups:
                raise ValueError("runtime evidence summary exceeds its publication bound")
            for arm in groups:
                arm["trials"] = arm["trials"][: len(arm["trials"]) // 2]
                arm["trial_details_truncated"] = True
            data["evidence_complete"] = False
        collector._write_generated_root_json(self._sidecar, self._sidecar.parent, data)
