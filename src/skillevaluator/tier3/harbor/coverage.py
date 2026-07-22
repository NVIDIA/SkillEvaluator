# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical SkillEvaluator agent-coverage policy and artifact primitives.

This module deliberately has no dependency on the Harbor runner.  It is the
small, fail-closed contract boundary shared by runner, CLI, and downstream
evidence producers.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import math
import os
import re
import secrets
import stat
import tomllib
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from skillevaluator.tier3.harbor.failure_taxonomy import (
    AGENT_FAILURE_TAXONOMY as _AGENT_FAILURE_TAXONOMY,
)
from skillevaluator.tier3.harbor.failure_taxonomy import (
    LAUNCHED_AGENT_FAILURE_TAXONOMY as _LAUNCHED_AGENT_FAILURE_TAXONOMY,
)
from skillevaluator.tier3.harbor.failure_taxonomy import (
    RUN_FAILURE_TAXONOMY as _RUN_FAILURE_TAXONOMY,
)
from skillevaluator.tier3.harbor.failure_taxonomy import (
    TRUSTED_AGENT_EXECUTION_EXCEPTIONS as _TRUSTED_AGENT_EXECUTION_EXCEPTIONS,
)
from skillevaluator.tier3.harbor.metrics import (
    CUSTOM_ONLY_METRIC_SET,
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    RESERVED_METRIC_NAMES,
)

SCHEMA_VERSION = "1.0"
CAPABILITY = "agent-coverage/1"
DATASET_DIGEST_ALGORITHM = "skill-eval-dataset-c14n/1"
STAGED_TASK_DIGEST_ALGORITHM = "skill-evaluator-staged-harbor-task-tree-c14n/1"
MAX_CANONICAL_BYTES = 4 * 1024 * 1024
MAX_STAGED_TREE_FILES = 100_000
MAX_STAGED_TREE_BYTES = 4 * 1024 * 1024 * 1024
MAX_SIDECAR_TRIALS = 1_000_000
MAX_REWARD_PROPERTIES = 256
MAX_RESULT_STEPS = 10_000

CoverageStatus = Literal["valid_full", "valid_degraded", "invalid"]
AgentStatus = Literal[
    "valid",
    "invalid_infrastructure",
    "invalid_configuration",
    "invalid_coverage",
    "not_evaluated_run_blocked",
]
Phase = Literal[
    "policy_validation",
    "dataset_validation",
    "task_generation",
    "preflight",
    "execution",
    "completed",
]
Disposition = Literal[
    "scored",
    "failed",
    "skipped_stop_on_pass",
    "not_run_agent_excluded",
    "not_run_agent_unavailable",
    "not_run_run_blocked",
]
Arm = Literal["with_skill", "baseline"]
RewardStrategy = Literal["single_step", "multi_step_mean", "multi_step_final"]
DatasetSnapshotKind = Literal["evals_json", "native_harbor"]
FailureOrigin = Literal[
    "trusted_preflight",
    "harbor_pre_instruction_phase",
    "trusted_adapter_marker",
    "trusted_execution_result",
    "run_scope",
]

_COVERAGE_STATUSES = frozenset({"valid_full", "valid_degraded", "invalid"})
_AGENT_STATUSES = frozenset(
    {
        "valid",
        "invalid_infrastructure",
        "invalid_configuration",
        "invalid_coverage",
        "not_evaluated_run_blocked",
    }
)
_PHASES = (
    "policy_validation",
    "dataset_validation",
    "task_generation",
    "preflight",
    "execution",
    "completed",
)
_PHASE_INDEX = {phase: index for index, phase in enumerate(_PHASES)}
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SCHEMA_VERSION_RE = re.compile(r"^1\.(0|[1-9][0-9]*)$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXTENSION_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_EXCEPTION_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_SAFE_METRIC_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "phase",
        "capabilities",
        "requested_policy_digest",
        "effective_policy_digest",
        "task_plan_digest",
        "task_plan_ref",
        "execution_ledger_digest",
        "execution_ledger_ref",
        "dataset_digest",
        "dataset_digest_algorithm",
        "status",
        "requested_policy",
        "authorized_tightening",
        "effective_policy",
        "policy_provenance",
        "requested_agents",
        "eligible_agents",
        "excluded_agents",
        "agents",
        "warnings",
        "blockers",
        "extensions",
    }
)
_ROOT_REQUIRED_FIELDS = _ROOT_FIELDS - {"extensions"}
_POLICY_FIELDS = frozenset({"mode", "min_valid_agents", "required_agents"})
_CAPABILITY_FIELDS = frozenset({"requested", "provided"})
_AGENT_FIELDS = frozenset(
    {
        "base_agent",
        "occurrence",
        "requested_model",
        "resolved_model",
        "model_source",
        "status",
        "score_eligible",
        "reason_code",
        "failure_stage",
        "failure_origin",
        "evidence_ref",
        "evidence_file_digest",
        "viewer_url",
        "with_skill",
        "baseline",
    }
)
_AGENT_REQUIRED_FIELDS = frozenset(
    {
        "base_agent",
        "occurrence",
        "requested_model",
        "resolved_model",
        "model_source",
        "status",
        "score_eligible",
        "with_skill",
        "baseline",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "expected_cases",
        "scored_cases",
        "exceptions",
        "expected_attempts",
        "scored_attempts",
        "failed_attempts",
        "skipped_attempts",
        "not_run_attempts",
    }
)
_WARNING_FIELDS = frozenset(
    {
        "code",
        "agent",
        "reason_code",
        "failure_stage",
        "failure_origin",
        "evidence_ref",
        "evidence_file_digest",
    }
)
_FAILURE_FIELDS = frozenset(
    {"scope", "stage", "reason_code", "origin", "agent", "evidence_ref", "evidence_file_digest"}
)


class ContractError(ValueError):
    """Raised when evidence cannot satisfy the coverage contract."""


def _require_nonempty_text(value: object, name: str, *, limit: int = 1024) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > limit:
        raise ContractError(f"{name} must be a non-empty bounded string")
    return value


def _require_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_RE.fullmatch(value):
        raise ContractError(f"{name} is not a valid occurrence identity")
    return value


def harbor_model_for_agent(base_agent: str, resolved_model: str) -> str:
    """Return the exact pinned Harbor model identity for a resolved agent model."""

    _require_identity(base_agent, "base_agent")
    return _require_nonempty_text(resolved_model, "resolved_model")


def _require_positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ContractError(f"{name} must be a {qualifier} integer")
    return value


def _require_unique_strings(values: object, name: str, *, identity: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ContractError(f"{name} must be an ordered list of strings")
    parsed: list[str] = []
    for index, value in enumerate(values):
        item = (
            _require_identity(value, f"{name}[{index}]")
            if identity
            else _require_nonempty_text(value, f"{name}[{index}]")
        )
        if item in parsed:
            raise ContractError(f"{name} contains duplicate value {item!r}")
        parsed.append(item)
    return tuple(parsed)


@dataclass(frozen=True)
class AgentOccurrence:
    result_key: str
    base_agent: str
    occurrence: int
    requested_model: str | None
    resolved_model: str
    model_source: str

    def __post_init__(self) -> None:
        _require_identity(self.result_key, "result_key")
        _require_identity(self.base_agent, "base_agent")
        _require_positive_int(self.occurrence, "occurrence")
        if self.requested_model is not None:
            _require_nonempty_text(self.requested_model, "requested_model")
        _require_nonempty_text(self.resolved_model, "resolved_model")
        _require_nonempty_text(self.model_source, "model_source")


@dataclass(frozen=True)
class CoveragePolicy:
    mode: Literal["all_selected", "any_valid"]
    min_valid_agents: int
    required_agents: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"all_selected", "any_valid"}:
            raise ContractError(f"unsupported coverage policy mode: {self.mode!r}")
        _require_positive_int(self.min_valid_agents, "min_valid_agents", allow_zero=True)
        if not isinstance(self.required_agents, tuple):
            raise ContractError("required_agents must be a tuple")
        _require_unique_strings(self.required_agents, "required_agents", identity=True)


@dataclass(frozen=True)
class PolicyResolution:
    requested: CoveragePolicy
    effective: CoveragePolicy
    requested_digest: str
    effective_digest: str
    provenance: Literal["trusted_caller", "trusted_caller_plus_authorized_tightening"]
    authorized_tightening: CoveragePolicy | None = None


@dataclass(frozen=True)
class FailureRecord:
    scope: Literal["run", "agent"]
    stage: str
    reason_code: str
    origin: FailureOrigin | None = None
    agent: str | None = None
    evidence_ref: str | None = None
    evidence_file_digest: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"run", "agent"}:
            raise ContractError(f"unknown failure scope: {self.scope!r}")
        _require_nonempty_text(self.stage, "failure stage", limit=128)
        _require_nonempty_text(self.reason_code, "failure reason_code", limit=128)
        pair = (self.stage, self.reason_code)
        taxonomy = _RUN_FAILURE_TAXONOMY if self.scope == "run" else _AGENT_FAILURE_TAXONOMY
        if pair not in taxonomy:
            raise ContractError(f"failure stage/reason pair is not in the trusted {self.scope} taxonomy")
        origin = self.origin
        if origin is None:
            if self.scope == "run":
                origin = "run_scope"
            elif self.stage in {"agent_readiness", "preflight"}:
                origin = "trusted_preflight"
            else:
                raise ContractError("launched agent failure requires an explicit trusted origin")
            object.__setattr__(self, "origin", origin)
        if origin not in {
            "trusted_preflight",
            "harbor_pre_instruction_phase",
            "trusted_adapter_marker",
            "trusted_execution_result",
            "run_scope",
        }:
            raise ContractError("failure origin is unsupported")
        if self.scope == "run" and origin != "run_scope":
            raise ContractError("run-scoped failure requires origin=run_scope")
        if self.scope == "agent":
            if origin == "trusted_preflight" and self.stage not in {"agent_readiness", "preflight"}:
                raise ContractError("trusted_preflight origin is restricted to readiness/preflight")
            if origin == "harbor_pre_instruction_phase" and pair != (
                "agent_adapter_bootstrap",
                "adapter_initialization_failed",
            ):
                raise ContractError("Harbor pre-instruction origin is restricted to adapter setup")
            if origin == "trusted_adapter_marker" and pair not in _LAUNCHED_AGENT_FAILURE_TAXONOMY:
                raise ContractError("trusted adapter marker is outside the launched-agent taxonomy")
            if origin == "trusted_execution_result" and self.stage != "agent_execution":
                raise ContractError("trusted execution result is restricted to agent execution")
            if origin == "run_scope":
                raise ContractError("agent-scoped failure cannot use run_scope origin")
        if self.scope == "run" and self.agent is not None:
            raise ContractError("run-scoped blocker cannot name an agent")
        if self.scope == "agent":
            _require_identity(self.agent, "agent-scoped failure agent")
        if self.evidence_ref is not None:
            _validated_ref_text(self.evidence_ref)
        if (self.evidence_ref is None) != (self.evidence_file_digest is None):
            raise ContractError("diagnostic reference and file digest are all-or-none")
        if self.evidence_file_digest is not None:
            _validate_digest(self.evidence_file_digest, "evidence_file_digest", nullable=False)


@dataclass(frozen=True)
class PreflightReport:
    """Closed structured readiness report shared by runner preflight producers."""

    run_blockers: tuple[FailureRecord, ...] = ()
    agent_exclusions: dict[str, FailureRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not isinstance(item, FailureRecord) or item.scope != "run" for item in self.run_blockers):
            raise ContractError("preflight run blockers must be typed run-scoped failures")
        exclusions = dict(self.agent_exclusions)
        for agent, failure in exclusions.items():
            if failure.scope != "agent" or failure.agent != agent:
                raise ContractError("preflight agent exclusion must match its occurrence key")
        object.__setattr__(self, "agent_exclusions", exclusions)


@dataclass(frozen=True)
class CoverageDecision:
    status: CoverageStatus
    eligible_agents: tuple[str, ...]
    excluded_agents: tuple[str, ...]


@dataclass(frozen=True, order=True)
class ExpectedAttempt:
    """One immutable logical attempt from the pre-execution task plan."""

    agent: str
    arm: Arm
    case_id: str
    ordinal: int

    def __post_init__(self) -> None:
        _require_identity(self.agent, "attempt agent")
        if self.arm not in {"with_skill", "baseline"}:
            raise ContractError(f"unsupported attempt arm: {self.arm!r}")
        _require_nonempty_text(self.case_id, "attempt case_id", limit=256)
        ordinal = _require_positive_int(self.ordinal, "attempt ordinal")
        if ordinal > 1000:
            raise ContractError("attempt ordinal cannot exceed 1000")


@dataclass(frozen=True)
class AttemptRecord:
    """Observed disposition for exactly one planned logical attempt."""

    expected: ExpectedAttempt
    disposition: Disposition
    trial_ref: str | None
    trial_file_digest: str | None
    passed: bool | None = None
    score: float | None = None
    failure: FailureRecord | None = None
    caused_by: ExpectedAttempt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expected, ExpectedAttempt):
            raise ContractError("attempt record expected must be an ExpectedAttempt")
        if self.disposition not in {
            "scored",
            "failed",
            "skipped_stop_on_pass",
            "not_run_agent_excluded",
            "not_run_agent_unavailable",
            "not_run_run_blocked",
        }:
            raise ContractError(f"unsupported attempt disposition: {self.disposition!r}")
        if self.caused_by is not None and not isinstance(self.caused_by, ExpectedAttempt):
            raise ContractError("attempt caused_by must be an ExpectedAttempt")


def derive_attempt_passed(
    score: float,
    pass_threshold: float,
    *,
    supplied: bool | None = None,
) -> bool:
    """Derive pass from the exact unrounded score and reject contradictions."""

    if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(float(score)):
        raise ContractError("attempt score must be finite")
    if (
        isinstance(pass_threshold, bool)
        or not isinstance(pass_threshold, int | float)
        or not math.isfinite(float(pass_threshold))
    ):
        raise ContractError("pass threshold must be finite")
    derived = float(score) >= float(pass_threshold)
    if supplied is not None and (not isinstance(supplied, bool) or supplied is not derived):
        raise ContractError("collector-supplied pass boolean disagrees with unrounded canonical score")
    return derived


def resolve_occurrences(
    base_agents: Sequence[str],
    *,
    requested_models: Sequence[str | None],
    resolved_models: Sequence[str],
    model_sources: Sequence[str],
) -> tuple[AgentOccurrence, ...]:
    """Resolve selected agents into stable, ordered six-field identities."""

    if isinstance(base_agents, (str, bytes)):
        raise ContractError("base_agents must be an ordered sequence")
    bases = tuple(_require_identity(agent, "base_agent") for agent in base_agents)
    if not bases:
        raise ContractError("at least one agent must be selected")
    requested = tuple(requested_models)
    resolved = tuple(resolved_models)
    sources = tuple(model_sources)
    if not (len(bases) == len(requested) == len(resolved) == len(sources)):
        raise ContractError("agent/model identity fields must have equal lengths")

    totals = Counter(bases)
    seen: Counter[str] = Counter()
    result_keys: set[str] = set()
    occurrences: list[AgentOccurrence] = []
    for base, requested_model, resolved_model, source in zip(bases, requested, resolved, sources, strict=True):
        seen[base] += 1
        occurrence = seen[base]
        result_key = base if totals[base] == 1 else f"{base}-{occurrence}"
        if result_key in result_keys:
            raise ContractError(f"resolved result-key collision: {result_key!r}")
        result_keys.add(result_key)
        occurrences.append(
            AgentOccurrence(
                result_key=result_key,
                base_agent=base,
                occurrence=occurrence,
                requested_model=requested_model,
                resolved_model=resolved_model,
                model_source=source,
            )
        )
    return tuple(occurrences)


def _validated_occurrences(agents: Sequence[AgentOccurrence]) -> tuple[AgentOccurrence, ...]:
    if isinstance(agents, (str, bytes)):
        raise ContractError("agents must be ordered occurrence records")
    parsed = tuple(agents)
    if not parsed or any(not isinstance(agent, AgentOccurrence) for agent in parsed):
        raise ContractError("at least one valid agent occurrence is required")
    keys = tuple(agent.result_key for agent in parsed)
    if len(keys) != len(set(keys)):
        raise ContractError("agent occurrence result keys must be unique")
    base_occurrences = [(agent.base_agent, agent.occurrence) for agent in parsed]
    if len(base_occurrences) != len(set(base_occurrences)):
        raise ContractError("agent base/occurrence identities must be unique")
    totals = Counter(agent.base_agent for agent in parsed)
    seen: Counter[str] = Counter()
    for agent in parsed:
        seen[agent.base_agent] += 1
        expected_occurrence = seen[agent.base_agent]
        expected_key = (
            agent.base_agent if totals[agent.base_agent] == 1 else f"{agent.base_agent}-{expected_occurrence}"
        )
        if agent.occurrence != expected_occurrence or agent.result_key != expected_key:
            raise ContractError("agent result-key identity does not match ordered base/occurrence identities")
    return parsed


def resolve_required_agents(required_agents: Sequence[str], agents: Sequence[AgentOccurrence]) -> tuple[str, ...]:
    """Resolve required occurrence keys; accept a base only when unambiguous."""

    occurrences = _validated_occurrences(agents)
    requested = _require_unique_strings(required_agents, "required_agents", identity=True)
    by_key = {agent.result_key: agent for agent in occurrences}
    resolved: set[str] = set()
    for value in requested:
        if value in by_key:
            key = value
        else:
            matches = [agent.result_key for agent in occurrences if agent.base_agent == value]
            if not matches:
                raise ContractError(f"required agent {value!r} was not selected")
            if len(matches) != 1:
                raise ContractError(f"required base agent {value!r} is ambiguous; use an occurrence key")
            key = matches[0]
        if key in resolved:
            raise ContractError(f"required agent {value!r} resolves to a duplicate occurrence")
        resolved.add(key)
    return tuple(agent.result_key for agent in occurrences if agent.result_key in resolved)


def _normalize_policy(
    policy: CoveragePolicy,
    agents: tuple[AgentOccurrence, ...],
    *,
    label: str,
) -> CoveragePolicy:
    if not isinstance(policy, CoveragePolicy):
        raise ContractError(f"{label} must be a CoveragePolicy")
    count = len(agents)
    required = resolve_required_agents(policy.required_agents, agents)
    if policy.mode == "all_selected":
        minimum = count if policy.min_valid_agents == 0 else policy.min_valid_agents
        if minimum != count:
            raise ContractError("all-selected policy requires min_valid_agents to equal selected-agent count")
    else:
        minimum = 1 if policy.min_valid_agents == 0 else policy.min_valid_agents
        if minimum < 1 or minimum > count:
            raise ContractError(f"{label} min_valid_agents must be between 1 and {count}")
    return CoveragePolicy(policy.mode, minimum, required)


def _policy_object(policy: CoveragePolicy) -> dict[str, object]:
    return {
        "mode": policy.mode,
        "min_valid_agents": policy.min_valid_agents,
        "required_agents": list(policy.required_agents),
    }


def resolve_policy(
    requested_policy: CoveragePolicy,
    agents: Sequence[AgentOccurrence],
    capabilities: Sequence[str],
    *,
    authorized_tightening: CoveragePolicy | None = None,
    allow_content_tightening: bool = True,
) -> PolicyResolution:
    """Normalize caller policy and deterministically join authorized tightening."""

    occurrences = _validated_occurrences(agents)
    parsed_capabilities = _require_unique_strings(capabilities, "capabilities")
    unknown = set(parsed_capabilities) - {CAPABILITY}
    if unknown:
        raise ContractError(f"unsupported capability request: {sorted(unknown)!r}")
    requested = _normalize_policy(requested_policy, occurrences, label="requested policy")
    if requested.mode == "any_valid" and CAPABILITY not in parsed_capabilities:
        raise ContractError(f"any-valid policy requires negotiated capability {CAPABILITY}")

    tightening: CoveragePolicy | None = None
    if authorized_tightening is not None:
        if not allow_content_tightening:
            raise ContractError("authorized policy tightening is disabled by the trusted caller")
        tightening = _normalize_policy(authorized_tightening, occurrences, label="authorized tightening")

    if tightening is None:
        effective = requested
        provenance: Literal["trusted_caller", "trusted_caller_plus_authorized_tightening"] = "trusted_caller"
    else:
        mode: Literal["all_selected", "any_valid"] = (
            "all_selected" if "all_selected" in {requested.mode, tightening.mode} else "any_valid"
        )
        if mode == "all_selected":
            minimum = len(occurrences)
        else:
            minimum = max(requested.min_valid_agents, tightening.min_valid_agents)
        required_keys = set(requested.required_agents) | set(tightening.required_agents)
        required = tuple(agent.result_key for agent in occurrences if agent.result_key in required_keys)
        effective = CoveragePolicy(mode, minimum, required)
        provenance = "trusted_caller_plus_authorized_tightening"

    return PolicyResolution(
        requested=requested,
        effective=effective,
        requested_digest=canonical_digest(_policy_object(requested)),
        effective_digest=canonical_digest(_policy_object(effective)),
        provenance=provenance,
        authorized_tightening=tightening,
    )


def _partition_values(values: Sequence[str], name: str) -> tuple[str, ...]:
    try:
        return _require_unique_strings(values, name, identity=True)
    except ContractError as error:
        raise ContractError(f"invalid coverage partition: {error}") from error


def calculate_coverage(
    policy: CoveragePolicy,
    requested_agents: Sequence[str],
    eligible_agents: Sequence[str],
    excluded_agents: Sequence[str],
    blockers: Sequence[FailureRecord],
) -> CoverageDecision:
    """Classify an explicit, exact occurrence partition under trusted policy."""

    requested = _partition_values(requested_agents, "requested_agents")
    eligible = _partition_values(eligible_agents, "eligible_agents")
    excluded = _partition_values(excluded_agents, "excluded_agents")
    if not requested:
        raise ContractError("invalid coverage partition: requested_agents cannot be empty")
    eligible_set = set(eligible)
    excluded_set = set(excluded)
    requested_set = set(requested)
    if eligible_set & excluded_set or eligible_set | excluded_set != requested_set:
        raise ContractError("invalid coverage partition: requested agents must equal eligible plus excluded")
    if eligible != tuple(agent for agent in requested if agent in eligible_set):
        raise ContractError("invalid coverage partition: eligible agents are not in requested order")
    if excluded != tuple(agent for agent in requested if agent in excluded_set):
        raise ContractError("invalid coverage partition: excluded agents are not in requested order")

    if not isinstance(policy, CoveragePolicy) or policy.min_valid_agents < 1:
        raise ContractError("coverage policy must be normalized before calculation")
    if not set(policy.required_agents).issubset(requested_set):
        raise ContractError("coverage policy required agents are not a subset of requested agents")
    if policy.mode == "all_selected" and policy.min_valid_agents != len(requested):
        raise ContractError("all-selected coverage policy does not match requested-agent count")
    if policy.mode == "any_valid" and policy.min_valid_agents > len(requested):
        raise ContractError("any-valid coverage minimum exceeds requested-agent count")

    if isinstance(blockers, (str, bytes)):
        raise ContractError("blockers must be structured failure records")
    parsed_blockers = tuple(blockers)
    for blocker in parsed_blockers:
        if not isinstance(blocker, FailureRecord) or blocker.scope != "run":
            raise ContractError("blockers must contain only typed run-scoped failures")

    required_ok = set(policy.required_agents).issubset(eligible_set)
    if parsed_blockers or not required_ok or len(eligible) < policy.min_valid_agents:
        status: CoverageStatus = "invalid"
    elif not excluded:
        status = "valid_full"
    else:
        status = "valid_degraded"
    return CoverageDecision(status=status, eligible_agents=eligible, excluded_agents=excluded)


def _validate_json_value(value: object, *, path: str = "$", seen: set[int] | None = None, depth: int = 0) -> None:
    if depth > 256:
        raise ContractError(f"{path} exceeds the canonical JSON nesting limit")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(cast("float", value)):
            raise ContractError(f"{path} canonical JSON number must be finite")
        return
    if type(value) not in {list, dict}:
        raise ContractError(f"{path} contains non-JSON type {type(value).__name__}")

    active = seen if seen is not None else set()
    identity = id(value)
    if identity in active:
        raise ContractError(f"{path} contains a cyclic JSON value")
    active.add(identity)
    try:
        if type(value) is list:
            for index, item in enumerate(cast("list[object]", value)):
                _validate_json_value(item, path=f"{path}[{index}]", seen=active, depth=depth + 1)
            return
        for key, item in cast("dict[object, object]", value).items():
            if type(key) is not str:
                raise ContractError(f"{path} JSON objects require string keys")
            _validate_json_value(item, path=f"{path}.{key}", seen=active, depth=depth + 1)
    finally:
        active.remove(identity)


def canonical_json_bytes(value: object, *, trailing_newline: bool = False) -> bytes:
    """Encode bounded canonical JSON in the contract's selected byte domain."""

    _validate_json_value(value)
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ContractError(f"value cannot be encoded as canonical JSON: {error}") from error
    if trailing_newline:
        encoded += b"\n"
    if len(encoded) > MAX_CANONICAL_BYTES:
        raise ContractError(f"canonical JSON exceeds {MAX_CANONICAL_BYTES} bytes")
    return encoded


def _sha256_digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def canonical_digest(value: object) -> str:
    """Return a policy/object self-digest (canonical bytes, no newline)."""

    return _sha256_digest(canonical_json_bytes(value))


def _absolute_lexical(path: Path) -> Path:
    # Deliberately normalize lexically without resolving symlinks. Artifact
    # containment validation resolves each component through dirfd operations.
    return Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100


def _reject_parent_syntax(path: Path, name: str) -> None:
    if ".." in path.parts:
        raise ContractError(f"{name} cannot contain parent traversal")


def _open_trusted_root(trusted_root: Path) -> tuple[Path, int]:
    raw_root = Path(trusted_root)
    _reject_parent_syntax(raw_root, "trusted root")
    root = _absolute_lexical(raw_root)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None or os.open not in os.supports_dir_fd:
        raise ContractError("secure dirfd-anchored artifact access is unavailable on this platform")
    flags = os.O_RDONLY | nofollow | directory
    current_fd = os.open(root.anchor, flags)
    try:
        for component in root.parts[1:]:
            child_fd = _open_child_directory(current_fd, component)
            os.close(current_fd)
            current_fd = child_fd
        return root, current_fd
    except ContractError as error:
        os.close(current_fd)
        raise ContractError(f"trusted root is invalid: {error}") from error
    except BaseException:
        os.close(current_fd)
        raise


def _relative_parts(root: Path, path: Path) -> tuple[str, ...]:
    raw_path = Path(path)
    _reject_parent_syntax(raw_path, "artifact path")
    target = _absolute_lexical(raw_path)
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ContractError(f"artifact path is outside trusted root: {target}") from error
    if not relative.parts:
        raise ContractError("artifact path must name a file below the trusted root")
    for component in relative.parts:
        if component in {"", ".", ".."} or "/" in component or "\x00" in component:
            raise ContractError("artifact path has an unsafe component")
    return cast("tuple[str, ...]", relative.parts)


def _open_child_directory(parent_fd: int, component: str) -> int:
    try:
        entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ContractError(f"artifact parent does not exist: {component}") from error
    if stat.S_ISLNK(entry.st_mode):
        raise ContractError(f"artifact parent is a symlink: {component}")
    if not stat.S_ISDIR(entry.st_mode):
        raise ContractError(f"artifact parent is not a directory: {component}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        child_fd = os.open(component, flags, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ContractError(f"artifact parent is a symlink or not a directory: {component}") from error
        raise
    opened = os.fstat(child_fd)
    if (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino):
        os.close(child_fd)
        raise ContractError(f"artifact parent changed while it was opened: {component}")
    return child_fd


def _open_parent(root_fd: int, parent_parts: Sequence[str]) -> int:
    current_fd = os.dup(root_fd)
    try:
        for component in parent_parts:
            child_fd = _open_child_directory(current_fd, component)
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def ensure_artifact_parent(path: Path, *, trusted_root: Path) -> None:
    """Create artifact parent directories without following symlinks."""

    root, root_fd = _open_trusted_root(Path(trusted_root))
    current_fd = -1
    try:
        parts = _relative_parts(root, Path(path))
        current_fd = os.dup(root_fd)
        for component in parts[:-1]:
            try:
                os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    pass
            child_fd = _open_child_directory(current_fd, component)
            os.close(current_fd)
            current_fd = child_fd
    finally:
        if current_fd >= 0:
            os.close(current_fd)
        os.close(root_fd)


def atomic_write_json(path: Path, value: object, *, trusted_root: Path) -> str:
    """Publish canonical JSON once using dirfd anchoring and an exclusive link."""

    data = canonical_json_bytes(value, trailing_newline=True)
    root, root_fd = _open_trusted_root(Path(trusted_root))
    parent_fd = -1
    temp_name: str | None = None
    final_name: str | None = None
    published = False
    try:
        parts = _relative_parts(root, Path(path))
        final_name = parts[-1]
        parent_fd = _open_parent(root_fd, parts[:-1])
        try:
            os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(errno.EEXIST, "canonical artifact already exists", os.fspath(path))

        temp_name = f".{final_name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        temp_fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        try:
            view = memoryview(data)
            while view:
                written = os.write(temp_fd, view)
                if written <= 0:
                    raise OSError(errno.EIO, "short write while persisting canonical artifact")
                view = view[written:]
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)

        os.link(
            temp_name,
            final_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temp_name, dir_fd=parent_fd)
        temp_name = None
        os.fsync(parent_fd)
        return _sha256_digest(data)
    except BaseException:
        if parent_fd >= 0:
            if published and final_name is not None:
                with contextlib.suppress(OSError):
                    os.unlink(final_name, dir_fd=parent_fd)
            if temp_name is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temp_name, dir_fd=parent_fd)
            with contextlib.suppress(OSError):
                os.fsync(parent_fd)
        raise
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(root_fd)


def _validated_ref_text(ref: object) -> str:
    if (
        not isinstance(ref, str)
        or not ref
        or len(ref) > 4096
        or "\\" in ref
        or any(
            ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F or ord(character) in {0x2028, 0x2029}
            for character in ref
        )
    ):
        raise ContractError("evidence reference must be a non-empty POSIX relative path")
    pure = PurePosixPath(ref)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in ref.split("/")):
        raise ContractError(f"evidence reference escapes or is non-canonical: {ref!r}")
    if str(pure) != ref:
        raise ContractError(f"evidence reference is not canonical: {ref!r}")
    return ref


def normalized_refs_overlap(left: str, right: str) -> bool:
    """Return whether safe refs alias or contain one another cross-platform."""

    left_ref = _validated_ref_text(left)
    right_ref = _validated_ref_text(right)
    left_parts = tuple(unicodedata.normalize("NFC", part).casefold() for part in left_ref.split("/"))
    right_parts = tuple(unicodedata.normalize("NFC", part).casefold() for part in right_ref.split("/"))
    common = min(len(left_parts), len(right_parts))
    return left_parts[:common] == right_parts[:common]


def normalized_identity_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _open_relative_regular(trusted_root: Path, ref: str) -> tuple[int, int]:
    canonical_ref = _validated_ref_text(ref)
    _root, root_fd = _open_trusted_root(trusted_root)
    parts = canonical_ref.split("/")
    parent_fd = -1
    try:
        parent_fd = _open_parent(root_fd, parts[:-1])
        name = parts[-1]
        try:
            entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ContractError(f"evidence reference does not exist: {ref}") from error
        if stat.S_ISLNK(entry.st_mode):
            raise ContractError(f"evidence reference is a symlink: {ref}")
        if not stat.S_ISREG(entry.st_mode):
            raise ContractError(f"evidence reference is not a regular file: {ref}")
        if entry.st_nlink != 1:
            raise ContractError(f"evidence reference is hard-linked: {ref}")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino):
                raise ContractError(f"evidence reference changed while it was opened: {ref}")
            if opened.st_nlink != 1:
                raise ContractError(f"evidence reference is hard-linked: {ref}")
            if opened.st_size > MAX_CANONICAL_BYTES:
                raise ContractError(f"evidence reference is too large: {ref}")
        except BaseException:
            os.close(fd)
            raise
        return fd, opened.st_size
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(root_fd)


def verified_relative_ref(trusted_root: Path, ref: str) -> str:
    """Verify that a reference is a bounded regular file below a trusted root."""

    fd, _size = _open_relative_regular(Path(trusted_root), ref)
    os.close(fd)
    return ref


def _artifact_bytes(trusted_root: Path, ref: str) -> bytes:
    fd, expected_size = _open_relative_regular(trusted_root, ref)
    try:
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise ContractError(f"artifact changed while reading: {ref}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ContractError(f"artifact grew while reading: {ref}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _artifact_digest(trusted_root: Path, ref: str) -> str:
    return _sha256_digest(_artifact_bytes(trusted_root, ref))


_FAILURE_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "stage",
        "reason_code",
        "origin",
        "skill_logic_started",
        "agent",
        "process_exit_code",
        "process_signal",
        "http_status",
        "exception_type",
    }
)
_FAILURE_EVIDENCE_REQUIRED = frozenset(
    {"schema_version", "scope", "stage", "reason_code", "origin", "skill_logic_started"}
)


def validate_failure_evidence(
    evidence: object,
    *,
    expected: FailureRecord | None = None,
) -> None:
    """Validate the whitelist-only retained diagnostic sidecar."""

    canonical_json_bytes(evidence)
    obj = _expect_exact_fields(
        evidence,
        required=_FAILURE_EVIDENCE_REQUIRED,
        allowed=_FAILURE_EVIDENCE_FIELDS,
        name="failure_evidence",
    )
    if obj["schema_version"] != SCHEMA_VERSION:
        raise ContractError("failure_evidence.schema_version is unsupported")
    if not isinstance(obj["skill_logic_started"], bool):
        raise ContractError("failure_evidence.skill_logic_started must be boolean")
    failure = FailureRecord(
        scope=cast("Literal['run', 'agent']", obj["scope"]),
        stage=obj["stage"],
        reason_code=obj["reason_code"],
        origin=obj["origin"],
        agent=obj.get("agent"),
    )
    if failure.scope == "agent":
        started = obj["skill_logic_started"]
        if failure.origin == "trusted_execution_result":
            if not started:
                raise ContractError("trusted execution failure evidence requires skill_logic_started=true")
        elif started:
            raise ContractError("pre-semantic agent failure evidence requires skill_logic_started=false")
    if expected is not None and (
        failure.scope,
        failure.stage,
        failure.reason_code,
        failure.origin,
        failure.agent,
    ) != (expected.scope, expected.stage, expected.reason_code, expected.origin, expected.agent):
        raise ContractError("failure evidence identity does not match its typed failure")
    exit_code = obj.get("process_exit_code")
    signal = obj.get("process_signal")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int) or not 0 <= exit_code <= 255
    ):
        raise ContractError("failure_evidence.process_exit_code must be between 0 and 255")
    if signal is not None and (isinstance(signal, bool) or not isinstance(signal, int) or not 1 <= signal <= 64):
        raise ContractError("failure_evidence.process_signal must be between 1 and 64")
    if exit_code is not None and signal is not None:
        raise ContractError("failure evidence cannot contain both process exit code and signal")
    status = obj.get("http_status")
    if status is not None and (isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599):
        raise ContractError("failure_evidence.http_status must be between 100 and 599")
    if (
        failure.scope == "agent"
        and status == 400
        and (
            failure.origin != "trusted_adapter_marker"
            or (failure.stage, failure.reason_code)
            != ("agent_adapter_bootstrap", "adapter_model_protocol_negotiation_failed")
        )
    ):
        raise ContractError("agent-scoped HTTP 400 is permitted only for proven adapter protocol negotiation")
    exception_type = obj.get("exception_type")
    if exception_type is not None and (
        not isinstance(exception_type, str)
        or len(exception_type) > 256
        or not _SAFE_EXCEPTION_TYPE_RE.fullmatch(exception_type)
    ):
        raise ContractError("failure_evidence.exception_type is not a safe typed identifier")
    if failure.origin == "trusted_execution_result":
        expected_reason = _TRUSTED_AGENT_EXECUTION_EXCEPTIONS.get(str(exception_type or ""))
        if expected_reason != failure.reason_code:
            raise ContractError("trusted execution failure evidence requires a matching typed exception")


def write_failure_evidence(
    trusted_root: Path,
    relative_ref: str,
    failure: FailureRecord,
    *,
    skill_logic_started: bool,
    process_exit_code: int | None = None,
    process_signal: int | None = None,
    http_status: int | None = None,
    exception_type: str | None = None,
) -> str:
    """Exclusively publish one safe typed failure-evidence sidecar."""

    if not isinstance(failure, FailureRecord):
        raise ContractError("failure must be a typed FailureRecord")
    ref = _validated_ref_text(relative_ref)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "scope": failure.scope,
        "stage": failure.stage,
        "reason_code": failure.reason_code,
        "origin": failure.origin,
        "skill_logic_started": skill_logic_started,
    }
    if failure.agent is not None:
        payload["agent"] = failure.agent
    payload.update(
        {
            key: value
            for key, value in (
                ("process_exit_code", process_exit_code),
                ("process_signal", process_signal),
                ("http_status", http_status),
                ("exception_type", exception_type),
            )
            if value is not None
        }
    )
    validate_failure_evidence(payload, expected=failure)
    root = Path(trusted_root)
    return atomic_write_json(root.joinpath(*ref.split("/")), payload, trusted_root=root)


def _verify_failure_evidence_binding(
    trusted_root: Path,
    failure: FailureRecord,
) -> None:
    if failure.evidence_ref is None or failure.evidence_file_digest is None:
        raise ContractError("retained typed failure requires diagnostic reference and exact file digest")
    data = _artifact_bytes(trusted_root, failure.evidence_ref)
    if _sha256_digest(data) != failure.evidence_file_digest:
        raise ContractError("failure evidence file digest mismatch")
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ContractError("failure evidence is not valid JSON") from error
    if canonical_json_bytes(payload, trailing_newline=True) != data:
        raise ContractError("failure evidence must use canonical JSON bytes")
    validate_failure_evidence(payload, expected=failure)


def _task_plan_baseline_required(data: bytes) -> bool:
    try:
        plan = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"task plan is not valid UTF-8 JSON: {error}") from error
    if type(plan) is not dict:
        raise ContractError("task plan must be a JSON object with boolean baseline_required")
    baseline_required = cast("dict[str, object]", plan).get("baseline_required")
    if type(baseline_required) is not bool:
        raise ContractError("task plan baseline_required must be boolean")
    if canonical_json_bytes(plan, trailing_newline=True) != data:
        raise ContractError("task plan must use canonical JSON bytes")
    return cast("bool", baseline_required)


def _expect_exact_fields(
    value: object, *, required: frozenset[str], allowed: frozenset[str], name: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    keys = set(value)
    missing = required - keys
    unknown = keys - allowed
    if missing:
        raise ContractError(f"{name} is missing required fields: {sorted(missing)!r}")
    if unknown:
        raise ContractError(f"{name} has unexpected fields: {sorted(unknown)!r}")
    return value


def _parse_policy_object(value: object, name: str) -> CoveragePolicy:
    obj = _expect_exact_fields(value, required=_POLICY_FIELDS, allowed=_POLICY_FIELDS, name=name)
    mode = obj["mode"]
    minimum = obj["min_valid_agents"]
    if mode not in {"all_selected", "any_valid"}:
        raise ContractError(f"{name}.mode is unsupported")
    _require_positive_int(minimum, f"{name}.min_valid_agents")
    required = _require_unique_strings(obj["required_agents"], f"{name}.required_agents", identity=True)
    return CoveragePolicy(cast("Literal['all_selected', 'any_valid']", mode), cast("int", minimum), required)


def _validate_digest(value: object, name: str, *, nullable: bool) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase sha256 digest")
    return value


def _validate_nullable_ref(value: object, name: str) -> str | None:
    if value is None:
        return None
    try:
        return _validated_ref_text(value)
    except ContractError as error:
        raise ContractError(f"{name} is invalid: {error}") from error


def _validate_summary(value: object, name: str, *, allow_null: bool) -> dict[str, Any] | None:
    if value is None and allow_null:
        return None
    obj = _expect_exact_fields(value, required=_SUMMARY_FIELDS, allowed=_SUMMARY_FIELDS, name=name)
    expected = _require_positive_int(obj["expected_cases"], f"{name}.expected_cases", allow_zero=True)
    scored = _require_positive_int(obj["scored_cases"], f"{name}.scored_cases", allow_zero=True)
    exceptions = _require_positive_int(obj["exceptions"], f"{name}.exceptions", allow_zero=True)
    expected_attempts = _require_positive_int(obj["expected_attempts"], f"{name}.expected_attempts", allow_zero=True)
    scored_attempts = _require_positive_int(obj["scored_attempts"], f"{name}.scored_attempts", allow_zero=True)
    failed_attempts = _require_positive_int(obj["failed_attempts"], f"{name}.failed_attempts", allow_zero=True)
    skipped_attempts = _require_positive_int(obj["skipped_attempts"], f"{name}.skipped_attempts", allow_zero=True)
    not_run_attempts = _require_positive_int(obj["not_run_attempts"], f"{name}.not_run_attempts", allow_zero=True)
    if expected_attempts < expected:
        raise ContractError(f"{name}.expected_attempts cannot be less than expected_cases")
    if scored > expected or scored > scored_attempts:
        raise ContractError(f"{name}.scored_cases is inconsistent with case and attempt counts")
    if exceptions != failed_attempts:
        raise ContractError(f"{name}.exceptions must equal failed_attempts")
    observed_attempts = scored_attempts + failed_attempts + skipped_attempts + not_run_attempts
    if observed_attempts != expected_attempts:
        raise ContractError(f"{name} attempt disposition counters do not partition expected_attempts")
    return obj


def _manifest_occurrences(
    manifest: Mapping[str, Any],
    *,
    completed: bool,
    baseline_required: bool | None,
) -> tuple[AgentOccurrence, ...]:
    requested = _require_unique_strings(manifest["requested_agents"], "requested_agents", identity=True)
    agents_obj = manifest["agents"]
    if not isinstance(agents_obj, dict):
        raise ContractError("agents must be an occurrence-keyed object")
    if set(agents_obj) != set(requested):
        raise ContractError("agent-map keys must exactly equal requested_agents")

    occurrences: list[AgentOccurrence] = []
    for result_key in requested:
        name = f"agents.{result_key}"
        agent = _expect_exact_fields(
            agents_obj[result_key],
            required=_AGENT_REQUIRED_FIELDS,
            allowed=_AGENT_FIELDS,
            name=name,
        )
        occurrence = AgentOccurrence(
            result_key=result_key,
            base_agent=_require_identity(agent["base_agent"], f"{name}.base_agent"),
            occurrence=_require_positive_int(agent["occurrence"], f"{name}.occurrence"),
            requested_model=(
                None
                if agent["requested_model"] is None
                else _require_nonempty_text(agent["requested_model"], f"{name}.requested_model")
            ),
            resolved_model=_require_nonempty_text(agent["resolved_model"], f"{name}.resolved_model"),
            model_source=_require_nonempty_text(agent["model_source"], f"{name}.model_source"),
        )
        occurrences.append(occurrence)
        status_value = agent["status"]
        if status_value not in _AGENT_STATUSES:
            raise ContractError(f"{name}.status is unsupported")
        if not isinstance(agent["score_eligible"], bool):
            raise ContractError(f"{name}.score_eligible must be boolean")
        if (status_value == "valid") != agent["score_eligible"]:
            raise ContractError(f"{name}.score_eligible disagrees with agent status")
        if status_value in {"invalid_infrastructure", "invalid_configuration", "invalid_coverage"}:
            reason_code = _require_nonempty_text(agent.get("reason_code"), f"{name}.reason_code", limit=128)
            failure_stage = _require_nonempty_text(agent.get("failure_stage"), f"{name}.failure_stage", limit=128)
            if (failure_stage, reason_code) not in _AGENT_FAILURE_TAXONOMY:
                raise ContractError(f"{name} failure stage/reason is not in the trusted agent taxonomy")
            failure_origin = _require_nonempty_text(agent.get("failure_origin"), f"{name}.failure_origin", limit=64)
            FailureRecord(
                "agent",
                failure_stage,
                reason_code,
                origin=cast("FailureOrigin", failure_origin),
                agent=result_key,
            )
        elif status_value == "valid" and any(
            agent.get(key) is not None
            for key in (
                "reason_code",
                "failure_stage",
                "failure_origin",
                "evidence_ref",
                "evidence_file_digest",
            )
        ):
            raise ContractError(f"{name} valid entry cannot carry failure fields")
        evidence_ref = _validate_nullable_ref(agent.get("evidence_ref"), f"{name}.evidence_ref")
        evidence_file_digest = _validate_digest(
            agent.get("evidence_file_digest"), f"{name}.evidence_file_digest", nullable=True
        )
        if (evidence_ref is None) != (evidence_file_digest is None):
            raise ContractError(f"{name} diagnostic reference and file digest are all-or-none")
        if (
            completed
            and status_value in {"invalid_infrastructure", "invalid_configuration", "invalid_coverage"}
            and (evidence_ref is None or evidence_file_digest is None)
        ):
            raise ContractError(f"{name} completed invalid agent requires retained typed evidence")
        if evidence_ref is not None and completed is False and status_value == "valid":
            raise ContractError(f"{name} cannot attach failure evidence to a valid entry")
        viewer_url = agent.get("viewer_url")
        if viewer_url is not None:
            viewer = _require_nonempty_text(viewer_url, f"{name}.viewer_url", limit=4096)
            if not viewer.startswith(("https://", "http://")):
                raise ContractError(f"{name}.viewer_url must be HTTP(S)")
        with_skill = _validate_summary(agent["with_skill"], f"{name}.with_skill", allow_null=not completed)
        if completed:
            if baseline_required is None:
                raise ContractError("completed manifest requires verified task plan baseline_required")
            if baseline_required is True and agent["baseline"] is None:
                raise ContractError(f"{name}.baseline is required when task plan baseline_required is true")
            baseline = _validate_summary(
                agent["baseline"],
                f"{name}.baseline",
                allow_null=not baseline_required,
            )
            if baseline_required is False and baseline is not None:
                raise ContractError(f"{name}.baseline must be null when task plan baseline_required is false")
        else:
            baseline = _validate_summary(agent["baseline"], f"{name}.baseline", allow_null=True)
            if with_skill is not None or baseline is not None:
                raise ContractError(f"{name} noncompleted phase requires both arm summaries to be null")
        if agent["score_eligible"]:
            for arm_name, summary in (("with_skill", with_skill), ("baseline", baseline)):
                if arm_name == "baseline" and summary is None:
                    continue
                if (
                    summary is None
                    or summary["expected_cases"] <= 0
                    or summary["expected_attempts"] <= 0
                    or summary["scored_attempts"] <= 0
                ):
                    raise ContractError(f"{name} score-eligible {arm_name} arm requires a non-zero expected set")
                if (
                    summary["scored_cases"] != summary["expected_cases"]
                    or summary["scored_attempts"] + summary["skipped_attempts"] != summary["expected_attempts"]
                    or summary["failed_attempts"] != 0
                    or summary["exceptions"] != 0
                    or summary["not_run_attempts"] != 0
                ):
                    raise ContractError(f"{name} score-eligible {arm_name} arm is not complete")
    parsed = _validated_occurrences(occurrences)
    expected = resolve_occurrences(
        tuple(agent.base_agent for agent in parsed),
        requested_models=tuple(agent.requested_model for agent in parsed),
        resolved_models=tuple(agent.resolved_model for agent in parsed),
        model_sources=tuple(agent.model_source for agent in parsed),
    )
    if parsed != expected:
        raise ContractError("agent result-key identity does not match ordered base/occurrence identities")
    return parsed


def _parse_failures(values: object, name: str) -> tuple[FailureRecord, ...]:
    if not isinstance(values, list):
        raise ContractError(f"{name} must be a list")
    failures: list[FailureRecord] = []
    for index, value in enumerate(values):
        item_name = f"{name}[{index}]"
        obj = _expect_exact_fields(
            value,
            required=frozenset({"scope", "stage", "reason_code", "origin"}),
            allowed=_FAILURE_FIELDS,
            name=item_name,
        )
        failures.append(
            FailureRecord(
                scope=cast("Literal['run', 'agent']", obj["scope"]),
                stage=obj["stage"],
                reason_code=obj["reason_code"],
                origin=obj["origin"],
                agent=obj.get("agent"),
                evidence_ref=obj.get("evidence_ref"),
                evidence_file_digest=obj.get("evidence_file_digest"),
            )
        )
    return tuple(failures)


def _validate_warnings(values: object, requested: set[str]) -> tuple[dict[str, str], ...]:
    if not isinstance(values, list):
        raise ContractError("warnings must be a list")
    warnings: list[dict[str, str]] = []
    for index, value in enumerate(values):
        name = f"warnings[{index}]"
        obj = _expect_exact_fields(
            value,
            required=frozenset({"code"}),
            allowed=_WARNING_FIELDS,
            name=name,
        )
        parsed = {"code": _require_nonempty_text(obj["code"], f"{name}.code", limit=128)}
        if "agent" in obj:
            agent = _require_identity(obj["agent"], f"{name}.agent")
            if agent not in requested:
                raise ContractError(f"{name}.agent was not requested")
            parsed["agent"] = agent
        if "reason_code" in obj:
            parsed["reason_code"] = _require_nonempty_text(obj["reason_code"], f"{name}.reason_code", limit=128)
        if "failure_stage" in obj:
            parsed["failure_stage"] = _require_nonempty_text(obj["failure_stage"], f"{name}.failure_stage", limit=128)
        if "failure_origin" in obj:
            parsed["failure_origin"] = _require_nonempty_text(obj["failure_origin"], f"{name}.failure_origin", limit=64)
        if len({field for field in ("reason_code", "failure_stage", "failure_origin") if field in parsed}) not in {
            0,
            3,
        }:
            raise ContractError(f"{name} failure stage, reason_code, and origin are all-or-none")
        if "reason_code" in parsed:
            if "agent" not in parsed:
                raise ContractError(f"{name} typed agent failure warning requires agent")
            if (parsed["failure_stage"], parsed["reason_code"]) not in _AGENT_FAILURE_TAXONOMY:
                raise ContractError(f"{name} failure stage/reason is not in the trusted agent taxonomy")
            FailureRecord(
                "agent",
                parsed["failure_stage"],
                parsed["reason_code"],
                origin=cast("FailureOrigin", parsed["failure_origin"]),
                agent=parsed["agent"],
            )
        if "evidence_ref" in obj:
            parsed["evidence_ref"] = _validated_ref_text(obj["evidence_ref"])
        if "evidence_file_digest" in obj:
            parsed["evidence_file_digest"] = cast(
                "str",
                _validate_digest(obj["evidence_file_digest"], f"{name}.evidence_file_digest", nullable=False),
            )
        if ("evidence_ref" in parsed) != ("evidence_file_digest" in parsed):
            raise ContractError(f"{name} diagnostic reference and file digest are all-or-none")
        warnings.append(parsed)
    return tuple(warnings)


def _validate_capabilities(value: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    obj = _expect_exact_fields(
        value,
        required=_CAPABILITY_FIELDS,
        allowed=_CAPABILITY_FIELDS,
        name="capabilities",
    )
    requested = _require_unique_strings(obj["requested"], "capabilities.requested")
    provided = _require_unique_strings(obj["provided"], "capabilities.provided")
    unknown = (set(requested) | set(provided)) - {CAPABILITY}
    if unknown:
        raise ContractError(f"unsupported capability in manifest: {sorted(unknown)!r}")
    if CAPABILITY not in provided:
        raise ContractError(f"manifest producer did not provide {CAPABILITY}")
    if not set(requested).issubset(provided):
        raise ContractError("requested capabilities were not provided")
    return requested, provided


def _validate_extensions(value: object) -> None:
    if not isinstance(value, dict):
        raise ContractError("extensions must be an object")
    for key in value:
        if not isinstance(key, str) or not _EXTENSION_KEY_RE.fullmatch(key):
            raise ContractError(f"extensions key must be namespaced: {key!r}")
    canonical_json_bytes(value)


def _require_phase_group(
    manifest: Mapping[str, Any],
    *,
    phase_index: int,
    first_phase: str,
    fields: Sequence[str],
) -> None:
    required = phase_index >= _PHASE_INDEX[first_phase]
    for field_name in fields:
        present = manifest[field_name] is not None
        if present != required:
            state = "non-null" if required else "null"
            raise ContractError(f"phase {manifest['phase']!r} requires {field_name} to be {state}")


def validate_manifest(manifest: object, *, trusted_root: Path | None = None) -> None:
    """Validate schema shape, phase matrix, policy join, partitions, and refs."""

    canonical_json_bytes(manifest)
    obj = _expect_exact_fields(
        manifest,
        required=_ROOT_REQUIRED_FIELDS,
        allowed=_ROOT_FIELDS,
        name="agent_coverage",
    )
    schema_version = obj["schema_version"]
    if not isinstance(schema_version, str) or not _SCHEMA_VERSION_RE.fullmatch(schema_version):
        raise ContractError(f"unsupported agent-coverage schema version: {schema_version!r}")
    _require_nonempty_text(obj["run_id"], "run_id", limit=256)
    phase = obj["phase"]
    if phase not in _PHASE_INDEX:
        raise ContractError(f"unsupported terminal phase: {phase!r}")
    phase_index = _PHASE_INDEX[cast("str", phase)]
    status = obj["status"]
    if status not in _COVERAGE_STATUSES:
        raise ContractError(f"unsupported coverage status: {status!r}")
    requested_capabilities, _provided = _validate_capabilities(obj["capabilities"])
    if "extensions" in obj:
        _validate_extensions(obj["extensions"])

    _require_phase_group(
        obj,
        phase_index=phase_index,
        first_phase="dataset_validation",
        fields=(
            "requested_policy_digest",
            "effective_policy_digest",
            "requested_policy",
            "effective_policy",
            "policy_provenance",
        ),
    )
    if phase == "policy_validation" and obj["authorized_tightening"] is not None:
        raise ContractError("phase 'policy_validation' requires authorized_tightening to be null")
    _require_phase_group(
        obj,
        phase_index=phase_index,
        first_phase="task_generation",
        fields=("dataset_digest", "dataset_digest_algorithm"),
    )
    _require_phase_group(
        obj,
        phase_index=phase_index,
        first_phase="preflight",
        fields=("task_plan_digest", "task_plan_ref"),
    )
    _require_phase_group(
        obj,
        phase_index=phase_index,
        first_phase="execution",
        fields=("execution_ledger_digest", "execution_ledger_ref"),
    )

    requested_policy_digest = _validate_digest(obj["requested_policy_digest"], "requested_policy_digest", nullable=True)
    effective_policy_digest = _validate_digest(obj["effective_policy_digest"], "effective_policy_digest", nullable=True)
    task_plan_digest = _validate_digest(obj["task_plan_digest"], "task_plan_digest", nullable=True)
    execution_ledger_digest = _validate_digest(obj["execution_ledger_digest"], "execution_ledger_digest", nullable=True)
    _validate_digest(obj["dataset_digest"], "dataset_digest", nullable=True)
    task_plan_ref = _validate_nullable_ref(obj["task_plan_ref"], "task_plan_ref")
    ledger_ref = _validate_nullable_ref(obj["execution_ledger_ref"], "execution_ledger_ref")
    if obj["dataset_digest"] is not None and obj["dataset_digest_algorithm"] != DATASET_DIGEST_ALGORITHM:
        raise ContractError(f"dataset_digest_algorithm must be {DATASET_DIGEST_ALGORITHM}")

    referenced: list[tuple[str, str, str]] = []
    if task_plan_ref is not None and task_plan_digest is not None:
        referenced.append(("task_plan_digest", task_plan_ref, task_plan_digest))
    if ledger_ref is not None and execution_ledger_digest is not None:
        referenced.append(("execution_ledger_digest", ledger_ref, execution_ledger_digest))
    if referenced and trusted_root is None:
        raise ContractError("trusted_root is required to verify manifest artifact references")
    baseline_required: bool | None = None
    if trusted_root is not None:
        root = Path(trusted_root)
        for field, ref, expected_digest in referenced:
            artifact_bytes = _artifact_bytes(root, ref)
            if _sha256_digest(artifact_bytes) != expected_digest:
                raise ContractError(f"{field} does not match referenced artifact bytes")
            if field == "task_plan_digest":
                baseline_required = _task_plan_baseline_required(artifact_bytes)

    completed = phase == "completed"
    occurrences = _manifest_occurrences(
        obj,
        completed=completed,
        baseline_required=baseline_required,
    )
    requested_agents = tuple(agent.result_key for agent in occurrences)
    eligible_agents = _require_unique_strings(obj["eligible_agents"], "eligible_agents", identity=True)
    excluded_agents = _require_unique_strings(obj["excluded_agents"], "excluded_agents", identity=True)
    requested_set = set(requested_agents)
    warnings = _validate_warnings(obj["warnings"], requested_set)
    blockers = _parse_failures(obj["blockers"], "blockers")
    if any(blocker.scope != "run" for blocker in blockers):
        raise ContractError("top-level blockers may contain only run-scoped failures")

    policy_resolution: PolicyResolution | None = None
    if phase_index >= _PHASE_INDEX["dataset_validation"]:
        requested_policy = _parse_policy_object(obj["requested_policy"], "requested_policy")
        effective_policy = _parse_policy_object(obj["effective_policy"], "effective_policy")
        tightening = (
            None
            if obj["authorized_tightening"] is None
            else _parse_policy_object(obj["authorized_tightening"], "authorized_tightening")
        )
        policy_resolution = resolve_policy(
            requested_policy,
            occurrences,
            requested_capabilities,
            authorized_tightening=tightening,
        )
        if policy_resolution.requested != requested_policy:
            raise ContractError("requested_policy is not normalized")
        if policy_resolution.effective != effective_policy:
            raise ContractError("effective_policy does not equal deterministic trusted-policy join")
        if policy_resolution.authorized_tightening != tightening:
            raise ContractError("authorized_tightening is not normalized")
        if obj["policy_provenance"] != policy_resolution.provenance:
            raise ContractError("policy_provenance does not match policy resolution")
        if requested_policy_digest != policy_resolution.requested_digest:
            raise ContractError("requested_policy_digest does not match canonical policy object")
        if effective_policy_digest != policy_resolution.effective_digest:
            raise ContractError("effective_policy_digest does not match canonical policy object")

    decision: CoverageDecision | None = None
    if policy_resolution is not None:
        decision = calculate_coverage(
            policy_resolution.effective,
            requested_agents,
            eligible_agents,
            excluded_agents,
            blockers,
        )

    agents_obj = cast("dict[str, dict[str, Any]]", obj["agents"])
    for key in requested_agents:
        is_eligible = key in set(eligible_agents)
        if agents_obj[key]["score_eligible"] != is_eligible:
            raise ContractError(f"agents.{key}.score_eligible disagrees with eligible_agents")

    if completed:
        if decision is None or decision.status != status:
            raise ContractError("completed coverage status disagrees with policy and partition")
        if status == "valid_full" and excluded_agents:
            raise ContractError("valid_full cannot contain excluded agents")
        if status == "valid_degraded" and (not eligible_agents or not excluded_agents):
            raise ContractError("valid_degraded requires eligible and excluded agents")
        if status == "valid_degraded":
            if len(warnings) != len(excluded_agents):
                raise ContractError(
                    "valid_degraded requires exactly one optional_agent_excluded warning per excluded agent"
                )
            warning_agents = [warning.get("agent") for warning in warnings]
            if len(set(warning_agents)) != len(warning_agents) or set(warning_agents) != set(excluded_agents):
                raise ContractError(
                    "valid_degraded requires exactly one optional_agent_excluded warning per excluded agent"
                )
            by_agent = {warning["agent"]: warning for warning in warnings}
            for key in excluded_agents:
                agent = agents_obj[key]
                expected_warning = {
                    "code": "optional_agent_excluded",
                    "agent": key,
                    "reason_code": _require_nonempty_text(
                        agent.get("reason_code"), f"agents.{key}.reason_code", limit=128
                    ),
                    "failure_stage": _require_nonempty_text(
                        agent.get("failure_stage"), f"agents.{key}.failure_stage", limit=128
                    ),
                    "failure_origin": _require_nonempty_text(
                        agent.get("failure_origin"), f"agents.{key}.failure_origin", limit=64
                    ),
                    "evidence_ref": _validated_ref_text(agent.get("evidence_ref")),
                    "evidence_file_digest": cast(
                        "str",
                        _validate_digest(
                            agent.get("evidence_file_digest"),
                            f"agents.{key}.evidence_file_digest",
                            nullable=False,
                        ),
                    ),
                }
                if by_agent[key] != expected_warning:
                    raise ContractError(
                        f"optional_agent_excluded warning for {key!r} does not exactly match excluded agent evidence"
                    )
    else:
        if status != "invalid":
            raise ContractError(f"phase {phase!r} permits coverage status invalid only")
        if not blockers:
            raise ContractError(f"phase {phase!r} invalid result requires a typed run blocker")
        if eligible_agents or excluded_agents != requested_agents:
            raise ContractError(f"phase {phase!r} requires every requested agent to be excluded")
        if any(agents_obj[key]["status"] != "not_evaluated_run_blocked" for key in requested_agents):
            raise ContractError(f"phase {phase!r} requires blocked status for every requested agent")

    failure_bindings: list[FailureRecord] = [blocker for blocker in blockers if blocker.evidence_ref is not None]
    for warning in warnings:
        if "evidence_ref" in warning:
            failure_bindings.append(
                FailureRecord(
                    "agent",
                    warning["failure_stage"],
                    warning["reason_code"],
                    origin=cast("FailureOrigin", warning["failure_origin"]),
                    agent=warning["agent"],
                    evidence_ref=warning["evidence_ref"],
                    evidence_file_digest=warning["evidence_file_digest"],
                )
            )
    for key, agent in agents_obj.items():
        if agent.get("evidence_ref") is not None:
            failure_bindings.append(
                FailureRecord(
                    "agent",
                    agent["failure_stage"],
                    agent["reason_code"],
                    origin=cast("FailureOrigin", agent["failure_origin"]),
                    agent=key,
                    evidence_ref=agent["evidence_ref"],
                    evidence_file_digest=agent["evidence_file_digest"],
                )
            )
    if failure_bindings and trusted_root is None:
        raise ContractError("trusted_root is required to verify manifest artifact references")
    if trusted_root is not None:
        root = Path(trusted_root)
        seen_bindings: set[tuple[str, str]] = set()
        for failure in failure_bindings:
            assert failure.evidence_ref is not None
            assert failure.evidence_file_digest is not None
            binding = (failure.evidence_ref, failure.evidence_file_digest)
            if binding in seen_bindings:
                continue
            seen_bindings.add(binding)
            _verify_failure_evidence_binding(root, failure)


def write_manifest(trusted_root: Path, manifest: Mapping[str, object]) -> str:
    """Validate and exclusively publish ``agent_coverage.json``."""

    root = Path(trusted_root)
    validate_manifest(manifest, trusted_root=root)
    return atomic_write_json(root / "agent_coverage.json", manifest, trusted_root=root)


# Expected-attempt plan and execution-ledger contract -------------------------

_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "dataset_snapshot_kind",
        "dataset_digest_algorithm",
        "dataset_digest",
        "agents",
        "cases",
        "arm_task_sets",
        "baseline_required",
        "n_attempts",
        "stop_on_pass",
        "pass_threshold",
        "grading_mode",
        "reward_contract",
        "attempts",
    }
)
_REWARD_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "grading_mode",
        "metric_set",
        "default_metrics",
        "custom_metrics",
        "custom_grader_schema_digest",
        "combination_rule",
    }
)
_CUSTOM_METRIC_FIELDS = frozenset({"name", "range"})
_PLAN_AGENT_FIELDS = frozenset(
    {"result_key", "base_agent", "occurrence", "requested_model", "resolved_model", "model_source"}
)
_PLAN_CASE_FIELDS = frozenset({"case_id", "harbor_task_name", "reward_strategy"})
_ARM_TASK_SET_FIELDS = frozenset(
    {
        "arm",
        "root_ref",
        "digest_algorithm",
        "skill_payload_digest",
        "task_set_digest",
        "tasks",
    }
)
_ARM_TASK_FIELDS = frozenset({"case_id", "harbor_task_name", "reward_strategy", "staged_task_digest"})
_ATTEMPT_FIELDS = frozenset({"agent", "arm", "case_id", "ordinal"})
_JOB_EVIDENCE_FIELDS = frozenset(
    {
        "job_id",
        "job_name",
        "agent",
        "arm",
        "schedule_ref",
        "schedule_file_digest",
        "results_ref",
        "results_file_digest",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "schema_version",
        "plan_digest",
        "job_id",
        "job_name",
        "agent",
        "arm",
        "resolved_model",
        "harbor_model",
        "reward_contract_digest",
        "task_set_digest",
        "trials",
    }
)
_SCHEDULE_TRIAL_FIELDS = frozenset(
    {
        "trial_name",
        "agent",
        "arm",
        "case_id",
        "ordinal",
        "reward_strategy",
        "staged_task_digest",
    }
)
_RESULTS_FIELDS = frozenset(
    {
        "schema_version",
        "plan_digest",
        "schedule_file_digest",
        "job_id",
        "job_name",
        "agent",
        "arm",
        "resolved_model",
        "harbor_model",
        "reward_contract_digest",
        "task_set_digest",
        "trials",
    }
)
_RESULT_TRIAL_FIELDS = frozenset(
    {
        "trial_name",
        "agent",
        "arm",
        "case_id",
        "ordinal",
        "reward_strategy",
        "staged_task_digest",
        "state",
        "verifier_result_present",
        "rewards",
        "steps",
        "exception_type",
        "skill_logic_started",
        "agent_failure",
        "trial_ref",
        "trial_file_digest",
    }
)
_RESULT_STEP_FIELDS = frozenset({"step_name", "verifier_result_present", "rewards", "exception_type"})
_MINIMAL_TRIAL_FIELDS = frozenset(
    {
        "schema_version",
        "plan_digest",
        "job_id",
        *(_RESULT_TRIAL_FIELDS - {"trial_ref", "trial_file_digest"}),
    }
)


def _json_clone(value: object) -> Any:
    """Return a plain-JSON deep copy after canonical-domain validation."""

    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _exact_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    fd = -1
    try:
        before = Path(path).lstat()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        opened = os.fstat(fd)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mode)
        if not stat.S_ISREG(opened.st_mode) or identity != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mode,
        ):
            raise ContractError(f"semantic dataset file changed before hashing: {path}")
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mode) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
        ):
            raise ContractError(f"semantic dataset file changed while hashing: {path}")
    except OSError as error:
        raise ContractError(f"cannot hash semantic dataset file {path}: {error}") from error
    finally:
        if fd >= 0:
            os.close(fd)
    return f"sha256:{digest.hexdigest()}"


def _semantic_file_digest(value: bytes | Path, name: str) -> str:
    if isinstance(value, bytes):
        return _sha256_digest(value)
    if not isinstance(value, Path):
        raise ContractError(f"{name} must be bytes or a Path")
    try:
        entry = value.lstat()
    except OSError as error:
        raise ContractError(f"{name} cannot be inspected: {error}") from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ContractError(f"{name} must be a regular non-symlink file")
    return _exact_file_sha256(value)


def build_evals_json_snapshot(
    *,
    entries: Sequence[Mapping[str, Any]],
    evaluation_config: Mapping[str, Any],
    referenced_files: Mapping[str, bytes | Path],
) -> dict[str, object]:
    """Build the complete variant-neutral semantic snapshot for evals JSON."""

    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence) or not entries:
        raise ContractError("evals_json snapshot requires a non-empty ordered entries list")
    parsed_entries: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ContractError(f"entries[{index}] must be an object")
        parsed = _json_clone(dict(entry))
        case_id = _require_nonempty_text(parsed.get("id"), f"entries[{index}].id", limit=256)
        if case_id in case_ids:
            raise ContractError(f"evals_json snapshot contains duplicate case ID {case_id!r}")
        case_ids.add(case_id)
        parsed_entries.append(parsed)

    if not isinstance(evaluation_config, Mapping):
        raise ContractError("evaluation_config must be an object")
    normalized_config = _json_clone(dict(evaluation_config))
    file_rows: list[dict[str, str]] = []
    for raw_ref in sorted(referenced_files):
        ref = _validated_ref_text(raw_ref)
        file_rows.append(
            {
                "path": ref,
                "file_digest": _semantic_file_digest(referenced_files[raw_ref], f"referenced_files[{ref!r}]"),
            }
        )
    snapshot: dict[str, object] = {
        "kind": "evals_json",
        "entries": parsed_entries,
        "evaluation_config": normalized_config,
        "referenced_files": file_rows,
    }
    canonical_json_bytes(snapshot)
    return snapshot


def build_native_harbor_snapshot(dataset_root: Path, *, task_ids: Sequence[str]) -> dict[str, object]:
    """Build the complete semantic snapshot for native Harbor source tasks."""

    root = Path(dataset_root)
    if root.is_symlink() or not root.is_dir():
        raise ContractError("native Harbor dataset root must be a regular directory")
    parsed_ids = _require_unique_strings(task_ids, "task_ids")
    if not parsed_ids:
        raise ContractError("native Harbor snapshot requires at least one task")
    task_rows: list[dict[str, object]] = []
    for task_id in parsed_ids:
        relative = _validated_ref_text(task_id)
        if "/" in relative:
            raise ContractError("native Harbor task IDs must be direct repository-relative directories")
        task_dir = root / relative
        if task_dir.is_symlink() or not task_dir.is_dir():
            raise ContractError(f"native Harbor task is not a regular directory: {task_id}")
        task_toml = task_dir / "task.toml"
        if task_toml.is_symlink() or not task_toml.is_file():
            raise ContractError(f"native Harbor task {task_id!r} has no regular task.toml")
        try:
            config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise ContractError(f"native Harbor task {task_id!r} has invalid task.toml: {error}") from error
        files: list[dict[str, str]] = []
        for path in sorted(task_dir.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            relative_path = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ContractError(f"native Harbor semantic file cannot be a symlink: {relative_path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ContractError(f"native Harbor semantic entry is not a regular file: {relative_path}")
            files.append({"path": _validated_ref_text(relative_path), "file_digest": _exact_file_sha256(path)})
        task_rows.append({"task_id": task_id, "config": _json_clone(config), "files": files})
    snapshot: dict[str, object] = {
        "kind": "native_harbor",
        "task_ids": list(parsed_ids),
        "tasks": task_rows,
    }
    canonical_json_bytes(snapshot)
    return snapshot


def build_harbor_case_map(task_paths: Sequence[Path], *, case_ids: Sequence[str] | None = None) -> list[dict[str, str]]:
    """Resolve each staged task's logical case ID and pinned reward strategy."""

    if isinstance(task_paths, (str, bytes)) or not isinstance(task_paths, Sequence) or not task_paths:
        raise ContractError("task_paths must be a non-empty ordered list")
    paths = [Path(path) for path in task_paths]
    if case_ids is not None:
        if isinstance(case_ids, (str, bytes)) or len(case_ids) != len(paths):
            raise ContractError("case_ids must have exactly one value per staged task")
        resolved_case_ids = [
            _require_nonempty_text(value, f"case_ids[{index}]", limit=256) for index, value in enumerate(case_ids)
        ]
    else:
        resolved_case_ids = []
        for path in paths:
            try:
                config = tomllib.loads((path / "task.toml").read_text(encoding="utf-8"))
            except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
                raise ContractError(f"cannot resolve staged task config for {path}: {error}") from error
            metadata = config.get("metadata")
            raw_case = metadata.get("entry_id") if isinstance(metadata, dict) else None
            resolved_case_ids.append(str(raw_case) if raw_case is not None else path.name)
    if len(set(resolved_case_ids)) != len(resolved_case_ids):
        raise ContractError("staged case map contains duplicate case IDs")

    rows: list[dict[str, str]] = []
    staged_names: set[str] = set()
    for index, (path, case_id) in enumerate(zip(paths, resolved_case_ids, strict=True)):
        task_name = _require_identity(path.name, f"task_paths[{index}].name")
        task_name_key = normalized_identity_key(task_name)
        if task_name_key in staged_names:
            raise ContractError("staged Harbor task-name collision")
        staged_names.add(task_name_key)
        task_toml = path / "task.toml"
        if task_toml.is_symlink() or not task_toml.is_file():
            raise ContractError(f"staged Harbor task has no regular task.toml: {task_name}")
        try:
            config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise ContractError(f"staged Harbor task {task_name!r} has invalid task.toml: {error}") from error
        steps = config.get("steps")
        if isinstance(steps, list) and steps:
            raw_strategy = config.get("multi_step_reward_strategy", "mean")
            if raw_strategy == "mean":
                strategy = "multi_step_mean"
            elif raw_strategy == "final":
                strategy = "multi_step_final"
            else:
                raise ContractError(f"staged Harbor task {task_name!r} has unknown reward strategy")
        else:
            strategy = "single_step"
        rows.append(
            {
                "case_id": case_id,
                "harbor_task_name": task_name,
                "reward_strategy": strategy,
            }
        )
    return rows


def staged_task_digest(task_dir: Path) -> str:
    """Digest one fully staged Harbor task without following symlinks."""

    root = Path(task_dir)
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise ContractError(f"staged task cannot be inspected: {error}") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ContractError("staged task must be a regular non-symlink directory")
    files: list[dict[str, object]] = []
    total_bytes = 0
    normalized_paths: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        try:
            entry = path.lstat()
        except OSError as error:
            raise ContractError(f"staged task entry cannot be inspected: {relative}: {error}") from error
        if stat.S_ISLNK(entry.st_mode):
            raise ContractError(f"staged task entry cannot be a symlink: {relative}")
        if stat.S_ISDIR(entry.st_mode):
            continue
        if not stat.S_ISREG(entry.st_mode):
            raise ContractError(f"staged task entry must be a regular file: {relative}")
        if entry.st_nlink != 1:
            raise ContractError(f"staged task entry cannot be hard-linked: {relative}")
        canonical_relative = _validated_ref_text(relative)
        collision_key = unicodedata.normalize("NFC", canonical_relative).casefold()
        if collision_key in normalized_paths:
            raise ContractError(f"staged task paths collide after case folding: {relative}")
        normalized_paths.add(collision_key)
        total_bytes += entry.st_size
        if len(files) >= MAX_STAGED_TREE_FILES or total_bytes > MAX_STAGED_TREE_BYTES:
            raise ContractError("staged task tree exceeds the bounded c14n limits")
        files.append(
            {
                "path": canonical_relative,
                "size": entry.st_size,
                "executable": bool(entry.st_mode & 0o111),
                "file_digest": _exact_file_sha256(path),
            }
        )
    if not files:
        raise ContractError("staged task must contain at least one regular file")
    return canonical_digest({"digest_algorithm": STAGED_TASK_DIGEST_ALGORITHM, "files": files})


def staged_tree_digest(tree_root: Path) -> str:
    """Digest an arbitrary staged payload tree with the task c14n algorithm."""

    return staged_task_digest(tree_root)


def build_staged_arm_task_set(
    trusted_root: Path,
    *,
    arm: Arm,
    task_root: Path,
    cases: Sequence[Mapping[str, str]],
    skill_payload_path: Path | None,
) -> dict[str, object]:
    """Bind one arm's exact staged root and ordered task bytes into the plan."""

    if arm not in {"with_skill", "baseline"}:
        raise ContractError("staged task-set arm is unsupported")
    root = Path(trusted_root)
    task_root_path = Path(task_root)
    try:
        root_resolved = root.resolve(strict=True)
        task_root_resolved = task_root_path.resolve(strict=True)
        root_ref = task_root_resolved.relative_to(root_resolved).as_posix()
    except (OSError, ValueError) as error:
        raise ContractError(f"staged task root must be inside the trusted run root: {error}") from error
    _validated_ref_text(root_ref)
    if root_ref != f"staged/{arm}":
        raise ContractError("arm task roots must use the disjoint fixed staged/<arm> references")
    if task_root_path.is_symlink() or not task_root_path.is_dir():
        raise ContractError("staged task root must be a regular non-symlink directory")
    parsed_cases = [_parse_plan_case(case, index) for index, case in enumerate(cases)]
    expected_task_names = {case["harbor_task_name"] for case in parsed_cases}
    try:
        root_entries = list(task_root_path.iterdir())
    except OSError as error:
        raise ContractError(f"staged task root cannot be enumerated: {error}") from error
    actual_task_names = {entry.name for entry in root_entries}
    if actual_task_names != expected_task_names or any(
        entry.is_symlink() or not entry.is_dir() for entry in root_entries
    ):
        raise ContractError("staged arm root contains an extra, missing, or unsafe task entry")
    tasks: list[dict[str, str]] = []
    for case in parsed_cases:
        task_path = task_root_path / case["harbor_task_name"]
        try:
            if task_path.resolve(strict=True).parent != task_root_resolved:
                raise ContractError("staged task path escapes its arm root")
        except OSError as error:
            raise ContractError(f"staged task cannot be resolved: {error}") from error
        tasks.append({**case, "staged_task_digest": staged_task_digest(task_path)})
    if arm == "with_skill":
        if skill_payload_path is None:
            raise ContractError("with_skill arm requires an exact staged skill payload")
        skill_payload_digest: str | None = staged_tree_digest(skill_payload_path)
    else:
        if skill_payload_path is not None:
            raise ContractError("baseline arm must have an explicit-null skill payload")
        skill_payload_digest = None
    core = {
        "arm": arm,
        "root_ref": root_ref,
        "digest_algorithm": STAGED_TASK_DIGEST_ALGORITHM,
        "skill_payload_digest": skill_payload_digest,
        "tasks": tasks,
    }
    return {**core, "task_set_digest": canonical_digest(core)}


def verify_staged_arm_task_roots(
    trusted_root: Path,
    plan: Mapping[str, Any],
) -> dict[Arm, tuple[Path, tuple[Path, ...]]]:
    """Rehash the exact immutable staged task roots bound by a valid plan."""

    validate_expected_attempt_plan(plan)
    root = Path(trusted_root)
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"trusted run root cannot be resolved: {error}") from error
    verified: dict[Arm, tuple[Path, tuple[Path, ...]]] = {}
    for raw_task_set in cast("list[dict[str, Any]]", plan["arm_task_sets"]):
        arm = cast("Arm", raw_task_set["arm"])
        root_ref = _validated_ref_text(raw_task_set["root_ref"])
        task_root = root.joinpath(*root_ref.split("/"))
        verified[arm] = verify_staged_task_root_against_plan(
            canonical_root,
            task_root,
            plan,
            arm=arm,
        )
    return verified


def verify_staged_task_root_against_plan(
    trusted_root: Path,
    task_root: Path,
    plan: Mapping[str, Any],
    *,
    arm: Arm,
) -> tuple[Path, tuple[Path, ...]]:
    """Rehash any confined staged root against one arm of a valid plan."""

    validate_expected_attempt_plan(plan)
    if arm not in {"with_skill", "baseline"}:
        raise ContractError("staged task-set arm is unsupported")
    raw_task_set = next(
        (item for item in cast("list[dict[str, Any]]", plan["arm_task_sets"]) if item["arm"] == arm),
        None,
    )
    if raw_task_set is None:
        raise ContractError(f"immutable plan does not bind a staged {arm} root")

    root = Path(trusted_root)
    task_root = Path(task_root)
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"trusted run root cannot be resolved: {error}") from error
    try:
        entry = task_root.lstat()
        resolved_task_root = task_root.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"staged {arm} root cannot be inspected: {error}") from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise ContractError(f"staged {arm} root must be a regular non-symlink directory")
    try:
        resolved_task_root.relative_to(canonical_root)
    except ValueError as error:
        raise ContractError(f"staged {arm} root escapes the trusted run root") from error

    expected_tasks = cast("list[dict[str, str]]", raw_task_set["tasks"])
    expected_names = [task["harbor_task_name"] for task in expected_tasks]
    try:
        actual_entries = list(task_root.iterdir())
    except OSError as error:
        raise ContractError(f"staged {arm} root cannot be enumerated: {error}") from error
    if {entry.name for entry in actual_entries} != set(expected_names):
        raise ContractError(f"staged {arm} root contains an extra or missing task")

    task_paths: list[Path] = []
    for task in expected_tasks:
        task_path = task_root / task["harbor_task_name"]
        try:
            task_entry = task_path.lstat()
            resolved_task = task_path.resolve(strict=True)
        except OSError as error:
            raise ContractError(f"staged {arm} task cannot be inspected: {error}") from error
        if stat.S_ISLNK(task_entry.st_mode) or not stat.S_ISDIR(task_entry.st_mode):
            raise ContractError(f"staged {arm} task must be a regular non-symlink directory")
        if resolved_task.parent != resolved_task_root:
            raise ContractError(f"staged {arm} task escapes its immutable root")
        if staged_task_digest(task_path) != task["staged_task_digest"]:
            raise ContractError(f"staged {arm} task digest disagrees with the immutable plan")
        task_paths.append(task_path)
    return task_root, tuple(task_paths)


def _parse_arm_task_set(value: object, index: int) -> dict[str, object]:
    obj = _expect_exact_fields(
        value,
        required=_ARM_TASK_SET_FIELDS,
        allowed=_ARM_TASK_SET_FIELDS,
        name=f"arm_task_sets[{index}]",
    )
    if obj["arm"] not in {"with_skill", "baseline"}:
        raise ContractError(f"arm_task_sets[{index}].arm is unsupported")
    if obj["digest_algorithm"] != STAGED_TASK_DIGEST_ALGORITHM:
        raise ContractError(f"arm_task_sets[{index}].digest_algorithm is unsupported")
    payload_digest = _validate_digest(
        obj["skill_payload_digest"],
        f"arm_task_sets[{index}].skill_payload_digest",
        nullable=True,
    )
    if (obj["arm"] == "with_skill") != (payload_digest is not None):
        raise ContractError("with_skill requires a payload digest and baseline requires null")
    root_ref = _validated_ref_text(obj["root_ref"])
    if root_ref != f"staged/{obj['arm']}":
        raise ContractError("arm task roots must use the disjoint fixed staged/<arm> references")
    raw_tasks = obj["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ContractError(f"arm_task_sets[{index}].tasks must be a non-empty list")
    tasks: list[dict[str, str]] = []
    normalized_task_names: set[str] = set()
    for task_index, raw_task in enumerate(raw_tasks):
        task = _expect_exact_fields(
            raw_task,
            required=_ARM_TASK_FIELDS,
            allowed=_ARM_TASK_FIELDS,
            name=f"arm_task_sets[{index}].tasks[{task_index}]",
        )
        parsed = _parse_plan_case({field: task[field] for field in _PLAN_CASE_FIELDS}, task_index)
        task_name_key = normalized_identity_key(parsed["harbor_task_name"])
        if task_name_key in normalized_task_names:
            raise ContractError("arm task set contains a normalized task-name collision")
        normalized_task_names.add(task_name_key)
        digest = _validate_digest(
            task["staged_task_digest"],
            f"arm_task_sets[{index}].tasks[{task_index}].staged_task_digest",
            nullable=False,
        )
        tasks.append({**parsed, "staged_task_digest": cast("str", digest)})
    core = {
        "arm": obj["arm"],
        "root_ref": root_ref,
        "digest_algorithm": STAGED_TASK_DIGEST_ALGORITHM,
        "skill_payload_digest": payload_digest,
        "tasks": tasks,
    }
    digest = _validate_digest(obj["task_set_digest"], f"arm_task_sets[{index}].task_set_digest", nullable=False)
    if digest != canonical_digest(core):
        raise ContractError(f"arm_task_sets[{index}].task_set_digest is stale")
    return {**core, "task_set_digest": digest}


def _validate_snapshot_against_plan_cases(
    snapshot: Mapping[str, Any],
    *,
    kind: DatasetSnapshotKind,
    cases: Sequence[Mapping[str, str]],
    grading_mode: str,
) -> None:
    case_ids = [case["case_id"] for case in cases]
    if kind == "evals_json":
        entries = snapshot.get("entries")
        if (
            not isinstance(entries, list)
            or [entry.get("id") for entry in entries if isinstance(entry, dict)] != case_ids
        ):
            raise ContractError("evals semantic snapshot case IDs do not match the plan case map")
        config = snapshot.get("evaluation_config")
        if not isinstance(config, dict):
            raise ContractError("evals semantic snapshot evaluation_config is malformed")
        snapshot_mode = config.get("grading_mode")
        if snapshot_mode is not None and snapshot_mode != grading_mode:
            raise ContractError("semantic snapshot grading mode does not match the immutable plan")
        return
    tasks = snapshot.get("tasks")
    if not isinstance(tasks, list):
        raise ContractError("native Harbor semantic snapshot tasks are malformed")
    derived: list[tuple[str, str, str]] = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("config"), dict):
            raise ContractError("native Harbor semantic snapshot task is malformed")
        config = task["config"]
        metadata_value = config.get("metadata")
        raw_case_id = metadata_value.get("entry_id") if isinstance(metadata_value, dict) else None
        case_id = str(raw_case_id) if raw_case_id is not None else task.get("task_id")
        steps = config.get("steps")
        if isinstance(steps, list) and steps:
            raw_strategy = config.get("multi_step_reward_strategy", "mean")
            strategy = {"mean": "multi_step_mean", "final": "multi_step_final"}.get(raw_strategy)
            if strategy is None:
                raise ContractError("native Harbor semantic snapshot has an unknown reward strategy")
        else:
            strategy = "single_step"
        derived.append((case_id, task.get("task_id"), strategy))
    expected = [(case["case_id"], case["harbor_task_name"], case["reward_strategy"]) for case in cases]
    if derived != expected:
        raise ContractError("native Harbor semantic snapshot does not match the staged case map")


def _parse_plan_case(value: object, index: int) -> dict[str, str]:
    obj = _expect_exact_fields(value, required=_PLAN_CASE_FIELDS, allowed=_PLAN_CASE_FIELDS, name=f"cases[{index}]")
    strategy = obj["reward_strategy"]
    if strategy not in {"single_step", "multi_step_mean", "multi_step_final"}:
        raise ContractError(f"cases[{index}].reward_strategy is unsupported")
    return {
        "case_id": _require_nonempty_text(obj["case_id"], f"cases[{index}].case_id", limit=256),
        "harbor_task_name": _require_identity(obj["harbor_task_name"], f"cases[{index}].harbor_task_name"),
        "reward_strategy": cast("str", strategy),
    }


def _attempt_object(attempt: ExpectedAttempt) -> dict[str, object]:
    return {
        "agent": attempt.agent,
        "arm": attempt.arm,
        "case_id": attempt.case_id,
        "ordinal": attempt.ordinal,
    }


def build_reward_contract(
    grading_mode: Literal["default", "default_plus_custom", "custom_only"],
    *,
    custom_metrics: Sequence[Mapping[str, object]] = (),
    custom_grader_schema_digest: str | None = None,
) -> dict[str, object]:
    """Build the closed numeric reward allowlist bound before execution."""

    if grading_mode not in {"default", "default_plus_custom", "custom_only"}:
        raise ContractError(f"unsupported grading_mode: {grading_mode!r}")
    parsed_custom: list[dict[str, object]] = []
    for index, raw in enumerate(custom_metrics):
        item = _expect_exact_fields(
            raw,
            required=_CUSTOM_METRIC_FIELDS,
            allowed=_CUSTOM_METRIC_FIELDS,
            name=f"reward_contract.custom_metrics[{index}]",
        )
        name = _require_nonempty_text(item["name"], f"reward_contract.custom_metrics[{index}].name", limit=128)
        if not _SAFE_METRIC_NAME_RE.fullmatch(name) or name in RESERVED_METRIC_NAMES or name == "reward":
            raise ContractError("reward contract contains an unsafe or reserved custom metric name")
        if item["range"] != "unit_interval":
            raise ContractError("reward contract custom metric range is unsupported")
        parsed_custom.append({"name": name, "range": "unit_interval"})
    names = [cast("str", item["name"]) for item in parsed_custom]
    if len(names) != len(set(names)):
        raise ContractError("reward contract custom metric names must be unique")

    schema_digest = _validate_digest(
        custom_grader_schema_digest,
        "reward_contract.custom_grader_schema_digest",
        nullable=True,
    )
    if grading_mode == "default":
        if parsed_custom or schema_digest is not None:
            raise ContractError("SkillEvaluator-default reward contract cannot declare a custom grader")
        metric_set = DEFAULT_METRIC_SET
        default_metrics = list(DEFAULT_METRICS)
        combination_rule = "default_mean"
    elif grading_mode == "default_plus_custom":
        if len(parsed_custom) > 249:
            raise ContractError("SkillEvaluator-plus-custom cannot exceed 249 custom metrics")
        if not parsed_custom or schema_digest is None:
            raise ContractError(
                "SkillEvaluator-plus-custom reward contract requires declared metrics and a schema digest"
            )
        metric_set = DEFAULT_METRIC_SET
        default_metrics = list(DEFAULT_METRICS)
        combination_rule = "default_mean_with_advisory_custom"
    else:
        if len(parsed_custom) > 255:
            raise ContractError("custom-only cannot exceed 255 custom metrics")
        if parsed_custom and schema_digest is None:
            raise ContractError("custom-only declared metrics require a custom grader schema digest")
        if not parsed_custom and schema_digest is not None:
            raise ContractError("custom-only reward contract cannot retain an unused schema digest")
        metric_set = CUSTOM_ONLY_METRIC_SET
        default_metrics = []
        combination_rule = "custom_overall"
    return {
        "schema_version": SCHEMA_VERSION,
        "grading_mode": grading_mode,
        "metric_set": metric_set,
        "default_metrics": default_metrics,
        "custom_metrics": parsed_custom,
        "custom_grader_schema_digest": schema_digest,
        "combination_rule": combination_rule,
    }


def validate_reward_contract(value: object) -> dict[str, object]:
    obj = _expect_exact_fields(
        value,
        required=_REWARD_CONTRACT_FIELDS,
        allowed=_REWARD_CONTRACT_FIELDS,
        name="reward_contract",
    )
    if not isinstance(obj["custom_metrics"], list):
        raise ContractError("reward_contract.custom_metrics must be a list")
    rebuilt = build_reward_contract(
        cast("Any", obj["grading_mode"]),
        custom_metrics=cast("list[Mapping[str, object]]", obj["custom_metrics"]),
        custom_grader_schema_digest=cast("str | None", obj["custom_grader_schema_digest"]),
    )
    if obj != rebuilt:
        raise ContractError("reward_contract fields disagree with canonical grading semantics")
    return rebuilt


def build_expected_attempt_plan(
    *,
    run_id: str,
    dataset_snapshot_kind: DatasetSnapshotKind,
    semantic_snapshot: Mapping[str, Any],
    agents: Sequence[AgentOccurrence],
    cases: Sequence[Mapping[str, str]],
    baseline_required: bool,
    n_attempts: int,
    stop_on_pass: bool,
    pass_threshold: float,
    grading_mode: Literal["default", "default_plus_custom", "custom_only"],
    reward_contract: Mapping[str, object] | None = None,
    arm_task_sets: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """Construct the immutable full Cartesian attempt plan."""

    _require_nonempty_text(run_id, "run_id", limit=256)
    if dataset_snapshot_kind not in {"evals_json", "native_harbor"}:
        raise ContractError(f"unsupported dataset snapshot kind: {dataset_snapshot_kind!r}")
    snapshot = _json_clone(dict(semantic_snapshot))
    if snapshot.get("kind") != dataset_snapshot_kind:
        raise ContractError("semantic snapshot kind does not match dataset_snapshot_kind")
    occurrences = _validated_occurrences(agents)
    if len(occurrences) > 128:
        raise ContractError("expected attempt plan cannot exceed 128 agent occurrences")
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence) or not cases:
        raise ContractError("cases must be a non-empty ordered list")
    if len(cases) > 100_000:
        raise ContractError("expected attempt plan cannot exceed 100000 cases")
    parsed_cases = [_parse_plan_case(value, index) for index, value in enumerate(cases)]
    case_ids = [case["case_id"] for case in parsed_cases]
    task_names = [case["harbor_task_name"] for case in parsed_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ContractError("case map contains duplicate case IDs")
    if len(task_names) != len({normalized_identity_key(name) for name in task_names}):
        raise ContractError("case map has a staged Harbor task-name collision")
    if not isinstance(baseline_required, bool) or not isinstance(stop_on_pass, bool):
        raise ContractError("baseline_required and stop_on_pass must be booleans")
    attempts_count = _require_positive_int(n_attempts, "n_attempts")
    if attempts_count > 1000:
        raise ContractError("n_attempts cannot exceed 1000")
    if isinstance(pass_threshold, bool) or not isinstance(pass_threshold, int | float):
        raise ContractError("pass_threshold must be a finite number")
    threshold = float(pass_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ContractError("pass_threshold must be finite and between 0 and 1")
    if grading_mode not in {"default", "default_plus_custom", "custom_only"}:
        raise ContractError(f"unsupported grading_mode: {grading_mode!r}")
    parsed_reward_contract = validate_reward_contract(
        build_reward_contract(grading_mode) if reward_contract is None else reward_contract
    )
    if parsed_reward_contract["grading_mode"] != grading_mode:
        raise ContractError("reward contract grading mode disagrees with the plan")

    arms: tuple[Arm, ...] = ("with_skill", "baseline") if baseline_required else ("with_skill",)
    attempt_product = len(occurrences) * len(arms) * attempts_count * len(parsed_cases)
    if attempt_product > 1_000_000:
        raise ContractError("expected attempt plan cannot exceed 1000000 attempts")
    _validate_snapshot_against_plan_cases(
        snapshot,
        kind=dataset_snapshot_kind,
        cases=parsed_cases,
        grading_mode=grading_mode,
    )
    if arm_task_sets is None or isinstance(arm_task_sets, (str, bytes)):
        raise ContractError("arm_task_sets must bind every planned arm")
    parsed_task_sets = [_parse_arm_task_set(value, index) for index, value in enumerate(arm_task_sets)]
    if [item["arm"] for item in parsed_task_sets] != list(arms):
        raise ContractError("arm_task_sets must exactly match the planned arm order")
    expected_case_rows = [(case["case_id"], case["harbor_task_name"], case["reward_strategy"]) for case in parsed_cases]
    if any(
        normalized_refs_overlap(left["root_ref"], right["root_ref"])
        for index, left in enumerate(parsed_task_sets)
        for right in parsed_task_sets[index + 1 :]
    ):
        raise ContractError("planned arm task roots must be disjoint")
    for item in parsed_task_sets:
        actual_case_rows = [
            (task["case_id"], task["harbor_task_name"], task["reward_strategy"])
            for task in cast("list[dict[str, Any]]", item["tasks"])
        ]
        if actual_case_rows != expected_case_rows:
            raise ContractError("arm task-set cases do not exactly match the immutable case map")
    expected = [
        ExpectedAttempt(agent.result_key, arm, case["case_id"], ordinal)
        for agent in occurrences
        for arm in arms
        for ordinal in range(1, attempts_count + 1)
        for case in parsed_cases
    ]
    plan: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "dataset_snapshot_kind": dataset_snapshot_kind,
        "dataset_digest_algorithm": DATASET_DIGEST_ALGORITHM,
        "dataset_digest": canonical_digest(snapshot),
        "agents": [
            {
                "result_key": agent.result_key,
                "base_agent": agent.base_agent,
                "occurrence": agent.occurrence,
                "requested_model": agent.requested_model,
                "resolved_model": agent.resolved_model,
                "model_source": agent.model_source,
            }
            for agent in occurrences
        ],
        "cases": parsed_cases,
        "arm_task_sets": parsed_task_sets,
        "baseline_required": baseline_required,
        "n_attempts": attempts_count,
        "stop_on_pass": stop_on_pass,
        "pass_threshold": threshold,
        "grading_mode": grading_mode,
        "reward_contract": parsed_reward_contract,
        "attempts": [_attempt_object(item) for item in expected],
    }
    validate_expected_attempt_plan(plan)
    return plan


def validate_expected_attempt_plan(plan: object) -> None:
    """Validate the closed plan shape and its exact ordered Cartesian product."""

    canonical_json_bytes(plan)
    obj = _expect_exact_fields(plan, required=_PLAN_FIELDS, allowed=_PLAN_FIELDS, name="expected_attempt_plan")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise ContractError("expected_attempt_plan.schema_version is unsupported")
    _require_nonempty_text(obj["run_id"], "expected_attempt_plan.run_id", limit=256)
    if obj["dataset_snapshot_kind"] not in {"evals_json", "native_harbor"}:
        raise ContractError("expected_attempt_plan.dataset_snapshot_kind is unsupported")
    if obj["dataset_digest_algorithm"] != DATASET_DIGEST_ALGORITHM:
        raise ContractError("expected_attempt_plan.dataset_digest_algorithm is unsupported")
    _validate_digest(obj["dataset_digest"], "expected_attempt_plan.dataset_digest", nullable=False)

    raw_agents = obj["agents"]
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ContractError("expected_attempt_plan.agents must be a non-empty list")
    if len(raw_agents) > 128:
        raise ContractError("expected_attempt_plan.agents exceeds 128 occurrences")
    occurrences: list[AgentOccurrence] = []
    for index, raw in enumerate(raw_agents):
        item = _expect_exact_fields(
            raw,
            required=_PLAN_AGENT_FIELDS,
            allowed=_PLAN_AGENT_FIELDS,
            name=f"expected_attempt_plan.agents[{index}]",
        )
        occurrences.append(
            AgentOccurrence(
                result_key=item["result_key"],
                base_agent=item["base_agent"],
                occurrence=item["occurrence"],
                requested_model=item["requested_model"],
                resolved_model=item["resolved_model"],
                model_source=item["model_source"],
            )
        )
    parsed_occurrences = _validated_occurrences(occurrences)

    raw_cases = obj["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ContractError("expected_attempt_plan.cases must be a non-empty list")
    if len(raw_cases) > 100_000:
        raise ContractError("expected_attempt_plan.cases exceeds 100000 cases")
    cases = [_parse_plan_case(value, index) for index, value in enumerate(raw_cases)]
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ContractError("expected_attempt_plan case IDs must be unique")
    if len({normalized_identity_key(case["harbor_task_name"]) for case in cases}) != len(cases):
        raise ContractError("expected_attempt_plan has a staged task-name collision")
    if not isinstance(obj["baseline_required"], bool) or not isinstance(obj["stop_on_pass"], bool):
        raise ContractError("expected_attempt_plan boolean options are invalid")
    count = _require_positive_int(obj["n_attempts"], "expected_attempt_plan.n_attempts")
    if count > 1000:
        raise ContractError("expected_attempt_plan.n_attempts cannot exceed 1000")
    threshold = obj["pass_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not math.isfinite(float(threshold)):
        raise ContractError("expected_attempt_plan.pass_threshold must be finite")
    if not 0 <= float(threshold) <= 1:
        raise ContractError("expected_attempt_plan.pass_threshold must be between 0 and 1")
    if obj["grading_mode"] not in {"default", "default_plus_custom", "custom_only"}:
        raise ContractError("expected_attempt_plan.grading_mode is unsupported")
    reward_contract = validate_reward_contract(obj["reward_contract"])
    if reward_contract["grading_mode"] != obj["grading_mode"]:
        raise ContractError("expected_attempt_plan reward contract grading mode disagrees")
    arms: tuple[Arm, ...] = ("with_skill", "baseline") if obj["baseline_required"] else ("with_skill",)
    if len(parsed_occurrences) * len(arms) * count * len(cases) > 1_000_000:
        raise ContractError("expected_attempt_plan exceeds 1000000 attempts")
    raw_task_sets = obj["arm_task_sets"]
    if not isinstance(raw_task_sets, list):
        raise ContractError("expected_attempt_plan.arm_task_sets must be a list")
    task_sets = [_parse_arm_task_set(value, index) for index, value in enumerate(raw_task_sets)]
    if [item["arm"] for item in task_sets] != list(arms):
        raise ContractError("expected_attempt_plan.arm_task_sets do not match planned arms")
    expected_case_rows = [(case["case_id"], case["harbor_task_name"], case["reward_strategy"]) for case in cases]
    if any(
        normalized_refs_overlap(left["root_ref"], right["root_ref"])
        for index, left in enumerate(task_sets)
        for right in task_sets[index + 1 :]
    ):
        raise ContractError("expected_attempt_plan arm task roots must be disjoint")
    for item in task_sets:
        actual_case_rows = [
            (task["case_id"], task["harbor_task_name"], task["reward_strategy"])
            for task in cast("list[dict[str, Any]]", item["tasks"])
        ]
        if actual_case_rows != expected_case_rows:
            raise ContractError("expected_attempt_plan arm task-set cases do not match cases")
    expected = [
        ExpectedAttempt(agent.result_key, arm, case["case_id"], ordinal)
        for agent in parsed_occurrences
        for arm in arms
        for ordinal in range(1, count + 1)
        for case in cases
    ]
    raw_attempts = obj["attempts"]
    if not isinstance(raw_attempts, list):
        raise ContractError("expected_attempt_plan.attempts must be a list")
    actual: list[ExpectedAttempt] = []
    for index, raw in enumerate(raw_attempts):
        item = _expect_exact_fields(
            raw, required=_ATTEMPT_FIELDS, allowed=_ATTEMPT_FIELDS, name=f"expected_attempt_plan.attempts[{index}]"
        )
        actual.append(ExpectedAttempt(item["agent"], item["arm"], item["case_id"], item["ordinal"]))
    if actual != expected:
        raise ContractError("expected_attempt_plan.attempts is not the exact ordered Cartesian product")


def write_expected_attempt_plan(trusted_root: Path, plan: Mapping[str, object]) -> str:
    validate_expected_attempt_plan(plan)
    root = Path(trusted_root)
    return atomic_write_json(root / "expected_attempt_plan.json", plan, trusted_root=root)


def verify_persisted_expected_attempt_plan(
    trusted_root: Path,
    plan: Mapping[str, Any],
    task_plan_digest: str,
) -> None:
    """Bind the in-memory plan to exact immutable on-disk canonical bytes."""

    expected_digest = _validate_digest(task_plan_digest, "task_plan_digest", nullable=False)
    data = _artifact_bytes(Path(trusted_root), "expected_attempt_plan.json")
    if _sha256_digest(data) != expected_digest:
        raise ContractError("task_plan_digest does not match exact persisted plan bytes")
    try:
        persisted = json.loads(data)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ContractError("persisted expected-attempt plan is not valid JSON") from error
    if canonical_json_bytes(persisted, trailing_newline=True) != data:
        raise ContractError("persisted expected-attempt plan is not canonical JSON")
    if persisted != plan:
        raise ContractError("in-memory expected-attempt plan differs from immutable persisted plan")
    validate_expected_attempt_plan(persisted)


def _expected_attempts(plan: Mapping[str, Any]) -> tuple[ExpectedAttempt, ...]:
    validate_expected_attempt_plan(plan)
    return tuple(
        ExpectedAttempt(item["agent"], item["arm"], item["case_id"], item["ordinal"])
        for item in cast("list[dict[str, Any]]", plan["attempts"])
    )


def _numeric_rewards(value: object, name: str, *, nullable: bool) -> dict[str, float] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    if len(value) > MAX_REWARD_PROPERTIES:
        raise ContractError(f"{name} exceeds the {MAX_REWARD_PROPERTIES}-property limit")
    parsed: dict[str, float] = {}
    for key, raw in value.items():
        metric = _require_nonempty_text(key, f"{name} key", limit=128)
        if not _SAFE_METRIC_NAME_RE.fullmatch(metric):
            raise ContractError(f"{name} contains an unsafe metric name")
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(float(raw)):
            raise ContractError(f"{name}.{metric} must be a finite number")
        parsed[metric] = float(raw)
    return parsed


def validate_projected_reward_contract(
    rewards: Mapping[str, float],
    reward_contract: Mapping[str, object],
    *,
    reward_strategy: str = "single_step",
) -> None:
    """Enforce the plan-bound exact numeric key set and declared ranges."""

    contract = validate_reward_contract(reward_contract)
    default_names = cast("list[str]", contract["default_metrics"])
    custom_rows = cast("list[dict[str, object]]", contract["custom_metrics"])
    custom_ranges = {cast("str", row["name"]): (0.0, 1.0) for row in custom_rows}
    expected = set(default_names) | set(custom_ranges) | {"overall"}
    if set(rewards) != expected:
        raise ContractError("completed Harbor reward keys disagree with the immutable reward contract")
    validate_projected_step_reward_contract(rewards, contract)
    if reward_strategy == "single_step" and contract["grading_mode"] in {"default", "default_plus_custom"}:
        derived_overall = round(sum(float(rewards[name]) for name in default_names) / len(default_names), 4)
        if not math.isclose(float(rewards["overall"]), derived_overall, rel_tol=0.0, abs_tol=1e-12):
            raise ContractError("completed Harbor overall disagrees with the canonical SkillEvaluator mean")


def validate_projected_step_reward_contract(
    rewards: Mapping[str, float], reward_contract: Mapping[str, object]
) -> None:
    """Validate a multi-step subset against the same closed allowlist/ranges."""

    contract = validate_reward_contract(reward_contract)
    default_names = cast("list[str]", contract["default_metrics"])
    custom_rows = cast("list[dict[str, object]]", contract["custom_metrics"])
    custom_ranges = {cast("str", row["name"]): (0.0, 1.0) for row in custom_rows}
    allowed = set(default_names) | set(custom_ranges) | {"overall"}
    if not set(rewards).issubset(allowed):
        raise ContractError("Harbor step reward contains a key outside the immutable reward contract")
    for name, raw in rewards.items():
        value = float(raw)
        minimum, maximum = custom_ranges.get(name, (0.0, 1.0))
        if not minimum <= value <= maximum:
            raise ContractError(f"completed Harbor reward {name!r} is outside its declared range")


def _validate_exception_type(value: object, name: str) -> str | None:
    if value is None:
        return None
    exception_type = _require_nonempty_text(value, name, limit=128)
    if len(exception_type) > 128 or not _SAFE_EXCEPTION_TYPE_RE.fullmatch(exception_type):
        raise ContractError(f"{name} is not a safe exception type")
    return exception_type


def validate_harbor_schedule(sidecar: object) -> tuple[dict[str, Any], ...]:
    obj = _expect_exact_fields(sidecar, required=_SCHEDULE_FIELDS, allowed=_SCHEDULE_FIELDS, name="schedule")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise ContractError("schedule.schema_version is unsupported")
    _validate_digest(obj["plan_digest"], "schedule.plan_digest", nullable=False)
    _require_nonempty_text(obj["job_id"], "schedule.job_id", limit=128)
    _require_nonempty_text(obj["job_name"], "schedule.job_name", limit=256)
    _require_identity(obj["agent"], "schedule.agent")
    _require_nonempty_text(obj["resolved_model"], "schedule.resolved_model", limit=1024)
    _require_nonempty_text(obj["harbor_model"], "schedule.harbor_model", limit=1024)
    _validate_digest(
        obj["reward_contract_digest"],
        "schedule.reward_contract_digest",
        nullable=False,
    )
    _validate_digest(obj["task_set_digest"], "schedule.task_set_digest", nullable=False)
    if obj["arm"] not in {"with_skill", "baseline"}:
        raise ContractError("schedule.arm is unsupported")
    if not isinstance(obj["trials"], list) or not obj["trials"]:
        raise ContractError("schedule.trials must be a non-empty list")
    if len(obj["trials"]) > MAX_SIDECAR_TRIALS:
        raise ContractError(f"schedule.trials exceeds the {MAX_SIDECAR_TRIALS}-item limit")
    trials: list[dict[str, Any]] = []
    names: set[str] = set()
    keys: set[ExpectedAttempt] = set()
    for index, raw in enumerate(obj["trials"]):
        item = _expect_exact_fields(
            raw, required=_SCHEDULE_TRIAL_FIELDS, allowed=_SCHEDULE_TRIAL_FIELDS, name=f"schedule.trials[{index}]"
        )
        trial_name = _require_nonempty_text(item["trial_name"], f"schedule.trials[{index}].trial_name", limit=256)
        expected = ExpectedAttempt(item["agent"], item["arm"], item["case_id"], item["ordinal"])
        if expected.agent != obj["agent"] or expected.arm != obj["arm"]:
            raise ContractError("schedule trial identity disagrees with job binding")
        if item["reward_strategy"] not in {"single_step", "multi_step_mean", "multi_step_final"}:
            raise ContractError("schedule trial reward_strategy is unsupported")
        _validate_digest(
            item["staged_task_digest"],
            f"schedule.trials[{index}].staged_task_digest",
            nullable=False,
        )
        if trial_name in names or expected in keys:
            raise ContractError("schedule contains a duplicate trial identity")
        names.add(trial_name)
        keys.add(expected)
        trials.append(dict(item))
    return tuple(trials)


def _aggregate_step_rewards(steps: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    present = [step["rewards"] or {} for step in steps if step["verifier_result_present"]]
    if not present:
        return None
    keys = {key for rewards in present for key in rewards}
    if not keys:
        return None
    return {key: sum(rewards.get(key, 0.0) for rewards in present) / len(present) for key in keys}


def _rewards_equal(left: dict[str, float] | None, right: dict[str, float] | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.keys() == right.keys() and all(left[key] == right[key] for key in left)


def validate_harbor_results(
    sidecar: object, *, reward_contract: Mapping[str, object] | None = None
) -> tuple[dict[str, Any], ...]:
    obj = _expect_exact_fields(sidecar, required=_RESULTS_FIELDS, allowed=_RESULTS_FIELDS, name="results")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise ContractError("results.schema_version is unsupported")
    _validate_digest(obj["plan_digest"], "results.plan_digest", nullable=False)
    _validate_digest(obj["schedule_file_digest"], "results.schedule_file_digest", nullable=False)
    _require_nonempty_text(obj["job_id"], "results.job_id", limit=128)
    _require_nonempty_text(obj["job_name"], "results.job_name", limit=256)
    _require_identity(obj["agent"], "results.agent")
    _require_nonempty_text(obj["resolved_model"], "results.resolved_model", limit=1024)
    _require_nonempty_text(obj["harbor_model"], "results.harbor_model", limit=1024)
    _validate_digest(
        obj["reward_contract_digest"],
        "results.reward_contract_digest",
        nullable=False,
    )
    _validate_digest(obj["task_set_digest"], "results.task_set_digest", nullable=False)
    if obj["arm"] not in {"with_skill", "baseline"}:
        raise ContractError("results.arm is unsupported")
    if not isinstance(obj["trials"], list):
        raise ContractError("results.trials must be a list")
    if len(obj["trials"]) > MAX_SIDECAR_TRIALS:
        raise ContractError(f"results.trials exceeds the {MAX_SIDECAR_TRIALS}-item limit")
    trials: list[dict[str, Any]] = []
    names: set[str] = set()
    keys: set[ExpectedAttempt] = set()
    refs: set[str] = set()
    for index, raw in enumerate(obj["trials"]):
        name = f"results.trials[{index}]"
        item = _expect_exact_fields(raw, required=_RESULT_TRIAL_FIELDS, allowed=_RESULT_TRIAL_FIELDS, name=name)
        trial_name = _require_nonempty_text(item["trial_name"], f"{name}.trial_name", limit=256)
        expected = ExpectedAttempt(item["agent"], item["arm"], item["case_id"], item["ordinal"])
        if expected.agent != obj["agent"] or expected.arm != obj["arm"]:
            raise ContractError("results trial identity disagrees with job binding")
        strategy = item["reward_strategy"]
        if strategy not in {"single_step", "multi_step_mean", "multi_step_final"}:
            raise ContractError(f"{name}.reward_strategy is unsupported")
        _validate_digest(item["staged_task_digest"], f"{name}.staged_task_digest", nullable=False)
        if item["state"] not in {"completed", "failed"}:
            raise ContractError(f"{name}.state is unsupported")
        if not isinstance(item["verifier_result_present"], bool):
            raise ContractError(f"{name}.verifier_result_present must be boolean")
        if not isinstance(item["skill_logic_started"], bool):
            raise ContractError(f"{name}.skill_logic_started must be boolean")
        raw_agent_failure = item["agent_failure"]
        if raw_agent_failure is not None:
            agent_failure = _expect_exact_fields(
                raw_agent_failure,
                required=frozenset({"stage", "reason_code", "origin"}),
                allowed=frozenset({"stage", "reason_code", "origin"}),
                name=f"{name}.agent_failure",
            )
            pair = (agent_failure["stage"], agent_failure["reason_code"])
            if pair not in _LAUNCHED_AGENT_FAILURE_TAXONOMY:
                raise ContractError(f"{name}.agent_failure is not an allowed launched-agent classification")
            origin = agent_failure["origin"]
            if origin == "harbor_pre_instruction_phase":
                if pair != ("agent_adapter_bootstrap", "adapter_initialization_failed"):
                    raise ContractError(f"{name}.agent_failure has invalid Harbor phase provenance")
            elif origin != "trusted_adapter_marker":
                raise ContractError(f"{name}.agent_failure origin is unsupported")
            if item["skill_logic_started"]:
                raise ContractError(f"{name} agent failure marker requires skill_logic_started=false")
            if item["state"] != "failed":
                raise ContractError(f"{name} agent failure marker requires state=failed")
        top_rewards = _numeric_rewards(item["rewards"], f"{name}.rewards", nullable=True)
        if (top_rewards is not None) != item["verifier_result_present"] and top_rewards is not None:
            raise ContractError(f"{name}.rewards requires a present verifier result")
        if item["state"] == "completed" and not top_rewards:
            raise ContractError(f"{name}.completed result requires a non-empty reward mapping")
        if item["state"] == "failed" and top_rewards:
            raise ContractError(f"{name}.failed result cannot carry a score-bearing reward mapping")
        if not isinstance(item["steps"], list):
            raise ContractError(f"{name}.steps must be a list")
        if len(item["steps"]) > MAX_RESULT_STEPS:
            raise ContractError(f"{name}.steps exceeds the {MAX_RESULT_STEPS}-item limit")
        steps: list[dict[str, Any]] = []
        for step_index, raw_step in enumerate(item["steps"]):
            step_name = f"{name}.steps[{step_index}]"
            step = _expect_exact_fields(
                raw_step, required=_RESULT_STEP_FIELDS, allowed=_RESULT_STEP_FIELDS, name=step_name
            )
            _require_nonempty_text(step["step_name"], f"{step_name}.step_name", limit=128)
            if not isinstance(step["verifier_result_present"], bool):
                raise ContractError(f"{step_name}.verifier_result_present must be boolean")
            rewards = _numeric_rewards(step["rewards"], f"{step_name}.rewards", nullable=True)
            if rewards is not None and not step["verifier_result_present"]:
                raise ContractError(f"{step_name}.rewards requires a present verifier result")
            _validate_exception_type(step["exception_type"], f"{step_name}.exception_type")
            parsed_step = dict(step)
            parsed_step["rewards"] = rewards
            steps.append(parsed_step)
        _validate_exception_type(item["exception_type"], f"{name}.exception_type")
        if item["state"] == "completed":
            if not item["skill_logic_started"]:
                raise ContractError(f"{name}.completed result requires a structurally started agent phase")
            if raw_agent_failure is not None:
                raise ContractError(f"{name}.completed result cannot carry an agent failure")
            if item["exception_type"] is not None or any(step["exception_type"] is not None for step in steps):
                raise ContractError(f"{name}.completed result cannot carry a typed execution exception")
            if reward_contract is not None:
                validate_projected_reward_contract(top_rewards or {}, reward_contract, reward_strategy=strategy)
                for step in steps:
                    if step["rewards"] is not None:
                        validate_projected_step_reward_contract(step["rewards"], reward_contract)
        elif any(step["rewards"] for step in steps):
            raise ContractError(f"{name}.failed result cannot carry step rewards")
        if strategy == "single_step":
            if steps:
                raise ContractError(f"{name} single_step cannot contain step projections")
        elif strategy == "multi_step_mean":
            if not _rewards_equal(top_rewards, _aggregate_step_rewards(steps)):
                raise ContractError(f"{name} top-level rewards disagree with multi_step_mean semantics")
        else:
            final = steps[-1]["rewards"] if steps and steps[-1]["verifier_result_present"] else None
            if not _rewards_equal(top_rewards, final):
                raise ContractError(f"{name} top-level rewards disagree with multi_step_final semantics")
        ref = _validated_ref_text(item["trial_ref"])
        _validate_digest(item["trial_file_digest"], f"{name}.trial_file_digest", nullable=False)
        if trial_name in names or expected in keys or ref in refs:
            raise ContractError("results contains a duplicate trial identity or reference")
        names.add(trial_name)
        keys.add(expected)
        refs.add(ref)
        parsed = dict(item)
        parsed["agent_failure"] = None if raw_agent_failure is None else dict(agent_failure)
        parsed["rewards"] = top_rewards
        parsed["steps"] = steps
        trials.append(parsed)
    return tuple(trials)


def _read_json_artifact(trusted_root: Path, ref: str) -> tuple[dict[str, Any], str]:
    data = _artifact_bytes(trusted_root, ref)
    try:
        value = json.loads(data)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ContractError(f"evidence artifact is not valid JSON: {ref}") from error
    if not isinstance(value, dict):
        raise ContractError(f"evidence artifact must contain an object: {ref}")
    if canonical_json_bytes(value, trailing_newline=True) != data:
        raise ContractError(f"evidence artifact must use canonical JSON bytes: {ref}")
    return value, _sha256_digest(data)


def _score_from_rewards(rewards: Mapping[str, float] | None) -> float | None:
    if not rewards:
        return None
    from skillevaluator.tier3.harbor.metrics import overall_score

    normalized: dict[str, Any] = dict(rewards)
    if "overall" not in normalized and "reward" in normalized:
        normalized["overall"] = normalized["reward"]
    score = float(overall_score(normalized))
    if not math.isfinite(score):
        raise ContractError("canonical trial score must be finite")
    return score


def _validated_job_evidence(
    plan: Mapping[str, Any],
    task_plan_digest: str,
    trusted_root: Path,
    values: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[ExpectedAttempt, dict[str, Any]]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError("job_evidence must be an ordered list")
    plan_cases = {case["case_id"]: case["reward_strategy"] for case in cast("list[dict[str, Any]]", plan["cases"])}
    parsed_rows: list[dict[str, Any]] = []
    projections: dict[ExpectedAttempt, dict[str, Any]] = {}
    refs: set[str] = set()
    job_ids: set[str] = set()
    job_names: set[str] = set()
    trial_names: set[str] = set()
    for index, raw in enumerate(values):
        row = _expect_exact_fields(
            raw, required=_JOB_EVIDENCE_FIELDS, allowed=_JOB_EVIDENCE_FIELDS, name=f"job_evidence[{index}]"
        )
        job_id = _require_nonempty_text(row["job_id"], f"job_evidence[{index}].job_id", limit=128)
        job_name = _require_nonempty_text(row["job_name"], f"job_evidence[{index}].job_name", limit=256)
        agent = _require_identity(row["agent"], f"job_evidence[{index}].agent")
        arm = row["arm"]
        if arm not in {"with_skill", "baseline"}:
            raise ContractError(f"job_evidence[{index}].arm is unsupported")
        schedule_ref = _validated_ref_text(row["schedule_ref"])
        results_ref = _validated_ref_text(row["results_ref"])
        schedule_digest = _validate_digest(
            row["schedule_file_digest"], f"job_evidence[{index}].schedule_file_digest", nullable=False
        )
        results_digest = _validate_digest(
            row["results_file_digest"], f"job_evidence[{index}].results_file_digest", nullable=False
        )
        if job_id in job_ids or job_name in job_names or schedule_ref in refs or results_ref in refs:
            raise ContractError("job_evidence contains duplicate jobs or physical references")
        if schedule_ref == results_ref:
            raise ContractError("job_evidence schedule and results references must be distinct")
        job_ids.add(job_id)
        job_names.add(job_name)
        refs.update({schedule_ref, results_ref})
        schedule, actual_schedule_digest = _read_json_artifact(trusted_root, schedule_ref)
        results, actual_results_digest = _read_json_artifact(trusted_root, results_ref)
        if schedule_digest != actual_schedule_digest:
            raise ContractError("job_evidence schedule file digest mismatch")
        if results_digest != actual_results_digest:
            raise ContractError("job_evidence results file digest mismatch")
        schedule_trials = validate_harbor_schedule(schedule)
        result_trials = validate_harbor_results(
            results,
            reward_contract=cast("Mapping[str, object]", plan["reward_contract"]),
        )
        plan_agent = next(
            (item for item in cast("list[dict[str, Any]]", plan["agents"]) if item["result_key"] == agent),
            None,
        )
        plan_task_set = next(
            (item for item in cast("list[dict[str, Any]]", plan["arm_task_sets"]) if item["arm"] == arm),
            None,
        )
        if plan_agent is None or plan_task_set is None:
            raise ContractError("job sidecar agent or arm is outside the immutable plan")
        expected_root = (
            task_plan_digest,
            job_id,
            job_name,
            agent,
            arm,
            plan_agent["resolved_model"],
            harbor_model_for_agent(plan_agent["base_agent"], plan_agent["resolved_model"]),
            canonical_digest(plan["reward_contract"]),
            plan_task_set["task_set_digest"],
        )
        if (
            schedule["plan_digest"],
            schedule["job_id"],
            schedule["job_name"],
            schedule["agent"],
            schedule["arm"],
            schedule["resolved_model"],
            schedule["harbor_model"],
            schedule["reward_contract_digest"],
            schedule["task_set_digest"],
        ) != expected_root:
            raise ContractError("schedule sidecar does not match ledger job binding")
        if (
            results["plan_digest"],
            results["job_id"],
            results["job_name"],
            results["agent"],
            results["arm"],
            results["resolved_model"],
            results["harbor_model"],
            results["reward_contract_digest"],
            results["task_set_digest"],
        ) != expected_root or results["schedule_file_digest"] != schedule_digest:
            raise ContractError("results sidecar does not match plan/schedule job binding")
        schedule_by_name = {trial["trial_name"]: trial for trial in schedule_trials}
        result_by_name = {trial["trial_name"]: trial for trial in result_trials}
        if schedule_by_name.keys() != result_by_name.keys():
            raise ContractError("results sidecar has missing or extra scheduled trial identities")
        for trial in schedule_trials:
            trial_name = trial["trial_name"]
            result = result_by_name[trial_name]
            identity_fields = (
                "agent",
                "arm",
                "case_id",
                "ordinal",
                "reward_strategy",
                "staged_task_digest",
            )
            if any(result[field] != trial[field] for field in identity_fields):
                raise ContractError("results projection identity disagrees with schedule")
            expected = ExpectedAttempt(result["agent"], result["arm"], result["case_id"], result["ordinal"])
            if expected in projections or trial_name in trial_names:
                raise ContractError("job sidecars contain duplicate logical or physical trials")
            if plan_cases.get(expected.case_id) != result["reward_strategy"]:
                raise ContractError("result reward strategy disagrees with immutable plan")
            planned_task = next(
                (
                    item
                    for item in cast("list[dict[str, Any]]", plan_task_set["tasks"])
                    if item["case_id"] == expected.case_id
                ),
                None,
            )
            if planned_task is None or planned_task["staged_task_digest"] != result["staged_task_digest"]:
                raise ContractError("result staged task digest disagrees with immutable arm binding")
            trial_artifact, actual_trial_digest = _read_json_artifact(trusted_root, result["trial_ref"])
            if actual_trial_digest != result["trial_file_digest"]:
                raise ContractError("retained trial file digest disagrees with results sidecar")
            _expect_exact_fields(
                trial_artifact,
                required=_MINIMAL_TRIAL_FIELDS,
                allowed=_MINIMAL_TRIAL_FIELDS,
                name=f"retained trial {trial_name!r}",
            )
            expected_trial_artifact = {
                "schema_version": SCHEMA_VERSION,
                "plan_digest": task_plan_digest,
                "job_id": job_id,
                **{field: result[field] for field in _RESULT_TRIAL_FIELDS - {"trial_ref", "trial_file_digest"}},
            }
            if canonical_json_bytes(trial_artifact) != canonical_json_bytes(expected_trial_artifact):
                raise ContractError("retained trial content disagrees with its structured projection")
            trial_names.add(trial_name)
            projections[expected] = result
        parsed_rows.append(dict(row))
    return parsed_rows, projections


def _failure_object(failure: FailureRecord) -> dict[str, object]:
    result: dict[str, object] = {
        "scope": failure.scope,
        "stage": failure.stage,
        "reason_code": failure.reason_code,
        "origin": failure.origin,
    }
    if failure.agent is not None:
        result["agent"] = failure.agent
    if failure.evidence_ref is not None:
        result["evidence_ref"] = failure.evidence_ref
        result["evidence_file_digest"] = failure.evidence_file_digest
    return result


def _record_object(record: AttemptRecord, *, derived_passed: bool | None) -> dict[str, object]:
    result: dict[str, object] = {
        "agent": record.expected.agent,
        "arm": record.expected.arm,
        "case_id": record.expected.case_id,
        "ordinal": record.expected.ordinal,
        "disposition": record.disposition,
        "trial_ref": record.trial_ref,
        "trial_file_digest": record.trial_file_digest,
        "passed": derived_passed,
        "score": record.score,
        "failure": None if record.failure is None else _failure_object(record.failure),
        "caused_by": None if record.caused_by is None else _attempt_object(record.caused_by),
    }
    return result


def _validate_agent_exclusion_authorizer(agent: str, failure: FailureRecord, *, trusted_root: Path) -> None:
    _require_identity(agent, "agent exclusion key")
    if failure.scope != "agent" or failure.agent != agent:
        raise ContractError("agent exclusion must be a matching typed agent-scoped failure")
    if failure.origin != "trusted_preflight" or failure.stage not in {
        "agent_readiness",
        "preflight",
    }:
        raise ContractError("agent exclusion requires trusted preflight provenance")
    if failure.evidence_ref is None or failure.evidence_file_digest is None:
        raise ContractError("agent exclusion requires retained typed preflight evidence")
    _verify_failure_evidence_binding(Path(trusted_root), failure)


def build_execution_ledger(
    *,
    plan: Mapping[str, Any],
    task_plan_digest: str,
    records: Sequence[AttemptRecord],
    trusted_root: Path,
    job_evidence: Sequence[Mapping[str, Any]],
    agent_exclusions: Mapping[str, FailureRecord] | None = None,
    run_blockers: Sequence[FailureRecord] = (),
) -> dict[str, object]:
    """Validate and construct the exact plan-bound execution partition."""

    validate_expected_attempt_plan(plan)
    plan_digest = _validate_digest(task_plan_digest, "task_plan_digest", nullable=False)
    verify_persisted_expected_attempt_plan(Path(trusted_root), plan, cast("str", plan_digest))
    expected = _expected_attempts(plan)
    expected_set = set(expected)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ContractError("records must be an ordered list")
    parsed_records = tuple(records)
    if any(not isinstance(record, AttemptRecord) for record in parsed_records):
        raise ContractError("records must contain AttemptRecord values")
    keys = [record.expected for record in parsed_records]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ContractError("execution ledger contains duplicate attempts")
    missing = expected_set - set(keys)
    extra = set(keys) - expected_set
    if extra:
        raise ContractError("execution ledger contains extra or unbound attempts")
    if missing:
        raise ContractError("execution ledger is missing planned attempts")
    by_key = {record.expected: record for record in parsed_records}

    exclusions = dict(agent_exclusions or {})
    for agent, failure in exclusions.items():
        if not isinstance(failure, FailureRecord):
            raise ContractError("agent exclusion must be a matching typed agent-scoped failure")
        _validate_agent_exclusion_authorizer(agent, failure, trusted_root=Path(trusted_root))
    blockers = tuple(run_blockers)
    if any(not isinstance(failure, FailureRecord) or failure.scope != "run" for failure in blockers):
        raise ContractError("run_blockers must contain typed run-scoped failures")
    for failure in blockers:
        if failure.evidence_ref is not None:
            _verify_failure_evidence_binding(Path(trusted_root), failure)

    evidence_rows, projections = _validated_job_evidence(
        plan, cast("str", plan_digest), Path(trusted_root), job_evidence
    )
    projection_keys = set(projections)
    launched_keys = {record.expected for record in parsed_records if record.disposition in {"scored", "failed"}}
    if projection_keys != launched_keys:
        raise ContractError("job sidecar projections do not exactly match launched ledger attempts")

    threshold = float(plan["pass_threshold"])
    stop_on_pass = cast("bool", plan["stop_on_pass"])
    for agent in {key.agent for key in expected}:
        agent_records = [by_key[key] for key in expected if key.agent == agent]
        if any(record.disposition == "not_run_agent_excluded" for record in agent_records) and any(
            record.disposition in {"scored", "failed"} for record in agent_records
        ):
            raise ContractError("agent exclusion must happen before any occurrence trial launches")
    if stop_on_pass:
        identities = {(key.agent, key.arm, key.case_id) for key in expected}
        for agent, arm, case_id in identities:
            seen_pass = False
            for ordinal in range(1, int(plan["n_attempts"]) + 1):
                record = by_key[ExpectedAttempt(agent, arm, case_id, ordinal)]
                if seen_pass and record.disposition != "skipped_stop_on_pass":
                    raise ContractError("stop-on-pass forbids work after the first scored pass")
                if (
                    record.disposition == "scored"
                    and isinstance(record.score, int | float)
                    and not isinstance(record.score, bool)
                    and float(record.score) >= threshold
                ):
                    seen_pass = True
    trial_refs: set[str] = set()
    output_entries: list[dict[str, object]] = []
    for key in expected:
        record = by_key[key]
        pair = (record.trial_ref is not None, record.trial_file_digest is not None)
        if pair[0] != pair[1]:
            raise ContractError("trial reference and file digest are all-or-none")
        derived_passed: bool | None = None
        if record.disposition != "not_run_agent_unavailable" and record.caused_by is not None:
            raise ContractError("caused_by is permitted only for not_run_agent_unavailable")
        if record.disposition == "scored":
            if not all(pair) or record.failure is not None:
                raise ContractError("scored attempt requires a trial binding and no failure")
            if (
                isinstance(record.score, bool)
                or not isinstance(record.score, int | float)
                or not math.isfinite(float(record.score))
            ):
                raise ContractError("scored attempt requires a finite canonical score")
            derived_passed = derive_attempt_passed(float(record.score), threshold, supplied=record.passed)
        elif record.disposition == "failed":
            if not all(pair) or record.score is not None or record.passed is not None:
                raise ContractError("failed launched attempt requires only a trial binding and typed failure")
            if not isinstance(record.failure, FailureRecord):
                raise ContractError("failed launched attempt requires a typed failure")
            if record.failure.scope == "agent" and record.failure.agent != key.agent:
                raise ContractError("failed attempt failure identity does not match its agent")
            if record.failure.evidence_ref is None:
                raise ContractError("failed launched attempt requires retained typed failure evidence")
            _verify_failure_evidence_binding(Path(trusted_root), record.failure)
        else:
            if any(pair) or record.score is not None or record.passed is not None or record.failure is not None:
                raise ContractError("unlaunched attempt fields must be null")
            if record.disposition == "skipped_stop_on_pass":
                if not stop_on_pass:
                    raise ContractError("stop-on-pass skip is not authorized by the immutable plan")
                prior = [
                    by_key[ExpectedAttempt(key.agent, key.arm, key.case_id, ordinal)]
                    for ordinal in range(1, key.ordinal)
                ]
                if not any(
                    candidate.disposition == "scored"
                    and isinstance(candidate.score, int | float)
                    and not isinstance(candidate.score, bool)
                    and float(candidate.score) >= threshold
                    for candidate in prior
                ):
                    raise ContractError("stop-on-pass skip requires a prior scored pass")
            elif record.disposition == "not_run_agent_excluded":
                if key.agent not in exclusions:
                    raise ContractError("not_run_agent_excluded requires matching typed exclusion")
            elif record.disposition == "not_run_agent_unavailable":
                cause = record.caused_by
                if cause is None or cause not in by_key:
                    raise ContractError("not_run_agent_unavailable requires a bound caused_by attempt")
                cause_record = by_key[cause]
                if cause.agent != key.agent or expected.index(cause) >= expected.index(key):
                    raise ContractError("not_run_agent_unavailable cause must be an earlier attempt for the same agent")
                if (
                    cause_record.disposition != "failed"
                    or cause_record.failure is None
                    or cause_record.failure.scope != "agent"
                    or cause_record.failure.agent != key.agent
                ):
                    raise ContractError(
                        "not_run_agent_unavailable cause must be a terminal agent-scoped failed attempt"
                    )
            elif record.disposition == "not_run_run_blocked":
                if not blockers:
                    raise ContractError("not_run_run_blocked requires a typed run blocker")

        if record.trial_ref is not None and record.trial_file_digest is not None:
            ref = _validated_ref_text(record.trial_ref)
            digest = _validate_digest(record.trial_file_digest, "trial_file_digest", nullable=False)
            if ref in trial_refs:
                raise ContractError("trial physical references must be unique")
            trial_refs.add(ref)
            if _artifact_digest(Path(trusted_root), ref) != digest:
                raise ContractError("trial file digest mismatch")
            projection = projections[key]
            if projection["trial_ref"] != ref or projection["trial_file_digest"] != digest:
                raise ContractError("ledger trial binding disagrees with results sidecar")
            projection_score = _score_from_rewards(projection["rewards"])
            if record.disposition == "scored":
                if (
                    projection["state"] != "completed"
                    or projection_score is None
                    or projection_score != float(record.score)
                ):
                    raise ContractError("scored ledger entry disagrees with structured result projection")
            elif projection["state"] != "failed" or projection_score is not None:
                raise ContractError("failed ledger entry disagrees with structured result projection")
            if (
                record.disposition == "failed"
                and record.failure is not None
                and record.failure.scope == "agent"
                and projection["skill_logic_started"]
            ):
                raise ContractError("agent-scoped launched failure requires skill_logic_started=false")
            if record.disposition == "failed" and record.failure is not None and record.failure.scope == "agent":
                expected_classification = {
                    "stage": record.failure.stage,
                    "reason_code": record.failure.reason_code,
                    "origin": record.failure.origin,
                }
                if projection["agent_failure"] != expected_classification:
                    raise ContractError("agent-scoped failure disagrees with its trusted structured marker")
        output_entries.append(_record_object(record, derived_passed=derived_passed))

    ledger: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_plan_digest": plan_digest,
        "reward_contract_digest": canonical_digest(plan["reward_contract"]),
        "job_evidence": evidence_rows,
        "entries": output_entries,
    }
    validate_execution_ledger(
        plan,
        ledger,
        trusted_root=trusted_root,
        agent_exclusions=exclusions,
        run_blockers=blockers,
    )
    return ledger


def validate_execution_ledger(
    plan: Mapping[str, Any],
    ledger: object,
    *,
    trusted_root: Path,
    agent_exclusions: Mapping[str, FailureRecord] | None = None,
    run_blockers: Sequence[FailureRecord] = (),
) -> None:
    """Revalidate a persisted ledger without trusting producer dataclasses."""

    validate_expected_attempt_plan(plan)
    obj = _expect_exact_fields(
        ledger,
        required=frozenset(
            {
                "schema_version",
                "task_plan_digest",
                "reward_contract_digest",
                "job_evidence",
                "entries",
            }
        ),
        allowed=frozenset(
            {
                "schema_version",
                "task_plan_digest",
                "reward_contract_digest",
                "job_evidence",
                "entries",
            }
        ),
        name="execution_ledger",
    )
    if obj["schema_version"] != SCHEMA_VERSION:
        raise ContractError("execution_ledger.schema_version is unsupported")
    digest = _validate_digest(obj["task_plan_digest"], "execution_ledger.task_plan_digest", nullable=False)
    reward_contract_digest = _validate_digest(
        obj["reward_contract_digest"],
        "execution_ledger.reward_contract_digest",
        nullable=False,
    )
    if reward_contract_digest != canonical_digest(plan["reward_contract"]):
        raise ContractError("execution_ledger reward contract digest disagrees with the plan")
    verify_persisted_expected_attempt_plan(Path(trusted_root), plan, cast("str", digest))
    if not isinstance(obj["job_evidence"], list) or not isinstance(obj["entries"], list):
        raise ContractError("execution_ledger job_evidence and entries must be lists")
    exclusions = dict(agent_exclusions or {})
    for agent, failure in exclusions.items():
        if not isinstance(failure, FailureRecord):
            raise ContractError("agent exclusion must be a matching typed agent-scoped failure")
        _validate_agent_exclusion_authorizer(agent, failure, trusted_root=Path(trusted_root))
    blockers = tuple(run_blockers)
    if any(not isinstance(failure, FailureRecord) or failure.scope != "run" for failure in blockers):
        raise ContractError("run_blockers must contain typed run-scoped failures")
    for failure in blockers:
        if failure.evidence_ref is not None:
            _verify_failure_evidence_binding(Path(trusted_root), failure)
    # Rehydrate through the single producer validator so runtime and persisted
    # instances cannot drift. Authorizations are recovered from explicit entry
    # failure types only by higher-level manifest validation; this structural
    # path verifies the closed shape and physical bindings.
    expected = _expected_attempts(plan)
    if len(obj["entries"]) != len(expected):
        raise ContractError("execution_ledger entries do not match the planned partition")
    actual_keys: list[ExpectedAttempt] = []
    refs: set[str] = set()
    for index, raw in enumerate(obj["entries"]):
        fields = frozenset(
            {
                "agent",
                "arm",
                "case_id",
                "ordinal",
                "disposition",
                "trial_ref",
                "trial_file_digest",
                "passed",
                "score",
                "failure",
                "caused_by",
            }
        )
        row = _expect_exact_fields(raw, required=fields, allowed=fields, name=f"execution_ledger.entries[{index}]")
        key = ExpectedAttempt(row["agent"], row["arm"], row["case_id"], row["ordinal"])
        actual_keys.append(key)
        if row["disposition"] not in {
            "scored",
            "failed",
            "skipped_stop_on_pass",
            "not_run_agent_excluded",
            "not_run_agent_unavailable",
            "not_run_run_blocked",
        }:
            raise ContractError("execution_ledger entry disposition is unsupported")
        if (row["trial_ref"] is None) != (row["trial_file_digest"] is None):
            raise ContractError("trial reference and file digest are all-or-none")
        if row["trial_ref"] is not None:
            ref = _validated_ref_text(row["trial_ref"])
            file_digest = _validate_digest(row["trial_file_digest"], "trial_file_digest", nullable=False)
            if ref in refs:
                raise ContractError("execution_ledger trial references must be unique")
            refs.add(ref)
            if _artifact_digest(Path(trusted_root), ref) != file_digest:
                raise ContractError("trial file digest mismatch")
        if row["disposition"] == "scored":
            score = row["score"]
            if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(float(score)):
                raise ContractError("scored execution_ledger entry requires finite score")
            expected_pass = derive_attempt_passed(float(score), float(plan["pass_threshold"]), supplied=row["passed"])
            if row["passed"] is not expected_pass or row["failure"] is not None:
                raise ContractError("scored execution_ledger entry has contradictory fields")
        elif row["disposition"] == "failed":
            if row["score"] is not None or row["passed"] is not None or not isinstance(row["failure"], dict):
                raise ContractError("failed execution_ledger entry has contradictory fields")
            failure = _parse_failures([row["failure"]], "execution_ledger failure")[0]
            if failure.evidence_ref is None:
                raise ContractError("failed execution_ledger entry requires retained typed evidence")
            _verify_failure_evidence_binding(Path(trusted_root), failure)
        elif any(row[field] is not None for field in ("trial_ref", "trial_file_digest", "passed", "score", "failure")):
            raise ContractError("unlaunched execution_ledger entry fields must be null")
        if row["disposition"] == "not_run_agent_unavailable":
            cause = _expect_exact_fields(
                row["caused_by"],
                required=_ATTEMPT_FIELDS,
                allowed=_ATTEMPT_FIELDS,
                name=f"execution_ledger.entries[{index}].caused_by",
            )
            ExpectedAttempt(cause["agent"], cause["arm"], cause["case_id"], cause["ordinal"])
        elif row["caused_by"] is not None:
            raise ContractError("caused_by is permitted only for not_run_agent_unavailable")
    if actual_keys != list(expected):
        raise ContractError("execution_ledger entries are not in immutable plan order")
    rows_by_key = dict(zip(actual_keys, obj["entries"], strict=True))
    positions = {key: index for index, key in enumerate(actual_keys)}
    for agent in {key.agent for key in actual_keys}:
        agent_rows = [rows_by_key[key] for key in actual_keys if key.agent == agent]
        if any(row["disposition"] == "not_run_agent_excluded" for row in agent_rows) and any(
            row["disposition"] in {"scored", "failed"} for row in agent_rows
        ):
            raise ContractError("persisted agent exclusion is mixed with launched trials")
    if plan["stop_on_pass"]:
        identities = {(key.agent, key.arm, key.case_id) for key in actual_keys}
        for agent, arm, case_id in identities:
            seen_pass = False
            for ordinal in range(1, int(plan["n_attempts"]) + 1):
                row = rows_by_key[ExpectedAttempt(agent, arm, case_id, ordinal)]
                if seen_pass and row["disposition"] != "skipped_stop_on_pass":
                    raise ContractError("persisted stop-on-pass ledger contains work after a pass")
                if row["disposition"] == "scored" and row["passed"] is True:
                    seen_pass = True
    for key, row in rows_by_key.items():
        if row["disposition"] == "skipped_stop_on_pass":
            if not plan["stop_on_pass"]:
                raise ContractError("persisted stop-on-pass skip is not authorized by the plan")
            prior_pass = any(
                candidate.agent == key.agent
                and candidate.arm == key.arm
                and candidate.case_id == key.case_id
                and candidate.ordinal < key.ordinal
                and rows_by_key[candidate]["disposition"] == "scored"
                and rows_by_key[candidate]["passed"] is True
                for candidate in actual_keys
            )
            if not prior_pass:
                raise ContractError("persisted stop-on-pass skip has no prior scored pass")
        elif row["disposition"] == "not_run_agent_excluded" and key.agent not in exclusions:
            raise ContractError("persisted agent exclusion has no typed authorizer")
        elif row["disposition"] == "not_run_run_blocked" and not blockers:
            raise ContractError("persisted run-blocked attempt has no typed authorizer")
        if row["disposition"] != "not_run_agent_unavailable":
            continue
        cause_obj = row["caused_by"]
        cause = ExpectedAttempt(cause_obj["agent"], cause_obj["arm"], cause_obj["case_id"], cause_obj["ordinal"])
        cause_row = rows_by_key.get(cause)
        if (
            cause_row is None
            or positions[cause] >= positions[key]
            or cause.agent != key.agent
            or cause_row["disposition"] != "failed"
            or not isinstance(cause_row["failure"], dict)
            or cause_row["failure"].get("scope") != "agent"
            or cause_row["failure"].get("agent") != key.agent
        ):
            raise ContractError("not_run_agent_unavailable has an invalid caused_by binding")
    _rows, projections = _validated_job_evidence(plan, cast("str", digest), Path(trusted_root), obj["job_evidence"])
    launched = {
        key for key, row in zip(actual_keys, obj["entries"], strict=True) if row["disposition"] in {"scored", "failed"}
    }
    if set(projections) != launched:
        raise ContractError("job sidecar projections do not exactly match launched ledger attempts")
    for key in launched:
        row = rows_by_key[key]
        projection = projections[key]
        if row["trial_ref"] != projection["trial_ref"] or row["trial_file_digest"] != projection["trial_file_digest"]:
            raise ContractError("execution_ledger trial binding disagrees with results sidecar")
        projection_score = _score_from_rewards(projection["rewards"])
        if row["disposition"] == "scored":
            if (
                projection["state"] != "completed"
                or projection_score is None
                or projection_score != float(row["score"])
            ):
                raise ContractError("scored execution_ledger entry disagrees with result projection")
        else:
            if projection["state"] != "failed" or projection_score is not None:
                raise ContractError("failed execution_ledger entry disagrees with result projection")
            if row["failure"]["scope"] == "agent" and projection["skill_logic_started"]:
                raise ContractError("agent-scoped failed entry requires skill_logic_started=false")
            if row["failure"]["scope"] == "agent":
                expected_classification = {
                    "stage": row["failure"]["stage"],
                    "reason_code": row["failure"]["reason_code"],
                    "origin": row["failure"]["origin"],
                }
                if projection["agent_failure"] != expected_classification:
                    raise ContractError("agent-scoped failure disagrees with its trusted structured marker")


def write_execution_ledger(
    trusted_root: Path,
    plan: Mapping[str, Any],
    ledger: Mapping[str, object],
    *,
    agent_exclusions: Mapping[str, FailureRecord] | None = None,
    run_blockers: Sequence[FailureRecord] = (),
) -> str:
    """Validate and exclusively publish ``execution_ledger.json``."""

    root = Path(trusted_root)
    validate_execution_ledger(
        plan,
        ledger,
        trusted_root=root,
        agent_exclusions=agent_exclusions,
        run_blockers=run_blockers,
    )
    return atomic_write_json(root / "execution_ledger.json", ledger, trusted_root=root)


def arm_summaries_from_ledger(
    plan: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, int]]]:
    """Recompute case and logical-attempt units independently from the ledger."""

    validate_expected_attempt_plan(plan)
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ContractError("execution ledger entries must be a list")
    agents = [item["result_key"] for item in cast("list[dict[str, Any]]", plan["agents"])]
    arms = ["with_skill", *(["baseline"] if plan["baseline_required"] else [])]
    case_ids = [item["case_id"] for item in cast("list[dict[str, Any]]", plan["cases"])]
    output: dict[str, dict[str, dict[str, int]]] = {}
    for agent in agents:
        output[agent] = {}
        for arm in arms:
            selected = [row for row in entries if row.get("agent") == agent and row.get("arm") == arm]
            by_case = {case_id: [row for row in selected if row.get("case_id") == case_id] for case_id in case_ids}
            complete_dispositions = {"scored", "skipped_stop_on_pass"}
            output[agent][arm] = {
                "expected_cases": len(case_ids),
                "scored_cases": sum(
                    bool(rows) and all(row.get("disposition") in complete_dispositions for row in rows)
                    for rows in by_case.values()
                ),
                "exceptions": sum(row.get("disposition") == "failed" for row in selected),
                "expected_attempts": len(selected),
                "scored_attempts": sum(row.get("disposition") == "scored" for row in selected),
                "failed_attempts": sum(row.get("disposition") == "failed" for row in selected),
                "skipped_attempts": sum(row.get("disposition") == "skipped_stop_on_pass" for row in selected),
                "not_run_attempts": sum(str(row.get("disposition", "")).startswith("not_run_") for row in selected),
            }
    return output
