# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import unicodedata
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from skillevaluator.tier3.harbor import coverage
from skillevaluator.tier3.harbor.coverage import (
    CAPABILITY,
    MAX_CANONICAL_BYTES,
    AgentOccurrence,
    AttemptRecord,
    ContractError,
    CoveragePolicy,
    ExpectedAttempt,
    FailureRecord,
    arm_summaries_from_ledger,
    atomic_write_json,
    build_evals_json_snapshot,
    build_execution_ledger,
    build_expected_attempt_plan,
    build_harbor_case_map,
    build_native_harbor_snapshot,
    build_reward_contract,
    build_staged_arm_task_set,
    calculate_coverage,
    canonical_digest,
    canonical_json_bytes,
    derive_attempt_passed,
    resolve_occurrences,
    resolve_policy,
    resolve_required_agents,
    staged_task_digest,
    validate_execution_ledger,
    validate_expected_attempt_plan,
    validate_failure_evidence,
    validate_manifest,
    validate_projected_reward_contract,
    validate_reward_contract,
    verified_relative_ref,
    write_expected_attempt_plan,
    write_failure_evidence,
    write_manifest,
)
from skillevaluator.tier3.harbor.failure_taxonomy import taxonomy_schema

SCHEMA_PATH = Path(__file__).parents[2] / "src/skillevaluator/tier3/harbor/schemas/agent_coverage_v1.schema.json"
PLAN_SCHEMA_PATH = (
    Path(__file__).parents[2] / "src/skillevaluator/tier3/harbor/schemas/expected_attempt_plan_v1.schema.json"
)
LEDGER_SCHEMA_PATH = (
    Path(__file__).parents[2] / "src/skillevaluator/tier3/harbor/schemas/execution_ledger_v1.schema.json"
)
FAILURE_SCHEMA_PATH = (
    Path(__file__).parents[2] / "src/skillevaluator/tier3/harbor/schemas/failure_evidence_v1.schema.json"
)
DATASET_FIXTURE_ROOT = Path(__file__).parents[2] / "contracts/fixtures/dataset-c14n-v1"


def _agents() -> tuple[AgentOccurrence, ...]:
    return (
        AgentOccurrence(
            "claude-code",
            "claude-code",
            1,
            "aws/anthropic/bedrock-claude-opus-4-6",
            "aws/anthropic/bedrock-claude-opus-4-6",
            "CLI --agent-model",
        ),
        AgentOccurrence(
            "codex",
            "codex",
            1,
            "openai/openai/gpt-5.4",
            "openai/openai/gpt-5.4",
            "CLI --agent-model",
        ),
    )


def _policy_dict(policy: CoveragePolicy) -> dict[str, object]:
    return {
        "mode": policy.mode,
        "min_valid_agents": policy.min_valid_agents,
        "required_agents": list(policy.required_agents),
    }


def _arm_summary(*, expected: int, scored: int, failed: int) -> dict[str, int]:
    return {
        "expected_cases": expected,
        "scored_cases": scored,
        "exceptions": failed,
        "expected_attempts": expected,
        "scored_attempts": scored,
        "failed_attempts": failed,
        "skipped_attempts": 0,
        "not_run_attempts": 0,
    }


def _schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _ref_schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema["$defs"]["relative_ref"])


def _completed_manifest(tmp_path: Path, *, baseline_required: bool = True) -> dict[str, object]:
    requested = CoveragePolicy("any_valid", 1, ())
    resolution = resolve_policy(requested, _agents(), (CAPABILITY,))
    diagnostic_dir = tmp_path / "diagnostics" / "codex"
    diagnostic_dir.mkdir(parents=True)
    evidence_ref = "diagnostics/codex/agent-runtime-failure.json"
    failure = FailureRecord(
        "agent",
        "agent_adapter_bootstrap",
        "adapter_model_protocol_negotiation_failed",
        origin="trusted_adapter_marker",
        agent="codex",
    )
    evidence_digest = write_failure_evidence(
        tmp_path,
        evidence_ref,
        failure,
        skill_logic_started=False,
        http_status=400,
        exception_type="ProtocolNegotiationError",
    )
    plan_digest = atomic_write_json(
        tmp_path / "expected_attempt_plan.json",
        {"schema_version": "1.0", "run_id": "run-1", "baseline_required": baseline_required},
        trusted_root=tmp_path,
    )
    ledger_digest = atomic_write_json(
        tmp_path / "execution_ledger.json",
        {"schema_version": "1.0", "task_plan_digest": plan_digest},
        trusted_root=tmp_path,
    )
    return {
        "schema_version": "1.0",
        "run_id": "run-1",
        "phase": "completed",
        "capabilities": {
            "requested": [CAPABILITY],
            "provided": [CAPABILITY],
        },
        "requested_policy_digest": resolution.requested_digest,
        "effective_policy_digest": resolution.effective_digest,
        "task_plan_digest": plan_digest,
        "task_plan_ref": "expected_attempt_plan.json",
        "execution_ledger_digest": ledger_digest,
        "execution_ledger_ref": "execution_ledger.json",
        "dataset_digest": "sha256:" + "d" * 64,
        "dataset_digest_algorithm": "skill-eval-dataset-c14n/1",
        "status": "valid_degraded",
        "requested_policy": _policy_dict(resolution.requested),
        "authorized_tightening": None,
        "effective_policy": _policy_dict(resolution.effective),
        "policy_provenance": resolution.provenance,
        "requested_agents": ["claude-code", "codex"],
        "eligible_agents": ["claude-code"],
        "excluded_agents": ["codex"],
        "agents": {
            "claude-code": {
                "base_agent": "claude-code",
                "occurrence": 1,
                "requested_model": "aws/anthropic/bedrock-claude-opus-4-6",
                "resolved_model": "aws/anthropic/bedrock-claude-opus-4-6",
                "model_source": "CLI --agent-model",
                "status": "valid",
                "score_eligible": True,
                "with_skill": _arm_summary(expected=4, scored=4, failed=0),
                "baseline": _arm_summary(expected=4, scored=4, failed=0) if baseline_required else None,
            },
            "codex": {
                "base_agent": "codex",
                "occurrence": 1,
                "requested_model": "openai/openai/gpt-5.4",
                "resolved_model": "openai/openai/gpt-5.4",
                "model_source": "CLI --agent-model",
                "status": "invalid_infrastructure",
                "score_eligible": False,
                "reason_code": "adapter_model_protocol_negotiation_failed",
                "failure_stage": "agent_adapter_bootstrap",
                "failure_origin": "trusted_adapter_marker",
                "evidence_ref": evidence_ref,
                "evidence_file_digest": evidence_digest,
                "viewer_url": "https://results.example/jobs/codex",
                "with_skill": _arm_summary(expected=4, scored=0, failed=4),
                "baseline": _arm_summary(expected=4, scored=0, failed=4) if baseline_required else None,
            },
        },
        "warnings": [
            {
                "code": "optional_agent_excluded",
                "agent": "codex",
                "reason_code": "adapter_model_protocol_negotiation_failed",
                "failure_stage": "agent_adapter_bootstrap",
                "failure_origin": "trusted_adapter_marker",
                "evidence_ref": evidence_ref,
                "evidence_file_digest": evidence_digest,
            }
        ],
        "blockers": [],
    }


def _early_manifest() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "run_id": "run-early",
        "phase": "policy_validation",
        "capabilities": {"requested": [CAPABILITY], "provided": [CAPABILITY]},
        "requested_policy_digest": None,
        "effective_policy_digest": None,
        "task_plan_digest": None,
        "task_plan_ref": None,
        "execution_ledger_digest": None,
        "execution_ledger_ref": None,
        "dataset_digest": None,
        "dataset_digest_algorithm": None,
        "status": "invalid",
        "requested_policy": None,
        "authorized_tightening": None,
        "effective_policy": None,
        "policy_provenance": None,
        "requested_agents": ["claude-code", "codex"],
        "eligible_agents": [],
        "excluded_agents": ["claude-code", "codex"],
        "agents": {
            "claude-code": {
                "base_agent": "claude-code",
                "occurrence": 1,
                "requested_model": "aws/anthropic/bedrock-claude-opus-4-6",
                "resolved_model": "aws/anthropic/bedrock-claude-opus-4-6",
                "model_source": "CLI --agent-model",
                "status": "not_evaluated_run_blocked",
                "score_eligible": False,
                "with_skill": None,
                "baseline": None,
            },
            "codex": {
                "base_agent": "codex",
                "occurrence": 1,
                "requested_model": "openai/openai/gpt-5.4",
                "resolved_model": "openai/openai/gpt-5.4",
                "model_source": "CLI --agent-model",
                "status": "not_evaluated_run_blocked",
                "score_eligible": False,
                "with_skill": None,
                "baseline": None,
            },
        },
        "warnings": [],
        "blockers": [
            {
                "scope": "run",
                "stage": "policy_validation",
                "reason_code": "invalid_policy",
                "origin": "run_scope",
                "agent": None,
                "evidence_ref": None,
                "evidence_file_digest": None,
            }
        ],
    }


_PHASES = (
    "policy_validation",
    "dataset_validation",
    "task_generation",
    "preflight",
    "execution",
    "completed",
)
_POLICY_PHASE_FIELDS = (
    "requested_policy_digest",
    "effective_policy_digest",
    "requested_policy",
    "effective_policy",
    "policy_provenance",
)
_DATASET_PHASE_FIELDS = ("dataset_digest", "dataset_digest_algorithm")
_PLAN_PHASE_FIELDS = ("task_plan_digest", "task_plan_ref")
_LEDGER_PHASE_FIELDS = ("execution_ledger_digest", "execution_ledger_ref")
_PHASE_NON_NULL_VALUES: dict[str, object] = {
    "requested_policy_digest": "sha256:" + "1" * 64,
    "effective_policy_digest": "sha256:" + "2" * 64,
    "requested_policy": {"mode": "any_valid", "min_valid_agents": 1, "required_agents": []},
    "authorized_tightening": {"mode": "any_valid", "min_valid_agents": 1, "required_agents": []},
    "effective_policy": {"mode": "any_valid", "min_valid_agents": 1, "required_agents": []},
    "policy_provenance": "trusted_caller",
    "dataset_digest": "sha256:" + "3" * 64,
    "dataset_digest_algorithm": "skill-eval-dataset-c14n/1",
    "task_plan_digest": "sha256:" + "4" * 64,
    "task_plan_ref": "expected_attempt_plan.json",
    "execution_ledger_digest": "sha256:" + "5" * 64,
    "execution_ledger_ref": "execution_ledger.json",
}


def _phase_field_cases() -> tuple[tuple[str, str, bool], ...]:
    cases: list[tuple[str, str, bool]] = []
    groups = (
        (_POLICY_PHASE_FIELDS, "dataset_validation"),
        (_DATASET_PHASE_FIELDS, "task_generation"),
        (_PLAN_PHASE_FIELDS, "preflight"),
        (_LEDGER_PHASE_FIELDS, "execution"),
    )
    for phase_index, phase in enumerate(_PHASES):
        for fields, first_phase in groups:
            first_index = _PHASES.index(first_phase)
            cases.extend((phase, field, phase_index >= first_index) for field in fields)
        if phase == "policy_validation":
            cases.append((phase, "authorized_tightening", False))
    return tuple(cases)


def _phase_manifest(tmp_path: Path, phase: str) -> dict[str, object]:
    manifest = _completed_manifest(tmp_path)
    if phase == "completed":
        return manifest

    manifest["phase"] = phase
    manifest["status"] = "invalid"
    manifest["eligible_agents"] = []
    manifest["excluded_agents"] = list(manifest["requested_agents"])  # type: ignore[arg-type]
    manifest["warnings"] = []
    manifest["blockers"] = [
        {
            "scope": "run",
            "stage": phase,
            "reason_code": f"{phase}_failed",
            "origin": "run_scope",
            "agent": None,
            "evidence_ref": None,
            "evidence_file_digest": None,
        }
    ]
    agents = manifest["agents"]
    assert isinstance(agents, dict)
    for agent in agents.values():
        assert isinstance(agent, dict)
        agent["status"] = "not_evaluated_run_blocked"
        agent["score_eligible"] = False
        agent["with_skill"] = None
        agent["baseline"] = None
        for field in (
            "reason_code",
            "failure_stage",
            "failure_origin",
            "evidence_ref",
            "evidence_file_digest",
            "viewer_url",
        ):
            agent.pop(field, None)

    if phase == "policy_validation":
        for field in (
            "requested_policy_digest",
            "effective_policy_digest",
            "requested_policy",
            "authorized_tightening",
            "effective_policy",
            "policy_provenance",
        ):
            manifest[field] = None
    if phase in {"policy_validation", "dataset_validation"}:
        manifest["dataset_digest"] = None
        manifest["dataset_digest_algorithm"] = None
    if phase in {"policy_validation", "dataset_validation", "task_generation"}:
        manifest["task_plan_digest"] = None
        manifest["task_plan_ref"] = None
    if phase in {"policy_validation", "dataset_validation", "task_generation", "preflight"}:
        manifest["execution_ledger_digest"] = None
        manifest["execution_ledger_ref"] = None
    return manifest


def _assert_schema_and_runtime_valid(manifest: dict[str, object], *, trusted_root: Path) -> None:
    _schema_validator().validate(manifest)
    validate_manifest(manifest, trusted_root=trusted_root)


def _assert_schema_and_runtime_invalid(manifest: dict[str, object], *, trusted_root: Path) -> None:
    with pytest.raises(ValidationError):
        _schema_validator().validate(manifest)
    with pytest.raises(ContractError):
        validate_manifest(manifest, trusted_root=trusted_root)


def _replace_task_plan(tmp_path: Path, manifest: dict[str, object], plan: object) -> None:
    path = tmp_path / "expected_attempt_plan.json"
    path.unlink()
    manifest["task_plan_digest"] = atomic_write_json(path, plan, trusted_root=tmp_path)


def test_resolve_occurrences_uses_stable_six_field_identity() -> None:
    occurrences = resolve_occurrences(
        ("codex", "claude-code", "codex"),
        requested_models=(None, "requested-claude", "requested-codex"),
        resolved_models=("default-codex", "resolved-claude", "resolved-codex"),
        model_sources=("provider default", "CLI --agent-model", "CLI --agent-model"),
    )
    assert occurrences == (
        AgentOccurrence("codex-1", "codex", 1, None, "default-codex", "provider default"),
        AgentOccurrence(
            "claude-code",
            "claude-code",
            1,
            "requested-claude",
            "resolved-claude",
            "CLI --agent-model",
        ),
        AgentOccurrence(
            "codex-2",
            "codex",
            2,
            "requested-codex",
            "resolved-codex",
            "CLI --agent-model",
        ),
    )


def test_required_base_name_must_resolve_uniquely() -> None:
    agents = resolve_occurrences(
        ("codex", "codex"),
        requested_models=(None, None),
        resolved_models=("model-1", "model-2"),
        model_sources=("default", "default"),
    )
    with pytest.raises(ContractError, match="ambiguous"):
        resolve_required_agents(("codex",), agents)
    assert resolve_required_agents(("codex-2",), agents) == ("codex-2",)


def test_policy_rejects_noncanonical_occurrence_identity() -> None:
    agents = (AgentOccurrence("codex-alias", "codex", 1, None, "model", "default"),)
    with pytest.raises(ContractError, match="result-key identity"):
        resolve_policy(CoveragePolicy("all_selected", 1, ()), agents, ())


def test_any_valid_requires_capability() -> None:
    with pytest.raises(ContractError, match="agent-coverage/1"):
        resolve_policy(CoveragePolicy("any_valid", 1, ()), _agents(), ())


def test_requested_any_valid_requires_capability_even_when_tightened_to_all_selected() -> None:
    with pytest.raises(ContractError, match="agent-coverage/1"):
        resolve_policy(
            CoveragePolicy("any_valid", 1, ()),
            _agents(),
            (),
            authorized_tightening=CoveragePolicy("all_selected", 2, ()),
        )


def test_unknown_capability_fails_closed() -> None:
    with pytest.raises(ContractError, match="unsupported capability"):
        resolve_policy(CoveragePolicy("all_selected", 2, ()), _agents(), ("future-contract/1",))


@pytest.mark.parametrize(
    "capabilities",
    [
        {"requested": [], "provided": [CAPABILITY]},
        {"requested": [CAPABILITY], "provided": []},
    ],
)
def test_any_valid_manifest_capability_negotiation_matches_schema_and_runtime(
    tmp_path: Path, capabilities: dict[str, list[str]]
) -> None:
    manifest = _completed_manifest(tmp_path)
    manifest["capabilities"] = capabilities
    _assert_schema_and_runtime_invalid(manifest, trusted_root=tmp_path)


def test_policy_validation_may_have_no_requested_capability() -> None:
    manifest = _early_manifest()
    manifest["capabilities"] = {"requested": [], "provided": [CAPABILITY]}
    _schema_validator().validate(manifest)
    validate_manifest(manifest)


def test_policy_defaults_are_normalized_before_digesting() -> None:
    all_selected = resolve_policy(CoveragePolicy("all_selected", 0, ()), _agents(), ())
    any_valid = resolve_policy(CoveragePolicy("any_valid", 0, ()), _agents(), (CAPABILITY,))
    assert all_selected.requested.min_valid_agents == 2
    assert any_valid.requested.min_valid_agents == 1
    assert all_selected.requested_digest == canonical_digest(_policy_dict(all_selected.requested))


def test_authorized_policy_is_joined_without_weakening_trusted_policy() -> None:
    resolution = resolve_policy(
        CoveragePolicy("any_valid", 1, ()),
        _agents(),
        (CAPABILITY,),
        authorized_tightening=CoveragePolicy("any_valid", 1, ("codex",)),
    )
    assert resolution.effective == CoveragePolicy("any_valid", 1, ("codex",))
    assert resolution.provenance == "trusted_caller_plus_authorized_tightening"
    assert resolution.authorized_tightening == CoveragePolicy("any_valid", 1, ("codex",))


def test_policy_join_preserves_orthogonal_minimum_and_required_agent_constraints() -> None:
    resolution = resolve_policy(
        CoveragePolicy("any_valid", 1, ("codex",)),
        _agents(),
        (CAPABILITY,),
        authorized_tightening=CoveragePolicy("any_valid", 2, ()),
    )
    assert resolution.effective == CoveragePolicy("any_valid", 2, ("codex",))

    reverse = resolve_policy(
        CoveragePolicy("any_valid", 2, ()),
        _agents(),
        (CAPABILITY,),
        authorized_tightening=CoveragePolicy("any_valid", 1, ("codex",)),
    )
    assert reverse.effective == CoveragePolicy("any_valid", 2, ("codex",))


def test_all_selected_join_accepts_weaker_independent_constraint() -> None:
    resolution = resolve_policy(
        CoveragePolicy("all_selected", 2, ()),
        _agents(),
        (CAPABILITY,),
        authorized_tightening=CoveragePolicy("any_valid", 1, ("codex",)),
    )
    assert resolution.effective == CoveragePolicy("all_selected", 2, ("codex",))


def test_symmetric_optional_exclusion_is_degraded() -> None:
    policy = resolve_policy(CoveragePolicy("any_valid", 1, ()), _agents(), (CAPABILITY,))
    assert (
        calculate_coverage(
            policy.effective,
            ("claude-code", "codex"),
            ("claude-code",),
            ("codex",),
            (),
        ).status
        == "valid_degraded"
    )
    assert (
        calculate_coverage(
            policy.effective,
            ("claude-code", "codex"),
            ("codex",),
            ("claude-code",),
            (),
        ).status
        == "valid_degraded"
    )


def test_required_exclusion_is_invalid() -> None:
    policy = resolve_policy(CoveragePolicy("any_valid", 1, ("codex",)), _agents(), (CAPABILITY,))
    assert (
        calculate_coverage(
            policy.effective,
            ("claude-code", "codex"),
            ("claude-code",),
            ("codex",),
            (),
        ).status
        == "invalid"
    )


@pytest.mark.parametrize(
    ("requested", "eligible", "excluded"),
    [
        (("claude-code", "codex"), ("claude-code",), ()),
        (("claude-code", "codex"), ("claude-code",), ("codex", "other")),
        (("claude-code", "codex"), ("claude-code",), ("claude-code", "codex")),
        (("claude-code", "codex"), ("claude-code", "claude-code"), ("codex",)),
        (("claude-code", "codex"), ("codex", "claude-code"), ()),
    ],
)
def test_coverage_rejects_invalid_partition(
    requested: tuple[str, ...], eligible: tuple[str, ...], excluded: tuple[str, ...]
) -> None:
    policy = resolve_policy(CoveragePolicy("any_valid", 1, ()), _agents(), (CAPABILITY,))
    with pytest.raises(ContractError, match="partition"):
        calculate_coverage(policy.effective, requested, eligible, excluded, ())


def test_all_selected_rejects_conflicting_minimum() -> None:
    with pytest.raises(ContractError, match="all-selected"):
        resolve_policy(CoveragePolicy("all_selected", 1, ()), _agents(), ())


def test_run_blocker_forces_invalid() -> None:
    policy = resolve_policy(CoveragePolicy("any_valid", 1, ()), _agents(), (CAPABILITY,))
    decision = calculate_coverage(
        policy.effective,
        ("claude-code", "codex"),
        ("claude-code", "codex"),
        (),
        (FailureRecord("run", "verifier", "grader_contract_failure"),),
    )
    assert decision.status == "invalid"


# ---------------------------------------------------------------------------
# Task 7: consolidated contract classification matrix.
#
# calculate_coverage takes NO lift/score argument -- coverage validity is
# purely structural (blockers, required-agent eligibility, min-valid count).
# So the behavior-matrix rows that differ only by quality ("pass" vs "low lift"
# vs "neutral") classify identically here: this contract computes coverage,
# while quality and release gating remain downstream decisions. Several rows overlap the single-purpose
# tests above; this table is the consolidated authority for the matrix.
# ---------------------------------------------------------------------------

_MATRIX_AGENTS: tuple[str, ...] = ("claude-code", "codex")


@pytest.mark.parametrize(
    ("case_id", "policy", "eligible", "excluded", "blockers", "expected"),
    [
        # Both agents run correctly -> valid_full (lift irrelevant to coverage).
        ("both_valid_full", CoveragePolicy("any_valid", 1, ()), ("claude-code", "codex"), (), (), "valid_full"),
        # One pass + one genuinely low-lift agent are BOTH structurally valid.
        ("pass_plus_low_lift_full", CoveragePolicy("any_valid", 1, ()), ("claude-code", "codex"), (), (), "valid_full"),
        # Symmetric degraded: claude valid + codex excluded, and the mirror.
        (
            "claude_valid_codex_excluded",
            CoveragePolicy("any_valid", 1, ()),
            ("claude-code",),
            ("codex",),
            (),
            "valid_degraded",
        ),
        (
            "codex_valid_claude_excluded",
            CoveragePolicy("any_valid", 1, ()),
            ("codex",),
            ("claude-code",),
            (),
            "valid_degraded",
        ),
        # Only the one valid agent has a low/neutral lift -> still valid_degraded.
        (
            "only_valid_low_lift_degraded",
            CoveragePolicy("any_valid", 1, ()),
            ("claude-code",),
            ("codex",),
            (),
            "valid_degraded",
        ),
        # Both harnesses invalid -> invalid.
        ("both_invalid", CoveragePolicy("any_valid", 1, ()), (), ("claude-code", "codex"), (), "invalid"),
        # An explicitly required agent is excluded -> invalid.
        ("required_excluded", CoveragePolicy("any_valid", 1, ("codex",)), ("claude-code",), ("codex",), (), "invalid"),
        # any-valid minimum of 2 not met by a single eligible agent -> invalid.
        ("min_two_not_met", CoveragePolicy("any_valid", 2, ()), ("claude-code",), ("codex",), (), "invalid"),
        # A shared run-scoped blocker forces invalid even with both eligible.
        (
            "shared_run_blocker",
            CoveragePolicy("any_valid", 1, ()),
            ("claude-code", "codex"),
            (),
            (FailureRecord("run", "verifier", "grader_contract_failure"),),
            "invalid",
        ),
        # all-selected requires every selected agent valid.
        ("all_selected_full", CoveragePolicy("all_selected", 2, ()), ("claude-code", "codex"), (), (), "valid_full"),
        (
            "all_selected_one_excluded_invalid",
            CoveragePolicy("all_selected", 2, ()),
            ("claude-code",),
            ("codex",),
            (),
            "invalid",
        ),
    ],
)
def test_contract_matrix_coverage_classification(
    case_id: str,
    policy: CoveragePolicy,
    eligible: tuple[str, ...],
    excluded: tuple[str, ...],
    blockers: tuple[FailureRecord, ...],
    expected: str,
) -> None:
    resolution = resolve_policy(policy, _agents(), (CAPABILITY,))
    decision = calculate_coverage(
        resolution.effective,
        _MATRIX_AGENTS,
        eligible,
        excluded,
        blockers,
    )
    assert decision.status == expected, case_id
    assert decision.eligible_agents == eligible
    assert decision.excluded_agents == excluded


def test_coverage_rejects_empty_requested_partition() -> None:
    policy = resolve_policy(CoveragePolicy("any_valid", 1, ()), _agents(), (CAPABILITY,))
    with pytest.raises(ContractError, match="requested_agents cannot be empty"):
        calculate_coverage(policy.effective, (), (), (), ())


@pytest.mark.parametrize(
    "capability",
    ["agent-coverage/2", "agent-coverage/0", "agent-coverage/1.0", "agent-coverage"],
)
def test_unsupported_capability_major_fails_closed(capability: str) -> None:
    # An unsupported major (or malformed) capability request never negotiates a
    # coverage contract -- it fails closed before any policy join.
    with pytest.raises(ContractError, match="unsupported capability"):
        resolve_policy(CoveragePolicy("any_valid", 1, ()), _agents(), (capability,))


def test_duplicate_base_agent_gets_stable_ordered_occurrence_keys() -> None:
    occurrences = resolve_occurrences(
        ["claude-code", "claude-code"],
        requested_models=[None, None],
        resolved_models=[
            "aws/anthropic/bedrock-claude-opus-4-6",
            "aws/anthropic/bedrock-claude-opus-4-7",
        ],
        model_sources=["CLI --agent-model", "CLI --agent-model"],
    )
    assert [agent.result_key for agent in occurrences] == ["claude-code-1", "claude-code-2"]
    assert [agent.occurrence for agent in occurrences] == [1, 2]
    # The two occurrences bind distinct resolved models under one base harness.
    assert occurrences[0].resolved_model != occurrences[1].resolved_model


def _valid_full_manifest(tmp_path: Path) -> dict[str, object]:
    """A completed valid_full manifest where one eligible agent has a neutral,
    low scored ratio. Coverage stays valid_full: scored ratio / lift is quality
    evidence for downstream quality policy, never a coverage downgrade."""
    requested = CoveragePolicy("any_valid", 1, ())
    resolution = resolve_policy(requested, _agents(), (CAPABILITY,))
    plan_digest = atomic_write_json(
        tmp_path / "expected_attempt_plan.json",
        {"schema_version": "1.0", "run_id": "run-full", "baseline_required": True},
        trusted_root=tmp_path,
    )
    ledger_digest = atomic_write_json(
        tmp_path / "execution_ledger.json",
        {"schema_version": "1.0", "task_plan_digest": plan_digest},
        trusted_root=tmp_path,
    )
    return {
        "schema_version": "1.0",
        "run_id": "run-full",
        "phase": "completed",
        "capabilities": {"requested": [CAPABILITY], "provided": [CAPABILITY]},
        "requested_policy_digest": resolution.requested_digest,
        "effective_policy_digest": resolution.effective_digest,
        "task_plan_digest": plan_digest,
        "task_plan_ref": "expected_attempt_plan.json",
        "execution_ledger_digest": ledger_digest,
        "execution_ledger_ref": "execution_ledger.json",
        "dataset_digest": "sha256:" + "d" * 64,
        "dataset_digest_algorithm": "skill-eval-dataset-c14n/1",
        "status": "valid_full",
        "requested_policy": _policy_dict(resolution.requested),
        "authorized_tightening": None,
        "effective_policy": _policy_dict(resolution.effective),
        "policy_provenance": resolution.provenance,
        "requested_agents": ["claude-code", "codex"],
        "eligible_agents": ["claude-code", "codex"],
        "excluded_agents": [],
        "agents": {
            "claude-code": {
                "base_agent": "claude-code",
                "occurrence": 1,
                "requested_model": "aws/anthropic/bedrock-claude-opus-4-6",
                "resolved_model": "aws/anthropic/bedrock-claude-opus-4-6",
                "model_source": "CLI --agent-model",
                "status": "valid",
                "score_eligible": True,
                "with_skill": _arm_summary(expected=4, scored=4, failed=0),
                "baseline": _arm_summary(expected=4, scored=4, failed=0),
            },
            "codex": {
                "base_agent": "codex",
                "occurrence": 1,
                "requested_model": "openai/openai/gpt-5.4",
                "resolved_model": "openai/openai/gpt-5.4",
                "model_source": "CLI --agent-model",
                "status": "valid",
                "score_eligible": True,
                # Neutral / low-lift arm: still valid, still score-eligible.
                "with_skill": _arm_summary(expected=4, scored=4, failed=0),
                "baseline": _arm_summary(expected=4, scored=4, failed=0),
            },
        },
        "warnings": [],
        "blockers": [],
    }


def test_valid_full_manifest_is_lift_independent(tmp_path: Path) -> None:
    manifest = _valid_full_manifest(tmp_path)
    # Runtime validation accepts the manifest and confirms valid_full with both
    # agents score-eligible -- no scores_invalid, no lift downgrade.
    validate_manifest(manifest, trusted_root=tmp_path)
    _schema_validator().validate(manifest)
    assert manifest["status"] == "valid_full"
    assert manifest["excluded_agents"] == []
    assert all(agent["score_eligible"] for agent in manifest["agents"].values())  # type: ignore[union-attr,index]


def test_canonical_domains_are_stable_and_distinct() -> None:
    value = {"unicode": "café", "a": 1}
    object_bytes = canonical_json_bytes(value)
    file_bytes = canonical_json_bytes(value, trailing_newline=True)
    assert object_bytes == b'{"a":1,"unicode":"caf\xc3\xa9"}'
    assert file_bytes == object_bytes + b"\n"
    assert canonical_digest(value) != "sha256:" + __import__("hashlib").sha256(file_bytes).hexdigest()
    with pytest.raises(ContractError, match="canonical JSON"):
        canonical_json_bytes({"score": float("nan")})


@pytest.mark.parametrize(
    "value",
    [
        {1: "x"},
        ("tuple",),
        {"nested": (1, 2)},
        {"custom": object()},
        b"bytes",
    ],
)
def test_canonical_json_rejects_non_json_types_before_encoder_coercion(value: object) -> None:
    with pytest.raises(ContractError, match="JSON"):
        canonical_json_bytes(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_every_non_finite_number(value: float) -> None:
    with pytest.raises(ContractError, match="finite"):
        canonical_json_bytes({"score": value})


def test_canonical_json_string_key_cannot_collide_with_integer_key() -> None:
    assert canonical_json_bytes({"1": "x"}) == b'{"1":"x"}'
    with pytest.raises(ContractError, match="string keys"):
        canonical_digest({1: "x"})


def test_canonical_write_is_exclusive_and_digest_stable(tmp_path: Path) -> None:
    path = tmp_path / "agent_coverage.json"
    digest = atomic_write_json(path, {"b": 2, "a": 1}, trusted_root=tmp_path)
    assert digest.startswith("sha256:")
    assert path.read_bytes() == b'{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"a": 1}, trusted_root=tmp_path)


def test_canonical_write_rejects_symlink_root_parent_and_destination(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ContractError, match="symlink"):
        atomic_write_json(linked_root / "x.json", {"a": 1}, trusted_root=linked_root)

    outside = tmp_path / "outside"
    outside.mkdir()
    (real_root / "linked-parent").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContractError, match="symlink"):
        atomic_write_json(
            real_root / "linked-parent" / "x.json",
            {"a": 1},
            trusted_root=real_root,
        )

    (real_root / "dangling.json").symlink_to(real_root / "missing.json")
    with pytest.raises(FileExistsError):
        atomic_write_json(real_root / "dangling.json", {"a": 1}, trusted_root=real_root)


def test_canonical_write_rejects_symlink_in_trusted_root_ancestry(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    real_root = real_parent / "root"
    real_root.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    trusted_root = linked_parent / "root"
    with pytest.raises(ContractError, match="trusted root.*symlink"):
        atomic_write_json(trusted_root / "x.json", {"a": 1}, trusted_root=trusted_root)


def test_canonical_write_cleans_temporary_after_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3.harbor import coverage

    def fail_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EIO, "injected publish failure")

    monkeypatch.setattr(coverage.os, "link", fail_link)
    with pytest.raises(OSError, match="injected publish failure"):
        atomic_write_json(tmp_path / "result.json", {"a": 1}, trusted_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("ref", ["/tmp/x", "../x", "a/../../x", "a\\..\\x", "", "."])
def test_evidence_ref_rejects_escape(tmp_path: Path, ref: str) -> None:
    with pytest.raises(ContractError):
        verified_relative_ref(tmp_path, ref)


def test_evidence_ref_requires_bounded_regular_in_root_file(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    evidence = diagnostics / "failure.json"
    evidence.write_text("{}\n", encoding="utf-8")
    assert verified_relative_ref(tmp_path, "diagnostics/failure.json") == "diagnostics/failure.json"

    (diagnostics / "linked.json").symlink_to(evidence)
    with pytest.raises(ContractError, match="symlink"):
        verified_relative_ref(tmp_path, "diagnostics/linked.json")

    oversized = diagnostics / "oversized.json"
    with oversized.open("wb") as stream:
        stream.truncate(MAX_CANONICAL_BYTES + 1)
    with pytest.raises(ContractError, match="too large"):
        verified_relative_ref(tmp_path, "diagnostics/oversized.json")


def test_evidence_ref_closes_file_descriptor_when_fstat_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "failure.json").write_text("{}\n", encoding="utf-8")
    real_fstat = os.fstat
    evidence_fds: list[int] = []

    def failing_fstat(fd):
        opened = real_fstat(fd)
        if stat.S_ISREG(opened.st_mode):
            evidence_fds.append(fd)
            raise OSError(errno.EIO, "injected fstat failure")
        return opened

    monkeypatch.setattr(coverage.os, "fstat", failing_fstat)

    with pytest.raises(OSError, match="injected fstat failure"):
        coverage._open_relative_regular(tmp_path, "diagnostics/failure.json")

    assert evidence_fds
    with pytest.raises(OSError) as closed:
        real_fstat(evidence_fds[-1])
    assert closed.value.errno == errno.EBADF


def test_early_phase_cannot_claim_degraded() -> None:
    manifest = _early_manifest()
    manifest["phase"] = "policy_validation"
    manifest["status"] = "valid_degraded"
    with pytest.raises(ContractError, match="phase"):
        validate_manifest(manifest)


def test_early_phase_requires_run_blocker_and_null_matrix() -> None:
    manifest = _early_manifest()
    validate_manifest(manifest)

    manifest["blockers"] = []
    with pytest.raises(ContractError, match="blocker"):
        validate_manifest(manifest)

    manifest = _early_manifest()
    manifest["dataset_digest"] = "sha256:" + "d" * 64
    with pytest.raises(ContractError, match="phase"):
        validate_manifest(manifest)


@pytest.mark.parametrize("phase", _PHASES)
def test_phase_matrix_is_accepted_by_schema_and_runtime(tmp_path: Path, phase: str) -> None:
    _assert_schema_and_runtime_valid(_phase_manifest(tmp_path, phase), trusted_root=tmp_path)


def test_dataset_validation_invalid_envelope_is_published_to_canonical_path(tmp_path: Path) -> None:
    manifest = _phase_manifest(tmp_path, "dataset_validation")
    digest = write_manifest(tmp_path, manifest)
    path = tmp_path / "agent_coverage.json"
    assert path.is_file()
    assert digest == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text()) == manifest


@pytest.mark.parametrize(("phase", "field", "must_be_present"), _phase_field_cases())
def test_every_phase_field_nullability_rule_matches_schema_and_runtime(
    tmp_path: Path,
    phase: str,
    field: str,
    must_be_present: bool,
) -> None:
    manifest = _phase_manifest(tmp_path, phase)
    if must_be_present:
        manifest[field] = None
    else:
        manifest[field] = deepcopy(_PHASE_NON_NULL_VALUES[field])
    _assert_schema_and_runtime_invalid(manifest, trusted_root=tmp_path)


@pytest.mark.parametrize("phase", _PHASES[:-1])
def test_noncompleted_phase_rejects_arm_summaries_in_schema_and_runtime(tmp_path: Path, phase: str) -> None:
    manifest = _phase_manifest(tmp_path, phase)
    agents = manifest["agents"]
    assert isinstance(agents, dict)
    claude = agents["claude-code"]
    assert isinstance(claude, dict)
    claude["with_skill"] = _arm_summary(expected=4, scored=0, failed=4)
    _assert_schema_and_runtime_invalid(manifest, trusted_root=tmp_path)


@pytest.mark.parametrize("case", ["status", "blocker", "eligible", "agent_status"])
def test_early_terminal_semantics_are_aligned_in_schema_and_runtime(tmp_path: Path, case: str) -> None:
    manifest = _phase_manifest(tmp_path, "policy_validation")
    if case == "status":
        manifest["status"] = "valid_degraded"
    elif case == "blocker":
        manifest["blockers"] = []
    elif case == "eligible":
        manifest["eligible_agents"] = ["claude-code"]
        manifest["excluded_agents"] = ["codex"]
    else:
        agents = manifest["agents"]
        assert isinstance(agents, dict)
        claude = agents["claude-code"]
        assert isinstance(claude, dict)
        claude["status"] = "invalid_infrastructure"
        claude["reason_code"] = "agent_runtime_failure"
        claude["failure_stage"] = "agent_startup"
    _assert_schema_and_runtime_invalid(manifest, trusted_root=tmp_path)


def test_completed_requires_with_skill_arm_in_schema_and_runtime(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    agents = manifest["agents"]
    assert isinstance(agents, dict)
    claude = agents["claude-code"]
    assert isinstance(claude, dict)
    claude["with_skill"] = None
    _assert_schema_and_runtime_invalid(manifest, trusted_root=tmp_path)


def test_completed_manifest_verifies_policy_partition_and_artifact_digests(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    validate_manifest(manifest, trusted_root=tmp_path)

    manifest["effective_policy_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ContractError, match="effective_policy_digest"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_valid_degraded_requires_one_exact_warning_per_excluded_agent(tmp_path: Path) -> None:
    missing = _completed_manifest(tmp_path)
    missing["warnings"] = []
    with pytest.raises(ContractError, match="optional_agent_excluded"):
        validate_manifest(missing, trusted_root=tmp_path)

    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    duplicate = _completed_manifest(duplicate_root)
    warnings = duplicate["warnings"]
    assert isinstance(warnings, list)
    warnings.append(deepcopy(warnings[0]))
    with pytest.raises(ContractError, match="optional_agent_excluded"):
        validate_manifest(duplicate, trusted_root=duplicate_root)

    extra_root = tmp_path / "extra"
    extra_root.mkdir()
    extra = _completed_manifest(extra_root)
    warnings = extra["warnings"]
    assert isinstance(warnings, list)
    warnings.append({"code": "unrelated_warning"})
    with pytest.raises(ContractError, match="optional_agent_excluded"):
        validate_manifest(extra, trusted_root=extra_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason_code", "different_reason"),
        ("failure_stage", "different_stage"),
        ("evidence_ref", "expected_attempt_plan.json"),
    ],
)
def test_valid_degraded_rejects_warning_that_mismatches_agent(tmp_path: Path, field: str, value: str) -> None:
    manifest = _completed_manifest(tmp_path)
    warnings = manifest["warnings"]
    assert isinstance(warnings, list)
    warning = warnings[0]
    assert isinstance(warning, dict)
    warning[field] = value
    with pytest.raises(ContractError, match="optional_agent_excluded|trusted agent taxonomy"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_completed_invalid_agent_requires_retained_evidence_not_only_viewer_url(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    agents = manifest["agents"]
    assert isinstance(agents, dict)
    codex = agents["codex"]
    assert isinstance(codex, dict)
    codex["evidence_ref"] = None
    codex["evidence_file_digest"] = None
    with pytest.raises(ContractError, match="retained typed evidence"):
        validate_manifest(manifest, trusted_root=tmp_path)


@pytest.mark.parametrize("field", ["agent", "reason_code", "failure_stage", "evidence_ref", "evidence_file_digest"])
def test_warning_optional_field_rejects_null_in_schema_and_runtime(tmp_path: Path, field: str) -> None:
    manifest = _completed_manifest(tmp_path)
    warnings = manifest["warnings"]
    assert isinstance(warnings, list)
    warning = warnings[0]
    assert isinstance(warning, dict)
    warning[field] = None
    _assert_schema_and_runtime_invalid(manifest, trusted_root=tmp_path)


@pytest.mark.parametrize(
    "ref",
    [
        "",
        "/absolute.json",
        ".",
        "./failure.json",
        "diagnostics/./failure.json",
        "diagnostics/../failure.json",
        "diagnostics//failure.json",
        "diagnostics/failure.json/",
        "diagnostics\\failure.json",
        "diagnostics/\x00failure.json",
        ".\n",
        "diagnostics/result.json\n",
        "diagnostics/ok.json\n\\escape",
        "diagnostics/\tfailure.json",
        "diagnostics/\x7ffailure.json",
        "diagnostics/\x85failure.json",
        "diagnostics/\u2028failure.json",
        "diagnostics/\u2029failure.json",
        "a" * 4097,
    ],
)
def test_canonical_ref_rules_match_schema_and_runtime(tmp_path: Path, ref: str) -> None:
    manifest = _completed_manifest(tmp_path)
    agents = manifest["agents"]
    warnings = manifest["warnings"]
    assert isinstance(agents, dict)
    assert isinstance(warnings, list)
    codex = agents["codex"]
    warning = warnings[0]
    assert isinstance(codex, dict)
    assert isinstance(warning, dict)
    codex["evidence_ref"] = ref
    warning["evidence_ref"] = ref
    _assert_schema_and_runtime_invalid(manifest, trusted_root=tmp_path)


@pytest.mark.parametrize(
    "ref",
    [
        "expected_attempt_plan.json",
        "diagnostics/codex/trial-1.json",
        ".hidden",
        "a/.hidden",
        "...",
        "a..",
        "é/β",
        "a" * 4096,
    ],
)
def test_canonical_relative_ref_shape_acceptance_matches_schema_and_runtime(ref: str) -> None:
    _ref_schema_validator().validate(ref)
    assert coverage._validated_ref_text(ref) == ref


@pytest.mark.parametrize(
    "ref",
    [
        "",
        "/absolute.json",
        ".",
        "..",
        "./failure.json",
        "diagnostics/./failure.json",
        "diagnostics/../failure.json",
        "diagnostics//failure.json",
        "diagnostics/failure.json/",
        "diagnostics\\failure.json",
        "diagnostics/\x00failure.json",
        ".\n",
        "diagnostics/result.json\n",
        "diagnostics/ok.json\n\\escape",
        "diagnostics/\tfailure.json",
        "diagnostics/\x7ffailure.json",
        "diagnostics/\x85failure.json",
        "diagnostics/\u2028failure.json",
        "diagnostics/\u2029failure.json",
        "a" * 4097,
    ],
)
def test_canonical_relative_ref_shape_rejection_matches_schema_and_runtime(ref: str) -> None:
    with pytest.raises(ValidationError):
        _ref_schema_validator().validate(ref)
    with pytest.raises(ContractError):
        coverage._validated_ref_text(ref)


def test_arm_summary_requires_case_and_attempt_counters(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    validate_manifest(manifest, trusted_root=tmp_path)

    agents = manifest["agents"]
    assert isinstance(agents, dict)
    claude = agents["claude-code"]
    assert isinstance(claude, dict)
    with_skill = claude["with_skill"]
    assert isinstance(with_skill, dict)
    del with_skill["expected_attempts"]
    with pytest.raises(ContractError, match="expected_attempts"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_score_eligible_agent_requires_every_expected_case_complete(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    agents = manifest["agents"]
    assert isinstance(agents, dict)
    claude = agents["claude-code"]
    assert isinstance(claude, dict)
    with_skill = claude["with_skill"]
    assert isinstance(with_skill, dict)
    with_skill["scored_cases"] = 3

    with pytest.raises(ContractError, match="score-eligible.*complete"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_completed_manifest_allows_explicit_skip_baseline(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path, baseline_required=False)
    _assert_schema_and_runtime_valid(manifest, trusted_root=tmp_path)


def test_score_eligible_agent_rejects_vacuous_zero_case_arm(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    agents = manifest["agents"]
    assert isinstance(agents, dict)
    claude = agents["claude-code"]
    assert isinstance(claude, dict)
    claude["with_skill"] = _arm_summary(expected=0, scored=0, failed=0)
    with pytest.raises(ContractError, match="non-zero"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_score_eligible_agent_allows_complete_attempts_with_authorized_skips(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    agents = manifest["agents"]
    assert isinstance(agents, dict)
    claude = agents["claude-code"]
    assert isinstance(claude, dict)
    with_skill = claude["with_skill"]
    assert isinstance(with_skill, dict)
    with_skill.update(
        expected_attempts=8,
        scored_attempts=4,
        skipped_attempts=4,
    )
    validate_manifest(manifest, trusted_root=tmp_path)


def test_completed_baseline_presence_is_bound_to_verified_task_plan(tmp_path: Path) -> None:
    required = _completed_manifest(tmp_path)
    agents = required["agents"]
    assert isinstance(agents, dict)
    for agent in agents.values():
        assert isinstance(agent, dict)
        agent["baseline"] = None
    with pytest.raises(ContractError, match="baseline_required"):
        validate_manifest(required, trusted_root=tmp_path)

    second_root = tmp_path / "not-required"
    second_root.mkdir()
    not_required = _completed_manifest(second_root, baseline_required=False)
    agents = not_required["agents"]
    assert isinstance(agents, dict)
    for agent in agents.values():
        assert isinstance(agent, dict)
        agent["baseline"] = _arm_summary(expected=4, scored=4, failed=0)
    with pytest.raises(ContractError, match="baseline_required"):
        validate_manifest(not_required, trusted_root=second_root)


@pytest.mark.parametrize("baseline_value", [None, 1, "false", []])
def test_verified_task_plan_requires_boolean_baseline_required(tmp_path: Path, baseline_value: object) -> None:
    manifest = _completed_manifest(tmp_path)
    plan: dict[str, object] = {"schema_version": "1.0", "run_id": "run-1"}
    if baseline_value is not None:
        plan["baseline_required"] = baseline_value
    _replace_task_plan(tmp_path, manifest, plan)

    with pytest.raises(ContractError, match="baseline_required"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_manifest_accepts_compatible_minor_only_through_namespaced_extensions(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    manifest["schema_version"] = "1.1"
    manifest["extensions"] = {"org.skillevaluator/diagnostics": {"trace_id": "trace-1"}}
    validate_manifest(manifest, trusted_root=tmp_path)

    manifest["extensions"] = {"diagnostics": {"trace_id": "trace-1"}}
    with pytest.raises(ContractError, match="namespaced"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_completed_manifest_rejects_changed_referenced_artifact(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    plan = tmp_path / "expected_attempt_plan.json"
    plan.write_bytes(b'{"run_id":"tampered","schema_version":"1.0"}\n')
    with pytest.raises(ContractError, match="task_plan_digest"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_manifest_rejects_unknown_fields_and_agent_map_mismatch(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    manifest["unexpected"] = True
    with pytest.raises(ContractError, match="unexpected field"):
        validate_manifest(manifest, trusted_root=tmp_path)

    del manifest["unexpected"]
    del manifest["agents"]["codex"]  # type: ignore[index]
    with pytest.raises(ContractError, match="agent-map"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_manifest_rejects_result_key_that_disagrees_with_six_field_identity(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    agents = manifest["agents"]
    assert isinstance(agents, dict)
    agents["codex-alias"] = agents.pop("codex")
    manifest["requested_agents"] = ["claude-code", "codex-alias"]
    manifest["excluded_agents"] = ["codex-alias"]
    manifest["warnings"][0]["agent"] = "codex-alias"  # type: ignore[index]
    with pytest.raises(ContractError, match="result-key identity"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_write_manifest_validates_and_never_overwrites(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    digest = write_manifest(tmp_path, manifest)
    assert digest.startswith("sha256:")
    with pytest.raises(FileExistsError):
        write_manifest(tmp_path, manifest)


def test_packaged_schema_declares_closed_contract_objects() -> None:
    schema_path = Path(__file__).parents[2] / "src/skillevaluator/tier3/harbor/schemas/agent_coverage_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$id"].endswith("agent_coverage_v1.schema.json")
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["policy"]["additionalProperties"] is False
    assert schema["$defs"]["agent"]["additionalProperties"] is False


def test_writer_rejects_target_outside_trusted_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ContractError, match="trusted root"):
        atomic_write_json(tmp_path / "outside.json", {"a": 1}, trusted_root=root)


def test_writer_requires_existing_regular_directory_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ContractError, match="trusted root"):
        atomic_write_json(missing / "x.json", {"a": 1}, trusted_root=missing)

    regular_file = tmp_path / "file"
    regular_file.write_text("x", encoding="utf-8")
    with pytest.raises(ContractError, match="trusted root"):
        atomic_write_json(regular_file / "x.json", {"a": 1}, trusted_root=regular_file)


def test_atomic_writer_fsyncs_and_publishes_one_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "canonical.json"
    atomic_write_json(path, {"a": 1}, trusted_root=tmp_path)
    stat_result = os.lstat(path)
    assert stat_result.st_nlink == 1
    assert path.is_file()


def _semantic_snapshot() -> dict[str, object]:
    return build_evals_json_snapshot(
        entries=[
            {"id": "case-1", "question": "first", "expected": ["one"]},
            {"id": "case-2", "question": "second", "expected": ["two"]},
        ],
        evaluation_config={"grading_mode": "default", "judge": {"temperature": 0}},
        referenced_files={
            "graders/domain.py": b"def grade():\n    return 1\n",
            "files/input.txt": b"fixture\n",
        },
    )


def _attempt_plan(
    *,
    baseline_required: bool = True,
    n_attempts: int = 2,
    stop_on_pass: bool = False,
) -> dict[str, object]:
    cases = [
        {"case_id": "case-1", "harbor_task_name": "case-1", "reward_strategy": "single_step"},
        {"case_id": "case-2", "harbor_task_name": "case-2", "reward_strategy": "single_step"},
    ]
    arms = ("with_skill", "baseline") if baseline_required else ("with_skill",)
    arm_task_sets = []
    for arm_index, arm in enumerate(arms):
        tasks = [
            {**case, "staged_task_digest": f"sha256:{arm_index + case_index + 1:064x}"}
            for case_index, case in enumerate(cases)
        ]
        core = {
            "arm": arm,
            "root_ref": f"staged/{arm}",
            "digest_algorithm": "skill-evaluator-staged-harbor-task-tree-c14n/1",
            "skill_payload_digest": "sha256:" + "f" * 64 if arm == "with_skill" else None,
            "tasks": tasks,
        }
        arm_task_sets.append({**core, "task_set_digest": canonical_digest(core)})
    return build_expected_attempt_plan(
        run_id="run-task-2",
        dataset_snapshot_kind="evals_json",
        semantic_snapshot=_semantic_snapshot(),
        agents=_agents(),
        cases=cases,
        baseline_required=baseline_required,
        n_attempts=n_attempts,
        stop_on_pass=stop_on_pass,
        pass_threshold=0.6,
        grading_mode="default",
        arm_task_sets=arm_task_sets,
    )


def _all_blocked_records(plan: dict[str, object]) -> list[AttemptRecord]:
    attempts = plan["attempts"]
    assert isinstance(attempts, list)
    return [
        AttemptRecord(
            ExpectedAttempt(
                agent=str(item["agent"]),
                arm=item["arm"],
                case_id=str(item["case_id"]),
                ordinal=int(item["ordinal"]),
            ),
            "not_run_run_blocked",
            None,
            None,
        )
        for item in attempts
    ]


def _persist_plan(tmp_path: Path, plan: dict[str, object]) -> str:
    return write_expected_attempt_plan(tmp_path, plan)


def test_plan_contains_every_occurrence_arm_case_and_ordinal() -> None:
    plan = _attempt_plan()
    attempts = plan["attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 2 * 2 * 2 * 2
    assert [(row["agent"], row["arm"], row["ordinal"], row["case_id"]) for row in attempts[:8]] == [
        ("claude-code", "with_skill", 1, "case-1"),
        ("claude-code", "with_skill", 1, "case-2"),
        ("claude-code", "with_skill", 2, "case-1"),
        ("claude-code", "with_skill", 2, "case-2"),
        ("claude-code", "baseline", 1, "case-1"),
        ("claude-code", "baseline", 1, "case-2"),
        ("claude-code", "baseline", 2, "case-1"),
        ("claude-code", "baseline", 2, "case-2"),
    ]
    validate_expected_attempt_plan(plan)


def test_plan_binds_dataset_agent_model_and_execution_options() -> None:
    plan = _attempt_plan(stop_on_pass=True)
    assert plan["dataset_digest"] == canonical_digest(_semantic_snapshot())
    assert plan["dataset_digest_algorithm"] == "skill-eval-dataset-c14n/1"
    assert plan["agents"][1] == {
        "result_key": "codex",
        "base_agent": "codex",
        "occurrence": 1,
        "requested_model": "openai/openai/gpt-5.4",
        "resolved_model": "openai/openai/gpt-5.4",
        "model_source": "CLI --agent-model",
    }
    assert plan["baseline_required"] is True
    assert plan["n_attempts"] == 2
    assert plan["stop_on_pass"] is True
    assert plan["pass_threshold"] == 0.6
    assert plan["grading_mode"] == "default"


def test_plan_second_write_is_rejected(tmp_path: Path) -> None:
    plan = _attempt_plan()
    write_expected_attempt_plan(tmp_path, plan)
    with pytest.raises(FileExistsError):
        write_expected_attempt_plan(tmp_path, plan)


def test_ledger_rejects_stale_digest_mutated_plan_and_post_write_plan_tamper(tmp_path: Path) -> None:
    plan = _attempt_plan(baseline_required=False, n_attempts=1)
    digest = _persist_plan(tmp_path, plan)
    records = _all_blocked_records(plan)
    blocker = [FailureRecord("run", "execution", "shared_configuration_invalid")]
    common = {
        "records": records,
        "trusted_root": tmp_path,
        "job_evidence": [],
        "run_blockers": blocker,
    }
    with pytest.raises(ContractError, match="exact persisted plan bytes"):
        build_execution_ledger(
            plan=plan,
            task_plan_digest="sha256:" + "0" * 64,
            **common,
        )

    mutated = deepcopy(plan)
    mutated["pass_threshold"] = 0.7
    with pytest.raises(ContractError, match="differs from immutable persisted plan"):
        build_execution_ledger(
            plan=mutated,
            task_plan_digest=digest,
            **common,
        )

    plan_path = tmp_path / "expected_attempt_plan.json"
    plan_path.write_bytes(plan_path.read_bytes().replace(b'"pass_threshold":0.6', b'"pass_threshold":0.7'))
    with pytest.raises(ContractError, match="exact persisted plan bytes"):
        build_execution_ledger(
            plan=plan,
            task_plan_digest=digest,
            **common,
        )


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.599999999, False), (0.6, True), (0.600000001, True)],
)
def test_pass_derivation_uses_exact_unrounded_threshold(score: float, expected: bool) -> None:
    assert derive_attempt_passed(score, 0.6) is expected
    with pytest.raises(ContractError, match="disagrees"):
        derive_attempt_passed(score, 0.6, supplied=not expected)


def test_case_map_rejects_staged_name_collision() -> None:
    with pytest.raises(ContractError, match="staged|collision"):
        build_expected_attempt_plan(
            run_id="collision",
            dataset_snapshot_kind="evals_json",
            semantic_snapshot=_semantic_snapshot(),
            agents=_agents(),
            cases=[
                {"case_id": "one", "harbor_task_name": "same", "reward_strategy": "single_step"},
                {"case_id": "two", "harbor_task_name": "same", "reward_strategy": "single_step"},
            ],
            baseline_required=False,
            n_attempts=1,
            stop_on_pass=False,
            pass_threshold=0.5,
            grading_mode="default",
        )


def test_evals_semantic_snapshot_is_stable_and_omits_transport_metadata() -> None:
    first = _semantic_snapshot()
    second = build_evals_json_snapshot(
        entries=deepcopy(first["entries"]),
        evaluation_config=deepcopy(first["evaluation_config"]),
        referenced_files={
            "files/input.txt": b"fixture\n",
            "graders/domain.py": b"def grade():\n    return 1\n",
        },
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert b"/Users/" not in canonical_json_bytes(first)
    assert [row["path"] for row in first["referenced_files"]] == [
        "files/input.txt",
        "graders/domain.py",
    ]


def test_native_semantic_snapshot_parses_config_and_hashes_every_task_file(tmp_path: Path) -> None:
    for task_name, strategy in (("task-a", "mean"), ("task-b", "final")):
        task = tmp_path / task_name
        (task / "tests").mkdir(parents=True)
        (task / "task.toml").write_text(
            f'schema_version = "1.3"\nmulti_step_reward_strategy = "{strategy}"\n[[steps]]\nname = "solve"\n',
            encoding="utf-8",
        )
        (task / "instruction.md").write_text(f"{task_name}\n", encoding="utf-8")
        (task / "tests" / "test.sh").write_bytes(b"#!/bin/sh\nexit 0\n")

    snapshot = build_native_harbor_snapshot(tmp_path, task_ids=["task-b", "task-a"])
    assert snapshot["task_ids"] == ["task-b", "task-a"]
    assert snapshot["tasks"][0]["config"]["multi_step_reward_strategy"] == "final"
    assert [row["path"] for row in snapshot["tasks"][0]["files"]] == [
        "task-b/instruction.md",
        "task-b/task.toml",
        "task-b/tests/test.sh",
    ]
    assert build_harbor_case_map(
        [tmp_path / "task-b", tmp_path / "task-a"],
        case_ids=["raw/b", "raw a"],
    ) == [
        {"case_id": "raw/b", "harbor_task_name": "task-b", "reward_strategy": "multi_step_final"},
        {"case_id": "raw a", "harbor_task_name": "task-a", "reward_strategy": "multi_step_mean"},
    ]


def test_staged_task_c14n_binds_executable_bit_and_rejects_hardlinks(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    script = task / "test.sh"
    script.write_bytes(b"#!/bin/sh\nexit 0\n")
    non_executable = staged_task_digest(task)
    script.chmod(0o755)
    assert staged_task_digest(task) != non_executable

    alias = task / "alias.sh"
    os.link(script, alias)
    with pytest.raises(ContractError, match="hard-linked"):
        staged_task_digest(task)


def test_staged_arm_binding_rejects_extra_root_entries(tmp_path: Path) -> None:
    arm_root = tmp_path / "staged/with_skill"
    task = arm_root / "case-1"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('schema_version = "1.0"\n', encoding="utf-8")
    payload = tmp_path / "skill-payload"
    payload.mkdir()
    (payload / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    cases = [{"case_id": "case-1", "harbor_task_name": "case-1", "reward_strategy": "single_step"}]
    binding = build_staged_arm_task_set(
        tmp_path,
        arm="with_skill",
        task_root=arm_root,
        cases=cases,
        skill_payload_path=payload,
    )
    assert binding["digest_algorithm"] == "skill-evaluator-staged-harbor-task-tree-c14n/1"
    (arm_root / "unbound.txt").write_text("must fail\n", encoding="utf-8")
    with pytest.raises(ContractError, match="extra|unsafe"):
        build_staged_arm_task_set(
            tmp_path,
            arm="with_skill",
            task_root=arm_root,
            cases=cases,
            skill_payload_path=payload,
        )


def test_staged_path_collision_key_normalizes_unicode_and_case() -> None:
    decomposed = "Cafe\N{COMBINING ACUTE ACCENT}.txt"
    composed_upper = "CAF\N{LATIN CAPITAL LETTER E WITH ACUTE}.TXT"
    assert (
        unicodedata.normalize("NFC", decomposed).casefold() == unicodedata.normalize("NFC", composed_upper).casefold()
    )


def test_evals_and_native_semantic_dataset_digests_have_shared_golden_bytes() -> None:
    expected_digests = json.loads((DATASET_FIXTURE_ROOT / "digests.json").read_text())
    evals_root = DATASET_FIXTURE_ROOT / "evals_json"
    evals_snapshot = build_evals_json_snapshot(
        entries=json.loads((evals_root / "entries.json").read_text()),
        evaluation_config=json.loads((evals_root / "evaluation_config.json").read_text()),
        referenced_files={
            "files/input.txt": evals_root / "files/input.txt",
            "graders/domain.py": evals_root / "graders/domain.py",
        },
    )
    assert (
        canonical_json_bytes(evals_snapshot, trailing_newline=True)
        == (evals_root / "expected.snapshot.json").read_bytes()
    )
    assert canonical_digest(evals_snapshot) == expected_digests["evals_json"]

    native_root = DATASET_FIXTURE_ROOT / "native_harbor"
    native_snapshot = build_native_harbor_snapshot(
        native_root,
        task_ids=json.loads((native_root / "task_ids.json").read_text()),
    )
    assert (
        canonical_json_bytes(native_snapshot, trailing_newline=True)
        == (native_root / "expected.snapshot.json").read_bytes()
    )
    assert canonical_digest(native_snapshot) == expected_digests["native_harbor"]


def test_ledger_rejects_missing_extra_and_duplicate_attempts(tmp_path: Path) -> None:
    plan = _attempt_plan(baseline_required=False, n_attempts=1)
    plan_digest = _persist_plan(tmp_path, plan)
    records = _all_blocked_records(plan)
    blocker = FailureRecord("run", "execution", "shared_configuration_invalid")
    kwargs = {
        "plan": plan,
        "task_plan_digest": plan_digest,
        "trusted_root": tmp_path,
        "job_evidence": [],
        "run_blockers": [blocker],
    }
    with pytest.raises(ContractError, match="missing"):
        build_execution_ledger(records=records[:-1], **kwargs)
    with pytest.raises(ContractError, match="duplicate"):
        build_execution_ledger(records=[*records, records[0]], **kwargs)
    extra = AttemptRecord(ExpectedAttempt("codex", "with_skill", "case-x", 1), "not_run_run_blocked", None, None)
    with pytest.raises(ContractError, match="extra|unbound"):
        build_execution_ledger(records=[*records[:-1], extra], **kwargs)


def test_missing_attempt_never_becomes_stop_on_pass_skip(tmp_path: Path) -> None:
    plan = _attempt_plan(baseline_required=False, n_attempts=2, stop_on_pass=True)
    plan_digest = _persist_plan(tmp_path, plan)
    records = _all_blocked_records(plan)
    records.pop()
    with pytest.raises(ContractError, match="missing"):
        build_execution_ledger(
            plan=plan,
            task_plan_digest=plan_digest,
            records=records,
            trusted_root=tmp_path,
            job_evidence=[],
            run_blockers=[FailureRecord("run", "execution", "shared_configuration_invalid")],
        )


def test_explicit_skip_requires_prior_scored_passing_attempt(tmp_path: Path) -> None:
    plan = _attempt_plan(baseline_required=False, n_attempts=2, stop_on_pass=True)
    plan_digest = _persist_plan(tmp_path, plan)
    records = _all_blocked_records(plan)
    records[2] = AttemptRecord(records[2].expected, "skipped_stop_on_pass", None, None)
    with pytest.raises(ContractError, match="prior scored pass"):
        build_execution_ledger(
            plan=plan,
            task_plan_digest=plan_digest,
            records=records,
            trusted_root=tmp_path,
            job_evidence=[],
            run_blockers=[FailureRecord("run", "execution", "shared_configuration_invalid")],
        )


def test_preflight_exclusion_emits_not_run_for_every_attempt(tmp_path: Path) -> None:
    plan = _attempt_plan(baseline_required=False, n_attempts=1)
    plan_digest = _persist_plan(tmp_path, plan)
    records = _all_blocked_records(plan)
    for index, record in enumerate(records):
        if record.expected.agent == "codex":
            records[index] = AttemptRecord(record.expected, "not_run_agent_excluded", None, None)
    (tmp_path / "diagnostics/codex").mkdir(parents=True)
    failure_ref = "diagnostics/codex/preflight.json"
    unbound = FailureRecord("agent", "preflight", "agent_runtime_unavailable", agent="codex")
    failure_digest = write_failure_evidence(
        tmp_path,
        failure_ref,
        unbound,
        skill_logic_started=False,
        exception_type="AgentRuntimeUnavailable",
    )
    exclusion = FailureRecord(
        "agent",
        "preflight",
        "agent_runtime_unavailable",
        agent="codex",
        evidence_ref=failure_ref,
        evidence_file_digest=failure_digest,
    )
    ledger = build_execution_ledger(
        plan=plan,
        task_plan_digest=plan_digest,
        records=records,
        trusted_root=tmp_path,
        job_evidence=[],
        agent_exclusions={"codex": exclusion},
        run_blockers=[FailureRecord("run", "execution", "shared_configuration_invalid")],
    )
    assert sum(row["disposition"] == "not_run_agent_excluded" for row in ledger["entries"]) == 2


def test_launched_failure_origin_cannot_authorize_zero_projection_exclusion(
    tmp_path: Path,
) -> None:
    plan = _attempt_plan(baseline_required=False, n_attempts=1)
    plan_digest = _persist_plan(tmp_path, plan)
    records = _all_blocked_records(plan)
    for index, record in enumerate(records):
        if record.expected.agent == "codex":
            records[index] = AttemptRecord(record.expected, "not_run_agent_excluded", None, None)
    invented = FailureRecord(
        "agent",
        "agent_adapter_bootstrap",
        "adapter_model_protocol_negotiation_failed",
        origin="trusted_adapter_marker",
        agent="codex",
    )

    with pytest.raises(ContractError, match="trusted preflight provenance"):
        build_execution_ledger(
            plan=plan,
            task_plan_digest=plan_digest,
            records=records,
            trusted_root=tmp_path,
            job_evidence=[],
            agent_exclusions={"codex": invented},
            run_blockers=[FailureRecord("run", "execution", "shared_configuration_invalid")],
        )


def test_arm_summary_keeps_case_and_attempt_counts_distinct(tmp_path: Path) -> None:
    plan = _attempt_plan(baseline_required=False, n_attempts=2)
    plan_digest = _persist_plan(tmp_path, plan)
    ledger = build_execution_ledger(
        plan=plan,
        task_plan_digest=plan_digest,
        records=_all_blocked_records(plan),
        trusted_root=tmp_path,
        job_evidence=[],
        run_blockers=[FailureRecord("run", "execution", "shared_configuration_invalid")],
    )
    summary = arm_summaries_from_ledger(plan, ledger)["claude-code"]["with_skill"]
    assert summary == {
        "expected_cases": 2,
        "scored_cases": 0,
        "exceptions": 0,
        "expected_attempts": 4,
        "scored_attempts": 0,
        "failed_attempts": 0,
        "skipped_attempts": 0,
        "not_run_attempts": 4,
    }

    tampered_skip = deepcopy(ledger)
    tampered_skip["entries"][0]["disposition"] = "skipped_stop_on_pass"
    with pytest.raises(ContractError, match="stop-on-pass"):
        validate_execution_ledger(
            plan,
            tampered_skip,
            trusted_root=tmp_path,
            run_blockers=[FailureRecord("run", "execution", "shared_configuration_invalid")],
        )

    tampered_exclusion = deepcopy(ledger)
    tampered_exclusion["entries"][0]["disposition"] = "not_run_agent_excluded"
    with pytest.raises(ContractError, match="no typed authorizer"):
        validate_execution_ledger(
            plan,
            tampered_exclusion,
            trusted_root=tmp_path,
            run_blockers=[FailureRecord("run", "execution", "shared_configuration_invalid")],
        )


@pytest.mark.parametrize("schema_path", [PLAN_SCHEMA_PATH, LEDGER_SCHEMA_PATH])
def test_task2_schemas_are_closed_draft_2020_12(schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


def test_reward_contract_accepts_real_template_shapes_and_checks_overall() -> None:
    default_values = {
        metric: value
        for metric, value in zip(
            coverage.DEFAULT_METRICS,
            (1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
            strict=True,
        )
    }
    default_values["overall"] = 0.5
    default_contract = build_reward_contract("default")
    validate_projected_reward_contract(default_values, default_contract)

    schema_digest = "sha256:" + "d" * 64
    plus_contract = build_reward_contract(
        "default_plus_custom",
        custom_metrics=[{"name": "domain_quality", "range": "unit_interval"}],
        custom_grader_schema_digest=schema_digest,
    )
    validate_projected_reward_contract({**default_values, "domain_quality": 0.75}, plus_contract)
    custom_contract = build_reward_contract(
        "custom_only",
        custom_metrics=[{"name": "domain_quality", "range": "unit_interval"}],
        custom_grader_schema_digest=schema_digest,
    )
    validate_projected_reward_contract({"overall": 0.0, "domain_quality": 1.0}, custom_contract)

    with pytest.raises(ContractError, match="canonical SkillEvaluator mean"):
        validate_projected_reward_contract({**default_values, "overall": 0.5001}, default_contract)
    with pytest.raises(ContractError, match="keys disagree"):
        validate_projected_reward_contract({"reward": 0.5}, custom_contract)
    with pytest.raises(ContractError, match="outside"):
        validate_projected_reward_contract({"overall": 0.5, "domain_quality": 1.000001}, custom_contract)

    multistep_mean = {metric: (0.5 if index == 0 else 0.0) for index, metric in enumerate(coverage.DEFAULT_METRICS)}
    multistep_mean["overall"] = 0.08335
    validate_projected_reward_contract(
        multistep_mean,
        default_contract,
        reward_strategy="multi_step_mean",
    )


def test_reward_contract_runtime_and_plan_schema_boundary_parity() -> None:
    validator = Draft202012Validator(json.loads(PLAN_SCHEMA_PATH.read_text()))
    schema_digest = "sha256:" + "d" * 64
    valid_contracts = [
        build_reward_contract("default"),
        build_reward_contract("custom_only"),
        build_reward_contract(
            "custom_only",
            custom_metrics=[{"name": "domain_quality", "range": "unit_interval"}],
            custom_grader_schema_digest=schema_digest,
        ),
        build_reward_contract(
            "default_plus_custom",
            custom_metrics=[{"name": "domain_quality", "range": "unit_interval"}],
            custom_grader_schema_digest=schema_digest,
        ),
    ]
    for contract in valid_contracts:
        plan = deepcopy(_attempt_plan(baseline_required=False, n_attempts=1))
        plan["grading_mode"] = contract["grading_mode"]
        plan["reward_contract"] = contract
        validate_expected_attempt_plan(plan)
        assert not list(validator.iter_errors(plan))

    base_custom = build_reward_contract(
        "custom_only",
        custom_metrics=[{"name": "domain_quality", "range": "unit_interval"}],
        custom_grader_schema_digest=schema_digest,
    )
    invalid_contracts: list[dict[str, object]] = []
    orphan_digest = build_reward_contract("custom_only")
    orphan_digest["custom_grader_schema_digest"] = schema_digest
    invalid_contracts.append(orphan_digest)
    duplicate = deepcopy(base_custom)
    duplicate["custom_metrics"].append(deepcopy(duplicate["custom_metrics"][0]))
    invalid_contracts.append(duplicate)
    reserved = deepcopy(base_custom)
    reserved["custom_metrics"][0]["name"] = "overall"
    invalid_contracts.append(reserved)
    plus_overflow = deepcopy(valid_contracts[-1])
    plus_overflow["custom_metrics"] = [{"name": f"metric_{index}", "range": "unit_interval"} for index in range(250)]
    invalid_contracts.append(plus_overflow)
    custom_overflow = deepcopy(base_custom)
    custom_overflow["custom_metrics"] = [{"name": f"metric_{index}", "range": "unit_interval"} for index in range(256)]
    invalid_contracts.append(custom_overflow)

    for contract in invalid_contracts:
        plan = deepcopy(_attempt_plan(baseline_required=False, n_attempts=1))
        plan["grading_mode"] = contract["grading_mode"]
        plan["reward_contract"] = contract
        with pytest.raises(ContractError):
            validate_reward_contract(contract)
        assert list(validator.iter_errors(plan)), contract


def test_runtime_and_schema_accept_same_plan_and_ledger(tmp_path: Path) -> None:
    plan = _attempt_plan(baseline_required=False, n_attempts=1)
    plan_digest = _persist_plan(tmp_path, plan)
    validate_expected_attempt_plan(plan)
    Draft202012Validator(json.loads(PLAN_SCHEMA_PATH.read_text())).validate(plan)
    ledger = build_execution_ledger(
        plan=plan,
        task_plan_digest=plan_digest,
        records=_all_blocked_records(plan),
        trusted_root=tmp_path,
        job_evidence=[],
        run_blockers=[FailureRecord("run", "execution", "shared_configuration_invalid")],
    )
    validate_execution_ledger(
        plan,
        ledger,
        trusted_root=tmp_path,
        run_blockers=[FailureRecord("run", "execution", "shared_configuration_invalid")],
    )
    Draft202012Validator(json.loads(LEDGER_SCHEMA_PATH.read_text())).validate(ledger)


@pytest.mark.parametrize(
    ("with_root", "baseline_root"),
    [
        ("staged/with_skill", "STAGED/WITH_SKILL/child"),
        ("staged/Caf\u00e9", "STAGED/cafe\u0301/child"),
    ],
)
def test_plan_rejects_normalized_or_nested_arm_task_roots(with_root: str, baseline_root: str) -> None:
    plan = _attempt_plan(baseline_required=True, n_attempts=1)
    for task_set, root_ref in zip(plan["arm_task_sets"], (with_root, baseline_root), strict=True):
        task_set["root_ref"] = root_ref
        task_set["task_set_digest"] = canonical_digest(
            {key: value for key, value in task_set.items() if key != "task_set_digest"}
        )

    with pytest.raises(ContractError, match="disjoint"):
        validate_expected_attempt_plan(plan)


def test_plan_rejects_casefolded_and_non_ascii_task_name_collisions() -> None:
    plan = _attempt_plan(baseline_required=False, n_attempts=1)
    plan["cases"][0]["harbor_task_name"] = "Task"
    plan["cases"][1]["harbor_task_name"] = "task"
    task_set = plan["arm_task_sets"][0]
    task_set["tasks"][0]["harbor_task_name"] = "Task"
    task_set["tasks"][1]["harbor_task_name"] = "task"
    task_set["task_set_digest"] = canonical_digest(
        {key: value for key, value in task_set.items() if key != "task_set_digest"}
    )
    with pytest.raises(ContractError, match="task-name collision"):
        validate_expected_attempt_plan(plan)

    non_ascii = deepcopy(plan)
    non_ascii["cases"][1]["harbor_task_name"] = "Caf\u00e9"
    with pytest.raises(ContractError, match="identity"):
        validate_expected_attempt_plan(non_ascii)


def test_failure_evidence_runtime_and_schema_parity(tmp_path: Path) -> None:
    failure = FailureRecord(
        "agent",
        "agent_adapter_bootstrap",
        "adapter_model_protocol_negotiation_failed",
        origin="trusted_adapter_marker",
        agent="codex",
    )
    ref = "diagnostics/codex/protocol.json"
    (tmp_path / "diagnostics/codex").mkdir(parents=True)
    digest = write_failure_evidence(
        tmp_path,
        ref,
        failure,
        skill_logic_started=False,
        http_status=400,
        exception_type="Adapter.ProtocolError",
    )
    payload = json.loads((tmp_path / ref).read_text())
    validate_failure_evidence(payload, expected=failure)
    schema = json.loads(FAILURE_SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    assert digest.startswith("sha256:")


@pytest.mark.parametrize(
    "unsafe",
    [
        {"message": "Authorization: Bearer secret"},
        {"headers": {"Authorization": "secret"}},
        {"response_body": "400 Bad Request"},
        {"environment": {"PRIVATE_PROVIDER_TOKEN": "secret"}},
        {"traceback": "raw model/tool output"},
    ],
)
def test_failure_evidence_rejects_free_form_or_secret_shaped_fields(unsafe: dict[str, object]) -> None:
    payload = {
        "schema_version": "1.0",
        "scope": "agent",
        "stage": "agent_adapter_bootstrap",
        "reason_code": "adapter_model_protocol_negotiation_failed",
        "origin": "trusted_adapter_marker",
        "agent": "codex",
        "skill_logic_started": False,
        **unsafe,
    }
    with pytest.raises(ContractError, match="unexpected"):
        validate_failure_evidence(payload)
    with pytest.raises(ValidationError):
        Draft202012Validator(json.loads(FAILURE_SCHEMA_PATH.read_text())).validate(payload)


def test_generic_or_content_induced_http_400_cannot_be_agent_scoped() -> None:
    with pytest.raises(ContractError, match="trusted agent taxonomy"):
        FailureRecord("agent", "agent_adapter_bootstrap", "model_endpoint_rejected", agent="codex")
    with pytest.raises(ContractError, match="skill_logic_started=false"):
        validate_failure_evidence(
            {
                "schema_version": "1.0",
                "scope": "agent",
                "stage": "agent_adapter_bootstrap",
                "reason_code": "adapter_model_protocol_negotiation_failed",
                "origin": "trusted_adapter_marker",
                "agent": "codex",
                "skill_logic_started": True,
                "http_status": 400,
            }
        )


def test_failure_taxonomy_runtime_schema_full_cartesian_parity() -> None:
    schema_validator = Draft202012Validator(json.loads(FAILURE_SCHEMA_PATH.read_text()))
    taxonomies = {
        "agent": coverage._AGENT_FAILURE_TAXONOMY,
        "run": coverage._RUN_FAILURE_TAXONOMY,
    }
    for scope, allowed in taxonomies.items():
        stages = {stage for stage, _reason in allowed}
        reasons = {reason for _stage, reason in allowed}
        for stage in stages:
            for reason in reasons:
                payload = {
                    "schema_version": "1.0",
                    "scope": scope,
                    "stage": stage,
                    "reason_code": reason,
                    "origin": (
                        "run_scope"
                        if scope == "run"
                        else (
                            "trusted_preflight"
                            if stage in {"agent_readiness", "preflight"}
                            else (
                                "harbor_pre_instruction_phase"
                                if (stage, reason) == ("agent_adapter_bootstrap", "adapter_initialization_failed")
                                else "trusted_adapter_marker"
                            )
                        )
                    ),
                    "skill_logic_started": False,
                }
                if scope == "agent":
                    payload["agent"] = "codex"
                runtime_ok = True
                try:
                    validate_failure_evidence(payload)
                except ContractError:
                    runtime_ok = False
                schema_ok = not list(schema_validator.iter_errors(payload))
                assert runtime_ok == schema_ok == ((stage, reason) in allowed)

    for stage, reason in coverage._AGENT_FAILURE_TAXONOMY:
        payload = {
            "schema_version": "1.0",
            "scope": "agent",
            "stage": stage,
            "reason_code": reason,
            "origin": "trusted_adapter_marker",
            "agent": "codex",
            "skill_logic_started": False,
            "http_status": 400,
        }
        runtime_ok = True
        try:
            validate_failure_evidence(payload)
        except ContractError:
            runtime_ok = False
        schema_ok = not list(schema_validator.iter_errors(payload))
        expected = (stage, reason) == (
            "agent_adapter_bootstrap",
            "adapter_model_protocol_negotiation_failed",
        )
        assert runtime_ok == schema_ok == expected


def test_failure_origin_runtime_schema_full_cartesian_parity() -> None:
    validator = Draft202012Validator(json.loads(FAILURE_SCHEMA_PATH.read_text()))
    origins = (
        "trusted_preflight",
        "harbor_pre_instruction_phase",
        "trusted_adapter_marker",
        "run_scope",
    )
    for scope, taxonomy in (
        ("agent", coverage._AGENT_FAILURE_TAXONOMY),
        ("run", coverage._RUN_FAILURE_TAXONOMY),
    ):
        for stage, reason in taxonomy:
            for origin in origins:
                payload = {
                    "schema_version": "1.0",
                    "scope": scope,
                    "stage": stage,
                    "reason_code": reason,
                    "origin": origin,
                    "skill_logic_started": False,
                }
                if scope == "agent":
                    payload["agent"] = "codex"
                runtime_ok = True
                try:
                    validate_failure_evidence(payload)
                except ContractError:
                    runtime_ok = False
                schema_ok = not list(validator.iter_errors(payload))
                assert runtime_ok == schema_ok, (scope, stage, reason, origin)


def test_packaged_schemas_embed_the_generated_shared_failure_taxonomy() -> None:
    failure_schema = json.loads(FAILURE_SCHEMA_PATH.read_text())
    ledger_schema = json.loads(LEDGER_SCHEMA_PATH.read_text())
    coverage_schema = json.loads(SCHEMA_PATH.read_text())
    assert failure_schema["oneOf"][0]["allOf"][0] == taxonomy_schema("agent")
    assert failure_schema["oneOf"][1]["allOf"][0] == taxonomy_schema("run")
    assert ledger_schema["$defs"]["agent_failure_taxonomy"] == taxonomy_schema("agent")
    assert ledger_schema["$defs"]["run_failure_taxonomy"] == taxonomy_schema("run")
    assert coverage_schema["$defs"]["agent_failure_taxonomy"] == taxonomy_schema("agent")
    assert coverage_schema["$defs"]["agent_entry_failure_taxonomy"] == taxonomy_schema(
        "agent", stage_field="failure_stage"
    )
    assert coverage_schema["$defs"]["run_failure_taxonomy"] == taxonomy_schema("run")


def test_diagnostic_reference_and_file_digest_are_all_or_none() -> None:
    with pytest.raises(ContractError, match="all-or-none"):
        FailureRecord(
            "agent",
            "agent_adapter_bootstrap",
            "adapter_model_protocol_negotiation_failed",
            origin="trusted_adapter_marker",
            agent="codex",
            evidence_ref="diagnostics/codex/protocol.json",
        )


def test_manifest_rejects_post_publication_failure_evidence_tamper(tmp_path: Path) -> None:
    manifest = _completed_manifest(tmp_path)
    evidence_ref = manifest["agents"]["codex"]["evidence_ref"]
    evidence = tmp_path / evidence_ref
    evidence.write_bytes(evidence.read_bytes().replace(b"ProtocolNegotiationError", b"ProtocolNegotiationErroR"))
    with pytest.raises(ContractError, match="digest mismatch"):
        validate_manifest(manifest, trusted_root=tmp_path)


def test_terminal_pre_skill_agent_failure_explicitly_accounts_for_all_later_attempts(tmp_path: Path) -> None:
    cases = [
        {"case_id": "case-1", "harbor_task_name": "case-1", "reward_strategy": "single_step"},
        {"case_id": "case-2", "harbor_task_name": "case-2", "reward_strategy": "single_step"},
    ]
    tasks = [{**case, "staged_task_digest": f"sha256:{index + 1:064x}"} for index, case in enumerate(cases)]
    arm_core = {
        "arm": "with_skill",
        "root_ref": "staged/with_skill",
        "digest_algorithm": "skill-evaluator-staged-harbor-task-tree-c14n/1",
        "skill_payload_digest": "sha256:" + "f" * 64,
        "tasks": tasks,
    }
    plan = build_expected_attempt_plan(
        run_id="terminal-agent-failure",
        dataset_snapshot_kind="evals_json",
        semantic_snapshot=_semantic_snapshot(),
        agents=(_agents()[1],),
        cases=cases,
        baseline_required=False,
        n_attempts=2,
        stop_on_pass=False,
        pass_threshold=0.6,
        grading_mode="default",
        arm_task_sets=[{**arm_core, "task_set_digest": canonical_digest(arm_core)}],
    )
    plan_digest = _persist_plan(tmp_path, plan)
    cause = ExpectedAttempt("codex", "with_skill", "case-1", 1)
    failure_ref = "diagnostics/codex/adapter-protocol.json"
    (tmp_path / "diagnostics/codex").mkdir(parents=True)
    unbound_failure = FailureRecord(
        "agent",
        "agent_adapter_bootstrap",
        "adapter_model_protocol_negotiation_failed",
        origin="trusted_adapter_marker",
        agent="codex",
    )
    failure_digest = write_failure_evidence(
        tmp_path,
        failure_ref,
        unbound_failure,
        skill_logic_started=False,
        http_status=400,
        exception_type="ProtocolNegotiationError",
    )
    failure = FailureRecord(
        "agent",
        "agent_adapter_bootstrap",
        "adapter_model_protocol_negotiation_failed",
        origin="trusted_adapter_marker",
        agent="codex",
        evidence_ref=failure_ref,
        evidence_file_digest=failure_digest,
    )
    (tmp_path / "harbor-evidence/job/trials").mkdir(parents=True)
    trial_ref = "harbor-evidence/job/trials/000001.json"
    trial_digest = atomic_write_json(
        tmp_path / trial_ref,
        {
            "schema_version": "1.0",
            "plan_digest": plan_digest,
            "job_id": "job-id",
            "trial_name": "case-1__trial",
            "agent": "codex",
            "arm": "with_skill",
            "case_id": "case-1",
            "ordinal": 1,
            "reward_strategy": "single_step",
            "staged_task_digest": plan["arm_task_sets"][0]["tasks"][0]["staged_task_digest"],
            "state": "failed",
            "verifier_result_present": False,
            "rewards": None,
            "steps": [],
            "exception_type": "ProtocolNegotiationError",
            "skill_logic_started": False,
            "agent_failure": {
                "stage": "agent_adapter_bootstrap",
                "reason_code": "adapter_model_protocol_negotiation_failed",
                "origin": "trusted_adapter_marker",
            },
        },
        trusted_root=tmp_path,
    )
    schedule_ref = "harbor-evidence/job/schedule.json"
    schedule_digest = atomic_write_json(
        tmp_path / schedule_ref,
        {
            "schema_version": "1.0",
            "plan_digest": plan_digest,
            "job_id": "job-id",
            "job_name": "job",
            "agent": "codex",
            "arm": "with_skill",
            "resolved_model": "openai/openai/gpt-5.4",
            "harbor_model": "openai/openai/gpt-5.4",
            "reward_contract_digest": canonical_digest(plan["reward_contract"]),
            "task_set_digest": plan["arm_task_sets"][0]["task_set_digest"],
            "trials": [
                {
                    "trial_name": "case-1__trial",
                    "agent": "codex",
                    "arm": "with_skill",
                    "case_id": "case-1",
                    "ordinal": 1,
                    "reward_strategy": "single_step",
                    "staged_task_digest": plan["arm_task_sets"][0]["tasks"][0]["staged_task_digest"],
                }
            ],
        },
        trusted_root=tmp_path,
    )
    results_ref = "harbor-evidence/job/results.json"
    results_digest = atomic_write_json(
        tmp_path / results_ref,
        {
            "schema_version": "1.0",
            "plan_digest": plan_digest,
            "schedule_file_digest": schedule_digest,
            "job_id": "job-id",
            "job_name": "job",
            "agent": "codex",
            "arm": "with_skill",
            "resolved_model": "openai/openai/gpt-5.4",
            "harbor_model": "openai/openai/gpt-5.4",
            "reward_contract_digest": canonical_digest(plan["reward_contract"]),
            "task_set_digest": plan["arm_task_sets"][0]["task_set_digest"],
            "trials": [
                {
                    "trial_name": "case-1__trial",
                    "agent": "codex",
                    "arm": "with_skill",
                    "case_id": "case-1",
                    "ordinal": 1,
                    "reward_strategy": "single_step",
                    "staged_task_digest": plan["arm_task_sets"][0]["tasks"][0]["staged_task_digest"],
                    "state": "failed",
                    "verifier_result_present": False,
                    "rewards": None,
                    "steps": [],
                    "exception_type": "ProtocolNegotiationError",
                    "skill_logic_started": False,
                    "agent_failure": {
                        "stage": "agent_adapter_bootstrap",
                        "reason_code": "adapter_model_protocol_negotiation_failed",
                        "origin": "trusted_adapter_marker",
                    },
                    "trial_ref": trial_ref,
                    "trial_file_digest": trial_digest,
                }
            ],
        },
        trusted_root=tmp_path,
    )
    records = [AttemptRecord(cause, "failed", trial_ref, trial_digest, failure=failure)]
    for raw in plan["attempts"][1:]:
        expected = ExpectedAttempt(raw["agent"], raw["arm"], raw["case_id"], raw["ordinal"])
        records.append(AttemptRecord(expected, "not_run_agent_unavailable", None, None, caused_by=cause))
    ledger = build_execution_ledger(
        plan=plan,
        task_plan_digest=plan_digest,
        records=records,
        trusted_root=tmp_path,
        job_evidence=[
            {
                "job_id": "job-id",
                "job_name": "job",
                "agent": "codex",
                "arm": "with_skill",
                "schedule_ref": schedule_ref,
                "schedule_file_digest": schedule_digest,
                "results_ref": results_ref,
                "results_file_digest": results_digest,
            }
        ],
    )
    assert [entry["disposition"] for entry in ledger["entries"]] == [
        "failed",
        "not_run_agent_unavailable",
        "not_run_agent_unavailable",
        "not_run_agent_unavailable",
    ]
    assert all(entry["caused_by"] == ledger["entries"][1]["caused_by"] for entry in ledger["entries"][1:])
