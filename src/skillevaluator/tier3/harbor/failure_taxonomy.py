# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source for the closed SkillEvaluator coverage-failure taxonomy."""

from __future__ import annotations

from typing import Literal

AGENT_FAILURE_REASONS: dict[str, tuple[str, ...]] = {
    "agent_readiness": (
        "adapter_unavailable",
        "model_unavailable",
        "agent_authentication_unavailable",
        "gateway_configuration_invalid",
    ),
    "preflight": (
        "agent_runtime_unavailable",
        "model_not_available",
        "gateway_configuration_invalid",
    ),
    "agent_adapter_bootstrap": (
        "adapter_initialization_failed",
        "adapter_model_protocol_negotiation_failed",
    ),
    "agent_process_bootstrap": ("process_spawn_failed", "process_start_timeout"),
    "agent_authentication": ("agent_authentication_failed",),
    "transport_before_semantic_event": (
        "transport_connection_failed",
        "transport_timeout",
        "model_rate_limited",
    ),
    "agent_execution": (
        "agent_execution_timeout",
        "agent_process_exit",
    ),
}

RUN_FAILURE_REASONS: dict[str, tuple[str, ...]] = {
    "policy_validation": ("invalid_policy", "policy_validation_failed"),
    "dataset_validation": ("missing_dataset", "invalid_dataset", "dataset_validation_failed"),
    "task_generation": ("task_generation_failed", "staged_task_collision"),
    "preflight": ("preflight_failed", "shared_environment_failure"),
    "execution": (
        "execution_failed",
        "shared_configuration_invalid",
        "evidence_integrity_failure",
        "evidence_production_failure",
        "unsatisfied_required_agent",
        "post_skill_unscored_failure",
    ),
    "verifier": ("grader_contract_failure", "verifier_contract_failure"),
    "manifest_validation": ("corrupt_manifest", "contradictory_manifest"),
    "security": ("evaluation_integrity_failure",),
}


def _pairs(values: dict[str, tuple[str, ...]]) -> frozenset[tuple[str, str]]:
    return frozenset((stage, reason) for stage, reasons in values.items() for reason in reasons)


AGENT_FAILURE_TAXONOMY = _pairs(AGENT_FAILURE_REASONS)
RUN_FAILURE_TAXONOMY = _pairs(RUN_FAILURE_REASONS)
LAUNCHED_AGENT_FAILURE_TAXONOMY = frozenset(
    (stage, reason)
    for stage, reasons in AGENT_FAILURE_REASONS.items()
    if stage
    in {
        "agent_adapter_bootstrap",
        "agent_process_bootstrap",
        "agent_authentication",
        "transport_before_semantic_event",
    }
    for reason in reasons
)

# Typed Harbor exceptions that are safe to classify as occurrence-scoped
# execution failures after the agent has started. These are process/runtime
# outcomes, not grader, dataset, or shared-evidence failures.
TRUSTED_AGENT_EXECUTION_EXCEPTIONS: dict[str, str] = {
    "AgentTimeoutError": "agent_execution_timeout",
    "NonZeroAgentExitCodeError": "agent_process_exit",
}


def taxonomy_schema(scope: Literal["agent", "run"], *, stage_field: str = "stage") -> dict[str, object]:
    """Generate the closed JSON-Schema branch embedded in packaged contracts."""

    values = AGENT_FAILURE_REASONS if scope == "agent" else RUN_FAILURE_REASONS
    branches: list[dict[str, object]] = []
    for stage, reasons in values.items():
        reason_schema: dict[str, object]
        if len(reasons) == 1:
            reason_schema = {"const": reasons[0]}
        else:
            reason_schema = {"enum": list(reasons)}
        branches.append(
            {
                "properties": {
                    stage_field: {"const": stage},
                    "reason_code": reason_schema,
                }
            }
        )
    return {"oneOf": branches}


__all__ = [
    "AGENT_FAILURE_REASONS",
    "AGENT_FAILURE_TAXONOMY",
    "LAUNCHED_AGENT_FAILURE_TAXONOMY",
    "RUN_FAILURE_REASONS",
    "RUN_FAILURE_TAXONOMY",
    "TRUSTED_AGENT_EXECUTION_EXCEPTIONS",
    "taxonomy_schema",
]
