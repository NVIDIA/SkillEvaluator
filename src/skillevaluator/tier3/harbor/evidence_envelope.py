# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed CI agent-eval evidence envelope shared by compatibility surfaces.

Both compatibility surfaces that publish a negotiated agent-coverage result
build the envelope here so the closed shape cannot drift between them. The
field set is exactly the seven-field closed envelope consumed by the migrated
contract (``schema_version``, ``skill_name``, ``run_dir``,
``agent_coverage_file_digest``, ``agent_coverage``, ``raw_agent_evidence``,
``diagnostic_agents``); the embedded sealed manifest and digest are the exact
ones produced by the evaluator and are never recomputed or fabricated here.
"""

from __future__ import annotations

from pathlib import Path

# Fields carried into the diagnostic view of an excluded occurrence: occurrence
# identity plus the structured failure reason. Every numeric score/lift field is
# deliberately absent so an excluded agent can never leak scored evidence.
_DIAGNOSTIC_IDENTITY_FIELDS = (
    "base_agent",
    "occurrence",
    "requested_model",
    "resolved_model",
    "model_source",
    "status",
    "score_eligible",
)
_DIAGNOSTIC_REASON_FIELDS = (
    "reason_code",
    "failure_stage",
    "failure_origin",
    "evidence_ref",
    "evidence_file_digest",
)


def coverage_agent_without_numeric_scores(result: dict, key: str) -> dict:
    """Diagnostic view of an excluded occurrence from the coverage manifest.

    Derived from the manifest's excluded-agent entry: occurrence identity plus
    the structured failure reason codes. Every numeric score/lift field is
    absent (not zeroed, not null-filled) so excluded agents never surface raw
    scored evidence.
    """
    coverage = result.get("agent_coverage") or {}
    entry = (coverage.get("agents") or {}).get(key, {})
    view: dict[str, object] = {}
    for field in _DIAGNOSTIC_IDENTITY_FIELDS:
        if field in entry:
            view[field] = entry[field]
    for field in _DIAGNOSTIC_REASON_FIELDS:
        if field in entry:
            view[field] = entry[field]
    return view


def _valid_coverage_manifest(result: dict) -> dict | None:
    """Return the sealed coverage manifest if the result carries a usable one."""
    coverage = result.get("agent_coverage")
    digest = result.get("agent_coverage_file_digest")
    run_dir = result.get("run_dir")
    if not isinstance(coverage, dict) or not isinstance(digest, str) or not run_dir:
        return None
    eligible = coverage.get("eligible_agents")
    excluded = coverage.get("excluded_agents")
    # An early-terminal negotiated result (e.g. policy_validation, preflight)
    # seals a schema-valid ``invalid`` manifest with an empty eligible partition
    # and carries NO top-level ``agents`` key -- nothing was ever evaluated. A
    # missing key is therefore an empty evidence map, not a malformed result.
    agents = result.get("agents") or {}
    if not isinstance(eligible, list) or not isinstance(excluded, list) or not isinstance(agents, dict):
        return None
    # Every eligible occurrence must carry raw Harbor evidence to embed. On an
    # early-invalid manifest eligible is empty, so this holds vacuously.
    if any(key not in agents for key in eligible):
        return None
    return coverage


def build_ci_evidence_envelope(skill_name: str, result: dict) -> dict | None:
    """Build the atomic CI evidence envelope from a sealed negotiated result.

    Returns ``None`` when the result does not carry a usable, sealed coverage
    manifest -- the caller then publishes a typed evidence-production sentinel
    instead of a shaped success artifact. The embedded manifest and digest are
    the exact ones the evaluator sealed; no field is recomputed or fabricated here.
    """
    coverage = _valid_coverage_manifest(result)
    if coverage is None:
        return None
    # Missing on an early-invalid result; eligible is empty there so the
    # comprehension below produces an empty raw-evidence map.
    agents = result.get("agents") or {}
    return {
        "schema_version": "1.0",
        "skill_name": skill_name,
        "run_dir": str(Path(result["run_dir"]).resolve()),
        "agent_coverage_file_digest": result["agent_coverage_file_digest"],
        "agent_coverage": coverage,
        "raw_agent_evidence": {key: agents[key] for key in coverage["eligible_agents"]},
        "diagnostic_agents": {
            key: coverage_agent_without_numeric_scores(result, key) for key in coverage["excluded_agents"]
        },
    }
