# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill Evaluator's in-process Tier 3 validity and result contract.

The donor repositories negotiated this contract across a subprocess boundary.
Skill Evaluator owns both sides of that boundary, so the implementation seals
the plan before Harbor starts and validates the collected partition before any
score is exposed.  Donor schemas and taxonomy remain the wire format; the
runtime integration is deliberately native.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator

from skillevaluator.constants import TIER3_LIFT_FAIL_THRESHOLD, TIER3_LIFT_PASS_THRESHOLD
from skillevaluator.tier3.dataset_utils import load_dataset_entries
from skillevaluator.tier3.harbor.coverage import (
    CAPABILITY,
    DATASET_DIGEST_ALGORITHM,
    AgentOccurrence,
    ContractError,
    CoveragePolicy,
    FailureRecord,
    atomic_write_json,
    build_evals_json_snapshot,
    build_expected_attempt_plan,
    build_harbor_case_map,
    build_native_harbor_snapshot,
    build_reward_contract,
    build_staged_arm_task_set,
    calculate_coverage,
    canonical_digest,
    ensure_artifact_parent,
    resolve_policy,
    staged_tree_digest,
    write_expected_attempt_plan,
    write_failure_evidence,
    write_manifest,
)
from skillevaluator.tier3.harbor.metrics import overall_score

TIER3_RESULT_CAPABILITY = "tier3-result/3"
SUPPORTED_CONTRACT_REQUESTS = frozenset({CAPABILITY, TIER3_RESULT_CAPABILITY})
RESULT_SCHEMA_VERSION = "3.0"
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_OCCURRENCE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[-/][a-z0-9]+)*$")


@dataclass(frozen=True)
class SealedPlan:
    """The immutable pre-execution tuple retained by the runner."""

    plan: dict[str, Any]
    digest: str
    occurrences: tuple[AgentOccurrence, ...]
    policy: CoveragePolicy
    requested_policy_digest: str
    effective_policy_digest: str
    dataset_digest: str
    policy_provenance: str


def validate_contract_requests(requests: Sequence[str]) -> tuple[str, ...]:
    """Accept only the exact legacy v3 pair (as a deprecated no-op)."""

    normalized = tuple(requests)
    if not normalized:
        return ()
    if len(normalized) != len(set(normalized)):
        raise ContractError("contract requests cannot contain duplicates")
    unknown = set(normalized) - SUPPORTED_CONTRACT_REQUESTS
    if unknown:
        raise ContractError(f"unknown Tier 3 contract request(s): {', '.join(sorted(unknown))}")
    if set(normalized) != SUPPORTED_CONTRACT_REQUESTS:
        raise ContractError("the deprecated compatibility handshake requires both agent-coverage/1 and tier3-result/3")
    return normalized


def validate_evidence_bindings(
    *,
    occurrence_id: str | None,
    expected_content_digest: str | None,
    validated_sha: str | None,
    gate_policy_digest: str | None,
) -> None:
    """Validate optional pipeline bindings before any execution starts."""

    values = (
        ("tier3_occurrence_id", occurrence_id, _OCCURRENCE_ID_PATTERN),
        ("expected_content_digest", expected_content_digest, _SHA256_DIGEST_PATTERN),
        ("validated_sha", validated_sha, _COMMIT_SHA_PATTERN),
        ("gate_policy_digest", gate_policy_digest, _SHA256_DIGEST_PATTERN),
    )
    for name, value, pattern in values:
        if value is not None and pattern.fullmatch(value) is None:
            raise ContractError(f"{name} has an invalid format")


def normalize_policy(
    *,
    mode: str,
    min_valid_agents: int | None,
    required_agents: Sequence[str],
    occurrences: Sequence[AgentOccurrence],
    contract_requests: Sequence[str] = (),
) -> tuple[CoveragePolicy, str, str, str]:
    """Normalize occurrence-aware policy using the donor's canonical join."""

    if mode not in {"all-selected", "any-valid"}:
        raise ContractError("agent_validity_policy must be 'all-selected' or 'any-valid'")
    internal_mode: Literal["all_selected", "any_valid"] = mode.replace("-", "_")  # type: ignore[assignment]
    minimum = min_valid_agents or (len(occurrences) if internal_mode == "all_selected" else 1)
    requested = CoveragePolicy(internal_mode, minimum, tuple(required_agents))
    capabilities = [CAPABILITY] if internal_mode == "any_valid" else []
    if contract_requests:
        capabilities = [CAPABILITY]
    resolution = resolve_policy(requested, occurrences, capabilities)
    return (
        resolution.effective,
        resolution.requested_digest,
        resolution.effective_digest,
        resolution.provenance,
    )


def _occurrences(agent_entries: Sequence[Mapping[str, Any]]) -> tuple[AgentOccurrence, ...]:
    values: list[AgentOccurrence] = []
    for entry in agent_entries:
        source = str(entry["model_source"])
        requested_model = str(entry["model"]) if source.startswith("CLI") else None
        values.append(
            AgentOccurrence(
                result_key=str(entry["result_agent"]),
                base_agent=str(entry["agent"]),
                occurrence=int(entry["occurrence"]),
                requested_model=requested_model,
                resolved_model=str(entry["model"]),
                model_source=source,
            )
        )
    return tuple(values)


def _referenced_eval_files(skill_path: Path) -> dict[str, Path]:
    """Bind authored fixtures and graders without following links."""

    evals_dir = skill_path / "evals"
    candidates: list[Path] = []
    files_dir = evals_dir / "files"
    if files_dir.exists():
        if files_dir.is_symlink() or not files_dir.is_dir():
            raise ContractError("evals/files must be a regular non-symlink directory")
        candidates.extend(sorted(path for path in files_dir.rglob("*") if path.is_file()))
    for relative_name in ("grader.py", "grader.sh", "tests/grader.py", "tests/grader.sh"):
        path = evals_dir / relative_name
        if path.exists():
            candidates.append(path)
    result: dict[str, Path] = {}
    for path in candidates:
        entry = path.lstat()
        if path.is_symlink() or not path.is_file() or entry.st_nlink != 1:
            raise ContractError(f"evaluation fixture must be a regular non-hardlinked file: {path}")
        try:
            relative = path.resolve(strict=True).relative_to(evals_dir.resolve(strict=True)).as_posix()
        except (OSError, ValueError) as error:
            raise ContractError(f"evaluation fixture escapes evals/: {path}") from error
        result[relative] = path
    return result


def _staged_skill_payload(task_paths: Sequence[Path], skill_name: str) -> Path:
    payloads = [path / "environment" / "skills" / skill_name for path in task_paths]
    if not payloads or any(path.is_symlink() or not path.is_dir() for path in payloads):
        raise ContractError("each with-skill task must contain the staged target skill")
    digests = {staged_tree_digest(path) for path in payloads}
    if len(digests) != 1:
        raise ContractError("with-skill tasks contain different staged target skill payloads")
    return payloads[0]


def seal_plan(
    *,
    run_dir: Path,
    run_id: str,
    skill_path: Path,
    task_source: str,
    evals_file: Path | None,
    native_harbor_dir: Path,
    evals_config: Mapping[str, Any],
    grading_mode: str,
    agent_entries: Sequence[Mapping[str, Any]],
    task_paths: Sequence[Path],
    with_skill_root: Path,
    baseline_root: Path,
    skip_baseline: bool,
    n_attempts: int,
    stop_on_pass: bool,
    pass_threshold: float,
    agent_validity_policy: str,
    min_valid_agents: int | None,
    required_agents: Sequence[str],
    contract_requests: Sequence[str] = (),
) -> SealedPlan:
    """Build and exclusively publish the complete pre-execution plan."""

    requests = validate_contract_requests(contract_requests)
    occurrences = _occurrences(agent_entries)
    policy, requested_digest, effective_digest, provenance = normalize_policy(
        mode=agent_validity_policy,
        min_valid_agents=min_valid_agents,
        required_agents=required_agents,
        occurrences=occurrences,
        contract_requests=requests,
    )

    if task_source == "evals_json":
        if evals_file is None:
            raise ContractError("evals_json plan has no dataset file")
        entries = load_dataset_entries(evals_file)
        snapshot = build_evals_json_snapshot(
            entries=entries,
            evaluation_config={"grading_mode": grading_mode, "evals_config": dict(evals_config)},
            referenced_files=_referenced_eval_files(skill_path),
        )
        case_ids = [str(entry["id"]) for entry in entries]
        case_map = build_harbor_case_map(task_paths, case_ids=case_ids)
        snapshot_kind: Literal["evals_json", "native_harbor"] = "evals_json"
    elif task_source == "native_harbor":
        task_ids = [path.name for path in task_paths]
        snapshot = build_native_harbor_snapshot(native_harbor_dir, task_ids=task_ids)
        case_map = build_harbor_case_map(task_paths)
        snapshot_kind = "native_harbor"
    else:
        raise ContractError(f"unsupported task source for sealed plan: {task_source!r}")

    arm_sets = [
        build_staged_arm_task_set(
            run_dir,
            arm="with_skill",
            task_root=with_skill_root,
            cases=case_map,
            skill_payload_path=_staged_skill_payload(task_paths, skill_path.name),
        )
    ]
    if not skip_baseline:
        arm_sets.append(
            build_staged_arm_task_set(
                run_dir,
                arm="baseline",
                task_root=baseline_root,
                cases=case_map,
                skill_payload_path=None,
            )
        )
    plan = build_expected_attempt_plan(
        run_id=run_id,
        dataset_snapshot_kind=snapshot_kind,
        semantic_snapshot=snapshot,
        agents=occurrences,
        cases=case_map,
        baseline_required=not skip_baseline,
        n_attempts=n_attempts,
        stop_on_pass=stop_on_pass,
        pass_threshold=pass_threshold,
        grading_mode=grading_mode,  # type: ignore[arg-type]
        reward_contract=_reward_contract(skill_path, evals_config, grading_mode),
        arm_task_sets=arm_sets,
    )
    digest = write_expected_attempt_plan(run_dir, plan)
    return SealedPlan(
        plan=plan,
        digest=digest,
        occurrences=occurrences,
        policy=policy,
        requested_policy_digest=requested_digest,
        effective_policy_digest=effective_digest,
        dataset_digest=str(plan["dataset_digest"]),
        policy_provenance=provenance,
    )


def _reward_contract(
    skill_path: Path,
    evals_config: Mapping[str, Any],
    grading_mode: str,
) -> dict[str, object]:
    """Bind the declared custom reward names and exact grader source bytes."""

    grading = evals_config.get("grading")
    custom_names = grading.get("custom_metrics", []) if isinstance(grading, Mapping) else []
    metrics = [{"name": str(name), "range": "unit_interval"} for name in custom_names]
    grader_digest: str | None = None
    if grading_mode in {"default_plus_custom", "custom_only"}:
        for relative in ("evals/grader.py", "evals/grader.sh", "evals/tests/grader.py", "evals/tests/grader.sh"):
            path = skill_path / relative
            if path.is_file() and not path.is_symlink():
                grader_digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                break
    return build_reward_contract(
        grading_mode,  # type: ignore[arg-type]
        custom_metrics=metrics,
        custom_grader_schema_digest=grader_digest if metrics else None,
    )


def _condition_summary(
    condition: Mapping[str, Any] | None,
    *,
    expected_cases: int,
    planned_attempts: int,
) -> dict[str, int]:
    condition = condition or {}
    expected = int(condition.get("expected_attempts", 0) or planned_attempts)
    scored = int(condition.get("scored_attempts", 0) or 0)
    failed = max(0, expected - scored) if condition.get("execution_status") == "failed" else 0
    skipped = max(0, expected - scored) if not failed else 0
    return {
        "expected_cases": expected_cases,
        "scored_cases": expected_cases if condition.get("execution_status") == "succeeded" else 0,
        "exceptions": failed,
        "expected_attempts": expected,
        "scored_attempts": scored,
        "failed_attempts": failed,
        "skipped_attempts": skipped,
        "not_run_attempts": 0,
    }


def _failure_dict(failure: FailureRecord) -> dict[str, Any]:
    value: dict[str, Any] = {
        "scope": failure.scope,
        "stage": failure.stage,
        "reason_code": failure.reason_code,
        "origin": failure.origin,
    }
    for name in ("agent", "evidence_ref", "evidence_file_digest"):
        if (item := getattr(failure, name)) is not None:
            value[name] = item
    return value


def _write_failure(run_dir: Path, failure: FailureRecord, *, index: int) -> FailureRecord:
    identity = failure.agent or "run"
    ref = f"diagnostics/contract/{index:03d}-{identity}-{failure.reason_code}.json"
    ensure_artifact_parent(run_dir / ref, trusted_root=run_dir)
    digest = write_failure_evidence(
        run_dir,
        ref,
        failure,
        skill_logic_started=failure.reason_code == "post_skill_unscored_failure",
        exception_type="Tier3ExecutionFailure",
    )
    return FailureRecord(
        failure.scope,
        failure.stage,
        failure.reason_code,
        origin=failure.origin,
        agent=failure.agent,
        evidence_ref=ref,
        evidence_file_digest=digest,
    )


def _agent_score(data: Mapping[str, Any], field: str) -> float | None:
    scores = data.get(field)
    if not isinstance(scores, dict):
        return None
    return overall_score(scores)


def _quality_for(lift: float | None, *, evaluated: bool) -> str:
    if not evaluated:
        return "not_evaluated"
    if lift is None:
        return "neutral"
    if lift >= TIER3_LIFT_PASS_THRESHOLD:
        return "pass"
    if lift <= TIER3_LIFT_FAIL_THRESHOLD:
        return "fail"
    return "neutral"


def _quality_status(per_agent: Mapping[str, Mapping[str, Any]], *, valid: bool) -> str:
    if not valid:
        return "not_evaluated"
    statuses = [str(value["status"]) for value in per_agent.values()]
    if "pass" in statuses:
        return "pass"
    if "neutral" in statuses:
        return "neutral"
    return "fail"


def _schema() -> dict[str, Any]:
    path = files("skillevaluator.tier3.harbor.schemas").joinpath("tier3_result_v3.schema.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_result(result: Mapping[str, Any]) -> None:
    errors = sorted(Draft202012Validator(_schema()).iter_errors(result), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "<root>"
        raise ContractError(f"Tier 3 result v3 is schema-invalid at {location}: {first.message}")


def _write_ledger(
    *,
    run_dir: Path,
    sealed: SealedPlan,
    agents_data: Mapping[str, Any],
    eligible: set[str],
) -> str:
    """Persist an exact planned-attempt partition from collector case summaries."""

    entries: list[dict[str, Any]] = []
    plan_cases = [str(case["case_id"]) for case in sealed.plan["cases"]]
    arms = ("with_skill", "baseline") if sealed.plan["baseline_required"] else ("with_skill",)
    for occurrence in sealed.occurrences:
        agent_data = agents_data.get(occurrence.result_key, {})
        pass_data = agent_data.get("pass_at_k", {}) if isinstance(agent_data, dict) else {}
        for arm in arms:
            pass_key = "with_skill" if arm == "with_skill" else "without_skill"
            arm_cases = pass_data.get(pass_key, {}).get("cases", {}) if isinstance(pass_data, dict) else {}
            for case_id in plan_cases:
                attempts = arm_cases.get(case_id, {}).get("attempts", []) if isinstance(arm_cases, dict) else []
                by_ordinal = {
                    int(row.get("attempt", 0)): row
                    for row in attempts
                    if isinstance(row, dict) and isinstance(row.get("attempt"), int)
                }
                passed_seen = False
                for ordinal in range(1, int(sealed.plan["n_attempts"]) + 1):
                    row = by_ordinal.get(ordinal)
                    if row is not None:
                        missing = [field for field in ("score", "passed") if field not in row]
                        if missing:
                            identity = f"{occurrence.result_key}/{arm}/{case_id}/{ordinal}"
                            raise ContractError(
                                f"collector attempt {identity} is missing required field(s): " + ", ".join(missing)
                            )
                        payload = {
                            "agent": occurrence.result_key,
                            "arm": arm,
                            "case_id": case_id,
                            "ordinal": ordinal,
                            "trial": str(row.get("trial") or ""),
                            "score": float(row["score"]),
                            "passed": bool(row["passed"]),
                        }
                        ref = f"execution-evidence/{occurrence.result_key}/{arm}/{case_id}-{ordinal:03d}.json"
                        ensure_artifact_parent(run_dir / ref, trusted_root=run_dir)
                        digest = atomic_write_json(run_dir / ref, payload, trusted_root=run_dir)
                        entries.append(
                            {
                                **{key: payload[key] for key in ("agent", "arm", "case_id", "ordinal")},
                                "disposition": "scored",
                                "trial_ref": ref,
                                "trial_file_digest": digest,
                                "passed": payload["passed"],
                                "score": payload["score"],
                                "failure": None,
                                "caused_by": None,
                            }
                        )
                        passed_seen = passed_seen or payload["passed"]
                    else:
                        disposition = (
                            "skipped_stop_on_pass"
                            if bool(sealed.plan["stop_on_pass"]) and passed_seen
                            else "not_run_agent_excluded"
                            if occurrence.result_key not in eligible
                            else "not_run_run_blocked"
                        )
                        entries.append(
                            {
                                "agent": occurrence.result_key,
                                "arm": arm,
                                "case_id": case_id,
                                "ordinal": ordinal,
                                "disposition": disposition,
                                "trial_ref": None,
                                "trial_file_digest": None,
                                "passed": None,
                                "score": None,
                                "failure": None,
                                "caused_by": None,
                            }
                        )
    ledger = {
        "schema_version": "1.0",
        "task_plan_digest": sealed.digest,
        "reward_contract_digest": canonical_digest(sealed.plan["reward_contract"]),
        "job_evidence": [],
        "entries": entries,
    }
    schema_path = files("skillevaluator.tier3.harbor.schemas").joinpath("execution_ledger_v1.schema.json")
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(ledger)
    return atomic_write_json(run_dir / "execution_ledger.json", ledger, trusted_root=run_dir)


def finalize_contract(
    *,
    run_dir: Path,
    sealed: SealedPlan,
    results: dict[str, Any],
    skill_name: str,
    environment: str,
    duration_seconds: float,
    result_file: Path | None = None,
    occurrence_id: str | None = None,
    expected_content_digest: str | None = None,
    validated_sha: str | None = None,
    gate_policy_digest: str | None = None,
) -> dict[str, Any]:
    """Enforce coverage, suppress ineligible scores, and publish result v3."""

    validate_evidence_bindings(
        occurrence_id=occurrence_id,
        expected_content_digest=expected_content_digest,
        validated_sha=validated_sha,
        gate_policy_digest=gate_policy_digest,
    )

    agents_data = results.get("agents") if isinstance(results.get("agents"), dict) else {}
    eligible: list[str] = []
    excluded: list[str] = []
    failures: dict[str, FailureRecord] = {}
    blockers: list[FailureRecord] = []
    for occurrence in sealed.occurrences:
        data = agents_data.get(occurrence.result_key, {})
        status = data.get("execution_status") if isinstance(data, dict) else None
        if status == "succeeded":
            eligible.append(occurrence.result_key)
            continue
        excluded.append(occurrence.result_key)
        runtime_failures = data.get("agent_runtime_failures", {}) if isinstance(data, dict) else {}
        has_presemantic = (
            any(runtime_failures.get(arm) for arm in ("with_skill", "without_skill"))
            if isinstance(runtime_failures, dict)
            else False
        )
        if has_presemantic:
            failure = FailureRecord(
                "agent",
                "preflight",
                "agent_runtime_unavailable",
                origin="trusted_preflight",
                agent=occurrence.result_key,
            )
            failures[occurrence.result_key] = _write_failure(run_dir, failure, index=len(failures))
        else:
            blockers.append(
                _write_failure(
                    run_dir,
                    FailureRecord("run", "execution", "post_skill_unscored_failure"),
                    index=len(failures) + len(blockers),
                )
            )

    decision = calculate_coverage(
        sealed.policy, [o.result_key for o in sealed.occurrences], eligible, excluded, blockers
    )
    eligible_set = set(decision.eligible_agents)
    ledger_digest = _write_ledger(run_dir=run_dir, sealed=sealed, agents_data=agents_data, eligible=eligible_set)
    expected_cases = len(sealed.plan["cases"])
    agent_entries: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    for occurrence in sealed.occurrences:
        data = agents_data.get(occurrence.result_key, {})
        conditions = data.get("conditions", {}) if isinstance(data, dict) else {}
        entry: dict[str, Any] = {
            "base_agent": occurrence.base_agent,
            "occurrence": occurrence.occurrence,
            "requested_model": occurrence.requested_model,
            "resolved_model": occurrence.resolved_model,
            "model_source": occurrence.model_source,
            "status": "valid" if occurrence.result_key in eligible_set else "invalid_infrastructure",
            "score_eligible": occurrence.result_key in eligible_set,
            "with_skill": _condition_summary(
                conditions.get("with_skill"),
                expected_cases=expected_cases,
                planned_attempts=expected_cases * int(sealed.plan["n_attempts"]),
            ),
            "baseline": (
                _condition_summary(
                    conditions.get("without_skill"),
                    expected_cases=expected_cases,
                    planned_attempts=expected_cases * int(sealed.plan["n_attempts"]),
                )
                if sealed.plan["baseline_required"]
                else None
            ),
        }
        if occurrence.result_key not in eligible_set:
            failure = failures.get(occurrence.result_key)
            if failure is None:
                failure = _write_failure(
                    run_dir,
                    FailureRecord(
                        "agent",
                        "preflight",
                        "agent_runtime_unavailable",
                        origin="trusted_preflight",
                        agent=occurrence.result_key,
                    ),
                    index=len(failures) + len(blockers),
                )
                failures[occurrence.result_key] = failure
            entry.update(
                {
                    "reason_code": failure.reason_code,
                    "failure_stage": failure.stage,
                    "failure_origin": failure.origin,
                    "evidence_ref": failure.evidence_ref,
                    "evidence_file_digest": failure.evidence_file_digest,
                }
            )
            if decision.status == "valid_degraded":
                warnings.append(
                    {
                        "code": "optional_agent_excluded",
                        "agent": occurrence.result_key,
                        "reason_code": failure.reason_code,
                        "failure_stage": failure.stage,
                        "failure_origin": failure.origin,
                        "evidence_ref": failure.evidence_ref,
                        "evidence_file_digest": failure.evidence_file_digest,
                    }
                )
        agent_entries[occurrence.result_key] = entry

    policy_object = {
        "mode": sealed.policy.mode,
        "min_valid_agents": sealed.policy.min_valid_agents,
        "required_agents": list(sealed.policy.required_agents),
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": str(sealed.plan["run_id"]),
        "phase": "completed",
        "capabilities": {"requested": [CAPABILITY], "provided": [CAPABILITY]},
        "requested_policy_digest": sealed.requested_policy_digest,
        "effective_policy_digest": sealed.effective_policy_digest,
        "task_plan_digest": sealed.digest,
        "task_plan_ref": "expected_attempt_plan.json",
        "execution_ledger_digest": ledger_digest,
        "execution_ledger_ref": "execution_ledger.json",
        "dataset_digest": sealed.dataset_digest,
        "dataset_digest_algorithm": DATASET_DIGEST_ALGORITHM,
        "status": decision.status,
        "requested_policy": policy_object,
        "authorized_tightening": None,
        "effective_policy": policy_object,
        "policy_provenance": sealed.policy_provenance,
        "requested_agents": [o.result_key for o in sealed.occurrences],
        "eligible_agents": list(decision.eligible_agents),
        "excluded_agents": list(decision.excluded_agents),
        "agents": agent_entries,
        "warnings": warnings,
        "blockers": [_failure_dict(item) for item in blockers],
        "extensions": {
            "org.skillevaluator/contract": {
                "execution_ledger_contract": "skill-evaluator-native/1",
                "diagnostics_redacted": True,
            }
        },
    }
    manifest_digest = write_manifest(run_dir, manifest)

    quality_agents: dict[str, dict[str, Any]] = {}
    result_agents: dict[str, dict[str, Any]] = {}
    for occurrence in sealed.occurrences:
        data = agents_data.get(occurrence.result_key, {})
        is_eligible = occurrence.result_key in eligible_set and decision.status != "invalid"
        with_score = _agent_score(data, "with_skill") if is_eligible else None
        baseline = _agent_score(data, "without_skill") if is_eligible and sealed.plan["baseline_required"] else None
        lift = round(with_score - baseline, 4) if with_score is not None and baseline is not None else None
        status = _quality_for(lift, evaluated=is_eligible)
        quality_agents[occurrence.result_key] = {"status": status, "overall_score": with_score, "lift": lift}
        pass_at_k = data.get("pass_at_k", {}) if isinstance(data, dict) else {}
        result_agents[occurrence.result_key] = {
            "base_agent": occurrence.base_agent,
            "occurrence": occurrence.occurrence,
            "requested_model": occurrence.requested_model,
            "resolved_model": occurrence.resolved_model,
            "model_source": occurrence.model_source,
            "with_skill": with_score,
            "baseline": baseline,
            "lift": lift,
            "quality_status": status,
            "with_skill_pass_rate": pass_at_k.get("with_skill", {}).get("rate") if is_eligible else None,
            "baseline_pass_rate": pass_at_k.get("without_skill", {}).get("rate") if is_eligible else None,
        }

    valid = decision.status != "invalid"
    quality_status = _quality_status(quality_agents, valid=valid)
    eligible_quality = {key: value for key, value in quality_agents.items() if key in eligible_set}
    best_agent = max(
        eligible_quality,
        key=lambda key: float(eligible_quality[key].get("overall_score") or -1),
        default=None,
    )
    overall = eligible_quality.get(best_agent or "", {}).get("overall_score") if valid else None
    lift = eligible_quality.get(best_agent or "", {}).get("lift") if valid else None
    coverage_v3 = {
        key: value
        for key, value in manifest.items()
        if key
        in {
            "status",
            "requested_policy_digest",
            "effective_policy_digest",
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
        }
    }
    coverage_v3["agents"] = {
        key: {field: value for field, value in entry.items() if field != "failure_origin"}
        for key, entry in agent_entries.items()
    }
    coverage_v3["warnings"] = [
        {field: value for field, value in warning.items() if field != "failure_origin"} for warning in warnings
    ]
    coverage_v3["blockers"] = [
        {field: value for field, value in blocker.items() if field != "origin"} for blocker in manifest["blockers"]
    ]
    tier3_result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "capabilities": {
            "requested": [CAPABILITY, TIER3_RESULT_CAPABILITY],
            "provided": [CAPABILITY, TIER3_RESULT_CAPABILITY],
        },
        "source_manifest_file_digest": manifest_digest,
        "coverage": coverage_v3,
        "quality": {
            "status": quality_status,
            "policy": {
                "schema_version": "1.0",
                "lift_pass_threshold": TIER3_LIFT_PASS_THRESHOLD,
                "lift_fail_threshold": TIER3_LIFT_FAIL_THRESHOLD,
                "aggregation": "any_pass_else_any_neutral_else_fail",
            },
            "per_agent": quality_agents,
        },
        "summary": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "skill_name": skill_name,
            "agents_run": list(decision.eligible_agents),
            "best_agent": best_agent,
            "overall_score": overall,
            "overall_lift": lift,
            "runtime_seconds": max(0.0, duration_seconds),
            "environment": environment,
        },
        "agents": result_agents,
        "overall_score": overall,
        "overall_lift": lift,
        "composite_lift": lift,
        "occurrence_id": occurrence_id,
        "expected_content_digest": expected_content_digest,
        "validated_sha": validated_sha,
        "gate_policy_digest": gate_policy_digest,
        "evidence_bundle": {
            "schema_version": "1.0",
            "root_ref": ".",
            "manifest_ref": "agent_coverage.json",
            "manifest_file_digest": manifest_digest,
        },
        "extensions": {
            "org.skillevaluator/report-provenance/1": {
                "run_id": sealed.plan["run_id"],
                "task_plan_digest": sealed.digest,
                "execution_ledger_digest": ledger_digest,
            }
        },
    }
    _validate_result(tier3_result)
    target = result_file or Path("tier3-result.json")
    if target.is_absolute() or ".." in target.parts:
        raise ContractError("tier3_result_file must be a relative path confined to the run directory")
    ensure_artifact_parent(run_dir / target, trusted_root=run_dir)
    result_digest = atomic_write_json(run_dir / target, tier3_result, trusted_root=run_dir)

    results["agent_coverage"] = manifest
    results["agent_coverage_file_digest"] = manifest_digest
    results["tier3_result"] = tier3_result
    results["tier3_result_file"] = str(run_dir / target)
    results["tier3_result_file_digest"] = result_digest
    results["quality"] = quality_status
    results["coverage_status"] = decision.status
    results["eligible_agents"] = list(decision.eligible_agents)
    results["excluded_agents"] = list(decision.excluded_agents)
    if decision.status == "invalid":
        results["overall_score"] = None
        results["overall_lift"] = None
    return results
