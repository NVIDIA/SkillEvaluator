# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Harbor 0.13.2 job plugin for plan-bound Skill Evaluator evidence."""

from __future__ import annotations

import json
import math
import stat
import unicodedata
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

from harbor.job import Job
from harbor.models.agent.context import AgentContext
from harbor.models.job.plugin import BaseJobPlugin
from harbor.models.job.result import JobResult
from harbor.models.trial.config import TrialConfig

from skillevaluator.tier3.harbor.coverage import (
    MAX_RESULT_STEPS,
    MAX_REWARD_PROPERTIES,
    ContractError,
    _artifact_bytes,
    _sha256_digest,
    _validated_ref_text,
    atomic_write_json,
    canonical_digest,
    canonical_json_bytes,
    ensure_artifact_parent,
    normalized_identity_key,
    normalized_refs_overlap,
    staged_task_digest,
    validate_harbor_results,
    validate_harbor_schedule,
    validate_projected_reward_contract,
    validate_projected_step_reward_contract,
    validate_reward_contract,
)
from skillevaluator.tier3.harbor.failure_taxonomy import LAUNCHED_AGENT_FAILURE_TAXONOMY

HARBOR_VERSION = "0.13.2"
AGENT_FAILURE_MARKER = "skillevaluator.agent_failure_v1"
TRUSTED_ADAPTER_MARKER_IMPORT_PATHS: frozenset[str] = frozenset(
    {
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalCodex",
    }
)
TRUSTED_SETUP_ADAPTERS = frozenset(
    {
        ("codex", None),
        ("claude-code", None),
        ("codex", "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalCodex"),
        ("claude-code", "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalClaudeCode"),
        ("opencode", "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalOpenCode"),
    }
)
_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "plan_digest",
        "job_name",
        "agent",
        "harbor_agent",
        "harbor_agent_import_path",
        "resolved_model",
        "harbor_model",
        "reward_contract",
        "arm",
        "task_root_ref",
        "protected_task_roots",
        "digest_algorithm",
        "skill_payload_digest",
        "task_set_digest",
        "schedule_ref",
        "results_ref",
        "retained_results_prefix",
        "ordinal_base",
        "expected_n_attempts",
        "arm_tasks",
        "cases",
    }
)
_CASE_FIELDS = frozenset({"case_id", "harbor_task_name", "reward_strategy", "staged_task_digest"})


def _exact_object(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be an object")
    if set(value) != set(fields):
        raise RuntimeError(f"{name} fields do not match the pinned SkillEvaluator binding contract")
    return value


def _bounded_text(value: object, name: str, *, limit: int) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > limit:
        raise RuntimeError(f"{name} must be a non-empty bounded string")
    return value


def _digest(value: object, name: str) -> str:
    text = _bounded_text(value, name, limit=71)
    if len(text) != 71 or not text.startswith("sha256:"):
        raise RuntimeError(f"{name} must be a lowercase sha256 digest")
    try:
        int(text[7:], 16)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a lowercase sha256 digest") from error
    if text != text.lower():
        raise RuntimeError(f"{name} must be a lowercase sha256 digest")
    return text


def _normalized_ref_parts(ref: str) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", part).casefold() for part in ref.split("/"))


def _require_namespace(ref: str, namespace: str, name: str) -> None:
    parts = _normalized_ref_parts(ref)
    if len(parts) < 2 or parts[0] != namespace.casefold():
        raise ValueError(f"{name} must be below the reserved {namespace}/ namespace")


def _task_name(config: TrialConfig) -> str:
    task = config.task
    if task.path is not None:
        return task.path.name
    if task.name is not None:
        return task.name.rsplit("/", 1)[-1]
    raise RuntimeError("Harbor trial config has no pinned task identity")


def _numeric_rewards(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    rewards = getattr(value, "rewards", None)
    if rewards is None:
        return None
    if not isinstance(rewards, dict):
        raise RuntimeError("Harbor verifier rewards have an unsupported shape")
    if len(rewards) > MAX_REWARD_PROPERTIES:
        raise RuntimeError(f"Harbor verifier rewards exceed the {MAX_REWARD_PROPERTIES}-property limit")
    parsed: dict[str, float] = {}
    for key, raw in rewards.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise RuntimeError("Harbor verifier reward key is invalid")
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(float(raw)):
            raise RuntimeError("Harbor verifier reward must be finite numeric data")
        parsed[key] = float(raw)
    return parsed


def _exception_type(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "exception_type", None)
    if not isinstance(raw, str) or not raw or len(raw) > 128:
        raise RuntimeError("Harbor exception type is not a safe typed identifier")
    parts = raw.split(".")
    if not raw.isascii() or any(
        not part
        or not (part[0].isalpha() or part[0] == "_")
        or any(not (character.isalnum() or character == "_") for character in part)
        for part in parts
    ):
        raise RuntimeError("Harbor exception type is not a safe typed identifier")
    return raw


def _context_started(context: object) -> bool:
    """Derive only a boolean from typed AgentContext counters/rollout cardinality."""

    if context is None:
        return False
    for field in ("n_input_tokens", "n_cache_tokens", "n_output_tokens"):
        value = getattr(context, field, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    rollouts = getattr(context, "rollout_details", None)
    return isinstance(rollouts, list) and bool(rollouts)


def _execution_started(value: object) -> bool:
    timing = getattr(value, "agent_execution", None)
    return timing is not None and getattr(timing, "started_at", None) is not None


def _execution_completed(value: object) -> bool:
    timing = getattr(value, "agent_execution", None)
    return (
        timing is not None
        and getattr(timing, "started_at", None) is not None
        and getattr(timing, "finished_at", None) is not None
    )


def _trial_skill_logic_started(
    trial: object,
    *,
    agent_started_event: bool,
    agent_failure: dict[str, str] | None,
) -> bool:
    if agent_failure is not None:
        return False
    if agent_started_event or _execution_started(trial):
        return True
    steps = getattr(trial, "step_results", None)
    if isinstance(steps, list) and any(_execution_started(step) for step in steps):
        return True
    if _context_started(getattr(trial, "agent_result", None)):
        return True
    return isinstance(steps, list) and any(_context_started(getattr(step, "agent_result", None)) for step in steps)


def _context_agent_failure(context: object, *, allow_adapter_marker: bool) -> dict[str, str] | None:
    """Project only a closed, typed adapter classification; never inspect text."""

    if type(context) is not AgentContext:
        return None
    metadata_value = context.metadata
    if not isinstance(metadata_value, dict) or AGENT_FAILURE_MARKER not in metadata_value:
        return None
    if not allow_adapter_marker:
        raise RuntimeError("agent failure marker came from an untrusted adapter path")
    marker = metadata_value[AGENT_FAILURE_MARKER]
    if not isinstance(marker, dict) or set(marker) != {"stage", "reason_code"}:
        raise RuntimeError("agent failure marker has an unsupported shape")
    pair = (marker["stage"], marker["reason_code"])
    if pair not in LAUNCHED_AGENT_FAILURE_TAXONOMY:
        raise RuntimeError("agent failure marker is outside the launched-agent taxonomy")
    return {
        "stage": pair[0],
        "reason_code": pair[1],
        "origin": "trusted_adapter_marker",
    }


def _trial_agent_failure(
    trial: object,
    *,
    allow_adapter_marker: bool,
    allow_setup_phase: bool,
    agent_started_event: bool,
) -> dict[str, str] | None:
    markers: list[dict[str, str]] = []
    top = _context_agent_failure(getattr(trial, "agent_result", None), allow_adapter_marker=allow_adapter_marker)
    if top is not None:
        markers.append(top)
    steps = getattr(trial, "step_results", None)
    if isinstance(steps, list):
        markers.extend(
            marker
            for step in steps
            if (
                marker := _context_agent_failure(
                    getattr(step, "agent_result", None),
                    allow_adapter_marker=allow_adapter_marker,
                )
            )
            is not None
        )
    if not markers:
        environment_timing = getattr(trial, "environment_setup", None)
        setup_timing = getattr(trial, "agent_setup", None)
        execution_timing = getattr(trial, "agent_execution", None)
        exception_info = getattr(trial, "exception_info", None)
        exception_type = getattr(exception_info, "exception_type", None)
        if (
            allow_setup_phase
            and not agent_started_event
            and environment_timing is not None
            and getattr(environment_timing, "started_at", None) is not None
            and getattr(environment_timing, "finished_at", None) is not None
            and setup_timing is not None
            and getattr(setup_timing, "started_at", None) is not None
            and getattr(setup_timing, "finished_at", None) is not None
            and execution_timing is None
            and getattr(trial, "agent_result", None) is None
            and getattr(trial, "step_results", None) is None
            and getattr(trial, "verifier", None) is None
            and getattr(trial, "verifier_result", None) is None
            and getattr(exception_info, "occurred_at", None) is not None
        ):
            setup_failures = {
                "AgentSetupTimeoutError": {
                    "stage": "agent_adapter_bootstrap",
                    "reason_code": "adapter_initialization_failed",
                    "origin": "harbor_pre_instruction_phase",
                },
                "NonZeroAgentExitCodeError": {
                    "stage": "agent_adapter_bootstrap",
                    "reason_code": "adapter_initialization_failed",
                    "origin": "harbor_pre_instruction_phase",
                },
            }
            return setup_failures.get(exception_type)
        return None
    step_values = getattr(trial, "step_results", None)
    if not agent_started_event or not (
        _execution_started(trial)
        or (isinstance(step_values, list) and any(_execution_started(step) for step in step_values))
    ):
        raise RuntimeError("trusted adapter marker lacks matching AGENT_START/execution timing")
    if any(marker != markers[0] for marker in markers[1:]):
        raise RuntimeError("trial contains conflicting agent failure markers")
    return markers[0]


class SkillEvaluatorResultPlugin(BaseJobPlugin):
    """Observe one existing Harbor job; add no execution of any kind."""

    def __init__(self, *, run_root: str, binding_ref: str, binding_file_digest: str, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"unsupported Skill Evaluator plugin kwargs: {sorted(kwargs)}")
        root = Path(run_root)
        if not root.is_absolute():
            raise ValueError("run_root must be absolute")
        try:
            root_stat = root.lstat()
        except OSError as error:
            raise ValueError(f"run_root cannot be inspected: {error}") from error
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("run_root must be a regular non-symlink directory")
        self._run_root = root
        self._binding_ref = _validated_ref_text(binding_ref)
        _require_namespace(self._binding_ref, "harbor-bindings", "binding_ref")
        self._binding_file_digest = _digest(binding_file_digest, "binding_file_digest")
        self._binding = self._load_binding()
        self._schedule: dict[str, Any] | None = None
        self._schedule_file_digest: str | None = None
        self._agent_started_trials: set[str] = set()

    def _load_binding(self) -> dict[str, Any]:
        data = _artifact_bytes(self._run_root, self._binding_ref)
        if _sha256_digest(data) != self._binding_file_digest:
            raise ValueError("binding_file_digest does not match exact binding bytes")
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError("Skill Evaluator plugin binding is not valid JSON") from error
        if canonical_json_bytes(parsed, trailing_newline=True) != data:
            raise ValueError("Skill Evaluator plugin binding must use canonical JSON bytes")
        binding = _exact_object(parsed, _BINDING_FIELDS, "binding")
        if binding["schema_version"] != "1.0":
            raise ValueError("unsupported Skill Evaluator plugin binding schema")
        _digest(binding["plan_digest"], "binding.plan_digest")
        for field, limit in (
            ("job_name", 256),
            ("agent", 128),
            ("harbor_agent", 128),
            ("resolved_model", 1024),
            ("harbor_model", 1024),
        ):
            _bounded_text(binding[field], f"binding.{field}", limit=limit)
        import_path = binding["harbor_agent_import_path"]
        if import_path is not None:
            _bounded_text(import_path, "binding.harbor_agent_import_path", limit=1024)
        if binding["arm"] not in {"with_skill", "baseline"}:
            raise ValueError("binding.arm is unsupported")
        try:
            binding["reward_contract"] = validate_reward_contract(binding["reward_contract"])
        except ContractError as error:
            raise ValueError("binding.reward_contract is invalid") from error
        if binding["digest_algorithm"] != "skill-evaluator-staged-harbor-task-tree-c14n/1":
            raise ValueError("binding.digest_algorithm is unsupported")
        payload_digest = binding["skill_payload_digest"]
        if payload_digest is not None:
            _digest(payload_digest, "binding.skill_payload_digest")
        if (binding["arm"] == "with_skill") != (payload_digest is not None):
            raise ValueError("binding skill payload digest disagrees with its arm")
        task_root_ref = _validated_ref_text(binding["task_root_ref"])
        protected_task_roots_raw = binding["protected_task_roots"]
        if not isinstance(protected_task_roots_raw, list) or not protected_task_roots_raw:
            raise ValueError("binding.protected_task_roots must be a non-empty list")
        protected_task_roots = [_validated_ref_text(ref) for ref in protected_task_roots_raw]
        if task_root_ref not in protected_task_roots:
            raise ValueError("binding task root is absent from protected_task_roots")
        if task_root_ref != f"staged/{binding['arm']}":
            raise ValueError("binding task root must use the fixed staged/<arm> reference")
        if any(
            normalized_refs_overlap(left, right)
            for index, left in enumerate(protected_task_roots)
            for right in protected_task_roots[index + 1 :]
        ):
            raise ValueError("binding protected task roots overlap")
        task_root = self._run_root.joinpath(*task_root_ref.split("/"))
        try:
            task_root_stat = task_root.lstat()
        except OSError as error:
            raise ValueError(f"binding task root cannot be inspected: {error}") from error
        if stat.S_ISLNK(task_root_stat.st_mode) or not stat.S_ISDIR(task_root_stat.st_mode):
            raise ValueError("binding task root must be a regular non-symlink directory")
        _digest(binding["task_set_digest"], "binding.task_set_digest")
        refs = [
            _validated_ref_text(binding["schedule_ref"]),
            _validated_ref_text(binding["results_ref"]),
            _validated_ref_text(binding["retained_results_prefix"]),
        ]
        artifact_refs = [self._binding_ref, *refs]
        if any(
            normalized_refs_overlap(left, right)
            for index, left in enumerate(artifact_refs)
            for right in artifact_refs[index + 1 :]
        ):
            raise ValueError("Skill Evaluator plugin binding paths overlap")
        if any(normalized_refs_overlap(ref, task_root) for ref in artifact_refs for task_root in protected_task_roots):
            raise ValueError("SkillEvaluator contract artifact path overlaps the immutable task root")
        if tuple(protected_task_roots) not in {
            ("staged/with_skill",),
            ("staged/with_skill", "staged/baseline"),
        }:
            raise ValueError("binding protected task roots must use the fixed ordered staged references")
        for name, ref in zip(
            ("schedule_ref", "results_ref", "retained_results_prefix"),
            refs,
            strict=True,
        ):
            _require_namespace(ref, "harbor-evidence", name)
        ordinal_base = binding["ordinal_base"]
        if isinstance(ordinal_base, bool) or not isinstance(ordinal_base, int) or ordinal_base < 1:
            raise ValueError("binding.ordinal_base must be a positive integer")
        expected_n_attempts = binding["expected_n_attempts"]
        if (
            isinstance(expected_n_attempts, bool)
            or not isinstance(expected_n_attempts, int)
            or not 1 <= expected_n_attempts <= 1000
        ):
            raise ValueError("binding.expected_n_attempts must be between 1 and 1000")
        arm_tasks = binding["arm_tasks"]
        if not isinstance(arm_tasks, list) or not arm_tasks:
            raise ValueError("binding.arm_tasks must be a non-empty list")
        arm_case_ids: set[str] = set()
        arm_task_names: set[str] = set()
        for index, raw in enumerate(arm_tasks):
            case = _exact_object(raw, _CASE_FIELDS, f"binding.arm_tasks[{index}]")
            case_id = _bounded_text(case["case_id"], f"binding.arm_tasks[{index}].case_id", limit=256)
            task_name = _bounded_text(
                case["harbor_task_name"], f"binding.arm_tasks[{index}].harbor_task_name", limit=128
            )
            if case["reward_strategy"] not in {"single_step", "multi_step_mean", "multi_step_final"}:
                raise ValueError("binding case reward strategy is unsupported")
            _digest(
                case["staged_task_digest"],
                f"binding.arm_tasks[{index}].staged_task_digest",
            )
            task_name_key = normalized_identity_key(task_name)
            if case_id in arm_case_ids or task_name_key in arm_task_names:
                raise ValueError("binding arm task map contains duplicate identities")
            arm_case_ids.add(case_id)
            arm_task_names.add(task_name_key)
        cases = binding["cases"]
        if not isinstance(cases, list) or not cases:
            raise ValueError("binding.cases must be a non-empty scheduled subset")
        arm_by_name = {case["harbor_task_name"]: case for case in arm_tasks}
        scheduled_names: set[str] = set()
        for index, raw in enumerate(cases):
            case = _exact_object(raw, _CASE_FIELDS, f"binding.cases[{index}]")
            task_name = case.get("harbor_task_name")
            if task_name in scheduled_names or arm_by_name.get(task_name) != case:
                raise ValueError("binding.cases is not a unique exact subset of arm_tasks")
            scheduled_names.add(task_name)
        task_set_core = {
            "arm": binding["arm"],
            "root_ref": task_root_ref,
            "digest_algorithm": binding["digest_algorithm"],
            "skill_payload_digest": payload_digest,
            "tasks": [
                {
                    "case_id": case["case_id"],
                    "harbor_task_name": case["harbor_task_name"],
                    "reward_strategy": case["reward_strategy"],
                    "staged_task_digest": case["staged_task_digest"],
                }
                for case in arm_tasks
            ],
        }
        if canonical_digest(task_set_core) != binding["task_set_digest"]:
            raise ValueError("binding.task_set_digest is stale")
        return binding

    @staticmethod
    def _require_pinned_job(job: Job) -> list[TrialConfig]:
        if metadata.version("harbor") != HARBOR_VERSION:
            raise RuntimeError(f"Skill Evaluator evidence plugin requires Harbor exactly {HARBOR_VERSION}")
        if type(job) is not Job:
            raise RuntimeError("Skill Evaluator evidence plugin requires the pinned Harbor Job shape")
        if job.is_resuming:
            raise RuntimeError("SkillEvaluator contract mode does not support Harbor resume")
        if job.config.retry.max_retries != 0:
            raise RuntimeError("SkillEvaluator contract mode requires Harbor retries to be disabled")
        trial_configs = getattr(job, "_trial_configs", None)
        if not isinstance(trial_configs, list) or len(trial_configs) != len(job):
            raise RuntimeError("Harbor private resolved schedule shape changed")
        if any(type(config) is not TrialConfig for config in trial_configs):
            raise RuntimeError("Harbor private resolved schedule contains an unknown config shape")
        names = [config.trial_name for config in trial_configs]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise RuntimeError("Harbor private resolved schedule has duplicate trial identities")
        if any(config.job_id != job.id for config in trial_configs):
            raise RuntimeError("Harbor private resolved schedule job identity mismatch")
        return trial_configs

    @staticmethod
    def _ensure_parent(root: Path, ref: str) -> None:
        target = root.joinpath(*ref.split("/"))
        ensure_artifact_parent(target, trusted_root=root)

    async def on_job_start(self, job: Job) -> None:
        trial_configs = self._require_pinned_job(job)
        binding = self._binding
        if job.config.job_name != binding["job_name"]:
            raise RuntimeError("Harbor job name does not match trusted Skill Evaluator binding")
        case_by_task = {case["harbor_task_name"]: case for case in binding["cases"]}
        arm_task_names = {case["harbor_task_name"] for case in binding["arm_tasks"]}
        task_root = self._run_root.joinpath(*binding["task_root_ref"].split("/"))
        try:
            root_entries = list(task_root.iterdir())
        except OSError as error:
            raise RuntimeError(f"Harbor arm task root cannot be enumerated: {error}") from error
        if {entry.name for entry in root_entries} != arm_task_names or any(
            entry.is_symlink() or not entry.is_dir() for entry in root_entries
        ):
            raise RuntimeError("Harbor arm root contains an extra, missing, or unsafe task entry")
        ordinal_counts: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        for config in trial_configs:
            if config.agent.model_name != binding["harbor_model"]:
                raise RuntimeError("Harbor model does not match trusted Skill Evaluator binding")
            expected_import_path = binding["harbor_agent_import_path"]
            if config.agent.import_path is not None:
                if config.agent.import_path != expected_import_path:
                    raise RuntimeError("Harbor agent import path does not match trusted Skill Evaluator binding")
            elif expected_import_path is not None or config.agent.name != binding["harbor_agent"]:
                raise RuntimeError("Harbor agent does not match trusted Skill Evaluator binding")
            task_name = _task_name(config)
            case = case_by_task.get(task_name)
            if case is None:
                raise RuntimeError(f"Harbor schedule contains unknown task outside SkillEvaluator binding: {task_name}")
            task_path = config.task.path
            if task_path is None:
                raise RuntimeError("SkillEvaluator contract mode requires a staged Harbor task path")
            try:
                resolved_task = task_path.resolve(strict=True)
                resolved_root = task_root.resolve(strict=True)
            except OSError as error:
                raise RuntimeError(f"Harbor staged task cannot be resolved: {error}") from error
            if task_path.is_symlink() or resolved_task.parent != resolved_root or resolved_task.name != task_name:
                raise RuntimeError("Harbor staged task is outside the arm-specific trusted root")
            if staged_task_digest(task_path) != case["staged_task_digest"]:
                raise RuntimeError("Harbor staged task bytes do not match the immutable arm binding")
            ordinal_counts[task_name] += 1
            rows.append(
                {
                    "trial_name": config.trial_name,
                    "agent": binding["agent"],
                    "arm": binding["arm"],
                    "case_id": case["case_id"],
                    "ordinal": binding["ordinal_base"] + ordinal_counts[task_name] - 1,
                    "reward_strategy": case["reward_strategy"],
                    "staged_task_digest": case["staged_task_digest"],
                }
            )
        if set(ordinal_counts) != set(case_by_task):
            raise RuntimeError("Harbor schedule is missing a task from the trusted Skill Evaluator binding")
        if any(count != binding["expected_n_attempts"] for count in ordinal_counts.values()):
            raise RuntimeError("Harbor schedule attempt count disagrees with the trusted Skill Evaluator binding")
        scheduled_names = {row["trial_name"] for row in rows}

        async def _record_agent_started(event: object) -> None:
            trial_id = getattr(event, "trial_id", None)
            if trial_id not in scheduled_names:
                raise RuntimeError("Harbor AGENT_START hook reported an unknown trial")
            self._agent_started_trials.add(trial_id)

        job.on_agent_started(_record_agent_started)
        schedule = {
            "schema_version": "1.0",
            "plan_digest": binding["plan_digest"],
            "job_id": str(job.id),
            "job_name": binding["job_name"],
            "agent": binding["agent"],
            "arm": binding["arm"],
            "resolved_model": binding["resolved_model"],
            "harbor_model": binding["harbor_model"],
            "reward_contract_digest": canonical_digest(binding["reward_contract"]),
            "task_set_digest": binding["task_set_digest"],
            "trials": rows,
        }
        validate_harbor_schedule(schedule)
        self._ensure_parent(self._run_root, binding["schedule_ref"])
        self._schedule_file_digest = atomic_write_json(
            self._run_root.joinpath(*binding["schedule_ref"].split("/")),
            schedule,
            trusted_root=self._run_root,
        )
        self._schedule = schedule

    @staticmethod
    def _minimal_steps(trial: object) -> list[dict[str, Any]]:
        raw_steps = getattr(trial, "step_results", None)
        if raw_steps is None:
            return []
        if not isinstance(raw_steps, list):
            raise RuntimeError("Harbor step_results has an unsupported shape")
        if len(raw_steps) > MAX_RESULT_STEPS:
            raise RuntimeError(f"Harbor step_results exceed the {MAX_RESULT_STEPS}-item limit")
        return [
            {
                "step_name": _bounded_text(getattr(step, "step_name", None), "step_name", limit=128),
                "verifier_result_present": getattr(step, "verifier_result", None) is not None,
                "rewards": _numeric_rewards(getattr(step, "verifier_result", None)),
                "exception_type": _exception_type(getattr(step, "exception_info", None)),
            }
            for step in raw_steps
        ]

    async def on_job_end(self, job_result: JobResult) -> None:
        if self._schedule is None or self._schedule_file_digest is None:
            raise RuntimeError("SkillEvaluator schedule was not published before job completion")
        if type(job_result) is not JobResult:
            raise RuntimeError("Harbor JobResult shape changed")
        schedule = self._schedule
        if str(job_result.id) != schedule["job_id"] or job_result.n_total_trials != len(schedule["trials"]):
            raise RuntimeError("Harbor JobResult does not match the captured schedule")
        raw_results = job_result.trial_results
        if not isinstance(raw_results, list):
            raise RuntimeError("Harbor JobResult trial_results shape changed")
        by_name: dict[str, object] = {}
        for trial in raw_results:
            name = getattr(trial, "trial_name", None)
            if not isinstance(name, str) or name in by_name:
                raise RuntimeError("Harbor JobResult contains a duplicate result identity")
            by_name[name] = trial
        scheduled_names = [row["trial_name"] for row in schedule["trials"]]
        if set(by_name) != set(scheduled_names):
            raise RuntimeError("Harbor JobResult has missing or extra result identities")

        binding = self._binding
        result_rows: list[dict[str, Any]] = []
        for index, scheduled in enumerate(schedule["trials"]):
            trial = by_name[scheduled["trial_name"]]
            agent_started_event = scheduled["trial_name"] in self._agent_started_trials
            raw_step_results = getattr(trial, "step_results", None)
            any_execution_started = _execution_started(trial) or (
                isinstance(raw_step_results, list) and any(_execution_started(step) for step in raw_step_results)
            )
            if any_execution_started and not agent_started_event:
                raise RuntimeError("Harbor agent timing exists without the pinned AGENT_START event")
            import_path = binding["harbor_agent_import_path"]
            agent_failure = _trial_agent_failure(
                trial,
                allow_adapter_marker=import_path in TRUSTED_ADAPTER_MARKER_IMPORT_PATHS,
                allow_setup_phase=(binding["harbor_agent"], import_path) in TRUSTED_SETUP_ADAPTERS,
                agent_started_event=agent_started_event,
            )
            raw_rewards = _numeric_rewards(getattr(trial, "verifier_result", None))
            verifier_present = getattr(trial, "verifier_result", None) is not None
            steps = self._minimal_steps(trial)
            exception_type = _exception_type(getattr(trial, "exception_info", None))
            any_step_exception = isinstance(raw_step_results, list) and any(
                _exception_type(getattr(step, "exception_info", None)) is not None for step in raw_step_results
            )
            strategy = scheduled["reward_strategy"]
            if strategy == "single_step":
                agent_phase_completed = agent_started_event and _execution_completed(trial)
            else:
                agent_phase_completed = (
                    agent_started_event
                    and isinstance(raw_step_results, list)
                    and bool(raw_step_results)
                    and all(_execution_completed(step) for step in raw_step_results)
                )
            unclassified_failure = agent_failure is None and (
                exception_type is not None or any_step_exception or (bool(raw_rewards) and not agent_phase_completed)
            )
            if agent_failure is not None or unclassified_failure:
                rewards = None
                steps = [{**step, "rewards": None} for step in steps]
                state = "failed"
            else:
                rewards = raw_rewards
                state = "completed" if verifier_present and bool(rewards) else "failed"
            if state == "failed":
                rewards = None
                steps = [{**step, "rewards": None} for step in steps]
            else:
                validate_projected_reward_contract(
                    rewards or {},
                    binding["reward_contract"],
                    reward_strategy=strategy,
                )
                for step in steps:
                    if step["rewards"] is not None:
                        validate_projected_step_reward_contract(step["rewards"], binding["reward_contract"])
            minimal: dict[str, Any] = {
                "schema_version": "1.0",
                "plan_digest": binding["plan_digest"],
                "job_id": schedule["job_id"],
                "trial_name": scheduled["trial_name"],
                "agent": scheduled["agent"],
                "arm": scheduled["arm"],
                "case_id": scheduled["case_id"],
                "ordinal": scheduled["ordinal"],
                "reward_strategy": scheduled["reward_strategy"],
                "staged_task_digest": scheduled["staged_task_digest"],
                "state": state,
                "verifier_result_present": verifier_present,
                "rewards": rewards,
                "steps": steps,
                "exception_type": exception_type,
                "skill_logic_started": _trial_skill_logic_started(
                    trial,
                    agent_started_event=agent_started_event,
                    agent_failure=agent_failure,
                ),
                "agent_failure": agent_failure,
            }
            trial_ref = f"{binding['retained_results_prefix']}/{index + 1:06d}.json"
            self._ensure_parent(self._run_root, trial_ref)
            trial_digest = atomic_write_json(
                self._run_root.joinpath(*trial_ref.split("/")), minimal, trusted_root=self._run_root
            )
            result_rows.append(
                {
                    "trial_name": minimal["trial_name"],
                    "agent": minimal["agent"],
                    "arm": minimal["arm"],
                    "case_id": minimal["case_id"],
                    "ordinal": minimal["ordinal"],
                    "reward_strategy": minimal["reward_strategy"],
                    "staged_task_digest": minimal["staged_task_digest"],
                    "state": minimal["state"],
                    "verifier_result_present": minimal["verifier_result_present"],
                    "rewards": minimal["rewards"],
                    "steps": minimal["steps"],
                    "exception_type": minimal["exception_type"],
                    "skill_logic_started": minimal["skill_logic_started"],
                    "agent_failure": minimal["agent_failure"],
                    "trial_ref": trial_ref,
                    "trial_file_digest": trial_digest,
                }
            )
        results = {
            "schema_version": "1.0",
            "plan_digest": binding["plan_digest"],
            "schedule_file_digest": self._schedule_file_digest,
            "job_id": schedule["job_id"],
            "job_name": schedule["job_name"],
            "agent": schedule["agent"],
            "arm": schedule["arm"],
            "resolved_model": schedule["resolved_model"],
            "harbor_model": schedule["harbor_model"],
            "reward_contract_digest": schedule["reward_contract_digest"],
            "task_set_digest": schedule["task_set_digest"],
            "trials": result_rows,
        }
        validate_harbor_results(results, reward_contract=binding["reward_contract"])
        self._ensure_parent(self._run_root, binding["results_ref"])
        atomic_write_json(
            self._run_root.joinpath(*binding["results_ref"].split("/")),
            results,
            trusted_root=self._run_root,
        )


__all__ = ["SkillEvaluatorResultPlugin"]
