# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from harbor.cli.job_plugins import attach_job_plugin, finalize_job_plugins
from harbor.job import Job
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import AgentConfig, TaskConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, StepResult, TimingInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from jsonschema import Draft202012Validator

from skillevaluator.tier3.harbor.coverage import (
    ContractError,
    atomic_write_json,
    build_reward_contract,
    canonical_digest,
    canonical_json_bytes,
    staged_task_digest,
)
from skillevaluator.tier3.harbor.result_plugin import SkillEvaluatorResultPlugin

ROOT = Path(__file__).parents[2]
SCHEDULE_SCHEMA = ROOT / "src/skillevaluator/tier3/harbor/schemas/harbor_schedule_v1.schema.json"
RESULTS_SCHEMA = ROOT / "src/skillevaluator/tier3/harbor/schemas/harbor_results_v1.schema.json"


def _binding(
    run_root: Path,
    *,
    task_names: tuple[str, ...],
    ordinal_base: int = 1,
    arm: str = "with_skill",
    import_path: str | None = None,
    resolved_model: str = "openai/gpt-test",
    harbor_model: str | None = None,
    scheduled_task_names: tuple[str, ...] | None = None,
    expected_n_attempts: int = 1,
    reward_contract: dict[str, object] | None = None,
) -> tuple[str, str]:
    ref = "harbor-bindings/job.json"
    (run_root / "harbor-bindings").mkdir(parents=True)
    task_root_ref = f"staged/{arm}"
    task_root = run_root / task_root_ref
    for name in task_names:
        task_dir = task_root / name
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.toml").write_text('schema_version = "1.0"\n', encoding="utf-8")
    cases = [
        {
            "case_id": name,
            "harbor_task_name": name,
            "reward_strategy": "single_step",
            "staged_task_digest": staged_task_digest(task_root / name),
        }
        for name in task_names
    ]
    task_set_core = {
        "arm": arm,
        "root_ref": task_root_ref,
        "digest_algorithm": "skill-evaluator-staged-harbor-task-tree-c14n/1",
        "skill_payload_digest": "sha256:" + "f" * 64 if arm == "with_skill" else None,
        "tasks": cases,
    }
    value = {
        "schema_version": "1.0",
        "plan_digest": "sha256:" + "a" * 64,
        "job_name": "job",
        "agent": "codex",
        "harbor_agent": "codex",
        "harbor_agent_import_path": import_path,
        "resolved_model": resolved_model,
        "harbor_model": harbor_model or resolved_model,
        "reward_contract": reward_contract or build_reward_contract("custom_only"),
        "arm": arm,
        "task_root_ref": task_root_ref,
        "protected_task_roots": [task_root_ref],
        "digest_algorithm": "skill-evaluator-staged-harbor-task-tree-c14n/1",
        "skill_payload_digest": "sha256:" + "f" * 64,
        "task_set_digest": canonical_digest(task_set_core),
        "schedule_ref": "harbor-evidence/job/schedule.json",
        "results_ref": "harbor-evidence/job/results.json",
        "retained_results_prefix": "harbor-evidence/job/trials",
        "ordinal_base": ordinal_base,
        "expected_n_attempts": expected_n_attempts,
        "arm_tasks": cases,
        "cases": [
            case for case in cases if scheduled_task_names is None or case["harbor_task_name"] in scheduled_task_names
        ],
    }
    digest = atomic_write_json(run_root / ref, value, trusted_root=run_root)
    return ref, digest


def _job(
    tmp_path: Path,
    *,
    task_names: tuple[str, ...] = ("case-1",),
    n_attempts: int = 1,
    arm: str = "with_skill",
    import_path: str | None = None,
    model_name: str = "openai/gpt-test",
) -> Job:
    task_root = tmp_path / f"staged/{arm}"
    for name in task_names:
        task_dir = task_root / name
        task_dir.mkdir(parents=True, exist_ok=True)
        task_toml = task_dir / "task.toml"
        if not task_toml.exists():
            task_toml.write_text('schema_version = "1.0"\n', encoding="utf-8")
    config = JobConfig(
        job_name="job",
        jobs_dir=tmp_path / "harbor-jobs",
        n_attempts=n_attempts,
        agents=[
            AgentConfig(
                name=None if import_path is not None else "codex",
                import_path=import_path,
                model_name=model_name,
            )
        ],
        tasks=[TaskConfig(path=task_root / task_names[0])],
    )
    task_configs = [TaskConfig(path=task_root / name) for name in task_names]
    return Job(config, _task_configs=task_configs, _metrics={"adhoc": []})


def _job_result(job: Job, *, reverse: bool = False, raw_metadata: bool = False) -> JobResult:
    now = datetime.now(UTC)
    trials: list[TrialResult] = []
    for config in job._trial_configs:
        trials.append(
            TrialResult(
                task_name=f"public/{config.task.path.name}",
                trial_name=config.trial_name,
                trial_uri="secret://must-not-persist",
                task_id=config.task.get_task_id(),
                task_checksum="checksum",
                config=config,
                agent_info=AgentInfo(name="codex", version="1", model_info=None),
                agent_result=(
                    None
                    if not raw_metadata
                    else {
                        "n_output_tokens": 1,
                        "metadata": {
                            "headers": {"Authorization": "Bearer secret"},
                            "trajectory": "raw model/tool content",
                        },
                    }
                ),
                verifier_result=VerifierResult(rewards={"overall": 0.75}),
                started_at=now,
                finished_at=now,
            )
        )
    if reverse:
        trials.reverse()
    return JobResult(
        id=job.id,
        started_at=now,
        finished_at=now,
        n_total_trials=len(trials),
        stats=JobStats.from_trial_results(trials),
        trial_results=trials,
    )


def _mark_successful_agent_phase(
    plugin: SkillEvaluatorResultPlugin, result: JobResult, *, multistep: bool = False
) -> None:
    now = datetime.now(UTC)
    for trial in result.trial_results:
        plugin._agent_started_trials.add(trial.trial_name)
        if multistep:
            assert isinstance(trial.step_results, list) and trial.step_results
            for step in trial.step_results:
                step.agent_execution = TimingInfo(started_at=now, finished_at=now)
        else:
            trial.agent_execution = TimingInfo(started_at=now, finished_at=now)


def _plugin(run_root: Path, binding_ref: str, binding_digest: str) -> SkillEvaluatorResultPlugin:
    return SkillEvaluatorResultPlugin(
        run_root=str(run_root),
        binding_ref=binding_ref,
        binding_file_digest=binding_digest,
    )


def _refresh_task_set_digest(binding: dict[str, object]) -> None:
    binding["task_set_digest"] = canonical_digest(
        {
            "arm": binding["arm"],
            "root_ref": binding["task_root_ref"],
            "digest_algorithm": binding["digest_algorithm"],
            "skill_payload_digest": binding["skill_payload_digest"],
            "tasks": binding["arm_tasks"],
        }
    )


def test_plugin_writes_schedule_before_trials_and_results_in_schedule_order(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-b", "case-a"), expected_n_attempts=2)
    job = _job(tmp_path, task_names=("case-b", "case-a"), n_attempts=2)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)

    asyncio.run(plugin.on_job_start(job))
    schedule_path = tmp_path / "harbor-evidence/job/schedule.json"
    assert schedule_path.exists()
    schedule = json.loads(schedule_path.read_text())
    assert [(row["ordinal"], row["case_id"]) for row in schedule["trials"]] == [
        (1, "case-b"),
        (1, "case-a"),
        (2, "case-b"),
        (2, "case-a"),
    ]

    result = _job_result(job, reverse=True)
    _mark_successful_agent_phase(plugin, result)
    asyncio.run(plugin.on_job_end(result))
    results = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())
    Draft202012Validator(json.loads(SCHEDULE_SCHEMA.read_text())).validate(schedule)
    Draft202012Validator(json.loads(RESULTS_SCHEMA.read_text())).validate(results)
    assert [row["trial_name"] for row in results["trials"]] == [row["trial_name"] for row in schedule["trials"]]
    assert all((tmp_path / row["trial_ref"]).is_file() for row in results["trials"])


def test_stop_on_pass_binding_preserves_absolute_ordinal(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",), ordinal_base=2)
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    schedule = json.loads((tmp_path / "harbor-evidence/job/schedule.json").read_text())
    assert schedule["trials"][0]["ordinal"] == 2


def test_stop_on_pass_subset_binds_full_two_case_arm_without_treating_other_case_as_extra(
    tmp_path: Path,
) -> None:
    binding_ref, binding_digest = _binding(
        tmp_path,
        task_names=("case-1", "case-2"),
        scheduled_task_names=("case-2",),
        ordinal_base=2,
    )
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(_job(tmp_path, task_names=("case-2",))))
    schedule = json.loads((tmp_path / "harbor-evidence/job/schedule.json").read_text())
    assert [(trial["case_id"], trial["ordinal"]) for trial in schedule["trials"]] == [("case-2", 2)]


def test_plugin_rejects_version_shape_resume_and_unknown_case_before_start(monkeypatch, tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    job = _job(tmp_path)

    monkeypatch.setattr("skillevaluator.tier3.harbor.result_plugin.metadata.version", lambda _name: "0.13.3")
    with pytest.raises(RuntimeError, match="0.13.2"):
        asyncio.run(plugin.on_job_start(job))

    monkeypatch.setattr("skillevaluator.tier3.harbor.result_plugin.metadata.version", lambda _name: "0.13.2")
    job.is_resuming = True
    with pytest.raises(RuntimeError, match="resume"):
        asyncio.run(plugin.on_job_start(job))

    other_job = _job(tmp_path / "other", task_names=("other",))
    with pytest.raises(RuntimeError, match="unknown|binding"):
        asyncio.run(plugin.on_job_start(other_job))


def test_plugin_rejects_symlinked_evidence_parent_without_writing_outside(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "harbor-evidence").symlink_to(outside, target_is_directory=True)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    with pytest.raises(ContractError, match="symlink"):
        asyncio.run(plugin.on_job_start(_job(tmp_path)))
    assert not (outside / "job").exists()


def test_plugin_rejects_missing_duplicate_or_extra_result_identity(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1", "case-2"))
    job = _job(tmp_path, task_names=("case-1", "case-2"))
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    result.trial_results.pop()
    with pytest.raises(RuntimeError, match="missing|extra"):
        asyncio.run(plugin.on_job_end(result))

    plugin = _plugin(tmp_path / "duplicate", *_binding(tmp_path / "duplicate", task_names=("case-1", "case-2")))
    duplicate_job = _job(tmp_path / "duplicate", task_names=("case-1", "case-2"))
    asyncio.run(plugin.on_job_start(duplicate_job))
    duplicate_result = _job_result(duplicate_job)
    duplicate_result.trial_results[1].trial_name = duplicate_result.trial_results[0].trial_name
    with pytest.raises(RuntimeError, match="duplicate|missing|extra"):
        asyncio.run(plugin.on_job_end(duplicate_result))


def test_plugin_never_serializes_raw_context_messages_headers_or_credentials(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job, raw_metadata=True)
    result.trial_results[0].exception_info = ExceptionInfo(
        exception_type="RuntimeError",
        exception_message="Authorization: Bearer secret response body",
        exception_traceback="raw traceback with tool output",
        occurred_at=datetime.now(UTC),
    )
    asyncio.run(plugin.on_job_end(result))
    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((tmp_path / "harbor-evidence").rglob("*.json"))
    )
    for forbidden in ("Bearer secret", "response body", "raw traceback", "raw model/tool", "headers"):
        assert forbidden not in persisted
    assert '"exception_type":"RuntimeError"' in persisted
    projection = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())["trials"][0]
    assert projection["agent_failure"] is None


def test_plugin_rejects_synthetic_marker_from_current_uninstrumented_adapter(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    result.trial_results[0].verifier_result = None
    result.trial_results[0].agent_result = AgentContext(
        metadata={
            "skillevaluator.agent_failure_v1": {
                "stage": "agent_adapter_bootstrap",
                "reason_code": "adapter_model_protocol_negotiation_failed",
            },
            "error": "400 Bad Request with secret response body",
        }
    )
    with pytest.raises(RuntimeError, match="untrusted adapter"):
        asyncio.run(plugin.on_job_end(result))
    assert not (tmp_path / "harbor-evidence/job/results.json").exists()


@pytest.mark.parametrize("exception_type", ["AgentSetupTimeoutError", "NonZeroAgentExitCodeError"])
def test_official_plugin_lifecycle_projects_only_pinned_pre_instruction_setup_failures(
    tmp_path: Path, exception_type: str
) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)
    now = datetime.now(UTC)

    async def scenario() -> None:
        plugin = await attach_job_plugin(
            job,
            "skillevaluator.tier3.harbor.result_plugin:SkillEvaluatorResultPlugin",
            kwargs={
                "run_root": str(tmp_path),
                "binding_ref": binding_ref,
                "binding_file_digest": binding_digest,
            },
        )
        result = _job_result(job)
        trial = result.trial_results[0]
        trial.agent_result = None
        trial.verifier_result = None
        trial.step_results = None
        trial.environment_setup = TimingInfo(started_at=now, finished_at=now)
        trial.agent_setup = TimingInfo(started_at=now, finished_at=now)
        trial.agent_execution = None
        trial.verifier = None
        trial.exception_info = ExceptionInfo(
            exception_type=exception_type,
            exception_message="400 Bad Request must not be retained",
            exception_traceback="secret traceback",
            occurred_at=now,
        )
        await finalize_job_plugins([plugin], result)

    asyncio.run(scenario())
    projection = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())["trials"][0]
    assert projection["agent_failure"] == {
        "stage": "agent_adapter_bootstrap",
        "reason_code": "adapter_initialization_failed",
        "origin": "harbor_pre_instruction_phase",
    }
    assert projection["skill_logic_started"] is False
    assert projection["state"] == "failed"
    assert projection["rewards"] is None


@pytest.mark.parametrize(
    ("exception_type", "omit_environment_timing", "step_results", "verifier_reward"),
    [
        ("FileNotFoundError", False, None, True),
        ("RuntimeError", False, None, False),
        ("AgentSetupTimeoutError", True, None, False),
        ("AgentSetupTimeoutError", False, [], False),
    ],
)
def test_setup_failure_mapper_fails_closed_when_any_trusted_phase_predicate_is_absent(
    tmp_path: Path,
    exception_type: str,
    omit_environment_timing: bool,
    step_results: list[StepResult] | None,
    verifier_reward: bool,
) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    trial = result.trial_results[0]
    now = datetime.now(UTC)
    trial.agent_result = None
    trial.verifier_result = VerifierResult(rewards={"reward": 1.0}) if verifier_reward else None
    trial.step_results = step_results
    trial.environment_setup = None if omit_environment_timing else TimingInfo(started_at=now, finished_at=now)
    trial.agent_setup = TimingInfo(started_at=now, finished_at=now)
    trial.agent_execution = None
    trial.verifier = None
    trial.exception_info = ExceptionInfo(
        exception_type=exception_type,
        exception_message="400 Bad Request",
        exception_traceback="secret traceback",
        occurred_at=now,
    )
    asyncio.run(plugin.on_job_end(result))
    projection = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())["trials"][0]
    assert projection["agent_failure"] is None
    assert projection["state"] == "failed"
    assert projection["rewards"] is None


def test_post_agent_start_native_turn_failed_nonzero_remains_unclassified_and_cannot_score(
    tmp_path: Path,
) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    trial = result.trial_results[0]
    now = datetime.now(UTC)
    plugin._agent_started_trials.add(trial.trial_name)
    trial.agent_result = AgentContext()
    trial.agent_execution = TimingInfo(started_at=now, finished_at=now)
    trial.exception_info = ExceptionInfo(
        exception_type="NonZeroAgentExitCodeError",
        exception_message="API Error: 400 Bad Request",
        exception_traceback="secret response body",
        occurred_at=now,
    )
    native_log = job.job_dir / trial.trial_name / "agent/codex.txt"
    native_log.parent.mkdir(parents=True)
    native_log.write_text(
        json.dumps({"type": "thread.started", "thread_id": "thread-1"})
        + "\n"
        + json.dumps({"type": "turn.started"})
        + "\n"
        + json.dumps({"type": "turn.failed", "error": {"message": "API Error: 400 Bad Request"}})
        + "\n",
        encoding="utf-8",
    )
    asyncio.run(plugin.on_job_end(result))
    result_text = (tmp_path / "harbor-evidence/job/results.json").read_text()
    projection = json.loads(result_text)["trials"][0]
    assert projection["state"] == "failed"
    assert projection["rewards"] is None
    assert projection["agent_failure"] is None
    assert projection["skill_logic_started"] is True
    assert "Bad Request" not in result_text


def test_trusted_codex_adapter_marker_projects_post_start_presemantic_failure(
    tmp_path: Path,
) -> None:
    import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalCodex"
    binding_ref, binding_digest = _binding(
        tmp_path,
        task_names=("case-1",),
        import_path=import_path,
    )
    job = _job(tmp_path, import_path=import_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    trial = result.trial_results[0]
    now = datetime.now(UTC)
    plugin._agent_started_trials.add(trial.trial_name)
    trial.agent_result = AgentContext(
        metadata={
            "skillevaluator.agent_failure_v1": {
                "stage": "agent_adapter_bootstrap",
                "reason_code": "adapter_model_protocol_negotiation_failed",
            }
        }
    )
    trial.agent_execution = TimingInfo(started_at=now, finished_at=now)
    trial.verifier_result = None
    trial.exception_info = ExceptionInfo(
        exception_type="NonZeroAgentExitCodeError",
        exception_message="must not be retained",
        exception_traceback="must not be retained",
        occurred_at=now,
    )

    asyncio.run(plugin.on_job_end(result))

    result_text = (tmp_path / "harbor-evidence/job/results.json").read_text()
    projection = json.loads(result_text)["trials"][0]
    assert projection["agent_failure"] == {
        "stage": "agent_adapter_bootstrap",
        "reason_code": "adapter_model_protocol_negotiation_failed",
        "origin": "trusted_adapter_marker",
    }
    assert projection["skill_logic_started"] is False
    assert projection["state"] == "failed"
    assert projection["rewards"] is None
    assert "must not be retained" not in result_text


def test_plugin_rejects_retry_model_mismatch_and_arm_swap_before_trials(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    retry_job = _job(tmp_path)
    retry_job.config.retry.max_retries = 1
    with pytest.raises(RuntimeError, match="retries"):
        asyncio.run(_plugin(tmp_path, binding_ref, binding_digest).on_job_start(retry_job))

    model_job = _job(tmp_path, model_name="openai/other")
    with pytest.raises(RuntimeError, match="model"):
        asyncio.run(_plugin(tmp_path, binding_ref, binding_digest).on_job_start(model_job))

    baseline_job = _job(tmp_path, arm="baseline")
    with pytest.raises(RuntimeError, match="arm-specific trusted root"):
        asyncio.run(_plugin(tmp_path, binding_ref, binding_digest).on_job_start(baseline_job))


@pytest.mark.parametrize(
    ("schedule_ref", "extra_root"),
    [
        ("staged/with_skill", None),
        ("staged/with_skill/evidence.json", None),
        ("staged", None),
        ("STAGED/WITH_SKILL/evidence.json", None),
        ("staged/cafe\u0301/evidence.json", "staged/Caf\u00e9"),
    ],
)
def test_plugin_rejects_artifact_paths_overlapping_any_protected_task_root(
    tmp_path: Path, schedule_ref: str, extra_root: str | None
) -> None:
    binding_ref, _ = _binding(tmp_path, task_names=("case-1",))
    binding_path = tmp_path / binding_ref
    binding = json.loads(binding_path.read_text())
    binding_path.unlink()
    binding["schedule_ref"] = schedule_ref
    if extra_root is not None:
        binding["protected_task_roots"].append(extra_root)
    binding_digest = atomic_write_json(binding_path, binding, trusted_root=tmp_path)

    with pytest.raises(ValueError, match="overlap"):
        _plugin(tmp_path, binding_ref, binding_digest)


def test_plugin_rejects_task_roots_outside_fixed_staged_namespace(tmp_path: Path) -> None:
    binding_ref, _ = _binding(tmp_path, task_names=("case-1",))
    binding_path = tmp_path / binding_ref
    binding = json.loads(binding_path.read_text())
    binding_path.unlink()
    binding["task_root_ref"] = "harbor-evidence/tasks"
    binding["protected_task_roots"] = ["harbor-evidence/tasks"]
    _refresh_task_set_digest(binding)
    binding_digest = atomic_write_json(binding_path, binding, trusted_root=tmp_path)

    with pytest.raises(ValueError, match="fixed staged"):
        _plugin(tmp_path, binding_ref, binding_digest)


def test_plugin_rejects_ancestor_descendant_overlap_between_artifact_refs(
    tmp_path: Path,
) -> None:
    binding_ref, _ = _binding(tmp_path, task_names=("case-1",))
    binding_path = tmp_path / binding_ref
    binding = json.loads(binding_path.read_text())
    binding_path.unlink()
    binding["schedule_ref"] = "harbor-evidence/job"
    binding["results_ref"] = "HARBOR-EVIDENCE/JOB/results.json"
    binding_digest = atomic_write_json(binding_path, binding, trusted_root=tmp_path)

    with pytest.raises(ValueError, match="paths overlap"):
        _plugin(tmp_path, binding_ref, binding_digest)


def test_local_import_path_with_name_none_uses_pinned_binding(tmp_path: Path) -> None:
    import_path = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalCodex"
    binding_ref, binding_digest = _binding(
        tmp_path,
        task_names=("case-1",),
        import_path=import_path,
    )
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(_job(tmp_path, import_path=import_path)))
    assert (tmp_path / "harbor-evidence/job/schedule.json").exists()


def test_official_lifecycle_leaves_results_missing_when_job_run_raises(monkeypatch, tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)

    async def scenario() -> None:
        await attach_job_plugin(
            job,
            "skillevaluator.tier3.harbor.result_plugin:SkillEvaluatorResultPlugin",
            kwargs={
                "run_root": str(tmp_path),
                "binding_ref": binding_ref,
                "binding_file_digest": binding_digest,
            },
        )

        async def fail_run():
            raise RuntimeError("job failed before finalize")

        monkeypatch.setattr(job, "run", fail_run)
        with pytest.raises(RuntimeError, match="before finalize"):
            await job.run()

    asyncio.run(scenario())
    assert (tmp_path / "harbor-evidence/job/schedule.json").exists()
    assert not (tmp_path / "harbor-evidence/job/results.json").exists()


def test_official_finalize_swallows_end_hook_failure_and_leaves_results_missing(monkeypatch, tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)

    async def scenario() -> None:
        plugin = await attach_job_plugin(
            job,
            "skillevaluator.tier3.harbor.result_plugin:SkillEvaluatorResultPlugin",
            kwargs={
                "run_root": str(tmp_path),
                "binding_ref": binding_ref,
                "binding_file_digest": binding_digest,
            },
        )

        async def fail_end(_result):
            raise RuntimeError("end hook failed")

        monkeypatch.setattr(plugin, "on_job_end", fail_end)
        await finalize_job_plugins([plugin], _job_result(job))

    asyncio.run(scenario())
    assert (tmp_path / "harbor-evidence/job/schedule.json").exists()
    assert not (tmp_path / "harbor-evidence/job/results.json").exists()


def test_plugin_multistep_mean_preserves_present_null_semantics(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    binding_path = tmp_path / binding_ref
    binding = json.loads(binding_path.read_text())
    binding_path.unlink()
    binding["cases"][0]["reward_strategy"] = "multi_step_mean"
    binding["arm_tasks"][0]["reward_strategy"] = "multi_step_mean"
    _refresh_task_set_digest(binding)
    binding_digest = atomic_write_json(binding_path, binding, trusted_root=tmp_path)
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    result.trial_results[0].step_results = [
        StepResult(step_name="one", verifier_result=VerifierResult(rewards={"overall": 1.0})),
        StepResult(step_name="two", verifier_result=VerifierResult(rewards=None)),
        StepResult(step_name="three", verifier_result=None),
    ]
    result.trial_results[0].verifier_result = VerifierResult(rewards={"overall": 0.5})
    _mark_successful_agent_phase(plugin, result, multistep=True)
    asyncio.run(plugin.on_job_end(result))
    projection = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())["trials"][0]
    assert [step["verifier_result_present"] for step in projection["steps"]] == [True, True, False]
    assert projection["rewards"] == {"overall": 0.5}


def test_plugin_mixed_multistep_mean_and_final_preserve_harbor_parity(tmp_path: Path) -> None:
    binding_ref, _binding_digest = _binding(tmp_path, task_names=("case-mean", "case-final"))
    binding_path = tmp_path / binding_ref
    binding = json.loads(binding_path.read_text())
    binding_path.unlink()
    binding["cases"][0]["reward_strategy"] = "multi_step_mean"
    binding["cases"][1]["reward_strategy"] = "multi_step_final"
    binding["arm_tasks"][0]["reward_strategy"] = "multi_step_mean"
    binding["arm_tasks"][1]["reward_strategy"] = "multi_step_final"
    _refresh_task_set_digest(binding)
    binding_digest = atomic_write_json(binding_path, binding, trusted_root=tmp_path)
    job = _job(tmp_path, task_names=("case-mean", "case-final"))
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    by_task = {trial.config.task.path.name: trial for trial in result.trial_results}
    by_task["case-mean"].step_results = [
        StepResult(step_name="one", verifier_result=VerifierResult(rewards={"overall": 1.0})),
        StepResult(step_name="two", verifier_result=VerifierResult(rewards=None)),
    ]
    by_task["case-mean"].verifier_result = VerifierResult(rewards={"overall": 0.5})
    by_task["case-final"].step_results = [
        StepResult(step_name="one", verifier_result=VerifierResult(rewards={"overall": 0.2})),
        StepResult(step_name="two", verifier_result=VerifierResult(rewards={"overall": 0.8})),
    ]
    by_task["case-final"].verifier_result = VerifierResult(rewards={"overall": 0.8})
    _mark_successful_agent_phase(plugin, result, multistep=True)
    asyncio.run(plugin.on_job_end(result))
    trials = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())["trials"]
    assert [(trial["reward_strategy"], trial["rewards"]["overall"]) for trial in trials] == [
        ("multi_step_mean", 0.5),
        ("multi_step_final", 0.8),
    ]


@pytest.mark.parametrize("schema_path", [SCHEDULE_SCHEMA, RESULTS_SCHEMA])
def test_plugin_schemas_are_closed_draft_2020_12(schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False


def test_minimal_result_bytes_are_canonical(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    _mark_successful_agent_phase(plugin, result)
    asyncio.run(plugin.on_job_end(result))
    result = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())
    trial_path = tmp_path / result["trials"][0]["trial_ref"]
    assert trial_path.read_bytes() == canonical_json_bytes(json.loads(trial_path.read_text()), trailing_newline=True)


def test_plugin_refuses_score_without_agent_start_and_completed_execution(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))

    asyncio.run(plugin.on_job_end(_job_result(job)))

    projection = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())["trials"][0]
    assert projection["state"] == "failed"
    assert projection["rewards"] is None
    assert projection["skill_logic_started"] is False


def test_plugin_accepts_real_default_projection_and_rejects_bad_overall(
    tmp_path: Path,
) -> None:
    contract = build_reward_contract("default")
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",), reward_contract=contract)
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    rewards = {
        metric: value
        for metric, value in zip(
            (
                "security",
                "skill_execution",
                "skill_efficiency",
                "accuracy",
                "goal_accuracy",
                "behavior_check",
            ),
            (1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
            strict=True,
        )
    }
    rewards["overall"] = 0.5
    result.trial_results[0].verifier_result = VerifierResult(rewards=rewards)
    _mark_successful_agent_phase(plugin, result)
    asyncio.run(plugin.on_job_end(result))
    projection = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())["trials"][0]
    assert projection["rewards"] == rewards

    bad_root = tmp_path / "bad-overall"
    binding_ref, binding_digest = _binding(bad_root, task_names=("case-1",), reward_contract=contract)
    job = _job(bad_root)
    plugin = _plugin(bad_root, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    result.trial_results[0].verifier_result = VerifierResult(rewards={**rewards, "overall": 0.51})
    _mark_successful_agent_phase(plugin, result)
    with pytest.raises(ContractError, match="canonical SkillEvaluator mean"):
        asyncio.run(plugin.on_job_end(result))
    assert not (bad_root / "harbor-evidence/job/trials/000001.json").exists()


@pytest.mark.parametrize(
    "rewards",
    [
        {"Authorization: Bearer sk-secret": 1.0},
        {"overall": 1.000001},
        {"overall": -0.000001},
    ],
)
def test_plugin_rejects_unbound_secret_shaped_or_out_of_range_rewards_before_persistence(
    tmp_path: Path, rewards: dict[str, float]
) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    result.trial_results[0].verifier_result = VerifierResult(rewards=rewards)
    _mark_successful_agent_phase(plugin, result)

    with pytest.raises(ContractError, match="reward|metric|range|keys"):
        asyncio.run(plugin.on_job_end(result))
    assert not (tmp_path / "harbor-evidence/job/trials/000001.json").exists()
    assert not (tmp_path / "harbor-evidence/job/results.json").exists()


def test_plugin_enforces_reward_and_step_projection_bounds(tmp_path: Path) -> None:
    binding_ref, binding_digest = _binding(tmp_path, task_names=("case-1",))
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    result.trial_results[0].verifier_result = VerifierResult(rewards={f"metric_{index}": 0.5 for index in range(257)})
    _mark_successful_agent_phase(plugin, result)
    with pytest.raises(RuntimeError, match="256-property"):
        asyncio.run(plugin.on_job_end(result))

    other_root = tmp_path / "steps"
    binding_ref, binding_digest = _binding(other_root, task_names=("case-1",))
    job = _job(other_root)
    plugin = _plugin(other_root, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    result.trial_results[0].step_results = [StepResult(step_name=f"step-{index}") for index in range(10_001)]
    with pytest.raises(RuntimeError, match="10000-item"):
        asyncio.run(plugin.on_job_end(result))


def test_multistep_exception_strips_partial_and_top_level_scores(tmp_path: Path) -> None:
    binding_ref, _ = _binding(tmp_path, task_names=("case-1",))
    binding_path = tmp_path / binding_ref
    binding = json.loads(binding_path.read_text())
    binding_path.unlink()
    binding["cases"][0]["reward_strategy"] = "multi_step_mean"
    binding["arm_tasks"][0]["reward_strategy"] = "multi_step_mean"
    _refresh_task_set_digest(binding)
    binding_digest = atomic_write_json(binding_path, binding, trusted_root=tmp_path)
    job = _job(tmp_path)
    plugin = _plugin(tmp_path, binding_ref, binding_digest)
    asyncio.run(plugin.on_job_start(job))
    result = _job_result(job)
    now = datetime.now(UTC)
    result.trial_results[0].step_results = [
        StepResult(step_name="one", verifier_result=VerifierResult(rewards={"overall": 1.0})),
        StepResult(
            step_name="two",
            verifier_result=VerifierResult(rewards={"overall": 0.0}),
            exception_info=ExceptionInfo(
                exception_type="RuntimeError",
                exception_message="must not persist",
                exception_traceback="must not persist",
                occurred_at=now,
            ),
        ),
    ]
    result.trial_results[0].verifier_result = VerifierResult(rewards={"overall": 0.5})
    _mark_successful_agent_phase(plugin, result, multistep=True)
    asyncio.run(plugin.on_job_end(result))
    projection = json.loads((tmp_path / "harbor-evidence/job/results.json").read_text())["trials"][0]
    assert projection["state"] == "failed"
    assert projection["rewards"] is None
    assert all(step["rewards"] is None for step in projection["steps"])
