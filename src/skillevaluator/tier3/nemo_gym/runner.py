# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Tier 3 cases on NeMo Gym instead of Harbor.

SkillEvaluator still turns ``evals.json`` into task folders with its own verifier
(``tests/test.sh``). NeMo Gym then owns execution: ``gym eval run <tasks>`` starts its
Harbor-format environment server, model server and agent harness, runs each case in a
Gym sandbox and grades it with that verifier. The rollouts are normalized back into the
trial layout the shared collector reads, so scores, lift and reports are unchanged.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.progress import NullProgressReporter, ProgressEvent, ProgressReporter
from skillevaluator.tier3.nemo_gym.backend import (
    NEMO_GYM_AGENTS,
    NEMO_GYM_BACKEND,
    NEMO_GYM_BIN_ENV,
    nemo_gym_bin,
    nemo_gym_capability_errors,
    nemo_gym_version,
)
from skillevaluator.tier3.nemo_gym.results import read_rollout_failures, read_rollouts, write_job_from_rollouts

MODEL_API_KEY_ENV = "SKILL_EVAL_NEMO_GYM_MODEL_API_KEY"
MODEL_URL_ENV = "SKILL_EVAL_NEMO_GYM_MODEL_URL"
SANDBOX_ENV = "SKILL_EVAL_NEMO_GYM_SANDBOX"
RUN_TIMEOUT_ENV = "SKILL_EVAL_NEMO_GYM_TIMEOUT_S"
HOST_ENV = "SKILL_EVAL_NEMO_GYM_HOST"


_SKILLS_DIR_LINE = re.compile(r"^skills_dir\s*=.*\n?", re.MULTILINE)


def stage_for_gym(arm_dir: Path) -> None:
    """Drop Harbor's ``skills_dir`` task hook, which Gym's CLI agents reject.

    The task image already carries the skills at each agent's native location
    (``/root/.agents/skills``, ``/root/.claude/skills``, ...), so the with-skill arm
    is unchanged and the baseline image still has none.
    """
    for task_toml in arm_dir.glob("*/task.toml"):
        text = task_toml.read_text(encoding="utf-8")
        stripped = _SKILLS_DIR_LINE.sub("", text)
        if stripped != text:
            task_toml.write_text(stripped, encoding="utf-8")


def _task_case_ids(arm_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for task_dir in sorted(p for p in arm_dir.iterdir() if (p / "task.toml").is_file()):
        metadata = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8")).get("metadata") or {}
        entry_id = metadata.get("entry_id")
        mapping[task_dir.name] = entry_id if isinstance(entry_id, str) and entry_id else task_dir.name
    return mapping


def _model_endpoint(model: str | None) -> tuple[str, str, str]:
    """Return ``(model, base_url, api_key)`` for Gym's OpenAI-compatible policy model server."""
    from skillevaluator.provider_config import resolve_llm_provider

    url = os.environ.get(MODEL_URL_ENV, "").strip()
    key = os.environ.get(MODEL_API_KEY_ENV, "").strip()
    if url and not key:
        # Never send the evaluator's provider key to a policy server it was not issued for.
        raise ValueError(f"{MODEL_URL_ENV} is set, so also set {MODEL_API_KEY_ENV} (use EMPTY for a keyless server)")
    resolved_model = model
    if not url or not key or not resolved_model:
        provider = resolve_llm_provider()
        url = url or (provider.base_url or "")
        key = key or (provider.api_key or "")
        resolved_model = resolved_model or provider.model
    if not url or not key or not resolved_model:
        raise ValueError(
            f"--backend nemo-gym needs a model, an OpenAI-compatible base URL and an API key "
            f"(--model, {MODEL_URL_ENV}, {MODEL_API_KEY_ENV}, or the SkillEvaluator LLM provider settings)"
        )
    return resolved_model, url, key


def verifier_judge_env(skill_path: Path, grading_mode: str, env_mode: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(task verifier env templates, Gym process env)`` carrying the judge settings.

    Mirrors the Harbor runner's staging. Gym has no job-level env layer, so the job-level
    judge overrides are folded into the task templates, which the Gym environment server
    resolves from its own environment; the values never enter a task folder or row.
    """
    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
    from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
    from skillevaluator.tier3.harbor.runner import (
        _VERIFIER_JUDGE_CONTROL_ENV_VARS,
        _VERIFIER_JUDGE_MODEL_ENV_VARS,
        _job_judge_subprocess_env,
        _job_judge_verifier_env,
        _provider_environment,
        _resolve_runtime_env,
    )

    try:
        provider = resolve_llm_provider()
        config, _config_path = load_evals_config(skill_path)
    except (ProviderConfigurationError, EvalsConfigError) as exc:
        raise ValueError(str(exc)) from exc
    runtime_env, runtime_errors = _resolve_runtime_env(
        (config.get("harbor") or {}).get("runtime_env"), env_mode=env_mode
    )
    if runtime_errors:
        raise ValueError("; ".join(runtime_errors))
    provider_env = _provider_environment(provider)
    values = {**runtime_env, **provider_env}
    templates = {
        name: f"${{{name}}}"
        for name in values
        if name not in _VERIFIER_JUDGE_MODEL_ENV_VARS
        and not (grading_mode == "custom_only" and name in _VERIFIER_JUDGE_CONTROL_ENV_VARS)
    }
    templates.update(_job_judge_verifier_env(provider_env, grading_mode))
    return templates, {**values, **_job_judge_subprocess_env(provider_env, grading_mode)}


def gym_eval_command(
    gym: str,
    arm_dir: Path,
    *,
    agent: str,
    model: str,
    model_url: str,
    rollouts_path: Path,
    n_attempts: int,
    n_concurrent: int | None,
    sandbox: str,
) -> list[str]:
    """Build the ``gym eval run`` command for one agent and arm; the API key is read from the environment."""
    command = [
        gym,
        "eval",
        "run",
        str(arm_dir),
        "--agent",
        NEMO_GYM_AGENTS[agent],
        "--sandbox",
        sandbox,
        "--model-type",
        "openai_model",
        "--model",
        model,
        "--model-url",
        model_url,
        "--output",
        str(rollouts_path),
        f"+policy_api_key=${{oc.env:{MODEL_API_KEY_ENV}}}",
        f"+num_repeats={n_attempts}",
        "+resume_from_cache=false",
    ]
    if n_concurrent:
        command += ["--concurrency", str(n_concurrent)]
    host = os.environ.get(HOST_ENV, "").strip()
    if host:
        command.append(f"+default_host={host}")
    return command


# Package-index settings Gym's own ``uv sync`` of its servers reads.
_GYM_HOST_ENV_VARS = frozenset({"UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL", "UV_INDEX", "UV_INDEX_URL"})
_GYM_SHUTDOWN_GRACE_SECONDS = 30.0


def _run_gym(command: list[str], *, cwd: Path, env: dict[str, str], log: Any, timeout_s: float) -> int | None:
    """Run ``gym eval run`` in its own process group; return its exit code, or None on timeout.

    On timeout or Ctrl-C the group gets SIGTERM and a grace period, so Gym's own
    shutdown stops its servers and removes its sandboxes, before SIGKILL.
    """
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def _stop() -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=_GYM_SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _stop()
        return None
    except BaseException:
        _stop()
        raise


def gym_launch_environment(credentials: dict[str, str]) -> dict[str, str]:
    """Return Gym's process environment: the host variables a Harbor backend may receive, plus its credentials."""
    from skillevaluator.tier3.harbor.runner import _HARBOR_BASE_ENV_VARS, _HARBOR_ENV_MODE_VARS

    allowed = _HARBOR_BASE_ENV_VARS | _HARBOR_ENV_MODE_VARS["docker"] | _GYM_HOST_ENV_VARS
    return {**{name: value for name, value in os.environ.items() if name in allowed}, **credentials}


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def run_nemo_gym_eval(
    *,
    skill_path: Path,
    agents: list[str],
    skip_baseline: bool,
    n_attempts: int | None,
    pass_threshold: float | None,
    n_concurrent: int | None,
    model: str | None,
    agent_models: dict[str, str] | None,
    skill_workspace_mode: str | None,
    include_skills: list[Path] | None,
    grading_mode: str | None,
    output_dir: Path,
    env_mode: str,
    timeout_multiplier: float | None,
    evaluated_source: dict[str, str] | None,
    override_cpus: int | None,
    override_memory_mb: int | None,
    repo_context_exclude_paths: tuple[Path, ...] = (),
    progress_reporter: ProgressReporter | None = None,
    skip_with_skill: bool = False,
    max_cases: int | None = None,
    stop_on_pass: bool | None = None,
    agent_runtime_preflight: bool | None = None,
    custom_dockerfile_mode: str | None = None,
) -> dict[str, Any]:
    """Run the selected arms on NeMo Gym and return the shared results mapping."""
    from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
    from skillevaluator.publication_identity import publication_target_from_path
    from skillevaluator.source_identity import normalized_evaluated_source
    from skillevaluator.tier3.case_ids import validate_case_ids
    from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
    from skillevaluator.tier3.harbor.collector import _score_definition_for_policy, collect_harbor_results
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS, score_policy_for_metrics
    from skillevaluator.tier3.harbor.report_data import dataset_snapshot_manifest
    from skillevaluator.tier3.harbor.runner import (
        _limit_staged_cases,
        _merge_launch_execution_errors,
        _native_entry_id,
        _persist_dataset_truth,
        _reserve_run_dir,
        _resolve_runtime_env,
        _validated_pass_threshold,
        _workspace_skills,
        _write_final_result,
    )
    from skillevaluator.tier3.results_location import publish_latest_results

    started_at = time.monotonic()
    publication_target = publication_target_from_path(skill_path)
    reporter = progress_reporter or NullProgressReporter()

    def _config_failure(detail: str) -> dict[str, Any]:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail=detail))
        return {"error": [detail]}

    gym = nemo_gym_bin()
    if gym is None:
        raise ValueError(
            f"--backend nemo-gym needs the NeMo Gym CLI: install nemo-gym (Python >=3.13.14) in its own "
            f"environment and set {NEMO_GYM_BIN_ENV} to its `gym` executable"
        )
    if skip_baseline and skip_with_skill:
        return _config_failure("skip_baseline and skip_with_skill are mutually exclusive (no arm would run)")
    try:
        provider = resolve_llm_provider()
        config, config_path = load_evals_config(skill_path)
    except (ProviderConfigurationError, EvalsConfigError) as exc:
        return _config_failure(str(exc))

    # Unset options resolve from evals/config.yml exactly as on Harbor; anything Gym cannot honor is rejected.
    harbor_config = config.get("harbor") or {}
    grading_config = config.get("grading") or {}
    workspace_config = config.get("skill_workspace") or {}
    attempts = n_attempts if n_attempts is not None else harbor_config.get("n_attempts", 1)
    concurrency = n_concurrent if n_concurrent is not None else harbor_config.get("n_concurrent", 4)
    max_cases = max_cases if max_cases is not None else harbor_config.get("max_cases")
    effective_grading = grading_mode or grading_config.get("mode", "default")
    workspace_mode = skill_workspace_mode or workspace_config.get("mode", "isolated")
    effective_timeout = (
        timeout_multiplier if timeout_multiplier is not None else harbor_config.get("timeout_multiplier", 1.0)
    )
    errors = nemo_gym_capability_errors(
        agents=agents,
        env_mode=env_mode,
        grading_mode=effective_grading,
        stop_on_pass=bool(stop_on_pass if stop_on_pass is not None else harbor_config.get("stop_on_pass", False)),
        custom_dockerfile_mode=custom_dockerfile_mode or harbor_config.get("custom_dockerfile_mode"),
        agent_runtime_preflight=bool(
            agent_runtime_preflight
            if agent_runtime_preflight is not None
            else harbor_config.get("agent_runtime_preflight", False)
        ),
    )
    if effective_timeout not in (1, 1.0):
        errors.append("--timeout-multiplier is not supported with --backend nemo-gym yet")
    if harbor_config.get("base_image_mode", "disabled") != "disabled":
        errors.append("harbor.base_image_mode reuse/rebuild applies to the Harbor backend only")
    if not _is_positive_int(attempts):
        errors.append("n_attempts must be >= 1")
    if not _is_positive_int(concurrency):
        errors.append("n_concurrent must be >= 1")
    if max_cases is not None and not _is_positive_int(max_cases):
        errors.append("max_cases must be >= 1")
    if harbor_config.get("task_source", "auto") == "native_harbor":
        errors.append("harbor.task_source native_harbor is not supported with --backend nemo-gym yet")
    errors += [
        f"harbor.{key} is not supported with --backend nemo-gym yet"
        for key in ("agent_workdir", "pre_agent_setup", "setup_commands", "passthrough_env")
        if harbor_config.get(key)
    ]
    resource_config = dict(harbor_config.get("resources") or {})
    if "storage_mb" in resource_config:
        errors.append("harbor.resources.storage_mb is not supported with --backend nemo-gym")
    include_values = [*workspace_config.get("include", []), *(include_skills or [])]
    if include_values and workspace_mode != "group":
        errors.append("include_skills requires skill_workspace.mode=group")
    if workspace_config.get("decoy_manifest"):
        errors.append("skill_workspace.decoy_manifest is not supported with --backend nemo-gym yet")
    raw_timeout = os.environ.get(RUN_TIMEOUT_ENV, "").strip()
    try:
        timeout_s = float(raw_timeout) if raw_timeout else 21600.0
    except ValueError:
        timeout_s = math.nan
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        errors.append(f"{RUN_TIMEOUT_ENV} must be a finite number of seconds greater than 0")
    if errors:
        return _config_failure("; ".join(errors))
    try:
        threshold = _validated_pass_threshold(
            pass_threshold if pass_threshold is not None else harbor_config.get("pass_threshold", 0.5)
        )
        workspace_skills = _workspace_skills(skill_path, include_values if workspace_mode == "group" else [])
    except ValueError as exc:
        return _config_failure(str(exc))

    gym_version = nemo_gym_version(gym)
    sandbox = os.environ.get(SANDBOX_ENV, "docker").strip() or "docker"
    execution_backend = {"name": NEMO_GYM_BACKEND, "version": gym_version, "sandbox": sandbox}
    resolved_models: dict[str, dict[str, str]] = {}
    endpoints: dict[str, tuple[str, str, str]] = {}
    config_agents = harbor_config.get("agents") or {}
    for agent in agents:
        cli_model = (agent_models or {}).get(agent) or model
        config_model = (config_agents.get(agent) or {}).get("model")
        endpoints[agent] = _model_endpoint(cli_model or config_model)
        source = "CLI" if cli_model else "evals/config.yml" if config_model else "public provider default"
        resolved_models[agent] = {"agent": agent, "model": endpoints[agent][0], "source": source}

    run_dir = _reserve_run_dir(output_dir, datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    tasks_root = run_dir / "_harbor-tasks"
    jobs_dir = run_dir / "_harbor-jobs"
    gym_root = run_dir / "_nemo-gym"
    gym_root.mkdir(parents=True, exist_ok=True)
    arms = [
        (arm, with_skill)
        for arm, with_skill, skipped in (("with", True, skip_with_skill), ("without", False, skip_baseline))
        if not skipped
    ]
    task_resources = {
        **resource_config,
        **{
            key: value
            for key, value in (("cpus", override_cpus), ("memory_mb", override_memory_mb))
            if value is not None
        },
    }
    # Judge credentials reach only the verifier: tasks carry ${NAME} templates that the
    # Gym environment server resolves from its own process environment.
    try:
        judge_verifier_env, judge_process_env = verifier_judge_env(skill_path, effective_grading, env_mode)
    except ValueError as exc:
        return _config_failure(str(exc))
    reporter.set_secret_values(set(judge_process_env.values()))
    # harbor.runtime_env reaches the agent as ${NAME} templates, resolved from Gym's environment.
    runtime_values, _runtime_errors = _resolve_runtime_env(harbor_config.get("runtime_env"), env_mode=env_mode)
    agent_runtime_env = {name: f"${{{name}}}" for name in runtime_values}
    reporter.emit(ProgressEvent(stage="configuration", state="ready", detail="evaluation config validated"))
    launch_errors: list[str] = []
    case_ids: list[str] = []
    case_selection: dict[str, Any] | None = None
    for agent in agents:
        reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="running", detail="NeMo Gym"))
        agent_model, model_url, api_key = endpoints[agent]
        for arm, with_skill in arms:
            arm_dir = tasks_root / agent / arm
            task_paths = generate_harbor_tasks(
                skill_path,
                arm_dir,
                with_skill=with_skill,
                workspace_skill_paths=workspace_skills,
                workspace_mode=workspace_mode,
                grading_mode=effective_grading,
                verifier_env=judge_verifier_env or None,
                repo_context_exclude_paths=repo_context_exclude_paths,
                task_resources=task_resources or None,
                runtime_env=agent_runtime_env or None,
                arm_suffix=f"-{arm}-skill" if len(arms) == 2 else "",
            )
            if max_cases is not None:
                selectors = validate_case_ids(task.name for task in task_paths)
                selector_case_ids = validate_case_ids(_native_entry_id(task) for task in task_paths)
                case_selection = _limit_staged_cases(
                    selectors, dict(zip(selectors, selector_case_ids, strict=True)), task_paths, max_cases=max_cases
                )
            stage_for_gym(arm_dir)
            task_case_ids = _task_case_ids(arm_dir)
            case_ids = case_ids or sorted(set(task_case_ids.values()))
            rollouts_path = gym_root / f"{agent}-{arm}" / "rollouts.jsonl"
            rollouts_path.parent.mkdir(parents=True, exist_ok=True)
            command = gym_eval_command(
                gym,
                arm_dir,
                agent=agent,
                model=agent_model,
                model_url=model_url,
                rollouts_path=rollouts_path,
                n_attempts=attempts,
                n_concurrent=concurrency,
                sandbox=sandbox,
            )
            log_path = rollouts_path.with_name("gym.log")
            reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="running", detail=f"NeMo Gym {arm}-skill arm"))
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    returncode = _run_gym(
                        command,
                        cwd=rollouts_path.parent,
                        env=gym_launch_environment({**judge_process_env, MODEL_API_KEY_ENV: api_key}),
                        log=log,
                        timeout_s=timeout_s,
                    )
                except OSError as exc:
                    launch_errors.append(f"{agent} {arm}-skill: gym eval run could not start: {exc}")
                else:
                    if returncode is None:
                        launch_errors.append(f"{agent} {arm}-skill: gym eval run timed out after {timeout_s:.0f}s")
                    elif returncode != 0:
                        launch_errors.append(f"{agent} {arm}-skill: gym eval run exited {returncode}; see {log_path}")
            launch_errors += write_job_from_rollouts(
                read_rollouts(rollouts_path, launch_errors),
                jobs_dir / f"{skill_path.name}-{agent}-{arm}",
                eval_name=f"{agent}__{agent_model}",
                task_case_ids=task_case_ids,
                n_attempts=attempts,
                failures=read_rollout_failures(rollouts_path, launch_errors),
            )
        reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="complete"))

    reporter.emit(ProgressEvent(stage="collection", state="running"))
    results = collect_harbor_results(
        skill_name=skill_path.name,
        agents=agents,
        output_dir=run_dir,
        jobs_dir=jobs_dir,
        skip_baseline=skip_baseline,
        skip_with_skill=skip_with_skill,
        n_attempts=attempts,
        pass_threshold=threshold,
        expected_cases=len(case_ids),
        expected_case_ids=case_ids,
        expected_trials=len(case_ids) * attempts,
        env_mode=env_mode,
        agent_models=resolved_models,
        launch_errors=launch_errors,
        grading_mode=effective_grading,
    )
    if launch_errors:
        _merge_launch_execution_errors(results, launch_errors)
    reporter.emit(ProgressEvent(stage="collection", state="complete", detail="NeMo Gym results collected"))

    # The same completion contract as a Harbor run, so result discovery (`view`, compare) finds Gym runs.
    try:
        config_file = str(config_path.relative_to(skill_path)) if config_path else "none"
    except ValueError:
        config_file = config_path.name if config_path else "none"
    run_config: dict[str, Any] = {
        "config_file": config_file,
        "harbor": {
            "environment": {"value": env_mode, "source": "CLI"},
            "environment_kwargs": {"keys": [], "sources": {}},
            "n_attempts": attempts,
            "stop_on_pass": False,
            "n_concurrent": concurrency,
            "timeout_multiplier": 1.0,
            "base_image_mode": "disabled",
            "jobs_retained": True,
        },
        "provider": {"name": provider.provider, "model": provider.model},
        "task_source": "evals_json",
        "grading": {"mode": effective_grading},
        "agents": resolved_models,
        "evaluated_source": normalized_evaluated_source(evaluated_source),
        "execution_backend": execution_backend,
    }
    if max_cases is not None:
        run_config["harbor"]["max_cases"] = max_cases
    if case_selection is not None:
        run_config["case_selection"] = case_selection
    selected_metrics = tuple(results.get("metrics", DEFAULT_METRICS))
    attempt_policy = dict(results.get("attempt_policy") or {})
    score_policy = results.get("score_policy") or attempt_policy.get("score_policy")
    if not isinstance(score_policy, str) or not score_policy.strip():
        score_policy = score_policy_for_metrics(selected_metrics)
    attempt_policy.update(
        {
            "max_attempts": attempts,
            "pass_threshold": float(threshold),
            "stop_on_pass": False,
            "score_policy": score_policy,
            "score_definition": attempt_policy.get("score_definition")
            or _score_definition_for_policy(selected_metrics, score_policy),
        }
    )
    # The same immutable dataset and publication identity a Harbor run records.
    try:
        dataset_truth = _persist_dataset_truth(run_dir, fallback_task_ids=case_ids)
    except Exception as exc:
        dataset_truth = None
        results.setdefault("warnings", []).append(f"Dataset snapshot was not recorded: {type(exc).__name__}")
    if dataset_truth is not None:
        results.update(
            {
                "evaluated_at": datetime.now(UTC).isoformat(),
                "evaluator_version": dataset_truth["evaluator_version"],
                "dataset_snapshot": dataset_snapshot_manifest(dataset_truth),
                "dataset_snapshot_path": str(run_dir / "dataset_snapshot.json"),
                "dataset_summary": dataset_truth["dataset_summary"],
                "dataset_digest": dataset_truth["dataset_digest"],
                "dataset_digest_algorithm": dataset_truth["dataset_digest_algorithm"],
            }
        )
    run_end_target = publication_target_from_path(skill_path)
    if run_end_target == publication_target:
        results["publication_target"] = dict(publication_target) if publication_target is not None else None
    else:
        results["publication_target"] = None
        results["publication_target_conflict"] = {
            "run_start": dict(publication_target) if publication_target is not None else None,
            "run_end": dict(run_end_target) if run_end_target is not None else None,
        }
    result_path = run_dir / "result.json"
    results.update(
        {
            "skill_name": skill_path.name,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "result_path": str(result_path),
            "execution_backend": execution_backend,
            "nemo_gym_dir": str(gym_root),
            "attempt_policy": attempt_policy,
            "run_config": run_config,
        }
    )
    if evaluated_source:
        results["evaluated_source"] = dict(evaluated_source)
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    try:
        results["report_path"] = str(
            render_agent_eval_html_report(skill_path, run_dir, env_mode=env_mode, engine_result=results)
        )
        results["report_status"] = "complete"
    except Exception as exc:
        results.setdefault("warnings", []).append(f"HTML report was not generated: {type(exc).__name__}")
        results["report_status"] = "degraded"
    results["duration_seconds"] = round(time.monotonic() - started_at, 3)
    final = _write_final_result(result_path, results)
    publish_latest_results(output_dir, run_dir.name)
    return final
