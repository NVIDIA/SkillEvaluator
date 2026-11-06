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

import json
import os
import re
import subprocess
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
) -> dict[str, Any]:
    """Run the with-skill and baseline arms on NeMo Gym and return the shared results mapping."""
    from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
    from skillevaluator.tier3.harbor.collector import collect_harbor_results
    from skillevaluator.tier3.harbor.runner import _reserve_run_dir

    reporter = progress_reporter or NullProgressReporter()
    gym = nemo_gym_bin()
    if gym is None:
        raise ValueError(
            f"--backend nemo-gym needs the NeMo Gym CLI: install nemo-gym (Python >=3.13.14) in its own "
            f"environment and set {NEMO_GYM_BIN_ENV} to its `gym` executable"
        )
    if timeout_multiplier not in (None, 1, 1.0):
        raise ValueError("--timeout-multiplier is not supported with --backend nemo-gym yet")
    attempts = n_attempts or 1
    threshold = 0.5 if pass_threshold is None else float(pass_threshold)
    gym_version = nemo_gym_version(gym)
    sandbox = os.environ.get(SANDBOX_ENV, "docker").strip() or "docker"
    timeout_s = float(os.environ.get(RUN_TIMEOUT_ENV, "21600"))
    resolved_models: dict[str, dict[str, str]] = {}
    endpoints: dict[str, tuple[str, str, str]] = {}
    for agent in agents:
        endpoints[agent] = _model_endpoint((agent_models or {}).get(agent) or model)
        resolved_models[agent] = {"model": endpoints[agent][0], "source": "nemo-gym"}

    run_dir = _reserve_run_dir(output_dir, datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    tasks_root = run_dir / "_harbor-tasks"
    jobs_dir = run_dir / "_harbor-jobs"
    gym_root = run_dir / "_nemo-gym"
    gym_root.mkdir(parents=True, exist_ok=True)
    arms = [("with", True)] + ([] if skip_baseline else [("without", False)])
    task_resources = {
        key: value for key, value in (("cpus", override_cpus), ("memory_mb", override_memory_mb)) if value is not None
    }
    # Judge credentials reach only the verifier: tasks carry ${NAME} templates that the
    # Gym environment server resolves from its own process environment.
    effective_grading = grading_mode or "default"
    try:
        judge_verifier_env, judge_process_env = verifier_judge_env(skill_path, effective_grading, env_mode)
    except ValueError as exc:
        reporter.emit(ProgressEvent(stage="configuration", state="failed", detail=str(exc)))
        return {"error": [str(exc)]}
    reporter.set_secret_values(set(judge_process_env.values()))
    reporter.emit(ProgressEvent(stage="configuration", state="ready", detail="evaluation config validated"))
    errors: list[str] = []
    case_ids: list[str] = []
    for agent in agents:
        reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="running", detail="NeMo Gym"))
        agent_model, model_url, api_key = endpoints[agent]
        for arm, with_skill in arms:
            arm_dir = tasks_root / agent / arm
            generate_harbor_tasks(
                skill_path,
                arm_dir,
                with_skill=with_skill,
                workspace_skill_paths=include_skills,
                workspace_mode=skill_workspace_mode or "isolated",
                grading_mode=effective_grading,
                verifier_env=judge_verifier_env or None,
                repo_context_exclude_paths=repo_context_exclude_paths,
                task_resources=task_resources or None,
                arm_suffix="" if skip_baseline else f"-{arm}-skill",
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
                n_concurrent=n_concurrent,
                sandbox=sandbox,
            )
            log_path = rollouts_path.with_name("gym.log")
            reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="running", detail=f"NeMo Gym {arm}-skill arm"))
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=rollouts_path.parent,
                        env={**os.environ, **judge_process_env, MODEL_API_KEY_ENV: api_key},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=timeout_s,
                        check=False,
                    )
                    if completed.returncode != 0:
                        errors.append(
                            f"{agent} {arm}-skill: gym eval run exited {completed.returncode}; see {log_path}"
                        )
                except subprocess.TimeoutExpired:
                    errors.append(f"{agent} {arm}-skill: gym eval run timed out after {timeout_s:.0f}s")
            errors += write_job_from_rollouts(
                read_rollouts(rollouts_path),
                jobs_dir / f"{skill_path.name}-{agent}-{arm}",
                eval_name=f"{agent}__{agent_model}",
                task_case_ids=task_case_ids,
                n_attempts=attempts,
                failures=read_rollout_failures(rollouts_path),
            )
        reporter.emit(ProgressEvent(stage=f"agent:{agent}", state="complete"))

    reporter.emit(ProgressEvent(stage="collection", state="running"))
    results = collect_harbor_results(
        skill_name=skill_path.name,
        agents=agents,
        output_dir=run_dir,
        jobs_dir=jobs_dir,
        skip_baseline=skip_baseline,
        n_attempts=attempts,
        pass_threshold=threshold,
        expected_cases=len(case_ids),
        expected_case_ids=case_ids,
        expected_trials=len(case_ids) * attempts,
        env_mode=env_mode,
        agent_models=resolved_models,
        launch_errors=errors,
    )
    reporter.emit(ProgressEvent(stage="collection", state="complete", detail="NeMo Gym results collected"))
    result_path = run_dir / "result.json"
    results.update(
        {
            "skill_name": skill_path.name,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "result_path": str(result_path),
            "execution_backend": {"name": NEMO_GYM_BACKEND, "version": gym_version, "sandbox": sandbox},
            "nemo_gym_dir": str(gym_root),
        }
    )
    if evaluated_source:
        results["evaluated_source"] = dict(evaluated_source)
    result_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    try:
        results["report_path"] = str(
            render_agent_eval_html_report(skill_path, run_dir, env_mode=env_mode, engine_result=results)
        )
    except Exception as exc:
        results["report_warning"] = f"HTML report was not generated: {type(exc).__name__}"
    result_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results
