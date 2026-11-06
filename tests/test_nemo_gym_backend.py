# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the selectable Tier 3 execution backend and the NeMo Gym adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.options import EvaluationOptions
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.nemo_gym.backend import (
    DEFAULT_EXECUTION_BACKEND,
    nemo_gym_bin,
    nemo_gym_capability_errors,
    normalize_backend,
)
from skillevaluator.tier3.nemo_gym.results import rollout_task_name, write_job_from_rollouts
from skillevaluator.tier3.nemo_gym.runner import MODEL_API_KEY_ENV, gym_eval_command, stage_for_gym


def test_backend_defaults_to_harbor_and_is_forwarded() -> None:
    assert DEFAULT_EXECUTION_BACKEND == "harbor"
    assert normalize_backend(None) == "harbor"
    assert normalize_backend("NeMo-Gym") == "nemo-gym"
    assert EvaluationOptions(skill_path=Path("s")).engine_kwargs()["backend"] == "harbor"


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown Tier 3 execution backend"):
        normalize_backend("modal")


def test_capability_errors_reject_unsupported_combinations() -> None:
    assert nemo_gym_capability_errors(agents=["codex"], env_mode="docker", grading_mode=None) == []
    errors = nemo_gym_capability_errors(
        agents=["codex", "gemini-cli"],
        env_mode="modal",
        grading_mode=None,
        harbor_keep_jobs=True,
        stop_on_pass=True,
    )
    joined = " | ".join(errors)
    for fragment in ("gemini-cli", "--env-mode docker", "--harbor-keep-jobs", "--stop-on-pass"):
        assert fragment in joined


def test_gym_bin_uses_configured_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gym = tmp_path / "gym"
    gym.write_text("#!/bin/sh\n")
    gym.chmod(0o755)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(gym))
    assert nemo_gym_bin() == str(gym)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(tmp_path / "missing"))
    assert nemo_gym_bin() is None


def test_gym_command_keeps_api_key_out_of_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_HOST", "172.17.0.1")
    command = gym_eval_command(
        "gym",
        tmp_path / "tasks",
        agent="codex",
        model="m",
        model_url="http://127.0.0.1:9/v1",
        rollouts_path=tmp_path / "r.jsonl",
        n_attempts=2,
        n_concurrent=3,
        sandbox="docker",
    )
    assert command[:4] == ["gym", "eval", "run", str(tmp_path / "tasks")]
    assert command[command.index("--agent") + 1] == "codex_agent"
    assert f"+policy_api_key=${{oc.env:{MODEL_API_KEY_ENV}}}" in command
    assert "+num_repeats=2" in command
    assert command[command.index("--concurrency") + 1] == "3"
    assert "+default_host=172.17.0.1" in command


def _rollout(tmp_path: Path, task: str, attempt: int, overall: float | None) -> dict:
    logs = tmp_path / "gym-logs" / f"{task}-{attempt}"
    logs.mkdir(parents=True)
    if overall is not None:
        (logs / "reward.json").write_text(json.dumps({"overall": overall, "correctness": overall}))
        (logs / "skill_evaluator_reward.json").write_text(json.dumps({"overall": overall}))
    (logs / "trajectory.json").write_text(json.dumps({"schema_version": "ATIF-v1.7", "steps": []}))
    return {
        "_ng_task_id": {"taskset": "t", "task_id": task},
        "episode_id": {"rollout_id": f"r{attempt}", "attempt": 0},
        "reward": 99.0,
        "verifier_return_code": 0 if overall is not None else None,
        "verifier_logs_dir": str(logs),
    }


def test_rollouts_normalize_into_collector_trials(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "calc-codex-with"
    rows = [_rollout(tmp_path, "calc-001", 1, 0.8), _rollout(tmp_path, "calc-002", 1, None)]
    assert rollout_task_name(rows[0]) == "calc-001"
    errors = write_job_from_rollouts(
        rows,
        job_dir,
        eval_name="codex__m",
        task_case_ids={"calc-001": "calc-001", "calc-002": "calc-002"},
        n_attempts=1,
    )
    assert len(errors) == 1 and "calc-002" in errors[0]
    trial = job_dir / "calc-001__gym001"
    reward = json.loads((trial / "verifier" / "reward.json").read_text())
    assert reward["entry_id"] == "calc-001" and reward["overall"] == 0.8
    assert (trial / "agent" / "trajectory.json").is_file()
    assert (trial / "verifier" / "skill_evaluator_reward.json").is_file()
    job = json.loads((job_dir / "result.json").read_text())
    assert job["n_total_trials"] == 2
    assert job["stats"]["n_errors"] == 1
    assert job["stats"]["evals"]["codex__m"]["reward_stats"]["reward"] == {"0.8": ["calc-001__gym001"]}


def test_normalized_gym_jobs_are_read_by_shared_collector(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    for arm, value in (("with", 0.9), ("without", 0.4)):
        write_job_from_rollouts(
            [_rollout(tmp_path / arm, "calc-001", 1, value)],
            jobs / f"calc-codex-{arm}",
            eval_name="codex__m",
            task_case_ids={"calc-001": "calc-001"},
            n_attempts=1,
        )
    results = collect_harbor_results(
        "calc",
        ["codex"],
        tmp_path / "out",
        jobs,
        expected_cases=1,
        expected_case_ids=["calc-001"],
        expected_trials=1,
    )
    agent = results["agents"]["codex"]
    assert agent["overall_with_skill"] == pytest.approx(0.9)
    assert agent["overall_without_skill"] == pytest.approx(0.4)
    assert agent["scored_attempts"]


def _evaluate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides):
    from skillevaluator.provider_config import ProviderConfig
    from skillevaluator.tier3 import commands as tier3_commands
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    skill = tmp_path / "skill"
    skill.mkdir(exist_ok=True)
    provider = ProviderConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4.1-mini",
    )
    calls: dict[str, dict] = {}
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "resolve_results_root", lambda *_args: tmp_path / "results")
    monkeypatch.setattr(tier3_commands, "run_harbor_eval", lambda **kw: calls.setdefault("harbor", kw) and {})
    monkeypatch.setattr(gym_runner, "run_nemo_gym_eval", lambda **kw: calls.setdefault("nemo-gym", kw) and {})
    kwargs = {
        "agents": "codex",
        "env_mode": "docker",
        "skip_baseline": False,
        "n_attempts": None,
        "pass_threshold": None,
        "n_concurrent": None,
        "max_agents": None,
        "model": None,
        "agent_model": (),
        "custom_dockerfile_mode": None,
        "skill_workspace_mode": None,
        "include_skills": (),
        "copy_repo": False,
        "grading_mode": None,
        "results_dir": None,
        "harbor_keep_jobs": False,
        "timeout_multiplier": None,
        "override_cpus": None,
        "override_memory_mb": None,
        "override_storage_mb": None,
    }
    kwargs.update(overrides)
    tier3_commands.evaluate(skill, **kwargs)
    return calls


def test_default_backend_dispatches_to_harbor_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _evaluate(monkeypatch, tmp_path)
    assert set(calls) == {"harbor"}


def test_nemo_gym_backend_dispatches_to_gym_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _evaluate(monkeypatch, tmp_path, backend="nemo-gym", n_attempts=2)
    assert set(calls) == {"nemo-gym"}
    assert calls["nemo-gym"]["agents"] == ["codex"] and calls["nemo-gym"]["n_attempts"] == 2


def test_nemo_gym_rejects_unsupported_env_before_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="docker"):
        _evaluate(monkeypatch, tmp_path, backend="nemo-gym", env_mode="e2b")


def test_stage_for_gym_drops_only_the_skills_dir_hook(tmp_path: Path) -> None:
    task = tmp_path / "calc-001"
    task.mkdir()
    (task / "task.toml").write_text(
        '[environment]\ncpus = 2\nskills_dir = "/workspace/skills"\n\n[environment.env]\nA = "1"\n'
    )
    stage_for_gym(tmp_path)
    text = (task / "task.toml").read_text()
    assert "skills_dir" not in text
    assert "cpus = 2" in text and 'A = "1"' in text


def test_verifier_judge_env_stages_templates_not_values(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.tier3.nemo_gym.runner import verifier_judge_env

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    (tmp_path / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n")
    templates, process_env = verifier_judge_env(tmp_path, "default", "docker")
    assert templates["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"
    assert "sk-test-secret" not in templates.values()
    assert process_env["OPENAI_API_KEY"] == "sk-test-secret"
