# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the selectable Tier 3 execution backend and the NeMo Gym adapter."""

from __future__ import annotations

import json
import os
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


@pytest.fixture
def keep_staged_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the staged task folders, which a Gym run removes at the end, for inspection."""
    from skillevaluator.tier3.harbor import artifact_retention

    class _Keep(artifact_retention.HarborArtifactLifecycle):
        def __init__(self, paths: list[Path], *, keep_requested: bool) -> None:
            del keep_requested
            super().__init__(paths, keep_requested=True)

    monkeypatch.setattr(artifact_retention, "HarborArtifactLifecycle", _Keep)


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


def test_nemo_gym_backend_forwards_copy_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _evaluate(monkeypatch, tmp_path, backend="nemo-gym", copy_repo=True)
    assert calls["nemo-gym"]["copy_repo"] is True


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


def test_nemo_gym_forwards_arm_case_and_model_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _evaluate(
        monkeypatch,
        tmp_path,
        backend="nemo-gym",
        skip_baseline=False,
        skip_with_skill=True,
        max_cases=1,
        agent_model=("codex=gpt-custom",),
    )
    gym_call = calls["nemo-gym"]
    assert gym_call["skip_with_skill"] is True and gym_call["max_cases"] == 1
    assert gym_call["agent_models"] == {"codex": "gpt-custom"}


def test_nemo_gym_rejects_agent_runtime_preflight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="agent-runtime-preflight"):
        _evaluate(monkeypatch, tmp_path, backend="nemo-gym", agent_runtime_preflight=True)


_FAKE_GYM = """#!{python}
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
if args == ["eval", "run", "--help"]:
    print("usage: gym eval run tasks --sandbox docker --agent AGENT --model-type TYPE --model MODEL --model-url URL --output OUTPUT --concurrency N\\n\\npositional arguments:\\n  tasks  Harbor task directory")
    sys.exit(0)
tasks = Path(args[2])
output = Path(args[args.index("--output") + 1])
repeats = int(next(a for a in args if a.startswith("+num_repeats=")).split("=", 1)[1])
Path(os.environ["FAKE_GYM_ENV_LOG"]).open("a").write(json.dumps({{"arm": tasks.name, "repeats": repeats, "env": sorted(os.environ), "argv": args}}) + "\\n")
with output.open("w") as rows:
    for task in sorted(p for p in tasks.iterdir() if (p / "task.toml").is_file()):
        for attempt in range(repeats + int(bool(os.environ.get("FAKE_GYM_EXTRA_ROLLOUT")))):
            logs = output.parent / "logs" / f"{{task.name}}-{{attempt}}"
            logs.mkdir(parents=True)
            if os.environ.get("FAKE_GYM_REWARD_TXT"):
                (logs / "reward.txt").write_text("1\\n")
            else:
                (logs / "reward.json").write_text(json.dumps({{"overall": 0.75}}))
                (logs / "skill_evaluator_reward.json").write_text(json.dumps({{"overall": 0.75}}))
            row = {{"_ng_task_id": {{"taskset": tasks.name, "task_id": task.name}}, "verifier_return_code": 0,
                    "verifier_logs_dir": str(logs), "episode_id": {{"rollout_id": f"r{{attempt}}", "attempt": 0}}}}
            if os.environ.get("FAKE_GYM_FAILURE"):
                row["failure"] = os.environ["FAKE_GYM_FAILURE"]
            rows.write(json.dumps(row) + "\\n")
"""


def _fake_gym_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str = "") -> tuple[Path, Path]:
    import sys

    from skillevaluator import provider_config
    from skillevaluator.provider_config import ProviderConfig
    from skillevaluator.tier3.eval_core import llm_judge

    # Keep report rendering real while replacing its unrelated remote advice request.
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_args, **_kwargs: ("[]", None))

    skill = tmp_path / "calc"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: calc\ndescription: Adds numbers.\n---\n# Calc\n")
    cases = [
        {"id": f"calc-00{i}", "prompt": f"Add {i}+{i}", "expected_output": str(2 * i), "assertions": ["adds"]}
        for i in (1, 2)
    ]
    (skill / "evals" / "evals.json").write_text(json.dumps({"skill_name": "calc", "evals": cases}))
    if config:
        (skill / "evals" / "config.yml").write_text(config)
    gym = tmp_path / "gym-venv" / "gym"
    gym.parent.mkdir()
    gym.write_text(_FAKE_GYM.format(python=sys.executable))
    gym.chmod(0o755)
    env_log = tmp_path / "gym-env.jsonl"
    provider = ProviderConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="judge-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4.1-mini",
    )
    monkeypatch.setattr(provider_config, "resolve_llm_provider", lambda: provider)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(gym))
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_MODEL_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv(MODEL_API_KEY_ENV, "policy-key")
    monkeypatch.setenv("FAKE_GYM_ENV_LOG", str(env_log))
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    monkeypatch.setattr(
        gym_runner,
        "_GYM_HOST_ENV_VARS",
        gym_runner._GYM_HOST_ENV_VARS
        | {"FAKE_GYM_ENV_LOG", "FAKE_GYM_REWARD_TXT", "FAKE_GYM_FAILURE", "FAKE_GYM_EXTRA_ROLLOUT"},
    )
    monkeypatch.setenv("GITHUB_TOKEN", "host-only-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-only-key")
    return skill, env_log


def _run_fake_gym(skill: Path, tmp_path: Path, **overrides) -> dict:
    from skillevaluator.tier3.nemo_gym.runner import run_nemo_gym_eval

    kwargs = {
        "skill_path": skill,
        "agents": ["codex"],
        "skip_baseline": False,
        "n_attempts": None,
        "pass_threshold": None,
        "n_concurrent": 1,
        "model": "policy-model",
        "agent_models": None,
        "skill_workspace_mode": None,
        "include_skills": None,
        "grading_mode": None,
        "output_dir": tmp_path / "results",
        "env_mode": "docker",
        "timeout_multiplier": None,
        "evaluated_source": None,
        "override_cpus": None,
        "override_memory_mb": None,
    }
    kwargs.update(overrides)
    return run_nemo_gym_eval(**kwargs)


@pytest.fixture
def incompatible_gym(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    gym = tmp_path / "released-gym"
    gym.write_text("#!/bin/sh\necho 'usage: gym eval run [--input INPUT] [--agent AGENT] [--output OUTPUT]'\n")
    gym.chmod(0o755)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(gym))
    return gym


def test_incompatible_gym_is_rejected_before_dataset_generation(incompatible_gym: Path, tmp_path: Path) -> None:
    from skillevaluator.tier3.workflow import nemo_gym_preflight_errors

    errors = nemo_gym_preflight_errors(tmp_path, {"agents": "codex", "env_mode": "docker"})
    assert any("incompatible" in error and "--sandbox" in error for error in errors)
    assert not (tmp_path / "evals").exists()


@pytest.mark.parametrize("command", [["evaluate", "--autopilot"], ["tier3"]])
def test_cli_rejects_incompatible_gym_before_generating_a_dataset(
    incompatible_gym: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    from click.testing import CliRunner

    from skillevaluator.cli import cli
    from skillevaluator.evaluation import EvaluationService
    from skillevaluator.tier3 import workflow

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: calc\ndescription: Adds numbers.\n---\n# Calc\n")
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must not generate")
    )
    monkeypatch.setattr(workflow, "_prepare_dataset", lambda *_args, **_kwargs: pytest.fail("must not generate"))
    result = CliRunner().invoke(cli, [*command, str(skill), "--backend", "nemo-gym", "--agents", "codex"])
    assert result.exit_code == 2, result.output
    assert "incompatible" in result.output
    assert not (skill / "evals").exists()


def test_direct_run_rejects_incompatible_gym_before_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    gym = Path(os.environ["SKILL_EVAL_NEMO_GYM_BIN"])
    gym.write_text("#!/bin/sh\necho 'usage: gym eval run [--input INPUT]'\n")
    result = _run_fake_gym(skill, tmp_path)
    assert any("incompatible" in error for error in result.get("error", []))
    assert not (tmp_path / "results").exists()


def test_doctor_rejects_incompatible_gym_despite_readable_version(
    incompatible_gym: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3 import commands
    from skillevaluator.tier3.nemo_gym import backend

    monkeypatch.setattr(backend, "nemo_gym_version", lambda _gym: "0.7.0")
    monkeypatch.setattr(commands, "_docker_daemon_error", lambda _env: None)
    rows = commands._nemo_gym_doctor_rows(["codex"], "docker", {})
    assert rows[0][0:2] == ("NeMo Gym CLI", "fail")
    assert "incompatible" in rows[0][2]
    assert not any(status == "pass" for _name, status, _detail in rows)


def test_gym_cli_probe_handles_unreadable_help(tmp_path: Path) -> None:
    from skillevaluator.tier3.nemo_gym.backend import nemo_gym_cli_error

    gym = tmp_path / "gym"
    gym.write_bytes(b"#!/bin/sh\nprintf '\\377'\n")
    gym.chmod(0o755)
    assert "incompatible" in (nemo_gym_cli_error(str(gym)) or "")


def test_gym_cli_probe_timeout_does_not_echo_subprocess_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from skillevaluator.tier3.nemo_gym import backend

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("gym", 15, output="policy-secret")

    monkeypatch.setattr(backend.subprocess, "run", timeout)
    error = backend.nemo_gym_cli_error("/missing/gym")
    assert error is not None and "15s" in error
    assert "policy-secret" not in error


def test_gym_cli_probe_omits_policy_credentials_and_ignores_raw_error_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from skillevaluator.tier3.nemo_gym.backend import nemo_gym_cli_error

    gym = tmp_path / "gym"
    gym.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "assert 'SKILL_EVAL_NEMO_GYM_MODEL_API_KEY' not in os.environ\n"
        "assert 'OPENAI_API_KEY' not in os.environ\n"
        "assert sys.argv[1:] == ['eval', 'run', '--help']\n"
        "print('policy-secret', file=sys.stderr)\nsys.exit(7)\n"
    )
    gym.chmod(0o755)
    monkeypatch.setenv(MODEL_API_KEY_ENV, "policy-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "judge-secret")
    error = nemo_gym_cli_error(str(gym))
    assert error is not None and "exited 7" in error
    assert "policy-secret" not in error


def test_fake_gym_run_honors_arm_case_limit_config_and_is_discoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_staged_tasks: None
) -> None:
    from skillevaluator.tier3.results_location import ordered_run_directories

    skill, env_log = _fake_gym_skill(tmp_path, monkeypatch, config="schema_version: 1\nharbor:\n  n_attempts: 2\n")
    result = _run_fake_gym(skill, tmp_path, skip_with_skill=True, max_cases=1)

    run_dir = Path(result["run_dir"])
    assert sorted(p.name for p in (run_dir / "_harbor-tasks" / "codex").iterdir()) == ["without"]
    assert len(list((run_dir / "_harbor-tasks" / "codex" / "without").glob("*/task.toml"))) == 1
    launches = [json.loads(line) for line in env_log.read_text().splitlines()]
    assert [(launch["arm"], launch["repeats"]) for launch in launches] == [("without", 2)]
    assert MODEL_API_KEY_ENV in launches[0]["env"]
    assert "GITHUB_TOKEN" not in launches[0]["env"] and "ANTHROPIC_API_KEY" not in launches[0]["env"]
    assert result["run_config"]["case_selection"]["selected_ids"] == ["calc-001"]
    assert result["attempt_policy"]["max_attempts"] == 2
    assert result["agents"]["codex"]["num_trials_with"] == 0
    assert ordered_run_directories(tmp_path / "results", completed_only=True) == [run_dir]
    assert json.loads((run_dir / "run_config.json").read_text()) == result["run_config"]


def test_excess_rollouts_fail_the_run_and_rerendered_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory

    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_GYM_EXTRA_ROLLOUT", "1")
    result = _run_fake_gym(skill, tmp_path, n_attempts=1, max_cases=1)
    assert result["execution_status"] == "failed"
    assert result["agents"]["codex"]["num_trials_with"] == 1
    assert result["agents"]["codex"]["num_trials_without"] == 1
    assert any("2/1 rollouts" in error for error in result["execution_errors"])
    validation = agent_eval_result_from_directory(skill, Path(result["run_dir"]))
    assert validation is not None and not validation.passed
    assert validation.metadata["agent_eval"]["summary"]["execution_status"] == "failed"


def test_fake_gym_run_rejects_config_options_gym_cannot_honor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill, env_log = _fake_gym_skill(
        tmp_path, monkeypatch, config="schema_version: 1\nharbor:\n  n_attempts: 2\n  stop_on_pass: true\n"
    )
    result = _run_fake_gym(skill, tmp_path)
    assert "stop-on-pass" in result["error"][0]
    assert not env_log.exists()


def test_cli_rejects_unsupported_gym_flags_before_autopilot_writes_a_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from skillevaluator.cli import cli
    from skillevaluator.evaluation import EvaluationService

    skill = tmp_path / "calc"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: calc\ndescription: Adds numbers.\n---\n# Calc\n")
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must not generate")
    )
    result = CliRunner().invoke(
        cli,
        ["evaluate", str(skill), "--backend", "nemo-gym", "--autopilot", "--harbor-keep-jobs", "--progress", "off"],
    )
    assert result.exit_code == 2, result.output
    assert "--harbor-keep-jobs" in result.output
    assert not (skill / "evals" / "evals.json").exists()


def test_fake_gym_run_uses_config_agent_model_and_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_staged_tasks: None
) -> None:
    config = "schema_version: 1\nharbor:\n  agents:\n    codex:\n      model: cfg-model\n  resources:\n    cpus: 3\n"
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch, config=config)
    result = _run_fake_gym(skill, tmp_path, model=None, skip_baseline=True)

    assert result["run_config"]["agents"]["codex"] == {
        "agent": "codex",
        "model": "cfg-model",
        "source": "evals/config.yml",
    }
    task_tomls = list((Path(result["run_dir"]) / "_harbor-tasks" / "codex" / "with").glob("*/task.toml"))
    assert task_tomls and all("cpus = 3" in path.read_text() for path in task_tomls)


@pytest.mark.parametrize(
    "harbor_config",
    ["  resources:\n    storage_mb: 4096\n"],
)
def test_fake_gym_run_rejects_harbor_only_config_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor_config: str
) -> None:
    skill, env_log = _fake_gym_skill(tmp_path, monkeypatch, config=f"schema_version: 1\nharbor:\n{harbor_config}")
    result = _run_fake_gym(skill, tmp_path)
    assert "not supported with --backend nemo-gym" in result["error"][0]
    assert not env_log.exists()


def test_fake_gym_run_rejects_pre_agent_setup_with_a_baseline_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill, env_log = _fake_gym_skill(
        tmp_path, monkeypatch, config="schema_version: 1\nharbor:\n  pre_agent_setup:\n    - echo hi\n"
    )
    result = _run_fake_gym(skill, tmp_path)
    assert "cannot run in a paired evaluation" in result["error"][0]
    assert not env_log.exists()


def test_fake_gym_run_stages_agent_workdir_and_pre_agent_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = "schema_version: 1\nharbor:\n  agent_workdir: /work\n  setup_commands:\n    - touch /tmp/ready\n"
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch, config=config)
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    staged: list[str] = []
    original_stage = gym_runner.stage_for_gym

    def recording_stage(arm_dir: Path, *args, **kwargs) -> None:
        original_stage(arm_dir, *args, **kwargs)
        staged.extend(path.read_text() for path in sorted(arm_dir.glob("*/task.toml")))

    monkeypatch.setattr(gym_runner, "stage_for_gym", recording_stage)
    result = _run_fake_gym(skill, tmp_path, skip_baseline=True)
    assert result.get("execution_status") == "succeeded", result
    assert staged
    for text in staged:
        assert 'workdir = "/work"' in text
        assert "[environment.healthcheck]" in text
        assert "touch /tmp/ready" in text


def test_fake_gym_run_rejects_an_invalid_agent_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill, env_log = _fake_gym_skill(
        tmp_path, monkeypatch, config="schema_version: 1\nharbor:\n  agent_workdir: relative/dir\n"
    )
    result = _run_fake_gym(skill, tmp_path)
    assert any("agent_workdir" in error for error in result["error"])
    assert not env_log.exists()


def test_truncated_rollout_line_is_skipped_and_reported(tmp_path: Path) -> None:
    from skillevaluator.tier3.nemo_gym.results import read_rollouts

    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_bytes(b'{"task_id": "a"}\n{"task_id": "b", "reward\xff')
    errors: list[str] = []
    assert read_rollouts(rollouts, errors) == [{"task_id": "a"}]
    assert "1 unreadable line" in errors[0]


def test_custom_model_url_requires_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.nemo_gym.runner import _model_endpoint

    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_MODEL_URL", "http://policy.example/v1")
    monkeypatch.delenv(MODEL_API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=MODEL_API_KEY_ENV):
        _model_endpoint("policy-model")
    monkeypatch.setenv(MODEL_API_KEY_ENV, "EMPTY")
    assert _model_endpoint("policy-model") == ("policy-model", "http://policy.example/v1", "EMPTY")


@pytest.mark.parametrize("value", ["abc", "0", "-5", "nan", "inf"])
def test_invalid_gym_timeout_is_rejected_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    skill, env_log = _fake_gym_skill(tmp_path, monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_TIMEOUT_S", value)
    result = _run_fake_gym(skill, tmp_path)
    assert "SKILL_EVAL_NEMO_GYM_TIMEOUT_S" in result["error"][0]
    assert not env_log.exists()


def test_gym_bin_is_absolute_and_version_follows_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from skillevaluator.tier3.nemo_gym.backend import _gym_interpreters, nemo_gym_bin

    real = tmp_path / "tools" / "gym"
    real.parent.mkdir()
    real.write_text(f"#!{sys.executable}\nprint('gym')\n")
    real.chmod(0o755)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "gym").symlink_to(real)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", "bin/gym")
    found = nemo_gym_bin()
    assert found is not None and Path(found).is_absolute()
    assert _gym_interpreters(found)[0] == Path(sys.executable)


def test_failed_gym_launch_is_reported_not_succeeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    failing = tmp_path / "failing-gym"
    failing.write_text("#!/bin/sh\nexit 3\n")
    failing.chmod(0o755)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(failing))
    result = _run_fake_gym(skill, tmp_path)
    assert result.get("execution_status") != "succeeded"
    assert "exited 3" in json.dumps(result)


def test_rerendered_report_keeps_a_failed_gym_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory, render_agent_eval_html_report

    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    gym = Path(os.environ["SKILL_EVAL_NEMO_GYM_BIN"])
    wrapper = tmp_path / "gym-then-exit-3"
    wrapper.write_text(f'#!/bin/sh\n"{gym}" "$@"\ncase "$*" in *--help*) exit 0;; esac\nexit 3\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(wrapper))
    result = _run_fake_gym(skill, tmp_path)
    run_dir = Path(result["run_dir"])
    assert result["execution_status"] == "failed"
    (run_dir / "report.html").unlink(missing_ok=True)

    report = render_agent_eval_html_report(skill, run_dir)
    assert "exited 3" in report.read_text(encoding="utf-8")
    validation = agent_eval_result_from_directory(skill, run_dir)
    assert validation is not None and not validation.passed
    payload = validation.metadata["agent_eval"]
    assert payload["summary"]["execution_status"] == "failed"
    assert any("exited 3" in error for error in payload["summary"]["execution_errors"])
    assert payload["agents"]["codex"]["execution_status"] == "failed"


def test_gym_run_publishes_latest_stages_runtime_env_and_records_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_staged_tasks: None
) -> None:
    from skillevaluator.tier3.results_location import publish_latest_results

    skill, _env_log = _fake_gym_skill(
        tmp_path, monkeypatch, config='schema_version: 1\nharbor:\n  runtime_env:\n    GYM_TEST_FLAG: "on"\n'
    )
    earlier = tmp_path / "results" / "20200101_000000_1_harbor"
    earlier.mkdir(parents=True)
    publish_latest_results(tmp_path / "results", earlier.name)
    result = _run_fake_gym(skill, tmp_path, skip_with_skill=True, max_cases=1)
    run_dir = Path(result["run_dir"])
    assert (tmp_path / "results" / "latest").resolve() == run_dir.resolve()
    task_toml = next((run_dir / "_harbor-tasks" / "codex" / "without").glob("*/task.toml")).read_text()
    assert '"GYM_TEST_FLAG" = "${GYM_TEST_FLAG}"' in task_toml
    assert "on" not in task_toml.split("GYM_TEST_FLAG", 1)[1].splitlines()[0]
    assert result["dataset_digest"] and (run_dir / "dataset_snapshot.json").is_file()
    assert "publication_target" in result


def test_config_errors_accept_decoys_and_timeout_multiplier(tmp_path: Path) -> None:
    from skillevaluator.tier3.nemo_gym.backend import nemo_gym_config_errors

    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  timeout_multiplier: 2\n"
        "skill_workspace:\n  mode: group\n  decoy_manifest: decoys.json\n"
    )
    assert nemo_gym_config_errors(tmp_path) == []


def test_fake_gym_run_stages_frozen_decoys_in_every_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import decoys
    from skillevaluator.tier3.decoy_identity import decoy_source_digest
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    config = "schema_version: 1\nskill_workspace:\n  mode: group\n  decoy_manifest: frozen.json\n"
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch, config=config)
    decoy = tmp_path / "catalog" / "decoy"
    decoy.mkdir(parents=True)
    (decoy / "SKILL.md").write_text("---\nname: decoy\ndescription: Use this decoy workflow.\n---\n# decoy\n")
    manifest = {"selected": [{"path": "decoy", "skill_name": "decoy", "source_digest": decoy_source_digest(decoy)}]}
    monkeypatch.setattr(decoys, "load_decoy_manifest", lambda *_: ([decoy], manifest))
    staged_arms: list[str] = []
    original_stage = gym_runner.stage_for_gym

    def recording_stage(arm_dir: Path, *args, **kwargs) -> None:
        original_stage(arm_dir, *args, **kwargs)
        if any(path.parent.name == "decoy" for path in arm_dir.rglob("SKILL.md")):
            staged_arms.append(arm_dir.name)

    monkeypatch.setattr(gym_runner, "stage_for_gym", recording_stage)
    result = _run_fake_gym(skill, tmp_path)
    assert result.get("execution_status") == "succeeded", result
    assert "with" in staged_arms
    run_config = json.loads((Path(result["run_dir"]) / "run_config.json").read_text())
    assert run_config["skill_workspace"] == {"mode": "group", "decoy_manifest": manifest}


def test_timeout_multiplier_scales_staged_task_timeouts(tmp_path: Path) -> None:
    import tomllib

    from skillevaluator.tier3.nemo_gym.runner import stage_for_gym

    task = tmp_path / "case-1"
    task.mkdir()
    (task / "task.toml").write_text(
        'instruction = """\n[agent]\ntimeout_sec = 1\n"""\n\n[agent]\ntimeout_sec = 300.0\n\n'
        '[verifier]\ntimeout_sec = 600  # judges\n\n[verifier.env]\nA = "1"\n\n[environment]\ncpus = 1\n'
    )
    stage_for_gym(tmp_path, timeout_multiplier=2.5)
    data = tomllib.loads((task / "task.toml").read_text())
    assert data["agent"]["timeout_sec"] == 750.0
    assert data["verifier"]["timeout_sec"] == 1500.0
    assert data["environment"]["build_timeout_sec"] == 1500.0
    assert data["environment"]["cpus"] == 1
    assert "timeout_sec = 1\n" in data["instruction"]


def test_timeout_multiplier_is_validated_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_staged_tasks: None
) -> None:
    import tomllib

    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    bad = _run_fake_gym(skill, tmp_path, timeout_multiplier=0.0)
    assert any("timeout_multiplier" in error for error in bad["error"])

    result = _run_fake_gym(skill, tmp_path, timeout_multiplier=2.0, skip_baseline=True, max_cases=1)
    run_dir = Path(result["run_dir"])
    task_toml = next((run_dir / "_harbor-tasks" / "codex" / "with").glob("*/task.toml")).read_text()
    assert tomllib.loads(task_toml)["agent"]["timeout_sec"] == 600.0
    run_config = json.loads((run_dir / "run_config.json").read_text())
    assert run_config["harbor"]["timeout_multiplier"] == 2.0


def test_passthrough_env_alias_reaches_the_gym_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_staged_tasks: None
) -> None:
    skill, _env_log = _fake_gym_skill(
        tmp_path, monkeypatch, config="schema_version: 1\nharbor:\n  passthrough_env:\n    - GYM_ALIAS_FLAG\n"
    )
    monkeypatch.setenv("GYM_ALIAS_FLAG", "on")
    result = _run_fake_gym(skill, tmp_path, skip_with_skill=True, max_cases=1)
    task_toml = next((Path(result["run_dir"]) / "_harbor-tasks" / "codex" / "without").glob("*/task.toml")).read_text()
    assert '"GYM_ALIAS_FLAG" = "${GYM_ALIAS_FLAG}"' in task_toml


def test_config_errors_let_cli_overrides_win(tmp_path: Path) -> None:
    from skillevaluator.tier3.nemo_gym.backend import nemo_gym_config_errors

    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  stop_on_pass: true\n  timeout_multiplier: 2\n"
        "  agent_runtime_preflight: true\n  custom_dockerfile_mode: preserve\n"
    )
    assert len(nemo_gym_config_errors(tmp_path)) == 2
    assert (
        nemo_gym_config_errors(
            tmp_path,
            stop_on_pass=False,
            timeout_multiplier=1.0,
            agent_runtime_preflight=False,
            custom_dockerfile_mode="rebase",
        )
        == []
    )


def test_workflow_preflight_rejects_environment_kwarg(tmp_path: Path) -> None:
    import click

    from skillevaluator.tier3.workflow import _preflight_nemo_gym

    with pytest.raises(click.UsageError, match="--environment-kwarg"):
        _preflight_nemo_gym(tmp_path, {"agents": "codex", "env_mode": "docker", "environment_kwarg": ("cpus=2",)})


def test_version_probe_ignores_the_callers_directory_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from skillevaluator.tier3.nemo_gym import backend

    python = tmp_path / "venv" / "python"
    python.parent.mkdir()
    python.write_text("")
    calls: list[dict] = []

    def _run(args, **kwargs):
        calls.append({"args": args, **kwargs})
        return subprocess.CompletedProcess(args, 0, stdout="0.7.0\n", stderr="")

    monkeypatch.setattr(backend, "_gym_interpreters", lambda _gym: [python])
    monkeypatch.setattr(backend.subprocess, "run", _run)
    monkeypatch.setenv("GITHUB_TOKEN", "host-only-token")
    assert backend.nemo_gym_version("gym") == "0.7.0"
    assert calls[0]["args"][1] == "-P"
    assert calls[0]["cwd"] == python.parent
    assert "GITHUB_TOKEN" not in calls[0]["env"]


def test_stage_for_gym_adds_job_level_judge_env(tmp_path: Path) -> None:
    import tomllib

    task = tmp_path / "arm" / "calc-001"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        'skills_dir = "/skills"\n[verifier]\ntimeout_sec = 1.0\n\n[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
    )
    stage_for_gym(
        tmp_path / "arm",
        {"LLM_JUDGE_MODEL": "${LLM_JUDGE_MODEL}", "OPENAI_API_KEY": "${OTHER}"},
    )
    staged = tomllib.loads((task / "task.toml").read_text())
    assert "skills_dir" not in staged
    assert staged["verifier"]["env"] == {"OPENAI_API_KEY": "${OPENAI_API_KEY}", "LLM_JUDGE_MODEL": "${LLM_JUDGE_MODEL}"}


def test_missing_attempts_become_failed_trials_and_keep_scored_ones(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    errors = write_job_from_rollouts(
        [_rollout(tmp_path / "with", "calc-001", 1, 0.9)],
        job_dir,
        eval_name="codex__m",
        task_case_ids={"calc-001": "calc-001"},
        n_attempts=2,
    )
    assert len(errors) == 1 and "1/2" in errors[0]
    assert len([path for path in job_dir.iterdir() if path.is_dir()]) == 2
    stats = json.loads((job_dir / "result.json").read_text())["stats"]
    assert stats["n_trials"] == 2 and stats["n_errors"] == 1
    assert stats["evals"]["codex__m"]["n_trials"] == 1


def test_rollout_reader_splits_only_on_newlines(tmp_path: Path) -> None:
    from skillevaluator.tier3.nemo_gym.results import read_rollouts

    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps({"text": "a\u2028b\u0085c"}, ensure_ascii=False) + "\n", encoding="utf-8")
    errors: list[str] = []
    assert read_rollouts(path, errors) == [{"text": "a\u2028b\u0085c"}] and errors == []


def test_codex_inherits_the_sandbox_environment_when_runtime_env_is_set(tmp_path: Path) -> None:
    kwargs = {
        "agent": "codex",
        "model": "m",
        "model_url": "http://127.0.0.1:9/v1",
        "rollouts_path": tmp_path / "r.jsonl",
        "n_attempts": 1,
        "n_concurrent": 1,
        "sandbox": "docker",
    }
    override = "++codex_agent.responses_api_agents.codex_agent.inherit_sandbox_environment=true"
    assert override in gym_eval_command("gym", tmp_path, inherit_sandbox_environment=True, **kwargs)
    assert override not in gym_eval_command("gym", tmp_path, **kwargs)


def test_model_url_with_userinfo_is_rejected() -> None:
    from skillevaluator.tier3.nemo_gym.backend import model_url_credential_error

    assert model_url_credential_error("https://user:pass@models.example/v1")
    assert model_url_credential_error("https://models.example/v1") is None


def test_gym_environment_keeps_uv_credentials_and_anchors_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3.nemo_gym.runner import gym_launch_environment

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_INDEX_INTERNAL_PASSWORD", "index-secret")
    monkeypatch.setenv("NETRC", "netrc")
    monkeypatch.setenv("SSL_CERT_FILE", "certs/ca.pem")
    env = gym_launch_environment({})
    assert env["UV_INDEX_INTERNAL_PASSWORD"] == "index-secret"
    assert env["NETRC"] == "netrc" or env["NETRC"] == str(tmp_path / "netrc")
    assert env["SSL_CERT_FILE"] == str(tmp_path / "certs" / "ca.pem")


def test_native_harbor_task_source_resolves_like_harbor(tmp_path: Path) -> None:
    from skillevaluator.tier3.nemo_gym.backend import uses_native_harbor_tasks

    skill = tmp_path / "skill"
    (skill / "evals" / "harbor" / "task-1").mkdir(parents=True)
    assert uses_native_harbor_tasks(skill, "auto")
    (skill / "evals" / "evals.json").write_text("{}")
    assert not uses_native_harbor_tasks(skill, "auto")
    assert uses_native_harbor_tasks(skill, "native_harbor")


def test_gym_log_is_redacted(tmp_path: Path) -> None:
    import sys

    from skillevaluator.tier3.nemo_gym.runner import _run_gym

    log_path = tmp_path / "gym.log"
    with log_path.open("w", encoding="utf-8") as log:
        code = _run_gym(
            [sys.executable, "-c", "print('key=super-secret-value done')"],
            cwd=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            log=log,
            timeout_s=30,
            secret_values={"super-secret-value"},
        )
    assert code == 0
    text = log_path.read_text()
    assert "super-secret-value" not in text and "done" in text


def test_gym_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    import os
    import sys

    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    pid_file = tmp_path / "child.pid"
    script = (
        "import os, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    with (tmp_path / "gym.log").open("w") as log:
        code = gym_runner._run_gym(
            [sys.executable, "-c", script], cwd=tmp_path, env=dict(os.environ), log=log, timeout_s=2
        )
    assert code is None
    child = int(pid_file.read_text())
    import time

    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("child of the Gym leader survived")


def test_gym_judge_override_reaches_the_staged_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_staged_tasks: None
) -> None:
    import tomllib

    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_MODEL", "judge-override-model")
    result = _run_fake_gym(skill, tmp_path)
    task_toml = next((Path(result["run_dir"]) / "_harbor-tasks" / "codex" / "with").glob("*/task.toml"))
    verifier_env = tomllib.loads(task_toml.read_text())["verifier"]["env"]
    assert verifier_env["LLM_JUDGE_MODEL"] == "${LLM_JUDGE_MODEL}"
    assert result["run_config"]["judge"]["model"] == "judge-override-model"


def test_gym_run_removes_staged_tasks_and_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    result = _run_fake_gym(skill, tmp_path)
    run_dir = Path(result["run_dir"])
    assert not (run_dir / "_harbor-tasks").exists() and not (run_dir / "_harbor-jobs").exists()
    assert (run_dir / "_nemo-gym").is_dir()
    assert result["run_config"]["harbor"]["jobs_retained"] is False
    assert json.loads((run_dir / "run_config.json").read_text())["harbor"]["jobs_retained"] is False


def test_doctor_reports_nemo_gym_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import commands
    from skillevaluator.tier3.nemo_gym import backend
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    gym = tmp_path / "gym"
    import sys

    gym.write_text(_FAKE_GYM.format(python=sys.executable))
    gym.chmod(0o755)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(gym))
    monkeypatch.setattr(backend, "nemo_gym_version", lambda _bin: "0.7.0")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(commands, "_docker_daemon_error", lambda _env: None)
    monkeypatch.setattr(gym_runner, "_model_endpoint", lambda model: (model or "m", "http://127.0.0.1:9/v1", "k"))
    rows = commands._nemo_gym_doctor_rows(["codex"], "docker", {})
    assert rows == [
        ("NeMo Gym CLI", "pass", f"{gym} (nemo-gym 0.7.0)"),
        ("NeMo Gym backend", "pass", "docker sandbox ready"),
    ]

    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(tmp_path / "missing"))
    rows = commands._nemo_gym_doctor_rows(["codex", "goose"], "local", {})
    assert rows[0][:2] == ("NeMo Gym CLI", "fail")
    details = " ".join(detail for _name, status, detail in rows[1:] if status == "fail")
    assert "goose" in details and "docker only" in details


def test_tier3_doctor_accepts_backend_option() -> None:
    from click.testing import CliRunner

    from skillevaluator.cli import cli

    result = CliRunner().invoke(cli, ["tier3", "doctor", "--help"])
    assert result.exit_code == 0 and "--backend" in result.output


def test_fake_gym_run_stages_native_harbor_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    config = "schema_version: 1\nharbor:\n  task_source: native_harbor\ngrading:\n  mode: custom_only\n"
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch, config=config)
    task = skill / "evals" / "harbor" / "case-001"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[metadata]\nentry_id = "case-001"\n\n'
        "[agent]\ntimeout_sec = 300.0\n\n[verifier]\ntimeout_sec = 600.0\n\n[environment]\n"
        'skills_dir = "/workspace/skills"\n'
    )
    (task / "instruction.md").write_text("Native instruction NATIVE-7341.\n")
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /workspace\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    staged: dict[str, str] = {}
    original_stage = gym_runner.stage_for_gym

    def recording_stage(arm_dir: Path, *args, **kwargs) -> None:
        original_stage(arm_dir, *args, **kwargs)
        for instruction in arm_dir.glob("*/instruction.md"):
            staged[arm_dir.name] = instruction.read_text()
            assert "skills_dir" not in (instruction.parent / "task.toml").read_text()

    monkeypatch.setattr(gym_runner, "stage_for_gym", recording_stage)
    # The fake Gym writes what the native test.sh writes: only reward.txt.
    monkeypatch.setenv("FAKE_GYM_REWARD_TXT", "1")
    result = _run_fake_gym(skill, tmp_path)
    assert set(staged) == {"with", "without"}
    assert all("NATIVE-7341" in text for text in staged.values())
    run_config = json.loads((Path(result["run_dir"]) / "run_config.json").read_text())
    assert run_config["task_source"] == "native_harbor"
    assert result.get("execution_status") != "failed", result.get("execution_errors")
    assert result["agents"]["codex"]["num_trials_with"] == 1
    assert result["agents"]["codex"]["num_trials_without"] == 1


def test_fake_gym_run_reuses_the_shared_base_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.harbor import adapter
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    skill, _env_log = _fake_gym_skill(
        tmp_path, monkeypatch, config="schema_version: 1\nharbor:\n  base_image_mode: rebuild\n"
    )
    builds: list[bool] = []

    def fake_build(*_args, force_rebuild: bool = False, **_kwargs) -> str:
        builds.append(force_rebuild)
        return "skillevaluator-base:test"

    monkeypatch.setattr(adapter, "build_eval_base_image", fake_build)
    dockerfiles: list[str] = []
    original_stage = gym_runner.stage_for_gym

    def recording_stage(arm_dir: Path, *args, **kwargs) -> None:
        original_stage(arm_dir, *args, **kwargs)
        dockerfiles.extend(path.read_text() for path in arm_dir.glob("*/environment/Dockerfile"))

    monkeypatch.setattr(gym_runner, "stage_for_gym", recording_stage)
    result = _run_fake_gym(skill, tmp_path)
    assert builds == [True]
    assert dockerfiles and all("FROM skillevaluator-base:test" in text for text in dockerfiles)
    run_config = json.loads((Path(result["run_dir"]) / "run_config.json").read_text())
    assert run_config["harbor"]["base_image_mode"] == "rebuild"


class _RecordingReporter:
    def __init__(self) -> None:
        self.plans: list = []
        self.events: list = []

    def start(self, plan) -> None:
        self.plans.append(plan)

    def set_secret_values(self, values) -> None:
        del values

    def emit(self, event) -> None:
        self.events.append(event)

    def heartbeat(self) -> None:
        pass

    def close(self) -> None:
        pass


def _native_task(skill: Path, extra: str = "") -> Path:
    task = skill / "evals" / "harbor" / "case-001"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[metadata]\nentry_id = "case-001"\n\n'
        '[agent]\ntimeout_sec = 300.0\n\n[environment]\nskills_dir = "/workspace/skills"\n' + extra
    )
    (task / "instruction.md").write_text("Native instruction.\n")
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return task


def test_codex_inherits_the_task_environment_without_runtime_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = "schema_version: 1\nharbor:\n  task_source: native_harbor\ngrading:\n  mode: custom_only\n"
    skill, env_log = _fake_gym_skill(tmp_path, monkeypatch, config=config)
    _native_task(skill, '\n[environment.env]\nSERVICE_URL = "http://svc:8080"\n')
    _run_fake_gym(skill, tmp_path, skip_baseline=True)
    override = "++codex_agent.responses_api_agents.codex_agent.inherit_sandbox_environment=true"
    launches = [json.loads(line) for line in env_log.read_text().splitlines()]
    assert launches and all(override in launch["argv"] for launch in launches)


def test_reward_txt_is_scored_like_harbor(tmp_path: Path) -> None:
    from skillevaluator.tier3.nemo_gym.results import read_verifier_rewards

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.txt").write_text("0.5\n")
    assert read_verifier_rewards(logs) == {"reward": 0.5}
    (logs / "reward.txt").write_text("nan\n")
    assert read_verifier_rewards(logs) is None
    (logs / "reward.json").write_text(json.dumps({"overall": 0.9}))
    assert read_verifier_rewards(logs) == {"overall": 0.9}
    row = {
        "_ng_task_id": {"taskset": "arm", "task_id": "t1"},
        "verifier_return_code": 0,
        "verifier_logs_dir": str(tmp_path / "txt"),
    }
    (tmp_path / "txt").mkdir()
    (tmp_path / "txt" / "reward.txt").write_text("1\n")
    job = tmp_path / "job"
    errors = write_job_from_rollouts([row], job, eval_name="codex__m", task_case_ids={"t1": "case-1"}, n_attempts=1)
    assert errors == []
    trial = json.loads((job / "t1__gym001" / "result.json").read_text())
    assert trial["verifier_result"]["rewards"] == {"entry_id": "case-1", "reward": 1.0}


def test_rollout_row_failures_are_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    secret = "gw-policy-0a1b2c3d4e5f6789"
    monkeypatch.setenv(MODEL_API_KEY_ENV, secret)
    monkeypatch.setenv("FAKE_GYM_FAILURE", f"AuthenticationError: 401 invalid key {secret}")
    result = _run_fake_gym(skill, tmp_path, skip_baseline=True, max_cases=1)
    run_dir = Path(result["run_dir"])
    assert "AuthenticationError" in json.dumps(result)
    assert secret not in json.dumps(result)
    assert all(secret not in path.read_text() for path in run_dir.rglob("*.json"))


def test_cli_checks_gym_settings_before_autopilot_writes_a_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from skillevaluator.cli import cli
    from skillevaluator.evaluation import EvaluationService

    skill = tmp_path / "calc"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: calc\ndescription: Adds numbers.\n---\n# Calc\n")
    gym = tmp_path / "gym"
    gym.write_text("#!/bin/sh\n")
    gym.chmod(0o755)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(gym))
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must not generate")
    )
    base = ["evaluate", str(skill), "--backend", "nemo-gym", "--autopilot", "--progress", "off"]
    cases = [
        ({"SKILL_EVAL_NEMO_GYM_TIMEOUT_S": "abc"}, [], "SKILL_EVAL_NEMO_GYM_TIMEOUT_S"),
        ({"SKILL_EVAL_NEMO_GYM_MODEL_URL": "http://127.0.0.1:9/v1"}, [], "SKILL_EVAL_NEMO_GYM_MODEL_API_KEY"),
        (
            {"SKILL_EVAL_NEMO_GYM_MODEL_URL": "http://u:p@127.0.0.1:9/v1", MODEL_API_KEY_ENV: "k"},
            [],
            "username or password",
        ),
        ({}, ["--override-cpus", "0"], "--override-cpus"),
        ({}, ["--override-memory-mb", "0"], "--override-memory-mb"),
    ]
    for env, flags, expected in cases:
        with monkeypatch.context() as patched:
            for name, value in env.items():
                patched.setenv(name, value)
            result = CliRunner().invoke(cli, [*base, *flags])
        assert result.exit_code == 2, (env, flags, result.output)
        assert expected in result.output, (expected, result.output)
        assert not (skill / "evals").exists()


def test_missing_dataset_is_rejected_before_a_run_dir_is_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    (skill / "evals" / "evals.json").unlink()
    result = _run_fake_gym(skill, tmp_path)
    assert "No evals/evals.json found" in json.dumps(result)
    results_root = tmp_path / "results"
    assert not results_root.exists() or not any(results_root.iterdir())


def test_native_task_mcp_servers_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.nemo_gym.backend import NATIVE_MCP_SERVERS_ERROR, nemo_gym_config_errors

    config = "schema_version: 1\nharbor:\n  task_source: native_harbor\ngrading:\n  mode: custom_only\n"
    skill, env_log = _fake_gym_skill(tmp_path, monkeypatch, config=config)
    _native_task(skill, '\n[[environment.mcp_servers]]\nname = "files"\ntransport = "stdio"\ncommand = "srv"\n')
    assert NATIVE_MCP_SERVERS_ERROR in nemo_gym_config_errors(skill)
    result = _run_fake_gym(skill, tmp_path)
    assert NATIVE_MCP_SERVERS_ERROR in json.dumps(result)
    assert not env_log.exists()
    staged = tmp_path / "staged" / "case-001"
    staged.mkdir(parents=True)
    (staged / "task.toml").write_text((skill / "evals" / "harbor" / "case-001" / "task.toml").read_text())
    with pytest.raises(ValueError, match="mcp_servers"):
        stage_for_gym(staged.parent)


def test_stage_for_gym_applies_resource_overrides_and_scales_the_default_verifier_timeout(tmp_path: Path) -> None:
    import tomllib

    with_env = tmp_path / "a" / "t1"
    with_env.mkdir(parents=True)
    (with_env / "task.toml").write_text(
        '[agent]\ntimeout_sec = 300.0\n\n[verifier.env]\nX = "1"\n\n[environment]\ncpus = 2\nmemory_mb = 4096\n'
    )
    stage_for_gym(with_env.parent, timeout_multiplier=2.0, resource_overrides={"cpus": 4, "memory_mb": 8192})
    data = tomllib.loads((with_env / "task.toml").read_text())
    assert data["environment"]["cpus"] == 4 and data["environment"]["memory_mb"] == 8192
    assert data["verifier"]["timeout_sec"] == 1200.0 and data["verifier"]["env"] == {"X": "1"}
    assert data["agent"]["timeout_sec"] == 600.0

    no_env = tmp_path / "b" / "t1"
    no_env.mkdir(parents=True)
    (no_env / "task.toml").write_text("[verifier]\ntimeout_sec = 100.0\n")
    stage_for_gym(no_env.parent, timeout_multiplier=2.0, resource_overrides={"cpus": 4})
    data = tomllib.loads((no_env / "task.toml").read_text())
    assert data["environment"]["cpus"] == 4 and data["verifier"]["timeout_sec"] == 200.0


def test_native_tasks_get_cpu_and_memory_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_staged_tasks: None
) -> None:
    import tomllib

    config = "schema_version: 1\nharbor:\n  task_source: native_harbor\ngrading:\n  mode: custom_only\n"
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch, config=config)
    _native_task(skill)
    result = _run_fake_gym(skill, tmp_path, skip_baseline=True, override_cpus=4, override_memory_mb=8192)
    staged = next((Path(result["run_dir"]) / "_harbor-tasks" / "codex" / "with").glob("*/task.toml"))
    environment = tomllib.loads(staged.read_text())["environment"]
    assert environment["cpus"] == 4 and environment["memory_mb"] == 8192


def test_results_root_is_excluded_from_the_repo_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    seen: list[tuple[Path, ...]] = []
    from skillevaluator.tier3.harbor import adapter

    original = adapter.generate_harbor_tasks

    def recording(*args, repo_context_exclude_paths=(), **kwargs):
        seen.append(tuple(repo_context_exclude_paths))
        return original(*args, repo_context_exclude_paths=repo_context_exclude_paths, **kwargs)

    monkeypatch.setattr(adapter, "generate_harbor_tasks", recording)
    del gym_runner
    _run_fake_gym(skill, tmp_path, skip_baseline=True, max_cases=1)
    assert seen and all(tmp_path / "results" in paths for paths in seen)


def test_failed_base_image_build_is_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.harbor import adapter

    skill, _env_log = _fake_gym_skill(
        tmp_path, monkeypatch, config="schema_version: 1\nharbor:\n  base_image_mode: rebuild\n"
    )
    monkeypatch.setattr(adapter, "build_eval_base_image", lambda *_args, **_kwargs: "")
    reporter = _RecordingReporter()
    _run_fake_gym(skill, tmp_path, skip_baseline=True, max_cases=1, progress_reporter=reporter)
    images = [event for event in reporter.events if event.stage == "docker-images" and event.state != "running"]
    assert [(event.state, event.detail) for event in images] == [
        ("degraded", "base image build failed; falling back to per-task Dockerfiles")
    ]


def test_gym_run_starts_the_reporter_with_its_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill, _env_log = _fake_gym_skill(tmp_path, monkeypatch)
    reporter = _RecordingReporter()
    result = _run_fake_gym(skill, tmp_path, progress_reporter=reporter)
    assert len(reporter.plans) == 1
    plan = reporter.plans[0]
    assert plan.output_dir == result["run_dir"] and plan.case_count == 2 and plan.matrix_trials == 4


def test_live_marks_score_reward_logs_and_skip_normalized_gym_jobs(tmp_path: Path) -> None:
    from skillevaluator.tier3.case_summary import live_case_marks

    run_dir = tmp_path / "run"
    arm = run_dir / "_nemo-gym" / "codex-with"
    arm.mkdir(parents=True)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text(json.dumps({"overall": 0.9}))
    txt_logs = tmp_path / "txt"
    txt_logs.mkdir()
    (txt_logs / "reward.txt").write_text("0\n")
    rows = [
        {"_ng_task_id": {"task_id": "calc-001"}, "verifier_return_code": 0, "verifier_logs_dir": str(logs)},
        {"_ng_task_id": {"task_id": "calc-002"}, "verifier_return_code": 0, "verifier_logs_dir": str(txt_logs)},
    ]
    (arm / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_job_from_rollouts(
        rows,
        run_dir / "_harbor-jobs" / "calc-codex-with",
        eval_name="codex__m",
        task_case_ids={"calc-001": "calc-001", "calc-002": "calc-002"},
        n_attempts=1,
    )
    marks, running = live_case_marks(run_dir, ["codex"])
    with_skill = marks["codex"]["with_skill"]
    assert len(with_skill["calc-001"]) == 1 and len(with_skill["calc-002"]) == 1
    from skillevaluator.tier3.case_summary import ERROR

    assert ERROR not in with_skill["calc-001"] + with_skill["calc-002"]
    assert with_skill["calc-001"] != with_skill["calc-002"]
    assert running == set()


def test_adc_refresh_follows_the_policy_key_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator import provider_config
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    monkeypatch.setattr(provider_config, "refresh_host_vertex_adc_environment", lambda *_args, **_kwargs: "token-c")
    base = {"OPENAI_API_KEY": "token-b", MODEL_API_KEY_ENV: "token-a"}
    refreshed, _timeout = gym_runner._refresh_adc_credentials(base, policy_uses_provider_key=True, timeout_s=60)
    assert refreshed[MODEL_API_KEY_ENV] == "token-c" and refreshed["OPENAI_API_KEY"] == "token-c"
    refreshed, _timeout = gym_runner._refresh_adc_credentials(base, policy_uses_provider_key=False, timeout_s=60)
    assert refreshed[MODEL_API_KEY_ENV] == "token-a"


def test_second_interrupt_during_stop_still_kills_the_gym_group(tmp_path: Path) -> None:
    import sys
    import time

    from skillevaluator.tier3.nemo_gym.runner import _run_gym

    class _InterruptingReporter(_RecordingReporter):
        def emit(self, event) -> None:
            del event
            raise KeyboardInterrupt

    pid_file = tmp_path / "leader.pid"
    script = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    with (tmp_path / "gym.log").open("w") as log, pytest.raises(KeyboardInterrupt):
        _run_gym(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=dict(os.environ),
            log=log,
            timeout_s=1,
            reporter=_InterruptingReporter(),
        )
    leader = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(leader, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("Gym leader survived a second interrupt")


def test_eperm_race_while_stopping_is_treated_as_exited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from skillevaluator.tier3.harbor import runner as harbor_runner
    from skillevaluator.tier3.nemo_gym.runner import _run_gym

    def racing_terminate(process, **_kwargs) -> None:
        process.kill()
        process.wait()
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(harbor_runner, "_terminate_harbor_process_tree", racing_terminate)
    with (tmp_path / "gym.log").open("w") as log:
        code = _run_gym(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            env=dict(os.environ),
            log=log,
            timeout_s=0.5,
        )
    assert code is None


def test_doctor_fails_without_a_policy_endpoint_or_docker_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3 import commands
    from skillevaluator.tier3.nemo_gym import backend
    from skillevaluator.tier3.nemo_gym import runner as gym_runner

    gym = tmp_path / "gym"
    import sys

    gym.write_text(_FAKE_GYM.format(python=sys.executable))
    gym.chmod(0o755)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(gym))
    monkeypatch.setattr(backend, "nemo_gym_version", lambda _bin: "0.7.0")

    def no_endpoint(_model):
        raise ValueError("--backend nemo-gym needs a model, an OpenAI-compatible base URL and an API key")

    monkeypatch.setattr(gym_runner, "_model_endpoint", no_endpoint)
    docker_daemon_error = commands._docker_daemon_error
    monkeypatch.setattr(commands, "_docker_daemon_error", lambda _env: None)
    rows = commands._nemo_gym_doctor_rows(["codex"], "docker", {})
    assert (
        "NeMo Gym backend",
        "fail",
        "codex: --backend nemo-gym needs a model, an OpenAI-compatible base URL and an API key",
    ) in rows

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\necho 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock' >&2\nexit 1\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    error = docker_daemon_error({"PATH": os.environ["PATH"]})
    assert error is not None and "Cannot connect to the Docker daemon" in error


def test_unsupported_gym_sandbox_is_rejected_before_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3.nemo_gym.backend import nemo_gym_settings_errors

    skill, env_log = _fake_gym_skill(tmp_path, monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_SANDBOX", "local")
    assert any("SKILL_EVAL_NEMO_GYM_SANDBOX=local" in error for error in nemo_gym_settings_errors())
    result = _run_fake_gym(skill, tmp_path)
    assert "SKILL_EVAL_NEMO_GYM_SANDBOX=local" in json.dumps(result)
    assert not env_log.exists()


def test_excess_rollouts_are_reported_and_not_collected(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text(json.dumps({"overall": 1.0}))
    row = {"_ng_task_id": {"task_id": "t1"}, "verifier_return_code": 0, "verifier_logs_dir": str(logs)}
    job = tmp_path / "job"
    errors = write_job_from_rollouts(
        [row, dict(row)], job, eval_name="codex__m", task_case_ids={"t1": "case-1"}, n_attempts=1
    )
    assert errors == ["NeMo Gym returned 2/1 rollouts for case case-1; the 1 extra rollout(s) were not collected"]
    assert sorted(path.name for path in job.iterdir() if path.is_dir()) == ["t1__gym001"]
