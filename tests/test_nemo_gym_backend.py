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
tasks = Path(args[2])
output = Path(args[args.index("--output") + 1])
repeats = int(next(a for a in args if a.startswith("+num_repeats=")).split("=", 1)[1])
Path(os.environ["FAKE_GYM_ENV_LOG"]).open("a").write(json.dumps({{"arm": tasks.name, "repeats": repeats, "env": sorted(os.environ)}}) + "\\n")
with output.open("w") as rows:
    for task in sorted(p for p in tasks.iterdir() if (p / "task.toml").is_file()):
        for attempt in range(repeats):
            logs = output.parent / "logs" / f"{{task.name}}-{{attempt}}"
            logs.mkdir(parents=True)
            (logs / "reward.json").write_text(json.dumps({{"overall": 0.75}}))
            (logs / "skill_evaluator_reward.json").write_text(json.dumps({{"overall": 0.75}}))
            row = {{"_ng_task_id": {{"taskset": tasks.name, "task_id": task.name}}, "verifier_return_code": 0,
                    "verifier_logs_dir": str(logs), "episode_id": {{"rollout_id": f"r{{attempt}}", "attempt": 0}}}}
            rows.write(json.dumps(row) + "\\n")
"""


def _fake_gym_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str = "") -> tuple[Path, Path]:
    import sys

    from skillevaluator import provider_config
    from skillevaluator.provider_config import ProviderConfig

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

    monkeypatch.setattr(gym_runner, "_GYM_HOST_ENV_VARS", gym_runner._GYM_HOST_ENV_VARS | {"FAKE_GYM_ENV_LOG"})
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


def test_fake_gym_run_honors_arm_case_limit_config_and_is_discoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_fake_gym_run_uses_config_agent_model_and_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    ["  pre_agent_setup:\n    - echo hi\n", "  resources:\n    storage_mb: 4096\n", "  task_source: native_harbor\n"],
)
def test_fake_gym_run_rejects_harbor_only_config_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor_config: str
) -> None:
    skill, env_log = _fake_gym_skill(tmp_path, monkeypatch, config=f"schema_version: 1\nharbor:\n{harbor_config}")
    result = _run_fake_gym(skill, tmp_path)
    assert "not supported with --backend nemo-gym" in result["error"][0]
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


def test_gym_run_publishes_latest_stages_runtime_env_and_records_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_config_errors_reject_decoys_and_timeout_multiplier(tmp_path: Path) -> None:
    from skillevaluator.tier3.nemo_gym.backend import nemo_gym_config_errors

    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  timeout_multiplier: 2\nskill_workspace:\n  decoy_manifest: decoys.json\n"
    )
    errors = " ".join(nemo_gym_config_errors(tmp_path))
    assert "timeout_multiplier" in errors and "decoy_manifest" in errors
