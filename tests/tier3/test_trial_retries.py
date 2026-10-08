# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Harbor trial retries (``--trial-retries``).

Harbor 0.24 retries a trial inside its own process: ``TrialQueue`` runs the same
``TrialConfig`` again, under the same trial name and directory, after deleting
the failed attempt's directory, and only when the attempt's exception name is in
``--retry-include`` and not in Harbor's default exclusions. The job then counts
the failed attempt only in ``stats.n_retries``. These tests pin that behavior
against the installed Harbor and check what SkillEvaluator builds, collects, and
records around it.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from click.testing import CliRunner
from rich.console import Console

pytest.importorskip("harbor")

from harbor.agents.installed.base import (
    AgentAuthenticationError,
    ApiError,
    ApiRateLimitError,
    NetworkConnectionError,
    NonZeroAgentExitCodeError,
    UnknownApiError,
)
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import RetryConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.result import ExceptionInfo, TrialResult
from harbor.trial.errors import (
    AgentSetupTimeoutError,
    AgentTimeoutError,
    EnvironmentStartTimeoutError,
    VerifierTimeoutError,
)
from harbor.trial.queue import TrialQueue

from skillevaluator import cli as cli_module
from skillevaluator.evaluation import EvaluationOptions, EvaluationService
from skillevaluator.models.result import ValidationResult
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3 import commands as tier3_commands
from skillevaluator.tier3.evals_config import EvalsConfigError, _validate_config
from skillevaluator.tier3.harbor import LOCAL_AGENT_IMPORT_PATHS, progress, runner, runtime_preflight
from skillevaluator.tier3.harbor.collector import (
    _agent_runtime_failure_reason,
    collect_harbor_results,
    harbor_job_retries,
    validate_harbor_job_result,
)
from skillevaluator.tier3.harbor.local_agents import (
    NVIDIA_BUILD_AGENT_IMPORT_PATHS,
    NVIDIA_BUILD_LOCAL_AGENT_IMPORT_PATHS,
    AgentSetupNetworkError,
    SetupNetworkErrorAgent,
    SkillEvaluatorCodex,
    SkillEvaluatorLocalOpenCode,
)
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.harbor.runner import build_harbor_run_command
from skillevaluator.tier3.harbor.runtime_preflight import validate_harbor_agent_only_job_result
from skillevaluator.tier3.plugin_native import NATIVE_AGENT_IMPORT_PATHS
from skillevaluator.tier3_environments import HARBOR_TRIAL_RETRY_EXCEPTIONS, MAX_TRIAL_RETRIES

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "skills" / "simple"
_NOW = datetime(2026, 10, 8, tzinfo=UTC)
_RETRY_FLAGS = {"--max-retries", "-r", "--retry-include", "--retry-exclude"}

# Harbor 0.24 exceptions for the agent's own task failures and the verifier's.
# None of them may ever be retried. NetworkConnectionError is one: Harbor raises
# it from the agent's task commands as well as from its install commands.
_TASK_PHASE_EXCEPTIONS = (
    AgentTimeoutError,
    VerifierTimeoutError,
    NonZeroAgentExitCodeError,
    NetworkConnectionError,
    ApiError,
    ApiRateLimitError,
    UnknownApiError,
    AgentAuthenticationError,
)
_TASK_PHASE_EXCEPTION_NAMES = (
    "RewardFileNotFoundError",
    "RewardFileEmptyError",
    "VerifierOutputParseError",
    "ApiUsageLimitError",
    "AgentSafetyRefusalError",
    "ModelNotFoundError",
    "CancelledError",
    "RuntimeError",
)


def _command(**kwargs: object) -> list[str]:
    return build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="retry-test",
        env_mode="docker",
        **kwargs,
    )


def test_default_trial_retries_add_no_flag_to_the_harbor_command() -> None:
    command = _command()

    assert command == _command(trial_retries=0)
    assert not _RETRY_FLAGS & set(command)


def test_trial_retries_pass_harbor_the_budget_and_only_the_infrastructure_exceptions() -> None:
    command = _command(trial_retries=2)

    retry_args = ["--max-retries", "2"]
    for name in ("EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "AgentSetupNetworkError"):
        retry_args += ["--retry-include", name]
    assert "NetworkConnectionError" not in command
    start = command.index("--max-retries")
    assert command[start : start + len(retry_args)] == retry_args
    # Nothing else changes, and Harbor's default exclusions stay in force because
    # SkillEvaluator never passes --retry-exclude, which would replace them.
    assert command[:start] + command[start + len(retry_args) :] == _command()


@pytest.mark.parametrize("value", [-1, MAX_TRIAL_RETRIES + 1, True, 1.5, "2", None])
def test_trial_retries_outside_the_supported_range_are_rejected(value: object) -> None:
    with pytest.raises(ValueError, match=f"trial_retries must be an integer from 0 to {MAX_TRIAL_RETRIES}"):
        _command(trial_retries=value)


def test_retry_include_names_are_the_names_harbor_records() -> None:
    raised = (
        EnvironmentStartTimeoutError("environment start timed out"),
        AgentSetupTimeoutError("agent setup timed out"),
        AgentSetupNetworkError("Agent setup failed with a network error: Command failed (exit 6): curl"),
    )

    # Harbor's retry check compares ExceptionInfo.exception_type, which is type(exc).__name__.
    assert tuple(ExceptionInfo.from_exception(exc).exception_type for exc in raised) == HARBOR_TRIAL_RETRY_EXCEPTIONS


def test_harbor_never_retries_a_task_or_verifier_failure_under_the_include_list() -> None:
    queue = TrialQueue(
        n_concurrent=1,
        retry_config=RetryConfig(max_retries=MAX_TRIAL_RETRIES, include_exceptions=set(HARBOR_TRIAL_RETRY_EXCEPTIONS)),
    )
    excluded = [cls.__name__ for cls in _TASK_PHASE_EXCEPTIONS] + list(_TASK_PHASE_EXCEPTION_NAMES)

    assert all(queue._should_retry_exception(name) for name in HARBOR_TRIAL_RETRY_EXCEPTIONS)
    assert not any(queue._should_retry_exception(name) for name in excluded)
    assert not set(excluded) & set(HARBOR_TRIAL_RETRY_EXCEPTIONS)
    # No default exclusion cancels an included name, and a parent class never matches its subclass.
    assert not set(HARBOR_TRIAL_RETRY_EXCEPTIONS) & RetryConfig().exclude_exceptions
    assert issubclass(NetworkConnectionError, NonZeroAgentExitCodeError)
    assert not queue._should_retry_exception(NonZeroAgentExitCodeError.__name__)
    assert not issubclass(AgentSetupNetworkError, NonZeroAgentExitCodeError)


class _NetworkFailingEnvironment:
    """Agent environment whose every command, after Harbor's setup mkdir, fails with curl's DNS error."""

    default_user = None

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, **_kwargs: object) -> SimpleNamespace:
        self.commands.append(command)
        if command.startswith("[ -d /installed-agent ]"):
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        return SimpleNamespace(return_code=6, stdout="", stderr="curl: (6) Could not resolve host: mirror.example.test")

    async def upload_file(self, *_args: object, **_kwargs: object) -> None:
        pass


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorLocalOpenCode])
def test_a_wrapper_raises_a_setup_network_failure_under_the_retried_name(tmp_path: Path, agent_class: type) -> None:
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(AgentSetupNetworkError, match="Could not resolve host") as raised:
        asyncio.run(agent.setup(_NetworkFailingEnvironment()))

    assert type(raised.value.__cause__) is NetworkConnectionError
    assert ExceptionInfo.from_exception(raised.value).exception_type in HARBOR_TRIAL_RETRY_EXCEPTIONS


@pytest.mark.parametrize("agent_class", [SkillEvaluatorCodex, SkillEvaluatorLocalOpenCode])
def test_a_wrapper_keeps_a_task_network_failure_unretried(tmp_path: Path, agent_class: type) -> None:
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")
    environment = _NetworkFailingEnvironment()

    with pytest.raises(NetworkConnectionError) as raised:
        asyncio.run(agent.run("Solve the task.", environment, AgentContext()))

    assert environment.commands
    assert type(raised.value) is NetworkConnectionError
    assert ExceptionInfo.from_exception(raised.value).exception_type not in HARBOR_TRIAL_RETRY_EXCEPTIONS


def _skillevaluator_agent_import_paths() -> set[str]:
    """Every SkillEvaluator Harbor agent wrapper a Tier 3 run can launch."""
    return {
        *runner.HARBOR_AGENT_IMPORT_PATHS.values(),
        *LOCAL_AGENT_IMPORT_PATHS.values(),
        *NVIDIA_BUILD_AGENT_IMPORT_PATHS.values(),
        *NVIDIA_BUILD_LOCAL_AGENT_IMPORT_PATHS.values(),
        *NATIVE_AGENT_IMPORT_PATHS.values(),
        *(base for _agent, base in NATIVE_AGENT_IMPORT_PATHS if base is not None),
    }


def test_every_skillevaluator_agent_wrapper_renames_a_setup_network_failure() -> None:
    paths = _skillevaluator_agent_import_paths()
    assert "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorGatewayOpenCode" in paths

    for path in sorted(paths):
        module_name, _, class_name = path.partition(":")
        agent_class = getattr(importlib.import_module(module_name), class_name)
        assert issubclass(agent_class, SetupNetworkErrorAgent), path
        # Nothing between the wrapper and the mixin replaces the setup that renames the failure.
        assert agent_class.setup is SetupNetworkErrorAgent.setup, path


def test_a_setup_network_failure_left_after_the_retries_is_an_agent_runtime_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "case-001__Ab3dE5f"
    trial_dir.mkdir()
    failure = AgentSetupNetworkError("Agent setup failed with a network error: Command failed (exit 6): curl")
    result = _trial_result(trial_dir, exception=failure)
    (trial_dir / "result.json").write_text(result.model_dump_json(indent=4), encoding="utf-8")

    assert _agent_runtime_failure_reason(trial_dir).startswith("AgentSetupNetworkError: Agent setup failed")


def _trial_result(
    trial_dir: Path,
    *,
    task_path: Path | None = None,
    exception: BaseException | None = None,
    verified: bool = True,
) -> TrialResult:
    task = str(task_path or trial_dir.parent / "tasks" / "case-001")
    return TrialResult.model_validate(
        {
            "id": uuid4(),
            "task_name": "nvidia/case-001",
            "trial_name": trial_dir.name,
            "trial_uri": trial_dir.as_uri(),
            "task_id": {"path": task},
            "task_checksum": "trial-retry-fixture",
            "config": {"task": {"path": task}, "trial_name": trial_dir.name, "trials_dir": str(trial_dir.parent)},
            "agent_info": {"name": "opencode", "version": "test", "model_info": {"name": "test-model"}},
            "agent_result": None if exception is not None else {"n_input_tokens": 10, "n_output_tokens": 2},
            "verifier_result": {"rewards": {"reward": 0.9}} if exception is None and verified else None,
            "exception_info": None if exception is None else ExceptionInfo.from_exception(exception),
            "started_at": _NOW,
            "finished_at": _NOW,
        }
    )


def _reward(entry_id: str, score: float) -> dict[str, object]:
    return {
        "entry_id": entry_id,
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, score),
        "overall": score,
    }


class _ScriptedTrial:
    """Stand-in for ``harbor.trial.trial.Trial`` whose attempts fail as scripted.

    Each attempt writes what a real attempt leaves in its trial directory: its
    agent setup log, then ``exception.txt`` when it fails or its verifier reward
    when it completes, and last ``result.json``, as Harbor's finalizer does.
    """

    def __init__(self, trial_dir: Path, failures: list[BaseException], attempts: list[int], *, verified: bool) -> None:
        self.paths = SimpleNamespace(trial_dir=trial_dir)
        self.config = SimpleNamespace(agent=SimpleNamespace(n_concurrent=None))
        self._failures = failures
        self._attempts = attempts
        self._verified = verified

    def add_hook(self, _event: object, _hook: object) -> None:
        pass

    async def run(self) -> TrialResult:
        self._attempts.append(len(self._attempts) + 1)
        trial_dir = self.paths.trial_dir
        (trial_dir / "agent" / "setup").mkdir(parents=True, exist_ok=True)
        (trial_dir / "agent" / "setup" / "stdout.txt").write_text(f"attempt {self._attempts[-1]}\n", encoding="utf-8")
        failure = self._failures.pop(0) if self._failures else None
        if failure is not None:
            (trial_dir / "exception.txt").write_text(f"{type(failure).__name__}: {failure}\n", encoding="utf-8")
        elif self._verified:
            entry_id = trial_dir.name.split("__")[0]
            (trial_dir / "verifier").mkdir(exist_ok=True)
            (trial_dir / "verifier" / "reward.json").write_text(json.dumps(_reward(entry_id, 0.9)), encoding="utf-8")
        result = _trial_result(trial_dir, exception=failure, verified=self._verified)
        (trial_dir / "result.json").write_text(result.model_dump_json(indent=4), encoding="utf-8")
        return result


def _run_scripted_trial(
    monkeypatch: pytest.MonkeyPatch,
    trial_dir: Path,
    failures: list[BaseException],
    *,
    verified: bool = True,
) -> tuple[TrialResult, list[int]]:
    """Run one trial through Harbor's own retry loop with the include list SkillEvaluator passes."""
    from harbor.trial import trial as harbor_trial

    attempts: list[int] = []

    async def create(_config: object) -> _ScriptedTrial:
        return _ScriptedTrial(trial_dir, failures, attempts, verified=verified)

    monkeypatch.setattr(harbor_trial.Trial, "create", create)
    queue = TrialQueue(
        n_concurrent=1,
        retry_config=RetryConfig(
            max_retries=2,
            include_exceptions=set(HARBOR_TRIAL_RETRY_EXCEPTIONS),
            min_wait_sec=0.0,
            max_wait_sec=0.0,
        ),
    )
    trial_config = SimpleNamespace(trial_name=trial_dir.name, agent=SimpleNamespace(n_concurrent=None))
    result = asyncio.run(queue._execute_trial_with_retries(trial_config))
    return result, attempts


def _write_job_result(job_dir: Path, trial_results: list[TrialResult], *, n_retries: int) -> None:
    """Write the job ``result.json`` exactly as Harbor's ``Job.run`` does once its trials finish."""
    stats = JobStats.from_trial_results(trial_results, n_total_trials=len(trial_results), n_retries=n_retries)
    job_result = JobResult(
        id=uuid4(),
        started_at=_NOW,
        updated_at=_NOW,
        finished_at=_NOW,
        n_total_trials=len(trial_results),
        stats=stats,
        trial_results=trial_results,
    )
    (job_dir / "result.json").write_text(
        job_result.model_dump_json(indent=4, exclude={"trial_results"}),
        encoding="utf-8",
    )


def test_harbor_retries_a_setup_failure_in_the_same_directory_and_keeps_only_the_last_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trial_dir = tmp_path / "jobs" / "demo-opencode-with" / "case-001__Ab3dE5f"
    trial_dir.mkdir(parents=True)

    result, attempts = _run_scripted_trial(
        monkeypatch,
        trial_dir,
        [AgentSetupNetworkError("Agent setup failed with a network error: Command failed (exit 6): curl")],
    )

    assert attempts == [1, 2]
    assert result.exception_info is None
    # Harbor deleted the failed attempt's directory: no superseded attempt is left to filter.
    assert sorted(path.name for path in trial_dir.parent.iterdir()) == [trial_dir.name]
    assert not (trial_dir / "exception.txt").exists()
    assert (trial_dir / "agent" / "setup" / "stdout.txt").read_text(encoding="utf-8") == "attempt 2\n"
    persisted = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    assert persisted["exception_info"] is None


@pytest.mark.parametrize(
    "failure",
    [
        AgentTimeoutError("agent timed out"),
        VerifierTimeoutError("verifier timed out"),
        NetworkConnectionError("Command failed (exit 1): opencode run"),
    ],
)
def test_harbor_keeps_a_task_phase_failure_without_retrying_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: BaseException
) -> None:
    trial_dir = tmp_path / "jobs" / "demo-opencode-with" / "case-001__Ab3dE5f"
    trial_dir.mkdir(parents=True)

    result, attempts = _run_scripted_trial(monkeypatch, trial_dir, [failure])

    assert attempts == [1]
    assert result.exception_info is not None
    assert result.exception_info.exception_type == type(failure).__name__
    assert (trial_dir / "exception.txt").is_file()


def test_a_retried_trial_is_collected_once_as_a_complete_scored_trial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job_dir = tmp_path / "jobs" / "demo-opencode-with"
    retried_dir = job_dir / "case-001__Ab3dE5f"
    retried_dir.mkdir(parents=True)
    retried, _ = _run_scripted_trial(
        monkeypatch,
        retried_dir,
        [AgentSetupTimeoutError("Agent setup timed out after 360.0 seconds")],
    )
    steady_dir = job_dir / "case-002__Gh7jK9m"
    steady_dir.mkdir()
    steady, _ = _run_scripted_trial(monkeypatch, steady_dir, [])
    _write_job_result(job_dir, [retried, steady], n_retries=1)

    assert validate_harbor_job_result(job_dir / "result.json", expected_trials=2) == (True, "")
    assert harbor_job_retries(job_dir) == 1

    results = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
        expected_cases=2,
        expected_case_ids=["case-001", "case-002"],
        expected_trials=2,
    )

    agent = results["agents"]["opencode"]
    with_skill = agent["conditions"]["with_skill"]
    assert results["execution_status"] == "succeeded"
    assert (with_skill["execution_status"], with_skill["scored_attempts"]) == ("succeeded", 2)
    assert not agent["trial_failures"]["with_skill"]
    trials_dir = tmp_path / "results" / "opencode" / "with-skill" / "trials"
    assert sorted(path.name for path in trials_dir.iterdir()) == ["case-001__Ab3dE5f", "case-002__Gh7jK9m"]


def test_a_retried_agent_only_preflight_trial_is_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / runtime_preflight.preflight_job_name("opencode")
    trial_dir = job_dir / "case-001__Ab3dE5f"
    trial_dir.mkdir(parents=True)
    # Verification is disabled in the preflight, so its trial carries an agent result and no reward.
    retried, attempts = _run_scripted_trial(
        monkeypatch,
        trial_dir,
        [EnvironmentStartTimeoutError("Environment start timed out after 600.0 seconds")],
        verified=False,
    )
    _write_job_result(job_dir, [retried], n_retries=len(attempts) - 1)

    assert validate_harbor_agent_only_job_result(job_dir / "result.json", expected_trials=1) == (True, "")
    assert harbor_job_retries(job_dir) == 1


@pytest.mark.parametrize("stats", [{}, {"n_retries": -1}, {"n_retries": True}, {"n_retries": "2"}])
def test_an_absent_or_invalid_retry_counter_reads_as_no_retries(tmp_path: Path, stats: dict[str, object]) -> None:
    (tmp_path / "result.json").write_text(json.dumps({"n_total_trials": 1, "stats": stats}), encoding="utf-8")

    assert harbor_job_retries(tmp_path) == 0
    assert harbor_job_retries(tmp_path / "missing") == 0


def test_runtime_preflight_passes_trial_retries_and_gives_each_attempt_its_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "tasks"
    (dataset / "case-001").mkdir(parents=True)
    (dataset / "case-001" / "task.toml").write_text('[task]\nname = "nvidia/case-001"\n', encoding="utf-8")
    captured: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime_preflight.subprocess, "run", run)
    monkeypatch.setattr(runtime_preflight, "validate_harbor_agent_only_job_result", lambda *_a, **_k: (True, ""))

    result = runtime_preflight.run_agent_runtime_preflight(
        dataset=dataset,
        agent="opencode",
        model="nvidia/model",
        env_mode="docker",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        timeout_seconds=300,
        trial_retries=2,
    )

    assert result.ok is True
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--max-retries") + 1] == "2"
    assert command.count("--retry-include") == len(HARBOR_TRIAL_RETRY_EXCEPTIONS)
    assert captured["timeout"] == 900


def test_run_harbor_launches_harbor_with_the_retry_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_process(command: list[str], **_kwargs: object) -> runner._BoundedHarborProcessResult:
        commands.append(command)
        return runner._BoundedHarborProcessResult(returncode=0, output_tail="", output_exceeded=False)

    monkeypatch.setattr(runner, "_run_bounded_harbor_process", fake_process)
    monkeypatch.setattr(runner, "_validate_harbor_job_result", lambda *_a, **_k: (True, ""))
    run_options = {
        "dataset": tmp_path / "dataset",
        "agent": "opencode",
        "job_name": "demo-opencode-with",
        "env_mode": "docker",
        "model": "nvidia/model",
        "jobs_dir": tmp_path / "jobs",
        "run_env": {},
        "n_attempts": 1,
        "n_concurrent": 1,
        "timeout_multiplier": 1.0,
        "override_cpus": None,
        "override_memory_mb": None,
        "override_storage_mb": None,
    }

    assert runner._run_harbor(**run_options) == (True, "")
    assert runner._run_harbor(**run_options, trial_retries=3) == (True, "")

    default_command, retry_command = commands
    assert not _RETRY_FLAGS & set(default_command)
    assert retry_command[retry_command.index("--max-retries") + 1] == "3"


def test_tier3_evaluate_forwards_trial_retries_and_leaves_the_default_to_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[EvaluationOptions] = []

    def evaluate(_self: object, options: EvaluationOptions, **_kwargs: object) -> dict[str, object]:
        captured.append(options)
        return {"execution_status": "succeeded", "execution_errors": []}

    monkeypatch.setattr(EvaluationService, "evaluate", evaluate)
    base = ["tier3", "evaluate", str(FIXTURE), "-a", "codex", "--progress", "off"]

    for argv in (base, [*base, "--trial-retries", "2"]):
        outcome = CliRunner().invoke(cli_module.cli, argv)
        assert outcome.exit_code == 0, outcome.output

    assert [options.trial_retries for options in captured] == [None, 2]
    rejected = CliRunner().invoke(cli_module.cli, [*base, "--trial-retries", str(MAX_TRIAL_RETRIES + 1)])
    assert rejected.exit_code == 2
    assert "--trial-retries" in rejected.output


def test_validate_forwards_trial_retries_and_catalog_children_add_the_flag_only_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def tier1(*_args: object, **_kwargs: object) -> list[ValidationResult]:
        result = ValidationResult(validator_name="SCHEMA")
        result.add_success("schema", "ok")
        return [result]

    def tier3(*_args: object, **kwargs: object) -> ValidationResult:
        captured.update(kwargs)
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "ok")
        return result

    monkeypatch.setattr(cli_module, "run_validation", tier1)
    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", tier3)
    argv = ["validate", "--no-autopilot", str(FIXTURE), "--no-llm", "--no-tier2", "--tier3", "--checks", "schema"]

    outcome = CliRunner().invoke(cli_module.cli, [*argv, "--trial-retries", "1"])

    assert outcome.exit_code == 0, outcome.output
    assert captured["trial_retries"] == 1

    def child_argv(trial_retries: int | None) -> list[str]:
        context = SimpleNamespace(params={"env_mode": "docker", "trial_retries": trial_retries})
        return cli_module._catalog_child_argv_from_ctx(context, FIXTURE, tmp_path / "out")

    assert "--trial-retries" not in child_argv(None)
    # An explicit 0 still overrides harbor.trial_retries in the child skill's config.
    for value in (0, 2):
        argv = child_argv(value)
        assert argv[argv.index("--trial-retries") + 1] == str(value)


def test_validate_helper_and_plugin_evaluation_carry_trial_retries_into_the_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[EvaluationOptions] = []
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: captured.append(options) or {},
    )

    cli_module._run_agent_eval_or_skip(
        FIXTURE,
        agents="codex",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=None,
        max_agents=None,
        validate_source=False,
        trial_retries=3,
    )

    package_path = tmp_path / "package"
    package_path.mkdir()
    prepared = SimpleNamespace(
        skipped=False,
        skip_reason=None,
        package_path=package_path,
        include_skills=(),
        unresolved_skill_refs=(),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=(),
        native_source=None,
        mcp_probe_targets=(),
        integration_evidence_error=lambda: None,
        provenance=lambda: {"plugin_name": "plugin", "partial": False},
    )
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: prepared)
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.tier3.result_display.render_evaluation_result", lambda *_a, **_k: None)
    plugin = tmp_path / "plugin"
    plugin.mkdir()

    outcome = CliRunner().invoke(
        cli_module.cli,
        ["tier3", "evaluate-plugin", str(plugin), "--trial-retries", "4", "--progress", "off"],
    )

    assert outcome.exit_code == 0, outcome.output
    assert [options.trial_retries for options in captured] == [3, 4]


def test_engine_entry_point_forwards_trial_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    provider = ProviderConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4.1-mini",
    )
    monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(tier3_commands, "resolve_results_root", lambda *_args: tmp_path / "results")
    monkeypatch.setattr(tier3_commands, "run_harbor_eval", lambda **kwargs: captured.update(kwargs) or {})

    EvaluationService().evaluate(EvaluationOptions(skill_path=FIXTURE, agents="codex", trial_retries=2))

    assert captured["trial_retries"] == 2


def _native_skill(tmp_path: Path, *, verifier: str = "", harbor: str = "") -> Path:
    skill_dir = tmp_path / "target-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: target-skill\ndescription: demo\n---\n# Body\n", encoding="utf-8")
    (skill_dir / "evals").mkdir()
    (skill_dir / "evals" / "config.yaml").write_text(
        f"schema_version: 1\nharbor:\n  task_source: native_harbor\n{harbor}grading:\n  mode: custom_only\n",
        encoding="utf-8",
    )
    for folder in ("case-001", "case-002"):
        task_dir = skill_dir / "evals" / "harbor" / folder
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Solve the task.\n", encoding="utf-8")
        (task_dir / "task.toml").write_text(
            f'schema_version = "1.3"\n\n[metadata]\nentry_id = "{folder}"\n\n'
            f'[task]\nname = "nvidia/{folder}"\n\n[environment]\n{verifier}',
            encoding="utf-8",
        )
        (task_dir / "tests").mkdir()
        (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    return skill_dir


def _offline_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ProviderConfig(
        provider="nv_build",
        model="nvidia/nemotron-3-nano-30b-a3b",
        api_key="nvapi-test",
        base_url="https://integrate.api.nvidia.com/v1",
        litellm_model="nvidia_nim/nvidia/nemotron-3-nano-30b-a3b",
    )
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected: runtime_preflight.ModelProbeResult(True, selected.provider, selected.model, "ok"),
    )


@pytest.mark.parametrize(
    ("stop_on_pass", "n_attempts", "jobs"),
    [
        (False, 1, 2),
        # One job per case and arm; each case passes on its first attempt. The merged
        # arm jobs sum their attempts' retries, which must not be counted twice.
        (True, 2, 4),
    ],
)
def test_run_records_the_configured_retries_and_the_retries_harbor_performed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stop_on_pass: bool, n_attempts: int, jobs: int
) -> None:
    skill_dir = _native_skill(tmp_path)
    _offline_engine(monkeypatch)
    launches: list[object] = []

    def fake_run_harbor(**kwargs: object) -> tuple[bool, str]:
        launches.append(kwargs.get("trial_retries"))
        dataset = Path(str(kwargs["dataset"]))
        job_dir = Path(str(kwargs["jobs_dir"])) / str(kwargs["job_name"])
        trial_results = []
        for folder in kwargs.get("include_task_names") or ("case-001", "case-002"):
            trial_dir = job_dir / f"{folder}__attempt1"
            trial_dir.mkdir(parents=True)
            trial_result = _trial_result(trial_dir, task_path=dataset / folder)
            (trial_dir / "result.json").write_text(trial_result.model_dump_json(indent=2), encoding="utf-8")
            trial_results.append(trial_result)
        # One trial of every job needed a retry after an infrastructure error.
        _write_job_result(job_dir, trial_results, n_retries=1)
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_run_harbor)

    results = runner.run_harbor_eval(
        skill_path=skill_dir,
        agents=["opencode"],
        output_dir=tmp_path / "eval-out",
        env_mode="docker",
        n_attempts=n_attempts,
        stop_on_pass=stop_on_pass,
        agent_runtime_preflight=False,
        trial_retries=2,
    )

    assert results["execution_status"] == "succeeded", results.get("execution_errors")
    assert launches == [2] * jobs
    harbor = results["run_config"]["harbor"]
    assert (harbor["trial_retries"], harbor["trial_retries_used"]) == (2, jobs)
    persisted = json.loads((Path(results["run_dir"]) / "run_config.json").read_text(encoding="utf-8"))
    assert (persisted["harbor"]["trial_retries"], persisted["harbor"]["trial_retries_used"]) == (2, jobs)


def test_trial_retries_refuse_a_single_step_task_that_verifies_in_a_separate_environment(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    (shared / "case-001").mkdir(parents=True)
    (shared / "case-001" / "task.toml").write_text('[task]\nname = "nvidia/case-001"\n', encoding="utf-8")
    multi_step = tmp_path / "multi-step"
    (multi_step / "case-001").mkdir(parents=True)
    # A multi-step task records a step's verifier failure on the step, which Harbor never retries.
    (multi_step / "case-001" / "task.toml").write_text(
        '[verifier]\nenvironment_mode = "separate"\n\n[[steps]]\nname = "one"\n', encoding="utf-8"
    )
    separate = tmp_path / "separate"
    (separate / "case-002").mkdir(parents=True)
    (separate / "case-002" / "task.toml").write_text("[verifier.environment]\ncpus = 1\n", encoding="utf-8")

    assert runner._separate_verifier_task([shared, multi_step]) is None
    assert runner._separate_verifier_task([shared, separate]) == separate / "case-002" / "task.toml"


def test_run_with_trial_retries_stops_before_harbor_for_a_separate_verifier_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    skill_dir = _native_skill(tmp_path, verifier='\n[verifier]\nenvironment_mode = "separate"\n')
    _offline_engine(monkeypatch)
    launches: list[object] = []
    monkeypatch.setattr(runner, "_run_harbor", lambda **kwargs: launches.append(kwargs) or (True, ""))

    results = runner.run_harbor_eval(
        skill_path=skill_dir,
        agents=["opencode"],
        output_dir=tmp_path / "eval-out",
        env_mode="docker",
        agent_runtime_preflight=False,
        trial_retries=1,
    )

    assert results["execution_status"] == "failed"
    [error] = results["execution_errors"]
    assert "trial_retries is not supported for a single-step task that verifies in a separate environment" in error
    assert launches == []


@pytest.mark.parametrize("value", [0, 1, MAX_TRIAL_RETRIES])
def test_evals_config_accepts_harbor_trial_retries(tmp_path: Path, value: int) -> None:
    config = _validate_config({"schema_version": 1, "harbor": {"trial_retries": value}}, tmp_path / "config.yml")

    assert config["harbor"]["trial_retries"] == value


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1, f"must be between 0 and {MAX_TRIAL_RETRIES}"),
        (MAX_TRIAL_RETRIES + 1, f"must be between 0 and {MAX_TRIAL_RETRIES}"),
        (True, "must be an integer"),
        (1.5, "must be an integer"),
        ("2", "must be an integer"),
    ],
)
def test_evals_config_rejects_an_invalid_harbor_trial_retries(tmp_path: Path, value: object, message: str) -> None:
    with pytest.raises(EvalsConfigError, match=f"harbor.trial_retries {message}"):
        _validate_config({"schema_version": 1, "harbor": {"trial_retries": value}}, tmp_path / "config.yml")


@pytest.mark.parametrize(("cli_value", "expected"), [(None, 3), (0, 0), (1, 1)])
def test_run_takes_trial_retries_from_the_config_unless_the_cli_sets_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cli_value: int | None, expected: int
) -> None:
    skill_dir = _native_skill(tmp_path, harbor="  trial_retries: 3\n")
    _offline_engine(monkeypatch)
    launches: list[object] = []
    plans: list[progress.Tier3RunPlan] = []

    class Reporter(progress.NullProgressReporter):
        def start(self, plan: progress.Tier3RunPlan) -> None:
            plans.append(plan)

    def fake_run_harbor(**kwargs: object) -> tuple[bool, str]:
        launches.append(kwargs.get("trial_retries"))
        return False, "stop after the first launch"

    monkeypatch.setattr(runner, "_run_harbor", fake_run_harbor)

    results = runner.run_harbor_eval(
        skill_path=skill_dir,
        agents=["opencode"],
        output_dir=tmp_path / "eval-out",
        env_mode="docker",
        agent_runtime_preflight=False,
        skip_baseline=True,
        trial_retries=cli_value,
        progress_reporter=Reporter(),
    )

    assert launches == [expected]
    assert results["run_config"]["harbor"]["trial_retries"] == expected
    # The resolved plans carry the effective value; the early plan before config loads carries the CLI value.
    assert [plan.trial_retries for plan in plans] == [cli_value, expected, expected]


def _banner_plan(trial_retries: int | None) -> progress.Tier3RunPlan:
    return progress.Tier3RunPlan(
        skill_name="demo",
        environment="docker",
        agents=("codex",),
        attempts=1,
        timeout_multiplier=1.0,
        trial_retries=trial_retries,
    )


def test_the_run_plan_shows_the_trial_retries_only_when_enabled() -> None:
    def plain(trial_retries: int | None) -> str:
        output = io.StringIO()
        reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)
        reporter.start(_banner_plan(trial_retries))
        reporter.close()
        return output.getvalue()

    def rich(trial_retries: int | None) -> str:
        reporter = progress.RichProgressReporter(stream=io.StringIO())
        reporter._live_plan = _banner_plan(trial_retries)
        output = io.StringIO()
        Console(file=output, force_terminal=False, width=200).print(reporter._build_live_table())
        return output.getvalue()

    assert "timeout=1x" in plain(2)
    assert "trial-retries=2" in plain(2)
    assert "Trial retries 2" in rich(2)
    for value in (None, 0):
        assert "trial-retries" not in plain(value)
        assert "Trial retries" not in rich(value)
