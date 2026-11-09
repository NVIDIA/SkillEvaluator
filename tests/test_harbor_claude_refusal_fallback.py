# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code's refusal model fallback stays off in every staged Tier 3 task.

When the model provider's safety classifier stops the selected model, Claude
Code 2.1.292 retries the turn on another model from a fixed table (Claude Opus
5.5 -> ``claude-opus-4-8``) under that bare Anthropic id. Behind a gateway that
serves only the selected model the retry failed with HTTP 403 and Harbor
recorded an ``UnknownApiError``; where it succeeded, one scored trial mixed two
models. With the fallback off, the CLI stops with ``model_refusal_no_fallback``
and Harbor 0.24 raises ``AgentSafetyRefusalError``.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID

import pytest
from click.testing import CliRunner
from harbor.agents.installed.base import (
    AgentSafetyRefusalError,
    ApiOverloadedError,
    NonZeroAgentExitCodeError,
    UnknownApiError,
)
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.job.result import JobResult, JobStats
from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from harbor.utils.env import is_sensitive_env_key
from rich.console import Console
from tests.claude_refusal_fixtures import (
    NO_FALLBACK_TEXT,
    SELECTED_MODEL,
    gateway_fallback_403_stream,
    init_event,
    line,
    max_turns_event,
    no_fallback_stream,
    result_event,
    tool_round,
)

from skillevaluator.cli import cli
from skillevaluator.evaluation import EvaluationService
from skillevaluator.provider_config import resolve_llm_provider
from skillevaluator.tier3.harbor import CONTAINER_AGENT_IMPORT_PATHS, LOCAL_AGENT_IMPORT_PATHS
from skillevaluator.tier3.harbor.adapter import (
    _write_task_toml,
    generate_harbor_tasks,
    stage_native_harbor_tasks,
)
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment
from skillevaluator.tier3.harbor.runner import (
    _CLAUDE_CODE_VERSION,
    _model_for_agent,
    _resolve_agent_runtime_plan,
    _resolve_runtime_env,
)
from skillevaluator.tier3.result_display import render_evaluation_result

_SWITCH = "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK"
# The collector's arm-level error when scored attempts fall short of expected ones.
_COVERAGE_ERROR = "Scored attempt coverage is"
# The switch's CHANGELOG entry, found by its opening words so that a later entry
# that names the variable does not change which entry the docs test reads.
_CHANGELOG_ENTRY_LEAD = "- A Tier 3 `claude-code` trial that the model provider's safety classifier stops"
# A released entry keeps the text it shipped with, so the later Documentation
# entry that corrects where that entry said the coverage error appears is the
# CHANGELOG text that must agree with the docs.
_CHANGELOG_CORRECTION_LEAD = "- The 0.8.0 entry for refused `claude-code` attempts said"
_SIMPLE_SKILL = Path(__file__).resolve().parent / "fixtures" / "skills" / "simple"
_FAKE_CREDENTIAL = "test-key"
_VERTEX_ENV = {
    "CLAUDE_CODE_USE_VERTEX": "1",
    "ANTHROPIC_VERTEX_PROJECT_ID": "test-project",
    "CLOUD_ML_REGION": "us-east5",
}
_ROUTE_ENV_PREFIXES = (
    "ANTHROPIC_",
    "AWS_",
    "CLAUDE_",
    "CLOUD_ML_",
    "CLOUDSDK_",
    "GCLOUD_",
    "GCP_",
    "GOOGLE_",
    "NVIDIA_",
    "OPENAI_",
    "SKILL_EVAL_",
)


def _clear_route_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(_ROUTE_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)


def _write_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "demo-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Demo Skill\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "prompt": "Use the skill."}]),
        encoding="utf-8",
    )
    return skill


def _staged_environment_env(task: Path) -> dict[str, str]:
    config = TaskConfig.model_validate_toml((task / "task.toml").read_text(encoding="utf-8"))
    return dict(config.environment.env)


# ---------------------------------------------------------------------------
# The fallback switch in the task environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_skill", [True, False])
def test_generated_tasks_disable_claude_code_refusal_fallback(tmp_path: Path, with_skill: bool) -> None:
    skill = _write_skill(tmp_path)

    [task] = generate_harbor_tasks(skill, tmp_path / "tasks", with_skill=with_skill)

    assert _staged_environment_env(task)[_SWITCH] == "1"


class _Route(NamedTuple):
    provider_env: dict[str, str]
    host_env: dict[str, str]
    # A name the route's own credential set stages, so the case proves it took that route.
    marker: str | None
    env_mode: str = "docker"
    agent_model: str | None = None
    environment_kwargs: dict[str, str] | None = None


_ROUTES = {
    "gateway": _Route(
        {
            "SKILL_EVAL_LLM_PROVIDER": "openai-compatible",
            "SKILL_EVAL_LLM_BASE_URL": "https://gateway.example/team/v1",
            "SKILL_EVAL_LLM_API_KEY": _FAKE_CREDENTIAL,
        },
        {},
        "ANTHROPIC_BASE_URL",
    ),
    "anthropic": _Route(
        {"SKILL_EVAL_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": _FAKE_CREDENTIAL},
        {"ANTHROPIC_API_KEY": _FAKE_CREDENTIAL},
        "ANTHROPIC_API_KEY",
    ),
    "bedrock": _Route(
        {"SKILL_EVAL_LLM_PROVIDER": "bedrock", "AWS_REGION": "us-east-1"},
        {
            "AWS_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": _FAKE_CREDENTIAL,
            "AWS_SECRET_ACCESS_KEY": _FAKE_CREDENTIAL,
        },
        "CLAUDE_CODE_USE_BEDROCK",
    ),
    "nv_build": _Route(
        {"SKILL_EVAL_LLM_PROVIDER": "nv_build", "NVIDIA_API_KEY": _FAKE_CREDENTIAL},
        {"NVIDIA_API_KEY": _FAKE_CREDENTIAL},
        None,
    ),
    # Claude Code on Vertex AI runs in GKE mode with Workload Identity and an
    # explicit Claude model; the evaluator provider can be any other one.
    "vertex": _Route(
        {"SKILL_EVAL_LLM_PROVIDER": "openai", "SKILL_EVAL_LLM_API_KEY": _FAKE_CREDENTIAL},
        _VERTEX_ENV,
        "CLAUDE_CODE_USE_VERTEX",
        env_mode="gke",
        agent_model=SELECTED_MODEL,
        environment_kwargs={"allow_workload_identity": "true"},
    ),
}


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_claude_code_refusal_fallback_is_off_for_every_provider_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: str
) -> None:
    _clear_route_env(monkeypatch)
    spec = _ROUTES[route]
    for name, value in spec.host_env.items():
        monkeypatch.setenv(name, value)
    provider = resolve_llm_provider(spec.provider_env)
    model, source = _model_for_agent("claude-code", cli_model=spec.agent_model, config_agents={}, provider=provider)
    plan = _resolve_agent_runtime_plan(
        provider=provider,
        agents=["claude-code"],
        models={"claude-code": model},
        model_sources={"claude-code": source},
        configured_runtime_env={},
        env_mode=spec.env_mode,
        environment_kwargs=spec.environment_kwargs,
    )["claude-code"]
    if spec.marker is not None:
        assert spec.marker in plan.staged_env

    [task] = generate_harbor_tasks(
        _write_skill(tmp_path), tmp_path / "tasks", with_skill=True, runtime_env=dict(plan.staged_env)
    )

    # The switch is evaluator-managed task environment, not a provider credential:
    # every route stages it, and no route's credential set carries or replaces it.
    assert _staged_environment_env(task)[_SWITCH] == "1"
    assert _SWITCH not in plan.staged_env
    assert _SWITCH not in plan.subprocess_env


def test_docs_scope_the_switch_to_the_pinned_claude_code_release() -> None:
    # The variable was checked with the pinned CLI only, and local mode runs the
    # host's CLI. A pin bump must re-check the variable and update the paragraph.
    page = Path(__file__).resolve().parents[1] / "docs" / "agents-and-sandboxes.mdx"
    [paragraph] = [
        " ".join(block.split()) for block in page.read_text(encoding="utf-8").split("\n\n") if f"{_SWITCH}=1" in block
    ]

    assert f"With the pinned Claude Code {_CLAUDE_CODE_VERSION}," in paragraph
    assert "[Local mode](#local-mode) runs the Claude Code already on your machine" in paragraph


def _refusal_doc_texts() -> dict[str, str]:
    """Return each place that documents the switch, whitespace-normalized."""
    root = Path(__file__).resolve().parents[1]
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    [release] = [section for section in changelog.split("\n## ")[1:] if _SWITCH in section]
    assert release.startswith("0.8.0 - "), release.splitlines()[0]
    entries = [block for block in release.split("\n\n") if " ".join(block.split()).startswith(_CHANGELOG_ENTRY_LEAD)]
    assert entries, "the 0.8.0 CHANGELOG entry for the refusal fallback switch is missing"
    corrections = [
        block for block in changelog.split("\n\n") if " ".join(block.split()).startswith(_CHANGELOG_CORRECTION_LEAD)
    ]
    assert len(corrections) == 1, "the CHANGELOG correction of the 0.8.0 refusal entry is missing or repeated"
    entry = corrections[0]
    blocks = (root / "docs" / "agents-and-sandboxes.mdx").read_text(encoding="utf-8").split("\n\n")
    [index] = [index for index, block in enumerate(blocks) if f"{_SWITCH}=1" in block]
    troubleshooting = (root / "docs" / "tier3-live-evaluation.mdx").read_text(encoding="utf-8")
    [accordion] = re.findall(
        r'<Accordion title="A claude-code trial fails with AgentSafetyRefusalError">(.*?)</Accordion>',
        troubleshooting,
        flags=re.DOTALL,
    )
    texts = {
        "CHANGELOG.md": entry,
        "agents-and-sandboxes.mdx": "\n\n".join(blocks[index : index + 2]),
        "tier3-live-evaluation.mdx": accordion,
    }
    return {name: " ".join(text.split()) for name, text in texts.items()}


def _write_harbor_job(job_dir: Path, case_ids: list[str], *, attempts: int, refused: str | None = None) -> None:
    """Persist a Harbor 0.24 job whose trials all score 1.0, except the ``refused`` attempt."""
    started = datetime(2026, 8, 25, tzinfo=UTC)
    trials: list[TrialResult] = []
    for case_id in case_ids:
        for attempt in range(1, attempts + 1):
            name = f"{case_id}__attempt{attempt}"
            trial_dir = job_dir / name
            trial_dir.mkdir(parents=True)
            at = started + timedelta(seconds=len(trials))
            task = {"path": str(job_dir / "task" / case_id)}
            refusal = {
                "exception_type": "AgentSafetyRefusalError",
                "exception_message": "Command failed (exit 1)",
                "exception_traceback": "",
                "occurred_at": at,
            }
            trial = TrialResult.model_validate(
                {
                    "id": UUID(int=len(trials) + 2),
                    "task_name": case_id,
                    "trial_name": name,
                    "trial_uri": trial_dir.as_uri(),
                    "task_id": task,
                    "task_checksum": "refusal-fixture",
                    "config": {"task": task, "trial_name": name},
                    "agent_info": {"name": "claude-code", "version": "test", "model_info": {"name": SELECTED_MODEL}},
                    "agent_result": {"n_input_tokens": 7, "n_cache_tokens": 2, "n_output_tokens": 3},
                    "started_at": at,
                    "finished_at": at,
                    "verifier_result": None if name == refused else {"rewards": {"overall": 1.0}},
                    "exception_info": refusal if name == refused else None,
                }
            )
            (trial_dir / "result.json").write_text(trial.model_dump_json(indent=2), encoding="utf-8")
            trials.append(trial)
    job = JobResult(
        id=UUID(int=1),
        started_at=started,
        updated_at=started,
        finished_at=started,
        n_total_trials=len(trials),
        stats=JobStats.from_trial_results(trials, n_total_trials=len(trials)),
        trial_results=trials,
    )
    (job_dir / "result.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")


def _collect_refusal(
    tmp_path: Path, *, cases: int, attempts: int, refused: str, skip_baseline: bool = True
) -> dict[str, Any]:
    """Collect a claude-code run in which the safety classifier stopped only ``refused``."""
    case_ids = [f"case-{index:03d}" for index in range(1, cases + 1)]
    jobs_dir = tmp_path / "jobs"
    _write_harbor_job(jobs_dir / "demo-claude-code-with", case_ids, attempts=attempts, refused=refused)
    if not skip_baseline:
        _write_harbor_job(jobs_dir / "demo-claude-code-without", case_ids, attempts=attempts)
    return collect_harbor_results(
        skill_name="demo",
        agents=["claude-code"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        skip_baseline=skip_baseline,
        n_attempts=attempts,
        expected_cases=cases,
        expected_case_ids=case_ids,
        expected_trials=cases * attempts,
        case_id_by_task_selector={case_id: case_id for case_id in case_ids},
    )


def test_a_refused_attempt_fails_attempt_coverage_as_the_docs_say(tmp_path: Path) -> None:
    # The documented example: one refusal in 20 cases with 3 attempts each.
    # Where the fallback used to finish a refused attempt on another model, the
    # attempt was scored. Now it is an unscored trial failure, so the arm's
    # attempt coverage is incomplete, its execution fails and the command exits 1.
    refused = "case-007__attempt2"
    results = _collect_refusal(tmp_path, cases=20, attempts=3, refused=refused)

    arm = results["agents"]["claude-code"]
    assert arm["execution_status"] == results["execution_status"] == "failed"
    assert (arm["expected_attempts"], arm["scored_attempts"]) == (60, 59)
    coverage = f"{_COVERAGE_ERROR} 59/60"
    # The run's result.json persists these execution errors.
    assert coverage in results["execution_errors"]
    # A refusal is a trial failure, reported as an unscoreable reward, and never
    # an agent runtime (infrastructure) failure.
    assert arm["agent_runtime_failures"]["with_skill"] == []
    assert [failure["trial"] for failure in arm["trial_failures"]["with_skill"]] == [refused]
    assert (
        f"Unscoreable reward in {refused}: AgentSafetyRefusalError: Command failed (exit 1)"
        in results["execution_errors"]
    )
    # `tier3 evaluate` exits 1 whenever this reports a failure.
    assert EvaluationService.failure_reason(results) is not None

    texts = _refusal_doc_texts()
    for name, text in texts.items():
        # Each place quotes the persisted message for exactly this example.
        assert re.findall(rf"`({_COVERAGE_ERROR} \d+/\d+)`", text) == [coverage], name
        for token in (
            "`result.json`",
            "complete attempt coverage",
            "`tier3 evaluate`",
            "`validate --block-on-agent-eval`",
            "exits 1",
        ):
            assert token in text, (name, token)
    # The model is chosen per agent, not per case.
    for name in ("CHANGELOG.md", "tier3-live-evaluation.mdx"):
        assert "`--agent-model claude-code=MODEL`" in texts[name], name
    assert "`Unscoreable reward`" in texts["tier3-live-evaluation.mdx"]


@pytest.mark.parametrize("skip_baseline", [True, False], ids=["with-skill-only", "with-baseline"])
def test_verbose_evaluate_output_lists_the_refused_trial(tmp_path: Path, skip_baseline: bool) -> None:
    # Troubleshooting points to `tier3 evaluate --verbose`. Without it, a run with
    # a completed baseline arm prints only a failure count, and the terminal never
    # quotes the coverage message when no arm is usable.
    refused = "case-001__attempt1"
    results = _collect_refusal(tmp_path, cases=1, attempts=1, refused=refused, skip_baseline=skip_baseline)
    stream = StringIO()

    render_evaluation_result(results, console=Console(file=stream, force_terminal=False, width=200), verbose=True)

    assert f"{refused}: AgentSafetyRefusalError: Command failed (exit 1)" in stream.getvalue()
    assert "`tier3 evaluate --verbose`" in _refusal_doc_texts()["tier3-live-evaluation.mdx"]


@pytest.mark.parametrize(
    ("gate", "expected_exit_code"),
    [("--block-on-agent-eval", 1), ("--no-block-on-agent-eval", 0)],
)
def test_validate_block_on_agent_eval_fails_a_run_with_a_refused_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gate: str, expected_exit_code: int
) -> None:
    # The docs say `validate --block-on-agent-eval` exits 1 too. The engine
    # returns the collector's own result for a refused attempt, so the gate reads
    # the failed execution and the coverage error that a refusal produces.
    _clear_route_env(monkeypatch)
    results = _collect_refusal(tmp_path, cases=1, attempts=1, refused="case-001__attempt1")
    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options, **_kwargs: results)
    skill = tmp_path / "simple"
    shutil.copytree(_SIMPLE_SKILL, skill)
    (skill / "evals").mkdir()
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "prompt": "Use the skill."}]), encoding="utf-8"
    )
    output_dir = tmp_path / "reports"

    result = CliRunner().invoke(
        cli,
        [
            "validate",
            "--no-autopilot",
            str(skill),
            "--no-llm",
            "--no-tier2",
            "--tier3",
            gate,
            "--agents",
            "claude-code",
            "--checks",
            "schema",
            "--report",
            "json",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == expected_exit_code, result.output
    report_text = next(output_dir.glob("skillevaluator-output-*.json")).read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["overall_passed"] is (expected_exit_code == 0)
    assert report["gating"]["tiers"]["3"]["blocking"] is (expected_exit_code == 1)
    assert f"{_COVERAGE_ERROR} 0/1" in report_text


def test_refusal_fallback_switch_is_not_redacted_as_a_secret() -> None:
    # Harbor replaces each sensitive value across the trial directory; a value of
    # "1" must never be treated as one.
    assert not is_sensitive_env_key(_SWITCH)


def test_skill_runtime_env_cannot_reenable_refusal_fallback(tmp_path: Path) -> None:
    resolved, errors = _resolve_runtime_env({_SWITCH: "0"}, env_mode="docker")

    assert resolved == {}
    assert errors == [f"harbor.runtime_env.{_SWITCH} controls the host process and is not allowed"]

    _write_task_toml(
        tmp_path,
        {"id": "case-001", "expected_skill": "demo-skill"},
        has_skill=True,
        runtime_env={_SWITCH: "0"},
    )

    assert _staged_environment_env(tmp_path)[_SWITCH] == "1"


def test_native_tasks_cannot_reenable_refusal_fallback(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    native_task = skill / "evals" / "harbor" / "case-001"
    native_task.mkdir(parents=True)
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n'
        f'[environment]\n\n[environment.env]\n{_SWITCH} = "0"\n',
        encoding="utf-8",
    )

    [staged] = stage_native_harbor_tasks(skill, tmp_path / "native")

    assert _staged_environment_env(staged)[_SWITCH] == "1"


@pytest.mark.skipif(os.name == "nt", reason="local mode requires a POSIX host")
def test_local_mode_agent_commands_keep_refusal_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_SWITCH, "0")
    _write_task_toml(tmp_path, {"id": "case-001", "expected_skill": "demo-skill"}, has_skill=True)
    task_config = TaskConfig.model_validate_toml((tmp_path / "task.toml").read_text(encoding="utf-8"))
    (tmp_path / "environment").mkdir()
    (tmp_path / "runtime").mkdir()
    environment = SkillEvaluatorLocalEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="refusal-fallback",
        session_id="refusal-fallback",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=task_config.environment,
        runtime_agent="claude-code",
        runtime_root=str(tmp_path / "runtime"),
        sandbox_mode="off",
    )

    # Harbor's Claude Code agent passes its own per-command variables; the
    # staged task value still reaches the CLI over the host's setting.
    command_env = environment._exec_env({"IS_SANDBOX": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})

    assert command_env[_SWITCH] == "1"


# ---------------------------------------------------------------------------
# Harbor 0.24's own classification of the CLI's refusal (pins a future bump)
# ---------------------------------------------------------------------------


def _load(import_path: str) -> type[ClaudeCode]:
    module, class_name = import_path.split(":")
    return getattr(importlib.import_module(module), class_name)


# Harbor's own agent, the container and local wrappers SkillEvaluator runs, and
# the container wrapper on the Vertex AI route, which has its own auth and exec
# rewriting. Each value is the agent class and the host credentials it reads.
_LANES: dict[str, tuple[Callable[[], type[ClaudeCode]], dict[str, str]]] = {
    "harbor": (lambda: ClaudeCode, {"ANTHROPIC_API_KEY": _FAKE_CREDENTIAL}),
    "container": (
        lambda: _load(CONTAINER_AGENT_IMPORT_PATHS["claude-code"]),
        {"ANTHROPIC_API_KEY": _FAKE_CREDENTIAL},
    ),
    "local": (lambda: _load(LOCAL_AGENT_IMPORT_PATHS["claude-code"]), {"ANTHROPIC_API_KEY": _FAKE_CREDENTIAL}),
    "vertex": (lambda: _load(CONTAINER_AGENT_IMPORT_PATHS["claude-code"]), _VERTEX_ENV),
}
_INSTRUCTION = "Use the skill."


class _TrialEnvironment(BaseEnvironment):
    """Harbor's own task and exec environment merging, with no process behind it.

    The command that carries the task instruction is the Claude Code launch. It
    exits 1 with the given stream, so Harbor classifies the launch command its
    agent really builds, not a copy of it.
    """

    def __init__(self, tmp_path: Path, task_env_config: EnvironmentConfig, launch_stdout: str) -> None:
        self.launch_stdout = launch_stdout
        self.launches: list[tuple[str, dict[str, str]]] = []
        super().__init__(
            environment_dir=tmp_path,
            environment_name="refusal-fallback",
            session_id="refusal-fallback",
            trial_paths=TrialPaths(tmp_path / "trial"),
            task_env_config=task_env_config,
        )

    @staticmethod
    def type() -> str:
        return "refusal-fallback"

    def _validate_definition(self) -> None:
        return None

    async def start(self, force_build: bool) -> None:
        _ = force_build

    async def stop(self, delete: bool) -> None:
        _ = delete

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        _ = (source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        _ = (source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        _ = (source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        _ = (source_dir, target_dir)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = (cwd, timeout_sec, user)
        merged = self._merge_env(env) or {}
        if _INSTRUCTION in command or _INSTRUCTION in merged.values():
            self.launches.append((command, merged))
            return ExecResult(return_code=1, stdout=self.launch_stdout, stderr=None)
        return ExecResult(return_code=0, stdout="", stderr=None)


def _run_trial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lane: str, stdout: str
) -> tuple[NonZeroAgentExitCodeError, str, dict[str, str]]:
    """Run the lane's real ``run()`` in a staged task; return its error, launch command and env."""
    _clear_route_env(monkeypatch)
    agent_class, host_env = _LANES[lane]
    for name, value in host_env.items():
        monkeypatch.setenv(name, value)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    _write_task_toml(task_dir, {"id": "case-001", "expected_skill": "demo-skill"}, has_skill=True)
    task_config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text(encoding="utf-8"))
    environment = _TrialEnvironment(tmp_path, task_config.environment, stdout)
    # The Harbor model id SkillEvaluator passes for Claude Code, on every route.
    agent = agent_class()(logs_dir=tmp_path / "agent", model_name=f"anthropic/{SELECTED_MODEL}")

    with pytest.raises(NonZeroAgentExitCodeError) as raised:
        asyncio.run(agent.run(_INSTRUCTION, environment=environment, context=AgentContext()))

    [(command, launch_env)] = environment.launches
    return raised.value, command, launch_env


@pytest.mark.parametrize("lane", sorted(_LANES))
def test_staged_switch_reaches_the_claude_code_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lane: str
) -> None:
    _, command, launch_env = _run_trial(monkeypatch, tmp_path, lane, no_fallback_stream())

    # The agent's own per-command variables (and the Vertex AI wrapper's
    # filtering of them) leave the staged task value in place.
    assert launch_env[_SWITCH] == "1"
    if lane == "vertex":
        assert launch_env["CLAUDE_CODE_USE_VERTEX"] == "1"
        assert "; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL CLAUDE_CODE_OAUTH_TOKEN; " in command


@pytest.mark.parametrize("lane", sorted(_LANES))
def test_disabled_fallback_refusal_is_harbors_safety_refusal_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lane: str
) -> None:
    error, _, _ = _run_trial(monkeypatch, tmp_path, lane, no_fallback_stream())

    assert type(error) is AgentSafetyRefusalError
    assert "Command failed (exit 1)" in str(error)


@pytest.mark.parametrize("lane", sorted(_LANES))
def test_enabled_fallback_gateway_403_reads_as_an_unknown_api_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lane: str
) -> None:
    # Why the switch is needed: with the fallback on, the refusal never reaches
    # the CLI's terminal events, so Harbor sees only the fallback model's 403.
    error, _, _ = _run_trial(monkeypatch, tmp_path, lane, gateway_fallback_403_stream())

    assert type(error) is UnknownApiError


# The terminal event of the failed launch and the error Harbor must raise for it.
_TERMINAL_EVENTS = {
    "overloaded": (lambda: result_event("API Error: Overloaded", status=529), ApiOverloadedError),
    # Matches no Harbor pattern, so the only refusal text in stdout is the
    # quoted text in the tool output and the model reply.
    "max_turns": (max_turns_event, NonZeroAgentExitCodeError),
}


@pytest.mark.parametrize("terminal", sorted(_TERMINAL_EVENTS))
@pytest.mark.parametrize("lane", sorted(_LANES))
def test_refusal_text_in_tool_output_or_model_text_is_not_a_safety_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lane: str, terminal: str
) -> None:
    # Harbor reads only the CLI's terminal events, and only for the
    # stream-json launch it builds itself. If a Harbor upgrade changed that
    # launch so this filter no longer applied, Harbor would scan all of stdout
    # and the max_turns case would become a safety refusal.
    quoted_event = no_fallback_stream().splitlines()[-3]
    model_text = line(
        {
            "type": "assistant",
            "message": {
                "model": SELECTED_MODEL,
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": f"The log says: {NO_FALLBACK_TEXT}"}],
            },
        }
    )
    terminal_event, expected = _TERMINAL_EVENTS[terminal]
    stdout = "\n".join(
        [
            init_event(),
            *tool_round(f"{quoted_event}\nRequest blocked\n{NO_FALLBACK_TEXT}"),
            model_text,
            terminal_event(),
        ]
    )

    error, _, _ = _run_trial(monkeypatch, tmp_path, lane, stdout)

    assert not isinstance(error, AgentSafetyRefusalError)
    assert type(error) is expected
