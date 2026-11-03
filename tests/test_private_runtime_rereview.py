# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for private runtime retries, custom routes and backend boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.options import EvaluationOptions
from skillevaluator.tier3 import plugin_execution as px
from skillevaluator.tier3 import plugin_native as native
from skillevaluator.tier3.harbor import adapter, runner
from test_private_runtime_review import _collected_setup_failure
from test_tier3_progress import _stub_runner
from tier3.test_plugin_execution import _prepared

CUSTOM_AGENT = "skillevaluator.tier3.harbor.installed_agents:SkillEvaluatorClaudeCode"


def test_wrapper_preserves_custom_backend_routes_in_both_arms(tmp_path, monkeypatch):
    calls = []
    _, skill = _stub_runner(monkeypatch, tmp_path, run_agent=lambda **kw: calls.append(kw) or [])
    monkeypatch.setattr(runner, "_agent_import_path", lambda *_: CUSTOM_AGENT)
    source = _prepared(tmp_path / "plugin").native_source
    hooks = px.NativeRuntimeHooks(source, native.resolve_plugin_load("wrapper", ["claude-code"], env_mode="docker"))
    with px.private_runtime(hooks):
        result = runner.run_harbor_eval(skill, ["claude-code"], output_dir=tmp_path / "results")
    assert result["execution_status"] == "succeeded", result
    assert len(calls) == 1
    assert calls[0]["agent_import_path"] == CUSTOM_AGENT
    assert calls[0]["with_agent_import_path"] == CUSTOM_AGENT


def test_unsupported_native_route_persists_failure_before_build_or_staging(tmp_path, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("refused native route must not build, stage, preflight or launch")

    _, skill = _stub_runner(monkeypatch, tmp_path, task_emitter=unexpected, run_agent=unexpected)
    monkeypatch.setattr(runner, "_agent_import_path", lambda *_: CUSTOM_AGENT)
    monkeypatch.setattr(runner, "build_eval_base_image", unexpected)
    source = _prepared(tmp_path / "plugin").native_source
    hooks = px.NativeRuntimeHooks(source, native.resolve_plugin_load("native", ["claude-code"], env_mode="docker"))
    with px.private_runtime(hooks):
        result = runner.run_harbor_eval(skill, ["claude-code"], output_dir=tmp_path / "results")
    assert result["execution_status"] == "failed"
    assert result["agents"] == {}
    assert "native" in " ".join(result["error"]).lower()
    persisted = json.loads(Path(result["result_path"]).read_text())
    assert persisted["execution_status"] == "failed"
    assert not (Path(result["run_dir"]) / "_harbor-tasks").exists()


LOCAL_ROUTES = [
    (key[0], value)
    for key, value in native.PRIVATE_AGENT_IMPORT_PATHS.items()
    if key[1] and ":SkillEvaluatorLocal" in key[1]
]


@pytest.mark.parametrize(("agent", "path"), LOCAL_ROUTES)
@pytest.mark.parametrize("env_mode", ["gke", "daytona"])
def test_remote_backends_reject_host_only_private_agents(tmp_path, agent, path, env_mode):
    with pytest.raises(ValueError, match="agent_import_path"):
        runner.build_harbor_run_command(
            dataset_path=tmp_path, agent=agent, job_name="test", env_mode=env_mode, agent_import_path=path
        )


@pytest.mark.parametrize(("agent", "path"), LOCAL_ROUTES)
def test_local_backend_keeps_private_local_agents(tmp_path, agent, path):
    command = runner.build_harbor_run_command(
        dataset_path=tmp_path, agent=agent, job_name="test", env_mode="local", agent_import_path=path
    )
    assert command[command.index("--agent") + 1] == path


def test_retry_before_reservation_does_not_publish_previous_evidence(tmp_path):
    package = _prepared(tmp_path / "plugin")
    options = EvaluationOptions(skill_path=tmp_path, agents="opencode", results_dir=tmp_path / "results")
    attempts = []

    def evaluate(current):
        hooks = px.active_runtime()
        attempts.append(hooks)
        if len(attempts) == 1:
            tasks = adapter.generate_harbor_tasks(current.skill_path, tmp_path / "tasks", with_skill=True)
            hooks.stage_with_arm("opencode", tasks)
            result = _collected_setup_failure(tmp_path / "attempt1")
            jobs = tmp_path / "attempt1" / "jobs"
            (jobs / "demo-opencode-with").rename(jobs / f"{current.skill_path.name}-opencode-with")
            (jobs / f"{current.skill_path.name}-opencode-with" / "case-001__attempt" / "agent").mkdir()
            hooks.collect_evidence(jobs, result)
            assert hooks.evidence and hooks.evidence["opencode"]["trials"] == 1
            return result
        return {"error": ["credential probe failed before run reservation"]}

    outcome = px.evaluate_prepared_plugin(package, options, work_dir=tmp_path, load="native", evaluate=evaluate)
    assert outcome.attempts == 2
    assert outcome.provenance.get("native_evidence", {}) == {}
    assert "native_evidence_error" not in outcome.provenance
    assert outcome.setup_attempt_results[0]["native_evidence"]


def test_new_attempt_discards_previous_staging_and_evidence_error(tmp_path):
    package = _prepared(tmp_path / "plugin")
    hooks = px.NativeRuntimeHooks(package.native_source, {})
    hooks.evidence = {"old": {"loaded": ["old-skill"]}}
    hooks.evidence_error = "old collection failure"
    hooks.stagings["opencode"] = object()
    hooks.collected_jobs.add(tmp_path)

    def evaluate(_):
        assert hooks.evidence == {} and hooks.evidence_error is None
        assert hooks.stagings == {} and hooks.collected_jobs == set()
        return {"error": ["no reservation"]}

    px.run_private_evaluation(
        options=EvaluationOptions(skill_path=tmp_path), hooks=hooks, provenance={}, evaluate=evaluate
    )


def test_native_mcp_uses_each_tasks_runtime_environment_even_with_cached_bundle(tmp_path):
    from tier3.test_native_mcp_policy import _source

    source = _source(tmp_path, {"command": "python", "env": {"FLAG": "${SAFE_FLAG}"}})
    hooks = px.NativeRuntimeHooks(source, native.resolve_plugin_load("native", ["claude-code"], env_mode="docker"))
    tasks = []
    for name, runtime_env in (("approved", {"SAFE_FLAG": "enabled"}), ("missing", {})):
        task = tmp_path / name
        environment = task / "environment"
        environment.mkdir(parents=True)
        (environment / "Dockerfile").write_text("FROM python:3.12-slim\n")
        adapter._write_task_toml(task, {"id": name}, True, runtime_env=runtime_env)
        tasks.append(task)
    hooks.stage_with_arm("claude-code", tasks[:1])
    config = tasks[0] / "environment/skilleval/native/claude-code/plugin/.mcp.json"
    assert json.loads(config.read_text())["mcpServers"]["fixture"]["env"] == {"FLAG": "${SAFE_FLAG}"}
    with pytest.raises(ValueError, match="SAFE_FLAG"):
        hooks.stage_with_arm("claude-code", tasks[1:])
    assert not (tasks[1] / "environment/skilleval").exists()
