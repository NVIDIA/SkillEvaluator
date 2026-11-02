# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private orchestration regressions through real task generation and collection."""

from __future__ import annotations

import copy
import json
import shutil
import threading
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from skillevaluator.evaluation.options import EvaluationOptions
from skillevaluator.tier3 import plugin_execution as px
from skillevaluator.tier3 import plugin_native as native
from skillevaluator.tier3.harbor import adapter, collector, runner
from skillevaluator.tier3.plugin_prepare import stage_wrapper_skill, wrapper_skill_name
from test_harbor_collector_runtime_failures import _write_actual_harbor_022_result
from test_tier3_progress import _stub_runner
from tier3.test_plugin_execution import _prepared


@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode"])
@pytest.mark.parametrize("keep", [False, True])
def test_runner_selects_native_before_workers_and_projects_only_native_layout(tmp_path, monkeypatch, agent, keep):
    package = _prepared(tmp_path / "plugin")
    calls = []
    owner = threading.get_ident()

    def launch(**kwargs):
        assert threading.get_ident() != owner
        assert px.active_runtime() is None  # ContextVars do not propagate through the worker pool.
        calls.append(kwargs)
        return []

    _, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        run_agent=launch,
        task_emitter=adapter.generate_harbor_tasks,
        skill_name=wrapper_skill_name(package.plugin_name),
    )
    shutil.rmtree(skill)
    stage_wrapper_skill(package, skill)
    hooks = px.NativeRuntimeHooks(
        package.native_source, native.resolve_plugin_load("native", [agent], env_mode="docker"), package=package
    )
    with px.private_runtime(hooks):
        result = runner.run_harbor_eval(skill, [agent], output_dir=tmp_path / "results", keep_harbor_jobs=keep)
    assert result["execution_status"] == "succeeded", result
    assert len(calls) == 1
    assert "Native" in calls[0]["with_agent_import_path"]
    assert "Private" in calls[0]["agent_import_path"]
    assert result["harbor_jobs_retained"] is keep
    persisted = json.loads(Path(result["result_path"]).read_text())
    assert persisted["harbor_jobs_retained"] is keep
    assert "native_evidence" in persisted
    tasks = Path(result["run_dir"]) / "_harbor-tasks"
    assert tasks.exists() is keep
    if keep:
        with_tasks = list((tasks / agent / "with").iterdir())
        task = next(path for path in with_tasks if (path / "environment").is_dir())
        assert not (task / "environment" / "skills" / skill.name).exists()
        assert (task / "environment" / "skills" / "a" / "SKILL.md").exists() is (agent != "claude-code")
        entry = json.loads((task / "tests" / "entry.json").read_text())
        assert "a" in entry["workspace_skill_names"]
        assert set(hooks.stagings[agent].workspace_skill_aliases()) <= set(entry["workspace_skill_names"])


@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode", "hermes"])
def test_wrapper_mcp_reaches_with_task_and_never_baseline(tmp_path, agent):
    package = _prepared(tmp_path / "plugin")
    source = replace(
        package.native_source,
        mcp_servers=({"name": "demo", "command": "python", "args": ["-m", "example"], "env": {"SAFE_FLAG": "yes"}},),
    )
    wrapper = stage_wrapper_skill(package, tmp_path / "wrapper")
    with_tasks = adapter.generate_harbor_tasks(wrapper.path, tmp_path / "with", with_skill=True)
    without_tasks = adapter.generate_harbor_tasks(wrapper.path, tmp_path / "without", with_skill=False)
    hooks = px.NativeRuntimeHooks(source, native.resolve_plugin_load("wrapper", [agent], env_mode="docker"))
    hooks.stage_with_arm(agent, with_tasks)
    for task in with_tasks:
        servers = tomllib.loads((task / "task.toml").read_text())["environment"]["mcp_servers"]
        assert servers == [
            {
                "name": "demo",
                "command": "python",
                "args": ["-m", "example"],
                "env": {"SAFE_FLAG": "yes"},
                "transport": "stdio",
            }
        ]
        assert json.loads((task / "mcp_servers.json").read_text()) == servers
    for task in without_tasks:
        assert not tomllib.loads((task / "task.toml").read_text())["environment"].get("mcp_servers")


def _collected_setup_failure(tmp_path, *, baseline_exception="AgentSetupNetworkError"):
    jobs = tmp_path / "jobs"
    for arm, exception in (("with", "AgentSetupNetworkError"), ("without", baseline_exception)):
        _write_actual_harbor_022_result(
            jobs / f"demo-opencode-{arm}", verifier_mode="null" if exception else "present", exception_type=exception
        )
    return collector.collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=tmp_path / "results",
        jobs_dir=jobs,
        skip_baseline=False,
        expected_cases=1,
        expected_case_ids=["case-001"],
        expected_trials=1,
    )


def test_setup_retry_accepts_real_collector_diagnostics_and_retains_failed_attempt(tmp_path):
    result = _collected_setup_failure(tmp_path)
    assert any("expected" in error.lower() or "missing" in error.lower() for error in result["execution_errors"])
    assert px.setup_only_failure(result)
    results = iter((result, {"execution_status": "succeeded"}))
    outcome = px.run_private_evaluation(
        EvaluationOptions(skill_path=tmp_path), hooks=None, provenance={}, evaluate=lambda _: next(results)
    )
    assert outcome.attempts == 2
    assert outcome.setup_attempt_results == [result]


@pytest.mark.parametrize("baseline_exception", [None, "AgentTimeoutError", "NonZeroAgentExitCodeError"])
def test_setup_retry_never_replays_completed_or_dispatched_baseline(tmp_path, baseline_exception):
    result = _collected_setup_failure(tmp_path, baseline_exception=baseline_exception)
    assert px.setup_only_failure(result) is None


@pytest.mark.parametrize("case", ["missing", "truncated", "unknown"])
def test_setup_retry_refuses_incomplete_trial_proof(tmp_path, case):
    result = copy.deepcopy(_collected_setup_failure(tmp_path))
    info = result["agents"]["opencode"]
    if case == "missing":
        info["conditions"]["without_skill"]["expected_attempts"] = 2
    elif case == "truncated":
        info["failure_detail_metadata"]["without_skill_trials"]["details_truncated"] = True
    else:
        info["trial_failures"]["without_skill"][0]["reason"] = "unknown"
    assert px.setup_only_failure(result) is None


def test_auto_records_agent_fallback_and_native_refuses_before_engine(tmp_path):
    package = _prepared(tmp_path / "plugin")
    package = replace(
        package, native_source=replace(package.native_source, refusals=(("skill", "a", "permission bypass"),))
    )
    options = EvaluationOptions(skill_path=tmp_path, agents="claude-code,codex")
    outcome = px.evaluate_prepared_plugin(
        package, options, work_dir=tmp_path, load="auto", evaluate=lambda _: {"execution_status": "succeeded"}
    )
    assert all(decision["mode"] == "wrapper" for decision in outcome.provenance["load_decisions"].values())
    with pytest.raises(px.PluginLoadError, match="permission bypass"):
        px.evaluate_prepared_plugin(
            package,
            options,
            work_dir=tmp_path,
            load="native",
            evaluate=lambda _: pytest.fail("must refuse before engine"),
        )


def test_private_resolution_canonicalizes_and_deduplicates_agents(tmp_path):
    package = _prepared(tmp_path / "plugin")
    options = EvaluationOptions(skill_path=tmp_path, agents="claude,claude-code,opencode,opencode")
    outcome = px.evaluate_prepared_plugin(
        package, options, work_dir=tmp_path, load="auto", evaluate=lambda _: {"execution_status": "succeeded"}
    )
    assert set(outcome.provenance["load_decisions"]) == {"claude-code", "opencode"}


@pytest.mark.parametrize("exception", [RuntimeError("launch failed"), KeyboardInterrupt()])
def test_private_exception_uses_engine_cleanup_for_jobs_and_tasks(tmp_path, monkeypatch, exception):
    package = _prepared(tmp_path / "plugin")

    def launch(**kwargs):
        raise exception

    _, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        run_agent=launch,
        task_emitter=adapter.generate_harbor_tasks,
        skill_name=wrapper_skill_name(package.plugin_name),
    )
    shutil.rmtree(skill)
    stage_wrapper_skill(package, skill)
    hooks = px.NativeRuntimeHooks(
        package.native_source, native.resolve_plugin_load("native", ["codex"], env_mode="docker")
    )
    with px.private_runtime(hooks), pytest.raises(type(exception)):
        runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")
    assert not list((tmp_path / "results").rglob("_harbor-jobs"))
    assert not list((tmp_path / "results").rglob("_harbor-tasks"))
