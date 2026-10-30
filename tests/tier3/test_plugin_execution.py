# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private plugin evaluation lifecycle over the existing engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.options import EvaluationOptions
from skillevaluator.tier3 import plugin_execution as px

OK = {"execution_status": "succeeded"}
SETUP = {"execution_status": "failed", "error": ["claude-code with-skill: AgentSetupNetworkError: dns"]}
AGENT = {"execution_status": "failed", "error": ["claude-code with-skill: NonZeroAgentExitCodeError: 1"]}


def _options(tmp_path: Path) -> EvaluationOptions:
    return EvaluationOptions(skill_path=tmp_path)


def _sequence(*results):
    calls: list[object] = []
    items = list(results)

    def evaluate(options):
        calls.append(px.active_runtime())
        item = items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return dict(item)

    return evaluate, calls


def test_success_publishes_before_cleanup_and_activates_hooks_only_inside_the_run(tmp_path: Path) -> None:
    order: list[str] = []
    hooks = object()
    evaluate, calls = _sequence(OK)
    outcome = px.run_private_evaluation(
        _options(tmp_path),
        hooks=hooks,  # type: ignore[arg-type]
        provenance={"plugin": "demo"},
        evaluate=evaluate,
        publish=lambda _: order.append("publish"),
        cleanup=lambda: order.append("cleanup"),
    )
    assert outcome.status == "succeeded" and outcome.attempts == 1
    assert order == ["publish", "cleanup"]
    assert calls == [hooks] and px.active_runtime() is None


def test_setup_only_failure_is_retried_within_the_bound(tmp_path: Path) -> None:
    evaluate, _ = _sequence(SETUP, SETUP, OK)
    outcome = px.run_private_evaluation(_options(tmp_path), hooks=None, provenance={}, evaluate=evaluate)
    assert outcome.status == "succeeded" and outcome.attempts == 3 and len(outcome.retry_reasons) == 2
    evaluate, _ = _sequence(SETUP, SETUP, SETUP)
    outcome = px.run_private_evaluation(_options(tmp_path), hooks=None, provenance={}, evaluate=evaluate)
    assert outcome.status == "failed" and outcome.attempts == 3


@pytest.mark.parametrize(
    "result",
    [AGENT, {"execution_status": "failed", "error": ["AgentSetupNetworkError", "verifier timed out"]}, {}],
)
def test_agent_verifier_and_ambiguous_failures_are_never_retried(tmp_path: Path, result: dict) -> None:
    evaluate, calls = _sequence(result, OK)
    outcome = px.run_private_evaluation(_options(tmp_path), hooks=None, provenance={}, evaluate=evaluate)
    assert outcome.attempts == 1 and outcome.status == "failed" and len(calls) == 1


def test_cancellation_is_not_retried_and_still_cleans_up(tmp_path: Path) -> None:
    cleaned: list[bool] = []
    evaluate, calls = _sequence(KeyboardInterrupt(), OK)
    with pytest.raises(KeyboardInterrupt):
        px.run_private_evaluation(
            _options(tmp_path),
            hooks=object(),  # type: ignore[arg-type]
            provenance={},
            evaluate=evaluate,
            cleanup=lambda: cleaned.append(True),
        )
    assert len(calls) == 1 and cleaned == [True] and px.active_runtime() is None


def test_publication_and_cleanup_failures_keep_the_run_status(tmp_path: Path) -> None:
    def boom(*_):
        raise OSError("disk full")

    evaluate, _ = _sequence(AGENT)
    outcome = px.run_private_evaluation(
        _options(tmp_path), hooks=None, provenance={}, evaluate=evaluate, publish=boom, cleanup=boom
    )
    assert outcome.status == "failed"
    assert outcome.publication_error == "OSError: disk full" == outcome.cleanup_error


def test_retry_bound_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_setup_retries"):
        px.run_private_evaluation(_options(tmp_path), hooks=None, provenance={}, evaluate=dict, max_setup_retries=9)


def test_ordinary_runs_have_no_active_runtime() -> None:
    from skillevaluator.tier3.harbor import runner

    assert runner._private_runtime_hooks() is None
    assert runner._private_with_agent_import_path("claude-code", None) is None


def _prepared(root: Path):
    import json

    from skillevaluator.plugin_inventory import plugin_inventory_for_root
    from skillevaluator.tier3.plugin_prepare import prepare_runtime_package

    files = {
        ".claude-plugin/plugin.json": {"name": "demo"},
        "skills/a/SKILL.md": "---\nname: a\ndescription: A.\n---\nBody.\n",
        "skills/a/evals/evals.json": [{"id": "one", "prompt": "do it"}],
    }
    for rel, value in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    inventory = plugin_inventory_for_root(root)
    assert inventory is not None
    return prepare_runtime_package(inventory)


def test_prepared_plugin_runs_wrapper_with_durable_results_and_removes_staging(tmp_path: Path) -> None:
    package = _prepared(tmp_path / "plugin")
    work = tmp_path / "work"
    work.mkdir()
    seen: dict = {}

    def evaluate(options):
        seen["skill"] = options.skill_path
        seen["results"] = options.results_dir
        seen["exists"] = (options.skill_path / "evals" / "evals.json").is_file()
        return dict(OK)

    outcome = px.evaluate_prepared_plugin(package, _options(tmp_path), work_dir=work, evaluate=evaluate)
    assert outcome.status == "succeeded" and seen["exists"]
    assert seen["results"] == tmp_path / "plugin" / "evals" / "results"
    assert outcome.provenance["case_sources"] == {"a-one": "skills/a/evals/evals.json"}
    assert outcome.provenance["source"]["manifest_sha256"] == package.identity.manifest_sha256
    assert not seen["skill"].exists() and list(work.iterdir()) == []


def test_native_hooks_stage_only_the_with_arm_of_native_agents(tmp_path: Path) -> None:
    from skillevaluator.tier3.plugin_native import resolve_plugin_load

    package = _prepared(tmp_path / "plugin")
    decisions = resolve_plugin_load("native", ["claude-code"], env_mode="docker")
    hooks = px.NativeRuntimeHooks(package.native_source, decisions)
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim", encoding="utf-8")
    hooks.stage_with_arm("claude-code", [task])
    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM python:3.12-slim\n") and "COPY" in dockerfile
    assert (task / "environment" / "skilleval").is_dir()
    assert hooks.with_agent_import_path("codex", None) is None
    hooks.stage_with_arm("codex", [tmp_path / "missing"])


def test_native_load_without_agents_is_refused(tmp_path: Path) -> None:
    package = _prepared(tmp_path / "plugin")
    with pytest.raises(px.PluginLoadError, match="explicit agents"):
        px.evaluate_prepared_plugin(package, _options(tmp_path), work_dir=tmp_path, load="native", evaluate=dict)


def test_native_evidence_is_read_before_cleanup_and_never_inferred_from_staging(tmp_path: Path) -> None:
    package = _prepared(tmp_path / "plugin")
    work = tmp_path / "work"
    work.mkdir()
    jobs = tmp_path / "jobs"

    def evaluate(options):
        assert options.harbor_keep_jobs
        task = tmp_path / "task"
        (task / "environment").mkdir(parents=True)
        (task / "environment" / "Dockerfile").write_text("FROM x\n", encoding="utf-8")
        px.active_runtime().stage_with_arm("claude-code", [task])
        (jobs / f"{options.skill_path.name}-claude-code-with" / "trial-1" / "agent").mkdir(parents=True)
        return {"execution_status": "succeeded", "harbor_jobs_dir": str(jobs)}

    options = EvaluationOptions(skill_path=tmp_path, agents="claude-code")
    outcome = px.evaluate_prepared_plugin(package, options, work_dir=work, load="native", evaluate=evaluate)
    assert outcome.publication_error is None, outcome.publication_error
    evidence = outcome.provenance["native_evidence"]["claude-code"]
    assert evidence["trials"] == 1
    assert evidence["loaded"] == []
    assert not jobs.exists() and list(work.iterdir()) == []


def _staged_claude_hooks(tmp_path: Path):
    from skillevaluator.tier3 import plugin_native

    package = _prepared(tmp_path / "plugin")
    decisions = plugin_native.resolve_plugin_load("native", ["claude-code"], env_mode="docker")
    hooks = px.NativeRuntimeHooks(package.native_source, decisions)
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM x\n", encoding="utf-8")
    hooks.stage_with_arm("claude-code", [task])
    plan = hooks.stagings["claude-code"].bundle.census_plan()
    return hooks, plan


def _trial(tmp_path: Path, plan, *, init: dict | None):
    from skillevaluator.tier3 import plugin_native

    logs = tmp_path / "jobs" / "w-claude-code-with" / "trial-1" / "agent"
    logs.mkdir(parents=True)
    census = {"agent": "claude-code", "mode": "native", "listed": plan["declared"], "loaded": [], "not_loaded": []}
    (logs / plugin_native.LOAD_CENSUS_FILENAME).write_text(json.dumps(census), encoding="utf-8")
    if init is not None:
        (logs / "claude-code.txt").write_text(json.dumps(init) + "\n", encoding="utf-8")


def test_claude_init_event_confirms_listed_plugin_skills(tmp_path: Path) -> None:
    hooks, plan = _staged_claude_hooks(tmp_path)
    plugin = plan["harness"]["plugin"]
    skills = [f"{plugin}:{d['name']}" for d in plan["declared"] if d["type"] == "skill"]
    assert skills
    init = {"type": "system", "subtype": "init", "plugins": [{"name": plugin}], "skills": skills}
    _trial(tmp_path, plan, init=init)
    evidence = px.collect_native_evidence(hooks, tmp_path / "jobs", "w")["claude-code"]
    assert {f"{plugin}:{e['name']}" for e in evidence["loaded"] if e["type"] == "skill"} == set(skills)


def test_sandbox_census_alone_never_reports_loaded(tmp_path: Path) -> None:
    hooks, plan = _staged_claude_hooks(tmp_path)
    _trial(tmp_path, plan, init=None)
    evidence = px.collect_native_evidence(hooks, tmp_path / "jobs", "w")["claude-code"]
    assert evidence["loaded"] == [] and evidence["trials"] == 1
