# SPDX-License-Identifier: Apache-2.0
"""Private arms use real emission/orchestration with an in-process fake Harbor."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from skillevaluator.evaluation.options import EvaluationOptions
from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.tier3 import plugin_execution as px
from skillevaluator.tier3.harbor import adapter, collector, runner
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.plugin_comparisons import ComparisonOptions, ComparisonRuntime
from skillevaluator.tier3.plugin_prepare import prepare_runtime_package, stage_wrapper_skill, wrapper_skill_name
from test_tier3_progress import _stub_runner


def _package(root: Path):
    files = {
        ".claude-plugin/plugin.json": {"name": "acceptance"},
        "skills/member/SKILL.md": "---\nname: member\ndescription: Fixture member.\n---\nUse the fixture member.\n",
        "skills/member/evals/evals.json": [
            {"id": f"case-{index}", "prompt": f"Perform fixture task {index}.", "cross_component": True}
            for index in range(1, 6)
        ],
        ".mcp.json": {"mcpServers": {"fixture": {"command": "python3", "args": ["-m", "fixture_server"]}}},
        "rules/policy.md": "# Fixture policy\nKeep outputs concise.\n",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    inventory = plugin_inventory_for_root(root)
    assert inventory is not None
    return prepare_runtime_package(inventory)


def _harbor_result(job: Path, tasks: list[Path], attempts: int) -> None:
    """Real Harbor model shapes, deterministic complete rewards, no agent/provider."""
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.result import TrialResult

    now = datetime(2026, 11, 2, tzinfo=UTC)
    variant = job.name.rsplit("-", 1)[-1]
    score = {"without": 0.2, "sumofparts": 0.4, "looseparts": 0.6, "with": 0.8}[variant]
    init = {
        "type": "system",
        "subtype": "init",
        "skills": ["acceptance:member"] if variant == "with" else [] if variant == "without" else ["member"],
        "plugins": [{"name": "acceptance"}] if variant == "with" else [],
        "mcp_servers": (
            [{"name": "plugin:acceptance:fixture" if variant == "with" else "fixture", "status": "connected"}]
            if variant in {"with", "looseparts"}
            else []
        ),
    }
    trials = []
    for task in tasks:
        entry = json.loads((task / "tests/entry.json").read_text())
        for ordinal in range(1, attempts + 1):
            trial_name = f"{task.name}__attempt-{ordinal}"
            trial = job / trial_name
            (trial / "verifier").mkdir(parents=True)
            (trial / "agent").mkdir()
            setup = task / "environment/skilleval/native/setup.sh"
            if setup.is_file():
                # Fake Harbor skips agent setup. Run only the emitted fixture's
                # copy/listing script, with every writable location in tmp_path,
                # so the collector reads the same census as a real native trial.
                home = trial / "home"
                home.mkdir()
                script = (
                    setup.read_text()
                    .replace("/skilleval/", str(task / "environment/skilleval") + "/")
                    .replace("/logs/agent", str(trial / "agent"))
                )
                completed = subprocess.run(
                    ["sh", "-c", script],
                    env={"PATH": os.environ["PATH"], "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude")},
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                assert completed.returncode == 0, completed.stderr
            (trial / "agent/claude-code.txt").write_text(json.dumps(init) + "\n")
            reward = {
                "entry_id": entry["id"],
                "_attempt_ordinal": ordinal,
                "metric_set": DEFAULT_METRIC_SET,
                **dict.fromkeys(DEFAULT_METRICS, score),
            }
            (trial / "verifier/reward.json").write_text(json.dumps(reward))
            result = TrialResult.model_validate(
                {
                    "id": UUID(int=len(trials) + 2),
                    "task_name": f"fixture/{task.name}",
                    "trial_name": trial_name,
                    "trial_uri": trial.as_uri(),
                    "task_id": {"path": str(task)},
                    "task_checksum": "acceptance-fixture",
                    "config": {"task": {"path": str(task)}, "trial_name": trial_name},
                    "agent_info": {
                        "name": "claude-code",
                        "version": "fixture",
                        "model_info": {"name": "fixture-model"},
                    },
                    "agent_result": {"n_input_tokens": 10, "n_output_tokens": 2},
                    "verifier_result": {"rewards": dict.fromkeys(DEFAULT_METRICS, score)},
                    "started_at": now,
                    "finished_at": now,
                    "step_results": None,
                }
            )
            (trial / "result.json").write_text(result.model_dump_json())
            trials.append(result)
    result = JobResult(
        id=UUID(int=1),
        started_at=now,
        updated_at=now,
        finished_at=now,
        n_total_trials=len(trials),
        stats=JobStats.from_trial_results(trials, n_total_trials=len(trials)),
        trial_results=trials,
    )
    (job / "result.json").write_text(result.model_dump_json())


def test_fake_harbor_native_setup_confirms_the_full_arm_and_caches_its_init_event(tmp_path):
    from skillevaluator.tier3 import plugin_native
    from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle

    package = _package(tmp_path / "plugin")
    task = tmp_path / "tasks/fixture"
    (task / "tests").mkdir(parents=True)
    (task / "tests/entry.json").write_text(json.dumps({"id": "case-1"}))
    staging = build_native_task_staging(
        "claude-code", plugin_native.HARNESS_ADAPTERS["claude-code"], package.native_source
    )
    stage_native_bundle(task / "environment", staging)
    job = tmp_path / "jobs/fixture-with"
    _harbor_result(job, [task], 2)

    plan = staging.bundle.census_plan()
    events = {}
    for trial in sorted(job.iterdir()):
        if not trial.is_dir():
            continue
        census = px._trial_census(
            plugin_native, "claude-code", trial / "agent", plan["declared"], plan, init_events=events
        )
        assert {(row["type"], row["name"]) for row in census["loaded"]} == {("skill", "member"), ("mcp", "fixture")}
        assert events[trial]["plugins"] == [{"name": "acceptance"}]
    assert len(events) == 2


def _engine(monkeypatch, root: Path, *, arm_count: int, route: str | None, keep: bool, ordinary: bool = False):
    package = _package(root / "plugin")
    real_pair = runner._run_agent_pair
    _, skill = _stub_runner(
        monkeypatch,
        root,
        run_agent=real_pair,
        collect=collector.collect_harbor_results,
        task_emitter=adapter.generate_harbor_tasks,
        skill_name=wrapper_skill_name(package.plugin_name),
    )
    shutil.rmtree(skill)
    stage_wrapper_skill(package, skill)
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: (
            {
                "harbor": {
                    "task_source": "evals_json",
                    "base_image_mode": "disabled",
                    "resources": {"cpus": 3, "memory_mb": 5120, "storage_mb": 3072},
                }
            },
            None,
        ),
    )
    monkeypatch.setattr(runner, "_agent_import_path", lambda *_args: route)
    calls, lock, first_wave = [], threading.Lock(), threading.Event()
    active, maximum = 0, 0

    def launch(**kwargs):
        nonlocal active, maximum
        tasks = sorted(path for path in kwargs["dataset"].iterdir() if path.is_dir())
        captured = {
            **kwargs,
            "task_names": [path.name for path in tasks],
            "tasks": tasks,
            "entries": [json.loads((path / "tests/entry.json").read_text()) for path in tasks],
            "tomls": [tomllib.loads((path / "task.toml").read_text()) for path in tasks],
            "files": [
                {path.relative_to(task).as_posix() for path in task.rglob("*") if path.is_file()} for task in tasks
            ],
        }
        with lock:
            calls.append(captured)
            active += kwargs["n_concurrent"]
            maximum = max(maximum, active)
            if len(calls) >= 2:
                first_wave.set()
        assert first_wave.wait(timeout=10), "runner did not share the requested condition budget"
        _harbor_result(kwargs["jobs_dir"] / kwargs["job_name"], tasks, kwargs["n_attempts"])
        with lock:
            active -= kwargs["n_concurrent"]
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", launch)
    seen = {}

    def evaluate(options):
        seen["runtime"] = px.active_runtime()
        seen["wrapper"] = options.skill_path
        return runner.run_harbor_eval(
            options.skill_path,
            ["claude-code"],
            output_dir=options.results_dir,
            n_attempts=options.n_attempts,
            n_concurrent=options.n_concurrent,
            max_cases=options.max_cases,
            keep_harbor_jobs=options.harbor_keep_jobs,
            override_cpus=options.override_cpus,
            override_memory_mb=options.override_memory_mb,
            override_storage_mb=options.override_storage_mb,
            agent_runtime_preflight=options.agent_runtime_preflight,
        )

    options = EvaluationOptions(
        skill_path=skill,
        agents="claude-code",
        results_dir=root / "results",
        n_attempts=2,
        n_concurrent=2,
        max_cases=2,
        harbor_keep_jobs=keep,
        override_cpus=4,
        override_memory_mb=6144,
        override_storage_mb=4096,
        agent_runtime_preflight=False,
    )
    if ordinary:
        result = evaluate(options)
        assert seen["runtime"] is None
    else:
        work = root / "work"
        work.mkdir()
        kwargs = {"comparison_options": ComparisonOptions(loose_parts=arm_count == 4)} if arm_count > 2 else {}
        outcome = px.evaluate_prepared_plugin(
            package, options, work_dir=work, load="native", evaluate=evaluate, **kwargs
        )
        assert outcome.cleanup_error is None and outcome.publication_error is None
        assert not seen["wrapper"].exists() and list(work.iterdir()) == []
        result = outcome.result
        assert isinstance(seen["runtime"], ComparisonRuntime) is (arm_count > 2)
    assert px.active_runtime() is None
    return result, calls, seen["runtime"], maximum


@pytest.mark.parametrize(
    "arm_count,route",
    [
        (2, None),
        (3, "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorClaudeCode"),
        (4, "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode"),
    ],
)
def test_real_runner_emits_private_arms_with_shared_cases_resources_and_budget(tmp_path, monkeypatch, arm_count, route):
    result, calls, runtime, maximum = _engine(monkeypatch, tmp_path, arm_count=arm_count, route=route, keep=True)
    assert result["execution_status"] == "succeeded", result
    assert len(calls) == arm_count
    assert maximum == 2
    by_variant = {call["job_name"].rsplit("-", 1)[1]: call for call in calls}
    assert set(by_variant) == {
        "with",
        "without",
        *(["sumofparts"] if arm_count > 2 else []),
        *(["looseparts"] if arm_count > 3 else []),
    }
    selected = by_variant["with"]["task_names"]
    assert len(selected) == 2
    for call in calls:
        assert call["task_names"] == selected
        assert call["n_attempts"] == 2 and call["expected_trials"] == 4
        assert call["n_concurrent"] == 1
        assert (call["override_cpus"], call["override_memory_mb"], call["override_storage_mb"]) == (4, 6144, 4096)
        for task in call["tomls"]:
            assert {key: task["environment"][key] for key in ("cpus", "memory_mb", "storage_mb")} == {
                "cpus": 3,
                "memory_mb": 5120,
                "storage_mb": 3072,
            }
    assert "Native" in by_variant["with"]["agent_import_path"]
    assert "Private" in by_variant["without"]["agent_import_path"]
    if arm_count > 2:
        assert by_variant["sumofparts"]["agent_import_path"] == by_variant["without"]["agent_import_path"]
        for files, task in zip(by_variant["sumofparts"]["files"], by_variant["sumofparts"]["tomls"], strict=True):
            assert "environment/skills/member/SKILL.md" in files
            assert {path for path in files if path.endswith("/SKILL.md")} == {"environment/skills/member/SKILL.md"}
            assert not any(
                "native/claude-code/plugin" in path or path.endswith("mcp.json") or path.endswith("mcp_servers.json")
                for path in files
            )
            assert not task["environment"].get("mcp_servers")
    if arm_count == 4:
        assert by_variant["looseparts"]["agent_import_path"].endswith(":LoosePartsNvidiaBuildClaudeCode")
        assert runtime.loose_plan.arm_for("claude-code").content["equal_across_arms"] is True


def test_comparison_evidence_is_saved_before_real_runner_cleanup(tmp_path, monkeypatch):
    result, _calls, runtime, _maximum = _engine(monkeypatch, tmp_path, arm_count=4, route=None, keep=False)
    assert result["execution_status"] == "succeeded", result
    run_dir = Path(result["run_dir"])
    assert not (run_dir / "_harbor-tasks").exists()
    assert not (run_dir / "_harbor-jobs").exists()
    saved = json.loads((run_dir / "private_comparisons.json").read_text())
    assert saved["status"] == "recorded" and saved["errors"] == []
    agent = saved["agents"]["claude-code"]
    assert agent["loose_parts"]["equal_across_arms"] is True
    assert agent["integration_claim"]["eligible"] is False
    assert agent["integration_claim"]["verdict"] == "inconclusive"
    ladder = agent["lift_ladder"]
    assert ladder["n_cases"] == 2
    assert ladder["excluded_cases"] == [] and ladder["failed_arms"] == []
    assert {name: step["estimate"] for name, step in ladder["steps"].items()} == pytest.approx(
        {"skill": 0.2, "capability": 0.2, "integration": 0.2, "plugin": 0.6}
    )
    assert {step["n_cases"] for step in ladder["steps"].values()} == {2}
    assert ladder["bundle"]["estimate"] == pytest.approx(0.4)
    assert agent["loose_parts"]["load_evidence"]["dropped_cases"] == []
    assert json.loads((run_dir / "claude-code/statistics.json").read_text())["lift_ladder"] == ladder
    assert runtime.expected_case_ids
    assert json.loads(Path(result["result_path"]).read_text())["private_comparisons"]


def test_ordinary_runner_has_two_arms_and_no_private_comparison_side_effects(tmp_path, monkeypatch):
    result, calls, runtime, _maximum = _engine(
        monkeypatch, tmp_path, arm_count=2, route=None, keep=False, ordinary=True
    )
    assert result["execution_status"] == "succeeded", result
    assert len(calls) == 2 and runtime is None
    assert all(call["agent_import_path"] is None for call in calls)
    assert "private_comparisons" not in result and "runtime_evidence" not in result
    assert "private_comparisons" not in result["run_config"]
    assert not (Path(result["run_dir"]) / "private_comparisons.json").exists()
