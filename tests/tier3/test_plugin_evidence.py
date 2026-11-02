# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private evidence uses the real preparation and lifecycle with synthetic artifacts."""

from __future__ import annotations

import json
import shutil

import pytest

from skillevaluator.evaluation.options import EvaluationOptions
from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.tier3 import plugin_evidence as evidence
from skillevaluator.tier3 import plugin_execution as execution
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS, DEFAULT_SCORE_POLICY
from skillevaluator.tier3.plugin_prepare import prepare_runtime_package


def prepared(root):
    files = {
        ".claude-plugin/plugin.json": {
            "name": "demo",
            "mcpServers": {"docs": {"url": "https://docs.example.invalid/mcp"}},
        },
        "skills/a/SKILL.md": "---\nname: a\ndescription: A.\n---\nBody.\n",
        "skills/a/evals/evals.json": [
            {"id": "one", "prompt": "do it", "expected_tools": ["MCP:docs/search"]},
            {"id": "two", "prompt": "another", "expected_tools": ["MCP:docs/other"]},
        ],
    }
    for rel, value in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
    return prepare_runtime_package(plugin_inventory_for_root(root))


def trajectory():
    return {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": "fixture-trial",
        "agent": {"name": "codex", "version": "test"},
        "final_metrics": {"total_prompt_tokens": 23, "total_completion_tokens": 5, "total_cost_usd": None},
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "message": "",
                "metrics": {"prompt_tokens": 23, "completion_tokens": 5},
                "tool_calls": [
                    {
                        "tool_call_id": "call",
                        "function_name": "mcp__docs__search",
                        "arguments": {"query": "argument-value-should-not-persist"},
                    }
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "call", "content": "found", "extra": {"tool_result_is_error": False}}
                    ]
                },
            }
        ],
    }


def write_trial(jobs, skill, arm="with", *, missing=False, steps=1):
    root = jobs / f"{skill}-codex-{arm}" / "a-one__attempt1"
    for index in range(steps):
        verifier = root / (f"steps/step-{index}/verifier" if steps > 1 else "verifier")
        verifier.mkdir(parents=True)
        reward = {
            **dict.fromkeys(DEFAULT_METRICS, 1.0),
            "entry_id": "a-one",
            "metric_set": DEFAULT_METRIC_SET,
            "score_policy": DEFAULT_SCORE_POLICY,
        }
        (verifier / "reward.json").write_text(json.dumps(reward))
    if not missing:
        for index in range(steps):
            agent = root / (f"steps/step-{index}/agent" if steps > 1 else "agent")
            agent.mkdir()
            (agent / "trajectory.json").write_text(json.dumps(trajectory()))
        if steps > 1:
            (root / "result.json").write_text(
                json.dumps({"step_results": [{"step_name": f"step-{index}"} for index in range(steps)]})
            )
    return root


def run_fixture(
    tmp_path,
    monkeypatch,
    *,
    missing=False,
    steps=1,
    options=None,
    variants=("with",),
    listed_only=False,
    hosted_search=False,
    forged_case=None,
    empty=False,
):
    package = prepared(tmp_path / "plugin")
    work = tmp_path / "work"
    work.mkdir()
    captured = {}

    def evaluate(settings):
        hooks = execution.active_runtime()
        captured["observer"] = hooks.observer
        captured["wrapper"] = settings.skill_path
        jobs = tmp_path / "results" / "_harbor-jobs"
        jobs.mkdir(parents=True)
        for variant in variants:
            if empty:
                (jobs / f"{settings.skill_path.name}-codex-{variant}").mkdir()
                continue
            trial = write_trial(jobs, settings.skill_path.name, variant, missing=missing, steps=steps)
            if forged_case:
                for path in trial.rglob("reward.json"):
                    row = json.loads(path.read_text())
                    row["entry_id"] = forged_case
                    path.write_text(json.dumps(row))
            if listed_only:
                hooks.trial_censuses[trial] = {
                    "loaded": [],
                    "listed": [{"type": "mcp", "name": "docs"}],
                    "not_loaded": [],
                }
            if hosted_search:
                path = trial / "agent/trajectory.json"
                data = json.loads(path.read_text())
                data["steps"][0]["tool_calls"][0]["function_name"] = "web_search_call"
                path.write_text(json.dumps(data))
        result = {
            "execution_status": "succeeded",
            "agents": {
                "codex": {
                    "conditions": {
                        "with_skill": {"execution_status": "succeeded", "scored_attempts": 1, "expected_attempts": 1},
                        "without_skill": {"execution_status": "skipped", "scored_attempts": 0, "expected_attempts": 0},
                    }
                }
            },
        }
        for arm, variant in evidence.ARM_SUFFIXES.items():
            if variant in variants:
                result["agents"]["codex"]["conditions"][arm] = {
                    "execution_status": "succeeded",
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                }
        hooks.collect_evidence(jobs, result)
        hooks.collect_evidence(jobs, result)
        shutil.rmtree(jobs)
        return result

    outcome = execution.evaluate_prepared_plugin(
        package,
        EvaluationOptions(
            skill_path=package.native_source.plugin_root, agents="codex", results_dir=tmp_path / "results"
        ),
        work_dir=work,
        evaluate=evaluate,
        evidence_options=options,
    )
    return outcome, captured


def test_default_collects_advisory_evidence_without_probing_and_survives_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "probe_mcp_servers", lambda *_a, **_k: pytest.fail("default host probe"))
    outcome, captured = run_fixture(tmp_path, monkeypatch)
    assert outcome.status == "succeeded" and outcome.attempts == 1
    assert not captured["wrapper"].exists()
    assert outcome.provenance["mcp_proof"]["docs"]["status"] == "used-successfully"
    summary = outcome.result["agents"]["codex"]["plugin_signals_summary"]["with_skill"]
    assert summary["mcp_calls"]["succeeded"] == 1
    record = captured["observer"].logical_trials["codex"]["with_skill"][0]
    assert record.case_id == "a-one" and record.reward["security"] == 1.0
    assert record.usage.get("total_cost_usd") is None
    assert record.first_turn_usage["prompt_tokens"] == 23
    sidecar = tmp_path / "results" / evidence.RUNTIME_EVIDENCE_FILENAME
    persisted = sidecar.read_text()
    assert "argument-value-should-not-persist" not in persisted
    assert json.loads(persisted)["agents"]["codex"]["with_skill"]["trials"][0]["case_id"] == "a-one"


def test_logical_trial_is_read_and_normalized_once_across_steps_and_repeated_collection(tmp_path, monkeypatch):
    from skillevaluator.tier3.eval_core import plugin_signals
    from skillevaluator.utils.secure_fs import SecureRoot

    reads, normalizations = [], []
    read = SecureRoot.read_bytes
    extract = plugin_signals._extract_calls

    def read_once(self, path, *args, **kwargs):
        if path.name == "trajectory.json":
            reads.append(path)
        return read(self, path, *args, **kwargs)

    def normalized(*args, **kwargs):
        normalizations.append(True)
        return extract(*args, **kwargs)

    monkeypatch.setattr(SecureRoot, "read_bytes", read_once)
    monkeypatch.setattr(plugin_signals, "_extract_calls", normalized)
    outcome, _ = run_fixture(tmp_path, monkeypatch, steps=2)
    assert len(reads) == 2 and len(normalizations) == 1
    assert outcome.result["agents"]["codex"]["plugin_signals_summary"]["with_skill"]["n_trials"] == 1


def test_missing_trajectory_is_explicit_and_never_proves_a_server(tmp_path, monkeypatch):
    outcome, captured = run_fixture(tmp_path, monkeypatch, missing=True)
    summary = outcome.result["agents"]["codex"]["plugin_signals_summary"]["with_skill"]
    assert summary["n_missing_trajectory"] == 1
    assert outcome.provenance["mcp_proof"]["docs"]["status"] == "declared"
    record = captured["observer"].logical_trials["codex"]["with_skill"][0]
    assert record.signals is None and record.usage is None and record.evidence_status == "missing"


def test_explicit_probe_forwards_scoped_grants_and_never_changes_execution_status(tmp_path, monkeypatch):
    seen = []

    def probe(targets, **kwargs):
        seen.append((targets, kwargs))
        return {"docs": {"status": "unreachable", "tools": [], "detail": "fake timeout"}}

    monkeypatch.setattr(evidence, "probe_mcp_servers", probe)
    outcome, _ = run_fixture(
        tmp_path,
        monkeypatch,
        missing=True,
        options=evidence.EvidenceOptions(probe_mcp=True, env_grants=("TOKEN@docs",)),
    )
    assert outcome.status == "succeeded" and len(seen) == 1
    assert seen[0][1]["expand_env"] == ("TOKEN@docs",)
    assert outcome.provenance["mcp_proof"]["docs"]["status"] == "unreachable"


def test_comparison_arm_jobs_use_the_existing_engine_variants(tmp_path, monkeypatch):
    outcome, captured = run_fixture(tmp_path, monkeypatch, variants=("with", "without", "sumofparts", "looseparts"))
    assert set(captured["observer"].logical_trials["codex"]) == {
        "with_skill",
        "without_skill",
        "sum_of_parts",
        "loose_parts",
    }
    assert len(outcome.result["agents"]["codex"]["plugin_signals_summary"]) == 4


def test_listed_only_census_without_init_keeps_load_unknown(tmp_path, monkeypatch):
    outcome, _ = run_fixture(tmp_path, monkeypatch, missing=True, listed_only=True)
    tally = outcome.result["agents"]["codex"]["plugin_signals_summary"]["with_skill"]["mcp_load"]["docs"]
    assert tally["loaded"] == tally["trials"] == 0
    assert tally["unknown"] == 1
    assert outcome.provenance["mcp_proof"]["docs"]["status"] == "declared"


def test_first_turn_hosted_search_is_not_a_context_measurement(tmp_path, monkeypatch):
    _, captured = run_fixture(tmp_path, monkeypatch, hosted_search=True)
    record = captured["observer"].logical_trials["codex"]["with_skill"][0]
    assert record.first_turn_usage is None
    assert record.first_turn_excluded_reason == "hosted_search"


def test_reward_cannot_forge_another_valid_staged_case_for_signals(tmp_path, monkeypatch):
    _, captured = run_fixture(tmp_path, monkeypatch, forged_case="a-two")
    record = captured["observer"].logical_trials["codex"]["with_skill"][0]
    assert record.case_id == record.reward["entry_id"] == "a-one"
    assert record.signals["tool_selection"]["recall"] == 1.0


@pytest.mark.parametrize("variants,empty", [((), False), (("with",), True)])
def test_missing_requested_job_or_empty_job_is_explicit_incomplete_evidence(tmp_path, monkeypatch, variants, empty):
    outcome, _ = run_fixture(tmp_path, monkeypatch, variants=variants, empty=empty)
    assert outcome.provenance["runtime_evidence"]["evidence_complete"] is False
    summary = outcome.result["agents"]["codex"]["plugin_signals_summary"]["with_skill"]
    assert summary["n_missing_trials"] == 1 and summary["evidence_complete"] is False


def test_selected_reward_roots_refuse_escape_and_read_only_the_requested_trial(tmp_path):
    jobs = tmp_path / "jobs"
    first = write_trial(jobs, "demo")
    other = first.with_name("other__attempt1")
    (other / "verifier").mkdir(parents=True)
    (other / "verifier/reward.json").write_text("{}")
    rows = evidence.collector._extract_rewards(first.parent, selected_trial_roots=[first])
    assert {row["_trial_root_name"] for row in rows} == {first.name}
    with pytest.raises(ValueError):
        evidence.collector._extract_rewards(first.parent, selected_trial_roots=[tmp_path])


def test_setup_retry_replaces_cached_trials_without_reprobing_or_rereading_rewards(tmp_path, monkeypatch):
    package = prepared(tmp_path / "plugin")
    work = tmp_path / "work"
    work.mkdir()
    attempts, probes, observers = [], [], []

    def probe(*_args, **_kwargs):
        probes.append(True)
        return {"docs": {"status": "reachable-host", "tools": [], "detail": "fake host proof"}}

    def evaluate(settings):
        hooks = execution.active_runtime()
        observers.append(hooks.observer)
        index = len(attempts)
        jobs = tmp_path / f"attempt-{index}" / "_harbor-jobs"
        root = write_trial(jobs, settings.skill_path.name, missing=index == 0)
        if index:
            root = root.rename(root.with_name("second-task__attempt1"))
            path = root / "agent/trajectory.json"
            data = json.loads(path.read_text())
            data["steps"][0]["tool_calls"][0]["function_name"] = "mcp__docs__other"
            path.write_text(json.dumps(data))
        selector_map = {"a-one": "a-one", "second-task" if index else "a-two": "a-two"}
        (root / "result.json").write_text(json.dumps({"task_id": {"path": root.name.split("__")[0]}}))
        hooks.configure_run(case_id_by_task_selector=selector_map)
        rows = evidence.collector._extract_rewards(root.parent, selector_map)
        scoreable, _ = evidence.collector._partition_scoreable_rewards(rows)
        hooks.observe_collected_trials("codex", "with_skill", root.parent, rows, scoreable, selector_map)
        condition = {
            "execution_status": "succeeded" if index else "failed",
            "expected_attempts": 1,
            "scored_attempts": 1 if index else 0,
        }
        info = {"conditions": {"with_skill": condition}}
        if not index:
            info["trial_failures"] = {"with_skill": [{"trial": "a-one", "reason": "AgentSetupNetworkError: dns"}]}
            info["failure_detail_metadata"] = {"with_skill_trials": {"details_total": 1, "details_truncated": False}}
        result = {"execution_status": condition["execution_status"], "agents": {"codex": info}}
        with monkeypatch.context() as local:
            local.setattr(evidence.collector, "_extract_rewards", lambda *_a, **_k: pytest.fail("reward reread"))
            hooks.collect_evidence(jobs, result)
        attempts.append(result)
        return result

    monkeypatch.setattr(evidence, "probe_mcp_servers", probe)
    outcome = execution.evaluate_prepared_plugin(
        package,
        EvaluationOptions(skill_path=package.native_source.plugin_root, agents="codex"),
        work_dir=work,
        evaluate=evaluate,
        evidence_options=evidence.EvidenceOptions(probe_mcp=True),
    )
    assert outcome.status == "succeeded" and outcome.attempts == 2
    assert len(probes) == 1 and observers[0] is observers[1]
    observer = observers[-1]
    record = observer.logical_trials["codex"]["with_skill"][0]
    assert record.case_id == "a-two" and record.signals["tool_selection"]["recall"] == 1.0
    assert record.usage["total_prompt_tokens"] == 23
    assert len(observer.artifacts._traces) == len(observer._rewards) == 1
    assert attempts[0]["runtime_evidence"]["mcp_proof"]["docs"]["status"] == "reachable-host"
    assert attempts[0]["agents"]["codex"]["plugin_signals_summary"]["with_skill"]["n_missing_trajectory"] == 1
    assert attempts[1]["runtime_evidence"]["mcp_proof"]["docs"]["status"] == "used-successfully"


def test_unscored_trace_keeps_trusted_identity_and_usage(tmp_path, monkeypatch):
    extract = evidence.collector._extract_rewards

    def no_scores(*args, **kwargs):
        rows = extract(*args, **kwargs)
        for row in rows:
            for metric in DEFAULT_METRICS:
                row.pop(metric, None)
        return rows

    monkeypatch.setattr(evidence.collector, "_extract_rewards", no_scores)
    _, captured = run_fixture(tmp_path, monkeypatch)
    record = captured["observer"].logical_trials["codex"]["with_skill"][0]
    assert record.case_id == "a-one" and record.reward is None
    assert record.usage["total_prompt_tokens"] == 23


def test_private_probe_schema_checks_arguments_without_persisting_values(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evidence,
        "probe_mcp_servers",
        lambda *_a, **_k: {
            "docs": {
                "status": "reachable-host",
                "tools": ["search"],
                "detail": "fixture",
                "input_schemas": {
                    "search": {"type": "object", "properties": {"query": {"type": "integer"}}, "required": ["query"]}
                },
            }
        },
    )
    outcome, captured = run_fixture(tmp_path, monkeypatch, options=evidence.EvidenceOptions(probe_mcp=True))
    signals = captured["observer"].logical_trials["codex"]["with_skill"][0].signals
    assert signals["arguments"]["checked"] == 1 and signals["arguments"]["passed"] == 0
    assert outcome.provenance["mcp_proof"]["docs"]["status"] == "used-successfully"
    assert (
        "argument-value-should-not-persist"
        not in (tmp_path / "results" / evidence.RUNTIME_EVIDENCE_FILENAME).read_text()
    )


def test_trial_cap_bounds_reward_and_trajectory_work_and_marks_truncation(tmp_path, monkeypatch):
    original = write_trial

    def two_trials(*args, **kwargs):
        root = original(*args, **kwargs)
        shutil.copytree(root, root.with_name("a-one__attempt2"))
        return root

    selected = []
    extract = evidence.collector._extract_rewards

    def bounded(*args, **kwargs):
        selected.extend(kwargs["selected_trial_roots"])
        return extract(*args, **kwargs)

    monkeypatch.setattr(evidence, "MAX_TRIALS_PER_ARM", 1)
    monkeypatch.setattr(evidence.collector, "_extract_rewards", bounded)
    monkeypatch.setitem(run_fixture.__globals__, "write_trial", two_trials)
    outcome, captured = run_fixture(tmp_path, monkeypatch)
    assert len(selected) == len(captured["observer"].artifacts._traces) == 1
    summary = outcome.result["agents"]["codex"]["plugin_signals_summary"]["with_skill"]
    assert summary["trials_truncated"] is True and summary["evidence_complete"] is False


def test_mutation_during_explicit_probe_aborts_before_dispatch_and_cleans_wrapper(tmp_path, monkeypatch):
    package = prepared(tmp_path / "plugin")
    work = tmp_path / "work"
    work.mkdir()

    def probe(*_args, **_kwargs):
        (tmp_path / "plugin/skills/a/SKILL.md").write_text("changed")
        return {}

    monkeypatch.setattr(evidence, "probe_mcp_servers", probe)
    with pytest.raises(ValueError, match="changed"):
        execution.evaluate_prepared_plugin(
            package,
            EvaluationOptions(skill_path=package.native_source.plugin_root, agents="codex"),
            work_dir=work,
            evidence_options=evidence.EvidenceOptions(probe_mcp=True),
            evaluate=lambda _o: pytest.fail("dispatch after source mutation"),
        )
    assert list(work.iterdir()) == [] and execution.active_runtime() is None


def test_observer_failure_is_advisory_and_does_not_retry(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("fixture observer failure")

    monkeypatch.setattr(evidence, "compute_plugin_signals", broken)
    outcome, _ = run_fixture(tmp_path, monkeypatch)
    assert outcome.status == "succeeded" and outcome.attempts == 1
    assert "fixture observer failure" in outcome.provenance["runtime_evidence"]["errors"][0]
    assert execution.active_runtime() is None


def test_missing_trial_census_keeps_server_load_unknown():
    loaded = evidence.LogicalTrialEvidence(
        "one", "a-one", None, None, {"loaded": [{"type": "mcp", "name": "docs"}]}, None, None, "missing"
    )
    missing = evidence.LogicalTrialEvidence("two", "a-two", None, None, None, None, None, "missing")
    assert evidence._load_tally((loaded, missing))["docs"] == {"loaded": 1, "trials": 1, "unknown": 1}
    assert evidence._proof_census((loaded, missing))["loaded"] == []


def test_cancellation_with_observer_restores_context_and_cleans_wrapper(tmp_path):
    package = prepared(tmp_path / "plugin")
    work = tmp_path / "work"
    work.mkdir()
    calls = []

    def cancel(settings):
        assert execution.active_runtime().observer is not None and settings.skill_path.exists()
        calls.append(True)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execution.evaluate_prepared_plugin(
            package,
            EvaluationOptions(skill_path=package.native_source.plugin_root, agents="codex"),
            work_dir=work,
            evaluate=cancel,
        )
    assert calls == [True] and execution.active_runtime() is None and list(work.iterdir()) == []
