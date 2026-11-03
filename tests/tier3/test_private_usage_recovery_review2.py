# SPDX-License-Identifier: Apache-2.0
"""Recorded Harbor usage survives missing ATIF without inventing observations."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from skillevaluator.tier3 import plugin_evidence as evidence
from skillevaluator.tier3.plugin_trace_artifacts import TrialTrace, TrialTraceArtifacts


def _observer(job, *, final=None):
    observer = evidence.RuntimeEvidenceObserver.__new__(evidence.RuntimeEvidenceObserver)
    observer._jobs_dir = job.parent
    observer.logical_trials, observer._rewards, observer.summaries = {}, {job: ([], [])}, {}
    observer.selector_map = {"selected-task": "logical-case"}
    observer.hooks = SimpleNamespace(trial_censuses={}, trial_init_events={})
    observer.artifacts = TrialTraceArtifacts()
    observer.context = SimpleNamespace(case_spec=lambda _: {}, arm_enabled=lambda _: False, mcp_input_schemas={})
    if final is not None:
        observer.artifacts._traces[next(job.iterdir())] = TrialTrace({"final_metrics": final, "steps": []}, {})
    return observer


def _trial(tmp_path, *, agent_result=None, steps=None, failed=False):
    job = tmp_path / "jobs" / "demo-codex-with"
    root = job / "demo-codex-with-selected-task-attempt001__trial"
    root.mkdir(parents=True)
    payload = {
        "trial_uri": root.as_uri(),
        "task_checksum": "fixture",
        "agent_info": {"name": "codex"},
        "config": {"task": {"path": "/tasks/selected-task"}},
        "agent_result": agent_result,
        "step_results": steps,
        "exception_info": {"exception_type": "AgentTimeoutError"} if failed else None,
    }
    (root / "result.json").write_text(json.dumps(payload))
    return job, root


@pytest.mark.parametrize("failed", [False, True])
def test_harbor_usage_and_physical_ordinal_are_kept_without_a_score(tmp_path, failed):
    job, _root = _trial(
        tmp_path,
        agent_result={"n_input_tokens": 10, "n_cache_tokens": 2, "n_output_tokens": 3, "cost_usd": 0.25},
        failed=failed,
    )
    record = _observer(job).collect_arm("codex", "with_skill", job)[0]
    assert record.usage == {"prompt_tokens": 10, "cached_tokens": 2, "completion_tokens": 3, "cost_usd": 0.25}
    assert record.reward is None and record.case_id == "logical-case"
    assert record.attempt_ordinal == 1
    assert record.first_turn_usage is None


def test_fallback_fills_only_missing_finite_fields_and_preserves_zero(tmp_path):
    job, _root = _trial(
        tmp_path, agent_result={"n_input_tokens": 10, "n_cache_tokens": 2, "n_output_tokens": 3, "cost_usd": 0.25}
    )
    record = _observer(
        job, final={"total_prompt_tokens": 0, "total_completion_tokens": None, "total_cost_usd": 0}
    ).collect_arm("codex", "with_skill", job)[0]
    assert record.usage == {
        "total_prompt_tokens": 0,
        "total_completion_tokens": None,
        "total_cost_usd": 0,
        "completion_tokens": 3,
        "cached_tokens": 2,
    }


def test_multistep_fallback_uses_each_complete_step_once_and_never_root_mirror(tmp_path):
    job, _root = _trial(
        tmp_path,
        agent_result={"n_input_tokens": 100, "n_output_tokens": 100, "cost_usd": 100},
        steps=[
            {"step_name": "first", "agent_result": {"n_input_tokens": 10, "n_output_tokens": 2, "cost_usd": 0}},
            {"step_name": "second", "agent_result": {"n_input_tokens": 20, "n_output_tokens": 3, "cost_usd": 0.25}},
        ],
    )
    record = _observer(job).collect_arm("codex", "with_skill", job)[0]
    assert record.usage == {"prompt_tokens": 30, "completion_tokens": 5, "cost_usd": 0.25}


def test_partial_or_invalid_step_counters_do_not_become_totals(tmp_path):
    job, _root = _trial(
        tmp_path,
        steps=[
            {"step_name": "first", "agent_result": {"n_input_tokens": 10, "n_output_tokens": 0, "cost_usd": 0.1}},
            {"step_name": "second", "agent_result": {"n_output_tokens": 0, "cost_usd": True}},
        ],
    )
    assert _observer(job).collect_arm("codex", "with_skill", job)[0].usage == {"completion_tokens": 0}


def test_result_usage_reuses_the_identity_read_and_arm_cache(tmp_path, monkeypatch):
    job, root = _trial(tmp_path, agent_result={"n_input_tokens": 0, "n_output_tokens": 0, "cost_usd": 0})
    observer = _observer(job, final={})
    read = evidence.collector._read_json
    reads = []

    def counted(path):
        reads.append(path)
        return read(path)

    monkeypatch.setattr(evidence.collector, "_read_json", counted)
    first = observer.collect_arm("codex", "with_skill", job)
    assert first[0].usage == {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0}
    assert observer.collect_arm("codex", "with_skill", job) is first
    assert reads == [root / "result.json"]


@pytest.mark.parametrize("value", [None, True, -1, float("inf"), "12"])
def test_invalid_harbor_counters_stay_unavailable(tmp_path, value):
    job, _root = _trial(tmp_path, agent_result={"n_input_tokens": value, "cost_usd": value})
    assert _observer(job).collect_arm("codex", "with_skill", job)[0].usage is None
