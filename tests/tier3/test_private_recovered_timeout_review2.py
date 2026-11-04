# SPDX-License-Identifier: Apache-2.0
"""Private statistics honor the collector's verified physical timeout recovery."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from skillevaluator.tier3 import plugin_evidence as evidence
from skillevaluator.tier3.plugin_comparisons import ComparisonOptions, ComparisonRuntime
from skillevaluator.tier3.plugin_trace_artifacts import TrialTraceArtifacts
from test_harbor_timeout_recovery import collect, native_attempt


def _attempt(root, ordinal, *, timeout):
    job = native_attempt(root, ordinal, timeout=timeout)
    trial = next(path for path in job.iterdir() if path.is_dir())
    payload = json.loads((trial / "result.json").read_text())
    payload["agent_result"] = {"n_input_tokens": 10, "n_cache_tokens": 2, "n_output_tokens": 3, "cost_usd": 0.25}
    (trial / "result.json").write_text(json.dumps(payload))
    aggregate = json.loads((job / "result.json").read_text())
    aggregate["trial_results"] = [payload]
    (job / "result.json").write_text(json.dumps(aggregate))
    return job


def _runtime(root, result):
    job = root / "jobs" / "demo-opencode-with"
    observer = evidence.RuntimeEvidenceObserver.__new__(evidence.RuntimeEvidenceObserver)
    observer._jobs_dir = job.parent
    observer.logical_trials, observer._rewards, observer.summaries = {}, {}, {}
    observer.selector_map = {"case-a": "case-a"}
    observer.hooks = SimpleNamespace(trial_censuses={}, trial_init_events={})
    observer.artifacts = TrialTraceArtifacts()
    observer.context = SimpleNamespace(case_spec=lambda _: {}, arm_enabled=lambda _: False, mcp_input_schemas={})
    observer.collect_arm("opencode", "with_skill", job)
    runtime = ComparisonRuntime(
        SimpleNamespace(observer=observer, decisions={}),
        SimpleNamespace(native_source=SimpleNamespace(member_skills=())),
        ComparisonOptions(member_skills=False),
        ["opencode"],
        baseline=False,
    )
    runtime.expected_case_ids = ("case-a",)
    runtime.n_attempts, runtime.stop_on_pass = 3, True
    runtime.jobs = {"opencode": {"with_skill": job}}
    runtime._collect_statistics(result, job.parent)
    return runtime.data["agents"]["opencode"], observer.logical_trials["opencode"]["with_skill"]


def test_verified_recovery_keeps_successful_statistics_and_all_physical_usage(tmp_path):
    failed = _attempt(tmp_path / "attempts", 1, timeout=True)
    passed = _attempt(tmp_path / "attempts", 2, timeout=False)
    result = collect(tmp_path, failed, passed)
    assert result["agents"]["opencode"]["conditions"]["with_skill"]["recovered_attempts_total"] == 1
    stats, records = _runtime(tmp_path, result)
    assert stats["conditions"]["with_skill"]["execution_status"] == "succeeded"
    assert stats["conditions"]["with_skill"]["expected_attempts"] == 1
    assert stats["conditions"]["with_skill"]["physical_expected_attempts"] == 2
    assert stats["reliability"]["with_skill"]["pass_at_k"] == 1
    assert stats["cost"]["with_skill"] == {
        "total_tokens": 22,
        "tokens_per_success": 22,
        "total_usd": 0.5,
        "usd_per_success": 0.5,
        "successes": 1,
        "observations_complete": True,
    }
    assert records[0].reward is None


@pytest.mark.parametrize("mutation", ["checksum", "exception"])
def test_unverified_recovery_stays_failed(tmp_path, mutation):
    failed = _attempt(tmp_path / "attempts", 1, timeout=True)
    passed = _attempt(tmp_path / "attempts", 2, timeout=False)
    trial = next(path for path in failed.iterdir() if path.is_dir())
    payload = json.loads((trial / "result.json").read_text())
    if mutation == "checksum":
        payload["task_checksum"] = "different"
    else:
        payload["exception_info"]["exception_type"] = "AgentSetupError"
    (trial / "result.json").write_text(json.dumps(payload))
    aggregate = json.loads((failed / "result.json").read_text())
    aggregate["trial_results"] = [payload]
    (failed / "result.json").write_text(json.dumps(aggregate))
    result = collect(tmp_path, failed, passed)
    stats, _records = _runtime(tmp_path, result)
    assert stats["conditions"]["with_skill"]["execution_status"] == "failed"
    assert "with_skill" not in stats["reliability"]
