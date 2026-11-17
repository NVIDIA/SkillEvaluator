# SPDX-License-Identifier: Apache-2.0
"""Paired execution coverage is independent of benchmark metric applicability."""

import copy

import pytest

from ci import tier_manifest
from skillevaluator.reporting.base import agent_eval_publication_evidence_complete


def complete_report(*, baseline_na=True):
    """Compact native shape, containing no task prompts or model output."""
    dimensions = [
        {"id": name, "with_skill": 0.8, "score": 0.8, "baseline": 0.4, "lift": 0.4}
        for name in ("security", "correctness", "discoverability", "effectiveness", "efficiency")
    ]
    agent = {
        "model": "fixture-model",
        "execution_status": "succeeded",
        "execution_errors": [],
        "with_skill": 0.8,
        "baseline": 0.4,
        "lift": 0.4,
        "dimensions": copy.deepcopy(dimensions),
        "expected_attempts": 4,
        "scored_attempts": 4,
        "conditions": {
            arm: {"execution_status": "succeeded", "execution_errors": [], "expected_attempts": 2, "scored_attempts": 2}
            for arm in ("with_skill", "without_skill")
        },
    }
    payload = {
        "schema_version": "2.0",
        "mode": "both",
        "verdict": "pass",
        "execution_status": "succeeded",
        "execution_errors": [],
        "skill_name": "demo",
        "environment": "local",
        "evaluator_version": "fixture-1",
        "publication_target": {
            "skill_name": "demo",
            "skill_digest": "sha256:" + "a" * 64,
            "skill_digest_algorithm": "skill-evaluator-source-tree/2",
        },
        "run_id": "fixture-run-001",
        "evaluated_at": "2026-01-01T00:00:00+00:00",
        "dataset_digest": "sha256:" + "b" * 64,
        "dataset_digest_algorithm": "skill-evaluator-dataset-snapshot/1",
        "dataset_summary": {"total_tasks": 2},
        "attempt_policy": {"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        "expected_attempts": 4,
        "scored_attempts": 4,
        "best_agent": "fixture-agent",
        "agents_run": ["fixture-agent"],
        "overall_score": 0.8,
        "overall_lift": 0.4,
        "dimensions": dimensions,
    }
    payload["summary"] = copy.deepcopy(payload)
    payload["agents"] = {"fixture-agent": agent}
    assert agent_eval_publication_evidence_complete(payload)
    if baseline_na:
        for dimension in [*dimensions, *agent["dimensions"]]:
            if dimension["id"] in {"discoverability", "efficiency"}:
                dimension.update(baseline=None, lift=None)
        payload["summary"]["dimensions"] = copy.deepcopy(dimensions)
    return {
        "tier3": payload,
        "incomplete_scans": [],
        "results": [{"validator": "AGENT_EVAL", "tier3": copy.deepcopy(payload), "incomplete_scans": []}],
        "publication": {"tier3": {"evidence_complete": not baseline_na}},
    }


def test_complete_execution_with_na_baseline_dimensions_is_not_incomplete():
    report = complete_report()
    assert not agent_eval_publication_evidence_complete(report["tier3"])
    original = copy.deepcopy(report)
    assert tier_manifest.tier3_execution_complete(report) is True
    assert report == original
    assert report["publication"]["tier3"]["evidence_complete"] is False


@pytest.mark.parametrize("verdict", ["pass", "neutral", "fail"])
def test_execution_complete_does_not_reinterpret_the_native_quality_verdict(verdict):
    report = complete_report()
    report["tier3"]["verdict"] = report["tier3"]["summary"]["verdict"] = verdict
    assert tier_manifest.tier3_execution_complete(report) is True


@pytest.mark.parametrize(
    "key", ["dataset_digest", "dataset_digest_algorithm", "publication_target", "run_id", "evaluated_at"]
)
def test_missing_canonical_run_provenance_remains_incomplete(key):
    report = complete_report()
    report["tier3"].pop(key)
    report["tier3"]["summary"].pop(key)
    assert tier_manifest.tier3_execution_complete(report) is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["tier3"]["agents"]["fixture-agent"]["conditions"].pop("without_skill"),
        lambda p: p["tier3"]["summary"].update(mode="with_skill"),
        lambda p: p["tier3"].update(publication_target_conflict="source changed during evaluation"),
        lambda p: p["tier3"].update(mode="with_skill"),
        lambda p: p["tier3"].update(mode="without_skill"),
        lambda p: p["tier3"].update(scored_attempts=3),
        lambda p: p["tier3"]["summary"].update(execution_status="failed"),
        lambda p: p["tier3"].update(verdict="inconclusive"),
        lambda p: p["tier3"]["agents"]["fixture-agent"]["conditions"]["without_skill"].update(scored_attempts=1),
        lambda p: p["tier3"]["agents"]["fixture-agent"]["conditions"]["with_skill"].update(
            execution_status="skipped", expected_attempts=0, scored_attempts=0
        ),
        lambda p: p["tier3"]["agents"]["fixture-agent"]["conditions"]["without_skill"].update(
            execution_errors=["fixture error"]
        ),
        lambda p: p["tier3"].update(case_selection={"total": 3, "selected": 2}),
        lambda p: p["results"][0].update(incomplete_scans=["live-agent"]),
        lambda p: p.update(incomplete_scans=["live-agent"]),
        lambda p: p.update(tier3=None),
        lambda p: p.update(results=[]),
    ],
)
def test_partial_baseline_only_or_contradictory_execution_remains_incomplete(mutate):
    report = complete_report()
    mutate(report)
    assert tier_manifest.tier3_execution_complete(report) is False
