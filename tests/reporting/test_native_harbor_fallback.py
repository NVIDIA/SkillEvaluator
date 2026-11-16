# SPDX-License-Identifier: Apache-2.0
"""Unusable primary Harbor metadata must not hide recorded legacy links."""

from copy import deepcopy

import pytest

from skillevaluator.reporting.harbor_viewer import (
    harbor_job_links,
    normalize_agent_eval_harbor_links,
    normalize_harbor_viewer_for_display,
)


@pytest.mark.parametrize(
    "primary",
    [
        {},
        {"jobs": [], "evidence_links": []},
        {"job_url": "javascript:alert(1)", "analysis_url": "data:text/html,unsafe"},
        {"job_url": "https://[invalid"},
        {"jobs": [{"url": "file:///tmp/unavailable"}, {"url": "//example.test/no-scheme"}]},
        {"evidence_links": [{"url": "javascript:alert(1)", "agent": "untrusted"}]},
    ],
    ids=["empty", "empty-lists", "unsafe-links", "malformed-url", "unusable-jobs", "unsafe-evidence"],
)
def test_unusable_primary_viewer_falls_back_to_recorded_summary_links(primary):
    summary = {
        "job_url": "https://example.test/recorded-job",
        "analysis_url": "https://example.test/recorded-analysis",
    }
    payload = {"harbor_viewer": primary, "summary": {"harbor_viewer": summary}}
    original = deepcopy(payload)

    assert normalize_harbor_viewer_for_display(payload) == summary
    projected = normalize_agent_eval_harbor_links(payload)
    assert projected["harbor_viewer"] == summary
    assert projected["summary"]["harbor_viewer"] == summary
    assert harbor_job_links(projected) == [
        {
            "url": "https://example.test/recorded-job",
            "analysis_url": "https://example.test/recorded-analysis",
            "name": "Job 1",
        }
    ]
    assert payload == original


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        (
            {"job_url": "https://example.test/primary-job"},
            {"job_url": "https://example.test/primary-job"},
        ),
        (
            {"analysis_url": "https://example.test/primary-analysis"},
            {"analysis_url": "https://example.test/primary-analysis"},
        ),
        (
            {"jobs": [{"url": "https://example.test/primary-job", "name": "Recorded primary"}]},
            {
                "job_url": "https://example.test/primary-job",
                "jobs": [{"url": "https://example.test/primary-job", "name": "Recorded primary"}],
            },
        ),
        (
            {"evidence_links": [{"url": "https://example.test/primary-trial?step=4", "agent": "agent-a"}]},
            {
                "evidence_links": [
                    {"url": "https://example.test/primary-trial?step=4", "agent": "agent-a", "label": "Step 4"}
                ]
            },
        ),
    ],
    ids=["job", "analysis-only", "job-list", "evidence-only"],
)
def test_usable_primary_viewer_retains_existing_precedence(primary, expected):
    payload = {
        "harbor_viewer": primary,
        "summary": {
            "harbor_viewer": {
                "job_url": "https://example.test/other-job",
                "analysis_url": "https://example.test/other-analysis",
            }
        },
    }
    original = deepcopy(payload)
    assert normalize_harbor_viewer_for_display(payload) == expected
    assert normalize_agent_eval_harbor_links(payload)["harbor_viewer"] == expected
    assert "other-job" not in repr(harbor_job_links(payload))
    assert payload == original


@pytest.mark.parametrize("summary", [None, [], "not-a-summary", {}, {"harbor_viewer": []}])
def test_unusable_summary_does_not_create_viewer_links(summary):
    payload = {"harbor_viewer": {"job_url": "javascript:alert(1)"}, "summary": summary}
    original = deepcopy(payload)

    assert normalize_harbor_viewer_for_display(payload) == {}
    assert harbor_job_links(payload) == []
    assert "harbor_viewer" not in normalize_agent_eval_harbor_links(payload)
    assert payload == original


def test_summary_fallback_preserves_each_job_and_recorded_evidence_association():
    payload = {
        "harbor_viewer": {"jobs": []},
        "summary": {
            "harbor_viewer": {
                "analysis_url": "https://example.test/run-analysis",
                "jobs": [
                    {
                        "url": "https://example.test/with-skill",
                        "analysis_url": "https://example.test/shared-analysis",
                        "name": "Agent A with skill",
                    },
                    {
                        "url": "https://example.test/baseline",
                        "analysis_url": "https://example.test/shared-analysis",
                        "name": "Agent A baseline",
                    },
                    {"url": "https://example.test/baseline", "name": "Duplicate"},
                    {"url": "javascript:alert(1)", "name": "Unsafe"},
                ],
                "evidence_links": [
                    {
                        "url": "https://example.test/with-skill/trial?step=7",
                        "agent": "agent-a",
                        "condition": "with_skill",
                        "case_id": "case-one",
                    },
                    {"url": "javascript:alert(1)", "agent": "unsafe"},
                ],
            }
        },
    }
    original = deepcopy(payload)

    projected = normalize_agent_eval_harbor_links(payload)
    assert harbor_job_links(projected) == [
        {
            "url": "https://example.test/with-skill",
            "analysis_url": "https://example.test/shared-analysis",
            "name": "Agent A with skill",
        },
        {"url": "https://example.test/baseline", "name": "Agent A baseline"},
        {"name": "Run analysis", "analysis_url": "https://example.test/run-analysis"},
    ]
    assert projected["harbor_viewer"]["evidence_links"] == [
        {
            "url": "https://example.test/with-skill/trial?step=7",
            "agent": "agent-a",
            "condition": "with_skill",
            "case_id": "case-one",
            "label": "Step 7",
        }
    ]
    assert payload == original
