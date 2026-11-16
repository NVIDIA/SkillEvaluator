# SPDX-License-Identifier: Apache-2.0
"""Recorded upload links survive display projection and offline reconstruction."""

import base64
import json
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.reporting.harbor_viewer import harbor_job_links, normalize_agent_eval_harbor_links
from skillevaluator.reporting.retained import retained_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


@pytest.mark.parametrize("status", ["succeeded", "incomplete", "failed"])
def test_upload_only_payload_keeps_each_recorded_job_in_display(status):
    payload = {
        "execution_status": status,
        "harbor_viewer": {
            "status": "partial",
            "uploads": [
                {
                    "viewer_url": "https://viewer.example.test/jobs/with",
                    "analysis_url": "https://viewer.example.test/review/with",
                    "uploaded_job_name": "With skill",
                },
                {"error": "Upload did not complete"},
                {
                    "job_url": "https://viewer.example.test/jobs/baseline?view=logs&case=one#step-2",
                    "job_name": "Baseline",
                },
                {"viewer_url": "javascript:alert(1)"},
                {"viewer_url": "https://[invalid"},
                None,
            ],
        },
    }
    original = deepcopy(payload)

    displayed = normalize_agent_eval_harbor_links(payload)

    assert harbor_job_links(displayed) == [
        {
            "url": "https://viewer.example.test/jobs/with",
            "analysis_url": "https://viewer.example.test/review/with",
            "name": "With skill",
        },
        {
            "url": "https://viewer.example.test/jobs/baseline?view=logs&case=one#step-2",
            "name": "Baseline",
        },
    ]
    assert payload == original


def test_uploads_enrich_only_the_same_job_without_replacing_canonical_metadata():
    payload = {
        "harbor_viewer": {
            "analysis_url": "https://viewer.example.test/review/run",
            "jobs": [
                {
                    "url": "https://viewer.example.test/jobs/with",
                    "analysis_url": "https://viewer.example.test/review/canonical",
                    "name": "Canonical job",
                },
                {"url": "https://viewer.example.test/jobs/baseline", "name": "Canonical baseline"},
            ],
            "uploads": [
                {
                    "viewer_url": "https://viewer.example.test/jobs/with",
                    "analysis_url": "https://viewer.example.test/review/duplicate",
                    "uploaded_job_name": "Duplicate name",
                },
                {
                    "viewer_url": "https://viewer.example.test/jobs/baseline",
                    "analysis_url": "https://viewer.example.test/review/baseline",
                },
                {"viewer_url": "https://viewer.example.test/jobs/second", "uploaded_job_name": "Second job"},
            ],
        },
    }
    original = deepcopy(payload)

    assert harbor_job_links(payload) == [
        {
            "url": "https://viewer.example.test/jobs/with",
            "analysis_url": "https://viewer.example.test/review/canonical",
            "name": "Canonical job",
        },
        {
            "url": "https://viewer.example.test/jobs/baseline",
            "analysis_url": "https://viewer.example.test/review/baseline",
            "name": "Canonical baseline",
        },
        {
            "url": "https://viewer.example.test/jobs/second",
            "name": "Second job",
        },
        {"name": "Run analysis", "analysis_url": "https://viewer.example.test/review/run"},
    ]
    assert payload == original


@pytest.mark.parametrize("uploads", [None, [], {}, True, 1, "malformed"])
def test_malformed_uploads_do_not_hide_existing_canonical_links(uploads):
    payload = {
        "harbor_viewer": {
            "job_url": "https://viewer.example.test/jobs/retained",
            "analysis_url": "https://viewer.example.test/review/retained",
            "uploads": uploads,
        },
    }
    original = deepcopy(payload)

    assert harbor_job_links(payload) == [
        {
            "url": "https://viewer.example.test/jobs/retained",
            "analysis_url": "https://viewer.example.test/review/retained",
            "name": "Job 1",
        }
    ]
    assert payload == original


def _write_retained_run(run: Path, viewer: object) -> Path:
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps({"skill_name": "example-skill", "run_id": "retained-run", "harbor_viewer": viewer}),
        encoding="utf-8",
    )
    for arm in ("with-skill", "without-skill"):
        condition = run / "opencode" / arm
        trial = condition / "trials" / "case__trial"
        trial.mkdir(parents=True)
        scores = dict.fromkeys(DEFAULT_METRICS, 1.0)
        (condition / "summary.json").write_text(
            json.dumps(
                {
                    "scores": scores,
                    "metrics": list(DEFAULT_METRICS),
                    "num_trials": 1,
                    "num_reward_rows": 1,
                    "execution_status": "succeeded",
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                }
            ),
            encoding="utf-8",
        )
        (trial / "reward.json").write_text(json.dumps({"entry_id": "case", **scores}), encoding="utf-8")
    return run


def test_directory_reconstruction_keeps_links_from_recorded_engine_result(tmp_path):
    run = _write_retained_run(
        tmp_path / "run",
        {"uploads": [{"viewer_url": "https://viewer.example.test/jobs/recorded", "uploaded_job_name": "Recorded"}]},
    )
    original = (run / "result.json").read_bytes()

    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)

    assert result is not None
    assert harbor_job_links(result.metadata["agent_eval"]) == [
        {
            "url": "https://viewer.example.test/jobs/recorded",
            "analysis_url": "https://viewer.example.test/jobs/recorded?tab=analysis",
            "name": "Recorded",
        }
    ]
    assert (run / "result.json").read_bytes() == original


@pytest.mark.parametrize("status", ["succeeded", "incomplete", "failed"])
def test_live_engine_links_win_per_job_and_keep_distinct_recorded_jobs(tmp_path, status):
    run = _write_retained_run(
        tmp_path / "run",
        {
            "uploads": [
                {
                    "viewer_url": "https://viewer.example.test/jobs/shared",
                    "analysis_url": "https://viewer.example.test/review/stale",
                    "uploaded_job_name": "Recorded shared",
                },
                {
                    "viewer_url": "https://viewer.example.test/jobs/baseline",
                    "analysis_url": "https://viewer.example.test/review/baseline",
                    "uploaded_job_name": "Recorded baseline",
                },
            ]
        },
    )
    live = {
        "execution_status": status,
        "harbor_viewer": {
            "uploads": [
                {
                    "viewer_url": "https://viewer.example.test/jobs/shared",
                    "analysis_url": "https://viewer.example.test/review/live",
                    "uploaded_job_name": "Live shared",
                }
            ]
        },
    }
    original = deepcopy(live)

    result = agent_eval_result_from_directory(tmp_path, run, engine_result=live, use_llm_judge=False)

    assert result is not None
    assert harbor_job_links(result.metadata["agent_eval"]) == [
        {
            "url": "https://viewer.example.test/jobs/shared",
            "analysis_url": "https://viewer.example.test/review/live",
            "name": "Live shared",
        },
        {
            "url": "https://viewer.example.test/jobs/baseline",
            "analysis_url": "https://viewer.example.test/review/baseline",
            "name": "Recorded baseline",
        },
    ]
    assert live == original


@pytest.mark.parametrize("viewer", [None, "malformed", {"uploads": True}, {"uploads": 1}])
def test_unusable_live_viewer_keeps_safe_recorded_uploads(tmp_path, viewer):
    run = _write_retained_run(
        tmp_path / "run", {"uploads": [{"viewer_url": "https://viewer.example.test/jobs/recorded"}]}
    )

    result = agent_eval_result_from_directory(
        tmp_path, run, engine_result={"harbor_viewer": viewer}, use_llm_judge=False
    )

    assert result is not None
    assert [job["url"] for job in harbor_job_links(result.metadata["agent_eval"])] == [
        "https://viewer.example.test/jobs/recorded"
    ]


class _HarborSurfaces(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.stack = []
        self.links = {"tile": set(), "overview": set()}
        self.canonical = ""
        self.encoding = None
        self.in_canonical = False
        self.feed(html)

    def handle_starttag(self, tag, items):
        attrs = dict(items)
        if tag == "script" and attrs.get("id") == "tier3-full":
            self.in_canonical = True
            self.encoding = attrs.get("data-encoding")
        if tag == "a":
            for _, parent in self.stack:
                if "tier-card" in parent.get("class", "").split() and parent.get("data-target-tab") == "tier3":
                    self.links["tile"].add(attrs.get("href"))
                if parent.get("id") == "tier3-page-overview":
                    self.links["overview"].add(attrs.get("href"))
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_canonical = False
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, value):
        if self.in_canonical:
            self.canonical += value

    def download(self):
        value = base64.b64decode(self.canonical) if self.encoding == "base64" else self.canonical
        return json.loads(value)


@pytest.mark.parametrize("carrier", ["canonical-uploads", "recorded-directory"])
def test_recorded_uploads_reach_both_html_surfaces_without_changing_download(tmp_path, carrier):
    viewer = {
        "uploads": [
            {
                "viewer_url": "https://viewer.example.test/jobs/report",
                "analysis_url": "https://viewer.example.test/review/report",
                "uploaded_job_name": "Recorded report job",
            }
        ]
    }
    run = _write_retained_run(tmp_path / "run", viewer if carrier == "recorded-directory" else None)
    original_result = (run / "result.json").read_bytes()
    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)
    assert result is not None
    if carrier == "canonical-uploads":
        result.metadata["agent_eval"]["harbor_viewer"] = deepcopy(viewer)
    original_payload = deepcopy(result.metadata["agent_eval"])
    retained = retained_results(json.loads(JSONReporter().render_all([result])))

    rendered = HTMLReporter(
        include_timestamp=False,
        requested_tiers=(1, 2, 3),
        displayed_tiers=(1, 2, 3),
    ).render_all(retained)

    surfaces = _HarborSurfaces(rendered)
    for links in surfaces.links.values():
        assert {"https://viewer.example.test/jobs/report", "https://viewer.example.test/review/report"} <= links
    assert surfaces.download() == original_payload
    assert retained[0].metadata["agent_eval"] == original_payload
    assert result.metadata["agent_eval"] == original_payload
    assert (run / "result.json").read_bytes() == original_result
