# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import _load_suggestions_v2
from skillevaluator.tier3.harbor import report


@pytest.mark.parametrize(
    ("score", "severity"),
    [
        (0.39, "critical"),
        (0.40, "warning"),
        (0.4545, "warning"),
        (0.50, "warning"),
        (0.6741, "warning"),
        (0.7999, "warning"),
        (0.80, "ok"),
    ],
)
def test_finding_boundaries(score, severity):
    findings = report._extract_findings([{"accuracy": score, "details": {"accuracy": {"reason": "Observed result"}}}])
    assert len(findings) == 1
    assert findings[0]["severity"] == severity
    assert findings[0]["score"] == score


def _payload(findings=None):
    return report._bounded_findings_payload(
        skill_name="example",
        agent="codex",
        findings=findings or [],
        suggestions=[],
        suggestion_mode="not_generated",
        suggestions_v2=[],
    )


def test_payload_and_docs_are_in_sync():
    payload = _payload()
    assert payload["severity_thresholds"] == {
        "ok_at_or_above": report.FINDING_OK_THRESHOLD,
        "critical_below": report.FINDING_CRITICAL_THRESHOLD,
        "scale": "advisory_metric",
    }
    assert payload["findings"] == []
    assert payload["findings_total"] == payload["findings_shown"] == 0
    assert payload["findings_truncated"] is False
    docs = (Path(__file__).resolve().parents[2] / "docs/reports.mdx").read_text()
    normalized_docs = " ".join(docs.split())
    for key in {*payload, "evidence_refs_omitted", "evidence_refs_omitted_reason"}:
        assert f"`{key}`" in docs
    assert f"`ok` at {report.FINDING_OK_THRESHOLD:.2f} or above" in docs
    assert f"`critical` below {report.FINDING_CRITICAL_THRESHOLD:.2f}" in docs
    assert "| Terminal score colors | Green ≥ 0.80, amber ≥ 0.60, red below 0.60 |" in docs
    assert "`validate --tier3` never writes it" in normalized_docs


def test_payload_skips_invalid_scores():
    payload = _payload([{"score": float("nan")}, {"score": None}])
    assert payload["findings"] == []
    assert payload["findings_total"] == 2
    assert payload["findings_truncated"] is True


def test_cli_legend():
    body = report._render_findings_body([])
    assert body.plain == (
        "  Advisory metric severity: ok ≥ 0.80 · warning 0.40\u2013<0.80 · critical < 0.40. "
        "Dimension verdicts use pass ≥ 0.50.\n\n"
    )
    assert body.spans[0].style == "dim"


@pytest.mark.parametrize("include_thresholds", [False, True])
def test_suggestion_reader_accepts_old_and_new_artifacts(tmp_path, include_thresholds):
    payload = _payload()
    if not include_thresholds:
        payload.pop("severity_thresholds")
    payload["suggestions_v2"] = [{"suggestion": "Check the result", "dimension": "accuracy", "evidence_refs": []}]
    (tmp_path / "codex").mkdir()
    (tmp_path / "codex/findings.json").write_text(json.dumps(payload))
    assert _load_suggestions_v2(tmp_path, {"codex": {}}) == [
        {"metric": "accuracy", "recommendation": "Check the result", "evidence_refs": []}
    ]
