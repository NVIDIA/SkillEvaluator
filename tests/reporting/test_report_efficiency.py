# SPDX-License-Identifier: Apache-2.0
"""Standalone reports carry compact exports and only the data their charts consume."""

import base64
import json
import re

from browser_navigation_fixtures import _agent_result

from skillevaluator.models import ValidationResult
from skillevaluator.reporting.html import HTMLReporter


def _script(output: str, script_id: str) -> str:
    match = re.search(r'<script\b[^>]*\bid="' + script_id + r'"[^>]*>(.*?)</script>', output, re.DOTALL)
    assert match is not None
    return match[1]


def test_chart_projection_excludes_unconsumed_evidence_and_keeps_measurements():
    result = _agent_result("synthetic", ["agent/雪"])
    payload = result.metadata["agent_eval"]
    payload["suggestions"] = ["UNCONSUMED-CHART-EVIDENCE-" + "x" * 40_000]
    output = HTMLReporter(include_timestamp=False).render_all([result])

    literal = re.search(r"var tier3Data = (.*)", output)
    assert literal is not None
    chart_data, _ = json.JSONDecoder().raw_decode(literal[1])
    assert set(chart_data) == {"dimensions", "agents", "trials"}
    assert "UNCONSUMED-CHART-EVIDENCE" not in literal[1]
    assert len(literal[1]) < 5_000
    assert all(set(row) == {"id"} for row in chart_data["dimensions"])
    assert set(chart_data["agents"]["agent/雪"]) == {"dimensions"}
    dimensions = chart_data["agents"]["agent/雪"]["dimensions"]
    assert next(row for row in dimensions if row["id"] == "correctness")["with_skill"] == 0.8
    assert all(set(row) <= {"id", "score", "with_skill"} for row in dimensions)
    assert chart_data["trials"][0] == {
        "agent": "agent/雪",
        "tokens": {"prompt": 100, "completion": 10, "cached": 0},
    }

    # The full retained payload remains owned by its canonical download block.
    canonical = _script(output, "tier3-full")
    if 'id="tier3-full" type="application/json" data-encoding="base64"' in output:
        canonical = base64.b64decode(canonical).decode("utf-8")
    assert json.loads(canonical) == payload
    assert json.loads(_script(output, "report-data"))["tier3"] == {"$ref": "#tier3-full"}


def test_embedded_report_json_is_compact_with_existing_ascii_and_html_escaping():
    result = ValidationResult(validator_name="SCHEMA", validator_description="雪 <script>& safety")
    result.add_success("synthetic", "雪 <script>& retained value")
    output = HTMLReporter(include_timestamp=False).render_all([result])
    raw = _script(output, "report-data")
    data = json.loads(raw)
    compact = (
        json.dumps(data, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    assert raw == compact
    assert "\\u96ea" in raw
    assert "\\u003cscript\\u003e\\u0026" in raw
    assert "雪 <script>& safety" in json.dumps(data, ensure_ascii=False)
