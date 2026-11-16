# SPDX-License-Identifier: Apache-2.0
"""Consolidated-table assertions must require the actual visible lift reason."""

import re

import pytest
from native_comparison_assertions import dimension_lift_labels
from test_tier3_unavailable_lift_labels import _agent, _html


@pytest.mark.parametrize("baseline", [False, True])
def test_missing_primary_overall_lift_reason_is_detected(baseline):
    payload, html = _html({"agent": _agent(baseline=baseline, judge_with="fallback")})
    assert len(dimension_lift_labels(html, ["agent"], payload["best_agent"])["agent"]) == 6
    overall = re.search(r"<tr>\s*<td><strong>Overall</strong>.*?</tr>", html, re.DOTALL)[0]
    cells = re.findall(r"<td(?:\s[^>]*)?>(.*?)</td>", overall, re.DOTALL)
    assert "no baseline" in cells[3] or "comparison unavailable" in cells[3]
    broken = html.replace(overall, overall.replace(cells[3], "N/A", 1), 1)
    with pytest.raises(AssertionError):
        dimension_lift_labels(broken, ["agent"], payload["best_agent"])
