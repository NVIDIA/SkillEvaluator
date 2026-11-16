# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from copy import deepcopy
from html.parser import HTMLParser

import pytest
from tests.reporting.report_markup_assertions import without_scripts

from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, build_agent_eval_payload
from skillevaluator.reporting import HTMLReporter
from skillevaluator.tier3.harbor.run_budget import BudgetController


class _ReportText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_style = False
        self.parts = []

    def handle_starttag(self, tag, _attrs):
        if tag == "style":
            self.in_style = True

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False

    def handle_data(self, data):
        if not self.in_style:
            self.parts.append(data)


@pytest.mark.parametrize("change", [None, "error", "started", "reason", "empty", "missing"])
def test_html_preserves_only_fully_denied_budget_context(change):
    budget = BudgetController(max_attempts_total=0, planned=4)
    for arm in ("with", "without"):
        for attempt in (1, 2):
            assert not budget.admit(agent="opencode", arm=arm, case_id="case", attempt=attempt)
    condition = {
        "execution_status": "skipped",
        "skip_reason": "budget_exhausted",
        "expected_attempts": 0,
        "scored_attempts": 0,
    }
    payload = build_agent_eval_payload(
        "skill",
        {
            "opencode": {
                "execution_status": "skipped",
                "execution_errors": [],
                "conditions": {"with_skill": condition, "without_skill": condition},
            }
        },
        run_config={"budget": budget.metadata()},
        use_llm_judge=False,
    )
    if change == "error":
        payload["execution_errors"] = ["real execution failure"]
    elif change == "started":
        payload["budget"]["started_attempts"] = 1
    elif change == "reason":
        payload["budget"]["not_run"][0]["reason"] = "execution_failed"
    elif change == "empty":
        payload["budget"]["not_run"] = []
    elif change == "missing":
        payload.pop("budget")
    original = deepcopy(payload)
    validation = _validation_result_from_payload(payload)
    rendered = HTMLReporter(report_scope="tier3").render_all([validation])
    # Check every reachable report view, excluding non-rendered script/style data.
    text = _ReportText()
    text.feed(without_scripts(rendered))
    text.close()
    visible_text = " ".join("".join(text.parts).split())
    assert ("Budget exhausted (attempts 0 of 0)" in visible_text) is (change is None)
    # Stale diagnostics remain in the original evidence export, never in the view.
    canonical = re.search(r'<script[^>]*id="tier3-full"[^>]*>(.*?)</script>', rendered, re.DOTALL)
    assert canonical is not None
    assert json.loads(canonical[1]) == original
    assert payload == original
    if change is None:
        assert not validation.errors
        assert 'data-verdict="incomplete"' in rendered
        assert "Live quality: Incomplete." in visible_text
        assert "with-skill not run due to budget" in visible_text
