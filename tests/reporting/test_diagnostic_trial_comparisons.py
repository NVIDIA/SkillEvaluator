# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed-arm trial diagnostics never become HTML skill comparisons."""

import base64
import json
import re
from copy import deepcopy

import pytest
from tests.reporting.test_dimension_explanations import _agent

from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, build_agent_eval_payload
from skillevaluator.reporting import html as html_module
from skillevaluator.reporting.html import HTMLReporter


def _trial_agent(failed_arm=None):
    info = _agent()
    if failed_arm is not None:
        key = "rewards" if failed_arm == "with_skill" else "rewards_baseline"
        info[f"diagnostic_{key}"] = [
            *info[key],
            {"entry_id": "case-1", "trial_id": "case-1__unscored", "_unscored": True},
        ]
        info[key] = []
        info["execution_status"] = "failed"
        info["conditions"][failed_arm] = {"execution_status": "failed"}
    return info


def _render(agents):
    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)
    original = deepcopy(payload)
    result = _validation_result_from_payload(payload)
    output = HTMLReporter(include_timestamp=False).render_all([result])
    match = re.search(r'<script\b([^>]*\bid="tier3-full"[^>]*)>(.*?)</script>', output, re.DOTALL)
    assert match is not None
    data = base64.b64decode(match[2]).decode() if 'data-encoding="base64"' in match[1] else match[2]
    assert json.loads(data) == original
    assert payload == original
    return output, payload


@pytest.mark.parametrize("failed_arm", [None, "with_skill", "without_skill"])
def test_html_compares_trials_only_when_both_arms_are_scored(failed_arm):
    output, payload = _render({"agent": _trial_agent(failed_arm)})

    # Pairing retained diagnostic rewards can still produce per-trial deltas.
    assert any(row.get("lift_scores") for row in payload["trials"])
    table = re.search(r'<table\b[^>]*id="t3-trials-table".*?<tbody>(.*?)</tbody>', output, re.DOTALL)
    assert table is not None
    assert len(re.findall(r"<tr\b", table[1])) == (3 if failed_arm else 2)
    assert ('id="tier3-lift-per-case"' in output) is (failed_arm is None)
    if failed_arm:
        key = "trials" if failed_arm == "with_skill" else "trials_baseline"
        assert all(row["diagnostic_only"] for row in payload["agents"]["agent"][key])
        assert any(row.get("scored") is False for row in payload["agents"]["agent"][key])


@pytest.mark.parametrize("failed_arm", ["with_skill", "without_skill"])
def test_html_keeps_complete_peer_trial_comparisons(failed_arm):
    output, _payload = _render({"failed": _trial_agent(failed_arm), "completed": _trial_agent()})

    comparison = re.search(r'<details\b[^>]*id="tier3-lift-per-case".*?</details>', output, re.DOTALL)
    assert comparison is not None
    assert "<h4>completed " in comparison[0]
    assert "<h4>failed " not in comparison[0]


def test_omitting_diagnostic_baseline_preview_rows_cannot_restore_a_comparison(monkeypatch):
    bounded_preview = html_module._bounded_tier3_preview

    def omit_baseline_rows(payload):
        preview, notice = bounded_preview(payload)
        for agent in preview["agents"].values():
            agent["trials_baseline"] = []
        return preview, notice

    monkeypatch.setattr(html_module, "_bounded_tier3_preview", omit_baseline_rows)
    output, _payload = _render({"agent": _trial_agent("without_skill")})

    assert 'id="tier3-lift-per-case"' not in output
