# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded telemetry stays scoped, bounded and distinct from missing evidence."""

import copy
import json
import re
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape


def _view(payload):
    from skillevaluator.reporting.telemetry_view import build_telemetry_view

    return build_telemetry_view(payload)


def _render(view):
    root = Path(__file__).parents[2] / "src/skillevaluator/reporting/templates"
    env = Environment(loader=FileSystemLoader(root), autoescape=select_autoescape(default=True))
    return env.get_template("telemetry.html.j2").module.render_telemetry(view)


def _arm(totals, *, status="complete", measured=2, reasons=None):
    return {
        "totals": totals,
        "coverage": {
            "expected_attempts": 2,
            "observed_attempts": 2,
            "measured_attempts": measured,
            "partial_attempts": 2 - measured,
            "status": status,
            "reason_codes": reasons or [],
            "llm_timing_sources": ["atif", "step_timestamps"],
        },
    }


def _payload():
    token_scope = {
        "with_skill": _arm(
            {
                "prompt_tokens": 100,
                "cached_tokens": 40,
                "uncached_prompt_tokens": 60,
                "completion_tokens": 20,
                "total_tokens": 120,
                "efficiency_tokens": 80,
            }
        ),
        "baseline": _arm(
            {
                "prompt_tokens": 200,
                "cached_tokens": 20,
                "uncached_prompt_tokens": 180,
                "completion_tokens": 40,
                "total_tokens": 240,
                "efficiency_tokens": 220,
            }
        ),
        "delta": {
            "prompt_tokens": -100,
            "cached_tokens": 20,
            "uncached_prompt_tokens": -120,
            "completion_tokens": -20,
            "total_tokens": -120,
            "efficiency_tokens": -140,
            "total_tokens_change_percent": -50.0,
            "status": "available",
            "reason_codes": [],
        },
    }
    latency_scope = {
        "with_skill": _arm({"llm_s": 10.5, "tool_exec_s": 0, "llm_ms_per_token": 87.5}),
        "baseline": _arm(
            {"llm_s": 20, "tool_exec_s": None, "llm_ms_per_token": 83.33},
            status="partial",
            measured=1,
            reasons=["missing_timing_evidence"],
        ),
        "delta": {
            "llm_s": None,
            "tool_exec_s": None,
            "llm_ms_per_token": None,
            "llm_s_change_percent": None,
            "llm_ms_per_token_change_percent": None,
            "status": "unavailable",
            "reason": "Baseline timing is incomplete.",
            "reason_codes": ["baseline_partial"],
        },
    }
    payload = {}
    for key, scope in (("token_usage", token_scope), ("latency_metrics", latency_scope)):
        agent = copy.deepcopy(scope)
        agent["cases"] = [{"entry_id": "case-one", **copy.deepcopy(scope)}]
        agent["arm_runtime_ms"] = {
            "with_skill": {"duration_ms": 35000},
            "without_skill": {"duration_ms": 50000},
        }
        payload[key] = {
            "source": "recorded_native",
            "schema_version": "1.0",
            "dataset": copy.deepcopy(scope),
            "agents": {"agent-a": agent},
        }
    return payload


def test_native_tables_keep_all_recorded_columns_scopes_conditions_and_reasons():
    view = _view(_payload())
    html = _render(view)
    assert view["has_tokens"] and view["has_latency"]
    for label in (
        "Dataset",
        "agent-a",
        "case-one",
        "With skill",
        "Without skill",
        "Change",
        "Prompt",
        "Cached",
        "Uncached prompt",
        "Completion",
        "Total",
        "Efficiency Tokens",
        "LLM (s)",
        "Tool (s)",
        "LLM ms/token",
        "LLM timing",
        "Arm runtime (ms)",
        "120",
        "240",
        "-120",
        "-50.00%",
        "10.5",
        "87.5",
        "35,000",
        "50,000",
        "native (ATIF)",
        "inferred (timestamps)",
        "1/2",
        "partial",
        "missing_timing_evidence",
        "baseline_partial",
        "Baseline timing is incomplete.",
    ):
        assert label in html
    assert 'id="tier3-token-usage"' in html and 'id="tier3-latency-metrics"' in html
    assert 'id="tier3-recorded-token-chart"' in html
    assert "<script" not in html


def test_token_chart_uses_both_arms_without_counting_cached_reads_twice():
    chart = _view(_payload())["token_chart"]
    assert chart == {
        "labels": ["agent-a · With skill", "agent-a · Without skill"],
        "datasets": [
            {"label": "Uncached prompt", "data": [60, 180]},
            {"label": "Cached prompt", "data": [40, 20]},
            {"label": "Completion", "data": [20, 40]},
        ],
    }


def test_chart_fallback_is_visible_until_the_shared_lifecycle_draws_chart():
    html = _render(_view(_payload()))
    canvas = re.search(r'<canvas\b[^>]*id="tier3-recorded-token-chart"[^>]*>', html).group()
    fallback = re.search(r'<p\b[^>]*id="tier3-recorded-token-chart-fallback"[^>]*>', html).group()
    assert " hidden" in canvas
    assert " hidden" not in fallback
    assert 'id="t3-recorded-token-usage-table"' in html
    assert 'id="t3-token-usage-table"' not in html
    for paragraph_id in ("tier3-recorded-token-chart-description", "tier3-recorded-token-chart-fallback"):
        paragraph = re.search(rf'<p\b[^>]*id="{paragraph_id}"[^>]*>.*?</p>', html).group()
        assert 'href="#t3-recorded-token-usage-table"' in paragraph
        assert "&quot;#" not in paragraph
    description = re.search(r'<p\b[^>]*id="tier3-recorded-token-chart-description"[^>]*>', html).group()
    assert 'class="chart-description"' in description


def test_zero_is_recorded_but_unknown_counters_and_unavailable_deltas_stay_na():
    payload = _payload()
    scope = payload["token_usage"]["dataset"]
    scope["with_skill"]["totals"] = {"total_tokens": 0, "completion_tokens": 0}
    scope["baseline"]["coverage"] = {}
    scope["delta"].update(status="unavailable", reason="Scopes differ.", total_tokens=999)
    html = _render(_view(payload))
    dataset = html.split("Dataset", 1)[1].split("agent-a", 1)[0]
    assert ">0<" in dataset
    assert "N/A" in dataset and "?/?" in dataset
    assert "Scopes differ." in dataset and "999" not in dataset


def test_partial_token_groups_are_labelled_and_missing_chart_components_remain_unknown():
    payload = _payload()
    arm = payload["token_usage"]["agents"]["agent-a"]["with_skill"]
    arm["coverage"].update(status="partial", measured_attempts=1, reason_codes=["missing_tokens"])
    chart = _view(payload)["token_chart"]
    assert chart["labels"][0] == "agent-a · With skill (partial)"
    del arm["totals"]["uncached_prompt_tokens"]
    chart = _view(payload)["token_chart"]
    assert chart["datasets"][0]["data"][0] is None
    assert "missing_tokens" in _render(_view(payload))


def test_recorded_timing_precision_is_preserved_without_recalculation():
    payload = _payload()
    payload["latency_metrics"]["dataset"]["with_skill"]["totals"]["llm_ms_per_token"] = 87.123456789
    assert "87.123456789" in _render(_view(payload))


def test_native_partial_delta_preserves_absolute_change_when_percent_is_unavailable():
    payload = _payload()
    delta = payload["token_usage"]["dataset"]["delta"]
    delta.update(
        status="partial",
        total_tokens=120,
        total_tokens_change_percent=None,
        reason="Baseline is zero; percentage unavailable.",
        reason_codes=["baseline_zero"],
    )
    html = _render(_view(payload))
    dataset = html.split("Dataset", 1)[1].split("agent-a", 1)[0]
    assert "+120" in dataset and "baseline_zero" in dataset
    assert "Baseline is zero; percentage unavailable." in dataset
    assert "%" not in dataset


def test_view_preserves_each_family_within_byte_limit_for_escape_heavy_text():
    payload = _payload()
    for family in ("token_usage", "latency_metrics"):
        agent = payload[family]["agents"].pop("agent-a")
        for scope in (agent, agent["cases"][0], payload[family]["dataset"]):
            for condition in ("with_skill", "baseline"):
                scope[condition]["coverage"]["reason_codes"] = ["\U0001f600<>\"'&" * 100] * 100
            scope["delta"]["reason"] = "\U0001f600<>\"'&" * 100
        agent["cases"] *= 1000
        payload[family]["agents"] = {f"agent-{i}" + "\U0001f600<>\"'&" * 100: agent for i in range(1000)}
    view = _view(payload)
    assert len(json.dumps(view).encode()) <= 256 * 1024
    assert len(_render(view).encode()) <= 512 * 1024
    assert "Display text shortened" in _render(view)


def test_combined_projection_byte_bound_includes_unicode_agent_labels_and_chart():
    payload = _payload()
    for family in ("token_usage", "latency_metrics"):
        agent = payload[family]["agents"]["agent-a"]
        agent["cases"] *= 1000
        payload[family]["agents"] = {"\U0001f600" * 240 + str(i): agent for i in range(32)}
    view = _view(payload)
    assert len(json.dumps(view).encode()) <= 256 * 1024


def test_html_expansion_of_escape_heavy_reasons_stays_bounded():
    payload = _payload()
    for family in ("token_usage", "latency_metrics"):
        agent = payload[family]["agents"]["agent-a"]
        for scope in (agent, agent["cases"][0], payload[family]["dataset"]):
            for condition in ("with_skill", "baseline"):
                scope[condition]["coverage"]["reason_codes"] = ["<" * 240] * 8
            scope["delta"]["reason"] = "<" * 240
        agent["cases"] *= 1000
        payload[family]["agents"] = {f"agent-{i}": agent for i in range(32)}
    html = _render(_view(payload))
    assert len(html.encode()) <= 512 * 1024
    assert "omitted" in html


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), -1, True, "123", {}, 10**10000],
    ids=["nan", "infinite", "negative", "boolean", "string", "mapping", "huge-int"],
)
def test_bad_numeric_values_never_become_recorded_measurements(bad):
    payload = {
        "token_usage": {"dataset": {"with_skill": _arm({"total_tokens": bad})}},
        "latency_metrics": {"dataset": {"with_skill": _arm({"llm_s": bad})}},
    }
    html = _render(_view(payload))
    assert "N/A" in html
    assert ">nan<" not in html and ">inf<" not in html and ">True<" not in html
    assert ">123<" not in html and ">-1<" not in html


def test_untrusted_text_is_escaped_and_projection_has_no_mutable_canonical_aliases():
    payload = _payload()
    attack = '<img src=x onerror="alert(1)">\ud800'
    payload["token_usage"]["agents"][attack] = payload["token_usage"]["agents"].pop("agent-a")
    payload["latency_metrics"]["dataset"]["delta"]["reason"] = attack
    before = copy.deepcopy(payload)
    view = _view(payload)
    html = _render(view)
    assert "<img" not in html and "&lt;img" in html
    assert "\ud800" not in html
    json.dumps(view, ensure_ascii=False, allow_nan=False).encode("utf-8")
    view["token_chart"]["datasets"][0]["data"].append(12345)
    view["token_sections"].clear()
    assert payload == before


def test_all_telemetry_text_is_redacted_before_shortening_and_controls_are_removed():
    payload = _payload()
    secret = "syntheticcredential123"
    unsafe = f"\x1b[31mBearer {secret}\x1b[0m\x00\ud800\u202e"
    for family in ("token_usage", "latency_metrics"):
        usage = payload[family]
        usage["agents"][unsafe] = usage["agents"].pop("agent-a")
        agent = usage["agents"][unsafe]
        usage["source"] = unsafe
        agent["cases"][0]["entry_id"] = unsafe
        for condition in ("with_skill", "baseline"):
            agent[condition]["coverage"]["reason_codes"] = [unsafe]
        usage["dataset"]["delta"]["reason"] = "x " * 110 + f" api_key={secret}"
    before = copy.deepcopy(payload)
    view = _view(payload)
    html = _render(view)
    projected = json.dumps(view, ensure_ascii=False)
    assert secret not in projected and secret not in html
    assert "redacted" in projected and "redacted" in html
    assert "\x1b" not in projected and "\x00" not in projected and "\ud800" not in projected
    assert "\u202e" not in projected
    assert payload == before


def test_large_inputs_are_bounded_without_walking_all_cases_or_extra_payloads():
    class BoundedCases(list):
        def __iter__(self):
            raise AssertionError("unbounded list traversal")

    payload = _payload()
    agent = payload["token_usage"]["agents"]["agent-a"]
    agent["cases"] = BoundedCases([{"entry_id": "x" * 10000, **copy.deepcopy(agent["cases"][0])}] * 100000)
    payload["token_usage"]["agents"] = {f"agent-{i}": agent for i in range(1000)}
    payload["token_usage"]["source"] = "x" * 1000000
    payload["irrelevant"] = {"recursive": payload}
    view = _view(payload)
    assert len(json.dumps(view).encode()) <= 256 * 1024
    assert len(_render(view).encode()) <= 512 * 1024
    assert "omitted" in _render(view).lower()
    assert len(view["token_chart"]["labels"]) <= 64


@pytest.mark.parametrize("payload", [None, [], "oops", {}, {"agents": {"agent-a": {"resources": {"tokens": 0}}}}])
def test_absent_native_telemetry_keeps_public_resources_separate(payload):
    view = _view(payload)
    assert view == {
        "token_sections": [],
        "latency_sections": [],
        "token_chart": None,
        "has_tokens": False,
        "has_latency": False,
    }
    assert _render(view).strip() == ""
