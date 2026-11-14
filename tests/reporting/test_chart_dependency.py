# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reports carry an offline, licensed chart runtime and accessible fallbacks."""

import hashlib
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting import html as html_module


class _Report(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.external: list[dict[str, str | None]] = []
        self.elements: dict[str, tuple[str, dict[str, str | None]]] = {}
        self.scripts: dict[str, str] = {}
        self.script_id: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.elements[element_id] = tag, attributes
        if tag == "script":
            if attributes.get("src"):
                self.external.append(attributes)
            self.script_id = element_id
            if element_id:
                self.scripts[element_id] = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.script_id = None

    def handle_data(self, data: str) -> None:
        if self.script_id:
            self.scripts[self.script_id] += data


def _tier3_html(*, agent_count: int = 1) -> str:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = build_agent_eval_payload(
        "example",
        {
            agent: {
                "execution_status": "succeeded",
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0},
                "rewards": [
                    {
                        "entry_id": "case-1",
                        "security": 1.0,
                        "_traj": {"steps": 4, "prompt_tokens": 10, "completion_tokens": 4, "cached_tokens": 2},
                    }
                ],
            }
            for agent in ["agent", "second-agent"][:agent_count]
        },
        use_llm_judge=False,
    )
    return HTMLReporter(include_timestamp=False).render_all([result])


def _parse(html: str) -> _Report:
    parser = _Report()
    parser.feed(html)
    return parser


def test_tier3_embeds_verified_chart_runtime_without_network_or_source_map() -> None:
    report = _parse(_tier3_html())
    assert report.external == []
    runtime = report.scripts["tier3-chart-runtime"]
    assert "Chart.js v4.5.1" in runtime
    assert "@kurkle/color v0.3.2" in runtime
    # SRI-verified upstream bytes with the source-map directive removed and one final newline.
    assert hashlib.sha256(runtime.encode()).hexdigest() == (
        "84d0e233daba702b8f77d669d8c137cad36d441a10f200b6f2d3ab553bdfcf6b"
    )
    assert "sourceMappingURL" not in runtime
    assert "</script" not in runtime.lower()


def test_reports_retain_both_bundled_library_licenses() -> None:
    html = _tier3_html()
    assert 'id="tier3-chart-notices"' in html
    assert "Copyright (c) 2014-2024 Chart.js Contributors" in html
    assert "Copyright (c) 2018-2021 Jukka Kurkela" in html
    assert html.count("Permission is hereby granted, free of charge") == 2
    assert html.count("THE SOFTWARE IS PROVIDED") == 2


@pytest.mark.parametrize("agent_count", [1, 2])
def test_chart_canvases_explain_data_and_keep_visible_failure_fallback(agent_count: int) -> None:
    html = _tier3_html(agent_count=agent_count)
    report = _parse(html)
    canvases = {key: attrs for key, (tag, attrs) in report.elements.items() if tag == "canvas"}
    token_chart = "tier3-recorded-token-chart" if "tier3-recorded-token-chart" in canvases else "tier3-token-chart"
    assert canvases.keys() == {"tier3-radar", "tier3-metric-bar", token_chart, "tier3-steps-chart"}
    for chart_id, attrs in canvases.items():
        assert attrs["role"] == "img"
        assert attrs["aria-label"]
        assert "hidden" in attrs
        description = attrs["aria-describedby"]
        assert description and description in report.elements
        _, fallback = report.elements[chart_id + "-fallback"]
        assert "hidden" not in fallback
    references = re.findall(r'class="chart-description"[^>]*>[^<]*<a href="#([^"]+)"', html)
    assert len(references) == 4
    assert all(report.elements[reference][0] == "table" for reference in references)
    assert references[:2] == ["tier3-dimension-table" if agent_count == 1 else "tier3-multi-agent-dimension-table"] * 2


def test_tier1_does_not_load_or_embed_chart_resources(monkeypatch) -> None:
    def forbidden():
        raise AssertionError("Tier 1 must not load chart assets")

    monkeypatch.setattr(html_module, "_load_chart_assets", forbidden)
    html = HTMLReporter(include_timestamp=False).render_all([ValidationResult(validator_name="QUALITY")])
    report = _parse(html)
    assert "tier3-chart-runtime" not in report.scripts
    assert "Chart.js v4.5.1" not in html
    assert "tier3-chart-notices" not in html


def test_chart_package_resources_are_cached_across_report_instances() -> None:
    html_module._load_chart_assets.cache_clear()
    try:
        _tier3_html()
        _tier3_html()
        info = html_module._load_chart_assets.cache_info()
        assert info.misses == 1
        assert info.hits == 1
    finally:
        html_module._load_chart_assets.cache_clear()


def test_chart_initialization_failure_keeps_fallback_and_recovers() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the generated chart controller")
    html = _tier3_html()
    controller = re.search(r"\(function\(\)\{\s+var tier3Data = .*?\}\)\(\);", html, re.DOTALL)
    assert controller
    script = r"""
const vm = require('node:vm');
const controller = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const ids = ['tier3-radar', 'tier3-metric-bar', 'tier3-recorded-token-chart', 'tier3-steps-chart'];
const elements = {};
ids.forEach(id => {
    elements[id] = {id, hidden: true};
    elements[id + '-fallback'] = {hidden: false};
});
const listeners = {};
let theme = 'dark';
const context = {
    window: {addEventListener: (name, fn) => {listeners[name] = fn;}},
    document: {
        documentElement: {getAttribute: () => theme, setAttribute: (name, value) => {theme = value;}},
        getElementById: id => elements[id] || null,
        addEventListener: (name, fn) => {listeners[name] = fn;}
    },
    getComputedStyle: () => ({getPropertyValue: () => theme === 'dark' ? '#e2e8f0' : '#0f172a'})
};
vm.createContext(context);
vm.runInContext(controller, context);
listeners.DOMContentLoaded();
const missing = ids.map(id => [elements[id].hidden, elements[id + '-fallback'].hidden]);
context.Chart = class {constructor() {throw new Error('Canvas unavailable');}};
context.window.rebuildTier3Charts();
const failed = ids.map(id => [elements[id].hidden, elements[id + '-fallback'].hidden]);
let released = 0;
// Like Chart.js 4.5.1: a context-less instance is registered with canvas=null,
// so getChart(canvas) cannot find it; only the constructed instance can release it.
context.Chart = class {
    constructor() {this.ctx = null;}
    destroy() {released++;}
    static getChart() {return undefined;}
};
context.window.rebuildTier3Charts();
const missingContext = ids.map(id => [elements[id].hidden, elements[id + '-fallback'].hidden]);
let built = 0, destroyed = 0;
const printAnimations = [];
const colors = [];
context.Chart = class {
    constructor(canvas, config) {
        this.ctx = {};
        built++;
        printAnimations.push(config.options.animation);
        colors.push(config.options.plugins.legend.labels.color);
    }
    destroy() {destroyed++;}
};
context.window.rebuildTier3Charts();
const recovered = ids.map(id => [elements[id].hidden, elements[id + '-fallback'].hidden]);
listeners.beforeprint();
listeners.afterprint();
process.stdout.write(JSON.stringify({missing, failed, missingContext, released, recovered, built, destroyed, printAnimations, colors, theme}));
"""
    result = subprocess.run(
        [node, "-e", script], input=json.dumps(controller.group()), capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    states = json.loads(result.stdout)
    assert states["missing"] == [[True, False]] * 4
    assert states["failed"] == [[True, False]] * 4
    assert states["missingContext"] == [[True, False]] * 4
    assert states["released"] == 4
    assert states["recovered"] == [[False, True]] * 4
    assert states["built"] == 12
    assert states["destroyed"] == 8
    assert states["printAnimations"][4:8] == [False] * 4
    assert states["colors"] == ["#e2e8f0"] * 4 + ["#0f172a"] * 4 + ["#e2e8f0"] * 4
    assert states["theme"] == "dark"


def test_charts_without_dimension_scores_link_to_an_openable_download_section() -> None:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = build_agent_eval_payload(
        "example",
        {"agent": {"execution_status": "failed", "expected_attempts": 1, "scored_attempts": 0, "with_skill": {}}},
        use_llm_judge=False,
    )
    html = HTMLReporter(include_timestamp=False).render_all([result])
    report = _parse(html)
    links = re.findall(r'class="chart-description"[^>]*>[^<]*<a href="#([^"]+)" onclick="([^"]+)"', html)
    dimension_links = [link for link in links if link[0] != "t3-trials-table"][:2]
    assert [target for target, _ in dimension_links] == ["tier3-json-downloads"] * 2
    assert report.elements["tier3-json-downloads"][0] == "details"
    # The section sits in the hidden Diagnostics sub-tab; the static handler reveals it.
    assert all("openTier3Reference" in handler for _, handler in dimension_links)
    assert "function openTier3Reference(id)" in html


def test_missing_chart_asset_keeps_the_report_and_visible_fallbacks(monkeypatch) -> None:
    class Missing:
        def joinpath(self, *_parts):
            return self

        def read_text(self, **_kwargs):
            raise FileNotFoundError("chart.umd.min.js")

    html_module._load_chart_assets.cache_clear()
    monkeypatch.setattr(html_module.resources, "files", lambda _package: Missing())
    try:
        html = _tier3_html()
    finally:
        html_module._load_chart_assets.cache_clear()
    report = _parse(html)
    assert "tier3-chart-runtime" not in report.scripts
    assert "tier3-chart-notices" not in html
    _, fallback = report.elements["tier3-radar-fallback"]
    assert "hidden" not in fallback


def test_chart_reference_opens_ancestor_evidence_and_reuses_shared_navigation() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the generated chart reference handler")
    html = _tier3_html()
    helper = re.search(r"function openTier3Reference\(id\).*?(?=\s*function showTier3SubTab)", html, re.DOTALL)
    assert helper
    script = r"""
const vm = require('vm');
const state = {lookups:0, focused:false, hash:null, tab:null, routed:[]};
const outer = {tagName:'DETAILS', open:false, parentElement:null};
const inner = {tagName:'DETAILS', open:false, parentElement:outer};
const page = {id:'tier3-page-trials', classList:{contains:()=>false}};
const target = {
    tagName:'TABLE', parentElement:inner, closest:()=>page,
    setAttribute:(key,value)=>{state[key]=value;},
    focus:()=>{state.focused=true;},
    scrollIntoView:()=>{}
};
const context = {
    document:{getElementById:()=>{state.lookups++; return target;}, querySelectorAll:()=>[{dataset:{tier3Tab:'trials'}}]},
    history:{replaceState:(_data,_title,hash)=>{state.hash=hash;}},
    showTier3SubTab:(tab)=>{state.tab=tab;}
};
vm.createContext(context);
vm.runInContext(HELPER, context);
const result = context.openTier3Reference('evidence-table');
const standalone = {outer:outer.open,inner:inner.open,focused:state.focused,tabindex:state.tabindex,hash:state.hash,tab:state.tab,result};
const before = state.lookups;
context.openReportDestination = id => {state.routed.push(id); return false;};
const routed = context.openTier3Reference('evidence-table');
process.stdout.write(JSON.stringify({standalone,routed,targets:state.routed,extraLookups:state.lookups-before}));
""".replace("HELPER", json.dumps(helper.group()))
    states = json.loads(subprocess.run([node, "-e", script], check=True, capture_output=True, text=True).stdout)
    assert states["standalone"] == {
        "outer": True,
        "inner": True,
        "focused": True,
        "tabindex": "-1",
        "hash": "#evidence-table",
        "tab": "trials",
        "result": False,
    }
    assert states["routed"] is False
    assert states["targets"] == ["evidence-table"]
    assert states["extraLookups"] == 0
