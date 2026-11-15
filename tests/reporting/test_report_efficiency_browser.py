# SPDX-License-Identifier: Apache-2.0
"""Offline browser regressions for chart lifecycle and complete filtered prints."""

import json
import os

import pytest
import test_report_browser as report_browser
from test_report_browser import _open, _subtab, _tier

from skillevaluator.reporting.html import HTMLReporter

browser = report_browser.browser
page = report_browser.page
reports = report_browser.reports

pytestmark = pytest.mark.skipif(
    os.environ.get("SKILLEVALUATOR_BROWSER_TESTS") != "1",
    reason="Enable offline Chromium checks with SKILLEVALUATOR_BROWSER_TESTS=1",
)


def _track_chart_constructions(page, theme="dark"):
    page.add_init_script(
        """
        window.chartConstructions = [];
        let runtime;
        Object.defineProperty(window, 'Chart', {
            configurable: true,
            get: () => runtime,
            set: value => {
                runtime = new Proxy(value, {
                    construct(target, args, newTarget) {
                        chartConstructions.push({id:args[0].id,theme:document.documentElement.dataset.theme});
                        return Reflect.construct(target, args, newTarget);
                    }
                });
            }
        });
        try { localStorage.setItem('skillevaluator-theme', SAVED_THEME); } catch (error) {}
    """.replace("SAVED_THEME", json.dumps(theme))
    )


def test_initial_visible_charts_apply_the_saved_theme_before_constructing(page, reports):
    directory, _, _ = reports
    _track_chart_constructions(page, theme="light")
    _open(page, directory, "canonical")
    events = page.evaluate("chartConstructions")
    assert {event["id"] for event in events} == {"tier3-radar", "tier3-metric-bar"}
    assert len(events) == 2
    assert all(event["theme"] == "light" for event in events)


def test_hidden_charts_wait_for_visibility_and_reuse_instances_across_theme_and_tabs(page, reports):
    directory, _, _ = reports
    _track_chart_constructions(page)
    _open(page, directory)
    assert page.evaluate("chartConstructions") == []

    _tier(page, "tier3")
    page.wait_for_function("chartConstructions.length === 2")
    assert page.evaluate("chartConstructions.map(row=>row.id).sort()") == ["tier3-metric-bar", "tier3-radar"]
    assert page.evaluate("chartConstructions.every(row=>row.theme==='dark')")
    assert page.locator("#tier3-radar").evaluate("node=>Chart.getChart(node).data.datasets[0].data").count(0.94) >= 1
    dark_color = page.locator("#tier3-radar").evaluate("node=>Chart.getChart(node).options.scales.r.ticks.color")
    page.evaluate("setReportTheme('light')")
    light_color = page.locator("#tier3-radar").evaluate("node=>Chart.getChart(node).options.scales.r.ticks.color")
    assert dark_color != light_color
    _subtab(page, "agents")
    _subtab(page, "overview")
    assert page.evaluate("chartConstructions.length") == 2
    _subtab(page, "trials")
    # The recorded token chart stays folded until its disclosure opens.
    page.locator("#tier3-page-trials canvas").evaluate_all("""nodes=>nodes.forEach(canvas=>{
        for(let node=canvas.parentElement;node;node=node.parentElement)
            if(node.tagName==='DETAILS')node.open=true;
    })""")
    page.wait_for_function("chartConstructions.length === 4")
    _subtab(page, "overview")
    _subtab(page, "trials")
    assert page.evaluate("chartConstructions.length") == 4


def test_folded_native_chart_is_lazy_and_print_draws_all_charts_synchronously(page, reports):
    directory, _, _ = reports
    _track_chart_constructions(page)
    _open(page, directory, "native-telemetry")
    assert page.evaluate("chartConstructions") == []
    _tier(page, "tier3")
    _subtab(page, "trials")
    assert page.evaluate("!Chart.getChart(document.getElementById('tier3-recorded-token-chart'))")
    page.locator("#tier3-token-usage > summary").click()
    page.wait_for_function("Chart.getChart(document.getElementById('tier3-recorded-token-chart'))")
    assert page.evaluate("chartConstructions.filter(row=>row.id==='tier3-recorded-token-chart').length") == 1
    colors = page.evaluate("""() => {
        const root=document.documentElement;
        const dark=getComputedStyle(root).getPropertyValue('--chart-label').trim();
        root.setAttribute('data-theme','light');
        const light=getComputedStyle(root).getPropertyValue('--chart-label').trim();
        root.setAttribute('data-theme','dark');
        return {dark,light};
    }""")
    assert colors["dark"] != colors["light"]
    page.evaluate("dispatchEvent(new Event('beforeprint'))")
    page.emulate_media(media="print")
    assert page.evaluate("""[...document.querySelectorAll('canvas[id^=tier3-]')].every(canvas=>{
        const chart=Chart.getChart(canvas);
        return chart && chart.width>0 && chart.height>0 && chart.options.animation===false;
    })""")
    assert (
        page.locator("canvas[id^=tier3-]").evaluate_all(
            "nodes=>nodes.map(canvas=>Chart.getChart(canvas).options.plugins.legend.labels.color)"
        )
        == [colors["light"]] * page.locator("canvas[id^=tier3-]").count()
    )
    assert page.evaluate("new Set(chartConstructions.map(row=>row.id)).size === chartConstructions.length")
    page.emulate_media(media="screen")
    page.evaluate("dispatchEvent(new Event('afterprint'))")
    assert page.locator("html").get_attribute("data-theme") == "dark"
    assert (
        page.locator("#tier3-recorded-token-chart").evaluate(
            "canvas=>Chart.getChart(canvas).options.plugins.legend.labels.color"
        )
        == colors["dark"]
    )
    assert page.evaluate("new Set(chartConstructions.map(row=>row.id)).size === chartConstructions.length")


def test_legacy_token_chart_keeps_totals_and_unknown_cached_measurements(page, tmp_path):
    result = report_browser._complete_result("synthetic", ("first", "second"), (0.94, 0.87))
    payload = result.metadata["agent_eval"]
    payload.pop("token_usage", None)
    payload["trials"][0]["tokens"]["cached"] = None
    (tmp_path / "legacy-tokens.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([result]), encoding="utf-8"
    )
    _open(page, tmp_path, "legacy-tokens")
    _subtab(page, "trials")
    chart = page.locator("#tier3-token-chart")
    assert chart.is_visible()
    assert page.locator("#tier3-recorded-token-chart").count() == 0
    data = chart.evaluate("node=>Chart.getChart(node).data")
    assert data["labels"] == ["first", "second"]
    assert {row["label"]: row["data"] for row in data["datasets"]} == {
        "Prompt (total)": [400, 400],
        "Completion": [40, 40],
        "Cached (included in prompt)": [None, 0],
    }


@pytest.mark.parametrize("fixture", ["full", "native-telemetry"])
def test_print_builds_charts_in_unvisited_tiers_synchronously(page, reports, fixture):
    directory, _, _ = reports
    _track_chart_constructions(page)
    _open(page, directory, fixture)
    assert page.evaluate("chartConstructions") == []
    assert page.evaluate("""() => {
        dispatchEvent(new Event('beforeprint'));
        return [...document.querySelectorAll('canvas[id^=tier3-]')].every(canvas=>{
            const chart=Chart.getChart(canvas);
            return chart && chart.ctx && chart.width>0 && chart.height>0 && chart.options.animation===false;
        });
    }""")
    page.emulate_media(media="print")
    assert page.evaluate("""[...document.querySelectorAll('canvas[id^=tier3-]')].every(canvas=>{
        const chart=Chart.getChart(canvas);
        return !canvas.hidden && chart.width>0 && chart.height>0;
    })""")
    assert page.evaluate("new Set(chartConstructions.map(row=>row.id)).size === chartConstructions.length")
    page.emulate_media(media="screen")
    page.evaluate("dispatchEvent(new Event('afterprint'))")
    assert page.locator("html").get_attribute("data-report-printing") is None
    assert page.locator("#tab-tier1").is_visible()
    assert not page.locator("#tab-tier3").is_visible()


@pytest.mark.parametrize("canvas_available", [True, False])
def test_dataset_search_prints_every_case_and_restores_filter_and_disclosures(page, reports, canvas_available):
    directory, _, _ = reports
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    if not canvas_available:
        page.add_init_script("HTMLCanvasElement.prototype.getContext=()=>null")
    _open(page, directory)
    _tier(page, "tier3")
    _subtab(page, "dataset")
    search = page.get_by_role("searchbox", name="Filter dataset", exact=True)
    search.fill("case-2")
    cards = page.locator("#t3-dataset-list > details")
    assert cards.count() == 4
    assert page.locator("#t3-dataset-list > details:visible").count() == 1
    state = cards.evaluate_all("nodes=>nodes.map(node=>({open:node.open,display:node.style.display}))")
    page.evaluate("dispatchEvent(new Event('beforeprint'))")
    page.emulate_media(media="print")
    assert page.locator("#t3-dataset-list > details:visible").count() == 4
    assert cards.evaluate_all(
        "nodes=>nodes.every(node=>node.open && node.querySelector('.t3-ds-body').getClientRects().length)"
    )
    page.emulate_media(media="screen")
    page.evaluate("dispatchEvent(new Event('afterprint'))")
    assert page.locator("html").get_attribute("data-report-printing") is None
    assert search.input_value() == "case-2"
    assert cards.evaluate_all("nodes=>nodes.map(node=>({open:node.open,display:node.style.display}))") == state
    assert page.locator("#t3-dataset-list > details:visible").count() == 1
    assert not errors
