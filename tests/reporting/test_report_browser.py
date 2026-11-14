# SPDX-License-Identifier: Apache-2.0
"""Exercise standalone report behavior in Chromium, without providers or network.

Opt in with SKILLEVALUATOR_BROWSER_TESTS=1 after installing the dev extra and
``python -m playwright install chromium``. Normal unit-test runs need no browser.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from browser_navigation_fixtures import _reward, _similarity_result, _tier1_results, render_fixtures

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

pytestmark = pytest.mark.skipif(
    os.environ.get("SKILLEVALUATOR_BROWSER_TESTS") != "1",
    reason="Enable the dedicated offline Chromium report checks with SKILLEVALUATOR_BROWSER_TESTS=1",
)


def _complete_result(skill: str, names: tuple[str, str], scores: tuple[float, float]) -> ValidationResult:
    cases = [
        {
            "id": f"case-{index}",
            "prompt": f"Use {skill} to summarize fictional input {index} without network access.",
            "expected_output": f"A summary of fictional input {index}.",
            "assertions": ["Reads the skill instructions.", "Writes a concise summary."],
            "expected_skill": skill,
        }
        for index in range(1, 5)
    ]
    agents = {
        name: {
            "execution_status": "succeeded",
            "expected_attempts": 8,
            "scored_attempts": 8,
            "with_skill": dict.fromkeys(DEFAULT_METRICS, score),
            "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.7),
            "rewards": [_reward(f"{case['id']}__attempt1", case["id"], score, 12) for case in cases],
            "rewards_baseline": [_reward(f"{case['id']}__baseline1", case["id"], 0.7, 20) for case in cases],
            "num_trials": 4,
            "num_trials_baseline": 4,
        }
        for name, score in zip(names, scores, strict=True)
    }
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = build_agent_eval_payload(skill, agents, dataset=cases, use_llm_judge=False)
    return result


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    if os.environ.get("SKILLEVALUATOR_REQUIRE_INSTALLED_REPORT") == "1":
        import skillevaluator

        assert Path(skillevaluator.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    directory = render_fixtures(tmp_path_factory.mktemp("browser-reports"))
    agent = _complete_result("demo", ('agent"[one]/雪', "agent.two"), (0.94, 0.87))
    (directory / "full.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([*_tier1_results(), _similarity_result(), agent]),
        encoding="utf-8",
    )
    if external := os.environ.get("SKILLEVALUATOR_BROWSER_REPORTS"):
        for name in ("full.html", "stress.html", "absent.html"):
            shutil.copyfile(Path(external) / name, directory / name)
    canonical = deepcopy(agent.metadata["agent_eval"])
    (directory / "canonical.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([agent]), encoding="utf-8"
    )
    failed = deepcopy(canonical)
    errors = [f"AgentTimeoutError: retained diagnostic {index}; original detail" for index in range(34)]
    blob = "; ".join(errors)
    failed.update(
        verdict="neutral",
        execution_status="failed",
        overall_score=None,
        overall_lift=None,
        trials=[],
        cases=[],
        dimensions=[],
        evaluator_cards=[],
        execution_errors=errors,
        conclusions=[
            {"title": "Evaluation incomplete", "message": blob, "severity": "fail", "source": "deterministic"}
        ],
        recommendations=[
            {"title": "Evaluation incomplete", "message": blob, "category": "action", "source": "deterministic"}
        ],
        suggestions=[blob],
        next_steps=[
            {
                "kind": "incomplete_run",
                "target": "evaluation",
                "problem": "8 planned attempts did not produce scores.",
                "action": "Rerun the incomplete evaluation before changing the skill or cases.",
                "check": "Every planned attempt has a score.",
                "strength": "supported",
                "evidence": [],
            }
        ],
    )
    # Recorded counters are independent of the cleared quality rows.
    failed["agents"] = {
        "agent": {
            "execution_status": "failed",
            "expected_attempts": 36,
            "scored_attempts": 28,
            "num_trials": 16,
            "num_trials_baseline": 12,
            "conditions": {
                "with_skill": {"execution_status": "failed", "expected_attempts": 18, "scored_attempts": 16},
                "without_skill": {"execution_status": "failed", "expected_attempts": 18, "scored_attempts": 12},
            },
        }
    }
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = failed
    result.add_error("Retained evaluation did not complete")
    (directory / "incomplete.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([result]), encoding="utf-8"
    )
    clean = ValidationResult(validator_name="Schema Check")
    clean.add_success("schema", "Valid schema")
    (directory / "tier1.html").write_text(HTMLReporter(include_timestamp=False).render_all([clean]), encoding="utf-8")
    # A second complete two-agent shape exercises a different recorded score.
    alternate = _complete_result("second-demo", ("agent-a", "agent-b"), (0.97, 0.79))
    (directory / "second.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([*_tier1_results(), _similarity_result(), alternate]),
        encoding="utf-8",
    )

    def section(key, title, value):
        return {
            "key": key,
            "title": title,
            "columns": ["Recorded observation"],
            "rows": [[value]],
            "total": 1,
            "omitted": 0,
            "notes": [],
        }

    static = {
        "schema_version": 1,
        "name": "synthetic-plugin",
        "status": "passed",
        "truncated": False,
        "sections": [
            section("components", "Component inventory", "sample-skill"),
            section("catalog_skills", "Catalog comparison", "No duplicate skills"),
        ],
    }
    runtime = {
        "schema_version": 1,
        "name": "synthetic-plugin",
        "status": "complete",
        "truncated": False,
        "run_id": "fixture-run",
        "execution_status": "succeeded",
        "evidence_complete": True,
        "source": {},
        "sections": [
            section("eligibility", "Measurement eligibility", "Complete recorded measurement"),
            section("loading", "Component loading", "Recorded native delivery"),
            section("comparisons", "Paired comparisons", "Four recorded pairs"),
            section("limitations", "Evidence limitations", "Synthetic offline fixture"),
        ],
    }
    (directory / "plugin.html").write_text(
        HTMLReporter(include_timestamp=False).render_all(
            [*_tier1_results(), _similarity_result(), agent], plugin_static=static, plugin_runtime=runtime
        ),
        encoding="utf-8",
    )
    (directory / "plugin-snapshots.json").write_text(json.dumps({"plugin_static": static, "plugin_runtime": runtime}))
    return directory, canonical, failed


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as runtime:
        instance = runtime.chromium.launch(
            executable_path=os.environ.get("SKILLEVALUATOR_TEST_BROWSER") or None, headless=True
        )
        yield instance
        instance.close()


@pytest.fixture
def page(browser, request):
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
    context.route("https://**/*", lambda route: route.abort())
    context.route("http://**/*", lambda route: route.abort())
    tab = context.new_page()
    yield tab
    if destination := os.environ.get("SKILLEVALUATOR_BROWSER_SCREENSHOTS"):
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^a-zA-Z0-9_.-]", "-", request.node.name)
        tab.screenshot(path=str(directory / f"{name}.png"))
    context.close()


def _open(page, directory: Path, name: str = "full"):
    page.goto((directory / f"{name}.html").as_uri())
    page.add_style_tag(
        content="*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}"
    )


def _tier(page, name: str):
    page.locator(f'.tier-card[data-target-tab="{name}"] .tier-card-label').click()
    assert page.locator(f"#tab-{name}").is_visible()


def _subtab(page, name: str):
    page.locator(f'[data-tier3-tab="{name}"]').click()
    assert page.locator(f"#tier3-page-{name}").is_visible()


@pytest.mark.parametrize("width", [1440, 760, 390])
def test_native_report_hierarchy_and_every_tier_fit_viewport(page, reports, width):
    directory, _, _ = reports
    page.set_viewport_size({"width": width, "height": 1000})
    for fixture in ("full", "second", "stress"):
        _open(page, directory, fixture)
        assert page.locator(".report-sidebar").count() == 0
        assert page.locator(".main-header").is_visible()
        styles = page.evaluate("""() => ({
            font:getComputedStyle(document.body).fontSize,
            line:getComputedStyle(document.body).lineHeight,
            header:getComputedStyle(document.querySelector('.main-header')).backgroundImage
        })""")
        assert styles["font"] == "16px"
        assert float(styles["line"].removesuffix("px")) == pytest.approx(25.6)
        assert "rgb(118, 185, 0)" in styles["header"]
        if width == 1440:
            title_box = page.locator(".agent-hero-title-block").bounding_box()
            score_box = page.locator(".hero-side").bounding_box()
            assert abs(title_box["y"] - score_box["y"]) <= 8
        for tier in ("tier1", "tier2", "tier3"):
            _tier(page, tier)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            if tier == "tier2":
                for scope in ("inter", "intra"):
                    page.locator("#dedup-tab-" + scope).click()
                    assert page.locator("#dedup-" + scope).is_visible()
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        for tab in ("overview", "suggestions", "agents", "trials", "dataset", "provenance"):
            _subtab(page, tab)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def test_theme_defaults_and_explicit_preference_survive_reload(page, reports):
    directory, _, _ = reports
    _open(page, directory, "tier1")
    assert page.locator("html").get_attribute("data-theme") == "light"
    _open(page, directory)
    assert page.locator("html").get_attribute("data-theme") == "dark"
    page.locator(".theme-toggle").click()
    assert page.locator("html").get_attribute("data-theme") == "light"
    page.reload()
    assert page.locator("html").get_attribute("data-theme") == "light"


def test_keyboard_navigation_reveals_nested_agent_and_finding_evidence(page, reports):
    directory, _, _ = reports
    _open(page, directory)
    _tier(page, "tier2")
    page.locator("#dedup-tab-inter").click()
    assert page.locator("#dedup-inter").is_visible()
    page.keyboard.press("Home")
    page.keyboard.press("Enter")
    assert page.locator("#dedup-intra").is_visible()
    _tier(page, "tier3")
    tabs = page.locator('[role="tab"][data-tier3-tab]')
    assert tabs.count() == 6
    tabs.nth(0).focus()
    page.keyboard.press("ArrowRight")
    page.keyboard.press("Enter")
    assert page.locator("#tier3-page-suggestions").is_visible()
    _subtab(page, "agents")
    agents = page.locator(".t3-agent-tab-btn")
    assert agents.count() == 2
    agents.nth(0).focus()
    page.keyboard.press("End")
    page.keyboard.press("Enter")
    assert agents.nth(1).get_attribute("aria-selected") == "true"
    anchor_id = page.locator(".t3-agent-anchor").nth(0).get_attribute("id")
    page.set_viewport_size({"width": 390, "height": 1000})
    page.goto((directory / "full.html").as_uri() + "#" + anchor_id)
    page.reload()
    page.wait_for_function("""() => {
        const header=document.querySelector('.main-header').getBoundingClientRect();
        const tabs=document.querySelector('.t3-subtabs').getBoundingClientRect();
        const anchor=document.querySelector('.t3-agent-panel.active .t3-agent-anchor').getBoundingClientRect();
        return tabs.top >= header.bottom - 2 && anchor.top >= header.bottom + tabs.height - 2;
    }""")
    page.set_viewport_size({"width": 1440, "height": 1000})
    _tier(page, "tier1")
    search = page.locator("#report-search")
    search.fill("a finding that does not exist")
    assert page.locator(".check-finding:visible").count() == 0
    page.get_by_role("button", name="Reset filters", exact=True).click()
    # A stored finding fragment must reveal its owning tier/card after reload.
    finding_id = page.locator(".check-finding[id]").nth(0).get_attribute("id")
    page.goto((directory / "full.html").as_uri() + "#" + finding_id)
    page.reload()
    assert page.locator("#" + finding_id).is_visible()
    assert page.evaluate("document.activeElement.id") == finding_id


@pytest.mark.parametrize("fixture", ["canonical", "incomplete"])
def test_full_payload_download_preserves_retained_canonical_object(page, reports, fixture, tmp_path):
    directory, canonical, failed = reports
    _open(page, directory, fixture)
    _tier(page, "tier3")
    _subtab(page, "provenance")
    with page.expect_download() as event:
        page.get_by_role("button", name="Download Full Tier 3 Payload", exact=True).click()
    destination = tmp_path / "tier3.json"
    event.value.save_as(destination)
    assert json.loads(destination.read_text()) == (canonical if fixture == "canonical" else failed)


def test_incomplete_insights_are_concise_and_full_errors_stay_in_diagnostics(page, reports):
    directory, _, failed = reports
    _open(page, directory, "incomplete")
    _tier(page, "tier3")
    _subtab(page, "suggestions")
    insights = page.locator("#tier3-page-suggestions").inner_text()
    assert "8" in insights and "36" in insights
    assert "Rerun the incomplete evaluation" in insights
    assert "retained diagnostic 0" not in insights
    assert len(insights) < 2500
    _subtab(page, "provenance")
    page.locator("#tier3-page-provenance details").evaluate_all("nodes=>nodes.forEach(node=>node.open=true)")
    diagnostics = page.locator("#tier3-page-provenance").inner_text()
    assert all(error in diagnostics for error in failed["execution_errors"])


def test_offline_charts_copy_and_json_export_work_without_browser_errors(page, reports):
    directory, _, _ = reports
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    _open(page, directory)
    _tier(page, "tier3")
    _subtab(page, "overview")
    assert page.locator("#tier3-radar").is_visible()
    assert page.evaluate("typeof Chart") == "function"
    _subtab(page, "suggestions")
    copy = page.locator("#tier3-page-suggestions .t3-copy-btn").nth(0)
    copy.click()
    page.wait_for_function(
        """() => [...document.querySelectorAll('#tier3-page-suggestions .t3-copy-btn')].some(node => /copied|copy failed/i.test(node.innerText))"""
    )
    with page.expect_download() as event:
        page.locator(".export-btn").click()
    assert event.value.suggested_filename.endswith(".json")
    assert not errors


def test_plugin_evidence_is_reachable_in_its_tiers_and_downloaded_once(page, reports, tmp_path):
    directory, _, _ = reports
    _open(page, directory, "plugin")
    page.set_viewport_size({"width": 390, "height": 1000})
    for tier, evidence in (("tier1", "plugin-static-components"), ("tier2", "plugin-static-catalog_skills")):
        _tier(page, tier)
        assert page.locator("#" + evidence).is_visible()
        assert page.locator("#" + evidence).count() == 1
    _tier(page, "tier3")
    for tab, key in (
        ("overview", "eligibility"),
        ("agents", "loading"),
        ("trials", "comparisons"),
        ("provenance", "limitations"),
    ):
        _subtab(page, tab)
        assert page.locator("#plugin-runtime-" + key).is_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    with page.expect_download() as event:
        page.locator(".export-btn").click()
    target = tmp_path / "report.json"
    event.value.save_as(target)
    exported = json.loads(target.read_text())
    expected = json.loads((directory / "plugin-snapshots.json").read_text())
    assert {key: exported[key] for key in expected} == expected


def test_print_exposes_evidence_and_restores_disclosures(page, reports):
    directory, _, _ = reports
    _open(page, directory)
    page.locator("#report-search").fill("no matching finding")
    assert page.locator(".check-finding:visible").count() == 0
    _tier(page, "tier3")
    _subtab(page, "dataset")
    page.evaluate("window.closedDetails=[...document.querySelectorAll('details:not([open]):not(.raw-output)')]")
    page.evaluate("dispatchEvent(new Event('beforeprint'))")
    page.emulate_media(media="print")
    assert page.evaluate(
        "[...document.querySelectorAll('.t3-page,.t3-agent-panel,.check-finding')].every(node=>node.getClientRects().length>0)"
    )
    assert page.locator("canvas:visible").count() == 4
    page.emulate_media(media="screen")
    page.evaluate("dispatchEvent(new Event('afterprint'))")
    assert page.evaluate("window.closedDetails.every(node=>!node.open)")
    _tier(page, "tier1")
    assert page.locator("#report-search").input_value() == "no matching finding"
    assert page.locator(".check-finding:visible").count() == 0


def test_no_javascript_keeps_all_recorded_evidence_readable(browser, reports):
    directory, _, _ = reports
    context = browser.new_context(java_script_enabled=False, viewport={"width": 390, "height": 1000})
    try:
        tab = context.new_page()
        tab.goto((directory / "full.html").as_uri())
        assert tab.evaluate(
            "[...document.querySelectorAll('.t3-page,.t3-agent-panel,.check-finding')].every(node=>node.getClientRects().length>0)"
        )
        assert tab.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    finally:
        context.close()
