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
from test_telemetry_view import _payload as _native_telemetry_payload

from skillevaluator.evaluation.tier3_expectations import build_case_diagnostics
from skillevaluator.evaluation.tier3_report import advisory_skip_result, build_agent_eval_payload
from skillevaluator.models import Finding, ValidationResult
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


def _edge_result(state):
    """Synthetic retained shapes shared with missing-arm/telemetry unit coverage."""
    with_rewards = [_reward("case-1__attempt1", "case-1", 0.9, 12)]
    baseline_rewards = [_reward("case-1__baseline1", "case-1", 0.7, 20)]
    info = {
        "execution_status": "succeeded",
        "expected_attempts": 2,
        "scored_attempts": 2,
        "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.9),
        "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.7),
        "rewards": with_rewards,
        "rewards_baseline": baseline_rewards,
        "num_trials": 1,
        "num_trials_baseline": 1,
        "rewards_complete": True,
        "rewards_baseline_complete": True,
        "conditions": {
            arm: {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1}
            for arm in ("with_skill", "without_skill")
        },
    }
    agents = {"complete-peer": info}
    if state == "mixed":
        agents["failed-peer"] = {
            "execution_status": "failed",
            "expected_attempts": 2,
            "scored_attempts": 0,
            "execution_errors": ["AgentTimeoutError: failed-peer case-1 synthetic retained diagnostic"],
            "with_skill": {},
            "without_skill": {},
            "rewards": [],
            "rewards_baseline": [],
            "num_trials": 0,
            "num_trials_baseline": 0,
        }
    elif state == "custom-only":
        info.update(
            with_skill={},
            without_skill={},
            metrics_with_skill=[],
            metrics_without_skill=[],
            overall_with_skill=0.45,
            overall_without_skill=0.45,
        )
        for rewards in (with_rewards, baseline_rewards):
            rewards[0] = {
                "entry_id": "case-1",
                "metric_set": "custom-only",
                "overall": 0.45,
                "custom_metrics": {"task_quality": 0.45},
            }
    elif state == "missing-baseline":
        info.update(
            without_skill={},
            rewards_baseline=[],
            num_trials_baseline=0,
            expected_attempts=1,
            scored_attempts=1,
            conditions={
                "with_skill": {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1},
                "without_skill": {"execution_status": "skipped"},
            },
        )
    elif state == "missing-telemetry":
        for rewards in (with_rewards, baseline_rewards):
            rewards[0].pop("_traj")
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = build_agent_eval_payload(
        "synthetic-edge",
        agents,
        dataset=[{"id": "case-1", "prompt": "Summarize synthetic input", "expected_output": "Synthetic summary"}],
        use_llm_judge=False,
    )
    assert result.metadata["agent_eval"] is not None
    if state == "mixed":
        result.add_error("failed-peer did not complete")
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
    failed["case_diagnostics"], _ = build_case_diagnostics(failed["dataset"], failed["agents"])
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = failed
    result.add_error("Retained evaluation did not complete")
    (directory / "incomplete.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([result]), encoding="utf-8"
    )
    (directory / "incomplete-full.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([*_tier1_results(), result]), encoding="utf-8"
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
    (directory / "static-plugin.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([clean], plugin_static=static), encoding="utf-8"
    )
    (directory / "plugin-snapshots.json").write_text(json.dumps({"plugin_static": static, "plugin_runtime": runtime}))
    for state in ("mixed", "custom-only", "missing-baseline", "missing-telemetry"):
        (directory / f"{state}.html").write_text(
            HTMLReporter(include_timestamp=False, report_scope="tier3").render_all([_edge_result(state)]),
            encoding="utf-8",
        )
    skipped = advisory_skip_result("Synthetic retained Tier 3 explicitly skipped", skill_name="synthetic-edge")
    skipped.metadata["optional_skip"] = True
    skipped.metadata["skip_reason"] = "Synthetic retained Tier 3 explicitly skipped"
    (directory / "skipped-tier3.html").write_text(
        HTMLReporter(include_timestamp=False, requested_tiers=(1, 2, 3), displayed_tiers=(1, 2, 3)).render_all(
            [clean, skipped]
        ),
        encoding="utf-8",
    )
    incomplete_static = deepcopy(static)
    incomplete_static.update(status="incomplete", truncated=True)
    incomplete_static["sections"][0].update(
        total=3, omitted=2, notes=["Inventory incomplete; retained observation only"]
    )
    incomplete_runtime = deepcopy(runtime)
    incomplete_runtime.update(status="incomplete", execution_status="failed", evidence_complete=False)
    incomplete_runtime["sections"][0]["rows"] = [["Not eligible: incomplete recorded arms"]]
    incomplete_runtime["sections"][1]["rows"] = [[None]]
    incomplete_runtime["sections"][2]["rows"] = [["Comparison unavailable; incomplete pair"]]
    incomplete_runtime["sections"][3]["rows"] = [["Synthetic retained runtime timeout; no behavior conclusion"]]
    (directory / "incomplete-plugin.html").write_text(
        HTMLReporter(include_timestamp=False).render_all(
            [_edge_result("mixed")], plugin_static=incomplete_static, plugin_runtime=incomplete_runtime
        ),
        encoding="utf-8",
    )
    similarity = ValidationResult(validator_name="Similarity Check")
    similarity.add_success("index_built", "Indexed synthetic collection", entry_count=3, content_type="skill")
    similarity.add_finding(
        Finding(
            category="SIMILARITY",
            check_name="EXACT_DUPLICATE",
            severity="high",
            message="Synthetic collection duplicate",
            file_path="skills/a/SKILL.md",
            metadata={"path_a": "skills/a", "path_b": "skills/b"},
        )
    )
    (directory / "nonzero-similarity.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([similarity]), encoding="utf-8"
    )
    native = ValidationResult(validator_name="AGENT_EVAL")
    native.metadata["agent_eval"] = build_agent_eval_payload(
        "native-telemetry",
        {
            "agent-a": {
                "execution_status": "succeeded",
                "expected_attempts": 4,
                "scored_attempts": 4,
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.9),
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.7),
                "rewards": [_reward(f"case-one__attempt{i}", "case-one", 0.9, 12) for i in (1, 2)],
                "rewards_baseline": [_reward(f"case-one__baseline{i}", "case-one", 0.7, 20) for i in (1, 2)],
                "num_trials": 2,
                "num_trials_baseline": 2,
                "conditions": {
                    arm: {"execution_status": "succeeded", "expected_attempts": 2, "scored_attempts": 2}
                    for arm in ("with_skill", "without_skill")
                },
            }
        },
        dataset=[{"id": "case-one", "prompt": "Summarize synthetic input", "expected_output": "Summary"}],
        use_llm_judge=False,
    )
    native.metadata["agent_eval"].update(_native_telemetry_payload())
    (directory / "native-telemetry.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([*_tier1_results(), _similarity_result(), native]),
        encoding="utf-8",
    )
    qualityless = deepcopy(native)
    qualityless.add_error("Synthetic retained quality-agent mapping unavailable")
    qualityless.metadata["agent_eval"].update(
        agents={},
        trials=[],
        cases=[],
        dimensions=[],
        best_agent=None,
        overall_score=None,
        overall_lift=None,
        summary={},
        execution_status="failed",
        verdict="neutral",
    )
    (directory / "native-no-quality-agents.html").write_text(
        HTMLReporter(include_timestamp=False, report_scope="tier3").render_all([qualityless]), encoding="utf-8"
    )
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
    page.locator("#report-theme").select_option("light")
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


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_active_native_controls_have_readable_computed_contrast(page, reports, theme, width):
    directory, _, _ = reports
    page.set_viewport_size({"width": width, "height": 1000})
    _open(page, directory)
    page.locator("html").evaluate("(node, theme) => node.dataset.theme=theme", theme)
    checked = []
    for tier in ("tier1", "tier2"):
        _tier(page, tier)
        controls = page.locator(
            ".view-toggle-btn:visible,.filter-btn:visible,.severity-filter-btn:visible,.nav-tab:visible"
        )
        assert controls.count() > 0
        for control in controls.all():
            # Inspect every actual control's active variant, including severity colors.
            colors = control.evaluate("""node => {
                node.classList.add('active');
                const style=getComputedStyle(node);
                return {foreground:style.color, background:style.backgroundColor, label:node.textContent.trim()};
            }""")

            def luminance(color):
                channels = [int(value) / 255 for value in re.findall(r"[0-9]+", color)[:3]]
                linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
                return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

            values = sorted((luminance(colors["foreground"]), luminance(colors["background"])))
            ratio = (values[1] + 0.05) / (values[0] + 0.05)
            assert ratio >= 4.5, (theme, width, colors, ratio)
            checked.append(colors["label"])
    assert len(checked) >= 8
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_static_only_native_disclosures_have_shared_styles_without_chart_runtime(page, reports, theme):
    directory, _, _ = reports
    _open(page, directory, "static-plugin")
    page.locator("html").evaluate("(node, theme) => node.dataset.theme=theme", theme)
    assert page.evaluate("typeof Chart") == "undefined"
    assert page.locator("#plugin-static-components").is_visible()
    details = page.locator("details.t3-details").first
    assert details.is_visible()
    details.evaluate("node=>node.open=false")
    styles = details.evaluate("""node => {
        const detail=getComputedStyle(node), summary=getComputedStyle(node.querySelector('summary'));
        return {radius:detail.borderRadius,border:detail.borderTopWidth,padding:summary.padding,
            display:summary.display,arrow:getComputedStyle(node.querySelector('summary'),'::after').content};
    }""")
    assert styles == {"radius": "12px", "border": "1px", "padding": "14.4px 16px", "display": "flex", "arrow": '"▼"'}
    details.locator("summary").click()
    assert details.get_attribute("open") is not None
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize(
    "fixture",
    [
        "mixed",
        "custom-only",
        "missing-baseline",
        "missing-telemetry",
        "skipped-tier3",
        "incomplete-plugin",
        "nonzero-similarity",
    ],
)
def test_retained_edge_state_matrix_is_truthful_accessible_and_has_no_browser_errors(page, reports, fixture, width):
    directory, _, _ = reports
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.set_viewport_size({"width": width, "height": 1000})
    _open(page, directory, fixture)
    if fixture == "nonzero-similarity":
        _tier(page, "tier2")
        page.locator("#dedup-tab-inter").click()
        scope = page.locator("#dedup-inter")
        assert "3 collection entries" in scope.inner_text()
        assert "Synthetic collection duplicate" in scope.inner_text()
        assert scope.locator("details").count() > 0
        scope.locator("details").evaluate_all("nodes=>nodes.forEach(node=>node.open=true)")
        assert scope.get_by_text("skills/a", exact=False).count() > 0
        assert "Changes required" in page.locator('.tier-card[data-target-tab="tier2"]').inner_text()
    elif fixture == "skipped-tier3":
        tile = page.locator('.tier-card[data-target-tab="tier3"]')
        assert "Skipped" in tile.inner_text()
        _tier(page, "tier3")
        assert (
            "Live evaluation skipped: Synthetic retained Tier 3 explicitly skipped"
            in page.locator("#tab-tier3").inner_text()
        )
        assert page.locator("#tier3-radar").count() == 0
    else:
        _tier(page, "tier3")
        overview = page.locator("#tier3-page-overview")
        if fixture in ("mixed", "incomplete-plugin"):
            assert "Other execution failures need attention" in overview.inner_text()
            assert "complete-peer" in overview.inner_text()
            _subtab(page, "agents")
            buttons = page.locator(".t3-agent-tab-btn")
            assert buttons.count() == 2
            buttons.filter(has_text="failed-peer").click()
            failed_panel = page.locator(".t3-agent-panel.active")
            assert failed_panel.get_attribute("data-tier3-agent") == "failed-peer"
            assert failed_panel.locator(".t3-kpi-value").all_text_contents() == ["N/A", "N/A", "0"]
            _subtab(page, "provenance")
            page.locator("#tier3-page-provenance details").evaluate_all("nodes=>nodes.forEach(node=>node.open=true)")
            assert (
                "AgentTimeoutError: failed-peer case-1 synthetic retained diagnostic"
                in page.locator("#tier3-page-provenance").inner_text()
            )
        elif fixture == "custom-only":
            assert "Neutral" in overview.inner_text()
            assert page.locator("#tier3-dimension-table").count() == 0
            _subtab(page, "agents")
            assert "Custom: task_quality" in page.locator("#tier3-page-agents").inner_text()
        elif fixture == "missing-baseline":
            assert "no baseline" in overview.inner_text()
            _subtab(page, "trials")
            page.locator("#tier3-page-trials details").evaluate_all("nodes=>nodes.forEach(node=>node.open=true)")
            table = page.locator("#t3-trials-table")
            assert table.is_visible()
            assert table.locator("tbody tr").count() == 1
            assert "With skill" in table.inner_text()
            assert "Without skill" not in table.inner_text()
        elif fixture == "missing-telemetry":
            _subtab(page, "trials")
            page.locator("#tier3-page-trials details").evaluate_all("nodes=>nodes.forEach(node=>node.open=true)")
            table = page.locator("#t3-trials-table")
            assert table.is_visible()
            headers = table.locator("thead th").all_text_contents()
            for label in ("Steps", "Tokens"):
                column = headers.index(label)
                assert [row.locator("td").nth(column).inner_text() for row in table.locator("tbody tr").all()] == [
                    "—",
                    "—",
                ]
            assert page.locator("#tier3-token-chart").count() == 0
        if fixture == "incomplete-plugin":
            _tier(page, "tier1")
            assert "Static snapshot · incomplete" in page.locator("#plugin-static-evidence").inner_text()
            inventory = page.locator("#plugin-static-components")
            assert inventory.is_visible()
            inventory.locator("summary").click()
            assert "2 more rows omitted (3 total)" in inventory.inner_text()
            _tier(page, "tier3")
            _subtab(page, "overview")
            assert (
                "Evidence incomplete · execution failed · advisory snapshot"
                in page.locator("#plugin-runtime-evidence").inner_text()
            )
            for tab, key, evidence in (
                ("overview", "eligibility", "Not eligible"),
                ("agents", "loading", "unknown"),
                ("trials", "comparisons", "Comparison unavailable"),
                ("provenance", "limitations", "no behavior conclusion"),
            ):
                _subtab(page, tab)
                section = page.locator("#plugin-runtime-" + key)
                assert section.is_visible()
                if section.get_attribute("open") is None:
                    section.locator("summary").click()
                assert evidence in section.inner_text()
        for tab in ("overview", "suggestions", "agents", "trials", "dataset", "provenance"):
            _subtab(page, tab)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert not errors


def test_chart_runtime_unavailable_exposes_fallbacks_and_linked_tables_without_errors(page, reports):
    directory, _, _ = reports
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    # Prevent the bundled inline runtime from creating Chart while all report JS executes normally.
    page.add_init_script("Object.defineProperty(window,'Chart',{get:()=>undefined,set:()=>{},configurable:false})")
    _open(page, directory)
    assert page.evaluate("typeof Chart") == "undefined"
    _tier(page, "tier3")
    token_chart = (
        "tier3-recorded-token-chart" if page.locator("#tier3-recorded-token-chart").count() else "tier3-token-chart"
    )
    for tab, chart_ids in (
        ("overview", ("tier3-radar", "tier3-metric-bar")),
        ("trials", (token_chart, "tier3-steps-chart")),
    ):
        _subtab(page, tab)
        for chart_id in chart_ids:
            canvas = page.locator("#" + chart_id)
            assert canvas.count() == 1
            canvas.evaluate("""node=>{
                for(let parent=node.parentElement;parent;parent=parent.parentElement)
                    if(parent.tagName==='DETAILS')parent.open=true;
            }""")
            assert not canvas.is_visible()
            assert page.locator("#" + chart_id + "-fallback").is_visible()
            link = page.locator("#" + chart_id + "-description a")
            target = link.get_attribute("href")
            link.click()
            assert page.locator(target).is_visible()
            assert page.locator(target).locator("table").count() or page.locator(target).evaluate(
                "node=>node.tagName==='TABLE'"
            )
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            _subtab(page, tab)
    assert not errors


@pytest.mark.parametrize("fixture", ["full", "incomplete", "mixed", "custom-only"])
def test_overview_is_a_concise_projection_and_full_insights_have_one_owner(page, reports, fixture):
    directory, _, _ = reports
    _open(page, directory, fixture)
    _tier(page, "tier3")
    overview = page.locator("#tier3-page-overview")
    assert overview.locator(".t3-sug-card").count() == 0
    assert overview.locator(".dashboard-card").count() <= 3
    assert page.locator("#run-metadata").get_attribute("open") is None
    assert page.locator(".nav-tab:visible").count() == 0
    assert page.locator(".header-status-badge").count() == 0
    if fixture == "incomplete":
        assert "28" in overview.inner_text() and "36" in overview.inner_text()
        assert "unavailable" in overview.inner_text().lower()
        assert overview.locator(".dashboard-card").filter(has_text="Overall Score").count() == 0
    _subtab(page, "suggestions")
    assert page.locator("#tier3-page-suggestions .t3-sug-card").count() > 0


def test_folded_plugin_evidence_is_closed_then_revealed_by_its_existing_link_and_search(page, reports):
    directory, _, _ = reports
    _open(page, directory, "plugin")
    _tier(page, "tier1")
    component = page.locator("#plugin-static-components")
    assert component.get_attribute("open") is None
    assert not component.locator("table").is_visible()
    page.locator("#report-search").fill("sample-skill")
    assert component.locator("table").is_visible()
    assert "sample-skill" in component.inner_text()
    page.goto((directory / "plugin.html").as_uri() + "#plugin-runtime-loading")
    loading = page.locator("#plugin-runtime-loading")
    assert loading.is_visible()
    assert loading.locator("table").is_visible()
    assert "Recorded native delivery" in loading.inner_text()


def test_mobile_overview_places_incomplete_result_and_recorded_action_before_metadata(page, reports):
    directory, _, _ = reports
    page.set_viewport_size({"width": 390, "height": 1000})
    _open(page, directory, "incomplete")
    _tier(page, "tier3")
    _subtab(page, "overview")
    overview = page.locator("#tier3-page-overview")
    overview.scroll_into_view_if_needed()
    action = overview.get_by_text("Rerun the incomplete evaluation", exact=False)
    assert action.count() == 1
    action_box = action.bounding_box()
    assert action_box is not None and action_box["y"] + action_box["height"] <= 1000
    assert page.locator(".main-header").bounding_box()["height"] <= 96
    assert "8" in overview.inner_text() and "36" in overview.inner_text()
    assert page.locator("#run-metadata").get_attribute("open") is None
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


@pytest.mark.parametrize("width", [1440, 390])
def test_native_recorded_telemetry_is_folded_complete_and_draws_one_condition_aware_chart(page, reports, width):
    directory, _, _ = reports
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.set_viewport_size({"width": width, "height": 1000})
    _open(page, directory, "native-telemetry")
    page.evaluate("filterValidators('fail')")
    _tier(page, "tier3")
    _subtab(page, "trials")
    tokens = page.locator("#tier3-token-usage")
    latency = page.locator("#tier3-latency-metrics")
    assert tokens.get_attribute("open") is None and latency.get_attribute("open") is None
    tokens.locator("summary").first.click()
    text = tokens.inner_text()
    for marker in (
        "Prompt",
        "Cached",
        "Uncached prompt",
        "Completion",
        "Total",
        "Efficiency Tokens",
        "Without skill",
        "case-one",
    ):
        assert marker.lower() in text.lower()
    assert tokens.locator("table").first.is_visible()
    assert tokens.locator("tbody tr:visible").count() == 9
    assert page.locator("#tier3-token-chart").count() == 0
    chart = page.locator("#tier3-recorded-token-chart")
    page.wait_for_function("""() => {
        const canvas=document.getElementById('tier3-recorded-token-chart');
        return canvas&&!canvas.hidden&&window.Chart?.getChart(canvas);
    }""")
    data = chart.evaluate("node=>Chart.getChart(node).data")
    assert any("With skill" in label for label in data["labels"])
    assert any("Without skill" in label for label in data["labels"])
    assert [series["label"] for series in data["datasets"]] == ["Uncached prompt", "Cached prompt", "Completion"]
    assert sum(series["data"][0] for series in data["datasets"]) == 120
    latency.locator("summary").first.click()
    for marker in (
        "LLM (s)",
        "Tool (s)",
        "LLM ms/token",
        "Arm runtime (ms)",
        "partial",
        "Baseline timing is incomplete.",
    ):
        assert marker.lower() in latency.inner_text().lower()
    page.locator("#t3-trials-table").evaluate("""node=>{
        for(let parent=node.parentElement;parent;parent=parent.parentElement)
            if(parent.tagName==='DETAILS')parent.open=true;
    }""")
    assert page.locator("#t3-trials-table tbody tr:visible").count() == 4
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert not errors


def test_native_recorded_chart_fallback_links_to_retained_table(page, reports):
    directory, _, _ = reports
    page.add_init_script("Object.defineProperty(window,'Chart',{get:()=>undefined,set:()=>{},configurable:false})")
    _open(page, directory, "native-telemetry")
    _tier(page, "tier3")
    _subtab(page, "trials")
    page.locator("#tier3-token-usage > summary").click()
    assert not page.locator("#tier3-recorded-token-chart").is_visible()
    fallback = page.locator("#tier3-recorded-token-chart-fallback")
    assert fallback.is_visible()
    fallback.locator("a").click()
    assert page.locator("#t3-recorded-token-usage-table").is_visible()


def test_recorded_chart_loads_even_when_quality_agent_mapping_is_unavailable(page, reports):
    directory, _, _ = reports
    _open(page, directory, "native-no-quality-agents")
    _tier(page, "tier3")
    _subtab(page, "trials")
    page.locator("#tier3-token-usage > summary").click()
    assert page.evaluate("typeof Chart") == "function"
    page.wait_for_function("""() => {
        const canvas=document.getElementById('tier3-recorded-token-chart');
        return canvas&&!canvas.hidden&&window.Chart.getChart(canvas);
    }""")
    assert page.locator("#t3-recorded-token-usage-table").is_visible()
    assert page.locator("#tier3-radar").count() == 0


def test_search_reveals_hidden_runtime_and_nested_dataset_evidence_then_restores_state(page, reports):
    directory, _, _ = reports
    _open(page, directory, "plugin")
    _tier(page, "tier1")
    page.locator("#report-search").fill("Recorded native delivery")
    loading = page.locator("#plugin-runtime-loading")
    assert page.locator("#tier3-page-agents").is_visible()
    assert loading.locator("table").is_visible()
    _tier(page, "tier1")
    page.locator("#report-search").fill("")
    assert loading.get_attribute("open") is None

    _open(page, directory, "incomplete")
    _tier(page, "tier3")
    _subtab(page, "dataset")
    cases = page.locator("#t3-dataset-list > details")
    assert cases.count() == 4
    later_case = cases.nth(1)
    assert later_case.get_attribute("open") is None
    search = page.get_by_role("searchbox", name="Filter dataset", exact=True)
    search.fill("No matched trial")
    assert later_case.get_attribute("open") is not None
    assert later_case.locator("details[open]").count() > 0
    assert "No matched trial" in later_case.inner_text()
    search.fill("")
    assert later_case.get_attribute("open") is None


def test_global_search_reveals_dataset_attempt_body_matches_and_preserves_user_disclosures(page, reports):
    directory, _, _ = reports
    _open(page, directory, "incomplete-full")
    _tier(page, "tier3")
    _subtab(page, "dataset")
    cases = page.locator("#t3-dataset-list > details")
    attempts = cases.locator("details.t3-details")
    assert attempts.count() == 8
    # Keep a user-opened case and attempt, while later cases remain folded.
    assert cases.nth(0).get_attribute("open") is not None
    attempts.nth(0).locator("summary").click()
    initial_state = page.locator("#t3-dataset-list details").evaluate_all("nodes=>nodes.map(node=>node.open)")
    query = "No retained trial result could be matched to this case"
    assert all(query not in summary for summary in attempts.locator("summary").all_text_contents())
    assert not attempts.nth(1).locator(".t3-details-body").is_visible()

    _tier(page, "tier1")
    search = page.locator("#report-search")
    search.fill(query)
    assert page.locator("#tier3-page-dataset").is_visible()
    assert "evidence sections matched" in page.locator("#report-search-count").inner_text()
    assert not page.locator("#report-no-matches").is_visible()
    for index in range(attempts.count()):
        body = attempts.nth(index).locator(".t3-details-body")
        assert body.is_visible()
        assert query in body.inner_text()

    # A changed query also restores only what the previous search opened.
    _tier(page, "tier1")
    search.fill("a query with no matching evidence")
    assert page.locator("#t3-dataset-list details").evaluate_all("nodes=>nodes.map(node=>node.open)") == initial_state
    search.fill(query)
    assert attempts.nth(1).get_attribute("open") is not None
    _tier(page, "tier1")
    search.fill("")
    assert page.locator("#t3-dataset-list details").evaluate_all("nodes=>nodes.map(node=>node.open)") == initial_state


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_overview_text_links_and_insights_action_have_readable_contrast(page, reports, theme, width):
    directory, _, _ = reports
    page.set_viewport_size({"width": width, "height": 1000})
    _open(page, directory, "incomplete")
    page.locator("html").evaluate("(node, theme)=>node.dataset.theme=theme", theme)
    _tier(page, "tier3")
    overview = page.locator("#tier3-page-overview")
    links = overview.locator("a:visible")
    assert links.count() >= 3
    for link in links.all():
        colors = link.evaluate("""node=>{
            const foreground=getComputedStyle(node).color;
            let parent=node,background;
            while(parent){
                background=getComputedStyle(parent).backgroundColor;
                if(background!=='rgba(0, 0, 0, 0)'&&background!=='transparent')break;
                parent=parent.parentElement;
            }
            return {foreground,background,label:node.textContent.trim()};
        }""")

        def luminance(color):
            channels = [int(value) / 255 for value in re.findall(r"[0-9]+", color)[:3]]
            linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
            return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

        values = sorted((luminance(colors["foreground"]), luminance(colors["background"])))
        ratio = (values[1] + 0.05) / (values[0] + 0.05)
        assert ratio >= 4.5, (theme, width, colors, ratio)


def test_sole_mobile_coverage_metric_uses_the_available_grid_width(page, reports):
    directory, _, _ = reports
    page.set_viewport_size({"width": 390, "height": 1000})
    _open(page, directory, "incomplete")
    _tier(page, "tier3")
    grid = page.locator(".t3-primary-metrics")
    cards = grid.locator(".dashboard-card")
    assert cards.count() == 1
    assert cards.first.bounding_box()["width"] >= grid.bounding_box()["width"] - 2
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def test_print_wraps_unbroken_retained_table_cells_within_the_page(page, tmp_path):
    result = ValidationResult(validator_name="Schema Check")
    result.add_success("schema_valid", "Synthetic static check")
    snapshot = {
        "schema_version": 1,
        "name": "synthetic-print",
        "status": "passed",
        "truncated": False,
        "sections": [
            {
                "key": "components",
                "title": "Component inventory",
                "columns": ["Recorded observation"],
                "rows": [["x" * 280]],
                "total": 1,
                "omitted": 0,
                "notes": [],
            }
        ],
    }
    (tmp_path / "print-table.html").write_text(
        HTMLReporter(include_timestamp=False).render_all([result], plugin_static=snapshot), encoding="utf-8"
    )
    page.set_viewport_size({"width": 794, "height": 1123})
    _open(page, tmp_path, "print-table")
    page.evaluate("dispatchEvent(new Event('beforeprint'))")
    page.emulate_media(media="print")
    bounds = page.locator("#plugin-static-components table").evaluate("""node=>({
        table:node.getBoundingClientRect().width,
        owner:node.closest('.table-scroll').getBoundingClientRect().width
    })""")
    assert bounds["table"] <= bounds["owner"] + 2, bounds
    page.emulate_media(media="screen")
    page.evaluate("dispatchEvent(new Event('afterprint'))")


@pytest.mark.parametrize("width", [1440, 390])
def test_tier1_top_issue_labels_stay_intact_while_long_evidence_wraps(page, tmp_path, width):
    result = ValidationResult(validator_name="Synthetic Plugin Schema")
    token = "recorded-command-" + "x" * 280
    message = (
        "A recorded hook command refers outside the synthetic plugin root; "
        "review its retained target before accepting this component: " + token
    )
    result.add_finding(
        Finding(
            category="PLUGIN_SCHEMA",
            severity="medium",
            check_name="hook_target",
            message=message,
            file_path="hooks/synthetic.json",
        )
    )
    (tmp_path / "tier1-long-evidence.html").write_text(
        HTMLReporter(
            include_timestamp=False, explicit_skill_name="synthetic-component-evidence-and-catalog-search"
        ).render_all([result]),
        encoding="utf-8",
    )
    page.set_viewport_size({"width": width, "height": 1000})
    _open(page, tmp_path, "tier1-long-evidence")
    table = page.locator(".top-issues-table").first
    assert table.is_visible()
    assert table.locator(".quality-pill").count() == 1
    broken = table.locator("th,.quality-pill,.severity-tag,.count-badge").evaluate_all(r"""nodes=>{
        const broken=[];
        for(const element of nodes){
            const walker=document.createTreeWalker(element,NodeFilter.SHOW_TEXT);
            while(walker.nextNode()){
                const node=walker.currentNode;
                for(const match of node.textContent.matchAll(/[A-Za-z]{2,}|\d+/g)){
                    const range=document.createRange();
                    range.setStart(node,match.index);range.setEnd(node,match.index+match[0].length);
                    if(new Set([...range.getClientRects()].map(rect=>Math.round(rect.top))).size>1)
                        broken.push(match[0]);
                }
            }
        }
        return broken;
    }""")
    assert not broken, (width, broken)
    pill = table.locator(".quality-pill")
    assert pill.evaluate("node=>node.scrollWidth<=node.clientWidth+1")
    evidence = table.locator("tbody td").first
    assert evidence.inner_text() == message
    assert evidence.evaluate(
        """(node,token)=>{
        const text=node.firstChild,range=document.createRange(),start=text.textContent.indexOf(token);
        range.setStart(text,start);range.setEnd(text,start+token.length);
        return new Set([...range.getClientRects()].map(rect=>Math.round(rect.top))).size>1;
    }""",
        token,
    )
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    page.set_viewport_size({"width": 794, "height": 1123})
    page.emulate_media(media="print")
    assert table.evaluate(
        "node=>node.getBoundingClientRect().width<=node.parentElement.getBoundingClientRect().width+2"
    )


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize(
    "fixture,tab,selector",
    [
        ("full", "overview", "#tier3-dimension-table"),
        ("native-telemetry", "trials", "#tier3-token-usage table,#tier3-latency-metrics table"),
    ],
)
def test_detailed_tables_keep_short_words_and_numbers_intact_with_bounded_scrolling(
    page, reports, width, fixture, tab, selector
):
    directory, _, _ = reports
    page.set_viewport_size({"width": width, "height": 1000})
    _open(page, directory, fixture)
    _tier(page, "tier3")
    _subtab(page, tab)
    tables = page.locator(selector)
    assert tables.count() > 0
    tables.evaluate_all("""nodes=>nodes.forEach(node=>{
        for(let parent=node.parentElement;parent;parent=parent.parentElement)
            if(parent.tagName==='DETAILS')parent.open=true;
    })""")
    broken = tables.evaluate_all(r"""tables=>{
        const broken=[];
        for(const table of tables){
            const walker=document.createTreeWalker(table,NodeFilter.SHOW_TEXT);
            while(walker.nextNode()){
                const node=walker.currentNode;
                if(node.parentElement.closest('code'))continue; // Long recorded identifiers may wrap.
                for(const match of node.textContent.matchAll(/[A-Za-z]{2,}|\d[\d,]*(?:\.\d+)?/g)){
                    const range=document.createRange();
                    range.setStart(node,match.index);range.setEnd(node,match.index+match[0].length);
                    const lines=new Set([...range.getClientRects()].map(rect=>Math.round(rect.top)));
                    if(lines.size>1)broken.push(match[0]);
                }
            }
        }
        return broken;
    }""")
    assert not broken, (fixture, width, broken)
    for table in tables.all():
        bounds = table.evaluate("""node=>{
            const wrapper=node.closest('.table-scroll');
            return {width:wrapper.getBoundingClientRect().width,scroll:wrapper.scrollWidth,
                client:wrapper.clientWidth,overflow:getComputedStyle(wrapper).overflowX};
        }""")
        assert bounds["width"] <= width
        if bounds["scroll"] > bounds["client"]:
            assert bounds["overflow"] == "auto"
            assert table.evaluate(
                "node=>{const parent=node.closest('.table-scroll');parent.scrollLeft=100;return parent.scrollLeft>0}"
            )
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
