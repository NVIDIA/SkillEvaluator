# SPDX-License-Identifier: Apache-2.0
"""Real-browser checks for generated full/stress fixtures, without project dependencies.

Run with: uv run --no-project --with playwright python
 tests/reporting/browser_navigation_checks.py REPORT_DIRECTORY OUTPUT_DIRECTORY
Uses system Chrome; no live evaluation or network resources are required.
REPORT_DIRECTORY contains full.html/stress.html plus absent.html rendered with
HTMLReporter(requested_tiers=(1,), displayed_tiers=(1, 2, 3)) and one Tier 1 result.
"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(page, condition, message):
    try:
        page.wait_for_function(condition, timeout=3000)
    except Exception as error:
        raise AssertionError(message) from error


def run(reports, output):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True
        )
        for width in (1440, 760, 390):
            for theme in ("light", "dark"):
                context = browser.new_context(
                    viewport={"width": width, "height": 1000}, color_scheme=theme, reduced_motion="reduce"
                )
                # Task 2 owns chart-runtime offline behavior; navigation must work without it.
                context.route("https://**/*", lambda route: route.abort())
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
                page.goto((reports / "full.html").as_uri())
                page.select_option("#report-theme", "system")
                check(page, f"() => document.documentElement.dataset.theme === '{theme}'", "system theme")
                opposite = "light" if theme == "dark" else "dark"
                page.emulate_media(color_scheme=opposite)
                check(page, f"() => document.documentElement.dataset.theme === '{opposite}'", "live system theme")
                page.select_option("#report-theme", theme)
                page.reload()
                check(page, f"() => document.documentElement.dataset.theme === '{theme}'", "explicit theme persists")
                page.emulate_media(color_scheme=opposite)
                check(
                    page,
                    f"() => document.documentElement.dataset.theme === '{theme}'",
                    "explicit preference ignores system",
                )
                page.evaluate("""() => {
                    window._navigationScroll=[];
                    const original=Element.prototype.scrollIntoView;
                    Element.prototype.scrollIntoView=function(options) { window._navigationScroll.push(options.behavior); original.call(this,options); };
                }""")
                check(
                    page,
                    "() => document.querySelector('.hero-command-status.verdict-chip.fail')?.innerText === 'Changes required' && document.querySelector('.hero-side .verdict-chip').innerText.startsWith('Tier 3 quality: ')",
                    "visible outcome hierarchy",
                )
                page.screenshot(path=str(output / f"hierarchy-{width}-{theme}.png"))
                # Every entry point synchronizes all other surfaces and focuses its panel.
                for tier in ("tier2", "tier3", "tier1"):
                    page.locator(f'.tier-card[data-target-tab="{tier}"]').click()
                    check(
                        page,
                        f"""() => document.activeElement.id === 'tab-{tier}' &&
                        document.querySelectorAll('.tab-content.active').length === 1 &&
                        document.querySelector('.tab-content.active').id === 'tab-{tier}' &&
                        document.querySelector('.tier-card[data-target-tab="{tier}"]').getAttribute('aria-pressed') === 'true' &&
                        document.querySelector('.report-sidebar .sidebar-current').dataset.reportTier === '{tier}' &&
                        document.querySelector('.mobile-tier-nav [aria-current]').dataset.reportTier === '{tier}'""",
                        "tier synchronization",
                    )
                tier_nav = ".mobile-tier-nav" if width <= 760 else ".report-sidebar"
                page.locator(f'{tier_nav} [data-report-tier="tier3"]').click()
                # Chart/table references need to reveal the containing local view as well.
                page.evaluate("""() => {
                    activateReportTier('tier1');
                    const ref=document.createElement('a'); ref.href='#t3-trials-table'; ref.id='test-trial-reference';
                    ref.textContent='Trials evidence'; document.body.append(ref);
                }""")
                page.locator("#test-trial-reference").click()
                check(
                    page,
                    "() => document.querySelector('#t3-trials-table').getClientRects().length > 0 && document.activeElement.id === 't3-trials-table'",
                    "generic local evidence link",
                )
                page.locator("#test-trial-reference").evaluate("(node) => node.remove()")
                # Manual-activation roving tabs: arrows move focus; Enter/Space changes view.
                overview = page.locator("#tier3-tab-overview")
                overview.click()
                overview.press("ArrowRight")
                check(
                    page,
                    "() => document.activeElement.id === 'tier3-tab-suggestions' && document.querySelector('#tier3-tab-overview').getAttribute('aria-selected') === 'true'",
                    "arrow focus",
                )
                page.keyboard.press("Enter")
                check(
                    page,
                    "() => document.querySelector('#tier3-page-suggestions').classList.contains('active')",
                    "enter activation",
                )
                page.keyboard.press("End")
                page.keyboard.press("Space")
                check(
                    page,
                    "() => document.querySelector('#tier3-page-provenance').classList.contains('active')",
                    "end/space activation",
                )
                page.keyboard.press("Home")
                page.keyboard.press("ArrowLeft")
                check(page, "() => document.activeElement.id === 'tier3-tab-provenance'", "left wraps")
                page.locator("#tier3-tab-agents").click()
                agent_tabs = page.locator(".t3-agent-tab-btn")
                agent_tabs.first.focus()
                page.keyboard.press("End")
                page.keyboard.press("Enter")
                check(
                    page,
                    "() => document.querySelector('.t3-agent-panel.active').id === document.activeElement.getAttribute('aria-controls')",
                    "agent keyboard",
                )
                # Deep links use stable panel IDs even when agent names contain selector syntax.
                page.evaluate("""() => {
                    const anchor=document.querySelector('.t3-agent-anchor');
                    anchor.dataset.tier3AgentName='agent"[雪]/name';
                    document.querySelector('.t3-agent-tab-btn').dataset.tier3Agent=anchor.dataset.tier3AgentName;
                    jumpToTier3Agent(anchor.id);
                }""")
                page.wait_for_timeout(50)
                check(
                    page,
                    "() => document.querySelector('.t3-agent-panel.active').id === document.querySelector('.t3-agent-anchor').dataset.tier3AgentPanelId",
                    "unusual agent link",
                )
                # Focus lands on the named panel, never on the aria-hidden scroll anchor.
                check(
                    page,
                    "() => document.activeElement.id === document.querySelector('.t3-agent-anchor').dataset.tier3AgentPanelId",
                    "agent link focus",
                )
                anchor_id = page.locator(".t3-agent-anchor").first.get_attribute("id")
                page.goto((reports / "full.html").as_uri() + "#" + anchor_id)
                page.reload()
                check(
                    page,
                    "() => document.activeElement.id === document.querySelector('.t3-agent-anchor').dataset.tier3AgentPanelId && document.activeElement.getAttribute('aria-hidden') !== 'true'",
                    "agent fragment reload focus",
                )
                # Sticky tier and local navigation occupy separate vertical bands.
                page.evaluate("() => window.scrollBy(0, 350)")
                if width <= 760:
                    check(
                        page,
                        "() => document.querySelector('.t3-subtabs').getBoundingClientRect().top >= document.querySelector('.mobile-tier-nav').getBoundingClientRect().bottom - 1",
                        "sticky navigation overlap",
                    )
                # A finding deep-link reveals Tier 1, resets incompatible filters, and focuses exact evidence.
                page.evaluate("""() => {
                    document.querySelector('#report-search').value='no such finding';
                    filterSkills('pass'); filterSeverity('critical-high'); applyReportSearch();
                    openFinding(document.querySelector('.check-finding[id]').id);
                }""")
                check(
                    page,
                    """() => document.querySelector('.tab-content.active').id === 'tab-tier1' &&
                    document.activeElement.classList.contains('check-finding') &&
                    document.activeElement.getClientRects().length > 0 &&
                    document.activeElement.classList.contains('finding-flash') &&
                    document.querySelector('#report-search').value === ''""",
                    "exact finding reveal",
                )
                page.evaluate("""() => {
                    activateReportTier('tier3'); document.querySelector('#tier3-tab-overview').focus();
                    const finding=document.querySelector('.check-finding[id]');
                    scrollToFinding(finding.closest('.skill-card').dataset.skillName, finding.dataset.issueKey);
                }""")
                check(page, "() => document.activeElement.classList.contains('check-finding')", "legacy finding focus")
                check(
                    page,
                    "() => window._navigationScroll.length > 0 && window._navigationScroll.every(mode=>mode === 'auto')",
                    "reduced motion navigation",
                )
                # Print exposes filtered evidence and all local pages; restores disclosures afterward.
                page.evaluate("""() => {
                    document.querySelector('#report-search').value='no such finding'; applyReportSearch();
                    window._closed = Array.from(document.querySelectorAll('details:not([open]):not(.raw-output)')).map(x=>x);
                    dispatchEvent(new Event('beforeprint'));
                }""")
                page.emulate_media(media="print")
                check(
                    page,
                    """() => [...document.querySelectorAll('.t3-page,.t3-agent-panel,.check-finding')].every(e=>e.getClientRects().length > 0) && window._closed.every(e=>e.open)""",
                    "print evidence",
                )
                page.emulate_media(media="screen")
                page.evaluate("() => dispatchEvent(new Event('afterprint'))")
                check(
                    page,
                    "() => window._closed.every(e=>!e.open) && document.querySelector('#report-search').value === 'no such finding'",
                    "print restoration",
                )
                page.locator('.report-sidebar [data-report-section="run-metadata"]').click()
                page.wait_for_timeout(50)
                check(
                    page,
                    "() => document.querySelector('.report-sidebar .sidebar-current').dataset.reportSection === 'run-metadata'",
                    "run details navigation",
                )
                page.locator('.report-sidebar [data-report-section="report-workspace"]').click()
                page.wait_for_timeout(50)
                check(
                    page,
                    "() => document.querySelector('.report-sidebar .sidebar-current').dataset.reportSection === 'report-workspace'",
                    "summary navigation",
                )
                for fixture in ("full", "stress"):
                    page.goto((reports / f"{fixture}.html").as_uri())
                    check(
                        page,
                        "() => [...document.querySelectorAll('.tier-card')].every(tile=>tile.scrollWidth <= tile.clientWidth + 1)",
                        "tier tile content clipping",
                    )
                    for tier in ("tier1", "tier2", "tier3"):
                        page.evaluate("(tier) => activateReportTier(tier)", tier)
                        check(
                            page,
                            "() => document.documentElement.scrollWidth <= innerWidth + 1",
                            "page horizontal overflow",
                        )
                        page.screenshot(path=str(output / f"{fixture}-{width}-{theme}-{tier}.png"))
                    page.evaluate("() => { activateReportTier('tier1'); toggleAllSkills(true); }")
                    check(
                        page,
                        """() => [...document.querySelectorAll('.skill-content.active,.validator-content.active,.top-issues-content.active,.table-container.active')].every(e=>getComputedStyle(e).maxHeight === 'none')""",
                        "expanded evidence cap",
                    )
                # A real URL reload must use the same reveal/focus path as an in-page link.
                page.goto((reports / "full.html").as_uri() + "#validator-1")
                page.reload()
                check(
                    page,
                    "() => document.querySelector('#results-by-validator').classList.contains('active') && document.querySelector('#validator-1').getClientRects().length > 0 && document.activeElement.id === 'validator-1' && document.querySelector('#validator-1 .validator-content').classList.contains('active')",
                    "validator fragment reload",
                )
                # Whole tiers and the skip link are sections: open at their top, never centred.
                page.goto((reports / "full.html").as_uri() + "#tab-tier3")
                page.reload()
                check(
                    page,
                    "() => document.querySelector('.tab-content.active').id === 'tab-tier3' && Math.abs(document.getElementById('tab-tier3').getBoundingClientRect().top) < 200",
                    "tier fragment reload",
                )
                page.goto((reports / "full.html").as_uri())
                page.evaluate("() => window.scrollTo(0, document.documentElement.scrollHeight)")
                page.locator(".skip-link").focus()
                page.keyboard.press("Enter")
                check(
                    page,
                    "() => document.activeElement.id === 'report-workspace' && Math.abs(document.getElementById('report-workspace').getBoundingClientRect().top) < 2",
                    "skip link",
                )
                page.goto((reports / "absent.html").as_uri())
                page.select_option("#report-theme", theme)
                contrast = page.evaluate(r"""() => {
                    const rgb = value => (value.match(/[\d.]+/g) || []).map(Number);
                    const luminance = color => color.slice(0,3).map(v=>v/255).map(v=>v<=0.04045?v/12.92:((v+0.055)/1.055)**2.4).reduce((sum,v,i)=>sum+v*[0.2126,0.7152,0.0722][i],0);
                    return [...document.querySelectorAll('.tier-card.skipped .tier-card-label')].map(label=>{
                        const tile=label.closest('.tier-card');
                        let background=rgb(getComputedStyle(tile).backgroundColor);
                        const foreground=rgb(getComputedStyle(label).color);
                        let opacity=1;
                        for(let node=label;node;node=node.parentElement) opacity*=Number(getComputedStyle(node).opacity);
                        if(background[3] === 0) background=rgb(getComputedStyle(tile.parentElement.closest('section')).backgroundColor);
                        const rendered=foreground.slice(0,3).map((v,i)=>v*opacity+background[i]*(1-opacity));
                        const values=[luminance(rendered),luminance(background)].sort((a,b)=>a-b);
                        return {tier:tile.dataset.targetTab,opacity,ratio:(values[1]+0.05)/(values[0]+0.05)};
                    });
                }""")
                assert len(contrast) == 2 and all(value["ratio"] >= 4.5 for value in contrast), contrast
                page.screenshot(path=str(output / f"absent-{width}-{theme}.png"))
                assert not errors, errors
                records.append(
                    {"width": width, "theme": theme, "errors": errors, "checks": "passed", "absentContrast": contrast}
                )
                context.close()
                no_js = browser.new_context(
                    viewport={"width": width, "height": 1000}, java_script_enabled=False, color_scheme=theme
                )
                page = no_js.new_page()
                page.goto((reports / "full.html").as_uri())
                check(
                    page,
                    "() => [...document.querySelectorAll('.t3-page,.t3-agent-panel,.check-finding')].every(e=>e.getClientRects().length > 0)",
                    "no-JS evidence",
                )
                check(page, "() => document.documentElement.scrollWidth <= innerWidth + 1", "no-JS page overflow")
                check(
                    page,
                    f"() => getComputedStyle(document.documentElement).colorScheme === '{theme}'",
                    "no-JS system theme",
                )
                no_js.close()
        storage_context = browser.new_context()
        storage_context.route("https://**/*", lambda route: route.abort())
        storage_context.add_init_script(
            "Object.defineProperty(window, 'localStorage', {get() { throw new Error('storage blocked'); }});"
        )
        storage_page = storage_context.new_page()
        storage_page.goto((reports / "full.html").as_uri())
        storage_page.select_option("#report-theme", "dark")
        check(storage_page, "() => document.documentElement.dataset.theme === 'dark'", "storage-independent theme")
        storage_context.close()
        browser.close()
    (output / "navigation-results.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.reports.resolve(), args.output.resolve())
