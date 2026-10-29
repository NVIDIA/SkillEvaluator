# SPDX-License-Identifier: Apache-2.0
"""HTML report layout: header, executive summary, Fix first, vocabulary, print and accessibility."""

import re
from html.parser import HTMLParser
from itertools import pairwise
from pathlib import Path

from skillevaluator.models import Finding, ValidationResult
from skillevaluator.reporting.html import HTMLReporter, fix_first_for_results, plain_check_name

TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "reporting" / "templates"


def _finding(severity: str, message: str, file_path: str = "", line: int | None = None) -> Finding:
    return Finding(
        category="SECURITY",
        severity=severity,
        check_name="check",
        message=message,
        file_path=file_path,
        line_number=line,
    )


def _results() -> list[ValidationResult]:
    """One skill: the same secret from two scanners, mixed severities, a failing quality score."""
    pii = ValidationResult(validator_name="PII Scan", validator_description="Detect PII")
    pii.add_finding(_finding("critical", "Hardcoded key in code", "scripts/fetch.py", 11))
    secrets = ValidationResult(validator_name="Secrets Detection", validator_description="Detect secrets")
    secrets.add_finding(_finding("critical", "API key found", "scripts/fetch.py", 11))
    risk = ValidationResult(validator_name="Code Risk Analysis", validator_description="Static analysis")
    risk.add_finding(_finding("medium", "Request without timeout", "scripts/fetch.py", 15))
    risk.add_finding(_finding("high", "Shell injection", "scripts/fetch.py", 21))
    lint = ValidationResult(validator_name="SCRIPT_LINT", validator_description="Lint scripts")
    lint.add_finding(_finding("low", "Line too long", "scripts/fetch.py", 3))
    quality = ValidationResult(validator_name="QUALITY", validator_description="Quality dimensions")
    quality.add_error("Quality score 66.2/100 is below the required 70/100")
    quality.metadata["quality_config"] = {"source": "packaged", "min_score": 70}
    quality.metadata["quality_scores"] = {
        "skill_name": "demo",
        "overall_score": 66.2,
        "grade": "D",
        "skill_type": "script-based",
        "dimensions": {"correctness": {"score": 66.2, "weight": 1.0, "issues_count": 0, "issues": []}},
    }
    clean = ValidationResult(validator_name="License Compliance", validator_description="Licenses")
    clean.add_success("license", "License found")
    return [pii, secrets, risk, lint, quality, clean]


def _render(results: list[ValidationResult]) -> str:
    return HTMLReporter(include_timestamp=False, explicit_skill_name="demo").render_all(results)


class _Outline(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.headings: list[int] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(int(tag[1]))
        self.ids.extend(value for key, value in attrs if key == "id" and value)


def _outline(html: str) -> _Outline:
    parser = _Outline()
    parser.feed(html)
    return parser


def test_fix_first_ranks_by_severity_and_lists_each_location_once() -> None:
    items = fix_first_for_results(_results(), skill_name="demo")

    assert [item["severity"] for item in items] == ["critical", "high", "high", "medium", "low"]
    # The secret both scanners report at scripts/fetch.py:11 takes one slot.
    assert [item["location"] for item in items].count("scripts/fetch.py:11") == 1
    assert items[0]["check"] == "PII Scan"
    # A failed check without a located finding keeps its reason in the list.
    assert {"kind": "check", "check": "Quality score"}.items() <= items[2].items()
    assert items[2]["message"] == "Quality score 66.2/100 is below the required 70/100"
    assert items[-1]["blocking"] is False


def test_fix_first_puts_an_unfinished_scan_first_when_incomplete() -> None:
    results = _results()
    secrets = results[1]
    secrets.mark_scan_incomplete("gitleaks")

    items = fix_first_for_results(results, skill_name="demo")

    assert items[0]["kind"] == "incomplete"
    # Without a diagnostic the report names the scanner and claims no cause.
    assert items[0]["message"] == "Secrets Detection did not finish (gitleaks)"


def _timed_out_scan() -> list[ValidationResult]:
    """A scan that kept one partial finding, then timed out."""
    scan = ValidationResult(validator_name="Security Scan", validator_description="Detect security vulnerabilities")
    scan.add_finding(_finding("high", "Tool Misuse: subprocess.run(shell=True)", "scripts/run.py", 7))
    scan.mark_scan_incomplete("skillspector")
    scan.add_error("skillspector timed out after 300 seconds; partial results only")
    clean = ValidationResult(validator_name="License Compliance", validator_description="Licenses")
    clean.add_success("license", "License found")
    return [scan, clean]


def test_incomplete_scan_with_partial_findings_keeps_its_diagnostic_visible() -> None:
    html = _render(_timed_out_scan())
    card = html.split('id="c-demo-security-scan"', 1)[1].split('<div class="check-finding', 1)[0]
    raw_output = re.findall(r'<details class="raw-output">.*?</details>', card, flags=re.DOTALL)
    outside_raw_output = re.sub(r'<details class="raw-output">.*?</details>', "", card, flags=re.DOTALL)

    # The reason the scan stopped is shown, and printed, outside the collapsed output.
    assert '<p class="scan-diagnostic">skillspector timed out after 300 seconds; partial results only</p>' in (
        outside_raw_output
    )
    # Only the line that repeats the finding card folds away.
    assert len(raw_output) == 1
    assert "[SECURITY-HIGH] Tool Misuse" in raw_output[0]
    assert "timed out" not in raw_output[0]


def test_incomplete_scan_names_the_diagnostic_instead_of_a_missing_scanner() -> None:
    results = _timed_out_scan()
    html = _render(results)
    summary = html.split('<section class="exec-summary', 1)[1].split("</section>", 1)[0]

    items = fix_first_for_results(results, skill_name="demo")

    assert items[0]["message"] == (
        "Security Scan did not finish: skillspector timed out after 300 seconds; partial results only"
    )
    assert "Security Scan: skillspector timed out after 300 seconds; partial results only." in summary
    assert "unavailable" not in items[0]["message"]
    assert "Install or enable" not in summary


def test_missing_scanner_asks_for_an_install() -> None:
    secrets = ValidationResult(validator_name="Secrets Detection", validator_description="Detect secrets")
    secrets.add_warning("gitleaks not installed - secrets scanning skipped. Install with: brew install gitleaks")
    secrets.mark_scan_incomplete("gitleaks")

    html = _render([secrets])
    summary = html.split('<section class="exec-summary', 1)[1].split("</section>", 1)[0]
    items = fix_first_for_results([secrets], skill_name="demo")

    assert items[0]["message"].startswith("Secrets Detection did not finish: gitleaks not installed")
    assert "Install or enable the scanner and run the evaluation again." in summary


def test_fix_first_links_open_existing_cards() -> None:
    html = _render(_results())
    ids = set(_outline(html).ids)
    summary = html.split('class="fix-first"', 1)[1].split("</section>", 1)[0]
    anchors = re.findall(r'href="#([^"]+)" class="fix-link"', summary)

    assert len(anchors) == 5
    assert set(anchors) <= ids
    assert "onclick=\"return openFinding(this.getAttribute('href').slice(1))\"" in summary


def test_report_ids_are_unique() -> None:
    ids = _outline(_render(_results())).ids

    assert len(ids) == len(set(ids))


def test_executive_summary_states_one_verdict_reason_and_four_kpis() -> None:
    html = _render(_results())
    summary = html.split('<section class="exec-summary', 1)[1].split("</section>", 1)[0]

    assert '<span class="verdict-chip fail">Changes required</span>' in summary
    assert re.search(r"4 of 6 checks failed: [^.]*PII Scan[^.]*\.", summary)
    assert "3 critical or high findings in those checks block the verdict." in summary
    assert re.findall(r"<dt>([^<]+)</dt>", summary) == [
        "Checks passed",
        "Findings",
        "Blocking critical + high",
        "Quality",
    ]
    assert "Minimum 70 to pass" in summary
    assert summary.count("A finding is one problem a check reports at one location") == 1
    assert "<h3>Fix first</h3>" in summary


def test_passing_report_lists_advisory_findings_as_worth_a_look() -> None:
    lint = ValidationResult(validator_name="SCRIPT_LINT")
    lint.add_finding(_finding("low", "Line too long", "scripts/fetch.py", 3))

    html = _render([lint])

    assert '<span class="verdict-chip pass">Passed</span>' in html
    assert "<h3>Worth a look</h3>" in html
    assert "1 advisory finding does not block." in html


def test_header_shows_one_verdict_and_skill_name_is_the_only_h1() -> None:
    html = _render(_results())
    header = html.split('<header class="main-header">', 1)[1].split("</header>", 1)[0]

    assert re.findall(r'class="header-status-badge verdict-badge (\w+)" data-verdict="(\w+)"', header) == [
        ("fail", "fail")
    ]
    assert "Changes required" in header
    assert "Download JSON" in header
    assert re.findall(r"<h1[^>]*>([^<]+)</h1>", html) == ["demo"]


def test_heading_levels_do_not_skip() -> None:
    for results in (_results(), [ValidationResult(validator_name="SCHEMA")]):
        levels = _outline(_render(results)).headings
        assert levels[0] == 1
        assert all(later <= earlier + 1 for earlier, later in pairwise(levels)), levels


def test_publication_eligibility_sits_next_to_the_verdict() -> None:
    html = _render(_results())
    hero = html.split('<section class="agent-hero"', 1)[1].split("</section>", 1)[0]

    assert 'data-publication-status="fail"' in hero
    assert '<span class="status-badge fail">Not eligible</span>' in hero


def test_folder_scan_keeps_its_skill_count_hint() -> None:
    html = HTMLReporter(include_timestamp=False).render_all([ValidationResult(validator_name="Security Scan")])
    hero = html.split('<section class="agent-hero"', 1)[1].split("</section>", 1)[0]

    assert re.search(r'<p class="hero-skills">Skills: \d+ of \d+ passed</p>', hero)


def test_check_names_are_plain_with_the_raw_id_beside_them() -> None:
    html = _render(_results())

    assert plain_check_name("SCRIPT_LINT") == "Script lint"
    assert plain_check_name("NEW_SCANNER") == "New scanner"
    assert plain_check_name("Secrets Detection") == "Secrets Detection"
    assert 'Quality score <code class="check-id">QUALITY</code>' in html
    assert 'Script lint <code class="check-id">SCRIPT_LINT</code>' in html


def test_finding_types_are_most_severe_first() -> None:
    html = _render(_results())
    table = html.split('class="top-issues-table"', 1)[1].split("</table>", 1)[0]
    severities = re.findall(r'class="severity-tag severity-(\w+)"', table)
    rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

    assert severities == sorted(severities, key=rank.__getitem__, reverse=True)
    # Only lists that really are sorted by severity say so.
    assert html.count("most severe first") == 1
    assert "most severe first</h3>" in html


def test_check_table_rows_are_keyboard_reachable_links() -> None:
    html = _render(_results())
    table = html.split('class="summary-table"', 1)[1].split("</table>", 1)[0]
    links = re.findall(r'<a class="validator-link" href="#validator-(\d+)">', table)

    assert len(links) == len(_results())
    assert all(f'id="validator-{index}"' in html for index in links)


def test_theme_toggle_has_a_fixed_name_and_reports_the_dark_state() -> None:
    html = _render(_results())
    button = html.split('class="theme-toggle"', 1)[1].split("</button>", 1)[0]

    assert 'aria-pressed="false"' in button
    assert '<span id="theme-text">Dark theme</span>' in button
    assert "toggle.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false')" in html
    assert "getElementById('theme-text').textContent" not in html
    assert "matchMedia('(prefers-color-scheme: dark)')" in html


def _rules(css: str):
    """Yield (media preludes, selector, declarations) for every style rule."""
    stack: list[str] = []
    prelude = ""
    depth_media: list[bool] = []
    index = 0
    while index < len(css):
        char = css[index]
        if css.startswith("/*", index):
            index = css.index("*/", index) + 2
            continue
        if char == "{":
            head = prelude.strip()
            if head.startswith("@"):
                stack.append(head)
                depth_media.append(True)
                prelude = ""
            else:
                end = css.index("}", index)
                yield tuple(stack), head, css[index + 1 : end]
                prelude = ""
                index = end + 1
                continue
        elif char == "}":
            if depth_media and depth_media.pop():
                stack.pop()
            prelude = ""
        else:
            prelude += char
        index += 1


def test_dark_colors_never_reach_print() -> None:
    html = _render(_results())
    css = "\n".join(re.findall(r"<style>(.*?)</style>", html, flags=re.DOTALL))
    dark_rules = [(media, selector, body) for media, selector, body in _rules(css) if 'data-theme="dark"' in selector]

    assert dark_rules
    for media, selector, body in dark_rules:
        literal_colors = re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", body)
        assert not literal_colors or any(query.startswith("@media screen") for query in media), selector


def test_print_shows_the_header_and_one_results_view() -> None:
    css = (TEMPLATES / "report_ui.css.j2").read_text(encoding="utf-8")
    print_rules = [(selector, body) for media, selector, body in _rules(css) if "@media print" in media]
    shown = " ".join(selector for selector, body in print_rules if "display:block" in body)
    hidden = " ".join(selector for selector, body in print_rules if "display:none" in body)

    assert ".main-header" in shown
    assert ".header-brand .se-logo" in shown
    assert ".header-controls" in hidden and ".skip-link" in hidden
    assert "#results-by-validator>.validator-card" in hidden


def test_logo_is_in_the_sidebar_and_the_header_with_unique_ids() -> None:
    html = _render(_results())

    assert html.count('class="se-logo"') == 2
    assert 'class="brand-mark"' not in html
    # Both copies sit next to the visible name, so they are decorative and carry no ids.
    assert len(re.findall(r'<svg class="se-logo"[^>]*aria-hidden="true"', html)) == 2
    assert ' id="se-logo' not in html


def test_no_workspace_font_size_is_below_eleven_pixels() -> None:
    css = (TEMPLATES / "report_ui.css.j2").read_text(encoding="utf-8")
    sizes = [float(size) for size in re.findall(r"font-size:(\d+(?:\.\d+)?)px", css)]

    assert sizes
    assert min(sizes) >= 11


def test_short_table_cells_do_not_break_mid_word() -> None:
    css = (TEMPLATES / "report_ui.css.j2").read_text(encoding="utf-8")

    assert "td,th{overflow-wrap:anywhere}" not in css
    assert "td,th{overflow-wrap:break-word}th{white-space:nowrap}" in css
