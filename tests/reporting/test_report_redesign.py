# SPDX-License-Identifier: Apache-2.0
"""HTML report layout: header, executive summary, Fix first, vocabulary, print and accessibility."""

import json
import re
from html.parser import HTMLParser
from itertools import pairwise
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
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
        self.scripts: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(int(tag[1]))
        self.ids.extend(value for key, value in attrs if key == "id" and value)
        if tag == "script":
            self._in_script = True
            self.scripts.append("")

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script:
            self.scripts[-1] += data


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


def test_quality_gate_stays_actionable_with_advisory_findings() -> None:
    quality = ValidationResult(validator_name="QUALITY", validator_description="Quality dimensions")
    quality.add_finding(_finding("medium", "Optional version missing", "SKILL.md"))
    reason = "Quality score 98.2/100 is below the required 100/100"
    quality.fail_gate(reason)

    items = fix_first_for_results([quality], skill_name="demo")
    assert items[0]["kind"] == "check"
    assert items[0]["message"] == reason
    assert items[0]["anchor"].startswith("c-")
    html = _render([quality])
    fix_first = html.split('class="fix-first"', 1)[1].split("</section>", 1)[0]
    assert reason in fix_first
    assert f'href="#{items[0]["anchor"]}"' in fix_first
    assert f'id="{items[0]["anchor"]}"' in html


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


def test_fix_first_uses_the_scan_root_for_a_repository_root_skill() -> None:
    risk = ValidationResult(validator_name="Code Risk Analysis", validator_description="Static analysis")
    risk.add_finding(_finding("high", "subprocess call with shell=True", "/workspace/scripts/fetch.py", 21))
    scan = ValidationResult(validator_name="Security Scan", validator_description="Detect vulnerabilities")
    scan.add_finding(_finding("high", "Tool Misuse: subprocess.run(command, shell=True", "scripts/fetch.py", 21))

    items = fix_first_for_results([risk, scan], skill_name=".", expected_skill_name="workspace", scan_root="/workspace")
    html = HTMLReporter(include_timestamp=False, explicit_skill_name=".", scan_root="/workspace").render_all(
        [risk, scan]
    )
    fix_first = html.split('class="fix-first"', 1)[1].split("</section>", 1)[0]

    # Absolute and relative paths to one file and line take one slot, shown relative to the skill.
    assert [item["location"] for item in items] == ["scripts/fetch.py:21"]
    assert re.findall(r"<code>([^<]+)</code>", fix_first) == ["scripts/fetch.py:21"]


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
    assert summary.count("A finding is one problem a check reports,") == 1
    assert "each location is an occurrence" in summary
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

    assert 'class="header-status-badge' not in header
    assert 'class="hero-command-status verdict-chip fail">Changes required</p>' in html
    assert "Download JSON" in header
    assert re.findall(r"<h1[^>]*>([^<]+)</h1>", html) == ["demo"]


def _gated(results: list[ValidationResult], tier: int, *, blocking: bool = True) -> list[ValidationResult]:
    for result in results:
        result.metadata["gating"] = {"tier": tier, "blocking": blocking}
    return results


def _header_verdict(html: str) -> list[tuple[str, str]]:
    return re.findall(r'<p class="hero-command-status verdict-chip (\w+)">([^<]+)</p>', html)


def _tier_card_verdicts(html: str) -> list[tuple[str, str]]:
    return re.findall(r'tier-card-label">(Tier \d)</span>\s*<span class="tier-card-verdict">([^<]+)</span>', html)


def test_header_keeps_required_gate_separate_from_skipped_execution() -> None:
    skipped = []
    for name in ("Secrets Detection", "PII Scan"):
        result = ValidationResult(validator_name=name)
        result.metadata.update(skipped=True, skip_reason="Disabled by --checks")
        skipped.append(result)

    html = _render(_gated(skipped, 1))
    summary = html.split('<section class="exec-summary', 1)[1].split("</section>", 1)[0]

    # The command gate passed; the evidence summary still says nothing ran.
    assert _header_verdict(html) == [("pass", "Passed required checks")]
    assert '<span class="verdict-chip warn">Skipped</span>' in summary
    assert "0 of 0 checks passed" not in summary


def test_failing_advisory_tier_is_labelled_advisory() -> None:
    def results(blocking: bool) -> list[ValidationResult]:
        license_check = ValidationResult(validator_name="License Compliance")
        license_check.add_success("license", "License found")
        similarity = ValidationResult(validator_name="Similarity Check")
        similarity.add_error("Similarity provider failed")
        return [*_gated([license_check], 1), *_gated([similarity], 2, blocking=blocking)]

    advisory = _render(results(blocking=False))
    blocking = _render(results(blocking=True))

    # A failing tier whose gate does not block never says "Changes required".
    assert _header_verdict(advisory) == [("pass", "Passed required checks")]
    assert _tier_card_verdicts(advisory) == [("Tier 1", "Passed"), ("Tier 2", "Failed (advisory)")]
    assert _header_verdict(blocking) == [("fail", "Changes required")]
    assert _tier_card_verdicts(blocking) == [("Tier 1", "Passed"), ("Tier 2", "Changes required")]


def test_failing_advisory_tier3_is_labelled_advisory() -> None:
    agent_eval = ValidationResult(validator_name="AGENT_EVAL")
    agent_eval.add_error("Harbor execution failed")

    html = HTMLReporter(include_timestamp=False).render_all(_gated([agent_eval], 3, blocking=False))

    assert _tier_card_verdicts(html) == [("Tier 3", "Failed (advisory)")]
    assert "Tier 3 quality: Failed (advisory)</span>" in html
    assert "Changes required" not in html.split("<main", 1)[1].split('id="tab-tier3"', 1)[0]


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


def test_theme_selector_has_a_name_and_system_light_dark_choices() -> None:
    html = _render(_results())
    assert '<label class="theme-control" for="report-theme">Theme' in html
    assert '<select id="report-theme"' in html
    assert all(f'<option value="{theme}">' in html for theme in ("system", "light", "dark"))
    assert "matchMedia('(prefers-color-scheme: dark)')" in html
    assert "localStorage.setItem('skillevaluator-theme', reportThemePreference)" in html


def test_tier3_charts_are_redrawn_with_print_colors() -> None:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = build_agent_eval_payload(
        "example",
        {
            "agent": {
                "execution_status": "succeeded",
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0},
                "rewards": [{"entry_id": "case-1", "security": 1.0}],
            }
        },
        use_llm_judge=False,
    )
    html = HTMLReporter(include_timestamp=False).render_all([result])
    script = next(script for script in _outline(html).scripts if "rebuildTier3Charts" in script)

    # A canvas keeps the colors it was drawn with, so printing redraws every
    # chart with the light (print) colors and restores the theme afterwards.
    # Print mode is enabled first; the shared disclosure handler then redraws
    # after it exposes inactive chart containers for synchronous measurement.
    assert 'window.addEventListener("beforeprint", function(){ printing = true; });' in script
    assert 'window.addEventListener("afterprint", function(){ redrawForPrint(false); });' in script
    assert 'window.matchMedia("print")' in script
    assert 'if (printing && theme === "dark") root.setAttribute("data-theme", "light");' in script
    # All four chart types' print redraw, colors and theme restoration are
    # executed in test_chart_dependency and the offline browser efficiency
    # tests rather than counting constructor sites.


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
    css = (TEMPLATES / "report_ui.css.j2").read_text(encoding="utf-8")
    print_rules = [(selector, body) for media, selector, body in _rules(css) if "@media print" in media]
    palette = next(body for selector, body in print_rules if "[data-theme=dark]" in selector)
    assert "--bg:#fff" in palette and "--dark-text:#1e293b" in palette
    assert "--badge-fail-bg:#fee2e2" in palette


def test_print_shows_the_header_and_one_results_view() -> None:
    css = (TEMPLATES / "report_ui.css.j2").read_text(encoding="utf-8")
    print_rules = [(selector, body) for media, selector, body in _rules(css) if "@media print" in media]
    shown = " ".join(selector for selector, body in print_rules if "display:block" in body)
    hidden = " ".join(selector for selector, body in print_rules if "display:none" in body)

    assert ".main-header" in shown
    assert ".header-controls" in hidden and ".skip-link" in hidden
    assert ".results-view:not(#results-by-skill)" in hidden


def test_native_brand_header_replaces_sidebar_and_decorative_logo() -> None:
    html = _render(_results())
    header = html.split('<header class="main-header">', 1)[1].split("</header>", 1)[0]
    assert '<div class="logo-section">' in header
    assert '<p class="header-title">SkillEvaluator</p>' in header
    assert '<aside class="report-sidebar"' not in html
    assert len(_outline(html).ids) == len(set(_outline(html).ids))


def test_native_body_uses_readable_sixteen_pixel_default() -> None:
    html = _render(_results())
    body = re.search(r"body\s*\{([^}]+)\}", html).group(1)
    assert "line-height: 1.6" in body
    assert "font-family: -apple-system" in body
    assert "font-size:" not in body  # Browser default is 16px; no workspace shrink.


def _px(value: str, unit: str) -> float:
    return float(value) * (16 if unit == "rem" else 1)


def _style_block_digests() -> list[str]:
    from hashlib import sha256

    template = (TEMPLATES / "report.html.j2").read_text(encoding="utf-8")
    # Comments are stripped, so only declarations change a digest.
    blocks = [
        re.sub(r"/\*.*?\*/", "", block, flags=re.DOTALL)
        for block in re.findall(r"<style>(.*?)</style>", template, re.DOTALL)
    ]
    return [sha256(block.encode()).hexdigest() for block in blocks]


def test_report_style_blocks_match_recorded_digests() -> None:
    """Snapshot of this template's style blocks: a CSS edit must update the digests on purpose."""
    assert _style_block_digests() == [
        "e09704d0a4a030b3fd7de5e063a5247c0770733ad8c5b380a24e194829c0364e",
        "2f7d5555e6ff116aa4f95c0e3a4b620a549267c9ec38c95a7dd3a30fd9a04bc6",
        "f02e0b64bdb8ebe7e9fd65c6fd45cace436a961a60a0b6552d18a06db7780cba",
        "1997da9ee6f5b7cdefabbebffe2fa3d0101fce091f533e2f59403c2e2ce58875",
    ]


def test_presentation_manifest_matches_the_template_and_its_checks() -> None:
    root = TEMPLATES.parents[3]
    manifest = json.loads((root / "docs" / "report-presentation-port.json").read_text(encoding="utf-8"))
    assert (root / manifest["template"]).samefile(TEMPLATES / "report.html.j2")
    base, tier3, *_ = _style_block_digests()
    recorded = {entry["block"]: entry["comment_free_sha256"] for entry in manifest["native_styles"]}
    assert recorded == {"base": base, "tier3": tier3}
    for check in manifest["drift_checks"]:
        path, _, name = check.partition("::")
        source = root / path
        assert source.is_file(), check
        assert not name or f"def {name}(" in source.read_text(encoding="utf-8"), check


def _contrast_ratio(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _hex_color(value: str) -> str:
    color = value.replace("!important", "").strip().lower()
    color = {"white": "#ffffff"}.get(color, color)
    if re.fullmatch(r"#[0-9a-f]{3}", color):
        color = "#" + "".join(char * 2 for char in color[1:])
    assert re.fullmatch(r"#[0-9a-f]{6}", color), f"not a plain colour: {value!r}"
    return color


def _screen_rules(html: str) -> list[tuple[str, str]]:
    """Every screen style rule in document order, as (selector, declarations)."""
    styles = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL))
    css = re.sub(r"/\*.*?\*/", "", styles, flags=re.DOTALL)
    return [
        (re.sub(r"\s+", " ", part.replace('"', "")).strip(), body)
        for media, selector, body in _rules(css)
        if all(prelude == "@media screen" for prelude in media)
        for part in selector.split(",")
    ]


def _specificity(selector: str) -> tuple[int, int]:
    return (
        len(re.findall(r"#[\w-]+", selector)),
        len(re.findall(r"\.[\w-]+|\[[^\]]+\]|(?<!:):[\w-]+", selector)),
    )


def _resolved_colors(rules: list[tuple[str, str]], theme: str, matching: list[str]) -> tuple[str, str]:
    """Foreground and background the cascade gives an element that `matching` selectors select."""
    variables: dict[str, str] = {}
    for selector, body in rules:
        if selector == ":root" or (theme == "dark" and selector == "[data-theme=dark]"):
            variables.update(re.findall(r"(--[\w-]+)\s*:\s*([^;]+)", body))

    def resolve(value: str) -> str:
        while match := re.search(r"var\((--[\w-]+)\)", value):
            value = value.replace(match.group(0), variables[match.group(1)])
        return _hex_color(value)

    applies = [
        selector
        for selector in matching
        if not selector.startswith("[data-theme=") or selector.startswith(f"[data-theme={theme}]")
    ]
    winners: dict[str, tuple[tuple[int, int, int], str]] = {}
    for order, (selector, body) in enumerate(rules):
        if selector not in applies:
            continue
        for prop, value in re.findall(r"(?:^|[;\s])(color|background|background-color)\s*:\s*([^;]+)", body):
            key = "color" if prop == "color" else "background"
            rank = (*_specificity(selector), order)
            if key not in winners or rank >= winners[key][0]:
                winners[key] = (rank, value)
    assert winners.keys() == {"color", "background"}, (matching, winners)
    return resolve(winners["color"][1]), resolve(winners["background"][1])


_CHIP_AND_BADGE_SELECTORS = {
    "pass tier chip": [".tier-card.pass .tier-card-verdict"],
    "fail tier chip": [".tier-card.fail .tier-card-verdict"],
    "warn tier chip": [".tier-card.warn .tier-card-verdict"],
    "skipped tier chip": [".tier-card.skipped .tier-card-verdict"],
    "issue count badge": [".count-badge"],
    "medium issue count badge": [".count-badge", ".count-badge.medium-count"],
    "low issue count badge": [".count-badge", ".count-badge.low-count", "[data-theme=dark] .count-badge.low-count"],
    "incomplete badge": [".t3-incomplete-badge"],
    "hovered skill pill": [".skill-pill", ".skill-pill:hover"],
    "hovered quality pill": [".quality-pill", ".quality-pill:hover"],
    "expanded validator header": [".validator-collapsible", ".validator-collapsible.active"],
    "back to top button": ["#backToTop"],
    "hovered back to top button": ["#backToTop", "#backToTop:hover"],
}


def test_contrast_helper_matches_the_published_ratios() -> None:
    assert round(_contrast_ratio("#ffffff", "#22c55e"), 2) == 2.28
    assert round(_contrast_ratio("#ffffff", "#eab308"), 2) == 1.92
    assert round(_contrast_ratio("#ffffff", "#ef4444"), 2) == 3.76


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("name", _CHIP_AND_BADGE_SELECTORS)
def test_status_chips_and_badges_have_readable_text(name: str, theme: str) -> None:
    rules = _screen_rules(_render(_results()))
    foreground, background = _resolved_colors(rules, theme, _CHIP_AND_BADGE_SELECTORS[name])
    assert _contrast_ratio(foreground, background) >= 4.5, (name, theme, foreground, background)


def test_table_overflow_is_scoped_to_scroll_containers() -> None:
    css = (TEMPLATES / "report_ui.css.j2").read_text(encoding="utf-8")
    assert ".table-scroll,.table-container,.summary-table-container,.top-issues-content,.contributor-content" in css
    assert "max-width:100%;overflow-x:auto" in css
