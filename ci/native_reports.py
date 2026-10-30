# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- reusable CI report entrypoint.
"""Native SkillEvaluator reports with a small, accessible multi-skill index.

Only retained JSON is consumed. All executable markup comes from trusted native
templates; artifact-supplied HTML is never copied into the published site.
"""

from __future__ import annotations

import base64
import hashlib
import re
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser
from importlib import resources
from pathlib import Path, PurePosixPath

from skillevaluator.reporting.html import VERDICT_WORDS, fix_first_for_results, render_logo
from skillevaluator.reporting.retained import render_retained_report, retained_results

try:
    import report_views as views
except ModuleNotFoundError:
    from ci import report_views as views

# The scan job mounts the repository here, so scanners record finding paths below it.
WORKSPACE = PurePosixPath("/workspace")


class _InlineCode(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.scripts: list[str] = []
        self.handlers: list[str] = []
        self.current: list[str] | None = None
        self.external_script = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.handlers.extend(value for key, value in attrs if key.startswith("on") and value)
        if tag == "script":
            if "src" in attributes:
                self.external_script = True
            elif attributes.get("type", "").casefold() not in {"application/json", "application/ld+json"}:
                self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.current is not None:
            self.scripts.append("".join(self.current))
            self.current = None


@lru_cache(maxsize=1)
def _trusted_inline_code() -> frozenset[str]:
    """Read executable code from trusted template sources, never report content.

    Retained CI reports support Tier 1 only. Their executable script and event
    handlers are static; values are carried in escaped data attributes or inert
    JSON. Dynamic Tier 3 scripts are deliberately outside this allowlist.
    """
    templates = resources.files("skillevaluator.reporting").joinpath("templates")
    source = templates.joinpath("report.html.j2").read_text()
    # This include contains static UI code; Jinja drops its final newline.
    source = source.replace(
        '{% include "report_ui.js.j2" %}', templates.joinpath("report_ui.js.j2").read_text().removesuffix("\n")
    )
    scripts = re.findall(r"<script\s*>(.*?)</script\s*>", source, flags=re.DOTALL | re.IGNORECASE)
    handlers = [match[1:-1] for match in re.findall(r"""\bon\w+\s*=\s*("[^"]*"|'[^']*')""", source)]
    return frozenset(
        code
        for code in [*scripts, *(unescape(handler) for handler in handlers)]
        if "{{" not in code and "{%" not in code
    )


def constrain_native_html(document: str) -> str:
    """Reject executable markup absent from the trusted Tier 1 template."""
    parser = _InlineCode()
    parser.feed(document)
    code = {*parser.scripts, *parser.handlers}
    if parser.external_script or not code <= _trusted_inline_code():
        raise ValueError("Unexpected executable markup in the native report")
    hashes = sorted(
        {"'sha256-" + base64.b64encode(hashlib.sha256(code.encode()).digest()).decode() + "'" for code in code}
    )
    policy = (
        "default-src 'none'; script-src 'unsafe-hashes' "
        + " ".join(hashes)
        + "; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; "
        "base-uri 'none'; form-action 'none'; object-src 'none'"
    )
    charset = re.search(r'<meta charset="[^"]+">', document, re.IGNORECASE)
    if charset is None:
        raise ValueError("The trusted native template must declare its character encoding.")
    # Keep the encoding declaration inside the browser's first 1024-byte window.
    position = charset.end()
    return (
        document[:position]
        + '<meta http-equiv="Content-Security-Policy" content="'
        + views._h(policy)
        + '">'
        + document[position:]
    )


_STYLE = """
:root{--brand-core:#76b900;--brand-green:#3e6500;--bg:#fff;--page:#f5f6f7;--tx:#24292f;--muted:#59636e;--line:#dfe3e6;--line-soft:#eaecef;--th:#fafbfc;--link:#3e6500;--focus:#3d6b00;
--pass-bg:#edf5e4;--pass-tx:#315b25;--fail-bg:#fff0ef;--fail-tx:#a3232b;--warn-bg:#fff5df;--warn-tx:#7a4e00;--rail:#1e2328;--rail-tx:#bfc6ce;--rail-link:#34402d;--rail-link-tx:#e6f2d4;
--se-logo-panel:#152117;--se-logo-panel-edge:#304631;--se-logo-ring-outer:#39523a;--se-logo-ring-middle:#6f9264;--se-logo-ring-inner:#dbe8d6;--se-logo-core:#76b900;--se-logo-core-edge:#9bdb34;--se-logo-check:#fff;
font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--tx);background:var(--page);font-size:14px}
@media screen and (prefers-color-scheme:dark){:root{--bg:#1e252c;--page:#151a1f;--tx:#e1e6eb;--muted:#aab5bf;--line:#3b454f;--line-soft:#303b46;--th:#252d35;--link:#b0d684;--focus:#b0d684;
--pass-bg:#293e27;--pass-tx:#c4e7a0;--fail-bg:#412a2d;--fail-tx:#ffc1c4;--warn-bg:#403826;--warn-tx:#f4d792;--rail:#0f1317}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--tx)}a{color:var(--link);text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid var(--focus);outline-offset:4px}
.rail{position:fixed;inset:0 auto 0 0;width:210px;background:var(--rail);color:white;padding:32px 24px}.brand{display:flex;align-items:center;gap:10px;font-size:17px;font-weight:700;letter-spacing:-.5px;color:#fff;white-space:nowrap}.se-logo{flex:none;display:block}.rail p{color:var(--rail-tx);font-size:12px;line-height:1.6}.rail a{display:block;color:var(--rail-link-tx);margin-top:30px;padding:12px;background:var(--rail-link);border-radius:6px;text-decoration:none}
main{margin-left:210px;max-width:1400px;padding:36px 44px 60px}.eyebrow{text-transform:uppercase;letter-spacing:.1em;font-size:11px;font-weight:700;color:var(--muted)}h1{font-size:28px;letter-spacing:-.7px;margin:8px 0 12px}h2{font-size:19px;letter-spacing:-.3px;margin:0 0 16px}p{line-height:1.6}.muted{color:var(--muted);font-size:12px}.status{display:inline-block;border-radius:999px;border:1px solid currentColor;padding:4px 10px;font-size:12px;font-weight:700;white-space:nowrap;background:var(--warn-bg);color:var(--warn-tx);print-color-adjust:exact;-webkit-print-color-adjust:exact}.pass{background:var(--pass-bg);color:var(--pass-tx)}.fail{background:var(--fail-bg);color:var(--fail-tx)}.neutral{background:var(--th);color:var(--muted)}.hero{display:flex;justify-content:space-between;align-items:start;gap:20px}.button{display:inline-block;padding:10px 14px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--tx);font-size:12px;font-weight:600;text-decoration:none;white-space:nowrap}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric,section{background:var(--bg);border:1px solid var(--line);border-radius:8px}.metric{padding:17px 20px}.metric strong{display:block;font-size:26px;letter-spacing:-.5px;margin-bottom:5px}.metric span{font-size:12px;color:var(--muted)}section{padding:24px;margin-bottom:20px}.scope{font-size:12px;color:var(--muted);line-height:1.7;padding:0 0 20px}.scope strong{color:var(--tx)}.attention{border-left:3px solid var(--fail-tx)}.attention p:last-child{margin-bottom:0}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left}th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);background:var(--th);white-space:nowrap}th,td{padding:14px 12px;border-bottom:1px solid var(--line-soft)}td{font-size:13px}td:first-child{font-weight:600;min-width:180px;max-width:320px;overflow-wrap:anywhere}td .muted{font-weight:400}tr:last-child td{border-bottom:0}.score{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}.evidence{white-space:nowrap}summary{cursor:pointer;font-size:12px;font-weight:600}details p{font-size:12px;overflow-wrap:anywhere}code{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}.empty{padding:20px;background:var(--warn-bg);color:var(--warn-tx);border-radius:6px}footer{font-size:12px;color:var(--muted);margin-top:24px}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}
@media(max-width:850px){.rail{position:static;width:auto;padding:16px 20px}.rail p,.rail a{display:none}main{margin:0;padding:24px 20px}.hero{flex-wrap:wrap}.metrics{grid-template-columns:repeat(2,1fr)}section{padding:18px}.metric{padding:16px}h1{font-size:25px}}
@media(max-width:600px){table,tbody{display:block}thead{display:none}tr{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid var(--line);padding:12px 0;gap:10px}td{display:block;padding:0;border:0}td:first-child{grid-column:1/-1;min-width:0;max-width:none}td:nth-child(3)::before{content:'Quality: ';font-weight:400;color:var(--muted)}td:nth-child(4)::before{content:'Errors: ';color:var(--muted)}td:nth-child(5)::before{content:'Warnings: ';color:var(--muted)}.evidence{grid-column:1/-1}tr:last-child{border-bottom:0}}
@media print{.rail{position:static;width:auto;padding:0 0 8px;background:none;color:#111}.rail p,.rail a,.button{display:none}.brand{color:#111}.rail .se-logo{--se-logo-panel:#f2f7ec;--se-logo-panel-edge:#d7e5cc;--se-logo-ring-outer:#b8d3a5;--se-logo-ring-middle:#83aa66;--se-logo-ring-inner:#315b25;--se-logo-core-edge:#3e6500}main{margin:0;padding:0;max-width:none}.table-wrap{overflow:visible}section,.metric{break-inside:avoid}body{background:white}a{color:inherit}}
"""


def brand_html(uid: str) -> str:
    """Logo and product name for the rail of the CI index and the Pages history.

    The logo is decorative next to the visible name; ``uid`` is kept so a
    labelled copy on the same page would still get unique ids.
    """
    return f'<div class="brand">{render_logo(uid, 28, "")}<span>SkillEvaluator</span></div>'


# Run and skill statuses in the report vocabulary. Anything unrecognised is
# shown as Incomplete, never as a pass.
_STATUS_VERDICTS = {
    "PASS": "pass",
    "FAIL": "fail",
    "passed": "pass",
    "failed": "fail",
    "skipped": "skipped",
}


def verdict_word(status: object) -> str:
    """Return the report's verdict word for a run or skill status."""
    return VERDICT_WORDS[_STATUS_VERDICTS.get(str(status), "incomplete")]


def render_native_reports(reports: Path, summary: dict, context: dict) -> dict[str, str]:
    """Write all available views and return per-skill rendering diagnostics."""
    reports = Path(reports)
    if reports.is_symlink() or not reports.is_dir():
        raise ValueError("A regular report directory is required.")
    models = []
    render_errors = {}
    first_fixes = {}
    for row in summary.get("skills", []):
        if row.get("overall_status") == "error" and re.fullmatch(r"skill-[0-9]{3}", row.get("report_dir", "")):
            # The publisher intentionally discards rejected child JSON, so its
            # directory may not exist yet. Create only this validated leaf.
            (reports / row["report_dir"]).mkdir(exist_ok=True)
    # Read each skill's JSON once; the Markdown views reuse these parsed reports.
    skills = views.read_skill_models(reports, summary)
    for skill in skills:
        row, directory, data, issue = skill["row"], skill["directory"], skill["data"], skill["issue"]
        status = views._skill_status(row, data, issue)
        native_url = None
        if row.get("overall_status") == "error":
            issue = row.get("error") or "The evaluator did not produce consistent report evidence."
        elif data and directory is not None:
            try:
                # Finding paths are shown relative to the skill, also for a skill at the repository root.
                scan_root = (WORKSPACE / row["path"]).as_posix()
                document = render_retained_report(
                    data,
                    skill_name=row["path"],
                    # The runner mounts repository-root skills at /workspace.
                    expected_skill_name=Path(row["path"]).name or "workspace",
                    context={
                        **context,
                        "report_index_url": "../index.html",
                        "canonical_json_url": "skillevaluator-tier1.json",
                    },
                    scan_root=scan_root,
                )
                views._write(directory / "skillevaluator-tier1.html", constrain_native_html(document))
                native_url = f"{directory.name}/skillevaluator-tier1.html"
                # Name the same first fix as the skill's own report, with its anchor.
                fixes = fix_first_for_results(
                    retained_results(data),
                    skill_name=row["path"],
                    expected_skill_name=Path(row["path"]).name or "workspace",
                    scan_root=scan_root,
                )
                if fixes:
                    first_fixes[row["report_dir"]] = fixes[0]
            except (ValueError, TypeError, KeyError) as exc:
                issue = f"Native report could not be rendered: {exc}"
        if issue:
            # One invalid display path or malformed child must not erase the
            # diagnostics for every other skill. Raw evidence is left untouched.
            render_errors[row["report_dir"]] = issue
            data = {}
            status = "INCOMPLETE"
            if directory is not None:
                document = (
                    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f"<title>Incomplete report · SkillEvaluator</title><style>{_STYLE}</style></head>"
                    '<body><main style="margin-left:0"><a href="../index.html">← All skills</a>'
                    '<h1>Report unavailable</h1><p class="status">Incomplete</p>'
                    f"<p>{views._h(issue)}</p><p>No passing result is claimed.</p></main></body></html>"
                )
                views._write(directory / "skillevaluator-tier1.html", constrain_native_html(document))
                native_url = f"{directory.name}/skillevaluator-tier1.html"
        quality = views._items(data.get("quality_summary"))
        models.append((row, data, issue, status, native_url, views._dict(quality[0]) if len(quality) == 1 else {}))
    # Retain GitHub's concise native surface and detailed text evidence.
    views.render_reports(reports, summary, context, render_errors=render_errors, models=skills)
    h = views._h
    passed = sum(item[3] == "PASS" for item in models)
    incomplete = sum(item[3] in {"INCOMPLETE", "ERROR", "UNKNOWN"} for item in models)
    failed = len(models) - passed - incomplete
    no_changes = views.is_noop_summary(summary)
    verdict_key = (
        "skipped"
        if no_changes
        else "pass"
        if models and passed == len(models)
        else "incomplete"
        if incomplete or not models
        else "fail"
    )
    verdict = VERDICT_WORDS[verdict_key]
    verdict_class = {"pass": "pass", "fail": "fail", "skipped": "neutral"}.get(verdict_key, "")
    run_url = views._url(context.get("run_url"))
    page = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">",
        f"<title>Skills · SkillEvaluator</title><style>{_STYLE}</style></head><body>",
        f'<aside class="rail">{brand_html("se-logo-index")}<p>Evaluation reports</p><a href="#skills" aria-current="page">All skills</a></aside><main>',
        '<header class="hero"><div><div class="eyebrow">Evaluation report · Tier 1</div><h1>Skill scan results</h1>',
        f'<span class="status {verdict_class}"><span class="sr-only">Verdict: </span>{verdict}</span>',
        # A no-op run explains itself once, in the skills section below.
        "</div>"
        if no_changes
        else '<p class="muted">Open a skill to see its findings and the evidence for each check.</p></div>',
        f'<a class="button" href="{h(run_url)}">View workflow run ↗</a>' if run_url else "",
        '</header><div class="metrics">',
    ]
    for count, label in (
        (len(models), "Skills scanned"),
        (passed, "Passed"),
        (failed, VERDICT_WORDS["fail"]),
        (incomplete, "Incomplete"),
    ):
        page.append(f'<div class="metric"><strong>{count}</strong><span>{label}</span></div>')
    llm = (
        "Disabled"
        if context.get("llm_enabled") is False
        else "Enabled"
        if context.get("llm_enabled") is True
        else "See individual reports"
    )
    scope = (
        "Tier 1: Not run (no skills selected) · LLM: Not run · Tier 2 and Tier 3: Not run"
        if no_changes
        else f"Tier 1 static checks · LLM: {llm} · Tier 2 and Tier 3: Not run"
    )
    page.append(f'</div><div class="scope"><strong>Scope</strong> &nbsp; {scope}</div>')
    attention = [item for item in models if item[3] != "PASS"]
    if attention or summary.get("error"):
        page.append('<section class="attention"><h2>Needs attention</h2>')
        if summary.get("error"):
            page.append(f"<p>{h(summary['error'])}</p>")
        for row, _data, issue, _status, url, _quality in attention:
            fix = first_fixes.get(row["report_dir"]) if not issue else None
            reason = issue or row.get("error") or (fix or {}).get("message") or "Open the report to inspect the checks."
            label = f'<a href="{h(url)}">{h(row["path"])}</a>' if url else h(row["path"])
            line = h(str(reason)[:2000])
            if fix and url:
                line = f'<a href="{h(url)}#{h(fix["anchor"])}">{line}</a>'
            if fix:
                line = f'{line} <span class="muted">{h(fix["check"])}</span>'
            page.append(f"<p><strong>{label}</strong><br>{line}</p>")
        page.append("</section>")
    page.append('<section id="skills"><h2>All skills</h2>')
    if models:
        page.append(
            '<p class="muted">Quality is a content score out of 100. A score below the minimum fails the'
            " Quality score check, which blocks the verdict.</p>"
        )
    if models:
        page.append(
            '<div class="table-wrap"><table><thead><tr><th scope="col">Skill</th><th scope="col">Verdict</th><th scope="col">Quality</th><th scope="col">Errors</th><th scope="col">Warnings</th><th scope="col">Report</th></tr></thead><tbody>'
        )
        for row, data, _issue, status, url, quality in models:
            css = "pass" if status == "PASS" else "fail" if status == "FAIL" else ""
            score = views._score(quality.get("overall_score"))
            grade = quality.get("grade", "")
            minimum = views._score(views._dict(data.get("quality_config")).get("min_score"))
            if minimum != "Not recorded" and score != "Not recorded":
                grade = " · ".join(part for part in (str(grade), f"min {minimum}") if part)
            link = f'<a href="{h(url)}">View report →</a>' if url else "Unavailable"
            page.append(
                f'<tr id="{h(row["report_dir"])}"><td>{h(row["path"])}</td><td><span class="status {css}">{h(verdict_word(status))}</span></td><td class="score">{h(score)} <span class="muted">{h(grade)}</span></td><td>{h(data.get("total_errors", "—"))}</td><td>{h(data.get("total_warnings", "—"))}</td><td class="evidence">{link}</td></tr>'
            )
        page.append("</tbody></table></div>")
    elif no_changes:
        page.append(
            '<p class="empty">No skills were selected for validation in this comparison. Validation did not run.</p>'
        )
        deleted = summary["selection"].get("deleted", [])
        if deleted:
            page.append('<p class="muted">Deleted skill directories (not scanned):</p><ul>')
            page.extend(f"<li><code>{h(path)}</code></li>" for path in deleted)
            page.append("</ul>")
    else:
        page.append(
            '<p class="empty">No skill results were recorded. Review the run error above or the workflow logs.</p>'
        )
    page.append("</section><section><details><summary>Run details and provenance</summary>")
    for label, value in (
        ("Commit", context.get("commit")),
        ("Workflow conclusion", context.get("ci_conclusion")),
        ("Runtime image", context.get("image")),
    ):
        if value:
            page.append(f"<p><strong>{label}</strong><br><code>{h(value)}</code></p>")
    page.append(
        '<p><a href="summary.json" download>Download run summary (JSON)</a></p></details></section><footer>Generated by SkillEvaluator · Static checks do not establish live agent behavior.</footer></main></body></html>'
    )
    views._write(reports / "index.html", "\n".join(page))
    return render_errors
