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
from pathlib import Path

from skillevaluator.reporting.retained import render_retained_report

try:
    import report_views as views
except ModuleNotFoundError:
    from ci import report_views as views


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
:root{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#24292f;background:#f5f6f7;font-size:14px}
*{box-sizing:border-box}body{margin:0}a{color:#487600;text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #76b900;outline-offset:4px}
.rail{position:fixed;inset:0 auto 0 0;width:210px;background:#1e2328;color:white;padding:32px 24px}.brand{font-size:17px;font-weight:700;letter-spacing:-.5px}.mark{display:inline-block;background:#76b900;width:11px;height:11px;border-radius:3px;margin-right:8px}.rail p{color:#bfc6ce;font-size:12px;line-height:1.6}.rail a{display:block;color:#e6f2d4;margin-top:30px;padding:12px;background:#34402d;border-radius:6px;text-decoration:none}
main{margin-left:210px;max-width:1400px;padding:36px 44px 60px}.eyebrow{text-transform:uppercase;letter-spacing:.1em;font-size:10px;font-weight:700;color:#59636e}h1{font-size:28px;letter-spacing:-.7px;margin:8px 0 12px}h2{font-size:19px;letter-spacing:-.3px;margin:0 0 16px}p{line-height:1.6}.muted{color:#59636e;font-size:12px}.status{display:inline-block;border-radius:5px;padding:5px 9px;font-size:11px;font-weight:700;background:#fff5df;color:#8a5800}.pass{background:#edf5e4;color:#357000}.fail{background:#fff0ef;color:#b4232d}.hero{display:flex;justify-content:space-between;align-items:start;gap:20px}.button{display:inline-block;padding:10px 14px;border:1px solid #dfe3e6;border-radius:6px;background:#fff;font-size:12px;font-weight:600;text-decoration:none;white-space:nowrap}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric,section{background:#fff;border:1px solid #dfe3e6;border-radius:8px}.metric{padding:17px 20px}.metric strong{display:block;font-size:26px;letter-spacing:-.5px;margin-bottom:5px}.metric span{font-size:12px;color:#59636e}section{padding:24px;margin-bottom:20px}.scope{font-size:12px;color:#59636e;line-height:1.7;padding:0 0 20px}.scope strong{color:#24292f}.attention{border-left:3px solid #b4232d}.attention p:last-child{margin-bottom:0}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left}th{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#59636e;background:#fafbfc}th,td{padding:14px 12px;border-bottom:1px solid #eaecef}td{font-size:12px}td:first-child{font-weight:600;min-width:180px;max-width:320px;overflow-wrap:anywhere}td .muted{font-weight:400}tr:last-child td{border-bottom:0}.score{font-variant-numeric:tabular-nums;font-weight:600}.evidence{white-space:nowrap}summary{cursor:pointer;font-size:12px;font-weight:600}details p{font-size:12px;overflow-wrap:anywhere}code{font-family:ui-monospace,monospace;font-size:11px;overflow-wrap:anywhere}.empty{padding:20px;background:#fff5df;border-radius:6px}footer{font-size:11px;color:#59636e;margin-top:24px}
@media(max-width:850px){.rail{position:static;width:auto;padding:16px 20px}.rail p,.rail a{display:none}main{margin:0;padding:24px 20px}.hero{flex-wrap:wrap}.metrics{grid-template-columns:repeat(2,1fr)}section{padding:18px}.metric{padding:16px}h1{font-size:25px}}
@media(max-width:600px){table,tbody{display:block}thead{display:none}tr{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid #dfe3e6;padding:12px 0;gap:10px}td{display:block;padding:0;border:0}td:first-child{grid-column:1/-1;min-width:0;max-width:none}td:nth-child(3)::before{content:'Quality: ';font-weight:400;color:#59636e}td:nth-child(4)::before{content:'Errors: ';color:#59636e}td:nth-child(5)::before{content:'Warnings: ';color:#59636e}.evidence{grid-column:1/-1}tr:last-child{border-bottom:0}}
@media print{.rail,.button{display:none}main{margin:0;padding:0;max-width:none}.table-wrap{overflow:visible}section,.metric{break-inside:avoid}body{background:white}a{color:inherit}}
"""


def render_native_reports(reports: Path, summary: dict, context: dict) -> dict[str, str]:
    """Write all available views and return per-skill rendering diagnostics."""
    reports = Path(reports)
    if reports.is_symlink() or not reports.is_dir():
        raise ValueError("A regular report directory is required.")
    models = []
    render_errors = {}
    for row in summary.get("skills", []):
        if row.get("overall_status") == "error" and re.fullmatch(r"skill-[0-9]{3}", row.get("report_dir", "")):
            # The publisher intentionally discards rejected child JSON, so its
            # directory may not exist yet. Create only this validated leaf.
            (reports / row["report_dir"]).mkdir(exist_ok=True)
        directory = views._directory(reports, row.get("report_dir"))
        data, issue = views._read(directory)
        status = views._skill_status(row, data, issue)
        native_url = None
        if row.get("overall_status") == "error":
            issue = row.get("error") or "The evaluator did not produce consistent report evidence."
        elif data and directory is not None:
            try:
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
                )
                views._write(directory / "skillevaluator-tier1.html", constrain_native_html(document))
                native_url = f"{directory.name}/skillevaluator-tier1.html"
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
    views.render_reports(reports, summary, context, render_errors=render_errors)
    h = views._h
    passed = sum(item[3] == "PASS" for item in models)
    incomplete = sum(item[3] in {"INCOMPLETE", "ERROR", "UNKNOWN"} for item in models)
    failed = len(models) - passed - incomplete
    no_changes = views.is_noop_summary(summary)
    verdict = (
        "No skills changed"
        if no_changes
        else "Passed"
        if models and passed == len(models)
        else "Incomplete"
        if incomplete or not models
        else "Failed"
    )
    verdict_class = "pass" if verdict == "Passed" else "fail" if verdict == "Failed" else ""
    run_url = views._url(context.get("run_url"))
    page = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">",
        f"<title>Skills · SkillEvaluator</title><style>{_STYLE}</style></head><body>",
        '<aside class="rail"><div class="brand"><span class="mark"></span>SkillEvaluator</div><p>Evaluation reports</p><a href="#skills" aria-current="page">All skills</a></aside><main>',
        '<header class="hero"><div><div class="eyebrow">Evaluation report · Tier 1</div><h1>Skill scan results</h1>',
        f'<span class="status {verdict_class}">{verdict}</span><p class="muted">Review findings, quality dimensions, and scanner evidence for each skill.</p></div>',
        f'<a class="button" href="{h(run_url)}">View workflow run ↗</a>' if run_url else "",
        '</header><div class="metrics">',
    ]
    for count, label in (
        (len(models), "Skills scanned"),
        (passed, "Passed"),
        (failed, "Failed"),
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
        for row, data, issue, _status, url, _quality in attention:
            reasons = [
                finding.get("message")
                for result in data.get("results", [])
                if not result.get("passed")
                for finding in result.get("findings", [])
                if finding.get("message")
            ]
            reason = issue or row.get("error") or next(iter(reasons), "Open the report to inspect the required checks.")
            label = f'<a href="{h(url)}">{h(row["path"])}</a>' if url else h(row["path"])
            page.append(f"<p><strong>{label}</strong><br>{h(str(reason)[:2000])}</p>")
        page.append("</section>")
    page.append(
        '<section id="skills"><h2>All skills</h2><p class="muted">Quality measures content quality out of 100. The scan verdict is a separate result.</p>'
    )
    if models:
        page.append(
            '<div class="table-wrap"><table><thead><tr><th scope="col">Skill</th><th scope="col">Scan result</th><th scope="col">Quality</th><th scope="col">Errors</th><th scope="col">Warnings</th><th scope="col">Report</th></tr></thead><tbody>'
        )
        for row, data, _issue, status, url, quality in models:
            css = "pass" if status == "PASS" else "fail" if status == "FAIL" else ""
            score = views._score(quality.get("overall_score"))
            grade = quality.get("grade", "")
            link = f'<a href="{h(url)}">View report →</a>' if url else "Unavailable"
            page.append(
                f'<tr id="{h(row["report_dir"])}"><td>{h(row["path"])}</td><td><span class="status {css}">{h(status)}</span></td><td class="score">{h(score)} <span class="muted">{h(grade)}</span></td><td>{h(data.get("total_errors", "—"))}</td><td>{h(data.get("total_warnings", "—"))}</td><td class="evidence">{link}</td></tr>'
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
