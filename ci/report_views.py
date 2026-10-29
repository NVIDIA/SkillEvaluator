# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- standalone CI helper loaded by the trusted publisher.
"""Render retained Tier 1 evidence into safe, portable CI report views."""

from __future__ import annotations

import html
import json
import math
import re
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

from skillevaluator.reporting.presentation import success_is_not_applicable

SUMMARY_LIMIT = 20 * 1024
DETAIL_LIMIT = 500 * 1024
SUMMARY_SKILL_LIMIT = 5
COMMENT_LIMIT = 50 * 1024
REPORT_LIMIT = 32 * 1024 * 1024


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _items(value: object) -> list:
    return value if isinstance(value, list) else []


def _text(value: object) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _h(value: object) -> str:
    return html.escape(_text(value), quote=True)


def _md(value: object) -> str:
    text = "".join(c if ord(c) >= 32 else " " for c in _text(value))
    # Break GitHub mention tokens while retaining readable author and finding text.
    text = text.replace("@", "@\u200b")
    escaped = re.sub(r"([\\`*_{}\[\]()#+.!|~-])", r"\\\1", text)
    return html.escape(escaped, quote=False)


def _url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 8192 or any(ord(c) <= 32 for c in value):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password:
            return value
    except ValueError:
        pass
    return None


def _status(result: dict) -> str:
    raw = result.get("status")
    raw = raw if isinstance(raw, str) else None
    if raw in {"incomplete", "error"} or result.get("incomplete_scans"):
        return "INCOMPLETE"
    if raw == "failed":
        return "FAIL"
    if result.get("skipped") is True or raw in {"skipped", "not_run"}:
        reason = _text(result.get("skip_reason", "")).casefold()
        return "NOT APPLICABLE" if "not applicable" in reason else "NOT RUN"
    if raw in {"not_applicable", "not-applicable"}:
        return "NOT APPLICABLE"
    if result.get("passed") is False:
        return "FAIL"
    if raw == "passed" and result.get("passed") is True:
        details = [_dict(item) for item in _items(result.get("success_details"))]
        if (
            details
            and not result.get("findings")
            and all(
                success_is_not_applicable(item.get("check"), item.get("message"), item.get("metadata"))
                for item in details
            )
        ):
            return "NOT APPLICABLE"
        return "PASS"
    return "UNKNOWN"


def _skill_status(row: dict, data: dict, issue: str | None) -> str:
    """Do not paint a skill green when aggregate and child verdicts disagree."""
    if issue or not _items(data.get("results")):
        return "INCOMPLETE"
    raw = data.get("overall_status")
    if not isinstance(raw, str) or raw not in {"passed", "failed", "incomplete"}:
        return "INCOMPLETE"
    expected = raw == "passed"
    if (
        type(data.get("overall_passed")) is not bool
        or data["overall_passed"] is not expected
        or type(row.get("passed")) is not bool
        or row["passed"] is not expected
        or row.get("overall_status") != raw
    ):
        return "INCOMPLETE"
    return {"passed": "PASS", "failed": "FAIL", "incomplete": "INCOMPLETE"}[raw]


def _score(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:g}"
    return "Not recorded"


def _directory(reports: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return None
    candidate = reports / value
    if candidate.is_symlink() or not candidate.is_dir():
        return None
    return candidate


def _read(directory: Path | None) -> tuple[dict, str | None]:
    if directory is None:
        return {}, "Report directory is missing or unsafe; no evidence loaded."
    path = directory / "skillevaluator-tier1.json"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > REPORT_LIMIT:
            return {}, "JSON report is missing, unsafe, or too large; no evidence loaded."
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return {}, "JSON report has no validator result list; evidence is incomplete."
        return data, None
    except (OSError, ValueError):
        return {}, "JSON report could not be read; evidence is incomplete."


def _links(
    directory: Path | None, report_url: str | None = None, *, anchor: str | None = None
) -> list[tuple[str, str]]:
    if directory is None:
        return []
    choices = [
        ("HTML", "skillevaluator-tier1.html"),
        ("JSON", "skillevaluator-tier1.json"),
        ("Markdown", "skillevaluator-tier1.md"),
        ("SARIF", "skillevaluator-tier1.sarif.json"),
        ("stdout log", "stdout.log"),
        ("stderr log", "stderr.log"),
    ]
    # Hosted HTML is regenerated from validated JSON using trusted templates.
    # Original Markdown, SARIF and logs remain download-artifact links.
    if report_url:
        choices = [("Native HTML", "skillevaluator-tier1.html"), ("JSON", "skillevaluator-tier1.json")]
    links = [("Skill details", f"{report_url.rstrip('/')}/index.html#{anchor}")] if report_url and anchor else []
    for label, name in choices:
        path = directory / name
        if path.is_file() and not path.is_symlink():
            relative = f"{directory.name}/{name}"
            links.append((label, f"{report_url.rstrip('/')}/{relative}" if report_url else relative))
    return links


def _md_links(links: list[tuple[str, str]]) -> str:
    return " · ".join(f"[{label}]({quote(url, safe='/:#?=&%')})" for label, url in links)


def _evidence(result: dict) -> list[tuple[str, str]]:
    rows = []
    for item in _items(result.get("success_details")):
        item = _dict(item)
        rows.append((f"Evidence: {_text(item.get('check'))}", _text(item.get("message"))))
        if item.get("metadata"):
            rows.append(("Evidence metadata", _text(item["metadata"])))
    for finding in _items(result.get("findings")):
        finding = _dict(finding)
        rows.append(
            (
                f"Finding: {_text(finding.get('check_name'))} ({_text(finding.get('severity'))})",
                _text(finding.get("message")),
            )
        )
        for key, label in [
            ("file_path", "File"),
            ("line_number", "Line"),
            ("line_content", "Source line"),
            ("suggestion", "Suggested fix"),
            ("metadata", "Finding metadata"),
        ]:
            if finding.get(key) is not None:
                rows.append((label, _text(finding[key])))
    legacy = _dict(result.get("legacy"))
    for key, label in [("errors", "Error"), ("warnings", "Warning"), ("messages", "Scanner message")]:
        rows.extend((label, _text(message)) for message in _items(legacy.get(key)))
    if result.get("skip_reason"):
        rows.append(("Skip reason", _text(result["skip_reason"])))
    if result.get("incomplete_scans"):
        rows.append(("Incomplete scanners", _text(result["incomplete_scans"])))
    return rows


def _llm_scope(models: list[dict], context: dict) -> str:
    stages = [_dict(_dict(model["data"].get("workflow")).get("stages")).get("llm") for model in models]
    recorded = [_dict(stage) for stage in stages if isinstance(stage, dict)]
    if recorded:
        values = sorted({_text(stage.get("status")) for stage in recorded})
        reasons = sorted({_text(stage["reason"]) for stage in recorded if stage.get("reason")})
        return "Recorded LLM stages: " + ", ".join(values) + (". " + " ".join(reasons) if reasons else "")
    if context.get("llm_enabled") is False:
        return "LLM checks disabled; no model-backed evaluation requested."
    if context.get("llm_enabled") is True:
        return "LLM checks requested; execution evidence is not recorded in the available reports."
    return "LLM execution scope not recorded; no model-backed evaluation is claimed."


def _markdown_blocks(lines: list[str]) -> list[str]:
    """Keep each generated table contiguous and atomic under display limits."""
    blocks = []
    table = []
    for line in lines:
        if line.startswith("|"):
            table.append(line)
            continue
        if table:
            blocks.append("\n".join(table))
            table = []
        blocks.append(line)
    if table:
        blocks.append("\n".join(table))
    return blocks


def _bounded(blocks: list[str], limit: int, footer: str) -> str:
    reserve = len(footer.encode("utf-8")) + 160
    parts = []
    size = 0
    truncated = False
    for block in _markdown_blocks(blocks):
        cost = len(block.encode("utf-8")) + 2
        if size + cost > limit - reserve:
            truncated = True
            break
        parts.append(block)
        size += cost
    if truncated:
        parts.append(
            "**Truncated for CI display limits. Full evidence remains in the HTML report and raw JSON/log artifacts.**"
        )
    parts.append(footer)
    return "\n\n".join(parts) + "\n"


def _write(path: Path, content: str) -> None:
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlink report view.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _brief(value: object, limit: int = 320) -> str:
    text = _text(value)
    return _md(text if len(text) <= limit else text[:limit] + "… (see full report)")


def is_noop_summary(summary: dict) -> bool:
    """Recognize only a complete, neutral Git changed-selection receipt."""
    if (
        not isinstance(summary, dict)
        or summary.get("overall_status") != "skipped"
        or summary.get("overall_passed") is not False
        or summary.get("skills") != []
        or summary.get("error")
    ):
        return False
    for key in ("total", "passed", "failed", "incomplete", "total_validators", "total_errors", "total_warnings"):
        if type(summary.get(key)) is not int or summary[key] != 0:
            return False
    counts = summary.get("severity_counts")
    if not isinstance(counts, dict) or any(
        type(counts.get(key)) is not int or counts[key] != 0 for key in ("critical", "high", "medium", "low")
    ):
        return False
    selection = summary.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("mode") != "changed"
        or selection.get("selected") != []
        or type(selection.get("discovered")) is not int
        or not 0 <= selection["discovered"] <= 10000
        or type(selection.get("changed_files")) is not int
        or not 0 <= selection["changed_files"] <= 100000
    ):
        return False
    if any(
        not isinstance(selection.get(key), str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", selection[key])
        for key in ("base", "head")
    ):
        return False
    if "base_ref" in selection and (
        not isinstance(selection["base_ref"], str)
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", selection["base_ref"])
    ):
        return False
    deleted = selection.get("deleted")
    if not isinstance(deleted, list) or len(deleted) > 10000:
        return False
    for value in deleted:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 4096
            or value.startswith("/")
            or ".." in value.split("/")
            or "\\" in value
            or ":" in value
            or any(ord(character) < 32 for character in value)
        ):
            return False
    return (
        bool(selection["discovered"] or deleted)
        and len(set(deleted)) == len(deleted)
        and selection["changed_files"] >= len(deleted)
    )


def _render_noop(reports: Path, summary: dict, context: dict) -> None:
    selection = summary["selection"]
    links = []
    hosted, run = _url(context.get("report_url")), _url(context.get("run_url"))
    if hosted:
        links.append(("Open full report", hosted.rstrip("/") + "/index.html"))
    if run:
        links.append(("Workflow run and logs", run))
    lines = [
        "# Skill scan",
        "**Tier 1 scan result: NO SKILLS CHANGED**",
        "No configured skill changed between the selected Git commits. Validation did not run.",
        f"Selection: 0 of {selection['discovered']} current skills selected.",
        "Tier 1: Not run (no skills selected). LLM: Not run. Tier 2: Not run. Tier 3: Not run.",
    ]
    if selection["deleted"]:
        lines.append(f"Deleted skill directories (not scanned): {len(selection['deleted'])}.")
        lines.extend(f"- {_brief(value, 240)}" for value in selection["deleted"][:5])
        if len(selection["deleted"]) > 5:
            lines.append(f"{len(selection['deleted']) - 5} more deleted directories are recorded in summary.json.")
        lines.append("These skill directories were removed completely; they were skipped, not evaluated.")
    lines.append("Full selection evidence: download the scan run's report artifact and open summary.json.")
    footer = _md_links(links)
    markdown = _bounded(lines, SUMMARY_LIMIT, footer)
    _write(reports / "summary.md", markdown)
    _write(reports / "details.md", markdown)
    _write(reports / "pr-comment.md", markdown)


def finding_repository_path(skill_path: object, file_path: object) -> str | None:
    """Map recorded scanner paths to safe repository-relative annotation paths."""
    if not isinstance(skill_path, str) or not isinstance(file_path, str) or not file_path:
        return None
    if any(ord(c) < 32 or ord(c) == 127 for c in skill_path + file_path) or "\\" in skill_path + file_path:
        return None
    if file_path.startswith("/workspace/"):
        path = PurePosixPath(file_path.removeprefix("/workspace/"))
    else:
        path = PurePosixPath(file_path)
        root = PurePosixPath(skill_path)
        if path.is_absolute() or root.is_absolute() or ".." in root.parts:
            return None
        if root != PurePosixPath(".") and not path.is_relative_to(root):
            path = root / path
    if path.is_absolute() or ".." in path.parts or ":" in str(path) or str(path) in {"", "."}:
        return None
    return str(path)


def _first_action(model: dict) -> tuple[str, str, str]:
    """Select one recorded problem, preferring unavailable scans over findings."""
    row = model["row"]
    if model["issue"] or row.get("error"):
        return (
            _text(row.get("error") or model["issue"]),
            "",
            "Inspect the failed workflow step and report artifacts, then rerun the scan.",
        )
    failed = [r for r in model["results"] if _status(r) in {"FAIL", "INCOMPLETE", "UNKNOWN"}]
    failed.sort(key=lambda r: _status(r) == "FAIL")
    for result in failed:
        legacy = _dict(result.get("legacy"))
        if _status(result) in {"INCOMPLETE", "UNKNOWN"}:
            errors = _items(legacy.get("errors"))
            warnings = _items(legacy.get("warnings"))
            reason = errors[0] if errors else result.get("skip_reason")
            reason = reason or (warnings[0] if warnings else None)
            reason = reason or (
                "Scans did not complete: " + _text(result["incomplete_scans"])
                if result.get("incomplete_scans")
                else "Execution evidence is incomplete."
            )
            return (
                f"{_text(result.get('validator'))}: {_text(reason)}",
                "",
                "Check this validator's diagnostics and rerun after restoring the missing scan.",
            )
        findings = [_dict(f) for f in _items(result.get("findings"))]
        severity = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        findings.sort(key=lambda f: severity.get(_text(f.get("severity")).lower(), 5))
        finding = findings[0] if findings else {}
        errors = _items(legacy.get("errors"))
        reason = finding.get("message") or (errors[0] if errors else "This validator failed without a recorded reason.")
        fix = (
            finding.get("suggestion")
            or "Open the full report and inspect this validator's diagnostics before rerunning."
        )
        location = finding_repository_path(row.get("path"), finding.get("file_path")) or ""
        line = finding.get("line_number")
        if location and type(line) is int and line > 0:
            location += f":{line}"
        # Only offer a concrete replacement for this exact, known schema failure.
        # Other naming failures and arbitrary findings retain their recorded fix.
        name_error = re.fullmatch(
            r"Field 'name': Value error, Skill name '([^'\n]+)' must be in kebab-case format "
            r"\(lowercase letters, numbers, and hyphens only, starting with a letter\)",
            _text(reason),
        )
        folder = _text(row.get("path")).rstrip("/").rsplit("/", 1)[-1]
        if (
            finding.get("category") == "SCHEMA"
            and finding.get("check_name") == "frontmatter_field"
            and _dict(finding.get("metadata")).get("field") == "name"
            and name_error
            and re.fullmatch(r"[a-z][a-z0-9-]{0,63}", folder)
        ):
            reason = f"Invalid skill name: '{name_error[1]}'."
            fix = f"Set name: {folder} in SKILL.md frontmatter to match the skill directory."
        return _text(reason), location, _text(fix)
    return (
        "The skill did not produce a complete passing result.",
        "",
        "Open the full report and inspect the failed workflow step before rerunning.",
    )


def _compact_summary(models: list[dict], summary: dict, context: dict, *, status: str, llm: str, footer: str) -> str:
    """Keep the Actions landing view small; detailed evidence lives in reports."""
    outcomes = [
        "INCOMPLETE"
        if model["skill_status"] == "PASS"
        and any(_status(r) in {"FAIL", "INCOMPLETE", "UNKNOWN"} for r in model["results"])
        else model["skill_status"]
        for model in models
    ]
    affected = [model for model, outcome in zip(models, outcomes, strict=True) if outcome != "PASS"]
    counts = {key: outcomes.count(key) for key in ("PASS", "FAIL", "INCOMPLETE")}
    mismatch = any(
        summary.get(key) != count
        for key, count in (
            ("total", len(models)),
            ("passed", counts["PASS"]),
            # The runtime rollup counts every non-passing skill as failed,
            # including its incomplete subset. Display the subsets separately.
            ("failed", counts["FAIL"] + counts["INCOMPLETE"]),
            ("incomplete", counts["INCOMPLETE"]),
        )
    )
    if (not models or mismatch) and status in {"PASSED", "PASS"}:
        status = "INCOMPLETE"
    if summary.get("error"):
        status = "ERROR"
    if status.startswith("INCOMPLETE"):
        status = "INCOMPLETE"
    lines = ["# Skill scan", f"**Tier 1 scan result: {_brief(status)}**"]
    lines.append(
        f"{len(models)} skills · {counts['PASS']} passed · {counts['FAIL']} failed · {counts['INCOMPLETE']} incomplete"
    )
    if summary.get("error"):
        lines.extend(
            [f"**Run error:** {_brief(summary['error'])}", "Correct the workflow configuration and rerun the scan."]
        )
    if mismatch:
        lines.append(
            "**Report evidence is missing or inconsistent.** Inspect the full report before relying on these results."
        )
    if affected:
        lines.append("## What to fix")
        lines.append("First issue per affected skill; the full report contains all findings.")
        for model in affected[:SUMMARY_SKILL_LIMIT]:
            reason, location, fix = _first_action(model)
            lines.append(f"**{_brief(model['row'].get('path'), 200)}**")
            lines.append(f"Problem: {_brief(reason)}")
            if location:
                lines.append(f"File: {_brief(location, 240)}")
            lines.append(f"Suggested fix: {_brief(fix)}")
        if len(affected) > SUMMARY_SKILL_LIMIT:
            lines.append(f"**{len(affected) - SUMMARY_SKILL_LIMIT} more affected skills:** open the full report.")
        lines.append("Push your fixes to rerun PR checks, or rerun the workflow after resolving scan errors.")
    elif models and status in {"PASSED", "PASS"}:
        lines.append("No blocking issues found. Advisory findings remain available in the full report.")
    elif not summary.get("error"):
        lines.append("Inspect the full report and failed workflow step; no complete passing scan is established.")
    deleted = _items(_dict(summary.get("selection")).get("deleted"))
    if deleted:
        lines.append(f"{len(deleted)} completely deleted skill directories were skipped; see summary.json.")
    conclusion = context.get("ci_conclusion", context.get("run_conclusion"))
    if conclusion is not None:
        lines.append(f"**CI run conclusion:** {_brief(conclusion, 80)}")
    lines.append(f"Tier 1 only · Tier 2: Not run · Tier 3: Not run. {_brief(llm, 240)}")
    # The source scan's artifact is also the fallback when reading a separate
    # publisher workflow's summary, or when the hosted URL is not deployed yet.
    artifact_help = (
        "Full evidence: download the scan run's report artifact and open index.html. Detailed Markdown: details.md."
    )
    # Percent encoding can expand a valid URL beyond the entire display budget.
    # Preserve the result and counts rather than spending that budget on links.
    if len(footer.encode("utf-8")) > SUMMARY_LIMIT // 2 or footer.startswith("Full evidence:"):
        footer = artifact_help
    else:
        lines.append(artifact_help)
    return _bounded(lines, SUMMARY_LIMIT, footer)


def render_reports(reports: Path, summary: dict, context: dict, *, render_errors: dict[str, str] | None = None) -> None:
    """Write a short Actions summary, detailed Markdown and a PR comment.

    Optional context keys: run_url, commit, image, report_url, llm_enabled, ci_conclusion.
    report_url is the hosted directory containing index.html and the artifacts.
    All report payloads are displayed as text, never executable markup.
    """
    reports = Path(reports)
    if reports.is_symlink() or not reports.is_dir():
        raise ValueError("A regular report directory is required.")
    if is_noop_summary(summary):
        _render_noop(reports, summary, context)
        return
    report_url = _url(context.get("report_url"))
    run_url = _url(context.get("run_url"))
    models = []
    for index, row in enumerate(_items(summary.get("skills")), 1):
        row = _dict(row)
        directory = _directory(reports, row.get("report_dir"))
        data, issue = _read(directory)
        if row.get("overall_status") == "error" or (render_errors or {}).get(row.get("report_dir")):
            issue = (render_errors or {}).get(row.get("report_dir")) or row.get("error") or "Report unavailable"
            data = {}
        results = [_dict(item) for item in _items(data.get("results"))]
        quality = [_dict(item) for item in _items(data.get("quality_summary"))]
        models.append(
            {
                "row": row,
                "directory": directory,
                "data": data,
                "issue": issue,
                "results": results,
                "quality": quality,
                "anchor": f"skill-{index:03d}",
            }
        )
    for model in models:
        model["skill_status"] = _skill_status(model["row"], model["data"], model["issue"])
        if (
            model["skill_status"] == "INCOMPLETE"
            and not model["issue"]
            and model["data"].get("overall_status") != "incomplete"
        ):
            model["issue"] = (
                "Summary and child report verdicts are missing or inconsistent; no passing verdict is claimed."
            )
    recorded = sum(bool(model["results"]) for model in models)
    status = _text(summary.get("overall_status", "unknown")).upper()
    if status == "SKIPPED":
        status = "INCOMPLETE"
    evidence_missing = any(model["issue"] or not model["results"] for model in models)
    aggregate_mismatch = summary.get("total") != len(models) or summary.get("passed") != sum(
        model["skill_status"] == "PASS" for model in models
    )
    evidence_missing = evidence_missing or aggregate_mismatch
    if evidence_missing and status in {"PASSED", "PASS"}:
        status = "INCOMPLETE (aggregate gate was recorded as passed; report evidence is missing)"
    llm = _llm_scope(models, context)
    scope = f"Tier 1: executed for {recorded}/{len(models)} skills with validator evidence; result {status}. Tier 2: Not run. Tier 3: Not run."
    displayed = {key: sum(model["skill_status"] == key for model in models) for key in ("PASS", "FAIL", "INCOMPLETE")}
    overview = f"{len(models)} skills · {displayed['PASS']} passed · {displayed['FAIL']} failed · {displayed['INCOMPLETE']} incomplete"
    counts = f"Errors: {_text(summary.get('total_errors'))} · Warnings: {_text(summary.get('total_warnings'))}"
    hosted = f"{report_url.rstrip('/')}/index.html" if report_url else None
    navigation = ([("Open full report", hosted)] if hosted else []) + (
        [("Workflow run and logs", run_url)] if run_url else []
    )
    footer = _md_links(navigation) or "Full evidence: open index.html in the downloaded report artifact."
    markdown = [
        "# SkillEvaluator evaluation report",
        f"**Tier 1 scan result: {_md(status)}**",
        _md(overview),
        _md(counts),
        "## Evaluation scope",
        _md(scope),
        _md(llm),
        "Static scanner outcomes do not establish behavioral usefulness. Not run tiers are not passes.",
    ]
    if navigation:
        markdown.append(_md_links(navigation))
    metadata = [
        ("CI run conclusion", context.get("ci_conclusion", context.get("run_conclusion"))),
        ("Commit", context.get("commit")),
        ("Runtime image", context.get("image")),
    ]
    for label, value in metadata:
        if value is not None:
            markdown.append(f"**{label}:** {_md(_text(value)[:4096])}")
    if summary.get("error"):
        markdown.append(f"**Run error:** {_md(summary['error'])}")
    comment_header = list(markdown)
    attention = []
    for model in models:
        for result in model["results"]:
            outcome = _status(result)
            if outcome not in {"FAIL", "INCOMPLETE", "UNKNOWN"}:
                continue
            findings = _items(result.get("findings"))
            finding = _dict(findings[0]) if findings else {}
            errors = _items(_dict(result.get("legacy")).get("errors"))
            reason = finding.get("message") or (errors[0] if errors else result.get("skip_reason"))
            reason = reason or "Execution evidence is incomplete; inspect the validator details."
            location = _text(finding.get("file_path"))
            if finding.get("line_number") is not None:
                location += ":" + _text(finding["line_number"])
            attention.append((model, result.get("validator"), outcome, reason, location, finding.get("suggestion")))
        if model["issue"]:
            attention.append((model, "Report evidence", "INCOMPLETE", model["issue"], None, None))
    if attention:
        markdown.append("## Needs attention")
        for model, validator, outcome, reason, location, suggestion in attention:
            label = _text(model["row"].get("path")) + " / " + _text(validator)
            markdown.append(f"**{_md(label)} ({outcome}):** {_md(_text(reason)[:2000])}")
            if location and location != "Not recorded":
                markdown.append(f"Location: {_md(location)}")
            if suggestion:
                markdown.append(f"Suggested fix: {_md(_text(suggestion)[:2000])}")
    markdown.extend(
        [
            "## Skills and scores",
            "| Skill | Tier 1 | Quality / 100 | Grade | Errors | Warnings |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    comments = [
        *comment_header,
        "| Skill | Status | Quality / 100 | Errors | Warnings |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in models:
        row, data = model["row"], model["data"]
        skill = _text(row.get("path"))
        skill_status = model["skill_status"]
        quality = model["quality"][0] if len(model["quality"]) == 1 else {}
        score, grade = _score(quality.get("overall_score")), _text(quality.get("grade"))
        errors, warnings = data.get("total_errors"), data.get("total_warnings")
        markdown.append(f"| {_md(skill)} | {skill_status} | {score} | {_md(grade)} | {_md(errors)} | {_md(warnings)} |")
        comments.append(f"| {_md(skill)} | {skill_status} | {score} / {_md(grade)} | {_md(errors)} | {_md(warnings)} |")
    for model in models:
        row = model["row"]
        skill = _text(row.get("path"))
        block = [f"## {_md(skill)}", _md_links(_links(model["directory"], report_url, anchor=model["anchor"]))]
        if row.get("error"):
            block.append(f"Run error: {_md(row['error'])}")
        if model["issue"]:
            block.append(_md(model["issue"]))
            comments.append(f"**{_md(skill)}:** {_md(model['issue'])}")
        for quality in model["quality"]:
            block.extend(
                [
                    "### Recorded quality dimensions",
                    "| Dimension | Score / 100 | Weight | Issues |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for name, dimension in _dict(quality.get("dimensions")).items():
                dimension = _dict(dimension)
                values = [name, _score(dimension.get("score")), dimension.get("weight"), dimension.get("issues_count")]
                block.append("| " + " | ".join(_md(value) for value in values) + " |")
        block.extend(
            [
                "### Validator outcomes",
                "| Validator | Outcome | Checks recorded | Errors | Warnings |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for result in model["results"]:
            numbers = _dict(result.get("summary"))
            name, outcome = result.get("validator"), _status(result)
            block.append(
                f"| {_md(name)} | {outcome} | {_md(numbers.get('checks_performed'))} | {_md(numbers.get('errors'))} | {_md(numbers.get('warnings'))} |"
            )
        for result in model["results"]:
            block.append(f"### {_md(result.get('validator'))} — evidence")
            for label, value in _evidence(result):
                block.append(f"**{_md(label)}:** {_md(value)}")
        markdown.extend(block)
        failures = [result for result in model["results"] if _status(result) in {"FAIL", "INCOMPLETE", "UNKNOWN"}]
        for result in failures:
            findings = _items(result.get("findings"))
            errors = _items(_dict(result.get("legacy")).get("errors"))
            message = (
                _dict(findings[0]).get("message") if findings else (errors[0] if errors else "See retained evidence.")
            )
            comments.append(f"**{_md(skill)} / {_md(result.get('validator'))}:** {_md(_text(message)[:2000])}")
            if findings:
                finding = _dict(findings[0])
                location = _text(finding.get("file_path"))
                if finding.get("line_number") is not None:
                    location += ":" + _text(finding["line_number"])
                comments.append(
                    f"Location: {_md(location[:2000])}. Suggested fix: {_md(_text(finding.get('suggestion'))[:2000])}"
                )
    _write(reports / "details.md", _bounded(markdown, DETAIL_LIMIT, footer))
    _write(reports / "summary.md", _compact_summary(models, summary, context, status=status, llm=llm, footer=footer))
    _write(reports / "pr-comment.md", _bounded(comments, COMMENT_LIMIT, footer))
