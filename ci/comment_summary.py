# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- platform-neutral CI comment renderer.
"""Render a bounded, skill-scoped comment from validated retained CI evidence."""

from __future__ import annotations

import html
import unicodedata
from urllib.parse import quote, urlsplit

try:
    from tier_manifest import validate_manifest
except ModuleNotFoundError:
    from ci.tier_manifest import validate_manifest

_CELL_CHARACTERS = 140
_COMMENT_BYTES = 52_000


def _text(value: str, limit: int = _CELL_CHARACTERS) -> str:
    """Keep each cell on one bounded line, without executable Markdown."""
    value = " ".join("".join(" " if unicodedata.category(c).startswith("C") else c for c in value).split())
    if len(value) > limit:
        value = value[: limit - 1].rstrip() + "…"
    return "".join(
        html.escape(c, quote=False)
        if c in "&<>"
        else "&#64;"
        if c == "@"
        else "&#124;"
        if c == "|"
        else "\\" + c
        if c in "\\`*_{}[]()#!"
        else c
        for c in value
    )


def _report_link(artifact_url: str) -> str:
    """A downloadable archive is one artifact, never a member URL directory."""
    if (
        not isinstance(artifact_url, str)
        or not artifact_url
        or len(artifact_url) > 8192
        or any(c.isspace() or unicodedata.category(c).startswith("C") or c == "\\" for c in artifact_url)
    ):
        raise ValueError("Invalid artifact URL")
    parsed = urlsplit(artifact_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid artifact URL")
    safe_url = quote(artifact_url, safe=":/?#[]@!$&'*+,;=%-._~")
    return f"[Report]({safe_url})"


def _reason(record: dict) -> str:
    return next((record[key] for key in ("error", "reason") if isinstance(record.get(key), str) and record[key]), "")


def _cell_text(status: str, issues: str, reason: str = "") -> str:
    prefix = f"{status.upper()} · {issues}"
    if not reason:
        return prefix
    return prefix + " · " + _text(reason, max(1, _CELL_CHARACTERS - len(prefix) - 3))


def _cell(record: dict | None, skill: dict | None, *, expected: bool) -> str:
    if record is None:
        reason = "No retained result" if expected else "Not requested"
        return _cell_text("not_run", "issues unknown", reason)
    if skill is None:
        if record["skills"]:
            return _cell_text("incomplete", "issues unknown", "No retained result for this skill")
        reason = _reason(record) or "No retained skill evidence"
        return _cell_text(record["status"], "issues unknown", reason)

    metrics = skill.get("metrics", {})
    count, critical, high = (metrics.get(key) for key in ("issue_count", "critical_count", "high_count"))
    counts_valid = all(type(value) is int and value >= 0 for value in (count, critical, high))
    known = metrics.get("issue_count_known") is True and counts_valid
    if known or (counts_valid and count > 0):
        prefix = "" if known else "at least "
        issues = f"{prefix}{count} issues"
        severity = ", ".join(f"{value} {name}" for value, name in ((critical, "critical"), (high, "high")) if value > 0)
        if severity:
            issues += f" ({severity})"
    else:
        issues = "issues unknown"
    top_issue = metrics.get("top_issue")
    reason = (top_issue if isinstance(top_issue, str) else "") or _reason(skill)
    if not reason and not known:
        reason = _reason(record) or ("Incomplete native evidence" if counts_valid else "Native evidence unavailable")
    return _cell_text(skill["status"], issues, reason)


def _policy(records: dict[int, dict], expected: list[int]) -> tuple[str, str]:
    required = [tier for tier in (1, 2, 3) if records.get(tier, {}).get("required", tier == 1 and tier in expected)]
    if required == [1]:
        policy = "Only Tier 1 blocks CI; Tier 2 and Tier 3 are advisory."
    else:
        required_label = ", ".join(f"Tier {tier}" for tier in required) or "none"
        advisory_label = ", ".join(f"Tier {tier}" for tier in (1, 2, 3) if tier not in required) or "none"
        policy = f"Required tiers: {required_label}. Advisory tiers: {advisory_label}."
    return policy, "Advisory only" if not required else ""


def render_comment_summary(manifest: dict, artifact_url: str) -> str:
    """Return one compact Markdown table; never read paths or mutate evidence.

    Callers derive each skill's ``metrics`` from verified retained JSON using
    ``summary_metrics.summarize_results``. Missing metrics remain unknown. The
    report link points unchanged to the combined artifact covering all tiers.
    Commit, producer receipt, and run link are supplied by the feedback adapter.
    """
    manifest = validate_manifest(manifest)
    report = _report_link(artifact_url)
    records = {record["tier"]: record for record in manifest["tiers"]}
    by_skill = {tier: {skill["path"]: skill for skill in record["skills"]} for tier, record in records.items()}
    paths = sorted({path for skills in by_skill.values() for path in skills})
    policy, gate = _policy(records, manifest["expected_tiers"])
    if not gate:
        gate = (
            "Skipped"
            if manifest["overall_passed"] and manifest["overall_status"] == "skipped"
            else "Passed"
            if manifest["overall_passed"]
            else "Not passed"
        )
    lines = [
        "## SkillEvaluator CI results",
        "",
        f"Required gate: **{gate}**",
        "",
        policy,
        "",
        "| Skill | Tier 1: Static and Security Validation | Tier 2: Deduplication | Tier 3: Live Agent Evaluation | Report |",
        "| --- | --- | --- | --- | --- |",
    ]
    size = len("\n".join(lines).encode("utf-8"))
    for index, path in enumerate(paths or [None]):
        cells = [
            _cell(records.get(tier), by_skill.get(tier, {}).get(path), expected=tier in manifest["expected_tiers"])
            for tier in (1, 2, 3)
        ]
        label = _text(path) if path is not None else "No skills evaluated"
        row = "| " + " | ".join([label, *cells, report]) + " |"
        row_bytes = len(row.encode("utf-8")) + 1
        # Reserve room for a truthful omission notice and the adapter's footer.
        if size + row_bytes > _COMMENT_BYTES - 200:
            lines.extend(["", f"{len(paths) - index} additional skills omitted; open a Report for complete evidence."])
            break
        lines.append(row)
        size += row_bytes
    return "\n".join([*lines, ""])
