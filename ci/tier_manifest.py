# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- platform-neutral CI report entrypoint.
"""Validate and render retained multi-tier CI outcomes without reading skill paths."""

from __future__ import annotations

import copy
import html
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

_STATUSES = frozenset({"passed", "failed", "incomplete", "skipped", "not_run", "needs_action", "neutral"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(value, depth: int = 0, budget: list[int] | None = None) -> None:
    budget = [100_000] if budget is None else budget
    budget[0] -= 1
    _require(budget[0] >= 0 and depth <= 24, "Manifest JSON exceeds its structural budget")
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), "Manifest contains a non-finite number")
    elif isinstance(value, str):
        _require(len(value) <= 1_000_000, "Manifest string is too large")
    elif isinstance(value, list):
        for item in value:
            _json(item, depth + 1, budget)
    elif isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str) and len(key) <= 4096, "Invalid manifest JSON key")
            _json(item, depth + 1, budget)
    else:
        raise ValueError("Manifest must contain JSON values only")


def _text(value, field: str, *, limit: int = 4096, empty: bool = False) -> str:
    _require(
        isinstance(value, str)
        and len(value) <= limit
        and (empty or bool(value))
        and not any(ord(c) < 32 or ord(c) == 127 for c in value),
        f"Invalid {field}",
    )
    return value


def _path(value, field: str, *, skill: bool = False) -> str:
    value = _text(value, field)
    if skill and value == ".":
        return value
    _require(
        not any(c in value for c in ("\\", ":", "%", "?", "#"))
        and not value.startswith("/")
        and not value.endswith("/")
        and all(part not in {"", ".", ".."} for part in value.split("/"))
        and PurePosixPath(value).as_posix() == value,
        f"Unsafe {field}",
    )
    return value


def validate_manifest(payload: dict) -> dict:
    """Return a copied canonical manifest; reject contradictory gate claims.

    ``expected_tiers`` defaults to all three tiers. A single-tier producer can
    explicitly declare its one expected tier. Missing expected required evidence
    fails the gate; an explicit no-work skip satisfies it with a skipped headline.
    Additional JSON provenance and historical declared gate policies are retained.
    New CI producers require only Tier 1; missing paid-tier evidence is advisory.
    """
    _require(isinstance(payload, dict), "Manifest must be an object")
    _json(payload)
    _require(type(payload.get("schema_version")) is int and payload["schema_version"] == 1, "Invalid schema_version")
    _require(payload.get("report_contract") == "skillevaluator-ci-v1", "Invalid report_contract")
    for key in ("commit", "base_commit"):
        value = payload.get(key)
        if key == "base_commit" and value is None:
            continue
        _require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value) is not None,
            f"Invalid {key}",
        )
    repository = _path(payload.get("repository"), "repository")
    _require("/" in repository and " " not in repository, "Invalid repository")
    _require(payload.get("platform") in ("github", "gitlab"), "Invalid platform")
    _require(payload.get("environment") in ("docker", "local"), "Invalid environment")
    for key in ("run_id", "run_attempt"):
        value = payload.get(key)
        _require(isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,29}", value) is not None, f"Invalid {key}")
    tiers = payload.get("tiers")
    _require(isinstance(tiers, list) and 0 < len(tiers) <= 3, "Manifest requires nonempty tiers")
    expected = payload.get("expected_tiers", [1, 2, 3])
    _require(
        isinstance(expected, list)
        and bool(expected)
        and all(type(tier) is int and tier in (1, 2, 3) for tier in expected)
        and len(expected) == len(set(expected)),
        "Invalid expected_tiers",
    )
    seen = set()
    for record in tiers:
        _require(isinstance(record, dict), "Tier must be an object")
        tier = record.get("tier")
        _require(type(tier) is int and tier in {1, 2, 3} and tier not in seen, "Invalid or duplicate tier")
        _require(tier in expected, "Unexpected tier outcome")
        seen.add(tier)
        _require(type(record.get("required")) is bool, "Tier required must be boolean")
        _require(tier != 1 or record["required"], "Tier 1 must be required")
        _require(isinstance(record.get("status"), str) and record["status"] in _STATUSES, "Invalid tier status")
        for key in ("reason", "error"):
            if key in record:
                _text(record[key], f"tier.{key}", empty=True)
        if "metrics" in record:
            _require(isinstance(record["metrics"], dict), "Invalid tier metrics")
        skills = record.get("skills")
        _require(isinstance(skills, list) and len(skills) <= 1000, "Invalid skills")
        paths = set()
        for skill in skills:
            _require(isinstance(skill, dict), "Skill must be an object")
            path = _path(skill.get("path"), "skill path", skill=True)
            _require(path not in paths, "Duplicate skill path")
            paths.add(path)
            _require(isinstance(skill.get("status"), str) and skill["status"] in _STATUSES, "Invalid skill status")
            for key in ("report_json", "report_html"):
                _require(key in skill, f"Missing {key}")
                if skill[key] is not None:
                    _path(skill[key], key)
            if "error" in skill:
                _text(skill["error"], "skill error", empty=True)
            if "metrics" in skill:
                _require(isinstance(skill["metrics"], dict), "Invalid skill metrics")
        if record["status"] == "passed":
            _require(bool(skills), "Passed tier cannot have empty skills")
            _require(all(skill["status"] == "passed" for skill in skills), "Tier status contradicts skills")
            _require(
                not record.get("error") and not any(skill.get("error") for skill in skills),
                "Passed tier contains execution errors",
            )
    required = [record for record in tiers if record["required"]]
    missing = set(expected) & {1} - seen

    def satisfies_gate(record):
        return record["status"] == "passed" or (
            record["status"] == "skipped" and not record["skills"] and bool(record.get("reason"))
        )

    overall = not missing and all(satisfies_gate(record) for record in required)
    observed = {record["status"] for record in required or tiers}
    status = (
        "skipped"
        if observed == {"skipped"} and overall
        else "passed"
        if overall and required
        else next(
            (
                candidate
                for candidate in ("failed", "incomplete", "needs_action", "not_run", "neutral", "skipped", "passed")
                if candidate in observed
            ),
            "not_run",
        )
    )
    if missing:
        status = "incomplete" if status != "failed" else status
    if "overall_passed" in payload:
        _require(
            type(payload["overall_passed"]) is bool and payload["overall_passed"] == overall,
            "Overall verdict contradicts required tiers",
        )
    validated = copy.deepcopy(payload)
    if "overall_status" in payload:
        _require(payload["overall_status"] == status, "Overall status contradicts tier outcomes")
    validated["overall_passed"] = overall
    validated["overall_status"] = status
    validated["expected_tiers"] = list(expected)
    return validated


def tier3_execution_complete(report: dict) -> bool:
    """Recognize complete paired execution independently of benchmark eligibility.

    Callers retain the native quality verdict and benchmark publication assessment
    separately. Inapplicable baseline dimensions do not make completed attempts
    incomplete; missing coverage, run/source binding, or execution errors do.
    This helper reads the canonical native ``tier3`` carrier and never modifies
    the supplied report. Imports stay lazy for platform-neutral manifest users.
    """
    from skillevaluator.reporting.base import (
        _agent_eval_attempt_coverage_complete,
        _agent_eval_consistent_text_field,
        _agent_eval_execution_errors_clear,
        _agent_eval_has_publication_target_conflict,
        _agent_eval_truth_state,
        agent_eval_case_selection_issue,
        agent_eval_publication_dataset_provenance,
        agent_eval_publication_evaluated_at,
        agent_eval_publication_run_id,
        agent_eval_publication_target,
        agent_eval_report_serialization_issue,
    )

    if not isinstance(report, dict) or report.get("incomplete_scans", []) != []:
        return False
    results = report.get("results")
    if (
        not isinstance(results, list)
        or not results
        or any(not isinstance(result, dict) or result.get("incomplete_scans", []) != [] for result in results)
    ):
        return False
    payload = report.get("tier3")
    if not isinstance(payload, dict) or agent_eval_report_serialization_issue(payload) is not None:
        return False
    verdict, execution_status, truth_consistent = _agent_eval_truth_state(payload)
    target = agent_eval_publication_target(payload)
    return bool(
        _agent_eval_consistent_text_field(payload, "mode") == "both"
        and truth_consistent
        and execution_status == "succeeded"
        and verdict in {"pass", "neutral", "fail"}
        and _agent_eval_execution_errors_clear(payload)
        and _agent_eval_attempt_coverage_complete(payload, require_baseline=True)
        and agent_eval_publication_dataset_provenance(payload) is not None
        and target is not None
        and target.skill_name == _agent_eval_consistent_text_field(payload, "skill_name")
        and not _agent_eval_has_publication_target_conflict(payload)
        and not any("publication_target_conflict" in result for result in results)
        and agent_eval_publication_run_id(payload) is not None
        and agent_eval_publication_evaluated_at(payload) is not None
        and agent_eval_case_selection_issue(payload) is None
    )


def _tiers(payload: dict) -> list[dict]:
    by_tier = {record["tier"]: record for record in payload["tiers"]}
    return [
        by_tier.get(
            tier,
            {
                "tier": tier,
                "status": "not_run",
                "required": tier == 1 and tier in payload["expected_tiers"],
                "skills": [],
                "reason": "No retained result" if tier in payload["expected_tiers"] else "Not requested",
            },
        )
        for tier in (1, 2, 3)
    ]


def _md(value) -> str:
    value = html.escape(str(value), quote=False)
    return re.sub(r"([\\`*_{}\[\]()#!|])", r"\\\1", value).replace("\r", " ").replace("\n", " ")


def _metrics(record: dict) -> str:
    metrics = record.get("metrics", {})
    return "; ".join(
        f"{key.replace('_', ' ')}: "
        + (
            json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
            if isinstance(value, list | dict)
            else str(value)
        )
        for key, value in metrics.items()
    )[:8000]


def _report_base(value: str) -> str:
    if not value:
        return ""
    value = _text(value, "report_url")
    parsed = urlsplit(value)
    _require(
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and "\\" not in value
        and not parsed.query
        and not parsed.fragment,
        "Invalid report_url",
    )
    return value.rstrip("/") + "/"


def _gate_label(payload: dict) -> str:
    if payload["overall_status"] == "skipped" and payload["overall_passed"]:
        return "Skipped"
    if not any(record["required"] for record in _tiers(payload)):
        return "Advisory only"
    return "Passed" if payload["overall_passed"] else "Not passed"


def _gate_policy(payload: dict) -> str:
    records = _tiers(payload)
    required = ", ".join(f"Tier {record['tier']}" for record in records if record["required"]) or "none"
    advisory = ", ".join(f"Tier {record['tier']}" for record in records if not record["required"]) or "none"
    return f"Required tiers: {required}. Advisory tiers: {advisory}."


def render_markdown(payload: dict, report_url: str = "") -> str:
    """Render the required gate, every tier status, and actual native evidence links."""
    payload = validate_manifest(payload)
    base = _report_base(report_url)
    lines = [
        "# SkillEvaluator CI results",
        "",
        f"Required gate: **{_gate_label(payload)}**",
        "",
        _gate_policy(payload),
        "",
        "| Tier | Status | Gate | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for record in _tiers(payload):
        detail = "; ".join(part for part in (record.get("reason", ""), _metrics(record)) if part)
        lines.append(
            f"| Tier {record['tier']} | {record['status']} | {'Required' if record['required'] else 'Advisory'} | {_md(detail) or 'See skill reports'} |"
        )
    for record in _tiers(payload):
        if not record["skills"]:
            continue
        lines.extend(
            [
                "",
                f"## Tier {record['tier']} skill evidence",
                "",
                "| Skill | Status | Reports | Findings and metrics |",
                "| --- | --- | --- | --- |",
            ]
        )
        for skill in record["skills"]:
            links = []
            for key, label in (("report_html", "Native HTML"), ("report_json", "JSON")):
                if skill[key]:
                    url = base + quote(skill[key], safe="/-._~")
                    links.append(f"[{label}]({url})")
            detail = "; ".join(part for part in (skill.get("error", ""), _metrics(skill)) if part)
            lines.append(
                f"| {_md(skill['path'])} | {skill['status']} | {' · '.join(links) or 'Unavailable'} | {_md(detail) or '—'} |"
            )
    lines.extend(["", f"Commit: `{payload['commit']}` · Environment: {payload['environment']}", ""])
    return "\n".join(lines)


def _render_html(payload: dict) -> str:
    h = html.escape
    lines = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">",
        "<title>Three-tier CI results · SkillEvaluator</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#24292f}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}a{color:#315b25}p{line-height:1.6}.table{overflow:auto}</style></head><body>",
        "<h1>SkillEvaluator CI results</h1>",
        f"<p>Required gate: <strong>{_gate_label(payload)}</strong></p>",
        f"<p>{_gate_policy(payload)}</p>",
        '<div class="table"><table><thead><tr><th>Tier</th><th>Status</th><th>Gate</th><th>Evidence</th></tr></thead><tbody>',
    ]
    for record in _tiers(payload):
        detail = "; ".join(part for part in (record.get("reason", ""), _metrics(record)) if part)
        lines.append(
            f"<tr><td>Tier {record['tier']}</td><td>{record['status']}</td><td>{'Required' if record['required'] else 'Advisory'}</td><td>{h(detail)}</td></tr>"
        )
    lines.append("</tbody></table></div>")
    for record in _tiers(payload):
        if not record["skills"]:
            continue
        lines.append(
            f'<h2>Tier {record["tier"]} skill evidence</h2><div class="table"><table><thead><tr><th>Skill</th><th>Status</th><th>Reports</th><th>Findings and metrics</th></tr></thead><tbody>'
        )
        for skill in record["skills"]:
            links = []
            for key, label in (("report_html", "Native HTML"), ("report_json", "JSON")):
                if skill[key]:
                    links.append(f'<a href="{h(quote(skill[key], safe="/-._~"), quote=True)}">{label}</a>')
            detail = "; ".join(part for part in (skill.get("error", ""), _metrics(skill)) if part)
            lines.append(
                f"<tr><td>{h(skill['path'])}</td><td>{skill['status']}</td><td>{' · '.join(links) or 'Unavailable'}</td><td>{h(detail)}</td></tr>"
            )
        lines.append("</tbody></table></div>")
    lines.append(
        f'<p>Commit: <code>{payload["commit"]}</code> · Environment: {payload["environment"]}</p><p><a href="ci-results.json">Run manifest (JSON)</a> · <a href="summary.md">Markdown summary</a></p></body></html>'
    )
    return "\n".join(lines)


def write_reports(reports: Path, payload: dict) -> None:
    """Write manifest, Markdown and HTML without opening linked native evidence."""
    payload = validate_manifest(payload)
    reports = Path(reports).absolute()
    for directory in (reports, *reports.parents):
        _require(not directory.is_symlink(), "Report directory must not contain symlinks")
    reports.mkdir(parents=True, exist_ok=True)
    _require(reports.is_dir(), "Report directory is required")
    contents = {
        "ci-results.json": json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "summary.md": render_markdown(payload),
        "index.html": _render_html(payload),
    }
    # Preflight every output so rejection does not leave an updated partial set.
    for name in contents:
        destination = reports / name
        _require(
            not destination.is_symlink() and (not destination.exists() or destination.is_file()),
            "Report output must be a regular file",
        )
    for name, content in contents.items():
        descriptor, temporary = tempfile.mkstemp(prefix=".ci-report-", dir=reports)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(content)
            Path(temporary).replace(reports / name)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
