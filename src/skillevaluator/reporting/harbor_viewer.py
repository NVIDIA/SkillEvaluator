# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display helpers for Harbor artifacts report links."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlsplit


def normalize_harbor_viewer_for_display(agent_eval: dict[str, Any]) -> dict[str, Any]:
    """Return display-safe Harbor links from a canonical Tier 3 payload."""
    raw = agent_eval.get("harbor_viewer")
    if not isinstance(raw, dict):
        summary = agent_eval.get("summary")
        raw = summary.get("harbor_viewer") if isinstance(summary, dict) else {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, Any] = {}
    job_url = safe_url(raw.get("job_url"))
    analysis_url = safe_url(raw.get("analysis_url"))
    if job_url:
        out["job_url"] = job_url
    if analysis_url:
        out["analysis_url"] = analysis_url

    jobs: list[dict[str, str]] = []
    seen_jobs: set[str] = set()
    for item in raw.get("jobs") or []:
        if not isinstance(item, dict):
            continue
        url = safe_url(item.get("url") or item.get("job_url"))
        if not url or url in seen_jobs:
            continue
        seen_jobs.add(url)
        job: dict[str, str] = {"url": url}
        item_analysis = safe_url(item.get("analysis_url"))
        if item_analysis:
            job["analysis_url"] = item_analysis
        if item.get("name"):
            job["name"] = str(item["name"])
        jobs.append(job)
    if jobs:
        out["jobs"] = jobs
        out.setdefault("job_url", jobs[0]["url"])
        if jobs[0].get("analysis_url"):
            out.setdefault("analysis_url", jobs[0]["analysis_url"])

    evidence: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    for item in raw.get("evidence_links") or []:
        if not isinstance(item, dict):
            continue
        url = safe_url(item.get("url"))
        if not url or url in seen_evidence:
            continue
        seen_evidence.add(url)
        entry = dict(item)
        entry["url"] = url
        entry["label"] = harbor_evidence_label(entry)
        evidence.append(entry)
    if evidence:
        out["evidence_links"] = evidence
    return out


def normalize_agent_eval_harbor_links(agent_eval: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with display-safe Harbor links in nested fields."""
    normalized = dict(agent_eval)
    harbor_viewer = normalize_harbor_viewer_for_display(normalized)
    if harbor_viewer:
        normalized["harbor_viewer"] = harbor_viewer
        summary = normalized.get("summary")
        if isinstance(summary, dict):
            normalized_summary = dict(summary)
            normalized_summary["harbor_viewer"] = {
                key: harbor_viewer[key] for key in ("job_url", "analysis_url") if harbor_viewer.get(key)
            }
            normalized["summary"] = normalized_summary
    else:
        summary = normalized.get("summary")
        if isinstance(summary, dict) and "harbor_viewer" in summary:
            normalized_summary = dict(summary)
            normalized_summary.pop("harbor_viewer", None)
            normalized["summary"] = normalized_summary

    normalized["recommendations"] = _sanitize_items_with_evidence(
        normalized.get("recommendations"),
        evidence_keys=("evidence",),
    )
    normalized["conclusions"] = _sanitize_items_with_evidence(
        normalized.get("conclusions"),
        evidence_keys=("evidence",),
    )
    normalized["suggestions_v2"] = _sanitize_items_with_evidence(
        normalized.get("suggestions_v2"),
        evidence_keys=("harbor_evidence", "evidence"),
    )
    return normalized


def _sanitize_items_with_evidence(raw: object, *, evidence_keys: tuple[str, ...]) -> object:
    if not isinstance(raw, list):
        return raw

    sanitized: list[Any] = []
    for item in raw:
        if not isinstance(item, dict):
            sanitized.append(item)
            continue
        entry = dict(item)
        for key in evidence_keys:
            evidence = entry.get(key)
            if not isinstance(evidence, dict):
                continue
            url = safe_url(evidence.get("url"))
            if not url:
                entry.pop(key, None)
                continue
            normalized_evidence = dict(evidence)
            normalized_evidence["url"] = url
            normalized_evidence["label"] = harbor_evidence_label(normalized_evidence)
            entry[key] = normalized_evidence
        sanitized.append(entry)
    return sanitized


def safe_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    return url


def harbor_evidence_label(evidence: dict[str, Any]) -> str:
    step = evidence.get("step")
    if not isinstance(step, int):
        url = evidence.get("url")
        step = step_number_from_url(url) if isinstance(url, str) else None
    if isinstance(step, int) and step > 0:
        return f"Step {step}"
    label = evidence.get("label") or evidence.get("entry_id") or evidence.get("trial_id")
    return str(label).strip() if label else "evidence"


def harbor_evidence_link_text(evidence: dict[str, Any]) -> str:
    return f"View {harbor_evidence_label(evidence)}"


def step_number_from_url(url: str) -> int | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    for key, value in parse_qsl(parts.query, keep_blank_values=False):
        if key in {"step", "trajectory_step", "trajectoryStep"}:
            try:
                step = int(value)
            except (TypeError, ValueError):
                return None
            return step if step > 0 else None
    fragment = parts.fragment.strip().lower()
    for prefix in ("step-", "trajectory-step-"):
        if fragment.startswith(prefix):
            try:
                step = int(fragment[len(prefix) :])
            except ValueError:
                return None
            return step if step > 0 else None
    return None
