# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display helpers for Harbor artifacts report links."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlsplit


def normalize_harbor_viewer_for_display(agent_eval: dict[str, Any]) -> dict[str, Any]:
    """Return display-safe Harbor links from a canonical Tier 3 payload."""
    primary = _normalize_harbor_viewer(agent_eval.get("harbor_viewer"))
    if primary:
        return primary
    summary = agent_eval.get("summary")
    return _normalize_harbor_viewer(summary.get("harbor_viewer") if isinstance(summary, dict) else None)


def _normalize_harbor_upload_jobs(raw: object) -> list[dict[str, str]]:
    """Project recorded upload URLs without deriving links or changing source data."""
    jobs: list[dict[str, str]] = []
    seen: set[str] = set()
    for upload in raw if isinstance(raw, list) else []:
        if not isinstance(upload, dict):
            continue
        url = safe_url(upload.get("viewer_url") or upload.get("job_url"))
        if not url or url in seen:
            continue
        seen.add(url)
        job = {"url": url}
        analysis = safe_url(upload.get("analysis_url"))
        if analysis:
            job["analysis_url"] = analysis
        name = upload.get("uploaded_job_name") or upload.get("job_name") or upload.get("original_job_name")
        if isinstance(name, str) and name.strip():
            job["name"] = name.strip()
        jobs.append(job)
    return jobs


def _normalize_harbor_viewer(raw: object) -> dict[str, Any]:
    """Normalize one source before choosing primary or legacy summary evidence."""
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
    raw_jobs = raw.get("jobs")
    for item in raw_jobs if isinstance(raw_jobs, list) else []:
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
    by_url = {job["url"]: job for job in jobs}
    for uploaded in _normalize_harbor_upload_jobs(raw.get("uploads")):
        retained = by_url.get(uploaded["url"])
        if retained is None:
            jobs.append(uploaded)
            by_url[uploaded["url"]] = uploaded
        else:
            # Canonical metadata wins; enrich only the same validated job URL.
            for key, value in uploaded.items():
                retained.setdefault(key, value)
    if jobs:
        out["jobs"] = jobs
        out.setdefault("job_url", jobs[0]["url"])
        if jobs[0].get("analysis_url"):
            out.setdefault("analysis_url", jobs[0]["analysis_url"])

    evidence: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    raw_evidence = raw.get("evidence_links")
    for item in raw_evidence if isinstance(raw_evidence, list) else []:
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


def harbor_job_links(agent_eval: dict[str, Any]) -> list[dict[str, str]]:
    """Project every normalized job and unmatched legacy link, without attribution."""
    viewer = normalize_harbor_viewer_for_display(agent_eval)
    jobs = [dict(job) for job in viewer.get("jobs", [])]
    job_url, analysis_url = viewer.get("job_url"), viewer.get("analysis_url")
    unmatched_analysis = bool(analysis_url) and not any(job.get("analysis_url") == analysis_url for job in jobs)
    if job_url and not any(job["url"] == job_url for job in jobs):
        legacy = {"url": job_url}
        if unmatched_analysis:
            # A top-level job_url/analysis_url pair describes one job.
            legacy["analysis_url"] = analysis_url
            unmatched_analysis = False
        jobs.append(legacy)
    for index, job in enumerate(jobs, 1):
        job["name"] = job.get("name") or f"Job {index}"
    if unmatched_analysis:
        # A run-level analysis is not a job; never number it as one.
        jobs.append({"name": "Run analysis", "analysis_url": analysis_url})
    seen_analysis: set[str] = set()
    for job in jobs:
        analysis = job.get("analysis_url")
        if analysis in seen_analysis:
            job.pop("analysis_url", None)
        elif analysis:
            seen_analysis.add(analysis)
    return jobs


def harbor_job_display_links(agent_eval: dict[str, Any]) -> list[dict[str, str]]:
    """Use exact recorded job associations for concise labels, retaining identifiers."""
    labels: dict[str, list[str]] = {}

    def record(trial: object, name: object, condition: str) -> None:
        if not isinstance(name, str) or not name:
            return
        viewer = trial.get("harbor_viewer") if isinstance(trial, dict) else None
        url = safe_url(viewer.get("job_url")) if isinstance(viewer, dict) else None
        if url:
            associated = labels.setdefault(url, [])
            label = f"{name} · {condition}"
            if label not in associated:
                associated.append(label)

    trials = agent_eval.get("trials")
    if isinstance(trials, list):
        for trial in trials:
            if isinstance(trial, dict):
                record(
                    trial,
                    trial.get("agent"),
                    "Without skill" if agent_eval.get("mode") == "baseline_only" else "With skill",
                )
    agents = agent_eval.get("agents")
    if isinstance(agents, dict):
        for name, agent in agents.items():
            if not isinstance(agent, dict):
                continue
            for key, condition in (("trials", "With skill"), ("trials_baseline", "Without skill")):
                trials = agent.get(key)
                if not isinstance(trials, list):
                    continue
                for trial in trials:
                    record(trial, name, condition)
    jobs = harbor_job_links(agent_eval)
    for index, job in enumerate(jobs, 1):
        job["display_label"] = "; ".join(labels.get(job.get("url", ""), [])) or (
            "Run analysis" if job.get("name") == "Run analysis" else f"Job {index}"
        )
    return jobs


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
        normalized.pop("harbor_viewer", None)
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
