# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HTML report generator for Harbor evaluation results.

Produces a self-contained HTML file with tabbed navigation:
  - Overview tab: radar chart + score comparison + lift
  - Per-agent tabs: detailed findings, behavior checks, error recovery
  - Suggestions tab: LLM-generated and contextual next steps
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRICS,
    DIMENSION_DEFINITIONS,
    LEGACY_METRICS,
    METRIC_DESCRIPTIONS,
    METRIC_DISPLAY,
    extract_custom_metrics,
)

logger = logging.getLogger(__name__)

DISPLAY_METRICS = list(DEFAULT_METRICS)
_METRIC_DISPLAY = METRIC_DISPLAY
_METRIC_DESC = METRIC_DESCRIPTIONS
_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_EVIDENCE_LINK_RE = re.compile(r"Evidence:\s*(?P<url>https?://[^\s<>\"]+)")

AGENT_COLORS = {
    "claude-code": "#76b900",
    "opencode": "#3b82f6",
    "codex": "#8b5cf6",
    "openhands": "#f59e0b",
    "cursor-cli": "#ec4899",
    "mini-swe-agent": "#06b6d4",
    "aider": "#14b8a6",
    "gemini-cli": "#f97316",
}


def _agent_color(name: str, idx: int = 0) -> str:
    fallback = ["#76b900", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899", "#14b8a6"]
    return AGENT_COLORS.get(name, fallback[idx % len(fallback)])


def _agent_model_label(
    agent: str,
    run_config: dict[str, Any],
    agents: dict[str, dict[str, Any]],
) -> str:
    run_agents = run_config.get("agents", {}) if isinstance(run_config, dict) else {}
    meta = run_agents.get(agent, {}) if isinstance(run_agents, dict) else {}
    if isinstance(meta, dict):
        model = str(meta.get("model") or "").strip()
        if model:
            return f"{agent} / {model}"

    agent_data = agents.get(agent, {})
    if isinstance(agent_data, dict):
        model = str(agent_data.get("model") or "").strip()
        if model:
            return f"{agent} / {model}"
    return agent


def _dimension_source_label(dimension: str) -> str:
    """Return a readable metric grouping label for a report-only dimension."""
    sources = DIMENSION_DEFINITIONS.get(dimension, {})
    if not sources:
        return ""
    total_weight = sum(sources.values())
    parts: list[str] = []
    for metric, weight in sources.items():
        label = _METRIC_DISPLAY.get(metric, metric)
        if len(sources) > 1 and total_weight > 0:
            parts.append(f"{label} ({weight / total_weight:.0%})")
        else:
            parts.append(label)
    return " + ".join(parts)


def _metrics_for_rewards(rewards: list[dict[str, Any]]) -> list[str]:
    if any(isinstance(r.get("security"), int | float) for r in rewards):
        return list(DEFAULT_METRICS)
    if any(any(isinstance(r.get(m), int | float) for m in LEGACY_METRICS) for r in rewards):
        return list(LEGACY_METRICS)
    return []


def _custom_metric_names_for_agent(agent_info: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    custom_scores = agent_info.get("custom_with_skill")
    if isinstance(custom_scores, dict):
        names.update(str(metric) for metric in custom_scores)
    rewards = agent_info.get("rewards", [])
    if isinstance(rewards, list):
        for reward in rewards:
            if not isinstance(reward, dict):
                continue
            names.update(extract_custom_metrics(reward))
            custom_details = reward.get("custom_details")
            if isinstance(custom_details, dict):
                names.update(str(metric) for metric in custom_details)
    return sorted(names)


def _skill_evaluator_metrics_for_agent(agent_info: dict[str, Any]) -> list[str]:
    configured = agent_info.get("metrics_with_skill")
    if isinstance(configured, list):
        return [str(m) for m in configured]
    scores = agent_info.get("with_skill", {})
    if isinstance(scores, dict) and "security" in scores:
        return list(DEFAULT_METRICS)
    rewards = agent_info.get("rewards", [])
    return _metrics_for_rewards(rewards) if isinstance(rewards, list) else []


def _metrics_for_agent(agent_info: dict[str, Any]) -> list[str]:
    metrics = _skill_evaluator_metrics_for_agent(agent_info)
    return metrics + [metric for metric in _custom_metric_names_for_agent(agent_info) if metric not in metrics]


def _metrics_for_agents(agents: dict[str, dict[str, Any]]) -> list[str]:
    saw_metrics = False
    for info in agents.values():
        metrics = _skill_evaluator_metrics_for_agent(info)
        if metrics:
            saw_metrics = True
        if "security" in metrics:
            return list(DEFAULT_METRICS)
    return list(LEGACY_METRICS) if saw_metrics else []


def _mean_numeric(values: list[Any]) -> float:
    numeric = [float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _agent_overall(agent_info: dict[str, Any], metrics: list[str]) -> float:
    scores = agent_info.get("with_skill", {})
    if metrics:
        return sum(float(scores.get(m, 0.0) or 0.0) for m in metrics) / len(metrics)
    custom_scores = agent_info.get("custom_with_skill", {})
    if isinstance(custom_scores, dict) and custom_scores:
        return _mean_numeric(list(custom_scores.values()))
    rewards = agent_info.get("rewards", [])
    if isinstance(rewards, list):
        return _mean_numeric([r.get("overall") for r in rewards if isinstance(r, dict)])
    return 0.0


def _reward_overall(reward: dict[str, Any], metrics: list[str]) -> float:
    if metrics:
        return sum(float(reward.get(m, 0.0) or 0.0) for m in metrics) / len(metrics)
    if isinstance(reward.get("overall"), int | float) and not isinstance(reward.get("overall"), bool):
        return float(reward["overall"])
    custom = reward.get("custom_metrics")
    if isinstance(custom, dict):
        values = []
        for value in custom.values():
            if isinstance(value, dict):
                value = value.get("score")
            values.append(value)
        return _mean_numeric(values)
    return 0.0


def _detail_for_metric(reward: dict[str, Any], metric: str) -> dict[str, Any]:
    details = reward.get("details")
    detail = details.get(metric) if isinstance(details, dict) else None
    if isinstance(detail, dict):
        return detail
    custom_details = reward.get("custom_details")
    custom_detail = custom_details.get(metric) if isinstance(custom_details, dict) else None
    return custom_detail if isinstance(custom_detail, dict) else {}


def _nonnegative_counter(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _condition_status(agent_info: dict[str, Any], condition: str) -> str:
    conditions = agent_info.get("conditions")
    data = conditions.get(condition) if isinstance(conditions, dict) else None
    return str(data.get("execution_status") or "unknown") if isinstance(data, dict) else "unknown"


def _both_conditions_succeeded(agent_info: dict[str, Any]) -> bool:
    return all(_condition_status(agent_info, condition) == "succeeded" for condition in ("with_skill", "without_skill"))


def _grading_mode(run_config: dict[str, Any]) -> str:
    grading = run_config.get("grading")
    mode = grading.get("mode") if isinstance(grading, dict) else None
    if isinstance(mode, dict):
        mode = mode.get("value")
    return str(mode or "").strip()


def _load_agent_data(results_dir: Path) -> dict[str, dict[str, Any]]:
    agents: dict[str, dict[str, Any]] = {}
    for agent_dir in sorted(results_dir.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue
        agent_name = agent_dir.name
        agent_info: dict[str, Any] = {"name": agent_name}
        condition_execution: dict[str, dict[str, Any]] = {}

        for variant in ("with-skill", "without-skill"):
            summary = agent_dir / variant / "summary.json"
            if summary.exists():
                try:
                    data = json.loads(summary.read_text(encoding="utf-8"))
                    key = "with_skill" if variant == "with-skill" else "without_skill"
                    agent_info[key] = data.get("scores", data)
                    metric_key = "metrics_with_skill" if variant == "with-skill" else "metrics_without_skill"
                    agent_info[metric_key] = data.get("metrics", [])
                    custom_key = "custom_with_skill" if variant == "with-skill" else "custom_without_skill"
                    if "custom_scores" in data:
                        agent_info[custom_key] = data.get("custom_scores", {})
                    dim_key = "dimensions_with_skill" if variant == "with-skill" else "dimensions_without_skill"
                    if "dimensions" in data:
                        agent_info[dim_key] = data.get("dimensions", {})
                    pass_key = "pass_with_skill" if variant == "with-skill" else "pass_without_skill"
                    if "pass_at_k" in data:
                        agent_info[pass_key] = data["pass_at_k"]
                    status = data.get("execution_status")
                    if status not in {"succeeded", "failed", "skipped"}:
                        status = "unknown"
                    errors = data.get("execution_errors")
                    condition_errors = [str(error) for error in errors] if isinstance(errors, list) else []
                    label = "With skill" if variant == "with-skill" else "Without skill"
                    job_failure = data.get("job_failure")
                    if job_failure:
                        condition_errors.append(f"{label} aggregate job: {job_failure}")
                    trial_failures = data.get("trial_failures")
                    if isinstance(trial_failures, list):
                        for failure in trial_failures:
                            if not isinstance(failure, dict):
                                continue
                            trial = failure.get("trial") or "unknown"
                            reason = failure.get("reason") or "Unknown Harbor trial failure"
                            condition_errors.append(f"{label} trial {trial}: {reason}")
                    condition_execution[key] = {
                        "execution_status": status,
                        "execution_errors": condition_errors,
                        "expected_attempts": _nonnegative_counter(data.get("expected_attempts")),
                        "scored_attempts": _nonnegative_counter(data.get("scored_attempts")),
                    }
                    if variant == "with-skill":
                        agent_info["num_trials"] = data.get("num_trials", 0)
                except (json.JSONDecodeError, OSError):
                    pass

        lift_file = agent_dir / "lift.json"
        if lift_file.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                agent_info["lift"] = json.loads(lift_file.read_text(encoding="utf-8"))

        pass_lift_file = agent_dir / "pass_at_k_lift.json"
        if pass_lift_file.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                agent_info["pass_lift"] = json.loads(pass_lift_file.read_text(encoding="utf-8"))

        custom_lift_file = agent_dir / "custom_lift.json"
        if custom_lift_file.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                agent_info["custom_lift"] = json.loads(custom_lift_file.read_text(encoding="utf-8"))

        for variant_key, variant_dir_name in [("rewards", "with-skill"), ("rewards_baseline", "without-skill")]:
            trial_list: list[dict[str, Any]] = []
            trials_dir = agent_dir / variant_dir_name / "trials"
            if trials_dir.exists():
                for trial_dir in sorted(trials_dir.iterdir()):
                    if not trial_dir.is_dir():
                        continue
                    rf = trial_dir / "reward.json"
                    if not rf.exists():
                        continue
                    try:
                        reward = json.loads(rf.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    if not reward.get("entry_id"):
                        reward["entry_id"] = trial_dir.name.split("__", 1)[0] if trial_dir.name else "unknown"
                    tf = trial_dir / "trajectory.json"
                    if tf.exists():
                        try:
                            traj = json.loads(tf.read_text(encoding="utf-8"))
                            fm = traj.get("final_metrics", {})
                            reward["_traj"] = {
                                "steps": len(traj.get("steps", [])),
                                "prompt_tokens": fm.get("total_prompt_tokens", 0),
                                "completion_tokens": fm.get("total_completion_tokens", 0),
                                "cached_tokens": fm.get("total_cached_tokens", 0),
                            }
                        except (json.JSONDecodeError, OSError):
                            pass
                    trial_list.append(reward)
            agent_info[variant_key] = trial_list

        if "with_skill" in agent_info:
            active_conditions = list(condition_execution.values())
            execution_errors = [
                error for condition in active_conditions for error in condition.get("execution_errors", []) if error
            ]
            if not active_conditions or any(
                condition.get("execution_status") in {"failed", "unknown"} for condition in active_conditions
            ):
                execution_status = "failed" if execution_errors else "unknown"
            elif all(condition.get("execution_status") == "skipped" for condition in active_conditions):
                execution_status = "skipped"
            else:
                execution_status = "succeeded"
            agent_info.update(
                {
                    "conditions": condition_execution,
                    "execution_status": execution_status,
                    "execution_errors": list(dict.fromkeys(execution_errors)),
                    "expected_attempts": sum(
                        _nonnegative_counter(condition.get("expected_attempts")) for condition in active_conditions
                    ),
                    "scored_attempts": sum(
                        _nonnegative_counter(condition.get("scored_attempts")) for condition in active_conditions
                    ),
                }
            )
            # A failed/unknown condition may leave partial or legacy score
            # artifacts on disk. Keep its failure and coverage metadata, but
            # remove every quality-bearing payload before any report surface
            # can interpret those incomplete values as valid scores.
            condition_quality_fields = {
                "with_skill": (
                    "with_skill",
                    "custom_with_skill",
                    "dimensions_with_skill",
                    "pass_with_skill",
                    "rewards",
                ),
                "without_skill": (
                    "without_skill",
                    "custom_without_skill",
                    "dimensions_without_skill",
                    "pass_without_skill",
                    "rewards_baseline",
                ),
            }
            for condition, fields in condition_quality_fields.items():
                condition_status = _condition_status(agent_info, condition)
                if condition_status == "succeeded":
                    continue
                condition_info = condition_execution.get(condition, {})
                for field in fields:
                    if field.startswith("pass_") and condition_status in {"failed", "unknown"}:
                        agent_info[field] = {
                            "attempts_used": _nonnegative_counter(condition_info.get("scored_attempts")),
                            "max_attempts_possible": _nonnegative_counter(condition_info.get("expected_attempts")),
                        }
                    else:
                        agent_info[field] = [] if field.startswith("rewards") else {}
            agents[agent_name] = agent_info
    return agents


def _sc(score: float) -> str:
    if score >= 0.8:
        return "#22c55e"
    if score >= 0.6:
        return "#eab308"
    return "#ef4444"


def _badge(score: float) -> str:
    if score >= 0.8:
        return '<span class="badge pass">PASS</span>'
    if score >= 0.6:
        return '<span class="badge warn">WARN</span>'
    return '<span class="badge fail">FAIL</span>'


def _metric_evidence_url(reward: dict[str, Any], metric: str) -> str:
    harbor_viewer = reward.get("harbor_viewer")
    if not isinstance(harbor_viewer, dict):
        return ""
    evidence_urls = harbor_viewer.get("evidence_urls")
    if not isinstance(evidence_urls, list):
        return ""
    metric_keys = _metric_evidence_keys(metric)
    for item in evidence_urls:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("metric") or "").strip().lower()
        url = str(item.get("url") or "").strip()
        if label in metric_keys and url:
            return url
    return ""


def _metric_evidence_keys(metric: str) -> set[str]:
    normalized = str(metric).strip().lower()
    aliases = {
        "skill_efficiency": {"skill_efficiency", "efficiency"},
        "skill_execution": {"skill_execution", "execution"},
    }
    return aliases.get(normalized, {normalized})


def _build_agent_findings(agent_info: dict[str, Any]) -> str:
    rewards = agent_info.get("rewards", [])
    if not rewards:
        if _condition_status(agent_info, "with_skill") != "succeeded":
            return (
                "<p class='subtle'>No valid with-skill score is available because execution did not complete "
                "successfully. Review Failure Details above.</p>"
            )
        return "<p class='subtle'>No trial data available.</p>"

    cards: list[str] = []
    for metric in _metrics_for_agent(agent_info):
        display = _METRIC_DISPLAY.get(metric, metric)
        desc = _METRIC_DESC.get(metric, "")
        scores = [r.get(metric, 0.0) for r in rewards if isinstance(r.get(metric), (int, float))]
        avg = sum(scores) / len(scores) if scores else 0.0

        items: list[str] = []
        for reward in rewards:
            d = _detail_for_metric(reward, metric)
            evidence_url = _metric_evidence_url(reward, metric)
            evidence_html = f" {_compact_evidence_link(evidence_url)}" if evidence_url else ""

            if metric == "behavior_check":
                for r in d.get("results", []):
                    ok = r.get("passed", False)
                    cls = "ok" if ok else "err"
                    sym = "&#10003;" if ok else "&#10007;"
                    items.append(
                        f'<li class="{cls}"><span class="sym">{sym}</span>{escape(r.get("reason", "")[:140])}{evidence_html}</li>'
                    )

            elif metric == "accuracy":
                for crit, passed in d.get("criteria", {}).items():
                    cls = "ok" if passed else "err"
                    sym = "&#10003;" if passed else "&#10007;"
                    items.append(f'<li class="{cls}"><span class="sym">{sym}</span>{crit}{evidence_html}</li>')
                if d.get("reason"):
                    items.append(f'<li class="note">{escape(d["reason"][:200])}{evidence_html}</li>')

            elif metric == "goal_accuracy":
                if d.get("reason"):
                    items.append(f'<li class="note">{escape(d["reason"][:250])}{evidence_html}</li>')

            elif metric == "security":
                if d.get("reason"):
                    items.append(f'<li class="note">{escape(d["reason"][:250])}{evidence_html}</li>')
                for finding in d.get("findings", []):
                    if isinstance(finding, str):
                        items.append(
                            f'<li class="err"><span class="sym">&#10007;</span>{escape(finding[:180])}{evidence_html}</li>'
                        )
                        continue
                    if not isinstance(finding, dict):
                        continue
                    score_impact = bool(finding.get("score_impact"))
                    cls = "err" if score_impact else "ok"
                    sym = "&#10007;" if score_impact else "&#9432;"
                    message = str(finding.get("message") or finding.get("type") or "")
                    attribution = str(finding.get("attribution") or "")
                    if attribution:
                        message = f"{message} | Attribution: {attribution.replace('_', ' ')}"
                    items.append(
                        f'<li class="{cls}"><span class="sym">{sym}</span>{escape(message[:220])}{evidence_html}</li>'
                    )
                    explanation = str(finding.get("attribution_explanation") or "")
                    if explanation:
                        items.append(f'<li class="sub">{escape(explanation[:220])}{evidence_html}</li>')

            elif metric == "skill_execution":
                for cn, cd in d.items():
                    if not isinstance(cd, dict):
                        continue
                    ok = cd.get("passed", True)
                    cls = "ok" if ok else "err"
                    sym = "&#10003;" if ok else "&#10007;"
                    items.append(
                        f'<li class="{cls}"><span class="sym">{sym}</span><b>{escape(cn)}</b>: {escape(cd.get("reason", "")[:120])}{evidence_html}</li>'
                    )
                    if cn == "error_recovery":
                        for corr in cd.get("corrections", []):
                            f_type = corr.get("fault", "?")
                            bcls = "fail" if f_type == "skill" else "warn"
                            items.append(
                                f'<li class="sub"><span class="badge {bcls}">{f_type}</span> {escape(corr.get("error", "")[:120])}{evidence_html}</li>'
                            )

            elif metric == "skill_efficiency":
                for cn, cd in d.items():
                    if not isinstance(cd, dict):
                        continue
                    ok = cd.get("passed", True)
                    cls = "ok" if ok else "err"
                    sym = "&#10003;" if ok else "&#10007;"
                    items.append(
                        f'<li class="{cls}"><span class="sym">{sym}</span><b>{escape(cn)}</b>: {escape(cd.get("reason", "")[:120])}{evidence_html}</li>'
                    )

            else:
                if d.get("reason"):
                    items.append(f'<li class="note">{escape(str(d["reason"])[:250])}{evidence_html}</li>')
                for finding in d.get("findings", []):
                    if isinstance(finding, str):
                        items.append(f'<li class="note">{escape(finding[:180])}{evidence_html}</li>')
                    elif isinstance(finding, dict):
                        message = str(finding.get("message") or finding.get("reason") or "")
                        if message:
                            items.append(f'<li class="note">{escape(message[:180])}{evidence_html}</li>')

        seen: set[str] = set()
        deduped: list[str] = []
        for it in items:
            k = it[:100].lower()
            if k not in seen:
                seen.add(k)
                deduped.append(it)

        list_html = f'<ul class="checks">{"".join(deduped[:10])}</ul>' if deduped else ""

        refs = []
        for reward in rewards:
            refs.extend(_detail_for_metric(reward, metric).get("evidence_refs", []) or [])
        ref_html = ""
        if refs:
            _seen: set[tuple[str | None, str, str | None]] = set()
            _lis: list[str] = []
            for r in refs:
                if isinstance(r, str):
                    # Legacy string ref: "source#/json_pointer"
                    if "#" in r:
                        r_source, _, r_pointer = r.partition("#")
                    else:
                        r_source, r_pointer = r, ""
                    key = (r_source, r_pointer, None)
                    if key in _seen:
                        continue
                    _seen.add(key)
                    _lis.append(
                        f'<li class="evidence"><code>evidence</code> <code>{escape(r_source + r_pointer)}</code> </li>'
                    )
                    continue
                loc = r.get("json_pointer") or r.get("path") or ""
                key = (r.get("source"), loc, r.get("kind"))
                if key in _seen:
                    continue
                _seen.add(key)
                _lis.append(
                    f'<li class="evidence"><code>{escape(str(r.get("kind", "")))}</code> '
                    f"<code>{escape(str(r.get('source', '')) + str(loc))}</code> "
                    f"{escape(str(r.get('label') or r.get('excerpt') or '')[:120])}</li>"
                )
            ref_html = (
                f'<details class="evidence-refs"><summary>Evidence ({len(_lis)})</summary>'
                f"<ul>{''.join(_lis[:8])}</ul></details>"
            )

        cards.append(f"""<div class="card" style="border-left-color:{_sc(avg)}">
  <div class="card-head"><div><span class="card-title">{escape(display)}</span><span class="card-desc">{escape(desc)}</span></div>
  <div class="card-score">{_badge(avg)} <span class="score" style="color:{_sc(avg)}">{avg:.2f}</span></div></div>
  {list_html}
  {ref_html}
</div>""")

    return "\n".join(cards)


def _build_execution_failure_details(agent_info: dict[str, Any]) -> str:
    status = str(agent_info.get("execution_status") or "unknown")
    if status == "succeeded":
        return ""
    expected = int(agent_info.get("expected_attempts", 0) or 0)
    scored = int(agent_info.get("scored_attempts", 0) or 0)
    errors = agent_info.get("execution_errors")
    items = [escape(str(error)) for error in errors] if isinstance(errors, list) else []
    if not items:
        items = ["This legacy result has no explicit successful execution status."]
    error_list = "".join(f"<li>{item}</li>" for item in items)
    return (
        '<div class="info-box failure-details"><h3>Failure Details — Execution Status: '
        f"{escape(status.upper())}</h3><p>Scored {scored} of {expected} expected logical attempts.</p>"
        f"<ul>{error_list}</ul></div>"
    )


def _load_dataset(skill_path: Path | None) -> list[dict[str, Any]]:
    """Load evals.json from the skill directory."""
    if not skill_path:
        return []
    evals_dir = skill_path / "evals"
    for name in ("evals.json", "evals.jsonl", "evals.yaml", "evals.yml", "dataset.json"):
        candidate = evals_dir / name
        if candidate.exists():
            try:
                from skillevaluator.tier3.dataset_utils import load_dataset_entries

                return load_dataset_entries(candidate)
            except (json.JSONDecodeError, OSError, ValueError):
                pass
    return []


def _load_staged_harbor_dataset(results_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    tasks_dir = results_dir / "_harbor-tasks"
    if not tasks_dir.exists():
        return entries
    # Real runs stage tasks below <agent>/<condition>/<case>, while older
    # artifacts used a flat <case> layout. Read both and collapse the duplicate
    # with-skill/baseline copies by dataset entry id.
    for entry_file in sorted(tasks_dir.rglob("tests/entry.json")):
        try:
            entry = json.loads(entry_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(entry, dict):
            entry_id = entry.get("id")
            identity = f"id:{entry_id}" if entry_id is not None else f"payload:{json.dumps(entry, sort_keys=True)}"
            if identity in seen:
                continue
            seen.add(identity)
            entries.append(entry)
    return entries


def _build_dataset_html(entries: list[dict[str, Any]]) -> str:
    """Build the Dataset tab content from evals.json entries."""
    if not entries:
        return '<p class="subtle">No dataset found. Generate one with: <code>skillevaluator create-eval-dataset &lt;skill&gt;</code></p>'

    cards: list[str] = []
    for entry in entries:
        eid = escape(str(entry.get("id", "?")))
        prompt = escape(str(entry.get("prompt") or entry.get("question") or "")[:500])
        expected_output = escape(str(entry.get("expected_output") or entry.get("ground_truth") or ""))[:400]
        expected_skill = escape(str(entry.get("expected_skill", "") or "none"))
        expected_script = escape(str(entry.get("expected_script", "") or "none"))
        assertions = entry.get("assertions", entry.get("expected_behavior", []))
        if isinstance(assertions, list) and assertions:
            assertion_items = "".join(f"<li>{escape(str(assertion)[:150])}</li>" for assertion in assertions)
            assertions_html = f'<div class="ds-field"><span class="ds-label">Assertions</span><ol class="ds-behaviors">{assertion_items}</ol></div>'
        else:
            assertions_html = ""

        cards.append(f"""<div class="ds-card">
  <div class="ds-head">
    <span class="ds-id">{eid}</span>
  </div>
  <div class="ds-field"><span class="ds-label">Prompt</span><div class="ds-value">{prompt}</div></div>
  <div class="ds-field"><span class="ds-label">Expected Output</span><div class="ds-value ds-gt">{expected_output or '<span class="subtle">not specified</span>'}</div></div>
  <div class="ds-meta">
    <span>Expected skill: <b>{expected_skill}</b></span>
    <span>Expected script: <b>{expected_script}</b></span>
  </div>
  {assertions_html}
</div>""")

    return f'<p class="subtle">{len(entries)} AgentSkills eval case(s) in dataset</p>\n' + "\n".join(cards)


def _build_suggestions_html(skill_name: str, agents: dict[str, dict[str, Any]]) -> str:
    all_rewards = [r for a in agents.values() for r in a.get("rewards", [])]

    def suggestion_items(suggestions: list[str]) -> str:
        return "".join(f"<li>{_format_suggestion_item(s)}</li>" for s in suggestions)

    if any(agent.get("execution_status") != "succeeded" for agent in agents.values()):
        return (
            '<div class="sug-box sug-warn"><h3>&#9888; Resolve Execution Failures</h3>'
            "<p>Resolve the execution failures shown in Failure Details, then rerun the evaluation before "
            "interpreting scores or quality suggestions.</p></div>"
        )

    try:
        from skillevaluator.tier3.harbor.report import (
            _extract_findings,
            _fallback_suggestions,
            _generate_suggestions,
            _passing_skill_suggestions,
            add_evidence_links_to_suggestions,
        )

        findings = _extract_findings(all_rewards)
        has_warning_findings = any(finding.get("severity") in {"critical", "warning"} for finding in findings)
        if not has_warning_findings:
            suggestions = add_evidence_links_to_suggestions(
                _passing_skill_suggestions(findings, all_rewards),
                all_rewards,
            )
            if suggestions:
                items = suggestion_items(suggestions)
                return f'<div class="sug-box sug-pass"><h3>&#128161; Next Steps</h3><ol>{items}</ol></div>'
            return """<div class="sug-box sug-pass"><h3>&#128161; Next Steps</h3><ol>
<li>Add <code>evals/environment/Dockerfile</code> if the skill depends on CLI tools or APIs.</li>
<li>Expand <code>evals.json</code> with more test cases: <code>skillevaluator create-eval-dataset --full</code></li>
<li>Run with additional agents to verify cross-agent compatibility.</li></ol></div>"""

        suggestions = _generate_suggestions(skill_name, findings, all_rewards)
        if suggestions:
            suggestions = add_evidence_links_to_suggestions(suggestions, all_rewards)
            items = suggestion_items(suggestions)
            return f'<div class="sug-box sug-warn"><h3>&#128161; Suggestions</h3><ol>{items}</ol></div>'
        suggestions = add_evidence_links_to_suggestions(
            _fallback_suggestions(findings),
            all_rewards,
        )
        if suggestions:
            items = suggestion_items(suggestions)
            return f'<div class="sug-box sug-warn"><h3>&#128161; Suggestions</h3><ol>{items}</ol></div>'
    except Exception as e:
        logger.debug("LLM suggestions failed: %s", e)

    return '<div class="sug-box sug-warn"><h3>&#128161; Suggestions</h3><p>Review the findings above and address critical/warning items.</p></div>'


def _format_suggestion_item(text: str) -> str:
    value = str(text)
    parts: list[str] = []
    cursor = 0
    for match in _EVIDENCE_LINK_RE.finditer(value):
        before = value[cursor : match.start()].rstrip()
        if before:
            parts.append(_linkify_urls(before))
            parts.append(" ")
        parts.append(_compact_evidence_link(match.group("url")))
        cursor = match.end()
    if cursor < len(value):
        remainder = value[cursor:].strip()
        if remainder:
            parts.append(_linkify_urls(remainder))
    if not parts:
        return _linkify_urls(value)
    return "".join(parts)


def _compact_evidence_link(url: str) -> str:
    safe_url = escape(url, quote=True)
    label = _evidence_link_label(url)
    aria_label = (
        f"View Harbor Log Viewer trajectory {label.lower()}"
        if label.startswith("Step ")
        else "View evidence in Harbor Log Viewer"
    )
    return (
        f'<a class="evidence-link" href="{safe_url}" target="_blank" '
        f'rel="noopener noreferrer" aria-label="{escape(aria_label, quote=True)}">'
        '<span class="evidence-link-icon" aria-hidden="true">&#128279;</span>'
        f"<span>{escape(label)}</span></a>"
    )


def _evidence_link_label(url: str) -> str:
    query = parse_qs(urlsplit(url).query)
    for key in ("step", "trajectory_step", "trajectoryStep"):
        values = query.get(key)
        if values and str(values[0]).strip():
            return f"Step {values[0]}"
    return "View evidence"


def _first_harbor_analysis_url(agents: dict[str, dict[str, Any]]) -> str:
    for agent_info in agents.values():
        rewards = agent_info.get("rewards", [])
        if not isinstance(rewards, list):
            continue
        for reward in rewards:
            if not isinstance(reward, dict):
                continue
            harbor_viewer = reward.get("harbor_viewer")
            if not isinstance(harbor_viewer, dict):
                continue
            analysis_url = str(harbor_viewer.get("analysis_url") or "").strip()
            if analysis_url:
                return analysis_url
            job_url = str(harbor_viewer.get("job_url") or "").strip()
            if job_url:
                return job_url
    return ""


def _build_harbor_analysis_html(agents: dict[str, dict[str, Any]]) -> str:
    analysis_url = _first_harbor_analysis_url(agents)
    if not analysis_url:
        return ""
    safe_url = escape(analysis_url, quote=True)
    return f"""<div class="info-box harbor-analysis">
  <h3>Harbor Analysis</h3>
  <p class="subtle">Open the generated Harbor/Skill Evaluator engineering analysis for this uploaded job.</p>
  <a class="evidence-link harbor-analysis-link" href="{safe_url}" target="_blank" rel="noopener noreferrer" aria-label="Open Harbor Analysis">
    <span class="evidence-link-icon" aria-hidden="true">&#128279;</span><span>Open Harbor Analysis</span>
  </a>
</div>"""


def _linkify_urls(text: str) -> str:
    value = str(text)
    parts: list[str] = []
    cursor = 0
    for match in _URL_RE.finditer(value):
        parts.append(escape(value[cursor : match.start()]))
        url = match.group(0)
        parts.append(f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(url)}</a>')
        cursor = match.end()
    parts.append(escape(value[cursor:]))
    return "".join(parts)


def _build_js(
    radar_labels: str,
    radar_js: str,
    bar_labels: str,
    bar_js: str,
    token_labels: str,
    token_prompt: str,
    token_completion: str,
    token_cached: str,
    steps_labels: str,
    steps_datasets: str,
) -> str:
    """Build the JS block separately to avoid f-string brace escaping issues."""
    return (
        "function go(btn,id){document.querySelectorAll('.nav button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));document.getElementById(id).classList.add('active')}\n"
        "function switchAgent(btn,name){document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');document.querySelectorAll('.tab-panel').forEach(p=>p.style.display='none');document.getElementById('agent-'+name).style.display='block'}\n"
        "function getCS(){const s=getComputedStyle(document.documentElement);return{grid:s.getPropertyValue('--chart-grid').trim(),tick:s.getPropertyValue('--chart-tick').trim(),label:s.getPropertyValue('--chart-label').trim()}}\n"
        "let allCharts=[];\n"
        "function rebuildCharts(){allCharts.forEach(c=>c.destroy());allCharts=[];const cs=getCS();\n"
        f"const rCtx=document.getElementById('cRadar');if(rCtx)allCharts.push(new Chart(rCtx,{{type:'radar',data:{{labels:{radar_labels},datasets:[{radar_js}]}},options:{{responsive:true,scales:{{r:{{min:0,max:1,ticks:{{stepSize:.2,color:cs.tick,backdropColor:'transparent'}},grid:{{color:cs.grid}},angleLines:{{color:cs.grid}},pointLabels:{{color:cs.label,font:{{size:11}}}}}}}},plugins:{{legend:{{labels:{{color:cs.label}}}}}}}}}}));\n"
        f"const bCtx=document.getElementById('cBar');if(bCtx)allCharts.push(new Chart(bCtx,{{type:'bar',data:{{labels:{bar_labels},datasets:[{bar_js}]}},options:{{responsive:true,scales:{{y:{{min:0,max:1,ticks:{{stepSize:.2,color:cs.tick}},grid:{{color:cs.grid}}}},x:{{ticks:{{color:cs.label,maxRotation:45}},grid:{{display:false}}}}}},plugins:{{legend:{{labels:{{color:cs.label}}}}}}}}}}));\n"
        f"const tCtx=document.getElementById('cTokens');if(tCtx)allCharts.push(new Chart(tCtx,{{type:'bar',data:{{labels:{token_labels},datasets:[{{label:'Prompt',data:{token_prompt},backgroundColor:'#3b82f6cc',borderRadius:3}},{{label:'Completion',data:{token_completion},backgroundColor:'#22c55ecc',borderRadius:3}},{{label:'Cached',data:{token_cached},backgroundColor:'#8b5cf6cc',borderRadius:3}}]}},options:{{responsive:true,scales:{{x:{{stacked:true,ticks:{{color:cs.tick}},grid:{{color:cs.grid}}}},y:{{stacked:true,ticks:{{color:cs.label}},grid:{{display:false}}}}}},plugins:{{legend:{{labels:{{color:cs.label}}}}}}}}}}));\n"
        f"const sCtx=document.getElementById('cSteps');if(sCtx)allCharts.push(new Chart(sCtx,{{type:'bar',data:{{labels:{steps_labels},datasets:[{steps_datasets}]}},options:{{responsive:true,scales:{{x:{{ticks:{{color:cs.tick}},grid:{{color:cs.grid}}}},y:{{ticks:{{color:cs.label}},grid:{{display:false}}}}}},plugins:{{legend:{{labels:{{color:cs.label}}}}}}}}}}));\n"
        "}\n"
        "function toggleTheme(){const h=document.documentElement;const t=h.getAttribute('data-theme')==='dark'?'light':'dark';h.setAttribute('data-theme',t);localStorage.setItem('theme',t);rebuildCharts()}\n"
        "(function(){const saved=localStorage.getItem('theme');if(saved)document.documentElement.setAttribute('data-theme',saved);rebuildCharts()})();\n"
    )


def generate_html_report(
    skill_name: str,
    results_dir: Path,
    output_path: Path | None = None,
    skill_path: Path | None = None,
) -> Path:
    if output_path is None:
        output_path = results_dir / "report.html"

    agents = _load_agent_data(results_dir)
    if not agents:
        raise ValueError(f"No agent results found in {results_dir}")

    dataset_entries = _load_dataset(skill_path) or _load_staged_harbor_dataset(results_dir)
    attempt_policy: dict[str, Any] = {"max_attempts": 1, "pass_threshold": 0.50}
    policy_file = results_dir / "attempt_policy.json"
    if policy_file.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            attempt_policy.update(json.loads(policy_file.read_text(encoding="utf-8")))
    run_config: dict[str, Any] = {}
    run_config_file = results_dir / "run_config.json"
    if run_config_file.exists():
        try:
            loaded_config = json.loads(run_config_file.read_text(encoding="utf-8"))
            if isinstance(loaded_config, dict):
                run_config = loaded_config
        except (json.JSONDecodeError, OSError):
            pass
    max_attempts = int(attempt_policy.get("max_attempts", 1) or 1)
    pass_threshold_value = attempt_policy.get("pass_threshold", 0.50)
    pass_threshold = float(0.50 if pass_threshold_value is None else pass_threshold_value)
    show_attempt_policy = max_attempts > 1

    timestamp = results_dir.name
    try:
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")  # noqa: DTZ007 -- local run-dir name, no tz to attach
        formatted_time = dt.strftime("%b %d, %Y %H:%M")
    except ValueError:
        formatted_time = timestamp

    agent_names = sorted(agents.keys())
    has_lift = any("lift" in agents[a] and _both_conditions_succeeded(agents[a]) for a in agent_names)
    has_custom_lift = any("custom_lift" in agents[a] and _both_conditions_succeeded(agents[a]) for a in agent_names)
    display_metrics = _metrics_for_agents(agents)
    metric_count = len(display_metrics)
    custom_only = _grading_mode(run_config) == "custom_only"
    if metric_count:
        score_definition_text = (
            f"Overall score is the mean of the {metric_count} evaluator metrics shown in this report."
        )
        metrics_label = f"{metric_count} evaluator checks (deterministic + LLM judge)"
        heatmap_text = f"Overall score per eval case per agent (average of {metric_count} evaluator metrics)."
    elif custom_only:
        score_definition_text = "Overall score is the user-provided reward overall."
        metrics_label = "custom-only user reward overall"
        heatmap_text = "Overall score per eval case per agent (user-provided reward overall)."
    else:
        score_definition_text = "No scored evaluator metrics are available for this run."
        metrics_label = "no valid scored metrics"
        heatmap_text = "No per-case score data is available."

    # --- Data for JS ---
    radar_labels = json.dumps([_METRIC_DISPLAY.get(m, m) for m in display_metrics])
    radar_datasets = []
    for i, name in enumerate(agent_names):
        color = _agent_color(name, i)
        scores = agents[name].get("with_skill", {})
        with_skill_succeeded = _condition_status(agents[name], "with_skill") == "succeeded"
        data = []
        for metric in display_metrics:
            value = scores.get(metric) if isinstance(scores, dict) and with_skill_succeeded else None
            data.append(
                round(float(value), 4) if isinstance(value, int | float) and not isinstance(value, bool) else None
            )
        radar_datasets.append(
            f'{{label:{json.dumps(name)},data:{json.dumps(data)},borderColor:"{color}",backgroundColor:"{color}20",pointBackgroundColor:"{color}",borderWidth:2,pointRadius:4}}'
        )
    radar_js = ",".join(radar_datasets)

    bar_labels = json.dumps([_METRIC_DISPLAY.get(m, m) for m in display_metrics])
    bar_datasets = []
    for i, name in enumerate(agent_names):
        color = _agent_color(name, i)
        scores = agents[name].get("with_skill", {})
        with_skill_succeeded = _condition_status(agents[name], "with_skill") == "succeeded"
        data = []
        for metric in display_metrics:
            value = scores.get(metric) if isinstance(scores, dict) and with_skill_succeeded else None
            data.append(
                round(float(value), 4) if isinstance(value, int | float) and not isinstance(value, bool) else None
            )
        bar_datasets.append(
            f'{{label:{json.dumps(name)},data:{json.dumps(data)},backgroundColor:"{color}cc",borderRadius:4}}'
        )
    bar_js = ",".join(bar_datasets)

    # --- Overview tab ---
    overall_cards = ""
    for name in agent_names:
        avg = _agent_overall(agents[name], display_metrics)
        trials = agents[name].get("num_trials", 0)
        with_skill_status = _condition_status(agents[name], "with_skill")
        baseline_status = _condition_status(agents[name], "without_skill")
        expected_attempts = _nonnegative_counter(agents[name].get("expected_attempts"))
        scored_attempts = _nonnegative_counter(agents[name].get("scored_attempts"))
        color = _agent_color(name, agent_names.index(name))
        lift_str = ""
        if "lift" in agents[name] and _both_conditions_succeeded(agents[name]):
            ld = agents[name]["lift"].get("overall", {}).get("delta", 0.0)
            lcolor = "#22c55e" if ld > 0 else ("#ef4444" if ld < 0 else "#94a3b8")
            lsign = "+" if ld > 0 else ""
            lift_str = f'<div class="lift-chip" style="color:{lcolor}">{lsign}{ld:.2f} lift</div>'
        elif "custom_lift" in agents[name] and _both_conditions_succeeded(agents[name]):
            ld = agents[name]["custom_lift"].get("overall", {}).get("delta", 0.0)
            lcolor = "#22c55e" if ld > 0 else ("#ef4444" if ld < 0 else "#94a3b8")
            lsign = "+" if ld > 0 else ""
            lift_str = f'<div class="lift-chip" style="color:{lcolor}">{lsign}{ld:.2f} custom lift</div>'
        if with_skill_status != "succeeded":
            coverage = (
                f"<br>{scored_attempts} of {expected_attempts} expected attempts scored" if expected_attempts else ""
            )
            overall_cards += f"""<div class="agent-card" style="border-top:3px solid #ef4444;opacity:.6">
  <div class="agent-name" style="color:{color}">{escape(name)}</div>
  <div class="agent-score" style="color:#ef4444">NO SCORE</div>
  <div class="subtle">No valid with-skill score{coverage}</div>
</div>"""
        elif trials == 0:
            overall_cards += f"""<div class="agent-card" style="border-top:3px solid #ef4444;opacity:.6">
  <div class="agent-name" style="color:{color}">{escape(name)}</div>
  <div class="agent-score" style="color:#ef4444">FAIL</div>
  <div class="subtle">0 trials — agent did not produce results</div>
</div>"""
        else:
            baseline_failure_note = (
                "<br>Baseline execution failed; lift unavailable" if baseline_status in {"failed", "unknown"} else ""
            )
            overall_cards += f"""<div class="agent-card" style="border-top:3px solid {color}">
  <div class="agent-name" style="color:{color}">{escape(name)}</div>
  <div class="agent-score" style="color:{_sc(avg)}">{avg:.2f}</div>
  <div class="subtle">{trials} trial(s){baseline_failure_note}</div>{lift_str}
</div>"""

    run_config_html = ""
    if run_config:
        harbor_cfg = run_config.get("harbor", {})

        def _is_default_source(source: Any) -> bool:
            source_text = str(source or "")
            return source_text in {"", "SkillEvaluator default", "auto", "none"} or source_text.startswith("auto (")

        def _cfg_cell(key: str, label: str, formatter=str) -> str:
            item = harbor_cfg.get(key, {})
            if not isinstance(item, dict):
                return ""
            value = item.get("value")
            source = item.get("source", "")
            if _is_default_source(source):
                return ""
            if value is None:
                value_text = "auto" if key == "timeout_multiplier" else "all" if key == "max_agents" else "none"
            elif isinstance(value, bool):
                value_text = "enabled" if value else "disabled"
            else:
                value_text = formatter(value)
            return f"<div><span>{escape(label)}</span><b>{escape(value_text)}</b><em>{escape(str(source))}</em></div>"

        env_mode_text = ""
        env_mode = harbor_cfg.get("environment") or harbor_cfg.get("env_mode", {})
        if isinstance(env_mode, dict) and env_mode.get("value"):
            env_mode_text = f"<div><span>Environment</span><b>{escape(str(env_mode['value']))}</b></div>"

        task_source_text = ""
        task_source = str(run_config.get("task_source", "evals_json"))
        task_source_source = str(run_config.get("task_source_source", ""))
        if task_source != "evals_json" or not _is_default_source(task_source_source):
            task_label = (
                "evals.json → Harbor single-step tasks"
                if task_source == "evals_json"
                else "evals/harbor → native Harbor tasks"
            )
            task_source_text = (
                f"<div><span>Task source</span><b>{escape(task_label)}</b><em>{escape(task_source_source)}</em></div>"
            )

        workspace_cfg = run_config.get("skill_workspace", {})
        workspace_text = ""
        if isinstance(workspace_cfg, dict):
            mode = workspace_cfg.get("mode", {})
            include = workspace_cfg.get("include", {})
            staged = workspace_cfg.get("staged_skills", [])
            staged_text = ", ".join(str(s) for s in staged) if isinstance(staged, list) and staged else "none"
            if isinstance(mode, dict) and not _is_default_source(mode.get("source", "")):
                workspace_text += (
                    f"<div><span>Workspace mode</span><b>{escape(str(mode.get('value', 'isolated')))}</b>"
                    f"<em>{escape(str(mode.get('source', '')))}</em></div>"
                )
            if isinstance(staged, list) and staged:
                workspace_text += (
                    f"<div><span>Included skills</span><b>{escape(staged_text)}</b>"
                    f"<em>{escape(str(include.get('source', '')) if isinstance(include, dict) else '')}</em></div>"
                )

        grading_cfg = run_config.get("grading", {})
        grading_text = ""
        if (
            isinstance(grading_cfg, dict)
            and isinstance(grading_cfg.get("mode"), dict)
            and not _is_default_source(grading_cfg["mode"].get("source", ""))
        ):
            mode = grading_cfg["mode"]
            grading_text = (
                f"<div><span>Grading mode</span><b>{escape(str(mode.get('value', 'default')))}</b>"
                f"<em>{escape(str(mode.get('source', '')))}</em></div>"
            )

        agent_rows = ""
        for agent, meta in run_config.get("agents", {}).items():
            if not isinstance(meta, dict):
                continue
            if _is_default_source(meta.get("source", "")):
                continue
            agent_rows += (
                f"<tr><td>{escape(str(agent))}</td><td>{escape(str(meta.get('model', '')))}</td>"
                f"<td>{escape(str(meta.get('source', '')))}</td></tr>"
            )
        agent_table = ""
        if agent_rows:
            agent_table = (
                '<div class="table-wrap compact"><table><thead><tr><th>Agent</th><th>Model</th>'
                f"<th>Source</th></tr></thead><tbody>{agent_rows}</tbody></table></div>"
            )

        config_file = str(run_config.get("config_file", "none") or "none")
        config_file_text = ""
        if config_file != "none":
            config_file_text = f"<div><span>Config file</span><b>{escape(config_file)}</b></div>"

        policy_cells = "".join(
            [
                config_file_text,
                _cfg_cell("n_attempts", "Attempts"),
                _cfg_cell("pass_threshold", "Pass threshold", lambda value: f"{float(value):.2f}"),
                _cfg_cell("n_concurrent", "Concurrency"),
                _cfg_cell("max_agents", "Max agents"),
                _cfg_cell("timeout_multiplier", "Timeout multiplier", lambda value: f"{float(value):.2f}x"),
                _cfg_cell("custom_dockerfile_mode", "Custom Dockerfile"),
                _cfg_cell("copy_repo", "Repo context"),
                env_mode_text,
                task_source_text,
                grading_text,
                workspace_text,
            ]
        )
        notes = " ".join(str(n) for n in run_config.get("notes", []) if n and "slash commands" not in str(n))
        notes_html = f'<p class="subtle">{escape(notes)}</p>' if notes else ""
        if policy_cells or agent_table or notes_html:
            run_config_html = f"""<div class="info-box run-config">
  <h3>Harbor Run Configuration</h3>
  <div class="policy-grid">
    {policy_cells}
  </div>
  {agent_table}
  {notes_html}
</div>"""

    attempt_policy_html = ""
    pass_summary_html = ""
    if show_attempt_policy:
        attempt_policy_html = f"""<div class="info-box attempt-policy">
  <h3>Attempt Policy</h3>
  <div class="policy-grid">
    <div><span>Max attempts per eval case</span><b>{max_attempts}</b></div>
    <div><span>Pass threshold</span><b>overall score &ge; {pass_threshold:.2f}</b></div>
  </div>
  <p class="subtle">{escape(score_definition_text)} Harbor runs every configured attempt for every eval case.</p>
</div>"""

        def _summary_counts(summary: dict[str, Any]) -> tuple[int, int, int]:
            scored = int(summary.get("attempts_used", 0) or 0)
            max_possible = int(summary.get("max_attempts_possible", 0) or 0)
            total_missing = max(0, max_possible - scored)
            return scored, max_possible, total_missing

        def _summary_row(
            agent_name: str,
            condition: str,
            summary: dict[str, Any],
            lift_cell: str = "",
            *,
            valid: bool,
        ) -> str:
            scored, max_possible, not_scored = _summary_counts(summary)
            if not valid:
                return (
                    f"<tr><td>{escape(agent_name)}</td><td>{escape(condition)}</td>"
                    '<td class="subtle">NO SCORE</td><td>--</td>'
                    f"<td>{scored} of {max_possible} scored</td><td>--</td>"
                    f'<td class="subtle">{not_scored} not scored</td>{lift_cell}</tr>'
                )
            rate = float(summary.get("rate", 0.0) or 0.0)
            passed = int(summary.get("passed_cases", 0) or 0)
            total = int(summary.get("total_cases", 0) or 0)
            avg_scored = scored / total if total else 0.0
            unscored_text = f"{not_scored} not scored"
            return (
                f"<tr><td>{escape(agent_name)}</td><td>{escape(condition)}</td>"
                f"<td><b>{rate:.0%}</b></td>"
                f"<td>{passed} of {total} cases</td>"
                f"<td>{scored} of {max_possible} scored</td>"
                f"<td>{avg_scored:.2f} scored/case</td>"
                f'<td class="subtle">{escape(unscored_text)}</td>{lift_cell}</tr>'
            )

        pass_rows = ""
        for name in agent_names:
            with_pass = agents[name].get("pass_with_skill", {})
            without_pass = agents[name].get("pass_without_skill", {})
            pass_lift = agents[name].get("pass_lift", {})
            with_status = _condition_status(agents[name], "with_skill")
            without_status = _condition_status(agents[name], "without_skill")
            valid_lift = _both_conditions_succeeded(agents[name])

            lift_cell = ""
            delta_value = pass_lift.get("delta") if isinstance(pass_lift, dict) else None
            if (
                has_lift
                and without_pass
                and valid_lift
                and isinstance(delta_value, int | float)
                and not isinstance(delta_value, bool)
            ):
                delta = float(delta_value)
                dc = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#94a3b8")
                ds = f"+{delta:.0%}" if delta > 0 else f"{delta:.0%}"
                lift_cell = f'<td style="color:{dc};font-weight:700">{ds}</td>'
            elif has_lift:
                lift_cell = "<td>--</td>"
            pass_rows += _summary_row(
                name,
                "With skill",
                with_pass,
                lift_cell,
                valid=with_status == "succeeded" and bool(with_pass),
            )

            if without_pass:
                baseline_lift = "<td>baseline</td>" if has_lift and valid_lift else ("<td>--</td>" if has_lift else "")
                pass_rows += _summary_row(
                    name,
                    "Without skill",
                    without_pass,
                    baseline_lift,
                    valid=without_status == "succeeded",
                )

        lift_col = "<th>Lift</th>" if has_lift else ""
        pass_summary_html = f"""<h3>Pass@{max_attempts} Summary</h3>
<p class="subtle">Pass@{max_attempts} is case-level: a case passes when any scored attempt reaches overall score &ge; {pass_threshold:.2f}. Attempt coverage shows scored rewards out of the maximum possible attempts; unscored attempts are not counted as pass or fail.</p>
<div class="table-wrap"><table><thead><tr><th>Agent</th><th>Condition</th><th>Pass@{max_attempts}</th><th>Cases Passed</th><th>Scored Attempts</th><th>Avg Scored / Case</th><th>Not Scored</th>{lift_col}</tr></thead><tbody>{pass_rows}</tbody></table></div>"""

    # --- Score comparison table ---
    score_table_head = "".join(f"<th>{escape(n)}</th>" for n in agent_names)
    score_table_rows = ""
    for metric in display_metrics:
        display = _METRIC_DISPLAY.get(metric, metric)
        cells = ""
        for name in agent_names:
            w = agents[name].get("with_skill", {}).get(metric)
            if (
                _condition_status(agents[name], "with_skill") != "succeeded"
                or not isinstance(w, int | float)
                or isinstance(w, bool)
            ):
                cells += '<td class="subtle">NO SCORE</td>'
                continue
            pct = int(w * 100)
            cells += f'<td><div class="bar-cell"><span class="val" style="color:{_sc(w)}">{w:.2f}</span><div class="mini-bar"><div style="width:{pct}%;background:{_sc(w)}"></div></div></div></td>'
        score_table_rows += f'<tr><td class="metric-name">{escape(display)}<span class="metric-hint">{escape(_METRIC_DESC.get(metric, ""))}</span></td>{cells}</tr>'

    dimension_html = ""
    if "security" in display_metrics:
        dimension_names = ["security", "correctness", "discoverability", "effectiveness", "efficiency"]
        dimension_rows = ""
        show_dimension_lift = any(
            _both_conditions_succeeded(agents[name])
            and isinstance(agents[name].get("dimensions_without_skill", {}).get(dimension, {}), dict)
            and "score" in agents[name].get("dimensions_without_skill", {}).get(dimension, {})
            for name in agent_names
            for dimension in dimension_names
        )
        for dimension in dimension_names:
            cells = ""
            any_score = False
            for name in agent_names:
                dim = agents[name].get("dimensions_with_skill", {}).get(dimension, {})
                score = dim.get("score") if isinstance(dim, dict) else None
                if _condition_status(agents[name], "with_skill") == "succeeded" and isinstance(score, int | float):
                    any_score = True
                    if show_dimension_lift:
                        baseline_dim = agents[name].get("dimensions_without_skill", {}).get(dimension, {})
                        baseline_score = baseline_dim.get("score") if isinstance(baseline_dim, dict) else None
                        if isinstance(baseline_score, int | float):
                            delta = float(score) - float(baseline_score)
                            dc = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#94a3b8")
                            ds = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
                            cells += f'<td><span class="val" style="color:{dc}">{ds}</span></td>'
                        else:
                            cells += '<td class="subtle">--</td>'
                    else:
                        cells += (
                            f'<td><span class="val" style="color:{_sc(float(score))}">{float(score):.2f}</span></td>'
                        )
                else:
                    cells += '<td class="subtle">--</td>'
            if any_score:
                source_label = _dimension_source_label(dimension)
                dimension_rows += (
                    f'<tr><td class="metric-name">{escape(dimension.title())}'
                    f'<span class="metric-hint">Group: {escape(source_label)}</span></td>{cells}</tr>'
                )
        if dimension_rows:
            if show_dimension_lift:
                dimension_caption = (
                    "Dimensions are report-only metric groups. Agent cells show dimension lift "
                    "(with skill minus without skill). Custom metrics do not change these scores."
                )
            else:
                dimension_caption = (
                    "Dimensions are report-only metric groups built from default evaluator metrics. "
                    "Custom metrics do not change these scores."
                )
            dimension_html = (
                "<h3>Default Dimension Summary</h3>"
                f'<p class="subtle">{escape(dimension_caption)}</p>'
                f'<div class="table-wrap"><table><thead><tr><th>Dimension</th>{score_table_head}</tr></thead><tbody>{dimension_rows}</tbody></table></div>'
            )

    custom_metric_names = sorted({metric for info in agents.values() for metric in info.get("custom_with_skill", {})})
    custom_html = ""
    if custom_metric_names:
        custom_rows = ""
        for metric in custom_metric_names:
            cells = ""
            for name in agent_names:
                value = agents[name].get("custom_with_skill", {}).get(metric)
                if _condition_status(agents[name], "with_skill") == "succeeded" and isinstance(value, int | float):
                    cells += f'<td><span class="val" style="color:{_sc(float(value))}">{float(value):.2f}</span></td>'
                else:
                    cells += '<td class="subtle">--</td>'
            custom_rows += f'<tr><td class="metric-name">{escape(metric)}</td>{cells}</tr>'
        custom_html = (
            "<h3>Custom Metrics</h3>"
            '<p class="subtle">Custom metrics are reported separately and do not change default evaluator dimensions.</p>'
            f'<div class="table-wrap"><table><thead><tr><th>Custom Metric</th>{score_table_head}</tr></thead><tbody>{custom_rows}</tbody></table></div>'
        )

    custom_lift_html = ""
    if has_custom_lift:
        custom_lift_metric_names = sorted(
            {metric for info in agents.values() for metric in info.get("custom_lift", {}) if metric != "overall"}
        )
        custom_lift_rows = ""
        if any("overall" in agents[name].get("custom_lift", {}) for name in agent_names):
            custom_lift_metric_names = ["overall", *custom_lift_metric_names]
        for metric in custom_lift_metric_names:
            label = "Overall Reward" if metric == "overall" else metric
            hint = (
                "User-provided overall score used for custom_only pass/lift."
                if metric == "overall"
                else "User-owned custom metric; does not affect default evaluator dimensions."
            )
            cells = ""
            any_metric_score = False
            for name in agent_names:
                ld = agents[name].get("custom_lift", {}).get(metric)
                values = (
                    [ld.get(key) for key in ("with_skill", "without_skill", "delta")] if isinstance(ld, dict) else []
                )
                if _both_conditions_succeeded(agents[name]) and all(
                    isinstance(value, int | float) and not isinstance(value, bool) for value in values
                ):
                    any_metric_score = True
                    w, wo, delta = (float(value) for value in values)
                    dc = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#94a3b8")
                    ds = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
                    cells += f'<td><span class="subtle">{wo:.2f}</span> &rarr; {w:.2f} <span style="color:{dc};font-weight:600">({ds})</span></td>'
                else:
                    cells += '<td class="subtle">--</td>'
            if any_metric_score:
                custom_lift_rows += f'<tr><td class="metric-name">{escape(label)}<span class="metric-hint">{escape(hint)}</span></td>{cells}</tr>'
        if custom_lift_rows:
            custom_lift_html = (
                "<h3>Custom Reward Lift (with skill &minus; without)</h3>"
                '<p class="subtle">Lift for BYOT/custom metrics is computed separately from default evaluator lift. '
                "Baseline is the same task run without the target skill staged.</p>"
                f'<div class="table-wrap"><table><thead><tr><th>Custom Metric</th>{score_table_head}</tr></thead><tbody>{custom_lift_rows}</tbody></table></div>'
            )
    elif custom_metric_names:
        custom_lift_html = (
            '<div class="info-box">'
            "<h3>Custom Reward Lift</h3>"
            '<p class="subtle">Custom lift is not available for this run because no without-skill baseline rewards were found. '
            "Run BYOT/custom grading without <code>--skip-baseline</code> to compare custom metrics with and without the target skill.</p>"
            "</div>"
        )

    default_score_html = ""
    default_chart_html = ""
    if display_metrics:
        default_chart_html = """<div class="chart-row">
    <div class="chart-box"><h3>Score Comparison (Radar)</h3><canvas id="cRadar"></canvas></div>
    <div class="chart-box"><h3>Scores by Agent &amp; Metric</h3><canvas id="cBar"></canvas></div>
  </div>"""
        default_score_html = (
            "<h3>Scores by Metric</h3>"
            f'<div class="table-wrap"><table><thead><tr><th>Metric</th>{score_table_head}</tr></thead><tbody>{score_table_rows}</tbody></table></div>'
        )

    custom_only_note_html = ""
    if not display_metrics and custom_only:
        custom_only_note_html = (
            '<div class="info-box">'
            "<h3>Custom Reward Mode</h3>"
            '<p class="subtle">This report is from BYOT <code>custom_only</code> grading. '
            "Default evaluator metric charts and dimensions are hidden because the user grader owns the scoring contract. "
            "Pass@k uses the user-provided overall reward.</p>"
            "</div>"
        )
    elif not display_metrics:
        custom_only_note_html = (
            '<div class="info-box">'
            "<h3>No Scored Metrics</h3>"
            '<p class="subtle">No scored evaluator metrics are available. Review Failure Details and rerun '
            "the evaluation before interpreting quality.</p>"
            "</div>"
        )

    # --- Lift table ---
    lift_html = ""
    if has_lift and display_metrics:
        lift_rows = ""
        for metric in display_metrics:
            display = _METRIC_DISPLAY.get(metric, metric)
            cells = ""
            for name in agent_names:
                ld = agents[name].get("lift", {}).get(metric, {})
                values = (
                    [ld.get(key) for key in ("with_skill", "without_skill", "delta")] if isinstance(ld, dict) else []
                )
                if not _both_conditions_succeeded(agents[name]) or not all(
                    isinstance(value, int | float) and not isinstance(value, bool) for value in values
                ):
                    cells += '<td class="subtle">--</td>'
                    continue
                w, wo, delta = (float(value) for value in values)
                dc = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#94a3b8")
                ds = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
                cells += f'<td><span class="subtle">{wo:.2f}</span> &rarr; {w:.2f} <span style="color:{dc};font-weight:600">({ds})</span></td>'
            lift_rows += f'<tr><td class="metric-name">{escape(display)}</td>{cells}</tr>'
        lift_html = f"""<h3>Skill Lift (with skill &minus; without)</h3>
<div class="table-wrap"><table><thead><tr><th>Metric</th>{score_table_head}</tr></thead><tbody>{lift_rows}</tbody></table></div>"""

    # --- Agent tabs ---
    agent_tab_buttons = ""
    agent_tab_panels = ""
    for i, name in enumerate(agent_names):
        color = _agent_color(name, i)
        active = " active" if i == 0 else ""
        agent_tab_buttons += f'<button class="tab-btn{active}" onclick=\'switchAgent(this,{json.dumps(name)})\' style="--accent:{color}">{escape(name)}</button>'
        failure_details_html = _build_execution_failure_details(agents[name])
        findings_html = _build_agent_findings(agents[name])
        display = "block" if i == 0 else "none"
        agent_tab_panels += f'<div class="tab-panel" id="agent-{name}" style="display:{display}">{failure_details_html}{findings_html}</div>'

    suggestions_html = _build_suggestions_html(skill_name, agents)
    harbor_analysis_html = _build_harbor_analysis_html(agents)
    dataset_html = _build_dataset_html(dataset_entries)

    # --- Trials tab data ---
    # Heatmap: eval case (rows) x agent (cols), colored by score
    all_entry_ids: list[str] = []
    seen_ids: set[str] = set()
    for name in agent_names:
        for r in agents[name].get("rewards", []):
            eid = r.get("entry_id", "?")
            if eid not in seen_ids:
                seen_ids.add(eid)
                all_entry_ids.append(eid)

    heatmap_rows = ""
    for eid in all_entry_ids:
        cells = ""
        for name in agent_names:
            match = [r for r in agents[name].get("rewards", []) if r.get("entry_id") == eid]
            if match:
                scores = [_reward_overall(r, display_metrics) for r in match]
                avg = sum(scores) / len(scores)
                cells += f'<td style="background:{_sc(avg)}22"><span class="val" style="color:{_sc(avg)}">{avg:.2f}</span></td>'
            else:
                cells += '<td class="subtle">--</td>'
        heatmap_rows += f'<tr><td class="metric-name" style="font-family:monospace">{escape(eid)}</td>{cells}</tr>'

    # Token usage per agent (stacked bar data)
    token_labels: list[str] = []
    token_prompt: list[int] = []
    token_completion: list[int] = []
    token_cached: list[int] = []
    for name in agent_names:
        rewards = agents[name].get("rewards", [])
        tp = sum(r.get("_traj", {}).get("prompt_tokens", 0) for r in rewards)
        tc = sum(r.get("_traj", {}).get("completion_tokens", 0) for r in rewards)
        tca = sum(r.get("_traj", {}).get("cached_tokens", 0) for r in rewards)
        token_labels.append(name)
        token_prompt.append(tp)
        token_completion.append(tc)
        token_cached.append(tca)
    token_labels_js = json.dumps(token_labels)
    token_prompt_js = json.dumps(token_prompt)
    token_completion_js = json.dumps(token_completion)
    token_cached_js = json.dumps(token_cached)

    # Steps per trial (grouped bar)
    steps_data: dict[str, list[int]] = {name: [] for name in agent_names}
    steps_labels: list[str] = []
    for eid in all_entry_ids:
        steps_labels.append(eid)
        for name in agent_names:
            match = [r for r in agents[name].get("rewards", []) if r.get("entry_id") == eid]
            steps_data[name].append(match[0].get("_traj", {}).get("steps", 0) if match else 0)
    steps_labels_js = json.dumps(steps_labels)
    steps_datasets = []
    for i, name in enumerate(agent_names):
        color = _agent_color(name, i)
        steps_datasets.append(
            f'{{label:{json.dumps(name)},data:{json.dumps(steps_data[name])},backgroundColor:"{color}cc",borderRadius:3}}'
        )
    steps_datasets_js = ",".join(steps_datasets)

    # Error recovery summary
    er_rows = ""
    has_er_data = False
    for name in agent_names:
        for r in agents[name].get("rewards", []):
            eid = r.get("entry_id", "?")
            er = r.get("details", {}).get("skill_execution", {}).get("error_recovery", {})
            corrections = er.get("corrections", [])
            if not corrections and er.get("first_attempt_clean", True):
                continue
            has_er_data = True
            sf = er.get("skill_faults", 0)
            af = er.get("agent_faults", 0)
            clean = er.get("first_attempt_clean", True)
            clean_badge = (
                '<span class="badge pass">clean</span>' if clean else '<span class="badge fail">retries needed</span>'
            )
            fault_cells = ""
            if sf > 0:
                fault_cells += f'<span class="badge fail">{sf} skill</span> '
            if af > 0:
                fault_cells += f'<span class="badge warn">{af} agent</span> '
            if not fault_cells:
                fault_cells = '<span class="subtle">none</span>'
            detail_items = ""
            for corr in corrections[:3]:
                f_type = corr.get("fault", "?")
                bcls = "fail" if f_type == "skill" else "warn"
                detail_items += f'<div class="subtle" style="margin:2px 0"><span class="badge {bcls}">{f_type}</span> {escape(corr.get("error", "")[:100])}</div>'
            er_rows += f'<tr><td style="font-family:monospace">{escape(name)}</td><td style="font-family:monospace">{escape(eid)}</td><td>{clean_badge}</td><td>{fault_cells}</td><td>{detail_items or "<span class=subtle>--</span>"}</td></tr>'

    er_html = ""
    if has_er_data:
        er_html = f"""<h3>Error Recovery</h3><p class="subtle">Trials where commands failed and were retried, with fault attribution.</p>
<div class="table-wrap"><table><thead><tr><th>Agent</th><th>Case</th><th>First Attempt</th><th>Faults</th><th>Details</th></tr></thead><tbody>{er_rows}</tbody></table></div>"""
    elif all(agents[name].get("execution_status") == "succeeded" for name in agent_names):
        er_html = '<h3>Error Recovery</h3><p class="subtle">All commands succeeded on first attempt across all trials. No error-retry patterns detected.</p>'
    else:
        er_html = (
            '<h3>Error Recovery</h3><p class="subtle">Error-recovery analysis is incomplete because execution '
            "did not succeed for every agent. Review Failure Details.</p>"
        )

    attempt_detail_html = ""
    if show_attempt_policy:
        missing_explainer = (
            "Not scored means no reward was found for that expected attempt; it is excluded from pass/fail."
        )

        def _attempt_counts(case_data: dict[str, Any]) -> tuple[int, int]:
            attempts_used = int(case_data.get("attempts_used", len(case_data.get("attempts", []))) or 0)
            missing = int(case_data.get("attempts_missing", max(0, max_attempts - attempts_used)) or 0)
            return attempts_used, missing

        def _attempt_summary(case_data: dict[str, Any]) -> str:
            attempts = case_data.get("attempts", [])
            parts: list[str] = []
            for attempt in attempts:
                idx = attempt.get("attempt", "?")
                score = float(attempt.get("score", 0.0) or 0.0)
                passed = bool(attempt.get("passed", False))
                cls = "pass" if passed else ("warn" if score >= 0.4 else "fail")
                label = "pass" if passed else "fail"
                parts.append(f'<span class="attempt-chip {cls}">A{idx}: {score:.2f} {label}</span>')

            _, missing = _attempt_counts(case_data)
            if missing:
                start = len(attempts) + 1
                end = max_attempts
                attempt_range = f"A{start}" if start == end else f"A{start}-A{end}"
                parts.append(f'<span class="attempt-chip muted">{attempt_range}: not scored</span>')
            return "<div class='attempt-list'>" + "".join(parts) + "</div>"

        def _attempt_count_text(count: int, word: str) -> str:
            if count <= 0:
                return "0"
            noun = "attempt" if count == 1 else "attempts"
            return f"{count} {noun} {word}"

        def _attempt_detail_cells(case_data: dict[str, Any]) -> str:
            _, missing = _attempt_counts(case_data)
            return f'<td class="subtle">{escape(_attempt_count_text(missing, "not scored"))}</td>'

        def _result_text(case_data: dict[str, Any]) -> tuple[str, str]:
            first = case_data.get("first_pass_attempt")
            attempts_used, missing = _attempt_counts(case_data)

            if first:
                result_text = f"passed on scored attempt {first}"
                badge_cls = "pass"
            else:
                result_text = f"failed across {attempts_used} scored attempt(s)"
                badge_cls = "fail"

            if missing:
                result_text += f"; {missing} not scored"
            return result_text, badge_cls

        attempt_rows = ""
        for name in agent_names:
            for condition, key in (("With skill", "pass_with_skill"), ("Without skill", "pass_without_skill")):
                summary = agents[name].get(key, {})
                cases = summary.get("cases", {})
                if not cases:
                    continue
                for eid, case_data in sorted(cases.items()):
                    result_text, badge_cls = _result_text(case_data)
                    attempt_rows += (
                        f"<tr><td>{escape(name)}</td><td>{escape(condition)}</td>"
                        f'<td style="font-family:monospace">{escape(str(eid))}</td>'
                        f"<td>{_attempt_summary(case_data)}</td>"
                        f"<td>{_attempt_counts(case_data)[0]} of {max_attempts} scored</td>"
                        f"{_attempt_detail_cells(case_data)}"
                        f'<td><span class="badge {badge_cls}">{escape(result_text)}</span></td></tr>'
                    )

        attempt_detail_html = f"""<h3>Attempt Details</h3>
<p class="subtle">Each row is one eval case. Attempt Scores lists the scored attempts only; a score at or above {pass_threshold:.2f} counts as passed for Pass@{max_attempts}. {missing_explainer}</p>
<div class="table-wrap"><table><thead><tr><th>Agent</th><th>Condition</th><th>Eval Case</th><th>Attempt Scores</th><th>Scored Attempts</th><th>Not Scored</th><th>Result</th></tr></thead><tbody>{attempt_rows}</tbody></table></div>"""

    # With vs without per trial
    lift_per_trial_rows = ""
    has_baseline_trials = any(agents[a].get("rewards_baseline") for a in agent_names)
    if has_baseline_trials and display_metrics:
        for name in agent_names:
            with_rewards = {r.get("entry_id"): r for r in agents[name].get("rewards", [])}
            without_rewards = {r.get("entry_id"): r for r in agents[name].get("rewards_baseline", [])}
            for eid in all_entry_ids:
                wr = with_rewards.get(eid)
                wor = without_rewards.get(eid)
                if not wr:
                    continue
                cells = f'<td style="font-family:monospace">{escape(name)}</td><td style="font-family:monospace">{escape(eid)}</td>'
                for m in display_metrics:
                    w_val = wr.get(m, 0.0) if wr else 0.0
                    wo_val = wor.get(m, 0.0) if wor else 0.0
                    delta = w_val - wo_val
                    dc = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#94a3b8")
                    ds = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
                    cells += f'<td><span class="subtle">{wo_val:.2f}</span>&rarr;{w_val:.2f} <span style="color:{dc};font-weight:600;font-size:.82em">({ds})</span></td>'
                lift_per_trial_rows += f"<tr>{cells}</tr>"

    custom_lift_per_trial_rows = ""
    custom_trial_metrics = list(custom_metric_names)
    if has_baseline_trials and (custom_trial_metrics or (not display_metrics and has_custom_lift)):
        if not display_metrics and has_custom_lift:
            custom_trial_metrics = ["overall", *custom_trial_metrics]
        for name in agent_names:
            with_rewards = {r.get("entry_id"): r for r in agents[name].get("rewards", [])}
            without_rewards = {r.get("entry_id"): r for r in agents[name].get("rewards_baseline", [])}
            for eid in all_entry_ids:
                wr = with_rewards.get(eid)
                wor = without_rewards.get(eid)
                if not wr:
                    continue
                cells = f'<td style="font-family:monospace">{escape(name)}</td><td style="font-family:monospace">{escape(eid)}</td>'
                wr_custom = extract_custom_metrics(wr)
                wor_custom = extract_custom_metrics(wor or {})
                for metric in custom_trial_metrics:
                    if metric == "overall":
                        w_val = _reward_overall(wr, display_metrics)
                        wo_val = _reward_overall(wor, display_metrics) if wor else 0.0
                    else:
                        w_val = wr_custom.get(metric, 0.0)
                        wo_val = wor_custom.get(metric, 0.0)
                    delta = w_val - wo_val
                    dc = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#94a3b8")
                    ds = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
                    cells += f'<td><span class="subtle">{wo_val:.2f}</span>&rarr;{w_val:.2f} <span style="color:{dc};font-weight:600;font-size:.82em">({ds})</span></td>'
                custom_lift_per_trial_rows += f"<tr>{cells}</tr>"

    # Compute best agent overall for hero
    best_agent = ""
    best_score: float | None = None
    for name in agent_names:
        if _condition_status(agents[name], "with_skill") != "succeeded" or agents[name].get("num_trials", 0) == 0:
            continue
        avg = _agent_overall(agents[name], display_metrics)
        if best_score is None or avg > best_score:
            best_score = avg
            best_agent = name
    best_agent_label = _agent_model_label(best_agent, run_config, agents) if best_agent else ""
    hero_score = f"{best_score:.2f}" if best_score is not None else "N/A"
    hero_color = _sc(best_score) if best_score is not None else "var(--tx2)"
    hero_agent = escape(best_agent_label) if best_agent_label else "No successfully scored agent"

    html = (
        f"""<!DOCTYPE html>
<html lang="en" data-theme="dark"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(skill_name)} — Eval Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{--r:10px;--nv:#76b900}}
[data-theme="dark"]{{--bg:#0c111b;--s1:#151d2e;--s2:#1e293b;--s3:#2a3650;--tx:#e2e8f0;--tx2:#94a3b8;--bdr:#334155;--chart-grid:#1e293b;--chart-tick:#94a3b8;--chart-label:#e2e8f0;--badge-pass-bg:#14532d;--badge-pass-tx:#bbf7d0;--badge-warn-bg:#713f12;--badge-warn-tx:#fef08a;--badge-fail-bg:#7f1d1d;--badge-fail-tx:#fecaca;--shadow:0 1px 3px #0005}}
[data-theme="light"]{{--bg:#f8fafc;--s1:#ffffff;--s2:#f1f5f9;--s3:#e2e8f0;--tx:#0f172a;--tx2:#64748b;--bdr:#cbd5e1;--chart-grid:#e2e8f0;--chart-tick:#64748b;--chart-label:#0f172a;--badge-pass-bg:#dcfce7;--badge-pass-tx:#166534;--badge-warn-bg:#fef9c3;--badge-warn-tx:#854d0e;--badge-fail-bg:#fee2e2;--badge-fail-tx:#991b1b;--shadow:0 1px 4px #0001}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--tx);line-height:1.6;transition:background .2s,color .2s}}
.wrap{{max-width:1140px;margin:0 auto;padding:20px 24px}}

/* Header + Hero */
.header{{padding:28px 0 20px;border-bottom:1px solid var(--bdr);margin-bottom:28px}}
.header-top{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}}
.header h1{{font-size:1.7em;color:var(--nv)}}
.header .meta{{color:var(--tx2);font-size:.85em;margin-top:6px}}
.hero{{display:flex;gap:20px;align-items:center;margin-top:16px;flex-wrap:wrap}}
.hero-score{{font-size:3em;font-weight:800;line-height:1}}
.hero-detail{{font-size:.85em;color:var(--tx2)}}
.hero-detail b{{color:var(--tx)}}
.subtle{{color:var(--tx2);font-size:.82em}}

/* Theme toggle */
.theme-toggle{{background:var(--s2);border:1px solid var(--bdr);border-radius:8px;padding:6px 14px;cursor:pointer;color:var(--tx);font-size:.85em;transition:.15s}}
.theme-toggle:hover{{background:var(--s3)}}

/* Nav */
.nav{{display:flex;gap:2px;background:var(--s1);border-radius:var(--r);padding:4px;margin-bottom:24px;overflow-x:auto}}
.nav button{{background:none;border:none;color:var(--tx2);padding:10px 20px;font-size:.9em;cursor:pointer;border-radius:8px;white-space:nowrap;transition:.15s}}
.nav button:hover{{background:var(--s2);color:var(--tx)}}
.nav button.active{{background:var(--s2);color:var(--tx);font-weight:600;box-shadow:0 1px 3px #0005}}
.page{{display:none}}.page.active{{display:block}}

/* Agent overview cards */
.agent-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:20px 0}}
.agent-card{{background:var(--s1);border-radius:var(--r);padding:18px;text-align:center}}
.agent-name{{font-weight:600;font-size:.95em;margin-bottom:4px}}
.agent-score{{font-size:2em;font-weight:700}}
.lift-chip{{font-size:.85em;font-weight:600;margin-top:4px}}
.info-box{{background:var(--s1);border:1px solid var(--bdr);border-radius:var(--r);padding:18px;margin:18px 0;box-shadow:var(--shadow)}}
.info-box h3{{font-size:1em;margin-bottom:10px;color:var(--tx)}}
.attempt-policy{{border-left:4px solid #8b5cf6}}
.run-config{{border-left:4px solid #38bdf8}}
.harbor-analysis{{border-left:4px solid var(--nv)}}
.failure-details{{border-left:4px solid #ef4444}}
.failure-details ul{{padding-left:20px}}
.policy-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:10px}}
.policy-grid div{{background:var(--s2);border:1px solid var(--bdr);border-radius:8px;padding:10px}}
.policy-grid span{{display:block;color:var(--tx2);font-size:.76em;text-transform:uppercase;letter-spacing:.4px}}
.policy-grid b{{display:block;color:var(--tx);font-size:.95em;margin-top:3px}}
.policy-grid em{{display:block;color:var(--tx2);font-size:.76em;font-style:normal;margin-top:2px}}

/* Charts */
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.chart-box{{background:var(--s1);border-radius:var(--r);padding:20px}}
.chart-box h3{{font-size:.95em;color:var(--tx2);margin-bottom:12px}}
@media(max-width:768px){{.chart-row{{grid-template-columns:1fr}}}}

/* Tables */
.table-wrap{{overflow-x:auto;margin:16px 0}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid var(--bdr)}}
th{{background:var(--s2);font-size:.78em;text-transform:uppercase;letter-spacing:.5px;color:var(--tx2);font-weight:600}}
td{{background:var(--s1)}}
.metric-name{{font-weight:600;min-width:140px}}
.metric-hint{{display:block;font-weight:400;font-size:.78em;color:var(--tx2)}}
.bar-cell{{display:flex;align-items:center;gap:10px}}
.val{{font-weight:700;min-width:36px}}
.mini-bar{{width:80px;height:7px;background:var(--s3);border-radius:4px;overflow:hidden}}
.mini-bar div{{height:100%;border-radius:4px}}

/* Agent detail tabs */
.agent-tabs{{display:flex;gap:2px;margin:16px 0 0;flex-wrap:wrap}}
.tab-btn{{background:var(--s2);border:none;color:var(--tx2);padding:8px 18px;font-size:.85em;cursor:pointer;border-radius:8px 8px 0 0;transition:.15s}}
.tab-btn:hover{{color:var(--tx)}}
.tab-btn.active{{background:var(--s1);color:var(--tx);font-weight:600;border-bottom:2px solid var(--accent,var(--nv))}}
.tab-panel{{background:var(--s1);border-radius:0 var(--r) var(--r) var(--r);padding:20px}}

/* Finding cards */
.card{{background:var(--s2);border-radius:8px;padding:14px 16px;margin:10px 0;border-left:4px solid var(--bdr)}}
.card-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}}
.card-title{{font-weight:600;font-size:.95em}}
.card-desc{{display:block;color:var(--tx2);font-size:.78em;margin-top:1px}}
.card-score{{text-align:right;white-space:nowrap}}
.score{{font-size:1.3em;font-weight:700;margin-left:6px}}
.checks{{list-style:none;margin:10px 0 0;padding:0}}
.checks li{{padding:4px 0;font-size:.85em;border-bottom:1px solid var(--s3)}}
.checks li:last-child{{border:none}}
.checks .sym{{display:inline-block;width:18px;font-weight:700}}
.checks .ok .sym{{color:#22c55e}}
.checks .err .sym{{color:#ef4444}}
.checks .note{{color:var(--tx2);font-style:italic;padding-left:18px}}
.checks .sub{{padding-left:24px}}

	/* Badges */
	.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.72em;font-weight:600;text-transform:uppercase;vertical-align:middle}}
	.pass{{background:var(--badge-pass-bg);color:var(--badge-pass-tx)}}.warn{{background:var(--badge-warn-bg);color:var(--badge-warn-tx)}}.fail{{background:var(--badge-fail-bg);color:var(--badge-fail-tx)}}
	.attempt-list{{display:flex;flex-wrap:wrap;gap:6px;min-width:180px}}
	.attempt-chip{{display:inline-block;padding:3px 8px;border-radius:4px;font-size:.76em;font-weight:600;white-space:nowrap}}
	.attempt-chip.muted{{background:var(--s2);color:var(--tx2);border:1px dashed var(--bdr)}}

	/* Dataset cards */
.ds-card{{background:var(--s2);border-radius:8px;padding:16px;margin:12px 0;border-left:4px solid var(--nv)}}
.ds-head{{display:flex;align-items:center;gap:10px;margin-bottom:12px}}
.ds-id{{font-weight:700;font-size:1em;color:var(--nv);font-family:monospace}}
.ds-field{{margin:8px 0}}
.ds-label{{display:block;font-size:.72em;text-transform:uppercase;letter-spacing:.5px;color:var(--tx2);font-weight:600;margin-bottom:2px}}
.ds-value{{font-size:.9em;line-height:1.5;white-space:pre-wrap}}
.ds-gt{{color:var(--tx2);font-style:italic}}
.ds-meta{{display:flex;gap:20px;font-size:.82em;color:var(--tx2);margin:8px 0;flex-wrap:wrap}}
.ds-behaviors{{padding-left:18px;margin:4px 0;font-size:.85em}}.ds-behaviors li{{margin:3px 0}}

/* Suggestions */
.sug-box{{background:var(--s1);border-radius:var(--r);padding:20px;margin:20px 0}}
.sug-box h3{{margin-bottom:10px}}.sug-box ol{{padding-left:20px}}.sug-box li{{margin:6px 0;font-size:.92em}}
.sug-pass{{border-left:4px solid #22c55e}}.sug-warn{{border-left:4px solid #eab308}}
.evidence-link{{display:inline-flex;align-items:center;gap:4px;margin-left:4px;font-size:.78em;font-weight:600;color:var(--nv);text-decoration:none;white-space:nowrap}}
.evidence-link:hover{{text-decoration:underline}}
.evidence-link-icon{{font-size:.95em;line-height:1}}

footer{{margin-top:40px;padding:16px 0;border-top:1px solid var(--bdr);color:var(--tx2);font-size:.78em;text-align:center}}
</style></head>
<body><div class="wrap">

<div class="header">
  <div class="header-top">
    <h1>{escape(skill_name)}</h1>
    <button class="theme-toggle" onclick="toggleTheme()">&#9788; / &#9790;</button>
  </div>
  <p class="meta">Skill Evaluator &mdash; {escape(formatted_time)} &mdash; {len(agent_names)} agent(s): {
            escape(", ".join(agent_names))
        }</p>
  <div class="hero">
    <div class="hero-score" style="color:{hero_color}">{hero_score}</div>
    <div class="hero-detail">
      <div><b>Best performing agent/model combination for this skill:</b> {hero_agent}</div>
      <div><b>Trials:</b> {sum(agents[a].get("num_trials", 0) for a in agent_names)} across {
            len(agent_names)
        } agent(s)</div>
      <div><b>Metrics:</b> {escape(metrics_label)}</div>
    </div>
  </div>
</div>

<div class="nav">
  <button class="active" onclick="go(this,'p-overview')">Overview</button>
  <button onclick="go(this,'p-trials')">Trials</button>
  <button onclick="go(this,'p-detail')">Agent Details</button>
  <button onclick="go(this,'p-dataset')">Dataset</button>
  <button onclick="go(this,'p-suggest')">Suggestions</button>
</div>

<!-- ===== OVERVIEW ===== -->
<div class="page active" id="p-overview">
  <div class="agent-grid">{overall_cards}</div>
  {harbor_analysis_html}
  {run_config_html}
  {attempt_policy_html}
  {pass_summary_html}
  {custom_only_note_html}
  {default_chart_html}
  {default_score_html}
  {dimension_html}
  {custom_html}
  {custom_lift_html}
  {lift_html}
</div>

<!-- ===== TRIALS ===== -->
<div class="page" id="p-trials">
  <h2>Score Heatmap</h2>
  <p class="subtle">{escape(heatmap_text)}</p>
  <div class="table-wrap"><table><thead><tr><th>Eval Case</th>{score_table_head}</tr></thead><tbody>{
            heatmap_rows
        }</tbody></table></div>

  <div class="chart-row">
    <div class="chart-box"><h3>Token Usage by Agent (total across trials)</h3><canvas id="cTokens"></canvas></div>
    <div class="chart-box"><h3>Steps per Eval Case</h3><canvas id="cSteps"></canvas></div>
  </div>

  {er_html}

  {attempt_detail_html}

  {
            ""
            if not lift_per_trial_rows
            else f'''<h3>Lift per Eval Case</h3>
  <p class="subtle">Per-trial with-skill vs without-skill comparison.</p>
  <div class="table-wrap"><table><thead><tr><th>Agent</th><th>Case</th>{"".join(f"<th>{_METRIC_DISPLAY.get(m, m)}</th>" for m in display_metrics)}</tr></thead><tbody>{lift_per_trial_rows}</tbody></table></div>'''
        }

  {
            ""
            if not custom_lift_per_trial_rows
            else f'''<h3>Custom Lift per Eval Case</h3>
  <p class="subtle">Per-trial with-skill vs without-skill comparison for user-owned reward fields.</p>
  <div class="table-wrap"><table><thead><tr><th>Agent</th><th>Case</th>{"".join(f"<th>{'Overall Reward' if m == 'overall' else escape(m)}</th>" for m in custom_trial_metrics)}</tr></thead><tbody>{custom_lift_per_trial_rows}</tbody></table></div>'''
        }
</div>

<!-- ===== AGENT DETAILS ===== -->
<div class="page" id="p-detail">
  <div class="agent-tabs">{agent_tab_buttons}</div>
  {agent_tab_panels}
</div>

<!-- ===== DATASET ===== -->
<div class="page" id="p-dataset">
  <h2>AgentSkills Dataset</h2>
  {dataset_html}
</div>

<!-- ===== SUGGESTIONS ===== -->
<div class="page" id="p-suggest">
  {suggestions_html}
</div>

<footer>Generated by <b>Skill Evaluator</b> &mdash; {escape(formatted_time)}</footer>
</div>

<script>
"""
        + _build_js(
            radar_labels,
            radar_js,
            bar_labels,
            bar_js,
            token_labels_js,
            token_prompt_js,
            token_completion_js,
            token_cached_js,
            steps_labels_js,
            steps_datasets_js,
        )
        + """
</script></body></html>"""
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
