# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor findings report — surfaces actionable insights from reward.json details.

Generates a Rich panel showing what failed, why, and LLM-generated suggestions
for skill developers.  Called after ``_display_harbor_results`` in the CLI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRICS,
    METRIC_DESCRIPTIONS,
    METRIC_DISPLAY,
    METRIC_QUESTIONS,
    extract_custom_metrics,
)
from skillevaluator.utils.redaction import redact_sensitive_data

logger = logging.getLogger(__name__)

DISPLAY_METRICS = DEFAULT_METRICS

_METRIC_LABELS = {
    "security": "SECURITY (unsafe operations, secret leakage, unauthorized access)",
    "skill_execution": "SKILL EXECUTION (activation, script run, workflow order, error recovery)",
    "skill_efficiency": "EFFICIENCY (routing, tool call productivity)",
    "accuracy": "ACCURACY (factual correctness, 5-criterion rubric)",
    "goal_accuracy": "GOAL ACCURACY (did agent achieve the user's goal?)",
    "behavior_check": "BEHAVIOR CHECK (expected workflow adherence)",
}


def _load_trial_rewards(results_dir: Path, agent: str) -> list[dict[str, Any]]:
    """Load all reward.json files for a given agent's with-skill trials."""
    trials_dir = results_dir / agent / "with-skill" / "trials"
    if not trials_dir.exists():
        return []
    rewards = []
    for reward_file in sorted(trials_dir.rglob("reward.json")):
        try:
            rewards.append(json.loads(reward_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", reward_file, e)
    return rewards


def _pick_best_agent(
    agents_data: dict[str, dict[str, Any]],
) -> str:
    """Select the agent with the highest overall with-skill score."""
    best_agent = ""
    best_score = -1.0
    for agent, data in agents_data.items():
        with_scores = data.get("with_skill", {})
        if not with_scores:
            continue
        metrics = [m for m in DISPLAY_METRICS if m in with_scores] or list(DISPLAY_METRICS)
        overall = sum(with_scores.get(m, 0.0) for m in metrics) / len(metrics)
        if overall > best_score:
            best_score = overall
            best_agent = agent
    return best_agent


def _agent_model_for_display(
    agent: str,
    harbor_result: dict[str, Any],
    agents_data: dict[str, dict[str, Any]],
) -> str:
    """Return the resolved user-facing model for an agent when available."""
    run_config = harbor_result.get("run_config", {})
    run_config_agents = run_config.get("agents", {}) if isinstance(run_config, dict) else {}
    meta = run_config_agents.get(agent, {}) if isinstance(run_config_agents, dict) else {}
    if isinstance(meta, dict):
        model = str(meta.get("model") or "").strip()
        if model:
            return model

    agent_data = agents_data.get(agent, {})
    if isinstance(agent_data, dict):
        model = str(agent_data.get("model") or "").strip()
        if model:
            return model
    return ""


def _agent_model_label(
    agent: str,
    harbor_result: dict[str, Any],
    agents_data: dict[str, dict[str, Any]],
) -> str:
    model = _agent_model_for_display(agent, harbor_result, agents_data)
    return f"{agent} / {model}" if model else agent


def _details_for_findings(reward: dict[str, Any]) -> dict[str, Any]:
    details = reward.get("details")
    out = dict(details) if isinstance(details, dict) else {}
    custom_details = reward.get("custom_details")
    if isinstance(custom_details, dict):
        for metric, detail in custom_details.items():
            out.setdefault(str(metric), detail)
    return out


def _finding_metric_names(rewards: list[dict[str, Any]]) -> list[str]:
    custom_names: set[str] = set()
    for reward in rewards:
        custom_names.update(extract_custom_metrics(reward))
        custom_details = reward.get("custom_details")
        if isinstance(custom_details, dict):
            custom_names.update(str(metric) for metric in custom_details)
    return list(DISPLAY_METRICS) + sorted(custom_names.difference(DISPLAY_METRICS))


def _metric_score(reward: dict[str, Any], metric: str) -> float:
    value = reward.get(metric)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return extract_custom_metrics(reward).get(metric, 0.0)


def _metric_label(metric: str) -> str:
    if metric not in DISPLAY_METRICS:
        return f"custom: {metric}"
    return _METRIC_LABELS.get(
        metric,
        f"{METRIC_DISPLAY.get(metric, metric).upper()} ({METRIC_DESCRIPTIONS.get(metric, '')})",
    )


def _extract_findings(rewards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract actionable findings from reward.json details across trials."""
    findings: list[dict[str, Any]] = []

    metric_names = _finding_metric_names(rewards)
    aggregated: dict[str, list[dict[str, Any]]] = {m: [] for m in metric_names}
    for reward in rewards:
        details = _details_for_findings(reward)
        for metric in metric_names:
            if metric in details:
                aggregated[metric].append(
                    {
                        "score": _metric_score(reward, metric),
                        "detail": details[metric],
                        "entry_id": reward.get("entry_id", "?"),
                    }
                )

    for metric in metric_names:
        trials = aggregated[metric]
        if not trials:
            continue

        avg_score = sum(t["score"] for t in trials) / len(trials)
        label = _metric_label(metric)

        metric_refs = []
        for t in trials:
            d = t["detail"]
            if isinstance(d, dict):
                metric_refs.extend(d.get("evidence_refs") or [])
        _seen: set[tuple[Any, ...]] = set()
        _refs: list[dict[str, Any]] = []
        for r in metric_refs:
            k = (r.get("source"), r.get("json_pointer"), r.get("kind"), r.get("path"))
            if k not in _seen:
                _seen.add(k)
                _refs.append(r)

        if avg_score >= 0.8:
            reasons = _collect_pass_reasons(metric, trials)
            findings.append(
                {
                    "metric": metric,
                    "label": label,
                    "severity": "ok",
                    "score": avg_score,
                    "reasons": reasons[:2],
                    "evidence_refs": _refs[:8],
                }
            )
        else:
            reasons = _collect_fail_reasons(metric, trials)
            severity = "critical" if avg_score < 0.4 else "warning"
            findings.append(
                {
                    "metric": metric,
                    "label": label,
                    "severity": severity,
                    "score": avg_score,
                    "reasons": reasons[:4],
                    "evidence_refs": _refs[:8],
                }
            )

    return findings


def _render_findings_body(findings: list[dict[str, Any]]) -> Any:
    """Render findings list into a Rich Text body (icon/label/score/question/reasons/evidence)."""
    from rich.text import Text

    body = Text()
    for finding in findings:
        if finding["severity"] == "critical":
            icon = "⚠ CRITICAL"
            style = "bold red"
        elif finding["severity"] == "warning":
            icon = "⚠ WARNING"
            style = "bold yellow"
        else:
            icon = "✓"
            style = "bold green"

        body.append(f"  {icon}: ", style=style)
        body.append(f"{finding['label']} ", style="bold white")
        body.append(f"{finding['score']:.2f}\n", style=style)
        question = METRIC_QUESTIONS.get(finding["metric"])
        if question:
            body.append(f"     {question}\n", style="dim cyan")

        reason_style = "dim green" if finding["severity"] == "ok" else "dim"
        for reason in finding["reasons"]:
            body.append(f"    → {reason}\n", style=reason_style)

        for ref in (finding.get("evidence_refs") or [])[:3]:
            if isinstance(ref, str):
                body.append(f"      evidence: {ref}\n", style="dim")
            else:
                loc = ref.get("json_pointer") or ref.get("path") or ""
                body.append(f"      evidence: {ref.get('source', '')}{loc}\n", style="dim")

        body.append("\n")
    return body


def _collect_fail_reasons(metric: str, trials: list[dict[str, Any]]) -> list[str]:
    """Extract human-readable failure reasons from trial details."""
    reasons: list[str] = []

    for trial in trials:
        detail = trial["detail"]

        if metric == "behavior_check":
            for r in detail.get("results", []):
                if not r.get("passed") and r.get("reason"):
                    reasons.append(r["reason"])

        elif metric == "accuracy":
            criteria = detail.get("criteria", {})
            for crit, passed in criteria.items():
                if not passed:
                    reasons.append(f"{crit} failed")
            if detail.get("reason"):
                reasons.append(detail["reason"])

        elif metric in ("goal_accuracy", "security"):
            if detail.get("reason"):
                reasons.append(detail["reason"])
            for finding in detail.get("findings", []):
                if isinstance(finding, str):
                    reasons.append(finding)
                elif isinstance(finding, dict):
                    message = str(finding.get("message") or finding.get("type") or "")
                    attribution = str(finding.get("attribution") or "")
                    explanation = str(finding.get("attribution_explanation") or "")
                    if attribution:
                        message = f"{message} | Attribution: {attribution.replace('_', ' ')}"
                    if explanation:
                        message = f"{message}. {explanation}"
                    if message:
                        reasons.append(message)

        elif metric == "skill_execution":
            for check_name, check_data in detail.items():
                if isinstance(check_data, dict) and not check_data.get("passed", True):
                    reasons.append(check_data.get("reason", f"{check_name} failed"))
                if isinstance(check_data, dict) and check_name == "error_recovery":
                    for corr in check_data.get("corrections", []):
                        fault = corr.get("fault", "unknown")
                        err = corr.get("error", "")[:100]
                        reasons.append(f"[{fault} fault] {err}")

        elif metric == "skill_efficiency":
            for check_name, check_data in detail.items():
                if isinstance(check_data, dict) and not check_data.get("passed", True):
                    reasons.append(check_data.get("reason", f"{check_name} failed"))

        else:
            if isinstance(detail, dict) and detail.get("reason"):
                reasons.append(str(detail["reason"]))
            if isinstance(detail, dict):
                for finding in detail.get("findings", []):
                    if isinstance(finding, str):
                        reasons.append(finding)
                    elif isinstance(finding, dict):
                        message = str(finding.get("message") or finding.get("reason") or "")
                        if message:
                            reasons.append(message)

    seen: set[str] = set()
    deduped: list[str] = []
    for r in reasons:
        key = r[:80].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _collect_pass_reasons(metric: str, trials: list[dict[str, Any]]) -> list[str]:
    """Extract concise success reasons."""
    reasons: list[str] = []
    for trial in trials:
        detail = trial["detail"]
        if not isinstance(detail, dict):
            continue

        if metric == "behavior_check":
            passed_count = sum(1 for r in detail.get("results", []) if r.get("passed"))
            total = len(detail.get("results", []))
            if total:
                reasons.append(f"{passed_count}/{total} expected behaviors observed")
            if detail.get("reason"):
                reasons.append(detail["reason"])

        elif metric == "accuracy":
            criteria = detail.get("criteria", {})
            passed = [k for k, v in criteria.items() if v]
            if passed:
                reasons.append(f"Passed: {', '.join(passed)}")
            if detail.get("reason"):
                reasons.append(detail["reason"])

        elif metric in ("goal_accuracy", "security"):
            if detail.get("reason"):
                reasons.append(detail["reason"])
            for finding in detail.get("findings", []):
                if not isinstance(finding, dict):
                    continue
                if finding.get("score_impact"):
                    continue
                message = str(finding.get("message") or "")
                attribution = str(finding.get("attribution") or "")
                if attribution:
                    message = f"{message} | Attribution: {attribution.replace('_', ' ')}"
                if message:
                    reasons.append(message)
            if detail.get("end_state"):
                reasons.append(detail["end_state"])

        else:
            if detail.get("reason"):
                reasons.append(str(detail["reason"]))
            for check_data in detail.values():
                if isinstance(check_data, dict) and check_data.get("passed") and check_data.get("reason"):
                    reasons.append(check_data["reason"])

    seen: set[str] = set()
    deduped: list[str] = []
    for r in reasons:
        key = r[:80].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _build_evidence_ref_lookup(rewards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a lookup from compact string key ``source#json_pointer`` to full dict ref.

    Iterates over all metrics in every reward's ``details`` dict, collecting
    ``evidence_refs`` entries.  The resulting mapping lets
    ``_generate_suggestions_structured`` resolve the compact string refs that
    the LLM returns into the richer dict form consumed by SkillEvaluator report
    templates (which read ``ref.kind``, ``ref.json_pointer``, ``ref.path``,
    ``ref.excerpt``).
    """
    lookup: dict[str, dict[str, Any]] = {}
    for reward in rewards:
        details = reward.get("details") or {}
        if not isinstance(details, dict):
            continue
        for metric_detail in details.values():
            if not isinstance(metric_detail, dict):
                continue
            for ref in metric_detail.get("evidence_refs") or []:
                if not isinstance(ref, dict):
                    continue
                source = ref.get("source") or ""
                pointer = ref.get("json_pointer") or ""
                if source or pointer:
                    key = f"{source}#{pointer}"
                    if key not in lookup:
                        lookup[key] = ref
    return lookup


def _resolve_evidence_ref(ref: Any, lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Resolve a single evidence ref to a dict.

    If ``ref`` is already a dict, return it unchanged.  If ``ref`` is a string
    of the form ``"source#json_pointer"``, look it up in *lookup* and return the
    full dict.  If the lookup misses, fall back to a minimal dict parsed from
    the string, with ``kind`` set to ``"evidence"``.
    """
    if isinstance(ref, dict):
        return ref
    ref_str = str(ref)
    if ref_str in lookup:
        return lookup[ref_str]
    # Parse the compact string into a minimal dict
    if "#" in ref_str:
        source, _, pointer = ref_str.partition("#")
    else:
        source, pointer = ref_str, ""
    return {"source": source, "json_pointer": pointer, "kind": "evidence"}


def _generate_suggestions_structured(
    skill_name: str,
    findings: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate max 4 LLM-powered structured suggestion objects for skill developers.

    Each object has keys: ``suggestion`` (str), ``dimension`` (metric name str),
    ``evidence_refs`` (list of dicts with keys ``source``, ``json_pointer``,
    ``kind``, and optionally ``path`` and ``excerpt``).

    The LLM is still asked to return compact ``"source#/json_pointer"`` strings
    (cheap, reliable).  After parsing the LLM response each string ref is
    resolved against a lookup built from ``rewards[*]["details"][metric]["evidence_refs"]``
    so that the richer dict fields (``kind``, ``path``, ``excerpt``) are
    preserved.  Unresolvable strings fall back to a minimal dict with
    ``kind="evidence"``.  Callers that consume ``suggestions_v2[*].evidence_refs``
    (including the SkillEvaluator report template) therefore always receive dicts rather
    than plain strings.
    """
    failed_findings = [f for f in findings if f["severity"] in ("critical", "warning")]
    if not failed_findings:
        return []

    summary_parts = []
    for f in failed_findings:
        summary_parts.append(f"- {f['label']} ({f['score']:.2f}): {'; '.join(f['reasons'][:2])}")
    findings_summary = "\n".join(summary_parts)

    behavior_failures = []
    error_recovery_info = []
    for reward in rewards:
        details = reward.get("details", {})
        bc = details.get("behavior_check", {})
        for r in bc.get("results", []):
            if not r.get("passed"):
                behavior_failures.append(r.get("reason", ""))
        er = details.get("skill_execution", {}).get("error_recovery", {})
        for corr in er.get("corrections", []):
            error_recovery_info.append(f"[{corr.get('fault', '?')}] {corr.get('error', '')[:150]}")

    evidence_lines = []
    for f in failed_findings:
        for ref in (f.get("evidence_refs") or [])[:3]:
            if isinstance(ref, dict):
                loc = ref.get("json_pointer") or ref.get("path") or ""
                evidence_lines.append(
                    f"  - [{f['metric']}] {ref.get('kind', '')} {ref.get('source', '')}{loc}: "
                    f"{str(ref.get('label') or ref.get('excerpt') or '')[:120]}"
                )
    evidence_block = "\n".join(evidence_lines) or "(no evidence refs)"

    prompt = f"""You are an expert skill evaluator. A skill named "{skill_name}" was tested with an AI coding agent and received the following evaluation results:

FINDINGS (metrics that scored below threshold):
{findings_summary}

FAILED BEHAVIORS:
{chr(10).join(f"- {b}" for b in behavior_failures[:6]) or "(none)"}

ERROR RECOVERY ISSUES:
{chr(10).join(f"- {e}" for e in error_recovery_info[:4]) or "(none)"}

EVIDENCE REFERENCES (cite the relevant ones as trajectory.json#/pointer in your suggestions):
{evidence_block}

Based on these results, provide exactly 3-4 specific, actionable suggestions for the skill developer to improve their skill. Focus on:
1. What the developer should fix in their skill (SKILL.md, scripts, config)
2. What they should add to evals/environment/ (Dockerfile, docker-compose) if tools are missing
3. Documentation gaps that caused agent failures

Keep each suggestion to 1-2 sentences. Be concrete — reference specific files, commands, or sections.
If the failures are purely environmental (e.g. CLI not installed in container), say so directly.

Respond with ONLY a JSON array of objects {{"suggestion": "...", "dimension": "<metric>", "evidence_refs": ["trajectory.json#/steps/N", ...]}}, no other text."""

    # Build the lookup once from all rewards' detail refs
    ref_lookup = _build_evidence_ref_lookup(rewards)

    try:
        from skillevaluator.tier3.eval_core.llm_judge import _extract_json, call_public_llm

        content, error = call_public_llm(prompt, max_tokens=1536)
        if error:
            logger.warning("LLM suggestion generation failed: %s", error)
            return [{"suggestion": s, "dimension": "", "evidence_refs": []} for s in _fallback_suggestions(findings)]

        parsed = _extract_json(content) if content else None
        if isinstance(parsed, list):
            result: list[dict[str, Any]] = []
            for item in parsed[:4]:
                if isinstance(item, dict):
                    raw_refs = list(item.get("evidence_refs") or [])
                    resolved_refs = [_resolve_evidence_ref(r, ref_lookup) for r in raw_refs]
                    result.append(
                        {
                            "suggestion": str(item.get("suggestion", "")),
                            "dimension": str(item.get("dimension", "")),
                            "evidence_refs": resolved_refs,
                        }
                    )
                elif isinstance(item, str):
                    result.append({"suggestion": item, "dimension": "", "evidence_refs": []})
            if result:
                return result
        if isinstance(parsed, dict) and "suggestions" in parsed:
            return [{"suggestion": str(s), "dimension": "", "evidence_refs": []} for s in parsed["suggestions"][:4]]

        return [{"suggestion": s, "dimension": "", "evidence_refs": []} for s in _fallback_suggestions(findings)]
    except Exception as e:
        logger.warning("LLM suggestion generation failed: %s", e)
        return [{"suggestion": s, "dimension": "", "evidence_refs": []} for s in _fallback_suggestions(findings)]


def _generate_suggestions(
    skill_name: str,
    findings: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
) -> list[str]:
    """Generate max 4 LLM-powered suggestions for skill developers.

    Delegates to ``_generate_suggestions_structured`` and extracts the plain
    suggestion strings for back-compatibility with callers that expect ``list[str]``.
    """
    return [
        str(o.get("suggestion", ""))
        for o in _generate_suggestions_structured(skill_name, findings, rewards)
        if o.get("suggestion")
    ]


def _fallback_suggestions(findings: list[dict[str, Any]]) -> list[str]:
    """Rule-based fallback when LLM is unavailable."""
    suggestions: list[str] = []
    for f in findings:
        if f["severity"] not in ("critical", "warning"):
            continue
        for reason in f["reasons"][:1]:
            r_lower = reason.lower()
            if "not found" in r_lower or "command not found" in r_lower:
                suggestions.append(
                    f"Add evals/environment/Dockerfile to install missing CLI tools "
                    f"required by the skill ({f['label']} scored {f['score']:.2f})."
                )
            elif "unsafe" in r_lower or "destructive" in r_lower or "secret" in r_lower:
                if "likely skill related" in r_lower:
                    suggestions.append(
                        "Add explicit safety rules to SKILL.md and any skill scripts: require confirmation before "
                        "destructive cleanup, forbid writing shell startup files, and avoid exposing secrets."
                    )
                elif "baseline" in r_lower or "environment" in r_lower:
                    suggestions.append(
                        "Review the eval prompt and sandbox permissions: the same unsafe behavior appeared in baseline, "
                        "so this may be a prompt/environment issue rather than a skill defect."
                    )
                else:
                    suggestions.append(
                        f"Review security evidence for {f['label']} ({f['score']:.2f}) and add guardrails or custom "
                        "grader checks for unsafe commands, sensitive files, and secret handling."
                    )
            elif "did not read" in r_lower or "skill.md" in r_lower:
                suggestions.append(
                    f"Review SKILL.md discoverability — agent failed to find or read it "
                    f"({f['label']} scored {f['score']:.2f})."
                )
            elif "not achieve" in r_lower or "did not complete" in r_lower:
                suggestions.append(
                    f"Check that the skill's workflow can complete end-to-end in the eval environment "
                    f"({f['label']} scored {f['score']:.2f})."
                )
            else:
                suggestions.append(f"{f['label']} scored {f['score']:.2f}: {reason}")
    return suggestions[:4]


def _passing_skill_suggestions(
    findings: list[dict[str, Any]],
    rewards: list[dict[str, Any]],
) -> list[str]:
    """Generate improvement suggestions even when all metrics pass."""
    suggestions: list[str] = []
    num_trials = len(rewards)

    lowest = min(findings, key=lambda f: f["score"]) if findings else None
    if lowest and lowest["score"] < 0.95:
        suggestions.append(
            f"Strengthen {lowest['label'].split('(')[0].strip()} "
            f"(scored {lowest['score']:.2f}) — review the reasons above and "
            f"refine expected_behavior or ground_truth in evals.json to better "
            f"match what agents actually do."
        )

    suggestions.append(
        "Add evals/environment/Dockerfile if the skill depends on CLI tools, "
        "databases, or APIs — this ensures consistent results across agents "
        "and CI environments."
    )

    if num_trials < 4:
        suggestions.append(
            f"Expand evals.json with more test cases (currently {num_trials}). "
            "Use 'skillevaluator create-eval-dataset --full' to generate a 4-bucket "
            "strategy covering explicit, implicit, contextual, and negative cases."
        )

    suggestions.append(
        "Run with additional agents ('skillevaluator evaluate <skill> --agents "
        "claude-code,codex,opencode --env-mode docker') to verify the skill works across "
        "different coding agents."
    )

    return suggestions[:4]


def _harbor_viewer_evidence_links(rewards: list[dict[str, Any]]) -> list[str]:
    links: list[str] = []
    for reward in _prioritized_evidence_rewards(rewards):
        harbor_viewer = reward.get("harbor_viewer")
        if not isinstance(harbor_viewer, dict):
            continue
        evidence_urls = harbor_viewer.get("evidence_urls")
        if isinstance(evidence_urls, list):
            for item in evidence_urls:
                if isinstance(item, dict) and item.get("url"):
                    links.append(str(item["url"]))
        trial_url = harbor_viewer.get("trial_url")
        if trial_url:
            links.append(str(trial_url))

    seen: set[str] = set()
    deduped: list[str] = []
    for link in links:
        if link in seen:
            continue
        seen.add(link)
        deduped.append(link)
    return deduped


def _prioritized_evidence_rewards(rewards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failing = [reward for reward in rewards if _reward_has_failing_signal(reward)]
    passing = [reward for reward in rewards if reward not in failing]
    return [*failing, *passing]


def _reward_has_failing_signal(reward: dict[str, Any]) -> bool:
    for metric in DISPLAY_METRICS:
        value = reward.get(metric)
        if isinstance(value, int | float) and not isinstance(value, bool) and float(value) < 0.8:
            return True
    details = reward.get("details")
    if not isinstance(details, dict):
        return False
    for detail in details.values():
        if not isinstance(detail, dict):
            continue
        score = detail.get("score")
        if isinstance(score, int | float) and not isinstance(score, bool) and float(score) < 0.8:
            return True
        results = detail.get("results")
        if isinstance(results, list) and any(
            isinstance(item, dict) and item.get("passed") is False for item in results
        ):
            return True
    return False


def add_evidence_links_to_suggestions(
    suggestions: list[str],
    rewards: list[dict[str, Any]],
) -> list[str]:
    """Append Harbor viewer evidence links to report suggestions when available."""
    links = _harbor_viewer_evidence_links(rewards)
    if not suggestions or not links:
        return suggestions

    linked: list[str] = []
    for index, suggestion in enumerate(suggestions):
        text = str(suggestion).rstrip()
        if "Evidence:" in text or "http://" in text or "https://" in text:
            linked.append(text)
            continue
        punctuation = "" if text.endswith((".", "!", "?")) else "."
        linked.append(f"{text}{punctuation} Evidence: {links[min(index, len(links) - 1)]}")
    return linked


def _write_findings_artifact(
    *,
    results_dir: Path,
    skill_name: str,
    agent: str,
    findings: list[dict[str, Any]],
    suggestions: list[str],
    suggestion_mode: str,
    suggestions_v2: list[dict[str, Any]] | None = None,
) -> Path | None:
    artifact = results_dir / agent / "findings.json"
    payload = {
        "skill_name": skill_name,
        "agent": agent,
        "suggestion_mode": suggestion_mode,
        "findings": findings,
        "suggestions": suggestions,
        "suggestions_v2": suggestions_v2 or [],
    }
    try:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(redact_sensitive_data(payload), indent=2), encoding="utf-8")
    except OSError as e:
        logger.debug("Failed to write findings artifact %s: %s", artifact, e)
        return None
    return artifact


def display_findings_report(
    harbor_result: dict[str, Any],
    skill_name: str,
    harbor_agents: list[str],
    results_dir: Path,
) -> set[str]:
    """Display the findings report panel after the score table.

    Return the atomic feedback messages printed in the detailed panel. The
    caller uses these to suppress only matching compact feedback items while
    preserving payload-only conclusions and recommendations.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    agents_data = harbor_result.get("agents", {})

    if len(harbor_agents) > 1:
        best_agent = _pick_best_agent(agents_data)
        if best_agent:
            best_agent_label = _agent_model_label(best_agent, harbor_result, agents_data)
            console.print(
                f"  [dim]Findings from best performing agent/model combination for your skill:[/dim] "
                f"[bold cyan]{best_agent_label}[/bold cyan]"
            )
            console.print()
    else:
        best_agent = harbor_agents[0] if harbor_agents else ""

    if not best_agent or best_agent not in agents_data:
        return set()

    agent_reports: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    report_agents = list(dict.fromkeys([*harbor_agents, *agents_data.keys()]))
    for agent in report_agents:
        if agent not in agents_data:
            continue
        rewards_for_agent = _load_trial_rewards(results_dir, agent)
        if not rewards_for_agent:
            continue
        findings_for_agent = _extract_findings(rewards_for_agent)
        if findings_for_agent:
            agent_reports[agent] = (findings_for_agent, rewards_for_agent)

    if best_agent not in agent_reports:
        return set()

    findings, rewards = agent_reports[best_agent]
    body = _render_findings_body(findings)
    rendered_messages = {
        str(message).strip()
        for finding in findings
        for message in (
            finding.get("label"),
            METRIC_QUESTIONS.get(str(finding.get("metric") or "")),
            *(finding.get("reasons") or []),
        )
        if str(message or "").strip()
    }

    structured = _generate_suggestions_structured(skill_name, findings, rewards)
    suggestions = [s["suggestion"] for s in structured]
    suggestion_mode = "remediation" if suggestions else "passing_next_steps"

    if suggestions:
        rendered_messages.update(str(suggestion).strip() for suggestion in suggestions if str(suggestion).strip())
        suggestions = add_evidence_links_to_suggestions(suggestions, rewards)
        body.append("  \U0001f4a1 SUGGESTIONS\n", style="bold cyan")
        for i, suggestion in enumerate(suggestions, 1):
            body.append(f"    {i}. {suggestion}\n", style="white")
    else:
        body.append("  \U0001f4a1 NEXT STEPS\n", style="bold cyan")
        suggestions = _passing_skill_suggestions(findings, rewards)
        rendered_messages.update(str(suggestion).strip() for suggestion in suggestions if str(suggestion).strip())
        suggestions = add_evidence_links_to_suggestions(suggestions, rewards)
        for i, suggestion in enumerate(suggestions, 1):
            body.append(f"    {i}. {suggestion}\n", style="white")
    rendered_messages.update(str(suggestion).strip() for suggestion in suggestions if str(suggestion).strip())

    for agent, (agent_findings, _agent_rewards) in agent_reports.items():
        is_best_agent = agent == best_agent
        agent_suggestions = suggestions if is_best_agent else []
        agent_suggestion_mode = suggestion_mode if is_best_agent else "not_generated"
        agent_suggestions_v2 = structured if is_best_agent else []
        _write_findings_artifact(
            results_dir=results_dir,
            skill_name=skill_name,
            agent=agent,
            findings=agent_findings,
            suggestions=agent_suggestions,
            suggestion_mode=agent_suggestion_mode,
            suggestions_v2=agent_suggestions_v2,
        )

    panel_agent_label = _agent_model_label(best_agent, harbor_result, agents_data)
    console.print(
        Panel(
            body,
            title=f"[bold]{skill_name} / {panel_agent_label} \u2014 Findings[/bold]",
            border_style="cyan",
            padding=(1, 1),
        )
    )
    console.print()
    return rendered_messages
