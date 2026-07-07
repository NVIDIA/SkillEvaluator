# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-as-Judge for richer Tier 3 Insights (conclusions + recommendations).

The Insights judge runs *after* the deterministic dimension judge has produced
per-dimension scores, verdicts, and explanations. It receives the canonical
Tier 3 payload and returns up to N additional, contextual conclusions and
actionable recommendations that go beyond the deterministic baselines (best
performing agent, weakest dimension, coverage expansion).

If the LLM is unavailable the judge falls back to an empty list, so the
deterministic conclusions and recommendations from
:mod:`skillevaluator.evaluation.tier3_normalizer` continue to render unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from skillevaluator.constants import (
    AGENT_EVAL_EVALUATORS,
    INSIGHTS_JUDGE_MAX_CONCLUSIONS,
    INSIGHTS_JUDGE_MAX_RECOMMENDATIONS,
    INSIGHTS_JUDGE_MAX_TOKENS,
    INSIGHTS_JUDGE_MODEL,
    INSIGHTS_JUDGE_TEMPERATURE,
)
from skillevaluator.inference.client import LLMClient
from skillevaluator.logging_config import get_logger

logger = get_logger(__name__)


_VALID_CONCLUSION_SEVERITIES = {"pass", "warn", "fail"}
_VALID_RECOMMENDATION_CATEGORIES = {
    "Update",
    "Add",
    "Implement",
    "Document",
    "Fix",
    "Test",
    "Improve",
    "Action",
}


_SYSTEM_PROMPT = """\
You are an expert Skill Quality Reviewer summarising a Tier 3 (live agent)
evaluation of an Agent Skill.

You will receive a JSON payload describing:
- Skill identity (name, agents run, best performing agent, runtime)
- Five SkillEvaluator dimensions (Security, Correctness, Discoverability,
  Effectiveness, Efficiency) with score, baseline, lift, verdict, and a
  human explanation.
- Per-evaluator scores (security, skill_execution, skill_efficiency, accuracy,
  goal_accuracy, behavior_check) for the best performing agent.
- A short selection of trial outcomes, error_recovery summaries, and
  baseline pairings.
- A handful of dataset cases the agent ran against.

The runner already provides three deterministic baseline observations
(\"best performing agent\", \"weakest dimension\", and an optional
\"coverage expansion\" suggestion). DO NOT repeat those exact observations.

Your job is to produce ADDITIONAL, contextual insights:

CONCLUSIONS (up to {max_conclusions}): observations the reviewer should know
- what worked, what regressed, where evidence points to systemic issues. Each
should be 1-2 sentences and cite a concrete signal (a metric, a case id,
an error message, or a comparison).

RECOMMENDATIONS (up to {max_recommendations}): concrete, actionable next
steps for the skill author. Each must be one short imperative sentence.
Tag each with a category from this set: {categories}.

SEVERITY: tag every conclusion and recommendation with one of:
- \"pass\"   - positive observation / low-effort recommendation
- \"warn\"   - needs attention / medium-effort recommendation
- \"fail\"   - regression or blocking issue / high-priority fix

IMPORTANT: Respond with ONLY a JSON object, no markdown, no preamble. Use
this exact schema:

{{
  \"conclusions\": [
    {{\"title\": \"<short title>\", \"message\": \"<1-2 sentences>\", \"severity\": \"pass|warn|fail\"}}
  ],
  \"recommendations\": [
    {{\"category\": \"<one of the allowed categories>\", \"title\": \"<short title>\", \"message\": \"<one imperative sentence>\", \"severity\": \"pass|warn|fail\"}}
  ]
}}
""".format(
    max_conclusions=INSIGHTS_JUDGE_MAX_CONCLUSIONS,
    max_recommendations=INSIGHTS_JUDGE_MAX_RECOMMENDATIONS,
    categories=", ".join(sorted(_VALID_RECOMMENDATION_CATEGORIES)),
)


def _summarize_dimensions(dimensions: list[dict]) -> list[dict]:
    """Project SkillEvaluator dimensions into a compact prompt-friendly shape."""
    summary: list[dict] = []
    for dim in dimensions or []:
        summary.append(
            {
                "id": dim.get("id"),
                "score": dim.get("score", dim.get("with_skill", 0.0)),
                "baseline": dim.get("baseline"),
                "lift": dim.get("lift"),
                "verdict": dim.get("verdict"),
                "explanation": (dim.get("explanation") or "")[:600],
                "evaluators": dim.get("evaluators") or [],
            }
        )
    return summary


def _summarize_evaluators(evaluators: dict) -> dict:
    """Drop any unknown evaluator fields and keep core scores."""
    out: dict = {}
    for name in AGENT_EVAL_EVALUATORS:
        entry = evaluators.get(name) if isinstance(evaluators, dict) else None
        if isinstance(entry, dict):
            out[name] = {
                "with_skill": entry.get("with_skill"),
                "baseline": entry.get("baseline"),
                "lift": entry.get("lift"),
            }
    return out


def _summarize_trials(trials: list[dict]) -> list[dict]:
    """Pick a representative slice of trials for the prompt."""
    if not trials:
        return []

    # Sort by overall score so failures and successes both appear.
    def _key(t: dict) -> float:
        overall = t.get("overall")
        return overall if isinstance(overall, (int, float)) else 0.5

    sorted_trials = sorted(trials, key=_key)
    chosen: list[dict] = sorted_trials[:3] + sorted_trials[-3:]
    seen: set[str] = set()
    compact: list[dict] = []
    for trial in chosen:
        tid = str(trial.get("trial_id") or trial.get("entry_id") or "")
        if tid in seen:
            continue
        seen.add(tid)
        scores = trial.get("scores") or {}
        compact.append(
            {
                "agent": trial.get("agent"),
                "entry_id": trial.get("entry_id"),
                "overall": trial.get("overall"),
                "scores": {k: v for k, v in scores.items() if v is not None},
                "baseline_overall": trial.get("baseline_overall"),
                "lift_scores": trial.get("lift_scores") or {},
                "warnings": (trial.get("warnings") or [])[:3],
                "error_recovery": trial.get("error_recovery") or {},
            }
        )
        if len(compact) >= 6:
            break
    return compact


def _summarize_dataset(dataset: list[dict]) -> list[dict]:
    """Trim dataset cases to the question + ground truth (plus id)."""
    summary: list[dict] = []
    for case in (dataset or [])[:5]:
        summary.append(
            {
                "id": case.get("id"),
                "question": (case.get("question") or "")[:300],
                "ground_truth": (case.get("ground_truth") or "")[:300],
                "expected_skill": case.get("expected_skill"),
                "expected_script": case.get("expected_script"),
            }
        )
    return summary


def _coerce_severity(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _VALID_CONCLUSION_SEVERITIES:
            return v
    return "warn"


def _coerce_category(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().capitalize()
        if v in _VALID_RECOMMENDATION_CATEGORIES:
            return v
    return "Action"


class InsightsJudge(LLMClient):
    """LLM judge that produces additional conclusions and recommendations."""

    default_model: str = INSIGHTS_JUDGE_MODEL
    default_max_tokens: int | None = INSIGHTS_JUDGE_MAX_TOKENS
    default_temperature: float | None = INSIGHTS_JUDGE_TEMPERATURE

    def get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def create_user_prompt(self, **kwargs: Any) -> str:
        canonical: dict = kwargs.get("canonical") or {}
        deterministic: dict = kwargs.get("deterministic") or {}

        prompt: dict = {
            "skill": {
                "name": canonical.get("skill_name"),
                "best_agent": canonical.get("best_agent"),
                "agents_run": canonical.get("agents_run") or [],
                "verdict": canonical.get("verdict"),
                "overall_score": canonical.get("overall_score"),
                "overall_lift": canonical.get("overall_lift"),
                "runtime_seconds": canonical.get("runtime_seconds"),
            },
            "dimensions": _summarize_dimensions(canonical.get("dimensions") or []),
            "best_agent_evaluators": _summarize_evaluators(canonical.get("evaluators") or {}),
            "trials": _summarize_trials(canonical.get("trials") or []),
            "dataset": _summarize_dataset(canonical.get("dataset") or []),
            "deterministic_observations": {
                "conclusions": [
                    {
                        "title": item.get("title"),
                        "message": item.get("message"),
                        "severity": item.get("severity"),
                    }
                    for item in (deterministic.get("conclusions") or [])
                ],
                "recommendations": list(deterministic.get("suggestions") or []),
            },
        }

        return (
            "Tier 3 evaluation context (already includes deterministic "
            "observations — produce ADDITIONAL insights, do not repeat):\n\n"
            + json.dumps(prompt, indent=2, default=str)
        )

    def parse_response(self, response_text: str, **_kwargs: Any) -> dict:
        content = response_text.strip().lstrip("\ufeff")
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0].strip()
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        conclusions: list[dict] = []
        for item in (data.get("conclusions") or [])[:INSIGHTS_JUDGE_MAX_CONCLUSIONS]:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            message = (item.get("message") or "").strip()
            if not title and not message:
                continue
            conclusions.append(
                {
                    "title": title or "Insight",
                    "message": message or title,
                    "severity": _coerce_severity(item.get("severity")),
                    "source": "llm",
                }
            )

        recommendations: list[dict] = []
        for item in (data.get("recommendations") or [])[:INSIGHTS_JUDGE_MAX_RECOMMENDATIONS]:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            message = (item.get("message") or "").strip()
            if not title and not message:
                continue
            recommendations.append(
                {
                    "title": title or message[:60],
                    "message": message or title,
                    "category": _coerce_category(item.get("category")),
                    "severity": _coerce_severity(item.get("severity")),
                    "source": "llm",
                }
            )

        return {"conclusions": conclusions, "recommendations": recommendations}

    def get_fallback_response(self, **_kwargs: Any) -> dict:
        return {"conclusions": [], "recommendations": []}


def build_insights(
    canonical: dict,
    deterministic: dict,
    *,
    use_llm: bool = True,
) -> dict:
    """Return ``{"conclusions": [...], "recommendations": [...]}`` from the LLM.

    The function never raises: when the LLM is unavailable it simply returns
    empty lists so callers can keep using their deterministic baselines.
    """
    if not use_llm:
        return {"conclusions": [], "recommendations": []}

    judge = InsightsJudge()
    return judge.process(canonical=canonical, deterministic=deterministic)


__all__ = [
    "InsightsJudge",
    "build_insights",
]
