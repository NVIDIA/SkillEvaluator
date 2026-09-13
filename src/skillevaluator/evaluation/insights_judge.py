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
import math
import re
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
_VALID_CLAIM_TYPES = {
    "general",
    "negative_control_failure",
    "dataset_case_removal",
}
_MAX_GROUNDED_CLAIM_TEXT_CHARS = 2_000
_NEGATIVE_CONTROL_SEMANTIC_PATTERNS = (
    re.compile(r"\bnegative[\s-]*control\b", re.IGNORECASE),
    re.compile(r"\b(?:scope[\s-]*(?:drift|misalignment)|capability[\s-]*overreach)\b", re.IGNORECASE),
    re.compile(r"\b(?:unintended|unnecessary|unnecessarily)\b.{0,48}\b(?:skill|capability)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:skill|capability)\b.{0,48}\b(?:over[\s-]*(?:applied|used)|overreach|overfitting)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:unrelated|out[\s-]*of[\s-]*scope)\b.{0,48}\b(?:work|task|request|skill|capability)\b", re.IGNORECASE
    ),
)


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

NEGATIVE-CONTROL INTERPRETATION:
- A dataset case whose expected_skill is null is an intentional negative
  control: the agent should complete the unrelated task without activating
  the evaluated skill.
- An explicit should_trigger value takes precedence over expected_skill, just
  as it does in the runtime grader. Otherwise, a falsey expected_skill marks a
  negative control; a case with neither routing field is unlabeled.
- A successful negative-control case is correct routing behavior. Do not infer
  unintended skill application, scope misalignment, or overfitting merely
  because the agent answered the unrelated task correctly.
- Report unintended skill application only when explicit execution evidence
  shows that the evaluated skill was read or invoked, or routing evidence for
  that case failed. A passing routing result is evidence against that warning;
  when invocation evidence is absent or ambiguous, do not issue the warning.
- Do not recommend removing an unrelated negative-control case merely because
  it is outside the skill's scope; that is what the case is designed to test.
- Every conclusion and recommendation MUST include a claim_type and a
  non-empty evidence_case_ids list containing 1-10 unique known case ids. The
  title and message together must name every evidence case id and no other case
  id. General claims may cite only skill-activation cases. Use claim_type
  "general" only for observations that do not concern negative-control routing
  or removal of a dataset case. Use
  claim_type "negative_control_failure" for unintended skill use, scope
  misalignment, or overfitting involving a negative control. Use claim_type
  "dataset_case_removal" for any recommendation to remove a case from the
  dataset.
- A negative_control_failure claim MUST identify every affected case in
  evidence_case_ids and is valid only when the supplied context shows failed
  routing or explicit invocation evidence for every identified negative-control
  case. A dataset_case_removal claim must not cite a negative control. Do not
  label a negative-control claim as general.

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
    {{\"claim_type\": \"general|negative_control_failure\", \"title\": \"<short title>\", \"message\": \"<1-2 sentences>\", \"severity\": \"pass|warn|fail\", \"evidence_case_ids\": [\"<case id>\"]}}
  ],
  \"recommendations\": [
    {{\"claim_type\": \"general|negative_control_failure|dataset_case_removal\", \"category\": \"<one of the allowed categories>\", \"title\": \"<short title>\", \"message\": \"<one imperative sentence>\", \"severity\": \"pass|warn|fail\", \"evidence_case_ids\": [\"<case id>\"]}}
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


def _case_routing(case: dict) -> tuple[str, bool | None]:
    """Mirror the runtime grader's routing precedence for prompt metadata."""
    if "should_trigger" in case:
        should_trigger = bool(case.get("should_trigger"))
    elif "expected_skill" in case:
        should_trigger = bool(case.get("expected_skill"))
    else:
        return "unlabeled", None
    return ("skill_activation", True) if should_trigger else ("negative_control", False)


def _summarize_case(case: dict) -> dict:
    case_type, skill_expected_to_activate = _case_routing(case)
    expected_behavior = case.get("expected_behavior")
    if isinstance(expected_behavior, list):
        expected_behavior = [str(item)[:300] for item in expected_behavior[:5]]
    elif isinstance(expected_behavior, str):
        expected_behavior = expected_behavior[:300]
    else:
        expected_behavior = None

    return {
        "id": case.get("id"),
        "case_type": case_type,
        "skill_expected_to_activate": skill_expected_to_activate,
        "question": (case.get("question") or "")[:300],
        "ground_truth": (case.get("ground_truth") or "")[:300],
        "expected_skill": case.get("expected_skill"),
        "should_trigger": case.get("should_trigger") if "should_trigger" in case else None,
        "expected_script": case.get("expected_script"),
        "expected_behavior": expected_behavior,
    }


def _case_lookup(dataset: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for case in dataset or []:
        if not isinstance(case, dict) or case.get("id") is None:
            continue
        lookup.setdefault(str(case["id"]), case)
    return lookup


def _summarize_trials(trials: list[dict], cases_by_id: dict[str, dict] | None = None) -> list[dict]:
    """Pick a representative slice without collapsing agents or attempts."""
    if not trials:
        return []
    cases_by_id = cases_by_id or {}

    def _key(indexed_trial: tuple[int, dict]) -> tuple[float, int]:
        index, trial = indexed_trial
        overall = trial.get("overall")
        score = overall if isinstance(overall, (int, float)) else 0.5
        return score, index

    indexed_trials = [(index, trial) for index, trial in enumerate(trials) if isinstance(trial, dict)]
    sorted_trials = sorted(indexed_trials, key=_key)

    def _has_trusted_negative_failure(indexed_trial: tuple[int, dict]) -> bool:
        _index, trial = indexed_trial
        entry_id = trial.get("entry_id")
        case = cases_by_id.get(str(entry_id)) if entry_id is not None else None
        if not isinstance(case, dict) or _case_routing(case)[1] is not False:
            return False
        if trial.get("invocation_evidence_source") != "trajectory":
            return False
        skill_invoked = trial.get("skill_invoked") if type(trial.get("skill_invoked")) is bool else None
        routing_passed = trial.get("routing_passed") if type(trial.get("routing_passed")) is bool else None
        if skill_invoked is not None and routing_passed is not None:
            return routing_passed is (not skill_invoked) and skill_invoked
        if skill_invoked is not None:
            return skill_invoked
        return routing_passed is False

    priority = [trial for trial in indexed_trials if _has_trusted_negative_failure(trial)]
    chosen = priority + sorted_trials[:3] + sorted_trials[-3:]

    attempt_counts: dict[tuple[str, str], int] = {}
    attempts_by_index: dict[int, int | str] = {}
    for index, trial in indexed_trials:
        identity = (str(trial.get("agent") or ""), str(trial.get("entry_id") or ""))
        attempt_counts[identity] = attempt_counts.get(identity, 0) + 1
        attempts_by_index[index] = trial.get("attempt") or attempt_counts[identity]

    seen_indices: set[int] = set()
    compact: list[dict] = []
    for index, trial in chosen:
        if index in seen_indices:
            continue
        seen_indices.add(index)
        scores = trial.get("scores") or {}
        entry_id = trial.get("entry_id")
        case = cases_by_id.get(str(entry_id)) if entry_id is not None else None
        entry = {
            "agent": trial.get("agent"),
            "trial_id": trial.get("trial_id"),
            "entry_id": entry_id,
            "attempt": attempts_by_index[index],
            "overall": trial.get("overall"),
            "scores": {k: v for k, v in scores.items() if v is not None},
            "baseline_overall": trial.get("baseline_overall"),
            "lift_scores": trial.get("lift_scores") or {},
            "warnings": (trial.get("warnings") or [])[:3],
            "error_recovery": trial.get("error_recovery") or {},
            "case": _summarize_case(case) if case is not None else None,
        }
        if trial.get("invocation_evidence_source") == "trajectory":
            for key in ("skill_invoked", "routing_passed"):
                if type(trial.get(key)) is bool:
                    entry[key] = trial[key]
            if "skill_invoked" in entry or "routing_passed" in entry:
                entry["invocation_evidence_source"] = "trajectory"
        compact.append(entry)
        if len(compact) >= 6:
            break
    return compact


def _summarize_dataset(dataset: list[dict]) -> list[dict]:
    """Trim dataset cases to the question + ground truth (plus id)."""
    return [_summarize_case(case) for case in (dataset or [])[:5] if isinstance(case, dict)]


def _dataset_for_sampled_trials(trials: list[dict], dataset: list[dict]) -> list[dict]:
    """Keep one copy of every case joined to a sampled trial, in trial order."""
    summary: list[dict] = []
    seen: set[str] = set()
    for trial in trials:
        case = trial.get("case")
        if not isinstance(case, dict) or case.get("id") is None:
            continue
        case_id = str(case["id"])
        if case_id in seen:
            continue
        seen.add(case_id)
        summary.append(case)
    return summary or _summarize_dataset(dataset)


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


def _strict_bool(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _finite_numeric(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) else None


def _trial_negative_control_failure(trial: dict) -> bool | None:
    """Interpret one trial, preferring strict invocation evidence over legacy scores."""
    if trial.get("invocation_evidence_source") == "trajectory":
        skill_invoked = _strict_bool(trial.get("skill_invoked"))
        routing_passed = _strict_bool(trial.get("routing_passed"))
        if skill_invoked is not None and routing_passed is not None:
            if routing_passed is not (not skill_invoked):
                return None
            return skill_invoked
        if skill_invoked is not None:
            return skill_invoked
        if routing_passed is not None:
            return not routing_passed

    scores = trial.get("scores") if isinstance(trial.get("scores"), dict) else {}
    numeric_scores = [
        numeric
        for value in (
            scores.get("skill_execution"),
            scores.get("skill_routing"),
            trial.get("skill_execution"),
            trial.get("skill_routing"),
        )
        if (numeric := _finite_numeric(value)) is not None
    ]
    if not numeric_scores:
        return None
    return any(score < 1.0 for score in numeric_scores)


def _negative_control_grounding(canonical: dict) -> tuple[dict[str, bool | None], dict[str, str]]:
    """Return per-case routing type and grounded negative-control failure state."""
    case_types: dict[str, str] = {}
    trial_states: dict[str, list[bool | None]] = {}
    for case in canonical.get("dataset") or []:
        if not isinstance(case, dict) or case.get("id") is None:
            continue
        case_id = str(case["id"])
        case_type, should_trigger = _case_routing(case)
        case_types[case_id] = case_type
        if should_trigger is False:
            trial_states[case_id] = []

    for trial in canonical.get("trials") or []:
        if not isinstance(trial, dict) or trial.get("entry_id") is None:
            continue
        case_id = str(trial["entry_id"])
        if case_id not in trial_states:
            continue
        trial_states[case_id].append(_trial_negative_control_failure(trial))

    evidence: dict[str, bool | None] = {}
    for case_id, states in trial_states.items():
        if any(state is True for state in states):
            evidence[case_id] = True
        elif not states or any(state is None for state in states):
            evidence[case_id] = None
        else:
            evidence[case_id] = False
    return evidence, case_types


def _declared_case_ids(item: dict) -> list[str] | None:
    raw_references = item.get("evidence_case_ids")
    if not isinstance(raw_references, list) or not 1 <= len(raw_references) <= 10:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in raw_references):
        return None
    references = [value.strip() for value in raw_references]
    return references if len(references) == len(set(references)) else None


def _claim_text(item: dict) -> str | None:
    text = f"{item.get('title') or ''} {item.get('message') or ''}"
    return text if len(text) <= _MAX_GROUNDED_CLAIM_TEXT_CHARS else None


def _text_case_ids(text: str, known_case_ids: set[str]) -> set[str]:
    references: set[str] = set()

    for case_id in known_case_ids:
        if re.search(rf"(?<![\w-]){re.escape(case_id)}(?![\w-])", text, re.IGNORECASE):
            references.add(case_id)
    return references


def _has_negative_control_semantics(text: str) -> bool:
    return any(pattern.search(text) for pattern in _NEGATIVE_CONTROL_SEMANTIC_PATTERNS)


def _claim_is_grounded(
    item: dict,
    grounding: tuple[dict[str, bool | None], dict[str, str]],
) -> bool:
    evidence, case_types = grounding
    claim_type = item.get("claim_type")
    if claim_type not in _VALID_CLAIM_TYPES:
        return False

    declared_case_ids = _declared_case_ids(item)
    if declared_case_ids is None:
        return False
    declared_set = set(declared_case_ids)
    known_case_ids = set(case_types)
    if not declared_set.issubset(known_case_ids):
        return False
    claim_text = _claim_text(item)
    if claim_text is None or _text_case_ids(claim_text, known_case_ids) != declared_set:
        return False

    declared_types = [case_types[case_id] for case_id in declared_case_ids]
    if claim_type == "general":
        return not _has_negative_control_semantics(claim_text) and all(
            case_type == "skill_activation" for case_type in declared_types
        )
    if claim_type == "dataset_case_removal":
        return all(case_type != "negative_control" for case_type in declared_types)

    return all(case_type == "negative_control" for case_type in declared_types) and all(
        evidence.get(case_id) is True for case_id in declared_case_ids
    )


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
        dataset = canonical.get("dataset") or []
        trials = _summarize_trials(canonical.get("trials") or [], _case_lookup(dataset))

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
            "trials": trials,
            "dataset": _dataset_for_sampled_trials(trials, dataset),
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

    def parse_response(self, response_text: str, **kwargs: Any) -> dict:
        content = response_text.strip().lstrip("\ufeff")
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0].strip()
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")
        canonical = kwargs.get("canonical") if isinstance(kwargs.get("canonical"), dict) else {}
        grounding = _negative_control_grounding(canonical)

        conclusions: list[dict] = []
        for item in (data.get("conclusions") or [])[:INSIGHTS_JUDGE_MAX_CONCLUSIONS]:
            if not isinstance(item, dict):
                continue
            if not _claim_is_grounded(item, grounding):
                continue
            title = (item.get("title") or "").strip()
            message = (item.get("message") or "").strip()
            if not title and not message:
                continue
            conclusions.append(
                {
                    "claim_type": item["claim_type"],
                    "title": title or "Insight",
                    "message": message or title,
                    "severity": _coerce_severity(item.get("severity")),
                    "source": "llm",
                    "evidence_case_ids": _declared_case_ids(item),
                }
            )

        recommendations: list[dict] = []
        for item in (data.get("recommendations") or [])[:INSIGHTS_JUDGE_MAX_RECOMMENDATIONS]:
            if not isinstance(item, dict):
                continue
            if not _claim_is_grounded(item, grounding):
                continue
            title = (item.get("title") or "").strip()
            message = (item.get("message") or "").strip()
            if not title and not message:
                continue
            recommendations.append(
                {
                    "claim_type": item["claim_type"],
                    "title": title or message[:60],
                    "message": message or title,
                    "category": _coerce_category(item.get("category")),
                    "severity": _coerce_severity(item.get("severity")),
                    "source": "llm",
                    "evidence_case_ids": _declared_case_ids(item),
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
