# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded failure reasons for Tier 3 evaluator scores.

Harbor graders record why a metric failed in ``reward["details"][metric]``: a
``reason`` or ``message`` string, per-finding ``message`` values, failing
behavior results, failed rubric criteria, and short ``evidence`` excerpts. This
module projects those details into the compact text the HTML report shows next
to a failing score, so a ``0.00`` cell or evaluator card explains itself without
opening the raw trajectory.

Redaction and length caps happen here rather than in the template because the
Tier 3 payload is embedded in every report, so unbounded grader text would both
inflate the artifact and could carry credentials an agent echoed.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from itertools import islice
from typing import Any

from skillevaluator.utils.redaction import redact_sensitive_text

# A per-trial metric score below this gate carries a reason. It matches the
# evaluator card's pass gate, so passing metrics add nothing to the payload.
TRIAL_PASS_THRESHOLD = 0.8

MAX_REASON_CHARS = 240
MAX_EXCERPT_CHARS = 300
_MAX_REASON_PARTS = 3
# Grader detail lists and nested checks are scanned up to this many entries;
# the first few failures already explain the score.
_MAX_DETAIL_ITEMS = 64
# At most this many candidate strings are redacted per metric. A reason keeps
# three distinct parts, and redaction dominates report build time when a grader
# records dozens of findings for every failing trial.
_MAX_REASON_CANDIDATES = 8
# Grader text is collapsed and clipped to this window before redaction so one
# oversized string cannot stall report generation. The window is measured after
# whitespace collapse, so padding cannot pull a credential cut at the window
# edge into the much shorter visible prefix.
_MAX_SCAN_CHARS = 16_384
_WORD_RE = re.compile(rf"\S{{1,{_MAX_SCAN_CHARS}}}")

# Keys whose values are handled by a dedicated branch and must not be re-read by
# the generic per-check sweep.
_STRUCTURED_DETAIL_KEYS = frozenset({"findings", "results", "criteria"})

# These findings store the detecting pattern or a fixed placeholder in
# ``evidence`` rather than the matched text, so quoting it back explains nothing.
_OPAQUE_EVIDENCE_TYPES = frozenset({"secret_leak", "secret_exposure"})

# Evaluator-card evidence keys that hold text built here rather than recorded by
# the grader. Trials carry theirs in ``score_reasons``.
FAILURE_REASON_EVIDENCE_KEYS = ("reason", "excerpt")


def _collapsed_prefix(value: str) -> str:
    """Return at most ``_MAX_SCAN_CHARS`` of ``value`` with whitespace runs collapsed."""
    words: list[str] = []
    size = 0
    for match in _WORD_RE.finditer(value):
        words.append(match.group())
        size += len(words[-1]) + 1
        if size > _MAX_SCAN_CHARS:
            break
    return " ".join(words)[:_MAX_SCAN_CHARS]


def sanitize_grader_text(value: object, max_len: int | None = None) -> str:
    """Collapse whitespace, redact credentials, and optionally clamp grader text.

    Like the Tier 3 findings report, this redacts the exact values of configured
    credentials first and then recognizable credential shapes.
    """
    # Imported here, like the other Tier 3 helpers of the report builders, so
    # importing the reporters does not load the Harbor and judge modules.
    from skillevaluator.tier3.eval_core.llm_judge import _redact_configured_credentials
    from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line

    if not isinstance(value, str):
        return ""
    text = _collapsed_prefix(value)
    if not text:
        return ""
    text = _redact_configured_credentials(text)
    return redact_sensitive_text(redact_secrets_in_log_line(text), max_len=max_len)


def _failed(entry: object) -> bool:
    """True when a grader sub-result explicitly records a failure."""
    if not isinstance(entry, dict):
        return False
    if entry.get("score_impact") is True:
        return True
    return entry.get("passed") is False


def _detail_list(detail: dict[str, Any], key: str) -> list[Any]:
    value = detail.get(key)
    return list(islice(value, _MAX_DETAIL_ITEMS)) if isinstance(value, list) else []


def _failed_check_reason(key: str, value: dict[str, Any]) -> str:
    """Prefer a grader-written reason over the internal check key."""
    for field in ("reason", "message"):
        text = value.get(field)
        if isinstance(text, str) and text.strip():
            return text
    label = key.replace("_", " ").replace("-", " ").strip()
    if not label:
        return "Check failed"
    return f"{label[0].upper()}{label[1:]} failed"


def _reason_candidates(detail: dict[str, Any]) -> Iterator[str]:
    """Yield candidate reason strings across the known grader detail shapes."""
    from skillevaluator.tier3.harbor.report import _applicable_accuracy_criteria

    for field in ("reason", "message"):
        if isinstance(detail.get(field), str):
            yield detail[field]

    for finding in _detail_list(detail, "findings"):
        if isinstance(finding, str):
            # Legacy and custom graders list findings as plain strings, which
            # the terminal findings report also treats as failure reasons.
            yield finding
        elif _failed(finding):
            yield str(finding.get("message") or finding.get("reason") or "")

    # behavior_check records one entry per expected behavior.
    for result in _detail_list(detail, "results"):
        if _failed(result):
            yield str(result.get("reason") or "")

    # Rubric graders record a bool per named criterion. Criteria the grader
    # excluded from the score (such as skill identification when no skill is
    # expected) did not fail it.
    criteria = _applicable_accuracy_criteria(detail)
    failed_criteria = sorted(
        str(name)[:128] for name, passed in islice(criteria.items(), _MAX_DETAIL_ITEMS) if passed is False
    )
    if failed_criteria:
        yield f"Failed criteria: {', '.join(failed_criteria)}"

    # Other graders nest one dict per named check.
    for key, value in islice(detail.items(), _MAX_DETAIL_ITEMS):
        if key not in _STRUCTURED_DETAIL_KEYS and _failed(value):
            yield _failed_check_reason(str(key)[:128], value)

    # Attribution says whether the skill or the environment caused the failure,
    # which is the first thing a reviewer asks after seeing the reason.
    if isinstance(detail.get("attribution_explanation"), str):
        yield detail["attribution_explanation"]


def failure_reason(detail: object) -> str:
    """Return one short, deduplicated, redacted reason for a failing metric."""
    if not isinstance(detail, dict):
        return ""
    parts: list[str] = []
    for candidate in islice(_reason_candidates(detail), _MAX_REASON_CANDIDATES):
        cleaned = sanitize_grader_text(candidate)
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
        if len(parts) >= _MAX_REASON_PARTS:
            break
    # The parts are already redacted, so clipping them before the final pass
    # cannot expose a credential, and the final pass stays short.
    return sanitize_grader_text("; ".join(parts)[: 4 * MAX_REASON_CHARS], MAX_REASON_CHARS)


def failure_excerpt(detail: object) -> str:
    """Return a short, redacted excerpt of the artifact that triggered the failure."""
    if not isinstance(detail, dict):
        return ""
    for finding in _detail_list(detail, "findings"):
        if not _failed(finding) or finding.get("type") in _OPAQUE_EVIDENCE_TYPES:
            continue
        excerpt = sanitize_grader_text(finding.get("evidence"), MAX_EXCERPT_CHARS)
        if excerpt:
            return excerpt
    return sanitize_grader_text(detail.get("evidence"), MAX_EXCERPT_CHARS)


def failure_reason_text_slots(payload: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield each ``(container, key)`` that holds a built failure reason or excerpt.

    Order follows the report: evaluator-card evidence per agent, then the
    drill-down rows (with-skill rows of an agent, then its baseline rows), then
    the top-level card copy. Size limits drop these first and the HTML preview
    restores them in this order while room remains.
    """
    agents = payload.get("agents") if isinstance(payload.get("agents"), dict) else {}
    agent_items = [(str(name), agent) for name, agent in agents.items() if isinstance(agent, dict)]

    def card_slots(cards: object) -> Iterator[tuple[dict[str, Any], str]]:
        for card in cards if isinstance(cards, list) else []:
            evidence = card.get("evidence") if isinstance(card, dict) else None
            for entry in evidence if isinstance(evidence, list) else []:
                if isinstance(entry, dict):
                    for key in FAILURE_REASON_EVIDENCE_KEYS:
                        if key in entry:
                            yield entry, key

    for _name, agent in agent_items:
        yield from card_slots(agent.get("evaluator_cards"))
    trials = payload.get("trials") if isinstance(payload.get("trials"), list) else []
    trials_by_agent: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        if isinstance(trial, dict):
            trials_by_agent.setdefault(str(trial.get("agent") or ""), []).append(trial)
    for name, agent in [*agent_items, *((name, {}) for name in trials_by_agent if name not in agents)]:
        baseline = agent.get("trials_baseline") if isinstance(agent.get("trials_baseline"), list) else []
        for trial in [*trials_by_agent.get(name, []), *baseline]:
            if isinstance(trial, dict) and "score_reasons" in trial:
                yield trial, "score_reasons"
    yield from card_slots(payload.get("evaluator_cards"))


def strip_failure_reason_text(payload: dict[str, Any]) -> int:
    """Remove built failure reasons and excerpts in place; return how many were removed."""
    removed = 0
    for container, key in list(failure_reason_text_slots(payload)):
        # The top-level card copy can share entries with the best agent's cards.
        if container.pop(key, None) is not None:
            removed += 1
    return removed
