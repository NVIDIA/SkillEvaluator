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
from collections.abc import Collection, Iterator, Mapping
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
# edge into the much shorter visible prefix. It holds whole words only: a word
# cut at the edge is dropped, because no redaction pass can match the prefix of
# a credential, and redacting earlier text can pull that prefix into view.
_MAX_SCAN_CHARS = 16_384
# One character longer than the window, so a word that does not fit is never
# mistaken for one that does.
_WORD_RE = re.compile(rf"\S{{1,{_MAX_SCAN_CHARS + 1}}}")
_LINE_BREAKS = ("\n", "\r")

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
    """Return the whole words of ``value`` that fit in ``_MAX_SCAN_CHARS``, whitespace collapsed.

    A whitespace run that contains a line break becomes one ``\\n`` and any
    other run one space. Line breaks stay because credential assignments such as
    ``token: value`` end at the line: flattening them first would let redaction
    mask the explanation on the next line too.
    """
    parts: list[str] = []
    size = 0
    end = 0
    for match in _WORD_RE.finditer(value):
        separator = ""
        if parts:
            gap = value[end : match.start()]
            separator = "\n" if any(brk in gap for brk in _LINE_BREAKS) else " "
        word = match.group()
        size += len(separator) + len(word)
        if size > _MAX_SCAN_CHARS:
            break
        parts.append(separator + word)
        end = match.end()
    return "".join(parts)


# Command-line credentials, which the shared redaction does not know about
# because it targets ``key=value`` pairs and recognizable token shapes. Grader
# excerpts quote raw command lines, so the report masks these shapes itself.
_CLI_VALUE = r"""(?:"[^"\r\n]*+"|'[^'\r\n]*+'|[^\s"'|;&]++)"""
_CLI_USER_PASSWORD_RE = re.compile(rf"(?<![\w-])(?P<flag>-u|--user)(?P<sep>\s++|=)(?=[^\s]*:)(?P<value>{_CLI_VALUE})")
_CLI_COOKIE_FLAG_RE = re.compile(rf"(?<![\w-])(?P<flag>-b)(?P<sep>\s++)(?=[^\s]*=)(?P<value>{_CLI_VALUE})")
_CLI_SECRET_FLAG_RE = re.compile(
    r"(?i)(?<![\w-])(?P<flag>--[a-z0-9-]*(?:password|passwd|passphrase|token|secret|api-?key|credential|cookie)"
    rf"[a-z0-9-]*)(?P<sep>\s++|=)(?!-)(?P<value>{_CLI_VALUE})"
)
_CLI_SSHPASS_RE = re.compile(rf"(?<![\w-])(?P<flag>sshpass\s++-p\s*+)(?P<value>{_CLI_VALUE})")
# ``-pSECRET`` is attached only for the MySQL family; elsewhere ``-p`` is a flag
# such as ``mkdir -p``.
_CLI_MYSQL_PASSWORD_RE = re.compile(
    r"(?P<head>\b(?:mysql|mysqldump|mysqladmin|mysqlimport|mariadb|mariadb-dump)\b[^|;&\r\n]*?\s)-p(?P<value>[^\s|;&]++)"
)
_COOKIE_HEADER_RE = re.compile(r"(?i)\b(?P<key>(?:set-)?cookie)(?P<sep>\s*+:\s*+)[^\r\n\"']++")


def _redact_command_line_credentials(text: str) -> str:
    """Mask password, token and cookie arguments of quoted shell command lines."""
    text = _CLI_USER_PASSWORD_RE.sub(r"\g<flag>\g<sep><redacted>", text)
    text = _CLI_COOKIE_FLAG_RE.sub(r"\g<flag>\g<sep><redacted>", text)
    text = _CLI_SECRET_FLAG_RE.sub(r"\g<flag>\g<sep><redacted>", text)
    text = _CLI_SSHPASS_RE.sub(r"\g<flag><redacted>", text)
    text = _CLI_MYSQL_PASSWORD_RE.sub(r"\g<head>-p<redacted>", text)
    return _COOKIE_HEADER_RE.sub(r"\g<key>\g<sep><redacted>", text)


def sanitize_grader_text(value: object, max_len: int | None = None) -> str:
    """Collapse whitespace, redact credentials, and optionally clamp grader text.

    Like the Tier 3 findings report, this redacts the exact values of configured
    credentials first and then recognizable credential shapes. Line breaks are
    kept (one per break) so a ``key: value`` assignment is masked only to the end
    of its line; HTML renders them as spaces, and tooltips as line breaks.
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
    text = _redact_command_line_credentials(_redact_configured_credentials(text))
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
    return f"Check `{label}` did not pass; no reason was recorded"


def _reason_candidates(
    detail: dict[str, Any],
    *,
    include_results: bool,
    metric: str | None,
    reward: dict[str, Any] | None,
) -> Iterator[str]:
    """Yield candidate reason strings across the known grader detail shapes."""
    from skillevaluator.tier3.harbor.report import (
        _applicable_accuracy_criteria,
        _expected_observed_suffix,
    )

    for field in ("reason", "message"):
        if isinstance(detail.get(field), str):
            yield detail[field]
    if isinstance(detail.get("expected_observed"), str):
        yield detail["expected_observed"]

    for finding in _detail_list(detail, "findings"):
        if isinstance(finding, str):
            # Legacy and custom graders list findings as plain strings, which
            # the terminal findings report also treats as failure reasons.
            yield finding
        elif _failed(finding):
            # Like the terminal findings report, name the finding type when the
            # grader wrote no message.
            yield str(finding.get("message") or finding.get("reason") or finding.get("type") or "")

    # behavior_check records one entry per expected behavior. Evaluator cards
    # already list these as ``failures``, so they leave them out of the reason.
    for result in _detail_list(detail, "results") if include_results else []:
        if _failed(result):
            yield str(result.get("reason") or "")

    # Rubric graders record a bool per named criterion. Criteria the grader
    # excluded from the score (such as skill identification when no skill is
    # expected) did not fail it.
    criteria = _applicable_accuracy_criteria(detail)
    failed_criteria = [str(name) for name, passed in islice(criteria.items(), _MAX_DETAIL_ITEMS) if passed is False]
    if failed_criteria:
        yield f"Failed criteria: {', '.join(failed_criteria)}"

    if metric in ("accuracy", "goal_accuracy"):
        suffix = _expected_observed_suffix(detail, reward)
        if suffix:
            yield suffix

    # Other graders nest one dict per named check.
    for key, value in islice(detail.items(), _MAX_DETAIL_ITEMS):
        if key not in _STRUCTURED_DETAIL_KEYS and _failed(value):
            yield _failed_check_reason(str(key)[:128], value)

    # Attribution says whether the skill or the environment caused the failure,
    # which is the first thing a reviewer asks after seeing the reason.
    if isinstance(detail.get("attribution_explanation"), str):
        yield detail["attribution_explanation"]


def failure_reason(
    detail: object,
    *,
    include_results: bool = True,
    metric: str | None = None,
    reward: dict[str, Any] | None = None,
    omit: Collection[str] = (),
) -> str:
    """Return one short, deduplicated, redacted reason for a failing metric.

    ``include_results=False`` leaves out failing behavior results, for evaluator
    cards that already show them as ``failures``.
    """
    if not isinstance(detail, dict) and reward is not None and metric:
        details = reward.get("details")
        detail = details.get(metric) if isinstance(details, Mapping) else None
    if not isinstance(detail, dict):
        return ""
    if metric in {"accuracy", "goal_accuracy"}:
        observed_pair = detail.get("expected_observed")
        observed_pair = observed_pair if isinstance(observed_pair, Mapping) else detail
        expected = observed_pair.get("expected")
        observed = observed_pair.get("observed")
        if isinstance(expected, str) and expected.strip() and isinstance(observed, str) and observed.strip():
            detail = {**detail, "expected_observed": f"Expected: {expected}; observed: {observed}"}
    parts: list[str] = []
    omitted_texts: set[str] = set()
    for text in omit:
        cleaned = sanitize_grader_text(text)
        if cleaned:
            omitted_texts.add(cleaned)
            omitted_texts.update(part for part in cleaned.split("; ") if part)
    # Only non-blank candidates count toward the cap, so entries without text
    # cannot crowd out the criteria, checks, or attribution that follow them.
    candidates = (
        cleaned
        for candidate in _reason_candidates(
            detail,
            include_results=include_results,
            metric=metric,
            reward=reward,
        )
        if (cleaned := sanitize_grader_text(candidate))
        and not any(cleaned == omitted or cleaned in omitted for omitted in omitted_texts)
    )
    for cleaned in islice(candidates, _MAX_REASON_CANDIDATES):
        if cleaned and any(cleaned == text or cleaned in text for text in omitted_texts):
            continue
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
        if len(parts) >= _MAX_REASON_PARTS:
            break
    # The parts are already redacted, so clipping them before the final pass
    # cannot expose a credential, and the final pass stays short.
    return sanitize_grader_text("; ".join(parts)[: 4 * MAX_REASON_CHARS], MAX_REASON_CHARS)


def evidence_unavailable_reason(reward: dict[str, Any]) -> str:
    failure = sanitize_grader_text(reward.get("trajectory_detail") or reward.get("error"), MAX_REASON_CHARS)
    if failure:
        return f"Trajectory evidence could not be loaded: {failure}"
    return "No trajectory evidence was recorded for this metric"


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


def failure_reason_text_slots(
    payload: dict[str, Any],
    *,
    include_top_level_cards: bool = True,
) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield each ``(container, key)`` that holds a built failure reason or excerpt.

    Order follows the report: evaluator-card evidence per agent, then the
    drill-down rows (with-skill rows of an agent, then its baseline rows), then
    the top-level card copy, which the HTML page does not render
    (``include_top_level_cards=False`` leaves it out). Size limits drop these
    first and the HTML preview restores them in this order while room remains.
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
    if include_top_level_cards:
        yield from card_slots(payload.get("evaluator_cards"))


def strip_failure_reason_text(payload: dict[str, Any]) -> int:
    """Remove built failure reasons and excerpts in place; return how many were removed."""
    removed = 0
    for container, key in list(failure_reason_text_slots(payload)):
        # The top-level card copy can share entries with the best agent's cards.
        if container.pop(key, None) is not None:
            removed += 1
    return removed
