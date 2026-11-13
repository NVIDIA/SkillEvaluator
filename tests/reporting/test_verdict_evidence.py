# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The full report must show the evidence and policy that decided the verdict.

Issue #192: a card declared ``Overall verdict: PASS`` while one agent carried
no dimension scores and another sat below the documented FAIL boundary, and the
report displayed nothing that explained the decision.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
from tests.golden.test_benchmark_card import _deterministic_results, _live_tier3_result

from skillevaluator.constants import DIMENSION_MAPPING
from skillevaluator.evaluation.tier3_report import advisory_skip_result
from skillevaluator.reporting import BenchmarkReporter
from skillevaluator.reporting.base import agent_eval_dimension_breakdown, agent_eval_dimension_verdict
from skillevaluator.reporting.benchmark import _overall_status, _publication_assessment

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

_EVIDENCE_HEADING = "### Why This Verdict"


def _dimensions(baseline: float, with_skill: float, overrides: dict[str, float] | None = None) -> list[dict]:
    rows = []
    for dimension in DIMENSION_MAPPING:
        score = (overrides or {}).get(dimension, with_skill)
        rows.append({"id": dimension, "baseline": baseline, "with_skill": score, "lift": score - baseline})
    return rows


def _succeeded_conditions() -> dict[str, dict[str, Any]]:
    return {
        arm: {
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 24,
            "scored_attempts": 24,
        }
        for arm in ("with_skill", "without_skill")
    }


def _issue_192_results() -> list[ValidationResult]:
    """Build the run issue #192 reports, as current ``main`` can produce it.

    ``claude-code`` is the agent whose dimension scores were withheld, so the
    gate reads nothing from it. ``codex`` scores Security below the FAIL
    boundary. ``gemini-cli`` passes every dimension, which is what the gate
    actually answers, so the run is still a PASS.
    """
    results = [*deepcopy(_deterministic_results()), deepcopy(_live_tier3_result())]
    payload = results[-1].metadata["agent_eval"]
    agents = payload["agents"]

    claude = agents["claude-code"]
    for field in ("dimensions", "baseline", "with_skill", "lift"):
        claude.pop(field, None)

    codex_dimensions = _dimensions(0.55, 0.88, {"security": 0.238})
    codex = agents["codex"]
    codex["dimensions"] = codex_dimensions
    codex["baseline"] = 0.55
    codex["with_skill"] = sum(row["with_skill"] for row in codex_dimensions) / len(codex_dimensions)
    codex["lift"] = codex["with_skill"] - 0.55

    gemini_dimensions = _dimensions(0.5, 0.9)
    agents["gemini-cli"] = {
        "model": "gemini-pro",
        "execution_status": "succeeded",
        "execution_errors": [],
        "expected_attempts": 48,
        "scored_attempts": 48,
        "conditions": _succeeded_conditions(),
        "baseline": 0.5,
        "with_skill": 0.9,
        "lift": 0.4,
        "dimensions": gemini_dimensions,
        "evaluators": {"accuracy": {"baseline": 0.5, "with_skill": 0.9}},
    }

    for container in (payload, payload["summary"]):
        container["best_agent"] = "gemini-cli"
        container["agents_run"] = ["claude-code", "codex", "gemini-cli"]
        container["overall_score"] = 0.9
        container["overall_lift"] = 0.4
        container["expected_attempts"] = 144
        container["scored_attempts"] = 144
    payload["dimensions"] = gemini_dimensions
    return results


def _render(results: list[ValidationResult]) -> str:
    return BenchmarkReporter(include_timestamp=True).render_all(results)


def _evidence_block(rendered: str) -> str:
    assert _EVIDENCE_HEADING in rendered, rendered[:400]
    block = rendered.partition(_EVIDENCE_HEADING)[2]
    return block.partition("\n## ")[0]


def test_issue_192_pass_shows_the_agent_and_dimensions_that_decided_it() -> None:
    results = _issue_192_results()
    rendered = _render(results)

    assert "Overall verdict: PASS" in rendered
    block = _evidence_block(rendered)
    # The rule that fired, not a restatement of the label.
    assert "- Decision: **PASS**" in block
    assert "Tier 3 dimension gate passed" in block
    # The agent that satisfied the gate, and the one that did not.
    assert "| Gemini Cli | pass |" in block
    assert "| Codex | fail |" in block
    # The dimension the reporter saw below the documented FAIL boundary.
    assert "Security 23.8% below the 40% FAIL boundary" in block
    # The agent with no dimension scores is named, not silently absent.
    assert "| Claude Code | not counted |" in block
    assert "no complete dimension set was recorded" in block
    # Why a failing agent does not contradict the PASS.
    assert "A failing agent does not overturn this gate" in block


def test_issue_192_evidence_block_precedes_the_metadata_section() -> None:
    rendered = _render(_issue_192_results())
    assert rendered.index(_EVIDENCE_HEADING) < rendered.index("## Evaluation Metadata")


def test_excluded_agent_arm_reason_is_rendered() -> None:
    results = _issue_192_results()
    payload = results[-1].metadata["agent_eval"]
    payload["agents"]["claude-code"]["conditions"]["without_skill"] = {
        "execution_status": "failed",
        "execution_errors": ["trial-7: AgentTimeoutError: deadline exceeded"],
        "expected_attempts": 24,
        "scored_attempts": 0,
    }

    block = _evidence_block(_render(results))
    assert "| Claude Code | not counted |" in block
    assert "baseline unavailable (0 of 24 attempts scored)" in block


def test_failing_agent_alone_renders_a_fail_with_its_blocking_reason() -> None:
    results = _issue_192_results()
    payload = results[-1].metadata["agent_eval"]
    del payload["agents"]["gemini-cli"]
    for container in (payload, payload["summary"]):
        container["best_agent"] = "codex"
        container["agents_run"] = ["claude-code", "codex"]
        container["verdict"] = "fail"
        container["expected_attempts"] = 96
        container["scored_attempts"] = 96
    payload["dimensions"] = payload["agents"]["codex"]["dimensions"]

    rendered = _render(results)
    assert "Overall verdict: FAIL" in rendered
    block = _evidence_block(rendered)
    assert "- Decision: **FAIL**" in block
    assert "- Blocking: " in block
    assert "| Codex | fail |" in block
    assert "Security 23.8% below the 40% FAIL boundary" in block
    assert "A failing agent does not overturn this gate" not in block


def test_neutral_verdict_names_the_dimension_below_the_pass_boundary() -> None:
    results = _issue_192_results()
    payload = results[-1].metadata["agent_eval"]
    del payload["agents"]["claude-code"]
    del payload["agents"]["codex"]
    neutral_dimensions = _dimensions(0.4, 0.9, {"efficiency": 0.45})
    gemini = payload["agents"]["gemini-cli"]
    gemini["dimensions"] = neutral_dimensions
    gemini["baseline"] = 0.4
    gemini["with_skill"] = sum(row["with_skill"] for row in neutral_dimensions) / len(neutral_dimensions)
    gemini["lift"] = gemini["with_skill"] - 0.4
    for container in (payload, payload["summary"]):
        container["best_agent"] = "gemini-cli"
        container["agents_run"] = ["gemini-cli"]
        container["verdict"] = "neutral"
        container["overall_score"] = gemini["with_skill"]
        container["overall_lift"] = gemini["lift"]
        container["expected_attempts"] = 48
        container["scored_attempts"] = 48
    payload["dimensions"] = neutral_dimensions

    rendered = _render(results)
    assert "Overall verdict: NEUTRAL" in rendered
    block = _evidence_block(rendered)
    assert "- Decision: **NEUTRAL**" in block
    assert "Efficiency 45.0% below the 50% PASS boundary" in block


def test_incomplete_verdict_lists_every_recorded_reason() -> None:
    results = _issue_192_results()
    payload = results[-1].metadata["agent_eval"]
    payload["agents"]["gemini-cli"]["conditions"]["without_skill"]["scored_attempts"] = 11

    rendered = _render(results)
    assert "Overall verdict: INCOMPLETE" in rendered
    block = _evidence_block(rendered)
    assert "- Decision: **INCOMPLETE**" in block
    assert "- Blocking: " in block
    assert "not publication-complete" in block


def test_skipped_tier3_is_named_with_its_publication_policy() -> None:
    results = [
        *deepcopy(_deterministic_results()),
        advisory_skip_result("Live evaluation runtime unavailable", skill_name="demo-skill"),
    ]

    block = _evidence_block(_render(results))
    assert "- Tier 3 live evaluation: skipped" in block
    assert "no dimension evidence was recorded" in block
    # The skip is advisory, so the card must still say what the policy asked for.
    assert "optional by policy" in block or "required for publication" in block


@pytest.mark.parametrize("tier3_state", ["absent", "skipped", "skipped-fail", "skipped-neutral", "skipped-unscored"])
def test_optional_tier3_without_a_passing_gate_does_not_claim_one(tier3_state: str) -> None:
    results = deepcopy(_deterministic_results())
    results[0].metadata["benchmark_policy"] = {"tier3_required": False}
    if tier3_state != "absent":
        skipped = advisory_skip_result("Live evaluation intentionally omitted", skill_name="demo-skill")
        if tier3_state != "skipped":
            agent = deepcopy(_issue_192_results()[-1].metadata["agent_eval"]["agents"]["codex"])
            if tier3_state == "skipped-neutral":
                agent["dimensions"] = _dimensions(0.55, 0.45)
                agent["with_skill"] = 0.45
                agent["lift"] = -0.1
            elif tier3_state == "skipped-unscored":
                agent.pop("dimensions")
            skipped.metadata["agent_eval"]["agents"] = {"codex": agent}
        results.append(skipped)

    rendered = _render(results)
    block = _evidence_block(rendered)

    assert "Overall verdict: PASS" in rendered
    assert "optional by policy" in block
    assert "Tier 3 dimension gate passed" not in block
    assert "and one did" not in block


def test_recorded_neutral_verdict_is_explained_when_every_agent_passes_the_dimension_gate() -> None:
    results = [*deepcopy(_deterministic_results()), deepcopy(_live_tier3_result())]
    payload = results[-1].metadata["agent_eval"]
    for container in (payload, payload["summary"]):
        container["verdict"] = "neutral"

    rendered = _render(results)
    block = _evidence_block(rendered)

    assert "Overall verdict: NEUTRAL" in rendered
    assert "| Claude Code | pass |" in block
    assert "| Codex | pass |" in block
    assert "recorded Tier 3 verdict is neutral" in block
    assert "no supported agent scored" not in block
    assert "One or more dimensions remain below PASS" not in rendered


def test_a_recorded_pass_without_passing_dimensions_does_not_invent_an_optional_policy() -> None:
    results = [*deepcopy(_deterministic_results()), deepcopy(_live_tier3_result())]
    for agent in results[-1].metadata["agent_eval"]["agents"].values():
        agent["skill_activation"] = {"status": "never_activated"}

    rendered = _render(results)
    block = _evidence_block(rendered)

    assert "Overall verdict: PASS" in rendered
    assert "required for publication" in block
    assert "optional by policy" not in block
    assert "Tier 3 dimension gate passed" not in block
    assert "recorded Tier 3 verdict is pass" in block


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.3999, "| Codex | fail | Security 39.99% below the 40% FAIL boundary |"),
        (0.4999, "| Codex | neutral | Security 49.99% below the 50% PASS boundary |"),
        (
            0.5005,
            "| Codex | pass | every dimension at or above the 50% PASS boundary (lowest: Security 50.1%) |",
        ),
    ],
)
def test_gate_evidence_preserves_score_bands_and_the_card_rounding_policy(score: float, expected: str) -> None:
    results = _issue_192_results()
    codex = results[-1].metadata["agent_eval"]["agents"]["codex"]
    codex["dimensions"] = _dimensions(0.55, 0.88, {"security": score})
    codex["with_skill"] = sum(row["with_skill"] for row in codex["dimensions"]) / len(codex["dimensions"])
    codex["lift"] = codex["with_skill"] - codex["baseline"]

    assert expected in _evidence_block(_render(results))


def test_evidence_block_is_absent_from_the_tier3_scoped_report() -> None:
    tier3_only = [_issue_192_results()[-1]]
    rendered = BenchmarkReporter(include_timestamp=True, report_scope="tier3").render_all(tier3_only)
    assert _EVIDENCE_HEADING not in rendered


def test_untrusted_agent_identity_cannot_inject_markdown_structure() -> None:
    results = _issue_192_results()
    payload = results[-1].metadata["agent_eval"]
    payload["agents"]["gemini-cli | ## Injected"] = payload["agents"].pop("gemini-cli")
    for container in (payload, payload["summary"]):
        container["best_agent"] = "gemini-cli | ## Injected"
        container["agents_run"] = ["claude-code", "codex", "gemini-cli | ## Injected"]

    block = _evidence_block(_render(results))
    injected = [line for line in block.splitlines() if line.startswith("|") and "Injected" in line]
    assert injected, block
    for line in injected:
        # The untrusted name keeps its text but cannot open a new table column
        # or a new section.
        assert "\\| ## Injected" in line
        assert len(line.split(" | ")) == 3, line
    assert "\n## Injected" not in block


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda _payload: None, id="pass"),
        pytest.param(lambda payload: payload["agents"].pop("gemini-cli"), id="fail"),
        pytest.param(
            lambda payload: payload["agents"]["codex"].update({"execution_status": "failed"}),
            id="agent-not-supported",
        ),
        pytest.param(
            lambda payload: payload["agents"]["gemini-cli"].update({"skill_activation": {"status": "never_activated"}}),
            id="never-activated",
        ),
    ],
)
def test_rendered_explanation_never_contradicts_the_computed_verdict(mutate) -> None:
    """The printed decision is the status ``_overall_status`` computes."""
    results = _issue_192_results()
    mutate(results[-1].metadata["agent_eval"])
    ae = results[-1].metadata["agent_eval"]

    status = _overall_status(results, ae, {}, expected_skill_name="demo-skill")
    rendered = _render(results)
    assert f"Overall verdict: {status}" in rendered
    assert f"- Decision: **{status}**" in _evidence_block(rendered)


def test_breakdown_folds_into_the_same_dimension_verdict() -> None:
    """The gate verdict is a fold of the breakdown the card renders."""
    payload = _issue_192_results()[-1].metadata["agent_eval"]
    breakdown = agent_eval_dimension_breakdown(payload)

    assert {entry.name: (entry.state, entry.verdict) for entry in breakdown} == {
        "claude-code": ("incomplete_evidence", None),
        "codex": ("scored", "fail"),
        "gemini-cli": ("scored", "pass"),
    }
    assert agent_eval_dimension_verdict(payload) == "pass"
    assert (
        _publication_assessment(
            _issue_192_results(),
            payload,
            expected_skill_name="demo-skill",
        ).tier3.status
        == "pass"
    )
