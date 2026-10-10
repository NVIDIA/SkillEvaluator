# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run status rule, tested through its one interface: ``resolve_run_status(results) -> RunStatus``.

The precedence the docs give (plugin-evaluation.mdx, "Reports and verdicts"): a real failure outranks missing
evidence, missing evidence outranks a pass, a skipped check is not a pass, an INCONCLUSIVE Integration never changes
a verdict, and a tier outside the exit gate never changes the headline. Every reporter renders what this returns.
"""

from __future__ import annotations

import itertools
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from _f90_gate_payloads import EDGE05_INTERVAL, EDGE05_WITH, EDGE05_WITHOUT, build_payload, scores, tier3
from _plugin_fixtures import STATISTICS, provenance

from skillevaluator.evaluation.tier3_report import (
    _validation_result_from_payload,
    advisory_skip_result,
    build_agent_eval_payload,
    dataset_required_result,
    incomplete_reason,
)
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.base import passes_required_gate
from skillevaluator.reporting.run_status import RunStatus, Status, TierStatus, resolve_run_status

PARTIAL = {"partial": True, "unresolved_skill_refs": ["github::example-org/example-repo::skills::missing"]}
_DIMENSIONS = ("security", "correctness", "discoverability", "effectiveness", "efficiency")

# ---------------------------------------------------------------------------
# Builders: results shaped the way ``validate`` produces and stamps them
# ---------------------------------------------------------------------------


def check(
    name: str,
    *,
    tier: int = 1,
    blocking: bool = True,
    high: int = 0,
    low: int = 0,
    incomplete: tuple[str, ...] = (),
    skipped: bool = False,
    optional: bool = False,
) -> ValidationResult:
    """A Tier 1 or Tier 2 check: HIGH findings fail it, a scanner that did not complete leaves it incomplete."""
    result = ValidationResult(validator_name=name, validator_description=f"{name} check")
    for index in range(high):
        result.add_finding(
            Finding("SECURITY", Severity.HIGH, f"high_{index}", f"{name} HIGH {index}", "SKILL.md", line_number=1)
        )
    for index in range(low):
        result.add_finding(Finding("SECURITY", Severity.LOW, f"low_{index}", f"{name} LOW {index}", "SKILL.md"))
    if not (high or low):
        result.add_success(name, "no issues")
    for tool in incomplete:
        result.add_warning(f"{tool} timed out")
        result.mark_scan_incomplete(tool)
    if skipped:
        result.add_warning(f"{name} did not run: prerequisite missing")
        result.metadata.update(skipped=True, execution_status="skipped", skip_reason=f"{name} prerequisite missing")
        if optional:
            result.metadata["optional"] = True
    result.metadata["gating"] = {"tier": tier, "blocking": blocking}
    return result


def gated(result: ValidationResult, *, blocking: bool) -> ValidationResult:
    result.metadata["gating"] = {"tier": 3, "blocking": blocking}
    return result


def cli_incomplete(result: ValidationResult) -> ValidationResult:
    """Stamp a Tier 3 run that did not complete the way ``validate`` does (``_incomplete_plugin_agent_eval_result``)."""
    provenance_record = result.metadata["agent_eval"].get("plugin_provenance") or {}
    result.passed = False
    result.metadata["execution_status"] = "skipped"
    result.metadata["skip_reason"] = incomplete_reason(provenance_record)
    return result


def agent(**overrides: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "with_skill": scores(),
        "without_skill": {},
        "execution_status": "succeeded",
        "rewards": [],
        "num_trials": 2,
        "model": "test-model",
    }
    info.update(overrides)
    return info


def multi_agent(**agents: dict[str, Any]) -> ValidationResult:
    payload = build_agent_eval_payload(
        "gate-demo", agents, use_llm_judge=False, run_config={"eval_target": {"kind": "plugin"}}
    )
    result = _validation_result_from_payload(payload)
    assert result is not None
    return result


def complete_plugin_run(
    *, verdict: str = "pass", score: float = 0.8, integration: dict[str, Any] | None = None
) -> ValidationResult:
    """A complete, publication-ready plugin run (every provenance field BENCHMARK.md requires)."""
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Run live agent evaluation")
    payload: dict[str, Any] = {
        "skill_name": "demo-plugin",
        "verdict": verdict,
        "execution_status": "succeeded",
        "evaluated_at": "2026-09-01T12:00:00+00:00",
        "evaluator_version": "0.3.0",
        "overall_lift": 0.3,
        "summary": {"environment": "docker", "verdict": verdict, "execution_status": "succeeded"},
        "dataset_summary": {"total_tasks": 4, "positive_tasks": 3, "negative_tasks": 1, "unclassified_tasks": 0},
        "dataset_digest": "sha256:" + "0123456789abcdef" * 4,
        "attempt_policy": {"max_attempts": 3, "pass_threshold": 0.5},
        "best_agent": "codex",
        "agents": {
            "codex": {
                "model": "gpt-codex",
                "execution_status": "succeeded",
                "dimensions": [
                    {"id": dimension, "baseline": 0.5, "with_skill": score, "lift": score - 0.5}
                    for dimension in _DIMENSIONS
                ],
                **deepcopy(STATISTICS),
            }
        },
        "plugin_provenance": provenance(partial=False),
        **deepcopy(STATISTICS),
    }
    if integration is not None:
        payload["integration"] = integration
    result.metadata["agent_eval"] = payload
    result.add_success("agent_eval", f"Live evaluation completed: verdict {verdict.upper()}")
    result.metadata["gating"] = {"tier": 3, "blocking": False}
    return result


def tiers(status: RunStatus) -> dict[int, Status]:
    return {number: tier.status for number, tier in status.tiers.items()}


# ---------------------------------------------------------------------------
# The precedence table
# ---------------------------------------------------------------------------


def test_a_real_failure_outranks_missing_evidence_across_results() -> None:
    # The saved hookify run: a schema failure, and a security scan whose SkillSpector run did not complete.
    schema = check("Plugin Schema & Bundle References", high=2)
    security = check("Security Scan", incomplete=("skillspector",))

    status = resolve_run_status([schema, security])

    assert status.headline is Status.FAILED
    assert status.tiers[1].status is Status.FAILED
    assert status.publication is Status.FAILED
    assert status.failed == (schema,)
    assert status.incomplete == (security,)
    # A FAIL still names what did not complete.
    assert status.missing == ("skillspector",)
    assert status.reasons == (
        "Plugin Schema & Bundle References failed in Tier 1",
        "skillspector did not complete",
    )
    assert status.of(schema) is Status.FAILED
    assert status.of(security) is Status.INCOMPLETE


def test_a_blocking_finding_outranks_missing_evidence_in_the_same_result() -> None:
    security = check("Security Scan", high=1, incomplete=("skillspector",))

    status = resolve_run_status([security])

    assert status.of(security) is Status.FAILED
    assert status.headline is Status.FAILED
    assert status.failed == (security,)
    assert status.incomplete == ()
    assert status.missing == ("skillspector",)


def test_missing_evidence_alone_is_incomplete_not_failed() -> None:
    # A LOW note and a plain error string are not blocking findings.
    security = check("Security Scan", low=1, incomplete=("skillspector",))
    security.add_error("skillspector reported partial analysis")

    status = resolve_run_status([check("Plugin Schema"), security])

    assert status.headline is Status.INCOMPLETE
    assert status.tiers[1].status is Status.INCOMPLETE
    assert status.publication is Status.INCOMPLETE
    assert status.of(security) is Status.INCOMPLETE
    assert status.reasons == ("skillspector did not complete",)


def test_a_clean_run_passes_and_publication_still_requires_tier3() -> None:
    results = [check("Plugin Schema"), check("Security Scan", low=1)]

    status = resolve_run_status(results)

    assert status.headline is Status.PASSED
    assert tiers(status) == {1: Status.PASSED, 2: Status.NOT_RUN, 3: Status.NOT_RUN}
    assert status.reasons == ()
    # BENCHMARK.md requires Tier 3 evidence by default; a catalog can waive it.
    assert status.publication is Status.INCOMPLETE
    results[0].metadata["benchmark_policy"] = {"tier3_required": False}
    assert resolve_run_status(results).publication is Status.PASSED


def test_no_tiers_run_is_a_vacuous_gate_and_never_a_publication_pass() -> None:
    status = resolve_run_status([])

    # Nothing failed the exit gate (validate exits 0), but nothing ran either.
    assert status.headline is Status.PASSED
    assert tiers(status) == {1: Status.NOT_RUN, 2: Status.NOT_RUN, 3: Status.NOT_RUN}
    assert status.publication is Status.INCOMPLETE
    assert status.reasons == ()


def test_a_skipped_check_is_not_a_pass() -> None:
    parity = check("Claude Plugin Validate", skipped=True)
    tier2 = check("Inter-Plugin Deduplication", tier=2, blocking=False, skipped=True, optional=True)

    status = resolve_run_status([check("Plugin Schema"), parity, tier2])

    assert status.of(parity) is Status.SKIPPED
    assert status.of(tier2) is Status.SKIPPED
    assert status.tiers[1].status is Status.PASSED  # the checks that ran passed
    assert status.tiers[2].status is Status.SKIPPED  # nothing in Tier 2 ran
    assert status.headline is Status.PASSED  # a skip never fails the exit gate
    # A skipped required check keeps the publication card INCOMPLETE; an optional skip does not.
    assert status.tiers[1].publication is Status.PASSED
    assert status.tiers[2].publication is Status.SKIPPED
    assert status.publication is Status.INCOMPLETE


def test_a_tier_that_only_skipped_a_required_check_is_publication_incomplete() -> None:
    status = resolve_run_status([check("Tier 2 Deduplication", tier=2, skipped=True)])

    assert status.tiers[2].status is Status.SKIPPED
    assert status.tiers[2].publication is Status.INCOMPLETE


def test_an_advisory_tier3_skip_is_skipped_and_a_gated_one_fails() -> None:
    advisory = gated(advisory_skip_result("Tier 3 live evaluation skipped: docker missing"), blocking=False)
    blocking = gated(advisory_skip_result("Tier 3 live evaluation skipped: docker missing"), blocking=True)

    advisory_status = resolve_run_status([check("Plugin Schema"), advisory])
    blocking_status = resolve_run_status([check("Plugin Schema"), blocking])

    assert advisory_status.of(advisory) is Status.SKIPPED
    assert advisory_status.tiers[3].status is Status.SKIPPED
    assert advisory_status.headline is Status.PASSED
    assert blocking_status.of(blocking) is Status.FAILED
    assert blocking_status.tiers[3].status is Status.FAILED
    assert blocking_status.headline is Status.FAILED
    assert blocking_status.publication is Status.FAILED


def test_an_advisory_tier3_without_a_task_source_did_not_run() -> None:
    result = dataset_required_result(
        Path("evals/evals.json"), [], blocking=False, skill_name="demo", source_kind="skill"
    )
    assert result.passed is True

    status = resolve_run_status([check("Plugin Schema"), result])

    assert status.of(result) is Status.SKIPPED
    assert status.tiers[3].status is Status.SKIPPED
    assert status.headline is Status.PASSED


@pytest.mark.parametrize("blocking", [False, True])
def test_a_partial_plugin_run_with_scored_trials_is_incomplete(blocking: bool) -> None:
    # Every arm scored, but a declared skill ref could not be resolved: a partial run, never a pass.
    result = gated(tier3(build_payload(scores(), plugin_provenance=PARTIAL)), blocking=blocking)
    assert result.metadata["agent_eval"]["agents"]["codex"]["execution_status"] == "succeeded"

    status = resolve_run_status([check("Plugin Schema"), result])

    assert status.of(result) is Status.INCOMPLETE
    assert status.tiers[3].status is Status.INCOMPLETE
    assert status.tiers[3].publication is Status.INCOMPLETE
    assert status.publication is Status.INCOMPLETE
    assert status.plugin is Status.INCOMPLETE
    if blocking:
        assert status.headline is Status.INCOMPLETE
        assert status.incomplete == (result,)
        assert status.missing == ("Tier 3",)
        assert status.advisory == ()
    else:
        # Outside the exit gate the headline is the gate's (validate exits 0), and it says Tier 3 was advisory.
        assert status.headline is Status.PASSED
        assert status.advisory == ("Tier 3 INCOMPLETE",)
        assert status.reasons == ("Tier 3 INCOMPLETE is advisory",)


def test_a_tier3_run_that_did_not_complete_is_incomplete() -> None:
    # The saved hookify run's Tier 3: Harbor errored, the provenance is partial, and validate stamped it.
    payload = build_payload(
        scores(),
        execution_status="failed",
        execution_errors=["Harbor job did not complete successfully: 2 errored"],
        plugin_provenance={"partial": True, "execution_incomplete": "Tier 3 plugin evaluation did not complete: boom"},
    )
    stamped = gated(cli_incomplete(tier3(payload)), blocking=False)
    unstamped = gated(tier3(deepcopy(payload)), blocking=True)

    assert resolve_run_status([stamped]).of(stamped) is Status.INCOMPLETE
    # Re-rendered without validate's stamp (``view``), the partial provenance alone still says INCOMPLETE.
    assert resolve_run_status([unstamped]).of(unstamped) is Status.INCOMPLETE
    assert resolve_run_status([unstamped]).headline is Status.INCOMPLETE


@pytest.mark.parametrize("blocking", [False, True])
def test_a_partial_run_whose_evaluated_parts_failed_is_failed(blocking: bool) -> None:
    fail_verdict = gated(tier3(build_payload(scores(security=0.0), plugin_provenance=PARTIAL)), blocking=blocking)
    regression = gated(
        tier3(build_payload(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL, plugin_provenance=PARTIAL)),
        blocking=blocking,
    )

    for result, label in ((fail_verdict, "Tier 3 verdict FAIL"), (regression, "Tier 3 Skill Lift regression")):
        status = resolve_run_status([check("Plugin Schema"), result])
        assert status.of(result) is Status.FAILED
        assert status.tiers[3].status is Status.FAILED
        assert status.publication is Status.FAILED
        assert status.publication_advisory is not blocking
        if blocking:
            assert status.headline is Status.FAILED
        else:
            assert status.headline is Status.PASSED
            assert status.advisory == (label,)


def test_an_inconclusive_integration_never_changes_a_verdict() -> None:
    inconclusive = {"verdict": "inconclusive", "measured": False, "reason": "No cross-component case completed."}
    measured = {"verdict": "real_integration", "measured": True, "integration_lift": 0.18}

    with_inconclusive = resolve_run_status([check("Plugin Schema"), complete_plugin_run(integration=inconclusive)])
    with_measured = resolve_run_status([check("Plugin Schema"), complete_plugin_run(integration=measured)])

    assert with_inconclusive.tiers[3].status is Status.PASSED
    assert with_inconclusive.headline is Status.PASSED
    assert with_inconclusive.publication is Status.PASSED
    assert (with_inconclusive.headline, with_inconclusive.publication, tiers(with_inconclusive)) == (
        with_measured.headline,
        with_measured.publication,
        tiers(with_measured),
    )


def test_a_neutral_tier3_run_is_neutral_not_passed() -> None:
    status = resolve_run_status([check("Plugin Schema"), complete_plugin_run(verdict="neutral", score=0.45)])

    assert status.tiers[3].status is Status.NEUTRAL
    assert status.publication is Status.NEUTRAL
    assert status.headline is Status.PASSED


def test_a_tier3_run_without_publication_provenance_passes_the_gate_but_not_the_card() -> None:
    # build_payload records no dataset digest, so BENCHMARK.md cannot publish it; the gate still passes.
    result = gated(tier3(build_payload(scores())), blocking=True)

    status = resolve_run_status([check("Plugin Schema"), result])

    assert status.tiers[3].status is Status.PASSED
    assert status.tiers[3].publication is Status.INCOMPLETE
    assert status.headline is Status.PASSED
    assert status.publication is Status.INCOMPLETE


def test_multi_agent_runs() -> None:
    # One agent passes: the dimension gate is per run, so the run passes.
    one_passes = multi_agent(codex=agent(), **{"claude-code": agent(with_skill=scores(security=0.0))})
    # Every agent is in the NEUTRAL band.
    both_neutral = multi_agent(
        codex=agent(with_skill=scores(security=0.45)), **{"claude-code": agent(with_skill=scores(security=0.45))}
    )
    # One agent crashed: the run did not produce a complete scored result.
    one_crashed = multi_agent(
        codex=agent(), **{"claude-code": agent(execution_status="failed", with_skill={}, num_trials=0)}
    )

    assert resolve_run_status([one_passes]).tiers[3].status is Status.PASSED
    assert resolve_run_status([both_neutral]).tiers[3].status is Status.NEUTRAL
    crashed = resolve_run_status([gated(one_crashed, blocking=False)])
    assert crashed.tiers[3].status is Status.FAILED
    assert crashed.headline is Status.PASSED
    assert crashed.advisory == ("Tier 3 FAIL",)
    assert resolve_run_status([gated(one_crashed, blocking=True)]).headline is Status.FAILED


def test_an_advisory_tier_never_changes_the_headline() -> None:
    advisory_incomplete = check("Context Deduplication", tier=2, blocking=False, incomplete=("embedding-provider",))
    advisory_failed = check("Inter-Skill Deduplication", tier=2, blocking=False, high=1)

    status = resolve_run_status([check("Plugin Schema"), advisory_incomplete, advisory_failed])

    assert status.headline is Status.PASSED
    assert status.tiers[2].status is Status.FAILED
    assert status.tiers[2].blocking is False
    # The terminal summary names the advisory tier instead of saying every validation passed.
    assert status.advisory == ("Tier 2 FAIL",)
    assert status.reasons == ("embedding-provider did not complete", "Tier 2 FAIL is advisory")
    # The plugin row stays fail-closed on missing evidence, even advisory evidence.
    assert status.plugin is Status.INCOMPLETE
    # BENCHMARK.md counts every missing scan.
    assert status.publication is Status.INCOMPLETE


def test_a_blocking_tier2_counts() -> None:
    blocking_incomplete = check("Context Deduplication", tier=2, incomplete=("embedding-provider",))

    status = resolve_run_status([check("Plugin Schema"), blocking_incomplete])

    assert status.headline is Status.INCOMPLETE
    assert status.tiers[2].blocking is True


def test_tier_membership_follows_the_recorded_gate() -> None:
    # validate stamps the tier; the name alone ("Similarity Check" has no "dedup") would say Tier 1.
    schema = check("Plugin Schema")
    similarity = check("Similarity Check", tier=2, blocking=False, high=1)
    agent_eval = complete_plugin_run()

    status = resolve_run_status([schema, similarity, agent_eval])

    assert status.tiers[1].results == (schema,)
    assert status.tiers[2].results == (similarity,)
    assert status.tiers[3].results == (agent_eval,)
    assert isinstance(status.tiers[3], TierStatus)


def test_tier_membership_without_a_recorded_gate() -> None:
    dedup = ValidationResult(validator_name="Context Deduplication")
    similarity = ValidationResult(validator_name="Similarity Check")
    agent_eval = complete_plugin_run()
    del agent_eval.metadata["gating"]
    schema = ValidationResult(validator_name="SCHEMA")

    status = resolve_run_status([schema, dedup, similarity, agent_eval])

    assert status.tiers[1].results == (schema,)
    assert status.tiers[2].results == (dedup, similarity)
    assert status.tiers[3].results == (agent_eval,)


# ---------------------------------------------------------------------------
# Invariants over every combination of tier outcomes and gates
# ---------------------------------------------------------------------------

_TIER1 = {
    "pass": lambda: [check("Plugin Schema")],
    "fail": lambda: [check("Plugin Schema", high=1)],
    "incomplete": lambda: [check("Plugin Schema"), check("Security Scan", incomplete=("skillspector",))],
    "fail+incomplete": lambda: [check("Plugin Schema", high=1), check("Security Scan", incomplete=("skillspector",))],
    "skipped": lambda: [check("Plugin Schema"), check("Claude Plugin Validate", skipped=True)],
}
_TIER2 = {
    "absent": lambda _blocking: [],
    "pass": lambda blocking: [check("Context Deduplication", tier=2, blocking=blocking)],
    "fail": lambda blocking: [check("Context Deduplication", tier=2, blocking=blocking, high=1)],
    "incomplete": lambda blocking: [
        check("Context Deduplication", tier=2, blocking=blocking, incomplete=("embedding-provider",))
    ],
    "skipped": lambda blocking: [
        check("Inter-Plugin Deduplication", tier=2, blocking=blocking, skipped=True, optional=True)
    ],
}
_TIER3 = {
    "absent": lambda _blocking: [],
    "pass": lambda blocking: [gated(complete_plugin_run(), blocking=blocking)],
    "neutral": lambda blocking: [gated(complete_plugin_run(verdict="neutral", score=0.45), blocking=blocking)],
    "fail": lambda blocking: [gated(tier3(build_payload(scores(security=0.0))), blocking=blocking)],
    "partial": lambda blocking: [gated(tier3(build_payload(scores(), plugin_provenance=PARTIAL)), blocking=blocking)],
    "partial-fail": lambda blocking: [
        gated(tier3(build_payload(scores(security=0.0), plugin_provenance=PARTIAL)), blocking=blocking)
    ],
    "not-run": lambda blocking: [gated(advisory_skip_result("docker missing"), blocking=blocking)],
}
_COMBINATIONS = [
    pytest.param(t1, t2, b2, t3, b3, id=f"t1={t1}-t2={t2}{'(gate)' if b2 else ''}-t3={t3}{'(gate)' if b3 else ''}")
    for t1, t2, b2, t3, b3 in itertools.product(_TIER1, _TIER2, (False, True), _TIER3, (False, True))
]


@pytest.mark.parametrize(("t1", "t2", "b2", "t3", "b3"), _COMBINATIONS)
def test_status_invariants(t1: str, t2: str, b2: bool, t3: str, b3: bool) -> None:
    results = [*_TIER1[t1](), *_TIER2[t2](b2), *_TIER3[t3](b3)]

    status = resolve_run_status(results)

    # The headline is the exit gate: PASSED exactly when validate exits 0.
    assert (status.headline is Status.PASSED) == all(passes_required_gate(result) for result in results)
    # FAILED exactly when a gating result failed for real; INCOMPLETE when gating results only miss evidence.
    gate_failures = [result for result in results if not passes_required_gate(result)]
    assert (status.headline is Status.FAILED) == any(status.of(r) is not Status.INCOMPLETE for r in gate_failures)
    assert set(map(id, status.failed)) | set(map(id, status.incomplete)) == set(map(id, gate_failures))
    # A gating tier that failed fails the headline; one that is only incomplete never passes it.
    for tier in status.tiers.values():
        gating_failures = [r for r in tier.results if not passes_required_gate(r)]
        if tier.status is Status.FAILED and any(status.of(r) is Status.FAILED for r in gating_failures):
            assert status.headline is Status.FAILED
        if tier.status is Status.INCOMPLETE and gating_failures:
            assert status.headline is not Status.PASSED
    # BENCHMARK.md is never kinder than the gate, and a Tier 3 FAIL outranks missing evidence there too.
    if status.headline is Status.FAILED:
        assert status.publication is Status.FAILED
    if status.headline is Status.INCOMPLETE:
        assert status.publication in {Status.INCOMPLETE, Status.FAILED}
        assert (status.publication is Status.FAILED) == (t3 in {"fail", "partial-fail"})
    # The plugin row is the headline, fail-closed on missing evidence.
    if status.headline is not Status.PASSED:
        assert status.plugin is status.headline
    # A skipped check is never a pass, and nothing is both failed and incomplete.
    for result in results:
        if result.metadata.get("skipped") and result.passed:
            assert status.of(result) is Status.SKIPPED
    assert not set(map(id, status.failed)) & set(map(id, status.incomplete))
    # Order of results never matters.
    shuffled = list(results)
    random.Random(f"{t1}{t2}{b2}{t3}{b3}").shuffle(shuffled)
    again = resolve_run_status(shuffled)
    assert (again.headline, again.publication, again.plugin, tiers(again)) == (
        status.headline,
        status.publication,
        status.plugin,
        tiers(status),
    )


# ---------------------------------------------------------------------------
# Review follow-ups: edge cases of the rule, and the reporters that render it
# ---------------------------------------------------------------------------


def test_a_skipped_check_that_failed_is_failed_on_the_card_too() -> None:
    parity = check("Claude Plugin Validate", skipped=True)
    parity.add_error("claude plugin validate crashed")

    status = resolve_run_status([check("Plugin Schema"), parity])

    assert status.of(parity) is Status.FAILED
    assert status.tiers[1].status is Status.FAILED
    assert status.tiers[1].publication is Status.FAILED
    assert status.headline is Status.FAILED


def test_a_bundled_skills_blocking_medium_finding_outranks_missing_evidence() -> None:
    skill = ValidationResult(validator_name="Secrets Detection")
    skill.add_finding(
        Finding("SECRET", Severity.MEDIUM, "generic-secret", "Possible secret", "SKILL.md"), fail_on_medium=True
    )
    merged = ValidationResult(validator_name="Secrets Detection")
    merged.merge_with_prefix(skill, "my-skill")
    merged.mark_scan_incomplete("gitleaks")
    merged.metadata["gating"] = {"tier": 1, "blocking": True}
    assert merged.errors and merged.errors[0].startswith("[my-skill] ")

    status = resolve_run_status([merged])

    assert status.of(merged) is Status.FAILED
    assert status.headline is Status.FAILED
    assert status.missing == ("gitleaks",)


def test_a_card_fail_the_gate_never_saw_does_not_claim_tier3_gated_the_run() -> None:
    from skillevaluator.reporting import BenchmarkReporter

    # The payload's verdict says pass, but its dimension scores say FAIL: the card is stricter than the gate,
    # which validate decides from the result (it passed, so the run exited 0).
    agent_eval = gated(complete_plugin_run(verdict="pass", score=0.3), blocking=True)
    results = [check("Plugin Schema"), agent_eval]

    status = resolve_run_status(results)
    card = BenchmarkReporter(include_timestamp=False, content_type="plugin", skill_name="demo-plugin").render_all(
        results
    )

    assert status.headline is Status.PASSED
    assert status.publication is Status.FAILED
    assert "Overall verdict: FAIL" in card
    assert "made Tier 3 gate this run" not in card


def test_markdown_counts_a_skipped_check_as_skipped_and_keeps_its_findings() -> None:
    from skillevaluator.reporting import MarkdownReporter

    skipped = check("Inter-Skill Deduplication", tier=2, blocking=False, skipped=True, optional=True, low=1)

    output = MarkdownReporter(include_timestamp=False).render_all([check("Plugin Schema"), skipped])

    assert "| ✅ Passed | 1 |" in output
    assert "| ⏭️ Skipped | 1 |" in output
    assert "### ⏭️ SKIPPED Inter-Skill Deduplication" in output
    assert "Inter-Skill Deduplication LOW 0" in output


def test_html_filters_match_the_row_statuses() -> None:
    import re

    from skillevaluator.reporting import HTMLReporter

    html = HTMLReporter(include_timestamp=False).render_all([check("Plugin Schema", high=1), check("Security Scan")])

    statuses = re.findall(r'class="validator-card" id="validator-\d+" data-status="([a-z]+)"', html)
    assert statuses == ["failed", "passed"]
    # "Passed Only" and "Failed Only" compare against those same values.
    assert "status === 'pass' ? value === 'passed' : (value === 'failed' || value === 'incomplete')" in html
