# SPDX-License-Identifier: Apache-2.0
"""Human reports retain recorded coverage and distinguish it from approval."""

from __future__ import annotations

import copy
import re
from dataclasses import FrozenInstanceError

import pytest

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import presentation
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.reporting.run_status import Status


def context(results, **kwargs):
    builder = getattr(presentation, "build_final_render_context", None)
    assert callable(builder), "final reports need one shared immutable presentation context"
    return builder(results, **kwargs)


def result(tier=1, **metadata):
    return ValidationResult(
        validator_name="AGENT_EVAL" if tier == 3 else f"Tier{tier}",
        metadata={
            "gating": {"tier": tier, "blocking": True},
            **metadata,
        },
    )


def test_tier_counts_exclude_na_and_do_not_leak_other_tiers():
    na = result()
    na.add_success("code_file_discovery", "No code files found - code risk analysis is not applicable")
    failed = result(2)
    failed.add_error("duplicate")
    incomplete = result(2)
    incomplete.mark_scan_incomplete("catalog")
    ctx = context(
        [result(), na, result(skipped=True), failed, incomplete], requested_tiers=(1, 2), displayed_tiers=(1, 2, 3)
    )
    assert ctx.command_status == Status.FAILED
    assert ctx.command_label == "Changes required"
    assert (ctx.tiers[1].passed, ctx.tiers[1].executed, ctx.tiers[1].not_applicable, ctx.tiers[1].skipped) == (
        1,
        1,
        1,
        1,
    )
    assert (ctx.tiers[2].passed, ctx.tiers[2].failed, ctx.tiers[2].incomplete, ctx.tiers[2].executed) == (0, 1, 1, 2)
    assert ctx.tiers[2].status == Status.FAILED
    assert ctx.tiers[3].status == Status.NOT_RUN
    assert not ctx.tiers[3].requested
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ctx.command_label = "changed"
    with pytest.raises(TypeError):
        ctx.tiers[1] = ctx.tiers[2]


def test_recorded_condition_coverage_outlives_preview_and_retry_rows():
    payload = {
        "agents": {
            "agent": {
                "conditions": {
                    "with_skill": {"execution_status": "succeeded", "scored_attempts": 101, "expected_attempts": 120},
                    "without_skill": {"execution_status": "failed", "scored_attempts": 87, "expected_attempts": 120},
                },
                "trials_baseline": [{}] * 64,
            }
        },
        "trials": [{}] * 64,
    }
    source = result(3, agent_eval=payload)
    original = copy.deepcopy(source)
    ctx = context([source], report_scope="tier3")
    assert ctx.coverage.text == "188 scored trials: 101 with skill, 87 baseline"
    assert ctx.coverage.with_skill.expected == 120
    assert ctx.coverage.baseline.status == "failed"
    assert ctx.publication_status is None
    assert ctx.publication_label == "Not assessed (Tier 3 only)"
    assert source == original


@pytest.mark.parametrize("bad", [None, True, -1, "3", 1.5])
def test_missing_or_malformed_counters_never_become_zero(bad):
    payload = {
        "agents": {
            "a": {
                "conditions": {
                    "with_skill": {"execution_status": "incomplete", "scored_attempts": bad},
                    "without_skill": {"execution_status": "skipped", "scored_attempts": 0, "expected_attempts": 0},
                }
            }
        },
        "trials": [{"overall": 0.8}],
    }
    coverage = context([result(3, agent_eval=payload)]).coverage
    assert coverage.with_skill.scored is None
    assert coverage.baseline.status == "skipped"
    assert "unavailable" in coverage.text
    assert "baseline skipped" in coverage.text
    assert "0 scored trials" not in coverage.text


def test_legacy_rows_are_observed_evidence_not_complete_totals():
    coverage = context(
        [
            result(
                3,
                agent_eval={
                    "trials": [{"overall": 0.8}] * 64,
                    "agents": {"a": {"trials_baseline": [{"overall": 0.2}] * 3}},
                },
            )
        ]
    ).coverage
    assert coverage.with_skill.scored is None
    assert coverage.with_skill.observed == 64
    assert coverage.baseline.observed == 3
    assert "64 retained with-skill rows" in coverage.text
    assert "3 retained baseline rows" in coverage.text
    assert "67 scored" not in coverage.text


def test_failed_result_outranks_incomplete_in_html_tile_and_header():
    failed = result()
    failed.add_error("unsafe")
    missing = result()
    missing.mark_scan_incomplete("scanner")
    html = HTMLReporter(tabs=[{"id": "tier1", "label": "Tier 1"}, {"id": "tier2", "label": "Tier 2"}]).render_all(
        [failed, missing]
    )
    tile = re.search(r'<div class="tier-card ([^"]+)"[^>]+data-target-tab="tier1"', html)
    assert tile and "fail" in tile[1]
    assert 'hero-command-status verdict-chip fail">Changes required</p>' in html
    assert "Changes required" in html


def test_full_cli_and_html_distinguish_required_pass_from_publication():
    source = result()
    terminal = re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render_all([source]))
    html = HTMLReporter().render_all([source])
    assert "Passed required checks" in terminal
    assert "Passed required checks" in html
    assert "Publication eligibility" in terminal
    assert "Publication eligibility" in html


def test_compact_panel_shares_incomplete_status_and_publication(tmp_path):
    from io import StringIO

    from rich.console import Console

    from skillevaluator.cli import _finish_pipeline_view
    from skillevaluator.reporting.console_ui import ValidateView

    source = result()
    source.mark_scan_incomplete("scanner")
    output = StringIO()
    view = ValidateView("demo", [(1, "Tier 1", "")], console=Console(file=output, width=120, color_system=None))
    _finish_pipeline_view(
        view,
        tier_gate_results=[source],
        tier3_result=None,
        gate_failed=True,
        output_dir=tmp_path,
        basename="report",
        report_formats=(),
        target_path=tmp_path,
        agent_eval=False,
    )
    assert "Incomplete" in output.getvalue()
    assert "INCOMPLETE" in output.getvalue()
    assert "Publication eligibility: Incomplete" in output.getvalue()
    assert "✗ FAIL" not in output.getvalue()


def test_html_coverage_uses_full_raw_counts_and_preserves_payload():
    source = result(
        3,
        agent_eval={
            "agents": {
                "a": {
                    "conditions": {
                        "with_skill": {
                            "execution_status": "succeeded",
                            "scored_attempts": 101,
                            "expected_attempts": 120,
                        },
                        "without_skill": {
                            "execution_status": "failed",
                            "scored_attempts": 87,
                            "expected_attempts": 120,
                        },
                    }
                }
            },
            "trials": [{"overall": 0.8}] * 100,
        },
    )
    original = copy.deepcopy(source)
    html = HTMLReporter().render(source)
    assert "188 scored trials: 101 with skill, 87 baseline" in html
    assert "Best agent score" in html
    assert source == original


def test_required_failure_cannot_be_replaced_by_completed_quality_verdict():
    source = result(3, agent_eval={"verdict": "pass", "overall_score": 0.8})
    source.add_error("agent setup failed")
    html = HTMLReporter(report_scope="tier3").render(source)
    assert re.search(r'hero-command-status verdict-chip fail">Changes required</p>', html)
    assert 'data-publication-status="not-assessed"' in html


def test_missing_requested_tier_is_never_reported_as_zero_checks_passed():
    source = result()
    ctx = context([source], requested_tiers=(1,), displayed_tiers=(1, 2, 3))
    output = re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter(presentation_context=ctx).render_all([source]))
    assert "Tier 2: Not run (not requested)" in output
    assert "Tier 3: Not run (not requested)" in output


def test_partial_expected_attempt_counts_remain_visible():
    source = result(
        3,
        agent_eval={
            "agents": {
                "a": {
                    "conditions": {
                        "with_skill": {"execution_status": "incomplete", "scored_attempts": 4, "expected_attempts": 10},
                        "without_skill": {"execution_status": "skipped", "scored_attempts": 0, "expected_attempts": 0},
                    }
                }
            }
        },
    )
    assert "With skill: 4 of 10 scored (incomplete)" in HTMLReporter().render(source)


def test_confirmed_failure_never_gets_green_pass_in_full_cli_details():
    source = result(confirmed_failure=True)
    source.mark_scan_incomplete("scanner")
    output = re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter().render_all([source]))
    assert "[PASS] Validation passed" not in output
    assert "[INCOMPLETE] scanner did not complete" not in output
    assert "[FAIL] Validation failed" in output


def test_all_na_tier_remains_na_without_claiming_execution():
    source = result()
    source.add_success("lint", "No scripts/ or tools/ directory found")
    ctx = context([source], displayed_tiers=(1, 2))
    assert getattr(ctx.tiers[1], "display_status", None) == "not_applicable"
    html = HTMLReporter(tabs=[{"id": "tier1", "label": "Tier 1"}, {"id": "tier2", "label": "Tier 2"}]).render(source)
    assert re.search(r'tier-card-label">Tier 1</span>\s*<span class="tier-card-verdict">N/A</span>', html)


@pytest.mark.parametrize("skipped_arm", ["with_skill", "without_skill"])
def test_one_arm_runs_preserve_explicit_skip_and_do_not_infer_baseline(skipped_arm):
    active = "without_skill" if skipped_arm == "with_skill" else "with_skill"
    source = result(
        3,
        agent_eval={
            "agents": {
                "a": {
                    "conditions": {
                        active: {"execution_status": "succeeded", "scored_attempts": 7, "expected_attempts": 7},
                        skipped_arm: {"execution_status": "skipped", "scored_attempts": 0, "expected_attempts": 0},
                    }
                }
            }
        },
    )
    coverage = context([source]).coverage
    assert coverage.text == (
        "7 scored trials: with skill skipped, 7 baseline"
        if skipped_arm == "with_skill"
        else "7 scored trials: 7 with skill, baseline skipped"
    )


def test_emitting_human_reports_preserves_canonical_json_and_return_value(tmp_path, monkeypatch):
    from io import StringIO

    from rich.console import Console

    from skillevaluator.reporting.json_reporter import JSONReporter
    from skillevaluator.tier1 import commands

    source = result(
        3,
        agent_eval={
            "agents": {
                "a": {
                    "conditions": {
                        "with_skill": {"execution_status": "failed", "scored_attempts": 3, "expected_attempts": 8},
                        "without_skill": {"execution_status": "skipped", "scored_attempts": 0, "expected_attempts": 0},
                    }
                }
            }
        },
    )
    source.add_error("execution failed")
    source.metadata["gating"]["blocking"] = False
    before = JSONReporter(include_timestamp=False).render(source)
    snapshot = copy.deepcopy(source)
    output = StringIO()
    monkeypatch.setattr(commands, "console", Console(file=output, color_system=None, width=160))
    assert (
        commands.emit_reports([source], report_formats=("cli", "html"), output_dir=tmp_path, basename="projection")
        is False
    )
    assert "Passed required checks" in output.getvalue()
    assert "3 scored trials: 3 with skill, baseline skipped" in (tmp_path / "projection.html").read_text(
        encoding="utf-8"
    )
    assert source == snapshot
    assert JSONReporter(include_timestamp=False).render(source) == before


def test_tier3_failure_tile_and_hero_keep_command_outcome_separate_from_quality():
    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

    conditions = {
        arm: {"execution_status": "succeeded", "execution_errors": [], "expected_attempts": 1, "scored_attempts": 1}
        for arm in ("with_skill", "without_skill")
    }
    payload = build_agent_eval_payload(
        "demo",
        {
            "good": {
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.4),
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 2,
                "scored_attempts": 2,
                "conditions": conditions,
                "num_trials": 1,
                "num_trials_baseline": 1,
            },
            "failed": {
                "execution_status": "failed",
                "execution_errors": ["agent setup failed"],
                "with_skill": {},
                "without_skill": {},
            },
        },
        dataset=[{"id": "case-a"}],
        attempt_policy={"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    source = result(3, agent_eval=payload)
    source.add_error("agent setup failed")
    html = HTMLReporter(report_scope="tier3").render(source)
    hero = html.split('<section class="agent-hero"', 1)[1].split("</section>", 1)[0]
    assert "Changes required" in hero
    assert re.search(r'<div class="tier-card fail[^\"]*"[^>]+data-target-tab="tier3"', hero)


def test_partial_tier3_evidence_assessment_is_shared_without_changing_command_gate():
    source = result(3, agent_eval={"verdict": "pass"})
    source.metadata["gating"]["blocking"] = False
    ctx = context([source])
    assert ctx.command_status == Status.PASSED
    assert ctx.tiers[3].display_status == "incomplete"
    output = re.sub(r"\x1b\[[0-9;]*m", "", CLIReporter(presentation_context=ctx).render_all([source]))
    assert "Tier 3: Incomplete evidence" in output
    html = HTMLReporter(presentation_context=ctx).render(source)
    assert re.search(r'tier-card-label">Tier 3</span>\s*<span class="tier-card-verdict">Incomplete</span>', html)


def test_explicit_tier3_skip_is_shared_even_when_publication_is_incomplete():
    source = result(
        3,
        agent_eval={
            "execution_status": "skipped",
            "verdict": "neutral",
            "provenance": {
                "advisory": True,
                "reason": "skipped",
                "message": "provider unavailable",
            },
        },
    )
    source.metadata["gating"]["blocking"] = False
    ctx = context([source])
    assert ctx.tiers[3].status == Status.SKIPPED
    assert ctx.tiers[3].display_status == "skipped"
    assert ctx.publication_status == Status.INCOMPLETE


def test_retained_row_observations_share_one_projection_budget(monkeypatch):
    from skillevaluator.reporting import base

    visited = []

    class CountedRows(list):
        def __iter__(self):
            for row in super().__iter__():
                visited.append(row)
                yield row

    monkeypatch.setattr(base, "AGENT_EVAL_REPORT_MAX_NODES", 100)
    rows = CountedRows([{}] * 10)
    coverage = presentation.trial_coverage(
        {
            "trials": rows,
            "agents": {f"agent-{i}": {"trials_baseline": rows} for i in range(12)},
        }
    )
    assert len(visited) <= 100
    assert coverage.with_skill.observed == 10
    assert coverage.baseline.observed is None
    assert coverage.baseline.scored is None
    assert "baseline scored trials unavailable" in coverage.text
    assert "0 retained baseline rows" not in coverage.text


def test_recorded_scored_counters_avoid_all_retained_row_iteration():
    visited = []

    class CountedRows(list):
        def __iter__(self):
            for row in super().__iter__():
                visited.append(row)
                yield row

    rows = CountedRows([{}] * 1000)
    conditions = {
        arm: {"execution_status": "succeeded", "scored_attempts": 9, "expected_attempts": 12}
        for arm in ("with_skill", "without_skill")
    }
    coverage = presentation.trial_coverage(
        {
            "trials": rows,
            "agents": {f"agent-{i}": {"conditions": conditions, "trials_baseline": rows} for i in range(120)},
        }
    )
    assert visited == []
    assert coverage.text == "2160 scored trials: 1080 with skill, 1080 baseline"
    assert coverage.with_skill.expected == 1440
    assert coverage.baseline.expected == 1440
    assert coverage.with_skill.observed is None
    assert coverage.baseline.observed is None


def test_exhausted_projection_budget_does_not_turn_other_arm_into_observed_zero(monkeypatch):
    from skillevaluator.reporting import base

    monkeypatch.setattr(base, "AGENT_EVAL_REPORT_MAX_NODES", 10)
    coverage = presentation.trial_coverage({"trials": [{}] * 11, "agents": {"a": {"trials_baseline": []}}})
    assert coverage.with_skill.observed is None
    assert coverage.baseline.observed is None
    assert "0 retained baseline rows" not in coverage.text


@pytest.mark.parametrize("width", [60, 80, 98])
@pytest.mark.parametrize(("skipped", "not_applicable"), [(1, 0), (0, 2), (2, 3)])
def test_zero_execution_counts_stay_explicit_at_terminal_widths(monkeypatch, width, skipped, not_applicable):
    monkeypatch.setenv("COLUMNS", str(width))
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("CI", "true")
    sources = [result(skipped=True) for _ in range(skipped)]
    for _ in range(not_applicable):
        source = result()
        source.add_success("lint", "No scripts/ or tools/ directory found")
        sources.append(source)
    output = CLIReporter().render_all(sources)
    assert f"Tier 1: {skipped} skipped, {not_applicable} N/A; no checks executed" in output
    assert "0 of 0 checks passed" not in output
    assert "Passed required checks" in output
    assert "Publication eligibility: Incomplete" in output


@pytest.mark.parametrize(
    ("mode", "present", "expected"),
    [
        ("with_skill_only", "with_skill", "3 scored trials: 3 with skill, baseline skipped"),
        ("baseline_only", "without_skill", "3 scored trials: with skill skipped, 3 baseline"),
    ],
)
def test_producer_one_arm_mode_marks_the_missing_arm_skipped(mode, present, expected):
    # --skip-baseline / --skip-with-skill omit the unexecuted condition; the
    # payload-level mode is the producer's explicit skip marker.
    counts = {"execution_status": "succeeded", "scored_attempts": 3, "expected_attempts": 3}
    coverage = presentation.trial_coverage({"mode": mode, "agents": {"a": {"conditions": {present: counts}}}})
    assert coverage.text == expected
    assert "unavailable" not in coverage.text


def test_advisory_tier3_skip_reports_skipped_arms_not_unavailable_counts():
    coverage = presentation.trial_coverage({"execution_status": "skipped", "agents": {}})
    assert (coverage.with_skill.status, coverage.baseline.status) == ("skipped", "skipped")
    assert "unavailable" not in coverage.text
    source = result(3, skipped=True, agent_eval={"execution_status": "skipped", "agents": {}})
    output = CLIReporter().render_all([source])
    assert "scored trials unavailable" not in output


def test_standalone_tier3_report_header_follows_failing_tier3_verdict():
    from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, build_agent_eval_payload
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

    def arm(n):
        return {"execution_status": "succeeded", "execution_errors": [], "expected_attempts": n, "scored_attempts": n}

    payload = build_agent_eval_payload(
        "demo",
        {
            "agent": {
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.2),
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.9),
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 2,
                "scored_attempts": 2,
                "conditions": {"with_skill": arm(1), "without_skill": arm(1)},
                "num_trials": 1,
                "num_trials_baseline": 1,
            }
        },
        dataset=[{"id": "case-a"}],
        attempt_policy={"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    source = _validation_result_from_payload(payload)
    assert payload["verdict"] == "fail" and "gating" not in source.metadata
    ctx = context([source], report_scope="tier3")
    assert ctx.command_status == Status.FAILED
    html = HTMLReporter(report_scope="tier3", tabs=[{"id": "tier3", "label": "Tier 3"}]).render_all([source])
    assert re.search(r'hero-command-status verdict-chip fail">Changes required</p>', html)
    assert "Passed required checks" not in html


def test_not_applicable_rows_show_their_recorded_reason():
    covered = ValidationResult(validator_name="Code Risk", metadata={"gating": {"tier": 1, "blocking": True}})
    covered.add_success("code_file_discovery", "No code files found - code risk analysis is not applicable")
    skipped = result(skipped=True, skip_reason="Lint is not applicable: no scripts directory")
    output = re.sub(r"\s+", " ", CLIReporter().render_all([covered, skipped]))
    assert "1 checks passed" not in output
    assert "Skipped (see warnings)" not in output
    assert "No code files found" in output
    assert "Lint is not applicable: no scripts directory" in output


def test_executed_tier_counts_stay_on_one_line_at_narrow_widths(monkeypatch):
    monkeypatch.setenv("COLUMNS", "60")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("CI", "true")
    passed = result()
    passed.add_success("lint", "Lint passed")
    failed = result()
    failed.add_error("Lint failed")
    output = CLIReporter().render_all([passed, failed, result(skipped=True)])
    assert "Tier 1: 1 of 2 checks passed; 1 failed, 0 incomplete, 1 skipped, 0 N/A" in output


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param([], id="empty-list"),
        pytest.param({}, id="empty-object"),
        pytest.param(["with_skill_only"], id="list-mode"),
        pytest.param({"baseline_only": True}, id="object-mode"),
        pytest.param(None, id="null"),
        pytest.param(1, id="number"),
        pytest.param(True, id="boolean"),
        pytest.param("unrecognized", id="unknown-string"),
    ],
)
@pytest.mark.parametrize("recorded", [False, True], ids=["unavailable", "recorded-conditions"])
def test_malformed_or_unknown_mode_preserves_coverage_evidence_and_payload(mode, recorded):
    agents = {}
    if recorded:
        agents = {
            "a": {
                "conditions": {
                    "with_skill": {"execution_status": "succeeded", "scored_attempts": 2, "expected_attempts": 3},
                    "without_skill": {"execution_status": "incomplete", "scored_attempts": 1, "expected_attempts": 3},
                }
            }
        }
    payload = {"mode": mode, "agents": agents}
    before = copy.deepcopy(payload)
    coverage = presentation.trial_coverage(payload)
    if recorded:
        assert coverage.with_skill.status == "succeeded"
        assert coverage.baseline.status == "incomplete"
        assert coverage.with_skill.scored == 2
        assert coverage.baseline.scored == 1
        assert coverage.with_skill.expected == coverage.baseline.expected == 3
        assert coverage.text == "3 scored trials: 2 with skill, 1 baseline"
    else:
        assert coverage.with_skill.status == coverage.baseline.status == "unavailable"
        assert coverage.with_skill.scored is coverage.baseline.scored is None
        assert coverage.with_skill.expected is coverage.baseline.expected is None
        assert "with skill scored trials unavailable" in coverage.text
        assert "baseline scored trials unavailable" in coverage.text
    assert payload == before


@pytest.fixture
def passing_tier3_counts_payload():
    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

    arm = {"execution_status": "succeeded", "execution_errors": [], "expected_attempts": 1, "scored_attempts": 1}
    return build_agent_eval_payload(
        "demo",
        {
            "agent": {
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.4),
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 2,
                "scored_attempts": 2,
                "conditions": {"with_skill": arm, "without_skill": arm},
                "num_trials": 1,
                "num_trials_baseline": 1,
            }
        },
        dataset=[{"id": "case-a"}],
        attempt_policy={"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        use_llm_judge=False,
    )


@pytest.mark.parametrize("tier", [1, 2, 3])
@pytest.mark.parametrize(
    ("adverse", "blocking", "expected", "label"),
    [
        ("fail", True, (1, 0, 0), "Changes required"),
        ("incomplete", True, (0, 1, 0), "Incomplete"),
        ("fail", False, (0, 0, 1), "Passed required checks"),
        ("incomplete", False, (0, 0, 1), "Passed required checks"),
    ],
)
def test_per_result_adverse_counts_handle_mixed_tier_gates(
    tier, adverse, blocking, expected, label, passing_tier3_counts_payload
):
    passing, unfavorable = result(tier), result(tier)
    if tier == 3:
        passing.metadata["agent_eval"] = copy.deepcopy(passing_tier3_counts_payload)
        unfavorable.metadata["agent_eval"] = copy.deepcopy(passing_tier3_counts_payload)
    unfavorable.metadata["gating"]["blocking"] = blocking
    if adverse == "fail":
        unfavorable.add_error("failed required or advisory check")
    else:
        unfavorable.mark_scan_incomplete("scanner")
    sources = [passing, unfavorable]
    before = copy.deepcopy(sources)
    ctx = context(sources)
    assert (ctx.required_failed, ctx.required_incomplete, ctx.advisory_unfavorable) == expected
    assert ctx.command_label == label
    assert sources == before


@pytest.mark.parametrize(
    ("verdict", "advisory", "expected", "label"),
    [
        ("fail", False, (1, 0, 0), "Changes required"),
        ("inconclusive", False, (0, 1, 0), "Incomplete"),
        ("fail", True, (0, 0, 1), "Passed required checks"),
        ("inconclusive", True, (0, 0, 1), "Passed required checks"),
    ],
)
def test_standalone_adverse_counts_follow_existing_headline_override(verdict, advisory, expected, label):
    source = ValidationResult(validator_name="AGENT_EVAL", passed=True, metadata={"agent_eval": {"verdict": verdict}})
    if advisory:
        source.metadata["gating"] = {"tier": 3, "blocking": False}
    before = copy.deepcopy(source)
    ctx = context([source], report_scope="tier3")
    assert (ctx.required_failed, ctx.required_incomplete, ctx.advisory_unfavorable) == expected
    assert ctx.command_label == label
    assert source == before


def test_gate_rejected_pass_shaped_tier3_counts_as_required_failure():
    source = result(3, agent_eval={"verdict": "pass"})
    before = copy.deepcopy(source)
    ctx = context([source])
    assert ctx.result_statuses[id(source)] == "pass"
    assert (ctx.required_failed, ctx.required_incomplete, ctx.advisory_unfavorable) == (1, 0, 0)
    assert ctx.command_label == "Changes required"
    assert source == before


def test_adverse_counts_default_for_existing_context_constructor_and_are_immutable():
    ctx = context([])
    legacy = presentation.FinalRenderContext(
        ctx.command_status,
        ctx.command_label,
        ctx.publication_status,
        ctx.publication_label,
        ctx.tiers,
        ctx.result_statuses,
        ctx.score_scope,
        ctx.coverage,
    )
    assert (legacy.required_failed, legacy.required_incomplete, legacy.advisory_unfavorable) == (0, 0, 0)
    with pytest.raises(FrozenInstanceError):
        legacy.required_failed = 1


def test_adverse_counts_accumulate_across_required_and_advisory_results_in_all_tiers(passing_tier3_counts_payload):
    failed = result(1)
    failed.add_error("required failure")
    missing = result(2)
    missing.mark_scan_incomplete("required-scanner")
    advisory_failed = result(2)
    advisory_failed.metadata["gating"]["blocking"] = False
    advisory_failed.add_error("advisory failure")
    passed = result(3, agent_eval=copy.deepcopy(passing_tier3_counts_payload))
    advisory_missing = result(3, agent_eval=copy.deepcopy(passing_tier3_counts_payload))
    advisory_missing.metadata["gating"]["blocking"] = False
    advisory_missing.mark_scan_incomplete("advisory-scanner")
    sources = [result(1), failed, missing, advisory_failed, passed, advisory_missing]
    before = copy.deepcopy(sources)
    ctx = context(sources)
    assert (ctx.required_failed, ctx.required_incomplete, ctx.advisory_unfavorable) == (1, 1, 2)
    assert ctx.command_status == Status.FAILED
    assert sources == before
