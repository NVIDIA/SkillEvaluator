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
    assert 'data-verdict="fail"' in html
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
    assert re.search(r'header-status-badge[^>]+data-verdict="fail"', html)
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
    assert "3 scored trials: 3 with skill, baseline skipped" in (tmp_path / "projection.html").read_text()
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
