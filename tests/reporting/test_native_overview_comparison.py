# SPDX-License-Identifier: Apache-2.0
"""Overview comparisons retain native detail without inventing scored evidence."""

from __future__ import annotations

import base64
import json
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import html as html_report
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def _comparison(**updates):
    return {
        "status": "partial",
        "label": "Recorded token-only comparison",
        "reason": "Only the same recorded sources can be compared.",
        "reason_code": "shared_sources_only",
        "with_skill": 0.8,
        "baseline": 0.6,
        "lift": 0.15,
        "sources": ["token_efficiency"],
        "measured": {
            "with_skill": {"score": 0.7, "sources": ["skill_efficiency", "token_efficiency"]},
            "baseline": {"score": 0.6, "sources": ["token_efficiency"]},
        },
        **updates,
    }


def _custom_payload(**updates):
    return {
        "execution_status": "succeeded",
        "best_agent": "agent",
        "agents": {"agent": {"execution_status": "succeeded", "metrics_with_skill": []}},
        "dimensions": [],
        "metric_ids": [],
        "evaluator_cards": [
            {"id": "task_quality", "label": "Custom: Task quality", "with_skill": 0.8, "baseline": 0.5, "lift": 0.2}
        ],
        **updates,
    }


def _metric_view(payload):
    helper = getattr(html_report, "_tier3_metric_comparison_view", None)
    assert callable(helper), "The Overview needs a bounded recorded custom-metric projection"
    return helper(payload)


def _context(monkeypatch, payload):
    result = ValidationResult(validator_name="AGENT_EVAL")
    if payload.get("execution_status") == "failed":
        result.add_error("Recorded runner did not complete")
    result.metadata["agent_eval"] = payload
    original = deepcopy(payload)
    reporter = HTMLReporter(include_timestamp=False)
    captured = {}

    def render(**values):
        captured.update(values)
        return "captured"

    monkeypatch.setattr(reporter._env, "get_template", lambda *_args, **_kwargs: SimpleNamespace(render=render))
    assert reporter.render_all([result]) == "captured"
    assert payload == original
    canonical = captured["tier3_canonical_data"]
    if captured["tier3_canonical_encoding"] == "base64":
        canonical = base64.b64decode(canonical)
    assert json.loads(canonical) == original
    return captured


def test_incomplete_standard_metrics_keep_all_known_dimension_comparisons_unavailable():
    payload = {
        "execution_status": "failed",
        "dimensions": [],
        "best_agent": None,
        "agents": {"failed": {"execution_status": "failed", "metrics_with_skill": list(DEFAULT_METRICS)}},
    }
    original = deepcopy(payload)
    rows = html_report._tier3_html_dimensions(payload)
    assert [row["id"] for row in rows] == ["security", "correctness", "discoverability", "effectiveness", "efficiency"]
    for row in rows:
        assert all(row[key] is None for key in ("score", "with_skill", "baseline", "lift", "verdict"))
        assert row["measurement_unavailable"] is True
        assert "did not complete" in " ".join(row["reasoning_bullets"])
    assert payload == original


@pytest.mark.parametrize(
    "metric,dimensions", [("accuracy", ["correctness"]), ("task_quality", []), ("token_efficiency", []), (None, [])]
)
def test_unavailable_rows_use_recorded_public_signal_mapping_only(metric, dimensions):
    payload = {"execution_status": "incomplete", "metric_ids": [metric] if metric else []}
    assert [row["id"] for row in html_report._tier3_html_dimensions(payload)] == dimensions


def test_successful_best_agent_does_not_import_failed_peer_configuration():
    payload = {
        "execution_status": "failed",
        "best_agent": "complete",
        "dimensions": [{"id": "correctness", "score": 0.8, "baseline": 0.5, "lift": 0.2}],
        "agents": {
            "complete": {"execution_status": "succeeded", "metrics_with_skill": ["accuracy"]},
            "failed": {"execution_status": "failed", "metrics_with_skill": list(DEFAULT_METRICS)},
        },
    }
    assert html_report._tier3_html_dimensions(payload) == payload["dimensions"]


def test_recorded_partial_comparison_keeps_its_values_without_recomputing_lift():
    raw = {
        "id": "efficiency",
        "score": 0.7,
        "baseline": None,
        "lift": None,
        "partial": True,
        "comparison": _comparison(),
    }
    original = deepcopy(raw)
    dimension = html_report._sanitize_tier3_dimension(raw)
    assert dimension["partial"] is True
    assert dimension["baseline"] is dimension["lift"] is None
    assert dimension["comparison"]["lift"] == 0.15  # Recorded paired lift, not 0.8 - 0.6.
    assert dimension["comparison"]["measured"]["with_skill"]["score"] == 0.7
    assert raw == original


@pytest.mark.parametrize(
    "invalid",
    ["0.7", True, [], {}, float("nan"), float("inf"), -0.2, 1.2, 10**500],
    ids=["text", "bool", "list", "dict", "nan", "inf", "negative", "too-large", "huge"],
)
def test_native_comparison_numeric_cells_are_typed_finite_and_bounded(invalid):
    dimension = html_report._sanitize_tier3_dimension(
        {
            "comparison": _comparison(
                with_skill=invalid, baseline=invalid, lift=invalid, measured={"baseline": {"score": invalid}}
            )
        }
    )
    comparison = dimension["comparison"]
    assert comparison["with_skill"] is comparison["baseline"] is None
    expected_lift = invalid if type(invalid) in (int, float) and -1 <= invalid <= 1 else None
    assert comparison["lift"] == expected_lift
    assert comparison["measured"]["baseline"]["score"] is None


@pytest.mark.parametrize("invalid", [None, "partial", [], 4])
def test_malformed_comparison_objects_are_not_sent_to_the_template(invalid):
    dimension = html_report._sanitize_tier3_dimension({"partial": "yes", "comparison": invalid})
    assert dimension["partial"] is False
    assert not dimension.get("comparison")


def test_malformed_comparison_text_and_sources_fail_closed():
    dimension = html_report._sanitize_tier3_dimension(
        {"comparison": _comparison(status=[], label={}, reason=[], reason_code=False, sources="accuracy")}
    )
    comparison = dimension["comparison"]
    assert comparison["status"] == "unavailable"
    assert comparison["label"] == comparison["reason"] == comparison["reason_code"] == ""
    assert comparison["sources"] == []
    assert all(comparison[key] is None for key in ("with_skill", "baseline", "lift"))


@pytest.mark.parametrize("status", ["not_applicable", "unavailable", "inconclusive"])
def test_noncomparable_status_does_not_restore_cached_comparison_numbers(status):
    dimension = html_report._sanitize_tier3_dimension({"comparison": _comparison(status=status)})
    assert dimension["comparison"]["status"] == status
    assert all(dimension["comparison"][key] is None for key in ("with_skill", "baseline", "lift"))


def test_never_activated_saved_dimensions_do_not_restore_nested_causal_comparison():
    raw = {
        "best_agent": "agent",
        "agents": {
            "agent": {
                "skill_activation": {"status": "never_activated"},
                "dimensions": [{"id": "efficiency", "score": 0.7, "comparison": _comparison()}],
            }
        },
        "dimensions": [{"id": "efficiency", "score": 0.7, "comparison": _comparison()}],
    }
    original = deepcopy(raw)
    safe = html_report._sanitize_tier3_display_payload(raw)
    for dimension in (safe["dimensions"][0], safe["agents"]["agent"]["dimensions"][0]):
        assert dimension["comparison"]["lift"] is None
        assert dimension["comparison"]["status"] == "inconclusive"
        assert "never activated" in dimension["comparison"]["reason"]
    assert raw == original


@pytest.mark.parametrize("scope", ["baseline_only", "summary_baseline_only", "agent_skipped", "root_skipped"])
def test_skipped_skill_arm_suppresses_root_and_agent_nested_comparison(scope):
    raw = {
        "best_agent": "agent",
        "agents": {"agent": {"dimensions": [{"id": "efficiency", "score": 0.7, "comparison": _comparison()}]}},
        "dimensions": [{"id": "efficiency", "score": 0.7, "comparison": _comparison()}],
    }
    if scope == "baseline_only":
        raw["mode"] = "baseline_only"
    elif scope == "summary_baseline_only":
        raw["summary"] = {"mode": "baseline_only"}
    else:
        owner = raw["agents"]["agent"] if scope == "agent_skipped" else raw
        owner["conditions"] = {"with_skill": {"execution_status": "skipped"}}
    original = deepcopy(raw)
    safe = html_report._sanitize_tier3_display_payload(raw)
    for dimension in (safe["dimensions"][0], safe["agents"]["agent"]["dimensions"][0]):
        comparison = dimension["comparison"]
        assert comparison["with_skill"] is comparison["baseline"] is comparison["lift"] is None
        assert comparison["status"] == "unavailable"
        assert (
            "Baseline-only" in comparison["reason"] if "baseline_only" in scope else "skipped" in comparison["reason"]
        )
        assert "never activated" not in comparison["reason"]
        assert comparison["measured"] == original["dimensions"][0]["comparison"]["measured"]
    assert raw == original


def test_collected_baseline_only_report_does_not_render_cached_causal_comparison(tmp_path):
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
    from skillevaluator.tier3.harbor.collector import collect_harbor_results
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET

    skill = tmp_path / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Synthetic comparison\n---\n# Demo\n")
    jobs = tmp_path / "jobs"
    job = jobs / "demo-opencode-without"
    verifier = job / "case__attempt" / "verifier"
    verifier.mkdir(parents=True)
    reward = {
        "entry_id": "case",
        "metric_set": DEFAULT_METRIC_SET,
        "overall": 0.67,
        **dict.fromkeys(DEFAULT_METRICS, 0.67),
    }
    (verifier / "reward.json").write_text(json.dumps(reward))
    (job / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_retries": 0,
                    "evals": {
                        "agent__model___harbor-tasks": {
                            "n_trials": 1,
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.67": ["case__attempt"]}},
                        }
                    },
                },
            }
        )
    )
    run_dir = tmp_path / "results"
    collected = collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=run_dir,
        jobs_dir=jobs,
        skip_with_skill=True,
        expected_cases=1,
        expected_case_ids=["case"],
        expected_trials=1,
    )
    result = agent_eval_result_from_directory(skill, run_dir, engine_result=collected, use_llm_judge=False)
    payload = result.metadata["agent_eval"]
    assert payload["mode"] == "baseline_only"
    best = payload["agents"][payload["best_agent"]]
    assert best["conditions"]["with_skill"]["execution_status"] == "skipped"
    for dimension in (payload["dimensions"][0], best["dimensions"][0]):
        dimension["comparison"] = _comparison()
    original = deepcopy(payload)
    reporter = HTMLReporter(include_timestamp=False)
    display = reporter._tier3_report_data([result])
    for dimension in (display["dimensions"][0], display["agents"][payload["best_agent"]]["dimensions"][0]):
        assert dimension["comparison"]["lift"] is None
        assert "Baseline-only" in dimension["comparison"]["reason"]
        assert dimension["comparison"]["measured"] == _comparison()["measured"]
    html = reporter.render_all([result])
    overview = html.split('id="tier3-page-overview"', 1)[1].split('id="tier3-page-trials"', 1)[0]
    assert "+0.15" not in overview
    assert "baseline-only" in overview.lower()
    embedded = re.search(r'<script(?P<attrs>[^>]*\bid="tier3-full"[^>]*)>(?P<data>.*?)</script>', html, re.DOTALL)
    assert embedded is not None
    canonical = embedded["data"]
    if 'data-encoding="base64"' in embedded["attrs"]:
        canonical = base64.b64decode(canonical)
    assert json.loads(canonical) == original
    assert payload == original


def test_custom_comparison_uses_recorded_card_and_preserves_paired_lift():
    payload = _custom_payload()
    original = deepcopy(payload)
    assert _metric_view(payload) == {
        "rows": [
            {
                "id": "task_quality",
                "label": "Custom: Task quality",
                "score": 0.8,
                "baseline": 0.5,
                "lift": 0.2,
                "reason": "",
            }
        ],
        "omitted": 0,
    }
    assert html_report._tier3_html_dimensions(payload) == []
    assert payload == original


def test_custom_metric_comparison_does_not_reconstruct_missing_baseline_from_rewards():
    payload = _custom_payload()
    payload["evaluator_cards"][0].update(baseline=None, lift=None)
    payload["agents"]["agent"]["trials_baseline"] = [{"custom_metrics": {"task_quality": 0.5}}]
    row = _metric_view(payload)["rows"][0]
    assert row["score"] == 0.8 and row["baseline"] is row["lift"] is None
    assert "baseline" in row["reason"].lower()


def test_standard_metric_dimensions_do_not_get_a_duplicate_custom_table():
    payload = _custom_payload(metric_ids=["accuracy"])
    assert _metric_view(payload) == {"rows": [], "omitted": 0}


def test_custom_table_has_explicit_omission_count_and_no_identifier_collisions():
    cards = [{"id": f"custom_{index}", "with_skill": 0.5} for index in range(70)]
    view = _metric_view(_custom_payload(evaluator_cards=cards))
    assert len(view["rows"]) == 64 and view["omitted"] == 6
    assert view["rows"][0]["id"] == "custom_0"
    assert view["rows"][-1]["id"] == "custom_63"
    collision = _custom_payload(
        evaluator_cards=[{"id": "Custom", "with_skill": 0.5}, {"id": "custom", "with_skill": 0.8}]
    )
    assert _metric_view(collision) == {"rows": [], "omitted": 0}


def test_baseline_only_custom_table_does_not_restore_a_skill_score_or_lift():
    row = _metric_view(_custom_payload(mode="baseline_only"))["rows"][0]
    assert row["score"] is row["lift"] is None and row["baseline"] == 0.5
    assert "Baseline-only" in row["reason"]


@pytest.mark.parametrize("scope", ["summary_baseline_only", "agent_skipped", "root_skipped"])
def test_custom_table_uses_the_same_explicit_skill_arm_guard(scope):
    payload = _custom_payload()
    if scope == "summary_baseline_only":
        payload["summary"] = {"mode": "baseline_only"}
    else:
        owner = payload["agents"]["agent"] if scope == "agent_skipped" else payload
        owner["conditions"] = {"with_skill": {"execution_status": "skipped"}}
    original = deepcopy(payload)
    row = _metric_view(payload)["rows"][0]
    assert row["score"] is row["lift"] is None and row["baseline"] == 0.5
    assert "Baseline-only" in row["reason"] if "baseline_only" in scope else "skipped" in row["reason"]
    assert "never activated" not in row["reason"]
    assert payload == original


def test_never_activated_custom_table_keeps_diagnostic_scores_but_no_causal_lift():
    payload = _custom_payload()
    payload["agents"]["agent"]["skill_activation"] = {"status": "never_activated"}
    row = _metric_view(payload)["rows"][0]
    assert row["score"] == 0.8 and row["baseline"] == 0.5 and row["lift"] is None
    assert "never activated" in row["reason"] and "diagnostic" in row["reason"]


def test_failed_custom_agent_does_not_gain_authoritative_scores_from_cached_cards():
    payload = _custom_payload(execution_status="failed")
    payload["agents"]["agent"]["execution_status"] = "failed"
    row = _metric_view(payload)["rows"][0]
    assert row["score"] is row["baseline"] is row["lift"] is None
    assert "did not complete" in row["reason"]


def test_incomplete_context_keeps_configured_dimensions_without_mutating_canonical(monkeypatch):
    payload = {
        "execution_status": "failed",
        "verdict": "fail",
        "expected_attempts": 36,
        "scored_attempts": 28,
        "agents": {
            "agent": {
                "execution_status": "failed",
                "expected_attempts": 36,
                "scored_attempts": 28,
                "metrics_with_skill": list(DEFAULT_METRICS),
            }
        },
    }
    context = _context(monkeypatch, payload)
    assert len(context["tier3_display_dimensions"]) == 5
    assert all(row["score"] is None for row in context["tier3_display_dimensions"])
    assert context["tier3_metric_comparison_rows"] == []
    assert context["tier3_metric_comparison_omitted"] == 0


def test_source_rejected_context_does_not_restore_untrusted_comparison_values(monkeypatch):
    payload = _custom_payload(
        execution_status="succeeded",
        verdict="pass",
        provenance={
            "source": "tier3_source_preflight",
            "source_validation": [{"path": "evals.json", "status": "error", "message": "Dataset rejected"}],
        },
    )
    context = _context(monkeypatch, payload)
    assert context["tier3_display_dimensions"] == []
    assert context["tier3_metric_comparison_rows"] == []


def test_mixed_execution_context_keeps_recorded_completed_agent_comparisons(monkeypatch):
    agent = {
        "execution_status": "succeeded",
        "expected_attempts": 2,
        "scored_attempts": 2,
        "num_trials": 1,
        "num_trials_baseline": 1,
        "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.9),
        "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.5),
        "rewards": [{**dict.fromkeys(DEFAULT_METRICS, 0.9), "entry_id": "case"}],
        "rewards_baseline": [{**dict.fromkeys(DEFAULT_METRICS, 0.5), "entry_id": "case"}],
        "rewards_complete": True,
        "rewards_baseline_complete": True,
    }
    payload = build_agent_eval_payload(
        "demo",
        {
            "complete": agent,
            "failed": {"execution_status": "failed", "execution_errors": ["Runner failed"], "expected_attempts": 2},
        },
        use_llm_judge=False,
    )
    context = _context(monkeypatch, payload)
    assert context["tier3"]["execution_status"] == "failed"
    assert context["tier3"]["best_agent"] == "complete"
    assert [row["score"] for row in context["tier3_display_dimensions"]] == [0.9] * 5
