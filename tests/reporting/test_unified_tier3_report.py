# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path

import pytest

from skillevaluator.evaluation.tier3_report import (
    _attach_harbor_evidence_to_conclusions,
    _attach_harbor_evidence_to_recommendations,
    _attach_harbor_evidence_to_suggestions_v2,
    _evaluator_cards,
    _metric_evidence,
    _normalize_trials,
    _raw_trial_rewards,
    _ReportBudget,
    agent_eval_result_from_directory,
    build_agent_eval_payload,
    render_agent_eval_html_report,
)
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter, JSONReporter
from skillevaluator.reporting import html as html_module
from skillevaluator.reporting.html import PackageLoader, _compact_json
from skillevaluator.tier3.harbor.collector import _paired_pass_comparison, _wilson_score_interval
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.harbor.report_data import build_dataset_snapshot


def _write_summary(
    run_dir: Path,
    *,
    status: str = "succeeded",
    errors: list[str] | None = None,
    score: float = 1.0,
) -> None:
    summary = run_dir / "opencode" / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    scores = dict.fromkeys(DEFAULT_METRICS, score) if status == "succeeded" else {}
    summary.write_text(
        json.dumps(
            {
                "scores": scores,
                "metrics": list(DEFAULT_METRICS),
                "num_trials": 1 if status == "succeeded" else 0,
                "execution_status": status,
                "execution_errors": errors or [],
                "expected_attempts": 1,
                "scored_attempts": 1 if status == "succeeded" else 0,
            }
        ),
        encoding="utf-8",
    )


def _write_authenticated_pre_status_summary(run_dir: Path, *, score: float = 0.8) -> None:
    summary = run_dir / "opencode" / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "harbor": {"n_attempts": {"value": 1, "source": "ACES default"}},
                "task_source": "evals_json",
                "agents": {
                    "opencode": {
                        "agent": "opencode",
                        "model": "test-model",
                        "source": "test default",
                        "occurrence": "1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "agent": "opencode",
                "model": "test-model",
                "model_source": "test default",
                "scores": {"security": score},
                "custom_scores": {},
                "metric_set": "aces-default-v2",
                "metrics": ["security"],
                "dimensions": {"safety": {"score": score, "sources": {"security": 1.0}}},
                "num_trials": 1,
                "pass_at_k": {
                    "k": 1,
                    "pass_threshold": 0.5,
                    "stop_on_pass": False,
                    "passed_cases": 1,
                    "failed_cases": 0,
                    "total_cases": 1,
                    "rate": 1.0,
                    "attempts_used": 1,
                    "max_attempts_possible": 1,
                    "avg_attempts_used": 1.0,
                    "extra_cases": [],
                    "cases": {
                        "case": {
                            "passed": True,
                            "first_pass_attempt": 1,
                            "attempts_used": 1,
                            "attempts_skipped": 0,
                            "attempts_missing": 0,
                            "best_score": score,
                            "attempts": [
                                {
                                    "attempt": 1,
                                    "trial": "case__attempt1",
                                    "score": score,
                                    "passed": True,
                                }
                            ],
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _render_agent_payload(payload: dict) -> str:
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload
    return HTMLReporter(include_timestamp=False).render_all([result])


def _tier3_page(html: str, page: str, next_page: str) -> str:
    return html.split(f'id="tier3-page-{page}"', 1)[1].split(f'id="tier3-page-{next_page}"', 1)[0]


def _embedded_tier3_payload(html: str) -> dict:
    match = re.search(
        r'<script type="application/json" id="tier3-full"(?P<attrs>[^>]*)>(?P<body>.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    body = match.group("body").strip()
    if 'data-encoding="base64"' in match.group("attrs"):
        body = base64.b64decode(body).decode("utf-8")
    return json.loads(body)


def test_normalize_trials_preserves_only_trusted_standard_invocation_evidence() -> None:
    standard, custom_spoof, untrusted = _normalize_trials(
        [
            {
                "entry_id": "standard",
                "metric_set": "skill-evaluator-default-v2",
                "skill_execution": 0.0,
                "skill_invoked": True,
                "routing_passed": False,
                "invocation_evidence_source": "trajectory",
            },
            {
                "entry_id": "custom",
                "metric_set": "custom-only",
                "overall": 1.0,
                "skill_execution": 0.0,
                "skill_invoked": True,
                "routing_passed": False,
                "invocation_evidence_source": "trajectory",
            },
            {
                "entry_id": "untrusted",
                "metric_set": "skill-evaluator-default-v2",
                "skill_execution": 0.0,
                "skill_invoked": 1,
                "routing_passed": 0,
                "invocation_evidence_source": "reward",
            },
        ],
        list(DEFAULT_METRICS),
    )

    assert standard["skill_invoked"] is True
    assert standard["routing_passed"] is False
    assert standard["invocation_evidence_source"] == "trajectory"
    assert "skill_execution" not in custom_spoof["scores"]
    for trial in (custom_spoof, untrusted):
        assert "skill_invoked" not in trial
        assert "routing_passed" not in trial
        assert "invocation_evidence_source" not in trial


def test_case_grounded_harbor_links_win_over_positional_fallback() -> None:
    evidence_links = [
        {"url": "https://harbor.example.test/case-1", "entry_id": "case-1", "label": "case-1"},
        {"url": "https://harbor.example.test/case-2", "entry_id": "case-2", "label": "case-2"},
    ]
    recommendations = _attach_harbor_evidence_to_recommendations(
        [
            {
                "title": "Fix case-2",
                "evidence_case_ids": ["case-2"],
                "evidence": evidence_links[0],
            }
        ],
        evidence_links,
    )
    suggestions = _attach_harbor_evidence_to_suggestions_v2(
        [{"recommendation": "Fix case-2", "evidence_case_ids": ["case-2"]}],
        evidence_links,
    )
    conclusions = _attach_harbor_evidence_to_conclusions(
        [{"title": "Failure in case-2", "evidence_case_ids": ["case-2"]}],
        evidence_links,
    )

    assert conclusions[0]["evidence"]["entry_id"] == "case-2"
    assert recommendations[0]["evidence"]["entry_id"] == "case-2"
    assert suggestions[0]["harbor_evidence"]["entry_id"] == "case-2"


def test_conclusions_without_case_grounding_do_not_receive_positional_evidence() -> None:
    evidence_links = [{"url": "https://harbor.example.test/case-1", "entry_id": "case-1", "label": "case-1"}]

    conclusions = _attach_harbor_evidence_to_conclusions(
        [{"title": "Aggregate result", "message": "No case-specific claim"}],
        evidence_links,
    )

    assert "evidence" not in conclusions[0]


def test_package_loader_reads_template_resources_as_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    class TemplateResource:
        def joinpath(self, _path: str) -> TemplateResource:
            return self

        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return "Per-trial evidence \u00d72"

    monkeypatch.setattr(html_module.resources, "files", lambda _package: TemplateResource())

    source, _, _ = PackageLoader("skillevaluator.reporting", "templates").get_source(
        html_module.Environment(), "report.html.j2"
    )

    assert source == "Per-trial evidence \u00d72"


def test_package_loader_fallback_reads_template_resources_as_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable_files(_package: str) -> None:
        raise AttributeError

    monkeypatch.setattr(html_module.resources, "files", unavailable_files)

    source, _, _ = PackageLoader("skillevaluator.reporting", "templates").get_source(
        html_module.Environment(), "report.html.j2"
    )

    assert "\u00d7" in source


def test_standalone_tier3_uses_generic_tier3_only_report(tmp_path: Path) -> None:
    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "results" / "20260709_120000"
    _write_summary(run_dir)

    report = render_agent_eval_html_report(skill, run_dir, use_llm_judge=False)
    html = report.read_text(encoding="utf-8")

    assert report == run_dir / "report.html"
    assert "Tier 3: Live Agent Evaluation" in html
    assert "Tier 1: Security and Static Validation" not in html
    assert "Tier 2: Deduplication" not in html
    assert 'data-tier3-tab="trials"' in html
    assert "Diagnostics" in html
    assert "SkillEvaluator" in html


def test_authenticated_pre_status_rerender_preserves_historical_scores(tmp_path: Path) -> None:
    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "results" / "20260709_120008"
    _write_authenticated_pre_status_summary(run_dir)

    tier3 = agent_eval_result_from_directory(skill, run_dir, use_llm_judge=False)

    assert tier3 is not None
    payload = tier3.metadata["agent_eval"]
    assert payload["execution_status"] == "succeeded"
    assert payload["overall_score"] is not None
    assert payload["agents"]["opencode"]["evaluators"]["security"]["with_skill"] == 0.8


def test_canonical_report_prefers_agentskills_dataset_fields() -> None:
    payload = build_agent_eval_payload(
        "hld-documents",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0, "goal_accuracy": 1.0},
                "rewards": [{"entry_id": "case-1", "security": 1.0, "goal_accuracy": 1.0}],
            }
        },
        dataset=[
            {
                "id": "case-1",
                "prompt": "Use hld-documents for this design.",
                "question": "legacy fallback should not be shown",
                "expected_output": "A complete HLD document is produced.",
                "ground_truth": "legacy fallback should not be shown",
                "assertions": ["The agent reads SKILL.md.", "The agent writes the HLD sections."],
                "expected_behavior": ["legacy fallback should not be shown"],
                "expected_skill": "hld-documents",
            }
        ],
        use_llm_judge=False,
    )
    assert payload is not None

    dataset_html = _tier3_page(_render_agent_payload(payload), "dataset", "suggestions")

    assert "<h2>AgentSkills Dataset</h2>" in dataset_html
    assert "1 AgentSkills eval case used for this Tier 3 run." in dataset_html
    assert '<span class="t3-ds-label">Prompt</span>' in dataset_html
    assert '<span class="t3-ds-label">Expected Output</span>' in dataset_html
    assert '<span class="t3-ds-label">Assertions</span>' in dataset_html
    assert "Use hld-documents for this design." in dataset_html
    assert "A complete HLD document is produced." in dataset_html
    assert "The agent reads SKILL.md." in dataset_html
    assert "legacy fallback should not be shown" not in dataset_html


def test_standalone_report_loads_staged_legacy_dataset_with_agentskills_labels(tmp_path: Path) -> None:
    skill = tmp_path / "hld-documents"
    skill.mkdir()
    run_dir = tmp_path / "results" / "20260709_120006"
    _write_summary(run_dir)
    entry_dir = run_dir / "_harbor-tasks" / "hld-documents-001" / "tests"
    entry_dir.mkdir(parents=True)
    (entry_dir / "entry.json").write_text(
        json.dumps(
            {
                "id": "hld-documents-001",
                "question": "Create an HLD for packet reordering.",
                "ground_truth": "A complete HLD document is produced.",
                "expected_behavior": [
                    "The agent reads the hld-documents skill.",
                    "The agent includes hardware register tables.",
                ],
                "expected_skill": "hld-documents",
            }
        ),
        encoding="utf-8",
    )

    report = render_agent_eval_html_report(skill, run_dir, use_llm_judge=False)
    dataset_html = _tier3_page(report.read_text(encoding="utf-8"), "dataset", "suggestions")

    assert "<h2>AgentSkills Dataset</h2>" in dataset_html
    assert '<span class="t3-ds-label">Prompt</span>' in dataset_html
    assert '<span class="t3-ds-label">Expected Output</span>' in dataset_html
    assert '<span class="t3-ds-label">Assertions</span>' in dataset_html
    assert "Create an HLD for packet reordering." in dataset_html
    assert "A complete HLD document is produced." in dataset_html
    assert "The agent includes hardware register tables." in dataset_html


def test_canonical_html_renders_evaluator_evidence_and_custom_metric_details() -> None:
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0, "goal_accuracy": 0.2},
                "custom_with_skill": {"domain_quality": 0.9},
                "rewards": [
                    {
                        "entry_id": "case-1",
                        "security": 1.0,
                        "goal_accuracy": 0.2,
                        "custom_metrics": {"domain_quality": 0.9},
                        "details": {
                            "goal_accuracy": {
                                "reason": "results file not produced",
                                "evidence_refs": [
                                    {
                                        "source": "trajectory.json",
                                        "json_pointer": "/steps/14",
                                        "kind": "tool_call",
                                    }
                                ],
                            },
                            "domain_quality": {
                                "score": 0.9,
                                "reason": "custom domain matched",
                                "evidence_refs": [
                                    {
                                        "source": "custom_reward.json",
                                        "json_pointer": "/details/domain_quality",
                                        "kind": "custom_metric",
                                    }
                                ],
                            },
                        },
                    }
                ],
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None
    assert "report_truncation" not in payload

    agents_html = _tier3_page(_render_agent_payload(payload), "agents", "dataset")

    assert "results file not produced" in agents_html
    assert "trajectory.json/steps/14" in agents_html
    assert "domain_quality" in agents_html
    assert "custom domain matched" in agents_html
    assert "custom_reward.json/details/domain_quality" in agents_html


def test_canonical_html_tolerates_legacy_string_evidence_refs() -> None:
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0, "goal_accuracy": 0.3},
                "rewards": [
                    {
                        "entry_id": "legacy-001",
                        "security": 1.0,
                        "goal_accuracy": 0.3,
                        "details": {
                            "goal_accuracy": {
                                "reason": "failure",
                                "evidence_refs": ["trajectory.json#/steps/14"],
                            }
                        },
                    }
                ],
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None

    agents_html = _tier3_page(_render_agent_payload(payload), "agents", "dataset")

    assert "failure" in agents_html
    assert "trajectory.json#/steps/14" in agents_html


def test_canonical_payload_deduplicates_repeated_metric_evidence() -> None:
    reward = {
        "entry_id": "case-1",
        "security": 1.0,
        "goal_accuracy": 0.5,
        "details": {"goal_accuracy": {"reason": "same representative evidence"}},
    }
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 2,
                "scored_attempts": 2,
                "with_skill": {"security": 1.0, "goal_accuracy": 0.5},
                "rewards": [dict(reward), dict(reward)],
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None

    goal_card = next(card for card in payload["agents"]["codex"]["evaluator_cards"] if card["id"] == "goal_accuracy")
    assert goal_card["evidence"] == [
        {
            "entry_id": "case-1",
            "score": 0.5,
            "notes": ["same representative evidence"],
            "failures": [],
            "checks": [],
            "evidence_refs": [],
            "occurrences": 2,
        }
    ]
    agents_html = _tier3_page(_render_agent_payload(payload), "agents", "dataset")
    assert "\u00d72" in agents_html
    assert "1 cases, 2 trials" in agents_html


def test_deduplicated_evidence_average_is_weighted_by_occurrences() -> None:
    rewards = [
        {
            "entry_id": "case-1",
            "security": 1.0,
            "goal_accuracy": score,
            "details": {"goal_accuracy": {"reason": reason}},
        }
        for score, reason in [(0.0, "same failure"), (0.0, "same failure"), (1.0, "eventual success")]
    ]
    payload = build_agent_eval_payload(
        "demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 3,
                "scored_attempts": 3,
                "with_skill": {"security": 1.0, "goal_accuracy": 1 / 3},
                "rewards": rewards,
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None

    agents_html = _tier3_page(_render_agent_payload(payload), "agents", "dataset")
    evidence_row = agents_html.split('<span class="t3-ev-id">case-1</span>', 1)[1].split("</li>", 1)[0]

    assert "1 cases, 3 trials" in agents_html
    assert ">0.33<" in evidence_row


def test_metric_evidence_does_not_scan_rewards_after_global_budget_is_exhausted() -> None:
    class NeverIteratedRewards:
        def __len__(self) -> int:
            return 50_000

        def __iter__(self):
            raise AssertionError("an exhausted evidence budget must not scan reward payloads")

    budget = _ReportBudget(evidence_remaining=0)

    evidence = _metric_evidence("goal_accuracy", NeverIteratedRewards(), budget)  # type: ignore[arg-type]

    assert evidence == []
    assert budget.omitted["evidence_entries"] == 50_000


def test_metric_evidence_bounds_work_after_per_card_output_is_full() -> None:
    class BoundedRewards:
        def __len__(self) -> int:
            return 50_000

        def __iter__(self):
            for index in range(64):
                yield {
                    "entry_id": f"case-{index}",
                    "goal_accuracy": 1.0,
                    "details": {"goal_accuracy": {"reason": f"evidence-{index}"}},
                }
            raise AssertionError("per-card evidence processing must have a finite scan budget")

    budget = _ReportBudget(evidence_remaining=256)

    evidence = _metric_evidence("goal_accuracy", BoundedRewards(), budget)  # type: ignore[arg-type]

    assert len(evidence) == 16
    assert budget.omitted["evidence_entries"] == 50_000 - len(evidence)


def test_exhausted_card_and_raw_reward_budgets_do_not_iterate_rewards() -> None:
    class NeverIteratedRewards:
        def __len__(self) -> int:
            return 50_000

        def __iter__(self):
            raise AssertionError("an exhausted report budget must not scan reward payloads")

    rewards = NeverIteratedRewards()
    card_budget = _ReportBudget(cards_remaining=0)
    raw_budget = _ReportBudget(raw_rewards_remaining=0)

    cards = _evaluator_cards(  # type: ignore[arg-type]
        {},
        rewards=rewards,
        custom_with_skill={},
        custom_without_skill={},
        custom_lift={},
        report_budget=card_budget,
    )
    raw = _raw_trial_rewards({"rewards": rewards}, raw_budget)

    assert cards == []
    assert raw == []
    assert card_budget.omitted["custom_metric_discovery_trials"] == 50_000
    assert raw_budget.omitted["raw_trial_rewards"] == 50_000


def test_custom_metric_name_discovery_is_capped_before_sorting() -> None:
    class ExplodingMetricMap(dict[str, float]):
        def __iter__(self):
            yield "metric-b"
            yield "metric-a"
            raise AssertionError("custom metric discovery must stop before a third key")

    custom_scores = ExplodingMetricMap({"metric-a": 0.8, "metric-b": 0.9, "metric-c": 1.0})
    budget = _ReportBudget(cards_remaining=1)

    cards = _evaluator_cards(
        {},
        rewards=[],
        custom_with_skill=custom_scores,
        custom_without_skill={},
        custom_lift={},
        report_budget=budget,
    )

    assert len(cards) == 1
    assert cards[0]["id"] == "metric-a"
    assert budget.omitted["evaluator_cards"] == 2


def test_reward_custom_metric_discovery_and_evidence_do_not_materialize_full_maps() -> None:
    class ExplodingMetricMap(dict[str, float]):
        def items(self):
            yield "metric-b", 0.9
            yield "metric-a", 0.8
            raise AssertionError("custom metric discovery must stop before a third reward-derived metric")

    custom_metrics = ExplodingMetricMap({"metric-a": 0.8, "metric-b": 0.9, "metric-c": 1.0})
    budget = _ReportBudget(cards_remaining=1)

    cards = _evaluator_cards(
        {},
        rewards=[
            {
                "entry_id": "case-1",
                "custom_metrics": custom_metrics,
                "custom_details": {"metric-a": {"reason": "bounded evidence"}},
            }
        ],
        custom_with_skill={},
        custom_without_skill={},
        custom_lift={},
        report_budget=budget,
    )

    assert len(cards) == 1
    assert cards[0]["id"] == "metric-a"
    assert cards[0]["evidence"][0]["score"] == 0.8
    assert budget.omitted["evaluator_cards"] > 0


def test_raw_reward_projection_bulk_stops_when_field_budget_is_full() -> None:
    class ExplodingReward(dict[str, float]):
        def items(self):
            for index in range(97):
                yield f"field-{index:03d}", float(index)
            raise AssertionError("raw reward projection must stop after detecting the exhausted field budget")

    reward = ExplodingReward({f"field-{index:03d}": float(index) for index in range(200)})
    budget = _ReportBudget(raw_rewards_remaining=1)

    projected = _raw_trial_rewards({"rewards": [reward]}, budget)

    assert len(projected[0]) == 96
    assert budget.omitted["raw_reward_fields"] == 104


def test_canonical_payload_bounds_high_cardinality_custom_report_details() -> None:
    metric_count = 100
    trial_count = 100
    custom_scores = {f"custom_metric_{index:03d}": 0.75 for index in range(metric_count)}
    custom_details = {
        metric: {
            "reason": f"representative evidence for {metric} " + ("x" * 2_000),
            "evidence_refs": [{"source": "custom_reward.json", "json_pointer": f"/details/{metric}"}],
        }
        for metric in custom_scores
    }
    rewards = [
        {
            "entry_id": f"case-{index:03d}",
            "security": 1.0,
            "custom_metrics": dict(custom_scores),
            "custom_details": custom_details,
        }
        for index in range(trial_count)
    ]

    payload = build_agent_eval_payload(
        "high-cardinality-demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": trial_count,
                "scored_attempts": trial_count,
                "with_skill": {"security": 1.0},
                "custom_with_skill": custom_scores,
                "rewards": rewards,
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None

    cards = payload["agents"]["codex"]["evaluator_cards"]
    raw_rewards = payload["provenance"]["raw_trial_rewards"]["codex"]
    encoded = json.dumps(payload, separators=(",", ":")).encode()

    assert len(cards) <= 64
    assert all(len(card["evidence"]) <= 16 for card in cards)
    assert sum(len(card["evidence"]) for card in cards) <= 256
    assert all("custom_details" not in reward for reward in raw_rewards)
    assert any(card["evidence"] for card in cards), "representative evidence should survive bounded reporting"
    assert payload["report_truncation"]["truncated"] is True
    assert payload["report_truncation"]["omitted"]["evaluator_cards"] > 0
    assert payload["report_truncation"]["omitted"]["evidence_entries"] > 0
    assert payload["report_truncation"]["payload_budget_bytes"] == 2 * 1024 * 1024
    assert len(encoded) <= payload["report_truncation"]["payload_budget_bytes"]

    html = _render_agent_payload(payload)
    assert "Embedded report details were bounded" in html
    assert "evaluator cards omitted:" in html
    assert str(payload["report_truncation"]["omitted"]["evaluator_cards"]) in html


def test_global_report_budget_prioritizes_best_agent_details() -> None:
    custom_scores = {f"custom_metric_{index:03d}": 0.2 for index in range(80)}
    low_agent_rewards = [
        {
            "entry_id": f"low-{index:03d}",
            "security": 0.1,
            "custom_metrics": custom_scores,
        }
        for index in range(256)
    ]
    payload = build_agent_eval_payload(
        "multi-agent-budget",
        {
            "aaa": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": len(low_agent_rewards),
                "scored_attempts": len(low_agent_rewards),
                "with_skill": {"security": 0.1},
                "custom_with_skill": custom_scores,
                "rewards": low_agent_rewards,
            },
            "zzz": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0, "goal_accuracy": 1.0},
                "rewards": [
                    {
                        "entry_id": "best-case",
                        "security": 1.0,
                        "goal_accuracy": 1.0,
                        "details": {"goal_accuracy": {"reason": "best-agent evidence"}},
                    }
                ],
            },
        },
        use_llm_judge=False,
    )
    assert payload is not None

    assert payload["best_agent"] == "zzz"
    best_cards = payload["agents"]["zzz"]["evaluator_cards"]
    assert best_cards
    assert payload["evaluator_cards"] == best_cards
    assert any(card["evidence"] for card in best_cards)
    assert payload["provenance"]["raw_trial_rewards"]["zzz"]


def test_payload_prunes_oversized_non_best_conditions_before_best_evidence() -> None:
    payload = build_agent_eval_payload(
        "multi-agent-pruning",
        {
            "aaa": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 0.1},
                "conditions": {"with_skill": {"diagnostic": "x" * (3 * 1024 * 1024)}},
                "rewards": [{"entry_id": "low-case", "security": 0.1}],
            },
            "zzz": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0, "goal_accuracy": 1.0},
                "rewards": [
                    {
                        "entry_id": "best-case",
                        "security": 1.0,
                        "goal_accuracy": 1.0,
                        "details": {"goal_accuracy": {"reason": "best-agent evidence"}},
                    }
                ],
            },
        },
        use_llm_judge=False,
    )
    assert payload is not None

    best_cards = payload["agents"]["zzz"]["evaluator_cards"]
    assert any(card["evidence"] for card in best_cards)
    assert any(card["evidence"] for card in payload["evaluator_cards"])
    assert payload["provenance"]["raw_trial_rewards"]["zzz"]
    assert payload["agents"]["aaa"]["conditions"] == {}
    assert payload["report_truncation"]["omitted"]["non_best_agent_details"] > 0


def test_non_finite_report_numbers_are_sanitized_before_canonical_json() -> None:
    payload = build_agent_eval_payload(
        "finite-json",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0, "goal_accuracy": 0.5},
                "custom_with_skill": {"unrepresentable": 10**10_000},
                "lift": {"security": float("inf")},
                "conditions": {"with_skill": {"score": float("-inf")}},
                "rewards": [
                    {
                        "entry_id": "case-1",
                        "security": 1.0,
                        "goal_accuracy": float("nan"),
                        "overall": float("inf"),
                        "details": {"goal_accuracy": {"reason": "invalid score"}},
                    }
                ],
            }
        },
        dataset=[{"id": "case-1", "score": float("inf")}],
        comparison={"delta": float("nan")},
        runtime_seconds=float("inf"),
        use_llm_judge=False,
    )
    assert payload is not None

    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert payload["runtime_seconds"] == 0.0
    assert payload["dataset"][0]["score"] is None
    assert payload["provenance"]["comparison"]["delta"] is None
    goal_card = next(card for card in payload["agents"]["codex"]["evaluator_cards"] if card["id"] == "goal_accuracy")
    assert goal_card["evidence"][0]["score"] is None
    assert not any(card["id"] == "unrepresentable" for card in payload["agents"]["codex"]["evaluator_cards"])
    assert all(
        not isinstance(value, float) or math.isfinite(value)
        for value in (payload["overall_score"], payload["overall_lift"], payload["composite_lift"])
    )

    html = _render_agent_payload(payload)
    assert _embedded_tier3_payload(html)["dataset"][0]["score"] is None


def test_canonical_html_serializer_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        _compact_json({"score": float("nan")})

    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = {"score": float("inf")}
    with pytest.raises(ValueError, match="Out of range float values"):
        JSONReporter(include_timestamp=False).render_all([result])


def test_sampled_evidence_carries_lower_bound_metadata_and_honest_wording() -> None:
    rewards = [
        {
            "entry_id": "case-1",
            "security": 1.0,
            "goal_accuracy": 0.5,
            "details": {"goal_accuracy": {"reason": "same sampled evidence"}},
        }
        for _ in range(100)
    ]
    payload = build_agent_eval_payload(
        "sampled-evidence",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 100,
                "scored_attempts": 100,
                "with_skill": {"security": 1.0, "goal_accuracy": 0.5},
                "rewards": rewards,
            }
        },
        use_llm_judge=False,
    )
    assert payload is not None

    goal_card = next(card for card in payload["agents"]["codex"]["evaluator_cards"] if card["id"] == "goal_accuracy")
    assert goal_card["evidence_sampling"] == {
        "truncated": True,
        "counts_are_lower_bounds": True,
        "scanned_trials": 64,
        "total_trials": 100,
        "represented_cases": 1,
        "represented_trials": 64,
    }

    agents_html = _tier3_page(_render_agent_payload(payload), "agents", "dataset")
    assert "&gt;=64 of 100 trials" in agents_html
    assert "1 cases, 64 trials" not in agents_html


def test_escape_heavy_payload_has_bounded_html_and_one_recoverable_canonical_copy() -> None:
    escape_heavy_prompt = "<" * 1_500_000
    payload = build_agent_eval_payload(
        "escape-heavy",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0},
                "rewards": [{"entry_id": "case-1", "security": 1.0}],
            }
        },
        dataset=[{"id": "case-1", "prompt": escape_heavy_prompt}],
        use_llm_judge=False,
    )
    assert payload is not None

    html = _render_agent_payload(payload)

    assert len(html.encode("utf-8")) <= 4 * 1024 * 1024
    assert html.count('id="tier3-full"') == 1
    assert 'id="tier3-raw-evaluators"' not in html
    assert 'id="tier3-raw-lift"' not in html
    assert 'id="tier3-comparison"' not in html
    assert _embedded_tier3_payload(html)["dataset"][0]["prompt"] == escape_heavy_prompt
    assert "HTML preview omitted" in html
    assert "characters" in html


def test_canonical_payload_enforces_total_serialized_budget() -> None:
    payload = build_agent_eval_payload(
        "oversized-comparison-demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0},
                "rewards": [{"entry_id": "case-1", "security": 1.0}],
            }
        },
        comparison={"unrendered_blob": "x" * (3 * 1024 * 1024)},
        use_llm_judge=False,
    )
    assert payload is not None

    encoded = json.dumps(payload, separators=(",", ":")).encode()
    truncation = payload["report_truncation"]

    assert payload["summary"]["execution_status"] == "succeeded"
    assert payload["provenance"]["comparison"] == {}
    assert truncation["truncated"] is True
    assert truncation["omitted"]["comparison_payloads"] == 1
    assert len(encoded) <= truncation["payload_budget_bytes"]


def test_unpaired_id_artifact_survives_loader_report_budget_and_html(tmp_path: Path) -> None:
    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "results" / "20260825_120000"
    agent_dir = run_dir / "opencode"
    case_count = 2_400

    for variant, rate, passed_cases in (("with-skill", 1.0, 1), ("without-skill", 0.0, 0)):
        summary_path = agent_dir / variant / "summary.json"
        summary_path.parent.mkdir(parents=True)
        score = rate
        summary_path.write_text(
            json.dumps(
                {
                    "scores": dict.fromkeys(DEFAULT_METRICS, score),
                    "metrics": list(DEFAULT_METRICS),
                    "num_trials": 1,
                    "execution_status": "succeeded",
                    "execution_errors": [],
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                    "overall_score": score,
                    "pass_at_k": {
                        "rate": rate,
                        "passed_cases": passed_cases,
                        "total_cases": 1,
                        "attempts_used": 1,
                        "max_attempts_possible": 1,
                    },
                }
            ),
            encoding="utf-8",
        )

    full_with_ids = [f"with-{index:04d}" for index in range(case_count)]
    full_without_ids = [f"without-{index:04d}" for index in range(case_count)]
    pass_lift_path = agent_dir / "pass_at_k_lift.json"
    pass_lift_path.write_text(
        json.dumps(
            {
                "with_skill": 1.0,
                "without_skill": 0.0,
                "delta": 1.0,
                "paired_comparison": {
                    "pairing_status": "unavailable",
                    "paired_cases": 0,
                    "with_skill_unpaired_case_ids": full_with_ids,
                    "without_skill_unpaired_case_ids": full_without_ids,
                },
            }
        ),
        encoding="utf-8",
    )
    assert pass_lift_path.stat().st_size < 2 * 1024 * 1024

    result = agent_eval_result_from_directory(skill, run_dir, use_llm_judge=False)

    assert result is not None
    payload = result.metadata["agent_eval"]
    paired = payload["agents"]["opencode"]["pass_at_k"]["lift"]["paired_comparison"]
    assert payload["agents"]
    assert payload["pass_at_k"]
    assert paired["with_skill_unpaired_case_count"] == case_count
    assert paired["without_skill_unpaired_case_count"] == case_count
    assert len(paired["with_skill_unpaired_case_ids"]) == 64
    assert len(paired["without_skill_unpaired_case_ids"]) == 64
    assert paired["with_skill_unpaired_case_ids_truncated"] is True
    assert paired["without_skill_unpaired_case_ids_truncated"] is True
    # The canonical payload intentionally carries both a best-agent projection
    # and the top-level pass@k projection, so the omission signal counts both
    # serialized copies that were bounded.
    assert payload["report_truncation"]["omitted"]["unpaired_case_ids"] == 4 * (case_count - 64)
    assert len(json.dumps(payload, separators=(",", ":")).encode()) <= 2 * 1024 * 1024

    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert "Paired comparison unavailable; exact test not computed." in html
    assert "Pass@" in html


def test_standalone_tier3_persists_the_canonical_feedback_payload(tmp_path: Path) -> None:
    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "results" / "20260709_120005"
    _write_summary(run_dir, score=0.2)
    engine_result: dict[str, object] = {"execution_status": "succeeded"}

    render_agent_eval_html_report(
        skill,
        run_dir,
        engine_result=engine_result,
        use_llm_judge=False,
    )
    persisted = json.loads(json.dumps(engine_result))

    assert persisted["tier3_feedback"] == {
        "schema_version": "1.0",
        "conclusions": persisted["tier3_feedback"]["conclusions"],
        "recommendations": persisted["tier3_feedback"]["recommendations"],
        "suggestions": persisted["tier3_feedback"]["suggestions"],
        "suggestions_v2": persisted["tier3_feedback"]["suggestions_v2"],
    }
    assert persisted["tier3_feedback"]["recommendations"]
    assert persisted["tier3_feedback"]["suggestions"]
    assert persisted["tier3_feedback"]["conclusions"]
    assert "agent_eval" not in persisted


def test_rerender_uses_run_owned_dataset_and_evaluator_version(tmp_path: Path) -> None:
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": "current-1"}, {"id": "current-2"}]),
        encoding="utf-8",
    )
    run_dir = tmp_path / "results" / "20260709_120006"
    _write_summary(run_dir)
    snapshot = build_dataset_snapshot(
        [{"id": "historical-1", "expected_skill": "demo"}],
        evaluator_version="evaluated-version-1.2.3",
    )
    (run_dir / "dataset_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    tier3 = agent_eval_result_from_directory(
        skill,
        run_dir,
        use_llm_judge=False,
    )

    assert tier3 is not None
    payload = tier3.metadata["agent_eval"]
    assert payload["dataset"] == [{"id": "historical-1", "expected_skill": "demo"}]
    assert payload["dataset_summary"] == snapshot["dataset_summary"]
    assert payload["dataset_digest"] == snapshot["dataset_digest"]
    assert payload["evaluator_version"] == "evaluated-version-1.2.3"


def test_legacy_rerender_does_not_claim_current_dataset_or_reader_version(tmp_path: Path) -> None:
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": "current-1"}, {"id": "current-2"}]),
        encoding="utf-8",
    )
    run_dir = tmp_path / "results" / "20260709_120007"
    _write_summary(run_dir)

    tier3 = agent_eval_result_from_directory(
        skill,
        run_dir,
        use_llm_judge=False,
    )

    assert tier3 is not None
    payload = tier3.metadata["agent_eval"]
    assert payload["dataset"] == []
    assert payload["dataset_summary"]["source"] == "unavailable"
    assert payload["dataset_digest"] is None
    assert payload["evaluator_version"] is None


def test_standalone_failed_tier3_report_shows_error_without_score(tmp_path: Path) -> None:
    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "results" / "20260709_120001"
    _write_summary(run_dir, status="failed", errors=["No Harbor job result"])

    report = render_agent_eval_html_report(skill, run_dir, use_llm_judge=False)
    html = report.read_text(encoding="utf-8")

    assert "No Harbor job result" in html
    assert "No successfully scored agent is available" in html
    assert ">N/A<" in html
    assert "Default agent is the top performer" not in html
    assert 'onclick="tier3CopyText(this, "' not in html


def test_view_regenerates_missing_report_with_canonical_renderer(monkeypatch, tmp_path: Path) -> None:
    from skillevaluator.tier3 import commands

    skill = tmp_path / "demo"
    skill.mkdir()
    results_root = tmp_path / "results"
    run_dir = results_root / skill.name / "20260709_120002"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps({"run_id": run_dir.name}),
        encoding="utf-8",
    )
    (results_root / skill.name / "latest").symlink_to(run_dir.name)
    calls: list[tuple[Path, Path]] = []

    def render(skill_path: Path, target: Path, **_kwargs) -> Path:
        calls.append((skill_path, target))
        report = target / "report.html"
        report.write_text("<html>canonical</html>", encoding="utf-8")
        return report

    monkeypatch.setattr(commands, "render_agent_eval_html_report", render)
    monkeypatch.setattr(commands.webbrowser, "open", lambda _uri: True)

    report = commands.view_results(skill, results_dir=results_root)

    assert calls == [(skill, run_dir)]
    assert report.read_text(encoding="utf-8") == "<html>canonical</html>"


def test_legacy_tier3_html_renderer_is_removed() -> None:
    package_root = Path(__file__).parents[2] / "src" / "skillevaluator"

    assert not (package_root / "tier3" / "harbor" / "html_report.py").exists()
    for source in package_root.rglob("*.py"):
        assert "generate_html_report" not in source.read_text(encoding="utf-8")


def test_generic_report_sections_match_executed_tier_combinations(tmp_path: Path) -> None:
    tier1 = ValidationResult(validator_name="Schema Check", validator_description="Tier 1")
    tier1.add_success("schema", "valid")
    tier2 = ValidationResult(validator_name="Similarity Check", validator_description="Tier 2")
    tier2.add_success("similarity", "no duplicates")

    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "results" / "20260709_120003"
    _write_summary(run_dir)
    tier3 = agent_eval_result_from_directory(skill, run_dir, use_llm_judge=False)
    assert tier3 is not None

    cases = (
        ([tier1], {"tier1"}),
        ([tier2], {"tier2"}),
        ([tier1, tier2], {"tier1", "tier2"}),
        ([tier3], {"tier3"}),
        ([tier1, tier2, tier3], {"tier1", "tier2", "tier3"}),
    )
    for results, expected in cases:
        html = HTMLReporter(include_timestamp=False).render_all(results)
        rendered = set(re.findall(r'data-target-tab="(tier[123])"', html))
        assert rendered == expected


def test_partial_pass_at_k_case_without_attempts_skipped_still_renders(tmp_path: Path) -> None:
    skill = tmp_path / "demo"
    skill.mkdir()
    run_dir = tmp_path / "results" / "20260709_120004"
    _write_summary(run_dir)
    summary_path = run_dir / "opencode" / "with-skill" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pass_at_k"] = {
        "cases": {
            "case-001": {
                "attempts": [{"attempt": 1, "score": 1.0}],
                "attempts_used": 1,
                "attempts_missing": 0,
                "best_score": 1.0,
                "passed": True,
                "first_pass_attempt": 1,
            }
        }
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = render_agent_eval_html_report(skill, run_dir, use_llm_judge=False)

    assert "Attempt Details" in report.read_text(encoding="utf-8")


def test_pass_at_k_uncertainty_and_paired_evidence_render() -> None:
    payload = build_agent_eval_payload(
        "paired-evidence-demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 8,
                "scored_attempts": 8,
                "overall_with_skill": 0.75,
                "overall_without_skill": 0.5,
                "pass_with_skill": {
                    "rate": 0.75,
                    "rate_interval": {"lower": 0.3006, "upper": 0.9544},
                    "passed_cases": 3,
                    "total_cases": 4,
                    "attempts_used": 4,
                    "max_attempts_possible": 4,
                },
                "pass_without_skill": {
                    "rate": 0.5,
                    "rate_interval": {"lower": 0.15, "upper": 0.85},
                    "passed_cases": 2,
                    "total_cases": 4,
                    "attempts_used": 4,
                    "max_attempts_possible": 4,
                },
                "pass_lift": {
                    "delta": 0.25,
                    "paired_comparison": {
                        "pairing_status": "complete",
                        "paired_cases": 4,
                        "with_skill_only_pass": 1,
                        "without_skill_only_pass": 0,
                        "both_pass": 2,
                        "neither_pass": 1,
                        "discordant_cases": 1,
                        "paired_rate_delta": 0.25,
                        "mcnemar_exact": {
                            "p_value": 1.0,
                            "p_value_text": "1",
                            "minimum_attainable_p_value": 1.0,
                            "minimum_attainable_p_value_text": "1",
                            "resolution_limited_at_alpha_0_05": True,
                        },
                    },
                },
            }
        },
        attempt_policy={"max_attempts": 1, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    assert payload is not None

    html = _render_agent_payload(payload)

    assert "30%\u201395%" in html
    assert "15%\u201385%" in html
    assert "skill-only passes 1" in html
    assert "baseline-only passes 0" in html
    assert "Pairing: complete" in html
    assert "Paired pass-rate delta: +25.0%" in html
    assert "Exact McNemar p=1" in html
    assert "Minimum attainable p with 1 observed discordant pair:" in html
    assert "resolution-limited at \N{GREEK SMALL LETTER ALPHA}=0.05" in html


def test_pass_at_k_render_preserves_exact_subnormal_mcnemar_probability() -> None:
    pair_count = 1_086
    case_ids = [f"case-{index}" for index in range(pair_count)]
    with_skill_cases = {case_id: {"passed": index < 1_085} for index, case_id in enumerate(case_ids)}
    without_skill_cases = {case_id: {"passed": index == 1_085} for index, case_id in enumerate(case_ids)}
    paired = _paired_pass_comparison(
        {"total_cases": pair_count, "cases": with_skill_cases},
        {"total_cases": pair_count, "cases": without_skill_cases},
    )

    payload = build_agent_eval_payload(
        "subnormal-mcnemar-probability-demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": pair_count * 2,
                "scored_attempts": pair_count * 2,
                "overall_with_skill": 1.0,
                "overall_without_skill": 0.0,
                "pass_with_skill": {
                    "rate": 1_085 / pair_count,
                    "passed_cases": 1_085,
                    "total_cases": pair_count,
                },
                "pass_without_skill": {
                    "rate": 1 / pair_count,
                    "passed_cases": 1,
                    "total_cases": pair_count,
                },
                "pass_lift": {
                    "delta": 1_084 / pair_count,
                    "paired_comparison": paired,
                },
            }
        },
        attempt_policy={"max_attempts": 1, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    assert payload is not None

    html = _render_agent_payload(payload)

    assert paired["mcnemar_exact"]["p_value_text"] == "2.622311314e-324"
    assert "Exact McNemar p=2.622311314e-324." in html
    assert "4.940656458e-324" not in html


def test_pass_at_k_render_preserves_narrow_intervals_and_small_paired_delta() -> None:
    payload = build_agent_eval_payload(
        "adaptive-pass-evidence-demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 20_000,
                "scored_attempts": 20_000,
                "overall_with_skill": 0.5,
                "overall_without_skill": 0.5,
                "pass_with_skill": {
                    "rate": 0.0,
                    "rate_interval": {"lower": 0.0, "upper": 0.0038},
                    "passed_cases": 0,
                    "total_cases": 1_000,
                },
                "pass_without_skill": {
                    "rate": 1.0,
                    "rate_interval": {"lower": 0.9962, "upper": 1.0},
                    "passed_cases": 1_000,
                    "total_cases": 1_000,
                },
                "pass_lift": {
                    "delta": -1.0,
                    "paired_comparison": {
                        "pairing_status": "complete",
                        "paired_cases": 10_000,
                        "with_skill_only_pass": 1,
                        "without_skill_only_pass": 0,
                        "both_pass": 0,
                        "neither_pass": 9_999,
                        "discordant_cases": 1,
                        "paired_rate_delta": 0.0001,
                    },
                },
            }
        },
        attempt_policy={"max_attempts": 1, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    assert payload is not None

    html = _render_agent_payload(payload)

    assert "0%\u20130.38%" in html
    assert "99.62%\u2013100%" in html
    assert "Paired pass-rate delta: +0.01%" in html


def test_pass_at_k_render_preserves_sub_basis_point_endpoint_uncertainty() -> None:
    payload = build_agent_eval_payload(
        "endpoint-precision-demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 200_000,
                "scored_attempts": 200_000,
                "overall_with_skill": 0.5,
                "overall_without_skill": 0.5,
                "pass_with_skill": {
                    "rate": 0.0,
                    "rate_interval": {"lower": 0.0, "upper": 0.000038413},
                    "passed_cases": 0,
                    "total_cases": 100_000,
                },
                "pass_without_skill": {
                    "rate": 1.0,
                    "rate_interval": {"lower": 0.999961587, "upper": 1.0},
                    "passed_cases": 100_000,
                    "total_cases": 100_000,
                },
                "pass_lift": {
                    "delta": -1.0,
                    "paired_comparison": {
                        "pairing_status": "complete",
                        "paired_cases": 25_000,
                        "with_skill_only_pass": 1,
                        "without_skill_only_pass": 0,
                        "both_pass": 0,
                        "neither_pass": 24_999,
                        "discordant_cases": 1,
                        "paired_rate_delta": 1 / 25_000,
                    },
                },
            }
        },
        attempt_policy={"max_attempts": 1, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    assert payload is not None

    html = _render_agent_payload(payload)

    assert "0%\u20130.004%" in html
    assert "99.996%\u2013100%" in html
    assert "Paired pass-rate delta: +0.004%" in html


def test_pass_at_k_render_preserves_distinct_central_high_n_wilson_bounds() -> None:
    interval = _wilson_score_interval(50_000, 100_000)
    assert interval is not None
    payload = build_agent_eval_payload(
        "central-interval-precision-demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 200_000,
                "scored_attempts": 200_000,
                "overall_with_skill": 0.5,
                "overall_without_skill": 0.5,
                "pass_with_skill": {
                    "rate": 0.5,
                    "rate_interval": interval,
                    "passed_cases": 50_000,
                    "total_cases": 100_000,
                },
                "pass_without_skill": {
                    "rate": 0.5,
                    "rate_interval": interval,
                    "passed_cases": 50_000,
                    "total_cases": 100_000,
                },
                "pass_lift": {"delta": 0.0},
            }
        },
        attempt_policy={"max_attempts": 1, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    assert payload is not None

    html = _render_agent_payload(payload)

    assert "49.7%\u201350.3%" in html
    assert "50%\u201350%" not in html


def test_pass_at_k_final_column_has_a_narrow_viewport_scroll_contract() -> None:
    payload = build_agent_eval_payload(
        "responsive-pass-evidence-demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 2,
                "scored_attempts": 2,
                "overall_with_skill": 1.0,
                "overall_without_skill": 0.0,
                "pass_with_skill": {"rate": 1.0, "passed_cases": 1, "total_cases": 1},
                "pass_without_skill": {"rate": 0.0, "passed_cases": 0, "total_cases": 1},
                "pass_lift": {"delta": 1.0},
            }
        },
        attempt_policy={"max_attempts": 1, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    assert payload is not None

    html = _render_agent_payload(payload)
    region = re.search(
        r'<div class="table-scroll" tabindex="0" role="region" '
        r'aria-label="Pass at K summary table">(?P<table>.*?)</div>',
        html,
        re.DOTALL,
    )

    assert region is not None
    assert 'class="summary-table pass-at-k-table"' in region.group("table")
    assert re.search(r"<th>Lift</th>\s*</tr>", region.group("table"))
    assert ".table-scroll { max-width: 100%; overflow-x: auto;" in html
    assert ".pass-at-k-table { min-width: 720px; }" in html


def test_unavailable_pairing_does_not_render_a_zero_evidence_row() -> None:
    payload = build_agent_eval_payload(
        "unpaired-evidence-demo",
        {
            "codex": {
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 2,
                "scored_attempts": 2,
                "overall_with_skill": 1.0,
                "overall_without_skill": 0.0,
                "pass_with_skill": {"rate": 1.0, "passed_cases": 1, "total_cases": 1},
                "pass_without_skill": {"rate": 0.0, "passed_cases": 0, "total_cases": 1},
                "pass_lift": {
                    "delta": 1.0,
                    "paired_comparison": {"pairing_status": "unavailable", "paired_cases": 0},
                },
            }
        },
        attempt_policy={"max_attempts": 1, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    assert payload is not None

    html = _render_agent_payload(payload)

    assert "Paired cases: 0" not in html
    assert "Paired comparison unavailable; exact test not computed." in html
