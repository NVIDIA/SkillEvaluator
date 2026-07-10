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
    _evaluator_cards,
    _metric_evidence,
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
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


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
    assert "Skill Evaluator" in html


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
