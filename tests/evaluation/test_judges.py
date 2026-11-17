# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Tier 3 LLM-as-judge modules (deterministic paths).

The dimension judge and insights judge fall back to deterministic / empty
output when no LLM is available, so these tests exercise that path without any
network access.
"""

from __future__ import annotations

import json
import time

import pytest

from skillevaluator.constants import DIMENSION_MAPPING
from skillevaluator.evaluation import (
    InsightsJudge,
    build_insights,
    compute_dimensions,
    compute_dimensions_deterministic,
)
from skillevaluator.evaluation.insights_judge import _weakens_requirements
from skillevaluator.evaluation.tier3_report import _normalize_trials
from skillevaluator.tier3.dataset_utils import normalize_dataset_entries
from skillevaluator.tier3.harbor.metrics import DIMENSION_DEFINITIONS


@pytest.fixture
def evaluators() -> dict:
    return {
        "security": {"with_skill": 0.9, "baseline": 0.8},
        "skill_execution": {"with_skill": 0.7, "baseline": 0.5},
        "skill_efficiency": {"with_skill": 0.6, "baseline": 0.6},
        "accuracy": {"with_skill": 0.8, "baseline": 0.7},
        "goal_accuracy": {"with_skill": 0.75, "baseline": 0.6},
        "behavior_check": {"with_skill": 0.85, "baseline": 0.8},
        "token_efficiency": {"with_skill": 0.5, "baseline": 0.4},
    }


class TestDimensionJudgeDeterministic:
    def test_produces_all_five_dimensions(self, evaluators: dict) -> None:
        dims = compute_dimensions_deterministic(evaluators)
        assert {d["id"] for d in dims} == set(DIMENSION_MAPPING)

    def test_scores_and_lift_present(self, evaluators: dict) -> None:
        dims = {d["id"]: d for d in compute_dimensions_deterministic(evaluators)}
        sec = dims["security"]
        assert sec["score"] == pytest.approx(0.9)
        assert sec["with_skill"] == pytest.approx(0.9)
        assert sec["baseline"] == pytest.approx(0.8)
        assert sec["lift"] == pytest.approx(0.1)
        assert sec["verdict"] == "PASS"
        assert sec["reasoning_bullets"]
        assert sec["explanation"]

    def test_verdict_thresholds(self, evaluators: dict) -> None:
        dims = {d["id"]: d for d in compute_dimensions_deterministic(evaluators)}
        # Efficiency maps directly to skill_efficiency (0.6), above the canonical 0.5 pass threshold.
        assert dims["efficiency"]["verdict"] == "PASS"
        assert dims["efficiency"]["score"] == pytest.approx(0.6)

    def test_harbor_and_report_dimension_contracts_match(self) -> None:
        expected = {
            dimension: dict(zip(config["evaluators"], config["weights"], strict=True))
            for dimension, config in DIMENSION_MAPPING.items()
        }
        assert expected == DIMENSION_DEFINITIONS

    def test_baseline_absent_yields_none_lift(self) -> None:
        evaluators = {"security": {"with_skill": 0.9}}
        dims = {d["id"]: d for d in compute_dimensions_deterministic(evaluators)}
        assert dims["security"]["baseline"] is None
        assert dims["security"]["lift"] is None

    def test_compute_dimensions_no_llm_uses_deterministic(self, evaluators: dict) -> None:
        dims = compute_dimensions(evaluators, [], 0.1, use_llm=False)
        assert {d["id"] for d in dims} == set(DIMENSION_MAPPING)


class TestInsightsJudgeFallback:
    def test_build_insights_no_llm_returns_empty(self, evaluators: dict) -> None:
        deterministic = {"conclusions": [], "recommendations": []}
        canonical = {"dimensions": compute_dimensions_deterministic(evaluators)}
        out = build_insights(canonical, deterministic, use_llm=False)
        assert out == {"conclusions": [], "recommendations": []}


class TestInsightsJudgeNegativeControls:
    def test_prompt_marks_expected_skill_null_as_negative_control(self) -> None:
        prompt = InsightsJudge().create_user_prompt(
            canonical={
                "dataset": [
                    {
                        "id": "negative-001",
                        "question": "Convert 42 degrees Fahrenheit to Celsius.",
                        "ground_truth": "5.6",
                        "expected_skill": None,
                        "expected_behavior": [
                            "The agent did not read or invoke the evaluated skill",
                        ],
                    }
                ]
            },
            deterministic={},
        )
        context = json.loads(prompt.split("\n\n", 1)[1])

        case = context["dataset"][0]
        assert case["case_type"] == "negative_control"
        assert case["skill_expected_to_activate"] is False
        assert case["expected_skill"] is None
        assert case["expected_behavior"] == [
            "The agent did not read or invoke the evaluated skill",
        ]

    @pytest.mark.parametrize(
        ("case", "expected_type", "expected_activation"),
        [
            ({"id": "positive-001", "expected_skill": "calculator"}, "skill_activation", True),
            ({"id": "unlabeled-001"}, "unlabeled", None),
        ],
    )
    def test_prompt_preserves_positive_and_unlabeled_case_semantics(
        self,
        case: dict,
        expected_type: str,
        expected_activation: bool | None,
    ) -> None:
        prompt = InsightsJudge().create_user_prompt(
            canonical={"dataset": [case]},
            deterministic={},
        )
        context = json.loads(prompt.split("\n\n", 1)[1])

        assert context["dataset"][0]["case_type"] == expected_type
        assert context["dataset"][0]["skill_expected_to_activate"] is expected_activation

    def test_prompt_bounds_negative_control_assertions(self) -> None:
        prompt = InsightsJudge().create_user_prompt(
            canonical={
                "dataset": [
                    {
                        "id": "negative-001",
                        "expected_skill": None,
                        "expected_behavior": ["x" * 500] * 10,
                    }
                ]
            },
            deterministic={},
        )
        context = json.loads(prompt.split("\n\n", 1)[1])

        assertions = context["dataset"][0]["expected_behavior"]
        assert len(assertions) == 5
        assert all(len(assertion) == 300 for assertion in assertions)

    def test_system_prompt_requires_execution_evidence_for_negative_control_warning(self) -> None:
        prompt = InsightsJudge().get_system_prompt()

        assert "expected_skill is null" in prompt
        assert "successful negative-control" in prompt
        assert "explicit execution evidence" in prompt
        assert "invocation evidence is absent or ambiguous" in prompt
        assert "Do not recommend removing an unrelated negative-control case" in prompt
        assert 'claim_type "negative_control_failure"' in prompt
        assert "evidence_case_ids list" in prompt
        assert "non-empty" in prompt
        assert "name every evidence case id" in prompt
        assert "general claims may cite only skill-activation cases" in prompt.lower()

    def test_prompt_uses_runtime_should_trigger_precedence_for_agentskills_cases(self) -> None:
        dataset = normalize_dataset_entries(
            {
                "skill_name": "calculator",
                "evals": [
                    {
                        "id": "negative-001",
                        "prompt": "Answer without the calculator skill.",
                        "should_trigger": False,
                    },
                    {
                        "id": "positive-001",
                        "prompt": "Use the calculator skill.",
                    },
                ],
            }
        )

        prompt = InsightsJudge().create_user_prompt(canonical={"dataset": dataset}, deterministic={})
        context = json.loads(prompt.split("\n\n", 1)[1])

        assert [case["case_type"] for case in context["dataset"]] == [
            "negative_control",
            "skill_activation",
        ]
        assert [case["skill_expected_to_activate"] for case in context["dataset"]] == [False, True]

    @pytest.mark.parametrize(
        "case",
        [
            {"id": "negative-null", "expected_skill": None},
            {"id": "negative-empty", "expected_skill": ""},
            {"id": "negative-explicit", "expected_skill": "calculator", "should_trigger": False},
        ],
    )
    def test_prompt_uses_runtime_routing_semantics_for_legacy_cases(self, case: dict) -> None:
        prompt = InsightsJudge().create_user_prompt(canonical={"dataset": [case]}, deterministic={})
        context = json.loads(prompt.split("\n\n", 1)[1])

        assert context["dataset"][0]["case_type"] == "negative_control"
        assert context["dataset"][0]["skill_expected_to_activate"] is False

    def test_sampled_trial_keeps_matching_case_metadata_beyond_first_five(self) -> None:
        dataset = [{"id": f"positive-{index:03d}", "expected_skill": "calculator"} for index in range(1, 6)] + [
            {
                "id": "negative-006",
                "expected_skill": None,
                "expected_behavior": ["Do not invoke the calculator skill"],
            }
        ]
        trials = [
            {
                "agent": "codex",
                "entry_id": case["id"],
                "overall": index / 10,
                "scores": {"skill_execution": 1.0},
            }
            for index, case in enumerate(dataset, start=1)
        ]

        prompt = InsightsJudge().create_user_prompt(
            canonical={"dataset": dataset, "trials": trials},
            deterministic={},
        )
        context = json.loads(prompt.split("\n\n", 1)[1])

        sampled = next(trial for trial in context["trials"] if trial["entry_id"] == "negative-006")
        assert sampled["case"]["case_type"] == "negative_control"
        assert sampled["case"]["expected_behavior"] == ["Do not invoke the calculator skill"]
        assert "negative-006" in {case["id"] for case in context["dataset"]}

    def test_trial_sampling_preserves_multi_agent_and_attempt_identity_without_trial_ids(self) -> None:
        trials = [
            {"agent": "codex", "entry_id": "shared", "trial_id": None, "overall": 0.1},
            {"agent": "codex", "entry_id": "shared", "trial_id": None, "overall": 0.2},
            {"agent": "claude-code", "entry_id": "shared", "trial_id": None, "overall": 0.3},
        ]

        prompt = InsightsJudge().create_user_prompt(
            canonical={"dataset": [{"id": "shared", "expected_skill": "calculator"}], "trials": trials},
            deterministic={},
        )
        sampled = json.loads(prompt.split("\n\n", 1)[1])["trials"]

        assert [(trial["agent"], trial["attempt"]) for trial in sampled] == [
            ("codex", 1),
            ("codex", 2),
            ("claude-code", 1),
        ]

    def test_trial_sampling_prioritizes_trusted_negative_control_failure_evidence(self) -> None:
        dataset = [{"id": f"positive-{index}", "expected_skill": "calculator"} for index in range(8)]
        dataset[4] = {"id": "negative-4", "expected_skill": None}
        trials = [
            {
                "entry_id": case["id"],
                "overall": index / 10,
                "scores": {"skill_execution": 1.0},
            }
            for index, case in enumerate(dataset)
        ]
        trials[4].update(
            {
                "skill_invoked": True,
                "invocation_evidence_source": "trajectory",
            }
        )

        prompt = InsightsJudge().create_user_prompt(
            canonical={"dataset": dataset, "trials": trials},
            deterministic={},
        )
        sampled = json.loads(prompt.split("\n\n", 1)[1])["trials"]

        evidence_trial = next(trial for trial in sampled if trial["entry_id"] == "negative-4")
        assert len(sampled) == 6
        assert evidence_trial["skill_invoked"] is True
        assert evidence_trial["invocation_evidence_source"] == "trajectory"

    def test_build_insights_suppresses_original_unsupported_negative_control_findings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "title": "Unintended Skill Application",
                        "message": (
                            "The agent correctly answers the unit conversion query but the skill isn't designed "
                            "for such tasks, indicating scope misalignment."
                        ),
                        "severity": "warn",
                    }
                ],
                "recommendations": [
                    {
                        "category": "Test",
                        "title": "Exclude Irrelevant Cases",
                        "message": (
                            "Remove unit conversion test cases from the dataset as they fall outside the skill's "
                            "intended scope."
                        ),
                        "severity": "warn",
                    }
                ],
            }
        )
        canonical = {
            "dataset": [{"id": "negative-001", "expected_skill": None}],
            "trials": [
                {
                    "agent": "codex",
                    "entry_id": "negative-001",
                    "overall": 1.0,
                    "scores": {"skill_execution": 1.0, "skill_routing": 1.0},
                }
            ],
        }

        monkeypatch.setattr(InsightsJudge, "completions", lambda *_args, **_kwargs: response)

        parsed = build_insights(canonical, {"conclusions": [], "suggestions": []})

        assert parsed == {"conclusions": [], "recommendations": []}

    def test_parse_preserves_negative_control_warning_with_failed_routing_evidence(self) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "negative_control_failure",
                        "title": "Unintended Skill Application",
                        "message": "The evaluated skill was invoked for negative-001.",
                        "severity": "fail",
                        "evidence_case_ids": ["negative-001"],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [{"id": "negative-001", "expected_skill": None}],
            "trials": [
                {
                    "agent": "codex",
                    "entry_id": "negative-001",
                    "scores": {"skill_execution": 0.0},
                }
            ],
        }

        parsed = InsightsJudge().parse_response(response, canonical=canonical)

        assert parsed["conclusions"] == [
            {
                "claim_type": "negative_control_failure",
                "title": "Unintended Skill Application",
                "message": "The evaluated skill was invoked for negative-001.",
                "severity": "fail",
                "source": "llm",
                "evidence_case_ids": ["negative-001"],
            }
        ]

    @pytest.mark.parametrize("evidence_case_ids", [["negative-001"], ["fabricated-999"]])
    def test_parse_rejects_special_claim_without_grounding(
        self,
        evidence_case_ids: list[str],
    ) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "negative_control_failure",
                        "title": "Scope Misalignment",
                        "message": "The evaluated skill activated outside its intended scope.",
                        "severity": "warn",
                        "evidence_case_ids": evidence_case_ids,
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [{"id": "negative-001", "expected_skill": None}],
            "trials": [
                {
                    "entry_id": "negative-001",
                    "scores": {"skill_execution": 1.0},
                }
            ],
        }

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    @pytest.mark.parametrize("claim_type", ["negative_control_failure", "general"])
    def test_parse_rejects_unintended_skill_use_paraphrase_without_routing_evidence(
        self,
        claim_type: str,
    ) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": claim_type,
                        "title": "Unintended Skill Use",
                        "message": "The evaluated skill was unnecessarily used for negative-001.",
                        "severity": "warn",
                        "evidence_case_ids": ["negative-001"],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [{"id": "negative-001", "expected_skill": None}],
            "trials": [
                {
                    "entry_id": "negative-001",
                    "scores": {"skill_execution": 1.0, "skill_routing": 1.0},
                }
            ],
        }

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    @pytest.mark.parametrize("claim_type", ["dataset_case_removal", "general"])
    def test_parse_rejects_unrelated_case_removal_paraphrase_without_routing_evidence(
        self,
        claim_type: str,
    ) -> None:
        response = json.dumps(
            {
                "conclusions": [],
                "recommendations": [
                    {
                        "claim_type": claim_type,
                        "category": "Test",
                        "title": "Remove unrelated case",
                        "message": "Remove negative-001 from the dataset because it is unrelated.",
                        "severity": "warn",
                        "evidence_case_ids": ["negative-001"],
                    }
                ],
            }
        )
        canonical = {
            "dataset": [{"id": "negative-001", "expected_skill": None}],
            "trials": [
                {
                    "entry_id": "negative-001",
                    "scores": {"skill_execution": 1.0, "skill_routing": 1.0},
                }
            ],
        }

        assert InsightsJudge().parse_response(response, canonical=canonical)["recommendations"] == []

    def test_parse_preserves_unrelated_insights_and_nonremoval_test_improvements(self) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "general",
                        "title": "Efficiency Regression",
                        "message": "Token use increased by 25% for positive-001.",
                        "severity": "warn",
                        "evidence_case_ids": ["positive-001"],
                    }
                ],
                "recommendations": [
                    {
                        "claim_type": "general",
                        "category": "Improve",
                        "title": "Clarify Assertions",
                        "message": "Remove ambiguity from positive-001 by making its expected output explicit.",
                        "severity": "warn",
                        "evidence_case_ids": ["positive-001"],
                    }
                ],
            }
        )
        canonical = {
            "dataset": [
                {"id": "negative-001", "expected_skill": None},
                {"id": "positive-001", "expected_skill": "calculator"},
            ],
            "trials": [
                {
                    "entry_id": "negative-001",
                    "scores": {"skill_execution": 1.0},
                }
            ],
        }

        parsed = InsightsJudge().parse_response(response, canonical=canonical)

        assert [item["title"] for item in parsed["conclusions"]] == ["Efficiency Regression"]
        assert [item["title"] for item in parsed["recommendations"]] == ["Clarify Assertions"]

    def test_parse_rejects_original_general_claim_with_empty_evidence_ids(self) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "general",
                        "title": "Unintended Skill Application",
                        "message": "The evaluated skill was unnecessarily used.",
                        "severity": "warn",
                        "evidence_case_ids": [],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {"dataset": [{"id": "negative-001", "expected_skill": None}]}

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    @pytest.mark.parametrize(
        ("title", "message"),
        [
            (
                "Efficiency positive-001",
                "The skill was unnecessarily over-applied to unrelated negative-control work.",
            ),
            (
                "Capability overreach positive-001",
                "The evaluated capability was unnecessarily used for unrelated work, indicating scope drift.",
            ),
        ],
    )
    def test_parse_rejects_general_claim_with_negative_control_semantics(
        self,
        title: str,
        message: str,
    ) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "general",
                        "title": title,
                        "message": message,
                        "severity": "warn",
                        "evidence_case_ids": ["positive-001"],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [
                {"id": "positive-001", "expected_skill": "calculator"},
                {"id": "negative-001", "expected_skill": None},
            ]
        }

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    def test_parse_rejects_case_ids_hidden_beyond_the_grounding_text_bound(self) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "general",
                        "title": "Efficiency positive-001",
                        "message": ("x" * 2_100) + " negative-001",
                        "severity": "warn",
                        "evidence_case_ids": ["positive-001"],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [
                {"id": "positive-001", "expected_skill": "calculator"},
                {"id": "negative-001", "expected_skill": None},
            ]
        }

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    @pytest.mark.parametrize(
        ("evidence_case_ids", "message"),
        [
            (["positive-001"], "Token use increased."),
            (["positive-001", "positive-001"], "Token use increased for positive-001."),
            (["fabricated-999"], "Token use increased for fabricated-999."),
        ],
    )
    def test_parse_requires_exact_unique_known_case_ids_named_in_prose(
        self,
        evidence_case_ids: list[str],
        message: str,
    ) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "general",
                        "title": "Efficiency regression",
                        "message": message,
                        "severity": "warn",
                        "evidence_case_ids": evidence_case_ids,
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {"dataset": [{"id": "positive-001", "expected_skill": "calculator"}]}

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    @pytest.mark.parametrize(
        ("trial", "accepted"),
        [
            ({"skill_invoked": True}, True),
            ({"routing_passed": False}, True),
            ({"skill_invoked": False}, False),
            ({"routing_passed": True}, False),
            ({"skill_invoked": True, "routing_passed": False}, True),
            ({"skill_invoked": False, "routing_passed": True}, False),
            ({"skill_invoked": True, "routing_passed": True, "scores": {"skill_execution": 0.0}}, False),
            ({"skill_invoked": False, "routing_passed": False, "scores": {"skill_execution": 0.0}}, False),
            ({"scores": {"skill_execution": 0.0}}, True),
            ({"scores": {"skill_execution": 1.0}}, False),
        ],
    )
    def test_parse_negative_control_uses_strict_evidence_before_legacy_numeric_fallback(
        self,
        trial: dict,
        accepted: bool,
    ) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "negative_control_failure",
                        "title": "Unintended invocation in negative-001",
                        "message": "The evaluated skill was invoked for negative-001.",
                        "severity": "fail",
                        "evidence_case_ids": ["negative-001"],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [{"id": "negative-001", "expected_skill": None}],
            "trials": [
                {
                    "entry_id": "negative-001",
                    **(
                        {"invocation_evidence_source": "trajectory"}
                        if "skill_invoked" in trial or "routing_passed" in trial
                        else {}
                    ),
                    **trial,
                }
            ],
        }

        parsed = InsightsJudge().parse_response(response, canonical=canonical)["conclusions"]

        assert bool(parsed) is accepted
        if accepted:
            assert parsed[0]["claim_type"] == "negative_control_failure"
            assert parsed[0]["evidence_case_ids"] == ["negative-001"]

    def test_parse_requires_every_cited_negative_control_to_have_failure_evidence(self) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "negative_control_failure",
                        "title": "Failures in negative-001 and negative-002",
                        "message": "The skill was invoked for negative-001 and negative-002.",
                        "severity": "fail",
                        "evidence_case_ids": ["negative-001", "negative-002"],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [
                {"id": "negative-001", "expected_skill": None},
                {"id": "negative-002", "expected_skill": None},
            ],
            "trials": [
                {
                    "entry_id": "negative-001",
                    "skill_invoked": True,
                    "invocation_evidence_source": "trajectory",
                },
                {
                    "entry_id": "negative-002",
                    "skill_invoked": False,
                    "invocation_evidence_source": "trajectory",
                },
            ],
        }

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    @pytest.mark.parametrize("source", [None, "custom-verifier"])
    def test_parse_does_not_trust_direct_or_custom_spoofed_invocation_booleans(
        self,
        source: str | None,
    ) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "negative_control_failure",
                        "title": "Unintended invocation in negative-001",
                        "message": "The evaluated skill was invoked for negative-001.",
                        "severity": "fail",
                        "evidence_case_ids": ["negative-001"],
                    }
                ],
                "recommendations": [],
            }
        )
        trial = {
            "entry_id": "negative-001",
            "skill_invoked": True,
            "routing_passed": False,
            "scores": {"skill_execution": 1.0},
        }
        if source is not None:
            trial["invocation_evidence_source"] = source
        canonical = {
            "dataset": [{"id": "negative-001", "expected_skill": None}],
            "trials": [trial],
        }

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    def test_parse_does_not_use_explicit_custom_metric_as_legacy_routing_grounding(self) -> None:
        trials = _normalize_trials(
            [
                {
                    "entry_id": "negative-001",
                    "metric_set": "custom-only",
                    "overall": 0.8,
                    "skill_execution": 0.25,
                }
            ],
            ["skill_execution"],
        )
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "negative_control_failure",
                        "title": "Unintended invocation in negative-001",
                        "message": "The evaluated skill was invoked for negative-001.",
                        "severity": "fail",
                        "evidence_case_ids": ["negative-001"],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [{"id": "negative-001", "expected_skill": None}],
            "trials": trials,
        }

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    def test_parse_treats_overflowing_legacy_numeric_evidence_as_unknown(self) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "negative_control_failure",
                        "title": "Unintended invocation in negative-001",
                        "message": "The evaluated skill was invoked for negative-001.",
                        "severity": "fail",
                        "evidence_case_ids": ["negative-001"],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [{"id": "negative-001", "expected_skill": None}],
            "trials": [
                {
                    "entry_id": "negative-001",
                    "scores": {"skill_execution": 10**10_000},
                }
            ],
        }

        assert InsightsJudge().parse_response(response, canonical=canonical)["conclusions"] == []

    def test_parse_preserves_ordered_grounding_for_positive_general_claim(self) -> None:
        response = json.dumps(
            {
                "conclusions": [
                    {
                        "claim_type": "general",
                        "title": "Positive evidence positive-002",
                        "message": "Efficiency improved in positive-001 and positive-002.",
                        "severity": "pass",
                        "evidence_case_ids": ["positive-002", "positive-001"],
                    }
                ],
                "recommendations": [],
            }
        )
        canonical = {
            "dataset": [
                {"id": "positive-001", "expected_skill": "calculator"},
                {"id": "positive-002", "expected_skill": "calculator"},
            ]
        }

        parsed = InsightsJudge().parse_response(response, canonical=canonical)["conclusions"]

        assert parsed[0]["claim_type"] == "general"
        assert parsed[0]["evidence_case_ids"] == ["positive-002", "positive-001"]


def test_regressed_dimension_reasoning_names_the_weakest_signal() -> None:
    from skillevaluator.evaluation.dimension_judge import _human_reasoning_bullets

    parts = ["goal_accuracy=0.25", "behavior_check=0.81"]
    regressed = _human_reasoning_bullets(with_skill=0.53, baseline=0.66, lift=-0.13, parts=parts)
    improved = _human_reasoning_bullets(with_skill=0.8, baseline=0.4, lift=0.4, parts=parts)

    assert regressed[-1] == "Weakest signal: goal_accuracy 0.25."
    assert improved[-1] == "Strongest signal: behavior_check 0.81."


# Verb modifiers a negator may govern: "without first deleting", "without ever having to delete the case".
_NEGATION_CHAINS = (
    "",
    "first",
    "outright",
    "ever",
    "even",
    "actually",
    "simply",
    "just",
    "also",
    "then",
    "again",
    "entirely",
    "completely",
    "fully",
    "altogether",
    "necessarily",
    "directly",
    "having to",
    "needing to",
    "need to",
    "the need to",
    "having to first",
    "being forced to",
    "being asked to",
    "trying to",
    "attempting to",
    "ever having to",
    "ever even having to first",
)
# (negator, whether the verb after it is in its base form): "without first deleting", "do not first delete".
_STRICT_NEGATORS = (
    ("without", False),
    ("instead of", False),
    ("rather than", False),
    ("there is no need to", True),
    ("should not", True),
)
_ALL_NEGATORS = (*_STRICT_NEGATORS, ("do not", True), ("never", True), ("don't", True), ("avoid", False))


# "not only deleting the case but also fixing the parser" recommends the deletion, so after "not", "do not",
# "should not", "never", "don't" or "avoid" these words void the negation. "without simply removing the case" and
# "instead of just dropping it" are different: there the negator governs the verb.
_ADDITIVE_WORDS = ("only", "just", "merely", "simply", "solely", "exclusively", "purely")
_ADDITIVE_PREFIXES = (*_ADDITIVE_WORDS, "necessarily just", "necessarily only", "even just")
_NOT_NEGATORS = (
    ("not", False),
    ("do not", True),
    ("should not", True),
    ("never", True),
    ("don't", True),
    ("avoid", False),
)


def _negated_advice(negators: tuple[tuple[str, bool], ...], case_subject: str) -> list[str]:
    sentences = []
    for negator, base_form in negators:
        for chain in _NEGATION_CHAINS:
            if negator in dict(_NOT_NEGATORS) and set(chain.split()) & set(_ADDITIVE_WORDS):
                continue
            base = base_form or chain.endswith(" to")
            for verb, subject in (("delete", case_subject), ("lower", "the pass threshold")):
                form = verb if base else verb.replace("lower", "lowering").replace("delete", "deleting")
                sentences.append(" ".join(f"Fix the parser {negator} {chain} {form} {subject}.".split()))
    return sentences


def _unrelated_negation_advice(negators: tuple[tuple[str, bool], ...], case_subject: str) -> list[str]:
    """A noun between the negator (and its modifiers) and the verb: the negator governs the noun, not the removal."""
    return [
        " ".join(f"Fix the parser {negator} {chain} {noun} by {verb} {subject}.".split())
        for negator, _ in negators
        for chain in _NEGATION_CHAINS
        for noun in ("failures", "errors")
        for verb, subject in (("deleting", case_subject), ("lowering", "the pass threshold"))
    ]


def _additive_advice(case_subject: str) -> list[str]:
    """Additive wording ("not only lowering the threshold but also fixing the parser") still recommends the removal."""
    sentences = []
    for negator, base_form in _NOT_NEGATORS:
        for additive in _ADDITIVE_PREFIXES:
            for verb, subject in (("delete", case_subject), ("lower", "the pass threshold")):
                form = verb if base_form else verb.replace("lower", "lowering").replace("delete", "deleting")
                sentences.append(f"Fix the parser and {negator} {additive} {form} {subject} but also add a test.")
    return sentences


class TestInsightsJudgeRecommendationQuality:
    @staticmethod
    def _recommendation(message: str, title: str = "Next step", case_id: str = "positive-001") -> dict:
        return {
            "claim_type": "general",
            "category": "Improve",
            "title": title,
            "message": message,
            "severity": "warn",
            "evidence_case_ids": [case_id],
        }

    def _parse(self, *messages: str, deterministic: dict | None = None, case_id: str = "positive-001") -> list[str]:
        response = json.dumps(
            {
                "conclusions": [],
                "recommendations": [self._recommendation(message, case_id=case_id) for message in messages],
            }
        )
        parsed = InsightsJudge().parse_response(
            response,
            canonical={"dataset": [{"id": case_id, "expected_skill": "calculator"}], "trials": []},
            deterministic=deterministic or {},
        )
        return [item["message"] for item in parsed["recommendations"]]

    @pytest.mark.parametrize(
        "message",
        [
            "Loosen the expected output of positive-001 so the agent passes.",
            "Relax the assertions for positive-001.",
            "Lower the pass threshold for positive-001 to 0.3.",
            "Soften positive-001's ground truth to accept partial answers.",
            "The pass threshold for positive-001 should be lowered to 0.3.",
            "Expectations for positive-001 ought to be relaxed so the agent passes.",
        ],
    )
    def test_drops_advice_that_weakens_requirements(self, message: str) -> None:
        assert self._parse(message) == []

    def test_keeps_advice_that_clarifies_requirements(self) -> None:
        message = "Make the expected output of positive-001 explicit in evals.json."
        assert self._parse(message) == [message]

    @pytest.mark.parametrize(
        "message",
        [
            "Remove the expected output mismatch by deleting the positive-001 case from evals/evals.json.",
            "Remove the assertion failure by dropping the `positive-001` case from evals/evals.json.",
            "Remove the expected output mismatch by implementing the parser, then delete the positive-001 case.",
            "Fix the mismatch by removing the positive-001 entry from evals/evals.json.",
            "Remove the expected output mismatch by deleting the positive-001 test case.",
            "Remove the failure by commenting out the positive-001 case in evals/evals.json.",
            "Remove the mismatch by disabling the dataset entry for positive-001.",
            "Delete the positive-001 row from the dataset.",
            "Skip the positive-001 case in evals/evals.json.",
            "The positive-001 case should be deleted from evals/evals.json.",
            "The positive-001 entry in evals/evals.json should be removed.",
            "The positive-001 test case must be dropped.",
            "Do not remove the expectation; delete the positive-001 case from the dataset.",
            "Delete the positive-001 case from evals/evals.json.",
            "Remove the test case for positive-001 from the dataset.",
            "Drop the eval case for positive-001.",
            "The test case should be deleted so positive-001 stops failing.",
            "Remove the special case for positive-001 from evals/evals.json.",
            "In evals/evals.json, remove the special case for positive-001.",
            "In the dataset, delete the edge case for positive-001.",
            "The special case in evals/evals.json should be removed so positive-001 passes.",
        ],
    )
    def test_drops_advice_that_deletes_a_dataset_case(self, message: str) -> None:
        assert self._parse(message) == []

    @pytest.mark.parametrize(
        "message",
        [
            "Remove the special case in parser.py for positive-001 and implement the general rounding algorithm.",
            "Remove the test case failure for positive-001 by implementing the rounding rule in SKILL.md",
            "Remove the special case in the script for positive-001 and implement the general rounding algorithm.",
            "Delete the default case branch in parser.py; implement the rounding rule positive-001 needs.",
            "Remove the base case in the recursive function that positive-001 exercises.",
            "Remove the corner case for zero in parser.py, which positive-001 hits.",
            "Remove the error case in parser.py and handle positive-001 generally.",
            "Remove the case failure in positive-001 by implementing the rounding rule in SKILL.md.",
            "Remove the case mismatch in positive-001 by implementing the rounding rule in SKILL.md.",
            "Remove the test case violation in positive-001 by implementing the rounding rule in SKILL.md.",
            "The positive-001 test case failure should be removed by implementing the rounding rule in SKILL.md.",
            "The special case in parser.py that positive-001 hits should be removed.",
            "Never remove the positive-001 test case; fix the rounding step instead.",
            "Fix the failing positive-001 test case by removing the special case in parser.py.",
        ],
    )
    def test_keeps_advice_that_removes_implementation_cases_and_case_failures(self, message: str) -> None:
        assert self._parse(message) == [message]

    @pytest.mark.parametrize(
        "case_id,message",
        [
            ("positive-001", "Delete the case for positive-001 so it stops failing."),
            ("positive-001", "Remove the case named positive-001 so the run passes."),
            ("positive-001", "Delete case positive-001."),
            ("positive-001", "Delete test case positive-001."),
            ("positive-001", "Fix the mismatch by deleting the case for positive-001."),
            ("positive-001", "Delete the entry for positive-001."),
            ("positive-001", "Remove positive-001 from evals/evals.json."),
            ("negative-007", "Remove the case for `negative-007`"),
            ("negative-007", 'Drop the "negative-007" case.'),
            ("syllable-count", "Delete the syllable-count case."),
            ("syllable-count", "Remove the case for syllable-count."),
            ("smoke", "Delete the smoke case."),
            ("smoke", "Remove the case for smoke so it stops failing."),
            ("positive-001", "The case for positive-001 should be deleted."),
            ("positive-001", "The positive-001 entry should be deleted."),
            ("positive-001", "The case for positive-001 in evals/evals.json can be removed."),
            ("syllable-count", "The syllable-count case should be removed."),
            ("smoke", "The smoke case must be dropped."),
            ("positive-001", "Exclude the case for positive-001 from the run."),
            ("positive-001", "Omit the case for positive-001."),
            ("positive-001", "Get rid of the positive-001 case."),
            ("positive-001", "Take out the case for positive-001."),
            ("positive-001", "Take positive-001 out of the dataset."),
            ("positive-001", "Take the positive-001 case out of evals/evals.json."),
            ("syllable-count", "Leave out the syllable-count case."),
            ("smoke", "Leave the smoke case out of the dataset."),
            ("positive-001", "Filter out the case for positive-001."),
            ("positive-001", "Filter positive-001 out of evals/evals.json."),
            ("smoke", "Discard the smoke case."),
            ("positive-001", "Strip the case for positive-001 from evals/evals.json."),
            ("positive-001", "Ignore the case for positive-001."),
            ("smoke", "Ignore the smoke case."),
            ("positive-001", "Stop running the case for positive-001."),
            ("syllable-count", "Retire the syllable-count case."),
            ("positive-001", "Replace the positive-001 case with an easier one."),
            ("positive-001", "Replace the case for positive-001 in evals/evals.json with a simpler one."),
            ("positive-001", "Replace positive-001 with a simpler case."),
            ("smoke", "Swap out the smoke case for an easier one."),
            ("positive-001", "Instead of keeping the case for positive-001, move on to the next one."),
            ("positive-001", "The positive-001 case should be excluded."),
            ("positive-001", "The case for positive-001 should be replaced with an easier one."),
            ("syllable-count", "The syllable-count case should be discarded."),
            ("smoke", "The smoke case can be left out."),
            ("positive-001", "The case for positive-001 should be ignored."),
            ("positive-001", "The positive-001 case should be taken out of the dataset."),
            ("syllable-count", "The syllable-count case must be filtered out."),
            ("positive-001", "The case for positive-001 can be stripped from evals/evals.json."),
            ("smoke", "The smoke case should be retired."),
        ],
    )
    def test_drops_advice_that_deletes_a_dataset_case_named_by_its_id(self, case_id: str, message: str) -> None:
        assert self._parse(message, case_id=case_id) == []

    @pytest.mark.parametrize(
        "case_id,message",
        [
            ("positive-001", "Remove the special case in parser.py for positive-001 and implement the general rule."),
            ("positive-001", "Remove the special case for positive-001 in parser.py and implement the general rule."),
            ("positive-001", "Remove the case for positive-001 from analyze_text.py and implement the general rule."),
            ("syllable-count", "Remove the syllable-count case in analyze_text.py and implement it generally."),
            ("smoke", "Remove the smoke case in analyze_text.py and implement it generally."),
            ("positive-001", "Remove the test case failure for positive-001 by implementing the rule in SKILL.md"),
            ("positive-001", "Do not delete the case for positive-001; implement the rounding rule in SKILL.md."),
            ("syllable-count", "Never remove the syllable-count case; implement the counting rule in SKILL.md."),
            ("smoke", "Do not drop the smoke case; implement the missing step in SKILL.md."),
            ("positive-001", "Remove the special case for zero-length input in parser.py, which positive-001 hits."),
            (
                "positive-001",
                "Remove the special case for zero-length input and implement the rule positive-001 needs.",
            ),
        ],
    )
    def test_keeps_advice_that_names_a_case_id_but_edits_the_implementation(self, case_id: str, message: str) -> None:
        assert self._parse(message, case_id=case_id) == [message]

    @pytest.mark.parametrize(
        "case_id,message",
        [
            ("positive-001", "Do not exclude the case for positive-001; implement the rule in SKILL.md."),
            (
                "positive-001",
                "Never replace the case for positive-001 with an easier one; implement the rule in SKILL.md.",
            ),
            ("smoke", "Never replace the smoke case; implement the missing step in SKILL.md."),
            ("syllable-count", "Don't ignore the syllable-count case; implement the counting rule in SKILL.md."),
            ("positive-001", "Do not take positive-001 out of the dataset; implement the rule in SKILL.md."),
            ("positive-001", "Instead of deleting the case for positive-001, implement the rounding rule in SKILL.md."),
            ("syllable-count", "Instead of removing the syllable-count case, implement the counting rule in SKILL.md."),
            ("smoke", "Implement the missing step in SKILL.md instead of dropping the smoke case."),
            ("positive-001", "Fix the rounding step rather than excluding the case for positive-001."),
            ("positive-001", "Make positive-001 pass without deleting the case for positive-001."),
            ("positive-001", "Discard the special case in parser.py for positive-001 and implement the general rule."),
            ("syllable-count", "Exclude the syllable-count case in analyze_text.py and implement it generally."),
            ("positive-001", "Take the special case for positive-001 out of parser.py and implement the general rule."),
            ("smoke", "Retire the smoke case from analyze_text.py and implement it generally."),
            ("positive-001", "Discard the test case failure for positive-001 by implementing the rule in SKILL.md."),
            ("positive-001", "Ignore case when comparing units, as positive-001 expects."),
            ("positive-001", "Strip whitespace from the answer, as positive-001 expects."),
            ("positive-001", "Take the case for positive-001 into account in SKILL.md."),
            ("positive-001", "Replace the special case for positive-001 in parser.py with the general rounding rule."),
        ],
    )
    def test_keeps_negated_instead_of_and_implementation_advice_for_the_wider_verbs(
        self, case_id: str, message: str
    ) -> None:
        assert self._parse(message, case_id=case_id) == [message]

    @pytest.mark.parametrize(
        "case_id,message",
        [
            ("positive-001", "Run without failures by deleting the positive-001 case from evals/evals.json."),
            ("positive-001", "Make the run pass without failures by dropping the case for positive-001."),
            ("smoke", "Get a green run without extra work by excluding the smoke case."),
            ("positive-001", "Rather than a fix, delete the case for positive-001."),
            ("positive-001", "Instead of the old approach, delete the case for positive-001 from the dataset."),
        ],
    )
    def test_drops_a_removal_when_the_negation_word_governs_something_else(self, case_id: str, message: str) -> None:
        assert self._parse(message, case_id=case_id) == []

    @pytest.mark.parametrize(
        "case_id,message",
        [
            ("positive-001", "Make positive-001 pass without deleting the case for positive-001."),
            ("positive-001", "Make positive-001 pass without simply removing the case for positive-001."),
            ("positive-001", "Fix the rounding step for positive-001 without ever dropping its case."),
            ("syllable-count", "Implement the counting rule instead of just dropping the syllable-count case."),
            ("smoke", "Fix the missing step in SKILL.md rather than excluding the smoke case."),
        ],
    )
    def test_keeps_advice_whose_negation_word_governs_the_removal_verb(self, case_id: str, message: str) -> None:
        assert self._parse(message, case_id=case_id) == [message]

    @pytest.mark.parametrize(
        "case_id,message",
        [
            ("smoke", "Implement the missing field without having to delete the smoke case."),
            ("smoke", "Fix the parser without first deleting the smoke case."),
            ("smoke", "Fix the parser without outright deleting the smoke case."),
            ("smoke", "Fix the parser without ever having to delete the smoke case."),
            ("smoke", "Fix the parser without even needing to drop the smoke case."),
            ("smoke", "Fix the parser without being forced to exclude the smoke case."),
            ("smoke", "Fix the parser instead of first deleting the smoke case."),
            ("smoke", "Fix the parser rather than simply dropping the smoke case."),
            ("smoke", "Do not first delete the smoke case; fix the parser."),
            ("smoke", "There is no need to delete the smoke case once the parser is fixed."),
            ("smoke", "You should not delete the smoke case; fix the parser."),
            ("positive-001", "Fix the parser without trying to delete the case for positive-001."),
            ("smoke", "Get smoke to pass without first lowering the pass threshold."),
            ("smoke", "Fix the parser so smoke passes without having to loosen the expected output."),
            ("smoke", "Make smoke pass without ever relaxing the assertions."),
            ("smoke", "Make smoke pass instead of outright softening the ground truth."),
        ],
    )
    def test_keeps_advice_whose_negator_governs_the_verb_through_modifiers(self, case_id: str, message: str) -> None:
        assert self._parse(message, case_id=case_id) == [message]

    @pytest.mark.parametrize(
        "case_id,message",
        [
            ("smoke", "Run without failures by deleting the smoke case from evals/evals.json."),
            ("smoke", "Run without errors by removing the smoke case from evals/evals.json."),
            ("positive-001", "Pass without errors by removing the positive-001 case from evals/evals.json."),
            ("smoke", "Pass without regressions by dropping the smoke case."),
            ("smoke", "Fix the parser without delay, deleting the smoke case."),
            ("smoke", "Fix the parser without having to wait, delete the smoke case."),
            ("smoke", "Fix the parser without ever fixing the parser by deleting the smoke case."),
            ("smoke", "Fix the parser without first fixing the parser. Delete the smoke case."),
            ("smoke", "Get smoke to pass without failures by lowering the pass threshold."),
            ("smoke", "Get smoke to pass without errors by loosening the expected output."),
            ("smoke", "Get smoke to pass without first fixing the parser, then relax the assertions."),
        ],
    )
    def test_drops_a_removal_that_a_negator_with_modifiers_does_not_govern(self, case_id: str, message: str) -> None:
        assert self._parse(message, case_id=case_id) == []

    @pytest.mark.parametrize("known", [("smoke",), ()])
    def test_a_negator_and_its_verb_modifiers_negate_the_removal_and_the_weakening(self, known: tuple) -> None:
        subject = "the smoke case" if known else "the case for positive-001"
        offenders = [s for s in _negated_advice(_ALL_NEGATORS, subject) if _weakens_requirements(s, known)]
        assert offenders == []

    @pytest.mark.parametrize("known", [("smoke",), ()])
    def test_a_noun_between_the_negator_and_the_verb_breaks_the_negation(self, known: tuple) -> None:
        subject = "the smoke case from evals/evals.json" if known else "the case for positive-001 from evals/evals.json"
        missed = [
            s for s in _unrelated_negation_advice(_STRICT_NEGATORS, subject) if not _weakens_requirements(s, known)
        ]
        assert missed == []

    @pytest.mark.parametrize(
        "case_id,message",
        [
            ("smoke", "Make smoke pass by not only lowering the pass threshold but also fixing the parser."),
            ("smoke", "Not only delete the smoke case but also fix the parser."),
            ("smoke", "Make smoke pass by not just lowering the pass threshold but also fixing the parser."),
            ("smoke", "Make smoke pass by not merely relaxing the assertions but also fixing the parser."),
            ("smoke", "Make smoke pass by not simply loosening the expected output but also fixing the parser."),
            ("smoke", "Make smoke pass by not solely lowering the pass threshold but also fixing the parser."),
            ("smoke", "Make smoke pass by not exclusively softening the ground truth but also fixing the parser."),
            (
                "smoke",
                "Make smoke pass by not necessarily just lowering the pass threshold but also fixing the parser.",
            ),
            (
                "smoke",
                "Make smoke pass by not necessarily only lowering the pass threshold but also fixing the parser.",
            ),
            ("smoke", "Fix the parser by not only dropping the smoke case but also adding a unit test."),
            ("smoke", "Not just removing the smoke case but also fixing the parser."),
            ("smoke", "Fix the parser by not merely excluding the smoke case but also adding a unit test."),
            ("positive-001", "Not simply delete the case for positive-001 but also fix the parser."),
            ("smoke", "Do not only lower the pass threshold for smoke but also fix the parser."),
            ("smoke", "Do not just delete the smoke case, fix the parser."),
            ("smoke", "Don't simply delete the smoke case; fix the parser."),
        ],
    )
    def test_drops_additive_wording_that_still_recommends_the_removal(self, case_id: str, message: str) -> None:
        assert self._parse(message, case_id=case_id) == []

    @pytest.mark.parametrize(
        "case_id,message",
        [
            ("smoke", "You should not lower the pass threshold for smoke; fix the parser."),
            ("smoke", "You must not delete the smoke case; fix the parser."),
            ("smoke", "Do not lower the pass threshold for smoke."),
            ("smoke", "Do not delete the smoke case but also fix the parser."),
            ("smoke", "Fix the parser and do not delete the smoke case."),
            ("smoke", "Fix the parser; it should not drop the smoke case."),
            ("smoke", "Fix the parser without simply removing the smoke case."),
            ("smoke", "Fix the parser instead of just dropping the smoke case."),
            ("smoke", "Fix the parser rather than merely excluding the smoke case."),
        ],
    )
    def test_keeps_genuine_prohibitions_next_to_additive_wording(self, case_id: str, message: str) -> None:
        assert self._parse(message, case_id=case_id) == [message]

    @pytest.mark.parametrize("known", [("smoke",), ()])
    def test_additive_not_only_wording_does_not_negate_the_removal_or_the_weakening(self, known: tuple) -> None:
        subject = "the smoke case" if known else "the case for positive-001"
        missed = [s for s in _additive_advice(subject) if not _weakens_requirements(s, known)]
        assert missed == []

    def test_negation_is_read_from_a_bounded_window_before_the_verb(self) -> None:
        padding = "Fix the parser. " * 2000
        assert not _weakens_requirements(padding + "Without ever having to first delete the smoke case.", ("smoke",))
        assert _weakens_requirements(
            padding + "Without ever having to first fix it, delete the smoke case.", ("smoke",)
        )

    @pytest.mark.parametrize(
        "message",
        [
            "Remove the special case for UTF-8 and use the standard decoder for positive-001.",
            "Drop the special case for ISO-8859 input and use the standard decoder for positive-001.",
            "Delete the utf-16 case and decode with the standard library for positive-001.",
            "Remove the UNIX case and normalise line endings generally for positive-001.",
            "Remove the v2-api case and call the current endpoint for positive-001.",
            "Remove the hard-coded case and compute the answer for positive-001 generally.",
            "Exclude the base-64 case from the encoder and handle positive-001 with the standard library.",
        ],
    )
    def test_keeps_advice_that_names_an_identifier_that_is_not_a_known_case_id(self, message: str) -> None:
        assert self._parse(message) == [message]

    @pytest.mark.parametrize(
        "message",
        [
            "Delete the utf-16 case from evals/evals.json, then document what positive-001 expects.",
            "Remove the special case for UTF-8 from the dataset and document what positive-001 expects.",
        ],
    )
    def test_drops_an_unlisted_identifier_when_the_sentence_points_at_the_dataset(self, message: str) -> None:
        assert self._parse(message) == []

    def test_identifier_shaped_names_count_only_without_case_ids_or_with_a_dataset_location(self) -> None:
        assert _weakens_requirements("Delete the syllable-count case.")
        assert _weakens_requirements("Delete the syllable-count case.", ())
        assert not _weakens_requirements("Delete the syllable-count case.", ("positive-001",))
        assert _weakens_requirements("Delete the syllable-count case.", ("syllable-count",))
        assert _weakens_requirements("Delete the syllable-count case from evals.json.", ("positive-001",))

    def test_case_ids_that_are_not_single_words_are_matched_by_name(self) -> None:
        ids = ("edge one", "positive-001")
        assert _weakens_requirements("Delete the edge one case.", ids)
        assert _weakens_requirements("Remove the case for edge one so it stops failing.", ids)
        assert not _weakens_requirements("Remove the edge case in parser.py and handle edge one generally.", ids)

    def test_a_list_of_case_ids_without_a_case_noun_is_checked_in_bounded_time(self) -> None:
        ids = tuple(f"case-{i:02d}" for i in range(20))
        text = "Remove the " + ", ".join(ids) + " overrides in parser.py and implement the general rule."
        for known in (ids, ()):
            started = time.perf_counter()
            assert _weakens_requirements(text, known) is False
            assert time.perf_counter() - started < 1.0

    def test_long_adversarial_actions_are_checked_in_bounded_time(self) -> None:
        ids = tuple(f"case-{i:03d}" for i in range(250))
        listed = ", ".join(ids[:150])
        hyphenated = " ".join(["a-b-c-d-e-f-g"] * 700)
        adversarial = {
            "list without a noun": (f"Remove the {listed} overrides in parser.py and implement the rule.", False),
            "list before the noun": (f"Delete the {listed} cases.", True),
            "repeated hyphen words": (f"Remove {hyphenated} case", False),
            "repeated lists": (("Remove the foo-bar, " * 300) + "baz-qux case", False),
            "repeated removals": ("Delete the case for case-001 and exclude the case " * 100, True),
            "removal verbs only": ("remove delete drop skip exclude discard omit " * 150, False),
            "nouns only": ("case cases entry row example test case " * 150, False),
            "locations only": ("in evals.json from the dataset of benchmark out of the test set " * 100, False),
            "one long identifier": ("remove the " + "positive-" * 600 + "x case", False),
            "requirement nouns": ("expectations are " * 400, False),
            "negated removals": ("Do not delete the case for case-001 and " * 300, False),
            "negators and modifiers without a verb": ("without ever having to first simply " * 300, False),
        }
        for label, (text, expected) in adversarial.items():
            for known in (ids, ()):
                started = time.perf_counter()
                result = _weakens_requirements(text, known)
                elapsed = time.perf_counter() - started
                assert elapsed < 1.0, f"{label} took {elapsed:.2f}s with {len(known)} case ids"
                if known:
                    assert result is expected, label

    def test_parse_checks_a_list_of_known_case_ids_in_bounded_time(self) -> None:
        ids = [f"case-{i:03d}" for i in range(250)]
        dataset = [{"id": case_id, "expected_skill": "calculator"} for case_id in ids]
        cited = ids[:10]
        benign = f"Remove the {', '.join(cited)} overrides in parser.py and implement the general rule."
        removal = f"Delete the {', '.join(cited)} cases from evals/evals.json."

        def recommendation(message: str) -> dict:
            return {**self._recommendation(message), "evidence_case_ids": cited}

        response = json.dumps({"conclusions": [], "recommendations": [recommendation(benign), recommendation(removal)]})
        started = time.perf_counter()
        parsed = InsightsJudge().parse_response(response, canonical={"dataset": dataset, "trials": []})
        assert time.perf_counter() - started < 1.0
        assert [item["message"] for item in parsed["recommendations"]] == [benign]

    @pytest.mark.parametrize("case_id", ["smoke", "syllable-count"])
    def test_drops_a_named_dataset_case_removal_in_the_title_even_when_the_message_is_benign(
        self, case_id: str
    ) -> None:
        message = f"Implement the missing step that {case_id} exercises in SKILL.md."
        recommendation = self._recommendation(message, title=f"Delete the {case_id} case", case_id=case_id)
        response = json.dumps({"conclusions": [], "recommendations": [recommendation]})
        parsed = InsightsJudge().parse_response(
            response,
            canonical={"dataset": [{"id": case_id, "expected_skill": "calculator"}], "trials": []},
        )
        assert parsed["recommendations"] == []

    @pytest.mark.parametrize(
        "message",
        [
            "Remove the expected output mismatch in positive-001 by implementing the parser in SKILL.md.",
            "Remove the assertion failure in positive-001 by implementing the parser in SKILL.md.",
            "Remove the `expected output` mismatch in positive-001 by implementing the parser in SKILL.md.",
            "Remove the expected output mismatch by fixing the parser; keep the positive-001 case in evals/evals.json.",
            "Do not delete the case; document the rounding rule that positive-001 depends on in SKILL.md.",
            "Never remove the positive-001 entry from evals/evals.json; fix the rounding step instead.",
            "In this case, document the rounding rule that positive-001 depends on in SKILL.md.",
            "The edge case in positive-001 should be handled in SKILL.md.",
            "Delete the cache entry, then document the rounding rule that positive-001 depends on in SKILL.md.",
            "Remove temp files per case, then document the rounding rule that positive-001 depends on in SKILL.md.",
        ],
    )
    def test_keeps_mismatch_fixes_negated_removals_and_ordinary_case_prose(self, message: str) -> None:
        assert self._parse(message) == [message]

    def test_drops_a_dataset_case_removal_in_the_title_even_when_the_message_is_benign(self) -> None:
        recommendation = self._recommendation(
            "Implement the rounding rule in SKILL.md.",
            title="Delete the positive-001 case from evals/evals.json",
        )
        response = json.dumps({"conclusions": [], "recommendations": [recommendation]})
        parsed = InsightsJudge().parse_response(
            response,
            canonical={"dataset": [{"id": "positive-001", "expected_skill": "calculator"}], "trials": []},
        )
        assert parsed["recommendations"] == []

    def test_keeps_negated_and_non_requirement_uses_of_weakening_verbs(self) -> None:
        messages = (
            "Do not lower the pass threshold for positive-001; fix the rounding step instead.",
            "Lower ambiguity in positive-001's acceptance criteria by documenting the rounding rule.",
            "Tell the agent to lowercase unit names, as positive-001's expected output does.",
        )
        assert self._parse(*messages) == list(messages)

    def test_does_not_match_a_requirement_across_title_and_message(self) -> None:
        message = "Document the expected JSON shape for positive-001 in SKILL.md."
        recommendation = self._recommendation(message, title="Lower token usage")
        response = json.dumps({"conclusions": [], "recommendations": [recommendation]})
        parsed = InsightsJudge().parse_response(
            response,
            canonical={"dataset": [{"id": "positive-001", "expected_skill": "calculator"}], "trials": []},
        )
        assert [item["message"] for item in parsed["recommendations"]] == [message]

    def test_drops_repeats_of_deterministic_and_earlier_recommendations(self) -> None:
        first = "Document the rounding rule that positive-001 depends on in SKILL.md."
        repeated = "Add a units example for positive-001 to SKILL.md."
        assert self._parse(
            first,
            first.upper(),
            repeated,
            deterministic={"suggestions": ["add a units example for positive-001 to  SKILL.md."]},
        ) == [first]

    def test_prompt_requires_distinct_actions_and_experiments_for_untested_causes(self) -> None:
        prompt = InsightsJudge().get_system_prompt()
        assert "do not restate a\n  deterministic recommendation" in prompt
        assert "Never recommend loosening, relaxing, or lowering an expectation" in prompt
        assert "Recommend a different agent or model only when the supplied trials compare" in prompt
