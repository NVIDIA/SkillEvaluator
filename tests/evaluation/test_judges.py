# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Tier 3 LLM-as-judge modules (deterministic paths).

The dimension judge and insights judge fall back to deterministic / empty
output when no LLM is available, so these tests exercise that path without any
network access.
"""

from __future__ import annotations

import json

import pytest

from skillevaluator.constants import DIMENSION_MAPPING
from skillevaluator.evaluation import (
    InsightsJudge,
    build_insights,
    compute_dimensions,
    compute_dimensions_deterministic,
)
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
