# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed result semantics for the Tier 3 LLM judges."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from skillevaluator.tier3.eval_core import llm_judge
from skillevaluator.tier3.harbor import report

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)
_CRITERIA = {
    "ACTION_CORRECT": False,
    "FACTUALLY_ACCURATE": True,
    "TASK_ADDRESSED": True,
    "ACTIONABLE": False,
    "FORMAT_FOLLOWED": True,
}
VALID_ACCURACY_RESPONSE = json.dumps({"criteria": _CRITERIA, "reason": "valid retry"})
# Five valid criteria, so a reply built on it can fail only on what follows.
_VALID_CRITERIA_PREFIX = '{"criteria": ' + json.dumps(_CRITERIA) + ", "
VALID_GOAL_RESPONSE = json.dumps(
    {
        "user_goal": "complete the task",
        "end_state": "task completed",
        "achieved": True,
        "reason": "valid retry",
    }
)


def _load_template_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("harbor_template_eval_judge_results", _TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_template = _load_template_module()


@pytest.mark.parametrize("module", [llm_judge, eval_template])
def test_behavior_inventory_preserves_original_requirements_and_missing_verdicts(module, monkeypatch):
    truncated = '{"results":[{"step":2,"passed":false,"reason":"file absent"},{"step":3'
    monkeypatch.setattr(module, "call_public_llm", _pair_script([(truncated, None)], []))
    result = module.judge_behavior_check("conversation", ["read the input", "write the file", "check the file"])

    assert result["score"] == 0.0
    assert result["results"] == [{"step": 2, "passed": False, "reason": "file absent"}]
    assert [item["requirement"] for item in result["expectations"]] == [
        "read the input",
        "write the file",
        "check the file",
    ]
    assert [item["status"] for item in result["expectations"]] == ["not_checked", "unmet", "not_checked"]
    assert all(item["reason"] for item in result["expectations"])


@pytest.mark.parametrize("module", [llm_judge, eval_template])
def test_behavior_inventory_keeps_all_expectations_after_judge_error(module, monkeypatch):
    monkeypatch.setattr(module, "call_public_llm", _pair_script([("", "judge unavailable")], []))
    result = module.judge_behavior_check("conversation", ["first", "second"])

    assert result["score"] is None
    assert result["status"] == "error"
    assert [item["requirement"] for item in result["expectations"]] == ["first", "second"]
    assert all(item["status"] == "not_checked" for item in result["expectations"])
    assert all(item["reason"] for item in result["expectations"])


@pytest.mark.parametrize("module", [llm_judge, eval_template])
def test_behavior_judge_can_abstain_without_calling_missing_evidence_a_violation(module, monkeypatch):
    response = json.dumps(
        {
            "results": [
                {"step": 1, "passed": True, "reason": "input read"},
                {"step": 2, "passed": None, "reason": "artifact contents not captured"},
            ]
        }
    )
    monkeypatch.setattr(module, "call_public_llm", _pair_script([(response, None)], []))
    result = module.judge_behavior_check("partial trace", ["read input", "verify contents"])
    assert result["score"] == 0.5
    assert [item["status"] for item in result["expectations"]] == ["met", "not_checked"]
    assert result["expectations"][1]["reason"] == "artifact contents not captured"


@pytest.mark.parametrize("module", [llm_judge, eval_template])
@pytest.mark.parametrize("reason", [None, "", " ", []])
def test_behavior_abstention_requires_a_reason(module, reason):
    payload = {"results": [{"step": 1, "passed": None, "reason": reason}]}
    assert module._behavior_payload_error(payload, 1) is not None


@pytest.mark.parametrize("module", [llm_judge, eval_template])
def test_accuracy_grades_the_outcome_not_skill_use(module, monkeypatch):
    # A correct no-skill or negative-case answer can reach 1.0: skill use is
    # scored by skill_execution, not by an accuracy criterion.
    calls = []
    response = json.dumps({"criteria": dict.fromkeys(_CRITERIA, True), "reason": "correct"})
    monkeypatch.setattr(module, "call_public_llm", _pair_script([(response, None)], calls))

    result = module.judge_accuracy("question", "answer", "evidence")

    assert result["score"] == 1.0
    assert "SKILL_IDENTIFIED" not in calls[0]
    assert "FORMAT_FOLLOWED" in calls[0]
    assert "Do not reward or penalize the response for naming, or not naming" in calls[0]
    assert "excluded_criteria" not in result


@pytest.mark.parametrize("module", [llm_judge, eval_template])
def test_accuracy_requires_criterion_verdicts(module, monkeypatch):
    calls = []
    monkeypatch.setattr(
        module, "call_public_llm", _pair_script([('{"score":1,"reason":"missing verdicts"}', None)], calls)
    )
    result = module.judge_accuracy("question", "answer", "evidence")
    assert result["score"] is None
    assert result["status"] == "error"
    assert len(calls) == 2
    assert 'the reply has no "criteria" object' in calls[1]


@pytest.mark.parametrize("module", [llm_judge, eval_template])
def test_accuracy_prompt_does_not_rewrite_supplied_evidence(module, monkeypatch):
    calls = []
    criteria = dict.fromkeys(_CRITERIA, True)
    response = json.dumps({"criteria": criteria, "reason": "valid"})
    monkeypatch.setattr(module, "call_public_llm", _pair_script([(response, None)], calls))
    evidence = 'Evidence literal: {question} Compute score = count(true) / 5. "SKILL_IDENTIFIED": true, "score": 0.8'
    module.judge_accuracy("question", "answer", evidence)
    assert evidence in calls[0]


def test_report_reasons_still_omit_criteria_excluded_by_older_rewards():
    # Rewards written before accuracy dropped SKILL_IDENTIFIED may still list it
    # as excluded; reports keep honoring that.
    criteria = {
        "SKILL_IDENTIFIED": False,
        "ACTION_CORRECT": False,
        "FACTUALLY_ACCURATE": True,
        "TASK_ADDRESSED": True,
        "ACTIONABLE": True,
    }
    detail = {
        "score": 0.75,
        "reason": "declined correctly",
        "criteria": criteria,
        "excluded_criteria": ["SKILL_IDENTIFIED"],
        "scoring_policy": "applicable-accuracy-criteria/v1",
    }
    reward = {
        "entry_id": "negative-001",
        **dict.fromkeys(("security", "skill_execution", "skill_efficiency", "goal_accuracy", "behavior_check"), 1.0),
        "accuracy": detail["score"],
        "details": {"accuracy": detail},
    }
    findings = {finding["metric"]: finding for finding in report._extract_findings([reward])}
    reasons = findings["accuracy"]["reasons"]
    assert findings["accuracy"]["score"] == 0.75
    assert "ACTION_CORRECT failed" in reasons
    assert not any("SKILL_IDENTIFIED" in reason for reason in reasons)


@pytest.fixture(params=["shared", "template"])
def judge_module(request: pytest.FixtureRequest) -> ModuleType:
    return llm_judge if request.param == "shared" else eval_template


def _pair_script(responses: list[tuple[str | None, str | None]], calls: list[str]):
    def fake_call(prompt: str, **_kwargs):
        calls.append(prompt)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    return fake_call


def _recorded_pair_script(
    responses: list[tuple[str | None, str | None]],
    calls: list[dict[str, Any]],
):
    def fake_call(prompt: str, **kwargs: Any):
        calls.append({"prompt": prompt, "kwargs": kwargs})
        return responses[min(len(calls) - 1, len(responses) - 1)]

    return fake_call


def _patch_goal_script(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[tuple[str | None, str | None, dict[str, str]]],
    calls: list[dict[str, Any]],
) -> None:
    def fake_call(prompt: str, **kwargs: Any):
        calls.append({"prompt": prompt, "kwargs": kwargs})
        content, error, provenance = responses[min(len(calls) - 1, len(responses) - 1)]
        if module is llm_judge:
            return content, error
        return content, error, provenance

    if module is llm_judge:
        monkeypatch.setattr(module, "call_public_llm", fake_call)
    else:
        monkeypatch.setattr(module, "_ragas_goal_accuracy_enabled", lambda: False)
        monkeypatch.setattr(module, "_call_public_llm_with_provenance", fake_call)


def _assert_error_result(result: dict) -> None:
    assert result["score"] is None
    assert result["status"] == "error"
    assert isinstance(result["reason"], str)
    assert 0 < len(result["reason"]) <= 512


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("not-json", id="malformed"),
        pytest.param(
            json.dumps({"criteria": {**_CRITERIA, "ACTIONABLE": "true"}, "score": 0.6}),
            id="schema-invalid",
        ),
    ],
)
def test_accuracy_retries_invalid_response_once_and_recovers(
    judge_module,
    monkeypatch,
    invalid: str,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _recorded_pair_script([(invalid, None), (VALID_ACCURACY_RESPONSE, None)], calls),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert result["score"] == 0.6
    assert result["criteria"] == _CRITERIA
    assert [call["kwargs"]["max_tokens"] for call in calls] == [judge_module.STRUCTURED_JUDGE_MAX_TOKENS] * 2
    assert "previous reply could not be parsed or validated" in calls[1]["prompt"]


def test_accuracy_retries_non_string_reason_once_and_recovers(judge_module, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    invalid = json.dumps({"score": 1.0, "reason": {"nested": "text"}})
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _recorded_pair_script([(invalid, None), (VALID_ACCURACY_RESPONSE, None)], calls),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 2
    assert result["score"] == 0.6
    assert result["reason"] == "valid retry"
    assert "previous reply could not be parsed or validated" in calls[1]["prompt"]


def test_accuracy_invalid_twice_errors_after_exactly_one_retry(judge_module, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _recorded_pair_script([("not-json", None), ("still-not-json", None)], calls),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 2
    _assert_error_result(result)
    assert "after retry" in result["reason"]
    assert "not-json" not in result["reason"]


def test_accuracy_initial_provider_error_does_not_retry(judge_module, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _recorded_pair_script([(None, "provider unavailable")], calls),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 1
    _assert_error_result(result)
    assert result["reason"].startswith("LLM judge error:")


def test_accuracy_retry_provider_error_is_bounded_and_redacted(judge_module, monkeypatch) -> None:
    credential = "dummy-accuracy-retry-secret-DO-NOT-RETAIN"
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _recorded_pair_script(
            [
                ("not-json", None),
                (None, f"retry failed with {credential} " + ("x" * 1000)),
            ],
            calls,
        ),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["reason"].startswith("LLM judge retry error:")
    assert credential not in result["reason"]
    assert "[REDACTED]" in result["reason"]


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("not-json", id="malformed"),
        pytest.param('{"achieved": "true", "score": 1.0}', id="schema-invalid"),
    ],
)
def test_goal_retries_invalid_response_once_and_recovers(
    judge_module,
    monkeypatch,
    invalid: str,
) -> None:
    calls: list[dict[str, Any]] = []
    _patch_goal_script(
        judge_module,
        monkeypatch,
        [
            (invalid, None, {"provider": "first-provider", "model": "first-model"}),
            (
                VALID_GOAL_RESPONSE,
                None,
                {"provider": "retry-provider", "model": "retry-model"},
            ),
        ],
        calls,
    )

    result = judge_module.judge_goal_accuracy("question", "ground truth", "agent response")

    assert result["score"] == 1.0
    assert result["user_goal"] == "complete the task"
    assert [call["kwargs"]["max_tokens"] for call in calls] == [judge_module.STRUCTURED_JUDGE_MAX_TOKENS] * 2
    assert "previous reply could not be parsed or validated" in calls[1]["prompt"]
    if judge_module is eval_template:
        assert result["provider"] == "retry-provider"
        assert result["model"] == "retry-model"


@pytest.mark.parametrize("field", ["reason", "user_goal", "end_state"])
def test_goal_retries_non_string_text_fields_once_and_recovers(
    judge_module,
    monkeypatch,
    field: str,
) -> None:
    calls: list[dict[str, Any]] = []
    invalid_payload = {
        "user_goal": "complete the task",
        "end_state": "task completed",
        "achieved": True,
        "score": 1.0,
        "reason": "valid judgment",
    }
    invalid_payload[field] = {"nested": "text"}
    _patch_goal_script(
        judge_module,
        monkeypatch,
        [
            (
                json.dumps(invalid_payload),
                None,
                {"provider": "first-provider", "model": "first-model"},
            ),
            (
                VALID_GOAL_RESPONSE,
                None,
                {"provider": "retry-provider", "model": "retry-model"},
            ),
        ],
        calls,
    )

    result = judge_module.judge_goal_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 2
    assert result["score"] == 1.0
    assert result["reason"] == "valid retry"
    assert result["user_goal"] == "complete the task"
    assert result["end_state"] == "task completed"
    assert "previous reply could not be parsed or validated" in calls[1]["prompt"]


def test_goal_invalid_twice_errors_after_exactly_one_retry(judge_module, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    _patch_goal_script(
        judge_module,
        monkeypatch,
        [
            ("not-json", None, {"provider": "first-provider", "model": "first-model"}),
            ("still-not-json", None, {"provider": "retry-provider", "model": "retry-model"}),
        ],
        calls,
    )

    result = judge_module.judge_goal_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 2
    _assert_error_result(result)
    assert "after retry" in result["reason"]
    assert "not-json" not in result["reason"]
    if judge_module is eval_template:
        assert result["provider"] == "retry-provider"
        assert result["model"] == "retry-model"


def test_goal_initial_provider_error_does_not_retry(judge_module, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    _patch_goal_script(
        judge_module,
        monkeypatch,
        [(None, "provider unavailable", {"provider": "first-provider", "model": "first-model"})],
        calls,
    )

    result = judge_module.judge_goal_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 1
    _assert_error_result(result)
    assert result["reason"].startswith("LLM judge error:")


def test_goal_retry_provider_error_is_bounded_redacted_and_uses_retry_provenance(
    judge_module,
    monkeypatch,
) -> None:
    credential = "dummy-goal-retry-secret-DO-NOT-RETAIN"
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    calls: list[dict[str, Any]] = []
    _patch_goal_script(
        judge_module,
        monkeypatch,
        [
            ("not-json", None, {"provider": "first-provider", "model": "first-model"}),
            (
                None,
                f"retry failed with {credential} " + ("x" * 1000),
                {"provider": "retry-provider", "model": "retry-model"},
            ),
        ],
        calls,
    )

    result = judge_module.judge_goal_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["reason"].startswith("LLM judge retry error:")
    assert credential not in result["reason"]
    assert "[REDACTED]" in result["reason"]
    if judge_module is eval_template:
        assert result["provider"] == "retry-provider"
        assert result["model"] == "retry-model"


@pytest.mark.parametrize(
    ("judge_name", "response"),
    [
        pytest.param("judge_accuracy", VALID_ACCURACY_RESPONSE, id="accuracy"),
        pytest.param("judge_goal_accuracy", VALID_GOAL_RESPONSE, id="goal-accuracy"),
    ],
)
def test_shared_structured_judge_keeps_explicit_max_tokens_override(
    monkeypatch,
    judge_name: str,
    response: str,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        llm_judge,
        "call_public_llm",
        _recorded_pair_script([(response, None)], calls),
    )

    judge = getattr(llm_judge, judge_name)
    result = judge("question", "ground truth", "agent response", max_tokens=2048)

    assert result["score"] is not None
    assert len(calls) == 1
    assert calls[0]["kwargs"]["max_tokens"] == 2048
    assert isinstance(calls[0]["kwargs"]["response_schema"], dict)
    assert isinstance(calls[0]["kwargs"]["schema_name"], str)


def test_judge_error_reserved_fields_cannot_be_overridden(judge_module) -> None:
    metadata = {
        "score": 1.0,
        "status": "ok",
        "reason": "spoofed success",
        "provider": "test-provider",
    }

    result = judge_module._judge_error("canonical failure", **metadata)

    assert result == {
        "score": None,
        "status": "error",
        "reason": "canonical failure",
        "provider": "test-provider",
    }


def _judge_goal_with_response(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    content: str | None,
    error: str | None = None,
    provenance: dict | None = None,
) -> dict:
    if module is llm_judge:
        monkeypatch.setattr(module, "call_public_llm", lambda *_args, **_kwargs: (content, error))
    else:
        monkeypatch.setattr(module, "_ragas_goal_accuracy_enabled", lambda: False)
        monkeypatch.setattr(
            module,
            "_call_public_llm_with_provenance",
            lambda *_args, **_kwargs: (content, error, provenance or {}),
        )
    return module.judge_goal_accuracy("question", "ground truth", "agent response")


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("", id="empty"),
        pytest.param("YES YES YES YES YES", id="prose-yes-count"),
        pytest.param("[]", id="list"),
        pytest.param("{}", id="empty-object"),
        pytest.param('prefix [{"score": 1.0}] suffix', id="prose-wrapped-list"),
        pytest.param(_VALID_CRITERIA_PREFIX + '"score": true, "score": 0.5}', id="duplicate-score"),
        pytest.param(_VALID_CRITERIA_PREFIX + '"score": NaN}', id="nan-score"),
        pytest.param(_VALID_CRITERIA_PREFIX + '"score": Infinity}', id="infinite-score"),
        pytest.param('{"criteria": {"SKILL_IDENTIFIED": true}}', id="incomplete-criteria"),
        pytest.param(
            json.dumps({"criteria": {**_CRITERIA, "ACTIONABLE": "true"}}),
            id="non-boolean-criterion",
        ),
        pytest.param(
            json.dumps({"criteria": {**_CRITERIA, "EXTRA": True}}),
            id="extra-criterion",
        ),
        pytest.param('{"score": 0.5, "criteria": []}', id="numeric-score-with-wrong-criteria"),
    ],
)
def test_accuracy_rejects_unscorable_response_shapes(judge_module, monkeypatch, content: str) -> None:
    monkeypatch.setattr(judge_module, "call_public_llm", lambda *_args, **_kwargs: (content, None))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    _assert_error_result(result)


@pytest.mark.parametrize(
    "score",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(0.5, id="fraction"),
        pytest.param(99, id="out-of-range"),
        pytest.param(10**400, id="huge-integer"),
    ],
)
def test_accuracy_rejects_score_only_replies(judge_module, monkeypatch, score: float) -> None:
    calls: list[str] = []
    content = json.dumps({"score": score, "reason": "model judgment"})
    monkeypatch.setattr(judge_module, "call_public_llm", _pair_script([(content, None)], calls))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 2
    _assert_error_result(result)


@pytest.mark.parametrize(
    "reported_score",
    [
        pytest.param(0.0, id="contradicting-zero"),
        pytest.param(-2, id="below-range"),
        pytest.param(4.2, id="above-range"),
        pytest.param(10**400, id="huge-integer"),
        pytest.param("0.8", id="string"),
        pytest.param(True, id="boolean"),
    ],
)
def test_accuracy_ignores_any_reported_score(judge_module, monkeypatch, reported_score) -> None:
    calls: list[str] = []
    content = json.dumps({"criteria": _CRITERIA, "score": reported_score, "reason": "model judgment"})
    monkeypatch.setattr(judge_module, "call_public_llm", _pair_script([(content, None)], calls))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 1
    assert result["score"] == 0.6
    assert result["criteria"] == _CRITERIA


def test_accuracy_normalizes_and_bounds_reason(judge_module, monkeypatch) -> None:
    calls: list[str] = []
    content = json.dumps({"criteria": _CRITERIA, "reason": "  " + ("x" * 600) + "  "})
    monkeypatch.setattr(judge_module, "call_public_llm", _pair_script([(content, None)], calls))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 1
    assert result["reason"] == ("x" * 509) + "..."
    assert len(result["reason"]) == 512


def test_accuracy_redacts_configured_secret_before_bounding_reason(judge_module, monkeypatch) -> None:
    credential = "SECRET-ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789"
    prefix = "x" * 490
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    content = json.dumps({"criteria": _CRITERIA, "reason": prefix + credential})
    monkeypatch.setattr(judge_module, "call_public_llm", lambda *_args, **_kwargs: (content, None))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert result["reason"] == prefix + "[REDACTED]"
    assert credential not in result["reason"]
    assert "SECRET-" not in result["reason"]


@pytest.mark.parametrize("invalid_score", [pytest.param(None, id="absent"), pytest.param(True, id="boolean")])
def test_accuracy_derives_score_only_from_complete_boolean_criteria(
    judge_module,
    monkeypatch,
    invalid_score: bool | None,
) -> None:
    payload = {"criteria": _CRITERIA, "reason": "criteria judgment"}
    if invalid_score is not None:
        payload["score"] = invalid_score
    content = json.dumps(payload)
    monkeypatch.setattr(judge_module, "call_public_llm", lambda *_args, **_kwargs: (content, None))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert result["score"] == 0.6
    assert result["criteria"] == _CRITERIA


def test_accuracy_provider_error_is_scoreless_bounded_and_redacted(judge_module, monkeypatch) -> None:
    credential = "dummy-judge-secret-DO-NOT-RETAIN"
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    detail = f"HTTP 401: Unauthorized - echoed {credential} " + ("x" * 1000)
    monkeypatch.setattr(judge_module, "call_public_llm", lambda *_args, **_kwargs: (None, detail))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    _assert_error_result(result)
    assert credential not in result["reason"]
    assert "[REDACTED]" in result["reason"]


def test_accuracy_accepts_unique_object_after_non_json_bracket_label(judge_module, monkeypatch) -> None:
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        lambda *_args, **_kwargs: ("Judge [draft]: " + json.dumps({"criteria": _CRITERIA}), None),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert result["score"] == 0.6
    assert result.get("status") != "error"


def test_accuracy_accepts_unique_object_before_unmatched_trailing_prose_without_retry(
    judge_module,
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(json.dumps({"criteria": _CRITERIA}) + "\nNote [draft", None)], calls),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 1
    assert result["score"] == 0.6
    assert result.get("status") != "error"


def test_accuracy_rejects_nonstandard_constant_in_reason(judge_module, monkeypatch) -> None:
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        lambda *_args, **_kwargs: ('{"score": 0.5, "reason": NaN}', None),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    _assert_error_result(result)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param('{"score": 0} {"score": 1}', id="adjacent-objects"),
        pytest.param(
            '{"score": 0}\n```json\n{"score": 1}\n```',
            id="object-and-conflicting-fence",
        ),
    ],
)
def test_accuracy_rejects_multiple_complete_json_documents(judge_module, monkeypatch, content: str) -> None:
    monkeypatch.setattr(judge_module, "call_public_llm", lambda *_args, **_kwargs: (content, None))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    _assert_error_result(result)


def test_accuracy_deep_nesting_is_scoreless_instead_of_raising(judge_module, monkeypatch) -> None:
    content = "[" * 10_000 + '{"score": 1.0}' + "]" * 10_000
    monkeypatch.setattr(judge_module, "call_public_llm", lambda *_args, **_kwargs: (content, None))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    _assert_error_result(result)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("", id="empty"),
        pytest.param("achieved: yes", id="prose"),
        pytest.param("[]", id="list"),
        pytest.param("{}", id="empty-object"),
        pytest.param('prefix [{"achieved": true}] suffix', id="prose-wrapped-list"),
        pytest.param('{"achieved": 1}', id="integer-achieved"),
        pytest.param('{"achieved": 1, "achieved": true}', id="duplicate-achieved"),
        pytest.param('{"achieved": "true"}', id="string-achieved"),
        pytest.param('{"achieved": false, "score": NaN}', id="nan-score"),
        pytest.param('{"achieved": true, "score": Infinity}', id="infinite-score"),
    ],
)
def test_goal_rejects_unscorable_response_shapes(judge_module, monkeypatch, content: str) -> None:
    result = _judge_goal_with_response(judge_module, monkeypatch, content=content)

    _assert_error_result(result)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param({"achieved": False, "score": 0.0}, 0.0, id="genuine-zero"),
        pytest.param({"achieved": True}, 1.0, id="derive-true"),
        pytest.param({"achieved": False}, 0.0, id="derive-false"),
        pytest.param({"achieved": False, "score": 1.0}, 0.0, id="contradicting-score"),
        pytest.param({"achieved": False, "score": 0.25}, 0.0, id="fractional-score"),
        pytest.param({"achieved": True, "score": -(10**400)}, 1.0, id="huge-integer-low"),
        pytest.param({"achieved": False, "score": 10**400}, 0.0, id="huge-integer-high"),
        pytest.param({"achieved": True, "score": True}, 1.0, id="boolean-score"),
        pytest.param({"achieved": True, "score": "0.8"}, 1.0, id="string-score"),
    ],
)
def test_goal_score_is_binary_from_achieved_and_ignores_reported_score(
    judge_module,
    monkeypatch,
    payload: dict,
    expected: float,
) -> None:
    payload = {"reason": "model judgment", "user_goal": "goal", "end_state": "state", **payload}

    result = _judge_goal_with_response(judge_module, monkeypatch, content=json.dumps(payload))

    assert result["score"] == expected
    assert result.get("status") != "error"


def test_goal_normalizes_and_bounds_text_fields(judge_module, monkeypatch) -> None:
    payload = {
        "achieved": True,
        "score": 1.0,
        "reason": "  " + ("r" * 600) + "  ",
        "user_goal": "  " + ("g" * 600) + "  ",
        "end_state": "  " + ("e" * 600) + "  ",
    }

    result = _judge_goal_with_response(judge_module, monkeypatch, content=json.dumps(payload))

    assert result["reason"] == ("r" * 509) + "..."
    assert result["user_goal"] == ("g" * 509) + "..."
    assert result["end_state"] == ("e" * 509) + "..."


def test_goal_accepts_unique_object_before_unmatched_trailing_prose_without_retry(
    judge_module,
    monkeypatch,
) -> None:
    calls: list[str] = []
    content = '{"achieved": true}\nNote [draft'
    if judge_module is llm_judge:
        monkeypatch.setattr(judge_module, "call_public_llm", _pair_script([(content, None)], calls))
    else:
        monkeypatch.setattr(judge_module, "_ragas_goal_accuracy_enabled", lambda: False)

        def fake_call(*_args, **_kwargs):
            calls.append("goal")
            return content, None, {}

        monkeypatch.setattr(judge_module, "_call_public_llm_with_provenance", fake_call)

    result = judge_module.judge_goal_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 1
    assert result["score"] == 1.0
    assert result.get("status") != "error"


def test_goal_rejects_nonstandard_constant_in_metadata(judge_module, monkeypatch) -> None:
    result = _judge_goal_with_response(
        judge_module,
        monkeypatch,
        content='{"achieved": true, "metadata": Infinity}',
    )

    _assert_error_result(result)


def test_template_goal_preserves_provenance_on_success_and_error(monkeypatch) -> None:
    provenance = {"provider": "openai", "model": "judge-model"}

    success = _judge_goal_with_response(
        eval_template,
        monkeypatch,
        content='{"achieved": true, "reason": "ok"}',
        provenance=provenance,
    )
    failure = _judge_goal_with_response(
        eval_template,
        monkeypatch,
        content=None,
        error="request timed out",
        provenance=provenance,
    )

    assert {key: success[key] for key in provenance} == provenance
    assert {key: failure[key] for key in provenance} == provenance
    _assert_error_result(failure)


def test_template_nonfinite_ragas_score_falls_back_to_custom(monkeypatch) -> None:
    custom_result = {
        "score": 0.0,
        "reason": "custom judgment",
        "method": "custom",
        "provider": "openai",
        "model": "judge-model",
    }
    monkeypatch.setattr(eval_template, "_ragas_goal_accuracy_enabled", lambda: True)
    monkeypatch.setattr(
        eval_template,
        "_judge_goal_accuracy_ragas",
        lambda *_args, **_kwargs: {"score": math.nan, "method": "ragas"},
    )
    monkeypatch.setattr(eval_template, "_judge_goal_accuracy_custom", lambda *_args, **_kwargs: custom_result)

    result = eval_template.judge_goal_accuracy("question", "ground truth", "agent response")

    assert result == custom_result


def test_behavior_first_provider_error_is_scoreless_without_retry(judge_module, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(None, "LLM judge model fallback exhausted: denied")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["behavior"])

    assert len(calls) == 1
    _assert_error_result(result)


def test_behavior_retries_overflowed_numeric_metadata_once_and_recovers(judge_module, monkeypatch) -> None:
    calls: list[str] = []
    overflowed = '{"results":[{"passed":true,"reason":"ok","metadata":1e999}],"score":1.0}'
    recovered = json.dumps({"results": [{"step": 1, "passed": True, "reason": "ok"}], "summary": "recovered"})
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(overflowed, None), (recovered, None)], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["behavior"])

    assert len(calls) == 2
    assert result["score"] == 1.0
    assert result["reason"] == "recovered"
    assert "previous reply could not be parsed" in calls[1]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("", id="empty"),
        pytest.param("not json", id="prose"),
        pytest.param("[]", id="list"),
        pytest.param("{}", id="empty-object"),
        pytest.param(
            'prefix [{"results": [{"step": 1, "passed": true}], "score": 1.0}] suffix',
            id="prose-wrapped-list",
        ),
        pytest.param('{"results": "wrong", "score": 0}', id="non-list-results"),
        pytest.param('{"results": [1], "score": 0}', id="non-object-result"),
        pytest.param('{"results": [{"step": 1, "passed": 1}], "score": 1}', id="non-boolean-passed"),
        pytest.param(
            '{"results": [{"step": 1, "passed": 1, "passed": true}], "score": 1}',
            id="duplicate-passed",
        ),
        pytest.param('{"results": [], "score": 0}', id="incomplete-results"),
        pytest.param(
            '{"results": [{"step": 1, "passed": false}, {"step": 2, "passed": false}], "score": 0}',
            id="extra-results",
        ),
        pytest.param('{"results": [], "score": true}', id="boolean-score"),
        pytest.param('{"results": [], "score": NaN}', id="nan-score"),
        pytest.param('{"results": [], "score": Infinity}', id="infinite-score"),
        pytest.param(
            '{"results": [{"step": 1, "passed": true}], "score": NaN}',
            id="valid-results-with-nan-score",
        ),
    ],
)
def test_behavior_retries_malformed_or_invalid_shapes_once_then_errors(
    judge_module,
    monkeypatch,
    content: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(judge_module, "call_public_llm", _pair_script([(content, None), (content, None)], calls))

    result = judge_module.judge_behavior_check("conversation", ["behavior"])

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["results"] == []


@pytest.mark.parametrize(
    ("payload", "expected_behaviors", "expected"),
    [
        pytest.param(
            {"results": [{"step": 1, "passed": False}], "summary": "failed"},
            ["b1"],
            0.0,
            id="explicit-zero",
        ),
        pytest.param(
            {"results": [{"step": 2, "passed": False}, {"step": 1, "passed": True}], "summary": "half"},
            ["b1", "b2"],
            0.5,
            id="derived-out-of-order",
        ),
        pytest.param(
            {"results": [{"step": 1, "passed": False}], "score": 10**400, "summary": "ignored score"},
            ["b1"],
            0.0,
            id="huge-integer-score-ignored",
        ),
        pytest.param(
            {"results": [{"step": 1, "passed": False}], "score": 1.0, "summary": "contradicting score"},
            ["b1"],
            0.0,
            id="contradicting-score-ignored",
        ),
    ],
)
def test_behavior_accepts_valid_zero_and_boolean_results(
    judge_module,
    monkeypatch,
    payload: dict,
    expected_behaviors: list[str],
    expected: float,
) -> None:
    calls: list[str] = []
    content = json.dumps(payload)
    monkeypatch.setattr(judge_module, "call_public_llm", _pair_script([(content, None)], calls))

    result = judge_module.judge_behavior_check("conversation", expected_behaviors)

    assert len(calls) == 1
    assert result["score"] == expected
    assert result.get("status") != "error"


def test_behavior_accepts_unique_object_before_unmatched_trailing_prose_without_retry(
    judge_module,
    monkeypatch,
) -> None:
    calls: list[str] = []
    content = '{"results":[{"step":1,"passed":true}]}\nNote [draft'
    monkeypatch.setattr(judge_module, "call_public_llm", _pair_script([(content, None)], calls))

    result = judge_module.judge_behavior_check("conversation", ["behavior"])

    assert len(calls) == 1
    assert result["score"] == 1.0
    assert result.get("status") != "error"


def test_behavior_retries_nonstandard_constant_in_metadata_then_errors(
    judge_module,
    monkeypatch,
) -> None:
    calls: list[str] = []
    content = '{"results":[{"step":1,"passed":true,"metadata":-Infinity}]}'
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(content, None), (content, None)], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["behavior"])

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["results"] == []


def test_behavior_retry_transport_failure_salvages_only_complete_first_entries(judge_module, monkeypatch) -> None:
    truncated = '{"results":[{"step":1,"passed":true,"reason":"observed"},{"step":2,"passed":false,"reason":"cut off'
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(truncated, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2", "b3"])

    assert len(calls) == 2
    assert result["score"] == round(1 / 3, 4)
    assert result["results"] == [{"step": 1, "passed": True, "reason": "observed"}]
    assert "1/3" in result["reason"]


def test_behavior_retry_transport_failure_without_salvage_is_redacted_error(judge_module, monkeypatch) -> None:
    credential = "dummy-retry-secret-DO-NOT-RETAIN"
    malformed = f"malformed response head containing {credential}"
    calls: list[str] = []
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(malformed, None), (None, f"timeout echoed {credential}")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["behavior"])

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["results"] == []
    assert credential not in result["reason"]
    assert "malformed response head" not in result["reason"]


@pytest.mark.parametrize(
    "unfinished_entry",
    [
        pytest.param("{bad", id="bare-identifier-key"),
        pytest.param('{"x":,', id="missing-value-before-comma"),
        pytest.param('{"passed":false "x":', id="missing-member-comma"),
        pytest.param('{"passed":truX', id="invalid-literal"),
        pytest.param(
            '{"passed":true,"step":-\N{ARABIC-INDIC DIGIT ONE}',
            id="non-ascii-number-digit",
        ),
        pytest.param('{"passed":NaN', id="non-standard-nan-literal"),
        pytest.param('{"reason":"bad\x01', id="raw-control-in-string"),
        pytest.param('{"reason":"bad\\q', id="invalid-string-escape"),
    ],
)
def test_behavior_salvage_rejects_noncompletable_current_entry(
    judge_module,
    monkeypatch,
    unfinished_entry: str,
) -> None:
    malformed = '{"results":[{"step":1,"passed":true},' + unfinished_entry
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(malformed, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["results"] == []


@pytest.mark.parametrize(
    "unfinished_entry",
    [
        pytest.param('{"step":2,"passed":false,"reason":"cut', id="unfinished-string"),
        pytest.param('{"step":2,"passed":false,"reason":', id="missing-value-at-end"),
        pytest.param('{"step":2,"passed":fal', id="partial-false-literal"),
    ],
)
def test_behavior_salvage_accepts_append_only_completable_current_entry(
    judge_module,
    monkeypatch,
    unfinished_entry: str,
) -> None:
    truncated = '{"results":[{"step":1,"passed":true},' + unfinished_entry
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(truncated, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert len(calls) == 2
    assert result["score"] == 0.5
    assert result["results"] == [{"step": 1, "passed": True}]


def test_behavior_retries_ambiguous_multiple_documents_then_errors(judge_module, monkeypatch) -> None:
    content = '{"results":[{"step":1,"passed":false}]} {"results":[{"step":1,"passed":true}]}'
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(content, None), (content, None)], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["behavior"])

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["results"] == []


def test_behavior_deep_nesting_is_scoreless_instead_of_raising(judge_module, monkeypatch) -> None:
    content = '{"results":[{"step":1,"passed":true,"metadata":' + "[" * 10_000 + "0" + "]" * 10_000 + "}]}"
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(content, None), (content, None)], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["behavior"])

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["results"] == []


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param(
            '{"metadata":'
            + "[" * 900
            + "0"
            + "]" * 900
            + ',"results":[{"step":1,"passed":true},{"step":2,"passed":false',
            id="top-level-field",
        ),
        pytest.param(
            '{"results":[{"step":1,"passed":true,"metadata":'
            + "[" * 900
            + "0"
            + "]" * 900
            + '},{"step":2,"passed":false',
            id="complete-entry",
        ),
        pytest.param(
            '{"results":[{"step":1,"passed":true},{"step":2,"passed":false,"metadata":'
            + "[" * 900
            + "0"
            + "]" * 900
            + ',"reason":"cut',
            id="current-partial-entry",
        ),
    ],
)
def test_behavior_salvage_rejects_excessive_nesting(
    judge_module,
    monkeypatch,
    malformed: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(malformed, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["results"] == []


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param(
            '{"metadata":NaN,"results":[{"step":1,"passed":true},{"step":2,"passed":false',
            id="top-level-field",
        ),
        pytest.param(
            '{"results":[{"step":1,"passed":true,"metadata":Infinity},{"step":2,"passed":false',
            id="complete-entry",
        ),
        pytest.param(
            '{"results":[{"step":1,"passed":true},{"step":2,"passed":false,"metadata":-Infinity,"reason":"cut',
            id="current-partial-entry",
        ),
    ],
)
def test_behavior_salvage_rejects_nonstandard_constants(
    judge_module,
    monkeypatch,
    malformed: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(malformed, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["results"] == []


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param('{"results":null,"metadata":[{"step":1,"passed":true}]', id="unrelated-array"),
        pytest.param('{"metadata":{"results":[{"step":1,"passed":true}]', id="nested-results"),
        pytest.param('{"results":[{"step":1,"passed":true} {"step":2,"passed":false', id="missing-comma"),
        pytest.param(
            '{"results":[{"step":1,"passed":true},{"step":2,"passed":truX}]}',
            id="closed-invalid-token",
        ),
        pytest.param(
            '{"results":[{"step":1,"passed":true},{"step":2,"passed":false,}]}',
            id="closed-trailing-comma",
        ),
        pytest.param('{"score":true,"results":[{"step":1,"passed":true},{"step":2,"passed":false', id="boolean-score"),
        pytest.param('{"score":"0.5","results":[{"step":1,"passed":true},{"step":2,"passed":false', id="string-score"),
        pytest.param('{"score":NaN,"results":[{"step":1,"passed":true},{"step":2,"passed":false', id="nan-score"),
        pytest.param(
            '{"score":Infinity,"results":[{"step":1,"passed":true},{"step":2,"passed":false', id="infinite-score"
        ),
        pytest.param(
            '[{"results":[{"step":1,"passed":true},{"step":2,"passed":false',
            id="array-wrapper",
        ),
        pytest.param(
            '[] {"results":[{"step":1,"passed":true},{"step":2,"passed":false',
            id="complete-array-prefix",
        ),
        pytest.param(
            '}] {"results":[{"step":1,"passed":true},{"step":2,"passed":false',
            id="structural-prefix",
        ),
        pytest.param(
            '{"score":true,"score":0.5,"results":[{"step":1,"passed":true},{"step":2,"passed":false',
            id="duplicate-score",
        ),
        pytest.param(
            '{"results":[{"step":1,"passed":1,"passed":true},{"step":2,"passed":false',
            id="duplicate-passed",
        ),
        pytest.param(
            '{"results":\N{NO-BREAK SPACE}[{"step":1,"passed":true},{"step":2,"passed":false',
            id="non-json-nbsp",
        ),
        pytest.param(
            '{"results":\f[{"step":1,"passed":true},{"step":2,"passed":false',
            id="non-json-form-feed",
        ),
        pytest.param(
            '{"results":\v[{"step":1,"passed":true},{"step":2,"passed":false',
            id="non-json-vertical-tab",
        ),
    ],
)
def test_behavior_salvage_rejects_invalid_top_level_prefixes(
    judge_module,
    monkeypatch,
    malformed: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(malformed, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert len(calls) == 2
    _assert_error_result(result)
    assert result["results"] == []


def test_behavior_salvage_keeps_finite_top_level_score_prefix(judge_module, monkeypatch) -> None:
    truncated = '{"score":0.5,"results":[{"step":1,"passed":true},{"step":2,"passed":false'
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(truncated, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert result["score"] == 0.5
    assert result["results"] == [{"step": 1, "passed": True}]


@pytest.mark.parametrize(
    "prefix", [pytest.param("Here is the JSON:\n", id="prose"), pytest.param("```json\n", id="fence")]
)
def test_behavior_salvage_tolerates_non_structural_wrappers(judge_module, monkeypatch, prefix: str) -> None:
    truncated = f'{prefix}{{"results":[{{"step":1,"passed":true}},{{"step":2,"passed":false'
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(truncated, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert result["score"] == 0.5
    assert result["results"] == [{"step": 1, "passed": True}]


def test_absent_reference_judges_are_explicitly_not_applicable(judge_module) -> None:
    for result in (
        judge_module.judge_accuracy("question", "", "agent response"),
        judge_module.judge_goal_accuracy("question", "", "agent response"),
        judge_module.judge_behavior_check("conversation", []),
    ):
        assert result["score"] is None
        assert result["status"] == "not_applicable"
        assert result["reason"].startswith("N/A:")
    assert judge_module.judge_behavior_check("conversation", [])["results"] == []


def test_shared_and_template_accuracy_core_results_match(monkeypatch) -> None:
    responses = [
        json.dumps({"score": 0.0, "reason": "valid zero"}),
        json.dumps({"criteria": _CRITERIA, "reason": "derived"}),
        '{"score": NaN}',
    ]
    for content in responses:
        monkeypatch.setattr(
            llm_judge,
            "call_public_llm",
            lambda *_args, _content=content, **_kwargs: (_content, None),
        )
        monkeypatch.setattr(
            eval_template,
            "call_public_llm",
            lambda *_args, _content=content, **_kwargs: (_content, None),
        )

        assert eval_template.judge_accuracy("q", "gt", "answer") == llm_judge.judge_accuracy("q", "gt", "answer")


def test_shared_and_template_behavior_results_match(monkeypatch) -> None:
    malformed = "not json"
    shared_calls: list[str] = []
    template_calls: list[str] = []
    monkeypatch.setattr(
        llm_judge,
        "call_public_llm",
        _pair_script([(malformed, None), (malformed, None)], shared_calls),
    )
    monkeypatch.setattr(
        eval_template,
        "call_public_llm",
        _pair_script([(malformed, None), (malformed, None)], template_calls),
    )

    shared = llm_judge.judge_behavior_check("conversation", ["behavior"])
    template = eval_template.judge_behavior_check("conversation", ["behavior"])

    assert template == shared
    assert len(shared_calls) == len(template_calls) == 2
