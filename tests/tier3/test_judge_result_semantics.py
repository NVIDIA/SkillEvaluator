# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed result semantics for the Tier 3 LLM judges."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest

from skillevaluator.tier3.eval_core import llm_judge

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)
_CRITERIA = {
    "SKILL_IDENTIFIED": True,
    "ACTION_CORRECT": False,
    "FACTUALLY_ACCURATE": True,
    "TASK_ADDRESSED": True,
    "ACTIONABLE": False,
}


def _load_template_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("harbor_template_eval_judge_results", _TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_template = _load_template_module()


@pytest.fixture(params=["shared", "template"])
def judge_module(request: pytest.FixtureRequest) -> ModuleType:
    return llm_judge if request.param == "shared" else eval_template


def _pair_script(responses: list[tuple[str | None, str | None]], calls: list[str]):
    def fake_call(prompt: str, **_kwargs):
        calls.append(prompt)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    return fake_call


def _assert_error_result(result: dict) -> None:
    assert result["score"] is None
    assert result["status"] == "error"
    assert isinstance(result["reason"], str)
    assert 0 < len(result["reason"]) <= 512


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
        pytest.param('{"score": true}', id="boolean-score"),
        pytest.param('{"score": true, "score": 0.5}', id="duplicate-score"),
        pytest.param('{"score": NaN}', id="nan-score"),
        pytest.param('{"score": Infinity}', id="infinite-score"),
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
    ("score", "expected"),
    [
        pytest.param(0.0, 0.0, id="genuine-zero"),
        pytest.param(-2, 0.0, id="clamp-low"),
        pytest.param(4.2, 1.0, id="clamp-high"),
        pytest.param(-(10**400), 0.0, id="clamp-huge-integer-low"),
        pytest.param(10**400, 1.0, id="clamp-huge-integer-high"),
    ],
)
def test_accuracy_accepts_finite_numeric_scores(judge_module, monkeypatch, score: float, expected: float) -> None:
    content = json.dumps({"score": score, "reason": "model judgment"})
    monkeypatch.setattr(judge_module, "call_public_llm", lambda *_args, **_kwargs: (content, None))

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert result["score"] == expected
    assert result.get("status") != "error"


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
        lambda *_args, **_kwargs: ('Judge [draft]: {"score": 0.5}', None),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert result["score"] == 0.5
    assert result.get("status") != "error"


def test_accuracy_accepts_unique_object_before_unmatched_trailing_prose_without_retry(
    judge_module,
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([('{"score": 0.5}\nNote [draft', None)], calls),
    )

    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert len(calls) == 1
    assert result["score"] == 0.5
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
        pytest.param('{"achieved": true, "score": true}', id="boolean-score"),
        pytest.param('{"achieved": true, "score": "0.8"}', id="string-score"),
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
        pytest.param({"achieved": False, "score": 0.25}, 0.25, id="finite-override"),
        pytest.param({"achieved": True, "score": -(10**400)}, 0.0, id="huge-integer-low"),
        pytest.param({"achieved": False, "score": 10**400}, 1.0, id="huge-integer-high"),
    ],
)
def test_goal_requires_boolean_achieved_and_uses_only_finite_numeric_optional_score(
    judge_module,
    monkeypatch,
    payload: dict,
    expected: float,
) -> None:
    payload = {"reason": "model judgment", "user_goal": "goal", "end_state": "state", **payload}

    result = _judge_goal_with_response(judge_module, monkeypatch, content=json.dumps(payload))

    assert result["score"] == expected
    assert result.get("status") != "error"


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


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("", id="empty"),
        pytest.param("not json", id="prose"),
        pytest.param("[]", id="list"),
        pytest.param("{}", id="empty-object"),
        pytest.param(
            'prefix [{"results": [{"passed": true}], "score": 1.0}] suffix',
            id="prose-wrapped-list",
        ),
        pytest.param('{"results": "wrong", "score": 0}', id="non-list-results"),
        pytest.param('{"results": [1], "score": 0}', id="non-object-result"),
        pytest.param('{"results": [{"passed": 1}], "score": 1}', id="non-boolean-passed"),
        pytest.param(
            '{"results": [{"passed": 1, "passed": true}], "score": 1}',
            id="duplicate-passed",
        ),
        pytest.param('{"results": [], "score": 0}', id="incomplete-results"),
        pytest.param(
            '{"results": [{"passed": false}, {"passed": false}], "score": 0}',
            id="extra-results",
        ),
        pytest.param('{"results": [], "score": true}', id="boolean-score"),
        pytest.param('{"results": [], "score": NaN}', id="nan-score"),
        pytest.param('{"results": [], "score": Infinity}', id="infinite-score"),
        pytest.param(
            '{"results": [{"passed": true}], "score": NaN}',
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
            {"results": [{"passed": False}], "score": 0.0, "summary": "failed"},
            ["b1"],
            0.0,
            id="explicit-zero",
        ),
        pytest.param(
            {"results": [{"passed": True}, {"passed": False}], "summary": "half"},
            ["b1", "b2"],
            0.5,
            id="derived",
        ),
        pytest.param(
            {"results": [{"passed": False}], "score": 10**400, "summary": "valid integer"},
            ["b1"],
            0.0,
            id="huge-integer-score",
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
    content = '{"results":[{"passed":true}]}\nNote [draft'
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
    content = '{"results":[{"passed":true,"metadata":-Infinity}]}'
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
    malformed = '{"results":[{"passed":true},' + unfinished_entry
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
        pytest.param('{"passed":false,"reason":"cut', id="unfinished-string"),
        pytest.param('{"passed":false,"reason":', id="missing-value-at-end"),
        pytest.param('{"passed":fal', id="partial-false-literal"),
    ],
)
def test_behavior_salvage_accepts_append_only_completable_current_entry(
    judge_module,
    monkeypatch,
    unfinished_entry: str,
) -> None:
    truncated = '{"results":[{"passed":true},' + unfinished_entry
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(truncated, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert len(calls) == 2
    assert result["score"] == 0.5
    assert result["results"] == [{"passed": True}]


def test_behavior_retries_ambiguous_multiple_documents_then_errors(judge_module, monkeypatch) -> None:
    content = '{"results":[{"passed":false}]} {"results":[{"passed":true}]}'
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
    content = '{"results":[{"passed":true,"metadata":' + "[" * 10_000 + "0" + "]" * 10_000 + "}]}"
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
            '{"metadata":' + "[" * 900 + "0" + "]" * 900 + ',"results":[{"passed":true},{"passed":false',
            id="top-level-field",
        ),
        pytest.param(
            '{"results":[{"passed":true,"metadata":' + "[" * 900 + "0" + "]" * 900 + '},{"passed":false',
            id="complete-entry",
        ),
        pytest.param(
            '{"results":[{"passed":true},{"passed":false,"metadata":' + "[" * 900 + "0" + "]" * 900 + ',"reason":"cut',
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
            '{"metadata":NaN,"results":[{"passed":true},{"passed":false',
            id="top-level-field",
        ),
        pytest.param(
            '{"results":[{"passed":true,"metadata":Infinity},{"passed":false',
            id="complete-entry",
        ),
        pytest.param(
            '{"results":[{"passed":true},{"passed":false,"metadata":-Infinity,"reason":"cut',
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
        pytest.param('{"results":null,"metadata":[{"passed":true}]', id="unrelated-array"),
        pytest.param('{"metadata":{"results":[{"passed":true}]', id="nested-results"),
        pytest.param('{"results":[{"passed":true} {"passed":false', id="missing-comma"),
        pytest.param(
            '{"results":[{"passed":true},{"passed":truX}]}',
            id="closed-invalid-token",
        ),
        pytest.param(
            '{"results":[{"passed":true},{"passed":false,}]}',
            id="closed-trailing-comma",
        ),
        pytest.param('{"score":true,"results":[{"passed":true},{"passed":false', id="boolean-score"),
        pytest.param('{"score":"0.5","results":[{"passed":true},{"passed":false', id="string-score"),
        pytest.param('{"score":NaN,"results":[{"passed":true},{"passed":false', id="nan-score"),
        pytest.param('{"score":Infinity,"results":[{"passed":true},{"passed":false', id="infinite-score"),
        pytest.param(
            '[{"results":[{"passed":true},{"passed":false',
            id="array-wrapper",
        ),
        pytest.param(
            '[] {"results":[{"passed":true},{"passed":false',
            id="complete-array-prefix",
        ),
        pytest.param(
            '}] {"results":[{"passed":true},{"passed":false',
            id="structural-prefix",
        ),
        pytest.param(
            '{"score":true,"score":0.5,"results":[{"passed":true},{"passed":false',
            id="duplicate-score",
        ),
        pytest.param(
            '{"results":[{"passed":1,"passed":true},{"passed":false',
            id="duplicate-passed",
        ),
        pytest.param(
            '{"results":\N{NO-BREAK SPACE}[{"passed":true},{"passed":false',
            id="non-json-nbsp",
        ),
        pytest.param(
            '{"results":\f[{"passed":true},{"passed":false',
            id="non-json-form-feed",
        ),
        pytest.param(
            '{"results":\v[{"passed":true},{"passed":false',
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
    truncated = '{"score":0.5,"results":[{"passed":true},{"passed":false'
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(truncated, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert result["score"] == 0.5
    assert result["results"] == [{"passed": True}]


@pytest.mark.parametrize(
    "prefix", [pytest.param("Here is the JSON:\n", id="prose"), pytest.param("```json\n", id="fence")]
)
def test_behavior_salvage_tolerates_non_structural_wrappers(judge_module, monkeypatch, prefix: str) -> None:
    truncated = f'{prefix}{{"results":[{{"passed":true}},{{"passed":false'
    calls: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "call_public_llm",
        _pair_script([(truncated, None), (None, "request timed out")], calls),
    )

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert result["score"] == 0.5
    assert result["results"] == [{"passed": True}]


def test_neutral_judge_skips_remain_unchanged(judge_module) -> None:
    assert judge_module.judge_accuracy("question", "", "agent response") == {
        "score": 1.0,
        "reason": "No ground_truth -- skipped",
    }
    assert judge_module.judge_goal_accuracy("question", "", "agent response") == {
        "score": 1.0,
        "reason": "No ground_truth -- skipped",
    }
    assert judge_module.judge_behavior_check("conversation", []) == {
        "score": 1.0,
        "reason": "No expected_behavior defined",
        "results": [],
    }


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
