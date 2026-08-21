# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for behavior_check judge response parsing.

Root cause (live-reproduced 2026-06-10 by replaying Harbor trial
``evaluator-plugin-001__DrQLA2j`` from run ``aces-nick-e2e-rebased-20260608-092041``
against a real OpenAI-compatible reasoning endpoint): the verifier judge ran ``openai/openai/gpt-5.5``
(a reasoning model, via ``[verifier.env] SKILL_EVAL_JUDGE_MODEL``) with
``max_tokens=1024``.  That budget includes hidden reasoning tokens; on 3 of 4
replay attempts the API returned ``finish_reason="length"`` with
``usage.reasoning_tokens=1024`` and EMPTY visible content, which
``judge_behavior_check`` scored as 0.0 "Could not parse judge response" --
indistinguishable from a genuine zero.  behavior_check is hit (not
accuracy/goal_accuracy) because its per-behavior ``results`` array is the
longest judge output, and gpt-5* judges run without ``temperature=0``
(``_supports_custom_temperature``), so reasoning length varies run-to-run --
hence intermittent (5/16 trials).

Fix contract under test:
- behavior judge requests a reasoning-safe ``max_tokens`` (>= 4096) by default;
- one retry max with a "ONLY minified JSON" reminder on parse failure;
- complete entries are salvaged from a truncated ``results`` array;
- final parse failure is a scoreless structured error with a bounded reason;
- ``_extract_json`` tolerates fenced / prose-wrapped / trailing-junk responses;
- the in-sandbox template copy (``harbor/templates/eval.py``) stays equivalent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from skillevaluator.tier3.eval_core import llm_judge

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)


def _load_template_module():
    spec = importlib.util.spec_from_file_location("harbor_template_eval_parse", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_template = _load_template_module()


# ---------------------------------------------------------------------------
# Captured offending shapes
# ---------------------------------------------------------------------------

# Exact live failure shape: gpt-5.5 spent the entire 1024-token completion
# budget on reasoning (finish_reason="length", reasoning_tokens=1024) and the
# visible message content came back EMPTY.
EMPTY_REASONING_BURN_RESPONSE = ""

# Adjacent live shape: when some budget is left after reasoning the judge emits
# the results array and hits the cap mid-string.  Structure taken from the
# captured 810-char PARSE_OK replay response (5 entries, minified, starts with
# {"results":[{"step":1,...); reasons redacted, cut applied inside entry 3.
TRUNCATED_RESULTS_RESPONSE = (
    '{"results":[{"step":1,"passed":true,"reason":"Skill doc opened before responding."},'
    '{"step":2,"passed":false,"reason":"Required run command was not executed."},'
    '{"step":3,"passed":true,"reason":"Completed result with artifact URL observed in the'
)

# Same truncation class, but every recovered entry passed: the cap cut entry 4
# of 7 mid-string, so exactly 3 complete all-passed entries are salvageable.
# Scoring those 3/3 instead of 3/7 silently inflates behavior_check to 1.0.
TRUNCATED_ALL_PASSED_RESPONSE = (
    '{"results":[{"step":1,"passed":true,"reason":"observed"},'
    '{"step":2,"passed":true,"reason":"observed"},'
    '{"step":3,"passed":true,"reason":"observed"},'
    '{"step":4,"passed":true,"reason":"cut by the completion cap mid-'
)

VALID_BEHAVIOR_RESPONSE = json.dumps(
    {
        "results": [
            {"step": 1, "passed": True, "reason": "observed"},
            {"step": 2, "passed": True, "reason": "observed"},
        ],
        "score": 1.0,
        "summary": "All expected behaviors were observed.",
    }
)

_PARSED_OBJECT = {"results": [{"step": 1, "passed": True, "reason": "ok"}], "score": 1.0, "summary": "fine"}
_MINIFIED = json.dumps(_PARSED_OBJECT)

# Model formatting drift shapes the parser must tolerate (same fix class).
TOLERATED_SHAPES = [
    pytest.param(_MINIFIED, _PARSED_OBJECT, id="plain-minified"),
    pytest.param(f"```json\n{_MINIFIED}\n```", _PARSED_OBJECT, id="json-fence"),
    pytest.param(f"```\n{_MINIFIED}\n```", _PARSED_OBJECT, id="bare-fence"),
    pytest.param(
        f"Here is my evaluation:\n```json\n{_MINIFIED}\n```\nLet me know if you need more.",
        _PARSED_OBJECT,
        id="prose-wrapped-fence",
    ),
    pytest.param(
        f"Sure! The verdict follows.\n{_MINIFIED}",
        _PARSED_OBJECT,
        id="leading-prose",
    ),
    pytest.param(
        f"{_MINIFIED}\nNote: entry {{step 1}} was judged leniently.",
        _PARSED_OBJECT,
        id="trailing-prose-with-braces",
    ),
]

REJECTED_SHAPES = [
    pytest.param(EMPTY_REASONING_BURN_RESPONSE, id="empty-reasoning-burn"),
    pytest.param(TRUNCATED_RESULTS_RESPONSE, id="truncated-mid-string"),
    pytest.param("no json here at all", id="no-json"),
]


# ---------------------------------------------------------------------------
# _extract_json tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "expected"), TOLERATED_SHAPES)
def test_extract_json_tolerates_observed_shapes(text, expected):
    assert llm_judge._extract_json(text) == expected


@pytest.mark.parametrize("text", REJECTED_SHAPES)
def test_extract_json_returns_none_for_unrecoverable_text(text):
    assert llm_judge._extract_json(text) is None


def test_extract_json_preserves_top_level_arrays():
    # harbor.report._generate_suggestions_structured relies on list passthrough.
    assert llm_judge._extract_json('[{"suggestion": "s"}]') == [{"suggestion": "s"}]
    assert eval_template.extract_json('[{"suggestion": "s"}]') == [{"suggestion": "s"}]


def test_extract_json_preserves_prose_wrapped_top_level_arrays():
    text = 'prefix [{"score": 1.0}] trailing prose'
    expected = [{"score": 1.0}]

    assert llm_judge._extract_json(text) == expected
    assert eval_template.extract_json(text) == expected


def test_extract_json_does_not_promote_object_from_truncated_outer_array():
    text = '[{"score":0}'

    assert llm_judge._extract_json(text) is None
    assert eval_template.extract_json(text) is None


def test_extract_json_ignores_non_json_bracket_label_before_unique_object():
    text = 'Judge [draft]: {"score": 0.5}'
    expected = {"score": 0.5}

    assert llm_judge._extract_json(text) == expected
    assert eval_template.extract_json(text) == expected


def test_extract_json_keeps_unique_object_before_unmatched_trailing_prose():
    text = '{"score": 0.5}\nNote [draft'
    expected = {"score": 0.5}

    assert llm_judge._extract_json(text) == expected
    assert eval_template.extract_json(text) == expected


def test_extract_json_ignores_closed_non_json_label_before_unique_array():
    text = 'Judge [draft]: [{"score": 0.5}]'
    expected = [{"score": 0.5}]

    assert llm_judge._extract_json(text) == expected
    assert eval_template.extract_json(text) == expected


def test_extract_json_does_not_cross_unclosed_structural_prefix():
    text = '[draft {"score": 0.5}'

    assert llm_judge._extract_json(text) is None
    assert eval_template.extract_json(text) is None


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"score": 0} {"score": 1}', id="adjacent-objects"),
        pytest.param(
            '{"score": 0}\n```json\n{"score": 1}\n```',
            id="object-and-conflicting-fence",
        ),
    ],
)
def test_extract_json_rejects_multiple_complete_documents(text):
    assert llm_judge._extract_json(text) is None
    assert eval_template.extract_json(text) is None


def test_extract_json_returns_none_for_excessive_nesting():
    text = "[" * 10_000 + '{"score": 1.0}' + "]" * 10_000

    assert llm_judge._extract_json(text) is None
    assert eval_template.extract_json(text) is None


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"score": true, "score": 0.5}', id="top-level"),
        pytest.param('{"results": [{"passed": 1, "passed": true}]}', id="nested"),
    ],
)
def test_extract_json_rejects_duplicate_object_members(text):
    assert llm_judge._extract_json(text) is None
    assert eval_template.extract_json(text) is None


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_extract_json_rejects_nonstandard_json_constants(constant):
    text = f'{{"metadata": {constant}}}'

    assert llm_judge._extract_json(text) is None
    assert eval_template.extract_json(text) is None


def test_behavior_judge_treats_list_payload_as_unparseable(monkeypatch):
    # A bare array is valid JSON but not a judge object; old code raised
    # AttributeError on parsed.get -- now it takes the structured-error path.
    calls: list[dict] = []
    monkeypatch.setattr(llm_judge, "call_public_llm", _scripted_hub(["[1, 2, 3]", "[1, 2, 3]"], calls))

    result = llm_judge.judge_behavior_check("conversation", ["b1"])

    assert len(calls) == 2
    assert result["score"] is None
    assert result["status"] == "error"
    assert "unparseable" in result["reason"]


# ---------------------------------------------------------------------------
# Salvage of truncated results arrays
# ---------------------------------------------------------------------------


def test_salvage_recovers_complete_entries_from_truncated_results():
    salvaged = llm_judge._salvage_behavior_results(TRUNCATED_RESULTS_RESPONSE)

    assert [r["step"] for r in salvaged] == [1, 2]
    assert [r["passed"] for r in salvaged] == [True, False]


def test_salvage_returns_empty_for_empty_or_alien_text():
    assert llm_judge._salvage_behavior_results(EMPTY_REASONING_BURN_RESPONSE) == []
    assert llm_judge._salvage_behavior_results("plain prose") == []


# ---------------------------------------------------------------------------
# Behavior judge: max_tokens, retry, salvage, diagnostic failure
# ---------------------------------------------------------------------------


def _scripted_hub(responses, calls):
    """Return a call_public_llm stub replaying *responses* and recording calls."""

    def fake_hub(prompt, **kwargs):
        calls.append({"prompt": prompt, "kwargs": kwargs})
        response = responses[min(len(calls) - 1, len(responses) - 1)]
        return response, None

    return fake_hub


def test_behavior_judge_requests_reasoning_safe_max_tokens(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(llm_judge, "call_public_llm", _scripted_hub([VALID_BEHAVIOR_RESPONSE], calls))

    llm_judge.judge_behavior_check("conversation", ["behavior one"])

    assert calls[0]["kwargs"].get("max_tokens", 0) >= 4096


def test_behavior_judge_keeps_explicit_max_tokens_override(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(llm_judge, "call_public_llm", _scripted_hub([VALID_BEHAVIOR_RESPONSE], calls))

    llm_judge.judge_behavior_check("conversation", ["behavior one"], max_tokens=2048)

    assert calls[0]["kwargs"]["max_tokens"] == 2048


def test_behavior_judge_retries_once_with_minified_json_reminder(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_judge,
        "call_public_llm",
        _scripted_hub([EMPTY_REASONING_BURN_RESPONSE, VALID_BEHAVIOR_RESPONSE], calls),
    )

    result = llm_judge.judge_behavior_check("conversation", ["b1", "b2"])

    assert len(calls) == 2  # exactly one retry
    assert "ONLY" in calls[1]["prompt"]
    assert "JSON" in calls[1]["prompt"]
    assert result["score"] == 1.0
    assert result["reason"] == "All expected behaviors were observed."


def test_behavior_judge_empty_reasoning_burn_yields_scoreless_error(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_judge,
        "call_public_llm",
        _scripted_hub([EMPTY_REASONING_BURN_RESPONSE, EMPTY_REASONING_BURN_RESPONSE], calls),
    )

    result = llm_judge.judge_behavior_check("conversation", ["b1"])

    assert len(calls) == 2  # one retry max, never more
    assert result["score"] is None
    assert result["status"] == "error"
    assert result["results"] == []
    # Diagnostic, filterable reason -- never the old ambiguous string.
    assert result["reason"] != "Could not parse judge response"
    assert "unparseable" in result["reason"]
    assert "response" in result["reason"].lower()


def test_behavior_judge_salvages_truncated_results_after_retry(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_judge,
        "call_public_llm",
        _scripted_hub([TRUNCATED_RESULTS_RESPONSE, TRUNCATED_RESULTS_RESPONSE], calls),
    )

    result = llm_judge.judge_behavior_check("conversation", ["b1", "b2", "b3"])

    assert len(calls) == 2
    # Two complete entries recovered (step 1 passed, step 2 failed) out of
    # THREE expected behaviors; the unrecovered third counts as not-passed.
    assert result["score"] == round(1 / 3, 4)
    assert [r["step"] for r in result["results"]] == [1, 2]
    assert "salvaged" in result["reason"].lower()
    assert "2/3" in result["reason"]


def test_behavior_judge_salvage_scores_against_expected_behavior_count(monkeypatch):
    # Inflation regression: 3 salvaged all-passed entries from 7 expected
    # behaviors must score 3/7 (unrecovered behaviors not-passed), never 3/3.
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_judge,
        "call_public_llm",
        _scripted_hub([TRUNCATED_ALL_PASSED_RESPONSE, TRUNCATED_ALL_PASSED_RESPONSE], calls),
    )
    behaviors = [f"behavior {i}" for i in range(1, 8)]  # 7 expected

    result = llm_judge.judge_behavior_check("conversation", behaviors)

    assert [r["step"] for r in result["results"]] == [1, 2, 3]
    assert result["score"] == round(3 / 7, 4)  # 0.4286, NOT 1.0
    assert "salvaged" in result["reason"].lower()
    assert "3/7" in result["reason"]
    assert "truncated" in result["reason"].lower()


def test_behavior_judge_success_path_unchanged(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(llm_judge, "call_public_llm", _scripted_hub([VALID_BEHAVIOR_RESPONSE], calls))

    result = llm_judge.judge_behavior_check("conversation", ["b1", "b2"])

    assert len(calls) == 1  # no retry on success
    assert result["score"] == 1.0


# ---------------------------------------------------------------------------
# Drift guards: in-sandbox template must stay equivalent to eval_core
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "expected"), TOLERATED_SHAPES)
def test_template_extract_json_matches_eval_core_on_tolerated_shapes(text, expected):
    assert eval_template.extract_json(text) == llm_judge._extract_json(text) == expected


@pytest.mark.parametrize("text", REJECTED_SHAPES)
def test_template_extract_json_matches_eval_core_on_rejected_shapes(text):
    assert eval_template.extract_json(text) is None
    assert llm_judge._extract_json(text) is None


def test_template_salvage_matches_eval_core():
    assert eval_template._salvage_behavior_results(TRUNCATED_RESULTS_RESPONSE) == llm_judge._salvage_behavior_results(
        TRUNCATED_RESULTS_RESPONSE
    )


@pytest.mark.parametrize(
    "responses",
    [
        pytest.param([EMPTY_REASONING_BURN_RESPONSE, EMPTY_REASONING_BURN_RESPONSE], id="empty-empty"),
        pytest.param([EMPTY_REASONING_BURN_RESPONSE, VALID_BEHAVIOR_RESPONSE], id="empty-then-valid"),
        pytest.param([TRUNCATED_RESULTS_RESPONSE, TRUNCATED_RESULTS_RESPONSE], id="truncated-truncated"),
        pytest.param([VALID_BEHAVIOR_RESPONSE], id="valid-first-try"),
    ],
)
def test_template_behavior_judge_matches_eval_core(monkeypatch, responses):
    shared_calls: list[dict] = []
    template_calls: list[dict] = []
    monkeypatch.setattr(llm_judge, "call_public_llm", _scripted_hub(responses, shared_calls))
    monkeypatch.setattr(eval_template, "call_public_llm", _scripted_hub(responses, template_calls))

    shared = llm_judge.judge_behavior_check("conversation", ["b1", "b2"])
    template = eval_template.judge_behavior_check("conversation", ["b1", "b2"])

    assert template == shared
    assert len(template_calls) == len(shared_calls)
    assert template_calls[0]["kwargs"].get("max_tokens", 0) >= 4096


def test_template_behavior_judge_max_tokens_matches_eval_core_constant():
    assert eval_template.BEHAVIOR_JUDGE_MAX_TOKENS == llm_judge.BEHAVIOR_JUDGE_MAX_TOKENS
    assert llm_judge.BEHAVIOR_JUDGE_MAX_TOKENS >= 4096


def test_template_default_judge_model_matches_central_constant():
    from skillevaluator.provider_config import CHAT_DEFAULT_OPENAI

    assert eval_template.DEFAULT_JUDGE_MODEL == llm_judge.DEFAULT_JUDGE_MODEL
    assert llm_judge.DEFAULT_JUDGE_MODEL == CHAT_DEFAULT_OPENAI
    assert CHAT_DEFAULT_OPENAI == "gpt-5.6-sol"


def test_template_gpt5_temperature_guard_matches_eval_core():
    for model in (
        "gpt-5.6-sol",
        "openai/gpt-5.6-sol",
        "openai/openai/gpt-5.6-sol",
        "gpt-5.4-mini",
        "gpt-4.1-mini",
        "claude-mythos-preview",
        "anthropic/claude-mythos-preview",
    ):
        assert eval_template._model_leaf(model) == llm_judge._model_leaf(model)
        assert eval_template._supports_custom_temperature(model) == llm_judge._supports_custom_temperature(model)


@pytest.mark.parametrize(
    ("model", "expected_temperature"),
    [
        ("claude-opus-4-8", None),
        ("claude-opus-5", None),
        ("claude-mythos-preview", None),
        ("claude-3-5-sonnet-20241022", 0.3),
    ],
)
def test_template_anthropic_request_uses_model_compatible_temperature(monkeypatch, model, expected_temperature):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"content":[{"type":"text","text":"Done"}]}'

    def fake_urlopen(request, timeout):
        assert timeout == 90
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(eval_template.urllib.request, "urlopen", fake_urlopen)

    content, error = eval_template._call_anthropic("prompt", model, 4096, 0.3)

    assert (content, error) == ("Done", None)
    assert captured["max_tokens"] == 4096
    if expected_temperature is None:
        assert "temperature" not in captured
    else:
        assert captured["temperature"] == expected_temperature


@pytest.mark.parametrize(
    ("model", "expected_temperature"),
    [
        ("us.anthropic.claude-opus-4-8", None),
        ("us.anthropic.claude-opus-5", None),
        ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", 0.3),
    ],
)
def test_template_bedrock_request_uses_model_compatible_temperature(monkeypatch, model, expected_temperature):
    import boto3

    client = MagicMock()
    client.converse.return_value = {"output": {"message": {"content": [{"text": "Done"}]}}}
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)

    content, error = eval_template._call_bedrock("prompt", model, 4096, 0.3)

    assert (content, error) == ("Done", None)
    inference_config = client.converse.call_args.kwargs["inferenceConfig"]
    assert inference_config["maxTokens"] == 4096
    if expected_temperature is None:
        assert "temperature" not in inference_config
    else:
        assert inference_config["temperature"] == expected_temperature


@pytest.mark.parametrize(
    "model",
    ["gpt-5.6-sol", "openai/gpt-5.6-sol", "openai/openai/gpt-5.6-sol"],
)
def test_template_native_openai_gpt5_payload_matches_eval_core(model):
    kwargs = {
        "model": model,
        "prompt": "Judge this response",
        "max_tokens": 321,
        "temperature": 0.25,
        "provider": "openai",
        "request_url": llm_judge.OPENAI_CHAT_URL,
    }

    template_payload = eval_template._chat_completion_payload(**kwargs)
    assert template_payload == llm_judge._chat_completion_payload(**kwargs)
    assert "max_completion_tokens" in template_payload
    assert "max_tokens" not in template_payload


def test_template_salvage_score_matches_eval_core_on_partial_recovery(monkeypatch):
    # Drift guard for the salvage denominator fix: 3 salvaged of 7 expected
    # must score 3/7 in BOTH the shared judge and the in-sandbox template.
    responses = [TRUNCATED_ALL_PASSED_RESPONSE, TRUNCATED_ALL_PASSED_RESPONSE]
    shared_calls: list[dict] = []
    template_calls: list[dict] = []
    monkeypatch.setattr(llm_judge, "call_public_llm", _scripted_hub(responses, shared_calls))
    monkeypatch.setattr(eval_template, "call_public_llm", _scripted_hub(responses, template_calls))
    behaviors = [f"behavior {i}" for i in range(1, 8)]  # 7 expected

    shared = llm_judge.judge_behavior_check("conversation", behaviors)
    template = eval_template.judge_behavior_check("conversation", behaviors)

    assert template == shared
    assert shared["score"] == round(3 / 7, 4)
    assert "3/7" in template["reason"]
