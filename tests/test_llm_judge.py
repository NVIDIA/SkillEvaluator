# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM judge request-payload compatibility tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from skillevaluator.provider_config import CHAT_CHEAP_OPENAI, CHAT_DEFAULT_OPENAI
from skillevaluator.tier3.eval_core import llm_judge


@pytest.mark.parametrize(
    "model",
    [
        CHAT_DEFAULT_OPENAI,
        f"openai/{CHAT_DEFAULT_OPENAI}",
        f"openai/openai/{CHAT_DEFAULT_OPENAI}",
    ],
)
def test_native_openai_gpt5_uses_max_completion_tokens_without_temperature(model: str) -> None:
    payload = llm_judge._chat_completion_payload(
        model=model,
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.25,
        provider="openai",
        request_url=llm_judge.OPENAI_CHAT_URL,
    )

    assert payload == {
        "model": model,
        "max_completion_tokens": 321,
        "messages": [{"role": "user", "content": "Judge this response"}],
    }
    assert "max_tokens" not in payload
    assert "temperature" not in payload


@pytest.mark.parametrize(
    "model",
    [
        CHAT_DEFAULT_OPENAI,
        CHAT_CHEAP_OPENAI,
        f"openai/{CHAT_DEFAULT_OPENAI}",
        f"openai/openai/{CHAT_DEFAULT_OPENAI}",
    ],
)
def test_gpt5_family_rejects_custom_temperature(model: str) -> None:
    assert not llm_judge._supports_custom_temperature(model)


def test_older_models_accept_custom_temperature() -> None:
    assert llm_judge._supports_custom_temperature("gpt-4.1-mini")
    assert llm_judge._supports_custom_temperature("claude-opus-4-6")
    assert llm_judge._supports_custom_temperature("claude-opus-4-20250514")
    assert llm_judge._supports_custom_temperature("claude-3-5-sonnet-20241022")


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-7",
        "claude-opus-4-8",
        "anthropic/claude-opus-4-8",
        "azure/anthropic/claude-opus-4-8",
        "us.anthropic.claude-opus-4-8",
        "bedrock/us.anthropic.claude-opus-4-8",
        "claude-opus-5",
        "anthropic/claude-opus-5",
        "us.anthropic.claude-opus-5",
        "bedrock/us.anthropic.claude-opus-5",
        "claude-sonnet-5",
        "claude-mythos-5",
        "claude-mythos-preview",
        "anthropic/claude-mythos-preview",
    ],
)
def test_newer_claude_models_reject_custom_temperature(model: str) -> None:
    assert not llm_judge._supports_custom_temperature(model)


def test_call_public_llm_uses_production_gpt5_payload_without_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("SKILL_EVAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

    with patch("openai.OpenAI", return_value=mock_openai):
        content, error = llm_judge.call_public_llm("Judge this response", max_tokens=4096, temperature=0.0)

    assert (content, error) == ("Done", None)
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_completion_tokens"] == 4096
    assert "temperature" not in call_kwargs


@pytest.mark.parametrize(
    ("judge_name", "valid_response"),
    [
        (
            "judge_accuracy",
            json.dumps(
                {
                    "criteria": {
                        "SKILL_IDENTIFIED": True,
                        "ACTION_CORRECT": True,
                        "FACTUALLY_ACCURATE": True,
                        "TASK_ADDRESSED": True,
                        "ACTIONABLE": True,
                    },
                    "score": 1.0,
                    "reason": "recovered",
                }
            ),
        ),
        (
            "judge_goal_accuracy",
            json.dumps({"achieved": True, "score": 1.0, "reason": "recovered"}),
        ),
    ],
)
def test_shared_structured_judges_retry_empty_sdk_response(
    monkeypatch: pytest.MonkeyPatch,
    judge_name: str,
    valid_response: str,
) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.delenv("SKILL_EVAL_LLM_MODEL", raising=False)
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=""))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content=valid_response))]),
    ]

    with patch("openai.OpenAI", return_value=mock_openai):
        result = getattr(llm_judge, judge_name)("question", "ground truth", "agent response")

    assert result["score"] == 1.0
    assert mock_openai.chat.completions.create.call_count == 2
    calls = mock_openai.chat.completions.create.call_args_list
    assert [call.kwargs["max_tokens"] for call in calls] == [4096, 4096]
    assert "previous reply could not be parsed or validated" in calls[1].kwargs["messages"][1]["content"]


def test_shared_structured_judge_does_not_retry_sdk_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = RuntimeError("transport unavailable")

    with patch("openai.OpenAI", return_value=mock_openai):
        result = llm_judge.judge_accuracy("question", "ground truth", "agent response")

    assert result["score"] is None
    assert result["status"] == "error"
    assert mock_openai.chat.completions.create.call_count == 1


@pytest.mark.parametrize(
    ("provider", "request_url"),
    [
        ("nv_build", llm_judge.NVIDIA_BUILD_CHAT_URL),
        ("openai", "https://openai-compatible.example/v1/chat/completions"),
    ],
)
def test_non_native_gpt5_requests_keep_max_tokens(provider: str, request_url: str) -> None:
    payload = llm_judge._chat_completion_payload(
        model=CHAT_DEFAULT_OPENAI,
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.0,
        provider=provider,
        request_url=request_url,
    )

    assert payload["max_tokens"] == 321
    assert "max_completion_tokens" not in payload
    assert "temperature" not in payload


def test_native_openai_non_gpt5_keeps_max_tokens() -> None:
    payload = llm_judge._chat_completion_payload(
        model="gpt-4.1-mini",
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.0,
        provider="openai",
        request_url=llm_judge.OPENAI_CHAT_URL,
    )

    assert payload["max_tokens"] == 321
    assert payload["temperature"] == 0.0
    assert "max_completion_tokens" not in payload


def test_completion_token_payload_resolves_provider_and_url_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)

    payload = llm_judge._chat_completion_payload(
        model=CHAT_DEFAULT_OPENAI,
        prompt="Judge this response",
        max_tokens=321,
        temperature=0.0,
    )

    assert payload["max_completion_tokens"] == 321
    assert "max_tokens" not in payload
    assert "temperature" not in payload


@pytest.mark.parametrize(
    "request_url",
    [
        "https://api.openai.com/v1/chat/completions",
        "https://api.openai.com/v1/chat/completions/",
        "HTTPS://API.OPENAI.COM/v1/chat/completions",
        "https://api.openai.com:443/v1/chat/completions",
    ],
)
def test_native_openai_completion_token_url_accepts_only_canonical_variants(request_url: str) -> None:
    assert llm_judge._is_native_openai_chat_url("OPENAI", request_url)


@pytest.mark.parametrize(
    ("provider", "request_url"),
    [
        ("nv_build", "https://api.openai.com/v1/chat/completions"),
        ("openai-compatible", "https://api.openai.com/v1/chat/completions"),
        ("openai", "http://api.openai.com/v1/chat/completions"),
        ("openai", "https://api.openai.com.evil.example/v1/chat/completions"),
        ("openai", "https://user@api.openai.com/v1/chat/completions"),
        ("openai", "https://api.openai.com/v1/chat/completions?route=proxy"),
        ("openai", "https://api.openai.com/v1/chat/completions?"),
        ("openai", "https://api.openai.com/v1/chat/completions#fragment"),
        ("openai", "https://api.openai.com/v1/chat/completions#"),
        ("openai", "https://api.openai.com/v1/chat/completions;proxy"),
        ("openai", "https://api.openai.com/v1/chat/completions;"),
        ("openai", "https://api.openai.com/v1/chat/completionsbeta"),
        ("openai", "https://api.openai.com:444/v1/chat/completions"),
        ("openai", "https://api.openai.com:/v1/chat/completions"),
        ("openai", "https://api.openai.com:invalid/v1/chat/completions"),
        ("openai", "https://api.openai.com\r/v1/chat/completions"),
        ("openai", "https://api.openai.com\n/v1/chat/completions"),
        ("openai", "https://api.openai.com\t/v1/chat/completions"),
        ("openai", " https://api.openai.com/v1/chat/completions"),
        ("openai", "https://api.openai.com/v1/chat/completions "),
    ],
)
def test_deceptive_openai_urls_keep_max_tokens(provider: str, request_url: str) -> None:
    assert not llm_judge._is_native_openai_chat_url(provider, request_url)


@pytest.mark.parametrize(
    ("judge_name", "call_args", "expected_schema_name", "valid_response"),
    [
        (
            "judge_accuracy",
            ("question", "ground truth", "agent response"),
            "accuracy_judgment",
            json.dumps(
                {
                    "criteria": {
                        "SKILL_IDENTIFIED": True,
                        "ACTION_CORRECT": True,
                        "FACTUALLY_ACCURATE": True,
                        "TASK_ADDRESSED": True,
                        "ACTIONABLE": True,
                    },
                    "score": 1.0,
                    "reason": "all good",
                }
            ),
        ),
        (
            "judge_goal_accuracy",
            ("question", "ground truth", "agent response"),
            "goal_accuracy_judgment",
            json.dumps(
                {
                    "user_goal": "do x",
                    "end_state": "did x",
                    "achieved": True,
                    "score": 1.0,
                    "reason": "achieved",
                }
            ),
        ),
        (
            "judge_behavior_check",
            ("conversation", ["behavior 1"]),
            "behavior_check_judgment",
            json.dumps(
                {
                    "results": [{"step": 1, "passed": True, "reason": "observed"}],
                    "score": 1.0,
                    "summary": "observed",
                }
            ),
        ),
    ],
)
def test_structured_judges_pass_openai_response_format_schema(
    monkeypatch: pytest.MonkeyPatch,
    judge_name: str,
    call_args: tuple,
    expected_schema_name: str,
    valid_response: str,
) -> None:
    """Verify OpenAI/compatible judge calls pass strict json_schema in both shared judge and template."""
    import importlib.util
    from pathlib import Path

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.delenv("SKILL_EVAL_LLM_MODEL", raising=False)

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=valid_response))]
    )
    with patch("openai.OpenAI", return_value=mock_openai):
        result = getattr(llm_judge, judge_name)(*call_args)

    assert result["score"] == 1.0
    sdk_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert "response_format" in sdk_kwargs
    rf = sdk_kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == expected_schema_name
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["type"] == "object"
    assert rf["json_schema"]["schema"]["additionalProperties"] is False

    # Verify Harbor template parity on the HTTP boundary
    template_path = (
        Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_template_eval_slice1", template_path)
    assert spec and spec.loader
    eval_template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_template)

    captured_requests: list[dict] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": valid_response}}]}).encode()

    monkeypatch.setattr(
        eval_template.urllib.request,
        "urlopen",
        lambda req, **_: captured_requests.append(json.loads(req.data)) or _Resp(),
    )
    template_result = getattr(eval_template, judge_name)(*call_args)
    assert template_result["score"] == 1.0
    assert len(captured_requests) == 1
    assert captured_requests[0]["response_format"] == rf


def test_structured_judges_pass_anthropic_output_config_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Anthropic judge calls pass output_config.format.json_schema instead of response_format."""
    import importlib.util
    from pathlib import Path

    valid_response = json.dumps(
        {
            "criteria": {
                "SKILL_IDENTIFIED": True,
                "ACTION_CORRECT": True,
                "FACTUALLY_ACCURATE": True,
                "TASK_ADDRESSED": True,
                "ACTIONABLE": True,
            },
            "score": 1.0,
            "reason": "all good",
        }
    )
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    mock_anthropic = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = valid_response
    mock_anthropic.messages.create.return_value = MagicMock(content=[mock_block])

    with patch("anthropic.Anthropic", return_value=mock_anthropic):
        result = llm_judge.judge_accuracy("question", "ground truth", "agent response")

    assert result["score"] == 1.0
    anth_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert "response_format" not in anth_kwargs
    assert "output_config" in anth_kwargs
    assert anth_kwargs["output_config"]["format"]["type"] == "json_schema"
    assert anth_kwargs["output_config"]["format"]["schema"] == llm_judge.ACCURACY_JSON_SCHEMA

    template_path = (
        Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_template_eval_anth", template_path)
    assert spec and spec.loader
    eval_template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_template)

    captured_requests: list[dict] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps({"content": [{"type": "text", "text": valid_response}]}).encode()

    monkeypatch.setattr(
        eval_template.urllib.request,
        "urlopen",
        lambda req, **_: captured_requests.append(json.loads(req.data)) or _Resp(),
    )
    template_result = eval_template.judge_accuracy("question", "ground truth", "agent response")
    assert template_result["score"] == 1.0
    assert "response_format" not in captured_requests[0]
    assert captured_requests[0]["output_config"] == anth_kwargs["output_config"]


def test_accuracy_and_behavior_prompts_use_boolean_instructions() -> None:
    """Verify judge prompts do not instruct 'YES or NO' strings when schema requires booleans."""
    assert "YES or NO" not in llm_judge.ACCURACY_PROMPT
    assert "YES (observed) or NO" not in llm_judge.BEHAVIOR_CHECK_PROMPT


@pytest.fixture(autouse=True)
def _clear_schema_unsupported_targets():
    from skillevaluator.inference import client as inference_client

    target_set = getattr(inference_client, "_SCHEMA_UNSUPPORTED_TARGETS", None)
    if isinstance(target_set, set):
        target_set.clear()
    yield
    if isinstance(target_set, set):
        target_set.clear()


@pytest.mark.parametrize("status_code", [400, 422])
def test_client_and_template_downgrade_on_400_422_and_memoize_across_calls(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    """Verify HTTP 400/422 on schema calls downgrades to prompt-only and memoizes the model."""
    import importlib.util
    import io
    import urllib.error
    from pathlib import Path

    valid_response = json.dumps(
        {
            "criteria": {
                "SKILL_IDENTIFIED": True,
                "ACTION_CORRECT": True,
                "FACTUALLY_ACCURATE": True,
                "TASK_ADDRESSED": True,
                "ACTIONABLE": True,
            },
            "score": 1.0,
            "reason": "downgraded ok",
        }
    )
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "legacy-unsupported-model")

    class _BadRequestError(Exception):
        def __init__(self, code: int):
            super().__init__(f"HTTP {code}: response_format of type 'json_schema' is not supported")
            self.status_code = code

    sdk_calls: list[dict] = []

    def fake_sdk_create(**kwargs):
        sdk_calls.append(kwargs)
        if "response_format" in kwargs:
            raise _BadRequestError(status_code)
        return MagicMock(choices=[MagicMock(message=MagicMock(content=valid_response))])

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = fake_sdk_create

    with patch("openai.OpenAI", return_value=mock_openai):
        # Call 1: tries with response_format -> gets 400/422 -> retries without response_format -> succeeds
        res1 = llm_judge.judge_accuracy("q1", "gt1", "ans1")
        # Call 2: memoized! Should skip response_format on first attempt (only 1 SDK call for Call 2)
        res2 = llm_judge.judge_accuracy("q2", "gt2", "ans2")

    assert res1["score"] == 1.0
    assert res2["score"] == 1.0
    assert len(sdk_calls) == 3
    assert "response_format" in sdk_calls[0]
    assert "response_format" not in sdk_calls[1]
    assert "response_format" not in sdk_calls[2]

    # Verify Harbor template parity for 400/422 downgrade + memoization
    template_path = (
        Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    )
    spec = importlib.util.spec_from_file_location(f"harbor_template_eval_{status_code}", template_path)
    assert spec and spec.loader
    eval_template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_template)

    http_requests: list[dict] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": valid_response}}]}).encode()

    def fake_urlopen(req, **_):
        payload = json.loads(req.data)
        http_requests.append(payload)
        if "response_format" in payload:
            raise urllib.error.HTTPError(
                req.full_url,
                status_code,
                "Bad Request",
                {},
                io.BytesIO(b'{"error": "json_schema not supported"}'),
            )
        return _Resp()

    monkeypatch.setattr(eval_template.urllib.request, "urlopen", fake_urlopen)
    tres1 = eval_template.judge_accuracy("q1", "gt1", "ans1")
    tres2 = eval_template.judge_accuracy("q2", "gt2", "ans2")

    assert tres1["score"] == 1.0
    assert tres2["score"] == 1.0
    assert len(http_requests) == 3
    assert "response_format" in http_requests[0]
    assert "response_format" not in http_requests[1]
    assert "response_format" not in http_requests[2]


def test_transient_429_preserves_schema_and_does_not_memoize_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify HTTP 429 retries with backoff while keeping response_format intact."""
    valid_response = json.dumps(
        {
            "criteria": {
                "SKILL_IDENTIFIED": True,
                "ACTION_CORRECT": True,
                "FACTUALLY_ACCURATE": True,
                "TASK_ADDRESSED": True,
                "ACTIONABLE": True,
            },
            "score": 1.0,
            "reason": "retried 429 with schema",
        }
    )
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0.01")

    class _RateLimitError(Exception):
        status_code = 429

    sdk_calls: list[dict] = []

    def fake_sdk_create(**kwargs):
        sdk_calls.append(kwargs)
        if len(sdk_calls) == 1:
            raise _RateLimitError("rate limited")
        return MagicMock(choices=[MagicMock(message=MagicMock(content=valid_response))])

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = fake_sdk_create

    with patch("openai.OpenAI", return_value=mock_openai):
        res = llm_judge.judge_accuracy("q", "gt", "ans")

    assert res["score"] == 1.0
    assert len(sdk_calls) == 2
    assert "response_format" in sdk_calls[0]
    assert "response_format" in sdk_calls[1]


def test_template_model_fallback_isolates_schema_memoization_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify memoizing primary-model as schema-unsupported does not strip schema from fallback-model."""
    import importlib.util
    import io
    import urllib.error
    from pathlib import Path

    valid_response = json.dumps(
        {
            "criteria": {
                "SKILL_IDENTIFIED": True,
                "ACTION_CORRECT": True,
                "FACTUALLY_ACCURATE": True,
                "TASK_ADDRESSED": True,
                "ACTIONABLE": True,
            },
            "score": 1.0,
            "reason": "fallback model with schema",
        }
    )
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "https://openai-compatible.example/v1")
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "old-primary-model")
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", "modern-fallback-model")

    template_path = (
        Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_template_eval_fallback_iso", template_path)
    assert spec and spec.loader
    eval_template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_template)

    http_requests: list[dict] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": valid_response}}]}).encode()

    def fake_urlopen(req, **_):
        payload = json.loads(req.data)
        http_requests.append(payload)
        if payload["model"] == "old-primary-model":
            if "response_format" in payload:
                raise urllib.error.HTTPError(
                    req.full_url,
                    400,
                    "Bad Request",
                    {},
                    io.BytesIO(b'{"error": "json_schema not supported"}'),
                )
            raise urllib.error.HTTPError(
                req.full_url,
                404,
                "Not Found",
                {},
                io.BytesIO(b'{"error": "model not found"}'),
            )
        return _Resp()

    monkeypatch.setattr(eval_template.urllib.request, "urlopen", fake_urlopen)
    res = eval_template.judge_accuracy("q", "gt", "ans")
    assert res["score"] == 1.0
    assert [r["model"] for r in http_requests] == [
        "old-primary-model",
        "old-primary-model",
        "modern-fallback-model",
    ]
    assert "response_format" in http_requests[0]
    assert "response_format" not in http_requests[1]
    assert "response_format" in http_requests[2]


def test_reasoning_token_exhaustion_missing_message_retries_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify missing 'message' or message=None on finish_reason='length' is treated as empty content."""
    import importlib.util
    from pathlib import Path

    valid_response = json.dumps(
        {
            "criteria": {
                "SKILL_IDENTIFIED": True,
                "ACTION_CORRECT": True,
                "FACTUALLY_ACCURATE": True,
                "TASK_ADDRESSED": True,
                "ACTIONABLE": True,
            },
            "score": 1.0,
            "reason": "recovered from missing message",
        }
    )
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=None)]),
        MagicMock(choices=[MagicMock(message=MagicMock(content=valid_response))]),
    ]
    with patch("openai.OpenAI", return_value=mock_openai):
        res = llm_judge.judge_accuracy("q", "gt", "ans")
    assert res["score"] == 1.0

    template_path = (
        Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_template_eval_missing_msg", template_path)
    assert spec and spec.loader
    eval_template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_template)

    responses = iter(
        [
            {"choices": [{"finish_reason": "length", "index": 0, "message": None}]},
            {"choices": [{"finish_reason": "stop", "index": 0, "message": {"content": valid_response}}]},
        ]
    )

    class _Resp:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps(self._data).encode()

    monkeypatch.setattr(eval_template.urllib.request, "urlopen", lambda _req, **_: _Resp(next(responses)))
    tres = eval_template.judge_accuracy("q", "gt", "ans")
    assert tres["score"] == 1.0


def test_persistent_400_after_schema_downgrade_fails_fast_without_infinite_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify if downgraded prompt-only call also returns HTTP 400, it fails fast after 2 calls."""
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret-key-9999999")
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "broken-model")

    class _BadRequestError(Exception):
        status_code = 400

    sdk_calls: list[dict] = []

    def fake_sdk_create(**kwargs):
        sdk_calls.append(kwargs)
        raise _BadRequestError("HTTP 400 echoed test-secret-key-9999999")

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = fake_sdk_create

    with patch("openai.OpenAI", return_value=mock_openai):
        res = llm_judge.judge_accuracy("q", "gt", "ans")

    assert res["score"] is None
    assert res["status"] == "error"
    assert "test-secret-key-9999999" not in res["reason"]
    assert "[REDACTED]" in res["reason"]
    assert len(sdk_calls) == 2
    assert "response_format" in sdk_calls[0]
    assert "response_format" not in sdk_calls[1]


def test_anthropic_downgrades_on_400_and_memoizes_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Anthropic provider downgrades output_config on HTTP 400 and memoizes across calls."""
    import importlib.util
    import io
    import urllib.error
    from pathlib import Path

    valid_response = json.dumps(
        {
            "criteria": {
                "SKILL_IDENTIFIED": True,
                "ACTION_CORRECT": True,
                "FACTUALLY_ACCURATE": True,
                "TASK_ADDRESSED": True,
                "ACTIONABLE": True,
            },
            "score": 1.0,
            "reason": "anthropic downgraded ok",
        }
    )
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "claude-3-haiku-20240307")

    class _BadRequestError(Exception):
        status_code = 400

    anth_calls: list[dict] = []

    def fake_messages_create(**kwargs):
        anth_calls.append(kwargs)
        if "output_config" in kwargs:
            raise _BadRequestError("output_config: Extra inputs are not permitted")
        block = MagicMock()
        block.type = "text"
        block.text = valid_response
        return MagicMock(content=[block])

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = fake_messages_create

    with patch("anthropic.Anthropic", return_value=mock_anthropic):
        res1 = llm_judge.judge_accuracy("q1", "gt1", "ans1")
        res2 = llm_judge.judge_accuracy("q2", "gt2", "ans2")

    assert res1["score"] == 1.0
    assert res2["score"] == 1.0
    assert len(anth_calls) == 3
    assert "output_config" in anth_calls[0]
    assert "output_config" not in anth_calls[1]
    assert "output_config" not in anth_calls[2]

    template_path = (
        Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_template_eval_anth_400", template_path)
    assert spec and spec.loader
    eval_template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_template)

    http_requests: list[dict] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return json.dumps({"content": [{"type": "text", "text": valid_response}]}).encode()

    def fake_urlopen(req, timeout=90):
        payload = json.loads(req.data)
        http_requests.append(payload)
        if "output_config" in payload:
            raise urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                {},
                io.BytesIO(b'{"error": "output_config not supported"}'),
            )
        return _Resp()

    monkeypatch.setattr(eval_template.urllib.request, "urlopen", fake_urlopen)
    tres1 = eval_template.judge_accuracy("q1", "gt1", "ans1")
    tres2 = eval_template.judge_accuracy("q2", "gt2", "ans2")

    assert tres1["score"] == 1.0
    assert tres2["score"] == 1.0
    assert len(http_requests) == 3
    assert "output_config" in http_requests[0]
    assert "output_config" not in http_requests[1]
    assert "output_config" not in http_requests[2]


def test_judge_schema_not_spoofed_by_transcript_containing_accuracy_tokens(monkeypatch):
    """Ensure judge_goal_accuracy and judge_behavior_check pass explicit schemas and are not spoofed by SKILL_IDENTIFIED/ACTION_CORRECT in the agent transcript."""
    import sys
    from types import SimpleNamespace

    from skillevaluator.inference import client as client_mod
    from skillevaluator.tier3.eval_core import llm_judge

    client_mod._SCHEMA_UNSUPPORTED_TARGETS.clear()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "google/gemini-3.8-flash")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123456")

    captured_kwargs: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            schema_name = kwargs.get("response_format", {}).get("json_schema", {}).get("name")
            if schema_name == "goal_accuracy_judgment":
                content = '{"user_goal": "g", "end_state": "e", "achieved": true, "score": 1.0, "reason": "ok"}'
            else:
                content = '{"results": [{"step": 1, "passed": true, "reason": "ok"}], "score": 1.0, "summary": "ok"}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    adversarial_transcript = "Checked SKILL_IDENTIFIED and ACTION_CORRECT in llm_judge.py"
    goal_res = llm_judge.judge_goal_accuracy("goal", "expected", adversarial_transcript)
    beh_res = llm_judge.judge_behavior_check(adversarial_transcript, ["rule"])

    assert goal_res["score"] == 1.0
    assert beh_res["score"] == 1.0
    assert captured_kwargs[0]["response_format"]["json_schema"]["name"] == "goal_accuracy_judgment"
    assert captured_kwargs[1]["response_format"]["json_schema"]["name"] == "behavior_check_judgment"


def test_schema_builders_and_downgrade_warning_log(monkeypatch, caplog):
    """Verify _build_openai_response_format, _build_anthropic_output_config, and warning log on 400 downgrade."""
    import logging
    import sys
    from types import SimpleNamespace

    from skillevaluator.inference import client as client_mod
    from skillevaluator.tier3.eval_core import llm_judge

    assert client_mod._build_openai_response_format({"type": "object"}, "my_schema") == {
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "strict": True,
            "schema": {"type": "object"},
        },
    }
    assert client_mod._build_anthropic_output_config({"type": "object"}) == {
        "format": {
            "type": "json_schema",
            "schema": {"type": "object"},
        }
    }

    client_mod._SCHEMA_UNSUPPORTED_TARGETS.clear()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "legacy/model-400")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123456")

    valid_response = json.dumps(
        {
            "criteria": {
                "SKILL_IDENTIFIED": True,
                "ACTION_CORRECT": True,
                "FACTUALLY_ACCURATE": True,
                "TASK_ADDRESSED": True,
                "ACTIONABLE": True,
            },
            "score": 1.0,
            "reason": "ok",
        }
    )

    class FakeBadRequestError(Exception):
        def __init__(self, message: str):
            super().__init__(message)
            self.status_code = 400

    class FakeCompletions:
        def create(self, **kwargs):
            if "response_format" in kwargs:
                raise FakeBadRequestError("400 response_format not supported")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=valid_response))])

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    with caplog.at_level(logging.WARNING):
        res = llm_judge.judge_accuracy("q", "gt", "ans")

    assert res["score"] == 1.0
    assert any("Structured output schema unsupported" in rec.message for rec in caplog.records)
