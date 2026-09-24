# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider diagnostics expose actionable protocol fields, never raw data."""

import json

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from skillevaluator.inference import LLMClientError
from skillevaluator.inference.diagnostics import llm_failure_diagnostic, safe_llm_labels


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422, 429, 500])
def test_http_status_without_echoing_response_or_request(status):
    request = httpx.Request("POST", "https://private-token@example.invalid/v1/chat/completions")
    response = httpx.Response(status, request=request)
    exception = APIStatusError(
        "message-private-token",
        response=response,
        body={"error": {"message": "body-private-token", "code": "model_not_found", "param": "model"}},
    )
    diagnostic = llm_failure_diagnostic(exception)
    assert f"HTTP {status}" in diagnostic
    assert "code=model_not_found" in diagnostic
    assert "param=model" in diagnostic
    assert "private-token" not in diagnostic
    assert "example.invalid" not in diagnostic


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        "private-token",
        {"error": "private-token"},
        {"error": {"code": "private-token", "param": "private-token", "type": "private-token"}},
        {"error": {"code": [], "param": {}, "type": ["invalid_request_error"]}},
    ],
)
def test_unknown_or_malformed_provider_fields_are_not_echoed(body):
    request = httpx.Request("POST", "https://example.invalid")
    exception = APIStatusError("private-token", response=httpx.Response(400, request=request), body=body)
    diagnostic = llm_failure_diagnostic(exception)
    assert "HTTP 400" in diagnostic
    assert "private-token" not in diagnostic
    assert "code=" not in diagnostic
    assert "param=" not in diagnostic
    assert "type=" not in diagnostic


@pytest.mark.parametrize(
    "exception, expected",
    [
        (APIConnectionError(request=httpx.Request("POST", "https://private-token@example.invalid")), "connection"),
        (APITimeoutError(request=httpx.Request("POST", "https://private-token@example.invalid")), "timed out"),
        (LLMClientError("LLM returned invalid JSON: Raw: private-token"), "structured-output"),
        (json.JSONDecodeError("private-token", "private-token", 0), "structured-output"),
        (ValueError("private-token"), "structured-output"),
        (RuntimeError("private-token"), "Unexpected"),
    ],
)
def test_transport_and_response_errors_do_not_echo_raw_content(exception, expected):
    diagnostic = llm_failure_diagnostic(exception)
    assert expected in diagnostic
    assert "private-token" not in diagnostic


@pytest.mark.parametrize(
    "model",
    [
        "https://private-token@example.invalid",
        "secret\nnew line",
        "sk-secret",
        "nvapi-secret",
        "m" * 161,
        None,
    ],
)
def test_model_labels_reject_urls_tokens_and_control_text(model):
    assert safe_llm_labels("nv_build", model) == ("nv_build", "unresolved")


def test_model_labels_preserve_real_model_ids_and_bound_unknown_provider():
    assert safe_llm_labels("nv_build", "nvidia/nemotron-3.5-lightning-30b-a3b") == (
        "nv_build",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
    )
    assert safe_llm_labels("private-token", "anthropic.claude-model-v1:0") == (
        "unresolved",
        "anthropic.claude-model-v1:0",
    )
