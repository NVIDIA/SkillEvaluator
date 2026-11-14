# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in judge policies: LLM_JUDGE_FALLBACK_POLICY and LLM_JUDGE_RETRY_POLICY."""

from __future__ import annotations

import io
import json
import ssl
import time
import urllib.error
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.conftest import load_harbor_eval_template

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.adapter import _verifier_env_block

PRIMARY = "family/primary"
ALLOWLIST = "family/backup,family/last,other/forbidden"
POLICY_VARS = ("LLM_JUDGE_FALLBACK_POLICY", "LLM_JUDGE_RETRY_POLICY")
GOAL_REPLY = json.dumps({"user_goal": "g", "end_state": "e", "achieved": False, "reason": "missing"})


class _Clock:
    """Stand-in for the verifier's ``time`` module with a manual monotonic clock."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def verifier(monkeypatch):
    module = load_harbor_eval_template("judge_policies_verifier")
    for name in (
        *POLICY_VARS,
        "LLM_JUDGE_FALLBACK_MODELS",
        "LLM_JUDGE_MODEL",
        "SKILL_EVAL_JUDGE_MODEL",
        "SKILL_EVAL_LLM_MODEL",
        "SKILL_EVAL_JUDGE_CHARACTER_FACTS",
        "SKILL_EVAL_LLM_JUDGE_BUDGET_SEC",
        "SKILL_EVAL_LLM_MAX_RETRIES",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY",
        "SKILL_EVAL_LLM_CREDENTIAL_SOURCE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "NVIDIA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "judge-test-key")
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "https://judge.example/v1")
    monkeypatch.setattr(module, "time", _Clock())
    return module


def strict(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_POLICY", "replace")
    monkeypatch.setenv("LLM_JUDGE_RETRY_POLICY", "strict")
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", ALLOWLIST)
    monkeypatch.setenv("LLM_JUDGE_MODEL", PRIMARY)


def completion(content, **choice):
    message = {"content": content}
    if "refusal" in choice:
        message["refusal"] = choice.pop("refusal")
    return io.BytesIO(json.dumps({"choices": [{"message": message, **choice}]}).encode())


def http_error(code=429, body=b'{"error":{"code":"capacity_exceeded"}}', headers=None):
    return urllib.error.HTTPError("https://judge.example/v1", code, "test", headers or {}, io.BytesIO(body))


def replay(verifier, monkeypatch, outcomes):
    """Answer each judge request with the next outcome and return the requested models."""
    requested = []
    pending = iter(outcomes)

    def open_request(request, timeout):
        requested.append(json.loads(request.data)["model"])
        outcome = next(pending)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(verifier.urllib.request, "urlopen", open_request)
    return requested


def goal_judge(verifier):
    return verifier._call_required_judge("goal_accuracy", verifier._judge_goal_accuracy_custom, "q", "truth", "e", "")


# ---------------------------------------------------------------------------
# Fallback policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "answered_by"),
    [("replace", "family/backup"), ("append", "us.anthropic.claude-opus-4-8")],
)
def test_refused_judge_is_answered_from_the_explicit_list_only_under_replace(
    verifier, monkeypatch, policy, answered_by
):
    import boto3

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_POLICY", policy)
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", "family/backup")
    client = MagicMock()
    client.converse.side_effect = [
        {"stopReason": "content_filtered", "output": {"message": {"content": []}}},
        {"stopReason": "end_turn", "output": {"message": {"content": [{"text": "judged"}]}}},
    ]
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)

    content, error, provenance = verifier._call_public_llm_with_provenance(
        "prompt", model="us.anthropic.claude-opus-5-5"
    )

    assert (content, error) == ("judged", None)
    assert provenance["model"] == answered_by
    assert [call.kwargs["modelId"] for call in client.converse.call_args_list] == [
        "us.anthropic.claude-opus-5-5",
        answered_by,
    ]


def _anthropic_request(verifier, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "judge-test-key")
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL")
    sent = []

    def open_request(request, timeout):
        sent.append(request)
        body = {"content": [{"type": "text", "text": "Done"}], "stop_reason": "end_turn"}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(verifier.urllib.request, "urlopen", open_request)
    assert verifier._call_anthropic("prompt", "claude-opus-5-5", 4096, 0.0) == ("Done", None)
    return sent[0]


@pytest.mark.parametrize(
    ("name", "value"), [("LLM_JUDGE_FALLBACK_POLICY", "replace"), ("LLM_JUDGE_RETRY_POLICY", "strict")]
)
def test_policy_turns_off_the_builtin_anthropic_refusal_fallback(verifier, monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    request = _anthropic_request(verifier, monkeypatch)

    assert "fallbacks" not in json.loads(request.data)
    assert request.get_header("Anthropic-beta") is None


@pytest.mark.parametrize(("fallback_policy", "retry_policy"), [(None, None), (" APPEND ", "Legacy")])
def test_default_policies_keep_the_builtin_anthropic_refusal_fallback(
    verifier, monkeypatch, fallback_policy, retry_policy
):
    for name, value in zip(POLICY_VARS, (fallback_policy, retry_policy), strict=True):
        if value is not None:
            monkeypatch.setenv(name, value)

    request = _anthropic_request(verifier, monkeypatch)

    assert json.loads(request.data)["fallbacks"] == [{"model": "claude-opus-4-8"}]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, [{"model": "claude-opus-4-8"}]),
        ("", [{"model": "claude-opus-4-8"}]),
        ("claude-opus-5", [{"model": "claude-opus-5"}]),
        ("none", None),
        ("Off", None),
        ("claude-opus-5-5", None),
    ],
)
def test_refusal_fallback_model_env_for_the_anthropic_judge(verifier, monkeypatch, value, expected):
    monkeypatch.delenv("SKILL_EVAL_LLM_REFUSAL_FALLBACK_MODEL", raising=False)
    if value is not None:
        monkeypatch.setenv("SKILL_EVAL_LLM_REFUSAL_FALLBACK_MODEL", value)

    request = _anthropic_request(verifier, monkeypatch)

    payload = json.loads(request.data)
    assert payload.get("fallbacks") == expected
    assert (request.get_header("Anthropic-beta") is not None) == (expected is not None)


def test_refusal_fallback_model_env_for_the_bedrock_judge(verifier, monkeypatch):
    import boto3

    monkeypatch.setenv("SKILL_EVAL_LLM_REFUSAL_FALLBACK_MODEL", "claude-opus-5")
    client = MagicMock()
    client.converse.side_effect = [
        {"stopReason": "content_filtered", "output": {"message": {"content": []}}},
        {"stopReason": "end_turn", "output": {"message": {"content": [{"text": "Done"}]}}},
    ]
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)

    verifier._call_bedrock("prompt", "us.anthropic.claude-opus-5-5", 4096, 0.0)

    assert [call.kwargs["modelId"] for call in client.converse.call_args_list] == [
        "us.anthropic.claude-opus-5-5",
        "us.anthropic.claude-opus-5",
    ]

    monkeypatch.setenv("SKILL_EVAL_LLM_REFUSAL_FALLBACK_MODEL", "none")
    client.converse.reset_mock()
    client.converse.side_effect = [{"stopReason": "content_filtered", "output": {"message": {"content": []}}}]
    with pytest.raises(verifier._JudgeRefusedError):
        verifier._call_bedrock("prompt", "us.anthropic.claude-opus-5-5", 4096, 0.0)
    assert client.converse.call_count == 1


def test_bedrock_refusal_fallback_equal_to_the_request_model_is_not_sent(verifier, monkeypatch):
    import boto3

    monkeypatch.setenv("SKILL_EVAL_LLM_REFUSAL_FALLBACK_MODEL", "claude-opus-5-5")
    client = MagicMock()
    client.converse.side_effect = [
        {"stopReason": "content_filtered", "output": {"message": {"content": []}}},
        {"stopReason": "content_filtered", "output": {"message": {"content": []}}},
    ]
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)

    with pytest.raises(verifier._JudgeRefusedError):
        verifier._call_bedrock("prompt", "us.anthropic.claude-opus-5-5", 4096, 0.0)

    assert [call.kwargs["modelId"] for call in client.converse.call_args_list] == ["us.anthropic.claude-opus-5-5"]


@pytest.mark.parametrize(
    ("value", "expected_calls"),
    [
        ("claude-opus-4-8", ["us.anthropic.claude-opus-5-5", "us.anthropic.claude-opus-4-8"]),
        ("us.anthropic.claude-opus-4-8", ["us.anthropic.claude-opus-5-5", "us.anthropic.claude-opus-4-8"]),
        ("global.anthropic.claude-opus-4-8", ["us.anthropic.claude-opus-5-5", "global.anthropic.claude-opus-4-8"]),
        ("anthropic.claude-opus-4-8", ["us.anthropic.claude-opus-5-5", "anthropic.claude-opus-4-8"]),
        ("us.anthropic.claude-opus-5-5", ["us.anthropic.claude-opus-5-5"]),
    ],
)
def test_bedrock_refusal_fallback_uses_a_qualified_bedrock_id_as_given(verifier, monkeypatch, value, expected_calls):
    import boto3

    monkeypatch.setenv("SKILL_EVAL_LLM_REFUSAL_FALLBACK_MODEL", value)
    client = MagicMock()
    client.converse.side_effect = [
        {"stopReason": "content_filtered", "output": {"message": {"content": []}}},
        {"stopReason": "end_turn", "output": {"message": {"content": [{"text": "Done"}]}}},
    ]
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)

    if len(expected_calls) == 1:
        with pytest.raises(verifier._JudgeRefusedError):
            verifier._call_bedrock("prompt", "us.anthropic.claude-opus-5-5", 4096, 0.0)
    else:
        assert verifier._call_bedrock("prompt", "us.anthropic.claude-opus-5-5", 4096, 0.0) == ("Done", None)

    assert [call.kwargs["modelId"] for call in client.converse.call_args_list] == expected_calls


@pytest.mark.parametrize(
    ("name", "value"), [("LLM_JUDGE_FALLBACK_POLICY", "replace"), ("LLM_JUDGE_RETRY_POLICY", "strict")]
)
def test_policy_turns_off_the_builtin_bedrock_refusal_fallback(verifier, monkeypatch, name, value):
    import boto3

    monkeypatch.setenv(name, value)
    client = MagicMock()
    client.converse.side_effect = [
        {"stopReason": "content_filtered", "output": {"message": {"content": []}}},
        {"stopReason": "end_turn", "output": {"message": {"content": [{"text": "Done"}]}}},
    ]
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)

    with pytest.raises(verifier._JudgeRefusedError):
        verifier._call_bedrock("prompt", "us.anthropic.claude-opus-5-5", 4096, 0.0)

    assert [call.kwargs["modelId"] for call in client.converse.call_args_list] == ["us.anthropic.claude-opus-5-5"]


# ---------------------------------------------------------------------------
# Strict retry policy
# ---------------------------------------------------------------------------


def test_strict_retries_capacity_once_then_records_the_model_that_answered(verifier, monkeypatch):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [http_error(), http_error(), completion(GOAL_REPLY)])

    result = goal_judge(verifier)

    assert requested == [PRIMARY, PRIMARY, "family/backup"]
    assert result["score"] == 0.0
    assert result["judge_model_used"] == "family/backup"
    assert result["judge_fallback_used"] is True


def test_strict_tries_at_most_three_models_twice_each(verifier, monkeypatch):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [http_error() for _ in range(8)])

    result = goal_judge(verifier)

    assert requested == [PRIMARY] * 2 + ["family/backup"] * 2 + ["family/last"] * 2
    assert result["score"] is None
    assert result["status"] == "error"
    assert "fallback exhausted" in result["reason"]


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_strict_availability_moves_on_only_after_one_retry(verifier, monkeypatch, status):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [http_error(status), http_error(status), completion("judged")])

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert requested == [PRIMARY, PRIMARY, "family/backup"]


def test_strict_connection_failure_moves_on_after_one_retry(verifier, monkeypatch):
    strict(monkeypatch)
    refused = urllib.error.URLError(ConnectionRefusedError("connection refused"))
    requested = replay(verifier, monkeypatch, [refused, refused, completion("judged")])

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert requested == [PRIMARY, PRIMARY, "family/backup"]


def test_strict_certificate_failure_ends_the_judge(verifier, monkeypatch):
    strict(monkeypatch)
    failure = urllib.error.URLError(ssl.SSLCertVerificationError("certificate verify failed"))
    requested = replay(verifier, monkeypatch, [failure, completion("judged")])

    content, error = verifier.call_public_llm("prompt")

    assert content is None and "certificate verify failed" in error
    assert requested == [PRIMARY]


def test_strict_model_access_denial_moves_to_the_explicit_list(verifier, monkeypatch):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [http_error(403, b"key_model_access_denied"), completion("judged")])

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert requested == [PRIMARY, "family/backup"]


@pytest.mark.parametrize(
    ("code", "body"),
    [
        (400, b"content_filter"),
        (401, b"unauthorized"),
        (403, b"policy rejection"),
        (409, b"conflict"),
        (429, b'{"error":{"code":"insufficient_quota"}}'),
        (429, b"key budget exhausted"),
        (501, b"not implemented"),
        (503, b"blocked by safety system"),
    ],
)
def test_strict_rejection_is_not_retried_or_handed_on(verifier, monkeypatch, code, body):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [http_error(code, body), completion("judged")])

    content, error = verifier.call_public_llm("prompt")

    assert content is None and f"HTTP {code}" in error
    assert requested == [PRIMARY]


def test_strict_stop_keeps_the_errors_of_earlier_models(verifier, monkeypatch):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [http_error(503), http_error(503), http_error(401, b"bad key")])

    content, error = verifier.call_public_llm("prompt")

    assert content is None
    assert error.startswith(f"{PRIMARY}: HTTP 503")
    assert error.endswith("family/backup: HTTP 401: test - bad key")
    assert requested == [PRIMARY, PRIMARY, "family/backup"]


def test_strict_reads_stop_words_only_at_the_start_of_an_error_page(verifier, monkeypatch):
    strict(monkeypatch)
    page = b"<html><title>Error 502</title>" + b" " * 600 + b"Performance and security by a proxy</html>"
    requested = replay(verifier, monkeypatch, [http_error(502, page), http_error(502, page), completion("judged")])

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert requested == [PRIMARY, PRIMARY, "family/backup"]


OPENAI_MODEL_NOT_FOUND = (
    b'{"error":{"message":"The model `family/primary` does not exist or you do not have access to it.",'
    b'"type":"invalid_request_error","param":null,"code":"model_not_found"}}'
)


@pytest.mark.parametrize(
    "body",
    [
        OPENAI_MODEL_NOT_FOUND,
        b'{"error":{"message":"The model `family/primary` does not exist.","type":"invalid_request_error"}}',
        b'{"type":"error","error":{"type":"not_found_error","message":"model: family/primary"}}',
    ],
)
def test_strict_missing_model_moves_to_the_explicit_list_without_a_retry(verifier, monkeypatch, body):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [http_error(404, body), completion("judged")])

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert requested == [PRIMARY, "family/backup"]


def test_strict_other_not_found_error_ends_the_judge(verifier, monkeypatch):
    strict(monkeypatch)
    body = b'{"type":"error","error":{"type":"not_found_error","message":"Not found"}}'
    requested = replay(verifier, monkeypatch, [http_error(404, body), completion("judged")])

    content, error = verifier.call_public_llm("prompt")

    assert content is None and "HTTP 404" in error
    assert requested == [PRIMARY]


def test_legacy_policy_keeps_its_model_error_rules(verifier, monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", ALLOWLIST)
    requested = replay(verifier, monkeypatch, [http_error(404, OPENAI_MODEL_NOT_FOUND), completion("judged")])

    content, error = verifier.call_public_llm("prompt", model=PRIMARY)

    assert content is None and "HTTP 404" in error
    assert requested == [PRIMARY]


def _bedrock_error(code, status, message):
    error = RuntimeError(message)
    error.response = {"Error": {"Code": code, "Message": message}, "ResponseMetadata": {"HTTPStatusCode": status}}
    return error


def _bedrock_client(monkeypatch, outcomes):
    import boto3

    client = MagicMock()
    client.converse.side_effect = outcomes
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)
    return client


@pytest.mark.parametrize(
    ("code", "status", "message"),
    [
        ("AccessDeniedException", 403, "You don't have access to the model with the specified model ID."),
        ("ResourceNotFoundException", 404, "Model not found."),
        ("ValidationException", 400, "The provided model identifier is invalid."),
    ],
)
def test_strict_bedrock_model_unavailable_moves_on_without_a_retry(verifier, monkeypatch, code, status, message):
    strict(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
    answered = {"stopReason": "end_turn", "output": {"message": {"content": [{"text": "judged"}]}}}
    client = _bedrock_client(monkeypatch, [_bedrock_error(code, status, message), answered])

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert [call.kwargs["modelId"] for call in client.converse.call_args_list] == [PRIMARY, "family/backup"]


def test_strict_bedrock_other_validation_error_ends_the_judge(verifier, monkeypatch):
    strict(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
    invalid = _bedrock_error("ValidationException", 400, "maxTokens: must be at most 8192")
    client = _bedrock_client(monkeypatch, [invalid, {"stopReason": "end_turn"}])

    content, error = verifier.call_public_llm("prompt")

    assert content is None and "maxTokens" in error
    assert [call.kwargs["modelId"] for call in client.converse.call_args_list] == [PRIMARY]


def test_strict_bedrock_stop_keeps_the_errors_of_earlier_models(verifier, monkeypatch):
    strict(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
    throttled = _bedrock_error("ThrottlingException", 429, "Too many requests")
    invalid = _bedrock_error("ValidationException", 400, "maxTokens: must be at most 8192")
    _bedrock_client(monkeypatch, [throttled, throttled, invalid])

    content, error = verifier.call_public_llm("prompt")

    assert content is None
    assert error == (
        f"{PRIMARY}: Bedrock request failed: Too many requests | "
        "family/backup: Bedrock request failed: maxTokens: must be at most 8192"
    )


def _bedrock_endpoint_error():
    from botocore.exceptions import EndpointConnectionError

    return EndpointConnectionError(endpoint_url="https://bedrock-runtime.example/model/converse")


@pytest.mark.parametrize(
    ("make_error", "requests_per_model"),
    [
        (lambda: _bedrock_error("ResourceNotFoundException", 404, "Model not found."), 1),
        (lambda: _bedrock_error("ThrottlingException", 429, "Too many requests"), 2),
        (_bedrock_endpoint_error, 2),
    ],
    ids=["model-not-found", "throttled", "endpoint-unreachable"],
)
def test_strict_bedrock_errors_that_move_on_keep_the_bedrock_label(
    verifier, monkeypatch, make_error, requests_per_model
):
    strict(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", "family/backup")
    error = make_error()
    client = _bedrock_client(monkeypatch, [error] * (2 * requests_per_model))

    content, detail = verifier.call_public_llm("prompt")

    assert content is None
    assert detail == (
        f"LLM judge model fallback exhausted: {PRIMARY}: Bedrock request failed: {error} | "
        f"family/backup: Bedrock request failed: {error}"
    )
    assert [call.kwargs["modelId"] for call in client.converse.call_args_list] == (
        [PRIMARY] * requests_per_model + ["family/backup"] * requests_per_model
    )


def test_strict_stop_words_in_an_echoed_model_id_do_not_end_the_judge(verifier, monkeypatch):
    strict(monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_MODEL", "vendor/content-safety-judge")
    missing = http_error(404, b'{"error":{"message":"invalid model: vendor/content-safety-judge"}}')
    requested = replay(verifier, monkeypatch, [missing, completion("judged")])

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert requested == ["vendor/content-safety-judge", "family/backup"]


def test_strict_stop_words_in_a_connection_error_do_not_end_the_judge(verifier, monkeypatch):
    strict(monkeypatch)
    tunnel = urllib.error.URLError(OSError("Tunnel connection failed: 503 security-proxy.example unavailable"))
    requested = replay(verifier, monkeypatch, [tunnel, tunnel, completion("judged")])

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert requested == [PRIMARY, PRIMARY, "family/backup"]


@pytest.mark.parametrize(
    "body",
    [
        b'{"error":{"code":"budget_exceeded","message":"Budget has been exceeded"}}',
        b'{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low"}}',
        b'{"error":{"code":"billing_hard_limit_reached","message":"Billing hard limit has been reached"}}',
        b'{"error":{"code":"content_filter","message":"The prompt was filtered"}}',
        b'{"detail":"Budget has been exceeded for this key"}',
        b'{"error":{"status":429},"note":"insufficient_quota"}',
    ],
)
def test_strict_structured_stop_errors_end_the_judge(verifier, monkeypatch, body):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [http_error(429, body), completion("judged")])

    content, error = verifier.call_public_llm("prompt")

    assert content is None and "HTTP 429" in error
    assert requested == [PRIMARY]


@pytest.mark.parametrize(
    "reply",
    [
        completion("ignored", refusal="I can't help with that."),
        completion("", finish_reason="content_filter"),
    ],
)
def test_strict_refusal_never_switches_models(verifier, monkeypatch, reply):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [reply, completion("judged")])

    content, error = verifier.call_public_llm("prompt")

    assert content is None and "declined the request" in error
    assert requested == [PRIMARY]


def test_strict_empty_reply_is_final(verifier, monkeypatch):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [completion(None), completion(GOAL_REPLY)])

    result = goal_judge(verifier)

    assert result["status"] == "error"
    assert "LLM_JUDGE_RETRY_POLICY=strict" in result["reason"]
    assert requested == [PRIMARY]


def test_strict_candidates_share_the_judge_time_budget(verifier, monkeypatch):
    strict(monkeypatch)
    clock = verifier.time
    requested = []

    def unavailable(request, timeout):
        requested.append(json.loads(request.data)["model"])
        clock.now += 100
        raise http_error()

    monkeypatch.setattr(verifier.urllib.request, "urlopen", unavailable)
    token = verifier._ACTIVE_JUDGE_DEADLINE.set(110.0)
    try:
        content, error = verifier.call_public_llm("prompt")
    finally:
        verifier._ACTIVE_JUDGE_DEADLINE.reset(token)

    assert content is None and "time budget exhausted" in error
    assert requested == [PRIMARY, PRIMARY]


def test_strict_retry_after_never_waits_past_the_judge_budget(verifier, monkeypatch):
    strict(monkeypatch)
    clock = verifier.time
    requested = replay(verifier, monkeypatch, [http_error(headers={"Retry-After": "20"}), completion("judged")])
    token = verifier._ACTIVE_JUDGE_DEADLINE.set(10.0)
    try:
        content, error = verifier.call_public_llm("prompt")
    finally:
        verifier._ACTIVE_JUDGE_DEADLINE.reset(token)

    assert content is None and "time budget exhausted" in error
    assert requested == [PRIMARY]
    assert clock.sleeps == []


def test_strict_retry_after_above_the_maximum_delay_moves_on_without_waiting(verifier, monkeypatch):
    strict(monkeypatch)
    clock = verifier.time
    requested = replay(verifier, monkeypatch, [http_error(headers={"Retry-After": "600"}), completion("judged")])

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert requested == [PRIMARY, "family/backup"]
    assert clock.sleeps == []


def test_strict_schema_downgrade_counts_toward_the_request_limit(verifier, monkeypatch):
    strict(monkeypatch)
    schema_rejected = http_error(400, b'{"error":{"message":"response_format is not supported"}}')
    requested = replay(verifier, monkeypatch, [http_error(), schema_rejected, completion("judged")])

    assert verifier.call_public_llm("prompt", response_schema={"type": "object"}) == ("judged", None)
    assert requested == [PRIMARY, PRIMARY, "family/backup"]


@pytest.mark.parametrize(
    "reply",
    [
        "not JSON",
        '{"achieved":"false"}',
        '{"achieved":false,"reason":7}',
        '{"achieved":true,"reason":"example"}\n{"achieved":false,"reason":"actual judgment"}',
        '{"achieved":true,"reason":"example"} trailing prose',
        'I cannot judge this request. The requested schema is {"achieved":true,"reason":"example"}.',
        'Refused.\n```json\n{"achieved":true,"reason":"example"}\n```',
        '```json\n{"achieved":true,"reason":"example"}\n```',
        '{"achieved":false,"achieved":true,"reason":"contradictory"}',
        '{"achieved":true,"reason":"example","invalid":NaN}',
    ],
)
def test_strict_completed_reply_that_is_not_one_valid_object_is_final(verifier, monkeypatch, reply):
    strict(monkeypatch)
    requested = replay(verifier, monkeypatch, [completion(reply), completion('{"achieved":true}')])

    result = goal_judge(verifier)

    assert result["score"] is None
    assert result["status"] == "error"
    assert "LLM_JUDGE_RETRY_POLICY=strict sends no retry" in result["reason"]
    assert requested == [PRIMARY]


@pytest.mark.parametrize("achieved", [False, True])
def test_strict_whole_object_keeps_the_completed_verdict(verifier, monkeypatch, achieved):
    strict(monkeypatch)
    reply = " \n" + json.dumps({"achieved": achieved, "reason": "actual judgment"}) + "\n "
    requested = replay(verifier, monkeypatch, [completion(reply), completion('{"achieved":true}')])

    result = goal_judge(verifier)

    assert result["score"] == float(achieved)
    assert result["reason"] == "actual judgment"
    assert requested == [PRIMARY]


def test_legacy_json_extraction_remains_permissive(verifier):
    assert verifier.extract_json('explanation: {"achieved":true} trailing prose') == {"achieved": True}


def test_strict_behavior_check_neither_retries_nor_salvages(verifier, monkeypatch):
    strict(monkeypatch)
    truncated = '{"results": [{"step": 1, "passed": true, "reason": "seen"}, {"step": 2, "pas'
    requested = replay(verifier, monkeypatch, [completion(truncated), completion(truncated)])

    result = verifier._call_required_judge(
        "behavior_check", verifier.judge_behavior_check, "conversation", ["first", "second"]
    )

    assert result["score"] is None
    assert result["status"] == "error"
    assert "LLM_JUDGE_RETRY_POLICY=strict" in result["reason"]
    assert requested == [PRIMARY]


def test_strict_goal_judge_skips_ragas(verifier, monkeypatch):
    strict(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "judge-test-key")
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-4o")
    monkeypatch.setattr(
        verifier, "_judge_goal_accuracy_ragas", lambda *_args: pytest.fail("RAGAS sends its own judge requests")
    )
    requested = replay(verifier, monkeypatch, [completion(GOAL_REPLY)])

    assert verifier.judge_goal_accuracy("q", "truth", "e")["score"] == 0.0
    assert requested == ["gpt-4o"]


def test_strict_bedrock_throttling_moves_on_after_one_retry(verifier, monkeypatch):
    import boto3

    strict(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
    throttled = RuntimeError("throttled")
    throttled.response = {"Error": {"Code": "ThrottlingException"}, "ResponseMetadata": {"HTTPStatusCode": 429}}
    client = MagicMock()
    client.converse.side_effect = [
        throttled,
        throttled,
        {"stopReason": "end_turn", "output": {"message": {"content": [{"text": "judged"}]}}},
    ]
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)

    assert verifier.call_public_llm("prompt") == ("judged", None)
    assert [call.kwargs["modelId"] for call in client.converse.call_args_list] == [PRIMARY, PRIMARY, "family/backup"]


@pytest.mark.parametrize(
    ("name", "value"), [("LLM_JUDGE_RETRY_POLICY", "typo"), ("LLM_JUDGE_FALLBACK_POLICY", "prepend")]
)
def test_invalid_policy_fails_closed_before_any_request(verifier, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        verifier.urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("invalid policy sent a request")
    )

    content, error = verifier.call_public_llm("prompt")
    result = goal_judge(verifier)

    assert content is None and error.startswith(f"{name} must be ")
    assert result["status"] == "error"
    assert name in result["reason"]


@pytest.mark.parametrize(
    ("name", "value"), [("LLM_JUDGE_FALLBACK_POLICY", "replce"), ("LLM_JUDGE_RETRY_POLICY", "strcit")]
)
def test_invalid_policy_keeps_the_ragas_goal_scorer_off(verifier, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "judge-test-key")
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-4o")
    monkeypatch.setattr(
        verifier, "_judge_goal_accuracy_ragas", lambda *_args: pytest.fail("RAGAS sent its own judge request")
    )
    monkeypatch.setattr(
        verifier.urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("invalid policy sent a request")
    )

    result = verifier._call_required_judge("goal_accuracy", verifier.judge_goal_accuracy, "q", "truth", "e", "")

    assert verifier._ragas_goal_accuracy_enabled() is False
    assert result["status"] == "error"
    assert name in result["reason"]


def test_default_policies_keep_legacy_retries_and_refusal_fallback(verifier, monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", ALLOWLIST)
    requested = replay(verifier, monkeypatch, [completion("ignored", refusal="no"), *[http_error() for _ in range(4)]])

    content, error = verifier.call_public_llm("prompt", model=PRIMARY)

    assert content is None and "HTTP 429" in error
    assert requested == [PRIMARY] + ["family/backup"] * 4


# ---------------------------------------------------------------------------
# Host forwarding
# ---------------------------------------------------------------------------


def test_task_verifier_env_forwards_the_judge_policies():
    env = _verifier_env_block({"LLM_JUDGE_FALLBACK_POLICY": "replace", "LLM_JUDGE_RETRY_POLICY": "strict"})

    assert 'LLM_JUDGE_FALLBACK_POLICY = "${LLM_JUDGE_FALLBACK_POLICY}"' in env
    assert 'LLM_JUDGE_RETRY_POLICY = "${LLM_JUDGE_RETRY_POLICY}"' in env


def test_host_judge_policies_reach_only_the_standard_verifier_job(monkeypatch):
    monkeypatch.setattr(
        runner.os, "environ", {"LLM_JUDGE_FALLBACK_POLICY": " replace ", "LLM_JUDGE_RETRY_POLICY": "Strict"}
    )
    provider = ProviderConfig(
        provider="openai",
        model="test-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
    )

    provider_env = runner._provider_environment(provider)

    assert provider_env["LLM_JUDGE_FALLBACK_POLICY"] == "replace"
    assert provider_env["LLM_JUDGE_RETRY_POLICY"] == "Strict"
    assert runner._job_judge_verifier_env(provider_env, "default") == {
        "LLM_JUDGE_FALLBACK_POLICY": "${LLM_JUDGE_FALLBACK_POLICY}",
        "LLM_JUDGE_RETRY_POLICY": "${LLM_JUDGE_RETRY_POLICY}",
    }
    assert runner._job_judge_verifier_env(provider_env, "custom_only") == {}
    configuration = runner._judge_model_config(provider, provider_env, "default")
    assert configuration["fallback_policy"] == "replace"
    assert configuration["retry_policy"] == "strict"


@pytest.mark.parametrize("name", POLICY_VARS)
def test_skill_runtime_env_cannot_set_a_judge_policy(name):
    resolved, errors = runner._resolve_runtime_env({name: "legacy"}, env_mode="docker")

    assert resolved == {}
    assert errors and "host process" in errors[0]


def test_unset_policies_are_not_recorded():
    provider = SimpleNamespace(provider="openai", model="test-model")

    configuration = runner._judge_model_config(provider, {}, "default")

    assert "fallback_policy" not in configuration
    assert "retry_policy" not in configuration


@pytest.mark.parametrize(
    ("name", "value"), [("LLM_JUDGE_FALLBACK_POLICY", "replce"), ("LLM_JUDGE_RETRY_POLICY", "strcit")]
)
def test_unknown_policy_is_not_recorded_as_applied(name, value):
    provider = SimpleNamespace(provider="openai", model="test-model")

    configuration = runner._judge_model_config(provider, {name: value}, "default")

    assert "fallback_policy" not in configuration
    assert "retry_policy" not in configuration


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("LLM_JUDGE_FALLBACK_POLICY", "replce", "LLM_JUDGE_FALLBACK_POLICY must be append or replace"),
        ("LLM_JUDGE_RETRY_POLICY", "strcit", "LLM_JUDGE_RETRY_POLICY must be legacy or strict"),
    ],
)
def test_unknown_policy_stops_the_run_before_any_trial(monkeypatch, tmp_path, name, value, expected):
    monkeypatch.setenv(name, value)

    result = _run_until_staging(monkeypatch, tmp_path)

    assert result["error"] == [expected]


def test_custom_only_run_ignores_the_judge_policies(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_JUDGE_RETRY_POLICY", "strcit")

    result = _run_until_staging(monkeypatch, tmp_path, grading_mode="custom_only")

    assert not any("LLM_JUDGE_RETRY_POLICY" in error for error in result.get("error", []))


def _run_until_staging(monkeypatch, tmp_path, **engine_kwargs):
    """Run the Harbor engine with the provider, prerequisites, staging and trials stubbed out."""
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "config.yml").write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "resolve_llm_provider",
        lambda: ProviderConfig(
            provider="openai",
            model="test-model",
            api_key="provider-key",
            base_url="https://provider.example/v1",
            litellm_model="openai/test-model",
        ),
    )
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "generate_harbor_tasks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "_run_agent_pair", lambda **_kwargs: pytest.fail("trials started"))

    return runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results", env_mode="docker", **engine_kwargs)
