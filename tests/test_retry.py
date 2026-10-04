# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for zero-dependency retry and rate limit utilities."""

from __future__ import annotations

import urllib.error
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.message import Message
from email.utils import format_datetime
from typing import Any

import pytest
from tests.conftest import load_harbor_eval_template

from skillevaluator.inference.retry import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    calculate_full_jitter_delay,
    is_retriable_status_code,
    parse_retry_after,
    resolve_retry_config,
    retry_call_with_backoff,
)


def _http_headers(retry_after: str | None = None) -> Message:
    """Build an HTTP header Message for urllib.error.HTTPError tests."""
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return headers


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (408, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (520, True),
        (529, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (501, False),
        (505, False),
        (200, False),
        (None, False),
    ],
)
def test_is_retriable_status_code(code: int | None, expected: bool) -> None:
    """Verify is_retriable_status_code correctly classifies transient status codes."""
    assert is_retriable_status_code(code) is expected


def test_host_and_verifier_retry_the_same_http_statuses() -> None:
    """Verify the host client and the Harbor verifier retry exactly the same HTTP statuses."""
    verifier = load_harbor_eval_template("harbor_template_retry_status_parity")
    host = {code for code in range(100, 600) if is_retriable_status_code(code)}
    template = {code for code in range(100, 600) if verifier._is_retriable_http_status(code)}
    assert host == template


@pytest.mark.parametrize(
    ("header_val", "fallback", "expected"),
    [
        ("10", 1.0, 10.0),
        ("0", 1.0, 0.0),
        ("2.5", 1.0, 2.5),
        ("-5", 1.0, 0.0),
        ("", 3.0, 3.0),
        (None, 4.0, 4.0),
        ("invalid", 2.0, 2.0),
    ],
)
def test_parse_retry_after_numeric_and_fallback(
    header_val: str | None,
    fallback: float,
    expected: float,
) -> None:
    """Verify parse_retry_after handles numeric strings, negatives, and fallbacks."""
    assert parse_retry_after(header_val, fallback) == pytest.approx(expected)


def test_parse_retry_after_http_date_future() -> None:
    """Verify parse_retry_after handles RFC-7231 HTTP dates in UTC."""
    future_time = datetime.now(UTC) + timedelta(seconds=15)
    formatted = format_datetime(future_time, usegmt=True)

    result = parse_retry_after(formatted, fallback_delay=1.0)
    assert 13.0 <= result <= 16.0


def test_parse_retry_after_http_date_past() -> None:
    """Verify parse_retry_after clamps past dates to 0.0."""
    past_time = datetime.now(UTC) - timedelta(seconds=15)
    formatted = format_datetime(past_time, usegmt=True)

    result = parse_retry_after(formatted, fallback_delay=1.0)
    assert result == 0.0


def test_parse_retry_after_naive_date() -> None:
    """Verify parse_retry_after correctly handles offset-naive parsed datetimes without raising."""
    result = parse_retry_after("Sun, 06 Nov 1994 08:49:37", fallback_delay=5.0)
    assert result == 0.0


@pytest.mark.parametrize("attempt", [0, 1, 2, 3, 5, 10])
def test_calculate_full_jitter_delay_bounds(attempt: int) -> None:
    """Verify calculate_full_jitter_delay generates non-negative values bounded by backoff."""
    base_delay = 1.0
    max_delay = 30.0
    expected_ceiling = min(max_delay, base_delay * (2.0**attempt))

    for _ in range(25):
        delay = calculate_full_jitter_delay(attempt, base_delay=base_delay, max_delay=max_delay)
        assert 0.0 <= delay <= expected_ceiling


def test_resolve_retry_config_defaults() -> None:
    """Verify resolve_retry_config returns sensible defaults when environment is empty."""
    config = resolve_retry_config({})
    assert config.max_retries == DEFAULT_MAX_RETRIES
    assert config.base_delay == DEFAULT_BASE_DELAY
    assert config.max_delay == DEFAULT_MAX_DELAY


def test_resolve_retry_config_custom_env() -> None:
    """Verify resolve_retry_config accepts standard SKILL_EVAL_LLM_* environment variables."""
    env = {
        "SKILL_EVAL_LLM_MAX_RETRIES": "5",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY": "2.5",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY": "45.0",
    }
    config = resolve_retry_config(env)
    assert config.max_retries == 5
    assert config.base_delay == 2.5
    assert config.max_delay == 45.0


def test_resolve_retry_config_defensive_fallbacks() -> None:
    """Verify resolve_retry_config safely falls back to defaults when values are malformed."""
    env = {
        "SKILL_EVAL_LLM_MAX_RETRIES": "invalid",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY": "-1.0",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY": "not-a-float",
    }
    config = resolve_retry_config(env)
    assert config.max_retries == DEFAULT_MAX_RETRIES
    assert config.base_delay == DEFAULT_BASE_DELAY
    assert config.max_delay == DEFAULT_MAX_DELAY


def test_resolve_retry_config_negative_values_fallback() -> None:
    """Verify resolve_retry_config falls back to defaults when values are negative."""
    env = {
        "SKILL_EVAL_LLM_MAX_RETRIES": "-5",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY": "-2.0",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY": "-10.0",
    }
    config = resolve_retry_config(env)
    assert config.max_retries == DEFAULT_MAX_RETRIES
    assert config.base_delay == DEFAULT_BASE_DELAY
    assert config.max_delay == DEFAULT_MAX_DELAY


@pytest.mark.parametrize("variable", ["SKILL_EVAL_LLM_RETRY_BASE_DELAY", "SKILL_EVAL_LLM_RETRY_MAX_DELAY"])
def test_nonfinite_retry_delay_falls_back_in_host_and_verifier(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    """Verify non-finite retry delay configuration falls back to defaults in host and verifier."""
    config = resolve_retry_config({variable: "inf"})
    assert config.base_delay == DEFAULT_BASE_DELAY
    assert config.max_delay == DEFAULT_MAX_DELAY

    verifier = load_harbor_eval_template("harbor_template_retry_nonfinite")
    monkeypatch.setenv(variable, "inf")
    assert verifier._resolve_eval_retry_config() == (
        verifier._DEFAULT_MAX_RETRIES,
        verifier._DEFAULT_BASE_DELAY,
        verifier._DEFAULT_MAX_DELAY,
    )


def test_verifier_preserves_rate_limit_error_body_when_retry_after_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the terminal 429 report retains its provider diagnostic body when Retry-After exceeds max_delay."""
    import io

    verifier = load_harbor_eval_template("harbor_template_retry_after_diagnostic")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "30")

    def rate_limited(request: Any, timeout: float = 90) -> Any:
        _ = timeout
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            _http_headers("3600"),
            io.BytesIO(b'{"error":"capacity returns in one hour"}'),
        )

    monkeypatch.setattr(verifier.urllib.request, "urlopen", rate_limited)
    request = verifier.urllib.request.Request("https://example.test/v1/chat/completions")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        verifier._urlopen_with_retry(request)

    assert "capacity returns in one hour" in verifier._format_http_error(exc_info.value)


def test_retry_call_with_backoff_immediate_success() -> None:
    """Verify retry_call_with_backoff executes operation and returns immediately on success."""
    calls = 0

    def op() -> str:
        nonlocal calls
        calls += 1
        return "success"

    result = retry_call_with_backoff(op)
    assert result == "success"
    assert calls == 1


def test_retry_call_with_backoff_succeeds_after_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify retry_call_with_backoff retries transient 429 and succeeds on subsequent attempt."""
    calls = 0
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)

    def op() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("http://api", 429, "Too Many Requests", hdrs=_http_headers(), fp=None)
        return "recovered"

    result = retry_call_with_backoff(op, max_retries=3, base_delay=1.0)
    assert result == "recovered"
    assert calls == 2
    assert len(slept) == 1
    assert slept[0] >= 0.0


def test_retry_call_with_backoff_respects_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify retry_call_with_backoff uses Retry-After header when present on HTTPError."""
    calls = 0
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)

    def op() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                "http://api",
                429,
                "Too Many Requests",
                hdrs=_http_headers("3.5"),
                fp=None,
            )
        return "done"

    result = retry_call_with_backoff(op, max_retries=2, base_delay=0.1)
    assert result == "done"
    assert calls == 2
    assert len(slept) == 1
    # 3.5 + small jitter buffer (+0.1-0.5)
    assert 3.5 <= slept[0] <= 4.1


def test_retry_call_with_backoff_fails_fast_on_non_retriable_error() -> None:
    """Verify retry_call_with_backoff raises immediately without retrying on non-retriable error."""
    calls = 0

    def op() -> str:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError("http://api", 401, "Unauthorized", hdrs=_http_headers(), fp=None)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        retry_call_with_backoff(op, max_retries=3)

    assert exc_info.value.code == 401
    assert calls == 1


def test_retry_call_with_backoff_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify retry_call_with_backoff exhausts retries and raises on persistent 429."""
    calls = 0
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)

    def op() -> str:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError("http://api", 429, "Too Many Requests", hdrs=_http_headers(), fp=None)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        retry_call_with_backoff(op, max_retries=3)

    assert exc_info.value.code == 429
    assert calls == 4  # 1 initial + 3 retries
    assert len(slept) == 3


def test_retry_call_with_backoff_caps_delay_at_max_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify retry_call_with_backoff caps single attempt sleep duration to max_delay."""
    calls = 0
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)

    def op() -> str:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise urllib.error.HTTPError("http://api", 503, "Service Unavailable", hdrs=_http_headers(), fp=None)
        return "finished"

    # With base_delay=100.0 and max_delay=5.0, delay is capped to 5.0
    result = retry_call_with_backoff(op, max_retries=3, base_delay=100.0, max_delay=5.0)
    assert result == "finished"
    assert calls == 3
    assert len(slept) == 2
    for duration in slept:
        assert duration <= 5.0


def test_retry_call_with_backoff_raises_if_retry_after_exceeds_max_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify retry_call_with_backoff raises immediately without sleeping if Retry-After exceeds max_delay."""
    calls = 0
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)

    def op() -> str:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            "http://api",
            429,
            "Too Many Requests",
            hdrs=_http_headers("3600"),
            fp=None,
        )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        retry_call_with_backoff(op, max_retries=3, max_delay=30.0)

    assert exc_info.value.code == 429
    assert calls == 1
    assert len(slept) == 0


def test_retry_call_with_backoff_invokes_on_retry_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify retry_call_with_backoff invokes on_retry callback with attempt and sleep duration."""
    calls = 0
    callback_calls: list[tuple[int, float]] = []
    monkeypatch.setattr("time.sleep", lambda _: None)

    def on_retry_cb(exc: BaseException, attempt: int, delay: float) -> None:
        _ = exc
        callback_calls.append((attempt, delay))

    def op() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("http://api", 503, "Service Unavailable", hdrs=_http_headers(), fp=None)
        return "ok"

    result = retry_call_with_backoff(op, max_retries=2, base_delay=0.1, on_retry=on_retry_cb)
    assert result == "ok"
    assert calls == 2
    assert len(callback_calls) == 1
    assert callback_calls[0][0] == 1
    assert callback_calls[0][1] >= 0.0


class _BedrockTransportHarness:
    """Test harness for driving _call_bedrock against botocore's URLLib3Session.send."""

    def __init__(self, verifier: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        import botocore.httpsession

        self.verifier = verifier
        self.monkeypatch = monkeypatch
        self.slept: list[float] = []
        self.requests_sent = 0
        self._handler: Callable[[Any], Any] | None = None
        monkeypatch.setattr(verifier.time, "sleep", self.slept.append)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
        monkeypatch.setenv("AWS_REGION", "us-west-2")

        def _send(_session: Any, request: Any) -> Any:
            self.requests_sent += 1
            assert self._handler is not None, "Configure harness handler before calling call_bedrock()"
            return self._handler(request)

        monkeypatch.setattr(botocore.httpsession.URLLib3Session, "send", _send)

    def set_handler(self, handler: Callable[[Any], Any]) -> None:
        """Register a per-request transport handler."""
        self._handler = handler

    @staticmethod
    def json_response(
        request: Any,
        status_code: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Build a botocore AWSResponse with a JSON body."""
        import json

        import botocore.awsrequest

        merged_headers = {"Content-Type": "application/json", **(headers or {})}
        resp = botocore.awsrequest.AWSResponse(request.url, status_code, merged_headers, None)
        resp._content = json.dumps(payload).encode("utf-8")
        return resp

    def throttled_response(self, request: Any, *, retry_after: str | None = None) -> Any:
        """Build an HTTP 429 ThrottlingException response."""
        headers = {"x-amzn-errortype": "ThrottlingException"}
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        return self.json_response(request, 429, {"message": "Rate exceeded"}, headers)

    def ok_response(self, request: Any, text: str = "ok") -> Any:
        """Build an HTTP 200 Converse output response."""
        return self.json_response(
            request,
            200,
            {"output": {"message": {"content": [{"text": text}]}}},
        )

    def call_bedrock(self, prompt: str = "Judge this") -> tuple[str | None, str | None]:
        """Invoke verifier._call_bedrock with default test model parameters."""
        return self.verifier._call_bedrock(
            prompt,
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            512,
            0.0,
        )


@pytest.fixture
def bedrock_harness(monkeypatch: pytest.MonkeyPatch) -> _BedrockTransportHarness:
    """Provide an isolated verifier template module wired to a botocore transport harness."""
    verifier = load_harbor_eval_template("harbor_template_bedrock_harness")
    return _BedrockTransportHarness(verifier, monkeypatch)


def test_bedrock_converse_transport_zero_retries_overrides_aws_max_attempts(
    bedrock_harness: _BedrockTransportHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _call_bedrock with SKILL_EVAL_LLM_MAX_RETRIES=0 sends only 1 request even when AWS_MAX_ATTEMPTS=4."""
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "0")
    bedrock_harness.set_handler(
        lambda req: (
            bedrock_harness.throttled_response(req)
            if bedrock_harness.requests_sent == 1
            else bedrock_harness.ok_response(req, "recovered")
        )
    )

    content, error = bedrock_harness.call_bedrock()
    assert content is None
    assert error is not None and "Bedrock request failed" in error
    assert bedrock_harness.requests_sent == 1
    assert bedrock_harness.slept == []


def test_bedrock_converse_transport_positive_retries_overrides_aws_max_attempts_one(
    bedrock_harness: _BedrockTransportHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _call_bedrock with SKILL_EVAL_LLM_MAX_RETRIES=3 retries 429 and succeeds even when AWS_MAX_ATTEMPTS=1."""
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0.05")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "0.2")
    bedrock_harness.set_handler(
        lambda req: (
            bedrock_harness.throttled_response(req)
            if bedrock_harness.requests_sent <= 2
            else bedrock_harness.ok_response(req, "recovered verdict")
        )
    )

    content, error = bedrock_harness.call_bedrock()
    assert (content, error) == ("recovered verdict", None)
    assert bedrock_harness.requests_sent == 3
    assert len(bedrock_harness.slept) == 2
    assert all(0.0 <= d <= 0.2 for d in bedrock_harness.slept)


def test_bedrock_converse_transport_exhausts_configured_retries_on_persistent_429(
    bedrock_harness: _BedrockTransportHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _call_bedrock with SKILL_EVAL_LLM_MAX_RETRIES=3 sends 4 total requests on persistent 429."""
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0.01")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "0.1")
    bedrock_harness.set_handler(bedrock_harness.throttled_response)

    content, error = bedrock_harness.call_bedrock()
    assert content is None
    assert error is not None and "Bedrock request failed" in error
    assert bedrock_harness.requests_sent == 4
    assert len(bedrock_harness.slept) == 3


def test_bedrock_converse_transport_fails_fast_on_non_retriable_400(
    bedrock_harness: _BedrockTransportHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _call_bedrock does not retry non-retriable HTTP 400 ValidationException."""
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "3")
    bedrock_harness.set_handler(
        lambda req: bedrock_harness.json_response(
            req,
            400,
            {"message": "Malformed input request"},
            {"x-amzn-errortype": "ValidationException"},
        )
    )

    content, error = bedrock_harness.call_bedrock()
    assert content is None
    assert error is not None and "Bedrock request failed" in error
    assert bedrock_harness.requests_sent == 1
    assert bedrock_harness.slept == []


def test_bedrock_converse_transport_respects_retry_after_and_max_delay_cap(
    bedrock_harness: _BedrockTransportHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _call_bedrock honors Retry-After within max_delay and aborts immediately when Retry-After exceeds max_delay."""
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0.1")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "5.0")

    retry_after_header = "2"
    bedrock_harness.set_handler(
        lambda req: (
            bedrock_harness.throttled_response(req, retry_after=retry_after_header)
            if bedrock_harness.requests_sent == 1
            else bedrock_harness.ok_response(req, "ok")
        )
    )

    content, error = bedrock_harness.call_bedrock()
    assert (content, error) == ("ok", None)
    assert bedrock_harness.requests_sent == 2
    assert len(bedrock_harness.slept) == 1
    assert 2.0 <= bedrock_harness.slept[0] <= 2.6

    # Exceed max_delay (3600s > 5.0s): must abort after 1 request without sleeping
    bedrock_harness.requests_sent = 0
    bedrock_harness.slept.clear()
    retry_after_header = "3600"
    content, error = bedrock_harness.call_bedrock()
    assert content is None
    assert error is not None and "Bedrock request failed" in error
    assert bedrock_harness.requests_sent == 1
    assert bedrock_harness.slept == []


def test_bedrock_converse_transport_retries_transient_connection_error(
    bedrock_harness: _BedrockTransportHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _call_bedrock retries transient botocore EndpointConnectionError up to SKILL_EVAL_LLM_MAX_RETRIES."""
    import botocore.exceptions

    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0.05")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "0.2")

    def _conn_then_ok(req: Any) -> Any:
        if bedrock_harness.requests_sent == 1:
            raise botocore.exceptions.EndpointConnectionError(endpoint_url=req.url)
        return bedrock_harness.ok_response(req, "reconnected")

    bedrock_harness.set_handler(_conn_then_ok)

    content, error = bedrock_harness.call_bedrock()
    assert (content, error) == ("reconnected", None)
    assert bedrock_harness.requests_sent == 2
    assert len(bedrock_harness.slept) == 1


@pytest.mark.parametrize("exc_cls", [FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError])
def test_bedrock_converse_does_not_retry_filesystem_os_error(
    bedrock_harness: _BedrockTransportHarness,
    monkeypatch: pytest.MonkeyPatch,
    exc_cls: type[OSError],
) -> None:
    """Verify _call_bedrock fails fast without retrying when local credential file resolution raises filesystem OSError."""
    from unittest.mock import MagicMock

    import boto3

    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "3")
    calls = 0
    client = MagicMock()

    def raise_fs_error(**_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise exc_cls("missing or unreadable AWS credential file")

    client.converse.side_effect = raise_fs_error
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)

    content, error = bedrock_harness.call_bedrock()
    assert content is None
    assert error is not None and "Bedrock request failed" in error
    assert calls == 1
    assert bedrock_harness.slept == []


def test_bedrock_converse_aborts_when_judge_deadline_exhausted_before_retry(
    bedrock_harness: _BedrockTransportHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _call_bedrock aborts without sleeping when _ACTIVE_JUDGE_DEADLINE is exceeded by the retry backoff."""
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "1.0")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "30.0")
    bedrock_harness.set_handler(lambda req: bedrock_harness.throttled_response(req, retry_after="5"))

    verifier = bedrock_harness.verifier
    token = verifier._ACTIVE_JUDGE_DEADLINE.set(verifier.time.monotonic() + 0.5)
    try:
        content, error = bedrock_harness.call_bedrock()
    finally:
        verifier._ACTIVE_JUDGE_DEADLINE.reset(token)

    assert content is None
    assert error is not None and "LLM judge time budget exhausted before retry" in error
    assert bedrock_harness.requests_sent == 1
    assert bedrock_harness.slept == []


@pytest.mark.parametrize(
    ("raw_val", "expected_budget"),
    [
        (None, 180.0),
        ("", 180.0),
        ("45", 45.0),
        ("12.5", 12.5),
        ("0", 180.0),
        ("-10", 180.0),
        ("nan", 180.0),
        ("inf", 180.0),
        ("-inf", 180.0),
        ("not-a-number", 180.0),
    ],
)
def test_resolve_judge_wall_time_budget_from_env(
    harbor_eval_template: Any,
    monkeypatch: pytest.MonkeyPatch,
    raw_val: str | None,
    expected_budget: float,
) -> None:
    """Verify _resolve_judge_wall_time_budget parses positive finite seconds and falls back to 180.0 on invalid inputs."""
    if raw_val is None:
        monkeypatch.delenv("SKILL_EVAL_LLM_JUDGE_BUDGET_SEC", raising=False)
    else:
        monkeypatch.setenv("SKILL_EVAL_LLM_JUDGE_BUDGET_SEC", raw_val)

    assert harbor_eval_template._resolve_judge_wall_time_budget() == pytest.approx(expected_budget)


def test_call_required_judge_enforces_skill_eval_llm_judge_budget_sec(
    harbor_eval_template: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _call_required_judge applies the SKILL_EVAL_LLM_JUDGE_BUDGET_SEC deadline to judge execution."""
    monkeypatch.setenv("SKILL_EVAL_LLM_JUDGE_BUDGET_SEC", "42.5")
    observed_remaining: list[float] = []

    def fake_judge() -> dict[str, Any]:
        observed_remaining.append(harbor_eval_template._remaining_judge_timeout(90.0))
        return {"score": 1.0, "reason": "ok"}

    result = harbor_eval_template._call_required_judge("accuracy", fake_judge)
    assert result["score"] == 1.0
    assert len(observed_remaining) == 1
    assert 40.0 <= observed_remaining[0] <= 42.5
