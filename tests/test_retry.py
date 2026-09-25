# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for zero-dependency retry and rate limit utilities."""

from __future__ import annotations

import urllib.error
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

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


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (200, False),
        (None, False),
    ],
)
def test_is_retriable_status_code(code: int | None, expected: bool) -> None:
    """Verify is_retriable_status_code correctly classifies transient status codes."""
    assert is_retriable_status_code(code) is expected


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
    """Nonfinite configuration must not become an invalid sleep duration on HTTP 429."""
    import importlib.util
    from pathlib import Path

    config = resolve_retry_config({variable: "inf"})
    assert config.base_delay == DEFAULT_BASE_DELAY
    assert config.max_delay == DEFAULT_MAX_DELAY

    template_path = (
        Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_template_retry_nonfinite", template_path)
    assert spec and spec.loader
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    monkeypatch.setenv(variable, "inf")
    assert verifier._resolve_eval_retry_config() == (
        verifier._DEFAULT_MAX_RETRIES,
        verifier._DEFAULT_BASE_DELAY,
        verifier._DEFAULT_MAX_DELAY,
    )


def test_verifier_preserves_rate_limit_error_body_when_retry_after_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The terminal 429 report must retain its provider diagnostic body."""
    import importlib.util
    import io
    from pathlib import Path

    template_path = (
        Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_template_retry_after_diagnostic", template_path)
    assert spec and spec.loader
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "30")

    def rate_limited(request, timeout=90):
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            {"Retry-After": "3600"},
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
            err = urllib.error.HTTPError("http://api", 429, "Too Many Requests", hdrs={}, fp=None)  # type: ignore[arg-type]
            raise err
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

    class MockHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            if key.lower() == "retry-after":
                return "3.5"
            return default

    def op() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            err = urllib.error.HTTPError("http://api", 429, "Too Many Requests", hdrs=MockHeaders(), fp=None)  # type: ignore[arg-type]
            raise err
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
        err = urllib.error.HTTPError("http://api", 401, "Unauthorized", hdrs={}, fp=None)  # type: ignore[arg-type]
        raise err

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
        err = urllib.error.HTTPError("http://api", 429, "Too Many Requests", hdrs={}, fp=None)  # type: ignore[arg-type]
        raise err

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
            err = urllib.error.HTTPError("http://api", 503, "Service Unavailable", hdrs={}, fp=None)  # type: ignore[arg-type]
            raise err
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

    class MockHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            if key.lower() == "retry-after":
                return "3600"
            return default

    def op() -> str:
        nonlocal calls
        calls += 1
        err = urllib.error.HTTPError("http://api", 429, "Too Many Requests", hdrs=MockHeaders(), fp=None)  # type: ignore[arg-type]
        raise err

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
        callback_calls.append((attempt, delay))

    def op() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("http://api", 503, "Service Unavailable", hdrs={}, fp=None)  # type: ignore[arg-type]
        return "ok"

    result = retry_call_with_backoff(op, max_retries=2, base_delay=0.1, on_retry=on_retry_cb)
    assert result == "ok"
    assert calls == 2
    assert len(callback_calls) == 1
    assert callback_calls[0][0] == 1
    assert callback_calls[0][1] >= 0.0
