# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide zero-dependency exponential backoff with full jitter and header-aware retry logic."""

from __future__ import annotations

import logging
import math
import os
import random
import time
import urllib.error
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BASE_DELAY: float = 1.0
DEFAULT_MAX_DELAY: float = 30.0
_RETRIABLE_HTTP_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES_ENV: str = "SKILL_EVAL_LLM_MAX_RETRIES"
_BASE_DELAY_ENV: str = "SKILL_EVAL_LLM_RETRY_BASE_DELAY"
_MAX_DELAY_ENV: str = "SKILL_EVAL_LLM_RETRY_MAX_DELAY"


@dataclass(frozen=True)
class RetryConfig:
    """Hold configuration parameters for retry and exponential backoff."""

    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY


def is_retriable_status_code(status_code: int | None) -> bool:
    """Return True if the given HTTP status code represents a transient, retriable condition."""
    return status_code in _RETRIABLE_HTTP_STATUS_CODES


def parse_retry_after(header_value: str | None, fallback_delay: float) -> float:
    """Parse a Retry-After header as seconds or HTTP date, falling back to default."""
    if not header_value:
        return fallback_delay
    clean_val = str(header_value).strip()
    try:
        return max(0.0, float(clean_val))
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(clean_val)
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        return max(0.0, (target - now).total_seconds())
    except Exception:
        return fallback_delay


def calculate_full_jitter_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    factor: float = 2.0,
) -> float:
    """Calculate delay using exponential backoff with full jitter (AWS standard)."""
    backoff = min(max_delay, base_delay * (factor**attempt))
    return random.uniform(0.0, backoff)


def _int_env(environ: Mapping[str, str], name: str, fallback: int) -> int:
    """Read a non-negative integer from the environment variable or return default."""
    raw = str(environ.get(name, "")).strip()
    if raw:
        try:
            val = int(raw)
            return val if val >= 0 else fallback
        except ValueError:
            return fallback
    return fallback


def _float_env(environ: Mapping[str, str], name: str, fallback: float) -> float:
    """Read a non-negative float from the environment variable or return default."""
    raw = str(environ.get(name, "")).strip()
    if raw:
        try:
            val = float(raw)
            return val if math.isfinite(val) and val >= 0.0 else fallback
        except ValueError:
            return fallback
    return fallback


def resolve_retry_config(environ: Mapping[str, str] | None = None) -> RetryConfig:
    """Resolve retry and backoff limits from environment variables with safe defaults."""
    env = os.environ if environ is None else environ
    max_retries = _int_env(env, _MAX_RETRIES_ENV, fallback=DEFAULT_MAX_RETRIES)
    base_delay = _float_env(env, _BASE_DELAY_ENV, fallback=DEFAULT_BASE_DELAY)
    raw_max_delay = _float_env(env, _MAX_DELAY_ENV, fallback=DEFAULT_MAX_DELAY)
    max_delay = max(base_delay, raw_max_delay)
    return RetryConfig(max_retries=max_retries, base_delay=base_delay, max_delay=max_delay)


def extract_http_status(exc: BaseException) -> int | None:
    """Extract numeric HTTP status code from an exception if available."""
    if isinstance(exc, urllib.error.HTTPError):
        return getattr(exc, "code", None)
    if hasattr(exc, "status_code") and isinstance(exc.status_code, int):
        return exc.status_code
    if hasattr(exc, "status") and isinstance(exc.status, int):
        return exc.status
    return None


def extract_retry_after(exc: BaseException) -> str | None:
    """Extract Retry-After header string from an exception if present."""
    if isinstance(exc, urllib.error.HTTPError) and exc.headers:
        return exc.headers.get("retry-after") or exc.headers.get("Retry-After")
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "headers") and response.headers:
        return response.headers.get("retry-after") or response.headers.get("Retry-After")
    return None


def is_retriable_exception(exc: BaseException) -> bool:
    """Return True if an exception represents a transient failure eligible for retry."""
    status = extract_http_status(exc)
    if status is not None:
        return is_retriable_status_code(status)

    if isinstance(exc, (TimeoutError, urllib.error.URLError, ConnectionError, OSError)):
        return True

    type_name = type(exc).__name__
    return any(keyword in type_name for keyword in ("RateLimit", "APIConnection", "Timeout", "InternalServer"))


def retry_call_with_backoff[T](
    operation: Callable[[], T],
    *,
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    on_retry: Callable[[BaseException, int, float], None] | None = None,
) -> T:
    """Execute operation, retrying on transient HTTP or network faults using full jitter."""
    cfg = resolve_retry_config()
    retries_limit = cfg.max_retries if max_retries is None else max(0, max_retries)
    ceiling_delay = cfg.max_delay if max_delay is None else max(0.0, max_delay)
    initial_delay = min(ceiling_delay, cfg.base_delay if base_delay is None else max(0.0, base_delay))

    attempt = 0

    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt >= retries_limit or not is_retriable_exception(exc):
                raise

            retry_after_str = extract_retry_after(exc)
            if retry_after_str is not None:
                parsed_delay = parse_retry_after(retry_after_str, fallback_delay=initial_delay)
                if parsed_delay > ceiling_delay:
                    raise
                delay = parsed_delay + random.uniform(0.1, 0.5)
            else:
                delay = calculate_full_jitter_delay(
                    attempt=attempt,
                    base_delay=initial_delay,
                    max_delay=ceiling_delay,
                )

            sleep_duration = min(delay, ceiling_delay)

            if on_retry is not None:
                on_retry(exc, attempt + 1, sleep_duration)
            else:
                status = extract_http_status(exc)
                status_label = f"HTTP {status}" if status is not None else type(exc).__name__
                logger.warning(
                    "LLM call encountered transient error (%s). Retrying in %.2fs (attempt %d/%d)...",
                    status_label,
                    sleep_duration,
                    attempt + 1,
                    retries_limit,
                )

            time.sleep(sleep_duration)
            attempt += 1
