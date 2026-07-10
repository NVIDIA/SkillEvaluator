# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light authenticated model-catalog discovery."""

from __future__ import annotations

import ipaddress
import json
import math
import time
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from http.client import HTTPException
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

if TYPE_CHECKING:
    from skillevaluator.provider_config import ProviderConfig

_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_CATALOG_PAGES = 100
_MAX_MODEL_ID_LENGTH = 512
_MAX_MODEL_RECORDS = 10_000
_NON_CHAT_MARKERS = (
    "dall-e",
    "embedding",
    "embed",
    "flux",
    "image",
    "moderation",
    "rerank",
    "speech",
    "text-to-image",
    "text-to-speech",
    "transcribe",
    "tts",
    "whisper",
)


class ModelCatalogError(RuntimeError):
    """Safe-to-display catalog error without response or credential content."""


class _RejectRedirects(HTTPRedirectHandler):
    """Never forward an authenticated catalog request across a redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        return None


def _urlopen_without_redirects(request: Request, *, timeout: float):
    parsed = urlsplit(request.full_url)
    handlers: list[Any] = [_RejectRedirects()]
    if parsed.hostname and _is_loopback_host(parsed.hostname):
        handlers.insert(0, ProxyHandler({}))
    return build_opener(*handlers).open(request, timeout=timeout)


# Module seam for bounded request-shape tests.
urlopen = _urlopen_without_redirects


@dataclass(frozen=True)
class ModelRecord:
    """One normalized model returned by a provider catalog."""

    id: str
    created: int | None = None


@dataclass(frozen=True)
class CatalogModel:
    """One filtered catalog entry selected for display."""

    id: str
    created: int | None
    is_configured: bool


def fetch_model_records(config: ProviderConfig, timeout_seconds: float = 15.0) -> tuple[ModelRecord, ...]:
    """Fetch and normalize the selected provider's authenticated ``/models`` catalog."""
    _validate_timeout(timeout_seconds)
    url, headers = _request_settings(config)
    records: list[ModelRecord] = []
    seen: set[str] = set()
    seen_cursors: set[str] = set()
    next_url = url
    remaining_bytes = _MAX_RESPONSE_BYTES
    deadline = time.monotonic() + timeout_seconds

    for page_number in range(1, _MAX_CATALOG_PAGES + 1):
        request_timeout = timeout_seconds if page_number == 1 else deadline - time.monotonic()
        if request_timeout <= 0:
            raise ModelCatalogError("model catalog request timed out")
        payload, response_bytes = _request_json(
            next_url,
            headers=headers,
            timeout_seconds=request_timeout,
            max_response_bytes=remaining_bytes,
        )
        remaining_bytes -= response_bytes
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ModelCatalogError("model catalog response has no data list")

        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str):
                continue
            normalized_id = model_id.strip()
            if (
                not normalized_id
                or len(normalized_id) > _MAX_MODEL_ID_LENGTH
                or _has_unsafe_control(normalized_id)
                or normalized_id in seen
            ):
                continue
            if len(records) >= _MAX_MODEL_RECORDS:
                raise ModelCatalogError("model catalog exceeded the safe record limit")
            created = item.get("created")
            normalized_created = (
                created if isinstance(created, int) and not isinstance(created, bool) and created >= 0 else None
            )
            records.append(ModelRecord(id=normalized_id, created=normalized_created))
            seen.add(normalized_id)

        if config.provider != "anthropic" or payload.get("has_more") is not True:
            break
        cursor = payload.get("last_id")
        if (
            not isinstance(cursor, str)
            or not cursor.strip()
            or len(cursor) > _MAX_MODEL_ID_LENGTH
            or _has_unsafe_control(cursor)
            or cursor in seen_cursors
        ):
            raise ModelCatalogError("model catalog pagination cursor is invalid")
        if remaining_bytes <= 0:
            raise ModelCatalogError("model catalog response exceeded the safe size limit")
        seen_cursors.add(cursor)
        next_url = f"{url}?{urlencode({'after_id': cursor})}"
    else:
        raise ModelCatalogError("model catalog exceeded the safe pagination limit")

    if data and not records:
        raise ModelCatalogError("model catalog response did not contain valid model records")
    return tuple(records)


def select_catalog_models(
    config: ProviderConfig,
    records: Iterable[ModelRecord],
    *,
    limit: int = 10,
) -> tuple[CatalogModel, ...]:
    """Return a bounded filtered view with the configured model first."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("model catalog selection limit must be a positive integer")

    unique: dict[str, ModelRecord] = {}
    for record in records:
        if isinstance(record, ModelRecord) and _is_chat_candidate(record.id):
            unique.setdefault(record.id, record)

    ranked = sorted(unique.values(), key=lambda record: record.id != config.model)
    return tuple(
        CatalogModel(
            id=record.id,
            created=record.created,
            is_configured=record.id == config.model,
        )
        for record in ranked[:limit]
    )


def _request_settings(config: ProviderConfig) -> tuple[str, dict[str, str]]:
    if config.provider == "bedrock":
        raise ModelCatalogError("bedrock does not expose this HTTP catalog; use skillevaluator doctor --verify-models")
    if config.provider not in {"nv_build", "openai", "openai-compatible", "anthropic"}:
        raise ModelCatalogError(f"{config.provider} does not expose a supported HTTP model catalog")

    api_key = config.api_key
    if not isinstance(api_key, str) or not api_key.strip():
        credential = config.credential_env or "provider API key"
        raise ModelCatalogError(f"{credential} is required for authenticated model discovery")
    if _has_unsafe_control(api_key):
        credential = config.credential_env or "provider API key"
        raise ModelCatalogError(f"{credential} contains invalid control characters")

    if config.provider == "anthropic":
        base_url = config.base_url or _ANTHROPIC_BASE_URL
        return _provider_url(base_url, ensure_v1=True), {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    if not config.base_url:
        raise ModelCatalogError(f"{config.provider} does not expose an HTTP model catalog")
    return _provider_url(config.base_url), {"Authorization": f"Bearer {api_key}"}


def _provider_url(base_url: str, *, ensure_v1: bool = False) -> str:
    if not isinstance(base_url, str) or not base_url or base_url != base_url.strip():
        raise ModelCatalogError("model catalog base URL is invalid")
    if _has_unsafe_control(base_url) or "\\" in base_url:
        raise ModelCatalogError("model catalog base URL is invalid")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        raise ModelCatalogError("model catalog base URL is invalid") from None

    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname
    if scheme not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ModelCatalogError("model catalog base URL must be absolute HTTP or HTTPS")
    if (
        parsed.username is not None
        or parsed.password is not None
        or "?" in base_url
        or "#" in base_url
        or parsed.netloc.endswith(":")
    ):
        raise ModelCatalogError("model catalog base URL must not contain credentials, query, or fragment")
    if ";" in parsed.path:
        raise ModelCatalogError("model catalog base URL path is invalid")
    if scheme == "http" and not _is_loopback_host(hostname):
        raise ModelCatalogError("model catalog base URL must use HTTPS unless it targets loopback")

    host = f"[{hostname}]" if ":" in hostname else hostname
    authority = f"{host}:{port}" if port is not None else host
    path = parsed.path.rstrip("/")
    if ensure_v1 and not path.endswith("/v1"):
        path = f"{path}/v1"
    return f"{scheme}://{authority}{path}/models"


def _request_json(
    url: str,
    *,
    headers: Mapping[str, str],
    timeout_seconds: float,
    max_response_bytes: int,
) -> tuple[Any, int]:
    request_headers = {
        **headers,
        "Accept": "application/json",
        "User-Agent": "SkillEvaluator/model-catalog",
    }
    try:
        request = Request(url, headers=request_headers, method="GET")
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310 - validated above
            raw = response.read(max_response_bytes + 1)
    except HTTPError as exc:
        raise ModelCatalogError(f"model catalog returned HTTP {exc.code}") from None
    except (HTTPException, TimeoutError, URLError, OSError) as exc:
        raise ModelCatalogError(f"model catalog request failed: {type(exc).__name__}") from None
    except (TypeError, ValueError):
        raise ModelCatalogError("model catalog request configuration is invalid") from None

    if len(raw) > max_response_bytes:
        raise ModelCatalogError("model catalog response exceeded the safe size limit")
    try:
        return json.loads(raw), len(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ModelCatalogError("model catalog returned invalid JSON") from None


def _validate_timeout(timeout_seconds: float) -> None:
    if (
        not isinstance(timeout_seconds, int | float)
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ModelCatalogError("model catalog timeout must be a positive number")


def _has_unsafe_control(value: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value)


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _is_chat_candidate(model_id: str) -> bool:
    lowered = model_id.casefold()
    return bool(model_id.strip()) and not any(marker in lowered for marker in _NON_CHAT_MARKERS)
