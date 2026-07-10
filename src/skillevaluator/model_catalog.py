# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light authenticated model-catalog discovery."""

from __future__ import annotations

import io
import ipaddress
import json
import math
import socket
import time
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from http.client import HTTPConnection, HTTPException, HTTPResponse, HTTPSConnection
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPHandler, HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

if TYPE_CHECKING:
    from skillevaluator.provider_config import ProviderConfig

_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_CATALOG_PAGES = 100
_MAX_MODEL_ID_LENGTH = 512
_MAX_MODEL_RECORDS = 10_000
_RESPONSE_READ_CHUNK_BYTES = 64 * 1024
_DNS_RESOLVER_SLOT = BoundedSemaphore(1)
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


def _remaining_deadline(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("model catalog request timed out")
    return remaining


class _DeadlineSocketRaw(io.RawIOBase):
    """Socket reader that reapplies one absolute deadline before every recv."""

    def __init__(self, sock: socket.socket, deadline: float) -> None:
        super().__init__()
        self._socket = sock
        self._deadline = deadline
        self._raw = sock.makefile("rb", buffering=0)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int | None:
        remaining = _remaining_deadline(self._deadline)
        self._socket.settimeout(remaining)
        result = self._raw.readinto(buffer)
        _remaining_deadline(self._deadline)
        return result

    def fileno(self) -> int:
        return self._raw.fileno()

    def close(self) -> None:
        if not self.closed:
            try:
                self._raw.close()
            finally:
                super().close()


class _DeadlineResponseSocket:
    """Minimal socket facade used by ``HTTPResponse`` to build its reader."""

    def __init__(self, sock: socket.socket, deadline: float) -> None:
        self._socket = sock
        self._deadline = deadline

    def makefile(self, mode: str) -> io.BufferedReader:
        if mode != "rb":
            raise ValueError("deadline response sockets support binary reads only")
        return io.BufferedReader(_DeadlineSocketRaw(self._socket, self._deadline))


class _DeadlineHTTPResponse(HTTPResponse):
    """HTTP response whose status, headers, and body share one deadline."""

    def __init__(
        self,
        sock: socket.socket,
        debuglevel: int = 0,
        method: str | None = None,
        url: str | None = None,
        *,
        deadline: float,
    ) -> None:
        super().__init__(
            _DeadlineResponseSocket(sock, deadline),
            debuglevel=debuglevel,
            method=method,
            url=url,
        )


class _DeadlineConnectionMixin:
    """Apply one request deadline across address attempts and HTTP I/O."""

    def __init__(self, *args: Any, deadline: float, **kwargs: Any) -> None:
        self._deadline = deadline
        super().__init__(*args, **kwargs)
        self.response_class = partial(_DeadlineHTTPResponse, deadline=deadline)
        self._create_connection = self._create_connection_before_deadline

    def _remaining_timeout(self) -> float:
        return _remaining_deadline(self._deadline)

    def _set_socket_timeout(self) -> None:
        remaining = self._remaining_timeout()
        if self.sock is None:
            self.timeout = remaining
        else:
            self.sock.settimeout(remaining)

    def _create_connection_before_deadline(
        self,
        address: tuple[str, int],
        _timeout: object,
        source_address: tuple[str, int] | None,
    ) -> socket.socket:
        host, port = address
        addresses = _getaddrinfo_before_deadline(host, port, self._deadline)
        self._remaining_timeout()
        last_error: OSError | None = None
        for address_info in addresses:
            family, socktype, proto, _canonname, socket_address = address_info
            candidate: socket.socket | None = None
            try:
                candidate = socket.socket(family, socktype, proto)
                candidate.settimeout(self._remaining_timeout())
                if source_address:
                    candidate.bind(source_address)
                candidate.connect(socket_address)
                candidate.settimeout(self._remaining_timeout())
                return candidate
            except OSError as exc:
                last_error = exc
                if candidate is not None:
                    candidate.close()

        if last_error is not None:
            raise last_error
        raise OSError("getaddrinfo returned no addresses")

    def connect(self) -> None:
        self.timeout = self._remaining_timeout()
        super().connect()
        self._set_socket_timeout()

    def send(self, data: Any) -> None:
        self._set_socket_timeout()
        super().send(data)
        self._remaining_timeout()

    def getresponse(self) -> HTTPResponse:
        self._set_socket_timeout()
        response = super().getresponse()
        self._remaining_timeout()
        return response


class _DeadlineHTTPConnection(_DeadlineConnectionMixin, HTTPConnection):
    pass


def _getaddrinfo_before_deadline(host: str, port: int, deadline: float) -> list[tuple[Any, ...]]:
    """Resolve one host without allowing DNS to hold the caller past its deadline.

    ``getaddrinfo`` has no portable timeout or cancellation API. One daemon
    resolver is therefore allowed in flight globally. If an OS resolver call
    stalls, callers fail on time and later requests cannot create an unbounded
    queue or thread leak; the slot is released only when that resolver returns.
    """
    if not _DNS_RESOLVER_SLOT.acquire(timeout=_remaining_deadline(deadline)):
        raise TimeoutError("model catalog request timed out")

    outcome: Queue[Any] = Queue(maxsize=1)

    def resolve() -> None:
        try:
            outcome.put_nowait(socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM))
        except BaseException as exc:
            outcome.put_nowait(exc)
        finally:
            _DNS_RESOLVER_SLOT.release()

    worker = Thread(target=resolve, name="skillevaluator-model-catalog-dns", daemon=True)
    try:
        worker.start()
    except BaseException:
        _DNS_RESOLVER_SLOT.release()
        raise

    try:
        resolved = outcome.get(timeout=_remaining_deadline(deadline))
    except Empty:
        raise TimeoutError("model catalog request timed out") from None
    if isinstance(resolved, BaseException):
        raise resolved
    return list(resolved)


class _DeadlineHTTPSConnection(_DeadlineConnectionMixin, HTTPSConnection):
    def connect(self) -> None:
        """Refresh the deadline between proxy tunneling and the TLS handshake."""
        self.timeout = self._remaining_timeout()
        HTTPConnection.connect(self)
        self._set_socket_timeout()
        server_hostname = self._tunnel_host or self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)
        self._set_socket_timeout()


def _request_deadline(request: Request) -> float:
    deadline = getattr(request, "_skillevaluator_deadline", None)
    if isinstance(deadline, int | float) and not isinstance(deadline, bool) and math.isfinite(deadline):
        return float(deadline)
    return time.monotonic() + float(request.timeout)


class _DeadlineHTTPHandler(HTTPHandler):
    def http_open(self, request: Request):
        return self.do_open(_DeadlineHTTPConnection, request, deadline=_request_deadline(request))


class _DeadlineHTTPSHandler(HTTPSHandler):
    def https_open(self, request: Request):
        return self.do_open(
            _DeadlineHTTPSConnection,
            request,
            context=self._context,
            deadline=_request_deadline(request),
        )


def _urlopen_without_redirects(request: Request, *, timeout: float):
    parsed = urlsplit(request.full_url)
    handlers: list[Any] = [_DeadlineHTTPHandler(), _DeadlineHTTPSHandler(), _RejectRedirects()]
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
            deadline=deadline,
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
    deadline: float,
) -> tuple[Any, int]:
    request_headers = {
        **headers,
        "Accept": "application/json",
        "User-Agent": "SkillEvaluator/model-catalog",
    }
    try:
        request = Request(url, headers=request_headers, method="GET")
        request._skillevaluator_deadline = deadline
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310 - validated above
            raw = _read_response_body(response, max_response_bytes=max_response_bytes, deadline=deadline)
    except HTTPError as exc:
        raise ModelCatalogError(f"model catalog returned HTTP {exc.code}") from None
    except TimeoutError:
        raise ModelCatalogError("model catalog request timed out") from None
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise ModelCatalogError("model catalog request timed out") from None
        raise ModelCatalogError(f"model catalog request failed: {type(exc).__name__}") from None
    except (HTTPException, OSError) as exc:
        raise ModelCatalogError(f"model catalog request failed: {type(exc).__name__}") from None
    except (TypeError, ValueError):
        raise ModelCatalogError("model catalog request configuration is invalid") from None

    if len(raw) > max_response_bytes:
        raise ModelCatalogError("model catalog response exceeded the safe size limit")
    try:
        return json.loads(raw), len(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ModelCatalogError("model catalog returned invalid JSON") from None


def _read_response_body(response: Any, *, max_response_bytes: int, deadline: float) -> bytes:
    """Read one response without letting byte trickles reset the wall-clock limit.

    ``urllib``'s timeout is a socket-inactivity timeout. A peer that sends one
    byte before each socket timeout can therefore keep a single ``read`` alive
    indefinitely. Real ``HTTPResponse`` objects expose ``read1``; using it keeps
    each iteration to one underlying buffered read, while resetting the socket
    timeout to the *remaining* absolute deadline before every iteration.

    Small test doubles and alternate response objects may expose only ``read``.
    They still receive a post-read deadline check, although only a real transport
    socket can interrupt a read that is already in progress.
    """
    read_once = getattr(response, "read1", None)
    if not callable(read_once):
        raw = response.read(max_response_bytes + 1)
        if time.monotonic() >= deadline:
            raise ModelCatalogError("model catalog request timed out")
        return raw

    chunks: list[bytes] = []
    total = 0
    while total <= max_response_bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelCatalogError("model catalog request timed out")
        _set_response_socket_timeout(response, remaining)
        chunk = read_once(min(_RESPONSE_READ_CHUNK_BYTES, max_response_bytes + 1 - total))
        if time.monotonic() >= deadline:
            raise ModelCatalogError("model catalog request timed out")
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _set_response_socket_timeout(response: Any, timeout_seconds: float) -> None:
    """Best-effort propagation of the remaining deadline to urllib's socket."""
    candidates = (
        getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None),
        getattr(getattr(response, "fp", None), "_sock", None),
        getattr(getattr(response, "raw", None), "_sock", None),
        getattr(response, "_sock", None),
    )
    for candidate in candidates:
        settimeout = getattr(candidate, "settimeout", None)
        if callable(settimeout):
            settimeout(timeout_seconds)
            return


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
