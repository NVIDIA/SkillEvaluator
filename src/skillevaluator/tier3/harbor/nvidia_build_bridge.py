# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Build compatibility translations and a dependency-free loopback bridge."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import socket
import stat
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import BoundedSemaphore, Condition, Lock, Thread, Timer
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPHandler, HTTPRedirectHandler, HTTPSHandler, Request, build_opener

BUILD_CHAT_COMPLETIONS_PATH = "/chat/completions"
HEALTH_PATH = "/healthz"
READINESS_TOKEN_HEADER = "X-SkillEvaluator-Bridge-Token"
RESPONSES_PATH = "/v1/responses"
MESSAGES_PATH = "/v1/messages"
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"
PRODUCTION_BUILD_BASE_URL = "https://integrate.api.nvidia.com/v1"
TEST_API_KEY_PREFIX = "test-"
MAX_REQUEST_BYTES = 1_000_000
MAX_BACKEND_RESPONSE_BYTES = 1_000_000
# A trusted run may choose a lower limit; this generous fixed guard prevents
# unbounded integer serialization while still allowing 128-Ki-token models.
MAX_OUTPUT_TOKENS_PER_REQUEST = 128 * 1024
REQUEST_READ_TIMEOUT_SECONDS = 10.0
REQUEST_HEADER_TIMEOUT_SECONDS = 10.0
BACKEND_CONNECT_TIMEOUT_SECONDS = 120.0
MAX_WORKERS = 16
MAX_REQUESTS_PER_BRIDGE = 256
MAX_SECRET_FILE_BYTES = 16_384
IN_PROCESS_START_TIMEOUT_SECONDS = 5.0
MAX_CHAT_TOOL_NAME_LENGTH = 64
_BACKEND_DNS_RESOLVER_SLOT = BoundedSemaphore(1)
DROPPED_RESPONSES_SERVER_TOOL_TYPES = {"web_search", "web_search_preview"}
DROPPED_CLAUDE_CODE_ORCHESTRATION_TOOLS = {
    "Agent",
    "Task",
    "CronCreate",
    "CronDelete",
    "CronList",
    "EnterWorktree",
    "ExitWorktree",
    "ReportFindings",
    "ScheduleWakeup",
    "SendMessage",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "WebFetch",
    "WebSearch",
    "Workflow",
}

BuildRequestTransport = Callable[[str, bytes], bytes]


@dataclass(frozen=True)
class BridgeConfig:
    """Configuration for a local NVIDIA Build protocol compatibility bridge."""

    api_key: str
    build_base_url: str
    host: str
    port: int
    log_path: Path
    readiness_token: str
    request_transport: BuildRequestTransport | None = None
    client_token: str | None = None
    allowed_model: str | None = None
    max_requests: int = MAX_REQUESTS_PER_BRIDGE
    max_output_tokens: int = MAX_OUTPUT_TOKENS_PER_REQUEST


@dataclass
class RunningBridge:
    """One authenticated in-process bridge with deterministic cleanup."""

    origin: str
    client_token: str
    _server: _BridgeHTTPServer
    _thread: Thread
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        if self._thread.is_alive():
            self._server.shutdown()
        self._server.close_active_requests()
        self._thread.join(timeout=IN_PROCESS_START_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("NVIDIA Build bridge thread did not stop")
        if not self._server.wait_for_workers(BACKEND_CONNECT_TIMEOUT_SECONDS + IN_PROCESS_START_TIMEOUT_SECONDS):
            raise RuntimeError("NVIDIA Build bridge requests did not drain")
        self._server.server_close()
        self._closed = True


@dataclass(frozen=True)
class _NamespaceToolTarget:
    namespace: str
    name: str
    kind: str


class BridgePayloadError(ValueError):
    """A malformed bridge payload that an HTTP layer should return as 400."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class _BackendError(RuntimeError):
    """A Build request or response which is unsafe to pass through unchanged."""

    def __init__(self, category: str, status_code: int = 502) -> None:
        super().__init__(category)
        self.category = category
        self.status_code = status_code


class NvidiaBuildBridgeHandler(BaseHTTPRequestHandler):
    """Serve the small OpenAI/Anthropic compatibility surface on loopback."""

    bridge_config: ClassVar[BridgeConfig]

    def do_GET(self) -> None:
        self._bridge_server().finish_request_headers(self.connection)
        route = _known_route(self.path)
        if route == HEALTH_PATH:
            supplied_token = self.headers.get(READINESS_TOKEN_HEADER, "")
            if not supplied_token or not hmac.compare_digest(supplied_token, self.bridge_config.readiness_token):
                self._send_error(403, "forbidden", "invalid readiness token")
                self._log(403, route, "forbidden")
                return
            self._send_json(200, {"status": "ok"})
            self._log(200, route, "none")
            return
        self._send_error(404, "not_found", "unknown endpoint")
        self._log(404, route, "not_found")

    def do_POST(self) -> None:
        self._bridge_server().finish_request_headers(self.connection)
        route = _known_route(self.path)
        if route not in {RESPONSES_PATH, MESSAGES_PATH, COUNT_TOKENS_PATH}:
            self._send_error(404, "not_found", "unknown endpoint")
            self._log(404, route, "not_found")
            return
        if not self._client_is_authorized():
            self._send_error(403, "forbidden", "invalid bridge client credential")
            self._log(403, route, "forbidden")
            return
        if not self._bridge_server().claim_request():
            self._send_error(429, "rate_limit", "bridge request budget exhausted")
            self._log(429, route, "request_budget_exhausted")
            return
        try:
            payload = self._read_json()
            if route == COUNT_TOKENS_PATH:
                self._send_json(200, {"input_tokens": 0})
                self._log(200, route, "none")
                return
            if route == RESPONSES_PATH:
                chat_request, custom_tool_names, namespace_tools = _responses_to_chat_request(
                    payload, max_output_tokens=self.bridge_config.max_output_tokens
                )
                _enforce_allowed_model(self.bridge_config, chat_request)
                events = _translate_backend(
                    _request_build(self.bridge_config, chat_request),
                    chat_completion_to_responses_events,
                    custom_tool_names,
                    namespace_tools,
                )
            elif route == MESSAGES_PATH:
                chat_request = _anthropic_to_chat_request(
                    payload, max_output_tokens=self.bridge_config.max_output_tokens
                )
                _enforce_allowed_model(self.bridge_config, chat_request)
                events = _translate_backend(
                    _request_build(self.bridge_config, chat_request), chat_completion_to_anthropic_events
                )
        except BridgePayloadError as error:
            error_type = {
                403: "forbidden",
                408: "request_timeout",
                413: "request_too_large",
                429: "rate_limit",
            }.get(error.status_code, "invalid_request")
            self._send_error(error.status_code, error_type, str(error))
            self._log(error.status_code, route, error_type)
            return
        except _BackendError as error:
            self._send_error(error.status_code, "backend_error", "Build backend request failed")
            self._log(error.status_code, route, error.category)
            return

        self._send_sse(events)
        self._log(200, route, "none")

    def _client_is_authorized(self) -> bool:
        expected = self.bridge_config.client_token
        if expected is None:
            return True
        supplied: list[str] = []
        authorization = self.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            supplied.append(authorization.removeprefix("Bearer "))
        api_key = self.headers.get("x-api-key", "")
        if api_key:
            supplied.append(api_key)
        return any(hmac.compare_digest(candidate, expected) for candidate in supplied)

    def _bridge_server(self) -> _BridgeHTTPServer:
        if not isinstance(self.server, _BridgeHTTPServer):
            raise RuntimeError("bridge handler requires a bounded bridge server")
        return self.server

    def _read_json(self) -> Any:
        raw_content_length = self.headers.get("Content-Length")
        if raw_content_length is None:
            raise BridgePayloadError("Content-Length is required")
        try:
            content_length = int(raw_content_length)
        except ValueError as error:
            raise BridgePayloadError("invalid Content-Length") from error
        if content_length < 0:
            raise BridgePayloadError("invalid Content-Length")
        if content_length > MAX_REQUEST_BYTES:
            raise BridgePayloadError("request body is too large", 413)
        previous_timeout = self.connection.gettimeout()
        try:
            deadline = time.monotonic() + REQUEST_READ_TIMEOUT_SECONDS
            body_parts: list[bytes] = []
            remaining = content_length
            while remaining:
                seconds_left = deadline - time.monotonic()
                if seconds_left <= 0:
                    raise TimeoutError("request body deadline expired")
                self.connection.settimeout(seconds_left)
                chunk = self.rfile.read1(min(remaining, 64 * 1024))
                if not chunk:
                    break
                body_parts.append(chunk)
                remaining -= len(chunk)
            body = b"".join(body_parts)
        except (OSError, TimeoutError) as error:
            raise BridgePayloadError("request body read timed out", 408) from error
        finally:
            self.connection.settimeout(previous_timeout)
        if len(body) != content_length:
            raise BridgePayloadError("request body read timed out", 408)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise BridgePayloadError("invalid JSON payload") from error

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: int, error_type: str, message: str) -> None:
        self._send_json(status, {"error": {"type": error_type, "message": message}})

    def _send_sse(self, events: list[dict[str, Any]]) -> None:
        data = b"".join(
            f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode() for event in events
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _log(self, status: int, route: str, category: str) -> None:
        self.bridge_config.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.bridge_config.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"status={status} path={route} error={category}\n")

    def log_message(self, _format: str, *_args: object) -> None:
        """Prevent the base handler from emitting unreviewed request diagnostics."""


def _known_route(request_path: str) -> str:
    """Return only a known route so logs cannot include query strings or secrets."""
    try:
        path = urlsplit(request_path).path
    except ValueError:
        return "unknown"
    return path if path in {HEALTH_PATH, RESPONSES_PATH, MESSAGES_PATH, COUNT_TOKENS_PATH} else "unknown"


def _build_endpoint(config: BridgeConfig) -> str:
    """Validate the permitted upstream origin and construct its only endpoint."""
    try:
        parsed = urlsplit(config.build_base_url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("build_base_url must have a valid port") from error
    if parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.path != "/v1":
        raise ValueError("build_base_url must be the approved origin with an exact /v1 path")

    if config.request_transport is None:
        if parsed.scheme != "https" or parsed.hostname != "integrate.api.nvidia.com" or port not in {None, 443}:
            raise ValueError(f"build_base_url must be {PRODUCTION_BUILD_BASE_URL}")
        return f"{PRODUCTION_BUILD_BASE_URL}{BUILD_CHAT_COMPLETIONS_PATH}"

    if not config.api_key.startswith(TEST_API_KEY_PREFIX):
        raise ValueError("test transport requires a test-only api_key")
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or port is None:
        raise ValueError("test transport requires an explicit 127.0.0.1 http build_base_url")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")) + BUILD_CHAT_COMPLETIONS_PATH


def _enforce_allowed_model(config: BridgeConfig, payload: dict[str, Any]) -> None:
    """Prevent an authenticated trial from turning the bridge into a general credential proxy."""
    if config.allowed_model is not None and payload.get("model") != config.allowed_model:
        raise BridgePayloadError("requested model is not allowed", 403)


def _request_build(config: BridgeConfig, payload: dict[str, Any]) -> Any:
    endpoint = _build_endpoint(config)
    request_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(request_body) > MAX_REQUEST_BYTES:
        raise BridgePayloadError("translated request body is too large", 413)
    deadline_controller: _BackendDeadline | None = None
    try:
        if config.request_transport is not None:
            response_body = config.request_transport(endpoint, request_body)
        else:
            request = Request(
                endpoint,
                data=request_body,
                headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            deadline_controller = _BackendDeadline(BACKEND_CONNECT_TIMEOUT_SECONDS)
            with deadline_controller:
                opener = build_opener(
                    _NoRedirect(),
                    _DeadlineHTTPHandler(deadline_controller),
                    _DeadlineHTTPSHandler(deadline_controller),
                )
                with opener.open(request, timeout=deadline_controller.remaining()) as response:
                    deadline_controller.register_response(response)
                    response_body = _read_backend_response(response, deadline_controller.deadline)
                    if deadline_controller.timed_out:
                        raise TimeoutError("Build backend deadline expired")
        if not isinstance(response_body, bytes) or len(response_body) > MAX_BACKEND_RESPONSE_BYTES:
            raise _BackendError("response_too_large")
        return json.loads(response_body.decode("utf-8"))
    except HTTPError as error:
        is_timeout = error.code in {408, 504} or (deadline_controller is not None and deadline_controller.timed_out)
        raise _BackendError("timeout" if is_timeout else "http_error", 504 if is_timeout else 502) from error
    except (HTTPException, OSError, TimeoutError, URLError) as error:
        is_timeout = (
            isinstance(error, TimeoutError)
            or (deadline_controller is not None and deadline_controller.timed_out)
            or "timed out" in str(getattr(error, "reason", error)).lower()
        )
        raise _BackendError("timeout" if is_timeout else "network_error", 504 if is_timeout else 502) from error
    except ValueError as error:
        if deadline_controller is not None and deadline_controller.timed_out:
            raise _BackendError("timeout", 504) from error
        raise _BackendError("invalid_backend_response") from error
    except (UnicodeDecodeError, RecursionError) as error:
        raise _BackendError("invalid_backend_response") from error


def _read_backend_response(response: Any, deadline: float) -> bytes:
    response_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    read1 = getattr(response, "read1", None)
    if response_socket is None or not callable(read1):
        raise _BackendError("invalid_backend_response")
    is_closed = getattr(response, "isclosed", None)
    parts: list[bytes] = []
    remaining = MAX_BACKEND_RESPONSE_BYTES + 1
    while remaining:
        # ``HTTPResponse.read1`` closes its socket as soon as the declared
        # Content-Length is consumed. Do not try to reset a timeout on that
        # already-complete descriptor during the next loop iteration.
        if callable(is_closed) and is_closed():
            break
        seconds_left = deadline - time.monotonic()
        if seconds_left <= 0:
            raise TimeoutError("Build backend deadline expired")
        response_socket.settimeout(seconds_left)
        chunk = read1(min(remaining, 64 * 1024))
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _translate_backend(completion: Any, translator: Any, *translator_args: Any) -> list[dict[str, Any]]:
    try:
        return list(translator(completion, *translator_args))
    except BridgePayloadError as error:
        raise _BackendError("invalid_backend_response") from error


class _NoRedirect(HTTPRedirectHandler):
    """Treat redirects as backend errors instead of forwarding to another host."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _BackendDeadline:
    """Abort a credentialed backend connection at one absolute wall-clock deadline."""

    def __init__(self, timeout: float) -> None:
        self.deadline = time.monotonic() + timeout
        self._lock = Lock()
        self._connections: list[Any] = []
        self._response_sockets: list[Any] = []
        self._expired = False
        self._finished = False
        self._timer = Timer(timeout, self._expire)
        self._timer.daemon = True

    def __enter__(self) -> _BackendDeadline:
        self._timer.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        with self._lock:
            self._finished = True
        self._timer.cancel()

    @property
    def expired(self) -> bool:
        with self._lock:
            return self._expired

    @property
    def timed_out(self) -> bool:
        return self.expired or time.monotonic() >= self.deadline

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Build backend deadline expired")
        return remaining

    def register_connection(self, connection: Any) -> None:
        with self._lock:
            if self._expired:
                raise TimeoutError("Build backend deadline expired")
            self._connections.append(connection)

    def register_response(self, response: Any) -> None:
        response_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if response_socket is None:
            raise _BackendError("invalid_backend_response")
        with self._lock:
            expired = self._expired
            if not expired:
                self._response_sockets.append(response_socket)
        if expired:
            self._abort_socket(response_socket)
            raise TimeoutError("Build backend deadline expired")

    def _expire(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._expired = True
            connections = list(self._connections)
            response_sockets = list(self._response_sockets)
        for connection in connections:
            self._abort_socket(getattr(connection, "sock", None))
        for response_socket in response_sockets:
            self._abort_socket(response_socket)

    @staticmethod
    def _abort_socket(backend_socket: Any) -> None:
        if backend_socket is None:
            return
        with suppress(OSError):
            backend_socket.shutdown(socket.SHUT_RDWR)
        with suppress(OSError):
            backend_socket.close()


class _DeadlineHTTPHandler(HTTPHandler):
    def __init__(self, deadline: _BackendDeadline) -> None:
        super().__init__()
        self._deadline = deadline

    def http_open(self, request: Any) -> Any:
        return self.do_open(self._connection, request)

    def _connection(self, host: str, **kwargs: Any) -> HTTPConnection:
        return _DeadlineHTTPConnection(host, deadline=self._deadline, **kwargs)


class _DeadlineHTTPSHandler(HTTPSHandler):
    def __init__(self, deadline: _BackendDeadline) -> None:
        super().__init__()
        self._deadline = deadline

    def https_open(self, request: Any) -> Any:
        return self.do_open(self._connection, request, context=self._context)

    def _connection(self, host: str, **kwargs: Any) -> HTTPSConnection:
        return _DeadlineHTTPSConnection(host, deadline=self._deadline, **kwargs)


def _resolve_backend_before_deadline(host: str, port: int, deadline: _BackendDeadline) -> list[tuple[Any, ...]]:
    """Resolve one backend host without allowing DNS to outlive the request deadline."""
    if not _BACKEND_DNS_RESOLVER_SLOT.acquire(timeout=deadline.remaining()):
        raise TimeoutError("Build backend deadline expired")

    outcome: Queue[Any] = Queue(maxsize=1)

    def resolve() -> None:
        try:
            outcome.put_nowait(socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM))
        except BaseException as error:
            outcome.put_nowait(error)
        finally:
            _BACKEND_DNS_RESOLVER_SLOT.release()

    worker = Thread(target=resolve, name="skillevaluator-build-backend-dns", daemon=True)
    try:
        worker.start()
    except BaseException:
        _BACKEND_DNS_RESOLVER_SLOT.release()
        raise

    try:
        resolved = outcome.get(timeout=deadline.remaining())
    except Empty:
        raise TimeoutError("Build backend deadline expired") from None
    if isinstance(resolved, BaseException):
        raise resolved
    return list(resolved)


class _DeadlineConnectionMixin:
    def __init__(self, host: str, *, deadline: _BackendDeadline, **kwargs: Any) -> None:
        self._deadline = deadline
        super().__init__(host, **kwargs)
        self._create_connection = self._create_connection_before_deadline
        deadline.register_connection(self)

    def _create_connection_before_deadline(
        self,
        address: tuple[str, int],
        _timeout: object,
        source_address: tuple[str, int] | None,
    ) -> socket.socket:
        host, port = address
        addresses = _resolve_backend_before_deadline(host, port, self._deadline)
        self._deadline.remaining()
        last_error: OSError | None = None
        for family, socktype, proto, _canonname, socket_address in addresses:
            self._deadline.remaining()
            candidate: socket.socket | None = None
            try:
                candidate = socket.socket(family, socktype, proto)
                candidate.settimeout(self._deadline.remaining())
                if source_address:
                    candidate.bind(source_address)
                candidate.connect(socket_address)
                candidate.settimeout(self._deadline.remaining())
                return candidate
            except OSError as error:
                last_error = error
                if candidate is not None:
                    candidate.close()

        if last_error is not None:
            raise last_error
        raise OSError("getaddrinfo returned no addresses")

    def connect(self) -> None:
        self.timeout = self._deadline.remaining()
        super().connect()
        if self.sock is not None:
            self.sock.settimeout(self._deadline.remaining())


class _DeadlineHTTPConnection(_DeadlineConnectionMixin, HTTPConnection):
    pass


class _DeadlineHTTPSConnection(_DeadlineConnectionMixin, HTTPSConnection):
    def connect(self) -> None:
        """Refresh the deadline between proxy tunneling and the TLS handshake."""
        self.timeout = self._deadline.remaining()
        HTTPConnection.connect(self)
        if self.sock is None:
            raise OSError("Build backend connection did not create a socket")
        self.sock.settimeout(self._deadline.remaining())
        server_hostname = self._tunnel_host or self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)
        self.sock.settimeout(self._deadline.remaining())


def _handler_for(config: BridgeConfig) -> type[NvidiaBuildBridgeHandler]:
    class ConfiguredNvidiaBuildBridgeHandler(NvidiaBuildBridgeHandler):
        bridge_config = config

    return ConfiguredNvidiaBuildBridgeHandler


def _write_ready_file(path: Path, *, origin: str, readiness_token: str) -> None:
    """Create the private readiness handoff without following an existing link."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("secure ready-file creation requires O_NOFOLLOW")
    payload = json.dumps(
        {"origin": origin, "readiness_token": readiness_token},
        separators=(",", ":"),
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while creating ready file")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)


def _read_ready_file(path: Path) -> tuple[str, str]:
    """Read and validate a private regular ready file through a no-follow fd."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("secure ready-file reading requires O_NOFOLLOW")
    descriptor = os.open(path, os.O_RDONLY | nofollow)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("ready file must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RuntimeError("ready file must have mode 0600")
        if metadata.st_uid != os.geteuid():
            raise RuntimeError("ready file must be owned by the current user")
        with os.fdopen(descriptor, encoding="utf-8") as ready_stream:
            descriptor = -1
            raw_ready = ready_stream.read(4097)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw_ready) > 4096:
        raise RuntimeError("ready file is too large")
    try:
        ready = json.loads(raw_ready)
    except (json.JSONDecodeError, RecursionError) as error:
        raise RuntimeError("ready file is not valid JSON") from error
    if not isinstance(ready, dict) or set(ready) != {"origin", "readiness_token"}:
        raise RuntimeError("ready file has an invalid shape")
    origin = ready["origin"]
    readiness_token = ready["readiness_token"]
    if not isinstance(origin, str) or not isinstance(readiness_token, str) or not readiness_token:
        raise RuntimeError("ready file has invalid values")
    if "\r" in readiness_token or "\n" in readiness_token:
        raise RuntimeError("ready file has an invalid readiness token")
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as error:
        raise RuntimeError("ready file has an invalid origin") from error
    expected_origin = f"http://127.0.0.1:{port}" if port is not None else ""
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or port is None
        or origin != expected_origin
    ):
        raise RuntimeError("ready file has an invalid origin")
    return origin, readiness_token


def check_ready_file(path: Path) -> str:
    """Authenticate the bound bridge described by a private ready file."""
    origin, readiness_token = _read_ready_file(path)
    _check_bridge_health(origin, readiness_token)
    return origin


def _check_bridge_health(origin: str, readiness_token: str) -> None:
    """Require one authenticated health response from a bound bridge."""
    request = Request(
        f"{origin}{HEALTH_PATH}",
        headers={READINESS_TOKEN_HEADER: readiness_token},
        method="GET",
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=1) as response:
            body = response.read(1025)
            status_code = response.status
    except (HTTPError, OSError, TimeoutError, URLError) as error:
        raise RuntimeError("authenticated bridge health check failed") from error
    try:
        health = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise RuntimeError("authenticated bridge health check returned invalid JSON") from error
    if status_code != 200 or health != {"status": "ok"}:
        raise RuntimeError("authenticated bridge health check returned an invalid response")


def start_in_process_bridge(
    *,
    api_key: str,
    build_base_url: str,
    log_path: Path,
    request_transport: BuildRequestTransport | None = None,
    allowed_model: str | None = None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS_PER_REQUEST,
) -> RunningBridge:
    """Start one authenticated loopback bridge in a managed daemon thread."""
    readiness_token = secrets.token_urlsafe(32)
    client_token = secrets.token_urlsafe(32)
    config = BridgeConfig(
        api_key=api_key,
        build_base_url=build_base_url,
        host="127.0.0.1",
        port=0,
        log_path=log_path,
        readiness_token=readiness_token,
        request_transport=request_transport,
        client_token=client_token,
        allowed_model=allowed_model,
        max_output_tokens=max_output_tokens,
    )
    server = _create_bridge_server(config)

    def run_server() -> None:
        try:
            server.serve_forever(poll_interval=0.05)
        finally:
            server.server_close()

    thread = Thread(target=run_server, name="skillevaluator-nvidia-build-bridge", daemon=True)
    try:
        thread.start()
    except BaseException:
        server.server_close()
        raise

    origin = f"http://127.0.0.1:{server.server_port}"
    handle = RunningBridge(origin, client_token, server, thread)
    deadline = time.monotonic() + IN_PROCESS_START_TIMEOUT_SECONDS
    while True:
        try:
            _check_bridge_health(origin, readiness_token)
            remaining_startup_time = max(0.0, deadline - time.monotonic())
            if not server.wait_for_workers(remaining_startup_time):
                raise RuntimeError("NVIDIA Build bridge readiness request did not drain")
            return handle
        except RuntimeError:
            if not thread.is_alive() or time.monotonic() >= deadline:
                handle.close()
                raise RuntimeError("NVIDIA Build bridge health check failed") from None
            time.sleep(0.025)


def serve(
    config: BridgeConfig,
    *,
    ready_file: Path | None = None,
    on_server_ready: Callable[[ThreadingHTTPServer], None] | None = None,
) -> None:
    """Run a bridge bound to loopback only."""
    server = _create_bridge_server(config)
    ready_file_created = False
    try:
        if ready_file is not None:
            origin = f"http://{config.host}:{server.server_port}"
            _write_ready_file(ready_file, origin=origin, readiness_token=config.readiness_token)
            ready_file_created = True
        if on_server_ready is not None:
            on_server_ready(server)
        server.serve_forever()
    finally:
        server.server_close()
        if ready_file_created and ready_file is not None:
            ready_file.unlink(missing_ok=True)


def _create_bridge_server(config: BridgeConfig) -> _BridgeHTTPServer:
    """Validate configuration and bind a dynamic loopback listener."""
    if not config.api_key:
        raise ValueError("api_key must not be empty")
    if "\r" in config.api_key or "\n" in config.api_key:
        raise ValueError("api_key must not contain CR or LF")
    if not config.readiness_token:
        raise ValueError("readiness_token must not be empty")
    if "\r" in config.readiness_token or "\n" in config.readiness_token:
        raise ValueError("readiness_token must not contain CR or LF")
    if config.client_token is not None and not config.client_token:
        raise ValueError("client_token must not be empty when configured")
    if config.client_token is not None and ("\r" in config.client_token or "\n" in config.client_token):
        raise ValueError("client_token must not contain CR or LF")
    if config.allowed_model is not None and not config.allowed_model:
        raise ValueError("allowed_model must not be empty when configured")
    if config.allowed_model is not None and ("\r" in config.allowed_model or "\n" in config.allowed_model):
        raise ValueError("allowed_model must not contain CR or LF")
    if (
        isinstance(config.max_requests, bool)
        or not isinstance(config.max_requests, int)
        or not 1 <= config.max_requests <= MAX_REQUESTS_PER_BRIDGE
    ):
        raise ValueError(f"max_requests must be between 1 and {MAX_REQUESTS_PER_BRIDGE}")
    if (
        isinstance(config.max_output_tokens, bool)
        or not isinstance(config.max_output_tokens, int)
        or not 1 <= config.max_output_tokens <= MAX_OUTPUT_TOKENS_PER_REQUEST
    ):
        raise ValueError(f"max_output_tokens must be between 1 and {MAX_OUTPUT_TOKENS_PER_REQUEST}")
    if config.host != "127.0.0.1":
        raise ValueError("bridge host must be 127.0.0.1")
    _build_endpoint(config)
    if config.request_transport is None and config.client_token is None:
        raise ValueError("production bridge requires a client_token")
    if config.request_transport is None and config.allowed_model is None:
        raise ValueError("production bridge requires an allowed_model")
    if config.port != 0:
        raise ValueError("bridge port must be 0 for dynamic loopback binding")
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    config.log_path.touch(exist_ok=True)
    server = _BridgeHTTPServer(
        (config.host, config.port),
        _handler_for(config),
        max_requests=config.max_requests,
    )
    server.daemon_threads = True
    return server


class _BridgeHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    request_queue_size = MAX_WORKERS

    def __init__(self, *args: Any, max_requests: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._worker_slots = BoundedSemaphore(MAX_WORKERS)
        self._worker_condition = Condition()
        self._active_workers = 0
        self._request_budget_lock = Lock()
        self._remaining_requests = max_requests
        self._request_state_lock = Lock()
        self._active_requests: set[Any] = set()
        self._header_timers: dict[Any, Timer] = {}
        self._closing_requests = False

    def claim_request(self) -> bool:
        """Atomically consume one authenticated request from this bridge's finite budget."""
        with self._request_budget_lock:
            if self._remaining_requests <= 0:
                return False
            self._remaining_requests -= 1
            return True

    @property
    def active_workers(self) -> int:
        with self._worker_condition:
            return self._active_workers

    def wait_for_workers(self, timeout: float) -> bool:
        with self._worker_condition:
            return self._worker_condition.wait_for(lambda: self._active_workers == 0, timeout=timeout)

    def get_request(self) -> tuple[Any, Any]:
        request, client_address = super().get_request()
        request.settimeout(REQUEST_HEADER_TIMEOUT_SECONDS)
        header_timer = Timer(REQUEST_HEADER_TIMEOUT_SECONDS, self._expire_request_headers, args=(request,))
        header_timer.daemon = True
        with self._request_state_lock:
            self._active_requests.add(request)
            self._header_timers[request] = header_timer
        header_timer.start()
        return request, client_address

    def finish_request_headers(self, request: Any) -> None:
        """Cancel the absolute request-line/header deadline after parsing completes."""
        with self._request_state_lock:
            header_timer = self._header_timers.pop(request, None)
        if header_timer is not None:
            header_timer.cancel()

    def release_request(self, request: Any) -> None:
        """Drop all server bookkeeping after a request socket is no longer owned."""
        with self._request_state_lock:
            self._active_requests.discard(request)
            header_timer = self._header_timers.pop(request, None)
        if header_timer is not None:
            header_timer.cancel()

    def close_active_requests(self) -> None:
        """Force blocked request readers to exit during deterministic shutdown."""
        with self._request_state_lock:
            self._closing_requests = True
            requests = list(self._active_requests)
            header_timers = list(self._header_timers.values())
            self._header_timers.clear()
        for header_timer in header_timers:
            header_timer.cancel()
        for request in requests:
            with suppress(OSError):
                request.shutdown(socket.SHUT_RDWR)
            request.close()

    def handle_error(self, request: Any, client_address: Any) -> None:
        if self._closing_requests and isinstance(sys.exception(), OSError):
            return
        super().handle_error(request, client_address)

    def _expire_request_headers(self, request: Any) -> None:
        with self._request_state_lock:
            header_timer = self._header_timers.pop(request, None)
        if header_timer is None:
            return
        with suppress(OSError):
            request.shutdown(socket.SHUT_RDWR)
        request.close()

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._worker_slots.acquire(blocking=False):
            self.release_request(request)
            request.close()
            return
        with self._worker_condition:
            self._active_workers += 1

        def bounded_worker() -> None:
            try:
                self.process_request_thread(request, client_address)
            finally:
                self.release_request(request)
                with self._worker_condition:
                    self._active_workers -= 1
                    self._worker_condition.notify_all()
                self._worker_slots.release()

        Thread(target=bounded_worker, daemon=self.daemon_threads).start()


def main(argv: list[str] | None = None) -> int:
    """Run the loopback bridge from a trial container."""
    parser = argparse.ArgumentParser(description="NVIDIA Build compatibility bridge")
    parser.add_argument("--build-base-url")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--client-token-file", type=Path)
    parser.add_argument("--allowed-model")
    parser.add_argument("--max-requests", type=int, default=MAX_REQUESTS_PER_BRIDGE)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS_PER_REQUEST)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--check-ready-file", type=Path)
    parser.add_argument("--log-path", type=Path, default=Path("nvidia-build-bridge.log"))
    arguments = parser.parse_args(argv)
    if arguments.check_ready_file is not None:
        try:
            origin = check_ready_file(arguments.check_ready_file)
        except (OSError, RuntimeError) as exc:
            parser.error(f"bridge readiness check failed: {exc}")
        sys.stdout.write(f"{origin}\n")
        sys.stdout.flush()
        return 0
    api_key = ""
    client_token = ""
    credential_error: OSError | RuntimeError | None = None
    try:
        if arguments.api_key_file is not None:
            api_key = _consume_private_text_file(arguments.api_key_file, "NVIDIA API key")
        else:
            api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if arguments.client_token_file is not None:
            client_token = _consume_private_text_file(arguments.client_token_file, "bridge client token")
    except (OSError, RuntimeError) as exc:
        credential_error = exc
    finally:
        if arguments.api_key_file is not None:
            arguments.api_key_file.unlink(missing_ok=True)
        if arguments.client_token_file is not None:
            arguments.client_token_file.unlink(missing_ok=True)
    if arguments.build_base_url is None:
        parser.error("--build-base-url is required when serving")
    if arguments.ready_file is None:
        parser.error("--ready-file is required when serving")
    if arguments.client_token_file is None:
        parser.error("--client-token-file is required when serving")
    if not (arguments.allowed_model or "").strip():
        parser.error("--allowed-model is required when serving")
    if credential_error is not None:
        parser.error(f"cannot read private bridge credential file: {credential_error}")
    if not api_key:
        parser.error("NVIDIA API key is required")
    if not client_token:
        parser.error("bridge client token is required")
    serve(
        BridgeConfig(
            api_key=api_key,
            build_base_url=arguments.build_base_url,
            host="127.0.0.1",
            port=arguments.port,
            log_path=arguments.log_path,
            readiness_token=secrets.token_urlsafe(32),
            client_token=client_token,
            allowed_model=arguments.allowed_model.strip(),
            max_requests=arguments.max_requests,
            max_output_tokens=arguments.max_output_tokens,
        ),
        ready_file=arguments.ready_file,
    )
    return 0


def _consume_private_text_file(path: Path, label: str) -> str:
    """Read a small owner-only regular file without following links, then remove it."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError(f"secure {label} reading requires O_NOFOLLOW")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{label} file must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RuntimeError(f"{label} file must have mode 0600")
        if metadata.st_uid != os.geteuid():
            raise RuntimeError(f"{label} file must be owned by the current user")
        with os.fdopen(descriptor, encoding="utf-8") as secret_stream:
            descriptor = -1
            value = secret_stream.read(MAX_SECRET_FILE_BYTES + 1)
        if len(value) > MAX_SECRET_FILE_BYTES:
            raise RuntimeError(f"{label} file is too large")
        return value.strip()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def responses_to_chat_request(payload: Any) -> dict[str, Any]:
    """Translate an OpenAI Responses request to a Build Chat Completions request."""
    translated, _custom_tool_names, _namespace_tools = _responses_to_chat_request(payload)
    return translated


def _responses_to_chat_request(
    payload: Any,
    *,
    max_output_tokens: int = MAX_OUTPUT_TOKENS_PER_REQUEST,
) -> tuple[dict[str, Any], set[str], dict[str, _NamespaceToolTarget]]:
    request = _require_object(payload)
    tools = request.get("tools")
    if tools is None:
        translated_tools: list[dict[str, Any]] | None = None
        omit_translated_tools = False
        custom_tool_names: set[str] = set()
        namespace_tools: dict[str, _NamespaceToolTarget] = {}
    else:
        response_tools = _require_list(tools, "tools")
        translated_tools, custom_tool_names, namespace_tools = _responses_tools_to_chat(response_tools)
        omit_translated_tools = not translated_tools and any(
            isinstance(tool, dict) and tool.get("type") in DROPPED_RESPONSES_SERVER_TOOL_TYPES
            for tool in response_tools
        )

    messages = _responses_messages(request.get("input", []), namespace_tools)
    instructions = request.get("instructions")
    if instructions is not None:
        messages.insert(0, {"role": "system", "content": _content_text(instructions)})

    translated: dict[str, Any] = {"model": _require_model(request), "messages": messages}
    if translated_tools is not None and not omit_translated_tools:
        translated["tools"] = translated_tools
    _copy_request_options(request, translated, {"temperature", "top_p", "parallel_tool_calls"})
    if "tool_choice" in request:
        if request["tool_choice"] == "required" and omit_translated_tools:
            raise BridgePayloadError("Responses tool_choice 'required' needs an executable tool for NVIDIA Build")
        translated["tool_choice"] = _responses_tool_choice_to_chat(request["tool_choice"], namespace_tools)
    if "max_output_tokens" in request:
        translated["max_tokens"] = _require_output_token_limit(
            request["max_output_tokens"], "max_output_tokens", max_output_tokens
        )
    return translated, custom_tool_names, namespace_tools


def anthropic_to_chat_request(payload: Any) -> dict[str, Any]:
    """Translate an Anthropic Messages request to a Build Chat Completions request."""
    return _anthropic_to_chat_request(payload)


def _anthropic_to_chat_request(
    payload: Any, *, max_output_tokens: int = MAX_OUTPUT_TOKENS_PER_REQUEST
) -> dict[str, Any]:
    request = _require_object(payload)
    messages: list[dict[str, Any]] = []
    if "system" in request:
        messages.append({"role": "system", "content": _content_text(request["system"])})

    for message in _require_list(request.get("messages", []), "messages"):
        if not isinstance(message, dict):
            raise BridgePayloadError("messages entries must be JSON objects")
        messages.extend(_anthropic_message_to_chat(message))

    translated: dict[str, Any] = {"model": _require_model(request), "messages": messages}
    tools = request.get("tools")
    translated_tools: list[dict[str, Any]] | None = None
    if tools is not None:
        translated_tools = []
        for tool in _require_list(tools, "tools"):
            chat_tool = _anthropic_tool_to_chat(tool)
            tool_name = chat_tool["function"]["name"]
            if tool_name in DROPPED_CLAUDE_CODE_ORCHESTRATION_TOOLS or not _is_chat_tool_name(tool_name):
                continue
            translated_tools.append(chat_tool)
        if translated_tools:
            translated["tools"] = translated_tools
    _copy_request_options(request, translated, {"temperature", "top_p"})
    if "tool_choice" in request:
        tool_choice = request["tool_choice"]
        declared_tool_names = {tool["function"]["name"] for tool in translated_tools or []}
        if (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "tool"
            and (
                tool_choice.get("name") in DROPPED_CLAUDE_CODE_ORCHESTRATION_TOOLS
                or not _is_chat_tool_name(tool_choice.get("name"))
            )
        ):
            raise BridgePayloadError("forced Anthropic tool is unsupported by NVIDIA Build")
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "any" and not declared_tool_names:
            raise BridgePayloadError("Anthropic tool_choice 'any' needs a declared executable tool for NVIDIA Build")
        if (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "tool"
            and tool_choice.get("name") not in declared_tool_names
        ):
            raise BridgePayloadError("forced Anthropic tool_choice requires a declared executable tool")
        translated["tool_choice"] = _anthropic_tool_choice_to_chat(tool_choice)
        if isinstance(tool_choice, dict) and tool_choice.get("disable_parallel_tool_use") is True:
            translated["parallel_tool_calls"] = False
    if "max_tokens" in request:
        translated["max_tokens"] = _require_output_token_limit(request["max_tokens"], "max_tokens", max_output_tokens)
    return translated


def chat_completion_to_responses_events(
    payload: Any,
    custom_tool_names: set[str] | None = None,
    namespace_tools: dict[str, _NamespaceToolTarget] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield terminal Responses API events for one non-streaming Chat Completion."""
    response_id, model, content, tool_calls = _chat_completion_parts(payload)
    output: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": _responses_response(response_id, model, "in_progress", []),
        }
    ]

    if content:
        message = {
            "type": "message",
            "id": f"{response_id}-message-0",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
            "status": "completed",
        }
        output.append(message)
        output_index = len(output) - 1
        events.extend(
            [
                {"type": "response.output_item.added", "output_index": output_index, "item": message},
                {
                    "type": "response.output_text.delta",
                    "output_index": output_index,
                    "item_id": message["id"],
                    "content_index": 0,
                    "delta": content,
                    "logprobs": [],
                },
                {
                    "type": "response.output_text.done",
                    "output_index": output_index,
                    "item_id": message["id"],
                    "content_index": 0,
                    "text": content,
                    "logprobs": [],
                },
                {"type": "response.output_item.done", "output_index": output_index, "item": message},
            ]
        )

    for tool_call in tool_calls:
        function = tool_call["function"]
        namespace_target = namespace_tools.get(function["name"]) if namespace_tools else None
        response_name = namespace_target.name if namespace_target else function["name"]
        namespace = {"namespace": namespace_target.namespace} if namespace_target else {}
        is_custom = (
            namespace_target.kind == "custom"
            if namespace_target
            else bool(custom_tool_names and function["name"] in custom_tool_names)
        )
        if is_custom:
            custom_input = _chat_custom_tool_input(function["arguments"])
            output_index = len(output)
            added_item = {
                "type": "custom_tool_call",
                "id": tool_call["id"],
                "call_id": tool_call["id"],
                "name": response_name,
                "input": "",
                **namespace,
            }
            done_item = {**added_item, "input": custom_input}
            output.append(done_item)
            events.extend(
                [
                    {"type": "response.output_item.added", "output_index": output_index, "item": added_item},
                    {
                        "type": "response.custom_tool_call_input.delta",
                        "output_index": output_index,
                        "item_id": added_item["id"],
                        "delta": custom_input,
                    },
                    {
                        "type": "response.custom_tool_call_input.done",
                        "output_index": output_index,
                        "item_id": added_item["id"],
                        "input": custom_input,
                    },
                    {"type": "response.output_item.done", "output_index": output_index, "item": done_item},
                ]
            )
            continue
        item = {
            "type": "function_call",
            "id": tool_call["id"],
            "call_id": tool_call["id"],
            "name": response_name,
            "arguments": function["arguments"],
            "status": "completed",
            **namespace,
        }
        output.append(item)
        events.extend(
            [
                {"type": "response.output_item.added", "output_index": len(output) - 1, "item": item},
                {"type": "response.output_item.done", "output_index": len(output) - 1, "item": item},
            ]
        )

    events.append(
        {"type": "response.completed", "response": _responses_response(response_id, model, "completed", output)}
    )
    for sequence_number, event in enumerate(events, start=1):
        yield {**event, "sequence_number": sequence_number}


def chat_completion_to_anthropic_events(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield terminal Anthropic Messages stream events for one Chat Completion."""
    response_id, model, content, tool_calls = _chat_completion_parts(payload)
    normalized_tool_calls = _anthropic_tool_calls(tool_calls)
    yield {
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }

    index = 0
    if content:
        yield {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}
        yield {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": content}}
        yield {"type": "content_block_stop", "index": index}
        index += 1

    for tool_call, _tool_input in normalized_tool_calls:
        function = tool_call["function"]
        yield {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": tool_call["id"], "name": function["name"], "input": {}},
        }
        yield {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": function["arguments"]},
        }
        yield {"type": "content_block_stop", "index": index}
        index += 1

    yield {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use" if normalized_tool_calls else "end_turn", "stop_sequence": None},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    yield {"type": "message_stop"}


def redact_bridge_text(value: Any) -> str:
    """Redact common credential-bearing text before bridge diagnostics are written."""
    text = str(value)
    text = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+", r"\1<redacted>", text)
    text = re.sub(r"(?i)\bnvapi-[a-z0-9_-]+\b", "nvapi-<redacted>", text)
    return re.sub(
        r"(?i)((?:[\"']?(?:nvidia_api_key|api[_-]?key)[\"']?)\s*[=:]\s*)([\"']?)[^\s,;}\]\"']+\2",
        r"\1\2<redacted>\2",
        text,
    )


def _require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise BridgePayloadError("payload must be a JSON object")
    return payload


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise BridgePayloadError(f"{field_name} must be a JSON array")
    return value


def _require_model(request: dict[str, Any]) -> str:
    model = request.get("model")
    if not isinstance(model, str) or not model:
        raise BridgePayloadError("model must be a non-empty string")
    return model


def _require_output_token_limit(value: Any, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise BridgePayloadError(f"{field_name} must be an integer between 1 and {maximum}")
    return value


def _copy_request_options(source: dict[str, Any], destination: dict[str, Any], names: set[str]) -> None:
    for name in names:
        if name in source:
            destination[name] = source[name]


def _responses_tool_choice_to_chat(choice: Any, namespace_tools: dict[str, _NamespaceToolTarget] | None = None) -> Any:
    if isinstance(choice, str) and choice in {"auto", "none", "required"}:
        return choice
    if isinstance(choice, dict) and choice.get("type") in DROPPED_RESPONSES_SERVER_TOOL_TYPES:
        raise BridgePayloadError(f"forced Responses {choice['type']} tool_choice is unsupported by NVIDIA Build")
    if (
        isinstance(choice, dict)
        and choice.get("type") in {"function", "custom"}
        and isinstance(choice.get("name"), str)
    ):
        name = choice["name"]
        namespace = choice.get("namespace")
        if namespace is not None:
            name = _namespace_chat_tool_name(
                namespace_tools or {}, namespace, name, choice["type"], "namespaced tool_choice"
            )
        return {"type": "function", "function": {"name": name}}
    raise BridgePayloadError("unsupported Responses tool_choice")


def _anthropic_tool_choice_to_chat(choice: Any) -> Any:
    if isinstance(choice, str) and choice in {"auto", "none", "required"}:
        return choice
    if not isinstance(choice, dict):
        raise BridgePayloadError("unsupported Anthropic tool_choice")
    choice_type = choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}
    raise BridgePayloadError("unsupported Anthropic tool_choice")


def _responses_response(response_id: str, model: str, status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 0.0,
        "completed_at": 0.0 if status == "completed" else None,
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "temperature": None,
        "top_p": None,
        "max_output_tokens": None,
        "error": None,
        "incomplete_details": None,
        "usage": None,
    }


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}:
                text = block.get("text")
                if not isinstance(text, str):
                    raise BridgePayloadError("text content blocks must include a string text value")
                parts.append(text)
            else:
                raise BridgePayloadError("unsupported content block")
        return "".join(parts)
    raise BridgePayloadError("content must be a string or text block array")


def _responses_messages(
    input_value: Any, namespace_tools: dict[str, _NamespaceToolTarget] | None = None
) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    messages: list[dict[str, Any]] = []
    for item in _require_list(input_value, "input"):
        if not isinstance(item, dict):
            raise BridgePayloadError("input entries must be JSON objects")
        item_type = item.get("type")
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise BridgePayloadError(f"{item_type} requires a call_id")
            messages.append({"role": "tool", "tool_call_id": call_id, "content": _content_text(item.get("output", ""))})
        elif item_type == "function_call":
            _append_responses_tool_call(messages, _responses_function_call_to_chat(item, namespace_tools))
        elif item_type == "custom_tool_call":
            _append_responses_tool_call(messages, _responses_custom_tool_call_to_chat(item, namespace_tools))
        elif item_type in {"message", "input_text", "output_text", "text", None}:
            role = item.get("role", "user")
            if role not in {"system", "user", "assistant", "developer"}:
                raise BridgePayloadError("message role is not supported")
            messages.append(
                {
                    "role": "system" if role == "developer" else role,
                    "content": _content_text(item.get("content", item.get("text", ""))),
                }
            )
        else:
            raise BridgePayloadError(f"unsupported Responses input type: {item_type}")
    return messages


def _append_responses_tool_call(messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") is None:
        messages[-1]["tool_calls"].extend(message["tool_calls"])
        return
    messages.append(message)


def _responses_function_call_to_chat(
    item: dict[str, Any], namespace_tools: dict[str, _NamespaceToolTarget] | None = None
) -> dict[str, Any]:
    call_id = item.get("call_id", item.get("id"))
    name = item.get("name")
    arguments = item.get("arguments", "{}")
    if not all(isinstance(value, str) and value for value in (call_id, name)) or not isinstance(arguments, str):
        raise BridgePayloadError("function_call requires string call_id, name, and arguments")
    namespace = item.get("namespace")
    if namespace is not None:
        name = _namespace_chat_tool_name(namespace_tools or {}, namespace, name, "function", "namespaced function_call")
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _responses_custom_tool_call_to_chat(
    item: dict[str, Any], namespace_tools: dict[str, _NamespaceToolTarget] | None = None
) -> dict[str, Any]:
    call_id = item.get("call_id", item.get("id"))
    name = item.get("name")
    custom_input = item.get("input")
    if (
        not isinstance(call_id, str)
        or not call_id
        or not isinstance(name, str)
        or not name
        or not isinstance(custom_input, str)
    ):
        raise BridgePayloadError("custom_tool_call requires string call_id, name, and input")
    namespace = item.get("namespace")
    if namespace is not None:
        name = _namespace_chat_tool_name(
            namespace_tools or {}, namespace, name, "custom", "namespaced custom_tool_call"
        )
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps({"input": custom_input}, separators=(",", ":")),
                },
            }
        ],
    }


def _responses_tool_to_chat(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise BridgePayloadError("Responses tools must be JSON objects")
    if tool.get("type") == "custom":
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise BridgePayloadError("custom tools require a name")
        raw_input_instruction = "Pass the raw custom tool input through the `input` string property."
        description = tool.get("description")
        if isinstance(description, str) and description:
            description = f"{description}\n\n{raw_input_instruction}"
        else:
            description = raw_input_instruction
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        }
    if tool.get("type") != "function":
        raise BridgePayloadError("unsupported Responses tool type")
    name = tool.get("name")
    parameters = tool.get("parameters", {})
    if not isinstance(name, str) or not name or not isinstance(parameters, dict):
        raise BridgePayloadError("function tools require a name and JSON object parameters")
    function: dict[str, Any] = {"name": name, "parameters": parameters}
    if isinstance(tool.get("description"), str):
        function["description"] = tool["description"]
    _copy_tool_strict(tool, function)
    return {"type": "function", "function": function}


def _responses_tools_to_chat(
    tools: list[Any],
) -> tuple[list[dict[str, Any]], set[str], dict[str, _NamespaceToolTarget]]:
    translated: list[dict[str, Any]] = []
    custom_tool_names: set[str] = set()
    namespace_tools: dict[str, _NamespaceToolTarget] = {}
    top_level_names: set[str] = set()

    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") in DROPPED_RESPONSES_SERVER_TOOL_TYPES:
            continue
        if isinstance(tool, dict) and tool.get("type") == "namespace":
            translated.extend(_responses_namespace_tool_to_chat(tool, namespace_tools, top_level_names))
            continue

        chat_tool = _responses_tool_to_chat(tool)
        chat_name = chat_tool["function"]["name"]
        if chat_name in namespace_tools:
            raise BridgePayloadError("Responses tool name collision after namespace flattening")
        translated.append(chat_tool)
        top_level_names.add(chat_name)
        if isinstance(tool, dict) and tool.get("type") == "custom":
            custom_tool_names.add(chat_name)

    return translated, custom_tool_names, namespace_tools


def _responses_namespace_tool_to_chat(
    tool: dict[str, Any],
    namespace_tools: dict[str, _NamespaceToolTarget],
    top_level_names: set[str],
) -> list[dict[str, Any]]:
    namespace = tool.get("name")
    if not isinstance(namespace, str) or not isinstance(tool.get("description"), str):
        raise BridgePayloadError("namespace tools require string name and description")
    _validate_namespace_name(namespace, "namespace name")
    nested_tools = _require_list(tool.get("tools"), "namespace tools")
    translated: list[dict[str, Any]] = []
    for nested_tool in nested_tools:
        if not isinstance(nested_tool, dict):
            raise BridgePayloadError("namespace tool entries must be JSON objects")
        nested_name = nested_tool.get("name")
        if not isinstance(nested_name, str):
            raise BridgePayloadError("namespaced tools require a string name")
        _validate_namespace_name(nested_name, "namespaced tool name")
        nested_kind = nested_tool.get("type")
        if nested_kind not in {"function", "custom"}:
            _responses_tool_to_chat(nested_tool)
            raise BridgePayloadError("namespace tools must contain function or custom tools")
        chat_name = f"{namespace}__{nested_name}"
        if len(chat_name) > MAX_CHAT_TOOL_NAME_LENGTH:
            raise BridgePayloadError("flattened namespace tool names must be at most 64 characters")
        if chat_name in namespace_tools or chat_name in top_level_names:
            raise BridgePayloadError("Responses tool name collision after namespace flattening")
        chat_tool = _responses_tool_to_chat(nested_tool)
        chat_tool["function"]["name"] = chat_name
        translated.append(chat_tool)
        namespace_tools[chat_name] = _NamespaceToolTarget(namespace, nested_name, nested_kind)
    return translated


def _validate_namespace_name(name: str, field_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise BridgePayloadError(f"{field_name} must contain only letters, digits, underscores, or hyphens")


def _is_chat_tool_name(name: Any) -> bool:
    return isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is not None


def _namespace_chat_tool_name(
    namespace_tools: dict[str, _NamespaceToolTarget],
    namespace: Any,
    name: str,
    kind: str,
    field_name: str,
) -> str:
    if not isinstance(namespace, str) or not namespace:
        raise BridgePayloadError(f"{field_name} requires a string namespace")
    for chat_name, target in namespace_tools.items():
        if target.namespace == namespace and target.name == name and target.kind == kind:
            return chat_name
    raise BridgePayloadError(f"{field_name} does not match a declared namespace tool")


def _anthropic_message_to_chat(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    if role not in {"user", "assistant", "system"}:
        raise BridgePayloadError("Anthropic message role must be user, assistant, or system")
    content = message.get("content", "")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    blocks = _require_list(content, "message content")
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise BridgePayloadError("message content entries must be JSON objects")
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(_content_text([block]))
        elif block_type == "tool_use":
            tool_calls.append(_anthropic_tool_use_to_chat(block))
        elif block_type == "tool_result":
            tool_messages.append(_anthropic_tool_result_to_chat(block))
        else:
            raise BridgePayloadError(f"unsupported Anthropic content type: {block_type}")

    translated: list[dict[str, Any]] = []
    if role == "system" and (tool_calls or tool_messages):
        raise BridgePayloadError("Anthropic system messages may contain only text")
    if text_parts or tool_calls:
        translated.append(
            {"role": role, "content": "".join(text_parts), **({"tool_calls": tool_calls} if tool_calls else {})}
        )
    translated.extend(tool_messages)
    return translated


def _anthropic_tool_use_to_chat(block: dict[str, Any]) -> dict[str, Any]:
    call_id = block.get("id")
    name = block.get("name")
    tool_input = block.get("input", {})
    if not all(isinstance(value, str) and value for value in (call_id, name)) or not isinstance(tool_input, dict):
        raise BridgePayloadError("tool_use requires id, name, and object input")
    if not _is_chat_tool_name(name):
        raise BridgePayloadError("tool_use name is unsupported by NVIDIA Build")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(tool_input, separators=(",", ":"), sort_keys=True)},
    }


def _anthropic_tool_result_to_chat(block: dict[str, Any]) -> dict[str, Any]:
    call_id = block.get("tool_use_id")
    if not isinstance(call_id, str) or not call_id:
        raise BridgePayloadError("tool_result requires a tool_use_id")
    return {"role": "tool", "tool_call_id": call_id, "content": _content_text(block.get("content", ""))}


def _anthropic_tool_to_chat(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise BridgePayloadError("Anthropic tools must be JSON objects")
    name = tool.get("name")
    parameters = tool.get("input_schema", {})
    if not isinstance(name, str) or not name or not isinstance(parameters, dict):
        raise BridgePayloadError("Anthropic tools require a name and object input_schema")
    function: dict[str, Any] = {"name": name, "parameters": parameters}
    if isinstance(tool.get("description"), str):
        function["description"] = tool["description"]
    _copy_tool_strict(tool, function)
    return {"type": "function", "function": function}


def _copy_tool_strict(source: dict[str, Any], destination: dict[str, Any]) -> None:
    if "strict" not in source:
        return
    if not isinstance(source["strict"], bool):
        raise BridgePayloadError("tool strict must be a boolean")
    destination["strict"] = source["strict"]


def _chat_completion_parts(payload: Any) -> tuple[str, str, str, list[dict[str, Any]]]:
    completion = _require_object(payload)
    choices = _require_list(completion.get("choices"), "choices")
    if not choices or not isinstance(choices[0], dict):
        raise BridgePayloadError("choices must contain a completion choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise BridgePayloadError("completion choice must include a message object")
    response_id = completion.get("id", "chatcmpl-bridge")
    model = completion.get("model", "")
    if not isinstance(response_id, str) or not response_id or not isinstance(model, str) or not model:
        raise BridgePayloadError("completion must include string id and model")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise BridgePayloadError("completion message content must be a string or null")
    raw_tool_calls = message.get("tool_calls", [])
    tool_calls = _require_list(raw_tool_calls, "tool_calls")
    for tool_call in tool_calls:
        _validate_chat_tool_call(tool_call)
    return response_id, model, content, tool_calls


def _anthropic_tool_calls(tool_calls: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    normalized: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for tool_call in tool_calls:
        function = tool_call["function"]
        try:
            tool_input = json.loads(function["arguments"])
        except json.JSONDecodeError as error:
            raise BridgePayloadError("tool call arguments must be valid JSON") from error
        if not isinstance(tool_input, dict):
            raise BridgePayloadError("tool call arguments must decode to a JSON object")
        normalized.append((tool_call, tool_input))
    return normalized


def _chat_custom_tool_input(arguments: str) -> str:
    try:
        tool_input = json.loads(arguments)
    except (json.JSONDecodeError, RecursionError) as error:
        raise BridgePayloadError("custom tool call arguments must be valid JSON") from error
    if not isinstance(tool_input, dict) or not isinstance(tool_input.get("input"), str):
        raise BridgePayloadError("custom tool call arguments must contain a string input")
    return tool_input["input"]


def _validate_chat_tool_call(tool_call: Any) -> None:
    if not isinstance(tool_call, dict):
        raise BridgePayloadError("tool calls must be JSON objects")
    function = tool_call.get("function")
    if not isinstance(tool_call.get("id"), str) or not tool_call["id"] or not isinstance(function, dict):
        raise BridgePayloadError("tool calls require an id and function object")
    if (
        not isinstance(function.get("name"), str)
        or not function["name"]
        or not isinstance(function.get("arguments"), str)
    ):
        raise BridgePayloadError("tool calls require function name and string arguments")


if __name__ == "__main__":
    raise SystemExit(main())
