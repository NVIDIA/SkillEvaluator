# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light, safe public model-catalog behavior."""

from __future__ import annotations

import io
import json
import time
from http.client import BadStatusLine, IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError, URLError

import pytest

from skillevaluator import model_catalog
from skillevaluator.model_catalog import (
    ModelCatalogError,
    ModelRecord,
    fetch_model_records,
    select_catalog_models,
)
from skillevaluator.provider_config import ProviderConfig


class _Response:
    def __init__(self, payload: object | bytes) -> None:
        self.raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.raw if size < 0 else self.raw[:size]


def _provider(
    provider: str,
    *,
    model: str = "configured-model",
    api_key: str | None = "top-secret-key",
    base_url: str | None = None,
) -> ProviderConfig:
    defaults = {
        "nv_build": "https://integrate.api.nvidia.com/v1",
        "openai": "https://api.openai.com/v1",
        "openai-compatible": "https://gateway.example.test/v1",
        "anthropic": None,
        "bedrock": None,
    }
    return ProviderConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=defaults[provider] if base_url is None else base_url,
        litellm_model=model,
        credential_env={
            "nv_build": "NVIDIA_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openai-compatible": "SKILL_EVAL_LLM_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(provider),
        region="us-west-2" if provider == "bedrock" else None,
    )


@pytest.mark.parametrize(
    ("provider", "expected_url"),
    [
        ("nv_build", "https://integrate.api.nvidia.com/v1/models"),
        ("openai", "https://api.openai.com/v1/models"),
        ("openai-compatible", "https://gateway.example.test/v1/models"),
    ],
)
def test_fetch_model_records_uses_bearer_catalog_request(monkeypatch, provider: str, expected_url: str) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, *, timeout):
        captured.update(url=request.full_url, headers=dict(request.header_items()), timeout=timeout)
        return _Response({"data": [{"id": "chat-model", "created": 7}]})

    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)

    records = fetch_model_records(_provider(provider), timeout_seconds=4.5)

    assert records == (ModelRecord("chat-model", 7),)
    assert captured["url"] == expected_url
    assert captured["timeout"] == 4.5
    assert captured["headers"]["Authorization"] == "Bearer top-secret-key"


def test_fetch_model_records_uses_anthropic_native_headers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, **_kwargs):
        captured.update(url=request.full_url, headers=dict(request.header_items()))
        return _Response({"data": [{"id": "claude-sonnet-4-5"}]})

    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)

    records = fetch_model_records(_provider("anthropic"))

    assert records == (ModelRecord("claude-sonnet-4-5"),)
    assert captured["url"] == "https://api.anthropic.com/v1/models"
    assert captured["headers"]["X-api-key"] == "top-secret-key"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"
    assert "Authorization" not in captured["headers"]


def test_fetch_model_records_follows_bounded_anthropic_cursor_pages(monkeypatch) -> None:
    urls: list[str] = []
    pages = iter(
        (
            {"data": [{"id": "model-a"}], "has_more": True, "last_id": "model-a"},
            {"data": [{"id": "configured-model"}], "has_more": False, "last_id": "configured-model"},
        )
    )

    def fake_urlopen(request, **_kwargs):
        urls.append(request.full_url)
        return _Response(next(pages))

    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)
    config = _provider("anthropic")

    records = fetch_model_records(config)

    assert records == (ModelRecord("model-a"), ModelRecord("configured-model"))
    assert urls == [
        "https://api.anthropic.com/v1/models",
        "https://api.anthropic.com/v1/models?after_id=model-a",
    ]
    assert select_catalog_models(config, records, limit=1)[0].id == "configured-model"


def test_fetch_model_records_rejects_repeated_anthropic_cursor(monkeypatch) -> None:
    monkeypatch.setattr(
        model_catalog,
        "urlopen",
        lambda *_args, **_kwargs: _Response({"data": [{"id": "model-a"}], "has_more": True, "last_id": "model-a"}),
    )

    with pytest.raises(ModelCatalogError, match="pagination"):
        fetch_model_records(_provider("anthropic"))


def test_anthropic_pagination_uses_one_overall_timeout(monkeypatch) -> None:
    timeouts: list[float] = []
    pages = iter(
        (
            {"data": [{"id": "model-a"}], "has_more": True, "last_id": "model-a"},
            {"data": [{"id": "model-b"}], "has_more": False, "last_id": "model-b"},
        )
    )

    def fake_urlopen(_request, *, timeout):
        timeouts.append(timeout)
        return _Response(next(pages))

    # Deadline creation, page-one post-read check, page-two timeout, and
    # page-two post-read check respectively.
    monotonic = iter((100.0, 100.0, 105.0, 105.0))
    monkeypatch.setattr(model_catalog.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)

    assert len(fetch_model_records(_provider("anthropic"), timeout_seconds=15.0)) == 2
    assert timeouts == [15.0, 10.0]


def test_anthropic_pagination_fails_when_overall_deadline_expires(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(_request, **_kwargs):
        nonlocal calls
        calls += 1
        return _Response({"data": [{"id": "model-a"}], "has_more": True, "last_id": "model-a"})

    monotonic = iter((100.0, 116.0))
    monkeypatch.setattr(model_catalog.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)

    with pytest.raises(ModelCatalogError, match="timed out"):
        fetch_model_records(_provider("anthropic"), timeout_seconds=15.0)
    assert calls == 1


def test_catalog_body_trickle_cannot_extend_the_wall_clock_deadline() -> None:
    payload = json.dumps({"data": [{"id": "slow-model"}]}).encode()

    class SlowBodyHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                for byte in payload:
                    self.wfile.write(bytes((byte,)))
                    self.wfile.flush()
                    time.sleep(0.025)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args) -> None:
            return None

    with ThreadingHTTPServer(("127.0.0.1", 0), SlowBodyHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            config = _provider(
                "openai-compatible",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
            )
            with pytest.raises(ModelCatalogError, match="timed out"):
                fetch_model_records(config, timeout_seconds=0.15)
            elapsed = time.monotonic() - started
        finally:
            server.shutdown()
            thread.join(timeout=2)

    # The old single response.read() waited for the complete slow body (~0.9s).
    # Allow ample scheduler jitter while still proving the configured deadline is
    # wall-clock bounded rather than reset by each arriving byte.
    assert elapsed < 0.5


def test_catalog_dns_resolution_cannot_extend_the_wall_clock_deadline(monkeypatch) -> None:
    calls = 0

    def slow_resolution(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        time.sleep(0.2)
        return []

    monkeypatch.setattr(model_catalog.socket, "getaddrinfo", slow_resolution)
    started = time.monotonic()
    try:
        with pytest.raises(ModelCatalogError, match="timed out"):
            fetch_model_records(
                _provider("openai-compatible", base_url="https://slow-resolver.example/v1"),
                timeout_seconds=0.05,
            )
        elapsed = time.monotonic() - started
        second_started = time.monotonic()
        with pytest.raises(ModelCatalogError, match="timed out"):
            fetch_model_records(
                _provider("openai-compatible", base_url="https://slow-resolver.example/v1"),
                timeout_seconds=0.05,
            )
        second_elapsed = time.monotonic() - second_started
    finally:
        # A deadline-bounded daemon resolver may still be completing the OS
        # call; let it release the one global resolver slot for later tests.
        time.sleep(0.2)

    assert elapsed < 0.15
    assert second_elapsed < 0.15
    assert calls == 1, "a stuck resolver must not create an unbounded thread or work queue"


@pytest.mark.parametrize(
    ("immediate", "trickled", "remainder"),
    [
        (
            b"",
            b"HTTP/1.1 200 OK\r\n",
            b'Content-Type: application/json\r\nContent-Length: 12\r\n\r\n{"data": []}',
        ),
        (
            b"HTTP/1.1 200 OK\r\n",
            b"X-Slow-Header: xxxxxxxxxxxxxxxxxxxx\r\n",
            b'Content-Type: application/json\r\nContent-Length: 12\r\n\r\n{"data": []}',
        ),
    ],
    ids=("status-line", "headers"),
)
def test_catalog_status_and_header_trickles_cannot_extend_the_wall_clock_deadline(
    immediate: bytes,
    trickled: bytes,
    remainder: bytes,
) -> None:
    class SlowProtocolHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                self.connection.sendall(immediate)
                for byte in trickled:
                    self.connection.sendall(bytes((byte,)))
                    time.sleep(0.025)
                self.connection.sendall(remainder)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args) -> None:
            return None

    with ThreadingHTTPServer(("127.0.0.1", 0), SlowProtocolHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            config = _provider(
                "openai-compatible",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
            )
            with pytest.raises(ModelCatalogError, match="timed out"):
                fetch_model_records(config, timeout_seconds=0.1)
            elapsed = time.monotonic() - started
        finally:
            server.shutdown()
            thread.join(timeout=2)

    # A socket-inactivity timeout is reset by every byte and takes roughly
    # 0.5-1.0s for these responses. The configured timeout is an absolute
    # deadline spanning connection setup, status, headers, and body parsing.
    assert elapsed < 0.3


def test_catalog_connect_attempts_share_one_absolute_deadline(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self, *, fail_connect: bool) -> None:
            self.fail_connect = fail_connect
            self.timeouts: list[float] = []
            self.closed = False

        def settimeout(self, timeout: float) -> None:
            self.timeouts.append(timeout)

        def bind(self, _source_address) -> None:
            return None

        def connect(self, _socket_address) -> None:
            if self.fail_connect:
                raise OSError("first address failed")

        def close(self) -> None:
            self.closed = True

    first = FakeSocket(fail_connect=True)
    second = FakeSocket(fail_connect=False)
    sockets = iter((first, second))
    monkeypatch.setattr(
        model_catalog.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (model_catalog.socket.AF_INET, model_catalog.socket.SOCK_STREAM, 6, "", ("127.0.0.1", 1)),
            (model_catalog.socket.AF_INET, model_catalog.socket.SOCK_STREAM, 6, "", ("127.0.0.2", 1)),
        ],
    )
    monkeypatch.setattr(model_catalog.socket, "socket", lambda *_args, **_kwargs: next(sockets))
    # Resolver-slot acquisition, resolver result wait, post-resolution check,
    # first address, then the second address before/after connect.
    monotonic = iter((100.0, 100.0, 100.0, 100.0, 100.08, 100.08))
    monkeypatch.setattr(model_catalog.time, "monotonic", lambda: next(monotonic))

    connection = model_catalog._DeadlineHTTPConnection("example.test", timeout=1.0, deadline=100.1)
    connected = connection._create_connection_before_deadline(("example.test", 443), None, None)

    assert connected is second
    assert first.closed is True
    assert first.timeouts == [pytest.approx(0.1)]
    assert second.timeouts == [pytest.approx(0.02), pytest.approx(0.02)]


def test_loopback_catalog_transport_bypasses_environment_proxy(monkeypatch) -> None:
    authorization: list[str | None] = []
    payload = b'{"data": [{"id": "loopback-model"}]}'

    class CatalogHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            authorization.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return None

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    with ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = _provider(
                "openai-compatible",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
            )
            records = fetch_model_records(config, timeout_seconds=1.0)
        finally:
            server.shutdown()
            thread.join(timeout=2)

    assert records == (ModelRecord("loopback-model"),)
    assert authorization == ["Bearer top-secret-key"]


def test_anthropic_pagination_enforces_aggregate_response_bytes(monkeypatch) -> None:
    first = json.dumps({"data": [{"id": "model-a"}], "has_more": True, "last_id": "model-a"}).encode()
    second = json.dumps({"data": [{"id": "model-b"}], "has_more": False, "last_id": "model-b"}).encode()
    pages = iter((_Response(first), _Response(second)))
    monkeypatch.setattr(model_catalog, "_MAX_RESPONSE_BYTES", len(first) + len(second) - 1)
    monkeypatch.setattr(model_catalog, "urlopen", lambda *_args, **_kwargs: next(pages))

    with pytest.raises(ModelCatalogError, match="safe size limit"):
        fetch_model_records(_provider("anthropic"))


def test_catalog_record_cap_is_fail_closed_without_off_by_one(monkeypatch) -> None:
    monkeypatch.setattr(model_catalog, "_MAX_MODEL_RECORDS", 2)
    monkeypatch.setattr(
        model_catalog,
        "urlopen",
        lambda *_args, **_kwargs: _Response({"data": [{"id": "a"}, {"id": "b"}]}),
    )
    assert [record.id for record in fetch_model_records(_provider("openai"))] == ["a", "b"]

    monkeypatch.setattr(
        model_catalog,
        "urlopen",
        lambda *_args, **_kwargs: _Response({"data": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}),
    )
    with pytest.raises(ModelCatalogError, match="safe record limit"):
        fetch_model_records(_provider("openai"))


def test_anthropic_page_cap_is_fail_closed(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(_request, **_kwargs):
        nonlocal calls
        calls += 1
        model_id = f"model-{calls}"
        return _Response({"data": [{"id": model_id}], "has_more": True, "last_id": model_id})

    monkeypatch.setattr(model_catalog, "_MAX_CATALOG_PAGES", 2)
    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)

    with pytest.raises(ModelCatalogError, match="safe pagination limit"):
        fetch_model_records(_provider("anthropic"))
    assert calls == 2


def test_bedrock_catalog_defers_to_existing_doctor_without_network(monkeypatch) -> None:
    monkeypatch.setattr(
        model_catalog,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network must not be called")),
    )

    with pytest.raises(ModelCatalogError, match=r"doctor --verify-models"):
        fetch_model_records(_provider("bedrock", api_key=None))


def test_fetch_normalizes_deduplicates_and_skips_unsafe_records(monkeypatch) -> None:
    payload = {
        "data": [
            {"id": " model-a ", "created": 3},
            {"id": "model-a", "created": 9},
            {"id": "model-b", "created": True},
            {"id": "bad\x1b[2J"},
            {"id": "bad\u202eoverride"},
            {"id": "bad\u200bzero-width"},
            {"id": "x" * 513},
            {"id": ""},
            None,
        ]
    }
    monkeypatch.setattr(model_catalog, "urlopen", lambda *_args, **_kwargs: _Response(payload))

    assert fetch_model_records(_provider("openai")) == (
        ModelRecord("model-a", 3),
        ModelRecord("model-b"),
    )


@pytest.mark.parametrize("payload", [None, [], {}, {"data": None}, {"data": {}}, {"data": [None, {"id": ""}]}])
def test_fetch_rejects_malformed_catalog(monkeypatch, payload: object) -> None:
    monkeypatch.setattr(model_catalog, "urlopen", lambda *_args, **_kwargs: _Response(payload))

    with pytest.raises(ModelCatalogError, match="catalog response"):
        fetch_model_records(_provider("openai"))


def test_fetch_never_exposes_http_body_reason_url_or_key(monkeypatch) -> None:
    error = HTTPError(
        "https://user:password@example.test/secret-path?token=query-secret",
        401,
        "reason contains top-secret-key",
        {},
        io.BytesIO(b"body contains top-secret-key"),
    )
    monkeypatch.setattr(model_catalog, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(ModelCatalogError) as caught:
        fetch_model_records(_provider("openai"))

    message = str(caught.value)
    assert message == "model catalog returned HTTP 401"
    assert all(secret not in message for secret in ("top-secret-key", "secret-path", "query-secret", "password"))


def test_fetch_redacts_transport_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        model_catalog,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("leaked top-secret-key")),
    )

    with pytest.raises(ModelCatalogError) as caught:
        fetch_model_records(_provider("nv_build"))

    assert str(caught.value) == "model catalog request failed: URLError"


@pytest.mark.parametrize(
    "failure",
    [BadStatusLine("wire leaked top-secret-key"), IncompleteRead(b"body leaked top-secret-key", 100)],
)
def test_fetch_redacts_low_level_http_protocol_failures(monkeypatch, failure: Exception) -> None:
    if isinstance(failure, IncompleteRead):

        class BrokenResponse(_Response):
            def read(self, size: int = -1) -> bytes:  # noqa: ARG002
                raise failure

        monkeypatch.setattr(model_catalog, "urlopen", lambda *_args, **_kwargs: BrokenResponse(b""))
    else:
        monkeypatch.setattr(
            model_catalog,
            "urlopen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )

    with pytest.raises(ModelCatalogError) as caught:
        fetch_model_records(_provider("openai"))

    assert str(caught.value) == f"model catalog request failed: {type(failure).__name__}"
    assert "top-secret-key" not in str(caught.value)


def test_fetch_rejects_oversized_and_invalid_json_responses(monkeypatch) -> None:
    monkeypatch.setattr(model_catalog, "urlopen", lambda *_args, **_kwargs: _Response(b"x" * (2 * 1024 * 1024 + 1)))
    with pytest.raises(ModelCatalogError, match="safe size limit"):
        fetch_model_records(_provider("openai"))

    monkeypatch.setattr(model_catalog, "urlopen", lambda *_args, **_kwargs: _Response(b"not-json"))
    with pytest.raises(ModelCatalogError, match="invalid JSON"):
        fetch_model_records(_provider("openai"))


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.test/v1",
        "http://192.168.1.2/v1",
        "http://169.254.1.1/v1",
        "http://0.0.0.0/v1",
        "file:///tmp/catalog",
        "https://user:password@example.test/v1",
        "https://example.test/v1?token=secret",
        "https://example.test/v1#fragment",
        "https://example.test\\@evil.test/v1",
        " https://example.test/v1",
        "https://example.test/v1\n",
        "https://example.test:invalid/v1",
    ],
)
def test_catalog_rejects_unsafe_base_urls(base_url: str) -> None:
    with pytest.raises(ModelCatalogError, match="model catalog base URL"):
        fetch_model_records(_provider("openai-compatible", base_url=base_url))


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.0.0.2", "[::1]"])
def test_catalog_allows_plain_http_only_for_loopback(monkeypatch, host: str) -> None:
    captured: dict[str, str] = {}

    def fake_urlopen(request, **_kwargs):
        captured["url"] = request.full_url
        return _Response({"data": []})

    monkeypatch.setattr(model_catalog, "urlopen", fake_urlopen)

    assert fetch_model_records(_provider("openai-compatible", base_url=f"http://{host}:8000/v1")) == ()
    assert captured["url"].endswith(":8000/v1/models")


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_authenticated_catalog_requests_never_follow_redirects(status: int) -> None:
    target_requests: list[dict[str, str]] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            target_requests.append({name.lower(): value for name, value in self.headers.items()})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def log_message(self, *_args) -> None:
            return None

    with ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler) as target:
        target_thread = Thread(target=target.serve_forever, daemon=True)
        target_thread.start()

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(status)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/models")
                self.end_headers()

            def log_message(self, *_args) -> None:
                return None

        with ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler) as redirect:
            redirect_thread = Thread(target=redirect.serve_forever, daemon=True)
            redirect_thread.start()
            try:
                config = _provider("openai-compatible", base_url=f"http://127.0.0.1:{redirect.server_port}/v1")
                with pytest.raises(ModelCatalogError, match=f"HTTP {status}"):
                    fetch_model_records(config)
            finally:
                redirect.shutdown()
                redirect_thread.join(timeout=2)

        target.shutdown()
        target_thread.join(timeout=2)

    assert target_requests == []


def test_catalog_selection_uses_configured_model_and_filters_non_chat_candidates() -> None:
    config = _provider("nv_build", model="minimaxai/minimax-m3")
    records = [
        ModelRecord("nvidia/nemotron-3-nano-30b-a3b", 100),
        ModelRecord("nvidia/nv-embed-v1", 500),
        ModelRecord("minimaxai/minimax-m3", 1),
        ModelRecord("provider/rerank-model", 600),
        ModelRecord("openai/gpt-oss-120b", 50),
    ]

    result = select_catalog_models(config, records, limit=2)

    assert [(item.id, item.is_configured) for item in result] == [
        ("minimaxai/minimax-m3", True),
        ("nvidia/nemotron-3-nano-30b-a3b", False),
    ]


def test_catalog_selection_is_bounded_and_validates_limit() -> None:
    config = _provider("openai", model="model-a")
    records = [ModelRecord("model-a"), ModelRecord("model-b"), ModelRecord("model-c")]

    assert [item.id for item in select_catalog_models(config, records, limit=1)] == ["model-a"]
    with pytest.raises(ValueError, match="positive integer"):
        select_catalog_models(config, records, limit=0)
