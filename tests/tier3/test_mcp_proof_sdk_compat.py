# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK compatibility retains the private URL probe's constrained transport."""

from __future__ import annotations

import asyncio
import builtins
import gzip
import netrc
import ssl
import threading
from contextlib import asynccontextmanager
from importlib import import_module
from importlib.metadata import version

import anyio
import pytest
import test_mcp_proof as protocol_peer
from test_mcp_proof import LOCAL, _FakeServer, _Handler
from test_mcp_proof import fake_mcp as fake_mcp  # noqa: PLC0414 -- pytest fixture re-export

from skillevaluator.tier3 import mcp_proof as proof


def _native_stack():
    modern = int(version("mcp").split(".", 1)[0]) >= 2
    return (
        (import_module("httpx2"), import_module("httpcore2"))
        if modern
        else (import_module("httpx"), import_module("httpcore"))
    )


def test_two_stream_transport_uses_real_session(monkeypatch, fake_mcp):
    """A modern two-stream boundary must work even in locked SDK1 CI."""
    from mcp.client import streamable_http

    native = streamable_http.streamable_http_client

    @asynccontextmanager
    async def two_streams(url, *, http_client):
        async with native(url, http_client=http_client) as streams:
            yield streams[:2]

    monkeypatch.setattr(streamable_http, "streamable_http_client", two_streams)

    entry = proof.probe_mcp_server({"name": "docs", "url": f"{fake_mcp}/mcp"}, allowed_private_hosts=LOCAL)

    assert entry["status"] == "reachable-host", entry
    assert entry["tools"] == ["search", "get_page"]


def test_session_owns_constrained_client_from_installed_sdk_family(monkeypatch, fake_mcp):
    clients = []
    original = proof._capped_client

    def recording_client(headers, **kwargs):
        client = original(headers, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(proof, "_capped_client", recording_client)

    entry = proof.probe_mcp_server({"name": "docs", "url": f"{fake_mcp}/mcp"}, allowed_private_hosts=LOCAL)

    expected = import_module("httpx2" if int(version("mcp").split(".", 1)[0]) >= 2 else "httpx")
    assert entry["status"] == "reachable-host", entry
    assert len(clients) == 1
    assert isinstance(clients[0], expected.AsyncClient)
    assert clients[0].is_closed
    assert clients[0].trust_env is False
    assert clients[0].follow_redirects is False


@pytest.mark.parametrize("oversize", [False, True])
def test_real_protocol_preserves_sanitized_and_bounded_argument_schema(monkeypatch, fake_mcp, oversize):
    schema = (
        {"type": "object", "properties": {f"p{i}": {"type": "string", "enum": ["a" * 50]} for i in range(64)}}
        if oversize
        else {
            "type": "object",
            "description": "not exported",
            "required": ["count"],
            "properties": {"count": {"type": "integer", "minimum": 1, "description": "not exported"}},
        }
    )
    monkeypatch.setattr(protocol_peer, "TOOLS", [{"name": "search", "inputSchema": schema}])

    entry = proof.probe_mcp_server({"name": "docs", "url": f"{fake_mcp}/mcp"}, allowed_private_hosts=LOCAL)

    assert entry["status"] == "reachable-host", entry
    assert entry["tools"] == ["search"]
    if oversize:
        assert "input_schemas" not in entry
    else:
        assert entry["input_schemas"] == {
            "search": {
                "type": "object",
                "required": ["count"],
                "properties": {"count": {"type": "integer", "minimum": 1}},
            }
        }


@pytest.mark.parametrize("malformed_type", [[["x"]], [{"x": "string"}], [{}]])
def test_untrusted_root_schema_type_is_not_exported(malformed_type):
    assert proof._input_schema({"type": malformed_type}) is None


@pytest.mark.parametrize(
    ("transport", "method", "status"),
    [("http", "POST", 307), ("http", "POST", 308), ("sse", "GET", 302), ("sse", "POST", 307)],
)
def test_same_origin_redirect_never_reaches_functional_target(monkeypatch, fake_mcp, transport, method, status):
    """SDK2 follows these itself unless the constrained transport refuses them."""
    seen = []
    source = "/messages" if transport == "sse" and method == "POST" else "/same-origin-redirect"
    sink = "/messages-target" if source == "/messages" else "/sse" if transport == "sse" else "/mcp"
    original = getattr(_Handler, f"do_{method}")

    def redirect_or_record(self):
        seen.append(self.path)
        if self.path.startswith(source) and not self.path.startswith(sink):
            self.send_response(status)
            self.send_header("Location", sink)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        original(self)

    monkeypatch.setattr(_Handler, f"do_{method}", redirect_or_record)
    start_path = "/sse" if source == "/messages" else source

    entry = proof.probe_mcp_server(
        {"name": "redirect", "url": f"{fake_mcp}{start_path}", "transport": transport}, allowed_private_hosts=LOCAL
    )

    assert not any(path.startswith(sink) for path in seen), seen
    assert entry["status"] == "unreachable", entry


@pytest.mark.skipif(int(version("mcp").split(".", 1)[0]) < 2, reason="SDK1 itself requires legacy HTTPX")
@pytest.mark.parametrize("transport", ["http", "sse"])
def test_sdk2_protocol_does_not_require_legacy_http_packages(monkeypatch, fake_mcp, transport):
    original = builtins.__import__

    def deny_legacy(name, *args, **kwargs):
        if name in {"httpx", "httpcore"} or name.startswith(("httpx.", "httpcore.")):
            raise ImportError("legacy HTTP package deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_legacy)
    path = "/sse" if transport == "sse" else "/mcp"

    entry = proof.probe_mcp_server(
        {"name": "docs", "url": f"{fake_mcp}{path}", "transport": transport}, allowed_private_hosts=LOCAL
    )

    assert entry["status"] == "reachable-host", entry
    assert entry["tools"] == ["search", "get_page"]


@pytest.mark.parametrize("mode", ["gzip", "chunks"])
def test_native_transport_rejects_encoding_and_aggregate_oversize(monkeypatch, fake_mcp, mode):
    monkeypatch.setattr(proof, "MAX_RESPONSE_BYTES", 16_384)
    sent_encodings = []

    def oversized(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        sent_encodings.append(self.headers.get("Accept-Encoding"))
        self.protocol_version = "HTTP/1.1"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if mode == "gzip":
            data = gzip.compress(b"x" * 32_768)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(data)))
        else:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if mode == "gzip":
                self.wfile.write(data)
            else:
                for _ in range(3):
                    self.wfile.write(b"2000\r\n" + b"x" * 8192 + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass  # The constrained client can close as soon as it rejects the response.

    monkeypatch.setattr(_Handler, "do_POST", oversized)

    entry = proof.probe_mcp_server(
        {"name": "large", "url": f"{fake_mcp}/mcp", "headers": {"Accept-Encoding": "gzip"}},
        allowed_private_hosts=LOCAL,
    )

    assert sent_encodings == ["identity"]
    assert entry["status"] == "unreachable", entry
    assert ("content encoding must be identity" if mode == "gzip" else "response exceeded 16384 bytes") in entry[
        "detail"
    ]


def test_native_transport_ignores_proxy_and_netrc_credentials(monkeypatch, tmp_path, fake_mcp):
    path = tmp_path / "fixture.netrc"
    path.write_text("machine 127.0.0.1 login ambient-fixture password never-send-this\n")
    monkeypatch.setenv("NETRC", str(path))
    monkeypatch.setenv("HTTP_PROXY", fake_mcp)
    monkeypatch.setenv("ALL_PROXY", fake_mcp)
    monkeypatch.setenv("NO_PROXY", "")

    def refuse_netrc(*_args, **_kwargs):
        raise AssertionError("the probe must not read ambient netrc credentials")

    monkeypatch.setattr(netrc, "netrc", refuse_netrc)
    requests = []
    original = _Handler.do_POST

    def record(self):
        requests.append((self.path, self.headers.get("Authorization")))
        original(self)

    monkeypatch.setattr(_Handler, "do_POST", record)

    entry = proof.probe_mcp_server({"name": "docs", "url": f"{fake_mcp}/mcp"}, allowed_private_hosts=LOCAL)

    assert entry["status"] == "reachable-host", entry
    assert requests and all(path == "/mcp" and authorization is None for path, authorization in requests)


def test_off_origin_sse_endpoint_never_contacts_live_sink(monkeypatch, fake_mcp):
    hits = []
    release = threading.Event()
    original_get, original_post = _Handler.do_GET, _Handler.do_POST

    def endpoint(self):
        if self.path != "/sse-to-sink":
            original_get(self)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(f"event: endpoint\ndata: {fake_mcp}/messages\n\n".encode())
        self.wfile.flush()
        release.wait(5)

    def record(self):
        hits.append(self.path)
        original_post(self)

    monkeypatch.setattr(_Handler, "do_GET", endpoint)
    monkeypatch.setattr(_Handler, "do_POST", record)
    server = _FakeServer()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        entry = proof.probe_mcp_server(
            {"name": "sse", "url": f"http://127.0.0.1:{server.server_port}/sse-to-sink", "transport": "sse"},
            allowed_private_hosts=LOCAL,
        )
    finally:
        release.set()
        server.sse_queue.put(None)
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)

    assert hits == []
    assert entry["status"] == "unreachable", entry
    assert "origin" in entry["detail"]


def test_native_pinned_pool_preserves_host_and_verifying_tls_identity(monkeypatch):
    http, core = _native_stack()
    seen = {"request": b""}

    class Stream(core.AsyncNetworkStream):
        sent = False

        async def read(self, max_bytes, timeout=None):
            del max_bytes, timeout
            if self.sent:
                return b""
            self.sent = True
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"

        async def write(self, buffer, timeout=None):
            del timeout
            seen["request"] += buffer

        async def aclose(self):
            seen["closed"] = True

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            del timeout
            seen["tls"] = (server_hostname, ssl_context.check_hostname, ssl_context.verify_mode)
            return self

    class Backend(core.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, **_kwargs):
            seen["address"] = (host, port)
            return Stream()

        async def sleep(self, seconds):
            pass

    monkeypatch.setattr(core, "AnyIOBackend", Backend)

    async def fetch():
        async with proof._capped_client(
            {},
            url="https://mcp.example.invalid/mcp",
            addresses=("192.0.2.10",),
            max_bytes=1024,
            connect_timeout=1,
            read_timeout=1,
            http_stack=(http, core),
        ) as client:
            assert (await client.get("https://mcp.example.invalid/mcp")).text == "OK"

    anyio.run(fetch)

    assert seen["address"] == ("192.0.2.10", 443)
    assert seen["tls"] == ("mcp.example.invalid", True, ssl.CERT_REQUIRED)
    assert b"Host: mcp.example.invalid\r\n" in seen["request"]
    assert seen["closed"] is True


def test_native_transport_refuses_unknown_pool_layout(monkeypatch):
    http, core = _native_stack()

    class UnsupportedTransport:
        _pool = object()

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(http, "AsyncHTTPTransport", UnsupportedTransport)

    with pytest.raises(RuntimeError, match="cannot pin"):
        proof._capped_client(
            {},
            url="https://mcp.example.invalid/mcp",
            addresses=("192.0.2.10",),
            max_bytes=1024,
            connect_timeout=1,
            read_timeout=1,
            http_stack=(http, core),
        )


def test_cancelled_native_session_closes_constrained_client(monkeypatch, fake_mcp):
    started, release = threading.Event(), threading.Event()
    clients = []
    original = proof._capped_client

    def record_client(headers, **kwargs):
        client = original(headers, **kwargs)
        clients.append(client)
        return client

    def stall(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        started.set()
        release.wait(5)

    monkeypatch.setattr(proof, "_capped_client", record_client)
    monkeypatch.setattr(_Handler, "do_POST", stall)

    async def cancel():
        task = asyncio.create_task(
            proof._probe_session(f"{fake_mcp}/mcp", "streamable-http", {}, addresses=LOCAL, total_timeout=10)
        )
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(cancel())

    assert clients and all(client.is_closed for client in clients)
