# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVIDIA Build protocol translation without network access."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread

import pytest
from anthropic.types import RawMessageStreamEvent
from openai.types.responses import ResponsesServerEvent
from pydantic import TypeAdapter

from skillevaluator.tier3.harbor import nvidia_build_bridge as bridge
from skillevaluator.tier3.harbor.nvidia_build_bridge import (
    BUILD_CHAT_COMPLETIONS_PATH,
    HEALTH_PATH,
    BridgePayloadError,
    anthropic_to_chat_request,
    chat_completion_to_anthropic_events,
    chat_completion_to_responses_events,
    redact_bridge_text,
    responses_to_chat_request,
)

CHAT_TOOL_RESPONSE = {
    "id": "chatcmpl-1",
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
        }
    ],
}

CHAT_TEXT_RESPONSE = {
    "id": "chatcmpl-text-1",
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello."}}],
}

CUSTOM_EXEC_TOOL = {
    "type": "custom",
    "name": "exec",
    "description": "Execute shell commands.",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: /.+/",
    },
}

CHAT_CUSTOM_TOOL_RESPONSE = {
    "id": "chatcmpl-custom-1",
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-exec-1",
                        "type": "function",
                        "function": {"name": "exec", "arguments": '{"input":"pwd"}'},
                    }
                ],
            },
        }
    ],
}

NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "multi_agent_v1",
    "description": "Delegate work to parallel agents.",
    "tools": [
        {
            "type": "function",
            "name": "spawn_agent",
            "description": "Spawn one agent.",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        },
        CUSTOM_EXEC_TOOL,
    ],
}

WEB_SEARCH_TOOL = {
    "type": "web_search",
    "search_context_size": "medium",
    "user_location": {"type": "approximate", "country": "US"},
}

CHAT_NAMESPACE_TOOL_RESPONSE = {
    "id": "chatcmpl-namespace-1",
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-spawn-1",
                        "type": "function",
                        "function": {
                            "name": "multi_agent_v1__spawn_agent",
                            "arguments": '{"task":"inspect"}',
                        },
                    }
                ],
            },
        }
    ],
}


class _FakeBuildHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        request = json.loads(body)
        self.server.requests.append((self.path, dict(self.headers), request))  # type: ignore[attr-defined]

        if request.get("model") == "return-backend-error":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail":"backend failed"}')
            return

        if request.get("model") == "return-invalid-completion":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")
            return

        if request.get("model") == "return-large-response":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"content": "x" * (bridge.MAX_BACKEND_RESPONSE_BYTES + 1)}).encode("utf-8"))
            return

        if request.get("model") == "return-deep-json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(("[" * 2_000 + "]" * 2_000).encode())
            return

        if request.get("model") == "return-custom-tool-flow":
            has_tool_output = any(message.get("role") == "tool" for message in request.get("messages", []))
            response = CHAT_TEXT_RESPONSE if has_tool_output else CHAT_CUSTOM_TOOL_RESPONSE
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        if request.get("model") == "return-namespace-tool-call":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(CHAT_NAMESPACE_TOOL_RESPONSE).encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(CHAT_TEXT_RESPONSE).encode("utf-8"))

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class _UnauthenticatedOldBridgeHandler(BaseHTTPRequestHandler):
    """Model the vulnerable health responder formerly reachable on port 18080."""

    def do_GET(self) -> None:
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


@dataclass
class _BridgeServices:
    url: str
    build: ThreadingHTTPServer
    bridge_server: ThreadingHTTPServer
    bridge_thread: Thread
    port: int
    log_path: Path
    api_key: str
    readiness_token: str
    ready_file: Path


def _request(
    url: str,
    payload: object | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status, dict(response.headers), response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read().decode("utf-8")


def _local_test_transport(endpoint: str, body: bytes) -> bytes:
    if json.loads(body).get("model") == "return-transport-value-error":
        raise ValueError("simulated backend value error with nvapi-secret")
    if json.loads(body).get("model") == "return-recursion-backend":
        return b"{}"
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Authorization": "Bearer test-transport-only-key", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read()


def _read_raw_http_response(client: socket.socket) -> str:
    chunks: list[bytes] = []
    client.settimeout(2)
    while True:
        try:
            chunk = client.recv(4096)
        except ConnectionResetError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8")


def _sse_events(body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for frame in body.strip().split("\n\n"):
        event_line, data_line = frame.split("\n")
        assert event_line.startswith("event: ")
        assert data_line.startswith("data: ")
        event = json.loads(data_line.removeprefix("data: "))
        assert event["type"] == event_line.removeprefix("event: ")
        events.append(event)
    return events


def test_serve_rejects_an_empty_api_key_before_binding() -> None:
    config = bridge.BridgeConfig(
        api_key="",
        build_base_url="http://127.0.0.1:8080/v1",
        host="127.0.0.1",
        port=-1,
        log_path=Path("nvidia-build-bridge.log"),
        readiness_token="test-readiness-token",
    )

    with pytest.raises(ValueError, match="api_key must not be empty"):
        bridge.serve(config)


def test_bridge_config_supports_a_per_run_client_token() -> None:
    assert "client_token" in bridge.BridgeConfig.__dataclass_fields__


def test_serve_rejects_api_keys_with_line_breaks_before_binding() -> None:
    config = bridge.BridgeConfig(
        api_key="nvapi-secret\nsecond-header: injected",
        build_base_url=bridge.PRODUCTION_BUILD_BASE_URL,
        host="127.0.0.1",
        port=-1,
        log_path=Path("nvidia-build-bridge.log"),
        readiness_token="test-readiness-token",
    )

    with pytest.raises(ValueError, match="api_key must not contain CR or LF"):
        bridge.serve(config)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://integrate.api.nvidia.com/v1",
        "https://127.0.0.1/v1",
        "https://10.0.0.1/v1",
        "https://example.com/v1",
        "https://user:password@integrate.api.nvidia.com/v1",
        "https://integrate.api.nvidia.com:bad-port/v1",
        "https://integrate.api.nvidia.com/v1/../redirect",
    ],
)
def test_serve_rejects_non_production_build_origins_before_binding(base_url: str) -> None:
    config = bridge.BridgeConfig(
        api_key="nvapi-production-key",
        build_base_url=base_url,
        host="127.0.0.1",
        port=-1,
        log_path=Path("nvidia-build-bridge.log"),
        readiness_token="test-readiness-token",
    )

    with pytest.raises(ValueError, match="build_base_url"):
        bridge.serve(config)


def test_test_transport_rejects_a_production_api_key_before_binding() -> None:
    config = bridge.BridgeConfig(
        api_key="nvapi-production-key",
        build_base_url="http://127.0.0.1:8080/v1",
        host="127.0.0.1",
        port=-1,
        log_path=Path("nvidia-build-bridge.log"),
        readiness_token="test-readiness-token",
        request_transport=_local_test_transport,
    )

    with pytest.raises(ValueError, match="test transport"):
        bridge.serve(config)


def test_bridge_rejects_a_request_budget_above_the_fixed_security_limit() -> None:
    config = bridge.BridgeConfig(
        api_key="test-budget-key",
        build_base_url="http://127.0.0.1:8080/v1",
        host="127.0.0.1",
        port=-1,
        log_path=Path("nvidia-build-bridge.log"),
        readiness_token="test-readiness-token",
        request_transport=_local_test_transport,
        max_requests=bridge.MAX_REQUESTS_PER_BRIDGE + 1,
    )

    with pytest.raises(ValueError, match="max_requests"):
        bridge.serve(config)


@pytest.mark.parametrize("value", [True, 0, bridge.MAX_OUTPUT_TOKENS_PER_REQUEST + 1])
def test_bridge_rejects_an_invalid_configured_output_token_ceiling(value: object) -> None:
    config = bridge.BridgeConfig(
        api_key="test-token-ceiling-key",
        build_base_url="http://127.0.0.1:8080/v1",
        host="127.0.0.1",
        port=0,
        log_path=Path("nvidia-build-bridge.log"),
        readiness_token="test-readiness-token",
        request_transport=_local_test_transport,
        max_output_tokens=value,  # type: ignore[arg-type]
    )
    server = None
    try:
        with pytest.raises(ValueError, match="max_output_tokens"):
            server = bridge._create_bridge_server(config)
    finally:
        if server is not None:
            server.server_close()


@pytest.fixture
def bridge_services(tmp_path: Path) -> _BridgeServices:
    build = ThreadingHTTPServer(("127.0.0.1", 0), _FakeBuildHandler)
    build.requests = []  # type: ignore[attr-defined]
    build_thread = Thread(target=build.serve_forever, daemon=True)
    build_thread.start()

    api_key = "test-config-key-for-log-test"
    readiness_token = "test-readiness-token"
    log_path = tmp_path / "nvidia-build-bridge.log"
    ready_file = tmp_path / "nvidia-build-bridge.ready"
    config = bridge.BridgeConfig(
        api_key=api_key,
        build_base_url=f"http://127.0.0.1:{build.server_port}/v1",
        host="127.0.0.1",
        port=0,
        log_path=log_path,
        readiness_token=readiness_token,
        request_transport=_local_test_transport,
    )
    started = Event()
    servers: list[ThreadingHTTPServer] = []

    def on_server_ready(server: ThreadingHTTPServer) -> None:
        servers.append(server)
        started.set()

    bridge_thread = Thread(
        target=bridge.serve,
        args=(config,),
        kwargs={"ready_file": ready_file, "on_server_ready": on_server_ready},
        daemon=True,
    )
    bridge_thread.start()
    assert started.wait(timeout=1)
    port = int(servers[0].server_port)
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            if (
                _request(
                    f"{base_url}/healthz",
                    headers={bridge.READINESS_TOKEN_HEADER: readiness_token},
                )[0]
                == 200
            ):
                break
        except urllib.error.URLError:
            time.sleep(0.025)
    else:
        pytest.fail("bridge server did not become healthy")

    yield _BridgeServices(
        base_url,
        build,
        servers[0],
        bridge_thread,
        port,
        log_path,
        api_key,
        readiness_token,
        ready_file,
    )
    servers[0].shutdown()
    bridge_thread.join(timeout=2)
    assert not bridge_thread.is_alive()
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    build.shutdown()
    build.server_close()
    build_thread.join(timeout=2)


def test_healthz_is_available_from_the_loopback_bridge(bridge_services: _BridgeServices) -> None:
    status, headers, body = _request(
        f"{bridge_services.url}/healthz",
        headers={bridge.READINESS_TOKEN_HEADER: bridge_services.readiness_token},
    )

    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert json.loads(body) == {"status": "ok"}


def test_healthz_rejects_requests_without_the_per_run_readiness_token(
    bridge_services: _BridgeServices,
) -> None:
    missing_status, _missing_headers, _missing_body = _request(f"{bridge_services.url}/healthz")
    wrong_status, _wrong_headers, _wrong_body = _request(
        f"{bridge_services.url}/healthz",
        headers={bridge.READINESS_TOKEN_HEADER: "wrong-token"},
    )

    assert missing_status == 403
    assert wrong_status == 403


def test_host_bridge_requires_the_per_run_client_token_before_forwarding(tmp_path: Path) -> None:
    backend_requests: list[dict[str, object]] = []

    def transport(_endpoint: str, body: bytes) -> bytes:
        backend_requests.append(json.loads(body))
        return json.dumps(CHAT_TEXT_RESPONSE).encode("utf-8")

    client_token = "per-run-client-token"
    config = bridge.BridgeConfig(
        api_key="test-host-bridge-key",
        build_base_url="http://127.0.0.1:8080/v1",
        host="127.0.0.1",
        port=0,
        log_path=tmp_path / "host-bridge.log",
        readiness_token="host-readiness-token",
        request_transport=transport,
        client_token=client_token,
        allowed_model="nvidia/model",
        max_requests=10,
    )
    started = Event()
    servers: list[ThreadingHTTPServer] = []

    def on_server_ready(server: ThreadingHTTPServer) -> None:
        servers.append(server)
        started.set()

    thread = Thread(
        target=bridge.serve,
        args=(config,),
        kwargs={"on_server_ready": on_server_ready},
        daemon=True,
    )
    thread.start()
    assert started.wait(timeout=1)
    origin = f"http://127.0.0.1:{servers[0].server_port}"
    try:
        responses_payload = {"model": "nvidia/model", "input": "Hello"}
        messages_payload = {
            "model": "nvidia/model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        assert _request(f"{origin}/v1/responses", responses_payload)[0] == 403
        assert (
            _request(
                f"{origin}/v1/responses",
                responses_payload,
                headers={"Authorization": "Bearer wrong-token"},
            )[0]
            == 403
        )
        assert _request(f"{origin}/v1/messages", messages_payload, headers={"x-api-key": "wrong-token"})[0] == 403
        assert backend_requests == []

        assert (
            _request(
                f"{origin}/v1/responses",
                {"model": "nvidia/alternate-model", "input": "Hello"},
                headers={"Authorization": f"Bearer {client_token}"},
            )[0]
            == 403
        )
        assert backend_requests == []

        assert (
            _request(
                f"{origin}/v1/responses",
                responses_payload,
                headers={"Authorization": f"Bearer {client_token}"},
            )[0]
            == 200
        )
        assert (
            _request(
                f"{origin}/v1/messages",
                messages_payload,
                headers={"x-api-key": client_token},
            )[0]
            == 200
        )
        assert (
            _request(
                f"{origin}/v1/messages/count_tokens",
                messages_payload,
                headers={"x-api-key": client_token},
            )[0]
            == 200
        )
        assert len(backend_requests) == 2
    finally:
        servers[0].shutdown()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_bridge_rejects_authenticated_requests_after_per_run_budget_is_exhausted(tmp_path: Path) -> None:
    backend_requests: list[dict[str, object]] = []

    def transport(_endpoint: str, body: bytes) -> bytes:
        backend_requests.append(json.loads(body))
        return json.dumps(CHAT_TEXT_RESPONSE).encode("utf-8")

    client_token = "per-run-client-token"
    config = bridge.BridgeConfig(
        api_key="test-budget-key",
        build_base_url="http://127.0.0.1:8080/v1",
        host="127.0.0.1",
        port=0,
        log_path=tmp_path / "budget-bridge.log",
        readiness_token="budget-readiness-token",
        request_transport=transport,
        client_token=client_token,
        allowed_model="nvidia/model",
        max_requests=2,
    )
    started = Event()
    servers: list[ThreadingHTTPServer] = []

    def on_server_ready(server: ThreadingHTTPServer) -> None:
        servers.append(server)
        started.set()

    thread = Thread(
        target=bridge.serve,
        args=(config,),
        kwargs={"on_server_ready": on_server_ready},
        daemon=True,
    )
    thread.start()
    assert started.wait(timeout=1)
    origin = f"http://127.0.0.1:{servers[0].server_port}"
    headers = {"Authorization": f"Bearer {client_token}"}
    payload = {"model": "nvidia/model", "input": "Hello"}
    try:
        assert _request(f"{origin}/v1/responses", payload, headers=headers)[0] == 200
        assert _request(f"{origin}/v1/responses", payload, headers=headers)[0] == 200
        status, _response_headers, body = _request(f"{origin}/v1/responses", payload, headers=headers)

        assert status == 429
        assert json.loads(body) == {
            "error": {
                "type": "rate_limit",
                "message": "bridge request budget exhausted",
            }
        }
        assert len(backend_requests) == 2
    finally:
        servers[0].shutdown()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_in_process_bridges_use_distinct_capabilities_and_close_cleanly(tmp_path: Path) -> None:
    def transport(_endpoint: str, _body: bytes) -> bytes:
        return json.dumps(CHAT_TEXT_RESPONSE).encode("utf-8")

    first = bridge.start_in_process_bridge(
        api_key="test-first-key",
        build_base_url="http://127.0.0.1:8080/v1",
        log_path=tmp_path / "first.log",
        request_transport=transport,
    )
    second = bridge.start_in_process_bridge(
        api_key="test-second-key",
        build_base_url="http://127.0.0.1:8081/v1",
        log_path=tmp_path / "second.log",
        request_transport=transport,
    )
    first_port = int(urllib.parse.urlsplit(first.origin).port or 0)
    second_port = int(urllib.parse.urlsplit(second.origin).port or 0)
    payload = {"model": "nvidia/model", "input": "Hello"}
    try:
        assert first.origin != second.origin
        assert first.client_token != second.client_token
        assert (
            _request(
                f"{first.origin}/v1/responses",
                payload,
                headers={"Authorization": f"Bearer {first.client_token}"},
            )[0]
            == 200
        )
        assert (
            _request(
                f"{first.origin}/v1/responses",
                payload,
                headers={"Authorization": f"Bearer {second.client_token}"},
            )[0]
            == 403
        )
    finally:
        first.close()
        second.close()
        first.close()

    for port in (first_port, second_port):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))


def test_in_process_bridge_enforces_its_configured_output_token_ceiling(tmp_path: Path) -> None:
    backend_requests: list[dict[str, object]] = []

    def transport(_endpoint: str, body: bytes) -> bytes:
        backend_requests.append(json.loads(body))
        return json.dumps(CHAT_TEXT_RESPONSE).encode("utf-8")

    running = bridge.start_in_process_bridge(
        api_key="test-token-limit-key",
        build_base_url="http://127.0.0.1:8080/v1",
        log_path=tmp_path / "token-limit.log",
        request_transport=transport,
        max_output_tokens=8,
    )
    headers = {"Authorization": f"Bearer {running.client_token}"}
    try:
        for path, field, payload in (
            ("/v1/responses", "max_output_tokens", {"model": "nvidia/model", "input": "Hello"}),
            ("/v1/messages", "max_tokens", {"model": "nvidia/model", "messages": []}),
        ):
            rejected = {**payload, field: 9}
            accepted = {**payload, field: 8}

            assert _request(f"{running.origin}{path}", rejected, headers=headers)[0] == 400
            assert _request(f"{running.origin}{path}", accepted, headers=headers)[0] == 200
    finally:
        running.close()

    assert [request["max_tokens"] for request in backend_requests] == [8, 8]


def test_build_request_timeout_allows_slow_agent_models() -> None:
    assert bridge.BACKEND_CONNECT_TIMEOUT_SECONDS >= 120


def test_backend_dns_resolution_obeys_deadline_without_late_request_or_resolver_growth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend_received = Event()
    backend_authorizations: list[str | None] = []
    resolver_calls = 0

    class BuildHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            backend_authorizations.append(self.headers.get("Authorization"))
            backend_received.set()
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), BuildHandler)
    backend_thread = Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()

    def slow_resolution(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        nonlocal resolver_calls
        resolver_calls += 1
        time.sleep(0.5)
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", backend.server_port),
            )
        ]

    config = bridge.BridgeConfig(
        api_key="credential-must-not-reach-backend",
        build_base_url=bridge.PRODUCTION_BUILD_BASE_URL,
        host="127.0.0.1",
        port=0,
        log_path=tmp_path / "backend-dns-deadline.log",
        readiness_token="backend-dns-deadline-token",
        client_token="client-token",
        allowed_model="nvidia/model",
    )
    monkeypatch.setattr(
        bridge,
        "_build_endpoint",
        lambda _config: f"http://slow-resolver.example:{backend.server_port}{BUILD_CHAT_COMPLETIONS_PATH}",
    )
    monkeypatch.setattr(bridge, "BACKEND_CONNECT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(bridge.socket, "getaddrinfo", slow_resolution)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")

    elapsed: list[float] = []
    try:
        for _ in range(2):
            started = time.monotonic()
            with pytest.raises(bridge._BackendError) as caught:
                bridge._request_build(config, {"model": "nvidia/model", "messages": []})
            elapsed.append(time.monotonic() - started)
            assert caught.value.category == "timeout"
            assert caught.value.status_code == 504
        time.sleep(0.55)
    finally:
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)

    assert all(duration < 0.2 for duration in elapsed)
    assert resolver_calls == 1, "a stuck resolver must not create an unbounded thread or work queue"
    assert backend_received.is_set() is False
    assert backend_authorizations == []


def test_backend_reads_a_complete_content_length_response_without_reusing_the_closed_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = json.dumps(CHAT_TEXT_RESPONSE).encode("utf-8")

    class BuildHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), BuildHandler)
    backend_thread = Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    config = bridge.BridgeConfig(
        api_key="test-content-length-key",
        build_base_url=bridge.PRODUCTION_BUILD_BASE_URL,
        host="127.0.0.1",
        port=0,
        log_path=tmp_path / "content-length.log",
        readiness_token="content-length-token",
        client_token="client-token",
        allowed_model="nvidia/model",
    )
    monkeypatch.setattr(
        bridge,
        "_build_endpoint",
        lambda _config: f"http://127.0.0.1:{backend.server_port}{BUILD_CHAT_COMPLETIONS_PATH}",
    )
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")

    try:
        response = bridge._request_build(config, {"model": "nvidia/model", "messages": []})
    finally:
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)

    assert response == CHAT_TEXT_RESPONSE


def test_backend_slow_drip_obeys_an_absolute_deadline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class SlowBuildHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b"null"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for index, byte in enumerate(body):
                if index:
                    time.sleep(0.03)
                try:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                except OSError:
                    break

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), SlowBuildHandler)
    backend_thread = Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    config = bridge.BridgeConfig(
        api_key="test-backend-deadline-key",
        build_base_url=bridge.PRODUCTION_BUILD_BASE_URL,
        host="127.0.0.1",
        port=0,
        log_path=tmp_path / "backend-deadline.log",
        readiness_token="backend-deadline-token",
        client_token="client-token",
        allowed_model="nvidia/model",
    )
    monkeypatch.setattr(
        bridge,
        "_build_endpoint",
        lambda _config: f"http://127.0.0.1:{backend.server_port}{BUILD_CHAT_COMPLETIONS_PATH}",
    )
    monkeypatch.setattr(bridge, "BACKEND_CONNECT_TIMEOUT_SECONDS", 0.05)
    try:
        with pytest.raises(bridge._BackendError) as caught:
            bridge._request_build(config, {"model": "nvidia/model", "messages": []})
    finally:
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)

    assert caught.value.category == "timeout"
    assert caught.value.status_code == 504


def test_backend_slow_drip_response_headers_are_interrupted_at_the_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class SlowBuildHeadersHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            response = b"".join(
                (
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-Slow: ",
                    b"x" * 160,
                    b"\r\nContent-Length: 4\r\n\r\nnull",
                )
            )
            for byte in response:
                try:
                    self.connection.sendall(bytes([byte]))
                except OSError:
                    break
                time.sleep(0.01)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), SlowBuildHeadersHandler)
    backend_thread = Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    config = bridge.BridgeConfig(
        api_key="test-backend-header-deadline-key",
        build_base_url=bridge.PRODUCTION_BUILD_BASE_URL,
        host="127.0.0.1",
        port=0,
        log_path=tmp_path / "backend-header-deadline.log",
        readiness_token="backend-header-deadline-token",
        client_token="client-token",
        allowed_model="nvidia/model",
    )
    monkeypatch.setattr(
        bridge,
        "_build_endpoint",
        lambda _config: f"http://127.0.0.1:{backend.server_port}{BUILD_CHAT_COMPLETIONS_PATH}",
    )
    monkeypatch.setattr(bridge, "BACKEND_CONNECT_TIMEOUT_SECONDS", 0.05)
    started = time.monotonic()
    try:
        with pytest.raises(bridge._BackendError) as caught:
            bridge._request_build(config, {"model": "nvidia/model", "messages": []})
    finally:
        elapsed = time.monotonic() - started
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)

    assert caught.value.category == "timeout"
    assert caught.value.status_code == 504
    assert elapsed < 0.5


def test_in_process_close_waits_for_active_backend_request(tmp_path: Path) -> None:
    backend_started = Event()
    release_backend = Event()
    close_finished = Event()

    def transport(_endpoint: str, _body: bytes) -> bytes:
        backend_started.set()
        assert release_backend.wait(timeout=5)
        return json.dumps(CHAT_TEXT_RESPONSE).encode("utf-8")

    running = bridge.start_in_process_bridge(
        api_key="test-worker-key",
        build_base_url="http://127.0.0.1:8080/v1",
        log_path=tmp_path / "worker.log",
        request_transport=transport,
    )
    request_result: list[int] = []
    request_aborted = Event()

    def make_request() -> None:
        try:
            request_result.append(
                _request(
                    f"{running.origin}/v1/responses",
                    {"model": "nvidia/model", "input": "Hello"},
                    headers={"Authorization": f"Bearer {running.client_token}"},
                )[0]
            )
        except (OSError, RemoteDisconnected, urllib.error.URLError):
            request_aborted.set()

    request_thread = Thread(target=make_request)
    close_thread = Thread(target=lambda: (running.close(), close_finished.set()))
    request_thread.start()
    assert backend_started.wait(timeout=2)
    close_thread.start()
    try:
        assert not close_finished.wait(timeout=0.8)
    finally:
        release_backend.set()
    request_thread.join(timeout=2)
    close_thread.join(timeout=2)

    assert request_result == []
    assert request_aborted.is_set()
    assert close_finished.is_set()
    assert not request_thread.is_alive()
    assert not close_thread.is_alive()


def test_in_process_close_force_drains_a_slow_header_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def transport(_endpoint: str, _body: bytes) -> bytes:
        return json.dumps(CHAT_TEXT_RESPONSE).encode("utf-8")

    running = bridge.start_in_process_bridge(
        api_key="test-slow-header-close-key",
        build_base_url="http://127.0.0.1:8080/v1",
        log_path=tmp_path / "slow-header-close.log",
        request_transport=transport,
    )
    port = int(urllib.parse.urlsplit(running.origin).port or 0)
    monkeypatch.setattr(bridge, "REQUEST_HEADER_TIMEOUT_SECONDS", 60.0)
    monkeypatch.setattr(bridge, "BACKEND_CONNECT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(bridge, "IN_PROCESS_START_TIMEOUT_SECONDS", 0.05)
    stop_sending = Event()
    client = socket.create_connection(("127.0.0.1", port), timeout=2)

    def drip_headers() -> None:
        while not stop_sending.is_set():
            try:
                client.sendall(b"P")
            except OSError:
                return
            time.sleep(0.015)

    sender = Thread(target=drip_headers)
    sender.start()
    close_error: BaseException | None = None
    try:
        deadline = time.monotonic() + 1
        while running._server.active_workers == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert running._server.active_workers == 1
        try:
            running.close()
        except BaseException as error:
            close_error = error
    finally:
        stop_sending.set()
        client.close()
        sender.join(timeout=1)
        assert not sender.is_alive()
        if not running._closed:
            running.close()

    assert close_error is None
    assert running._closed is True
    assert running._server.active_workers == 0
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def test_ready_file_records_dynamic_origin_and_is_private(bridge_services: _BridgeServices) -> None:
    ready = json.loads(bridge_services.ready_file.read_text(encoding="utf-8"))

    assert bridge_services.port != 18080
    assert ready == {
        "origin": bridge_services.url,
        "readiness_token": bridge_services.readiness_token,
    }
    assert not bridge_services.ready_file.is_symlink()
    assert bridge_services.ready_file.stat().st_mode & 0o777 == 0o600


def test_dynamic_bridge_ignores_prebound_old_port_and_checker_authenticates_health(tmp_path: Path) -> None:
    attacker = ThreadingHTTPServer(("127.0.0.1", 18080), _UnauthenticatedOldBridgeHandler)
    attacker_thread = Thread(target=attacker.serve_forever, daemon=True)
    attacker_thread.start()
    key_file = tmp_path / "nvidia-build-api-key"
    key_file.write_text("nvapi-test-secret", encoding="utf-8")
    key_file.chmod(0o600)
    client_token_file = tmp_path / "nvidia-build-client-token"
    client_token_file.write_text("per-run-client-token", encoding="utf-8")
    client_token_file.chmod(0o600)
    ready_file = tmp_path / "nvidia-build-bridge.ready"
    log_path = tmp_path / "nvidia-build-bridge.log"
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(bridge.__file__)),
                "--build-base-url",
                bridge.PRODUCTION_BUILD_BASE_URL,
                "--api-key-file",
                str(key_file),
                "--client-token-file",
                str(client_token_file),
                "--allowed-model",
                "nvidia/model",
                "--port",
                "0",
                "--ready-file",
                str(ready_file),
                "--log-path",
                str(log_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 5
        while not ready_file.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.025)
        assert process.poll() is None, process.communicate(timeout=1)
        assert ready_file.exists()
        ready = json.loads(ready_file.read_text(encoding="utf-8"))
        origin = ready["origin"]
        readiness_token = ready["readiness_token"]

        assert origin != "http://127.0.0.1:18080"
        assert _request("http://127.0.0.1:18080/healthz")[0] == 200
        assert _request(f"{origin}/healthz")[0] == 403
        assert (
            _request(
                f"{origin}/healthz",
                headers={bridge.READINESS_TOKEN_HEADER: readiness_token},
            )[0]
            == 200
        )

        checked = subprocess.run(
            [sys.executable, str(Path(bridge.__file__)), "--check-ready-file", str(ready_file)],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert checked.returncode == 0, checked.stderr
        assert checked.stdout.strip() == origin
        assert readiness_token not in checked.stdout
        assert readiness_token not in checked.stderr
        assert not key_file.exists()
        assert not client_token_file.exists()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        attacker.shutdown()
        attacker.server_close()
        attacker_thread.join(timeout=2)


def test_bridge_ready_file_creation_refuses_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "attacker-controlled-target"
    target.write_text("do-not-overwrite", encoding="utf-8")
    ready_file = tmp_path / "nvidia-build-bridge.ready"
    ready_file.symlink_to(target)
    key_file = tmp_path / "nvidia-build-api-key"
    api_key = "nvapi-must-not-appear-in-diagnostics"
    key_file.write_text(api_key, encoding="utf-8")
    key_file.chmod(0o600)
    client_token_file = tmp_path / "nvidia-build-client-token"
    client_token_file.write_text("per-run-client-token", encoding="utf-8")
    client_token_file.chmod(0o600)
    log_path = tmp_path / "nvidia-build-bridge.log"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(bridge.__file__)),
            "--build-base-url",
            bridge.PRODUCTION_BUILD_BASE_URL,
            "--api-key-file",
            str(key_file),
            "--client-token-file",
            str(client_token_file),
            "--allowed-model",
            "nvidia/model",
            "--port",
            "0",
            "--ready-file",
            str(ready_file),
            "--log-path",
            str(log_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )

    assert result.returncode != 0
    assert not key_file.exists()
    assert not client_token_file.exists()
    assert ready_file.is_symlink()
    assert target.read_text(encoding="utf-8") == "do-not-overwrite"
    assert api_key not in result.stdout
    assert api_key not in result.stderr


def test_responses_bridge_forwards_only_to_build_and_returns_sse(bridge_services: _BridgeServices) -> None:
    status, headers, body = _request(
        f"{bridge_services.url}/v1/responses",
        {"model": "nvidia/model", "input": "Hello"},
    )

    assert status == 200
    assert headers["Content-Type"].startswith("text/event-stream")
    assert _sse_events(body)[-1]["type"] == "response.completed"
    path, request_headers, _request_body = bridge_services.build.requests[0]  # type: ignore[attr-defined]
    assert path == "/v1/chat/completions"
    assert request_headers["Authorization"] == "Bearer test-transport-only-key"
    assert bridge_services.api_key not in request_headers["Authorization"]


def test_responses_custom_tool_round_trip_replays_full_codex_input(bridge_services: _BridgeServices) -> None:
    user_message = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Run pwd."}],
    }
    request_base = {
        "model": "return-custom-tool-flow",
        "tools": [CUSTOM_EXEC_TOOL],
        "store": False,
    }

    first_status, _first_headers, first_body = _request(
        f"{bridge_services.url}/v1/responses",
        {**request_base, "input": [user_message]},
    )
    assert first_status == 200
    first_events = _sse_events(first_body)
    custom_call = next(
        event["item"]
        for event in first_events
        if event["type"] == "response.output_item.done" and event["item"]["type"] == "custom_tool_call"
    )
    second_status, _second_headers, second_body = _request(
        f"{bridge_services.url}/v1/responses",
        {
            **request_base,
            "input": [
                user_message,
                custom_call,
                {
                    "type": "custom_tool_call_output",
                    "call_id": custom_call["call_id"],
                    "name": custom_call["name"],
                    "output": "workspace output",
                },
            ],
        },
    )

    assert second_status == 200
    assert _sse_events(second_body)[-1]["type"] == "response.completed"
    assert len(bridge_services.build.requests) == 2  # type: ignore[attr-defined]
    second_build_request = bridge_services.build.requests[1][2]  # type: ignore[attr-defined]
    assert second_build_request["messages"] == [
        {"role": "user", "content": "Run pwd."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-exec-1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": '{"input":"pwd"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-exec-1", "content": "workspace output"},
    ]
    log_text = bridge_services.log_path.read_text(encoding="utf-8")
    assert "Run pwd." not in log_text
    assert "workspace output" not in log_text
    assert bridge_services.api_key not in log_text


def test_responses_custom_tool_names_are_request_local(bridge_services: _BridgeServices) -> None:
    custom_status, _custom_headers, custom_body = _request(
        f"{bridge_services.url}/v1/responses",
        {
            "model": "return-custom-tool-flow",
            "input": "First request.",
            "tools": [CUSTOM_EXEC_TOOL],
            "store": False,
        },
    )
    function_status, _function_headers, function_body = _request(
        f"{bridge_services.url}/v1/responses",
        {
            "model": "return-custom-tool-flow",
            "input": "Second request.",
            "tools": [
                {
                    "type": "function",
                    "name": "exec",
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                    },
                }
            ],
            "store": False,
        },
    )

    assert custom_status == 200
    assert function_status == 200
    custom_done = next(event for event in _sse_events(custom_body) if event["type"] == "response.output_item.done")
    function_done = next(event for event in _sse_events(function_body) if event["type"] == "response.output_item.done")
    assert custom_done["item"]["type"] == "custom_tool_call"
    assert function_done["item"]["type"] == "function_call"


def test_responses_namespace_mapping_is_request_local_over_http(bridge_services: _BridgeServices) -> None:
    namespace_status, _namespace_headers, namespace_body = _request(
        f"{bridge_services.url}/v1/responses",
        {
            "model": "return-namespace-tool-call",
            "input": "Delegate this.",
            "tools": [NAMESPACE_TOOL],
            "store": False,
        },
    )
    plain_status, _plain_headers, plain_body = _request(
        f"{bridge_services.url}/v1/responses",
        {
            "model": "return-namespace-tool-call",
            "input": "Use the plain flattened function.",
            "tools": [
                {
                    "type": "function",
                    "name": "multi_agent_v1__spawn_agent",
                    "parameters": {"type": "object"},
                }
            ],
            "store": False,
        },
    )

    assert namespace_status == 200
    assert plain_status == 200
    namespace_done = next(
        event for event in _sse_events(namespace_body) if event["type"] == "response.output_item.done"
    )
    plain_done = next(event for event in _sse_events(plain_body) if event["type"] == "response.output_item.done")
    assert namespace_done["item"] == {
        "type": "function_call",
        "id": "call-spawn-1",
        "call_id": "call-spawn-1",
        "name": "spawn_agent",
        "namespace": "multi_agent_v1",
        "arguments": '{"task":"inspect"}',
        "status": "completed",
    }
    assert plain_done["item"]["name"] == "multi_agent_v1__spawn_agent"
    assert "namespace" not in plain_done["item"]
    build_request = bridge_services.build.requests[0][2]  # type: ignore[attr-defined]
    assert [tool["function"]["name"] for tool in build_request["tools"]] == [
        "multi_agent_v1__spawn_agent",
        "multi_agent_v1__exec",
    ]
    log_text = bridge_services.log_path.read_text(encoding="utf-8")
    assert "Delegate this." not in log_text
    assert "Use the plain flattened function." not in log_text
    assert bridge_services.api_key not in log_text


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_responses_mixed_web_search_request_forwards_only_executable_tools(
    bridge_services: _BridgeServices, tool_choice: str
) -> None:
    status, _headers, body = _request(
        f"{bridge_services.url}/v1/responses",
        {
            "model": "return-namespace-tool-call",
            "input": "Delegate without server search.",
            "tools": [WEB_SEARCH_TOOL, CUSTOM_EXEC_TOOL, NAMESPACE_TOOL],
            "tool_choice": tool_choice,
            "store": False,
        },
    )

    assert status == 200
    done = next(event for event in _sse_events(body) if event["type"] == "response.output_item.done")
    assert done["item"]["name"] == "spawn_agent"
    assert done["item"]["namespace"] == "multi_agent_v1"
    build_request = bridge_services.build.requests[0][2]  # type: ignore[attr-defined]
    assert [tool["function"]["name"] for tool in build_request["tools"]] == [
        "exec",
        "multi_agent_v1__spawn_agent",
        "multi_agent_v1__exec",
    ]
    assert build_request["tool_choice"] == tool_choice
    log_text = bridge_services.log_path.read_text(encoding="utf-8")
    assert "Delegate without server search." not in log_text
    assert bridge_services.api_key not in log_text


@pytest.mark.parametrize("tool_type", ["web_search", "web_search_preview"])
def test_forced_server_web_search_choice_is_a_structured_400(bridge_services: _BridgeServices, tool_type: str) -> None:
    status, _headers, body = _request(
        f"{bridge_services.url}/v1/responses",
        {
            "model": "nvidia/model",
            "input": "Search.",
            "tools": [{"type": tool_type}],
            "tool_choice": {"type": tool_type},
        },
    )

    assert status == 400
    assert json.loads(body) == {
        "error": {
            "type": "invalid_request",
            "message": f"forced Responses {tool_type} tool_choice is unsupported by NVIDIA Build",
        }
    }
    assert bridge_services.build.requests == []  # type: ignore[attr-defined]


def test_required_choice_with_only_server_web_search_is_a_structured_400(
    bridge_services: _BridgeServices,
) -> None:
    status, _headers, body = _request(
        f"{bridge_services.url}/v1/responses",
        {
            "model": "nvidia/model",
            "input": "Search.",
            "tools": [WEB_SEARCH_TOOL],
            "tool_choice": "required",
        },
    )

    assert status == 400
    assert json.loads(body) == {
        "error": {
            "type": "invalid_request",
            "message": "Responses tool_choice 'required' needs an executable tool for NVIDIA Build",
        }
    }
    assert bridge_services.build.requests == []  # type: ignore[attr-defined]


def test_messages_bridge_returns_anthropic_sse_and_message_stop(bridge_services: _BridgeServices) -> None:
    status, headers, body = _request(
        f"{bridge_services.url}/v1/messages",
        {"model": "nvidia/model", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert status == 200
    assert headers["Content-Type"].startswith("text/event-stream")
    assert _sse_events(body)[-1]["type"] == "message_stop"
    assert bridge_services.build.requests[0][0] == "/v1/chat/completions"  # type: ignore[attr-defined]


def test_messages_count_tokens_does_not_call_build(bridge_services: _BridgeServices) -> None:
    status, headers, body = _request(f"{bridge_services.url}/v1/messages/count_tokens", {"messages": []})

    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert json.loads(body) == {"input_tokens": 0}
    assert bridge_services.build.requests == []  # type: ignore[attr-defined]


def test_malformed_json_is_a_safe_structured_client_error(bridge_services: _BridgeServices) -> None:
    request = urllib.request.Request(
        f"{bridge_services.url}/v1/responses",
        data=b"{not-json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=2)

    assert caught.value.code == 400
    assert json.loads(caught.value.read()) == {"error": {"type": "invalid_request", "message": "invalid JSON payload"}}


def test_client_json_recursion_is_a_safe_structured_client_error(
    bridge_services: _BridgeServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_loads = json.loads

    def recursive_loads(value: object, *args: object, **kwargs: object) -> object:
        if value == "[]":
            raise RecursionError("simulated client JSON recursion")
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(bridge.json, "loads", recursive_loads)
    request = urllib.request.Request(
        f"{bridge_services.url}/v1/responses",
        data=b"[]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=2)

    assert caught.value.code == 400
    monkeypatch.undo()
    assert json.loads(caught.value.read())["error"]["type"] == "invalid_request"


def test_unknown_post_is_a_safe_404_without_parsing_a_body(bridge_services: _BridgeServices) -> None:
    with socket.create_connection(("127.0.0.1", bridge_services.port), timeout=2) as client:
        client.sendall(b"POST /unknown HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        response = _read_raw_http_response(client)

    status_line, body = response.split("\r\n\r\n", maxsplit=1)
    assert " 404 " in status_line
    assert json.loads(body)["error"]["type"] == "not_found"


def test_backend_error_is_safe_and_logs_never_expose_api_key(bridge_services: _BridgeServices) -> None:
    status, _headers, body = _request(
        f"{bridge_services.url}/v1/responses",
        {"model": "return-backend-error", "input": "Hello"},
    )

    assert status == 502
    assert json.loads(body)["error"]["type"] == "backend_error"
    log_text = bridge_services.log_path.read_text(encoding="utf-8")
    assert bridge_services.api_key not in log_text


def test_transport_value_error_is_a_safe_gateway_error(bridge_services: _BridgeServices) -> None:
    status, _headers, body = _request(
        f"{bridge_services.url}/v1/responses",
        {"model": "return-transport-value-error", "input": "Hello"},
    )

    assert status == 502
    assert "nvapi-secret" not in body
    assert json.loads(body)["error"]["type"] == "backend_error"


def test_invalid_backend_completion_is_a_safe_gateway_error(bridge_services: _BridgeServices) -> None:
    status, _headers, body = _request(
        f"{bridge_services.url}/v1/responses",
        {"model": "return-invalid-completion", "input": "Hello"},
    )

    assert status == 502
    assert json.loads(body)["error"]["type"] == "backend_error"


def test_oversized_request_is_a_structured_413(bridge_services: _BridgeServices) -> None:
    with socket.create_connection(("127.0.0.1", bridge_services.port), timeout=2) as client:
        client.sendall(
            f"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {bridge.MAX_REQUEST_BYTES + 1}\r\n\r\n".encode()
        )
        response = _read_raw_http_response(client)

    status_line, body = response.split("\r\n\r\n", maxsplit=1)
    assert " 413 " in status_line
    assert json.loads(body)["error"]["type"] == "request_too_large"


def test_translated_request_expansion_is_rejected_before_build_credentials_are_used(
    bridge_services: _BridgeServices,
) -> None:
    payload = {"model": "nvidia/model", "input": chr(0x1F600) * 249_900}
    incoming_body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    translated_body = json.dumps(responses_to_chat_request(payload), separators=(",", ":")).encode("utf-8")
    assert len(incoming_body) <= bridge.MAX_REQUEST_BYTES
    assert len(translated_body) > bridge.MAX_REQUEST_BYTES
    request = urllib.request.Request(
        f"{bridge_services.url}/v1/responses",
        data=incoming_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=2)

    assert caught.value.code == 413
    assert json.loads(caught.value.read())["error"]["type"] == "request_too_large"
    assert bridge_services.build.requests == []  # type: ignore[attr-defined]


def test_incomplete_request_body_times_out_with_a_structured_408(
    bridge_services: _BridgeServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "REQUEST_READ_TIMEOUT_SECONDS", 0.05)
    with socket.create_connection(("127.0.0.1", bridge_services.port), timeout=2) as client:
        client.sendall(
            b"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{"
        )
        response = _read_raw_http_response(client)

    status_line, body = response.split("\r\n\r\n", maxsplit=1)
    assert " 408 " in status_line
    assert json.loads(body)["error"]["type"] == "request_timeout"


def test_slow_drip_request_body_obeys_an_absolute_deadline(
    bridge_services: _BridgeServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "REQUEST_READ_TIMEOUT_SECONDS", 0.05)
    body = json.dumps({"model": "nvidia/model", "input": "Hello"}, separators=(",", ":")).encode("utf-8")
    with socket.create_connection(("127.0.0.1", bridge_services.port), timeout=2) as client:
        client.sendall(
            f"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        )
        for byte in body:
            try:
                client.sendall(bytes([byte]))
            except OSError:
                break
            time.sleep(0.015)
        response = _read_raw_http_response(client)

    status_line, response_body = response.split("\r\n\r\n", maxsplit=1)
    assert " 408 " in status_line
    assert json.loads(response_body)["error"]["type"] == "request_timeout"


def test_partial_headers_time_out_without_exceeding_the_worker_bound(
    bridge_services: _BridgeServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "REQUEST_HEADER_TIMEOUT_SECONDS", 0.05)
    clients = [
        socket.create_connection(("127.0.0.1", bridge_services.port), timeout=2) for _ in range(bridge.MAX_WORKERS + 2)
    ]
    try:
        for client in clients:
            client.sendall(b"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1")
        time.sleep(0.1)
        assert bridge_services.bridge_server.active_workers <= bridge.MAX_WORKERS
        assert (
            _request(
                f"{bridge_services.url}/healthz",
                headers={bridge.READINESS_TOKEN_HEADER: bridge_services.readiness_token},
            )[0]
            == 200
        )
    finally:
        for client in clients:
            client.close()


def test_slow_drip_request_line_and_headers_obey_an_absolute_deadline(
    bridge_services: _BridgeServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "REQUEST_HEADER_TIMEOUT_SECONDS", 0.05)
    stop_sending = Event()
    with socket.create_connection(("127.0.0.1", bridge_services.port), timeout=2) as client:

        def drip_headers() -> None:
            while not stop_sending.is_set():
                try:
                    client.sendall(b"P")
                except OSError:
                    return
                time.sleep(0.015)

        sender = Thread(target=drip_headers)
        sender.start()
        try:
            deadline = time.monotonic() + 1
            while bridge_services.bridge_server.active_workers == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert bridge_services.bridge_server.active_workers == 1
            deadline = time.monotonic() + 1
            while bridge_services.bridge_server.active_workers and time.monotonic() < deadline:
                time.sleep(0.005)
            assert bridge_services.bridge_server.active_workers == 0
        finally:
            stop_sending.set()
            sender.join(timeout=1)
            assert not sender.is_alive()


def test_oversized_backend_response_is_a_structured_502(bridge_services: _BridgeServices) -> None:
    status, _headers, body = _request(
        f"{bridge_services.url}/v1/responses",
        {"model": "return-large-response", "input": "Hello"},
    )

    assert status == 502
    assert json.loads(body)["error"]["type"] == "backend_error"


def test_backend_json_recursion_is_a_safe_structured_502(
    bridge_services: _BridgeServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_loads = json.loads

    def recursive_loads(value: object, *args: object, **kwargs: object) -> object:
        if value == "{}":
            raise RecursionError("simulated backend JSON recursion")
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(bridge.json, "loads", recursive_loads)
    status, _headers, body = _request(
        f"{bridge_services.url}/v1/responses",
        {"model": "return-recursion-backend", "input": "Hello"},
    )

    assert status == 502
    monkeypatch.undo()
    assert json.loads(body)["error"]["type"] == "backend_error"


def test_routes_are_relative_to_build_base_url() -> None:
    assert BUILD_CHAT_COMPLETIONS_PATH == "/chat/completions"
    assert HEALTH_PATH == "/healthz"


@pytest.mark.parametrize("protocol", ["responses", "anthropic"])
@pytest.mark.parametrize("value", [True, False, None, "16", 1.5, 0, -1, 10**100])
def test_protocol_translation_rejects_invalid_output_token_limits(protocol: str, value: object) -> None:
    if protocol == "responses":
        payload = {"model": "nvidia/model", "max_output_tokens": value}
        translator = responses_to_chat_request
        field = "max_output_tokens"
    else:
        payload = {"model": "nvidia/model", "messages": [], "max_tokens": value}
        translator = anthropic_to_chat_request
        field = "max_tokens"

    with pytest.raises(BridgePayloadError, match=rf"^{field} must be an integer between 1 and "):
        translator(payload)


def test_unsupported_responses_tool_error_does_not_echo_client_fields() -> None:
    with pytest.raises(BridgePayloadError, match=r"^unsupported Responses tool type$") as error:
        responses_to_chat_request(
            {
                "model": "nvidia/model",
                "tools": [
                    {
                        "type": "secret-shaped-type",
                        "name": "secret-shaped-name",
                        "secret": "must-not-appear",
                    }
                ],
            }
        )

    assert "secret-shaped" not in str(error.value)
    assert "must-not-appear" not in str(error.value)


def test_responses_server_web_search_tools_are_omitted_without_affecting_executable_mappings() -> None:
    translated, custom_tool_names, namespace_tools = bridge._responses_to_chat_request(
        {
            "model": "nvidia/model",
            "tools": [
                WEB_SEARCH_TOOL,
                CUSTOM_EXEC_TOOL,
                {"type": "web_search_preview", "search_context_size": "low"},
                NAMESPACE_TOOL,
            ],
        }
    )

    assert [tool["function"]["name"] for tool in translated["tools"]] == [
        "exec",
        "multi_agent_v1__spawn_agent",
        "multi_agent_v1__exec",
    ]
    assert custom_tool_names == {"exec"}
    assert set(namespace_tools) == {"multi_agent_v1__spawn_agent", "multi_agent_v1__exec"}


@pytest.mark.parametrize("tool_choice", ["auto", "none"])
def test_responses_only_server_web_search_omits_chat_tools_and_keeps_auto_or_none_choice(tool_choice: str) -> None:
    translated = responses_to_chat_request(
        {
            "model": "nvidia/model",
            "tools": [WEB_SEARCH_TOOL, {"type": "web_search_preview"}],
            "tool_choice": tool_choice,
        }
    )

    assert "tools" not in translated
    assert translated["tool_choice"] == tool_choice


def test_responses_text_and_tool_schema_become_chat_request() -> None:
    request = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "instructions": "Be concise.",
        "input": "What is the weather?",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather by city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }

    assert responses_to_chat_request(request) == {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is the weather?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }


def test_responses_custom_and_function_tools_become_chat_function_tools() -> None:
    request = {
        "model": "nvidia/model",
        "tools": [
            CUSTOM_EXEC_TOOL,
            {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
        ],
    }

    translated = responses_to_chat_request(request)

    assert translated["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "exec",
                "description": (
                    "Execute shell commands.\n\nPass the raw custom tool input through the `input` string property."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        },
    ]


def test_responses_namespace_tools_flatten_nested_function_and_custom_tools() -> None:
    translated = responses_to_chat_request(
        {
            "model": "nvidia/model",
            "tools": [
                NAMESPACE_TOOL,
                {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
            ],
        }
    )

    assert [tool["function"]["name"] for tool in translated["tools"]] == [
        "multi_agent_v1__spawn_agent",
        "multi_agent_v1__exec",
        "get_weather",
    ]
    assert translated["tools"][0]["function"] == {
        "name": "multi_agent_v1__spawn_agent",
        "description": "Spawn one agent.",
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
    }
    assert translated["tools"][1]["function"] == {
        "name": "multi_agent_v1__exec",
        "description": "Execute shell commands.\n\nPass the raw custom tool input through the `input` string property.",
        "parameters": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
            "additionalProperties": False,
        },
    }
    assert translated["tools"][2] == {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }


@pytest.mark.parametrize(
    ("tool_choice", "expected_name"),
    [
        (
            {"type": "function", "name": "spawn_agent", "namespace": "multi_agent_v1"},
            "multi_agent_v1__spawn_agent",
        ),
        ({"type": "custom", "name": "exec", "namespace": "multi_agent_v1"}, "multi_agent_v1__exec"),
    ],
)
def test_responses_namespaced_tool_choice_becomes_forced_flat_chat_function(
    tool_choice: dict[str, str], expected_name: str
) -> None:
    translated = responses_to_chat_request(
        {"model": "nvidia/model", "tools": [NAMESPACE_TOOL], "tool_choice": tool_choice}
    )

    assert translated["tool_choice"] == {"type": "function", "function": {"name": expected_name}}


@pytest.mark.parametrize(
    ("tools", "error"),
    [
        (
            [{"type": "namespace", "name": "bad/name", "description": "bad", "tools": []}],
            "namespace name",
        ),
        (
            [
                {
                    "type": "namespace",
                    "name": "safe",
                    "description": "bad nested name",
                    "tools": [{"type": "function", "name": "bad/name", "parameters": {}}],
                }
            ],
            "namespaced tool name",
        ),
        (
            [
                {
                    "type": "namespace",
                    "name": "n" * 40,
                    "description": "too long after flattening",
                    "tools": [{"type": "function", "name": "t" * 30, "parameters": {}}],
                }
            ],
            "64",
        ),
        (
            [
                {
                    "type": "namespace",
                    "name": "ns",
                    "description": "duplicate",
                    "tools": [
                        {"type": "function", "name": "work", "parameters": {}},
                        {"type": "custom", "name": "work"},
                    ],
                }
            ],
            "collision",
        ),
        (
            [
                {"type": "function", "name": "ns__work", "parameters": {}},
                {
                    "type": "namespace",
                    "name": "ns",
                    "description": "top-level collision",
                    "tools": [{"type": "function", "name": "work", "parameters": {}}],
                },
            ],
            "collision",
        ),
        (
            [{"type": "namespace", "name": "ns", "description": "bad tools", "tools": "not-a-list"}],
            "tools",
        ),
        (
            [
                {
                    "type": "namespace",
                    "name": "ns",
                    "description": "bad nested type",
                    "tools": [{"type": "shell", "name": "shell"}],
                }
            ],
            "unsupported Responses tool type",
        ),
    ],
)
def test_responses_namespace_tools_reject_malformed_or_colliding_names(
    tools: list[dict[str, object]], error: str
) -> None:
    with pytest.raises(BridgePayloadError, match=error):
        responses_to_chat_request({"model": "nvidia/model", "tools": tools})


def test_responses_custom_tool_choice_becomes_forced_chat_function() -> None:
    translated = responses_to_chat_request({"model": "nvidia/model", "tool_choice": {"type": "custom", "name": "exec"}})

    assert translated["tool_choice"] == {"type": "function", "function": {"name": "exec"}}


@pytest.mark.parametrize(
    "output",
    [
        "workspace output",
        [{"type": "input_text", "text": "workspace output"}],
    ],
)
@pytest.mark.parametrize(
    "optional_name",
    [pytest.param({}, id="without-name"), pytest.param({"name": "exec"}, id="with-name")],
)
def test_responses_custom_tool_replay_becomes_chat_tool_messages(output: object, optional_name: dict[str, str]) -> None:
    translated = responses_to_chat_request(
        {
            "model": "nvidia/model",
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call-exec-1",
                    "name": "exec",
                    "input": "pwd",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call-exec-1",
                    "output": output,
                    **optional_name,
                },
            ],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-exec-1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": '{"input":"pwd"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-exec-1", "content": "workspace output"},
    ]


def test_responses_parallel_mixed_tool_replay_becomes_one_chat_assistant_message() -> None:
    translated = responses_to_chat_request(
        {
            "model": "nvidia/model",
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call-exec-1",
                    "name": "exec",
                    "input": "pwd",
                },
                {
                    "type": "function_call",
                    "call_id": "call-weather-1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call-exec-1",
                    "output": "workspace output",
                },
                {"type": "function_call_output", "call_id": "call-weather-1", "output": "20C"},
            ],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-exec-1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": '{"input":"pwd"}'},
                },
                {
                    "id": "call-weather-1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-exec-1", "content": "workspace output"},
        {"role": "tool", "tool_call_id": "call-weather-1", "content": "20C"},
    ]


def test_responses_namespaced_function_and_custom_replay_use_flat_chat_names() -> None:
    translated = responses_to_chat_request(
        {
            "model": "nvidia/model",
            "tools": [NAMESPACE_TOOL],
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call-spawn-1",
                    "namespace": "multi_agent_v1",
                    "name": "spawn_agent",
                    "arguments": '{"task":"inspect"}',
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "call-exec-1",
                    "namespace": "multi_agent_v1",
                    "name": "exec",
                    "input": "pwd",
                },
                {"type": "function_call_output", "call_id": "call-spawn-1", "output": "agent-1"},
                {"type": "custom_tool_call_output", "call_id": "call-exec-1", "output": "workspace"},
            ],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-spawn-1",
                    "type": "function",
                    "function": {
                        "name": "multi_agent_v1__spawn_agent",
                        "arguments": '{"task":"inspect"}',
                    },
                },
                {
                    "id": "call-exec-1",
                    "type": "function",
                    "function": {"name": "multi_agent_v1__exec", "arguments": '{"input":"pwd"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-spawn-1", "content": "agent-1"},
        {"role": "tool", "tool_call_id": "call-exec-1", "content": "workspace"},
    ]


def test_responses_tool_result_becomes_chat_tool_message() -> None:
    request = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "input": [
            {"type": "function_call_output", "call_id": "call-1", "output": "42"},
        ],
    }

    translated = responses_to_chat_request(request)

    assert translated["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert translated["messages"] == [{"role": "tool", "tool_call_id": "call-1", "content": "42"}]


def test_anthropic_text_tool_and_tool_result_become_chat_request() -> None:
    request = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "system": "Be concise.",
        "messages": [
            {"role": "user", "content": "Use a tool."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking it up."},
                    {"type": "tool_use", "id": "call-1", "name": "get_weather", "input": {"city": "Paris"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "20C"}]},
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather by city.",
                "input_schema": {"type": "object"},
            }
        ],
    }

    assert anthropic_to_chat_request(request) == {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Use a tool."},
            {
                "role": "assistant",
                "content": "Looking it up.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "20C"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }


def test_anthropic_tool_result_becomes_chat_tool_message() -> None:
    request = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "42"}]},
        ],
    }

    translated = anthropic_to_chat_request(request)

    assert translated["messages"] == [{"role": "tool", "tool_call_id": "call-1", "content": "42"}]


def test_anthropic_system_message_role_used_by_claude_code_is_preserved() -> None:
    request = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Run the task."}]},
            {"role": "system", "content": "The following skills are available."},
        ],
    }

    translated = anthropic_to_chat_request(request)

    assert translated["messages"] == [
        {"role": "user", "content": "Run the task."},
        {"role": "system", "content": "The following skills are available."},
    ]


def test_claude_code_orchestration_tools_are_omitted_but_executable_and_custom_tools_remain() -> None:
    request = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{"role": "user", "content": "Run the task."}],
        "tools": [
            {"name": "Bash", "description": "Run shell", "input_schema": {"type": "object"}},
            {"name": "WebSearch", "description": "Server-side search", "input_schema": {"type": "object"}},
            {"name": "TaskCreate", "description": "Create task", "input_schema": {"type": "object"}},
            {"name": "mcp_custom", "description": "Custom MCP tool", "input_schema": {"type": "object"}},
            {"name": f"mcp_{'x' * 70}", "description": "Too long for Build", "input_schema": {"type": "object"}},
        ],
    }

    translated = anthropic_to_chat_request(request)

    assert [tool["function"]["name"] for tool in translated["tools"]] == ["Bash", "mcp_custom"]


def test_anthropic_required_tool_choice_rejects_an_all_filtered_tool_set() -> None:
    request = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{"role": "user", "content": "Search."}],
        "tools": [{"name": "WebSearch", "description": "Server-side search", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
    }

    with pytest.raises(BridgePayloadError, match="executable tool"):
        anthropic_to_chat_request(request)


@pytest.mark.parametrize("tool_choice", [{"type": "any"}, {"type": "tool", "name": "Bash"}])
def test_anthropic_forced_tool_choice_requires_a_declared_executable_tool(tool_choice: dict[str, str]) -> None:
    with pytest.raises(BridgePayloadError, match="declared executable tool"):
        anthropic_to_chat_request(
            {
                "model": "nvidia/nemotron-3-super-120b-a12b",
                "messages": [{"role": "user", "content": "Run it."}],
                "tool_choice": tool_choice,
            }
        )


def test_build_tool_call_emits_completed_responses_events() -> None:
    events = list(chat_completion_to_responses_events(CHAT_TOOL_RESPONSE))

    function_call = next(event["item"] for event in events if event["type"] == "response.output_item.done")
    assert function_call == {
        "type": "function_call",
        "id": "call-1",
        "call_id": "call-1",
        "name": "get_weather",
        "arguments": '{"city":"Paris"}',
        "status": "completed",
    }
    assert events[-1]["type"] == "response.completed"


def test_build_custom_tool_call_emits_schema_valid_responses_events() -> None:
    events = list(chat_completion_to_responses_events(CHAT_CUSTOM_TOOL_RESPONSE, {"exec"}))

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.output_item.done",
        "response.completed",
    ]
    added_item = events[1]["item"]
    assert added_item == {
        "type": "custom_tool_call",
        "id": "call-exec-1",
        "call_id": "call-exec-1",
        "name": "exec",
        "input": "",
    }
    assert events[2]["delta"] == "pwd"
    assert events[3]["input"] == "pwd"
    done_item = events[4]["item"]
    assert done_item == {**added_item, "input": "pwd"}
    assert events[-1]["response"]["output"] == [done_item]

    validator = TypeAdapter(ResponsesServerEvent)
    for event in events:
        validator.validate_python(event)


def test_build_mixed_tool_calls_keep_custom_and_function_event_types() -> None:
    completion = {
        **CHAT_CUSTOM_TOOL_RESPONSE,
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-exec-1",
                            "function": {"name": "exec", "arguments": '{"input":"pwd"}'},
                        },
                        {
                            "id": "call-weather-1",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        },
                    ],
                }
            }
        ],
    }

    events = list(chat_completion_to_responses_events(completion, {"exec"}))
    done_events = [event for event in events if event["type"] == "response.output_item.done"]

    assert [(event["output_index"], event["item"]["type"]) for event in done_events] == [
        (0, "custom_tool_call"),
        (1, "function_call"),
    ]
    assert done_events[1]["item"] == {
        "type": "function_call",
        "id": "call-weather-1",
        "call_id": "call-weather-1",
        "name": "get_weather",
        "arguments": '{"city":"Paris"}',
        "status": "completed",
    }
    validator = TypeAdapter(ResponsesServerEvent)
    for event in events:
        validator.validate_python(event)


def test_build_namespaced_tool_calls_restore_original_sdk_valid_items() -> None:
    _chat_request, custom_tool_names, namespace_tools = bridge._responses_to_chat_request(
        {"model": "nvidia/model", "tools": [NAMESPACE_TOOL]}
    )
    completion = {
        **CHAT_NAMESPACE_TOOL_RESPONSE,
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-spawn-1",
                            "function": {
                                "name": "multi_agent_v1__spawn_agent",
                                "arguments": '{"task":"inspect"}',
                            },
                        },
                        {
                            "id": "call-exec-1",
                            "function": {"name": "multi_agent_v1__exec", "arguments": '{"input":"pwd"}'},
                        },
                    ],
                }
            }
        ],
    }

    events = list(chat_completion_to_responses_events(completion, custom_tool_names, namespace_tools))
    done_events = [event for event in events if event["type"] == "response.output_item.done"]

    assert done_events[0]["item"] == {
        "type": "function_call",
        "id": "call-spawn-1",
        "call_id": "call-spawn-1",
        "name": "spawn_agent",
        "namespace": "multi_agent_v1",
        "arguments": '{"task":"inspect"}',
        "status": "completed",
    }
    assert done_events[1]["item"] == {
        "type": "custom_tool_call",
        "id": "call-exec-1",
        "call_id": "call-exec-1",
        "name": "exec",
        "namespace": "multi_agent_v1",
        "input": "pwd",
    }
    assert [event["type"] for event in events].count("response.custom_tool_call_input.done") == 1
    validator = TypeAdapter(ResponsesServerEvent)
    for event in events:
        validator.validate_python(event)


@pytest.mark.parametrize("arguments", ["not-json", "[]", "{}", '{"input":1}'])
def test_build_custom_tool_call_requires_a_string_input_argument(arguments: str) -> None:
    completion = {
        **CHAT_CUSTOM_TOOL_RESPONSE,
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "call-exec-1", "function": {"name": "exec", "arguments": arguments}}],
                }
            }
        ],
    }

    with pytest.raises(BridgePayloadError, match="custom tool call arguments"):
        list(chat_completion_to_responses_events(completion, {"exec"}))


def test_build_tool_call_emits_anthropic_tool_use_and_message_stop() -> None:
    events = list(chat_completion_to_anthropic_events(CHAT_TOOL_RESPONSE))

    assert {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "call-1", "name": "get_weather", "input": {}},
    } in events
    assert {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"city":"Paris"}'},
    } in events
    assert events[-1] == {"type": "message_stop"}


def test_anthropic_tool_arguments_are_validated_before_first_event() -> None:
    malformed_completion = {
        **CHAT_TOOL_RESPONSE,
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "call-1", "function": {"name": "get_weather", "arguments": "not-json"}}],
                }
            }
        ],
    }
    events = chat_completion_to_anthropic_events(malformed_completion)
    yielded: list[dict[str, object]] = []

    with pytest.raises(BridgePayloadError, match="tool call arguments must be valid JSON"):
        yielded.append(next(events))

    assert yielded == []


@pytest.mark.parametrize(
    ("translator", "payload"),
    [
        (
            responses_to_chat_request,
            {"model": "nvidia/model", "tools": [{"type": "function", "name": "x", "parameters": {}, "strict": True}]},
        ),
        (
            anthropic_to_chat_request,
            {"model": "nvidia/model", "messages": [], "tools": [{"name": "x", "input_schema": {}, "strict": False}]},
        ),
    ],
)
def test_tool_strictness_is_preserved_in_chat_tools(translator: object, payload: dict[str, object]) -> None:
    assert callable(translator)
    assert translator(payload)["tools"][0]["function"]["strict"] is payload["tools"][0]["strict"]


def test_parallel_tool_choice_is_preserved_for_responses_and_anthropic_requests() -> None:
    responses = responses_to_chat_request({"model": "nvidia/model", "parallel_tool_calls": False})
    anthropic = anthropic_to_chat_request(
        {"model": "nvidia/model", "messages": [], "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    )

    assert responses["parallel_tool_calls"] is False
    assert anthropic["tool_choice"] == "auto"
    assert anthropic["parallel_tool_calls"] is False


@pytest.mark.parametrize("completion", [CHAT_TOOL_RESPONSE, CHAT_TEXT_RESPONSE])
def test_responses_events_validate_with_openai_sdk_and_have_sequence_numbers(completion: dict[str, object]) -> None:
    events = list(chat_completion_to_responses_events(completion))
    validator = TypeAdapter(ResponsesServerEvent)

    assert [event["sequence_number"] for event in events] == list(range(1, len(events) + 1))
    for event in events:
        validator.validate_python(event)

    text_delta = next((event for event in events if event["type"] == "response.output_text.delta"), None)
    if text_delta is not None:
        assert text_delta["item_id"] == "chatcmpl-text-1-message-0"
        assert text_delta["content_index"] == 0
        assert text_delta["logprobs"] == []


@pytest.mark.parametrize("completion", [CHAT_TOOL_RESPONSE, CHAT_TEXT_RESPONSE])
def test_anthropic_events_validate_with_sdk(completion: dict[str, object]) -> None:
    events = list(chat_completion_to_anthropic_events(completion))
    validator = TypeAdapter(RawMessageStreamEvent)

    for event in events:
        validator.validate_python(event)


@pytest.mark.parametrize(
    ("translator", "payload", "expected"),
    [
        (
            responses_to_chat_request,
            {"model": "nvidia/model", "tool_choice": {"type": "function", "name": "x"}},
            {"type": "function", "function": {"name": "x"}},
        ),
        (
            anthropic_to_chat_request,
            {
                "model": "nvidia/model",
                "messages": [],
                "tools": [{"name": "x", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "tool", "name": "x"},
            },
            {"type": "function", "function": {"name": "x"}},
        ),
        (
            anthropic_to_chat_request,
            {
                "model": "nvidia/model",
                "messages": [],
                "tools": [{"name": "x", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "any"},
            },
            "required",
        ),
        (responses_to_chat_request, {"model": "nvidia/model", "tool_choice": "auto"}, "auto"),
        (anthropic_to_chat_request, {"model": "nvidia/model", "messages": [], "tool_choice": {"type": "auto"}}, "auto"),
    ],
)
def test_forced_tool_choice_translates_to_chat_contract(
    translator: object, payload: dict[str, object], expected: object
) -> None:
    assert callable(translator)
    assert translator(payload)["tool_choice"] == expected


@pytest.mark.parametrize("content", [0, False, []])
def test_falsey_non_string_completion_content_is_rejected(content: object) -> None:
    completion = {**CHAT_TEXT_RESPONSE, "choices": [{"message": {"content": content}}]}

    with pytest.raises(BridgePayloadError) as error:
        list(chat_completion_to_responses_events(completion))

    assert error.value.status_code == 400
    assert str(error.value) == "completion message content must be a string or null"


def test_malformed_payload_has_a_structured_client_error() -> None:
    with pytest.raises(BridgePayloadError) as error:
        responses_to_chat_request(["not", "an", "object"])

    assert error.value.status_code == 400
    assert str(error.value) == "payload must be a JSON object"


def test_redact_log_text_removes_simulated_nvidia_token() -> None:
    text = "Authorization: Bearer nvapi-simulated-token-for-test"

    assert "nvapi-simulated-token-for-test" not in redact_bridge_text(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"NVIDIA_API_KEY":"nvapi-secret"}', '{"NVIDIA_API_KEY":"<redacted>"}'),
        ("api-key=nvapi-secret", "api-key=<redacted>"),
    ],
)
def test_redact_log_text_masks_public_and_generic_api_key_values(text: str, expected: str) -> None:
    assert redact_bridge_text(text) == expected


def test_bridge_main_consumes_api_key_file_before_serving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_file = tmp_path / "nvidia-build-api-key"
    key_file.write_text("nvapi-test-secret", encoding="utf-8")
    key_file.chmod(0o600)
    client_token_file = tmp_path / "nvidia-build-client-token"
    client_token_file.write_text("per-run-client-token", encoding="utf-8")
    client_token_file.chmod(0o600)
    ready_file = tmp_path / "nvidia-build-bridge.ready"
    observed: dict[str, object] = {}

    def fake_serve(config: bridge.BridgeConfig, *, ready_file: Path) -> None:
        observed["api_key"] = config.api_key
        observed["base_url"] = config.build_base_url
        observed["port"] = config.port
        observed["has_readiness_token"] = bool(config.readiness_token)
        observed["client_token"] = config.client_token
        observed["allowed_model"] = config.allowed_model
        observed["max_requests"] = config.max_requests
        observed["max_output_tokens"] = config.max_output_tokens
        observed["ready_file"] = ready_file
        observed["key_file_exists"] = key_file.exists()
        observed["client_token_file_exists"] = client_token_file.exists()

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(bridge, "serve", fake_serve)

    result = bridge.main(
        [
            "--build-base-url",
            "https://integrate.api.nvidia.com/v1",
            "--api-key-file",
            str(key_file),
            "--client-token-file",
            str(client_token_file),
            "--allowed-model",
            "nvidia/test-model",
            "--max-requests",
            "17",
            "--max-output-tokens",
            "4096",
            "--port",
            "0",
            "--ready-file",
            str(ready_file),
        ]
    )

    assert result == 0
    assert observed == {
        "api_key": "nvapi-test-secret",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "port": 0,
        "has_readiness_token": True,
        "client_token": "per-run-client-token",
        "allowed_model": "nvidia/test-model",
        "max_requests": 17,
        "max_output_tokens": 4096,
        "ready_file": ready_file,
        "key_file_exists": False,
        "client_token_file_exists": False,
    }
