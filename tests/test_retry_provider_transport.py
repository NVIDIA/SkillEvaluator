# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise provider error wrapping over real loopback HTTP and TLS connections."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType

import pytest

from skillevaluator.inference.client import LLMClient
from skillevaluator.inference.retry import extract_http_status, is_retriable_exception, retry_call_with_backoff
from skillevaluator.inference.types import LLMClientError
from skillevaluator.provider_config import ProviderConfig


@pytest.fixture
def isolated_provider_environment(monkeypatch, tmp_path):
    """Use synthetic credentials and refuse any network address outside loopback."""
    for name in tuple(os.environ):
        if name.startswith(("AWS_", "BEDROCK_MANTLE_", "ANTHROPIC_AWS_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "loopback-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "loopback-secret")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("LITELLM_MODE", "PRODUCTION")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        monkeypatch.delenv(name, raising=False)
    connect = socket.socket.connect
    getaddrinfo = socket.getaddrinfo
    attempts = []

    def local_getaddrinfo(host, *args, **kwargs):
        assert host in ("localhost", "127.0.0.1", "::1", b"localhost", b"127.0.0.1", b"::1"), host
        return getaddrinfo(host, *args, **kwargs)

    def local_connect(sock, address):
        if isinstance(address, tuple):
            assert ipaddress.ip_address(address[0]).is_loopback, address
            attempts.append(address)
        return connect(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", local_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", local_connect)
    return attempts


class _Server(ThreadingHTTPServer):
    def __init__(self, statuses, tls_context=None, *, retry_after=None):
        self.statuses = statuses
        self.retry_after = retry_after
        self.requests: list[int] = []
        self.paths: list[str] = []
        self.bearer_requests = 0
        self.authorizations: list[str] = []
        self.connections = 0
        self.tls_context = tls_context
        super().__init__(("127.0.0.1", 0), _Handler)

    def get_request(self):
        connection, address = super().get_request()
        self.connections += 1
        if self.tls_context is not None:
            try:
                connection = self.tls_context.wrap_socket(connection, server_side=True)
            except BaseException:
                connection.close()
                raise
        return connection, address


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        status = self.server.statuses[min(len(self.server.requests), len(self.server.statuses) - 1)]
        self.server.requests.append(status)
        self.server.paths.append(self.path)
        self.server.authorizations.append(self.headers.get("Authorization", ""))
        self.server.bearer_requests += self.headers.get("Authorization", "").startswith("Bearer ")
        payload = {"message": "loopback response"}
        if status == 200:
            if self.path.endswith("/converse"):
                payload = {
                    "output": {"message": {"role": "assistant", "content": [{"text": "recovered"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3},
                    "metrics": {"latencyMs": 1},
                }
            elif self.path.endswith("/responses"):
                payload = {
                    "id": "loopback",
                    "object": "response",
                    "created_at": 0,
                    "model": "loopback",
                    "status": "completed",
                    "output": [
                        {
                            "id": "loopback",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "recovered", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                }
            elif self.path.endswith(("/invoke", "/messages")):
                payload = {
                    "id": "loopback",
                    "model": "loopback",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "recovered"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                }
            else:
                payload = {
                    "id": "loopback",
                    "object": "chat.completion",
                    "model": "loopback",
                    "created": 0,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "recovered"}, "finish_reason": "stop"}
                    ],
                }
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.server.retry_after is not None:
            self.send_header("Retry-After", self.server.retry_after)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@contextmanager
def _server(statuses, tls_context=None, *, retry_after=None):
    server = _Server(statuses, tls_context, retry_after=retry_after)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize(
    ("inner_retry_after", "outer_retry_after", "should_retry"),
    [
        ("garbage", "60", False),
        ("", "60", False),
        ("NaN", "60", False),
        ("-inf", "60", False),
        ("-5", "60", False),
        ("inf", "0", True),
        ("1e999", "0", True),
        ("0", "60", True),
        ("Wed, 21 Oct 2015 07:28:00 GMT", "60", True),
    ],
)
def test_wrapped_retry_after_preserves_valid_delay_over_http(
    isolated_provider_environment, inner_retry_after, outer_retry_after, should_retry, record_property
):
    with _server([429, 200], retry_after=inner_retry_after) as server:
        url = f"http://127.0.0.1:{server.server_port}"

        def operation():
            request = urllib.request.Request(url, data=b"{}", method="POST")
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    return response.status
            except urllib.error.HTTPError as inner:
                outer = urllib.error.HTTPError(url, 503, "wrapper", {"Retry-After": outer_retry_after}, None)
                raise outer from inner

        result = None
        error = None
        try:
            result = retry_call_with_backoff(operation, max_retries=1, max_delay=0.01, base_delay=0)
        except urllib.error.HTTPError as caught:
            error = caught

        record_property("inner_retry_after", inner_retry_after)
        record_property("outer_retry_after", outer_retry_after)
        record_property("http_statuses", server.requests)
        if should_retry:
            assert result == 200
            assert error is None
            assert server.requests == [429, 200]
        else:
            assert result is None
            assert error is not None
            assert error.code == 503
            assert server.requests == [429]


def _client(
    monkeypatch,
    endpoint,
    provider="bedrock",
    model="anthropic.claude-3-haiku-20240307-v1:0",
    endpoint_variable="AWS_BEDROCK_RUNTIME_ENDPOINT",
):
    if provider == "bedrock":
        pytest.importorskip("litellm")
        monkeypatch.setenv(endpoint_variable, endpoint)
        client = LLMClient(max_retries=2, retry_base_delay=0, retry_max_delay=0)
        client._provider_config = ProviderConfig("bedrock", model, None, None, f"bedrock/{model}", region="us-west-2")
        return client
    pytest.importorskip("openai")
    return LLMClient(
        model="loopback",
        api_key="loopback-key",
        base_url=endpoint,
        max_retries=2,
        retry_base_delay=0,
        retry_max_delay=0,
    )


@pytest.mark.parametrize("provider", ["bedrock", "openai"])
@pytest.mark.parametrize("status", [400, 408, 429, 500, 501, 502, 503, 504, 505, 520, 529])
def test_provider_http_status_survives_sdk_wrapping(
    isolated_provider_environment,
    monkeypatch,
    provider,
    status,
    record_property,
):
    expected = 1 if status in (400, 501, 505) else 3
    with _server([status]) as server:
        client = _client(monkeypatch, f"http://127.0.0.1:{server.server_port}", provider)
        with pytest.raises(Exception) as raised:
            client.completions("system", "user")
        record_property("provider", provider)
        record_property("status", status)
        record_property("http_requests", len(server.requests))
        record_property("outer_status", getattr(raised.value, "status_code", None))
        assert server.requests == [status] * expected
        assert extract_http_status(raised.value) == status


@pytest.mark.parametrize("model", ["anthropic.claude-3-haiku-20240307-v1:0", "converse/mantle/example"])
def test_bedrock_inference_rejects_http_bearer_override(isolated_provider_environment, monkeypatch, model):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    with _server([200]) as server:
        client = _client(monkeypatch, f"http://127.0.0.1:{server.server_port}", model=model)
        with pytest.raises(LLMClientError, match="AWS_BEDROCK_RUNTIME_ENDPOINT must use HTTPS"):
            client.completions("system", "user")
        assert server.connections == 0


_MANTLE_MODEL = "mantle/anthropic.claude-3-haiku-20240307-v1:0"
_PLATFORM_MODEL = "claude_platform/anthropic.claude-3-haiku-20240307-v1:0"
_OTHER_ROUTE_ENDPOINTS = [
    (_MANTLE_MODEL, "BEDROCK_MANTLE_API_BASE", "/anthropic/v1/messages"),
    (_PLATFORM_MODEL, "ANTHROPIC_AWS_BASE_URL", "/v1/messages"),
    (_PLATFORM_MODEL, "ANTHROPIC_AWS_API_BASE", "/v1/messages"),
]


@pytest.mark.parametrize("responses_prefix", ["", "responses/"])
@pytest.mark.parametrize(("model", "variable", "_path"), _OTHER_ROUTE_ENDPOINTS)
def test_bedrock_inference_rejects_http_bearer_override_on_other_routes(
    isolated_provider_environment, monkeypatch, responses_prefix, model, variable, _path
):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    monkeypatch.setenv("ANTHROPIC_AWS_WORKSPACE_ID", "loopback-workspace")
    with _server([200]) as server:
        client = _client(
            monkeypatch,
            f"http://127.0.0.1:{server.server_port}",
            model=f"{responses_prefix}{model}",
            endpoint_variable=variable,
        )
        with pytest.raises(LLMClientError, match=f"{variable} must use HTTPS"):
            client.completions("system", "user")
        assert server.connections == 0


@pytest.mark.parametrize(("model", "variable", "path"), _OTHER_ROUTE_ENDPOINTS)
def test_bedrock_inference_accepts_https_bearer_override_on_other_routes(
    isolated_provider_environment, monkeypatch, loopback_tls, model, variable, path
):
    litellm = pytest.importorskip("litellm")
    from litellm.caching.llm_caching_handler import LLMClientCache

    tls_context, ca_bundle = loopback_tls
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    monkeypatch.setenv("ANTHROPIC_AWS_WORKSPACE_ID", "loopback-workspace")
    monkeypatch.setenv("SSL_VERIFY", str(ca_bundle))
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache(), raising=False)
    monkeypatch.setattr(litellm, "drop_params", True)
    with _server([200], tls_context) as server, _server([200]) as unused:
        if variable == "ANTHROPIC_AWS_BASE_URL":
            monkeypatch.setenv("ANTHROPIC_AWS_API_BASE", f"http://127.0.0.1:{unused.server_port}")
        client = _client(
            monkeypatch, f"https://127.0.0.1:{server.server_port}", model=model, endpoint_variable=variable
        )
        assert client.completions("system", "user") == "recovered"
        assert server.paths == [path]
        assert server.bearer_requests == 1
        assert unused.connections == 0


@pytest.mark.parametrize("responses_bridge", [False, True])
def test_bedrock_inference_rejects_http_override_after_model_alias(
    isolated_provider_environment, monkeypatch, responses_bridge
):
    litellm = pytest.importorskip("litellm")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    model = "anthropic.claude-3-haiku-20240307-v1:0"
    aliases = {model: _MANTLE_MODEL} if responses_bridge else {f"bedrock/{model}": f"bedrock/{_MANTLE_MODEL}"}
    monkeypatch.setattr(litellm, "model_alias_map", aliases)
    with _server([200]) as sink:
        client = _client(
            monkeypatch,
            f"http://127.0.0.1:{sink.server_port}",
            model=f"responses/{model}" if responses_bridge else model,
            endpoint_variable="BEDROCK_MANTLE_API_BASE",
        )
        with pytest.raises(LLMClientError, match="BEDROCK_MANTLE_API_BASE must use HTTPS"):
            client.completions("system", "user")
        assert sink.connections == 0


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_bedrock_claude_platform_checks_global_api_base(
    isolated_provider_environment, monkeypatch, loopback_tls, scheme
):
    litellm = pytest.importorskip("litellm")
    from litellm.caching.llm_caching_handler import LLMClientCache

    tls_context, ca_bundle = loopback_tls
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    monkeypatch.setenv("ANTHROPIC_AWS_WORKSPACE_ID", "loopback-workspace")
    monkeypatch.setenv("SSL_VERIFY", str(ca_bundle))
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache(), raising=False)
    monkeypatch.setattr(litellm, "drop_params", True)
    with _server([200], tls_context if scheme == "https" else None) as sink, _server([200]) as unused:
        monkeypatch.setattr(litellm, "api_base", f"{scheme}://127.0.0.1:{sink.server_port}")
        client = _client(
            monkeypatch,
            f"http://127.0.0.1:{unused.server_port}",
            model=_PLATFORM_MODEL,
            endpoint_variable="ANTHROPIC_AWS_BASE_URL",
        )
        if scheme == "http":
            with pytest.raises(LLMClientError, match="LiteLLM api_base must use HTTPS"):
                client.completions("system", "user")
            assert sink.connections == 0
        else:
            assert client.completions("system", "user") == "recovered"
            assert sink.paths == ["/v1/messages"]
            assert sink.bearer_requests == 1
        assert unused.connections == 0


def test_bedrock_inference_checks_http_override_for_a_global_fallback(isolated_provider_environment, monkeypatch):
    litellm = pytest.importorskip("litellm")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    monkeypatch.setattr(litellm, "model_fallbacks", ["bedrock/mantle/example"])
    with _server([200]) as sink:
        monkeypatch.setenv("BEDROCK_MANTLE_API_BASE", f"http://127.0.0.1:{sink.server_port}")
        client = _client(monkeypatch, "https://127.0.0.1:9")
        with pytest.raises(LLMClientError, match="BEDROCK_MANTLE_API_BASE must use HTTPS"):
            client.completions("system", "user")
        assert sink.connections == 0
        assert not isolated_provider_environment


def test_bedrock_mantle_provider_checks_http_override_after_alias(isolated_provider_environment, monkeypatch):
    litellm = pytest.importorskip("litellm")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    model = "anthropic.claude-3-haiku-20240307-v1:0"
    monkeypatch.setattr(litellm, "model_alias_map", {f"bedrock/{model}": "bedrock_mantle/openai.gpt-5.5"})
    with _server([200]) as sink:
        monkeypatch.setenv("BEDROCK_MANTLE_API_BASE", f"http://127.0.0.1:{sink.server_port}")
        client = _client(monkeypatch, "https://127.0.0.1:9", model=model)
        # LiteLLM resolves the distinct Mantle provider's override as api_base.
        with pytest.raises(LLMClientError, match="LiteLLM api_base must use HTTPS"):
            client.completions("system", "user")
        assert sink.connections == 0
        assert not isolated_provider_environment


def test_bedrock_mantle_alias_uses_configured_region(isolated_provider_environment, monkeypatch, loopback_tls):
    litellm = pytest.importorskip("litellm")
    from litellm.caching.llm_caching_handler import LLMClientCache

    tls_context, ca_bundle = loopback_tls
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    monkeypatch.setenv("BEDROCK_MANTLE_REGION", "invalid_region")
    monkeypatch.setenv("SSL_VERIFY", str(ca_bundle))
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache(), raising=False)
    monkeypatch.setattr(litellm, "drop_params", True)
    model = "anthropic.claude-3-haiku-20240307-v1:0"
    monkeypatch.setattr(litellm, "model_alias_map", {f"bedrock/{model}": "bedrock_mantle/openai.gpt-5.5"})
    with _server([200], tls_context) as sink:
        client = _client(
            monkeypatch,
            f"https://127.0.0.1:{sink.server_port}",
            model=model,
            endpoint_variable="BEDROCK_MANTLE_API_BASE",
        )
        assert client.completions("system", "user") == "recovered"
        assert sink.paths == ["/openai/v1/responses"]
        assert sink.bearer_requests == 1


def test_bedrock_responses_bridge_checks_runtime_instead_of_chat_route(isolated_provider_environment, monkeypatch):
    litellm = pytest.importorskip("litellm")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    model = "mantle/example"
    registration = {"litellm_provider": "bedrock", "mode": "responses", "supported_endpoints": ["/v1/responses"]}
    monkeypatch.setitem(litellm.model_cost, model, registration)
    monkeypatch.setitem(litellm.model_cost, f"bedrock/{model}", registration)
    with _server([200]) as sink:
        client = _client(monkeypatch, f"http://127.0.0.1:{sink.server_port}", model=model)
        with pytest.raises(LLMClientError, match="AWS_BEDROCK_RUNTIME_ENDPOINT must use HTTPS"):
            client.completions("system", "user")
        assert sink.connections == 0


@pytest.mark.parametrize("model_prefix", ["", "bedrock/"])
def test_bedrock_mantle_uses_its_own_endpoint_when_runtime_is_also_set(
    isolated_provider_environment, monkeypatch, loopback_tls, model_prefix
):
    litellm = pytest.importorskip("litellm")
    from litellm.caching.llm_caching_handler import LLMClientCache

    tls_context, ca_bundle = loopback_tls
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    monkeypatch.setenv("SSL_VERIFY", str(ca_bundle))
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache(), raising=False)
    monkeypatch.setattr(litellm, "drop_params", False)
    with _server([200]) as runtime, _server([200], tls_context) as mantle:
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", f"http://127.0.0.1:{runtime.server_port}")
        client = _client(
            monkeypatch,
            f"https://127.0.0.1:{mantle.server_port}",
            model=f"{model_prefix}{_MANTLE_MODEL}",
            endpoint_variable="BEDROCK_MANTLE_API_BASE",
        )
        assert client.completions("system", "user") == "recovered"
        assert mantle.paths == ["/anthropic/v1/messages"]
        assert runtime.connections == 0


@pytest.mark.parametrize("bearer_token", ["", " \t "])
@pytest.mark.parametrize(("model", "variable", "path"), _OTHER_ROUTE_ENDPOINTS)
def test_bedrock_inference_ignores_http_override_on_other_routes_when_bearer_token_is_blank(
    isolated_provider_environment, monkeypatch, model, variable, path, bearer_token
):
    litellm = pytest.importorskip("litellm")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
    monkeypatch.setenv("ANTHROPIC_AWS_WORKSPACE_ID", "loopback-workspace")
    monkeypatch.setattr(litellm, "drop_params", True)
    with _server([200]) as server:
        client = _client(monkeypatch, f"http://127.0.0.1:{server.server_port}", model=model, endpoint_variable=variable)
        assert client.completions("system", "user") == "recovered"
        assert server.paths == [path]
        assert server.authorizations[0].startswith("AWS4-HMAC-SHA256 ")
        assert server.bearer_requests == 0


@pytest.mark.parametrize("bearer_token", ["", "   ", " \t "])
def test_bedrock_inference_signs_with_sigv4_when_bearer_token_is_blank(
    isolated_provider_environment, monkeypatch, bearer_token
):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
    with _server([200]) as server:
        client = _client(monkeypatch, f"http://127.0.0.1:{server.server_port}")
        assert client.completions("system", "user") == "recovered"
        assert server.requests == [200]
        assert len(server.authorizations) == 1
        assert server.authorizations[0].startswith("AWS4-HMAC-SHA256 ")
        assert "Bearer" not in server.authorizations[0]


@pytest.mark.parametrize("bearer_token", ["", "   ", " \t "])
def test_bedrock_responses_route_signs_with_sigv4_when_bearer_token_is_blank(
    isolated_provider_environment, monkeypatch, bearer_token
):
    litellm = pytest.importorskip("litellm")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
    monkeypatch.setattr(litellm, "drop_params", False)
    with _server([200]) as server:
        client = _client(monkeypatch, f"http://127.0.0.1:{server.server_port}", model="responses/us.openai.gpt-5.6-sol")
        assert client.completions("system", "user") == "recovered"
        assert server.paths == ["/openai/v1/responses"]
        assert len(server.authorizations) == 1
        assert server.authorizations[0].startswith("AWS4-HMAC-SHA256 ")
        assert "Bearer" not in server.authorizations[0]
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == bearer_token


def test_bedrock_responses_route_restores_blank_bearer_token_after_a_failed_call(
    isolated_provider_environment, monkeypatch
):
    litellm = pytest.importorskip("litellm")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", " \t ")
    monkeypatch.setattr(litellm, "drop_params", False)
    with _server([400]) as server:
        client = _client(monkeypatch, f"http://127.0.0.1:{server.server_port}", model="responses/us.openai.gpt-5.6-sol")
        with pytest.raises(Exception) as raised:
            client.completions("system", "user")
        assert extract_http_status(raised.value) == 400
        assert server.paths == ["/openai/v1/responses"]
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == " \t "


def test_blank_bedrock_bearer_token_is_restored_exactly(monkeypatch):
    from skillevaluator.inference.client import _bedrock_token_window

    name = "AWS_BEARER_TOKEN_BEDROCK"
    monkeypatch.setenv(name, " \t ")
    with _bedrock_token_window.blank():
        assert os.environ[name] == ""
        with _bedrock_token_window.blank():
            assert os.environ[name] == ""
        # An overlapping call finishing must not put the value back for the one still running.
        assert os.environ[name] == ""
    assert os.environ[name] == " \t "
    with pytest.raises(RuntimeError), _bedrock_token_window.blank():
        raise RuntimeError("call failed")
    assert os.environ[name] == " \t "
    monkeypatch.delenv(name)
    with _bedrock_token_window.blank():
        assert os.environ[name] == ""
    assert name not in os.environ


def test_bedrock_inference_validates_credentials_loaded_at_sdk_import(isolated_provider_environment, monkeypatch):
    sdk = pytest.importorskip("litellm")

    def unexpected_completion(**_kwargs):
        pytest.fail("Bedrock completion must not be called for an HTTP bearer endpoint")

    class LazyLiteLLM(ModuleType):
        def __getattr__(self, name):
            if name != "completion":
                return getattr(sdk, name)
            monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
            return unexpected_completion

    with _server([200]) as server:
        client = _client(monkeypatch, f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setitem(sys.modules, "litellm", LazyLiteLLM("litellm"))
        with pytest.raises(LLMClientError, match="AWS_BEDROCK_RUNTIME_ENDPOINT must use HTTPS"):
            client.completions("system", "user")
        assert server.connections == 0


@pytest.mark.parametrize("model", ["anthropic.claude-3-haiku-20240307-v1:0", "responses/us.openai.gpt-5.6-sol"])
def test_bedrock_inference_accepts_https_bearer_override(
    isolated_provider_environment, monkeypatch, loopback_tls, model
):
    litellm = pytest.importorskip("litellm")
    from litellm.caching.llm_caching_handler import LLMClientCache

    tls_context, ca_bundle = loopback_tls
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "loopback-bearer-token")
    monkeypatch.setenv("SSL_VERIFY", str(ca_bundle))
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache(), raising=False)
    monkeypatch.setattr(litellm, "drop_params", False)
    with _server([200], tls_context) as server, _server([200]) as unused:
        unused_endpoint = f"http://127.0.0.1:{unused.server_port}"
        monkeypatch.setenv("BEDROCK_MANTLE_API_BASE", unused_endpoint)
        monkeypatch.setenv("ANTHROPIC_AWS_BASE_URL", unused_endpoint)
        client = _client(monkeypatch, f"https://127.0.0.1:{server.server_port}", model=model)
        assert client.completions("system", "user") == "recovered"
        assert server.requests == [200]
        assert server.paths[0].endswith("/openai/v1/responses" if model.startswith("responses/") else "/converse")
        assert server.bearer_requests == 1
        assert unused.connections == 0
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "loopback-bearer-token"


@pytest.mark.parametrize("provider", ["bedrock", "openai"])
def test_provider_certificate_failure_is_not_retried(
    isolated_provider_environment,
    monkeypatch,
    loopback_tls,
    provider,
    record_property,
):
    tls_context, _ = loopback_tls
    with _server([503], tls_context) as server:
        client = _client(monkeypatch, f"https://127.0.0.1:{server.server_port}", provider)
        with pytest.raises(Exception) as raised:
            client.completions("system", "user")
        record_property("provider", provider)
        record_property("tcp_connections", server.connections)
        record_property("http_requests", len(server.requests))
        record_property("outer_status", getattr(raised.value, "status_code", None))
        assert not is_retriable_exception(raised.value)
        assert server.connections == 1
        assert server.requests == []


@pytest.mark.parametrize("provider", ["bedrock", "openai"])
def test_provider_recovers_after_transient_loopback_failures(
    isolated_provider_environment,
    monkeypatch,
    provider,
    record_property,
):
    with _server([529, 408, 200]) as server:
        client = _client(monkeypatch, f"http://127.0.0.1:{server.server_port}", provider)
        assert client.completions("system", "user") == "recovered"
        record_property("provider", provider)
        record_property("http_statuses", server.requests)
        assert server.requests == [529, 408, 200]


@pytest.mark.parametrize("provider", ["bedrock", "openai"])
def test_provider_connection_refused_still_retries(
    isolated_provider_environment,
    monkeypatch,
    provider,
    record_property,
):
    # Reserve a loopback port without listening so no unrelated process can
    # take it while the real SDK receives ECONNREFUSED.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        client = _client(monkeypatch, f"http://127.0.0.1:{port}", provider)
        with pytest.raises(Exception) as raised:
            client.completions("system", "user")
        attempts = [address for address in isolated_provider_environment if address[1] == port]
        record_property("provider", provider)
        record_property("connection_attempts", len(attempts))
        assert is_retriable_exception(raised.value)
        assert len(attempts) == 3
