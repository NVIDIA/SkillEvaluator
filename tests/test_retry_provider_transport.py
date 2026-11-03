# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise provider error wrapping over real loopback HTTP and TLS connections."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import ssl
import threading
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from skillevaluator.inference.client import LLMClient
from skillevaluator.inference.retry import extract_http_status, is_retriable_exception
from skillevaluator.provider_config import ProviderConfig


@pytest.fixture
def isolated_provider_environment(monkeypatch, tmp_path):
    """Use synthetic credentials and refuse any network address outside loopback."""
    for name in tuple(os.environ):
        if name.startswith("AWS_"):
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
    def __init__(self, statuses, tls_context=None):
        self.statuses = statuses
        self.requests: list[int] = []
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
        payload = {"message": "loopback response"}
        if status == 200:
            if self.path.endswith("/converse"):
                payload = {
                    "output": {"message": {"role": "assistant", "content": [{"text": "recovered"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3},
                    "metrics": {"latencyMs": 1},
                }
            elif self.path.endswith("/invoke"):
                payload = {
                    "id": "loopback",
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
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@contextmanager
def _server(statuses, tls_context=None):
    server = _Server(statuses, tls_context)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def _client(monkeypatch, endpoint, provider="bedrock"):
    if provider == "bedrock":
        pytest.importorskip("litellm")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", endpoint)
        client = LLMClient(max_retries=2, retry_base_delay=0, retry_max_delay=0)
        client._provider_config = ProviderConfig(
            "bedrock",
            "anthropic.claude-3-haiku-20240307-v1:0",
            None,
            None,
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            region="us-west-2",
        )
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


def _untrusted_tls_context(tmp_path: Path):
    x509 = pytest.importorskip("cryptography.x509")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(days=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), False)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "certificate.pem", tmp_path / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context


@pytest.mark.parametrize("provider", ["bedrock", "openai"])
def test_provider_certificate_failure_is_not_retried(
    isolated_provider_environment,
    monkeypatch,
    tmp_path,
    provider,
    record_property,
):
    with _server([503], _untrusted_tls_context(tmp_path)) as server:
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
