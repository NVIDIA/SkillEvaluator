# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in public MCP proof (``--probe-mcp``) against a local fake HTTP MCP server."""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.cli import _plugin_mcp_proof
from skillevaluator.tier3.mcp_proof import (
    NOT_REQUESTED_DETAIL,
    _check_endpoint,
    _ProbeRefused,
    apply_in_agent_mcp_proof,
    declared_mcp_proof,
    probe_mcp_server,
    probe_mcp_servers,
)
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

pytest.importorskip("mcp")

TOOLS = [{"name": "search", "inputSchema": {"type": "object"}}, {"name": "get_page", "inputSchema": {"type": "object"}}]
LOCAL = ("127.0.0.1",)


def _result(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    return {}


class _Handler(BaseHTTPRequestHandler):
    server: _FakeServer

    def log_message(self, *_args: Any) -> None:
        return

    def _empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_DELETE(self) -> None:
        self._empty(200)

    def do_GET(self) -> None:
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/sse-offorigin":
            # The endpoint event points at another origin; the client must refuse it at once.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"event: endpoint\ndata: http://127.0.0.1:9/messages?session_id=s1\n\n")
            self.wfile.flush()
            time.sleep(3)
            return
        if self.path != "/sse":
            self._empty(405)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(b"event: endpoint\ndata: /messages?session_id=s1\n\n")
        self.wfile.flush()
        while True:
            try:
                message = self.server.sse_queue.get(timeout=10)
            except queue.Empty:
                return
            if message is None:
                return
            self.wfile.write(b"event: message\ndata: " + json.dumps(message).encode() + b"\n\n")
            self.wfile.flush()

    def do_POST(self) -> None:
        self.server.seen_headers.append(dict(self.headers))
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if self.path == "/huge":
            data = b'{"jsonrpc":"2.0","id":0,"result":{"pad":"' + b"x" * (2 * 1024 * 1024) + b'"}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/redirect":
            self.send_response(307)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if "id" not in body:
            self._empty(202)
            return
        reply = {"jsonrpc": "2.0", "id": body["id"], "result": _result(body["method"], body.get("params") or {})}
        if self.path.startswith("/messages"):
            self.server.sse_queue.put(reply)
            self._empty(202)
            return
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", "session-1")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen_headers: list[dict[str, str]] = []
        self.sse_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()


@pytest.fixture
def fake_mcp() -> Iterator[str]:
    server = _FakeServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.sse_queue.put(None)
        server.shutdown()
        server.server_close()


def test_streamable_http_probe_lists_tools_with_opted_in_header_refs(fake_mcp: str) -> None:
    target = {
        "name": "docs",
        "url": f"{fake_mcp}/mcp",
        "transport": "http",
        "headers": {"Authorization": "Bearer ${DOCS_TOKEN}", "X-Missing": "$UNSET_VAR"},
    }

    entry = probe_mcp_server(
        target,
        allowed_private_hosts=LOCAL,
        expand_env=("DOCS_TOKEN", "UNSET_VAR"),
        environ={"DOCS_TOKEN": "tok-123"},
    )

    assert entry["status"] == "reachable-host"
    assert entry["tools"] == ["search", "get_page"]
    assert "streamable-http" in entry["detail"]
    assert "UNSET_VAR" in entry["detail"]
    assert "tok-123" not in json.dumps(entry)


def _spy_headers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    import skillevaluator.tier3.mcp_proof as proof

    seen: list[dict[str, str]] = []
    original = proof._capped_client

    def spy(headers: dict[str, str], **kwargs: Any):
        seen.append(dict(headers))
        return original(headers, **kwargs)

    monkeypatch.setattr(proof, "_capped_client", spy)
    return seen


def test_probe_expands_only_opted_in_variables(fake_mcp: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_headers(monkeypatch)
    target = {
        "name": "docs",
        "url": f"{fake_mcp}/mcp",
        "headers": {"Authorization": "Bearer ${DOCS_TOKEN}", "X-Trace": "$OTHER", "X-Client": "skilleval"},
    }

    probe_mcp_server(
        target, allowed_private_hosts=LOCAL, expand_env=("DOCS_TOKEN",), environ={"DOCS_TOKEN": "tok-123", "OTHER": "x"}
    )

    assert seen == [{"Authorization": "Bearer tok-123", "X-Client": "skilleval"}]


def test_probe_never_sends_host_variables_by_default(fake_mcp: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin picks both the URL and the header template; host secrets must not follow them."""
    seen = _spy_headers(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-host-secret-123")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_hostsecret456")
    target = {
        "name": "docs",
        "url": f"{fake_mcp}/mcp",
        "headers": {"X-A": "${ANTHROPIC_API_KEY}", "X-B": "$GITHUB_TOKEN", "X-Client": "skilleval"},
    }

    [entry] = probe_mcp_servers([target], allowed_private_hosts=LOCAL).values()

    assert seen == [{"X-Client": "skilleval"}]
    assert entry["status"] == "reachable-host"
    assert "ANTHROPIC_API_KEY" in entry["detail"] and "GITHUB_TOKEN" in entry["detail"]
    assert "--probe-mcp-env" in entry["detail"]
    assert "sk-ant-host-secret-123" not in json.dumps(entry)


def test_an_explicit_empty_environment_is_not_the_host_environment(
    fake_mcp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _spy_headers(monkeypatch)
    monkeypatch.setenv("DOCS_TOKEN", "from-the-host")
    target = {"name": "docs", "url": f"{fake_mcp}/mcp", "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}}

    entry = probe_mcp_server(target, allowed_private_hosts=LOCAL, expand_env=("DOCS_TOKEN",), environ={})

    assert seen == [{}]
    assert "headers skipped because unset: DOCS_TOKEN" in entry["detail"]


def test_probe_connects_only_to_policy_checked_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS rebinding: a name that re-resolves to a private address at connect time is never reached."""
    import anyio

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    real_getaddrinfo = socket.getaddrinfo

    def rebinding_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host in {"rebind.example", b"rebind.example"}:
            host = "127.0.0.1"
        return real_getaddrinfo(host, *args, **kwargs)

    real_connect = anyio.connect_tcp
    attempted: list[str] = []

    async def guarded_connect(remote_host: str, remote_port: int, **kwargs: Any) -> Any:
        attempted.append(str(remote_host))
        if remote_host == "93.184.216.34":
            raise OSError("public test address is unreachable")
        return await real_connect(remote_host, remote_port, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)
    monkeypatch.setattr(anyio, "connect_tcp", guarded_connect)
    target = {"name": "rebind", "url": f"https://rebind.example:{port}/mcp", "headers": {"X-Token": "${DOCS_TOKEN}"}}
    try:
        entry = probe_mcp_server(
            target,
            expand_env=("DOCS_TOKEN",),
            environ={"DOCS_TOKEN": "tok-123"},
            resolver=lambda _h, _p: ["93.184.216.34"],
            total_timeout=3,
        )
        listener.setblocking(False)
        with pytest.raises(BlockingIOError):
            listener.accept()
    finally:
        listener.close()

    assert attempted == ["93.184.216.34"]
    assert entry["status"] == "unreachable"


def test_pinned_client_refuses_other_hosts_and_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    import anyio
    import httpx

    from skillevaluator.tier3.mcp_proof import _capped_client

    attempted: list[str] = []

    async def recording_connect(remote_host: str, remote_port: int, **_kwargs: Any) -> Any:
        attempted.append(f"{remote_host}:{remote_port}")
        raise OSError("no network in tests")

    monkeypatch.setattr(anyio, "connect_tcp", recording_connect)

    async def fetch(url: str) -> None:
        client = _capped_client(
            {},
            url="https://mcp.example.com/mcp",
            addresses=("93.184.216.34", "93.184.216.35"),
            max_bytes=1024,
            connect_timeout=1,
            read_timeout=1,
        )
        async with client:
            await client.get(url)

    for url in ("https://metadata.example/mcp", "https://mcp.example.com:8443/mcp", "https://mcp.example.com/mcp"):
        with pytest.raises(httpx.ConnectError):
            anyio.run(fetch, url)

    assert attempted == ["93.184.216.34:443", "93.184.216.35:443"]


def test_sse_probe(fake_mcp: str) -> None:
    entry = probe_mcp_server(
        {"name": "legacy", "url": f"{fake_mcp}/sse", "transport": "sse"}, allowed_private_hosts=LOCAL
    )

    assert entry["status"] == "reachable-host", entry
    assert entry["tools"] == ["search", "get_page"]


def test_private_endpoints_are_not_probed_without_an_allowlist(fake_mcp: str) -> None:
    entry = probe_mcp_server({"name": "docs", "url": f"{fake_mcp}/mcp"})

    assert entry["status"] == "declared"
    assert "allowed_private_hosts" in entry["detail"]


@pytest.mark.parametrize(
    "url",
    ["https://169.254.169.254/mcp", "https://metadata.google.internal/mcp", "https://[fd00:ec2::254]/mcp"],
)
def test_metadata_endpoints_are_never_probed_even_when_allowlisted(url: str) -> None:
    entry = probe_mcp_server({"name": "meta", "url": url}, allowed_private_hosts=("0.0.0.0/0", "::/0", "*.internal"))

    assert entry["status"] == "declared"
    assert "metadata" in entry["detail"]


def test_public_names_resolving_to_private_addresses_are_refused() -> None:
    entry = probe_mcp_server(
        {"name": "pub", "url": "https://mcp.example.com/mcp"}, resolver=lambda _h, _p: ["10.0.0.7"]
    )

    assert entry["status"] == "declared"
    assert "10.0.0.7" in entry["detail"]


def test_plaintext_http_to_a_public_host_is_refused() -> None:
    entry = probe_mcp_server({"name": "pub", "url": "http://mcp.example.com/mcp"}, resolver=lambda _h, _p: ["1.1.1.1"])

    assert entry["status"] == "declared"
    assert "plaintext" in entry["detail"]


@pytest.mark.parametrize(
    ("url", "allowed", "addresses"),
    [
        ("https://mcp.corp.internal/mcp", ("mcp.corp.internal",), ["10.1.2.3"]),
        ("https://mcp.corp.internal/mcp", ("*.corp.internal",), ["10.1.2.3"]),
        ("https://mcp.corp.internal/mcp", ("10.0.0.0/8",), ["10.1.2.3"]),
        ("http://mcp.corp.internal/mcp", ("mcp.corp.internal",), ["10.1.2.3"]),
        ("http://mcp.corp.internal/mcp", ("10.0.0.0/8",), ["10.1.2.3"]),
        ("http://localhost:8080/mcp", ("localhost",), ["127.0.0.1", "::1"]),
        ("http://localhost:8080/mcp", ("127.0.0.1", "::1"), ["127.0.0.1", "::1"]),
    ],
)
def test_private_hosts_can_be_allowlisted_by_name_or_address(
    url: str, allowed: tuple[str, ...], addresses: list[str]
) -> None:
    kind, checked = _check_endpoint(url, "http", allowed, lambda _h, _p: addresses)

    assert (kind, checked) == ("streamable-http", tuple(addresses))


@pytest.mark.parametrize(
    ("url", "allowed", "addresses", "detail"),
    [
        # A name allowlist never admits a metadata address, and naming a metadata host never allowlists it.
        ("https://mcp.corp.internal/mcp", ("mcp.corp.internal",), ["169.254.169.254"], "metadata"),
        ("https://metadata.google.internal/mcp", ("metadata.google.internal",), ["10.0.0.1"], "metadata"),
        # Only the resolved ::1 is allowlisted, so the other loopback address is refused.
        ("http://localhost:8080/mcp", ("::1",), ["127.0.0.1", "::1"], "127.0.0.1"),
        # A name allowlist does not make plaintext http to a public address acceptable.
        ("http://mcp.example.com/mcp", ("mcp.example.com",), ["1.1.1.1"], "plaintext"),
        ("https://mcp.corp.internal/mcp", ("other.corp.internal",), ["10.1.2.3"], "allowed_private_hosts"),
    ],
)
def test_name_allowlisting_keeps_the_endpoint_policy(
    url: str, allowed: tuple[str, ...], addresses: list[str], detail: str
) -> None:
    with pytest.raises(_ProbeRefused) as refused:
        _check_endpoint(url, "http", allowed, lambda _h, _p: addresses)

    assert refused.value.status == "declared"
    assert detail in refused.value.detail


@pytest.mark.parametrize(
    ("target", "status"),
    [
        ({"name": "ws", "url": "wss://mcp.example.com/mcp"}, "unsupported"),
        ({"name": "grpc", "url": "https://mcp.example.com/mcp", "transport": "grpc"}, "unsupported"),
        ({"name": "ftp", "url": "ftp://mcp.example.com/mcp"}, "unsupported"),
    ],
)
def test_unsupported_transports(target: dict[str, Any], status: str) -> None:
    assert probe_mcp_server(target, resolver=lambda _h, _p: ["1.1.1.1"])["status"] == status


def test_unreachable_server() -> None:
    entry = probe_mcp_server({"name": "dead", "url": "http://127.0.0.1:1/mcp"}, allowed_private_hosts=LOCAL)

    assert entry["status"] == "unreachable"


def test_redirects_are_not_followed(fake_mcp: str) -> None:
    entry = probe_mcp_server({"name": "r", "url": f"{fake_mcp}/redirect"}, allowed_private_hosts=LOCAL)

    assert entry["status"] == "unreachable"


def test_oversized_responses_are_capped(fake_mcp: str) -> None:
    entry = probe_mcp_server({"name": "big", "url": f"{fake_mcp}/huge"}, allowed_private_hosts=LOCAL)

    assert entry["status"] == "unreachable"


def test_probe_servers_is_bounded_and_skips_nameless(fake_mcp: str) -> None:
    targets = [{"name": "", "url": f"{fake_mcp}/mcp"}, {"name": "docs", "url": f"{fake_mcp}/mcp"}]

    proof = probe_mcp_servers(targets, allowed_private_hosts=LOCAL)

    assert list(proof) == ["docs"]
    assert proof["docs"]["status"] == "reachable-host"


def _engine(by_server: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {"agents": {"codex": {"plugin_signals_summary": {"with_skill": {"mcp_calls": {"by_server": by_server}}}}}}


def test_in_agent_evidence_upgrades_and_never_downgrades() -> None:
    proof = {
        "docs": {"status": "reachable-host", "tools": ["search"], "detail": "ok"},
        "tracker": {"status": "declared", "tools": [], "detail": NOT_REQUESTED_DETAIL},
        "idle": {"status": "unreachable", "tools": [], "detail": "down"},
    }

    upgraded = apply_in_agent_mcp_proof(
        proof, _engine({"docs": {"total": 3, "succeeded": 2}, "tracker": {"total": 1, "succeeded": 0}})
    )

    assert upgraded["docs"]["status"] == "used-successfully"
    assert upgraded["docs"]["tools"] == ["search"]
    assert "host probe: ok" in upgraded["docs"]["detail"]
    # Called, but nothing succeeded: not proof that the server is reachable (proof M30).
    assert upgraded["tracker"]["status"] == "called-no-success"
    assert upgraded["idle"] == proof["idle"]
    assert apply_in_agent_mcp_proof(upgraded, None) == upgraded


def test_declared_default_without_probe() -> None:
    proof = declared_mcp_proof([{"name": "docs", "url": "https://docs.example.com/mcp"}])

    assert proof == {"docs": {"status": "declared", "tools": [], "detail": NOT_REQUESTED_DETAIL}}


def _plugin(tmp_path: Path) -> Path:
    plugin = tmp_path / "demo-plugin"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "demo-plugin",
                "description": "Demo plugin.",
                "mcpServers": {
                    "docs": {
                        "url": "https://docs.example.com/mcp",
                        "type": "http",
                        "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"},
                    },
                    "local": {"command": "npx", "args": ["-y", "@example/server@1.2.3"]},
                },
            }
        ),
        encoding="utf-8",
    )
    skill = plugin / "skills" / "alpha"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: alpha\ndescription: Alpha skill.\n---\nBody\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(json.dumps([{"id": "c1", "question": "q"}]), encoding="utf-8")
    (plugin / "agents").mkdir()
    (plugin / "agents" / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Reviews.\n---\nPrompt\n", encoding="utf-8"
    )
    return plugin


def test_package_exposes_url_probe_targets_and_cli_defaults_to_declared(tmp_path: Path) -> None:
    prepared = prepare_plugin_eval_package(_plugin(tmp_path), stage_root=tmp_path / "stage")

    assert [(t["name"], t["url"], t["transport"]) for t in prepared.mcp_probe_targets] == [
        ("docs", "https://docs.example.com/mcp", "http")
    ]
    assert prepared.mcp_probe_targets[0]["headers"] == {"Authorization": "Bearer ${DOCS_TOKEN}"}
    # Header values never reach provenance.
    assert "DOCS_TOKEN" not in json.dumps(prepared.provenance())
    assert _plugin_mcp_proof(prepared, probe_mcp=False) == {
        "docs": {"status": "declared", "tools": [], "detail": NOT_REQUESTED_DETAIL}
    }
    components = json.loads(
        (prepared.package_path / "evals" / "environment" / "plugin_runtime_components.json").read_text(encoding="utf-8")
    )
    assert components == {"subagents": ["reviewer"], "commands": []}


def test_validate_forwards_probe_mcp_env_to_the_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import skillevaluator.cli as cli_module
    from skillevaluator.evaluation import EvaluationService

    captured: dict[str, Any] = {}

    def fake_probe(targets: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("skillevaluator.tier3.mcp_proof.probe_mcp_servers", fake_probe)
    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options, **_kwargs: {})
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.agent_eval_result_from_run", lambda *_a, **_k: None)

    cli_module._run_agent_eval_or_skip(
        _plugin(tmp_path),
        agents="codex",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=1,
        max_agents=1,
        kind="plugin",
        probe_mcp=True,
        probe_mcp_env=("DOCS_TOKEN",),
        allowed_private_hosts=("10.0.0.0/8",),
    )

    assert captured == {"allowed_private_hosts": ("10.0.0.0/8",), "expand_env": ("DOCS_TOKEN",)}


@pytest.mark.parametrize("command", [["validate"], ["tier3", "evaluate-plugin"]])
def test_probe_mcp_env_rejects_names_that_are_not_variables(command: list[str], tmp_path: Path) -> None:
    from click.testing import CliRunner

    from skillevaluator.cli import cli

    result = CliRunner().invoke(cli, [*command, str(tmp_path), "--probe-mcp", "--probe-mcp-env", "NOT-A-NAME"])

    assert result.exit_code != 0
    assert "not an environment variable name" in result.output


# --------------------------------------------------------------------------- #
# Probe deadlines, fatal transport errors, output hygiene, env scoping        #
# --------------------------------------------------------------------------- #
def test_an_oversized_response_ends_the_probe_at_once_without_mcp_tracebacks(
    fake_mcp: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    started = time.monotonic()

    entry = probe_mcp_server({"name": "big", "url": f"{fake_mcp}/huge"}, allowed_private_hosts=LOCAL, total_timeout=10)

    assert time.monotonic() - started < 5
    assert entry["status"] == "unreachable"
    assert "response exceeded 1048576 bytes" in entry["detail"]
    assert not [record for record in caplog.records if record.name.startswith("mcp")]


def test_an_off_origin_sse_endpoint_fails_fast(fake_mcp: str) -> None:
    started = time.monotonic()

    entry = probe_mcp_server(
        {"name": "sse", "url": f"{fake_mcp}/sse-offorigin", "transport": "sse"},
        allowed_private_hosts=LOCAL,
        total_timeout=10,
    )

    assert time.monotonic() - started < 5
    assert entry["status"] == "unreachable"
    assert "origin" in entry["detail"]


def test_a_slow_dns_lookup_counts_against_the_probe_deadline() -> None:
    def slow_resolver(_host: str, _port: int) -> list[str]:
        time.sleep(5)
        return ["93.184.216.34"]

    started = time.monotonic()
    entry = probe_mcp_server(
        {"name": "docs", "url": "https://docs.example.com/mcp"}, resolver=slow_resolver, total_timeout=0.5
    )

    assert time.monotonic() - started < 2
    assert entry["status"] == "unreachable"
    assert "DNS resolution timed out" in entry["detail"]


def test_probed_tool_names_and_details_are_printable(fake_mcp: str, monkeypatch: pytest.MonkeyPatch) -> None:
    hostile = [
        {"name": "\x1b[31mred\x1b[0m", "inputSchema": {"type": "object"}},
        {"name": "\x1b]0;pwned\x07title", "inputSchema": {"type": "object"}},
        {"name": "bell\x07\x9bcsi", "inputSchema": {"type": "object"}},
    ]
    monkeypatch.setitem(globals(), "TOOLS", hostile)

    entry = probe_mcp_server({"name": "docs", "url": f"{fake_mcp}/mcp"}, allowed_private_hosts=LOCAL)

    assert entry["status"] == "reachable-host"
    assert entry["tools"] == ["red", "title", "bellcsi"]
    assert not any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in json.dumps(entry, ensure_ascii=False))


def _scoped_targets(fake_mcp: str) -> list[dict[str, Any]]:
    port = fake_mcp.rsplit(":", 1)[1]
    headers = {"Authorization": "Bearer ${DOCS_TOKEN}"}
    return [
        {"name": "docs", "url": f"http://docs.test:{port}/mcp", "headers": headers},
        {"name": "collector", "url": f"http://collector.test:{port}/mcp", "headers": headers},
    ]


@pytest.mark.parametrize(
    ("grant", "expected"),
    [
        ("DOCS_TOKEN=docs.test", [{"Authorization": "Bearer tok-123"}, {}]),
        ("DOCS_TOKEN@collector", [{}, {"Authorization": "Bearer tok-123"}]),
        ("DOCS_TOKEN", [{"Authorization": "Bearer tok-123"}, {"Authorization": "Bearer tok-123"}]),
    ],
)
def test_probe_mcp_env_can_be_scoped_to_one_host_or_server(
    fake_mcp: str, monkeypatch: pytest.MonkeyPatch, grant: str, expected: list[dict[str, str]]
) -> None:
    seen = _spy_headers(monkeypatch)

    proof = probe_mcp_servers(
        _scoped_targets(fake_mcp),
        allowed_private_hosts=LOCAL,
        expand_env=(grant,),
        environ={"DOCS_TOKEN": "tok-123"},
        resolver=lambda _host, _port: ["127.0.0.1"],
    )

    assert seen == expected
    for (name, entry), headers in zip(proof.items(), expected, strict=True):
        host = f"{name}.test"
        assert entry["status"] == "reachable-host"
        assert (f"sent DOCS_TOKEN to {host}" in entry["detail"]) is bool(headers), entry["detail"]
        assert "tok-123" not in json.dumps(entry)


def test_cli_prints_which_variables_go_to_which_host_before_probing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        "skillevaluator.tier3.mcp_proof.probe_mcp_servers", lambda *_a, **_k: order.append("probe") or {}
    )
    prepared = type(
        "Prepared",
        (),
        {"mcp_probe_targets": _scoped_targets("http://127.0.0.1:8080")},
    )()

    _plugin_mcp_proof(prepared, probe_mcp=True, probe_mcp_env=("DOCS_TOKEN=docs.test",))

    output = capsys.readouterr().out
    assert "MCP probe may send DOCS_TOKEN to docs.test (server docs" in " ".join(output.split())
    assert "collector.test" not in output
    assert order == ["probe"]


@pytest.mark.parametrize(
    "value", ["DOCS_TOKEN=", "DOCS_TOKEN=bad host", "DOCS_TOKEN=docs.test:443", "DOCS_TOKEN@", "1TOKEN=docs.test"]
)
def test_probe_mcp_env_rejects_malformed_scopes(value: str, tmp_path: Path) -> None:
    from click.testing import CliRunner

    from skillevaluator.cli import cli

    result = CliRunner().invoke(cli, ["tier3", "evaluate-plugin", str(tmp_path), "--probe-mcp-env", value])

    assert result.exit_code != 0
    assert "--probe-mcp-env" in result.output
