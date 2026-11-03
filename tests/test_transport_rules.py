# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP retry rules and NVIDIA Build bridge tool aliasing."""

from __future__ import annotations

import importlib.util
import ssl
import urllib.error
from pathlib import Path

import pytest

from skillevaluator.inference.retry import is_retriable_exception, is_retriable_status_code
from skillevaluator.tier3 import harbor
from skillevaluator.tier3.harbor.nvidia_build_bridge import (
    BridgePayloadError,
    anthropic_to_chat_request,
    chat_completion_to_anthropic_events,
)


def _template():
    path = Path(harbor.__file__).parent / "templates" / "eval.py"
    spec = importlib.util.spec_from_file_location("skilleval_eval_template", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("status", "retriable"),
    [
        (408, True),
        (429, True),
        (500, True),
        (520, True),
        (529, True),
        (501, False),
        (505, False),
        (400, False),
        (404, False),
        (True, False),
    ],
)
def test_engine_and_verifier_share_the_status_rule(status, retriable) -> None:
    assert is_retriable_status_code(status) is retriable
    assert _template()._is_retriable_http_status(status) is retriable


def test_certificate_failures_are_terminal_in_engine_and_verifier() -> None:
    wrapped = urllib.error.URLError(ssl.SSLCertVerificationError("bad ca"))
    assert _template()._is_certificate_failure(wrapped)
    assert not _template()._is_certificate_failure(urllib.error.URLError(ConnectionResetError()))
    assert not is_retriable_exception(wrapped)


def _tool(name: str) -> dict:
    return {"name": name, "description": "d", "input_schema": {"type": "object"}}


def test_long_tool_names_alias_stably_and_map_back_on_responses() -> None:
    long_name = f"mcp__plugin__{'x' * 70}"
    request = {
        "model": "nvidia/m",
        "messages": [{"role": "user", "content": "go"}],
        "tools": [_tool(long_name)],
        "tool_choice": {"type": "tool", "name": long_name},
    }
    first = anthropic_to_chat_request(request)
    alias = first["tools"][0]["function"]["name"]
    assert len(alias) == 64 and alias == anthropic_to_chat_request(request)["tools"][0]["function"]["name"]
    assert first["tool_choice"]["function"]["name"] == alias
    from skillevaluator.tier3.harbor.nvidia_build_bridge import _anthropic_to_chat_request

    _, names = _anthropic_to_chat_request(request)
    completion = {
        "id": "c",
        "model": "nvidia/m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "t1", "type": "function", "function": {"name": alias, "arguments": "{}"}}],
                },
            }
        ],
    }
    starts = [
        e for e in chat_completion_to_anthropic_events(completion, names) if e.get("type") == "content_block_start"
    ]
    assert [e["content_block"]["name"] for e in starts if e["content_block"]["type"] == "tool_use"] == [long_name]


def test_aliases_that_collide_within_one_request_are_refused() -> None:
    from skillevaluator.tier3.harbor import nvidia_build_bridge as bridge

    long_name = f"mcp_{'y' * 80}"
    alias = bridge._chat_tool_alias(long_name)
    request = {
        "model": "nvidia/m",
        "messages": [{"role": "user", "content": "go"}],
        "tools": [_tool(long_name), _tool(alias)],
    }
    with pytest.raises(BridgePayloadError, match="collision"):
        anthropic_to_chat_request(request)


def test_non_text_tool_results_keep_their_pairing_as_placeholders() -> None:
    request = {
        "model": "nvidia/m",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "shot", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "ok "},
                            {"type": "image", "source": {"type": "base64", "data": "AA=="}},
                        ],
                    }
                ],
            },
        ],
        "tools": [_tool("shot")],
    }
    tool_messages = [m for m in anthropic_to_chat_request(request)["messages"] if m["role"] == "tool"]
    assert tool_messages[0]["tool_call_id"] == "t1"
    assert tool_messages[0]["content"].startswith("ok [image block omitted")


def _loopback(statuses: list[int]):
    import http.server
    import threading

    seen: list[int] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            status = statuses[min(len(seen), len(statuses) - 1)]
            seen.append(status)
            self.send_response(status)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


@pytest.mark.parametrize(("statuses", "calls", "ok"), [([503, 408, 200], 3, True), ([501, 200], 1, False)])
def test_verifier_retries_transient_loopback_statuses_only(monkeypatch, statuses, calls, ok) -> None:
    import urllib.request

    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "0")
    server, seen = _loopback(statuses)
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/")
        module = _template()
        if ok:
            assert module._urlopen_with_retry(request, timeout=10) == b"ok"
        else:
            with pytest.raises(urllib.error.HTTPError):
                module._urlopen_with_retry(request, timeout=10)
    finally:
        server.shutdown()
    assert len(seen) == calls
