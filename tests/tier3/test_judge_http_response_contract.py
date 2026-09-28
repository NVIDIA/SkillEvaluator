# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the standalone verifier against an endpoint that defaults to SSE."""

from __future__ import annotations

import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest


@pytest.mark.parametrize("provider", ["nv_build", "openai", "openai-compatible", "anthropic"])
def test_standalone_judge_requests_json_from_streaming_default_endpoint(monkeypatch, provider):
    template_path = Path(__file__).resolve().parents[2] / "src/skillevaluator/tier3/harbor/templates/eval.py"
    spec = importlib.util.spec_from_file_location("judge_http_response_contract", template_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    requests = []
    judge_reply = json.dumps({"score": 1.0, "reason": "The answer matches the expected result."})

    class StreamingDefaultEndpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "payload": payload})
            if payload.get("stream") is not False:
                # Omission is a real protocol mismatch: a single JSON parser
                # cannot consume the endpoint's default event stream.
                body = b'data: {"choices":[{"delta":{"content":"judge output"}}]}\n\ndata: [DONE]\n\n'
                content_type = "text/event-stream"
            else:
                response = (
                    {"content": [{"type": "text", "text": judge_reply}]}
                    if provider == "anthropic"
                    else {"choices": [{"message": {"content": judge_reply}}]}
                )
                body = json.dumps(response).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    for name in (
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "SKILL_EVAL_LLM_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "LLM_JUDGE_MODEL",
        "SKILL_EVAL_JUDGE_MODEL",
        "LLM_JUDGE_FALLBACK_MODELS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "test-judge")
    credential = {
        "nv_build": "NVIDIA_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openai-compatible": "SKILL_EVAL_LLM_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider]
    monkeypatch.setenv(credential, "local-test-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    with ThreadingHTTPServer(("127.0.0.1", 0), StreamingDefaultEndpoint) as server:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        monkeypatch.setenv(
            "SKILL_EVAL_LLM_BASE_URL", base_url + "/chat/completions" if provider == "nv_build" else base_url
        )
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = verifier.judge_accuracy("What is two plus two?", "4", "4")
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert result["score"] == 1.0
    assert len(requests) == 1
    assert requests[0]["path"] == ("/v1/messages" if provider == "anthropic" else "/v1/chat/completions")
    assert requests[0]["payload"]["stream"] is False
    assert requests[0]["payload"]["model"] == "test-judge"
