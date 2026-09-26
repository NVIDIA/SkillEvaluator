# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 command wrappers preserve useful failure diagnostics."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.tier2 import commands


def test_similarity_check_wraps_unexpected_validator_exceptions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class BrokenSimilarityValidator:
        def __init__(self, **_kwargs):
            pass

        def validate(self, _content_path: Path):
            raise RuntimeError("provider body contains private-token")

    monkeypatch.setattr(commands, "SimilarityValidator", BrokenSimilarityValidator)

    (result,) = commands.run_similarity_check(tmp_path, catalog=tmp_path / "catalog.json")

    assert result.validator_name == "Similarity Check"
    assert result.validator_description == "Tier 2 check"
    assert result.passed is False
    assert result.status == "incomplete"
    assert result.findings == []
    assert "unexpected error" in result.errors[0]
    assert "private-token" not in result.errors[0]


@pytest.mark.parametrize("during_construction", [False, True])
def test_context_check_preserves_incomplete_contract_for_unexpected_errors(
    tmp_path: Path, monkeypatch, during_construction: bool
) -> None:
    class BrokenValidator:
        def __init__(self, **_kwargs):
            if during_construction:
                raise RuntimeError("private provider details")

        def validate(self, _path):
            raise RuntimeError("private provider details")

    monkeypatch.setattr(commands, "IntraSkillValidator", BrokenValidator)

    (result,) = commands.run_context_optimization_check(tmp_path)

    assert result.status == "incomplete"
    assert not result.passed
    assert result.findings == []
    assert "private provider details" not in result.errors[0]


def test_detected_duplicate_remains_failed_not_incomplete(tmp_path: Path, monkeypatch) -> None:
    result = ValidationResult(validator_name="Context Deduplication")
    result.add_finding(Finding("DUPLICATE", Severity.HIGH, "duplicate", "Repeated instructions", "SKILL.md"))

    class DuplicateValidator:
        def __init__(self, **_kwargs):
            pass

        def validate(self, _path):
            return result

    monkeypatch.setattr(commands, "IntraSkillValidator", DuplicateValidator)

    (actual,) = commands.run_context_optimization_check(tmp_path)

    assert actual.status == "failed"
    assert actual.findings[0].category == "DUPLICATE"


def test_http_410_is_incomplete_in_real_cli_and_reports(tmp_path: Path, monkeypatch) -> None:
    """Exercise the actual SDK against a local HTTP service, without external calls."""
    requests = []

    class RetiredModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            response = json.dumps(
                {"error": {"message": "Retired model: provider-private-token", "type": "model_retired"}}
            ).encode()
            self.send_response(410)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RetiredModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("## Section A\n" + "a" * 200 + "\n## Section B\n" + "b" * 200)
    reports = tmp_path / "reports"
    for purpose in ("LLM", "EMBEDDING"):
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_PROVIDER", "openai-compatible")
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_API_KEY", "local-test-key")
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_MODEL", "local-test-model")
    try:
        invocation = CliRunner().invoke(cli, ["tier2", str(skill), "-o", str(reports)])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert invocation.exit_code == 1, invocation.output
    assert len(requests) == 1
    assert requests[0][0] == "/v1/embeddings"
    assert requests[0][1]["model"] == "local-test-model"
    assert "INCOMPLETE" in invocation.output
    payload = json.loads((reports / "skillevaluator-tier2.json").read_text())
    html = (reports / "skillevaluator-tier2.html").read_text()
    assert payload["overall_status"] == "incomplete"
    assert payload["overall_passed"] is False
    assert payload["results"][0]["status"] == "incomplete"
    assert payload["results"][0]["findings"] == []
    assert "Incomplete" in html
    for output in (invocation.output, json.dumps(payload), html):
        assert "provider-private-token" not in output


@pytest.mark.parametrize("chat_status", [200, 400, 410, 429])
def test_chat_http_diagnostics_after_successful_embedding(tmp_path: Path, monkeypatch, chat_status: int) -> None:
    """Run collection, SDK requests, clustering, analysis and reports over loopback HTTP."""
    requests = []

    class ChatHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path == "/v1/embeddings":
                status = 200
                payload = {
                    "data": [
                        {"index": index, "embedding": [float(dimension == index // 2) for dimension in range(6)]}
                        for index in range(len(body["input"]))
                    ]
                }
            elif self.path == "/v1/chat/completions":
                status = chat_status
                if status == 200:
                    payload = {
                        "id": "local-completion",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {
                                            "verdict": "RELATED_BUT_DISTINCT",
                                            "confidence": 0.9,
                                            "reasoning": "Separate tasks",
                                            "suggestion": "Keep both",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                else:
                    payload = {
                        "error": {
                            "message": "provider-private-token with private input content",
                            "code": "unsupported_parameter" if status == 400 else "model_retired",
                            "param": "temperature",
                            "type": "provider-private-token",
                        }
                    }
            else:
                status, payload = 404, {}
            response = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("x-should-retry", "false")
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "\n".join(f"## Section {letter}\n{letter.lower() * 200}" for letter in "ABCDEFGHIJKL")
    )
    reports = tmp_path / "reports"
    for purpose in ("LLM", "EMBEDDING"):
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_PROVIDER", "openai-compatible")
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_API_KEY", "local-test-key")
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_MODEL", f"local-{purpose.lower()}-model")
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("SKILL_EVAL_LLM_RETRY_MAX_DELAY", "0")
    try:
        invocation = CliRunner().invoke(cli, ["tier2", str(skill), "-o", str(reports)])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert requests[0][0] == "/v1/embeddings"
    chat_requests = [body for path, body in requests if path == "/v1/chat/completions"]
    assert len(chat_requests) == 6 * (3 if chat_status == 429 else 1)
    assert all(body["model"] == "local-llm-model" and body["stream"] is False for body in chat_requests)
    payload = json.loads((reports / "skillevaluator-tier2.json").read_text())
    html = (reports / "skillevaluator-tier2.html").read_text()
    result = payload["results"][0]
    assert result["findings"] == []
    analysis = result["llm_analysis"]
    assert analysis["provider"] == "openai-compatible"
    assert analysis["model"] == "local-llm-model"
    assert analysis["clusters_total"] == 6
    if chat_status == 200:
        assert invocation.exit_code == 0, invocation.output
        assert payload["overall_status"] == "passed"
        assert analysis["clusters_completed"] == 6
        assert analysis["clusters_failed"] == 0
        assert analysis["failures"] == []
    else:
        assert invocation.exit_code == 1, invocation.output
        assert payload["overall_status"] == "incomplete"
        assert result["status"] == "incomplete"
        assert analysis["clusters_failed"] == 6
        assert analysis["clusters_completed"] == 0
        errors = result["legacy"]["errors"]
        assert len(errors) == len(analysis["failures"]) == 1
        assert analysis["failures"][0]["count"] == 6
        for output in (invocation.output, json.dumps(payload), html):
            assert f"HTTP {chat_status}" in output
            assert "6 of 6 content clusters" in output
            assert "local-llm-model" in output
            assert "provider-private-token" not in output
            assert "private input content" not in output
            assert "local-test-key" not in output
        if chat_status == 400:
            assert "code=unsupported_parameter" in errors[0]
            assert "param=temperature" in errors[0]
        elif chat_status == 410:
            assert "SKILL_EVAL_LLM_MODEL" in errors[0]
        elif chat_status == 429:
            assert "quota" in errors[0]
