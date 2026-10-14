# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution diagnostics remain visible alongside genuine content findings."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models import Finding, Severity, ValidationResult


class _VisibleHTMLText(HTMLParser):
    """Exclude diagnostics present only in the report's embedded JSON."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, _attrs) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag) -> None:
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data) -> None:
        if not self.hidden:
            self.parts.append(data)


@pytest.mark.parametrize("outcome", ["mixed", "errors_only", "finding_only"])
def test_tier2_cli_reports_preserve_analysis_errors_with_findings(tmp_path: Path, monkeypatch, outcome: str) -> None:
    """Exercise collection, clustering, SDK requests and all reports over loopback HTTP."""
    requests: list[str] = []

    class AnalysisHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(self.path)
            if self.path == "/v1/embeddings":
                payload = {
                    "data": [
                        {"index": index, "embedding": [float(index // 2 == dimension) for dimension in range(2)]}
                        for index in range(len(body["input"]))
                    ]
                }
            elif self.path == "/v1/chat/completions":
                first_cluster = "Section A" in body["messages"][-1]["content"]
                verdict = "DUPLICATE" if first_cluster else "RELATED_BUT_DISTINCT"
                if outcome == "errors_only" or (outcome == "mixed" and not first_cluster):
                    verdict = "INVALID"
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
                                        "verdict": verdict,
                                        "confidence": 0.9,
                                        "reasoning": "Repeated instructions",
                                        "suggestion": "Consolidate the repeated sections",
                                    }
                                ),
                            },
                        }
                    ],
                }
            else:
                self.send_error(404)
                return
            response = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), AnalysisHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "\n".join(f"## Section {letter}\n{letter.lower() * 200}" for letter in "ABCD"), encoding="utf-8"
    )
    reports = tmp_path / "reports"
    for purpose in ("LLM", "EMBEDDING"):
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_PROVIDER", "openai-compatible")
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_API_KEY", "local-test-key")
        monkeypatch.setenv(f"SKILL_EVAL_{purpose}_MODEL", "local-test-model")
    try:
        invocation = CliRunner().invoke(cli, ["tier2", str(skill), "-r", "cli,json,html,markdown", "-o", str(reports)])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert invocation.exit_code == 1, invocation.output
    assert requests.count("/v1/embeddings") == 1
    assert requests.count("/v1/chat/completions") == 2
    payload = json.loads((reports / "skillevaluator-tier2.json").read_text(encoding="utf-8"))
    result = payload["results"][0]
    failed_clusters = {"mixed": 1, "errors_only": 2, "finding_only": 0}[outcome]
    assert result["status"] == ("incomplete" if failed_clusters else "failed")
    assert result["llm_analysis"]["clusters_total"] == 2
    assert result["llm_analysis"]["clusters_failed"] == failed_clusters
    assert result["llm_analysis"]["clusters_completed"] == 2 - failed_clusters
    # Provider failures remain execution evidence rather than duplicate findings.
    assert [finding["check_name"] for finding in result["findings"]] == (
        [] if outcome == "errors_only" else ["duplicate"]
    )

    html = _VisibleHTMLText()
    html.feed((reports / "skillevaluator-tier2.html").read_text(encoding="utf-8"))
    # Earlier logger output must not mask omissions from the actual CLI report.
    marker = "SkillEvaluator Validation Results"
    assert marker in invocation.output
    outputs = {
        "cli": invocation.output[invocation.output.index(marker) :],
        "html": " ".join(html.parts),
        "markdown": (reports / "skillevaluator-tier2.md").read_text(encoding="utf-8"),
    }
    outputs = {name: " ".join(text.split()) for name, text in outputs.items()}
    diagnostic = "LLM analysis did not complete"
    expected_diagnostics = int(failed_clusters > 0)
    assert {name: text.count(diagnostic) for name, text in outputs.items()} == dict.fromkeys(
        outputs, expected_diagnostics
    )
    if failed_clusters:
        for text in outputs.values():
            assert f"{failed_clusters} of 2 content clusters" in text
            assert "structured-output support" in text
    duplicate_message = "Duplicate content found within SKILL.md:"
    # Markdown intentionally repeats the finding in its table and details.
    expected_findings = {"cli": 1, "html": 1, "markdown": 2}
    assert {name: text.count(duplicate_message) for name, text in outputs.items()} == {
        name: count if outcome != "errors_only" else 0 for name, count in expected_findings.items()
    }


@pytest.mark.parametrize("prefixed", [False, True])
def test_additional_errors_excludes_only_existing_finding_messages_without_mutation(prefixed: bool) -> None:
    from skillevaluator.reporting.base import additional_errors

    result = ValidationResult(validator_name="Context Deduplication")
    result.add_finding(Finding("DUPLICATE", Severity.HIGH, "duplicate", "Repeated instructions", "SKILL.md"))
    diagnostic = "LLM analysis failed while reviewing Repeated instructions"
    result.add_error(diagnostic)
    if prefixed:
        merged = ValidationResult(validator_name="Context Deduplication")
        merged.merge_with_prefix(result, "sample-skill")
        result = merged
        diagnostic = f"[sample-skill] {diagnostic}"
    before = asdict(result)

    assert additional_errors(result) == [diagnostic]
    assert asdict(result) == before
