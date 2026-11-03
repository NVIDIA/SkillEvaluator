# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check 24: failure words in a good MCP answer must not mark the call failed.

Shapes follow the proof's check-24 e03 (good docs answers that talk about
errors), p02 (an unflagged server "Error: ..." text is a success), the skeptic
probes ("AT-412:", "400-500 rps", "Error: ENOENT means ..."), and the failure
texts of p01, p05 and p07 that must keep failing. Without a structured flag,
failure text is read only on a result's lead lines, and only harness,
transport and HTTP status text counts.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals

DECLARED = {"skill": [], "mcp": ["deepwiki", "reltools"]}
CODEX_STATUS = "Wall time: 0.1305 seconds\nOutput:\n"


def _outcomes(contents: list[str], *, codex: bool) -> list[bool | None]:
    steps = []
    servers = {}
    for index, content in enumerate(contents):
        call_id = f"call_{index}"
        fn = "ask_question" if codex else "mcp__plugin_docs_deepwiki__ask_question"
        servers[call_id] = "deepwiki"
        steps.append(
            {
                "source": "agent",
                "message": "",
                "tool_calls": [{"tool_call_id": call_id, "function_name": fn, "arguments": {"q": "x"}}],
                "observation": {"results": [{"source_call_id": call_id, "content": content}]},
            }
        )
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "codex" if codex else "claude-code"},
        "steps": [{"source": "user", "message": "Ask the docs."}, *steps],
    }
    signals = compute_plugin_signals(trajectory, {}, declared=DECLARED, mcp_call_servers=servers if codex else None)
    assert signals is not None
    return [activation["succeeded"] for activation in signals["activations"]]


GOOD_ANSWERS = [
    "403: Forbidden is returned when the token lacks the repo scope; add the scope and retry.",
    "Permission denied errors happen when the socket file is owned by root; run as the docker group.",
    "Streaming is not available on the free plan; the paid plan enables it with stream=True.",
    "Unauthorized requests get a 401 response; pass the API key in the Authorization header.",
]


def test_good_answers_that_talk_about_errors_succeed_in_claude_shape() -> None:
    # check-24 e03, Claude Code shape: the answer is a JSON result body.
    contents = [json.dumps({"result": text}) for text in GOOD_ANSWERS]
    assert _outcomes(contents, codex=False) == [True, True, True, True]


def test_good_answers_that_talk_about_errors_succeed_in_codex_shape() -> None:
    # check-24 e03, Codex shape: status lines, then the result body (no codex.txt status).
    contents = [CODEX_STATUS + json.dumps({"result": text}) for text in GOOD_ANSWERS]
    assert _outcomes(contents, codex=True) == [True, True, True, True]


def test_good_answers_succeed_as_plain_text_and_in_an_mcp_text_block() -> None:
    # check-24 e03 answers as a server sends them without a {"result": ...}
    # wrapper: plain text, Claude Code's MCP block, and Codex's printed block.
    plain = list(GOOD_ANSWERS)
    claude_block = [json.dumps({"type": "text", "text": text}) for text in GOOD_ANSWERS]
    codex_block = [
        CODEX_STATUS + json.dumps([{"type": "text", "text": text}], separators=(",", ":")) for text in GOOD_ANSWERS
    ]
    assert _outcomes(plain, codex=False) == [True, True, True, True]
    assert _outcomes(claude_block, codex=False) == [True, True, True, True]
    assert _outcomes([CODEX_STATUS + text for text in plain], codex=True) == [True, True, True, True]
    assert _outcomes(codex_block, codex=True) == [True, True, True, True]


def _p02_trajectory(*, codex: bool) -> dict[str, Any]:
    # check-24 p02: five calls with the same unflagged server error text; four
    # carry a structured flag the code reads, the fifth (the control) has none.
    text = "Error: unknown tag '1.3.9' for atlas - known tags: 1.3.0, 1.4.2"
    body = CODEX_STATUS + json.dumps([{"type": "text", "text": text}], separators=(",", ":")) if codex else text
    results = [
        {"content": body, "is_error": True},
        {"content": body, "extra": {"isError": True}},
        {"content": {"isError": True, "content": [{"type": "text", "text": text}]}},
        {"content": body, "error": "tool failed"},
        {"content": body},
    ]
    fn = "list_changes" if codex else "mcp__plugin_release-kit_reltools__list_changes"
    steps = [
        {
            "source": "agent",
            "message": "",
            "tool_calls": [{"tool_call_id": f"c{index}", "function_name": fn, "arguments": {"project": "atlas"}}],
            "observation": {"results": [{"source_call_id": f"c{index}", **result}]},
        }
        for index, result in enumerate(results)
    ]
    return {"agent": {"name": "codex" if codex else "claude-code"}, "steps": steps}


@pytest.mark.parametrize("codex", [False, True], ids=["claude-code", "codex"])
def test_p02_unflagged_server_error_text_is_the_one_success(codex: bool) -> None:
    # Proof truth for check-24 p02: 5 total, 1 succeeded, 4 failed, 0 unknown in
    # both harness shapes. A server's own "Error: ..." text with no flag is an
    # answer, so Claude Code and Codex must agree on it.
    servers = {f"c{index}": "reltools" for index in range(5)} if codex else None
    signals = compute_plugin_signals(_p02_trajectory(codex=codex), {}, declared=DECLARED, mcp_call_servers=servers)
    assert signals is not None
    calls = signals["mcp_calls"]
    assert (calls["total"], calls["succeeded"], calls["failed"], calls["unknown"]) == (5, 1, 4, 0)
    assert calls["success_rate"] == 0.2
    assert [a["succeeded"] for a in signals["activations"]] == [False, False, False, False, True]


@pytest.mark.parametrize(
    "text",
    [
        "AT-412: Fix race in cache eviction\nAT-4821: Add retry budget to the sync client",
        "400-500 rps sustained during the soak test",
        "Peak load was 400-500 rps; p99 stayed under 40 ms.",
        "Permission denied errors happen when the socket file is owned by root.",
        "Streaming is not available on the free plan; upgrade to enable it.",
        "Forbidden words list: none configured.\nAll checks green.",
        "Error handling: wrap the call in try/except and retry once.",
        "Fix error: handle timeouts in the sync client",
        "Streaming is not available.",
        "Error: ENOENT means the path does not exist. Create the directory first.",
        "ERROR: disk full on node-3 at 12:00 (1 match in app.log)",
        "TypeError: is raised when an operation is applied to the wrong type.",
        "Release notes ready.\nFixed: retries after status code 503 from the registry",
        "404: page moved to /docs/v2",
        "429 - Too Many Requests means you hit the rate limit; back off and retry.",
        "Error: unknown tag '1.3.9' for atlas - known tags: 1.3.0, 1.4.2",
        "Get started: 404 Not Found pages are cached for an hour.",
        "Error handling: 404: page moved to /docs/v2",
        "Error: 500 - boom",
        "HTTP/1.1 404 responses are cached by the CDN for an hour.",
    ],
    ids=[
        "issue-key-412",
        "rps-range-start",
        "rps-range-mid",
        "perm-sentence",
        "not-available-mid",
        "forbidden-word",
        "error-handling-heading",
        "changelog-fix-error",
        "feature-not-available",
        "error-prefix-doc",
        "log-search-hit",
        "exception-doc",
        "status-code-in-last-line",
        "status-number-colon",
        "status-reason-in-sentence",
        "p02-unflagged-server-error",
        "request-word-in-a-heading",
        "error-heading-then-status-number",
        "error-prefix-status-without-reason",
        "http-version-status-in-prose",
    ],
)
def test_skeptic_probes_are_not_failures(text: str) -> None:
    wrapped = json.dumps([{"type": "text", "text": text}], separators=(",", ":"))
    assert _outcomes([text], codex=False) == [True]
    assert _outcomes([json.dumps({"type": "text", "text": text})], codex=False) == [True]
    assert _outcomes([CODEX_STATUS + wrapped], codex=True) == [True]


@pytest.mark.parametrize(
    "text",
    [
        "MCP error -32603: Internal error while reading the tag index",
        "401 - Unauthorized: missing release token",
        "Permission denied: notes_path must be inside /workspace",
        "MCP error -32000: Connection closed",
        "<tool_use_error>Error: No such tool available: mcp__plugin_docs_deepwiki__ask_question</tool_use_error>",
        "tool call error: deepwiki: connection refused",
        "error: failed to connect to MCP server 'deepwiki'",
        "HTTP/1.1 503 Service Unavailable",
        "Request failed with status code 404",
        "404 Not Found",
        "HTTP Error 404: Not Found",
        "AxiosError: Request failed with status code 401",
        "403: Forbidden - release token lacks scope",
        "Tool 'ask_question' is not available.",
        # A status line after a request line or an error clause (#180's cases).
        "GET /repos/x: 404 - Not Found",
        "Error calling tool: 404: not found",
        "Failed to fetch /v1/notes: 502 Bad Gateway",
        # An HTTP/x.y response status line needs no reason phrase.
        "HTTP/1.1 404 - x",
        "HTTP/2 500",
        "HTTP/1.1 502: upstream closed the connection",
    ],
)
def test_real_failure_texts_still_fail(text: str) -> None:
    wrapped = json.dumps([{"type": "text", "text": text}], separators=(",", ":"))
    assert _outcomes([text], codex=False) == [False]
    assert _outcomes([json.dumps({"type": "text", "text": text})], codex=False) == [False]
    assert _outcomes([CODEX_STATUS + wrapped], codex=True) == [False]


def test_an_error_line_appended_after_a_long_answer_still_fails() -> None:
    body = "x" * 5000 + "\n\n[error] tool reported failure"
    assert _outcomes([body], codex=False) == [False]


def test_harness_success_status_wins_over_error_words() -> None:
    # A Codex ``completed`` status (from codex.txt) is the harness's own outcome.
    trajectory = {
        "agent": {"name": "codex"},
        "steps": [
            {
                "source": "agent",
                "tool_calls": [{"tool_call_id": "c1", "function_name": "ask_question", "arguments": {}}],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "c1",
                            "content": CODEX_STATUS + "Permission denied: see the access guide",
                            "extra": {"harness_status": "completed"},
                        }
                    ]
                },
            }
        ],
    }
    signals: dict[str, Any] | None = compute_plugin_signals(
        trajectory, {}, declared=DECLARED, mcp_call_servers={"c1": "deepwiki"}
    )
    assert signals is not None
    assert signals["mcp_calls"]["succeeded"] == 1
