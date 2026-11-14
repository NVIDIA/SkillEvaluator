# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RAGAS goal_accuracy requests and replies are saved as judge evidence, like the prompt judges' attempts.

The RAGAS modules are fakes, but their requests go through the real OpenAI SDK
and httpx client; a stubbed httpx transport answers them. The stub fails the
test if the SDK's async client is not an httpx client, and DNS lookups and
connections beyond loopback are refused, so a request the stub misses fails
instead of reaching the provider. Proxy settings are cleared, so a developer's
proxy changes neither result.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from tests.conftest import load_harbor_eval_template

from skillevaluator.tier3.harbor import runner

_OPENAI_TEST_KEY = "test-key-fake"
_CHAT_URL = "https://api.openai.com/v1/chat/completions"
# Both spellings are deleted. NO_PROXY is listed so that a lower-case no_proxy,
# which urllib prefers, cannot override the NO_PROXY=* that no_network sets.
_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
# Proxies a developer's shell may set. The SOCKS one fails to build without socksio.
_PROXY_SETTINGS = [
    pytest.param("HTTPS_PROXY", "http://proxy.invalid:3128", id="https-proxy"),
    pytest.param("ALL_PROXY", "socks5://proxy.invalid:1080", id="socks-proxy"),
]
_JUDGE_MESSAGES = [{"role": "user", "content": "judge"}]
_ACCURACY_REPLY = json.dumps(
    {
        "criteria": {
            "ACTION_CORRECT": True,
            "FACTUALLY_ACCURATE": True,
            "TASK_ADDRESSED": True,
            "ACTIONABLE": True,
            "FORMAT_FOLLOWED": True,
        },
        "reason": "judged from evidence",
    }
)


def _host_text(host: Any) -> Any:
    # anyio passes the IDNA-encoded host name as bytes.
    return host.decode("ascii", "replace") if isinstance(host, bytes) else host


def _is_loopback(host: Any) -> bool:
    if _host_text(host) == "localhost":
        return True
    try:
        return ipaddress.ip_address(_host_text(host)).is_loopback
    except (TypeError, ValueError):
        return False


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Refuse DNS lookups and connections beyond loopback, and fail the test if one was tried.

    The provider stub answers httpx requests only. A request it misses must fail
    here, not reach the real provider with the fake key. The verifier's client
    reads proxy settings like any httpx client, so they are cleared and
    NO_PROXY=* makes httpx ignore the system proxy too: a developer's proxy must
    not change which host the guard sees, and a SOCKS proxy must not make the
    client fail to build.
    """
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    blocked: list[Any] = []
    getaddrinfo = socket.getaddrinfo
    connect = socket.socket.connect

    def local_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_loopback(host):
            blocked.append(_host_text(host))
            raise OSError(f"network access is blocked in this test: {_host_text(host)!r}")
        return getaddrinfo(host, *args, **kwargs)

    def local_connect(sock: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple) and not _is_loopback(address[0]):
            blocked.append(_host_text(address[0]))
            raise OSError(f"network access is blocked in this test: {_host_text(address[0])!r}")
        return connect(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", local_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", local_connect)
    yield blocked
    assert not blocked, f"the test tried to reach the network: {blocked}"


@pytest.fixture
def verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the verifier with a canonical OpenAI non-reasoning judge, which routes goal_accuracy to RAGAS."""
    module = load_harbor_eval_template(f"harbor_eval_ragas_evidence_{tmp_path.name}")
    agent_dir = tmp_path / "logs" / "agent"
    verifier_dir = tmp_path / "logs" / "verifier"
    tests_dir = tmp_path / "tests"
    for directory in (agent_dir, verifier_dir, tests_dir):
        directory.mkdir(parents=True)
    module.VERIFIER_DIR = verifier_dir
    module.ATIF_PATH = agent_dir / "trajectory.json"
    module.ENTRY_PATH = tests_dir / "entry.json"
    module.REWARD_JSON = verifier_dir / "reward.json"
    module.REWARD_TXT = verifier_dir / "reward.txt"
    module.SKILL_EVALUATOR_REWARD_JSON = verifier_dir / "skill_evaluator_reward.json"

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", _OPENAI_TEST_KEY)
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-4.1")
    for name in (
        "SKILL_EVAL_JUDGE_MODEL",
        "SKILL_EVAL_LLM_MODEL",
        "LLM_JUDGE_FALLBACK_MODELS",
        "LLM_JUDGE_FALLBACK_POLICY",
        "LLM_JUDGE_RETRY_POLICY",
        "OPENAI_BASE_URL",
        "SKILL_EVAL_LLM_BASE_URL",
        # Character facts select the prompt goal judge instead of RAGAS.
        "SKILL_EVAL_JUDGE_CHARACTER_FACTS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert module._ragas_goal_accuracy_enabled()
    return module


def _install_ragas(
    monkeypatch: pytest.MonkeyPatch, *, messages: list[dict[str, Any]] | None = None, score: float = 1.0
) -> None:
    """Install the RAGAS 0.4 modules the verifier imports; the metric sends one request through the OpenAI SDK."""

    class Message:
        def __init__(self, *, content: str):
            self.content = content

    class Metric:
        def __init__(self, *, llm):
            self.llm = llm

        async def ascore(self, **_kwargs):
            await self.llm.client.chat.completions.create(model=self.llm.model, messages=messages or _JUDGE_MESSAGES)
            return SimpleNamespace(value=score)

    modules = {
        name: ModuleType(name)
        for name in (
            "ragas",
            "ragas.llms",
            "ragas.llms.base",
            "ragas.messages",
            "ragas.metrics",
            "ragas.metrics.collections",
        )
    }
    modules["ragas.llms.base"].llm_factory = lambda model, *, client: SimpleNamespace(model=model, client=client)
    modules["ragas.messages"].AIMessage = Message
    modules["ragas.messages"].HumanMessage = Message
    modules["ragas.metrics.collections"].AgentGoalAccuracyWithReference = Metric
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _chat_reply(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }


class _CutOffBody(httpx.AsyncByteStream):
    """A reply body that stops partway, as when the connection drops mid-reply."""

    async def __aiter__(self):
        yield b'{"choices":['
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


def _sdk_transport_class() -> type[httpx.AsyncBaseTransport]:
    """Return the transport class behind the OpenAI SDK's default async client, or fail the test.

    The stub answers httpx requests only. If the SDK builds its client on another
    HTTP stack, patching httpx would miss the verifier's requests, so fail before
    any is sent.
    """
    client_class = openai.DefaultAsyncHttpxClient
    if not (isinstance(client_class, type) and issubclass(client_class, httpx.AsyncClient)):
        pytest.fail(
            f"openai.DefaultAsyncHttpxClient is {client_class!r}, not an httpx.AsyncClient; "
            "update the provider stub for this SDK's HTTP stack"
        )
    # trust_env=False: a proxy setting must not change or break the class this finds.
    return type(client_class(trust_env=False)._transport)


def _stub_provider(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> list[httpx.Request]:
    """Answer the SDK's async httpx requests in order: a response, an error status, "cut-off", or a 200 JSON body."""
    transport_class = _sdk_transport_class()
    received: list[httpx.Request] = []

    async def handle_async_request(_transport, request: httpx.Request) -> httpx.Response:
        received.append(request)
        reply = replies[min(len(received) - 1, len(replies) - 1)]
        if isinstance(reply, httpx.Response):
            return reply
        if reply == "cut-off":
            return httpx.Response(200, headers={"content-type": "application/json"}, stream=_CutOffBody())
        if isinstance(reply, int):
            # The provider echoes the key, so the saved problem must be redacted.
            error = {"error": {"message": f"temporary outage for {_OPENAI_TEST_KEY}", "type": "server_error"}}
            return httpx.Response(reply, headers={"retry-after-ms": "1"}, json=error)
        return httpx.Response(200, json=reply)

    monkeypatch.setattr(transport_class, "handle_async_request", handle_async_request)
    return received


@contextmanager
def _judge_evidence(verifier: ModuleType) -> Iterator[None]:
    """Collect judge evidence the way main() does while it runs the judges."""
    token = verifier._ACTIVE_JUDGE_EVIDENCE.set([])
    try:
        yield
    finally:
        verifier._ACTIVE_JUDGE_EVIDENCE.reset(token)


def _attempts(verifier: ModuleType) -> list[dict[str, Any]]:
    payload = json.loads((verifier.VERIFIER_DIR / "judge_attempts.json").read_text(encoding="utf-8"))
    assert payload["judge_contract"] == verifier.JUDGE_CONTRACT
    return payload["attempts"]


def test_ragas_request_and_reply_are_saved_redacted(verifier: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    leaked = "nvapi-" + "s" * 40
    _install_ragas(
        monkeypatch,
        messages=[
            {"role": "system", "content": "Compare the end state with the reference."},
            {"role": "user", "content": f"The agent printed {leaked}"},
        ],
    )
    received = _stub_provider(monkeypatch, [_chat_reply(f"achieved; the agent used {_OPENAI_TEST_KEY}")])

    with _judge_evidence(verifier):
        result = verifier._call_required_judge(
            "goal_accuracy", verifier.judge_goal_accuracy, "question", "expected", "agent answer"
        )

    assert (result["score"], result["method"]) == (1.0, "ragas")
    assert [str(request.url) for request in received] == [_CHAT_URL]
    exchange, outcome = _attempts(verifier)
    assert (exchange["metric"], exchange["attempt"], exchange["method"], exchange["status"]) == (
        "goal_accuracy",
        1,
        "ragas",
        "replied",
    )
    assert exchange["model"] == "gpt-4.1"
    assert exchange["problem"] is None
    assert exchange["prompt"].startswith(
        "[system]\nCompare the end state with the reference.\n\n[user]\nThe agent printed "
    )
    assert len(exchange["prompt_sha256"]) == 64
    assert exchange["reply"].startswith("achieved; the agent used ")
    assert (outcome["method"], outcome["status"], outcome["score"]) == ("ragas", "accepted", 1.0)
    assert "prompt" not in outcome
    assert "reply" not in outcome
    saved = (verifier.VERIFIER_DIR / "judge_attempts.json").read_text(encoding="utf-8")
    assert leaked not in saved
    assert _OPENAI_TEST_KEY not in saved


def test_ragas_reply_behind_a_rejected_score_is_saved(verifier: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ragas(monkeypatch, score=0.5)
    received = _stub_provider(monkeypatch, [_chat_reply("ok")])
    custom = {"score": 0.0, "reason": "custom judgment", "method": "custom"}
    monkeypatch.setattr(verifier, "_judge_goal_accuracy_custom", lambda *_args, **_kwargs: custom)

    with _judge_evidence(verifier):
        assert verifier.judge_goal_accuracy("question", "expected", "agent answer") == custom

    assert len(received) == 1
    exchange, failure = _attempts(verifier)
    assert (exchange["status"], exchange["prompt"], exchange["reply"]) == ("replied", "[user]\njudge", "ok")
    assert (failure["method"], failure["status"], failure["model"]) == ("ragas", "error", "gpt-4.1")
    assert "non-binary" in failure["problem"]


@pytest.mark.parametrize(
    ("first_reply", "problem"),
    [(503, "HTTP 503: "), ("cut-off", "the reply was not received (RemoteProtocolError)")],
    ids=["http-error", "cut-off"],
)
def test_failed_ragas_provider_replies_are_saved_with_their_prompt(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch, first_reply: Any, problem: str
) -> None:
    _install_ragas(monkeypatch)
    received = _stub_provider(monkeypatch, [first_reply, _chat_reply("ok")])

    with _judge_evidence(verifier):
        result = verifier.judge_goal_accuracy("question", "expected", "agent answer")

    assert (result["score"], result["method"]) == (1.0, "ragas")
    assert len(received) == 2
    failed, exchange, outcome = _attempts(verifier)
    assert (failed["method"], failed["status"], failed["prompt"]) == ("ragas", "error", "[user]\njudge")
    assert failed["problem"].startswith(problem)
    assert _OPENAI_TEST_KEY not in failed["problem"]
    assert "reply" not in failed
    assert [exchange["status"], outcome["status"]] == ["replied", "accepted"]
    # --max-judge-calls counts the RAGAS score once, not each request or the SDK's retry.
    assert runner._budget_usage(verifier.VERIFIER_DIR) == {"judge_calls": 1}


def test_long_error_body_is_redacted_before_it_is_shortened(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key that straddles the 2000th character is masked whole, not cut in two and saved in part."""
    cut = 2000
    body = "x" * (cut - len(_OPENAI_TEST_KEY) + 1) + _OPENAI_TEST_KEY + "x" * 100
    assert body[:cut].endswith(_OPENAI_TEST_KEY[:-1])
    _install_ragas(monkeypatch)
    received = _stub_provider(
        monkeypatch, [httpx.Response(503, headers={"retry-after-ms": "1"}, text=body), _chat_reply("ok")]
    )

    with _judge_evidence(verifier):
        result = verifier.judge_goal_accuracy("question", "expected", "agent answer")

    assert (result["score"], result["method"]) == (1.0, "ragas")
    assert len(received) == 2
    failed, exchange, outcome = _attempts(verifier)
    assert failed["status"] == "error"
    assert failed["problem"].startswith("HTTP 503: xxx")
    # The shortened problem keeps the end of the body, where the key was.
    assert failed["problem"].endswith("[REDACTED]" + "x" * 100)
    saved = (verifier.VERIFIER_DIR / "judge_attempts.json").read_text(encoding="utf-8")
    assert _OPENAI_TEST_KEY[:-1] not in saved
    assert [exchange["status"], outcome["status"]] == ["replied", "accepted"]


def test_ragas_exchange_with_an_unreadable_request_names_the_selected_judge_model(verifier: ModuleType) -> None:
    """A request body that is not a chat request names no model, so the entry names the selected judge model."""
    request = httpx.Request("POST", _CHAT_URL, content=b"not a chat request")
    response = httpx.Response(200, json=_chat_reply("ok"), request=request)

    with _judge_evidence(verifier):
        verifier._record_ragas_exchange(response)

    (exchange,) = _attempts(verifier)
    assert (exchange["status"], exchange["prompt"], exchange["reply"]) == ("replied", "not a chat request", "ok")
    assert exchange["model"] == "gpt-4.1"


def test_ragas_evidence_reads_every_request_message_and_tool_call_replies(verifier: ModuleType) -> None:
    request = {
        "model": "judge-model",
        "messages": [
            {"role": "system", "content": "rubric"},
            {"role": "user", "content": [{"type": "text", "text": "conversation"}]},
        ],
    }
    tool_reply = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"type": "function", "function": {"name": "Verdict", "arguments": '{"verdict": 1}'}}
                    ],
                }
            }
        ]
    }

    assert verifier._ragas_request_text(json.dumps(request).encode()) == (
        "judge-model",
        '[system]\nrubric\n\n[user]\n[{"type": "text", "text": "conversation"}]',
    )
    assert verifier._ragas_request_text(b"not a chat request") == (None, "not a chat request")
    assert verifier._ragas_reply_text(json.dumps(tool_reply).encode()) == '{"verdict": 1}'
    assert verifier._ragas_reply_text(json.dumps(_chat_reply("ok")).encode()) == "ok"
    assert verifier._ragas_reply_text(b"not a chat completion") == "not a chat completion"


def test_ragas_evidence_masks_credentials_at_json_escaped_line_starts(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RAGAS writes the agent's messages into its prompt as JSON, and a JSON-mode reply quotes them as JSON too.

    Each line break in the agent's text is therefore saved escaped, as the two
    characters backslash and n, before the next line's header or flag.
    """
    agent_text = (
        "HTTP/1.1 200 OK\nAuthorization: Bearer bbbbbbbbbbbb\nCookie: session=cccccccccccc\n"
        "--token tttttttttttt\n-u admin:pppppppppppp"
    )
    prompt = "Judge whether the agent reached the goal.\ninput: " + json.dumps(
        {"user_input": [{"content": agent_text, "type": "ai"}]}, indent=4
    )
    _install_ragas(monkeypatch, messages=[{"role": "user", "content": prompt}])
    received = _stub_provider(monkeypatch, [_chat_reply(json.dumps({"end_state": agent_text, "verdict": 1}))])

    with _judge_evidence(verifier):
        result = verifier.judge_goal_accuracy("question", "expected", "agent answer")

    assert (result["score"], result["method"]) == (1.0, "ragas")
    assert len(received) == 1
    exchange, outcome = _attempts(verifier)
    assert [exchange["status"], outcome["status"]] == ["replied", "accepted"]
    for field in ("prompt", "reply"):
        assert (
            "\\nAuthorization: Bearer <redacted>\\nCookie: <redacted>\\n--token <redacted>\\n-u <redacted>"
            in exchange[field]
        )
    saved = (verifier.VERIFIER_DIR / "judge_attempts.json").read_text(encoding="utf-8")
    for secret in ("bbbbbbbbbbbb", "cccccccccccc", "tttttttttttt", "pppppppppppp"):
        assert secret not in saved


def test_ragas_evidence_masks_consecutive_credential_lines_in_tool_calls(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool call's arguments reach the RAGAS prompt and reply as JSON in JSON, so their line breaks are escaped twice.

    A flag value that ends one line must not take the next line's flag or
    header name with it and leave that line's value in clear.
    """
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "function_name": "Write",
                        "arguments": {
                            "file_path": "deploy.flags",
                            "content": "--api-key aaaaaaaaaaaa\n--token bbbbbbbbbbbb\n",
                        },
                    },
                    {
                        "function_name": "Bash",
                        "arguments": {"command": "docker login -u bot -p cccccccccccc\nCookie: session=dddddddddddd"},
                    },
                ],
            },
        ]
    }
    secrets = ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc", "dddddddddddd")
    agent_text = verifier.build_conversation_summary(trajectory, "Set up the deploy flags.")
    assert all(secret in agent_text for secret in secrets)
    prompt = "Judge whether the agent reached the goal.\ninput: " + json.dumps(
        {"user_input": [{"content": agent_text, "type": "ai"}]}, indent=4
    )
    _install_ragas(monkeypatch, messages=[{"role": "user", "content": prompt}])
    received = _stub_provider(monkeypatch, [_chat_reply(json.dumps({"end_state": agent_text, "verdict": 1}))])

    with _judge_evidence(verifier):
        result = verifier.judge_goal_accuracy("question", "expected", "agent answer")

    assert (result["score"], result["method"]) == (1.0, "ragas")
    assert len(received) == 1
    exchange, outcome = _attempts(verifier)
    assert [exchange["status"], outcome["status"]] == ["replied", "accepted"]
    for field in ("prompt", "reply"):
        assert "deploy.flags" in exchange[field]
        assert "docker login -u bot -p <redacted>" in exchange[field]
    saved = (verifier.VERIFIER_DIR / "judge_attempts.json").read_text(encoding="utf-8")
    for secret in secrets:
        assert secret not in saved


def test_ragas_evidence_masks_private_key_blocks_in_json_prompts_and_replies(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private key block in the agent's text spans "\\n" escapes in the RAGAS prompt and reply, and is masked whole."""
    label = "RSA PRIVATE " + "KEY"
    body = ("MIIEbodyLineOne", "MIIEbodyLineTwo")
    agent_text = "\n".join(("Wrote the key:", f"-----BEGIN {label}-----", *body, f"-----END {label}-----", "done"))
    prompt = "Judge whether the agent reached the goal.\ninput: " + json.dumps(
        {"user_input": [{"content": agent_text, "type": "ai"}]}, indent=4
    )
    _install_ragas(monkeypatch, messages=[{"role": "user", "content": prompt}])
    received = _stub_provider(monkeypatch, [_chat_reply(json.dumps({"end_state": agent_text, "verdict": 1}))])

    with _judge_evidence(verifier):
        result = verifier.judge_goal_accuracy("question", "expected", "agent answer")

    assert (result["score"], result["method"]) == (1.0, "ragas")
    assert len(received) == 1
    exchange, outcome = _attempts(verifier)
    assert [exchange["status"], outcome["status"]] == ["replied", "accepted"]
    for field in ("prompt", "reply"):
        assert "Wrote the key:\\n<redacted private key>\\ndone" in exchange[field]
    saved = (verifier.VERIFIER_DIR / "judge_attempts.json").read_text(encoding="utf-8")
    assert "-----END" not in saved
    for line in body:
        assert line not in saved


@pytest.mark.parametrize(
    ("message", "finish_reason", "kept"),
    [
        pytest.param(
            {"role": "assistant", "content": None, "refusal": "I can't help with that request."},
            "stop",
            "I can't help with that request.",
            id="refusal",
        ),
        pytest.param(
            {"role": "assistant", "content": None},
            "content_filter",
            '"finish_reason": "content_filter"',
            id="content-filter",
        ),
        pytest.param({"role": "assistant", "content": ""}, "length", '"finish_reason": "length"', id="empty-text"),
    ],
)
def test_ragas_reply_without_text_or_tool_calls_is_saved_as_the_reply_body(
    verifier: ModuleType, message: dict[str, Any], finish_reason: str, kept: str
) -> None:
    """A refusal or a filtered reply is what makes RAGAS fail, so the entry keeps why there was no answer."""
    reply = _chat_reply("")
    reply["choices"][0].update(message=message, finish_reason=finish_reason)
    request = httpx.Request("POST", _CHAT_URL, json={"model": "gpt-4.1", "messages": _JUDGE_MESSAGES})

    with _judge_evidence(verifier):
        verifier._record_ragas_exchange(httpx.Response(200, content=json.dumps(reply).encode(), request=request))

    (exchange,) = _attempts(verifier)
    assert exchange["status"] == "replied"
    assert kept in exchange["reply"]
    assert exchange["reply_chars"] == len(exchange["reply"])


@pytest.mark.parametrize("broken", ["request-parser", "http-client"])
def test_ragas_evidence_failure_never_changes_the_score(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, broken: str
) -> None:
    _install_ragas(monkeypatch)
    received = _stub_provider(monkeypatch, [_chat_reply("ok")])

    def fail(*_args: Any, **_kwargs: Any):
        raise RuntimeError(f"evidence code failed for {_OPENAI_TEST_KEY}")

    if broken == "request-parser":
        monkeypatch.setattr(verifier, "_ragas_request_text", fail)
    else:
        monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", fail)

    with caplog.at_level(logging.WARNING, logger=verifier.logger.name), _judge_evidence(verifier):
        result = verifier.judge_goal_accuracy("question", "expected", "agent answer")

    assert (result["score"], result["method"]) == (1.0, "ragas")
    assert len(received) == 1
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == ["Could not save RAGAS judge evidence: RuntimeError"]
    (outcome,) = _attempts(verifier)
    assert (outcome["method"], outcome["status"], outcome["score"]) == ("ragas", "accepted", 1.0)


def test_ragas_evidence_write_failure_is_logged_like_the_other_judges(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """_record_judge_attempt logs a failed judge_attempts.json write itself, so the RAGAS warning is not logged."""
    _install_ragas(monkeypatch)
    received = _stub_provider(monkeypatch, [_chat_reply("ok")])
    write_text = Path.write_text

    def refuse_evidence_write(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name == "judge_attempts.json.tmp":
            raise PermissionError("read-only verifier directory")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", refuse_evidence_write)

    with caplog.at_level(logging.WARNING, logger=verifier.logger.name), _judge_evidence(verifier):
        result = verifier.judge_goal_accuracy("question", "expected", "agent answer")
        kept = list(verifier._ACTIVE_JUDGE_EVIDENCE.get())

    assert (result["score"], result["method"]) == (1.0, "ragas")
    assert len(received) == 1
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == ["Could not save judge evidence: PermissionError"] * 2
    assert [(attempt["method"], attempt["status"]) for attempt in kept] == [("ragas", "replied"), ("ragas", "accepted")]
    assert not (verifier.VERIFIER_DIR / "judge_attempts.json").exists()


@pytest.mark.parametrize("helper", ["exchange", "http-client"])
def test_ragas_evidence_helpers_reraise_the_judge_deadline(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, helper: str
) -> None:
    """Pins the helpers only: they re-raise the deadline alarm instead of logging it as an evidence failure.

    In a real request the alarm leaves the httpx hook into the OpenAI SDK, which
    treats it as a connection error and retries; the RAGAS time budget
    (asyncio.wait_for) still bounds the run. An alarm while the HTTP client is
    built must leave _judge_goal_accuracy_ragas, not be logged while RAGAS runs
    on past the deadline.
    """

    def deadline(*_args: Any, **_kwargs: Any):
        raise TimeoutError("LLM judge time budget exhausted")

    with caplog.at_level(logging.WARNING, logger=verifier.logger.name), pytest.raises(TimeoutError):
        if helper == "exchange":
            monkeypatch.setattr(verifier, "_record_judge_attempt", deadline)
            request = httpx.Request("POST", _CHAT_URL, json={"model": "gpt-4.1", "messages": _JUDGE_MESSAGES})
            verifier._record_ragas_exchange(httpx.Response(200, json=_chat_reply("ok"), request=request))
        else:
            monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", deadline)
            verifier._ragas_evidence_http_client(openai)

    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_verifier_run_saves_ragas_exchanges_with_the_other_judges(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier.ATIF_PATH.write_text(
        json.dumps({"steps": [{"source": "agent", "message": "I completed the task."}]}), encoding="utf-8"
    )
    verifier.ENTRY_PATH.write_text(
        json.dumps({"id": "case-1", "has_skill": False, "question": "Complete the task", "ground_truth": "Done"}),
        encoding="utf-8",
    )
    _install_ragas(monkeypatch)
    received = _stub_provider(monkeypatch, [_chat_reply("ok")])
    verifier.call_public_llm = lambda _prompt, **_kwargs: (_ACCURACY_REPLY, None)
    verifier._call_public_llm_with_provenance = lambda _prompt, **_kwargs: (
        _ACCURACY_REPLY,
        None,
        {"provider": "openai", "model": "gpt-4.1"},
    )

    verifier.main()

    assert len(received) == 1
    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["goal_accuracy"] == 1.0
    assert rich["details"]["goal_accuracy"]["method"] == "ragas"
    assert [(attempt["metric"], attempt["method"], attempt["status"]) for attempt in _attempts(verifier)] == [
        ("accuracy", "llm", "accepted"),
        ("goal_accuracy", "ragas", "replied"),
        ("goal_accuracy", "ragas", "accepted"),
    ]
    # The RAGAS request entry does not count toward --max-judge-calls, as on main.
    assert runner._budget_usage(verifier.VERIFIER_DIR.parent) == {"judge_calls": 2}


@pytest.mark.parametrize(("variable", "proxy"), _PROXY_SETTINGS)
def test_proxy_settings_leave_the_provider_stub_answering(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch, variable: str, proxy: str
) -> None:
    """A proxy setting must not break the stub or route the verifier's request around it."""
    # no_network's NO_PROXY=* stays set, as it does when the proxy comes from the developer's shell.
    monkeypatch.setenv(variable, proxy)
    _install_ragas(monkeypatch)
    received = _stub_provider(monkeypatch, [_chat_reply("ok")])

    with _judge_evidence(verifier):
        result = verifier.judge_goal_accuracy("question", "expected", "agent answer")

    assert (result["score"], result["method"]) == (1.0, "ragas")
    assert [str(request.url) for request in received] == [_CHAT_URL]
    assert [(attempt["method"], attempt["status"]) for attempt in _attempts(verifier)] == [
        ("ragas", "replied"),
        ("ragas", "accepted"),
    ]


def test_provider_stub_fails_before_any_request_when_the_sdk_client_is_not_httpx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OtherStackClient:
        """Stands in for an SDK whose default async client is built on another HTTP library."""

    monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", OtherStackClient)

    with pytest.raises(pytest.fail.Exception, match=r"not an httpx\.AsyncClient"):
        _stub_provider(monkeypatch, [_chat_reply("ok")])


@pytest.mark.parametrize(("variable", "proxy"), [pytest.param(None, None, id="no-proxy"), *_PROXY_SETTINGS])
def test_requests_the_stub_misses_never_reach_the_network(
    no_network: list[Any], monkeypatch: pytest.MonkeyPatch, variable: str | None, proxy: str | None
) -> None:
    if variable:
        # Undo no_network's NO_PROXY=* and set a proxy: only trust_env=False keeps it out of this client.
        monkeypatch.delenv("NO_PROXY")
        monkeypatch.setenv(variable, proxy)

    async def send_unstubbed_request() -> None:
        # trust_env=False keeps any proxy out of the way, so the guard sees the provider host.
        async with openai.DefaultAsyncHttpxClient(trust_env=False) as client:
            await client.post(_CHAT_URL, json={"model": "gpt-4.1", "messages": _JUDGE_MESSAGES})

    with pytest.raises(httpx.ConnectError, match="network access is blocked"):
        asyncio.run(send_unstubbed_request())
    with pytest.raises(OSError, match="network access is blocked"):
        socket.getaddrinfo("192.0.2.1", 443)
    with socket.socket() as sock, pytest.raises(OSError, match="network access is blocked"):
        sock.connect(("192.0.2.1", 443))
    assert socket.getaddrinfo("127.0.0.1", 443)

    # Clear the record so the guard's own teardown check passes.
    blocked = list(no_network)
    no_network.clear()
    assert blocked == ["api.openai.com", "192.0.2.1", "192.0.2.1"]
