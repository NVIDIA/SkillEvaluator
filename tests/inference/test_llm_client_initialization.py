# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent first calls must share one SDK client without serializing requests."""

from __future__ import annotations

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from threading import Barrier, BrokenBarrierError
from unittest.mock import patch

import httpx
import pytest

from skillevaluator.inference import LLMClient

_PROVIDERS = [
    ("openai", "OpenAI", "OPENAI_API_KEY", "gpt-4.1"),
    ("anthropic", "Anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-6"),
]


@pytest.fixture(params=_PROVIDERS, ids=lambda value: value[0])
def provider(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    name, sdk_name, key_env, model = request.param
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", name)
    monkeypatch.setenv(key_env, "test-key")
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", model)
    for setting in ("SKILL_EVAL_LLM_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(setting, raising=False)
    module = importlib.import_module(name)
    return module, sdk_name, model


def test_concurrent_first_calls_construct_one_sdk_client(provider) -> None:
    module, sdk_name, _model = provider
    sdk_class = getattr(module, sdk_name)
    start = Barrier(5)
    constructors = Barrier(5)
    created = []
    client = LLMClient()

    def create(**kwargs):
        instance = sdk_class(**kwargs)
        created.append(instance)
        # Keep overlapping initializations in flight. With one initializer,
        # the rendezvous expires and releases it; this is not a speed assertion.
        with suppress(BrokenBarrierError):
            constructors.wait(timeout=0.25)
        return instance

    def get(_index):
        start.wait(timeout=10)
        return client._get_client()

    try:
        with patch.object(module, sdk_name, side_effect=create), ThreadPoolExecutor(max_workers=5) as pool:
            returned = list(pool.map(get, range(5)))
        assert len(created) == 1
        assert all(instance is created[0] for instance in returned)
        assert client._get_client() is created[0]
    finally:
        for instance in created:
            instance.close()


def test_failed_initialization_can_be_retried(provider) -> None:
    # Existing retry behavior must survive the initialization synchronization.
    module, sdk_name, _model = provider
    sdk_class = getattr(module, sdk_name)
    client = LLMClient()
    instance = sdk_class(api_key="test-key", max_retries=0)
    try:
        with patch.object(module, sdk_name, side_effect=[RuntimeError("constructor failed"), instance]) as create:
            with pytest.raises(RuntimeError, match="constructor failed"):
                client._get_client()
            assert client._client is None
            assert client._get_client() is instance
            assert client._get_client() is instance
        assert create.call_count == 2
    finally:
        instance.close()


def test_shared_sdk_client_keeps_completion_requests_concurrent(provider) -> None:
    module, sdk_name, model = provider
    requests = Barrier(2)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = "done-" + payload["messages"][-1]["content"]
        requests.wait(timeout=10)
        if module.__name__ == "openai":
            body = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
        else:
            body = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = LLMClient(http_client=transport, max_retries=0)

        def complete(prompt):
            response = client.completions("system", prompt)
            return response, client.model_used

        sdk_class = getattr(module, sdk_name)
        with patch.object(module, sdk_name, wraps=sdk_class) as create, ThreadPoolExecutor(max_workers=2) as pool:
            returned = list(pool.map(complete, ["first", "second"]))
        assert returned == [("done-first", model), ("done-second", model)]
        assert create.call_count == 1
