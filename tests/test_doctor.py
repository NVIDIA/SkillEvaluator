# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 scanner and Tier 2 provider readiness behind ``doctor --tier``."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from skillevaluator import doctor as readiness
from skillevaluator.utils import tool_runner
from skillevaluator.utils.tool_runner import ExternalTool, Tools

_SCANNERS = ("skillspector", "gitleaks", "bandit", "semgrep", "pip_audit")
_VERSION_OUTPUT = {
    "skillspector": "SkillSpector v2.11.2\n",
    "gitleaks": "8.21.2\n",
    "bandit": "bandit 1.9.4\n  python version = 3.12.13\n",
    "semgrep": "1.166.0\n",
    "pip-audit": "pip-audit 2.10.1\n",
}
_PROVIDER_ENV = (
    "SKILL_EVAL_LLM_PROVIDER",
    "SKILL_EVAL_LLM_MODEL",
    "SKILL_EVAL_LLM_API_KEY",
    "SKILL_EVAL_LLM_BASE_URL",
    "SKILL_EVAL_EMBEDDING_PROVIDER",
    "SKILL_EVAL_EMBEDDING_MODEL",
    "SKILL_EVAL_EMBEDDING_API_KEY",
    "SKILL_EVAL_EMBEDDING_BASE_URL",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
)


def _rows(**options) -> dict[str, tuple[str, str]]:
    return {name: (status, detail) for name, status, detail in readiness.readiness_rows(**options)}


@pytest.fixture
def scanners(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Resolve every scanner to a fake path and answer its version probe in-process."""
    for name in _SCANNERS:
        tool = getattr(Tools, name)
        monkeypatch.setattr(tool, "_path", f"/doctor-tools/{tool.command}")
        monkeypatch.setattr(tool, "_configuration_error", None)
        monkeypatch.setattr(tool, "resolution_source", "automatic")
    output = dict(_VERSION_OUTPUT)
    calls: list[tuple[list[str], dict]] = []

    def run(command, **kwargs):
        calls.append((list(command), kwargs))
        name = command[0].rsplit("/", 1)[-1]
        return subprocess.CompletedProcess(command, 0, stdout=output[name], stderr="")

    monkeypatch.setattr(tool_runner.subprocess, "run", run)
    # Tier 1 readiness is static: it passes with no provider configured.
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(calls=calls, output=output)


@pytest.mark.parametrize(
    ("checks", "commands"),
    [
        (None, ["skillspector", "bandit", "semgrep", "gitleaks"]),
        ("security", ["skillspector"]),
        ("code", ["bandit", "semgrep", "gitleaks"]),
        ("deps,security", ["skillspector", "pip-audit"]),
    ],
)
def test_tier1_probes_each_scanner_the_selected_checks_run(scanners, checks, commands) -> None:
    assert readiness.doctor(tier="1", checks=checks, timeout=4) == 0

    assert [call[0] for call in scanners.calls] == [
        [f"/doctor-tools/{name}", "version" if name == "gitleaks" else "--version"] for name in commands
    ]
    assert all(call[1]["timeout"] == 4 for call in scanners.calls)


def test_tier1_reports_each_scanner_version_and_path(scanners) -> None:
    rows = _rows(tier="1", checks="security,code-integrity,dependency")

    assert rows == {
        "CLI package": rows["CLI package"],
        "SkillSpector": ("pass", "2.11.2 at /doctor-tools/skillspector"),
        "Bandit": ("pass", "1.9.4 at /doctor-tools/bandit"),
        "Semgrep": ("pass", "1.166.0 at /doctor-tools/semgrep"),
        "Gitleaks": ("pass", "8.21.2 at /doctor-tools/gitleaks"),
        "pip-audit": ("pass", "2.10.1 at /doctor-tools/pip-audit"),
    }


def test_tier1_checks_without_scanners_report_nothing_to_probe(scanners) -> None:
    rows = _rows(tier="1", checks="schema,pii,quality")

    assert rows["Tier 1 scanners"][0] == "pass"
    assert scanners.calls == []


def test_tier1_version_probes_keep_credentials_out_of_skillspector(scanners, monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-canary")
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")

    assert readiness.doctor(tier="1", checks="security,code") == 0

    environments = {call[0][0].rsplit("/", 1)[-1]: call[1]["env"] for call in scanners.calls}
    assert "secret-canary" not in environments["skillspector"].values()
    assert "NVIDIA_API_KEY" not in environments["skillspector"]
    # The semgrep probe skips Semgrep's network update check, as the scan does.
    assert environments["semgrep"]["SEMGREP_ENABLE_VERSION_CHECK"] == "0"
    assert environments["bandit"] is None


def test_tier1_reports_an_explicit_executable_override(scanners, monkeypatch, tmp_path) -> None:
    executable = tmp_path / "bandit"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("SKILLEVALUATOR_BANDIT_PATH", str(executable))
    monkeypatch.setattr(Tools, "bandit", ExternalTool("Bandit", "bandit", override_env="SKILLEVALUATOR_BANDIT_PATH"))

    rows = _rows(tier="1", checks="code")

    assert rows["Bandit"] == ("pass", f"1.9.4 at {executable.resolve()} (from SKILLEVALUATOR_BANDIT_PATH)")


@pytest.mark.parametrize("failure", ["missing", "malformed override", "timeout", "exit status"])
def test_tier1_scanner_failure_does_not_hide_other_scanners(scanners, monkeypatch, failure) -> None:
    if failure == "missing":
        monkeypatch.setattr(Tools.bandit, "_path", None)
    elif failure == "malformed override":
        monkeypatch.setenv("SKILLEVALUATOR_BANDIT_PATH", "relative/bandit")
        monkeypatch.setattr(
            Tools, "bandit", ExternalTool("Bandit", "bandit", override_env="SKILLEVALUATOR_BANDIT_PATH")
        )
    else:
        answer = tool_runner.subprocess.run

        def failing_bandit(command, **kwargs):
            if command[0].endswith("/bandit"):
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="secret-canary")
                return subprocess.CompletedProcess(command, 2, stdout="secret-canary", stderr="secret-canary")
            return answer(command, **kwargs)

        monkeypatch.setattr(tool_runner.subprocess, "run", failing_bandit)

    rows = _rows(tier="1", checks="code", timeout=4)

    status, detail = rows["Bandit"]
    assert status == "fail"
    assert "secret-canary" not in detail
    expected = {
        "missing": "not installed. Bundled with the security extra",
        "malformed override": "SKILLEVALUATOR_BANDIT_PATH must name an absolute executable file",
        "timeout": "Bandit timed out after 4 seconds",
        "exit status": "version probe exited with status 2",
    }[failure]
    assert expected in detail
    assert rows["Semgrep"][0] == rows["Gitleaks"][0] == "pass"
    assert readiness.doctor(tier="1", checks="code", timeout=4) == 1


@pytest.mark.parametrize(
    ("command", "output", "status"),
    [
        ("bandit", "bandit 1.9.3\n", "fail"),
        ("bandit", "bandit 1.9.4\n", "pass"),
        ("gitleaks", "v7.6.1\n", "fail"),
        ("gitleaks", "v8.0.0\n", "pass"),
    ],
)
def test_tier1_rejects_scanners_older_than_the_supported_release(scanners, command, output, status) -> None:
    scanners.output[command] = output

    rows = _rows(tier="1", checks="code")

    label = {"bandit": "Bandit", "gitleaks": "Gitleaks"}[command]
    assert rows[label][0] == status
    if status == "fail":
        assert "is older than the supported" in rows[label][1]


def test_tier1_unreported_version_warns_without_failing(scanners) -> None:
    scanners.output["semgrep"] = "development build\n"

    rows = _rows(tier="1", checks="code")

    assert rows["Semgrep"] == ("warn", "version not reported; /doctor-tools/semgrep")
    assert readiness.doctor(tier="1", checks="code") == 0


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_readiness_rejects_a_non_finite_or_non_positive_timeout(timeout) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        readiness.readiness_rows(tier="1", timeout=timeout)


def test_readiness_only_covers_tiers_1_and_2() -> None:
    with pytest.raises(ValueError, match="tier 1 or 2"):
        readiness.readiness_rows(tier="3")


@pytest.fixture
def provider_api(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Route the real provider SDKs through an in-process transport."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret-canary")
    requests: list[httpx.Request] = []
    responses: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint in responses:
            return responses[endpoint]
        if endpoint == "embeddings":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "m",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.6, 0.8]}],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "probe",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            },
        )

    class _TransportClient(httpx.Client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", _TransportClient)
    return SimpleNamespace(requests=requests, responses=responses)


def test_tier2_resolves_providers_offline_by_default(provider_api) -> None:
    rows = _rows(tier="2")

    assert rows["Embedding provider"] == ("pass", "nv_build / nvidia/nemotron-3-embed-1b")
    assert rows["LLM provider"][0] == "pass"
    assert rows["LLM provider"][1].startswith("nv_build / ")
    assert "Embedding model" not in rows
    assert provider_api.requests == []


def test_tier2_missing_provider_fails_without_network(provider_api, monkeypatch, capsys) -> None:
    monkeypatch.delenv("SKILL_EVAL_LLM_PROVIDER")
    monkeypatch.delenv("NVIDIA_API_KEY")

    assert readiness.doctor(tier="2", verify_models=True) == 1

    rows = _rows(tier="2", verify_models=True)
    assert rows["Embedding provider"][0] == rows["LLM provider"][0] == "fail"
    assert "No provider is configured" in rows["LLM provider"][1]
    assert "Embedding model" not in rows
    assert provider_api.requests == []


def test_tier2_without_the_extra_reports_it_missing(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "_module_available", lambda _module: False)

    rows = _rows(tier="2", verify_models=True)

    assert rows["Tier 2 extra"][0] == "fail"
    assert "tier2" in rows["Tier 2 extra"][1]


def test_tier2_live_probe_sends_one_bounded_request_per_model(provider_api) -> None:
    rows = _rows(tier="2", verify_models=True, timeout=4)

    assert rows["Embedding model"] == ("pass", "returned a 2-dimension vector")
    assert rows["Deduplication model"] == ("pass", "returned a chat completion")
    assert [request.url.path for request in provider_api.requests] == ["/v1/embeddings", "/v1/chat/completions"]
    assert all(request.extensions["timeout"]["read"] == 4 for request in provider_api.requests)
    chat = json.loads(provider_api.requests[1].content)
    assert chat["model"] == rows["LLM provider"][1].split(" / ", 1)[1]


@pytest.mark.parametrize("status", [401, 500])
def test_tier2_live_failures_are_reported_without_provider_bodies(provider_api, capsys, status) -> None:
    body = {"error": {"message": "secret-canary nvapi-secret-canary", "code": "secret-canary"}}
    provider_api.responses["embeddings"] = httpx.Response(status, json=body)
    provider_api.responses["completions"] = httpx.Response(status, json=body)

    assert readiness.doctor(tier="2", verify_models=True, timeout=4) == 1

    output = capsys.readouterr().out
    assert f"HTTP {status}" in output
    assert "secret-canary" not in output
    # The chat probe still runs after the embedding probe fails, and neither retries.
    assert [request.url.path for request in provider_api.requests] == ["/v1/embeddings", "/v1/chat/completions"]
