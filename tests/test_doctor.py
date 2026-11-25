# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 scanner and Tier 2 provider readiness behind ``doctor --tier``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner

from skillevaluator import doctor as readiness
from skillevaluator.cli import cli
from skillevaluator.constants import CONTENT_DEDUP_LLM_MAX_TOKENS
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


def test_tier1_reports_managed_provisioning_without_naming_an_override(scanners, monkeypatch, tmp_path) -> None:
    """No scanner is installed, so SkillSpector runs through uv and Gitleaks is the managed binary."""
    monkeypatch.setenv("SKILLEVALUATOR_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(
        tool_runner, "resolve_tool_path", lambda command, **_kwargs: "/env/bin/uv" if command == "uv" else None
    )
    monkeypatch.setattr(tool_runner, "gitleaks_platform", lambda: ("linux", "x64"))
    skillspector = ExternalTool(
        "skillspector", "skillspector", override_env="SKILLEVALUATOR_SKILLSPECTOR_PATH", managed=True
    )
    gitleaks = ExternalTool("Gitleaks", "gitleaks", managed=True)
    # The wheel and the binary are already in place, so the probes download nothing.
    monkeypatch.setattr(skillspector, "_ensure", None)
    monkeypatch.setattr(gitleaks, "_ensure", None)
    monkeypatch.setattr(Tools, "skillspector", skillspector)
    monkeypatch.setattr(Tools, "gitleaks", gitleaks)
    scanners.output["uv"] = "SkillSpector v2.13.7\n"

    rows = _rows(tier="1", checks="security,code")

    assert rows["SkillSpector"] == ("pass", "2.13.7 at /env/bin/uv (uv tool run)")
    assert rows["Gitleaks"] == ("pass", f"8.21.2 at {gitleaks.path} (managed)")
    assert scanners.calls[0][0][:3] == ["/env/bin/uv", "tool", "run"]
    assert scanners.calls[0][0][-2:] == ["skillspector", "--version"]


def test_tier1_reports_source_selected_during_probe(scanners, monkeypatch) -> None:
    tool = Tools.skillspector

    def run(*args, **kwargs):
        tool._path = "/doctor-tools/uv"
        tool.resolution_source = "uv tool run"
        return tool_runner.ToolResult(True, "SkillSpector v2.13.7", "", 0)

    monkeypatch.setattr(tool, "run", run)

    assert _rows(tier="1", checks="security")["SkillSpector"] == (
        "pass",
        "2.13.7 at /doctor-tools/uv (uv tool run)",
    )


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
        "missing": "not installed. Bundled with SkillEvaluator",
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


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 3600.5, 1e300])
def test_readiness_rejects_a_timeout_outside_the_supported_range(timeout) -> None:
    with pytest.raises(ValueError, match="finite, positive and at most 3600 seconds"):
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


def _request_timeouts(requests: list[httpx.Request]) -> list[dict[str, float]]:
    return [request.extensions["timeout"] for request in requests]


# 5.0 equals httpx's default client timeout, which the provider SDKs would
# replace with their own 600-second default unless the bound is passed through.
@pytest.mark.parametrize("timeout", [4.0, 5.0])
def test_tier2_live_probe_sends_one_bounded_request_per_model(provider_api, timeout) -> None:
    rows = _rows(tier="2", verify_models=True, timeout=timeout)

    assert rows["Embedding model"] == ("pass", "returned a 2-dimension vector")
    assert rows["Deduplication model"] == ("pass", "returned a chat completion")
    assert [request.url.path for request in provider_api.requests] == ["/v1/embeddings", "/v1/chat/completions"]
    bound = dict.fromkeys(("connect", "read", "write", "pool"), timeout)
    assert _request_timeouts(provider_api.requests) == [bound, bound]
    chat = json.loads(provider_api.requests[1].content)
    assert chat["model"] == rows["LLM provider"][1].split(" / ", 1)[1]
    # The probe uses the output ceiling of Tier 2 duplicate analysis.
    assert chat["max_tokens"] == CONTENT_DEDUP_LLM_MAX_TOKENS


def test_tier2_anthropic_chat_probe_is_bounded_by_the_timeout(provider_api, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("SKILL_EVAL_EMBEDDING_PROVIDER", "nv_build")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-canary")
    provider_api.responses["messages"] = httpx.Response(
        200,
        json={
            "id": "probe",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )

    rows = _rows(tier="2", verify_models=True, timeout=5.0)

    assert rows["Deduplication model"] == ("pass", "returned a chat completion")
    assert [request.url.path for request in provider_api.requests] == ["/v1/embeddings", "/v1/messages"]
    bound = dict.fromkeys(("connect", "read", "write", "pool"), 5.0)
    assert _request_timeouts(provider_api.requests) == [bound, bound]


def test_tier2_bedrock_chat_probe_passes_the_timeout_to_litellm(provider_api, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("SKILL_EVAL_EMBEDDING_PROVIDER", "nv_build")
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    # Bedrock goes through litellm rather than the shared HTTP client.
    fake_litellm = ModuleType("litellm")
    fake_litellm.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    rows = _rows(tier="2", verify_models=True, timeout=5.0)

    assert rows["Deduplication model"] == ("pass", "returned a chat completion")
    assert [call["timeout"] for call in calls] == [5.0]


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


def test_readiness_payload_rolls_up_worst_status() -> None:
    payload = readiness.readiness_payload(
        command="doctor",
        tiers=[
            ("1", [("SkillSpector", "warn", "version not reported; /x")]),
            ("2", [("Embedding provider", "pass", "nv_build / m")]),
            ("3", [("Harbor agents", "fail", "Unknown: nope")]),
        ],
    )

    assert payload["schema_version"] == 1
    assert payload["command"] == "doctor"
    assert [tier["tier"] for tier in payload["tiers"]] == ["1", "2", "3"]
    assert [tier["status"] for tier in payload["tiers"]] == ["warn", "pass", "fail"]
    assert payload["status"] == "fail"
    assert payload["tiers"][0]["checks"][0] == {
        "check": "SkillSpector",
        "status": "warn",
        "detail": "version not reported; /x",
    }


def test_readiness_payload_strips_terminal_controls() -> None:
    hostile = "\x1b]8;;https://example.invalid\x1b\\click\x1b]8;;\x1b\\\x9b2J\x1b[31mred\tok"
    payload = readiness.readiness_payload(command="doctor", tiers=[("2", [(f"LLM {hostile}", "warn", hostile)])])

    check = payload["tiers"][0]["checks"][0]
    assert check == {"check": "LLM click2Jred\tok", "status": "warn", "detail": "click2Jred\tok"}


def test_default_table_output_matches_shared_renderer(scanners, capsys) -> None:
    rows = readiness.readiness_rows(tier="1", checks="code")

    assert readiness.doctor(tier="1", checks="code") == 0
    printed = capsys.readouterr().out
    readiness.print_readiness_table(rows, title="SkillEvaluator Doctor (Tier 1)")
    rendered = capsys.readouterr().out

    assert printed == rendered
    assert "SkillEvaluator Doctor (Tier 1)" in printed
    assert "Bandit" in printed


def test_cli_without_json_prints_the_same_table(scanners, monkeypatch, capsys) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    assert readiness.doctor(tier="1", checks="code") == 0
    expected = capsys.readouterr().out

    result = CliRunner().invoke(cli, ["doctor", "--tier", "1", "--checks", "code"])

    assert result.exit_code == 0, result.output
    assert result.output == expected
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output)


def test_json_tier1_ready(scanners) -> None:
    result = CliRunner().invoke(cli, ["doctor", "--tier", "1", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["command"] == "doctor"
    assert payload["status"] == "pass"
    assert payload["tiers"][0]["tier"] == "1"
    names = [check["check"] for check in payload["tiers"][0]["checks"]]
    assert names[0] == "CLI package"
    assert "SkillSpector" in names


def test_json_names_a_missing_scanner_and_exits_one(scanners, monkeypatch) -> None:
    monkeypatch.setattr(Tools.bandit, "_path", None)
    result = CliRunner().invoke(cli, ["doctor", "--tier", "1", "--checks", "code", "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    bandit = next(check for check in payload["tiers"][0]["checks"] if check["check"] == "Bandit")
    assert bandit["status"] == "fail"
    assert "not installed" in bandit["detail"]


def test_json_reports_a_malformed_override(scanners, monkeypatch) -> None:
    monkeypatch.setenv("SKILLEVALUATOR_BANDIT_PATH", "relative/bandit")
    monkeypatch.setattr(Tools, "bandit", ExternalTool("Bandit", "bandit", override_env="SKILLEVALUATOR_BANDIT_PATH"))
    result = CliRunner().invoke(cli, ["doctor", "--tier", "1", "--checks", "code", "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    bandit = next(check for check in payload["tiers"][0]["checks"] if check["check"] == "Bandit")
    assert bandit["status"] == "fail"
    assert "SKILLEVALUATOR_BANDIT_PATH must name an absolute executable file" in bandit["detail"]


def test_json_warn_only_exits_zero(scanners) -> None:
    scanners.output["semgrep"] = "development build\n"
    result = CliRunner().invoke(cli, ["doctor", "--tier", "1", "--checks", "code", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "warn"
    semgrep = next(check for check in payload["tiers"][0]["checks"] if check["check"] == "Semgrep")
    assert semgrep["status"] == "warn"
    assert "version not reported" in semgrep["detail"]


def test_json_stdout_contains_only_json(scanners) -> None:
    result = CliRunner().invoke(cli, ["doctor", "--tier", "1", "--json"])

    assert result.exit_code == 0, result.output
    json.loads(result.output)
    assert result.output.lstrip().startswith("{")
    assert "SkillEvaluator Doctor" not in result.output
    stderr = getattr(result, "stderr", "") or ""
    assert "SkillEvaluator Doctor" not in stderr


def test_json_stdout_parses_when_skillspector_path_is_set(scanners, monkeypatch, tmp_path) -> None:
    executable = tmp_path / "skillspector"
    executable.write_text("#!/bin/sh\necho 'SkillSpector v2.13.4'\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("SKILLEVALUATOR_SKILLSPECTOR_PATH", str(executable))
    loaded: list[object] = []

    class _LoadOnUse:
        def __getattr__(self, name: str):
            if not loaded:
                loaded.append(
                    ExternalTool(
                        "skillspector",
                        "skillspector",
                        prefer_interpreter_sibling=True,
                        override_env="SKILLEVALUATOR_SKILLSPECTOR_PATH",
                        managed=True,
                    )
                )
            return getattr(loaded[0], name)

    monkeypatch.setattr(Tools, "skillspector", _LoadOnUse())
    result = CliRunner().invoke(cli, ["doctor", "--tier", "1", "--checks", "security", "--json"])

    payload = json.loads(result.stdout)
    assert payload["command"] == "doctor"
    spector = next(check for check in payload["tiers"][0]["checks"] if check["check"] == "SkillSpector")
    assert "SKILLEVALUATOR_SKILLSPECTOR_PATH" in spector["detail"]
    assert "WARNING" not in result.stdout
    assert "Using explicit" not in result.stdout


def test_fresh_process_with_skillspector_path_keeps_stdout_json() -> None:
    """Pins the import-time half: building ``Tools`` must not log to stdout."""
    env = {**os.environ, "SKILLEVALUATOR_SKILLSPECTOR_PATH": sys.executable}

    imported = subprocess.run(
        [sys.executable, "-c", "import skillevaluator.cli"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout == ""

    doctor_run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from skillevaluator.cli import cli; "
            "sys.argv = ['skillevaluator', 'doctor', '--tier', '1', '--checks', 'security', '--json']; cli()",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    payload = json.loads(doctor_run.stdout)
    assert payload["command"] == "doctor"
    assert "Using explicit" not in doctor_run.stdout
    assert "Using explicit" in doctor_run.stderr


def test_health_check_json_uses_command_name(scanners) -> None:
    result = CliRunner().invoke(cli, ["health-check", "--tier", "1", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "health-check"
    assert payload["tiers"][0]["tier"] == "1"


def test_json_tier2_offline(provider_api) -> None:
    result = CliRunner().invoke(cli, ["doctor", "--tier", "2", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tiers"][0]["tier"] == "2"
    checks = {check["check"]: check for check in payload["tiers"][0]["checks"]}
    assert checks["Embedding provider"]["status"] == "pass"
    assert "Embedding model" not in checks
    assert provider_api.requests == []
