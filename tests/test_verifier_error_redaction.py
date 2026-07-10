# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-redaction regressions for retained Tier 3 verifier errors."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from skillevaluator.inference.client import LLMClient
from skillevaluator.tier3.eval_core import llm_judge

_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)

_CREDENTIALS = {
    "OPENAI_API_KEY": "dummy-openai-credential-DO-NOT-USE",
    "NVIDIA_API_KEY": "dummy-nvidia-credential-DO-NOT-USE",
    "ANTHROPIC_API_KEY": "dummy-anthropic-credential-DO-NOT-USE",
    "SKILL_EVAL_LLM_API_KEY": "dummy-generic-credential-DO-NOT-USE",
    "AWS_ACCESS_KEY_ID": "dummy-aws-access-key-id-DO-NOT-USE",
    "AWS_SECRET_ACCESS_KEY": "dummy-aws-secret-credential-DO-NOT-USE",
    "AWS_SECURITY_TOKEN": "dummy-aws-legacy-security-token-DO-NOT-USE",
    "AWS_SESSION_TOKEN": "dummy-aws-session-credential-DO-NOT-USE",
}

_PROVIDER_ENV = (
    *_CREDENTIALS,
    "SKILL_EVAL_LLM_PROVIDER",
    "SKILL_EVAL_LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "LLM_JUDGE_FALLBACK_MODELS",
    "LLM_JUDGE_MODEL",
    "SKILL_EVAL_JUDGE_MODEL",
    "SKILL_EVAL_LLM_MODEL",
    "AWS_PROFILE",
)


@pytest.fixture(autouse=True)
def _clean_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every regression hermetic even when the invoking shell has credentials."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def verifier_module(tmp_path: Path):
    module_name = f"harbor_template_error_redaction_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(module_name, _TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    module.VERIFIER_DIR = tmp_path
    module.REWARD_JSON = tmp_path / "reward.json"
    module.REWARD_TXT = tmp_path / "reward.txt"
    module.SKILL_EVALUATOR_REWARD_JSON = tmp_path / "skill_evaluator_reward.json"
    return module


def _http_error(body: str, *, code: int = 401, reason: str = "Unauthorized") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://provider.invalid/chat/completions",
        code,
        reason,
        hdrs=None,
        fp=io.BytesIO(body.encode()),
    )


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


@pytest.mark.parametrize("module_fixture", ["source_judge", "verifier_module"])
def test_http_error_formatter_redacts_every_configured_credential(
    module_fixture: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = llm_judge if module_fixture == "source_judge" else request.getfixturevalue(module_fixture)
    for name, credential in _CREDENTIALS.items():
        monkeypatch.setenv(name, credential)

    detail = module._format_http_error(_http_error("echo: " + " | ".join(_CREDENTIALS.values())))

    assert detail.startswith("HTTP 401: Unauthorized - echo:")
    assert "[REDACTED]" in detail
    for credential in _CREDENTIALS.values():
        assert credential not in detail


@pytest.mark.parametrize("module_fixture", ["source_judge", "verifier_module"])
def test_http_error_formatter_redacts_overlapping_credentials_longest_first(
    module_fixture: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = llm_judge if module_fixture == "source_judge" else request.getfixturevalue(module_fixture)
    monkeypatch.setenv("OPENAI_API_KEY", "shared-dummy-secret")
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "shared-dummy-secret-with-suffix")

    detail = module._format_http_error(_http_error("echo: shared-dummy-secret-with-suffix"))

    assert detail.endswith("echo: [REDACTED]")
    assert "with-suffix" not in detail


@pytest.mark.parametrize("module_fixture", ["source_judge", "verifier_module"])
def test_http_error_formatter_redacts_before_truncating_body(
    module_fixture: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = llm_judge if module_fixture == "source_judge" else request.getfixturevalue(module_fixture)
    credential = _CREDENTIALS["OPENAI_API_KEY"]
    monkeypatch.setenv("OPENAI_API_KEY", credential)

    detail = module._format_http_error(_http_error("x" * 490 + credential + " trailing diagnostics"))

    assert "[REDACTED]" in detail
    assert "dummy-openai-credential" not in detail


@pytest.mark.parametrize("module_fixture", ["source_judge", "verifier_module"])
def test_http_error_fallback_detection_uses_full_raw_body_before_display_truncation(
    module_fixture: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = llm_judge if module_fixture == "source_judge" else request.getfixturevalue(module_fixture)
    credential = "S" * 200
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    body = "x" * 450 + credential + " model not found"

    detail, should_try_fallback = module._format_http_error_with_fallback(
        _http_error(body, code=403, reason="Forbidden")
    )

    assert should_try_fallback is True
    assert "model not found" in detail
    assert credential not in detail


@pytest.mark.parametrize("module_fixture", ["source_judge", "verifier_module"])
def test_exact_secret_redaction_uses_existing_eight_character_minimum(
    module_fixture: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = llm_judge if module_fixture == "source_judge" else request.getfixturevalue(module_fixture)
    monkeypatch.setenv("OPENAI_API_KEY", "1234567")

    assert module._redact_configured_credentials("value=1234567") == "value=1234567"

    monkeypatch.setenv("OPENAI_API_KEY", "12345678")
    assert module._redact_configured_credentials("value=12345678") == "value=[REDACTED]"


def test_generated_call_public_llm_redacts_http_error_body(
    verifier_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _CREDENTIALS["OPENAI_API_KEY"]
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", credential)

    def fail_request(*_args, **_kwargs):
        raise _http_error(f'{{"error":"credential echoed: {credential}"}}')

    monkeypatch.setattr(verifier_module.urllib.request, "urlopen", fail_request)

    content, error = verifier_module.call_public_llm("safe prompt", allow_model_fallback=False)

    assert content is None
    assert error is not None
    assert credential not in error
    assert "[REDACTED]" in error
    assert "HTTP 401: Unauthorized" in error


def test_generated_call_public_llm_redacts_generic_exception(
    verifier_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _CREDENTIALS["NVIDIA_API_KEY"]
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", credential)

    def fail_request(*_args, **_kwargs):
        raise RuntimeError(f"transport included {credential} in diagnostics")

    monkeypatch.setattr(verifier_module.urllib.request, "urlopen", fail_request)

    content, error = verifier_module.call_public_llm("safe prompt")

    assert content is None
    assert error is not None
    assert credential not in error
    assert error == (
        f"Public provider call failed for {verifier_module.DEFAULT_JUDGE_MODEL}: "
        "transport included [REDACTED] in diagnostics"
    )


@pytest.mark.parametrize(
    ("provider", "credential_name", "provider_call"),
    [
        ("anthropic", "ANTHROPIC_API_KEY", "_call_anthropic"),
        ("bedrock", "AWS_SECRET_ACCESS_KEY", "_call_bedrock"),
    ],
)
def test_generated_call_public_llm_redacts_provider_returned_errors(
    verifier_module,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    credential_name: str,
    provider_call: str,
) -> None:
    credential = _CREDENTIALS[credential_name]
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
    monkeypatch.setenv(credential_name, credential)
    monkeypatch.setattr(
        verifier_module,
        provider_call,
        lambda *_args, **_kwargs: (None, f"{provider} diagnostic echoed {credential}"),
    )

    content, error = verifier_module.call_public_llm("safe prompt")

    assert content is None
    assert error == f"{provider} diagnostic echoed [REDACTED]"
    assert credential not in error


def test_generated_call_public_llm_preserves_fallback_detection_and_redacts_exhausted_error(
    verifier_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deliberately overlap the credential with the categorical error token. The
    # verifier must decide whether to fall back before redacting retained text.
    credential = "key_model_access_denied"
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", credential)
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", "fallback-model")
    calls = 0

    def deny_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise _http_error(
            f'{{"error":"key_model_access_denied", "credential":"{credential}"}}',
            code=403,
            reason="Forbidden",
        )

    monkeypatch.setattr(verifier_module.urllib.request, "urlopen", deny_model)

    content, error = verifier_module.call_public_llm("safe prompt", model="primary-model")

    assert content is None
    assert calls == 2
    assert error is not None
    assert error.startswith("LLM judge model fallback exhausted:")
    assert "primary-model" in error
    assert "fallback-model" in error
    assert "[REDACTED]" in error
    assert credential not in error


def test_generated_call_public_llm_does_not_redact_successful_content(
    verifier_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _CREDENTIALS["OPENAI_API_KEY"]
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    response = _FakeResponse({"choices": [{"message": {"content": f"model content: {credential}"}}]})
    monkeypatch.setattr(verifier_module.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    content, error = verifier_module.call_public_llm("safe prompt")

    assert content == f"model content: {credential}"
    assert error is None


def test_source_call_public_llm_redacts_generic_client_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    credential = _CREDENTIALS["ANTHROPIC_API_KEY"]

    def fail_completion(*_args, **_kwargs):
        raise RuntimeError(f"SDK diagnostic echoed {credential}")

    monkeypatch.setattr(LLMClient, "completions", fail_completion)

    content, error = llm_judge.call_public_llm("safe prompt", api_key=credential)

    assert content is None
    assert error == "Public provider call failed: SDK diagnostic echoed [REDACTED]"
    assert credential not in error


def test_recursive_sanitizer_preserves_container_types(verifier_module, monkeypatch: pytest.MonkeyPatch) -> None:
    credential = _CREDENTIALS["OPENAI_API_KEY"]
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    value = {"errors": [f"list {credential}", (f"tuple {credential}",)]}

    sanitized = verifier_module._sanitize_error_value(value)

    assert isinstance(sanitized, dict)
    assert isinstance(sanitized["errors"], list)
    assert isinstance(sanitized["errors"][1], tuple)
    assert sanitized == {"errors": ["list [REDACTED]", ("tuple [REDACTED]",)]}


def test_write_reward_outputs_recursively_redacts_credentials_in_actual_artifacts(
    verifier_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, credential in _CREDENTIALS.items():
        monkeypatch.setenv(name, credential)
    result = {
        "score": 0.0,
        "error": {
            "primary": f"OpenAI returned {_CREDENTIALS['OPENAI_API_KEY']}",
            "nested": [
                f"NVIDIA returned {_CREDENTIALS['NVIDIA_API_KEY']}",
                (
                    f"Anthropic returned {_CREDENTIALS['ANTHROPIC_API_KEY']}",
                    {
                        "generic": _CREDENTIALS["SKILL_EVAL_LLM_API_KEY"],
                        _CREDENTIALS["OPENAI_API_KEY"]: "credential used as an error mapping key",
                        "aws": [
                            _CREDENTIALS["AWS_ACCESS_KEY_ID"],
                            _CREDENTIALS["AWS_SECRET_ACCESS_KEY"],
                            _CREDENTIALS["AWS_SECURITY_TOKEN"],
                            _CREDENTIALS["AWS_SESSION_TOKEN"],
                        ],
                    },
                ),
            ],
        },
    }

    verifier_module.write_reward_outputs(result, 0.0)

    artifacts = [
        verifier_module.SKILL_EVALUATOR_REWARD_JSON,
        verifier_module.REWARD_JSON,
        verifier_module.REWARD_TXT,
    ]
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in artifacts)
    for credential in _CREDENTIALS.values():
        assert credential not in artifact_text
    assert "[REDACTED]" in verifier_module.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8")
    assert json.loads(verifier_module.REWARD_JSON.read_text(encoding="utf-8")) == {
        "score": 0.0,
        "overall": 0.0,
    }
    assert _CREDENTIALS["OPENAI_API_KEY"] in result["error"]["primary"]


def test_write_reward_outputs_preserves_fixed_schema_keys_when_a_credential_matches_one(
    verifier_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "security")
    result = {
        "score": 0.5,
        "security": 1.0,
        "error": "provider echoed security",
    }

    verifier_module.write_reward_outputs(result, 0.75)

    rich_reward = json.loads(verifier_module.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric_reward = json.loads(verifier_module.REWARD_JSON.read_text(encoding="utf-8"))
    assert rich_reward == {
        "score": 0.5,
        "security": 1.0,
        "error": "provider echoed [REDACTED]",
    }
    assert numeric_reward == {
        "score": 0.5,
        "security": 1.0,
        "overall": 0.75,
    }
