# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- trusted reusable CI helper.
"""Bounded CI inputs mapped to an isolated runtime provider environment."""

from __future__ import annotations

import re
import unicodedata

from skillevaluator.provider_config import (
    ProviderConfigurationError,
    _normalize_anthropic_base_url,
    resolve_embedding_provider,
    resolve_llm_provider,
)

PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "nv_build": "NVIDIA_API_KEY",
    "openai-compatible": "SKILL_EVAL_LLM_API_KEY",
}
CREDENTIAL_KEYS = frozenset((*PROVIDER_KEYS.values(), "SKILL_EVAL_EMBEDDING_API_KEY"))
SAFE_ENV = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "XDG_RUNTIME_DIR",
        "SYSTEMROOT",
        "SKILLEVALUATOR_RUNTIME_DIR",
    }
)


def _model(value: str) -> str:
    if (
        not value.strip()
        or len(value) > 256
        or any(c.isspace() or unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in value)
    ):
        raise ValueError("Model configuration contains an unsupported value.")
    return value


def _api_root(value: str) -> str:
    # Reuse the runtime's strict authority/path validation without Anthropic's
    # protocol-specific removal of the final /v1 segment.
    if len(value) > 4096:
        raise ValueError("Provider API root configuration is invalid.")
    try:
        _normalize_anthropic_base_url(value, variable="API root")
    except (ValueError, UnicodeError):
        raise ValueError("Provider API root configuration is invalid.") from None
    return value.rstrip("/")


def required_credentials(child: dict[str, str], tier: int) -> frozenset[str]:
    """Credential variable names required by the selected tier, never values."""
    keys = {PROVIDER_KEYS[child["SKILL_EVAL_LLM_PROVIDER"]]}
    if tier == 2:
        embedding = child["SKILL_EVAL_EMBEDDING_PROVIDER"]
        keys.add("SKILL_EVAL_EMBEDDING_API_KEY" if embedding == "openai-compatible" else PROVIDER_KEYS[embedding])
    return frozenset(keys)


def model_environment(env: dict[str, str], tier: int) -> dict[str, str]:
    """Forward selected provider settings, never ambient CI or vendor credentials."""
    if tier not in {1, 2, 3}:
        raise ValueError("Select a supported evaluation tier.")
    base_provider = env.get("SKILL_EVAL_PROVIDER") or "openai"
    provider = (env.get(f"SKILL_EVAL_TIER{tier}_PROVIDER") or base_provider) if tier != 2 else base_provider
    embedding = env.get("SKILL_EVAL_EMBEDDING_PROVIDER") or (
        "openai" if base_provider == "anthropic" else base_provider
    )
    if provider not in PROVIDER_KEYS or (tier == 2 and embedding not in {"openai", "nv_build", "openai-compatible"}):
        raise ValueError("Select supported chat and embedding providers.")
    child = {key: value for key, value in env.items() if key in SAFE_ENV}
    child.update(
        {
            "SKILL_EVAL_LLM_PROVIDER": provider,
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
            "SKILLEVALUATOR_TELEMETRY_DISABLED": "1",
            "SKILLEVALUATOR_HARBOR_VIEWER_UPLOAD": "false",
        }
    )
    selected_providers = {provider}
    if tier == 2:
        child["SKILL_EVAL_EMBEDDING_PROVIDER"] = embedding
        selected_providers.add(embedding)
    for selected in selected_providers - {"openai-compatible"}:
        key = PROVIDER_KEYS[selected]
        if env.get(key):
            child[key] = env[key]
    if provider == "openai-compatible" and env.get("SKILL_EVAL_LLM_API_KEY"):
        child["SKILL_EVAL_LLM_API_KEY"] = env["SKILL_EVAL_LLM_API_KEY"]
    same_provider = provider == base_provider
    model = env.get(f"SKILL_EVAL_TIER{tier}_MODEL") if tier != 2 else None
    # Tier 1 may opt into the provider default independently of Tier 2.
    tier1_default = tier == 1 and "SKILL_EVAL_TIER1_MODEL" in env
    model = model or (env.get("SKILL_EVAL_MODEL") if same_provider and not tier1_default else None)
    if model:
        child["SKILL_EVAL_LLM_MODEL"] = child["LLM_JUDGE_MODEL"] = _model(model)
    base_url = env.get(f"SKILL_EVAL_TIER{tier}_BASE_URL") if tier != 2 else None
    base_url = base_url or (env.get("SKILL_EVAL_BASE_URL") if same_provider else None)
    if base_url and provider != "nv_build":
        child["SKILL_EVAL_LLM_BASE_URL"] = _api_root(base_url)
    if tier == 2:
        if embedding == "openai-compatible":
            embedding_key = env.get("SKILL_EVAL_EMBEDDING_API_KEY") or env.get("SKILL_EVAL_LLM_API_KEY")
            if embedding_key:
                child["SKILL_EVAL_EMBEDDING_API_KEY"] = embedding_key
            embedding_url = env.get("SKILL_EVAL_EMBEDDING_BASE_URL") or env.get("SKILL_EVAL_BASE_URL")
            if embedding_url:
                child["SKILL_EVAL_EMBEDDING_BASE_URL"] = _api_root(embedding_url)
        elif embedding == "openai" and env.get("SKILL_EVAL_EMBEDDING_BASE_URL"):
            child["SKILL_EVAL_EMBEDDING_BASE_URL"] = _api_root(env["SKILL_EVAL_EMBEDDING_BASE_URL"])
        if env.get("SKILL_EVAL_EMBEDDING_MODEL"):
            child["SKILL_EVAL_EMBEDDING_MODEL"] = _model(env["SKILL_EVAL_EMBEDDING_MODEL"])
    # Validate through the native runtime with non-secret placeholders so absent
    # credentials remain a handled CI precondition rather than an exception.
    validation = dict(child)
    for key in CREDENTIAL_KEYS:
        validation.setdefault(key, "ci-validation-placeholder")
    try:
        resolve_llm_provider(validation)
        # Only Tier 2 executes embeddings. A chat-only job (for example the
        # Tier 1 image job, which receives no embedding settings) must not fail
        # on an embedding route it never uses.
        if tier == 2:
            resolve_embedding_provider(validation)
    except (ProviderConfigurationError, ValueError):
        raise ValueError("Selected provider configuration is incomplete or invalid.") from None
    # Only the selected tier's routes are validated or receive credentials.
    for key in CREDENTIAL_KEYS - required_credentials(child, tier):
        child.pop(key, None)
    return child


def agent_model_overrides(env: dict[str, str]) -> tuple[str, ...]:
    """Parse bounded comma/newline native agent model choices without rewriting IDs."""
    raw = env.get("SKILL_EVAL_AGENT_MODELS", "")
    if not raw.strip():
        return ()
    error = "Native agent model configuration is invalid."
    if len(raw) > 4096 or any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} and c != "\n" for c in raw):
        raise ValueError(error)
    from skillevaluator.tier3.commands import parse_agent_model_overrides

    entries = tuple(entry.strip() for entry in re.split(r"[,\n]", raw) if entry.strip())
    try:
        parsed = parse_agent_model_overrides(entries)
        if len(entries) > 3 or set(parsed) - {"claude-code", "codex", "opencode"}:
            raise ValueError(error)
        for models in parsed.values():
            for model in models:
                _model(model)
    except ValueError:
        raise ValueError(error) from None
    return entries
