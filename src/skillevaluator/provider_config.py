# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public LLM and embedding provider configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

PUBLIC_NVIDIA_BUILD_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

_CHAT_DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-5",
    "nv_build": "meta/llama-3.1-8b-instruct",
    "bedrock": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}
_EMBEDDING_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "nv_build": "nvidia/nv-embed-v1",
}
_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic", "nv_build", "bedrock", "openai-compatible"})


class ProviderConfigurationError(ValueError):
    """Raised when a selected public provider is not fully configured."""


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider values safe to pass to the relevant SDK."""

    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    litellm_model: str
    region: str | None = None


def resolve_llm_provider(environ: Mapping[str, str] | None = None) -> ProviderConfig:
    """Resolve the public provider used for LLM-backed checks and judging."""
    env = _environment(environ)
    provider = _selected_provider(env, "SKILL_EVAL_LLM_PROVIDER")
    _validate_provider(provider, variable="SKILL_EVAL_LLM_PROVIDER")
    model = env.get("SKILL_EVAL_LLM_MODEL") or _default_chat_model(provider)

    if provider == "openai":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=_required(env, "OPENAI_API_KEY"),
            base_url=(env.get("SKILL_EVAL_LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/"),
            litellm_model=f"openai/{model}",
        )
    if provider == "anthropic":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=_required(env, "ANTHROPIC_API_KEY"),
            base_url=(env.get("SKILL_EVAL_LLM_BASE_URL") or env.get("ANTHROPIC_BASE_URL") or None),
            litellm_model=f"anthropic/{model}",
        )
    if provider == "nv_build":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=_required(env, "NVIDIA_API_KEY"),
            base_url=(env.get("SKILL_EVAL_LLM_BASE_URL") or PUBLIC_NVIDIA_BUILD_BASE_URL).rstrip("/"),
            litellm_model=f"openai/{model}",
        )
    if provider == "bedrock":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=None,
            base_url=None,
            litellm_model=f"bedrock/{model}",
            region=env.get("AWS_REGION") or "us-west-2",
        )

    return ProviderConfig(
        provider=provider,
        model=_required(env, "SKILL_EVAL_LLM_MODEL"),
        api_key=_required(env, "SKILL_EVAL_LLM_API_KEY"),
        base_url=_required(env, "SKILL_EVAL_LLM_BASE_URL").rstrip("/"),
        litellm_model=f"openai/{_required(env, 'SKILL_EVAL_LLM_MODEL')}",
    )


def resolve_embedding_provider(environ: Mapping[str, str] | None = None) -> ProviderConfig:
    """Resolve the embedding provider used by Tier 2 semantic overlap checks."""
    env = _environment(environ)
    provider = (
        env.get("SKILL_EVAL_EMBEDDING_PROVIDER")
        or env.get("SKILL_EVAL_LLM_PROVIDER")
        or _selected_provider(env, "SKILL_EVAL_EMBEDDING_PROVIDER")
    ).lower()
    if provider in {"anthropic", "bedrock"}:
        raise ProviderConfigurationError(
            f"SKILL_EVAL_EMBEDDING_PROVIDER is required because {provider} does not provide embeddings."
        )
    _validate_provider(provider, variable="SKILL_EVAL_EMBEDDING_PROVIDER")

    if provider == "openai":
        return ProviderConfig(
            provider=provider,
            model=env.get("SKILL_EVAL_EMBEDDING_MODEL") or _EMBEDDING_DEFAULT_MODELS[provider],
            api_key=_required(env, "OPENAI_API_KEY"),
            base_url=(env.get("SKILL_EVAL_EMBEDDING_BASE_URL") or env.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip(
                "/"
            ),
            litellm_model=f"openai/{env.get('SKILL_EVAL_EMBEDDING_MODEL') or _EMBEDDING_DEFAULT_MODELS[provider]}",
        )
    if provider == "nv_build":
        return ProviderConfig(
            provider=provider,
            model=env.get("SKILL_EVAL_EMBEDDING_MODEL") or _EMBEDDING_DEFAULT_MODELS[provider],
            api_key=_required(env, "NVIDIA_API_KEY"),
            base_url=(env.get("SKILL_EVAL_EMBEDDING_BASE_URL") or PUBLIC_NVIDIA_BUILD_BASE_URL).rstrip("/"),
            litellm_model=f"openai/{env.get('SKILL_EVAL_EMBEDDING_MODEL') or _EMBEDDING_DEFAULT_MODELS[provider]}",
        )

    model = env.get("SKILL_EVAL_EMBEDDING_MODEL")
    if not model:
        raise ProviderConfigurationError("SKILL_EVAL_EMBEDDING_MODEL is required for openai-compatible embeddings.")
    return ProviderConfig(
        provider=provider,
        model=model,
        api_key=env.get("SKILL_EVAL_EMBEDDING_API_KEY") or _required(env, "SKILL_EVAL_LLM_API_KEY"),
        base_url=(env.get("SKILL_EVAL_EMBEDDING_BASE_URL") or _required(env, "SKILL_EVAL_LLM_BASE_URL")).rstrip("/"),
        litellm_model=f"openai/{model}",
    )


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _required(environ: Mapping[str, str], variable: str) -> str:
    value = environ.get(variable, "").strip()
    if not value:
        raise ProviderConfigurationError(f"{variable} is required for the selected provider.")
    return value


def _selected_provider(environ: Mapping[str, str], variable: str) -> str:
    configured = environ.get(variable, "").strip().lower()
    if configured:
        return configured
    if environ.get("NVIDIA_API_KEY", "").strip():
        return "nv_build"
    if environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    raise ProviderConfigurationError(f"{variable} is required when no public provider credential is configured.")


def _validate_provider(provider: str, *, variable: str) -> None:
    if provider not in _SUPPORTED_PROVIDERS:
        choices = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise ProviderConfigurationError(f"{variable} must be one of: {choices}.")


def _default_chat_model(provider: str) -> str:
    try:
        return _CHAT_DEFAULT_MODELS[provider]
    except KeyError as exc:
        raise ProviderConfigurationError("SKILL_EVAL_LLM_MODEL is required for openai-compatible providers.") from exc
