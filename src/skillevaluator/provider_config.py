# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public LLM and embedding provider configuration."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

PUBLIC_NVIDIA_BUILD_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Pinned frontier chat defaults (not floating aliases like ``gpt-5`` / ``claude-opus-latest``).
# Harbor ``templates/eval.py`` cannot import this module — keep its local
# ``DEFAULT_JUDGE_MODEL`` in sync via the drift test in
# ``tests/tier3/test_judge_parse_robustness.py``.
CHAT_DEFAULT_OPENAI = "gpt-5.6-sol"
CHAT_DEFAULT_ANTHROPIC = "claude-opus-5"
CHAT_DEFAULT_BEDROCK = "us.anthropic.claude-opus-5"
# Lower-cost OpenAI alternative for ``SKILL_EVAL_LLM_MODEL`` overrides.
CHAT_CHEAP_OPENAI = "gpt-5.4-mini"

CHAT_DEFAULT_MODELS = {
    "openai": CHAT_DEFAULT_OPENAI,
    "anthropic": CHAT_DEFAULT_ANTHROPIC,
    "nv_build": "nvidia/nemotron-3-nano-30b-a3b",
    "bedrock": CHAT_DEFAULT_BEDROCK,
}
_EMBEDDING_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "nv_build": "nvidia/nv-embed-v1",
}
_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic", "nv_build", "bedrock", "openai-compatible"})
_NO_CUSTOM_TEMPERATURE_MODEL_IDS = frozenset({"claude-mythos-preview"})
_ANTHROPIC_BEDROCK_PREFIX_RE = re.compile(r"^(?:(?:[a-z]{2}|global)\.)?anthropic\.")
_VERSIONED_CLAUDE_MODEL_RE = re.compile(
    r"^claude-[a-z][a-z-]*-(?P<major>\d+)"
    r"(?:-(?P<minor>\d{1,2}))?"
    r"(?:-(?:\d{8}|latest))?"
    r"(?:-v\d+)?(?::\d+)?$"
)


def _model_leaf(model: str) -> str:
    """Return a normalized model ID for capability checks only."""
    leaf = str(model or "").strip().casefold().rsplit("/", 1)[-1]
    return _ANTHROPIC_BEDROCK_PREFIX_RE.sub("", leaf, count=1)


def _supports_custom_temperature(model: str) -> bool:
    """Return whether ``model`` accepts a non-default temperature value."""
    leaf = _model_leaf(model)
    if leaf.startswith("gpt-5") or leaf in _NO_CUSTOM_TEMPERATURE_MODEL_IDS:
        return False

    match = _VERSIONED_CLAUDE_MODEL_RE.fullmatch(leaf)
    if match is None:
        return True
    version = (int(match.group("major")), int(match.group("minor") or 0))
    return version < (4, 7)


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
    credential_env: str | None = None
    base_url_env: str | None = None

    def child_environment(self) -> dict[str, str]:
        """Return this provider's public credential settings for a child process."""
        environment: dict[str, str] = {}
        if self.credential_env and self.api_key:
            environment[self.credential_env] = self.api_key

        if self.base_url_env and self.base_url:
            environment[self.base_url_env] = self.base_url
        elif self.provider == "openai" and self.base_url:
            environment["OPENAI_BASE_URL"] = self.base_url
        elif self.provider == "anthropic" and self.base_url:
            environment["ANTHROPIC_BASE_URL"] = self.base_url
        elif self.provider == "openai-compatible" and self.base_url:
            environment["SKILL_EVAL_LLM_BASE_URL"] = self.base_url
        elif self.provider == "bedrock" and self.region:
            environment["AWS_REGION"] = self.region

        return environment


def resolve_llm_provider(environ: Mapping[str, str] | None = None) -> ProviderConfig:
    """Resolve the public provider used for LLM-backed checks and judging."""
    env = _environment(environ)
    provider = _selected_provider(env, "SKILL_EVAL_LLM_PROVIDER")
    _validate_provider(provider, variable="SKILL_EVAL_LLM_PROVIDER")
    configured_model = env.get("SKILL_EVAL_LLM_MODEL")
    if configured_model is None:
        model = _default_chat_model(provider)
    else:
        model = configured_model.strip()
        if not model:
            raise ProviderConfigurationError("SKILL_EVAL_LLM_MODEL must be a non-empty string when set.")

    if provider == "openai":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=_required(env, "OPENAI_API_KEY"),
            base_url=(env.get("SKILL_EVAL_LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/"),
            litellm_model=f"openai/{model}",
            credential_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
        )
    if provider == "anthropic":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=_required(env, "ANTHROPIC_API_KEY"),
            base_url=(env.get("SKILL_EVAL_LLM_BASE_URL") or env.get("ANTHROPIC_BASE_URL") or None),
            litellm_model=f"anthropic/{model}",
            credential_env="ANTHROPIC_API_KEY",
            base_url_env="ANTHROPIC_BASE_URL",
        )
    if provider == "nv_build":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=_required(env, "NVIDIA_API_KEY"),
            base_url=PUBLIC_NVIDIA_BUILD_BASE_URL,
            litellm_model=f"openai/{model}",
            credential_env="NVIDIA_API_KEY",
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
        model=model,
        api_key=_required(env, "SKILL_EVAL_LLM_API_KEY"),
        base_url=_required(env, "SKILL_EVAL_LLM_BASE_URL").rstrip("/"),
        litellm_model=f"openai/{model}",
        credential_env="SKILL_EVAL_LLM_API_KEY",
        base_url_env="SKILL_EVAL_LLM_BASE_URL",
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
            f"SKILL_EVAL_EMBEDDING_PROVIDER is required because {provider} does not provide embeddings. "
            "Set SKILL_EVAL_EMBEDDING_PROVIDER=nv_build|openai|openai-compatible (NVIDIA_API_KEY or "
            "OPENAI_API_KEY supply the first two)."
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
            credential_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
        )
    if provider == "nv_build":
        return ProviderConfig(
            provider=provider,
            model=env.get("SKILL_EVAL_EMBEDDING_MODEL") or _EMBEDDING_DEFAULT_MODELS[provider],
            api_key=_required(env, "NVIDIA_API_KEY"),
            base_url=PUBLIC_NVIDIA_BUILD_BASE_URL,
            litellm_model=f"openai/{env.get('SKILL_EVAL_EMBEDDING_MODEL') or _EMBEDDING_DEFAULT_MODELS[provider]}",
            credential_env="NVIDIA_API_KEY",
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
        credential_env=(
            "SKILL_EVAL_EMBEDDING_API_KEY"
            if env.get("SKILL_EVAL_EMBEDDING_API_KEY", "").strip()
            else "SKILL_EVAL_LLM_API_KEY"
        ),
        base_url_env="SKILL_EVAL_EMBEDDING_BASE_URL",
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
    available = [
        provider
        for provider, credential in (
            ("nv_build", "NVIDIA_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
        )
        if environ.get(credential, "").strip()
    ]
    if len(available) > 1:
        raise ProviderConfigurationError(
            f"{variable} is required when multiple public provider credentials are configured."
        )
    if available:
        return available[0]
    prefix = variable.removesuffix("_PROVIDER")
    if "EMBEDDING" in variable:
        # Anthropic/Bedrock have no embedding models: recommending them (or the
        # ANTHROPIC_API_KEY auto-detection) here would send the user straight
        # into the "does not provide embeddings" rejection below.
        raise ProviderConfigurationError(
            f"No provider is configured ({variable} unset and no credential found). Set one of: "
            "NVIDIA_API_KEY for NVIDIA Build (build.nvidia.com) or OPENAI_API_KEY (auto-detected) — "
            f"or set {variable}=openai|nv_build|openai-compatible explicitly "
            f"(openai-compatible also needs {prefix}_BASE_URL, {prefix}_API_KEY, and {prefix}_MODEL)."
        )
    raise ProviderConfigurationError(
        f"No provider is configured ({variable} unset and no credential found). Set one of: "
        "NVIDIA_API_KEY for NVIDIA Build (build.nvidia.com), OPENAI_API_KEY, or ANTHROPIC_API_KEY "
        f"(auto-detected) — or set {variable}=openai|anthropic|nv_build|bedrock|openai-compatible "
        f"explicitly (openai-compatible also needs {prefix}_BASE_URL, {prefix}_API_KEY, and {prefix}_MODEL)."
    )


def _validate_provider(provider: str, *, variable: str) -> None:
    if provider not in _SUPPORTED_PROVIDERS:
        choices = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise ProviderConfigurationError(f"{variable} must be one of: {choices}.")


def _default_chat_model(provider: str) -> str:
    try:
        return CHAT_DEFAULT_MODELS[provider]
    except KeyError as exc:
        raise ProviderConfigurationError("SKILL_EVAL_LLM_MODEL is required for openai-compatible providers.") from exc
