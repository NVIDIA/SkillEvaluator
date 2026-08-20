# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public LLM and embedding provider configuration."""

from __future__ import annotations

import ipaddress
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import quote, unquote_to_bytes, urlsplit, urlunsplit

import idna

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
_ANTHROPIC_DNS_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_ANTHROPIC_INTERNAL_LABEL_RE = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?$")
_ANTHROPIC_IPV6_ZONE_RE = re.compile(r"^[A-Za-z0-9._~-]+$")
_ANTHROPIC_PATH_SAFE = "/:@!$&'()*+,;=-._~%"
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_HEX_DIGIT_BYTES = frozenset(b"0123456789abcdefABCDEF")
_UNRESERVED_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
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
            base_url=_anthropic_base_url(env),
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


def _anthropic_base_url(environ: Mapping[str, str]) -> str | None:
    for variable in ("SKILL_EVAL_LLM_BASE_URL", "ANTHROPIC_BASE_URL"):
        if value := environ.get(variable):
            return _normalize_anthropic_base_url(value, variable=variable)
    return None


def _normalize_anthropic_base_url(value: str, *, variable: str) -> str:
    error = (
        f"{variable} must be an absolute HTTP or HTTPS URL representing an API root without credentials, query, fragment, "
        "whitespace, control characters, backslashes, an invalid authority, or a /v1/messages endpoint."
    )
    if "\\" in value or any(
        character.isspace() or unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value
    ):
        raise ProviderConfigurationError(error)
    if "?" in value or "#" in value:
        raise ProviderConfigurationError(error)

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise ProviderConfigurationError(error) from None

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
        or parsed.netloc.endswith(":")
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ProviderConfigurationError(error)

    authority = _canonical_anthropic_authority(parsed.netloc, hostname)
    path = _canonical_anthropic_path(parsed.path)
    if authority is None or path is None:
        raise ProviderConfigurationError(error)

    path = path.rstrip("/")
    if path.endswith("/v1/messages"):
        raise ProviderConfigurationError(error)
    if path.endswith("/v1"):
        path = path.removesuffix("/v1")
    return urlunsplit((parsed.scheme, authority, path, "", ""))


def _canonical_anthropic_authority(netloc: str, hostname: str) -> str | None:
    if netloc.startswith("["):
        closing_bracket = netloc.find("]")
        if closing_bracket < 0:
            return None
        literal = netloc[1:closing_bracket]
        suffix = netloc[closing_bracket + 1 :]
        if literal.casefold() != hostname.casefold() or (
            suffix and (not suffix.startswith(":") or not suffix[1:].isascii() or not suffix[1:].isdigit())
        ):
            return None

        address = literal
        zone = ""
        if "%" in literal:
            address, separator, zone = literal.partition("%25")
            if not separator or "%" in address or "%" in zone or not _ANTHROPIC_IPV6_ZONE_RE.fullmatch(zone):
                return None
        try:
            ipaddress.IPv6Address(address)
        except ValueError:
            return None
        return f"[{address}{'%25' + zone if zone else ''}]{suffix}"

    if "%" in netloc or "[" in netloc or "]" in netloc:
        return None
    host = netloc
    suffix = ""
    if ":" in netloc:
        host, port = netloc.rsplit(":", maxsplit=1)
        if ":" in host or not port.isascii() or not port.isdigit():
            return None
        suffix = f":{port}"
    if host.casefold() != hostname.casefold():
        return None

    if "." in hostname and all(character in "0123456789." for character in hostname):
        try:
            ipaddress.IPv4Address(hostname)
        except ValueError:
            return None
        return f"{host}{suffix}"

    trailing_dot = host.endswith(".")
    dns_name = host.removesuffix(".")
    if not dns_name:
        return None
    if dns_name.isascii() and "_" in dns_name:
        canonical_name = dns_name.lower()
        label_pattern = _ANTHROPIC_INTERNAL_LABEL_RE
    else:
        try:
            canonical_name = idna.encode(dns_name.lower()).decode("ascii")
        except idna.IDNAError:
            return None
        label_pattern = _ANTHROPIC_DNS_LABEL_RE
    if len(canonical_name) > 253 or not all(label_pattern.fullmatch(label) for label in canonical_name.split(".")):
        return None
    return f"{canonical_name}{'.' if trailing_dot else ''}{suffix}"


def _canonical_anthropic_path(path: str) -> str | None:
    canonical: list[str] = []
    index = 0
    while index < len(path):
        character = path[index]
        if character != "%":
            canonical.append(character)
            index += 1
            continue

        if index + 2 >= len(path) or path[index + 1] not in _HEX_DIGITS or path[index + 2] not in _HEX_DIGITS:
            return None
        octet = int(path[index + 1 : index + 3], 16)
        if octet in {0x2F, 0x5C, 0x7F} or octet < 0x20:
            return None
        if octet in _UNRESERVED_BYTES:
            canonical.append(chr(octet))
        else:
            canonical.append(f"%{octet:02X}")
        index += 3

    canonical_path = "".join(canonical)
    if "//" in canonical_path.rstrip("/"):
        return None
    decoded_octets = unquote_to_bytes(canonical_path)
    # A decoded percent is safe as data unless it opens a second escape layer.
    if any(
        decoded_octets[index] == 0x25
        and index + 2 < len(decoded_octets)
        and decoded_octets[index + 1] in _HEX_DIGIT_BYTES
        and decoded_octets[index + 2] in _HEX_DIGIT_BYTES
        for index in range(len(decoded_octets))
    ):
        return None
    decoded_path = decoded_octets.decode("utf-8", errors="replace")
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in decoded_path):
        return None
    if any(segment in {".", ".."} for segment in decoded_path.split("/")):
        return None
    return quote(canonical_path, safe=_ANTHROPIC_PATH_SAFE)


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
