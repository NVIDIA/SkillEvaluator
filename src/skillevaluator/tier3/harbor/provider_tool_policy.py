# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider capability policy for server-hosted agent tools."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from skillevaluator.provider_config import ProviderConfig

SERVER_TOOL_POLICY_PROVIDER_COMPATIBLE_V1 = "provider_compatible_v1"
SERVER_TOOL_POLICIES = frozenset({SERVER_TOOL_POLICY_PROVIDER_COMPATIBLE_V1})

# The requested policy is resolved before Harbor starts. Agent wrappers only
# consume one of these decisions; skills cannot select or override them.
CLAUDE_SERVER_TOOL_POLICY_ENV = "SKILLEVALUATOR_CLAUDE_SERVER_TOOL_POLICY"
CLAUDE_SERVER_TOOL_POLICY_NATIVE_V1 = "native_server_tools_v1"
CLAUDE_SERVER_TOOL_POLICY_NATIVE_NO_EXPERIMENTAL_BETAS_V1 = "native_server_tools_no_experimental_betas_v1"
CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1 = "disable_web_search_fetch_v1"
CLAUDE_SERVER_TOOL_POLICY_BRIDGE_V1 = "bridge_normalized_server_tools_v1"
CLAUDE_SERVER_TOOL_POLICY_RESOLUTIONS = frozenset(
    {
        CLAUDE_SERVER_TOOL_POLICY_NATIVE_V1,
        CLAUDE_SERVER_TOOL_POLICY_NATIVE_NO_EXPERIMENTAL_BETAS_V1,
        CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1,
        CLAUDE_SERVER_TOOL_POLICY_BRIDGE_V1,
    }
)

_NVIDIA_INFERENCE_GATEWAY_HOST = "inference-api.nvidia.com"
_NVIDIA_INFERENCE_AZURE_ANTHROPIC_PREFIX = "azure/anthropic/"
_AWS_BEDROCK_ANTHROPIC_PREFIX = "aws/anthropic/bedrock-"


def validate_server_tool_policy(policy: str | None) -> str | None:
    """Return a normalized public policy name or reject an unknown contract."""
    if policy is None:
        return None
    normalized = str(policy).strip()
    if normalized not in SERVER_TOOL_POLICIES:
        choices = ", ".join(sorted(SERVER_TOOL_POLICIES))
        raise ValueError(f"server_tool_policy must be one of: {choices}")
    return normalized


def resolve_claude_server_tool_policy(
    provider: ProviderConfig,
    requested_policy: str | None,
) -> str | None:
    """Resolve one immutable Claude Code tool decision for a provider route.

    Anthropic's native API supports Claude Code's server-hosted WebSearch and
    WebFetch tools. Compatibility-gateway support is resolved from the actual
    gateway and model profile: NVIDIA inference-api's Azure Anthropic route is
    verified to support both tools, while its AWS Bedrock route does not.
    Unknown gateway/model combinations fail closed. NVIDIA Build uses
    SkillEvaluator's compatibility bridge, which already normalizes unsupported
    server tools.
    """
    policy = validate_server_tool_policy(requested_policy)
    if policy is None:
        return None

    if provider.provider == "nv_build":
        return CLAUDE_SERVER_TOOL_POLICY_BRIDGE_V1

    model = str(provider.model or "").strip().casefold()
    if model.startswith(_AWS_BEDROCK_ANTHROPIC_PREFIX):
        return CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1

    base_url = str(provider.base_url or "").strip()
    if provider.provider == "anthropic" and _is_native_anthropic_url(base_url):
        return CLAUDE_SERVER_TOOL_POLICY_NATIVE_V1
    if provider.provider == "anthropic" and _supports_gateway_anthropic_server_tools(provider):
        # Claude Code 2.1.237 otherwise adds the gateway-incompatible
        # clear_thinking context-management beta. The wrapper disables only
        # experimental betas; adaptive thinking and native Web tools remain.
        return CLAUDE_SERVER_TOOL_POLICY_NATIVE_NO_EXPERIMENTAL_BETAS_V1

    return CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1


def _is_native_anthropic_url(base_url: str) -> bool:
    if not base_url:
        return True
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "https"
        and parsed.hostname is not None
        and parsed.hostname.casefold() == "api.anthropic.com"
        and port is None
    )


def _supports_gateway_anthropic_server_tools(provider: ProviderConfig) -> bool:
    """Return whether a gateway/model profile has verified server-tool support."""
    try:
        parsed = urlsplit(str(provider.base_url or "").strip())
        port = parsed.port
    except ValueError:
        return False
    hostname = parsed.hostname
    if (
        parsed.scheme.casefold() != "https"
        or hostname is None
        or hostname.rstrip(".").casefold() != _NVIDIA_INFERENCE_GATEWAY_HOST
        or port not in {None, 443}
    ):
        return False

    model = str(provider.model or "").strip().casefold()
    return model.startswith(_NVIDIA_INFERENCE_AZURE_ANTHROPIC_PREFIX)
