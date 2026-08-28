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
CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1 = "disable_web_search_fetch_v1"
CLAUDE_SERVER_TOOL_POLICY_BRIDGE_V1 = "bridge_normalized_server_tools_v1"
CLAUDE_SERVER_TOOL_POLICY_RESOLUTIONS = frozenset(
    {
        CLAUDE_SERVER_TOOL_POLICY_NATIVE_V1,
        CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1,
        CLAUDE_SERVER_TOOL_POLICY_BRIDGE_V1,
    }
)


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
    WebFetch tools. Compatibility gateways are not assumed to support those
    Anthropic-specific request blocks. NVIDIA Build uses SkillEvaluator's
    compatibility bridge, which already normalizes unsupported server tools.
    """
    policy = validate_server_tool_policy(requested_policy)
    if policy is None:
        return None

    if provider.provider == "nv_build":
        return CLAUDE_SERVER_TOOL_POLICY_BRIDGE_V1

    base_url = str(provider.base_url or "").strip()
    if provider.provider == "anthropic" and _is_native_anthropic_url(base_url):
        return CLAUDE_SERVER_TOOL_POLICY_NATIVE_V1

    return CLAUDE_SERVER_TOOL_POLICY_DISABLE_WEB_V1


def _is_native_anthropic_url(base_url: str) -> bool:
    if not base_url:
        return True
    parsed = urlsplit(base_url)
    return (
        parsed.scheme.casefold() == "https"
        and parsed.hostname is not None
        and parsed.hostname.casefold() == "api.anthropic.com"
        and parsed.port is None
    )
