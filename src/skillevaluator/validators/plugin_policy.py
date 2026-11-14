# SPDX-License-Identifier: Apache-2.0
"""Bounded operator allowlists shared by component checks and runtime proof."""

from __future__ import annotations

import ipaddress
import unicodedata
from urllib.parse import urlsplit


def normalize_plugin_allowlist(value: object, key: str, max_chars: int) -> tuple[str, ...]:
    """Validate entries without DNS and preserve the existing host matcher rules."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or len(value) > 64:
        raise ValueError(f"{key} must be a list of at most 64 strings")
    if not value:
        return ()
    # Default policies stay cheap for ordinary skills and base installations.
    from skillevaluator.plugin_component_risk import hook_url_entry_problem
    from skillevaluator.validators.mcp_static import HostAllowlist, classify_endpoint_host
    from skillevaluator.validators.url_policy import url_ambiguities

    normalized = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip() or len(raw) > max_chars:
            raise ValueError(f"{key} entries must be nonempty strings of at most {max_chars} characters")
        if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in raw):
            raise ValueError(f"{key} entries cannot contain control or invisible characters")
        entry = raw.strip()
        if "://" in entry:
            if (
                key != "hooks.allowed_urls"
                or hook_url_entry_problem(entry) is not None
                or url_ambiguities(entry, percent_in_host=True)
            ):
                raise ValueError(f"{key} contains an invalid URL prefix")
            parsed = urlsplit(entry)
            if parsed.scheme.lower() not in {"http", "https"}:
                raise ValueError(f"{key} URL prefixes must use HTTP or HTTPS")
            host = parsed.hostname
        else:
            if any(char.isspace() for char in entry) or any(char in entry for char in "@?#\\"):
                raise ValueError(f"{key} contains an invalid host pattern")
            if "/" in entry or ":" in entry:
                try:
                    ipaddress.ip_network(
                        entry[1:-1] if entry.startswith("[") and entry.endswith("]") else entry, strict=False
                    )
                except ValueError as exc:
                    raise ValueError(f"{key} contains an invalid IP address or network") from exc
            if "*" in entry and (not entry.startswith("*.") or "*" in entry[2:]):
                raise ValueError(f"{key} contains an invalid host wildcard")
            hosts = HostAllowlist.from_entries((entry,))
            if not (hosts.names or hosts.suffixes or hosts.networks):
                raise ValueError(f"{key} contains an invalid host pattern")
            host = entry
        endpoint = classify_endpoint_host(host) if host else None
        if endpoint is not None and endpoint.kind == "metadata":
            raise ValueError(f"{key} cannot allow cloud metadata endpoints")
        normalized.append(entry)
    return tuple(dict.fromkeys(normalized))


def plugin_policy_block(data: dict, name: str, known: set[str]) -> dict:
    """Reject misspelled controls instead of silently widening operator intent."""
    block = data.get(name)
    if block is None:
        return {}
    if not isinstance(block, dict) or not all(isinstance(key, str) for key in block) or set(block) - known:
        raise ValueError(f"{name} policy must be a mapping with only {', '.join(sorted(known))}")
    return block


def overlay_sets(data: dict, block: str, key: str) -> bool:
    value = data.get(block)
    return isinstance(value, dict) and key in value
