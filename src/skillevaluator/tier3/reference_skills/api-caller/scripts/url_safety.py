# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL checks shared by the api-caller scripts."""

import re
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def validate_http_url(url: str) -> str:
    """Reject local-file and custom URL schemes before making a request."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Only absolute HTTP or HTTPS URLs are supported")
    return url


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 section 5.2.4 dot-segment removal that keeps empty segments."""
    output: list[str] = []
    segments = path.split("/")
    for segment in segments:
        if segment == ".":
            continue
        if segment == "..":
            if len(output) > 1:
                output.pop()
            continue
        output.append(segment)
    if segments[-1] in {".", ".."}:
        output.append("")
    return "/".join(output)


def _resolve_reference(base: str, reference: str) -> str:
    """Resolve a URL reference against base per RFC 3986.

    urljoin drops empty segments when it merges a relative path, so 'api//v2'
    would lose its '//'. Relative-path references are merged here instead;
    everything else still goes through urljoin, which keeps them intact.
    """
    ref = urlsplit(reference)
    if ref.scheme or ref.netloc or not ref.path or ref.path.startswith("/"):
        return urljoin(base, reference)
    base_parts = urlsplit(base)
    if base_parts.netloc and not base_parts.path:
        merged = "/" + ref.path
    else:
        merged = base_parts.path[: base_parts.path.rfind("/") + 1] + ref.path
    return urlunsplit((base_parts.scheme, base_parts.netloc, _remove_dot_segments(merged), ref.query, ref.fragment))


def _swagger2_base_url(spec: dict, spec_url: str | None, *, operation: dict | None = None) -> str:
    """Build a Swagger 2.0 base URL from schemes, host and basePath, defaulting to the spec URL."""
    origin = urlparse(spec_url) if spec_url else None
    raw_schemes = (
        operation["schemes"]
        if isinstance(operation, dict) and operation.get("schemes") is not None
        else spec.get("schemes")
    )
    schemes = [] if raw_schemes is None else raw_schemes
    if not isinstance(schemes, list):
        raise ValueError("Swagger 2.0 schemes must be a list")
    http_schemes = [scheme for scheme in schemes if scheme in {"http", "https"}]
    if schemes and not http_schemes:
        raise ValueError("Swagger 2.0 spec declares no HTTP or HTTPS scheme")
    if origin and origin.scheme in http_schemes:
        scheme = origin.scheme
    elif "https" in http_schemes:
        scheme = "https"
    elif http_schemes:
        scheme = http_schemes[0]
    else:
        scheme = origin.scheme if origin else ""

    host = spec.get("host") or (origin.netloc if origin else "")
    if not isinstance(host, str):
        raise ValueError("Swagger 2.0 host must be a string")
    base_path = spec.get("basePath") or "/"
    if not isinstance(base_path, str):
        raise ValueError("Swagger 2.0 basePath must be a string")
    if not base_path.startswith("/"):
        base_path = "/" + base_path

    if not scheme or not host:
        return base_path
    return f"{scheme}://{host}{base_path}"


def _select_openapi_servers(spec: dict, path_item: dict | None, operation: dict | None) -> list:
    """Return the most specific non-empty OpenAPI 3 servers list."""
    for scope in (operation, path_item, spec):
        if not isinstance(scope, dict):
            continue
        servers = scope.get("servers")
        if servers is None:
            continue
        if not isinstance(servers, list):
            raise ValueError("servers must be a list")
        if servers:
            return servers
    return []


def resolve_openapi_server_url(
    spec: dict,
    spec_url: str | None = None,
    *,
    path_item: dict | None = None,
    operation: dict | None = None,
) -> str:
    """Return the API base URL: servers[0].url (OpenAPI 3) or schemes/host/basePath (Swagger 2.0).

    OpenAPI 3 servers are checked on the operation, then the path item, then the
    document, falling back to the spec URL origin; Swagger 2.0 operation schemes
    override document schemes. Relative URLs are resolved against the spec URL,
    which should be the final URL after redirects. Server variable defaults are
    applied.
    """
    if str(spec.get("swagger", "")).startswith("2"):
        return _swagger2_base_url(spec, spec_url, operation=operation)
    servers = _select_openapi_servers(spec, path_item, operation)
    if servers:
        server = servers[0] if isinstance(servers[0], dict) else {}
        server_url = server.get("url")
        if not isinstance(server_url, str):
            raise ValueError("server URL must be a string")
        if not server_url.strip():
            # An empty relative reference resolves to the spec document itself, and
            # appending paths to that is wrong. Use the document's directory instead.
            server_url = "." if spec_url else "/"
        variables = server.get("variables") or {}
    else:
        server_url = "/"
        variables = {}

    def replace_variable(match: re.Match[str]) -> str:
        name = match.group(1)
        variable = variables.get(name) if isinstance(variables, dict) else None
        if not isinstance(variable, dict) or "default" not in variable or variable["default"] is None:
            raise ValueError(f"server variable '{name}' has no default")
        return str(variable["default"])

    server_url = _PLACEHOLDER.sub(replace_variable, server_url)
    resolved = _resolve_reference(spec_url, server_url) if spec_url else server_url
    if urlparse(resolved).scheme:
        validate_http_url(resolved)
    return resolved


def join_url_path(base: str, path: str, *, query: str = "") -> str:
    """Append an operation path and encoded query string to a server URL."""
    parts = urlsplit(base)
    joined_path = parts.path
    if path:
        base_ends_with_slash = joined_path.endswith("/")
        operation_starts_with_slash = path.startswith("/")
        if base_ends_with_slash and operation_starts_with_slash:
            joined_path = joined_path[:-1]
        elif not base_ends_with_slash and not operation_starts_with_slash:
            joined_path += "/"
        joined_path += path
    combined_query = "&".join(filter(None, (parts.query, query)))
    return urlunsplit((parts.scheme, parts.netloc, joined_path, combined_query, parts.fragment))
