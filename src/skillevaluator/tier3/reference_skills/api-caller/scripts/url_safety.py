# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL checks shared by the api-caller scripts."""

import re
from urllib.parse import urljoin, urlparse

_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def validate_http_url(url: str) -> str:
    """Reject local-file and custom URL schemes before making a request."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Only absolute HTTP or HTTPS URLs are supported")
    return url


def _swagger2_base_url(spec: dict, spec_url: str | None) -> str:
    """Build a Swagger 2.0 base URL from schemes, host and basePath, defaulting to the spec URL."""
    origin = urlparse(spec_url) if spec_url else None
    schemes = spec.get("schemes") or []
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


def resolve_openapi_server_url(spec: dict, spec_url: str | None = None) -> str:
    """Return the API base URL: servers[0].url (OpenAPI 3) or schemes/host/basePath (Swagger 2.0).

    Relative URLs are resolved against the spec URL, which should be the final URL
    after redirects. Server variable defaults are applied.
    """
    if str(spec.get("swagger", "")).startswith("2"):
        return _swagger2_base_url(spec, spec_url)
    servers = spec.get("servers") or []
    if servers:
        server = servers[0] if isinstance(servers[0], dict) else {}
        server_url = server.get("url")
        if not isinstance(server_url, str):
            raise ValueError("server URL must be a string")
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
    if spec_url:
        return urljoin(spec_url, server_url)
    return server_url


def join_url_path(base: str, path: str) -> str:
    """Join a server base URL and an operation path without '//' or dropped segments."""
    if not path:
        return base
    if base.endswith("/") and path.startswith("/"):
        return base[:-1] + path
    if not base.endswith("/") and not path.startswith("/"):
        return f"{base}/{path}"
    return base + path
