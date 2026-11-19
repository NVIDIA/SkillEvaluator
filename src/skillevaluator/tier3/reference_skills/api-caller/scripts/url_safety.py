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


def resolve_openapi_server_url(spec: dict, spec_url: str | None = None) -> str:
    """Return servers[0].url resolved against the spec URL, with variable defaults applied."""
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
