#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parse an OpenAPI spec and return available operations with their parameters.
Helps the LLM understand what endpoints are available and how to call them.
"""

import json
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse


def _validate_http_url(url: str) -> str:
    """Reject local-file and custom URL schemes before fetching a spec."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Only absolute HTTP or HTTPS URLs are supported")
    return url


def fetch_spec(url: str) -> dict:
    """Fetch OpenAPI spec from URL."""
    try:
        req = urllib.request.Request(_validate_http_url(url), headers={"User-Agent": "api-caller/1.0"})
        # _validate_http_url() rejects non-HTTP schemes before this request.
        with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310
            content = response.read().decode("utf-8")
            return json.loads(content)
    except Exception as e:
        return {"error": f"Failed to fetch spec: {e!s}"}


def parse_parameters(params: list) -> list:
    """Parse parameter definitions."""
    result = []
    for p in params:
        param_info = {
            "name": p.get("name"),
            "in": p.get("in"),  # path, query, header, cookie
            "required": p.get("required", False),
            "type": p.get("schema", {}).get("type", "string"),
            "description": p.get("description", "")[:100],  # Truncate long descriptions
        }
        result.append(param_info)
    return result


def parse_request_body(body: dict) -> dict:
    """Parse request body schema."""
    if not body:
        return None

    content = body.get("content", {})
    json_content = content.get("application/json", {})
    schema = json_content.get("schema", {})

    # Extract properties if it's an object
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    if properties:
        fields = []
        for name, prop in properties.items():
            fields.append(
                {
                    "name": name,
                    "type": prop.get("type", "string"),
                    "required": name in required,
                    "description": prop.get("description", "")[:50],
                }
            )
        return {"fields": fields}

    return {"type": schema.get("type", "object")}


def parse_spec(spec: dict, filter_path: str | None = None) -> dict:
    """Parse OpenAPI spec and extract operations."""
    info = spec.get("info", {})
    base_url = ""

    # Get base URL from servers
    servers = spec.get("servers", [])
    if servers:
        base_url = servers[0].get("url", "")

    paths = spec.get("paths", {})
    operations = []

    for path, methods in paths.items():
        # Optional: filter by path substring
        if filter_path and filter_path.lower() not in path.lower():
            continue

        for method, details in methods.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                op = {
                    "method": method.upper(),
                    "path": path,
                    "full_url": f"{base_url}{path}" if base_url else path,
                    "operation_id": details.get("operationId", ""),
                    "summary": details.get("summary", "")[:100],
                    "parameters": parse_parameters(details.get("parameters", [])),
                }

                # Parse request body for POST/PUT/PATCH
                if method in ["post", "put", "patch"]:
                    body = parse_request_body(details.get("requestBody"))
                    if body:
                        op["request_body"] = body

                operations.append(op)

    return {
        "api_name": info.get("title", "Unknown API"),
        "version": info.get("version", ""),
        "base_url": base_url,
        "total_operations": len(operations),
        "operations": operations[:20],  # Limit to first 20 to avoid context bloat
    }


def main():
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    "error": "Usage: parse_openapi.py <spec_url> [filter_path]",
                    "example": "parse_openapi.py https://api.weather.gov/openapi.json points",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    spec_url = sys.argv[1]
    filter_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Fetch the spec
    spec = fetch_spec(spec_url)
    if "error" in spec:
        print(json.dumps(spec, indent=2), file=sys.stderr)
        sys.exit(1)

    # Parse and return operations
    result = parse_spec(spec, filter_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
