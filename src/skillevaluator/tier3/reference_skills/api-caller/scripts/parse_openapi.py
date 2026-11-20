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

from url_safety import join_url_path, resolve_openapi_server_url, validate_http_url


def fetch_spec(url: str) -> tuple[dict, str]:
    """Fetch OpenAPI spec from URL. Returns the spec and its final URL after redirects."""
    try:
        req = urllib.request.Request(validate_http_url(url), headers={"User-Agent": "api-caller/1.0"})
        # validate_http_url() rejects non-HTTP schemes before this request.
        with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310
            content = response.read().decode("utf-8")
            return json.loads(content), getattr(response, "url", None) or url
    except Exception as e:
        return {"error": f"Failed to fetch spec: {e!s}"}, url


def parse_parameters(params: list) -> list:
    """Parse parameter definitions."""
    result = []
    for p in params:
        schema = p.get("schema")
        if schema is None:
            schema = {}
        description = p.get("description")
        param_info = {
            "name": p.get("name"),
            "in": p.get("in"),  # path, query, header, cookie
            "required": p.get("required", False),
        }
        if isinstance(schema, bool):
            param_info["schema"] = schema
        elif isinstance(schema, dict):
            param_info["type"] = schema.get("type", "string")
        else:
            param_info["type"] = "string"
        param_info["description"] = description[:100] if isinstance(description, str) else ""
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


def parse_spec(spec: dict, filter_path: str | None = None, spec_url: str | None = None) -> dict:
    """Parse OpenAPI spec and extract operations."""
    info = spec.get("info", {})
    base_url = resolve_openapi_server_url(spec, spec_url)

    paths = spec.get("paths", {})
    operations = []

    for path, methods in paths.items():
        # Optional: filter by path substring
        if filter_path and filter_path.lower() not in path.lower():
            continue
        if not isinstance(methods, dict):
            continue

        for method, details in methods.items():
            if method in ["get", "post", "put", "delete", "patch"] and isinstance(details, dict):
                parameters = details.get("parameters", [])
                path_parameters = methods.get("parameters", [])
                if (
                    isinstance(path_parameters, list)
                    and path_parameters
                    and isinstance(parameters, list)
                    and all(
                        isinstance(parameter, dict)
                        and "$ref" not in parameter
                        and isinstance(parameter.get("name"), str)
                        and isinstance(parameter.get("in"), str)
                        for parameter in [*path_parameters, *parameters]
                    )
                ):
                    merged = {(parameter["name"], parameter["in"]): parameter for parameter in path_parameters}
                    merged.update({(parameter["name"], parameter["in"]): parameter for parameter in parameters})
                    parameters = list(merged.values())

                operation_base_url = resolve_openapi_server_url(spec, spec_url, path_item=methods, operation=details)
                op = {
                    "method": method.upper(),
                    "path": path,
                    "full_url": join_url_path(operation_base_url, path),
                    "operation_id": details.get("operationId", ""),
                    "summary": details.get("summary", "")[:100],
                    "parameters": parse_parameters(parameters),
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
    spec, final_url = fetch_spec(spec_url)
    if "error" in spec:
        print(json.dumps(spec, indent=2), file=sys.stderr)
        sys.exit(1)

    # Parse and return operations
    try:
        result = parse_spec(spec, filter_path, spec_url=final_url)
    except ValueError as exc:
        print(json.dumps({"error": f"Invalid input: {exc}"}, indent=2), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
