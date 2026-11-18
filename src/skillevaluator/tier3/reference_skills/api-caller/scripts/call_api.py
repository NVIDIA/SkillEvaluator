#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call a REST endpoint directly or through an OpenAPI operation. See SKILL.md for usage."""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from url_safety import validate_http_url

_UNSET = object()


def make_request(
    url: str, method: str = "GET", data: object = _UNSET, headers: dict | None = None, timeout: int = 30
) -> dict:
    """Make an HTTP request and return the response."""

    result = {
        "success": False,
        "status_code": None,
        "headers": {},
        "data": None,
        "error": None,
        "url": url,
        "method": method,
    }

    try:
        url = validate_http_url(url)
        # Prepare headers
        req_headers = {
            "User-Agent": "skills-agent/1.0 (api-caller skill)",
            "Accept": "application/json, application/geo+json, */*",
        }
        if headers:
            req_headers.update(headers)

        # Prepare body
        body = None
        if data is not _UNSET:
            body = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        # Create request
        req = Request(url, data=body, headers=req_headers, method=method)

        # Make request
        # validate_http_url() rejects non-HTTP schemes before this request.
        with urlopen(req, timeout=timeout) as response:  # nosec B310
            result["status_code"] = response.status
            result["headers"] = dict(response.headers)

            # Read and parse response
            content = response.read().decode("utf-8")

            # Try to parse as JSON
            try:
                result["data"] = json.loads(content)
            except json.JSONDecodeError:
                # Return raw text if not JSON
                result["data"] = content

            result["success"] = True

    except HTTPError as e:
        result["status_code"] = e.code
        result["error"] = f"HTTP {e.code}: {e.reason}"
        try:
            error_content = e.read().decode("utf-8")
            try:
                result["data"] = json.loads(error_content)
            except json.JSONDecodeError:
                result["data"] = error_content
        except Exception:
            pass

    except URLError as e:
        result["error"] = f"URL Error: {e.reason!s}"

    except TimeoutError:
        result["error"] = f"Request timed out after {timeout} seconds"

    except Exception as e:
        result["error"] = f"Error: {e!s}"

    return result


def fetch_openapi_spec(spec_url: str) -> dict:
    """Fetch and parse an OpenAPI spec."""
    result = make_request(spec_url)
    if result["success"] and isinstance(result["data"], dict):
        return result["data"]
    return None


def find_operation(spec: dict, operation_id: str) -> tuple:
    """Find an operation in the OpenAPI spec by operation ID."""
    paths = spec.get("paths", {})

    for path, methods in paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete", "patch") and details.get("operationId") == operation_id:
                return path, method.upper(), details

    return None, None, None


def build_url_from_spec(spec: dict, path: str, params: dict) -> str:
    """Build a URL from OpenAPI spec and parameters."""
    servers = spec.get("servers", [])
    base_url = servers[0]["url"] if servers else ""

    # Replace path parameters
    url_path = path
    query_params = {}

    for key, value in params.items():
        placeholder = "{" + key + "}"
        if placeholder in url_path:
            path_value = str(value)
            if path_value in {".", ".."}:
                raise ValueError(f"Path parameter '{key}' must not be '.' or '..'")
            url_path = url_path.replace(placeholder, quote(path_value, safe=""))
        else:
            query_params[key] = value

    url = base_url + url_path

    if query_params:
        url += "?" + urlencode(query_params)

    return url


def _parse_json_argument(value: str | None, label: str, expected_type: type | None = None) -> object | None:
    """Parse and type-check a JSON command-line argument."""
    if value is None:
        return None

    parsed = json.loads(value)
    if expected_type is not None and not isinstance(parsed, expected_type):
        raise ValueError(f"{label} must be a JSON {expected_type.__name__}")
    return parsed


def _print_json(value: dict, *, file=None) -> None:
    """Print a stable, human-readable JSON object."""
    print(json.dumps(value, indent=2, default=str), file=file)


def main():
    parser = argparse.ArgumentParser(description="Generic API Caller")
    parser.add_argument("--url", help="Direct API endpoint URL")
    parser.add_argument(
        "--method", default="GET", choices=["GET", "POST", "PUT", "DELETE", "PATCH"], help="HTTP method (default: GET)"
    )
    parser.add_argument("--data", help="Request body as JSON string")
    parser.add_argument("--headers", help="Custom headers as JSON string")
    parser.add_argument("--spec", help="OpenAPI spec URL")
    parser.add_argument("--operation", help="Operation ID from OpenAPI spec")
    parser.add_argument("--params", help="Path/query parameters as JSON")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")

    args = parser.parse_args()

    # Parse JSON arguments
    try:
        data = _UNSET if args.data is None else _parse_json_argument(args.data, "--data")
        headers = _parse_json_argument(args.headers, "--headers", dict)
        params = _parse_json_argument(args.params, "--params", dict) or {}
        if headers and not all(isinstance(value, str) for value in headers.values()):
            raise ValueError("--headers must map header names to string values")
    except (json.JSONDecodeError, ValueError) as exc:
        _print_json({"success": False, "error": f"Invalid input: {exc}"}, file=sys.stderr)
        sys.exit(1)

    # Determine URL and method
    url = args.url
    method = args.method

    # If using direct URL with params, append query parameters
    if url and params and not args.spec:
        # Check if URL already has query params
        if "?" in url:
            url += "&" + urlencode(params)
        else:
            url += "?" + urlencode(params)

    # If using OpenAPI spec
    if args.spec and args.operation:
        spec = fetch_openapi_spec(args.spec)
        if not spec:
            _print_json({"success": False, "error": f"Failed to fetch OpenAPI spec from {args.spec}"}, file=sys.stderr)
            sys.exit(1)

        path, spec_method, _operation = find_operation(spec, args.operation)
        if not path:
            # List available operations
            available = []
            for p, methods in spec.get("paths", {}).items():
                for m, details in methods.items():
                    if "operationId" in details:
                        available.append(f"{m.upper()} {p} ({details['operationId']})")

            _print_json(
                {
                    "success": False,
                    "error": f"Operation '{args.operation}' not found in spec",
                    "available_operations": available[:20],  # Show first 20
                },
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            url = build_url_from_spec(spec, path, params)
        except ValueError as exc:
            _print_json({"success": False, "error": f"Invalid input: {exc}"}, file=sys.stderr)
            sys.exit(1)
        method = spec_method

        print(f"Resolved OpenAPI operation: {method} {url}", file=sys.stderr)

    if not url:
        _print_json(
            {"success": False, "error": "No URL provided. Use --url or --spec with --operation"}, file=sys.stderr
        )
        sys.exit(1)

    # Make the request
    result = make_request(url, method, data, headers, args.timeout)

    # Pretty print successful results to stdout and handled failures to stderr.
    _print_json(result, file=None if result["success"] else sys.stderr)

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
