#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generic API Caller

Call any REST API endpoint directly or using an OpenAPI spec.

Usage:
    # Direct call
    python call_api.py --url "https://api.example.com/data" --method GET

    # With headers
    python call_api.py --url "https://api.example.com/data" --headers '{"Authorization": "Bearer TOKEN"}'

    # POST with body
    python call_api.py --url "https://api.example.com/items" --method POST --data '{"name": "test"}'

    # Using OpenAPI spec
    python call_api.py --spec "https://api.weather.gov/openapi.json" --operation "point" --params '{"latitude": 37.7749}'
"""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


def _validate_http_url(url: str) -> str:
    """Reject local-file and custom URL schemes before making a request."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Only absolute HTTP or HTTPS URLs are supported")
    return url


def make_request(
    url: str, method: str = "GET", data: dict | None = None, headers: dict | None = None, timeout: int = 30
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
        url = _validate_http_url(url)
        # Prepare headers
        req_headers = {
            "User-Agent": "skills-agent/1.0 (api-caller skill)",
            "Accept": "application/json, application/geo+json, */*",
        }
        if headers:
            req_headers.update(headers)

        # Prepare body
        body = None
        if data:
            body = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        # Create request
        req = Request(url, data=body, headers=req_headers, method=method)

        # Make request
        # _validate_http_url() rejects non-HTTP schemes before this request.
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
            url_path = url_path.replace(placeholder, str(value))
        else:
            query_params[key] = value

    url = base_url + url_path

    if query_params:
        url += "?" + urlencode(query_params)

    return url


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
    data = json.loads(args.data) if args.data else None
    headers = json.loads(args.headers) if args.headers else None
    params = json.loads(args.params) if args.params else {}

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
            print(json.dumps({"success": False, "error": f"Failed to fetch OpenAPI spec from {args.spec}"}, indent=2))
            sys.exit(1)

        path, spec_method, _operation = find_operation(spec, args.operation)
        if not path:
            # List available operations
            available = []
            for p, methods in spec.get("paths", {}).items():
                for m, details in methods.items():
                    if "operationId" in details:
                        available.append(f"{m.upper()} {p} ({details['operationId']})")

            print(
                json.dumps(
                    {
                        "success": False,
                        "error": f"Operation '{args.operation}' not found in spec",
                        "available_operations": available[:20],  # Show first 20
                    },
                    indent=2,
                )
            )
            sys.exit(1)

        url = build_url_from_spec(spec, path, params)
        method = spec_method

        print("# Resolved from OpenAPI spec:")
        print(f"# {method} {url}")
        print()

    if not url:
        print(
            json.dumps({"success": False, "error": "No URL provided. Use --url or --spec with --operation"}, indent=2)
        )
        sys.exit(1)

    # Make the request
    result = make_request(url, method, data, headers, args.timeout)

    # Pretty print result
    print(json.dumps(result, indent=2, default=str))

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
