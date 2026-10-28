# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- standalone helper shared by the trusted CI entry points.

"""GitHub REST client and identity patterns shared by the trusted CI helpers.

Standard library only. The PR comment job runs this module with the runner's
system Python from a sparse checkout that holds just it and comment_report.py.
"""

from __future__ import annotations

import json
import re
import urllib.request

# GitHub commit, tree and blob identities are SHA-1. SHA-256 repositories are not supported.
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
GHCR_DIGEST = re.compile(r"ghcr\.io/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}")
MAX_API_BYTES = 16 * 1024 * 1024
TIMEOUT_SECONDS = 60


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _request, _file, _code, _message, _headers, _url):
        return None


def bounded_read(
    response,
    limit: int,
    error: type[Exception] = ValueError,
    message: str = "The GitHub API response exceeds its size limit.",
) -> bytes:
    """Read at most ``limit`` bytes, rejecting a larger declared or actual body."""
    declared = response.headers.get("Content-Length")
    if declared and (not declared.isdigit() or int(declared) > limit):
        raise error(message)
    payload = response.read(limit + 1)
    if len(payload) > limit:
        raise error(message)
    return payload


def strict_json(raw: bytes, error: type[Exception] = ValueError, message: str = "A JSON document is invalid."):
    """Parse UTF-8 JSON and reject duplicate keys and NaN or Infinity."""

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    def invalid_constant(_value):
        raise ValueError("non-finite number")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise error(message) from exc


class GitHubAPI:
    """Token-authenticated REST client limited to one repository's endpoints.

    Subclasses set ``error`` to their own fixed-diagnostic exception type.
    """

    error: type[Exception] = ValueError
    json_error = "The GitHub API response is invalid JSON."
    user_agent = "SkillEvaluator-CI"

    def __init__(self, repository: str, token: str):
        self.repository, self.token = repository, token
        self.opener = urllib.request.build_opener(_NoRedirect())

    def _request(self, method: str, path: str, payload=None):
        base = f"/repos/{self.repository}"
        # A commit comparison is the one endpoint whose path needs "..": allow exactly two full SHAs.
        comparison = re.fullmatch(re.escape(base) + r"/compare/[0-9a-f]{40}\.\.\.[0-9a-f]{40}", path)
        if not (path == base or path.startswith(base + "/")) or (".." in path and not comparison) or "\\" in path:
            raise self.error("Only this repository's GitHub API endpoints are supported.")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": self.user_agent,
        }
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"https://api.github.com{path}", data=data, headers=headers, method=method)
        return self.opener.open(request, timeout=TIMEOUT_SECONDS)

    def _call(self, method: str, path: str, payload=None):
        with self._request(method, path, payload) as response:
            return strict_json(bounded_read(response, MAX_API_BYTES, self.error), self.error, self.json_error)

    def get(self, path: str):
        return self._call("GET", path)

    def post(self, path: str, payload: dict):
        return self._call("POST", path, payload)

    def patch(self, path: str, payload: dict):
        return self._call("PATCH", path, payload)
