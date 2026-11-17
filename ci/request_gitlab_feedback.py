#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: T201 — command-line helper diagnostics
"""Request protected-default-branch feedback with the running job's token."""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping

MAX_RESPONSE_BYTES = 1024 * 1024
MAX_ID = 2**63 - 1
SKIPPED = "Automatic MR feedback is skipped for this pipeline; the aggregate still gates Tier 1."
# GitLab refuses a trigger the job's user or token may not make with 400-404.
REFUSED_STATUSES = frozenset({400, 401, 403, 404})
REFUSED = (
    "GitLab refused the request: the pipeline user needs permission to run pipelines on the protected "
    "default branch and to pass pipeline variables."
)
NO_JOBS = "HTTP 400 can also mean the default-branch configuration added no jobs to the pipeline."
SERVER_ERROR = "GitLab did not create the feedback pipeline; retry this job, and check the GitLab API if it persists."
TRANSPORT_ERROR = "GitLab could not be reached; check the runner's access to the GitLab API, then retry this job."
INVALID_CONTEXT = (
    "This job needs a same-project merge request pipeline with GitLab's predefined CI variables and an HTTPS API URL."
)
INVALID_RESPONSE = "GitLab's response did not confirm a default-branch pipeline in this project."


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _identifier(value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]{0,18}", value) or int(value) > MAX_ID:
        raise ValueError("Invalid identifier")
    return int(value)


def _https_url(value: str):
    if not value or any(ord(character) <= 32 or ord(character) == 127 for character in value):
        raise ValueError("Invalid API origin")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or "\\" in value
    ):
        raise ValueError("Invalid API origin")
    _ = parsed.port
    return parsed


def _configuration(env: Mapping[str, str]) -> tuple[str, int, str, str, str]:
    project = _identifier(env.get("CI_PROJECT_ID", ""))
    source = env.get("CI_PIPELINE_ID", "")
    _identifier(source)
    if env.get("CI_PIPELINE_SOURCE") != "merge_request_event" or env.get("CI_MERGE_REQUEST_SOURCE_PROJECT_ID") != str(
        project
    ):
        raise ValueError("Not a same-project MR")
    api = _https_url(env.get("CI_API_V4_URL", ""))
    server = _https_url(env.get("CI_SERVER_URL", ""))
    if api.netloc != server.netloc or api.path.rstrip("/") != server.path.rstrip("/") + "/api/v4":
        raise ValueError("API origin does not match GitLab")
    ref = env.get("CI_DEFAULT_BRANCH", "")
    if (
        not ref
        or len(ref) > 1024
        or ref == "@"
        or ref.startswith("-")
        or ref.endswith(".")
        or ".." in ref
        or "@{" in ref
        or re.search(r"[~^:?*\[\\\x00-\x20\x7f]", ref)
        or any(not part or part.startswith(".") or part.endswith(".lock") for part in ref.split("/"))
    ):
        raise ValueError("Invalid default branch")
    token = env.get("CI_JOB_TOKEN", "")
    if not token or len(token) > 8192 or any(ord(character) <= 32 or ord(character) == 127 for character in token):
        raise ValueError("Invalid job token")
    endpoint = urllib.parse.urlunsplit((api.scheme, api.netloc, api.path.rstrip("/"), "", ""))
    return endpoint, project, source, ref, token


def _skipped(reason: str, hint: str) -> int:
    print(f"Feedback pipeline could not be requested ({reason}).")
    print(SKIPPED)
    print(hint)
    # The request job is allow_failure: a failure warns without blocking the MR.
    return 2


def main() -> int:
    try:
        base, project, source, ref, token = _configuration(os.environ)
        payload = urllib.parse.urlencode(
            {
                "token": token,
                "ref": ref,
                "variables[SKILL_EVAL_FEEDBACK]": "true",
                "variables[SKILL_EVAL_SOURCE_PIPELINE_ID]": source,
            }
        ).encode()
        request = urllib.request.Request(
            f"{base}/projects/{project}/trigger/pipeline",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        )
    except ValueError as error:
        return _skipped(type(error).__name__, INVALID_CONTEXT)
    try:
        # No inherited proxy can redirect this credential-bearing request.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=30) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code not in REFUSED_STATUSES:
            return _skipped(f"HTTP {error.code}", SERVER_ERROR)
        return _skipped(f"HTTP {error.code}", REFUSED + (" " + NO_JOBS if error.code == 400 else ""))
    except (OSError, ValueError, http.client.HTTPException) as error:
        return _skipped(type(error).__name__, TRANSPORT_ERROR)
    try:
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("Oversized response")
        result = json.loads(body)
        if (
            not isinstance(result, dict)
            or type(result.get("id")) is not int
            or not 0 < result["id"] <= MAX_ID
            or type(result.get("project_id")) is not int
            or result["project_id"] != project
            or result.get("ref") != ref
            or result.get("source") != "pipeline"
        ):
            raise ValueError("Invalid feedback pipeline")
    except ValueError as error:
        return _skipped(type(error).__name__, INVALID_RESPONSE)
    print(f"Feedback pipeline requested (pipeline {result['id']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
