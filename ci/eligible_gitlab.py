#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: T201 — command-line helper diagnostics
"""Fail closed before a provider-backed GitLab MR evaluation."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Mapping


def eligible(env: Mapping[str, str], current: dict | None) -> bool:
    if env.get("CI_PIPELINE_SOURCE") == "merge_request_event":
        if current is None:
            return False
        project = env.get("CI_PROJECT_ID")
        return (
            env.get("CI_MERGE_REQUEST_SOURCE_PROJECT_ID") == project
            and str(current.get("source_project_id")) == project
            and str(current.get("target_project_id")) == project
            and current.get("state") == "opened"
            and current.get("sha") == env.get("CI_COMMIT_SHA")
        )
    return (
        bool(env.get("CI_DEFAULT_BRANCH"))
        and env.get("CI_COMMIT_BRANCH") == env["CI_DEFAULT_BRANCH"]
        and env.get("CI_COMMIT_REF_PROTECTED") == "true"
        and env.get("CI_PIPELINE_SOURCE") in {"push", "web"}
    )


def main() -> int:
    env = os.environ
    current = None
    if env.get("CI_PIPELINE_SOURCE") == "merge_request_event":
        base = env.get("CI_API_V4_URL", "")
        parsed = urllib.parse.urlsplit(base)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            print("GitLab admission requires an HTTPS API endpoint.")
            return 2
        project = env.get("CI_PROJECT_ID", "")
        iid = env.get("CI_MERGE_REQUEST_IID", "")
        if not project.isdigit() or not iid.isdigit():
            return 2
        request = urllib.request.Request(
            f"{base}/projects/{project}/merge_requests/{iid}", headers={"JOB-TOKEN": env.get("CI_JOB_TOKEN", "")}
        )
        # The API call reads current MR state using only the short-lived job token.
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                current = json.loads(response.read(1024 * 1024))
        except (OSError, ValueError):
            print("Current merge-request head could not be verified.")
            return 2
    if not eligible(env, current):
        print("Provider jobs require the current same-project MR head or a protected default branch.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
