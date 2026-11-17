#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: T201 — command-line helper diagnostics
"""Fail closed before a provider-backed GitLab evaluation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


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
        _default_branch_candidate(env)
        and isinstance(current, dict)
        and current.get("name") == env["CI_DEFAULT_BRANCH"]
        and current.get("protected") is True
        and isinstance(current.get("commit"), dict)
        and current["commit"].get("id") == env["CI_COMMIT_SHA"]
    )


def _default_branch_candidate(env: Mapping[str, str]) -> bool:
    return (
        bool(env.get("CI_COMMIT_SHA"))
        and bool(env.get("CI_DEFAULT_BRANCH"))
        and env.get("CI_COMMIT_BRANCH") == env["CI_DEFAULT_BRANCH"]
        and env.get("CI_COMMIT_REF_PROTECTED") == "true"
        and env.get("CI_PIPELINE_SOURCE") in {"push", "web"}
    )


def main() -> int:
    env = os.environ
    current = None
    is_mr = env.get("CI_PIPELINE_SOURCE") == "merge_request_event"
    if is_mr or _default_branch_candidate(env):
        kind = "merge-request" if is_mr else "default-branch"
        base = env.get("CI_API_V4_URL", "")
        try:
            parsed = urllib.parse.urlsplit(base)
        except ValueError:
            print("GitLab admission requires an HTTPS API endpoint.")
            return 2
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.path.rstrip("/").endswith("/api/v4")
        ):
            print("GitLab admission requires an HTTPS API endpoint.")
            return 2
        project = env.get("CI_PROJECT_ID", "")
        iid = env.get("CI_MERGE_REQUEST_IID", "")
        if not project.isdigit() or (is_mr and not iid.isdigit()):
            return 2
        resource = (
            f"merge_requests/{iid}"
            if is_mr
            else "repository/branches/" + urllib.parse.quote(env["CI_DEFAULT_BRANCH"], safe="")
        )
        # GitLab added job-token authentication for the MR endpoint in 18.4.
        # Older servers use a separate project Reporter/read_api token, never
        # the publisher credential or the local onboarding user's PAT.
        read_token = env.get("SKILLEVALUATOR_GITLAB_READ_TOKEN", "").strip()
        headers = {"PRIVATE-TOKEN": read_token} if read_token else {"JOB-TOKEN": env.get("CI_JOB_TOKEN", "")}
        request = urllib.request.Request(f"{base.rstrip('/')}/projects/{project}/{resource}", headers=headers)
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=30) as response:
                payload = response.read(1024 * 1024 + 1)
                if len(payload) > 1024 * 1024:
                    raise ValueError("Oversized admission response")
                current = json.loads(payload)
                if not isinstance(current, dict):
                    raise ValueError("Invalid admission response")
        except urllib.error.HTTPError as error:
            print(f"Current {kind} head could not be verified (HTTP {error.code}).")
            if is_mr or error.code in {401, 403, 404}:
                print(
                    "Configure or verify SKILLEVALUATOR_GITLAB_READ_TOKEN with project Reporter/read_api access. "
                    "This separate read token is required for MR admission before GitLab 18.4 and whenever "
                    "CI_JOB_TOKEN cannot read the current-head endpoint."
                )
            return 2
        except urllib.error.URLError as error:
            print(f"Current {kind} head could not be verified (transport {type(error.reason).__name__}).")
            return 2
        except (OSError, ValueError) as error:
            print(f"Current {kind} head could not be verified ({type(error).__name__}).")
            return 2
    if not eligible(env, current):
        print("Provider jobs require the current same-project MR head or a protected default branch.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
