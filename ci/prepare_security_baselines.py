# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- trusted GitLab bootstrap, before the offline scan.
"""Resolve the approved GitLab policy commit; print only its exact object ID."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# The entrypoint uses Python isolated mode. Add only its image-owned sibling
# directory, never the candidate checkout or a consumer-provided PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from security_baselines import SHA
from select_changed_skills import SelectionError, _git


def policy_source(workspace: Path, environ: dict[str, str]) -> str:
    """Never use the candidate or merge base as MR policy authority."""
    merge_request = environ.get("CI_PIPELINE_SOURCE") == "merge_request_event"
    revision = environ.get("CI_MERGE_REQUEST_TARGET_BRANCH_SHA" if merge_request else "CI_COMMIT_SHA", "")
    if revision:
        if not SHA.fullmatch(revision):
            raise ValueError("The platform policy revision must be an exact commit ID.")
        resolved = _git(workspace, "rev-parse", revision + "^{commit}").decode().strip()
        if resolved != revision:
            raise ValueError("The platform policy revision must identify an available commit.")
        return revision
    if not merge_request:
        raise ValueError("Security baselines require the platform's current commit.")
    project = environ.get("CI_PROJECT_ID", "")
    if not project or project != environ.get("CI_MERGE_REQUEST_TARGET_PROJECT_ID"):
        raise ValueError(
            "Fork MR security baselines require an available exact target-branch commit; source origin is not trusted."
        )
    branch = environ.get("CI_MERGE_REQUEST_TARGET_BRANCH_NAME", "")
    if not branch:
        raise ValueError("Security baselines require the merge request's target branch.")
    reference = "refs/heads/" + branch
    _git(workspace, "check-ref-format", reference)
    env = {key: value for key, value in environ.items() if key in ("PATH", "LANG", "LC_ALL")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")
    # Only same-project MRs fetch origin. Preserve the full checkout's ancestry
    # for changed-skill selection: a depth-limited fetch can mark an existing
    # ancestor shallow and disconnect the recorded MR diff base. Existing
    # checkout authentication may be required; no remote URL or stderr is logged.
    process = subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "fetch",
            "--no-tags",
            "--no-filter",
            "origin",
            reference,
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=60,
    )
    if process.returncode:
        raise ValueError("Could not fetch the approved target branch for security baselines.")
    revision = _git(workspace, "rev-parse", "FETCH_HEAD^{commit}").decode().strip()
    if not SHA.fullmatch(revision):
        raise ValueError("The fetched policy revision is not an exact commit.")
    return revision


def main() -> int:
    try:
        print(policy_source(Path(os.environ["CI_PROJECT_DIR"]).resolve(strict=True), dict(os.environ)))
    except (KeyError, OSError, ValueError, SelectionError, subprocess.TimeoutExpired):
        print(
            "Could not establish the trusted target/current policy commit; security baselines were not applied.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
