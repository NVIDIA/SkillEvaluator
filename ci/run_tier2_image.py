# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- trusted standalone CI launcher with fixed diagnostics.
"""Run protected Tier 2 evaluation using the published image for the reviewed tools commit."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from model_config import tier2_credentials
    from select_changed_skills import _git, has_control_character
except ModuleNotFoundError:
    from ci.model_config import tier2_credentials
    from ci.select_changed_skills import _git, has_control_character

IMAGE = re.compile(r"ghcr\.io/nvidia/skillevaluator-tier2@sha256:[0-9a-f]{64}")
SHA = re.compile(r"[0-9a-f]{40}")
CONTEXT = frozenset(
    {
        "DIFF_BASE_SHA",
        "DIFF_HEAD_SHA",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "REPORT_COMMIT",
        "REPORT_REPOSITORY",
        "REPORT_RUN_URL",
        "SKILLEVALUATOR_TOOLS_REF",
        "SKILL_EVAL_BASE_URL",
        "SKILL_EVAL_EMBEDDING_BASE_URL",
        "SKILL_EVAL_EMBEDDING_MODEL",
        "SKILL_EVAL_EMBEDDING_PROVIDER",
        "SKILL_EVAL_ENV_MODE",
        "SKILL_EVAL_MODEL",
        "SKILL_EVAL_PATHS",
        "SKILL_EVAL_PROVIDER",
        "SKILL_EVAL_SCAN_MODE",
        "SKILL_EVAL_TIMEOUT_SECONDS",
    }
)


def _mount_paths(workspace: Path, reports: Path) -> tuple[Path, Path]:
    for path in (workspace, reports):
        if has_control_character(str(path)) or any(char in str(path) for char in ',"'):
            raise ValueError("Tier 2 image mount paths are invalid.")
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("Tier 2 image mount paths cannot use symlinks.")
    workspace, reports = workspace.resolve(strict=True), reports.resolve()
    if workspace.is_relative_to(reports) or reports.is_relative_to(workspace):
        raise ValueError("Tier 2 reports must be outside the candidate checkout.")
    git_dir = workspace / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise ValueError("Tier 2 image requires a self-contained Git checkout.")
    return workspace, reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--image", required=True)
    args = parser.parse_args(argv)
    try:
        if not IMAGE.fullmatch(args.image):
            raise ValueError("tier2-image must be a ghcr.io/nvidia/skillevaluator-tier2 SHA-256 digest.")
        workspace, reports = _mount_paths(args.workspace, args.reports)
        revision = os.environ.get("SKILLEVALUATOR_TOOLS_REF", "")
        if not SHA.fullmatch(revision):
            raise ValueError("Tier 2 image requires an immutable tools-ref commit.")
        head = os.environ.get("REPORT_COMMIT") or os.environ.get("DIFF_HEAD_SHA", "")
        diff_head = os.environ.get("DIFF_HEAD_SHA", "")
        if not SHA.fullmatch(head) or (diff_head and diff_head != head):
            raise ValueError("Tier 2 image requires the same evaluated commit for selection and reporting.")
        if _git(workspace, "rev-parse", "HEAD").decode().strip() != head:
            raise ValueError("Tier 2 image checkout does not match the evaluated commit.")
        reports.mkdir(parents=True, exist_ok=True)
        host = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "LC_CTYPE") if key in os.environ}
        host.setdefault("PATH", os.defpath)
        with tempfile.TemporaryDirectory(prefix="skillevaluator-anonymous-docker-") as config:
            # An explicit empty registry entry also disables auto-discovered credential helpers.
            (Path(config) / "config.json").write_text('{"auths":{"ghcr.io":{}}}', encoding="utf-8")
            docker = ["docker", "--config", config]
            if subprocess.run([*docker, "pull", args.image], cwd=reports.parent, env=host, check=False).returncode:
                print("Could not pull the published Tier 2 image anonymously.")
                return 1
            inspected = subprocess.run(
                [
                    *docker,
                    "image",
                    "inspect",
                    "--format",
                    '{{index .Config.Labels "org.opencontainers.image.revision"}}',
                    args.image,
                ],
                cwd=reports.parent,
                env=host,
                capture_output=True,
                text=True,
                check=False,
            )
            if inspected.returncode or inspected.stdout.splitlines() != [revision]:
                raise ValueError("Tier 2 image OCI revision must match tools-ref.")
            child = {key: os.environ[key] for key in CONTEXT if key in os.environ}
            child.update(tier2_credentials(dict(os.environ)))
            command = [
                *docker,
                "run",
                "--rm",
                "--pull=never",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec,mode=1777,size=1g",
                "--env",
                "HOME=/tmp",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--mount",
                f"type=bind,src={workspace},dst=/workspace,readonly",
                "--mount",
                f"type=bind,src={reports},dst=/reports",
            ]
            for key in sorted(child):
                command.extend(("--env", key))
            command.extend((args.image, "--workspace", "/workspace", "--reports", "/reports"))
            return subprocess.run(command, cwd=reports.parent, env={**host, **child}, check=False).returncode
    except ValueError as error:
        print(error)
    except OSError:
        print(
            "Could not start Tier 2 evaluation; the image requires a reachable local Docker daemon and writable reports."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
