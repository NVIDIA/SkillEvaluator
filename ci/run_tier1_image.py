# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- trusted standalone CI launcher with fixed diagnostics.
"""Launch authorized LLM Tier 1 scans in the immutable complete scanner image.

Admission and protected-environment authorization belong to the calling job.
This helper checks the exact checkout before exposing the selected model key.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Isolated Python omits the script directory. Add only the pinned tools
# checkout's helpers, never the consumer working directory or PYTHONPATH.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from model_config import PROVIDER_KEYS, model_environment
    from select_changed_skills import _git, changed_paths, pull_request, setup_changed, skill_root_changed, skill_roots
except ModuleNotFoundError:
    from ci.model_config import PROVIDER_KEYS, model_environment
    from ci.select_changed_skills import (
        _git,
        changed_paths,
        pull_request,
        setup_changed,
        skill_root_changed,
        skill_roots,
    )

IMAGE = re.compile(r"ghcr\.io/nvidia/skillevaluator-tier1@sha256:[0-9a-f]{64}")
SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
CONTEXT = frozenset(
    {
        "DIFF_BASE_SHA",
        "DIFF_HEAD_SHA",
        "REPORT_COMMIT",
        "REPORT_RUN_URL",
        "REPORT_URL",
        "REPORT_BASE_URL",
        "CALLER_WORKFLOW_SHA",
        "GITHUB_ACTIONS",
    }
)
# Transport settings are only for Docker itself; no ambient credentials or
# user Docker configuration are inherited. Model keys never reach the pull.
HOST_ENV = frozenset({"PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT"})


class NoSkillChanges(Exception):
    """An automatic PR/MR scan changed no file under a configured skill root."""


def _paths_safe(workspace: Path, reports: Path) -> tuple[Path, Path]:
    for path in (workspace, reports):
        if any(ord(c) < 32 or ord(c) == 127 or c == "," for c in str(path)):
            raise ValueError
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError
    workspace = workspace.resolve(strict=True)
    reports = reports.resolve()
    if not workspace.is_dir() or reports.is_relative_to(workspace) or workspace.is_relative_to(reports):
        raise ValueError
    if reports.exists() and not reports.is_dir():
        raise ValueError
    return workspace, reports


def _environment(workspace: Path, image: str, source: dict[str, str]) -> dict[str, str]:
    head = source.get("REPORT_COMMIT") or source.get("DIFF_HEAD_SHA", "")
    if not SHA.fullmatch(head):
        raise ValueError
    if any(value and value != head for value in (source.get("REPORT_COMMIT"), source.get("DIFF_HEAD_SHA"))):
        raise ValueError
    top = _git(workspace, "rev-parse", "--show-toplevel").decode().strip()
    if Path(top).resolve(strict=True) != workspace or _git(workspace, "rev-parse", "HEAD").decode().strip() != head:
        raise ValueError
    roots = source.get("SKILL_EVAL_PATHS", "skills")
    configured = skill_roots(workspace, roots)
    if _git(
        workspace, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching", "--", *configured
    ):
        raise ValueError
    mode = source.get("SKILL_EVAL_SCAN_MODE", "auto")
    if mode not in {"auto", "all"}:
        raise ValueError
    mode = "changed" if mode == "auto" and pull_request(source) else "all"
    if mode == "changed":
        changed = changed_paths(workspace, source, head)
        # The paid LLM pass needs a changed skill (select_changed_skills.skills_changed).
        # A setup-only change keeps the keyless scan of every skill as the gate.
        if not skill_root_changed(changed, configured):
            raise NoSkillChanges
        if setup_changed(changed):
            mode = "all"
    requested_provider = source.get("SKILL_EVAL_TIER1_PROVIDER") or source.get("SKILL_EVAL_PROVIDER") or "openai"
    if requested_provider == "openai-compatible":
        raise ValueError("gateway")
    selected = model_environment(source, 1)
    provider = selected["SKILL_EVAL_LLM_PROVIDER"]
    if provider not in {"openai", "anthropic", "nv_build"}:
        raise ValueError("gateway")
    if selected.get("SKILL_EVAL_LLM_BASE_URL") or source.get("SKILL_EVAL_TIER1_BASE_URL"):
        raise ValueError("gateway")
    key = selected.get(PROVIDER_KEYS[provider], "")
    if not key.strip():
        raise ValueError("key")
    # The enriched evidence replaces the keyless Tier 1 gate, so it must apply
    # the same reviewed baselines. The image validates the policy commit's
    # manifest; an inherited POLICY_SOURCE_SHA is ignored unless opted in.
    baselines = source.get("SECURITY_BASELINES", "false")
    if baselines not in {"true", "false"}:
        raise ValueError
    policy = source.get("POLICY_SOURCE_SHA", "") if baselines == "true" else ""
    if baselines == "true" and not SHA.fullmatch(policy):
        raise ValueError
    child = {name: source[name] for name in CONTEXT if name in source}
    child["SECURITY_BASELINES"] = baselines
    if policy:
        child["POLICY_SOURCE_SHA"] = policy
    child.update(
        REPORT_COMMIT=head,
        DIFF_HEAD_SHA=head,
        REPORT_IMAGE=image,
        SKILL_PATHS=roots,
        SCAN_MODE=mode,
        MIN_SCORE=source.get("SKILL_EVAL_MIN_SCORE", "70"),
        SCAN_WORKERS=source.get("SKILL_EVAL_SCAN_WORKERS", "4"),
        LLM_ENABLED="true",
        LLM_PROVIDER=provider,
        LLM_MODEL=selected.get("SKILL_EVAL_LLM_MODEL", ""),
        LLM_API_KEY=key,
    )
    return child


def main(argv: list[str] | None = None) -> int:
    """Run the immutable image or its installed entrypoint; never execute candidate code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--in-image", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not IMAGE.fullmatch(args.image):
            raise ValueError
        workspace, reports = _paths_safe(args.workspace, args.reports)
        source = dict(os.environ)
        try:
            child = _environment(workspace, args.image, source)
        except NoSkillChanges:
            print(
                "No changed file is under a configured skill root, so the LLM-enriched Tier 1 scan was skipped. "
                "The keyless Tier 1 scan remains the gate."
            )
            return 0
        reports.mkdir(parents=True, exist_ok=True)
        host = {name: value for name, value in source.items() if name in HOST_ENV}
        host.setdefault("PATH", os.defpath)
        if args.in_image:
            # Keep sibling trusted helper imports; ignore Python environment and
            # user site packages, and do not add the consumer cwd to sys.path.
            host["PATH"] = (
                "/opt/skillevaluator/.venv/bin:/opt/semgrep/bin:/opt/skillspector/bin:/usr/local/bin:/usr/bin:/bin"
            )
            return subprocess.run(
                [
                    "/opt/skillevaluator/.venv/bin/python",
                    "-E",
                    "-s",
                    "/opt/skillevaluator/ci/run_tier1.py",
                    "--workspace",
                    str(workspace),
                    "--reports",
                    str(reports),
                ],
                cwd="/opt/skillevaluator",
                env={**host, **child},
                check=False,
            ).returncode
        with tempfile.TemporaryDirectory(prefix="skillevaluator-anonymous-docker-") as docker_config:
            docker = ["docker", "--config", docker_config]
            pulled = subprocess.run([*docker, "pull", args.image], cwd=reports.parent, env=host, check=False)
            if pulled.returncode:
                return 1
            command = [
                *docker,
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,nosuid,noexec,size=1g",
                "--tmpfs",
                "/audit-tmp:rw,nosuid,noexec,mode=1777,size=1g",
                "--env",
                "TMPDIR=/audit-tmp",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--mount",
                f"type=bind,src={workspace},dst=/workspace,readonly",
                "--mount",
                f"type=bind,src={reports},dst=/reports",
            ]
            for name in sorted(child):
                command.extend(["--env", name])
            command.append(args.image)
            return subprocess.run(command, cwd=reports.parent, env={**host, **child}, check=False).returncode
    except ValueError as error:
        message = {
            "gateway": "Tier 1 image supports openai, anthropic, and nv_build with their default endpoints only.",
            "key": "Tier 1 LLM evaluation requires the selected provider credential.",
        }.get(str(error), "Tier 1 image configuration or evaluated checkout is invalid.")
        print(message)
        return 1
    except Exception:
        print("Tier 1 image evaluation could not start or complete; no result is claimed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
