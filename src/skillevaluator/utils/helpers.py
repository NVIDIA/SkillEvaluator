# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities for SkillEvaluator."""

import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from skillevaluator.constants import SCAN_EXCLUDED_DIRS, SKILL_MANIFEST_FILE, SKILL_MANIFEST_VARIANTS


def make_timestamped_basename(prefix: str, suffix: str = "") -> str:
    """Return ``<prefix>-YYYYMMDDHHMMSS<suffix>`` for report artifacts.

    Used so each combined ``validate`` run writes a distinct, sortable report
    file rather than overwriting the previous one (SkillEvaluator parity). ``suffix``
    is the optional file extension (e.g. ``".html"``); omit it to get the bare
    timestamped basename.
    """
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{stamp}{suffix}"


def find_skills_in_directory(root_path: Path) -> list[Path]:
    """Find all skill directories containing SKILL.md.

    Uses case-insensitive manifest detection per SkillEvaluator spec.
    Deduplicates results when both SKILL.md and skill.md exist.

    Args:
        root_path: Root directory or SKILL.md file path to search

    Returns:
        Sorted list of unique paths to skill directories
    """
    skill_dirs: set[Path] = set()

    if root_path.is_file():
        if root_path.name.upper() == SKILL_MANIFEST_FILE.upper():
            skill_dirs.add(root_path.parent)
        return sorted(skill_dirs)

    for manifest_name in SKILL_MANIFEST_VARIANTS:
        for skill_md in root_path.rglob(manifest_name):
            skill_dirs.add(skill_md.parent)

    return sorted(skill_dirs)


def find_bundled_plugin_skills(plugin_root: Path) -> list[Path]:
    """Find live skills under a plugin's ``skills/`` directory."""
    skills_root = plugin_root / "skills"
    if not skills_root.is_dir():
        return []
    return [
        skill_dir
        for skill_dir in find_skills_in_directory(skills_root)
        if not any(part in SCAN_EXCLUDED_DIRS for part in skill_dir.relative_to(skills_root).parts)
    ]


def resolve_git_remote_url(local_path: Path) -> str | None:
    """Resolve a local path to a browsable HTTPS URL if inside a git repo.

    Detects the git remote origin, converts SSH/HTTPS URLs to a browsable
    HTTPS URL, and appends the relative path within the repo.

    Examples:
        /home/user/project/skills/ with remote git@github.com:org/project.git
        -> https://github.com/org/project/tree/main/skills

    Args:
        local_path: Absolute path to resolve

    Returns:
        HTTPS URL string, or None if not inside a git repo
    """
    resolved = local_path.resolve()

    try:
        # Find the git repo root
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(resolved if resolved.is_dir() else resolved.parent),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Get the remote origin URL
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Get the current branch.
        # In CI pipelines (detached HEAD), git returns "HEAD" so prefer an
        # explicitly supplied branch name.
        branch = os.environ.get("GITHUB_REF_NAME", "")
        if not branch:
            try:
                branch = subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=repo_root,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                branch = "main"
        if branch == "HEAD":
            branch = "main"

        # Convert SSH URL to HTTPS.
        https_url = _ssh_to_https(remote_url)
        if not https_url:
            return None

        # Compute the relative path within the repo
        try:
            rel_path = str(resolved.relative_to(repo_root))
        except ValueError:
            rel_path = ""

        if rel_path and rel_path != ".":
            tree_segment = "/tree/" if https_url.startswith("https://github.com/") else "/-/tree/"
            return f"{https_url}{tree_segment}{branch}/{rel_path}"
        return https_url

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _ssh_to_https(remote_url: str) -> str | None:
    """Convert a git remote URL to a browsable HTTPS URL.

    Handles:
        ssh://git@host:port/group/repo.git -> https://host/group/repo
        git@host:group/repo.git            -> https://host/group/repo
        https://host/group/repo.git        -> https://host/group/repo
    """
    url = remote_url.strip().rstrip("/")

    # Remove .git suffix
    url = url.removesuffix(".git")

    # ssh://git@host:port/path
    match = re.match(r"ssh://[^@]+@([^:/]+)(?::\d+)?(/.*)", url)
    if match:
        return f"https://{match.group(1)}{match.group(2)}"

    # git@host:path
    match = re.match(r"[^@]+@([^:]+):(.+)", url)
    if match:
        return f"https://{match.group(1)}/{match.group(2)}"

    # Already HTTPS — strip any embedded credentials before rendering a link.
    if url.startswith("https://"):
        match = re.match(r"https://[^@]+@(.+)", url)
        if match:
            return f"https://{match.group(1)}"
        return url

    return None


def get_skill_name_from_path(skill_path: Path) -> str:
    """Extract skill name from path.

    Args:
        skill_path: Path to skill directory

    Returns:
        Skill name (directory name)
    """
    if skill_path.is_file():
        return skill_path.parent.name
    return skill_path.name
