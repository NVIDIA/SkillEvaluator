# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities for SkillEvaluator."""

import os
import re
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    SCAN_EXCLUDED_DIRS,
    SKILL_MANIFEST_VARIANTS,
)
from skillevaluator.utils.secure_fs import SecureFile, discover_secure_files, stat_is_link_or_reparse


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
    try:
        metadata = root_path.lstat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ValueError(f"Cannot inspect skill root safely: {exc}") from exc
    if stat_is_link_or_reparse(metadata):
        raise ValueError(f"Skill root is a symlink, junction, or reparse point: {root_path.name}")
    if not stat.S_ISDIR(metadata.st_mode):
        if root_path.name not in SKILL_MANIFEST_VARIANTS:
            return []
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Refusing selected manifest that is not a regular file: {root_path.name}")
        if getattr(metadata, "st_nlink", 1) != 1:
            raise ValueError(f"Refusing hard-linked selected manifest: {root_path.name}")
        return [root_path.parent]

    manifests = _discover_skill_manifests(root_path)
    return [(root_path / manifest.relative_path).parent for manifest in manifests]


def _discover_skill_manifests(root_path: Path) -> list[SecureFile]:
    """Return one securely discovered manifest identity per skill directory."""
    manifests = discover_secure_files(
        root_path,
        selected=lambda relative: relative.name in SKILL_MANIFEST_VARIANTS,
        excluded_dirs=SCAN_EXCLUDED_DIRS,
        max_paths=CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    )
    priority = {name: index for index, name in enumerate(SKILL_MANIFEST_VARIANTS)}
    selected: dict[Path, SecureFile] = {}
    for manifest in manifests:
        directory = manifest.relative_path.parent
        current = selected.get(directory)
        if current is None or priority[manifest.relative_path.name] < priority[current.relative_path.name]:
            selected[directory] = manifest
    return [selected[directory] for directory in sorted(selected)]


def find_bundled_plugin_skill_manifests(plugin_root: Path) -> list[SecureFile]:
    """Return retained manifest identities for live ``<plugin_root>/skills`` entries."""
    skills_root = plugin_root / "skills"
    try:
        metadata = skills_root.lstat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ValueError(f"Cannot inspect bundled plugin skills safely: {exc}") from exc
    if stat_is_link_or_reparse(metadata):
        raise ValueError("Plugin skills root is a symlink, junction, or reparse point")
    if not stat.S_ISDIR(metadata.st_mode):
        return []
    return _discover_skill_manifests(skills_root)


def find_bundled_plugin_skills(plugin_root: Path) -> list[Path]:
    """Find live, regular skills under a plugin's ``skills/`` directory."""
    skills_root = plugin_root / "skills"
    return [
        skills_root / manifest.relative_path.parent for manifest in find_bundled_plugin_skill_manifests(plugin_root)
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
