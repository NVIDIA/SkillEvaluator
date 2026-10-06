# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities for SkillEvaluator."""

import os
import re
import stat
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    PLUGIN_TREE_MAX_DISCOVERED_PATHS,
    PLUGIN_TREE_PRUNED_DIRS,
    SCAN_ARTIFACT_DIRS,
    SCAN_EXCLUDED_DIRS,
    SKILL_MANIFEST_VARIANTS,
)
from skillevaluator.utils.secure_fs import (
    MAX_SECURE_DIRECTORY_DEPTH,
    SecureFile,
    discover_secure_files,
    lstat_walk,
    stat_is_link_or_reparse,
)


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


# Preference among the manifest spellings of one skill folder: SKILL.md first.
_SKILL_MANIFEST_RANK = {name: rank for rank, name in enumerate(SKILL_MANIFEST_VARIANTS)}


def preferred_skill_manifests(files: Iterable[SecureFile]) -> list[SecureFile]:
    """Pick one manifest per skill folder, ``SKILL.md`` over ``skill.md``.

    ``files`` are skill manifests (each named like one of
    :data:`~skillevaluator.constants.SKILL_MANIFEST_VARIANTS`) from secure
    discovery. The result holds the preferred one for each
    ``relative_path.parent``, ordered by that folder as paths compare (part by
    part, so ``a/b`` comes before ``a-b``). Nothing is read.
    """
    best: dict[Path, SecureFile] = {}
    for file in files:
        folder = file.relative_path.parent
        current = best.get(folder)
        if (
            current is None
            or _SKILL_MANIFEST_RANK[file.relative_path.name] < _SKILL_MANIFEST_RANK[current.relative_path.name]
        ):
            best[folder] = file
    return [best[folder] for folder in sorted(best)]


def _discover_skill_manifests(root_path: Path) -> list[SecureFile]:
    """Return one securely discovered manifest identity per skill directory."""
    manifests = discover_secure_files(
        root_path,
        selected=lambda relative: relative.name in SKILL_MANIFEST_VARIANTS,
        excluded_dirs=SCAN_EXCLUDED_DIRS,
        max_paths=CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    )
    return preferred_skill_manifests(manifests)


def _plugin_skills_root(plugin_root: Path, skills_dir: str = "skills") -> Path | None:
    """Return ``<plugin_root>/<skills_dir>`` when it is a real directory; raise on a link.

    ``skills_dir`` is a plugin-root-relative POSIX path (``skills`` or a
    declared folder such as ``my-skills``). Every component is checked without
    following links.
    """
    relative = PurePosixPath(skills_dir)
    walk = lstat_walk(plugin_root, relative)
    if isinstance(walk.error, FileNotFoundError):
        return None
    if walk.error is not None:  # NotADirectoryError (a part was replaced meanwhile) or another OSError
        raise ValueError(f"Cannot inspect plugin skills safely: {walk.error}") from walk.error
    if walk.outcome == "link":
        raise ValueError(f"Plugin skills folder is a symlink, junction, or reparse point: {skills_dir}")
    if walk.outcome == "not_dir" or (walk.metadata is not None and not stat.S_ISDIR(walk.metadata.st_mode)):
        return None
    return plugin_root.joinpath(*relative.parts)


def find_bundled_plugin_skill_manifests(plugin_root: Path, skills_dir: str = "skills") -> list[SecureFile]:
    """Return retained manifest identities for the skills in a plugin skills folder.

    ``skills_dir`` is the plugin-root-relative skills folder: ``skills`` (the
    default) or a folder a manifest declares, such as ``my-skills``. The
    scan-exclusion names (``evals``, ``results``, ``versions``, and their
    dotted forms) mark a skill's own evaluation output and snapshots, so they
    are skipped inside a skill. A folder with one of those names directly
    under the skills folder is not inside a skill: clients load
    ``skills/evals/`` and ``skills/versions/v2/`` like any other skill folder,
    so it is searched too. The returned identities are all relative to the
    skills folder.
    """
    skills_root = _plugin_skills_root(plugin_root, skills_dir)
    if skills_root is None:
        return []
    manifests = _discover_skill_manifests(skills_root)
    for name in sorted(SCAN_ARTIFACT_DIRS):
        child = skills_root / name
        try:
            metadata = child.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"Cannot inspect bundled plugin skills safely: {exc}") from exc
        if stat_is_link_or_reparse(metadata):
            raise ValueError(f"Plugin skills folder is a symlink, junction, or reparse point: {skills_dir}/{name}")
        if not stat.S_ISDIR(metadata.st_mode):
            continue
        # Discovery records ``child`` normalized and absolute, so its parent is the skills folder
        # in the form the identities found above record it.
        manifests.extend(
            SecureFile(found.root.parent, found.path, Path(name) / found.relative_path, found.metadata)
            for found in _discover_skill_manifests(child)
        )
    return sorted(manifests, key=lambda manifest: manifest.relative_path.parent)


def find_unscanned_plugin_skill_manifests(plugin_root: Path, skills_dir: str = "skills") -> list[PurePosixPath]:
    """Return plugin-root-relative ``SKILL.md`` paths that bundled-skill discovery skips.

    These sit in a scan-excluded folder (``evals``, ``results``, ``versions``,
    or a dotted form) inside the skills folder ``skills_dir`` (``skills`` or a
    declared one), below the first level. Tier 1 does not scan them, but a
    client that searches a skills folder recursively (Codex) loads them. The
    walk reads names only, never follows links, and is bounded like discovery.
    """
    try:
        skills_root = _plugin_skills_root(plugin_root, skills_dir)
    except ValueError:
        return []  # reported by bundled-skill discovery
    if skills_root is None:
        return []
    base = PurePosixPath(skills_dir)
    found: list[PurePosixPath] = []
    budget = CONTENT_DEDUP_MAX_DISCOVERED_PATHS
    # (directory, path relative to the skills folder, inside a skipped folder)
    pending: list[tuple[Path, PurePosixPath, bool]] = [(skills_root, PurePosixPath(), False)]
    while pending and budget > 0:
        directory, relative, skipped = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            budget -= 1
            if budget <= 0:
                break
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            child = relative / entry.name
            if not is_dir:
                if skipped and entry.name in SKILL_MANIFEST_VARIANTS:
                    found.append(base / child)
                continue
            if entry.name in SCAN_EXCLUDED_DIRS and entry.name not in SCAN_ARTIFACT_DIRS:
                continue  # VCS, virtualenv, package, and bytecode caches
            if len(child.parts) < MAX_SECURE_DIRECTORY_DEPTH:
                artifact = entry.name in SCAN_ARTIFACT_DIRS and len(child.parts) > 1
                pending.append((Path(entry.path), child, skipped or artifact))
    return sorted(found)


def find_skill_manifest_in(skill_dir: Path) -> SecureFile | None:
    """Return the manifest identity of one skill folder (``SKILL.md`` first), or ``None``.

    Only the folder's own entries are listed, without following links; a link
    or special entry there raises :class:`ValueError`.
    """
    manifests = discover_secure_files(
        skill_dir,
        selected=lambda relative: len(relative.parts) == 1 and relative.name in SKILL_MANIFEST_VARIANTS,
        max_paths=CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
        max_depth=1,
    )
    preferred = preferred_skill_manifests(manifests)
    return preferred[0] if preferred else None


def find_bundled_plugin_skills(plugin_root: Path) -> list[Path]:
    """Find live, regular skills under a plugin's ``skills/`` directory."""
    skills_root = plugin_root / "skills"
    return [
        skills_root / manifest.relative_path.parent for manifest in find_bundled_plugin_skill_manifests(plugin_root)
    ]


def verify_plugin_tree(plugin_root: Path) -> int:
    """Walk the whole plugin tree without following links; fail closed on unsafe entries.

    Whole-plugin Tier 1 scanners read root-owned plugin content as well as
    bundled skills, so every entry they can reach must be a regular,
    single-link file or a real directory contained by the plugin root.
    Symlinks, junctions and other reparse points, hard links, and special
    files raise :class:`~skillevaluator.utils.secure_fs.SecurePathError` (a
    ``ValueError``) before any scanner reads content. Only the recognized
    contained ``CLAUDE.md -> AGENTS.md`` alias is tolerated. Directories that
    scanners never enter are pruned. Returns the number of verified files.
    """

    def is_non_directory(relative: Path) -> bool:
        # Secure discovery re-checks every entry from its own no-follow
        # metadata; this only keeps real directories unselected so they are
        # descended rather than rejected. An entry that cannot be inspected
        # stays selected and therefore fails closed.
        try:
            return not stat.S_ISDIR((plugin_root / relative).lstat().st_mode)
        except OSError:
            return True

    files = discover_secure_files(
        plugin_root,
        selected=is_non_directory,
        excluded_dirs=PLUGIN_TREE_PRUNED_DIRS,
        max_paths=PLUGIN_TREE_MAX_DISCOVERED_PATHS,
    )
    return len(files)


def resolve_git_root(local_path: Path) -> Path | None:
    """Return the containing Git repository root without importing optional tiers."""
    resolved = local_path.resolve()
    working_dir = resolved if resolved.is_dir() else resolved.parent
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(working_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return Path(root).resolve() if root else None


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
    repo_root = resolve_git_root(resolved)
    if repo_root is None:
        return None
    https_url = git_origin_https_url(repo_root)
    if not https_url:
        return None

    try:
        # Get the current branch.
        # In CI pipelines (detached HEAD), git returns "HEAD" so prefer an
        # explicitly supplied branch name.
        branch = os.environ.get("GITHUB_REF_NAME", "")
        if re.fullmatch(r"\d+/merge", branch):
            # A pull request merge ref belongs to the workflow repository, but
            # local_path may point into a different checkout. Resolve the
            # revision from the repository that contains local_path.
            try:
                branch = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo_root),
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                branch = ""
        if not branch:
            try:
                branch = subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=str(repo_root),
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                branch = "main"
        if branch == "HEAD":
            branch = "main"

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


@dataclass(frozen=True)
class GitOrigin:
    """The ``origin`` remote of a repository (see :func:`git_origin`).

    ``https_url`` is set for an ssh, SCP-style, or https remote. For any other
    remote only ``scheme`` describes it (``"http"``, or ``None`` for a local
    path), never its text, which may hold credentials.
    """

    configured: bool = False
    https_url: str | None = None
    scheme: str | None = None


_URL_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*")


def git_origin(git_root: Path) -> GitOrigin:
    """Read the ``origin`` remote of the repository at *git_root* with one ``git remote get-url origin``.

    Only ``ssh://``, SCP-style (``git@host:group/repo``), and ``https://``
    remotes give an HTTPS URL, with any credentials stripped (see
    :func:`_ssh_to_https`). Any other remote, such as ``http://``, ``git://``,
    ``file://``, or a local path, gives none, so repository identity derived
    from it fails closed.
    """
    try:
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=str(git_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return GitOrigin()
    https_url = _ssh_to_https(remote_url)
    if https_url is not None:
        return GitOrigin(configured=True, https_url=https_url)
    scheme, separator, _rest = remote_url.partition("://")
    if separator and _URL_SCHEME_RE.fullmatch(scheme):
        return GitOrigin(configured=True, scheme=scheme.lower())
    return GitOrigin(configured=True)


def git_origin_https_url(git_root: Path) -> str | None:
    """Return the ``origin`` remote at *git_root* as an HTTPS URL, or ``None`` (see :func:`git_origin`)."""
    return git_origin(git_root).https_url


def _ssh_to_https(remote_url: str) -> str | None:
    """Convert a git remote URL to a browsable HTTPS URL, without credentials.

    Handles:
        ssh://git@host:port/group/repo.git        -> https://host/group/repo
        git@host:group/repo.git                   -> https://host/group/repo
        https://user:token@host:port/group/repo.git -> https://host:port/group/repo

    The SSH port is dropped (it is not the web port); an HTTPS port is kept.
    Any other remote gives ``None``.
    """
    url = remote_url.strip().rstrip("/").removesuffix(".git")
    if "://" not in url:
        # SCP-style user@host:path. Only a string without a scheme can be one:
        # "https://user@host:8443/group/repo" would otherwise read as the path
        # "8443/group/repo" on host "host".
        match = re.match(r"[^@]+@([^:]+):(.+)", url)
        return f"https://{match.group(1)}/{match.group(2)}" if match else None
    try:
        parts = urlsplit(url)
        hostname, port = parts.hostname, parts.port
    except ValueError:  # an unparseable host or port
        return None
    if parts.scheme not in {"https", "ssh"} or not hostname:
        return None
    if ":" in hostname:  # an IPv6 address
        hostname = f"[{hostname}]"
    if parts.scheme == "https" and port is not None:
        hostname = f"{hostname}:{port}"
    return f"https://{hostname}{parts.path}"


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
