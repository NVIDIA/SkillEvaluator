# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, bounded plugin skill discovery and scan-visibility records."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    PLUGIN_COMPONENT_MAX_ITEMS,
    PLUGIN_TREE_MAX_DISCOVERED_PATHS,
    SCAN_ARTIFACT_DIRS,
    SCAN_EXCLUDED_DIRS,
    SKILL_MANIFEST_VARIANTS,
)
from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_formats import DEFAULT_SKILLS_DIR
from skillevaluator.plugin_paths import PluginRootReader, _plugin_finding
from skillevaluator.utils.secure_fs import (
    MAX_SECURE_DIRECTORY_DEPTH,
    SecureFile,
    discover_secure_files,
    lstat_walk,
    stat_is_link_or_reparse,
)

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


def _unscanned_skill_finding(reader: PluginRootReader, manifest_rel: PurePosixPath) -> Finding:
    """HIGH for a skill a client loads from a declared skills folder that Tier 1 never scans.

    The skill sits deep in an ``evals/``, ``results/``, or ``versions/`` folder
    inside a skill of the declared folder (a skill's own evaluation output or
    snapshots, which the whole-tree security, secret, and Unicode scans prune).
    Codex searches the folder recursively and loads it. Every client view
    reports it with the same text, so the merged inventory keeps one finding.
    """
    return _plugin_finding(
        Severity.HIGH,
        "plugin_skill_in_unscanned_folder",
        f"'{manifest_rel.as_posix()}' is a skill inside a folder that Tier 1 scans skip (evals/, results/, "
        "versions/). A client loads it from the declared skills folder, but SkillEvaluator never security-scans it",
        reader.display(manifest_rel),
        "Give the skill folder a different name, or move evaluation output and version snapshots out of the "
        "declared skills folder.",
        metadata={"path": manifest_rel.as_posix()},
    )


_DEPENDENCY_DIRS = SCAN_EXCLUDED_DIRS - SCAN_ARTIFACT_DIRS


_DEPENDENCY_DIRS_TEXT = "node_modules/, .venv/, .git/, __pycache__/"


def _loaded_dependency_skill(parts: tuple[str, ...], *, recursive: bool) -> str | None:
    """The dependency folder of a ``SKILL.md`` a client loads from a skills folder, or ``None``.

    ``parts`` is the manifest path relative to the skills folder. Claude Code
    loads every ``<skills folder>/<name>/SKILL.md``, hidden names included, so
    ``skills/node_modules/SKILL.md`` and ``skills/.venv/SKILL.md`` are skills.
    Codex (``recursive``) searches the folder at any depth but skips hidden
    folders, so it also loads ``skills/x/node_modules/pkg/SKILL.md``.
    """
    folders = parts[:-1]
    if len(folders) == 1 and folders[0] in _DEPENDENCY_DIRS:
        return folders[0]
    if not recursive or any(part.startswith(".") for part in folders):
        return None
    return next((part for part in folders if part in _DEPENDENCY_DIRS), None)


def find_dependency_folder_skills(
    root: Path, skills_dir: str = DEFAULT_SKILLS_DIR, *, recursive: bool = True
) -> tuple[list[PurePosixPath], bool]:
    """Skills a client loads from a dependency folder inside a skills folder, and whether the search finished.

    Bundled-skill discovery prunes ``node_modules/``, ``.venv/``, ``.git/``,
    and ``__pycache__/`` like the whole-tree scans, so a ``SKILL.md`` there is
    never listed or scanned, yet clients load it
    (:func:`_loaded_dependency_skill`). The paths are plugin-root relative.
    The walk reads names only, never follows links, skips hidden folders
    below the first level (no client loads skills from them), and stops after
    :data:`~skillevaluator.constants.PLUGIN_TREE_MAX_DISCOVERED_PATHS` entries.
    """
    start = Path(root)
    for part in PurePosixPath(skills_dir).parts:
        start = start / part
        try:
            metadata = start.lstat()
        except OSError:
            return [], True
        if stat_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            return [], True  # a linked skills folder is reported by skill discovery
    base = PurePosixPath(skills_dir)
    found: list[PurePosixPath] = []
    budget = PLUGIN_TREE_MAX_DISCOVERED_PATHS
    # Breadth first, so the folders clients load first are searched first.
    # (directory, path relative to the skills folder, whether its subfolders are searched)
    pending: list[tuple[Path, PurePosixPath, bool]] = [(start, PurePosixPath(), True)]
    while pending:
        directory, relative, descend = pending.pop(0)
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            budget -= 1
            if budget < 0:
                return sorted(found), False
            child = relative / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                if entry.name in SKILL_MANIFEST_VARIANTS and _loaded_dependency_skill(child.parts, recursive=recursive):
                    found.append(base / child)
                continue
            if not descend or not stat.S_ISDIR(metadata.st_mode) or stat_is_link_or_reparse(metadata):
                continue
            searched = recursive and not entry.name.startswith(".") and len(child.parts) < MAX_SECURE_DIRECTORY_DEPTH
            if searched or (not relative.parts and entry.name in _DEPENDENCY_DIRS):
                # A first-level dependency folder is a skill folder (its own SKILL.md); only a
                # recursive client searches below it, and never below a hidden one.
                pending.append((Path(entry.path), child, searched))
    return sorted(found), True


def dependency_folder_skill_findings(
    display: Callable[[PurePosixPath], str], skills_dir: str, found: list[PurePosixPath], complete: bool
) -> list[Finding]:
    """HIGH for each skill a client loads from a dependency folder; LOW when the search stopped early.

    ``display`` maps a plugin-root-relative path to the finding's file path.
    Every client view and the bundled-skill check report the same text, so a
    merged inventory keeps one finding per skill.
    """
    findings: list[Finding] = []
    for rel in found[:PLUGIN_COMPONENT_MAX_ITEMS]:
        folder = next(part for part in rel.parts if part in _DEPENDENCY_DIRS)
        findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_skill_in_unscanned_folder",
                f"'{rel.as_posix()}' is a skill inside '{folder}/', a folder that Tier 1 scans skip "
                f"({_DEPENDENCY_DIRS_TEXT}). A client loads it from '{skills_dir}/' (Claude Code loads every "
                f"'{skills_dir}/<name>/' folder, and Codex searches the folder at any depth), but SkillEvaluator "
                "never lists or security-scans it",
                display(rel),
                "Give the skill folder a different name, or keep dependency folders, VCS metadata, and bytecode "
                "caches out of skills folders.",
                metadata={"path": rel.as_posix()},
            )
        )
    if not complete:
        findings.append(
            _plugin_finding(
                Severity.LOW,
                "plugin_skill_scan_incomplete",
                f"the search for skills inside dependency folders in '{skills_dir}/' stopped after "
                f"{PLUGIN_TREE_MAX_DISCOVERED_PATHS} entries; a client may load skills from the rest",
                display(PurePosixPath(skills_dir)),
                "Keep dependency folders out of skills folders.",
                metadata={"path": skills_dir},
            )
        )
    return findings


_ENV_TEMPLATE_SUFFIXES = frozenset({"example", "sample", "template", "dist", "defaults", "tmpl"})
_ENV_SCAN_MAX_DEPTH = 32


def is_env_file(name: str) -> bool:
    """Whether *name* is a ``.env`` or ``.env.*`` file, in any letter case, that is not a template.

    Tier 1 flags these names and native Tier 3 staging leaves them out of the
    plugin copy, so both use this one rule.
    """
    folded = name.casefold()
    return folded == ".env" or (folded.startswith(".env.") and folded.rsplit(".", 1)[-1] not in _ENV_TEMPLATE_SUFFIXES)


def _scan_env_entries(
    entries: list[os.DirEntry[str]], rel_dir: PurePosixPath, depth: int, hits: list[PurePosixPath], budget: int
) -> tuple[list[str], int, bool]:
    """Return child names, remaining entry budget, and whether every child was inspectable."""
    children: list[str] = []
    complete = True
    for entry in entries:
        budget -= 1
        if budget < 0:
            break
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            complete = False
            continue
        if not is_dir:
            if is_env_file(entry.name):
                hits.append(rel_dir / entry.name)
        elif entry.name not in SCAN_EXCLUDED_DIRS:
            if depth < _ENV_SCAN_MAX_DEPTH:
                children.append(entry.name)
            else:
                complete = False
    return children, budget, complete


def find_environment_files(root: Path) -> tuple[list[PurePosixPath], bool]:
    """Names-only walk for shipped ``.env`` / ``.env.*`` files; files are never opened.

    Returns the hits and whether the walk was complete (``False`` when the entry
    budget ran out, the depth limit was reached, or a directory could not be inspected).

    On POSIX every directory is opened relative to its parent descriptor with
    ``O_NOFOLLOW``, so a directory swapped for a symlink is never listed. The walk
    is depth-first and opens a subdirectory only after its previous sibling's
    subtree is closed, so open descriptors grow with depth (at most
    ``_ENV_SCAN_MAX_DEPTH + 1``), not with the number of sibling directories.
    """
    hits: list[PurePosixPath] = []
    budget = CONTENT_DEDUP_MAX_DISCOVERED_PATHS
    complete = True
    if os.name == "posix" and os.open in os.supports_dir_fd and os.scandir in os.supports_fd:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            root_fd = os.open(root, flags)
        except OSError:
            return hits, False
        # Frames: (fd, rel_dir, depth, subdirectories left to visit; None until listed).
        stack: list[tuple[int, PurePosixPath, int, list[str] | None]] = [(root_fd, PurePosixPath(), 0, None)]
        try:
            while stack:
                fd, rel_dir, depth, children = stack[-1]
                if children is None:
                    try:
                        with os.scandir(fd) as iterator:
                            entries = sorted(iterator, key=lambda item: item.name)
                    except OSError:
                        complete = False
                        entries = []
                    found, budget, inspected = _scan_env_entries(entries, rel_dir, depth, hits, budget)
                    complete = complete and inspected
                    children = found[::-1]  # pop() visits them in name order
                    stack[-1] = (fd, rel_dir, depth, children)
                if budget < 0 or not children:
                    stack.pop()
                    os.close(fd)
                    continue
                name = children.pop()
                try:
                    child_fd = os.open(name, flags, dir_fd=fd)
                except OSError:
                    complete = False
                    continue
                stack.append((child_fd, rel_dir / name, depth + 1, None))
        finally:
            for fd, *_frame in stack:
                os.close(fd)
        return sorted(hits), complete and budget >= 0

    pending: list[tuple[Path, PurePosixPath, int]] = [(root, PurePosixPath(), 0)]
    while pending and budget >= 0:
        directory, rel_dir, depth = pending.pop()
        try:
            if stat_is_link_or_reparse(directory.lstat()):
                continue
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError:
            complete = False
            continue
        children, budget, inspected = _scan_env_entries(entries, rel_dir, depth, hits, budget)
        complete = complete and inspected
        pending.extend((directory / name, rel_dir / name, depth + 1) for name in children)
    return sorted(hits), complete and budget >= 0
