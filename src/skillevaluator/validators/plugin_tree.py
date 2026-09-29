# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scan a plugin's whole tree exactly once.

Folder-oriented Tier 1 validators discover ``SKILL.md`` files and scan each
skill directory. For a plugin that bundles skills under ``skills/`` that walk
alone never reaches root-owned plugin content such as ``scripts/``,
``hooks/``, ``commands/``, ``agents/`` or ``.mcp.json``.

Whole-plugin checks run inside :func:`plugin_tree_scope`. Within it,
:meth:`ValidatorBase._validate_folder_or_skill` validates each bundled skill
as its own unit (findings keep the ``[<skill>]`` prefix, and skill-relative
finding paths are rebased onto the plugin root), then validates the plugin
root once with every bundled-skill subtree excluded. File walkers prune those
subtrees (:func:`plugin_tree_exclusions`), and external scanners without an
exact path exclude run on a staged copy (:func:`plugin_tree_scan_view`), so
every file is scanned by exactly one pass.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import shutil
import stat
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from skillevaluator.logging_config import get_logger
from skillevaluator.utils.secure_fs import stat_is_link_or_reparse

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

logger = get_logger(__name__)

__all__ = [
    "PluginTree",
    "active_plugin_tree",
    "is_plugin_tree_root",
    "plugin_relative_dir",
    "plugin_tree_exclusions",
    "plugin_tree_scan_view",
    "plugin_tree_scope",
    "rebase_relative_finding_paths",
    "rewrite_finding_path_prefix",
]


def _lexical(path: Path) -> Path:
    """Return an absolute lexical path without resolving links."""
    return Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100 - lexical, never resolved


@dataclass(frozen=True)
class PluginTree:
    """A plugin root and the bundled skill directories that own subtrees of it."""

    root: Path
    skill_dirs: tuple[Path, ...]

    @property
    def root_key(self) -> Path:
        return _lexical(self.root)


_ACTIVE_PLUGIN_TREE: contextvars.ContextVar[PluginTree | None] = contextvars.ContextVar(
    "skillevaluator_plugin_tree", default=None
)


@contextlib.contextmanager
def plugin_tree_scope(root: Path, skill_dirs: Iterable[Path]) -> Iterator[PluginTree]:
    """Make whole-plugin validators scan *root* and its bundled skills exactly once.

    *skill_dirs* are the bundled skill directories returned by the secure,
    no-follow plugin discovery, spelled relative to *root* as given.
    """
    tree = PluginTree(root=root, skill_dirs=tuple(skill_dirs))
    token = _ACTIVE_PLUGIN_TREE.set(tree)
    try:
        yield tree
    finally:
        _ACTIVE_PLUGIN_TREE.reset(token)


def active_plugin_tree() -> PluginTree | None:
    """Return the plugin tree currently in scope, if any."""
    return _ACTIVE_PLUGIN_TREE.get()


def is_plugin_tree_root(path: Path) -> bool:
    """Return whether *path* is the root of the plugin tree currently in scope."""
    tree = _ACTIVE_PLUGIN_TREE.get()
    return tree is not None and _lexical(path) == tree.root_key


def plugin_relative_dir(path: Path) -> PurePosixPath | None:
    """Return *path* relative to the plugin root in scope, or ``None`` outside it."""
    tree = _ACTIVE_PLUGIN_TREE.get()
    if tree is None:
        return None
    try:
        relative = _lexical(path).relative_to(tree.root_key)
    except ValueError:
        return None
    return PurePosixPath(*relative.parts)


def plugin_tree_exclusions(path: Path) -> frozenset[tuple[str, ...]]:
    """Return bundled-skill subtrees strictly inside *path*, as relative part tuples.

    These subtrees are validated by their own per-skill pass, so a scan of
    *path* (the plugin root, or a skill that nests another bundled skill) must
    skip them. Outside a plugin tree scope nothing is excluded.
    """
    tree = _ACTIVE_PLUGIN_TREE.get()
    if tree is None:
        return frozenset()
    base = _lexical(path)
    excluded: set[tuple[str, ...]] = set()
    for skill_dir in tree.skill_dirs:
        try:
            relative = _lexical(skill_dir).relative_to(base)
        except ValueError:
            continue
        if relative.parts:
            excluded.add(relative.parts)
    return frozenset(excluded)


def rebase_relative_finding_paths(result: ValidationResult, directory: PurePosixPath) -> None:
    """Prefix relative finding paths in *result* with the plugin-relative *directory*.

    Validators report many paths relative to the scanned skill. Rebasing them
    onto the plugin root keeps each location attributable to its component
    after the ``[<skill>]`` prefix is stripped. Absolute paths and
    placeholders such as ``<plugin-skills>`` are left unchanged.
    """
    prefix = directory.as_posix()
    if prefix in ("", "."):
        return
    for finding in result.findings:
        file_path = finding.file_path
        if not file_path or file_path.startswith(("<", "[")):
            continue
        if Path(file_path).is_absolute() or PurePosixPath(file_path.replace("\\", "/")).is_absolute():
            continue
        finding.file_path = f"{prefix}/{file_path}"


def rewrite_finding_path_prefix(result: ValidationResult, old: str, new: str) -> None:
    """Map finding paths reported under a staged copy *old* back to *new*."""
    old = old.rstrip(os.sep)
    for finding in result.findings:
        file_path = finding.file_path
        if file_path == old:
            finding.file_path = new
        elif file_path.startswith(old + os.sep):
            finding.file_path = new + file_path[len(old) :]


def _staging_ignore(source: Path, excluded_subtrees: frozenset[tuple[str, ...]], excluded_dir_names: frozenset[str]):
    """Build a ``copytree`` ignore hook for a plugin scan view.

    The view drops bundled-skill subtrees, excluded directory names, and every
    link, reparse point, or special file. Whole-plugin scans verify the tree
    before they start, so dropping such entries here is defense in depth: an
    external scanner never follows a link out of the view.
    """

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).relative_to(source).parts
        dropped: set[str] = set()
        for name in names:
            try:
                metadata = Path(directory, name).lstat()
            except OSError:
                dropped.add(name)
                continue
            if stat_is_link_or_reparse(metadata):
                dropped.add(name)
            elif stat.S_ISDIR(metadata.st_mode):
                if name in excluded_dir_names or (*relative, name) in excluded_subtrees:
                    dropped.add(name)
            elif not stat.S_ISREG(metadata.st_mode) or (*relative, name) in excluded_subtrees:
                dropped.add(name)
        return dropped

    return ignore


@contextlib.contextmanager
def plugin_tree_scan_view(path: Path, *, excluded_dir_names: Iterable[str] = ()) -> Iterator[Path | None]:
    """Stage *path* without its bundled-skill subtrees for an external scanner.

    Yields ``None`` when *path* owns no bundled-skill subtree (scan it in
    place) or when staging fails; callers then scan in place, which can repeat
    bundled-skill findings but never loses coverage. Otherwise yields a
    temporary copy, named like *path*, that is removed on exit. Map reported
    absolute paths back with :func:`rewrite_finding_path_prefix`.
    """
    excluded = plugin_tree_exclusions(path)
    if not excluded or not path.is_dir():
        yield None
        return

    source = path.resolve()
    with tempfile.TemporaryDirectory(prefix="skillevaluator-plugin-root-") as temp_dir:
        view: Path | None = Path(temp_dir) / source.name
        try:
            shutil.copytree(
                source,
                view,
                symlinks=True,
                ignore=_staging_ignore(source, excluded, frozenset(excluded_dir_names)),
            )
        except (OSError, shutil.Error) as exc:
            logger.warning("Could not stage the plugin root without bundled skills (%s); scanning in place", exc)
            view = None
        yield view
