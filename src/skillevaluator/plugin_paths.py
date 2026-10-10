# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-bounded reads and declared component paths of a plugin.

Every path a plugin declares is plugin-controlled input.
:func:`normalize_declared_path` turns a manifest path into a contained
root-relative POSIX path, or names its problem. :class:`PluginRootReader`
classifies paths with ``lstat`` on every component (links are never
followed) and reads files through
:class:`~skillevaluator.utils.secure_fs.SecureRoot` with a byte bound. The
finding helpers report a declared path that is missing, escapes the root, is
unsafe, lacks the ``./`` prefix, starts with a root variable, or lies in a
folder Tier 1 scans skip, and a names-only walk notes the shipped folders the
scans skip and refuses links out of the plugin inside them. The component
inventory (:mod:`skillevaluator.plugin_components`) and the MCP collection
(:mod:`skillevaluator.plugin_mcp`) share them.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    PLUGIN_TREE_MAX_DISCOVERED_PATHS,
    PLUGIN_TREE_PRUNED_DIRS,
    SCAN_ARTIFACT_DIRS,
    SCAN_EXCLUDED_DIRS,
)
from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_formats import CLAUDE_PROFILE, DEFAULT_SKILLS_DIR, FormatProfile
from skillevaluator.utils.secure_fs import (
    MAX_SECURE_DIRECTORY_DEPTH,
    SecurePathError,
    SecureRoot,
    discover_secure_files,
    lstat_walk,
    stat_is_link_or_reparse,
)

# The category of every plugin schema and component finding; a policy overlay changes a severity with
# PLUGIN_SCHEMA.<check>.
PLUGIN_CATEGORY = "PLUGIN_SCHEMA"
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
# A longer declared path is "invalid": no component path comes close, and it bounds the work per path.
_MAX_DECLARED_PATH_CHARS = 4096


# --------------------------------------------------------------------------- #
# Root-bounded reads                                                          #
# --------------------------------------------------------------------------- #
PathKind = Literal["missing", "file", "dir", "link", "special"]


@dataclass(frozen=True)
class DeclaredPath:
    raw: str
    rel: PurePosixPath | None
    problem: str | None = None  # empty | escape | invalid | placeholder
    dot_relative: bool = True
    # The root placeholder (``${CURSOR_PLUGIN_ROOT}``) the path started with, if any.
    root_variable: str | None = None


def normalize_declared_path(raw: str, root_prefixes: Iterable[str]) -> DeclaredPath:
    """Normalize one manifest path to a contained root-relative POSIX path.

    Claude Code requires ``./``-relative paths (``"."``/``"./"`` names the root).
    ``root_prefixes`` are the root placeholders the format's client expands in
    manifest paths: callers pass :attr:`FormatProfile.manifest_path_prefixes`,
    which only Cursor's profile fills, so any other leading ``${...}`` is the
    ``placeholder`` problem. A root placeholder names the root only when a
    separator (``/`` or ``\\``) or nothing follows it: a client expands it as
    text, so ``${CURSOR_PLUGIN_ROOT}foo/x.sh`` loads ``<root>foo/x.sh`` beside
    the root, an escape. Absolute paths, home-relative paths, a drive letter in
    any part (``./C:/Users``), and ``..`` segments are escapes; any other colon
    (an NTFS data stream, ``x.md:hidden``) is ``invalid``. A path that starts
    with a root placeholder records it as ``root_variable``.
    """
    text = raw.strip()
    if not text:
        return DeclaredPath(raw, None, "empty")
    if "\x00" in text or len(text) > _MAX_DECLARED_PATH_CHARS:
        return DeclaredPath(raw, None, "invalid")
    normalized = text.replace("\\", "/")
    dot_relative = normalized in {".", "./"} or normalized.startswith("./")
    root_variable: str | None = None
    matched = [prefix for prefix in root_prefixes if prefix and normalized.startswith(prefix)]
    if matched:
        root_variable = max(matched, key=len)
        below_root = normalized[len(root_variable) :]
        if below_root and not below_root.startswith("/"):
            return DeclaredPath(raw, None, "escape")  # "<root>foo/x.sh" is beside the root, not in it
        normalized = "./" + below_root.lstrip("/")
        dot_relative = True
    if normalized.startswith(("/", "~")) or _WINDOWS_DRIVE_RE.match(normalized):
        return DeclaredPath(raw, None, "escape")
    if normalized.startswith("${"):
        return DeclaredPath(raw, None, "placeholder")
    if "${" in normalized or normalized.startswith("$"):
        return DeclaredPath(raw, None, "invalid")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    # Windows reads a character and a colon as a drive in any part, so joining
    # "./C:/Users/x" to the root leaves it; any other colon names a data stream.
    if any(part == ".." or part[1:2] == ":" for part in parts):
        return DeclaredPath(raw, None, "escape")
    if any(":" in part for part in parts):
        return DeclaredPath(raw, None, "invalid")
    rel = PurePosixPath(*parts) if parts else PurePosixPath(".")
    return DeclaredPath(raw, rel, None, dot_relative, root_variable)


class PluginRootReader:
    """No-follow classification and bounded reads beneath one plugin root.

    Security-relevant JSON configs (hooks, LSP, monitors, settings, MCP) draw on
    their own read budget, so large skill or rule files read for the context-cost
    estimate cannot starve them.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))  # noqa: PTH100 - lexical, never resolved
        self.bytes_read = 0
        self.config_bytes_read = 0

    def display(self, rel: PurePosixPath | str) -> str:
        rel_text = str(rel)
        return str(self.root) if rel_text in {"", "."} else str(self.root / rel_text)

    def kind(self, rel: PurePosixPath, *, allow_hard_links: bool = False) -> PathKind:
        """Classify ``rel`` with ``lstat`` on every component (links are never followed).

        A regular file with more than one hard link is ``special`` unless
        ``allow_hard_links`` (hook scripts, which are only scanned for evidence).
        """
        if not rel.parts or str(rel) == ".":
            return "dir"
        walk = lstat_walk(self.root, rel)
        if walk.outcome in ("missing", "not_dir"):
            return "missing"
        if walk.outcome == "link":
            return "link"
        metadata = walk.metadata
        if walk.outcome == "error" or metadata is None:
            return "special"
        if stat.S_ISDIR(metadata.st_mode):
            return "dir"
        if stat.S_ISREG(metadata.st_mode):
            return "file" if allow_hard_links or getattr(metadata, "st_nlink", 1) == 1 else "special"
        return "special"

    def _read_bytes(
        self, rel: PurePosixPath, max_bytes: int, *, config: bool = False, allow_hardlinks: bool = False
    ) -> bytes:
        """Bounded, anchored, no-follow read counted against a read budget; raises :class:`SecurePathError`.

        ``config`` charges the read to the separate config budget.
        ``allow_hardlinks`` also reads a regular file with more than one link.
        """
        used = self.config_bytes_read if config else self.bytes_read
        budget = "config" if config else "inventory"
        remaining = CONTENT_DEDUP_MAX_TOTAL_BYTES - used
        if remaining <= 0:
            raise SecurePathError("total_size_limit", f"Plugin {budget} read budget exhausted.")
        try:
            with SecureRoot(self.root) as secure_root:
                raw, _metadata = secure_root.read_bytes(
                    Path(*rel.parts), min(max_bytes, remaining), allow_hardlinks=allow_hardlinks
                )
        except SecurePathError as exc:
            if exc.code == "file_size_limit" and remaining < max_bytes:
                raise SecurePathError(
                    "total_size_limit", f"Plugin {budget} read budget exhausted.", relative_path=rel.as_posix()
                ) from exc
            raise
        if config:
            self.config_bytes_read += len(raw)
        else:
            self.bytes_read += len(raw)
        return raw

    def read_text(self, rel: PurePosixPath, max_bytes: int, *, config: bool = False) -> str:
        """Bounded, anchored, no-follow UTF-8 read; raises :class:`SecurePathError`.

        ``config`` charges the read to the separate config budget.
        """
        raw = self._read_bytes(rel, max_bytes, config=config)
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SecurePathError(
                "invalid_text_encoding", f"File is not valid UTF-8: {rel.as_posix()}", relative_path=rel.as_posix()
            ) from exc

    def read_script_bytes(self, rel: PurePosixPath, max_bytes: int) -> bytes:
        """Whole, bounded, anchored, no-follow read of a script a hook runs; hard links are allowed.

        A hook script is only scanned for evidence (no file content leaves the
        hook risk analyzer), so a hard-linked file, such as a ``node_modules``
        file pnpm links to its store, is read like any other regular file. Links
        anywhere in the path, special files, a file that changes while it is
        read, a file over ``max_bytes``, and an exhausted read budget raise
        :class:`SecurePathError`.
        """
        return self._read_bytes(rel, max_bytes, allow_hardlinks=True)

    def list_files(self, rel_dir: PurePosixPath, *, suffixes: tuple[str, ...] | None = None) -> list[PurePosixPath]:
        """Securely list regular files below a contained directory (raises on links).

        Nothing is pruned: a client loads a nested component folder whatever its
        name, so ``commands/evals/`` and ``agents/node_modules/`` are listed too,
        although the Tier 1 whole-tree scans skip them (see
        :func:`_in_unscanned_folder`).
        """
        start = self.root if str(rel_dir) == "." else self.root / rel_dir.as_posix()

        def _selected(relative: Path) -> bool:
            return suffixes is None or relative.name.lower().endswith(suffixes)

        files = discover_secure_files(
            start,
            selected=_selected,
            max_paths=CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
            allow_context_alias=False,
            refuse_selected_dirs=False,
        )
        base = PurePosixPath() if str(rel_dir) == "." else rel_dir
        return [base / file.relative_path.as_posix() for file in files]


# --------------------------------------------------------------------------- #
# Path findings                                                               #
# --------------------------------------------------------------------------- #
def _plugin_finding(
    severity: Severity,
    check_name: str,
    message: str,
    file_path: str,
    suggestion: str,
    *,
    category: str = PLUGIN_CATEGORY,
    metadata: dict[str, Any] | None = None,
) -> Finding:
    return Finding(
        category=category,
        severity=severity,
        check_name=check_name,
        message=message,
        file_path=file_path,
        suggestion=suggestion,
        metadata=metadata or {},
    )


def _path_problem_finding(
    reader: PluginRootReader,
    field_name: str,
    declared: DeclaredPath,
    manifest_rel: str,
    problem: str,
    rel: PurePosixPath | None = None,
    *,
    reference: str = CLAUDE_PROFILE.reference,
) -> Finding:
    """HIGH finding for a declared component path that is missing, escapes, or is unsafe."""
    raw = declared.raw
    # Lets the inventory attribute the finding to the (broken) component it names.
    metadata = {"plugin_component_ref": raw}
    if problem == "escape":
        return _plugin_finding(
            Severity.HIGH,
            "plugin_component_path_escape",
            f"'{field_name}' path {raw!r} is absolute or escapes the plugin root",
            reader.display(manifest_rel),
            "Use a './'-relative path that stays inside the plugin root (no '..', no absolute or home paths).",
            metadata=metadata,
        )
    if problem == "missing":
        return _plugin_finding(
            Severity.HIGH,
            "plugin_component_path_missing",
            f"'{field_name}' path {raw!r} does not exist in the plugin",
            reader.display(manifest_rel),
            "Ship the referenced file/directory inside the plugin root or remove the declaration.",
            metadata=metadata,
        )
    if problem == "unsafe":
        return _plugin_finding(
            Severity.HIGH,
            "plugin_component_path_unsafe",
            f"'{field_name}' path {raw!r} is (or passes through) a symlink, hard link, or special file; "
            "it was not followed",
            reader.display(rel if rel is not None else manifest_rel),
            "Replace links with regular files and directories contained in the plugin root.",
            metadata=metadata,
        )
    if declared.problem == "placeholder":
        return _plugin_finding(
            Severity.HIGH,
            "plugin_component_path_invalid",
            f"'{field_name}' path {raw!r} starts with a variable. Clients expand root variables such as "
            "${CLAUDE_PLUGIN_ROOT} only in hook commands and MCP server fields, not in manifest component paths, "
            "so the client rejects or ignores this path",
            reader.display(manifest_rel),
            f"Write the path relative to the plugin root with a leading './', as the {reference} requires.",
            metadata=metadata,
        )
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_invalid",
        f"'{field_name}' entry {raw!r} is not a valid component path for this field",
        reader.display(manifest_rel),
        f"Fix the '{field_name}' value to match the {reference}.",
        metadata=metadata,
    )


# The folders named in plugin_component_path_unscanned messages, and the advice they give.
_UNSCANNED_FOLDERS_TEXT = "evals/, results/, versions/, node_modules/, .venv/, .git/, __pycache__/"
_UNSCANNED_SUGGESTION = "Move the component out of dependency, VCS, evaluation-output, and version-snapshot folders."


def _unscanned_folder(rel: PurePosixPath) -> str | None:
    """The first folder of a plugin-root-relative path that Tier 1 whole-tree scans skip, or ``None``.

    The security, secret, and Unicode scans prune every
    :data:`~skillevaluator.constants.SCAN_EXCLUDED_DIRS` name at any depth:
    ``evals/``, ``results/``, and ``versions/`` (and their dotted forms),
    ``node_modules/``, ``.venv/``, ``.git/``, and ``__pycache__/``. Only an
    ``evals``/``results``/``versions`` folder at the first level of
    ``skills/`` (``skills/evals/``) is scanned anyway, because bundled-skill
    discovery scans it as a skill.
    """
    parts = rel.parts
    for index, part in enumerate(parts):
        if part not in SCAN_EXCLUDED_DIRS:
            continue
        if part in SCAN_ARTIFACT_DIRS and index == 1 and parts[0] == DEFAULT_SKILLS_DIR:
            continue
        return part
    return None


def _in_unscanned_folder(rel: PurePosixPath) -> bool:
    """Whether a plugin-root-relative path is inside a folder that Tier 1 whole-tree scans skip."""
    return _unscanned_folder(rel) is not None


def _unscanned_path_finding(
    reader: PluginRootReader, field_name: str, declared: DeclaredPath, manifest_rel: str
) -> Finding | None:
    """HIGH when a declared component lives in a folder that Tier 1 whole-tree scans skip.

    The security, secret, and Unicode scans prune evaluation output and
    version snapshots (``evals/``, ``results/``, ``versions/``), dependency
    folders (``node_modules/``, ``.venv/``), VCS metadata (``.git/``), and
    bytecode caches (:func:`_unscanned_folder`). A component the manifest
    loads from there would never be scanned.
    """
    rel = declared.rel
    folder = _unscanned_folder(rel) if rel is not None else None
    if folder is None:
        return None
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_unscanned",
        f"'{field_name}' path {declared.raw!r} is inside '{folder}/', a folder that Tier 1 whole-tree scans skip "
        f"({_UNSCANNED_FOLDERS_TEXT}), so the client loads files that are never security-scanned",
        reader.display(manifest_rel),
        _UNSCANNED_SUGGESTION,
        metadata={"plugin_component_ref": declared.raw},
    )


def _unscanned_packaged_finding(
    reader: PluginRootReader, component_type: str, rel: PurePosixPath, folder_rel: PurePosixPath
) -> Finding | None:
    """HIGH for a component file a client loads from a folder the scans skip, below the folder it lists.

    Clients load nested component folders whatever their name: Claude Code
    loads ``agents/``, ``commands/``, and ``output-styles/`` at any depth, so
    ``agents/evals/x.md`` is a live subagent even though the whole-tree scans
    prune ``evals/`` (see :meth:`PluginRootReader.list_files`). A listed folder
    that is itself inside such a folder already has its declared-path finding
    (:func:`_unscanned_path_finding`), so its files get no second one.
    """
    folder = _unscanned_folder(rel)
    if folder is None or _in_unscanned_folder(folder_rel):
        return None
    label = component_type.replace("_", " ")
    return _plugin_finding(
        Severity.HIGH,
        "plugin_component_path_unscanned",
        f"{label} '{rel.as_posix()}' is inside '{folder}/', a folder that Tier 1 whole-tree scans skip "
        f"({_UNSCANNED_FOLDERS_TEXT}). The client loads it from '{folder_rel.as_posix()}/', but it is never "
        "security-scanned",
        reader.display(rel),
        _UNSCANNED_SUGGESTION,
        metadata={"plugin_component_ref": rel.as_posix()},
    )


def _style_finding(
    reader: PluginRootReader,
    field_name: str,
    declared: DeclaredPath,
    manifest_rel: str,
    *,
    profile: FormatProfile = CLAUDE_PROFILE,
) -> Finding:
    """The finding for a declared path a client does not accept as written (no leading ``./``, or ``./`` alone).

    Its severity follows the client: Claude Code rejects the whole manifest
    (HIGH, the plugin does not load); Codex drops the value and loads the
    format's default location instead (MEDIUM).
    """
    if declared.rel is not None and str(declared.rel) == "." and declared.dot_relative:
        problem = "names the plugin root itself"
        fix = "Name the component folder or file, for example './skills/'."
    else:
        problem = "does not start with './'"
        fix = f"Write the path as './{declared.rel.as_posix() if declared.rel else declared.raw}'."
    if profile.rejects_undotted_paths:
        severity = Severity.HIGH
        outcome = f"{profile.label.removesuffix(' plugin')} rejects the whole manifest and does not load the plugin"
    else:
        severity = Severity.MEDIUM
        outcome = f"the {profile.label} loader ignores this value and loads the default location instead"
    return _plugin_finding(
        severity,
        "plugin_component_path_style",
        f"'{field_name}' path {declared.raw!r} {problem}; {outcome}",
        reader.display(manifest_rel),
        fix,
        metadata={"plugin_component_ref": declared.raw},
    )


def _root_variable_finding(
    reader: PluginRootReader, field_name: str, declared: DeclaredPath, manifest_rel: str, profile: FormatProfile
) -> Finding:
    """MEDIUM for a component path that starts with a root variable the client documents only elsewhere (Cursor)."""
    return _plugin_finding(
        Severity.MEDIUM,
        "plugin_component_path_root_variable",
        f"'{field_name}' path {declared.raw!r} starts with {declared.root_variable}. The {profile.label} reference "
        "documents root variables for hook commands and MCP server fields, not for manifest component paths, so "
        "the client may not expand it there and may not load this component. SkillEvaluator still checks the files "
        "it names",
        reader.display(manifest_rel),
        f"Write the path relative to the plugin root, for example './{declared.rel.as_posix() if declared.rel else ''}'.",
        metadata={"plugin_component_ref": declared.raw},
    )


# --------------------------------------------------------------------------- #
# Folders the whole-tree scans skip                                           #
# --------------------------------------------------------------------------- #
# Dependency folders and VCS metadata a plugin may ship that Tier 1 whole-tree
# scans skip. Their presence is noted (LOW) so a reader knows what was not
# scanned. A .git folder at the plugin root is the checkout's own metadata.
_NOTED_UNSCANNED_DIRS = frozenset({"node_modules", ".venv", ".git"})
_MAX_UNSCANNED_LINK_FINDINGS = 20
# A virtual environment links its interpreter to the system Python by design.
_VENV_INTERPRETER_RE = re.compile(r"^(?:python|pypy)(?:\d+(?:\.\d+)?)?(?:\.exe)?$")


@dataclass
class _PrunedWalk:
    noted: list[PurePosixPath] = field(default_factory=list)
    escaping_links: list[PurePosixPath] = field(default_factory=list)
    complete: bool = True


def _link_leaves_root(real_root: str, directory: str, name: str) -> bool:
    """Whether a link (or reparse point) leads outside the plugin root; unresolvable links count as outside.

    An absolute target is outside: it names a host path, not a plugin file.
    A relative target is resolved through every link on the way
    (``os.path.realpath``; targets are never read), so a chain such as
    ``node_modules/up -> ..`` and ``node_modules/evil -> up/../outside.md``
    is caught, and compared with the resolved root (``real_root``). A link
    loop or a path that cannot be resolved counts as outside; a dangling link
    whose missing target would be inside the root does not.
    """
    link = os.path.join(directory, name)  # noqa: PTH118 - plain string path
    try:
        target = os.readlink(link)  # noqa: PTH115 - reads the link text only
    except (OSError, ValueError):
        return True
    if not target or PurePosixPath(target).is_absolute() or _WINDOWS_DRIVE_RE.match(target) or target[0] in "/\\":
        return True
    try:
        resolved = os.path.realpath(link, strict=True)
    except (FileNotFoundError, NotADirectoryError):
        resolved = os.path.realpath(link)  # dangling: resolve as far as the existing parts go
    except (OSError, ValueError, RuntimeError):
        return True  # a loop, or a part that cannot be inspected
    return resolved != real_root and not resolved.startswith(real_root.rstrip(os.sep) + os.sep)


def _is_venv_interpreter(directory: Path, name: str) -> bool:
    """A ``python*`` link in the ``bin/`` (``Scripts/``) folder of a virtual environment (next to ``pyvenv.cfg``)."""
    if directory.name not in {"bin", "Scripts"} or not _VENV_INTERPRETER_RE.match(name):
        return False
    try:
        return stat.S_ISREG((directory.parent / "pyvenv.cfg").lstat().st_mode)
    except OSError:
        return False


def _walk_pruned_folders(root: Path) -> _PrunedWalk:
    """Names-only walk that finds the folders the whole-tree walk prunes and the links inside them.

    The whole-tree verification (``verify_plugin_tree``) refuses every link
    outside these folders and never enters them. Inside them, a link that
    leads out of the plugin root is recorded; links that stay inside (for
    example ``node_modules/.bin`` entries) and a virtual environment's
    interpreter links are not. Links are never followed, and the walk is
    bounded like the whole-tree walk.
    """
    walk = _PrunedWalk()
    root_path = Path(os.path.abspath(os.fspath(root)))  # noqa: PTH100 - lexical, never resolved
    real_root = os.path.realpath(root_path)  # only to compare where links lead
    budgets = {False: PLUGIN_TREE_MAX_DISCOVERED_PATHS, True: PLUGIN_TREE_MAX_DISCOVERED_PATHS}
    # (directory, root-relative path, inside a pruned folder)
    pending: list[tuple[Path, PurePosixPath, bool]] = [(root_path, PurePosixPath(), False)]
    while pending:
        directory, rel_dir, pruned = pending.pop()
        try:
            if stat_is_link_or_reparse(directory.lstat()):
                continue
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError:
            walk.complete = False
            continue
        for entry in entries:
            budgets[pruned] -= 1
            if budgets[pruned] < 0:
                walk.complete = False
                break
            rel = rel_dir / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat_is_link_or_reparse(metadata):
                if (
                    pruned
                    and not _is_venv_interpreter(directory, entry.name)
                    and _link_leaves_root(real_root, str(directory), entry.name)
                ):
                    walk.escaping_links.append(rel)
                continue
            if not stat.S_ISDIR(metadata.st_mode) or len(rel.parts) >= MAX_SECURE_DIRECTORY_DEPTH:
                continue
            enters_pruned = not pruned and entry.name in PLUGIN_TREE_PRUNED_DIRS
            if enters_pruned and entry.name == ".git" and not rel_dir.parts:
                continue  # the checkout's own metadata: no client loads it, and it can be large
            if enters_pruned and entry.name in _NOTED_UNSCANNED_DIRS:
                walk.noted.append(rel)
            pending.append((Path(entry.path), rel, pruned or enters_pruned))
    return walk


def _pruned_folder_findings(reader: PluginRootReader) -> list[Finding]:
    """Say which shipped folders Tier 1 does not scan, and refuse links out of the plugin inside them."""
    walk = _walk_pruned_folders(reader.root)
    findings: list[Finding] = []
    for rel in walk.escaping_links[:_MAX_UNSCANNED_LINK_FINDINGS]:
        folder = _unscanned_folder(rel) or rel.parts[0]
        findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_unscanned_folder_link",
                f"'{rel.as_posix()}' is a symlink or reparse point that leads outside the plugin root, inside "
                f"'{folder}/', a folder Tier 1 whole-tree scans skip. A client that runs or reads it uses content "
                "SkillEvaluator never checked",
                reader.display(rel),
                "Replace the link with the regular file or folder it points to, inside the plugin root, or remove it.",
                metadata={"path": rel.as_posix()},
            )
        )
    if len(walk.escaping_links) > _MAX_UNSCANNED_LINK_FINDINGS:
        findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_unscanned_folder_link",
                f"{len(walk.escaping_links)} links lead outside the plugin root from folders Tier 1 scans skip; only "
                f"the first {_MAX_UNSCANNED_LINK_FINDINGS} are listed",
                reader.display("."),
                "Replace links with regular files and folders inside the plugin root.",
            )
        )
    if walk.noted:
        shown = ", ".join(f"'{rel.as_posix()}/'" for rel in walk.noted[:10])
        more = f" and {len(walk.noted) - 10} more" if len(walk.noted) > 10 else ""
        findings.append(
            _plugin_finding(
                Severity.LOW,
                "plugin_unscanned_folders",
                f"Tier 1 whole-tree security, secret, and Unicode scans skip {shown}{more} (dependency folders and "
                "VCS metadata). Their files are not scanned; a declared component or MCP server that loads from "
                "them is reported separately",
                reader.display("."),
                "Do not ship dependency folders; let the client install pinned dependencies, or vendor reviewed "
                "code outside these folders.",
                metadata={"paths": [rel.as_posix() for rel in walk.noted[:10]]},
            )
        )
    if not walk.complete:
        findings.append(
            _plugin_finding(
                Severity.LOW,
                "plugin_unscanned_folder_scan_incomplete",
                f"the link check of folders Tier 1 scans skip stopped early (more than "
                f"{PLUGIN_TREE_MAX_DISCOVERED_PATHS} entries, or a folder could not be listed); later entries were "
                "not checked for links",
                reader.display("."),
                "Keep dependency folders and generated output out of the plugin package.",
            )
        )
    return findings
