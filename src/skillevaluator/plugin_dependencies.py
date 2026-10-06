# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, fail-closed resolution of bundle-reference plugin dependencies.

A bundle-reference plugin (``agent_plugin.yaml``) declares ``skills.refs`` and
``rules.refs`` as canonical ``<source>::<owner/repo>::<kind>::<name>`` references
(or equivalent ``{source, repo, path}`` selectors). This module holds the
network-free reference helpers shared by Tier 1 validation and Tier 3 plugin
staging, plus the Tier 1 classifier that labels every declared reference:

``provided``
    The reference names this repository and resolves to a component inside
    the plugin root. ``path`` is relative to the plugin root.
``referenced``
    The reference names this repository and resolves outside the plugin root.
    ``path`` is relative to the repository root.
``missing``
    The reference names this repository but nothing valid exists at its
    repository path. This is the only blocking state.
``external``
    The reference names a different repository. It is not fetched or verified.
``unresolved``
    Repository identity could not be established (no git ``origin`` at the
    repository root), the reference is not canonical, or its target could only
    be reached through a link. Advisory only; never gates.

Repository identity fails closed: a reference is compared with the local
repository only when the repository root is the git top-level and its
``origin`` remote yields an ``<owner>/<repo>`` slug. Path existence alone never
proves identity. Every filesystem probe inspects each path component with
``lstat`` and refuses symlinks and reparse points, so resolution cannot follow a
link out of the repository root. Probes only test existence and type; they
never read or copy content.

This module must stay importable in the base (Tier 1) install: it must not
import :mod:`skillevaluator.tier3`.
"""

from __future__ import annotations

import re
import stat
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from skillevaluator.constants import SKILL_MANIFEST_VARIANTS
from skillevaluator.deduplication.plugin.ref_utils import normalize_ref
from skillevaluator.utils.helpers import resolve_git_root
from skillevaluator.utils.secure_fs import stat_is_link_or_reparse
from skillevaluator.utils.structured_data import require_bounded_string

# Canonical dependency-ref sources. These mirror ``PluginSelector.source`` in
# :mod:`skillevaluator.models.plugin`.
REMOTE_REF_SOURCES = frozenset({"github", "git"})

# Repo-root content dirs a canonical ref's <kind> segment may name, per resolution
# kind. normalize_ref uses the ref's FIRST path segment as <kind>, so real
# bundle-reference layouts carry ref_kind "team-skills"/"team-rules" (e.g.
# team-skills/<team>/<plugin>/<skill>), while the simplified fixture layout carries
# "skills"/"rules". Resolution and containment use the ref's OWN content root, so a
# ref can only reach a recognized content dir -- never .git/, secrets/, or a sibling.
CONTENT_ROOTS: dict[str, tuple[str, ...]] = {
    "skills": ("skills", "team-skills"),
    "rules": ("rules", "team-rules"),
}

MAX_PLUGIN_MANIFEST_ITEMS = 256
MAX_PLUGIN_MANIFEST_TEXT_CHARS = 16_384
# Tier 1 classifies every declared ref, past MAX_PLUGIN_MANIFEST_ITEMS too (the
# Tier 1 YAML reader already caps a list at 1,024 items), so a ref over the
# staging limit still reaches the missing-dependency gate.
MAX_CLASSIFIED_REFS = 1_024

DEPENDENCY_STATES = ("provided", "referenced", "missing", "external", "unresolved")
# Reported ref labels are truncated so a pathological ref cannot bloat reports.
MAX_REF_LABEL_CHARS = 512


# ---------------------------------------------------------------------------
# Reference helpers (shared with Tier 3 plugin staging)
# ---------------------------------------------------------------------------


def find_repo_root(plugin_dir: Path) -> Path:
    """Return the enclosing catalog root by layout, or ``plugin_dir`` itself."""
    for parent in [plugin_dir, *plugin_dir.parents]:
        if (parent / "plugins").exists() and any((parent / child).exists() for child in ("skills", "team-skills")):
            return parent
    if plugin_dir.parent.name == "plugins":
        return plugin_dir.parent.parent
    return plugin_dir


def parse_canonical_ref(ref: Any) -> tuple[str, str, str, str] | None:
    """Parse a canonical ref into ``(source, repo, kind, name)`` or ``None``.

    Reuses the canonical-string producer :func:`normalize_ref` and splits it back
    into its four segments, so parsing never diverges from the producer and the
    :func:`~skillevaluator.models.plugin._validate_canonical_ref` validator. A ref
    that is not a confidently-parseable 4-segment canonical ID returns ``None``.
    """
    canonical = normalize_ref(ref)
    if not canonical:
        return None
    segments = canonical.split("::")
    if len(segments) != 4:
        return None
    source, repo, kind, name = (segment.strip() for segment in segments)
    if not (source and repo and kind and name):
        return None
    return source, repo, kind, name


def slug_from_remote_url(url: str) -> str | None:
    """Extract the ``<group>/<repo>`` slug from a git-remote URL.

    The sole caller (:func:`local_repo_slug`) passes a URL that
    :func:`~skillevaluator.utils.helpers.resolve_git_remote_url` has already
    normalized to HTTPS -- SSH ``ssh://`` and SCP-style (``git@host:group/repo``)
    remotes are converted by ``_ssh_to_https`` first -- so in practice this
    receives an ``https://host/group/repo[/-/tree/...]`` URL. The SCP and
    ``ssh://`` forms are nonetheless handled directly here as defense-in-depth,
    so the slug is correct no matter how the URL reaches this function (a
    standard URI would otherwise dump an SCP string verbatim into ``path``).
    """
    text = url.strip()
    if "://" in text:
        path = urlparse(text).path
    else:
        # SCP-style SSH shorthand ([user@]host:group/repo(.git)) is not a URI, so
        # take the segment after the first ':' when the string looks like one.
        scp = re.match(r"^[^/@]+@[^/:]+:(?P<path>.+)$", text)
        path = scp.group("path") if scp else text
    path = path.strip("/")
    if "/-/" in path:  # strip GitLab web suffixes like '/-/tree/main'
        path = path.split("/-/", 1)[0]
    path = path.removesuffix(".git")
    slug = path.strip("/")
    # Canonical plugin refs are normalized to lowercase. Git hosting treats the
    # owner/repository portion case-insensitively, so normalize the remote slug
    # the same way before comparing identities.
    return slug.lower() or None


# Remote URL forms whose path names a hosted <owner>/<repo>: https, http, ssh,
# git, and SCP-style ``[user@]host:owner/repo``. A local path or file:// URL
# names no hosted repository, so it never proves identity.
_HOSTED_REMOTE_RE = re.compile(
    r"^(?:(?:https?|ssh|git(?:\+ssh)?)://[^/]+/.+|(?:[^/@:\s]+@[^/:\s]+|[^/@:\s]+\.[^/@:\s]+):(?!/).+)$",
    re.IGNORECASE,
)


def origin_remote_url(clone_root: Path) -> str | None:
    """The raw ``origin`` URL of the git repository at ``clone_root``, or ``None`` when there is none."""
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=str(clone_root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return url or None


def _identity_of(clone_root: Path) -> tuple[str | None, str | None]:
    """``(slug, None)`` when ``clone_root`` is a git top-level with a hosted ``origin``, else ``(None, reason)``.

    The reason names the actual problem, so the advice fits: not a git
    repository, not its top-level, no ``origin`` remote, or an ``origin`` that
    is a local path. An ``http://`` origin names its repository as well as an
    ``https://`` one does, so it proves identity too.
    """
    name = clone_root.name or str(clone_root)
    try:
        git_root = resolve_git_root(clone_root)
    except (OSError, RuntimeError):
        git_root = None
    if git_root is None:
        return None, (
            f"'{name}' is not inside a git repository, so same-repository references cannot be told apart "
            "from external ones (validate from the plugin's git clone, or pass --repo-root <git top-level>)"
        )
    if git_root != clone_root.resolve():
        return None, (
            f"'{name}' is not the git top-level (the top-level is '{git_root.name}'), so repository-relative "
            f"references would resolve against the wrong folder (pass --repo-root <the '{git_root.name}' folder>)"
        )
    url = origin_remote_url(clone_root)
    if url is None:
        return None, (
            f"the git repository '{name}' has no 'origin' remote, so its <owner>/<repo> is unknown "
            "(add one with 'git remote add origin <url>', or validate from a clone that has it)"
        )
    if not _HOSTED_REMOTE_RE.match(url):
        return None, (
            f"the 'origin' remote of '{name}' is a local path, not a hosted <owner>/<repo>, so it cannot be "
            "compared with references (validate from a clone of the hosted repository)"
        )
    slug = slug_from_remote_url(url)
    if not slug or "/" not in slug:
        return None, f"the 'origin' remote of '{name}' does not name an <owner>/<repo>"
    return slug, None


def local_repo_slug(clone_root: Path) -> str | None:
    """Return the ``<group>/<repo>`` slug of ``clone_root``'s git origin, or ``None``.

    Fails closed: the slug is trusted only when ``clone_root`` is itself the git
    top-level. For a subdirectory the remote URL would carry a browse-path
    suffix and, more importantly, repository-relative refs would be resolved
    against the wrong base.
    """
    return _identity_of(clone_root)[0]


def iter_raw_refs(section: Any, *, limit: int | None = MAX_PLUGIN_MANIFEST_ITEMS) -> list[Any]:
    """Return the raw ref entries (str or mapping) for a dependency section.

    ``limit`` is the staging limit (Tier 3 refuses more refs). Tier 1 passes
    ``None`` so every ref is classified up to :data:`MAX_CLASSIFIED_REFS`, and
    reports the excess as a finding instead of skipping the gate.
    """
    if not section:
        return []
    refs = section.get("refs", section) if isinstance(section, dict) else section
    if refs is None:
        return []
    if not isinstance(refs, list):
        raise ValueError("Plugin manifest refs must be a list")
    cap = MAX_CLASSIFIED_REFS if limit is None else limit
    if len(refs) > cap:
        raise ValueError(f"Plugin manifest refs exceed the {cap}-item limit")
    return refs


def ref_source(ref: Any) -> str | None:
    """Return the source system of a dependency ref, or ``None``."""
    if isinstance(ref, str):
        segments = ref.split("::")
        return segments[0].strip() if len(segments) >= 2 else None
    if isinstance(ref, dict):
        source = ref.get("source")
        if source is None:
            return None
        return (
            require_bounded_string(
                source,
                "Plugin reference source",
                max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
            ).strip()
            or None
        )
    return None


def ref_name(ref: Any) -> str | None:
    """Return the trailing resource name of a dependency ref, or ``None``."""
    if isinstance(ref, str):
        tail = ref.split("::")[-1] if "::" in ref else ref
        name = tail.strip().split("/")[-1].strip()
        return name or None
    if isinstance(ref, dict):
        raw_path = ref.get("path")
        if raw_path is None:
            return None
        path = require_bounded_string(
            raw_path,
            "Plugin reference path",
            max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
        ).strip()
        if path:
            return path.split("/")[-1].strip() or None
    return None


def ref_label(ref: Any) -> str:
    """Return a stable, human-readable label for reporting a ref."""
    canonical = normalize_ref(ref)
    if canonical:
        return canonical
    name = ref_name(ref)
    if name:
        return name
    raise ValueError("Plugin reference must be a canonical string or scalar selector object")


def is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves inside ``root``."""
    try:
        return path.resolve().is_relative_to(root)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Repository identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepositoryIdentity:
    """The repository a plugin lives in, as far as it can be proven offline.

    ``local_slug`` is ``None`` whenever identity is unproven; ``reason`` then
    explains why so reports can label every dependency ``unresolved``.
    """

    clone_root: Path
    local_slug: str | None
    reason: str | None = None
    repo_root_ignored: bool = False


def resolve_repository_identity(
    plugin_root: Path,
    repo_root: Path | None = None,
    *,
    slug_for: Callable[[Path], str | None] | None = None,
) -> RepositoryIdentity:
    """Establish the repository root and its ``origin`` slug for ``plugin_root``.

    Root precedence: an explicit ``repo_root`` (``--repo-root``) that contains
    the plugin, then the git top-level containing the plugin, then the catalog
    layout heuristic. The slug is trusted only when the chosen root is the git
    top-level with an ``origin`` remote that names a hosted repository. Tier 1
    and Tier 3 both call this, so both tiers resolve refs against the same root.
    ``slug_for`` replaces the slug lookup (Tier 3 passes its own seam).
    """
    plugin_real = plugin_root.expanduser().resolve()
    ignored = False
    clone_root: Path | None = None
    if repo_root is not None:
        override = repo_root.expanduser().resolve()
        if plugin_real.is_relative_to(override):
            clone_root = override
        else:
            ignored = True
    if clone_root is None:
        git_root = resolve_git_root(plugin_real)
        if git_root is not None and plugin_real.is_relative_to(git_root):
            clone_root = git_root
        else:
            clone_root = find_repo_root(plugin_real).resolve()

    if slug_for is not None:
        slug = slug_for(clone_root)
        problem = None if slug else "no <owner>/<repo> could be read from the repository's 'origin' remote"
    else:
        slug, problem = _identity_of(clone_root)
    reason = None
    if slug is None:
        reason = f"repository identity unknown: {problem}"
    return RepositoryIdentity(clone_root=clone_root, local_slug=slug, reason=reason, repo_root_ignored=ignored)


# ---------------------------------------------------------------------------
# No-follow probing and classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Probe:
    outcome: str  # present | absent | wrong_type | link | error
    detail: str = ""


def _lstat(path: Path) -> tuple[Any, str | None]:
    try:
        return path.lstat(), None
    except (FileNotFoundError, NotADirectoryError):
        return None, None
    except OSError as exc:
        return None, str(exc)


def _exact_entry(parent: Path, name: str, listings: dict[Path, frozenset[str] | None]) -> str | None:
    """``None`` when ``parent`` holds an entry spelled exactly ``name``, else the entry that differs only in case.

    On a case-insensitive filesystem ``lstat`` finds ``Skills/helper`` for
    ``skills/helper``; a case-sensitive filesystem (and a client running there)
    does not. Comparing the exact spelling makes the result the same on both.
    """
    if parent not in listings:
        try:
            listings[parent] = frozenset(entry.name for entry in parent.iterdir())
        except OSError:
            listings[parent] = None
    entries = listings[parent]
    if entries is None or name in entries:
        return None
    lowered = name.casefold()
    return next((entry for entry in sorted(entries) if entry.casefold() == lowered), "")


def probe_repository_path(
    clone_root: Path,
    relative: Path,
    *,
    want_skill_dir: bool,
    listings: dict[Path, frozenset[str] | None] | None = None,
) -> _Probe:
    """Probe ``clone_root / relative`` without following any link.

    Each component is inspected with ``lstat``; a symlink or reparse point at
    any level refuses the probe, so the result can never describe a location
    outside ``clone_root``. Each component must also be spelled exactly as on
    disk, so the result does not depend on the filesystem's case rules. Skills
    must be directories holding a regular ``SKILL.md``; rules must be regular
    single-link files. The probe is a metadata-only, point-in-time label:
    nothing is read, copied, or staged on the strength of it (Tier 3 staging
    re-verifies with its own secure copy).
    """
    listings = {} if listings is None else listings
    current = clone_root
    metadata = None
    for index, part in enumerate(relative.parts):
        parent = current
        current = current / part
        so_far = Path(*relative.parts[: index + 1]).as_posix()
        metadata, error = _lstat(current)
        if error is not None:
            return _Probe("error", f"cannot inspect '{so_far}': {error}")
        if metadata is None:
            # On a case-sensitive filesystem a case-only typo is simply absent, so
            # name the other spelling here too; otherwise only macOS/Windows say why.
            other = _exact_entry(parent, part, listings)
            spelled = f" ('{other}' differs only in letter case)" if other else ""
            return _Probe("absent", f"'{so_far}' does not exist{spelled}")
        other = _exact_entry(parent, part, listings)
        if other is not None:
            spelled = f" ('{other}' differs only in letter case)" if other else ""
            return _Probe("absent", f"'{so_far}' does not exist{spelled}")
        if stat_is_link_or_reparse(metadata):
            return _Probe("link", f"'{so_far}' is a symlink or reparse point")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            return _Probe("absent", f"'{so_far}' is not a directory")
    if metadata is None:
        return _Probe("absent", "empty reference path")

    if want_skill_dir:
        if not stat.S_ISDIR(metadata.st_mode):
            return _Probe("wrong_type", f"'{relative.as_posix()}' is not a skill directory")
        for manifest_name in SKILL_MANIFEST_VARIANTS:
            manifest_metadata, error = _lstat(current / manifest_name)
            if error is not None:
                return _Probe("error", f"cannot inspect '{relative.as_posix()}/{manifest_name}': {error}")
            if manifest_metadata is None:
                continue
            if stat_is_link_or_reparse(manifest_metadata):
                return _Probe("link", f"'{relative.as_posix()}/{manifest_name}' is a symlink or reparse point")
            if stat.S_ISREG(manifest_metadata.st_mode):
                return _Probe("present")
        return _Probe("wrong_type", f"'{relative.as_posix()}' has no SKILL.md")

    if not stat.S_ISREG(metadata.st_mode):
        return _Probe("wrong_type", f"'{relative.as_posix()}' is not a regular file")
    if getattr(metadata, "st_nlink", 1) != 1:
        return _Probe("link", f"'{relative.as_posix()}' is hard-linked")
    return _Probe("present")


# Why a ref is ``unresolved`` (DependencyRow.cause); reports give each cause its own advice.
CAUSE_IDENTITY = "identity"
CAUSE_MALFORMED = "malformed"
CAUSE_SOURCE = "source"
CAUSE_UNSAFE = "unsafe"
CAUSE_CONTENT_ROOT = "content_root"
CAUSE_LINK = "link"
CAUSE_ERROR = "error"


@dataclass(frozen=True)
class DependencyRow:
    """Classification of one declared dependency reference.

    ``declared`` counts how often the manifest lists the same ref (written the
    same way or as an equivalent selector); the ref is classified and gated once.
    ``cause`` says why an ``unresolved`` ref could not be checked.
    """

    ref: str
    state: str
    path: str | None
    reason: str
    declared: int = 1
    cause: str | None = field(default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"ref": self.ref, "state": self.state, "path": self.path, "reason": self.reason}
        if self.declared > 1:
            row["declared"] = self.declared
        return row


@dataclass(frozen=True)
class DependencyResolution:
    """Per-reference classification for a bundle-reference manifest.

    ``declared_counts`` is the number of refs each section lists, duplicates
    included, so a caller can report sections over the staging limit.
    """

    skills: tuple[DependencyRow, ...]
    rules: tuple[DependencyRow, ...]
    identity: RepositoryIdentity
    declared_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def rows(self) -> tuple[DependencyRow, ...]:
        return (*self.skills, *self.rules)

    def status_counts(self) -> dict[str, int]:
        return status_counts(self.rows)

    def over_limit(self) -> dict[str, int]:
        """Sections that list more refs than Tier 3 stages (``MAX_PLUGIN_MANIFEST_ITEMS``)."""
        return {kind: count for kind, count in self.declared_counts.items() if count > MAX_PLUGIN_MANIFEST_ITEMS}

    def to_metadata(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "skills": [row.to_dict() for row in self.skills],
            "rules": [row.to_dict() for row in self.rules],
        }


def status_counts(rows: Iterable[DependencyRow]) -> dict[str, int]:
    """Count rows per state, always including every state key."""
    counts = dict.fromkeys(DEPENDENCY_STATES, 0)
    for row in rows:
        counts[row.state] += 1
    return counts


def _bundled_hint(name: str | None, bundled_by_leaf: Mapping[str, str]) -> str:
    bundled = bundled_by_leaf.get(name or "")
    if bundled is None:
        return ""
    return f"; a bundled skill '{bundled}' shares its name but is not assumed to satisfy it"


def classify_ref(
    ref: Any,
    *,
    kind: str,
    plugin_root: Path,
    identity: RepositoryIdentity,
    bundled_by_leaf: Mapping[str, str] | None = None,
    listings: dict[Path, frozenset[str] | None] | None = None,
) -> DependencyRow:
    """Classify one ``skills``/``rules`` reference (``kind``) without network access."""
    hints = bundled_by_leaf if kind == "skills" and bundled_by_leaf else {}
    try:
        label = ref_label(ref)
        leaf = ref_name(ref)
    except ValueError as exc:
        return DependencyRow(
            "<invalid reference>", "unresolved", None, f"invalid reference: {exc}", cause=CAUSE_MALFORMED
        )
    if len(label) > MAX_REF_LABEL_CHARS:
        label = label[: MAX_REF_LABEL_CHARS - 3] + "..."

    parsed = parse_canonical_ref(ref)
    if parsed is None:
        return DependencyRow(
            label,
            "unresolved",
            None,
            "not a canonical <source>::<owner/repo>::<kind>::<name> reference",
            cause=CAUSE_MALFORMED,
        )
    source, repo, ref_kind, name = parsed
    if source not in REMOTE_REF_SOURCES:
        return DependencyRow(label, "unresolved", None, f"unsupported reference source '{source}'", cause=CAUSE_SOURCE)
    if identity.local_slug is None:
        return DependencyRow(
            label,
            "unresolved",
            None,
            f"{identity.reason or 'repository identity unknown'}; advisory only{_bundled_hint(leaf, hints)}",
            cause=CAUSE_IDENTITY,
        )
    if repo != identity.local_slug:
        return DependencyRow(
            label,
            "external",
            None,
            f"references repository '{repo}', not this repository '{identity.local_slug}'; "
            f"not fetched or verified offline{_bundled_hint(leaf, hints)}",
        )

    relative_name = Path(name)
    kind_path = Path(ref_kind)
    if (
        relative_name.is_absolute()
        or relative_name.anchor
        or not relative_name.parts
        or any(part in {"", ".", ".."} for part in relative_name.parts)
        or "\\" in name
        or kind_path.anchor
        or kind_path.parts != (ref_kind,)
        or ref_kind in {".", ".."}
        or "\\" in ref_kind
    ):
        return DependencyRow(
            label, "unresolved", None, f"unsafe reference path '{ref_kind}/{name}'", cause=CAUSE_UNSAFE
        )

    relative = Path(ref_kind) / relative_name
    plugin_real = plugin_root.expanduser().resolve()
    target = identity.clone_root / relative
    # Outside the plugin root a ref may only reach a recognized content root
    # (never .git/, secrets/, ...); inside it, any bundled path is eligible.
    allowed_roots = CONTENT_ROOTS[kind]
    if ref_kind not in allowed_roots and not target.is_relative_to(plugin_real):
        return DependencyRow(
            label,
            "unresolved",
            None,
            f"'{ref_kind}' is not a recognized {kind} content root ({', '.join(allowed_roots)})",
            cause=CAUSE_CONTENT_ROOT,
        )

    probe = probe_repository_path(identity.clone_root, relative, want_skill_dir=kind == "skills", listings=listings)
    repo_path = relative.as_posix()
    if probe.outcome in {"absent", "wrong_type"}:
        component = "skill" if kind == "skills" else "rule"
        return DependencyRow(
            label,
            "missing",
            None,
            f"same-repository {component} not found at '{repo_path}' ({probe.detail}){_bundled_hint(leaf, hints)}",
        )
    if probe.outcome == "link":
        return DependencyRow(
            label,
            "unresolved",
            None,
            f"refusing to follow a link while resolving '{repo_path}': {probe.detail}; advisory only",
            cause=CAUSE_LINK,
        )
    if probe.outcome == "error":
        return DependencyRow(label, "unresolved", None, probe.detail, cause=CAUSE_ERROR)

    if target.is_relative_to(plugin_real):
        return DependencyRow(
            label,
            "provided",
            target.relative_to(plugin_real).as_posix(),
            "bundled inside the plugin root",
        )
    return DependencyRow(
        label,
        "referenced",
        repo_path,
        "resolved in this repository outside the plugin root; its content is not validated in this run",
    )


def classify_section(
    refs: Iterable[Any],
    *,
    kind: str,
    plugin_root: Path,
    identity: RepositoryIdentity,
    bundled_by_leaf: Mapping[str, str] | None = None,
    listings: dict[Path, frozenset[str] | None] | None = None,
) -> list[tuple[Any, DependencyRow]]:
    """Classify each distinct ref of one section once: ``[(first raw ref, row), ...]`` in declaration order.

    Refs that name the same dependency (the same canonical id, however written)
    share one row whose ``declared`` counts them, so a duplicate is never
    counted, gated, or reported twice.
    """
    listings = {} if listings is None else listings
    order: list[str] = []
    first: dict[str, Any] = {}
    rows: dict[str, DependencyRow] = {}
    counts: dict[str, int] = {}
    for index, ref in enumerate(refs):
        row = classify_ref(
            ref,
            kind=kind,
            plugin_root=plugin_root,
            identity=identity,
            bundled_by_leaf=bundled_by_leaf,
            listings=listings,
        )
        key = normalize_ref(ref) if isinstance(ref, str | dict) else None
        key = key or f"\0{index}"
        if key not in rows:
            order.append(key)
            first[key] = ref
            rows[key] = row
        counts[key] = counts.get(key, 0) + 1
    result: list[tuple[Any, DependencyRow]] = []
    for key in order:
        row = rows[key]
        if counts[key] > 1:
            row = DependencyRow(
                row.ref,
                row.state,
                row.path,
                f"{row.reason} (declared {counts[key]} times)",
                declared=counts[key],
                cause=row.cause,
            )
        result.append((first[key], row))
    return result


def classify_plugin_dependencies(
    manifest: Mapping[str, Any],
    plugin_root: Path,
    identity: RepositoryIdentity,
    *,
    bundled_skills: Iterable[str] = (),
    limit: int | None = MAX_PLUGIN_MANIFEST_ITEMS,
) -> DependencyResolution:
    """Classify every ``skills.refs`` / ``rules.refs`` entry of a manifest mapping.

    ``bundled_skills`` are plugin-root-relative ids (``skills/<name>``) used only
    to annotate reasons; a same-named bundled skill never changes a state.
    ``limit`` is the per-section ref limit; Tier 1 passes ``None`` to classify
    every ref (up to :data:`MAX_CLASSIFIED_REFS`) and report the excess itself.

    Raises:
        ValueError: If a refs section is not a list or exceeds the item limit.
    """
    bundled_by_leaf: dict[str, str] = {}
    for bundled in bundled_skills:
        bundled_by_leaf.setdefault(bundled.rsplit("/", 1)[-1], bundled)
    listings: dict[Path, frozenset[str] | None] = {}
    sections: dict[str, tuple[DependencyRow, ...]] = {}
    declared: dict[str, int] = {}
    for kind in ("skills", "rules"):
        refs = iter_raw_refs(manifest.get(kind), limit=limit)
        declared[kind] = len(refs)
        sections[kind] = tuple(
            row
            for _ref, row in classify_section(
                refs,
                kind=kind,
                plugin_root=plugin_root,
                identity=identity,
                bundled_by_leaf=bundled_by_leaf,
                listings=listings,
            )
        )
    return DependencyResolution(
        skills=sections["skills"], rules=sections["rules"], identity=identity, declared_counts=declared
    )


def dependency_status_counts_for_manifest(
    manifest: Mapping[str, Any],
    plugin_root: Path,
    repo_root: Path | None = None,
    *,
    identity: RepositoryIdentity | None = None,
) -> dict[str, int]:
    """Return per-state counts for a bundle-reference manifest (all zero on malformed refs).

    ``identity`` reuses an identity already resolved for the plugin (Tier 3
    passes the one its resolver uses), so counts and staging cannot disagree.
    """
    try:
        resolution = classify_plugin_dependencies(
            manifest,
            plugin_root,
            identity or resolve_repository_identity(plugin_root, repo_root),
        )
    except ValueError:
        return dict.fromkeys(DEPENDENCY_STATES, 0)
    return resolution.status_counts()


# ---------------------------------------------------------------------------
# Tier 1 gate wording (one place, so the advice fits the actual cause)
# ---------------------------------------------------------------------------

MAX_UNVERIFIED_REF_FINDINGS = 20

_CAUSE_ADVICE = {
    CAUSE_MALFORMED: (
        "Write the reference as <source>::<owner>/<repo>::<kind>::<name> (or a {source, repo, path} selector "
        "whose path is <kind>/<name>)."
    ),
    CAUSE_SOURCE: "Use a supported reference source (github or git).",
    CAUSE_UNSAFE: "Name the component by its path below its content folder, without '..', '.' or backslashes.",
    CAUSE_CONTENT_ROOT: (
        "Point the reference at skills/ or team-skills/ (skills), or rules/ or team-rules/ (rules), or at a "
        "path inside the plugin."
    ),
    CAUSE_LINK: "Replace the link with the real folder or file, so the reference can be checked without following it.",
    CAUSE_ERROR: "Make the referenced path readable, then validate again.",
}


def unverified_groups(resolution: DependencyResolution) -> list[tuple[str, list[DependencyRow], str]]:
    """Unresolved refs grouped for reporting: ``[(cause, rows, suggestion), ...]``.

    Refs unresolved because the repository identity is unknown share one cause
    and one piece of advice (about the clone or ``--repo-root``); every other
    cause is about the ref itself (its spelling, source, path, or a link), so
    the advice points at the ref, not at the git setup.
    """
    groups: dict[str, list[DependencyRow]] = {}
    for row in resolution.rows:
        if row.state == "unresolved":
            groups.setdefault(row.cause or CAUSE_ERROR, []).append(row)
    ordered: list[tuple[str, list[DependencyRow], str]] = []
    identity_rows = groups.pop(CAUSE_IDENTITY, None)
    if identity_rows:
        reason = resolution.identity.reason or "repository identity unknown"
        ordered.append((CAUSE_IDENTITY, identity_rows, reason))
    for cause, rows in groups.items():
        ordered.append((cause, rows, _CAUSE_ADVICE.get(cause, _CAUSE_ADVICE[CAUSE_ERROR])))
    return ordered


def record_unverified_dependencies(resolution: DependencyResolution, manifest_path: str, result: Any) -> None:
    """Add the Tier 1 findings for refs the missing-dependency gate could not check, and for over-limit sections.

    Each identity problem gives one MEDIUM ``plugin_dependency_unverified``
    finding (all its refs share the cause); each malformed, unsafe, linked, or
    unreadable ref gives its own, with advice about the ref. A section with more
    refs than Tier 3 stages gives a MEDIUM ``plugin_dependency_limit`` finding.
    None of these block; they keep "the gate did not run" out of the passing rows.
    """
    from skillevaluator.models.result import Finding, Severity

    for kind, count in resolution.over_limit().items():
        result.add_finding(
            Finding(
                category="PLUGIN_SCHEMA",
                severity=Severity.MEDIUM,
                check_name="plugin_dependency_limit",
                message=(
                    f"The manifest declares {count} {kind} refs. Every ref was classified here, but the component "
                    f"inventory lists only the first {MAX_PLUGIN_MANIFEST_ITEMS} and Tier 3 does not stage a "
                    f"manifest with more than {MAX_PLUGIN_MANIFEST_ITEMS} refs in one section."
                ),
                file_path=manifest_path,
                suggestion=f"Split the plugin, or keep each refs list at {MAX_PLUGIN_MANIFEST_ITEMS} entries or fewer.",
                metadata={"section": kind, "declared": count, "limit": MAX_PLUGIN_MANIFEST_ITEMS},
            )
        )
    sections = {id(row): "skills" for row in resolution.skills} | {id(row): "rules" for row in resolution.rules}
    listed = 0
    for cause, rows, advice in unverified_groups(resolution):
        if cause == CAUSE_IDENTITY:
            refs = [row.ref for row in rows]
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.MEDIUM,
                    check_name="plugin_dependency_unverified",
                    message=(f"The missing-dependency gate did not run for {len(rows)} declared ref(s): {advice}."),
                    file_path=manifest_path,
                    suggestion=(
                        "Validate from the plugin's git clone whose 'origin' remote names the hosted repository, "
                        "or pass --repo-root <git top-level>."
                    ),
                    metadata={"state": "unresolved", "cause": cause, "refs": refs[:32], "ref_count": len(refs)},
                )
            )
            continue
        for row in rows:
            if listed >= MAX_UNVERIFIED_REF_FINDINGS:
                continue
            listed += 1
            result.add_finding(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.MEDIUM,
                    check_name="plugin_dependency_unverified",
                    message=(
                        f"Declared {sections.get(id(row), 'skills')} dependency '{row.ref}' could not be checked: "
                        f"{row.reason}."
                    ),
                    file_path=manifest_path,
                    suggestion=advice,
                    metadata={
                        "ref": row.ref,
                        "state": row.state,
                        "cause": cause,
                        "section": sections.get(id(row), "skills"),
                    },
                )
            )
    remaining = sum(len(rows) for cause, rows, _advice in unverified_groups(resolution) if cause != CAUSE_IDENTITY)
    if remaining > listed:
        result.add_message(f"{remaining - listed} more unverified dependency ref(s) are not listed individually")
