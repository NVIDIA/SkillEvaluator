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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from skillevaluator.constants import SKILL_MANIFEST_VARIANTS
from skillevaluator.deduplication.plugin.ref_utils import normalize_ref
from skillevaluator.utils.helpers import git_origin_https_url, resolve_git_root
from skillevaluator.utils.secure_fs import lstat_walk, stat_is_link_or_reparse
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
    return _split_canonical(normalize_ref(ref))


def _split_canonical(canonical: str | None) -> tuple[str, str, str, str] | None:
    """Split a :func:`normalize_ref` result into ``(source, repo, kind, name)`` (see :func:`parse_canonical_ref`)."""
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

    :func:`local_repo_slug` passes a URL that
    :func:`~skillevaluator.utils.helpers.git_origin_https_url` has already
    normalized to HTTPS -- SSH ``ssh://`` and SCP-style (``git@host:group/repo``)
    remotes are converted by ``_ssh_to_https`` first -- so in practice this
    receives an ``https://host/group/repo`` URL. The SCP and
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


def local_repo_slug(clone_root: Path, *, git_root: Path | None = None) -> str | None:
    """Return the ``<group>/<repo>`` slug of ``clone_root``'s git origin, or ``None``.

    Fails closed: the slug is trusted only when ``clone_root`` is itself the git
    top-level. For a subdirectory the remote URL would carry a browse-path
    suffix and, more importantly, repository-relative refs would be resolved
    against the wrong base. A caller that already resolved the git top-level
    of ``clone_root`` passes it as ``git_root``, so only ``git remote get-url
    origin`` runs. Only ssh, SCP-style, and https origins give a slug
    (:func:`~skillevaluator.utils.helpers.git_origin_https_url`).
    """
    try:
        if git_root is None:
            git_root = resolve_git_root(clone_root)
        if git_root is None or git_root != clone_root.resolve():
            return None
    except (OSError, RuntimeError):
        return None
    url = git_origin_https_url(git_root)
    return slug_from_remote_url(url) if url else None


def iter_raw_refs(section: Any) -> list[Any]:
    """Return the raw ref entries (str or mapping) for a dependency section."""
    if not section:
        return []
    refs = section.get("refs", section) if isinstance(section, dict) else section
    if refs is None:
        return []
    if not isinstance(refs, list):
        raise ValueError("Plugin manifest refs must be a list")
    if len(refs) > MAX_PLUGIN_MANIFEST_ITEMS:
        raise ValueError(f"Plugin manifest refs exceed the {MAX_PLUGIN_MANIFEST_ITEMS}-item limit")
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
    return _ref_label_from(ref, normalize_ref(ref))


def _ref_label_from(ref: Any, canonical: str | None) -> str:
    """:func:`ref_label` of a ref whose :func:`normalize_ref` result is ``canonical``."""
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


def resolve_repository_identity(plugin_root: Path, repo_root: Path | None = None) -> RepositoryIdentity:
    """Establish the repository root and its ``origin`` slug for ``plugin_root``.

    Root precedence: an explicit ``repo_root`` (``--repo-root``) that contains
    the plugin, then the git top-level containing the plugin, then the catalog
    layout heuristic shared with Tier 3. The slug is trusted only when the
    chosen root is the git top-level with an ``origin`` remote.
    """
    plugin_real = plugin_root.expanduser().resolve()
    ignored = False
    clone_root: Path | None = None
    # The git top-level of clone_root, when it is already known.
    clone_git_root: Path | None = None
    if repo_root is not None:
        override = repo_root.expanduser().resolve()
        if plugin_real.is_relative_to(override):
            clone_root = override
        else:
            ignored = True
    if clone_root is None:
        git_root = resolve_git_root(plugin_real)
        if git_root is not None and plugin_real.is_relative_to(git_root):
            clone_root = clone_git_root = git_root
        else:
            clone_root = find_repo_root(plugin_real).resolve()

    slug = local_repo_slug(clone_root, git_root=clone_git_root)
    reason = None
    if slug is None:
        reason = (
            f"repository identity unknown: '{clone_root.name or clone_root}' is not a git top-level with an "
            "'origin' remote, so same-repository references cannot be told apart from external ones "
            "(run inside the plugin's git clone or pass --repo-root <git top-level>)"
        )
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


def probe_repository_path(clone_root: Path, relative: Path, *, want_skill_dir: bool) -> _Probe:
    """Probe ``clone_root / relative`` without following any link.

    Each component is inspected with ``lstat``; a symlink or reparse point at
    any level refuses the probe, so the result can never describe a location
    outside ``clone_root``. Skills must be directories holding a regular
    ``SKILL.md``; rules must be regular single-link files. The probe is a
    metadata-only, point-in-time label: nothing is read, copied, or staged on
    the strength of it (Tier 3 staging re-verifies with its own secure copy).
    """
    walk = lstat_walk(clone_root, relative)
    if walk.failing_index is not None:
        so_far = Path(*relative.parts[: walk.failing_index + 1]).as_posix()
        if walk.outcome == "error":
            return _Probe("error", f"cannot inspect '{so_far}': {walk.error}")
        if walk.outcome == "missing":
            return _Probe("absent", f"'{so_far}' does not exist")
        if walk.outcome == "link":
            return _Probe("link", f"'{so_far}' is a symlink or reparse point")
        return _Probe("absent", f"'{so_far}' is not a directory")
    metadata = walk.metadata
    if metadata is None:
        return _Probe("absent", "empty reference path")
    current = clone_root / relative

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


@dataclass(frozen=True)
class DependencyRow:
    """Classification of one declared dependency reference."""

    ref: str
    state: str
    path: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "state": self.state, "path": self.path, "reason": self.reason}


@dataclass(frozen=True)
class DependencyResolution:
    """Per-reference classification for a bundle-reference manifest."""

    skills: tuple[DependencyRow, ...]
    rules: tuple[DependencyRow, ...]
    identity: RepositoryIdentity

    @property
    def rows(self) -> tuple[DependencyRow, ...]:
        return (*self.skills, *self.rules)

    def status_counts(self) -> dict[str, int]:
        return status_counts(self.rows)

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
) -> DependencyRow:
    """Classify one ``skills``/``rules`` reference (``kind``) without network access."""
    return _classify_ref(
        ref,
        kind=kind,
        plugin_real=plugin_root.expanduser().resolve(),
        identity=identity,
        bundled_by_leaf=bundled_by_leaf,
    )


def _classify_ref(
    ref: Any,
    *,
    kind: str,
    plugin_real: Path,
    identity: RepositoryIdentity,
    bundled_by_leaf: Mapping[str, str] | None,
) -> DependencyRow:
    """:func:`classify_ref` for a plugin root that is already resolved (``plugin_real``)."""
    hints = bundled_by_leaf if kind == "skills" and bundled_by_leaf else {}
    canonical = normalize_ref(ref)
    try:
        label = _ref_label_from(ref, canonical)
        leaf = ref_name(ref)
    except ValueError as exc:
        return DependencyRow("<invalid reference>", "unresolved", None, f"invalid reference: {exc}")
    if len(label) > MAX_REF_LABEL_CHARS:
        label = label[: MAX_REF_LABEL_CHARS - 3] + "..."

    parsed = _split_canonical(canonical)
    if parsed is None:
        return DependencyRow(
            label, "unresolved", None, "not a canonical <source>::<owner/repo>::<kind>::<name> reference"
        )
    source, repo, ref_kind, name = parsed
    if source not in REMOTE_REF_SOURCES:
        return DependencyRow(label, "unresolved", None, f"unsupported reference source '{source}'")
    if identity.local_slug is None:
        return DependencyRow(
            label,
            "unresolved",
            None,
            f"{identity.reason or 'repository identity unknown'}; advisory only{_bundled_hint(leaf, hints)}",
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
        return DependencyRow(label, "unresolved", None, f"unsafe reference path '{ref_kind}/{name}'")

    relative = Path(ref_kind) / relative_name
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
        )

    probe = probe_repository_path(identity.clone_root, relative, want_skill_dir=kind == "skills")
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
        )
    if probe.outcome == "error":
        return DependencyRow(label, "unresolved", None, probe.detail)

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


def classify_plugin_dependencies(
    manifest: Mapping[str, Any],
    plugin_root: Path,
    identity: RepositoryIdentity,
    *,
    bundled_skills: Iterable[str] = (),
) -> DependencyResolution:
    """Classify every ``skills.refs`` / ``rules.refs`` entry of a manifest mapping.

    ``bundled_skills`` are plugin-root-relative ids (``skills/<name>``) used only
    to annotate reasons; a same-named bundled skill never changes a state.

    Raises:
        ValueError: If a refs section is not a list or exceeds the item limit.
    """
    bundled_by_leaf: dict[str, str] = {}
    for bundled in bundled_skills:
        bundled_by_leaf.setdefault(bundled.rsplit("/", 1)[-1], bundled)
    plugin_real = plugin_root.expanduser().resolve()
    sections: dict[str, tuple[DependencyRow, ...]] = {}
    for kind in ("skills", "rules"):
        sections[kind] = tuple(
            _classify_ref(
                ref,
                kind=kind,
                plugin_real=plugin_real,
                identity=identity,
                bundled_by_leaf=bundled_by_leaf,
            )
            for ref in iter_raw_refs(manifest.get(kind))
        )
    return DependencyResolution(skills=sections["skills"], rules=sections["rules"], identity=identity)


def dependency_status_counts_for_manifest(
    manifest: Mapping[str, Any],
    plugin_root: Path,
    repo_root: Path | None = None,
) -> dict[str, int]:
    """Return per-state counts for a bundle-reference manifest (all zero on malformed refs)."""
    try:
        resolution = classify_plugin_dependencies(
            manifest,
            plugin_root,
            resolve_repository_identity(plugin_root, repo_root),
        )
    except ValueError:
        return dict.fromkeys(DEPENDENCY_STATES, 0)
    return resolution.status_counts()
