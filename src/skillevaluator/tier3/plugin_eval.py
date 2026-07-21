# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for evaluating ``agent_plugin.yaml`` plugin manifests.

The Tier 3 runner evaluates skill directories. Plugin evaluation prepares a
temporary skill-shaped package from an agent plugin manifest, then reuses the
normal Harbor-backed live evaluation path.

Public offline scope
--------------------
A plugin is a *bundle-reference* artifact: ``skills.refs`` / ``rules.refs`` are
canonical remote references (``source: github|git``) and ``mcp`` entries may be
provider-scoped. SkillEvaluator does **not** fetch remote
references -- that deferred "bundle-reference resolution" is a later phase.

What Phase 1 *can* evaluate locally, without any network:

* **Contained skills** physically bundled under ``<plugin>/skills`` -- discovered
  with the shared, symlink-safe :func:`find_bundled_plugin_skills` (the same
  discovery Tier 1/2 use), plus any local skills the caller supplies via
  ``include_skills`` (the ``--include-skills`` escape hatch).
* **Contained rule files** that resolve to a real file *inside* the plugin root
  (symlink-contained) -- embedded into the with-plugin wrapper so they are
  actually exercised.
* **Runnable MCP servers** declared with a ``command``/``url`` (a documented
  local-testing extension). These are staged **with-plugin-only** via
  ``plugin_mcp_servers.toml`` so they never leak into the without-plugin
  baseline (which would invalidate lift).

Anything that only resolves remotely is recorded as *unresolved* and named in the
report rather than silently mis-resolved to a local path or scored as a pass. If
a plugin has **no** locally-resolvable component at all, preparation returns a
skipped package (an honest optional-skip; the caller exits 0 without a run).
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from skillevaluator.constants import (
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    SCAN_EXCLUDED_DIRS,
)
from skillevaluator.deduplication.plugin.ref_utils import normalize_ref
from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS, find_eval_file, load_dataset_entries
from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line
from skillevaluator.utils.helpers import find_bundled_plugin_skills, resolve_git_remote_url

# Shared with Harbor's runtime find_evals_file() and the report loader so a
# dataset accepted/staged here is resolvable downstream (MR !29 review 59316232).
_EVAL_DATASET_NAMES = tuple(f"evals{extension}" for extension in DATASET_EXTENSIONS)

# Canonical dependency-ref sources that Phase 1 cannot resolve offline. These
# mirror ``PluginSelector.source`` in :mod:`skillevaluator.models.plugin`.
_REMOTE_REF_SOURCES = frozenset({"github", "git"})

# Repo-root content dirs a canonical ref's <kind> segment may name, per resolution
# kind. normalize_ref uses the ref's FIRST path segment as <kind>, so real
# bundle-reference layouts carry ref_kind "team-skills"/"team-rules" (e.g.
# team-skills/<team>/<plugin>/<skill>), while the simplified fixture layout carries
# "skills"/"rules". Resolution and containment use the ref's OWN content root, so a
# ref can only reach a recognized content dir -- never .git/, secrets/, or a sibling.
_CONTENT_ROOTS: dict[str, tuple[str, ...]] = {
    "skills": ("skills", "team-skills"),
    "rules": ("rules", "team-rules"),
}

# Filename for the plugin's own runnable MCP servers. Kept distinct from the
# task-environment ``mcp_servers.toml`` so the adapter can stage it for the
# with-plugin arm only (see ``adapter.generate_harbor_tasks``).
PLUGIN_MCP_SERVERS_FILENAME = "plugin_mcp_servers.toml"


@dataclass(frozen=True)
class PluginEvalPackage:
    """A prepared plugin package ready for ``EvaluationService.evaluate``.

    When ``skipped`` is True the plugin had nothing locally evaluable in Phase 1;
    ``package_path`` is ``None`` and the caller should optional-skip (exit 0).
    """

    plugin_name: str
    package_path: Path | None
    include_skills: tuple[Path, ...]
    unresolved_mcp_servers: tuple[str, ...]
    runnable_mcp_servers: tuple[str, ...]
    rule_refs: tuple[str, ...]
    staged_rules: tuple[str, ...] = ()
    unresolved_skill_refs: tuple[str, ...] = ()
    unresolved_rule_refs: tuple[str, ...] = ()
    mcp_unsupported_config: tuple[str, ...] = ()
    skipped: bool = False
    skip_reason: str | None = None

    def provenance(self) -> dict[str, Any]:
        """Durable record of what a plugin run did and did NOT evaluate.

        Distinguishes a PARTIAL run (some declared components deferred as
        unresolvable remote refs / provider-only MCP that contribute nothing to
        the run) from a full one. Persisted into the agent_eval payload and a
        run-dir sidecar so it survives the temp package cleanup, instead of only
        living in the deleted ``plugin-eval-metadata.json`` (MR !29 review
        59316231).
        """
        unresolved_skill = list(self.unresolved_skill_refs)
        unresolved_rule = list(self.unresolved_rule_refs)
        provider_only_mcp = list(self.unresolved_mcp_servers)
        mcp_unsupported_config = list(self.mcp_unsupported_config)
        return {
            "plugin_name": self.plugin_name,
            "evaluated_member_skills": [path.name for path in self.include_skills],
            "staged_rules": list(self.staged_rules),
            "runnable_mcp_servers": list(self.runnable_mcp_servers),
            "unresolved_skill_refs": unresolved_skill,
            "unresolved_rule_refs": unresolved_rule,
            "provider_only_mcp_servers": provider_only_mcp,
            "mcp_unsupported_config": mcp_unsupported_config,
            "partial": bool(unresolved_skill or unresolved_rule or provider_only_mcp or mcp_unsupported_config),
        }


def _stage_agent_plugin_manifest(
    dest: Path, manifest_path: Path, manifest: dict[str, Any], *, contained_form: bool
) -> None:
    """Write the staged ``agent_plugin.yaml`` for the eval package.

    A bundle-reference manifest is copied verbatim. A *contained* manifest is
    ``.claude-plugin/plugin.json``, whose ``skills``/``rules`` are directory
    pointers (e.g. ``"./skills/"``) rather than the canonical ref LISTS the
    ``agent_plugin.yaml`` schema expects. Copying that JSON verbatim would stage
    a file whose ``skills`` is a bare string; no current consumer re-reads the
    staged manifest, but a future one calling :func:`_iter_raw_refs` on it would
    hit ``ValueError: refs must be a list``. So for contained plugins we stage a
    normalized YAML that drops those string directory-pointers -- keeping the
    file honest YAML (contained skills are discovered from ``skills/`` on disk,
    not from a ref list).
    """
    if not contained_form:
        shutil.copy2(manifest_path, dest)
        return
    normalized = {
        key: value for key, value in manifest.items() if key not in {"skills", "rules"} or isinstance(value, list)
    }
    dest.write_text(yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8", newline="\n")


def prepare_plugin_eval_package(
    plugin_path: Path,
    *,
    stage_root: Path,
    evals_source: Path | None = None,
    include_skills: tuple[Path, ...] = (),
    repo_root: Path | None = None,
) -> PluginEvalPackage:
    """Materialize an ``agent_plugin.yaml`` as a skill-shaped evaluation target.

    Args:
        plugin_path: Plugin directory or direct path to ``agent_plugin.yaml``.
        stage_root: Temporary directory under which the package is written.
        evals_source: Optional explicit workflow eval source. May point to an
            evals directory, a skill/plugin directory containing ``evals/``, or
            a single supported dataset file.
        include_skills: Additional local skill directories supplied by the caller
            (the ``--include-skills`` escape hatch for refs Phase 1 cannot fetch).

    Returns:
        Prepared package metadata. If nothing is locally evaluable, a package
        with ``skipped=True`` and ``package_path=None``.

    Raises:
        ValueError: If the manifest is malformed, or local components exist but
            no eval dataset/task source can be found.
    """
    from skillevaluator.cli_core import resolve_plugin_path

    manifest_path = _manifest_path(plugin_path)
    contained_form = _is_contained_manifest(manifest_path)
    # Share the CLI's one root-normalization helper (resolve_plugin_path) so a
    # direct contained manifest (.claude-plugin/plugin.json) anchors at <plugin> --
    # identical to the write root evaluate-plugin resolves and the root view/compare
    # read (MR !52 review). A plugin-directory input is returned unchanged.
    plugin_dir = resolve_plugin_path(plugin_path)
    plugin_root = plugin_dir.resolve()
    manifest = _load_manifest(manifest_path)
    plugin_name = _plugin_name(manifest, plugin_dir)

    # Layer-1 intra-repo resolver: canonical skill/rule refs whose <repo> is the
    # plugin's own clone are resolved to real dirs/files under the clone root
    # (widened, slug-verified containment); everything else stays unresolved.
    resolver = _make_intra_repo_resolver(plugin_dir, plugin_root, repo_root)

    # Contained skills: symlink-safe discovery shared with Tier 1/2, plus any
    # caller-supplied local skills, plus intra-repo-resolved bundle skill refs.
    # Canonical refs to OTHER repos are never treated as paths.
    contained_skills = tuple(path.resolve() for path in find_bundled_plugin_skills(plugin_dir))
    extra_skills = tuple(dict.fromkeys(path.expanduser().resolve() for path in include_skills))
    # Track WHICH canonical skill refs actually resolved intra-repo, keyed by the
    # EXACT canonical ref (not basename), so a foreign same-basename ref from a
    # different repo is never silently covered by a sibling repo's resolution
    # (fail-open: a resolved `alpha` must not cover `other/repo::skills::alpha`).
    intra_repo_skill_paths: list[Path] = []
    resolved_skill_refs: set[str] = set()
    if not contained_form:
        for ref in _iter_raw_refs(manifest.get("skills")):
            resolved = resolver.resolve_skill(ref)
            if resolved is None:
                continue
            intra_repo_skill_paths.append(resolved)
            canonical = normalize_ref(ref)
            if canonical:
                resolved_skill_refs.add(canonical)
    intra_repo_skills = tuple(intra_repo_skill_paths)
    member_skills = tuple(dict.fromkeys((*contained_skills, *extra_skills, *intra_repo_skills)))
    local_skill_names = {path.name for path in member_skills}

    # Contained plugins bundle their skills under skills/ (discovered above); the
    # manifest 'skills' key is a directory pointer (e.g. "./skills/"), not a
    # canonical ref list, so there are no unresolved remote skill refs. For bundle-
    # reference plugins a ref is "covered" when a local component (contained,
    # --include-skills, or intra-repo-resolved above) carries its trailing name.
    unresolved_skill_refs: tuple[str, ...] = (
        ()
        if contained_form
        else _unresolved_refs(
            manifest.get("skills"),
            covered_names=local_skill_names,
            resolved_refs=resolved_skill_refs,
        )
    )

    # A contained manifest may express 'rules' as a directory pointer (e.g.
    # "./rules/") rather than a canonical ref list. That string must not reach
    # ref-parsing (_iter_raw_refs would raise "refs must be a list"); instead, like
    # contained skills (discovered from skills/ on disk), contained rule files are
    # discovered from <plugin>/rules/ and staged so they are actually exercised --
    # honoring the contained-plugin contract rather than silently dropping them.
    # Bundle-reference plugins resolve their refs as before. MR !52 review.
    rules_section = manifest.get("rules")
    if contained_form and not isinstance(rules_section, list):
        contained_rules = _discover_contained_rule_files(plugin_root)
        staged_rules = tuple(contained_rules)
        unresolved_rule_refs = ()
        all_rule_refs = tuple(rule.name for rule in contained_rules)
    else:
        staged_rules, unresolved_rule_refs, all_rule_refs = _resolve_rules(
            rules_section, plugin_dir, plugin_root, resolver
        )
    runnable_mcp, provider_mcp, mcp_unsupported_config = _split_mcp_servers(manifest)

    # Optional-skip: nothing to evaluate locally in Phase 1. Honest skip rather
    # than a with-plugin run identical to baseline (a meaningless zero lift).
    if not (member_skills or staged_rules or runnable_mcp):
        return PluginEvalPackage(
            plugin_name=plugin_name,
            package_path=None,
            include_skills=(),
            unresolved_mcp_servers=tuple(server["name"] for server in provider_mcp),
            runnable_mcp_servers=(),
            rule_refs=tuple(all_rule_refs),
            unresolved_skill_refs=unresolved_skill_refs,
            unresolved_rule_refs=unresolved_rule_refs,
            mcp_unsupported_config=tuple(mcp_unsupported_config),
            skipped=True,
            skip_reason=_skip_reason(unresolved_skill_refs, unresolved_rule_refs, provider_mcp),
        )

    package_path = _fresh_package_dir(stage_root, plugin_name)
    _stage_agent_plugin_manifest(
        package_path / "agent_plugin.yaml", manifest_path, manifest, contained_form=contained_form
    )
    _write_plugin_skill_md(
        package_path / "SKILL.md",
        manifest=manifest,
        plugin_name=plugin_name,
        include_skills=member_skills,
        staged_rules=staged_rules,
        unresolved_skill_refs=unresolved_skill_refs,
        unresolved_rule_refs=unresolved_rule_refs,
        provider_mcp_servers=tuple(server["name"] for server in provider_mcp),
    )

    metadata = {
        "plugin_name": plugin_name,
        "manifest_path": str(manifest_path.resolve()),
        "member_skills": [str(path) for path in member_skills],
        "rule_refs": list(all_rule_refs),
        "staged_rules": [rule.name for rule in staged_rules],
        "unresolved_skill_refs": list(unresolved_skill_refs),
        "unresolved_rule_refs": list(unresolved_rule_refs),
        "runnable_mcp_servers": runnable_mcp,
        "provider_mcp_servers": provider_mcp,
    }
    (package_path / "plugin-eval-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    evals_dir = package_path / "evals"
    resolved_source = _resolve_evals_source(plugin_dir, evals_source)
    if resolved_source is not None:
        _copy_evals_source(resolved_source, evals_dir)
    else:
        _write_combined_member_evals(evals_dir, member_skills, plugin_name=plugin_name)

    _write_plugin_mcp_servers_toml(evals_dir, runnable_mcp)
    return PluginEvalPackage(
        plugin_name=plugin_name,
        package_path=package_path,
        include_skills=member_skills,
        unresolved_mcp_servers=tuple(server["name"] for server in provider_mcp),
        runnable_mcp_servers=tuple(server["name"] for server in runnable_mcp),
        mcp_unsupported_config=tuple(mcp_unsupported_config),
        rule_refs=tuple(all_rule_refs),
        staged_rules=tuple(rule.name for rule in staged_rules),
        unresolved_skill_refs=unresolved_skill_refs,
        unresolved_rule_refs=unresolved_rule_refs,
    )


def write_plugin_provenance(run_dir: Path, provenance: dict[str, Any]) -> Path | None:
    """Persist plugin provenance next to the run so it survives temp cleanup.

    Writes ``plugin_provenance.json`` into the durable run directory (best
    effort). Complements the copy embedded in the agent_eval payload, so even the
    standalone ``evaluate-plugin`` path (which builds no report payload) leaves a
    durable record of a partial run (MR !29 review 59316231).
    """
    try:
        run_path = Path(run_dir)
        if not run_path.is_dir():
            return None
        target = run_path / "plugin_provenance.json"
        target.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        return target
    except OSError:
        return None


def _is_contained_manifest(path: Path) -> bool:
    """Whether *path* is a contained-plugin manifest (``.claude-plugin/plugin.json``).

    Mirrors the Tier 1 detection (``cli_core._is_contained_plugin_manifest``) so
    the plugin-eval path accepts exactly the manifest forms Tier 1 does.
    """
    return path.name == PLUGIN_CONTAINED_MANIFEST_FILE and path.parent.name == PLUGIN_CONTAINED_MANIFEST_DIR


def _manifest_path(plugin_path: Path) -> Path:
    candidate = plugin_path.expanduser().resolve()
    if candidate.is_file():
        if candidate.name in {"agent_plugin.yaml", "agent_plugin.yml"} or _is_contained_manifest(candidate):
            return candidate
        raise ValueError(f"Plugin manifest must be agent_plugin.yaml/.yml or .claude-plugin/plugin.json: {candidate}")
    # Directory: the bundle-reference manifest wins (matches Tier 1 precedence),
    # then the contained .claude-plugin/plugin.json form.
    for name in ("agent_plugin.yaml", "agent_plugin.yml"):
        manifest = candidate / name
        if manifest.exists():
            return manifest.resolve()
    contained = candidate / PLUGIN_CONTAINED_MANIFEST_DIR / PLUGIN_CONTAINED_MANIFEST_FILE
    if contained.exists():
        return contained.resolve()
    raise ValueError(f"No agent_plugin.yaml or .claude-plugin/plugin.json found under {candidate}")


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a YAML mapping")
    return data


def _plugin_name(manifest: dict[str, Any], plugin_dir: Path) -> str:
    name = str(manifest.get("name") or plugin_dir.name).strip()
    if not name:
        raise ValueError("Plugin manifest requires a non-empty name")
    return name


def _find_repo_root(plugin_dir: Path) -> Path:
    for parent in [plugin_dir, *plugin_dir.parents]:
        if (parent / "plugins").exists() and any((parent / child).exists() for child in ("skills", "team-skills")):
            return parent
    if plugin_dir.parent.name == "plugins":
        return plugin_dir.parent.parent
    return plugin_dir


def _parse_canonical_ref(ref: Any) -> tuple[str, str, str, str] | None:
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


def _slug_from_remote_url(url: str) -> str | None:
    """Extract the ``<group>/<repo>`` slug from a git-remote URL.

    The sole caller (:func:`_local_repo_slug`) passes a URL that
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
    return path.strip("/") or None


def _local_repo_slug(clone_root: Path) -> str | None:
    """Best-effort ``<group>/<repo>`` slug of the clone's git origin, or ``None``."""
    url = resolve_git_remote_url(clone_root)
    return _slug_from_remote_url(url) if url else None


@dataclass(frozen=True)
class _IntraRepoResolver:
    """Layer-1 resolver for canonical refs that live in the plugin's own clone.

    A canonical ``<source>::<repo>::<kind>::<name>`` ref is resolved to a real path
    under its repo-root content dir (``<clone_root>/<ref_kind>/<name>``, where
    ``ref_kind`` is the ref's first path segment -- ``skills``/``team-skills`` for a
    skill or ``rules``/``team-rules`` for a rule) only when:

    * detection is active (there is an enclosing repo above the plugin), AND
    * the ref names a public remote source (github/git), AND
    * ``ref_kind`` is a recognized content root for the resolution kind, AND
    * the ref ``<repo>`` matches the local clone's git-origin slug -- OR no git
      origin is available, in which case path-existence under the content root is
      the sole signal, AND
    * the resolved path stays *inside* that content root.

    Containment is widened from the plugin root to the ref's content root
    (``<clone_root>/<ref_kind>``) for these slug-verified refs ONLY; symlink / ``..``
    escapes outside that content root are rejected, so a ref can only reach a
    recognized skills/rules dir. Refs to other repos are never resolved *when the
    local slug is known* (the CI path). When no git origin is available the slug
    cannot be verified, so resolution falls back to path-existence -- a documented
    fail-open in that degraded local case; see :meth:`_repo_matches`.
    """

    clone_root: Path
    local_slug: str | None
    active: bool

    def _repo_matches(self, repo: str) -> bool:
        # Fail-closed when the clone's slug is KNOWN: require an exact match so a
        # foreign-repo ref (a different <group>/<repo>) never resolves intra-repo.
        # This is the CI path -- a checked-out plugin repo has a git origin, so the
        # slug is known and cross-repo refs correctly stay unresolved -> INCOMPLETE.
        #
        # KNOWN LIMITATION (accepted, not a bug): when the slug is INDETERMINATE (no
        # git origin and no override) we fall back to path-existence under the clone
        # root, so in that degraded local case a same-named local component can
        # satisfy a foreign ref. A strictly fail-closed variant would require an
        # explicit slug source, intentionally deferred to keep the CLI surface
        # minimal. Containment (`_is_within`) still hard-gates every resolved path
        # inside the clone root, so this can never read outside the clone.
        return self.local_slug is None or repo == self.local_slug

    def _resolve(self, ref: Any, *, kind: str, want_dir: bool) -> Path | None:
        if not self.active:
            return None
        parsed = _parse_canonical_ref(ref)
        if parsed is None:
            return None
        source, repo, ref_kind, name = parsed
        # The canonical <kind> segment is the ref's REPO-ROOT content dir: skills live
        # under skills/ or team-skills/, rules under rules/ or team-rules/. A real
        # bundle-reference ref names team-skills/team-rules; the simplified fixture
        # layout names skills/rules. Both are accepted; a ref naming any other content
        # root (e.g. ``private``, ``.git``) is rejected here.
        if (
            source not in _REMOTE_REF_SOURCES
            or ref_kind not in _CONTENT_ROOTS.get(kind, ())
            or not self._repo_matches(repo)
        ):
            return None
        rel = Path(name)
        # Reject absolute names and any '..' traversal so a ref can never climb out of
        # its content root (e.g. ``team-rules::../private/credential.txt``). Legitimate
        # nested names (``team-skills::l4e/l4e-bringup/<skill>``) are preserved.
        if rel.is_absolute() or ".." in rel.parts:
            return None
        # Resolve under the ref's OWN content root (<clone_root>/<ref_kind>/<name>), so
        # real team-skills/ and team-rules/ repo-relative layouts resolve -- not just a
        # fixed <clone_root>/<kind> dir.
        content_root = (self.clone_root / ref_kind).resolve()
        try:
            resolved = (content_root / rel).resolve()
        except OSError:
            return None
        # Containment: the resolved path must stay under its content root (itself under
        # the clone root). ``_is_within`` resolves symlinks, so a component symlinked or
        # '..'-ed outside its content root is rejected -- a ref can only reach a
        # recognized skills/rules content dir, never .git/, secrets/, or a sibling.
        if not _is_within(resolved, content_root):
            return None
        if want_dir:
            if not resolved.is_dir():
                return None
            return resolved if (resolved / "SKILL.md").is_file() or (resolved / "skill.md").is_file() else None
        return resolved if resolved.is_file() else None

    def resolve_skill(self, ref: Any) -> Path | None:
        """Resolve an intra-repo ``skills`` ref to a local skill directory."""
        return self._resolve(ref, kind="skills", want_dir=True)

    def resolve_rule(self, ref: Any) -> Path | None:
        """Resolve an intra-repo ``rules`` ref to a local rule file."""
        return self._resolve(ref, kind="rules", want_dir=False)


def _make_intra_repo_resolver(plugin_dir: Path, plugin_root: Path, repo_root: Path | None) -> _IntraRepoResolver:
    """Build the intra-repo resolver, honoring an optional ``--repo-root`` override.

    ``repo_root`` (CLI ``--repo-root``) is a determinism override for CI; when it
    does not actually contain the plugin it is ignored in favor of the layout
    heuristic. Resolution is inactive when the clone root equals the plugin root
    (a standalone plugin with no enclosing repo to resolve into).
    """
    clone_root = _find_repo_root(plugin_dir).resolve()
    if repo_root is not None:
        override = repo_root.expanduser().resolve()
        if _is_within(plugin_root, override):
            clone_root = override
    # Compare RESOLVED paths on both sides: clone_root is already resolved, so a
    # symlinked standalone plugin_root must be resolved too, else `active` wrongly
    # turns True and activates the resolver with no slug (fail-open). (Greptile P1)
    active = clone_root != plugin_root.resolve()
    local_slug = _local_repo_slug(clone_root) if active else None
    return _IntraRepoResolver(clone_root=clone_root, local_slug=local_slug, active=active)


def _iter_raw_refs(section: Any) -> list[Any]:
    """Return the raw ref entries (str or mapping) for a dependency section."""
    if not section:
        return []
    refs = section.get("refs", section) if isinstance(section, dict) else section
    if refs is None:
        return []
    if not isinstance(refs, list):
        raise ValueError("Plugin manifest refs must be a list")
    return refs


def _ref_source(ref: Any) -> str | None:
    """Return the source system of a dependency ref, or ``None``."""
    if isinstance(ref, str):
        segments = ref.split("::")
        return segments[0].strip() if len(segments) >= 2 else None
    if isinstance(ref, dict):
        return str(ref.get("source") or "").strip() or None
    return None


def _ref_name(ref: Any) -> str | None:
    """Return the trailing resource name of a dependency ref, or ``None``."""
    if isinstance(ref, str):
        tail = ref.split("::")[-1] if "::" in ref else ref
        name = tail.strip().split("/")[-1].strip()
        return name or None
    if isinstance(ref, dict):
        path = str(ref.get("path") or "").strip()
        if path:
            return path.split("/")[-1].strip() or None
    return None


def _ref_label(ref: Any) -> str:
    """Return a stable, human-readable label for reporting an unresolved ref."""
    canonical = normalize_ref(ref)
    if canonical:
        return canonical
    return _ref_name(ref) or repr(ref)


def _unresolved_refs(
    section: Any,
    *,
    covered_names: set[str],
    resolved_refs: frozenset[str] | set[str] = frozenset(),
) -> tuple[str, ...]:
    """Return labels for refs that Phase 1 cannot resolve to a local component.

    A ref is covered when EITHER its exact canonical ref resolved intra-repo
    (``resolved_refs``), OR a local component (contained skill / ``--include-skills``)
    carries its trailing name AND that name is declared only once in this section.
    A basename shared by 2+ declared refs is AMBIGUOUS: a bare local component has
    no repo identity, so it cannot satisfy a *specific* repo's ref -- such refs stay
    unresolved unless exactly resolved, closing the fail-open where a resolved
    ``alpha`` covered a foreign ``other/repo::skills::alpha``.
    """
    raw_refs = _iter_raw_refs(section)
    name_counts: dict[str, int] = {}
    for ref in raw_refs:
        name = _ref_name(ref)
        if name:
            name_counts[name] = name_counts.get(name, 0) + 1
    labels: list[str] = []
    for ref in raw_refs:
        canonical = normalize_ref(ref)
        if canonical and canonical in resolved_refs:
            continue
        name = _ref_name(ref)
        if name and name in covered_names and name_counts.get(name, 0) == 1:
            continue
        labels.append(_ref_label(ref))
    return tuple(labels)


def _resolve_rules(
    section: Any, plugin_dir: Path, plugin_root: Path, resolver: _IntraRepoResolver
) -> tuple[tuple[Path, ...], tuple[str, ...], tuple[str, ...]]:
    """Resolve rule refs to contained files; report remote/unresolved ones.

    Returns ``(staged_rule_files, unresolved_labels, all_labels)``. A rule file is
    staged when it resolves to a real file inside the plugin root (symlink-
    contained, mirroring :func:`find_bundled_plugin_skills`) OR when a canonical
    remote ref names *this* clone and resolves intra-repo under the clone root.
    """
    staged: list[Path] = []
    unresolved: list[str] = []
    all_labels: list[str] = []
    seen: set[Path] = set()
    for ref in _iter_raw_refs(section):
        label = _ref_label(ref)
        all_labels.append(label)
        if _ref_source(ref) in _REMOTE_REF_SOURCES:
            # A remote rule ref whose <repo> is this clone resolves intra-repo to a
            # real file under the clone root; otherwise it stays unresolved.
            intra = resolver.resolve_rule(ref)
            if intra is None:
                unresolved.append(label)
            elif intra not in seen:
                staged.append(intra)
                seen.add(intra)
            continue
        resolved = _resolve_contained_file(ref, plugin_dir, plugin_root)
        if resolved is not None and resolved not in seen:
            staged.append(resolved)
            seen.add(resolved)
        elif resolved is None:
            unresolved.append(label)
    return tuple(staged), tuple(unresolved), tuple(all_labels)


def _resolve_contained_file(ref: Any, plugin_dir: Path, plugin_root: Path) -> Path | None:
    """Resolve a path-like ref to a file contained within the plugin root."""
    path_str = ref if isinstance(ref, str) else (ref.get("path") if isinstance(ref, dict) else None)
    if not path_str or "::" in str(path_str):
        return None
    path = Path(str(path_str))
    bases = [plugin_dir]
    repo_root = _find_repo_root(plugin_dir)
    if repo_root != plugin_dir:
        bases.append(repo_root)
    for base in bases:
        candidate = path if path.is_absolute() else base / path
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and _is_within(resolved, plugin_root):
            return resolved
    return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root)
    except OSError:
        return False


def _discover_contained_rule_files(plugin_root: Path) -> list[Path]:
    """Discover rule files bundled under ``<plugin_root>/rules`` for a contained
    plugin whose manifest expresses ``rules`` as a directory pointer ("./rules/").

    Mirrors ``find_bundled_plugin_skills``: only real files whose resolved path
    stays inside the plugin root are returned (symlink-escape safe), so a ``rules``
    symlink cannot capture a host file. Sorted for deterministic staging.
    """
    rules_root = plugin_root / "rules"
    if not rules_root.is_dir():
        return []
    discovered: list[Path] = []
    for entry in sorted(rules_root.rglob("*")):
        if any(part in SCAN_EXCLUDED_DIRS for part in entry.relative_to(rules_root).parts):
            continue
        try:
            resolved = entry.resolve()
        except OSError:
            continue
        if resolved.is_file() and _is_within(resolved, plugin_root):
            discovered.append(resolved)
    return discovered


def _reject_unsafe_mcp_declaration(name: Any, config: dict[str, Any]) -> None:
    """Fail closed before a runnable MCP declaration reaches Harbor.

    The direct Tier 3 command does not run Tier 1 first. Reuse the same network-free
    declaration policy here so shell smuggling, insecure endpoints, inline secrets,
    malformed transports, and other blocking findings can never be executed merely
    because the caller selected Tier 3 directly.
    """
    from skillevaluator.validators.mcp_static import validate_mcp_server_declaration

    blocking = [
        finding
        for finding in validate_mcp_server_declaration(name, config, "<plugin manifest>")
        if finding.severity.value in {"critical", "high"}
    ]
    if blocking:
        checks = ", ".join(dict.fromkeys(finding.check_name for finding in blocking))
        raise ValueError(
            f"Plugin manifest MCP server {str(name).strip()!r} failed static safety validation "
            f"({checks}); refusing to stage or execute it."
        )


def _normalize_mcp_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize both manifest MCP forms into the bundle-reference list shape.

    Bundle-reference ``agent_plugin.yaml`` uses a top-level ``mcp`` *list* of
    ``{name, provider}`` / ``{name, command|url, transport}`` objects. A standard
    contained ``.claude-plugin/plugin.json`` uses a top-level ``mcpServers`` *map*
    (name -> config). Both are flattened to the list shape that
    :func:`_split_mcp_servers` (and ``_write_plugin_mcp_servers_toml``) expect, so a
    contained MCP-only plugin is recognized instead of silently optional-skipped.
    MR !52 review.
    """
    raw_servers = manifest.get("mcp")
    if raw_servers:
        if not isinstance(raw_servers, list):
            raise ValueError("Plugin manifest mcp must be a list")
        return list(raw_servers)

    mcp_servers = manifest.get("mcpServers")
    if not mcp_servers:
        return []
    if not isinstance(mcp_servers, dict):
        raise ValueError("Plugin manifest mcpServers must be an object")

    normalized: list[dict[str, Any]] = []
    for name, config in mcp_servers.items():
        if not isinstance(config, dict):
            raise ValueError(f"Plugin manifest mcpServers[{name!r}] must be an object")
        # Fail closed: a raw inline credential must never be flattened into the
        # persisted toml (only ${ENV} references may reach the artifact).
        _reject_unsafe_mcp_declaration(name, config)
        entry: dict[str, Any] = {"name": str(name).strip()}
        # env/headers are declared config the eval runtime cannot apply (Harbor's
        # per-server MCPServerConfig has no such field). Record their presence so a
        # server evaluated WITHOUT its declared config marks the run INCOMPLETE
        # rather than reading as a faithful pass (Tier 1 also surfaces an advisory).
        unsupported_fields = [field for field in ("env", "headers") if config.get(field)]
        if unsupported_fields:
            entry["_unsupported_fields"] = unsupported_fields
        # Standard Claude stdio config: {"command": ..., "args": [...]}; remote
        # config: {"type": "sse"|"http", "url": ...}.
        if config.get("command"):
            # Keep argv STRUCTURE: command is the program; args stay a list so a
            # spaced arg (e.g. "path with spaces") is one token, not re-split. The
            # runtime (Harbor MCPServerConfig.args: list[str]) and every agent
            # adapter consume a separate args list.
            entry["command"] = str(config["command"])
            args = config.get("args")
            if args:
                entry["args"] = [str(arg) for arg in args]
            entry["transport"] = str(config.get("transport") or config.get("type") or "stdio")
        elif config.get("url"):
            entry["url"] = str(config["url"])
            transport = config.get("transport") or config.get("type")
            if transport:
                entry["transport"] = str(transport)
        else:
            # No command/url -> provider-only, so it is named as unresolved.
            entry["provider"] = str(config.get("provider") or config.get("type") or "")
        normalized.append(entry)
    return normalized


def _split_mcp_servers(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """Split MCP entries into runnable (command/url) vs provider-only.

    Canonical ``PluginMcpEntry`` entries carry ``name`` + ``provider`` and are
    *not* runnable offline (returned as provider-only, contributing nothing to
    the run). Entries with a ``command``/``url`` are a documented local-testing
    extension and are staged with-plugin-only.
    """
    raw_servers = _normalize_mcp_entries(manifest)

    runnable: list[dict[str, Any]] = []
    provider_only: list[dict[str, str]] = []
    unsupported_config: list[str] = []
    for idx, raw in enumerate(raw_servers):
        if not isinstance(raw, dict):
            raise ValueError(f"Plugin manifest mcp[{idx}] must be an object")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError(f"Plugin manifest mcp[{idx}] requires a name")
        if raw.get("command") or raw.get("url"):
            declaration = {key: value for key, value in raw.items() if key not in {"name", "_unsupported_fields"}}
            _reject_unsafe_mcp_declaration(name, declaration)
            server: dict[str, Any] = {"name": name}
            for key in ("url", "command", "transport"):
                if raw.get(key):
                    server[key] = str(raw[key])
            if raw.get("args"):
                server["args"] = [str(a) for a in raw["args"]]
            if "command" in server and "transport" not in server:
                server["transport"] = "stdio"
            runnable.append(server)
            if raw.get("_unsupported_fields"):
                unsupported_config.append(name)
        else:
            provider_only.append({"name": name, "provider": str(raw.get("provider") or "")})
    return runnable, provider_only, unsupported_config


def _skip_reason(
    unresolved_skill_refs: tuple[str, ...],
    unresolved_rule_refs: tuple[str, ...],
    provider_mcp: list[dict[str, str]],
) -> str:
    parts: list[str] = []
    if unresolved_skill_refs:
        parts.append(f"{len(unresolved_skill_refs)} remote skill ref(s)")
    if unresolved_rule_refs:
        parts.append(f"{len(unresolved_rule_refs)} remote rule ref(s)")
    if provider_mcp:
        parts.append(f"{len(provider_mcp)} provider-only MCP server(s)")
    detail = ", ".join(parts) if parts else "no declared dependencies"
    return (
        "Plugin has no locally-resolvable components to evaluate in Phase 1 "
        f"({detail}). Remote bundle-reference resolution is deferred to a later phase; "
        "bundle the skills under <plugin>/skills or pass --include-skills to evaluate now."
    )


def _fresh_package_dir(stage_root: Path, plugin_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", plugin_name).strip("-._") or "plugin"
    package_path = stage_root.expanduser().resolve() / f"{safe_name}-plugin-eval"
    if package_path.exists():
        raise ValueError(f"Plugin evaluation staging path already exists: {package_path}")
    package_path.mkdir(parents=True)
    return package_path


def _write_plugin_skill_md(
    path: Path,
    *,
    manifest: dict[str, Any],
    plugin_name: str,
    include_skills: tuple[Path, ...],
    staged_rules: tuple[Path, ...],
    unresolved_skill_refs: tuple[str, ...],
    unresolved_rule_refs: tuple[str, ...],
    provider_mcp_servers: tuple[str, ...],
) -> None:
    description = str(manifest.get("description") or f"Plugin evaluation wrapper for {plugin_name}.").strip()
    member_lines = "\n".join(f"- {skill.name}: staged as a plugin member skill." for skill in include_skills)
    if not member_lines:
        member_lines = "- No member skills were staged; evaluate the plugin wrapper, rules, and tools."

    rule_sections = "\n\n".join(_render_rule_block(rule) for rule in staged_rules)
    if not rule_sections:
        rule_sections = "- No contained rule files were staged."

    unresolved_lines = _render_unresolved(unresolved_skill_refs, unresolved_rule_refs, provider_mcp_servers)

    frontmatter = yaml.safe_dump(
        {
            "name": plugin_name,
            "description": description,
            "metadata": {"generated_by": "skillevaluator-plugin-eval"},
        },
        sort_keys=False,
    ).strip()
    content = f"""---
{frontmatter}
---

# {plugin_name}

This is a generated plugin evaluation wrapper. The plugin member skills are
staged alongside this wrapper during the with-plugin Harbor run. Route each task
to the most relevant member skill and follow that member skill's `SKILL.md`.

## Member Skills

{member_lines}

## Plugin Rules

{rule_sections}

## Unresolved / Deferred Dependencies

{unresolved_lines}
"""
    path.write_text(content, encoding="utf-8")


def _render_rule_block(rule: Path) -> str:
    try:
        body = rule.read_text(encoding="utf-8").strip()
    except OSError:
        body = "(rule file could not be read)"
    return f"### {rule.name}\n\n{body}"


def _render_unresolved(
    unresolved_skill_refs: tuple[str, ...],
    unresolved_rule_refs: tuple[str, ...],
    provider_mcp_servers: tuple[str, ...],
) -> str:
    lines: list[str] = []
    for ref in unresolved_skill_refs:
        lines.append(f"- skill (remote, deferred): {ref}")
    for ref in unresolved_rule_refs:
        lines.append(f"- rule (remote, deferred): {ref}")
    for name in provider_mcp_servers:
        lines.append(f"- MCP (provider-only, not runnable offline): {name}")
    return "\n".join(lines) or "- None."


def _resolve_evals_source(plugin_dir: Path, evals_source: Path | None) -> Path | None:
    if evals_source is not None:
        return _normalize_evals_source(evals_source)
    plugin_evals = plugin_dir / "evals"
    if plugin_evals.exists() and _contains_evals_source(plugin_evals):
        # Reject a plugin-controlled evals/ that escapes the plugin root BEFORE
        # resolving it (resolving first would erase the symlink identity and let
        # the escaped target masquerade as the source root).
        _reject_symlink_escapes(plugin_evals, plugin_dir, label="plugin evals directory")
        return plugin_evals.resolve()
    return None


def _normalize_evals_source(source: Path) -> Path:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.name not in _EVAL_DATASET_NAMES and source.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml"}:
            raise ValueError(f"Unsupported evals dataset file: {source}")
        return source
    if not source.is_dir():
        raise ValueError(f"Eval source does not exist: {source}")
    if _contains_evals_source(source):
        return source
    nested = source / "evals"
    if nested.exists() and _contains_evals_source(nested):
        return nested.resolve()
    raise ValueError(f"Eval source must contain an eval dataset or evals/harbor: {source}")


def _contains_evals_source(path: Path) -> bool:
    return any((path / name).exists() for name in _EVAL_DATASET_NAMES) or (path / "harbor").exists()


def _reject_symlink_escapes(path: Path, containment_root: Path, *, label: str) -> None:
    """Reject ``path`` (and any entry beneath it) that resolves outside ``containment_root``.

    ``shutil.copytree``/``copy2`` and ``Path.is_file()`` all DEREFERENCE
    symlinks, so a plugin-controlled ``evals/`` or member ``evals/files/*``
    symlink would otherwise capture an arbitrary readable host file into the
    generated package — and thence the task context — *before* the
    sandbox isolation boundary begins (MR !29 review 59912118).

    The boundary is an INDEPENDENTLY-resolved trusted root (the plugin dir or the
    member skill dir), never ``path`` itself: resolving the thing we are trying to
    bound would let a symlinked ``path`` adopt its own escaped target as the root.
    ``path`` itself is bounds-checked (so a symlinked ``evals``/``files`` root that
    escapes is caught) and so is every symlinked descendant, at any depth.
    """
    root_real = containment_root.resolve()

    def _escapes(candidate: Path) -> bool:
        resolved = candidate.resolve()
        return resolved != root_real and root_real not in resolved.parents

    if _escapes(path):
        raise ValueError(
            f"Refusing to stage {label}: '{path}' resolves to '{path.resolve()}', outside "
            f"its source root '{root_real}'. Symlinks that escape the source are rejected to "
            "prevent host-file capture before sandbox isolation."
        )
    if path.is_dir():
        for entry in path.rglob("*"):
            if entry.is_symlink() and _escapes(entry):
                raise ValueError(
                    f"Refusing to stage {label}: symlink '{entry}' resolves to "
                    f"'{entry.resolve()}', outside its source root '{root_real}'."
                )


def _copy_evals_source(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if source.is_file():
        # ``source`` is already normalized/resolved (see _normalize_evals_source,
        # which rejects a symlinked --evals-source before resolving), so a standalone
        # dataset file is a real file here.
        target_name = source.name if source.name in _EVAL_DATASET_NAMES else f"evals{source.suffix.lower()}"
        shutil.copy2(source, dest / target_name)
        return
    # Belt-and-suspenders: reject any symlinked descendant that escapes the (already
    # validated) source dir before copytree dereferences it. The plugin-controlled
    # roots are validated pre-resolution at their discovery sites.
    _reject_symlink_escapes(source, source, label="evals source directory")
    shutil.copytree(source, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("results", "__pycache__", ".git"))


def _write_combined_member_evals(evals_dir: Path, include_skills: tuple[Path, ...], *, plugin_name: str) -> None:
    evals_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    staged_files: dict[str, tuple[str, Path]] = {}
    for skill_dir in include_skills:
        eval_file = find_eval_file(skill_dir)
        if eval_file is None:
            continue
        skill_entries = load_dataset_entries(eval_file)
        for idx, entry in enumerate(skill_entries, start=1):
            combined = dict(entry)
            source_id = str(combined.get("id") or f"case-{idx:03d}")
            combined["id"] = _unique_eval_id(_safe_combined_eval_id(skill_dir.name, source_id), seen_ids)
            combined.setdefault("expected_skill", skill_dir.name)
            combined["plugin_eval_source_skill"] = skill_dir.name
            combined["plugin_eval_target"] = plugin_name
            entries.append(combined)
        _stage_member_files(
            eval_file.parent / "files",
            evals_dir / "files",
            skill_dir.name,
            staged_files,
            containment_root=skill_dir,
        )

    if not entries:
        raise ValueError(
            "Plugin eval requires --evals-source, plugin/evals, or at least one member skill with evals/evals.*"
        )

    (evals_dir / "evals.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _stage_member_files(
    files_dir: Path,
    dest_root: Path,
    skill_name: str,
    staged_files: dict[str, tuple[str, Path]],
    *,
    containment_root: Path,
) -> None:
    """Copy a member skill's ``evals/files`` tree, failing on cross-skill collisions.

    Combining several member skills into one dataset must not silently overwrite
    a fixture from one skill with a same-named fixture from another. On a
    genuine collision we fail fast and point at ``--evals-source``.
    """
    if not files_dir.exists():
        return
    # copy2 + is_file() dereference symlinks, so a member evals/files symlink (the
    # files/ dir itself, or an entry beneath it) could pull a host file into the
    # staged package. Bound against the member skill root, NOT files_dir itself
    # (MR !29 review 59912118).
    _reject_symlink_escapes(files_dir, containment_root, label=f"member '{skill_name}' eval files")
    for src in sorted(p for p in files_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(files_dir).as_posix()
        prior = staged_files.get(rel)
        if prior is not None and not _same_file(prior[1], src):
            raise ValueError(
                f"Plugin eval fixture collision on 'files/{rel}': member skills "
                f"'{prior[0]}' and '{skill_name}' provide different content. "
                "Author a combined dataset and pass it via --evals-source."
            )
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        staged_files[rel] = (skill_name, src)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def _safe_combined_eval_id(skill_name: str, source_id: str) -> str:
    raw = f"{skill_name}-{source_id}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._")
    return safe or "plugin-eval-case"


def _unique_eval_id(base_id: str, seen_ids: set[str]) -> str:
    candidate = base_id
    suffix = 2
    while candidate in seen_ids:
        candidate = f"{base_id}-{suffix}"
        suffix += 1
    seen_ids.add(candidate)
    return candidate


def _write_plugin_mcp_servers_toml(evals_dir: Path, servers: list[dict[str, Any]]) -> None:
    """Write the plugin's runnable MCP servers to a with-plugin-only file.

    Kept distinct from ``mcp_servers.toml`` (the shared task environment) so the
    adapter stages it for the with-plugin arm only, never the baseline.
    """
    if not servers:
        return
    env_dir = evals_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    mcp_file = env_dir / PLUGIN_MCP_SERVERS_FILENAME
    if mcp_file.exists():
        return

    lines: list[str] = []
    for server in servers:
        lines.append("[[mcp_servers]]")
        for key in ("name", "url", "command", "transport"):
            if key in server:
                raw = server[key]
                # Manifests may reference secret handles/env names; never emit a
                # raw secret. Redact known key shapes from command/url before write.
                value = redact_secrets_in_log_line(raw) if isinstance(raw, str) else raw
                lines.append(f"{key} = {json.dumps(value)}")
        args = server.get("args")
        if args:
            # Preserve argv structure as a real TOML array (a spaced arg stays one
            # token); redact each element so a secret cannot leak via args.
            redacted = [redact_secrets_in_log_line(a) if isinstance(a, str) else a for a in args]
            lines.append(f"args = {json.dumps(redacted)}")
        lines.append("")
    mcp_file.write_text("\n".join(lines), encoding="utf-8")
