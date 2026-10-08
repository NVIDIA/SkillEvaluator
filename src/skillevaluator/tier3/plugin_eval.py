# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for evaluating ``agent_plugin.yaml`` plugin manifests.

The Tier 3 runner evaluates skill directories. Plugin evaluation prepares a
temporary skill-shaped package from an agent plugin manifest, then reuses the
normal Harbor-backed live evaluation path.

Public offline scope
--------------------
A plugin is a *bundle-reference* artifact: ``skills.refs`` / ``rules.refs`` are
canonical remote references (``source: github|gitlab|git``) and ``mcp`` entries may be
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
  baseline (which would invalidate lift). Contained plugins may declare them in
  any documented ``mcpServers`` form (inline map, ``.json`` path, or an array
  of both) and in the root ``.mcp.json``; every source is read through the
  bounded, no-follow plugin-root reader and passes the Tier 1 static checks
  (fail closed) before anything is staged. A server launched from plugin files
  (``${CLAUDE_PLUGIN_ROOT}`` or a relative path) is not staged: the plugin tree
  never reaches the task environment, so it is reported unsupported and the run
  INCOMPLETE rather than counted as runnable.

Every declared and packaged component is also reported in
``provenance()['component_coverage']`` (staged / not_staged / unsupported /
unavailable / invalid) with a static ``context_cost`` estimate and the MCP
``mcp_pinning`` summary. Coverage is report-only: it does not change the
``partial`` (INCOMPLETE) semantics.

Anything that only resolves remotely is recorded as *unresolved* and named in the
report rather than silently mis-resolved to a local path or scored as a pass. If
a plugin has **no** locally-resolvable component at all, preparation returns a
skipped package (an honest optional-skip; the caller exits 0 without a run).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, NamedTuple

import yaml

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    DESCRIPTION_MAX_LENGTH,
    NAME_MAX_LENGTH,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_CURSOR_MANIFEST_TYPE,
    PLUGIN_MANIFEST_RELATIVE_PATHS,
    PLUGIN_NAME_MAX_REPORT_CHARS,
    SCAN_EXCLUDED_DIRS,
)
from skillevaluator.models.result import Severity
from skillevaluator.plugin_components import (
    MCP_JSON,
    Component,
    CostRow,
    PluginInventory,
    PluginRootReader,
    build_plugin_inventory,
    coverage_row,
    normalize_declared_path,
    parse_markdown,
    problem_reason,
    summarize_coverage,
)

# Network-free reference, identity, and bound helpers shared with Tier 1
# (``skillevaluator.plugin_dependencies`` must stay importable without Tier 3
# extras). Private aliases keep this module's historical names.
from skillevaluator.plugin_dependencies import CONTENT_ROOTS as _CONTENT_ROOTS
from skillevaluator.plugin_dependencies import (
    DEPENDENCY_STATES,
    MAX_PLUGIN_MANIFEST_ITEMS,
    MAX_PLUGIN_MANIFEST_TEXT_CHARS,
    dependency_status_counts_for_manifest,
)
from skillevaluator.plugin_dependencies import REMOTE_REF_SOURCES as _REMOTE_REF_SOURCES
from skillevaluator.plugin_dependencies import find_repo_root as _find_repo_root
from skillevaluator.plugin_dependencies import is_within as _is_within
from skillevaluator.plugin_dependencies import iter_raw_refs as _iter_raw_refs
from skillevaluator.plugin_dependencies import local_repo_slug as _local_repo_slug
from skillevaluator.plugin_dependencies import parse_canonical_ref as _parse_canonical_ref
from skillevaluator.plugin_dependencies import ref_label as _ref_label
from skillevaluator.plugin_dependencies import ref_name as _ref_name
from skillevaluator.plugin_dependencies import ref_source as _ref_source
from skillevaluator.plugin_formats import (
    CLAUDE_PROFILE,
    manifest_syntax,
    normalized_component_manifest,
    parse_manifest_text,
    profile_for,
)
from skillevaluator.plugin_mcp import mcp_pinning_summary
from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS, load_dataset_entries, normalize_dataset_entries
from skillevaluator.tier3.eval_core.plugin_signals import validate_plugin_case_fields
from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line

# The generated with-plugin files the Harbor adapter reads (and their name cap).
# PLUGIN_MCP_SERVERS_FILENAME stays distinct from the task-environment
# ``mcp_servers.toml`` so the adapter can stage it for the with-plugin arm only.
from skillevaluator.tier3.harbor.adapter import (
    MAX_PLUGIN_RUNTIME_NAMES,
    PLUGIN_MCP_SERVERS_FILENAME,
    PLUGIN_RUNTIME_COMPONENTS_FILENAME,
)
from skillevaluator.tier3.harbor.secure_copy import UnsafeStagingError, copy_file_secure, copytree_secure
from skillevaluator.tier3.plugin_native import (
    HARNESS_ADAPTERS,
    AgentLoadDecision,
    ClaudeCodeAdapter,
    NativePluginSource,
    PluginLoadError,
    adapter_for,
    apply_native_refusals,
    build_native_source,
    foreign_root_var_re,
    member_skill_hook_refusal,
    native_component_types,
    opencode_agent_name,
    plugin_root_var_names,
    resolve_plugin_load,
    to_claude_root,
)
from skillevaluator.utils.helpers import find_bundled_plugin_skills
from skillevaluator.utils.secure_fs import (
    SecurePathError,
    SecureRoot,
    discover_secure_files,
    secure_atomic_write_text,
    secure_read_path_text,
    stat_is_link_or_reparse,
)
from skillevaluator.utils.structured_data import (
    StructuredDataError,
    load_bounded_json,
    load_bounded_yaml,
    require_bounded_string,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from skillevaluator.plugin_formats import FormatProfile
    from skillevaluator.plugin_manifest import PluginManifestLocation

#: The generated package directory is ``<plugin><PLUGIN_EVAL_PACKAGE_SUFFIX>``; the Harbor
#: runner strips the suffix to recognize the wrapper skill by the plugin's name.
PLUGIN_EVAL_PACKAGE_SUFFIX = "-plugin-eval"

# Shared with Harbor's runtime find_evals_file() and the report loader so a
# dataset accepted/staged here is resolvable downstream.
_EVAL_DATASET_NAMES = tuple(f"evals{extension}" for extension in DATASET_EXTENSIONS)

# Install-time variables a harness expands when it loads an installed plugin
# (Claude Code's CLAUDE_PLUGIN_ROOT/DATA, plus each manifest format's own root
# placeholder such as ${PLUGIN_ROOT} or ${CURSOR_PLUGIN_ROOT}), and the path forms
# that point into the plugin tree. The wrapper runtime expands none of them and
# never copies the plugin tree into the task environment, so an MCP server
# launched through them cannot start there (see _launches_from_plugin_files).
_RELATIVE_PATH_PREFIXES = ("./", "../", ".\\", "..\\")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:/")
# ``${user_config.<key>}`` values Claude Code fills from the plugin's userConfig.
_USER_CONFIG_REF_RE = re.compile(r"\$\{user_config\.([A-Za-z0-9_.-]+)\}")
# Placeholder used to test whether a launch needs more than the plugin-root variable.
_ROOTED_PLACEHOLDER = "/plugin-root"


@functools.lru_cache(maxsize=16)
def _plugin_install_var_re(root_prefixes: tuple[str, ...] = ()) -> re.Pattern[str]:
    """Match a plugin-root variable, braced (``${PLUGIN_ROOT}``) or bare (``$PLUGIN_ROOT``)."""
    names = "|".join(re.escape(name) for name in plugin_root_var_names(root_prefixes))
    return re.compile(r"\$\{?(?:" + names + r")\b")


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
    staged_rules: tuple[str, ...] = ()
    unresolved_skill_refs: tuple[str, ...] = ()
    unresolved_rule_refs: tuple[str, ...] = ()
    mcp_unsupported_config: tuple[str, ...] = ()
    dataset_case_count: int = 0
    cross_component_case_count: int = 0
    dependency_status_counts: tuple[tuple[str, int], ...] = ()
    skipped: bool = False
    skip_reason: str | None = None
    # Native plugin loading (``--plugin-load native|auto``): the bounded plugin
    # snapshot the harness adapters stage for the with-plugin arm. ``None`` in
    # the default wrapper mode.
    native_source: NativePluginSource | None = dataclass_field(default=None, compare=False, hash=False, repr=False)
    # Report-only static inventory outputs (C2). ``None`` for packages built
    # without an inventory (e.g. constructed directly in tests).
    component_coverage: dict[str, Any] | None = dataclass_field(default=None, compare=False, hash=False, repr=False)
    context_cost: dict[str, Any] | None = dataclass_field(default=None, compare=False, hash=False, repr=False)
    mcp_pinning: dict[str, Any] | None = dataclass_field(default=None, compare=False, hash=False, repr=False)
    # Author-supplied URL MCP servers for the opt-in ``--probe-mcp`` host probe:
    # ``{"name", "url", "transport", "headers"}``. Header values are the declared
    # literals or ``${VAR}`` references; they are never persisted.
    mcp_probe_targets: tuple[dict[str, Any], ...] = dataclass_field(default=(), compare=False, hash=False, repr=False)

    def provenance(self) -> dict[str, Any]:
        """Durable record of what a plugin run did and did NOT evaluate.

        Distinguishes a PARTIAL run (some declared components deferred as
        unresolvable remote refs / provider-only MCP that contribute nothing to
        the run) from a full one. Persisted into the agent_eval payload and a
        run-dir sidecar so it survives the temp package cleanup, instead of only
        living only in temporary generated-package state.
        """
        unresolved_skill = list(self.unresolved_skill_refs)
        unresolved_rule = list(self.unresolved_rule_refs)
        provider_only_mcp = list(self.unresolved_mcp_servers)
        mcp_unsupported_config = list(self.mcp_unsupported_config)
        status_counts = {**dict.fromkeys(DEPENDENCY_STATES, 0), **dict(self.dependency_status_counts)}
        provenance: dict[str, Any] = {
            "plugin_name": self.plugin_name,
            "evaluated_member_skills": [path.name for path in self.include_skills],
            "staged_rules": list(self.staged_rules),
            "runnable_mcp_servers": list(self.runnable_mcp_servers),
            "unresolved_skill_refs": unresolved_skill,
            "unresolved_rule_refs": unresolved_rule,
            "provider_only_mcp_servers": provider_only_mcp,
            "mcp_unsupported_config": mcp_unsupported_config,
            "dataset_case_count": self.dataset_case_count,
            "cross_component_case_count": self.cross_component_case_count,
            "integration_evidence_ready": self.cross_component_case_count > 0,
            # A missing same-repository dependency can never be evaluated, so it alone makes the run partial.
            "partial": bool(
                unresolved_skill
                or unresolved_rule
                or provider_only_mcp
                or mcp_unsupported_config
                or status_counts.get("missing")
            ),
            # Offline classification of declared skill/rule refs (same classifier as Tier 1).
            "dependency_status_counts": status_counts,
        }
        # Report-only: unsupported component types are listed here, never gated.
        if self.component_coverage is not None:
            provenance["component_coverage"] = self.component_coverage
        if self.context_cost is not None:
            provenance["context_cost"] = self.context_cost
        if self.mcp_pinning is not None:
            provenance["mcp_pinning"] = self.mcp_pinning
        return provenance

    @property
    def incomplete_skip(self) -> bool:
        """Nothing was locally evaluable because declared components could not be resolved or run.

        Such a run is INCOMPLETE, not an optional skip: a plugin whose only
        refs are external, or whose only component is a provider-only MCP
        server, must not pass Tier 3 with exit 0.
        """
        return self.skipped and bool(self.provenance()["partial"])

    def integration_evidence_error(self) -> str | None:
        """Explain why an Integration arm would not test composition."""
        if not self.include_skills:
            return (
                "The plugin has no member skills, so Integration has no parts to compare it with "
                "(it needs member skills and a dataset case with cross_component=true naming two or more of them)"
            )
        if self.cross_component_case_count > 0:
            return None
        return (
            "Integration evaluation requires at least one dataset case with "
            "cross_component=true and two or more expected_skills that are member skills of the plugin"
        )


@dataclass(frozen=True)
class _StagedRule:
    """One bounded rule snapshot safe to embed in the generated wrapper."""

    name: str
    content: str


def _stage_agent_plugin_manifest(
    dest: Path,
    manifest_text: str,
    manifest: dict[str, Any],
    *,
    contained_form: bool,
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
        dest.write_text(manifest_text, encoding="utf-8", newline="")
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
    plugin_load: str = "wrapper",
    agents: str | Sequence[str] | None = None,
    env_mode: str | None = None,
    policy: Any = None,
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
        plugin_load: ``wrapper`` (default), ``native``, or ``auto``. Any value
            other than ``wrapper`` also snapshots the plugin for the native
            harness adapters (:mod:`skillevaluator.tier3.plugin_native`) when
            at least one agent can load it natively.
        agents: The selected agents (``-a`` value or a list; ``None`` means the
            provider default). With *env_mode* this resolves the per-agent load
            plan here, so the skip decision, the INCOMPLETE rule, and the native
            snapshot follow what each with-plugin arm really stages.
        env_mode: The run's environment mode; ``None`` leaves the plan unknown
            (every with-plugin arm is then treated like the wrapper).
        policy: The validation policy (``validate --policy`` / ``--profile``);
            ``None`` resolves the default profile, as Tier 1 does. Its
            ``severity_overrides`` and ``mcp.allowed_private_hosts`` apply to
            the MCP static checks that gate staging, so Tier 3 blocks exactly
            what Tier 1 blocks.

    Returns:
        Prepared package metadata. If nothing is locally evaluable, a package
        with ``skipped=True`` and ``package_path=None``.

    Raises:
        ValueError: If the manifest is malformed, or local components exist but
            no eval dataset/task source can be found.
    """
    if policy is None:
        from skillevaluator.validators.policy import resolve_policy

        policy = resolve_policy()
    plugin = _locate_plugin(plugin_path, allowed_private_hosts=tuple(policy.mcp_allowed_private_hosts))
    # Layer-1 intra-repo resolver: canonical skill/rule refs whose <repo> is the
    # plugin's own clone are resolved to real dirs/files under the clone root
    # (widened, slug-verified containment); everything else stays unresolved.
    # It uses Tier 1's repository identity, so both tiers resolve the same root.
    resolver = _make_intra_repo_resolver(plugin.plugin_dir, plugin.plugin_root, repo_root, stage_root)
    dependency_status_counts = (
        ()
        if plugin.contained
        else tuple(
            dependency_status_counts_for_manifest(
                plugin.manifest, plugin.plugin_dir, repo_root, identity=resolver.identity
            ).items()
        )
    )
    member_skills, unresolved_skill_refs = _resolve_member_skills(plugin, include_skills, resolver)
    _refuse_member_skill_hooks(member_skills)
    staged_rules, unresolved_rule_refs = _resolve_plugin_rules(plugin, resolver)
    mcp = _split_mcp_servers(
        plugin.component_manifest,
        plugin.inventory,
        plugin.contained,
        plugin_root=plugin.plugin_root,
        root_prefixes=plugin.profile.root_prefixes,
        policy=policy,
        manifest_type=plugin.location.manifest_type,
    )

    # Native and auto loading need the evals source before the load plan: the
    # task source it pins decides whether an agent can load the plugin natively,
    # and the native whole-plugin copy must leave it out. The wrapper resolves it
    # only when it stages the package, so a plugin with nothing locally
    # evaluable is skipped without its eval source being read or validated.
    resolved_source = _resolve_evals_source(plugin.plugin_dir, evals_source) if plugin_load != "wrapper" else None
    native_plan, native_source, staging_source = _native_snapshot(
        plugin,
        plugin_load=plugin_load,
        agents=agents,
        env_mode=env_mode,
        evals_source=resolved_source,
        member_skills=member_skills,
        staged_rules=staged_rules,
        mcp=mcp,
    )
    claude_native = _claude_native_arms(native_plan)
    user_config_defaults = _user_config_defaults(plugin.manifest)
    mcp_unsupported_config = _unsupported_mcp_for_plan(mcp, native_plan, user_config_defaults)
    skipped = not (
        member_skills or staged_rules or mcp.runnable or _native_loaded_types(plugin_load, native_plan, native_source)
    )
    mcp_coverage = _McpCoverage(
        mcp,
        unsupported=tuple(mcp_unsupported_config),
        plugin_file_agents=claude_native,
        plugin_file_gap_notes=_plugin_file_gap_notes(mcp, native_plan, user_config_defaults, mcp_unsupported_config),
    )
    report_only = _inventory_provenance(
        plugin.inventory,
        plugin_root=plugin.plugin_root,
        contained=plugin.contained,
        skipped=skipped,
        member_skills=member_skills,
        staged_rule_names=tuple(rule.name for rule in staged_rules),
        unresolved_skill_refs=unresolved_skill_refs,
        unresolved_rule_refs=unresolved_rule_refs,
        mcp=mcp_coverage,
        claude_skill_dirs=_claude_skill_dirs(native_source) if claude_native else (),
        claude_native_agents=claude_native,
        arm_staging=_arm_staging(plugin_load, native_plan, staging_source),
        cost_agents=_cost_agents(agents),
    )

    # Optional-skip: nothing to evaluate locally in Phase 1. Honest skip rather
    # than a with-plugin run identical to baseline (a meaningless zero lift).
    if skipped:
        return PluginEvalPackage(
            plugin_name=plugin.name,
            package_path=None,
            include_skills=(),
            unresolved_mcp_servers=mcp.provider_names,
            runnable_mcp_servers=(),
            unresolved_skill_refs=unresolved_skill_refs,
            unresolved_rule_refs=unresolved_rule_refs,
            mcp_unsupported_config=tuple(mcp_unsupported_config),
            dependency_status_counts=dependency_status_counts,
            skipped=True,
            skip_reason=_skip_reason(
                unresolved_skill_refs,
                unresolved_rule_refs,
                mcp.provider_only,
                [name for name, gaps in mcp.gaps.items() if gaps[0].startswith("plugin_files")],
                unrooted_mcp=[name for name, gaps in mcp.gaps.items() if gaps[0] == "plugin_files_unrooted"],
            ),
            **report_only,
        )

    package_path = _fresh_package_dir(stage_root, plugin.name)
    if plugin_load == "wrapper":
        resolved_source = _resolve_evals_source(plugin.plugin_dir, evals_source)
    dataset_cases = _stage_package_files(
        package_path,
        plugin,
        member_skills=member_skills,
        staged_rules=staged_rules,
        unresolved_skill_refs=unresolved_skill_refs,
        unresolved_rule_refs=unresolved_rule_refs,
        mcp=mcp,
        evals_source=resolved_source,
    )
    return PluginEvalPackage(
        plugin_name=plugin.name,
        native_source=native_source,
        package_path=package_path,
        include_skills=member_skills,
        unresolved_mcp_servers=mcp.provider_names,
        # A native Claude Code arm also starts the servers that launch from plugin files.
        runnable_mcp_servers=mcp.runnable_names + (mcp.plugin_file_names if claude_native else ()),
        mcp_unsupported_config=tuple(mcp_unsupported_config),
        staged_rules=tuple(rule.name for rule in staged_rules),
        unresolved_skill_refs=unresolved_skill_refs,
        unresolved_rule_refs=unresolved_rule_refs,
        dataset_case_count=len(dataset_cases),
        cross_component_case_count=_cross_component_case_count(dataset_cases, member_skills),
        dependency_status_counts=dependency_status_counts,
        mcp_probe_targets=tuple(mcp.probe_targets[:MAX_PLUGIN_MANIFEST_ITEMS]),
        **report_only,
    )


@dataclass(frozen=True)
class _LocatedPlugin:
    """The selected plugin manifest, read and parsed once, with the plugin's static inventory."""

    location: PluginManifestLocation
    manifest_text: str
    manifest: dict[str, Any]
    name: str
    description: str
    #: Every declared and packaged component (the selected manifest plus any
    #: additional manifests): the MCP declarations from every mcpServers form and
    #: default MCP file for staging, the declared skills and rules of the newer
    #: formats, and the report-only coverage / context-cost / pinning provenance.
    inventory: PluginInventory

    @property
    def contained(self) -> bool:
        return self.location.contained

    @property
    def plugin_dir(self) -> Path:
        return self.location.root

    @property
    def plugin_root(self) -> Path:
        return self.location.secure_file.root

    @property
    def manifest_rel(self) -> str:
        return self.location.manifest_filename

    @property
    def profile(self) -> FormatProfile:
        return profile_for(self.location.manifest_type)

    @property
    def stage_from_inventory(self) -> bool:
        """Whether this format's skills and rules are staged from what the inventory found."""
        return self.contained and self.profile.stage_from_inventory

    @property
    def component_manifest(self) -> dict[str, Any]:
        """The manifest's component declarations in Claude Code field names (a contained format's own view)."""
        if not self.contained:
            return self.manifest
        return normalized_component_manifest(self.location.manifest_type, self.manifest) or {}


def _locate_plugin(plugin_path: Path, *, allowed_private_hosts: tuple[str, ...] = ()) -> _LocatedPlugin:
    """Read the selected manifest and inventory the plugin; *allowed_private_hosts* is the policy's MCP allowlist."""
    location = _manifest_location(plugin_path)
    manifest_text = location.read_text()
    manifest = _load_manifest_text(manifest_text, location.path, location.manifest_type)
    name = _plugin_name(manifest, location.root)
    description = _plugin_description(manifest, name)
    inventory = build_plugin_inventory(
        location.secure_file.root,
        manifest,
        contained=location.contained,
        manifest_rel=location.manifest_filename,
        allowed_private_hosts=allowed_private_hosts,
        manifest_type=location.manifest_type,
        # Inventoried, never staged, so coverage reports what only another client loads.
        additional=location.parsed_additional(),
    )
    return _LocatedPlugin(location, manifest_text, manifest, name, description, inventory)


def _resolve_member_skills(
    plugin: _LocatedPlugin, include_skills: tuple[Path, ...], resolver: _IntraRepoResolver
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """The member skills to stage, and the skill refs that stay unresolved.

    Members are the plugin's own skills (symlink-safe discovery shared with
    Tier 1/2, or what the inventory found for the newer formats), the
    caller-supplied local skills, and the bundle skill refs Tier 1 proves
    (:func:`_resolve_skill_refs`). Canonical refs to OTHER repos are never
    treated as paths.
    """
    contained_skills = (
        _inventory_skill_dirs(plugin.inventory, plugin.plugin_root)
        if plugin.stage_from_inventory
        else tuple(path.resolve() for path in find_bundled_plugin_skills(plugin.plugin_dir))
    )
    extra_skills = tuple(dict.fromkeys(path.expanduser().resolve() for path in include_skills))
    if plugin.contained:
        # A contained manifest's 'skills' key is a directory pointer (e.g.
        # "./skills/"), not a canonical ref list, so there are no remote skill refs.
        return tuple(dict.fromkeys((*contained_skills, *extra_skills))), ()

    # A bundle-reference ref is covered only when Tier 1 would call it provided or
    # referenced, or (external/unresolved) by a same-named --include-skills
    # directory. A same-named bundled skill never covers a missing or external ref.
    intra_repo_skills, unresolved = _resolve_skill_refs(
        plugin.manifest.get("skills"),
        resolver=resolver,
        plugin_dir=plugin.plugin_dir,
        contained=contained_skills,
        included=extra_skills,
    )
    return tuple(dict.fromkeys((*contained_skills, *extra_skills, *intra_repo_skills))), unresolved


def _resolve_plugin_rules(
    plugin: _LocatedPlugin, resolver: _IntraRepoResolver
) -> tuple[tuple[_StagedRule, ...], tuple[str, ...]]:
    """The rule files to stage (read once, bounded), and the rule refs that stay unresolved.

    A contained manifest may express 'rules' as a directory pointer (e.g.
    "./rules/") rather than a canonical ref list. That string must not reach
    ref-parsing (_iter_raw_refs would raise "refs must be a list"); instead, like
    contained skills (discovered from skills/ on disk), contained rule files are
    discovered from <plugin>/rules/ and staged so they are actually exercised --
    honoring the contained-plugin contract rather than silently dropping them.
    Bundle-reference plugins resolve their refs.
    """
    rules_section = plugin.manifest.get("rules")
    if plugin.stage_from_inventory:
        return tuple(_inventory_rule_files(plugin.inventory, plugin.plugin_root)), ()
    if plugin.contained and not isinstance(rules_section, list):
        return tuple(_discover_contained_rule_files(plugin.plugin_root)), ()
    return _resolve_rules(rules_section, plugin.plugin_dir, plugin.plugin_root, resolver)


def _native_snapshot(
    plugin: _LocatedPlugin,
    *,
    plugin_load: str,
    agents: str | Sequence[str] | None,
    env_mode: str | None,
    evals_source: Path | None,
    member_skills: tuple[Path, ...],
    staged_rules: tuple[_StagedRule, ...],
    mcp: _McpSplit,
) -> tuple[dict[str, AgentLoadDecision] | None, NativePluginSource | None, NativePluginSource | None]:
    """The per-agent load plan, the plugin snapshot for the native adapters, and the snapshot the coverage reads.

    The plan comes first, and the plugin is snapshotted only when some agent
    loads it natively, so the skip decision and the INCOMPLETE rule follow what
    each with-plugin arm really stages. A component that enables a permission
    bypass then refuses (``native``) or falls back (``auto``) for the agents
    that would stage it. A wrapper run stages no native source, but its
    coverage rows still name the components that ``--plugin-load native``
    would refuse, so it reads the refusals from a snapshot it never stages
    (none when native loading cannot read the plugin: no native hint either).
    """
    plan = _preview_plugin_load_plan(plugin_load, agents, env_mode, _preview_task_source(evals_source))
    refusal_only = plugin_load == "wrapper"
    if not refusal_only and plan is not None and not any(decision.native for decision in plan.values()):
        return plan, None, None
    manifest_type = plugin.location.manifest_type
    # Newer contained formats stage through their Claude-field-name view; the
    # Claude Code and bundle-reference manifests stage as they are.
    newer_contained_format = plugin.contained and manifest_type != PLUGIN_CONTAINED_MANIFEST_TYPE
    build_source = functools.partial(
        build_native_source,
        inventory=plugin.inventory,
        manifest=plugin.component_manifest if newer_contained_format else plugin.manifest,
        plugin_root=plugin.plugin_root,
        contained=plugin.contained,
        manifest_rel=plugin.manifest_rel,
        plugin_name=plugin.name,
        description=plugin.description,
        member_skills=member_skills,
        rules=tuple((rule.name, rule.content) for rule in staged_rules),
        mcp_servers=mcp.runnable,
        plugin_file_mcp_servers=mcp.plugin_file,
        mcp_declared=mcp.declared,
        nest_flat_hooks=manifest_type == PLUGIN_CURSOR_MANIFEST_TYPE,
        hook_dialect="cursor" if manifest_type == PLUGIN_CURSOR_MANIFEST_TYPE else "claude",
        hook_root_prefixes=plugin.profile.root_prefixes,
        # An --evals-source inside the plugin root must not reach the agent
        # through the native whole-plugin copy.
        excluded_paths=_native_excluded_evals_paths(evals_source, plugin.plugin_root),
    )
    if refusal_only:
        try:
            return plan, None, build_source()
        except (ValueError, OSError):
            return plan, None, None
    source = build_source()
    return _apply_native_refusals(plugin_load, plan, source), source, source


def _stage_package_files(
    package_path: Path,
    plugin: _LocatedPlugin,
    *,
    member_skills: tuple[Path, ...],
    staged_rules: tuple[_StagedRule, ...],
    unresolved_skill_refs: tuple[str, ...],
    unresolved_rule_refs: tuple[str, ...],
    mcp: _McpSplit,
    evals_source: Path | None,
) -> list[dict[str, Any]]:
    """Write the package: the staged manifest, the wrapper ``SKILL.md``, the evals, and the generated files.

    The evals are a copy of *evals_source*, or the member skills' datasets
    combined when there is none. Returns the dataset cases, after checking
    their advisory plugin-signal fields.
    """
    _stage_agent_plugin_manifest(
        package_path / "agent_plugin.yaml",
        plugin.manifest_text,
        plugin.manifest,
        contained_form=plugin.contained,
    )
    _write_plugin_skill_md(
        package_path / "SKILL.md",
        plugin_name=plugin.name,
        plugin_description=plugin.description,
        include_skills=member_skills,
        staged_rules=staged_rules,
        unresolved_skill_refs=unresolved_skill_refs,
        unresolved_rule_refs=unresolved_rule_refs,
        provider_mcp_servers=mcp.provider_names,
    )
    evals_dir = package_path / "evals"
    if evals_source is not None:
        _copy_evals_source(evals_source, evals_dir)
    else:
        _write_combined_member_evals(evals_dir, member_skills, plugin_name=plugin.name)

    dataset_path = next((evals_dir / name for name in _EVAL_DATASET_NAMES if (evals_dir / name).exists()), None)
    if dataset_path is None and not (evals_dir / "harbor").exists():
        raise ValueError(f"Prepared plugin package has no evaluation dataset: {package_path}")
    dataset_cases = load_dataset_entries(dataset_path) if dataset_path is not None else []
    _reject_invalid_plugin_signal_fields(dataset_cases)

    _write_plugin_mcp_servers_toml(evals_dir, mcp.runnable)
    _write_plugin_runtime_components(evals_dir, plugin.inventory, plugin_name=plugin.name)
    return dataset_cases


def _cross_component_case_count(dataset_cases: list[dict[str, Any]], member_skills: tuple[Path, ...]) -> int:
    """Cases that can support an Integration claim: ``cross_component`` naming two or more member skills.

    A composition case must name two or more of the plugin's own member
    skills: names that are not members cannot be staged in the member-skills
    arm. Native Harbor sources can be valid for effectiveness without carrying
    this structured composition metadata.
    """
    member_names = {skill.name.strip().casefold() for skill in member_skills}
    return sum(
        1
        for case in dataset_cases
        if case.get("cross_component") is True
        and isinstance(case.get("expected_skills"), list)
        and len({str(name).strip().casefold() for name in case["expected_skills"]} & member_names) >= 2
    )


def _planned_agents(agents: str | Sequence[str] | None) -> list[str] | None:
    """The agents a run will use, resolved like the Tier 3 engine does; ``None`` when unknown."""
    from skillevaluator.tier3.commands import parse_agents, resolve_agents

    if isinstance(agents, str):
        return parse_agents(agents) or None
    if agents is not None:
        return [str(agent) for agent in agents] or None
    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider

    try:
        return resolve_agents(None, provider=resolve_llm_provider().provider)
    except (ProviderConfigurationError, ValueError):
        return None


def _preview_task_source(source: Path | None) -> str:
    """The task source the runner picks for the staged package.

    The staged ``evals/`` is a copy of *source*, so a ``harbor.task_source``
    pinned in its ``config.yml`` wins, as in the runner; otherwise the runner's
    ``auto`` rule applies (a dataset first, then ``evals/harbor``).
    """
    pinned = _pinned_task_source(source)
    if pinned is not None:
        return pinned
    if source is None or source.is_file() or any((source / name).exists() for name in _EVAL_DATASET_NAMES):
        return "evals_json"
    return "native_harbor" if (source / "harbor").exists() else "evals_json"


def _pinned_task_source(source: Path | None) -> str | None:
    """``harbor.task_source`` from the evals source's ``config.yml``, or ``None`` when unset or unreadable.

    An invalid config fails the run at the runner's configuration stage, so
    the preview only needs the value of a config the runner accepts.
    """
    from skillevaluator.constants import PLUGIN_CONFIG_MAX_BYTES
    from skillevaluator.tier3.evals_config import CONFIG_FILENAMES
    from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_yaml

    if source is None or not source.is_dir():
        return None
    for name in CONFIG_FILENAMES:
        path = source / name
        if not path.exists():
            continue
        try:
            # The package copy dereferences an in-source link, so read its target too
            # (a link that escapes the source fails the copy later).
            target = path.resolve() if path.is_symlink() else path
            raw = load_bounded_yaml(secure_read_path_text(target, PLUGIN_CONFIG_MAX_BYTES))
        except (SecurePathError, StructuredDataError, OSError, ValueError):
            return None
        harbor = raw.get("harbor") if isinstance(raw, dict) else None
        value = harbor.get("task_source") if isinstance(harbor, dict) else None
        return value if value in {"evals_json", "native_harbor"} else None
    return None


def _preview_plugin_load_plan(
    plugin_load: str, agents: str | Sequence[str] | None, env_mode: str | None, task_source: str
) -> dict[str, AgentLoadDecision] | None:
    """Per-agent load decisions for this run, or ``None`` when the agents or environment are unknown.

    Raises ``PluginLoadError`` for ``native`` with an agent or environment that
    cannot load natively, before anything else is checked or staged.
    """
    if plugin_load == "wrapper" or env_mode is None:
        return None
    planned = _planned_agents(agents)
    if not planned:
        return None
    return dict(resolve_plugin_load(plugin_load, planned, env_mode=env_mode, task_source=task_source))


def _apply_native_refusals(
    plugin_load: str, plan: dict[str, AgentLoadDecision] | None, source: NativePluginSource
) -> dict[str, AgentLoadDecision] | None:
    """Turn bypass refusals into errors (``native``) or wrapper fallbacks (``auto``), per agent.

    Only the component types an agent's adapter stages natively count. With an
    unknown plan, ``native`` refuses any bypass (every adapter might stage it)
    and ``auto`` leaves the decision to the runner.
    """
    if plan is None:
        if plugin_load == "native" and source.refusals:
            raise PluginLoadError(source.refusals[0][2])
        return None
    return apply_native_refusals(plugin_load, plan, source)


def _copies_tree(agent: str, decision: AgentLoadDecision) -> bool:
    """Whether this with-plugin arm loads the plugin natively through a copied plugin tree (Claude Code)."""
    adapter = adapter_for(agent)
    return decision.native and adapter is not None and adapter.copies_plugin_tree


def _claude_native_arms(plan: dict[str, AgentLoadDecision] | None) -> tuple[str, ...]:
    """The with-plugin arms that load the plugin natively through a copied plugin tree (Claude Code)."""
    return tuple(sorted(agent for agent, decision in (plan or {}).items() if _copies_tree(agent, decision)))


def _unsupported_mcp_for_plan(
    mcp: _McpSplit, plan: dict[str, AgentLoadDecision] | None, defaults: set[str]
) -> list[str]:
    """MCP servers some with-plugin arm cannot fully apply (the run is then INCOMPLETE).

    When every arm is a native Claude Code arm (the plugin tree is copied, and
    the staged ``.mcp.json`` keeps env, headers, and userConfig defaults), a
    server is unsupported only for what Claude Code cannot apply either.
    """
    if not _every_arm_claude_native(plan):
        return mcp.unsupported_config
    return [
        name for name, gaps in mcp.gaps.items() if not _claude_applies(gaps, mcp.user_config.get(name, ()), defaults)
    ]


def _every_arm_claude_native(plan: dict[str, AgentLoadDecision] | None) -> bool:
    """Whether every with-plugin arm is a native Claude Code arm (a known plan only)."""
    return bool(plan) and all(_copies_tree(agent, decision) for agent, decision in (plan or {}).items())


def _plugin_file_gap_notes(
    mcp: _McpSplit, plan: dict[str, AgentLoadDecision] | None, defaults: set[str], unsupported: Sequence[str]
) -> dict[str, str]:
    """Why each unsupported plugin-file MCP server leaves the run INCOMPLETE, from its own gaps."""
    every_arm_claude = _every_arm_claude_native(plan)
    plugin_file_names = {str(server["name"]) for server in mcp.plugin_file}
    notes: dict[str, str] = {}
    for name in unsupported:
        if name not in plugin_file_names:
            continue
        parts: list[str] = [] if every_arm_claude else ["not started in the other with-plugin arms"]
        missing = [key for key in mcp.user_config.get(name, ()) if key not in defaults]
        if missing:
            parts.append(f"its ${{user_config.*}} value(s) with no default stay unfilled ({', '.join(missing)})")
        other = [gap for gap in mcp.gaps.get(name, ()) if gap not in _CLAUDE_NATIVE_APPLIES and gap != "user_config"]
        if other:
            parts.append(f"its {', '.join(other)} config is not applied")
        notes[name] = "; ".join(parts or ["not started in the other with-plugin arms"])
    return notes


def _native_loaded_types(
    plugin_load: str, plan: dict[str, AgentLoadDecision] | None, source: NativePluginSource | None
) -> set[str]:
    """Component types some native with-plugin arm loads (with an unknown plan: any adapter under ``native``)."""
    if source is None:
        return set()
    if plan is None:
        adapters = list(HARNESS_ADAPTERS.values()) if plugin_load == "native" else []
    else:
        adapters = [adapter_for(agent) for agent, decision in plan.items() if decision.native]
    loaded: set[str] = set()
    for adapter in adapters:
        if adapter is not None:
            loaded |= native_component_types(adapter, source)
    return loaded


def _claude_skill_dirs(source: NativePluginSource | None) -> tuple[str, ...]:
    """Plugin-relative skill directories the native Claude Code plugin loads in place."""
    if source is None:
        return ()
    return tuple(rel for _name, rel, copy_from in ClaudeCodeAdapter().staged_skills(source) if copy_from is None)


def _reject_invalid_plugin_signal_fields(dataset_cases: list[dict[str, Any]]) -> None:
    """Fail fast on malformed advisory plugin-signal case fields before any agent runs."""
    problems = [
        f"case {str(case.get('id') or index)[:128]!r}: {problem}"
        for index, case in enumerate(dataset_cases)
        for problem in validate_plugin_case_fields(case)
    ]
    if problems:
        shown = "; ".join(problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        raise ValueError(f"Invalid plugin signal fields in the evaluation dataset: {shown}{more}")


_TIER3_COVERAGE_NOTE = (
    "Tier 3 wrapper: plugin rules are embedded in the generated wrapper SKILL.md and load on demand with it."
)
_WRAPPER_RULE_REASON = "rule embedded in the generated wrapper SKILL.md"


class _NativeArm(NamedTuple):
    """One native with-plugin arm: its adapter, that adapter's component modes, and what it stages here."""

    adapter_id: str
    modes: dict[str, str]
    #: The component types this arm stages natively from this plugin.
    types: frozenset[str]


@dataclass(frozen=True)
class _ArmStaging:
    """How the with-plugin arms stage the plugin, for the coverage reasons.

    ``native`` maps each native arm's agent to its :class:`_NativeArm`;
    ``wrapper`` lists the arms that load the generated wrapper (empty for a
    wrapper run, where every arm does).
    """

    native: dict[str, _NativeArm]
    wrapper: tuple[str, ...]
    #: ``(type, name)`` of components native staging refuses (a permission bypass).
    refused: frozenset[tuple[str, str]] = frozenset()
    #: False when the refusals are unknown (a wrapper run whose refusal pass could not read the plugin).
    refusals_known: bool = True

    def native_for(self, component_type: str) -> list[str]:
        return sorted(agent for agent, arm in self.native.items() if component_type in arm.types)

    def wrapper_rule_arms(self) -> list[str]:
        wrapped = [agent for agent, arm in self.native.items() if arm.modes.get("rule") == "wrapper"]
        return sorted([*self.wrapper, *wrapped])


def _arm_staging(
    plugin_load: str, plan: dict[str, AgentLoadDecision] | None, source: NativePluginSource | None
) -> _ArmStaging | None:
    """The per-arm staging for the coverage reasons; ``None`` when a native or auto plan is not known."""
    refused = frozenset((kind, name) for kind, name, _message in getattr(source, "refusals", ()) or ())
    if plugin_load == "wrapper":
        return _ArmStaging(native={}, wrapper=(), refused=refused, refusals_known=source is not None)
    if plan is None:
        return None
    native: dict[str, _NativeArm] = {}
    wrapper: list[str] = []
    for agent, decision in plan.items():
        adapter = adapter_for(agent) if decision.native else None
        if adapter is None or source is None:
            wrapper.append(agent)
            continue
        types = frozenset(native_component_types(adapter, source))
        native[agent] = _NativeArm(adapter.adapter_id, adapter.component_modes(), types)
    return _ArmStaging(native=native, wrapper=tuple(sorted(wrapper)), refused=refused)


def _rule_reason(staging: _ArmStaging | None, wrapper_reason: str = _WRAPPER_RULE_REASON) -> str:
    """Where the with-plugin arms put a staged rule: native rules, the wrapper SKILL.md, or both."""
    if staging is None:
        return wrapper_reason
    native = staging.native_for("rule")
    if not native:
        return wrapper_reason
    reason = "staged as a native rule for " + ", ".join(native)
    wrapped = staging.wrapper_rule_arms()
    if wrapped:
        reason += "; embedded in the generated wrapper SKILL.md for " + ", ".join(wrapped)
    return reason


def _natively_staged(row: dict[str, Any], agents: list[str]) -> dict[str, Any]:
    """Record on a coverage row the agents whose native arm stages it.

    The load census compares these agents, never the reason text, to tell
    whether the plan already said the row is staged natively for them.
    """
    if agents:
        row["native_agents"] = agents
    return row


def _native_capable_agents(component_type: str) -> list[str]:
    return sorted(
        agent
        for agent, adapter in HARNESS_ADAPTERS.items()
        if adapter.component_modes().get(component_type) == "native"
    )


def _refuse_member_skill_hooks(member_skills: tuple[Path, ...]) -> None:
    """Every load mode stages member skills as written, so a bypass flag in their frontmatter hooks blocks staging."""
    refusal = member_skill_hook_refusal(member_skills)
    if refusal is not None:
        raise PluginLoadError(refusal)


def _other_type_row(component: Component, staging: _ArmStaging | None, agents: Sequence[str] = ()) -> dict[str, Any]:
    """Coverage row of a hook, subagent, command, or other type the generated wrapper does not stage.

    The reason follows the resolved plan: ``staged natively for <agents>`` when
    a native arm stages the type, why each other arm does not (the wrapper, or
    that agent's native adapter), and the wrapper note when no arm is native.
    A skill's frontmatter hooks follow the run's *agents* instead: only Claude
    Code runs them, so they are ``unsupported`` when no arm is claude-code.
    """
    from skillevaluator.plugin_components import is_skill_frontmatter_hook

    kind = component.type
    if is_skill_frontmatter_hook(component):
        others = [agent for agent in agents if agent != "claude-code"]
        ignored = f"{', '.join(others)} {'ignores' if len(others) == 1 else 'ignore'} them"
        if others and len(others) == len(agents):
            return coverage_row(
                component,
                "unsupported",
                f"only Claude Code runs skill-frontmatter hooks, and no with-plugin arm is claude-code ({ignored})",
            )
        native = staging.native_for(kind) if staging is not None else []
        wrapped = (
            f"census-wrapped in the native {', '.join(native)} arm"
            if native
            else "staged without the hook census, so its runs are not counted"
        )
        if others:
            wrapped += f"; {ignored}"
        return coverage_row(
            component,
            "staged",
            f"skill-frontmatter hooks are staged with their member skill, and Claude Code runs them while the skill "
            f"is active; {wrapped}",
        )
    capable = _native_capable_agents(kind)
    if not capable:
        return coverage_row(
            component, "unsupported", f"SkillEvaluator does not stage {kind} components yet (inventoried only)"
        )
    wrapper_note = f"the generated wrapper does not stage {kind} components"
    if staging is None or not staging.native:
        hint = f"; --plugin-load native stages them for {', '.join(capable)}"
        if staging is not None and (kind, component.name) in staging.refused:
            hint = "; --plugin-load native refuses this one, because it enables a permission bypass"
        elif staging is not None and not staging.refusals_known:
            hint = ""
        return coverage_row(component, "unsupported", wrapper_note + hint)
    others = [f"{agent} ({wrapper_note})" for agent in staging.wrapper]
    for agent, arm in sorted(staging.native.items()):
        if kind in arm.types:
            continue
        if arm.modes.get(kind) == "native":
            others.append(f"{agent} (its native adapter ({arm.adapter_id}) found nothing of this type to load)")
        else:
            others.append(f"{agent} (unsupported by the {agent} native adapter ({arm.adapter_id}))")
    native = staging.native_for(kind)
    if not native:
        return coverage_row(component, "unsupported", "not staged for " + "; ".join(others))
    reason = "staged natively for " + ", ".join(native)
    if others:
        reason += "; not staged for " + "; ".join(others)
    return _natively_staged(coverage_row(component, "staged", reason), native)


_SKIPPED_PACKAGE_NOTE = "the plugin package was skipped (nothing locally evaluable)"


class _McpCoverage(NamedTuple):
    """The plugin's MCP servers as the coverage rows report them for the resolved plan."""

    split: _McpSplit
    #: Servers some with-plugin arm cannot fully apply (the run is reported INCOMPLETE).
    unsupported: tuple[str, ...]
    #: The native Claude Code arms, which start the servers that launch from plugin files.
    plugin_file_agents: tuple[str, ...]
    #: Why each unsupported plugin-file server still leaves the run INCOMPLETE.
    plugin_file_gap_notes: dict[str, str]


def _inventory_provenance(
    inventory: PluginInventory,
    *,
    plugin_root: Path,
    contained: bool,
    skipped: bool,
    member_skills: tuple[Path, ...],
    staged_rule_names: tuple[str, ...],
    unresolved_skill_refs: tuple[str, ...],
    unresolved_rule_refs: tuple[str, ...],
    mcp: _McpCoverage,
    claude_skill_dirs: tuple[str, ...] = (),
    claude_native_agents: tuple[str, ...] = (),
    arm_staging: _ArmStaging | None = None,
    cost_agents: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the report-only C2 ``component_coverage`` / ``context_cost`` / ``mcp_pinning``.

    *claude_skill_dirs* are the skill directories a native Claude Code arm's
    staged ``plugin.json`` loads beyond the wrapper, and *claude_native_agents*
    the native Claude Code arms that load them. *arm_staging* (the
    resolved per-agent plan) says where each arm stages rules, hooks,
    subagents, and commands, so a native arm's rows do not describe the
    wrapper.
    """
    member_resolved = {path.resolve() for path in member_skills}
    rows: list[dict[str, Any]] = []
    for component in inventory.components:
        if component.problem is not None:
            rows.append(coverage_row(component, "invalid", problem_reason(component)))
        elif component.declared_by:
            rows.append(_cross_client_row(component))
        elif component.type == "skill":
            rows.append(
                _skill_row(
                    component,
                    plugin_root=plugin_root,
                    member_skills=member_resolved,
                    member_names=_unique_names(path.name for path in member_skills),
                    claude_skill_dirs=set(claude_skill_dirs),
                    claude_native_agents=claude_native_agents,
                    unresolved_refs=unresolved_skill_refs,
                    skipped=skipped,
                )
            )
        elif component.type == "rule":
            rows.append(
                _rule_row(
                    component,
                    staged_rule_names=set(staged_rule_names),
                    unique_rule_names=_unique_names(staged_rule_names),
                    unresolved_refs=unresolved_rule_refs,
                    skipped=skipped,
                    arm_staging=arm_staging,
                )
            )
        elif component.type == "mcp":
            rows.append(_mcp_coverage_row(component, contained, mcp))
        else:
            rows.append(_other_type_row(component, arm_staging, cost_agents))
    return {
        "component_coverage": summarize_coverage(rows),
        "context_cost": inventory.context_cost(
            extra_rows=_external_member_skill_costs(member_skills, plugin_root),
            **_tier3_cost_view(arm_staging, cost_agents),
        ),
        "mcp_pinning": mcp_pinning_summary(inventory.mcp.effective),
    }


def _cross_client_row(component: Component) -> dict[str, Any]:
    """Coverage row of a component another client loads, through its own rules or an additional manifest."""
    state = "unsupported" if component.support == "unsupported" else "not_staged"
    source = (
        f"loaded only by {component.loaded_by}"
        if component.loaded_by
        else f"declared only by the additional manifest {component.declared_by}"
    )
    return coverage_row(component, state, f"{source}; Tier 3 stages the selected manifest's components")


def _unique_names(names: Iterable[str]) -> frozenset[str]:
    """The names exactly one staged member has."""
    counts = Counter(names)
    return frozenset(name for name, count in counts.items() if count == 1)


def _ref_member(component: Component, staged_names: frozenset[str]) -> str | None:
    """The member name a resolved skill or rule ref was staged under, or ``None``.

    Staging names the member after the ref's trailing name, whichever way the
    ref resolved: the bundled skill directory, the same-repository snapshot, or
    the same-named ``--include-skills`` directory for an external ref
    (``gitlab::<group>/<repo>::skills::release-notes`` is staged as
    ``release-notes``), and a rule after its file name. The load census and the
    activation labels name it that way, so the row records it. A name no staged
    member has is not recorded, and neither is one two staged members share:
    evidence for that name cannot say which of them it was (the task
    environment stages only the first skill directory with a given name).
    """
    member = _ref_name(component.name)
    return member if member and member in staged_names else None


def _skill_row(
    component: Component,
    *,
    plugin_root: Path,
    member_skills: set[Path],
    member_names: frozenset[str],
    claude_skill_dirs: set[str],
    claude_native_agents: tuple[str, ...],
    unresolved_refs: tuple[str, ...],
    skipped: bool,
) -> dict[str, Any]:
    """Coverage row of one skill: a staged member skill, a native Claude Code skill directory, or neither.

    *member_names* are the directory names exactly one staged member skill has.
    """
    if component.path == ".":
        return coverage_row(
            component,
            "not_staged",
            "a root SKILL.md single-skill plugin is inventoried only; Tier 3 stages skill directories",
        )
    if component.path is None:
        if component.name in unresolved_refs:
            return coverage_row(component, "unavailable", "remote skill reference is not resolvable offline")
        if skipped:
            return coverage_row(component, "not_staged", _SKIPPED_PACKAGE_NOTE)
        return coverage_row(
            component,
            "staged",
            "skill reference resolved to a local member skill",
            member=_ref_member(component, member_names),
        )
    if (plugin_root / component.path).resolve() in member_skills:
        return coverage_row(component, "staged", "bundled skill staged as a plugin member skill")
    if skipped:
        return coverage_row(component, "not_staged", _SKIPPED_PACKAGE_NOTE)
    if component.path in claude_skill_dirs:
        row = coverage_row(
            component,
            "staged",
            "declared skill directory staged by the native claude-code arm only (Claude Code loads the skill "
            "directories plugin.json declares); the other arms do not stage it",
        )
        return _natively_staged(row, list(claude_native_agents))
    return coverage_row(
        component,
        "not_staged",
        "only skills under skills/ are staged by Tier 3 for this manifest format; this declared skill directory "
        "is inventoried only",
    )


def _rule_row(
    component: Component,
    *,
    staged_rule_names: set[str],
    unique_rule_names: frozenset[str],
    unresolved_refs: tuple[str, ...],
    skipped: bool,
    arm_staging: _ArmStaging | None,
) -> dict[str, Any]:
    """Coverage row of one rule: staged (natively or in the wrapper), unavailable, or not staged.

    *unique_rule_names* are the names exactly one staged rule has.
    """
    native_agents = arm_staging.native_for("rule") if arm_staging is not None else []
    if component.path is None:
        if component.name in unresolved_refs:
            return coverage_row(component, "unavailable", "remote rule reference is not resolvable offline")
        if skipped:
            return coverage_row(component, "not_staged", _SKIPPED_PACKAGE_NOTE)
        reason = f"rule reference resolved and {_rule_reason(arm_staging, 'embedded in the wrapper')}"
        row = coverage_row(component, "staged", reason, member=_ref_member(component, unique_rule_names))
        return _natively_staged(row, native_agents)
    if {component.name, PurePosixPath(component.path).name, component.path.removeprefix("rules/")} & staged_rule_names:
        return _natively_staged(coverage_row(component, "staged", _rule_reason(arm_staging)), native_agents)
    return coverage_row(
        component,
        "not_staged",
        _SKIPPED_PACKAGE_NOTE if skipped else "rule file is inventoried but was not staged by Tier 3",
    )


def _mcp_coverage_row(component: Component, contained: bool, mcp: _McpCoverage) -> dict[str, Any]:
    if component.bundle:
        return coverage_row(component, "unsupported", "MCP bundles (.mcpb/.dxt) are not unpacked or staged")
    declaration = component.mcp
    if declaration is not None and declaration.source == "mcp_json" and not contained:
        return coverage_row(
            component, "not_staged", "agent_plugin.yaml plugins stage MCP servers from their 'mcp' list only"
        )
    if component.name in mcp.split.runnable_names:
        reason = "runnable MCP server staged for the with-plugin arm only"
        if component.name in mcp.unsupported:
            reason += (
                "; its env/headers or ${user_config.*} values are not applied by the runtime (run reported INCOMPLETE)"
            )
        return coverage_row(component, "staged", reason)
    if mcp.plugin_file_agents and component.name in mcp.split.plugin_file_names:
        reason = (
            "MCP server launches from plugin files; staged for the native claude-code arm, which copies the plugin "
            "tree and expands ${CLAUDE_PLUGIN_ROOT}"
        )
        if component.name in mcp.unsupported:
            note = mcp.plugin_file_gap_notes.get(component.name, "not started in the other with-plugin arms")
            reason += f"; {note} (run reported INCOMPLETE)"
        return _natively_staged(coverage_row(component, "staged", reason), list(mcp.plugin_file_agents))
    if component.name in mcp.unsupported:
        # Not runnable: _split_mcp_servers keeps plugin-file launches out of the toml.
        return coverage_row(
            component,
            "unsupported",
            "MCP server launches from plugin files (a plugin-root variable, a relative path, or cwd) that Tier 3 "
            "does not stage into the task environment; not started (run reported INCOMPLETE)",
        )
    if component.name in mcp.split.provider_names:
        return coverage_row(component, "unavailable", "provider-only MCP server is not runnable offline")
    return coverage_row(component, "not_staged", "MCP server was not staged")


def _external_member_skill_costs(member_skills: tuple[Path, ...], plugin_root: Path) -> list[CostRow]:
    """Context-cost rows for member skills staged from outside the plugin root."""
    from skillevaluator.plugin_components import cost_chars, model_hidden_traits

    root = plugin_root.resolve()
    rows: list[CostRow] = []
    for skill_dir in member_skills:
        if skill_dir.resolve().is_relative_to(root):
            continue
        for variant in ("SKILL.md", "skill.md"):
            try:
                text = secure_read_path_text(skill_dir / variant, CONTENT_DEDUP_MAX_FILE_BYTES)
            except (SecurePathError, OSError):
                continue
            parsed = parse_markdown(text)
            rows.append(
                CostRow(
                    "skill",
                    skill_dir.name,
                    cost_chars((parsed.name or "") + (parsed.description or "")),
                    cost_chars(parsed.body),
                    "member skill staged from outside the plugin root; always-on: name + description; "
                    "on-demand: SKILL.md body",
                    traits=model_hidden_traits(parsed.frontmatter),
                )
            )
            break
    return rows


def _cost_agents(agents: str | Sequence[str] | None) -> tuple[str, ...]:
    """The run's agents for the context-cost view; empty when they cannot be resolved."""
    try:
        return tuple(_planned_agents(agents) or ())
    except ValueError:
        return ()


def _tier3_cost_view(staging: _ArmStaging | None, agents: Sequence[str]) -> dict[str, Any]:
    """The harness and load mode the Tier 3 static estimate describes, and the note that goes with it.

    A native arm of a modeled harness wins (Claude Code or Codex); a wrapper
    run describes the first modeled agent's wrapper. When the plan is not
    known yet, the plugin's own native view stays; when no agent is modeled,
    the plugin's own harness is shown with the run's load mode.
    """
    from skillevaluator.plugin_components import COST_HARNESS_LABELS, COST_NATIVE, COST_VIEWS, COST_WRAPPER

    modeled = [agent for agent in agents if agent in {harness for harness, _mode in COST_VIEWS}]
    if staging is None:
        return {
            "extra_notes": (
                "Tier 3: the load mode of each arm is decided at run time; by_harness gives the native and "
                "wrapper estimates.",
            )
        }
    native = [agent for agent in sorted(staging.native) if agent in modeled]
    wrapped = [agent for agent in modeled if agent not in staging.native]
    views = [(agent, COST_NATIVE) for agent in native] + [(agent, COST_WRAPPER) for agent in wrapped]
    notes: list[str] = []
    if any(mode == COST_WRAPPER for _agent, mode in views) or not staging.native:
        notes.append(_TIER3_COVERAGE_NOTE)
    if len(views) > 1:
        shown = ", ".join(f"{COST_HARNESS_LABELS[agent]} ({mode})" for agent, mode in views)
        notes.append(f"Tier 3 arms load the plugin differently ({shown}); by_harness gives each estimate.")
    if not views:
        unmodeled = ", ".join(agents) or "the run's agents"
        notes.append(f"No context-cost model for {unmodeled}; the estimate shows the plugin's own harness.")
        return {"load_mode": COST_NATIVE if staging.native else COST_WRAPPER, "extra_notes": tuple(notes)}
    harness, load_mode = views[0]
    return {"harness": harness, "load_mode": load_mode, "extra_notes": tuple(notes)}


def write_plugin_provenance(run_dir: Path, provenance: dict[str, Any]) -> Path | None:
    """Persist plugin provenance next to the run so it survives temp cleanup.

    Writes ``plugin_provenance.json`` into the durable run directory (best
    effort). Complements the copy embedded in the agent_eval payload, so even the
    standalone ``evaluate-plugin`` path (which builds no report payload) leaves a
    durable record of a partial run.
    """
    try:
        run_path = Path(run_dir)
        target = run_path / "plugin_provenance.json"
        secure_atomic_write_text(
            target,
            json.dumps(provenance, indent=2),
            CONTENT_DEDUP_MAX_TOTAL_BYTES,
        )
        return target
    except (OSError, SecurePathError):
        return None


def _stageable_component(component: Component, component_type: str) -> bool:
    return (
        component.type == component_type
        and component.support == "evaluated"
        and component.problem is None
        and component.declared_by is None
        and component.path not in {None, "", "."}
    )


def _inventory_skill_dirs(inventory: PluginInventory, plugin_root: Path) -> tuple[Path, ...]:
    """Skill directories the selected manifest declares or packages (newer formats).

    The inventory classified every path without following links, so each
    directory is a regular directory inside the plugin root.
    """
    return tuple(
        dict.fromkeys(
            (plugin_root / component.path).resolve()
            for component in inventory.components
            if _stageable_component(component, "skill") and component.path
        )
    )


def _inventory_rule_files(inventory: PluginInventory, plugin_root: Path) -> list[_StagedRule]:
    """Read the selected manifest's rule files through the anchored, no-follow plugin root."""
    rules = [component for component in inventory.components if _stageable_component(component, "rule")]
    _check_rule_bounds(count=len(rules))
    staged: list[_StagedRule] = []
    total = 0
    try:
        with SecureRoot(plugin_root) as secure_root:
            for component in rules:
                content = secure_root.read_text(Path(str(component.path)), CONTENT_DEDUP_MAX_FILE_BYTES).strip()
                total += len(content.encode("utf-8"))
                _check_rule_bounds(total_bytes=total)
                staged.append(_StagedRule(name=component.name, content=content))
    except SecurePathError as exc:
        raise ValueError(f"Refusing unsafe or unbounded plugin rules: {exc}") from exc
    return staged


def _manifest_location(plugin_path: Path) -> PluginManifestLocation:
    from skillevaluator.plugin_manifest import locate_plugin_manifest

    located = locate_plugin_manifest(plugin_path)
    if located is None:
        raise ValueError(
            f"No supported plugin manifest ({', '.join(PLUGIN_MANIFEST_RELATIVE_PATHS)}) found under {plugin_path}"
        )
    return located


def _load_manifest_text(raw_text: str, manifest_path: Path, manifest_type: str) -> dict[str, Any]:
    syntax = manifest_syntax(manifest_type)
    # The YAML parser skips a leading byte-order mark itself; the JSON parser does not.
    text = raw_text.lstrip("\ufeff") if syntax == "json" else raw_text
    try:
        data = parse_manifest_text(manifest_type, text)
    except StructuredDataError as exc:
        raise ValueError(f"{manifest_path} is not valid bounded {syntax.upper()}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a manifest object")
    return data


def _plugin_name(manifest: dict[str, Any], plugin_dir: Path) -> str:
    """The plugin's own name. Plugin names are not skill names: Claude Code sets no length limit and Codex loads
    long names, so Tier 1 passes them and this bound is the report bound, not the 64-character skill limit."""
    raw_name = manifest.get("name")
    if raw_name is None or (isinstance(raw_name, str) and not raw_name.strip()):
        raw_name = plugin_dir.name
    return require_bounded_string(raw_name, "Plugin manifest name", max_chars=PLUGIN_NAME_MAX_REPORT_CHARS).strip()


# Room for the package suffix in one file name (255 bytes on common file systems).
_PACKAGE_NAME_MAX_CHARS = 200


def _wrapper_skill_name(plugin_name: str) -> str:
    """The generated wrapper ``SKILL.md`` name: the plugin name, cut to the skill name limit when longer.

    A longer name keeps its start and gets a short hash of the whole name, so two long plugin names that share
    a prefix stay apart.
    """
    if len(plugin_name) <= NAME_MAX_LENGTH:
        return plugin_name
    digest = hashlib.sha256(plugin_name.encode("utf-8")).hexdigest()[:8]
    return f"{plugin_name[: NAME_MAX_LENGTH - len(digest) - 1].rstrip('-_. ')}-{digest}"


def _plugin_description(manifest: dict[str, Any], plugin_name: str) -> str:
    raw_description = manifest.get("description")
    if raw_description is None or (isinstance(raw_description, str) and not raw_description.strip()):
        raw_description = f"Plugin evaluation wrapper for {plugin_name}."
    return require_bounded_string(
        raw_description,
        "Plugin manifest description",
        max_chars=DESCRIPTION_MAX_LENGTH,
    ).strip()


@dataclass(frozen=True)
class _IntraRepoResolver:
    """Layer-1 resolver for canonical refs that live in the plugin's own clone.

    A canonical ``<source>::<repo>::<kind>::<name>`` ref is resolved to a real path
    under its repo-root content dir (``<clone_root>/<ref_kind>/<name>``, where
    ``ref_kind`` is the ref's first path segment -- ``skills``/``team-skills`` for a
    skill or ``rules``/``team-rules`` for a rule) only when:

    * detection is active (there is an enclosing repo above the plugin), AND
    * the ref names a public remote source (github/gitlab/git), AND
    * ``ref_kind`` is a recognized content root for the resolution kind, AND
    * the ref ``<repo>`` matches the local clone's git-origin slug, AND
    * every lexical component below the clone root is opened without following
      links or reparse points.

    Containment is widened from the plugin root to the ref's content root
    (``<clone_root>/<ref_kind>``) for these slug-verified refs ONLY; symlink / ``..``
    escapes outside that content root are rejected, so a ref can only reach a
    recognized skills/rules dir. Refs stay unresolved when the local origin slug
    is unavailable because path existence alone cannot prove repository identity.
    """

    clone_root: Path
    local_slug: str | None
    active: bool
    snapshot_root: Path
    # The repository identity Tier 1 uses for the same plugin (same root, same slug).
    identity: Any = dataclass_field(default=None, repr=False, compare=False)
    _snapshot_cache: dict[tuple[str, str, str], Path | None] = dataclass_field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def _repo_matches(self, repo: str) -> bool:
        return self.local_slug is not None and repo == self.local_slug

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
        if rel.is_absolute() or not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
            return None
        cache_key = (kind, ref_kind, rel.as_posix())
        if cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key]
        content_root = self.clone_root / ref_kind
        try:
            content_root_metadata = content_root.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ValueError(f"Cannot inspect intra-repo plugin content root safely: {content_root}: {exc}") from exc
        if stat_is_link_or_reparse(content_root_metadata):
            raise ValueError(f"Refusing symlink or reparse-point intra-repo plugin content root: {content_root}")
        if not stat.S_ISDIR(content_root_metadata.st_mode):
            return None

        candidate = content_root / rel
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ValueError(f"Cannot inspect intra-repo plugin {kind} ref safely: {candidate}: {exc}") from exc
        if stat_is_link_or_reparse(metadata):
            raise ValueError(f"Refusing linked intra-repo plugin {kind} ref: {candidate}")

        snapshot = self.snapshot_root / kind / ref_kind / rel
        if want_dir:
            if not stat.S_ISDIR(metadata.st_mode):
                return None
            try:
                copytree_secure(candidate, snapshot, allowed_root=self.clone_root)
            except (OSError, UnsafeStagingError) as exc:
                raise ValueError(f"Refusing unsafe intra-repo plugin skill '{candidate}': {exc}") from exc
            resolved = snapshot if (snapshot / "SKILL.md").is_file() or (snapshot / "skill.md").is_file() else None
            self._snapshot_cache[cache_key] = resolved
            return resolved

        if not stat.S_ISREG(metadata.st_mode) or getattr(metadata, "st_nlink", 1) != 1:
            raise ValueError(f"Refusing non-regular intra-repo plugin rule: {candidate}")
        try:
            copy_file_secure(candidate, snapshot, allowed_root=self.clone_root)
        except (OSError, UnsafeStagingError) as exc:
            raise ValueError(f"Refusing unsafe intra-repo plugin rule '{candidate}': {exc}") from exc
        self._snapshot_cache[cache_key] = snapshot
        return snapshot

    def resolve_skill(self, ref: Any) -> Path | None:
        """Resolve an intra-repo ``skills`` ref to a private local snapshot."""
        return self._resolve(ref, kind="skills", want_dir=True)

    def resolve_rule(self, ref: Any) -> Path | None:
        """Resolve an intra-repo ``rules`` ref to a private local snapshot."""
        return self._resolve(ref, kind="rules", want_dir=False)


def _make_intra_repo_resolver(
    plugin_dir: Path,
    plugin_root: Path,
    repo_root: Path | None,
    stage_root: Path,
) -> _IntraRepoResolver:
    """Build the intra-repo resolver from the same repository identity Tier 1 uses.

    Root precedence matches Tier 1 (:func:`resolve_repository_identity`):
    ``--repo-root`` when it contains the plugin, then the git top-level that
    contains the plugin, then the catalog layout. A ``--repo-root`` that does
    not contain the plugin is ignored, as at Tier 1. Resolution is active only
    when the root's ``origin`` slug is known (the same fail-closed rule).
    """
    from skillevaluator.plugin_dependencies import resolve_repository_identity

    del plugin_root  # the identity is resolved from the plugin directory, as at Tier 1
    identity = resolve_repository_identity(plugin_dir, repo_root, slug_for=_local_repo_slug)
    snapshot_root = stage_root.expanduser().absolute() / ".intra-repo-snapshots"
    return _IntraRepoResolver(
        clone_root=identity.clone_root,
        local_slug=identity.local_slug,
        active=identity.local_slug is not None,
        snapshot_root=snapshot_root,
        identity=identity,
    )


def _resolve_skill_refs(
    section: Any,
    *,
    resolver: _IntraRepoResolver,
    plugin_dir: Path,
    contained: tuple[Path, ...],
    included: tuple[Path, ...],
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Stage the skill refs Tier 1 can prove, and return ``(staged skill dirs, unresolved labels)``.

    Each ref is classified by the Tier 1 classifier with the resolver's
    identity, so both tiers agree:

    * ``provided``: the skill at that path inside the plugin (a bundled skill,
      or another skill folder inside the plugin root);
    * ``referenced``: a private snapshot of the same-repository skill;
    * ``missing``: never covered. A bundled skill with the same name is not
      the skill the ref names (it is not at that repository path);
    * ``external`` or ``unresolved``: covered only by an ``--include-skills``
      directory with the ref's trailing name, when that name is declared once.
      That flag is the explicit way to supply a skill Tier 3 cannot fetch; a
      bundled skill with the same name never covers such a ref.

    A ref listed twice is reported once. A same-repository ref that reaches its
    target through a link is refused with ``ValueError``, as staging always did.
    """
    from skillevaluator.plugin_dependencies import classify_section

    raw_refs = _iter_raw_refs(section)
    pairs = classify_section(raw_refs, kind="skills", plugin_root=plugin_dir, identity=resolver.identity)
    contained_dirs = set(contained)
    included_names: dict[str, int] = {}
    for path in included:
        included_names[path.name] = included_names.get(path.name, 0) + 1
    name_counts: dict[str, int] = {}
    for ref, _row in pairs:
        name = _ref_name(ref)
        if name:
            name_counts[name] = name_counts.get(name, 0) + 1
    staged: list[Path] = []
    unresolved: list[str] = []
    for ref, row in pairs:
        if row.state == "provided" and row.path:
            target = (plugin_dir / row.path).resolve()
            if target not in contained_dirs:
                staged.append(target)
            continue
        if row.state == "referenced" or row.cause == "link":
            # A linked target is refused outright (ValueError), as staging always did; never staged.
            resolved = resolver.resolve_skill(ref)
            if resolved is not None and row.state == "referenced":
                staged.append(resolved)
                continue
        elif row.state in {"external", "unresolved"}:
            name = _ref_name(ref)
            if name and name_counts.get(name) == 1 and included_names.get(name) == 1:
                continue
        unresolved.append(_ref_label(ref))
    return tuple(dict.fromkeys(staged)), tuple(unresolved)


def _resolve_rules(
    section: Any, plugin_dir: Path, plugin_root: Path, resolver: _IntraRepoResolver
) -> tuple[tuple[_StagedRule, ...], tuple[str, ...]]:
    """Resolve rule refs to contained files; report remote/unresolved ones.

    Returns ``(staged_rule_files, unresolved_labels)``. A canonical
    remote ref is classified by the Tier 1 classifier with the resolver's
    identity (same root, same case-exact probe), so both tiers agree: a
    ``provided`` rule (inside the plugin root) is staged from the plugin, a
    ``referenced`` rule from a private snapshot of this repository, and a
    ``missing``, ``external``, or ``unresolved`` rule is never staged. A
    path-like ref is staged when it resolves to a real file inside the plugin
    root (symlink-contained, mirroring :func:`find_bundled_plugin_skills`). A
    canonical ref listed twice is reported once. Each staged rule is read once,
    here, through a bounded no-follow read, and the staged set shares the
    contained ``rules/`` aggregate byte bound.
    """
    from skillevaluator.plugin_dependencies import CAUSE_LINK, classify_ref, normalize_ref

    staged: list[_StagedRule] = []
    unresolved: list[str] = []
    seen: set[Path] = set()
    total_bytes = 0

    def _stage(path: Path) -> None:
        nonlocal total_bytes
        rule = _load_rule_path(path)
        # MAX_PLUGIN_MANIFEST_ITEMS per-file-bounded refs must not add up to a
        # wrapper the contained rules/ form would reject.
        total_bytes += len(rule.content.encode("utf-8"))
        _check_rule_bounds(total_bytes=total_bytes)
        staged.append(rule)
        seen.add(path)

    listings: dict[Path, frozenset[str] | None] = {}
    classified: set[str] = set()
    for ref in _iter_raw_refs(section):
        label = _ref_label(ref)
        if _ref_source(ref) in _REMOTE_REF_SOURCES:
            key = normalize_ref(ref) if isinstance(ref, str | dict) else None
            if key is not None:
                if key in classified:
                    continue
                classified.add(key)
            row = classify_ref(ref, kind="rules", plugin_root=plugin_dir, identity=resolver.identity, listings=listings)
            if row.state == "provided" and row.path:
                target = (plugin_dir / row.path).resolve()
                if _is_within(target, plugin_root):
                    if target not in seen:
                        _stage(target)
                    continue
            elif row.state == "referenced" or row.cause == CAUSE_LINK:
                # A linked target is refused outright (ValueError), as staging always did; never staged.
                intra = resolver.resolve_rule(ref)
                if intra is not None and row.state == "referenced":
                    if intra not in seen:
                        _stage(intra)
                    continue
            unresolved.append(label)
            continue
        resolved = _resolve_contained_file(ref, plugin_dir, plugin_root)
        if resolved is not None and resolved not in seen:
            _stage(resolved)
        elif resolved is None:
            unresolved.append(label)
    return tuple(staged), tuple(unresolved)


def _resolve_contained_file(ref: Any, plugin_dir: Path, plugin_root: Path) -> Path | None:
    """Resolve a path-like ref to a file contained within the plugin root."""
    path_str = ref if isinstance(ref, str) else (ref.get("path") if isinstance(ref, dict) else None)
    if path_str is None:
        return None
    path_str = require_bounded_string(
        path_str,
        "Contained plugin reference path",
        max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
    )
    if not path_str or "::" in path_str:
        return None
    path = Path(path_str)
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


def _check_rule_bounds(*, count: int = 0, total_bytes: int = 0) -> None:
    """Fail closed once the staged plugin rules exceed the shared file-count or aggregate byte bound.

    Every staged rule body is held in memory and embedded in the generated
    wrapper (the with-plugin arm's skill context), so the contained ``rules/``
    form, the inventory's rule files, and resolved rule refs share these bounds.
    """
    if count > MAX_PLUGIN_MANIFEST_ITEMS:
        raise ValueError(f"Plugin rules exceed the {MAX_PLUGIN_MANIFEST_ITEMS}-file limit")
    if total_bytes > CONTENT_DEDUP_MAX_TOTAL_BYTES:
        raise ValueError(f"Plugin rules exceed the {CONTENT_DEDUP_MAX_TOTAL_BYTES}-byte total limit")


def _load_rule_path(path: Path) -> _StagedRule:
    """Read one resolved rule through its parent anchor with a hard byte limit."""
    try:
        content = secure_read_path_text(path, CONTENT_DEDUP_MAX_FILE_BYTES).strip()
    except SecurePathError as exc:
        raise ValueError(f"Refusing unsafe or unbounded plugin rule '{path.name}': {exc}") from exc
    return _StagedRule(name=path.name, content=content)


def _discover_contained_rule_files(plugin_root: Path) -> list[_StagedRule]:
    """Discover rule files bundled under ``<plugin_root>/rules`` for a contained
    plugin whose manifest expresses ``rules`` as a directory pointer ("./rules/").

    Mirrors ``find_bundled_plugin_skills``: only real files whose resolved path
    stays inside the plugin root are returned (symlink-escape safe), so a ``rules``
    symlink cannot capture a host file. Sorted for deterministic staging.
    """
    rules_root = plugin_root / "rules"
    try:
        rules_root.lstat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ValueError(f"Cannot inspect contained plugin rules safely: {exc}") from exc

    try:
        files = discover_secure_files(
            rules_root,
            selected=lambda _relative: True,
            excluded_dirs=SCAN_EXCLUDED_DIRS,
            max_paths=CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
            allow_context_alias=False,
        )
        _check_rule_bounds(count=len(files), total_bytes=sum(file.metadata.st_size for file in files))
        with SecureRoot(rules_root) as secure_root:
            return [
                _StagedRule(
                    name=file.relative_path.as_posix(),
                    content=secure_root.read_file_text(file, CONTENT_DEDUP_MAX_FILE_BYTES).strip(),
                )
                for file in files
            ]
    except SecurePathError as exc:
        raise ValueError(f"Refusing unsafe or unbounded contained plugin rules: {exc}") from exc


_URL_USERINFO_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*://)[^\s/@'\"]+@")


def _effective_severity(finding: Any, policy: Any) -> Severity:
    """The finding's severity after the policy's ``severity_overrides``, as Tier 1 applies them."""
    current = finding.severity if isinstance(finding.severity, Severity) else Severity(str(finding.severity).lower())
    return policy.severity_for(finding.category, finding.check_name, current) if policy is not None else current


def _blocking(findings: Any, policy: Any) -> list[Any]:
    return [
        finding for finding in findings if _effective_severity(finding, policy) in (Severity.CRITICAL, Severity.HIGH)
    ]


def _refusal_detail(finding: Any) -> str:
    """A blocking finding's message for a Tier 3 refusal, never with a credential in it.

    An inline-secret finding names only its check; any other message loses URL
    user information and every known token shape.
    """
    from skillevaluator.utils.redaction import redact_sensitive_text

    check = str(finding.check_name)
    if "secret" in check or "credential" in check:
        return "an inline credential (the value is not shown)"
    text = _URL_USERINFO_RE.sub(r"\1<redacted>@", str(finding.message))
    return redact_secrets_in_log_line(redact_sensitive_text(text, max_len=MAX_PLUGIN_MANIFEST_TEXT_CHARS))


def _reject_unsafe_mcp_declaration(
    name: Any, config: dict[str, Any], *, policy: Any = None, manifest_type: str | None = None
) -> None:
    """Fail closed before a runnable MCP declaration reaches Harbor.

    The direct Tier 3 command does not run Tier 1 first. Reuse the same network-free
    declaration policy here so shell smuggling, insecure endpoints, inline secrets,
    malformed transports, and other blocking findings can never be executed merely
    because the caller selected Tier 3 directly. The *policy*'s private-host
    allowlist and severity overrides apply exactly as in Tier 1, and the
    refusal never repeats a credential. *manifest_type* names the format whose
    client loads the server, so a Codex plugin's ``${VAR:-default}`` URL (which
    Codex does not expand) is refused here as Tier 1 reports it.
    """
    from skillevaluator.validators.mcp_static import validate_mcp_server_declaration

    safe_name = require_bounded_string(
        name,
        "Plugin MCP server name",
        max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
    ).strip()
    for field in ("command", "url", "transport", "type", "provider"):
        if field in config and config[field] is not None:
            require_bounded_string(
                config[field],
                f"Plugin MCP server {field}",
                max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
                allow_empty=True,
            )
    args = config.get("args")
    if args is not None:
        if not isinstance(args, list):
            raise ValueError("Plugin MCP server args must be a list of strings")
        if len(args) > MAX_PLUGIN_MANIFEST_ITEMS:
            raise ValueError(f"Plugin MCP server args exceed the {MAX_PLUGIN_MANIFEST_ITEMS}-item limit")
        for index, arg in enumerate(args):
            require_bounded_string(
                arg,
                f"Plugin MCP server args[{index}]",
                max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
                allow_empty=True,
            )
    for field in ("env", "headers"):
        values = config.get(field)
        if values is None:
            continue
        if not isinstance(values, dict) or len(values) > MAX_PLUGIN_MANIFEST_ITEMS:
            raise ValueError(f"Plugin MCP server {field} must be a bounded object")
        for key, value in values.items():
            require_bounded_string(
                key,
                f"Plugin MCP server {field} key",
                max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
            )
            require_bounded_string(
                value,
                f"Plugin MCP server {field} value",
                max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
                allow_empty=True,
            )

    hosts = tuple(policy.mcp_allowed_private_hosts) if policy is not None else ()
    findings = validate_mcp_server_declaration(
        safe_name, config, "<plugin manifest>", allowed_private_hosts=hosts, manifest_type=manifest_type
    )
    blocking = _blocking(findings, policy)
    if blocking:
        first = blocking[0]
        raise ValueError(
            f"Plugin manifest MCP server '{safe_name}' failed blocking static validation "
            f"({first.check_name}): {_refusal_detail(first)}"
        )


def _reject_unloaded_mcp_findings(inventory: PluginInventory, policy: Any) -> None:
    """Refuse a plugin whose MCP servers Tier 1 blocks even though Tier 3 does not stage them.

    Another client can load the same folder (Claude Code reads a root
    ``.mcp.json`` with ``--plugin-dir``), and a bundle-reference plugin's root
    ``.mcp.json`` is inventoried but never staged. Their per-server findings
    are Tier 1 findings, so a blocking one (after the policy) refuses the run
    here too. Broken config files that nothing stages stay non-blocking.
    """
    server_findings = [
        *inventory.mcp.server_findings,
        *(
            finding
            for finding in inventory.findings
            if finding.category == "MCP_DECLARATION" and (finding.metadata or {}).get("mcp_server")
        ),
    ]
    blocking = _blocking(server_findings, policy)
    if blocking:
        first = blocking[0]
        server = str((first.metadata or {}).get("mcp_server") or "")[:MAX_PLUGIN_MANIFEST_TEXT_CHARS]
        raise ValueError(
            f"Plugin MCP server '{server}' (not staged, but loaded from this folder by a client or checked by Tier 1) "
            f"failed blocking static validation ({first.check_name}): {_refusal_detail(first)}"
        )


_YAML_MCP_KEYS = frozenset({"name", "provider"})


def _reject_runnable_yaml_mcp_entry(entry: dict[str, Any], idx: int) -> None:
    """Hold an ``agent_plugin.yaml`` ``mcp`` entry to the manifest schema Tier 1 validates: ``name`` and ``provider``."""
    from skillevaluator.models.plugin import MCP_NAME_PATTERN

    extra = sorted(str(key)[:64] for key in entry if key not in _YAML_MCP_KEYS)
    name, provider = entry.get("name"), entry.get("provider")
    if extra:
        raise ValueError(
            f"Plugin manifest mcp[{idx}] is not a name and provider entry: agent_plugin.yaml MCP entries take only "
            f"'name' and 'provider' (Tier 1 reports the rest as schema errors); unsupported keys: {', '.join(extra)}"
        )
    if not isinstance(name, str) or not re.fullmatch(MCP_NAME_PATTERN, name.strip()):
        raise ValueError(f"Plugin manifest mcp[{idx}].name must be a valid MCP server name (the manifest schema)")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"Plugin manifest mcp[{idx}] needs a non-empty provider (the manifest schema)")


def _launches_from_plugin_files(config: dict[str, Any], *, root_prefixes: tuple[str, ...] = ()) -> bool:
    """Whether a runnable MCP server's launch config points into the plugin tree.

    The documented form for a server shipped inside a plugin is
    ``"command": "${CLAUDE_PLUGIN_ROOT}/servers/x"``, which Claude Code expands when
    it loads the installed plugin; the Codex, Cursor, and Agent Plugins formats
    use their own placeholders (*root_prefixes*, such as ``${PLUGIN_ROOT}`` and
    ``${CURSOR_PLUGIN_ROOT}``). The wrapper runtime passes the staged config to
    the agent verbatim and packages only the generated wrapper, member skills,
    and evals, so a plugin-root variable (braced or bare), a ``./`` or ``../`` path
    in ``command``/``args``/``cwd`` (including ``--flag=./path``), a relative
    ``cwd`` (including ``.``, which Codex resolves against the plugin directory),
    or a relative ``command`` path all name files the with-plugin arm does not have.
    """
    args = config.get("args")
    launch = [config.get("command"), config.get("cwd"), *(args if isinstance(args, list) else ())]
    values = [value.strip() for value in launch if isinstance(value, str)]
    url = config.get("url")
    var_re = _plugin_install_var_re(tuple(root_prefixes))
    if any(var_re.search(value) for value in (*values, url if isinstance(url, str) else "")):
        return True
    cwd = config.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        rooted = cwd.strip().replace("\\", "/")
        if not rooted.startswith(("/", "~", "$")) and not _WINDOWS_DRIVE_RE.match(rooted):
            return True
    for value in values:
        path = value.split("=", 1)[1] if value.startswith("-") and "=" in value else value
        # A bare "./" (or "../") is the arm's working directory, which exists there.
        if path.startswith(_RELATIVE_PATH_PREFIXES) and path.replace("\\", "/").rstrip("/") not in {".", ".."}:
            return True
    command = config.get("command")
    if not isinstance(command, str):
        return False
    # A command containing a separator is a path (never a PATH lookup); unless it
    # is absolute (or env-rooted) it resolves against the arm's working directory.
    command = command.strip().replace("\\", "/")
    return "/" in command and not command.startswith(("/", "~", "$")) and not _WINDOWS_DRIVE_RE.match(command)


def _plugin_root_launch(server: dict[str, Any], *, root_prefixes: tuple[str, ...]) -> dict[str, Any] | None:
    """A plugin-file launch rewritten for a copied plugin tree, or ``None``.

    Claude Code copies of the plugin expand ``${CLAUDE_PLUGIN_ROOT}`` (and
    ``${CLAUDE_PLUGIN_DATA}``) in ``command``/``args``/``url``, so a launch whose
    only plugin-file reference is a plugin-root variable can start there; a
    format's own placeholder (``${PLUGIN_ROOT}``, ``${CURSOR_PLUGIN_ROOT}``) is
    rewritten to ``${CLAUDE_PLUGIN_ROOT}``. Relative paths and ``cwd`` cannot be
    expressed (Claude Code's MCP config has no working directory), so those
    servers stay unsupported everywhere.
    """
    cwd = server.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return None
    var_re = _plugin_install_var_re(root_prefixes)

    def _rooted(value: Any) -> Any:
        return var_re.sub(_ROOTED_PLACEHOLDER, value) if isinstance(value, str) else value

    probe = {key: _rooted(server.get(key)) for key in ("command", "url")}
    if isinstance(server.get("args"), list):
        probe["args"] = [_rooted(arg) for arg in server["args"]]
    if _launches_from_plugin_files(probe, root_prefixes=root_prefixes):
        return None
    foreign = foreign_root_var_re(tuple(root_prefixes))
    rewritten = {key: to_claude_root(value, foreign) for key, value in server.items() if key != "args"}
    if isinstance(server.get("args"), list):
        rewritten["args"] = [to_claude_root(arg, foreign) for arg in server["args"]]
    return rewritten


def _unstaged_root_mcp_json(manifest: dict[str, Any], plugin_root: Path) -> str | None:
    """Finding path of an ``agent_plugin.yaml`` plugin's implicit root ``.mcp.json``, or ``None``.

    Mirrors :func:`~skillevaluator.plugin_mcp.collect_mcp_declarations`: the
    root file is inventoried as the default ``mcp_json`` source (never staged for
    this manifest form) unless an ``mcpServers`` path names it explicitly -- then it
    is a staged ``path_ref`` source whose findings must keep blocking. Paths are
    read with the Claude Code profile's placeholders, as the inventory reads this
    manifest form: none, so ``${CLAUDE_PLUGIN_ROOT}/.mcp.json`` is a broken
    placeholder path (blocking on its own), not an explicit name for the root file.
    """
    declared = manifest.get("mcpServers")
    entries = declared if isinstance(declared, list) else [declared]
    prefixes = CLAUDE_PROFILE.manifest_path_prefixes
    if any(isinstance(entry, str) and normalize_declared_path(entry, prefixes).rel == MCP_JSON for entry in entries):
        return None
    return PluginRootReader(plugin_root).display(MCP_JSON)


def _normalize_mcp_entries(
    manifest: dict[str, Any],
    inventory: PluginInventory,
    contained_form: bool,
    *,
    plugin_root: Path | None = None,
    policy: Any = None,
    manifest_type: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize every manifest MCP form into the bundle-reference list shape.

    Bundle-reference ``agent_plugin.yaml`` uses a top-level ``mcp`` *list* of
    ``{name, provider}`` objects, the manifest schema Tier 1 validates; a
    runnable ``{name, command|url, transport}`` entry is refused, as Tier 1
    reports it (:func:`_reject_runnable_yaml_mcp_entry`). A contained
    ``.claude-plugin/plugin.json`` declares servers in any documented Claude Code
    form -- an inline ``mcpServers`` map, a ``.json`` path, or an array mixing
    both -- merged over the root ``.mcp.json`` (a later same-name server replaces
    an earlier one). The inventory collected those declarations through the
    bounded, no-follow plugin-root reader; here every declaration (including a
    shadowed one) must pass the blocking static checks before the effective
    servers are flattened to the list shape :func:`_split_mcp_servers` expects.

    This is the one validation of the MCP entries: every returned entry has a
    stripped, bounded string ``name``, and its launch fields passed
    :func:`_reject_unsafe_mcp_declaration`.
    """
    raw_servers = manifest.get("mcp")
    if raw_servers:
        if not isinstance(raw_servers, list):
            raise ValueError("Plugin manifest mcp must be a list")
        if len(raw_servers) > MAX_PLUGIN_MANIFEST_ITEMS:
            raise ValueError(f"Plugin manifest mcp exceeds the {MAX_PLUGIN_MANIFEST_ITEMS}-item limit")
        normalized_entries: list[dict[str, Any]] = []
        for idx, entry in enumerate(raw_servers):
            if not isinstance(entry, dict):
                raise ValueError(f"Plugin manifest mcp[{idx}] must be an object")
            _reject_runnable_yaml_mcp_entry(entry, idx)
            config = {key: value for key, value in entry.items() if key != "name"}
            _reject_unsafe_mcp_declaration(entry.get("name"), config, policy=policy, manifest_type=manifest_type)
            normalized_entries.append({**entry, "name": entry["name"].strip()})
        _reject_unloaded_mcp_findings(inventory, policy)
        return normalized_entries

    collection = inventory.mcp
    # Fail closed on config-source problems: escapes, absolute paths, symlinks,
    # missing/oversize/invalid config files, and malformed mcpServers values --
    # in the sources that will be staged. An agent_plugin.yaml plugin's implicit
    # root .mcp.json is inventoried (Tier 1 reports it) but never staged, so it
    # blocks here no more than it does when the manifest has an 'mcp' list.
    # Without the plugin root the finding cannot be attributed: keep it (fail closed).
    # Every config-source finding, so a policy that raises one to HIGH blocks here as in Tier 1.
    blocking = list(collection.findings)
    if not contained_form and plugin_root is not None:
        unstaged = _unstaged_root_mcp_json(manifest, plugin_root)
        blocking = [finding for finding in blocking if finding.file_path != unstaged]
    blocking = _blocking(blocking, policy)
    if blocking:
        first = blocking[0]
        raise ValueError(
            f"Plugin MCP configuration failed blocking static validation ({first.check_name}): {_refusal_detail(first)}"
        )

    def _stageable(source: str) -> bool:
        # agent_plugin.yaml plugins stage from their 'mcp' list only; a root
        # .mcp.json is inventoried and statically validated but not staged.
        return source in {"inline", "path_ref"} or (source == "mcp_json" and contained_form)

    declarations = [decl for decl in collection.declarations if _stageable(decl.source)]
    if len(declarations) > MAX_PLUGIN_MANIFEST_ITEMS:
        raise ValueError(f"Plugin MCP declarations exceed the {MAX_PLUGIN_MANIFEST_ITEMS}-item limit")
    for declaration in declarations:
        if not isinstance(declaration.config, dict):
            raise ValueError(f"Plugin manifest mcpServers[{declaration.name!r}] must be an object")
        # Fail closed: a raw inline credential must never be flattened into the
        # persisted toml (only ${ENV} references may reach the artifact).
        _reject_unsafe_mcp_declaration(declaration.name, declaration.config, policy=policy, manifest_type=manifest_type)
    _reject_unloaded_mcp_findings(inventory, policy)

    normalized: list[dict[str, Any]] = []
    for declaration in collection.effective:
        if not _stageable(declaration.source):
            continue
        # Validated above: the effective declarations are some of the stageable ones.
        config = declaration.config
        entry: dict[str, Any] = {"name": declaration.name.strip()}
        # env/headers are declared config the eval runtime cannot apply (Harbor's
        # per-server MCPServerConfig has no such field). Record their presence so a
        # server evaluated WITHOUT its declared config marks the run INCOMPLETE
        # rather than reading as a faithful pass (Tier 1 also surfaces an advisory).
        unsupported_fields = [field for field in ("env", "headers") if config.get(field)]
        unsupported_fields.extend(declaration.unapplied)
        if unsupported_fields:
            entry["_unsupported_fields"] = unsupported_fields
        # The native Claude Code adapter applies env/headers itself (never the
        # wrapper TOML or the other harness configs).
        declared = {field: config[field] for field in ("env", "headers") if isinstance(config.get(field), dict)}
        if declared:
            entry["_declared"] = declared
        # Standard Claude stdio config: {"command": ..., "args": [...]}; remote
        # config: {"type": "sse"|"http", "url": ...}.
        if config.get("command"):
            # Keep argv STRUCTURE: command is the program; args stay a list so a
            # spaced arg (e.g. "path with spaces") is one token, not re-split. The
            # runtime (Harbor MCPServerConfig.args: list[str]) and every agent
            # adapter consume a separate args list.
            entry["command"] = config["command"]
            args = config.get("args")
            if args:
                entry["args"] = list(args)
            entry["transport"] = config.get("transport") or config.get("type") or "stdio"
            # Never staged (the runtime has no cwd field); carried only so
            # _split_mcp_servers can see a cwd that points into the plugin tree.
            if config.get("cwd") is not None:
                entry["cwd"] = config["cwd"]
        elif config.get("url"):
            entry["url"] = config["url"]
            transport = config.get("transport") or config.get("type")
            if transport:
                entry["transport"] = transport
        else:
            # No command/url -> provider-only, so it is named as unresolved.
            entry["provider"] = config.get("provider") or config.get("type") or ""
        normalized.append(entry)
    return normalized


@dataclass(frozen=True)
class _McpSplit:
    """Plugin MCP servers sorted by how each with-plugin arm can run them.

    ``gaps`` maps every server some arm cannot fully apply to what is missing:
    ``plugin_files`` (launches from plugin files; only a copied plugin tree can
    start it, and ``plugin_file`` holds the rewritten launch), ``plugin_files_unrooted``
    (relative paths or ``cwd`` nothing can start), ``env``/``headers`` and other
    declared fields the wrapper runtime drops, and ``user_config`` (a
    ``${user_config.*}`` value only Claude Code fills in). ``probe_targets`` are
    the runnable URL servers for the opt-in ``--probe-mcp`` host probe.
    """

    runnable: list[dict[str, Any]]
    provider_only: list[dict[str, str]]
    plugin_file: list[dict[str, Any]]
    gaps: dict[str, tuple[str, ...]]
    declared: dict[str, dict[str, Any]]
    user_config: dict[str, tuple[str, ...]]
    probe_targets: list[dict[str, Any]]

    @property
    def unsupported_config(self) -> list[str]:
        """Servers with config the wrapper runtime cannot apply (the run is INCOMPLETE)."""
        return list(self.gaps)

    @property
    def runnable_names(self) -> tuple[str, ...]:
        return tuple(server["name"] for server in self.runnable)

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(server["name"] for server in self.provider_only)

    @property
    def plugin_file_names(self) -> tuple[str, ...]:
        return tuple(server["name"] for server in self.plugin_file)


# What a native Claude Code arm applies itself (it copies the plugin tree and
# reads the staged .mcp.json, env, headers, and userConfig defaults).
_CLAUDE_NATIVE_APPLIES = frozenset({"plugin_files", "env", "headers"})


def _runnable_server(entry: dict[str, Any]) -> dict[str, Any]:
    """The launch fields of a validated MCP entry that reach the staged server list.

    A ``command`` server defaults to ``stdio`` and a ``url`` server to ``http``
    (streamable HTTP), as native Claude Code and ``--probe-mcp`` assume; Harbor
    would otherwise read a URL server with no transport as ``sse``.
    """
    server: dict[str, Any] = {"name": entry["name"]}
    for key in ("url", "command", "transport"):
        if entry.get(key):
            server[key] = entry[key]
    if entry.get("args"):
        server["args"] = list(entry["args"])
    if "transport" not in server:
        server["transport"] = "stdio" if "command" in server else "http"
    return server


def _probe_target(entry: dict[str, Any], declared: dict[str, Any]) -> dict[str, Any]:
    """A runnable URL server for the ``--probe-mcp`` host probe: its transport and declared header references.

    Header values are the declared literals or ``${VAR}`` references; they are never persisted.
    """
    raw_headers = declared.get("headers")
    headers = (
        {str(key): str(value) for key, value in list(raw_headers.items())[:32] if isinstance(value, str)}
        if isinstance(raw_headers, dict)
        else {}
    )
    return {
        "name": entry["name"],
        "url": entry["url"],
        "transport": entry.get("transport") or entry.get("type") or "",
        "headers": headers,
    }


def _user_config_keys(raw: dict[str, Any], declared: dict[str, Any]) -> tuple[str, ...]:
    values: list[Any] = [raw.get("command"), raw.get("url"), *(raw.get("args") or [])]
    for block in declared.values():
        if isinstance(block, dict):
            values.extend(block.values())
    keys = {key for value in values if isinstance(value, str) for key in _USER_CONFIG_REF_RE.findall(value)}
    return tuple(sorted(keys))


def _split_mcp_servers(
    manifest: dict[str, Any],
    inventory: PluginInventory,
    contained_form: bool,
    *,
    plugin_root: Path | None = None,
    root_prefixes: tuple[str, ...] = (),
    policy: Any = None,
    manifest_type: str | None = None,
) -> _McpSplit:
    """Split MCP entries into runnable (command/url), plugin-file, and provider-only.

    Canonical ``PluginMcpEntry`` entries carry ``name`` + ``provider`` and are
    *not* runnable offline (returned as provider-only, contributing nothing to
    the run). Entries with a ``command``/``url`` are a documented local-testing
    extension and are staged with-plugin-only -- unless they launch from plugin
    files the wrapper does not stage (:func:`_launches_from_plugin_files`, using
    the manifest format's *root_prefixes*). Those never reach the wrapper TOML or
    the skip decision as runnable; the ones a copied plugin tree can start are
    kept in ``plugin_file`` for the native Claude Code adapter.
    """
    runnable: list[dict[str, Any]] = []
    provider_only: list[dict[str, str]] = []
    plugin_file: list[dict[str, Any]] = []
    gaps: dict[str, tuple[str, ...]] = {}
    declared_by_name: dict[str, dict[str, Any]] = {}
    user_config: dict[str, tuple[str, ...]] = {}
    probe_targets: list[dict[str, Any]] = []
    # Every entry was validated (bounded strings, blocking static checks) by _normalize_mcp_entries.
    for raw in _normalize_mcp_entries(
        manifest, inventory, contained_form, plugin_root=plugin_root, policy=policy, manifest_type=manifest_type
    ):
        name = raw["name"]
        if not (raw.get("command") or raw.get("url")):
            provider_only.append({"name": name, "provider": raw.get("provider") or ""})
            continue
        declared = raw.get("_declared")
        if not isinstance(declared, dict):
            declared = {field: raw[field] for field in ("env", "headers") if isinstance(raw.get(field), dict)}
        if declared:
            declared_by_name[name] = declared
        server_gaps: list[str] = [str(field) for field in raw.get("_unsupported_fields") or ()]
        if keys := _user_config_keys(raw, declared):
            user_config[name] = keys
            server_gaps.append("user_config")
        server = _runnable_server(raw)
        if _launches_from_plugin_files(raw, root_prefixes=root_prefixes):
            # Staged verbatim, this server could not start in a wrapper arm, yet it
            # would count as runnable: a plugin whose only component it is would run
            # a "complete" evaluation with a meaningless zero lift instead of an
            # honest skip. Keep it out of the toml and the runnable list; a copied
            # plugin tree (native Claude Code) can still start the rooted form.
            cwd = raw.get("cwd")
            rooted = _plugin_root_launch({**server, "cwd": cwd} if cwd else server, root_prefixes=root_prefixes)
            if rooted is not None:
                plugin_file.append(rooted)
                server_gaps.insert(0, "plugin_files")
            else:
                server_gaps.insert(0, "plugin_files_unrooted")
        else:
            runnable.append(server)
            if server.get("url"):
                probe_targets.append(_probe_target(raw, declared))
        if server_gaps:
            gaps[name] = tuple(dict.fromkeys(server_gaps))
    return _McpSplit(runnable, provider_only, plugin_file, gaps, declared_by_name, user_config, probe_targets)


def _claude_applies(gaps: tuple[str, ...], user_config: tuple[str, ...], defaults: set[str]) -> bool:
    """Whether a native Claude Code arm applies everything the wrapper drops for one server."""
    for gap in gaps:
        if gap == "user_config":
            if any(key not in defaults for key in user_config):
                return False
        elif gap not in _CLAUDE_NATIVE_APPLIES:
            return False
    return True


def _user_config_defaults(manifest: dict[str, Any]) -> set[str]:
    """``userConfig`` keys that declare a default (Claude Code applies it when unset)."""
    declared = manifest.get("userConfig")
    if not isinstance(declared, dict):
        return set()
    return {str(key) for key, spec in declared.items() if isinstance(spec, dict) and "default" in spec}


def _skip_reason(
    unresolved_skill_refs: tuple[str, ...],
    unresolved_rule_refs: tuple[str, ...],
    provider_mcp: list[dict[str, str]],
    unsupported_mcp: list[str] | tuple[str, ...] = (),
    *,
    unrooted_mcp: list[str] | tuple[str, ...] = (),
) -> str:
    """Why nothing was evaluated, with advice for each kind of component that is actually present.

    ``unsupported_mcp`` names the servers launched from unstaged plugin files;
    ``unrooted_mcp`` the ones among them that no copied plugin can start either
    (relative paths or a relative ``cwd``).
    """
    parts: list[str] = []
    advice: list[str] = []
    if unresolved_skill_refs:
        parts.append(f"{len(unresolved_skill_refs)} unresolved skill ref(s)")
        advice.append(
            "Add the referenced skills to this repository at their reference paths, or pass --include-skills "
            "with a local copy of a remote skill (a bundled skill with the same name does not satisfy a reference)."
        )
    if unresolved_rule_refs:
        parts.append(f"{len(unresolved_rule_refs)} unresolved rule ref(s)")
        advice.append("Add the referenced rules to this repository at their reference paths.")
    if provider_mcp:
        parts.append(f"{len(provider_mcp)} provider-only MCP server(s)")
        advice.append(
            "A provider-only MCP server (a name and provider, with no command or url) is not run in Phase 1; "
            "declare a command or url to test it locally."
        )
    if unsupported_mcp:
        # In a skipped package these are only plugin-file launches: a server with
        # just unapplied env/headers is still runnable, so it never reaches here.
        parts.append(f"{len(unsupported_mcp)} MCP server(s) launched from unstaged plugin files")
        unrooted = set(unrooted_mcp)
        if any(name not in unrooted for name in unsupported_mcp):
            advice.append(
                "Only a native Claude Code arm copies plugin files and can start such a server: run the "
                "claude-code agent with --plugin-load native or auto."
            )
        if unrooted:
            advice.append(
                "Start a server with ${CLAUDE_PLUGIN_ROOT} paths instead of relative paths or a relative cwd, "
                "so a copied plugin can start it."
            )
    if not parts:
        return "Plugin has no locally-resolvable components to evaluate in Phase 1 (no declared dependencies)."
    deferred = (
        " Remote bundle-reference resolution is deferred to a later phase."
        if unresolved_skill_refs or unresolved_rule_refs
        else ""
    )
    return (
        f"Plugin has no locally-resolvable components to evaluate in Phase 1 ({', '.join(parts)}), "
        f"so nothing it declares was evaluated.{deferred} {' '.join(advice)}"
    )


def _fresh_package_dir(stage_root: Path, plugin_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", plugin_name).strip("-._")[:_PACKAGE_NAME_MAX_CHARS] or "plugin"
    package_path = stage_root.expanduser().resolve() / f"{safe_name}{PLUGIN_EVAL_PACKAGE_SUFFIX}"
    if package_path.exists():
        raise ValueError(f"Plugin evaluation staging path already exists: {package_path}")
    package_path.mkdir(parents=True)
    return package_path


def _write_plugin_skill_md(
    path: Path,
    *,
    plugin_name: str,
    plugin_description: str,
    include_skills: tuple[Path, ...],
    staged_rules: tuple[_StagedRule, ...],
    unresolved_skill_refs: tuple[str, ...],
    unresolved_rule_refs: tuple[str, ...],
    provider_mcp_servers: tuple[str, ...],
) -> None:
    member_lines = "\n".join(f"- {skill.name}: staged as a plugin member skill." for skill in include_skills)
    if not member_lines:
        member_lines = "- No member skills were staged; evaluate the plugin wrapper, rules, and tools."

    rule_sections = "\n\n".join(_render_rule_block(rule) for rule in staged_rules)
    if not rule_sections:
        rule_sections = "- No contained rule files were staged."

    unresolved_lines = _render_unresolved(unresolved_skill_refs, unresolved_rule_refs, provider_mcp_servers)

    frontmatter = yaml.safe_dump(
        {
            "name": _wrapper_skill_name(plugin_name),
            "description": plugin_description,
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


def _render_rule_block(rule: _StagedRule) -> str:
    return f"### {rule.name}\n\n{rule.content}"


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


def _native_excluded_evals_paths(source: Path | None, plugin_root: Path) -> tuple[Path, ...]:
    """Eval data a native whole-plugin copy skips: the source, or its datasets when it is the plugin root."""
    if source is None:
        return ()
    if source.is_dir() and source.resolve() == plugin_root.resolve():
        return tuple(source / name for name in (*_EVAL_DATASET_NAMES, "harbor") if os.path.lexists(source / name))
    return (source,)


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
    sandbox isolation boundary begins.

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


def _load_member_eval_dataset(
    skill_dir: Path,
    *,
    snapshot_dir: Path,
    snapshot_index: int,
) -> tuple[Path, list[dict[str, Any]]] | None:
    """Snapshot and load one member dataset without following authored paths."""
    layouts = (("evals", "evals"), ("eval", "dataset"))
    for directory, stem in layouts:
        for extension in DATASET_EXTENSIONS:
            candidate = skill_dir / directory / f"{stem}{extension}"
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ValueError(f"Cannot inspect member dataset safely: {candidate}: {exc}") from exc
            if stat_is_link_or_reparse(metadata):
                raise ValueError(f"Refusing linked member dataset: {candidate}")
            if not stat.S_ISREG(metadata.st_mode) or getattr(metadata, "st_nlink", 1) != 1:
                raise ValueError(f"Refusing non-regular member dataset: {candidate}")
            if metadata.st_size > CONTENT_DEDUP_MAX_FILE_BYTES:
                raise ValueError(f"Member dataset exceeds the {CONTENT_DEDUP_MAX_FILE_BYTES}-byte limit: {candidate}")

            snapshot = snapshot_dir / f"member-{snapshot_index}{extension}"
            try:
                copy_file_secure(candidate, snapshot, allowed_root=skill_dir)
            except UnsafeStagingError as exc:
                raise ValueError(f"Refusing unsafe member dataset '{candidate}': {exc}") from exc
            try:
                raw_text = secure_read_path_text(snapshot, CONTENT_DEDUP_MAX_FILE_BYTES)
                return candidate, _parse_member_dataset_text(raw_text, extension)
            except SecurePathError as exc:
                raise ValueError(f"Cannot read snapshotted member dataset safely: {candidate}: {exc}") from exc
    return None


def _parse_member_dataset_text(raw_text: str, suffix: str) -> list[dict[str, Any]]:
    """Parse a bounded member-dataset snapshot in its declared format."""
    try:
        if suffix == ".jsonl":
            data: Any = [load_bounded_json(line) for raw_line in raw_text.splitlines() if (line := raw_line.strip())]
        elif suffix == ".json":
            data = load_bounded_json(raw_text)
        elif suffix in {".yaml", ".yml"}:
            data = load_bounded_yaml(raw_text)
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")
    except StructuredDataError as exc:
        raise ValueError(f"Invalid bounded member evaluation dataset: {exc}") from exc
    return normalize_dataset_entries(data)


def _write_combined_member_evals(evals_dir: Path, include_skills: tuple[Path, ...], *, plugin_name: str) -> None:
    evals_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    staged_files: dict[str, tuple[str, Path]] = {}
    with tempfile.TemporaryDirectory(prefix="member-dataset-snapshots-", dir=evals_dir.parent) as temp_dir:
        snapshot_dir = Path(temp_dir)
        for skill_index, skill_dir in enumerate(include_skills):
            loaded = _load_member_eval_dataset(
                skill_dir,
                snapshot_dir=snapshot_dir,
                snapshot_index=skill_index,
            )
            if loaded is None:
                continue
            eval_file, skill_entries = loaded
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
    # staged package. Bound against the member skill root, NOT files_dir itself.
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

    The file name is reserved for SkillEvaluator. The staged evals may be a copy
    of the plugin's own ``evals/``, and an authored copy of this file would skip
    the MCP safety checks and redaction, so its presence fails staging.

    Strings are TOML basic strings (``toml_quote``), not JSON: ``json.dumps``
    writes an emoji as a surrogate-pair escape and leaves DEL raw, and TOML
    rejects both. The text is parsed back before it is written, so a value TOML
    cannot hold fails here instead of silently dropping every plugin server.
    """
    import tomllib

    from skillevaluator.tier3.toml_utils import toml_quote

    env_dir = evals_dir / "environment"
    mcp_file = env_dir / PLUGIN_MCP_SERVERS_FILENAME
    if os.path.lexists(mcp_file):
        raise ValueError(
            f"Refusing evals source that provides environment/{PLUGIN_MCP_SERVERS_FILENAME}: SkillEvaluator "
            "generates that file from the plugin's validated MCP servers. Declare plugin servers in the plugin "
            "manifest, or shared task servers in environment/mcp_servers.toml."
        )
    if not servers:
        return
    env_dir.mkdir(parents=True, exist_ok=True)

    def toml_value(raw: Any) -> str:
        # Manifests may reference secret handles/env names; never emit a raw
        # secret. Redact known key shapes from strings before write.
        if isinstance(raw, str):
            return toml_quote(redact_secrets_in_log_line(raw))
        if isinstance(raw, bool | int | float):
            return json.dumps(raw)
        raise ValueError(f"plugin MCP value of type {type(raw).__name__} cannot be written as TOML")

    lines: list[str] = []
    for server in servers:
        lines.append("[[mcp_servers]]")
        for key in ("name", "url", "command", "transport"):
            if key in server:
                lines.append(f"{key} = {toml_value(server[key])}")
        args = server.get("args")
        if args:
            # Preserve argv structure as a real TOML array (a spaced arg stays one
            # token); redact each element so a secret cannot leak via args.
            lines.append("args = [" + ", ".join(toml_value(arg) for arg in args) + "]")
        lines.append("")
    text = "\n".join(lines)
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"generated {PLUGIN_MCP_SERVERS_FILENAME} is not valid TOML: {exc}") from exc
    mcp_file.write_text(text, encoding="utf-8")


def _write_plugin_runtime_components(evals_dir: Path, inventory: PluginInventory, *, plugin_name: str) -> None:
    """Record declared subagent and command names for report-only activation coverage.

    ``subagent_aliases`` maps the name a harness may call a staged subagent by
    (OpenCode stages an agent named like a built-in as ``<plugin>-<name>``) back
    to the declared name. The key is written only when some agent is renamed.
    """
    names: dict[str, Any] = {"subagents": [], "commands": []}
    aliases: dict[str, str] = {}
    for component in inventory.components:
        key = {"agent": "subagents", "command": "commands"}.get(component.type)
        if key is None or component.problem is not None or not component.name:
            continue
        if component.name not in names[key] and len(names[key]) < MAX_PLUGIN_RUNTIME_NAMES:
            names[key].append(component.name)
            staged = opencode_agent_name(plugin_name, component.name) if key == "subagents" else component.name
            if staged.casefold() != component.name.casefold():
                aliases.setdefault(staged, component.name)
    if aliases:
        names["subagent_aliases"] = aliases
    env_dir = evals_dir / "environment"
    target = env_dir / PLUGIN_RUNTIME_COMPONENTS_FILENAME
    if not (names["subagents"] or names["commands"]) and not target.exists():
        return
    env_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(names, indent=2), encoding="utf-8")
