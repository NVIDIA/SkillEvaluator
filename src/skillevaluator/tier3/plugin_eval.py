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

import json
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    DESCRIPTION_MAX_LENGTH,
    NAME_MAX_LENGTH,
    PLUGIN_CONTAINED_MANIFEST_DIR,
    PLUGIN_CONTAINED_MANIFEST_FILE,
    SCAN_EXCLUDED_DIRS,
)
from skillevaluator.deduplication.plugin.ref_utils import normalize_ref
from skillevaluator.models.result import Severity
from skillevaluator.plugin_components import (
    MCP_JSON,
    CostRow,
    PluginInventory,
    PluginRootReader,
    build_plugin_inventory,
    coverage_row,
    manifest_rel_for,
    mcp_pinning_summary,
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
from skillevaluator.plugin_dependencies import slug_from_remote_url as _slug_from_remote_url  # noqa: F401
from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS, load_dataset_entries, normalize_dataset_entries
from skillevaluator.tier3.eval_core.plugin_signals import validate_plugin_case_fields
from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line
from skillevaluator.tier3.harbor.secure_copy import UnsafeStagingError, copy_file_secure, copytree_secure
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
    from skillevaluator.plugin_manifest import PluginManifestLocation

# Shared with Harbor's runtime find_evals_file() and the report loader so a
# dataset accepted/staged here is resolvable downstream.
_EVAL_DATASET_NAMES = tuple(f"evals{extension}" for extension in DATASET_EXTENSIONS)

# Filename for the plugin's own runnable MCP servers. Kept distinct from the
# task-environment ``mcp_servers.toml`` so the adapter can stage it for the
# with-plugin arm only (see ``adapter.generate_harbor_tasks``).
PLUGIN_MCP_SERVERS_FILENAME = "plugin_mcp_servers.toml"

# Install-time variables Claude Code expands when it loads an installed plugin,
# plus the path forms that point into the plugin tree. The eval runtime expands
# neither and never copies the plugin tree into the task environment, so an MCP
# server launched through them cannot start (see _launches_from_plugin_files).
_PLUGIN_INSTALL_VAR_RE = re.compile(r"\$\{?CLAUDE_PLUGIN_(?:ROOT|DATA)\b")
_RELATIVE_PATH_PREFIXES = ("./", "../", ".\\", "..\\")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:/")


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
    dataset_case_count: int = 0
    cross_component_case_count: int = 0
    dependency_status_counts: tuple[tuple[str, int], ...] = ()
    skipped: bool = False
    skip_reason: str | None = None
    # Report-only static inventory outputs (C2). ``None`` for packages built
    # without an inventory (e.g. constructed directly in tests).
    component_coverage: dict[str, Any] | None = dataclass_field(default=None, compare=False, hash=False, repr=False)
    context_cost: dict[str, Any] | None = dataclass_field(default=None, compare=False, hash=False, repr=False)
    mcp_pinning: dict[str, Any] | None = dataclass_field(default=None, compare=False, hash=False, repr=False)

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
            "partial": bool(unresolved_skill or unresolved_rule or provider_only_mcp or mcp_unsupported_config),
            # Offline classification of declared skill/rule refs (same classifier as Tier 1).
            "dependency_status_counts": {
                **dict.fromkeys(DEPENDENCY_STATES, 0),
                **dict(self.dependency_status_counts),
            },
        }
        # Report-only: unsupported component types are listed here, never gated.
        if self.component_coverage is not None:
            provenance["component_coverage"] = self.component_coverage
        if self.context_cost is not None:
            provenance["context_cost"] = self.context_cost
        if self.mcp_pinning is not None:
            provenance["mcp_pinning"] = self.mcp_pinning
        return provenance

    def integration_evidence_error(self) -> str | None:
        """Explain why an Integration arm would not test composition."""
        if self.cross_component_case_count > 0:
            return None
        return (
            "Integration evaluation requires at least one dataset case with "
            "cross_component=true and two or more expected_skills"
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
    location = _manifest_location(plugin_path)
    manifest_path = location.path
    contained_form = _is_contained_manifest(manifest_path)
    plugin_dir = location.root
    plugin_root = location.secure_file.root
    manifest_text = location.read_text()
    manifest = _load_manifest_text(manifest_text, manifest_path)
    plugin_name = _plugin_name(manifest, plugin_dir)
    plugin_description = _plugin_description(manifest, plugin_name)

    # Layer-1 intra-repo resolver: canonical skill/rule refs whose <repo> is the
    # plugin's own clone are resolved to real dirs/files under the clone root
    # (widened, slug-verified containment); everything else stays unresolved.
    resolver = _make_intra_repo_resolver(plugin_dir, plugin_root, repo_root, stage_root)
    dependency_status_counts = (
        () if contained_form else tuple(dependency_status_counts_for_manifest(manifest, plugin_dir, repo_root).items())
    )

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
    # Bundle-reference plugins resolve their refs as before.
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
    # Static inventory of every declared/packaged component: supplies the MCP
    # declarations from all mcpServers forms (+ root .mcp.json) for staging and
    # the report-only coverage / context-cost / pinning provenance.
    inventory = build_plugin_inventory(
        plugin_root,
        manifest,
        contained=contained_form,
        manifest_rel=manifest_rel_for(manifest_path, plugin_dir),
    )
    runnable_mcp, provider_mcp, mcp_unsupported_config = _split_mcp_servers(
        manifest, inventory, contained_form, plugin_root=plugin_root
    )
    skipped = not (member_skills or staged_rules or runnable_mcp)
    report_only = _inventory_provenance(
        inventory,
        plugin_root=plugin_root,
        contained=contained_form,
        member_skills=member_skills,
        staged_rule_names=tuple(rule.name for rule in staged_rules),
        unresolved_skill_refs=unresolved_skill_refs,
        unresolved_rule_refs=unresolved_rule_refs,
        runnable_names=tuple(server["name"] for server in runnable_mcp),
        provider_names=tuple(server["name"] for server in provider_mcp),
        unsupported_config=tuple(mcp_unsupported_config),
        skipped=skipped,
    )

    # Optional-skip: nothing to evaluate locally in Phase 1. Honest skip rather
    # than a with-plugin run identical to baseline (a meaningless zero lift).
    if skipped:
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
            dependency_status_counts=dependency_status_counts,
            skipped=True,
            skip_reason=_skip_reason(unresolved_skill_refs, unresolved_rule_refs, provider_mcp, mcp_unsupported_config),
            **report_only,
        )

    package_path = _fresh_package_dir(stage_root, plugin_name)
    _stage_agent_plugin_manifest(
        package_path / "agent_plugin.yaml",
        manifest_text,
        manifest,
        contained_form=contained_form,
    )
    _write_plugin_skill_md(
        package_path / "SKILL.md",
        plugin_name=plugin_name,
        plugin_description=plugin_description,
        include_skills=member_skills,
        staged_rules=staged_rules,
        unresolved_skill_refs=unresolved_skill_refs,
        unresolved_rule_refs=unresolved_rule_refs,
        provider_mcp_servers=tuple(server["name"] for server in provider_mcp),
    )

    evals_dir = package_path / "evals"
    resolved_source = _resolve_evals_source(plugin_dir, evals_source)
    if resolved_source is not None:
        _copy_evals_source(resolved_source, evals_dir)
    else:
        _write_combined_member_evals(evals_dir, member_skills, plugin_name=plugin_name)

    dataset_path = next((evals_dir / name for name in _EVAL_DATASET_NAMES if (evals_dir / name).exists()), None)
    if dataset_path is None and not (evals_dir / "harbor").exists():
        raise ValueError(f"Prepared plugin package has no evaluation dataset: {package_path}")
    # Native Harbor sources can be valid for effectiveness without carrying the
    # structured composition metadata required for an Integration claim.
    dataset_cases = load_dataset_entries(dataset_path) if dataset_path is not None else []
    _reject_invalid_plugin_signal_fields(dataset_cases)
    cross_component_case_count = sum(
        1
        for case in dataset_cases
        if case.get("cross_component") is True
        and isinstance(case.get("expected_skills"), list)
        and len({str(name).strip() for name in case["expected_skills"] if str(name).strip()}) >= 2
    )

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
        dataset_case_count=len(dataset_cases),
        cross_component_case_count=cross_component_case_count,
        dependency_status_counts=dependency_status_counts,
        **report_only,
    )


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


def _inventory_provenance(
    inventory: PluginInventory,
    *,
    plugin_root: Path,
    contained: bool,
    member_skills: tuple[Path, ...],
    staged_rule_names: tuple[str, ...],
    unresolved_skill_refs: tuple[str, ...],
    unresolved_rule_refs: tuple[str, ...],
    runnable_names: tuple[str, ...],
    provider_names: tuple[str, ...],
    unsupported_config: tuple[str, ...],
    skipped: bool,
) -> dict[str, Any]:
    """Build the report-only C2 ``component_coverage`` / ``context_cost`` / ``mcp_pinning``."""
    member_resolved = {path.resolve() for path in member_skills}
    staged_rules = set(staged_rule_names)
    rows: list[dict[str, Any]] = []
    skip_note = "the plugin package was skipped (nothing locally evaluable)"
    for component in inventory.components:
        if component.problem is not None:
            rows.append(coverage_row(component, "invalid", problem_reason(component)))
            continue
        if component.type == "skill":
            if component.path is None:
                if component.name in unresolved_skill_refs:
                    rows.append(
                        coverage_row(component, "unavailable", "remote skill reference is not resolvable offline")
                    )
                elif skipped:
                    rows.append(coverage_row(component, "not_staged", skip_note))
                else:
                    rows.append(coverage_row(component, "staged", "skill reference resolved to a local member skill"))
            elif (plugin_root / component.path).resolve() in member_resolved:
                rows.append(coverage_row(component, "staged", "bundled skill staged as a plugin member skill"))
            else:
                rows.append(
                    coverage_row(
                        component,
                        "not_staged",
                        skip_note
                        if skipped
                        else "only skills under skills/ are staged by Tier 3; this declared "
                        "skill directory is inventoried only",
                    )
                )
        elif component.type == "rule":
            if component.path is None:
                if component.name in unresolved_rule_refs:
                    rows.append(
                        coverage_row(component, "unavailable", "remote rule reference is not resolvable offline")
                    )
                elif skipped:
                    rows.append(coverage_row(component, "not_staged", skip_note))
                else:
                    rows.append(
                        coverage_row(component, "staged", "rule reference resolved and embedded in the wrapper")
                    )
            elif {
                component.name,
                PurePosixPath(component.path).name,
                component.path.removeprefix("rules/"),
            } & staged_rules:
                rows.append(coverage_row(component, "staged", "rule embedded in the generated wrapper SKILL.md"))
            else:
                rows.append(
                    coverage_row(
                        component,
                        "not_staged",
                        skip_note if skipped else "rule file is inventoried but was not staged by Tier 3",
                    )
                )
        elif component.type == "mcp":
            rows.append(_mcp_coverage_row(component, contained, runnable_names, provider_names, unsupported_config))
        else:
            rows.append(
                coverage_row(
                    component,
                    "unsupported",
                    f"SkillEvaluator does not stage {component.type} components yet (inventoried only)",
                )
            )
    return {
        "component_coverage": summarize_coverage(rows),
        "context_cost": inventory.context_cost(
            extra_rows=_external_member_skill_costs(member_skills, plugin_root),
            extra_notes=(_TIER3_COVERAGE_NOTE,),
        ),
        "mcp_pinning": mcp_pinning_summary(inventory.mcp.effective),
    }


def _mcp_coverage_row(
    component: Any,
    contained: bool,
    runnable_names: tuple[str, ...],
    provider_names: tuple[str, ...],
    unsupported_config: tuple[str, ...],
) -> dict[str, Any]:
    if component.bundle:
        return coverage_row(component, "unsupported", "MCP bundles (.mcpb/.dxt) are not unpacked or staged")
    declaration = component.mcp
    if declaration is not None and declaration.source == "mcp_json" and not contained:
        return coverage_row(
            component, "not_staged", "agent_plugin.yaml plugins stage MCP servers from their 'mcp' list only"
        )
    if component.name in runnable_names:
        reason = "runnable MCP server staged for the with-plugin arm only"
        if component.name in unsupported_config:
            reason += "; its env/headers are not applied by the runtime (run reported INCOMPLETE)"
        return coverage_row(component, "staged", reason)
    if component.name in unsupported_config:
        # Not runnable: _split_mcp_servers keeps plugin-file launches out of the toml.
        return coverage_row(
            component,
            "unsupported",
            "MCP server launches from plugin files (${CLAUDE_PLUGIN_ROOT} or a relative path) that Tier 3 "
            "does not stage into the task environment; not started (run reported INCOMPLETE)",
        )
    if component.name in provider_names:
        return coverage_row(component, "unavailable", "provider-only MCP server is not runnable offline")
    return coverage_row(component, "not_staged", "MCP server was not staged")


def _external_member_skill_costs(member_skills: tuple[Path, ...], plugin_root: Path) -> list[CostRow]:
    """Context-cost rows for member skills staged from outside the plugin root."""
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
                    len(parsed.name or "") + len(parsed.description or ""),
                    len(parsed.body),
                    "member skill staged from outside the plugin root; always-on: name + description; "
                    "on-demand: SKILL.md body",
                )
            )
            break
    return rows


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


def _is_contained_manifest(path: Path) -> bool:
    """Whether *path* is a contained-plugin manifest (``.claude-plugin/plugin.json``).

    Mirrors the Tier 1 detection (``cli_core._is_contained_plugin_manifest``) so
    the plugin-eval path accepts exactly the manifest forms Tier 1 does.
    """
    return path.name == PLUGIN_CONTAINED_MANIFEST_FILE and path.parent.name == PLUGIN_CONTAINED_MANIFEST_DIR


def _manifest_path(plugin_path: Path) -> Path:
    return _manifest_location(plugin_path).path


def _manifest_location(plugin_path: Path) -> PluginManifestLocation:
    from skillevaluator.plugin_manifest import locate_plugin_manifest

    located = locate_plugin_manifest(plugin_path)
    if located is None:
        raise ValueError(f"No agent_plugin.yaml or .claude-plugin/plugin.json found under {plugin_path}")
    return located


def _load_manifest_text(raw_text: str, manifest_path: Path) -> dict[str, Any]:
    try:
        data = (
            load_bounded_json(raw_text.lstrip("\ufeff"))
            if _is_contained_manifest(manifest_path)
            else load_bounded_yaml(raw_text)
        )
    except StructuredDataError as exc:
        syntax = "JSON" if _is_contained_manifest(manifest_path) else "YAML"
        raise ValueError(f"{manifest_path} is not valid bounded {syntax}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a manifest object")
    return data


def _plugin_name(manifest: dict[str, Any], plugin_dir: Path) -> str:
    raw_name = manifest.get("name")
    if raw_name is None or (isinstance(raw_name, str) and not raw_name.strip()):
        raw_name = plugin_dir.name
    return require_bounded_string(raw_name, "Plugin manifest name", max_chars=NAME_MAX_LENGTH).strip()


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
    * the ref names a public remote source (github/git), AND
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
    active = clone_root != plugin_root.resolve()
    local_slug = _local_repo_slug(clone_root) if active else None
    snapshot_root = stage_root.expanduser().absolute() / ".intra-repo-snapshots"
    return _IntraRepoResolver(
        clone_root=clone_root,
        local_slug=local_slug,
        active=active,
        snapshot_root=snapshot_root,
    )


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
) -> tuple[tuple[_StagedRule, ...], tuple[str, ...], tuple[str, ...]]:
    """Resolve rule refs to contained files; report remote/unresolved ones.

    Returns ``(staged_rule_files, unresolved_labels, all_labels)``. A rule file is
    staged when it resolves to a real file inside the plugin root (symlink-
    contained, mirroring :func:`find_bundled_plugin_skills`) OR when a canonical
    remote ref names *this* clone and resolves intra-repo under the clone root.
    Each staged rule is read once, here, through a bounded no-follow read, and
    the staged set shares the contained ``rules/`` aggregate byte bound.
    """
    staged: list[_StagedRule] = []
    unresolved: list[str] = []
    all_labels: list[str] = []
    seen: set[Path] = set()
    total_bytes = 0

    def _stage(path: Path) -> None:
        nonlocal total_bytes
        rule = _load_rule_path(path)
        # Every staged body is held in memory and embedded in the wrapper (the
        # with-plugin arm's skill context), so MAX_PLUGIN_MANIFEST_ITEMS
        # per-file-bounded refs must not add up to a wrapper the contained
        # rules/ form would reject.
        total_bytes += len(rule.content.encode("utf-8"))
        _reject_rules_over_total_limit(total_bytes)
        staged.append(rule)
        seen.add(path)

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
                _stage(intra)
            continue
        resolved = _resolve_contained_file(ref, plugin_dir, plugin_root)
        if resolved is not None and resolved not in seen:
            _stage(resolved)
        elif resolved is None:
            unresolved.append(label)
    return tuple(staged), tuple(unresolved), tuple(all_labels)


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


def _reject_rules_over_total_limit(total_bytes: int) -> None:
    """Fail closed once staged plugin rules exceed the shared aggregate byte bound."""
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
        if len(files) > MAX_PLUGIN_MANIFEST_ITEMS:
            raise ValueError(f"Plugin rules exceed the {MAX_PLUGIN_MANIFEST_ITEMS}-file limit")
        _reject_rules_over_total_limit(sum(file.metadata.st_size for file in files))
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


def _reject_unsafe_mcp_declaration(name: Any, config: dict[str, Any]) -> None:
    """Fail closed before a runnable MCP declaration reaches Harbor.

    The direct Tier 3 command does not run Tier 1 first. Reuse the same network-free
    declaration policy here so shell smuggling, insecure endpoints, inline secrets,
    malformed transports, and other blocking findings can never be executed merely
    because the caller selected Tier 3 directly.
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

    blocking = [
        finding
        for finding in validate_mcp_server_declaration(safe_name, config, "<plugin manifest>")
        if finding.severity in (Severity.CRITICAL, Severity.HIGH)
    ]
    if blocking:
        first = blocking[0]
        raise ValueError(
            f"Plugin manifest MCP server '{safe_name}' failed blocking static validation "
            f"({first.check_name}): {first.message}"
        )


def _launches_from_plugin_files(config: dict[str, Any]) -> bool:
    """Whether a runnable MCP server's launch config points into the plugin tree.

    The documented form for a server shipped inside a plugin is
    ``"command": "${CLAUDE_PLUGIN_ROOT}/servers/x"``, which Claude Code expands when
    it loads the installed plugin. The eval runtime passes the staged config to
    the agent verbatim and packages only the generated wrapper, member skills,
    and evals, so ``${CLAUDE_PLUGIN_ROOT}`` / ``${CLAUDE_PLUGIN_DATA}``, a ``./`` or
    ``../`` path in ``command``/``args``/``cwd`` (including ``--flag=./path``), or a
    relative ``command`` path all name files the with-plugin arm does not have.
    """
    args = config.get("args")
    launch = [config.get("command"), config.get("cwd"), *(args if isinstance(args, list) else ())]
    values = [value.strip() for value in launch if isinstance(value, str)]
    url = config.get("url")
    if any(_PLUGIN_INSTALL_VAR_RE.search(value) for value in (*values, url if isinstance(url, str) else "")):
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


def _unstaged_root_mcp_json(manifest: dict[str, Any], plugin_root: Path) -> str | None:
    """Finding path of an ``agent_plugin.yaml`` plugin's implicit root ``.mcp.json``, or ``None``.

    Mirrors :func:`~skillevaluator.plugin_components.collect_mcp_declarations`: the
    root file is inventoried as the default ``mcp_json`` source (never staged for
    this manifest form) unless an ``mcpServers`` path names it explicitly -- then it
    is a staged ``path_ref`` source whose findings must keep blocking.
    """
    declared = manifest.get("mcpServers")
    entries = declared if isinstance(declared, list) else [declared]
    if any(isinstance(entry, str) and normalize_declared_path(entry).rel == MCP_JSON for entry in entries):
        return None
    return PluginRootReader(plugin_root).display(MCP_JSON)


def _normalize_mcp_entries(
    manifest: dict[str, Any],
    inventory: PluginInventory,
    contained_form: bool,
    *,
    plugin_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Normalize every manifest MCP form into the bundle-reference list shape.

    Bundle-reference ``agent_plugin.yaml`` uses a top-level ``mcp`` *list* of
    ``{name, provider}`` / ``{name, command|url, transport}`` objects. A contained
    ``.claude-plugin/plugin.json`` declares servers in any documented Claude Code
    form -- an inline ``mcpServers`` map, a ``.json`` path, or an array mixing
    both -- merged over the root ``.mcp.json`` (a later same-name server replaces
    an earlier one). The inventory collected those declarations through the
    bounded, no-follow plugin-root reader; here every declaration (including a
    shadowed one) must pass the blocking static checks before the effective
    servers are flattened to the list shape :func:`_split_mcp_servers` expects.
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
            name = entry.get("name")
            _reject_unsafe_mcp_declaration(name, {key: value for key, value in entry.items() if key != "name"})
            normalized_entries.append(entry)
        return normalized_entries

    collection = inventory.mcp
    # Fail closed on config-source problems: escapes, absolute paths, symlinks,
    # missing/oversize/invalid config files, and malformed mcpServers values --
    # in the sources that will be staged. An agent_plugin.yaml plugin's implicit
    # root .mcp.json is inventoried (Tier 1 reports it) but never staged, so it
    # blocks here no more than it does when the manifest has an 'mcp' list.
    # Without the plugin root the finding cannot be attributed: keep it (fail closed).
    blocking = collection.blocking_source_findings
    if not contained_form and plugin_root is not None:
        unstaged = _unstaged_root_mcp_json(manifest, plugin_root)
        blocking = [finding for finding in blocking if finding.file_path != unstaged]
    if blocking:
        first = blocking[0]
        raise ValueError(
            f"Plugin MCP configuration failed blocking static validation ({first.check_name}): {first.message}"
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
        _reject_unsafe_mcp_declaration(declaration.name, declaration.config)

    normalized: list[dict[str, Any]] = []
    for declaration in collection.effective:
        if not _stageable(declaration.source):
            continue
        config = declaration.config
        safe_name = require_bounded_string(
            declaration.name,
            "Plugin MCP server name",
            max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
        ).strip()
        entry: dict[str, Any] = {"name": safe_name}
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


def _split_mcp_servers(
    manifest: dict[str, Any],
    inventory: PluginInventory,
    contained_form: bool,
    *,
    plugin_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """Split MCP entries into runnable (command/url) vs provider-only.

    Canonical ``PluginMcpEntry`` entries carry ``name`` + ``provider`` and are
    *not* runnable offline (returned as provider-only, contributing nothing to
    the run). Entries with a ``command``/``url`` are a documented local-testing
    extension and are staged with-plugin-only -- unless they launch from plugin
    files the eval does not stage (:func:`_launches_from_plugin_files`); those
    are returned only in the unsupported-config list.
    """
    raw_servers = _normalize_mcp_entries(manifest, inventory, contained_form, plugin_root=plugin_root)

    runnable: list[dict[str, Any]] = []
    provider_only: list[dict[str, str]] = []
    unsupported_config: list[str] = []
    for idx, raw in enumerate(raw_servers):
        if not isinstance(raw, dict):
            raise ValueError(f"Plugin manifest mcp[{idx}] must be an object")
        name = require_bounded_string(
            raw.get("name"),
            f"Plugin manifest mcp[{idx}].name",
            max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
        ).strip()
        if (raw.get("command") or raw.get("url")) and _launches_from_plugin_files(raw):
            # Staged verbatim, this server could not start in the with-plugin arm,
            # yet it would count as runnable: a plugin whose only component it is
            # would run a "complete" evaluation with a meaningless zero lift instead
            # of an honest skip. Keep it out of the toml and the skip decision, and
            # record it as config the runtime cannot apply (run INCOMPLETE).
            unsupported_config.append(name)
        elif raw.get("command") or raw.get("url"):
            server: dict[str, Any] = {"name": name}
            for key in ("url", "command", "transport"):
                if raw.get(key):
                    server[key] = require_bounded_string(
                        raw[key],
                        f"Plugin manifest mcp[{idx}].{key}",
                        max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
                    )
            if raw.get("args"):
                args = raw["args"]
                if not isinstance(args, list) or len(args) > MAX_PLUGIN_MANIFEST_ITEMS:
                    raise ValueError(f"Plugin manifest mcp[{idx}].args must be a bounded list")
                server["args"] = [
                    require_bounded_string(
                        arg,
                        f"Plugin manifest mcp[{idx}].args[{arg_index}]",
                        max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
                        allow_empty=True,
                    )
                    for arg_index, arg in enumerate(args)
                ]
            if "command" in server and "transport" not in server:
                server["transport"] = "stdio"
            runnable.append(server)
            if raw.get("_unsupported_fields"):
                unsupported_config.append(name)
        else:
            provider = raw.get("provider") or ""
            provider_only.append(
                {
                    "name": name,
                    "provider": require_bounded_string(
                        provider,
                        f"Plugin manifest mcp[{idx}].provider",
                        max_chars=MAX_PLUGIN_MANIFEST_TEXT_CHARS,
                        allow_empty=True,
                    ),
                }
            )
    return runnable, provider_only, unsupported_config


def _skip_reason(
    unresolved_skill_refs: tuple[str, ...],
    unresolved_rule_refs: tuple[str, ...],
    provider_mcp: list[dict[str, str]],
    unsupported_mcp: list[str] | tuple[str, ...] = (),
) -> str:
    parts: list[str] = []
    if unresolved_skill_refs:
        parts.append(f"{len(unresolved_skill_refs)} remote skill ref(s)")
    if unresolved_rule_refs:
        parts.append(f"{len(unresolved_rule_refs)} remote rule ref(s)")
    if provider_mcp:
        parts.append(f"{len(provider_mcp)} provider-only MCP server(s)")
    if unsupported_mcp:
        # In a skipped package these are only plugin-file launches: a server with
        # just unapplied env/headers is still runnable, so it never reaches here.
        parts.append(f"{len(unsupported_mcp)} MCP server(s) launched from unstaged plugin files")
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
            "name": plugin_name,
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
