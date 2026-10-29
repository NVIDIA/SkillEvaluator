# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private discovery of declared and packaged members across plugin manifest formats.

Discovery reports declarations, origins, bounded source text/configuration and
path/read failures. It never invokes MCP/hook policies, estimates scores or
context cost, selects a runtime, or registers a public validation workflow.
The records remain distinct from proof that a component was staged or loaded.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NamedTuple

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_DISCOVERED_PATHS,
    CONTENT_DEDUP_MAX_FILE_BYTES,
    PLUGIN_COMPONENT_MAX_ITEMS,
    PLUGIN_CONFIG_MAX_BYTES,
    SCAN_ARTIFACT_DIRS,
    SCAN_EXCLUDED_DIRS,
    SKILL_MANIFEST_VARIANTS,
)
from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_formats import (
    CODEX_PROFILE,
    DEFAULT_SKILLS_DIR,
    FormatProfile,
    codex_accepts_path,
    declared_value_replaces_default,
    normalized_component_manifest,
    profile_for,
)
from skillevaluator.plugin_manifest import PluginManifestFile, PluginManifestPathError, locate_plugin_manifest
from skillevaluator.plugin_markdown import parse_markdown
from skillevaluator.plugin_mcp import McpDeclaration, collect_mcp_declarations
from skillevaluator.plugin_member_paths import (
    _unscanned_skill_finding,
    dependency_folder_skill_findings,
    find_dependency_folder_skills,
    find_environment_files,
)
from skillevaluator.plugin_paths import (
    DeclaredPath,
    PluginRootReader,
    _in_unscanned_folder,
    _path_problem_finding,
    _plugin_finding,
    _root_variable_finding,
    _style_finding,
    _unscanned_packaged_finding,
    _unscanned_path_finding,
    normalize_declared_path,
)
from skillevaluator.plugin_refs import normalize_ref
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_json
from skillevaluator.validators.url_policy import report_text

Support = Literal["evaluated", "static_only", "unsupported"]
Origin = Literal["declared", "packaged", "declared+packaged"]


@dataclass
class PluginMember:
    """One declaration or client-visible packaged member, without policy or runtime conclusions."""

    type: str
    name: str
    origin: Origin
    path: str | None
    support: Support
    problem: str | None = None
    manifest_type: str = ""
    declared_by: str = ""
    frontmatter: dict[str, Any] = field(default_factory=dict)
    frontmatter_error: str | None = None
    text: str | None = None
    config: Any = None
    mcp: McpDeclaration | None = None
    bundle: bool = False
    effective: bool = True
    # False means a later declaration overrides this record in the same client view.


@dataclass
class PluginDiscovery:
    """Passive members and visible discovery failures for one or more manifest views."""

    components: list[PluginMember] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    unread_files: int = 0
    manifests: tuple[PluginManifestFile, ...] = ()
    environment_files: tuple[PurePosixPath, ...] = ()
    environment_scan_complete: bool = True

    @property
    def complete(self) -> bool:
        """Whether discovery recorded no blocked read, invalid path, or exhausted scan.

        This is a discovery result, not schema, policy, or runtime acceptance.
        """
        return (
            self.environment_scan_complete
            and not self.unread_files
            and not any(member.problem for member in self.components)
            and not any(
                f.severity in (Severity.HIGH, Severity.CRITICAL)
                or f.check_name.endswith(("scan_truncated", "scan_incomplete", "list_truncated"))
                for f in self.findings
            )
        )


_TYPE_SUPPORT: dict[str, Support] = {
    "skill": "evaluated",
    "rule": "evaluated",
    "mcp": "evaluated",
    "hook": "unsupported",
    "agent": "unsupported",
    "command": "unsupported",
    "lsp": "unsupported",
    "output_style": "unsupported",
    "monitor": "unsupported",
    "settings": "unsupported",
    "app": "unsupported",
    "extension": "unsupported",
}


_NAMESPACE_DIR_RE = re.compile(r"^[a-z][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")


_CODEX_IGNORED_SCALAR_FIELDS = frozenset({"skills", "commands", "hooks"})


class _JsonSource(NamedTuple):
    """One loaded source of a JSON-config field (hooks, LSP servers, monitors, Codex apps)."""

    name: str  # the file, or "inline" / "inline[<index>]" for a value written in the manifest
    origin: Origin
    rel: str  # root-relative file that holds the config (the manifest for an inline value)
    config: Any  # the parsed config; None when the file could not be read or parsed safely


class _ResolvedPath(NamedTuple):
    """A declared component path that can be loaded (:meth:`_MemberCollector._resolve_component`)."""

    rel: PurePosixPath  # contained root-relative path
    kind: str  # "file" or "dir"


def _section_refs(section: Any) -> list[Any]:
    if isinstance(section, dict):
        refs = section.get("refs")
    else:
        refs = section
    if not isinstance(refs, list):
        return []
    return refs


def _ref_label(ref: Any) -> str:
    """Same label Tier 3 uses for unresolved refs (canonical ID or trailing name)."""
    canonical = normalize_ref(ref)
    if canonical:
        return canonical
    if isinstance(ref, dict) and isinstance(ref.get("path"), str):
        return ref["path"].strip().split("/")[-1] or repr(ref)
    return str(ref)


class _MemberCollector:
    def __init__(
        self,
        reader: PluginRootReader,
        manifest: dict[str, Any] | None,
        *,
        contained: bool,
        manifest_rel: str,
        profile: FormatProfile,
    ) -> None:
        self.reader, self.manifest = reader, manifest
        self.contained, self.manifest_rel, self.profile = contained, manifest_rel, profile
        self.inventory = PluginDiscovery()
        self._keys: dict[tuple[str, str, str], PluginMember] = {}
        self._type_counts: Counter[str] = Counter()
        self._truncated: set[str] = set()

    def collect(self) -> PluginDiscovery:
        self.skills()
        self.rules()
        self.mcp()
        self.hooks()
        self.markdown_components("agent", "agents", self.profile.default_agents_dir)
        self.markdown_components("command", "commands", self.profile.default_commands_dir)
        self.markdown_components("output_style", "outputStyles", self.profile.default_output_styles_dir)
        self.lsp()
        self.monitors()
        self.settings()
        self.apps()
        self.extensions()
        self.openai_extension()
        return self.inventory

    @property
    def manifest_display(self) -> str:
        return self.reader.display(self.manifest_rel)

    def _add(self, component: PluginMember) -> PluginMember:
        component.manifest_type = component.manifest_type or self.profile.manifest_type
        component.declared_by = self.manifest_rel
        key = (component.type, component.path or "", component.name)
        existing = self._keys.get(key)
        if existing is not None and component.type not in {"mcp", "lsp"}:
            if existing.origin != component.origin:
                existing.origin = "declared+packaged"
            return existing
        if self._type_counts[component.type] >= PLUGIN_COMPONENT_MAX_ITEMS:
            if component.type not in self._truncated:
                self._truncated.add(component.type)
                self.inventory.findings.append(
                    _plugin_finding(
                        Severity.LOW,
                        "plugin_component_scan_truncated",
                        f"More than {PLUGIN_COMPONENT_MAX_ITEMS} {component.type} components; the inventory lists "
                        "only the first ones",
                        self.reader.display("."),
                        "Split very large plugins; the static inventory is bounded.",
                    )
                )
            return component
        self._keys[key] = component
        self._type_counts[component.type] += 1
        self.inventory.components.append(component)
        return component

    def _broken(self, component_type: str, declared: DeclaredPath | None, raw: Any, problem: str) -> None:
        path = declared.rel.as_posix() if declared is not None and declared.rel is not None else None
        name = raw if isinstance(raw, str) else repr(raw)
        self._add(PluginMember(component_type, name, "declared", path, _TYPE_SUPPORT[component_type], problem=problem))

    def _declared_field(self, field_name: str) -> Any:
        if self.manifest is None or not self.contained:
            return None
        return self.manifest.get(field_name)

    def _declared_values(self, field_name: str, value: Any) -> list[Any]:
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        self._check_item_count(f"'{field_name}'", len(values))
        return values[:PLUGIN_COMPONENT_MAX_ITEMS]

    def _check_item_count(
        self, what: str, count: int, rel: str | None = None, *, limit: int = PLUGIN_COMPONENT_MAX_ITEMS
    ) -> None:
        """Expose every bounded-list cut before declarations or directory entries are dropped."""
        if count <= limit:
            return
        self.inventory.findings.append(
            _plugin_finding(
                Severity.HIGH,
                "plugin_component_list_truncated",
                f"{what} has {count} entries; only the first {limit} are inspected. Discovery is incomplete.",
                self.reader.display(rel or self.manifest_rel),
                f"Keep this declaration list or directory scan within its {limit}-entry limit.",
            )
        )

    def _declared_replaces(self, field_name: str, value: Any) -> bool:
        """Whether a declared field suppresses default-folder discovery for its type.

        Codex drops a path it does not accept and loads the default location
        instead (:func:`~skillevaluator.plugin_formats.declared_value_replaces_default`).
        """
        return declared_value_replaces_default(self.profile, field_name, value)

    def _codex_drops(self, raw: Any, profile: FormatProfile) -> bool:
        """Whether Codex ignores this declared path value (and loads the default location instead)."""
        return self.contained and profile.codex_path_rules and not codex_accepts_path(raw)

    def _resolve_declared(
        self,
        field_name: str,
        raw: Any,
        *,
        kinds: tuple[str, ...] = ("file", "dir"),
        wrong_kind_field: str | None = None,
        style: bool = True,
        profile: FormatProfile | None = None,
    ) -> tuple[DeclaredPath | None, str]:
        """Normalize + classify one declared path and record its findings: ``(path, kind or problem)``.

        The second value is the path's kind when it is one of ``kinds``, and
        otherwise the problem: ``escape``, ``invalid``, ``missing``, or
        ``unsafe``. A file where the field takes a folder (or the reverse) is
        ``invalid``; its finding names ``wrong_kind_field`` when given. The
        kind is ``dropped`` when the client ignores the value: Codex drops any
        path it does not accept (no leading ``./``, ``./`` alone, ``..``, a root
        variable) and loads the default location instead, so the value names
        no component. Its findings are still reported. ``profile`` overrides
        the builder's format profile (the Codex rules of
        ``extensions["com.openai"]``).
        """
        profile = profile or self.profile
        dropped = self._codex_drops(raw, profile)
        if not isinstance(raw, str):
            if dropped and field_name in _CODEX_IGNORED_SCALAR_FIELDS:
                # Codex drops a skills, commands, or hooks value of the wrong type, loads the default location,
                # and still installs the plugin: MEDIUM, as check 1 rates it for the Codex client.
                self.inventory.findings.append(
                    _plugin_finding(
                        Severity.MEDIUM,
                        "plugin_component_path_invalid",
                        f"'{field_name}' entry {report_text(repr(raw), 80)} is not a path; Codex ignores the value, "
                        "loads the default location, and still installs the plugin",
                        self.reader.display(self.manifest_rel),
                        f"Write '{field_name}' as a './'-relative path (or a list of them), or remove it to use the "
                        "default location.",
                        metadata={"plugin_component_ref": repr(raw)},
                    )
                )
                return None, "dropped"
            finding = _path_problem_finding(
                self.reader, field_name, DeclaredPath(repr(raw), None), self.manifest_rel, "invalid"
            )
            self.inventory.findings.append(finding)
            return None, "dropped" if dropped else "invalid"
        declared = normalize_declared_path(raw, profile.manifest_path_prefixes)
        if declared.problem is not None or declared.rel is None:
            problem = declared.problem if declared.problem == "escape" else "invalid"
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader, field_name, declared, self.manifest_rel, problem, reference=profile.reference
                )
            )
            return declared, "dropped" if dropped else problem
        if dropped:
            if style:
                self.inventory.findings.append(
                    _style_finding(self.reader, field_name, declared, self.manifest_rel, profile=profile)
                )
            return declared, "dropped"
        if style and self.contained and profile.require_dot_relative and not declared.dot_relative:
            self.inventory.findings.append(
                _style_finding(self.reader, field_name, declared, self.manifest_rel, profile=profile)
            )
        if self.contained and declared.root_variable is not None:
            self.inventory.findings.append(
                _root_variable_finding(self.reader, field_name, declared, self.manifest_rel, profile)
            )
        # Skill folders a client loads are scanned as skill units even in such a folder
        # (client_skill_dirs_outside_tree_scans), so only other components get this.
        if field_name != "skills" and (
            unscanned := _unscanned_path_finding(self.reader, field_name, declared, self.manifest_rel)
        ):
            self.inventory.findings.append(unscanned)
        if self.contained and (wrong_kind := self._wrong_file_kind(field_name, raw, profile)) is not None:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_path_invalid",
                    f"'{field_name}' entry {raw!r} must name a {wrong_kind} file; "
                    f"{profile.label.removesuffix(' plugin')} rejects the whole manifest otherwise and does not "
                    "load the plugin",
                    self.reader.display(self.manifest_rel),
                    f"List each {wrong_kind} file by its own './'-relative path, as the {profile.reference} requires.",
                    metadata={"plugin_component_ref": raw},
                )
            )
            return declared, "invalid"
        kind = self.reader.kind(declared.rel)
        if kind == "missing":
            self.inventory.findings.append(
                _path_problem_finding(self.reader, field_name, declared, self.manifest_rel, "missing")
            )
            return declared, "missing"
        if kind in {"link", "special"}:
            self.inventory.findings.append(
                _path_problem_finding(self.reader, field_name, declared, self.manifest_rel, "unsafe", declared.rel)
            )
            return declared, "unsafe"
        if kind not in kinds:
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader, wrong_kind_field or field_name, declared, self.manifest_rel, "invalid"
                )
            )
            return declared, "invalid"
        return declared, kind

    @staticmethod
    def _wrong_file_kind(field_name: str, raw: str, profile: FormatProfile) -> str | None:
        """The file kind a field's paths must name when ``raw`` does not (Claude Code: ``.md`` agents, ``.json`` configs).

        The client's schema compares the raw text, so a folder (``./agents/``)
        or another suffix fails it.
        """
        if field_name in profile.markdown_file_fields and not raw.endswith(".md"):
            return "Markdown (.md)"
        if field_name in profile.json_file_fields and not raw.endswith(".json"):
            return "JSON (.json)"
        return None

    def _resolve_component(
        self,
        component_type: str,
        field_name: str,
        raw: Any,
        *,
        kinds: tuple[str, ...] = ("file", "dir"),
        name: str | None = None,
        wrong_kind_field: str | None = None,
        style: bool = True,
    ) -> _ResolvedPath | None:
        """A declared component path that can be loaded, or ``None`` when it cannot.

        A path that cannot be loaded gets its finding (:meth:`_resolve_declared`)
        and a broken ``component_type`` component named ``name`` (by default
        the declared value) in its place. A value the client drops (Codex loads
        the default location instead) names no component, so it gets none.
        """
        declared, kind = self._resolve_declared(
            field_name, raw, kinds=kinds, wrong_kind_field=wrong_kind_field, style=style
        )
        if kind in kinds and declared is not None and declared.rel is not None:
            return _ResolvedPath(declared.rel, kind)
        if kind != "dropped":
            self._broken(component_type, declared, raw if name is None else name, kind)
        return None

    def _list_dir(
        self, component_type: str, rel_dir: PurePosixPath, suffixes: tuple[str, ...] | None
    ) -> list[PurePosixPath] | None:
        """List a component folder, nested folders included (see :meth:`PluginRootReader.list_files`).

        A folder that holds an unsafe entry is HIGH, with a broken row, and gives ``None``.
        """
        try:
            return self.reader.list_files(rel_dir, suffixes=suffixes)
        except SecurePathError as exc:
            offending = exc.relative_path if exc.relative_path not in {"", "."} else ""
            location = (rel_dir / offending) if offending else rel_dir
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_path_unsafe",
                    f"{component_type} directory '{rel_dir.as_posix()}' contains an unsafe entry "
                    f"({exc}); it was not followed",
                    self.reader.display(location),
                    "Replace symlinks, hard links, and special files with regular contained files.",
                )
            )
            self._add(
                PluginMember(
                    component_type,
                    rel_dir.as_posix(),
                    "packaged",
                    rel_dir.as_posix(),
                    _TYPE_SUPPORT[component_type],
                    problem="unsafe",
                )
            )
            return None

    def _read(self, rel: PurePosixPath, max_bytes: int = CONTENT_DEDUP_MAX_FILE_BYTES) -> str | None:
        try:
            return self.reader.read_text(rel, max_bytes)
        except (SecurePathError, OSError):
            self.inventory.unread_files += 1
            return None

    def _read_lenient(self, rel: PurePosixPath) -> str | None:
        """Like :meth:`_read`, but a file that is not UTF-8 is decoded with replacement characters.

        Claude Code still lists such a component (a Latin-1 agent shows up with
        U+FFFD where the bytes were), so the cost estimate must count it.
        """
        try:
            return self.reader.read_text(rel, CONTENT_DEDUP_MAX_FILE_BYTES)
        except SecurePathError as exc:
            if exc.code != "invalid_text_encoding":
                self.inventory.unread_files += 1
                return None
        except OSError:
            self.inventory.unread_files += 1
            return None
        try:
            raw = self.reader._read_bytes(rel, CONTENT_DEDUP_MAX_FILE_BYTES)
        except (SecurePathError, OSError):
            self.inventory.unread_files += 1
            return None
        return raw.decode("utf-8-sig", errors="replace")

    def _load_json(self, rel: PurePosixPath, field_name: str) -> Any:
        try:
            text = self.reader.read_text(rel, PLUGIN_CONFIG_MAX_BYTES, config=True)
            return load_bounded_json(text)
        except (SecurePathError, StructuredDataError, OSError) as exc:
            # Blocking: an unread hooks/LSP/monitor/settings config skips the
            # permission-bypass and env-override checks, so fail closed.
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_unreadable",
                    f"{field_name} config '{rel.as_posix()}' could not be read or parsed safely: {exc}",
                    self.reader.display(rel),
                    f"Keep '{rel.as_posix()}' a regular UTF-8 JSON file under {PLUGIN_CONFIG_MAX_BYTES} bytes.",
                )
            )
            return None

    def _json_sources(
        self,
        component_type: str,
        field_name: str,
        declared_value: Any,
        default: PurePosixPath | None,
        *,
        merge_default: bool = True,
        profile: FormatProfile | None = None,
    ) -> Iterator[_JsonSource]:
        """The loaded sources of a JSON-config field: its default file, inline values, and declared files.

        ``hooks`` and ``lspServers`` merge with their default file; monitors replace it
        (``merge_default=False``) when declared. For Codex, a declared value
        replaces the default only when Codex keeps it; otherwise the default
        file is loaded, as Codex does, and a declared path Codex drops is not
        yielded. A declared path that cannot be loaded is not yielded either:
        it gets its finding and a broken ``component_type`` component without a
        path. ``profile`` overrides the builder's format profile (the Codex
        rules of ``extensions["com.openai"]``).
        """
        profile = profile or self.profile
        declared_values = self._declared_values(field_name, declared_value)
        explicit: set[PurePosixPath] = set()
        for raw in declared_values:
            if isinstance(raw, str) and not self._codex_drops(raw, profile):
                declared = normalize_declared_path(raw, profile.manifest_path_prefixes)
                if declared.rel is not None:
                    explicit.add(declared.rel)
        if merge_default:
            load_default = True
        elif profile.codex_path_rules:
            load_default = not declared_value_replaces_default(profile, field_name, declared_value)
        else:
            load_default = not declared_values
        default_kind = self.reader.kind(default) if load_default and default is not None else "missing"
        if default_kind == "file" and default not in explicit:
            config = self._load_json(default, field_name)
            yield _JsonSource(default.as_posix(), "packaged", default.as_posix(), config)
        elif default_kind in {"link", "special"} and default is not None:
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader,
                    field_name,
                    DeclaredPath(default.as_posix(), default),
                    self.manifest_rel,
                    "unsafe",
                    default,
                )
            )
        for index, raw in enumerate(declared_values):
            if isinstance(raw, dict | list) and not isinstance(raw, str):
                name = f"inline[{index}]" if len(declared_values) > 1 else "inline"
                yield _JsonSource(name, "declared", self.manifest_rel, raw)
                continue
            declared, kind = self._resolve_declared(field_name, raw, kinds=("file",), profile=profile)
            if kind == "dropped":
                continue  # the client ignores the value and loads the default file instead
            if kind != "file" or declared is None or declared.rel is None:
                self._broken(component_type, None, raw, kind)  # a broken config source has no path
                continue
            origin: Origin = "declared+packaged" if declared.rel == default else "declared"
            config = self._load_json(declared.rel, field_name)
            yield _JsonSource(declared.rel.as_posix(), origin, declared.rel.as_posix(), config)

    def skills(self) -> None:
        from skillevaluator.plugin_member_paths import find_bundled_plugin_skill_manifests

        declared_default = False
        declared_value = self.manifest.get("skills") if self.manifest is not None else None
        default_dir = PurePosixPath(DEFAULT_SKILLS_DIR)
        if self.contained and self.manifest is not None:
            for raw in self._declared_values("skills", declared_value):
                resolved = self._resolve_component("skill", "skills", raw, kinds=("dir",))
                if resolved is None:
                    continue
                if resolved.rel == default_dir:
                    declared_default = True
                    continue
                self._declared_skill_dir(resolved.rel)
                self._flag_dependency_folder_skills(resolved.rel)
        elif self.manifest is not None:
            refs = _section_refs(self.manifest.get("skills"))
            self._check_item_count("Skill references", len(refs))
            for ref in refs[:PLUGIN_COMPONENT_MAX_ITEMS]:
                self._add(PluginMember("skill", _ref_label(ref), "declared", None, "evaluated"))

        if self.contained and self._declared_replaces("skills", declared_value) and not declared_default:
            return  # the declared paths replace skills/ discovery for this format
        try:
            manifests = find_bundled_plugin_skill_manifests(self.reader.root)
        except ValueError as exc:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_path_unsafe",
                    str(exc),
                    self.reader.display(default_dir),
                    "Use regular contained skill directories.",
                )
            )
            manifests = []
        if self.profile.skills_immediate_children_only:
            manifests = [manifest for manifest in manifests if len(manifest.relative_path.parts) == 2]
        if (
            not manifests
            and declared_value is None
            and self.profile.root_skill_fallback
            and self.reader.kind(default_dir) == "missing"
        ):
            self._root_skill()
        self._check_item_count("Discovered skills", len(manifests))
        for manifest in manifests[:PLUGIN_COMPONENT_MAX_ITEMS]:
            skill_dir = default_dir / manifest.relative_path.parent.as_posix()
            origin: Origin = "declared+packaged" if declared_default else "packaged"
            component = self._add(
                PluginMember(
                    "skill", manifest.relative_path.parent.as_posix(), origin, skill_dir.as_posix(), "evaluated"
                )
            )
            self._read_member(component, skill_dir / manifest.relative_path.name)

    def _root_skill(self) -> None:
        """A root ``SKILL.md`` makes a single-skill plugin (Cursor) when no skills are declared or bundled."""
        for variant in SKILL_MANIFEST_VARIANTS:
            if self.reader.kind(PurePosixPath(variant)) == "file":
                component = self._add(PluginMember("skill", self.reader.root.name, "packaged", ".", "evaluated"))
                self._read_member(component, PurePosixPath(variant))
                return

    def _declared_skill_dir(self, rel_dir: PurePosixPath) -> None:
        """A declared skills dir holds ``SKILL.md`` directly or ``<name>/SKILL.md`` folders.

        Codex searches a declared skills folder at any depth, like ``skills/``.
        A child named ``evals``, ``results``, or ``versions`` is a skill folder
        like any other here (clients load it). The whole-tree walk prunes such
        folder names, so Tier 1 scans these skills as their own units
        (:func:`client_skill_dirs_outside_tree_scans`).
        """
        if self.profile.skills_recursive and str(rel_dir) != ".":
            self._declared_skill_tree(rel_dir)
            return
        candidates: list[PurePosixPath] = []
        for variant in SKILL_MANIFEST_VARIANTS:
            if self.reader.kind(rel_dir / variant) == "file":
                candidates.append(rel_dir / variant)
                break
        if not candidates:
            candidates = self._child_skill_manifests(rel_dir)
        for manifest_rel in candidates:
            skill_dir = manifest_rel.parent
            name = skill_dir.name if str(skill_dir) != "." else self.reader.root.name
            component = self._add(PluginMember("skill", name, "declared", skill_dir.as_posix(), "evaluated"))
            self._read_member(component, manifest_rel)

    def _child_skill_manifests(self, rel_dir: PurePosixPath) -> list[PurePosixPath]:
        """The ``SKILL.md`` of each skill folder directly in a declared skills dir.

        A linked child folder or skill manifest is reported and not followed.
        """
        start = self.reader.root if str(rel_dir) == "." else self.reader.root / rel_dir.as_posix()
        try:
            with os.scandir(start) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            entries = []
        manifests: list[PurePosixPath] = []
        self._check_item_count(
            "Skill directory entries", len(entries), rel_dir.as_posix(), limit=CONTENT_DEDUP_MAX_DISCOVERED_PATHS
        )
        for entry in entries[:CONTENT_DEDUP_MAX_DISCOVERED_PATHS]:
            if entry.name.startswith(".") or (
                entry.name in SCAN_EXCLUDED_DIRS and entry.name not in SCAN_ARTIFACT_DIRS
            ):
                continue  # hidden folders, VCS, virtualenv, package, and bytecode caches
            child = PurePosixPath(entry.name) if str(rel_dir) == "." else rel_dir / entry.name
            if str(rel_dir) == "." and entry.name == DEFAULT_SKILLS_DIR:
                continue  # the default skills/ scan covers it
            if entry.is_symlink():
                self.inventory.findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "plugin_component_path_unsafe",
                        f"skills directory entry '{child.as_posix()}' is a symlink; it was not followed",
                        self.reader.display(child),
                        "Replace linked skill directories with regular contained directories.",
                    )
                )
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
            for variant in SKILL_MANIFEST_VARIANTS:
                kind = self.reader.kind(child / variant)
                if kind == "file":
                    manifests.append(child / variant)
                    break
                if kind in {"link", "special"}:
                    self.inventory.findings.append(
                        _path_problem_finding(
                            self.reader,
                            "skills",
                            DeclaredPath((child / variant).as_posix(), child / variant),
                            self.manifest_rel,
                            "unsafe",
                            child / variant,
                        )
                    )
                    break
        return manifests

    def _declared_skill_tree(self, rel_dir: PurePosixPath) -> None:
        """Every skill below a declared folder, at any depth (Codex recursive discovery).

        The folder is searched like ``skills/``: an ``evals``, ``results``, or
        ``versions`` child is searched as a skill folder (Tier 1 scans it as its
        own unit), and a ``SKILL.md`` deeper in such a folder, inside a skill's
        own evaluation output or snapshots, is HIGH
        ``plugin_skill_in_unscanned_folder``, as it is under ``skills/``.
        """
        from skillevaluator.plugin_member_paths import (
            find_bundled_plugin_skill_manifests,
            find_unscanned_plugin_skill_manifests,
        )

        try:
            manifests = find_bundled_plugin_skill_manifests(self.reader.root, rel_dir.as_posix())
        except ValueError as exc:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_path_unsafe",
                    f"skills directory '{rel_dir.as_posix()}' contains an unsafe entry ({exc}); it was not followed",
                    self.reader.display(rel_dir),
                    "Replace symlinks, hard links, and special files with regular contained files.",
                )
            )
            self._add(
                PluginMember("skill", rel_dir.as_posix(), "declared", rel_dir.as_posix(), "evaluated", problem="unsafe")
            )
            return
        self._check_item_count("Discovered skills", len(manifests))
        for manifest in manifests[:PLUGIN_COMPONENT_MAX_ITEMS]:
            parent = PurePosixPath(manifest.relative_path.parent.as_posix())
            skill_dir = rel_dir / parent if str(parent) != "." else rel_dir
            name = parent.as_posix() if str(parent) != "." else rel_dir.name
            component = self._add(PluginMember("skill", name, "declared", skill_dir.as_posix(), "evaluated"))
            self._read_member(component, skill_dir / manifest.relative_path.name)
        for manifest_rel in find_unscanned_plugin_skill_manifests(self.reader.root, rel_dir.as_posix()):
            self._flag_unscanned_skill(rel_dir, manifest_rel)

    def _flag_unscanned_skill(self, rel_dir: PurePosixPath, manifest_rel: PurePosixPath) -> None:
        """HIGH for a skill deep in a skill's own artifact folder that Codex loads from declared folder ``rel_dir``.

        Skills under ``skills/`` are reported by the bundled-skill validator, so
        they are not repeated here.
        """
        if rel_dir.parts[:1] == (DEFAULT_SKILLS_DIR,):
            return
        if _in_unscanned_folder(manifest_rel.parent):
            self.inventory.findings.append(_unscanned_skill_finding(self.reader, manifest_rel))

    def _flag_dependency_folder_skills(self, rel_dir: PurePosixPath) -> None:
        """HIGH for a skill a client loads from a dependency folder inside declared skills folder ``rel_dir``.

        Skill discovery prunes ``node_modules/``, ``.venv/``, ``.git/``, and
        ``__pycache__/``, so such a skill is never listed or scanned
        (:func:`find_dependency_folder_skills`). Skills under ``skills/`` are
        reported by the bundled-skill validator, so they are not repeated here.
        """
        if rel_dir.parts[:1] == (DEFAULT_SKILLS_DIR,):
            return
        found, complete = find_dependency_folder_skills(
            self.reader.root, rel_dir.as_posix(), recursive=self.profile.skills_recursive
        )
        self.inventory.findings.extend(
            dependency_folder_skill_findings(self.reader.display, rel_dir.as_posix(), found, complete)
        )

    def rules(self) -> None:
        declared_default = False
        section = self.manifest.get("rules") if self.manifest is not None else None
        if self.contained and section is not None:
            for raw in self._declared_values("rules", section):
                if isinstance(raw, str) and "::" in raw:
                    self._add(PluginMember("rule", _ref_label(raw), "declared", None, "evaluated"))
                    continue
                resolved = self._resolve_component("rule", "rules", raw, style=False)
                if resolved is None:
                    continue
                if resolved.rel == PurePosixPath(self.profile.default_rules_dir or "rules"):
                    declared_default = True
                elif resolved.kind == "dir":
                    self._rule_dir(resolved.rel, "declared")
                else:
                    self._rule_file(resolved.rel, "declared")
        elif self.manifest is not None and not self.contained:
            # Bundle-reference refs are inventoried by label only; resolving them
            # (and flagging dangling ones) is the dependency-resolution report's job.
            refs = _section_refs(section)
            self._check_item_count("Rule references", len(refs))
            for ref in refs[:PLUGIN_COMPONENT_MAX_ITEMS]:
                self._add(PluginMember("rule", _ref_label(ref), "declared", None, "evaluated"))
        default_name = self.profile.default_rules_dir
        if default_name is None:
            return
        if self.contained and self._declared_replaces("rules", section) and not declared_default:
            return  # the declared paths replace rules/ discovery for this format
        default_dir = PurePosixPath(default_name)
        if self.reader.kind(default_dir) == "dir":
            self._rule_dir(default_dir, "declared+packaged" if declared_default else "packaged")
        elif self.reader.kind(default_dir) == "link":
            self.inventory.findings.append(
                _path_problem_finding(
                    self.reader,
                    "rules",
                    DeclaredPath(default_name, default_dir),
                    self.manifest_rel,
                    "unsafe",
                    default_dir,
                )
            )
            self._add(PluginMember("rule", default_name, "packaged", default_name, "evaluated", problem="unsafe"))

    def _rule_dir(self, rel_dir: PurePosixPath, origin: Origin, *, support: Support = "evaluated") -> None:
        """Every rule file in a rules folder, at any depth; one in a folder the scans skip is also HIGH."""
        files = self._list_dir("rule", rel_dir, self.profile.rule_suffixes)
        for rel in files or []:
            if (unscanned := _unscanned_packaged_finding(self.reader, "rule", rel, rel_dir)) is not None:
                self.inventory.findings.append(unscanned)
            self._rule_file(rel, origin, name=rel.relative_to(rel_dir).as_posix(), support=support)

    def hooks(self) -> None:
        declared_value = self._declared_field("hooks")
        default = PurePosixPath(self.profile.default_hooks_file) if self.profile.default_hooks_file else None
        merge_default = not self.profile.declared_replaces_default
        for source in self._json_sources("hook", "hooks", declared_value, default, merge_default=merge_default):
            self._hook(source)

    def _suffixes(self, component_type: str) -> tuple[str, ...]:
        if component_type == "agent":
            return self.profile.agent_suffixes
        if component_type == "command":
            return self.profile.command_suffixes
        return (".md",)

    def markdown_components(self, component_type: str, field_name: str, default_dir: str | None) -> None:
        declared_value = self._declared_field(field_name)
        suffixes = self._suffixes(component_type)
        # Codex loads the default folder when it drops the declared value.
        codex_default = self.profile.codex_path_rules and not self._declared_replaces(field_name, declared_value)
        if declared_value is None or codex_default:
            if default_dir is not None and self.reader.kind(PurePosixPath(default_dir)) == "dir":
                self._markdown_dir(component_type, PurePosixPath(default_dir), "packaged")
            elif default_dir is not None and self.reader.kind(PurePosixPath(default_dir)) == "link":
                self._list_dir(component_type, PurePosixPath(default_dir), suffixes)
            if declared_value is None:
                return
        if component_type == "command" and isinstance(declared_value, dict):
            self._command_map(declared_value)
            return
        for raw in self._declared_values(field_name, declared_value):
            resolved = self._resolve_component(component_type, field_name, raw)
            if resolved is None:
                continue
            rel = resolved.rel
            origin: Origin = (
                "declared+packaged"
                if default_dir is not None and (rel.parts[:1] == (default_dir,) or rel == PurePosixPath(default_dir))
                else "declared"
            )
            if resolved.kind == "dir":
                self._markdown_dir(component_type, rel, origin)
            else:
                self._markdown_file(component_type, rel, origin)

    def _markdown_dir(self, component_type: str, rel_dir: PurePosixPath, origin: Origin) -> None:
        """Every agent, command, or output style a client loads from a folder, at any depth.

        Nothing is pruned: Claude Code also loads ``agents/evals/x.md``, so a
        file in a folder the whole-tree scans skip is inventoried, checked for
        privileges, and reported as unscanned (HIGH).
        """
        for rel in self._list_dir(component_type, rel_dir, self._suffixes(component_type)) or []:
            if (unscanned := _unscanned_packaged_finding(self.reader, component_type, rel, rel_dir)) is not None:
                self.inventory.findings.append(unscanned)
            self._markdown_file(component_type, rel, origin)

    def _command_map(self, commands: dict[str, Any]) -> None:
        """The object form of ``commands``: each entry has a ``source`` file or folder, or inline ``content`` text."""
        self._check_item_count("'commands' map", len(commands))
        for index, (command_name, entry) in enumerate(commands.items()):
            if index >= PLUGIN_COMPONENT_MAX_ITEMS:
                break
            if (
                not isinstance(entry, dict)
                or ("source" in entry) == ("content" in entry)
                or not isinstance(entry.get("content", ""), str)
            ):
                self.inventory.findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "plugin_component_path_invalid",
                        f"commands[{command_name!r}] must set exactly one of 'source' or 'content' (text)",
                        self.manifest_display,
                        "Give each command map entry either a 'source' file path or inline 'content' text.",
                    )
                )
                self._add(
                    PluginMember("command", str(command_name), "declared", None, "unsupported", problem="invalid")
                )
            elif "content" in entry:
                self._inline_command(command_name, entry)
            else:
                self._command_source(command_name, entry)

    def _command_source(self, command_name: str, entry: dict[str, Any]) -> None:
        """A ``commands`` map entry's ``source``: a command file, or a folder of them (:meth:`_command_map_folder`)."""
        source = entry.get("source")
        resolved = self._resolve_component(
            "command", f"commands[{command_name!r}].source", source, name=str(command_name)
        )
        if resolved is None:
            return
        if resolved.kind == "dir":
            self._command_map_folder(str(command_name), DeclaredPath(str(source), resolved.rel), entry)
            return
        description = entry.get("description") if isinstance(entry.get("description"), str) else ""
        component = self._add(
            PluginMember("command", str(command_name), "declared", resolved.rel.as_posix(), "unsupported")
        )
        self._command_text(component, resolved.rel, description, entry)

    def _command_map_folder(self, command_name: str, declared: DeclaredPath, entry: dict[str, Any]) -> None:
        """A ``commands`` map entry whose source is a folder: one command per Markdown file directly in it.

        Claude Code accepts "a command file or skill directory" as the source.
        For a folder it loads each ``.md`` file directly inside as a command
        named after the file (``cmds/a.md`` is ``a``, whatever the map key; a
        ``SKILL.md`` is ``SKILL``) and does not search subfolders. Each row
        names its file, so Tier 3 stages and checks the file itself. A folder
        without Markdown files loads no command: the entry is an invalid row.
        """
        rel_dir = declared.rel
        if rel_dir is None:
            return
        files = self._list_dir("command", rel_dir, self._suffixes("command"))
        if files is None:
            return  # an unsafe entry: reported, with a broken row
        direct = [rel for rel in files if rel.parent == rel_dir]
        if not direct:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "plugin_command_folder_empty",
                    f"commands[{command_name!r}].source {declared.raw!r} is a folder with no Markdown (.md) file "
                    "directly in it, so Claude Code loads no command from it",
                    self.manifest_display,
                    "Point the source at the command's .md file, or put the command files directly in the folder.",
                    metadata={"plugin_component": {"type": "command", "name": command_name}},
                )
            )
            self._broken("command", declared, command_name, "invalid")
            return
        self._check_item_count("Command folder members", len(direct), rel_dir.as_posix())
        for rel in direct[:PLUGIN_COMPONENT_MAX_ITEMS]:
            component = self._add(PluginMember("command", rel.stem, "declared", rel.as_posix(), "unsupported"))
            self._command_text(component, rel, "", entry)

    def apps(self) -> None:
        """Inventory Codex app (connector) declarations: ``apps`` or the default ``.app.json``.

        An app is a hosted connector that SkillEvaluator cannot stage, so every
        declared alias is inventoried as ``unsupported``.
        """
        declared_value = self._declared_field("apps")
        default_name = self.profile.default_apps_file
        if default_name is None and declared_value is None:
            return
        default = PurePosixPath(default_name) if default_name else None
        self._add_apps(self._json_sources("app", "apps", declared_value, default, merge_default=False))

    def _add_apps(self, sources: Iterable[_JsonSource]) -> None:
        """One ``app`` component per alias in each ``.app.json`` ``apps`` map; a file without aliases is one."""
        for name, origin, rel, config in sources:
            apps = config.get("apps") if isinstance(config, dict) else None
            if not isinstance(apps, dict) or not apps:
                self._add(PluginMember("app", name, origin, rel, "unsupported", config=config))
                continue
            self._check_item_count("App aliases", len(apps), rel)
            for alias in list(apps)[:PLUGIN_COMPONENT_MAX_ITEMS]:
                self._add(PluginMember("app", str(alias), origin, rel, "unsupported", config=apps[alias]))

    def extensions(self) -> None:
        """Inventory Agent Plugins client-extension namespaces.

        A namespace is declared in the manifest ``extensions`` object and/or
        ships a top-level reverse-domain directory (spec section 8). Extensions
        have no portable semantics, so each namespace is ``unsupported``. Inside
        a namespace directory, the documented client layouts (``agents/``,
        ``commands/``, ``rules/``, and ``hooks/hooks.json``) are inventoried as
        their own types, all ``unsupported``. Policy checks remain separate.
        """
        if not self.profile.client_extensions:
            return
        declared = self.manifest.get("extensions") if self.manifest is not None else None
        namespaces: dict[str, Origin] = {}
        if isinstance(declared, dict):
            self._check_item_count("Extension namespaces", len(declared))
            for key in list(declared)[:PLUGIN_COMPONENT_MAX_ITEMS]:
                namespaces[str(key)] = "declared"
        try:
            with os.scandir(self.reader.root) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            entries = []
        directories: set[str] = set()
        self._check_item_count("Root directory entries", len(entries), ".", limit=CONTENT_DEDUP_MAX_DISCOVERED_PATHS)
        for entry in entries[:CONTENT_DEDUP_MAX_DISCOVERED_PATHS]:
            if not _NAMESPACE_DIR_RE.match(entry.name) or self.reader.kind(PurePosixPath(entry.name)) != "dir":
                continue
            directories.add(entry.name)
            namespaces[entry.name] = "declared+packaged" if entry.name in namespaces else "packaged"
        for namespace, origin in namespaces.items():
            path = namespace if namespace in directories else self.manifest_rel
            self._add(
                PluginMember(
                    "extension",
                    namespace,
                    origin,
                    path,
                    "unsupported",
                    config=declared.get(namespace) if isinstance(declared, dict) else None,
                )
            )
            if namespace in directories:
                self._namespace_components(PurePosixPath(namespace))

    def _namespace_components(self, base: PurePosixPath) -> None:
        """The documented client layouts in a namespace folder, inventoried as their own (unsupported) types."""
        for component_type, folder in (("agent", "agents"), ("command", "commands")):
            if self.reader.kind(base / folder) == "dir":
                self._markdown_dir(component_type, base / folder, "packaged")
        if self.reader.kind(base / "rules") == "dir":
            self._rule_dir(base / "rules", "packaged", support="unsupported")
        hooks_file = base / "hooks" / "hooks.json"
        if self.reader.kind(hooks_file) == "file":
            config = self._load_json(hooks_file, "hooks")
            self._hook(_JsonSource(hooks_file.as_posix(), "packaged", hooks_file.as_posix(), config))

    def openai_extension(self) -> None:
        """Hooks and apps in ``extensions["com.openai"]`` of an Agent Plugins manifest.

        OpenAI documents this object as the place for Codex hooks (a path, an
        array of paths, an inline hooks object, or an array of them) and app
        mappings (a path to an ``.app.json``), read with the Codex path rules.
        Their raw configuration is retained for the separate hook policy engine.
        """
        if not self.profile.client_extensions or self.manifest is None:
            return
        extensions = self.manifest.get("extensions")
        openai = extensions.get("com.openai") if isinstance(extensions, dict) else None
        if not isinstance(openai, dict):
            return
        # Codex path rules, messages, and hook dialect for these fields.
        if openai.get("hooks") is not None:
            for source in self._json_sources("hook", "hooks", openai.get("hooks"), None, profile=CODEX_PROFILE):
                if source.rel == self.manifest_rel:  # an inline hooks object
                    source = source._replace(name=f"extensions.com.openai.hooks:{source.name}")
                self._hook(source, profile=CODEX_PROFILE)
        if openai.get("apps") is not None:
            self._add_apps(self._json_sources("app", "apps", openai.get("apps"), None, profile=CODEX_PROFILE))

    def _read_member(
        self, component: PluginMember, rel: PurePosixPath, *, lenient: bool = False, strip: bool = False
    ) -> None:
        text = self._read_lenient(rel) if lenient else self._read(rel)
        if text is not None and strip:
            text = text.strip()
        component.text = text
        if text is None:
            component.problem = "unreadable"
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_unreadable",
                    f"Member '{rel}' could not be read safely.",
                    self.reader.display(rel),
                    "Use a regular bounded file.",
                )
            )
            return
        parsed = parse_markdown(text)
        component.frontmatter = parsed.frontmatter
        component.frontmatter_error = parsed.frontmatter_error
        if component.type in {"skill", "command"} and isinstance(parsed.frontmatter.get("hooks"), dict):
            source = f"{rel.as_posix()}#hooks"
            self._add(
                PluginMember("hook", source, "packaged", source, "unsupported", config=parsed.frontmatter["hooks"])
            )
        if parsed.frontmatter_error:
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_unreadable",
                    parsed.frontmatter_error,
                    self.reader.display(rel),
                    "Keep frontmatter within the parser limits.",
                )
            )

    def _rule_file(
        self, rel: PurePosixPath, origin: Origin, *, name: str | None = None, support: Support = "evaluated"
    ) -> None:
        self._read_member(
            self._add(PluginMember("rule", name or rel.name, origin, rel.as_posix(), support)), rel, strip=True
        )

    def _markdown_file(self, component_type: str, rel: PurePosixPath, origin: Origin) -> None:
        component = PluginMember(component_type, rel.stem, origin, rel.as_posix(), _TYPE_SUPPORT[component_type])
        self._read_member(component, rel, lenient=True)
        if isinstance(component.frontmatter.get("name"), str) and component.frontmatter["name"]:
            component.name = component.frontmatter["name"]
        self._add(component)

    def _inline_command(self, command_name: str, entry: dict[str, Any]) -> None:
        self._add(
            PluginMember(
                "command",
                str(command_name),
                "declared",
                None,
                "unsupported",
                text=entry.get("content", ""),
                config=entry,
            )
        )

    def _command_text(
        self, component: PluginMember, rel: PurePosixPath, _description: str, entry: dict[str, Any]
    ) -> None:
        self._read_member(component, rel)
        component.config = entry

    def mcp(self) -> None:
        collection = collect_mcp_declarations(
            self.reader,
            self.manifest,
            contained=self.contained,
            manifest_rel=self.manifest_rel,
            profile=self.profile,
            validate_servers=False,
        )
        self.inventory.findings.extend(collection.findings)
        effective = {id(item) for item in collection.effective}
        for declaration in collection.declarations:
            origin: Origin = "packaged" if declaration.source == "mcp_json" else "declared"
            self._add(
                PluginMember(
                    "mcp",
                    declaration.name,
                    origin,
                    declaration.file,
                    "evaluated" if declaration.runnable else "static_only",
                    config=declaration.config,
                    mcp=declaration,
                    effective=id(declaration) in effective,
                )
            )
        for raw, path, problem in collection.broken_sources:
            self._add(PluginMember("mcp", raw, "declared", path, "unsupported", problem=problem))
        for raw, path in collection.bundles:
            self._add(PluginMember("mcp", raw, "declared", path, "unsupported", bundle=True))

    def _hook(self, source: _JsonSource, *, profile: FormatProfile | None = None) -> None:
        name, origin, rel, config = source
        profile = profile or self.profile
        problem = None
        if config is not None and (
            not isinstance(config, dict) or ("hooks" in config and not isinstance(config["hooks"], dict))
        ):
            problem = "invalid"
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_unreadable",
                    f"Hooks config '{name}' must map events to handlers.",
                    self.reader.display(rel),
                    "Write a hooks object.",
                )
            )
        self._add(
            PluginMember(
                "hook",
                name,
                origin,
                rel,
                "unsupported",
                problem=problem,
                config=config,
                manifest_type=profile.manifest_type,
            )
        )

    def lsp(self) -> None:
        declared = self._declared_field("lspServers")
        default = PurePosixPath(self.profile.default_lsp_file) if self.profile.default_lsp_file else None
        if declared is None and default is None:
            return
        applied: dict[str, PluginMember] = {}
        for name, origin, rel, config in self._json_sources("lsp", "lspServers", declared, default):
            servers = config.get("lspServers", config) if isinstance(config, dict) else None
            if isinstance(servers, dict) and servers:
                self._check_item_count(f"LSP servers in {rel}", len(servers), rel)
                for server_name, server in list(servers.items())[:PLUGIN_COMPONENT_MAX_ITEMS]:
                    name = str(server_name)
                    member = PluginMember("lsp", name, origin, rel, "unsupported", config=server)
                    previous = applied.get(name)
                    if isinstance(server, dict):
                        if previous is not None:
                            previous.effective = False
                        if previous is not None and isinstance(previous.config, dict):
                            self.inventory.findings.append(
                                _plugin_finding(
                                    Severity.MEDIUM,
                                    "plugin_lsp_duplicate_name",
                                    f"LSP server '{name}' is declared more than once; Claude Code keeps only the later declaration.",
                                    self.reader.display(rel),
                                    "Declare each LSP server name once.",
                                    metadata={"plugin_component": {"type": "lsp", "name": name}},
                                )
                            )
                        applied[name] = member
                    elif previous is not None:
                        member.effective = False
                    else:
                        applied[name] = member
                    self._add(member)
            else:
                self._add(PluginMember("lsp", name, origin, rel, "unsupported", config=config))

    def monitors(self) -> None:
        if self.profile.default_monitors_file is None:
            return
        experimental = self.manifest.get("experimental") if self.manifest else None
        value = (
            experimental.get("monitors")
            if isinstance(experimental, dict) and "monitors" in experimental
            else self._declared_field("monitors")
        )
        default = PurePosixPath(self.profile.default_monitors_file)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            sources = [_JsonSource("inline", "declared", self.manifest_rel, value)]
        else:
            sources = self._json_sources(
                "monitor", "experimental.monitors", value, default, merge_default=value is None
            )
        for name, origin, rel, config in sources:
            entries = config.get("monitors", config) if isinstance(config, dict) else config
            if not isinstance(entries, list) or not entries:
                self._add(PluginMember("monitor", name, origin, rel, "unsupported", config=config))
                continue
            self._check_item_count(f"Monitors in {rel}", len(entries), rel)
            for index, entry in enumerate(entries[:PLUGIN_COMPONENT_MAX_ITEMS]):
                label = (
                    entry.get("name")
                    if isinstance(entry, dict) and isinstance(entry.get("name"), str)
                    else f"{name}[{index}]"
                )
                self._add(PluginMember("monitor", label, origin, rel, "unsupported", config=entry))

    def settings(self) -> None:
        for path in self.profile.settings_files:
            rel = PurePosixPath(path)
            kind = self.reader.kind(rel)
            if kind in {"missing", "dir"}:
                continue
            problem = None if kind == "file" else "unsafe"
            config = self._load_json(rel, "settings") if problem is None else None
            if problem:
                self.inventory.findings.append(
                    _path_problem_finding(
                        self.reader, "settings", DeclaredPath(path, rel), self.manifest_rel, "unsafe", rel
                    )
                )
            self._add(PluginMember("settings", path, "packaged", path, "unsupported", problem=problem, config=config))
        if self.contained and self.manifest and "settings" in self.manifest:
            self._add(
                PluginMember(
                    "settings",
                    "plugin.json#settings",
                    "declared",
                    self.manifest_rel,
                    "unsupported",
                    config=self.manifest["settings"],
                )
            )
        rel = PurePosixPath(".codex/config.toml")
        kind = self.reader.kind(rel)
        if kind in {"missing", "dir"}:
            return
        import tomllib

        try:
            config = tomllib.loads(self.reader.read_text(rel, PLUGIN_CONFIG_MAX_BYTES, config=True))
            problem = None
        except (SecurePathError, OSError, ValueError, RecursionError) as exc:
            config, problem = None, "unreadable"
            self.inventory.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_component_unreadable",
                    f"Codex config could not be read: {report_text(str(exc), 160)}",
                    self.reader.display(rel),
                    "Use a regular bounded TOML file.",
                )
            )
        self._add(
            PluginMember(
                "settings", rel.as_posix(), "packaged", rel.as_posix(), "unsupported", problem=problem, config=config
            )
        )


def discover_plugin_members(path: Path, *, include_additional: bool = True) -> PluginDiscovery | None:
    """Locate and read actual plugin declarations without activating public validators.

    Each supported co-located manifest gets its own profile and provenance.
    Parse failures stay explicit findings; declarations from another view are
    never mistaken for confirmed loads. Reads share one per-operation budget.
    """
    try:
        located = locate_plugin_manifest(path)
    except PluginManifestPathError as exc:
        return PluginDiscovery(
            findings=[
                _plugin_finding(
                    Severity.HIGH, "plugin_component_path_unsafe", str(exc), str(path), "Use regular contained files."
                )
            ]
        )
    if located is None:
        return None
    manifests = (located, *located.additional) if include_additional else (located,)
    result = PluginDiscovery(manifests=manifests)
    environment_files, result.environment_scan_complete = find_environment_files(located.root)
    result.environment_files = tuple(environment_files)
    reader = PluginRootReader(located.root)
    for manifest in manifests:
        try:
            parsed = manifest.parse_for_audit()
        except PluginManifestPathError as exc:
            result.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_manifest_unreadable",
                    str(exc),
                    str(manifest.declared_path),
                    "Keep the manifest a stable regular file.",
                )
            )
            continue
        if parsed is None:
            result.findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_manifest_unreadable",
                    f"Manifest '{manifest.manifest_filename}' could not be parsed safely.",
                    str(manifest.declared_path),
                    "Provide a bounded valid manifest.",
                )
            )
            continue
        collector = _MemberCollector(
            reader,
            normalized_component_manifest(manifest.manifest_type, parsed),
            contained=manifest.manifest_type != "agent_bundle_yaml",
            manifest_rel=manifest.manifest_filename,
            profile=profile_for(manifest.manifest_type),
        )
        discovered = collector.collect()
        result.components.extend(discovered.components)
        result.findings.extend(discovered.findings)
        result.unread_files += discovered.unread_files
    return result
