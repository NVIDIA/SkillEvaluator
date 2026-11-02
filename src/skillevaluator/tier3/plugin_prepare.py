# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a private plugin inventory for a runtime evaluation.

Preparation is passive: it reads only the plugin root through the inventory's
anchored reader, never fetches dependencies, installs packages, or probes an
endpoint. Runnable configuration comes from the selected manifest's effective
rows only; rows of overridden or additional client views stay audit records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    PLUGIN_CURSOR_MANIFEST_TYPE,
)
from skillevaluator.plugin_config_risk import _nested_hook_groups
from skillevaluator.plugin_formats import profile_for
from skillevaluator.plugin_markdown import parse_markdown
from skillevaluator.plugin_mcp import _server_launch_files
from skillevaluator.plugin_paths import PluginRootReader
from skillevaluator.publication_identity import publication_source_digest
from skillevaluator.tier3 import plugin_native as native
from skillevaluator.utils.path_security import canonicalize_trusted_root_alias
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot

if TYPE_CHECKING:
    from skillevaluator.plugin_discovery import PluginMember
    from skillevaluator.plugin_inventory import PluginInventory

_FRONTMATTER_HOOKS = "#hooks"
_TEXT_TYPES = frozenset({"agent", "command", "output_style"})
_RUNTIME_TYPES = frozenset({"skill", "rule", "mcp"})


class PreparationError(ValueError):
    """The inventory cannot be prepared for a runtime evaluation."""


@dataclass(frozen=True)
class SourceIdentity:
    """The analyzed source a prepared package was built from."""

    root: str
    manifest: str
    manifest_sha256: str
    source_tree_sha256: str

    @classmethod
    def of(cls, inventory: PluginInventory) -> SourceIdentity:
        identity = _source_identity(inventory.root, inventory.selected_manifest.manifest_filename)
        if identity.manifest_sha256 != inventory.manifest_sha256:
            raise PreparationError("plugin source changed after analysis: manifest contents differ")
        return identity


def _source_identity(root: Path, manifest: str) -> SourceIdentity:
    # Normalize only trusted OS aliases; authored links must still be refused.
    root = canonicalize_trusted_root_alias(Path(os.path.abspath(root)))  # noqa: PTH100
    # The default results root creates <plugin>/evals/ during the run; a
    # directory with no authored content does not change the source identity.
    digest = publication_source_digest(root, ignore_empty_directories=True)
    if digest is None:
        raise PreparationError("Refusing linked, unsafe, or unstable plugin source tree")
    try:
        with SecureRoot(root) as reader:
            raw, _ = reader.read_bytes(Path(manifest), CONTENT_DEDUP_MAX_TOTAL_BYTES)
    except (SecurePathError, OSError) as exc:
        raise PreparationError(f"cannot identify plugin manifest '{manifest}': {exc}") from exc
    return SourceIdentity(str(root), manifest, hashlib.sha256(raw).hexdigest(), digest)


@dataclass(frozen=True)
class AuditRow:
    """An inventory row that is not runnable configuration, with the reason it is kept."""

    type: str
    name: str
    path: str | None
    reason: str
    declared_by: str = ""


@dataclass(frozen=True)
class PreparedRuntimePackage:
    """One plugin inventory prepared for the runtime arms of an evaluation."""

    identity: SourceIdentity
    plugin_name: str
    description: str
    skills: tuple[PluginMember, ...]
    rules: tuple[PluginMember, ...]
    mcp_servers: tuple[Mapping[str, Any], ...]
    native_source: native.NativePluginSource
    audit: tuple[AuditRow, ...] = ()
    policy_findings: tuple[Any, ...] = ()
    incomplete_reasons: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.incomplete_reasons

    @property
    def runnable(self) -> bool:
        """Whether there is anything a runtime arm can load."""
        return bool(self.skills or self.rules or self.mcp_servers or self.native_source.texts)


def prepare_runtime_package(
    inventory: PluginInventory,
    *,
    expected: SourceIdentity | None = None,
    blocking_findings: Sequence[Any] = (),
    excluded_paths: Sequence[Path] = (),
) -> PreparedRuntimePackage:
    """Build the prepared package from the selected manifest's effective rows.

    ``expected`` is the identity the caller analyzed; a different root,
    manifest, or authored source-tree digest is refused instead of preparing a changed
    source. ``blocking_findings`` are findings the caller's effective policy
    rates blocking; any of them refuses preparation.
    """
    identity = SourceIdentity.of(inventory)
    if expected is not None and identity != expected:
        raise PreparationError(f"plugin source changed after analysis: expected {expected}, found {identity}")
    if blocking_findings:
        names = ", ".join(sorted({str(getattr(finding, "check_name", finding)) for finding in blocking_findings}))
        raise PreparationError(f"plugin policy blocks runtime evaluation: {names}")
    manifest = inventory.selected_manifest.parse_for_audit()
    if not isinstance(manifest, dict):
        raise PreparationError(f"selected manifest is unreadable: {identity.manifest}")
    selected = inventory.selected_components
    selected_ids = {id(member) for member in selected}
    audit = [_audit_row(member) for member in inventory.components if id(member) not in selected_ids]
    skills: list[PluginMember] = []
    rules: list[PluginMember] = []
    mcp: list[Mapping[str, Any]] = []
    for member in selected:
        if member.problem is not None:
            audit.append(AuditRow(member.type, member.name, member.path, f"unresolved: {member.problem}"))
        elif member.type == "skill" and member.path:
            skills.append(member)
        elif member.type == "rule":
            rules.append(member)
        elif member.type == "mcp" and member.mcp is not None:
            if member.mcp.runnable:
                mcp.append({"name": member.mcp.name, **member.mcp.config})
            else:
                audit.append(
                    AuditRow("mcp", member.name, member.path, f"not runnable: {member.mcp.kind or 'no launch'}")
                )
    plugin_name = str(manifest.get("name") or inventory.root.name)
    incomplete = [] if inventory.complete else ["plugin inventory is incomplete"]
    source = build_native_source(
        inventory, manifest, plugin_name=plugin_name, excluded_paths=excluded_paths, mcp_servers=mcp
    )
    for server in source.plugin_file_mcp_servers:
        audit.append(
            AuditRow(
                "mcp", str(server["name"]), None, "requires plugin tree; unavailable to wrapper/non-tree-copy adapters"
            )
        )
    for kind, name in source.other:
        if kind == "mcp":
            audit.append(AuditRow("mcp", name, None, "launch cwd is unsupported by runtime adapters"))
    package = PreparedRuntimePackage(
        identity=identity,
        plugin_name=plugin_name,
        description=str(manifest.get("description") or ""),
        skills=tuple(skills),
        rules=tuple(rules),
        mcp_servers=tuple(mcp),
        native_source=source,
        audit=tuple(audit),
        policy_findings=tuple(inventory.findings),
        incomplete_reasons=tuple(incomplete),
    )
    verify_source_identity(package)
    return package


def verify_source_identity(package: PreparedRuntimePackage, inventory: PluginInventory | None = None) -> None:
    """Re-read disk identity before/after staging, even when the caller retains an old inventory.

    The versioned publication digest includes authored evaluation inputs and
    excludes only its established generated-output/dependency roots. It is an
    observed-change guard, not an atomic snapshot against concurrent writers.
    """
    root = inventory.root if inventory is not None else package.native_source.plugin_root
    manifest = inventory.selected_manifest.manifest_filename if inventory is not None else package.identity.manifest
    if _source_identity(root, manifest) != package.identity:
        raise PreparationError("plugin source changed between preparation and staging")


def _audit_row(member: PluginMember) -> AuditRow:
    if member.loaded_by:
        reason = f"additional client view ({member.loaded_by})"
    elif not member.effective:
        reason = "overridden by an effective declaration"
    else:
        reason = "declared by an additional manifest"
    return AuditRow(member.type, member.name, member.path, reason, member.declared_by)


def build_native_source(
    inventory: PluginInventory,
    manifest: Mapping[str, Any],
    *,
    plugin_name: str,
    excluded_paths: Sequence[Path] = (),
    mcp_servers: Sequence[Mapping[str, Any]] = (),
) -> native.NativePluginSource:
    """Snapshot what a native adapter can stage from the selected effective rows.

    Text components and hook files are read through the anchored root reader
    (bounded, never following links). A component enabling a permission bypass
    is recorded in ``refusals`` so every adapter that would stage its type
    natively refuses it.
    """
    reader = PluginRootReader(inventory.root)
    manifest_rel = inventory.selected_manifest.manifest_filename
    texts: list[native.NativeTextComponent] = []
    hooks: list[native.NativeHookSource] = []
    other: list[tuple[str, str]] = []
    refusals: list[tuple[str, str, str]] = []
    settings: dict[str, Any] = {}
    lsp_servers: dict[str, Any] = {}
    manifest_type = inventory.selected_manifest.manifest_type
    nest = manifest_type == PLUGIN_CURSOR_MANIFEST_TYPE
    root_prefixes = tuple(profile_for(manifest_type).root_prefixes)
    dialect = "cursor" if nest else "claude"

    def refuse(kind: str, name: str, message: str | None) -> None:
        if message is not None:
            refusals.append((kind, name, message))

    for member in inventory.selected_components:
        if member.problem is not None or member.type in _RUNTIME_TYPES:
            continue
        if member.type in _TEXT_TYPES:
            text = _member_text(reader, manifest, member)
            if text is None:
                continue
            where = f"plugin {member.type} '{member.name}'"
            refuse(
                member.type,
                member.name,
                native._bypass_frontmatter_refusal(parse_markdown(text).frontmatter, where=where),
            )
            texts.append(native.NativeTextComponent(member.type, member.name, member.path, text))
        elif member.type == "hook":
            rel = member.path or ""
            if rel.endswith(_FRONTMATTER_HOOKS):
                config, rel, hook_dialect = member.config, rel.removesuffix(_FRONTMATTER_HOOKS), "claude"
            else:
                config = (
                    native._inline_hook_config(manifest, member.name)
                    if rel == manifest_rel
                    else (native._read_plugin_json(reader, rel, "hooks") if rel else None)
                )
                config, hook_dialect = (_nested_hook_groups(config) if nest else config), dialect
            if not isinstance(config, dict):
                continue
            refuse("hook", member.name, native._bypass_refusal(config, where=f"plugin hooks '{member.name}'"))
            hooks.append(native.NativeHookSource(member.name, rel, config, hook_dialect, root_prefixes))
        else:
            other.append((member.type, member.name))
            if member.type == "settings" and member.path == native._PLUGIN_SETTINGS_FILE:
                applied = native._plugin_settings(native._read_plugin_json(reader, member.path, "settings"))
                refuse("settings", member.name, native._bypass_refusal(applied, where="plugin settings.json"))
                settings.update(applied)
            elif member.type == "lsp" and member.path:
                config = (
                    manifest.get("lspServers")
                    if member.path == manifest_rel
                    else native._read_plugin_json(reader, member.path, "LSP servers")
                )
                server = native._lsp_server_config(config, member.name)
                if server is not None:
                    refuse(
                        "lsp", member.name, native._bypass_refusal(server, where=f"plugin LSP server '{member.name}'")
                    )
                    lsp_servers[member.name] = server
    if reader.kind(PurePosixPath("bin")) == "dir":
        other.append(("bin", "bin"))
    portable_mcp, plugin_file_mcp = [], []
    pattern = native.foreign_root_var_re(root_prefixes)
    plugin_file = native._plugin_file_checker(inventory.root)
    for config in mcp_servers:
        server = dict(config)
        if server.get("cwd"):
            # No current native config projection preserves cwd. Keep explicit
            # evidence rather than launch in an unrelated task directory.
            other.append(("mcp", str(server["name"])))
            continue
        paths = _server_launch_files(
            server, relative_cwd=profile_for(manifest_type).mcp_dialect == "codex", root=inventory.root
        )
        launch_values = [server.get("command"), *(server.get("args") or [])]
        has_relative_file = any(_plugin_launch_path(value, plugin_file=plugin_file) != value for value in launch_values)
        if paths or has_relative_file:
            for key in ("command", "args"):
                value = server.get(key)
                if isinstance(value, list):
                    server[key] = [
                        _plugin_launch_path(native.to_claude_root(arg, pattern), plugin_file=plugin_file)
                        for arg in value
                    ]
                elif isinstance(value, str):
                    server[key] = _plugin_launch_path(native.to_claude_root(value, pattern), plugin_file=plugin_file)
            plugin_file_mcp.append(server)
        else:
            portable_mcp.append(server)
    return native.NativePluginSource(
        plugin_name=plugin_name,
        description=str(manifest.get("description") or ""),
        contained=True,
        plugin_root=inventory.root,
        manifest=dict(manifest),
        manifest_rel=manifest_rel,
        member_skills=tuple(inventory.root / member.path for member in inventory.selected_skills if member.path),
        rules=tuple(
            (member.name, member.text or "")
            for member in inventory.selected_components
            if member.type == "rule" and not member.problem
        ),
        mcp_servers=tuple(portable_mcp),
        plugin_file_mcp_servers=tuple(plugin_file_mcp),
        mcp_declared={str(server["name"]): _claude_declared_mcp(server, pattern) for server in mcp_servers},
        texts=tuple(texts),
        hooks=tuple(hooks),
        other=tuple(other),
        excluded_paths=tuple(excluded_paths),
        refusals=tuple(refusals),
        settings=settings,
        lsp_servers=lsp_servers,
    )


def _claude_declared_mcp(server: Mapping[str, Any], pattern: re.Pattern[str] | None) -> dict[str, Any]:
    """The declared server with its env/header values in Claude Code's plugin-root spelling."""
    declared = dict(server)
    for key in ("env", "headers"):
        values = declared.get(key)
        if isinstance(values, Mapping):
            declared[key] = {name: native.to_claude_root(value, pattern) for name, value in values.items()}
    return declared


def _plugin_launch_path(value: Any, *, plugin_file: Callable[[str], bool] | None) -> Any:
    """Anchor explicit ./ launch paths when Claude copies the plugin tree."""
    if not isinstance(value, str):
        return value
    prefix, sep, path = value.partition("=") if value.startswith("-") else ("", "", value)
    if path.startswith("./"):
        return prefix + sep + native.CLAUDE_PLUGIN_ROOT_VAR + "/" + path[2:]
    if not path.startswith(("-", "$", "/", "~")) and plugin_file is not None and plugin_file(path):
        return prefix + sep + native.CLAUDE_PLUGIN_ROOT_VAR + "/" + path
    return value


def _member_text(reader: PluginRootReader, manifest: Mapping[str, Any], member: PluginMember) -> str | None:
    if member.text is not None:
        return member.text
    if member.path is None:
        return native._inline_command_content(manifest, member.name) if member.type == "command" else None
    try:
        return reader.read_text(PurePosixPath(member.path), CONTENT_DEDUP_MAX_FILE_BYTES)
    except (SecurePathError, OSError) as exc:
        raise PreparationError(f"Refusing unsafe or unreadable plugin {member.type} '{member.path}': {exc}") from exc


#: Upper bound on combined evaluation cases (bounded work before any agent runs).
MAX_DATASET_CASES = 1000
_EVALUATOR_DIRS = frozenset({"evals", "eval"})
_MEMBER_LAYOUTS = (("evals", "evals"), ("eval", "dataset"))


def is_evaluator_dir(path: Path) -> bool:
    """Whether *path* is a skill's verifier-only dataset directory.

    ``evals/`` always is. A legacy ``eval/`` is only when it holds the
    ``dataset.*`` this module reads, so runtime helpers that a skill keeps in
    ``eval/`` still reach the agent.
    """
    from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS

    name = path.name.casefold()
    if name == "evals":
        return True
    return name == "eval" and any(os.path.lexists(path / f"dataset{ext}") for ext in DATASET_EXTENSIONS)


@dataclass(frozen=True)
class WrapperSkill:
    """The generated skill the existing engine evaluates for a prepared plugin."""

    name: str
    path: Path
    #: Case id -> the plugin-root-relative dataset file it came from (provenance, not the wrapper path).
    case_sources: Mapping[str, str] = field(default_factory=dict)


def wrapper_skill_name(plugin_name: str) -> str:
    return f"plugin-{native.plugin_slug(plugin_name)}"


def stage_wrapper_skill(
    package: PreparedRuntimePackage,
    destination: Path,
    *,
    evals_source: Path | None = None,
    max_cases: int = MAX_DATASET_CASES,
) -> WrapperSkill:
    """Write the wrapper skill and its combined dataset at ``destination`` (all or nothing).

    Member skills are copied without their evaluation directories, so datasets,
    answers, and fixtures are visible to the verifier only. An explicit
    ``evals_source`` replaces the member datasets. Case ids are deterministic
    and unique; fixtures that collide across members are refused.
    """
    import shutil
    import tempfile

    from skillevaluator.tier3.harbor.secure_copy import UnsafeStagingError, copytree_secure

    if os.path.lexists(destination):
        raise PreparationError(f"wrapper destination already exists: {destination}")
    verify_source_identity(package)
    name = wrapper_skill_name(package.plugin_name)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        entries, sources, fixtures = _collect_dataset(package, evals_source, max_cases=max_cases)
        if not entries:
            raise PreparationError("plugin has no evaluation dataset: pass an eval source or add member evals")
        if len(entries) > max_cases:
            raise PreparationError(f"plugin dataset has {len(entries)} cases, over the {max_cases}-case limit")
        evals_dir = temp / "evals"
        evals_dir.mkdir()
        (evals_dir / "evals.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
        for rel, source in fixtures.items():
            target = evals_dir / "files" / rel
            if source is None:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source)
        for member in package.skills:
            member_dir = package.native_source.plugin_root / str(member.path)
            try:
                copytree_secure(member_dir, temp / "skills" / member_dir.name, ignore=_evaluator_ignore(member_dir))
            except (UnsafeStagingError, OSError) as exc:
                raise PreparationError(f"Refusing to stage member skill '{member.name}': {exc}") from exc
        (temp / "SKILL.md").write_text(_wrapper_markdown(name, package), encoding="utf-8")
        verify_source_identity(package)
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return WrapperSkill(name, destination, sources)


def _evaluator_ignore(member_dir: Path):
    member_dir = member_dir.resolve()

    def ignore(directory: str, contents: list[str]) -> list[str]:
        if Path(directory).resolve() != member_dir:
            return []
        return [entry for entry in contents if is_evaluator_dir(Path(directory) / entry)]

    return ignore


def _wrapper_markdown(name: str, package: PreparedRuntimePackage) -> str:
    description = package.description or f"Use the {package.plugin_name} plugin."
    lines = [
        "---",
        f"name: {name}",
        f"description: {json.dumps(description)}",
        "---",
        f"# {package.plugin_name}",
        "",
    ]
    for member in package.skills:
        lines.append(f"- `skills/{Path(str(member.path)).name}/SKILL.md`: {member.name}")
    for rule_name, text in package.native_source.rules:
        lines.extend(["", f"## Rule: {rule_name}", "", text.strip()])
    return "\n".join(lines).rstrip() + "\n"


def _collect_dataset(
    package: PreparedRuntimePackage, evals_source: Path | None, *, max_cases: int
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, bytes | None]]:
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    fixtures: dict[str, bytes | None] = {}
    owners: dict[str, str] = {}
    root = package.native_source.plugin_root

    def collect(dataset: Path, skill: str | None) -> None:
        source_entries = _read_dataset(dataset)
        if len(entries) + len(source_entries) > max_cases:
            raise PreparationError(f"plugin dataset exceeds the {max_cases}-case limit")
        for index, entry in enumerate(source_entries, start=1):
            base_id = str(entry.get("id") or f"case-{index:03d}")
            case_id = _unique_id(_safe_id(f"{skill}-{base_id}" if skill else base_id), seen)
            combined = {**entry, "id": case_id, "plugin_eval_target": package.plugin_name}
            if skill is not None:
                combined["plugin_eval_source_skill"] = skill
                combined.setdefault("expected_skill", skill)
            # Resolve against the original dataset before moving it. In
            # particular, absent files means this member's defaults, never
            # defaults belonging to another member in the combined dataset.
            combined["files"] = _collect_case_inputs(dataset, entry, case_id, fixtures, owners)
            entries.append(combined)
            sources[case_id] = dataset.relative_to(root).as_posix() if skill else str(dataset)

    if evals_source is not None:
        dataset = _dataset_file(evals_source, (("", "evals"),)) if evals_source.is_dir() else evals_source
        if dataset is None:
            raise PreparationError(f"no evaluation dataset in {evals_source}")
        collect(dataset, None)
    else:
        for member in package.skills:
            member_dir = root / str(member.path)
            dataset = _dataset_file(member_dir, _MEMBER_LAYOUTS)
            if dataset is not None:
                collect(dataset, member_dir.name)
    return entries, sources, fixtures


def _collect_case_inputs(
    dataset: Path,
    entry: dict[str, Any],
    owner: str,
    fixtures: dict[str, bytes | None],
    owners: dict[str, str],
) -> list[str]:
    """Reuse the engine's path and control-file rules, preserving input-relative names."""
    import tempfile

    from skillevaluator.tier3.harbor.adapter import _entry_file_refs, _resolve_entry_file_ref, _stage_task_inputs
    from skillevaluator.tier3.harbor.secure_copy import UnsafeStagingError

    selected: dict[str, bytes | None] = {}
    try:
        # Keep the preparation byte bound before the engine copies inputs.
        # Like secure_copy, this assumes the source remains quiescent while
        # staging; the package identity is rechecked before publication.
        input_files_dir = dataset.parent / "files"
        if "files" not in entry:
            _read_fixtures(input_files_dir, owner, {}, {})
        for ref in _entry_file_refs(entry):
            source, _ = _resolve_entry_file_ref(
                ref, skill_path=dataset.parent.parent, evals_dir=dataset.parent, input_files_dir=input_files_dir
            )
            if source.is_dir():
                _read_fixtures(source, owner, {}, {})
            else:
                _read_regular(source)
        with tempfile.TemporaryDirectory(prefix="skilleval-plugin-input-") as temporary:
            staging = Path(temporary)
            _stage_task_inputs(
                staging,
                input_files_dir=dataset.parent / "files",
                entry=entry,
                source_skill_path=dataset.parent.parent,
                evals_dir=dataset.parent,
            )
            _read_fixtures(staging / "input", owner, selected, {})
    except (OSError, ValueError, UnsafeStagingError) as exc:
        raise PreparationError(f"cannot prepare input fixtures for '{owner}': {exc}") from exc
    for rel, data in selected.items():
        # Empty directory selections must not acquire another case's files;
        # file/directory prefix collisions are likewise never silently merged.
        conflict = next(
            (
                existing
                for existing, content in fixtures.items()
                if (existing == rel and content != data)
                or (existing != rel and (existing.startswith(rel + "/") or rel.startswith(existing + "/")))
            ),
            None,
        )
        if conflict is not None:
            raise PreparationError(
                f"fixture collision on 'files/{rel}': '{owners[conflict]}' and '{owner}' provide different content"
            )
        fixtures[rel], owners[rel] = data, owners.get(rel, owner)
    return [f"files/{rel}" for rel in selected]


def _dataset_file(directory: Path, layouts: Sequence[tuple[str, str]]) -> Path | None:
    from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS

    for folder, stem in layouts:
        for extension in DATASET_EXTENSIONS:
            candidate = directory / folder / f"{stem}{extension}"
            if os.path.lexists(candidate):
                return candidate
    return None


def _read_regular(path: Path) -> bytes:
    try:
        root = canonicalize_trusted_root_alias(Path(os.path.abspath(path.parent)))  # noqa: PTH100
        with SecureRoot(root) as reader:
            data, _ = reader.read_bytes(Path(path.name), CONTENT_DEDUP_MAX_FILE_BYTES)
        return data
    except (SecurePathError, OSError) as exc:
        raise PreparationError(f"Refusing linked, unsafe, or unreadable evaluation file '{path}': {exc}") from exc


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    from skillevaluator.tier3.dataset_utils import normalize_dataset_entries
    from skillevaluator.utils.structured_data import StructuredDataError, load_bounded_json, load_bounded_yaml

    text = _read_regular(path).decode("utf-8", errors="strict")
    suffix = path.suffix.casefold()
    try:
        if suffix == ".jsonl":
            data: Any = [load_bounded_json(line) for raw in text.splitlines() if (line := raw.strip())]
        elif suffix == ".json":
            data = load_bounded_json(text)
        else:
            data = load_bounded_yaml(text)
        return normalize_dataset_entries(data)
    except (StructuredDataError, ValueError, TypeError) as exc:
        raise PreparationError(f"invalid evaluation dataset {path}: {exc}") from exc


def _read_fixtures(files_dir: Path, owner: str, fixtures: dict[str, bytes | None], owners: dict[str, str]) -> None:
    if not os.path.lexists(files_dir):
        return
    if files_dir.is_symlink() or not files_dir.is_dir():
        raise PreparationError(f"Refusing linked or non-directory fixtures: {files_dir}")
    for directory, dirnames, filenames in os.walk(files_dir, followlinks=False):
        dirnames.sort()
        if not dirnames and not filenames and Path(directory) != files_dir:
            rel = Path(directory).relative_to(files_dir).as_posix()
            fixtures[rel], owners[rel] = None, owner
        for entry in [*dirnames, *sorted(filenames)]:
            path = Path(directory) / entry
            if path.is_symlink():
                raise PreparationError(f"Refusing linked fixture: {path}")
        for filename in sorted(filenames):
            path = Path(directory) / filename
            rel = path.relative_to(files_dir).as_posix()
            data = _read_regular(path)
            if rel in fixtures and fixtures[rel] != data:
                raise PreparationError(
                    f"fixture collision on 'files/{rel}': '{owners[rel]}' and '{owner}' provide different content"
                )
            fixtures[rel], owners[rel] = data, owners.get(rel, owner)


def _safe_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._") or "plugin-eval-case"


def _unique_id(base: str, seen: set[str]) -> str:
    candidate, suffix = base, 2
    while candidate in seen:
        candidate, suffix = f"{base}-{suffix}", suffix + 1
    seen.add(candidate)
    return candidate
