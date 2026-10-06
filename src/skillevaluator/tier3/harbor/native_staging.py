# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a native plugin bundle into one with-plugin Harbor task.

The bundle is written to ``<task>/environment/skilleval/`` (the Docker build
context) and copied to ``/skilleval`` in the image. Every copy of authored
content goes through the secure copy (single-link regular files and
directories only, contained, never following links). Generated files are
written only beneath the bundle root, and never through an existing link.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from skillevaluator.plugin_components import is_env_file
from skillevaluator.tier3.harbor.secure_copy import UnsafeStagingError, copy_file_secure, copytree_secure
from skillevaluator.tier3.plugin_native import (
    BUNDLE_DIRNAME,
    CONTAINER_ROOT,
    HOOK_CENSUS_TEMPLATE,
    HarnessAdapter,
    NativeBundle,
    NativePluginSource,
    PluginLoadError,
    declared_plugin_paths,
    native_refusal,
)

# Plugin-root entries a native Claude Code plugin copy never carries beyond the
# runtime skill copy rules, which already skip evaluator data (evals/, results/),
# VCS, and caches: the component files the adapter marks unsupported or
# regenerates, and the project's own Claude Code settings.
_PLUGIN_TREE_IGNORED_ROOT = frozenset(
    {
        ".claude-plugin",
        ".mcp.json",
        ".lsp.json",
        "settings.json",
        "monitors",
        ".claude",
    }
)


@dataclass(frozen=True)
class NativeTaskStaging:
    """One agent's native staging plan for its with-plugin arm."""

    agent: str
    adapter: HarnessAdapter
    source: NativePluginSource
    bundle: NativeBundle

    @property
    def stage_wrapper_skill(self) -> bool:
        return self.adapter.stage_wrapper_skill

    @property
    def stage_member_skills(self) -> bool:
        return self.adapter.stage_member_skills

    @property
    def plugin_mcp_via_task(self) -> bool:
        return self.adapter.plugin_mcp_via_task

    def workspace_skill_aliases(self) -> list[str]:
        """Extra names the harness may report for the staged skills (for routing grades)."""
        return list(self.bundle.skill_aliases)


def build_native_task_staging(agent: str, adapter: HarnessAdapter, source: NativePluginSource) -> NativeTaskStaging:
    """Build one agent's bundle; refuse a component this adapter would stage with a permission bypass."""
    refusal = native_refusal(adapter, source)
    if refusal is not None:
        raise PluginLoadError(refusal)
    return NativeTaskStaging(agent=agent, adapter=adapter, source=source, bundle=adapter.build(source))


def _declared_unsupported_paths(source: NativePluginSource) -> set[str]:
    """Manifest-declared LSP and monitor files, which the native copy never carries."""
    manifest = source.manifest
    experimental = manifest.get("experimental") if isinstance(manifest.get("experimental"), dict) else {}
    return {
        path.as_posix()
        for value in (manifest.get("lspServers"), manifest.get("monitors"), experimental.get("monitors"))
        for path in declared_plugin_paths(value)
    }


def _plugin_tree_ignore(source: NativePluginSource, excluded_roots: Sequence[Path] = ()):
    from skillevaluator.tier3.harbor.adapter import _runtime_skill_copy_ignore

    root = source.plugin_root.resolve()
    declared = _declared_unsupported_paths(source)
    # The runtime skill copy rules too: results/output roots, authenticated
    # generated output (earlier runs, staged with-skill tasks), nested skill
    # evals/, and the evals source when they sit inside the plugin root.
    runtime_ignore = _runtime_skill_copy_ignore(source.plugin_root, (*excluded_roots, *source.excluded_paths))

    def _ignore(directory: str, contents: list[str]) -> list[str]:
        current = Path(directory).resolve()
        ignored = {name for name in contents if is_env_file(name)}
        ignored.update(runtime_ignore(directory, contents))
        if current == root:
            ignored.update(name for name in contents if name in _PLUGIN_TREE_IGNORED_ROOT)
        try:
            rel_dir = current.relative_to(root)
        except ValueError:
            return sorted(ignored)
        for name in contents:
            rel = (PurePosixPath(rel_dir.as_posix()) / name).as_posix().removeprefix("./")
            if rel in declared:
                ignored.add(name)
            # A skill's own evals/ (datasets and expected outputs) never reach the agent.
            if name.casefold() == "evals" and (current / "SKILL.md").is_file():
                ignored.add(name)
        return sorted(ignored)

    return _ignore


def _write_generated(bundle_root: Path, rel: str, text: str) -> None:
    rel_path = PurePosixPath(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts or not rel_path.parts:
        raise UnsafeStagingError(f"Generated native bundle path escapes the bundle: {rel}")
    target = bundle_root.joinpath(*rel_path.parts)
    current = bundle_root
    for part in rel_path.parts[:-1]:
        current = current / part
        if os.path.lexists(current):
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise UnsafeStagingError(f"Refusing to write the native bundle through a non-directory: {rel}")
        else:
            current.mkdir()
    if os.path.lexists(target):
        metadata = target.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise UnsafeStagingError(f"Refusing to overwrite a linked or special file in the native bundle: {rel}")
        target.unlink()
    target.write_text(text, encoding="utf-8")


def stage_native_bundle(
    env_dir: Path,
    staging: NativeTaskStaging,
    *,
    excluded_roots: Sequence[Path] = (),
) -> list[str]:
    """Write the bundle into ``env_dir/skilleval`` and return the Dockerfile COPY lines."""
    from skillevaluator.tier3.harbor.adapter import _runtime_skill_copy_ignore, _sanitize_staged_runtime_skills

    bundle_root = env_dir / BUNDLE_DIRNAME
    if os.path.lexists(bundle_root):
        raise UnsafeStagingError(f"Native plugin bundle path already exists: {bundle_root}")
    bundle_root.mkdir(parents=True)
    bundle = staging.bundle
    copy_file_secure(
        HOOK_CENSUS_TEMPLATE,
        bundle_root / "hook_census.sh",
        allowed_root=HOOK_CENSUS_TEMPLATE.parent,
    )
    if bundle.plugin_tree is not None:
        destination = bundle_root.joinpath(*PurePosixPath(bundle.plugin_tree).parts)
        try:
            copytree_secure(
                staging.source.plugin_root,
                destination,
                ignore=_plugin_tree_ignore(staging.source, excluded_roots),
                allowed_root=staging.source.plugin_root,
            )
        except (UnsafeStagingError, OSError) as exc:
            raise ValueError(f"Refusing to stage the plugin tree natively: {exc}") from exc
        _sanitize_staged_runtime_skills(destination)
    for source_dir, dest_rel in bundle.trees:
        destination = bundle_root.joinpath(*PurePosixPath(dest_rel).parts)
        if os.path.lexists(destination):
            # The adapter picks free names; an existing path would silently load
            # another skill under this member skill's name.
            raise UnsafeStagingError(f"Native member skill destination already exists: {dest_rel}")
        copytree_secure(
            source_dir,
            destination,
            ignore=_runtime_skill_copy_ignore(source_dir, excluded_roots),
        )
        _sanitize_staged_runtime_skills(destination)
    for rel, text in sorted(bundle.generated.items()):
        _write_generated(bundle_root, rel, text)
    return [f"COPY {BUNDLE_DIRNAME}/ {CONTAINER_ROOT}/"]
