# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor Adapter -- converts evals/evals.json to Harbor task directories.

Generates two Harbor datasets from a single evals.json:
  - harbor-tasks/        (with skill installed)
  - harbor-tasks-baseline/ (without skill, reference skills only)

Each dataset entry becomes one Harbor task directory with:
  instruction.md, task.toml, environment/Dockerfile or environment/skills,
  tests/eval.py, tests/entry.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import tomllib
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from skillevaluator.tier3.case_ids import safe_child, validate_case_ids, validate_output_directory_path
from skillevaluator.tier3.harbor.secure_copy import (
    copy_file_secure,
    copytree_secure,
    tree_content_fingerprint_secure,
)
from skillevaluator.tier3.output_provenance import (
    GENERATED_OUTPUT_MARKER,
    is_generated_output_root,
    output_provenance_key_path,
    validate_generated_output_replacement,
    validate_provenance_key_outside,
    write_generated_output_marker,
)
from skillevaluator.tier3.toml_utils import toml_quote
from skillevaluator.utils.process_environment import child_process_env
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
_EVAL_CORE_DIR = Path(__file__).resolve().parent.parent / "eval_core"
_BASE_IMAGE_PREFIX = "skillevaluator-base"
_MAX_REPO_CONTEXT_FILE_BYTES = 10 * 1024 * 1024
_MAX_REPO_CONTEXT_TOTAL_BYTES = 200 * 1024 * 1024
_MAX_EVALUATOR_FILE_BYTES = 64 * 1024 * 1024
_MAX_EVALS_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_MCP_CONFIG_BYTES = 1024 * 1024
_EVALUATOR_DATASET_FILENAMES = (
    "evals.json",
    "evals.jsonl",
    "evals.yaml",
    "evals.yml",
    "dataset.json",
    "dataset.jsonl",
    "dataset.yaml",
    "dataset.yml",
)
_EVALUATOR_ONLY_TASK_INPUT_FILES = frozenset(
    {
        *(name.casefold() for name in _EVALUATOR_DATASET_FILENAMES),
        "benchmark_" + "conversion_report.md",
        "config.yaml",
        "config.yml",
        "eval.md",
        "grader.py",
        "grader.sh",
        GENERATED_OUTPUT_MARKER.casefold(),
    }
)
_EVALUATOR_ONLY_TASK_INPUT_ROOTS = frozenset({"environment", "harbor", "results", "tests"})
_REPO_CONTEXT_IGNORE_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    "target",
    ".env",
    ".env.local",
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "secrets.json",
}
_REPO_CONTEXT_IGNORE_PARTS = {("evals", "results")}
_REPO_CONTEXT_IGNORE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_NATIVE_SOURCE_IGNORE_NAMES = ("results", "__pycache__", ".git", GENERATED_OUTPUT_MARKER)
_NATIVE_SOURCE_IGNORE = shutil.ignore_patterns(*_NATIVE_SOURCE_IGNORE_NAMES)
_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE = os.name == "posix"
_REPO_CONTEXT_PUBLIC_ENV_SUFFIXES = (".dist", ".example", ".sample", ".template")
_REPO_CONTEXT_SENSITIVE_NAMES = {
    ".git-credentials",
    ".gitcredentials",
    ".npmrc",
    ".pypirc",
    ".terraformrc",
    ".yarnrc",
    ".yarnrc.yml",
    "_netrc",
    "access_tokens.db",
    "application_default_credentials.json",
    "credentials.db",
    "credentials.tfrc.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "terraform.rc",
}
_REPO_CONTEXT_SENSITIVE_COMPONENTS = {
    ".azure",
}
_REPO_CONTEXT_SENSITIVE_PARTS = {
    (".aws", "config"),
    (".aws", "credentials"),
    (".config", "doctl", "config.yaml"),
    (".config", "gcloud"),
    (".config", "gh", "hosts.yml"),
    (".config", "pypoetry", "auth.toml"),
    (".docker", "config.json"),
    (".kube", "config"),
    (".pulumi", "credentials.json"),
    (".terraform.d", "credentials.tfrc.json"),
}
_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
_PLAIN_RELATIVE_PATH_RE = re.compile(r"(?<![\w/])(?:\.\.?/)+(?:[^\s\])<>\"']+)")
_LOCAL_LINK_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_COMPOSE_ENV_RE = re.compile(r"(?<!\$)(?:\$\$)*\$(?:\{([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*))")
_COMPOSE_NAMED_VOLUME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COMPOSE_SSH_PATH_RE = re.compile(r"^(?:[^/\\\s@:]+@)?[^/\\\s:]+:.+$")
_COMPOSE_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_COMPOSE_ALLOWED_TOP_LEVEL_KEYS = frozenset({"networks", "services", "version", "volumes"})
_COMPOSE_SIDECAR_ALLOWED_KEYS = frozenset(
    {
        "build",
        "cap_drop",
        "command",
        "depends_on",
        "entrypoint",
        "environment",
        "expose",
        "healthcheck",
        "image",
        "init",
        "networks",
        "platform",
        "ports",
        "pull_policy",
        "read_only",
        "stop_grace_period",
        "stop_signal",
        "tmpfs",
        "user",
        "volumes",
        "working_dir",
    }
)
_COMPOSE_MAIN_ALLOWED_KEYS = frozenset({"depends_on"})
_COMPOSE_ALLOWED_BUILD_KEYS = frozenset(
    {
        "additional_contexts",
        "args",
        "context",
        "dockerfile",
        "dockerfile_inline",
        "labels",
        "no_cache",
        "pull",
        "target",
    }
)
_COMPOSE_ALLOWED_NETWORK_KEYS = frozenset({"attachable", "enable_ipv4", "enable_ipv6", "internal", "labels"})
_COMPOSE_ALLOWED_VOLUME_KEYS = frozenset({"labels"})
_VERIFIER_JUDGE_MODEL_ENV_VARS = frozenset({"LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"})
_VERIFIER_PROVIDER_ENV_VARS = frozenset(
    {
        "SKILL_EVAL_LLM_PROVIDER",
        "SKILL_EVAL_LLM_MODEL",
        "SKILL_EVAL_LLM_API_KEY",
        "SKILL_EVAL_LLM_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "NVIDIA_API_KEY",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_SDK_LOAD_CONFIG",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    }
)


def _verifier_env_vars(runtime_env: dict[str, str] | None = None) -> tuple[str, ...]:
    """Return public provider variables explicitly staged for the verifier."""
    return tuple(sorted(set(runtime_env or {}).intersection(_VERIFIER_PROVIDER_ENV_VARS)))


def _verifier_env_block(runtime_env: dict[str, str] | None = None, indent: str = "") -> str:
    return "\n".join(f'{indent}{name} = "${{{name}}}"' for name in _verifier_env_vars(runtime_env))


def _find_repo_root(path: Path) -> Path | None:
    """Return the git repo root for *path*, falling back to parent .git search."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip()).resolve()
    except Exception:
        pass

    current = path.resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _repo_context_root(path: Path) -> Path:
    """Return the exact source root used by repo-context staging."""
    try:
        repo_root = _find_repo_root(path)
        resolved = path.resolve()
        return repo_root or resolved.parent
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve repository context root for: {path}") from exc


def validate_output_provenance_key_location(
    skill_path: Path,
    output_dir: Path,
    *,
    reference_skills_dir: Path | None,
    workspace_skill_paths: Sequence[Path],
) -> None:
    """Keep the private marker key out of every tree that may reach an agent."""
    protected_roots = [skill_path, output_dir, *workspace_skill_paths]
    if reference_skills_dir is not None:
        protected_roots.append(reference_skills_dir)
    repo_root = _find_repo_root(skill_path)
    if repo_root is not None:
        protected_roots.append(repo_root)
    for protected_root in protected_roots:
        validate_provenance_key_outside(protected_root)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


_SKILL_MANIFEST_NAMES = ("SKILL.md", "skill.md")


def _skill_manifest(path: Path) -> Path | None:
    for name in _SKILL_MANIFEST_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def _iter_skill_manifests(root: Path) -> list[Path]:
    manifests: set[Path] = set()
    for name in _SKILL_MANIFEST_NAMES:
        manifests.update(root.rglob(name))
    return sorted(manifests)


def _find_target_manifest_payload(
    root: Path,
    target_skill: Path,
    *,
    runtime_projection: bool = False,
    excluded_roots: Sequence[Path] = (),
    ignored_parts: frozenset[str] = frozenset(),
    ignored_path_predicate: Callable[[Path], bool] | None = None,
) -> Path | None:
    """Find a renamed file containing the exact evaluated-skill instructions."""
    target_manifest = _skill_manifest(target_skill)
    if target_manifest is None:
        return None
    target_payload = target_manifest.read_bytes()
    runtime_ignore = _runtime_skill_copy_ignore(root, excluded_roots) if runtime_projection else None
    for candidate in root.rglob("*"):
        if ignored_path_predicate is not None and ignored_path_predicate(candidate):
            continue
        relative_parts = candidate.relative_to(root).parts
        if ignored_parts.intersection(relative_parts):
            continue
        if runtime_ignore is not None and _runtime_projection_path_is_ignored(candidate, root, runtime_ignore):
            continue
        try:
            metadata = candidate.lstat()
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1 and metadata.st_size >= len(target_payload):
            overlap = max(0, len(target_payload) - 1)
            previous = b""
            try:
                with candidate.open("rb") as stream:
                    while chunk := stream.read(64 * 1024):
                        combined = previous + chunk
                        if target_payload in combined:
                            return candidate
                        previous = combined[-overlap:] if overlap else b""
            except OSError:
                continue
    return None


def _casefold_contained(candidate: Path, root: Path) -> bool:
    """Return whether *candidate* is lexically below *root*, ignoring path casing."""
    candidate_parts = tuple(part.casefold() for part in candidate.parts)
    root_parts = tuple(part.casefold() for part in root.parts)
    return len(candidate_parts) >= len(root_parts) and candidate_parts[: len(root_parts)] == root_parts


def _path_is_excluded(path: Path, excluded_roots: Sequence[Path]) -> bool:
    """Check lexical and resolved containment using cross-platform path casing."""

    for excluded_root in excluded_roots:
        if _casefold_contained(path.absolute(), excluded_root.absolute()):
            return True
        try:
            resolved_path = path.resolve()
            resolved_root = excluded_root.resolve()
        except (OSError, RuntimeError):
            continue
        if _casefold_contained(resolved_path, resolved_root):
            return True
    return False


def _paths_equivalent(first: Path, second: Path) -> bool:
    """Return whether two paths identify the same cross-platform path."""
    return _path_is_excluded(first, (second,)) and _path_is_excluded(second, (first,))


def _path_is_link_or_reparse(path: Path, metadata: os.stat_result | None = None) -> bool:
    """Return whether a path is a symlink, junction, or Windows reparse point."""
    if metadata is None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = getattr(metadata, "st_file_attributes", 0) or 0
    is_junction = getattr(path, "is_junction", None)
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(file_attributes & reparse_flag)
        or (callable(is_junction) and is_junction())
    )


def _path_is_canonically_contained(path: Path, root: Path) -> bool:
    """Require exact-case lexical and resolved containment for safety exceptions."""
    for candidate, canonical_root in (
        (path.absolute(), root.absolute()),
        (path.resolve(), root.resolve()),
    ):
        try:
            candidate.relative_to(canonical_root)
        except ValueError:
            return False
    return True


def _authenticated_generated_output_ancestor(
    path: Path,
    root: Path,
    authenticated_roots: set[Path] | None = None,
) -> Path | None:
    """Find an authenticated generated-output boundary or reject an unsafe marker."""
    lexical_path = path.absolute()
    for authenticated_root in authenticated_roots or ():
        if _casefold_contained(lexical_path, authenticated_root.absolute()):
            return authenticated_root

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        current = path.parent
    except OSError as exc:
        raise ValueError(f"Cannot inspect potential generated output path: {path}") from exc
    else:
        current = (
            path if stat.S_ISDIR(metadata.st_mode) and not _path_is_link_or_reparse(path, metadata) else path.parent
        )

    lexical_root = root.absolute()
    try:
        resolved_root = root.resolve()
    except (OSError, RuntimeError):
        resolved_root = lexical_root

    def _inside_root(candidate: Path) -> bool:
        absolute_candidate = candidate.absolute()
        if _casefold_contained(absolute_candidate, lexical_root) or _casefold_contained(
            absolute_candidate, resolved_root
        ):
            return True
        try:
            return _casefold_contained(candidate.resolve(), resolved_root)
        except (OSError, RuntimeError):
            return False

    while _inside_root(current):
        marker = current / GENERATED_OUTPUT_MARKER
        if os.path.lexists(marker):
            if _path_is_link_or_reparse(current):
                raise ValueError(f"Generated output marker is under an unsafe linked directory: {current}")
            try:
                authenticated = is_generated_output_root(current)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Generated output marker is invalid or cannot be authenticated: {current}") from exc
            if not authenticated:
                raise ValueError(f"Generated output marker is invalid or cannot be authenticated: {current}")
            if authenticated_roots is not None:
                authenticated_roots.add(current)
            return current
        if _paths_equivalent(current, root):
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _authenticated_generated_output_latest_alias(
    path: Path,
    root: Path,
    authenticated_roots: set[Path] | None = None,
) -> bool:
    """Recognize only ``latest`` links to an authenticated immediate run child."""
    if path.name != "latest":
        return False
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if not stat.S_ISLNK(metadata.st_mode):
        return False
    try:
        relative_target = path.readlink()
    except OSError:
        return False
    target_text = os.fspath(relative_target)
    if (
        relative_target.is_absolute()
        or len(relative_target.parts) != 1
        or target_text in {"", ".", ".."}
        or "/" in target_text
        or "\\" in target_text
    ):
        return False
    target = path.parent / relative_target
    try:
        target_metadata = target.lstat()
    except OSError:
        return False
    if _path_is_link_or_reparse(target, target_metadata) or not stat.S_ISDIR(target_metadata.st_mode):
        return False
    authenticated = _authenticated_generated_output_ancestor(target, root, authenticated_roots)
    return authenticated is not None and _paths_equivalent(authenticated, target)


def _inside_declared_generated_dataset(path: Path, declared_roots: Sequence[Path]) -> bool:
    """Return whether *path* is below a staged dataset marker in a declared output root."""
    for declared_root in declared_roots:
        if not _path_is_excluded(path, (declared_root,)):
            continue
        if _authenticated_generated_output_ancestor(path, declared_root) is not None:
            return True
    return False


def _target_runtime_source_roots(
    skill_path: Path,
    declared_output_roots: Sequence[Path] = (),
) -> tuple[Path, ...]:
    """Return standard and nested-package runtime subtrees that output cannot replace."""
    roots = {skill_path / name for name in ("assets", "references", "scripts", "templates")}
    default_results_root = skill_path / "evals" / "results"
    for manifest in _iter_skill_manifests(skill_path):
        if _path_is_canonically_contained(manifest, default_results_root):
            continue
        owner = manifest.parent
        if not _paths_equivalent(owner, skill_path) and not _inside_declared_generated_dataset(
            manifest, declared_output_roots
        ):
            roots.add(owner)
    return tuple(sorted(roots))


def _validate_output_roots_outside_evaluator_sources(
    skill_path: Path,
    output_roots: Sequence[Path],
) -> None:
    """Reject output roots that would be copied back as evaluator source data."""
    evals_dir = skill_path / "evals"
    results_dir = evals_dir / "results"
    for output_root in output_roots:
        if not _path_is_excluded(output_root, (skill_path,)):
            continue
        if _path_is_excluded(output_root, (evals_dir,)) and not _path_is_canonically_contained(
            output_root, results_dir
        ):
            raise ValueError(
                f"Generated output root must not be inside evaluator source directory '{evals_dir}': {output_root}"
            )
        if _paths_equivalent(output_root, skill_path):
            raise ValueError(f"Generated output root must not equal the runtime skill root: {skill_path}")
        for source_root in _target_runtime_source_roots(skill_path, output_roots):
            if _path_is_excluded(output_root, (source_root,)) or _path_is_excluded(source_root, (output_root,)):
                raise ValueError(
                    f"Generated output root must not overlap runtime skill source '{source_root}': {output_root}"
                )


def _validate_staging_output_location(
    skill_path: Path,
    output_dir: Path,
    *,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: Sequence[Path] = (),
    declared_output_roots: Sequence[Path] = (),
) -> None:
    """Reject a staging destination that could delete evaluator source data."""
    validate_output_directory_path(output_dir)
    evals_dir = skill_path / "evals"
    results_dir = evals_dir / "results"
    output_is_results = _path_is_canonically_contained(output_dir, results_dir)
    if not output_is_results and (
        _path_is_excluded(output_dir, (evals_dir,)) or _path_is_excluded(evals_dir, (output_dir,))
    ):
        raise ValueError(f"Staging output directory overlaps evaluator source directory '{evals_dir}': {output_dir}")
    for source_root in _target_runtime_source_roots(skill_path, declared_output_roots):
        if _path_is_excluded(output_dir, (source_root,)) or _path_is_excluded(source_root, (output_dir,)):
            raise ValueError(f"Staging output directory overlaps runtime skill source '{source_root}': {output_dir}")
    output_inside_skill = _path_is_excluded(output_dir, (skill_path,))
    output_contains_skill = _path_is_excluded(skill_path, (output_dir,))
    declared_in_skill = any(
        _path_is_excluded(output_dir, (declared_root,))
        and _path_is_excluded(declared_root, (skill_path,))
        and not _paths_equivalent(declared_root, skill_path)
        for declared_root in declared_output_roots
    )
    if output_contains_skill or (output_inside_skill and not output_is_results and not declared_in_skill):
        raise ValueError(f"Staging output directory overlaps runtime skill source '{skill_path}': {output_dir}")

    runtime_sources = [*(workspace_skill_paths or [])]
    if reference_skills_dir is not None:
        runtime_sources.append(reference_skills_dir)
    for source_root in runtime_sources:
        if _path_is_excluded(skill_path, (source_root,)) and _path_is_excluded(output_dir, (skill_path,)):
            continue
        if _path_is_excluded(output_dir, (source_root,)) or _path_is_excluded(source_root, (output_dir,)):
            raise ValueError(f"Staging output directory overlaps runtime skill source '{source_root}': {output_dir}")


def validate_results_root_location(
    skill_path: Path,
    results_root: Path,
    *,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: Sequence[Path] = (),
) -> None:
    """Validate that a result root cannot become evaluator input or build context."""
    validate_output_directory_path(results_root)
    if _path_is_excluded(skill_path, (results_root,)):
        raise ValueError(f"Generated output root must not contain the runtime skill source: {results_root}")
    _validate_output_roots_outside_evaluator_sources(skill_path, (results_root,))
    runtime_sources = [*(workspace_skill_paths or [])]
    if reference_skills_dir is not None:
        runtime_sources.append(reference_skills_dir)
    for source_root in runtime_sources:
        if _path_is_excluded(skill_path, (source_root,)) and _path_is_excluded(results_root, (skill_path,)):
            continue
        if _path_is_excluded(results_root, (source_root,)):
            raise ValueError(
                f"Generated output root must not be inside runtime skill source '{source_root}': {results_root}"
            )


def _skill_scoped_excluded_roots(skill_root: Path, excluded_roots: Sequence[Path]) -> tuple[Path, ...]:
    """Keep only generated-output roots located inside this skill tree."""
    scoped: list[Path] = []
    for excluded_root in excluded_roots:
        if _paths_equivalent(excluded_root, skill_root):
            raise ValueError(f"Generated output root must not equal the runtime skill root: {skill_root}")
        if _path_is_excluded(excluded_root, (skill_root,)):
            scoped.append(excluded_root)
    return tuple(scoped)


def _is_skill_evals_path(path: Path, skill_root: Path) -> bool:
    """Return whether *path* is under ``evals/`` owned by any nested skill package."""
    for candidate, root in (
        (path.absolute(), skill_root.absolute()),
        (path.resolve(), skill_root.resolve()),
    ):
        try:
            rel = candidate.relative_to(root)
        except ValueError:
            continue
        for index, part in enumerate(rel.parts):
            if part.casefold() != "evals":
                continue
            owner = root.joinpath(*rel.parts[:index])
            if _skill_manifest(owner) is not None and (owner / part).is_dir():
                return True
    return False


def _runtime_skill_copy_ignore(skill_root: Path, excluded_roots: Sequence[Path] = ()):
    """Build a root-aware ignore callback for an agent-visible skill copy."""
    resolved_root = skill_root.resolve()
    excluded_roots = _skill_scoped_excluded_roots(skill_root, excluded_roots)
    authenticated_output_roots: set[Path] = set()
    if (
        _authenticated_generated_output_ancestor(
            skill_root,
            skill_root,
            authenticated_output_roots,
        )
        is not None
    ):
        raise ValueError(f"Runtime skill source must not be a generated output root: {skill_root}")

    def _ignore(directory: str, contents: list[str]) -> list[str]:
        current = Path(directory)
        ignored = {name for name in contents if name in {"results", "__pycache__", ".git"}}
        if current.resolve() == resolved_root:
            ignored.update(name for name in contents if name.casefold() == "evals")
        for name in contents:
            candidate = current / name
            if (
                _is_skill_evals_path(candidate, skill_root)
                or _path_is_excluded(candidate, excluded_roots)
                or _authenticated_generated_output_ancestor(
                    candidate,
                    skill_root,
                    authenticated_output_roots,
                )
                is not None
                or _authenticated_generated_output_latest_alias(
                    candidate,
                    skill_root,
                    authenticated_output_roots,
                )
            ):
                ignored.add(name)
        return sorted(ignored)

    return _ignore


def _runtime_projection_path_is_ignored(path: Path, root: Path, ignore) -> bool:
    """Apply a copytree ignore callback to every component of one source path."""
    current = root
    for part in path.relative_to(root).parts:
        if part in ignore(str(current), [part]):
            return True
        current /= part
    return False


def _runtime_skill_fingerprint(skill_root: Path, excluded_roots: Sequence[Path] = ()) -> str | None:
    """Hash the agent-visible runtime tree independent of its directory name."""
    if _skill_manifest(skill_root) is None:
        return None
    ignore = _runtime_skill_copy_ignore(skill_root, excluded_roots)
    digest = hashlib.sha256()
    pending = [skill_root]
    while pending:
        directory = pending.pop()
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        ignored = set(ignore(str(directory), [entry.name for entry in entries]))
        for entry in entries:
            if entry.name in ignored:
                continue
            path = Path(entry.path)
            metadata = entry.stat(follow_symlinks=False)
            if _path_is_link_or_reparse(path, metadata):
                return None
            relative = path.relative_to(skill_root).as_posix().encode("utf-8")
            if stat.S_ISDIR(metadata.st_mode):
                digest.update(b"D\\0" + relative + b"\\0")
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    return None
                digest.update(b"F\\0" + relative + b"\\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                return None
    return digest.hexdigest()


def _check_baseline_skill_candidates_do_not_alias_target(
    target_skill: Path,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None,
    *,
    excluded_roots: Sequence[Path] = (),
) -> None:
    target_fingerprint = _runtime_skill_fingerprint(target_skill, excluded_roots)
    target_manifest = _skill_manifest(target_skill)
    if target_fingerprint is None or target_manifest is None:
        return
    target_manifest_digest = hashlib.sha256(target_manifest.read_bytes()).digest()
    candidates: list[Path] = []
    if reference_skills_dir and reference_skills_dir.exists():
        candidates.extend(
            candidate
            for candidate in reference_skills_dir.iterdir()
            if candidate.is_dir() and not candidate.name.startswith(".")
        )
    candidates.extend(workspace_skill_paths or [])
    seen: set[Path] = set()
    for candidate in candidates:
        if _paths_equivalent(candidate, target_skill):
            continue
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_dir():
            continue
        seen.add(resolved)
        candidate_manifest = _skill_manifest(candidate)
        manifest_matches = candidate_manifest is not None and (
            hashlib.sha256(candidate_manifest.read_bytes()).digest() == target_manifest_digest
        )
        payload_alias = _find_target_manifest_payload(
            candidate,
            target_skill,
            runtime_projection=True,
            excluded_roots=excluded_roots,
        )
        if (
            manifest_matches
            or payload_alias is not None
            or _runtime_skill_fingerprint(candidate, excluded_roots) == target_fingerprint
        ):
            raise ValueError(
                f"Baseline skill candidate '{candidate}' is an alias of target skill '{target_skill.name}'. "
                "A baseline must not contain the evaluated skill under another directory name."
            )


@dataclass(frozen=True, slots=True)
class _BaselineAliasValidation:
    """Run-scoped proof that one exact baseline source set was prevalidated."""

    source_key: tuple[Any, ...]


def _baseline_alias_source_key(
    target_skill: Path,
    reference_skills_dir: Path | None,
    workspace_skill_paths: Sequence[Path] | None,
    excluded_roots: Sequence[Path],
) -> tuple[Any, ...]:
    def _path_key(path: Path) -> str:
        return os.path.normcase(os.fspath(path.expanduser().resolve(strict=False)))

    return (
        _path_key(target_skill),
        _path_key(reference_skills_dir) if reference_skills_dir is not None else None,
        tuple(sorted(_path_key(path) for path in workspace_skill_paths or ())),
        tuple(sorted(_path_key(path) for path in excluded_roots)),
    )


def _prevalidate_baseline_skill_candidates(
    target_skill: Path,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None,
    *,
    excluded_roots: Sequence[Path] = (),
) -> _BaselineAliasValidation:
    """Validate configured baseline sources once before a multi-agent run."""
    _check_baseline_skill_candidates_do_not_alias_target(
        target_skill,
        reference_skills_dir,
        workspace_skill_paths,
        excluded_roots=excluded_roots,
    )
    return _BaselineAliasValidation(
        _baseline_alias_source_key(
            target_skill,
            reference_skills_dir,
            workspace_skill_paths,
            excluded_roots,
        )
    )


def _baseline_alias_validation_matches(
    validation: _BaselineAliasValidation,
    target_skill: Path,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None,
    excluded_roots: Sequence[Path],
) -> bool:
    return validation.source_key == _baseline_alias_source_key(
        target_skill,
        reference_skills_dir,
        workspace_skill_paths,
        excluded_roots,
    )


def _check_staged_baseline_does_not_contain_target(env_dir: Path, target_skill: Path) -> None:
    """Fail closed if any agent-visible staged file is the evaluated instructions."""
    payload = _find_target_manifest_payload(env_dir, target_skill)
    if payload is not None:
        raise ValueError(
            f"Baseline environment contains the target skill instructions in {payload}. "
            "Remove copied or renamed target content from reference skills, task input, and repo context."
        )


def _repo_path_is_skill_evals(path: Path, root: Path) -> bool:
    """Identify evaluator-only roots without dropping unrelated nested ``evals``."""
    return _is_skill_evals_path(path, root)


def _strip_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if not target:
        return ""
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    elif any(ch.isspace() for ch in target):
        # Markdown allows optional titles: [x](../file.md "title").
        target = target.split(None, 1)[0]
    target = target.split("#", 1)[0].split("?", 1)[0]
    target = unquote(target)
    return target.strip().strip("'\"")


def _is_local_link_target(target: str) -> bool:
    if not target or target.startswith("#"):
        return False
    return not _LOCAL_LINK_SCHEME_RE.match(target)


def _discover_skill_link_targets(skill_md: Path) -> list[str]:
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return []

    targets: list[str] = []
    for match in _MARKDOWN_LINK_RE.finditer(text):
        target = _strip_link_target(match.group(1))
        if _is_local_link_target(target):
            targets.append(target)

    for match in _PLAIN_RELATIVE_PATH_RE.finditer(text):
        target = _strip_link_target(match.group(0).rstrip(".,;:"))
        if _is_local_link_target(target):
            targets.append(target)

    seen: set[str] = set()
    unique: list[str] = []
    for target in targets:
        if target not in seen:
            seen.add(target)
            unique.append(target)
    return unique


def _safe_repo_context_rel(path: Path, root: Path) -> Path | None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    try:
        rel = path.absolute().relative_to(root.absolute())
    except ValueError:
        return None
    if any(part in {"", ".", ".."} for part in rel.parts):
        return None
    return rel


def _staged_relative_link_path(skill_name: str, raw_target: str) -> Path | None:
    target = _strip_link_target(raw_target)
    if not target or Path(target).is_absolute():
        return None
    normalized = posixpath.normpath(posixpath.join("skills", skill_name, target.replace("\\", "/")))
    if normalized in {"", "."} or normalized.startswith("../"):
        return None
    rel = Path(normalized)
    if rel.parts and rel.parts[0].casefold() in {"input", "output", "repo"}:
        raise ValueError(f"SKILL.md link resolves into reserved runtime path '/workspace/{rel.parts[0]}': {raw_target}")
    # Runtime skills and agent configuration are projected separately after
    # recursive evaluator-only filtering. Linked repository content must never
    # create an alternate project-level discovery/configuration namespace.
    if rel.parts and rel.parts[0].casefold() in {
        "skills",
        ".agents",
        ".claude",
        ".cline",
        ".codex",
        ".config",
        ".cursor",
        ".gemini",
        ".opencode",
        ".qwen",
    }:
        return None
    return rel


def _repo_context_ignore_file(
    path: Path,
    root: Path,
    authenticated_output_roots: set[Path] | None = None,
) -> bool:
    if authenticated_output_roots is None:
        authenticated_output_roots = set()
    if _authenticated_generated_output_ancestor(path, root, authenticated_output_roots) is not None:
        return True
    if _authenticated_generated_output_latest_alias(path, root, authenticated_output_roots):
        return True
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    if _paths_equivalent(path, output_provenance_key_path()):
        return True
    if _repo_path_is_skill_evals(path, root):
        return True
    try:
        rel = path.absolute().relative_to(root.absolute())
    except ValueError:
        return True
    parts = tuple(part.casefold() for part in rel.parts)
    name = parts[-1]
    if any(part in _REPO_CONTEXT_IGNORE_NAMES for part in parts):
        return True
    if any(part in _REPO_CONTEXT_SENSITIVE_COMPONENTS for part in parts):
        return True
    if name in {".env", ".envrc"}:
        return True
    if name.startswith(".env.") and not name.endswith(_REPO_CONTEXT_PUBLIC_ENV_SUFFIXES):
        return True
    if name in _REPO_CONTEXT_SENSITIVE_NAMES:
        return True
    if name.endswith(_REPO_CONTEXT_IGNORE_SUFFIXES):
        return True
    for ignored in (*_REPO_CONTEXT_IGNORE_PARTS, *_REPO_CONTEXT_SENSITIVE_PARTS):
        if any(parts[index : index + len(ignored)] == ignored for index in range(len(parts) - len(ignored) + 1)):
            return True
    return False


def _copy_repo_context_file(
    src: Path,
    dest_root: Path,
    rel: Path,
    *,
    total_bytes: list[int],
    source_root: Path,
) -> bool:
    if not src.is_file():
        return False
    try:
        size = src.stat().st_size
    except OSError:
        return False
    if size > _MAX_REPO_CONTEXT_FILE_BYTES:
        logger.warning("Skipping linked repo file over size limit: %s", src)
        return False
    if total_bytes[0] + size > _MAX_REPO_CONTEXT_TOTAL_BYTES:
        logger.warning("Skipping repo context after total size limit: %s", src)
        return False
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy_file_secure(src, dest, allowed_root=source_root)
    total_bytes[0] += size
    return True


def _git_context_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [root / line for line in result.stdout.splitlines() if line.strip()]


def _iter_repo_context_files(
    root: Path,
    authenticated_output_roots: set[Path],
    excluded_roots: Sequence[Path] = (),
) -> list[Path]:
    """Enumerate repository files without inspecting excluded output trees."""
    git_files = _git_context_files(root)
    if git_files:
        return sorted(
            path
            for path in git_files
            if not _path_is_excluded(path, excluded_roots)
            and path.is_file()
            and not _repo_context_ignore_file(path, root, authenticated_output_roots)
        )

    files: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                # This lexical-first check must happen before stat/is_file or
                # descent: result trees can contain unsafe or unreadable user
                # artifacts and are explicitly outside the staged context.
                if _path_is_excluded(path, excluded_roots):
                    continue
                if _repo_context_ignore_file(path, root, authenticated_output_roots):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False) or entry.is_symlink():
                    files.append(path)
    return sorted(files)


def _stage_repo_context(
    env_dir: Path,
    *,
    source_skill_path: Path | None,
    mode: str,
    exclude_source_skill: bool = False,
    excluded_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Stage repo files referenced by SKILL.md, or the full repo when requested."""
    if not source_skill_path:
        return {"mode": "none", "files": []}

    source_skill_path = source_skill_path.resolve()
    skill_md = _skill_manifest(source_skill_path)
    repo_root = _repo_context_root(source_skill_path)
    authenticated_output_roots: set[Path] = set()
    if (
        _authenticated_generated_output_ancestor(
            repo_root,
            repo_root,
            authenticated_output_roots,
        )
        is not None
    ):
        raise ValueError(f"Repo context root must not be a generated output root: {repo_root}")

    resolved_excluded_roots: list[Path] = []
    for excluded_root in excluded_roots:
        candidate = excluded_root.resolve()
        if mode == "full" and _paths_equivalent(candidate, repo_root):
            raise ValueError(f"Repo context output root must not contain the repository root: {candidate}")
        if _path_is_excluded(repo_root, (candidate,)):
            continue
        if _path_is_excluded(candidate, (repo_root,)):
            resolved_excluded_roots.append(candidate)

    def _is_excluded_output(path: Path) -> bool:
        return _path_is_excluded(path, resolved_excluded_roots)

    repo_dest = env_dir / "repo"
    linked_root_dest = env_dir / "repo-linked-root"
    total_bytes = [0]
    staged: list[dict[str, str]] = []

    if mode == "full":
        for src in _iter_repo_context_files(
            repo_root,
            authenticated_output_roots,
            resolved_excluded_roots,
        ):
            if _is_excluded_output(src):
                continue
            if _repo_context_ignore_file(src, repo_root, authenticated_output_roots):
                continue
            if exclude_source_skill and _is_relative_to(src, source_skill_path):
                continue
            rel = _safe_repo_context_rel(src, repo_root)
            if rel is None:
                continue
            if _copy_repo_context_file(
                src,
                repo_dest,
                rel,
                total_bytes=total_bytes,
                source_root=repo_root,
            ):
                staged.append({"source": str(src), "container": f"/workspace/repo/{rel.as_posix()}"})
        metadata = {"mode": "full", "repo_root": str(repo_root), "files": staged}
        if staged:
            (env_dir / "repo-context.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    if mode != "linked" or skill_md is None:
        return {"mode": mode, "repo_root": str(repo_root), "files": []}

    for raw_target in _discover_skill_link_targets(skill_md):
        target = _strip_link_target(raw_target)
        raw_target_path = source_skill_path / target if not Path(target).is_absolute() else Path(target)
        lexical_target_path = Path(os.path.abspath(raw_target_path))  # noqa: PTH100 -- normalize without resolving
        if _path_is_excluded(lexical_target_path, resolved_excluded_roots):
            logger.warning("Skipping SKILL.md link into generated output: %s", raw_target)
            continue
        try:
            target_path = lexical_target_path.resolve(strict=True)
        except (OSError, RuntimeError):
            logger.warning("Skipping unreadable SKILL.md repo link: %s", raw_target)
            continue
        try:
            if not target_path.is_file():
                continue
        except OSError:
            continue
        if not _is_relative_to(target_path, repo_root):
            logger.warning("Skipping SKILL.md link outside repo root: %s", raw_target)
            continue
        if _is_excluded_output(target_path):
            logger.warning("Skipping SKILL.md link into generated output: %s", raw_target)
            continue
        if _repo_context_ignore_file(
            lexical_target_path,
            repo_root,
            authenticated_output_roots,
        ) or _repo_context_ignore_file(
            target_path,
            repo_root,
            authenticated_output_roots,
        ):
            logger.warning("Skipping ignored SKILL.md repo link: %s", raw_target)
            continue
        if _is_relative_to(target_path, source_skill_path):
            continue
        rel = _safe_repo_context_rel(target_path, repo_root)
        if rel is None:
            continue
        copied = _copy_repo_context_file(
            target_path,
            repo_dest,
            rel,
            total_bytes=total_bytes,
            source_root=repo_root,
        )
        compat_rel = _staged_relative_link_path(source_skill_path.name, raw_target)
        if compat_rel is not None:
            _copy_repo_context_file(
                target_path,
                linked_root_dest,
                compat_rel,
                total_bytes=total_bytes,
                source_root=repo_root,
            )
        if copied:
            staged.append(
                {
                    "source": str(target_path),
                    "container": f"/workspace/repo/{rel.as_posix()}",
                    "link": raw_target,
                }
            )

    metadata = {"mode": "linked", "repo_root": str(repo_root), "files": staged}
    if staged:
        (env_dir / "repo-context.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _collect_all_skill_deps(
    skill_path: Path | None,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None = None,
    *,
    excluded_roots: Sequence[Path] = (),
) -> tuple[list[str], list[str]]:
    """Collect pip and apt deps from target skill + staged workspace skills."""
    all_pip: set[str] = set()
    all_apt: set[str] = set()

    skill_dirs: list[Path] = []
    if skill_path and skill_path.exists():
        skill_dirs.append(skill_path)
    if reference_skills_dir and reference_skills_dir.exists():
        for ref in reference_skills_dir.iterdir():
            if ref.is_dir() and not ref.name.startswith("."):
                skill_dirs.append(ref)
    for workspace_skill in workspace_skill_paths or []:
        if workspace_skill.exists():
            skill_dirs.append(workspace_skill)

    for sd in skill_dirs:
        runtime_ignore = _runtime_skill_copy_ignore(sd, excluded_roots)
        for req_file in sd.rglob("requirements.txt"):
            if _runtime_projection_path_is_ignored(req_file, sd, runtime_ignore):
                continue
            metadata = req_file.lstat()
            if (
                _path_is_link_or_reparse(req_file, metadata)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ValueError(f"Runtime dependency manifest must be a regular non-linked file: {req_file}")
            for line in req_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    all_pip.add(line)
        for apt_file in sd.rglob("apt-packages.txt"):
            if _runtime_projection_path_is_ignored(apt_file, sd, runtime_ignore):
                continue
            metadata = apt_file.lstat()
            if (
                _path_is_link_or_reparse(apt_file, metadata)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ValueError(f"Runtime dependency manifest must be a regular non-linked file: {apt_file}")
            for line in apt_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    all_apt.add(line)

    return sorted(all_pip), sorted(all_apt)


def build_eval_base_image(
    skill_path: Path,
    reference_skills_dir: Path | None = None,
    *,
    workspace_skill_paths: list[Path] | None = None,
    evaluator_skill_path: Path | None = None,
    excluded_roots: Sequence[Path] = (),
    force_rebuild: bool = False,
    action_out: list[str] | None = None,
) -> str:
    """Pre-build a Docker base image with verifier and public-provider dependencies.

    Builds once and tags with a content hash so rebuilds only happen when
    dependencies change.  Subsequent calls return instantly when the image exists
    unless ``force_rebuild`` is true.

    When the skill provides a custom ``evals/environment/Dockerfile``, skip
    collecting deps from the skill itself — the custom Dockerfile handles
    those.  We still collect deps from reference skills.

    Returns the image tag (e.g. ``skillevaluator-base:a1b2c3d4e5f6``) or ``""``
    on failure (callers fall back to full per-task Dockerfiles).
    """
    evaluator_source = evaluator_skill_path or skill_path
    has_custom_dockerfile = (
        evaluator_source
        and (evaluator_source / "evals" / "environment" / "Dockerfile").is_file()
        and _validate_custom_dockerfile(evaluator_source / "evals" / "environment" / "Dockerfile") is None
    )

    if has_custom_dockerfile:
        extra_pip, extra_apt = _collect_all_skill_deps(
            None,
            reference_skills_dir,
            workspace_skill_paths,
            excluded_roots=excluded_roots,
        )
    else:
        extra_pip, extra_apt = _collect_all_skill_deps(
            skill_path,
            reference_skills_dir,
            workspace_skill_paths,
            excluded_roots=excluded_roots,
        )

    lines = [
        "FROM python:3.12-slim",
        "",
        "RUN apt-get -o Acquire::Retries=3 update && \\",
        "    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \\",
        "    bash curl git jq ripgrep \\",
    ]
    if extra_apt:
        lines[-1] += " \\"
        lines.append("    " + " ".join(extra_apt) + " \\")
    lines.append("    && rm -rf /var/lib/apt/lists/*")

    lines.extend(["", *_verifier_install_lines(extra_pip)])

    lines.extend(
        [
            "",
            "RUN mkdir -p /workspace/skills /workspace/input /workspace/output \\",
            "    /logs/verifier /logs/agent",
            "",
            "WORKDIR /workspace",
        ]
    )

    content = "\n".join(lines) + "\n"
    tag_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    image_tag = f"{_BASE_IMAGE_PREFIX}:{tag_hash}"

    try:
        check = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True,
            timeout=10,
            env=child_process_env(),
        )
        if check.returncode == 0 and not force_rebuild:
            logger.debug("Base image %s already exists, skipping build", image_tag)
            if action_out is not None:
                action_out.append("reused")
            return image_tag
    except (subprocess.TimeoutExpired, OSError):
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "Dockerfile").write_text(content, encoding="utf-8")

        logger.debug("Building eval base image %s ...", image_tag)
        try:
            cmd = ["docker", "build"]
            if force_rebuild:
                cmd.extend(["--pull", "--no-cache"])
            cmd.extend(["-t", image_tag, tmpdir])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=child_process_env(),
            )
        except subprocess.TimeoutExpired:
            logger.error("Base image build timed out after 600s")
            return ""

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            logger.error("Base image build failed: %s", stderr_tail)
            if action_out is not None:
                action_out.append("failed")
            return ""

        logger.debug("Built eval base image: %s", image_tag)
    if action_out is not None:
        action_out.append("rebuilt" if force_rebuild else "built")
    return image_tag


def _set_task_docker_image(task_dir: Path, image_tag: str) -> None:
    """Inject ``docker_image`` into a task's ``task.toml`` so Harbor skips building."""
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return
    content = toml_path.read_text(encoding="utf-8")
    if "docker_image" in content:
        return
    import re

    replaced = re.sub(
        r"(\[environment\]\s*\n)",
        rf'\1docker_image = "{image_tag}"\n',
        content,
        count=1,
    )
    if replaced == content:
        logger.warning("Could not inject docker_image into %s — [environment] section not found", toml_path)
        return
    toml_path.write_text(replaced, encoding="utf-8")


def _task_environment_context_hash(environment_dir: Path) -> str:
    """Hash the Docker context shape, modes, paths, and file contents."""
    digest = hashlib.sha256()
    for path in sorted(
        environment_dir.rglob("*"), key=lambda candidate: candidate.relative_to(environment_dir).as_posix()
    ):
        relative = path.relative_to(environment_dir).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            digest.update(f"D\0{relative}\0{mode:o}\0".encode())
        elif stat.S_ISREG(metadata.st_mode):
            digest.update(f"F\0{relative}\0{mode:o}\0".encode())
            digest.update(path.read_bytes())
            digest.update(b"\0")
        else:
            raise ValueError(f"Task Docker context contains an unsupported path: {path}")
    return digest.hexdigest()


def prebuild_task_environments(dataset_dirs: list[Path]) -> int:
    """Pre-build Docker images for generated task environments.

    After ``generate_harbor_tasks`` has written all task directories, this
    function builds each task's exact environment, tags it independently, and
    sets ``docker_image`` only in that task's ``task.toml`` so Harbor uses the prebuilt
    image directly (skipping ``docker compose build`` entirely).

    Returns the number of tasks configured to use prebuilt images.
    """
    # Harbor currently tears down trial environments with ``docker compose down
    # --rmi all``.  When multiple tasks share one injected ``docker_image``,
    # the first completed trial can remove the shared prebuilt image while
    # sibling trials still need it, causing later trials to fail with Docker
    # pull errors.  Keep this optimization opt-in until Harbor exposes a safe
    # way to retain prebuilt images during cleanup.
    if os.environ.get("SKILL_EVAL_HARBOR_PREBUILD_TASK_ENVS") != "1":
        logger.debug("Skipping Harbor task environment pre-build; set SKILL_EVAL_HARBOR_PREBUILD_TASK_ENVS=1 to opt in")
        return 0

    rewritten = 0

    for dataset_dir in dataset_dirs:
        if not dataset_dir or not dataset_dir.exists():
            continue

        task_dirs = [
            d
            for d in sorted(dataset_dir.iterdir())
            if d.is_dir() and not d.name.startswith((".", "_")) and (d / "environment" / "Dockerfile").exists()
        ]
        if not task_dirs:
            continue

        for task_dir in task_dirs:
            environment_dir = task_dir / "environment"
            if any((environment_dir / name).exists() for name in ("docker-compose.yaml", "docker-compose.yml")):
                logger.warning(
                    "Skipping task environment pre-build for %s because Compose build overrides require native Compose semantics",
                    task_dir,
                )
                continue
            try:
                context_hash = _task_environment_context_hash(environment_dir)
            except (OSError, ValueError) as error:
                logger.warning("Skipping unsafe task environment pre-build for %s: %s", task_dir, error)
                continue
            task_identity = hashlib.sha256(f"{dataset_dir.resolve()}::{task_dir.name}".encode()).hexdigest()[:8]
            tag = f"skillevaluator-env:{context_hash[:12]}-{task_identity}"

            try:
                check = subprocess.run(
                    ["docker", "image", "inspect", tag],
                    capture_output=True,
                    timeout=10,
                    env=child_process_env(),
                )
                if check.returncode == 0:
                    logger.debug("Pre-built env %s exists, reusing for task %s", tag, task_dir.name)
                    _set_task_docker_image(task_dir, tag)
                    rewritten += 1
                    continue
            except (subprocess.TimeoutExpired, OSError):
                pass

            logger.debug("Pre-building task environment as %s ...", tag)
            try:
                result = subprocess.run(
                    ["docker", "build", "-t", tag, str(environment_dir)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=child_process_env(),
                )
            except subprocess.TimeoutExpired:
                logger.warning("Environment pre-build timed out for %s", tag)
                continue

            if result.returncode != 0:
                stderr_tail = (result.stderr or "")[-500:]
                logger.warning("Environment pre-build failed for %s: %s", tag, stderr_tail)
                continue

            _set_task_docker_image(task_dir, tag)
            rewritten += 1
            logger.debug("Pre-built env %s for task %s", tag, task_dir.name)

    return rewritten


def _load_evals(evals_path: Path) -> list[dict[str, Any]]:
    """Load normalized dataset entries from evals.json/jsonl/yaml."""
    from skillevaluator.tier3.dataset_utils import normalize_dataset_entries

    payload = _read_regular_evals_file(evals_path).decode("utf-8")
    suffix = evals_path.suffix.casefold()
    if suffix == ".jsonl":
        data: Any = [json.loads(line) for raw_line in payload.splitlines() if (line := raw_line.strip())]
    elif suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(payload)
    elif suffix == ".json":
        data = json.loads(payload)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")
    return normalize_dataset_entries(data)


def _evaluator_node_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _snapshot_evaluator_parent_path(path: Path, allowed_root: Path, *, label: str) -> tuple[tuple[Path, tuple], ...]:
    """Capture every real directory component anchoring an evaluator file."""
    lexical_path = path.absolute()
    lexical_root = allowed_root.absolute()
    try:
        lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside its evaluator root: {path}") from exc
    snapshot: list[tuple[Path, tuple]] = []
    current = lexical_path.parent
    while True:
        try:
            component = current.lstat()
        except OSError as exc:
            raise ValueError(f"Cannot inspect {label.lower()} path: {current}") from exc
        if _path_is_link_or_reparse(current, component):
            raise ValueError(f"{label} path must not contain symlinks or reparse points: {current}")
        if not stat.S_ISDIR(component.st_mode):
            raise ValueError(f"{label} parent path must contain only directories: {current}")
        snapshot.append((current, _evaluator_node_fingerprint(component)))
        if current == lexical_root:
            break
        if lexical_root not in current.parents:
            raise ValueError(f"{label} resolves outside its evaluator root: {path}")
        current = current.parent
    return tuple(snapshot)


def _validate_evaluator_parent_snapshot(snapshot: tuple[tuple[Path, tuple], ...], *, label: str) -> None:
    for component_path, expected in snapshot:
        try:
            component = component_path.lstat()
        except OSError as exc:
            raise ValueError(f"{label} path changed while it was read: {component_path}") from exc
        if (
            _path_is_link_or_reparse(component_path, component)
            or not stat.S_ISDIR(component.st_mode)
            or _evaluator_node_fingerprint(component) != expected
        ):
            raise ValueError(f"{label} path changed while it was read: {component_path}")


def _read_regular_evals_file(
    path: Path,
    *,
    label: str = "Eval dataset",
    max_bytes: int = _MAX_EVALUATOR_FILE_BYTES,
    allowed_root: Path | None = None,
) -> bytes:
    """Read one bounded evaluator file without following links or accepting swaps."""
    parent_snapshot = (
        _snapshot_evaluator_parent_path(path, allowed_root, label=label) if allowed_root is not None else ()
    )
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"Cannot inspect {label.lower()} file: {path}") from exc
    if _path_is_link_or_reparse(path, before) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be a regular non-linked file: {path}")
    if before.st_size > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte safety limit: {path}")
    secure_root_path = allowed_root.absolute() if allowed_root is not None else path.parent.absolute()
    try:
        relative_path = path.absolute().relative_to(secure_root_path)
        root_metadata = secure_root_path.lstat()
        with SecureRoot(secure_root_path, expected=root_metadata) as secure_root:
            payload, opened = secure_root.read_bytes(relative_path, max_bytes, expected=before)
    except (OSError, SecurePathError, ValueError) as exc:
        raise ValueError(f"{label} changed while it was read: {path}") from exc

    try:
        named_after = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} changed while it was read: {path}") from exc
    if (
        _path_is_link_or_reparse(path, named_after)
        or len(payload) != opened.st_size
        or (
            _PATH_DESCRIPTOR_IDENTITIES_COMPARABLE
            and _evaluator_node_fingerprint(named_after) != _evaluator_node_fingerprint(opened)
        )
        or (
            not _PATH_DESCRIPTOR_IDENTITIES_COMPARABLE
            and (
                _evaluator_node_fingerprint(named_after) != _evaluator_node_fingerprint(before)
                or len(payload) != named_after.st_size
            )
        )
    ):
        raise ValueError(f"{label} changed while it was read: {path}")
    if parent_snapshot:
        _validate_evaluator_parent_snapshot(parent_snapshot, label=label)
    return payload


def _validate_evals_source_directory(skill_path: Path) -> None:
    """Reject a link/reparse-point evaluator root before loading any assets."""
    evals_dir = skill_path / "evals"
    try:
        metadata = evals_dir.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"Cannot inspect evaluator source directory: {evals_dir}") from exc
    if _path_is_link_or_reparse(evals_dir, metadata):
        raise ValueError(
            f"evals directory must be a real directory, not a symlink, reparse point, or junction: {evals_dir}"
        )
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Evaluator source path must be a directory: {evals_dir}")


def find_evals_file(skill_path: Path) -> Path | None:
    """Return the first supported SkillEvaluator eval dataset for a skill, if present."""
    evals_dir = skill_path / "evals"
    _validate_evals_source_directory(skill_path)
    for name in ("evals.json", "evals.jsonl", "evals.yaml", "evals.yml", "dataset.json", "dataset.jsonl"):
        candidate = evals_dir / name
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"Cannot inspect eval dataset file: {candidate}") from exc
        if (
            _path_is_link_or_reparse(candidate, metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ValueError(f"Eval dataset must be a regular non-linked file: {candidate}")
        return candidate
    return None


def _preflight_generated_tasks(entries: list[dict[str, Any]], output_dir: Path) -> list[tuple[dict[str, Any], str]]:
    case_ids = validate_case_ids(entry.get("id") for entry in entries)
    prepared = [({**entry, "id": case_id}, case_id) for entry, case_id in zip(entries, case_ids, strict=True)]
    for _entry, case_id in prepared:
        safe_child(output_dir, case_id)
    return prepared


def _write_instruction(task_dir: Path, question: str) -> None:
    (task_dir / "instruction.md").write_text(question + "\n", encoding="utf-8")


def _load_mcp_servers(skill_path: Path) -> list[dict[str, Any]]:
    """Load MCP server declarations from evals/environment/mcp_servers.toml."""
    evals_dir = skill_path / "evals"
    environment_dir = evals_dir / "environment"
    mcp_file = environment_dir / "mcp_servers.toml"
    if not os.path.lexists(environment_dir):
        return []
    parent_snapshot = _snapshot_evaluator_parent_path(mcp_file, evals_dir, label="MCP server configuration")
    if not os.path.lexists(mcp_file):
        _validate_evaluator_parent_snapshot(parent_snapshot, label="MCP server configuration")
        return []
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    payload = _read_regular_evals_file(
        mcp_file,
        label="MCP server configuration",
        max_bytes=_MAX_MCP_CONFIG_BYTES,
        allowed_root=evals_dir,
    )
    try:
        data = tomllib.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            logger.warning("mcp_servers.toml: expected a TOML table, got %s", type(data).__name__)
            return []
        servers = data.get("mcp_servers", [])
        if not isinstance(servers, list):
            logger.warning("mcp_servers.toml: expected [[mcp_servers]] array, got %s", type(servers).__name__)
            return []
        valid = []
        for s in servers:
            if not isinstance(s, dict) or "name" not in s:
                logger.warning("mcp_servers.toml: skipping entry missing 'name': %s", s)
                continue
            if "url" not in s and "command" not in s:
                logger.warning("mcp_servers.toml: entry '%s' needs 'url' or 'command'", s.get("name"))
                continue
            if "command" in s and "transport" not in s:
                s = {**s, "transport": "stdio"}
                logger.debug("mcp_servers.toml: inferred transport=stdio for '%s'", s["name"])
            valid.append(s)
        if valid:
            logger.debug("Loaded %d MCP server(s) from %s", len(valid), mcp_file)
        return valid
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        logger.warning("Failed to parse %s: %s", mcp_file, e)
        return []


def _task_resource_value(resources: dict[str, int] | None, key: str, default: int) -> int:
    if not resources or key not in resources:
        return default
    return int(resources[key])


def _write_task_toml(
    task_dir: Path,
    entry: dict[str, Any],
    has_skill: bool,
    mcp_servers: list[dict[str, Any]] | None = None,
    docker_image: str = "",
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
) -> None:
    entry_id = entry.get("id", "unknown")
    expected_skill = entry.get("expected_skill") or "none"
    if not isinstance(entry_id, str):
        raise TypeError("entry id must be a string before Harbor TOML serialization")
    if not isinstance(expected_skill, str):
        raise TypeError("expected_skill must be a string before Harbor TOML serialization")
    if not isinstance(docker_image, str):
        raise TypeError("docker_image must be a string before Harbor TOML serialization")
    docker_image_line = f"docker_image = {_toml_quote(docker_image)}\n" if docker_image else ""
    cpus = _task_resource_value(task_resources, "cpus", 2)
    memory_mb = _task_resource_value(task_resources, "memory_mb", 4096)
    storage_mb = _task_resource_value(task_resources, "storage_mb", 2048)
    workdir_line = f"workdir = {_toml_quote(agent_workdir)}\n" if agent_workdir else ""

    content = f"""schema_version = "1.3"

[task]
name = {_toml_quote(f"nvidia/skillevaluator-{entry_id}")}
description = {_toml_quote(f"Skill evaluation task for {expected_skill}")}

[metadata]
skill = {_toml_quote(expected_skill)}
entry_id = {_toml_quote(entry_id)}
has_skill = {str(has_skill).lower()}

[agent]
timeout_sec = 300.0

[verifier]
timeout_sec = 180.0

[verifier.env]
{_verifier_env_block(verifier_env if verifier_env is not None else runtime_env)}

[environment]
{docker_image_line}cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = {storage_mb}
{workdir_line}\
network_mode = "public"
skills_dir = "/workspace/skills"
"""

    content += _runtime_env_toml_block(runtime_env)
    content += _pre_agent_setup_healthcheck_toml_block(pre_agent_setup)

    if mcp_servers:
        for srv in mcp_servers:
            content += "\n[[environment.mcp_servers]]\n"
            for key, val in srv.items():
                if not isinstance(key, str):
                    raise TypeError("MCP TOML keys must be strings")
                content += f"{_toml_quote(key)} = {_toml_value(val)}\n"

    tomllib.loads(content)
    (task_dir / "task.toml").write_text(content, encoding="utf-8")


def _toml_quote(value: str) -> str:
    """Return a TOML-compatible quoted string."""
    return toml_quote(value)


def _toml_value(value: Any) -> str:
    """Serialize the documented MCP TOML scalar and string-list values."""

    if isinstance(value, str):
        return _toml_quote(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "[" + ", ".join(_toml_quote(item) for item in value) + "]"
    raise TypeError("MCP TOML values must be strings or lists of strings")


def _runtime_env_toml_block(runtime_env: dict[str, str] | None) -> str:
    runtime_env = {**(runtime_env or {}), **_EVALUATOR_MANAGED_RUNTIME_ENV}
    lines = ["", "[environment.env]"]
    for key in sorted(runtime_env):
        lines.append(f"{_toml_quote(key)} = {_toml_quote(runtime_env[key])}")
    return "\n".join(lines) + "\n"


def _pre_agent_setup_command(pre_agent_setup: list[str] | None) -> str:
    commands = [cmd.strip() for cmd in (pre_agent_setup or []) if cmd and cmd.strip()]
    if not commands:
        return ""
    script = "set -euo pipefail\n" + "\n".join(commands)
    return "bash -lc " + shlex.quote(script)


def _pre_agent_setup_healthcheck_toml_block(pre_agent_setup: list[str] | None) -> str:
    command = _pre_agent_setup_command(pre_agent_setup)
    if not command:
        return ""
    return (
        "\n[environment.healthcheck]\n"
        f"command = {_toml_quote(command)}\n"
        "interval_sec = 5.0\n"
        "timeout_sec = 120.0\n"
        "retries = 1\n"
    )


def _write_entry_json(
    task_dir: Path,
    entry: dict[str, Any],
    has_skill: bool,
    *,
    workspace_mode: str = "isolated",
    workspace_skill_names: list[str] | None = None,
    grading_mode: str = "default",
    custom_grader: bool = False,
) -> None:
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    entry_with_flag = {
        **entry,
        "has_skill": has_skill,
        "skill_workspace_mode": workspace_mode,
        "workspace_skill_names": workspace_skill_names or [],
        "grading_mode": grading_mode,
        "custom_grader": custom_grader,
    }
    (tests_dir / "entry.json").write_text(json.dumps(entry_with_flag, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_test_sh(task_dir: Path, *, grading_mode: str, custom_grader: bool) -> None:
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    prefix = '#!/bin/bash\nset -euo pipefail\ntests_dir="${HARBOR_TESTS_DIR:-/tests}"\n'
    if grading_mode == "custom_only":
        script = prefix + 'python3 "${tests_dir}/custom_grader_runner.py" --mode custom_only\n'
    elif grading_mode == "default_plus_custom" and custom_grader:
        script = (
            prefix
            + 'python3 "${tests_dir}/eval.py"\n'
            + 'python3 "${tests_dir}/custom_grader_runner.py" --mode default_plus_custom\n'
        )
    else:
        script = prefix + 'python3 "${tests_dir}/eval.py"\n'
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(script, encoding="utf-8")
    test_sh.chmod(0o755)


def _copy_verifier(task_dir: Path) -> None:
    """Copy the standalone eval.py verifier into the task's tests/ directory."""
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    src = TEMPLATES_DIR / "eval.py"
    if src.exists():
        shutil.copy2(src, tests_dir / "eval.py")
    else:
        logger.warning("Verifier template not found at %s", src)
    lc = _EVAL_CORE_DIR / "log_converters.py"
    if lc.exists():
        shutil.copy2(lc, tests_dir / "log_converters.py")
    else:
        logger.warning("log_converters helper not found at %s", lc)


def _has_symlink_component(path: Path, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if _path_is_link_or_reparse(current):
            return True
    return False


def _copy_custom_grader(
    task_dir: Path,
    skill_path: Path,
    grading_mode: str,
    *,
    evals_dir: Path | None = None,
) -> bool:
    """Copy user custom grader support into a generated task, if configured."""
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    runner_src = TEMPLATES_DIR / "custom_grader_runner.py"
    if runner_src.exists():
        shutil.copy2(runner_src, tests_dir / "custom_grader_runner.py")

    evals_dir = evals_dir or skill_path / "evals"
    grader_candidates = [
        (evals_dir / "grader.py", tests_dir / "grader.py"),
        (evals_dir / "grader.sh", tests_dir / "grader.sh"),
        (evals_dir / "tests" / "grader.py", tests_dir / "grader.py"),
        (evals_dir / "tests" / "grader.sh", tests_dir / "grader.sh"),
    ]
    for grader, destination in grader_candidates:
        if _has_symlink_component(grader, evals_dir):
            raise ValueError(f"custom grader must be a non-symlinked regular file contained under evals/: {grader}")
        if not grader.exists():
            continue
        if not grader.is_file():
            raise ValueError(f"custom grader must be a non-symlinked regular file contained under evals/: {grader}")

        resolved_evals = evals_dir.resolve(strict=True)
        resolved_grader = grader.resolve(strict=True)
        if not resolved_grader.is_relative_to(resolved_evals):
            raise ValueError(f"custom grader must be a non-symlinked regular file contained under evals/: {grader}")

        copy_file_secure(grader, destination, allowed_root=evals_dir)
        if destination.suffix == ".sh":
            destination.chmod(0o755)
        return True

    if grading_mode == "custom_only":
        raise FileNotFoundError("grading.mode=custom_only requires evals/grader.py or evals/grader.sh")
    return False


def _collect_txt_deps(skills_dir: Path, filename: str) -> list[str]:
    """Collect non-comment lines from all instances of ``filename`` inside skill dirs."""
    deps: list[str] = []
    for dep_file in skills_dir.rglob(filename):
        for line in dep_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(line)
    return sorted(set(deps))


def _validate_custom_dockerfile(path: Path) -> str | None:
    """Basic validation. Returns error message or None if OK."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Cannot read Dockerfile: {e}"
    if path.stat().st_size > 20_000:
        return "Dockerfile exceeds 20KB limit"
    lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines or not lines[0].upper().startswith("FROM"):
        return "Dockerfile must start with a FROM instruction"
    return None


_DOCKER_HEREDOC_RE = re.compile(
    r"<<(?P<strip>-?)(?:(?P<quote>['\"])(?P<quoted>[^'\"\s]+)(?P=quote)|(?P<bare>[^\s'\"<>]+))(?=\s|$)"
)


def _dockerfile_heredocs(line: str) -> list[tuple[str, bool]]:
    """Return heredoc delimiters introduced by one Dockerfile instruction."""
    heredocs: list[tuple[str, bool]] = []
    quote: str | None = None
    index = 0
    while index < len(line):
        character = line[index]
        if quote is not None:
            if character == quote:
                quote = None
            elif character == "\\" and quote == '"':
                index += 1
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            index += 1
            continue
        if character == "\\":
            index += 2
            continue
        if character == "<" and (index == 0 or line[index - 1].isspace()):
            match = _DOCKER_HEREDOC_RE.match(line, index)
            if match is not None:
                delimiter = match.group("quoted") or match.group("bare")
                heredocs.append((delimiter, match.group("strip") == "-"))
                index = match.end()
                continue
        index += 1
    return heredocs


def _dockerfile_logical_lines_from_content(content: str, *, source: str = "Dockerfile") -> list[str]:
    """Return Dockerfile instructions after applying its escape directive."""
    raw_lines = content.splitlines()
    escape_char = "\\"
    for raw_line in raw_lines:
        directive = re.match(r"^\s*#\s*escape\s*=\s*([`\\])\s*$", raw_line, flags=re.IGNORECASE)
        if directive:
            escape_char = directive.group(1)
            break
        if raw_line.strip() and not raw_line.lstrip().startswith("#"):
            break

    logical_lines: list[str] = []
    pending = ""
    active_heredocs: list[tuple[str, bool]] = []
    for raw_line in raw_lines:
        if active_heredocs:
            delimiter, strip_tabs = active_heredocs[0]
            candidate = raw_line.lstrip("\t") if strip_tabs else raw_line
            if candidate == delimiter:
                active_heredocs.pop(0)
            continue
        line = raw_line.strip()
        if not line or (line.startswith("#") and not pending):
            continue
        pending = f"{pending} {line}".strip()
        trailing_escapes = len(pending) - len(pending.rstrip(escape_char))
        if trailing_escapes % 2 == 1:
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        active_heredocs.extend(_dockerfile_heredocs(pending))
        pending = ""
    if pending:
        logical_lines.append(pending)
        active_heredocs.extend(_dockerfile_heredocs(pending))
    if active_heredocs:
        raise ValueError(f"Cannot safely parse unterminated Dockerfile heredoc in {source}")
    return logical_lines


def _dockerfile_logical_lines(path: Path) -> list[str]:
    return _dockerfile_logical_lines_from_content(path.read_text(encoding="utf-8"), source=str(path))


def _dockerfile_final_explicit_user(content: str) -> str | None:
    """Return the statically known USER inherited by the final stage."""
    user: str | None = None
    stage_users: dict[str, str | None] = {}
    current_stage_name: str | None = None
    current_stage_index = -1

    def _save_current_stage() -> None:
        if current_stage_index < 0:
            return
        stage_users[str(current_stage_index)] = user
        if current_stage_name is not None:
            stage_users[current_stage_name.casefold()] = user

    for line in _dockerfile_logical_lines_from_content(content):
        parts = line.split(None, 1)
        if not parts:
            continue
        instruction = parts[0].upper()
        if instruction == "FROM" and len(parts) == 2:
            _save_current_stage()
            try:
                from_parts = shlex.split(parts[1])
            except ValueError:
                from_parts = parts[1].split()
            non_options = [part for part in from_parts if not part.startswith("--")]
            base = non_options[0] if non_options else ""
            alias = next(
                (non_options[index + 1] for index, value in enumerate(non_options[:-1]) if value.upper() == "AS"),
                None,
            )
            user = stage_users.get(base.casefold())
            current_stage_index += 1
            current_stage_name = alias
        elif instruction == "USER" and len(parts) == 2:
            user = parts[1].strip()
    return user


def _replace_single_from_image(content: str, base_image: str) -> tuple[str, str]:
    """Replace one Dockerfile stage image while preserving options and its alias."""
    from_lines = [
        line for line in _dockerfile_logical_lines_from_content(content) if line.split(None, 1)[0].upper() == "FROM"
    ]
    if len(from_lines) != 1:
        raise ValueError(
            "Custom Dockerfile rebase mode requires exactly one FROM instruction; "
            "use preserve mode for multi-stage Dockerfiles"
        )
    match = re.search(r"^(?P<prefix>[ \t]*FROM[ \t]+)(?P<body>[^\r\n]+)", content, flags=re.IGNORECASE | re.MULTILINE)
    if match is None:
        raise ValueError("Custom Dockerfile rebase mode could not locate its FROM instruction safely")
    body = match.group("body")
    if body.rstrip().endswith(("\\", "`")):
        raise ValueError(
            "Custom Dockerfile rebase mode does not support a continued FROM instruction; use preserve mode"
        )
    tokens = list(re.finditer(r'"[^"\r\n]*"|\'[^\'\r\n]*\'|\S+', body))
    image_token = next((token for token in tokens if not token.group(0).startswith("--")), None)
    if image_token is None:
        raise ValueError("Custom Dockerfile FROM instruction does not contain an image")
    body_start = match.start("body")
    image_start = body_start + image_token.start()
    image_end = body_start + image_token.end()
    return content[:image_start] + base_image + content[image_end:], from_lines[0]


def _dockerfile_instruction_sources(line: str) -> list[str]:
    """Return build-context sources for one COPY/ADD instruction."""
    parts = line.split(None, 1)
    if len(parts) != 2 or parts[0].upper() not in {"COPY", "ADD"}:
        return []
    payload = parts[1].strip()
    from_stage = False
    while payload.startswith("--"):
        option_parts = payload.split(None, 1)
        option = option_parts[0]
        if len(option_parts) == 1:
            raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD options: {line}")
        payload = option_parts[1].lstrip()
        if option == "--from":
            stage_parts = payload.split(None, 1)
            from_stage = True
            if len(stage_parts) == 1:
                raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD --from option: {line}")
            payload = stage_parts[1].lstrip()
        elif option.startswith("--from="):
            from_stage = True
    if from_stage or not payload:
        return []
    if _docker_expression_has_literal_dollar(payload):
        raise ValueError(f"Cannot safely parse quoted or escaped Dockerfile COPY/ADD variable: {line}")
    if re.match(r'^\[\s*"', payload):
        try:
            values = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse custom Dockerfile COPY/ADD JSON form: {line}") from exc
        if isinstance(values, list) and len(values) >= 2 and all(isinstance(value, str) for value in values):
            return values[:-1]
        raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD JSON form: {line}")
    try:
        values = shlex.split(payload)
    except ValueError as exc:
        raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD form: {line}") from exc
    if len(values) < 2:
        raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD form: {line}")
    sources = values[:-1]
    heredoc_sources = [source for source in sources if source.startswith("<<")]
    malformed_heredocs = [source for source in heredoc_sources if _DOCKER_HEREDOC_RE.fullmatch(source) is None]
    if malformed_heredocs:
        raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD heredoc: {line}")
    return [source for source in sources if source not in heredoc_sources]


def _dockerfile_named_context_sources(line: str) -> list[tuple[str, str]]:
    """Return ``(--from context, source)`` pairs for one COPY/ADD."""
    parts = line.split(None, 1)
    if len(parts) != 2 or parts[0].upper() not in {"COPY", "ADD"}:
        return []
    payload = parts[1].strip()
    from_context: str | None = None
    while payload.startswith("--"):
        option_parts = payload.split(None, 1)
        option = option_parts[0]
        if len(option_parts) == 1:
            raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD options: {line}")
        payload = option_parts[1].lstrip()
        if option == "--from":
            context_parts = payload.split(None, 1)
            if len(context_parts) == 1:
                raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD --from option: {line}")
            from_context = context_parts[0]
            payload = context_parts[1].lstrip()
        elif option.startswith("--from="):
            from_context = option.partition("=")[2]
    if from_context is None or not payload:
        return []
    if "$" in from_context or _docker_expression_has_literal_dollar(payload):
        raise ValueError(f"Cannot safely parse dynamic Dockerfile named-context source: {line}")
    if re.match(r'^\[\s*"', payload):
        try:
            values = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse custom Dockerfile COPY/ADD JSON form: {line}") from exc
        if not (isinstance(values, list) and len(values) >= 2 and all(isinstance(value, str) for value in values)):
            raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD JSON form: {line}")
        sources = values[:-1]
    else:
        try:
            values = shlex.split(payload)
        except ValueError as exc:
            raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD form: {line}") from exc
        if len(values) < 2:
            raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD form: {line}")
        sources = values[:-1]
    heredoc_sources = [source for source in sources if source.startswith("<<")]
    malformed_heredocs = [source for source in heredoc_sources if _DOCKER_HEREDOC_RE.fullmatch(source) is None]
    if malformed_heredocs:
        raise ValueError(f"Cannot safely parse custom Dockerfile COPY/ADD heredoc: {line}")
    return [(from_context, source) for source in sources if source not in heredoc_sources]


def _dockerfile_named_build_context_sources(path: Path) -> list[tuple[str, str]]:
    """Return statically named build-context sources from a Dockerfile."""
    sources: list[tuple[str, str]] = []
    for line in _dockerfile_logical_lines(path):
        parts = line.split(None, 1)
        effective_line = parts[1].strip() if len(parts) == 2 and parts[0].upper() == "ONBUILD" else line
        sources.extend(_dockerfile_named_context_sources(effective_line))
    return sources


def _dockerfile_build_context_sources(path: Path) -> list[str]:
    """Return COPY/ADD build-context sources from a Dockerfile."""
    sources: list[str] = []
    for line in _dockerfile_logical_lines(path):
        parts = line.split(None, 1)
        effective_line = parts[1].strip() if len(parts) == 2 and parts[0].upper() == "ONBUILD" else line
        sources.extend(_dockerfile_instruction_sources(effective_line))
    return sources


def _normalize_docker_context_source(source: str) -> str:
    """Normalize a COPY/ADD source using Docker build-context traversal rules."""
    normalized = posixpath.normpath(source.replace("\\", "/")).lstrip("/")
    while normalized.startswith("../"):
        normalized = normalized[3:]
    if normalized == "..":
        return "."
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


_DOCKER_VARIABLE_RE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")


def _docker_expression_has_literal_dollar(value: str) -> bool:
    """Return whether Docker keeps a dollar literal that shlex would erase."""
    return "$$" in value or re.search(r"\\+\$", value) is not None or re.search(r"'[^']*\$[^']*'", value) is not None


def _resolve_docker_source_variables(source: str, defaults: dict[str, str | None]) -> tuple[str, bool]:
    """Resolve known static Docker variables and report any unresolved reference."""
    resolved = source
    for _ in range(len(defaults) + 1):
        changed = False

        def _replace(match: re.Match[str]) -> str:
            nonlocal changed
            name = match.group("braced") or match.group("plain")
            if name not in defaults or defaults[name] is None:
                return match.group(0)
            changed = True
            return defaults[name]

        updated = _DOCKER_VARIABLE_RE.sub(_replace, resolved)
        resolved = updated
        if not changed:
            break
    return resolved, "$" in resolved


def _dockerfile_resolved_build_context_sources(
    path: Path,
    *,
    build_arg_overrides: dict[str, str | None] | None = None,
) -> list[tuple[str, str, bool]]:
    """Resolve COPY/ADD sources using ARG/ENV values visible at each instruction."""
    global_args: dict[str, str | None] = {}
    stage_args: dict[str, str | None] = {}
    stage_env: dict[str, str | None] = {}
    stage_onbuild: list[str] = []
    saved_stages: dict[str, tuple[dict[str, str | None], dict[str, str | None], list[str]]] = {}
    current_stage_name: str | None = None
    current_stage_index = -1
    in_stage = False
    resolved_sources: list[tuple[str, str, bool]] = []
    build_arg_overrides = build_arg_overrides or {}

    def _apply_arg(payload: str, *, stage_scope: bool) -> None:
        name, separator, value = payload.partition("=")
        name = name.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(f"Cannot safely parse custom Dockerfile ARG: {payload}")
        target = stage_args if stage_scope else global_args
        visible = {**stage_args, **stage_env} if stage_scope else dict(global_args)
        if name in build_arg_overrides:
            target[name] = build_arg_overrides[name]
        elif separator:
            raw_value = value.strip()
            if _docker_expression_has_literal_dollar(raw_value):
                raise ValueError(f"Cannot safely parse quoted or escaped Dockerfile ARG variable: {payload}")
            resolved, unresolved = _resolve_docker_source_variables(raw_value.strip("\"'"), visible)
            target[name] = None if unresolved else resolved
        elif stage_scope and name in target:
            return
        elif stage_scope and name in global_args:
            target[name] = global_args[name]
        else:
            target[name] = None

    def _apply_env(payload: str) -> None:
        if _docker_expression_has_literal_dollar(payload):
            raise ValueError(f"Cannot safely parse quoted or escaped Dockerfile ENV variable: {payload}")
        try:
            assignments = shlex.split(payload)
        except ValueError as exc:
            raise ValueError(f"Cannot safely parse custom Dockerfile ENV: {payload}") from exc
        pairs: list[tuple[str, str]] = []
        if assignments and all("=" in assignment for assignment in assignments):
            pairs = [tuple(assignment.split("=", 1)) for assignment in assignments]
        elif len(assignments) >= 2:
            pairs = [(assignments[0], " ".join(assignments[1:]))]
        if not pairs:
            raise ValueError(f"Cannot safely parse custom Dockerfile ENV: {payload}")
        visible = {**stage_args, **stage_env}
        updates: dict[str, str | None] = {}
        for name, value in pairs:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError(f"Cannot safely parse custom Dockerfile ENV: {payload}")
            resolved, unresolved = _resolve_docker_source_variables(value, visible)
            updates[name] = None if unresolved else resolved
        stage_env.update(updates)

    def _append_context_sources(line: str) -> None:
        visible = {**stage_args, **stage_env}
        for source in _dockerfile_instruction_sources(line):
            resolved, unresolved = _resolve_docker_source_variables(source, visible)
            resolved_sources.append((source, resolved, unresolved))

    def _run_onbuild_trigger(trigger: str) -> None:
        trigger_parts = trigger.split(None, 1)
        if len(trigger_parts) != 2:
            raise ValueError(f"Cannot safely parse custom Dockerfile ONBUILD trigger: {trigger}")
        trigger_instruction, trigger_payload = trigger_parts[0].upper(), trigger_parts[1].strip()
        if trigger_instruction == "ARG":
            _apply_arg(trigger_payload, stage_scope=True)
        elif trigger_instruction == "ENV":
            _apply_env(trigger_payload)
        elif trigger_instruction in {"COPY", "ADD"}:
            _append_context_sources(trigger)
        elif trigger_instruction in {"FROM", "MAINTAINER", "ONBUILD"}:
            raise ValueError(f"Cannot safely parse unsupported Dockerfile ONBUILD trigger: {trigger}")

    for line in _dockerfile_logical_lines(path):
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        instruction, payload = parts[0].upper(), parts[1].strip()

        if instruction == "FROM":
            if in_stage:
                saved_stage = (dict(stage_args), dict(stage_env), list(stage_onbuild))
                saved_stages[str(current_stage_index)] = saved_stage
                if current_stage_name is not None:
                    saved_stages[current_stage_name.casefold()] = saved_stage
            try:
                from_parts = shlex.split(payload)
            except ValueError:
                from_parts = payload.split()
            non_options = [part for part in from_parts if not part.startswith("--")]
            raw_base = non_options[0] if non_options else ""
            base, base_unresolved = _resolve_docker_source_variables(raw_base, global_args)
            alias = next(
                (non_options[index + 1] for index, value in enumerate(non_options[:-1]) if value.upper() == "AS"),
                None,
            )
            inherited_args, inherited_env, inherited_onbuild = (
                saved_stages.get(base.casefold(), ({}, {}, [])) if not base_unresolved else ({}, {}, [])
            )
            stage_args = dict(inherited_args)
            stage_env = dict(inherited_env)
            stage_onbuild = []
            current_stage_name = alias
            current_stage_index += 1
            in_stage = True
            for trigger in inherited_onbuild:
                _run_onbuild_trigger(trigger)
            continue

        if instruction == "ARG":
            _apply_arg(payload, stage_scope=in_stage)
            continue

        if instruction == "ENV" and in_stage:
            _apply_env(payload)
            continue

        if instruction == "ONBUILD" and in_stage:
            stage_onbuild.append(payload)
            continue

        if instruction not in {"COPY", "ADD"}:
            continue
        _append_context_sources(line)

    return resolved_sources


def _compose_build_arg_overrides(env_dir: Path, dockerfile: Path) -> dict[str, str | None]:
    """Return literal build-arg overrides for Compose services that build *dockerfile*."""
    import yaml

    compose_files = [env_dir / name for name in ("docker-compose.yaml", "docker-compose.yml")]
    compose_files = [compose for compose in compose_files if compose.is_file()]
    if not compose_files:
        return {}

    overrides: dict[str, str | None] = {}
    for compose in compose_files:
        try:
            content = yaml.safe_load(compose.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Cannot safely resolve custom Docker Compose build args: {exc}") from exc
        services = content.get("services", {}) if isinstance(content, dict) else {}
        if not isinstance(services, dict):
            continue
        for service_name, service in services.items():
            if service_name != "main" or not isinstance(service, dict):
                continue
            build = service.get("build")
            if not isinstance(build, dict):
                continue
            context = build.get("context", ".")
            dockerfile_name = build.get("dockerfile", "Dockerfile")
            if not isinstance(context, str) or not isinstance(dockerfile_name, str):
                continue
            candidate = (compose.parent / context / dockerfile_name).resolve()
            if candidate != dockerfile.resolve():
                continue
            args = build.get("args", {})
            if isinstance(args, list):
                items = [item.partition("=") if isinstance(item, str) else ("", "", "") for item in args]
                parsed = {name: value if separator else None for name, separator, value in items if name}
            elif isinstance(args, dict):
                parsed = {str(name): None if value is None else str(value) for name, value in args.items()}
            else:
                raise ValueError("Cannot safely resolve custom Docker Compose build args")
            for name, value in parsed.items():
                if value is not None and "$" in value:
                    value = None
                if name in overrides and overrides[name] != value:
                    overrides[name] = None
                else:
                    overrides[name] = value
    return overrides


def _ensure_empty_docker_input_compatibility(
    context_dir: Path,
    dockerfile: Path,
    *,
    has_input: bool,
    context_is_reserved_input: bool = False,
    build_arg_overrides: dict[str, str | None] | None = None,
) -> None:
    """Keep directory-level ``COPY input/`` valid for one Docker build context."""
    resolved_sources: list[str] = []
    unresolved_sources: list[str] = []
    for source, resolved, unresolved in _dockerfile_resolved_build_context_sources(
        dockerfile,
        build_arg_overrides=build_arg_overrides or {},
    ):
        if unresolved:
            unresolved_sources.append(source)
        else:
            resolved_sources.append(resolved)
    for source in _dockerfile_main_mount_context_sources(dockerfile):
        if "$" in source:
            unresolved_sources.append(source)
        else:
            resolved_sources.append(source)
    normalized_sources = [_normalize_docker_context_source(source) for source in resolved_sources]
    for source in normalized_sources:
        first_component = source.split("/", 1)[0]
        if first_component.casefold() == "input" and first_component != "input":
            raise ValueError("Custom Dockerfile input sources must use the canonical lowercase path 'input/'")
    if has_input:
        return
    if unresolved_sources:
        raise ValueError(
            "Custom Dockerfile uses an ambiguous input source for an eval entry with no staged input files; "
            "use a literal path or a static ARG/ENV default"
        )
    if any(source.casefold().startswith("input/") for source in normalized_sources):
        raise ValueError("Custom Dockerfile copies a specific input path, but this eval entry stages no input files")
    if context_is_reserved_input:
        if any(source not in {"", "."} for source in normalized_sources):
            raise ValueError(
                "Custom Dockerfile copies a specific path from the input build context, "
                "but this eval entry stages no input files"
            )
        return
    # External base images can carry ONBUILD COPY input/ triggers that are not
    # visible in the authored Dockerfile. An empty reserved directory keeps
    # those builds valid without exposing any fixture bytes.
    (context_dir / "input").mkdir(parents=True, exist_ok=True)


def _ensure_empty_custom_docker_input_compatibility(env_dir: Path, dockerfile: Path, *, has_input: bool) -> None:
    """Keep directory-level ``COPY input/`` valid for the managed main build."""
    _ensure_empty_docker_input_compatibility(
        env_dir,
        dockerfile,
        has_input=has_input,
        build_arg_overrides=_compose_build_arg_overrides(env_dir, dockerfile),
    )


def _literal_compose_build_args(build: dict[str, Any]) -> dict[str, str | None]:
    raw_args = build.get("args", {})
    if isinstance(raw_args, list):
        items = [item.partition("=") if isinstance(item, str) else ("", "", "") for item in raw_args]
        parsed = {name: value if separator else None for name, separator, value in items if name}
    elif isinstance(raw_args, dict):
        parsed = {str(name): None if value is None else str(value) for name, value in raw_args.items()}
    else:
        raise ValueError("Cannot safely resolve custom Docker Compose build args")
    return {name: None if value is not None and "$" in value else value for name, value in parsed.items()}


def _compose_build_context_values(build: str | dict[str, Any]) -> list[str]:
    """Return validated primary and named build-context paths."""
    if isinstance(build, str):
        return [build]
    values = [str(build.get("context", "."))]
    additional = build.get("additional_contexts", {})
    if isinstance(additional, list):
        values.extend(item.split("=", 1)[1] for item in additional if isinstance(item, str) and "=" in item)
    elif isinstance(additional, dict):
        values.extend(str(value) for value in additional.values())
    return values


def _compose_additional_build_contexts(build: str | dict[str, Any]) -> dict[str, str]:
    """Return named Compose build contexts after structural validation."""
    if not isinstance(build, dict):
        return {}
    additional = build.get("additional_contexts", {})
    if isinstance(additional, list):
        return {
            name: value
            for item in additional
            if isinstance(item, str) and "=" in item
            for name, _separator, value in [item.partition("=")]
            if name
        }
    if isinstance(additional, dict):
        return {str(name): str(value) for name, value in additional.items()}
    return {}


def _dockerfile_mount_context_sources(path: Path) -> list[tuple[str | None, str]]:
    """Return BuildKit RUN-mount contexts and their source paths."""
    sources: list[tuple[str | None, str]] = []
    for line in _dockerfile_logical_lines(path):
        parts = line.split(None, 1)
        effective_line = parts[1].strip() if len(parts) == 2 and parts[0].upper() == "ONBUILD" else line
        effective_parts = effective_line.split(None, 1)
        if len(effective_parts) != 2 or effective_parts[0].upper() != "RUN":
            continue
        try:
            tokens = shlex.split(effective_parts[1])
        except ValueError as exc:
            raise ValueError(f"Cannot safely parse custom Dockerfile RUN options: {effective_line}") from exc
        for token in tokens:
            if not token.startswith("--"):
                break
            if not token.startswith("--mount="):
                continue
            mount = token.partition("=")[2]
            options = dict(item.partition("=")[::2] for item in mount.split(",") if "=" in item)
            source = options.get("source", options.get("src", "."))
            sources.append((options.get("from"), source))
    return sources


def _dockerfile_main_mount_context_sources(path: Path) -> list[str]:
    """Return source paths for BuildKit mounts from the primary context."""
    return [source for context, source in _dockerfile_mount_context_sources(path) if context is None]


def _dockerfile_named_mount_context_sources(path: Path) -> list[tuple[str, str]]:
    """Return source paths for BuildKit mounts from named contexts."""
    return [(context, source) for context, source in _dockerfile_mount_context_sources(path) if context is not None]


def _validate_empty_named_input_context_sources(
    dockerfile: Path,
    reserved_named_contexts: set[str],
    has_input: bool,
) -> None:
    """Reject specific reads from an explicitly empty named input context."""
    if has_input or not reserved_named_contexts:
        return
    named_sources = [
        *_dockerfile_named_build_context_sources(dockerfile),
        *_dockerfile_named_mount_context_sources(dockerfile),
    ]
    for context, source in named_sources:
        if context not in reserved_named_contexts:
            continue
        if "$" in source or _normalize_docker_context_source(source) not in {"", "."}:
            raise ValueError(
                "Custom Dockerfile reads a specific path from an input context, "
                "but this eval entry stages no input files"
            )


def _ensure_empty_compose_input_compatibility(env_dir: Path, *, has_input: bool) -> None:
    """Apply explicit-empty input compatibility to every validated Compose build."""
    compose_path = _custom_compose_path(env_dir)
    if compose_path is None:
        return
    import yaml

    content = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = content.get("services", {}) if isinstance(content, dict) else {}
    recipes: list[tuple[Path, Path | None, str | None, dict[str, str | None], bool, set[str]]] = []
    for service in services.values():
        if not isinstance(service, dict) or "build" not in service:
            continue
        build = service["build"]
        if not has_input:
            for value in _compose_build_context_values(build):
                normalized = _normalize_docker_context_source(value)
                first_component = normalized.split("/", 1)[0]
                if first_component.casefold() == "input" and first_component != "input":
                    raise ValueError(
                        "Custom Docker Compose input contexts must use the canonical lowercase path 'input'"
                    )
                if normalized.casefold() == "input":
                    (env_dir / "input").mkdir(parents=True, exist_ok=True)
                elif normalized.casefold().startswith("input/"):
                    raise ValueError(
                        "Custom Docker Compose references a specific input context, but this eval entry stages no input files"
                    )
        if isinstance(build, str):
            context_dir = (env_dir / build).resolve()
            context_is_reserved_input = _normalize_docker_context_source(build).casefold() == "input"
            recipes.append((context_dir, context_dir / "Dockerfile", None, {}, context_is_reserved_input, set()))
            continue
        if not isinstance(build, dict):
            continue
        context_dir = (env_dir / str(build.get("context", "."))).resolve()
        inline = build.get("dockerfile_inline")
        dockerfile = None if inline is not None else context_dir / str(build.get("dockerfile", "Dockerfile"))
        context_is_reserved_input = (
            _normalize_docker_context_source(str(build.get("context", "."))).casefold() == "input"
        )
        reserved_named_contexts = {
            name
            for name, value in _compose_additional_build_contexts(build).items()
            if _normalize_docker_context_source(value).casefold() == "input"
        }
        recipes.append(
            (
                context_dir,
                dockerfile,
                inline,
                _literal_compose_build_args(build),
                context_is_reserved_input,
                reserved_named_contexts,
            )
        )

    initial_input = {
        context_dir: (
            has_input
            if _paths_equivalent(context_dir, env_dir) or context_is_reserved_input
            else (context_dir / "input").is_dir()
        )
        for context_dir, _dockerfile, _inline, _args, context_is_reserved_input, _named in recipes
    }
    for context_dir, dockerfile, inline, build_args, context_is_reserved_input, reserved_named_contexts in recipes:
        if (
            dockerfile is not None
            and _paths_equivalent(context_dir, env_dir)
            and _paths_equivalent(dockerfile, env_dir / "Dockerfile")
        ):
            continue
        if inline is not None:
            with tempfile.TemporaryDirectory(prefix="skillevaluator-compose-inline-") as temporary:
                parsed_dockerfile = Path(temporary) / "Dockerfile"
                parsed_dockerfile.write_text(str(inline).replace("$$", "$"), encoding="utf-8")
                _ensure_empty_docker_input_compatibility(
                    context_dir,
                    parsed_dockerfile,
                    has_input=initial_input[context_dir],
                    context_is_reserved_input=context_is_reserved_input,
                    build_arg_overrides=build_args,
                )
                _validate_empty_named_input_context_sources(parsed_dockerfile, reserved_named_contexts, has_input)
            continue
        if dockerfile is None or not dockerfile.is_file():
            raise ValueError(f"Custom Docker Compose build Dockerfile does not exist: {dockerfile}")
        _ensure_empty_docker_input_compatibility(
            context_dir,
            dockerfile,
            has_input=initial_input[context_dir],
            context_is_reserved_input=context_is_reserved_input,
            build_arg_overrides=build_args,
        )
        _validate_empty_named_input_context_sources(dockerfile, reserved_named_contexts, has_input)


def _compose_strings(value: Any) -> list[str]:
    strings: list[str] = []
    seen_containers: set[int] = set()
    nodes = 0

    def _visit(node: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if depth > 64 or nodes > 10_000:
            raise ValueError("Custom Docker Compose structure exceeds safe traversal limits")
        if isinstance(node, str):
            strings.append(node)
            return
        if not isinstance(node, (dict, list)):
            return
        identity = id(node)
        if identity in seen_containers:
            raise ValueError("Custom Docker Compose aliases or cycles are not supported")
        seen_containers.add(identity)
        children = node.values() if isinstance(node, dict) else node
        for child in children:
            _visit(child, depth + 1)

    _visit(value, 0)
    return strings


def _validate_compose_interpolation(content: dict[str, Any], allowed_env: set[str]) -> None:
    referenced = {
        match.group(1) or match.group(2)
        for value in _compose_strings(content)
        for match in _COMPOSE_ENV_RE.finditer(value)
    }
    undeclared = sorted(referenced - allowed_env)
    if undeclared:
        raise ValueError(
            f"Custom Docker Compose uses undeclared interpolation variables: {', '.join(undeclared)}; "
            "declare each variable in harbor.runtime_env"
        )


def _is_relative_compose_path(value: object, root: Path) -> bool:
    if not isinstance(value, str) or not value or "$" in value:
        return False
    if (
        Path(value).is_absolute()
        or _COMPOSE_WINDOWS_PATH_RE.match(value)
        or _COMPOSE_SSH_PATH_RE.match(value)
        or value.startswith(("\\", "~"))
        or "://" in value
    ):
        return False
    parts = Path(value.replace("\\", "/")).parts
    if ".." in parts:
        return False
    return _is_relative_to(root / value, root)


def _validate_compose_host_passthrough(value: object, *, allowed_env: set[str], field: str) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        passthrough = {str(name) for name, configured in value.items() if configured is None}
    elif isinstance(value, list):
        if any(not isinstance(item, str) for item in value):
            raise ValueError(f"Custom Docker Compose {field} list entries must be strings")
        passthrough = {item for item in value if "=" not in item}
    else:
        raise ValueError(f"Custom Docker Compose {field} must be a mapping or list")

    undeclared = sorted(passthrough - allowed_env)
    if undeclared:
        raise ValueError(
            f"Custom Docker Compose {field} passes undeclared host variables: {', '.join(undeclared)}; "
            "declare each variable in harbor.runtime_env"
        )


def _require_string_mapping_keys(value: dict[object, object], *, field: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"Custom Docker Compose {field} keys must be strings")


def _validate_compose_build(
    service_name: str,
    build: object,
    environment_dir: Path,
    *,
    allowed_env: set[str],
) -> None:
    if isinstance(build, str):
        if not _is_relative_compose_path(build, environment_dir):
            raise ValueError(f"Custom Docker Compose service '{service_name}' has an unsafe build context")
        return
    if not isinstance(build, dict):
        raise ValueError(f"Custom Docker Compose service '{service_name}' build must be a path or mapping")
    _require_string_mapping_keys(build, field=f"service '{service_name}' build")

    unsupported_keys = sorted(set(build) - _COMPOSE_ALLOWED_BUILD_KEYS)
    if unsupported_keys:
        raise ValueError(
            f"Custom Docker Compose service '{service_name}' build cannot set: {', '.join(unsupported_keys)}"
        )

    context = build.get("context", ".")
    if not _is_relative_compose_path(context, environment_dir):
        raise ValueError(f"Custom Docker Compose service '{service_name}' has an unsafe build context")
    context_dir = environment_dir / str(context)

    dockerfile = build.get("dockerfile")
    inline = build.get("dockerfile_inline")
    if dockerfile is not None and inline is not None:
        raise ValueError(f"Custom Docker Compose service '{service_name}' cannot set both dockerfile forms")
    if dockerfile is not None and not _is_relative_compose_path(dockerfile, context_dir):
        raise ValueError(f"Custom Docker Compose service '{service_name}' has an unsafe build dockerfile")
    if inline is not None and (not isinstance(inline, str) or not inline.strip() or len(inline) > 20_000):
        raise ValueError(f"Custom Docker Compose service '{service_name}' has an invalid inline Dockerfile")
    if service_name == "main" and "target" in build:
        raise ValueError("Custom Docker Compose service 'main' cannot select an authored build target")

    additional_contexts = build.get("additional_contexts", {})
    if isinstance(additional_contexts, list):
        values = [
            item.split("=", 1)[1] if isinstance(item, str) and "=" in item else None for item in additional_contexts
        ]
    elif isinstance(additional_contexts, dict):
        _require_string_mapping_keys(additional_contexts, field=f"service '{service_name}' build additional_contexts")
        values = list(additional_contexts.values())
    else:
        raise ValueError(
            f"Custom Docker Compose service '{service_name}' build additional_contexts must be a mapping or list"
        )
    if any(not _is_relative_compose_path(value, environment_dir) for value in values):
        raise ValueError(f"Custom Docker Compose service '{service_name}' has unsafe build additional_contexts")
    _validate_compose_host_passthrough(
        build.get("args"),
        allowed_env=allowed_env,
        field=f"service '{service_name}' build.args",
    )


def _is_host_bind_volume(volume: object) -> bool:
    if isinstance(volume, dict):
        if any("$" in value for value in _compose_strings(volume)):
            return True
        volume_type = volume.get("type")
        if volume_type == "bind":
            return True
        if volume_type == "tmpfs":
            return "source" in volume
        if volume_type != "volume":
            return True
        source = volume.get("source")
        return source is not None and (not isinstance(source, str) or not _COMPOSE_NAMED_VOLUME_RE.fullmatch(source))
    if not isinstance(volume, str):
        return True
    if "$" in volume:
        return True
    if ":" not in volume:
        return False
    if _COMPOSE_WINDOWS_PATH_RE.match(volume):
        return True
    source = volume.split(":", 1)[0]
    return not bool(_COMPOSE_NAMED_VOLUME_RE.fullmatch(source))


def _validate_and_sanitize_custom_compose(
    compose_path: Path,
    *,
    allowed_env: set[str],
) -> None:
    """Reject Docker-host escape features and remove sidecar host ports.

    When Harbor runs multiple trials concurrently, each gets its own compose
    project.  Fixed host port mappings (e.g. ``"5432:5432"``) cause all but the
    first trial to fail with a port-already-in-use error.  Sidecar services
    don't need host ports — the main container reaches them via the compose
    network hostname (e.g. ``postgres:5432``).
    """
    import yaml

    try:
        content = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Custom Docker Compose file cannot be read safely: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError("Custom Docker Compose file must contain a top-level mapping")
    _require_string_mapping_keys(content, field="top-level")

    unsupported_top_level = sorted(set(content) - _COMPOSE_ALLOWED_TOP_LEVEL_KEYS)
    if unsupported_top_level:
        raise ValueError(f"Custom Docker Compose top-level cannot set: {', '.join(unsupported_top_level)}")

    services = content.get("services")
    if not isinstance(services, dict) or not services:
        raise ValueError("Custom Docker Compose services must be a non-empty mapping")

    changed = False
    for svc_name, service in services.items():
        if not isinstance(svc_name, str) or not isinstance(service, dict):
            raise ValueError("Custom Docker Compose services must map names to service mappings")
        _require_string_mapping_keys(service, field=f"service '{svc_name}'")
        if "build" in service and "image" in service:
            raise ValueError(f"Custom Docker Compose service '{svc_name}' cannot set image together with build")

        volumes = service.get("volumes", [])
        if not isinstance(volumes, list):
            raise ValueError(f"Custom Docker Compose service '{svc_name}' volumes must be a list")
        if any(_is_host_bind_volume(volume) for volume in volumes):
            raise ValueError(f"Custom Docker Compose service '{svc_name}' cannot use a host bind mount")

        allowed_service_keys = _COMPOSE_MAIN_ALLOWED_KEYS if svc_name == "main" else _COMPOSE_SIDECAR_ALLOWED_KEYS
        unsupported_keys = sorted(set(service) - allowed_service_keys)
        if unsupported_keys:
            if svc_name == "main":
                raise ValueError(
                    "Custom Docker Compose service 'main' may set only depends_on; "
                    f"unsupported: {', '.join(unsupported_keys)}"
                )
            raise ValueError(f"Custom Docker Compose service '{svc_name}' cannot set: {', '.join(unsupported_keys)}")
        if "build" in service:
            _validate_compose_build(
                svc_name,
                service["build"],
                compose_path.parent,
                allowed_env=allowed_env,
            )
        _validate_compose_host_passthrough(
            service.get("environment"),
            allowed_env=allowed_env,
            field=f"service '{svc_name}' environment",
        )

        if svc_name != "main" and "ports" in service:
            del service["ports"]
            changed = True
            logger.debug("Stripped host port mapping from sidecar service '%s'", svc_name)

    volumes = content.get("volumes", {})
    if not isinstance(volumes, dict):
        raise ValueError("Custom Docker Compose top-level volumes must be a mapping")
    for volume_name, volume in volumes.items():
        if not isinstance(volume_name, str):
            raise ValueError("Custom Docker Compose volume names must be strings")
        if volume is None:
            continue
        if not isinstance(volume, dict):
            raise ValueError(f"Custom Docker Compose volume '{volume_name}' must be a mapping")
        _require_string_mapping_keys(volume, field=f"volume '{volume_name}'")
        unsupported_keys = sorted(set(volume) - _COMPOSE_ALLOWED_VOLUME_KEYS)
        if unsupported_keys:
            raise ValueError(f"Custom Docker Compose volume '{volume_name}' cannot set: {', '.join(unsupported_keys)}")

    networks = content.get("networks", {})
    if not isinstance(networks, dict):
        raise ValueError("Custom Docker Compose top-level networks must be a mapping")
    for network_name, network in networks.items():
        if not isinstance(network_name, str):
            raise ValueError("Custom Docker Compose network names must be strings")
        if network is None:
            continue
        if not isinstance(network, dict):
            raise ValueError(f"Custom Docker Compose network '{network_name}' must be a mapping")
        _require_string_mapping_keys(network, field=f"network '{network_name}'")
        unsupported_keys = sorted(set(network) - _COMPOSE_ALLOWED_NETWORK_KEYS)
        if unsupported_keys:
            raise ValueError(
                f"Custom Docker Compose network '{network_name}' cannot set: {', '.join(unsupported_keys)}"
            )

    _validate_compose_interpolation(content, allowed_env)

    if changed:
        compose_path.write_text(yaml.dump(content, default_flow_style=False), encoding="utf-8")


def _custom_compose_path(environment_dir: Path) -> Path | None:
    candidates = [
        environment_dir / name
        for name in ("docker-compose.yaml", "docker-compose.yml")
        if (environment_dir / name).exists()
    ]
    if len(candidates) > 1:
        raise ValueError("Custom environment must not contain both docker-compose.yaml and docker-compose.yml")
    return candidates[0] if candidates else None


_VERIFIER_DEPS = "ragas~=0.4.0 langchain-community<0.4.2 openai>=1.0 anthropic>=0.40 boto3>=1.34 idna>=3.10,<4"
_VERIFIER_IMPORT_SMOKE = (
    "from ragas import SingleTurnSample; "
    "from ragas.llms.base import llm_factory; "
    "from ragas.messages import AIMessage, HumanMessage; "
    "from ragas.metrics.collections import AgentGoalAccuracyWithReference"
)
_MANAGED_PATH_SHELL = 'PATH="${PATH:+$PATH:}/usr/local/bin:/usr/bin:/bin"; export PATH; exec "$@"'


def _managed_run(command: str, *args: str) -> str:
    """Build an exec-form RUN that remains usable after an authored PATH override."""
    return "RUN " + json.dumps(["/bin/sh", "-c", _MANAGED_PATH_SHELL, "skillevaluator", command, *args])


def _verifier_install_lines(extra_requirements: Sequence[str] = ()) -> list[str]:
    """Install verifier and skill requirements in one resolver transaction."""
    requirements = [*_VERIFIER_DEPS.split(), *extra_requirements]
    return [
        _managed_run("python", "-m", "pip", "install", "--no-cache-dir", *requirements),
        _managed_run("python", "-c", _VERIFIER_IMPORT_SMOKE),
    ]


_WORKSPACE_SKILL_PATH = "COPY skills/ /workspace/skills/"
_AGENT_SKILL_PATHS = [
    "COPY skills/ /root/.claude/skills/",
    "COPY skills/ /root/.agents/skills/",
    "COPY skills/ /root/.config/opencode/skills/",
]
_RUNTIME_SKILL_DIRS = (
    "/etc/codex/skills",
    "/tmp/codex-home/skills",
    "/workspace/skills",
    "/workspace/.agents/skills",
    "/workspace/.claude/commands",
    "/workspace/.claude/skills",
    "/workspace/.cline/skills",
    "/workspace/.codex/skills",
    "/workspace/.config/goose/skills",
    "/workspace/.config/opencode/skills",
    "/workspace/.cursor/skills",
    "/workspace/.gemini/skills",
    "/workspace/.gemini/extensions",
    "/workspace/.opencode/skills",
    "/workspace/.qwen/skills",
    "/root/.claude/commands",
    "/root/.claude/skills",
    "/root/.cline/skills",
    "/root/.agents/skills",
    "/root/.codex/skills",
    "/root/.config/goose/skills",
    "/root/.config/opencode/skills",
    "/root/.gemini/extensions",
    "/root/.qwen/skills",
    "/logs/agent/sessions/skills",
    "/tmp/agent-home/sessions/skills",
)
_PROJECT_SKILL_RELATIVE_DIRS = (
    ".agents/skills",
    ".claude/commands",
    ".claude/skills",
    ".cline/skills",
    ".codex/skills",
    ".config/goose/skills",
    ".config/opencode/skills",
    ".cursor/skills",
    ".gemini/skills",
    ".gemini/extensions",
    ".opencode/skills",
    ".qwen/skills",
)
_RUNTIME_PROJECTION_ROOTS = ("/workspace/input", "/workspace/repo", *_RUNTIME_SKILL_DIRS)
_RUNTIME_PROJECTION_CASE_PATTERNS = (
    *(pattern for root in _RUNTIME_PROJECTION_ROOTS for pattern in (root, f"{root}/*")),
    *(pattern for relative in _PROJECT_SKILL_RELATIVE_DIRS for pattern in (f"*/{relative}", f"*/{relative}/*")),
)
_RUNTIME_DISCOVERY_ENV_NAMES = frozenset(
    {
        "CLAUDE_CODE_DISABLE_POLICY_SKILLS",
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        "GEMINI_CLI_HOME",
        "HOME",
        "OPENCODE_CONFIG_DIR",
        "USERPROFILE",
        "XDG_CONFIG_HOME",
    }
)
_RUNTIME_LOADER_ENV_NAMES = frozenset(
    {
        "BASH_ENV",
        "CLASSPATH",
        "DYLD_INSERT_LIBRARIES",
        "ENV",
        "GCONV_PATH",
        "JAVA_TOOL_OPTIONS",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "LOCPATH",
        "LUA_CPATH",
        "LUA_INIT",
        "LUA_PATH",
        "NLSPATH",
        "NODE_OPTIONS",
        "PERL5LIB",
        "PERL5OPT",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "RUBYLIB",
        "RUBYOPT",
        "ZDOTDIR",
        "_JAVA_OPTIONS",
    }
)
_RUNTIME_LOADER_ENV_PREFIXES = ("BASH_FUNC_",)
_RUNTIME_LOADER_ENV_RESET = "ENV " + " ".join(f'{name}=""' for name in sorted(_RUNTIME_LOADER_ENV_NAMES))
_RUNTIME_POLICY_ENV_RESET = 'ENV CLAUDE_CODE_DISABLE_POLICY_SKILLS="1"'
_EVALUATOR_MANAGED_RUNTIME_ENV = {
    **dict.fromkeys(_RUNTIME_LOADER_ENV_NAMES, ""),
    "CLAUDE_CODE_DISABLE_POLICY_SKILLS": "1",
}
_CLEAR_SKILL_ROOT_FUNCTION = (
    "clear_skill_root() { target=$1; "
    'if [ -d "$target" ] && [ ! -L "$target" ]; then '
    'for child in "$target"/* "$target"/.[!.]* "$target"/..?*; do '
    '[ -e "$child" ] || [ -L "$child" ] || continue; /bin/rm -rf -- "$child"; done; '
    'else /bin/rm -rf -- "$target"; fi; }; '
)
_PROJECT_SKILL_ANCESTOR_RESET = (
    "set -eu; current=$(/bin/pwd -P); while :; do "
    "for rel in "
    + " ".join(_PROJECT_SKILL_RELATIVE_DIRS)
    + '; do /bin/rm -rf -- "$current/$rel"; /bin/mkdir -p -- "$current/$rel"; done; '
    '[ "$current" = / ] && break; current=${current%/*}; [ -n "$current" ] || current=/; done'
)
_HOME_SKILL_RESET = (
    "set -eu; " + _CLEAR_SKILL_ROOT_FUNCTION + 'home=${HOME:-/root}; case "$home" in /*) ;; *) '
    "echo 'SkillEvaluator requires an absolute HOME for runtime skill isolation' >&2; exit 78;; esac; "
    "for rel in " + " ".join(_PROJECT_SKILL_RELATIVE_DIRS) + '; do clear_skill_root "$home/$rel"; done'
)
_PASSWD_HOME_SKILL_RESET = (
    "set -eu; " + _CLEAR_SKILL_ROOT_FUNCTION + "[ -r /etc/passwd ] || exit 0; "
    'while IFS=: read -r _ _ _ _ _ home _; do case "$home" in /*) ;; *) continue;; esac; '
    '[ -d "$home" ] || continue; '
    "for rel in "
    + " ".join(_PROJECT_SKILL_RELATIVE_DIRS)
    + '; do clear_skill_root "$home/$rel"; done; done < /etc/passwd'
)
_DISCOVERY_ENV_SKILL_RESET = (
    "set -eu; "
    + _CLEAR_SKILL_ROOT_FUNCTION
    + 'if [ -n "${CLAUDE_CONFIG_DIR:-}" ]; then case "$CLAUDE_CONFIG_DIR" in /*) ;; *) exit 78;; esac; '
    'clear_skill_root "$CLAUDE_CONFIG_DIR/skills"; clear_skill_root "$CLAUDE_CONFIG_DIR/commands"; fi; '
    'if [ -n "${CODEX_HOME:-}" ]; then case "$CODEX_HOME" in /*) ;; *) exit 78;; esac; '
    'clear_skill_root "$CODEX_HOME/skills"; fi; '
    'if [ -n "${GEMINI_CLI_HOME:-}" ]; then case "$GEMINI_CLI_HOME" in /*) ;; *) exit 78;; esac; '
    'clear_skill_root "$GEMINI_CLI_HOME/.gemini/skills"; '
    'clear_skill_root "$GEMINI_CLI_HOME/.gemini/extensions"; fi; '
    'if [ -n "${OPENCODE_CONFIG_DIR:-}" ]; then case "$OPENCODE_CONFIG_DIR" in /*) ;; *) exit 78;; esac; '
    'clear_skill_root "$OPENCODE_CONFIG_DIR/skills"; fi; '
    'if [ -n "${XDG_CONFIG_HOME:-}" ]; then case "$XDG_CONFIG_HOME" in /*) ;; *) exit 78;; esac; '
    'clear_skill_root "$XDG_CONFIG_HOME/opencode/skills"; fi'
)
_RUNTIME_PROJECTION_OVERLAP_PREFLIGHT = (
    'set -eu; check_path() { value=$1; case "$value" in ' + "|".join(_RUNTIME_PROJECTION_CASE_PATTERNS) + ") "
    "echo 'SkillEvaluator agent workdir/home overlaps an evaluator-controlled runtime projection' >&2; "
    'exit 78;; esac; }; check_control_path() { value=$1; check_path "$value"; '
    'if [ -d "$value" ]; then canonical=$(CDPATH= cd -P -- "$value" 2>/dev/null && /bin/pwd -P) || exit 78; '
    'check_path "$canonical"; elif [ -e "$value" ] || [ -L "$value" ]; then '
    "echo 'SkillEvaluator discovery home must resolve to a directory' >&2; exit 78; fi; }; "
    'current=$(/bin/pwd -P); check_path "$current"; '
    'home=${HOME:-/root}; case "$home" in /*) check_control_path "$home";; *) '
    "echo 'SkillEvaluator requires an absolute HOME for runtime skill isolation' >&2; exit 78;; esac; "
    'for controlled_home in "${CLAUDE_CONFIG_DIR:-}" "${CODEX_HOME:-}" "${GEMINI_CLI_HOME:-}" '
    '"${OPENCODE_CONFIG_DIR:-}" '
    '"${XDG_CONFIG_HOME:-}"; do [ -z "$controlled_home" ] || check_control_path "$controlled_home"; done; '
    "[ -r /etc/passwd ] || exit 0; while IFS=: read -r _ _ _ _ _ passwd_home _; do "
    'case "$passwd_home" in /*) '
    '[ -e "$passwd_home" ] && [ ! -d "$passwd_home" ] && continue; '
    'check_control_path "$passwd_home";; esac; done < /etc/passwd'
)
_RUNTIME_REPO_RESET_LINES = [
    'RUN ["/bin/rm", "-rf", "/workspace/repo"]',
    'RUN ["/bin/mkdir", "-p", "/workspace/repo"]',
]
_RUNTIME_ROOT_PREFLIGHT = "RUN " + json.dumps(
    [
        "/bin/sh",
        "-c",
        'PATH="${PATH:+$PATH:}/usr/local/bin:/usr/bin:/bin"; export PATH; '
        'if [ "$(id -u)" -ne 0 ]; then echo \'SkillEvaluator final runtime projection requires root; '
        "declare the Dockerfile final USER explicitly so it can be restored' >&2; exit 78; fi",
    ]
)


def _validated_agent_workdir(agent_workdir: str | None) -> str | None:
    """Return a normalized absolute container workdir or fail closed."""
    if agent_workdir is None:
        return None
    if not isinstance(agent_workdir, str) or not agent_workdir:
        raise ValueError("agent_workdir must be a non-empty absolute container path")
    normalized = posixpath.normpath(agent_workdir)
    typed_normalized = agent_workdir.rstrip("/") or "/"
    if (
        not normalized.startswith("/")
        or normalized != typed_normalized
        or re.fullmatch(r"/[A-Za-z0-9._/-]*", normalized) is None
    ):
        raise ValueError("agent_workdir must be a normalized absolute POSIX path")
    if any(normalized == root or normalized.startswith(f"{root}/") for root in _RUNTIME_PROJECTION_ROOTS):
        raise ValueError("agent_workdir must not be inside an evaluator-controlled runtime projection")
    parts = tuple(part for part in normalized.split("/") if part)
    discovery_parts = tuple(tuple(relative.split("/")) for relative in _PROJECT_SKILL_RELATIVE_DIRS)
    if any(
        parts[index : index + len(candidate)] == candidate
        for candidate in discovery_parts
        for index in range(len(parts) - len(candidate) + 1)
    ):
        raise ValueError("agent_workdir must not be inside an agent skill discovery directory")
    return normalized


def _validate_runtime_discovery_env(runtime_env: dict[str, str] | None) -> None:
    collisions = sorted(name for name in (runtime_env or {}) if name.upper() in _RUNTIME_DISCOVERY_ENV_NAMES)
    if collisions:
        raise ValueError(
            "harbor.runtime_env cannot override agent skill discovery location(s): " + ", ".join(collisions)
        )


def _validate_runtime_loader_env(runtime_env: dict[str, str] | None) -> None:
    collisions = sorted(
        name
        for name in (runtime_env or {})
        if name.upper() in _RUNTIME_LOADER_ENV_NAMES or name.upper().startswith(_RUNTIME_LOADER_ENV_PREFIXES)
    )
    if collisions:
        raise ValueError("harbor.runtime_env cannot override process loader environment: " + ", ".join(collisions))


def _runtime_skill_reset_lines(agent_workdir: str | None = None) -> list[str]:
    """Replace fixed, effective-home, and project-ancestor skill roots."""
    lines = [
        "RUN " + json.dumps(["/bin/rm", "-rf", *_RUNTIME_SKILL_DIRS]),
        "RUN " + json.dumps(["/bin/mkdir", "-p", *_RUNTIME_SKILL_DIRS]),
        "RUN " + json.dumps(["/bin/sh", "-c", _PROJECT_SKILL_ANCESTOR_RESET]),
        "RUN " + json.dumps(["/bin/sh", "-c", _HOME_SKILL_RESET]),
        "RUN " + json.dumps(["/bin/sh", "-c", _PASSWD_HOME_SKILL_RESET]),
        "RUN " + json.dumps(["/bin/sh", "-c", _DISCOVERY_ENV_SKILL_RESET]),
    ]
    validated_workdir = _validated_agent_workdir(agent_workdir)
    if validated_workdir is not None:
        lines.extend(
            [
                f"WORKDIR {validated_workdir}",
                "RUN " + json.dumps(["/bin/sh", "-c", _PROJECT_SKILL_ANCESTOR_RESET]),
            ]
        )
    return lines


def _runtime_copy_lines(
    agent_config_lines: list[str],
    include_input: bool,
    include_repo: bool = False,
    include_repo_linked_root: bool = False,
    agent_workdir: str | None = None,
) -> list[str]:
    _ = include_input
    lines = [
        _RUNTIME_ROOT_PREFLIGHT,
        *_runtime_skill_reset_lines(agent_workdir),
        *_RUNTIME_REPO_RESET_LINES,
        _WORKSPACE_SKILL_PATH,
        *_AGENT_SKILL_PATHS,
        *agent_config_lines,
    ]
    if include_repo:
        lines.append("COPY repo/ /workspace/repo/")
    if include_repo_linked_root:
        lines.append("COPY repo-linked-root/ /workspace/")
    return lines


def _append_task_input_projection(
    content: str,
    *,
    include_input: bool,
    agent_config_lines: list[str],
    restore_user: str | None = None,
) -> str:
    """Append the final evaluator-owned replacement of ``/workspace/input``."""
    separator = "" if content.endswith("\n") else "\n"
    lines = [
        "",
        "# SkillEvaluator: replace task input after all authored/base-image layers",
        'RUN ["/bin/rm", "-rf", "/workspace/input"]',
        'RUN ["/bin/mkdir", "-p", "/workspace/input"]',
    ]
    if include_input:
        lines.append("COPY input/ /workspace/input/")
    lines.extend(
        [
            "RUN " + json.dumps(["/bin/sh", "-c", _RUNTIME_PROJECTION_OVERLAP_PREFLIGHT]),
            "RUN " + json.dumps(["/bin/sh", "-c", _PROJECT_SKILL_ANCESTOR_RESET]),
            _WORKSPACE_SKILL_PATH,
            *_AGENT_SKILL_PATHS,
            *agent_config_lines,
            _RUNTIME_LOADER_ENV_RESET,
            _RUNTIME_POLICY_ENV_RESET,
            "ENTRYPOINT []",
            "HEALTHCHECK NONE",
        ]
    )
    if restore_user is not None and restore_user.casefold() != "root":
        lines.extend(["", f"USER {restore_user}"])
    return content + separator + "\n".join(lines) + "\n"


def _extend_task_input_projection(lines: list[str], *, include_input: bool, agent_config_lines: list[str]) -> None:
    """Add the final evaluator-owned input replacement to generated Dockerfiles."""
    lines.extend(
        [
            "",
            'RUN ["/bin/rm", "-rf", "/workspace/input"]',
            'RUN ["/bin/mkdir", "-p", "/workspace/input"]',
        ]
    )
    if include_input:
        lines.append("COPY input/ /workspace/input/")
    lines.extend(
        [
            "RUN " + json.dumps(["/bin/sh", "-c", _RUNTIME_PROJECTION_OVERLAP_PREFLIGHT]),
            "RUN " + json.dumps(["/bin/sh", "-c", _PROJECT_SKILL_ANCESTOR_RESET]),
            _WORKSPACE_SKILL_PATH,
            *_AGENT_SKILL_PATHS,
            *agent_config_lines,
            _RUNTIME_LOADER_ENV_RESET,
            _RUNTIME_POLICY_ENV_RESET,
            "ENTRYPOINT []",
            "HEALTHCHECK NONE",
        ]
    )


def _append_evaluator_runtime_lines(content: str, lines: list[str], *, elevate: bool = False) -> str:
    """Append evaluator-owned instructions without trusting authored text matches."""
    separator = "" if content.endswith("\n") else "\n"
    user_line = "USER root\n" if elevate else ""
    return content + separator + "\n# SkillEvaluator: final runtime projection\n" + user_line + "\n".join(lines) + "\n"


def _write_agent_configs(env_dir: Path) -> list[str]:
    """Use Harbor's agent integrations and provider-native environment variables."""
    _ = env_dir
    return []


def _rebase_custom_dockerfile_content(
    content: str,
    base_image: str,
    *,
    agent_config_lines: list[str],
    include_input: bool,
    include_repo: bool = False,
    include_repo_linked_root: bool = False,
    agent_workdir: str | None = None,
) -> tuple[str, str]:
    """Return custom Dockerfile content layered on top of the eval base image.

    The base image already contains: python:3.12-slim, system packages
    (bash/curl/git/jq), verifier dependencies, and the
    standard directory structure (/workspace/skills, /logs, etc.).

    The custom Dockerfile's FROM line is replaced so the skill author's
    additions (extra apt/pip packages, COPY, RUN) layer on top.  Multi-agent
    skill discovery paths are appended if not already present.
    Returns the rebased content and original ``FROM`` instruction. Rebase mode
    fails closed for missing or multi-stage ``FROM`` instructions.
    """
    restore_user = _dockerfile_final_explicit_user(content)
    rebased_source, original_from = _replace_single_from_image(content, base_image)
    rebased_content = _append_evaluator_runtime_lines(
        rebased_source + f"\n# SkillEvaluator: original base was {original_from}\n",
        [
            *_verifier_install_lines(),
            *_runtime_copy_lines(
                agent_config_lines,
                include_input,
                include_repo=include_repo,
                include_repo_linked_root=include_repo_linked_root,
                agent_workdir=agent_workdir,
            ),
        ],
        elevate=True,
    )
    rebased_content = _append_task_input_projection(
        rebased_content,
        include_input=include_input,
        agent_config_lines=agent_config_lines,
        restore_user=restore_user,
    )
    return rebased_content, original_from


def _ensure_verifier_deps(
    dockerfile_path: Path,
    *,
    agent_config_lines: list[str],
    include_input: bool,
    include_repo: bool = False,
    include_repo_linked_root: bool = False,
    agent_workdir: str | None = None,
) -> None:
    """Append verifier deps and runtime COPY lines to a custom Dockerfile."""
    content = dockerfile_path.read_text(encoding="utf-8")
    restore_user = _dockerfile_final_explicit_user(content)
    user_prefix = "USER root\n" if restore_user is not None else ""
    additions = f"\n{user_prefix}" + "\n".join(_verifier_install_lines()) + "\n"
    updated = _append_evaluator_runtime_lines(
        content + additions,
        _runtime_copy_lines(
            agent_config_lines,
            include_input,
            include_repo=include_repo,
            include_repo_linked_root=include_repo_linked_root,
            agent_workdir=agent_workdir,
        ),
    )
    updated = _append_task_input_projection(
        updated,
        include_input=include_input,
        agent_config_lines=agent_config_lines,
        restore_user=restore_user,
    )
    dockerfile_path.write_text(updated, encoding="utf-8")
    logger.debug("Appended verifier deps + agent paths to %s", dockerfile_path)


def _entry_file_refs(entry: dict[str, Any]) -> list[str]:
    raw_files = entry.get("files")
    if raw_files is None:
        return []
    if isinstance(raw_files, str):
        raw_items = [raw_files]
    elif isinstance(raw_files, list):
        raw_items = raw_files
    else:
        raise ValueError(f"evals.json entry '{entry.get('id', '<unknown>')}' files must be a string or list of strings")

    refs: list[str] = []
    for idx, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, str):
            raise ValueError(f"evals.json entry '{entry.get('id', '<unknown>')}' files[{idx}] must be a string")
        ref = raw_item.strip()
        if ref:
            refs.append(ref)
    return refs


def _is_evaluator_only_task_input(relative: Path) -> bool:
    """Return whether an eval-relative path is evaluator control material."""
    if not relative.parts:
        return True
    top_level = relative.parts[0].casefold()
    return top_level in _EVALUATOR_ONLY_TASK_INPUT_ROOTS or (
        len(relative.parts) == 1 and top_level in _EVALUATOR_ONLY_TASK_INPUT_FILES
    )


def _resolve_entry_file_ref(
    ref: str,
    *,
    skill_path: Path,
    evals_dir: Path,
    input_files_dir: Path | None,
) -> tuple[Path, Path]:
    if "\x00" in ref:
        raise ValueError("evals.json files entries cannot contain NUL bytes")

    ref_path = Path(ref)
    if ref_path.is_absolute():
        raise ValueError(f"evals.json files entry must be relative to evals/: {ref}")
    if ".." in ref_path.parts:
        raise ValueError(f"evals.json files entry cannot traverse parent directories: {ref}")
    if _LOCAL_LINK_SCHEME_RE.match(ref):
        raise ValueError(f"evals.json files entry uses unsupported URI scheme: {ref}")

    if ref_path.parts and ref_path.parts[0] == "evals":
        candidates = [evals_dir.joinpath(*ref_path.parts[1:])]
    else:
        candidates = [evals_dir / ref_path]

    source_path = next((candidate.absolute() for candidate in candidates if candidate.exists()), None)
    if source_path is None:
        raise FileNotFoundError(f"evals.json files entry does not exist: {ref}")
    source = source_path.resolve()

    resolved_evals_dir = evals_dir.resolve()
    if source == resolved_evals_dir or not _is_relative_to(source, resolved_evals_dir):
        raise ValueError(f"evals.json files entry resolves outside evals/: {ref}")
    source_relative = source.relative_to(resolved_evals_dir)
    if _is_evaluator_only_task_input(source_relative):
        raise ValueError(f"evals.json files entry selects evaluator-only material: {ref}")
    if _has_symlink_component(source_path, evals_dir.absolute()):
        raise ValueError(f"evals.json files entry must not contain symlinks: {ref}")

    resolved_input_files_dir = input_files_dir.resolve() if input_files_dir and input_files_dir.exists() else None
    if resolved_input_files_dir and _is_relative_to(source, resolved_input_files_dir):
        rel = source.relative_to(resolved_input_files_dir)
    else:
        rel = source_relative

    return source_path, rel


def _copy_input_ref(source: Path, input_dir: Path, rel: Path, *, allowed_root: Path) -> None:
    dest = input_dir if rel == Path() else input_dir / rel
    if source.is_dir():
        if dest.exists() and dest.is_file():
            dest.unlink()
        copytree_secure(source, dest, dirs_exist_ok=True, allowed_root=allowed_root)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    copy_file_secure(source, dest, allowed_root=allowed_root)


def _stage_task_inputs(
    env_dir: Path,
    *,
    input_files_dir: Path | None,
    entry: dict[str, Any],
    source_skill_path: Path,
    evals_dir: Path,
) -> bool:
    input_dir = env_dir / "input"
    if os.path.lexists(input_dir) and (_path_is_link_or_reparse(input_dir) or not input_dir.is_dir()):
        input_dir.unlink()
    elif input_dir.exists():
        shutil.rmtree(input_dir)

    if "files" not in entry and input_files_dir and input_files_dir.exists():
        copytree_secure(input_files_dir, input_dir, dirs_exist_ok=True, allowed_root=evals_dir)

    for ref in _entry_file_refs(entry):
        source, rel = _resolve_entry_file_ref(
            ref,
            skill_path=source_skill_path,
            evals_dir=evals_dir,
            input_files_dir=input_files_dir,
        )
        input_dir.mkdir(parents=True, exist_ok=True)
        _copy_input_ref(source, input_dir, rel, allowed_root=evals_dir)

    return input_dir.exists()


def _entry_declares_task_input(entry: dict[str, Any], input_files_dir: Path | None) -> bool:
    """Return whether eval metadata should replace a native task's own input."""
    return "files" in entry or bool(input_files_dir is not None and input_files_dir.exists())


def _remove_staged_path(path: Path) -> None:
    if _path_is_link_or_reparse(path) or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _sanitize_staged_runtime_skills(skills_dir: Path) -> None:
    """Remove evaluator-only roots from authored or evaluator-copied skill bundles."""
    if not skills_dir.exists():
        return
    eval_roots: set[Path] = set()
    for manifest in _iter_skill_manifests(skills_dir):
        skill_root = manifest.parent
        if not skill_root.is_dir():
            continue
        eval_roots.update(child for child in skill_root.iterdir() if child.name.casefold() == "evals")
    for eval_root in sorted(eval_roots, key=lambda path: len(path.parts), reverse=True):
        if os.path.lexists(eval_root):
            _remove_staged_path(eval_root)


def _write_dockerfile(
    task_dir: Path,
    skill_path: Path | None,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None,
    has_skill: bool,
    input_files_dir: Path | None = None,
    entry: dict[str, Any] | None = None,
    evals_dir: Path | None = None,
    exclude_skill_name: str | None = None,
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    repo_context_skill_path: Path | None = None,
    repo_context_mode: str = "linked",
    compose_env_names: set[str] | None = None,
    repo_context_exclude_paths: Sequence[Path] = (),
    agent_workdir: str | None = None,
    baseline_aliases_prevalidated: bool = False,
) -> None:
    """Generate a Dockerfile that installs skills into the container.

    Environment resolution order:
      1. ``skill_path/evals/environment/Dockerfile`` -- developer's custom Dockerfile
      2. Pre-built base image (when *base_image* is set) -- only COPY layers
      3. ``scripts/requirements.txt`` + ``scripts/apt-packages.txt`` -- auto-detected deps
      4. Default generic Dockerfile
    """
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    if not has_skill and skill_path is not None and not baseline_aliases_prevalidated:
        _check_baseline_skill_candidates_do_not_alias_target(
            skill_path,
            reference_skills_dir,
            workspace_skill_paths,
            excluded_roots=repo_context_exclude_paths,
        )

    skills_dir = env_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    if has_skill and skill_path and skill_path.exists():
        dest = skills_dir / skill_path.name
        if dest.exists():
            shutil.rmtree(dest)
        copytree_secure(
            skill_path,
            dest,
            dirs_exist_ok=True,
            ignore=_runtime_skill_copy_ignore(skill_path, repo_context_exclude_paths),
        )

    if reference_skills_dir and reference_skills_dir.exists():
        for ref_skill in reference_skills_dir.iterdir():
            if ref_skill.is_dir() and not ref_skill.name.startswith("."):
                if not has_skill and exclude_skill_name and ref_skill.name == exclude_skill_name:
                    continue
                dest = skills_dir / ref_skill.name
                if not dest.exists():
                    copytree_secure(
                        ref_skill,
                        dest,
                        dirs_exist_ok=True,
                        ignore=_runtime_skill_copy_ignore(ref_skill, repo_context_exclude_paths),
                    )

    for workspace_skill in workspace_skill_paths or []:
        if not workspace_skill.exists() or not workspace_skill.is_dir():
            continue
        if not has_skill and exclude_skill_name and workspace_skill.name == exclude_skill_name:
            continue
        dest = skills_dir / workspace_skill.name
        if dest.exists():
            continue
        copytree_secure(
            workspace_skill,
            dest,
            dirs_exist_ok=True,
            ignore=_runtime_skill_copy_ignore(workspace_skill, repo_context_exclude_paths),
        )

    _sanitize_staged_runtime_skills(skills_dir)

    include_input = False
    if entry is not None and skill_path is not None and evals_dir is not None:
        include_input = _stage_task_inputs(
            env_dir,
            input_files_dir=input_files_dir,
            entry=entry,
            source_skill_path=skill_path,
            evals_dir=evals_dir,
        )

    effective_repo_context_mode = "full" if repo_context_mode == "full" else ("linked" if has_skill else "none")
    _stage_repo_context(
        env_dir,
        source_skill_path=repo_context_skill_path,
        mode=effective_repo_context_mode,
        exclude_source_skill=repo_context_mode == "full" and not has_skill,
        excluded_roots=repo_context_exclude_paths,
    )
    if not has_skill and skill_path is not None:
        _check_staged_baseline_does_not_contain_target(env_dir, skill_path)

    agent_config_lines = _write_agent_configs(env_dir)
    include_repo = (env_dir / "repo").exists()
    include_repo_linked_root = (env_dir / "repo-linked-root").exists()

    custom_env_dir = (
        evals_dir / "environment"
        if evals_dir is not None
        else skill_path / "evals" / "environment"
        if skill_path is not None
        else None
    )

    if custom_env_dir and custom_env_dir.exists():
        private_custom_env = tempfile.TemporaryDirectory(prefix="skillevaluator-custom-environment-")
        staged_custom_env = Path(private_custom_env.name) / "environment"
        try:
            copytree_secure(custom_env_dir, staged_custom_env, allowed_root=evals_dir)
            if not has_skill and skill_path:
                _check_custom_environment_does_not_stage_target(staged_custom_env, skill_path)

            custom_dockerfile = staged_custom_env / "Dockerfile"
            custom_dockerfile_accepted = False
            if custom_dockerfile.exists():
                err = _validate_custom_dockerfile(custom_dockerfile)
                if err:
                    logger.warning("Custom Dockerfile rejected (%s): %s", custom_env_dir / "Dockerfile", err)
                else:
                    shutil.copy2(custom_dockerfile, env_dir / "Dockerfile")
                    custom_dockerfile_accepted = True
                    logger.debug("Using custom Dockerfile from %s", custom_env_dir / "Dockerfile")

            compose_file = _custom_compose_path(staged_custom_env)
            if compose_file is not None:
                shutil.copy2(compose_file, env_dir / "docker-compose.yaml")
                _validate_and_sanitize_custom_compose(
                    env_dir / "docker-compose.yaml",
                    allowed_env=compose_env_names or set(),
                )
                logger.debug("Copied docker-compose.yaml from %s", custom_env_dir / compose_file.name)

            for subdir in staged_custom_env.iterdir():
                if subdir.is_dir() and subdir.name not in ("__pycache__", ".git") and subdir.name.casefold() != "input":
                    dest = env_dir / subdir.name
                    if not dest.exists():
                        copytree_secure(subdir, dest, allowed_root=staged_custom_env)
                        logger.debug("Copied sidecar dir %s", subdir.name)

            _ensure_empty_compose_input_compatibility(env_dir, has_input=include_input)

            if custom_dockerfile_accepted:
                _ensure_empty_custom_docker_input_compatibility(
                    env_dir,
                    env_dir / "Dockerfile",
                    has_input=include_input,
                )
                if base_image and custom_dockerfile_mode == "rebase":
                    dockerfile_path = env_dir / "Dockerfile"
                    rebased = _rebase_custom_dockerfile_content(
                        dockerfile_path.read_text(encoding="utf-8"),
                        base_image,
                        agent_config_lines=agent_config_lines,
                        include_input=include_input,
                        include_repo=include_repo,
                        include_repo_linked_root=include_repo_linked_root,
                        agent_workdir=agent_workdir,
                    )
                    if rebased is not None:
                        rebased_content, original_from = rebased
                        dockerfile_path.write_text(rebased_content, encoding="utf-8")
                        logger.warning(
                            "Rebased custom Dockerfile from '%s' onto '%s'",
                            original_from,
                            base_image,
                        )
                else:
                    _ensure_verifier_deps(
                        env_dir / "Dockerfile",
                        agent_config_lines=agent_config_lines,
                        include_input=include_input,
                        include_repo=include_repo,
                        include_repo_linked_root=include_repo_linked_root,
                        agent_workdir=agent_workdir,
                    )
                return
        finally:
            private_custom_env.cleanup()

    if base_image:
        dockerfile_lines = [
            f"FROM {base_image}",
            "",
            *_runtime_copy_lines(
                agent_config_lines,
                include_input,
                include_repo=include_repo,
                include_repo_linked_root=include_repo_linked_root,
                agent_workdir=agent_workdir,
            ),
        ]
        _extend_task_input_projection(
            dockerfile_lines,
            include_input=include_input,
            agent_config_lines=agent_config_lines,
        )

        (env_dir / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")
        return

    pip_reqs = _collect_txt_deps(skills_dir, "requirements.txt")
    apt_pkgs = _collect_txt_deps(skills_dir, "apt-packages.txt")

    dockerfile_lines = [
        "FROM python:3.12-slim",
        "",
        "RUN apt-get -o Acquire::Retries=3 update && \\",
        "    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \\",
        "    bash curl git jq ripgrep \\",
    ]
    if apt_pkgs:
        dockerfile_lines[-1] += " \\"
        dockerfile_lines.append("    " + " ".join(apt_pkgs) + " \\")
    dockerfile_lines.append("    && rm -rf /var/lib/apt/lists/*")

    dockerfile_lines.extend(["", *_verifier_install_lines(pip_reqs)])

    dockerfile_lines.extend(
        [
            "",
            "RUN mkdir -p /workspace/skills /workspace/input /workspace/output \\",
            "    /logs/verifier /logs/agent",
            "",
            *_runtime_copy_lines(
                agent_config_lines,
                include_input,
                include_repo=include_repo,
                include_repo_linked_root=include_repo_linked_root,
                agent_workdir=agent_workdir,
            ),
        ]
    )
    _extend_task_input_projection(
        dockerfile_lines,
        include_input=include_input,
        agent_config_lines=agent_config_lines,
    )

    dockerfile_lines.extend(
        [
            "",
            "WORKDIR /workspace",
        ]
    )

    (env_dir / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")


def _copy_skill_dirs(
    *,
    env_dir: Path,
    skill_path: Path | None,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None,
    has_skill: bool,
    exclude_skill_name: str | None,
    excluded_roots: Sequence[Path] = (),
) -> None:
    """Stage target/workspace skills into a task environment."""
    skills_dir = env_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    if has_skill and skill_path and skill_path.exists():
        dest = skills_dir / skill_path.name
        if dest.exists():
            shutil.rmtree(dest)
        copytree_secure(
            skill_path,
            dest,
            dirs_exist_ok=True,
            ignore=_runtime_skill_copy_ignore(skill_path, excluded_roots),
        )

    if reference_skills_dir and reference_skills_dir.exists():
        for ref_skill in reference_skills_dir.iterdir():
            if ref_skill.is_dir() and not ref_skill.name.startswith("."):
                if not has_skill and exclude_skill_name and ref_skill.name == exclude_skill_name:
                    continue
                dest = skills_dir / ref_skill.name
                if not dest.exists():
                    copytree_secure(
                        ref_skill,
                        dest,
                        dirs_exist_ok=True,
                        ignore=_runtime_skill_copy_ignore(ref_skill, excluded_roots),
                    )

    for workspace_skill in workspace_skill_paths or []:
        if not workspace_skill.exists() or not workspace_skill.is_dir():
            continue
        if not has_skill and exclude_skill_name and workspace_skill.name == exclude_skill_name:
            continue
        dest = skills_dir / workspace_skill.name
        if dest.exists():
            continue
        copytree_secure(
            workspace_skill,
            dest,
            dirs_exist_ok=True,
            ignore=_runtime_skill_copy_ignore(workspace_skill, excluded_roots),
        )

    _sanitize_staged_runtime_skills(skills_dir)


def _write_default_environment_dockerfile(
    env_dir: Path,
    *,
    base_image: str,
    agent_config_lines: list[str],
    include_input: bool,
    include_repo: bool = False,
    include_repo_linked_root: bool = False,
    include_verifier_deps: bool = True,
    agent_workdir: str | None = None,
) -> None:
    if base_image:
        dockerfile_lines = [
            f"FROM {base_image}",
            "",
            *_runtime_copy_lines(
                agent_config_lines,
                include_input,
                include_repo=include_repo,
                include_repo_linked_root=include_repo_linked_root,
                agent_workdir=agent_workdir,
            ),
        ]
        _extend_task_input_projection(
            dockerfile_lines,
            include_input=include_input,
            agent_config_lines=agent_config_lines,
        )
        (env_dir / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")
        return

    pip_reqs = _collect_txt_deps(env_dir / "skills", "requirements.txt")
    apt_pkgs = _collect_txt_deps(env_dir / "skills", "apt-packages.txt")

    dockerfile_lines = [
        "FROM python:3.12-slim",
        "",
        "RUN apt-get -o Acquire::Retries=3 update && \\",
        "    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \\",
        "    bash curl git jq ripgrep \\",
    ]
    if apt_pkgs:
        dockerfile_lines[-1] += " \\"
        dockerfile_lines.append("    " + " ".join(apt_pkgs) + " \\")
    dockerfile_lines.append("    && rm -rf /var/lib/apt/lists/*")
    if include_verifier_deps:
        dockerfile_lines.extend(["", *_verifier_install_lines(pip_reqs)])
    elif pip_reqs:
        dockerfile_lines.append(_managed_run("python", "-m", "pip", "install", "--no-cache-dir", *pip_reqs))
    dockerfile_lines.extend(
        [
            "",
            "RUN mkdir -p /workspace/skills /workspace/input /workspace/output \\",
            "    /logs/verifier /logs/agent",
            "",
            *_runtime_copy_lines(
                agent_config_lines,
                include_input,
                include_repo=include_repo,
                include_repo_linked_root=include_repo_linked_root,
                agent_workdir=agent_workdir,
            ),
        ]
    )
    _extend_task_input_projection(
        dockerfile_lines,
        include_input=include_input,
        agent_config_lines=agent_config_lines,
    )
    dockerfile_lines.extend(["", "WORKDIR /workspace"])
    (env_dir / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")


def _prepare_native_environment(
    task_dir: Path,
    *,
    skill_path: Path,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None,
    has_skill: bool,
    base_image: str,
    custom_dockerfile_mode: str,
    grading_mode: str,
    repo_context_mode: str = "linked",
    compose_env_names: set[str] | None = None,
    repo_context_exclude_paths: Sequence[Path] = (),
    agent_workdir: str | None = None,
    baseline_aliases_prevalidated: bool = False,
) -> None:
    """Stage SkillEvaluator runtime additions into a copied native Harbor task."""
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    if not has_skill and not baseline_aliases_prevalidated:
        _check_baseline_skill_candidates_do_not_alias_target(
            skill_path,
            reference_skills_dir,
            workspace_skill_paths,
            excluded_roots=repo_context_exclude_paths,
        )
    _copy_skill_dirs(
        env_dir=env_dir,
        skill_path=skill_path if has_skill else None,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths,
        has_skill=has_skill,
        exclude_skill_name=skill_path.name,
        excluded_roots=repo_context_exclude_paths,
    )
    agent_config_lines = _write_agent_configs(env_dir)
    effective_repo_context_mode = "full" if repo_context_mode == "full" else ("linked" if has_skill else "none")
    _stage_repo_context(
        env_dir,
        source_skill_path=skill_path,
        mode=effective_repo_context_mode,
        exclude_source_skill=repo_context_mode == "full" and not has_skill,
        excluded_roots=repo_context_exclude_paths,
    )
    if not has_skill:
        _check_staged_baseline_does_not_contain_target(env_dir, skill_path)
    include_input = (env_dir / "input").exists()
    include_repo = (env_dir / "repo").exists()
    include_repo_linked_root = (env_dir / "repo-linked-root").exists()
    dockerfile_path = env_dir / "Dockerfile"
    compose_path = _custom_compose_path(env_dir)
    if compose_path is not None:
        _validate_and_sanitize_custom_compose(compose_path, allowed_env=compose_env_names or set())
        if compose_path.name == "docker-compose.yml":
            compose_path = compose_path.replace(env_dir / "docker-compose.yaml")
        _ensure_empty_compose_input_compatibility(env_dir, has_input=include_input)

    if dockerfile_path.exists():
        err = _validate_custom_dockerfile(dockerfile_path)
        if err:
            raise ValueError(f"{dockerfile_path}: {err}")
        _ensure_empty_custom_docker_input_compatibility(
            env_dir,
            dockerfile_path,
            has_input=include_input,
        )
        if base_image and custom_dockerfile_mode == "rebase":
            rebased = _rebase_custom_dockerfile_content(
                dockerfile_path.read_text(encoding="utf-8"),
                base_image,
                agent_config_lines=agent_config_lines,
                include_input=include_input,
                include_repo=include_repo,
                include_repo_linked_root=include_repo_linked_root,
                agent_workdir=agent_workdir,
            )
            if rebased is not None:
                rebased_content, original_from = rebased
                dockerfile_path.write_text(rebased_content, encoding="utf-8")
                logger.warning(
                    "Rebased custom Dockerfile from '%s' onto '%s'",
                    original_from,
                    base_image,
                )
        elif grading_mode == "custom_only":
            content = dockerfile_path.read_text(encoding="utf-8")
            restore_user = _dockerfile_final_explicit_user(content)
            updated = _append_evaluator_runtime_lines(
                content,
                _runtime_copy_lines(
                    agent_config_lines,
                    include_input,
                    include_repo=include_repo,
                    include_repo_linked_root=include_repo_linked_root,
                    agent_workdir=agent_workdir,
                ),
                elevate=restore_user is not None,
            )
            updated = _append_task_input_projection(
                updated,
                include_input=include_input,
                agent_config_lines=agent_config_lines,
                restore_user=restore_user,
            )
            dockerfile_path.write_text(updated, encoding="utf-8")
        else:
            _ensure_verifier_deps(
                dockerfile_path,
                agent_config_lines=agent_config_lines,
                include_input=include_input,
                include_repo=include_repo,
                include_repo_linked_root=include_repo_linked_root,
                agent_workdir=agent_workdir,
            )
        return

    _write_default_environment_dockerfile(
        env_dir,
        base_image=base_image,
        agent_config_lines=agent_config_lines,
        include_input=include_input,
        include_repo=include_repo,
        include_repo_linked_root=include_repo_linked_root,
        include_verifier_deps=grading_mode != "custom_only",
        agent_workdir=agent_workdir,
    )


def _native_task_dirs(
    dataset_dir: Path,
    *,
    ignore: Callable[[str, list[str]], Iterable[str]] | None = None,
) -> list[Path]:
    """Return native Harbor task directories from a copied dataset."""
    children = sorted(dataset_dir.iterdir())
    ignored = set(ignore(os.fspath(dataset_dir), [path.name for path in children])) if ignore is not None else set()
    return [
        p
        for p in children
        if p.name not in ignored and p.is_dir() and not p.name.startswith((".", "_")) and (p / "task.toml").exists()
    ]


def _native_source_path_is_ignored(path: Path, native_dir: Path) -> bool:
    """Apply the native copy ignore callback to every lexical path component."""
    parent = native_dir
    for part in path.relative_to(native_dir).parts:
        if part in _NATIVE_SOURCE_IGNORE(os.fspath(parent), [part]):
            return True
        parent /= part
    return False


def _native_entry_id(task_dir: Path) -> str:
    task_toml = task_dir / "task.toml"
    try:
        import tomllib

        data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except Exception:
        return task_dir.name
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    if isinstance(metadata, dict) and metadata.get("entry_id"):
        return str(metadata["entry_id"])
    return task_dir.name


def _environment_reference_names(value: object) -> set[str]:
    """Return portable shell-style environment references from a TOML value."""
    if not isinstance(value, str):
        return set()
    dollar_references = {
        braced or plain
        for braced, plain in re.findall(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}|\$([A-Za-z_][A-Za-z0-9_]*)\b",
            value,
        )
    }
    percent_references = set(re.findall(r"%([A-Za-z_][A-Za-z0-9_]*)%", value))
    return dollar_references | percent_references


def _judge_model_env_controls(environment: dict[str, Any]) -> set[str]:
    """Return reserved judge-model names used as keys or value references."""
    authored_keys = {name for name in environment if name.upper() in _VERIFIER_JUDGE_MODEL_ENV_VARS}
    authored_references = {
        reference
        for value in environment.values()
        for reference in _environment_reference_names(value)
        if reference.upper() in _VERIFIER_JUDGE_MODEL_ENV_VARS
    }
    return authored_keys | authored_references


def _validate_native_agent_judge_model_controls(task_toml: Path, environment_env: dict[str, Any]) -> None:
    """Keep host judge controls out of the native task's agent environment."""
    controls = sorted(_judge_model_env_controls(environment_env))
    if controls:
        raise ValueError(
            f"Native Harbor task [environment.env] cannot name or reference evaluator-controlled judge model "
            f"variable(s): {', '.join(controls)}: {task_toml}"
        )


def _native_task_workdir(task_dir: Path, *, allow_docker_image: bool = False) -> str | None:
    """Read and validate the workdir Harbor will use for a native task."""
    try:
        import tomllib

        data = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot parse native Harbor task config: {task_dir / 'task.toml'}") from exc
    environment = data.get("environment", {}) if isinstance(data, dict) else {}
    if not isinstance(environment, dict):
        raise ValueError(f"Native Harbor task [environment] must be a table: {task_dir / 'task.toml'}")
    environment_env = environment.get("env", {})
    if not isinstance(environment_env, dict):
        raise ValueError(f"Native Harbor task [environment.env] must be a table: {task_dir / 'task.toml'}")
    _validate_native_agent_judge_model_controls(task_dir / "task.toml", environment_env)
    _validate_runtime_discovery_env(environment_env)
    _validate_runtime_loader_env(environment_env)
    skills_dir = environment.get("skills_dir")
    if skills_dir not in (None, "/workspace/skills"):
        raise ValueError(
            "Native Harbor task skills_dir must use the evaluator-controlled /workspace/skills projection: "
            f"{task_dir / 'task.toml'}"
        )
    docker_image = environment.get("docker_image")
    if docker_image not in (None, "") and not allow_docker_image:
        raise ValueError(
            "Native Harbor task docker_image bypasses the evaluator-controlled runtime projection; "
            "declare the image in environment/Dockerfile instead: "
            f"{task_dir / 'task.toml'}"
        )
    workdir = environment.get("workdir")
    if workdir is not None and not isinstance(workdir, str):
        raise ValueError(f"Native Harbor task workdir must be a string: {task_dir / 'task.toml'}")
    return _validated_agent_workdir(workdir)


def _ensure_native_skills_dir(task_dir: Path) -> None:
    """Pin native tasks to the evaluator-owned runtime skill projection."""
    task_toml = task_dir / "task.toml"
    content = task_toml.read_text(encoding="utf-8")
    lines = content.splitlines()
    if "[environment]" in lines:
        environment_index = lines.index("[environment]")
        section_end = next(
            (index for index in range(environment_index + 1, len(lines)) if lines[index].strip().startswith("[")),
            len(lines),
        )
        if any(line.strip().startswith("skills_dir") for line in lines[environment_index + 1 : section_end]):
            return
        lines.insert(environment_index + 1, 'skills_dir = "/workspace/skills"')
    else:
        lines.extend(["", "[environment]", 'skills_dir = "/workspace/skills"'])
    task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_skill_evaluator_verifier_env(task_dir: Path, *, verifier_env: dict[str, str] | None) -> None:
    """Ensure staged native tasks forward configured public provider variables."""
    task_toml = task_dir / "task.toml"
    content = task_toml.read_text(encoding="utf-8")
    env_lines = [f'{name} = "${{{name}}}"' for name in _verifier_env_vars(verifier_env)]
    if all(line in content for line in env_lines):
        return

    lines = content.splitlines()
    if "[verifier.env]" in lines:
        idx = lines.index("[verifier.env]") + 1
        existing = set(lines)
        for line in reversed(env_lines):
            if line not in existing:
                lines.insert(idx, line)
        task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    env_block = ["[verifier.env]", *env_lines]
    if "[verifier]" in lines:
        start = lines.index("[verifier]") + 1
        insert_at = len(lines)
        for idx in range(start, len(lines)):
            line = lines[idx].strip()
            if line.startswith("[") and line.endswith("]"):
                insert_at = idx
                break
        lines[insert_at:insert_at] = ["", *env_block]
        task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    insert_at = lines.index("[environment]") if "[environment]" in lines else len(lines)
    lines[insert_at:insert_at] = ["[verifier]", "timeout_sec = 180.0", "", *env_block, ""]
    task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _insert_table_block(lines: list[str], anchor: str, block: list[str]) -> None:
    """Insert a TOML table block after *anchor* and before the next table."""
    if anchor not in lines:
        lines.extend(["", anchor])
    start = lines.index(anchor) + 1
    insert_at = len(lines)
    for idx in range(start, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            insert_at = idx
            break
    prefix = [] if insert_at == 0 or (insert_at > 0 and lines[insert_at - 1] == "") else [""]
    lines[insert_at:insert_at] = [*prefix, *block]


def _ensure_environment_env(task_dir: Path, runtime_env: dict[str, str]) -> None:
    runtime_env = {**runtime_env, **_EVALUATOR_MANAGED_RUNTIME_ENV}

    task_toml = task_dir / "task.toml"
    lines = task_toml.read_text(encoding="utf-8").splitlines()
    header = "[environment.env]"
    rendered = {key: f"{_toml_quote(key)} = {_toml_quote(runtime_env[key])}" for key in sorted(runtime_env)}
    env_lines = list(rendered.values())
    if header in lines:
        idx = lines.index(header) + 1
        env_end = len(lines)
        for end_idx in range(idx, len(lines)):
            stripped = lines[end_idx].strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                env_end = end_idx
                break
        seen_keys: set[str] = set()
        updated_section: list[str] = []
        for line in lines[idx:env_end]:
            assignment = line.split("=", 1)[0].strip() if line.strip() and "=" in line else ""
            matching_key = next(
                (key for key in runtime_env if assignment in {key, _toml_quote(key)}),
                None,
            )
            if matching_key is not None:
                if matching_key in seen_keys:
                    continue
                updated_section.append(rendered[matching_key])
                seen_keys.add(matching_key)
            else:
                updated_section.append(line)
        for key, line in rendered.items():
            if key not in seen_keys:
                updated_section.append(line)
        lines[idx:env_end] = updated_section
        task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    block = [header, *env_lines]
    _insert_table_block(lines, "[environment]", block)
    task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_pre_agent_setup_healthcheck(task_dir: Path, pre_agent_setup: list[str]) -> None:
    command = _pre_agent_setup_command(pre_agent_setup)
    if not command:
        return

    task_toml = task_dir / "task.toml"
    lines = task_toml.read_text(encoding="utf-8").splitlines()
    header = "[environment.healthcheck]"
    if header in lines:
        raise ValueError(
            f"{task_toml}: harbor.pre_agent_setup cannot be injected because "
            "the native Harbor task already defines [environment.healthcheck]"
        )

    block = [
        header,
        f"command = {_toml_quote(command)}",
        "interval_sec = 5.0",
        "timeout_sec = 120.0",
        "retries = 1",
    ]
    _insert_table_block(lines, "[environment.env]" if "[environment.env]" in lines else "[environment]", block)
    task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_runtime_env_and_pre_agent_setup(
    task_dir: Path,
    *,
    runtime_env: dict[str, str] | None,
    pre_agent_setup: list[str] | None,
) -> None:
    _ensure_environment_env(task_dir, runtime_env or {})
    _ensure_pre_agent_setup_healthcheck(task_dir, pre_agent_setup or [])


def _load_entries_by_id(skill_path: Path) -> dict[str, dict[str, Any]]:
    evals_file = find_evals_file(skill_path)
    if not evals_file:
        return {}
    entries = _load_evals(evals_file)
    case_ids = validate_case_ids(entry.get("id") for entry in entries)
    return {case_id: {**entry, "id": case_id} for entry, case_id in zip(entries, case_ids, strict=True)}


def _dockerfile_copy_or_add_mentions_skill(line: str, target_skill_name: str) -> bool:
    stripped = line.strip()
    if not stripped.upper().startswith(("COPY ", "ADD ")):
        return False
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    path_tokens = [token for token in tokens[1:] if not token.startswith("--")]
    for token in path_tokens:
        normalized = re.sub(r"[\[\]\",:]+", "/", token).replace("\\", "/")
        if target_skill_name in [segment for segment in normalized.split("/") if segment]:
            return True
    return False


def _check_custom_environment_does_not_stage_target(custom_env_dir: Path, target_skill: Path) -> None:
    target_skill_name = target_skill.name
    manifests = _iter_skill_manifests(custom_env_dir)
    if manifests:
        raise ValueError(
            f"Baseline custom eval environment contains an unmanaged skill package at {manifests[0]}. "
            "Configure reference/workspace skills through SkillEvaluator instead so baseline staging stays uncontaminated."
        )
    payload = _find_target_manifest_payload(custom_env_dir, target_skill)
    if payload is not None:
        raise ValueError(
            f"Baseline custom eval environment contains the target skill instructions in {payload}. "
            "The evaluated skill must not be embedded under an alias."
        )
    for dockerfile in custom_env_dir.rglob("Dockerfile"):
        try:
            lines = dockerfile.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if _dockerfile_copy_or_add_mentions_skill(line, target_skill_name):
                raise ValueError(
                    f"Custom eval Dockerfile appears to copy target skill '{target_skill_name}' in {dockerfile}. "
                    "Use SkillEvaluator skill staging instead so baseline stays uncontaminated."
                )


def _check_native_source_does_not_stage_target(
    native_dir: Path,
    target_skill: Path,
    *,
    with_skill: bool,
) -> None:
    def _is_ignored(path: Path) -> bool:
        return _native_source_path_is_ignored(path, native_dir)

    target_skill_name = target_skill.name
    for skill_md in _iter_skill_manifests(native_dir):
        if _is_ignored(skill_md):
            continue
        if "environment" not in skill_md.relative_to(native_dir).parts:
            continue
        if not with_skill:
            raise ValueError(
                f"Baseline native environment contains an unmanaged skill package at {skill_md}. "
                "Configure reference/workspace skills through SkillEvaluator instead."
            )
        if skill_md.parent.name == target_skill_name:
            raise ValueError(
                f"BYOT source already stages target skill '{target_skill_name}' at {skill_md}. "
                "SkillEvaluator must control with-skill and baseline staging."
            )
    if not with_skill:
        for environment_dir in native_dir.rglob("environment"):
            if not environment_dir.is_dir() or _is_ignored(environment_dir):
                continue
            payload = _find_target_manifest_payload(
                environment_dir,
                target_skill,
                ignored_path_predicate=_is_ignored,
            )
            if payload is not None:
                raise ValueError(
                    f"Baseline native environment contains the target skill instructions in {payload}. "
                    "The evaluated skill must not be embedded under an alias."
                )
    for dockerfile in native_dir.rglob("Dockerfile"):
        if _is_ignored(dockerfile):
            continue
        try:
            lines = dockerfile.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if _dockerfile_copy_or_add_mentions_skill(line, target_skill_name):
                raise ValueError(
                    f"BYOT Dockerfile appears to copy target skill '{target_skill_name}' in {dockerfile}. "
                    "Use SkillEvaluator workspace staging instead so baseline stays uncontaminated."
                )


def _stage_native_harbor_tasks_into(
    skill_path: Path,
    output_dir: Path,
    *,
    evaluator_skill_path: Path,
    with_skill: bool = True,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: list[Path] | None = None,
    workspace_mode: str = "isolated",
    grading_mode: str = "default",
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    copy_repo: bool = False,
    repo_context_exclude_paths: Sequence[Path] = (),
    private_repo_context_exclude_paths: Sequence[Path] = (),
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
    baseline_aliases_prevalidated: bool = False,
) -> list[Path]:
    """Build native Harbor tasks inside a private, caller-owned directory.

    The source tree is copied first and all SkillEvaluator injections happen only in the
    staged result directory.
    """
    _validate_runtime_discovery_env(runtime_env)
    _validate_runtime_loader_env(runtime_env)
    evals_dir = evaluator_skill_path / "evals"
    native_dir = evals_dir / "harbor"
    if not native_dir.exists():
        raise FileNotFoundError(f"No native Harbor task source found at {native_dir}")
    _validate_staging_output_location(
        skill_path,
        output_dir,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths or (),
    )
    _validate_output_roots_outside_evaluator_sources(skill_path, repo_context_exclude_paths)
    _check_native_source_does_not_stage_target(native_dir, skill_path, with_skill=with_skill)
    entries_by_id = _load_entries_by_id(evaluator_skill_path)
    source_task_dirs = _native_task_dirs(native_dir, ignore=_NATIVE_SOURCE_IGNORE)
    if not source_task_dirs:
        raise ValueError(f"No Harbor task directories with task.toml found in {native_dir}")
    validate_case_ids(path.name for path in source_task_dirs)
    for source_task_dir in source_task_dirs:
        _native_task_workdir(source_task_dir)
    _ = task_resources
    _ = agent_workdir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    copytree_secure(
        native_dir,
        output_dir,
        ignore=_NATIVE_SOURCE_IGNORE,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dirs = _native_task_dirs(output_dir)

    if grading_mode in ("default", "default_plus_custom") and not entries_by_id:
        raise FileNotFoundError(
            "Native Harbor tasks with SkillEvaluator default grading require evals/evals.json metadata matching task IDs"
        )

    workspace_skill_names = sorted({p.name for p in workspace_skill_paths or []})
    effective_excluded_roots = (*repo_context_exclude_paths, *private_repo_context_exclude_paths)
    if not with_skill and not baseline_aliases_prevalidated:
        _check_baseline_skill_candidates_do_not_alias_target(
            skill_path,
            reference_skills_dir,
            workspace_skill_paths,
            excluded_roots=effective_excluded_roots,
        )
        baseline_aliases_prevalidated = True
    for task_dir in task_dirs:
        entry_id = _native_entry_id(task_dir)
        native_agent_workdir = _native_task_workdir(task_dir)
        _ensure_native_skills_dir(task_dir)
        entry = entries_by_id.get(entry_id)
        if grading_mode in ("default", "default_plus_custom") and entry is None:
            raise ValueError(f"Native Harbor task '{entry_id}' has no matching entry in evals/evals.json")

        _ensure_runtime_env_and_pre_agent_setup(
            task_dir,
            runtime_env=runtime_env,
            pre_agent_setup=pre_agent_setup,
        )

        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        _copy_verifier(task_dir)
        shutil.copy2(TEMPLATES_DIR / "custom_grader_runner.py", tests_dir / "custom_grader_runner.py")

        custom_grader = (tests_dir / "grader.py").exists() or (tests_dir / "grader.sh").exists()
        if (not custom_grader and grading_mode != "custom_only") or (
            not custom_grader and not (tests_dir / "test.sh").exists()
        ):
            custom_grader = _copy_custom_grader(task_dir, skill_path, grading_mode, evals_dir=evals_dir)

        if grading_mode in ("default", "default_plus_custom"):
            _ensure_skill_evaluator_verifier_env(
                task_dir,
                verifier_env=verifier_env if verifier_env is not None else runtime_env,
            )
            _write_entry_json(
                task_dir,
                entry or {"id": entry_id},
                with_skill,
                workspace_mode=workspace_mode,
                workspace_skill_names=workspace_skill_names,
                grading_mode=grading_mode,
                custom_grader=custom_grader,
            )
            _write_test_sh(task_dir, grading_mode=grading_mode, custom_grader=custom_grader)
        elif custom_grader:
            _write_entry_json(
                task_dir,
                entry or {"id": entry_id},
                with_skill,
                workspace_mode=workspace_mode,
                workspace_skill_names=workspace_skill_names,
                grading_mode=grading_mode,
                custom_grader=True,
            )
            _write_test_sh(task_dir, grading_mode=grading_mode, custom_grader=True)
        elif not (tests_dir / "test.sh").exists():
            raise FileNotFoundError(
                f"custom_only native Harbor task '{entry_id}' requires tests/grader.py or tests/test.sh"
            )

        if entry is not None and _entry_declares_task_input(entry, evals_dir / "files"):
            _stage_task_inputs(
                task_dir / "environment",
                input_files_dir=evals_dir / "files",
                entry=entry,
                source_skill_path=skill_path,
                evals_dir=evals_dir,
            )
        _prepare_native_environment(
            task_dir,
            skill_path=skill_path,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skill_paths,
            has_skill=with_skill,
            base_image=base_image,
            custom_dockerfile_mode=custom_dockerfile_mode,
            grading_mode=grading_mode,
            repo_context_mode="full" if copy_repo else "linked",
            compose_env_names=set(runtime_env or {}),
            repo_context_exclude_paths=effective_excluded_roots,
            agent_workdir=native_agent_workdir,
            baseline_aliases_prevalidated=baseline_aliases_prevalidated,
        )

    if not (output_dir / "dataset.toml").exists():
        _write_dataset_toml(output_dir, [p.name for p in task_dirs])
    _copy_metric_py(output_dir)
    return task_dirs


def _private_task_staging_parent(
    skill_path: Path,
    *,
    reference_skills_dir: Path | None,
    workspace_skill_paths: Sequence[Path],
) -> Path:
    """Choose a temp parent outside every runtime skill source.

    Preserve the normal system temporary location unless TMPDIR places task
    staging inside a runtime skill source. In that case, use a sibling of the
    target skill and ascend only when that location is itself inside a
    configured reference or workspace source.
    """
    external_sources = [*workspace_skill_paths]
    if reference_skills_dir is not None:
        external_sources.append(reference_skills_dir)
    candidate = Path(tempfile.gettempdir())
    if not any(_path_is_excluded(candidate, (source,)) for source in (skill_path, *external_sources)):
        return candidate

    candidate = skill_path.parent
    while candidate.parent != candidate and any(_path_is_excluded(candidate, (source,)) for source in external_sources):
        candidate = candidate.parent
    return candidate


def stage_native_harbor_tasks(
    skill_path: Path,
    output_dir: Path,
    *,
    with_skill: bool = True,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: list[Path] | None = None,
    workspace_mode: str = "isolated",
    grading_mode: str = "default",
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    copy_repo: bool = False,
    repo_context_exclude_paths: Sequence[Path] = (),
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
    evaluator_skill_path: Path | None = None,
    _baseline_alias_validation: _BaselineAliasValidation | None = None,
) -> list[Path]:
    """Stage native tasks privately, then publish one exact output snapshot."""

    if evaluator_skill_path is None:
        with private_evaluator_skill_snapshot(skill_path, task_source="native_harbor") as private_skill_path:
            return stage_native_harbor_tasks(
                skill_path,
                output_dir,
                with_skill=with_skill,
                reference_skills_dir=reference_skills_dir,
                workspace_skill_paths=workspace_skill_paths,
                workspace_mode=workspace_mode,
                grading_mode=grading_mode,
                base_image=base_image,
                custom_dockerfile_mode=custom_dockerfile_mode,
                copy_repo=copy_repo,
                repo_context_exclude_paths=repo_context_exclude_paths,
                runtime_env=runtime_env,
                verifier_env=verifier_env,
                pre_agent_setup=pre_agent_setup,
                task_resources=task_resources,
                agent_workdir=agent_workdir,
                evaluator_skill_path=private_skill_path,
                _baseline_alias_validation=_baseline_alias_validation,
            )

    baseline_aliases_prevalidated = False
    if not with_skill and _baseline_alias_validation is not None:
        if not _baseline_alias_validation_matches(
            _baseline_alias_validation,
            skill_path,
            reference_skills_dir,
            workspace_skill_paths,
            repo_context_exclude_paths,
        ):
            raise ValueError("Run-scoped baseline alias validation does not match the requested source set")
        baseline_aliases_prevalidated = True

    validate_output_provenance_key_location(
        skill_path,
        output_dir,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths or (),
    )
    _validate_staging_output_location(
        skill_path,
        output_dir,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths or (),
        declared_output_roots=repo_context_exclude_paths,
    )
    output_requires_provenance = _path_is_excluded(output_dir, (skill_path,))
    if output_requires_provenance:
        validate_generated_output_replacement(output_dir)

    private_staging_parent = _private_task_staging_parent(
        skill_path,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths or (),
    )
    with tempfile.TemporaryDirectory(
        prefix="skillevaluator-native-tasks-",
        dir=private_staging_parent,
    ) as temporary:
        private_root = Path(temporary).resolve(strict=True)
        private_output = private_root / "dataset"
        private_tasks = _stage_native_harbor_tasks_into(
            skill_path,
            private_output,
            evaluator_skill_path=evaluator_skill_path,
            with_skill=with_skill,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skill_paths,
            workspace_mode=workspace_mode,
            grading_mode=grading_mode,
            base_image=base_image,
            custom_dockerfile_mode=custom_dockerfile_mode,
            copy_repo=copy_repo,
            repo_context_exclude_paths=(
                *repo_context_exclude_paths,
                output_dir,
                private_root,
            ),
            private_repo_context_exclude_paths=(evaluator_skill_path.parent,),
            runtime_env=runtime_env,
            verifier_env=verifier_env,
            pre_agent_setup=pre_agent_setup,
            task_resources=task_resources,
            agent_workdir=agent_workdir,
            baseline_aliases_prevalidated=baseline_aliases_prevalidated,
        )
        relative_tasks = [task.relative_to(private_output) for task in private_tasks]
        if output_requires_provenance:
            write_generated_output_marker(private_output, destination=output_dir)
            validate_generated_output_replacement(output_dir)
        copytree_secure(
            private_output,
            output_dir,
            replace_existing=True,
            allowed_root=private_root,
        )
    return [output_dir / relative for relative in relative_tasks]


def _write_dataset_toml(output_dir: Path, task_dirs: list[str]) -> None:
    """Generate a minimal dataset.toml for the Harbor dataset."""
    lines = [
        "[dataset]",
        'name = "nvidia/skillevaluator"',
        'description = "SkillEvaluator skill evaluation dataset"',
        "",
    ]
    for task_name in sorted(task_dirs):
        lines.append("[[tasks]]")
        lines.append(f"name = {_toml_quote(f'nvidia/{task_name}')}")
        lines.append("")

    (output_dir / "dataset.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_metric_py(output_dir: Path) -> None:
    """Copy the custom metric.py for Harbor dataset-level aggregation."""
    src = TEMPLATES_DIR / "metric.py"
    if src.exists():
        shutil.copy2(src, output_dir / "metric.py")


def _generate_harbor_tasks_into(
    skill_path: Path,
    output_dir: Path,
    *,
    evaluator_skill_path: Path,
    with_skill: bool = True,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: list[Path] | None = None,
    workspace_mode: str = "isolated",
    grading_mode: str = "default",
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    copy_repo: bool = False,
    repo_context_exclude_paths: Sequence[Path] = (),
    private_repo_context_exclude_paths: Sequence[Path] = (),
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
    baseline_aliases_prevalidated: bool = False,
) -> list[Path]:
    """Generate Harbor task directories inside a private output directory.

    Args:
        skill_path: Path to the skill directory (contains SKILL.md, scripts/, evals/)
        output_dir: Where to write the Harbor dataset
        with_skill: If True, install the skill in the container. If False, baseline run.
        reference_skills_dir: Optional parent directory of reference/decoy
            skills to stage. Prefer ``workspace_skill_paths`` for new code.
        workspace_skill_paths: Explicit sibling/custom skills to stage into
            the agent workspace for this dataset.
        workspace_mode: ``isolated`` or ``group``; stored in entry metadata for
            workspace-aware scoring.
        grading_mode: ``default``, ``default_plus_custom``, or ``custom_only``.
        base_image: Pre-built Docker image tag to use as base (skips heavy pip
            installs in per-task Dockerfiles).  Empty string = full build.
        custom_dockerfile_mode: ``rebase`` replaces a valid custom Dockerfile's
            FROM with the eval base image. ``preserve`` keeps the custom FROM and
            appends verifier/runtime dependencies.
        runtime_env: Runtime environment variables to expose inside Harbor task
            environments using Harbor's native ``[environment.env]`` template.
        pre_agent_setup: Commands to run as a Harbor environment healthcheck
            before the agent starts.
        task_resources: Optional Harbor task ``[environment]`` resource values
            (``cpus``, ``memory_mb``, ``storage_mb``). Missing keys use SkillEvaluator
            defaults.
        agent_workdir: Optional default working directory for agent command
            execution inside the Harbor task environment.

    Returns:
        List of generated task directory paths.
    """
    _validate_runtime_discovery_env(runtime_env)
    _validate_runtime_loader_env(runtime_env)
    agent_workdir = _validated_agent_workdir(agent_workdir)
    evals_dir = evaluator_skill_path / "evals"
    evals_file = find_evals_file(evaluator_skill_path)
    _validate_staging_output_location(
        skill_path,
        output_dir,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths or (),
    )
    _validate_output_roots_outside_evaluator_sources(skill_path, repo_context_exclude_paths)

    if not evals_file:
        raise FileNotFoundError(f"No evals dataset found in {evals_dir}")

    entries = _load_evals(evals_file)
    if not entries:
        raise ValueError(f"Empty dataset: {evals_file}")
    workspace_skill_paths = workspace_skill_paths or []
    workspace_skill_names = sorted({p.name for p in workspace_skill_paths})

    input_files_dir = evals_dir / "files"
    if not input_files_dir.exists():
        input_files_dir = None

    mcp_servers = _load_mcp_servers(evaluator_skill_path)
    prepared_entries = _preflight_generated_tasks(entries, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dirs: list[str] = []
    task_paths: list[Path] = []
    effective_excluded_roots = (*repo_context_exclude_paths, *private_repo_context_exclude_paths)
    if not with_skill and not baseline_aliases_prevalidated:
        _check_baseline_skill_candidates_do_not_alias_target(
            skill_path,
            reference_skills_dir,
            workspace_skill_paths,
            excluded_roots=effective_excluded_roots,
        )
        baseline_aliases_prevalidated = True

    for normalized_entry, case_id in prepared_entries:
        task_dir = safe_child(output_dir, case_id)

        if os.path.lexists(task_dir):
            if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
                raise ValueError(
                    f"cannot safely replace generated task {case_id!r}: "
                    "this platform does not provide symlink-attack-resistant recursive deletion"
                )
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True)

        _write_instruction(task_dir, normalized_entry.get("question", ""))
        _write_task_toml(
            task_dir,
            normalized_entry,
            with_skill,
            mcp_servers=mcp_servers,
            runtime_env=runtime_env,
            verifier_env=verifier_env,
            pre_agent_setup=pre_agent_setup,
            task_resources=task_resources,
            agent_workdir=agent_workdir,
        )
        _copy_verifier(task_dir)
        custom_grader = _copy_custom_grader(task_dir, skill_path, grading_mode, evals_dir=evals_dir)
        _write_entry_json(
            task_dir,
            normalized_entry,
            with_skill,
            workspace_mode=workspace_mode,
            workspace_skill_names=workspace_skill_names,
            grading_mode=grading_mode,
            custom_grader=custom_grader,
        )
        _write_test_sh(task_dir, grading_mode=grading_mode, custom_grader=custom_grader)

        _write_dockerfile(
            task_dir,
            skill_path=skill_path,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skill_paths,
            has_skill=with_skill,
            input_files_dir=input_files_dir,
            entry=normalized_entry,
            evals_dir=evals_dir,
            exclude_skill_name=skill_path.name,
            base_image=base_image,
            custom_dockerfile_mode=custom_dockerfile_mode,
            repo_context_skill_path=skill_path,
            repo_context_mode="full" if copy_repo else "linked",
            compose_env_names=set(runtime_env or {}),
            repo_context_exclude_paths=effective_excluded_roots,
            agent_workdir=agent_workdir,
            baseline_aliases_prevalidated=baseline_aliases_prevalidated,
        )

        task_dirs.append(case_id)
        task_paths.append(task_dir)
        logger.debug("Generated task: %s (has_skill=%s)", case_id, with_skill)

    _write_dataset_toml(output_dir, task_dirs)
    _copy_metric_py(output_dir)

    logger.debug(
        "Generated %d Harbor tasks in %s (with_skill=%s)",
        len(task_paths),
        output_dir,
        with_skill,
    )
    return task_paths


@dataclass(frozen=True)
class _EvaluatorProjectionSelection:
    """Evaluator paths selected before the single secure-copy transaction."""

    exact_paths: frozenset[tuple[str, ...]]
    whole_roots: frozenset[tuple[str, ...]]
    signature: tuple[Any, ...]


def _projection_path_exists(path: Path) -> bool:
    """Check lexical presence without silently dropping a broken link."""
    return os.path.lexists(path)


def _selected_environment_projection(evals_dir: Path) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Select the custom environment as one compatibility-preserving unit."""
    exact: set[tuple[str, ...]] = set()
    whole: set[tuple[str, ...]] = set()
    environment = evals_dir / "environment"
    if not _projection_path_exists(environment):
        return exact, whole
    # Custom Docker handling first creates a secure copy of this complete tree
    # before it validates Dockerfile/Compose and selects sidecars. Keeping the
    # directory whole preserves authored environment behavior while unrelated
    # evals/ siblings remain outside the snapshot.
    whole.add(("environment",))
    return exact, whole


def _selected_grader_paths(evals_dir: Path) -> tuple[tuple[str, ...], ...]:
    candidates = (("grader.py",), ("grader.sh",), ("tests", "grader.py"), ("tests", "grader.sh"))
    return tuple(relative for relative in candidates if _projection_path_exists(evals_dir.joinpath(*relative)))


def _resolved_projection_task_source(
    requested_task_source: str | None,
    evals_config: dict[str, Any],
    *,
    dataset_path: Path | None,
    native_dir: Path,
) -> str:
    configured = requested_task_source
    if configured is None:
        harbor_config = evals_config.get("harbor", {})
        configured = harbor_config.get("task_source", "auto") if isinstance(harbor_config, dict) else "auto"
    if configured not in {"auto", "evals_json", "native_harbor"}:
        configured = "auto"
    if configured == "auto":
        if dataset_path is not None:
            return "evals_json"
        if native_dir.exists():
            return "native_harbor"
    return configured


def _native_projection_entry_ids(native_dir: Path) -> tuple[str, ...]:
    """Read native task IDs without following unsafe authored nodes.

    The secure projection copy validates the complete selected ``harbor/`` tree
    as one transaction. Selection only needs immediate task identities so it
    can avoid fixtures referenced exclusively by generated entries.
    """
    try:
        native_metadata = native_dir.lstat()
        parent_metadata = native_dir.parent.lstat()
    except OSError:
        return ()
    if (
        _path_is_link_or_reparse(native_dir, native_metadata)
        or not stat.S_ISDIR(native_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or native_metadata.st_dev != parent_metadata.st_dev
    ):
        return ()
    try:
        children = sorted(native_dir.iterdir())
    except OSError:
        return ()
    ignored = set(_NATIVE_SOURCE_IGNORE(os.fspath(native_dir), [path.name for path in children]))
    entry_ids: list[str] = []
    for task_dir in children:
        if task_dir.name in ignored or task_dir.name.startswith((".", "_")):
            continue
        try:
            task_metadata = task_dir.lstat()
        except OSError:
            continue
        if (
            _path_is_link_or_reparse(task_dir, task_metadata)
            or not stat.S_ISDIR(task_metadata.st_mode)
            or task_metadata.st_dev != native_metadata.st_dev
        ):
            continue
        task_toml = task_dir / "task.toml"
        if not _projection_path_exists(task_toml):
            continue
        entry_id = task_dir.name
        try:
            payload = _read_regular_evals_file(
                task_toml,
                label="Native Harbor task configuration",
                allowed_root=native_dir,
            )
            data = tomllib.loads(payload.decode("utf-8"))
            metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
            if isinstance(metadata, dict) and metadata.get("entry_id"):
                entry_id = str(metadata["entry_id"])
        except (UnicodeError, ValueError):
            # Staging owns the detailed invalid-task diagnostic after the secure
            # snapshot exists. Directory-name fallback remains non-invasive.
            pass
        entry_ids.append(entry_id)
    return tuple(sorted(entry_ids))


def _decode_projection_config(payload: bytes) -> tuple[dict[str, Any], bool]:
    """Decode only the source-selection hint from already-secured config bytes."""
    import yaml

    try:
        raw = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError):
        return {}, False
    if not isinstance(raw, dict):
        return {}, False
    harbor = raw.get("harbor")
    if harbor is not None and not isinstance(harbor, dict):
        return {}, False
    return raw, True


def _evaluator_projection_selection(
    skill_path: Path,
    *,
    task_source: str | None,
    bind_full_evidence_sources: bool,
) -> _EvaluatorProjectionSelection:
    """Resolve the evaluator projection and a post-copy selection signature."""
    import yaml

    from skillevaluator.tier3.dataset_utils import normalize_dataset_entries
    from skillevaluator.tier3.evals_config import CONFIG_FILENAMES

    evals_dir = skill_path / "evals"
    config_path = next(
        (evals_dir / name for name in CONFIG_FILENAMES if _projection_path_exists(evals_dir / name)),
        None,
    )
    evals_config: dict[str, Any] = {}
    config_valid = True
    config_digest: str | None = None
    if config_path is not None:
        config_payload = _read_regular_evals_file(
            config_path,
            label="Eval configuration",
            max_bytes=_MAX_EVALS_CONFIG_BYTES,
            allowed_root=evals_dir,
        )
        config_digest = hashlib.sha256(config_payload).hexdigest()
        evals_config, config_valid = _decode_projection_config(config_payload)

    dataset_path = find_evals_file(skill_path)
    dataset_valid = True
    entries: list[dict[str, Any]] = []
    dataset_digest: str | None = None
    if dataset_path is not None:
        dataset_payload = _read_regular_evals_file(dataset_path, allowed_root=evals_dir)
        dataset_digest = hashlib.sha256(dataset_payload).hexdigest()
        try:
            decoded = dataset_payload.decode("utf-8")
            suffix = dataset_path.suffix.casefold()
            if suffix == ".jsonl":
                raw_entries: Any = [json.loads(line) for raw_line in decoded.splitlines() if (line := raw_line.strip())]
            elif suffix in {".yaml", ".yml"}:
                raw_entries = yaml.safe_load(decoded)
            elif suffix == ".json":
                raw_entries = json.loads(decoded)
            else:
                raise ValueError(f"Unsupported dataset format: {suffix}")
            entries = normalize_dataset_entries(raw_entries)
        except (UnicodeError, TypeError, ValueError, yaml.YAMLError):
            # Invalid datasets do not have a meaningful fixture selection. Copy
            # the dataset itself and let normal task staging report its error.
            dataset_valid = False

    resolved_task_source = _resolved_projection_task_source(
        task_source,
        evals_config,
        dataset_path=dataset_path,
        native_dir=evals_dir / "harbor",
    )
    native_entry_ids: tuple[str, ...] = ()
    native_dir = evals_dir / "harbor"
    if resolved_task_source == "native_harbor" and _projection_path_exists(native_dir):
        native_entry_ids = _native_projection_entry_ids(native_dir)
    exact_paths: set[tuple[str, ...]] = set()
    whole_roots: set[tuple[str, ...]] = set()
    if config_path is not None:
        exact_paths.add((config_path.name,))
    if dataset_path is not None:
        exact_paths.add((dataset_path.name,))
    if resolved_task_source == "native_harbor" and _projection_path_exists(evals_dir / "harbor"):
        whole_roots.add(("harbor",))

    environment_exact, environment_whole = _selected_environment_projection(evals_dir)
    exact_paths.update(environment_exact)
    whole_roots.update(environment_whole)

    available_grader_paths = _selected_grader_paths(evals_dir)
    grader_paths = (
        available_grader_paths
        if bind_full_evidence_sources and resolved_task_source == "evals_json"
        else available_grader_paths[:1]
    )
    exact_paths.update(grader_paths)

    fixture_signature: tuple[Any, ...] = ("invalid-or-absent-dataset",)
    if dataset_path is not None and dataset_valid:
        files_dir = evals_dir / "files"
        native_entry_id_set = set(native_entry_ids)
        fixture_entries = (
            [entry for entry in entries if str(entry.get("id")) in native_entry_id_set]
            if resolved_task_source == "native_harbor"
            else entries
        )
        implicit_shared_files = (
            (bind_full_evidence_sources and resolved_task_source == "evals_json")
            or any("files" not in entry for entry in fixture_entries)
        ) and _projection_path_exists(files_dir)
        selected_refs: set[tuple[str, ...]] = set()
        if implicit_shared_files:
            whole_roots.add(("files",))
        for entry in fixture_entries:
            for ref in _entry_file_refs(entry):
                source, _staged_relative = _resolve_entry_file_ref(
                    ref,
                    skill_path=skill_path,
                    evals_dir=evals_dir,
                    input_files_dir=files_dir if files_dir.exists() else None,
                )
                relative = source.absolute().relative_to(evals_dir.absolute()).parts
                try:
                    source_metadata = source.lstat()
                except OSError:
                    exact_paths.add(relative)
                    selected_refs.add(relative)
                    continue
                if stat.S_ISDIR(source_metadata.st_mode) and not _path_is_link_or_reparse(source, source_metadata):
                    whole_roots.add(relative)
                else:
                    exact_paths.add(relative)
                selected_refs.add(relative)
        fixture_signature = (
            "all-files" if implicit_shared_files else "declared-only",
            tuple(sorted(selected_refs)),
        )

    signature = (
        config_path.name if config_path is not None else None,
        config_valid,
        config_digest,
        resolved_task_source,
        dataset_path.name if dataset_path is not None else None,
        dataset_valid,
        dataset_digest,
        bind_full_evidence_sources,
        native_entry_ids,
        fixture_signature,
        grader_paths,
        tuple(sorted(environment_exact)),
        tuple(sorted(environment_whole)),
    )
    return _EvaluatorProjectionSelection(
        exact_paths=frozenset(exact_paths),
        whole_roots=frozenset(whole_roots),
        signature=signature,
    )


def _projection_includes(
    relative: tuple[str, ...],
    *,
    exact_paths: frozenset[tuple[str, ...]],
    whole_roots: frozenset[tuple[str, ...]],
) -> bool:
    if any(relative == exact[: len(relative)] for exact in exact_paths):
        return True
    return any(relative == whole[: len(relative)] or whole == relative[: len(whole)] for whole in whole_roots)


@contextmanager
def private_evaluator_skill_snapshot(
    skill_path: Path,
    *,
    task_source: str | None = None,
    bind_full_evidence_sources: bool = False,
) -> Iterator[Path]:
    """Yield one selective, immutable evaluator projection for a complete run."""
    source_evals_dir = skill_path / "evals"
    _validate_evals_source_directory(skill_path)
    if not source_evals_dir.is_dir():
        raise FileNotFoundError(f"No evaluator source directory found at {source_evals_dir}")

    selection = _evaluator_projection_selection(
        skill_path,
        task_source=task_source,
        bind_full_evidence_sources=bind_full_evidence_sources,
    )
    source_evals_canonical = source_evals_dir.resolve(strict=True)

    with tempfile.TemporaryDirectory(prefix="skillevaluator-evals-snapshot-") as snapshot_root_text:
        snapshot_root = Path(snapshot_root_text)
        evaluator_skill_path = snapshot_root / skill_path.name
        evaluator_skill_path.mkdir()
        try:
            snapshot_source_relative = snapshot_root.resolve(strict=True).relative_to(source_evals_canonical).parts
        except (OSError, ValueError):
            snapshot_source_relative = ()

        def _ignore_unselected_evaluator_paths(current: str, names: list[str]) -> set[str]:
            try:
                current_relative = Path(current).resolve(strict=True).relative_to(source_evals_canonical).parts
            except (OSError, ValueError):
                return set(names)
            ignored = {
                name
                for name in names
                if not _projection_includes(
                    (*current_relative, name),
                    exact_paths=selection.exact_paths,
                    whole_roots=selection.whole_roots,
                )
            }
            if snapshot_source_relative:
                ignored.update(name for name in names if (*current_relative, name) == snapshot_source_relative)
            if current_relative and current_relative[0].casefold() == "harbor":
                ignored.update(_NATIVE_SOURCE_IGNORE(current, names))
            return ignored

        source_content_fingerprint = tree_content_fingerprint_secure(
            source_evals_dir,
            ignore=_ignore_unselected_evaluator_paths,
            allowed_root=source_evals_dir,
        )

        copytree_secure(
            source_evals_dir,
            evaluator_skill_path / "evals",
            ignore=_ignore_unselected_evaluator_paths,
            allowed_root=source_evals_dir,
        )
        copied_selection = _evaluator_projection_selection(
            evaluator_skill_path,
            task_source=task_source,
            bind_full_evidence_sources=bind_full_evidence_sources,
        )
        if copied_selection.signature != selection.signature:
            raise ValueError("Evaluator source selection changed while its private snapshot was created")
        copied_content_fingerprint = tree_content_fingerprint_secure(evaluator_skill_path / "evals")
        if copied_content_fingerprint != source_content_fingerprint:
            raise ValueError("Evaluator source selection changed while its private snapshot was created")
        yield evaluator_skill_path


def generate_harbor_tasks(
    skill_path: Path,
    output_dir: Path,
    *,
    with_skill: bool = True,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: list[Path] | None = None,
    workspace_mode: str = "isolated",
    grading_mode: str = "default",
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    copy_repo: bool = False,
    repo_context_exclude_paths: Sequence[Path] = (),
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
    evaluator_skill_path: Path | None = None,
    _baseline_alias_validation: _BaselineAliasValidation | None = None,
) -> list[Path]:
    """Generate tasks from one private evals snapshot, then publish exactly."""

    if evaluator_skill_path is None:
        if find_evals_file(skill_path) is None:
            raise FileNotFoundError(f"No evals dataset found in {skill_path / 'evals'}")
        with private_evaluator_skill_snapshot(skill_path, task_source="evals_json") as private_skill_path:
            return generate_harbor_tasks(
                skill_path,
                output_dir,
                with_skill=with_skill,
                reference_skills_dir=reference_skills_dir,
                workspace_skill_paths=workspace_skill_paths,
                workspace_mode=workspace_mode,
                grading_mode=grading_mode,
                base_image=base_image,
                custom_dockerfile_mode=custom_dockerfile_mode,
                copy_repo=copy_repo,
                repo_context_exclude_paths=repo_context_exclude_paths,
                runtime_env=runtime_env,
                verifier_env=verifier_env,
                pre_agent_setup=pre_agent_setup,
                task_resources=task_resources,
                agent_workdir=agent_workdir,
                evaluator_skill_path=private_skill_path,
                _baseline_alias_validation=_baseline_alias_validation,
            )
    if find_evals_file(evaluator_skill_path) is None:
        raise FileNotFoundError(f"No evals dataset found in {evaluator_skill_path / 'evals'}")

    baseline_aliases_prevalidated = False
    if not with_skill and _baseline_alias_validation is not None:
        if not _baseline_alias_validation_matches(
            _baseline_alias_validation,
            skill_path,
            reference_skills_dir,
            workspace_skill_paths,
            repo_context_exclude_paths,
        ):
            raise ValueError("Run-scoped baseline alias validation does not match the requested source set")
        baseline_aliases_prevalidated = True

    validate_output_provenance_key_location(
        skill_path,
        output_dir,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths or (),
    )
    _validate_staging_output_location(
        skill_path,
        output_dir,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths or (),
        declared_output_roots=repo_context_exclude_paths,
    )
    output_requires_provenance = _path_is_excluded(output_dir, (skill_path,))
    if output_requires_provenance:
        validate_generated_output_replacement(output_dir)

    private_staging_parent = _private_task_staging_parent(
        skill_path,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths or (),
    )
    with tempfile.TemporaryDirectory(
        prefix="skillevaluator-generated-tasks-",
        dir=private_staging_parent,
    ) as temporary:
        private_root = Path(temporary).resolve(strict=True)
        private_output = private_root / "dataset"
        private_tasks = _generate_harbor_tasks_into(
            skill_path,
            private_output,
            evaluator_skill_path=evaluator_skill_path,
            with_skill=with_skill,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skill_paths,
            workspace_mode=workspace_mode,
            grading_mode=grading_mode,
            base_image=base_image,
            custom_dockerfile_mode=custom_dockerfile_mode,
            copy_repo=copy_repo,
            repo_context_exclude_paths=(
                *repo_context_exclude_paths,
                output_dir,
                private_root,
            ),
            private_repo_context_exclude_paths=(evaluator_skill_path.parent,),
            runtime_env=runtime_env,
            verifier_env=verifier_env,
            pre_agent_setup=pre_agent_setup,
            task_resources=task_resources,
            agent_workdir=agent_workdir,
            baseline_aliases_prevalidated=baseline_aliases_prevalidated,
        )
        relative_tasks = [task.relative_to(private_output) for task in private_tasks]
        if output_requires_provenance:
            write_generated_output_marker(private_output, destination=output_dir)
            validate_generated_output_replacement(output_dir)
        copytree_secure(
            private_output,
            output_dir,
            replace_existing=True,
            allowed_root=private_root,
        )
    return [output_dir / relative for relative in relative_tasks]
