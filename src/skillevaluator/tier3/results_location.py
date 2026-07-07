# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared result-location helpers for local skill evaluation commands."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ENV_RESULTS_DIR = "SKILLEVALUATOR_RESULTS_DIR"
_RUN_TIMESTAMP_FORMATS = ("%Y%m%d_%H%M%S", "%Y-%m-%d_%H%M%S")


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def legacy_results_root(skill_path: Path) -> Path:
    """Return the historical in-skill results directory."""
    return skill_path.expanduser().resolve() / "evals" / "results"


def skill_results_name(skill_path: Path) -> str:
    """Return the directory name used under an external results root."""
    return skill_path.expanduser().resolve().name


def external_results_root(root: str | Path, skill_path: Path) -> Path:
    """Return ``<root>/<skill-name>`` for a global or CLI results root."""
    return _expand(root) / skill_results_name(skill_path)


def env_results_root(skill_path: Path, *, environ: dict[str, str] | None = None) -> Path | None:
    """Return the env-configured results root for a skill, if configured."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_RESULTS_DIR)
    if not raw:
        return None
    return external_results_root(raw, skill_path)


def resolve_results_root(
    skill_path: Path,
    cli_results_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve the per-skill results root for writes.

    Precedence:
    1. command ``--results-dir`` root
    2. ``SKILLEVALUATOR_RESULTS_DIR`` root
    3. legacy ``<skill>/evals/results``
    """
    if cli_results_dir is not None:
        return external_results_root(cli_results_dir, skill_path)
    configured = env_results_root(skill_path, environ=environ)
    if configured is not None:
        return configured
    return legacy_results_root(skill_path)


def iter_candidate_results_roots(
    skill_path: Path,
    cli_results_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> list[Path]:
    """Return read candidates in precedence order, with legacy fallback.

    Read commands should honor the same primary resolution as write commands,
    but falling back to the legacy location avoids hiding old runs when a user
    has newly configured ``SKILLEVALUATOR_RESULTS_DIR``.
    """
    roots: list[Path] = []
    if cli_results_dir is not None:
        roots.append(external_results_root(cli_results_dir, skill_path))
        configured = env_results_root(skill_path, environ=environ)
        if configured is not None and configured not in roots:
            roots.append(configured)
    else:
        roots.append(resolve_results_root(skill_path, environ=environ))
    legacy = legacy_results_root(skill_path)
    if legacy not in roots:
        roots.append(legacy)
    return roots


def _run_timestamp(name: str) -> datetime | None:
    for timestamp_format in _RUN_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(name, timestamp_format)  # noqa: DTZ007 -- directory names have no timezone
        except ValueError:
            continue
    return None


def _newest_completed_run(root: Path) -> Path | None:
    """Return the newest complete timestamped run without relying on symlinks."""
    try:
        children = root.iterdir()
    except OSError:
        return None

    completed: list[tuple[datetime, Path]] = []
    try:
        for candidate in children:
            timestamp = _run_timestamp(candidate.name)
            if timestamp is None or candidate.name.startswith((".", "_")) or candidate.is_symlink():
                continue
            try:
                if not candidate.is_dir():
                    continue
                result = json.loads((candidate / "result.json").read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
            if isinstance(result, dict) and result.get("run_id") == candidate.name:
                completed.append((timestamp, candidate))
    except OSError:
        return None
    return max(completed, default=(None, None), key=lambda item: item[0])[1]


def resolve_latest_results(
    skill_path: Path,
    cli_results_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Return the best available ``latest`` results path for read workflows."""
    roots = iter_candidate_results_roots(
        skill_path,
        cli_results_dir,
        environ=environ,
    )
    for root in roots:
        latest = root / "latest"
        if latest.exists():
            return latest
        fallback = _newest_completed_run(root)
        if fallback is not None:
            return fallback
    return roots[0] / "latest"


def resolve_explicit_or_latest_results(
    skill_path: Path,
    from_results: str | Path | None = None,
    cli_results_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve a specific run path or the latest run for refinement/reporting."""
    if from_results is not None:
        return _expand(from_results)
    return resolve_latest_results(skill_path, cli_results_dir, environ=environ)


def git_root_for(path: Path) -> Path | None:
    """Return the containing git repo root for ``path``, if it is in a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path.expanduser().resolve()), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root).resolve() if root else None


def gitignore_entry_for_skill_results(skill_path: Path, repo_root: Path | None = None) -> str | None:
    """Return a repo-root-relative ignore entry for a skill's generated results."""
    skill_path = skill_path.expanduser().resolve()
    repo_root = repo_root or git_root_for(skill_path)
    if repo_root is None:
        return None
    try:
        rel = legacy_results_root(skill_path).relative_to(repo_root)
    except ValueError:
        return None
    return f"/{rel.as_posix()}/"


def ensure_skill_results_gitignore(skill_path: Path) -> tuple[Path | None, str | None, bool]:
    """Ensure the legacy in-repo results directory is ignored.

    Returns ``(.gitignore path, entry, changed)``. If the skill is not inside a
    git repository, returns ``(None, None, False)``.
    """
    repo_root = git_root_for(skill_path)
    entry = gitignore_entry_for_skill_results(skill_path, repo_root=repo_root)
    if repo_root is None or entry is None:
        return None, None, False

    gitignore_path = repo_root / ".gitignore"
    try:
        text = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    except OSError:
        return gitignore_path, entry, False

    lines = {line.strip() for line in text.splitlines()}
    normalized_lines = {line.lstrip("/") for line in lines}
    if entry in lines or entry.lstrip("/") in normalized_lines:
        return gitignore_path, entry, False

    suffix = "" if not text or text.endswith("\n") else "\n"
    gitignore_path.write_text(f"{text}{suffix}{entry}\n", encoding="utf-8")
    return gitignore_path, entry, True
