# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- standalone trusted CI helper.
"""Select skill trees using two explicit Git commits, without executing Git hooks.

A rename is deliberately represented as a deletion plus an addition so both
skill trees are considered. Removing a manifest while leaving tracked content
is an error; removing the entire skill is recorded as a deletion, not a pass.
"""

from __future__ import annotations

import os
import re
import selectors
import subprocess
import time
from pathlib import Path, PurePosixPath

MAX_GIT_BYTES = 16 * 1024 * 1024
MAX_FILES = 100_000
MAX_DISCOVERED_SKILLS = 10_000
MAX_SELECTED_SKILLS = 100
GIT_TIMEOUT = 30
_SHA = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")


class SelectionError(ValueError):
    """A selection cannot be established from the supplied checkout."""

    def __init__(self, message: str, *, selection: dict | None = None):
        super().__init__(message)
        self.selection = selection


def _git(workspace: Path, *arguments: str) -> bytes:
    # Do not inherit GIT_DIR, GIT_CONFIG_COUNT, external diff commands or user
    # config. The checkout is read-only and optional index updates are disabled.
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_OPTIONAL_LOCKS="0")
    command = [
        "git",
        "--literal-pathspecs",
        "-C",
        str(workspace),
        "-c",
        f"safe.directory={workspace}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.allow=never",
        *arguments,
    ]
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env) as process:
            assert process.stdout is not None
            chunks = []
            size = 0
            deadline = time.monotonic() + GIT_TIMEOUT
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise SelectionError("Git selection exceeded its time limit.")
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_GIT_BYTES:
                            raise SelectionError("Git selection exceeded its output limit.")
                        chunks.append(chunk)
                if process.wait(timeout=max(0.1, deadline - time.monotonic())):
                    raise SelectionError(
                        "Git could not resolve the requested selection; verify checkout history and commits."
                    )
            except BaseException:
                process.kill()
                process.wait()
                raise
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SelectionError("Git selection could not run or timed out; no full scan was substituted.") from error
    return b"".join(chunks)


def _paths(output: bytes) -> list[str]:
    if output and not output.endswith(b"\0"):
        raise SelectionError("Git returned an incomplete path list.")
    values = output.split(b"\0")[:-1] if output else []
    if len(values) > MAX_FILES:
        raise SelectionError("Git selection exceeded its file count limit.")
    try:
        paths = [value.decode("utf-8", errors="strict") for value in values]
    except UnicodeError as error:
        raise SelectionError("Git selection requires UTF-8 paths.") from error
    for path in paths:
        if (
            not path
            or len(path.encode()) > 4096
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or any(ord(character) < 32 or ord(character) == 127 for character in path)
        ):
            raise SelectionError("Git returned an unsupported or unsafe path.")
    return paths


def _roots(workspace: Path, configured: str) -> list[str]:
    values = [line.strip() for line in configured.splitlines() if line.strip()]
    if not values or len(values) > 100 or len(configured) > 16384:
        raise SelectionError("SKILL_PATHS must specify between 1 and 100 bounded relative directories.")
    roots = set()
    for value in values:
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in value
            or ":" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise SelectionError("Skill roots must be relative directories without parent traversal.")
        current = workspace
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise SelectionError("Symlinks are not supported in skill roots.")
        if current.exists() and not current.is_dir():
            raise SelectionError("A configured skill root is not a directory.")
        roots.add(relative.as_posix())
    return sorted(roots)


def _within(path: str, directory: str) -> bool:
    return directory in {".", path} or path.startswith(directory + "/")


def _manifests(paths: list[str]) -> set[str]:
    found = {PurePosixPath(path).parent.as_posix() for path in paths if PurePosixPath(path).name == "SKILL.md"}
    if len(found) > MAX_DISCOVERED_SKILLS:
        raise SelectionError("Git selection exceeded its discovered skill limit.")
    return found


def select_changed_skills(workspace: Path, roots: str, base: str, head: str) -> tuple[list[Path], dict]:
    """Return existing changed skills and a bounded, deterministic audit record.

    Commits must be full object IDs and the working tree must match ``head``.
    The unique merge base defines the PR delta. No network fetch or full-scan fallback is performed.
    """
    workspace = workspace.resolve(strict=True)
    configured = _roots(workspace, roots)
    if not isinstance(base, str) or not isinstance(head, str) or not _SHA.fullmatch(base) or not _SHA.fullmatch(head):
        raise SelectionError("Changed selection requires full DIFF_BASE_SHA and DIFF_HEAD_SHA commit IDs.")
    top = _git(workspace, "rev-parse", "--show-toplevel").decode().strip()
    if Path(top).resolve(strict=True) != workspace:
        raise SelectionError("The workspace must be the root of its Git checkout.")
    requested_base, requested_head = base.lower(), head.lower()
    base = _git(workspace, "rev-parse", "--verify", base + "^{commit}").decode().strip()
    head = _git(workspace, "rev-parse", "--verify", head + "^{commit}").decode().strip()
    if base != requested_base or head != requested_head:
        raise SelectionError("DIFF_BASE_SHA and DIFF_HEAD_SHA must identify commits, not tag objects.")
    if _git(workspace, "rev-parse", "HEAD").decode().strip() != head:
        raise SelectionError("The checked-out HEAD does not match DIFF_HEAD_SHA; no scan was substituted.")
    base_ref = base
    try:
        bases = _git(workspace, "merge-base", "--all", base_ref, head).decode().splitlines()
    except SelectionError as error:
        raise SelectionError(
            "The requested commits have no available common merge base; no full scan was substituted."
        ) from error
    if len(bases) != 1 or not _SHA.fullmatch(bases[0]):
        raise SelectionError(
            "A unique Git merge base is required; ambiguous or missing history cannot select PR skills."
        )
    base = bases[0]
    if _git(
        workspace, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching", "--", *configured
    ):
        raise SelectionError(
            "Configured skill roots contain modified, untracked or ignored content; checkout must match the selected head."
        )
    before_files = _paths(_git(workspace, "ls-tree", "-r", "-z", "--name-only", base, "--", *configured))
    after_files = _paths(_git(workspace, "ls-tree", "-r", "-z", "--name-only", head, "--", *configured))
    for root in configured:
        if not (workspace / root).is_dir() and not any(_within(path, root) for path in before_files):
            raise SelectionError("A configured skill root is missing from both the checkout and base revision.")
    before = _manifests(before_files)
    after = _manifests(after_files)
    if not before and not after:
        raise SelectionError("No SKILL.md files exist in the configured roots at either commit.")
    changed = _paths(
        _git(
            workspace,
            "diff",
            "--name-only",
            "-z",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--ignore-submodules=all",
            base,
            head,
            "--",
            *configured,
        )
    )
    removed = sorted(before - after)
    deleted = [directory for directory in removed if not any(_within(path, directory) for path in after_files)]
    selected = sorted(directory for directory in after if any(_within(path, directory) for path in changed))
    selection = {
        "mode": "changed",
        "base": base,
        "base_ref": base_ref,
        "head": head,
        "discovered": len(after),
        "selected": selected,
        "deleted": deleted,
        "changed_files": len(changed),
    }
    invalid = [directory for directory in removed if directory not in deleted]
    if invalid:
        names = ", ".join(repr(directory) for directory in invalid[:5])
        raise SelectionError(
            f"Deleted SKILL.md left tracked content in {names}. Restore the manifest or remove the entire skill directory.",
            selection=selection,
        )
    if len(selected) > MAX_SELECTED_SKILLS:
        raise SelectionError("The CI integration supports at most 100 selected skills per run.", selection=selection)
    return [workspace / directory for directory in selected], selection
