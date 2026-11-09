# SPDX-License-Identifier: Apache-2.0
"""Git discovery and subprocess helpers for repository onboarding."""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import yaml

EXCLUDED = {
    "test",
    "tests",
    "fixture",
    "fixtures",
    "example",
    "examples",
    "vendor",
    "vendored",
    "node_modules",
    ".venv",
}

OWNER_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")


class SetupError(Exception):
    """A safely described setup failure."""


def run(args: list[str], *, cwd: Path | None = None, optional: bool = False, timeout: int = 60) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    if args[0] in {"gh", "git"}:
        env["GH_HOST"] = "github.com"
        env.pop("GH_REPO", None)
    try:
        result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SetupError(f"Could not run {args[0]}; check that it is installed and responsive.") from exc
    if result.returncode and not optional:
        # Do not echo subprocess stderr: authentication helpers can include credentials.
        raise SetupError(f"{args[0]} could not complete this step. Check authentication, access and repository state.")
    return result.stdout if not result.returncode else ""


def git(repo: Path, *args: str, optional: bool = False, network: bool = False) -> str:
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false"]
    if not network:
        command += ["-c", "protocol.allow=never"]
    else:
        # Accepted SSH origins are normalized to HTTPS for API and push parity.
        # Use the authenticated gh account without changing the user's Git config.
        command += [
            "-c",
            "credential.helper=",
            "-c",
            "credential.https://github.com.helper=!gh auth git-credential",
        ]
    try:
        return run([*command, *args], cwd=repo, optional=optional)
    except SetupError as exc:
        if network and args and args[0] == "push":
            raise SetupError(
                f"{exc} Pushing workflow files also requires workflow write permission. "
                "For a GitHub CLI OAuth login, run: gh auth refresh -h github.com -s workflow"
            ) from exc
        raise


def repository_root(path: Path) -> Path:
    repo = path.resolve(strict=True)
    top = Path(git(repo, "rev-parse", "--show-toplevel").strip()).resolve(strict=True)
    if top != repo:
        raise SetupError("Run from the Git repository root, or pass that root using --repo.")
    return repo


def parse_origin[T](repo: Path, parse: Callable[[str], T]) -> T:
    """Parse origin as configured, then as rewritten by Git's url.<base>.insteadOf rules.

    `git remote get-url` applies insteadOf rewrites, so a user-level rule such as
    https://github.com/ -> ssh://git@github.com/ would hide a supported origin, while
    an alias origin such as gh:owner/repo only parses after the rewrite.
    """
    configured = git(repo, "config", "--get", "remote.origin.url", optional=True).strip()
    rewritten = git(repo, "remote", "get-url", "origin").strip()
    first_error: SetupError | None = None
    for value in dict.fromkeys(v for v in (configured, rewritten) if v):
        try:
            return parse(value)
        except SetupError as exc:
            first_error = first_error or exc
    raise first_error or SetupError("The repository has no origin remote.")


def parse_remote(value: str) -> str:
    if value.startswith("git@github.com:"):
        name = value.removeprefix("git@github.com:")
    else:
        parts = urlsplit(value)
        if (
            parts.scheme != "https"
            or parts.hostname != "github.com"
            or parts.username
            or parts.password
            or parts.port
            or parts.query
            or parts.fragment
        ):
            raise SetupError(
                "CI onboarding supports a credential-free github.com HTTPS or git@github.com SSH origin only."
            )
        name = parts.path.removeprefix("/")
    name = name.removesuffix(".git")
    if not OWNER_REPO.fullmatch(name):
        raise SetupError("The origin does not identify a supported GitHub repository.")
    return name


def safe_relative(value: str) -> str:
    if value == ".":
        return value
    p = PurePosixPath(value)
    if (
        not value
        or p.is_absolute()
        or any(x in {"", ".", ".."} for x in value.split("/"))
        or "\\" in value
        or ":" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise SetupError("Skill paths must be plain repository-relative directories.")
    return p.as_posix()


def no_symlinks(repo: Path, relative: str) -> None:
    import stat

    current = repo
    for part in PurePosixPath(relative).parts:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(metadata.st_mode) or reparse:
            raise SetupError("Symlink and Windows junction paths are not supported by CI onboarding.")


def discover(repo: Path) -> list[dict]:
    output = git(repo, "ls-files", "-z", "--cached")
    if len(output.encode()) > 16 * 1024 * 1024:
        raise SetupError("Repository discovery exceeded the discovery size limit.")
    manifests = []
    for path in output.split("\0"):
        if not path or PurePosixPath(path).name != "SKILL.md":
            continue
        safe_relative(path)
        no_symlinks(repo, path)
        if not (repo / path).is_file():
            raise SetupError("A tracked skill manifest is missing. Restore it before onboarding.")
        parents = PurePosixPath(path).parent.parts
        group = None
        for i, part in enumerate(parents):
            if part == "skills" or part.endswith("-skills"):
                group = PurePosixPath(*parents[: i + 1]).as_posix()
                break
        if group is None:
            group = PurePosixPath(*parents[:-1]).as_posix() if len(parents) > 1 else "."
        skill_dir = PurePosixPath(path).parent.as_posix()
        if group == ".":
            group = skill_dir
        manifests.append((skill_dir, group, not bool(set(parents) & EXCLUDED)))
    if not manifests:
        raise SetupError("No tracked SKILL.md files were found. Commit your skills first.")
    if len(manifests) > 1 and any(skill == "." for skill, _, _ in manifests):
        raise SetupError(
            "A root-level SKILL.md together with nested skills cannot be safely isolated by CI onboarding."
        )
    # If fixtures are nested inside a broad skill root, use individual skill
    # directories so recursive CI discovery cannot silently include them.
    mixed_roots = {
        group for _, group, default in manifests if not default and not (set(PurePosixPath(group).parts) & EXCLUDED)
    }
    counts: Counter[str] = Counter()
    defaults = {}
    for skill_dir, group, default in manifests:
        root = skill_dir if group in mixed_roots else group
        counts[root] += 1
        defaults[root] = default
    if any(other != root and (root == "." or other.startswith(root + "/")) for root in counts for other in counts):
        raise SetupError(
            "Overlapping discovered skill roots cannot be isolated safely by CI onboarding. Choose a non-overlapping skill layout."
        )
    return [{"root": root, "count": count, "default": defaults[root]} for root, count in sorted(counts.items())]


def infer_branch(repo: Path, explicit: str | None) -> str:
    branch = explicit
    if branch is None:
        remote_head = git(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD", optional=True).strip()
        branch = remote_head.removeprefix("origin/") if remote_head else None
    if branch is None:
        branch = git(repo, "config", "--local", "--get", "init.defaultBranch", optional=True).strip() or None
    if branch is None:
        branch = git(repo, "symbolic-ref", "--short", "HEAD", optional=True).strip() or None
    if not branch or branch.startswith("-") or any(ord(c) < 32 for c in branch):
        raise SetupError("Cannot infer a default branch. Provide --default-branch.")
    # check-ref-format has no stdout on success, so check it without optional mode.
    git(repo, "check-ref-format", f"refs/heads/{branch}")
    return branch


class WorkflowDumper(yaml.SafeDumper):
    pass


def represent_string(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


def require_clean(repo: Path) -> None:
    if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise SetupError("Save or commit existing changes first. Onboarding requires a clean checkout.")


def require_identity(repo: Path) -> None:
    """Fail before changing GitHub settings if setup cannot create commits."""
    try:
        for key in ("user.name", "user.email"):
            if not git(repo, "config", "--get", key).strip():
                raise SetupError("Missing Git identity")
        for variable in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
            git(repo, "var", variable)
    except SetupError as exc:
        raise SetupError(
            "Set your Git commit identity in this repository first: "
            'git config user.name "Your Name" and git config user.email "you@example.com".'
        ) from exc


WorkflowDumper.add_representer(str, represent_string)
